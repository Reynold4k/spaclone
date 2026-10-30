"""Tissue regions from the transcriptome: germinal centres, plasma-cell (ASC) areas, T zones, follicles, red pulp.

Two families of rules:

* **marker-share regions** (``"germinal_centre"``, ``"plasma_cell"``): the share of a marker set among all counts
  within ``smooth_um`` of each tissue unit; units in the top ``top_frac`` are "hot"; hot units are grouped into
  region instances by DBSCAN (``dbscan_eps_um``, ``min_samples``), clusters with fewer than ``min_units`` hot units
  are dropped, and the kept clusters are widened by ``widen_um`` (every tissue unit within that distance of a
  kept hot unit joins the nearest one's instance). The defaults are the archive's AIR-SPACE rules
  (``04_external_validation_2026-10/workflow/X3_airspace_claims.py`` and ``X3b_airspace_gc_sharing.py``):
  germinal centres = GC-B share (Aicda, S1pr2, Gcsam, Rgs13, Mef2b) within 30 µm in the top 8%, DBSCAN 25 µm /
  8 units, ≥ 30 units, widened by 25 µm; ASC area = plasma-cell share (Jchain, Xbp1, Prdm1, Sdc1) within 30 µm in the
  top 8% and not already a germinal centre (no instances).
* **cell-type-mask regions** (``"t_zone"``, ``"follicle"``, ``"red_pulp"``): masks of the pixels where T,
  B-lineage or erythroid units dominate (``spaclone.tl.mask_from_types``: 10-µm grid, 30-µm Gaussian, Otsu), T zones
  = connected components of the T-rich mask, compartments T zone > red pulp (erythroid-rich or outside the B-rich
  area) > follicle (archive ``T_common.py``). These need the mask utilities of ``spaclone.tl.structures``.

Labels are written to one region column, so the order of calls sets the priority: a unit keeps an earlier label
unless ``overwrite=True`` (call germinal centres before the ASC area, as the archive does).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone.gex.qc import _bool_obs, _groups, _record, gene_index, get_counts, get_xy, radius_sum

__all__ = ["REGION_LABELS", "REGION_MARKERS", "call_regions", "marker_share"]

logger = logging.getLogger("spaclone")

#: Marker sets of the share-based regions (archive X3).
REGION_MARKERS: dict[str, dict[str, list[str]]] = {
    "germinal_centre": {
        "mouse": ["Aicda", "S1pr2", "Gcsam", "Rgs13", "Mef2b"],
        "human": ["AICDA", "S1PR2", "GCSAM", "RGS13", "MEF2B", "SERPINA9"],
    },
    "plasma_cell": {"mouse": ["Jchain", "Xbp1", "Prdm1", "Sdc1"], "human": ["JCHAIN", "XBP1", "PRDM1", "SDC1"]},
}

#: Default label of each region kind in the region column.
REGION_LABELS: dict[str, str] = {
    "germinal_centre": "GC",
    "plasma_cell": "ASC",
    "t_zone": "T zone",
    "follicle": "follicle",
    "red_pulp": "red pulp",
}
_ALIASES = {
    "gc": "germinal_centre",
    "germinal_center": "germinal_centre",
    "asc": "plasma_cell",
    "pc": "plasma_cell",
    "plasma": "plasma_cell",
    "tzone": "t_zone",
    "t zone": "t_zone",
    "red pulp": "red_pulp",
}
_SHARE_DEFAULTS: dict[str, dict[str, float]] = {
    "germinal_centre": {
        "top_frac": 0.08,
        "smooth_um": 30.0,
        "dbscan_eps_um": 25.0,
        "min_samples": 8,
        "min_units": 30,
        "widen_um": 25.0,
    },
    "plasma_cell": {
        "top_frac": 0.08,
        "smooth_um": 30.0,
        "dbscan_eps_um": 0.0,
        "min_samples": 8,
        "min_units": 0,
        "widen_um": 0.0,
    },
}
_SHARE_KEY = {"germinal_centre": "gc_share", "plasma_cell": "asc_share"}
#: Cell classes behind the masks (labels of :data:`spaclone.gex.MARKERS` and common deconvolution names).
_MASK_TYPES = {
    "T": ("T", "Tfh", "T lineage", "T_cells", "T cells"),
    "B": ("B", "GC_B", "ASC", "B lineage", "B_cells", "B cells", "plasma"),
    "erythroid": ("erythroid", "erythrocyte", "Erythrocyte", "Erythrocytes"),
}


def marker_share(
    adata: ad.AnnData,
    genes: Sequence[str],
    *,
    radius_um: float = 30.0,
    tissue_key: str | None = "in_tissue",
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """Share of marker-gene counts among all counts within a radius of each tissue unit.

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts and ``obsm[spatial_key]`` in µm.
    genes
        Marker genes (missing ones are skipped).
    radius_um
        Disc radius (inclusive; the unit itself counts). 0 = the unit alone.
    tissue_key
        Boolean ``obs`` column; only tissue units enter the sums and only they get a share.
    library_key
        ``obs`` column naming the section; discs stay within a section.
    layer
        Layer holding raw counts.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    numpy.ndarray
        ``sum(marker counts) / max(sum(all counts), 1)`` over the disc; NaN for off-tissue units.
    """
    counts = get_counts(adata, layer)
    pos, missing = gene_index(adata.var_names, list(genes))
    if not len(pos):
        raise ValueError(f"none of the marker genes {list(genes)} is in the data")
    if missing:
        logger.info("marker_share: genes not in the data: %s", missing)
    mk = np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel()
    tot = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    tissue = _bool_obs(adata, tissue_key)
    xy = get_xy(adata, spatial_key)
    lib = _lib_codes(adata, library_key)
    t = np.flatnonzero(tissue)
    sums = radius_sum(xy[t], np.column_stack([mk[t], tot[t]]), radius_um, groups=lib[t])
    out = np.full(adata.n_obs, np.nan)
    out[t] = sums[:, 0] / np.maximum(sums[:, 1], 1.0)
    return out


def _lib_codes(adata: ad.AnnData, library_key: str | None) -> np.ndarray:
    lib = np.zeros(adata.n_obs, dtype=np.int64)
    for k, (_, idx) in enumerate(_groups(adata, library_key).items()):
        lib[idx] = k
    return lib


def _kind(kind: str) -> str:
    k = _ALIASES.get(kind.lower(), kind.lower())
    if k not in REGION_LABELS:
        raise ValueError(f"unknown region kind {kind!r}; choose from {sorted(REGION_LABELS)}")
    return k


def call_regions(
    adata: ad.AnnData,
    *,
    kind: str,
    markers: Sequence[str] | None = None,
    species: str = "mouse",
    top_frac: float | None = None,
    smooth_um: float | None = None,
    dbscan_eps_um: float | None = None,
    min_samples: int | None = None,
    min_units: int | None = None,
    widen_um: float | None = None,
    tissue_key: str | None = "in_tissue",
    library_key: str | None = None,
    layer: str | None = None,
    cell_type_key: str = "cell_type",
    types: Mapping[str, Sequence[str]] | None = None,
    pixel_um: float = 10.0,
    min_area_um2: float = 8000.0,
    region_key: str = "region",
    instance_key: str = "region_id",
    share_key: str | None = None,
    label: str | None = None,
    background: str = "other",
    overwrite: bool = False,
    relabel: bool = True,
    spatial_key: str = "spatial",
) -> pd.DataFrame:
    """Call tissue regions and their instances (individual germinal centres, T zones, ...).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts, ``obsm[spatial_key]`` in µm and (for share kinds) a tissue column.
    kind
        ``"germinal_centre"``, ``"plasma_cell"`` (alias ``"asc"``), ``"t_zone"``, ``"follicle"`` or ``"red_pulp"``.
    markers
        Marker genes of a share kind (default :data:`REGION_MARKERS` for ``species``).
    species
        ``"mouse"`` or ``"human"`` (default markers).
    top_frac
        Share kinds: units whose share is at least the ``1 - top_frac`` quantile over tissue units (and > 0) are hot.
    smooth_um
        Share kinds: radius (µm) of the disc over which counts are summed. Mask kinds: standard deviation (µm) of the
        Gaussian smoothing of the type counts.
    dbscan_eps_um, min_samples
        DBSCAN neighbourhood radius (µm, inclusive) and minimum neighbourhood size (the unit included); 0 disables
        clustering (every hot unit is in the region; no instances).
    min_units
        Clusters with fewer hot units are dropped.
    widen_um
        Kept clusters grow by this distance (0 = no widening).
    tissue_key
        Boolean ``obs`` column of tissue units; only they enter shares, quantiles and clusters.
    library_key
        ``obs`` column naming the section; every step runs per section.
    layer
        Layer holding raw counts.
    cell_type_key
        Mask kinds: ``obs`` column of unit classes (from :func:`~spaclone.gex.score_cell_types` or a deconvolution
        top class); units whose class is in none of the mask groups are left out of the masks.
    types
        Mask kinds: ``{"T": [...], "B": [...], "erythroid": [...]}`` classes behind each mask (defaults cover the
        built-in marker labels and common deconvolution names).
    pixel_um, min_area_um2
        Mask kinds: grid pixel and smallest island / hole kept.
    region_key, instance_key
        ``obs`` columns written: the region label (categorical; ``background`` elsewhere on tissue, NaN off tissue)
        and the instance id (e.g. ``"GC_0"``; prefixed ``"{library}|"`` with ``library_key``; NaN outside instances).
    share_key
        Share kinds: ``obs`` column of the share (default ``"gc_share"`` / ``"asc_share"``).
    label
        Region label (default :data:`REGION_LABELS`).
    background
        Label of tissue units in no region.
    overwrite
        If False, units that already carry another region label keep it.
    relabel
        Number instances 0..n-1 in DBSCAN order; False keeps DBSCAN's labels (gaps where clusters were dropped).
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    pandas.DataFrame
        One row per region instance: ``region_id``, ``region``, ``kind``, ``library``, ``n_units`` (after widening),
        ``n_core`` (hot units), ``x_um``, ``y_um`` (centroid of the units), ``mean_share``.

    Notes
    -----
    Parameter defaults per kind (``None`` = default):

    ====================  ========  =========  =============  ===========  =========  ========
    kind                  top_frac  smooth_um  dbscan_eps_um  min_samples  min_units  widen_um
    ====================  ========  =========  =============  ===========  =========  ========
    germinal_centre       0.08      30         25             8            30         25
    plasma_cell           0.08      30         0 (off)        –            –          0
    t_zone etc. (masks)   –         30 (σ)     –              –            –          –
    ====================  ========  =========  =============  ===========  =========  ========

    The archive's first germinal-centre rule (X3: ``region_maps_*.parquet``) is ``top_frac=0.04,
    dbscan_eps_um=30, min_samples=10, min_units=40, widen_um=0``.

    References
    ----------
    Ester et al. 1996, KDD-96 pp. 226–231, and Schubert et al. 2017, doi:10.1145/3068335 (DBSCAN); Jiang et al. 2026,
    doi:10.1073/pnas.2504742123 (AIR-SPACE germinal centres from GEX); Otsu 1979, doi:10.1109/TSMC.1979.4310076
    (mask threshold).
    """
    k = _kind(kind)
    lab_name = label or REGION_LABELS[k]
    if k in _SHARE_DEFAULTS:
        p = dict(_SHARE_DEFAULTS[k])
        for name, val in (
            ("top_frac", top_frac),
            ("smooth_um", smooth_um),
            ("dbscan_eps_um", dbscan_eps_um),
            ("min_samples", min_samples),
            ("min_units", min_units),
            ("widen_um", widen_um),
        ):
            if val is not None:
                p[name] = val
        if not 0 < p["top_frac"] < 1:
            raise ValueError("top_frac must be between 0 and 1")
        genes = list(markers) if markers is not None else REGION_MARKERS[k][species]
        new_lab, new_id, share, table = _share_regions(
            adata, k, genes, p, lab_name, tissue_key, library_key, layer, relabel, spatial_key
        )
        adata.obs[share_key or _SHARE_KEY[k]] = share
        params: dict[str, Any] = {"kind": k, "markers": genes, **p}
    else:
        p = {"smooth_um": 30.0 if smooth_um is None else smooth_um, "pixel_um": pixel_um, "min_area_um2": min_area_um2}
        new_lab, new_id, table = _mask_regions(
            adata, k, lab_name, cell_type_key, types, p, tissue_key, library_key, relabel, spatial_key
        )
        params = {"kind": k, "cell_type_key": cell_type_key, **p}
    _merge_labels(
        adata, new_lab, new_id, lab_name, region_key, instance_key, background, overwrite, _bool_obs(adata, tissue_key)
    )
    if len(table):
        table["n_units"] = table.region_id.map(adata.obs[instance_key].value_counts()).fillna(0).astype(int)
    _record(
        adata,
        f"call_regions_{lab_name}",
        {
            **params,
            "label": lab_name,
            "region_key": region_key,
            "instance_key": instance_key,
            "n_instances": len(table),
        },
    )
    return table


def _share_regions(adata, k, genes, p, lab_name, tissue_key, library_key, layer, relabel, spatial_key):
    from sklearn.cluster import DBSCAN

    share = marker_share(
        adata,
        genes,
        radius_um=p["smooth_um"],
        tissue_key=tissue_key,
        library_key=library_key,
        layer=layer,
        spatial_key=spatial_key,
    )
    tissue = _bool_obs(adata, tissue_key)
    xy = get_xy(adata, spatial_key)
    is_lab = np.zeros(adata.n_obs, dtype=bool)
    inst = np.full(adata.n_obs, None, dtype=object)
    rows = []
    for lib, idx in _groups(adata, library_key).items():
        t = idx[tissue[idx]]
        if not len(t):
            continue
        s = share[t]
        q = np.quantile(s, 1 - p["top_frac"])
        hot = (s >= q) & (s > 0)
        prefix = f"{lib}|" if library_key is not None else ""
        if p["dbscan_eps_um"] <= 0:
            is_lab[t[hot]] = True
            continue
        hpos = t[hot]
        lab = np.full(len(t), -1, dtype=np.int64)
        if len(hpos) >= p["min_samples"]:
            cl = DBSCAN(eps=p["dbscan_eps_um"], min_samples=int(p["min_samples"])).fit_predict(xy[hpos])
            sizes = pd.Series(cl[cl >= 0]).value_counts()
            keep = np.sort(sizes[sizes >= p["min_units"]].index.to_numpy())
            lab[np.flatnonzero(hot)] = np.where(np.isin(cl, keep), cl, -1)
        else:
            keep = np.empty(0, dtype=np.int64)
        core = lab >= 0
        if p["widen_um"] > 0 and core.any():
            d, j = cKDTree(xy[t[core]]).query(xy[t])
            lab = np.where(d <= p["widen_um"], lab[core][j], -1)
        number = {c: (i if relabel else int(c)) for i, c in enumerate(keep)}
        for c in keep:
            m = lab == c
            rid = f"{prefix}{lab_name}_{number[c]}"
            is_lab[t[m]] = True
            inst[t[m]] = rid
            rows.append(
                {
                    "region_id": rid,
                    "region": lab_name,
                    "kind": k,
                    "library": lib,
                    "n_core": int((core & m).sum()),
                    "x_um": float(xy[t[m], 0].mean()),
                    "y_um": float(xy[t[m], 1].mean()),
                    "mean_share": float(np.mean(s[m])),
                }
            )
    return (
        is_lab,
        inst,
        share,
        pd.DataFrame(rows, columns=["region_id", "region", "kind", "library", "n_core", "x_um", "y_um", "mean_share"]),
    )


def _mask_regions(adata, k, lab_name, cell_type_key, types, p, tissue_key, library_key, relabel, spatial_key):
    from spaclone import tl

    need = ("Grid", "mask_from_types", "zones", "compartments")
    if not all(hasattr(tl, n) for n in need):
        raise NotImplementedError(
            f"region kind {k!r} needs spaclone.tl.structures ({', '.join(need)}), which is not available in this "
            "installation"
        )
    if cell_type_key not in adata.obs:
        raise KeyError(f"obs[{cell_type_key!r}] (unit classes) is missing; run spaclone.gex.score_cell_types first")
    groups = {g: tuple(v) for g, v in _MASK_TYPES.items()}
    if types:
        groups.update({g: tuple(v) for g, v in types.items()})
    tissue = _bool_obs(adata, tissue_key)
    xy = get_xy(adata, spatial_key)
    cls = adata.obs[cell_type_key].astype(str).to_numpy()
    grp = np.full(adata.n_obs, "", dtype=object)
    for g, members in groups.items():
        grp[np.isin(cls, list(members))] = g
    is_lab = np.zeros(adata.n_obs, dtype=bool)
    inst = np.full(adata.n_obs, None, dtype=object)
    rows = []
    for lib, idx in _groups(adata, library_key).items():
        t = idx[tissue[idx]]
        typed = t[grp[t] != ""]
        if not len(typed):
            continue
        units = pd.DataFrame({"x_um": xy[typed, 0], "y_um": xy[typed, 1], "group": grp[typed]})
        grid = tl.Grid.from_points(units.x_um, units.y_um, pixel_um=p["pixel_um"])

        def mask(g, units=units, grid=grid):
            if not (units.group == g).any():
                return np.zeros(grid.shape, dtype=bool)
            return tl.mask_from_types(
                units, types=g, type_key="group", grid=grid, smooth_um=p["smooth_um"], min_area_um2=p["min_area_um2"]
            ).values.astype(bool)

        m_t = mask("T")
        if k == "t_zone":
            region = m_t
        else:
            m_b, m_e = mask("B"), mask("erythroid")
            comp = tl.compartments(
                [("T zone", m_t), ("red pulp", m_e), ("red pulp", ~m_b)], default="follicle", grid=grid
            )
            code = list(comp.categories).index("follicle" if k == "follicle" else "red pulp")
            region = np.asarray(comp.values) == code
        zl = tl.zones(tl.Raster(region, grid)).values
        i, j, inside = grid.index(xy[t, 0], xy[t, 1])
        z = np.where(inside, zl[i, j], 0)
        prefix = f"{lib}|" if library_key is not None else ""
        ids = np.unique(z[z > 0])
        for n, zid in enumerate(ids):
            m = z == zid
            rid = f"{prefix}{lab_name}_{n if relabel else int(zid)}"
            is_lab[t[m]] = True
            inst[t[m]] = rid
            rows.append(
                {
                    "region_id": rid,
                    "region": lab_name,
                    "kind": k,
                    "library": lib,
                    "n_core": int(m.sum()),
                    "x_um": float(xy[t[m], 0].mean()),
                    "y_um": float(xy[t[m], 1].mean()),
                    "mean_share": np.nan,
                }
            )
    return (
        is_lab,
        inst,
        pd.DataFrame(rows, columns=["region_id", "region", "kind", "library", "n_core", "x_um", "y_um", "mean_share"]),
    )


def _merge_labels(adata, new_lab, new_id, lab_name, region_key, instance_key, background, overwrite, tissue):
    if region_key in adata.obs:
        cur = adata.obs[region_key].astype(object).to_numpy().copy()
    else:
        cur = np.full(adata.n_obs, None, dtype=object)
    if instance_key in adata.obs:
        cid = adata.obs[instance_key].astype(object).to_numpy().copy()
        cid = np.where(pd.isna(cid), None, cid)
    else:
        cid = np.full(adata.n_obs, None, dtype=object)
    cur = np.where(pd.isna(cur), None, cur)
    free = np.array([(c is None) or (c == background) or (c == lab_name) for c in cur], dtype=bool)
    # previous calls of the same label are replaced
    prev_same = np.array([c == lab_name for c in cur], dtype=bool)
    cur[prev_same] = background
    cid[prev_same] = None
    take = new_lab & (free | overwrite)
    cur[take] = lab_name
    cid[take] = new_id[take]
    cur[tissue & np.array([c is None for c in cur], dtype=bool)] = background
    cur[~tissue] = None
    cid[~tissue] = None
    cats = list(dict.fromkeys([c for c in cur if c is not None and c != background] + [background]))
    adata.obs[region_key] = pd.Categorical(cur, categories=cats)
    ids = [c for c in dict.fromkeys(cid) if c is not None]
    adata.obs[instance_key] = pd.Categorical(cid, categories=ids)
