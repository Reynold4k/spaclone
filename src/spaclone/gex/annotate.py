"""Cell typing of spots, bins, beads or cells from marker modules, and import of deconvolution results.

A 10-µm bead or bin can hold parts of more than one cell, so a "B unit" here is a unit dominated by B-lineage
markers, not a segmented B cell. :func:`score_cell_types` reproduces the archive's bead typing
(``04_external_validation_2026-10/workflow/X6_airspace_bcr_bead_celltype.py``): each module's share of the unit's counts,
divided by that module's mean share over tissue units (enrichment); the unit is the module with the highest
enrichment, or ``"unassigned"`` with fewer than two marker counts (``min_module_counts=1``; the default 2 also
asks for two counts of the winning module's own genes).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone._utils import SpacloneWarning
from spaclone.gex.qc import _bool_obs, _record, gene_index, get_counts, get_xy

__all__ = ["LINEAGES", "MARKERS", "celltype_composition", "import_deconvolution", "score_cell_types"]

logger = logging.getLogger("spaclone")


class MarkerWarning(SpacloneWarning):
    """A marker module has no gene in the data, or its genes are never detected."""


class JoinWarning(SpacloneWarning):
    """Few units could be matched when joining an external table."""


#: Built-in marker modules (mouse and human symbols). B, plasma (ASC), T, myeloid and stromal are the archive's X6
#: modules; GC B is the archive's X3 germinal-centre B set.
MARKERS: dict[str, dict[str, list[str]]] = {
    "immune_mouse": {
        "B": ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Pax5", "Cr2", "Fcer2a"],
        "GC_B": ["Aicda", "S1pr2", "Gcsam", "Rgs13", "Mef2b"],
        "ASC": ["Jchain", "Xbp1", "Prdm1", "Sdc1", "Mzb1"],
        "T": ["Cd3e", "Cd3d", "Cd3g", "Cd4", "Cd8a", "Cd8b1"],
        "Tfh": ["Il21", "Pdcd1", "Icos", "Tox2", "Ascl2", "Sh2d1a"],
        "myeloid": ["Lyz2", "Cd68", "C1qa", "C1qb", "Csf1r"],
        "stromal": ["Col1a1", "Col3a1", "Pdpn", "Ccl21a", "Pecam1", "Cdh5"],
        "erythroid": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-b1", "Hbb-b2", "Alas2", "Gypa"],
    },
    "immune_human": {
        "B": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "CR2", "FCER2"],
        "GC_B": ["AICDA", "S1PR2", "GCSAM", "RGS13", "MEF2B", "SERPINA9"],
        "ASC": ["JCHAIN", "XBP1", "PRDM1", "SDC1", "MZB1"],
        "T": ["CD3E", "CD3D", "CD3G", "CD4", "CD8A", "CD8B"],
        "Tfh": ["IL21", "PDCD1", "ICOS", "CXCL13", "TOX2", "SH2D1A"],
        "myeloid": ["LYZ", "CD68", "C1QA", "C1QB", "CSF1R"],
        "stromal": ["COL1A1", "COL3A1", "PDPN", "CCL21", "PECAM1", "CDH5"],
        "erythroid": ["HBA1", "HBA2", "HBB", "HBD", "ALAS2", "GYPA"],
    },
}

#: Classes that are commonly merged when reporting (labels of the built-in modules).
LINEAGES: dict[str, list[str]] = {"B lineage": ["B", "GC_B", "ASC"], "T lineage": ["T", "Tfh"]}


def _resolve_markers(markers: str | Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    if isinstance(markers, str):
        if markers not in MARKERS:
            raise ValueError(f"unknown marker set {markers!r}; built-in sets: {sorted(MARKERS)}")
        return {k: list(v) for k, v in MARKERS[markers].items()}
    out = {str(k): [str(g) for g in v] for k, v in markers.items()}
    if not out:
        raise ValueError("markers is empty")
    return out


def score_cell_types(
    adata: ad.AnnData,
    *,
    markers: str | Mapping[str, Sequence[str]] = "immune_mouse",
    method: Literal["enrichment", "scanpy_score"] = "enrichment",
    min_marker_counts: float = 2,
    min_module_counts: float = 2,
    tissue_key: str | None = "in_tissue",
    layer: str | None = None,
    key_added: str = "cell_type",
    unassigned: str = "unassigned",
    rng: int = 0,
) -> pd.DataFrame:
    """Type every unit by marker modules.

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts (``layer``, else ``layers["counts"]``, else ``X``).
    markers
        ``"immune_mouse"``, ``"immune_human"`` (see :data:`MARKERS`) or a ``{class: [genes]}`` mapping. Missing genes
        are skipped; a module with no gene present is dropped with a :class:`MarkerWarning`.
    method
        ``"enrichment"``: share of the unit's counts from the module's genes, divided by the module's mean share over
        tissue units; the class is the module with the highest enrichment (ties: first module). ``"scanpy_score"``:
        ``scanpy.tl.score_genes`` on the log-normalised ``X`` (run :func:`~spaclone.gex.preprocess` first); the class
        is the module with the highest score, ``unassigned`` if no score is positive.
    min_marker_counts
        Units with fewer counts of all marker genes together are ``unassigned``.
    min_module_counts
        A unit can only take a class whose own genes have at least this many counts in it (the best module that
        passes wins; none passing = ``unassigned``). The archive's X6 rule is ``min_module_counts=1``. The default 2
        removes a bias of the enrichment ratio towards modules of rare genes (GC B, Tfh): one count of a rare gene
        gives a huge share / mean-share ratio, so with a floor of 1 such modules win units on a single molecule.
    tissue_key
        Boolean ``obs`` column of tissue units, the reference of the mean shares (all units if absent). Off-tissue
        units are typed too.
    layer
        Layer holding raw counts.
    key_added
        ``obs`` column of the class (categorical, modules then ``unassigned``). Enrichments (or scores) go to
        ``obsm[f"{key_added}_enrichment"]`` (or ``_score``) and module shares to ``obsm[f"{key_added}_share"]``.
    unassigned
        Label of units below the marker floor.
    rng
        Seed of ``scanpy.tl.score_genes`` (control genes).

    Returns
    -------
    pandas.DataFrame
        One row per module: ``genes_used``, ``genes_missing``, ``mean_share`` (tissue), ``n_units`` and ``pct_units``
        (share of tissue units with that class), plus the ``unassigned`` row.

    References
    ----------
    Tirosh et al. 2016, doi:10.1126/science.aad0501 (module scores, ``scanpy_score``); Jiang et al. 2026,
    doi:10.1073/pnas.2504742123 (AIR-SPACE beads typed by markers).
    """
    mods = _resolve_markers(markers)
    counts = get_counts(adata, layer)
    var = adata.var_names
    used: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for k, genes in mods.items():
        pos, miss = gene_index(var, genes)
        used[k] = [str(var[p]) for p in pos]
        missing[k] = miss
    empty = [k for k, g in used.items() if not g]
    if empty:
        warnings.warn(f"marker module(s) {empty} have no gene in the data and are dropped", MarkerWarning, stacklevel=2)
    names = [k for k in mods if used[k]]
    if not names:
        raise ValueError("no marker module has a gene in the data")
    tissue = _bool_obs(adata, tissue_key)
    if not tissue.any():
        raise ValueError(f"no unit is in tissue ({tissue_key!r})")
    total = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    mod_counts = np.column_stack(
        [np.asarray(counts[:, gene_index(var, used[k])[0]].sum(axis=1)).ravel() for k in names]
    ).astype(float)
    union, _ = gene_index(var, sorted({g for k in names for g in used[k]}))
    marker_total = np.asarray(counts[:, union].sum(axis=1), dtype=float).ravel()
    share = mod_counts / np.maximum(total, 1.0)[:, None]
    mean_share = share[tissue].mean(axis=0)
    if method == "enrichment":
        dead = mean_share <= 0
        if dead.any():
            warnings.warn(
                f"marker module(s) {[names[i] for i in np.flatnonzero(dead)]} are never detected in tissue",
                MarkerWarning,
                stacklevel=2,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(dead[None, :], -np.inf, share / np.where(dead, 1.0, mean_share)[None, :])
        own = mod_counts >= min_module_counts
        best = np.argmax(np.where(own, value, -np.inf), axis=1)
        pick = np.arange(len(best)), best
        ok = (marker_total >= min_marker_counts) & own[pick] & np.isfinite(value[pick])
        store = "enrichment"
    elif method == "scanpy_score":
        import scanpy as sc

        if "log1p" not in adata.uns:
            raise ValueError("method='scanpy_score' needs log-normalised X; run spaclone.gex.preprocess first")
        value = np.empty((adata.n_obs, len(names)))
        for i, k in enumerate(names):
            res = sc.tl.score_genes(adata, used[k], score_name="_spaclone_score", random_state=rng, copy=True)
            value[:, i] = res.obs["_spaclone_score"].to_numpy(dtype=float)
        own = mod_counts >= min_module_counts
        best = np.argmax(np.where(own, value, -np.inf), axis=1)
        pick = np.arange(len(best)), best
        ok = (marker_total >= min_marker_counts) & own[pick] & (value[pick] > 0)
        store = "score"
    else:
        raise ValueError(f"method must be 'enrichment' or 'scanpy_score', got {method!r}")
    cats = [*names, unassigned]
    codes = np.where(ok, best, len(names))
    adata.obs[key_added] = pd.Categorical.from_codes(codes, categories=cats)
    adata.obsm[f"{key_added}_{store}"] = pd.DataFrame(value, index=adata.obs_names, columns=names)
    adata.obsm[f"{key_added}_share"] = pd.DataFrame(share, index=adata.obs_names, columns=names)
    _record(
        adata,
        f"score_cell_types_{key_added}",
        {
            "markers": used,
            "method": method,
            "min_marker_counts": min_marker_counts,
            "min_module_counts": min_module_counts,
            "tissue_key": tissue_key,
        },
    )
    lab = codes[tissue]
    rows = []
    for i, k in enumerate(cats):
        rows.append(
            {
                "cell_type": k,
                "genes_used": ",".join(used.get(k, [])),
                "genes_missing": ",".join(missing.get(k, [])),
                "mean_share": float(mean_share[i]) if i < len(names) else np.nan,
                "n_units": int((lab == i).sum()),
                "pct_units": 100.0 * float((lab == i).mean()),
            }
        )
    return pd.DataFrame(rows)


def celltype_composition(
    adata: ad.AnnData,
    select: Any,
    *,
    type_key: str = "cell_type",
    depth_key: str | None = None,
    n_depth_bins: int = 20,
    tissue_key: str | None = "in_tissue",
    groups: Mapping[str, Sequence[str]] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Class composition of selected units against the composition expected from their depth alone.

    For example, are the beads that carry BCR molecules B cells more often than beads of the same GEX depth? Tissue
    units are cut into ``n_depth_bins`` quantile bins of depth; the depth-matched expectation of a class is the mean,
    over the selected units, of that class's share among all tissue units of the same depth bin (archive X6).

    Parameters
    ----------
    adata
        AnnData typed by :func:`score_cell_types` (or any categorical ``obs[type_key]``).
    select
        Units to describe: a boolean array over ``obs``, the name of a boolean ``obs`` column, or obs names. Only
        tissue units are used.
    type_key
        ``obs`` column of the class.
    depth_key
        ``obs`` column of the depth; default the unit's total counts.
    n_depth_bins
        Number of depth quantile bins (tied quantiles are merged).
    tissue_key
        Boolean ``obs`` column of tissue units (all units if absent).
    groups
        Extra rows that sum classes, e.g. ``{"B lineage": ["B", "ASC"]}``.
    layer
        Layer holding raw counts (for the default depth).

    Returns
    -------
    pandas.DataFrame
        ``cell_type``, ``n_selected``, ``observed_pct``, ``depth_matched_pct``, ``ratio`` (observed / expected) and
        ``baseline_pct`` (share among all tissue units).
    """
    tissue = _bool_obs(adata, tissue_key)
    sel = _as_mask(adata, select) & tissue
    lab = adata.obs[type_key].astype(str).to_numpy()
    if depth_key is None:
        depth = np.asarray(get_counts(adata, layer).sum(axis=1), dtype=float).ravel()
    else:
        depth = adata.obs[depth_key].to_numpy(dtype=float)
    dt = depth[tissue]
    edges = np.unique(np.quantile(dt, np.linspace(0, 1, n_depth_bins + 1)))
    dbin = np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, max(len(edges) - 2, 0))
    cats = (
        list(adata.obs[type_key].cat.categories)
        if isinstance(adata.obs[type_key].dtype, pd.CategoricalDtype)
        else sorted(set(lab[tissue]))
    )
    tb = dbin[tissue]
    n_bins = int(dbin.max()) + 1 if len(dbin) else 0
    cnt_bin = np.bincount(tb, minlength=n_bins).astype(float)
    rows = []
    for c in cats:
        hit = lab == c
        frac = np.bincount(tb, weights=hit[tissue].astype(float), minlength=n_bins) / np.maximum(cnt_bin, 1)
        rows.append(
            {
                "cell_type": c,
                "n_selected": int(sel.sum()),
                "observed_pct": 100.0 * float(hit[sel].mean()) if sel.any() else np.nan,
                "depth_matched_pct": 100.0 * float(frac[dbin[sel]].mean()) if sel.any() else np.nan,
                "baseline_pct": 100.0 * float(hit[tissue].mean()),
            }
        )
    out = pd.DataFrame(rows)
    for name, members in (groups or {}).items():
        part = out[out.cell_type.isin(list(members))]
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "cell_type": name,
                            "n_selected": int(sel.sum()),
                            "observed_pct": part.observed_pct.sum(),
                            "depth_matched_pct": part.depth_matched_pct.sum(),
                            "baseline_pct": part.baseline_pct.sum(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    out["ratio"] = out.observed_pct / out.depth_matched_pct.where(out.depth_matched_pct > 0)
    return out


def _as_mask(adata: ad.AnnData, select: Any) -> np.ndarray:
    if isinstance(select, str):
        return _bool_obs(adata, select) if select in adata.obs else np.zeros(adata.n_obs, dtype=bool)
    arr = np.asarray(select)
    if arr.dtype == bool:
        if arr.shape != (adata.n_obs,):
            raise ValueError("a boolean selection must have one value per unit")
        return arr
    return adata.obs_names.isin(pd.Index(arr).astype(str))


_C2L_PREFIXES = (
    "q05cell_abundance_w_sf_",
    "q95cell_abundance_w_sf_",
    "meanscell_abundance_w_sf_",
    "stdscell_abundance_w_sf_",
    "q05_cell_abundance_w_sf_",
    "means_cell_abundance_w_sf_",
)


def _read_table(table: Any, kind: str) -> pd.DataFrame:
    if isinstance(table, str | Path):
        p = Path(table)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffix == ".h5ad":
            return _read_table(ad.read_h5ad(p), kind)
        sep = "\t" if p.suffix in (".tsv", ".txt") or p.name.endswith((".tsv.gz", ".txt.gz")) else ","
        return pd.read_csv(p, sep=sep, index_col=0)
    if isinstance(table, ad.AnnData):
        keys = [k for k in ("q05_cell_abundance_w_sf", "means_cell_abundance_w_sf") if k in table.obsm]
        if not keys:
            raise ValueError("the AnnData has no cell2location abundance in obsm (q05_cell_abundance_w_sf)")
        m = table.obsm[keys[0]]
        df = m.copy() if isinstance(m, pd.DataFrame) else pd.DataFrame(np.asarray(m), index=table.obs_names)
        df.index = table.obs_names
        if "spatial" in table.obsm:
            xy = np.asarray(table.obsm["spatial"], dtype=float)
            df["_x"], df["_y"] = xy[:, 0], xy[:, 1]
        return df
    if isinstance(table, pd.DataFrame):
        return table.copy()
    raise TypeError(f"table must be a DataFrame, an AnnData or a path, got {type(table).__name__}")


def import_deconvolution(
    adata: ad.AnnData,
    table: Any,
    *,
    kind: Literal["rctd", "cell2location", "generic"] = "rctd",
    on: Literal["index", "coordinates"] = "index",
    spot_key: str | None = None,
    x_key: str = "x",
    y_key: str = "y",
    coord_scale: float = 1.0,
    max_dist_um: float | None = None,
    normalize: bool | None = None,
    spatial_key: str = "spatial",
    key_added: str | None = None,
) -> pd.DataFrame:
    """Attach RCTD / cell2location cell-type weights to units, joined by unit id or by position, never by row order.

    Parameters
    ----------
    adata
        Spot-level AnnData.
    table
        Weights: a DataFrame (rows = units, numeric columns = cell types), a csv / tsv / parquet path, or a
        cell2location output AnnData (``obsm["q05_cell_abundance_w_sf"]``). RCTD columns ``first_type``,
        ``second_type`` and ``spot_class`` are carried over; cell2location column prefixes are stripped.
    kind
        ``"rctd"`` (weights normalised to proportions by default), ``"cell2location"`` (abundances, not
        normalised) or ``"generic"``.
    on
        ``"index"``: the table index (or ``spot_key`` column) must hold the unit ids of ``adata.obs_names``.
        ``"coordinates"``: each table row (``x_key``, ``y_key`` × ``coord_scale`` = µm) is matched to the nearest
        unit within ``max_dist_um``; a unit claimed by two rows raises.
    spot_key
        Column of the table holding unit ids (default: its index).
    x_key, y_key, coord_scale
        Table coordinates and their factor to µm (``on="coordinates"``).
    max_dist_um
        Largest allowed distance for a coordinate match; default half the median nearest-neighbour distance between
        units.
    normalize
        Divide each row by its sum (default True for RCTD, False otherwise).
    spatial_key
        ``obsm`` key of the unit coordinates.
    key_added
        Prefix of the outputs (default ``kind``): ``obsm[key]`` (DataFrame aligned to ``obs``, NaN for unmatched
        units), ``obs[f"{key}_top"]`` (largest weight) and, for RCTD, ``obs[f"{key}_first_type"]``.

    Returns
    -------
    pandas.DataFrame
        One-row report: ``n_table``, ``n_units``, ``n_matched``, ``n_unmatched_rows``, ``max_dist_um`` (coordinates).

    Raises
    ------
    ValueError
        If no table row matches a unit id (e.g. a table indexed 0..n-1, which would otherwise be joined by row order
        — the archive's ``RCTD_bestcell.h5ad`` was misaligned exactly so), or if coordinate matches are not one to one.

    References
    ----------
    Cable et al. 2022, doi:10.1038/s41587-021-00830-w (RCTD); Kleshchevnikov et al. 2022,
    doi:10.1038/s41587-021-01139-4 (cell2location).
    """
    if kind not in ("rctd", "cell2location", "generic"):
        raise ValueError(f"kind must be 'rctd', 'cell2location' or 'generic', got {kind!r}")
    key = key_added or kind
    df = _read_table(table, kind)
    if spot_key is not None:
        df = df.set_index(spot_key)
    df.index = df.index.astype(str)
    labels = {c: df.pop(c) for c in ("first_type", "second_type", "spot_class") if c in df.columns}
    coords = None
    if on == "coordinates":
        cx, cy = ("_x", "_y") if "_x" in df.columns and x_key not in df.columns else (x_key, y_key)
        if cx not in df.columns or cy not in df.columns:
            raise KeyError(f"on='coordinates' needs columns {x_key!r} and {y_key!r} in the table")
        coords = np.column_stack([df[cx].to_numpy(float), df[cy].to_numpy(float)]) * coord_scale
    df = df.drop(columns=[c for c in ("_x", "_y", x_key, y_key) if c in df.columns])
    weights = df.select_dtypes(include=[np.number])
    if weights.shape[1] == 0:
        raise ValueError("the table has no numeric cell-type column")
    if kind == "cell2location":
        cols = []
        for c in weights.columns.astype(str):
            for p in _C2L_PREFIXES:
                if c.startswith(p):
                    c = c[len(p) :]
                    break
            cols.append(c)
        weights.columns = cols
    report: dict[str, Any] = {"n_table": len(weights), "n_units": int(adata.n_obs)}
    if on == "index":
        pos = adata.obs_names.get_indexer(weights.index)
        n_hit = int((pos >= 0).sum())
        if n_hit == 0:
            raise ValueError(
                "no table row matches a unit id (adata.obs_names); refusing to join by row order. Index the table "
                "by unit id (spot_key=) or join by position (on='coordinates')."
            )
        if weights.index.duplicated().any():
            raise ValueError("the table index holds duplicated unit ids")
        row_of_unit = np.full(adata.n_obs, -1, dtype=np.int64)
        row_of_unit[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
    elif on == "coordinates":
        xy = get_xy(adata, spatial_key)
        tree = cKDTree(xy)
        if max_dist_um is None:
            d_nn, _ = tree.query(xy, k=2)
            max_dist_um = 0.5 * float(np.median(d_nn[:, 1]))
        d, j = tree.query(coords, k=1)
        ok = d <= max_dist_um
        if len(np.unique(j[ok])) < ok.sum():
            raise ValueError(
                "coordinate matches are not one to one (two table rows map to one unit); lower "
                "max_dist_um or check coord_scale"
            )
        row_of_unit = np.full(adata.n_obs, -1, dtype=np.int64)
        row_of_unit[j[ok]] = np.flatnonzero(ok)
        report["max_dist_um"] = max_dist_um
    else:
        raise ValueError(f"on must be 'index' or 'coordinates', got {on!r}")
    matched = row_of_unit >= 0
    report["n_matched"] = int(matched.sum())
    report["n_unmatched_rows"] = int(len(weights) - matched.sum())
    if matched.sum() < 0.5 * min(len(weights), adata.n_obs):
        warnings.warn(
            f"only {int(matched.sum())} of {len(weights)} table rows matched a unit", JoinWarning, stacklevel=2
        )
    w = weights.to_numpy(dtype=float)
    if normalize if normalize is not None else kind == "rctd":
        s = w.sum(axis=1, keepdims=True)
        w = np.divide(w, s, out=np.full_like(w, np.nan), where=s > 0)
    full = np.full((adata.n_obs, w.shape[1]), np.nan)
    full[matched] = w[row_of_unit[matched]]
    out = pd.DataFrame(full, index=adata.obs_names, columns=[str(c) for c in weights.columns])
    adata.obsm[key] = out
    top = np.full(adata.n_obs, np.nan, dtype=object)
    good = matched & np.isfinite(full).any(axis=1)
    top[good] = out.columns.to_numpy()[np.nanargmax(full[good], axis=1)]
    adata.obs[f"{key}_top"] = pd.Categorical(top, categories=list(out.columns))
    for name, s in labels.items():
        vals = np.full(adata.n_obs, np.nan, dtype=object)
        vals[matched] = s.astype(str).to_numpy()[row_of_unit[matched]]
        adata.obs[f"{key}_{name}"] = pd.Categorical(vals)
    _record(adata, f"import_deconvolution_{key}", {"kind": kind, "on": on, **report})
    return pd.DataFrame([report])
