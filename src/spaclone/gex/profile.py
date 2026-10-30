"""What is a region made of? Marker pseudobulk inside a region, around it and over the tissue, and per-unit module maps.

:func:`profile_regions` characterises given regions (for example BCR class-switch / mutation hotspots) from the
transcriptome alone. For every marker set it pools the raw counts of the set's genes over the units inside a region,
over the rest of a window around it (units inside any region left out) and over the whole tissue, as UMIs per 10,000
UMIs of all genes (pseudobulk CP10k), and reports the ratios inside / window and inside / tissue. The region's label is
the cell-type set with the highest ratio to its window; it is ``"<first> / <second> (mixed)"`` when the second set's
ratio is within ``2 ** mixed_log2`` (1.23×) of the first. The readout sets (cycling, switched Ig) are reported but never
label a region. This is the rule of the 2026-10 archive (``04_external_validation_2026-10/workflow/qc/T8f_celltype_overlap.py``),
fixed before looking at the data. It is descriptive: no test, no p-value.

:func:`module_scores` gives one score per unit and marker set for drawing: the mean over the set's genes of the
z-scored ``log1p(CP10k)`` (z over all units, population standard deviation), then the mean over the unit and its
neighbours within ``smooth_um`` (22.5 µm: the unit and its 8 neighbours on a 10-µm grid, plus the 4 units two steps
away along the axes).

The built-in cell-type sets (:data:`PROFILE_MARKERS`) contain no immunoglobulin gene, so a region found from BCR
isotype calls is characterised independently of those calls; Ig genes given in a cell-type set are dropped with a
warning. Jchain is not an immunoglobulin locus gene and stays in the antibody-secreting-cell set.

References
----------
Squair et al. 2021, doi:10.1038/s41467-021-25960-2 (pseudobulk over units instead of per-unit tests); Tirosh et al.
2016, doi:10.1126/science.aad0501 (marker module scores); Chen et al. 2022, doi:10.1016/j.cell.2022.04.003
(Stereo-seq bins).
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spaclone._utils import SpacloneWarning
from spaclone.gex.qc import _bool_obs, get_counts, get_xy, radius_graph

__all__ = [
    "PROFILE_MARKERS",
    "PROFILE_READOUTS",
    "RegionProfiles",
    "is_ig_gene",
    "module_scores",
    "profile_markers",
    "profile_regions",
    "unit_mask",
]

logger = logging.getLogger("spaclone")

#: Cell-type marker sets of :func:`profile_regions` (archive T8f, fixed 2026-10-28 before looking). No Ig gene.
PROFILE_MARKERS: dict[str, dict[str, list[str]]] = {
    "immune_mouse_no_ig": {
        "GC B": ["Aicda", "S1pr2", "Rgs13", "Gcsam", "Mef2b", "Bcl6", "Mybl1", "Lmo2"],
        "plasmablast/ASC": ["Jchain", "Xbp1", "Prdm1", "Irf4", "Mzb1"],
        "plasma cell": ["Sdc1", "Tnfrsf17", "Slamf7"],
        "follicular B": ["Fcer2a", "Sell", "Klf2", "Ms4a1"],
        "MZ B": ["Cr2", "Cd1d1", "Cd9", "Dtx1", "S1pr3"],
        "T": ["Cd3e", "Cd3g", "Cd3d", "Cd4", "Cd8b1"],
        "myeloid": ["Lyz2", "C1qa", "C1qb", "Cd68", "Csf1r"],
        "erythroid": ["Car2", "Alas2", "Gypa", "Klf1", "Hbb-bs"],
    },
    "immune_human_no_ig": {
        "GC B": ["AICDA", "S1PR2", "RGS13", "GCSAM", "MEF2B", "BCL6", "MYBL1", "LMO2"],
        "plasmablast/ASC": ["JCHAIN", "XBP1", "PRDM1", "IRF4", "MZB1"],
        "plasma cell": ["SDC1", "TNFRSF17", "SLAMF7"],
        "follicular B": ["FCER2", "SELL", "KLF2", "MS4A1"],
        "MZ B": ["CR2", "CD1C", "CD9", "DTX1", "S1PR3"],
        "T": ["CD3E", "CD3G", "CD3D", "CD4", "CD8B"],
        "myeloid": ["LYZ", "C1QA", "C1QB", "CD68", "CSF1R"],
        "erythroid": ["CA2", "ALAS2", "GYPA", "KLF1", "HBB"],
    },
}

#: Readout sets reported next to the cell types (never used for the label): cycling (plasmablast vs plasma cell) and
#: switched Ig RNA (a GEX check of a class-switch call).
PROFILE_READOUTS: dict[str, dict[str, list[str]]] = {
    "immune_mouse_no_ig": {
        "cycling": ["Mki67", "Top2a", "Stmn1", "Birc5"],
        "switched Ig": ["Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha"],
    },
    "immune_human_no_ig": {
        "cycling": ["MKI67", "TOP2A", "STMN1", "BIRC5"],
        "switched Ig": ["IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2"],
    },
}

#: Immunoglobulin locus genes (heavy and light constant, V, D, J and the surrogate light chain), any case.
_IG = re.compile(
    r"^(IGH[MDGAE][0-9A-Z]{0,3}|IGH[VDJ][0-9A-Z][-0-9A-Z/.]*|IG[KL][CVJ][0-9A-Z]?[-0-9A-Z/.]*|IGLL\d*)$", re.I
)


class ProfileWarning(SpacloneWarning):
    """A marker set lost genes (Ig genes dropped, genes absent from the data) or a region has no window units."""


def is_ig_gene(names: Sequence[str] | pd.Index) -> np.ndarray:
    """Whether each gene name is an immunoglobulin locus gene (``Ighg2b``, ``IGKC``, ``Ighv1-26``, ``Iglc1`` ...).

    Parameters
    ----------
    names
        Gene symbols (mouse or human; case is ignored).

    Returns
    -------
    numpy.ndarray
        Boolean, one value per name. ``Jchain`` and genes such as ``Ighmbp2`` are not Ig locus genes.
    """
    return np.array([bool(_IG.match(str(n))) and not str(n).upper().startswith("IGHMBP") for n in names], dtype=bool)


def profile_markers(
    markers: str | Mapping[str, Sequence[str]] = "immune_mouse_no_ig",
    readouts: str | Mapping[str, Sequence[str]] | None = "auto",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Resolve the cell-type and readout sets of :func:`profile_regions` and :func:`module_scores`.

    Parameters
    ----------
    markers
        Name of a built-in set (:data:`PROFILE_MARKERS`) or ``{set name: genes}``.
    readouts
        ``"auto"``: the built-in readouts of a built-in ``markers`` name (none for a custom mapping); a built-in name;
        a mapping; or ``None``.

    Returns
    -------
    cell_types, readouts
        Two ``{set name: genes}`` dictionaries; Ig genes are removed from the cell-type sets (with a warning).
    """
    if isinstance(markers, str):
        if markers not in PROFILE_MARKERS:
            raise ValueError(f"unknown marker set {markers!r}; built-in sets: {sorted(PROFILE_MARKERS)}")
        cells = {k: list(v) for k, v in PROFILE_MARKERS[markers].items()}
    else:
        cells = {str(k): [str(g) for g in v] for k, v in markers.items()}
    if not cells:
        raise ValueError("markers is empty")
    if readouts == "auto":
        reads = {k: list(v) for k, v in PROFILE_READOUTS.get(markers, {}).items()} if isinstance(markers, str) else {}
    elif readouts is None:
        reads = {}
    elif isinstance(readouts, str):
        if readouts not in PROFILE_READOUTS:
            raise ValueError(f"unknown readout set {readouts!r}; built-in sets: {sorted(PROFILE_READOUTS)}")
        reads = {k: list(v) for k, v in PROFILE_READOUTS[readouts].items()}
    else:
        reads = {str(k): [str(g) for g in v] for k, v in readouts.items()}
    clash = set(cells) & set(reads)
    if clash:
        raise ValueError(f"set names used both as cell type and readout: {sorted(clash)}")
    for name, genes in cells.items():
        ig = [g for g, f in zip(genes, is_ig_gene(genes), strict=True) if f]
        if ig:
            warnings.warn(
                f"cell-type set {name!r}: immunoglobulin genes {ig} removed (a region's label must not depend on the "
                "Ig RNA that the BCR isotype calls also measure)",
                ProfileWarning,
                stacklevel=3,
            )
            cells[name] = [g for g in genes if g not in ig]
        if not cells[name]:
            raise ValueError(f"cell-type set {name!r} has no gene left")
    return cells, reads


# --------------------------------------------------------------------------------------------------- helpers


def _present(var_names: pd.Index, genes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    pos = var_names.get_indexer(list(genes))
    return pos[pos >= 0], [g for g, p in zip(genes, pos, strict=True) if p < 0]


def _set_columns(counts: sparse.csr_matrix, var_names: pd.Index, sets: Mapping[str, Sequence[str]]):
    """Per-unit UMIs of each set and the missing genes."""
    umi, missing = {}, {}
    for name, genes in sets.items():
        pos, miss = _present(var_names, genes)
        missing[name] = miss
        umi[name] = (
            np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel() if len(pos) else np.zeros(counts.shape[0])
        )
    return umi, missing


def _warn_missing(missing: Mapping[str, list[str]], sets: Mapping[str, Sequence[str]]) -> None:
    empty = [k for k, v in missing.items() if len(v) == len(sets[k])]
    part = {k: v for k, v in missing.items() if v and k not in empty}
    if part:
        logger.info("marker genes not in the data: %s", part)
    if empty:
        warnings.warn(f"no gene of set(s) {empty} is in the data; their scores are 0", ProfileWarning, stacklevel=3)


# ------------------------------------------------------------------------------------------------- unit selection


def unit_mask(adata: ad.AnnData, units: Any = None) -> np.ndarray:
    """Boolean selection of GEX units from a column, a column and its values, or one flag per unit.

    Parameters
    ----------
    adata
        Spot-level AnnData.
    units
        ``None``: every unit. A string: a boolean (or 0/1) ``obs`` column. ``(column, values)``: the units whose
        ``obs[column]`` is one of ``values`` (one value or a list, compared as strings), e.g.
        ``("cell2loc_anno", ["B_cells"])``. A boolean array with one value per unit, or a boolean Series indexed by
        ``obs_names`` (every unit must be present).

    Returns
    -------
    numpy.ndarray
        One boolean per unit, in ``obs`` order.

    Raises
    ------
    KeyError
        If the column is not in ``obs``.
    ValueError
        If a single column is not boolean (pass ``(column, values)``), or the flags do not match the units.
    """
    n = adata.n_obs
    if units is None:
        return np.ones(n, dtype=bool)
    if isinstance(units, str):
        if units not in adata.obs.columns:
            raise KeyError(f"units: obs column {units!r} not found")
        col = adata.obs[units]
        if pd.api.types.is_bool_dtype(col.dtype):
            return col.fillna(False).to_numpy(dtype=bool)
        if pd.api.types.is_numeric_dtype(col.dtype) and set(pd.unique(col.dropna())) <= {0, 1}:
            return col.fillna(0).to_numpy() != 0
        raise ValueError(
            f"units: obs column {units!r} is not boolean; pass (column, values), e.g. ({units!r}, ['B_cells'])"
        )
    if isinstance(units, tuple) and len(units) == 2 and isinstance(units[0], str):
        col_name, values = units
        if col_name not in adata.obs.columns:
            raise KeyError(f"units: obs column {col_name!r} not found")
        vals = [values] if isinstance(values, str) or np.ndim(values) == 0 else list(values)
        if not vals:
            raise ValueError("units: (column, values) needs at least one value")
        col = adata.obs[col_name].astype(object)
        out = col.notna().to_numpy() & col.astype(str).isin([str(v) for v in vals]).to_numpy()
        if not out.any():
            seen = list(pd.unique(col.dropna().astype(str)))[:8]
            raise ValueError(f"units: no unit has obs[{col_name!r}] in {vals} (values present: {seen})")
        return out
    if isinstance(units, pd.Series):
        if not units.index.equals(adata.obs_names):
            missing = adata.obs_names.difference(units.index)
            if len(missing):
                raise ValueError(f"units: the Series lacks {len(missing)} units (e.g. {list(missing[:3])})")
            units = units.reindex(adata.obs_names)
        units = units.to_numpy()
    arr = np.asarray(units)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"units: expected one flag per unit ({n}), got shape {arr.shape}")
    if arr.dtype != bool:
        if not np.isin(pd.unique(arr), [0, 1, True, False]).all():
            raise ValueError("units: flags must be boolean (or 0/1)")
        arr = arr.astype(bool)
    return arr


# ------------------------------------------------------------------------------------------------- module scores


def module_scores(
    adata: ad.AnnData,
    markers: str | Mapping[str, Sequence[str]] = "immune_mouse_no_ig",
    *,
    readouts: str | Mapping[str, Sequence[str]] | None = "auto",
    smooth_um: float | None = 22.5,
    tissue_key: str | None = None,
    units: Any = None,
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    key_added: str | None = None,
) -> pd.DataFrame:
    """Per-unit marker-module scores, smoothed over a small neighbourhood (maps of the archive's T8f).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts (``layer``, else ``layers["counts"]``, else ``X``) and ``obsm[spatial_key]``
        in µm.
    markers
        Built-in name (:data:`PROFILE_MARKERS`) or ``{set name: genes}``.
    readouts
        Readout sets scored as well (see :func:`profile_markers`).
    smooth_um
        Radius (µm, inclusive) of the neighbourhood mean, the unit itself included; ``None`` or 0 = no smoothing.
    tissue_key
        Boolean ``obs`` column: only these units are z-scored and smoothed (others get NaN). ``None``: every unit.
    units
        Further restriction of the scored units (see :func:`unit_mask`): a boolean ``obs`` column, ``(column,
        values)`` such as ``("cell2loc_anno", ["B_cells"])``, or one flag per unit. Only the selected tissue units are
        z-scored, and a unit's neighbourhood mean uses selected neighbours only (the archive's figure scored B-lineage
        bins only and greyed the others); the others get NaN. ``None``: every tissue unit.
    library_key
        ``obs`` column naming the section; neighbourhoods stay within a section.
    layer
        Layer with raw counts.
    spatial_key
        ``obsm`` key of the coordinates.
    key_added
        If given, the scores are also written to ``obs[f"{key_added}_{set}"]``.

    Returns
    -------
    pandas.DataFrame
        One row per unit (index = ``obs_names``), one column per set (cell types, then readouts):
        ``mean_g z(log1p(CP10k_g))`` averaged over the unit and its neighbours. CP10k uses the unit's total counts
        over all genes; z uses the mean and the population standard deviation over the scored units (genes without
        variance count as 0).

    Notes
    -----
    Descriptive (no test). Missing genes are skipped; a set without any present gene scores 0.

    References
    ----------
    Tirosh et al. 2016, doi:10.1126/science.aad0501.
    """
    cells, reads = profile_markers(markers, readouts)
    sets = {**cells, **reads}
    counts = get_counts(adata, layer)
    names = pd.Index(adata.var_names).astype(str)
    use = _bool_obs(adata, tissue_key) if tissue_key is not None else np.ones(adata.n_obs, dtype=bool)
    use = use & unit_mask(adata, units)
    idx = np.flatnonzero(use)
    sub = counts[idx]
    tot = np.asarray(counts.sum(axis=1), dtype=float).ravel()[idx]
    scale = 1e4 / np.maximum(tot, 1.0)
    w = None
    if smooth_um is not None and smooth_um > 0 and len(idx):
        xy = get_xy(adata, spatial_key)[idx]
        groups = adata.obs[library_key].astype(str).to_numpy()[idx] if library_key is not None else None
        a = radius_graph(xy, float(smooth_um), groups=groups)
        a = (a + sparse.identity(len(idx), format="csr")).tocsr()
        deg = np.asarray(a.sum(axis=1)).ravel()
        w = (sparse.diags(1.0 / deg) @ a).tocsr()
    out = pd.DataFrame(index=adata.obs_names.copy())
    missing = {}
    for name, genes in sets.items():
        pos, miss = _present(names, genes)
        missing[name] = miss
        col = np.full(adata.n_obs, np.nan)
        if len(pos) and len(idx):
            lg = np.log1p(sub[:, pos].toarray() * scale[:, None])
            sd = lg.std(axis=0)
            sd[sd == 0] = 1.0
            z = ((lg - lg.mean(axis=0)) / sd).mean(axis=1)
            col[idx] = w @ z if w is not None else z
        elif len(idx):
            col[idx] = 0.0
        out[name] = col
    _warn_missing(missing, sets)
    if key_added is not None:
        for name in sets:
            adata.obs[f"{key_added}_{name}"] = out[name].to_numpy()
    out.attrs["spaclone"] = {
        "module_scores": {
            "cell_types": cells,
            "readouts": reads,
            "smooth_um": smooth_um,
            "tissue_key": tissue_key,
            "units": units if (units is None or isinstance(units, str | tuple)) else "flags per unit",
            "n_scored": len(idx),
        }
    }
    return out


# ----------------------------------------------------------------------------------------------- region profiles


@dataclass(frozen=True, repr=False)
class RegionProfiles:
    """Result of :func:`profile_regions`.

    Attributes
    ----------
    table
        One row per region and marker set: ``region_id``, ``set``, ``is_cell_type``, ``n_units_inside``,
        ``n_units_window``, ``n_units_tissue``, ``umi_inside``, ``umi_window``, ``umi_tissue``, ``cp10k_inside``,
        ``cp10k_window``, ``cp10k_tissue`` (UMIs of the set per 10,000 UMIs of all genes, pooled over the units),
        ``ratio_window`` and ``ratio_tissue`` (inside over window / tissue; NaN when the denominator is 0),
        ``frac_units_detected_inside`` (share of inside units with at least one UMI of the set).
    labels
        One row per region: ``region_id``, ``label`` (top set, or ``"<top> / <second> (mixed)"``), ``top``,
        ``top_ratio_window``, ``top_ratio_tissue``, ``second``, ``second_ratio_window``, ``mixed``,
        ``n_units_inside``, ``n_units_window``, ``ranking`` (``"set=ratio;..."`` by decreasing ratio to the window) and
        one ``<readout>_ratio_window`` column per readout set.
    units
        Per GEX unit (index = ``obs_names``): ``region_id`` (0 = no region; the first region when masks overlap) and
        ``in_window`` (in the window of at least one region, outside every region).
    params
        Marker sets, missing genes, window rule, ``mixed_log2``.
    """

    table: pd.DataFrame
    labels: pd.DataFrame
    units: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        lab = "; ".join(f"{r.region_id}: {r.label}" for r in self.labels.itertuples()) or "no region"
        return f"RegionProfiles({len(self.labels)} regions: {lab})"


def _region_masks(adata: ad.AnnData, regions: Any, spatial_key: str) -> tuple[list[Any], np.ndarray]:
    """Region ids and an (n_units, n_regions) boolean membership matrix."""
    from spaclone.tl.structures import Raster

    n = adata.n_obs
    if isinstance(regions, Raster):
        xy = get_xy(adata, spatial_key)
        lab = np.asarray(regions.sample(xy[:, 0], xy[:, 1], fill=0))
        return _from_labels(lab)
    if isinstance(regions, str):
        if regions not in adata.obs.columns:
            raise KeyError(f"regions: obs column {regions!r} not found")
        return _from_labels(adata.obs[regions].to_numpy())
    if isinstance(regions, pd.DataFrame):
        if len(regions) != n:
            raise ValueError(f"regions: a mask table needs one row per unit ({n}), got {len(regions)}")
        if not regions.index.equals(adata.obs_names):
            regions = regions.reindex(adata.obs_names)
            if regions.isna().any().any():
                raise ValueError("regions: the mask table's index does not match obs_names")
        return list(regions.columns), regions.fillna(False).to_numpy(dtype=bool)
    if isinstance(regions, Mapping):
        ids = list(regions)
        m = np.column_stack([np.asarray(regions[k], dtype=bool) for k in ids]) if ids else np.zeros((n, 0), bool)
        if m.shape[0] != n:
            raise ValueError(f"regions: each mask needs one value per unit ({n})")
        return ids, m
    arr = np.asarray(regions)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError("regions must be a label Raster, an obs column, per-unit labels, or masks per region")
    return _from_labels(arr)


def _from_labels(lab: np.ndarray) -> tuple[list[Any], np.ndarray]:
    s = pd.Series(lab)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() == s.notna().sum():
        inside = num.notna() & (num > 0)
        vals = num.where(inside)
        ids = sorted(int(v) if float(v).is_integer() else float(v) for v in vals.dropna().unique())
    else:
        inside = s.notna() & (s.astype(str) != "") & (s.astype(str) != "0")
        vals = s.where(inside)
        ids = sorted(vals.dropna().astype(str).unique())
        vals = vals.astype(object).where(inside)
    m = np.zeros((len(s), len(ids)), dtype=bool)
    v = vals.to_numpy(dtype=object)
    for k, r in enumerate(ids):
        m[:, k] = np.array([x == r for x in v], dtype=bool) if len(v) else np.zeros(0, bool)
    return ids, m


def _window_masks(
    adata: ad.AnnData, window: Any, ids: list[Any], inside: np.ndarray, spatial_key: str
) -> tuple[np.ndarray, pd.DataFrame | None, str]:
    """(n_units, n_regions) window masks (before removing region units), the boxes used, and the rule in words."""
    n = adata.n_obs
    if isinstance(window, int | float | np.integer | np.floating) and not isinstance(window, bool):
        pad = float(window)
        if pad < 0:
            raise ValueError("window pad must be >= 0 µm")
        xy = get_xy(adata, spatial_key)
        rows = []
        for k, r in enumerate(ids):
            p = xy[inside[:, k]]
            if not len(p):
                rows.append({"region_id": r, "x0": np.nan, "x1": np.nan, "y0": np.nan, "y1": np.nan})
                continue
            lo, hi = p.min(axis=0) - pad, p.max(axis=0) + pad
            rows.append({"region_id": r, "x0": lo[0], "x1": hi[0], "y0": lo[1], "y1": hi[1]})
        boxes = pd.DataFrame(rows, columns=["region_id", "x0", "x1", "y0", "y1"])
        return _box_masks(xy, boxes, ids), boxes, f"bounding box of the region's units +/- {pad:g} um"
    if isinstance(window, pd.DataFrame) and {"x0", "x1", "y0", "y1"} <= set(window.columns):
        boxes = window.copy()
        if "region_id" not in boxes.columns:
            boxes = boxes.reset_index().rename(columns={boxes.index.name or "index": "region_id"})
        xy = get_xy(adata, spatial_key)
        return _box_masks(xy, boxes, ids), boxes, "given boxes (x0, x1, y0, y1 um)"
    if isinstance(window, Mapping):
        miss = [r for r in ids if r not in window]
        if miss:
            raise KeyError(f"window: no mask for region(s) {miss}")
        m = np.column_stack([np.asarray(window[r], dtype=bool) for r in ids]) if ids else np.zeros((n, 0), bool)
        if m.shape[0] != n:
            raise ValueError(f"window: each mask needs one value per unit ({n})")
        return m, None, "given unit masks"
    if isinstance(window, pd.DataFrame):
        miss = [r for r in ids if r not in window.columns]
        if miss:
            raise KeyError(f"window: no mask column for region(s) {miss}")
        w = window.reindex(adata.obs_names) if not window.index.equals(adata.obs_names) else window
        return w[ids].fillna(False).to_numpy(dtype=bool), None, "given unit masks"
    raise ValueError("window must be a pad in µm, a box table (region_id, x0, x1, y0, y1) or unit masks per region")


def _box_masks(xy: np.ndarray, boxes: pd.DataFrame, ids: list[Any]) -> np.ndarray:
    look = boxes.set_index("region_id")
    out = np.zeros((len(xy), len(ids)), dtype=bool)
    for k, r in enumerate(ids):
        if r not in look.index:
            raise KeyError(f"window: no box for region {r!r}")
        b = look.loc[r]
        if not np.isfinite([b.x0, b.x1, b.y0, b.y1]).all():
            continue
        out[:, k] = (xy[:, 0] >= b.x0) & (xy[:, 0] <= b.x1) & (xy[:, 1] >= b.y0) & (xy[:, 1] <= b.y1)
    return out


def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def profile_regions(
    adata: ad.AnnData,
    regions: Any,
    *,
    markers: str | Mapping[str, Sequence[str]] = "immune_mouse_no_ig",
    readouts: str | Mapping[str, Sequence[str]] | None = "auto",
    window: float | pd.DataFrame | Mapping[Any, Any] = 100.0,
    exclude: Literal["regions", "own"] = "regions",
    mixed_log2: float = 0.3,
    tissue_key: str | None = "in_tissue",
    layer: str | None = None,
    spatial_key: str = "spatial",
) -> RegionProfiles:
    """Which cell type does each region overlap? Marker pseudobulk inside vs around vs whole tissue (archive T8f).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts (``layer``, else ``layers["counts"]``, else ``X``) and ``obsm[spatial_key]``
        in µm.
    regions
        The regions: a label :class:`~spaclone.tl.Raster` (id > 0 inside, sampled at the units), an ``obs`` column or
        an array of per-unit region ids (0, NaN or ``""`` = none), or masks: a DataFrame with one boolean column per
        region (index = ``obs_names``) or ``{region id: boolean array}`` (masks may overlap).
    markers
        Cell-type sets: a built-in name (:data:`PROFILE_MARKERS`) or ``{set name: genes}``; Ig genes are removed.
    readouts
        Readout sets reported but never used for the label (see :func:`profile_markers`).
    window
        The surroundings of each region: a pad in µm around the bounding box of the region's units, a box table
        (``region_id``, ``x0``, ``x1``, ``y0``, ``y1`` in µm), or unit masks (as ``regions``, one per region).
    exclude
        Units removed from every window: ``"regions"`` (default, the archive's rule) = units inside any region;
        ``"own"`` = units of the region itself only.
    mixed_log2
        The label is ``"<top> / <second> (mixed)"`` when ``log2(top ratio / second ratio)`` is below this (0.3 =
        1.23×).
    tissue_key
        Boolean ``obs`` column of tissue units; regions, windows and the whole-tissue reference use tissue units only.
        ``None`` or an absent column: every unit.
    layer
        Layer with raw counts.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    RegionProfiles
        ``table`` (region × set), ``labels`` (one row per region), ``units`` (per-unit region and window flags) and
        ``params``.

    Notes
    -----
    Descriptive, no test: the ratios compare pooled counts and carry no uncertainty; with few units inside a region
    (fewer than ~30) or a sparse window, treat the label as a hint. The rule is the archive's T8f: pseudobulk UMIs per
    10,000 UMIs of each set inside the region vs the rest of its window (other regions excluded) and vs the whole
    tissue; label = cell-type set with the highest ratio to the window, mixed when the second is within 1.23×.

    References
    ----------
    Squair et al. 2021, doi:10.1038/s41467-021-25960-2.

    Examples
    --------
    >>> prof = profile_regions(gex, hot.labels, window=100.0)  # doctest: +SKIP
    >>> prof.labels[["region_id", "label", "top_ratio_window"]]  # doctest: +SKIP
    """
    if exclude not in ("regions", "own"):
        raise ValueError("exclude must be 'regions' or 'own'")
    if not mixed_log2 >= 0:
        raise ValueError("mixed_log2 must be >= 0")
    cells, reads = profile_markers(markers, readouts)
    sets = {**cells, **reads}
    counts = get_counts(adata, layer)
    names = pd.Index(adata.var_names).astype(str)
    tissue = _bool_obs(adata, tissue_key) if (tissue_key is not None and tissue_key in adata.obs) else None
    tissue = np.ones(adata.n_obs, dtype=bool) if tissue is None else tissue
    ids, inside = _region_masks(adata, regions, spatial_key)
    inside = inside & tissue[:, None]
    win_raw, boxes, win_rule = _window_masks(adata, window, ids, inside, spatial_key)
    any_region = inside.any(axis=1)
    win = win_raw & tissue[:, None] & (~any_region[:, None] if exclude == "regions" else ~inside)
    tot = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    umi, missing = _set_columns(counts, names, sets)
    _warn_missing(missing, sets)
    tot_t = tot[tissue].sum()
    rows = []
    lab_rows = []
    for k, r in enumerate(ids):
        mi, mw = inside[:, k], win[:, k]
        if not mw.any():
            warnings.warn(
                f"region {r!r} has no unit in its window: ratios to the window are NaN", ProfileWarning, stacklevel=2
            )
        ti, tw = tot[mi].sum(), tot[mw].sum()
        ratios = {}
        for s in sets:
            u = umi[s]
            ui, uw, ut = float(u[mi].sum()), float(u[mw].sum()), float(u[tissue].sum())
            ci = ui / ti * 1e4 if ti > 0 else np.nan
            cw = uw / tw * 1e4 if tw > 0 else np.nan
            ct = ut / tot_t * 1e4 if tot_t > 0 else np.nan
            rw = ci / cw if (np.isfinite(cw) and cw > 0) else np.nan
            rt = ci / ct if (np.isfinite(ct) and ct > 0) else np.nan
            ratios[s] = (rw, rt)
            rows.append(
                {
                    "region_id": r,
                    "set": s,
                    "is_cell_type": s in cells,
                    "n_units_inside": int(mi.sum()),
                    "n_units_window": int(mw.sum()),
                    "n_units_tissue": int(tissue.sum()),
                    "umi_inside": ui,
                    "umi_window": uw,
                    "umi_tissue": ut,
                    "cp10k_inside": ci,
                    "cp10k_window": cw,
                    "cp10k_tissue": ct,
                    "ratio_window": rw,
                    "ratio_tissue": rt,
                    "frac_units_detected_inside": float((u[mi] > 0).mean()) if mi.any() else np.nan,
                }
            )
        rc = pd.Series({s: ratios[s][0] for s in cells}, dtype=float).sort_values(ascending=False, kind="stable")
        top = rc.index[0] if len(rc) and np.isfinite(rc.iloc[0]) else None
        second = rc.index[1] if len(rc) > 1 and np.isfinite(rc.iloc[1]) else None
        if top is None:
            label, mixed = "undetermined", False
        elif second is None or rc.iloc[1] <= 0:
            label, mixed = top, False
        else:
            mixed = bool(np.log2(rc.iloc[0] / rc.iloc[1]) < mixed_log2)
            label = f"{top} / {second} (mixed)" if mixed else top
        lr = {
            "region_id": r,
            "label": label,
            "top": top,
            "top_ratio_window": float(rc.iloc[0]) if top is not None else np.nan,
            "top_ratio_tissue": float(ratios[top][1]) if top is not None else np.nan,
            "second": second,
            "second_ratio_window": float(rc.iloc[1]) if second is not None else np.nan,
            "mixed": mixed,
            "n_units_inside": int(mi.sum()),
            "n_units_window": int(mw.sum()),
            "ranking": ";".join(f"{s}={v:.2f}" for s, v in rc.items()),
        }
        for s in reads:
            lr[f"{_safe_name(s)}_ratio_window"] = ratios[s][0]
        lab_rows.append(lr)
    table = pd.DataFrame(rows)
    if not len(table):
        table = pd.DataFrame(
            columns=[
                "region_id", "set", "is_cell_type", "n_units_inside", "n_units_window", "n_units_tissue",
                "umi_inside", "umi_window", "umi_tissue", "cp10k_inside", "cp10k_window", "cp10k_tissue",
                "ratio_window", "ratio_tissue", "frac_units_detected_inside",
            ]
        )  # fmt: skip
    lab_cols = [
        "region_id", "label", "top", "top_ratio_window", "top_ratio_tissue", "second", "second_ratio_window",
        "mixed", "n_units_inside", "n_units_window", "ranking", *[f"{_safe_name(s)}_ratio_window" for s in reads],
    ]  # fmt: skip
    labels = pd.DataFrame(lab_rows, columns=lab_cols)
    first = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    region_of = np.array([ids[i] if i >= 0 else 0 for i in first], dtype=object)
    units = pd.DataFrame(
        {"region_id": region_of, "in_window": win.any(axis=1), "tissue": tissue}, index=adata.obs_names.copy()
    )
    params = {
        "cell_types": cells,
        "readouts": reads,
        "missing_genes": {k: v for k, v in missing.items() if v},
        "window": win_rule,
        "boxes": boxes,
        "exclude": exclude,
        "mixed_log2": mixed_log2,
        "tissue_key": tissue_key if tissue_key in adata.obs else None,
        "layer": layer,
    }
    return RegionProfiles(table=table, labels=labels, units=units, params=params)
