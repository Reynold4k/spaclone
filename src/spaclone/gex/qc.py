"""Quality control of the spot-level transcriptome: metrics, tissue calls, off-tissue-aware filters and square bins.

Spots, bins, beads and segmented cells are the observations of an :class:`anndata.AnnData` whose
``obsm["spatial"]`` holds micrometre coordinates. Off-tissue units are kept by default: the location-trust tests
(``spaclone.qc``) need them as the reference for what an empty capture area looks like.

The private helpers at the bottom of this module (count matrix, coordinates, radius sums, parameter records) are
shared by every ``spaclone.gex`` module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from spaclone._utils import SpacloneWarning

__all__ = ["bin", "bin_ids", "call_tissue", "filter", "qc_metrics"]

logger = logging.getLogger("spaclone")

SPATIAL = "spatial"


class CountsWarning(SpacloneWarning):
    """The matrix used as raw counts holds non-integer values (it may already be normalised)."""


# ------------------------------------------------------------------------------------------------------------ public


def qc_metrics(
    adata: ad.AnnData,
    *,
    mito_prefix: str | Sequence[str] = ("mt-", "MT-"),
    in_tissue_key: str | None = "in_tissue",
    library_key: str | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Per-unit and per-gene QC metrics (scanpy's ``calculate_qc_metrics``), summarised on and off tissue.

    Parameters
    ----------
    adata
        Spot-level AnnData (spots, bins, beads or cells as ``obs``).
    mito_prefix
        Gene-name prefix(es) of mitochondrial genes; marked in ``var["mt"]``.
    in_tissue_key
        Boolean ``obs`` column separating tissue from off-tissue units; the summary is split by it when present.
    library_key
        ``obs`` column naming the section / library; the summary is split by it when given.
    layer
        Layer holding raw counts. By default ``layers["counts"]`` when present, else ``X``.

    Returns
    -------
    pandas.DataFrame
        One row per (library, tissue) group: ``n_units``, ``median_counts``, ``median_genes``, ``median_pct_mt``,
        ``total_counts`` and ``share_of_counts`` (share of all counts of the library).

    Notes
    -----
    Writes the scanpy columns ``obs["total_counts"]``, ``obs["n_genes_by_counts"]``, ``obs["total_counts_mt"]``,
    ``obs["pct_counts_mt"]`` and ``var["mt"]``, ``var["n_cells_by_counts"]``, ``var["mean_counts"]``,
    ``var["pct_dropout_by_counts"]``, ``var["total_counts"]``; parameters go to ``uns["spaclone"]["qc_metrics"]``.

    References
    ----------
    Wolf, Angerer & Theis 2018, doi:10.1186/s13059-017-1382-0 (scanpy); McCarthy et al. 2017,
    doi:10.1093/bioinformatics/btw777 (scater QC metrics).
    """
    import scanpy as sc

    layer = _counts_layer(adata, layer)
    prefixes = (mito_prefix,) if isinstance(mito_prefix, str) else tuple(mito_prefix)
    adata.var["mt"] = adata.var_names.astype(str).str.startswith(prefixes) if prefixes else False
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, layer=layer, inplace=True)
    _record(
        adata, "qc_metrics", {"mito_prefix": list(prefixes), "layer": layer, "n_mito_genes": int(adata.var.mt.sum())}
    )
    obs = adata.obs
    frame = pd.DataFrame(
        {
            "library": obs[library_key].astype(str).to_numpy() if library_key is not None else "all",
            "tissue": "all",
            "total": obs["total_counts"].to_numpy(dtype=float),
            "genes": obs["n_genes_by_counts"].to_numpy(dtype=float),
            "pct_mt": obs["pct_counts_mt"].to_numpy(dtype=float),
        },
        index=obs.index,
    )
    if in_tissue_key is not None and in_tissue_key in obs:
        frame["tissue"] = np.where(_bool_obs(adata, in_tissue_key), "in_tissue", "off_tissue")
    frame["lib_total"] = frame.groupby("library").total.transform("sum").to_numpy()
    out = frame.groupby(["library", "tissue"], sort=True).agg(
        n_units=("total", "size"),
        median_counts=("total", "median"),
        median_genes=("genes", "median"),
        median_pct_mt=("pct_mt", "median"),
        total_counts=("total", "sum"),
        lib_total=("lib_total", "first"),
    )
    out["share_of_counts"] = out.total_counts / out.lib_total.where(out.lib_total > 0)
    return out.drop(columns="lib_total").reset_index()


def call_tissue(
    adata: ad.AnnData,
    *,
    min_counts: float = 100,
    radius_um: float = 100.0,
    min_neighbors: int = 15,
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = SPATIAL,
    key_added: str = "in_tissue",
) -> pd.DataFrame:
    """Call tissue units from the transcriptome: enough counts, and enough such units around.

    A unit is in tissue if it has at least ``min_counts`` counts and at least ``min_neighbors`` *other* units with
    ``min_counts`` counts lie within ``radius_um`` (same library). The second condition removes isolated beads
    ("smear") outside the section. With the defaults this is the AIR-SPACE rule used by the archive (X3): 100
    counts, 15 neighbours within 100 µm.

    Parameters
    ----------
    adata
        Spot-level AnnData with ``obsm[spatial_key]`` in µm.
    min_counts
        Minimum counts of a unit (and of the neighbours that count).
    radius_um
        Neighbourhood radius (µm, inclusive).
    min_neighbors
        Minimum number of other qualifying units within ``radius_um``.
    library_key
        ``obs`` column naming the section; neighbours are searched within a section only.
    layer
        Layer holding raw counts (default ``layers["counts"]`` when present, else ``X``).
    spatial_key
        ``obsm`` key of the coordinates.
    key_added
        Boolean ``obs`` column written.

    Returns
    -------
    pandas.DataFrame
        Per library: ``n_units``, ``n_min_counts`` (units with enough counts) and ``n_tissue``.

    References
    ----------
    Jiang et al. 2026, doi:10.1073/pnas.2504742123 (AIR-SPACE; beads with < 15 neighbours within 100 µm removed).
    """
    counts = get_counts(adata, layer)
    total = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    xy = get_xy(adata, spatial_key)
    libs = _groups(adata, library_key)
    ok = total >= min_counts
    tissue = np.zeros(adata.n_obs, dtype=bool)
    rows = []
    for lib, idx in libs.items():
        sel = idx[ok[idx]]
        if len(sel):
            tree = cKDTree(xy[sel])
            nn = tree.query_ball_point(xy[sel], r=radius_um, return_length=True) - 1
            tissue[sel[nn >= min_neighbors]] = True
        rows.append({"library": lib, "n_units": len(idx), "n_min_counts": len(sel), "n_tissue": int(tissue[idx].sum())})
    adata.obs[key_added] = tissue
    _record(
        adata,
        "call_tissue",
        {"min_counts": min_counts, "radius_um": radius_um, "min_neighbors": min_neighbors, "key_added": key_added},
    )
    return pd.DataFrame(rows)


def filter(
    adata: ad.AnnData,
    *,
    min_counts: float | None = None,
    max_counts: float | None = None,
    min_genes: int | None = None,
    max_mito: float | None = None,
    min_units_per_gene: int | None = None,
    off_tissue: Literal["keep", "filter", "drop"] = "keep",
    in_tissue_key: str | None = "in_tissue",
    layer: str | None = None,
) -> ad.AnnData:
    """Filter units (and optionally genes) by QC thresholds without silently losing off-tissue units.

    Parameters
    ----------
    adata
        Spot-level AnnData. Missing metrics are computed with :func:`qc_metrics`.
    min_counts, max_counts
        Bounds on ``obs["total_counts"]``.
    min_genes
        Minimum ``obs["n_genes_by_counts"]``.
    max_mito
        Maximum ``obs["pct_counts_mt"]`` in percent (0–100).
    min_units_per_gene
        Remove genes detected (count > 0) in fewer kept units.
    off_tissue
        ``"keep"`` (default): thresholds apply to tissue units only; off-tissue units are kept, because the
        location-trust tests use them. ``"filter"``: thresholds apply to every unit. ``"drop"``: remove every
        off-tissue unit. Without ``in_tissue_key`` every unit counts as tissue.
    in_tissue_key
        Boolean ``obs`` column of tissue units.
    layer
        Layer holding raw counts (for metrics and the gene filter).

    Returns
    -------
    anndata.AnnData
        A filtered copy; ``uns["spaclone"]["filter"]`` holds the thresholds and how many units each one removed.
    """
    if off_tissue not in ("keep", "filter", "drop"):
        raise ValueError(f"off_tissue must be 'keep', 'filter' or 'drop', got {off_tissue!r}")
    need = {"total_counts", "n_genes_by_counts", "pct_counts_mt"}
    if not need <= set(adata.obs.columns):
        qc_metrics(adata, in_tissue_key=in_tissue_key, layer=layer)
    obs = adata.obs
    has_tissue = in_tissue_key is not None and in_tissue_key in obs
    tissue = _bool_obs(adata, in_tissue_key) if has_tissue else np.ones(adata.n_obs, dtype=bool)
    eligible = tissue if off_tissue == "keep" else np.ones(adata.n_obs, dtype=bool)
    fails: dict[str, np.ndarray] = {}
    total = obs["total_counts"].to_numpy(dtype=float)
    if min_counts is not None:
        fails["min_counts"] = total < min_counts
    if max_counts is not None:
        fails["max_counts"] = total > max_counts
    if min_genes is not None:
        fails["min_genes"] = obs["n_genes_by_counts"].to_numpy(dtype=float) < min_genes
    if max_mito is not None:
        fails["max_mito"] = np.nan_to_num(obs["pct_counts_mt"].to_numpy(dtype=float), nan=0.0) > max_mito
    drop = np.zeros(adata.n_obs, dtype=bool)
    report: dict[str, Any] = {"n_units_before": int(adata.n_obs), "off_tissue": off_tissue}
    for name, f in fails.items():
        report[f"removed_{name}"] = int((f & eligible).sum())
        drop |= f & eligible
    if off_tissue == "drop" and has_tissue:
        report["removed_off_tissue"] = int((~tissue).sum())
        drop |= ~tissue
    out = adata[~drop].copy()
    report["n_units_after"] = int(out.n_obs)
    report["n_off_tissue_kept"] = int((~tissue[~drop]).sum()) if has_tissue else 0
    if min_units_per_gene is not None:
        det = np.asarray((get_counts(out, layer) > 0).sum(axis=0)).ravel()
        keep_g = det >= min_units_per_gene
        report["removed_genes"] = int((~keep_g).sum())
        out = out[:, keep_g].copy()
    report.update(
        {
            "min_counts": min_counts,
            "max_counts": max_counts,
            "min_genes": min_genes,
            "max_mito": max_mito,
            "min_units_per_gene": min_units_per_gene,
        }
    )
    _record(out, "filter", report)
    logger.info("gex.filter: %d -> %d units", adata.n_obs, out.n_obs)
    return out


def bin_ids(
    x_um: Any, y_um: Any, *, bin_um: float, origin_um: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Integer square-bin indices of positions: ``ix = floor((x - x0) / bin_um)`` and the same along y.

    With the default origin (0, 0) a Stereo-seq ``binN`` (DNB pitch 0.5 µm) is ``bin_um = N / 2``:
    ``floor(x_um / bin_um) = floor(x_dnb / N)``, the Stereo-seq bin of the DNB. Use the same call to put receptor
    records into the bins made by :func:`bin`.

    Parameters
    ----------
    x_um, y_um
        Coordinates in µm.
    bin_um
        Bin side in µm.
    origin_um
        Lower corner of bin (0, 0).

    Returns
    -------
    ix, iy
        ``int64`` arrays.
    """
    if not bin_um > 0:
        raise ValueError("bin_um must be positive")
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("coordinates must be finite")
    return (
        np.floor((x - origin_um[0]) / bin_um).astype(np.int64),
        np.floor((y - origin_um[1]) / bin_um).astype(np.int64),
    )


def bin(
    adata: ad.AnnData,
    *,
    bin_um: float,
    origin_um: tuple[float, float] = (0.0, 0.0),
    library_key: str | None = None,
    layer: str | None = None,
    in_tissue_key: str | None = "in_tissue",
    position: Literal["centre", "mean"] = "centre",
    spatial_key: str = SPATIAL,
    key_added: str | None = "bin_id",
) -> ad.AnnData:
    """Aggregate units (DNBs, spots, beads) into square bins by summing their counts.

    Parameters
    ----------
    adata
        Spot-level AnnData with ``obsm[spatial_key]`` in µm and raw counts.
    bin_um
        Bin side in µm (Stereo-seq bin20 = 10 µm, bin50 = 25 µm, bin100 = 50 µm).
    origin_um
        Lower corner of bin (0, 0); see :func:`bin_ids`.
    library_key
        ``obs`` column naming the section; bins never mix sections.
    layer
        Layer holding raw counts (default ``layers["counts"]`` when present, else ``X``).
    in_tissue_key
        Boolean ``obs`` column; a bin is in tissue when at least half of its units are
        (``obs["in_tissue_frac"]`` keeps the share).
    position
        Bin coordinate: the bin ``"centre"`` or the ``"mean"`` position of its units.
    spatial_key
        ``obsm`` key of the coordinates.
    key_added
        If given, ``adata.obs[key_added]`` receives every unit's bin id (the input is modified), so receptor records
        can be mapped to the same bins.

    Returns
    -------
    anndata.AnnData
        One observation per non-empty bin; obs index ``"{ix}_{iy}"`` (prefixed ``"{library}|"`` with
        ``library_key``), columns ``bin_ix``, ``bin_iy``, ``n_units`` (and the library and tissue columns);
        ``X`` = summed counts (CSR); ``obsm["spatial"]`` in µm; ``var`` copied.
    """
    counts = get_counts(adata, layer)
    xy = get_xy(adata, spatial_key)
    ix, iy = bin_ids(xy[:, 0], xy[:, 1], bin_um=bin_um, origin_um=origin_um)
    keys = pd.DataFrame({"ix": ix, "iy": iy})
    if library_key is not None:
        keys.insert(0, "lib", adata.obs[library_key].astype(str).to_numpy())
    grouped = keys.groupby(list(keys.columns), sort=True)
    codes = grouped.ngroup().to_numpy()
    u = grouped.size().reset_index()[list(keys.columns)]
    n_bins = len(u)
    member = sparse.csr_matrix((np.ones(adata.n_obs), (codes, np.arange(adata.n_obs))), shape=(n_bins, adata.n_obs))
    xb = (member @ counts).tocsr()
    names = u["ix"].astype(str) + "_" + u["iy"].astype(str)
    if library_key is not None:
        names = u["lib"].astype(str) + "|" + names
    n_units = np.asarray(member.sum(axis=1)).ravel()
    obs = pd.DataFrame(
        {
            "bin_ix": u["ix"].to_numpy(np.int64),
            "bin_iy": u["iy"].to_numpy(np.int64),
            "n_units": n_units.astype(np.int64),
        },
        index=pd.Index(names.to_numpy(), dtype=str),
    )
    if library_key is not None:
        obs[library_key] = u["lib"].to_numpy()
    if in_tissue_key is not None and in_tissue_key in adata.obs:
        frac = (member @ _bool_obs(adata, in_tissue_key).astype(float)) / n_units
        obs["in_tissue_frac"] = frac
        obs[in_tissue_key] = frac >= 0.5
    if position == "centre":
        pos = np.column_stack([origin_um[0] + (obs.bin_ix + 0.5) * bin_um, origin_um[1] + (obs.bin_iy + 0.5) * bin_um])
    elif position == "mean":
        pos = (member @ xy) / n_units[:, None]
    else:
        raise ValueError(f"position must be 'centre' or 'mean', got {position!r}")
    out = ad.AnnData(X=xb, obs=obs, var=adata.var.copy())
    out.obsm[SPATIAL] = np.asarray(pos, dtype=float)
    if key_added is not None:
        adata.obs[key_added] = pd.Categorical.from_codes(codes, categories=obs.index)
    params = {
        "bin_um": bin_um,
        "origin_um": list(origin_um),
        "position": position,
        "n_units": int(adata.n_obs),
        "n_bins": int(n_bins),
        "library_key": library_key,
    }
    src = adata.uns.get("spaclone", {}) if isinstance(adata.uns.get("spaclone"), dict) else {}
    out.uns["spaclone"] = {k: v for k, v in src.items() if k in ("source_unit", "um_per_unit", "technology")}
    _record(out, "bin", params)
    return out


# ------------------------------------------------------------------------------------------- shared private helpers


def _counts_layer(adata: ad.AnnData, layer: str | None) -> str | None:
    """Layer holding raw counts: the given one, else ``"counts"`` when present, else ``None`` (= ``X``)."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not in adata.layers ({list(adata.layers)})")
        return layer
    return "counts" if "counts" in adata.layers else None


def get_counts(adata: ad.AnnData, layer: str | None = None, *, check: bool = True) -> sparse.csr_matrix:
    """Raw count matrix as CSR (``layer``, else ``layers["counts"]``, else ``X``); warns if it is not integer."""
    name = _counts_layer(adata, layer)
    m = adata.layers[name] if name is not None else adata.X
    if m is None:
        raise ValueError("adata has no expression matrix")
    m = sparse.csr_matrix(m) if not sparse.issparse(m) else m.tocsr()
    if check and m.nnz:
        sample = m.data[: min(m.nnz, 100_000)]
        if not np.allclose(sample, np.round(sample)):
            import warnings

            warnings.warn(
                f"the matrix used as raw counts ({name or 'X'}) holds non-integer values; pass layer= with raw counts",
                CountsWarning,
                stacklevel=3,
            )
    return m


def get_xy(adata: ad.AnnData, spatial_key: str = SPATIAL) -> np.ndarray:
    """``(n, 2)`` float coordinates in µm from ``obsm[spatial_key]``."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] (coordinates in µm) is missing")
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"obsm[{spatial_key!r}] must have two columns (x, y in µm)")
    xy = xy[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError(f"obsm[{spatial_key!r}] contains NaN or infinite coordinates")
    return xy


def _bool_obs(adata: ad.AnnData, key: str | None) -> np.ndarray:
    """Boolean array of an ``obs`` column (all True when ``key`` is None or absent)."""
    if key is None or key not in adata.obs:
        return np.ones(adata.n_obs, dtype=bool)
    v = adata.obs[key]
    if v.dtype == bool:
        return v.to_numpy()
    if pd.api.types.is_numeric_dtype(v):
        return v.fillna(0).to_numpy() != 0
    s = v.astype(str).str.lower()
    return s.isin(["true", "1", "yes", "in_tissue", "tissue"]).to_numpy()


def _groups(adata: ad.AnnData, library_key: str | None) -> dict[Any, np.ndarray]:
    """Row positions of each library (one group "all" without ``library_key``)."""
    if library_key is None:
        return {"all": np.arange(adata.n_obs)}
    if library_key not in adata.obs:
        raise KeyError(f"library_key {library_key!r} not in adata.obs")
    lab = adata.obs[library_key].astype(str).to_numpy()
    return dict(pd.Series(np.arange(adata.n_obs)).groupby(lab, sort=True).indices.items())


def gene_index(var_names: Sequence[str] | pd.Index, genes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Positions of the ``genes`` present in ``var_names`` (in the given order) and the list of missing ones."""
    vn = pd.Index(var_names).astype(str)
    pos = vn.get_indexer(list(genes))
    return pos[pos >= 0], [g for g, p in zip(genes, pos, strict=True) if p < 0]


def gene_sum(counts: sparse.csr_matrix, var_names: Sequence[str] | pd.Index, genes: Sequence[str]) -> np.ndarray:
    """Per-unit sum of the counts of ``genes`` (missing genes are skipped; zeros if none is present)."""
    pos, _ = gene_index(var_names, genes)
    if not len(pos):
        return np.zeros(counts.shape[0])
    return np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel()


def radius_graph(xy: np.ndarray, radius_um: float, *, groups: np.ndarray | None = None) -> sparse.csr_matrix:
    """Symmetric 0/1 adjacency of unit pairs within ``radius_um`` (inclusive), self excluded, within groups."""
    n = len(xy)
    if groups is None:
        pairs = cKDTree(xy).query_pairs(radius_um, output_type="ndarray")
    else:
        parts = []
        for idx in pd.Series(np.arange(n)).groupby(np.asarray(groups)).indices.values():
            if len(idx) > 1:
                p = cKDTree(xy[idx]).query_pairs(radius_um, output_type="ndarray")
                parts.append(idx[p])
        pairs = np.vstack(parts) if parts else np.empty((0, 2), dtype=np.int64)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def radius_sum(xy: np.ndarray, values: np.ndarray, radius_um: float, *, groups: np.ndarray | None = None) -> np.ndarray:
    """Sum of ``values`` (1-D or 2-D, rows = units) over the units within ``radius_um`` of each unit, itself included."""
    if radius_um <= 0:
        return np.asarray(values, dtype=float).copy()
    a = radius_graph(xy, radius_um, groups=groups)
    v = np.asarray(values, dtype=float)
    return a @ v + v


def _record(adata: ad.AnnData, step: str, params: dict[str, Any]) -> None:
    """Store the parameters of a step in ``adata.uns["spaclone"][step]``."""
    store = adata.uns.get("spaclone")
    if not isinstance(store, dict):
        store = {}
    store[step] = _jsonable(params)
    adata.uns["spaclone"] = store


def _jsonable(v: Any) -> Any:
    """Make a parameter record writable to h5ad: drop None entries, tuples to lists, numpy scalars to Python."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if x is not None}
    if isinstance(v, list | tuple):
        return [("none" if x is None else _jsonable(x)) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v
