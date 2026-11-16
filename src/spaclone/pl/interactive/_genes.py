"""Gene expression inside an interactive figure: which genes, and how their values are stored.

**Which genes** (``genes=`` of :func:`spaclone.pl.explore` and :func:`spaclone.pl.interactive.gex_map`): a list of
names (exactly those; names not in the data are reported), a number ``k`` (the ``k`` most variable genes of the drawn
units, plus the immune marker genes of :data:`spaclone.gex.MARKERS` and the immunoglobulin constant genes found in
the data), ``"all"``, or ``0`` / ``None`` (no gene picker). The most variable genes are ranked by the normalised
dispersion of log1p(counts per 10,000) (variance over mean, as a z-score within 20 bins of mean expression, the
rule of Satija et al. 2015), among genes detected in at least 0.5% of the units (and 10 units).

**How they are stored**: as a sparse column per gene (only the units where the gene is detected), unit positions
as differences in variable-length bytes, all compressed with zlib. Integer counts are stored exactly (one byte when
every count is below 256, else two); the figure then shows either the counts or log1p(counts per 10,000) computed in
the browser from them and the unit's total counts (exact up to float rounding). Values that are not counts (already
normalised data) are stored as 8-bit levels of each gene's own range [0, max]: a stored value is within max / 510 of
the original (half a level); the colour bar says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from spaclone.pl.interactive import _encode as E

#: Immunoglobulin constant and joining-chain genes always offered when present (mouse and human symbols).
IG_GENES = ("Ighm", "Ighd", "Ighg3", "Ighg1", "Ighg2b", "Ighg2c", "Ighg2a", "Ighe", "Igha", "Igkc", "Iglc1", "Iglc2",
            "Jchain", "IGHM", "IGHD", "IGHG3", "IGHG1", "IGHA1", "IGHG2", "IGHG4", "IGHE", "IGHA2", "IGKC", "IGLC1",
            "IGLC2", "JCHAIN")  # fmt: skip


def matrix_of(adata: Any, rows: np.ndarray, cols: np.ndarray, layer: str | None) -> sparse.csc_matrix:
    """``adata`` values of ``rows`` x ``cols`` as CSC (in-memory, backed, sparse or dense matrices)."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if getattr(adata, "isbacked", False):
        sub = adata[np.sort(rows)].to_memory()
        X = sub.layers[layer] if layer is not None else sub.X
        order = np.argsort(np.argsort(rows))
        X = X[order]
    else:
        X = adata.layers[layer] if layer is not None else adata.X
        X = X[rows]
    X = X[:, cols]
    if sparse.issparse(X):
        return sparse.csc_matrix(X)
    return sparse.csc_matrix(np.asarray(X))


def _totals(adata: Any, rows: np.ndarray, layer: str | None) -> np.ndarray:
    """Total counts of the drawn units (``obs["total_counts"]`` when present, else the sum over every gene)."""
    if layer is None and "total_counts" in adata.obs.columns:
        t = pd.to_numeric(adata.obs["total_counts"], errors="coerce").to_numpy(dtype=float)[rows]
        if np.isfinite(t).all() and (t > 0).all():
            return t
    X = matrix_of(adata, rows, np.arange(adata.n_vars), layer)
    return np.asarray(X.sum(axis=1)).ravel().astype(float)


def _is_counts(X: sparse.spmatrix) -> bool:
    d = X.data
    return bool(len(d) == 0 or (np.all(d >= 0) and np.all(np.equal(np.mod(d, 1), 0))))


def select_genes(
    units: Sequence[tuple[Any, np.ndarray]],
    genes: Any,
    *,
    layer: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """The genes a figure carries, and how they were chosen.

    Parameters
    ----------
    units
        ``(adata, rows)`` per sample: the drawn units.
    genes
        See the module notes.
    layer
        AnnData layer of the values (default ``X``).

    Returns
    -------
    list, dict
        Gene names (in the order the picker lists them) and a report (``rule``, ``asked``, ``missing``, ``n_genes``).
    """
    if not units:
        return [], {"rule": "no transcriptome", "n_genes": 0}
    names0 = [str(g) for g in units[0][0].var_names]
    common = set(names0)
    for ad_, _ in units[1:]:
        common &= {str(g) for g in ad_.var_names}
    universe = [g for g in names0 if g in common]
    if genes is None or (isinstance(genes, int | np.integer) and not isinstance(genes, bool) and int(genes) == 0):
        return [], {"rule": "none", "n_genes": 0}
    if isinstance(genes, str):
        if genes == "all":
            return universe, {"rule": "all", "n_genes": len(universe)}
        genes = [genes]
    if not isinstance(genes, int | np.integer):
        asked = [str(g) for g in genes]
        low = {g.lower(): g for g in universe}
        found, missing = [], []
        for g in asked:
            hit = g if g in common else low.get(g.lower())
            (found if hit is not None else missing).append(hit if hit is not None else g)
        return list(dict.fromkeys(found)), {"rule": "list", "asked": len(asked), "missing": missing,
                                             "n_genes": len(set(found))}  # fmt: skip
    k = int(genes)
    if k < 0:
        raise ValueError("genes must be a non-negative number, a list of names, 'all' or None")
    hv = _variable(units, universe, layer)
    chosen = hv[:k]
    from spaclone.gex.annotate import MARKERS

    markers = [g for sets in MARKERS.values() for gs in sets.values() for g in gs] + list(IG_GENES)
    extra = [g for g in dict.fromkeys(markers) if g in common and g not in chosen]
    return chosen + extra, {
        "rule": f"top {k} variable genes + {len(extra)} marker genes",
        "n_genes": len(chosen) + len(extra),
    }


def _variable(units: Sequence[tuple[Any, np.ndarray]], universe: list[str], layer: str | None) -> list[str]:
    """Genes by normalised dispersion of log1p(counts per 10,000), pooled over the samples' drawn units."""
    n_g = len(universe)
    s1 = np.zeros(n_g)
    s2 = np.zeros(n_g)
    det = np.zeros(n_g)
    n = 0
    for ad_, rows in units:
        cols = ad_.var_names.get_indexer(universe)
        X = matrix_of(ad_, rows, cols, layer).tocsr().astype(np.float64)
        tot = np.asarray(X.sum(axis=1)).ravel()
        if _is_counts(X):
            scale = np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0.0)
            X = sparse.diags(scale) @ X
            X.data = np.log1p(X.data)
        s1 += np.asarray(X.sum(axis=0)).ravel()
        s2 += np.asarray(X.multiply(X).sum(axis=0)).ravel()
        det += np.asarray((X > 0).sum(axis=0)).ravel()
        n += X.shape[0]
    if n == 0:
        return []
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.log(np.where(mean > 0, var / mean, np.nan))
    ok = np.isfinite(disp) & (det >= max(10, 0.005 * n))
    z = np.full(n_g, -np.inf)
    if ok.any():
        bins = pd.qcut(pd.Series(mean[ok]), q=min(20, int(ok.sum())), labels=False, duplicates="drop").to_numpy()
        dv = disp[ok]
        zz = np.zeros(ok.sum())
        for b in np.unique(bins):
            m = bins == b
            sd = dv[m].std()
            zz[m] = (dv[m] - dv[m].mean()) / sd if sd > 0 else 0.0
        z[ok] = zz
    order = np.lexsort((np.arange(n_g), -z))
    return [universe[i] for i in order if np.isfinite(z[i])]


def encode(
    spec: Any,
    key: str,
    adata: Any,
    rows: np.ndarray,
    names: Sequence[str],
    *,
    layer: str | None = None,
    unit_word: str = "units",
) -> dict[str, Any]:
    """Store the genes ``names`` of the units ``rows`` of ``adata`` in ``spec`` under ``key``; returns the entry."""
    cols = adata.var_names.get_indexer(list(names))
    X = matrix_of(adata, rows, cols, layer)
    X.sort_indices()
    if len(X.data) and np.nanmin(X.data) < 0:
        raise ValueError(
            "gene values must be non-negative (counts or log-normalised values); the data hold negative values "
            "(scaled data?): pass layer= with counts or log-normalised values"
        )
    counts = _is_counts(X)
    ind, ptr = X.indices.astype(np.int64), X.indptr.astype(np.int64)
    d = ind.copy()
    if len(d):
        d[1:] -= ind[:-1]
        starts = ptr[:-1][ptr[1:] > ptr[:-1]]
        d[starts] = ind[starts]
    raw = E.varint_bytes(d)
    nb = np.ones(len(d), dtype=np.int64)
    t = d >> 7
    while (t > 0).any():
        nb += t > 0
        t >>= 7
    boff = np.concatenate([[0], np.cumsum(nb)])
    bptr = boff[ptr]
    entry: dict[str, Any] = {"names": spec.table(list(names)), "n": int(len(rows)), "unit_word": unit_word}
    entry["ptr"] = spec.blob(ptr, "u4")
    entry["bptr"] = spec.blob(bptr, "u4")
    entry["idx"] = spec.raw_blob(E.encode_bytes(raw, compress=True))
    if counts:
        vals = np.rint(X.data).astype(np.int64)
        vmax = int(vals.max()) if len(vals) else 0
        dt = "u1" if vmax <= 255 else ("u2" if vmax <= 65535 else "f4")
        entry["val"] = spec.blob(vals, dt, compress=True)
        entry["counts"] = True
        entry["totals"] = spec.blob(_totals(adata, rows, layer), "f4")
        entry["default_norm"] = "norm"
    else:
        data = X.data.astype(float)
        gmax = np.zeros(len(names))
        for g in range(len(names)):
            seg = data[ptr[g] : ptr[g + 1]]
            gmax[g] = seg.max() if len(seg) else 0.0
        scale = np.where(gmax > 0, gmax / 255.0, 1.0)
        per = np.repeat(scale, np.diff(ptr))
        codes = np.clip(np.rint(data / per), 0, 255).astype(np.int64)
        entry["val"] = spec.blob(codes, "u1", compress=True)
        entry["scale"] = spec.blob(scale, "f4")
        entry["counts"] = False
        entry["unit"] = f"value ({layer or 'X'})"
    spec.genes[key] = entry
    spec.arrays[f"genes:{key}"] = X  # for the static picture and the tests
    return entry


def values(spec: Any, key: str, name: str, *, norm: bool = True) -> np.ndarray:
    """A gene's values per drawn unit as the figure computes them (for tests and the static picture)."""
    entry = spec.genes[key]
    names = spec.tables[entry["names"]]
    g = names.index(name)
    X = spec.arrays[f"genes:{key}"]
    col = np.asarray(X[:, g].toarray()).ravel().astype(float)
    if entry.get("counts"):
        if norm:
            tot = spec.arrays[entry["totals"]].astype(float)
            col = np.log1p(col / np.where(tot > 0, tot, 1.0) * 1e4)
        return col
    sc = float(spec.arrays[entry["scale"]][g])
    return np.rint(col / sc) * sc
