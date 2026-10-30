"""Markov transition models on spatial, expression or joint neighbour graphs, and transition path theory between states.

A *transition model* turns a time-like score per unit (a CSR potential, an SHM score, any pseudotime) and a neighbour
graph into a Markov chain whose steps prefer later units. It re-implements the kernels sciCSR uses through CellRank 1.x
(Ng et al. 2023): the pseudotime kernel (``threshold_scheme="hard"``: the Palantir-style removal of edges into the
pseudotime past that keeps the closest ``frac_to_keep`` neighbours; ``"soft"``: the VIA-style down-weighting of past
edges) combined with the density-normalised connectivity kernel of the same graph, ``T = w T_time + (1 - w) T_conn``
(sciCSR: ``w = 0.8``). The units can be cells, spots, beads or inferred cells (loci), and the graph is the tissue
(``graph="spatial"``: the k nearest units in µm with Gaussian weights), the transcriptome (``"expression"``: a
scanpy kNN graph, the sciCSR / CellRank setting) or both (``"joint"``), so the chain moves between places as well as
between expression states. ``backend="cellrank"`` computes the same matrices with an installed CellRank (1.x or 2.x).

:func:`tpt_flux` applies transition path theory (TPT) between a source and a target set of states, which may be
isotypes, cell types or tissue regions (follicle, germinal centre, ASC foci, red pulp from
:func:`spaclone.gex.call_regions`): TPT runs on the unit-level chain and is coarse-grained onto the states, exactly
as sciCSR's ``fitTPT`` does (forward and backward committors, gross flux ``pi_i q-_i T_ij q+_j``, net flux,
stationary distribution, rate, mean first-passage time and dominant pathways), and every pairwise flux is tested
against a stated randomisation null. :func:`transition_displacement` projects the transition matrix onto tissue
coordinates (expected displacement per unit and step, in µm) and :func:`flow_grid` averages it on a grid for arrow
or stream plots (:func:`spaclone.pl.csr_flow`).

References
----------
Ng et al. 2023, doi:10.1038/s41592-023-02060-1 (sciCSR); Lange et al. 2022, doi:10.1038/s41592-021-01346-6
(CellRank); Setty et al. 2019, doi:10.1038/s41587-019-0068-4 (Palantir); Stassen et al. 2021,
doi:10.1038/s41467-021-25773-3 (VIA); E & Vanden-Eijnden 2006, doi:10.1007/s10955-005-9003-9; Metzner, Schütte &
Vanden-Eijnden 2009, doi:10.1137/070699500 (TPT for Markov jump processes); Noé et al. 2009,
doi:10.1073/pnas.0905466106 (fluxes, coarse-graining, pathways); Hoffmann et al. 2022, doi:10.1088/2632-2153/ac3de0
(deeptime); Bergen et al. 2020, doi:10.1038/s41587-020-0591-3 (scVelo embedding projection); Hummer & Szabo 2015,
doi:10.1021/jp508375q (stationary-weighted lumping); Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree
from scipy.stats import norm

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    LowPowerWarning,
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    mc_pvalue,
    null_summary,
    rng_entropy,
)

__all__ = [
    "FluxResult",
    "TransitionModel",
    "coarse_grain",
    "flow_grid",
    "neighbor_graph",
    "stationary_distribution",
    "tpt_flux",
    "transition_displacement",
    "transition_model",
]

logger = logging.getLogger("spaclone")

GraphKind = Literal["spatial", "expression", "joint"]
NullKind = Literal["time_within_states", "time", "labels"]

#: CellRank's hard-threshold scheme never keeps more than this many "closest" neighbours regardless of time.
_MAX_CLOSE = 30


# ===================================================================================================== input helpers
def _is_adata(data: Any) -> bool:
    return hasattr(data, "obs") and hasattr(data, "obsm") and hasattr(data, "obs_names")


def _obs_names(data: Any) -> pd.Index:
    if _is_adata(data):
        return pd.Index(data.obs_names.astype(str))
    if isinstance(data, pd.DataFrame):
        return pd.Index(data.index.astype(str))
    raise TypeError(f"expected an AnnData or a pandas DataFrame of units, got {type(data).__name__}")


def _column(data: Any, key: str | Sequence[Any] | np.ndarray | pd.Series, what: str) -> np.ndarray:
    """A per-unit column given by name (obs / DataFrame column) or as values."""
    n = len(_obs_names(data))
    if isinstance(key, str):
        table = data.obs if _is_adata(data) else data
        if key not in table.columns:
            where = "obs" if _is_adata(data) else "the units table"
            raise KeyError(f"{what}: column {key!r} not in {where}")
        vals = table[key]
        return vals.to_numpy() if not isinstance(vals.dtype, pd.CategoricalDtype) else np.asarray(vals.astype(object))
    vals = key.to_numpy() if isinstance(key, pd.Series) else np.asarray(key)
    if len(vals) != n:
        raise ValueError(f"{what}: {len(vals)} values for {n} units")
    return vals


def _coords(data: Any, *, spatial_key: str, x_key: str, y_key: str, required: bool) -> np.ndarray | None:
    if _is_adata(data):
        if spatial_key in data.obsm:
            xy = np.asarray(data.obsm[spatial_key], dtype=float)[:, :2]
            return xy
    elif isinstance(data, pd.DataFrame) and x_key in data.columns and y_key in data.columns:
        return np.column_stack([data[x_key].to_numpy(dtype=float), data[y_key].to_numpy(dtype=float)])
    if required:
        where = f"obsm[{spatial_key!r}]" if _is_adata(data) else f"columns {x_key!r} / {y_key!r}"
        raise KeyError(f"spatial coordinates not found ({where}); pass spatial_key / x_key / y_key")
    return None


def _as_states(values: Any) -> pd.Categorical:
    if isinstance(values, pd.Categorical):
        cat = values
    elif isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        cat = values.array
    else:
        arr = np.asarray(values, dtype=object)
        if pd.isna(arr).any():
            raise ValueError("states: every unit needs a state label (found missing values)")
        cat = pd.Categorical(arr.astype(str))
    if pd.isna(np.asarray(cat, dtype=object)).any():
        raise ValueError("states: every unit needs a state label (found missing values)")
    return cat.remove_unused_categories()


def _as_csr(m: Any, n: int, what: str) -> sparse.csr_matrix:
    out = sparse.csr_matrix(m, dtype=float)
    if out.shape != (n, n):
        raise ValueError(f"{what}: expected a {n} x {n} matrix, got {out.shape}")
    return out


# ===================================================================================================== graphs
def _knn_pairs(
    xy: np.ndarray, k: int, max_distance_um: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed k-nearest-neighbour pairs (rows, cols, distances) with ties at the k-th distance included.

    Every unit within the distance of its k-th nearest other unit is a neighbour, so equidistant units on regular
    grids are all kept and the graph does not depend on the order of the units. Returns also the k-th distance.
    """
    n = len(xy)
    if n < 2:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0), np.zeros(n)
    kk = int(min(k, n - 1))
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=kk + 1)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    d_other = np.where(idx != np.arange(n)[:, None], dist, np.inf)
    dk = np.sort(d_other, axis=1)[:, kk - 1]
    radius = dk * (1 + 1e-9) + 1e-9
    if max_distance_um is not None:
        radius = np.minimum(radius, float(max_distance_um) * (1 + 1e-12))
    lists = tree.query_ball_point(xy, r=radius)
    lens = np.fromiter((len(v) for v in lists), dtype=np.int64, count=n)
    rows = np.repeat(np.arange(n), lens)
    cols = np.concatenate([np.asarray(v, dtype=np.int64) for v in lists]) if lens.sum() else np.zeros(0, np.int64)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    d = np.sqrt(((xy[rows] - xy[cols]) ** 2).sum(axis=1))
    return rows, cols, d, dk


def _spatial_connectivities(
    xy: np.ndarray, *, k: int, bandwidth_um: float | None, max_distance_um: float | None
) -> tuple[sparse.csr_matrix, float]:
    """Symmetric Gaussian weights on the k-nearest-neighbour graph of the coordinates (µm), ties included."""
    n = len(xy)
    if n < 2:
        return sparse.csr_matrix((n, n)), float("nan")
    rows, cols, d, dk = _knn_pairs(xy, k, max_distance_um)
    if bandwidth_um is None:
        h = float(np.median(dk)) / 2.0
        if not np.isfinite(h) or h <= 0:
            h = 1.0
    else:
        h = float(bandwidth_um)
        if not h > 0:
            raise ValueError(f"bandwidth_um must be positive, got {bandwidth_um}")
    w = np.exp(-(d**2) / (2.0 * h * h))
    a = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    c = a.maximum(a.T).tocsr()  # union of the directed kNN edges; the weight is symmetric in the distance
    c.setdiag(0.0)
    c.eliminate_zeros()
    c.sort_indices()
    return c, h


def _expression_connectivities(
    adata: Any, *, k: int, use_rep: str | None, n_pcs: int | None, connectivities_key: str, rng: RngLike
) -> sparse.csr_matrix:
    """``obsp[connectivities_key]`` if present, else a scanpy kNN graph (PCA on ``X`` unless ``use_rep``)."""
    n = adata.n_obs
    if connectivities_key in adata.obsp:
        return _as_csr(adata.obsp[connectivities_key], n, f"obsp[{connectivities_key!r}]")
    import anndata as ad
    import scanpy as sc

    seed = int(as_rng(rng).integers(2**31 - 1))
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"use_rep {use_rep!r} not in obsm")
        tmp = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names), obsm={use_rep: np.asarray(adata.obsm[use_rep])})
    else:
        tmp = ad.AnnData(X=adata.X, obs=pd.DataFrame(index=adata.obs_names))
        sc.pp.pca(tmp, n_comps=min(50, tmp.n_obs - 1, tmp.n_vars - 1), random_state=seed)
        use_rep = "X_pca"
    sc.pp.neighbors(tmp, n_neighbors=int(k), use_rep=use_rep, n_pcs=n_pcs, random_state=seed)
    return sparse.csr_matrix(tmp.obsp["connectivities"], dtype=float)


def neighbor_graph(
    data: Any,
    gex: Any = None,
    *,
    graph: GraphKind = "spatial",
    k: int = 30,
    bandwidth_um: float | None = None,
    max_distance_um: float | None = None,
    joint_alpha: float = 0.5,
    connectivities_key: str = "connectivities",
    use_rep: str | None = None,
    n_pcs: int | None = None,
    spatial_key: str = "spatial",
    x_key: str = S.X,
    y_key: str = S.Y,
    rng: RngLike = None,
) -> sparse.csr_matrix:
    """Symmetric neighbour graph (connectivities) of the units in tissue space, expression space or both.

    Parameters
    ----------
    data
        AnnData of units (``obsm[spatial_key]`` in µm; expression in ``X`` or ``obsp[connectivities_key]``) or a units
        table with ``x_key`` / ``y_key`` (µm).
    gex
        AnnData whose ``obs_names`` contain the units' ids, for the expression graph of a units table.
    graph
        ``"spatial"``: the ``k`` nearest units in µm (units tied with the k-th distance included, so the graph does
        not depend on the order of the units), weighted ``exp(-d^2 / (2 h^2))`` and symmetrised by union;
        ``"expression"``: ``obsp[connectivities_key]`` when present (e.g. from ``scanpy.pp.neighbors``), else a scanpy
        kNN graph with ``k`` neighbours on 50 principal components (or ``use_rep``); ``"joint"``:
        ``joint_alpha * spatial + (1 - joint_alpha) * expression``, each scaled to a maximum of 1.
    k
        Neighbours per unit.
    bandwidth_um
        Gaussian bandwidth ``h`` of the spatial weights (default: half the median distance to the k-th neighbour).
    max_distance_um
        Spatial edges longer than this are dropped (default: none), so isolated units do not link to far ones.
    joint_alpha
        Weight of the spatial graph in ``"joint"``.
    connectivities_key, use_rep, n_pcs
        Expression graph options (see ``graph``).
    spatial_key, x_key, y_key
        Coordinates of AnnData (``obsm``) or table units.
    rng
        Seed of scanpy's PCA / neighbour search when the expression graph is computed.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``n x n`` symmetric non-negative weights with a zero diagonal, in the order of the units.
    """
    names = _obs_names(data)
    if graph not in ("spatial", "expression", "joint"):
        raise ValueError(f"graph must be 'spatial', 'expression' or 'joint', got {graph!r}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    cs = ce = None
    if graph in ("spatial", "joint"):
        xy = _coords(data, spatial_key=spatial_key, x_key=x_key, y_key=y_key, required=True)
        cs, _ = _spatial_connectivities(xy, k=k, bandwidth_um=bandwidth_um, max_distance_um=max_distance_um)
    if graph in ("expression", "joint"):
        src = data if _is_adata(data) else gex
        if src is None:
            raise ValueError("an expression graph needs an AnnData (data or gex=)")
        if src is not data:
            missing = names.difference(pd.Index(src.obs_names.astype(str)))
            if len(missing):
                raise KeyError(f"{len(missing)} units not in gex.obs_names (e.g. {missing[0]!r})")
            src = src[np.asarray(names)]
        ce = _expression_connectivities(
            src, k=k, use_rep=use_rep, n_pcs=n_pcs, connectivities_key=connectivities_key, rng=rng
        )
    if graph == "spatial":
        return cs
    if graph == "expression":
        return ce
    if not 0 <= joint_alpha <= 1:
        raise ValueError(f"joint_alpha must lie in [0, 1], got {joint_alpha}")

    def scale(m: sparse.csr_matrix) -> sparse.csr_matrix:
        return m / m.max() if m.nnz and m.max() > 0 else m

    out = (joint_alpha * scale(cs) + (1.0 - joint_alpha) * scale(ce)).tocsr()
    out.setdiag(0.0)
    out.eliminate_zeros()
    return out


# ===================================================================================================== kernels
def _row_normalise(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-stochastic copy; empty rows get a self-loop (CellRank's rule for absorbing units)."""
    m = sparse.csr_matrix(m, dtype=float, copy=True)
    s = np.asarray(m.sum(axis=1)).ravel()
    empty = np.flatnonzero(s == 0)
    if len(empty):
        m = (m + sparse.csr_matrix((np.ones(len(empty)), (empty, empty)), shape=m.shape)).tocsr()
        s = np.asarray(m.sum(axis=1)).ravel()
    return sparse.diags(1.0 / s) @ m


def _biased_connectivities(
    conn: sparse.csr_matrix,
    time: np.ndarray,
    *,
    scheme: str,
    frac_to_keep: float,
    b: float,
    nu: float,
    ties: str = "keep",
) -> sparse.csr_matrix:
    """The pseudotime-biased graph of CellRank's ``PseudotimeKernel``.

    ``ties="keep"`` keeps every neighbour whose weight equals the weakest of the ``k`` strongest (order-invariant);
    ``"cellrank"`` breaks such ties like CellRank's ``np.flip(np.argsort(w))`` with a stable sort (later neighbours in
    the row's storage order first). Without ties the two are identical.
    """
    conn = sparse.csr_matrix(conn, dtype=float)
    n = conn.shape[0]
    indptr, cols, data = conn.indptr, conn.indices, conn.data
    deg = np.diff(indptr)
    row = np.repeat(np.arange(n), deg)
    t_i, t_j = time[row], time[cols]
    if scheme == "hard":
        if not 0 <= frac_to_keep <= 1:
            raise ValueError(f"frac_to_keep must lie in [0, 1], got {frac_to_keep}")
        k_thresh = np.clip(np.floor(deg * frac_to_keep).astype(np.int64), 0, _MAX_CLOSE)
        pos = np.arange(len(data)) - indptr[row]
        # CellRank: np.flip(np.argsort(conn_row)) -> descending weight, ties by descending position in the row
        order = np.lexsort((-pos, -data, row))
        if ties == "cellrank":
            rank = np.empty(len(order), dtype=np.int64)
            rank[order] = np.arange(len(order)) - indptr[row[order]]
            close = rank < k_thresh[row]
        elif ties == "keep":
            thr = np.full(n, np.inf)
            has = (k_thresh > 0) & (deg > 0)
            thr[has] = data[order][indptr[:-1][has] + k_thresh[has] - 1]
            close = data >= thr[row]
        else:
            raise ValueError(f"ties must be 'keep' or 'cellrank', got {ties!r}")
        keep = close | (t_i <= t_j)
        new = np.where(keep, data, 0.0)
    elif scheme == "soft":
        past = t_j < t_i
        w = np.ones_like(data)
        with np.errstate(over="ignore"):
            w[past] = 2.0 / (1.0 + np.exp(b * (t_i[past] - t_j[past]))) ** (1.0 / nu)
        new = data * w
    else:
        raise ValueError(f"threshold_scheme must be 'hard' or 'soft', got {scheme!r}")
    out = sparse.csr_matrix((new, cols.copy(), indptr.copy()), shape=conn.shape)
    out.eliminate_zeros()
    return out


def _connectivity_kernel(conn: sparse.csr_matrix, *, density_normalize: bool) -> sparse.csr_matrix:
    m = sparse.csr_matrix(conn, dtype=float, copy=True)
    if density_normalize:
        q = np.asarray(conn.sum(axis=0)).ravel()
        with np.errstate(divide="ignore"):
            qi = np.where(q > 0, 1.0 / q, 0.0)
        m = (sparse.diags(qi) @ m @ sparse.diags(qi)).tocsr()
    return _row_normalise(m)


def _native_kernels(conn: sparse.csr_matrix, time: np.ndarray, p: dict) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    biased = _biased_connectivities(
        conn, time, scheme=p["threshold_scheme"], frac_to_keep=p["frac_to_keep"], b=p["b"], nu=p["nu"], ties=p["ties"]
    )
    return _row_normalise(biased), _connectivity_kernel(conn, density_normalize=p["density_normalize"])


def _cellrank_kernels(
    conn: sparse.csr_matrix, time: np.ndarray, p: dict
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """The same two kernels from an installed CellRank (2.x ``cellrank.kernels`` or 1.x ``cellrank.tl.kernels``)."""
    try:
        from cellrank.kernels import ConnectivityKernel, PseudotimeKernel  # CellRank >= 2
    except ImportError:
        try:
            from cellrank.tl.kernels import ConnectivityKernel, PseudotimeKernel  # CellRank 1.x (sciCSR's API)
        except ImportError as err:
            raise ImportError("backend='cellrank' needs CellRank (pip install cellrank)") from err
    import anndata as ad

    n = conn.shape[0]
    tmp = ad.AnnData(obs=pd.DataFrame({"_spaclone_time": np.asarray(time, float)}, index=[str(i) for i in range(n)]))
    tmp.obsp["connectivities"] = sparse.csr_matrix(conn, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk = PseudotimeKernel(tmp, time_key="_spaclone_time")
        kw = {"threshold_scheme": p["threshold_scheme"], "show_progress_bar": False}
        if p["threshold_scheme"] == "hard":
            kw["frac_to_keep"] = p["frac_to_keep"]
        else:
            kw.update(b=p["b"], nu=p["nu"])
        pk.compute_transition_matrix(**kw)
        ck = ConnectivityKernel(tmp).compute_transition_matrix(density_normalize=p["density_normalize"])
    return sparse.csr_matrix(pk.transition_matrix, dtype=float), sparse.csr_matrix(ck.transition_matrix, dtype=float)


def _combine(pt: sparse.csr_matrix, ck: sparse.csr_matrix, weight: float) -> sparse.csr_matrix:
    return (weight * pt + (1.0 - weight) * ck).tocsr()


# ===================================================================================================== stationary
def _nd_order(xy: np.ndarray, conn: sparse.csr_matrix, leaf: int = 64) -> np.ndarray:
    """Geometric nested-dissection ordering of spatial units (separators last), for sparse LU with little fill.

    The units are split recursively at the median of the longer coordinate axis; the units of the lower half that
    have a graph neighbour in the upper half form the separator and are eliminated after both halves. On spatial kNN
    graphs this keeps the LU fill near ``n log n`` where general-purpose orderings can take minutes (SuperLU's MMD on
    a 63,000-unit Stereo-seq section: 351 s, against a second or two here).
    """
    n = len(xy)
    adj = sparse.csr_matrix(conn, dtype=bool)
    adj = (adj + adj.T).tocsr()
    out: list[np.ndarray] = []
    stack: list[tuple[np.ndarray, bool]] = [(np.arange(n), False)]
    mark = np.zeros(n, dtype=bool)
    while stack:
        idx, done = stack.pop()
        if done or len(idx) <= leaf:
            out.append(idx)
            continue
        pts = xy[idx]
        axis = int(np.argmax(np.ptp(pts, axis=0)))
        med = np.median(pts[:, axis])
        low = pts[:, axis] <= med
        if low.all() or not low.any():
            out.append(idx)
            continue
        lo, hi = idx[low], idx[~low]
        mark[hi] = True
        touch = np.asarray(adj[lo] @ mark.astype(np.int8)).ravel() > 0
        mark[hi] = False
        sep, core = lo[touch], lo[~touch]
        # LIFO: push the separator first (eliminated last), then the upper half, then the lower half
        stack.append((sep, True))
        stack.append((hi, False))
        stack.append((core, False))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _sub_order(order: np.ndarray | None, keep: np.ndarray, n: int) -> np.ndarray | None:
    """The global ordering restricted to the units ``keep`` (local positions, same relative order)."""
    if order is None:
        return None
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return np.argsort(rank[keep], kind="stable")


class _LU:
    """Sparse LU of a square matrix with solves of ``A x = b`` and ``A' x = b`` (one factorisation, both systems).

    With an ordering (nested dissection of spatial units) the matrix is permuted symmetrically and factorised without
    reordering or pivoting (the matrices are non-singular M-matrices, diagonally dominant); else SuperLU's COLAMD.
    """

    def __init__(self, a: sparse.spmatrix, order: np.ndarray | None = None):
        a = sparse.csc_matrix(a, dtype=float)
        self.order = order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if order is not None:
                ap = a[order][:, order].tocsc()
                self.lu = spla.splu(ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
            else:
                self.lu = spla.splu(a, permc_spec="COLAMD")

    def solve(self, b: np.ndarray, trans: str = "N") -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if self.order is None:
            return np.asarray(self.lu.solve(b, trans=trans), dtype=float)
        x = np.empty_like(b)
        x[self.order] = self.lu.solve(b[self.order], trans=trans)
        return x


def stationary_distribution(transition_matrix: Any, *, order: np.ndarray | None = None) -> np.ndarray:
    """Stationary distribution ``pi`` of an irreducible row-stochastic matrix (``pi T = pi``, ``sum(pi) = 1``).

    Parameters
    ----------
    transition_matrix
        ``n x n`` row-stochastic matrix (dense or sparse).
    order
        Optional elimination ordering of the units (a permutation; spaclone uses a nested dissection of the
        coordinates of spatial units); default: SuperLU's COLAMD.

    Returns
    -------
    numpy.ndarray
        ``pi``, non-negative and summing to 1. Solved directly: with ``pi_r = 1`` for one unit ``r`` (the one with the
        most links), the others solve the sparse non-singular system ``(I - T_{-r,-r})' x = T_{r,-r}'``, which keeps
        the sparsity of ``T`` (a normalisation row would fill the factorisation), and ``pi`` is ``x`` normalised. If
        that system is singular (a reducible chain, whose stationary distribution is not unique) the leading left
        eigenvector is used and a :class:`~spaclone._utils.SpacloneWarning` is raised.
    """
    t = sparse.csr_matrix(transition_matrix, dtype=float)
    n = t.shape[0]
    if t.shape != (n, n):
        raise ValueError(f"transition matrix must be square, got {t.shape}")
    if n == 1:
        return np.ones(1)
    r = int(np.argmax(np.diff(t.indptr) + np.diff(t.tocsc().indptr)))
    keep = np.r_[0:r, r + 1 : n]
    a = (sparse.identity(n - 1, format="csr") - t[keep][:, keep]).T.tocsc()
    rhs = np.asarray(t[r][:, keep].todense()).ravel()
    try:
        x = _LU(a, _sub_order(order, keep, n)).solve(rhs)
    except RuntimeError:
        x = np.full(n - 1, np.nan)
    pi = np.insert(np.asarray(x, dtype=float), r, 1.0)
    if not np.all(np.isfinite(pi)) or pi.min() < -1e-8 * max(1.0, float(np.nanmax(np.abs(pi)))):
        warnings.warn(
            "stationary_distribution: the chain is not irreducible (the graph is disconnected or has absorbing "
            "units), so the stationary distribution is not unique; using the leading left eigenvector",
            SpacloneWarning,
            stacklevel=2,
        )
        vals, vecs = spla.eigs(t.T, k=1, which="LR")
        pi = np.abs(np.real(vecs[:, 0]))
    pi = np.maximum(np.real(pi), 0.0)
    return pi / pi.sum()


def _membership(codes: np.ndarray, m: int) -> sparse.csr_matrix:
    n = len(codes)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, m))


def coarse_grain(model_or_matrix: Any, states: Any = None, *, pi: np.ndarray | None = None) -> pd.DataFrame:
    """State-to-state transition matrix of a unit-level chain, lumped with stationary weights.

    ``P_IJ = sum_{i in I} pi_i sum_{j in J} T_ij / sum_{i in I} pi_i``: the probability that a unit drawn from the
    stationary distribution within state ``I`` steps into state ``J`` (Hummer & Szabo 2015). Rows sum to 1 and the
    lumped stationary distribution ``sum_{i in I} pi_i`` is stationary for ``P``.

    Parameters
    ----------
    model_or_matrix
        A :class:`TransitionModel` or a row-stochastic ``n x n`` matrix.
    states
        State label per unit (default: the model's states).
    pi
        Stationary distribution (computed when omitted).

    Returns
    -------
    pandas.DataFrame
        ``m x m`` matrix indexed by state (rows: from, columns: to).
    """
    if isinstance(model_or_matrix, TransitionModel):
        t = model_or_matrix.transition_matrix
        if states is None:
            if model_or_matrix.states is None:
                raise ValueError("coarse_grain: pass states= (the model has none)")
            states = model_or_matrix.states
        elif isinstance(states, str):
            raise TypeError("coarse_grain: pass the state labels, not a column name")
        if pi is None:
            pi = model_or_matrix.pi
    else:
        t = sparse.csr_matrix(model_or_matrix, dtype=float)
        if states is None:
            raise ValueError("coarse_grain: pass states=")
    cat = _as_states(states)
    if len(cat) != t.shape[0]:
        raise ValueError(f"coarse_grain: {len(cat)} state labels for {t.shape[0]} units")
    if pi is None:
        pi = stationary_distribution(t)
    m = len(cat.categories)
    mem = _membership(np.asarray(cat.codes), m)
    flow = (mem.T @ sparse.diags(pi) @ t @ mem).toarray()
    w = np.asarray(mem.T @ pi).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        p = flow / w[:, None]
    idx = pd.Index(cat.categories.astype(str), name="from_state")
    return pd.DataFrame(p, index=idx, columns=pd.Index(cat.categories.astype(str), name="to_state"))


# ===================================================================================================== the model
@dataclass(frozen=True, eq=False, repr=False)
class TransitionModel:
    """Unit-level Markov chain of :func:`transition_model`.

    Attributes
    ----------
    transition_matrix
        ``T = weight * pseudotime_matrix + (1 - weight) * connectivity_matrix`` (rows sum to 1).
    pseudotime_matrix
        Row-normalised pseudotime-biased graph.
    connectivity_matrix
        Row-normalised (density-normalised) connectivity kernel.
    connectivities
        The symmetric neighbour graph the kernels were built on.
    time
        Time key per unit.
    obs_names
        Unit ids, in matrix order.
    coords
        ``(n, 2)`` coordinates in µm (``None`` when the units have none).
    states
        State label per unit (``None`` unless given).
    params
        Every argument that defines the model.
    dropped
        Ids of units outside the kept graph component.
    """

    transition_matrix: sparse.csr_matrix
    pseudotime_matrix: sparse.csr_matrix
    connectivity_matrix: sparse.csr_matrix
    connectivities: sparse.csr_matrix
    time: np.ndarray
    obs_names: pd.Index
    coords: np.ndarray | None = None
    states: pd.Categorical | None = None
    params: dict[str, Any] = field(default_factory=dict)
    dropped: pd.Index = field(default_factory=lambda: pd.Index([], dtype=object))

    @property
    def n_units(self) -> int:
        """Number of units in the chain."""
        return self.transition_matrix.shape[0]

    @cached_property
    def order(self) -> np.ndarray | None:
        """Nested-dissection elimination ordering of the units (spatial coordinates), or None (no coordinates)."""
        if self.coords is None or self.n_units < 2000:
            return None
        return _nd_order(self.coords, self.connectivities)

    @cached_property
    def pi(self) -> np.ndarray:
        """Stationary distribution of :attr:`transition_matrix` (computed once)."""
        return stationary_distribution(self.transition_matrix, order=self.order)

    def stationary_distribution(self) -> pd.Series:
        """Stationary probability of every unit, indexed by unit id."""
        return pd.Series(self.pi, index=self.obs_names, name="stationary")

    def coarse_grain(self, states: Any = None) -> pd.DataFrame:
        """State-to-state transition matrix (see :func:`coarse_grain`)."""
        return coarse_grain(self, states)

    def __repr__(self) -> str:
        st = "" if self.states is None else f", {len(self.states.categories)} states"
        return (
            f"TransitionModel({self.n_units} units, graph={self.params.get('graph')!r}, time={self.params.get('time_key')!r}, "
            f"weight={self.params.get('weight')}, scheme={self.params.get('threshold_scheme')!r}{st})"
        )


def transition_model(
    data: Any,
    gex: Any = None,
    *,
    time_key: str | Sequence[float] | np.ndarray,
    graph: GraphKind | sparse.spmatrix | np.ndarray = "spatial",
    states: str | Sequence[Any] | None = None,
    weight: float = 0.8,
    k: int = 30,
    threshold_scheme: Literal["hard", "soft"] = "hard",
    frac_to_keep: float = 0.3,
    ties: Literal["keep", "cellrank"] = "keep",
    b: float = 10.0,
    nu: float = 0.5,
    density_normalize: bool = True,
    bandwidth_um: float | None = None,
    max_distance_um: float | None = None,
    joint_alpha: float = 0.5,
    connectivities_key: str = "connectivities",
    use_rep: str | None = None,
    n_pcs: int | None = None,
    component: Literal["largest", "all"] = "largest",
    backend: Literal["native", "cellrank"] = "native",
    spatial_key: str = "spatial",
    x_key: str = S.X,
    y_key: str = S.Y,
    rng: RngLike = None,
) -> TransitionModel:
    """Markov chain of units that prefers steps towards a later time key, on a spatial, expression or joint graph.

    The sciCSR / CellRank 1.x construction, re-implemented: the pseudotime kernel biases the neighbour graph by the
    time key and is row-normalised; the connectivity kernel row-normalises the density-normalised graph
    ``Q C Q`` (``Q = diag(1 / column sums)``); the chain is ``T = weight * T_time + (1 - weight) * T_conn``.

    Parameters
    ----------
    data
        AnnData of units or a units table (e.g. :func:`spaclone.bcr.csr_potential`), one row per cell, spot, bead
        or locus.
    gex
        AnnData holding the units' transcriptomes, for an expression or joint graph of a units table.
    time_key
        Column (``obs`` / table) or values of the time-like score, e.g. ``"csr_pot"`` or ``"shm"``. Missing values
        are not allowed.
    graph
        ``"spatial"``, ``"expression"`` or ``"joint"`` (see :func:`neighbor_graph`), or a precomputed symmetric
        ``n x n`` connectivity matrix (e.g. the ``obsp["connectivities"]`` sciCSR computed, for exact replication).
    states
        Optional column or labels of states (isotypes, cell types, regions) stored for :func:`tpt_flux` and
        :func:`coarse_grain`.
    weight
        Weight of the pseudotime kernel (sciCSR: 0.8); ``1 - weight`` goes to the connectivity kernel. Below 1 the
        chain is irreducible on a connected graph.
    k
        Neighbours per unit of a graph built here (sciCSR's scanpy default is 15; 30 here).
    threshold_scheme
        ``"hard"`` (CellRank default): edges to units with an earlier time are removed, except that the
        ``min(30, floor(frac_to_keep * degree))`` strongest edges of each unit are always kept. ``"soft"``: edges
        into the past are multiplied by ``2 / (1 + exp(b * dt)) ** (1 / nu)``.
    frac_to_keep, b, nu
        Parameters of the two schemes (CellRank defaults).
    ties
        Hard scheme: neighbours tied in weight with the weakest of the always-kept strongest edges. ``"keep"``
        (default) keeps them all, so the chain does not depend on the order of the units; on regular spatial grids
        (Stereo-seq bins, Visium spots) many neighbours are equidistant, and an order-based choice would add a drift
        towards the units listed last. ``"cellrank"`` reproduces CellRank's ``np.flip(np.argsort(w))`` choice (with a
        stable sort). With distinct weights, as in expression graphs, the two are identical.
    density_normalize
        Density-normalise the connectivity kernel (CellRank default True).
    bandwidth_um, max_distance_um, joint_alpha, connectivities_key, use_rep, n_pcs
        Graph options (see :func:`neighbor_graph`).
    component
        ``"largest"``: keep the largest connected component of the graph (the chain of a disconnected graph is
        reducible and has no unique stationary distribution); the other units are listed in ``dropped`` and a
        :class:`~spaclone._utils.SpacloneWarning` is raised. ``"all"``: keep every unit.
    backend
        ``"native"`` (numpy / scipy, this module) or ``"cellrank"`` (an installed CellRank 1.x or 2.x computes the two
        kernels on the same graph; the result agrees with ``"native"`` to floating-point precision).
    spatial_key, x_key, y_key
        Coordinates (µm) used for spatial graphs and kept for :func:`transition_displacement`.
    rng
        Seed of the expression graph's PCA / neighbour search when it is computed here.

    Returns
    -------
    TransitionModel

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Lange et al. 2022, doi:10.1038/s41592-021-01346-6; Setty et al.
    2019, doi:10.1038/s41587-019-0068-4; Stassen et al. 2021, doi:10.1038/s41467-021-25773-3.

    Examples
    --------
    >>> pot = spaclone.bcr.csr_potential(loci, impute="spatial")  # doctest: +SKIP
    >>> model = spaclone.tl.transition_model(pot, time_key="csr_pot", graph="spatial", states="region")  # doctest: +SKIP
    """
    names = _obs_names(data)
    n = len(names)
    if n < 2:
        raise ValueError("transition_model needs at least 2 units")
    if not 0 <= weight <= 1:
        raise ValueError(f"weight must lie in [0, 1], got {weight}")
    if backend not in ("native", "cellrank"):
        raise ValueError(f"backend must be 'native' or 'cellrank', got {backend!r}")
    if component not in ("largest", "all"):
        raise ValueError(f"component must be 'largest' or 'all', got {component!r}")
    time = pd.to_numeric(pd.Series(_column(data, time_key, "time_key")), errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError(f"time_key: {int((~np.isfinite(time)).sum())} units have a missing or non-finite time")
    st = None if states is None else _as_states(_column(data, states, "states"))
    xy = _coords(data, spatial_key=spatial_key, x_key=x_key, y_key=y_key, required=False)
    if isinstance(graph, str):
        conn = neighbor_graph(
            data,
            gex,
            graph=graph,
            k=k,
            bandwidth_um=bandwidth_um,
            max_distance_um=max_distance_um,
            joint_alpha=joint_alpha,
            connectivities_key=connectivities_key,
            use_rep=use_rep,
            n_pcs=n_pcs,
            spatial_key=spatial_key,
            x_key=x_key,
            y_key=y_key,
            rng=rng,
        )
        graph_name = graph
    else:
        conn = _as_csr(graph, n, "graph")
        graph_name = "precomputed"
    if (abs(conn - conn.T) > 1e-12 * max(1.0, abs(conn).max() if conn.nnz else 1.0)).nnz:
        logger.info("transition_model: the graph is not symmetric (as CellRank, it is used as given)")
    keep = np.ones(n, dtype=bool)
    n_comp, lab = csgraph.connected_components(conn, directed=True, connection="weak")
    if n_comp > 1 and component == "largest":
        big = np.bincount(lab).argmax()
        keep = lab == big
        warnings.warn(
            f"transition_model: the graph has {n_comp} connected components; keeping the largest "
            f"({int(keep.sum())} of {n} units; the others are in model.dropped)",
            SpacloneWarning,
            stacklevel=2,
        )
        idx = np.flatnonzero(keep)
        conn = conn[idx][:, idx].tocsr()
        time = time[idx]
        xy = None if xy is None else xy[idx]
        st = None if st is None else st[idx].remove_unused_categories()
    params = {
        "time_key": time_key if isinstance(time_key, str) else "<values>",
        "graph": graph_name,
        "weight": float(weight),
        "k": int(k),
        "threshold_scheme": threshold_scheme,
        "frac_to_keep": float(frac_to_keep),
        "ties": ties,
        "b": float(b),
        "nu": float(nu),
        "density_normalize": bool(density_normalize),
        "bandwidth_um": bandwidth_um,
        "max_distance_um": max_distance_um,
        "joint_alpha": float(joint_alpha),
        "component": component,
        "n_components": int(n_comp),
        "backend": backend,
        "states": states if isinstance(states, str) or states is None else "<values>",
    }
    if backend == "cellrank":
        pt, ck = _cellrank_kernels(conn, time, params)
    else:
        pt, ck = _native_kernels(conn, time, params)
    t = _combine(pt, ck, weight)
    return TransitionModel(
        transition_matrix=t,
        pseudotime_matrix=pt,
        connectivity_matrix=ck,
        connectivities=conn,
        time=time,
        obs_names=names[keep],
        coords=xy,
        states=st,
        params=params,
        dropped=names[~keep],
    )


# ===================================================================================================== TPT core
def _committors(
    t: sparse.csr_matrix, pi: np.ndarray, in_a: np.ndarray, in_b: np.ndarray, order: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Forward (reach B before A) and backward (came from A rather than B) committors of the unit-level chain.

    One factorisation of ``I - T_II`` (intermediate units) gives both: ``(I - T_II) q+ = T_IB 1`` and, for
    ``y = pi_I q-_I``, the transposed system ``(I - T_II)' y = T_AI' pi_A`` (the time-reversed chain
    ``pi_j T_ji / pi_i``, the adjoint system of deeptime's backward committor).
    """
    n = t.shape[0]
    ii = np.flatnonzero(~(in_a | in_b))
    q_plus = np.where(in_b, 1.0, 0.0)
    q_minus = np.where(in_a, 1.0, 0.0)
    if len(ii):
        t_i = t[ii]
        lu = _LU(sparse.identity(len(ii), format="csc") - t_i[:, ii], _sub_order(order, ii, n))
        q_plus[ii] = lu.solve(np.asarray(t_i[:, np.flatnonzero(in_b)].sum(axis=1)).ravel())
        a_idx = np.flatnonzero(in_a)
        rhs = np.asarray(t[a_idx][:, ii].T @ pi[a_idx]).ravel()
        y = lu.solve(rhs, trans="T")
        with np.errstate(divide="ignore", invalid="ignore"):
            q_minus[ii] = np.where(pi[ii] > 0, y / pi[ii], 0.0)
    return np.clip(q_plus, 0.0, 1.0), np.clip(q_minus, 0.0, 1.0)


def _gross_flux(t: sparse.csr_matrix, pi: np.ndarray, q_minus: np.ndarray, q_plus: np.ndarray) -> sparse.csr_matrix:
    f = (sparse.diags(pi * q_minus) @ t @ sparse.diags(q_plus)).tocsr()
    f.setdiag(0.0)
    f.eliminate_zeros()
    return f


def _coarse_flux(f: sparse.csr_matrix, codes: np.ndarray, m: int) -> np.ndarray:
    mem = _membership(codes, m)
    fc = (mem.T @ f @ mem).toarray()
    np.fill_diagonal(fc, 0.0)
    return fc


def _percent(f: np.ndarray) -> np.ndarray:
    tot = f.sum()
    return f * 100.0 / tot if tot > 0 else np.zeros_like(f)


def _tpt_native(
    t: sparse.csr_matrix,
    pi: np.ndarray | None,
    codes: np.ndarray,
    m: int,
    src: np.ndarray,
    tgt: np.ndarray,
    order: np.ndarray | None = None,
) -> dict[str, Any]:
    if pi is None:
        pi = stationary_distribution(t, order=order)
    in_a, in_b = np.isin(codes, src), np.isin(codes, tgt)
    qp, qm = _committors(t, pi, in_a, in_b, order)
    f = _gross_flux(t, pi, qm, qp)
    gross = _coarse_flux(f, codes, m)
    return {"pi": pi, "q_plus": qp, "q_minus": qm, "gross": gross}


def _tpt_deeptime(
    t: sparse.csr_matrix,
    pi: np.ndarray | None,
    codes: np.ndarray,
    m: int,
    src: np.ndarray,
    tgt: np.ndarray,
    order: np.ndarray | None = None,
) -> dict[str, Any]:
    try:
        import deeptime.markov.tools.analysis as msmana
        from deeptime.markov import ReactiveFlux
        from deeptime.markov.tools.flux import flux_matrix, to_netflux
    except ImportError as err:
        raise ImportError("backend='deeptime' needs deeptime (pip install deeptime)") from err
    a = np.flatnonzero(np.isin(codes, src))
    b = np.flatnonzero(np.isin(codes, tgt))
    if pi is None:
        pi = np.asarray(msmana.stationary_distribution(t), dtype=float)
    qp = np.asarray(msmana.committor(t, a, b, forward=True), dtype=float)
    qm = np.asarray(msmana.committor(t, a, b, forward=False, mu=pi), dtype=float)
    gross_u = flux_matrix(t, pi, qm, qp, netflux=False)
    rf = ReactiveFlux(
        a, b, net_flux=to_netflux(gross_u), stationary_distribution=pi, qminus=qm, qplus=qp, gross_flux=gross_u
    )
    sets = [np.flatnonzero(codes == j) for j in range(m)]
    tpt_sets, coarse = rf.coarse_grain(sets)
    g = coarse.gross_flux
    g = np.asarray(g.toarray() if sparse.issparse(g) else g, dtype=float)
    # deeptime orders the coarse sets source, intermediates, target; map back to the state order
    pos = [int(codes[next(iter(s))]) for s in tpt_sets]
    gross = np.zeros((m, m))
    gross[np.ix_(pos, pos)] = g
    return {"pi": pi, "q_plus": qp, "q_minus": qm, "gross": gross}


def _widest_path(g: np.ndarray, s: int, t: int) -> tuple[list[int] | None, float]:
    """Path from s to t maximising its smallest edge capacity (modified Dijkstra on a dense matrix)."""
    n = g.shape[0]
    best = np.full(n, -np.inf)
    best[s] = np.inf
    prev = np.full(n, -1)
    done = np.zeros(n, dtype=bool)
    while True:
        cand = np.where(~done, best, -np.inf)
        u = int(np.argmax(cand))
        if not cand[u] > 0:
            break
        done[u] = True
        if u == t:
            break
        cap = np.minimum(best[u], g[u])
        upd = (~done) & (g[u] > 0) & (cap > best)
        best[upd] = cap[upd]
        prev[upd] = u
    if not done[t] or prev[t] < 0 and s != t:
        return None, 0.0
    path = [t]
    while path[-1] != s:
        path.append(int(prev[path[-1]]))
    return path[::-1], float(best[t])


def _pathways(net: np.ndarray, src: np.ndarray, tgt: np.ndarray, *, fraction: float = 1.0, maxiter: int = 1000):
    """Dominant reaction pathways of a net-flux network (bottleneck decomposition, Metzner et al. 2009)."""
    m = net.shape[0]
    g = np.zeros((m + 2, m + 2))
    g[:m, :m] = net
    s_node, t_node = m, m + 1
    g[s_node, src] = net[src, :].sum(axis=1)
    g[tgt, t_node] = net[:, tgt].sum(axis=0)
    total = g[s_node].sum()
    paths, caps = [], []
    if not total > 0:
        return paths, caps, 0.0
    tol = 1e-12 * total
    captured = 0.0
    for _ in range(maxiter):
        path, cap = _widest_path(g, s_node, t_node)
        if path is None or not cap > tol:
            break
        for u, v in zip(path[:-1], path[1:], strict=False):
            g[u, v] -= cap
        g[g < tol] = 0.0
        paths.append(path[1:-1])
        caps.append(cap)
        captured += cap
        if captured / total >= fraction - 1e-12:
            break
    return paths, caps, float(total)


def _state_codes(cat: pd.Categorical, labels: Any, what: str) -> np.ndarray:
    labels = [labels] if isinstance(labels, str) or not isinstance(labels, Sequence) else list(labels)
    cats = list(cat.categories.astype(str))
    missing = [str(x) for x in labels if str(x) not in cats]
    if missing:
        raise ValueError(f"{what}: state(s) {missing} not among the states {cats}")
    return np.array([cats.index(str(x)) for x in labels], dtype=np.int64)


def _permute_within(values: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    if strata is None:
        return values[rng.permutation(len(values))]
    order = np.argsort(strata, kind="stable")
    bounds = np.flatnonzero(np.diff(strata[order])) + 1
    for grp in np.split(order, bounds):
        if len(grp) > 1:
            out[grp] = values[grp[rng.permutation(len(grp))]]
    return out


@dataclass(frozen=True, repr=False)
class FluxResult(TestResult):
    """Result of :func:`tpt_flux`: a :class:`~spaclone.TestResult` plus state, unit and pathway tables.

    Attributes
    ----------
    states
        One row per state: ``role`` (source / intermediate / target), ``n_units``, ``unit_share``, ``stationary``
        (lumped stationary probability) with a bootstrap 95% interval (``stationary_lo``, ``stationary_hi``), and the
        stationary-weighted mean forward and backward committors (``q_plus``, ``q_minus``).
    units
        One row per unit: ``state``, ``stationary``, ``q_plus``, ``q_minus`` and the coordinates when known.
    pathways
        Dominant pathways from source to target: ``path`` (states joined by " > "), ``flux`` and ``percent`` of the
        total reactive flux.
    gross, net
        ``m x m`` gross and net flux in percent of their totals (rows: from, columns: to).
    null_draws
        One row per null draw: ``rate``, ``mfpt`` and ``total_gross_flux`` (the sum of the coarse gross flux).
    """

    states: pd.DataFrame | None = None
    units: pd.DataFrame | None = None
    pathways: pd.DataFrame | None = None
    gross: pd.DataFrame | None = None
    net: pd.DataFrame | None = None
    null_draws: pd.DataFrame | None = None

    def __repr__(self) -> str:
        return TestResult.__repr__(self)


def tpt_flux(
    model: TransitionModel,
    *,
    source: str | Sequence[str],
    target: str | Sequence[str],
    states: Sequence[Any] | None = None,
    n_random: int = 100,
    null: NullKind = "time_within_states",
    statistic: Literal["net", "gross", "net_percent", "gross_percent"] = "net",
    alternative: Alternative = "greater",
    correction: Literal["fdr_bh"] | None = "fdr_bh",
    permutations: np.ndarray | None = None,
    n_bootstrap: int = 100,
    fraction: float = 1.0,
    backend: Literal["native", "deeptime"] = "native",
    rng: RngLike = None,
) -> FluxResult:
    """Transition path theory from a source to a target set of states, with a randomisation test of every flux.

    TPT runs on the unit-level chain with ``A`` = the units of the source state(s) and ``B`` = those of the target
    state(s); the gross flux ``f_ij = pi_i q-_i T_ij q+_j`` is summed over the units of each pair of states (the
    diagonal is dropped) and the net flux is ``max(F_IJ - F_JI, 0)``; both are reported in probability per step of
    the stationary chain (``*_raw``) and in percent of their totals, as in sciCSR's ``fitTPT``. The rate
    ``k_AB = total flux / sum_i pi_i q-_i`` (reactive trajectories per step) is tested as well.

    Parameters
    ----------
    model
        A :class:`TransitionModel`.
    source, target
        State label(s) of ``A`` and ``B`` (disjoint). sciCSR's advice applies: choose states far apart (e.g. IgM and
        IgA, or follicle and red pulp) so that every intermediate transition is in the network.
    states
        State label per unit (default: the model's ``states``).
    n_random
        Null draws (0 = no test).
    null
        Randomisation design (see Null model).
    statistic
        Flux tested per pair: ``"net"`` (directed flux, probability per step; default), ``"gross"``, or their shares
        of the total, ``"net_percent"`` / ``"gross_percent"`` (sciCSR plots the gross percent). Shares only ask how the
        reactive flux is routed: when every path must cross the same states (a chain of regions) they are fixed by
        the topology and cannot respond to the time key; the raw fluxes and the rate can.
    alternative
        Direction of the Monte-Carlo p-value (``"greater"``: more flux than under the null).
    correction
        ``"fdr_bh"`` (Benjamini–Hochberg over the ``m (m - 1)`` ordered pairs) or None.
    permutations
        ``null="labels"`` only: an ``(n_random, n_units)`` array of unit orders replacing the random draws (draw ``b``
        gives the first ``n_1`` units of row ``b`` to the first state, the next ``n_2`` to the second, ...), e.g. to
        replay sciCSR's ``random.seed(i + 1); random.shuffle`` draws exactly.
    n_bootstrap
        Bootstrap draws of the stationary distribution (units resampled within each state, as sciCSR).
    fraction
        Share of the total flux the pathway decomposition must explain.
    backend
        ``"native"`` or ``"deeptime"`` (an installed deeptime computes the stationary distribution, committors,
        fluxes and coarse-graining).
    rng
        Seed or generator of the null and the bootstrap.

    Returns
    -------
    FluxResult
        ``table``: one row per ordered pair of states with ``from_state``, ``to_state``, ``gross_flux`` and
        ``net_flux`` (percent), ``gross_flux_raw`` and ``net_flux_raw``, ``statistic``, ``expected`` / ``null_mean``,
        ``null_sd``, ``z``, ``p_value``, ``q_value``, ``n_perm`` and ``p_pooled_normal`` (sciCSR's significance: the
        upper normal tail of the observed gross percent against the mean and sd pooled over all entries of all null
        gross-percent matrices; not a calibrated p-value, reported for comparison only). ``summary``: ``total_flux``,
        ``rate``, ``mfpt`` (in steps of the chain) and, with a null, ``rate_null_mean``, ``rate_z``,
        ``rate_p_value``; ``n_units``, ``n_states``, ``n_significant`` (q < 0.05). ``null``: the null draws of the
        tested statistic, shape ``(n_random, m, m)``.

    Null model
    ----------
    * ``"time_within_states"`` (default): the time key is permuted among the units of each state; the graph, the
      state labels, the source and target sets and the kernel settings stay fixed, and the chain and TPT are
      recomputed for every draw. It keeps each state's distribution of the time key, so a flux is significant when
      the arrangement of the time key on the graph (e.g. earlier units on the follicle side of a follicle–GC border)
      carries more flux between the two states than the states' time distributions alone. It is exact under
      exchangeability of the time key within states, but degenerate when the time key is constant within states
      (e.g. a CSR potential from the isotype alone with the isotypes as states): then the null equals the data and
      a :class:`~spaclone._utils.LowPowerWarning` is raised.
    * ``"time"``: the time key is permuted over all units (graph and states fixed): is there more flux than when the
      time key is unrelated to the graph and to the states?
    * ``"labels"`` (sciCSR's ``fitTPT`` randomisation): the transition matrix stays fixed and the units are dealt at
      random into states of the observed sizes. Random states are scattered over the graph while real states are
      contiguous (in tissue or in expression), so this null is not exchangeable with spatially or transcriptionally
      coherent states and its p-values are not calibrated there; it is provided to reproduce sciCSR.

    References
    ----------
    Metzner, Schütte & Vanden-Eijnden 2009, doi:10.1137/070699500; Noé et al. 2009, doi:10.1073/pnas.0905466106;
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.

    Examples
    --------
    >>> res = spaclone.tl.tpt_flux(model, source="follicle", target="red pulp", n_random=200, rng=0)  # doctest: +SKIP
    >>> res.table.sort_values("q_value").head()  # doctest: +SKIP
    """
    if not isinstance(model, TransitionModel):
        raise TypeError("tpt_flux expects a TransitionModel (from spaclone.tl.transition_model)")
    check_alternative(alternative)
    if null not in ("time_within_states", "time", "labels"):
        raise ValueError(f"null must be 'time_within_states', 'time' or 'labels', got {null!r}")
    if statistic not in ("net", "gross", "net_percent", "gross_percent"):
        raise ValueError(f"statistic must be 'net', 'gross', 'net_percent' or 'gross_percent', got {statistic!r}")
    if correction not in ("fdr_bh", None):
        raise ValueError(f"correction must be 'fdr_bh' or None, got {correction!r}")
    if backend not in ("native", "deeptime"):
        raise ValueError(f"backend must be 'native' or 'deeptime', got {backend!r}")
    if states is None:
        if model.states is None:
            raise ValueError("tpt_flux: pass states= (the model has none)")
        cat = model.states
    else:
        if isinstance(states, str):
            raise TypeError("tpt_flux: pass the state labels (one per unit of the model), not a column name")
        cat = _as_states(states)
        if len(cat) != model.n_units:
            raise ValueError(f"tpt_flux: {len(cat)} state labels for {model.n_units} units")
    codes = np.asarray(cat.codes, dtype=np.int64)
    names = [str(c) for c in cat.categories]
    m = len(names)
    src = _state_codes(cat, source, "source")
    tgt = _state_codes(cat, target, "target")
    if np.intersect1d(src, tgt).size:
        raise ValueError("source and target states must be disjoint")
    rng = as_rng(rng)
    core = _tpt_deeptime if backend == "deeptime" else _tpt_native
    t = model.transition_matrix
    order = model.order if backend == "native" else None
    obs = core(t, model.pi if backend == "native" else None, codes, m, src, tgt, order)
    pi, qp, qm, gross_raw = obs["pi"], obs["q_plus"], obs["q_minus"], obs["gross"]
    net_raw = np.maximum(gross_raw - gross_raw.T, 0.0)
    gross_pct, net_pct = _percent(gross_raw), _percent(net_raw)
    not_src = np.setdiff1d(np.arange(m), src)

    def _rate(net: np.ndarray, pi_: np.ndarray, qm_: np.ndarray) -> tuple[float, float]:
        tot = float(net[np.ix_(src, not_src)].sum())
        den = float(np.sum(pi_ * qm_))
        return tot, (tot / den if den > 0 else float("nan"))

    total_flux, rate = _rate(net_raw, pi, qm)
    mfpt = 1.0 / rate if rate and np.isfinite(rate) and rate > 0 else float("inf")

    # ---------------------------------------------------------------- null draws
    n_random = int(n_random)
    null_raw = null_rate = None
    if n_random > 0:
        if null == "time_within_states":
            var = pd.Series(model.time).groupby(codes).var(ddof=0).fillna(0.0)
            if not (var > 0).any():
                warnings.warn(
                    "tpt_flux: the time key is constant within every state, so the within-state shuffle null equals "
                    "the data (all p = 1); use null='time', or states that are not a function of the time key",
                    LowPowerWarning,
                    stacklevel=2,
                )
        if permutations is not None:
            if null != "labels":
                raise ValueError("permutations= is only used with null='labels'")
            permutations = np.asarray(permutations, dtype=np.int64)
            if permutations.shape != (n_random, model.n_units):
                raise ValueError(
                    f"permutations must have shape ({n_random}, {model.n_units}), got {permutations.shape}"
                )
        null_raw = np.empty((n_random, m, m))
        null_rate = np.empty(n_random)
        p = model.params
        sizes = np.bincount(codes, minlength=m)
        ck = model.connectivity_matrix
        for i in range(n_random):
            if null == "labels":
                perm = permutations[i] if permutations is not None else rng.permutation(model.n_units)
                codes_b = np.empty(model.n_units, dtype=np.int64)
                codes_b[perm] = np.repeat(np.arange(m), sizes)
                res = core(t, pi, codes_b, m, src, tgt, order)
            else:
                strata = codes if null == "time_within_states" else None
                tb = _permute_within(model.time, strata, rng)
                pt_b = _row_normalise(
                    _biased_connectivities(
                        model.connectivities,
                        tb,
                        scheme=p["threshold_scheme"],
                        frac_to_keep=p["frac_to_keep"],
                        b=p["b"],
                        nu=p["nu"],
                        ties=p.get("ties", "keep"),
                    )
                )
                res = core(_combine(pt_b, ck, p["weight"]), None, codes, m, src, tgt, order)
            g = res["gross"]
            null_raw[i] = g
            null_rate[i] = _rate(np.maximum(g - g.T, 0.0), res["pi"], res["q_minus"])[1]

    # ---------------------------------------------------------------- pair table
    off = ~np.eye(m, dtype=bool)
    fr, to = np.nonzero(off)

    def _stat(g: np.ndarray) -> np.ndarray:
        """The tested statistic of gross-flux matrices of shape (..., m, m)."""
        if statistic.startswith("net"):
            g = np.maximum(g - np.swapaxes(g, -1, -2), 0.0)
        if statistic.endswith("percent"):
            tot = g.sum(axis=(-1, -2), keepdims=True)
            g = np.divide(g * 100.0, tot, out=np.zeros_like(g), where=tot > 0)
        return g

    obs_stat = _stat(gross_raw)[fr, to]
    tab = pd.DataFrame(
        {
            "from_state": [names[i] for i in fr],
            "to_state": [names[j] for j in to],
            "gross_flux": gross_pct[fr, to],
            "net_flux": net_pct[fr, to],
            "gross_flux_raw": gross_raw[fr, to],
            "net_flux_raw": net_raw[fr, to],
            "statistic": obs_stat,
        }
    )
    rate_test: dict[str, float] = {}
    if n_random > 0:
        null_stat = _stat(null_raw)
        nd = null_stat[:, fr, to]
        mu, sd, _, _ = null_summary(nd)
        tab["expected"] = mu
        tab["null_mean"] = mu
        tab["null_sd"] = sd
        with np.errstate(invalid="ignore", divide="ignore"):
            tab["z"] = np.where(sd > 0, (obs_stat - mu) / sd, np.nan)
        tab["p_value"] = mc_pvalue(nd, obs_stat, alternative=alternative)
        tab["q_value"] = bh(tab["p_value"].to_numpy()) if correction == "fdr_bh" else np.nan
        tab["n_perm"] = n_random
        null_gpct = np.array([_percent(g) for g in null_raw])
        pooled_mu, pooled_sd = float(null_gpct.mean()), float(null_gpct.std())
        tab["p_pooled_normal"] = norm.sf(gross_pct[fr, to], loc=pooled_mu, scale=pooled_sd) if pooled_sd > 0 else np.nan
        null_out = null_stat
        r_mu, r_sd = float(np.nanmean(null_rate)), float(np.nanstd(null_rate, ddof=1)) if n_random > 1 else np.nan
        rate_test = {
            "rate_null_mean": r_mu,
            "rate_null_sd": r_sd,
            "rate_z": (rate - r_mu) / r_sd if r_sd and np.isfinite(r_sd) and r_sd > 0 else float("nan"),
            "rate_p_value": float(mc_pvalue(null_rate, rate, alternative=alternative)[0]),
        }
    else:
        for c in ("expected", "null_mean", "null_sd", "z", "p_value", "q_value", "p_pooled_normal"):
            tab[c] = np.nan
        tab["n_perm"] = 0
        null_out = None

    # ---------------------------------------------------------------- states, units, pathways
    role = np.array(["intermediate"] * m, dtype=object)
    role[src] = "source"
    role[tgt] = "target"
    n_units = np.bincount(codes, minlength=m)
    pi_state = np.bincount(codes, weights=pi, minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        qp_state = np.bincount(codes, weights=pi * qp, minlength=m) / pi_state
        qm_state = np.bincount(codes, weights=pi * qm, minlength=m) / pi_state
    boot = np.empty((max(int(n_bootstrap), 0), m))
    members = [np.flatnonzero(codes == j) for j in range(m)]
    for bi in range(boot.shape[0]):
        for j, mem in enumerate(members):
            boot[bi, j] = pi[mem[rng.integers(0, len(mem), len(mem))]].sum() if len(mem) else 0.0
    lo = np.quantile(boot, 0.025, axis=0) if boot.shape[0] else np.full(m, np.nan)
    hi = np.quantile(boot, 0.975, axis=0) if boot.shape[0] else np.full(m, np.nan)
    states_tab = pd.DataFrame(
        {
            "role": role,
            "n_units": n_units,
            "unit_share": n_units / n_units.sum(),
            "stationary": pi_state,
            "stationary_lo": lo,
            "stationary_hi": hi,
            "q_plus": qp_state,
            "q_minus": qm_state,
        },
        index=pd.Index(names, name="state"),
    )
    units = pd.DataFrame(
        {"state": pd.Categorical.from_codes(codes, categories=names), "stationary": pi, "q_plus": qp, "q_minus": qm},
        index=model.obs_names,
    )
    if model.coords is not None:
        units.insert(0, S.Y, model.coords[:, 1])
        units.insert(0, S.X, model.coords[:, 0])
    paths, caps, tot = _pathways(net_raw, src, tgt, fraction=fraction)
    pathways = pd.DataFrame(
        {
            "path": [" > ".join(names[j] for j in pth) for pth in paths],
            "flux": caps,
            "percent": [100.0 * c / tot if tot > 0 else np.nan for c in caps],
        }
    )
    idx = pd.Index(names, name="from_state")
    cols = pd.Index(names, name="to_state")
    summary = {
        "total_flux": total_flux,
        "rate": rate,
        "mfpt": mfpt,
        "n_units": int(model.n_units),
        "n_states": m,
        "source": ",".join(names[j] for j in src),
        "target": ",".join(names[j] for j in tgt),
        "null": null,
        "n_random": n_random,
        "n_significant": int((tab["q_value"] < 0.05).sum()) if n_random > 0 and correction else 0,
        **rate_test,
    }
    if n_random > 0 and n_random < 100:
        logger.info("tpt_flux: %d null draws give p-values no smaller than %.3g", n_random, 1 / (n_random + 1))
    params = {
        "source": list(np.asarray(names)[src]),
        "target": list(np.asarray(names)[tgt]),
        "null": null,
        "statistic": statistic,
        "n_random": n_random,
        "alternative": alternative,
        "correction": correction,
        "n_bootstrap": int(n_bootstrap),
        "backend": backend,
        "replayed_permutations": permutations is not None,
        "model": dict(model.params),
        "rng_entropy": rng_entropy(rng),
        "null_design": {
            "time_within_states": "time key permuted among the units of each state; graph, states, A/B fixed",
            "time": "time key permuted over all units; graph, states, A/B fixed",
            "labels": "units dealt at random into states of the observed sizes; transition matrix fixed",
        }[null],
    }
    return FluxResult(
        name="tpt_flux",
        table=tab,
        summary=summary,
        params=params,
        null=null_out,
        states=states_tab,
        units=units,
        pathways=pathways,
        gross=pd.DataFrame(gross_pct, index=idx, columns=cols),
        net=pd.DataFrame(net_pct, index=idx, columns=cols),
        null_draws=None
        if null_raw is None
        else pd.DataFrame(
            {
                "rate": null_rate,
                "mfpt": np.divide(1.0, null_rate, out=np.full(n_random, np.inf), where=null_rate > 0),
                "total_gross_flux": null_raw.sum(axis=(1, 2)),
            }
        ),
    )


# ===================================================================================================== projection
def transition_displacement(
    model: TransitionModel,
    *,
    method: Literal["expected", "unit"] = "expected",
    baseline: Literal["connectivity"] | None = "connectivity",
    coords: np.ndarray | None = None,
) -> pd.DataFrame:
    """Projection of the transition matrix onto tissue coordinates: where each unit's next step is expected to go.

    Parameters
    ----------
    model
        A :class:`TransitionModel` with coordinates (or pass ``coords``).
    method
        ``"expected"``: the expected displacement per step, ``sum_j T_ij (x_j - x_i)`` in µm. ``"unit"``: the
        CellRank / scVelo embedding projection, ``sum_j T_ij u_ij - mean_j u_ij`` over the unit vectors ``u_ij`` to
        the graph neighbours (dimensionless, direction only).
    baseline
        ``"expected"`` only: ``"connectivity"`` subtracts the displacement of the undirected connectivity kernel on
        the same graph, ``sum_j T_conn,ij (x_j - x_i)``, so the result is the drift caused by the time key
        (``weight * (d_time - d_conn)``) and units at a tissue edge, whose neighbours all lie inwards, show no
        spurious inward drift. None keeps the raw expected displacement.
    coords
        ``(n, 2)`` coordinates in µm overriding the model's (e.g. a UMAP for expression arrows).

    Returns
    -------
    pandas.DataFrame
        One row per unit: ``x_um``, ``y_um``, ``dx_um``, ``dy_um``, ``speed_um`` and ``time``.

    References
    ----------
    Bergen et al. 2020, doi:10.1038/s41587-020-0591-3; Lange et al. 2022, doi:10.1038/s41592-021-01346-6.
    """
    xy = model.coords if coords is None else np.asarray(coords, dtype=float)[:, :2]
    if xy is None:
        raise ValueError("transition_displacement: the model has no coordinates; pass coords=")
    if len(xy) != model.n_units:
        raise ValueError(f"coords: {len(xy)} rows for {model.n_units} units")
    t = model.transition_matrix
    if method == "expected":
        d = t @ xy - xy
        if baseline == "connectivity":
            d = d - (model.connectivity_matrix @ xy - xy)
        elif baseline is not None:
            raise ValueError(f"baseline must be 'connectivity' or None, got {baseline!r}")
    elif method == "unit":
        c = model.connectivities.tocoo()
        r, cl = c.row, c.col
        dx = xy[cl] - xy[r]
        nrm = np.linalg.norm(dx, axis=1)
        u = np.divide(dx, nrm[:, None], out=np.zeros_like(dx), where=nrm[:, None] > 0)
        probs = np.asarray(t[r, cl]).ravel()
        deg = np.bincount(r, minlength=model.n_units).astype(float)
        d = np.column_stack([np.bincount(r, weights=probs * u[:, j], minlength=model.n_units) for j in range(2)])
        mean_u = np.column_stack([np.bincount(r, weights=u[:, j], minlength=model.n_units) for j in range(2)])
        with np.errstate(invalid="ignore", divide="ignore"):
            d = d - np.where(deg[:, None] > 0, mean_u / deg[:, None], 0.0)
    else:
        raise ValueError(f"method must be 'expected' or 'unit', got {method!r}")
    return pd.DataFrame(
        {
            S.X: xy[:, 0],
            S.Y: xy[:, 1],
            "dx_um": d[:, 0],
            "dy_um": d[:, 1],
            "speed_um": np.hypot(d[:, 0], d[:, 1]),
            "time": model.time,
        },
        index=model.obs_names,
    )


def flow_grid(
    displacement: pd.DataFrame,
    *,
    grid_um: float | None = None,
    n_grid: int = 40,
    smooth: float = 0.5,
    min_mass: float = 1.0,
    x_key: str = S.X,
    y_key: str = S.Y,
    dx_key: str = "dx_um",
    dy_key: str = "dy_um",
) -> pd.DataFrame:
    """Gaussian-weighted mean displacement on a regular grid, for arrow and stream plots of the flow on the tissue.

    Parameters
    ----------
    displacement
        Per-unit vectors (:func:`transition_displacement`).
    grid_um
        Grid spacing in µm (default: the larger side of the bounding box / ``n_grid``).
    n_grid
        Grid points along the larger side when ``grid_um`` is None.
    smooth
        Gaussian standard deviation in grid spacings (scVelo's default 0.5).
    min_mass
        Grid points whose summed weight is below this (about the number of units within one standard deviation)
        are left out.
    x_key, y_key, dx_key, dy_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per kept grid point: ``x_um``, ``y_um``, ``dx_um``, ``dy_um`` (weighted means), ``mass`` and the
        integer grid indices ``ix``, ``iy``; ``attrs["spaclone"]["grid_um"]`` holds the spacing.

    References
    ----------
    Bergen et al. 2020, doi:10.1038/s41587-020-0591-3 (velocity on a grid).
    """
    xy = displacement[[x_key, y_key]].to_numpy(dtype=float)
    v = displacement[[dx_key, dy_key]].to_numpy(dtype=float)
    ok = np.isfinite(xy).all(1) & np.isfinite(v).all(1)
    xy, v = xy[ok], v[ok]
    if not len(xy):
        return pd.DataFrame(columns=[S.X, S.Y, "dx_um", "dy_um", "mass", "ix", "iy"])
    lo, hi = xy.min(0), xy.max(0)
    h = float(grid_um) if grid_um is not None else max(float((hi - lo).max()) / max(int(n_grid), 1), 1e-9)
    gx = np.arange(lo[0], hi[0] + h * 0.5, h)
    gy = np.arange(lo[1], hi[1] + h * 0.5, h)
    ix, iy = np.meshgrid(np.arange(len(gx)), np.arange(len(gy)), indexing="xy")
    pts = np.column_stack([gx[ix.ravel()], gy[iy.ravel()]])
    sigma = smooth * h
    tree = cKDTree(xy)
    pairs = tree.query_ball_point(pts, r=3.0 * sigma)
    lens = np.fromiter((len(p) for p in pairs), dtype=np.int64, count=len(pairs))
    gi = np.repeat(np.arange(len(pts)), lens)
    ui = np.concatenate([np.asarray(p, dtype=np.int64) for p in pairs]) if lens.sum() else np.zeros(0, np.int64)
    d2 = ((pts[gi] - xy[ui]) ** 2).sum(1)
    w = np.exp(-d2 / (2 * sigma * sigma))
    mass = np.bincount(gi, weights=w, minlength=len(pts))
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.bincount(gi, weights=w * v[ui, 0], minlength=len(pts)) / mass
        my = np.bincount(gi, weights=w * v[ui, 1], minlength=len(pts)) / mass
    keep = mass >= min_mass
    out = pd.DataFrame(
        {
            S.X: pts[keep, 0],
            S.Y: pts[keep, 1],
            "dx_um": mx[keep],
            "dy_um": my[keep],
            "mass": mass[keep],
            "ix": ix.ravel()[keep],
            "iy": iy.ravel()[keep],
        }
    )
    out.attrs["spaclone"] = {
        "flow_grid": {
            "grid_um": h,
            "smooth": smooth,
            "min_mass": min_mass,
            "nx": len(gx),
            "ny": len(gy),
            "x0": float(gx[0]),
            "y0": float(gy[0]),
        }
    }
    return out
