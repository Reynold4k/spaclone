"""Per-clone spatial scan: is a clone concentrated in some window beyond random relabelling of its chain's cells?

For one chain (and section) the pool is the chain's units: by default one unit per (grid cell, clone) pair
(``units="bins"``: a clone counts once in every ``grid_um`` cell it occupies), or one per (spot, clone) pair
(``units="records"``). For a clone with ``N_c`` of the ``N`` units, a window ``w`` holding ``T_w`` units of the chain is
expected to hold ``mu_w = N_c T_w / N`` units of the clone if the clone were a random draw from the chain's cells; no
model of where receptors are captured is needed. The statistic is Kulldorff's log-likelihood ratio maximised over
windows centred on every cell of a ``grid_um`` grid (circles or squares of several sizes), and its null distribution
comes from random relabelling.

Two points go beyond a textbook scan (survey S3 §A):

* **Monte-Carlo p-values that can pass BH.** A Besag–Clifford sequential p-value never falls below its floor, and with
  ``m`` clones Benjamini–Hochberg's first threshold ``alpha / m`` lies below the floor of a fixed 9,999-draw test once
  ``m`` exceeds about 500. With ``stopping="bh"`` (default) the scan keeps drawing only for clones whose BH decision is
  still open (MMCTest logic) and raises their cap far enough; the reported p-values stay valid whatever cap was used.
* **Cells, not spots, are exchangeable.** One cell spills its molecules over neighbouring spots, so a clone's units come
  in cell-sized clumps. Relabelling units one by one (``null="units"``, the archive's ClonalScan) breaks the clumps
  apart, and every clone looks concentrated: 15-100 % of null clones at alpha = 0.05 once cells cover 1.1-6 spots (F4a
  design study). The default null (``null="blocks"``) relabels whole cell-scale *blocks* -- the units of one clone
  within ``block_radius_um`` of a seed unit -- taken in random order until the relabelled clone has as many units as
  the observed one, and recomputes the maximum on the drawn units. The clone is still counted in units, not merged into
  loci: single-linkage loci chain a dense germinal-centre clone into a few loci (power for a clone with half of its
  150 cells packed together: <= 0.21 counted in loci, >= 0.98 counted in units), and relabelling ready-made loci is
  anti-conservative for large clones where units are dense, because their loci merge more there (19-30 % of null large
  clones in clumped pools of cells, 97-100 % when half or all of the molecules are misplaced).

References
----------
Kulldorff & Nagarwalla 1995, doi:10.1002/sim.4780140809; Kulldorff 1997, doi:10.1080/03610929708831995; Besag &
Clifford 1991, doi:10.1093/biomet/78.2.301; Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x;
Gandy & Hahn 2014, doi:10.1111/sjos.12085 (MMCTest); Tarone 1990, doi:10.2307/2531456; Gilbert 2005,
doi:10.1111/j.1467-9876.2005.00475.x; Hurlbert 1984, doi:10.2307/1942661 and Zimmerman et al. 2021,
doi:10.1038/s41467-021-21038-1 (the exchangeable unit: pseudoreplication of dependent observations).
"""

from __future__ import annotations

import hashlib
import logging
import math
import multiprocessing
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import xlogy

from spaclone import _schema as S
from spaclone._utils import LowPowerWarning, RngLike, SpacloneWarning, TestResult, as_rng, bh, rng_entropy

__all__ = ["clonal_scan", "scan_llr"]

logger = logging.getLogger("spaclone")

_COLUMNS = [
    "clone_id", "n_records", "n_units", "n_blocks", "statistic", "null_mean", "null_sd", "z", "p_value", "q_value", "n_perm",
    "n_exceed", "n_drawn", "n_ties", "p_plain", "p_min", "testable", "significant", "stop", "best_radius_um",
    "best_x_um", "best_y_um", "best_observed", "best_total", "best_expected", "best_ratio",
]  # fmt: skip
_TIE = 1e-12  # relative tolerance: null maxima within this of the observed count as exceedances (ties are conservative)
_EPS_MMC = 1e-3  # error budget of the confidence bounds that settle BH decisions early (MMCTest)
_KEYS_PER_BATCH = 4_000_000  # scatter keys per vectorised batch of null draws


# ------------------------------------------------------------------------------------------------------- statistic


def scan_llr(
    observed: Any,
    total: Any,
    n_clone: float,
    n_units: float,
    *,
    model: Literal["poisson", "bernoulli"] = "poisson",
) -> np.ndarray:
    """Kulldorff log-likelihood ratio of scan windows (0 for windows without excess).

    Parameters
    ----------
    observed
        Units of the clone in each window (``W``).
    total
        Units of the chain in each window (``T``), the clone's own included.
    n_clone
        Units of the clone in the whole pool (``N_c``); a scalar or one value per window.
    n_units
        Units of the chain in the pool (``N``).
    model
        ``"poisson"``: conditional Poisson LLR with expectation ``mu = N_c T / N``,
        ``W ln(W / mu) + (N_c - W) ln((N_c - W) / (N_c - mu))`` where ``W > mu``. ``"bernoulli"``: Kulldorff's
        Bernoulli (case / control) LLR of the 2 x 2 table window x clone, where the window's share of clone units
        exceeds the share outside.

    Returns
    -------
    numpy.ndarray
        LLR per window (float), 0 where the window holds no excess.

    References
    ----------
    Kulldorff 1997, doi:10.1080/03610929708831995.
    """
    w = np.asarray(observed, dtype=float)
    t = np.asarray(total, dtype=float)
    w, t, ncb = np.broadcast_arrays(w, t, np.asarray(n_clone, dtype=float))
    out = np.zeros(w.shape)
    n = float(n_units)
    if model == "poisson":
        mu = ncb * t / n
        m = (w > mu) & (mu > 0)
        ww, uu, nc = w[m], mu[m], ncb[m]
        rest = nc - ww
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = ww * np.log(ww / uu)
            t2 = np.where(rest > 0, rest * np.log(rest / np.maximum(nc - uu, 1e-12)), 0.0)
        out[m] = t1 + t2
    elif model == "bernoulli":
        n_out = n - t
        c_out = ncb - w
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (t > 0) & (n_out > 0) & (w * n_out > c_out * t)
            c, ti, co, no, nc = w[m], t[m], c_out[m], n_out[m], ncb[m]
            l1 = xlogy(c, c / ti) + xlogy(ti - c, (ti - c) / ti)
            l2 = xlogy(co, co / no) + xlogy(no - co, (no - co) / no)
            l0 = xlogy(nc, nc / n) + xlogy(n - nc, (n - nc) / n)
        out[m] = l1 + l2 - l0
    else:
        raise ValueError(f"model must be 'poisson' or 'bernoulli', got {model!r}")
    return out


# --------------------------------------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class _Windows:
    """Scan windows on a square grid: offsets (dy, dx) of the cells of each window, and their row decomposition."""

    radii_um: tuple[float, ...]
    shape: str
    grid_um: float
    offsets: tuple[np.ndarray, ...]  # per size: (K_s, 2) int64 (dy, dx)
    rects: tuple[tuple[tuple[int, int, int, int], ...], ...]  # per size: (dy0, dy1, dx0, dx1), inclusive
    pad: int
    reach: float  # two cells can share a window only if their distance (p_norm, in cells) is <= reach
    p_norm: float

    @property
    def n_sizes(self) -> int:
        return len(self.radii_um)

    @property
    def n_offsets(self) -> int:
        return int(sum(len(o) for o in self.offsets))


def _make_windows(radii_um: Sequence[float], grid_um: float, shape: str) -> _Windows:
    if shape not in ("circle", "square"):
        raise ValueError(f"shape must be 'circle' or 'square', got {shape!r}")
    radii = tuple(float(r) for r in radii_um)
    if not radii or any(not np.isfinite(r) or r < 0 for r in radii):
        raise ValueError("radii_um must be a non-empty sequence of finite radii >= 0")
    if not (np.isfinite(grid_um) and grid_um > 0):
        raise ValueError("grid_um must be a positive number")
    offsets, rects = [], []
    pad = 0
    for r in radii:
        rc = r / grid_um
        big = int(np.floor(rc + 1e-9))
        d = np.arange(-big, big + 1)
        dy, dx = np.meshgrid(d, d, indexing="ij")
        if shape == "circle":
            keep = dy.astype(float) ** 2 + dx.astype(float) ** 2 <= rc * rc + 1e-9
        else:
            keep = np.ones(dy.shape, dtype=bool)
        off = np.column_stack([dy[keep], dx[keep]]).astype(np.int64)
        offsets.append(off)
        half = {int(a): int(np.abs(off[off[:, 0] == a, 1]).max()) for a in np.unique(off[:, 0])}
        rows = sorted(half)
        rr, start = [], rows[0]
        for i, a in enumerate(rows):
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            if nxt is None or nxt != a + 1 or half[nxt] != half[a]:
                rr.append((start, a, -half[a], half[a]))
                start = nxt
        rects.append(tuple(rr))
        pad = max(pad, big)
    big_r = max(radii) / grid_um
    if shape == "square":
        reach, p_norm = 2.0 * np.floor(big_r + 1e-9), np.inf
    else:
        reach, p_norm = 2.0 * big_r + 1e-9, 2.0
    return _Windows(radii, shape, float(grid_um), tuple(offsets), tuple(rects), pad, float(reach), p_norm)


def _integral(counts: np.ndarray, pad: int) -> np.ndarray:
    """Zero-padded integral image (cumulative sums clipped at the grid border) for window sums by rectangles."""
    ny, nx = counts.shape
    integ = np.zeros((ny + 1 + 2 * pad, nx + 1 + 2 * pad), dtype=np.int32)
    integ[pad + 1 : pad + 1 + ny, pad + 1 : pad + 1 + nx] = counts.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    integ[pad + 1 + ny :, :] = integ[pad + ny : pad + 1 + ny, :]
    integ[:, pad + 1 + nx :] = integ[:, pad + nx : pad + 1 + nx]
    return integ


def _rect_sums(integ: np.ndarray, shape: tuple[int, int], pad: int, rects) -> np.ndarray:
    ny, nx = shape
    out = np.zeros((ny, nx), dtype=np.int32)
    for dy0, dy1, dx0, dx1 in rects:
        ya, yb, xa, xb = pad + dy0, pad + dy1 + 1, pad + dx0, pad + dx1 + 1
        out += integ[yb : yb + ny, xb : xb + nx]
        out -= integ[ya : ya + ny, xb : xb + nx]
        out -= integ[yb : yb + ny, xa : xa + nx]
        out += integ[ya : ya + ny, xa : xa + nx]
    return out


def _window_sums(counts: np.ndarray, win: _Windows, s: int) -> np.ndarray:
    """Units in the window of size ``s`` centred on every cell of the grid (exact integers, zero outside the grid)."""
    return _rect_sums(_integral(counts, win.pad), counts.shape, win.pad, win.rects[s])


# ------------------------------------------------------------------------------------------------------------ pool


@dataclass
class _Pool:
    """The record units of one stratum (chain x section), their grid cells and the window totals."""

    gy: np.ndarray  # (N,) int64
    gx: np.ndarray
    ny: int
    nx: int
    totals: np.ndarray  # (S, ny * nx) int32: chain units per window, for every centre cell
    t_min: np.ndarray  # (N,) int64: smallest window total over the windows containing each unit
    spot: np.ndarray | None  # (N,) int64 spot codes when several units share a spot (distinct-spot draws)
    xy: np.ndarray  # (N, 2) float coordinates in µm (locus merging)
    win: _Windows
    model: str
    t_unique: tuple[np.ndarray, ...] = ()  # per size: distinct positive window totals
    x_span: float = 0.0  # extent of the units along x (separates batched draws in one KD-tree)
    dup_rate: float = 0.0  # probability that two random units share a spot
    lin_p: np.ndarray | None = None  # (N,) cell index in the grid padded by win.pad on every side
    totals_p: np.ndarray | None = None  # (S, padded cells) int32 totals, 0 for centres outside the grid
    off_lin: tuple[np.ndarray, ...] = ()  # per size: window offsets as padded linear steps
    nx_p: int = 0

    @property
    def n(self) -> int:
        return len(self.gy)

    @property
    def n_cells(self) -> int:
        return self.ny * self.nx

    @property
    def n_cells_p(self) -> int:
        return (self.ny + 2 * self.win.pad) * (self.nx + 2 * self.win.pad)

    @property
    def dense_cost(self) -> float:
        """Rough cost (array-element operations) of evaluating every window of the grid for one draw."""
        return self.n_cells * (4.0 * sum(len(r) for r in self.win.rects) + 3.0 * self.win.n_sizes + 6.0)


_POOL_CACHE: dict[bytes, tuple[_Pool, float, float]] = {}


def _build_pool(
    x: np.ndarray,
    y: np.ndarray,
    spot: np.ndarray | None,
    win: _Windows,
    model: str,
    origin: tuple[float, float] | None = None,
) -> tuple[_Pool, float, float]:
    """Grid cells, window totals and the smallest window total of every unit (cached on the content of the pool).

    The grid starts at ``origin`` (default: the smallest coordinates of the units).
    """
    h = hashlib.blake2b(digest_size=20)
    for arr in (np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(y, dtype=float)):
        h.update(arr.tobytes())
    if spot is not None:
        h.update(np.ascontiguousarray(spot, dtype=np.int64).tobytes())
    h.update(repr((win.radii_um, win.shape, win.grid_um, model, spot is None, origin)).encode())
    key = h.digest()
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    x0, y0 = (float(x.min()), float(y.min())) if origin is None else (float(origin[0]), float(origin[1]))
    gx = np.floor_divide(x - x0, win.grid_um).astype(np.int64)
    gy = np.floor_divide(y - y0, win.grid_um).astype(np.int64)
    nx, ny = int(gx.max()) + 1, int(gy.max()) + 1
    counts = np.bincount(gy * nx + gx, minlength=nx * ny).reshape(ny, nx)
    integ = _integral(counts, win.pad)
    totals = np.stack([_rect_sums(integ, (ny, nx), win.pad, win.rects[s]).ravel() for s in range(win.n_sizes)])
    p = win.pad
    nyp, nxp = ny + 2 * p, nx + 2 * p
    totals_p = np.zeros((win.n_sizes, nyp, nxp), dtype=np.int32)
    totals_p[:, p : p + ny, p : p + nx] = totals.reshape(win.n_sizes, ny, nx)
    totals_p = totals_p.reshape(win.n_sizes, -1)
    off_lin = tuple((o[:, 0] * nxp + o[:, 1]).astype(np.int64) for o in win.offsets)
    lin_p = (gy + p) * nxp + (gx + p)
    # smallest window total among the windows (any size, centre on the grid) that contain each occupied cell
    occ = np.unique(lin_p)
    big = np.iinfo(np.int32).max
    tmin_cell = np.full(len(occ), big, dtype=np.int64)
    chunk = max(1, 4_000_000 // max(1, max(len(o) for o in off_lin)))
    for s, off in enumerate(off_lin):
        tp = totals_p[s]
        for a in range(0, len(occ), chunk):
            vals = tp[occ[a : a + chunk, None] + off[None, :]]
            vals = np.where(vals > 0, vals, big)  # 0 only outside the grid (a window holding the cell has T >= 1)
            np.minimum(tmin_cell[a : a + chunk], vals.min(axis=1), out=tmin_cell[a : a + chunk])
    t_min = tmin_cell[np.searchsorted(occ, lin_p)]
    n = len(x)
    if spot is not None and len(np.unique(spot)) == n:
        spot = None  # every unit on its own spot: plain draws
    t_unique = tuple(np.unique(totals[s][totals[s] > 0]) for s in range(win.n_sizes))
    xy = np.column_stack([x, y]).astype(float)
    pool = _Pool(gy, gx, ny, nx, totals, t_min, spot, xy, win, model, t_unique)
    pool.lin_p, pool.totals_p, pool.off_lin, pool.nx_p = lin_p, totals_p, off_lin, nxp
    pool.x_span = float(x.max() - x0)
    pool.dup_rate = float((np.bincount(spot).astype(float) ** 2).sum()) / n / n if spot is not None else 1.0 / n
    while len(_POOL_CACHE) >= 2:
        _POOL_CACHE.pop(next(iter(_POOL_CACHE)))
    _POOL_CACHE[key] = (pool, x0, y0)
    return pool, x0, y0


# ----------------------------------------------------------------------------------------------- null draws (rng)


def _draw_one(pool: _Pool, n_c: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``n_c`` pool units drawn at random, never two on one spot.

    The observed units are distinct (spot, clone) pairs, so a relabelling must not put two units of a clone on one
    spot: units are taken in random order and a unit on an already used spot is skipped (the legacy
    ``_draw_distinct_dnb``).
    """
    n = pool.n
    if pool.spot is None:
        return rng.choice(n, n_c, replace=False)
    k = min(n, int(n_c * 1.15) + 16)
    while True:
        idx = rng.choice(n, k, replace=False)
        _, first = np.unique(pool.spot[idx], return_index=True)
        keep = np.sort(first)
        if len(keep) >= n_c or k == n:
            return idx[keep[:n_c]]
        k = min(n, int(k * 1.5) + 1)


def _kept(pool: _Pool, seqs: np.ndarray) -> np.ndarray:
    """Positions of each row of i.i.d. draws that are the first draw of their unit and of their spot (mask)."""
    bsz, k = seqs.shape
    key = pool.spot[seqs] if pool.spot is not None else seqs
    width = int(key.max()) + 1
    flat = (np.arange(bsz, dtype=np.int64)[:, None] * width + key).ravel()
    _, first = np.unique(flat, return_index=True)
    mask = np.zeros(bsz * k, dtype=bool)
    mask[first] = True
    return mask.reshape(bsz, k)


def _draw_batch(pool: _Pool, n_c: int, bsz: int, rng: np.random.Generator) -> np.ndarray:
    """``bsz`` independent null draws of ``n_c`` units, shape ``(bsz, n_c)``.

    Units are drawn i.i.d. and a draw is skipped when it repeats a unit or a spot already drawn, which is the same
    sequential rule as drawing without replacement and skipping used spots (:func:`_draw_one`).
    """
    n = pool.n
    if n_c * 4 > n:
        return np.stack([_draw_one(pool, n_c, rng) for _ in range(bsz)])
    k = n_c + 32 + int(3.0 * n_c * n_c * max(pool.dup_rate, 1.0 / n))
    seqs = rng.integers(0, n, size=(bsz, k))
    kept = _kept(pool, seqs)
    take = kept & (np.cumsum(kept, axis=1) <= n_c)
    ok = take.sum(axis=1) == n_c
    out = np.empty((bsz, n_c), dtype=np.int64)
    out[ok] = seqs[ok][take[ok]].reshape(-1, n_c)
    for r in np.flatnonzero(~ok):  # extend the row's sequence until it holds n_c valid units (no conditioning)
        seq = seqs[r]
        while True:
            seq = np.r_[seq, rng.integers(0, n, size=k)]
            m = _kept(pool, seq[None, :])[0]
            if m.sum() >= n_c:
                out[r] = seq[np.flatnonzero(m)[:n_c]]
                break
    return out


@dataclass(frozen=True)
class _Blocks:
    """Cell-scale blocks of a pool: the units of one clone within ``radius_um`` of a seed unit (leader clustering)."""

    ptr: np.ndarray  # (B + 1,) offsets of each block's members in ``members``
    members: np.ndarray  # (N,) pool unit indices grouped by block, canonical (pool) order inside a block
    size: np.ndarray  # (B,) units per block
    radius_um: float

    @property
    def n(self) -> int:
        return len(self.size)

    @classmethod
    def from_ids(cls, ids: np.ndarray, radius_um: float) -> _Blocks:
        size = np.bincount(ids).astype(np.int64)
        return cls(np.r_[0, np.cumsum(size)].astype(np.int64), np.argsort(ids, kind="stable"), size, float(radius_um))


def _leader_blocks(xy: np.ndarray, codes: np.ndarray, radius_um: float) -> np.ndarray:
    """Block id of every unit (leader clustering within each clone, in canonical unit order).

    Within a clone, the first unit (pool order) that is not yet in a block seeds a block that takes every unit of the
    clone within ``radius_um`` of the seed that is not yet in a block. A cell whose units all lie within ``radius_um``
    of each other and away from other cells of its clone is one block; blocks never grow beyond ``radius_um`` around
    their seed, however dense the clone (single-linkage loci can chain across a whole germinal centre).
    """
    n = len(codes)
    out = np.full(n, -1, np.int64)
    if n == 0:
        return out
    r = float(radius_um) + 1e-9
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    ends = np.r_[starts[1:], n]
    seg = np.repeat(np.arange(len(starts)), ends - starts)
    first = order[starts]
    d = np.hypot(xy[order, 0] - xy[first[seg], 0], xy[order, 1] - xy[first[seg], 1])
    single = np.maximum.reduceat(d, starts) <= r
    # clones that form one block: numbered first, in clone order
    out[order[single[seg]]] = np.cumsum(single)[seg[single[seg]]] - 1
    nb = int(single.sum())
    for s in np.flatnonzero(~single):
        mem = order[starts[s] : ends[s]]
        pts = xy[mem]
        tree = cKDTree(pts)
        free = np.ones(len(mem), bool)
        for i in range(len(mem)):
            if free[i]:
                nbr = np.asarray(tree.query_ball_point(pts[i], r), dtype=np.int64)
                nbr = nbr[free[nbr]]
                out[mem[nbr]] = nb
                free[nbr] = False
                nb += 1
    return out


def _median_cluster_size(xy: np.ndarray, codes: np.ndarray, radius_um: float, max_units: int = 20_000) -> float:
    """Median size of the single-linkage clusters of same-clone units within ``radius_um`` (singletons included).

    Computed on the clones of the smallest codes up to about ``max_units`` units (a diagnostic; deterministic).
    """
    if len(codes) > max_units:
        sizes = np.bincount(codes)
        keep_codes = np.flatnonzero(np.cumsum(sizes) - sizes < max_units)
        sel = np.isin(codes, keep_codes)
        xy, codes = xy[sel], codes[sel]
    n = len(codes)
    if n < 2:
        return 1.0
    span = float(np.ptp(xy[:, 0])) + 2.0 * radius_um + 1.0
    pts = np.column_stack([xy[:, 0] + codes.astype(float) * span, xy[:, 1]])
    pairs = cKDTree(pts).query_pairs(radius_um, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0
    adj = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    return float(np.median(np.bincount(lab)))


def _expand_blocks(pool: _Pool, blocks: _Blocks, rows: np.ndarray, blk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Units of the given (row, block) sequence in order, without units on a spot already used by their row."""
    sz = blocks.size[blk]
    tot = int(sz.sum())
    cum = np.cumsum(sz) - sz
    pos = np.repeat(blocks.ptr[blk], sz) + (np.arange(tot, dtype=np.int64) - np.repeat(cum, sz))
    units = blocks.members[pos]
    urow = np.repeat(rows, sz)
    if pool.spot is not None:
        width = int(pool.spot.max()) + 1
        _, first = np.unique(urow * width + pool.spot[units], return_index=True)
        ok = np.zeros(tot, dtype=bool)
        ok[first] = True
        units, urow = units[ok], urow[ok]
    return units, urow


def _draw_blocks(pool: _Pool, blocks: _Blocks, n_c: int, bsz: int, rng: np.random.Generator) -> np.ndarray:
    """``bsz`` null clones of ``n_c`` units built from whole blocks, shape ``(bsz, n_c)``.

    Blocks are taken in uniformly random order (i.i.d. draws with repeated blocks skipped, i.e. the prefix of a random
    permutation), their units in pool order; a unit on a spot the draw already holds is skipped, and the draw stops
    at ``n_c`` units, so only the last block can be cut.
    """
    nb = blocks.n
    need = n_c * nb / pool.n  # expected blocks per draw
    out = np.empty((bsz, n_c), dtype=np.int64)
    if need * 4 > nb:
        for r in range(bsz):
            perm = rng.permutation(nb)
            units, _ = _expand_blocks(pool, blocks, np.zeros(nb, np.int64), perm)
            out[r] = units[:n_c]
        return out
    k = int(need * 1.25 + 6.0 * np.sqrt(need) + 16)
    seqs = rng.integers(0, nb, size=(bsz, k))
    flat = (np.arange(bsz, dtype=np.int64)[:, None] * nb + seqs).ravel()
    _, first = np.unique(flat, return_index=True)
    keep = np.zeros(bsz * k, dtype=bool)
    keep[first] = True
    rows = np.repeat(np.arange(bsz, dtype=np.int64), k)[keep]
    units, urow = _expand_blocks(pool, blocks, rows, seqs.ravel()[keep])
    cnt = np.bincount(urow, minlength=bsz)
    rank = np.arange(len(urow)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ok = cnt >= n_c
    take = (rank < n_c) & ok[urow]
    out[ok] = units[take].reshape(-1, n_c)
    for r in np.flatnonzero(~ok):  # extend the row's block sequence until it holds n_c units (no conditioning)
        seq = seqs[r]
        while True:
            seq = np.r_[seq, rng.integers(0, nb, size=k)]
            _, f = np.unique(seq, return_index=True)
            f = np.sort(f)
            u, _ = _expand_blocks(pool, blocks, np.zeros(len(f), np.int64), seq[f])
            if len(u) >= n_c:
                out[r] = u[:n_c]
                break
    return out


def _merge(pool: _Pool, idx: np.ndarray, radius_um: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Units of each draw merged into loci (single linkage at ``radius_um``), one representative per locus.

    Returns the representatives' pool indices and draw numbers (sorted by draw) and the number of loci per draw. The
    representative of a locus is its member with the smallest pool index (pool order is canonical), so observed and
    null clones are treated alike. Without a radius every unit is its own locus.
    """
    bsz, n_c = idx.shape
    flat = idx.ravel()
    b = np.repeat(np.arange(bsz, dtype=np.int64), n_c)
    if radius_um is None or n_c < 2:
        return flat, b, np.full(bsz, n_c, dtype=np.int64)
    pts = pool.xy[flat].copy()
    pts[:, 0] += b * (pool.x_span + 2.0 * radius_um + 1.0)
    pairs = cKDTree(pts).query_pairs(radius_um, output_type="ndarray")
    if len(pairs) == 0:
        return flat, b, np.full(bsz, n_c, dtype=np.int64)
    m = len(flat)
    adj = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    _, lab = connected_components(adj, directed=False)
    order = np.lexsort((flat, lab))
    first = order[np.r_[True, lab[order][1:] != lab[order][:-1]]]
    first = first[np.argsort(b[first], kind="stable")]
    return flat[first], b[first], np.bincount(b[first], minlength=bsz).astype(np.int64)


# --------------------------------------------------------------------------------------- window counts of a clone


def _scatter(lin_p: np.ndarray, b: np.ndarray, pool: _Pool) -> tuple[np.ndarray, np.ndarray]:
    """Sorted keys ``(b * S + s) * padded_cells + centre`` of every window holding >= 1 of the given units, and counts."""
    ns, ncp = pool.win.n_sizes, pool.n_cells_p
    keys = [((b[:, None] * ns + s) * ncp + lin_p[:, None] + off[None, :]).ravel() for s, off in enumerate(pool.off_lin)]
    return np.unique(np.concatenate(keys), return_counts=True)


def _observed(units: np.ndarray, pool: _Pool) -> dict[str, Any]:
    """Maximum LLR of one clone (its units or loci representatives) and its best window.

    Ties: the last size, then the first centre in row-major order (the archive's rule).
    """
    win, n_c = pool.win, len(units)
    cells_gy, cells_gx = pool.gy[units], pool.gx[units]
    per_size = []
    if n_c * win.n_offsets > 4.0 * pool.n_cells:
        counts = np.bincount(cells_gy * pool.nx + cells_gx, minlength=pool.n_cells).reshape(pool.ny, pool.nx)
        integ = _integral(counts, win.pad)
        for s in range(win.n_sizes):
            w = _rect_sums(integ, counts.shape, win.pad, win.rects[s]).ravel()
            cand = np.flatnonzero(w > 0)
            llr = scan_llr(w[cand], pool.totals[s][cand], n_c, pool.n, model=pool.model)
            i = int(np.argmax(llr)) if len(llr) else -1
            per_size.append((float(llr[i]), int(cand[i]), int(w[cand[i]])) if i >= 0 else (0.0, -1, 0))
    else:
        p = win.pad
        key, cnt = _scatter(pool.lin_p[units], np.zeros(n_c, dtype=np.int64), pool)
        tot = pool.totals_p.ravel()[key]
        inside = tot > 0  # centres outside the grid are not windows
        key, cnt, tot = key[inside], cnt[inside], tot[inside]
        s_of = key // pool.n_cells_p
        lp = key % pool.n_cells_p
        lin = (lp // pool.nx_p - p) * pool.nx + (lp % pool.nx_p - p)
        llr = scan_llr(cnt, tot, n_c, pool.n, model=pool.model)
        for s in range(win.n_sizes):
            sel = np.flatnonzero(s_of == s)
            if len(sel) == 0:
                per_size.append((0.0, -1, 0))
                continue
            i = sel[int(np.argmax(llr[sel]))]
            per_size.append((float(llr[i]), int(lin[i]), int(cnt[i])))
    stat = max(v for v, _, _ in per_size)
    best = (0.0, -1, -1, 0, 0)
    for s, (v, c, w) in enumerate(per_size):
        if c >= 0 and v >= stat - 1e-9:
            best = (v, s, c, w, int(pool.totals[s][c]))
    return {"statistic": stat, "size": best[1], "centre": best[2], "observed": best[3], "total": best[4]}


def _null_maxima(
    pool: _Pool,
    n_c: int,
    n_draws: int,
    rng: np.random.Generator,
    stop: tuple[float, int] | None = None,
    merge_um: float | None = None,
    blocks: _Blocks | None = None,
) -> np.ndarray:
    """Maximum LLR of ``n_draws`` random relabellings of ``n_c`` units (stops early once ``stop = (S, h)`` is met).

    With ``blocks`` the units are drawn as whole cell-scale blocks (:func:`_draw_blocks`), otherwise one by one. With
    ``merge_um`` the drawn units are merged into loci before the statistic, exactly as the observed clone was.
    Large clones go to the whole-grid evaluation directly; otherwise a first batch measures the cost of the sparse
    evaluation and switches if the grid is cheaper. Both evaluations give identical maxima.
    """
    win = pool.win
    keys_cap = pool.dense_cost / 60.0
    mode = "dense" if n_c * win.n_offsets > keys_cap else None
    out = np.empty(n_draws)
    done, exceed, keys_per_draw = 0, 0, 1.0
    while done < n_draws:
        if mode == "dense":
            bsz = min(4, n_draws - done)
        elif mode is None:
            bsz = min(8, n_draws - done)
        else:
            bsz = min(int(np.clip(_KEYS_PER_BATCH // max(1.0, keys_per_draw), 1, 512)), n_draws - done)
        drawn = _draw_batch(pool, n_c, bsz, rng) if blocks is None else _draw_blocks(pool, blocks, n_c, bsz, rng)
        units, b, m = _merge(pool, drawn, merge_um)
        if mode == "dense":
            vals = _max_dense(units, b, m, pool)
        else:
            vals, nkeys = _max_sparse(units, b, m, pool)
            if mode is None:
                keys_per_draw = max(1.0, nkeys / bsz)
                mode = "dense" if keys_per_draw > keys_cap else "sparse"
        out[done : done + bsz] = vals
        if stop is not None:
            exceed += int((vals >= stop[0] - _TIE * abs(stop[0])).sum())
            if exceed >= stop[1]:
                return out[: done + bsz]
        done += bsz
    return out


def _segments(b: np.ndarray, bsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Start positions of each draw's block in ``b`` (sorted) and the draws present."""
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]]) if len(b) else np.empty(0, np.int64)
    return starts, b[starts]


def _single_unit_bound(units: np.ndarray, b: np.ndarray, m: np.ndarray, pool: _Pool) -> np.ndarray:
    """Best LLR of windows holding one unit of each draw: bounded via the units' smallest window totals."""
    bsz = len(m)
    out = np.zeros(bsz)
    if len(units) == 0:
        return out
    starts, present = _segments(b, bsz)
    tmin = np.minimum.reduceat(pool.t_min[units], starts)
    out[present] = scan_llr(1.0, tmin, m[present], pool.n, model=pool.model)
    return out


def _max_dense(units: np.ndarray, b: np.ndarray, m: np.ndarray, pool: _Pool) -> np.ndarray:
    win = pool.win
    res = _single_unit_bound(units, b, m, pool)
    starts, present = _segments(b, len(m))
    ends = np.r_[starts[1:], len(b)]
    for a, e, row in zip(starts, ends, present, strict=True):
        u = units[a:e]
        n_c = int(m[row])
        counts = np.bincount(pool.gy[u] * pool.nx + pool.gx[u], minlength=pool.n_cells).reshape(pool.ny, pool.nx)
        integ = _integral(counts, win.pad)
        best = res[row]
        for s in range(win.n_sizes):
            w = _rect_sums(integ, counts.shape, win.pad, win.rects[s]).ravel()
            cand = np.flatnonzero(w >= 2)
            if len(cand):
                wc, tc = w[cand], pool.totals[s][cand]
                ex = wc.astype(np.int64) * pool.n > tc.astype(np.int64) * n_c
                if ex.any():
                    best = max(best, float(scan_llr(wc[ex], tc[ex], n_c, pool.n, model=pool.model).max()))
        res[row] = best
    return res


def _max_sparse(units: np.ndarray, b: np.ndarray, m: np.ndarray, pool: _Pool) -> tuple[np.ndarray, int]:
    """Null maxima of a batch of draws, and the number of scatter keys it needed.

    Windows holding exactly one unit of the clone are bounded by the unit's smallest window total (``t_min``; the LLR
    falls with the expectation); windows holding two or more contain only units that have another unit of the same
    draw within the window reach, so only those units are scattered.
    """
    win = pool.win
    bsz = len(m)
    best = _single_unit_bound(units, b, m, pool)
    if len(units) < 2:
        return best, 0
    step = float(pool.nx + 2.0 * win.reach + 4.0)
    pts = np.column_stack([pool.gx[units] + b * step, pool.gy[units]]).astype(float)
    d, _ = cKDTree(pts).query(pts, k=2, p=win.p_norm, distance_upper_bound=win.reach + 1e-9)
    sel = np.flatnonzero(np.isfinite(d[:, 1]))
    nkeys = len(sel) * win.n_offsets
    if len(sel):
        key, cnt = _scatter(pool.lin_p[units[sel]], b[sel], pool)
        two = cnt >= 2
        key, cnt = key[two], cnt[two]
        span = win.n_sizes * pool.n_cells_p
        tot = pool.totals_p.ravel()[key % span]
        bk = key // span
        n_c = m[bk]
        ex = cnt.astype(np.int64) * pool.n > tot.astype(np.int64) * n_c
        if ex.any():
            key, cnt, tot, bk, n_c = key[ex], cnt[ex], tot[ex], bk[ex], n_c[ex]
            llr = scan_llr(cnt, tot, n_c, pool.n, model=pool.model)
            starts = np.flatnonzero(np.r_[True, bk[1:] != bk[:-1]])
            s2 = np.zeros(bsz)
            s2[bk[starts]] = np.maximum.reduceat(llr, starts)
            best = np.maximum(best, s2)
    return best, nkeys


# ------------------------------------------------------------------------------------- attainable p (pre-filter)


def _p_min(n_c: int, pool: _Pool) -> float:
    """Lower bound on the smallest p-value the exact relabelling test of a clone of ``n_c`` units can reach.

    One unit: exact (the share of pool units whose smallest window total is the pool's minimum). More units: the
    hypergeometric probability that the windows reaching the largest attainable LLR hold that configuration.
    """
    if n_c <= 1:
        t_star = int(pool.t_min.min())
        return float((pool.t_min == t_star).mean())
    best, best_lp = -np.inf, -np.inf
    for s in range(pool.win.n_sizes):
        tv = pool.t_unique[s]
        wv = np.minimum(n_c, tv)
        llr = scan_llr(wv, tv, n_c, pool.n, model=pool.model)
        i = int(np.argmax(llr))
        if llr[i] > best + 1e-12:
            best = float(llr[i])
            best_lp = float(stats.hypergeom.logpmf(wv[i], pool.n, tv[i], n_c))
        elif abs(llr[i] - best) <= 1e-12:
            best_lp = max(best_lp, float(stats.hypergeom.logpmf(wv[i], pool.n, tv[i], n_c)))
    return float(np.exp(best_lp)) if np.isfinite(best_lp) else 0.0


def _tarone_testable(p_min: np.ndarray, alpha: float) -> np.ndarray:
    """Tarone (1990) / Gilbert (2005): smallest K with #{p_min <= alpha / K} <= K; testable = p_min <= alpha / K."""
    m = len(p_min)
    if m == 0:
        return np.zeros(0, dtype=bool)
    srt = np.sort(p_min)
    for k in range(1, m + 1):
        if np.searchsorted(srt, alpha / k, side="right") <= k:
            return p_min <= alpha / k
    return np.ones(m, dtype=bool)  # pragma: no cover - k = m always satisfies the condition


# ------------------------------------------------------------------------------------------------ worker plumbing

_WORKER: dict[str, Any] = {}


def _init_worker(pool: _Pool, merge_um: float | None, blocks: _Blocks | None = None) -> None:
    _WORKER["pool"] = pool
    _WORKER["merge_um"] = merge_um
    _WORKER["blocks"] = blocks


def _task(args: tuple[int, int, int, dict, tuple[float, int] | None]) -> tuple[int, np.ndarray, dict]:
    key, n_c, n_draws, state, stop = args
    gen = np.random.Generator(np.random.PCG64())
    gen.bit_generator.state = state
    maxima = _null_maxima(_WORKER["pool"], n_c, n_draws, gen, stop, _WORKER["merge_um"], _WORKER["blocks"])
    return key, maxima, gen.bit_generator.state


# -------------------------------------------------------------------------------------------------- the main loop


@dataclass
class _Clone:
    code: int
    n_c: int
    stat: float
    gen: np.random.Generator
    base: list
    n: int = 0  # draws made
    g: int = 0  # exceedances among them
    ties: int = 0  # base draws whose maximum equals the observed (counted as exceedances)
    stop_at: int = 0  # draw index (1-based) of the h-th exceedance, 0 if not reached
    stop: str = ""
    cap: int = 0


def _bh_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    return bh(p) <= alpha


def _run_stratum(
    pool: _Pool,
    clones: list[_Clone],
    *,
    n_perm: int,
    n_perm_max: int,
    h: int,
    stopping: str,
    alpha: float,
    n_jobs: int,
    merge_um: float | None,
    blocks: _Blocks | None = None,
) -> None:
    """Draw relabellings for every clone until its p-value is settled (Besag–Clifford, optionally BH-driven caps)."""
    if not clones:
        return
    for c in clones:
        c.cap = n_perm
    n_stages = 1 + max(0, math.ceil(math.log2((n_perm_max + 1) / (n_perm + 1))))
    eps = _EPS_MMC / (len(clones) * n_stages)
    executor = None
    if n_jobs > 1:
        methods = multiprocessing.get_all_start_methods()
        ctx = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=ctx, initializer=_init_worker, initargs=(pool, merge_um, blocks)
        )
    try:
        active = list(clones)
        while active:
            # base draws are always made in full (null summaries); later batches stop at the h-th exceedance
            tasks = [(c, c.cap - c.n, (c.stat, h - c.g) if c.n >= n_perm else None) for c in active]
            if executor is None:
                for c, need, stop in tasks:
                    _absorb(c, _null_maxima(pool, c.n_c, need, c.gen, stop, merge_um, blocks), n_perm, h)
            else:
                futs = [
                    executor.submit(_task, (i, c.n_c, need, c.gen.bit_generator.state, stop))
                    for i, (c, need, stop) in enumerate(tasks)
                ]
                for f in futs:
                    i, maxima, state = f.result()
                    c = tasks[i][0]
                    c.gen.bit_generator.state = state
                    _absorb(c, maxima, n_perm, h)
            nxt = []
            for c in active:
                if c.stop_at:
                    c.stop = "exceedances"
                elif c.n >= n_perm_max:
                    c.stop = "cap"
                else:
                    nxt.append(c)
            if stopping == "bh" and nxt:
                nxt = _bh_settle(clones, nxt, n_perm_max, h, alpha, eps)
            for c in nxt:
                c.cap = min(2 * c.n + 1, n_perm_max)
            active = nxt
    finally:
        if executor is not None:
            executor.shutdown()


def _absorb(c: _Clone, maxima: np.ndarray, n_perm: int, h: int) -> None:
    hit = maxima >= c.stat - _TIE * abs(c.stat)
    if c.n < n_perm:
        head = maxima[: n_perm - c.n]
        c.base.extend(head.tolist())
        c.ties += int((np.abs(head - c.stat) <= _TIE * max(abs(c.stat), 1e-300)).sum())
    if not c.stop_at:
        pos = np.flatnonzero(hit)
        if c.g + len(pos) >= h:
            c.stop_at = c.n + int(pos[h - c.g - 1]) + 1
    c.g += int(hit.sum())
    c.n += len(maxima)


def _p_value(c: _Clone, h: int, stopping: str) -> tuple[float, int]:
    """Besag–Clifford p-value and the draws it rests on."""
    if c.stop_at:
        return h / c.stop_at, c.stop_at
    if stopping == "besag_clifford":
        return (c.g + 1) / (c.n + 1), c.n
    return h / (c.n + 1), c.n


def _bh_settle(
    clones: list[_Clone], running: list[_Clone], n_perm_max: int, h: int, alpha: float, eps: float
) -> list[_Clone]:
    """Running clones whose BH decision is still open (MMCTest logic); the others stop."""
    idx = {id(c): i for i, c in enumerate(clones)}
    p_now = np.array([_p_value(c, h, "bh")[0] for c in clones])
    lo = p_now.copy()
    run_i = np.array([idx[id(c)] for c in running])
    g = np.array([min(c.g, c.n) for c in running], dtype=float)
    n = np.array([c.n for c in running], dtype=float)
    cp_lo = np.where(g > 0, stats.beta.ppf(eps / 2, np.maximum(g, 1e-12), n - g + 1), 0.0)
    lo[run_i] = np.maximum(h / (n_perm_max + 1), cp_lo)
    rej_now = _bh_reject(p_now, alpha)
    rej_lo = _bh_reject(lo, alpha)
    keep = []
    for c, i in zip(running, run_i, strict=True):
        if rej_now[i]:
            c.stop = "rejected"  # further draws can only lower its p-value
        elif not rej_lo[i]:
            c.stop = "decided"  # not rejected even at the smallest p-value it can plausibly reach
        else:
            keep.append(c)
    return keep


# ------------------------------------------------------------------------------------------------------- public API


def _as_list(v: str | Sequence[str] | None) -> list[str] | None:
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def clonal_scan(
    air: pd.DataFrame,
    *,
    locus: str | Sequence[str] | None = None,
    units: Literal["bins", "records", "loci"] = "bins",
    null: Literal["blocks", "units"] = "blocks",
    block_radius_um: float = 25.0,
    radii_um: Sequence[float] = (10.0, 20.0, 30.0, 50.0, 75.0, 110.0),
    shape: Literal["circle", "square"] = "circle",
    grid_um: float = 10.0,
    model: Literal["poisson", "bernoulli"] = "poisson",
    min_units: int = 20,
    clones: Sequence[str] | None = None,
    n_perm: int = 199,
    n_perm_max: int | None = None,
    h: int = 10,
    stopping: Literal["bh", "besag_clifford"] = "bh",
    prefilter: bool = True,
    alpha: float = 0.05,
    loci_radius_um: float = 15.0,
    by: str | Sequence[str] | None = None,
    rng: RngLike = None,
    n_jobs: int = 1,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
    library_key: str = S.LIBRARY,
) -> TestResult:
    """Scan every clone for spatial concentration against random relabelling of its chain's cells.

    Each chain (and section) is scanned on its own: its units are the pool, and every clone with at least
    ``min_units`` cell-scale blocks is tested. The statistic of a clone is Kulldorff's log-likelihood ratio maximised
    over windows of the given radii centred on every cell of a ``grid_um`` grid covering the pool; the p-value is a
    Besag–Clifford sequential Monte-Carlo p-value against random relabellings of whole cell-scale blocks that recompute
    the maximum over the same windows; q-values are Benjamini–Hochberg within the chain.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or call) with spot ids, coordinates in µm, chain and clone.
    locus
        Chain(s) to scan, e.g. ``"IGK"``; ``None`` scans every chain present.
    units
        What the clone and the pool are counted in. ``"bins"`` (default): one unit per (``grid_um`` cell, clone) pair,
        placed at the cell centre, so that a cell with many molecules weighs about as much as its area.
        ``"records"``: one unit per (spot, clone) pair, the archive's ClonalScan unit (at sub-cellular resolution a
        cell with many molecules weighs as many units as it has spots). ``"loci"``: the clone's (spot, clone) units
        merged by single linkage at ``loci_radius_um`` (as :func:`spaclone.pp.merge_loci`), each locus placed at its
        first spot in the canonical (x, y) order, against a pool of (spot, clone) units (the spaclone 1.0.dev0
        default). Single-linkage loci chain the cells of a dense clone together, so a clone packed into a germinal
        centre becomes a few loci and is missed (power <= 0.21 against >= 0.98 counted in units, F4a).
    null
        ``"blocks"`` (default): whole cell-scale blocks are relabelled (see Notes); calibrated whether a clone's
        labels are random over cells or over molecules. ``"units"``: units are relabelled one by one, the archive's
        design, valid only when units are independent placements (every cell on one unit, or every molecule placed
        at random); with cells covering several units it calls most null clones concentrated (a
        :class:`~spaclone._utils.SpacloneWarning` is raised when same-clone units form clusters of a median of more
        than 1.5 units within ``block_radius_um``).
    block_radius_um
        Radius of the blocks in µm (leader clustering, see Notes): at least the diameter of the largest cell plus the
        diagonal of a unit (a grid cell for ``units="bins"``, a spot for ``"records"``), so that one cell is never
        split between blocks; larger blocks only make the test more conservative. The default 25 µm keeps cells up
        to about 12 µm whole on the default 10-µm grid (a 15-µm radius split plasma-like 12-µm cells: up to 11 %
        false positives at alpha = 0.05, F4a).
    radii_um
        Window radii in µm. A window holds the grid cells whose centres lie within the radius of the window's centre
        cell (``shape="circle"``) or within the square of that half-side (``shape="square"``). The archive's windows
        (squares of 10, 30, 50 and 110 µm on a 10-µm grid) are ``shape="square", radii_um=(5, 15, 25, 55)``.
    shape
        ``"circle"`` (default) or ``"square"``.
    grid_um
        Side of the grid cells on which units are counted and windows are centred.
    model
        Likelihood: ``"poisson"`` (conditional Poisson, the archive's) or ``"bernoulli"`` (case / control). Both give
        exact tests under random relabelling; they rank windows slightly differently for large clones.
    min_units
        Clones with fewer cell-scale units are not tested (and do not count towards the BH denominator): blocks with
        ``null="blocks"`` (and ``units`` "bins" or "records"), loci with ``units="loci"``, units with
        ``null="units"``.
    clones
        Test only these clone ids (all units still form the pool), e.g. a planted clone in a power study.
    n_perm
        Relabellings always made per tested clone; they give ``null_mean``, ``null_sd``, ``z`` and ``p_plain``.
    n_perm_max
        Largest number of relabellings for one clone. ``None``: 9,999 with ``stopping="besag_clifford"``; with
        ``stopping="bh"`` the smallest value on the doubling ladder from ``n_perm`` for which the p-value floor
        ``h / (n_perm_max + 1)`` is at most half of BH's first threshold ``alpha / m`` (at least 9,999).
    h
        Exceedances that stop the sequential test of a clone (Besag–Clifford).
    stopping
        ``"bh"`` (default): after the base draws a clone keeps drawing (caps doubling up to ``n_perm_max``) only while
        its BH decision is open: it stops when it has ``h`` exceedances, when BH already rejects it at its current
        p-value (more draws can only lower it), or when it would not be rejected even at the smallest p-value it can
        plausibly still reach (Clopper–Pearson bound, error 0.001 over clones and stages). ``"besag_clifford"``: every
        clone draws until ``h`` exceedances or ``n_perm_max`` (the archive's rule).
    prefilter
        Leave out clones whose smallest attainable p-value under exact relabelling cannot pass Tarone's threshold
        (Gilbert's discrete-FDR procedure: they would only enlarge ``m``); in practice clones of one unit.
    alpha
        FDR level of the BH decisions (column ``significant``).
    loci_radius_um
        Merge radius of ``units="loci"`` (µm).
    by
        Strata scanned separately (each its own pool and BH family). ``None``: the chain and library columns present.
    rng
        Seed or generator; each clone gets its own child stream, so results do not depend on ``n_jobs``.
    n_jobs
        Worker processes.
    spot_key, x_key, y_key, locus_key, clone_key, library_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per clone with at least ``min_units`` cell-scale units: the strata columns,
        ``clone_id``, ``n_records`` (units of the clone: (grid cell, clone) pairs with ``units="bins"``, (spot,
        clone) pairs otherwise), ``n_units`` (what the statistic counts: loci with ``units="loci"``, else
        ``n_records``), ``n_blocks`` (its blocks; NaN with ``null="units"``), ``statistic`` (max LLR), ``null_mean``,
        ``null_sd``, ``z`` (from the base draws), ``p_value``, ``q_value``,
        ``n_perm`` (draws the p-value rests on), ``n_exceed``, ``p_plain`` (base-draw Monte-Carlo p-value), ``p_min``
        (attainable lower bound), ``testable``, ``significant`` (q <= alpha), ``stop`` (why sampling stopped), and the
        best window: ``best_radius_um``, ``best_x_um``, ``best_y_um`` (its centre), ``best_observed``, ``best_total``
        (chain units in it), ``best_expected``, ``best_ratio``. ``null``: base draws, shape ``(n_perm, n_rows)``.

    Notes
    -----
    Null model (``null="blocks"``): random relabelling of cells within each stratum (chain x section). The units of
    every clone of the stratum are grouped into blocks by leader clustering: in canonical unit order, the first unit
    not yet in a block seeds a block holding every unit of its clone within ``block_radius_um`` of the seed that is not
    yet in a block. A relabelled clone takes blocks of the stratum (of any clone) in uniformly random order, with all
    their units (a unit on a spot or grid cell the clone already holds is skipped), until it has the observed clone's
    number of units; the last block may be cut. The positions, the pool, the blocks, the clone's number of units and
    the window set are fixed; every relabelling recomputes the maximum over the same windows (with ``units="loci"``
    the drawn units are merged into loci first, as the observed clone's were). If clone labels are random over cells
    and every cell lies within one block, the observed clone is one draw from this null; blocks that hold several
    cells of one clone (dense clones), or several independently placed molecules (misplaced receptor records), make
    the null clumpier than the observed clone, i.e. the test conservative.

    Calibration (F4a design study; 200 chips per setting; cells of radius 4 µm, or 15 % plasma-like cells of radius
    6 µm with 5x molecules, covering 1.1-9.4 spots; uniform or clumped pools; labels random over cells, over molecules,
    or half of the molecules misplaced; 8 small (20-50 cells) and one large (15 % of the cells) null clone per chip):
    per-clone false-positive rate at alpha = 0.05 of the default design 0-4.1 % for small and 0-8.0 % for large
    clones, within binomial tolerance in all 80 cells; power at alpha = 0.05 for a clone of 30 cells with 15 of them
    packed together 0.92-1.0, and for a germinal-centre-like clone (75 of its 150 cells packed in a clump) 0.99-1.0.
    The legacy designs on the same chips: ``units="records", null="units"`` 15-100 % false positives as soon as cells
    cover more than one spot; ``units="loci", null="units"`` up to 27 %. On the reviewer's chips (review 2026-10-30,
    B4: cells covering 4-5 spots) the default called 3.4 % of null clones and 2 % of chips at FDR 5 %, against
    22 % and 39 % for ``units="loci", null="units"``.

    Null model (``null="units"``): conditional on the positions of the stratum's units, the units of a clone are a
    uniformly random subset of the units of the same size, never two on one spot (units are taken in random order and a
    unit on a used spot is skipped). With ``units="loci"`` the drawn units are merged into loci exactly as the
    observed clone's were, and the statistic uses the number of loci. Exact when units are exchangeable (labels random
    over molecules, or every cell on one unit), anti-conservative when cells cover several units.

    Legacy settings: the archive's ClonalScan (03 B5, B6) is ``units="records", null="units", shape="square",
    radii_um=(5, 15, 25, 55), stopping="besag_clifford", prefilter=False``; the spaclone 1.0.dev0 default is
    ``units="loci", null="units"``. The first reproduces the archived numbers exactly, the second the package's
    earlier validation rows (``validation/results/m4_scan.csv``).

    p-values: Besag–Clifford: ``h / L`` if the ``h``-th exceedance (null maximum >= observed; ties count) comes at
    draw ``L``. A clone that stops before that reports ``h / (n + 1)`` after ``n`` draws with ``stopping="bh"``,
    which is at least ``h / L`` for the draw ``L`` at which the ``h``-th exceedance would have come, so the p-value is
    valid whatever rule chose ``n``; ``stopping="besag_clifford"`` reports ``(g + 1) / (n + 1)`` at the fixed cap,
    the archive's convention. With ``m`` tested clones a single genuine clone needs about ``h m / alpha`` draws to
    pass BH; the "bh" rule spends them only on clones whose decision depends on them.

    References
    ----------
    Kulldorff & Nagarwalla 1995, doi:10.1002/sim.4780140809; Kulldorff 1997, doi:10.1080/03610929708831995; Besag &
    Clifford 1991, doi:10.1093/biomet/78.2.301; Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x; Gandy & Hahn 2014, doi:10.1111/sjos.12085; Tarone 1990,
    doi:10.2307/2531456; Gilbert 2005, doi:10.1111/j.1467-9876.2005.00475.x.

    Examples
    --------
    >>> res = clonal_scan(air, locus="IGK", rng=0)  # doctest: +SKIP
    >>> res.table.query("significant")  # doctest: +SKIP
    """
    if units not in ("bins", "records", "loci"):
        raise ValueError(f"units must be 'bins', 'records' or 'loci', got {units!r}")
    if null not in ("blocks", "units"):
        raise ValueError(f"null must be 'blocks' or 'units', got {null!r}")
    if not (np.isfinite(block_radius_um) and block_radius_um >= 0):
        raise ValueError("block_radius_um must be a finite number >= 0")
    if stopping not in ("bh", "besag_clifford"):
        raise ValueError(f"stopping must be 'bh' or 'besag_clifford', got {stopping!r}")
    if model not in ("poisson", "bernoulli"):
        raise ValueError(f"model must be 'poisson' or 'bernoulli', got {model!r}")
    if n_perm < 1 or h < 1:
        raise ValueError("n_perm and h must be >= 1")
    if n_perm_max is not None and n_perm_max < n_perm:
        raise ValueError("n_perm_max must be >= n_perm")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if min_units < 1:
        raise ValueError("min_units must be >= 1")
    S.validate_air(air, required=(spot_key, x_key, y_key, clone_key), numeric=(x_key, y_key))
    win = _make_windows(radii_um, grid_um, shape)
    gen = as_rng(rng)
    strata = _as_list(by) if by is not None else [c for c in (locus_key, library_key) if c in air.columns]
    missing = [c for c in strata if c not in air.columns]
    if missing:
        raise KeyError(f"clonal_scan: strata column(s) {missing} not in the table")
    data = air[air[clone_key].notna()]
    loci_sel = _as_list(locus)
    if loci_sel is not None:
        if locus_key not in data.columns:
            raise KeyError(f"clonal_scan: locus={locus!r} needs the column {locus_key!r}")
        data = data[data[locus_key].isin(loci_sel)]
    only = None if clones is None else set(map(str, clones))
    groups = list(data.groupby(strata, sort=True, dropna=False, observed=True)) if strata else [((), data)]
    seeds = _spawn(gen, len(groups))
    tables, nulls, info = [], [], {}
    for (gkey, g), seed in zip(groups, seeds, strict=True):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        tab, nul, inf = _scan_stratum(
            g,
            dict(zip(strata, gkey, strict=True)),
            win,
            seed,
            units=units,
            null=null,
            block_radius_um=block_radius_um,
            model=model,
            min_units=min_units,
            only=only,
            n_perm=n_perm,
            n_perm_max=n_perm_max,
            h=h,
            stopping=stopping,
            prefilter=prefilter,
            alpha=alpha,
            loci_radius_um=loci_radius_um,
            n_jobs=n_jobs,
            keys=(spot_key, x_key, y_key, clone_key),
        )
        tables.append(tab)
        nulls.append(nul)
        info["|".join(map(str, gkey)) or "all"] = inf
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    null_draws = np.concatenate(nulls, axis=1) if nulls and sum(n.shape[1] for n in nulls) else None
    tested = table[table["testable"]] if len(table) else table
    summary = {
        "n_tested": int(len(tested)),
        "n_significant": int(tested["significant"].sum()) if len(tested) else 0,
        "min_p": float(tested["p_value"].min()) if len(tested) else float("nan"),
        "min_q": float(tested["q_value"].min()) if len(tested) else float("nan"),
        "n_perm_total": int(tested["n_drawn"].sum()) if len(tested) else 0,
        "alpha": alpha,
    }
    params = {
        "units": units,
        "null_design": null,
        "block_radius_um": block_radius_um if null == "blocks" else None,
        "radii_um": list(win.radii_um),
        "shape": shape,
        "grid_um": grid_um,
        "model": model,
        "min_units": min_units,
        "clones": None if clones is None else list(clones),
        "n_perm": n_perm,
        "n_perm_max": n_perm_max,
        "h": h,
        "stopping": stopping,
        "prefilter": prefilter,
        "alpha": alpha,
        "loci_radius_um": loci_radius_um,
        "by": strata,
        "locus": loci_sel,
        "null": (
            f"random relabelling of whole cell-scale blocks (same-clone units within {block_radius_um:g} um of a seed)"
            if null == "blocks"
            else "random relabelling of the stratum's (spot, clone) units, one by one"
        ),
        "strata": info,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="clonal_scan", table=table, summary=summary, params=params, null=null_draws)


def _spawn(gen: np.random.Generator, k: int) -> list[np.random.SeedSequence]:
    ss = getattr(gen.bit_generator, "seed_seq", None)
    if ss is None:  # pragma: no cover - generators built from a raw bit generator
        ss = np.random.SeedSequence(int(gen.integers(0, 2**63 - 1)))
    return list(ss.spawn(k))


def _scan_stratum(
    g: pd.DataFrame,
    stratum: dict[str, Any],
    win: _Windows,
    seed: np.random.SeedSequence,
    *,
    units: str,
    null: str,
    block_radius_um: float,
    model: str,
    min_units: int,
    only: set[str] | None,
    n_perm: int,
    n_perm_max: int | None,
    h: int,
    stopping: str,
    prefilter: bool,
    alpha: float,
    loci_radius_um: float,
    n_jobs: int,
    keys: tuple[str, str, str, str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    spot_key, x_key, y_key, clone_key = keys
    u = g.drop_duplicates([spot_key, clone_key])
    u = pd.DataFrame(
        {
            "clone": u[clone_key].astype(str).to_numpy(),
            "x": u[x_key].to_numpy(float),
            "y": u[y_key].to_numpy(float),
            "spot": u[spot_key].astype(str).to_numpy(),
        }
    )
    origin = None
    if units == "bins" and len(u):  # one unit per (grid cell, clone), at the centre of the cell
        ox, oy = float(u["x"].min()), float(u["y"].min())
        gxx = np.floor_divide(u["x"].to_numpy() - ox, win.grid_um).astype(np.int64)
        gyy = np.floor_divide(u["y"].to_numpy() - oy, win.grid_um).astype(np.int64)
        u = pd.DataFrame(
            {
                "clone": u["clone"].to_numpy(),
                "x": ox + (gxx + 0.5) * win.grid_um,
                "y": oy + (gyy + 0.5) * win.grid_um,
                "spot": (gyy * (int(gxx.max()) + 1) + gxx).astype(str),
            }
        ).drop_duplicates(["spot", "clone"])
        origin = (ox, oy)
    # canonical unit order, so that results do not depend on the row order of the input
    u = u.sort_values(["x", "y", "spot", "clone"], kind="stable", ignore_index=True)
    spot_codes = pd.factorize(u["spot"])[0].astype(np.int64)
    info: dict[str, Any] = {"n_units": int(len(u))}
    if len(u) < 2:
        return pd.DataFrame(columns=list(stratum) + _COLUMNS), np.empty((n_perm, 0)), info
    pool, x0, y0 = _build_pool(u["x"].to_numpy(), u["y"].to_numpy(), spot_codes, win, model, origin)
    info.update({"grid": [pool.ny, pool.nx], "x0_um": x0, "y0_um": y0})
    merge_um = float(loci_radius_um) if units == "loci" else None
    codes, names = pd.factorize(u["clone"], sort=True)
    sizes = np.bincount(codes)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(sizes)]
    cand = np.flatnonzero(sizes >= min_units)  # loci and blocks never outnumber records
    if only is not None:
        cand = cand[np.isin(names[cand], list(only))]
    blocks, n_blk = None, None
    if null == "blocks":
        blocks = _Blocks.from_ids(_leader_blocks(pool.xy, codes.astype(np.int64), block_radius_um), block_radius_um)
        n_blk = np.bincount(codes[blocks.members[blocks.ptr[:-1]]], minlength=len(names))
        info.update({"n_blocks": int(blocks.n), "units_per_block_median": float(np.median(blocks.size))})
    else:
        med = _median_cluster_size(pool.xy, codes.astype(np.int64), block_radius_um)
        info["units_per_cluster_median"] = med
        if med > 1.5:
            warnings.warn(
                f"clonal_scan: null='units' relabels (spot, clone) units one by one, but same-clone units here come "
                f"in clusters of a median {med:g} units within {block_radius_um:g} um (cells spanning several spots): "
                "the null breaks the clusters apart and the p-values are anti-conservative. Use null='blocks'.",
                SpacloneWarning,
                stacklevel=3,
            )
    reps, n_loci = {}, {}
    for k in cand:
        mem = order[bounds[k] : bounds[k + 1]]
        r_, _, mm = _merge(pool, mem[None, :], merge_um)
        reps[k], n_loci[k] = r_, int(mm[0])
    # cell-scale size of a clone: its blocks (records counted, block null), its loci (units="loci") or its records
    scale = {k: int(n_blk[k]) if (n_blk is not None and units != "loci") else n_loci[k] for k in cand}
    test = np.array([k for k in cand if scale[k] >= min_units], dtype=np.int64)
    if len(test) == 0:
        return pd.DataFrame(columns=list(stratum) + _COLUMNS), np.empty((n_perm, 0)), info
    p_min = np.array([_p_min(int(sizes[k]), pool) for k in test])
    testable = _tarone_testable(p_min, alpha) if prefilter else np.ones(len(test), dtype=bool)
    m = int(testable.sum())
    if n_perm_max is None:
        if stopping == "besag_clifford":
            cap = max(9999, n_perm)
        else:
            need = 2.0 * h * max(m, 1) / alpha
            cap = n_perm
            while cap + 1 < need or cap < 9999:
                cap = 2 * cap + 1
    else:
        cap = int(n_perm_max)
    floor = (1.0 if stopping == "besag_clifford" else h) / (cap + 1)
    if m and floor > alpha / m:
        k_need = math.ceil(floor * m / alpha - 1e-9)
        warnings.warn(
            f"clonal_scan: the p-value floor {floor:.2g} of {cap} draws exceeds BH's first threshold alpha/m = "
            f"{alpha / m:.2g} ({m} clones): a clone at the floor is called only together with at least {k_need - 1} "
            "others. Use stopping='bh' or raise n_perm_max.",
            LowPowerWarning,
            stacklevel=3,
        )
    info.update({"n_tested": m, "n_perm_max": cap})
    child = seed.spawn(len(test))
    work: list[_Clone] = []
    obs_rows = []
    for j, k in enumerate(test):
        ob = _observed(reps[k], pool)
        obs_rows.append(ob)
        if testable[j]:
            work.append(
                _Clone(code=j, n_c=int(sizes[k]), stat=ob["statistic"], gen=np.random.default_rng(child[j]), base=[])
            )
    _run_stratum(
        pool,
        work,
        n_perm=n_perm,
        n_perm_max=cap,
        h=h,
        stopping=stopping,
        alpha=alpha,
        n_jobs=n_jobs,
        merge_um=merge_um,
        blocks=blocks,
    )
    by_code = {c.code: c for c in work}
    rows, base = [], np.full((n_perm, len(test)), np.nan)
    for j, k in enumerate(test):
        ob = obs_rows[j]
        r = dict(stratum)
        r.update(
            {
                "clone_id": names[k],
                "n_records": int(sizes[k]),
                "n_units": n_loci[k],
                "n_blocks": int(n_blk[k]) if n_blk is not None else np.nan,
                "statistic": ob["statistic"],
            }
        )
        c = by_code.get(j)
        if c is not None:
            nb = np.asarray(c.base, dtype=float)
            base[: len(nb), j] = nb
            p, n_used = _p_value(c, h, stopping)
            mu, sd = float(nb.mean()), float(nb.std(ddof=1)) if len(nb) > 1 else float("nan")
            r.update(
                {
                    "null_mean": mu,
                    "null_sd": sd,
                    "z": (ob["statistic"] - mu) / sd if sd > 0 else float("nan"),
                    "p_value": p,
                    "n_perm": int(n_used),
                    "n_exceed": int(min(c.g, h) if c.stop_at else c.g),
                    "n_drawn": int(c.n),
                    "n_ties": int(c.ties),
                    "p_plain": float((1 + (nb >= ob["statistic"] - _TIE * abs(ob["statistic"])).sum()) / (len(nb) + 1)),
                    "stop": c.stop,
                }
            )
        else:
            r.update(
                {
                    "null_mean": np.nan,
                    "null_sd": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "n_perm": 0,
                    "n_exceed": 0,
                    "n_drawn": 0,
                    "n_ties": 0,
                    "p_plain": np.nan,
                    "stop": "untestable",
                }
            )
        r["p_min"] = float(p_min[j])
        r["testable"] = bool(testable[j])
        s, cen = ob["size"], ob["centre"]
        if s >= 0:
            cy, cx = divmod(cen, pool.nx)
            mu_b = n_loci[k] * ob["total"] / pool.n
            r.update(
                {
                    "best_radius_um": win.radii_um[s],
                    "best_x_um": x0 + (cx + 0.5) * win.grid_um,
                    "best_y_um": y0 + (cy + 0.5) * win.grid_um,
                    "best_observed": ob["observed"],
                    "best_total": ob["total"],
                    "best_expected": mu_b,
                    "best_ratio": ob["observed"] / mu_b if mu_b > 0 else np.nan,
                }
            )
        else:
            r.update(
                {
                    "best_radius_um": np.nan,
                    "best_x_um": np.nan,
                    "best_y_um": np.nan,
                    "best_observed": 0,
                    "best_total": 0,
                    "best_expected": np.nan,
                    "best_ratio": np.nan,
                }
            )
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab["q_value"] = bh(tab["p_value"].to_numpy(dtype=float))
    tab["significant"] = tab["q_value"].to_numpy(dtype=float) <= alpha
    front = list(stratum) + [
        "clone_id",
        "n_records",
        "n_units",
        "n_blocks",
        "statistic",
        "null_mean",
        "null_sd",
        "z",
        "p_value",
        "q_value",
        "n_perm",
        "n_exceed",
        "n_drawn",
        "n_ties",
        "p_plain",
        "p_min",
        "testable",
        "significant",
        "stop",
    ]
    tab = tab[front + [c for c in tab.columns if c not in front]]
    return tab, base, info
