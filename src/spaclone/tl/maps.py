"""Smoothed share maps of a binary mark, map-level calibrated hotspots, and excess of a mark in given regions.

A share map is a kernel binary regression (Nadaraya–Watson): with Gaussian weights ``w_i(s)`` of the units around
pixel ``s``, the share is ``Σ w m / Σ w``. The numerator is a subset of the denominator and both use the same kernel,
so the map is not a ratio of two independent densities (Kelsall & Diggle 1998). Its precision is the Kish effective
number of units ``n_eff = (Σ w)² / Σ w²``. Against a per-unit expected probability ``p_i`` (a global share, or a
covariate-matched one such as the mutated share of the unit's isotype class: indirect standardisation) the
standardised residual is ``z = (Σ w m − Σ w p) / sqrt(Σ w² p (1 − p))``.

Pointwise thresholds give no map-level error control. :func:`hotspots` therefore compares ``z`` with the per-shuffle
map maximum (max-T, family-wise over the section) or with the per-shuffle maximum cluster mass, with the mark
shuffled among units (within covariate strata) or among whole clones, while positions stay fixed.
:func:`region_excess` tests regions found with one mark (e.g. class-switch hotspots) for an excess of another mark
(e.g. mutation) beyond what the regions' covariate mix predicts.

Clone-level marks. Switched, mutated and expanded are properties of a lineage: the loci of one clone share them and
sit together. Shuffling such a mark among loci treats a clone of 30 mutated loci as 30 independent coin flips, and a
single expanded clone then looks like a hotspot (in simulations the unit-level null rejected on 99% of null maps).
With ``blocks=<clone column>`` the default null therefore relabels whole clones: a clone swaps its marks with clones
of the same size and the same isotype mix, and inside a clone the marks are re-dealt among its loci of each isotype
(``block_design="matched"``). The statistic is then studentised pixel by pixel with the mean and standard deviation
of the shuffled maps (``standardise="permutation"``), because clone-level shuffles do not have the binomial
variance that the ``z`` formula assumes.

References
----------
Kelsall & Diggle 1995, doi:10.2307/3318678; Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128 (kernel binary
regression); Kish 1965, Survey Sampling, Wiley (effective sample size); Nichols & Holmes 2002, doi:10.1002/hbm.1058
(max-T and cluster-mass permutation thresholds); Bullmore et al. 1999, doi:10.1109/42.750253 (cluster mass);
Westfall & Young 1993, Resampling-Based Multiple Testing, Wiley (studentised max-T); Hazelton & Davies 2009,
doi:10.1002/bimj.200810495 (pointwise tolerance contours, for contrast); Hoehn et al. 2022,
doi:10.1371/journal.pcbi.1009885 (permutations that respect lineage structure).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    LowPowerWarning,
    RngLike,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    logger,
    mc_pvalue,
    rng_entropy,
)
from spaclone.tl.structures import Grid, Raster

__all__ = ["HotspotResult", "ShareMap", "hotspots", "region_excess", "share_map"]

NeffKernel = Literal["exact", "legacy"]
Correction = Literal["maxT", "cluster_mass"] | None
ThresholdRule = Literal["rank", "quantile"]
BlockDesign = Literal["matched", "free"]
Standardise = Literal["auto", "binomial", "permutation"]

#: Largest number of (shuffled map x eligible pixel) values kept in memory by the permutation-studentised max-T;
#: beyond it the shuffled maps are recomputed in a second pass (identical results, twice the smoothing time).
_STORE_LIMIT = 12_500_000


# ------------------------------------------------------------------------------------------------- input helpers


def _column(table: pd.DataFrame, spec: Any, what: str) -> np.ndarray:
    """A column name or an array with one value per row."""
    if isinstance(spec, str):
        if spec not in table.columns:
            raise KeyError(f"{what}: column {spec!r} not in the table")
        return table[spec].to_numpy()
    arr = np.asarray(spec)
    if arr.ndim != 1 or arr.shape[0] != len(table):
        raise ValueError(f"{what}: expected {len(table)} values, got shape {arr.shape}")
    return arr


def _binary(values: np.ndarray, what: str) -> np.ndarray:
    """0/1 floats from a bool or 0/1 column; raises on missing or other values."""
    v = pd.array(values)
    if pd.isna(v).any():
        raise ValueError(f"{what} has missing values; drop those units first")
    out = np.asarray(v, dtype=float)
    if not np.isin(out, (0.0, 1.0)).all():
        raise ValueError(f"{what} must be binary (bool or 0/1)")
    return out


def _codes(values: np.ndarray, what: str) -> np.ndarray:
    """Integer codes 0..k-1 in sorted order of the values (deterministic, independent of row order)."""
    codes, _ = pd.factorize(pd.Series(values), sort=True)
    if (codes < 0).any():
        raise ValueError(f"{what} has missing values")
    return codes.astype(np.int64)


def _strata(table: pd.DataFrame, within: str | Sequence[str] | None) -> np.ndarray | None:
    """Integer stratum codes (sorted order of the stratum values) or None."""
    if within is None:
        return None
    cols = [within] if isinstance(within, str) else list(within)
    for c in cols:
        if c not in table.columns:
            raise KeyError(f"within: column {c!r} not in the table")
    if len(cols) == 1:
        return _codes(table[cols[0]].to_numpy(), f"within column {cols[0]!r}")
    key = pd.MultiIndex.from_frame(table[cols].astype(object))
    codes, _ = pd.factorize(key, sort=True)
    if (codes < 0).any():
        raise ValueError(f"within columns {cols} have missing values")
    return codes.astype(np.int64)


def _groups(codes: np.ndarray) -> list[np.ndarray]:
    """Positions of each code in increasing code order; positions inside a group are increasing."""
    if not len(codes):
        return []
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)


class _Shuffler:
    """Random relabelling of a binary mark among units, positions fixed.

    Without blocks, the mark is permuted among the units of each stratum (strata in sorted order, one
    ``rng.permutation`` per stratum and draw; without strata one ``rng.permutation(n)`` per draw). With blocks, the
    block-level mark (constant within a block) is permuted among the blocks of each stratum and written to every
    unit of the receiving block.
    """

    def __init__(self, mark: np.ndarray, strata: np.ndarray | None, blocks: np.ndarray | None) -> None:
        self.mark = mark
        self.n = len(mark)
        self.single = strata is None and blocks is None
        st = np.zeros(self.n, dtype=np.int64) if strata is None else strata
        if blocks is None:
            self.blocks = None
            self.groups = _groups(st)
            return
        self.blocks = blocks
        nb = int(blocks.max()) + 1 if self.n else 0
        first = np.full(nb, -1, dtype=np.int64)
        first[blocks[::-1]] = np.arange(self.n)[::-1]
        bmark, bst = mark[first], st[first]
        if (mark != bmark[blocks]).any():
            raise ValueError("blocks: the mark varies inside a block; block shuffles need a block-level mark")
        if (st != bst[blocks]).any():
            raise ValueError("blocks: a stratum varies inside a block")
        self.block_mark = bmark
        self.groups = _groups(bst)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One shuffled copy of the mark."""
        if self.single:
            return self.mark[rng.permutation(self.n)]
        if self.blocks is None:
            out = self.mark.copy()
            for pos in self.groups:
                out[pos] = self.mark[pos][rng.permutation(len(pos))]
            return out
        bm = self.block_mark.copy()
        for pos in self.groups:
            bm[pos] = self.block_mark[pos][rng.permutation(len(pos))]
        return bm[self.blocks]

    @staticmethod
    def describe(within: Any, blocks: Any) -> dict[str, Any]:
        """The null design in words, for ``params``."""
        return {
            "design": "random labelling (Permute the mark, positions fixed)",
            "within": within,
            "blocks": blocks,
            "exchangeable": "blocks" if blocks is not None else "units",
        }


class _MatchedShuffler:
    """Clone-level relabelling that keeps every block's size and stratum mix (``block_design="matched"``).

    A block's *signature* is its number of units in each stratum. Blocks of one signature exchange their marks: in a
    draw the blocks of a signature group are permuted, and each receiving block takes its donor's marks stratum by
    stratum, dealt to its own units of that stratum in a random order. A block with a unique signature keeps its marks,
    re-dealt among its own units of each stratum. Neither the mark nor the stratum needs to be constant inside a block,
    and every stratum keeps its number of marked units in every draw (a covariate-matched expectation is unchanged).

    Draws use ``rng.random(n_blocks)`` (order of blocks inside signature groups) and then one ``rng.random`` value per
    unit of a (block, stratum) cell with several units (order of the units inside each such cell).
    """

    def __init__(self, mark: np.ndarray, strata: np.ndarray | None, blocks: np.ndarray) -> None:
        n = len(mark)
        self.mark = mark
        self.n = n
        st = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata, dtype=np.int64)
        bl = np.asarray(blocks, dtype=np.int64)
        order = np.lexsort((np.arange(n), st, bl))  # canonical order: block, stratum, row
        b_s, s_s = bl[order], st[order]
        new_cell = np.ones(n, dtype=bool)
        if n:
            new_cell[1:] = (b_s[1:] != b_s[:-1]) | (s_s[1:] != s_s[:-1])
        cell_of = np.cumsum(new_cell) - 1  # cell of every sorted position
        cell_start = np.flatnonzero(new_cell)
        cell_size = np.diff(np.r_[cell_start, n])
        cell_block = b_s[cell_start]
        cell_stratum = s_s[cell_start]
        n_blocks = int(bl.max()) + 1 if n else 0
        n_strata = int(st.max()) + 1 if n else 0
        comp = np.zeros((n_blocks, n_strata), dtype=np.int64)  # units per (block, stratum)
        comp[cell_block, cell_stratum] = cell_size
        present = comp.sum(1) > 0
        sig = np.full(n_blocks, -1, dtype=np.int64)
        if present.any():
            _, inv = np.unique(comp[present], axis=0, return_inverse=True)
            sig[present] = np.asarray(inv).ravel()
        self.order = order
        self.cell_of = cell_of
        self.cell_start = cell_start
        self.block_first_cell = np.full(n_blocks, -1, dtype=np.int64)
        self.block_first_cell[cell_block[::-1]] = np.arange(len(cell_start))[::-1]
        self.block_ids = np.flatnonzero(present)
        self.sig = sig[self.block_ids]
        self.canon = self.block_ids[np.lexsort((self.block_ids, self.sig))]  # blocks by signature, then id
        self.mark_sorted = mark[order]
        self.cell_index = np.arange(len(cell_start)) - self.block_first_cell[cell_block]  # rank of a cell in its block
        self.cell_block = cell_block
        # units whose mark can change: a mixed cell, or a cell whose slot differs between blocks of the signature
        marked = np.bincount(cell_of, weights=self.mark_sorted, minlength=len(cell_start))
        slot = sig[cell_block] * max(n_strata, 1) + cell_stratum
        lo = pd.Series(marked).groupby(slot).transform("min").to_numpy()
        hi = pd.Series(marked).groupby(slot).transform("max").to_numpy()
        movable_cell = (lo != hi) | ((marked > 0) & (marked < cell_size))
        mov = np.zeros(n, dtype=bool)
        mov[order] = movable_cell[cell_of]
        self.movable = mov
        multi = cell_size[cell_of] > 1
        self.multi_pos = np.flatnonzero(multi)  # sorted positions in cells of several units (grouped by cell)
        self.multi_cell = cell_of[self.multi_pos]
        self.n_signatures = int(len(np.unique(self.sig))) if len(self.sig) else 0
        self.n_exchangeable_blocks = int(
            (pd.Series(self.sig).map(pd.Series(self.sig).value_counts()) > 1).sum() if len(self.sig) else 0
        )

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One shuffled copy of the mark."""
        if not self.n:
            return self.mark.copy()
        key = rng.random(len(self.block_ids))
        shuffled = self.block_ids[np.lexsort((key, self.sig))]  # same signature groups, random order inside each
        donor = np.arange(len(self.block_first_cell))
        donor[self.canon] = shuffled
        cell_donor = self.block_first_cell[donor[self.cell_block]] + self.cell_index
        offset = np.zeros(self.n, dtype=np.int64)
        if len(self.multi_pos):  # each multi-unit cell's positions in random order
            order = np.lexsort((rng.random(len(self.multi_pos)), self.multi_cell))
            offset[self.multi_pos] = self.multi_pos[order] - self.cell_start[self.multi_cell]
        src = self.cell_start[cell_donor[self.cell_of]] + offset
        out = np.empty(self.n, dtype=self.mark.dtype)
        out[self.order] = self.mark_sorted[src]
        return out

    def describe(self, within: Any, blocks: Any) -> dict[str, Any]:
        """The null design in words, for ``params``."""
        return {
            "design": "matched block relabelling (positions fixed)",
            "within": within,
            "blocks": blocks,
            "exchangeable": "blocks of the same size and stratum mix; units of a block within its strata",
            "n_signatures": self.n_signatures,
            "n_exchangeable_blocks": self.n_exchangeable_blocks,
            "movable_units": int(self.movable.sum()),
        }


#: Units of one block closer than this are parts of one cell (records, spots), not separate cells.
_SUBCELLULAR_UM = 5.0


def _check_cell_scale(table: pd.DataFrame, blocks: str, x_key: str, y_key: str) -> int:
    """Log a warning when units of one block lie within ``_SUBCELLULAR_UM`` of each other; return how many do."""
    from scipy.spatial import cKDTree

    if x_key not in table.columns or y_key not in table.columns or len(table) < 2:
        return 0
    xy = table[[x_key, y_key]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        return 0
    blk = pd.factorize(table[blocks])[0]
    pairs = cKDTree(xy).query_pairs(_SUBCELLULAR_UM, output_type="ndarray")
    if not len(pairs):
        return 0
    pairs = pairs[blk[pairs[:, 0]] == blk[pairs[:, 1]]]
    n = int(len(np.unique(pairs))) if len(pairs) else 0
    if n:
        logger.warning(
            "%d units lie within %g um of another unit of the same block: they look like records or spots of one cell. "
            "Clone-level shuffles assume cell-scale units (loci from pp.merge_loci); re-dealing marks among the "
            "records of one cell makes the null too narrow when the mark varies inside a clone",
            n,
            _SUBCELLULAR_UM,
        )
    return n


def _make_shuffler(
    mark: np.ndarray,
    table: pd.DataFrame,
    *,
    within: Any,
    blocks: str | None,
    block_design: str,
) -> tuple[_Shuffler | _MatchedShuffler, dict[str, Any]]:
    """The shuffler of a design and its description."""
    st = _strata(table, within)
    if blocks is None:
        sh: _Shuffler | _MatchedShuffler = _Shuffler(mark, st, None)
        return sh, sh.describe(within, None)
    bl = _codes(_column(table, blocks, "blocks"), f"blocks column {blocks!r}")
    if block_design == "free":
        sh = _Shuffler(mark, st, bl)
        return sh, {**sh.describe(within, blocks), "block_design": "free"}
    sh = _MatchedShuffler(mark, st, bl)
    return sh, {**sh.describe(within, blocks), "block_design": "matched"}


def _critical(null: np.ndarray, alpha: float, rule: ThresholdRule) -> float:
    """Critical value of per-shuffle maxima: an observed statistic is significant when it exceeds it.

    ``"rank"``: the k-th largest of the B finite draws, k = floor(α (B + 1)); a statistic exceeds it exactly when its
    Monte-Carlo p-value ``(1 + #{draws >= t}) / (B + 1)`` is at most α, so the family-wise error is at most α.
    ``"quantile"``: ``numpy.quantile(draws, 1 - α)`` (linear interpolation; the 2026-10 archive's rule), which can
    exceed α by up to ``1 / (B + 1)``.
    """
    v = np.asarray(null, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.inf
    if rule == "quantile":
        return float(np.quantile(v, 1 - alpha))
    if rule != "rank":
        raise ValueError("threshold_rule must be 'rank' or 'quantile'")
    k = int(np.floor(alpha * (len(v) + 1) + 1e-9))
    if k < 1:
        warnings.warn(
            f"{len(v)} shuffles cannot reach alpha = {alpha:g} (need at least {int(np.ceil(1 / alpha)) - 1}); "
            "nothing can be significant",
            LowPowerWarning,
            stacklevel=3,
        )
        return np.inf
    return float(np.sort(v)[len(v) - k])


# --------------------------------------------------------------------------------------------- smoothing engine


@dataclass
class _Units:
    """Units on the grid: rows kept, pixel indices, mark, expected probabilities."""

    grid: Grid
    rows: np.ndarray  # positions (in the input table) of the units on the grid
    i: np.ndarray
    j: np.ndarray
    mark: np.ndarray
    p: np.ndarray
    kind: str  # "global", "match" or "given"
    classes: np.ndarray | None  # match_by codes (kind "match")
    n_off_grid: int


def _units(
    table: pd.DataFrame,
    *,
    mark: Any,
    grid: Grid | None,
    tissue: Any,
    pixel_um: float,
    expected: Any,
    match_by: str | None,
    x_key: str,
    y_key: str,
) -> _Units:
    for c in (x_key, y_key):
        if c not in table.columns:
            raise KeyError(f"coordinate column {c!r} not in the table; pass x_key / y_key")
    if expected is not None and match_by is not None:
        raise ValueError("give either expected (per-unit probabilities) or match_by (a covariate column), not both")
    m_all = _binary(_column(table, mark, "mark"), "mark")
    x, y = table[x_key].to_numpy(dtype=float), table[y_key].to_numpy(dtype=float)
    if grid is None and isinstance(tissue, Raster):
        grid = tissue.grid
    if grid is None:
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            raise ValueError("no unit has finite coordinates")
        grid = Grid.from_points(x[ok], y[ok], pixel_um=pixel_um)
    i, j, inside = grid.index(x, y)
    rows = np.flatnonzero(inside)
    n_off = int(len(table) - len(rows))
    if n_off:
        logger.warning("%d of %d units fall off the grid and are left out of the map", n_off, len(table))
    if not len(rows):
        raise ValueError("no unit falls on the grid")
    m = m_all[rows]
    classes = None
    if expected is not None:
        p = np.asarray(_column(table, expected, "expected"), dtype=float)[rows]
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise ValueError("expected probabilities must lie in [0, 1]")
        kind = "given"
    elif match_by is not None:
        classes = _codes(_column(table, match_by, "match_by")[rows], f"match_by column {match_by!r}")
        p = (np.bincount(classes, weights=m) / np.bincount(classes))[classes]
        kind = "match"
    else:
        p = np.full(len(rows), m.mean())
        kind = "global"
    return _Units(grid, rows, i[rows], j[rows], m, p, kind, classes, n_off)


class _Engine:
    """Gaussian smoothing on the grid: Σw (n1), Σw² (n2), n_eff, expectation and z."""

    def __init__(self, u: _Units, *, sigma_um: float, neff_kernel: NeffKernel, truncate: float = 4.0) -> None:
        if not sigma_um > 0:
            raise ValueError("sigma_um must be positive")
        if neff_kernel not in ("exact", "legacy"):
            raise ValueError("neff_kernel must be 'exact' or 'legacy'")
        self.u = u
        self.sg = sigma_um / u.grid.pixel_um
        self.neff_kernel = neff_kernel
        self.truncate = truncate
        radius = int(truncate * self.sg + 0.5)  # scipy's window
        x = np.arange(-radius, radius + 1)
        phi = np.exp(-0.5 / self.sg**2 * x**2)
        phi = phi / phi.sum()
        self.w2 = np.ascontiguousarray((phi**2)[::-1])
        ones = np.ones(len(u.mark))
        self.n1 = self.s1(ones)
        self.n2 = self.s2(ones)
        self.neff = np.where(self.n2 > 0, self.n1**2 / np.maximum(self.n2, 1e-300), 0.0)

    def raster(self, v: np.ndarray) -> np.ndarray:
        g = np.zeros(self.u.grid.shape)
        np.add.at(g, (self.u.i, self.u.j), v)
        return g

    def s1(self, v: np.ndarray) -> np.ndarray:
        """Σ_i w_i(s) v_i (scipy's normalised discrete Gaussian, window ±4 σ, zero outside the grid)."""
        return ndi.gaussian_filter(self.raster(v), self.sg, mode="constant", truncate=self.truncate)

    def s2(self, v: np.ndarray) -> np.ndarray:
        """Σ_i w_i(s)² v_i."""
        g = self.raster(v)
        if self.neff_kernel == "legacy":
            # 2026-10 archive (T3 / T7): Gaussian of sd sigma/sqrt(2) divided by 4 pi sigma^2 with scipy's default
            # truncation, so its window reaches only 2.83 sigma (n_eff inflated where the nearest units are far)
            return ndi.gaussian_filter(g, self.sg / np.sqrt(2), mode="constant") / (4 * np.pi * self.sg**2)
        g = ndi.correlate1d(g, self.w2, axis=0, mode="constant")
        return ndi.correlate1d(g, self.w2, axis=1, mode="constant")

    def expectation(self, p: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Σ w p and Σ w² p (1 − p) on the full grid."""
        if kind == "global":
            p0 = float(p[0]) if len(p) else 0.0
            return p0 * self.n1, p0 * (1 - p0) * self.n2
        return self.s1(p), self.s2(p * (1 - p))


def _z(s1: np.ndarray, e: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(Σ w m − Σ w p) / sqrt(Σ w² p (1 − p)); NaN where the variance is 0 (no information)."""
    out = np.full(np.shape(s1), np.nan)
    ok = v > 0
    out[ok] = (s1[ok] - e[ok]) / np.sqrt(v[ok])
    return out


# ------------------------------------------------------------------------------------------------------ share map


@dataclass(frozen=True, eq=False)
class ShareMap:
    """A smoothed share map and its standardised residuals.

    Attributes
    ----------
    grid
        The pixel grid (arrays are indexed ``[i, j]``, first axis x).
    share
        Smoothed share ``Σ w m / Σ w`` (NaN where no unit has weight).
    expected
        Smoothed expected share ``Σ w p / Σ w``.
    z
        Standardised residual ``(Σ w m − Σ w p) / sqrt(Σ w² p (1 − p))`` (NaN where the variance is 0).
    n_eff
        Kish effective number of units ``(Σ w)² / Σ w²``.
    eligible
        Pixels used for inference: tissue and ``n_eff >= min_neff``.
    p0
        Share of marked units among the units on the grid.
    n_units, n_marked
        Units on the grid and marked units among them.
    params
        Arguments that define the map.
    tissue
        The tissue mask the map was computed on (``None`` when none was given); plots draw the section outline and
        the map over it.
    """

    grid: Grid
    share: np.ndarray
    expected: np.ndarray
    z: np.ndarray
    n_eff: np.ndarray
    eligible: np.ndarray
    p0: float
    n_units: int
    n_marked: int
    params: dict[str, Any] = field(default_factory=dict)
    tissue: np.ndarray | None = None

    _LAYERS = ("share", "expected", "z", "n_eff", "eligible")

    def __repr__(self) -> str:
        return (
            f"ShareMap({self.n_marked}/{self.n_units} marked units, p0={self.p0:.4g}, sigma "
            f"{self.params.get('sigma_um')} µm; {int(self.eligible.sum())} eligible pixels of {self.grid.pixel_um:g} µm)"
        )

    def raster(self, layer: str = "share") -> Raster:
        """One layer as a :class:`~spaclone.tl.Raster`.

        Parameters
        ----------
        layer
            ``"share"``, ``"expected"``, ``"z"``, ``"n_eff"`` or ``"eligible"``.

        Returns
        -------
        Raster
            The layer on the map's grid.
        """
        if layer not in self._LAYERS:
            raise ValueError(f"layer must be one of {self._LAYERS}")
        return Raster(getattr(self, layer), self.grid, name=layer)

    def sample(self, x_um: Any, y_um: Any, *, layer: str = "share", fill: Any = np.nan) -> np.ndarray:
        """Values of a layer at the pixels holding the points.

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm.
        layer
            Layer name, as in :meth:`raster`.
        fill
            Value for points off the grid.

        Returns
        -------
        numpy.ndarray
            One value per point.
        """
        return self.raster(layer).sample(x_um, y_um, fill=fill)


def _tissue_array(tissue: Raster | np.ndarray | None, grid: Grid) -> np.ndarray:
    if tissue is None:
        return np.ones(grid.shape, dtype=bool)
    if isinstance(tissue, Raster):
        if tissue.grid != grid:
            raise ValueError("tissue raster is on a different grid than the map")
        return tissue.values.astype(bool)
    arr = np.asarray(tissue, dtype=bool)
    if arr.shape != grid.shape:
        raise ValueError(f"tissue mask has shape {arr.shape}, the grid has {grid.shape}")
    return arr


def _build(
    table: pd.DataFrame,
    *,
    mark: Any,
    sigma_um: float,
    pixel_um: float,
    grid: Grid | None,
    tissue: Any,
    expected: Any,
    match_by: str | None,
    min_neff: float,
    neff_kernel: NeffKernel,
    x_key: str,
    y_key: str,
) -> tuple[_Units, _Engine, ShareMap, np.ndarray, np.ndarray]:
    u = _units(
        table,
        mark=mark,
        grid=grid,
        tissue=tissue,
        pixel_um=pixel_um,
        expected=expected,
        match_by=match_by,
        x_key=x_key,
        y_key=y_key,
    )
    eng = _Engine(u, sigma_um=sigma_um, neff_kernel=neff_kernel)
    e, v = eng.expectation(u.p, u.kind)
    if not (v > 0).any():
        raise ValueError("the expected variance is 0 everywhere (the mark is constant within every class)")
    s1 = eng.s1(u.mark)
    pos = eng.n1 > 1e-12
    n1 = np.maximum(eng.n1, 1e-300)
    share = np.where(pos, s1 / n1, np.nan)
    exp_share = np.where(pos, e / n1, np.nan)
    z = _z(s1, e, v)
    tis = _tissue_array(tissue, u.grid)
    elig = tis & (eng.neff >= min_neff)
    params = {
        "sigma_um": sigma_um,
        "pixel_um": u.grid.pixel_um,
        "grid": u.grid,
        "min_neff": min_neff,
        "neff_kernel": neff_kernel,
        "expected": u.kind if expected is None else "given per unit",
        "match_by": match_by,
        "n_off_grid": u.n_off_grid,
    }
    sm = ShareMap(
        u.grid, share, exp_share, z, eng.neff, elig, float(u.mark.mean()), len(u.mark), int(u.mark.sum()), params,
        tissue=tis if tissue is not None else None,
    )  # fmt: skip
    return u, eng, sm, e, v


def share_map(
    table: pd.DataFrame,
    *,
    mark: str | Any,
    sigma_um: float = 50.0,
    pixel_um: float = 10.0,
    tissue: Raster | np.ndarray | None = None,
    expected: str | Any | None = None,
    match_by: str | None = None,
    min_neff: float = 10.0,
    grid: Grid | None = None,
    neff_kernel: NeffKernel = "exact",
    x_key: str = S.X,
    y_key: str = S.Y,
) -> ShareMap:
    """Smoothed share of a binary mark over units, with effective sample size and standardised residuals.

    Parameters
    ----------
    table
        One row per unit (normally loci: one inferred cell each, so a cell spread over several spots counts once).
    mark
        Binary column (or array): switched, mutated, expanded, ...
    sigma_um
        Gaussian standard deviation in µm, the same for numerator and denominator.
    pixel_um
        Pixel side when neither ``grid`` nor a ``tissue`` raster gives the grid (the grid then spans the units).
    tissue
        Boolean mask of the tissue (a :class:`~spaclone.tl.Raster`, whose grid the map then uses, or an array on
        ``grid``); eligible pixels are tissue pixels with ``n_eff >= min_neff``.
    expected
        Column (or array) of each unit's expected probability of the mark, e.g. from a model; the residual is then
        ``Σ w (m − p) / sqrt(Σ w² p (1 − p))``.
    match_by
        Categorical column; each unit's expected probability is the share of marked units in its class
        (covariate-matched expectation, e.g. the mutated share of the unit's isotype). Exclusive with ``expected``.
        Without either, every unit's expectation is the global share ``p0``.
    min_neff
        Minimum effective number of units for a pixel to be eligible (low-``n_eff`` edge pixels otherwise dominate
        map maxima).
    grid
        Grid of the map, e.g. the grid of the tissue masks; units off the grid are left out (logged).
    neff_kernel
        ``"exact"``: ``Σ w²`` with the squared weights of the same kernel, truncated at the same radius (the Kish
        definition). ``"legacy"``: the 2026-10 archive's approximation (Gaussian of sd ``σ/√2`` with scipy's
        default window of 2.83 σ), kept only to reproduce archived maps; it inflates ``n_eff`` where the nearest
        units are far.
    x_key, y_key
        Coordinate columns (µm).

    Returns
    -------
    ShareMap
        Frozen object with the grid, ``share``, ``expected``, ``z``, ``n_eff``, ``eligible`` and ``.sample()``.

    Notes
    -----
    The kernel is scipy's normalised discrete Gaussian (window ±4 σ, zero outside the grid). The share is computed
    wherever units have weight; use ``eligible`` before reading or thresholding it. ``z`` is a standardised
    residual, not a calibrated test statistic: use :func:`hotspots` for inference. For a Gaussian kernel and unit
    density λ, ``n_eff ≈ λ π (2σ)²`` (the units within 2σ); it halves at tissue edges.

    References
    ----------
    Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Kish 1965, Survey Sampling, Wiley; Davies et al. 2018,
    doi:10.1002/sim.7577.
    """
    _, _, sm, _, _ = _build(
        table,
        mark=mark,
        sigma_um=sigma_um,
        pixel_um=pixel_um,
        grid=grid,
        tissue=tissue,
        expected=expected,
        match_by=match_by,
        min_neff=min_neff,
        neff_kernel=neff_kernel,
        x_key=x_key,
        y_key=y_key,
    )
    return sm


# ------------------------------------------------------------------------------------------------------- hotspots


@dataclass(frozen=True, repr=False)
class HotspotResult(TestResult):
    """Result of :func:`hotspots`: a :class:`~spaclone.TestResult` plus the map, the label raster and thresholds.

    Attributes
    ----------
    map
        The observed :class:`ShareMap`.
    labels
        Integer :class:`~spaclone.tl.Raster`, ``k`` = pixels of hotspot ``k`` in ``table`` (0 = none).
    threshold
        ``{"hot": z, "cold": z}`` map-level critical values (cluster-mass critical values for
        ``correction="cluster_mass"``, the pointwise p-value level for ``correction=None``).
    z_map
        The standardised map that was thresholded, NaN outside the pixels used for inference: the binomial residual
        ``map.z`` (``standardise="binomial"``) or the permutation-studentised excess ``(S − mean) / sd`` over the
        observed and shuffled maps (``standardise="permutation"``).
    """

    map: ShareMap | None = None
    labels: Raster | None = None
    threshold: dict[str, float] = field(default_factory=dict)
    z_map: Raster | None = None


def _label_by_area(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components numbered by decreasing area (ties: raster order)."""
    lab, n = ndi.label(mask)
    if not n:
        return lab, 0
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = np.argsort(-area, kind="stable")
    new = np.zeros(n + 1, dtype=lab.dtype)
    new[order + 1] = np.arange(1, n + 1)
    return new[lab], n


def _max_mass(t: np.ndarray, u: float) -> float:
    """Largest cluster mass Σ (t − u) over 4-connected clusters with t > u (0 if there is none)."""
    supra = t > u
    if not supra.any():
        return 0.0
    lab, n = ndi.label(supra)
    return float(np.max(ndi.sum(t - u, lab, index=np.arange(1, n + 1))))


def _block_neff(eng: _Engine, u: _Units, movable: np.ndarray, block_codes: np.ndarray | None) -> np.ndarray:
    """Effective number of blocks whose marks the null can move, per pixel: (Σ_b W_b)² / Σ_b W_b².

    ``W_b(s)`` is the kernel weight of block ``b``'s movable units at pixel ``s``; ``Σ_b W_b²`` is approximated by
    placing each block's movable units at their centroid (exact for single-unit blocks and for compact blocks).
    """
    mv = movable.astype(float)
    n1 = eng.s1(mv)
    if block_codes is None:
        n2 = eng.s2(mv)
    else:
        b = block_codes[movable]
        if not len(b):
            return np.zeros(u.grid.shape)
        cnt = np.bincount(b).astype(float)
        keep = cnt > 0
        ci = np.rint(np.bincount(b, weights=u.i[movable].astype(float))[keep] / cnt[keep]).astype(np.int64)
        cj = np.rint(np.bincount(b, weights=u.j[movable].astype(float))[keep] / cnt[keep]).astype(np.int64)
        g = np.zeros(u.grid.shape)
        np.add.at(g, (ci, cj), cnt[keep] ** 2)
        if eng.neff_kernel == "legacy":
            n2 = ndi.gaussian_filter(g, eng.sg / np.sqrt(2), mode="constant") / (4 * np.pi * eng.sg**2)
        else:
            g = ndi.correlate1d(g, eng.w2, axis=0, mode="constant")
            n2 = ndi.correlate1d(g, eng.w2, axis=1, mode="constant")
    return np.where(n2 > 1e-300, n1**2 / np.maximum(n2, 1e-300), 0.0)


def _studentised_null(
    eng: _Engine,
    u: _Units,
    elig: np.ndarray,
    draw: Any,
    n_perm: int,
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Pass 1 of the permutation-studentised max-T.

    Draws ``n_perm`` shuffled marks (``draw(b)``), packs them into ``packed`` and returns the per-pixel mean and
    standard deviation (ddof = 1) of ``S = Σ w m`` over the observed and the shuffled maps (``n_perm + 1`` maps), plus
    the shuffled ``S`` on the eligible pixels when they fit in memory (else ``None``: pass 2 recomputes them).
    """
    s0 = eng.s1(u.mark)[elig]
    acc1 = np.zeros_like(s0)
    acc2 = np.zeros_like(s0)
    store = np.empty((n_perm, len(s0))) if n_perm * len(s0) <= _STORE_LIMIT else None
    for b in range(n_perm):
        mb = draw(b)
        packed[b] = np.packbits(mb > 0.5)
        sb = eng.s1(mb)[elig]
        if store is not None:
            store[b] = sb
        d = sb - s0
        acc1 += d
        acc2 += d * d
    m = n_perm + 1
    mean = s0 + acc1 / m
    var = np.maximum(acc2 - acc1 * acc1 / m, 0.0) / max(m - 1, 1)
    return mean, np.sqrt(var), store


def hotspots(
    table: pd.DataFrame,
    *,
    mark: str | Any,
    sigma_um: float = 50.0,
    pixel_um: float = 10.0,
    tissue: Raster | np.ndarray | None = None,
    expected: str | Any | None = None,
    match_by: str | None = None,
    min_neff: float = 10.0,
    n_perm: int = 1000,
    within: str | Sequence[str] | None = None,
    blocks: str | None = None,
    block_design: BlockDesign = "matched",
    standardise: Standardise = "auto",
    min_blocks: float = 5.0,
    correction: Correction = "maxT",
    alpha: float = 0.05,
    alternative: Alternative = "greater",
    cluster_z: float = 3.0,
    threshold_rule: ThresholdRule = "rank",
    grid: Grid | None = None,
    neff_kernel: NeffKernel = "exact",
    null_marks: Any = None,
    min_units: int = 0,
    x_key: str = S.X,
    y_key: str = S.Y,
    rng: RngLike = None,
) -> HotspotResult:
    """Hotspots of a binary mark with family-wise error control over the whole map.

    The observed :func:`share_map` is standardised; a hotspot is a 4-connected set of eligible pixels whose
    standardised value exceeds the map-level critical value of the per-shuffle map maximum (``"maxT"``), or a cluster
    of pixels above ``cluster_z`` whose mass exceeds the critical value of the per-shuffle maximum cluster mass
    (``"cluster_mass"``).

    Parameters
    ----------
    table
        One row per unit: loci (one inferred cell each, :func:`spaclone.pp.merge_loci`), with coordinates and the
        mark. Several rows per cell (molecules, spots) make one cell count several times.
    mark, sigma_um, pixel_um, tissue, expected, match_by, min_neff, grid, neff_kernel, x_key, y_key
        As in :func:`share_map`.
    n_perm
        Number of shuffled maps; the smallest p-value is ``1 / (n_perm + 1)``.
    within
        Stratum column(s): marks move only between units of the same stratum (e.g. isotype class). Defaults to
        ``match_by``, so the null keeps the covariate-matched expectation (with ``block_design="free"`` and
        ``blocks``, no default: see there).
    blocks
        Column of exchangeable blocks, normally the clone id, for marks that belong to a lineage (mutated, switched).
        ``None``: every unit is exchangeable on its own (valid only when units of one clone carry independent marks,
        e.g. one unit per clone).
    block_design
        With ``blocks``: ``"matched"`` (default) — a block exchanges its marks only with blocks of the same size and
        the same number of units in every stratum, and inside a block the marks are re-dealt among its units of each
        stratum; neither the mark nor the strata need to be constant inside a block, and every stratum keeps its
        number of marked units. ``"free"`` — the 1.0 design: a block-level mark (constant inside each block) is
        permuted among all blocks of a stratum whatever their size (strata constant inside blocks; without
        ``within`` whole clones move across classes and a ``match_by`` expectation is recomputed on every map). It is
        anti-conservative when the mark depends on clone size or class (simulations: 11% family-wise error at
        α = 0.05) and is kept to reproduce 1.0 results.
    standardise
        ``"binomial"``: ``z = (Σ w m − Σ w p) / sqrt(Σ w² p (1 − p))`` (the 2026-10 archive; exact standardisation
        when units are exchangeable one by one). ``"permutation"``: ``z = (S − mean) / sd`` per pixel with
        ``S = Σ w m`` and the mean and standard deviation taken over the observed and the ``n_perm`` shuffled maps,
        so every pixel has unit null variance whatever the clone structure (studentised max-T; pixels where the null
        cannot move the mark have no variance and are left out). ``"auto"`` (default): ``"permutation"`` for the
        matched block design, ``"binomial"`` otherwise.
    min_blocks
        ``standardise="permutation"`` only: a pixel is also eligible only when the effective number of blocks whose
        marks the null can move, ``(Σ_b W_b)² / Σ_b W_b²`` with ``W_b`` the kernel weight of block ``b``, reaches
        this value (so a single rarely marked clone cannot set the map maximum, as ``min_neff`` does for units).
    correction
        ``"maxT"`` (peak height), ``"cluster_mass"`` (cluster extent and height above ``cluster_z``) or ``None``
        (pointwise Monte-Carlo p <= α per pixel: no map-level error control, display only).
    alpha
        Family-wise error rate of the map.
    alternative
        ``"greater"`` (hotspots), ``"less"`` (cold spots) or ``"two-sided"`` (both, α/2 each).
    cluster_z
        Cluster-forming threshold on the standardised map for ``"cluster_mass"``.
    threshold_rule
        ``"rank"``: a pixel (cluster) is called when its map-level Monte-Carlo p-value is at most α, i.e. its
        statistic exceeds the ``floor(α (n_perm + 1))``-th largest shuffled maximum (exact family-wise error).
        ``"quantile"``: it exceeds ``numpy.quantile`` of the shuffled maxima at ``1 − α`` (the 2026-10 archive's
        rule; up to ``1 / (n_perm + 1)`` too liberal).
    null_marks
        Optional ``(n_perm, len(table))`` array of shuffled marks (one null data set per row, in the row order of
        ``table``), used instead of the built-in shuffles: a custom design or a replayed archive null.
    min_units
        Smallest number of units inside a hotspot. A connected set of pixels above the threshold that holds fewer
        units is dropped and counted in ``summary["n_dropped_min_units"]``. Such sets occur next to a strong
        hotspot, where the kernel alone lifts pixels that hold no unit of their own. Default 0: every set is kept,
        as in 1.0. Dropping sets never adds a hotspot, so the family-wise error stays at most α.
    rng
        Seed or generator; shuffles are drawn from it in order (unit design: one ``permutation`` per stratum and map;
        matched block design: ``random(n_blocks)`` then ``random(n_units)`` per map).

    Returns
    -------
    HotspotResult
        ``table``: one row per hotspot (``hotspot`` 1 = largest; ``direction``, ``n_pixels``, ``area_um2``,
        ``x_um``, ``y_um`` (centroid), ``peak_z``, ``statistic`` (peak z, cluster mass, or smallest pointwise p),
        ``units``, ``marked``, ``share``, ``expected`` (Σ p of its units), ``ratio`` (marked / expected),
        ``null_mean``, ``null_sd`` (marked units inside under the shuffles), ``p_value`` (map-level: share of
        shuffled maps whose maximum reaches the hotspot's statistic), ``p_count_naive``, ``n_perm``). ``null``:
        per-shuffle maximum of ``z`` (hot) and of ``−z`` (cold), or maximum cluster masses. ``labels``,
        ``threshold``, ``z_map`` and ``map`` as attributes; thresholds also in ``summary``.

    Notes
    -----
    Null model (``blocks=None``): random labelling of units. Unit positions, the eligible pixels and the multiset of
    marks in each ``within`` stratum are fixed; every assignment of the marks to the units of a stratum is equally
    likely.

    Null model (``blocks``, ``block_design="matched"``): random relabelling of whole clones. Positions, strata and
    eligible pixels are fixed. Clones are grouped by signature (number of units in each stratum); within a signature
    group every assignment of the clones' mark profiles to the clones is equally likely, and within a clone every
    assignment of the marks of a stratum to the clone's units of that stratum is equally likely. Each stratum keeps
    its number of marked units; a clone keeps its number of marked units of each stratum unless it trades them with
    a clone of the same signature. A clone whose signature is unique is never traded: one expanded mutated clone is
    one observation, not a hotspot. The null holds when the mark does not depend on where a clone sits once its size
    and its stratum mix are given; the mark may depend on clone size and stratum (expanded and switched clones are
    mutated more often) and may vary inside a clone.

    Null model (``block_design="free"``): block-level marks permuted among the blocks of each ``within`` stratum
    (the 1.0 design), see ``block_design``.

    Under the stated null the share of maps with any hotspot is at most α (plus Monte-Carlo error with
    ``threshold_rule="quantile"``); with ``standardise="permutation"`` the observed map enters the per-pixel mean and
    standard deviation like every shuffled map, so the observed and the shuffled maxima are exchangeable.

    A mark that is a function of clone size (e.g. ``expanded``) cannot be tested by relabelling: the matched design
    then moves nothing and no pixel is eligible (warning). Use :func:`spaclone.tl.clone_dispersion` or
    :func:`spaclone.tl.clonal_scan` (random placement of clones) instead.

    The map-level p-value supports "a hotspot exists here", not its exact boundary. ``p_count_naive`` (marked units
    inside the hotspot vs the same shuffles) is descriptive: the region was selected on these data, so it is not a
    valid test. To test regions found with one mark for an excess of another, use :func:`region_excess`.

    References
    ----------
    Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Nichols & Holmes 2002, doi:10.1002/hbm.1058; Bullmore et
    al. 1999, doi:10.1109/42.750253; Westfall & Young 1993, Resampling-Based Multiple Testing, Wiley; Davison &
    Hinkley 1997, doi:10.1017/CBO9780511802843; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Hoehn et al. 2022,
    doi:10.1371/journal.pcbi.1009885.
    """
    check_alternative(alternative)
    if correction not in ("maxT", "cluster_mass", None):
        raise ValueError("correction must be 'maxT', 'cluster_mass' or None")
    if threshold_rule not in ("rank", "quantile"):
        raise ValueError("threshold_rule must be 'rank' or 'quantile'")
    if block_design not in ("matched", "free"):
        raise ValueError("block_design must be 'matched' or 'free'")
    if standardise not in ("auto", "binomial", "permutation"):
        raise ValueError("standardise must be 'auto', 'binomial' or 'permutation'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if min_units < 0:
        raise ValueError("min_units must be >= 0")
    gen = as_rng(rng)
    u, eng, sm, e_obs, v_obs = _build(
        table,
        mark=mark,
        sigma_um=sigma_um,
        pixel_um=pixel_um,
        grid=grid,
        tissue=tissue,
        expected=expected,
        match_by=match_by,
        min_neff=min_neff,
        neff_kernel=neff_kernel,
        x_key=x_key,
        y_key=y_key,
    )
    elig = sm.eligible
    if not elig.any():
        raise ValueError("no eligible pixel (lower min_neff, widen sigma_um or check the tissue mask)")
    matched = blocks is not None and block_design == "matched"
    if within is None and match_by is not None and (blocks is None or matched):
        within = match_by
    n_subcell = _check_cell_scale(table.iloc[u.rows], blocks, x_key, y_key) if (matched and null_marks is None) else 0
    if standardise == "auto":
        standardise = "permutation" if (matched and null_marks is None) else "binomial"
    sub = table.iloc[u.rows] if (within is not None or blocks is not None) else table
    shuffler: _Shuffler | _MatchedShuffler | None
    if null_marks is not None:
        nm = np.asarray(null_marks, dtype=float)
        if nm.ndim != 2 or nm.shape[1] != len(table):
            raise ValueError(f"null_marks must have shape (n_perm, {len(table)})")
        nm = nm[:, u.rows]
        n_perm = nm.shape[0]
        shuffler = None
        design: dict[str, Any] = {"design": "user-supplied null marks", "n_perm": n_perm}
    else:
        if n_perm < 1:
            raise ValueError("n_perm must be >= 1")
        shuffler, design = _make_shuffler(u.mark, sub, within=within, blocks=blocks, block_design=block_design)
    sides = {"greater": ("hot",), "less": ("cold",), "two-sided": ("hot", "cold")}[alternative]
    a_side = alpha / len(sides)
    sign = {"hot": 1.0, "cold": -1.0}
    null = np.empty((n_perm, len(sides)))
    exceed = {s: np.zeros(int(elig.sum()), dtype=np.int64) for s in sides} if correction is None else None
    packed = np.empty((n_perm, (len(u.mark) + 7) // 8), dtype=np.uint8)
    tfull = np.full(u.grid.shape, -np.inf)
    n_marked_obs = u.mark.sum()

    def draw(b: int) -> np.ndarray:
        return nm[b] if shuffler is None else shuffler.draw(gen)

    if standardise == "binomial":
        z_obs_e = sm.z[elig]
        e_e, v_e = e_obs[elig], v_obs[elig]
        tot_obs = np.bincount(u.classes, weights=u.mark) if u.kind == "match" else None
        for b in range(n_perm):
            mb = draw(b)
            packed[b] = np.packbits(mb > 0.5)
            e_b, v_b = e_e, v_e
            if u.kind == "global" and mb.sum() != n_marked_obs:
                p0b = mb.mean()
                e_b, v_b = (p0b * eng.n1)[elig], (p0b * (1 - p0b) * eng.n2)[elig]
            elif u.kind == "match" and not np.array_equal(np.bincount(u.classes, weights=mb), tot_obs):
                pb = (np.bincount(u.classes, weights=mb) / np.bincount(u.classes))[u.classes]
                ef, vf = eng.expectation(pb, "match")
                e_b, v_b = ef[elig], vf[elig]
            zb = _z(eng.s1(mb)[elig], e_b, v_b)
            for k, s in enumerate(sides):
                t = sign[s] * zb
                fin = np.isfinite(t)
                if correction == "cluster_mass":
                    tfull[elig] = np.where(fin, t, -np.inf)
                    null[b, k] = _max_mass(tfull, cluster_z)
                else:
                    null[b, k] = np.max(t[fin]) if fin.any() else -np.inf
                    if exceed is not None:
                        exceed[s] += fin & (t >= sign[s] * z_obs_e - 1e-12)
        z_obs_full = np.where(elig & np.isfinite(sm.z), sm.z, np.nan)
        stud: dict[str, Any] = {}
    else:
        mean, sd, store = _studentised_null(eng, u, elig, draw, n_perm, packed)
        if shuffler is None:  # custom null: a unit is movable when its mark changes in some draw
            bits = np.unpackbits(packed, axis=1, count=len(u.mark)).astype(bool)
            movable = (bits != (u.mark > 0.5)[None, :]).any(axis=0)
            bcodes = None
        else:
            movable = shuffler.movable if isinstance(shuffler, _MatchedShuffler) else _unit_movable(u.mark, sub, within)
            bcodes = _codes(_column(sub, blocks, "blocks"), f"blocks column {blocks!r}") if blocks is not None else None
        nb = _block_neff(eng, u, movable, bcodes)
        e_idx = np.flatnonzero(elig.ravel())
        ok_e = (sd > 0) & (nb.ravel()[e_idx] >= min_blocks)
        if not ok_e.any():
            warnings.warn(
                "the null moves the mark nowhere on the eligible pixels (a mark that is a function of clone size or "
                "strata cannot be tested by relabelling, or too few exchangeable clones): no hotspot can be called",
                LowPowerWarning,
                stacklevel=2,
            )
        s0 = eng.s1(u.mark)[elig]
        with np.errstate(invalid="ignore", divide="ignore"):
            z0 = np.where(ok_e, (s0 - mean) / np.where(ok_e, sd, 1.0), np.nan)
        zfull = np.full(u.grid.shape, np.nan)
        zfull[elig] = z0
        s0_ok = s0[ok_e]
        for b in range(n_perm):
            if store is not None:
                sb = store[b]
            else:
                sb = eng.s1(np.unpackbits(packed[b], count=len(u.mark)).astype(float))[elig]
            zb = np.full(len(sb), np.nan)
            zb[ok_e] = (sb[ok_e] - mean[ok_e]) / sd[ok_e]
            for k, s in enumerate(sides):
                t = sign[s] * zb
                fin = np.isfinite(t)
                if correction == "cluster_mass":
                    tfull[elig] = np.where(fin, t, -np.inf)
                    null[b, k] = _max_mass(tfull, cluster_z)
                else:
                    null[b, k] = np.max(t[fin]) if fin.any() else -np.inf
                    if exceed is not None:
                        ge = np.zeros(len(sb), dtype=bool)
                        if s == "hot":
                            ge[ok_e] = sb[ok_e] >= s0_ok - 1e-12 * np.maximum(1.0, np.abs(s0_ok))
                        else:
                            ge[ok_e] = sb[ok_e] <= s0_ok + 1e-12 * np.maximum(1.0, np.abs(s0_ok))
                        exceed[s] += ge
        z_obs_full = zfull
        stud = {
            "pixels_without_null_variance": int((sd <= 0).sum()),
            "pixels_below_min_blocks": int(((sd > 0) & ~ok_e).sum()),
            "movable_units": int(movable.sum()),
        }
        elig = elig & np.isfinite(zfull)

    # ---- observed hotspots
    labels = np.zeros(u.grid.shape, dtype=np.int32)
    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    gx, gy = u.grid.centres()
    next_id = 0
    n_dropped = 0
    for k, s in enumerate(sides):
        sgn = sign[s]
        t_obs = np.nan_to_num(sgn * z_obs_full, nan=-np.inf)
        p_pix = None
        if correction == "maxT":
            crit = _critical(null[:, k], a_side, threshold_rule)
            thresholds[s] = sgn * crit
            flag = elig & (t_obs > crit)
        elif correction == "cluster_mass":
            crit = _critical(null[:, k], a_side, threshold_rule)
            thresholds[s] = crit
            flag = elig & (t_obs > cluster_z)
        else:
            crit = a_side
            p_pix = np.ones(u.grid.shape)
            ex = exceed[s]
            if standardise == "permutation":
                ex = ex[np.isfinite(z_obs_full[sm.eligible])]
            p_pix[elig] = (1 + ex) / (1 + n_perm)
            thresholds[s] = a_side
            flag = elig & np.isfinite(z_obs_full) & (p_pix <= a_side)
        lab, n = _label_by_area(flag)
        n_inside = np.bincount(lab[u.i, u.j], minlength=n + 1)
        keep = []
        for h in range(1, n + 1):
            if n_inside[h] < min_units:  # spill-over of the kernel: no (or too few) units of its own
                n_dropped += 1
                continue
            pix = lab == h
            peak = float(np.max(t_obs[pix]))
            if correction == "cluster_mass":
                stat = float(np.sum(t_obs[pix] - cluster_z))
                if not stat > crit:
                    continue
                p_map = float(mc_pvalue(null[:, k], stat, alternative="greater")[0])
            elif correction == "maxT":
                stat = peak
                p_map = float(mc_pvalue(null[:, k], peak, alternative="greater")[0])
            else:
                stat = float(np.min(p_pix[pix]))
                p_map = stat
            keep.append((pix, peak, stat, p_map))
        keep.sort(key=lambda r: -int(r[0].sum()))  # cluster mass may drop some: renumber the kept ones by area
        for pix, peak, stat, p_map in keep:
            next_id += 1
            labels[pix] = next_id
            ii, jj = np.nonzero(pix)
            rows.append(
                {
                    "hotspot": next_id,
                    "direction": s,
                    "n_pixels": int(pix.sum()),
                    "area_um2": float(pix.sum() * u.grid.pixel_area_um2),
                    "x_um": float(gx[ii].mean()),
                    "y_um": float(gy[jj].mean()),
                    "peak_z": sgn * peak,
                    "statistic": sgn * stat if correction == "maxT" else stat,
                    "p_value": p_map,
                }
            )

    # ---- units inside each hotspot: observed vs expected and vs the same shuffles (descriptive)
    unit_lab = labels[u.i, u.j]
    n_h = next_id
    obs_cnt = np.bincount(unit_lab, weights=u.mark, minlength=n_h + 1)
    exp_cnt = np.bincount(unit_lab, weights=u.p, minlength=n_h + 1)
    n_cnt = np.bincount(unit_lab, minlength=n_h + 1)
    null_cnt = np.zeros((n_perm, n_h + 1))
    if n_h:
        for b in range(n_perm):
            mb = np.unpackbits(packed[b], count=len(u.mark)).astype(float)
            null_cnt[b] = np.bincount(unit_lab, weights=mb, minlength=n_h + 1)
    for r in rows:
        h = r["hotspot"]
        nl = null_cnt[:, h]
        r["units"] = int(n_cnt[h])
        r["marked"] = int(obs_cnt[h])
        r["share"] = obs_cnt[h] / n_cnt[h] if n_cnt[h] else np.nan
        r["expected"] = float(exp_cnt[h])
        r["ratio"] = obs_cnt[h] / exp_cnt[h] if exp_cnt[h] > 0 else np.nan
        r["null_mean"] = float(nl.mean())
        r["null_sd"] = float(nl.std(ddof=1)) if n_perm > 1 else np.nan
        alt = "greater" if r["direction"] == "hot" else "less"
        r["p_count_naive"] = float(mc_pvalue(nl, obs_cnt[h], alternative=alt)[0])
        r["n_perm"] = n_perm
    cols = [
        "hotspot",
        "direction",
        "n_pixels",
        "area_um2",
        "x_um",
        "y_um",
        "peak_z",
        "statistic",
        "units",
        "marked",
        "share",
        "expected",
        "ratio",
        "null_mean",
        "null_sd",
        "p_value",
        "p_count_naive",
        "n_perm",
    ]
    tab = pd.DataFrame({c: [r[c] for r in rows] for c in cols}) if rows else pd.DataFrame(columns=cols)
    summary: dict[str, Any] = {
        "n_hotspots": sum(r["direction"] == "hot" for r in rows),
        "n_coldspots": sum(r["direction"] == "cold" for r in rows),
        "n_dropped_min_units": n_dropped,
    }
    for s in sides:
        summary[f"threshold_{s}"] = thresholds[s]
    fin_obs = np.isfinite(z_obs_full)
    summary.update(
        {
            "max_z": float(np.max(z_obs_full[fin_obs])) if fin_obs.any() else np.nan,
            "min_z": float(np.min(z_obs_full[fin_obs])) if fin_obs.any() else np.nan,
            "p0": sm.p0,
            "n_units": sm.n_units,
            "n_marked": sm.n_marked,
            "n_eligible": int(elig.sum()),
            "eligible_area_um2": float(elig.sum() * u.grid.pixel_area_um2),
        }
    )
    if correction == "maxT":
        for k, s in enumerate(sides):
            t_ext = sign[s] * z_obs_full[fin_obs]
            obs_ext = float(np.max(t_ext)) if len(t_ext) else -np.inf
            summary[f"p_map_{s}"] = float(mc_pvalue(null[:, k], obs_ext, alternative="greater")[0])
            summary[f"null_any_{s}"] = float((null[:, k] > sign[s] * thresholds[s]).mean())
    params = dict(sm.params)
    params.update(
        {
            "n_perm": n_perm,
            "within": within,
            "blocks": blocks,
            "block_design": block_design if blocks is not None else None,
            "standardise": standardise,
            "min_blocks": min_blocks if standardise == "permutation" else None,
            "correction": correction,
            "cluster_z": cluster_z if correction == "cluster_mass" else None,
            "alternative": alternative,
            "alpha": alpha,
            "threshold_rule": threshold_rule,
            "min_units": min_units,
            "design": design,
            "rng_entropy": rng_entropy(gen),
            "subcellular_units": n_subcell,
            **stud,
        }
    )
    return HotspotResult(
        name="hotspots",
        table=tab,
        summary=summary,
        params=params,
        null=null if len(sides) > 1 else null[:, 0],
        map=sm,
        labels=Raster(labels, u.grid, name="hotspots"),
        threshold=thresholds,
        z_map=Raster(np.where(elig, z_obs_full, np.nan), u.grid, name="z"),
    )


def _unit_movable(mark: np.ndarray, table: pd.DataFrame, within: Any) -> np.ndarray:
    """Units whose mark the unit-level design can change: those in strata holding both marks."""
    st = _strata(table, within)
    if st is None:
        st = np.zeros(len(mark), dtype=np.int64)
    lo = pd.Series(mark).groupby(st).transform("min").to_numpy()
    hi = pd.Series(mark).groupby(st).transform("max").to_numpy()
    return lo != hi


# --------------------------------------------------------------------------------------- excess in given regions


def _region_ids(table: pd.DataFrame, regions: Any, x_key: str, y_key: str) -> np.ndarray:
    if isinstance(regions, HotspotResult):
        regions = regions.labels
    if isinstance(regions, Raster):
        for c in (x_key, y_key):
            if c not in table.columns:
                raise KeyError(f"coordinate column {c!r} not in the table; pass x_key / y_key")
        x, y = table[x_key].to_numpy(dtype=float), table[y_key].to_numpy(dtype=float)
        return np.asarray(regions.sample(x, y, fill=0))
    return _column(table, regions, "regions")


def region_excess(
    table: pd.DataFrame,
    *,
    mark: str | Any,
    regions: Any,
    expected: str | Any | None = None,
    match_by: str | None = None,
    within: str | Sequence[str] | None = None,
    blocks: str | None = None,
    block_design: BlockDesign = "matched",
    n_perm: int = 10_000,
    min_units: int = 5,
    alternative: Alternative = "greater",
    pooled: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
    rng: RngLike = None,
    keep_null: bool = False,
) -> TestResult:
    """Do given regions hold more (or fewer) marked units than their covariate-matched expectation?

    Typical use: class-switch (CSR) hotspots, found with the switched mark, tested for an excess of mutated loci
    beyond what their isotype mix predicts (the germinal-centre signature).

    Parameters
    ----------
    table
        One row per unit (loci) with the mark.
    mark
        Binary column (or array).
    regions
        Region of each unit: a column / array of region ids (0, negative or missing = outside), or a label
        :class:`~spaclone.tl.Raster` (e.g. ``hotspots(...).labels``) or a :class:`HotspotResult`, sampled at the
        unit positions (off the raster = outside).
    expected, match_by
        Expected probability of each unit, as in :func:`share_map` (default: the global share).
    within
        Strata of the shuffles; defaults to ``match_by`` (except with ``blocks`` and ``block_design="free"``).
    blocks
        Exchangeable blocks: the clone id for marks that belong to a lineage (mutated, switched); ``None`` = every unit
        on its own.
    block_design
        With ``blocks``: ``"matched"`` (default; clones trade marks only with clones of the same size and stratum mix,
        and marks are re-dealt among a clone's units of each stratum) or ``"free"`` (the 1.0 design: block-level marks
        permuted among all blocks of a stratum), as in :func:`hotspots`.
    n_perm
        Number of shuffles.
    min_units
        Regions with fewer units are not tested (they still count in the pooled row).
    alternative
        Direction of the test.
    pooled
        Add a row ``region = "all"`` for all regions together.
    x_key, y_key
        Coordinate columns, used when ``regions`` is a raster.
    rng
        Seed or generator (one ``permutation`` per stratum and shuffle).
    keep_null
        Keep the ``(n_perm, regions + 1)`` null counts (column 0 = outside every region).

    Returns
    -------
    TestResult
        ``table``: ``region``, ``units``, ``marked``, ``share``, ``statistic`` (= marked), ``expected`` (Σ p),
        ``expected_raw`` (global share × units), ``ratio`` (marked / expected), ``null_mean``, ``null_sd``,
        ``null_lo``, ``null_hi`` (2.5% / 97.5%), ``z``, ``p_value`` (Monte-Carlo), ``q_value`` (BH over regions, not
        the pooled row), ``n_perm``.

    Notes
    -----
    Null model (``blocks=None``): random labelling within strata. Positions, regions and each stratum's multiset of
    marks are fixed; marks are permuted among the units of each stratum. With ``match_by`` as strata the expected
    count of a region equals the null mean, so an excess means more marked units than the region's covariate mix
    predicts.

    Null model (``blocks``, ``block_design="matched"``): random relabelling of whole clones, as in :func:`hotspots`:
    clones of the same signature (units per stratum) trade their mark profiles, and inside a clone the marks of a
    stratum are re-dealt among its units of that stratum; positions, regions and strata are fixed. A region's null
    mean is then its expectation given the isotype mix *and* the clone sizes of its units, and ``expected`` (Σ p)
    stays the covariate-matched expectation for reference. Use it whenever the mark belongs to a lineage: unit-level
    shuffles count a clone of 30 mutated loci as 30 independent units and give far too small null variances.

    The regions must not have been selected on this mark (select them with another mark, another section or held-out
    units).

    References
    ----------
    Davison & Hinkley 1997, doi:10.1017/CBO9780511802843; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585;
    Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if expected is not None and match_by is not None:
        raise ValueError("give either expected or match_by, not both")
    gen = as_rng(rng)
    m = _binary(_column(table, mark, "mark"), "mark")
    raw = pd.Series(_region_ids(table, regions, x_key, y_key))
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().sum() == raw.notna().sum():  # numeric ids: <= 0 is outside
        inside = num.notna() & (num > 0)
        ids = num[inside]
    else:
        inside = raw.notna() & (raw.astype(str) != "")
        ids = raw[inside].astype(str)
    uniq = np.sort(ids.unique()) if len(ids) else np.array([])
    code = np.zeros(len(table), dtype=np.int64)
    if len(ids):
        code[inside.to_numpy()] = np.searchsorted(uniq, ids.to_numpy()) + 1
    if expected is not None:
        p = np.asarray(_column(table, expected, "expected"), dtype=float)
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise ValueError("expected probabilities must lie in [0, 1]")
        kind = "given per unit"
    elif match_by is not None:
        cl = _codes(_column(table, match_by, "match_by"), f"match_by column {match_by!r}")
        p = (np.bincount(cl, weights=m) / np.bincount(cl))[cl]
        kind = "match"
        if within is None and (blocks is None or block_design == "matched"):
            within = match_by
    else:
        p = np.full(len(m), m.mean())
        kind = "global"
    if block_design not in ("matched", "free"):
        raise ValueError("block_design must be 'matched' or 'free'")
    shuffler, design = _make_shuffler(m, table, within=within, blocks=blocks, block_design=block_design)
    nr = len(uniq)
    obs = np.bincount(code, weights=m, minlength=nr + 1)
    exp = np.bincount(code, weights=p, minlength=nr + 1)
    cnt = np.bincount(code, minlength=nr + 1)
    null = np.empty((n_perm, nr + 1))
    for b in range(n_perm):
        null[b] = np.bincount(code, weights=shuffler.draw(gen), minlength=nr + 1)
    p0 = float(m.mean()) if len(m) else np.nan
    sel = []
    for k in range(1, nr + 1):
        if cnt[k] >= min_units:
            rid = uniq[k - 1]
            sel.append((rid.item() if hasattr(rid, "item") else rid, obs[k], exp[k], cnt[k], null[:, k]))
    if pooled and nr and cnt[1:].sum() >= min_units:
        sel.append(("all", obs[1:].sum(), exp[1:].sum(), cnt[1:].sum(), null[:, 1:].sum(axis=1)))
    out = []
    for rid, o, e, n, nl in sel:
        mu = float(nl.mean())
        sd = float(nl.std(ddof=1)) if len(nl) > 1 else np.nan
        out.append(
            {
                "region": rid,
                "units": int(n),
                "marked": int(o),
                "share": o / n,
                "statistic": float(o),
                "expected": float(e),
                "expected_raw": p0 * n,
                "ratio": o / e if e > 0 else np.nan,
                "null_mean": mu,
                "null_sd": sd,
                "null_lo": float(np.quantile(nl, 0.025)),
                "null_hi": float(np.quantile(nl, 0.975)),
                "z": (o - mu) / sd if sd > 0 else np.nan,
                "p_value": float(mc_pvalue(nl, o, alternative=alternative)[0]),
                "n_perm": n_perm,
            }
        )
    cols = [
        "region",
        "units",
        "marked",
        "share",
        "statistic",
        "expected",
        "expected_raw",
        "ratio",
        "null_mean",
        "null_sd",
        "null_lo",
        "null_hi",
        "z",
        "p_value",
        "n_perm",
    ]
    tab = pd.DataFrame(out, columns=cols)
    per = np.array([not (isinstance(r, str) and r == "all") for r in tab.region], dtype=bool)
    q = np.full(len(tab), np.nan)
    if per.any():
        q[per] = bh(tab.p_value.to_numpy()[per])
    tab.insert(len(tab.columns) - 1, "q_value", q)
    summary: dict[str, Any] = {"regions": nr, "tested": int(per.sum()), "units": len(m), "p0": p0}
    if len(tab) and not per[-1]:
        summary.update(
            {
                "pooled_marked": int(tab.marked.iloc[-1]),
                "pooled_expected": float(tab.expected.iloc[-1]),
                "pooled_ratio": float(tab.ratio.iloc[-1]),
                "pooled_p": float(tab.p_value.iloc[-1]),
            }
        )
    params = {
        "n_perm": n_perm,
        "expected": kind,
        "match_by": match_by,
        "within": within,
        "blocks": blocks,
        "block_design": block_design if blocks is not None else None,
        "min_units": min_units,
        "alternative": alternative,
        "design": design,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="region_excess", table=tab, summary=summary, params=params, null=null if keep_null else None)
