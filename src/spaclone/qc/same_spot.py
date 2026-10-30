"""Same-spot test: does a receptor molecule sit on a spot whose GEX holds its own constant-region transcript?

In Stereo-seq-type data the receptor (XCR) library and the GEX library are made from the same cDNA on the same spot
barcodes (0.5 µm DNBs). A genuinely located IGK molecule is an Igkc cDNA on its own DNB, so the GEX library sees an
Igkc molecule on that DNB with probability ``p_det`` (the GEX per-molecule detection rate). A molecule whose barcode
was swapped in library PCR sits on another cDNA-carrying DNB and meets Igkc only by chance. For each spot that holds
receptor molecules of one locus we record a *hit* (GEX holds at least one molecule of the locus's constant genes on
that spot) and the *chance* ``c_i`` of a hit for a molecule that was placed there at random. With located molecules
the hit probability of record ``i`` is

    π_i = c_i + θ_i · (1 − c_i),

the Rogan–Gladen misclassification model (Sp = 1 − c_i, Se = p_det + (1 − p_det) c_i), where ``θ_i`` is the
probability that record ``i`` is a located molecule whose own transcript the GEX library detected. The located
fraction ``ρ`` follows from ``θ = ρ · p_det``.

**Chance model** (``chance=``). The default, ``"empirical"``, measures the chance on the data instead of deriving it
from a model of molecules: ``c_i`` is the share of *reference spots* that hold the constant gene, among the
GEX-occupied spots that carry no record of the locus (``background="receptor_free"``) and have the record's number
of GEX molecules ``m_i`` (``m_bins``), the same amount of constant-gene RNA on the spots around it (``context_um``,
default 4 µm, a label-free proxy of "inside a cell that expresses the gene") and optionally the same spot label
(``strata``, e.g. the cell type), in the smallest square block of ``tile_um`` tiles around the record that holds
``min_reference`` such spots. Reference spots are weighted by their GEX molecules (cDNA): a swapped molecule takes
the barcode of a random captured cDNA molecule, so a spot is drawn in proportion to its cDNA. This is the exact null
of placing each swapped molecule on a random cDNA molecule of its stratum and needs no assumption about how
constant-region RNA is spread over spots, which matters because it is not spread at all: it sits, in bursts, on the
spots of the B / plasma (T) cells that express it. Its identifying assumption is that, within a stratum, swaps land
in proportion to the cDNA the GEX library *sees*; cDNA it missed (bursty Ig RNA at low saturation) would draw swaps
to spots that hold the gene and give hits above chance without located molecules (an upward bias, so upper bounds
stay conservative while a "signal" needs :func:`shift_test` and the clone-level tests). The analytic alternatives treat
the spot's molecules as a random subset of its tile's molecules: ``"conditional"`` (``c_i = 1 − C(T − S, m_i) /
C(T, m_i)``, the 1.0.0.dev0 default) and ``"binomial"`` (``c_i = 1 − (1 − S/T)^{m_i}``, the archived estimate).
They over-state the chance of a randomly placed molecule (Ig RNA is clustered) and under-state it for a molecule on
its own B / plasma cell (whose other Ig RNA counts as chance), so hits fall below "chance" on broken data and θ is
over-estimated on good data (review 2026-10-30, B1).

**Estimand and estimator** (``estimator=``). The default, ``"ipw"``, estimates the plain mean of ``θ_i`` over the
tested records, ``θ̂ = mean[(y_i − c_i) / (1 − c_i)]``, the located fraction (times detection) among the molecules
the test can see. A hit counts 1 whatever its chance, so a located molecule's own transcript, which adds one
molecule to its spot and so raises the spot's chance, does not bias it (it biases the moment and ML estimates
towards 0); a spot that would hold the gene by chance anyway carries little information and makes the interval
wide, which is honest. The interval inverts ``(θ̂ − θ) / sqrt(V(θ))``,
with ``V(θ)`` the larger of the Bernoulli variance at ``θ`` and a block (cluster-robust) variance over square
blocks of ``block_um`` (the analytic equivalent of a tile-block bootstrap; it carries the over-dispersion of hits,
e.g. molecules of one cell and bursts), plus the sampling variance of the empirical chances (finite reference spots,
shared by neighbouring records), against a t quantile with (blocks − 1) degrees of freedom. ``"exact"`` is the Rogan–Gladen moment estimate ``θ̂ = (H − Σc_i) / (N − Σc_i)``
with the exact Poisson-binomial interval (widened to the block interval when ``variance="cluster"``); it estimates a
``(1 − c_i)``-weighted mean of ``θ_i``, which down-weights located molecules on spots rich in constant-region RNA
(the B / plasma cells where they would sit), so its upper bound is not a bound on the fraction located. ``"ml"`` is
maximum likelihood (biased towards 0 under the alternative). Because the test only sees GEX-occupied spots, a
located molecule enters more often when its own transcript was detected (an empty spot becomes occupied by it):
``θ`` is ``ρ`` times the detection rate *given an occupied spot*, which is at least ``p_det``, so ``ρ = θ / p_det``
with the library-wide ``p_det`` is conservative (too large).

**Deficit rule.** Fewer hits than chance, beyond what the interval allows (``deficit_p <= (1 − conf_level) / 2``,
the larger of the exact Poisson-binomial and the block p-value), means that the chance model over-states the chance;
then no bound is reported (``rho_pdet_hi`` is NaN, ``bound_reliable`` is False, ``bound_note`` says why), so a
mis-specified chance can never be read as "no located molecules".

**What the test cannot see.** A molecule whose template switch happened inside its constant region keeps the
acceptor's constant region and barcode and the donor's V(D)J ("right isotype, wrong clone"). It sits on a spot that
holds its chain's constant-region RNA and counts as located here, although its clone identity is random. The
same-spot test is a molecule-level test; whether clone identity follows position needs the clone-level tests
(:func:`spaclone.qc.colocalisation`, :func:`spaclone.qc.pairing`).

Pitfalls handled here (survey S3 §D): one record per spot, never per molecule (a barcode sink would count hundreds of
times); single-clone spots only by default; GEX-empty spots carry no chance model and are excluded.

References
----------
Rogan & Gladen 1978, doi:10.1093/oxfordjournals.aje.a112510; Lang & Reiczigel 2014,
doi:10.1016/j.prevetmed.2013.09.015 (prevalence with estimated sensitivity); Reiczigel et al. 2010,
doi:10.1017/S0950268810000385 (exact intervals); Hong 2013, doi:10.1016/j.csda.2012.10.006 (Poisson-binomial);
Costes et al. 2004, doi:10.1529/biophysj.103.038422 (displacement controls); Horvitz & Thompson 1952,
doi:10.1080/01621459.1952.10483446 (inverse weighting); Wilson 1927, doi:10.1080/01621459.1927.10502953 (score
intervals); Liang & Zeger 1986, doi:10.1093/biomet/73.1.13 (cluster-robust variance); Künsch 1989,
doi:10.1214/aos/1176347265 (block resampling of dependent data).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import fft, optimize, special, stats

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, bh, mc_pvalue, rng_entropy
from spaclone.nulls import bin_labels

__all__ = [
    "CONTEXT_BINS",
    "HUMAN_CONSTANT_GENES",
    "M_BINS",
    "MOUSE_CONSTANT_GENES",
    "MULTIPLICITY_BINS",
    "GexLookup",
    "SameSpotResult",
    "located_fraction",
    "same_spot",
    "shift_test",
]

logger = logging.getLogger("spaclone")

#: Constant-region genes per locus, mouse (Ensembl / SAW gene symbols).
MOUSE_CONSTANT_GENES: dict[str, tuple[str, ...]] = {
    "IGH": ("Ighm", "Ighd", "Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha", "Ighe"),
    "IGK": ("Igkc",),
    "IGL": ("Iglc1", "Iglc2", "Iglc3"),
    "TRA": ("Trac",),
    "TRB": ("Trbc1", "Trbc2"),
    "TRG": ("Tcrg-C1", "Tcrg-C2", "Tcrg-C3", "Tcrg-C4"),
    "TRD": ("Trdc",),
}
#: Constant-region genes per locus, human (HGNC symbols).
HUMAN_CONSTANT_GENES: dict[str, tuple[str, ...]] = {
    "IGH": ("IGHM", "IGHD", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHE"),
    "IGK": ("IGKC",),
    "IGL": ("IGLC1", "IGLC2", "IGLC3", "IGLC7"),
    "TRA": ("TRAC",),
    "TRB": ("TRBC1", "TRBC2"),
    "TRG": ("TRGC1", "TRGC2"),
    "TRD": ("TRDC",),
}
#: Edges of the clones-per-spot classes reported by :func:`same_spot` (``1``, ``2``, ``3-5``, ``6-20``, ``>=21``).
MULTIPLICITY_BINS: tuple[float, ...] = (1, 2, 3, 6, 21, np.inf)

#: Edges of the GEX-molecule classes of the empirical chance: exact counts 1-4, then roughly logarithmic classes.
M_BINS: tuple[float, ...] = (1, 2, 3, 4, 5, 7, 10, 15, 25, 40, 70, np.inf)
#: Edges of the classes of constant-gene molecules around a spot (``context_um``) used by the empirical chance.
CONTEXT_BINS: tuple[float, ...] = (0, 1, 2, 4, 8, 16, 32, np.inf)

ChanceModel = Literal["empirical", "conditional", "binomial"]
Background = Literal["receptor_free", "all"]
Estimator = Literal["ipw", "exact", "ml"]
Variance = Literal["cluster", "binomial"]
_SHIFT = np.int64(32)
_LOW = np.int64((1 << 32) - 1)
_MAXC = np.int64(1 << 31)
_TINY = float(np.finfo(float).tiny)


class ZeroChanceWarning(SpacloneWarning):
    """Hits were given a chance of zero (sparse tiles with ``chance="binomial", leave_own_out=True``)."""


def _null_text(
    chance: str, leave_own_out: bool, background: str, strata: str | None = None, context_um: float | None = None
) -> str:
    if chance == "empirical":
        ref = (
            "GEX-occupied spots without a record of the locus"
            if background == "receptor_free"
            else "GEX-occupied spots"
        )
        by = "the same class of GEX molecules (m_bins)"
        if strata:
            by += f", the same {strata!r} label"
        if context_um:
            by += f" and the same class of constant-gene molecules within {context_um:g} um"
        return (
            "rho = 0: each record's spot is a random cDNA molecule's spot within its stratum, i.e. a spot drawn with "
            f"probability proportional to its GEX molecules among the {ref} with {by} in the smallest square block of "
            "tiles around the record holding min_reference such spots; c_i = cDNA-weighted share of those reference "
            "spots that hold a constant-gene molecule. Conditional on the reference spots; records independent."
        )
    bg = "the tile's spots without records of the locus" if background == "receptor_free" else "all spots of the tile"
    bg += " (the record's spot excluded)" if leave_own_out else " plus the record's own spot"
    if chance == "conditional" and not leave_own_out:
        how = (
            f"its m_i GEX molecules are a random subset (without replacement) of the molecules of {bg}: "
            "c_i = 1 - C(T - S, m_i) / C(T, m_i)"
        )
    elif chance == "conditional":
        how = (
            f"its m_i GEX molecules are a random draw from the molecules of {bg}: "
            "c_i = 1 - C(T - S, m_i) / C(T, m_i) (unbiased for 1 - (1 - q)^m_i)"
        )
    else:
        how = f"its m_i GEX molecules are independent draws with the gene share q of {bg}: c_i = 1 - (1 - q)^m_i"
    return (
        "rho = 0: each record's spot is a GEX-occupied spot whose constant-gene content is independent of the record; "
        f"{how}. Conditional on the spots, their GEX molecule counts and the tile compositions; records independent."
    )


# ------------------------------------------------------------------------------------------------------ GEX lookup


def _reduce_by_key(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    return sk[starts], np.add.reduceat(np.asarray(counts)[order].astype(np.int64, copy=False), starts)


def _reduce_by_key_float(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return np.empty(0, np.int64), np.empty(0, float)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    return sk[starts], np.add.reduceat(np.asarray(values, dtype=float)[order], starts)


class _RefDesign:
    """Which reference spots gave each record its empirical chance, for the sampling variance of the chances.

    ``variance(coef)`` is the variance of ``Σ_i coef_i · c_i`` when every reference spot ``s`` holds the gene
    independently with its (tile, stratum) share ``p_s``: ``Σ_s (w_s · Σ_{i: s in R_i} coef_i / W_i)² p_s (1 − p_s)``.
    """

    def __init__(self, tab: dict[str, Any], k: np.ndarray, tx: np.ndarray, ty: np.ndarray, r: np.ndarray,
                 w: np.ndarray) -> None:  # fmt: skip
        self.tab, self.k, self.tx, self.ty, self.r, self.w = tab, k, tx, ty, r, w

    def subset(self, idx: np.ndarray) -> _RefDesign:
        return _RefDesign(self.tab, self.k[idx], self.tx[idx], self.ty[idx], self.r[idx], self.w[idx])

    def variance(self, coef: np.ndarray) -> float:
        coef = np.asarray(coef, dtype=float)
        sel = (coef != 0) & np.isfinite(coef) & (self.w > 0)
        if not sel.any():
            return 0.0
        v = coef[sel] / self.w[sel]
        k, tx, ty, r = self.k[sel], self.tx[sel], self.ty[sel], self.r[sel]
        bx, by = self.tab["bx"], self.tab["by"]
        codes, vals = [], []
        for rr in np.unique(r):
            m = r == rr
            km, xm, ym, vm = k[m] << 40, tx[m] - bx, ty[m] - by, v[m]
            for dx in range(-int(rr), int(rr) + 1):
                for dy in range(-int(rr), int(rr) + 1):
                    codes.append(km | ((xm + dx) << 20) | (ym + dy))
                    vals.append(vm)
        gk, gv = _reduce_by_key_float(np.concatenate(codes), np.concatenate(vals))
        ck, cw, ch, cw2 = self.tab["ck"], self.tab["cw"], self.tab["ch"], self.tab["cw2"]
        g = _find(gk, gv, ck)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(cw > 0, ch / np.where(cw > 0, cw, 1), 0.0)
        return float((g**2 * p * (1 - p) * cw2).sum())


def _find(sorted_keys: np.ndarray, values: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.zeros(len(k), dtype=values.dtype if len(values) else np.int64)
    if len(sorted_keys) == 0 or len(k) == 0:
        return out
    i = np.clip(np.searchsorted(sorted_keys, k), 0, len(sorted_keys) - 1)
    ok = (sorted_keys[i] == k) & (k >= 0)
    out[ok] = values[i[ok]]
    return out


def _log_hyper_none(t: np.ndarray, s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Log P(no constant-gene molecule among m drawn without replacement from t molecules of which s are), vector.

    ``-inf`` where ``m > t - s`` (every draw must hit). Requires ``0 <= s <= t`` and ``0 <= m <= t``.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(len(m))
    imp = m > t - s
    out[imp] = -np.inf
    ok = ~imp & (m > 0) & (s > 0)
    small = ok & (m <= 64)
    if small.any():
        ts, ss, ms = t[small], s[small], m[small]
        acc = np.zeros(len(ms))
        for i in range(int(ms.max())):
            live = i < ms
            acc[live] += np.log1p(-ss[live] / (ts[live] - i))
        out[small] = acc
    big = ok & (m > 64)
    if big.any():
        tb, sb, mb = t[big], s[big], m[big].astype(float)
        out[big] = (
            special.gammaln(tb - sb + 1)
            - special.gammaln(tb - sb - mb + 1)
            - special.gammaln(tb + 1)
            + special.gammaln(tb - mb + 1)
        )
    return out


class GexLookup:
    """Spot-level GEX molecule counts on a regular grid, for same-spot lookups.

    Holds, for every spot with at least one GEX molecule (on or off tissue), the total molecules over all genes and,
    for each *loaded* gene, the molecules of that gene. Receptor positions are matched to GEX spots on the grid of
    pitch ``spot_um`` (0.5 µm for Stereo-seq DNBs) anchored at ``origin_um``, so positions must be spot coordinates,
    not jittered.

    Build it with :meth:`from_gef` (SAW ``raw.gef``, whole chip), :meth:`from_stereo`, :meth:`from_arrays`
    (long-format counts) or :meth:`from_anndata` (gridded spot-level AnnData such as binned Stereo-seq or Visium HD).

    Parameters
    ----------
    ix, iy
        Integer grid indices ((position − origin) / ``spot_um``) of the occupied spots.
    total
        GEX molecules per occupied spot, all genes.
    genes
        ``{gene: (ix, iy, count)}`` for the loaded genes (spots with ≥ 1 molecule of the gene).
    spot_um
        Grid pitch in µm.
    origin_um
        Position (µm) of grid index ``(0, 0)``.
    extent
        ``(min_ix, max_ix, min_iy, max_iy)`` of the measured area (default: the occupied spots).
    meta
        Free-form metadata (source file, chip, units).
    """

    def __init__(
        self,
        ix: np.ndarray,
        iy: np.ndarray,
        total: np.ndarray,
        genes: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        spot_um: float,
        origin_um: tuple[float, float] = (0.0, 0.0),
        extent: tuple[int, int, int, int] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        ix = np.asarray(ix, dtype=np.int64)
        iy = np.asarray(iy, dtype=np.int64)
        total = np.asarray(total)
        if not (len(ix) == len(iy) == len(total)):
            raise ValueError("ix, iy and total must have the same length")
        if not spot_um > 0:
            raise ValueError("spot_um must be positive")
        self.spot_um = float(spot_um)
        self.origin_um = (float(origin_um[0]), float(origin_um[1]))
        if extent is None:
            extent = (int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())) if len(ix) else (0, 0, 0, 0)
        self.extent = tuple(int(v) for v in extent)
        self._ox = min(self.extent[0], int(ix.min()) if len(ix) else 0)
        self._oy = min(self.extent[2], int(iy.min()) if len(iy) else 0)
        self._keys, self._total = _reduce_by_key(self._key(ix, iy), total)
        self._genes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for g, (gx, gy, gc) in genes.items():
            gk, gcnt = _reduce_by_key(self._key(np.asarray(gx, np.int64), np.asarray(gy, np.int64)), gc)
            keep = gcnt > 0
            self._genes[str(g)] = (gk[keep], gcnt[keep])
        self.meta = dict(meta or {})
        self._tile_cache: dict[int, tuple] = {}
        self._set_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._set_tile_cache: dict[tuple[tuple[str, ...], int], Any] = {}
        self._labels: dict[str, tuple[np.ndarray, list[Any]]] = {}
        self._ctx_cache: dict[tuple[tuple[str, ...], float], np.ndarray] = {}
        self._emp_cache: dict[tuple, dict[str, Any]] = {}

    # -- construction -------------------------------------------------------------------------------------------
    def _key(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        a, b = ix - self._ox, iy - self._oy
        if len(a) and (a.min() < 0 or b.min() < 0 or a.max() >= _MAXC or b.max() >= _MAXC):
            raise ValueError("spot indices out of range for the grid")
        return (a << _SHIFT) | b

    @classmethod
    def from_arrays(
        cls,
        x_um: Any,
        y_um: Any,
        gene: Any,
        count: Any = None,
        *,
        spot_um: float = 0.5,
        origin_um: tuple[float, float] = (0.0, 0.0),
        genes: Sequence[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> GexLookup:
        """Build from long-format GEX counts: one entry per (spot, gene) or per molecule.

        Parameters
        ----------
        x_um, y_um
            Spot positions (µm) of each entry; they are snapped to the grid of pitch ``spot_um`` at ``origin_um``.
        gene
            Gene name of each entry.
        count
            Molecules per entry (default 1 each).
        spot_um
            Grid pitch (µm).
        origin_um
            Position of grid index ``(0, 0)``.
        genes
            Genes whose spots are kept for lookups (default: all). Totals always use every entry.
        meta
            Metadata stored on the object.
        """
        x = np.asarray(x_um, dtype=float)
        y = np.asarray(y_um, dtype=float)
        gname = np.asarray(gene, dtype=object)
        cnt = np.ones(len(x), dtype=np.int64) if count is None else np.asarray(count, dtype=np.int64)
        if not (len(x) == len(y) == len(gname) == len(cnt)):
            raise ValueError("x_um, y_um, gene and count must have the same length")
        if (cnt < 0).any():
            raise ValueError("counts must be non-negative")
        keep = cnt > 0
        x, y, gname, cnt = x[keep], y[keep], gname[keep], cnt[keep]
        ix = np.rint((x - origin_um[0]) / spot_um).astype(np.int64)
        iy = np.rint((y - origin_um[1]) / spot_um).astype(np.int64)
        codes, uniq = pd.factorize(pd.Series(gname, dtype=object), sort=True)
        wanted = [str(g) for g in uniq] if genes is None else list(dict.fromkeys(str(g) for g in genes))
        gene_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        pos = {str(g): i for i, g in enumerate(uniq)}
        for g in wanted:
            j = pos.get(g)
            if j is None:
                gene_data[g] = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
                continue
            rows = order[bounds[j] : bounds[j + 1]]
            gene_data[g] = (ix[rows], iy[rows], cnt[rows])
        return cls(ix, iy, cnt, gene_data, spot_um=spot_um, origin_um=origin_um, meta=meta)

    @classmethod
    def from_stereo(cls, gef: Any) -> GexLookup:
        """Build from the :class:`spaclone.io.StereoGEX` returned by :func:`spaclone.io.read_gef_bin1`."""
        m = gef.meta
        extent = (m["min_x"], m["max_x"], m["min_y"], m["max_y"])
        return cls(gef.x, gef.y, gef.total, gef.genes, spot_um=m["um_per_dnb"], extent=extent, meta=m)

    @classmethod
    def from_gef(cls, path: str | Path, genes: Sequence[str], **kwargs: Any) -> GexLookup:
        """Read a SAW ``raw.gef`` (whole chip, bin1) and keep the DNBs of ``genes``.

        The file is read with :func:`spaclone.io.read_gef_bin1`, which receives ``**kwargs``.
        """
        from spaclone.io.stereo import read_gef_bin1

        return cls.from_stereo(read_gef_bin1(path, genes, **kwargs))

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        genes: Sequence[str] | None = None,
        *,
        spot_um: float,
        spatial_key: str = "spatial",
        layer: str | None = None,
        tol: float = 1e-3,
        labels: str | Sequence[str] | None = None,
    ) -> GexLookup:
        """Build from a spot-level AnnData whose spots lie on a regular grid (``obsm[spatial_key]`` in µm).

        Parameters
        ----------
        adata
            AnnData with counts in ``X`` (or ``layers[layer]``), one observation per spot (bin).
        genes
            Genes to load for lookups (default: every constant gene of :data:`MOUSE_CONSTANT_GENES` and
            :data:`HUMAN_CONSTANT_GENES` found in ``var_names``). Totals always use all genes.
        spot_um
            Grid pitch in µm (e.g. 2 for Visium HD 2-µm bins).
        spatial_key, layer
            Where to find positions and counts.
        tol
            Largest allowed distance from the grid, in pitches; positions are snapped to the grid anchored at the
            first spot.
        labels
            ``obs`` column(s) loaded as spot labels (e.g. ``"cell_type"``) for the ``strata`` of the empirical chance
            (see :meth:`add_labels`).
        """
        from scipy import sparse

        xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
        if len(xy) == 0:
            raise ValueError("adata has no observations")
        origin = (
            float(xy[0, 0] - spot_um * np.floor(xy[0, 0] / spot_um)),
            float(xy[0, 1] - spot_um * np.floor(xy[0, 1] / spot_um)),
        )
        fx = (xy[:, 0] - origin[0]) / spot_um
        fy = (xy[:, 1] - origin[1]) / spot_um
        ix, iy = np.rint(fx).astype(np.int64), np.rint(fy).astype(np.int64)
        if max(np.abs(fx - ix).max(), np.abs(fy - iy).max()) > tol:
            raise ValueError(f"spot positions are not on a grid of pitch {spot_um} µm")
        if pd.DataFrame({"x": ix, "y": iy}).duplicated().any():
            raise ValueError("two observations fall on the same grid position")
        mat = adata.layers[layer] if layer is not None else adata.X
        total = np.asarray(mat.sum(axis=1)).ravel()
        names = [str(v) for v in adata.var_names]
        if genes is None:
            known = {g for d in (MOUSE_CONSTANT_GENES, HUMAN_CONSTANT_GENES) for gs in d.values() for g in gs}
            genes = [g for g in names if g in known]
        col = {g: i for i, g in enumerate(names)}
        gene_data = {}
        for g in genes:
            if g not in col:
                gene_data[str(g)] = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
                continue
            v = mat[:, col[g]]
            v = np.asarray(v.toarray() if sparse.issparse(v) else v).ravel()
            nz = np.flatnonzero(v > 0)
            gene_data[str(g)] = (ix[nz], iy[nz], np.rint(v[nz]).astype(np.int64))
        keep = total > 0
        out = cls(
            ix[keep],
            iy[keep],
            np.rint(total[keep]).astype(np.int64),
            gene_data,
            spot_um=spot_um,
            origin_um=origin,
            meta={"source": "anndata", "spot_um": spot_um},
        )
        for name in [labels] if isinstance(labels, str) else list(labels or []):
            if name not in adata.obs:
                raise KeyError(f"labels: column {name!r} not in adata.obs")
            out.add_labels(name, xy[:, 0], xy[:, 1], adata.obs[name].to_numpy())
        return out

    def add_labels(self, name: str, x_um: Any, y_um: Any, values: Any, *, bin_um: float | None = None) -> GexLookup:
        """Attach a spot label (e.g. the cell type) to every occupied spot, for the ``strata`` of the empirical chance.

        Parameters
        ----------
        name
            Name of the label set (passed as ``strata=name`` to :func:`same_spot`).
        x_um, y_um, values
            Labelled positions (µm) and their labels. Without ``bin_um`` each occupied spot takes the label given at
            its own grid position; with ``bin_um`` the positions are square bins of that side (aligned at the grid
            origin, e.g. bin-20 cell-type calls on DNBs) and every spot takes its bin's label. Occupied spots without a
            label get their own stratum (``NA``).
        bin_um
            Side of the labelled bins (a multiple of the spot pitch), or ``None`` for spot-level labels.

        Returns
        -------
        GexLookup
            ``self`` (labels are stored in place).

        Notes
        -----
        Labels must not be computed from the spot's own constant-gene molecules (e.g. a per-DNB call that uses Igkc):
        a located molecule's own transcript would then move its spot into a stratum of high chance and the located
        fraction would be under-estimated. Cell-level or bin-level calls are safe.
        """
        x = np.asarray(x_um, dtype=float)
        y = np.asarray(y_um, dtype=float)
        vals = pd.Series(np.asarray(values, dtype=object))
        if not (len(x) == len(y) == len(vals)):
            raise ValueError("x_um, y_um and values must have the same length")
        codes, levels = pd.factorize(vals, sort=True, use_na_sentinel=True)
        if bin_um is None:
            qk = self._qkeys(*self._grid(x, y))
        else:
            bpx = self._tile_px(bin_um)
            bx, by = self._grid(x, y)
            qk = ((bx // bpx) << _SHIFT) | (by // bpx)
        order = np.argsort(qk, kind="stable")
        sk, sc = qk[order], codes[order].astype(np.int64)
        first = np.r_[True, sk[1:] != sk[:-1]] if len(sk) else np.zeros(0, bool)
        sk, sc = sk[first], sc[first]
        if bin_um is None:
            look = self._keys
        else:
            ax = ((self._keys >> _SHIFT) + self._ox) // bpx
            ay = ((self._keys & _LOW) + self._oy) // bpx
            look = (ax << _SHIFT) | ay
        lab = np.full(len(self._keys), -1, dtype=np.int64)
        if len(sk):
            i = np.clip(np.searchsorted(sk, look), 0, len(sk) - 1)
            ok = sk[i] == look
            lab[ok] = sc[i[ok]]
        self._labels[str(name)] = (lab, [*levels])
        return self

    # -- properties ---------------------------------------------------------------------------------------------
    @property
    def genes(self) -> tuple[str, ...]:
        """Genes whose spots were loaded."""
        return tuple(self._genes)

    @property
    def n_spots(self) -> int:
        """Number of spots holding at least one GEX molecule."""
        return len(self._keys)

    @property
    def labels(self) -> tuple[str, ...]:
        """Names of the spot label sets attached with :meth:`add_labels`."""
        return tuple(self._labels)

    def __repr__(self) -> str:
        return (
            f"GexLookup({self.n_spots:,} occupied spots, pitch {self.spot_um:g} µm, "
            f"{int(self._total.sum()):,} molecules, genes={list(self.genes)})"
        )

    # -- lookups ------------------------------------------------------------------------------------------------
    def _grid(self, x_um: Any, y_um: Any) -> tuple[np.ndarray, np.ndarray]:
        ix = np.rint((np.asarray(x_um, dtype=float) - self.origin_um[0]) / self.spot_um).astype(np.int64)
        iy = np.rint((np.asarray(y_um, dtype=float) - self.origin_um[1]) / self.spot_um).astype(np.int64)
        return ix, iy

    def _qkeys(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        a, b = ix - self._ox, iy - self._oy
        ok = (a >= 0) & (a < _MAXC) & (b >= 0) & (b < _MAXC)
        return np.where(ok, (np.where(ok, a, 0) << _SHIFT) | np.where(ok, b, 0), -1)

    def _gene_set(self, genes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(sorted(set(genes)))
        if key in self._set_cache:
            return self._set_cache[key]
        missing = [g for g in key if g not in self._genes]
        if missing:
            raise KeyError(f"gene(s) {missing} were not loaded into this GexLookup (loaded: {list(self.genes)})")
        ks = [self._genes[g][0] for g in key]
        cs = [self._genes[g][1] for g in key]
        out = _reduce_by_key(np.concatenate(ks), np.concatenate(cs)) if ks else (np.empty(0, np.int64),) * 2
        self._set_cache[key] = out
        return out

    def spot_counts(self, x_um: Any, y_um: Any, genes: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """GEX molecules at the given spots: ``(total over all genes, molecules of genes)``; 0 for empty spots."""
        k = self._qkeys(*self._grid(x_um, y_um))
        m = _find(self._keys, self._total, k)
        if genes is None:
            return m, np.zeros_like(m)
        sk, sc = self._gene_set(genes)
        return m, _find(sk, sc, k)

    def _tile_px(self, tile_um: float) -> int:
        r = float(tile_um) / self.spot_um
        t = int(round(r))
        if t < 1 or abs(r - t) > 1e-6 * max(r, 1.0):
            raise ValueError(f"tile_um={tile_um} must be a positive multiple of the spot pitch {self.spot_um} µm")
        return t

    def _tiles(self, tpx: int) -> tuple:
        if tpx in self._tile_cache:
            return self._tile_cache[tpx]
        tx = ((self._keys >> _SHIFT) + self._ox) // tpx
        ty = ((self._keys & _LOW) + self._oy) // tpx
        e = self.extent
        x0 = min(e[0] // tpx, int(tx.min())) if len(tx) else e[0] // tpx
        y0 = min(e[2] // tpx, int(ty.min())) if len(ty) else e[2] // tpx
        ntx = (max(e[1] // tpx, int(tx.max())) if len(tx) else e[1] // tpx) - x0 + 1
        nty = (max(e[3] // tpx, int(ty.max())) if len(ty) else e[3] // tpx) - y0 + 1
        if ntx * nty <= 50_000_000:
            dense = np.bincount((tx - x0) * nty + (ty - y0), weights=self._total, minlength=ntx * nty)
            out = ("dense", x0, y0, ntx, nty, dense)
        else:  # pragma: no cover - very fine tiles on very large chips
            uk, sums = _reduce_by_key(((tx - x0) << _SHIFT) | (ty - y0), self._total)
            out = ("sparse", x0, y0, ntx, nty, (uk, sums.astype(float)))
        self._tile_cache[tpx] = out
        return out

    def _set_tiles(self, genes: tuple[str, ...], tpx: int) -> Any:
        key = (genes, tpx)
        if key in self._set_tile_cache:
            return self._set_tile_cache[key]
        kind, x0, y0, ntx, nty, _ = self._tiles(tpx)
        sk, sc = self._gene_set(genes)
        tx = ((sk >> _SHIFT) + self._ox) // tpx - x0
        ty = ((sk & _LOW) + self._oy) // tpx - y0
        if kind == "dense":
            out: Any = np.bincount(tx * nty + ty, weights=sc, minlength=ntx * nty)
        else:  # pragma: no cover
            uk, sums = _reduce_by_key((tx << _SHIFT) | ty, sc)
            out = (uk, sums.astype(float))
        self._set_tile_cache[key] = out
        return out

    @staticmethod
    def _tile_values(table: Any, kind: str, x0: int, y0: int, ntx: int, nty: int, ix, iy, tpx) -> np.ndarray:
        tx = ix // tpx - x0
        ty = iy // tpx - y0
        ok = (tx >= 0) & (tx < ntx) & (ty >= 0) & (ty < nty)
        out = np.zeros(len(ix), dtype=float)
        if kind == "dense":
            out[ok] = table[tx[ok] * nty + ty[ok]]
        else:  # pragma: no cover
            uk, sums = table
            out[ok] = _find(uk, sums, (tx[ok] << _SHIFT) | ty[ok])
        return out

    def evaluate(
        self,
        x_um: Any,
        y_um: Any,
        genes: Sequence[str],
        *,
        tile_um: float = 25.0,
        chance: ChanceModel = "conditional",
        leave_own_out: bool = False,
        exclude: tuple[Any, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Hit and chance of a constant-gene molecule at the given spots.

        Parameters
        ----------
        x_um, y_um
            Spot positions (µm).
        genes
            The gene set (e.g. the constant genes of one locus).
        tile_um
            Side of the square tiles (aligned at the grid origin) whose molecules define the chance.
        chance
            ``"conditional"``: hypergeometric chance (the spot's molecules are a random subset of the background
            molecules of its tile); ``"binomial"``: ``1 − (1 − q)^m`` with the background gene share ``q``. See
            :func:`same_spot`.
        leave_own_out
            Remove the looked-up spot's own molecules from the background.
        exclude
            ``(x_um, y_um)`` of spots whose molecules are removed from the tile background (e.g. every spot carrying
            receptor records of the locus, whose genuine own transcripts would otherwise raise the chance of all
            records in the tile). A looked-up spot that is itself excluded is put back (unless ``leave_own_out``).

        Returns
        -------
        dict
            ``m`` (GEX molecules on the spot), ``own`` (molecules of ``genes`` on it), ``hit`` (``own > 0``),
            ``tile_total`` and ``tile_set`` (the background molecules used by the chance model), ``q`` (``tile_set /
            tile_total``; NaN for an empty background) and ``chance`` (0 for empty spots; NaN when the chance model is
            undefined: an empty background, or fewer background molecules than molecules on the spot).
        """
        if chance not in ("conditional", "binomial"):
            raise ValueError(f"chance must be 'conditional' or 'binomial', got {chance!r}")
        gs = tuple(sorted(set(genes)))
        ix, iy = self._grid(x_um, y_um)
        k = self._qkeys(ix, iy)
        m = _find(self._keys, self._total, k)
        sk, sc = self._gene_set(gs)
        own = _find(sk, sc, k)
        tpx = self._tile_px(tile_um)
        kind, x0, y0, ntx, nty, tot_table = self._tiles(tpx)
        t_all = self._tile_values(tot_table, kind, x0, y0, ntx, nty, ix, iy, tpx)
        t_set = self._tile_values(self._set_tiles(gs, tpx), kind, x0, y0, ntx, nty, ix, iy, tpx)
        focal_in_bg = np.ones(len(m), dtype=bool)
        if exclude is not None:
            ex, ey = self._grid(*exclude)
            ek = np.unique(self._qkeys(ex, ey))
            ek = ek[ek >= 0]
            em = _find(self._keys, self._total, ek).astype(float)
            eo = _find(sk, sc, ek).astype(float)
            occ_e = em > 0
            ek, em, eo = ek[occ_e], em[occ_e], eo[occ_e]
            big = np.int64(1 << 31)
            etile = (((ek >> _SHIFT) + self._ox) // tpx - x0) * big + (((ek & _LOW) + self._oy) // tpx - y0)
            qx, qy = ix // tpx - x0, iy // tpx - y0
            qtile = np.where((qx >= 0) & (qy >= 0), qx * big + qy, -1)
            ut, inv = np.unique(etile, return_inverse=True)
            sum_m = np.bincount(inv.ravel(), weights=em, minlength=len(ut))
            sum_o = np.bincount(inv.ravel(), weights=eo, minlength=len(ut))
            t_all = t_all - _find(ut, sum_m, qtile)
            t_set = t_set - _find(ut, sum_o, qtile)
            focal_in_bg = ~np.isin(k, ek)
            t_all = t_all + np.where(focal_in_bg, 0.0, m)
            t_set = t_set + np.where(focal_in_bg, 0.0, own)
        if leave_own_out:
            t_all = t_all - m
            t_set = t_set - own
        mf = m.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(t_all > 0, t_set / np.where(t_all > 0, t_all, 1.0), np.nan)
            if chance == "binomial":
                c = np.where(m > 0, 1.0 - (1.0 - q) ** mf, 0.0)
            else:
                c = np.zeros(len(m))
                occ = m > 0
                defined = occ & (t_all >= m) & (t_all > 0)
                c[occ & ~defined] = np.nan
                if defined.any():
                    lp = _log_hyper_none(t_all[defined], t_set[defined], m[defined])
                    c[defined] = -np.expm1(lp)
        return {"m": m, "own": own, "hit": own > 0, "tile_total": t_all, "tile_set": t_set, "q": q, "chance": c}

    def _empirical_tables(
        self,
        gs: tuple[str, ...],
        ek: np.ndarray,
        tpx: int,
        edges: np.ndarray,
        strata: str | None,
        context_um: float | None,
        context_bins: Sequence[float],
        weights: str,
        max_pool: int,
    ) -> dict[str, Any] | None:
        """Reference-spot sums per (stratum, tile) and per stratum for :meth:`evaluate_empirical`."""
        keys, tot = self._keys, self._total
        if len(keys) == 0:
            return None
        sk, sc = self._gene_set(gs)
        hit_all = _find(sk, sc, keys) > 0
        ref = np.ones(len(keys), dtype=bool)
        if len(ek):
            pos = np.clip(np.searchsorted(keys, ek), 0, len(keys) - 1)
            ref[pos[keys[pos] == ek]] = False
        nm = len(edges) - 1
        k_all = np.clip(np.searchsorted(edges, tot, side="right") - 1, 0, nm - 1).astype(np.int64)
        n_extra = 1
        extra = np.zeros(len(keys), dtype=np.int64)
        out: dict[str, Any] = {}
        if strata is not None:
            lab, levels = self._labels[strata]
            extra = np.where(lab < 0, len(levels), lab).astype(np.int64)
            n_extra = len(levels) + 1
        if context_um is not None:
            ce = np.asarray(context_bins, dtype=float)
            nc = len(ce) - 1
            ca = np.clip(np.searchsorted(ce, self.context_counts(gs, context_um), side="right") - 1, 0, nc - 1)
            out["ctx_class"] = ca.astype(np.int64)
            extra, n_extra = extra * nc + ca, n_extra * nc
        k_all = k_all * n_extra + extra
        tx_all = ((keys >> _SHIFT) + self._ox) // tpx
        ty_all = ((keys & _LOW) + self._oy) // tpx
        bx = int(tx_all.min()) - max_pool - 1 - (1 << 18)
        by = int(ty_all.min()) - max_pool - 1 - (1 << 18)
        w_all = tot.astype(np.int64) if weights == "cdna" else np.ones(len(keys), dtype=np.int64)
        r_code = ((k_all << 40) | ((tx_all - bx) << 20) | (ty_all - by))[ref]
        wr, hr = w_all[ref], w_all[ref] * hit_all[ref]
        ck, cn = _reduce_by_key(r_code, np.ones(len(r_code), dtype=np.int64))
        _, cw = _reduce_by_key(r_code, wr)
        _, ch = _reduce_by_key(r_code, hr)
        _, cw2 = _reduce_by_key(r_code, wr * wr)
        out.update({"n_extra": n_extra, "bx": bx, "by": by, "ck": ck, "cn": cn, "cw": cw, "ch": ch, "cw2": cw2})
        return out

    def context_counts(self, genes: Sequence[str], context_um: float) -> np.ndarray:
        """Molecules of ``genes`` on the other spots within ``context_um`` (square window) of every occupied spot.

        Aligned with the occupied spots (:meth:`spot_arrays`); the spot's own molecules are not counted. On very large
        grids the window is evaluated on a coarser grid (a multiple of the pitch, at most a quarter of the window),
        which keeps the count local to about a cell.
        """
        gs = tuple(sorted(set(genes)))
        key = (gs, float(context_um))
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        sk, sc = self._gene_set(gs)
        r = max(1, int(round(float(context_um) / self.spot_um)))
        ax = (self._keys >> _SHIFT) + self._ox
        ay = (self._keys & _LOW) + self._oy
        out = np.zeros(len(self._keys), dtype=np.int64)
        if len(sk) and len(ax):
            gx = (sk >> _SHIFT) + self._ox
            gy = (sk & _LOW) + self._oy
            f = 1
            span = (max(ax.max(), gx.max()) - min(ax.min(), gx.min()) + 1) * (
                max(ay.max(), gy.max()) - min(ay.min(), gy.min()) + 1
            )
            while span / (f * f) > 6e7 and f < max(1, r // 2):
                f += 1
            rc = max(1, int(round(r / f)))
            x0, y0 = min(ax.min(), gx.min()) // f, min(ay.min(), gy.min()) // f
            nxg = int(max(ax.max(), gx.max()) // f - x0 + 1)
            nyg = int(max(ay.max(), gy.max()) // f - y0 + 1)
            grid = np.zeros((nxg + 1, nyg + 1), dtype=np.int64)
            np.add.at(grid, (gx // f - x0 + 1, gy // f - y0 + 1), sc)
            cs = grid.cumsum(axis=0).cumsum(axis=1)
            qx, qy = ax // f - x0 + 1, ay // f - y0 + 1
            a0, a1 = np.clip(qx - rc - 1, 0, nxg), np.clip(qx + rc, 0, nxg)
            b0, b1 = np.clip(qy - rc - 1, 0, nyg), np.clip(qy + rc, 0, nyg)
            out = cs[a1, b1] - cs[a0, b1] - cs[a1, b0] + cs[a0, b0] - _find(sk, sc, self._keys)
            out = np.maximum(out, 0)
        self._ctx_cache[key] = out
        return out

    def evaluate_empirical(
        self,
        x_um: Any,
        y_um: Any,
        genes: Sequence[str],
        *,
        exclude: tuple[Any, Any] | None = None,
        tile_um: float = 25.0,
        m_bins: Sequence[float] = M_BINS,
        strata: str | None = None,
        context_um: float | None = None,
        context_bins: Sequence[float] = CONTEXT_BINS,
        min_reference: int = 10,
        max_pool: int = 8,
        weights: Literal["cdna", "spots"] = "cdna",
    ) -> dict[str, np.ndarray]:
        """Empirical chance of a constant-gene hit at the given spots, from comparable reference spots.

        For every looked-up spot, the reference spots are the occupied spots that are not in ``exclude``, fall in
        the same class of GEX molecules (``m_bins``), carry the same ``strata`` label and the same class of
        constant-gene molecules within ``context_um`` (``context_bins``), and lie in the smallest square block of
        ``(2r + 1)²`` tiles of ``tile_um`` around the spot's tile (``r = 0 .. max_pool``) that holds at least
        ``min_reference`` of them (else the largest block, whatever it holds; no chance without any reference
        spot: the whole chip is never used, because it mixes tissue regions). The chance is their share holding a
        molecule of ``genes``, each weighted by its GEX molecules (``weights="cdna"``: a swapped molecule lands on a
        random cDNA molecule) or equally (``"spots"``).

        Parameters
        ----------
        x_um, y_um
            Spot positions (µm).
        genes
            The gene set (e.g. the constant genes of one locus).
        exclude
            ``(x_um, y_um)`` of spots that never serve as reference (the spots carrying records of the locus, whose
            located molecules' own transcripts would raise every chance).
        tile_um, m_bins, strata, context_um, context_bins, min_reference, max_pool, weights
            See above; ``strata`` names a label set attached with :meth:`add_labels`.

        Returns
        -------
        dict
            ``m``, ``own``, ``hit``, ``chance`` (0 for empty spots, NaN without any reference spot), ``ref_n``
            (reference spots used), ``pool_um`` (half-side of the block of tiles used), ``stratum`` (code of the
            stratum) and ``design`` (which reference spots made each chance, for their sampling variance).
        """
        gs = tuple(sorted(set(genes)))
        if min_reference < 1 or max_pool < 0:
            raise ValueError("min_reference must be >= 1 and max_pool >= 0")
        if weights not in ("cdna", "spots"):
            raise ValueError("weights must be 'cdna' or 'spots'")
        edges = np.asarray(m_bins, dtype=float)
        if edges[0] > 1 or not np.all(np.diff(edges) > 0):
            raise ValueError("m_bins must be increasing and start at <= 1")
        if strata is not None and strata not in self._labels:
            raise KeyError(f"strata={strata!r} is not a label set of this GexLookup (labels: {list(self.labels)})")
        ix, iy = self._grid(x_um, y_um)
        qk = self._qkeys(ix, iy)
        m = _find(self._keys, self._total, qk)
        sk, sc = self._gene_set(gs)
        own = _find(sk, sc, qk)
        keys = self._keys
        nm = len(edges) - 1
        tpx = self._tile_px(tile_um)
        ek = np.zeros(0, np.int64)
        if exclude is not None:
            ek = np.unique(self._qkeys(*self._grid(*exclude)))
            ek = ek[ek >= 0]
        ckey = (gs, tpx, tuple(edges), strata, context_um, tuple(context_bins), weights, max_pool, ek.tobytes())
        tab = self._emp_cache.get(ckey)
        if tab is None:
            tab = self._empirical_tables(gs, ek, tpx, edges, strata, context_um, context_bins, weights, max_pool)
            if len(self._emp_cache) >= 4:
                self._emp_cache.pop(next(iter(self._emp_cache)))
            self._emp_cache[ckey] = tab
        if tab is None or len(keys) == 0:
            nan = np.full(len(m), np.nan)
            return {"m": m, "own": own, "hit": own > 0, "chance": np.where(m > 0, nan, 0.0), "ref_n": np.zeros(len(m)),
                    "pool_um": nan, "stratum": np.full(len(m), -1)}  # fmt: skip
        n_extra, bx, by = tab["n_extra"], tab["bx"], tab["by"]
        ck, cn, cw, ch = tab["ck"], tab["cn"], tab["cw"], tab["ch"]
        extra_q = np.zeros(len(qk), dtype=np.int64)
        if strata is not None:
            lab, levels = self._labels[strata]
            la = np.where(lab < 0, len(levels), lab)
            lq = _find(keys, la + 1, qk) - 1
            extra_q = np.where(lq < 0, len(levels), lq)
        if context_um is not None:
            nc = len(context_bins) - 1
            cq = _find(keys, tab["ctx_class"] + 1, qk) - 1
            extra_q = extra_q * nc + np.where(cq < 0, 0, cq)
        tx_q, ty_q = ix // tpx, iy // tpx

        def mclass(v: np.ndarray) -> np.ndarray:
            return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nm - 1).astype(np.int64)

        kq = mclass(m) * n_extra + extra_q
        acc = np.zeros((3, len(m)))
        rad = np.full(len(m), max_pool, dtype=np.int64)
        todo = np.flatnonzero(m > 0)
        for r in range(max_pool + 1):
            if len(todo) == 0:
                break
            ring = [(0, 0)] if r == 0 else (
                [(d, -r) for d in range(-r, r + 1)] + [(d, r) for d in range(-r, r + 1)]
                + [(-r, d) for d in range(-r + 1, r)] + [(r, d) for d in range(-r + 1, r)]
            )  # fmt: skip
            for dx, dy in ring:
                c = (kq[todo] << 40) | ((tx_q[todo] + dx - bx) << 20) | (ty_q[todo] + dy - by)
                acc[0, todo] += _find(ck, cn, c)
                acc[1, todo] += _find(ck, cw, c)
                acc[2, todo] += _find(ck, ch, c)
            done = acc[0, todo] >= min_reference
            rad[todo[done]] = r
            todo = todo[~done]
        with np.errstate(divide="ignore", invalid="ignore"):
            chance = np.where(m > 0, np.where(acc[1] > 0, acc[2] / np.where(acc[1] > 0, acc[1], 1.0), np.nan), 0.0)
        design = _RefDesign(tab, kq, tx_q, ty_q, rad, np.where(m > 0, acc[1], 0.0))
        return {"m": m, "own": own, "hit": own > 0, "chance": chance, "ref_n": acc[0],
                "pool_um": np.where(m > 0, (rad + 0.5) * tile_um, np.nan), "stratum": np.where(m > 0, kq, -1),
                "design": design}  # fmt: skip

    def spot_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x_um, y_um, total)`` of every occupied spot."""
        ix = (self._keys >> _SHIFT) + self._ox
        iy = (self._keys & _LOW) + self._oy
        return ix * self.spot_um + self.origin_um[0], iy * self.spot_um + self.origin_um[1], self._total.copy()

    def grid_counts(self, bin_um: float, gene_sets: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
        """GEX molecules in square bins covering the whole measured area (empty bins included).

        Parameters
        ----------
        bin_um
            Bin side (a multiple of the spot pitch); bins are aligned at the grid origin.
        gene_sets
            ``{column: genes}`` counted besides the total (genes must be loaded).

        Returns
        -------
        pandas.DataFrame
            ``x_um``, ``y_um`` (lower-left corner of the bin), ``total`` and one column per gene set; bins ordered
            with ``x`` varying slowest.
        """
        bpx = self._tile_px(bin_um)
        bx0, bx1 = self.extent[0] // bpx, self.extent[1] // bpx
        by0, by1 = self.extent[2] // bpx, self.extent[3] // bpx
        nbx, nby = bx1 - bx0 + 1, by1 - by0 + 1

        def count(keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
            bx = ((keys >> _SHIFT) + self._ox) // bpx - bx0
            by = ((keys & _LOW) + self._oy) // bpx - by0
            ok = (bx >= 0) & (bx < nbx) & (by >= 0) & (by < nby)
            return np.bincount(bx[ok] * nby + by[ok], weights=vals[ok], minlength=nbx * nby)

        gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1), np.arange(by0, by1 + 1), indexing="ij")
        out = pd.DataFrame(
            {
                "x_um": gx.ravel() * bpx * self.spot_um + self.origin_um[0],
                "y_um": gy.ravel() * bpx * self.spot_um + self.origin_um[1],
            }
        )
        out["total"] = count(self._keys, self._total)
        for name, gs in (gene_sets or {}).items():
            sk, sc = self._gene_set(gs)
            out[name] = count(sk, sc)
        return out


# ------------------------------------------------------------------------------------ Poisson-binomial upper tail


def _conv_rows(pmf: np.ndarray, direct_max: int = 64) -> np.ndarray:
    """Pmf of the sum of independent variables whose pmfs are the rows of ``pmf`` (support 0, 1, ...).

    A merge tree in which every level convolves all pairs of rows at once: directly while the rows are short
    (``<= direct_max``), by one batched real FFT above (absolute error ~1e-15 of the largest probability; negative
    round-off is clipped).
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    while pmf.shape[0] > 1:
        if pmf.shape[0] % 2:
            delta = np.zeros((1, pmf.shape[1]))
            delta[0, 0] = 1.0
            pmf = np.vstack([pmf, delta])
        a, b = pmf[0::2], pmf[1::2]
        length = a.shape[1]
        if length <= direct_max:
            out = np.zeros((a.shape[0], 2 * length - 1))
            for j in range(length):
                out[:, j : j + length] += a[:, j : j + 1] * b
        else:
            nfft = fft.next_fast_len(2 * length - 1, real=True)
            out = fft.irfft(fft.rfft(a, nfft, axis=1) * fft.rfft(b, nfft, axis=1), nfft, axis=1)[:, : 2 * length - 1]
            np.clip(out, 0.0, None, out=out)
        pmf = out
    return pmf[0]


def _pb_pmf(p: np.ndarray, block: int = 16) -> np.ndarray:
    """Pmf of a sum of independent Bernoulli(p_i): exact recursion within blocks of ``block``, then :func:`_conv_rows`."""
    n = len(p)
    if n == 0:
        return np.ones(1)
    nb = -(-n // block)
    pp = np.zeros(nb * block)
    pp[:n] = p
    pp = pp.reshape(nb, block)
    pmf = np.zeros((nb, block + 1))
    pmf[:, 0] = 1.0
    for j in range(block):
        q = pp[:, j : j + 1]
        pmf[:, 1:] = pmf[:, 1:] * (1 - q) + pmf[:, :-1] * q
        pmf[:, 0] *= 1 - q[:, 0]
    return _conv_rows(pmf)[: n + 1]


def discrete_sum_sf(pmf: Any, k: int) -> float:
    """Upper tail ``P(T >= k)`` of ``T = Σ_s J_s`` for independent ``J_s`` with pmf rows ``pmf[s, j] = P(J_s = j)``.

    Same construction as :func:`poisson_binomial_sf`: summed directly below the mean, exponentially tilted to mean
    ``k`` above it, so that tail probabilities keep their relative precision; floored at the smallest positive double.
    Used for the exact stratified permutation tests of :func:`spaclone.qc.umi_match` (sums of hypergeometric
    variables).
    """
    pm = np.atleast_2d(np.asarray(pmf, dtype=float))
    pm = np.clip(pm, 0.0, None)
    pm = pm / pm.sum(axis=1, keepdims=True)
    width = pm.shape[1]
    j = np.arange(width)
    nz = pm > 0
    lo_s = np.argmax(nz, axis=1)
    hi_s = width - 1 - np.argmax(nz[:, ::-1], axis=1)
    k = int(k) - int(lo_s.sum())  # shift every variable to start at 0
    pm = np.array([np.roll(r, -o) for r, o in zip(pm, lo_s, strict=True)]) if lo_s.any() else pm
    span = hi_s - lo_s
    keep = span > 0
    pm, span = pm[keep], span[keep]
    if k <= 0:
        return 1.0
    total = int(span.sum())
    if k > total:
        return 0.0
    pm = pm[:, : int(span.max()) + 1]
    j = np.arange(pm.shape[1])
    logp = np.log(np.where(pm > 0, pm, 1.0)) + np.where(pm > 0, 0.0, -np.inf)
    if k == total:
        return float(max(np.exp(logp[np.arange(len(pm)), span].sum()), _TINY))
    mean = float((pm * j).sum())
    if k <= mean:
        return float(min(1.0, _conv_rows(pm)[k:].sum()))

    def tilted_mean(t: float) -> float:
        w = logp + t * j
        w = np.exp(w - w.max(axis=1, keepdims=True))
        return float(((w * j).sum(axis=1) / w.sum(axis=1)).sum()) - k

    hi = 1.0
    while tilted_mean(hi) < 0:
        hi *= 2.0
    t = optimize.brentq(tilted_mean, 0.0, hi, xtol=1e-12, rtol=1e-12, maxiter=500)
    w = logp + t * j
    log_m = special.logsumexp(w, axis=1)
    pt = np.exp(w - log_m[:, None])
    conv = _conv_rows(pt)
    tail = float((conv[k:] * np.exp(-t * np.arange(len(conv) - k))).sum())
    lp = float(log_m.sum()) - t * k + np.log(tail) if tail > 0 else -np.inf
    return float(max(np.exp(lp), _TINY)) if np.isfinite(lp) else _TINY


def poisson_binomial_sf(p: Any, k: int, *, log: bool = False) -> float:
    """Upper tail ``P(X >= k)`` of a sum of independent Bernoulli(``p_i``) (Poisson-binomial distribution).

    Below the mean the tail is summed from the pmf; above it the pmf is computed under an exponentially tilted
    measure whose mean is ``k`` (so the terms that matter carry most of the tilted mass and keep full precision) and
    the tail is untilted exactly: ``P(X >= k) = M(t) e^{-tk} Σ_{j>=k} P_t(j) e^{-t(j-k)}``. Accurate in relative terms
    far into the tail; returned values are floored at the smallest positive double (the true value is then smaller).

    Parameters
    ----------
    p
        Success probabilities.
    k
        Threshold.
    log
        Return the natural logarithm instead.

    References
    ----------
    Hong 2013, doi:10.1016/j.csda.2012.10.006; Butler 2007, doi:10.1017/CBO9780511619083 (exponential tilting).
    """
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    k = int(k)
    ones = int((p >= 1.0).sum())
    p = p[(p > 0.0) & (p < 1.0)]
    k -= ones
    n = len(p)
    if k <= 0:
        return 0.0 if log else 1.0
    if k > n:
        return -np.inf if log else 0.0
    if k == n:
        lp = float(np.log(p).sum())
    elif k <= p.sum():
        pmf = _pb_pmf(p)
        lp = float(np.log(min(1.0, pmf[k:].sum())))
    else:
        lg = special.logit(p)

        def tilted_mean(t: float) -> float:
            return float(special.expit(lg + t).sum()) - k

        hi = 1.0
        while tilted_mean(hi) < 0:
            hi *= 2.0
        t = optimize.brentq(tilted_mean, 0.0, hi, xtol=1e-12, rtol=1e-12, maxiter=500)
        pt = special.expit(lg + t)
        log_m = float((np.log1p(-p) + np.logaddexp(0.0, lg + t)).sum())
        pmf_t = _pb_pmf(pt)
        tail = float((pmf_t[k:] * np.exp(-t * np.arange(n - k + 1))).sum())
        lp = log_m - t * k + np.log(tail) if tail > 0 else -np.inf
    if log:
        return lp
    return float(max(np.exp(lp), _TINY)) if np.isfinite(lp) else _TINY


def poisson_binomial_cdf(p: Any, k: int) -> float:
    """Lower tail ``P(X <= k)`` of a Poisson-binomial sum, as the upper tail of ``n − X`` (same precision)."""
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    return poisson_binomial_sf(1.0 - p, len(p) - int(k))


def _exact_interval(c: np.ndarray, h: int, tail: float) -> tuple[float, float, bool]:
    """Interval for theta in pi_i = c_i + theta (1 - c_i) by inverting the exact Poisson-binomial law of the hits.

    Lower end: smallest theta with P_theta(H >= h) >= tail; upper end: largest theta with P_theta(H <= h) >= tail
    (Clopper-Pearson construction with record-specific chance). The third value flags data with fewer hits than any
    theta in [0, 1] makes plausible (upper end set to 0).
    """
    n = len(c)
    e, v = float(c.sum()), float((c * (1 - c)).sum())
    zq = float(stats.norm.ppf(1 - tail))

    def pi(t: float) -> np.ndarray:
        return c + t * (1.0 - c)

    def solve(f: Callable[[float], float], guess: float) -> float:
        """Root of a monotone f on [0, 1], bracketed around a normal-approximation guess."""
        a, b = 0.0, min(1.0, max(guess, 1e-9))
        fa, fb = f(a), f(b)
        while fa * fb > 0 and b < 1.0:
            a, fa = b, fb
            b = min(1.0, 2.0 * b)
            fb = f(b)
        if fa * fb > 0:  # pragma: no cover - no sign change on [0, 1]
            return b
        return float(optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-5, maxiter=200))

    th = (h - e) / (n - e) if n > e else 0.0
    half = zq * np.sqrt(v + max(h, 1)) / max(n - e, 1.0)
    if h == 0:
        lo = 0.0
    else:
        g = lambda t: poisson_binomial_sf(pi(t), h) - tail
        lo = 0.0 if g(0.0) >= 0 else solve(g, max(th - half, 1e-9))
    if h >= n:
        return lo, 1.0, False
    f = lambda t: poisson_binomial_cdf(pi(t), h) - tail
    if f(0.0) <= 0:
        return lo, 0.0, True
    return lo, solve(f, max(th, 0.0) + 2.0 * half), False


# --------------------------------------------------------------------------------------- likelihood and intervals


def _loglik(theta: float, c_hit: np.ndarray, n0: int) -> float:
    """Log-likelihood of theta = rho * p_det up to a constant: hits add log(c + theta (1 - c)), misses log(1 - theta)."""
    if theta < 0 or theta > 1:
        return -np.inf
    with np.errstate(divide="ignore"):
        lh = np.log(c_hit + theta * (1.0 - c_hit)).sum() if len(c_hit) else 0.0
        lm = n0 * np.log1p(-theta) if n0 else 0.0
    return float(lh + lm)


def _mle(c_hit: np.ndarray, n0: int) -> float:
    if len(c_hit) == 0:
        return 0.0
    if n0 == 0:
        return 1.0

    def score(t: float) -> float:
        with np.errstate(divide="ignore"):
            return float(((1.0 - c_hit) / (c_hit + t * (1.0 - c_hit))).sum() - n0 / (1.0 - t))

    s0 = score(0.0)
    if s0 <= 0:
        return 0.0
    lo = 0.0 if np.isfinite(s0) else 1e-300  # a hit with zero chance: the likelihood vanishes at theta = 0
    hi = 1.0 - 1e-15
    if score(hi) >= 0:  # pragma: no cover - only with a vanishing number of misses
        return 1.0
    return float(optimize.brentq(score, lo, hi, xtol=1e-15, rtol=1e-12, maxiter=500))


def _profile_interval(
    fun: Callable[[float], float], x_hat: float, crit: float, lo: float = 0.0, hi: float = 1.0
) -> tuple[float, float]:
    """{x in [lo, hi]: fun(x_hat) - fun(x) <= crit} for a unimodal log-likelihood ``fun``."""
    top = fun(x_hat)

    def g(x: float) -> float:
        v = fun(x)
        return top - (v if np.isfinite(v) else -1e300) - crit

    if x_hat <= lo:
        a = lo
    else:
        a_end = lo if np.isfinite(fun(lo)) else lo + (x_hat - lo) * 1e-12
        a = lo if g(a_end) <= 0 else float(optimize.brentq(g, a_end, x_hat, xtol=1e-15, rtol=1e-12, maxiter=500))
    if x_hat >= hi:
        b = hi
    else:
        b_end = hi if np.isfinite(fun(hi)) else hi - (hi - x_hat) * 1e-12
        b = hi if g(b_end) <= 0 else float(optimize.brentq(g, x_hat, b_end, xtol=1e-15, rtol=1e-12, maxiter=500))
    return a, b


def _parse_p_det(p_det: Any) -> tuple[float, float] | None:
    if p_det is None:
        return None
    if isinstance(p_det, Mapping):
        est, se = float(p_det["p_det"]), float(p_det.get("se") or 0.0)
    elif isinstance(p_det, tuple | list | np.ndarray):
        if len(p_det) != 2:
            raise ValueError("p_det as a sequence must be (estimate, standard error)")
        est, se = float(p_det[0]), float(p_det[1])
    else:
        est, se = float(p_det), 0.0
    if not 0 < est <= 1:
        raise ValueError(f"p_det must lie in (0, 1], got {est}")
    if not np.isfinite(se) or se < 0:
        raise ValueError("the standard error of p_det must be finite and non-negative")
    return est, se


def _block_codes(blocks: Any, n: int) -> tuple[np.ndarray, int]:
    if blocks is None:
        return np.zeros(n, dtype=np.int64), 1
    b = np.asarray(blocks).ravel()
    if len(b) != n:
        raise ValueError("blocks must have one label per record")
    codes = pd.factorize(pd.Series(b), sort=False, use_na_sentinel=False)[0].astype(np.int64)
    return codes, int(codes.max()) + 1 if n else 0


class _Moments:
    """Estimating function of θ for one of the moment estimators, with model and block variances."""

    def __init__(
        self,
        y: np.ndarray,
        c: np.ndarray,
        codes: np.ndarray,
        n_blocks: int,
        *,
        weighted: bool,
        cluster: bool,
        vref: Callable[[float], float] | None = None,
    ) -> None:
        self.y, self.c = y.astype(float), c
        self.vref = vref or (lambda t: 0.0)
        self.codes, self.B = codes, n_blocks
        self.weighted, self.cluster = weighted, cluster and n_blocks >= 2
        self.n = len(y)
        if weighted:  # unweighted mean of θ_i: inverse-chance weights, θ solves Σ u_i = θ Σ d_i
            self.w = 1.0 / (1.0 - c)
            self.u, self.d = (self.y - c) * self.w, np.ones(self.n)
        else:
            self.w = np.ones(self.n)
            self.u, self.d = self.y - c, 1.0 - c

    def estimate(self) -> float:
        return float(self.u.sum() / self.d.sum()) if self.d.sum() > 0 else np.nan

    def score(self, t: float) -> float:
        return float(self.u.sum() - t * self.d.sum())

    def variances(self, t: float) -> tuple[float, float]:
        """(model variance, block variance) of the score at θ = t."""
        pi = np.clip(self.c + t * (1.0 - self.c), 0.0, 1.0)
        vm = float((self.w**2 * pi * (1.0 - pi)).sum())
        if not self.cluster:
            return vm, np.nan
        a = np.bincount(self.codes, weights=self.u - t * self.d, minlength=self.B)
        return vm, float(self.B / (self.B - 1) * (a**2).sum())

    def sd(self, t: float) -> float:
        vm, vc = self.variances(t)
        return float(np.sqrt((max(vm, vc) if np.isfinite(vc) else vm) + self.vref(t)))

    def quantile(self, tail: float) -> float:
        return float(stats.t.ppf(1 - tail, self.B - 1)) if self.cluster else float(stats.norm.ppf(1 - tail))

    def interval(self, tail: float, th: float) -> tuple[float, float]:
        """Score interval {θ in [0, 1]: |score(θ)| <= q · sd(θ)}."""
        q = self.quantile(tail)

        def lo_f(t: float) -> float:
            return self.score(t) - q * self.sd(t)

        def hi_f(t: float) -> float:
            return self.score(t) + q * self.sd(t)

        start = min(max(th, 0.0), 1.0)
        lo = 0.0 if lo_f(0.0) <= 0 else float(optimize.brentq(lo_f, 0.0, max(start, 1e-12), xtol=1e-12))
        if hi_f(1.0) >= 0:
            hi = 1.0
        elif hi_f(0.0) <= 0:
            hi = 0.0
        else:
            a = start if hi_f(start) > 0 else 0.0
            hi = float(optimize.brentq(hi_f, a, 1.0, xtol=1e-12))
        return lo, hi

    def null_tails(self) -> tuple[float, float, float]:
        """(upper p, lower p, z) of the block test of θ = 0 (normal reference without blocks)."""
        s = self.score(0.0)
        sd = self.sd(0.0)
        z = s / sd if sd > 0 else np.nan
        if not np.isfinite(z):
            return np.nan, np.nan, np.nan
        if self.cluster:
            df = self.B - 1
            return float(stats.t.sf(z, df)), float(stats.t.cdf(z, df)), float(z)
        return float(stats.norm.sf(z)), float(stats.norm.cdf(z)), float(z)


def located_fraction(
    hit: Any,
    chance: Any,
    *,
    p_det: float | tuple[float, float] | Mapping[str, float] | None = None,
    conf_level: float = 0.95,
    method: Literal["ipw", "exact", "ml", "plugin"] = "ipw",
    blocks: Any = None,
    variance: Variance = "cluster",
    chance_cap: float = 0.9,
    chance_design: Any = None,
) -> dict[str, Any]:
    """Fraction of receptor molecules sitting on their own spot, from per-spot hits and chances.

    Parameters
    ----------
    hit
        Boolean per spot record: the spot's GEX holds a molecule of the locus's constant gene(s).
    chance
        Per-record chance of a hit under the null, ``c_i`` (records with non-finite chance are dropped).
    p_det
        Probability that the GEX library sees a genuine molecule's own transcript: a number, an
        ``(estimate, standard error)`` pair, or a mapping with keys ``p_det`` and ``se`` (as returned by
        :func:`spaclone.io.p_det_from_saturation`). Its standard error widens the interval of ``ρ``: by a
        Berger–Boos union over a ``1 − α/10`` interval of ``p_det`` (``"ipw"``, ``"exact"``) or by profiling a
        normal likelihood for its estimate (``"ml"``).
    conf_level
        Two-sided level of the intervals.
    method
        ``"ipw"`` (default): the plain mean of ``θ_i`` over records, ``θ̂ = mean[(y_i − c_i) / (1 − c_i)]``
        (inverse-chance weighting; a hit counts 1 whatever its chance), with a score interval (see Notes); records
        whose chance exceeds ``chance_cap`` carry almost no information and are left out (``n_capped``).
        ``"exact"``: the moment
        estimate ``θ̂ = (H − Σc_i) / (N − Σc_i)`` (truncated at 0) with the interval from inverting the exact
        Poisson-binomial distribution of the hits under ``π_i(θ)`` (Clopper–Pearson with record-specific chance),
        widened to the block score interval when ``blocks`` are given; it estimates a ``(1 − c_i)``-weighted mean of
        ``θ_i``. ``"ml"``: maximum likelihood with a profile-likelihood interval (biased towards 0 under the
        alternative). ``"plugin"``: the archived estimate, ``θ = (c − c̄) / (1 − c̄)`` with the concordance ``c``, the
        mean chance ``c̄`` and Clopper–Pearson bounds of ``c`` (truncated at 0), kept to reproduce archived numbers
        (``p_det`` is then used without its standard error, and a deficit does not suppress the bounds).
    blocks
        Per-record block label (e.g. a 100-µm square): with ``variance="cluster"`` the score variance is the larger
        of the Bernoulli variance and the block (cluster-robust) variance, and quantiles are t with (blocks − 1)
        degrees of freedom. ``None``: Bernoulli variance only.
    variance
        ``"cluster"`` (default; needs ``blocks``) or ``"binomial"`` (records independent, the 1.0.0.dev0 interval).
    chance_design
        The ``design`` returned by :meth:`GexLookup.evaluate_empirical` for these records (``same_spot`` passes it):
        the sampling variance of the empirical chances (finite reference spots, shared by neighbouring records) is
        added to the score variance. ``None``: chances taken as known.
    chance_cap
        ``"ipw"``: records with a larger chance are left out of the estimate (a spot that holds the gene by chance
        almost surely carries no information on its molecule, and its weight ``1 / (1 − c)`` would dominate the
        variance); ``n_capped`` counts them, and the estimate then refers to the other records.

    Returns
    -------
    dict
        ``n``, ``hits``, ``expected`` (= Σ c_i, the null mean of the hits), ``null_sd`` (sd of the hits under
        ``ρ = 0``, block-inflated when larger), ``ratio`` (hits / expected), ``concordance``, ``chance`` (mean),
        ``z``, ``p_value`` (more hits than chance: the larger of the exact Poisson-binomial ``P(hits ≥ observed)`` and
        the block p-value), ``p_value_exact``, ``p_value_block``, ``phi`` (block variance / Bernoulli variance under
        ``ρ = 0``), ``n_blocks``, ``deficit`` (fewer hits than chance beyond the interval: the chance model
        over-states the chance), ``deficit_p``, ``rho_pdet``, ``rho_pdet_lo``, ``rho_pdet_hi`` (NaN when
        ``deficit``), ``bound_reliable``, ``bound_note``, ``n_capped`` and, when ``p_det`` is given, ``p_det``,
        ``p_det_se``, ``rho``, ``rho_lo``, ``rho_hi``, ``rho_capped``.

    Notes
    -----
    Null model: ``ρ = 0``; each record is a hit with probability ``c_i``, independently between blocks; the exact
    p-value treats records as independent, the block p-value only blocks. Model: ``π_i = c_i + θ_i (1 − c_i)``
    (Rogan–Gladen with record-specific specificity). ``"ipw"`` and ``"exact"`` intervals are
    ``{θ: |S(θ)| <= q · sd(S(θ))}`` for the score ``S(θ) = Σ (u_i(θ) − θ d_i)`` of their estimating equation
    (``u = (y − c*)/(1 − c*), d = 1`` and ``u = y − c, d = 1 − c``), ``sd`` the larger of the Bernoulli and the block
    standard deviation at ``θ`` (Wilson-type, so the interval does not collapse at 0); ``"exact"`` also takes the
    union with the exact Poisson-binomial interval. A **deficit** (``deficit_p``, the larger of
    ``P_0(H <= h)`` and the block lower p-value, at most ``(1 − conf_level) / 2``) means that the chance model
    over-states the chance; then ``rho_pdet``, the bounds and ``rho`` are NaN and ``bound_reliable`` is False,
    because a deficit is evidence against the chance model, not evidence for ``ρ = 0``. ``ρ`` is capped at 1
    (``rho_capped`` flags data that would need ``ρ > 1``).

    References
    ----------
    Rogan & Gladen 1978, doi:10.1093/oxfordjournals.aje.a112510; Lang & Reiczigel 2014,
    doi:10.1016/j.prevetmed.2013.09.015; Reiczigel et al. 2010, doi:10.1017/S0950268810000385; Clopper & Pearson
    1934, doi:10.1093/biomet/26.4.404; Berger & Boos 1994, doi:10.1080/01621459.1994.10476784; Hong 2013,
    doi:10.1016/j.csda.2012.10.006; Horvitz & Thompson 1952, doi:10.1080/01621459.1952.10483446; Wilson 1927,
    doi:10.1080/01621459.1927.10502953; Liang & Zeger 1986, doi:10.1093/biomet/73.1.13.
    """
    y = np.asarray(hit, dtype=bool).ravel()
    c = np.asarray(chance, dtype=float).ravel()
    if y.shape != c.shape:
        raise ValueError("hit and chance must have the same shape")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must lie in (0, 1)")
    if method not in ("ipw", "exact", "ml", "plugin"):
        raise ValueError("method must be 'ipw', 'exact', 'ml' or 'plugin'")
    if variance not in ("cluster", "binomial"):
        raise ValueError("variance must be 'cluster' or 'binomial'")
    if not 0 < chance_cap < 1:
        raise ValueError("chance_cap must lie in (0, 1)")
    ok = np.isfinite(c)
    codes, n_blocks = _block_codes(None if blocks is None else np.asarray(blocks).ravel()[ok], int(ok.sum()))
    design = chance_design.subset(np.flatnonzero(ok)) if chance_design is not None else None
    y, c = y[ok], np.clip(c[ok], 0.0, 1.0)
    pd_ = _parse_p_det(p_det)
    n = int(len(y))
    h = int(y.sum())
    e = float(c.sum())
    cluster = variance == "cluster" and blocks is not None
    v_ref1 = design.variance(np.ones(n)) if (design is not None and n) else 0.0
    mo0 = _Moments(y, c, codes, n_blocks, weighted=False, cluster=cluster, vref=lambda t: (1.0 - t) ** 2 * v_ref1)
    vm0, vc0 = mo0.variances(0.0) if n else (np.nan, np.nan)
    v_null = (max(vm0, vc0) if np.isfinite(vc0) else vm0) + v_ref1
    pb_up = poisson_binomial_sf(c, h) if n else np.nan
    pb_lo = poisson_binomial_cdf(c, h) if n else np.nan
    bl_up, bl_lo, _ = mo0.null_tails() if (n and (cluster or v_ref1 > 0)) else (np.nan, np.nan, np.nan)
    out: dict[str, Any] = {
        "n": n,
        "hits": h,
        "expected": e,
        "null_sd": float(np.sqrt(v_null)) if n else np.nan,
        "ratio": h / e if e > 0 else np.nan,
        "concordance": h / n if n else np.nan,
        "chance": e / n if n else np.nan,
        "z": (h - e) / np.sqrt(v_null) if (n and v_null > 0) else np.nan,
        "p_value": float(np.nanmax([pb_up, bl_up])) if n else np.nan,
        "p_value_exact": pb_up,
        "p_value_block": bl_up,
        "phi": (vc0 / vm0) if (cluster and vm0 > 0) else np.nan,
        "n_blocks": n_blocks if cluster else 0,
        "chance_sd": float(np.sqrt(v_ref1)),
    }
    alpha = 1 - conf_level
    deficit_p = float(np.nanmax([pb_lo, bl_lo])) if n else np.nan
    deficit = bool(n and deficit_p <= alpha / 2)
    out.update({"deficit": deficit, "deficit_p": deficit_p})
    keys_rho = ("p_det", "p_det_se", "rho", "rho_lo", "rho_hi", "rho_capped")
    if n == 0:
        out.update({"rho_pdet": np.nan, "rho_pdet_lo": np.nan, "rho_pdet_hi": np.nan, "bound_reliable": False,
                    "bound_note": "no records", "n_capped": 0})  # fmt: skip
        if pd_ is not None:
            out.update(dict(zip(keys_rho, (pd_[0], pd_[1], np.nan, np.nan, np.nan, False), strict=True)))
        return out
    note = (
        f"unreliable: {h} hits against {e:.1f} expected (ratio {h / e:.3g}, deficit p = {deficit_p:.2g}): the "
        "chance model over-states the chance here, so no bound is given; see shift_test or a placebo"
        if deficit
        else "ok"
    )
    out.update({"bound_reliable": not deficit, "bound_note": note, "n_capped": 0})
    if method == "plugin":
        cbar = float(c.mean())
        conc = float(y.mean())
        hi = 1.0 if h == n else float(stats.beta.ppf(1 - alpha / 2, h + 1, n - h))
        lo = float(stats.beta.ppf(alpha / 2, h, n - h + 1)) if h > 0 else 0.0

        def plug(v: float, pdet: float) -> float:
            d = pdet * (1.0 - cbar)
            return max(0.0, (v - cbar) / d) if d > 0 else np.nan

        out.update({"rho_pdet": plug(conc, 1.0), "rho_pdet_lo": plug(lo, 1.0), "rho_pdet_hi": plug(hi, 1.0)})
        out["p_value"] = pb_up  # the archive's p-value
        if deficit:
            out["bound_note"] = note.replace("so no bound is given", "so the archived bound is not reliable")
        if pd_ is not None:
            p = pd_[0]
            rr = plug(conc, p)
            vals = (p, pd_[1], rr, plug(lo, p), plug(hi, p), bool(rr > 1))
            out.update(dict(zip(keys_rho, vals, strict=True)))
        return out

    def finish_rho(th: float, th_lo: float, th_hi: float, se: float, gamma: float) -> None:
        if pd_ is None:
            return
        p_hat = pd_[0]
        zg = float(stats.norm.ppf(1 - gamma / 2)) if gamma > 0 else 0.0
        p_lo, p_hi = max(p_hat - zg * se, 1e-12), min(p_hat + zg * se, 1.0)
        r = th / p_hat
        vals = (p_hat, se, min(r, 1.0), min(th_lo / p_hi, 1.0), min(th_hi / p_lo, 1.0), bool(r > 1))
        out.update(dict(zip(keys_rho, vals, strict=True)))

    if method in ("ipw", "exact"):
        se = pd_[1] if pd_ is not None else 0.0
        gamma = alpha / 10.0 if se > 0 else 0.0
        tail = (alpha - gamma) / 2.0
        if method == "ipw":
            keep = c <= chance_cap
            out["n_capped"] = int((~keep).sum())
            kc = codes[keep]
            kc, nb = (pd.factorize(kc)[0].astype(np.int64), len(np.unique(kc))) if keep.any() else (kc, 0)
            v_ref_ipw = 0.0
            if design is not None and keep.any():
                yk, ck_ = y[keep], c[keep]
                v_ref_ipw = design.subset(np.flatnonzero(keep)).variance((1.0 - yk) / (1.0 - ck_) ** 2)
            mo = _Moments(y[keep], c[keep], kc, nb, weighted=True, cluster=cluster, vref=lambda t: v_ref_ipw)
            raw = mo.estimate() if keep.any() else np.nan
            th_lo, th_hi = mo.interval(tail, raw) if np.isfinite(raw) else (np.nan, np.nan)
            th = max(0.0, raw) if np.isfinite(raw) else np.nan
        else:
            th = max(0.0, (h - e) / (n - e)) if n > e else 0.0
            th_lo, th_hi, _ = _exact_interval(c, h, tail)
            if cluster:
                lo2, hi2 = mo0.interval(tail, th)
                th_lo, th_hi = min(th_lo, lo2), max(th_hi, hi2)
        th_lo, th_hi = min(th_lo, th), max(th_hi, th)
        if deficit:
            th, th_lo, th_hi = np.nan, np.nan, np.nan
        out.update({"rho_pdet": th, "rho_pdet_lo": th_lo, "rho_pdet_hi": th_hi})
        finish_rho(th, th_lo, th_hi, se, gamma)
        return out
    crit = float(stats.chi2.ppf(conf_level, 1)) / 2.0
    c_hit = c[y]
    n0 = n - h

    def ll(t: float) -> float:
        return _loglik(t, c_hit, n0)

    th = _mle(c_hit, n0)
    th_lo, th_hi = _profile_interval(ll, th, crit)
    if deficit:
        th, th_lo, th_hi = np.nan, np.nan, np.nan
    out.update({"rho_pdet": th, "rho_pdet_lo": th_lo, "rho_pdet_hi": th_hi})
    if pd_ is None:
        return out
    p_hat, se = pd_
    if se == 0 or deficit:
        r = th / p_hat
        vals = (p_hat, se, min(r, 1.0), min(th_lo / p_hat, 1.0), min(th_hi / p_hat, 1.0), bool(r > 1))
        out.update(dict(zip(keys_rho, vals, strict=True)))
        return out
    p_min = 1e-9

    def prof(rho: float) -> float:
        """Profile log-likelihood of rho, maximising over p_det with a normal likelihood for its estimate."""
        if rho <= 0:
            return ll(0.0)
        p_max = min(1.0, 1.0 / rho)

        def neg(p: float) -> float:
            return -(ll(rho * p) - 0.5 * ((p - p_hat) / se) ** 2)

        res = optimize.minimize_scalar(neg, bounds=(p_min, p_max), method="bounded", options={"xatol": 1e-12})
        return -float(res.fun)

    if th <= p_hat:
        r_hat = th / p_hat
    else:
        res = optimize.minimize_scalar(
            lambda r: -prof(r), bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12}
        )
        r_hat = float(res.x)
    r_lo, r_hi = _profile_interval(prof, r_hat, crit)
    out.update(dict(zip(keys_rho, (p_hat, se, r_hat, r_lo, r_hi, bool(th > p_hat)), strict=True)))
    return out


# ------------------------------------------------------------------------------------------------- spot records


def _resolve_genes(
    constant_genes: Mapping[str, Sequence[str]] | None, loci: Sequence[str], gex: GexLookup
) -> tuple[dict[str, tuple[str, ...]], str]:
    """Constant genes per tested locus; ``None`` picks the species table whose genes are loaded in ``gex``."""
    if constant_genes is not None:
        return {str(k): tuple(v) for k, v in constant_genes.items()}, "given"
    loaded = set(gex.genes)
    best, name = None, ""
    for label, table in (("mouse", MOUSE_CONSTANT_GENES), ("human", HUMAN_CONSTANT_GENES)):
        hit = sum(g in loaded for lc in loci for g in table.get(str(lc), ()))
        if hit and (best is None or hit > best[0]):
            best, name = (hit, table), label
    if best is None:
        raise ValueError(
            "constant_genes=None: none of the mouse or human constant-region genes of the loci "
            f"{sorted(map(str, loci))} are loaded in the GexLookup (loaded: {sorted(loaded)}); pass constant_genes"
        )
    table = best[1]
    out = {}
    for lc in loci:
        gs = tuple(g for g in table.get(str(lc), ()) if g in loaded)
        if gs:
            out[str(lc)] = gs
    return out, name


def _spot_records(
    air: pd.DataFrame,
    gex: GexLookup,
    *,
    genes: Mapping[str, Sequence[str]],
    tile_um: float,
    chance: ChanceModel,
    leave_own_out: bool,
    background: Background,
    sink_min_clones: int,
    zone_key: str | None,
    spot_key: str,
    x_key: str,
    y_key: str,
    locus_key: str,
    clone_key: str,
    empirical: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One record per (spot, locus) with the GEX lookup; see :func:`same_spot`."""
    s, spots = pd.factorize(air[spot_key], sort=False, use_na_sentinel=False)
    lo, loci = pd.factorize(air[locus_key].astype(object), sort=True, use_na_sentinel=False)
    if air[clone_key].isna().any():
        raise ValueError(f"column {clone_key!r} has missing clone ids")
    c, clones = pd.factorize(air[clone_key], sort=False, use_na_sentinel=False)
    s, lo, c = s.astype(np.int64), lo.astype(np.int64), c.astype(np.int64)
    n_spots, n_loci, n_clones = len(spots), len(loci), max(len(clones), 1)
    if n_spots * max(n_loci, 1) * n_clones < 2**62:
        key = pd.unique((s * n_loci + lo) * n_clones + c)
        ts, tl = key // (n_loci * n_clones), (key // n_clones) % n_loci
    else:  # pragma: no cover - huge tables
        trip = pd.DataFrame({"s": s, "l": lo, "c": c}).drop_duplicates(ignore_index=True)
        ts, tl = trip["s"].to_numpy(), trip["l"].to_numpy()
    mult_all = np.bincount(ts, minlength=n_spots)
    mult_sl = np.bincount(ts * n_loci + tl, minlength=n_spots * n_loci)
    sl = s * n_loci + lo
    rec, first, inv = np.unique(sl, return_index=True, return_inverse=True)
    n_rec = np.bincount(inv.ravel(), minlength=len(rec))
    rs, rl = rec // n_loci, rec % n_loci
    if zone_key:
        zc = pd.factorize(air[zone_key], sort=False, use_na_sentinel=False)[0]
        zs = pd.DataFrame({"s": s, "z": zc}).drop_duplicates()
        if zs["s"].duplicated().any():
            raise ValueError(f"zone column {zone_key!r} varies inside a spot; zones must be spot-level")
        zcol = air[zone_key]
        if isinstance(zcol.dtype, pd.CategoricalDtype):
            zone = pd.Categorical(zcol.to_numpy()[first], categories=zcol.dtype.categories, ordered=zcol.dtype.ordered)
        else:
            zone = zcol.to_numpy()[first]
    else:
        zone = np.full(len(rec), "all", dtype=object)
    x = air[x_key].to_numpy(dtype=float)[first]
    y = air[y_key].to_numpy(dtype=float)[first]
    out = pd.DataFrame(
        {
            spot_key: np.asarray(spots, dtype=object)[rs],
            locus_key: np.asarray(loci, dtype=object)[rl],
            x_key: x,
            y_key: y,
            "zone": zone,
            "n_records": n_rec,
            "n_clones": mult_sl[rec],
            "mult_all": mult_all[rs],
        }
    )
    nrow = len(out)
    cols = {
        "m": np.zeros(nrow, dtype=np.int64),
        "own_mids": np.zeros(nrow, dtype=np.int64),
        "tile_total": np.full(nrow, np.nan),
        "tile_set": np.full(nrow, np.nan),
        "q_tile": np.full(nrow, np.nan),
        "chance": np.full(nrow, np.nan),
    }
    designs: dict[str, Any] = {}
    if chance == "empirical":
        cols.update(
            {"ref_n": np.zeros(nrow), "pool_um": np.full(nrow, np.nan), "stratum": np.full(nrow, -1, dtype=np.int64),
             "design_row": np.full(nrow, -1, dtype=np.int64)}
        )  # fmt: skip
    tested = np.zeros(nrow, dtype=bool)
    for j, name in enumerate(loci):
        gs = genes.get(str(name))
        if not gs:
            logger.info("same_spot: no constant genes for locus %r; its records are not tested", name)
            continue
        sel = rl == j
        excl = (x[sel], y[sel]) if background == "receptor_free" else None
        if chance == "empirical":
            ev = gex.evaluate_empirical(x[sel], y[sel], list(gs), exclude=excl, tile_um=tile_um, **(empirical or {}))
            for k in ("ref_n", "pool_um", "stratum"):
                cols[k][sel] = ev[k]
            cols["design_row"][sel] = np.arange(int(sel.sum()))
            designs[str(name)] = ev["design"]
            cols["m"][sel], cols["own_mids"][sel], cols["chance"][sel] = ev["m"], ev["own"], ev["chance"]
            tested[sel] = True
            continue
        ev = gex.evaluate(
            x[sel], y[sel], list(gs), tile_um=tile_um, chance=chance, leave_own_out=leave_own_out, exclude=excl
        )
        cols["m"][sel], cols["own_mids"][sel] = ev["m"], ev["own"]
        cols["tile_total"][sel], cols["tile_set"][sel] = ev["tile_total"], ev["tile_set"]
        cols["q_tile"][sel], cols["chance"][sel] = ev["q"], ev["chance"]
        tested[sel] = True
    for k, v in cols.items():
        out[k] = v
    out["hit"] = out["own_mids"].to_numpy() > 0
    out["tested"] = tested
    out["occupied"] = out["m"].to_numpy() > 0
    out["singleton"] = (out["n_clones"].to_numpy() == 1) & (out["mult_all"].to_numpy() < sink_min_clones)
    return out, designs


@dataclass(frozen=True)
class SameSpotResult(TestResult):
    """Result of :func:`same_spot`.

    Attributes
    ----------
    by_multiplicity
        Hits / expected by locus × zone × clones-per-spot class, counted per spot (``level="spot"``, the independent
        observation) and per record (``level="record"``, which lets barcode sinks count many times).
    records
        One row per (spot, locus): position, ``zone``, ``n_records``, clones of the locus on the spot (``n_clones``),
        clones of all loci (``mult_all``), GEX molecules ``m``, constant-gene molecules ``own_mids``, ``chance``,
        ``hit``, ``tested``, ``occupied``, ``singleton``, ``block`` (the block of the block variance) and, by chance
        model, the reference spots used (``ref_n``), the half-side of their block of tiles (``pool_um``) and the
        stratum code (empirical), or the tile's molecules (``tile_total``,
        ``tile_set``, ``q_tile``; analytic).
    """

    by_multiplicity: pd.DataFrame = field(default_factory=pd.DataFrame)
    records: pd.DataFrame = field(default_factory=pd.DataFrame)


def _zone_order(z: pd.Series) -> list:
    if isinstance(z.dtype, pd.CategoricalDtype):
        return [v for v in z.cat.categories if (z == v).any()]
    return sorted(pd.unique(z), key=str)


def same_spot(
    air: pd.DataFrame,
    gex: GexLookup,
    *,
    constant_genes: Mapping[str, Sequence[str]] | None = None,
    singleton_only: bool = True,
    sink_min_clones: int = 20,
    tile_um: float = 25.0,
    chance: ChanceModel = "empirical",
    background: Background = "receptor_free",
    leave_own_out: bool = False,
    strata: str | None = None,
    context_um: float | None = 4.0,
    m_bins: Sequence[float] = M_BINS,
    context_bins: Sequence[float] = CONTEXT_BINS,
    min_reference: int = 10,
    max_pool: int = 8,
    p_det: float | tuple[float, float] | Mapping[str, float] | None = None,
    zones: str | None = None,
    estimator: Estimator = "ipw",
    variance: Variance = "cluster",
    block_um: float = 100.0,
    legacy: bool = False,
    conf_level: float = 0.95,
    multiplicity_bins: Sequence[float] = MULTIPLICITY_BINS,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> SameSpotResult:
    """Do receptor molecules sit on spots whose GEX holds their own constant-region transcript more often than chance?

    Parameters
    ----------
    air
        Receptor table on the same spot grid as ``gex`` (e.g. Stereo-seq DNB coordinates × 0.5 µm); one row per
        molecule or per (spot, clone) unit.
    gex
        :class:`GexLookup` with the constant genes of every tested locus loaded (and the ``strata`` labels, if used).
    constant_genes
        ``{locus: genes}``; a hit is any molecule of these genes on the spot. Loci without an entry are not tested.
        ``None`` uses :data:`MOUSE_CONSTANT_GENES` or :data:`HUMAN_CONSTANT_GENES`, whichever has genes loaded in
        ``gex`` (restricted to the loaded genes).
    singleton_only
        Estimate on spots holding exactly one clone of the locus and fewer than ``sink_min_clones`` clones in all
        (several clones on one 0.5 µm DNB cannot all be its cells' own; a sink would dominate the hits).
    sink_min_clones
        Clones per spot (all loci) from which a spot is a barcode sink (see :func:`spaclone.pp.barcode_sinks`).
    tile_um
        Side of the tiles (a multiple of the spot pitch): the smallest block of the empirical chance's reference
        spots, or the background of the analytic chances.
    chance
        ``"empirical"`` (default): the cDNA-weighted share of comparable reference spots holding the constant gene
        (same class of GEX molecules, ``strata`` label and ``context_um`` class, in the smallest block of tiles
        around the record with ``min_reference`` of them; see :meth:`GexLookup.evaluate_empirical`). It assumes that,
        within these strata, a swapped molecule lands in proportion to the spot's *observed* GEX molecules; if swaps
        are drawn to cDNA the GEX library missed (bursty Ig RNA at low saturation), hits exceed this chance without
        any located molecule (an upward bias: upper bounds stay conservative, a "signal" needs :func:`shift_test`
        and the clone-level tests).
        ``"conditional"``: analytic within-tile chance ``1 − C(T − S, m)/C(T, m)`` (the 1.0.0.dev0 default);
        ``"binomial"``: ``1 − (1 − S/T)^m`` (archived). ``T`` and ``S`` are the background molecules of the tile and
        the constant-gene molecules among them. The analytic chances assume constant-region RNA is mixed within the
        tile, which it is not (see the module notes).
    background
        Which spots serve as background / reference: ``"receptor_free"`` (default) the spots that carry no record of
        the locus (plus, for the analytic chances, the record's own spot); ``"all"`` every spot (archived). With
        ``"all"``, the own transcripts of located molecules raise the chance of every record near them and bias
        ``ρ`` towards 0.
    leave_own_out
        Analytic chances: remove the record's own spot from the background (see :meth:`GexLookup.evaluate`); with
        ``chance="binomial"`` a hit in a sparse tile then gets a chance of zero (warned).
    strata
        Empirical chance: name of a spot label set of ``gex`` (:meth:`GexLookup.add_labels`, e.g. ``"cell_type"``;
        :meth:`GexLookup.from_anndata` loads them with ``labels=``). Reference spots must carry the record spot's
        label. A located molecule sits on its own B / plasma (T) cell, whose other constant-region RNA counts as
        chance; without labels its chance comes from a mixture of cell types and ``ρ`` is over-estimated.
    context_um
        Empirical chance (ignored by the analytic chances): also match the number of constant-gene molecules on the
        other spots within this distance (classes ``context_bins``; default 4 µm, about one cell radius), a label-free proxy of "on or next to a cell
        that expresses the gene": a located molecule's spot is surrounded by its cell's other constant-region RNA,
        which counts as chance. ``None``: no context (θ is then over-estimated when molecules are located).
    m_bins, context_bins, min_reference, max_pool
        Empirical chance: classes of GEX molecules per spot and of context molecules, the reference spots needed and
        the largest block of tiles searched (half-side ``(max_pool + 0.5) · tile_um``; its reference spots are used
        even when fewer than ``min_reference``, and the sampling variance of the chances enters the interval).
    p_det
        GEX detection probability of the own transcript (number, ``(estimate, se)`` or the dict from
        :func:`spaclone.io.p_det_from_saturation`); adds ``ρ`` and its interval. ``ρ·p_det`` needs no ``p_det``.
    zones
        Column of ``air`` with a spot-level zone label (e.g. from :func:`spaclone.qc.tissue_zones`); results are
        given per locus × zone. ``None``: one zone ``"all"``.
    estimator
        ``"ipw"`` (default): the plain mean of ``θ_i`` over the tested records (inverse-chance weighting) with a
        score interval; ``"exact"``: Rogan–Gladen moment estimate with the exact Poisson-binomial interval
        (a ``(1 − c)``-weighted mean); ``"ml"``: maximum likelihood (biased towards 0 under the alternative). See
        :func:`located_fraction`.
    variance
        ``"cluster"`` (default): block (cluster-robust) variance over squares of ``block_um``, never smaller than the
        Bernoulli variance; ``"binomial"``: records independent (the 1.0.0.dev0 interval).
    block_um
        Side of the blocks of the block variance (larger than a cell and than the reference blocks).
    legacy
        Reproduce the archived Stereo-XCR-seq analysis (workflow B2): ``chance="binomial"``, ``background="all"``,
        ``leave_own_out=False``, ``variance="binomial"`` (these arguments are overridden) and the plug-in estimate
        with Clopper–Pearson bounds (see :func:`located_fraction`); ``deficit`` is still reported.
    conf_level
        Two-sided level of the intervals.
    multiplicity_bins
        Edges of the clones-per-spot classes of ``by_multiplicity``.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    SameSpotResult
        ``table``: one row per locus × zone with ``n_spots`` (records used), ``statistic`` (= ``hits``, the number of
        records whose spot holds the constant gene), ``expected`` (its null mean Σ c_i), ``null_sd``, ``ratio``,
        ``z``, ``p_value`` (one-sided, more hits than chance; the larger of the exact and the block p-value),
        ``p_value_exact``, ``p_value_block`` (block and reference-sampling variance), ``q_value`` (BH over rows),
        ``phi`` (over-dispersion), ``n_blocks``, ``chance_sd``,
        ``concordance``, ``chance``, ``rho_pdet`` / ``rho_pdet_lo`` / ``rho_pdet_hi`` (``ρ·p_det`` and interval;
        NaN when ``deficit``), ``deficit`` (significantly fewer hits than chance: the chance model over-states the
        chance), ``deficit_p``, ``bound_reliable``, ``bound_note``, ``n_capped``, ``rho`` columns when ``p_det`` is
        given, ``n_spots_occupied`` (all GEX-occupied spot records of the locus in the zone), ``n_no_chance``
        (records without a chance model), ``n_zero_chance_hits`` and the naive per-record contrast
        ``naive_records``, ``naive_hits``, ``naive_expected``, ``naive_ratio`` (every record counted, sinks and
        multi-clone spots included). ``by_multiplicity`` and ``records`` as in :class:`SameSpotResult`.

    Notes
    -----
    Null model: ``ρ = 0``, i.e. each record's spot holds no transcript of its own molecule. Empirical chance: a swapped
    molecule took the barcode of a random captured cDNA molecule, so its spot is drawn with probability proportional
    to the spot's GEX molecules among the reference spots of its stratum (GEX-occupied, receptor-free, same class of
    GEX molecules and, if given, the same label and context class, in the smallest block of tiles around it with
    ``min_reference`` such spots); ``c_i`` is the cDNA-weighted share of those spots that hold a constant-gene
    molecule. Analytic chances: the spot's ``m_i`` GEX molecules are a random subset of the tile's background
    molecules (``"conditional"``) or independent draws with its gene share (``"binomial"``). The test conditions on
    the spots, their GEX molecules and the reference spots; records are independent between blocks of ``block_um``
    (``variance="cluster"``; with ``"binomial"`` all records are independent). Under the alternative
    ``π_i = c_i + θ_i (1 − c_i)``. One record per spot and locus: molecules on one spot share one GEX lookup and are
    not independent. The test is molecule-level: constant-region-homologous template switches ("right isotype, wrong
    clone") count as located; whether clone identity follows position needs :func:`spaclone.qc.colocalisation` and
    :func:`spaclone.qc.pairing`. A significant deficit of hits (``deficit``) means the chance model is wrong for these
    data; no bound is then reported. Compare with :func:`shift_test` and with an empirical placebo
    (:func:`spaclone.nulls.placebo`).

    References
    ----------
    Rogan & Gladen 1978, doi:10.1093/oxfordjournals.aje.a112510; Lang & Reiczigel 2014,
    doi:10.1016/j.prevetmed.2013.09.015; Hong 2013, doi:10.1016/j.csda.2012.10.006; Horvitz & Thompson 1952,
    doi:10.1080/01621459.1952.10483446; Liang & Zeger 1986, doi:10.1093/biomet/73.1.13; Künsch 1989,
    doi:10.1214/aos/1176347265; Costes et al. 2004, doi:10.1529/biophysj.103.038422.
    """
    if sink_min_clones < 2:
        raise ValueError("sink_min_clones must be >= 2")
    if chance not in ("empirical", "conditional", "binomial"):
        raise ValueError(f"chance must be 'empirical', 'conditional' or 'binomial', got {chance!r}")
    if background not in ("receptor_free", "all"):
        raise ValueError(f"background must be 'receptor_free' or 'all', got {background!r}")
    if legacy:
        chance, background, leave_own_out, variance = "binomial", "all", False, "binomial"
    loo = bool(leave_own_out)
    if estimator not in ("ipw", "exact", "ml"):
        raise ValueError(f"estimator must be 'ipw', 'exact' or 'ml', got {estimator!r}")
    if variance not in ("cluster", "binomial"):
        raise ValueError(f"variance must be 'cluster' or 'binomial', got {variance!r}")
    if not block_um > 0:
        raise ValueError("block_um must be positive")
    if chance != "empirical" and strata is not None:
        raise ValueError("strata applies to chance='empirical' only")
    if chance != "empirical":
        context_um = None  # the analytic chances have no context classes
    method = "plugin" if legacy else estimator
    req = [spot_key, x_key, y_key, locus_key, clone_key] + ([zones] if zones else [])
    S.validate_air(air, required=req, numeric=(x_key, y_key))
    loci_present = sorted(pd.unique(air[locus_key].astype(object)), key=str)
    genes, species = _resolve_genes(constant_genes, loci_present, gex)
    emp = {"m_bins": tuple(m_bins), "strata": strata, "context_um": context_um, "context_bins": tuple(context_bins),
           "min_reference": int(min_reference), "max_pool": int(max_pool)}  # fmt: skip
    rec, designs = _spot_records(
        air,
        gex,
        genes=genes,
        tile_um=tile_um,
        chance=chance,
        leave_own_out=loo,
        background=background,
        sink_min_clones=sink_min_clones,
        zone_key=zones,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        locus_key=locus_key,
        clone_key=clone_key,
        empirical=emp,
    )
    bx = np.floor(rec[x_key].to_numpy(dtype=float) / block_um).astype(np.int64)
    by = np.floor(rec[y_key].to_numpy(dtype=float) / block_um).astype(np.int64)
    rec["block"] = (bx - bx.min() if len(bx) else bx) * (1 << 31) + (by - by.min() if len(by) else by)
    use = rec[(rec["tested"] & rec["occupied"]).to_numpy()]
    finite = np.isfinite(use["chance"].to_numpy())
    zero_hits = int((use["hit"].to_numpy() & (use["chance"].to_numpy() == 0)).sum())
    if zero_hits and chance != "empirical":
        warnings.warn(
            f"same_spot: {zero_hits} hit(s) have a chance of zero (tiles with no other constant-gene molecule); "
            "they count as impossible under the null and inflate the evidence for located molecules. Use "
            "leave_own_out=False (the default) or larger tiles.",
            ZeroChanceWarning,
            stacklevel=2,
        )
    rows = []
    zone_levels = _zone_order(rec["zone"])
    for locus in sorted(pd.unique(use[locus_key]), key=str):
        ul = use[(use[locus_key] == locus).to_numpy()]
        fl = np.isfinite(ul["chance"].to_numpy())
        for zone in zone_levels:
            zmask = (ul["zone"] == zone).to_numpy()
            if not zmask.any():
                continue
            r = ul[zmask & fl]
            pick = r[r["singleton"].to_numpy()] if singleton_only else r
            lf = located_fraction(
                pick["hit"].to_numpy(),
                pick["chance"].to_numpy(),
                p_det=p_det,
                conf_level=conf_level,
                method=method,
                blocks=pick["block"].to_numpy() if variance == "cluster" else None,
                variance=variance,
                chance_design=designs[str(locus)].subset(pick["design_row"].to_numpy())
                if str(locus) in designs
                else None,
            )
            w = r["n_records"].to_numpy(dtype=float)
            nh = float((w * r["hit"].to_numpy()).sum())
            ne = float((w * r["chance"].to_numpy()).sum())
            n_used = lf.pop("n")
            row = {
                locus_key: locus,
                "zone": zone,
                "n_spots": n_used,
                "statistic": lf["hits"],
                **lf,
                "n_spots_occupied": int(zmask.sum()),
                "n_no_chance": int((zmask & ~fl).sum()),
                "n_zero_chance_hits": int((pick["hit"].to_numpy() & (pick["chance"].to_numpy() == 0)).sum()),
                "naive_records": int(w.sum()),
                "naive_hits": int(round(nh)),
                "naive_expected": ne,
                "naive_ratio": nh / ne if ne > 0 else np.nan,
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table.insert(table.columns.get_loc("p_value") + 1, "q_value", bh(table["p_value"].to_numpy()))
    by_m = _by_multiplicity(use[finite], zone_levels, multiplicity_bins, locus_key)
    summary: dict[str, Any] = {}
    if len(table):
        for lc, zn, ratio, hi, dfc in zip(
            table[locus_key], table["zone"], table["ratio"], table["rho_pdet_hi"], table["deficit"], strict=True
        ):
            summary[f"ratio[{lc}/{zn}]"] = ratio
            summary[f"rho_pdet_hi[{lc}/{zn}]"] = hi
            if dfc:
                summary[f"deficit[{lc}/{zn}]"] = True
    summary["occupied_share"] = float(rec.loc[rec["tested"], "occupied"].mean()) if rec["tested"].any() else np.nan
    params = {
        "constant_genes": {k: list(v) for k, v in genes.items()},
        "constant_genes_source": species,
        "singleton_only": singleton_only,
        "sink_min_clones": sink_min_clones,
        "tile_um": tile_um,
        "chance": chance,
        "background": background,
        "leave_own_out": loo,
        "strata": strata if chance == "empirical" else None,
        "context_um": context_um if chance == "empirical" else None,
        "m_bins": [float(v) for v in m_bins] if chance == "empirical" else None,
        "context_bins": [float(v) for v in context_bins] if (chance == "empirical" and context_um) else None,
        "min_reference": int(min_reference) if chance == "empirical" else None,
        "max_pool": int(max_pool) if chance == "empirical" else None,
        "estimator": method,
        "variance": variance,
        "block_um": float(block_um) if variance == "cluster" else None,
        "legacy": legacy,
        "p_det": _parse_p_det(p_det),
        "zones": zones,
        "conf_level": conf_level,
        "multiplicity_bins": list(multiplicity_bins),
        "null": _null_text(chance, loo, background, strata, context_um),
    }
    rec = rec.drop(columns=["design_row"], errors="ignore")
    return SameSpotResult(
        name="same_spot", table=table, summary=summary, params=params, by_multiplicity=by_m, records=rec
    )


def _by_multiplicity(
    bm: pd.DataFrame, zone_levels: list, multiplicity_bins: Sequence[float], locus_key: str
) -> pd.DataFrame:
    cols = ["level", locus_key, "zone", "multiplicity", "n", "hits", "expected", "ratio", "concordance", "chance"]
    cols += ["excess", "share_of_excess_in_zone"]
    if len(bm) == 0:
        return pd.DataFrame(columns=cols)
    labels = list(dict.fromkeys(bin_labels(np.asarray(multiplicity_bins[:-1], dtype=float), multiplicity_bins)))
    mcode = pd.Categorical(bin_labels(bm["n_clones"].to_numpy(), multiplicity_bins), categories=labels).codes
    lcode, lev_l = pd.factorize(bm[locus_key].astype(object), sort=True)
    zone_pos = {z: i for i, z in enumerate(zone_levels)}
    zcode = np.array([zone_pos[z] for z in bm["zone"].to_numpy()], dtype=np.int64)
    nl, nz, nm = len(lev_l), len(zone_levels), len(labels)
    cell = (lcode.astype(np.int64) * nz + zcode) * nm + mcode.astype(np.int64)
    hit = bm["hit"].to_numpy(dtype=float)
    chance = bm["chance"].to_numpy(dtype=float)
    parts = []
    for level, w in (("record", bm["n_records"].to_numpy(dtype=float)), ("spot", np.ones(len(bm)))):
        size = nl * nz * nm
        n = np.bincount(cell, weights=w, minlength=size)
        h = np.bincount(cell, weights=w * hit, minlength=size)
        e = np.bincount(cell, weights=w * chance, minlength=size)
        nz_cells = np.flatnonzero(n > 0)
        li, rem = np.divmod(nz_cells, nz * nm)
        zi, mi = np.divmod(rem, nm)
        g = pd.DataFrame(
            {
                "level": level,
                locus_key: np.asarray(lev_l, dtype=object)[li],
                "zone": [zone_levels[i] for i in zi],
                "multiplicity": pd.Categorical.from_codes(mi, categories=labels, ordered=True),
                "n": np.rint(n[nz_cells]).astype(np.int64),
                "hits": np.rint(h[nz_cells]).astype(np.int64),
                "expected": e[nz_cells],
            }
        )
        parts.append(g)
    by_m = pd.concat(parts, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        by_m["ratio"] = by_m["hits"] / by_m["expected"].where(by_m["expected"] > 0)
        by_m["concordance"] = by_m["hits"] / by_m["n"]
        by_m["chance"] = by_m["expected"] / by_m["n"]
    by_m["excess"] = by_m["hits"] - by_m["expected"]
    tot = by_m.groupby(["level", locus_key, "zone"], observed=True, sort=False)["excess"].transform("sum")
    by_m["share_of_excess_in_zone"] = by_m["excess"] / tot.where(tot != 0)
    return by_m[cols]


def shift_test(
    air: pd.DataFrame,
    gex: GexLookup,
    *,
    max_shift: int = 3,
    constant_genes: Mapping[str, Sequence[str]] | None = None,
    singleton_only: bool = True,
    sink_min_clones: int = 20,
    tile_um: float = 25.0,
    chance: ChanceModel = "empirical",
    background: Background = "receptor_free",
    leave_own_out: bool = False,
    strata: str | None = None,
    context_um: float | None = 4.0,
    m_bins: Sequence[float] = M_BINS,
    min_reference: int = 10,
    max_pool: int = 8,
    zones: str | None = None,
    zone: Any = None,
    max_records: int | None = None,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> TestResult:
    """Look the GEX up at displaced spots: a molecule that sits on its own transcript peaks only at offset (0, 0).

    Parameters
    ----------
    air, gex
        As in :func:`same_spot`.
    max_shift
        Offsets ``dx, dy`` from ``-max_shift`` to ``max_shift`` spots (grid steps of ``gex.spot_um``).
    constant_genes, singleton_only, sink_min_clones, tile_um, chance, background, leave_own_out, zones
        As in :func:`same_spot`; the records are the GEX-occupied spot records of :func:`same_spot`, and every spot
        carrying records of the locus is excluded from the background of displaced lookups too.
    strata, context_um, m_bins, min_reference, max_pool
        Empirical chance (as in :func:`same_spot`), evaluated at the displaced spots.
    zone
        Use only records of this zone (requires ``zones``).
    max_records
        Subsample at most this many records per locus (with ``rng``).
    rng
        Seed or generator for the subsample.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per locus × offset with ``dx``, ``dy`` (spots), ``dx_um``, ``dy_um``, ``n``, ``n_occupied``,
        ``p_occupied``, ``hits``, ``p_gene_given_occupied``, ``expected``, ``ratio`` (hits / chance-expected hits among
        occupied displaced spots with a chance model). ``summary``: per locus the ratio at (0, 0), the median ratio at
        the other offsets and ``p_value`` = rank of the (0, 0) ratio among all offsets (one-sided).

    Notes
    -----
    Null model: displaced lookups keep each record's local GEX context (tissue, cell density, the cDNA richness of
    receptor-carrying spots) but cannot meet the molecule's own transcript; if the (0, 0) excess over chance is no
    larger than at displaced offsets, the hits are not molecule-level. The offsets are not independent of each other
    (neighbouring DNBs share cells), so the rank p-value (smallest ``1 / (2·max_shift + 1)²``) is descriptive.

    References
    ----------
    Costes et al. 2004, doi:10.1529/biophysj.103.038422 (displacement controls for co-localisation).
    """
    if max_shift < 1:
        raise ValueError("max_shift must be >= 1")
    if zone is not None and zones is None:
        raise ValueError("zone requires zones (the zone column)")
    if chance not in ("empirical", "conditional", "binomial"):
        raise ValueError(f"chance must be 'empirical', 'conditional' or 'binomial', got {chance!r}")
    gen = as_rng(rng)
    req = [spot_key, x_key, y_key, locus_key, clone_key] + ([zones] if zones else [])
    S.validate_air(air, required=req, numeric=(x_key, y_key))
    loci_present = sorted(pd.unique(air[locus_key].astype(object)), key=str)
    genes, _ = _resolve_genes(constant_genes, loci_present, gex)
    emp = {"m_bins": tuple(m_bins), "strata": strata, "context_um": context_um, "min_reference": int(min_reference),
           "max_pool": int(max_pool)}  # fmt: skip
    rec, _ = _spot_records(
        air,
        gex,
        genes=genes,
        tile_um=tile_um,
        chance=chance,
        leave_own_out=leave_own_out,
        background=background,
        sink_min_clones=sink_min_clones,
        zone_key=zones,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        locus_key=locus_key,
        clone_key=clone_key,
        empirical=emp,
    )
    receptor_spots = {
        str(lc): (g[x_key].to_numpy(dtype=float), g[y_key].to_numpy(dtype=float))
        for lc, g in rec.groupby(locus_key, sort=False)
    }
    keep = rec["tested"] & rec["occupied"]
    if singleton_only:
        keep &= rec["singleton"]
    if zone is not None:
        keep &= rec["zone"] == zone
    rec = rec[keep.to_numpy()]
    offs = [(dx, dy) for dx in range(-max_shift, max_shift + 1) for dy in range(-max_shift, max_shift + 1)]
    rows, summary = [], {}
    for locus in sorted(pd.unique(rec[locus_key]), key=str):
        r = rec[(rec[locus_key] == locus).to_numpy()]
        if max_records is not None and len(r) > max_records:
            r = r.iloc[np.sort(gen.choice(len(r), size=max_records, replace=False))]
        x, y = r[x_key].to_numpy(dtype=float), r[y_key].to_numpy(dtype=float)
        gs = list(genes[str(locus)])
        ratios = {}
        excl = receptor_spots[str(locus)] if background == "receptor_free" else None
        for dx, dy in offs:
            if chance == "empirical":
                ev = gex.evaluate_empirical(
                    x + dx * gex.spot_um, y + dy * gex.spot_um, gs, exclude=excl, tile_um=tile_um, **emp
                )
            else:
                ev = gex.evaluate(
                    x + dx * gex.spot_um,
                    y + dy * gex.spot_um,
                    gs,
                    tile_um=tile_um,
                    chance=chance,
                    leave_own_out=leave_own_out,
                    exclude=excl,
                )
            occ = ev["m"] > 0
            fin = occ & np.isfinite(ev["chance"])
            h = int(ev["hit"][occ].sum())
            e = float(ev["chance"][fin].sum())
            ratio = float(ev["hit"][fin].sum()) / e if e > 0 else np.nan
            ratios[(dx, dy)] = ratio
            rows.append(
                {
                    locus_key: locus,
                    "dx": dx,
                    "dy": dy,
                    "dx_um": dx * gex.spot_um,
                    "dy_um": dy * gex.spot_um,
                    "n": len(r),
                    "n_occupied": int(occ.sum()),
                    "p_occupied": float(occ.mean()) if len(r) else np.nan,
                    "hits": h,
                    "p_gene_given_occupied": h / occ.sum() if occ.any() else np.nan,
                    "expected": e,
                    "ratio": ratio,
                }
            )
        others = np.array([v for k, v in ratios.items() if k != (0, 0)], dtype=float)
        r0 = ratios[(0, 0)]
        summary[f"ratio_at_0[{locus}]"] = r0
        summary[f"median_ratio_shifted[{locus}]"] = float(np.nanmedian(others)) if len(others) else np.nan
        summary[f"p_value[{locus}]"] = float(mc_pvalue(others, r0, alternative="greater")[0])
    params = {
        "max_shift": max_shift,
        "constant_genes": {k: list(v) for k, v in genes.items()},
        "singleton_only": singleton_only,
        "sink_min_clones": sink_min_clones,
        "tile_um": tile_um,
        "chance": chance,
        "background": background,
        "leave_own_out": leave_own_out,
        "strata": strata,
        "context_um": context_um,
        "zones": zones,
        "zone": zone,
        "max_records": max_records,
        "rng_entropy": rng_entropy(gen),
        "null": "displaced GEX lookups (same local context, cannot hit the molecule's own transcript)",
    }
    return TestResult(name="shift_test", table=pd.DataFrame(rows), summary=summary, params=params)
