"""Same-clone co-localisation: do records of one clone sit closer together than their labels alone predict?

A genuine spatial receptor assay places the records of one clone where that clone's cells are, so pairs of records
of the same clone are found at short distances more often than pairs drawn at random. If a record's position is
only a readout of where receptor RNA happened to be captured (barcode swapping, ambient RNA, template switching onto
a random barcode), clone identity is independent of position given the local amount of receptor RNA, and the share of
same-clone pairs at short range equals what label shuffling within depth strata gives. This is the technology-trust
discriminator of the archive (chapter 04, X5 test T3).

The statistic is the mark-equality version of the mark connection function (P(same clone | distance band)),
computed as a pooled pair share: among all pairs of units of the same clone (clones with ``min_clone_size`` to
``max_clone_size`` units), the share whose distance falls in each band.

The exchangeable unit must be about one cell. On beads or segmented cells a (spot, clone) unit is one; on sub-cellular
spots (Stereo-seq DNBs, 2 µm bins) one cell covers several spots, and permuting the labels of single spots treats the
spots of one cell as independent cells, which makes the null distribution too narrow (and the clone-size window count
spots instead of cells). The default therefore merges the spots of one clone within ``loci_radius_um`` into loci
(:func:`spaclone.pp.merge_loci`) and permutes clone labels among loci. Barcode sinks (spots carrying tens of clones)
hold copies of the abundant clones, so two sinks make same-clone pairs that no permutation reproduces; they are removed
first.

References
----------
Illian et al. 2008, doi:10.1002/9780470725160; Baddeley, Rubak & Turner 2015, doi:10.1201/b19708 (mark connection
function, random labelling); Shimatani 2001, doi:10.1016/S0378-1127(00)00352-2 (spatial Simpson index); Besag & Diggle
1977, doi:10.2307/2346974 (Monte-Carlo tests); Myllymäki et al. 2017, doi:10.1111/rssb.12172 (distance bands are
multiple tests: global, max-type inference).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone._utils import (
    LowPowerWarning,
    RngLike,
    TestResult,
    as_rng,
    bh,
    max_t_pvalue,
    mc_pvalue,
    null_summary,
    rng_entropy,
)
from spaclone.nulls import Permute
from spaclone.pp.loci import merge_loci

__all__ = ["ColocalisationResult", "colocalisation"]

logger = logging.getLogger("spaclone")

Closed = Literal["both", "left", "right"]
Units = Literal["loci", "records"]


@dataclass(frozen=True)
class ColocalisationResult(TestResult):
    """Result of :func:`colocalisation`.

    Attributes
    ----------
    clones
        Per-clone contributions (``per_clone=True``), one row per (library, locus, band, clone in the size window):
        ``size``, ``pairs_total`` (same-clone pairs of the clone), ``pairs_within`` (of them in the band), ``share``,
        ``contribution`` (share of the pooled numerator of the band) and the per-clone null means. Pooled pair counts
        weight a clone by about n²/2, so a few large clones can dominate the pooled share; this table shows by how
        much.
    units
        The units the test used, one row each: ``library_id``, ``locus``, the spot (for loci: the spot of the
        locus' record with most reads), clone and coordinate (for loci: centroid) columns, the strata columns (for
        loci: the most frequent stratum of its spots), ``locus_id`` and ``n_spots`` (loci), ``clone_size`` (units of
        the clone in its library and locus) and ``in_window`` (the clone size lies in the window, so the unit
        contributes pairs).
    """

    clones: pd.DataFrame | None = None
    units: pd.DataFrame | None = None


def _as_list(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _check_bands(bands_um: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    bands = [(float(lo), float(hi)) for lo, hi in bands_um]
    if not bands:
        raise ValueError("bands_um must contain at least one (lo, hi) band")
    for lo, hi in bands:
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo < hi):
            raise ValueError(f"each band must satisfy 0 <= lo < hi < inf, got ({lo}, {hi})")
    return bands


def _band_mask(d: np.ndarray, lo: float, hi: float, closed: Closed) -> np.ndarray:
    if closed == "both":
        return (d >= lo) & (d <= hi)
    if closed == "left":
        return (d >= lo) & (d < hi)
    if closed == "right":
        return (d > lo) & (d <= hi)
    raise ValueError(f"closed must be 'both', 'left' or 'right', got {closed!r}")


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def _same_group_pairs(group: np.ndarray) -> np.ndarray:
    """All unordered pairs ``(i, j)``, ``i < j``, of rows sharing a group code; shape ``(n_pairs, 2)``."""
    n = len(group)
    if n < 2:
        return np.empty((0, 2), np.int64)
    order = np.argsort(group, kind="stable")
    g = group[order]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    size = np.diff(np.r_[start, n])
    pos = np.arange(n) - np.repeat(start, size)  # position inside the group
    n_after = np.repeat(size, size) - pos - 1  # partners following each row in its group
    first = np.repeat(np.arange(n), n_after)
    if not len(first):
        return np.empty((0, 2), np.int64)
    # offsets 1..n_after for every row: a running counter reset at each row
    csum = np.cumsum(n_after)
    off = np.arange(len(first)) - np.repeat(csum - n_after, n_after) + 1
    i, j = order[first], order[first + off]
    return np.column_stack([np.minimum(i, j), np.maximum(i, j)]).astype(np.int64)


def _pair_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Integer key of unordered pairs (``a < b`` assumed)."""
    return a.astype(np.int64) * np.int64(max(n, 1)) + b.astype(np.int64)


class _PairShare:
    """Pair geometry of one sample (fixed) and the pair-share statistic of any labelling of its units.

    ``excl`` lists unit pairs that count nowhere (numerator and denominator) because, carrying the same clone, they
    would be one unit: pairs on the same spot ((spot, clone) units) or loci on a shared spot. With ``excl_um`` every
    pair at most that far apart also counts nowhere (loci: closer than the merge radius they would have merged).
    """

    def __init__(
        self,
        xy: np.ndarray,
        excl: np.ndarray,
        codes: np.ndarray,
        bands: list[tuple[float, float]],
        *,
        min_size: int,
        max_size: int | None,
        closed: Closed,
        excl_um: float | None = None,
    ) -> None:
        n = len(xy)
        self.n_codes = int(codes.max()) + 1 if len(codes) else 0
        size = np.bincount(codes, minlength=self.n_codes)
        ok = size >= min_size
        if max_size is not None:
            ok &= size <= max_size
        self.size, self.ok = size, ok
        # the multiset of labels is invariant under permutation, so this part of the denominator is a constant
        self.tot = float((size[ok] * (size[ok] - 1) / 2).sum())
        r_max = max(hi for _, hi in bands)
        tree = cKDTree(xy) if n > 1 else None
        pairs = tree.query_pairs(r_max, output_type="ndarray") if tree is not None else np.empty((0, 2), int)
        pairs = pairs.astype(np.int64, copy=False)
        d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) if len(pairs) else np.empty(0)
        ex_keys = _pair_keys(excl[:, 0], excl[:, 1], n) if len(excl) else np.empty(0, np.int64)
        if excl_um is not None and tree is not None:
            if excl_um <= r_max:
                close = pairs[d <= excl_um]
            else:
                close = tree.query_pairs(excl_um, output_type="ndarray").astype(np.int64, copy=False)
            ex_keys = np.concatenate([ex_keys, _pair_keys(close[:, 0], close[:, 1], n)])
        ex_keys = np.unique(ex_keys)
        keep = ~np.isin(_pair_keys(pairs[:, 0], pairs[:, 1], n), ex_keys) if len(ex_keys) else np.ones(len(pairs), bool)
        self.exclude = len(ex_keys) > 0
        self.ss_a, self.ss_b = ex_keys // np.int64(max(n, 1)), ex_keys % np.int64(max(n, 1))
        self.a, self.b = pairs[keep, 0], pairs[keep, 1]
        dk = d[keep]
        self.band_idx = [np.flatnonzero(_band_mask(dk, lo, hi, closed)) for lo, hi in bands]

    def __call__(self, lab: np.ndarray, *, per_clone: bool = False) -> tuple[np.ndarray, float, np.ndarray | None]:
        """Numerator per band, denominator, and (optionally) per-label numerators ``(n_bands, n_codes)``."""
        la = lab[self.a]
        hit = (la == lab[self.b]) & self.ok[la]
        num = np.array([np.count_nonzero(hit[ix]) for ix in self.band_idx], dtype=np.int64)
        den = self.tot
        if self.exclude and len(self.ss_a):
            ls = lab[self.ss_a]
            den = self.tot - float(np.count_nonzero((ls == lab[self.ss_b]) & self.ok[ls]))
        pcs = None
        if per_clone:
            pcs = np.vstack([np.bincount(la[ix][hit[ix]], minlength=self.n_codes) for ix in self.band_idx]).astype(
                np.int64
            )
        return num, den, pcs

    def expected(self, codes: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        """Expected share per band under random labelling within ``stratum`` (ratio of expectations).

        For units i in stratum s and j in stratum t the probability that both carry the same clone of the window is
        ``sum_c n_cs n_ct / (N_s N_t)`` (s != t) or ``sum_c n_cs (n_cs - 1) / (N_s (N_s - 1))`` (s == t). The
        denominator is the constant ``tot`` minus the expected number of excluded same-clone pairs, so the value is
        exact when no pair is excluded.
        """
        if not len(codes):
            return np.full(len(self.band_idx), np.nan)
        n_st = int(stratum.max()) + 1
        win = np.flatnonzero(self.ok)
        cnt = np.bincount(stratum * self.n_codes + codes, minlength=n_st * self.n_codes).reshape(n_st, self.n_codes)
        a = cnt[:, win].astype(float)
        n_s = np.bincount(stratum, minlength=n_st).astype(float)
        g = a @ a.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pi = g / np.outer(n_s, n_s)
            diag = (np.diag(g) - a.sum(axis=1)) / (n_s * (n_s - 1))
        pi[np.diag_indices(n_st)] = diag
        pi = np.nan_to_num(pi, nan=0.0, posinf=0.0)
        pab = pi[stratum[self.a], stratum[self.b]]
        num = np.array([pab[ix].sum() for ix in self.band_idx])
        den = self.tot
        if self.exclude and len(self.ss_a):
            den = self.tot - float(pi[stratum[self.ss_a], stratum[self.ss_b]].sum())
        return num / den if den > 0 else np.full(len(num), np.nan)


def _strata_groups(stratum: np.ndarray) -> list[np.ndarray]:
    """Row positions of each stratum code, strata in increasing code order, positions ascending."""
    order = np.argsort(stratum, kind="stable")
    st = stratum[order]
    cut = np.flatnonzero(st[1:] != st[:-1]) + 1
    return np.split(order, cut) if len(order) else []


def _shuffle_within(codes: np.ndarray, groups: list[np.ndarray], gen: np.random.Generator) -> np.ndarray:
    """One random labelling: ``codes`` permuted inside every group.

    Consumes the generator exactly as :class:`spaclone.nulls.Permute` with ``within`` applied to a table sorted by
    stratum (one ``permutation(len(group))`` per group, groups in increasing order), and as the archive's
    ``lab[ix] = clone[rng.permutation(ix)]`` loop, without copying a DataFrame per draw.
    """
    lab = codes.copy()
    for pos in groups:
        lab[pos] = codes[pos[gen.permutation(len(pos))]]
    return lab


def _share(num: np.ndarray, den: float) -> np.ndarray:
    if den > 0:
        return num / den
    return np.full(len(num), np.nan)


def _sink_records(
    lib: np.ndarray, spot: np.ndarray, locus: np.ndarray, clone: np.ndarray, min_clones: int
) -> tuple[np.ndarray, int]:
    """Records on barcode sinks: spots holding at least ``min_clones`` distinct (locus, clone) pairs, per library.

    Returns the record mask and the number of sink spots (the rule of :func:`spaclone.pp.barcode_sinks`).
    """
    t = pd.DataFrame({"lib": lib, "spot": spot, "locus": locus, "clone": clone})
    sp = t.groupby(["lib", "spot"], sort=False, dropna=False).ngroup().to_numpy()
    n_sp = int(sp.max()) + 1 if len(sp) else 0
    trip = pd.DataFrame({"s": sp, "l": t["locus"].to_numpy(), "c": t["clone"].to_numpy()}).drop_duplicates()
    sink = np.bincount(trip["s"].to_numpy(), minlength=n_sp) >= min_clones
    return sink[sp], int(sink.sum())


def _mode_codes(owner: np.ndarray, code: np.ndarray, n_owner: int) -> np.ndarray:
    """Most frequent code per owner (ties: the smallest code)."""
    n_code = int(code.max()) + 1 if len(code) else 1
    cnt = np.bincount(owner * n_code + code, minlength=n_owner * n_code).reshape(n_owner, n_code)
    return cnt.argmax(axis=1)


def colocalisation(
    air: pd.DataFrame,
    *,
    strata: str | Sequence[str] | None = None,
    within: str | Sequence[str] | None = None,
    units: Units = "loci",
    loci_radius_um: float = 15.0,
    sink_min_clones: int | None = 20,
    bands_um: Sequence[tuple[float, float]] = ((0.0, 20.0), (20.0, 100.0)),
    min_clone_size: int = 2,
    max_clone_size: int | None = 50,
    exclude_same_spot: bool = True,
    n_perm: int = 200,
    closed: Closed = "both",
    per_clone: bool = False,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> ColocalisationResult:
    """Share of same-clone unit pairs per distance band, observed vs clone labels shuffled.

    Records are first collapsed to one row per (library, locus, spot, clone) (a clone's many molecules on one spot
    count once) and, with ``units="loci"``, the spots of one clone within ``loci_radius_um`` are merged into loci
    (about one cell each). For every library and locus separately, pairs of units are found once with a k-d tree and
    only the labels are shuffled.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or cell call) with spot id, coordinates (µm) and clone id; rows
        with a missing clone id are dropped. Optional ``locus`` and ``library_id`` columns split the test.
    strata, within
        Column(s) of ``air`` that define the depth strata of the stratified null (``within`` is an alias, the name
        used by the rest of the package; give one), e.g. the bin of local Ig RNA of the spot
        (``nulls.bin_labels(ig_rna, nulls.IG_RNA_BINS)``). Must be constant within a (spot, clone) unit; a locus takes
        the most frequent stratum of its spots. With ``None`` only the whole-sample null is run.
    units
        The exchangeable unit. ``"loci"`` (default): same-clone (spot, clone) units within ``loci_radius_um`` are
        merged into loci by :func:`spaclone.pp.merge_loci` (single linkage), loci are the units of the pairs and of
        the clone-size window, and labels are permuted among loci. ``"records"``: (spot, clone) units (the archive's
        T3), right when a spot is a cell or a bead holding about one cell; on sub-cellular spots it treats the spots
        of one cell as independent cells.
    loci_radius_um
        Merge radius of the loci (``units="loci"``), at least the spread of one cell's molecules. Large radii merge
        neighbouring cells of one clone, which removes clustering below that scale from the test (dense same-clone
        clusters such as germinal centres become a few loci); smaller radii keep it but split large cells. In the
        package simulator (2-µm spots, 8-µm cells; 100-200 chips per scenario, ``fixes/F2``) the 20–100 µm
        false-positive rate stayed within 2.0-7.0% for radii 5, 8 and 15 µm in every null, including cells that cover
        several spots, while the power at 30% genuine molecules was 23%, 17% and 10% (the archive's (spot, clone)
        units with the 50-unit window: 14%, and 24% false positives when every cell belongs to an expanded clone).
    sink_min_clones
        Remove records on barcode sinks first: spots holding at least this many distinct clones (all loci, per
        library; :func:`spaclone.pp.barcode_sinks`). ``None`` keeps them (two sinks make same-clone pairs that no
        permutation reproduces).
    bands_um
        Distance bands ``(lo, hi)`` in µm. Bands may overlap (e.g. ``(0, 100)`` with its parts). The 0–20 µm band is
        dominated by neighbouring cells or beads, where RNA spilling over from one cell can create same-clone pairs;
        read it separately from the longer bands. With ``units="loci"`` pairs closer than ``loci_radius_um`` count
        nowhere, so a band below that radius has no pairs.
    min_clone_size, max_clone_size
        Clones with this many units (loci or (spot, clone) units; inclusive; ``max_clone_size=None`` for no upper
        limit) contribute pairs. The window travels with the labels: under the null a unit counts if the label it
        receives is a clone of that size.
    exclude_same_spot
        Pairs of units that would be one unit had they carried the same clone count nowhere (numerator and
        denominator): two (spot, clone) units on one spot (several clones on one bead or DNB) or, for loci, two loci
        sharing a spot or closer than ``loci_radius_um``. Under shuffling such a pair may receive the same clone.
    n_perm
        Number of label shuffles per null design. The smallest p-value is ``1 / (n_perm + 1)``; a
        :class:`~spaclone._utils.LowPowerWarning` says when that cannot pass Benjamini–Hochberg at 0.05 on its own
        across the libraries × loci of one design and band (use ``n_perm >= 20 ×`` that number).
    closed
        Which band edges are inclusive: ``"both"`` (``lo <= d <= hi``, the archive definition; a pair exactly on a
        shared edge counts in both bands), ``"left"`` (``lo <= d < hi``) or ``"right"`` (``lo < d <= hi``).
    per_clone
        Also return per-clone contributions in ``result.clones``.
    rng
        Seed or generator. The generator is consumed in a fixed order (library, then locus, then the stratified
        design, then the whole-sample design), so results are reproducible.
    spot_key, x_key, y_key, clone_key, locus_key, library_key
        Column names.

    Returns
    -------
    ColocalisationResult
        ``table``: one row per (library, locus, design, band) with ``units``, ``n_units``, ``n_spot_clone``
        ((spot, clone) units before merging), ``n_clones``, ``n_clones_in_window``, ``pairs_total`` (denominator:
        same-clone pairs of clones in the window, excluded pairs removed), ``pairs_within`` (numerator),
        ``statistic`` (share), ``expected`` (analytic expectation under the design, a ratio of expectations),
        ``null_mean``, ``null_sd``, ``null_lo`` / ``null_hi`` (2.5% / 97.5%), ``ratio`` (statistic / null mean),
        ``z``, ``p_value`` (one-sided, greater), ``p_maxT`` (family-wise over the bands of the design, max-T),
        ``q_value`` (Benjamini–Hochberg over the libraries and loci of the same design and band), ``top_clone_share``
        (share of the observed numerator from the clone contributing most) and ``n_perm``. ``null`` has shape
        ``(n_perm, n_rows)`` in table order; ``units`` lists the units; ``summary`` also reports the removed sinks.

    Null model
    ----------
    Random labelling, positions fixed. ``"stratified"``: clone labels are permuted among the units (loci, or
    (spot, clone) units) of the same library, locus and stratum, so each stratum keeps its multiset of clone labels
    and everything the strata explain (detection driven by local receptor RNA) is kept; territory structure is not.
    ``"whole_sample"``: labels are permuted among all units of the library and locus. Clone sizes, unit positions and
    spot membership are fixed in both; barcode sinks are removed before.

    Notes
    -----
    With ``units="loci"`` the result does not depend on the row order of ``air`` (loci come in a canonical order).
    With ``units="records"`` the null draws follow the order in which the (spot, clone) units first appear.

    Legacy setting: ``units="records", sink_min_clones=None`` reproduces ``t3`` of
    ``04_external_validation_2026-10/workflow/X5_depth_tests.py`` (same pair-share definition, clone-size window, same-bead
    exclusion and denominator); with integer strata codes, units in the same order and the same generator the
    shuffles are identical draw for draw. Strata are visited in sorted order of their values.

    References
    ----------
    Illian et al. 2008, doi:10.1002/9780470725160; Baddeley, Rubak & Turner 2015, doi:10.1201/b19708; Besag & Diggle
    1977, doi:10.2307/2346974; Westfall & Young 1993 (max-T, ISBN 978-0-471-55761-6); Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x; Kebschull & Zador 2015, doi:10.1093/nar/gkv717 (barcode sinks).

    Examples
    --------
    >>> air["ig_bin"] = spaclone.nulls.bin_labels(air["ig_rna"], spaclone.nulls.IG_RNA_BINS)  # doctest: +SKIP
    >>> res = spaclone.qc.colocalisation(air, within="ig_bin", rng=0)  # doctest: +SKIP
    >>> res.table[["design", "band", "statistic", "null_mean", "ratio", "p_value"]]  # doctest: +SKIP
    """
    S.validate_air(air, required=(spot_key, x_key, y_key, clone_key), numeric=(x_key, y_key))
    bands = _check_bands(bands_um)
    if strata is not None and within is not None and _as_list(strata) != _as_list(within):
        raise ValueError("strata and within are aliases; give one of them")
    strata_cols = _as_list(strata if strata is not None else within)
    missing = [c for c in strata_cols if c not in air.columns]
    if missing:
        raise KeyError(f"strata column(s) {missing} not in the receptor table")
    if units not in ("loci", "records"):
        raise ValueError(f"units must be 'loci' or 'records', got {units!r}")
    if units == "loci" and not (np.isfinite(loci_radius_um) and loci_radius_um >= 0):
        raise ValueError("loci_radius_um must be a finite number >= 0")
    if sink_min_clones is not None and sink_min_clones < 1:
        raise ValueError("sink_min_clones must be >= 1 (or None to keep sinks)")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if min_clone_size < 2:
        raise ValueError("min_clone_size must be >= 2 (a pair needs two units)")
    if max_clone_size is not None and max_clone_size < min_clone_size:
        raise ValueError("max_clone_size must be >= min_clone_size")
    if closed not in ("both", "left", "right"):
        raise ValueError(f"closed must be 'both', 'left' or 'right', got {closed!r}")
    gen = as_rng(rng)

    df = air.loc[air[clone_key].notna()]
    n_dropped = len(air) - len(df)
    if n_dropped:
        logger.info("colocalisation: %d records without a clone id dropped", n_dropped)
    lib = df[library_key] if library_key in df.columns else pd.Series("", index=df.index)
    loc = df[locus_key] if locus_key in df.columns else pd.Series("", index=df.index)
    df = df.assign(_lib=lib.astype(str).to_numpy(), _loc=loc.astype(str).to_numpy())
    n_sink_spots = n_sink_records = 0
    if sink_min_clones is not None and len(df):
        on_sink, n_sink_spots = _sink_records(
            df["_lib"].to_numpy(), df[spot_key].to_numpy(), df["_loc"].to_numpy(), df[clone_key].to_numpy(),
            int(sink_min_clones),
        )  # fmt: skip
        n_sink_records = int(on_sink.sum())
        if n_sink_records:
            logger.info("colocalisation: %d records on %d barcode sinks removed", n_sink_records, n_sink_spots)
            df = df.loc[~on_sink]
    ukeys = ["_lib", "_loc", spot_key, clone_key]
    if strata_cols:
        nun = df.groupby(ukeys, sort=False, dropna=False)[strata_cols].nunique(dropna=False)
        bad = [c for c in strata_cols if (nun[c] > 1).any()]
        if bad:
            raise ValueError(f"strata column(s) {bad} vary inside a (spot, clone) unit; strata must be unit-level")
    spot_units = df.drop_duplicates(ukeys)
    if strata_cols:  # integer code per strata combination, sorted by value (the archive's visiting order)
        st_all = spot_units.groupby(strata_cols, sort=True, dropna=False).ngroup().to_numpy()
        st_first = pd.Series(np.arange(len(st_all))).groupby(st_all).first().to_numpy()
    else:
        st_all = np.zeros(len(spot_units), np.int64)
        st_first = np.zeros(1, np.int64)

    rows: list[dict[str, Any]] = []
    nulls_out: list[np.ndarray] = []
    clone_rows: list[pd.DataFrame] = []
    unit_tabs: list[pd.DataFrame] = []
    for (lib_v, loc_v), ix in spot_units.groupby(["_lib", "_loc"], sort=True).indices.items():
        sub = spot_units.iloc[ix]
        st_sub = st_all[ix]
        spot_c, _ = pd.factorize(sub[spot_key], sort=False)
        if units == "records":
            xy = S.coords(sub, x_key=x_key, y_key=y_key)
            codes, uniq = pd.factorize(sub[clone_key], sort=False)
            excl = _same_group_pairs(spot_c) if exclude_same_spot else np.empty((0, 2), np.int64)
            excl_um = None
            st_codes = st_sub
            tab = pd.DataFrame({spot_key: sub[spot_key].to_numpy(), clone_key: sub[clone_key].to_numpy()})
        else:
            loci, member = merge_loci(
                sub[[spot_key, x_key, y_key, clone_key]], radius_um=loci_radius_um, clone_key=clone_key, by=(),
                spot_key=spot_key, x_key=x_key, y_key=y_key,
            )  # fmt: skip
            mem = member.to_numpy()
            xy = S.coords(loci, x_key=x_key, y_key=y_key)
            codes, uniq = pd.factorize(loci[clone_key], sort=False)
            excl = np.empty((0, 2), np.int64)
            if exclude_same_spot:  # loci sharing a spot (several clones on one spot) would merge under one label
                pr = _same_group_pairs(spot_c)
                if len(pr):
                    la, lb = mem[pr[:, 0]], mem[pr[:, 1]]
                    keep_pr = la != lb
                    lp = np.unique(_pair_keys(np.minimum(la, lb)[keep_pr], np.maximum(la, lb)[keep_pr], len(loci)))
                    excl = np.column_stack([lp // max(len(loci), 1), lp % max(len(loci), 1)]).astype(np.int64)
            excl_um = float(loci_radius_um) if exclude_same_spot else None
            st_codes = _mode_codes(mem, st_sub, len(loci)) if strata_cols else np.zeros(len(loci), np.int64)
            tab = pd.DataFrame(
                {
                    spot_key: loci[spot_key].to_numpy(),
                    clone_key: loci[clone_key].to_numpy(),
                    "locus_id": loci["locus_id"].to_numpy(),
                    "n_spots": loci["n_spots"].to_numpy(),
                }
            )
        ps = _PairShare(
            xy,
            excl,
            codes,
            bands,
            min_size=min_clone_size,
            max_size=max_clone_size,
            closed=closed,
            excl_um=excl_um,
        )
        num0, den0, pc0 = ps(codes, per_clone=True)
        obs = _share(num0, den0)
        # strata in sorted order of their values, units in table order inside each (the archive's visiting order)
        designs: list[tuple[str, np.ndarray]] = []
        if strata_cols:
            designs.append(("stratified", pd.factorize(st_codes, sort=True)[0].astype(np.int64)))
        designs.append(("whole_sample", np.zeros(len(codes), np.int64)))
        base = {
            "library_id": lib_v,
            "locus": loc_v,
            "units": units,
            "n_units": len(codes),
            "n_spot_clone": len(sub),
            "n_clones": len(uniq),
            "n_clones_in_window": int(ps.ok.sum()),
            "pairs_total": den0,
        }
        tops = pc0.max(axis=1) if pc0.shape[1] else np.zeros(len(bands), dtype=np.int64)
        pc_null_means: dict[str, np.ndarray] = {}
        for dname, d_codes in designs:
            groups = _strata_groups(d_codes)
            null = np.empty((n_perm, len(bands)))
            pc_acc = np.zeros_like(pc0, dtype=float) if per_clone else None
            for i in range(n_perm):
                lab = _shuffle_within(codes, groups, gen)
                num, den, pcs = ps(lab, per_clone=per_clone)
                null[i] = _share(num, den)
                if per_clone:
                    pc_acc += pcs
            if per_clone:
                pc_null_means[dname] = pc_acc / n_perm
            expected = ps.expected(codes, d_codes)
            mu, sd, lo, hi = null_summary(null)
            p = mc_pvalue(null, obs, alternative="greater")
            finite = np.isfinite(obs) & np.isfinite(null).all(axis=0)
            p_max = np.full(len(bands), np.nan)
            if finite.any():
                p_max[finite] = max_t_pvalue(null[:, finite], obs[finite], alternative="greater")
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = obs / mu
                z = (obs - mu) / sd
            for k, (b_lo, b_hi) in enumerate(bands):
                rows.append(
                    {
                        **base,
                        "design": dname,
                        "band": _band_label(b_lo, b_hi),
                        "lo_um": b_lo,
                        "hi_um": b_hi,
                        "pairs_within": int(num0[k]),
                        "statistic": obs[k],
                        "expected": expected[k],
                        "null_mean": mu[k],
                        "null_sd": sd[k],
                        "null_lo": lo[k],
                        "null_hi": hi[k],
                        "ratio": ratio[k],
                        "z": z[k],
                        "p_value": p[k],
                        "p_maxT": p_max[k],
                        "top_clone_share": float(tops[k] / num0[k]) if num0[k] else np.nan,
                        "n_perm": n_perm,
                    }
                )
            nulls_out.append(null)
        if per_clone and ps.ok.any():
            okc = np.flatnonzero(ps.ok)
            for k, (b_lo, b_hi) in enumerate(bands):
                cd = pd.DataFrame(
                    {
                        "library_id": lib_v,
                        "locus": loc_v,
                        "band": _band_label(b_lo, b_hi),
                        "clone_id": np.asarray(uniq)[okc],
                        "size": ps.size[okc],
                        "pairs_total": ps.size[okc] * (ps.size[okc] - 1) // 2,
                        "pairs_within": pc0[k, okc],
                    }
                )
                cd["share"] = cd.pairs_within / cd.pairs_total
                cd["contribution"] = cd.pairs_within / num0[k] if num0[k] else np.nan
                for dname, m in pc_null_means.items():
                    cd[f"null_mean_{dname}"] = m[k, okc]
                clone_rows.append(cd.sort_values("pairs_within", ascending=False, kind="stable"))
        ut = pd.DataFrame({"library_id": lib_v, "locus": loc_v}, index=range(len(codes)))
        for c in tab.columns:
            ut[c] = tab[c].to_numpy()
        ut[x_key], ut[y_key] = xy[:, 0], xy[:, 1]
        for c in strata_cols:
            ut[c] = spot_units[c].to_numpy()[st_first[st_codes]] if len(st_codes) else []
        ut["clone_size"] = ps.size[codes]
        ut["in_window"] = ps.ok[codes]
        unit_tabs.append(ut)

    cols = [
        "library_id",
        "locus",
        "design",
        "band",
        "lo_um",
        "hi_um",
        "units",
        "n_units",
        "n_spot_clone",
        "n_clones",
        "n_clones_in_window",
        "pairs_total",
        "pairs_within",
        "statistic",
        "expected",
        "null_mean",
        "null_sd",
        "null_lo",
        "null_hi",
        "ratio",
        "z",
        "p_value",
        "p_maxT",
        "q_value",
        "top_clone_share",
        "n_perm",
    ]
    table = pd.DataFrame(rows, columns=[c for c in cols if c != "q_value"])
    q = np.full(len(table), np.nan)
    m_max = 0
    for _, ix in table.groupby(["design", "band"], sort=False).indices.items():
        q[ix] = bh(table.p_value.to_numpy()[ix])
        m_max = max(m_max, int(np.isfinite(table.p_value.to_numpy()[ix]).sum()))
    table.insert(cols.index("q_value"), "q_value", q)
    if m_max > 1 and 1 / (n_perm + 1) > 0.05 / m_max:
        warnings.warn(
            f"colocalisation: with {m_max} libraries x loci per design and band the smallest p-value 1/{n_perm + 1} "
            "cannot pass Benjamini-Hochberg at 0.05 on its own; use n_perm >= 20 x the number of rows",
            LowPowerWarning,
            stacklevel=2,
        )
    null_arr = np.hstack(nulls_out) if nulls_out else np.empty((n_perm, 0))
    summary: dict[str, Any] = {
        "n_tests": int(table[["library_id", "locus"]].drop_duplicates().shape[0]),
        "n_units": int(table.drop_duplicates(["library_id", "locus"]).n_units.sum()) if len(table) else 0,
        "units": units,
        "n_sink_spots": n_sink_spots,
        "n_sink_records": n_sink_records,
    }
    for (dname, band), t in table.groupby(["design", "band"], sort=False):
        summary[f"ratio[{dname}|{band}]"] = float(np.nanmedian(t.ratio)) if t.ratio.notna().any() else np.nan
        summary[f"n_p05[{dname}|{band}]"] = int((t.p_value <= 0.05).sum())
    params = {
        "strata": strata_cols,
        "units": units,
        "loci_radius_um": float(loci_radius_um) if units == "loci" else None,
        "sink_min_clones": sink_min_clones,
        "bands_um": bands,
        "min_clone_size": min_clone_size,
        "max_clone_size": max_clone_size,
        "exclude_same_spot": exclude_same_spot,
        "closed": closed,
        "n_perm": n_perm,
        "unit": "(library, locus, locus_id)" if units == "loci" else "(library, locus, spot, clone)",
        "designs": {
            "stratified": Permute("clone", within=["library", "locus", *strata_cols]).describe()
            if strata_cols
            else None,
            "whole_sample": Permute("clone", within=["library", "locus"]).describe(),
        },
        "alternative": "greater",
        "rng_entropy": rng_entropy(gen),
    }
    clones = pd.concat(clone_rows, ignore_index=True) if clone_rows else (pd.DataFrame() if per_clone else None)
    units_out = pd.concat(unit_tabs, ignore_index=True) if unit_tabs else pd.DataFrame()
    return ColocalisationResult(
        name="colocalisation",
        table=table,
        summary=summary,
        params=params,
        null=null_arr,
        clones=clones,
        units=units_out,
    )
