"""Heavy/light chain pairing beyond chance: do the two chains of the same cells meet in the same bins?

A B cell or plasma cell (a T cell) puts its heavy and its light transcripts (beta and alpha) into the same 10–20 µm of
tissue, so if a share of the receptor records is where its cell was, a heavy clone co-occurs in small bins with its
own light partner more often than with other light clones. Under barcode swapping the two chains are amplified in
separate reactions and land independently, and the co-occurrence of any heavy and light clone is what their
placements alone give.

For bins b of side ``bin_um`` let ``H[b, h]`` and ``L[b, k]`` count the (spot, clone) units of heavy clone h and light
clone k. The co-occurrence of a pair is ``O[h, k] = sum_b H[b, h] L[b, k]``; each pair is scored against the null by
``z = (O - mean) / sd`` (sd floored at the Poisson value, at least 0.5), each heavy clone's best partner (max z) against
the null distribution of the maximum, and the global statistic is the mean best z over the heavy clones.

The null keeps every light unit where it is and randomises the heavy units, so each light clone's own placement
(technical or not) is part of the null. A light clone that sits on multi-clone spots (chimeras, barcode sinks) meets
every heavy clone there; permuting heavy labels only within strata of clones per spot keeps each heavy clone's
tendency to sit on such spots, which a whole-sample permutation mistakes for pairing (archive 03, B5a vs B5c).

References
----------
Howie et al. 2015, doi:10.1126/scitranslmed.aac5624 (pairSEQ co-occurrence pairing); Gotelli 2000,
doi:10.1890/0012-9658(2000)081[2606:NMAOSC]2.0.CO;2 (fixed-fixed null models); Strona et al. 2014,
doi:10.1038/ncomms5114 (curveball algorithm); Besag & Clifford 1989, doi:10.1093/biomet/76.4.633 (exact Monte-Carlo
tests from Markov chains); Westfall & Young 1993 (max-statistic adjustment, ISBN 978-0-471-55761-6); Benjamini &
Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from spaclone import _schema as S
from spaclone._utils import RngLike, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy

__all__ = ["CLONES_PER_SPOT_BINS", "PairingResult", "pairing"]

logger = logging.getLogger("spaclone")

#: Strata of clones per spot (all loci) for the chimera-aware null: 1, 2, 3-5, >= 6 (archive B5c).
CLONES_PER_SPOT_BINS: tuple[float, ...] = (1, 2, 3, 6, np.inf)


@dataclass(frozen=True)
class PairingResult(TestResult):
    """Result of :func:`pairing`.

    Attributes
    ----------
    clones
        One row per tested heavy clone (and library, heavy and light locus): ``heavy``, ``n_heavy`` (units),
        ``best_light`` (partner with the largest z), ``n_light``, ``O`` (co-occurrence), ``mu`` (null mean of the
        pair), ``z``, ``z_second`` (z of the second-best partner), ``p_pair`` (Monte-Carlo p of the best pair's
        co-occurrence; descriptive only: the partner was chosen as the best on the same data, so this is not a valid
        p-value), ``p_value`` (p of the best z against the null of each heavy clone's maximum, which accounts for the
        choice of partner), ``q_value`` (Benjamini–Hochberg over the heavy clones of the test) and ``O_same_spot``
        (spots holding both clones).
    """

    clones: pd.DataFrame | None = None


def _as_tuple(v: str | Sequence[str]) -> tuple[str, ...]:
    return (v,) if isinstance(v, str) else tuple(v)


def _groups(codes: np.ndarray) -> list[np.ndarray]:
    """Positions of each code, codes in increasing order, positions ascending (empty codes skipped)."""
    order = np.argsort(codes, kind="stable")
    st = codes[order]
    cut = np.flatnonzero(st[1:] != st[:-1]) + 1
    return np.split(order, cut) if len(order) else []


def _stratum_codes(mult: np.ndarray, edges: Sequence[float] | None) -> np.ndarray:
    """Half-open bins ``[e_i, e_i+1)`` of clones per spot; values beyond the edges join the outer bins."""
    if edges is None:
        return np.zeros(len(mult), np.int64)
    e = np.asarray(edges, dtype=float)
    return np.clip(np.searchsorted(e, mult, side="right") - 1, 0, len(e) - 2).astype(np.int64)


def _curveball_sweep(
    row: np.ndarray, col: np.ndarray, n_cols: int, row_stratum: np.ndarray, gen: np.random.Generator
) -> np.ndarray:
    """One sweep of curveball trades on a binary rows × columns matrix given as units ``(row[u], col[u])``.

    Rows are paired at random inside their stratum; in every pair the columns held by both rows stay, and the columns
    held by only one of them are dealt out again at random, each row keeping its number of columns (Strona et al.
    2014). Row sums, column sums inside every row stratum and binariness are preserved; every trade is symmetric, so
    the uniform distribution on such matrices is stationary. Returns the new row of every unit (columns move with
    their units).
    """
    n_rows = len(row_stratum)
    order = np.lexsort((gen.random(n_rows), row_stratum))
    st = row_stratum[order]
    starts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
    sizes = np.diff(np.r_[starts, n_rows])
    gstart = np.repeat(starts, sizes)
    pos = np.arange(n_rows) - gstart
    pair_sorted = gstart + (pos // 2) * 2  # id of a pair: sorted position of its first row
    pair_sorted[(np.repeat(sizes, sizes) % 2 == 1) & (pos == np.repeat(sizes, sizes) - 1)] = -1  # odd row out
    pair = np.empty(n_rows, np.int64)
    pair[order] = pair_sorted
    g = pair[row]
    act = np.flatnonzero(g >= 0)
    if not len(act):
        return row
    key = g[act] * np.int64(n_cols) + col[act]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mov = act[cnt[inv] == 1]  # columns held by one row of the pair only
    if not len(mov):
        return row
    gm = g[mov]
    a = mov[np.lexsort((gen.random(len(mov)), gm))]
    b = mov[np.lexsort((row[mov], gm))]
    new = row.copy()
    new[a] = row[b]
    return new


def pairing(
    air: pd.DataFrame,
    *,
    heavy: str | Sequence[str] = ("IGH",),
    light: str | Sequence[str] = ("IGK", "IGL"),
    bin_um: float = 10.0,
    null: Literal["stratified", "curveball"] = "stratified",
    strata_bins: Sequence[float] | None = CLONES_PER_SPOT_BINS,
    singleton_only: bool = False,
    min_units: int = 50,
    n_perm: int = 999,
    n_sweeps: int = 50,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> PairingResult:
    """Heavy/light co-occurrence in small bins against a chimera-aware null.

    Records are collapsed to (library, spot, locus, clone) units first (chains come from separate reactions, so each
    is deduplicated on its own). Every heavy locus is tested against every light locus, separately per library.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or unit) with spot, coordinates (µm), locus and clone; barcode
        sinks should already be removed (they behave like giant multiplets in which every heavy and light clone
        co-occur). Records with a missing clone are dropped.
    heavy, light
        Heavy and light loci, e.g. ``("IGH",)`` and ``("IGK", "IGL")``, or ``("TRB",)`` and ``("TRA",)``.
    bin_um
        Side of the square bins (µm) in which co-occurrence is counted. A cell spans many Stereo-seq DNBs, so its
        two chains rarely share a DNB; 10 µm (bin20) is about one cell.
    null
        ``"stratified"``: heavy labels permuted among the heavy units of the same stratum (the archive's null).
        ``"curveball"``: fixed–fixed randomisation of the binary spot × heavy-clone matrix by curveball trades between
        spots of the same stratum, sampled by the parallel method of Besag & Clifford.
    strata_bins
        Edges of the strata of clones per spot (units of any locus on the spot, half-open bins); ``None`` for a
        single stratum (the whole-sample null that B5a used, which mistakes clone-specific crowding for pairing).
    singleton_only
        Keep only units on spots holding a single unit of any locus, where multi-clone spots cannot create
        co-occurrence.
    min_units
        Heavy and light clones with at least this many units in the pool are tested.
    n_perm
        Number of null data sets.
    n_sweeps
        Curveball sweeps from the observed matrix to the hub and from the hub to every null data set (each sweep
        offers every spot one trade). The test is exact for any value, but an under-mixed chain leaves the null data
        close to the observed matrix and the test conservative. A spot holding k heavy clones that trades with
        spots holding one changes about one clone per sweep, so it needs about k log k sweeps to be renewed; inside
        the default strata spots of similar crowding trade with each other and mix faster. Check that the result is
        stable when ``n_sweeps`` is doubled. Without strata the chain mixes more slowly: use at least 5 times the
        largest number of heavy clones on a spot.
    rng
        Seed or generator; consumed library by library, then heavy locus and light locus in the given order.
    spot_key, x_key, y_key, clone_key, locus_key, library_key
        Column names.

    Returns
    -------
    PairingResult
        ``table``: one row per (library, heavy locus, light locus) with ``design``, ``n_heavy_units``,
        ``n_light_units``, ``n_heavy`` and ``n_light`` (tested clones), ``statistic`` (mean best z over heavy
        clones), ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``, ``z``, ``p_value`` (global, one-sided),
        ``n_q05`` (heavy clones with q < 0.05), ``min_q``, ``median_p``, ``q_floor`` (the smallest q-value a single
        heavy clone can reach, n_heavy / (n_perm + 1); above 0.05 no clone can be called on its own),
        ``max_heavy_per_best_light`` (hub light clones: the largest number of heavy clones sharing one best partner),
        ``n_distinct_best_light``, ``testable`` (False when fewer than two clones of either chain have ``min_units``
        units: statistic and p-value are NaN, an untested combination and not a negative result), ``note`` and
        ``n_perm``.
        ``clones``: per heavy clone, see :class:`PairingResult`. ``null``: the global statistic of every null data
        set, ``(n_perm, n_rows)``.

    Null model
    ----------
    Light units, all positions and the spots' units are fixed; heavy clone labels are randomised among the heavy
    units of the same library and locus. ``"stratified"``: random labelling inside strata of clones per spot, so each
    heavy clone keeps its number of units in every stratum (its tendency to sit on multi-clone spots).
    ``"curveball"``: uniform over binary spot × heavy-clone matrices with the observed number of heavy clones on
    every spot and the observed number of spots of every heavy clone inside every stratum; unlike label permutation
    it never puts one clone twice on a spot.

    Notes
    -----
    Reproduces ``03_BCR_final_2026-10/workflow/B5c_pairing.py`` (``"stratified"`` with the default strata is the
    ``no_sinks_strat`` pool, ``strata_bins=None`` the ``no_sinks`` pool, ``singleton_only=True`` the
    ``singleton_dnb`` pool); with units in the same order and the same generator the permutations are identical draw
    for draw. A margin-preserving null does not remove co-clustering of distinct clones, which also raises O; read
    ``z_second`` (a genuine partner stands out from the second best) and the hub columns.

    References
    ----------
    Howie et al. 2015, doi:10.1126/scitranslmed.aac5624; Strona et al. 2014, doi:10.1038/ncomms5114; Besag & Clifford
    1989, doi:10.1093/biomet/76.4.633; Gotelli 2000, doi:10.1890/0012-9658(2000)081[2606:NMAOSC]2.0.CO;2.

    Examples
    --------
    >>> res = spaclone.qc.pairing(units, heavy="IGH", light="IGK", rng=0)  # doctest: +SKIP
    >>> res.table[["light_locus", "statistic", "null_mean", "p_value"]]  # doctest: +SKIP
    """
    S.validate_air(air, required=(spot_key, x_key, y_key, clone_key, locus_key), numeric=(x_key, y_key))
    heavy_t, light_t = _as_tuple(heavy), _as_tuple(light)
    if set(heavy_t) & set(light_t):
        raise ValueError("a locus cannot be both heavy and light")
    if null not in ("stratified", "curveball"):
        raise ValueError(f"null must be 'stratified' or 'curveball', got {null!r}")
    if not bin_um > 0:
        raise ValueError("bin_um must be positive")
    if n_perm < 1 or n_sweeps < 1 or min_units < 1:
        raise ValueError("n_perm, n_sweeps and min_units must be >= 1")
    if strata_bins is not None:
        e = np.asarray(strata_bins, dtype=float)
        if len(e) < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("strata_bins must be at least two increasing edges")
    gen = as_rng(rng)
    df = air.loc[air[clone_key].notna()]
    lib = df[library_key].astype(str).to_numpy() if library_key in df.columns else np.full(len(df), "", object)
    df = df.assign(_lib=lib)
    units = df.drop_duplicates(["_lib", spot_key, locus_key, clone_key])
    spot_code = units.groupby(["_lib", spot_key], sort=False).ngroup().to_numpy()
    mult = np.bincount(spot_code)[spot_code]  # units of any locus on the unit's spot
    if singleton_only:
        units, spot_code, mult = units[mult == 1], spot_code[mult == 1], mult[mult == 1]
    stratum = _stratum_codes(mult, strata_bins)
    design = ("curveball" if null == "curveball" else "stratified") if strata_bins is not None else None
    design = design or ("curveball_whole_sample" if null == "curveball" else "whole_sample")
    arr = {
        "lib": units["_lib"].to_numpy(),
        "loc": units[locus_key].astype(str).to_numpy(),
        "x": units[x_key].to_numpy(float),
        "y": units[y_key].to_numpy(float),
        "spot": spot_code,
        "stratum": stratum,
        "clone": units[clone_key].to_numpy(),
    }

    rows: list[dict[str, Any]] = []
    clone_tabs: list[pd.DataFrame] = []
    nulls_out: list[np.ndarray] = []
    for lib_v in sorted(pd.unique(arr["lib"])):
        in_lib = arr["lib"] == lib_v
        for h_loc in heavy_t:
            for l_loc in light_t:
                ih = np.flatnonzero(in_lib & (arr["loc"] == h_loc))
                il = np.flatnonzero(in_lib & (arr["loc"] == l_loc))
                base = {
                    "library_id": lib_v,
                    "heavy_locus": h_loc,
                    "light_locus": l_loc,
                    "design": design,
                    "bin_um": float(bin_um),
                    "n_heavy_units": len(ih),
                    "n_light_units": len(il),
                }
                hc = pd.Series(arr["clone"][ih]).value_counts(sort=True)
                lc = pd.Series(arr["clone"][il]).value_counts(sort=True)
                hkeep, lkeep = pd.Index(hc.index[hc >= min_units]), pd.Index(lc.index[lc >= min_units])
                if len(hkeep) < 2 or len(lkeep) < 2:
                    logger.info(
                        "pairing: %s %s x %s untested (%d / %d clones)", lib_v, h_loc, l_loc, len(hkeep), len(lkeep)
                    )
                    rows.append(
                        {
                            **base,
                            "n_heavy": len(hkeep),
                            "n_light": len(lkeep),
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "testable": False,
                            "note": f"untested: {len(hkeep)} heavy and {len(lkeep)} light clones with >= {min_units} "
                            "units (needs 2 of each)",
                            "n_perm": n_perm,
                        }
                    )
                    nulls_out.append(np.full(n_perm, np.nan))
                    continue
                r, ct, gnull = _one_test(arr, ih, il, hc, lc, hkeep, lkeep, null, bin_um, n_perm, n_sweeps, gen)
                rows.append({**base, **r, "testable": True, "note": "", "n_perm": n_perm})
                clone_tabs.append(ct.assign(library_id=lib_v, heavy_locus=h_loc, light_locus=l_loc, design=design))
                nulls_out.append(gnull)
    cols = [
        "library_id", "heavy_locus", "light_locus", "design", "bin_um", "n_heavy_units", "n_light_units", "n_heavy",
        "n_light", "statistic", "null_mean", "null_sd", "null_lo", "null_hi", "z", "p_value", "n_q05", "min_q",
        "median_p", "q_floor", "max_heavy_per_best_light", "n_distinct_best_light", "testable", "note", "n_perm",
    ]  # fmt: skip
    table = pd.DataFrame(rows).reindex(columns=cols)
    lead = ["library_id", "heavy_locus", "light_locus", "design"]
    clones = pd.concat(clone_tabs, ignore_index=True) if clone_tabs else pd.DataFrame(columns=lead)
    clones = clones[lead + [c for c in clones.columns if c not in lead]]
    tested = table[table.statistic.notna()]
    summary = {
        "n_tests": int(len(tested)),
        "n_untestable": int((~table.testable.astype(bool)).sum()) if len(table) else 0,
        "min_p": float(tested.p_value.min()) if len(tested) else np.nan,
        "max_z": float(tested.z.max()) if len(tested) else np.nan,
        "n_heavy_q05": int(tested.n_q05.sum()) if len(tested) else 0,
        "max_hub": int(tested.max_heavy_per_best_light.max()) if len(tested) else 0,
    }
    params = {
        "heavy": heavy_t,
        "light": light_t,
        "bin_um": float(bin_um),
        "null": null,
        "design": design,
        "strata_bins": None if strata_bins is None else [float(v) for v in strata_bins],
        "singleton_only": singleton_only,
        "min_units": min_units,
        "n_perm": n_perm,
        "n_sweeps": n_sweeps if null == "curveball" else None,
        "unit": "(library, spot, locus, clone)",
        "alternative": "greater",
        "rng_entropy": rng_entropy(gen),
    }
    null_arr = np.column_stack(nulls_out) if nulls_out else np.empty((n_perm, 0))
    return PairingResult(name="pairing", table=table, summary=summary, params=params, null=null_arr, clones=clones)


def _unit_entries(bins: np.ndarray, ptr: np.ndarray, idx: np.ndarray, val: np.ndarray):
    """For units in ``bins``, the entries of their bin's row of a CSR matrix: (unit position, column, value)."""
    lens = ptr[bins + 1] - ptr[bins]
    unit = np.repeat(np.arange(len(bins)), lens)
    pos = np.repeat(ptr[bins] - (np.cumsum(lens) - lens), lens) + np.arange(int(lens.sum()))
    return unit, idx[pos], val[pos]


def _one_test(arr, ih, il, hc, lc, hkeep, lkeep, null, bin_um, n_perm, n_sweeps, gen):
    """Observed and null co-occurrence of one heavy x light combination; returns (row, clone table, global null).

    ``O[h, k] = sum over heavy units u of clone h of L[bin(u), k]``: every heavy unit carries the light counts of its
    bin as sparse entries, so a null data set costs one ``bincount`` over those entries.
    """
    bx = np.floor(np.r_[arr["x"][ih], arr["x"][il]] / bin_um).astype(np.int64)
    by = np.floor(np.r_[arr["y"][ih], arr["y"][il]] / bin_um).astype(np.int64)
    key = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    bidx = np.unique(key)
    rows = np.searchsorted(bidx, key)
    rows_h, rows_l = rows[: len(ih)], rows[len(ih) :]
    n_bins, n_h, n_l = len(bidx), len(hkeep), len(lkeep)
    lcol = lkeep.get_indexer(arr["clone"][il])
    m = lcol >= 0
    lmat = sparse.csr_matrix((np.ones(int(m.sum())), (rows_l[m], lcol[m])), shape=(n_bins, n_l))
    lmat.sum_duplicates()
    ptr, lidx, lval = lmat.indptr.astype(np.int64), lmat.indices.astype(np.int64), lmat.data
    codes, uniq = pd.factorize(arr["clone"][ih])
    col_h = hkeep.get_indexer(uniq)[codes].astype(np.int64)  # tested column of each heavy unit, -1 if not tested
    tested = np.flatnonzero(col_h >= 0)

    def counts(col: np.ndarray, unit_bins: np.ndarray) -> np.ndarray:
        u, k, w = _unit_entries(unit_bins, ptr, lidx, lval)
        return np.bincount(col[u] * n_l + k, weights=w, minlength=n_h * n_l).reshape(n_h, n_l)

    obs = counts(col_h[tested], rows_h[tested])
    ob = np.empty((n_perm, n_h, n_l), np.float32)
    if null == "stratified":
        eu, ek, ew = _unit_entries(rows_h, ptr, lidx, lval)  # entries stay with the positions
        groups = _groups(arr["stratum"][ih])
        for b in range(n_perm):
            c = col_h.copy()
            for g in groups:
                c[g] = col_h[g[gen.permutation(len(g))]]
            lab = c[eu]
            ok = lab >= 0
            ob[b] = np.bincount(lab[ok] * n_l + ek[ok], weights=ew[ok], minlength=n_h * n_l).reshape(n_h, n_l)
    else:
        spot_code, spot_first = pd.factorize(arr["spot"][ih])
        first = np.full(len(spot_first), -1, np.int64)
        first[spot_code[::-1]] = np.arange(len(ih))[::-1]
        bin_of_spot = rows_h[first]
        row_st = arr["stratum"][ih][first]
        n_cols = int(codes.max()) + 1
        hub = spot_code.astype(np.int64)
        for _ in range(n_sweeps):
            hub = _curveball_sweep(hub, codes, n_cols, row_st, gen)
        for b in range(n_perm):
            cur = hub
            for _ in range(n_sweeps):
                cur = _curveball_sweep(cur, codes, n_cols, row_st, gen)
            ob[b] = counts(col_h[tested], bin_of_spot[cur[tested]])
    mu = ob.mean(axis=0, dtype=np.float64)
    sd = np.sqrt(np.maximum(ob.var(axis=0, dtype=np.float64), 0.0))
    sd = np.maximum(sd, np.sqrt(np.maximum(mu, 0.25)))  # floor: Poisson sd, at least 0.5
    z = (obs - mu) / sd
    order_z = np.argsort(-z, axis=1, kind="stable")
    j, j2 = order_z[:, 0], order_z[:, 1]
    i = np.arange(n_h)
    zmax = z[i, j]
    zbmax = np.empty((n_perm, n_h))
    for s in range(0, n_perm, 64):  # chunks keep the standardised null small in memory
        zbmax[s : s + 64] = ((ob[s : s + 64] - mu) / sd).max(axis=2)
    p_max = mc_pvalue(zbmax, zmax, alternative="greater")
    p_pair = (1 + (ob[:, i, j] >= obs[i, j] - 1e-9).sum(axis=0)) / (n_perm + 1)
    q = bh(p_max)
    # co-occurrence of the best pairs on the very same spot
    hs, ls = arr["spot"][ih], arr["spot"][il]
    same = np.array(
        [len(np.intersect1d(hs[col_h == a], ls[lcol == b], assume_unique=False)) for a, b in zip(i, j, strict=True)]
    )
    ct = pd.DataFrame(
        {
            "heavy": hkeep.to_numpy(),
            "n_heavy": hc.reindex(hkeep).to_numpy(),
            "best_light": lkeep.to_numpy()[j],
            "n_light": lc.reindex(lkeep[j]).to_numpy(),
            "O": obs[i, j],
            "mu": mu[i, j],
            "z": zmax,
            "z_second": z[i, j2],
            "p_pair": p_pair,
            "p_value": p_max,
            "q_value": q,
            "O_same_spot": same,
        }
    )
    g_obs, g_null = float(zmax.mean()), zbmax.mean(axis=1)
    mu_g, sd_g, lo_g, hi_g = (float(v[0]) for v in null_summary(g_null))
    deg = ct.best_light.value_counts()
    row = {
        "n_heavy": n_h,
        "n_light": n_l,
        "statistic": g_obs,
        "null_mean": mu_g,
        "null_sd": sd_g,
        "null_lo": lo_g,
        "null_hi": hi_g,
        "z": (g_obs - mu_g) / sd_g if sd_g > 0 else np.nan,
        "p_value": float(mc_pvalue(g_null, g_obs, alternative="greater")[0]),
        "n_q05": int((q < 0.05).sum()),
        "min_q": float(q.min()),
        "median_p": float(np.median(p_max)),
        "q_floor": min(1.0, n_h / (n_perm + 1)),
        "max_heavy_per_best_light": int(deg.max()),
        "n_distinct_best_light": int(len(deg)),
    }
    return row, ct.sort_values("p_value", kind="stable").reset_index(drop=True), g_null
