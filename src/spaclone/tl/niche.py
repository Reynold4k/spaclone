"""Niche profiles: where a labelled group of units sits relative to a structure border, against a baseline group.

The position covariate is a signed distance to a structure border (or a compartment). For each distance band (or
compartment) ``k`` the profile is the share of the group's units in that band, ``g_k = n_group,k / n_group``,
compared with the baseline's share ``b_k`` through the ratio ``g_k / b_k``. This is the binned form of the
``rhohat`` ratio estimator against a baseline (Baddeley et al. 2012): normalising each band by its baseline units, not
by raw counts or area, removes the band's capture efficiency and unit density.

References
----------
Baddeley et al. 2012, doi:10.4310/SII.2012.v5.n2.a7 (intensity as a function of a covariate relative to a
baseline); Myllymäki et al. 2017, doi:10.1111/rssb.12172 and Myllymäki et al. 2021, doi:10.1016/j.spasta.2020.100436
(global envelopes); Westfall & Young 1993, Resampling-Based Multiple Testing, Wiley (max-T); Feng et al. 2023,
doi:10.1038/s41467-023-37822-0 (structure-based tissue analysis).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    RngLike,
    TestResult,
    as_rng,
    check_alternative,
    max_t_pvalue,
    mc_pvalue,
    rng_entropy,
)
from spaclone.tl.maps import _binary, _codes, _column, _critical, _groups, _strata

__all__ = ["niche_profile"]

logger = logging.getLogger("spaclone")


def _resolve_blocks(
    table: pd.DataFrame, blocks: Any, clone_key: str | None, g: np.ndarray, valid: np.ndarray, st: np.ndarray
) -> tuple[np.ndarray | None, str]:
    """Exchangeable block of every valid unit (``None``: units one by one) and a description of the rule.

    ``blocks="auto"``: a label that is constant within every clone (all of a clone's units in the group or none) is
    a clone-level label, so whole clones (within each stratum) are the exchangeable blocks; a label that varies within
    clones is a unit-level mark. Units without a clone are blocks of their own.
    """
    if blocks is None:
        return None, "units"
    if not (isinstance(blocks, str) and blocks == "auto"):
        return _codes(_column(table, blocks, "blocks")[valid], f"blocks column {blocks!r}"), str(blocks)
    if clone_key is None or clone_key not in table.columns:
        logger.info("niche_profile: no clone column %r; units are exchanged one by one", clone_key)
        return None, "units (no clone column)"
    cl = pd.Series(table[clone_key].to_numpy()[valid])
    codes = pd.factorize(cl, sort=True)[0].astype(np.int64)
    gv = g[valid]
    has = codes >= 0
    n_cl = int(codes.max()) + 1 if has.any() else 0
    n_in = np.bincount(codes[has], weights=gv[has].astype(float), minlength=n_cl)
    n_all = np.bincount(codes[has], minlength=n_cl)
    if ((n_in > 0) & (n_in < n_all)).any():
        return None, "units (the group label varies within clones)"
    sv = st[valid]
    key = np.where(has, codes * (int(sv.max()) + 1) + sv, -1 - np.arange(len(codes)))
    return _codes(key, "blocks"), f"{clone_key} (auto)"


def _band_labels(edges: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    lo = np.r_[-np.inf, edges]
    hi = np.r_[edges, np.inf]
    w0 = edges[1] - edges[0] if len(edges) > 1 else 0.0
    w1 = edges[-1] - edges[-2] if len(edges) > 1 else 0.0
    mid = np.r_[edges[0] - w0 / 2, (edges[:-1] + edges[1:]) / 2, edges[-1] + w1 / 2]
    labels = [f"<{edges[0]:g}"] + [f"[{a:g}, {b:g})" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    labels.append(f">={edges[-1]:g}")
    return labels, lo, hi, mid


def _covariate(
    table: pd.DataFrame, distance_um: Any, bands_um: Any, compartment: Any
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Band (or category) code of every unit, validity, labels, band limits and mid-points."""
    if (distance_um is None) == (compartment is None):
        raise ValueError("give exactly one of distance_um (with bands_um) or compartment")
    if distance_um is not None:
        if bands_um is None:
            raise ValueError("distance_um needs bands_um (band edges in µm)")
        edges = np.asarray(bands_um, dtype=float)
        if edges.ndim != 1 or len(edges) < 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError("bands_um must be finite, increasing band edges")
        val = np.asarray(_column(table, distance_um, "distance_um"), dtype=float)
        ok = ~np.isnan(val)
        band = np.digitize(np.where(ok, val, 0.0), edges)
        labels, lo, hi, mid = _band_labels(edges)
        return band, ok, labels, lo, hi, mid
    if bands_um is not None:
        raise ValueError("bands_um applies to distance_um, not to compartment")
    if isinstance(compartment, str):
        if compartment not in table.columns:
            raise KeyError(f"compartment: column {compartment!r} not in the table")
        ser = table[compartment].reset_index(drop=True)  # keeps a Categorical's order
    else:
        ser = compartment.reset_index(drop=True) if isinstance(compartment, pd.Series) else pd.Series(compartment)
        if len(ser) != len(table):
            raise ValueError(f"compartment: expected {len(table)} values, got {len(ser)}")
    ok = ser.notna().to_numpy()
    cats = list(ser.cat.categories) if isinstance(ser.dtype, pd.CategoricalDtype) else sorted(ser[ok].unique().tolist())
    lookup = {c: k for k, c in enumerate(cats)}
    band = np.array([lookup.get(v, -1) if o else -1 for v, o in zip(ser.tolist(), ok, strict=True)], dtype=np.int64)
    nan = np.full(len(cats), np.nan)
    return band, ok, [str(c) for c in cats], nan, nan.copy(), nan.copy()


def niche_profile(
    table: pd.DataFrame,
    *,
    group: str | Any,
    distance_um: str | Any | None = None,
    bands_um: Sequence[float] | None = None,
    compartment: str | Any | None = None,
    baseline: str | Any | None = None,
    pool: Literal["union", "baseline"] = "union",
    n_perm: int = 1000,
    within: str | Sequence[str] | None = None,
    blocks: str | None = "auto",
    clone_key: str | None = S.CLONE,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    structure_key: str | None = None,
    n_boot: int = 0,
    rng: RngLike = None,
    keep_null: bool = True,
) -> TestResult:
    """Profile of a group of units across distance bands (or compartments) against a baseline, with chance bands.

    Parameters
    ----------
    table
        One row per unit (loci), holding the group label, the baseline label and the position covariate.
    group
        Boolean column (or array): the labelled group (e.g. iNKT alpha loci, loci of expanded clones).
    distance_um
        Column (or array) of signed distances in µm to a structure border, e.g.
        :func:`~spaclone.tl.signed_distance` sampled at the units (negative inside); used with ``bands_um``. Units with
        a missing value are left out; ``±inf`` falls in the outer bands.
    bands_um
        Band edges ``e_0 < … < e_m`` in µm; bands are ``(−inf, e_0)``, ``[e_0, e_1)``, …, ``[e_m, +inf)`` (the tails
        are pooled, as ``numpy.digitize``).
    compartment
        Instead of a distance: a categorical column (or array), e.g. :func:`~spaclone.tl.compartments` sampled at the
        units; categories in sorted order, or the order of a pandas ``Categorical``.
    baseline
        Boolean column (or array) of the reference units (e.g. all other alpha loci). Default: every unit of the
        table (the group included).
    pool
        Units the group label is redistributed over under the null: ``"union"`` = group ∪ baseline (random
        labelling) or ``"baseline"`` = a group-sized sample of the baseline units (the group's bands are a sample of
        the baseline's).
    n_perm
        Number of random draws.
    within
        Stratum column(s): the group label is redistributed within each stratum (same count per stratum).
    blocks
        Exchangeable blocks whose units carry the label together. ``"auto"`` (default): when ``clone_key`` is a column
        and the group label is constant within every clone (all of a clone's units in the group or none, e.g. loci of
        expanded clones), whole clones (within each ``within`` stratum) are the blocks; a label that varies within
        clones, or a table without the clone column, keeps units one by one. A column name: those blocks, e.g. the
        merged structure id (a germinal centre spanning several sections) when the label belongs to a structure.
        ``None``: units one by one (the archive's TR3 / TR7 rule), valid only when the units are independent cells --
        e.g. invariant iNKT alpha loci, whose shared "clone" is many unrelated cells, not a lineage.
    clone_key
        Clone column used by ``blocks="auto"``.
    alternative
        Direction of the pointwise and family-wise p-values.
    alpha
        Level of the pointwise bands (``α/2`` and ``1 − α/2`` quantiles) and of the global envelope.
    structure_key
        Column naming the structure (cluster) of each unit; with ``n_boot > 0`` the observed ratio gets a
        percentile confidence interval from resampling whole structures with replacement (cluster bootstrap).
    n_boot
        Number of cluster-bootstrap replicates (0: none).
    rng
        Seed or generator. Each null draw is one ``rng.choice(pool, size, replace=False)`` per stratum (block draws:
        over blocks); bootstrap draws follow the null draws.
    keep_null
        Keep the ``(n_perm, bands)`` null shares.

    Returns
    -------
    TestResult
        ``table``: one row per band with ``band``, ``label``, ``lo_um``, ``hi_um``, ``mid_um``, ``group_units``,
        ``baseline_units``, ``statistic`` (the group's share in the band), ``expected`` (the baseline's share),
        ``ratio`` (statistic / expected: the rhohat ratio), ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``
        (pointwise ``α/2`` / ``1 − α/2`` quantiles of the drawn share), ``ratio_lo``, ``ratio_hi`` (the same over
        ``expected``), ``env_lo``, ``env_hi`` and ``ratio_env_lo``, ``ratio_env_hi`` (studentised global envelope at
        level α), ``z``, ``p_value`` (pointwise Monte-Carlo), ``p_maxT`` (family-wise over bands), ``n_perm`` and, with
        a bootstrap, ``ci_lo``, ``ci_hi`` of the ratio. ``summary["global_p"]`` is the smallest ``p_maxT``: the
        global test of "the group follows the baseline across all bands"; ``summary["blocks"]`` names the
        exchangeable unit used and ``summary["n_group_blocks"]`` the number of group blocks.

    Notes
    -----
    Null model: random labelling conditional on positions. The covariate value of every unit (its band) and the
    baseline units are fixed; the group label goes to a uniformly random subset of the pool units of the group's size
    (per stratum), or -- with blocks -- to a uniformly random set of as many pool blocks as the group holds, whole
    blocks with all their units (per stratum). The bands are many tests: read single bands against the global envelope
    or ``p_maxT``, not the pointwise band. Derive the border from data that do not define the group (a circular border
    makes any profile significant), and floor surroundings beyond the border halo (bins within ~20 µm of a called
    germinal-centre border are ~25% inside).

    The loci of one expanded clone sit together, so a clone-level label shuffled locus by locus is pseudoreplicated:
    with 6 expanded clones of 25 loci placed independently of the border, unit shuffles rejected 97-99 % of null
    tables at alpha = 0.05 and clone blocks (the default) 0-1 %; power for the same clones placed on the border
    0.86 (F4a study, 300 tables each; review 2026-10-30, M2). Block draws ignore block size: when the group's clones
    are much larger than the pool's other clones the test is conservative.

    The two-sided pointwise p doubles the smaller one-sided Monte-Carlo p (``spaclone._utils.mc_pvalue``); the
    2026-10 archive (TR3) counted draws at least as far from the draw mean instead. Both are valid; the returned
    ``null`` lets either be computed.

    References
    ----------
    Baddeley et al. 2012, doi:10.4310/SII.2012.v5.n2.a7; Myllymäki et al. 2017, doi:10.1111/rssb.12172;
    Nichols & Holmes 2002, doi:10.1002/hbm.1058; Hurlbert 1984, doi:10.2307/1942661 (pseudoreplication); Davison &
    Hinkley 1997, doi:10.1017/CBO9780511802843 (Monte-Carlo p-values and the cluster bootstrap).
    """
    check_alternative(alternative)
    if pool not in ("union", "baseline"):
        raise ValueError("pool must be 'union' or 'baseline'")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if n_boot and structure_key is None:
        raise ValueError("n_boot needs structure_key (the structure of each unit)")
    gen = as_rng(rng)
    g = _binary(_column(table, group, "group"), "group").astype(bool)
    b = np.ones(len(table), bool) if baseline is None else _binary(_column(table, baseline, "baseline"), "baseline")
    b = b.astype(bool)
    band, ok, labels, lo, hi, mid = _covariate(table, distance_um, bands_um, compartment)
    nb = len(labels)
    valid = ok & (g | b)
    in_pool = valid if pool == "union" else valid & b
    gu, bu = valid & g, valid & b
    n_g, n_b = int(gu.sum()), int(bu.sum())
    if n_g == 0 or n_b == 0:
        raise ValueError(f"need group and baseline units with a covariate value (group {n_g}, baseline {n_b})")
    obs = np.bincount(band[gu], minlength=nb) / n_g
    ref = np.bincount(band[bu], minlength=nb) / n_b

    # ---- null draws
    st = _strata(table, within)
    st = np.zeros(len(table), dtype=np.int64) if st is None else st
    bl, block_rule = _resolve_blocks(table, blocks, clone_key, g, valid, st)
    n_group_blocks = n_g if bl is None else int(len(np.unique(bl[g[valid]])))
    null = np.empty((n_perm, nb))
    if bl is None:
        groups = []
        for s in np.unique(st[valid]):
            pos = np.flatnonzero(in_pool & (st == s))
            k = int((gu & (st == s)).sum())
            if k > len(pos):
                raise ValueError(f"stratum {s}: {k} group units but only {len(pos)} pool units")
            if k:
                groups.append((pos, k))
        n_drawn = sum(k for _, k in groups)
        for i in range(n_perm):
            drawn = np.concatenate([pos[gen.choice(len(pos), k, replace=False)] for pos, k in groups])
            null[i] = np.bincount(band[drawn], minlength=nb) / n_drawn
    else:
        vpos = np.flatnonzero(valid)
        nblk = int(bl.max()) + 1
        gb = np.zeros(nblk, bool)
        gb[bl[g[vpos]]] = True
        if (gb[bl] != g[vpos]).any():
            raise ValueError("blocks: the group label varies inside a block")
        bst = np.full(nblk, -1)
        bst[bl] = st[vpos]
        if (bst[bl] != st[vpos]).any():
            raise ValueError("blocks: a stratum varies inside a block")
        pool_units = in_pool[vpos]
        pool_blk = np.zeros(nblk, bool)
        pool_blk[bl[pool_units]] = True
        pu = np.flatnonzero(pool_units)
        members = dict(zip(np.unique(bl[pu]), _groups(bl[pu]), strict=True))
        members = {c: pu[idx] for c, idx in members.items()}
        groups = []
        for s in np.unique(bst):
            cand = np.flatnonzero(pool_blk & (bst == s))
            k = int((gb & (bst == s)).sum())
            if k > len(cand):
                raise ValueError(f"stratum {s}: {k} group blocks but only {len(cand)} pool blocks")
            if k:
                groups.append((cand, k))
        bvals = band[vpos]
        for i in range(n_perm):
            chosen = np.concatenate([cand[gen.choice(len(cand), k, replace=False)] for cand, k in groups])
            units = np.concatenate([members[c] for c in chosen])
            null[i] = np.bincount(bvals[units], minlength=nb) / len(units)

    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1) if n_perm > 1 else np.full(nb, np.nan)
    qlo, qhi = np.quantile(null, alpha / 2, axis=0), np.quantile(null, 1 - alpha / 2, axis=0)
    sd_safe = np.where(sd > 0, sd, np.inf)
    zn = (null - mu) / sd_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (obs - mu) / sd_safe, np.nan)
    env_sd = np.where(np.isfinite(sd), sd, 0.0)
    if alternative == "two-sided":
        c = _critical(np.abs(zn).max(axis=1), alpha, "rank")
        env_lo, env_hi = mu - c * env_sd, mu + c * env_sd
    elif alternative == "greater":
        c = _critical(zn.max(axis=1), alpha, "rank")
        env_lo, env_hi = np.full(nb, -np.inf), mu + c * env_sd
    else:
        c = _critical((-zn).max(axis=1), alpha, "rank")
        env_lo, env_hi = mu - c * env_sd, np.full(nb, np.inf)
    p = mc_pvalue(null, obs, alternative=alternative)
    p_fw = max_t_pvalue(null, obs, alternative=alternative)
    with np.errstate(divide="ignore", invalid="ignore"):
        tab = pd.DataFrame(
            {
                "band": np.arange(nb),
                "label": labels,
                "lo_um": lo,
                "hi_um": hi,
                "mid_um": mid,
                "group_units": np.bincount(band[gu], minlength=nb),
                "baseline_units": np.bincount(band[bu], minlength=nb),
                "statistic": obs,
                "expected": ref,
                "ratio": obs / ref,
                "null_mean": mu,
                "null_sd": sd,
                "null_lo": qlo,
                "null_hi": qhi,
                "ratio_lo": qlo / ref,
                "ratio_hi": qhi / ref,
                "env_lo": env_lo,
                "env_hi": env_hi,
                "ratio_env_lo": env_lo / ref,
                "ratio_env_hi": env_hi / ref,
                "z": z,
                "p_value": p,
                "p_maxT": p_fw,
                "n_perm": n_perm,
            }
        )

    # ---- cluster bootstrap of the observed ratio over structures
    if n_boot > 0:
        sid = _codes(_column(table, structure_key, "structure_key")[valid], f"structure column {structure_key!r}")
        ns = int(sid.max()) + 1
        vband = band[valid]
        G = np.zeros((ns, nb))
        R = np.zeros((ns, nb))
        np.add.at(G, (sid[g[valid]], vband[g[valid]]), 1.0)
        np.add.at(R, (sid[b[valid]], vband[b[valid]]), 1.0)
        mult = np.stack([np.bincount(gen.integers(0, ns, ns), minlength=ns) for _ in range(n_boot)])
        gbt, rbt = mult @ G, mult @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = (gbt / gbt.sum(axis=1, keepdims=True)) / (rbt / rbt.sum(axis=1, keepdims=True))
        rat = np.where(np.isfinite(rat), rat, np.nan)
        with np.errstate(all="ignore"):
            tab["ci_lo"] = np.nanquantile(rat, alpha / 2, axis=0) if np.isfinite(rat).any() else np.nan
            tab["ci_hi"] = np.nanquantile(rat, 1 - alpha / 2, axis=0) if np.isfinite(rat).any() else np.nan
    finite = np.isfinite(p_fw)
    summary = {
        "n_group": n_g,
        "n_baseline": n_b,
        "n_pool": int(in_pool.sum()),
        "bands": nb,
        "global_p": float(np.min(p_fw[finite])) if finite.any() else np.nan,
        "most_extreme": labels[int(np.argmin(np.where(finite, p_fw, np.inf)))] if finite.any() else None,
        "n_missing": int((~ok & (g | b)).sum()),
        "blocks": block_rule,
        "n_group_blocks": n_group_blocks,
    }
    params = {
        "covariate": "distance_um" if distance_um is not None else "compartment",
        "column": (
            distance_um if isinstance(distance_um, str) else compartment if isinstance(compartment, str) else "array"
        ),
        "bands_um": None if bands_um is None else [float(v) for v in bands_um],
        "pool": pool,
        "n_perm": n_perm,
        "within": within,
        "blocks": blocks,
        "blocks_used": block_rule,
        "clone_key": clone_key,
        "alternative": alternative,
        "alpha": alpha,
        "structure_key": structure_key,
        "n_boot": n_boot,
        "design": {
            "design": "random labelling (group label to a random subset of the pool, without replacement)",
            "pool": pool,
            "within": within,
            "blocks": block_rule,
        },
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="niche_profile", table=tab, summary=summary, params=params, null=null if keep_null else None)
