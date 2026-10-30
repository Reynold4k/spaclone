"""Clone sharing between regions (e.g. germinal centres) compared with random placement.

For the clones with at least ``min_records`` records inside regions, two measures summarise how clones spread over
regions: the share of these clones present in two or more regions, and the mean share of a clone's in-region records
that sit in its main region. Clones that expand inside one germinal centre and stay there give few multi-region
clones and a high main-region share; clones seeded into many germinal centres, or receptor records placed at random,
give the opposite.

References
----------
Besag & Diggle 1977, doi:10.2307/2346974 (Monte-Carlo tests); Davison & Hinkley 1997,
doi:10.1017/CBO9780511802843; Jiang et al. 2026 (AIR-SPACE; multi-germinal-centre clones),
doi:10.1073/pnas.2504742123.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    check_alternative,
    mc_pvalue,
    rng_entropy,
)
from spaclone.nulls import NullDesign

__all__ = ["region_sharing"]

_BUILTIN = {
    "regions": "clone labels permuted among the chain's records inside regions",
    "tissue": "clone labels permuted among all of the chain's records (random placement over the records' positions)",
}
_MEASURES = ("multi_region_share", "main_region_share")
_DENSE_MAX = 1 << 22  # largest clone x region table counted densely


def _inside(region: pd.Series, background: Any) -> np.ndarray:
    """Records inside a region: not missing, not negative (numeric ids) and not ``background``."""
    ok = region.notna().to_numpy()
    num = pd.to_numeric(region, errors="coerce")
    if num.notna().sum() == region.notna().sum():
        ok = ok & ~(num.to_numpy() < 0)
    return ok & (region.to_numpy() != background)


def _stats(
    clone: np.ndarray, region: np.ndarray, n_regions: int, min_records: int, weights: np.ndarray | None = None
) -> tuple[float, float, int]:
    """(share of clones in >= 2 regions, mean main-region share, clones) over clones with >= min_records units.

    Presence and eligibility count units; the main-region share is computed on ``weights`` (records per unit) when
    given, so that merging two cells of one clone into one unit does not change it.
    """
    if not len(clone):
        return np.nan, np.nan, 0
    key = clone.astype(np.int64) * n_regions + region
    n_cells = (int(clone.max()) + 1) * n_regions
    if n_cells <= _DENSE_MAX:  # clone x region count table (same values, same clone order as the sparse path)
        tab = np.bincount(key, minlength=n_cells).reshape(-1, n_regions)
        tot = tab.sum(axis=1)
        keep = tot >= min_records
        if not keep.any():
            return np.nan, np.nan, 0
        sub = tab[keep]
        if weights is None:
            main = sub.max(axis=1) / tot[keep]
        else:
            wt = np.bincount(key, weights=weights, minlength=n_cells).reshape(-1, n_regions)[keep]
            main = wt.max(axis=1) / wt.sum(axis=1)
        return float(((sub > 0).sum(axis=1) >= 2).mean()), float(main.mean()), int(keep.sum())
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cl = uk // n_regions
    starts = np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]])
    tot = np.add.reduceat(cnt, starts)
    nreg = np.diff(np.r_[starts, len(cl)])
    keep = tot >= min_records
    if not keep.any():
        return np.nan, np.nan, 0
    if weights is None:
        main = np.maximum.reduceat(cnt, starts) / tot
    else:
        w = np.bincount(inv, weights=weights, minlength=len(uk))
        main = np.maximum.reduceat(w, starts) / np.add.reduceat(w, starts)
    return float((nreg[keep] >= 2).mean()), float(main[keep].mean()), int(keep.sum())


def _first_rows(codes: np.ndarray) -> np.ndarray:
    """Position of the first row of every code (codes 0..k-1)."""
    first = np.full(int(codes.max()) + 1 if len(codes) else 0, -1, np.int64)
    first[codes[::-1]] = np.arange(len(codes))[::-1]
    return first


def _units(
    sub: pd.DataFrame,
    *,
    unit_key: str | None,
    region_key: str,
    clone_key: str,
    spot_key: str,
    x_key: str,
    y_key: str,
    library_key: str,
    loci_radius_um: float,
) -> tuple[pd.DataFrame, str]:
    """One row per cell-scale unit of a chain's records (the unit's first record, ``n_records`` added).

    Units never span regions or sections: the region (missing values included) and the library are part of the key,
    so every unit has one region. ``"auto"``: ``locus_id`` column, else same-clone spots within ``loci_radius_um``
    (:func:`spaclone.pp.merge_loci`), else one unit per (spot, clone), else records with a warning.
    """
    if unit_key is None:
        return sub, "records"
    cols = sub.columns
    rule = unit_key
    if unit_key == "auto":
        if S.LOCUS_ID in cols:
            rule = S.LOCUS_ID
        elif all(c in cols for c in (spot_key, x_key, y_key)):
            rule = "loci"
        elif spot_key in cols:
            rule = "spot"
        else:
            warnings.warn(
                "region_sharing: no locus, spot or coordinate columns to tell which records come from one cell; "
                "records are counted and shuffled one by one, which over-states how confined clones are when a cell "
                "contributes several molecules. Pass spot_key / x_key / y_key, unit_key=, or unit_key=None.",
                SpacloneWarning,
                stacklevel=3,
            )
            return sub, "records (no unit column)"
    reg = sub[region_key].astype(object).where(sub[region_key].notna(), "\x00missing")
    lib = [sub[library_key].astype(object).to_numpy()] if library_key in cols else []
    if rule == "loci":
        missing = [c for c in (spot_key, x_key, y_key) if c not in cols]
        if missing:
            raise KeyError(f"region_sharing: unit_key='loci' needs column(s) {missing}")
        from spaclone.pp.loci import merge_loci

        t = pd.DataFrame(
            {
                S.SPOT: sub[spot_key].astype(str).to_numpy(),
                S.X: sub[x_key].to_numpy(dtype=float),
                S.Y: sub[y_key].to_numpy(dtype=float),
                S.CLONE: sub[clone_key].to_numpy(),
                "_reg": reg.to_numpy(),
                **({"_lib": lib[0]} if lib else {}),
            }
        )
        code = merge_loci(t, radius_um=loci_radius_um, by=["_reg"] + (["_lib"] if lib else [])).membership.to_numpy()
        rule = f"loci ({loci_radius_um:g} um)"
    else:
        key_col = spot_key if rule == "spot" else rule
        if key_col not in cols:
            raise KeyError(f"region_sharing: unit column {key_col!r} not in the table")
        key = pd.MultiIndex.from_arrays(
            [sub[clone_key].astype(object).to_numpy(), sub[key_col].astype(object).to_numpy(), reg.to_numpy(), *lib]
        )
        code = pd.factorize(key, sort=True)[0].astype(np.int64)
        rule = f"({key_col}, clone)" if rule == "spot" else str(rule)
    first = _first_rows(code)
    units = sub.iloc[first].reset_index(drop=True)
    units["n_records"] = np.bincount(code, minlength=len(first)).astype(np.int64)
    return units, rule


def _measures(
    table: pd.DataFrame, region_key: str, clone_key: str, background: Any, min_records: int, weighted: bool = False
) -> tuple:
    """Measures of a (randomised) chain table, recomputed from scratch."""
    ins = _inside(table[region_key], background)
    rc, ru = pd.factorize(table[region_key].where(ins), sort=True)
    cc = pd.factorize(table[clone_key], sort=True)[0]
    pi = np.flatnonzero(ins)
    w = table["n_records"].to_numpy(dtype=float)[pi] if weighted else None
    return _stats(cc[pi], rc[pi], max(len(ru), 1), min_records, w)


def region_sharing(
    air: pd.DataFrame,
    *,
    region_key: str,
    min_records: int = 2,
    design: str | NullDesign | Sequence[str | NullDesign] = ("regions", "tissue"),
    n_perm: int = 500,
    clone_key: str = S.CLONE,
    chain_key: str | None = S.LOCUS,
    background: Any = -1,
    alternative: Alternative | Literal["confined", "dispersed"] = "confined",
    rng: RngLike = None,
    keep_null: bool = False,
    unit_key: str | None = "auto",
    loci_radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    library_key: str = S.LIBRARY,
) -> TestResult:
    """Are clones shared between regions more, or less, than under random placement?

    Parameters
    ----------
    air
        One row per receptor record (molecule, UMI or cell call) with its clone and region.
    region_key
        Column with the region of each record (e.g. germinal-centre id). Missing, negative or ``background`` = not in
        a region; 0 is a region (DBSCAN-style ids, -1 = noise).
    min_records
        Clones need at least this many units (records with ``unit_key=None``) inside regions to be counted.
    design
        Null design(s), evaluated in order within each permutation (one draw per design per permutation):
        ``"regions"`` permutes clone labels among the chain's records inside regions; ``"tissue"`` permutes them
        among all of the chain's records (random placement over the observed record positions); or any
        :class:`~spaclone.nulls.NullDesign` (e.g. ``Permute("clone_id", within="library_id")`` or a ``Relocate``)
        applied to the chain's table, after which the measures are recomputed from scratch.
    n_perm
        Number of permutations.
    clone_key
        Clone column.
    chain_key
        Chain column; measures and shuffles are computed per chain, chains in sorted order (``None``: all records
        together).
    background
        Region value meaning "not in a region".
    alternative
        ``"confined"`` (default): clones are confined to fewer regions than chance (``multi_region_share`` below and
        ``main_region_share`` above the null); ``"dispersed"``: the reverse; ``"two-sided"``: twice the smaller
        one-sided p-value (the archive's rows); ``"greater"`` / ``"less"``: the direction of each measure as it is.
        With cell-scale units the confinement direction is the calibrated one (see Notes).
    rng
        Seed or generator.
    keep_null
        Keep the null draws (shape ``(n_perm, rows)`` in table order).
    unit_key
        Cell-scale unit in which records are moved and counted, so that the molecules of one cell count once.
        ``"auto"`` (default): the ``locus_id`` column when present, else same-clone spots within ``loci_radius_um``
        merged by single linkage (:func:`spaclone.pp.merge_loci`; needs the spot and coordinate columns), else one
        unit per (spot, clone), else records, with a :class:`~spaclone._utils.SpacloneWarning`. ``"loci"``,
        ``"spot"`` or a column name force a rule; ``None``: every record is a unit (the archive's X3b rule).
    loci_radius_um
        Merge radius of locus units (µm).
    spot_key, x_key, y_key, library_key
        Columns used to build units (units never span sections).

    Returns
    -------
    TestResult
        ``table``: one row per chain × design × measure (``multi_region_share``, ``main_region_share``) with
        ``statistic`` (observed value), ``clones``, ``regions``, ``records_in_regions`` (units inside regions),
        ``units`` (the unit rule), ``null_mean``, ``null_sd``, ``z``, ``p_value``, ``alternative`` (of the row),
        ``n_perm``.

    Notes
    -----
    Null model: ``"regions"`` keeps every unit's region and each region's number of units of the chain, and makes
    which clone a region unit belongs to random; ``"tissue"`` keeps all unit positions and each clone's number of
    units, and places each clone's units at random among all of the chain's units. A unit moves with all its records.
    Units never span regions or sections. Presence in a region and the ``min_records`` rule count units; the
    main-region share weighs units by their records, so that two cells of one clone merged into one locus count as
    they would separately (counting merged loci once makes clones look spread: same-clone loci are never within
    ``loci_radius_um`` of each other, random labels are). Deduplicate molecules (barcode, UMI, chain) first.

    With records as units (``unit_key=None``), the several molecules of one cell are pseudoreplicates: they share the
    cell's region, so clones look confined to one region far more often than records shuffled one by one predict
    (100 % of null chips at alpha = 0.05 in the F4a study, for both designs and both measures). Cell-scale units
    remove this, but a locus holds two cells of one clone only when they lie close together in one region, which
    random labels of loci do not reproduce: relabelled clones are a little more confined than real ones. The
    confinement direction is therefore conservative (false-positive rate 0.3-3 % at alpha = 0.05 for both measures and
    designs, bead and sub-cellular data) and the dispersion direction anti-conservative for ``main_region_share``
    (15-29 %; ``multi_region_share`` 4-8 %); a two-sided test inherits the latter (F4a study, 300 null chips each).

    References
    ----------
    Besag & Diggle 1977, doi:10.2307/2346974; Davison & Hinkley 1997, doi:10.1017/CBO9780511802843; Jiang et al.
    2026, doi:10.1073/pnas.2504742123.
    """
    if alternative not in ("confined", "dispersed"):
        check_alternative(alternative)
    alt_of = {
        "confined": {_MEASURES[0]: "less", _MEASURES[1]: "greater"},
        "dispersed": {_MEASURES[0]: "greater", _MEASURES[1]: "less"},
    }.get(alternative, dict.fromkeys(_MEASURES, alternative))
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    for c in (region_key, clone_key):
        if c not in air.columns:
            raise KeyError(f"region_sharing: column {c!r} not in the table")
    designs = [design] if isinstance(design, str | NullDesign) else list(design)
    if not designs:
        raise ValueError("give at least one design")
    for d in designs:
        if isinstance(d, str) and d not in _BUILTIN:
            raise ValueError(f"design must be one of {sorted(_BUILTIN)} or a NullDesign, got {d!r}")
        if not isinstance(d, str | NullDesign):
            raise TypeError(f"design must be a name or a NullDesign, got {type(d).__name__}")
    names = [d if isinstance(d, str) else repr(d) for d in designs]
    gen = as_rng(rng)
    if chain_key is not None and chain_key in air.columns:
        chains = sorted(air[chain_key].dropna().unique().tolist())
        parts = [(c, air[(air[chain_key] == c).to_numpy()]) for c in chains]
    elif chain_key is not None:
        raise KeyError(f"region_sharing: chain column {chain_key!r} not in the table (chain_key=None pools chains)")
    else:
        parts = [("all", air)]
    rows, nulls = [], []
    unit_rules = {}
    for chain, sub in parts:
        sub, unit_rules[chain] = _units(
            sub.reset_index(drop=True),
            unit_key=unit_key,
            region_key=region_key,
            clone_key=clone_key,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            library_key=library_key,
            loci_radius_um=loci_radius_um,
        )
        sub = sub.reset_index(drop=True)
        inside = _inside(sub[region_key], background)
        rcodes, runiq = pd.factorize(sub[region_key].where(inside), sort=True)
        n_reg = max(len(runiq), 1)
        ccodes = pd.factorize(sub[clone_key], sort=True)[0]
        if (ccodes < 0).any():
            raise ValueError(f"{clone_key!r} has missing values")
        ipos = np.flatnonzero(inside)
        weighted = unit_rules[chain] not in ("records", "records (no unit column)")
        w_in = sub["n_records"].to_numpy(dtype=float)[ipos] if weighted else None
        obs_m, obs_s, n_cl = _stats(ccodes[ipos], rcodes[ipos], n_reg, min_records, w_in)
        draws = np.full((n_perm, len(designs), 2), np.nan)
        for b in range(n_perm):
            for k, d in enumerate(designs):
                if d == "regions":
                    m, s, _ = _stats(ccodes[ipos][gen.permutation(len(ipos))], rcodes[ipos], n_reg, min_records, w_in)
                elif d == "tissue":
                    lab = ccodes[gen.permutation(len(ccodes))]
                    m, s, _ = _stats(lab[ipos], rcodes[ipos], n_reg, min_records, w_in)
                else:
                    r = d(sub, gen).reset_index(drop=True)
                    m, s, _ = _measures(r, region_key, clone_key, background, min_records, weighted)
                draws[b, k] = (m, s)
        for k, name in enumerate(names):
            for j, measure, o in ((0, _MEASURES[0], obs_m), (1, _MEASURES[1], obs_s)):
                nl = draws[:, k, j]
                fin = nl[np.isfinite(nl)]
                mu = float(fin.mean()) if len(fin) else np.nan
                sd = float(fin.std(ddof=1)) if len(fin) > 1 else np.nan
                rows.append(
                    {
                        "chain": chain,
                        "design": name,
                        "measure": measure,
                        "statistic": o,
                        "clones": n_cl,
                        "regions": len(runiq),
                        "records_in_regions": len(ipos),
                        "units": unit_rules[chain],
                        "null_mean": mu,
                        "null_sd": sd,
                        "z": (o - mu) / sd if sd > 0 else np.nan,
                        "p_value": float(mc_pvalue(nl, o, alternative=alt_of[measure])[0]),
                        "alternative": alt_of[measure],
                        "n_perm": n_perm,
                    }
                )
                nulls.append(nl)
    tab = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for r in tab.itertuples():
        if r.design == names[0]:
            summary[f"{r.chain} {r.measure}"] = r.statistic
            summary[f"{r.chain} {r.measure} null[{r.design}]"] = r.null_mean
    params = {
        "region_key": region_key,
        "clone_key": clone_key,
        "chain_key": chain_key,
        "min_records": min_records,
        "unit_key": unit_key,
        "units": unit_rules,
        "loci_radius_um": loci_radius_um,
        "design": [_BUILTIN[d] if isinstance(d, str) else d.describe() for d in designs],
        "n_perm": n_perm,
        "background": background,
        "alternative": alternative,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(
        name="region_sharing",
        table=tab,
        summary=summary,
        params=params,
        null=np.column_stack(nulls) if keep_null and nulls else None,
    )
