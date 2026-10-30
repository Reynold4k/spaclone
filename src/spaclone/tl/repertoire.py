"""Repertoire descriptives made spatial: clone sizes, clonal expansion, diversity, overlap, V/J usage, spectratype.

Single-cell repertoire toolkits (scirpy, immunarch, scRepertoire) describe a repertoire per sample. Spatial data ask the
same questions per region, niche or section, and two things change.

* **The unit.** One cell can spill receptor molecules over several spots, and one spot can hold several cells, so a clone
  is counted in *units*: records (molecules, contigs or cell calls), spots (one clone on one spot counts once) or loci
  (same-clone spots within ``radius_um`` merged, about one cell; :func:`spaclone.pp.merge_loci`). :func:`clone_units`
  builds them and every function here takes ``unit=``.
* **The comparison.** Regions hold different numbers of units, and several statistics have built-in answers at
  different sample sizes: richness, Chao1, the share of expanded clones and the Gini coefficient all move with the
  number of units. Groups are therefore compared after rarefaction to a common number of units and against the same
  statistic re-computed under a null design. The test of a difference in diversity is random placement of intact
  clones (``placebo="clones"``, :class:`CloneRelocation`): the cells of a clone sit together in tissue, so the null
  must move them together. Shuffling clone labels over the units (``placebo="labels"``) shows only how large a
  difference the unit counts produce; its p-values reject whenever clones are spatially clustered.

Example: in the AIR-SPACE lymph nodes the germinal centres hold fewer IGH molecules than the plasma-cell area and the
Gini coefficient of clone sizes is lower there (0.13-0.44 vs 0.46-0.57), and clone labels shuffled over the tissue give
a similar difference, so the raw contrast is mostly a unit-count effect (archive ``X3b``). Rarefied to the GC's size
and tested against random placement of intact clones, the GC Gini is lower than placement predicts on days 10 and 21
(p 0.004-0.02) and not different on days 7 and 14 (validation ``m9``).
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import (
    LowPowerWarning,
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    rng_entropy,
)
from spaclone.nulls import NullDesign, Permute
from spaclone.nulls import placebo as run_placebo
from spaclone.pp.clones import gene_name
from spaclone.pp.loci import merge_loci

__all__ = [
    "DIVERSITY_METRICS",
    "OVERLAP_METRICS",
    "UNITS",
    "CloneRelocation",
    "CloneUnits",
    "cdr3_length",
    "clonal_expansion",
    "clone_sizes",
    "clone_units",
    "diversity",
    "gene_usage",
    "overlap",
]

#: Units in which clones are counted (see :func:`clone_units`).
UNITS: tuple[str, ...] = ("locus", "spot", "record")
#: Named diversity metrics of :func:`diversity` (plus ``"top<k>"``, e.g. ``"top10"``, and callables).
DIVERSITY_METRICS: tuple[str, ...] = (
    "richness",
    "shannon",
    "simpson",
    "inverse_simpson",
    "evenness",
    "gini",
    "chao1",
    "expanded",
)
#: Similarity metrics of :func:`overlap`; each is 1 for identical and 0 for disjoint repertoires.
OVERLAP_METRICS: tuple[str, ...] = ("morisita_horn", "jaccard", "sorensen", "bhattacharyya")

Unit = Literal["locus", "spot", "record"]
_TOP = re.compile(r"^top(\d+)$")
_RESERVED = {
    "kind",
    "group",
    "group_a",
    "group_b",
    "metric",
    "n_units",
    "n_clones",
    "value",
    "rarefied",
    "rarefied_lo",
    "rarefied_hi",
    "rarefied_to",
    "statistic",
    "expected",
    "null_mean",
    "null_sd",
    "null_lo",
    "null_hi",
    "z",
    "p_value",
    "p_maxT",
    "q_value",
    "n_perm",
}


def _as_list(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key) -> dict[str, str]:
    return {
        "clone_key": clone_key,
        "spot_key": spot_key,
        "x_key": x_key,
        "y_key": y_key,
        "locus_key": locus_key,
        "library_key": library_key,
        "reads_key": reads_key,
        "locus_id_key": locus_id_key,
    }


def _first_seen_codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Group codes numbered in the order in which each group is first seen."""
    if not cols:
        return np.zeros(len(df), np.int64)
    return df.groupby(cols, sort=False, dropna=False, observed=True).ngroup().to_numpy(np.int64)


# ---------------------------------------------------------------------------------------------------------- units


class CloneUnits(NamedTuple):
    """Result of :func:`clone_units` (unpack as ``units, membership = clone_units(air)``).

    Attributes
    ----------
    units
        One row per unit, in the order of each unit's first record in the input.
    membership
        ``unit_id`` (row of ``units``) of every retained record, indexed like the input; records without a clone id,
        without a group value or (for spot and locus units) without a spot are left out.
    """

    units: pd.DataFrame
    membership: pd.Series


def clone_units(
    air: pd.DataFrame,
    *,
    unit: Unit = "locus",
    by: str | Sequence[str] | None = None,
    carry: str | Sequence[str] = (),
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> CloneUnits:
    """Turn receptor records into the units in which clones are counted.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or cell call) with a clone column.
    unit
        ``"record"``: every row counts. ``"spot"``: one clone on one spot counts once. ``"locus"``: same-clone spots
        within ``radius_um`` merged by single linkage (about one cell, :func:`spaclone.pp.merge_loci`); if the table
        already has a ``locus_id_key`` column (for example the membership returned by ``merge_loci``), those loci are
        used as they are.
    by
        Group column(s) (region, niche, section). Units never span groups: loci are merged within each group, so a cell
        that straddles a region border counts once on each side.
    carry
        Further columns whose value is taken from each unit's first record (for example a depth stratum).
    radius_um
        Linkage distance of ``unit="locus"`` in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names. The chain (``locus_key``) and the section (``library_key``) are strata when present: units never
        span chains or sections.

    Returns
    -------
    CloneUnits
        ``units`` has ``unit_id``, the ``by`` columns, the chain and section columns (when present), the ``carry``
        columns, the clone, the spot of the unit's first record, the coordinates (the spot position for records and
        spots, the centroid of the distinct spots for loci), ``n_records``, ``n_spots`` and, with a read column,
        ``reads_max`` and ``reads_sum``. ``membership`` maps every retained record to its unit.

    Notes
    -----
    Units are numbered in the order of their first record, so the result follows the input order; the rarefaction
    draws of :func:`diversity` index this order.
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    by_cols = _as_list(by)
    carry_cols = [c for c in _as_list(carry) if c not in by_cols]
    use_ids = unit == "locus" and locus_id_key in air.columns
    need = [clone_key, *by_cols, *carry_cols]
    if unit != "record":
        need.append(spot_key)
    if unit == "locus" and not use_ids:
        need += [x_key, y_key]
    missing = [c for c in dict.fromkeys(need) if c not in air.columns]
    if missing:
        raise KeyError(f"column(s) {missing} not in the receptor table; pass the matching *_key arguments")
    keep = np.array(air[clone_key].notna().to_numpy(dtype=bool), dtype=bool)
    for c in by_cols + ([spot_key] if unit != "record" else []):
        keep &= air[c].notna().to_numpy(dtype=bool)
    a = air if keep.all() else air.loc[keep]
    strata = [c for c in (library_key, locus_key) if c in a.columns and c not in by_cols and c != clone_key]
    n = len(a)
    if unit == "record":
        code = np.arange(n, dtype=np.int64)
    elif unit == "spot":
        code = _first_seen_codes(a, by_cols + strata + [spot_key, clone_key])
    elif use_ids:
        code = _first_seen_codes(a, by_cols + strata + [locus_id_key])
    else:
        _, member = merge_loci(
            a,
            radius_um=radius_um,
            clone_key=clone_key,
            by=by_cols + strata,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            locus_key=locus_key,
            library_key=library_key,
            reads_key=reads_key,
        )
        code = pd.factorize(member.to_numpy())[0].astype(np.int64)
    n_units = int(code.max()) + 1 if n else 0
    first = np.full(n_units, -1, np.int64)
    first[code[::-1]] = np.arange(n)[::-1]
    has_spot = spot_key in a.columns
    cols = list(dict.fromkeys([*by_cols, *strata, *carry_cols, clone_key] + ([spot_key] if has_spot else [])))
    units = a.iloc[first][cols].reset_index(drop=True)
    units.insert(0, "unit_id", np.arange(n_units, dtype=np.int64))
    n_spots = np.ones(n_units, np.int64)
    if x_key in a.columns and y_key in a.columns:
        x = a[x_key].to_numpy(dtype=float)
        y = a[y_key].to_numpy(dtype=float)
        if unit == "locus":
            site = _first_seen_codes(pd.DataFrame({"u": code, "s": a[spot_key].to_numpy()}), ["u", "s"])
            n_site = int(site.max()) + 1 if n else 0
            sfirst = np.full(n_site, -1, np.int64)
            sfirst[site[::-1]] = np.arange(n)[::-1]
            site_unit = code[sfirst]
            n_spots = np.bincount(site_unit, minlength=n_units).astype(np.int64)
            units[x_key] = np.bincount(site_unit, weights=x[sfirst], minlength=n_units) / np.maximum(n_spots, 1)
            units[y_key] = np.bincount(site_unit, weights=y[sfirst], minlength=n_units) / np.maximum(n_spots, 1)
        else:
            units[x_key] = x[first]
            units[y_key] = y[first]
    elif unit == "locus" and has_spot:
        n_spots = np.asarray(pd.Series(a[spot_key].to_numpy()).groupby(code).nunique().reindex(range(n_units)))
    units["n_records"] = np.bincount(code, minlength=n_units).astype(np.int64)
    units["n_spots"] = n_spots
    if reads_key in a.columns:
        g = pd.Series(a[reads_key].to_numpy(dtype=float)).groupby(code)
        units["reads_max"] = g.max().reindex(range(n_units)).to_numpy(dtype=float)
        units["reads_sum"] = g.sum(min_count=1).reindex(range(n_units)).to_numpy(dtype=float)
    membership = pd.Series(code, index=a.index, name="unit_id")
    return CloneUnits(units, membership)


def _groups(units: pd.DataFrame, by_cols: list[str]) -> tuple[list[tuple], list[np.ndarray]]:
    """Group keys in sorted (or categorical) order and the unit positions of each group, in unit order."""
    if not by_cols:
        return [()], [np.arange(len(units))]
    if len(units) == 0:
        return [], []
    gb = units.groupby(by_cols, sort=True, observed=True, dropna=False)
    codes = gb.ngroup().to_numpy()
    sizes = gb.size()
    keys = [k if isinstance(k, tuple) else (k,) for k in sizes.index]
    order = np.argsort(codes, kind="stable")
    return keys, np.split(order, np.cumsum(sizes.to_numpy())[:-1])


def _label(key: tuple) -> str:
    return "|".join(str(v) for v in key) if key else "all"


# ------------------------------------------------------------------------------------------- sizes and expansion


def _expansion_categories(breakpoints: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    bp = np.array(sorted({int(b) for b in breakpoints}), dtype=np.int64)
    if len(bp) == 0 or bp[0] < 1:
        raise ValueError("breakpoints must be one or more integers >= 1")
    return bp, [f"<= {b}" for b in bp] + [f"> {bp[-1]}"]


def _expansion_labels(sizes: Any, breakpoints: Sequence[int]) -> pd.Categorical:
    bp, cats = _expansion_categories(breakpoints)
    idx = np.searchsorted(bp, np.asarray(sizes, dtype=np.int64), side="left")
    return pd.Categorical.from_codes(idx, categories=cats, ordered=True)


def clone_sizes(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    unit: Unit = "locus",
    breakpoints: Sequence[int] = (1, 2, 5, 20),
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> pd.DataFrame:
    """Size of every clone, per group, in units.

    Parameters
    ----------
    air
        Receptor table with a clone column (and spot / coordinate columns for spot and locus units).
    by
        Group column(s); ``None`` counts over the whole table.
    unit
        ``"locus"``, ``"spot"`` or ``"record"`` (see :func:`clone_units`).
    breakpoints
        Class limits of the ``expansion`` column, as in :func:`scirpy.tl.clonal_expansion`: ``(1, 2, 5, 20)`` gives
        ``"<= 1"``, ``"<= 2"``, ``"<= 5"``, ``"<= 20"`` and ``"> 20"``.
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per (group, clone): the ``by`` columns, the clone, ``size`` (units), ``n_records``, ``n_spots``
        (distinct spots), ``frequency`` (size / units of the group), ``rank`` (1 = largest; ties broken by records,
        then by clone id) and ``expansion``. Sorted by group, then by rank.
    """
    by_cols = _as_list(by)
    cu = clone_units(
        air,
        unit=unit,
        by=by_cols,
        radius_um=radius_um,
        **_keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key),
    )
    u = cu.units
    gcols = [*by_cols, clone_key]
    tab = u.groupby(gcols, sort=False, observed=True, dropna=False).agg(
        size=("unit_id", "size"), n_records=("n_records", "sum"), n_spots=("n_spots", "sum")
    )
    tab = tab.reset_index()
    if unit == "record":
        a = air.loc[cu.membership.index]
        if spot_key in a.columns:
            lib = [library_key] if library_key in a.columns else []
            uid = _first_seen_codes(a, lib + [spot_key])
            d = a[gcols].copy()
            d["_spot"] = uid
            ns = d.groupby(gcols, sort=False, observed=True, dropna=False)["_spot"].nunique().rename("n_spots")
            tab = tab.drop(columns="n_spots").merge(ns.reset_index(), on=gcols, how="left")
            tab["n_spots"] = tab["n_spots"].to_numpy(dtype=np.int64)
        else:
            tab["n_spots"] = np.nan
    total = tab.groupby(by_cols, observed=True, dropna=False)["size"].transform("sum") if by_cols else tab["size"].sum()
    tab["frequency"] = tab["size"] / total
    tab = tab.sort_values(
        [*by_cols, "size", "n_records", clone_key], ascending=[True] * len(by_cols) + [False, False, True]
    ).reset_index(drop=True)
    tab["rank"] = (
        (tab.groupby(by_cols, observed=True, dropna=False).cumcount() + 1) if by_cols else np.arange(len(tab)) + 1
    )
    tab["expansion"] = _expansion_labels(tab["size"], breakpoints)
    return tab


def clonal_expansion(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    unit: Unit = "locus",
    breakpoints: Sequence[int] = (1, 2, 5, 20),
    expanded_in: str | Sequence[str] | None = None,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> pd.DataFrame:
    """Share of each group's units (and clones) in each clone-size class.

    Parameters
    ----------
    air
        Receptor table with a clone column.
    by
        Group column(s) of the summary (region, niche, section); ``None`` summarises the whole table.
    unit
        ``"locus"``, ``"spot"`` or ``"record"`` (see :func:`clone_units`).
    breakpoints
        Class limits, as in :func:`scirpy.tl.clonal_expansion`: ``(1, 2, 5, 20)`` gives ``"<= 1"`` (singletons),
        ``"<= 2"``, ``"<= 5"``, ``"<= 20"`` and ``"> 20"``.
    expanded_in
        Column(s) within which a clone's size is counted (scirpy's ``expanded_in``). ``None`` counts over the whole
        table, so a unit in a region is classed by its clone's size across the section(s); pass ``"library_id"`` to
        count per section, or the ``by`` column to count within each group.
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per (group, class), every class present for every group: the ``by`` columns, ``expansion``
        (ordered categorical), ``n_units``, ``share_units`` (of the group's units), ``n_clones`` and ``share_clones``
        (of the group's distinct clones).

    Notes
    -----
    Sizes counted within each group (``expanded_in=by``) grow with the group's number of units, so groups of different
    size are not comparable that way; compare them with :func:`diversity` (``"expanded"``, ``"top10"``), which
    rarefies. Sizes counted over the whole table (the default) avoid that artefact.
    """
    by_cols = _as_list(by)
    exp_cols = _as_list(expanded_in)
    _, cats = _expansion_categories(breakpoints)
    cu = clone_units(
        air,
        unit=unit,
        by=list(dict.fromkeys(by_cols + exp_cols)),
        radius_um=radius_um,
        **_keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key),
    )
    u = cu.units
    size = u.groupby([*exp_cols, clone_key], observed=True, dropna=False)["unit_id"].transform("size")
    cat_codes = np.asarray(_expansion_labels(size.to_numpy(), breakpoints).codes, dtype=np.int64)
    clone_codes = pd.factorize(u[clone_key].to_numpy())[0]
    keys, pos = _groups(u, by_cols)
    rows = []
    k = len(cats)
    for key, p in zip(keys, pos, strict=True):
        n_u = np.bincount(cat_codes[p], minlength=k)
        pairs = np.unique(clone_codes[p] * k + cat_codes[p])
        n_c = np.bincount(pairs % k, minlength=k)
        n_clones = len(np.unique(clone_codes[p]))
        for i, c in enumerate(cats):
            row = dict(zip(by_cols, key, strict=True))
            row.update(
                expansion=c,
                n_units=int(n_u[i]),
                share_units=n_u[i] / max(len(p), 1),
                n_clones=int(n_c[i]),
                share_clones=n_c[i] / max(n_clones, 1),
            )
            rows.append(row)
    out = pd.DataFrame(rows, columns=[*by_cols, "expansion", "n_units", "share_units", "n_clones", "share_clones"])
    out["expansion"] = pd.Categorical(out["expansion"], categories=cats, ordered=True)
    return out


# --------------------------------------------------------------------------------------------------- diversity


def _gini(c: np.ndarray) -> float:
    x = np.sort(np.asarray(c, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _richness(c: np.ndarray) -> float:
    return float(len(c))


def _shannon(c: np.ndarray) -> float:
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _simpson(c: np.ndarray) -> float:
    p = c / c.sum()
    return float(1.0 - (p * p).sum())


def _inverse_simpson(c: np.ndarray) -> float:
    p = c / c.sum()
    return float(1.0 / (p * p).sum())


def _evenness(c: np.ndarray) -> float:
    return _shannon(c) / np.log(len(c)) if len(c) >= 2 else np.nan


def _chao1(c: np.ndarray) -> float:
    f1 = int((c == 1).sum())
    f2 = int((c == 2).sum())
    return float(len(c) + f1 * (f1 - 1) / (2 * (f2 + 1)))


def _expanded(c: np.ndarray) -> float:
    return float(1.0 - (c == 1).sum() / c.sum())


def _top_share(c: np.ndarray, k: int) -> float:
    return float(np.sort(c)[::-1][:k].sum() / c.sum())


_METRIC_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "richness": _richness,
    "shannon": _shannon,
    "simpson": _simpson,
    "inverse_simpson": _inverse_simpson,
    "evenness": _evenness,
    "gini": _gini,
    "chao1": _chao1,
    "expanded": _expanded,
}


class _Metric(NamedTuple):
    name: str
    fn: Callable[[np.ndarray], float]


def _resolve_metrics(metrics: Any) -> list[_Metric]:
    if isinstance(metrics, str) or callable(metrics):
        metrics = [metrics]
    out = []
    for m in metrics:
        if callable(m):
            out.append(_Metric(getattr(m, "__name__", repr(m)), m))
        elif isinstance(m, str) and m in _METRIC_FUNCS:
            out.append(_Metric(m, _METRIC_FUNCS[m]))
        elif isinstance(m, str) and (t := _TOP.match(m)) and int(t.group(1)) >= 1:
            out.append(_Metric(m, functools.partial(_top_share, k=int(t.group(1)))))
        else:
            raise ValueError(f"unknown diversity metric {m!r}; use one of {DIVERSITY_METRICS}, 'top<k>' or a callable")
    names = [m.name for m in out]
    if len(set(names)) < len(names) or not names:
        raise ValueError(f"metrics must be distinct and non-empty, got {names}")
    return out


def _values(c: np.ndarray, mets: list[_Metric]) -> np.ndarray:
    if len(c) == 0:
        return np.full(len(mets), np.nan)
    return np.array([m.fn(c) for m in mets], dtype=float)


def _mean(d: np.ndarray) -> float:
    if np.isfinite(d).all():
        return float(d.mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(d))


def _quantiles(d: np.ndarray) -> tuple[float, float]:
    if np.isfinite(d).all():
        return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))


def _group_values(
    codes_list: list[np.ndarray],
    mets: list[_Metric],
    rarefy_to: int | None,
    n_rarefy: int,
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Full-sample values and rarefaction summaries (mean, 2.5% and 97.5%) per group and metric.

    Draws are taken group by group, ``n_rarefy`` times ``gen.choice(N, rarefy_to, replace=False)`` each; a group with
    exactly ``rarefy_to`` units is used whole without drawing.
    """
    g_n, m_n = len(codes_list), len(mets)
    full, mean, lo, hi = (np.full((g_n, m_n), np.nan) for _ in range(4))
    draws_out: list[np.ndarray | None] = [None] * g_n
    for g, codes in enumerate(codes_list):
        n = len(codes)
        if n == 0:
            continue
        k = int(codes.max()) + 1
        cnt = np.bincount(codes, minlength=k)
        full[g] = _values(cnt[cnt > 0], mets)
        if rarefy_to is None or n < rarefy_to:
            continue
        draws = np.empty((m_n, n_rarefy))
        if n == rarefy_to:
            draws[:] = full[g][:, None]
        else:
            for r in range(n_rarefy):
                c = np.bincount(codes[gen.choice(n, rarefy_to, replace=False)], minlength=k)
                draws[:, r] = _values(c[c > 0], mets)
        for m in range(m_n):
            mean[g, m] = _mean(draws[m])
            lo[g, m], hi[g, m] = _quantiles(draws[m])
        draws_out[g] = draws
    return full, mean, lo, hi, draws_out


def _codes_for(units: pd.DataFrame, by_cols: list[str], labels: list[str], clone_key: str) -> list[np.ndarray]:
    keys, pos = _groups(units, by_cols)
    at = {_label(k): p for k, p in zip(keys, pos, strict=True)}
    clone = units[clone_key].to_numpy()
    return [pd.factorize(clone[at[lab]])[0] if lab in at else np.empty(0, np.int64) for lab in labels]


def _contrast_pairs(contrasts: Any, labels: list[str]) -> list[tuple[str, str]]:
    if contrasts is None or len(labels) < 2:
        return []
    if isinstance(contrasts, str):
        if contrasts != "pairwise":
            raise ValueError(f"contrasts must be 'pairwise', None or a list of (group, group) pairs, got {contrasts!r}")
        return [(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
    pairs = []
    for a, b in contrasts:
        a, b = str(a), str(b)
        if a not in labels or b not in labels:
            raise ValueError(f"contrast ({a!r}, {b!r}) names an unknown group; groups are {labels}")
        pairs.append((a, b))
    return pairs


class CloneRelocation(NullDesign):
    """Random placement of intact clones: permute clone centroids, carry every clone's footprint along.

    Within each section and chain, the centroids of the clones (mean position of their rows) are permuted among the
    clones; every row keeps its offset from its clone's centroid, so a clone keeps its shape and its number of rows.
    The group of a moved row is the group of the nearest row of the same section at its new position (the group map is
    the observed one). Under the identity permutation every row stays in its own group, so the observed data are one
    draw of the design. :func:`diversity` and :func:`overlap` apply it to the receptor records and rebuild the units
    (spots, loci) inside the new groups for every placebo data set, as for the observed data.

    Parameters
    ----------
    units
        Table (records or units, one row each) with coordinates, the clone and the group columns.
    by
        Group column(s) re-assigned from the new positions.
    within
        Stratum columns (section, chain): clones trade centroids only inside a stratum.
    section
        Column of the section (library); groups are looked up among units of the same section.
    clone_key, x_key, y_key
        Column names.

    Notes
    -----
    Null model: conditional on the clone footprints (unit offsets from the centroid) and on the set of centroid
    positions of each section and chain, every assignment of the centroids to the clones is equally likely: where a
    clone sits carries no information about its size or shape.
    """

    def __init__(
        self,
        units: pd.DataFrame,
        *,
        by: Sequence[str],
        within: Sequence[str],
        section: str | None,
        clone_key: str,
        x_key: str,
        y_key: str,
    ) -> None:
        for c in [*by, *within, clone_key, x_key, y_key]:
            if c not in units.columns:
                raise KeyError(f"placebo='clones' needs column {c!r} in the unit table (unit coordinates required)")
        self.by = list(by)
        self.within = list(within)
        self.section = section
        self.clone_key, self.x_key, self.y_key = clone_key, x_key, y_key
        self._ready = False

    def _prepare(self, df: pd.DataFrame) -> None:
        from scipy.spatial import cKDTree

        x = df[self.x_key].to_numpy(dtype=float)
        y = df[self.y_key].to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("placebo='clones' needs finite unit coordinates")
        strata = _first_seen_codes(df, self.within)
        self._clone = _first_seen_codes(df, [*self.within, self.clone_key])
        n_cl = int(self._clone.max()) + 1 if len(df) else 0
        cnt = np.bincount(self._clone, minlength=n_cl).astype(float)
        cx = np.bincount(self._clone, weights=x, minlength=n_cl) / np.maximum(cnt, 1)
        cy = np.bincount(self._clone, weights=y, minlength=n_cl) / np.maximum(cnt, 1)
        self._cx, self._cy = cx, cy
        self._dx, self._dy = x - cx[self._clone], y - cy[self._clone]
        cl_stratum = np.zeros(n_cl, dtype=np.int64)
        cl_stratum[self._clone] = strata
        self._cl_groups = [np.flatnonzero(cl_stratum == k) for k in range(int(strata.max()) + 1 if len(df) else 0)]
        self._group = _first_seen_codes(df, self.by)
        self._group_values = df[self.by].iloc[np.unique(self._group, return_index=True)[1]].reset_index(drop=True)
        sec = _first_seen_codes(df, [self.section]) if self.section else np.zeros(len(df), np.int64)
        self._sec = sec
        self._trees = {}
        for k in np.unique(sec):
            pos = np.flatnonzero(sec == k)
            self._trees[int(k)] = (cKDTree(np.column_stack([x[pos], y[pos]])), pos)
        self._n = len(df)
        self._ready = True

    def describe(self) -> dict[str, Any]:
        """Return the parameters that define this design."""
        return {
            "design": "CloneRelocation",
            "by": self.by,
            "within": self.within,
            "section": self.section,
            "moves": "clone centroids permuted within strata; footprints carried; groups from the nearest unit",
        }

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Return a copy of the unit table with the groups of the relocated units."""
        if not self._ready or len(df) != self._n:
            self._prepare(df)
        new_c = np.arange(len(self._cx))
        for g in self._cl_groups:
            new_c[g] = g[rng.permutation(len(g))]
        nx = self._cx[new_c][self._clone] + self._dx
        ny = self._cy[new_c][self._clone] + self._dy
        grp = np.empty(self._n, dtype=np.int64)
        for k, (tree, pos) in self._trees.items():
            mine = np.flatnonzero(self._sec == k)
            _, j = tree.query(np.column_stack([nx[mine], ny[mine]]))
            grp[mine] = self._group[pos[j]]
        out = df.copy()
        vals = self._group_values.iloc[grp].reset_index(drop=True)
        for c in self.by:
            out[c] = vals[c].to_numpy()
        return out


def _placeable(
    air: pd.DataFrame, by_cols: Sequence[str], clone_key: str, spot_key: str | None, x_key: str, y_key: str
) -> pd.DataFrame:
    """Records that the clone relocation can move: clone, groups, spot (for spot / locus units) and coordinates."""
    for c in [x_key, y_key]:
        if c not in air.columns:
            raise KeyError(f"placebo='clones' moves clones in space: column {c!r} (coordinates in um) is needed")
    ok = np.array(air[clone_key].notna().to_numpy(dtype=bool), dtype=bool)
    for c in [*by_cols, *([spot_key] if spot_key else [])]:
        ok &= air[c].notna().to_numpy(dtype=bool)
    ok &= np.isfinite(air[x_key].to_numpy(dtype=float)) & np.isfinite(air[y_key].to_numpy(dtype=float))
    return air if ok.all() else air.loc[ok]


def _resolve_design(
    placebo: Any,
    units: pd.DataFrame,
    air: pd.DataFrame,
    clone_key: str,
    locus_key: str,
    library_key: str,
    by_cols: Sequence[str] = (),
    x_key: str = S.X,
    y_key: str = S.Y,
    unit: str = "locus",
    spot_key: str = S.SPOT,
) -> tuple[NullDesign, pd.DataFrame, bool]:
    """Return (design, data the design acts on, whether that data is the unit table)."""
    if isinstance(placebo, str):
        if placebo == "labels":
            within = [c for c in (library_key, locus_key) if c in units.columns and c not in by_cols]
            return Permute(clone_key, within=within or None), units, True
        if placebo == "clones":
            if not by_cols:
                raise ValueError("placebo='clones' relocates clones between groups: it needs by=")
            rec = _placeable(air, by_cols, clone_key, spot_key if unit != "record" else None, x_key, y_key)
            strata = [c for c in (library_key, locus_key) if c in rec.columns]
            design = CloneRelocation(
                rec,
                by=list(by_cols),
                within=strata,
                section=library_key if library_key in rec.columns else None,
                clone_key=clone_key,
                x_key=x_key,
                y_key=y_key,
            )
            return design, rec, False
        raise ValueError(f"placebo must be None, 'clones', 'labels' or a NullDesign, got {placebo!r}")
    if isinstance(placebo, NullDesign):
        return placebo, air, False
    raise TypeError(
        f"placebo must be None, 'clones', 'labels' or a spaclone.nulls.NullDesign, got {type(placebo).__name__}"
    )


def _diversity_statistic(
    df: pd.DataFrame,
    *,
    from_units: bool,
    unit: str,
    by_cols: list[str],
    labels: list[str],
    pairs: list[tuple[str, str]],
    mets: list[_Metric],
    rarefy_to: int | None,
    n_rarefy: int,
    seed: int,
    radius_um: float,
    keys: dict[str, str],
    rarefy_min: bool = False,
) -> dict[str, float]:
    u = df if from_units else clone_units(df, unit=unit, by=by_cols, radius_um=radius_um, **keys).units
    codes = _codes_for(u, by_cols, labels, keys["clone_key"])
    if rarefy_min:  # rarefy to the smallest group of this data set (group sizes change when clones move)
        sizes = [len(c) for c in codes if len(c)]
        rarefy_to = int(min(sizes)) if sizes else None
    full, mean, *_ = _group_values(codes, mets, rarefy_to, n_rarefy, np.random.default_rng(seed))
    stat = mean if rarefy_to is not None else full
    out: dict[str, float] = {}
    for g, lab in enumerate(labels):
        for m, met in enumerate(mets):
            out[f"{lab}|{met.name}"] = stat[g, m]
    for a, b in pairs:
        ia, ib = labels.index(a), labels.index(b)
        for m, met in enumerate(mets):
            out[f"{a} - {b}|{met.name}"] = stat[ia, m] - stat[ib, m]
    return out


def diversity(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = "region",
    metrics: Sequence[str | Callable[[np.ndarray], float]] = ("richness", "shannon", "simpson", "gini", "chao1"),
    unit: Unit = "locus",
    rarefy: bool = True,
    rarefy_to: int | None = None,
    n_rarefy: int = 200,
    placebo: NullDesign | Literal["clones", "labels"] | None = None,
    n_perm: int = 200,
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
    contrasts: Literal["pairwise"] | Sequence[tuple[str, str]] | None = "pairwise",
    n_jobs: int = 1,
    rng: RngLike = None,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> TestResult:
    """Clonal diversity per group, rarefied to a common number of units, with a clone-level test of differences.

    Parameters
    ----------
    air
        Receptor table with a clone column and the group column(s).
    by
        Group column(s) (region, niche, section); ``None`` treats the table as one group.
    metrics
        Names among :data:`DIVERSITY_METRICS`, ``"top<k>"`` or callables of the clone-size vector (positive integers in
        no particular order) returning a float. ``richness`` = clones; ``shannon`` = -sum p ln p; ``simpson`` =
        1 - sum p^2 (Gini-Simpson); ``inverse_simpson`` = 1 / sum p^2; ``evenness`` = shannon / ln(richness)
        (Pielou); ``gini`` = Gini coefficient of the clone sizes (observed clones only); ``chao1`` = bias-corrected
        Chao1, S + F1 (F1 - 1) / (2 (F2 + 1)); ``expanded`` = share of units in clones with >= 2 units;
        ``"top<k>"`` = share of units in the k largest clones.
    unit
        ``"locus"``, ``"spot"`` or ``"record"`` (see :func:`clone_units`).
    rarefy
        Rarefy every group to ``rarefy_to`` units (without replacement, ``n_rarefy`` draws) and use the mean as the
        statistic. Groups with exactly ``rarefy_to`` units are used whole.
    rarefy_to
        Common number of units; ``None`` = the smallest group's (in every placebo data set, the smallest group of that
        data set). Groups with fewer units get NaN (with a warning).
    n_rarefy
        Rarefaction draws per group.
    placebo
        ``None`` (no test). ``"clones"`` (recommended): random placement of intact clones
        (:class:`CloneRelocation`): within a section and chain the clone centroids are permuted among the clones,
        every clone carries its footprint (its records' offsets from its centroid) to its new centroid, a moved record
        takes the group of the nearest observed record there, and the units are rebuilt inside the new groups. Clones
        stay whole, so the test asks whether a group holds larger (or smaller) clones than random placement of the
        tissue's clones gives, borders cutting clones included (see Notes). Needs coordinates; records without them
        are left out of the placebo. ``"labels"``: clone labels permuted among the units of the same section and chain (1.0
        behaviour, kept for the archive): its null mean shows the difference that unit counts alone produce, but its
        p-values test whether units of one clone are scattered at random, which spatially clustered clones never are
        (a warning says so). Or any :class:`~spaclone.nulls.NullDesign`, applied to the receptor records, after which
        the units, the groups and the statistic are rebuilt from scratch: for groups made of several structures (e.g.
        germinal centres) ``Permute(by, blocks="gc_id")`` exchanges whole structures between groups, and
        ``Relocate(slots, slot_columns=[..., "region"])`` moves spots with their records into new regions.
    n_perm
        Placebo data sets.
    alternative
        Direction of the Monte-Carlo p-values.
    contrasts
        ``"pairwise"`` (every pair of groups, first minus second in group order), ``None`` or a list of
        ``(group, group)`` label pairs.
    n_jobs
        Worker processes for the placebo (results do not depend on it).
    rng
        Seed or generator. Without a placebo the rarefaction draws come from it directly, group by group; with a
        placebo one seed is drawn from it for the rarefaction (the same subsampling then applies to the observed data
        and to every placebo data set, so the rarefied statistic is a fixed function of the data) and the placebo
        data sets get spawned generators.
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per (group, metric) (``kind="group"``) and per (contrast, metric) (``kind="contrast"``) with
        ``group`` (label; ``"a - b"`` for contrasts), the ``by`` columns, ``group_a`` / ``group_b``, ``metric``,
        ``n_units``, ``n_clones``, ``value`` (whole group), ``rarefied`` (mean of the draws), ``rarefied_lo`` /
        ``rarefied_hi`` (2.5% and 97.5% of the draws; for contrasts, of the differences of independent draws),
        ``rarefied_to``, ``statistic`` (``rarefied`` if ``rarefy`` else ``value``), and with a placebo ``expected`` /
        ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``, ``z``, ``p_value``, ``p_maxT`` (family-wise over all
        rows), ``q_value`` (Benjamini-Hochberg within each kind) and ``n_perm``. ``null``: placebo draws (one column
        per row of the table, in table order).

    Notes
    -----
    Null model (``placebo="clones"``): random placement of intact clones. Conditional on the clone footprints (the
    offsets of every clone's records from its centroid) and on the set of centroid positions of each section and
    chain, every assignment of the centroids to the clones is equally likely: where a clone sits carries no
    information about its size or shape. The group map is fixed (a moved record takes the group of the nearest
    observed record of the section), and units are rebuilt within the groups for every placebo data set, so a clone
    cut by a group border is split into units in the same way as in the observed data. Under this null big and small clones are equally likely to sit in any group, so a contrast tests "group
    A holds more expanded clones than group B" with every clone kept together; the null distribution includes the
    difference that unequal group areas, unit counts and clones cut by group borders produce, and its spread reflects
    how much a few large clones can move a group's diversity. In simulations of spatially clustered clones with equal
    clone-size distributions everywhere (two halves, a quarter against the rest, a germinal-centre-sized disc
    against the rest, clones cut or squashed at the section edge; 200 data sets each, loci and spot units) the
    contrasts rejected 2-8% of the time at α = 0.05, where the label placebo rejected 16-100%. When a region holds
    the largest clones, richness and the Gini coefficient detect it (power 28-99% with spot units); Simpson's index
    has almost no power when a few giant clones dominate, because it is set by where those few clones sit, and dense
    clones merge into few loci, so expansion is better seen with ``unit="spot"`` than with ``unit="locus"``. Clones
    scattered over the whole section have their centroid near its middle and barely move, so the test is
    conservative for them.

    Null model (``placebo="labels"``): conditional on the units, their groups, sections and chains, the clone labels
    are exchangeable among the units of each section and chain; which clone a unit belongs to carries no information
    about its group. This holds only when the units of a clone are scattered at random over the tissue, so with
    clustered clones every group looks less diverse than its placebo and contrasts reject too often; its null mean is
    still the difference built into the unit counts (for example a lower Gini where units are fewer). With a
    :class:`~spaclone.nulls.NullDesign` the null is whatever the design states.

    The Gini coefficient of the observed clone sizes rises with the number of units (singletons dominate small
    samples), and so do richness, Chao1 and the expanded share; never compare them across groups of different size
    without rarefaction or a placebo.

    References
    ----------
    Shannon 1948, doi:10.1002/j.1538-7305.1948.tb01338.x; Simpson 1949, doi:10.1038/163688a0; Hill 1973,
    doi:10.2307/1934352; Chao 1984 (Scand J Stat 11:265); Heck et al. 1975, doi:10.2307/1934716 (rarefaction);
    Chao et al. 2014, doi:10.1890/13-0133.1 (rarefaction and extrapolation with Hill numbers); Deltas 2003,
    doi:10.1162/rest.2003.85.1.226 (small-sample bias of the Gini coefficient); Phipson & Smyth 2010,
    doi:10.2202/1544-6115.1585 (Monte-Carlo p-values); Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    mets = _resolve_metrics(metrics)
    if n_rarefy < 1:
        raise ValueError("n_rarefy must be >= 1")
    by_cols = _as_list(by)
    gen = as_rng(rng)
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    u = clone_units(air, unit=unit, by=by_cols, radius_um=radius_um, **keys).units
    gkeys, pos = _groups(u, by_cols)
    if not gkeys:
        raise ValueError("no units with a clone id (and a group value) in the table")
    labels = [_label(k) for k in gkeys]
    codes_list = _codes_for(u, by_cols, labels, clone_key)
    n_units = [len(c) for c in codes_list]
    n_clones = [int(c.max()) + 1 if len(c) else 0 for c in codes_list]
    rt: int | None = None
    if rarefy:
        rt = int(rarefy_to) if rarefy_to is not None else int(min(n_units))
        if rt < 1:
            raise ValueError("rarefy_to must be >= 1")
        small = [lab for lab, n in zip(labels, n_units, strict=True) if n < rt]
        if small:
            warnings.warn(
                f"group(s) {small} have fewer than rarefy_to={rt} units; their rarefied values are NaN",
                LowPowerWarning,
                stacklevel=2,
            )
    pairs = _contrast_pairs(contrasts, labels)
    res = None
    seed = None
    if placebo is None:
        full, mean, lo, hi, draws = _group_values(codes_list, mets, rt, n_rarefy, gen)
    else:
        seed = int(gen.integers(0, 2**63 - 1))
        design, data, from_units = _resolve_design(
            placebo, u, air, clone_key, locus_key, library_key, by_cols, x_key, y_key, unit, spot_key
        )
        if isinstance(placebo, str) and placebo == "labels":
            warnings.warn(
                "placebo='labels' p-values test whether the units of a clone are scattered at random between the "
                "groups; spatially clustered clones reject that whatever the diversities. Use placebo='clones' to "
                "test a difference in diversity",
                SpacloneWarning,
                stacklevel=2,
            )
        stat = functools.partial(
            _diversity_statistic,
            from_units=from_units,
            unit=unit,
            by_cols=by_cols,
            labels=labels,
            pairs=pairs,
            mets=mets,
            rarefy_to=rt,
            n_rarefy=n_rarefy,
            seed=seed,
            radius_um=radius_um,
            keys=keys,
            rarefy_min=bool(rarefy and rarefy_to is None),
        )
        res = run_placebo(
            stat, data, design, n_perm=n_perm, alternative=alternative, rng=gen, n_jobs=n_jobs, name="diversity"
        )
        full, mean, lo, hi, draws = _group_values(codes_list, mets, rt, n_rarefy, np.random.default_rng(seed))
    rows = []
    for g, (lab, key) in enumerate(zip(labels, gkeys, strict=True)):
        for m, met in enumerate(mets):
            row: dict[str, Any] = {"kind": "group", "group": lab}
            row.update({c: v for c, v in zip(by_cols, key, strict=True) if c not in _RESERVED})
            row.update(
                group_a=None,
                group_b=None,
                metric=met.name,
                n_units=n_units[g],
                n_clones=n_clones[g],
                value=full[g, m],
                rarefied=mean[g, m],
                rarefied_lo=lo[g, m],
                rarefied_hi=hi[g, m],
                rarefied_to=rt,
                component=f"{lab}|{met.name}",
            )
            rows.append(row)
    for a, b in pairs:
        ia, ib = labels.index(a), labels.index(b)
        for m, met in enumerate(mets):
            dlo = dhi = np.nan
            if draws[ia] is not None and draws[ib] is not None:
                dlo, dhi = _quantiles(draws[ia][m] - draws[ib][m])
            rows.append(
                {
                    "kind": "contrast",
                    "group": f"{a} - {b}",
                    "group_a": a,
                    "group_b": b,
                    "metric": met.name,
                    "n_units": np.nan,
                    "n_clones": np.nan,
                    "value": full[ia, m] - full[ib, m],
                    "rarefied": mean[ia, m] - mean[ib, m],
                    "rarefied_lo": dlo,
                    "rarefied_hi": dhi,
                    "rarefied_to": rt,
                    "component": f"{a} - {b}|{met.name}",
                }
            )
    table = pd.DataFrame(rows)
    table["statistic"] = table["rarefied"] if rarefy else table["value"]
    null = None
    if res is not None:
        pt = res.table.set_index("component")
        comp = table["component"]
        table["expected"] = comp.map(pt["null_mean"]).to_numpy(dtype=float)
        table["null_mean"] = table["expected"]
        table["null_sd"] = comp.map(pt["null_sd"]).to_numpy(dtype=float)
        table["null_lo"] = comp.map(pt["null_lo"]).to_numpy(dtype=float)
        table["null_hi"] = comp.map(pt["null_hi"]).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            table["z"] = (table["statistic"] - table["null_mean"]) / table["null_sd"]
        table["p_value"] = comp.map(pt["p_value"]).to_numpy(dtype=float)
        table["p_maxT"] = comp.map(pt["p_maxT"]).to_numpy(dtype=float)
        q = np.full(len(table), np.nan)
        for kind in ("group", "contrast"):
            sel = (table["kind"] == kind).to_numpy()
            q[sel] = bh(table.loc[sel, "p_value"].to_numpy(dtype=float))
        table["q_value"] = q
        table["n_perm"] = n_perm
        order = [list(res.table["component"]).index(c) for c in comp]
        null = res.null[:, order] if res.null is not None else None
    else:
        for c in ("expected", "null_mean", "null_sd", "null_lo", "null_hi", "z", "p_value", "p_maxT", "q_value"):
            table[c] = np.nan
        table["n_perm"] = 0
    table = table.drop(columns="component")
    summary: dict[str, Any] = {"groups": len(labels), "rarefied_to": rt, "n_perm": n_perm if res is not None else 0}
    con = table[table["kind"] == "contrast"]
    if len(con):
        summary.update(
            contrast=f"{con['group'].iloc[0]} ({con['metric'].iloc[0]})",
            difference=float(con["statistic"].iloc[0]),
            p_value=float(con["p_value"].iloc[0]),
        )
    params = {
        "by": by_cols,
        "metrics": [m.name for m in mets],
        "unit": unit,
        "radius_um": radius_um,
        "rarefy": rarefy,
        "rarefy_to": rt,
        "n_rarefy": n_rarefy,
        "contrasts": pairs,
        "placebo": None if res is None else res.params["design"],
        "n_perm": n_perm if res is not None else 0,
        "alternative": alternative,
        "rng_entropy": rng_entropy(gen),
        "rarefaction_seed": seed,
    }
    return TestResult(name="diversity", table=table, summary=summary, params=params, null=null)


# ----------------------------------------------------------------------------------------------------- overlap


def _overlap_matrices(
    clone_codes: np.ndarray, pos: list[np.ndarray], mets: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise similarity matrices (G x G) and per-group units, clones and shared-clone counts."""
    from scipy import sparse

    g_n = len(pos)
    k = int(clone_codes.max()) + 1 if len(clone_codes) else 0
    rows = np.concatenate([np.full(len(p), g) for g, p in enumerate(pos)]) if g_n else np.empty(0, np.int64)
    cols = np.concatenate([clone_codes[p] for p in pos]) if g_n else np.empty(0, np.int64)
    c = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(g_n, k)).tocsr()
    c.sum_duplicates()
    tot = np.asarray(c.sum(axis=1)).ravel()
    b = (c > 0).astype(float)
    rich = np.asarray(b.sum(axis=1)).ravel()
    shared = (b @ b.T).toarray()
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if "morisita_horn" in mets:
            dot = (c @ c.T).toarray()
            sq = np.asarray(c.multiply(c).sum(axis=1)).ravel() / np.maximum(tot, 1) ** 2
            out["morisita_horn"] = 2 * dot / ((sq[:, None] + sq[None, :]) * np.outer(tot, tot))
        if "jaccard" in mets:
            out["jaccard"] = shared / (rich[:, None] + rich[None, :] - shared)
        if "sorensen" in mets:
            out["sorensen"] = 2 * shared / (rich[:, None] + rich[None, :])
        if "bhattacharyya" in mets:
            p = sparse.diags(1.0 / np.maximum(tot, 1)) @ c
            r = p.sqrt()
            out["bhattacharyya"] = (r @ r.T).toarray()
    return out, tot, rich, shared, c


def _resolve_overlap(metric: str | Sequence[str]) -> list[str]:
    mets = [metric] if isinstance(metric, str) else list(metric)
    bad = [m for m in mets if m not in OVERLAP_METRICS]
    if bad or not mets:
        raise ValueError(f"unknown overlap metric(s) {bad}; use {OVERLAP_METRICS}")
    return list(dict.fromkeys(mets))


def _overlap_statistic(
    df: pd.DataFrame,
    *,
    from_units: bool,
    unit: str,
    by_cols: list[str],
    labels: list[str],
    mets: list[str],
    radius_um: float,
    keys: dict[str, str],
) -> dict[str, float]:
    u = df if from_units else clone_units(df, unit=unit, by=by_cols, radius_um=radius_um, **keys).units
    gkeys, pos = _groups(u, by_cols)
    at = {_label(k): p for k, p in zip(gkeys, pos, strict=True)}
    codes = pd.factorize(u[keys["clone_key"]].to_numpy())[0]
    ps = [at.get(lab, np.empty(0, np.int64)) for lab in labels]
    mats = _overlap_matrices(codes, ps, mets)[0]
    out = {}
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            for m in mets:
                out[f"{a} vs {labels[j]}|{m}"] = float(mats[m][i, j])
    return out


def overlap(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str],
    metric: str | Sequence[str] = "morisita_horn",
    unit: Unit = "locus",
    placebo: NullDesign | Literal["clones", "labels"] | None = None,
    n_perm: int = 200,
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
    n_jobs: int = 1,
    rng: RngLike = None,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> TestResult:
    """Repertoire overlap between every pair of groups, optionally against a placebo.

    Parameters
    ----------
    air
        Receptor table with a clone column and the group column(s).
    by
        Group column(s) (region, niche, section); at least two groups are needed.
    metric
        One or several of :data:`OVERLAP_METRICS`, all similarities in [0, 1]: ``morisita_horn`` =
        2 sum x y / ((sum x^2 / X^2 + sum y^2 / Y^2) X Y) on unit counts (Horn's abundance overlap, little affected by
        sample size); ``jaccard`` = shared clones / clones in either group; ``sorensen`` = 2 shared / (clones of A +
        clones of B); ``bhattacharyya`` = sum sqrt(p q) on clone frequencies.
    unit
        ``"locus"``, ``"spot"`` or ``"record"`` (see :func:`clone_units`).
    placebo
        ``None``; ``"clones"`` (random placement of intact clones, :class:`CloneRelocation`: overlap expected when
        clones keep their shape but sit anywhere); ``"labels"`` (clone labels permuted among the units of each section
        and chain, across groups: valid only when the units of a clone are scattered at random); or a
        :class:`~spaclone.nulls.NullDesign` applied to the records.
    n_perm
        Placebo data sets.
    alternative
        Direction of the Monte-Carlo p-values (``"less"``: less overlap than random placement, i.e. segregated
        repertoires).
    n_jobs
        Worker processes for the placebo.
    rng
        Seed or generator for the placebo.
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per (pair, metric) with ``group_a``, ``group_b``, ``metric``, ``value``, ``n_units_a``,
        ``n_units_b``, ``n_clones_a``, ``n_clones_b``, ``shared_clones``, ``statistic`` (= value) and, with a placebo,
        ``expected`` / ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``, ``z``, ``p_value``, ``p_maxT``,
        ``q_value`` (Benjamini-Hochberg over all rows) and ``n_perm``. ``table.pivot(index="group_a",
        columns="group_b", values="value")`` gives the upper triangle of the matrix.

    Notes
    -----
    Null model (``placebo="clones"``): conditional on the clone footprints and on each section's and chain's set of
    clone centroids, every assignment of the centroids to the clones is equally likely (:class:`CloneRelocation`); a
    clone is shared by two groups when its relocated footprint straddles them.

    Null model (``placebo="labels"``): conditional on the units, their groups, sections and chains, clone labels are
    exchangeable among the units of each section and chain, so every group receives a random sample of the section's
    clones of its own size. Jaccard and Sorensen overlaps grow with group size under this null; the placebo gives the
    overlap expected at the observed sizes. Spatially clustered clones make groups share fewer clones than this null
    whatever their repertoires, so its p-values for "less" overlap are not evidence of segregated repertoires.

    References
    ----------
    Horn 1966, doi:10.1086/282436 (Morisita-Horn index); Jaccard 1912, doi:10.1111/j.1469-8137.1912.tb05611.x;
    Sorensen 1948 (Biol Skr 5:1); Bhattacharyya 1943 (Bull Calcutta Math Soc 35:99); Phipson & Smyth 2010,
    doi:10.2202/1544-6115.1585; Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    mets = _resolve_overlap(metric)
    by_cols = _as_list(by)
    if not by_cols:
        raise ValueError("overlap needs a group column (by=)")
    gen = as_rng(rng)
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    u = clone_units(air, unit=unit, by=by_cols, radius_um=radius_um, **keys).units
    gkeys, pos = _groups(u, by_cols)
    if len(gkeys) < 2:
        raise ValueError(f"overlap needs at least two groups, got {len(gkeys)}")
    labels = [_label(k) for k in gkeys]
    codes = pd.factorize(u[clone_key].to_numpy())[0]
    mats, tot, rich, shared, _ = _overlap_matrices(codes, pos, mets)
    rows = []
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            for m in mets:
                rows.append(
                    {
                        "group_a": a,
                        "group_b": labels[j],
                        "metric": m,
                        "value": float(mats[m][i, j]),
                        "n_units_a": int(tot[i]),
                        "n_units_b": int(tot[j]),
                        "n_clones_a": int(rich[i]),
                        "n_clones_b": int(rich[j]),
                        "shared_clones": int(shared[i, j]),
                    }
                )
    table = pd.DataFrame(rows)
    table["statistic"] = table["value"]
    null = None
    res = None
    if placebo is not None:
        design, data, from_units = _resolve_design(
            placebo, u, air, clone_key, locus_key, library_key, by_cols, x_key, y_key, unit, spot_key
        )
        stat = functools.partial(
            _overlap_statistic,
            from_units=from_units,
            unit=unit,
            by_cols=by_cols,
            labels=labels,
            mets=mets,
            radius_um=radius_um,
            keys=keys,
        )
        res = run_placebo(
            stat, data, design, n_perm=n_perm, alternative=alternative, rng=gen, n_jobs=n_jobs, name="overlap"
        )
        pt = res.table.set_index("component")
        comp = table["group_a"] + " vs " + table["group_b"] + "|" + table["metric"]
        table["expected"] = comp.map(pt["null_mean"]).to_numpy(dtype=float)
        table["null_mean"] = table["expected"]
        for c in ("null_sd", "null_lo", "null_hi", "p_value", "p_maxT"):
            table[c] = comp.map(pt[c]).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            table["z"] = (table["statistic"] - table["null_mean"]) / table["null_sd"]
        table["q_value"] = bh(table["p_value"].to_numpy(dtype=float))
        table["n_perm"] = n_perm
        order = [list(res.table["component"]).index(c) for c in comp]
        null = res.null[:, order] if res.null is not None else None
    else:
        for c in ("expected", "null_mean", "null_sd", "null_lo", "null_hi", "z", "p_value", "p_maxT", "q_value"):
            table[c] = np.nan
        table["n_perm"] = 0
    base = ["group_a", "group_b", "metric", "value", "n_units_a", "n_units_b", "n_clones_a", "n_clones_b"]
    tail = ["expected", "null_mean", "null_sd", "null_lo", "null_hi", "z", "p_value", "p_maxT", "q_value", "n_perm"]
    table = table[[*base, "shared_clones", "statistic", *tail]]
    summary: dict[str, Any] = {"groups": len(labels), "pairs": len(table) // len(mets), "metric": mets[0]}
    first = table[table["metric"] == mets[0]]
    summary["mean_value"] = float(first["value"].mean())
    params = {
        "by": by_cols,
        "metrics": mets,
        "unit": unit,
        "radius_um": radius_um,
        "placebo": None if res is None else res.params["design"],
        "n_perm": n_perm if res is not None else 0,
        "alternative": alternative,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="overlap", table=table, summary=summary, params=params, null=null)


# ---------------------------------------------------------------------------------------- gene usage, spectratype


def _mode(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Most common ``value`` per ``keys`` group (ties: the smallest value); rows with a missing value ignored."""
    d = df.dropna(subset=[value])
    cnt = d.groupby([*keys, value], sort=False, observed=True, dropna=False).size().rename("_n").reset_index()
    cnt = cnt.sort_values([*keys, "_n", value], ascending=[True] * len(keys) + [False, True])
    return cnt.drop_duplicates(keys)[[*keys, value]]


def _weighted_counts(
    air: pd.DataFrame,
    values: pd.Series,
    *,
    by_cols: list[str],
    unit: str,
    weight: str,
    radius_um: float,
    keys: dict[str, str],
) -> pd.DataFrame:
    """Counts of a record-level value per group, weighted by records, units or clones (mode per unit / clone)."""
    if weight not in ("clones", "units", "records"):
        raise ValueError(f"weight must be 'clones', 'units' or 'records', got {weight!r}")
    cu = clone_units(air, unit=unit, by=by_cols, radius_um=radius_um, **keys)
    u = cu.units
    member = cu.membership.to_numpy()
    rec = pd.DataFrame({"unit_id": member, "_v": values.loc[cu.membership.index].to_numpy(dtype=object)})
    rec["_v"] = rec["_v"].where(pd.notna(rec["_v"]), None)
    for c in [*by_cols, keys["clone_key"]]:
        rec[c] = u[c].to_numpy()[member]
    if weight == "records":
        items = rec.dropna(subset=["_v"])
    elif weight == "units":
        items = _mode(rec, ["unit_id"], "_v").merge(u[["unit_id", *by_cols]], on="unit_id")
    else:
        items = _mode(rec, [*by_cols, keys["clone_key"]], "_v")
    if by_cols:
        cnt = items.groupby([*by_cols, "_v"], observed=True, dropna=False).size().rename("count").reset_index()
        cnt["share"] = cnt["count"] / cnt.groupby(by_cols, observed=True, dropna=False)["count"].transform("sum")
    else:
        cnt = items.groupby("_v").size().rename("count").reset_index()
        cnt["share"] = cnt["count"] / cnt["count"].sum()
    return cnt


def gene_usage(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    gene: str = S.V_CALL,
    unit: Unit = "locus",
    weight: Literal["clones", "units", "records"] = "clones",
    allele: bool = False,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> pd.DataFrame:
    """V, J (or C) gene usage per group.

    Parameters
    ----------
    air
        Receptor table with the gene call column and a clone column.
    by
        Group column(s); ``None`` = the whole table.
    gene
        Gene call column (``"v_call"``, ``"j_call"``, ``"c_call"``, ...). Calls are reduced to the gene of the first
        call (allele dropped, :func:`spaclone.pp.gene_name`) unless ``allele=True``.
    unit
        Unit of ``weight="units"`` (see :func:`clone_units`).
    weight
        ``"clones"``: each clone counts once per group, with its most common gene (robust to expansion; the usual
        clonotype-level usage); ``"units"``: each unit counts once with its most common gene; ``"records"``: every
        record counts.
    allele
        Keep the full call instead of the gene.
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        The ``by`` columns, ``gene``, ``count`` and ``share`` (of the group's counted items with a call), sorted by
        group and decreasing count.
    """
    if gene not in air.columns:
        raise KeyError(f"gene column {gene!r} not in the receptor table")
    by_cols = _as_list(by)
    vals = air[gene].astype("string") if allele else gene_name(air[gene])
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    cnt = _weighted_counts(air, vals, by_cols=by_cols, unit=unit, weight=weight, radius_um=radius_um, keys=keys)
    cnt = cnt.rename(columns={"_v": "gene"})
    cnt = cnt.sort_values([*by_cols, "count", "gene"], ascending=[True] * len(by_cols) + [False, True])
    return cnt.reset_index(drop=True)[[*by_cols, "gene", "count", "share"]]


def cdr3_length(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    unit: Unit = "locus",
    weight: Literal["clones", "units", "records"] = "units",
    junction_key: str = S.JUNCTION_AA,
    anchors: bool = True,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> pd.DataFrame:
    """CDR3 length distribution (spectratype) per group.

    Parameters
    ----------
    air
        Receptor table with a junction column and a clone column.
    by
        Group column(s); ``None`` = the whole table.
    unit
        Unit of ``weight="units"`` (see :func:`clone_units`).
    weight
        ``"units"`` (each unit counts once with its most common length), ``"clones"`` (each clone once per group) or
        ``"records"``.
    junction_key
        Junction column (amino acids by default; a nucleotide junction gives lengths in nucleotides).
    anchors
        Count the two conserved anchor residues (junction length, as scirpy's ``spectratype``); ``False`` gives the
        IMGT CDR3 length (junction - 2).
    radius_um
        Linkage distance of locus units in µm.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        The ``by`` columns, ``length``, ``count`` and ``share``; every length between the overall minimum and maximum
        is listed for every group (count 0 where absent).

    References
    ----------
    Pannetier et al. 1993, doi:10.1073/pnas.90.9.4319 (CDR3 length spectratyping).
    """
    if junction_key not in air.columns:
        raise KeyError(f"junction column {junction_key!r} not in the receptor table")
    by_cols = _as_list(by)
    jx = air[junction_key].astype("string")
    lengths = jx.str.len().astype("Float64") - (0 if anchors else 2)
    lengths = lengths.where(jx.notna() & (jx != ""))
    vals = pd.Series(lengths.to_numpy(dtype=float, na_value=np.nan), index=air.index)
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    cnt = _weighted_counts(air, vals, by_cols=by_cols, unit=unit, weight=weight, radius_um=radius_um, keys=keys)
    cnt = cnt.rename(columns={"_v": "length"})
    cnt["length"] = cnt["length"].astype(float).astype(np.int64)
    if len(cnt) == 0:
        return pd.DataFrame(columns=[*by_cols, "length", "count", "share"])
    full = np.arange(cnt["length"].min(), cnt["length"].max() + 1)
    if by_cols:
        groups = cnt[by_cols].drop_duplicates()
        grid = groups.merge(pd.DataFrame({"length": full}), how="cross")
        out = grid.merge(cnt, on=[*by_cols, "length"], how="left")
    else:
        out = pd.DataFrame({"length": full}).merge(cnt, on="length", how="left")
    out["count"] = out["count"].fillna(0).astype(np.int64)
    out["share"] = out["share"].fillna(0.0)
    return out.sort_values([*by_cols, "length"]).reset_index(drop=True)[[*by_cols, "length", "count", "share"]]
