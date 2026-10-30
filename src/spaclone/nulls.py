"""Null designs and the placebo engine.

A *null design* turns a receptor table into one random version of itself that keeps everything the null hypothesis
says should be kept. Two designs cover the tests in this package:

* :class:`Permute` exchanges the values of some columns between rows (positions fixed), optionally only inside
  strata and optionally moving whole blocks (all rows of one spot or cell) together. This is random labelling
  (spatstat ``rlabel(group=)``).
* :class:`Relocate` moves whole units (all rows sharing a unit id, with every carried attribute such as read counts,
  isotype or clone) to random slots drawn from a slot table, inside matching strata and/or local windows, optionally
  weighted. This is random placement.

:func:`placebo` re-runs *any* statistic on many random versions of the data and compares the observed value with the
resulting placebo distribution. The statistic must contain every downstream step (deduplication, gating, voting,
thresholds), so the placebo is never partial. A metric that is reproduced by random placement cannot be evidence
that receptors sit where their cells were.

References
----------
Besag & Diggle 1977, doi:10.2307/2346974 (Monte-Carlo tests); Gel et al. 2016, doi:10.1093/bioinformatics/btv562
(randomiser / evaluator split in regioneR); Markello & Misic 2021, doi:10.1016/j.neuroimage.2021.118052 (the choice of
null drives false positives; compare designs).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import _schema as S
from ._utils import (
    Alternative,
    RngLike,
    TestResult,
    as_rng,
    check_alternative,
    max_t_pvalue,
    mc_pvalue,
    null_summary,
    rng_entropy,
)

__all__ = [
    "CROWDING_BINS",
    "IG_RNA_BINS",
    "NullDesign",
    "Permute",
    "PlaceboResult",
    "Relocate",
    "bin_labels",
    "compare_designs",
    "placebo",
]

#: Bin edges of local Ig RNA (molecules of Ig constant-region genes in a unit) used for depth strata.
IG_RNA_BINS: tuple[float, ...] = (0, 1, 3, 10, 30, 100, 300, np.inf)
#: Bin edges of crowding (receptor molecules of one chain on one spot) used for placement strata.
CROWDING_BINS: tuple[float, ...] = (1, 2, 3, 4, 6, 11, np.inf)


def bin_labels(values: Any, edges: Sequence[float] = IG_RNA_BINS) -> np.ndarray:
    """Label values by half-open bins ``[edges[i], edges[i + 1])``; returns strings such as ``"3-9"`` or ``">=300"``.

    Integer-valued edges produce integer labels (``[3, 10)`` is written ``"3-9"``).
    """
    v = np.asarray(values, dtype=float)
    e = np.asarray(edges, dtype=float)
    idx = np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(e) - 2)
    labs = []
    for lo, hi in zip(e[:-1], e[1:], strict=True):
        if np.isinf(hi):
            labs.append(f">={lo:g}")
        elif float(lo).is_integer() and float(hi).is_integer() and hi - lo == 1:
            labs.append(f"{lo:g}")
        elif float(lo).is_integer() and float(hi).is_integer():
            labs.append(f"{lo:g}-{hi - 1:g}")
        else:
            labs.append(f"{lo:g}-{hi:g}")
    out = np.array(labs, dtype=object)[idx]
    out[~np.isfinite(v)] = "NA"
    return out


def _as_list(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _group_positions(df: pd.DataFrame, keys: list[str]) -> list[np.ndarray]:
    """Row positions of each group defined by ``keys`` (all rows when ``keys`` is empty)."""
    if not keys:
        return [np.arange(len(df))]
    return list(df.groupby(keys, sort=False, observed=True, dropna=False).indices.values())


def _take(series: pd.Series, positions: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(series.array.take(positions), index=index, name=series.name)


class NullDesign:
    """Base class: a callable ``design(df, rng) -> randomised copy of df`` with a ``describe()`` method."""

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:  # pragma: no cover - abstract
        """Return one randomised copy of ``df``."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:  # pragma: no cover - abstract
        """Return the parameters that define the design."""
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.describe().items() if k != "design")
        return f"{type(self).__name__}({args})"


class Permute(NullDesign):
    """Random labelling: exchange the values of ``columns`` between rows, positions fixed.

    Parameters
    ----------
    columns
        Columns permuted jointly (a row's values stay together), e.g. ``"clone_id"`` or ``["clone_id", "c_call"]``.
    within
        Column(s) defining strata; values are exchanged only between rows of the same stratum (e.g. an Ig RNA depth
        bin, the library, the isotype class).
    blocks
        Column whose rows move together, e.g. ``"spot_id"``: block-level values of ``columns`` (which must be
        constant within a block) are permuted among blocks of the same stratum and written to every row of the
        receiving block.

    Notes
    -----
    Null model: conditional on the observed positions and on the multiset of labels in each stratum, every
    assignment of labels to rows (or blocks) is equally likely.
    """

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        within: str | Sequence[str] | None = None,
        blocks: str | None = None,
    ) -> None:
        self.columns = _as_list(columns)
        if not self.columns:
            raise ValueError("Permute needs at least one column")
        self.within = _as_list(within)
        self.blocks = blocks

    def describe(self) -> dict[str, Any]:
        """Return the parameters that define this design."""
        return {"design": "Permute", "columns": self.columns, "within": self.within, "blocks": self.blocks}

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Return a randomised copy of ``df``."""
        missing = [c for c in self.columns + self.within + _as_list(self.blocks) if c not in df.columns]
        if missing:
            raise KeyError(f"Permute: column(s) {missing} not in the table")
        out = df.copy()
        n = len(df)
        if self.blocks is None:
            src = np.arange(n)
            for pos in _group_positions(df, self.within):
                src[pos] = pos[rng.permutation(len(pos))]
        else:
            codes, _ = pd.factorize(df[self.blocks], use_na_sentinel=False)
            first = np.full(codes.max() + 1 if n else 0, -1, dtype=np.int64)
            first[codes[::-1]] = np.arange(n)[::-1]
            heads = df.iloc[first]
            for c in self.columns + self.within:
                nun = df.groupby(codes, sort=False)[c].nunique(dropna=False)
                if (nun > 1).any():
                    raise ValueError(
                        f"Permute(blocks={self.blocks!r}): column {c!r} varies inside a block; block-level "
                        "permutation needs block-constant columns"
                    )
            bsrc = np.arange(len(first))
            for pos in _group_positions(heads.reset_index(drop=True), self.within):
                bsrc[pos] = pos[rng.permutation(len(pos))]
            src = first[bsrc][codes]
        for c in self.columns:
            out[c] = _take(df[c], src, df.index)
        return out


class Relocate(NullDesign):
    """Random placement: move whole units to random slots.

    Parameters
    ----------
    slots
        Table of places a unit can be moved to: one row per slot with ``spot_id``, ``x_um``, ``y_um`` and every
        stratum column named in ``within``. Further columns listed in ``slot_columns`` (e.g. ``"cell_type"``,
        ``"in_tissue"``) are written to the moved rows, because they belong to the place, not to the unit.
    unit
        Column identifying a unit; all rows of a unit move together and keep every other column (clone, reads,
        isotype, ...). Usually the spot (all molecules of one DNB) or a (spot, clone) key.
    within
        Stratum column(s) present in both the receptor table (constant within a unit) and ``slots``; a unit moves only
        to a slot of the same stratum (e.g. crowding class, GEX depth bin, library).
    window_um
        If given, a unit moves only to slots in the same square tile of this side length (local placement).
    weights
        Column of ``slots`` with non-negative sampling weights (e.g. RNA molecules of a cell).
    replace
        Draw slots with replacement (several units may land on one slot). Without replacement, a stratum with more
        units than slots raises ``ValueError``.
    slot_columns
        Columns copied from the slot to the moved rows. Defaults to the slot id and coordinates.
    spot_key, x_key, y_key
        Names of the unit-position columns in the receptor table (and in ``slots``).

    Notes
    -----
    Null model: conditional on the units, their attributes and the strata, each unit's position is a random draw
    from the eligible slots (with the given weights); positions carry no information about the unit.
    """

    def __init__(
        self,
        slots: pd.DataFrame,
        *,
        unit: str = S.SPOT,
        within: str | Sequence[str] | None = None,
        window_um: float | None = None,
        weights: str | None = None,
        replace: bool = False,
        slot_columns: Sequence[str] | None = None,
        spot_key: str = S.SPOT,
        x_key: str = S.X,
        y_key: str = S.Y,
    ) -> None:
        self.unit = unit
        self.within = _as_list(within)
        self.window_um = window_um
        self.weights = weights
        self.replace = replace
        self.spot_key, self.x_key, self.y_key = spot_key, x_key, y_key
        self.slot_columns = list(slot_columns) if slot_columns is not None else [spot_key, x_key, y_key]
        need = set(self.slot_columns) | set(self.within) | ({weights} if weights else set())
        if window_um is not None:
            need |= {x_key, y_key}
        missing = [c for c in need if c not in slots.columns]
        if missing:
            raise KeyError(f"Relocate: slot table lacks column(s) {sorted(missing)}")
        if window_um is not None and not window_um > 0:
            raise ValueError("window_um must be positive")
        self.slots = slots.reset_index(drop=True)
        w = None
        if weights:
            w = self.slots[weights].to_numpy(dtype=float)
            if (w < 0).any() or not np.isfinite(w).all():
                raise ValueError("slot weights must be finite and non-negative")
        self._w = w
        self._slot_groups = self._index(self.slots, self.x_key, self.y_key)

    def _keys(self, df: pd.DataFrame, x_key: str, y_key: str) -> pd.DataFrame:
        k = df[self.within].copy() if self.within else pd.DataFrame(index=df.index)
        if self.window_um is not None:
            k["_tile_x"] = np.floor(df[x_key].to_numpy(dtype=float) / self.window_um).astype(np.int64)
            k["_tile_y"] = np.floor(df[y_key].to_numpy(dtype=float) / self.window_um).astype(np.int64)
        return k

    def _index(self, df: pd.DataFrame, x_key: str, y_key: str) -> dict[Any, np.ndarray]:
        k = self._keys(df, x_key, y_key)
        if k.shape[1] == 0:
            return {(): np.arange(len(df))}
        return {
            (key if isinstance(key, tuple) else (key,)): pos
            for key, pos in k.groupby(list(k.columns), sort=False, observed=True, dropna=False).indices.items()
        }

    def describe(self) -> dict[str, Any]:
        """Return the parameters that define this design."""
        return {
            "design": "Relocate",
            "unit": self.unit,
            "within": self.within,
            "window_um": self.window_um,
            "weights": self.weights,
            "replace": self.replace,
            "n_slots": len(self.slots),
            "slot_columns": self.slot_columns,
        }

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Return a copy of ``df`` with every unit moved to a random eligible slot."""
        missing = [c for c in [self.unit, *self.within] if c not in df.columns]
        if missing:
            raise KeyError(f"Relocate: column(s) {missing} not in the receptor table")
        n = len(df)
        out = df.copy()
        if n == 0:
            return out
        codes, _ = pd.factorize(df[self.unit], use_na_sentinel=False)
        n_units = codes.max() + 1
        first = np.full(n_units, -1, dtype=np.int64)
        first[codes[::-1]] = np.arange(n)[::-1]
        heads = df.iloc[first].reset_index(drop=True)
        for c in self.within:
            nun = df.groupby(codes, sort=False)[c].nunique(dropna=False)
            if (nun > 1).any():
                raise ValueError(f"Relocate: stratum column {c!r} varies inside a unit ({self.unit!r})")
        unit_groups = self._index(heads, self.x_key, self.y_key)
        target = np.empty(n_units, dtype=np.int64)
        for key, upos in unit_groups.items():
            spos = self._slot_groups.get(key)
            if spos is None or len(spos) == 0:
                raise ValueError(f"Relocate: no slot for stratum {key!r} ({len(upos)} units)")
            p = None
            if self._w is not None:
                w = self._w[spos]
                if w.sum() <= 0:
                    raise ValueError(f"Relocate: slot weights sum to 0 in stratum {key!r}")
                p = w / w.sum()
            if not self.replace and len(upos) > len(spos):
                raise ValueError(
                    f"Relocate: stratum {key!r} has {len(upos)} units but {len(spos)} slots; use replace=True"
                )
            target[upos] = spos[rng.choice(len(spos), size=len(upos), replace=self.replace, p=p)]
        rows = target[codes]
        for c in self.slot_columns:
            out[c] = _take(self.slots[c], rows, df.index)
        return out


# ---------------------------------------------------------------------------------------------------- placebo engine


def _as_vector(value: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(value, Mapping):
        names = [str(k) for k in value]
        return np.array([float(v) for v in value.values()], dtype=float), names
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float), [str(i) for i in value.index]
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.ndim != 1:
        raise ValueError("a placebo statistic must return a scalar, a 1-D array, a Series or a dict of scalars")
    return arr, ([("value" if len(arr) == 1 else str(i)) for i in range(len(arr))])


def _one(args: tuple[Callable, pd.DataFrame, NullDesign, np.random.SeedSequence, bool]) -> np.ndarray:
    statistic, data, design, ss, pass_rng = args
    randomised = design(data, np.random.default_rng(ss))
    if pass_rng:
        v, _ = _as_vector(statistic(randomised, rng=np.random.default_rng(ss.spawn(1)[0])))
    else:
        v, _ = _as_vector(statistic(randomised))
    return v


@dataclass(frozen=True, repr=False)
class PlaceboResult(TestResult):
    """Result of :func:`placebo`; ``table`` has one row per component of the statistic."""

    observed: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def p_value(self) -> float | np.ndarray:
        """Monte-Carlo p-value(s) of the observed statistic against the placebo distribution."""
        p = self.table["p_value"].to_numpy()
        return float(p[0]) if len(p) == 1 else p


def placebo(
    statistic: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    design: NullDesign,
    *,
    n_perm: int = 200,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    n_jobs: int = 1,
    keep_null: bool = True,
    name: str = "placebo",
    pass_rng: bool = False,
) -> PlaceboResult:
    """Compare a statistic on the observed data with its distribution under a null design.

    Parameters
    ----------
    statistic
        Function of a receptor table returning a scalar, a 1-D array, a ``Series`` or a ``dict`` of scalars. It must
        run the complete pipeline that produces the metric (deduplication, filters, voting, thresholds).
    data
        The observed receptor table.
    design
        A :class:`Permute` or :class:`Relocate` (or any :class:`NullDesign`).
    n_perm
        Number of placebo data sets; the smallest attainable p-value is ``1 / (n_perm + 1)``.
    alternative
        Direction in which the observed value is expected to exceed the placebo if the receptors are positional.
    rng
        Seed or generator. Each placebo data set gets its own spawned generator, so results do not depend on
        ``n_jobs``.
    n_jobs
        Worker processes (the statistic must then be picklable: a module-level function or ``functools.partial``).
    keep_null
        Keep the placebo draws in ``result.null``.
    name
        Name of the result.
    pass_rng
        Call ``statistic(data, rng=generator)`` with its own generator for the observed data and for every placebo
        data set (for statistics that are themselves random, e.g. rarefied diversity), so their randomness is
        reproducible and independent of the design's.

    Returns
    -------
    PlaceboResult
        ``table`` columns: ``component``, ``observed``, ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi`` (2.5% and
        97.5%), ``ratio`` (observed / null mean), ``standardised_gap`` ((observed − null median) / null sd),
        ``p_value`` (Monte-Carlo), ``p_maxT`` (family-wise over components, for vector statistics), ``mc_se``
        (Monte-Carlo standard error of the p-value), ``n_perm``.

    Notes
    -----
    Null model: whatever ``design`` states; the statistic itself is re-computed from scratch on each placebo data set.
    """
    check_alternative(alternative)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    gen = as_rng(rng)
    obs, names = _as_vector(statistic(data, rng=gen.spawn(1)[0]) if pass_rng else statistic(data))
    seeds = (
        gen.bit_generator.seed_seq.spawn(n_perm)
        if hasattr(gen.bit_generator, "seed_seq")
        else [np.random.SeedSequence(int(x)) for x in gen.integers(0, 2**63 - 1, size=n_perm)]
    )
    jobs = [(statistic, data, design, s, pass_rng) for s in seeds]
    if n_jobs == 1:
        draws = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            draws = list(ex.map(_one, jobs, chunksize=max(1, n_perm // (4 * n_jobs))))
    null = np.vstack(draws)
    if null.shape[1] != len(obs):
        raise ValueError("the statistic returned a different number of components on placebo data")
    mu, sd, lo, hi = null_summary(null)
    med = np.nanmedian(null, axis=0)
    p = mc_pvalue(null, obs, alternative=alternative)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = obs / mu
        gap = (obs - med) / sd
    table = pd.DataFrame(
        {
            "component": names,
            "observed": obs,
            "null_mean": mu,
            "null_sd": sd,
            "null_lo": lo,
            "null_hi": hi,
            "ratio": ratio,
            "standardised_gap": gap,
            "p_value": p,
            "p_maxT": max_t_pvalue(null, obs, alternative=alternative) if len(obs) > 1 else p,
            "mc_se": np.sqrt(p * (1 - p) / n_perm),
            "n_perm": n_perm,
        }
    )
    summary = {
        "observed": obs[0] if len(obs) == 1 else obs,
        "null_mean": mu[0] if len(mu) == 1 else mu,
        "p_value": p[0] if len(p) == 1 else p,
    }
    params = {
        "design": design.describe(),
        "n_perm": n_perm,
        "alternative": alternative,
        "pass_rng": pass_rng,
        "rng_entropy": rng_entropy(gen),
        "statistic": getattr(statistic, "__name__", repr(statistic)),
    }
    return PlaceboResult(
        name=name, table=table, summary=summary, params=params, null=null if keep_null else None, observed=obs
    )


def compare_designs(
    statistic: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    designs: Mapping[str, NullDesign],
    *,
    n_perm: int = 200,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    n_jobs: int = 1,
) -> TestResult:
    """Run :func:`placebo` under several null designs and stack the tables (column ``design``).

    A conclusion that holds under one design and flips under another is not robust; report both.
    """
    gen = as_rng(rng)
    children = gen.spawn(len(designs))
    tables, params = [], {}
    for (label, design), child in zip(designs.items(), children, strict=True):
        r = placebo(
            statistic, data, design, n_perm=n_perm, alternative=alternative, rng=child, n_jobs=n_jobs, keep_null=False
        )
        tables.append(r.table.assign(design=label))
        params[label] = r.params
    table = pd.concat(tables, ignore_index=True)
    summary = {f"p[{d}]": float(t.p_value.iloc[0]) for d, t in zip(designs, tables, strict=True)}
    return TestResult(name="compare_designs", table=table, summary=summary, params=params)
