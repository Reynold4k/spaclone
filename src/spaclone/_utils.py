"""Shared helpers: random generators, Monte-Carlo p-values, multiple testing and the result object."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("spaclone")

Alternative = Literal["greater", "less", "two-sided"]
RngLike = int | np.random.Generator | np.random.SeedSequence | None


class SpacloneWarning(UserWarning):
    """Base class of the warnings raised by spaclone."""


class LowPowerWarning(SpacloneWarning):
    """A test was run with too few units or permutations to reach the requested resolution."""


def as_rng(rng: RngLike = None) -> np.random.Generator:
    """Normalise a seed, a ``SeedSequence`` or a generator into a ``numpy.random.Generator`` (SPEC 7)."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def rng_entropy(rng: np.random.Generator) -> Any:
    """Return a value that identifies the generator's seed, for the ``params`` of a result (best effort)."""
    seed_seq = getattr(rng.bit_generator, "seed_seq", None)
    return getattr(seed_seq, "entropy", None)


def mc_pvalue(null: np.ndarray, observed: np.ndarray | float, *, alternative: Alternative) -> np.ndarray:
    """Monte-Carlo p-value ``(1 + #{T* at least as extreme as T}) / (B + 1)``.

    Parameters
    ----------
    null
        Null draws, shape ``(B,)`` or ``(B, k)``. NaN draws are ignored (B is then the number of finite draws).
    observed
        Observed statistic, scalar or shape ``(k,)``.
    alternative
        ``"greater"`` counts null draws ``>= observed``; ``"less"`` counts ``<= observed``; ``"two-sided"`` doubles
        the smaller one-sided p-value (capped at 1).

    Returns
    -------
    numpy.ndarray
        p-values, never 0; the smallest possible value is ``1 / (B + 1)``.

    References
    ----------
    Davison & Hinkley 1997, doi:10.1017/CBO9780511802843; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    null = np.asarray(null, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if null.ndim == 1:
        null = null[:, None]
        obs = np.atleast_1d(obs)
    finite = np.isfinite(null)
    b = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        ge = ((null >= obs - 1e-12 * np.abs(obs)) & finite).sum(axis=0)
        le = ((null <= obs + 1e-12 * np.abs(obs)) & finite).sum(axis=0)
    p_ge = (1 + ge) / (1 + b)
    p_le = (1 + le) / (1 + b)
    if alternative == "greater":
        p = p_ge
    elif alternative == "less":
        p = p_le
    elif alternative == "two-sided":
        p = np.minimum(1.0, 2 * np.minimum(p_ge, p_le))
    else:
        raise ValueError(f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}")
    p = np.where(np.isfinite(obs), p, np.nan)
    return p


def max_t_pvalue(null: np.ndarray, observed: np.ndarray, *, alternative: Alternative = "greater") -> np.ndarray:
    """Family-wise (max-T) adjusted Monte-Carlo p-values over the columns of ``null``.

    Each component is standardised by its null mean and standard deviation; the adjusted p-value of component j is
    the share of null draws whose maximum standardised statistic over all components reaches component j's
    standardised observed value (Westfall & Young 1993; Nichols & Holmes 2002, doi:10.1002/hbm.1058).
    """
    null = np.asarray(null, dtype=float)
    obs = np.asarray(observed, dtype=float)
    mu = np.nanmean(null, axis=0)
    sd = np.nanstd(null, axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    z_null = (null - mu) / sd
    z_obs = (obs - mu) / sd
    if alternative == "less":
        z_null, z_obs = -z_null, -z_obs
    elif alternative == "two-sided":
        z_null, z_obs = np.abs(z_null), np.abs(z_obs)
    elif alternative != "greater":
        raise ValueError(f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}")
    mx = np.nanmax(np.where(np.isfinite(z_null), z_null, -np.inf), axis=1)
    b = np.isfinite(mx).sum()
    return (1 + (mx[:, None] >= z_obs[None, :] - 1e-12).sum(axis=0)) / (1 + b)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values; NaN p-values stay NaN and do not count towards m."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def null_summary(null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sd (ddof=1), 2.5% and 97.5% quantiles of null draws along axis 0."""
    null = np.asarray(null, dtype=float)
    if null.ndim == 1:
        null = null[:, None]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns give NaN summaries, not warnings
        return (
            np.nanmean(null, axis=0),
            np.nanstd(null, axis=0, ddof=1),
            np.nanquantile(null, 0.025, axis=0),
            np.nanquantile(null, 0.975, axis=0),
        )


@dataclass(frozen=True)
class TestResult:
    """Result of a spaclone test.

    Attributes
    ----------
    name
        Test name, e.g. ``"same_spot"``.
    table
        Tidy table, one row per tested unit, with standard columns when meaningful (``statistic``, ``expected``,
        ``null_mean``, ``null_sd``, ``z``, ``p_value``, ``q_value``, ``n_perm``).
    summary
        Headline numbers.
    params
        Every argument that defines the test, including the null design and the generator's seed entropy.
    null
        Optional null draws.
    """

    __test__ = False  # not a pytest test class

    name: str
    table: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    null: np.ndarray | None = None

    def __repr__(self) -> str:
        head = ", ".join(f"{k}={_fmt(v)}" for k, v in list(self.summary.items())[:6])
        return f"TestResult({self.name}: {head}; table {self.table.shape[0]}×{self.table.shape[1]})"


def _fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        return f"{v:.4g}"
    return str(v)


def check_alternative(alternative: str) -> None:
    """Raise if ``alternative`` is not one of the three allowed strings."""
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}")
