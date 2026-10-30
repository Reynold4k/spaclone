"""Observed-vs-placebo plots.

:func:`placement` draws the observed statistic as a point and its null (placebo) distribution as an interval, for
every row of a test whose table has ``statistic``, ``null_mean``, ``null_lo`` and ``null_hi`` columns
(``qc.colocalisation``, ``qc.own_type_share``, ``qc.read_floor``, ``qc.pairing``, ``qc.composition``, ``nulls.placebo``
after renaming ``observed`` to ``statistic``). :func:`placebo` draws the whole placebo distribution of a
:func:`spaclone.nulls.placebo` result (histogram, or one strip per component) with the observed value and its p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from spaclone._utils import TestResult
from spaclone.pl._colors import CHROME, LOCUS_SLOT, SLOTS, fixed_colors

__all__ = ["placebo", "placement"]

# first three slots of the reference categorical palette (validated all-pairs in light and dark mode)
_SERIES = {mode: SLOTS[mode][:3] for mode in ("light", "dark")}
_INK = {mode: (CHROME[mode]["ink"], CHROME[mode]["ink2"], CHROME[mode]["grid"]) for mode in ("light", "dark")}
_SURFACE = {mode: CHROME[mode]["bg"] for mode in ("light", "dark")}
#: Neutral fill of placebo (null) draws.
_NULL_FILL = {mode: CHROME[mode]["null"] for mode in ("light", "dark")}

_DEFAULTS: dict[str, dict[str, Any]] = {
    "colocalisation": {"x": "band", "hue": "design", "ylabel": "share of same-clone pairs in the band"},
    "own_type_share": {"x": "min_reads", "hue": None, "log_x": True, "ylabel": "share of records on their own type"},
    "read_floor": {"x": "floor", "hue": "side", "log_x": True, "ylabel": "share of clonotypes sharing a partner"},
    "pairing": {"x": "light_locus", "hue": "design", "ylabel": "mean best-partner z"},
    "composition": {"x": "measure", "hue": "locus", "standardise": True, "ylabel": "standardised statistic (z)"},
}


def _hue_vocabulary(col: pd.Series) -> list[str]:
    """Levels of a hue column in a fixed order: declared categories, else order of first appearance in the result."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return [str(c) for c in col.cat.categories]
    return [str(v) for v in pd.unique(col.dropna())]


def _hue_colours(vocab: list[str], style: str) -> dict[str, str]:
    """Chains keep their fixed colour; other levels take the slots by their position in ``vocab``."""
    named = {v: SLOTS[style][LOCUS_SLOT[v]] for v in vocab if v in LOCUS_SLOT}
    return fixed_colors(vocab, mode=style, named=named)


def _table_and_defaults(result: TestResult | pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(result, TestResult):
        table = result.table
        d = dict(_DEFAULTS.get(result.name, {}))
    else:
        table = pd.DataFrame(result)
        d = {}
    if "statistic" not in table.columns and "observed" in table.columns:  # nulls.placebo tables
        table = table.rename(columns={"observed": "statistic"})
        d.setdefault("x", "component")
    return table, d


def placement(
    result: TestResult | pd.DataFrame,
    *,
    x: str | None = None,
    hue: str | None = None,
    query: str | None = None,
    standardise: bool | None = None,
    log_x: bool | None = None,
    ax: Axes | None = None,
    style: Literal["light", "dark"] = "dark",
    ylabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot the observed statistic against its null interval for every row of a test table.

    Parameters
    ----------
    result
        A :class:`~spaclone.TestResult` (or its table) with ``statistic``, ``null_mean``, ``null_lo``, ``null_hi``.
    x
        Column on the x axis (categories, or numbers such as read floors). Defaults depend on the test: ``band`` for
        ``colocalisation`` (``library_id`` when there are several libraries), ``min_reads`` for ``own_type_share``,
        ``floor`` for ``read_floor``, ``light_locus`` for ``pairing``, ``measure`` for ``composition``.
    hue
        Column whose levels are drawn side by side in different colours (at most three levels; use ``query`` or
        several axes for more). Defaults: ``design`` (``colocalisation``, ``pairing``), ``side`` (``read_floor``).
    query
        ``DataFrame.query`` string selecting rows before plotting, e.g. ``"design == 'stratified'"``.
    standardise
        Plot ``(value - null_mean) / null_sd`` instead of raw values (default for ``composition``, whose measures
        have different scales).
    log_x
        Logarithmic x axis for numeric x (default for read floors).
    ax
        Axes to draw into; a new figure is created when ``None``.
    style
        ``"dark"`` (the house style, default) or ``"light"`` surface.
    ylabel, title
        Axis label and title; sensible defaults per test.

    Returns
    -------
    matplotlib.axes.Axes
        The axes; nothing is shown or saved.
    """
    table, d = _table_and_defaults(result)
    if style not in _SERIES:
        raise ValueError("style must be 'light' or 'dark'")
    full = table  # colours follow the hue level over the whole result, whatever query= keeps
    if query is not None:
        table = table.query(query)
    need = {"statistic", "null_mean", "null_lo", "null_hi"}
    missing = sorted(need - set(table.columns))
    if missing:
        raise ValueError(f"the result table lacks column(s) {missing}")
    name = result.name if isinstance(result, TestResult) else ""
    if name == "colocalisation" and x is None and table["library_id"].nunique() > 1:
        if query is None:
            first = table["design"].iloc[0]
            table = table[table["design"] == first]
        d.update(x="library_id", hue="band")
    if name == "pairing" and x is None and table["library_id"].nunique() > 1:
        d.update(x="library_id", hue="light_locus")
    if name == "own_type_share" and x is None and "library_id" in table.columns and query is None:
        if (table["library_id"] == "all").any():
            table = table[table["library_id"] == "all"]
    x = x or d.get("x") or table.columns[0]
    hue = hue if hue is not None else d.get("hue")
    if hue is not None and (hue not in table.columns or full[hue].nunique() <= 1):
        hue = None  # a single level over the whole result needs no colour key
    standardise = d.get("standardise", False) if standardise is None else standardise
    table = table.dropna(subset=["statistic"])
    if table.empty:
        raise ValueError("no finite statistic to plot")

    y = table["statistic"].to_numpy(float)
    mu = table["null_mean"].to_numpy(float)
    lo = table["null_lo"].to_numpy(float)
    hi = table["null_hi"].to_numpy(float)
    if standardise:
        sd = table["null_sd"].to_numpy(float) if "null_sd" in table.columns else np.full(len(table), np.nan)
        sd = np.where(sd > 0, sd, np.nan)
        y, lo, hi, mu = (y - mu) / sd, (lo - mu) / sd, (hi - mu) / sd, np.zeros(len(table))

    levels: list[str | None] = [None]
    hue_colour: dict[str, str] = {}
    if hue is not None:  # colours follow the level over the whole result, whatever query= keeps
        vocab = _hue_vocabulary(full[hue] if hue in full.columns else table[hue])
        present = set(table[hue].astype(str))
        levels = [v for v in vocab if v in present] or [None]
        hue_colour = _hue_colours(vocab, style)
    if len(levels) > len(_SERIES[style]):
        raise ValueError(f"at most {len(_SERIES[style])} hue levels; select rows with query= or plot several axes")
    xv = table[x]
    if pd.api.types.is_numeric_dtype(xv) and xv.isna().all():  # e.g. one level without read counts
        xv = pd.Series("all records", index=table.index)
    numeric = pd.api.types.is_numeric_dtype(xv)
    if numeric:
        xpos = xv.to_numpy(float)
        cats = None
    else:
        cats = list(pd.unique(xv.astype(str)))
        xpos = np.array([cats.index(v) for v in xv.astype(str)], dtype=float)
    log_x = bool(d.get("log_x", False) if log_x is None else log_x) and numeric and bool((xpos > 0).all())

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(1.4 + 1.1 * max(3, len(np.unique(xpos))), 3.6))
    ink, ink2, grid = _INK[style]
    surface = _SURFACE[style]
    if style == "dark":
        ax.figure.set_facecolor(surface)
        ax.set_facecolor(surface)
    n_lev = len(levels)
    width = 0.5 if not numeric else 0.12
    handles = []
    for i, lev in enumerate(levels):
        c = _SERIES[style][0] if lev is None else hue_colour[lev]
        m = np.ones(len(table), bool) if lev is None else (table[hue].astype(str) == lev).to_numpy()
        off = (i - (n_lev - 1) / 2) * (width / max(n_lev, 1)) if n_lev > 1 else 0.0
        px = xpos[m] * 10**off if log_x else xpos[m] + (off * (np.ptp(xpos) or 1.0) if numeric else off)
        ax.vlines(px, lo[m], hi[m], color=c, lw=2, alpha=0.6, zorder=2)
        ax.scatter(px, mu[m], marker="_", s=110, color=c, linewidths=2, zorder=3)
        ax.scatter(px, y[m], marker="o", s=64, color=c, edgecolors=surface, linewidths=1.5, zorder=4)
        if lev is not None:
            handles.append(Line2D([], [], ls="", marker="o", ms=8, color=c, mec=surface, label=f"{hue} = {lev}"))
    handles += [
        Line2D([], [], ls="", marker="o", ms=8, color=ink2, mec=surface, label="observed"),
        Line2D([0, 1], [0, 0], color=ink2, lw=2, marker="_", ms=10, mew=2, label="null mean and 95% interval"),
    ]
    if cats is not None:
        ax.set_xticks(range(len(cats)), cats)
        ax.set_xlim(-0.6, len(cats) - 0.4)
    elif log_x:
        ax.set_xscale("log")
        ticks = np.unique(xpos)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel(x.replace("_", " "), color=ink2)
    ax.set_ylabel(ylabel or d.get("ylabel", "statistic"), color=ink2)
    if title:
        ax.set_title(title, color=ink, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink2)
    ax.tick_params(colors=ink2)
    leg = ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for txt in leg.get_texts():  # outside the plot area: the key never covers an interval
        txt.set_color(ink)
    return ax


def _component_index(names: list[str], component: str | int | None) -> int | None:
    if component is None:
        return None
    if isinstance(component, int | np.integer) and not isinstance(component, bool):
        if not -len(names) <= int(component) < len(names):
            raise IndexError(f"component {component} out of range for {len(names)} components")
        return int(component) % len(names)
    if str(component) not in names:
        raise KeyError(f"component {component!r} not in the result; components: {names}")
    return names.index(str(component))


def _style_axes(ax: Axes, style: str, *, grid_axis: str, created: bool) -> tuple[str, str, str]:
    ink, ink2, grid = _INK[style]
    if style == "dark":
        if created:  # never repaint a figure the caller owns (it may hold panels in another style)
            ax.figure.set_facecolor(_SURFACE[style])
        ax.set_facecolor(_SURFACE[style])
    ax.grid(axis=grid_axis, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink2)
    ax.tick_params(colors=ink2)
    return ink, ink2, grid


def _fmt_p(p: float) -> str:
    return "p = NA" if not np.isfinite(p) else f"p = {p:.2g}" if p < 0.001 else f"p = {p:.3g}"


def placebo(
    result: Any,
    *,
    component: str | int | None = None,
    kind: Literal["auto", "hist", "strip"] = "auto",
    bins: int | str | Sequence[float] = "auto",
    ax: Axes | None = None,
    style: Literal["light", "dark"] = "dark",
    xlabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Draw the placebo (null) distribution of a :func:`spaclone.nulls.placebo` result with the observed value and p.

    One component (a scalar statistic, or the one chosen with ``component``): a histogram of the placebo draws and the
    observed value as a vertical line; the legend gives the observed value and its Monte-Carlo p-value. Several
    components (``component=None``): one strip per component, the placebo draws as dots and the observed value as a
    large dot, each component standardised by its placebo mean and standard deviation so that components on
    different scales share one axis, with each component's p-value on the right and the smallest family-wise
    (max-T) p-value in the legend.

    Parameters
    ----------
    result
        A :class:`~spaclone.nulls.PlaceboResult` that kept its draws (``placebo(..., keep_null=True)``, the default).
    component
        Component to draw, by name (the ``component`` column of the table) or position; ``None``: every component
        (a histogram when there is only one).
    kind
        ``"hist"``, ``"strip"`` or ``"auto"`` (histogram for one component, strips for several).
    bins
        Histogram bins (``numpy.histogram_bin_edges``; at most 60). Integer-valued draws spanning at most 60 values get
        one bar per value.
    ax
        Axes to draw into; a new figure when ``None``.
    style
        ``"dark"`` (the house style, default) or ``"light"`` surface (with a supplied ``ax`` the figure itself is
        left as it is).
    xlabel, title
        Axis label (default: the statistic's name, the component name, or "standardised" for strips) and title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes; nothing is shown or saved.

    Examples
    --------
    >>> from spaclone import nulls, pl  # doctest: +SKIP
    >>> res = nulls.placebo(stat, air, nulls.Permute(["clone_id"]), n_perm=200, rng=0)  # doctest: +SKIP
    >>> ax = pl.placebo(res)  # doctest: +SKIP
    """
    if style not in _SERIES:
        raise ValueError("style must be 'light' or 'dark'")
    if kind not in ("auto", "hist", "strip"):
        raise ValueError("kind must be 'auto', 'hist' or 'strip'")
    null = getattr(result, "null", None)
    table = getattr(result, "table", None)
    if null is None or table is None:
        raise ValueError("the result keeps no placebo draws: run nulls.placebo(..., keep_null=True)")
    null = np.asarray(null, dtype=float)
    null = null[:, None] if null.ndim == 1 else null
    obs = getattr(result, "observed", None)
    if (obs is None or not np.size(obs)) and "observed" not in table.columns:
        raise ValueError(
            "pl.placebo draws nulls.placebo results (an 'observed' statistic and its draws); for a placement test "
            f"({getattr(result, 'name', 'this result')!r}) use pl.placement"
        )
    obs = np.asarray(table["observed"] if obs is None or not np.size(obs) else obs, dtype=float).ravel()
    names = [str(c) for c in table["component"]] if "component" in table.columns else [str(i) for i in range(len(obs))]
    if null.shape[1] != len(obs):
        raise ValueError(f"null draws have {null.shape[1]} components, the observed statistic {len(obs)}")
    pvals = table["p_value"].to_numpy(dtype=float) if "p_value" in table.columns else np.full(len(obs), np.nan)
    n_perm = null.shape[0]
    params = getattr(result, "params", None) or {}
    alt = params.get("alternative", "greater")
    j = _component_index(names, component)
    if kind == "auto":
        kind = "hist" if (j is not None or len(obs) == 1) else "strip"
    if kind == "hist" and j is None:
        if len(obs) > 1:
            raise ValueError("a histogram shows one component: pass component= (name or position)")
        j = 0
    blue = _SERIES[style][0]
    surface = _SURFACE[style]
    created = ax is None
    if created:
        import matplotlib.pyplot as plt

        rows = len(obs) if (kind == "strip" and j is None) else 1
        _, ax = plt.subplots(figsize=(6.0, 3.6) if kind == "hist" else (6.5, max(2.4, 0.42 * rows + 1.5)))

    if kind == "hist":
        x = null[:, j]
        x = x[np.isfinite(x)]
        o = float(obs[j])
        if len(x) and np.allclose(x, np.round(x)) and np.ptp(x) <= 60:
            edges = np.arange(x.min() - 0.5, x.max() + 1.5)
        else:
            edges = np.histogram_bin_edges(x if len(x) else [0.0, 1.0], bins=bins)
            if len(edges) > 61:
                edges = np.linspace(edges[0], edges[-1], 61)
        counts, edges = np.histogram(x, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=_NULL_FILL[style], edgecolor=surface,
               linewidth=0.5, zorder=2)  # fmt: skip
        ink, ink2, _ = _style_axes(ax, style, grid_axis="y", created=created)
        ax.axvline(o, color=blue, lw=2.2, zorder=4)
        lo_x, hi_x = (min(edges[0], o), max(edges[-1], o)) if np.isfinite(o) else (edges[0], edges[-1])
        pad = 0.04 * ((hi_x - lo_x) or 1.0)
        ax.set_xlim(lo_x - pad, hi_x + pad)
        ax.set_ylim(0, max(counts.max(initial=0), 1) * 1.25)
        stat = str(params.get("statistic", ""))
        default_x = names[j] if len(names) > 1 else (stat if stat and not stat.startswith("<") else "statistic")
        ax.set_xlabel(xlabel or default_x, color=ink2)
        ax.set_ylabel("placebo data sets", color=ink2)
        handles = [
            Patch(facecolor=_NULL_FILL[style], edgecolor="none", label=f"placebo: {n_perm:,} draws"),
            Line2D([0, 1], [0, 0], color=blue, lw=2.2,
                   label=f"observed {o:.4g}, {_fmt_p(float(pvals[j]))} ({alt})"),
        ]  # fmt: skip
        leg = ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    else:
        idx = list(range(len(obs))) if j is None else [j]
        sub = null[:, idx]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=1) if n_perm > 1 else np.full(len(idx), np.nan)
        scale = np.maximum(np.abs(mu), np.nanmax(np.abs(sub), axis=0, initial=0.0))
        ok = np.isfinite(sd) & (sd > 1e-9 * np.maximum(scale, 1e-300))  # float noise is no placebo variance
        gen = np.random.default_rng(0)  # fixed jitter: the same figure every time
        ink, ink2, _ = _style_axes(ax, style, grid_axis="x", created=created)
        lim = 1.0
        for r, (c, m_, s_, good) in enumerate(zip(idx, mu, sd, ok, strict=True)):
            if not good:
                continue
            z = (null[:, c] - m_) / s_
            z = z[np.isfinite(z)]
            zo = (obs[c] - m_) / s_
            lim = max(lim, float(np.max(np.abs(z))) if len(z) else 1.0, abs(zo) if np.isfinite(zo) else 0.0)
            ax.scatter(z, r + gen.uniform(-0.22, 0.22, len(z)), s=7, color=_NULL_FILL[style], lw=0, zorder=2)
            ax.scatter([zo], [r], s=70, color=blue, edgecolors=surface, linewidths=1.5, zorder=4)
        lim *= 1.08
        ax.set_xlim(-lim, lim)
        for r, (c, good) in enumerate(zip(idx, ok, strict=True)):
            text = _fmt_p(float(pvals[c])) if good else "no placebo variance"
            ax.annotate(text, xy=(1.0, r), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=ink)  # fmt: skip
        ax.axvline(0, color=ink2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_yticks(range(len(idx)), [names[c] for c in idx])
        ax.set_ylim(len(idx) - 0.5, -0.5)
        ax.set_xlabel(xlabel or "standardised: (value − placebo mean) / placebo SD", color=ink2)
        handles = [
            Line2D([], [], ls="", marker="o", ms=4, color=_NULL_FILL[style], label=f"placebo: {n_perm:,} draws"),
            Line2D([], [], ls="", marker="o", ms=8, color=blue, mec=surface, label="observed"),
        ]
        if len(idx) > 1 and "p_maxT" in table.columns:
            pmax = float(np.nanmin(table["p_maxT"].to_numpy(dtype=float)[idx]))
            handles.append(Line2D([], [], ls="", label=f"family-wise (max-T) smallest {_fmt_p(pmax)}"))
        leg = ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                        ncol=len(handles), borderaxespad=0.3, handletextpad=0.3, columnspacing=1.2)  # fmt: skip
    for txt in leg.get_texts():
        txt.set_color(ink)
    if title:
        ax.set_title(title, color=ink, loc="left", pad=22 if kind == "strip" else 6)
    return ax
