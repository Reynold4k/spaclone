"""Repertoire plots: diversity with placebo bands, overlap heatmap, gene usage, spectratype, expansion, dispersion.

Every function takes ``ax=None`` and returns the Axes; nothing is shown or saved. Colours follow one validated palette:
observed values in blue, placebo (null) ranges in neutral grey, magnitudes on a single blue ramp, groups by name
(chains and house classes such as regions in their fixed colours, other groups by their sorted position; never cycled).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Patch

from spaclone.pl._colors import CHROME, NEUTRAL, RAMPS, SLOTS, fixed_colors, ink_on, locus_colors, named_colors

__all__ = ["clonal_expansion", "clone_dispersion", "diversity", "gene_usage", "overlap", "spectratype"]

_LIGHT = CHROME["light"]
INK = _LIGHT["ink"]
INK_2 = _LIGHT["ink2"]
GRID = _LIGHT["grid"]
OBSERVED = SLOTS["light"][0]
NULL_FILL = _LIGHT["null"]
NULL_EDGE = _LIGHT["null_edge"]
MUTED = NEUTRAL["light"]
#: Categorical order (validated light-mode slots); groups take them in this order.
CATEGORICAL = SLOTS["light"]
_SEQ = RAMPS["blue"]["light"]
SEQUENTIAL = LinearSegmentedColormap.from_list("spaclone_blue", _SEQ)
_ORDINAL = RAMPS["blue_ordinal"]["light"]


def _ax(ax: Axes | None, size: tuple[float, float] = (5.0, 3.6)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _style(ax: Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _table(obj: Any) -> pd.DataFrame:
    return obj.table if hasattr(obj, "table") else obj


def _group_cols(table: pd.DataFrame, value_cols: Sequence[str]) -> list[str]:
    return [c for c in table.columns if c not in value_cols]


def _labels(table: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series("all", index=table.index)
    return table[cols].astype(str).agg("|".join, axis=1)


def diversity(
    result: Any,
    *,
    metric: str | None = None,
    kind: str = "group",
    ax: Axes | None = None,
    color: str = OBSERVED,
) -> Axes:
    """Diversity per group (or per contrast) with its rarefaction interval and the placebo range.

    Parameters
    ----------
    result
        :class:`~spaclone.TestResult` of :func:`spaclone.tl.diversity` (or its ``table``).
    metric
        Metric to show; defaults to the first one in the table.
    kind
        ``"group"`` (one point per group) or ``"contrast"`` (one point per pairwise difference, with a line at 0).
    ax
        Axes to draw on; a new figure when ``None``.
    color
        Colour of the observed values.

    Returns
    -------
    matplotlib.axes.Axes
        Points = the statistic (rarefied mean when rarefied), bars = 2.5-97.5% of the rarefaction draws, grey boxes =
        2.5-97.5% of the placebo with a tick at its mean.
    """
    t = _table(result)
    metric = metric or str(t["metric"].iloc[0])
    t = t[(t["metric"] == metric) & (t["kind"] == kind)].reset_index(drop=True)
    if t.empty:
        raise ValueError(f"no {kind!r} rows for metric {metric!r}")
    ax = _ax(ax)
    x = np.arange(len(t))
    has_null = "null_lo" in t and t["null_lo"].notna().any()
    if has_null:
        for i, r in t.iterrows():
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.28, r["null_lo"]), 0.56, r["null_hi"] - r["null_lo"], facecolor=NULL_FILL, edgecolor="none"
                )
            )
            ax.plot([i - 0.28, i + 0.28], [r["null_mean"]] * 2, color=NULL_EDGE, lw=1.5)
    y = t["statistic"].to_numpy(dtype=float)
    lo = t["rarefied_lo"].to_numpy(dtype=float) if "rarefied_lo" in t else np.full(len(t), np.nan)
    hi = t["rarefied_hi"].to_numpy(dtype=float) if "rarefied_hi" in t else np.full(len(t), np.nan)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.errorbar(
            x[ok], y[ok], yerr=np.vstack([y[ok] - lo[ok], hi[ok] - y[ok]]), fmt="none", ecolor=color, elinewidth=2
        )
    ax.scatter(x, y, s=64, color=color, zorder=3, edgecolor=_LIGHT["bg"], linewidth=1.5)
    if kind == "contrast":
        ax.axhline(0, color=NULL_EDGE, lw=1, ls="--")
    ax.set_xticks(x, t["group"].astype(str), rotation=0 if len(t) <= 6 else 45, ha="center" if len(t) <= 6 else "right")
    ax.set_xlim(-0.6, len(t) - 0.4)
    rare = "rarefied_to" in t and pd.notna(t["rarefied_to"].iloc[0])
    lab = metric + (f" (rarefied to {int(t['rarefied_to'].iloc[0])} units)" if rare else "")
    ax.set_ylabel(lab if kind == "group" else f"difference in {lab}")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=color, label="observed")]
    if ok.any():
        handles[0].set_label("observed (95% of rarefaction draws)")
    if has_null:
        handles.append(Patch(facecolor=NULL_FILL, edgecolor="none", label="placebo 95% (tick: mean)"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return ax


def overlap(
    result: Any,
    *,
    metric: str | None = None,
    ax: Axes | None = None,
    cmap: Any = None,
    annotate: bool = True,
    alpha: float = 0.05,
) -> Axes:
    """Heatmap of pairwise repertoire overlap.

    Parameters
    ----------
    result
        :class:`~spaclone.TestResult` of :func:`spaclone.tl.overlap` (or its ``table``).
    metric
        Metric to show; defaults to the first one.
    ax
        Axes to draw on; a new figure when ``None``.
    cmap
        Colormap (default: a single-hue blue ramp, 0 to 1).
    annotate
        Write the value in every cell, with ``*`` when the placebo q-value is below ``alpha``.
    alpha
        Level of the ``*`` marks.

    Returns
    -------
    matplotlib.axes.Axes
        Symmetric matrix of the similarity (1 on the diagonal).
    """
    t = _table(result)
    metric = metric or str(t["metric"].iloc[0])
    t = t[t["metric"] == metric]
    groups = list(dict.fromkeys([*t["group_a"].astype(str), *t["group_b"].astype(str)]))
    k = len(groups)
    pos = {g: i for i, g in enumerate(groups)}
    mat = np.eye(k)
    star = np.zeros((k, k), bool)
    for _, r in t.iterrows():
        i, j = pos[str(r["group_a"])], pos[str(r["group_b"])]
        mat[i, j] = mat[j, i] = r["value"]
        q = r.get("q_value", np.nan)
        star[i, j] = star[j, i] = bool(pd.notna(q) and q < alpha)
    ax = _ax(ax, (1.2 + 0.7 * k, 0.9 + 0.7 * k))
    im = ax.imshow(mat, cmap=cmap or SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(k), groups, rotation=45, ha="right")
    ax.set_yticks(range(k), groups)
    if annotate:
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                v = mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}{'*' if star[i, j] else ''}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=ink_on(to_hex(im.cmap(im.norm(v)))),
                )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(metric.replace("_", " "), color=INK)
    cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelcolor=INK)
    return ax


def gene_usage(
    table: pd.DataFrame, *, top: int = 20, ax: Axes | None = None, cmap: Any = None, color: str = OBSERVED
) -> Axes:
    """Gene usage: a genes x groups heatmap of shares (bars for a single group).

    Parameters
    ----------
    table
        Output of :func:`spaclone.tl.gene_usage`.
    top
        Number of genes shown (largest mean share over groups).
    ax
        Axes to draw on; a new figure when ``None``.
    cmap
        Colormap of the heatmap (default: single-hue blue ramp from 0).
    color
        Bar colour for a single group.

    Returns
    -------
    matplotlib.axes.Axes
    """
    gcols = _group_cols(table, ("gene", "count", "share"))
    lab = _labels(table, gcols)
    wide = table.assign(_g=lab.to_numpy()).pivot_table(index="gene", columns="_g", values="share", aggfunc="sum")
    wide = wide.fillna(0.0)
    order = wide.mean(axis=1).sort_values(ascending=False).index[:top]
    wide = wide.loc[order]
    if wide.shape[1] == 1:
        ax = _ax(ax, (4.8, 0.25 * len(wide) + 1.0))
        y = np.arange(len(wide))[::-1]
        ax.barh(y, 100 * wide.iloc[:, 0].to_numpy(), color=color, height=0.7)
        ax.set_yticks(y, wide.index)
        ax.set_xlabel("share (%)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        _style(ax)
        return ax
    ax = _ax(ax, (1.6 + 0.6 * wide.shape[1], 0.25 * len(wide) + 1.2))
    im = ax.imshow(100 * wide.to_numpy(), cmap=cmap or SEQUENTIAL, vmin=0, aspect="auto")
    ax.set_yticks(range(len(wide)), wide.index)
    ax.set_xticks(range(wide.shape[1]), wide.columns, rotation=45, ha="right")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("share (%)", color=INK)
    cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelcolor=INK)
    return ax


def spectratype(table: pd.DataFrame, *, ax: Axes | None = None, colors: Sequence[str] | None = None) -> Axes:
    """CDR3 length distributions, one line per group.

    Parameters
    ----------
    table
        Output of :func:`spaclone.tl.cdr3_length`.
    ax
        Axes to draw on; a new figure when ``None``.
    colors
        Slot colours for the groups, given to the groups in sorted order (default: chains and house classes such as
        regions in their fixed colours, other groups by sorted position in :data:`CATEGORICAL`); groups beyond the
        slots are drawn in grey.

    Returns
    -------
    matplotlib.axes.Axes
    """
    gcols = _group_cols(table, ("length", "count", "share"))
    lab = _labels(table, gcols)
    ax = _ax(ax)
    groups = list(dict.fromkeys(lab))
    vocab = sorted(groups)
    if colors is None:
        named = {g: locus_colors("light")[g] for g in vocab if g in locus_colors("light")}
        named |= {g: c for g, c in named_colors(vocab, "light").items() if g not in named}  # regions, cell classes
        cmap = fixed_colors(vocab, mode="light", named=named, slots=CATEGORICAL, beyond=MUTED)
    else:
        cmap = fixed_colors(vocab, mode="light", slots=tuple(colors), beyond=MUTED)
    for g in groups:
        sub = table[lab.to_numpy() == g]
        ax.plot(sub["length"], 100 * sub["share"], color=cmap[g], lw=2, marker="o", ms=4, label=g)
    ax.set_xlabel("CDR3 length (aa)")
    ax.set_ylabel("share (%)")
    if len(groups) > 1:
        ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return ax


def clonal_expansion(table: pd.DataFrame, *, value: str = "share_units", ax: Axes | None = None) -> Axes:
    """Stacked bars of each group's units (or clones) by clone-size class.

    Parameters
    ----------
    table
        Output of :func:`spaclone.tl.clonal_expansion`.
    value
        ``"share_units"`` or ``"share_clones"``.
    ax
        Axes to draw on; a new figure when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    gcols = _group_cols(table, ("expansion", "n_units", "share_units", "n_clones", "share_clones"))
    lab = _labels(table, gcols)
    t = table.assign(_g=lab.to_numpy())
    cats = (
        list(t["expansion"].cat.categories) if hasattr(t["expansion"], "cat") else list(dict.fromkeys(t["expansion"]))
    )
    groups = list(dict.fromkeys(t["_g"]))
    wide = t.pivot_table(index="_g", columns="expansion", values=value, aggfunc="sum", observed=False)
    wide = wide.reindex(index=groups, columns=cats).fillna(0.0)
    ax = _ax(ax)
    cols = (
        _ORDINAL[-len(cats) :]
        if len(cats) <= len(_ORDINAL)
        else [SEQUENTIAL(v) for v in np.linspace(0.3, 1, len(cats))]
    )
    bottom = np.zeros(len(groups))
    x = np.arange(len(groups))
    for c, col in zip(cats, cols, strict=True):
        v = 100 * wide[c].to_numpy()
        ax.bar(x, v, 0.65, bottom=bottom, color=col, edgecolor=_LIGHT["bg"], linewidth=1.5, label=str(c))
        bottom += v
    ax.set_xticks(x, groups)
    ax.set_ylabel(("units" if value == "share_units" else "clones") + " (%)")
    ax.set_ylim(0, 100)
    ax.legend(
        title="clone size", frameon=False, fontsize=9, title_fontsize=9, bbox_to_anchor=(1.0, 1.0), loc="upper left"
    )
    _style(ax)
    return ax


def clone_dispersion(result: Any, *, measure: str | None = None, ax: Axes | None = None, color: str = OBSERVED) -> Axes:
    """Per-clone dispersion: observed / null ratio against clone size, significant clones highlighted.

    Parameters
    ----------
    result
        :class:`~spaclone.TestResult` of :func:`spaclone.tl.clone_dispersion` (or its ``table``).
    measure
        Measure to show; defaults to the first one.
    ax
        Axes to draw on; a new figure when ``None``.
    color
        Colour of the significant clones (others grey).

    Returns
    -------
    matplotlib.axes.Axes
    """
    t = _table(result)
    measure = measure or str(t["measure"].iloc[0])
    t = t[t["measure"] == measure]
    ax = _ax(ax)
    sig = t["significant"].to_numpy(dtype=bool)
    ax.scatter(t.loc[~sig, "n_units"], t.loc[~sig, "ratio"], s=14, color=MUTED, lw=0, label="not significant")
    ax.scatter(t.loc[sig, "n_units"], t.loc[sig, "ratio"], s=22, color=color, lw=0, label="clustered (q < alpha)")
    ax.axhline(1, color=NULL_EDGE, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("units of the clone")
    name = {"mean_nn": "mean nearest-neighbour distance", "mst": "minimum-spanning-tree length"}.get(measure, measure)
    ax.set_ylabel(f"{name}\nobserved / random units")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return ax
