"""Quality-control plots of receptor records.

Read support and its knee, valid / productive shares by chain, clones per spot, and a per-spot QC map.

Every function takes ``ax=None`` and returns the Axes (``qc_panel`` returns a Figure); none calls ``show`` or
``savefig``. Chains keep one colour everywhere (IGH blue, IGK orange, IGL aqua, TRA yellow, TRB magenta, TRG green,
TRD violet), so a filter that drops a chain never repaints the others; other groups take the validated slots by their
sorted position. The spot map follows the image convention (y increases downwards) unless ``invert_y=False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl._colors import CHROME, RAMPS, SLOTS, fixed_colors, locus_colors
from spaclone.pl._frame import SectionFrame, attach

__all__ = ["LOCUS_COLORS", "clones_per_spot", "qc_panel", "read_support", "spot_qc_map", "valid_shares"]

#: Fixed colour of every chain (categorical slots of the validated house palette, light mode).
LOCUS_COLORS: dict[str, str] = locus_colors("light")
_INK, _INK2, _MUTED, _GRID, _AXIS = (CHROME["light"][k] for k in ("ink", "ink2", "muted", "grid", "axis"))
_BLUE_RAMP = list(RAMPS["blue"]["light"])


def _group_colors(names: Sequence[Any]) -> dict[str, str]:
    """Chains in their fixed colours; any other group by its sorted position among the groups (never cycled)."""
    vocab = sorted({str(n) for n in names})
    return fixed_colors(vocab, mode="light", named={n: LOCUS_COLORS[n] for n in vocab if n in LOCUS_COLORS})


def _ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.6))
    return ax


def _style(ax, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.xaxis.label.set_color(_INK2)
    ax.yaxis.label.set_color(_INK2)
    ax.title.set_color(_INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def read_support(
    data: Any,
    *,
    by: str | Sequence[str] | None = S.LOCUS,
    share: bool = True,
    knee: bool = True,
    ax=None,
    reads_key: str = S.READS,
):
    """Records kept against the read floor, one line per group, with the knee of each curve.

    Parameters
    ----------
    data
        A receptor table (the curve is computed with :func:`spaclone.pp.read_support_curve`) or the
        :class:`~spaclone.pp.ReadSupport` result.
    by
        Grouping of the curves when ``data`` is a table (default: the chain; ``None`` = library and chain).
    share
        Plot the share of records kept (otherwise the count).
    knee
        Mark each curve's knee (filled circle) and inflection (open diamond), see :func:`spaclone.pp.knee_point`.
        Their read floors are written in the legend, not next to the points, so labels never overlap.
    ax
        Axes to draw into.
    reads_key
        Reads column of the table.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.lines import Line2D

    from spaclone.pp.metrics import ReadSupport, read_support_curve

    if isinstance(data, ReadSupport):
        curve, kn = data
    else:
        b = [] if by is None else ([by] if isinstance(by, str) else list(by))
        b = [c for c in b if c in data.columns]
        curve, kn = read_support_curve(data, by=b, reads_key=reads_key)
    ax = _ax(ax)
    gcols = [c for c in curve.columns if c not in ("min_reads", "records", "share_records", "spots", "share_spots")]
    ycol = "share_records" if share else "records"
    groups = list(curve.groupby(gcols, sort=True)) if gcols else [((), curve)]
    keys = [k if isinstance(k, tuple) else (k,) for k, _ in groups]
    colors = _group_colors([k[-1] if k else "" for k in keys])
    handles, floors_all = [], []
    for key, (_, g) in zip(keys, groups, strict=True):
        label = " / ".join(str(k) for k in key) if key else "all"
        col = colors[str(key[-1]) if key else ""]
        ax.step(g.min_reads, g[ycol], where="post", color=col, linewidth=1.6)
        floors: dict[str, float] = {}
        if knee and len(kn):
            sel = np.ones(len(kn), bool)
            for c, v in zip(gcols, key, strict=True):
                sel &= (kn[c] == v).to_numpy()
            if sel.any() and np.isfinite(float(kn.loc[sel, "knee_reads"].iloc[0])):
                for name, mk in (("knee", "o"), ("inflection", "D")):
                    kx = float(kn.loc[sel, f"{name}_reads"].iloc[0])
                    if not np.isfinite(kx):
                        continue
                    yy = g[ycol].to_numpy()[
                        np.searchsorted(g.min_reads.to_numpy(), kx, side="left").clip(0, len(g) - 1)
                    ]
                    ax.plot(
                        [kx],
                        [yy],
                        mk,
                        ms=7,
                        mfc=col if name == "knee" else CHROME["light"]["bg"],
                        mec=col,
                        mew=1.5,
                        zorder=3,
                    )
                    floors[name] = kx
        floors_all.append(floors)
        text = ", ".join(f"{n} {v:g}" for n, v in floors.items())
        handles.append(Line2D([], [], color=col, lw=1.6, label=f"{label}: {text}" if text else label))
    marked = [n for n in ("knee", "inflection") if any(n in f for f in floors_all)]
    # marker key; with a single curve it carries that curve's floors (the curve itself needs no legend entry)
    one = floors_all[0] if len(groups) == 1 else {}
    key_handles = [
        Line2D([], [], ls="", marker="o", ms=6, mfc=_INK2, mec=_INK2, label="knee"),
        Line2D([], [], ls="", marker="D", ms=6, mfc=CHROME["light"]["bg"], mec=_INK2, mew=1.5, label="inflection"),
    ]
    for h in key_handles:
        name = h.get_label()
        if name in one:
            h.set_label(f"{name}: {one[name]:g} reads")
    shown = (handles if len(groups) > 1 else []) + [
        h for h, n in zip(key_handles, ("knee", "inflection"), strict=True) if n in marked
    ]
    ax.set_xscale("log")
    if not share:
        ax.set_yscale("log")
    ax.set_xlabel("read floor (reads per record)")
    ax.set_ylabel("share of records kept" if share else "records kept")
    if shown:
        leg = ax.legend(handles=shown, frameon=False, fontsize=8, labelcolor=_INK2, loc="best")
        leg.set_zorder(5)
    _style(ax)
    return ax


def valid_shares(data: Any, *, ax=None, by: str = S.LOCUS):
    """Share of records with a valid junction and share productive, per chain.

    Parameters
    ----------
    data
        A receptor table, or the ``table`` of :func:`spaclone.pp.qc_metrics` grouped by chain.
    ax
        Axes to draw into.
    by
        Chain column.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from spaclone.pp.metrics import QCMetrics, qc_metrics

    if isinstance(data, QCMetrics):
        t = data.table
    elif isinstance(data, pd.DataFrame) and "valid_share" in data.columns:
        t = data
    else:
        t = qc_metrics(data, by=[by]).table
    if by in t.columns and t[by].duplicated().any():  # several libraries: record-weighted shares per chain
        w = t["n_records"].to_numpy(dtype=float)
        agg = {"n_records": t.groupby(by, sort=True)["n_records"].sum()}
        for col in ("valid_share", "productive_share"):
            if col in t.columns:
                num = pd.Series(t[col].to_numpy(dtype=float) * w).groupby(t[by].to_numpy()).sum(min_count=1)
                agg[col] = num / agg["n_records"].to_numpy(dtype=float)
        t = pd.DataFrame(agg).rename_axis(by).reset_index()
    ax = _ax(ax)
    labels = t[by].astype(str).tolist() if by in t.columns else ["all"]
    y = np.arange(len(labels))
    h = 0.36
    series = [
        ("valid junction", "valid_share", SLOTS["light"][0]),
        ("productive", "productive_share", SLOTS["light"][1]),
    ]
    shown = [s for s in series if s[1] in t.columns and t[s[1]].notna().any()]
    for k, (name, col, color) in enumerate(shown):
        off = (k - (len(shown) - 1) / 2) * (h + 0.04)
        v = t[col].to_numpy(dtype=float)
        ax.barh(y + off, v, height=h, color=color, label=name, edgecolor=CHROME["light"]["bg"], linewidth=1)
        for yy, vv in zip(y + off, v, strict=True):
            if np.isfinite(vv):
                ax.text(min(vv, 1.0) + 0.01, yy, f"{100 * vv:.1f}%", va="center", fontsize=8, color=_INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of records")
    if len(shown) > 1:  # above the bars, so it never covers a bar or its value
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(shown),
                  labelcolor=_INK2, borderaxespad=0.2)  # fmt: skip
    _style(ax, grid_axis="x")
    return ax


def clones_per_spot(
    data: Any,
    *,
    sink_min_clones: int | None = 20,
    ax=None,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
):
    """Histogram of distinct clones per spot, with the barcode-sink threshold.

    Parameters
    ----------
    data
        A receptor table, or the ``spots`` table of :func:`spaclone.pp.qc_metrics` (column ``n_clones``).
    sink_min_clones
        Draw the sink threshold (spots with at least this many clones); ``None`` hides it.
    ax
        Axes to draw into.
    spot_key, clone_key
        Columns of the table.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from spaclone.pp.metrics import QCMetrics, qc_metrics

    if isinstance(data, QCMetrics):
        sp = data.spots
    elif isinstance(data, pd.DataFrame) and "n_clones" in data.columns and clone_key not in data.columns:
        sp = data
    else:
        sp = qc_metrics(data, by=(), spot_key=spot_key, clone_key=clone_key, sink_min_clones=None).spots
    k = sp["n_clones"].to_numpy(dtype=np.int64)
    ax = _ax(ax)
    if len(k):
        vals, cnt = np.unique(k, return_counts=True)
        ax.bar(vals, cnt, width=0.9, color=SLOTS["light"][0], edgecolor=CHROME["light"]["bg"], linewidth=0.5)
        ax.set_yscale("log")
    if sink_min_clones is not None:
        ax.axvline(sink_min_clones - 0.5, color=_INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.annotate(
            f"sinks: >= {sink_min_clones} clones",
            (sink_min_clones - 0.5, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=_INK2,
        )
    ax.set_xlabel("distinct clones per spot")
    ax.set_ylabel("spots")
    _style(ax)
    return ax


def spot_qc_map(
    data: Any,
    *,
    color: str = "n_records",
    ax=None,
    s: float | None = None,
    log: bool = True,
    cmap: Any = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
):
    """Map of a per-spot QC metric (records, reads, clones, valid share...) at the spots' positions in µm.

    Parameters
    ----------
    data
        A receptor table, or the ``spots`` table of :func:`spaclone.pp.qc_metrics`.
    color
        Spot column to colour by (``n_records``, ``n_reads``, ``n_clones``, ``n_loci``, ``valid_share``, ``is_sink``).
    ax
        Axes to draw into.
    s
        Marker size (points^2); default scales with the number of spots.
    log
        Log-scale the colour of count columns.
    cmap
        Matplotlib colormap; default a single-hue blue ramp (light = low).
    invert_y
        Draw y increasing downwards (image convention, the default of every spaclone map); ``False`` puts y = 0 at
        the bottom.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster) that hold 95% of the
        spots, with any piece next to them; specks far from the section (e.g. a stray cluster of records on the glass)
        are left out of the view and counted in an INFO log message. ``False`` frames every spot.
    x_key, y_key
        Coordinate columns (µm).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.colors as mcolors

    from spaclone.pp.metrics import QCMetrics, qc_metrics

    if isinstance(data, QCMetrics):
        sp = data.spots
    elif isinstance(data, pd.DataFrame) and "n_records" in data.columns and color in data.columns:
        sp = data
    else:
        sp = qc_metrics(data, by=()).spots
    if color not in sp.columns:
        raise KeyError(f"spot_qc_map: column {color!r} not in the spot table")
    ax = _ax(ax)
    v = sp[color].to_numpy(dtype=float)
    order = np.argsort(v, kind="stable")  # high values drawn last
    cm = cmap or mcolors.LinearSegmentedColormap.from_list("spaclone_blue", _BLUE_RAMP)
    norm = None
    if log and color.startswith("n_") and np.nanmax(v) > 0:
        norm = mcolors.LogNorm(vmin=max(np.nanmin(v[v > 0]), 1e-9), vmax=np.nanmax(v))
    size = s if s is not None else float(np.clip(20000.0 / max(len(sp), 1), 0.3, 12))
    fr = SectionFrame(sp[[x_key, y_key]].to_numpy(dtype=float), rotate, crop=crop)
    attach(ax, fr)
    pxy = fr(sp[[x_key, y_key]].to_numpy(dtype=float))
    sc = ax.scatter(
        pxy[order, 0],
        pxy[order, 1],
        c=v[order],
        s=size,
        cmap=cm,
        norm=norm,
        linewidths=0,
        rasterized=len(sp) > 5000,
    )
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(color, color=_INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=_INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel(fr.labels[0])
    ax.set_ylabel(fr.labels[1])
    fr.set_view(ax, invert_y=invert_y)
    _style(ax, grid_axis="")
    return ax


def qc_panel(
    air: pd.DataFrame,
    *,
    color: str = "n_records",
    figsize: tuple[float, float] = (12.0, 8.5),
    invert_y: bool = True,
    rotate: bool | float = True,
):
    """Four QC panels of a receptor table: read support, valid / productive shares, clones per spot, spot map.

    Parameters
    ----------
    air
        Receptor table.
    color
        Spot metric of the map panel.
    figsize
        Figure size in inches.
    invert_y, rotate
        Orientation of the spot map (see :func:`spot_qc_map`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from spaclone.pp.metrics import qc_metrics

    m = qc_metrics(air, by=[S.LOCUS] if S.LOCUS in air.columns else ())
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    if S.READS in air.columns:
        read_support(air, ax=axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, "no read counts", ha="center", va="center", color=_MUTED)
        axes[0, 0].set_axis_off()
    valid_shares(m, ax=axes[0, 1])
    clones_per_spot(m, ax=axes[1, 0])
    spot_qc_map(m, color=color, ax=axes[1, 1], invert_y=invert_y, rotate=rotate)
    return fig
