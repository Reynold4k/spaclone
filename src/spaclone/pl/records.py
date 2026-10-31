"""Map of receptor records on the section (BCR blue, TCR pink; optional dark house style)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from spaclone import _schema as S
from spaclone.io._container import SpatialAIR
from spaclone.pl._colors import (
    CHROME,
    LOCUS_SLOT,
    NEUTRAL,
    RECEPTOR_COLORS,
    SLOTS,
    TRUTH_SLOT,
    fixed_colors,
    vocabulary,
)
from spaclone.pl._frame import SectionFrame, attach

__all__ = ["spatial_records"]

#: Colours of the two styles (from :mod:`spaclone.pl._colors`): BCR / TCR pairs pass the validator's colour-vision
#: and contrast checks on their surface; ``categorical`` is the 8-slot order used for any other column.
STYLES: dict[str, dict[str, Any]] = {
    mode: {
        "face": CHROME[mode]["bg"],
        "ink": CHROME[mode]["ink"],
        "muted": CHROME[mode]["muted"],
        "tissue": CHROME[mode]["tissue_strong"],
        "glass": CHROME[mode]["glass"],
        "other": NEUTRAL[mode],
        "BCR": RECEPTOR_COLORS[mode]["BCR"],
        "TCR": RECEPTOR_COLORS[mode]["TCR"],
        "categorical": list(SLOTS[mode]),
    }
    for mode in ("light", "dark")
}
#: Fixed category orders, so that a colour always means the same value.
FIXED_ORDER = {
    S.LOCUS: list(S.ALL_LOCI),
    "truth": ["genuine", "swapped", "sink"],
}
#: Fixed slot of every value of the fixed-order columns (a value keeps its colour when others are absent).
_FIXED_SLOT = {S.LOCUS: LOCUS_SLOT, "truth": TRUTH_SLOT}
_MISSING = {"nan", "None", "<NA>", "NaN"}


def _nice_length(span: float) -> float:
    """A round scale-bar length near a fifth of ``span`` (1, 2 or 5 × a power of ten)."""
    target = max(span / 5, 1e-9)
    p = 10 ** np.floor(np.log10(target))
    return float(min((1, 2, 5, 10), key=lambda m: abs(m * p - target)) * p)


def spatial_records(
    air: pd.DataFrame | SpatialAIR,
    gex: ad.AnnData | None = None,
    *,
    color: str | None = "receptor",
    palette: Mapping[str, str] | None = None,
    highlight: Sequence[Any] | None = None,
    background: Literal["gex", "none"] | pd.DataFrame | np.ndarray = "gex",
    style: Literal["light", "dark"] = "dark",
    size: float | None = None,
    alpha: float = 0.85,
    max_categories: int = 8,
    legend: bool = True,
    scale_bar_um: float | None = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
) -> Axes:
    """Draw receptor records at their positions, optionally over the spots of the section.

    Parameters
    ----------
    air
        Receptor table, or a :class:`~spaclone.io.SpatialAIR` (its ``gex`` is then the default background).
    gex
        Spot-level AnnData whose ``obsm["spatial"]`` (µm) is drawn as the background; ``obs["in_tissue"]``, when
        present, separates tissue from glass.
    color
        ``"receptor"`` (BCR blue, TCR pink, from ``locus_key``), the name of a column (categorical), or ``None``
        (one colour). Colours follow the value, never the values present: ``locus`` (IGH blue, IGK orange, IGL aqua,
        TRA yellow, TRB magenta, TRG green, TRD violet) and ``truth`` (genuine blue, swapped orange, sink aqua) have
        fixed colours; any other column takes the categorical slots in the order of its declared categories
        (categorical dtype: subsets keep the categories, so a subset keeps the colours of the whole table) or of the
        sorted values of the table passed (give a categorical column or ``palette`` to fix colours across tables).
        With more categories than ``max_categories``, the most frequent ``max_categories - 1`` keep their colours and
        the rest are pooled as "other" (grey); a column with more than eight values cannot give every value a fixed
        colour, so its kept values take the slots in category order. Missing values are drawn grey.
    palette
        ``{value: colour}`` overrides for a categorical ``color`` column.
    highlight
        Values of the ``color`` column to draw in colour; the other records are drawn grey underneath. The highlighted
        values take the colours before the ``max_categories`` cut, so a rare value is never pooled into "other"
        because unhighlighted values are more frequent; only when more values are highlighted than
        ``max_categories`` are the least frequent of them pooled as "other".
    background
        ``"gex"`` (the spots of ``gex``), ``"none"``, or positions in µm (``DataFrame`` with ``x_key`` / ``y_key`` or
        an ``(n, 2)`` array).
    style
        ``"dark"`` (the black house style, default) or ``"light"`` (white).
    size
        Marker area in points² (default: smaller for more records; with ``highlight``, the highlighted records are
        sized by their own number, never smaller than the grey ones, so a few of them stay visible over many).
    alpha
        Marker opacity.
    max_categories
        Most colours used for a categorical column (at most 8).
    legend
        Draw a legend when two or more categories are shown.
    scale_bar_um
        Length of the scale bar in µm (default: a round length near a fifth of the width; ``0`` for none). Axes
        ticks are hidden; coordinates are micrometres.
    invert_y
        Draw y downwards, as in tissue images (rows = y).
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the background, or of
        the records) that hold 95% of the points, with any piece next to them; specks far from the section (e.g. a stray
        cluster of records on the glass) are left out of the view and counted in an INFO log message. ``False`` frames
        every point.
    title
        Axes title.
    ax
        Axes to draw on (default: a new figure 7 inches wide, as tall as the section needs).
    x_key, y_key, locus_key
        Column names.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes (the figure is neither shown nor saved).

    Examples
    --------
    >>> from spaclone.datasets import synthetic_example
    >>> ax = spatial_records(synthetic_example(), color="truth", style="dark")  # doctest: +SKIP
    """
    if isinstance(air, SpatialAIR):
        gex = air.gex if gex is None else gex
        air = air.air
    if style not in STYLES:
        raise ValueError(f"style must be 'light' or 'dark', got {style!r}")
    for c in (x_key, y_key):
        if c not in air.columns:
            raise KeyError(f"spatial_records: column {c!r} not in the receptor table")
    pal = STYLES[style]
    xy = air[[x_key, y_key]].to_numpy(dtype=float)
    n = len(xy)
    s = size if size is not None else float(np.clip(4000 / max(n, 1), 0.5, 12))

    # background spots
    bg = None
    if isinstance(background, str):
        if background not in ("gex", "none"):
            raise ValueError("background must be 'gex', 'none' or positions")
        if background == "gex" and gex is not None and "spatial" in gex.obsm:
            bg = np.asarray(gex.obsm["spatial"], dtype=float)[:, :2]
            tissue = gex.obs["in_tissue"].to_numpy(dtype=bool) if "in_tissue" in gex.obs.columns else None
    else:
        bg = (
            background[[x_key, y_key]].to_numpy(dtype=float)
            if isinstance(background, pd.DataFrame)
            else np.asarray(background, dtype=float)
        )
        tissue = None
    frame = SectionFrame(bg if bg is not None and len(bg) else xy, rotate, crop=crop)
    if ax is None:  # as tall as the section needs, with room for the legend and the scale bar
        _, ax = plt.subplots(figsize=frame.figsize(7.0, extra_height=0.9, lo=2.5, hi=9.0))
    fig = ax.figure
    fig.set_facecolor(pal["face"])
    ax.set_facecolor(pal["face"])
    attach(ax, frame)
    xy = frame(xy)
    if bg is not None and len(bg):
        bg = frame(bg)
        bs = float(np.clip(20000 / len(bg), 0.2, 6))
        if tissue is not None:
            ax.scatter(*bg[~tissue].T, s=bs, c=pal["glass"], lw=0, marker="s", rasterized=True, zorder=0)
            ax.scatter(*bg[tissue].T, s=bs, c=pal["tissue"], lw=0, marker="s", rasterized=True, zorder=0)
        else:
            ax.scatter(*bg.T, s=bs, c=pal["tissue"], lw=0, marker="s", rasterized=True, zorder=0)

    # categories
    hl = None if highlight is None else {str(h) for h in highlight}
    if color is None:
        labels = np.full(n, "records", dtype=object)
        order, colours = ["records"], {"records": pal["BCR"]}
        raw = labels
    elif color == "receptor":
        loc = air[locus_key].astype(str).str.upper().to_numpy(dtype=object)
        labels = np.where(np.isin(loc, S.BCR_LOCI), "BCR", np.where(np.isin(loc, S.TCR_LOCI), "TCR", "other"))
        order = [c for c in ("BCR", "TCR", "other") if (labels == c).any()]
        colours = {"BCR": pal["BCR"], "TCR": pal["TCR"], "other": pal["other"]}
        raw = labels
    else:
        if color not in air.columns:
            raise KeyError(f"spatial_records: column {color!r} not in the receptor table")
        col = air[color]
        vals = col.astype(str).to_numpy(dtype=object)
        vals = np.where(col.isna().to_numpy(), "NA", vals)
        raw = vals
        k = max(1, min(int(max_categories), 8))
        vocab = [v for v in vocabulary(col, order=FIXED_ORDER.get(color)) if v not in _MISSING]
        present = pd.Series(vals[vals != "NA"]).value_counts()
        ranked = [v for v in vocab if v in present.index]
        if hl is not None:  # highlighted values take the colours first: the others are drawn grey anyway
            ranked = [v for v in ranked if v in hl]
        if len(ranked) <= k:
            keep = ranked
        else:  # most frequent k - 1 (ties: category order), drawn in category order
            pos = {v: i for i, v in enumerate(vocab)}
            top = set(sorted(ranked, key=lambda v: (-present[v], pos[v]))[: k - 1])
            keep = [v for v in ranked if v in top]
        mode = "dark" if style == "dark" else "light"
        slot_of = _FIXED_SLOT.get(color, {})
        named = {v: SLOTS[mode][slot_of[v]] for v in vocab if v in slot_of}
        colours = fixed_colors(vocab if len(vocab) <= 8 else keep, mode=mode, named=named, palette=palette)
        colours |= {"other": pal["other"], "NA": pal["other"]}
        labels = np.where(np.isin(vals, keep) | (vals == "NA"), vals, "other")
        order = list(keep) + [c for c in ("other", "NA") if (labels == c).any()]
    shown = np.ones(n, dtype=bool)  # records drawn in their category colour
    s_shown = s
    if hl is not None:
        shown = np.isin(raw, list(hl))
        ax.scatter(*xy[~shown].T, s=s, c=pal["other"], lw=0, alpha=0.5, rasterized=n > 5000, zorder=1)
        order = [c for c in order if (shown & (labels == c)).any()]
        if size is None:  # a few highlighted records stay visible over many grey ones
            s_shown = max(s, float(np.clip(4000 / max(int(shown.sum()), 1), 0.5, 12)))
    handles = []
    counts = {cat: int((shown & (labels == cat)).sum()) for cat in order}
    stack = sorted(order, key=lambda c: -counts[c])  # large groups underneath, small ones stay visible
    for cat in order:
        m = shown & (labels == cat)
        z = 2 + stack.index(cat)
        ax.scatter(*xy[m].T, s=s_shown, c=colours[cat], lw=0, alpha=alpha, rasterized=n > 5000, zorder=z)
        handles.append(
            Line2D([], [], ls="", marker="o", ms=7, mfc=colours[cat], mec="none", label=f"{cat} ({counts[cat]:,})")
        )
    if legend and len(handles) >= 2:  # under the map, so it never covers the tissue
        rows = -(-len(handles) // 6)  # at most six entries per row, the rows balanced (8 entries: 4 + 4)
        lg = ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=-(-len(handles) // rows),
            fontsize=9.5, frameon=False, handletextpad=0.3, columnspacing=1.2,
        )  # fmt: skip
        lg.set_zorder(20)
        for t in lg.get_texts():
            t.set_color(pal["ink"])

    # frame: equal aspect, no ticks, scale bar
    ext = frame.extent
    if ext is not None:  # the main tissue: a detached speck far from the section does not shrink the map
        lo, hi = np.array([ext[0], ext[2]]), np.array([ext[1], ext[3]])
        pad = 0.03 * max(float((hi - lo).max()), 1.0)
        frame.set_view(ax, invert_y=invert_y, pad_um=pad)
        span = float(hi[0] - lo[0]) or 1.0
        bar = _nice_length(span) if scale_bar_um is None else float(scale_bar_um)
        if bar > 0:
            x0 = lo[0]
            yb = (hi[1] + 0.5 * pad) if invert_y else (lo[1] - 0.5 * pad)
            ax.plot([x0, x0 + bar], [yb, yb], color=pal["ink"], lw=2, solid_capstyle="butt", zorder=10)
            label = f"{bar / 1000:g} mm" if bar >= 1000 else f"{bar:g} µm"
            ax.annotate(label, (x0 + bar / 2, yb), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", color=pal["ink"], fontsize=9)  # fmt: skip
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    if title:
        ax.set_title(title, color=pal["ink"], fontsize=11, loc="left")
    return ax
