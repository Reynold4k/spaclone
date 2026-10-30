"""Clone maps (minimum-spanning-tree links between the units of a clone) and clonal-scan results.

Clone maps follow the conventions of every spaclone map: y increases downwards (image convention) unless
``invert_y=False``, and colours come from the validated categorical slots of :mod:`spaclone.pl._colors`, given to the
clones in the order they are named (``clones=``) or ranked (largest first), never cycled: clones past the eighth slot
are grey. Only the first three slots stay apart for every pair, so a map of more than three clones should be read with
its legend; ``palette`` fixes a clone's colour across figures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError

from spaclone import _schema as S
from spaclone._utils import TestResult
from spaclone.pl._colors import CHROME, NEUTRAL, SLOTS, fixed_colors
from spaclone.pl._frame import SectionFrame, attach

__all__ = ["clonal_scan", "clone_map", "mst_edges"]

#: Colours of the two house styles (from :mod:`spaclone.pl._colors`; validated pair: blue / orange on the background).
STYLES: dict[str, dict[str, Any]] = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "fg": CHROME[mode]["ink"],
        "muted": NEUTRAL[mode],
        "grid": CHROME[mode]["axis"],
        "accent": SLOTS[mode][0],
        "accent2": SLOTS[mode][1],
        "palette": list(SLOTS[mode]),
    }
    for mode in ("light", "dark")
}


def _style(style: str) -> dict[str, Any]:
    if style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}, got {style!r}")
    return STYLES[style]


def _apply_style(ax: Axes, st: dict[str, Any]) -> None:
    ax.set_facecolor(st["bg"])
    ax.figure.set_facecolor(st["bg"])
    for sp in ax.spines.values():
        sp.set_color(st["grid"])
    ax.tick_params(colors=st["fg"], which="both")
    ax.xaxis.label.set_color(st["fg"])
    ax.yaxis.label.set_color(st["fg"])
    ax.title.set_color(st["fg"])


def mst_edges(xy: np.ndarray) -> np.ndarray:
    """Edges ``(i, j)`` of the Euclidean minimum spanning tree of points (Delaunay graph; exact for distinct points).

    Duplicated points are joined at length 0. Returns an ``(n - 1, 2)`` integer array for ``n >= 2`` connected points.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.full(len(uniq), -1)
    first[inv[::-1]] = np.arange(n)[::-1]
    dup = [(first[inv[i]], i) for i in range(n) if first[inv[i]] != i]
    edges = []
    m = len(uniq)
    if m >= 2:
        if m <= 3:
            ii, jj = np.triu_indices(m, 1)
        else:
            try:
                tri = Delaunay(uniq)
                s = tri.simplices
                pairs = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
                pairs = np.unique(np.sort(pairs, axis=1), axis=0)
                ii, jj = pairs[:, 0], pairs[:, 1]
            except QhullError:  # collinear points
                order = np.lexsort((uniq[:, 1], uniq[:, 0]))
                ii, jj = order[:-1], order[1:]
        w = np.hypot(*(uniq[ii] - uniq[jj]).T) + 1e-12
        mst = minimum_spanning_tree(coo_matrix((w, (ii, jj)), shape=(m, m))).tocoo()
        edges = [(first[a], first[b]) for a, b in zip(mst.row, mst.col, strict=True)]
    return np.asarray(edges + dup, dtype=np.int64).reshape(-1, 2)


def clone_map(
    air: pd.DataFrame,
    clones: str | Sequence[str] | None = None,
    *,
    n_top: int = 3,
    loci: str | Sequence[str] | None = "heavy",
    ax: Axes | None = None,
    links: bool = True,
    max_link_um: float | None = None,
    background: bool = True,
    style: Literal["light", "dark"] = "dark",
    size: float = 6.0,
    background_size: float = 1.0,
    legend: bool = True,
    scalebar_um: float | None = None,
    palette: Mapping[str, str] | None = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    locus_key: str = S.LOCUS,
) -> Axes:
    """Map of clones: their units coloured and linked by the clone's minimum spanning tree, over all other units.

    Parameters
    ----------
    air
        Receptor table (records or loci) with positions in µm and clone ids.
    clones
        Clone id(s) to draw, in the order that assigns their colours; ``None``: the ``n_top`` clones with most
        distinct spots, largest first.
    n_top
        Number of clones drawn when ``clones`` is None (default 3: the slots that stay apart for every pair).
    loci
        Chains whose clones are ranked when ``clones`` is None: ``"heavy"`` (IGH, TRB, TRD; the default, since a light
        chain is shared by many B-cell clones), a locus or a list of loci, or None for every chain. Tables without a
        locus column, or without any of these chains, rank every clone.
    ax
        Axes to draw into (a new figure 8 inches wide and as tall as the section needs otherwise).
    links
        Draw the minimum-spanning-tree links between the distinct positions of each clone.
    max_link_um
        Links longer than this are not drawn (the tree is still computed on all units).
    background
        Draw every unit of the table in grey underneath.
    style
        ``"dark"`` (black, the house style; default) or ``"light"`` (white).
    size, background_size
        Marker sizes of the clones and of the background.
    legend
        Show a legend with each clone and its number of distinct spots (``"clone 45 · 56 spots"``), above the map.
    scalebar_um
        Length of a scale bar in µm (none if ``None``); with a scale bar the axes are hidden.
    palette
        ``{clone_id: colour}`` overrides, to keep a clone's colour across figures.
    invert_y
        Draw y increasing downwards (image convention, the default of every spaclone map); ``False`` puts y = 0 at
        the bottom.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the table's
        positions) that hold 95% of the points, with any piece next to them; specks far from the section (e.g. a stray
        cluster of records on the glass) are left out of the view and counted in an INFO log message. ``False`` frames
        every point.
    x_key, y_key, clone_key, spot_key, locus_key
        Column names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    st = _style(style)
    d = air[air[clone_key].notna()]
    if clones is None:
        pool = d
        if loci is not None and locus_key in d.columns:
            want = set(S.HEAVY_LOCI) if loci == "heavy" else ({loci} if isinstance(loci, str) else set(loci))
            sel = d[locus_key].astype(str).str.upper().isin({str(v).upper() for v in want}).to_numpy()
            pool = d[sel] if sel.any() else d
        key = [spot_key, clone_key] if spot_key in pool.columns else [x_key, y_key, clone_key]
        top = pool.drop_duplicates(key)[clone_key].value_counts()
        chosen = list(top.index[:n_top])
    else:
        chosen = [clones] if isinstance(clones, str) else list(clones)
    fr = SectionFrame(air[[x_key, y_key]].to_numpy(dtype=float), rotate, crop=crop)
    if ax is None:  # as tall as the section needs, with room for the legend above it and the axis labels below
        top_in, bottom_in = 0.3 + (0.2 * len(chosen) if legend else 0.0), 0.65
        hgt = float(np.clip(8.0 * 0.775 * fr.aspect() * 1.03 + top_in + bottom_in, 3.0, 12.0))
        fig, ax = plt.subplots(figsize=(8.0, hgt))
        fig.subplots_adjust(left=0.125, right=0.9, bottom=bottom_in / hgt, top=1 - top_in / hgt)
    _apply_style(ax, st)
    attach(ax, fr)
    if background:
        bxy = fr(air[[x_key, y_key]].to_numpy(dtype=float))
        ax.scatter(
            bxy[:, 0], bxy[:, 1], s=background_size, c=st["muted"], linewidths=0, alpha=0.5, zorder=1, rasterized=True
        )
    names = [str(c) for c in chosen]
    colours = fixed_colors(names, mode="dark" if style == "dark" else "light", palette=palette, beyond=st["muted"])
    for c in chosen:
        g = d[d[clone_key] == c]
        if len(g) == 0:
            continue
        xy = fr(g[[x_key, y_key]].drop_duplicates().to_numpy(dtype=float))
        col = colours[str(c)]
        if links and len(xy) > 1:
            e = mst_edges(xy)
            seg = np.stack([xy[e[:, 0]], xy[e[:, 1]]], axis=1)
            if max_link_um is not None:
                seg = seg[np.hypot(*(seg[:, 0] - seg[:, 1]).T) <= max_link_um]
            ax.add_collection(LineCollection(seg, colors=col, linewidths=0.8, alpha=0.8, zorder=2))
        name = f"clone {c}" if str(c)[:1].isdigit() else str(c)
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=col, linewidths=0, zorder=3, label=f"{name} · {len(xy):,} spots")
    ax.set_xlabel(fr.labels[0])
    ax.set_ylabel(fr.labels[1])
    fr.set_view(ax, invert_y=invert_y)
    if scalebar_um:  # lower left of the display, whichever way y runs (get_ylim gives bottom, top)
        ax.set_xticks([])  # the scale bar replaces the axes, as on every spaclone tissue map
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bx, by = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([bx, bx + scalebar_um], [by, by], color=st["fg"], lw=2, zorder=4)
        ax.text(
            bx + scalebar_um / 2,
            by + 0.015 * (y1 - y0),
            f"{scalebar_um:g} µm",
            color=st["fg"],
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if legend and chosen:  # above the map, so it never covers the tissue
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=8, frameon=False, markerscale=2,
                        borderaxespad=0.2)  # fmt: skip
        for t in leg.get_texts():
            t.set_color(st["fg"])
    return ax


def clonal_scan(
    result: TestResult,
    *,
    kind: Literal["size", "qq"] = "size",
    ax: Axes | None = None,
    style: Literal["light", "dark"] = "dark",
    label_top: int = 5,
) -> Axes:
    """Plot a :func:`spaclone.tl.clonal_scan` result.

    Parameters
    ----------
    result
        Result of :func:`spaclone.tl.clonal_scan`.
    kind
        ``"size"``: -log10 p against clone size (its cell-scale blocks when the scan used the block null, else its
        units), significant clones highlighted, with the BH threshold of the largest p-value called (or of the first
        rank when none is called) and the Monte-Carlo floor. ``"qq"``: observed against uniform -log10 p (tested
        clones), with the floor.
    ax
        Axes to draw into (a new figure otherwise).
    style
        ``"dark"`` (house style, default) or ``"light"``.
    label_top
        Label this many clones with the smallest p-values (``"size"``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    st = _style(style)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    _apply_style(ax, st)
    t = result.table
    if "testable" in t.columns:
        t = t[t["testable"]]
    alpha = float(result.params.get("alpha", 0.05))
    if len(t) == 0:
        ax.text(0.5, 0.5, "no clone tested", transform=ax.transAxes, ha="center", color=st["fg"])
        return ax
    lp = -np.log10(t["p_value"].to_numpy(dtype=float))
    sig = t["significant"].to_numpy(dtype=bool)
    floor = None
    if "n_perm" in t.columns and len(t):
        h = result.params.get("h", 1)
        num = 1.0 if result.params.get("stopping") == "besag_clifford" else float(h)
        floor = -np.log10(num / (t["n_drawn"].max() + 1)) if "n_drawn" in t.columns else None
    if kind == "size":
        blocks = "n_blocks" in t.columns and bool(np.isfinite(t["n_blocks"].to_numpy(dtype=float)).all())
        n = t["n_blocks" if blocks else "n_units"].to_numpy(dtype=float)
        if (~sig).any():  # legend entries only for groups that are drawn
            ax.scatter(n[~sig], lp[~sig], s=12, c=st["muted"], linewidths=0, label="not called")
        if sig.any():
            ax.scatter(n[sig], lp[sig], s=22, c=st["accent2"], linewidths=0, label=f"q <= {alpha:g}")
        m = len(t)
        thr = t.loc[sig, "p_value"].max() if sig.any() else alpha / m
        ax.axhline(-np.log10(thr), color=st["accent"], lw=1, ls="--", label="BH threshold")
        if floor is not None:
            ax.axhline(floor, color=st["grid"], lw=1, ls=":", label="Monte-Carlo floor")
        ax.set_xscale("log")
        ax.set_xlabel("cell-scale blocks of the clone" if blocks else "units of the clone")
        ax.set_ylabel("-log10 p")
        top = np.argsort(t["p_value"].to_numpy(dtype=float))[:label_top]
        for i in top:
            ax.annotate(
                str(t["clone_id"].iloc[i])[-24:],
                (n[i], lp[i]),
                fontsize=7,
                color=st["fg"],
                xytext=(3, 3),
                textcoords="offset points",
            )
    elif kind == "qq":
        o = np.sort(lp)[::-1]
        e = -np.log10(np.arange(1, len(o) + 1) / (len(o) + 1))
        ax.scatter(e, o, s=12, c=st["accent"], linewidths=0)
        lim = max(e.max(), o.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=st["muted"], lw=1)
        if floor is not None:
            ax.axhline(floor, color=st["grid"], lw=1, ls=":", label="Monte-Carlo floor")
        ax.set_xlabel("expected -log10 p (uniform)")
        ax.set_ylabel("observed -log10 p")
    else:
        raise ValueError(f"kind must be 'size' or 'qq', got {kind!r}")
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(fontsize=8, frameon=False)
        for tx in leg.get_texts():
            tx.set_color(st["fg"])
    return ax
