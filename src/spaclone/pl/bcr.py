"""BCR plots: isotype maps, SHM and CSR share maps with hotspot outlines, CSR transition matrices, trees next to tissue.

Every function takes ``ax=None`` (a new figure is made) and returns the ``Axes`` (or the ``Figure`` for multi-panel
plots); none calls ``show()`` or ``savefig()``. ``style="dark"`` is the project house style (black background): CSR
maps use the blue ramp with yellow hotspot outlines, SHM maps a green ramp with vermilion hotspot outlines and
mutated units as vermilion dots. A hotspot result with a permutation-studentised map (clone-level nulls, the default)
is drawn as that map (``layer="z_perm"``, diverging colours) unless another ``layer`` is asked for. Every map follows
the image convention (y increases downwards) unless ``invert_y=False``; colours follow the value (isotypes and
house classes fixed, other values by their sorted position), never the values present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl._colors import (
    CHROME,
    ISOTYPE_CLASSES,
    ISOTYPE_COLORS,
    MARKS,
    RAMPS,
    SLOTS,
    fixed_colors,
    ink_on,
    mode_of,
    named_colors,
    vocabulary,
)
from spaclone.pl._colors import (
    cmap as house_cmap,
)
from spaclone.pl._frame import SectionFrame, attach, frame_of

__all__ = ["csr_share_map", "csr_transitions", "isotype_map", "shm_share_map", "tree_tissue"]

#: Isotype class colours (:data:`spaclone.pl._colors.ISOTYPE_COLORS`) on black (house style) and on white.
ISOTYPE_COLORS_DARK = {k: ISOTYPE_COLORS["dark"][k] for k in (*ISOTYPE_CLASSES, "unknown")}
ISOTYPE_COLORS_LIGHT = {k: ISOTYPE_COLORS["light"][k] for k in (*ISOTYPE_CLASSES, "unknown")}
_PAL = {
    mode: {"bg": CHROME[mode]["bg"], "fg": CHROME[mode]["ink"], "tissue": CHROME[mode]["tissue_strong"],
           "line": CHROME[mode]["ink2"]}
    for mode in ("dark", "light")
}  # fmt: skip


def _palette(style: str | None) -> dict[str, str]:
    if style not in (None, "light", "dark"):
        raise ValueError("style must be None, 'light' or 'dark'")
    return _PAL["dark" if style == "dark" else "light"]


def _axes(ax: Any, style: str | None, figsize: tuple[float, float]) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if style == "dark":
        pal = _palette(style)
        ax.figure.set_facecolor(pal["bg"])
        ax.set_facecolor(pal["bg"])
        ax.tick_params(colors=pal["fg"])
        for sp in ax.spines.values():
            sp.set_color(pal["fg"])
        ax.xaxis.label.set_color(pal["fg"])
        ax.yaxis.label.set_color(pal["fg"])
        ax.title.set_color(pal["fg"])
    return ax


def _colors_for(
    values: Sequence[Any], palette: Mapping[str, str] | None, style: str | None, vocab: Sequence[Any] | None = None
) -> dict[Any, str]:
    """Colours by value.

    Isotype classes and house classes (regions, cell types) are fixed; any other value takes a validated slot by its
    position in ``vocab`` (the column's full sorted list), never by the values drawn.
    """
    mode = mode_of(style)
    iso = ISOTYPE_COLORS_DARK if mode == "dark" else ISOTYPE_COLORS_LIGHT
    full = list(dict.fromkeys(str(v) for v in [*(vocab if vocab is not None else ()), *values]))
    named = {v: iso[v] for v in full if v in iso}
    named |= {v: c for v, c in named_colors(full, mode).items() if v not in named}  # regions, cell classes
    cols = fixed_colors(full, mode=mode, named=named, palette=palette, slots=SLOTS[mode])
    return {v: cols[str(v)] for v in values}


def isotype_map(
    units: pd.DataFrame,
    *,
    isotype_key: str = "isotype",
    x_key: str = S.X,
    y_key: str = S.Y,
    ax: Any = None,
    style: str | None = "dark",
    s: float = 4.0,
    palette: Mapping[str, str] | None = None,
    order: Sequence[str] = ("unknown", "IgM", "IgD", "IgE", "IgG", "IgA"),
    background: pd.DataFrame | None = None,
    legend: bool = True,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    title: str | None = None,
) -> Any:
    """Units (loci, spots, records) on the tissue, coloured by isotype class.

    Parameters
    ----------
    units
        Table with coordinates (µm) and an isotype column (e.g. loci with :func:`spaclone.bcr.call_isotype`).
    isotype_key, x_key, y_key
        Column names.
    ax
        Axes to draw on; a new figure if None.
    style
        None / ``"light"`` or ``"dark"`` (house style).
    s
        Marker size (points²).
    palette
        Colours by class, overriding the defaults (IgM light blue, IgD blue, IgG red, IgA orange, IgE purple).
    order
        Drawing order (later classes on top: switched units over unswitched ones by default); other values are
        drawn first.
    background
        Optional table of all tissue positions (``x_key`` / ``y_key``) drawn underneath in grey.
    legend
        Add a legend with the number of units per class.
    invert_y
        Draw y increasing downwards (image convention, the default of every spaclone map); ``False`` puts y = 0 at
        the bottom.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster) that hold 95% of the
        points, with any piece next to them; specks far from the section (e.g. a stray cluster of records on the glass)
        are left out of the view and counted in an INFO log message. ``False`` frames every point.
    title
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    for c in (isotype_key, x_key, y_key):
        if c not in units.columns:
            raise KeyError(f"isotype_map: column {c!r} not in the table")
    pal = _palette(style)
    ax = _axes(ax, style, (8.0, 6.0))
    uxy = units[[x_key, y_key]].to_numpy(dtype=float)
    bxy = background[[x_key, y_key]].to_numpy(dtype=float) if background is not None else None
    fr = SectionFrame(bxy if bxy is not None and len(bxy) else uxy, rotate, crop=crop)
    attach(ax, fr)
    uxy = fr(uxy)
    if bxy is not None:
        bxy = fr(bxy)
        ax.scatter(bxy[:, 0], bxy[:, 1], s=max(s / 4, 0.2), c=pal["tissue"], lw=0, rasterized=True, zorder=0)
    iso = units[isotype_key].astype("string").fillna("unknown").astype(str)
    present = list(pd.unique(iso))
    ranked = [v for v in present if v not in order] + [v for v in order if v in present]
    cols = _colors_for(ranked, palette, style, vocabulary(iso))
    for z, v in enumerate(ranked, start=1):
        m = (iso == v).to_numpy()
        ax.scatter(
            uxy[m, 0],
            uxy[m, 1],
            s=s,
            c=cols[v],
            lw=0,
            rasterized=True,
            zorder=z,
            label=f"{v} ({int(m.sum()):,})",
        )
    ax.set_xlabel(fr.labels[0])
    ax.set_ylabel(fr.labels[1])
    fr.set_view(ax, invert_y=invert_y)
    if title:
        ax.set_title(title, color=pal["fg"])
    if legend:
        leg = ax.legend(frameon=False, markerscale=3, fontsize=9, loc="best")
        for t in leg.get_texts():
            t.set_color(pal["fg"])
    return ax


def _unit_xy(ax: Any, units: pd.DataFrame, x_key: str, y_key: str) -> np.ndarray:
    """Unit positions in the orientation the map on ``ax`` was drawn with."""
    xy = units[[x_key, y_key]].to_numpy(dtype=float)
    fr = frame_of(ax)
    return fr(xy) if fr is not None else xy


def _share_panel(
    result: Any,
    *,
    ax: Any,
    style: str | None,
    ramp: list[str],
    hot_color: str,
    hot_name: str,
    other: Any,
    other_color: str,
    other_name: str,
    outlines: Mapping[str, Any] | None,
    outline_colors: Mapping[str, str] | None,
    **kwargs: Any,
) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    from spaclone import pl

    draw = getattr(pl, "share_map", None)
    if draw is None:  # pragma: no cover - depends on spaclone.pl.maps
        raise ImportError("spaclone.pl.share_map is not available in this installation")
    lines: dict[str, Any] = {}
    colors: dict[str, str] = {}
    labels = getattr(result, "labels", None)
    if labels is not None:
        tab = getattr(result, "table", None)
        hot = tab.hotspot[tab.direction == "hot"].to_numpy() if tab is not None and len(tab) else []
        lines[hot_name] = np.isin(labels.values, hot)
        colors[hot_name] = hot_color
    if other is not None:
        lab2 = getattr(other, "labels", other)
        vals = lab2.values if hasattr(lab2, "values") else np.asarray(lab2)
        lines[other_name] = np.asarray(vals) > 0
        colors[other_name] = other_color
    for k, v in (outlines or {}).items():
        lines[k] = v
        if outline_colors and k in outline_colors:
            colors[k] = outline_colors[k]
    from spaclone.pl.maps import _Z_LAYERS, _resolve_layer

    layer = _resolve_layer(result, kwargs.pop("layer", None))
    cmap = kwargs.pop("cmap", None)
    if cmap is None and layer not in _Z_LAYERS:  # standardised layers keep the diverging map of share_map
        cmap = LinearSegmentedColormap.from_list("spaclone_bcr", ramp)
    return draw(
        result, layer=layer, ax=ax, style=style, cmap=cmap, hotspots=False, outlines=lines, outline_colors=colors,
        **kwargs,
    )  # fmt: skip


def shm_share_map(
    result: Any,
    *,
    ax: Any = None,
    style: str | None = "dark",
    csr: Any = None,
    units: pd.DataFrame | None = None,
    mark: str = "mutated",
    x_key: str = S.X,
    y_key: str = S.Y,
    s: float = 6.0,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """SHM share map (green) with SHM hotspots outlined in vermilion and, optionally, CSR hotspots in yellow.

    Parameters
    ----------
    result
        :func:`spaclone.bcr.shm_hotspots` result (map and hotspots) or :func:`spaclone.bcr.shm_map` result.
    ax
        Axes; a new figure if None.
    style
        ``"dark"`` (default, house style) or None / ``"light"``.
    csr
        Optional :func:`spaclone.bcr.csr_hotspots` result (or label raster) whose hotspots are outlined in yellow.
    units
        Optional evaluable units drawn as dots: mutated in vermilion, others in grey.
    mark, x_key, y_key, s
        Mutated column, coordinates and dot size of ``units``.
    outlines, outline_colors
        Further boolean rasters to outline (e.g. T zones) and their colours.
    **kwargs
        Passed to :func:`spaclone.pl.share_map` (``layer``, ``vmin``, ``vmax``, ``colorbar``, ``invert_y``, ``rotate``,
        ``title``, ...). ``layer`` defaults to ``"z_perm"`` (the permutation-studentised map that was thresholded)
        when ``result`` carries one, else ``"share"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, style, (8.0, 6.0))
    ax = _share_panel(
        result,
        ax=ax,
        style=style,
        ramp=RAMPS["shm"][mode_of(style)],
        hot_color=MARKS[mode_of(style)]["shm"],
        hot_name="SHM hotspot",
        other=csr,
        other_color=MARKS[mode_of(style)]["csr"],
        other_name="CSR hotspot",
        outlines=outlines,
        outline_colors=outline_colors,
        **kwargs,
    )
    if units is not None:
        m = units[mark].fillna(False).astype(bool).to_numpy()
        uxy = _unit_xy(ax, units, x_key, y_key)
        ax.scatter(uxy[~m, 0], uxy[~m, 1], s=s * 0.3, c=MARKS[mode_of(style)]["nomut"], lw=0, zorder=7)
        ax.scatter(
            uxy[m, 0],
            uxy[m, 1],
            s=s,
            c=MARKS[mode_of(style)]["shm"],
            lw=0.4,
            edgecolors=CHROME["light"]["bg"],
            zorder=8,
        )
    return ax


def csr_share_map(
    result: Any,
    *,
    ax: Any = None,
    style: str | None = "dark",
    units: pd.DataFrame | None = None,
    isotype_key: str = "isotype",
    mark: str = "switched",
    x_key: str = S.X,
    y_key: str = S.Y,
    s: float = 4.0,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Switched-share map (blue) with CSR hotspots outlined in yellow; optionally switched units as dots.

    Parameters
    ----------
    result
        :func:`spaclone.bcr.csr_hotspots` (map and hotspots) or :func:`spaclone.bcr.csr_map` result.
    ax
        Axes; a new figure if None.
    style
        ``"dark"`` (default, house style) or None / ``"light"``.
    units
        Optional units: switched ones drawn as dots coloured by isotype (IgG red, IgA orange).
    isotype_key, mark, x_key, y_key, s
        Columns and dot size of ``units``.
    outlines, outline_colors
        Further boolean rasters to outline and their colours.
    **kwargs
        Passed to :func:`spaclone.pl.share_map` (``layer``, ``vmin``, ``vmax``, ``colorbar``, ``invert_y``, ``rotate``,
        ``title``, ...). ``layer`` defaults to ``"z_perm"`` (the permutation-studentised map that was thresholded)
        when ``result`` carries one, else ``"share"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, style, (8.0, 6.0))
    ax = _share_panel(
        result,
        ax=ax,
        style=style,
        ramp=RAMPS["csr"][mode_of(style)],
        hot_color=MARKS[mode_of(style)]["csr"],
        hot_name="CSR hotspot",
        other=None,
        other_color="",
        other_name="",
        outlines=outlines,
        outline_colors=outline_colors,
        **kwargs,
    )
    if units is not None:
        sw = units[mark].fillna(False).astype(bool).to_numpy()
        iso = units[isotype_key].astype("string").fillna("unknown").astype(str) if isotype_key in units else None
        cols = _colors_for(["IgG", "IgA", "IgE", "unknown"], None, style)
        c = [cols.get(v, cols["unknown"]) for v in (iso[sw] if iso is not None else ["unknown"] * int(sw.sum()))]
        uxy = _unit_xy(ax, units, x_key, y_key)
        ax.scatter(uxy[sw, 0], uxy[sw, 1], s=s, c=c, lw=0, zorder=7)
    return ax


def csr_transitions(
    result: Any,
    *,
    value: str = "ratio",
    ax: Any = None,
    style: str | None = "dark",
    cmap: Any = None,
    annotate: bool = True,
    alpha: float = 0.05,
) -> Any:
    """Upstream x downstream matrix of a :func:`spaclone.bcr.csr_transitions` result.

    Parameters
    ----------
    result
        The test result (its table).
    value
        Column shown as colour: ``"ratio"`` (observed / null mean; ``null="global"``), ``"statistic"``, ``"share"``,
        ``"n_clones_both"`` or ``"z"``.
    ax
        Axes; a new figure if None.
    style
        ``"dark"`` (house style, default) or None / ``"light"``.
    cmap
        Colormap (default: the house blue ramp; the house diverging map centred at 0 for ``"z"``).
    annotate
        Write the number of clones carrying both isotypes in each cell, with ``*`` where ``q <= alpha``.
    alpha
        FDR level of the stars.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.colors import to_hex

    tab = result.table if hasattr(result, "table") else result
    if value not in tab.columns:
        raise KeyError(f"csr_transitions: column {value!r} not in the result table")
    pal = _palette(style)
    names_up = tab.drop_duplicates("upstream").sort_values("upstream_position").upstream.tolist()
    names_dn = tab.drop_duplicates("downstream").sort_values("downstream_position").downstream.tolist()
    names = list(dict.fromkeys(names_up + names_dn))
    k = len(names)
    mat = np.full((k, k), np.nan)
    pos = {n: i for i, n in enumerate(names)}
    for r in tab.itertuples():
        mat[pos[r.upstream], pos[r.downstream]] = getattr(r, value)
    ax = _axes(ax, style, (1.0 + 0.7 * k, 0.8 + 0.6 * k))
    fin = mat[np.isfinite(mat)]
    if value == "z":
        lim = float(np.abs(fin).max()) if len(fin) else 1.0
        im = ax.imshow(mat, cmap=cmap or house_cmap("diverging", mode_of(style)), vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(mat, cmap=cmap or house_cmap("blue", mode_of(style)))
    if annotate:
        for r in tab.itertuples():
            star = "*" if np.isfinite(r.q_value) and r.q_value <= alpha else ""
            ax.text(
                pos[r.downstream],
                pos[r.upstream],
                f"{int(r.n_clones_both)}{star}",
                ha="center",
                va="center",
                fontsize=8,
                color=ink_on(to_hex(im.cmap(im.norm(mat[pos[r.upstream], pos[r.downstream]])))),
            )
    ax.set_xticks(range(k), names, rotation=45, ha="right")
    ax.set_yticks(range(k), names)
    ax.set_xlabel("downstream isotype (3')")
    ax.set_ylabel("upstream isotype (5')")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(value, color=pal["fg"])
    cb.ax.tick_params(colors=pal["fg"])
    return ax


def _tree_layout(tree: Any) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Coordinates of every node (x: mutations from the germline, y: tip row) and the tips in drawing order."""
    ch = tree.children()
    depth = tree.depths()
    root = tree.root
    size = np.zeros(len(tree.parent), dtype=np.int64)
    for v in tree._postorder(ch):
        size[v] = 1 if not ch[v] else size[ch[v]].sum()
    order: list[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        if not ch[v]:
            if tree.label[v] != "Germline":
                order.append(v)
            continue
        stack.extend(sorted(ch[v], key=lambda c: size[c]))  # larger subtree drawn first (popped last -> top)
    y = np.full(len(tree.parent), np.nan)
    for row, v in enumerate(order):
        y[v] = row
    for v in tree._postorder(ch):
        if ch[v]:
            ys = [y[c] for c in ch[v] if np.isfinite(y[c])]
            y[v] = float(np.mean(ys)) if ys else np.nan
    return np.nan_to_num(depth, nan=0.0), y, order


def tree_tissue(
    tree: Any,
    tips: pd.DataFrame,
    *,
    color_key: str = "isotype",
    label_key: str = "sequence_id",
    x_key: str = S.X,
    y_key: str = S.Y,
    background: pd.DataFrame | None = None,
    style: str | None = "dark",
    palette: Mapping[str, str] | None = None,
    connect: bool = True,
    figsize: tuple[float, float] = (13.333, 6.0),
    s: float = 30.0,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
) -> Any:
    """A lineage tree (one row per tip, x = mutations from the germline) next to the tips' tissue positions.

    Parameters
    ----------
    tree
        A :class:`spaclone.bcr.LineageTree`.
    tips
        Tip table of that tree (:func:`spaclone.bcr.tip_table`): the tree tip label (``label_key``), coordinates
        in µm and the colour column; several rows may share a tree tip (collapsed sequences at several places).
    color_key
        Column colouring tips in both panels (isotype, region, location).
    label_key, x_key, y_key
        Column names.
    background
        Optional table of tissue positions (``x_key`` / ``y_key``) drawn in grey under the tips.
    style
        ``"dark"`` (house style, default) or None / ``"light"``.
    palette
        Colours by value of ``color_key`` (isotype and house-class colours by default; other values by their
        sorted position).
    connect
        Join every tree row to its tissue position(s) with a faint line.
    figsize
        Figure size in inches.
    s
        Marker size of tips.
    invert_y
        Draw the tissue panel with y increasing downwards (image convention, as every spaclone map); ``False`` puts
        y = 0 at the bottom.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the background, or of
        the tips) that hold 95% of the points, with any piece next to them; specks far from the section (e.g. a stray
        cluster of records on the glass) are left out of the view and counted in an INFO log message. ``False`` frames
        every point.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    for c in (label_key, x_key, y_key):
        if c not in tips.columns:
            raise KeyError(f"tree_tissue: column {c!r} not in the tip table")
    pal = _palette(style)
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.4]})
    for ax in (ax_t, ax_m):
        _axes(ax, style, figsize)
        ax.patch.set_visible(False)  # the figure-level connecting lines are drawn first: let them show beneath
    x, y, order = _tree_layout(tree)
    for v, p in enumerate(tree.parent):
        if p < 0 or not np.isfinite(y[v]):
            continue
        ax_t.plot([x[p], x[v]], [y[v], y[v]], color=pal["line"], lw=1.0, zorder=1)
        ax_t.plot([x[p], x[p]], [y[p], y[v]], color=pal["line"], lw=1.0, zorder=1)
    vals = tips[color_key].astype("string").fillna("unknown").astype(str) if color_key in tips else None
    keys = list(pd.unique(vals)) if vals is not None else ["tip"]
    cols = _colors_for(keys, palette, style, vocabulary(vals) if vals is not None else None)
    fallback = SLOTS[mode_of(style)][0]
    first = tips.drop_duplicates(label_key).set_index(label_key)
    tip_col = {}
    for v in order:
        lab = tree.label[v]
        val = str(first.loc[lab, color_key]) if (vals is not None and lab in first.index) else "tip"
        tip_col[v] = cols.get(val, fallback)
        ax_t.scatter(x[v], y[v], s=s, c=tip_col[v], lw=0.3, edgecolors=pal["bg"], zorder=3)
    ax_t.set_xlabel("mutations from the germline")
    ax_t.set_yticks([])
    ax_t.set_ylim(len(order) - 0.5, -0.5)
    for sp in ("top", "right", "left"):
        ax_t.spines[sp].set_visible(False)
    txy = tips[[x_key, y_key]].to_numpy(dtype=float)
    bxy = background[[x_key, y_key]].to_numpy(dtype=float) if background is not None else None
    fr = SectionFrame(bxy if bxy is not None and len(bxy) else txy, rotate, crop=crop)
    attach(ax_m, fr)
    txy = fr(txy)
    if bxy is not None:
        bxy = fr(bxy)
        ax_m.scatter(bxy[:, 0], bxy[:, 1], s=0.5, c=pal["tissue"], lw=0, rasterized=True, zorder=0)
    c_pts = [cols.get(str(v), fallback) for v in vals] if vals is not None else cols["tip"]
    ax_m.scatter(txy[:, 0], txy[:, 1], s=s, c=c_pts, lw=0.3, edgecolors=pal["bg"], zorder=3)
    ax_m.set_xlabel(fr.labels[0])
    ax_m.set_ylabel(fr.labels[1])
    fr.set_view(ax_m, invert_y=invert_y)
    if connect:
        row = {tree.label[v]: v for v in order}
        for lab, xx, yy in zip(tips[label_key], txy[:, 0], txy[:, 1], strict=True):
            v = row.get(lab)
            if v is None:
                continue
            con = ConnectionPatch(
                xyA=(x[v], y[v]),
                coordsA=ax_t.transData,
                xyB=(xx, yy),
                coordsB=ax_m.transData,
                color=tip_col[v],
                lw=0.4,
                alpha=0.35,
                zorder=0,
            )
            fig.add_artist(con)
    handles = [plt.Line2D([], [], ls="", marker="o", color=cols[k_], label=k_) for k_ in keys]
    leg = ax_m.legend(handles=handles, frameon=False, fontsize=9, loc="best", title=color_key)
    for t in [*leg.get_texts(), leg.get_title()]:
        t.set_color(pal["fg"])
    ax_t.set_title(f"clone {tree.clone_id}", color=pal["fg"], fontsize=10)
    return fig
