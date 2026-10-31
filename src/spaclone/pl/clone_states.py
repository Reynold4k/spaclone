"""Threadfin clone states on tissue (:mod:`spaclone.ext.threadfin`): clone-axis maps, niche profiles, spanning clones.

All functions take the :class:`~spaclone.ext.threadfin.ThreadfinSpatialResult` of :func:`spaclone.ext.threadfin.run`
(they use its ``units``, ``clones``, ``context`` and ``landscape`` tables only) and, optionally, the
:class:`~spaclone.TestResult` of the matching readout, whose null is drawn next to the observed values. ``style="dark"``
is the project house style (black background). No function calls ``show()`` or ``savefig()``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from scipy import ndimage as ndi
from scipy.sparse.csgraph import minimum_spanning_tree

from spaclone import _schema as S
from spaclone.pl._colors import CHROME, DIVERGING, MARKS, NEUTRAL, REGION_COLORS, SLOTS
from spaclone.pl._frame import SectionFrame, attach, frame_of

__all__ = ["clone_axis_profile", "clone_state_map", "spanning_clones"]

logger = logging.getLogger("spaclone")

#: Surface colours of the two styles (dark = the project's black house style), from :mod:`spaclone.pl._colors`.
_PALETTE: dict[str, dict[str, str]] = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "fg": CHROME[mode]["ink"],
        "muted": CHROME[mode]["muted"],
        "grid": CHROME[mode]["grid"],
        "tissue": CHROME[mode]["tissue"],
        "t_edge": REGION_COLORS[mode]["T zone"],
        "region": MARKS[mode]["hot"],
        "band": CHROME[mode]["ink"],
    }
    for mode in ("dark", "light")
}
#: Categorical slots (validated reference palette; never cycled: names beyond the slots are drawn in grey).
_SLOTS = SLOTS
_NEUTRAL = NEUTRAL
#: Fixed colours of the compartments and regions (a compartment keeps its colour in every plot; the non-B/T area is the
#: red pulp's tissue-neutral name, see :func:`spaclone.ext.threadfin.spatial_context`, and keeps its colour).
_NAMED = {
    mode: {
        "follicle": REGION_COLORS[mode]["follicle"],
        "T zone": REGION_COLORS[mode]["T zone"],
        "red pulp": REGION_COLORS[mode]["red pulp"],
        "non-B/T area": REGION_COLORS[mode]["red pulp"],
        "hotspot": SLOTS[mode][3],
    }
    for mode in ("dark", "light")
}
#: Compartments listed first, in this order, in compartment panels (T zone, follicle, red pulp under either name).
_COMPARTMENT_ORDER = ("T zone", "follicle", "red pulp", "non-B/T area")
#: Diverging maps (blue, neutral grey, red) for signed axis values.
_DIVERGING = DIVERGING


def _mode(style: str | None) -> str:
    if style not in (None, "light", "dark"):
        raise ValueError("style must be 'dark', 'light' or None")
    return "dark" if style == "dark" else "light"


def _diverging(mode: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"spaclone_div_{mode}", list(_DIVERGING[mode]), N=256)


def _colours(names: Sequence[str], mode: str) -> dict[str, str]:
    named = _NAMED[mode]
    free = [c for c in _SLOTS[mode] if c not in named.values()]
    out, k = {}, 0
    for n in names:
        if n in named:
            out[n] = named[n]
        elif k < len(free):
            out[n] = free[k]
            k += 1
        else:
            out[n] = _NEUTRAL[mode]
    return out


def _style_axes(ax: Axes, pal: dict[str, str], *, frame: bool = True) -> None:
    ax.set_facecolor(pal["bg"])
    ax.figure.set_facecolor(pal["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(pal["muted"] if frame else pal["bg"])
    ax.tick_params(colors=pal["fg"], which="both")
    ax.xaxis.label.set_color(pal["fg"])
    ax.yaxis.label.set_color(pal["fg"])
    ax.title.set_color(pal["fg"])


def _units(result: Any) -> pd.DataFrame:
    u = getattr(result, "units", None)
    if not isinstance(u, pd.DataFrame) or S.X not in u.columns:
        raise TypeError("expected the result of spaclone.ext.threadfin.run (with a units table)")
    return u


def _axis_col(result: Any, axis: int) -> str:
    c = f"axis_{int(axis)}"
    if c not in _units(result).columns:
        raise ValueError(f"axis {axis} was not kept (run(n_axes=...))")
    return c


def _pix_size(ax: Axes, unit_um: float, grow: float = 1.1) -> float:
    """Marker area (pt²) of a square that covers ``unit_um`` on the axes (call after the limits are set)."""
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    return float((w_pt / max(abs(x1 - x0), 1e-9) * unit_um * grow) ** 2)


def _frame(ax: Axes, fr: SectionFrame, pad: float, invert_y: bool) -> None:
    """View of the main tissue of ``fr`` (see :class:`spaclone.pl._frame.SectionFrame`), no ticks, no spines."""
    fr.set_view(ax, invert_y=invert_y, pad_um=pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _outline(ax: Axes, raster: Any, colour: str, lw: float, *, smooth: float = 1.2, zorder: int = 3) -> None:
    """Contour (level 0.5) of a boolean raster after a light Gaussian smoothing (pixels -> smooth outline)."""
    m = np.asarray(raster.values) > 0
    if not m.any():
        return
    gx, gy = raster.grid.centres()
    fr = frame_of(ax)
    ax.contour(
        gx,
        gy,
        ndi.gaussian_filter(m.astype(float), smooth).T,
        levels=[0.5],
        colors=[colour],
        linewidths=lw,
        zorder=zorder,
        **({} if fr is None else {"transform": fr.affine() + ax.transData}),
    )


def _union_label(ctx: Any) -> str:
    """Name of the B-rich area joined with the T zones, from the context's own compartment names.

    ``"white pulp (B-rich + T zones)"`` when the context names a red pulp (which
    :func:`spaclone.ext.threadfin.spatial_context` does only where the labels show an erythroid-rich red pulp, as in
    a spleen), else the tissue-neutral ``"B-rich + T zones"`` (e.g. a lymph node).
    """
    comp = getattr(ctx, "compartments", None)
    cats = set(getattr(comp, "categories", None) or ())
    return "white pulp (B-rich + T zones)" if "red pulp" in cats else "B-rich + T zones"


def _draw_context(ax: Axes, result: Any, pal: dict[str, str], outlines: bool, regions: bool) -> list[tuple]:
    """Outlines of the B-rich area, T zones and regions of the result's context; returns legend entries."""
    ctx = getattr(result, "context", None)
    handles: list[tuple] = []
    if ctx is None:
        return handles
    if outlines:
        if "B-rich" in ctx.masks:
            b = ctx.masks["B-rich"]
            t = ctx.masks.get("T zone")
            wp = type(b)(np.asarray(b.values, bool) | (np.asarray(t.values, bool) if t is not None else False), b.grid)
            _outline(ax, wp, pal["fg"], 0.6)
            handles.append((_union_label(ctx), pal["fg"]))
        if "T zone" in ctx.masks:
            _outline(ax, ctx.masks["T zone"], pal["t_edge"], 0.8)
            handles.append(("T zone", pal["t_edge"]))
    if regions:
        for name, lab in ctx.regions.items():
            _outline(ax, type(lab)(np.asarray(lab.values) > 0, lab.grid), pal["region"], 1.4, smooth=0.8, zorder=4)
            handles.append((str(name), pal["region"]))
    return handles


def _legend_lines(ax: Axes, entries: list[tuple], pal: dict[str, str], *, above: bool = True) -> int:
    """Outline legend in rows above the axes (outside the map); returns the number of rows."""
    from matplotlib.lines import Line2D

    if not entries:
        return 0
    hs = [Line2D([0, 1], [0, 0], color=c, lw=1.6, label=lab) for lab, c in entries]
    ncol = min(len(hs), 3)
    if above:
        leg = ax.legend(
            handles=hs,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=ncol,
            frameon=False,
            fontsize=7.5,
            labelcolor=pal["fg"],
            handlelength=1.4,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
    else:
        leg = ax.legend(handles=hs, loc="upper right", frameon=False, fontsize=8, labelcolor=pal["fg"])
    leg.set_zorder(10)
    return int(np.ceil(len(hs) / ncol))


def _scalebar(ax: Axes, length_um: float, pal: dict[str, str]) -> None:
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    if length_um > 0.25 * width:  # a round length that fits
        nice = [v for v in (10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000) if v <= 0.25 * width]
        if not nice:
            return
        length_um = float(nice[-1])
    yb = y0 + 0.03 * (y1 - y0)
    xb = x1 - 0.03 * width - length_um
    ax.plot([xb, xb + length_um], [yb, yb], color=pal["fg"], lw=2, solid_capstyle="butt", zorder=6)
    lab = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
    ax.annotate(lab, (xb + length_um / 2, yb), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=pal["fg"], fontsize=8, zorder=6)  # fmt: skip


def _sym_norm(values: np.ndarray, vmax: float | None, quantile: float) -> Normalize:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    m = float(vmax) if vmax is not None else (float(np.quantile(np.abs(v), quantile)) if len(v) else 1.0)
    m = m if m > 0 else 1.0
    return Normalize(-m, m)


def clone_state_map(
    result: Any,
    *,
    color: str = "clone_axis",
    axis: int = 1,
    ax: Axes | None = None,
    style: str | None = "dark",
    unit_um: float | None = None,
    background: bool = True,
    outlines: bool = True,
    regions: bool = True,
    cmap: Any = None,
    vmax: float | None = None,
    quantile: float = 0.98,
    end_labels: tuple[str, str] | None = None,
    colorbar: bool = True,
    legend: bool = True,
    scalebar_um: float | None = 500.0,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    title: str | None = None,
) -> Axes:
    """Tissue coloured by Threadfin clone states.

    Parameters
    ----------
    result
        :func:`spaclone.ext.threadfin.run` result.
    color
        ``"clone_axis"``: units of profiled clones coloured by their clone's axis value; ``"axis"``: every unit by its
        own projected position; ``"programme"``: units of profiled clones by Threadfin programme; or any column of
        ``result.units`` (numeric: diverging around 0 for signed values, else sequential; categorical: fixed slots).
    axis
        Clone-profile axis.
    ax
        Axes to draw into (otherwise a new figure 12 inches wide, as tall as the section, its legend and its colour
        bar need).
    style
        ``"dark"`` (house style, black) or ``"light"`` / None.
    unit_um
        Side of a unit's square marker in µm (default: the median nearest-neighbour distance of the units, i.e. the
        bin size).
    background
        Draw every unit underneath in the tissue colour.
    outlines, regions
        Draw the context's outlines (the B-rich area with the T zones, called white pulp where the context names a
        red pulp, and the T zones) and its regions (yellow).
    cmap
        Colour map for numeric colours (default: diverging blue - grey - red).
    vmax, quantile
        Symmetric colour limits: ``±vmax``, else ``±`` the ``quantile`` of the absolute values.
    end_labels
        Names of the negative and positive ends written under the colour bar (e.g. ``("red-pulp end",
        "white-pulp end")`` on a spleen).
    colorbar, legend, scalebar_um, invert_y, title
        Colour bar, legend of outlines / categories, scale bar length (None = none), image-like y axis, title.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the units) that hold
        95% of the units, with any piece next to them; specks far from the section (e.g. a stray cluster of records on
        the glass) are left out of the view and counted in an INFO log message. ``False`` frames every unit.

    Returns
    -------
    matplotlib.axes.Axes
    """
    u = _units(result)
    mode = _mode(style)
    pal = _PALETTE[mode]
    fr = SectionFrame(np.c_[u[S.X].to_numpy(float), u[S.Y].to_numpy(float)], rotate, crop=crop)
    x, y = fr(u[S.X].to_numpy(float), u[S.Y].to_numpy(float)).T
    if unit_um is None:
        from scipy.spatial import cKDTree

        sub = np.c_[x, y][:: max(1, len(x) // 5000)]
        d, _ = cKDTree(sub).query(sub, k=2)
        unit_um = float(np.median(d[:, 1])) if len(sub) > 1 else 10.0
        unit_um = unit_um if unit_um > 0 else 10.0
    layout = None
    if ax is None:  # a slide-wide figure shaped like the section: the map fills its width, nothing is left empty
        layout = _Layout(fr, 3 * unit_um, colorbar=colorbar, title=title is not None)
        ax = layout.axes()
    attach(ax, fr)
    _style_axes(ax, pal, frame=False)
    square = MarkerStyle("s").rotated(rad=-fr.angle if invert_y else fr.angle)
    _frame(ax, fr, pad=3 * unit_um, invert_y=invert_y)
    s = _pix_size(ax, unit_um)
    if background:
        ax.scatter(x, y, s=s, c=pal["tissue"], marker=square, lw=0, rasterized=True, zorder=1)
    entries = _draw_context(ax, result, pal, outlines, regions)
    sm = None
    cat_entries: list[tuple] = []
    if color in ("clone_axis", "axis"):
        col = f"clone_{_axis_col(result, axis)}" if color == "clone_axis" else _axis_col(result, axis)
        v = u[col].to_numpy(float)
        sel = np.isfinite(v)
        norm = _sym_norm(v[sel], vmax, quantile)
        cm = cmap or _diverging(mode)
        order = np.argsort(np.abs(v[sel]))  # extreme values drawn last
        xs, ys, vs = x[sel][order], y[sel][order], v[sel][order]
        sm = ax.scatter(xs, ys, s=s, c=vs, cmap=cm, norm=norm, marker=square, lw=0, rasterized=True, zorder=2)
        label = f"{'clone' if color == 'clone_axis' else 'unit'} axis {axis}"
    else:
        key = "programme" if color == "programme" else color
        if key not in u.columns:
            raise KeyError(f"result.units has no column {key!r}")
        v = u[key]
        if key == "programme" or not pd.api.types.is_numeric_dtype(v):
            sel = v.notna().to_numpy() & (v.astype(str).to_numpy() != "nan")
            cats = sorted(pd.unique(v[sel].astype(str)), key=lambda t: (len(t), t))
            cols = _colours(cats, mode)
            for c in cats:
                m = sel & (v.astype(str).to_numpy() == c)
                ax.scatter(x[m], y[m], s=s, c=cols[c], marker=square, lw=0, rasterized=True, zorder=2)
            cat_entries = [(f"{c} ({int((v.astype(str) == c).sum()):,})", cols[c]) for c in cats]
            label = key
        else:
            vals = v.to_numpy(float)
            sel = np.isfinite(vals)
            signed = (vals[sel] < 0).any() and (vals[sel] > 0).any()
            if signed:
                norm = _sym_norm(vals[sel], vmax, quantile)
                cm = cmap or _diverging(mode)
            else:
                lo, hi = np.quantile(vals[sel], [1 - quantile, quantile]) if sel.any() else (0, 1)
                norm = Normalize(lo, hi if hi > lo else lo + 1)
                cm = cmap or LinearSegmentedColormap.from_list("seq", [pal["tissue"], _SLOTS[mode][0]])
            sm = ax.scatter(
                x[sel], y[sel], s=s, c=vals[sel], cmap=cm, norm=norm, marker=square, lw=0, rasterized=True, zorder=2
            )
            label = key
    if colorbar and (sm is not None or cat_entries):
        from matplotlib.colorbar import make_axes

        cax = (
            layout.colorbar_axes()
            if layout is not None
            else make_axes(ax, orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.6)[0]
        )
        if sm is not None:
            cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(label, color=pal["fg"])
            cb.ax.tick_params(colors=pal["fg"], labelsize=8)
            cb.outline.set_edgecolor(pal["muted"])
            if end_labels:
                kw = {"transform": cb.ax.transAxes, "va": "bottom", "color": pal["fg"], "fontsize": 8}
                cb.ax.text(0.0, 1.25, f"← {end_labels[0]}", ha="left", **kw)
                cb.ax.text(1.0, 1.25, f"{end_labels[1]} →", ha="right", **kw)
        else:  # categories go where a colour bar would be, so panels keep the same size
            from matplotlib.lines import Line2D

            cax.set_axis_off()
            hs = [Line2D([], [], color=c, lw=0, marker="s", ms=7, label=lab) for lab, c in cat_entries]
            cax.legend(
                handles=hs,
                loc="center",
                ncol=min(len(hs), 4),
                frameon=False,
                fontsize=8,
                labelcolor=pal["fg"],
                handletextpad=0.3,
                columnspacing=1.0,
            )
    elif cat_entries:
        entries = cat_entries + entries
    rows = _legend_lines(ax, entries, pal) if legend else 0
    if layout is not None:
        layout.finish(rows)
    if scalebar_um:
        _scalebar(ax, scalebar_um, pal)
    if title is not None:
        ax.set_title(title, color=pal["fg"], fontsize=11, pad=6 + 11 * rows)
    return ax


class _Layout:
    """A figure 12 inches wide whose map fills the width: room above for legend rows and a title, below for a bar."""

    width, side = 12.0, 0.02
    row_in, title_in, bar_in, gap_in = 0.2, 0.3, 0.85, 0.12

    def __init__(self, fr: SectionFrame, pad: float, *, colorbar: bool, title: bool) -> None:
        e = fr.extent or (0.0, 1.0, 0.0, 1.0)
        w, h = max(e[1] - e[0], 1e-9) + 2 * pad, max(e[3] - e[2], 0.0) + 2 * pad
        self.map_w = self.width * (1 - 2 * self.side)
        self.map_h = float(np.clip(self.map_w * h / w, 1.5, 14.0))
        self.title = self.title_in if title else 0.0
        self.bottom = self.bar_in if colorbar else self.gap_in
        self.fig = plt.figure(figsize=(self.width, self._height(1)), dpi=200)
        self.ax: Axes | None = None
        self.cax: Axes | None = None

    def _height(self, rows: int) -> float:
        return self.bottom + self.map_h + self.gap_in + self.row_in * max(rows, 1) + self.title

    def _place(self, rows: int) -> None:
        hgt = self._height(rows)
        self.fig.set_size_inches(self.width, hgt)
        if self.ax is not None:
            self.ax.set_position([self.side, self.bottom / hgt, 1 - 2 * self.side, self.map_h / hgt])
        if self.cax is not None:
            self.cax.set_position([0.3, (self.bottom - 0.47) / hgt, 0.4, 0.12 / hgt])

    def axes(self) -> Axes:
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self._place(1)
        return self.ax

    def colorbar_axes(self) -> Axes:
        self.cax = self.fig.add_axes([0, 0, 1, 1])
        self._place(1)
        return self.cax

    def finish(self, rows: int) -> None:
        """Grow the figure for the legend rows, keeping the map's size in inches (marker sizes stay right)."""
        if rows > 1:
            self._place(rows)


def _decile_table(result: Any, axis: int, n_bins: int, compartment_key: str) -> pd.DataFrame:
    """Observed compartment shares of member units by clone-axis bin (as :func:`niche_by_axis`)."""
    u = _units(result)
    col = f"clone_{_axis_col(result, axis)}"
    ck = result.params.get("clone_key", S.CLONE) if hasattr(result, "params") else S.CLONE
    pr = u[u[col].notna()]
    cl = pr.groupby(ck, sort=False)[col].first()
    b = pd.qcut(cl.rank(method="first"), n_bins, labels=[f"D{k + 1}" for k in range(n_bins)])
    tab = pd.crosstab(pr[ck].map(b), pr[compartment_key].astype(str), normalize="index")
    return tab.reindex([f"D{k + 1}" for k in range(n_bins)])


def clone_axis_profile(
    result: Any,
    test: Any = None,
    *,
    axis: int = 1,
    n_bins: int = 10,
    compartment_key: str = "compartment",
    categories: Sequence[str] | None = None,
    expression: pd.DataFrame | None = None,
    genes: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    n_genes: int = 6,
    style: str | None = "dark",
    figsize: tuple[float, float] = (13.333, 5.5),
) -> Figure:
    """Niche composition and marker expression along a clone axis.

    Left: share of the member units of each clone-axis bin in each compartment (observed, solid) with the null mean
    and its 95% band from :func:`spaclone.ext.threadfin.niche_by_axis` when ``test`` is given. Right: mean marker
    expression of the member units per bin, z-scored per gene (heat map), for ``genes`` or, by default, the genes at
    both ends of the axis from ``result.landscape`` (axis 1).

    Parameters
    ----------
    result
        :func:`spaclone.ext.threadfin.run` result (with compartments).
    test
        Optional :func:`spaclone.ext.threadfin.niche_by_axis` result of the same axis and bins.
    axis, n_bins
        Clone-profile axis and number of clone bins.
    compartment_key
        ``result.units`` column of compartments.
    categories
        Compartments (and ``test`` flag categories) drawn; default every compartment.
    expression
        Units × genes table of expression values (e.g. log-normalised counts), indexed like ``result.units``;
        default ``result.adata.layers["log_norm"]`` when the full result is at hand.
    genes
        Genes of the heat map (a list, or ``{group: [genes]}`` whose group means are drawn).
    n_genes
        Genes per end taken from the landscape when ``genes`` is None.
    style
        ``"dark"`` (house style) or ``"light"`` / None.
    figsize
        Figure size in inches (16:9 slide width by default).

    Returns
    -------
    matplotlib.figure.Figure
    """
    u = _units(result)
    has_comp = compartment_key in u.columns
    if not has_comp:
        logger.info(
            "clone_axis_profile: result.units has no %r column (run ext.threadfin.run with context=); "
            "the compartment panel is left out",
            compartment_key,
        )
    mode = _mode(style)
    pal = _PALETTE[mode]
    obs = _decile_table(result, axis, n_bins, compartment_key) if has_comp else pd.DataFrame()
    for c in list(categories or []) if has_comp else []:  # boolean flags (e.g. in a region) drawn like compartments
        if c not in obs.columns and c in u.columns:
            col_ = f"clone_{_axis_col(result, axis)}"
            ck_ = result.params.get("clone_key", S.CLONE)
            pr_ = u[u[col_].notna()]
            cl_ = pr_.groupby(ck_, sort=False)[col_].first()
            b_ = pd.qcut(cl_.rank(method="first"), n_bins, labels=[f"D{k + 1}" for k in range(n_bins)])
            obs[c] = pr_[c].fillna(False).astype(float).groupby(pr_[ck_].map(b_), observed=False).mean()
    cats = (
        list(categories)
        if categories is not None
        else [c for c in _COMPARTMENT_ORDER if c in obs.columns]
        + [c for c in obs.columns if c not in (*_COMPARTMENT_ORDER, "outside")]
    )
    # expression panel input
    expr = expression
    want_default = genes is None and getattr(result, "landscape", None) is not None
    if expr is None and (genes is not None or want_default):
        ad_ = getattr(getattr(result, "threadfin", None), "adata", None)
        if ad_ is not None and "log_norm" in ad_.layers:
            want: list[str] = []
            if genes is None and result.landscape is not None and len(result.landscape) and int(axis) == 1:
                lg = result.landscape.sort_values("corr_axis1")
                want = list(lg.gene.head(n_genes)) + list(lg.gene.tail(n_genes))
            elif genes is not None:
                want = [g for gs in (genes.values() if isinstance(genes, Mapping) else [genes]) for g in gs]
            want = [g for g in dict.fromkeys(want) if g in ad_.var_names]
            if want:
                m = ad_[:, want].layers["log_norm"]
                m = m.toarray() if hasattr(m, "toarray") else np.asarray(m)
                expr = pd.DataFrame(m, index=ad_.obs_names, columns=want)
    has_expr = expr is not None and expr.shape[1] > 0
    if not has_comp and not has_expr:
        raise ValueError(
            f"clone_axis_profile: nothing to draw: result.units has no {compartment_key!r} column (run "
            "ext.threadfin.run with context=) and no expression was given (expression= or genes=)"
        )
    fig = plt.figure(figsize=figsize, facecolor=pal["bg"])
    gs = fig.add_gridspec(
        1,
        int(has_comp) + int(has_expr),
        width_ratios=[1.15, 1] if (has_expr and has_comp) else [1],
        wspace=0.28,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.14,
    )
    if has_comp:
        _compartment_panel(fig.add_subplot(gs[0, 0]), obs, cats, test, axis, n_bins, mode, pal)
    if has_expr:
        ax2 = fig.add_subplot(gs[0, int(has_comp)])
        _style_axes(ax2, pal)
        col = f"clone_{_axis_col(result, axis)}"
        ck = result.params.get("clone_key", S.CLONE)
        pr = u[u[col].notna()]
        cl = pr.groupby(ck, sort=False)[col].first()
        b = pd.qcut(cl.rank(method="first"), n_bins, labels=False)
        ub = pr[ck].map(b)
        e = expr.reindex(pr.index)
        if isinstance(genes, Mapping):
            e = pd.DataFrame({k: e[[g for g in v if g in e.columns]].mean(axis=1) for k, v in genes.items()})
        means = e.groupby(ub.to_numpy()).mean().reindex(range(n_bins)).T
        z = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, np.nan), axis=0)
        lim = float(np.nanmax(np.abs(z.to_numpy()))) if np.isfinite(z.to_numpy()).any() else 1.0
        im = ax2.imshow(
            z.to_numpy(float), aspect="auto", cmap=_diverging(mode), vmin=-lim, vmax=lim, interpolation="nearest"
        )
        ax2.set_xticks(range(n_bins), [f"D{k + 1}" for k in range(n_bins)])
        ax2.set_yticks(range(z.shape[0]), list(z.index), fontsize=8)
        ax2.set_xlabel(f"clone axis {axis} bin (low → high)")
        ax2.set_title("Marker expression of member units (z-score per gene)", fontsize=11, color=pal["fg"], loc="left")
        cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
        cb.ax.tick_params(colors=pal["fg"], labelsize=8)
        cb.outline.set_edgecolor(pal["muted"])
        cb.set_label("z", color=pal["fg"])
    return fig


def _compartment_panel(
    ax: Axes, obs: pd.DataFrame, cats: list[str], test: Any, axis: int, n_bins: int, mode: str, pal: dict[str, str]
) -> None:
    """Share of member units per clone-axis bin in each compartment (observed, with the null band when tested)."""
    _style_axes(ax, pal)
    xs = np.arange(1, n_bins + 1)
    cols = _colours(cats, mode)
    t = test.table if test is not None else None
    for c in cats:
        if c in obs.columns:
            ax.plot(xs, obs[c].to_numpy(float), color=cols[c], lw=2, marker="o", ms=5, label=c, zorder=3)
        if t is not None:
            tt = t[(t.kind == "share") & (t.category == c)]
            if len(tt) == n_bins:
                ax.fill_between(
                    xs,
                    tt.null_lo.to_numpy(float),
                    tt.null_hi.to_numpy(float),
                    color=cols[c],
                    alpha=0.18,
                    lw=0,
                    zorder=1,
                )
                ax.plot(xs, tt.null_mean.to_numpy(float), color=cols[c], lw=1, ls="--", zorder=2)
    ax.set_xticks(xs, [f"D{k}" for k in xs])
    ax.set_xlabel(f"clone axis {axis} bin (low → high)")
    ax.set_ylabel("share of member units")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=pal["grid"], lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=pal["fg"], loc="upper left", ncol=2)
    leg.set_zorder(10)
    sub = "solid: observed" + ("; dashed + band: null mean and 95% (clone labels permuted)" if t is not None else "")
    ax.set_title(f"Compartments of clone members along axis {axis}\n{sub}", fontsize=11, color=pal["fg"], loc="left")


def _mst_segments(xy: np.ndarray) -> np.ndarray:
    """Segments of the Euclidean minimum spanning tree of distinct points."""
    p = np.unique(xy, axis=0)
    if len(p) < 2:
        return np.empty((0, 2, 2))
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    i, j = minimum_spanning_tree(d).nonzero()
    return np.stack([p[i], p[j]], axis=1)


def _short(clone: str) -> str:
    parts = str(clone).split("|")
    if len(parts) == 3:
        return f"{parts[0].replace('IGHV', 'V')} {parts[2]}"
    return str(clone)


def spanning_clones(
    result: Any,
    test: Any = None,
    *,
    axis: int = 1,
    clones: Sequence[str] | None = None,
    n_top: int = 4,
    min_units: int = 5,
    ncols: int = 2,
    style: str | None = "dark",
    outlines: bool = True,
    regions: bool = False,
    vmax: float | None = None,
    unit_um: float | None = None,
    figsize: tuple[float, float] | None = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
) -> Figure:
    """The largest spanning clones on the tissue, linked by their minimum spanning trees.

    Each clone's units are coloured by their own position on the axis, so a clone whose members sit at both ends of
    the axis shows both colours.

    Parameters
    ----------
    result
        :func:`spaclone.ext.threadfin.run` result.
    test
        Optional :func:`spaclone.ext.threadfin.spanning_clones` result (picks the clones and adds their p-values).
    axis
        Clone-profile axis.
    clones
        Clone ids to draw (default: the ``n_top`` spanning clones with most units).
    n_top, min_units
        Number of clones; units of an expanded clone (when spanning is computed here).
    ncols
        Panels per row.
    style
        ``"dark"`` (house style) or ``"light"`` / None.
    outlines, regions
        Context outlines (the B-rich area with the T zones, and the T zones) and regions.
    vmax
        Colour limit (default: 98th percentile of the absolute positions of profiled units).
    unit_um
        Marker side of the background units (default: bin size).
    figsize
        Figure size (default: 13.333 in wide, rows as tall as the section needs).
    invert_y
        Image-like y axis.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the units) that hold
        95% of the units, with any piece next to them; specks far from the section (e.g. a stray cluster of records on
        the glass) are left out of the view and counted in an INFO log message. ``False`` frames every unit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    u = _units(result)
    mode = _mode(style)
    pal = _PALETTE[mode]
    col = _axis_col(result, axis)
    ck = result.params.get("clone_key", S.CLONE)
    pr = u[u[f"clone_{col}"].notna()]
    if clones is None:
        if test is not None and "spanning" in test.table.columns:
            cand = test.table[test.table.spanning.astype(bool)].sort_values("n_units", ascending=False)
            clones = list(cand[ck].astype(str).head(n_top))
        else:
            q1, q2 = np.quantile(pr[col], [1 / 3, 2 / 3])
            g = pr.groupby(ck)
            stats = pd.DataFrame({"n": g.size(), "lo": g[col].min(), "hi": g[col].max()})
            stats = stats[(stats.n >= min_units) & (stats.lo <= q1) & (stats.hi >= q2)]
            clones = list(stats.sort_values("n", ascending=False).index[:n_top].astype(str))
    clones = [str(c) for c in clones]
    if not clones:
        raise ValueError("no spanning clone to draw")
    n = len(clones)
    nrow = int(np.ceil(n / ncols))
    fr = SectionFrame(np.c_[u[S.X].to_numpy(float), u[S.Y].to_numpy(float)], rotate, crop=crop)
    x, y = fr(u[S.X].to_numpy(float), u[S.Y].to_numpy(float)).T
    # panels as tall as the section needs, a two-line title above each, the colour bar in a band at the bottom
    panel_w = 13.333 * 0.98 / ncols
    panel_h = float(np.clip(panel_w * fr.aspect(), 1.2, 6.0))
    title_in, bar_in = 0.55, 0.9
    size = figsize or (13.333, nrow * (panel_h + title_in) + bar_in)
    fig = plt.figure(figsize=size, facecolor=pal["bg"])
    gs = fig.add_gridspec(nrow, ncols, left=0.01, right=0.99, top=1 - title_in / size[1], bottom=bar_in / size[1],
                          wspace=0.04, hspace=title_in / panel_h)  # fmt: skip
    square = MarkerStyle("s").rotated(rad=-fr.angle if invert_y else fr.angle)
    norm = _sym_norm(pr[col].to_numpy(float), vmax, 0.98)
    cm = _diverging(mode)
    tt = test.table.set_index(test.table[ck].astype(str)) if test is not None else None
    sm = None
    for k, c in enumerate(clones):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        attach(ax, fr)
        _style_axes(ax, pal, frame=False)
        uu = unit_um
        if uu is None:
            from scipy.spatial import cKDTree

            sub = np.c_[x, y][:: max(1, len(x) // 5000)]
            d, _ = cKDTree(sub).query(sub, k=2)
            uu = float(np.median(d[:, 1])) if len(sub) > 1 else 10.0
        _frame(ax, fr, pad=3 * uu, invert_y=invert_y)
        s = _pix_size(ax, uu)
        ax.scatter(x, y, s=s, c=pal["tissue"], marker=square, lw=0, rasterized=True, zorder=1)
        _draw_context(ax, result, pal, outlines, regions)
        m = u[ck].astype(str).to_numpy() == c
        if not m.any():
            raise KeyError(f"clone {c!r} not in result.units")
        pts = np.c_[x[m], y[m]]
        ax.add_collection(LineCollection(_mst_segments(pts), colors=pal["fg"], lw=0.8, alpha=0.8, zorder=4))
        sm = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=max(18.0, 6 * s),
            c=u.loc[m, col].to_numpy(float),
            cmap=cm,
            norm=norm,
            lw=0.4,
            edgecolor=pal["bg"],
            zorder=5,
        )
        head = f"{_short(c)}\n{int(m.sum())} units"
        if tt is not None and c in tt.index:
            r = tt.loc[c]
            head += f" · axis range {r.axis_range:.2f}"
            if "p_range" in r and np.isfinite(r.p_range):
                head += f" (p {r.p_range:.2g})"
            if "p_cross" in r and np.isfinite(r.p_cross):
                head += f"\nlow↔high ends {r.cross_um:.0f} µm (p {r.p_cross:.2g})"
        ax.set_title(head, fontsize=8.5, color=pal["fg"], loc="left")
    if sm is not None:
        cax = fig.add_axes([0.35, 0.42 / size[1], 0.3, 0.13 / size[1]])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(f"unit position on axis {axis}", color=pal["fg"])
        cb.ax.tick_params(colors=pal["fg"], labelsize=8)
        cb.outline.set_edgecolor(pal["muted"])
    return fig
