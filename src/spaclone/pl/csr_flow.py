"""Transition plots on tissue: flow fields over a CSR-potential map, TPT flux matrices, stationary distributions.

* :func:`csr_flow`: units on the section coloured by the time key (CSR potential, SHM score, a committor), with the
  projected transition matrix drawn as grid arrows or streamlines (:func:`spaclone.tl.transition_displacement`,
  :func:`spaclone.tl.flow_grid`); with ``coords=`` the same arrows go on any embedding (sciCSR's ``plot_arrows``).
* :func:`flux_matrix`: the gross or net TPT flux between states as a bubble matrix (colour = percent of the
  flux, size = significance against the randomisation null), like sciCSR's ``plotFluxMatrix``.
* :func:`stationary_distribution`: the lumped stationary distribution per state with bootstrap intervals next to the
  observed share of units, like sciCSR's ``plotStationaryDistribution``.

Every function takes ``ax=None`` (a new figure is made), returns the ``Axes`` and never calls ``show()`` or
``savefig()``. ``style="dark"`` is the project house style (black surface).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl._frame import SectionFrame, attach

__all__ = ["csr_flow", "flux_matrix", "stationary_distribution"]


def _c(mode: str, key: str) -> str:
    """A surface / ink colour of the house theme (:func:`spaclone.pl._colors.theme`)."""
    from spaclone.pl._colors import theme

    return theme(mode)[key]


def _mode(style: str | None) -> str:
    if style not in (None, "light", "dark"):
        raise ValueError("style must be None, 'light' or 'dark'")
    return "dark" if style == "dark" else "light"


def _axes(ax: Any, mode: str, figsize: tuple[float, float]) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ink = _c(mode, "ink")
    ax.figure.set_facecolor(_c(mode, "bg"))
    ax.set_facecolor(_c(mode, "bg"))
    ax.tick_params(colors=_c(mode, "muted"), labelcolor=ink)
    for sp in ax.spines.values():
        sp.set_color(_c(mode, "muted"))
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    ax.title.set_color(ink)
    return ax


def _ramp_cmap(name: str, mode: str) -> Any:
    """Colormap of a house ramp (:data:`spaclone.pl._colors.RAMPS`)."""
    from spaclone.pl._colors import cmap

    return cmap(name, mode)


def _state_colors(states: Sequence[str], mode: str, palette: Mapping[str, str] | None) -> dict[str, str]:
    """Colour by state name: the package's region / cell-class and isotype colours, else fixed slots."""
    from spaclone.pl._colors import GREY_LABELS, ISOTYPE_COLORS, NEUTRAL, fixed_colors, named_colors

    named = named_colors(states, mode)
    named.update({s: c for s, c in ISOTYPE_COLORS[mode].items() if s in states and s not in named})
    neutral = {s: NEUTRAL[mode] for s in states if s in GREY_LABELS}
    return fixed_colors(states, mode=mode, named=named, palette=palette, neutral=neutral)


# ===================================================================================================== flow field
def csr_flow(
    data: Any,
    displacement: pd.DataFrame | None = None,
    *,
    color: str | Sequence[float] | np.ndarray | None = "time",
    kind: str = "stream",
    coords: np.ndarray | None = None,
    grid_um: float | None = None,
    n_grid: int = 40,
    smooth: float = 0.5,
    min_mass: float = 1.0,
    density: float = 1.4,
    arrow_color: str | None = None,
    s: float = 3.0,
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    label: str | None = None,
    ax: Any = None,
    style: str | None = "dark",
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    scalebar_um: float | None = None,
    palette: Mapping[str, str] | None = None,
    title: str | None = None,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> Any:
    """Flow of a transition model over the tissue: units coloured by the time key, the projected chain as arrows.

    Parameters
    ----------
    data
        A :class:`spaclone.tl.TransitionModel` (arrows from :func:`spaclone.tl.transition_displacement` unless
        ``displacement`` is given) or a units table with ``x_um`` / ``y_um`` and ``dx_um`` / ``dy_um``.
    displacement
        Per-unit vectors overriding the model's projection (e.g. ``method="unit"`` or another baseline).
    color
        Unit colours: ``"time"`` (the model's time key), a column of the table, per-unit values (e.g. the
        forward committor ``res.units.q_plus`` of :func:`spaclone.tl.tpt_flux`; or labels such as each unit's region,
        drawn in the house region colours with a legend) or None (units in grey).
    kind
        ``"stream"`` (streamlines on the flow grid), ``"grid"`` (one arrow per grid point), ``"units"`` (one arrow
        per unit; for small data) or ``"none"`` (the coloured units only, e.g. a committor map). Streamlines and grid
        arrows are drawn only on grid cells that hold a unit, so they stop at the edge of the tissue.
    coords
        ``(n, 2)`` coordinates replacing the tissue positions (e.g. a UMAP); the displacement is recomputed on them.
    grid_um, n_grid, smooth, min_mass
        Flow grid (:func:`spaclone.tl.flow_grid`).
    density
        Streamline density.
    arrow_color
        Arrow / streamline colour (default: the ink of the style).
    s
        Unit marker size; 0 draws the arrows only (e.g. over :func:`spaclone.pl.gex_regions` on the same axes).
    cmap, vmin, vmax
        Colour map and limits of the units (default: a single-hue blue ramp).
    colorbar
        Draw a colour bar.
    label
        Colour bar label (default from ``color``; the CSR potential of :func:`spaclone.bcr.csr_potential` is
        labelled "CSR potential").
    ax
        Axes to draw on (default: a new figure 12 inches wide, as tall as the section needs).
    style
        ``"dark"`` (house style, default) or None / ``"light"``.
    invert_y
        Draw y increasing downwards (image convention, the house default for tissue maps); False draws it upwards.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the units) that hold
        95% of the units, with any piece next to them; specks far from the section (e.g. a stray cluster of records on
        the glass) are left out of the view and counted in an INFO log message. ``False`` frames every unit.
    scalebar_um
        Length of the scale bar of a tissue map (default: a round length near a fifth of the section; 0 for none).
    palette
        Colours by label when ``color`` holds labels.
    title
        Axes title.
    x_key, y_key
        Coordinate columns of a table.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from spaclone.pl._colors import set_y_down
    from spaclone.tl.transitions import TransitionModel, flow_grid, transition_displacement

    mode = _mode(style)
    if kind not in ("stream", "grid", "units", "none"):
        raise ValueError(f"kind must be 'stream', 'grid', 'units' or 'none', got {kind!r}")
    if isinstance(data, TransitionModel):
        disp = displacement if displacement is not None else transition_displacement(data, coords=coords)
        if isinstance(color, str) and color == "time":
            values = data.time
            label = label or _nice_label(str(data.params.get("time_key", "time")))
        elif isinstance(color, str):
            raise ValueError("with a model, color must be 'time', per-unit values or None")
        else:
            values = None if color is None else _as_values(color)
    elif isinstance(data, pd.DataFrame):
        disp = displacement if displacement is not None else data
        if isinstance(color, str):
            key = "csr_pot" if color == "time" and "csr_pot" in data.columns else color
            if key == "time" and "time" in disp.columns:
                values = disp["time"].to_numpy(dtype=float)
            elif key in data.columns:
                values = _as_values(data[key].to_numpy())
            else:
                raise KeyError(f"color column {key!r} not in the table")
            label = label or _nice_label(key)
        else:
            values = None if color is None else _as_values(color)
    else:
        raise TypeError("csr_flow expects a TransitionModel or a units DataFrame")
    for c in (x_key, y_key, "dx_um", "dy_um"):
        if c not in disp.columns:
            raise KeyError(f"csr_flow: column {c!r} missing (pass a displacement table)")
    xy = disp[[x_key, y_key]].to_numpy(dtype=float)
    if values is not None and len(values) != len(xy):
        raise ValueError(f"color: {len(values)} values for {len(xy)} units")
    tissue = coords is None
    fr = SectionFrame(xy, rotate if tissue else False, crop=crop and tissue)
    if fr.rotated:  # positions and vectors turned together, so the flow grid is built in display coordinates
        p = fr(xy)
        v = disp[["dx_um", "dy_um"]].to_numpy(dtype=float) @ fr.R.T
        disp = disp.assign(**{x_key: p[:, 0], y_key: p[:, 1], "dx_um": v[:, 0], "dy_um": v[:, 1]})
        xy = p
    if ax is None:  # a figure shaped like the section: the map fills its width, the keys go in a band below
        import matplotlib.pyplot as plt

        e = fr.extent if tissue else None
        if e is not None:
            pad = 0.04 * max(e[1] - e[0], e[3] - e[2], 1.0)
            map_h = float(np.clip(11.52 * (e[3] - e[2] + 2 * pad) / (e[1] - e[0] + 2 * pad), 1.5, 11.0))
            top, bottom = (0.45 if title else 0.1), 0.75
            fig, ax = plt.subplots(figsize=(12.0, map_h + top + bottom), dpi=200)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=bottom / (map_h + top + bottom),
                                top=1 - top / (map_h + top + bottom))  # fmt: skip
        else:
            _, ax = plt.subplots(figsize=(7.5, 7.5), dpi=200)
    ax = _axes(ax, mode, (7.5, 7.5))
    attach(ax, fr)
    labels = None
    if values is not None and not np.issubdtype(np.asarray(values).dtype, np.number):
        labels, values = pd.Series(np.asarray(values, dtype=object)).astype(str).to_numpy(), None
    ink = arrow_color or _c(mode, "ink")
    if s <= 0:
        pass  # arrows only (e.g. over a region map drawn on the same axes)
    elif labels is not None:
        _label_units(ax, xy, labels, s, mode, palette)
    elif values is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=_c(mode, "muted"), linewidths=0, rasterized=True)
    else:
        cm = cmap if cmap is not None else _ramp_cmap("csr", mode)
        order = np.argsort(values, kind="stable")
        sc = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=values[order],
            s=s,
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        if colorbar:
            cb_label = label or ""
            if title and cb_label.strip().lower() == title.strip().lower():
                cb_label = ""  # the axes title already names the colour
            _discreet_colorbar(ax, sc, cb_label, mode)
    if kind == "none":
        pass
    elif kind == "units":
        v = disp[["dx_um", "dy_um"]].to_numpy(dtype=float)
        med = np.nanmedian(np.hypot(v[:, 0], v[:, 1]))
        span = float(np.ptp(xy, axis=0).max()) or 1.0
        scale = med / (0.02 * span) if med > 0 else 1.0
        ax.quiver(
            xy[:, 0], xy[:, 1], v[:, 0], v[:, 1], color=ink, angles="xy", scale_units="xy", scale=scale, width=0.0025
        )
    else:
        g = flow_grid(disp, grid_um=grid_um, n_grid=n_grid, smooth=smooth, min_mass=min_mass, x_key=x_key, y_key=y_key)
        if len(g):
            meta = g.attrs["spaclone"]["flow_grid"]
            h = meta["grid_um"]
            occ = _occupied(xy, meta)  # grid cells that hold a unit: no arrow off the tissue
            g = g[occ[g.iy.to_numpy(), g.ix.to_numpy()]]
            if kind == "grid":
                sp = np.hypot(g.dx_um, g.dy_um).to_numpy()
                med = np.nanmedian(sp)
                scale = med / (0.8 * h) if med > 0 else 1.0
                ax.quiver(
                    g[x_key],
                    g[y_key],
                    g.dx_um,
                    g.dy_um,
                    color=ink,
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    width=0.003,
                    headwidth=4,
                    headlength=4.5,
                )
            else:
                nx, ny = meta["nx"], meta["ny"]
                gx = meta["x0"] + h * np.arange(nx)
                gy = meta["y0"] + h * np.arange(ny)
                u = np.full((ny, nx), np.nan)
                v = np.full((ny, nx), np.nan)
                u[g.iy.to_numpy(), g.ix.to_numpy()] = g.dx_um.to_numpy()
                v[g.iy.to_numpy(), g.ix.to_numpy()] = g.dy_um.to_numpy()
                speed = np.hypot(u, v)
                ref = np.nanpercentile(speed, 90) if np.isfinite(speed).any() else 1.0
                lw = 0.4 + 1.4 * np.clip(np.nan_to_num(speed / ref if ref > 0 else speed), 0, 1)
                ax.streamplot(
                    gx,
                    gy,
                    np.ma.masked_invalid(u),
                    np.ma.masked_invalid(v),
                    density=density,
                    color=ink,
                    linewidth=lw,
                    arrowsize=0.8,
                    broken_streamlines=True,
                )
    if tissue:  # the main tissue, with a margin below it for the scale bar
        e = fr.extent
        fr.set_view(ax, invert_y=invert_y, pad_um=0.04 * max(e[1] - e[0], e[3] - e[2], 1.0) if e else None)
    else:
        ax.set_aspect("equal")
        set_y_down(ax, invert_y)
    if tissue:  # a tissue map: no axes, a scale bar
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        _scale_bar(ax, xy, mode, scalebar_um, invert_y)
    else:
        ax.set_xlabel("embedding 1")
        ax.set_ylabel("embedding 2")
    if title:
        ax.set_title(title, color=_c(mode, "ink"), loc="left", fontsize=11)
    return ax


#: Colour-bar labels of known time keys.
_NICE_LABELS = {"csr_pot": "CSR potential", "csr_potential": "CSR potential"}


def _nice_label(key: str) -> str:
    return _NICE_LABELS.get(key, key)


def _occupied(xy: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """``(ny, nx)`` mask of the flow-grid cells (nearest grid point) that hold at least one unit."""
    nx, ny, h = int(meta["nx"]), int(meta["ny"]), float(meta["grid_um"])
    occ = np.zeros((ny, nx), dtype=bool)
    if len(xy):
        ix = np.rint((xy[:, 0] - meta["x0"]) / h).astype(np.int64)
        iy = np.rint((xy[:, 1] - meta["y0"]) / h).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ[iy[ok], ix[ok]] = True
    return occ


def _as_values(color: Any) -> np.ndarray:
    """Per-unit colour values: floats, or the labels themselves when they are not numbers."""
    arr = np.asarray(color)
    return arr.astype(float) if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool else arr.astype(object)


def _label_units(ax: Any, xy: np.ndarray, labels: np.ndarray, s: float, mode: str, palette: Any) -> None:
    """Units coloured by label (regions, classes) with a legend under the map."""
    from matplotlib.lines import Line2D

    from spaclone.pl._colors import GREY_LABELS, NEUTRAL

    names = [n for n in pd.unique(labels) if n not in ("nan", "None")]
    cols = _state_colors(sorted(names), mode, palette)
    counts = pd.Series(labels).value_counts()
    for name in sorted(names, key=lambda n: -counts.get(n, 0)):  # large classes underneath
        m = labels == name
        col = NEUTRAL[mode] if name in GREY_LABELS else cols[name]
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=col, linewidths=0, rasterized=True, zorder=1)
    handles = [
        Line2D([], [], ls="", marker="o", ms=7, mfc=NEUTRAL[mode] if n in GREY_LABELS else cols[n], mec="none",
               label=f"{n} ({int(counts.get(n, 0)):,})")
        for n in sorted(names, key=lambda n: -counts.get(n, 0))
    ]  # fmt: skip
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.01),
        ncol=min(len(handles), 5),
        frameon=False,
        fontsize=9.5,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color(_c(mode, "ink"))


def _discreet_colorbar(ax: Any, mappable: Any, label: str, mode: str) -> None:
    """A thin horizontal colour bar under the lower right corner of the map: end values, label on its left."""
    cax = ax.inset_axes([0.76, -0.045, 0.22, 0.016], zorder=12)
    cb = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
    lo, hi = cb.mappable.norm.vmin, cb.mappable.norm.vmax
    if lo is not None and hi is not None and np.isfinite([lo, hi]).all():
        cb.set_ticks([lo, hi])
        cb.set_ticklabels([f"{lo:.2g}", f"{hi:.2g}"])
    cb.ax.tick_params(colors=_c(mode, "muted"), labelcolor=_c(mode, "muted"), labelsize=8, length=0, pad=2)
    cb.outline.set_visible(False)
    if label:
        cb.ax.text(-0.04, 0.5, label, transform=cb.ax.transAxes, ha="right", va="center", color=_c(mode, "ink"),
                   fontsize=9)  # fmt: skip


def _scale_bar(ax: Any, xy: np.ndarray, mode: str, length_um: float | None, invert_y: bool) -> None:
    """A scale bar at the lower left of a tissue map (in the margin under the tissue)."""
    span = abs(float(np.diff(ax.get_xlim())[0])) if len(xy) else 0.0
    if length_um is None:
        target = span / 5 if span > 0 else 0.0
        p = 10 ** np.floor(np.log10(target)) if target > 0 else 0.0
        length_um = float(min((1, 2, 5, 10), key=lambda m: abs(m * p - target)) * p) if p else 0.0
    if not length_um or length_um <= 0:
        return
    x0, x1 = ax.get_xlim()
    y_lo, y_hi = sorted(ax.get_ylim())
    xb = min(x0, x1) + 0.03 * abs(x1 - x0)
    yb = y_hi - 0.04 * (y_hi - y_lo) if invert_y else y_lo + 0.04 * (y_hi - y_lo)
    ax.plot([xb, xb + length_um], [yb, yb], color=_c(mode, "ink"), lw=2, solid_capstyle="butt", zorder=10)
    text = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
    ax.annotate(text, (xb + length_um / 2, yb), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=_c(mode, "ink"), fontsize=8.5)  # fmt: skip


# ===================================================================================================== flux matrix
def flux_matrix(
    result: Any,
    *,
    value: str = "net",
    order: Sequence[str] | None = None,
    mask_improbable: bool = False,
    alpha: float = 0.05,
    use: str = "q_value",
    annotate: bool = True,
    label_min: float = 5.0,
    ax: Any = None,
    style: str | None = "dark",
    title: str | None = None,
) -> Any:
    """TPT flux between every pair of states as a bubble matrix (sciCSR's ``plotFluxMatrix``).

    Parameters
    ----------
    result
        :class:`spaclone.tl.FluxResult` of :func:`spaclone.tl.tpt_flux`.
    value
        ``"net"`` or ``"gross"`` flux, in percent of its total (the colour).
    order
        State order on both axes (default: the result's order).
    mask_improbable
        Hide pairs whose target precedes their source in ``order`` (for isotypes in locus order: switches back
        towards Cμ, which CSR cannot make), as sciCSR does by default.
    alpha
        Pairs with ``use`` below it are outlined and labelled.
    use
        ``"q_value"`` (default) or ``"p_value"``: the bubble size is ``-log10`` of it (capped at 3).
    annotate
        Write the flux share of the outlined pairs and of pairs carrying at least ``label_min`` percent.
    label_min
        Share (percent) from which a pair is labelled whatever its significance.
    ax, style, title
        Axes, ``"dark"`` (house style, default) or None / ``"light"``, title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.lines import Line2D

    mode = _mode(style)
    if value not in ("net", "gross"):
        raise ValueError(f"value must be 'net' or 'gross', got {value!r}")
    mat = result.net if value == "net" else result.gross
    if mat is None:
        raise ValueError("flux_matrix: the result has no flux matrix")
    states = list(order) if order is not None else [str(s) for s in mat.index]
    missing = [s for s in states if s not in mat.index]
    if missing:
        raise ValueError(f"order: unknown states {missing}")
    tab = result.table.set_index(["from_state", "to_state"])
    pos = {s: i for i, s in enumerate(states)}
    xs, ys, fl, sig = [], [], [], []
    for a in states:
        for b in states:
            if a == b or (mask_improbable and pos[b] < pos[a]):
                continue
            xs.append(pos[a])
            ys.append(pos[b])
            fl.append(float(mat.loc[a, b]))
            pv = tab[use].get((a, b), np.nan) if use in tab.columns else np.nan
            sig.append(float(pv))
    fl, sig = np.asarray(fl), np.asarray(sig)
    has_p = np.isfinite(sig).any()
    size = np.where(np.isfinite(sig), np.clip(-np.log10(np.maximum(sig, 1e-300)), 0, 3), 0.0) if has_p else None
    area = 40 + 110 * (size if size is not None else np.full(len(fl), 1.5))
    m = len(states)
    ax = _axes(ax, mode, (1.1 * m + 2.6, 1.0 * m + 1.6))
    cm = _ramp_cmap("seq", mode)
    vmax = max(float(np.nanmax(fl)) if len(fl) else 1.0, 1e-12)
    from matplotlib.colors import PowerNorm

    sc = ax.scatter(xs, ys, s=area, c=fl, cmap=cm, norm=PowerNorm(0.5, vmin=0, vmax=vmax), linewidths=0, zorder=2)
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    hit = np.isfinite(sig) & (sig < alpha) if has_p else np.zeros(len(fl), dtype=bool)
    if hit.any():
        ax.scatter(
            xs_a[hit],
            ys_a[hit],
            s=area[hit] + 70,
            facecolors="none",
            edgecolors=_c(mode, "ink"),
            linewidths=1.2,
            zorder=3,
        )
    if annotate:
        lab = hit | (fl >= label_min)
        for x, y, f in zip(xs_a[lab], ys_a[lab], fl[lab], strict=True):
            ax.annotate(
                f"{f:.0f}%",
                (x, y),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
                color=_c(mode, "ink"),
                zorder=4,
            )
    ax.set_xticks(range(m), states, rotation=45, ha="left")
    ax.set_yticks(range(m), states)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("from")
    ax.set_ylabel("to")
    ax.set_xlim(-0.6, m - 0.4)
    ax.set_ylim(m - 0.4, -0.6)
    ax.grid(True, color=_c(mode, "grid"), linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.03, shrink=0.55, aspect=18)
    cb_label = f"{value} flux (% of total)"
    cb.set_label("" if title and title.strip().lower() == cb_label.lower() else cb_label, color=_c(mode, "muted"),
                 fontsize=8.5)  # fmt: skip
    cb.ax.tick_params(colors=_c(mode, "muted"), labelcolor=_c(mode, "muted"), labelsize=8, length=0)
    cb.outline.set_visible(False)
    if has_p:
        lab = "q" if use == "q_value" else "p"
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=np.sqrt(40 + 110 * k),
                markerfacecolor=_c(mode, "muted"),
                markeredgewidth=0,
                label=f"{lab} = {10**-k:g}",
            )
            for k in (0, 1, 2, 3)
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=_c(mode, "ink"),
                label=f"{lab} < {alpha:g}",
            )
        )
        leg = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=5,
            frameon=False,
            fontsize=8,
            columnspacing=1.2,
            handletextpad=0.3,
            borderaxespad=0,
        )
        for t in leg.get_texts():
            t.set_color(_c(mode, "ink"))
    if title:
        ax.set_title(title, color=_c(mode, "ink"), pad=40, loc="left", fontsize=11)
    return ax


# ===================================================================================================== stationary
def _lumped_stationary(model: Any) -> pd.DataFrame:
    """States table of a :class:`spaclone.tl.TransitionModel`: units, unit share and lumped stationary probability."""
    if model.states is None:
        raise ValueError(
            "stationary_distribution: the TransitionModel has no states; build it with "
            "spaclone.tl.transition_model(..., states=...) or pass the result of spaclone.tl.tpt_flux"
        )
    cat = pd.Categorical(model.states)
    codes = np.asarray(cat.codes)
    ok = codes >= 0
    m = len(cat.categories)
    pi = np.asarray(model.pi, dtype=float)
    n = np.bincount(codes[ok], minlength=m)
    return pd.DataFrame(
        {
            "n_units": n,
            "unit_share": n / max(int(n.sum()), 1),
            "stationary": np.bincount(codes[ok], weights=pi[ok], minlength=m),
            "stationary_lo": np.nan,
            "stationary_hi": np.nan,
        },
        index=pd.Index([str(c) for c in cat.categories], name="state"),
    )


def stationary_distribution(
    result: Any,
    *,
    order: Sequence[str] | None = None,
    show_share: bool = True,
    palette: Mapping[str, str] | None = None,
    ax: Any = None,
    style: str | None = "dark",
    title: str | None = None,
) -> Any:
    """Lumped stationary distribution per state with bootstrap 95% intervals (sciCSR's ``plotStationaryDistribution``).

    Parameters
    ----------
    result
        :class:`spaclone.tl.FluxResult` of :func:`spaclone.tl.tpt_flux` (its ``states`` table, with bootstrap 95%
        intervals), or a :class:`spaclone.tl.TransitionModel` built with ``states`` (the stationary distribution of
        its chain lumped by state, without intervals).
    order
        State order (default: the result's order).
    show_share
        Mark the observed share of units in each state (hollow circle): a state whose stationary probability exceeds
        its share of units is one the chain accumulates in.
    palette
        Colours by state (defaults: house colours of regions and classes, then the fixed categorical slots).
    ax, style, title
        Axes, ``"dark"`` (house style, default) or None / ``"light"``, title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from spaclone.tl.transitions import TransitionModel

    mode = _mode(style)
    if isinstance(result, TransitionModel):
        st = _lumped_stationary(result)
    else:
        st = getattr(result, "states", None)
        if st is None and hasattr(result, "states"):
            raise ValueError("stationary_distribution: the result has no states table")
        if not isinstance(st, pd.DataFrame):
            raise TypeError(
                "stationary_distribution expects the result of spaclone.tl.tpt_flux (a FluxResult, whose states "
                "table has bootstrap intervals) or a spaclone.tl.TransitionModel built with states, got "
                f"{type(result).__name__}"
            )
    every = [str(s) for s in st.index]
    st = st.set_axis(every)
    states = [str(s) for s in order] if order is not None else every
    st = st.loc[states]
    cols = _state_colors(every, mode, palette)  # every state of the result
    ax = _axes(ax, mode, (max(3.0, 0.9 * len(states) + 1.6), 3.6))
    x = np.arange(len(states))
    ax.bar(
        x,
        st["stationary"],
        width=0.62,
        color=[cols[s] for s in states],
        edgecolor=_c(mode, "bg"),
        linewidth=2,
        zorder=2,
    )
    lo = (st["stationary"] - st["stationary_lo"]).clip(lower=0).to_numpy()
    hi = (st["stationary_hi"] - st["stationary"]).clip(lower=0).to_numpy()
    if np.isfinite(lo).any():
        ax.errorbar(
            x, st["stationary"], yerr=[lo, hi], fmt="none", ecolor=_c(mode, "ink"), elinewidth=1.2, capsize=0, zorder=3
        )
    handles = [Patch(facecolor=_c(mode, "muted"), label="stationary probability")]
    if show_share:
        ax.scatter(
            x, st["unit_share"], s=46, facecolors=_c(mode, "bg"), edgecolors=_c(mode, "ink"), linewidths=1.4, zorder=4
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=_c(mode, "bg"),
                markeredgecolor=_c(mode, "ink"),
                label="share of units",
            )
        )
    ax.set_xticks(x, states, rotation=30 if max(len(s) for s in states) > 4 else 0, ha="right")
    ax.set_ylabel("probability")
    ax.set_ylim(
        0,
        max(
            float(np.nanmax(st["stationary_hi"].fillna(st["stationary"]))),
            float(st["unit_share"].max()) if show_share else 0.0,
        )
        * 1.15
        + 1e-9,
    )
    ax.grid(True, axis="y", color=_c(mode, "grid"), linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(_c(mode, "ink"))
    if title:
        ax.set_title(title, color=_c(mode, "ink"), loc="left", fontsize=11)
    return ax
