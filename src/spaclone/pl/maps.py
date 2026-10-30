"""Share maps with hotspot outlines, and niche profiles with chance bands.

Both functions take ``ax=None`` (a new figure is made), return the ``Axes`` and never call ``show()`` or
``savefig()``. ``style="dark"`` is the project house style: black background, B / BCR in blue, T / TCR in pink,
hotspot outlines in yellow. Maps follow the image convention (y increases downwards) unless ``invert_y=False``.
Standardised layers (``z``, ``z_perm``) use a diverging map: blue below 0, a neutral grey at 0, red above.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spaclone.pl._colors import CHROME, DIVERGING, MARKS, OUTLINE_SLOTS, RAMPS, RECEPTOR_COLORS
from spaclone.pl._frame import SectionFrame, attach

__all__ = ["niche_profile", "share_map"]

_DARK, _LIGHT = (
    {"bg": CHROME[m]["bg"], "fg": CHROME[m]["ink"], "tissue": CHROME[m]["tissue"], "hot": MARKS[m]["hot"],
     "cold": MARKS[m]["cold"]}
    for m in ("dark", "light")
)  # fmt: skip
#: Sequential ramp of the share layer by receptor family (BCR blue to ice, TCR magenta).
_RAMPS = {"BCR": "csr", "TCR": "tcr"}
#: Colours of extra outlines, by their position among the outlines drawn without an explicit colour.
_OUTLINES = OUTLINE_SLOTS
_LAYER_LABEL = {
    "share": "share (smoothed)",
    "expected": "expected share",
    "z": "z (binomial residual)",
    "z_perm": "z (permutation-studentised)",
    "n_eff": "effective units",
}
_Z_LAYERS = ("z", "z_perm")


def _resolve_layer(result: Any, layer: str | None) -> str:
    """The layer to draw.

    ``layer``, or by default ``"z_perm"`` for a hotspot result whose thresholded map is the permutation-studentised
    one (clone-level nulls), else ``"share"``.
    """
    if layer is None:
        z_map = getattr(result, "z_map", None)
        perm = (getattr(result, "params", None) or {}).get("standardise") == "permutation"
        return "z_perm" if (z_map is not None and perm) else "share"
    if layer not in _LAYER_LABEL:
        raise ValueError(f"layer must be one of {sorted(_LAYER_LABEL)}")
    if layer == "z_perm" and getattr(result, "z_map", None) is None:
        raise ValueError("layer='z_perm' needs a HotspotResult (its z_map); a ShareMap has share, expected, z, n_eff")
    return layer


def _diverging(style: str | None) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    mode = "dark" if style == "dark" else "light"
    return LinearSegmentedColormap.from_list(f"spaclone_diverging_{mode}", DIVERGING[mode])


def _palette(style: str | None) -> dict[str, str]:
    if style not in (None, "light", "dark"):
        raise ValueError("style must be None, 'light' or 'dark'")
    return _DARK if style == "dark" else _LIGHT


def _axes(ax: Any, style: str | None, figsize: tuple[float, float]) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    pal = _palette(style)
    if style == "dark":
        ax.figure.set_facecolor(pal["bg"])
        ax.set_facecolor(pal["bg"])
        ax.tick_params(colors=pal["fg"])
        for sp in ax.spines.values():
            sp.set_color(pal["fg"])
        ax.xaxis.label.set_color(pal["fg"])
        ax.yaxis.label.set_color(pal["fg"])
        ax.title.set_color(pal["fg"])
    return ax


def _finite(values: Any) -> np.ndarray:
    """Float array with non-finite values (empty baseline bands, one-sided envelopes) set to NaN."""
    v = np.asarray(values, dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def _contour(
    ax: Any,
    mask: np.ndarray,
    grid: Any,
    color: str,
    lw: float,
    ls: str = "solid",
    halo: str | None = None,
    transform: Any = None,
) -> None:
    from matplotlib import patheffects
    from scipy import ndimage as ndi

    if not mask.any():
        return
    gx, gy = grid.centres()
    cs = ax.contour(
        gx,
        gy,
        ndi.gaussian_filter(mask.astype(float), 0.7).T,
        levels=[0.5],
        colors=[color],
        linewidths=lw,
        linestyles=ls,
        **({} if transform is None else {"transform": transform}),
    )
    if halo is not None:  # a thin surface-coloured rim keeps the outline readable over pixels of its own hue
        cs.set(path_effects=[patheffects.withStroke(linewidth=lw + 1.6, foreground=halo), patheffects.Normal()])


def share_map(
    result: Any,
    *,
    layer: str | None = None,
    ax: Any = None,
    style: str | None = "dark",
    receptor: str = "BCR",
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    quantiles: tuple[float, float] = (0.02, 0.995),
    hotspots: bool = True,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    colorbar: bool = True,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    title: str | None = None,
) -> Any:
    """Draw a share map (or another layer) on the eligible pixels, with hotspot outlines.

    Parameters
    ----------
    result
        A :class:`~spaclone.tl.ShareMap` or a :class:`~spaclone.tl.HotspotResult` (its map and hotspot labels).
    layer
        ``"share"``, ``"expected"``, ``"z"`` (the map's binomial residual), ``"z_perm"`` (the hotspot result's
        ``z_map``: the standardised map that was thresholded, permutation-studentised under clone-level nulls) or
        ``"n_eff"``. ``z`` layers use diverging colours centred at 0. Default: ``"z_perm"`` when the result is a
        :class:`~spaclone.tl.HotspotResult` with a permutation-studentised ``z_map`` (``standardise="permutation"``,
        the default with ``blocks``), whose binomial ``z`` would overstate the evidence; ``"share"`` otherwise.
    ax
        Axes to draw on; a new figure if None (8 inches wide, as tall as the section needs).
    style
        None / ``"light"`` (white background) or ``"dark"`` (house style: black background).
    receptor
        ``"BCR"`` (blue ramp) or ``"TCR"`` (pink ramp); ignored when ``cmap`` is given or for ``layer="z"``.
    cmap
        Matplotlib colormap.
    vmin, vmax
        Colour limits; by default the ``quantiles`` of the layer on eligible pixels (symmetric about 0 for ``z``
        layers).
    quantiles
        Quantiles used for the default colour limits.
    hotspots
        Outline the hotspots of a ``HotspotResult`` (hot solid, cold dashed).
    outlines
        Extra structures to outline, ``{name: boolean Raster}`` on the map's grid (e.g. T zones).
    outline_colors
        Colours of ``outlines`` by name; the others take a fixed colour by their position among the outlines without
        one.
    colorbar
        Add a colour bar (for ``z_perm`` with ``correction="maxT"``, a line marks the map-level hotspot threshold).
    invert_y
        Draw y increasing downwards (image convention, the default of every spaclone map); ``False`` puts y = 0 at
        the bottom.
    rotate
        Turn the section so that its long axis is horizontal, as every spaclone map does (``False``: keep the
        original coordinates; a number: that angle in degrees).
    crop
        Frame the main tissue: the largest pieces of the section (8-connected on a 50 µm raster of the map's pixels)
        that hold 95% of the pixels, with any piece next to them; specks far from the section (e.g. a stray cluster of
        records on the glass) are left out of the view and counted in an INFO log message. ``False`` frames every pixel.
    title
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    sm = result.map if getattr(result, "map", None) is not None else result
    labels = getattr(result, "labels", None)
    layer = _resolve_layer(result, layer)
    pal = _palette(style)
    grid = sm.grid
    if layer == "z_perm":
        vals = np.asarray(result.z_map.values, dtype=float)
        vals = np.where(sm.eligible & np.isfinite(vals), vals, np.nan)
    else:
        vals = np.where(sm.eligible, getattr(sm, layer), np.nan)
    fin = vals[np.isfinite(vals)]
    if layer in _Z_LAYERS:
        lim = float(np.quantile(np.abs(fin), quantiles[1])) if len(fin) else 1.0
        lim = lim if lim > 0 else 1.0
        vmin = -lim if vmin is None else vmin
        vmax = lim if vmax is None else vmax
        cmap = _diverging(style) if cmap is None else cmap
    elif vmin is None or vmax is None:
        q = np.quantile(fin, quantiles) if len(fin) else (0.0, 1.0)
        vmin = float(q[0]) if vmin is None else vmin
        vmax = float(q[1]) if vmax is None else vmax
    if cmap is None:
        stops = RAMPS[_RAMPS.get(receptor.upper(), "csr")]["dark" if style == "dark" else "light"]
        cmap = LinearSegmentedColormap.from_list(f"spaclone_{receptor}", stops)
    weighted = (~sm.eligible) & (sm.n_eff > 0)
    gx, gy = grid.centres()
    ii, jj = np.nonzero(sm.eligible | (sm.n_eff > 0))
    pts = np.column_stack([gx[ii], gy[jj]]) if len(ii) else np.column_stack([gx[[0, -1]], gy[[0, -1]]])
    fr = SectionFrame(pts, rotate, crop=crop)
    ax = _axes(ax, style, (8.0, float(np.clip((5.4 if colorbar else 7.4) * fr.aspect(0.75) + 1.3, 3.0, 9.0))))
    attach(ax, fr)
    tr = fr.affine() + ax.transData
    ax.imshow(
        np.where(weighted, 1.0, np.nan).T,
        origin="lower",
        extent=grid.extent,
        cmap=ListedColormap([pal["tissue"]]),
        interpolation="nearest",
        zorder=0,
        transform=tr,
    )
    im = ax.imshow(
        vals.T,
        origin="lower",
        extent=grid.extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=1,
        transform=tr,
    )
    # on a light diverging map the red / blue hotspot outlines would vanish into the pixels: outline in ink
    hot_c, cold_c = (pal["fg"], pal["fg"]) if (layer in _Z_LAYERS and style != "dark") else (pal["hot"], pal["cold"])
    if hotspots and labels is not None:
        tab = getattr(result, "table", None)
        lab = labels.values
        has = tab is not None and len(tab)
        hot_ids = tab.hotspot[tab.direction == "hot"].to_numpy() if has else []
        cold_ids = tab.hotspot[tab.direction == "cold"].to_numpy() if has else []
        _contour(ax, np.isin(lab, hot_ids), grid, hot_c, 1.4, halo=pal["bg"], transform=tr)
        _contour(ax, np.isin(lab, cold_ids), grid, cold_c, 1.2, "dashed", halo=pal["bg"], transform=tr)
    extra = _OUTLINES["dark" if style == "dark" else "light"]
    k = 0
    for name, mask in (outlines or {}).items():
        col = (outline_colors or {}).get(name)
        if col is None:
            col = extra[k % len(extra)]
            k += 1
        m = mask.values if hasattr(mask, "values") else np.asarray(mask)
        _contour(ax, m.astype(bool), grid, col, 0.9, halo=pal["bg"], transform=tr)
    half = 0.75 * float(grid.pixel_um) if hasattr(grid, "pixel_um") else 0.0
    # the main tissue: a detached speck far from the section does not shrink the map
    fr.set_view(ax, invert_y=invert_y, pad_um=half)
    ax.set_xlabel(fr.labels[0])
    ax.set_ylabel(fr.labels[1])
    if title:
        ax.set_title(title, color=pal["fg"])
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        label = _LAYER_LABEL[layer]
        if layer == "z_perm" and (getattr(result, "params", None) or {}).get("standardise") == "binomial":
            label = _LAYER_LABEL["z"]
        cb.set_label(label, color=pal["fg"])
        cb.ax.tick_params(colors=pal["fg"])
        cb.outline.set_edgecolor(pal["fg"])
        thr = (getattr(result, "threshold", None) or {}).get("hot")
        maxt = (getattr(result, "params", None) or {}).get("correction") == "maxT"
        if layer == "z_perm" and maxt and thr is not None and np.isfinite(thr) and vmin <= thr <= vmax:
            cb.ax.axhline(thr, color=pal["fg"], lw=1.2)
    return ax


def niche_profile(
    result: Any,
    *,
    ax: Any = None,
    style: str | None = "dark",
    receptor: str = "TCR",
    color: str | None = None,
    label: str | None = None,
    ratio: bool = True,
    band: bool = True,
    envelope: bool = True,
    min_baseline: int = 0,
    xlabel: str | None = None,
) -> Any:
    """Draw a niche profile (ratio to the baseline per band) with its pointwise band and global envelope.

    Parameters
    ----------
    result
        Result of :func:`spaclone.tl.niche_profile`.
    ax
        Axes to draw on; a new figure if None.
    style
        None / ``"light"`` or ``"dark"`` (house style).
    receptor
        ``"BCR"`` (blue) or ``"TCR"`` (pink) line colour; ignored when ``color`` is given.
    color
        Line colour.
    label
        Legend label of the profile.
    ratio
        Plot the ratio to the baseline (True) or the group's share per band (False).
    band
        Shade the pointwise chance band.
    envelope
        Draw the global (studentised, family-wise) envelope as dashed lines.
    min_baseline
        Leave out bands with fewer baseline units (ratios there are noisy).
    xlabel
        Axis label; by default the distance or nothing for compartments.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    tab = result.table
    pal = _palette(style)
    ax = _axes(ax, style, (7.0, 4.5))
    rec = RECEPTOR_COLORS["dark" if style == "dark" else "light"]
    col = color or rec.get(receptor.upper(), rec["TCR"])
    t = tab[tab.baseline_units >= min_baseline]
    numeric = len(t) > 0 and bool(np.isfinite(t.mid_um.to_numpy(dtype=float)).all())
    x = t.mid_um.to_numpy(dtype=float) if numeric else np.arange(len(t), dtype=float)
    cols = (
        ("ratio", "ratio_lo", "ratio_hi", "ratio_env_lo", "ratio_env_hi")
        if ratio
        else ("statistic", "null_lo", "null_hi", "env_lo", "env_hi")
    )
    y, lo, hi, elo, ehi = (_finite(t[c]) for c in cols)
    if band:
        if numeric:
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, label="chance (pointwise)")
        else:
            ax.vlines(x, lo, hi, color=col, alpha=0.3, lw=8, label="chance (pointwise)")
    if envelope:
        ax.plot(x, elo, ls="--", color=col, lw=1, alpha=0.8, label="global envelope")
        ax.plot(x, ehi, ls="--", color=col, lw=1, alpha=0.8)
    ax.plot(x, y, marker="o", ms=4, lw=2 if numeric else 0, color=col, label=label or "observed")
    if ratio:
        ax.axhline(1.0, color=pal["fg"], lw=0.8, ls=":")
    else:
        ax.plot(x, t.expected.to_numpy(dtype=float), marker="_", ms=14, lw=0, color=pal["fg"], label="baseline")
    if numeric:
        ax.axvline(0, color=pal["fg"], lw=0.8)
        ax.set_xlabel(xlabel or "signed distance to the border (µm)")
    else:
        ax.set_xticks(x, t.label.tolist())
        if xlabel:
            ax.set_xlabel(xlabel)
    ax.set_ylabel("ratio to baseline" if ratio else "share of the group")
    leg = ax.legend(frameon=False, fontsize=9)
    if style == "dark":
        for txt in leg.get_texts():
            txt.set_color(pal["fg"])
    return ax
