"""Plots of the spot-level transcriptome: spatial maps of genes, cell types and regions; DE volcano and dot plots.

Every function takes ``ax=None`` and returns the Axes; nothing is shown or saved. ``style="dark"`` is the house
style (black background, B blue, T pink, ASC amber); the default is a light figure. Category colours follow the
entity, never its rank: a class keeps its colour whatever else is on the map, and a class without a house colour takes
a slot by its position in the column's full list of categories, not among the classes drawn. Maps follow the image
convention (y increases downwards) unless ``invert_y=False``.

Palette notes: every colour comes from :mod:`spaclone.pl._colors` (validated with the dataviz palette checker on black
and on white). Regions have their own set (:data:`spaclone.pl._colors.REGION_COLORS`): follicle blue, germinal centre
(GC) orange, T zone magenta, ASC area violet and red pulp green, every pair at least 15 OKLab Delta E apart with full
colour vision and 8 under protanopia and deuteranopia, in both modes; a column of region labels (or named like
``region``) is drawn with it. Maps of the nine cell classes (B blue, T pink, ASC amber, GC B aqua, myeloid orange,
stromal green, Tfh violet, erythroid red) cannot keep every pair apart, so read them with the legend, or fold minor
classes with ``groups=``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import sparse
from scipy.spatial import cKDTree

from spaclone.pl._colors import (
    CELL_COLORS,
    CHROME,
    GREY_LABELS,
    RAMPS,
    REGION_ALIASES,
    REGION_COLORS,
    SLOTS,
    fixed_colors,
    named_colors,
    vocabulary,
)
from spaclone.pl._frame import SectionFrame, attach, frame_of

__all__ = ["HOUSE_COLORS", "gex_dotplot", "gex_map", "gex_regions", "gex_volcano"]

_SURFACE = {m: CHROME[m]["bg"] for m in CHROME}
_INK = {m: CHROME[m]["ink"] for m in CHROME}
_MUTED = {m: CHROME[m]["muted"] for m in CHROME}
_NEUTRAL = {m: CHROME[m]["unassigned"] for m in CHROME}
_BACKGROUND = {m: CHROME[m]["background"] for m in CHROME}

#: Fixed colours of the built-in cell classes and regions (dark, light). Region labels take the validated region set
#: (:data:`spaclone.pl._colors.REGION_COLORS`); ``"ASC"`` here is the cell class (amber), and a region map draws the
#: ASC area in its region colour (violet).
HOUSE_COLORS: dict[str, tuple[str, str]] = dict(CELL_COLORS) | {
    name: (
        REGION_COLORS["dark"][REGION_ALIASES.get(name, name)],
        REGION_COLORS["light"][REGION_ALIASES.get(name, name)],
    )
    for name in ("follicle", "T zone", "GC", "PC", "red pulp")
}
#: Fixed slot order for classes without a house colour (the reference categorical order), as (dark, light) pairs.
_SLOTS = list(zip(SLOTS["dark"], SLOTS["light"], strict=True))
_GREY_LABELS = set(GREY_LABELS)


def _mode(style: str | None) -> str:
    if style not in (None, "dark", "light"):
        raise ValueError(f"style must be None, 'light' or 'dark', got {style!r}")
    return "dark" if style == "dark" else "light"


def _prepare_ax(ax: Axes | None, mode: str, *, figsize=(8, 8)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_SURFACE[mode])
    ax.figure.set_facecolor(_SURFACE[mode])
    for sp in ax.spines.values():
        sp.set_color(_MUTED[mode])
    ax.tick_params(colors=_MUTED[mode], labelcolor=_MUTED[mode])
    ax.xaxis.label.set_color(_INK[mode])
    ax.yaxis.label.set_color(_INK[mode])
    ax.title.set_color(_INK[mode])
    return ax


def _category_colors(
    cats: Sequence[str],
    mode: str,
    palette: Mapping[str, str] | None,
    vocab: Sequence[str] | None = None,
    column: str | None = None,
) -> dict[str, str]:
    """Colours of ``cats``.

    Palette overrides, grey labels, house colours, then free slots by position in ``vocab`` (the column's full category
    list), so a class never takes its colour from the classes drawn beside it.
    """
    k = 0 if mode == "dark" else 1
    full = list(dict.fromkeys([*(vocab if vocab is not None else ()), *cats]))
    named = named_colors(full, mode, column=column)  # region set for a region column, else cell classes
    grey = {
        c: (_BACKGROUND[mode] if c in ("other", "background") else _NEUTRAL[mode]) for c in full if c in _GREY_LABELS
    }
    slots = tuple(s[k] for s in _SLOTS)
    out = fixed_colors(full, mode=mode, named=named, neutral=grey, palette=palette, slots=slots, beyond=_NEUTRAL[mode])
    return {c: out[c] for c in cats}


def _seq_cmap(mode: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"spaclone_seq_{mode}", list(RAMPS["seq"][mode]))


def _point_size(ax: Axes, xy: np.ndarray, size: float | None) -> float:
    if size is not None:
        return float(size)
    sub = xy if len(xy) <= 20000 else xy[np.random.default_rng(0).choice(len(xy), 20000, replace=False)]
    d, _ = cKDTree(sub).query(sub, k=2) if len(sub) > 1 else (np.array([[0, 1.0]]), None)
    step = float(np.median(d[:, 1])) if len(sub) > 1 else 1.0
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    return max((w_pt / abs(x1 - x0) * step) ** 2, 0.05)


def _frame(ax: Axes, fr: SectionFrame, invert_y: bool = True) -> None:
    fr.set_view(ax, invert_y=invert_y)  # the main tissue; image convention by default: y increases downwards
    ax.set_xticks([])
    ax.set_yticks([])


def _values(adata: ad.AnnData, key: str, layer: str | None) -> tuple[np.ndarray, bool]:
    """Values to colour by: an obs column (categorical or numeric) or a gene; returns (values, is_categorical)."""
    if key in adata.obs:
        v = adata.obs[key]
        if (
            isinstance(v.dtype, pd.CategoricalDtype)
            or v.dtype == object
            or pd.api.types.is_string_dtype(v)
            or v.dtype == bool
        ):
            return v.astype(object).to_numpy(), True
        return v.to_numpy(dtype=float), False
    if key in adata.var_names:
        m = adata.layers[layer] if layer is not None else adata.X
        col = m[:, adata.var_names.get_loc(key)]
        col = col.toarray() if sparse.issparse(col) else np.asarray(col)
        return np.asarray(col, dtype=float).ravel(), False
    raise KeyError(f"{key!r} is neither an obs column nor a gene")


def _scale_bar(ax: Axes, mode: str, length_um: float | None) -> None:
    """Scale bar in the lower right corner of the display, whichever way the y axis runs."""
    x0, x1 = sorted(ax.get_xlim())
    ylim = ax.get_ylim()
    y_bottom, y_top = ylim  # display bottom and top in data units (y_bottom > y_top when y runs downwards)
    span = x1 - x0
    if length_um is None:
        nice = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        length_um = float(nice[np.argmin(np.abs(nice - span / 6))])
    xs = x1 - 0.04 * span - length_um
    ys = y_bottom + 0.04 * (y_top - y_bottom)
    ax.plot([xs, xs + length_um], [ys, ys], color=_INK[mode], lw=2, solid_capstyle="butt", zorder=5)
    from matplotlib import patheffects

    ax.text(
        xs + length_um / 2,
        ys + 0.015 * (y_top - y_bottom),
        f"{length_um:g} µm",
        color=_INK[mode],
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=5,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground=_SURFACE[mode])],  # readable over the data
    )


def gex_map(
    adata: ad.AnnData,
    color: str,
    *,
    ax: Axes | None = None,
    style: str | None = "dark",
    size: float | None = None,
    palette: Mapping[str, str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    layer: str | None = None,
    mask: Any = None,
    legend: bool = True,
    scale_bar_um: float | None = None,
    title: str | None = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    spatial_key: str = "spatial",
) -> Axes:
    """Spatial map of a gene, a cell type, a region or any ``obs`` column, in µm.

    Parameters
    ----------
    adata
        AnnData with ``obsm[spatial_key]`` in µm.
    color
        ``obs`` column (categorical: one colour per class; numeric: sequential colour map) or gene name (values of
        ``layer`` or ``X``; run :func:`spaclone.gex.preprocess` for log-normalised values).
    ax
        Axes to draw on (default: a new 8 × 8 inch figure).
    style
        ``None`` / ``"light"`` (white) or ``"dark"`` (house style: black background).
    size
        Marker area in points²; default: one unit spacing wide.
    palette
        ``{class: colour}`` overrides for categorical maps. Without one, a class has its house colour
        (:data:`HOUSE_COLORS`) or the slot of its position in the column's full category list (the declared
        categories, else the sorted values of every unit), so masking units or folding classes never repaints the
        others; classes past the eighth free slot are grey.
    groups
        Fold classes for display, e.g. ``{"B lineage": ["B", "GC_B", "ASC"]}``; each folded class is listed and drawn
        once.
    cmap, vmin, vmax
        Colour map and limits of numeric maps (default: one-hue sequential map, 1st–99th percentile).
    layer
        Layer of gene values.
    mask
        Boolean array: draw only these units (others are faint background).
    legend
        Draw the legend (categorical) or the colour bar (numeric).
    scale_bar_um
        Length of the scale bar (default: a round length near a sixth of the width; 0 = none).
    title
        Axes title (default ``color``; ``""`` for none). The legend or colour bar is titled ``color`` only when the
        axes has no title, so a name is never printed twice.
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
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mode = _mode(style)
    ax = _prepare_ax(ax, mode)
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    fr = SectionFrame(xy, rotate, crop=crop)
    attach(ax, fr)
    xy = fr(xy)
    values, categorical = _values(adata, color, layer)
    show = np.ones(adata.n_obs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    ax_title = title if title is not None else color
    _frame(ax, fr, invert_y=invert_y)
    s = _point_size(ax, xy, size)
    if (~show).any():
        ax.scatter(xy[~show, 0], xy[~show, 1], s=s, c=_BACKGROUND[mode], lw=0, rasterized=True, zorder=0)
    if categorical:
        lab = np.array(
            ["nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in values], dtype=object
        )
        col = adata.obs[color] if color in adata.obs else pd.Series(lab)
        vocab = [c for c in vocabulary(col) if c != "nan"]  # every class of the column, not only those drawn
        folded = {str(m): str(name) for name, members in (groups or {}).items() for m in members}
        lab = np.array([folded.get(c, c) for c in lab], dtype=object)
        vocab = list(dict.fromkeys(folded.get(c, c) for c in [*map(str, groups or {}), *vocab]))
        drawn = set(lab[show])
        cats = [c for c in vocab if c in drawn] + sorted(drawn - set(vocab))  # each class once
        colors = _category_colors(cats, mode, palette, vocab, column=color)
        grey = [c for c in cats if colors[c] in (_NEUTRAL[mode], _BACKGROUND[mode])]
        handles = {}
        for c in grey + [c for c in cats if c not in grey]:  # neutral classes underneath
            m = show & (lab == c)
            handles[c] = ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=s,
                c=colors[c],
                lw=0,
                rasterized=True,
                zorder=1 if c in grey else 2,
                label=f"{c} ({int(m.sum()):,})",
            )
        if legend:
            order = [c for c in cats if c not in grey] + grey  # classes in category order, neutral ones last
            leg = ax.legend(
                handles=[handles[c] for c in order],
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                markerscale=max(1.0, 6 / np.sqrt(s)),
                labelcolor=_INK[mode],
                fontsize=9,
            )
            if not ax_title:  # the axes title already names the variable
                leg.set_title(color, prop={"size": 9})
                leg.get_title().set_color(_INK[mode])
    else:
        v = values[show]
        fin = np.isfinite(v)
        lo = np.nanpercentile(v[fin], 1) if vmin is None and fin.any() else (vmin or 0.0)
        hi = np.nanpercentile(v[fin], 99) if vmax is None and fin.any() else (vmax or 1.0)
        hi = hi if hi > lo else lo + 1e-9
        order = np.argsort(np.where(fin, v, -np.inf), kind="stable")  # high values on top
        pts = xy[show][order]
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=s,
            c=v[order],
            cmap=cmap or _seq_cmap(mode),
            norm=Normalize(lo, hi),
            lw=0,
            rasterized=True,
            zorder=2,
        )
        if legend:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=_MUTED[mode], labelcolor=_INK[mode], labelsize=8)
            if not ax_title:
                cb.set_label(color, color=_INK[mode], fontsize=9)
    if scale_bar_um != 0:
        _scale_bar(ax, mode, scale_bar_um)
    ax.set_title(ax_title, color=_INK[mode], fontsize=11, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def gex_regions(
    adata: ad.AnnData,
    *,
    region_key: str = "region",
    instance_key: str | None = "region_id",
    ax: Axes | None = None,
    style: str | None = "dark",
    size: float | None = None,
    palette: Mapping[str, str] | None = None,
    label_instances: bool = True,
    title: str | None = None,
    invert_y: bool = True,
    rotate: bool | float = True,
    spatial_key: str = "spatial",
) -> Axes:
    """Map of called regions with every instance labelled at its centroid (labels are the secondary encoding).

    Regions keep the fixed, validated region colours (:data:`spaclone.pl._colors.REGION_COLORS`): follicle blue,
    germinal centre (GC) orange, T zone magenta, ASC area violet, red pulp green; unassigned units grey.

    Parameters
    ----------
    adata
        AnnData after :func:`spaclone.gex.call_regions`.
    region_key, instance_key
        ``obs`` columns of the region label and the instance id.
    ax, style, size, palette, title, invert_y, rotate, spatial_key
        As :func:`gex_map`.
    label_instances
        Write each instance id (without its kind prefix) at the instance centroid.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mode = _mode(style)
    ax = gex_map(
        adata,
        region_key,
        ax=ax,
        style=style,
        size=size,
        palette=palette,
        title=title or region_key,
        invert_y=invert_y,
        rotate=rotate,
        spatial_key=spatial_key,
    )
    if label_instances and instance_key is not None and instance_key in adata.obs:
        xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
        fr = frame_of(ax)
        xy = fr(xy) if fr is not None else xy
        ids = adata.obs[instance_key].astype(object).to_numpy()
        ok = ~pd.isna(ids)
        if ok.any():
            frame = pd.DataFrame({"id": ids[ok].astype(str), "x": xy[ok, 0], "y": xy[ok, 1]})
            for rid, d in frame.groupby("id", sort=False):
                ax.text(
                    d.x.mean(),
                    d.y.mean(),
                    rid.split("|")[-1].rsplit("_", 1)[-1],
                    color=_INK[mode],
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=6,
                    fontweight="bold",
                )
    return ax


def gex_volcano(
    result: Any,
    *,
    group: str | None = None,
    ax: Axes | None = None,
    style: str | None = "dark",
    q: float = 0.05,
    lfc: float = 1.0,
    label_top: int = 10,
) -> Axes:
    """Volcano plot of :func:`spaclone.gex.rank_genes_regions` (log2 fold change vs −log10 p).

    Parameters
    ----------
    result
        The ``TestResult`` (or its table).
    group
        Region class to show (default: the first in the table).
    ax, style
        As :func:`gex_map`.
    q, lfc
        Genes with q-value < ``q`` and absolute log2 fold change ≥ ``lfc`` are coloured (up blue, down orange).
    label_top
        Label this many genes with the smallest p-values among the coloured ones.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mode = _mode(style)
    ax = _prepare_ax(ax, mode, figsize=(6, 5))
    tab = result.table if hasattr(result, "table") else pd.DataFrame(result)
    if group is None and len(tab):
        group = str(tab["group"].iloc[0])
    t = tab[tab["group"].astype(str) == str(group)] if group is not None else tab
    x = t["log2fc"].to_numpy(dtype=float)
    y = -np.log10(np.clip(t["p_value"].to_numpy(dtype=float), 1e-300, 1))
    sig = (t["q_value"].to_numpy(dtype=float) < q) & (np.abs(x) >= lfc)
    k = 0 if mode == "dark" else 1
    up, down = _SLOTS[0][k], _SLOTS[1][k]
    ax.scatter(x[~sig], y[~sig], s=8, c=_NEUTRAL[mode], lw=0, zorder=1)
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s=10, c=up, lw=0, zorder=2, label=f"higher in {group}")
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s=10, c=down, lw=0, zorder=2, label="lower")
    if label_top and sig.any():
        idx = np.flatnonzero(sig)[np.argsort(-y[sig])][:label_top]
        for i in idx:
            ax.annotate(
                str(t["gene"].iloc[i]),
                (x[i], y[i]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=_INK[mode],
            )
    for v in (-lfc, lfc):
        ax.axvline(v, color=_MUTED[mode], lw=0.6, ls=(0, (2, 2)), zorder=0)
    ref = str(t["reference"].iloc[0]) if len(t) else "reference"
    ax.set_xlabel(f"log2 fold change ({group} vs {ref})")
    ax.set_ylabel("−log10 p")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=_INK[mode], loc="upper left")
    if leg is not None:
        leg.set_zorder(5)
    return ax


def gex_dotplot(
    adata: ad.AnnData,
    markers: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    groupby: str,
    ax: Axes | None = None,
    style: str | None = "dark",
    layer: str | None = None,
    standard_scale: bool = True,
) -> Axes:
    """Dot plot of marker genes per group: dot area = share of units with the gene detected, colour = mean value.

    Parameters
    ----------
    adata
        AnnData (log-normalised ``X`` after :func:`spaclone.gex.preprocess`, or ``layer``).
    markers
        Genes, or ``{module: genes}`` (modules are separated by gaps on the x axis).
    groupby
        ``obs`` column of the groups (clusters, cell types, regions).
    ax, style
        As :func:`gex_map`.
    layer
        Layer of values (default ``X``).
    standard_scale
        Scale each gene's group means to 0–1.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mode = _mode(style)
    mods = dict(markers) if isinstance(markers, Mapping) else {"": list(markers)}
    genes, xpos, ticks, x = [], [], [], 0.0
    for name, gs in mods.items():
        present = [g for g in gs if g in adata.var_names]
        if not present:
            continue
        if ticks:
            x += 0.6
        ticks.append((x + (len(present) - 1) / 2, name))
        for g in present:
            genes.append(g)
            xpos.append(x)
            x += 1
    if not genes:
        raise ValueError("none of the marker genes is in the data")
    m = adata.layers[layer] if layer is not None else adata.X
    sub = m[:, adata.var_names.get_indexer(genes)]
    sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    grp = adata.obs[groupby].astype(str).to_numpy()
    col = adata.obs[groupby]
    cats = list(col.cat.categories.astype(str)) if isinstance(col.dtype, pd.CategoricalDtype) else sorted(set(grp))
    cats = [c for c in cats if (grp == c).any()]
    mean = np.array([sub[grp == c].mean(axis=0) for c in cats])
    frac = np.array([(sub[grp == c] > 0).mean(axis=0) for c in cats])
    if standard_scale:
        lo, hi = mean.min(axis=0), mean.max(axis=0)
        mean = (mean - lo) / np.where(hi > lo, hi - lo, 1)
    ax = _prepare_ax(ax, mode, figsize=(max(4, 0.35 * len(genes) + 2), max(3, 0.35 * len(cats) + 1.5)))
    gx, gy = np.meshgrid(np.asarray(xpos), np.arange(len(cats)))
    sc = ax.scatter(
        gx.ravel(),
        gy.ravel(),
        s=(frac.ravel() * 120) + 0.5,
        c=mean.ravel(),
        cmap=_seq_cmap(mode),
        vmin=0,
        vmax=1 if standard_scale else None,
        lw=0,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(genes, rotation=90, fontsize=8, color=_INK[mode])
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=8, color=_INK[mode])
    ax.set_ylim(len(cats) - 0.5, -0.5)
    if any(name for _, name in ticks):
        top = ax.secondary_xaxis("top")
        top.set_xticks([p for p, _ in ticks])
        top.set_xticklabels([n for _, n in ticks], fontsize=8, color=_INK[mode])
        top.tick_params(length=0)
        for sp in top.spines.values():
            sp.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("mean (scaled)" if standard_scale else "mean", color=_INK[mode], fontsize=8)
    cb.ax.tick_params(colors=_MUTED[mode], labelcolor=_INK[mode], labelsize=7)
    return ax
