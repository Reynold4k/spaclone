"""Plots of the molecule-level location tests: same-spot hits by clones per spot, and density against tissue distance.

Every function takes ``ax=None``, returns the Axes and never calls ``show()`` or ``savefig()``. Colours follow the
locus (the same locus has the same colour in every plot: IGH blue, IGK orange, IGL aqua, TRA yellow, TRB magenta, as
in :data:`spaclone.pl.LOCUS_COLORS`); GEX (cDNA) profiles are drawn in neutral ink.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spaclone.pl._colors import CHROME, LOCUS_SLOT, SLOTS, fixed_colors

__all__ = ["density_profile", "same_spot_multiplicity"]

# validated categorical palette (light / dark surface), slots in fixed order; colour follows the locus
_SLOTS = {mode: list(slots) for mode, slots in SLOTS.items()}
_LOCUS_SLOT = dict(LOCUS_SLOT)
_INK = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "fg": CHROME[mode]["ink"],
        "muted": CHROME[mode]["ink2"],
        "grid": CHROME[mode]["grid"],
    }
    for mode in ("light", "dark")
}


def _theme(style: str | None) -> dict[str, Any]:
    mode = "dark" if style == "dark" else "light"
    return {"mode": mode, "slots": _SLOTS[mode], **_INK[mode]}


def _colours(groups: Sequence[Any], theme: dict[str, Any]) -> dict[str, str]:
    """Colour of every group: loci in their fixed slot, other groups by sorted position (never by the groups drawn)."""
    vocab = sorted({str(g) for g in groups})
    named = {g: theme["slots"][_LOCUS_SLOT[g.upper()]] for g in vocab if g.upper() in _LOCUS_SLOT}
    return fixed_colors(vocab, mode=theme["mode"], named=named)


def _axes(ax: Any, theme: dict[str, Any], styled: bool, figsize: tuple[float, float]) -> Any:
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)
    if styled:
        if created:  # never repaint a figure the caller owns (it may hold panels in another style)
            ax.figure.set_facecolor(theme["bg"])
        ax.set_facecolor(theme["bg"])
        for sp in ax.spines.values():
            sp.set_color(theme["muted"])
        ax.tick_params(colors=theme["muted"], labelcolor=theme["fg"])
        ax.xaxis.label.set_color(theme["fg"])
        ax.yaxis.label.set_color(theme["fg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _table(result: Any, attr: str) -> pd.DataFrame:
    tab = getattr(result, attr, None)
    if tab is None:
        if isinstance(result, pd.DataFrame):
            return result
        raise TypeError(f"expected a result with a {attr!r} table or a DataFrame")
    return tab


def same_spot_multiplicity(
    result: Any,
    *,
    level: str = "spot",
    zone: Any = None,
    loci: Sequence[str] | None = None,
    p_det: float | None = None,
    conf_level: float = 0.95,
    ax: Any = None,
    style: str | None = "dark",
) -> Any:
    """Same-spot hits ÷ chance-expected hits by clones per spot, with the chance line at 1.

    Parameters
    ----------
    result
        :class:`spaclone.qc.SameSpotResult` (its ``by_multiplicity`` table) or that table.
    level
        ``"spot"`` (one record per spot, the independent observation) or ``"record"`` (every record counted, as a
        naive analysis would; barcode sinks then dominate the high classes).
    zone
        Zone to show; ``None`` sums over all zones.
    loci
        Loci to show (default all, in the table's order).
    p_det
        If given, also mark where each class would lie if every molecule sat on its own spot
        (``(E + p_det (n − E)) / E``), the size of the signal a genuine library would give.
    conf_level
        Level of the exact Poisson intervals of the hits.
    ax
        Axes to draw into (default: a new figure).
    style
        ``"dark"`` (the house style, default) or ``"light"`` (with a supplied ``ax`` the figure itself is left as it
        is); ``None`` keeps the current Matplotlib style.

    Returns
    -------
    matplotlib.axes.Axes
    """
    th = _theme(style)
    bm = _table(result, "by_multiplicity")
    locus_key = "locus" if "locus" in bm.columns else bm.columns[1]
    b = bm[bm["level"] == level]
    if zone is not None:
        b = b[b["zone"] == zone]
    if len(b) == 0:
        raise ValueError(f"no rows for level={level!r}, zone={zone!r}")
    cats = (
        list(b["multiplicity"].cat.categories)
        if isinstance(b["multiplicity"].dtype, pd.CategoricalDtype)
        else (list(dict.fromkeys(b["multiplicity"])))
    )
    g = b.groupby([locus_key, "multiplicity"], observed=True, sort=False)[["n", "hits", "expected"]].sum().reset_index()
    show = list(loci) if loci is not None else list(dict.fromkeys(g[locus_key]))
    created = ax is None
    ax = _axes(ax, th, style is not None, (5.0, 3.4))
    used = _colours(g[locus_key].astype(str), th)
    width = 0.6 / max(len(show), 1)
    a = 1 - conf_level
    for i, lc in enumerate(show):
        s = g[g[locus_key] == lc].set_index("multiplicity").reindex(cats).dropna(subset=["n"])
        if len(s) == 0:
            continue
        xpos = np.array([cats.index(c) for c in s.index], dtype=float) + (i - (len(show) - 1) / 2) * width
        h, e = s["hits"].to_numpy(float), s["expected"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = h / e
            lo = np.where(h > 0, stats.chi2.ppf(a / 2, 2 * h) / 2, 0.0) / e
            hi = stats.chi2.ppf(1 - a / 2, 2 * (h + 1)) / 2 / e
        col = used[str(lc)]
        ax.errorbar(xpos, ratio, yerr=[ratio - lo, hi - ratio], fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.2,
                    capsize=0, label=str(lc), zorder=3)  # fmt: skip
        if p_det is not None:
            n = s["n"].to_numpy(float)
            full = (e + p_det * (n - e)) / e
            ax.scatter(xpos, full, marker="_", s=120, color=col, linewidths=1.6, zorder=2)
    ax.axhline(1.0, color=th["muted"], lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.annotate("chance", xy=(1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=th["muted"])  # fmt: skip
    if p_det is not None:
        ax.annotate(f"— every molecule on its own spot (p_det {p_det:g})", xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(2, -2), textcoords="offset points", ha="left", va="top", fontsize=7,
                    color=th["muted"])  # fmt: skip
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("clones of the locus on the spot")
    ax.set_ylabel("hits ÷ chance-expected hits")
    ax.grid(axis="y", color=th["grid"], lw=0.6)
    ax.set_axisbelow(True)
    if len(show) > 1 or p_det is not None:
        leg = ax.legend(frameon=False, fontsize=8, title="locus", title_fontsize=8)
        for t in leg.get_texts():
            t.set_color(th["fg"])
        leg.get_title().set_color(th["fg"])
    if created:
        ax.figure.tight_layout()
    return ax


def density_profile(
    result: Any,
    *,
    groups: Sequence[str] | None = None,
    relative: bool = True,
    xlim_um: tuple[float, float] | None = None,
    ax: Any = None,
    style: str | None = "dark",
) -> Any:
    """Records (and GEX molecules) per µm² against the signed distance to the tissue edge.

    Parameters
    ----------
    result
        :func:`spaclone.qc.density_profile` result, or its table.
    groups
        Groups to draw (default every group: receptor loci in colour, GEX profiles in neutral ink).
    relative
        Plot the density relative to the tissue core (``relative``); else the density per µm².
    xlim_um
        Distance range shown (µm).
    ax
        Axes to draw into (default: a new figure).
    style
        ``"dark"`` (the house style, default) or ``"light"`` (with a supplied ``ax`` the figure itself is left as it
        is); ``None`` keeps the current Matplotlib style.

    Returns
    -------
    matplotlib.axes.Axes
    """
    th = _theme(style)
    t = _table(result, "table")
    ycol = "relative" if relative else "density"
    show = list(groups) if groups is not None else list(dict.fromkeys(t["group"]))
    created = ax is None
    ax = _axes(ax, th, style is not None, (5.0, 3.4))
    used = _colours([grp for grp in t["group"].astype(str).unique() if not grp.startswith("gex:")], th)
    dashes = [(0, ()), (0, (4, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (2, 2))]
    k_gex = 0
    for grp in show:
        s = t[t["group"] == grp].sort_values("shell_mid")
        y = s[ycol].to_numpy(float)
        ok = np.isfinite(y) & (y > 0)
        if not ok.any():
            continue
        if str(grp).startswith("gex:"):
            ax.plot(s["shell_mid"].to_numpy()[ok], y[ok], color=th["muted"], lw=1.4, ls=dashes[k_gex % len(dashes)],
                    label=f"GEX {grp[4:]}", zorder=2)  # fmt: skip
            k_gex += 1
        else:
            col = used[str(grp)]
            ax.plot(s["shell_mid"].to_numpy()[ok], y[ok], color=col, lw=1.8, marker="o", ms=3.5, label=str(grp),
                    zorder=3)  # fmt: skip
    ax.axvline(0.0, color=th["muted"], lw=0.9, zorder=1)
    ax.annotate("tissue edge", xy=(0.0, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=th["muted"])  # fmt: skip
    ax.set_yscale("log")
    if xlim_um is not None:
        ax.set_xlim(*xlim_um)
    ax.set_xlabel("signed distance to the tissue edge (µm; < 0 inside)")
    ax.set_ylabel("density ÷ tissue core" if relative else "per µm²")
    ax.grid(axis="y", color=th["grid"], lw=0.6, which="major")
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(th["fg"])
    if created:
        ax.figure.tight_layout()
    return ax
