"""One-page figure of a location-trust report: the overall verdict, then the molecule level and the clone level."""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from spaclone.pl._colors import CHROME, VERDICT_COLORS

__all__ = ["trust_report"]

#: Surfaces, ink and verdict colours of the two styles (from :mod:`spaclone.pl._colors`).
_THEMES = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "ink": CHROME[mode]["ink"],
        "ink2": CHROME[mode]["ink2"],
        "grid": CHROME[mode]["grid"],
        "band": CHROME[mode]["panel"],
        **VERDICT_COLORS[mode],
    }
    for mode in ("light", "dark")
}
_LEVELS = (
    ("molecule", "Molecule-level placement", ("same_spot", "own_type_share", "umi_match")),
    ("clone", "Clone-level placement", ("colocalisation", "pairing", "partner_sharing")),
)
_LABELS = {
    "same_spot": "Same spot as its own transcript",
    "own_type_share": "On its own cell type",
    "umi_match": "UMI found as its own transcript",
    "colocalisation": "Same-clone pairs 20–100 µm apart",
    "pairing": "Heavy and light chain together",
    "partner_sharing": "Clone loci share a partner chain",
}
_SHOWN = ("signal", "not detected", "inconclusive", "untestable", "info")
_LEVEL_IN, _GROUP_IN, _ROW_IN, _AXIS_IN, _LINE_IN = 0.44, 0.34, 0.27, 0.62, 0.22


def _fmt_tick(v: float, _pos: int) -> str:
    return f"{v:g}"


def trust_report(report: Any, *, style: str = "dark", width: float = 10.0, compact: bool = False):
    """Plot a :class:`~spaclone.qc.TrustReport`: observed ÷ chance for every test row, by level and test.

    The title is the overall verdict; below it, the molecule level (same spot, own cell type, UMI match) and the clone
    level (heavy-chain co-localisation, heavy/light pairing, partner-chain sharing), each headed by its verdict. A dot
    is the observed statistic divided by its chance (placebo) value, coloured by the row's verdict (hollow: information
    only, e.g. light-chain co-localisation; no dot: untestable); the vertical line marks chance, and the text on the
    right gives the verdict and the BH q-value of the level's family.

    Parameters
    ----------
    report
        The result of :func:`spaclone.run`.
    style
        ``"dark"`` (the house style, default) or ``"light"``.
    width
        Figure width in inches; the height follows the number of rows.
    compact
        A card-sized summary instead: the verdict of each level, then one row per test (its main row: the heavy
        chain where there is one) with observed ÷ chance and the BH q-value, in large type.

    Returns
    -------
    matplotlib.figure.Figure
    """
    th = _THEMES[style]
    t = report.tests()
    t = t[t.test.isin(_LABELS) & t.verdict.isin(_SHOWN)].reset_index(drop=True)
    status = getattr(report, "status", {}) or {}
    if compact:
        return _compact(t, status, th, width)
    sections = []
    for level, title, tests in _LEVELS:
        groups = [(test, t[t.test == test]) for test in tests if (t.test == test).any()]
        sections.append((level, title, groups))
    n_rows = sum(max(sum(len(g) for _, g in groups), 1) for _, _, groups in sections)
    n_groups = sum(len(groups) for _, _, groups in sections)
    wrapped = textwrap.wrap(str(report.verdict).split(". ")[0].rstrip(".") + ".", int(width * 10.0))[:4]
    header_in = 0.30 + _LINE_IN * len(wrapped) + 0.30
    height = header_in + _LEVEL_IN * len(sections) + _GROUP_IN * n_groups + _ROW_IN * n_rows + _AXIS_IN
    fig = plt.figure(figsize=(width, height), facecolor=th["bg"])
    x0, x1 = 0.33, 0.74
    plot_h_in = height - header_in - _AXIS_IN
    ax = fig.add_axes((x0, _AXIS_IN / height, x1 - x0, plot_h_in / height), facecolor=th["bg"])
    ax.set_ylim(plot_h_in, 0)  # y in inches from the top of the plot area
    ratio = t.ratio.to_numpy(float)
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    lo = min(0.5, float(finite.min()) * 0.8) if len(finite) else 0.5
    hi = max(2.0, float(finite.max()) * 1.25) if len(finite) else 2.0
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.axvline(1, color=th["ink2"], lw=1, zorder=1)

    def fig_y(y_in: float) -> float:
        return 1 - (header_in + y_in) / height

    y = 0.0
    for level, title, groups in sections:
        st = str(status.get(level, "")) or "not tested"
        fig.text(
            0.03, fig_y(y + _LEVEL_IN * 0.55), title, color=th["ink"], fontsize=11.5, fontweight="bold", va="center"
        )
        fig.text(0.765, fig_y(y + _LEVEL_IN * 0.55), st, color=th.get(st, th["ink"]), fontsize=11, va="center",
                 fontweight="bold")  # fmt: skip
        y += _LEVEL_IN
        if not groups:
            fig.text(0.05, fig_y(y + _ROW_IN / 2), "no test could be run", color=th["ink2"], fontsize=9.5, va="center")
            y += _ROW_IN
            continue
        for test, g in groups:
            ax.axhspan(y, y + _GROUP_IN * 0.92, color=th["band"], lw=0, zorder=0)
            fig.text(0.04, fig_y(y + _GROUP_IN * 0.46), _LABELS[test], color=th["ink"], fontsize=10, va="center")
            y += _GROUP_IN
            for row in g.itertuples(index=False):
                yc = y + _ROW_IN / 2
                v = str(row.verdict)
                if np.isfinite(row.ratio) and row.ratio > 0 and v != "untestable":
                    if v == "info":
                        ax.scatter(row.ratio, yc, s=52, facecolors="none", edgecolors=th["info"], linewidths=1.4,
                                   zorder=3)  # fmt: skip
                    else:
                        ax.scatter(row.ratio, yc, s=58, color=th[v], edgecolors=th["bg"], linewidths=1.2, zorder=3)
                where = " · ".join(x for x in (str(row.locus), str(row.library_id)) if x and x not in ("all", "nan"))
                fig.text(0.06, fig_y(yc), where, color=th["ink2"], fontsize=9.5, va="center", ha="left")
                q = getattr(row, "q_value", np.nan)
                qtxt = "" if not np.isfinite(q) else f"q {q:.2g}"
                label = "light chain (info)" if v == "info" else v
                fig.text(0.765, fig_y(yc), label, color=th["ink"] if v != "untestable" else th["ink2"], fontsize=9.5,
                         va="center", ha="left")  # fmt: skip
                fig.text(0.92, fig_y(yc), qtxt, color=th["ink2"], fontsize=9.5, va="center", ha="left")
                y += _ROW_IN
    ticks = [v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= v <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(axis="x", colors=th["ink2"], labelsize=9, length=0, pad=4)
    ax.set_xlabel("observed ÷ chance (placebo)", color=th["ink2"], fontsize=9)
    top = 1 - 0.30 / height
    fig.text(0.03, top, "\n".join(wrapped), color=th["ink"], fontsize=12, va="top", ha="left", fontweight="bold",
             linespacing=1.25)  # fmt: skip
    sub = f"molecule level: {status.get('molecule', '')}   ·   clone level: {status.get('clone', '')}"
    fig.text(0.03, 1 - (0.30 + _LINE_IN * len(wrapped) + 0.12) / height, sub, color=th["ink2"], fontsize=9.5,
             va="top", ha="left")  # fmt: skip
    return fig


def _compact(t: Any, status: dict, th: dict, width: float):
    """The card-sized trust report: level verdicts, then one row per test with observed ÷ chance and q."""
    rows = []
    for level, title, tests in _LEVELS:
        picked = []
        for test in tests:
            g = t[t.test == test]
            if not len(g):
                continue
            main = g[g.verdict != "info"]
            main = main if len(main) else g
            heavy = main[main.locus.astype(str).str.startswith(("IGH", "TRB"))]
            main = heavy if len(heavy) else main
            picked.append((test, main.sort_values("q_value", na_position="last").iloc[0]))
        rows.append((level, title, picked))
    n_rows = sum(max(len(p), 1) for _, _, p in rows)
    height = 1.25 + 0.48 * len(rows) + 0.5 * n_rows + 0.15
    fig = plt.figure(figsize=(width, height), facecolor=th["bg"])

    def fy(y_in: float) -> float:
        return 1 - y_in / height

    x = 0.04
    for level, label in (("molecule", "Molecule level"), ("clone", "Clone level")):
        st = str(status.get(level, "")) or "not tested"
        fig.text(x, fy(0.42), label, color=th["ink2"], fontsize=12, va="center", ha="left")
        fig.text(x, fy(0.86), st, color=th.get(st, th["ink"]), fontsize=19, fontweight="bold", va="center", ha="left")
        x += 0.48
    y = 1.25
    for _level, title, picked in rows:
        fig.text(0.04, fy(y + 0.24), title.upper(), color=th["ink2"], fontsize=10, va="center", ha="left")
        fig.add_artist(plt.Line2D([0.04, 0.96], [fy(y + 0.45)] * 2, color=th["grid"], lw=0.8))
        y += 0.48
        if not picked:
            fig.text(0.04, fy(y + 0.25), "no test could be run", color=th["ink2"], fontsize=12, va="center")
            y += 0.5
            continue
        for test, r in picked:
            v = str(r.verdict)
            yc = y + 0.25
            fig_y = fy(yc)  # columns: test | observed ÷ chance | q | verdict
            fig.text(0.04, fig_y, _LABELS[test], color=th["ink"], fontsize=11.5, va="center", ha="left")
            ratio = float(r.ratio) if np.isfinite(r.ratio) else np.nan
            fig.text(0.62, fig_y, "" if not np.isfinite(ratio) else f"{ratio:.2g}×", color=th["ink"], fontsize=12,
                     va="center", ha="right", fontweight="bold")  # fmt: skip
            q = float(getattr(r, "q_value", np.nan))
            fig.text(0.645, fig_y, "" if not np.isfinite(q) else f"q {q:.2g}", color=th["ink2"], fontsize=11,
                     va="center", ha="left")  # fmt: skip
            fig.text(0.96, fig_y, v, color=th.get(v, th["ink"]), fontsize=11, va="center", ha="right")
            y += 0.5
    return fig
