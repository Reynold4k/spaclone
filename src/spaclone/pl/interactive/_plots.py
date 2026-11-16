"""Interactive forms of the main non-spatial plots: volcano, matrices, diversity, expansion, null distributions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone.pl import _colors as C
from spaclone.pl.interactive import _spec as P
from spaclone.pl.interactive._figure import InteractiveFigure
from spaclone.pl.interactive._util import call_text, check_style

__all__ = [
    "clonal_expansion",
    "clonal_scan",
    "csr_transitions",
    "diversity",
    "flux_matrix",
    "gex_volcano",
    "niche_profile",
    "overlap",
    "placebo",
    "read_support",
    "stationary_distribution",
    "trust_report",
]


def _axis(label: str | None = None, *, scale: str = "lin", values: Any = None, rng: Sequence[float] | None = None,
          cats: Sequence[str] | None = None, fmt: str | None = None, grid: bool = True, pad: float = 0.05,
          zero: bool = False, **kw: Any) -> dict[str, Any]:  # fmt: skip
    """An axis of a plot panel; ``values`` (data units) set the range when ``rng`` is not given."""
    a: dict[str, Any] = {"scale": scale, "grid": grid}
    if label:
        a["label"] = label
    if cats is not None:
        a["scale"] = "cat"
        a["cats"] = [str(c) for c in cats]
        a["grid"] = False
    elif rng is not None:
        a["range"] = [float(rng[0]), float(rng[1])]
    elif values is not None:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v) & ((v > 0) if scale == "log" else True)]
        if not len(v):
            lo, hi = (1.0, 10.0) if scale == "log" else (0.0, 1.0)
        elif scale == "log":
            lo, hi = float(v.min()) / (1 + pad * 4), float(v.max()) * (1 + pad * 4)
        else:
            lo, hi = float(v.min()), float(v.max())
            if zero:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            span = (hi - lo) or (abs(hi) or 1.0)
            lo, hi = lo - pad * span, hi + pad * span
        a["range"] = [lo, hi]
    if fmt:
        a["fmt"] = fmt
    a.update({k: v for k, v in kw.items() if v is not None})
    return a


def _raw(values: Any, scale: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, np.log10(v), np.nan)
    return v


def _plot_spec(title: str, call: str, style: str, *, x: dict, y: dict, aspect: float = 1.45, side: bool = True,
               refs: Sequence[dict] = (), margin: dict | None = None, zoom: str = "xy",
               panel_title: str | None = None) -> tuple[P.Spec, dict]:  # fmt: skip
    spec = P.Spec(style=style, title=title, call=call)
    axes = {"x": x, "y": y, "zoom": zoom}
    panel = spec.panel("plot", "plot", area=[0, 0, 1, 1], axes=axes, refs=list(refs) or None, margin=margin,
                       title=panel_title)  # fmt: skip
    spec.layout = {"cols": 1, "rows": 1, "aspect": aspect, "height_hint": 520, "side": None if side else "none",
                   "tools": ["pan", "zoom", "reset", "png"]}  # fmt: skip
    return spec, panel


def gex_volcano(
    result: Any,
    *,
    group: str | None = None,
    style: str = "dark",
    q: float = 0.05,
    lfc: float = 1.0,
    label_top: int = 10,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.gex_volcano`: every region class, a gene search, gene names on hover.

    Parameters
    ----------
    result, group, style, q, lfc, label_top
        As in :func:`spaclone.pl.gex_volcano` (``group``: the class shown first; the switcher lists every class).
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    mode = style
    tab = result.table if hasattr(result, "table") else pd.DataFrame(result)
    groups = [str(g) for g in pd.unique(tab["group"].astype(str))]
    if not groups:
        raise ValueError("gex_volcano: the table has no rows")
    first = str(group) if group is not None else groups[0]
    if first not in groups:
        raise KeyError(f"group {group!r} not in the table (groups: {groups})")
    xs = tab["log2fc"].to_numpy(dtype=float)
    ys = -np.log10(np.clip(tab["p_value"].to_numpy(dtype=float), 1e-300, 1))
    call = call_text("spaclone.pl.interactive.gex_volcano", result, group=group, q=q, lfc=lfc, style=style)
    spec, panel = _plot_spec(title or "differential expression", call, style,
                             x=_axis("log2 fold change", values=xs, zero=True), y=_axis("−log10 p", values=ys, zero=True),
                             refs=[{"x": -lfc, "dash": [3, 3]}, {"x": lfc, "dash": [3, 3]}])  # fmt: skip
    k = 0 if mode == "dark" else 1
    up, down = C.SLOTS[mode][0], C.SLOTS[mode][1]
    # genes that are not significant: the static grey, lifted on black so that the cloud stays visible behind the
    # coloured genes on a screen
    neutral = "#5e5d59" if mode == "dark" else C.CHROME[mode]["unassigned"]
    ch = C.CHROME[mode]
    gene_tables = {}
    for g in groups:
        t = tab[tab["group"].astype(str) == g].reset_index(drop=True)
        x = t["log2fc"].to_numpy(dtype=float)
        y = -np.log10(np.clip(t["p_value"].to_numpy(dtype=float), 1e-300, 1))
        qv = t["q_value"].to_numpy(dtype=float)
        sig = (qv < q) & (np.abs(x) >= lfc)
        ref = str(t["reference"].iloc[0]) if "reference" in t.columns and len(t) else "reference"
        cls = np.where(sig & (x > 0), f"higher in {g}", np.where(sig & (x < 0), "lower", "not significant"))
        names = [f"higher in {g}", "lower", "not significant"]
        enc, names, codes = P.cat_color(
            spec, cls, {names[0]: up, "lower": down, "not significant": neutral}, names=names
        )
        order = np.argsort(np.where(cls == "not significant", 0, 1), kind="stable")
        hover = {"title": P.hover_cat(spec, t["gene"].astype(str).to_numpy(dtype=object), "gene"),
                 "rows": [P.hover_num(spec, x, "log2 fold change", exact=True), P.hover_num(spec, t["p_value"], "p", fmt="p", exact=True),
                          P.hover_num(spec, qv, "q", fmt="p", exact=True)]}  # fmt: skip
        lay = P.points(spec, np.c_[x, y], color=enc, size={"px": 5.0}, hover=hover, when={"group": g}, order=order,
                       name="genes", hkey="volcano")  # fmt: skip
        panel["layers"].append(lay)
        counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
        P.legend_cat(spec, f"{g} vs {ref}", names, enc["colors"], counts, layer="volcano", when={"group": g})
        if label_top and sig.any():
            idx = np.flatnonzero(sig)[np.argsort(-y[sig], kind="stable")][:label_top]
            tl = P.text(spec, np.c_[x[idx], y[idx]], t["gene"].astype(str).iloc[idx].tolist(), size=10.5, weight="500",
                        anchor="start", baseline="bottom", dx=5, dy=-3, when={"group": g}, color=ch["ink"])  # fmt: skip
            tl["declutter"] = True
            panel["layers"].append(tl)
        tid = spec.table(t["gene"].astype(str).tolist())
        gene_tables[g] = tid
        cid = spec.blob(np.arange(len(t)), "u4" if len(t) > 65000 else "u2")
        hl = P.points(spec, np.c_[x, y], color={"kind": "highlight", "by": "gene", "codes": cid, "table": tid,
                      "colors": [ch["ink"]] * 8, "other": None}, size={"px": 9.0}, edge={"color": ch["bg"], "px": 1.5},
                      when={"group": g}, name="searched", label_highlight=True, hover=False)  # fmt: skip
        hl["color"].pop("other")
        panel["layers"].append(hl)
    spec.control("group", "select", label="class", options=[{"value": g, "label": g} for g in groups], value=first)
    spec.control("gene", "search", label="find", source={"by": "group", "tables": gene_tables}, placeholder="gene",
                 noun="gene", nouns="genes", multi=True, max=8, value=[], colors=[ch["ink"]] * 8)  # fmt: skip
    spec.note(f"coloured: q < {q:g} and |log2 fold change| ≥ {lfc:g}")
    del k
    return InteractiveFigure(spec, name="gex_volcano")


def _matrix_figure(
    *, title: str, call: str, style: str, rows: list[str], cols: list[str], layers: list[tuple[str, str, np.ndarray, dict]],
    hover_rows: list[dict], xlabel: str, ylabel: str, labels: dict[str, list[str]] | None = None, name: str,
    control_label: str = "value", first: str | None = None,
) -> InteractiveFigure:  # fmt: skip
    """A heatmap figure: one layer of cells per value (switcher), cell values on hover."""
    spec, panel = _plot_spec(title, call, style, x=_axis(xlabel, cats=cols, rotate=True), y=_axis(ylabel, cats=rows, flip=True),
                             aspect=1.25, zoom="none")  # fmt: skip
    panel["axes"]["square"] = True
    nr, nc = len(rows), len(cols)
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    boxes = np.c_[jj.ravel() - 0.48, ii.ravel() - 0.48, jj.ravel() + 0.48, ii.ravel() + 0.48]
    keys = []
    for key, lab, mat, opts in layers:
        vals = np.asarray(mat, dtype=float).ravel()
        fin = vals[np.isfinite(vals)]
        if opts.get("diverging"):
            lim = float(np.abs(fin).max()) if len(fin) else 1.0
            dom = (-lim, lim)
            ramp = list(C.DIVERGING["dark" if style == "dark" else "light"])
        else:
            dom = (float(fin.min()) if len(fin) else 0.0, float(fin.max()) if len(fin) else 1.0)
            if opts.get("zero"):
                dom = (0.0, dom[1])
            if dom[1] <= dom[0]:
                dom = (dom[0], dom[0] + 1.0)
            ramp = list(C.RAMPS["blue"]["dark" if style == "dark" else "light"])
        keep = np.isfinite(vals) | opts.get("keep_nan", False)
        fill = {"values": spec.blob(vals[keep], "f4"), "ramp": [P._hex(c) for c in ramp], "domain": [dom[0], dom[1]]}
        hov = {
            "rows": [{"label": r["label"], "list": list(np.asarray(r["list"], dtype=object)[keep])} for r in hover_rows]
        }
        lay = P.rects(spec, boxes[keep], fill=fill, hover=hov, when={"value": key},
                      labels=[t for t, k_ in zip(labels[key], keep, strict=True) if k_] if labels and key in labels else None)  # fmt: skip
        panel["layers"].append(lay)
        P.legend_colorbar(spec, lab, ramp, dom, fmt=opts.get("fmt"), when={"value": key}, note=opts.get("note"))
        keys.append((key, lab))
    spec.control("value", "select", label=control_label, options=[{"value": k, "label": lab} for k, lab in keys],
                 value=first or keys[0][0])  # fmt: skip
    spec.layout["aspect"] = float(np.clip((nc + 3) / (nr + 2.5), 0.8, 2.2))
    return InteractiveFigure(spec, name=name)


def csr_transitions(
    result: Any,
    *,
    value: str = "ratio",
    style: str = "dark",
    annotate: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.csr_transitions`: the upstream x downstream matrix with a value switcher.

    Hover a cell for its clones carrying both isotypes, the ratio to the null, and its p- and q-values.

    Parameters
    ----------
    result, value, style, annotate, alpha
        As in :func:`spaclone.pl.csr_transitions` (``value`` is shown first; the switcher lists the others).
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    tab = result.table if hasattr(result, "table") else result
    if value not in tab.columns:
        raise KeyError(f"csr_transitions: column {value!r} not in the result table")
    up = tab.drop_duplicates("upstream").sort_values("upstream_position", kind="stable").upstream.astype(str).tolist()
    dn = (
        tab.drop_duplicates("downstream")
        .sort_values("downstream_position", kind="stable")
        .downstream.astype(str)
        .tolist()
    )
    names = list(dict.fromkeys(up + dn))
    k = len(names)
    pos = {n: i for i, n in enumerate(names)}
    cols = [c for c in ("ratio", "statistic", "share", "n_clones_both", "z") if c in tab.columns]
    mats = {c: np.full((k, k), np.nan) for c in cols + ["p_value", "q_value"] if c in tab.columns}
    for r in tab.itertuples():
        for c in mats:
            mats[c][pos[str(r.upstream)], pos[str(r.downstream)]] = getattr(r, c)
    labels = None
    if annotate and "n_clones_both" in mats:
        nb = mats["n_clones_both"].ravel()
        qv = mats.get("q_value", np.full((k, k), np.nan)).ravel()
        txt = [
            ("" if not np.isfinite(v) else f"{int(v)}{'*' if np.isfinite(q_) and q_ <= alpha else ''}")
            for v, q_ in zip(nb, qv, strict=True)
        ]
        labels = dict.fromkeys(cols, txt)
    pair = [f"{a} → {b}" for a in names for b in names]
    hover = [{"label": "switch", "list": pair}]
    for c, lab in (
        ("n_clones_both", "clones with both"),
        ("ratio", "ratio to the null"),
        ("p_value", "p"),
        ("q_value", "q"),
    ):
        if c in mats:
            hover.append(
                {
                    "label": lab,
                    "list": [
                        ("–" if not np.isfinite(v) else (f"{v:.3g}" if c != "n_clones_both" else f"{int(v)}"))
                        for v in mats[c].ravel()
                    ],
                }
            )
    layers = [(c, c.replace("_", " "), mats[c], {"diverging": c == "z", "zero": c != "z"}) for c in cols]
    call = call_text("spaclone.pl.interactive.csr_transitions", result, value=value, style=style)
    fig = _matrix_figure(title=title or "class switches within clones", call=call, style=style, rows=names, cols=names,
                         layers=layers, hover_rows=hover, xlabel="downstream isotype (3')", ylabel="upstream isotype (5')",
                         labels=labels, name="csr_transitions", first=value)  # fmt: skip
    if annotate:
        fig._spec.note(f"cells: clones carrying both isotypes; * q ≤ {alpha:g}")
    return fig


def flux_matrix(
    result: Any,
    *,
    value: str = "net",
    order: Sequence[str] | None = None,
    mask_improbable: bool = False,
    alpha: float = 0.05,
    use: str = "q_value",
    style: str = "dark",
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.flux_matrix`: net and gross TPT flux between states, with a switcher.

    Cells are coloured by the flux in percent of its total; hover a cell for the flux and its ``use`` value.

    Parameters
    ----------
    result, value, order, mask_improbable, alpha, use, style
        As in :func:`spaclone.pl.flux_matrix`.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    mats = {v: getattr(result, v, None) for v in ("net", "gross")}
    mats = {k: m for k, m in mats.items() if m is not None}
    if value not in mats:
        raise ValueError(f"value must be one of {list(mats)}")
    m0 = mats[value]
    states = list(order) if order is not None else [str(s) for s in m0.index]
    pos = {s: i for i, s in enumerate(states)}
    tab = result.table.set_index(["from_state", "to_state"]) if hasattr(result, "table") else None
    layers = []
    hover_vals = {}
    for key, m in mats.items():
        mat = np.full((len(states), len(states)), np.nan)
        for a in states:
            for b in states:
                if a == b or (mask_improbable and pos[b] < pos[a]):
                    continue
                mat[pos[b], pos[a]] = float(m.loc[a, b])
        layers.append((key, f"{key} flux (% of total)", mat, {"zero": True, "fmt": None}))
        hover_vals[key] = mat
    pv = np.full((len(states), len(states)), np.nan)
    if tab is not None and use in tab.columns:
        for a in states:
            for b in states:
                pv[pos[b], pos[a]] = tab[use].get((a, b), np.nan)
    pair = [f"{a} → {b}" for b in states for a in states]
    hover = [{"label": "flux", "list": pair}]
    for key, mat in hover_vals.items():
        hover.append(
            {"label": f"{key} (%)", "list": [("–" if not np.isfinite(v) else f"{v:.2f}") for v in mat.ravel()]}
        )
    hover.append(
        {
            "label": "q" if use == "q_value" else "p",
            "list": [("–" if not np.isfinite(v) else f"{v:.3g}") for v in pv.ravel()],
        }
    )
    labels = {key: [("" if not np.isfinite(v) else (f"{v:.0f}%" + ("*" if np.isfinite(p_) and p_ < alpha else "")))
                    for v, p_ in zip(mat.ravel(), pv.ravel(), strict=True)] for key, mat in hover_vals.items()}  # fmt: skip
    call = call_text("spaclone.pl.interactive.flux_matrix", result, value=value, style=style)
    fig = _matrix_figure(title=title or "transition flux between states", call=call, style=style, rows=states, cols=states,
                         layers=layers, hover_rows=hover, xlabel="from", ylabel="to", labels=labels, name="flux_matrix",
                         first=value)  # fmt: skip
    fig._spec.note(f"* {'q' if use == 'q_value' else 'p'} < {alpha:g}")
    return fig


def overlap(result: Any, *, metric: str | None = None, style: str = "light", alpha: float = 0.05,
            title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.overlap`: the overlap matrix of every metric, with a switcher.

    Parameters
    ----------
    result, metric, alpha
        As in :func:`spaclone.pl.overlap`.
    style
        ``"light"`` (as the static plot) or ``"dark"``.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    t = result.table if hasattr(result, "table") else result
    metrics = [str(mm) for mm in pd.unique(t["metric"].astype(str))]
    first = metric or metrics[0]
    groups = list(dict.fromkeys([*t["group_a"].astype(str), *t["group_b"].astype(str)]))
    k = len(groups)
    pos = {g: i for i, g in enumerate(groups)}
    layers, labels = [], {}
    qtxt = {}
    for mm in metrics:
        mat = np.eye(k)
        star = np.zeros((k, k), bool)
        qm = np.full((k, k), np.nan)
        for _, r in t[t["metric"].astype(str) == mm].iterrows():
            i, j = pos[str(r["group_a"])], pos[str(r["group_b"])]
            mat[i, j] = mat[j, i] = r["value"]
            qv = r.get("q_value", np.nan)
            qm[i, j] = qm[j, i] = qv
            star[i, j] = star[j, i] = bool(pd.notna(qv) and qv < alpha)
        layers.append((mm, mm.replace("_", " "), mat, {"zero": True}))
        labels[mm] = [
            ("" if i == j else f"{mat[i, j]:.2f}{'*' if star[i, j] else ''}") for i in range(k) for j in range(k)
        ]
        qtxt[mm] = qm
    hover = [{"label": "pair", "list": [f"{a} · {b}" for a in groups for b in groups]}]
    call = call_text("spaclone.pl.interactive.overlap", result, metric=metric, style=style)
    return _matrix_figure(title=title or "repertoire overlap", call=call, style=style, rows=groups, cols=groups, layers=layers,
                          hover_rows=hover, xlabel="", ylabel="", labels=labels, name="overlap", control_label="metric",
                          first=first)  # fmt: skip


def diversity(result: Any, *, metric: str | None = None, kind: str = "group", style: str = "light",
              title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.diversity`: every metric of the table, with a switcher.

    Points are the statistic, bars the 95% of the rarefaction draws, grey boxes the placebo's 95% (tick: its mean);
    hover a group for every number.

    Parameters
    ----------
    result, metric, kind
        As in :func:`spaclone.pl.diversity`.
    style
        ``"light"`` (as the static plot) or ``"dark"``.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    mode = style
    t0 = result.table if hasattr(result, "table") else result
    t0 = t0[t0["kind"] == kind]
    metrics = [str(m) for m in pd.unique(t0["metric"].astype(str))]
    if not metrics:
        raise ValueError(f"no {kind!r} rows in the table")
    first = metric or metrics[0]
    groups = list(dict.fromkeys(t0["group"].astype(str)))
    allv = np.concatenate(
        [
            t0[c].to_numpy(dtype=float)
            for c in ("statistic", "rarefied_lo", "rarefied_hi", "null_lo", "null_hi")
            if c in t0
        ]
    )
    call = call_text("spaclone.pl.interactive.diversity", result, metric=metric, kind=kind, style=style)
    spec, panel = _plot_spec(title or "diversity", call, style, x=_axis(None, cats=groups, rotate=len(groups) > 6),
                             y=_axis(None, values=allv, zero=kind == "contrast"), aspect=1.5,
                             refs=[{"y": 0.0, "dash": [4, 3]}] if kind == "contrast" else ())  # fmt: skip
    obs = C.SLOTS[mode][0]
    ch = C.CHROME[mode]
    for mm in metrics:
        t = t0[t0["metric"].astype(str) == mm].reset_index(drop=True)
        x = np.array([groups.index(str(g)) for g in t["group"].astype(str)], dtype=float)
        w = {"metric": mm}
        if "null_lo" in t and t["null_lo"].notna().any():
            panel["layers"].append(
                P.rects(spec, np.c_[x - 0.28, t["null_lo"], x + 0.28, t["null_hi"]], fill=ch["null"], when=w)
            )
            panel["layers"].append(P.segments(spec, np.c_[x - 0.28, t["null_mean"], x + 0.28, t["null_mean"]].reshape(-1, 2, 2),
                                              color=ch["null_edge"], width=1.5, when=w))  # fmt: skip
        if "rarefied_lo" in t:
            ok = t["rarefied_lo"].notna() & t["rarefied_hi"].notna()
            if ok.any():
                xx = x[ok.to_numpy()]
                panel["layers"].append(P.segments(spec, np.c_[xx, t.loc[ok, "rarefied_lo"], xx, t.loc[ok, "rarefied_hi"]].reshape(-1, 2, 2),
                                                  color=obs, width=2.0, when=w))  # fmt: skip
        hover = {"title": P.hover_cat(spec, t["group"].astype(str).to_numpy(dtype=object), "group"),
                 "rows": [P.hover_num(spec, t[c], lab, exact=True) for c, lab in (("statistic", mm), ("rarefied_lo", "rarefaction 2.5%"),
                          ("rarefied_hi", "rarefaction 97.5%"), ("null_mean", "placebo mean"), ("p_value", "p"), ("q_value", "q")) if c in t]}  # fmt: skip
        panel["layers"].append(P.points(spec, np.c_[x, t["statistic"].to_numpy(dtype=float)], color={"kind": "const", "color": obs},
                                        size={"px": 10.0}, edge={"color": ch["bg"], "px": 1.5}, hover=hover, when=w))  # fmt: skip
    panel["title_when"] = [{"when": {"metric": mm}, "title": mm} for mm in metrics]
    spec.control(
        "metric", "select", label="metric", options=[{"value": mm, "label": mm} for mm in metrics], value=first
    )
    P.legend_lines(
        spec,
        None,
        [("observed (bar: 95% of rarefaction draws)", obs, "line"), ("placebo 95% (tick: mean)", ch["null"], "box")],
    )
    # the y range follows the metric
    ranges = {}
    for mm in metrics:
        t = t0[t0["metric"].astype(str) == mm]
        vals = np.concatenate(
            [
                t[c].to_numpy(dtype=float)
                for c in ("statistic", "rarefied_lo", "rarefied_hi", "null_lo", "null_hi")
                if c in t
            ]
        )
        ranges[mm] = _axis(None, values=vals, zero=kind == "contrast")["range"]
    panel["axes"]["y_by"] = {"metric": ranges}
    return InteractiveFigure(spec, name="diversity")


def clonal_expansion(table: pd.DataFrame, *, value: str = "share_units", style: str = "light",
                     title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.clonal_expansion`: stacked bars, units or clones (switcher).

    Parameters
    ----------
    table, value
        As in :func:`spaclone.pl.clonal_expansion`.
    style
        ``"light"`` (as the static plot) or ``"dark"``.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.repertoire import _group_cols, _labels

    style = check_style(style)
    gcols = _group_cols(table, ("expansion", "n_units", "share_units", "n_clones", "share_clones"))
    lab = _labels(table, gcols)
    t = table.assign(_g=lab.to_numpy())
    cats = (
        list(t["expansion"].cat.categories) if hasattr(t["expansion"], "cat") else list(dict.fromkeys(t["expansion"]))
    )
    cats = [str(c) for c in cats]
    groups = list(dict.fromkeys(t["_g"]))
    ramp = C.RAMPS["blue_ordinal"]["light" if style == "light" else "dark"]
    cols = (
        list(ramp[-len(cats) :])
        if len(cats) <= len(ramp)
        else [C.RAMPS["blue"][style][int(i)] for i in np.linspace(0, 6, len(cats))]
    )
    call = call_text("spaclone.pl.interactive.clonal_expansion", table, value=value, style=style)
    spec, panel = _plot_spec(title or "clonal expansion", call, style, x=_axis(None, cats=groups),
                             y=_axis("share (%)", rng=(0, 100)), aspect=1.5, zoom="none")  # fmt: skip
    for v in ("share_units", "share_clones"):
        if v not in t.columns:
            continue
        wide = t.pivot_table(index="_g", columns="expansion", values=v, aggfunc="sum", observed=False)
        wide.columns = [str(c) for c in wide.columns]
        wide = wide.reindex(index=groups, columns=cats).fillna(0.0)
        nn = t.pivot_table(
            index="_g", columns="expansion", values=v.replace("share", "n"), aggfunc="sum", observed=False
        )
        nn.columns = [str(c) for c in nn.columns]
        nn = nn.reindex(index=groups, columns=cats).fillna(0)
        boxes, codes, hov_g, hov_c, hov_v, hov_n = [], [], [], [], [], []
        for gi, g in enumerate(groups):
            bottom = 0.0
            for ci, c in enumerate(cats):
                h = 100 * float(wide.loc[g, c])
                boxes.append([gi - 0.33, bottom, gi + 0.33, bottom + h])
                codes.append(ci)
                hov_g.append(g)
                hov_c.append(c)
                hov_v.append(f"{h:.1f}%")
                hov_n.append(f"{int(nn.loc[g, c]):,}")
                bottom += h
        fill = {"codes": spec.blob(np.array(codes), "u1"), "colors": [P._hex(c) for c in cols]}
        panel["layers"].append(P.rects(spec, boxes, fill=fill, when={"value": v}, hkey="expansion",
                                       hover={"rows": [{"label": "group", "list": hov_g}, {"label": "clone size", "list": hov_c},
                                                       {"label": "share", "list": hov_v}, {"label": "units" if v == "share_units" else "clones", "list": hov_n}]}))  # fmt: skip
    P.legend_cat(spec, "clone size", cats, cols, None, layer="expansion", shape="square")
    opts = [v for v in ("share_units", "share_clones") if v in t.columns]
    spec.control(
        "value", "select", label="share of", options=[{"value": v, "label": v.split("_")[1]} for v in opts], value=value
    )
    return InteractiveFigure(spec, name="clonal_expansion")


def stationary_distribution(result: Any, *, order: Sequence[str] | None = None, style: str = "dark",
                            title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.stationary_distribution`: stationary share per state with its interval.

    Parameters
    ----------
    result, order, style
        As in :func:`spaclone.pl.stationary_distribution` (a stationary-distribution table or a transition model).
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.csr_flow import _lumped_stationary, _state_colors
    from spaclone.tl.transitions import TransitionModel

    style = check_style(style)
    if isinstance(result, TransitionModel):
        tab = _lumped_stationary(result)
    else:
        tab = getattr(result, "states", None)
        if not isinstance(tab, pd.DataFrame):
            raise TypeError(
                "stationary_distribution expects a FluxResult (spaclone.tl.tpt_flux) or a TransitionModel with states"
            )
    every = [str(s) for s in tab.index]
    tab = tab.set_axis(every)
    st = [str(s) for s in order] if order is not None else every
    t = tab.loc[st].reset_index(drop=True)
    cols = _state_colors(every, style, None)
    val = "stationary"
    lo = "stationary_lo" if "stationary_lo" in t.columns and t["stationary_lo"].notna().any() else None
    hi = "stationary_hi" if lo else None
    v = t[val].to_numpy(dtype=float)
    if "unit_share" in t.columns:
        t["share"] = t["unit_share"]
    call = call_text("spaclone.pl.interactive.stationary_distribution", result, style=style)
    spec, panel = _plot_spec(title or "stationary distribution", call, style, x=_axis(None, cats=st),
                             y=_axis("share", values=np.r_[v, 0, t[hi].to_numpy(dtype=float) if hi else v], fmt="pct", zero=True),
                             aspect=1.5, zoom="none")  # fmt: skip
    x = np.arange(len(st), dtype=float)
    fill = {"codes": spec.blob(np.arange(len(st)), "u1"), "colors": [P._hex(cols[s]) for s in st]}
    hover = {
        "rows": [{"label": "state", "list": st}, {"label": "stationary share", "list": [f"{100 * a:.1f}%" for a in v]}]
    }
    if lo and hi:
        hover["rows"].append(
            {
                "label": "95% interval",
                "list": [f"{100 * a:.1f}–{100 * b:.1f}%" for a, b in zip(t[lo], t[hi], strict=True)],
            }
        )
    if "share" in t.columns:
        hover["rows"].append(
            {"label": "share of units", "list": [f"{100 * a:.1f}%" for a in t["share"].to_numpy(dtype=float)]}
        )
    panel["layers"].append(P.rects(spec, np.c_[x - 0.32, np.zeros(len(st)), x + 0.32, v], fill=fill, hover=hover))
    if lo and hi:
        panel["layers"].append(
            P.segments(spec, np.c_[x, t[lo], x, t[hi]].reshape(-1, 2, 2), color=C.CHROME[style]["ink"], width=1.5)
        )
    if "share" in t.columns:
        panel["layers"].append(P.points(spec, np.c_[x, t["share"].to_numpy(dtype=float)], color={"kind": "const", "color": C.CHROME[style]["ink"]},
                                        size={"px": 7.0}, hover=False, shape="square"))  # fmt: skip
        P.legend_lines(spec, None, [("stationary share (bar: 95% interval)", C.CHROME[style]["ink2"], "line"),
                                    ("observed share of units", C.CHROME[style]["ink"], "line")])  # fmt: skip
    return InteractiveFigure(spec, name="stationary_distribution")


def placebo(result: Any, *, component: str | int | None = None, bins: int = 40, style: str = "dark",
            title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.placebo`: the null distribution and the observed value, per component.

    Parameters
    ----------
    result, component, style
        As in :func:`spaclone.pl.placebo` (a vector statistic gets a component switcher).
    bins
        Histogram bins.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    null = getattr(result, "null", None)
    table = getattr(result, "table", None)
    if null is None or table is None:
        raise ValueError("the result keeps no placebo draws: run nulls.placebo(..., keep_null=True)")
    null = np.asarray(null, dtype=float)
    null = null[:, None] if null.ndim == 1 else null
    obs = getattr(result, "observed", None)  # as the static plot: the observed statistic, else the table's column
    if (obs is None or not np.size(obs)) and "observed" not in table.columns:
        raise ValueError("pl.interactive.placebo draws nulls.placebo results (an 'observed' statistic and its draws)")
    obs = np.asarray(table["observed"] if obs is None or not np.size(obs) else obs, dtype=float).ravel()
    names = [str(c) for c in table["component"]] if "component" in table.columns else [str(i) for i in range(len(obs))]
    if null.shape[1] != len(obs):
        raise ValueError(f"null draws have {null.shape[1]} components, the observed statistic {len(obs)}")
    pvals = table["p_value"].to_numpy(dtype=float) if "p_value" in table.columns else np.full(len(obs), np.nan)
    if component is None:
        first = 0
    elif isinstance(component, str):
        if component not in names:
            raise ValueError(f"no component {component!r}; the result has {names}")
        first = names.index(component)
    else:
        first = int(component)
    ch = C.CHROME[style]

    def _range(k: int) -> list[float]:  # the component's own values: components may have different scales
        vals = np.r_[null[:, k], obs[k]]
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return [0.0, 1.0]
        lo, hi = float(vals.min()), float(vals.max())
        pad = (hi - lo) * 0.06 if hi > lo else max(abs(lo) * 0.05, 1.0)  # all draws equal: a window around them
        return [lo - pad, hi + pad]

    xr = {str(k): _range(k) for k in range(len(names))}
    call = call_text("spaclone.pl.interactive.placebo", result, component=component, style=style)
    spec, panel = _plot_spec(title or "observed vs placebo", call, style, x=_axis("statistic", rng=xr[str(first)]),
                             y=_axis("placebo draws", rng=(0, 1)), aspect=1.6)  # fmt: skip
    yr = {}
    for k in range(len(names)):
        v = null[:, k]
        v = v[np.isfinite(v)]
        both = np.r_[v, obs[k]] if np.isfinite(obs[k]) else v
        if len(both) and both.min() == both.max():  # every draw the same: one bar, a visible width
            half = (xr[str(k)][1] - xr[str(k)][0]) / (2 * bins)
            edges = np.array([both[0] - half, both[0] + half])
        else:
            edges = np.histogram_bin_edges(both, bins=bins)
        cnt, _ = np.histogram(v, edges)
        w = {"component": str(k)}
        panel["layers"].append(P.rects(spec, np.c_[edges[:-1], np.zeros(len(cnt)), edges[1:], cnt], fill=ch["null"], when=w,
                                       hover={"rows": [{"label": "bin", "list": [f"{a:.3g} to {b:.3g}" for a, b in zip(edges[:-1], edges[1:], strict=True)]},
                                                       {"label": "draws", "list": [str(int(c)) for c in cnt]}]}))  # fmt: skip
        yr[str(k)] = [0.0, float(cnt.max()) * 1.12 if len(cnt) else 1.0]
        if np.isfinite(obs[k]):
            top = yr[str(k)][1]
            panel["layers"].append(
                P.segments(spec, [[[obs[k], 0], [obs[k], top]]], color=C.SLOTS[style][1], width=2.2, when=w)
            )
            ptxt = f"p = {pvals[k]:.3g}" if k < len(pvals) and np.isfinite(pvals[k]) else ""
            right = obs[k] > (xr[str(k)][0] + xr[str(k)][1]) / 2  # the label on the side with room
            panel["layers"].append(P.text(spec, [(obs[k], top * 0.97)], [f"observed {obs[k]:.4g} {ptxt}".strip()], size=11,
                                          anchor="end" if right else "start", baseline="top", dx=-6 if right else 6,
                                          when=w, color=C.SLOTS[style][1]))  # fmt: skip
    panel["axes"]["y_by"] = {"component": yr}
    panel["axes"]["x_by"] = {"component": xr}
    if len(names) > 1:
        spec.control("component", "select", label="component", options=[{"value": str(k), "label": n} for k, n in enumerate(names)],
                     value=str(first))  # fmt: skip
    else:
        spec.control("component", "select", label="", options=[{"value": "0", "label": names[0]}], value="0")
    P.legend_lines(spec, None, [("placebo draws", ch["null"], "box"), ("observed", C.SLOTS[style][1], "line")])
    return InteractiveFigure(spec, name="placebo")


def clonal_scan(result: Any, *, kind: str = "size", style: str = "dark", label_top: int = 5,
                title: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.clonal_scan`: −log10 p against clone size, every clone on hover.

    Parameters
    ----------
    result, kind, style, label_top
        As in :func:`spaclone.pl.clonal_scan` (``kind="size"``; the Q-Q view stays static).
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    t = result.table
    if "testable" in t.columns:
        t = t[t["testable"]]
    t = t.reset_index(drop=True)
    if not len(t):
        raise ValueError("clonal_scan: no clone was tested")
    alpha = float(result.params.get("alpha", 0.05))
    lp = -np.log10(t["p_value"].to_numpy(dtype=float))
    sig = t["significant"].to_numpy(dtype=bool)
    blocks = "n_blocks" in t.columns and bool(np.isfinite(t["n_blocks"].to_numpy(dtype=float)).all())
    n = t["n_blocks" if blocks else "n_units"].to_numpy(dtype=float)
    thr = t.loc[sig, "p_value"].max() if sig.any() else alpha / len(t)
    st = {"accent": C.SLOTS[style][0], "accent2": C.SLOTS[style][1], "muted": C.NEUTRAL[style]}
    call = call_text("spaclone.pl.interactive.clonal_scan", result, style=style)
    spec, panel = _plot_spec(title or "clonal scan", call, style,
                             x=_axis("cell-scale blocks of the clone" if blocks else "units of the clone", scale="log", values=n),
                             y=_axis("−log10 p", values=np.r_[lp, -np.log10(thr), 0], zero=True),
                             refs=[{"y": float(-np.log10(thr)), "dash": [4, 3], "color": st["accent"]}])  # fmt: skip
    cls = np.where(sig, f"q ≤ {alpha:g}", "not called")
    enc, names, codes = P.cat_color(spec, cls, {f"q ≤ {alpha:g}": st["accent2"], "not called": st["muted"]})
    hover = {"title": P.hover_cat(spec, t["clone_id"].astype(str).to_numpy(dtype=object), "clone"),
             "rows": [P.hover_num(spec, n, "blocks" if blocks else "units", exact=True), P.hover_num(spec, t["p_value"], "p", fmt="p", exact=True)]
             + ([P.hover_num(spec, t["q_value"], "q", fmt="p", exact=True)] if "q_value" in t else [])}  # fmt: skip
    panel["layers"].append(P.points(spec, np.c_[np.log10(n), lp], color=enc, size={"px": 6.0}, hover=hover,
                                    order=np.argsort(sig, kind="stable"), hkey="scan"))  # fmt: skip
    if label_top:
        top = np.argsort(t["p_value"].to_numpy(dtype=float))[:label_top]
        tl = P.text(spec, np.c_[np.log10(n[top]), lp[top]], [str(c)[-24:] for c in t["clone_id"].iloc[top]], size=10,
                    weight="500", anchor="start", baseline="bottom", dx=4, dy=-3)  # fmt: skip
        tl["declutter"] = True
        panel["layers"].append(tl)
    counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
    P.legend_cat(spec, None, names, enc["colors"], counts, layer="scan")
    P.legend_lines(spec, None, [("BH threshold", st["accent"], "dash")])
    return InteractiveFigure(spec, name="clonal_scan")


def read_support(data: Any, *, by: str | Sequence[str] | None = "locus", share: bool = True, style: str = "light",
                 title: str | None = None, reads_key: str = "consensus_count") -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.read_support`: records kept against the read floor, per group.

    Parameters
    ----------
    data, by, share, reads_key
        As in :func:`spaclone.pl.read_support`.
    style
        ``"light"`` (as the static plot) or ``"dark"``.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.qc import _group_colors
    from spaclone.pp.metrics import ReadSupport, read_support_curve

    style = check_style(style)
    if isinstance(data, ReadSupport):
        curve, kn = data
    else:
        b = [] if by is None else ([by] if isinstance(by, str) else list(by))
        b = [c for c in b if c in data.columns]
        curve, kn = read_support_curve(data, by=b, reads_key=reads_key)
    gcols = [c for c in curve.columns if c not in ("min_reads", "records", "share_records", "spots", "share_spots")]
    ycol = "share_records" if share else "records"
    groups = list(curve.groupby(gcols, sort=True)) if gcols else [((), curve)]
    keys = [k if isinstance(k, tuple) else (k,) for k, _ in groups]
    colors = _group_colors([k[-1] if k else "" for k in keys])
    call = call_text("spaclone.pl.interactive.read_support", data, by=by, share=share, style=style)
    spec, panel = _plot_spec(title or "read support", call, style,
                             x=_axis("read floor (reads per record)", scale="log", values=curve["min_reads"]),
                             y=_axis("share of records kept" if share else "records kept", scale="lin" if share else "log",
                                     values=curve[ycol], fmt="pct" if share else None, zero=share))  # fmt: skip
    items = []
    for key, (_, g) in zip(keys, groups, strict=True):
        label = " / ".join(str(k) for k in key) if key else "all"
        col = colors[str(key[-1]) if key else ""]
        xs = g["min_reads"].to_numpy(dtype=float)
        ys = g[ycol].to_numpy(dtype=float)
        sx = np.repeat(xs, 2)[1:]
        sy = np.repeat(ys, 2)[:-1]
        step = np.c_[np.log10(sx), sy if share else np.log10(np.maximum(sy, 1e-12))]
        panel["layers"].append(P.paths(spec, [step], stroke=col, width=1.8, name=label))
        hv = {"title": label, "rows": [P.hover_num(spec, xs, "read floor", exact=True),
                                       P.hover_num(spec, ys, ycol.replace("_", " "), fmt="pct" if share else None, exact=True)]}  # fmt: skip
        panel["layers"].append(P.points(spec, np.c_[np.log10(xs), ys if share else np.log10(np.maximum(ys, 1e-12))],
                                        color={"kind": "const", "color": col}, size={"px": 3.0}, hover=hv))  # fmt: skip
        items.append((label, col, "line"))
    P.legend_lines(spec, None, items)
    if len(kn):
        spec.note("knee and inflection read floors: " + "; ".join(
            f"{' / '.join(str(r[c]) for c in gcols) or 'all'}: knee {r['knee_reads']:g}" for _, r in kn.iterrows()
            if np.isfinite(r.get("knee_reads", np.nan))))  # fmt: skip
    return InteractiveFigure(spec, name="read_support")


def trust_report(report: Any, *, style: str = "dark", title: str | None = None) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.trust_report`: observed ÷ chance of every test row, with every number on hover.

    Parameters
    ----------
    report, style
        As in :func:`spaclone.pl.trust_report`.
    title
        Figure title (default: the verdict).

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.report import _LABELS, _SHOWN

    style = check_style(style)
    th = C.VERDICT_COLORS[style]
    t = report.tests()
    t = t[t.test.isin(_LABELS) & t.verdict.isin(_SHOWN)].reset_index(drop=True)
    if not len(t):
        raise ValueError("trust_report: no test row to draw")
    rows = [
        f"{_LABELS[r.test]} · {r.locus}" if "locus" in t.columns and pd.notna(r.locus) else _LABELS[r.test]
        for r in t.itertuples()
    ]
    ratio = t["ratio"].to_numpy(dtype=float)
    fin = ratio[np.isfinite(ratio) & (ratio > 0)]
    lo = min(0.5, float(fin.min()) * 0.8) if len(fin) else 0.5
    hi = max(2.0, float(fin.max()) * 1.25) if len(fin) else 2.0
    call = call_text("spaclone.pl.interactive.trust_report", report, style=style)
    spec, panel = _plot_spec(title or "location trust", call, style,
                             x=_axis("observed ÷ chance", scale="log", rng=(lo, hi)), y=_axis(None, cats=rows, flip=True),
                             refs=[{"x": 1.0, "color": C.CHROME[style]["ink2"]}], aspect=1.6, zoom="x")  # fmt: skip
    verdicts = [v for v in _SHOWN if (t.verdict == v).any()]
    enc, names, codes = P.cat_color(spec, t["verdict"].astype(str), {v: th[v] for v in verdicts}, names=verdicts)
    shown = {"info": "VJ chain (info)"}  # the static report's wording of an information-only row
    ok = np.isfinite(ratio) & (ratio > 0)
    idx = np.flatnonzero(ok)
    hv_rows = [P.hover_cat(spec, t.loc[idx, "verdict"].astype(str).map(lambda v: shown.get(v, v)).to_numpy(dtype=object),
                           "verdict")]  # fmt: skip
    for c, lab in (
        ("n", "n"),
        ("observed", "observed"),
        ("chance", "chance"),
        ("ratio", "observed ÷ chance"),
        ("p_value", "p"),
        ("q_value", "q"),
    ):
        if c in t.columns:
            hv_rows.append(
                P.hover_num(spec, t.loc[idx, c], lab, exact=True, fmt="p" if c in ("p_value", "q_value") else None)
            )
    if "note" in t.columns:
        hv_rows.append(P.hover_cat(spec, t.loc[idx, "note"].astype("string").to_numpy(dtype=object), "note"))
    cid = spec.blob(codes[idx], "u1")
    panel["layers"].append(P.points(spec, np.c_[np.log10(ratio[idx]), idx.astype(float)], color={**enc, "codes": cid},
                                    size={"px": 11.0}, edge={"color": C.CHROME[style]["bg"], "px": 1.2},
                                    hover={"title": P.hover_cat(spec, [rows[i] for i in idx], "test"), "rows": hv_rows},
                                    hkey="verdict"))  # fmt: skip
    counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
    P.legend_cat(spec, "verdict", [shown.get(n, n) for n in names], enc["colors"], counts, layer="verdict")
    panel["margin"] = {"l": 280}
    spec.subtitle = str(report.verdict).split(". ")[0].rstrip(".") + "."
    spec.layout["aspect"] = float(np.clip(1.9 * 10 / max(len(rows), 4), 0.9, 2.4))
    return InteractiveFigure(spec, name="trust_report")


def niche_profile(result: Any, *, style: str = "dark", receptor: str = "TCR", ratio: bool = True, min_baseline: int = 0,
                  title: str | None = None, xlabel: str | None = None) -> InteractiveFigure:  # fmt: skip
    """Interactive form of :func:`spaclone.pl.niche_profile`: the profile, its chance band and global envelope.

    Parameters
    ----------
    result, style, receptor, ratio, min_baseline, xlabel
        As in :func:`spaclone.pl.niche_profile`.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    tab = result.table
    t = tab[tab.baseline_units >= min_baseline].reset_index(drop=True)
    numeric = len(t) > 0 and bool(np.isfinite(t.mid_um.to_numpy(dtype=float)).all())
    x = t.mid_um.to_numpy(dtype=float) if numeric else np.arange(len(t), dtype=float)
    cols = (
        ("ratio", "ratio_lo", "ratio_hi", "ratio_env_lo", "ratio_env_hi")
        if ratio
        else ("statistic", "null_lo", "null_hi", "env_lo", "env_hi")
    )
    y, lo, hi, elo, ehi = (
        np.where(np.isfinite(t[c].to_numpy(dtype=float)), t[c].to_numpy(dtype=float), np.nan) for c in cols
    )
    col = C.RECEPTOR_COLORS[style].get(receptor.upper(), C.RECEPTOR_COLORS[style]["TCR"])
    allv = np.r_[y, lo, hi, elo, ehi, 1.0 if ratio else 0.0]
    call = call_text("spaclone.pl.interactive.niche_profile", result, receptor=receptor, style=style)
    spec, panel = _plot_spec(title or "niche profile", call, style,
                             x=_axis(xlabel or ("signed distance to the border (µm)" if numeric else None), values=x) if numeric
                             else _axis(None, cats=t.label.astype(str).tolist()),
                             y=_axis("ratio to baseline" if ratio else "share of the group", values=allv),
                             refs=([{"y": 1.0, "dash": [2, 3]}] if ratio else []) + ([{"x": 0.0}] if numeric else []))  # fmt: skip
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ring = np.r_[np.c_[x[ok], lo[ok]], np.c_[x[ok], hi[ok]][::-1]]
        panel["layers"].append(P.paths(spec, [ring], closed=True, fill=col, fill_alpha=0.16, name="chance band"))
    for e in (elo, ehi):
        okk = np.isfinite(e)
        if okk.any():
            panel["layers"].append(
                P.paths(spec, [np.c_[x[okk], e[okk]]], stroke=col, width=1.0, dash=[4, 3], alpha=0.85)
            )
    oky = np.isfinite(y)
    if oky.sum() > 1 and numeric:
        panel["layers"].append(P.paths(spec, [np.c_[x[oky], y[oky]]], stroke=col, width=2.0))
    hover = {"title": P.hover_cat(spec, t.label.astype(str).to_numpy(dtype=object), "band") if "label" in t else None,
             "rows": [P.hover_num(spec, y, "ratio" if ratio else "share", exact=True), P.hover_num(spec, lo, "chance 2.5%", exact=True),
                      P.hover_num(spec, hi, "chance 97.5%", exact=True), P.hover_num(spec, t.baseline_units, "baseline units", exact=True)]}  # fmt: skip
    if hover["title"] is None:
        hover.pop("title")
    panel["layers"].append(
        P.points(spec, np.c_[x, y], color={"kind": "const", "color": col}, size={"px": 7.0}, hover=hover)
    )
    P.legend_lines(
        spec, None, [("observed", col, "line"), ("chance (pointwise)", col, "box"), ("global envelope", col, "dash")]
    )
    return InteractiveFigure(spec, name="niche_profile")
