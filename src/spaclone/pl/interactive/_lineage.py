"""Interactive forms of the lineage trees next to the tissue: hover a tip in the tree to find it on the section."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl import _colors as C
from spaclone.pl.interactive import _geo as G
from spaclone.pl.interactive import _spec as P
from spaclone.pl.interactive._figure import InteractiveFigure
from spaclone.pl.interactive._util import call_text, check_style

__all__ = ["lineage_tree", "tree_tissue"]


def _tree_axes(xmax: float, n_rows: int) -> dict[str, Any]:
    return {"x": {"label": "mutations from the germline", "range": [-0.04 * xmax - 0.8, xmax * 1.06 + 0.5], "grid": True},
            "y": {"range": [-1.0, n_rows], "flip": True, "hide": True}, "zoom": "xy"}  # fmt: skip


def _raster_layers(spec: P.Spec, panel: dict[str, Any], maps: Mapping[str, Any], m: np.ndarray, mode: str) -> list:
    """Tissue fill, switched-share image and outlines of a lineage workflow's maps; returns the legend items."""
    from spaclone.pl.atlas import _continuous_rgba, _fill_from, _mask_fill, _mask_lines

    th = C.LINEAGE_STYLES[mode]
    items = []
    tissue = maps.get("tissue")
    clip = None
    if tissue is not None:
        rings = [
            G.apply(m, r)
            for r in G.rings_from_paths(_mask_fill(np.asarray(tissue.values, dtype=bool), tissue.grid, 0.8))
        ]
        if rings:
            tl = P.paths(spec, rings, closed=True, fill=th["tissue"], name="tissue")
            tl["hover"] = False
            panel["layers"].append(tl)
            clip = tl["id"]
    share, elig = maps.get("share"), maps.get("share_eligible")
    if share is not None:
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        vals = np.asarray(share.values, dtype=float)
        e = (np.asarray(elig.values, dtype=bool) if elig is not None else np.ones(vals.shape, bool)) & np.isfinite(vals)
        if tissue is not None and tissue.grid == share.grid:
            e &= np.asarray(tissue.values, dtype=bool)
        fin = vals[e]
        lo, hi = np.quantile(fin, [0.02, 0.995]) if len(fin) else (0.0, 1.0)
        hi = float(max(hi, lo + 1e-9))
        filled, support = _fill_from(vals, e)
        tis = np.asarray(tissue.values, dtype=bool) if (tissue is not None and tissue.grid == share.grid) else e
        rgba = _continuous_rgba(
            filled,
            e,
            support,
            tis,
            LinearSegmentedColormap.from_list("spc", th["ramp"]),
            Normalize(float(lo), hi),
            th["tissue"],
        )
        panel["layers"].append(P.image(spec, G.grid_rows(rgba), G.grid_image(share.grid, m), values=G.grid_rows(np.where(e, vals, np.nan)),
                                       value_label="switched share (IgG+IgA)", value_fmt="pct", clip=clip, name="switched share"))  # fmt: skip
        P.legend_colorbar(spec, "switched share (IgG+IgA)", th["ramp"], (float(lo), hi), fmt="pct0")
    b_rich, t_rich, hot = maps.get("b_rich"), maps.get("t_rich"), maps.get("hotspots")
    for raster, col, lw, name, smooth in ((b_rich, th["fg"], 0.8, "B-rich area", 1.5), (t_rich, th["t_edge"], 1.0, "T zone", 1.5),
                                          (hot, th["hot_edge"], 1.6, "CSR hotspot", 0.7)):  # fmt: skip
        if raster is None:
            continue
        mask = np.asarray(raster.values, dtype=float) > 0
        if name == "B-rich area" and t_rich is not None and t_rich.grid == raster.grid:
            mask = mask | (np.asarray(t_rich.values, dtype=float) > 0)
        lines = _mask_lines(mask, raster.grid, smooth)
        if not lines:
            continue
        pl_ = P.paths(spec, [G.apply(m, v) for v in lines], stroke=col, width=lw, name=name, okey=f"out:{name}")
        pl_["hover"] = False
        panel["layers"].append(pl_)
        items.append((name, col, "line", f"out:{name}"))
    return items


def lineage_tree(
    result: Any,
    tree_id: str | None = None,
    *,
    grouping: str = "exact",
    style: str = "dark",
    background: str = "auto",
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.lineage_tree`: every drawn tree of a lineage workflow, next to the tissue.

    A switcher lists the trees the workflow drew (exact clones and related groups); hover a tip in the tree to ring its
    positions on the section, or a tip on the section to ring it in the tree; tips are coloured by isotype, with
    their location class, mutations and CDR3 on hover.

    Parameters
    ----------
    result
        A :class:`spaclone.bcr.LineageWorkflowResult`.
    tree_id
        Tree shown first (default: the first drawn tree of ``grouping``).
    grouping
        Grouping of ``tree_id`` (``"exact"`` or ``"related"``).
    style
        ``"dark"`` or ``"light"``.
    background
        ``"auto"`` / ``"share"`` (the switched-share map when the result has one) or ``"tissue"``.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.bcr.lineage_workflow import _grouping
    from spaclone.pl.lineage_gallery import _ISO, _Section, _split_clone, _Tree

    style = check_style(style)
    mode = style
    th = C.LINEAGE_STYLES[mode]
    loc_col = C.LOCATION_COLORS[mode]
    entries = []
    for g in ("exact", "related"):
        if g not in getattr(result, "trees", {}):
            continue
        for r in result.drawn(g).itertuples():
            entries.append((g, str(r.tree_id), int(r.rank), int(r.n_tips)))
    if not entries:
        raise ValueError("lineage_tree: the result has no drawn tree")
    g0 = _grouping(grouping)
    first = next((f"{g}:{t}" for g, t, *_ in entries if g == g0 and (tree_id is None or t == str(tree_id))), None)
    if first is None:
        raise KeyError(f"lineage_tree: no drawn {g0} tree {tree_id!r}")
    maps = dict(result.maps or {})
    if background == "tissue":
        maps["share"] = None
    rec_xy = result.records[[S.X, S.Y]].to_numpy(dtype=float) if len(result.records) else np.zeros((0, 2))
    sec = _Section(maps, rec_xy if len(rec_xy) else np.zeros((3, 2)))
    m = np.eye(3)
    m[:2, :2] = sec.R
    m[:2, 2] = -sec.R @ sec.c0
    call = call_text("spaclone.pl.interactive.lineage_tree", result, tree_id=tree_id, grouping=grouping, style=style)
    spec = P.Spec(style=style, title=title or "lineage trees on the tissue", call=call)
    tree_panel = spec.panel("tree", "plot", area=[0, 0, 1, 1], title=None)
    map_panel = spec.panel("map", "map", area=[1, 0, 1, 1], invert_y=True, home=[float(v) for v in sec.box], pad=0.01)
    items = _raster_layers(spec, map_panel, maps, m, mode)
    options = []
    xmax_all, rows_all = {}, {}
    iso_names = ["IgM", "IgD", "IgG", "IgA", "unknown"]
    for g, tid, rank, n_tips in entries:
        key = f"{g}:{tid}"
        tips = result.tree_tips(tid, grouping=g)
        if not len(tips):
            continue
        tr = _Tree(result.trees[g][tid], tips)
        lv = tr.leaves
        xy = G.apply(m, lv[[S.X, S.Y]].to_numpy(dtype=float))
        order = tr.order(xy[:, 0])
        row, ys = tr.layout(order)
        segs = np.asarray(tr.segments(row, ys), dtype=float).reshape(-1, 2, 2)
        w = {"tree": key}
        tree_panel["layers"].append(P.segments(spec, segs, color=th["tree"], width=1.1, when=w))
        n = len(lv)
        guide = np.c_[lv["div"].to_numpy(dtype=float), row, np.full(n, max(tr.div.values()) * 1.04 + 0.5), row].reshape(
            -1, 2, 2
        )
        tree_panel["layers"].append(P.segments(spec, guide, color=th["guide"], width=0.5, dash=[1, 2], when=w))
        tree_panel["layers"].append(P.points(spec, [(0.0, ys[tr.germ])], color={"kind": "const", "color": th["germ"]},
                                             size={"px": 9.0}, shape="square", when=w, hover={"title": "germline"}))  # fmt: skip
        iso = lv["isotype"].astype(str).where(lv["isotype"].astype(str).isin(iso_names), "unknown")
        enc, names, codes = P.cat_color(spec, iso.to_numpy(dtype=object), _ISO[mode], names=iso_names)
        hover = {"title": P.hover_cat(spec, lv["tip_id"].astype(str).to_numpy(dtype=object), "tip"),
                 "rows": [P.hover_cat(spec, iso.to_numpy(dtype=object), "isotype"),
                          P.hover_cat(spec, lv["location"].astype(str).to_numpy(dtype=object), "location"),
                          P.hover_num(spec, lv["mutations"], "V mutations", exact=True),
                          P.hover_cat(spec, lv[S.JUNCTION_AA].astype(str).to_numpy(dtype=object), "CDR3") if S.JUNCTION_AA in lv else None,
                          P.hover_num(spec, lv["n_records"], "records", exact=True) if "n_records" in lv else None]}  # fmt: skip
        hover["rows"] = [r for r in hover["rows"] if r is not None]
        s_pt = float(np.clip(1700.0 / n, 8, 48))
        dpx = float(np.sqrt(s_pt) * 1.25)
        tree_panel["layers"].append(P.points(spec, np.c_[lv["div"].to_numpy(dtype=float), row], color=enc, size={"px": dpx},
                                             hover=hover, when=w, key=np.arange(n), key_group="tips", hkey="tree_iso"))  # fmt: skip
        mut = np.argsort(lv["mutations"].to_numpy(), kind="stable")
        map_panel["layers"].append(P.points(spec, xy, color=enc, size={"px": dpx}, edge={"color": th["bg"], "px": 0.6},
                                            hover=hover, when=w, key=np.arange(n), key_group="tips", order=mut, hkey="tree_iso"))  # fmt: skip
        counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
        P.legend_cat(spec, "tip isotype", names, enc["colors"], counts, layer="tree_iso", when=w)
        loc = lv["location"].astype(str)
        locs = [k_ for k_ in dict.fromkeys(loc) if k_]
        P.legend_cat(spec, "tip location", locs, [loc_col.get(k_, loc_col["unknown"]) for k_ in locs],
                     [int((loc == k_).sum()) for k_ in locs], shape="square", when=w)  # fmt: skip
        v, j, cdr3 = _split_clone(tid)
        lab = f"{g} {rank} · {v} {j} {cdr3}".strip() + f" ({n_tips} tips)"
        options.append({"value": key, "label": lab})
        xmax_all[key] = float(max(max(tr.div.values()), 1.0))
        rows_all[key] = n
        n_loci = lv["locus_id"].nunique() if "locus_id" in lv else n
        spec.note(f"{n} tips · {n_loci} loci · V mutations per tip: up to {int(lv['mutations'].max())} "
                  f"(median {lv['mutations'].median():g})", when=w)  # fmt: skip
    tree_panel["axes"] = _tree_axes(xmax_all[first], rows_all[first])
    tree_panel["axes"]["x_by"] = {"tree": {k: [-0.04 * x - 0.8, x * 1.06 + 0.5] for k, x in xmax_all.items()}}
    tree_panel["axes"]["y_by"] = {"tree": {k: [-1.0, float(r)] for k, r in rows_all.items()}}
    tree_panel["margin"] = {"l": 12, "t": 12}
    if items:
        P.legend_lines(spec, "map", [(a, b, c) for a, b, c, _ in items], layers=[d for *_, d in items])
    spec.control("tree", "select", label="tree", options=options, value=first)
    box = sec.box
    asp = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    spec.layout = {
        "cols": 2,
        "rows": 1,
        "colw": [1.0, 1.7],
        "aspect": float(np.clip(asp * 0.9 + 0.6, 1.4, 3.6)),
        "height_hint": 620,
    }
    return InteractiveFigure(spec, name="lineage_tree")


def tree_tissue(
    tree: Any,
    tips: pd.DataFrame,
    *,
    color_key: str = "isotype",
    label_key: str = "sequence_id",
    x_key: str = S.X,
    y_key: str = S.Y,
    background: pd.DataFrame | None = None,
    style: str = "dark",
    palette: Mapping[str, str] | None = None,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.tree_tissue`: a lineage tree and the tips' tissue positions, linked.

    Hover a tip in the tree to ring its positions on the section, and the other way round.

    Parameters
    ----------
    tree, tips, color_key, label_key, x_key, y_key, background, style, palette, rotate, crop, invert_y
        As in :func:`spaclone.pl.tree_tissue`.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.bcr import _colors_for, _tree_layout

    style = check_style(style)
    mode = style
    for c in (label_key, x_key, y_key):
        if c not in tips.columns:
            raise KeyError(f"tree_tissue: column {c!r} not in the tip table")
    x, y, order = _tree_layout(tree)
    call = call_text("spaclone.pl.interactive.tree_tissue", tree, tips, color_key=color_key, style=style)
    spec = P.Spec(style=style, title=title or f"clone {tree.clone_id}", call=call)
    n_rows = len(order)
    xmax = float(np.nanmax(x)) if len(x) else 1.0
    tp = spec.panel(
        "tree", "plot", area=[0, 0, 1, 1], axes=_tree_axes(max(xmax, 1.0), n_rows), margin={"l": 12, "t": 12}
    )
    segs = []
    for v, p in enumerate(tree.parent):
        if p < 0 or not np.isfinite(y[v]):
            continue
        segs.append([[x[p], y[v]], [x[v], y[v]]])
        segs.append([[x[p], y[p]], [x[p], y[v]]])
    pal = {"line": C.CHROME[mode]["ink2"], "bg": C.CHROME[mode]["bg"], "tissue": C.CHROME[mode]["tissue_strong"]}
    if segs:
        tp["layers"].append(P.segments(spec, np.asarray(segs, dtype=float), color=pal["line"], width=1.0))
    vals = (
        tips[color_key].astype("string").fillna("unknown").astype(str)
        if color_key in tips
        else pd.Series("tip", index=tips.index)
    )
    keys = list(pd.unique(vals))
    cols = _colors_for(keys, palette, style, C.vocabulary(vals))
    row_of = {tree.label[v]: k for k, v in enumerate(order)}
    first = tips.drop_duplicates(label_key).set_index(label_key)
    tlabels = [tree.label[v] for v in order]
    tvals = [str(first.loc[lab, color_key]) if (color_key in tips and lab in first.index) else "tip" for lab in tlabels]
    enc_t, names, codes_t = P.cat_color(spec, tvals, cols, names=keys)
    tp["layers"].append(P.points(spec, np.c_[x[order], y[order]], color=enc_t, size={"px": 8.0}, edge={"color": pal["bg"], "px": 0.6},
                                 hover={"title": P.hover_cat(spec, tlabels, "tip"), "rows": [P.hover_cat(spec, tvals, color_key)]},
                                 key=np.arange(len(order)), key_group="tips", hkey="tips"))  # fmt: skip
    txy = tips[[x_key, y_key]].to_numpy(dtype=float)
    bxy = background[[x_key, y_key]].to_numpy(dtype=float) if background is not None else None
    fr = G.frame_for(bxy if bxy is not None and len(bxy) else txy, rotate, crop)
    mm = G.affine(fr)
    mp = spec.panel("map", "map", area=[1, 0, 1, 1], invert_y=bool(invert_y))
    if bxy is not None and len(bxy):
        mp["layers"].append(P.points(spec, G.apply(mm, bxy), color={"kind": "const", "color": pal["tissue"]}, size={"px": 2.0},
                                     hover=False, name="tissue"))  # fmt: skip
    keyv = np.array([row_of.get(lab, -1) for lab in tips[label_key]], dtype=np.int64)
    ok = keyv >= 0
    enc_m, _, codes_m = P.cat_color(spec, vals.to_numpy(dtype=object)[ok], cols, names=keys)
    mp["layers"].append(P.points(spec, G.apply(mm, txy[ok]), color=enc_m, size={"px": 8.0}, edge={"color": pal["bg"], "px": 0.6},
                                 hover={"title": P.hover_cat(spec, tips[label_key].astype(str).to_numpy(dtype=object)[ok], "tip"),
                                        "rows": [P.hover_cat(spec, vals.to_numpy(dtype=object)[ok], color_key)]},
                                 key=keyv[ok], key_group="tips", hkey="tips"))  # fmt: skip
    mp["home"] = G.home_box(G.apply(mm, bxy if bxy is not None and len(bxy) else txy), fr)
    counts = np.bincount(codes_m, minlength=len(names) + 1)[: len(names)]
    P.legend_cat(spec, color_key, names, enc_m["colors"], counts, layer="tips")
    spec.frames["_"] = G.inverse(fr)
    spec.layout = {"cols": 2, "rows": 1, "colw": [1.0, 1.4], "aspect": 2.0, "height_hint": 560}
    del codes_t
    return InteractiveFigure(spec, name="tree_tissue")
