"""Interactive forms of the share maps and of the hotspot atlas (overview, atlas, region zoom)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl import _colors as C
from spaclone.pl.interactive import _geo as G
from spaclone.pl.interactive import _spec as P
from spaclone.pl.interactive._figure import InteractiveFigure
from spaclone.pl.interactive._hotspots import HotspotLayers, atlas_data
from spaclone.pl.interactive._util import call_text, check_style

__all__ = ["csr_share_map", "hotspot_atlas", "hotspot_overview", "region_zoom", "share_map", "shm_share_map"]


def _short_label(label: str) -> str:
    from spaclone.pl.atlas import _short_label as short

    return short(label)


_LAYER_LABEL = {
    "share": "share (smoothed)",
    "expected": "expected share",
    "z": "z (binomial residual)",
    "z_perm": "z (permutation-studentised)",
    "n_eff": "effective units",
}


def _layer_values(result: Any, sm: Any, layer: str) -> np.ndarray:
    if layer == "z_perm":
        vals = np.asarray(result.z_map.values, dtype=float)
        return np.where(np.asarray(sm.eligible, dtype=bool) & np.isfinite(vals), vals, np.nan)
    return np.where(np.asarray(sm.eligible, dtype=bool), np.asarray(getattr(sm, layer), dtype=float), np.nan)


def share_map(
    result: Any,
    *,
    layer: str | None = None,
    style: str = "dark",
    receptor: str = "BCR",
    vmin: float | None = None,
    vmax: float | None = None,
    quantiles: tuple[float, float] = (0.02, 0.995),
    hotspots: bool = True,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    units: pd.DataFrame | None = None,
    unit_layers: Sequence[dict[str, Any]] = (),
    _ramp: Sequence[str] | None = None,
    _hot_color: str | None = None,
    _hot_name: str = "hotspot",
    _name: str = "share_map",
    _call: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.share_map`: a share map with a switcher over its layers.

    The switcher lists the map's layers (share, expected share, z, effective units, and the permutation-studentised z
    of a hotspot result); hover a pixel for its value. Hotspots are outlined (hot solid, cold dashed) and every outline
    can be shown or hidden from the legend.

    Parameters
    ----------
    result, layer, style, receptor, vmin, vmax, quantiles
        As in :func:`spaclone.pl.share_map` (``vmin`` / ``vmax`` apply to the first layer shown).
    hotspots, outlines, outline_colors, rotate, crop, invert_y, title
        As in :func:`spaclone.pl.share_map`.
    units, unit_layers
        Internal (used by :func:`csr_share_map` and :func:`shm_share_map`).

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.atlas import _continuous_rgba, _fill_from, _mask_fill, _mask_lines
    from spaclone.pl.maps import _display_tissue, _resolve_layer

    style = check_style(style)
    mode = style
    sm = result.map if getattr(result, "map", None) is not None else result
    labels = getattr(result, "labels", None)
    first = _resolve_layer(result, layer)
    layers = [lay for lay in ("share", "expected", "z", "z_perm", "n_eff")
              if (lay != "z_perm" or getattr(result, "z_map", None) is not None) and hasattr(sm, lay if lay != "z_perm" else "share")]  # fmt: skip
    grid = sm.grid
    tissue = _display_tissue(sm)
    gx, gy = grid.centres()
    valid0 = np.asarray(sm.eligible, dtype=bool)
    ii, jj = np.nonzero(tissue | valid0)
    pts = np.column_stack([gx[ii], gy[jj]]) if len(ii) else np.column_stack([gx[[0, -1]], gy[[0, -1]]])
    from spaclone.pl._frame import SectionFrame

    fr = SectionFrame(pts, rotate, crop=crop)
    m = G.affine(fr)
    call = _call or call_text("spaclone.pl.interactive.share_map", result, layer=layer, style=style)
    spec = P.Spec(style=style, title=title or "share map", call=call)
    panel = spec.panel("map", "map", area=[0, 0, 1, 1], invert_y=bool(invert_y))
    pal = {"bg": C.CHROME[mode]["bg"], "tissue": C.CHROME[mode]["tissue"], "edge": C.ATLAS_STYLES[mode]["tissue_edge"]}
    fill = [G.apply(m, r) for r in G.rings_from_paths(_mask_fill(tissue, grid, 1.5))]
    clip_id = None
    if fill:
        tl = P.paths(spec, fill, closed=True, fill=pal["tissue"], name="tissue")
        tl["hover"] = False
        panel["layers"].append(tl)
        clip_id = tl["id"]
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ramp = (
        list(_ramp)
        if _ramp is not None
        else list(C.RAMPS[{"BCR": "csr", "TCR": "tcr"}.get(receptor.upper(), "csr")][mode])
    )
    for lay_name in layers:
        vals = _layer_values(result, sm, lay_name)
        fin = vals[np.isfinite(vals)]
        if lay_name in ("z", "z_perm"):
            lim = float(np.quantile(np.abs(fin), quantiles[1])) if len(fin) else 1.0
            lim = lim if lim > 0 else 1.0
            lo, hi = -lim, lim
            stops = list(C.DIVERGING[mode])
        else:
            q = np.quantile(fin, quantiles) if len(fin) else (0.0, 1.0)
            lo, hi = float(q[0]), float(q[1])
            stops = ramp
        if lay_name == first:
            lo = vmin if vmin is not None else lo
            hi = vmax if vmax is not None else hi
        hi = hi if hi > lo else lo + 1e-9
        cmap = LinearSegmentedColormap.from_list("spc", stops)
        valid = valid0 & np.isfinite(vals)
        filled, support = _fill_from(vals, valid)
        rgba = _continuous_rgba(filled, valid, support, tissue, cmap, Normalize(lo, hi), pal["tissue"])
        lab = _LAYER_LABEL[lay_name]
        if lay_name == "z_perm" and (getattr(result, "params", None) or {}).get("standardise") == "binomial":
            lab = _LAYER_LABEL["z"]
        fmt = "pct" if lay_name in ("share", "expected") else None
        il = P.image(spec, G.grid_rows(rgba), G.grid_image(grid, m), values=G.grid_rows(vals), value_label=lab,
                     value_fmt=fmt, clip=clip_id, when={"layer": lay_name}, name=lab)  # fmt: skip
        panel["layers"].append(il)
        ticks = None
        thr = (getattr(result, "threshold", None) or {}).get("hot")
        note = None
        if (
            lay_name == "z_perm"
            and thr is not None
            and np.isfinite(thr)
            and (getattr(result, "params", None) or {}).get("correction") == "maxT"
        ):
            note = f"map-level hotspot threshold: z = {thr:.2f}"
        P.legend_colorbar(spec, lab, stops, (lo, hi), fmt="pct0" if fmt else None, when={"layer": lay_name}, note=note,
                          ticks=ticks)  # fmt: skip
    edge = [G.apply(m, v) for v in _mask_lines(tissue, grid, 1.5)]
    if edge:
        el = P.paths(spec, edge, stroke=pal["edge"], width=0.9, name="tissue edge")
        el["hover"] = False
        panel["layers"].append(el)
    line_items, line_keys = [], []
    hot_c = _hot_color or C.MARKS[mode]["hot"]
    if hotspots and labels is not None:
        tab = getattr(result, "table", None)
        lab = np.asarray(labels.values)
        has = tab is not None and len(tab)
        for direction, col, dash, name in (
            ("hot", hot_c, None, _hot_name),
            ("cold", C.MARKS[mode]["cold"], [5, 4], "cold spot"),
        ):
            ids = tab.hotspot[tab.direction == direction].to_numpy() if has else []
            lines = _mask_lines(np.isin(lab, ids), grid, 0.7) if len(ids) else []
            if not lines:
                continue
            pl_ = P.paths(spec, [G.apply(m, v) for v in lines], stroke=col, width=1.5, dash=dash, halo=pal["bg"],
                          name=name, okey=f"out:{direction}")  # fmt: skip
            pl_["hover"] = False
            panel["layers"].append(pl_)
            line_items.append((name, col, "dash" if dash else "line"))
            line_keys.append(f"out:{direction}")
    extra = C.OUTLINE_SLOTS[mode]
    k = 0
    for name, mask in (outlines or {}).items():
        col = (outline_colors or {}).get(name)
        if col is None:
            col = extra[k % len(extra)]
            k += 1
        mk = mask.values if hasattr(mask, "values") else np.asarray(mask)
        lines = _mask_lines(np.asarray(mk, dtype=bool), grid, 0.7)
        if not lines:
            continue
        pl_ = P.paths(spec, [G.apply(m, v) for v in lines], stroke=col, width=1.2, halo=pal["bg"], name=name,
                      okey=f"out:{name}")  # fmt: skip
        pl_["hover"] = False
        panel["layers"].append(pl_)
        line_items.append((name, col, "line"))
        line_keys.append(f"out:{name}")
    for ul in unit_layers:
        panel["layers"].append(ul["layer"](spec, m))
        if ul.get("legend"):
            ul["legend"](spec)
    P.legend_lines(spec, "outlines", line_items, layers=line_keys)
    spec.control(
        "layer", "select", label="layer", options=[{"value": v, "label": _LAYER_LABEL[v]} for v in layers], value=first
    )
    box = fr.extent or (float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max()))
    pad = 0.75 * float(grid.pixel_um)
    panel["home"] = [box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad]
    asp = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    spec.layout = {"cols": 1, "rows": 1, "aspect": float(np.clip(asp * 1.04, 0.75, 3.2)), "height_hint": 600}
    spec.frames["_"] = G.inverse(fr)
    return InteractiveFigure(spec, name=_name)


def _unit_dots(units: pd.DataFrame, mask: np.ndarray, colors: Any, *, x_key: str, y_key: str, um: float, edge: str | None,
               name: str, hover_cols: Sequence[str]) -> Any:  # fmt: skip
    """A deferred points layer of units (built once the map's frame is known)."""

    def build(spec: P.Spec, m: np.ndarray) -> dict[str, Any]:
        sub = units[mask]
        xy = G.apply(m, sub[[x_key, y_key]].to_numpy(dtype=float))
        if isinstance(colors, str):
            color = {"kind": "const", "color": colors}
        else:
            enc, _, _ = P.cat_color(
                spec, np.asarray(colors, dtype=object)[mask], {c: c for c in set(np.asarray(colors)[mask])}
            )
            color = enc
        rows = []
        for c in hover_cols:
            if c in sub.columns:
                v = sub[c]
                rows.append(P.hover_num(spec, pd.to_numeric(v, errors="coerce"), c) if pd.api.types.is_numeric_dtype(v)
                            and not pd.api.types.is_bool_dtype(v) else P.hover_cat(spec, v.astype("string").to_numpy(dtype=object), c))  # fmt: skip
        title = (
            P.hover_cat(spec, "clone " + sub[S.CLONE].astype("string").fillna("NA").astype(str), "clone")
            if S.CLONE in sub.columns
            else None
        )
        lay = P.points(spec, xy, color=color, size={"um": um, "min": 2.5, "max": 30}, edge={"color": edge, "px": 0.6} if edge else None,
                       name=name, okey=f"dots:{name}", hover={"title": title, "rows": rows} if title else {"rows": rows})  # fmt: skip
        return lay

    return build


def shm_share_map(
    result: Any,
    *,
    style: str = "dark",
    csr: Any = None,
    units: pd.DataFrame | None = None,
    mark: str = "mutated",
    x_key: str = S.X,
    y_key: str = S.Y,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.shm_share_map`: the mutated share (green), SHM hotspots in vermilion.

    Parameters
    ----------
    result, style, csr, units, mark, x_key, y_key, outlines, outline_colors
        As in :func:`spaclone.pl.shm_share_map` (``units``: mutated units in vermilion, the others grey; each set can
        be hidden from the legend).
    **kwargs
        Passed to :func:`share_map` (``layer``, ``vmin``, ``vmax``, ``rotate``, ``title``, ...).

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    mode = style
    lines = dict(outlines or {})
    colors = dict(outline_colors or {})
    if csr is not None:
        lab2 = getattr(csr, "labels", csr)
        vals = lab2.values if hasattr(lab2, "values") else np.asarray(lab2)
        lines["CSR hotspot"] = np.asarray(vals) > 0
        colors["CSR hotspot"] = C.MARKS[mode]["csr"]
    uls = []
    if units is not None:
        m_ = units[mark].fillna(False).astype(bool).to_numpy()
        uls.append({"layer": _unit_dots(units, ~m_, C.MARKS[mode]["nomut"], x_key=x_key, y_key=y_key, um=4.0, edge=None,
                                        name="not mutated", hover_cols=("isotype", "excess", "n_records"))})  # fmt: skip
        uls.append({"layer": _unit_dots(units, m_, C.MARKS[mode]["shm"], x_key=x_key, y_key=y_key, um=7.0,
                                        edge=C.CHROME["light"]["bg"], name="mutated", hover_cols=("isotype", "excess", "n_records"))})  # fmt: skip
    call = call_text("spaclone.pl.interactive.shm_share_map", result, csr=csr, units=units, style=style)
    fig = share_map(result, style=style, outlines=lines, outline_colors=colors, unit_layers=uls,
                    _ramp=C.RAMPS["shm"][mode], _hot_color=C.MARKS[mode]["shm"], _hot_name="SHM hotspot",
                    _name="shm_share_map", _call=call, title=kwargs.pop("title", None) or "SHM · mutated share",
                    **kwargs)  # fmt: skip
    if units is not None:
        P.legend_cat(
            fig._spec, "units", ["mutated", "not mutated"], [C.MARKS[mode]["shm"], C.MARKS[mode]["nomut"]], None
        )
    return fig


def csr_share_map(
    result: Any,
    *,
    style: str = "dark",
    units: pd.DataFrame | None = None,
    isotype_key: str = "isotype",
    mark: str = "switched",
    x_key: str = S.X,
    y_key: str = S.Y,
    outlines: Mapping[str, Any] | None = None,
    outline_colors: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.csr_share_map`: the switched share (blue), CSR hotspots in yellow.

    Parameters
    ----------
    result, style, units, isotype_key, mark, x_key, y_key, outlines, outline_colors
        As in :func:`spaclone.pl.csr_share_map` (``units``: switched units as dots coloured by isotype).
    **kwargs
        Passed to :func:`share_map`.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.bcr import _colors_for

    style = check_style(style)
    mode = style
    uls = []
    if units is not None:
        sw = units[mark].fillna(False).astype(bool).to_numpy()
        iso = (
            units[isotype_key].astype("string").fillna("unknown").astype(str).to_numpy()
            if isotype_key in units
            else np.full(len(units), "unknown")
        )
        cols = _colors_for(["IgG", "IgA", "IgE", "unknown"], None, style)
        colors = np.array([cols.get(v, cols["unknown"]) for v in iso], dtype=object)
        uls.append({"layer": _unit_dots(units, sw, colors, x_key=x_key, y_key=y_key, um=5.0, edge=None, name="switched",
                                        hover_cols=(isotype_key, "n_records", "switched_share"))})  # fmt: skip
    call = call_text("spaclone.pl.interactive.csr_share_map", result, units=units, style=style)
    return share_map(result, style=style, outlines=outlines, outline_colors=outline_colors, unit_layers=uls,
                     _ramp=C.RAMPS["csr"][mode], _hot_color=C.MARKS[mode]["csr"], _hot_name="CSR hotspot",
                     _name="csr_share_map", _call=call, title=kwargs.pop("title", None) or "CSR · switched share",
                     **kwargs)  # fmt: skip


# ------------------------------------------------------------------------------------------------ the hotspot atlas


def _atlas(
    result: Any,
    *,
    channels: Sequence[str],
    region: Any,
    style: str,
    rotate: bool | float,
    invert_y: bool,
    window_pad_um: float,
    min_window_um: float,
    title: str,
    call: str,
    name: str,
    regions: Any = None,
    max_regions: int | None = None,
) -> InteractiveFigure:
    style = check_style(style)
    rot = atlas_data(result, style=style, rotate=rotate, invert_y=invert_y)
    m = G.affine(rot.frame)
    spec = P.Spec(style=style, title=title, call=call)
    data = rot
    ids = list(data.regions.index)
    if regions is not None and not (isinstance(regions, int | np.integer) and not isinstance(regions, bool)):
        want = list(regions)
        missing = [r for r in want if r not in ids]
        if missing:
            raise KeyError(f"regions {missing} not in result.regions (available: {ids})")
        ids = [r for r in ids if r in want]
    elif isinstance(regions, int | np.integer) and not isinstance(regions, bool):
        ids = ids[: int(regions)]
    box = data.frame.section_box(data.tissue)
    sec_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    windows = {str(r): [float(v) for v in data.window(r, 1.0, window_pad_um, min_window_um)] for r in ids}
    homes = {"all": sec_box} | {k: [w[0], w[1], w[2], w[3]] for k, w in windows.items()}
    first = "all" if region is None else str(region)
    if first not in homes:
        raise KeyError(f"region {region!r} not in result.regions (available: {ids})")
    names = {"csr": "CSR · switched share", "shm": "SHM · mutated share", "gex": "transcriptome"}
    n = len(channels)
    flat = atlas_data(result, style=style, rotate=False, invert_y=invert_y)  # original µm; shared by the panels
    for k, ch in enumerate(channels):
        panel = spec.panel(ch, "map", area=[k, 0, 1, 1], invert_y=bool(invert_y), link="section", home_by="region",
                           home_map=homes, title=names[ch], pad=0.02)  # fmt: skip
        if ch == "gex":  # as the static atlas: the transcriptome tile belongs to a region's zoom, not to the overview
            panel["when"] = {"region": [str(r) for r in ids]}
        hl = HotspotLayers(spec, panel, result, m, style=style, data=flat)
        hl.tissue()
        if ch in ("csr", "shm"):
            if hl.share_map(ch, legend=True) is None:
                panel["notes_when"] = [{"when": {}, "text": f"no {ch.upper()} map in the result"}]
            hl.add_edge()
            hl.hotspot_outline("csr", okey="csr_hotspots")
            if ch == "shm":
                hl.hotspot_outline("shm", okey="shm_hotspots")
            hl.loci_dots(ch, okey=f"dots_{ch}", min_zoom=None)
            for lay in panel["layers"]:
                if str(lay.get("okey", "")).startswith(f"dots_{ch}"):
                    lay["max_view_um"] = 1500.0
        else:
            hl.add_edge()
            hl.gc_outline(okey="gc")
            for r in ids:
                lab = data.regions.loc[r].get("gex_label") if "gex_label" in data.regions.columns else None
                if isinstance(lab, str) and lab.strip():
                    hl.module_image(lab, second=data.regions.loc[r].get("gex_second_label"), when={"region": str(r)})
            # move module images under the outlines: images were appended after the outlines
            imgs = [q for q in panel["layers"] if q["type"] == "image"]
            rest = [q for q in panel["layers"] if q["type"] != "image"]
            panel["layers"] = rest[:1] + imgs + rest[1:]
        reg, _ = hl.regions(okey="regions", control="region", width=2.0 if ch != "gex" else 1.6, labels=False,
                            when={"region": [str(r) for r in ids]})  # fmt: skip
        # numbered windows of the overview
        rects = []
        labs = []
        for r in ids:
            w = windows[str(r)]
            rects.append([w[0], w[2], w[1], w[3]])
            labs.append(str(r))
        if rects:
            rl = P.rects(spec, rects, stroke=C.ATLAS_STYLES[style]["frame"], width=0.9, when={"region": "all"},
                         hover={"rows": [{"list": [data.region_name(r) for r in ids], "label": "window"}]},
                         actions=[{"set": ["region", str(r)]} for r in ids])  # fmt: skip
            rl["okey"] = "windows"
            panel["layers"].append(rl)
            tl = P.text(spec, [(w[0], w[2]) for w in (windows[str(r)] for r in ids)], labs, size=12, anchor="start",
                        baseline="bottom", dy=-2, when={"region": "all"})  # fmt: skip
            tl["okey"] = "windows"
            panel["layers"].append(tl)
        tw = []
        for r in ids:
            row = data.regions.loc[r]
            if ch == "csr":
                share = row.get("switched_share")
                t = (data.region_name(r) if data.from_csr(r) else f"CSR · region {r}") + (
                    f" · {float(share):.0%} switched" if pd.notna(share) else "")  # fmt: skip
            elif ch == "shm":
                share = row.get("mutated_share")
                t = "SHM" + (f" · {float(share):.0%} mutated" if pd.notna(share) else "")
            else:
                lab = row.get("gex_label")
                t = (f"{data.region_name(r)} · {_short_label(lab)}" if isinstance(lab, str) and lab.strip()
                     else f"{data.region_name(r)} · GEX n/a")  # fmt: skip
            tw.append({"when": {"region": str(r)}, "title": t})
        panel["title_when"] = tw
        subs = []  # the region's clone-level p-values: a second line of the panel title (the static atlas writes
        # them on the region's transcriptome tile)
        for r in ids:
            line = data.clone_p_line(r)
            if line and (ch == "gex" or (ch == channels[0] and "gex" not in channels)):
                subs.append({"when": {"region": str(r)}, "text": line})
        if subs:
            panel["subtitle_when"] = subs
        void = reg
        del void
    # legends
    pal = C.ATLAS_STYLES[style]
    items = [("CSR hotspot", pal["csr"], "line")]
    keys: list[str | None] = ["csr_hotspots"]
    if data.hotspot_lines("shm"):
        items.append(("SHM hotspot", pal["shm"], "line"))
        keys.append("shm_hotspots")
    if ids:
        items.append(("region (click: zoom)", pal["csr"], "box"))
        keys.append("regions")
        items.append(("window", pal["frame"], "box"))
        keys.append("windows")
    if "gex" in channels and data.gc_lines():
        items.append(("GC B (transcriptome)", pal["gcb"], "line"))
        keys.append("gc")
    P.legend_lines(spec, "outlines", items, layers=keys)
    if "csr" in channels:
        _, cls = data.isotype_dots()
        present = [c for c in ("IgG", "IgA", "IgE") if (cls == c).any()]
        P.legend_lines(spec, "loci (zoomed in)", [(c, pal["iso"][c], "line") for c in present] + [("IgM / IgD", pal["nomut"], "line")],
                       layers=[f"dots_csr:{c}" for c in present] + ["dots_csr:not switched"])  # fmt: skip
    if "shm" in channels:
        P.legend_lines(spec, None, [(data.mutation_label(), pal["mut"], "line"), ("not mutated", pal["nomut"], "line")],
                       layers=[f"dots_shm:{data.mutation_label()}", "dots_shm:not mutated"])  # fmt: skip
    for sec in spec.legend:
        for it in sec.get("items", []):
            if it.get("shape") == "line" and it.get("layer", "").startswith("dots"):
                it["shape"] = "point"
    if ids:
        spec.control("region", "select", label="region",
                     options=[{"value": "all", "label": "whole section"}] + [{"value": str(r), "label": data.region_name(r)} for r in ids],
                     value=first)  # fmt: skip
    else:
        spec.control(
            "region", "select", label="region", options=[{"value": "all", "label": "whole section"}], value="all"
        )
        spec.note("no CSR ∩ SHM region and no CSR hotspot region in this result")
    asp = (sec_box[1] - sec_box[0]) / max(sec_box[3] - sec_box[2], 1e-9)
    spec.layout = {"cols": n, "rows": 1, "aspect": float(np.clip(n * asp * 1.02, 0.8, 4.5)), "height_hint": 560,
                   "reflow": "gex" in channels}  # fmt: skip
    spec.frames["_"] = G.inverse(rot.frame)
    if max_regions is not None and len(data.regions) > len(ids):
        spec.note(f"{len(ids)} of {len(data.regions)} regions listed (regions= chooses)")
    return InteractiveFigure(spec, name=name)


def hotspot_overview(
    result: Any,
    *,
    regions: Any = None,
    style: str = "dark",
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.hotspot_overview`: the CSR and SHM maps, zoomed together.

    The numbered windows mark the regions; click one (or pick it in the region list) to zoom both maps to it. Zoomed
    in (a view narrower than 1.5 mm), the loci appear as dots: switched loci by isotype (IgG red, IgA orange, IgE
    violet) over unswitched ones (grey), mutated loci in vermilion.

    Parameters
    ----------
    result, regions, style, rotate, invert_y, window_pad_um, min_window_um
        As in :func:`spaclone.pl.hotspot_overview`.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    call = call_text("spaclone.pl.interactive.hotspot_overview", result, regions=regions, style=style)
    return _atlas(result, channels=("csr", "shm"), region=None, style=style, rotate=rotate, invert_y=invert_y,
                  window_pad_um=window_pad_um, min_window_um=min_window_um, title=title or "CSR and SHM hotspots",
                  call=call, name="hotspot_overview", regions=regions)  # fmt: skip


def hotspot_atlas(
    result: Any,
    *,
    regions: Any = None,
    style: str = "dark",
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.hotspot_atlas`: CSR map, SHM map and transcriptome, zoomed together.

    The whole section shows the CSR and SHM maps, as the static overview. Pick a region (list or click) to zoom both
    to it; a third panel then shows the region's transcriptome tile, as the static zoom does: the module score of the
    cell type the region's profile points to, with the region outline and the germinal-centre B area, and the
    region's clone-level p-values under its title.

    Parameters
    ----------
    result, regions, style, rotate, invert_y, window_pad_um, min_window_um
        As in :func:`spaclone.pl.hotspot_atlas`.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    call = call_text("spaclone.pl.interactive.hotspot_atlas", result, regions=regions, style=style)
    return _atlas(result, channels=("csr", "shm", "gex"), region=None, style=style, rotate=rotate, invert_y=invert_y,
                  window_pad_um=window_pad_um, min_window_um=min_window_um, title=title or "hotspot atlas", call=call,
                  name="hotspot_atlas", regions=regions)  # fmt: skip


def region_zoom(
    result: Any,
    region_id: Any,
    *,
    channels: str | Sequence[str] = ("csr", "shm", "gex"),
    style: str = "dark",
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    title: str | None = None,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.region_zoom`: the hotspot atlas opened on one region.

    Parameters
    ----------
    result, region_id, channels, style, rotate, invert_y, window_pad_um, min_window_um
        As in :func:`spaclone.pl.region_zoom`; the other regions stay one click away.
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    chans = [channels] if isinstance(channels, str) else list(channels)
    bad = [c for c in chans if c not in ("csr", "shm", "gex")]
    if bad or not chans:
        raise ValueError(f"channels must be among ('csr', 'shm', 'gex'), got {channels!r}")
    call = call_text("spaclone.pl.interactive.region_zoom", result, region_id, channels=channels, style=style)
    return _atlas(result, channels=chans, region=region_id, style=style, rotate=rotate, invert_y=invert_y,
                  window_pad_um=window_pad_um, min_window_um=min_window_um, title=title or f"region {region_id}",
                  call=call, name="region_zoom")  # fmt: skip
