"""Interactive forms of the class-switch maps: the CSR map (isotype share maps, transitions) and the CSR flow."""

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
from spaclone.pl.interactive._util import call_text, check_style

__all__ = ["csr_flow", "csr_map"]


def _stream_paths(gx: np.ndarray, gy: np.ndarray, u: np.ndarray, v: np.ndarray, density: float) -> tuple[list, list]:
    """Streamlines of a flow field (matplotlib's integrator), as polylines with their widths, and arrow vertices."""
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot()
    speed = np.hypot(u, v)
    ref = np.nanpercentile(speed, 90) if np.isfinite(speed).any() else 1.0
    lw = 0.4 + 1.4 * np.clip(np.nan_to_num(speed / ref if ref > 0 else speed), 0, 1)
    sp = ax.streamplot(gx, gy, np.ma.masked_invalid(u), np.ma.masked_invalid(v), density=density, linewidth=lw,
                       arrowsize=0.8, broken_streamlines=True)  # fmt: skip
    segs = sp.lines.get_segments()
    widths = np.broadcast_to(np.asarray(sp.lines.get_linewidths(), dtype=float), (len(segs),))
    lines: list[tuple[np.ndarray, float]] = []
    cur: list[np.ndarray] = []
    wsum = 0.0
    for s, w in zip(segs, widths, strict=True):
        if cur and np.allclose(cur[-1], s[0]):
            cur.append(s[1])
            wsum += w
        else:
            if len(cur) >= 2:
                lines.append((np.asarray(cur), wsum / (len(cur) - 1)))
            cur = [s[0], s[1]]
            wsum = w
    if len(cur) >= 2:
        lines.append((np.asarray(cur), wsum / (len(cur) - 1)))
    return lines, []


def csr_flow(
    data: Any,
    displacement: pd.DataFrame | None = None,
    *,
    color: Any = "time",
    grid_um: float | None = None,
    n_grid: int = 40,
    smooth: float = 0.5,
    min_mass: float = 1.0,
    density: float = 1.4,
    style: str = "dark",
    label: str | None = None,
    palette: Mapping[str, str] | None = None,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.csr_flow`: units coloured by the time key, the flow as streamlines.

    A switcher shows the streamlines, grid arrows, or the units alone; hover a unit for its value.

    Parameters
    ----------
    data, displacement, color, grid_um, n_grid, smooth, min_mass, density
        As in :func:`spaclone.pl.csr_flow` (tissue coordinates only).
    style, label, palette, rotate, crop, invert_y, title, x_key, y_key
        As in :func:`spaclone.pl.csr_flow`.
    max_points
        Most units drawn. ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.csr_flow import _as_values, _grid_arrows, _nice_label, _occupied, _state_colors
    from spaclone.tl.transitions import TransitionModel, flow_grid, transition_displacement

    style = check_style(style)
    mode = style
    if isinstance(data, TransitionModel):
        disp = displacement if displacement is not None else transition_displacement(data)
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
    xy0 = disp[[x_key, y_key]].to_numpy(dtype=float)
    fr = G.frame_for(xy0, rotate, crop)
    m = G.affine(fr)
    xy = G.apply(m, xy0)
    vec = disp[["dx_um", "dy_um"]].to_numpy(dtype=float) @ m[:2, :2].T
    d2 = disp.assign(**{x_key: xy[:, 0], y_key: xy[:, 1], "dx_um": vec[:, 0], "dy_um": vec[:, 1]})
    call = call_text(
        "spaclone.pl.interactive.csr_flow", data, color=color if isinstance(color, str) else "<values>", style=style
    )
    spec = P.Spec(style=style, title=title or "class-switch flow", call=call)
    panel = spec.panel("map", "map", area=[0, 0, 1, 1], invert_y=bool(invert_y))
    ch = C.CHROME[mode]
    n = len(xy)
    keep = G.thin(n, max_points)
    idx = np.arange(n) if keep is None else keep
    if keep is not None:
        spec.note(f"units: a random {len(idx):,} of {n:,} are drawn (max_points=None draws all)")
    spacing = G.spacing_um(xy[idx]) if len(idx) > 1 else 10.0
    size = {"um": float(spacing) * 0.9, "min": 1.6, "max": 40.0}
    rows = []
    if values is not None and np.issubdtype(np.asarray(values).dtype, np.number):
        v = np.asarray(values, dtype=float)[idx]
        ramp = list(C.RAMPS["csr"][mode])
        enc, dom = P.num_color(spec, v, ramp, quantiles=(0.0, 1.0))
        rows.append(P.hover_num(spec, v, label or "value"))
        lay = P.points(spec, xy[idx], color=enc, size=size, hover={"rows": rows}, name=label or "units")
        P.legend_colorbar(spec, label or "", ramp, dom)
    elif values is not None:
        labs = pd.Series(np.asarray(values, dtype=object)[idx]).astype(str).to_numpy(dtype=object)
        names = sorted(n_ for n_ in pd.unique(labs) if n_ not in ("nan", "None"))
        cols = _state_colors(names, mode, palette)
        enc, names, codes = P.cat_color(spec, labs, cols, names=names)
        lay = P.points(
            spec, xy[idx], color=enc, size=size, hover={"rows": [P.hover_cat(spec, labs, "label")]}, name="units"
        )
        P.legend_cat(
            spec,
            None,
            names,
            enc["colors"],
            np.bincount(codes, minlength=len(names) + 1)[: len(names)],
            layer=lay["id"],
        )
    else:
        lay = P.points(
            spec, xy[idx], color={"kind": "const", "color": ch["muted"]}, size=size, hover=False, name="units"
        )
    panel["layers"].append(lay)
    g = flow_grid(d2, grid_um=grid_um, n_grid=n_grid, smooth=smooth, min_mass=min_mass, x_key=x_key, y_key=y_key)
    kinds = [("none", "units only")]
    if len(g):
        meta = g.attrs["spaclone"]["flow_grid"]
        h = meta["grid_um"]
        occ = _occupied(xy, meta)
        g = g[occ[g.iy.to_numpy(), g.ix.to_numpy()]]
        nx, ny = meta["nx"], meta["ny"]
        gx = meta["x0"] + h * np.arange(nx)
        gy = meta["y0"] + h * np.arange(ny)
        uu = np.full((ny, nx), np.nan)
        vv = np.full((ny, nx), np.nan)
        uu[g.iy.to_numpy(), g.ix.to_numpy()] = g.dx_um.to_numpy()
        vv[g.iy.to_numpy(), g.ix.to_numpy()] = g.dy_um.to_numpy()
        lines, _ = _stream_paths(gx, gy, uu, vv, density)
        if lines:
            ws = np.array([w for _, w in lines])
            for lo, hi, wpx in ((0, 0.9, 0.9), (0.9, 1.4, 1.5), (1.4, 99, 2.2)):
                sel = [ln for ln, w in lines if lo <= w < hi]
                if not sel:
                    continue
                at = []
                off = 0
                for ln in sel:
                    if len(ln) >= 3:
                        at.append(off + len(ln) // 2 - 1)
                    off += len(ln)
                pl_ = P.paths(spec, sel, stroke=ch["ink"], width=wpx, when={"kind": "stream"}, name="streamlines",
                              arrows={"at": spec.blob(np.asarray(at, dtype=np.int64), "u4"), "size": 7.0, "color": ch["ink"]})  # fmt: skip
                pl_["hover"] = False
                panel["layers"].append(pl_)
            del ws
            kinds.insert(0, ("stream", "streamlines"))
        ua, va = _grid_arrows(g.dx_um.to_numpy(dtype=float), g.dy_um.to_numpy(dtype=float), h)
        okk = np.isfinite(ua) & np.isfinite(va) & ((ua != 0) | (va != 0))
        if okk.any():
            cx, cy = g[x_key].to_numpy(dtype=float)[okk], g[y_key].to_numpy(dtype=float)[okk]
            a0 = np.c_[cx - ua[okk] / 2, cy - va[okk] / 2]
            a1 = np.c_[cx + ua[okk] / 2, cy + va[okk] / 2]
            arrows = [np.vstack([p0, p1]) for p0, p1 in zip(a0, a1, strict=True)]
            pl_ = P.paths(spec, arrows, stroke=ch["ink"], width=1.2, when={"kind": "grid"}, name="grid arrows",
                          arrows={"at": spec.blob(np.arange(len(arrows)) * 2, "u4"), "size": 6.0, "color": ch["ink"], "pos": "end"})  # fmt: skip
            pl_["hover"] = False
            panel["layers"].append(pl_)
            kinds.insert(1 if kinds[0][0] == "stream" else 0, ("grid", "grid arrows"))
    spec.control(
        "kind", "select", label="flow", options=[{"value": k, "label": lab} for k, lab in kinds], value=kinds[0][0]
    )
    panel["home"] = G.home_box(xy, fr)
    hb = panel["home"]
    asp = (hb[1] - hb[0]) / max(hb[3] - hb[2], 1e-9)
    spec.layout = {"cols": 1, "rows": 1, "aspect": float(np.clip(asp * 1.04, 0.75, 3.2)), "height_hint": 620}
    spec.frames["_"] = G.inverse(fr)
    return InteractiveFigure(spec, name="csr_flow")


def csr_map(
    data: Any,
    *,
    species: str = "mouse",
    level: str = "auto",
    style: str = "dark",
    transitions: bool = True,
    transition_test: Any = None,
    n_perm: int = 199,
    rng: Any = 0,
    csr: Any = None,
    grid: Any = None,
    tissue: Any = None,
    sigma_um: float | None = None,
    min_neff: float | None = None,
    pixel_um: float = 10.0,
    min_loci: int = 1,
    rotate: bool | float = True,
    invert_y: bool = True,
    title: str | None = None,
    isotype_key: str | None = None,
    subclass_key: str | None = None,
    clone_key: str = S.CLONE,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.csr_map`: one map with a switcher over the isotype maps.

    The switcher lists the switched share (CSR hotspots outlined), the dominant switched class and the share map of
    every isotype in locus order; next to it, the class switches within clones as an upstream x downstream matrix
    (hover a cell for its clones and q-value).

    Parameters
    ----------
    data, species, level, style, transition_test, n_perm, rng, csr, grid, tissue
        As in :func:`spaclone.pl.csr_map`.
    sigma_um, min_neff, pixel_um, min_loci, rotate, invert_y, isotype_key, subclass_key, clone_key, x_key, y_key
        As in :func:`spaclone.pl.csr_map`.
    transitions
        Draw the transitions matrix (the static figure's arcs become a matrix here).
    title
        Figure title.

    Returns
    -------
    InteractiveFigure
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from spaclone.pl.atlas import _continuous_rgba, _fill_from, _Frame, _labels_array, _mask_fill, _mask_lines
    from spaclone.pl.csr_map import (
        _ORDER,
        _SWITCHED,
        _CSRInputs,
        _dominant_rgba,
        _iso_color,
        _panel_names,
        _ramp,
        _share_maps,
        _switched_map,
        _transitions,
    )

    style = check_style(style)
    mode = style
    if species not in _ORDER:
        raise ValueError(f"species must be one of {sorted(_ORDER)}")
    pal = C.ATLAS_STYLES[mode]
    inp = _CSRInputs(data, species=species, csr=csr, grid=grid, tissue=tissue, sigma_um=sigma_um, min_neff=min_neff,
                     pixel_um=pixel_um, isotype_key=isotype_key, subclass_key=subclass_key, clone_key=clone_key,
                     x_key=x_key, y_key=y_key)  # fmt: skip
    cls, lab, lvl = inp.labels(level)
    names = _panel_names(lab, species, lvl, min_loci)
    iso_maps = _share_maps(inp, lab, names)
    class_maps = (
        _share_maps(inp, cls, list(_SWITCHED)) if lvl == "subclass" else {c: iso_maps.get(c) for c in _SWITCHED}
    )
    tis = np.asarray(inp.tissue.values, dtype=bool)
    frame = _Frame(inp.grid, tis, rotate)
    m = G.affine(frame)
    call = call_text("spaclone.pl.interactive.csr_map", data, species=species, level=level, style=style)
    spec = P.Spec(style=style, title=title or "class switching on the tissue", call=call)
    test = transition_test
    if transitions and test is None:
        try:
            test = _transitions(inp, cls, lab, lvl, n_perm, rng)
        except (ValueError, KeyError) as err:
            spec.note(f"no transitions panel: {err}")
            test = None
    two = bool(transitions and test is not None and len(getattr(test, "table", [])))
    panel = spec.panel("map", "map", area=[0, 0, 1, 1], invert_y=bool(invert_y))
    fill = [G.apply(m, r) for r in G.rings_from_paths(_mask_fill(tis, inp.grid, 1.5))]
    clip = None
    if fill:
        tl = P.paths(spec, fill, closed=True, fill=pal["tissue"], name="tissue")
        tl["hover"] = False
        panel["layers"].append(tl)
        clip = tl["id"]
    options = []
    sm_sw = _switched_map(inp, cls)

    def add_map(key: str, label: str, sm: Any, ramp: Sequence[str], legend_label: str) -> None:
        if sm is None:
            return
        vals = np.asarray(sm.share, dtype=float)
        elig = np.asarray(sm.eligible, dtype=bool) & tis & np.isfinite(vals)
        fin = vals[elig]
        lo, hi = np.quantile(fin, [0.02, 0.995]) if len(fin) else (0.0, 1.0)
        hi = float(hi) if hi > lo else float(lo) + 1e-9
        filled, support = _fill_from(vals, elig)
        rgba = _continuous_rgba(filled, elig, support, tis, LinearSegmentedColormap.from_list("spc", list(ramp)),
                                Normalize(float(lo), hi), pal["tissue"])  # fmt: skip
        panel["layers"].append(P.image(spec, G.grid_rows(rgba), G.grid_image(inp.grid, m), values=G.grid_rows(np.where(elig, vals, np.nan)),
                                       value_label=legend_label, value_fmt="pct", clip=clip, when={"map": key}, name=label))  # fmt: skip
        P.legend_colorbar(spec, legend_label, ramp, (float(lo), hi), fmt="pct0", when={"map": key},
                          note="smoothed share of loci with an isotype; 2nd to 99.5th percentile")  # fmt: skip
        options.append((key, label))

    add_map("switched", "switched share", sm_sw, pal["csr_ramp"], "switched share (IgG, IgA, IgE)")
    rgba, present = _dominant_rgba(class_maps, tis, style, pal)
    if rgba is not None:
        panel["layers"].append(P.image(spec, G.grid_rows(rgba), G.grid_image(inp.grid, m), clip=clip, when={"map": "dominant"},
                                       name="dominant switched class"))  # fmt: skip
        P.legend_cat(spec, "dominant switched class (brighter: more switched)", present, [_iso_color(c, style) for c in present],
                     None, shape="square", when={"map": "dominant"})  # fmt: skip
        options.append(("dominant", "dominant switched class"))
    for nm in names:
        add_map(f"iso:{nm}", nm, iso_maps.get(nm), _ramp(_iso_color(nm.split(" ")[0], style), style), f"{nm} share")
    edge = [G.apply(m, v) for v in _mask_lines(tis, inp.grid, 1.5)]
    if edge:
        el = P.paths(spec, edge, stroke=pal["tissue_edge"], width=0.9, name="tissue edge")
        el["hover"] = False
        panel["layers"].append(el)
    hot = _labels_array(inp.csr, inp.grid) if inp.csr is not None else None
    hot_lines = _mask_lines(hot > 0, inp.grid, 0.7) if hot is not None else []
    if hot_lines:
        hl = P.paths(spec, [G.apply(m, v) for v in hot_lines], stroke=pal["csr"], width=1.4, halo=pal["bg"], name="CSR hotspot",
                     okey="csr_hotspots")  # fmt: skip
        hl["hover"] = False
        panel["layers"].append(hl)
        P.legend_lines(spec, "outlines", [("CSR hotspot", pal["csr"], "line")], layers=["csr_hotspots"])
    box = frame.section_box(tis)
    panel["home"] = [float(v) for v in box]
    asp = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    spec.control("map", "select", label="map", options=[{"value": k, "label": lab_} for k, lab_ in options],
                 value=options[0][0] if options else None)  # fmt: skip
    if two:
        tab = test.table
        up = (
            tab.drop_duplicates("upstream")
            .sort_values("upstream_position", kind="stable")
            .upstream.astype(str)
            .tolist()
        )
        dn = (
            tab.drop_duplicates("downstream")
            .sort_values("downstream_position", kind="stable")
            .downstream.astype(str)
            .tolist()
        )
        nms = list(dict.fromkeys(up + dn))
        k = len(nms)
        pos = {n_: i for i, n_ in enumerate(nms)}
        nb = np.full((k, k), np.nan)
        qv = np.full((k, k), np.nan)
        for r in tab.itertuples():
            nb[pos[str(r.upstream)], pos[str(r.downstream)]] = getattr(r, "n_clones_both", np.nan)
            qv[pos[str(r.upstream)], pos[str(r.downstream)]] = getattr(r, "q_value", np.nan)
        p2 = spec.panel("switches", "plot", area=[1, 0, 1, 1], title="class switches within clones",
                        axes={"x": {"scale": "cat", "cats": nms, "rotate": True, "label": "downstream (3')"},
                              "y": {"scale": "cat", "cats": nms, "flip": True, "label": "upstream (5')"}, "zoom": "none",
                              "square": True})  # fmt: skip
        ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        ok = np.isfinite(nb.ravel())
        boxes = np.c_[jj.ravel() - 0.47, ii.ravel() - 0.47, jj.ravel() + 0.47, ii.ravel() + 0.47][ok]
        vals = nb.ravel()[ok]
        ramp = list(C.RAMPS["blue"][mode])
        lab_txt = [
            f"{int(v)}{'*' if np.isfinite(q_) and q_ <= 0.05 else ''}"
            for v, q_ in zip(vals, qv.ravel()[ok], strict=True)
        ]
        pairs = [f"{nms[i]} → {nms[j]}" for i, j in zip(ii.ravel()[ok], jj.ravel()[ok], strict=True)]
        p2["layers"].append(P.rects(spec, boxes, fill={"values": spec.blob(vals, "f4"), "ramp": [P._hex(c) for c in ramp],
                                                       "domain": [0.0, float(max(vals.max(), 1.0)) if len(vals) else 1.0]},
                                    labels=lab_txt, hover={"rows": [{"label": "switch", "list": pairs},
                                                                    {"label": "clones with both", "list": [str(int(v)) for v in vals]},
                                                                    {"label": "q", "list": [("–" if not np.isfinite(q_) else f"{q_:.3g}") for q_ in qv.ravel()[ok]]}]}))  # fmt: skip
        spec.note("matrix: clones holding loci of both isotypes (* q ≤ 0.05, within-clone shuffles)")
        spec.layout = {
            "cols": 2,
            "rows": 1,
            "colw": [1.7, 1.0],
            "aspect": float(np.clip(asp * 1.6, 1.2, 3.4)),
            "height_hint": 600,
        }
    else:
        spec.layout = {"cols": 1, "rows": 1, "aspect": float(np.clip(asp * 1.04, 0.75, 3.2)), "height_hint": 600}
    spec.frames["_"] = G.inverse(frame)
    return InteractiveFigure(spec, name="csr_map")
