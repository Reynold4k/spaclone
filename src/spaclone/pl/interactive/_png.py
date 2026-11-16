"""The static picture of an interactive figure: its first view drawn with matplotlib from the same specification.

The picture is what a notebook frontend shows when it does not run scripts (GitHub's viewer, nbconvert to PDF, an
untrusted notebook) and what a saved file shows if its engine cannot start. It is drawn from the specification and
the arrays behind it, so it shows the same layers, colours and view as the interactive figure before any interaction.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from spaclone.pl._colors import CHROME

#: Width of the picture in pixels (at 100 dpi) and its largest height.
WIDTH_PX = 1100
MAX_HEIGHT_PX = 1400
_DPI = 100.0


def _matches(when: dict[str, Any] | None, state: dict[str, Any]) -> bool:
    if not when:
        return True
    for k, want in when.items():
        have = state.get(k)
        if isinstance(want, list):
            if have not in want:
                return False
        elif want != have:
            return False
    return True


def initial_state(spec: dict[str, Any]) -> dict[str, Any]:
    """Control values of the first view."""
    return {c["id"]: c.get("value") for c in spec.get("controls", [])}


def _z(L: dict[str, Any]) -> float:
    """Drawing order on a map, as the figure: filled shapes (tissue), images, links, points, lines, outlines, labels."""
    if "z" in L:
        return float(L["z"])
    t = L.get("type")
    if t == "paths" and L.get("fill"):
        return 0.0
    if t == "image":
        return 0.5
    if t == "links":
        return 1.5
    if t == "points" or (t == "rects" and L.get("fill")):
        return 2.0
    if t == "segs":
        return 3.0
    if t == "text":
        return 5.0
    return 4.0


def visible_layers(panel: dict[str, Any], state: dict[str, Any], layers_off: set[str] | None = None,
                   view_um: float | None = None) -> list[dict]:  # fmt: skip
    """Layers of ``panel`` shown in ``state`` at the first view (``view_um``: its width in µm, for zoom-gated layers)."""
    off = layers_off or set()
    out = []
    for L in panel.get("layers", []):
        if not _matches(L.get("when"), state) or L["id"] in off or L.get("okey", L["id"]) in off or L.get("min_zoom"):
            continue
        if L.get("max_view_um") and (view_um is None or view_um > L["max_view_um"]):
            continue
        if L.get("needs") and not state.get(L["needs"]):
            continue
        if L.get("needs_any") and not state.get(L["needs_any"]):
            continue
        out.append(L)
    return out


def _ramp(stops: list[str]) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("spc", stops, N=256)


def _point_colors(
    L: dict[str, Any],
    arrays: dict[str, np.ndarray],
    state: dict[str, Any],
    hidden: dict,
    tables: dict[str, list[str]] | None = None,
) -> tuple:
    """RGBA per point and a drawing order (None: skip the point)."""
    from matplotlib.colors import LogNorm, Normalize, to_rgba

    n = int(L["n"])
    c = L.get("color", {"kind": "const", "color": "#888888"})
    kind = c.get("kind")
    order = None
    alpha = float(L.get("alpha", 1.0))
    if kind == "cat":
        codes = np.asarray(arrays[c["codes"]], dtype=np.int64)
        pal = [to_rgba(col, alpha) for col in c["colors"]] + [
            to_rgba(c.get("nan", "#000000"), alpha if "nan" in c else 0)
        ]
        hid = hidden.get(L.get("hkey", L["id"]), set())
        rgba = np.array(pal)[np.clip(codes, 0, len(pal) - 1)]
        show = ~np.isin(codes, list(hid)) & (rgba[:, 3] > 0)
    elif kind == "num":
        v = np.asarray(arrays[c["values"]], dtype=float)
        lo, hi = c["domain"]
        norm = LogNorm(lo, hi, clip=True) if c.get("scale") == "log" else Normalize(lo, hi, clip=True)
        cm = _ramp(c["ramp"])
        ok = np.isfinite(v) & ((v > 0) if c.get("scale") == "log" else True)
        rgba = np.zeros((n, 4))
        rgba[ok] = cm(norm(v[ok]))
        if "nan" in c:
            rgba[~ok] = to_rgba(c["nan"])
        rgba[:, 3] *= alpha
        show = ok | ("nan" in c)
        if c.get("order", True):
            order = np.argsort(np.where(ok, v, -np.inf), kind="stable")
    elif kind == "gene":
        cm = _ramp(c["ramp"])
        rgba = np.tile(np.asarray(cm(0.0)), (n, 1))
        show = np.ones(n, dtype=bool)
    elif kind == "highlight":
        codes = np.asarray(arrays[c["codes"]], dtype=np.int64)
        names = (tables or {}).get(c.get("table"), [])
        chosen = list(state.get(c["by"]) or [])
        slot = {nm: k for k, nm in enumerate(chosen)}
        rgba = np.zeros((n, 4))
        sel = np.full(n, -1)
        for k_code, nm in enumerate(names):
            if nm in slot:
                sel[codes == k_code] = slot[nm]
        hl = sel >= 0
        cols = [to_rgba(x) for x in c["colors"]]
        rgba[hl] = np.array(cols)[np.minimum(sel[hl], len(cols) - 1)]
        if c.get("other"):
            rgba[~hl] = to_rgba(c["other"], c.get("other_alpha", 1.0))
        show = hl | bool(c.get("other"))
        order = np.concatenate([np.flatnonzero(~hl), np.flatnonzero(hl)])
    else:
        rgba = np.tile(np.asarray(to_rgba(c.get("color", "#888888"), alpha)), (n, 1))
        show = np.ones(n, dtype=bool)
    if order is None and L.get("order") is not None:
        order = np.asarray(arrays[L["order"]], dtype=np.int64)
    if order is None:
        order = np.arange(n)
    order = order[show[order]]
    return rgba, order


def _axes_px(ax: Any) -> tuple[float, float]:
    bb = ax.get_window_extent()
    return float(bb.width), float(bb.height)


def render(spec: Any, d: dict[str, Any], *, state: dict[str, Any] | None = None, width_px: int = WIDTH_PX) -> bytes:
    """PNG of the figure's first view (or of ``state``).

    Parameters
    ----------
    spec
        The :class:`~spaclone.pl.interactive._spec.Spec` (its arrays).
    d
        Its dictionary (:meth:`Spec.to_dict`).
    state
        Control values (default: the first view's).
    width_px
        Picture width in pixels.
    """
    import matplotlib

    matplotlib.use(matplotlib.get_backend())  # never switch the user's backend
    from matplotlib.figure import Figure

    st = dict(initial_state(d))
    st.update(state or {})
    mode = "dark" if d.get("style") != "light" else "light"
    ch = CHROME[mode]
    arrays = spec.arrays
    hidden: dict[str, set] = {}
    for h in d.get("hidden", []):
        hidden.setdefault(h["layer"], set()).add(h["cat"])
    off = set(d.get("layers_off", []))
    lay = d.get("layout", {})
    cols, rows = int(lay.get("cols", 1)), int(lay.get("rows", 1))
    aspect = float(lay.get("aspect", 1.5))
    panels = [p for p in d.get("panels", []) if _matches(p.get("when"), st)]
    if lay.get("reflow"):  # as the figure: the panels shown share the row
        n_all, cols = cols, max(len(panels), 1)
        panels = [{**p, "area": [k, 0, 1, 1]} for k, p in enumerate(panels)]
        aspect *= cols / max(n_all, 1)
    legend = [s for s in d.get("legend", []) if _matches(s.get("when"), st)]
    side = lay.get("side") != "none" and bool(legend)
    W = width_px / _DPI
    side_w = 2.45 if side else 0.0
    pan_w = W - side_w - 0.3
    pan_h = float(np.clip(pan_w / aspect, 2.6, MAX_HEIGHT_PX / _DPI - 1.0))
    head = 0.55 if d.get("subtitle") else 0.42
    foot = 0.12 + 0.17 * len([n for n in d.get("notes", []) if _matches(n.get("when"), st)])
    H = pan_h + head + foot + 0.1
    fig = Figure(figsize=(W, H), dpi=_DPI, facecolor=ch["bg"])
    fig.text(0.14 / W, 1 - 0.1 / H, str(d.get("title", "")), color=ch["ink"], fontsize=11, fontweight="bold",
             va="top", ha="left")  # fmt: skip
    if d.get("subtitle"):
        fig.text(0.14 / W, 1 - 0.33 / H, str(d["subtitle"]), color=ch["muted"], fontsize=8.5, va="top", ha="left")  # fmt: skip
    gap = 0.1
    cw = (pan_w - (cols - 1) * gap) / cols
    rh = (pan_h - (rows - 1) * gap) / rows
    for p in panels:
        a = p.get("area", [0, 0, 1, 1])
        x0 = 0.14 + a[0] * (cw + gap)
        w = cw * a[2] + gap * (a[2] - 1)
        top = head + a[1] * (rh + gap)
        h = rh * a[3] + gap * (a[3] - 1)
        if p.get("kind") != "map":  # a chart: room for tick labels, axis titles and the panel title (as the figure)
            ml, mb, mt, mr = _chart_margins(p, w, h)
            x0, w, top, h = x0 + ml, max(w - ml - mr, 0.3), top + mt, max(h - mt - mb, 0.3)
        _panel(fig, p, (x0 / W, 1 - (top + h) / H, w / W, h / H), arrays, st, hidden, off, mode, d)
    if side:
        _legend(fig, legend, (W - side_w + 0.05) / W, 1 - head / H, side_w / W, arrays, st, mode, W, H)
    y = 0.05 / H
    for n in reversed([n for n in d.get("notes", []) if _matches(n.get("when"), st)]):
        fig.text(0.14 / W, y, n["text"], color=CHROME[mode]["muted"] if not n.get("warn") else "#d95926", fontsize=7.5,
                 va="bottom")  # fmt: skip
        y += 0.17 / H
    fig.text(1 - 0.14 / W, 0.05 / H, f"spaclone {d.get('version', '')} · static view of an interactive figure",
             color=ch["muted"], fontsize=7, ha="right", va="bottom")  # fmt: skip
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=ch["bg"], metadata={"Software": None})
    from PIL import Image

    from spaclone.pl.interactive._html import png_bytes

    img = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return png_bytes(img)


def _chart_margins(p: dict[str, Any], w: float, h: float) -> tuple[float, float, float, float]:
    """Left, bottom, top and right margins (inches) of a chart panel, as the figure's plot area leaves them."""
    ax = p.get("axes", {})
    m = p.get("margin") or {}
    x, y = ax.get("x", {}), ax.get("y", {})
    left = 0.12 if y.get("hide") else 0.58
    bottom = 0.1 if x.get("hide") else 0.42
    if y.get("scale") == "cat" and not y.get("hide"):
        longest = max((len(str(c)) for c in y.get("cats", [])), default=0)
        left = max(left, min(longest * 0.062 + 0.22 + (0.14 if y.get("label") else 0.0), w * 0.42))
    if x.get("scale") == "cat" and not x.get("hide") and x.get("rotate"):
        longest = max((len(str(c)) for c in x.get("cats", [])), default=0)
        bottom = max(bottom, min(longest * 0.047 + 0.3, h * 0.45))
    left = m["l"] / _DPI if "l" in m else left
    bottom = m["b"] / _DPI if "b" in m else bottom
    top = m["t"] / _DPI if "t" in m else (0.3 if p.get("title") or p.get("title_when") else 0.12)
    right = m["r"] / _DPI if "r" in m else 0.16
    return left, bottom, top, right


def _panel(fig: Any, p: dict[str, Any], rect: tuple, arrays: dict, st: dict, hidden: dict, off: set, mode: str,
           d: dict) -> None:  # fmt: skip
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.transforms import Affine2D

    ch = CHROME[mode]
    is_map = p.get("kind") == "map"
    ax = fig.add_axes(rect)
    ax.set_facecolor(ch["bg"])
    hb0 = p.get("home")
    if p.get("home_by"):
        hb0 = p.get("home_map", {}).get(str(st.get(p["home_by"])), hb0)
    layers = visible_layers(p, st, off, view_um=(hb0[1] - hb0[0]) if (is_map and hb0) else None)
    if is_map:
        layers = sorted(layers, key=_z)  # stable: the given order within a kind
    title = p.get("title", "")
    for tw in p.get("title_when", []):
        if _matches(tw.get("when"), st):
            title = tw["title"]
            break
    if is_map:
        hb = p.get("home")
        if p.get("home_by"):
            hb = p.get("home_map", {}).get(str(st.get(p["home_by"])), hb)
        hb = hb or [0, 1, 0, 1]
        pad = p.get("pad", 0.03)
        bw, bh = (hb[1] - hb[0]) or 1.0, (hb[3] - hb[2]) or 1.0
        # the padded home box, widened to the axes' own shape (1:1 aspect without matplotlib overriding the limits)
        cx, cy = (hb[0] + hb[1]) / 2, (hb[2] + hb[3]) / 2
        hw, hh = bw * (1 + 2 * pad) / 2, bh * (1 + 2 * pad) / 2
        apx_w, apx_h = _axes_px(ax)
        if apx_w > 0 and apx_h > 0:
            if hw / hh < apx_w / apx_h:
                hw = hh * apx_w / apx_h
            else:
                hh = hw * apx_h / apx_w
        ax.set_xlim(cx - hw, cx + hw)
        if p.get("invert_y", True):
            ax.set_ylim(cy + hh, cy - hh)
        else:
            ax.set_ylim(cy - hh, cy + hh)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        fig.canvas.draw() if hasattr(fig.canvas, "draw") else None
    else:
        _plot_axes(ax, p, mode, st)
    wpx, hpx = _axes_px(ax)
    x0, x1 = ax.get_xlim()
    k = wpx / max(abs(x1 - x0), 1e-12)  # px per data unit along x
    z = 1
    clips: dict[str, Any] = {}
    for L in layers:
        t = L["type"]
        z += 1
        if t == "points":
            rgba, order = _point_colors(L, arrays, st, hidden, d.get("tables"))
            if not len(order):
                continue
            x = arrays[L["x"]][order]
            y = arrays[L["y"]][order]
            if not is_map:
                x, y = _plot_xy(p, x, y)
            s = L.get("size", {"px": 3})
            dpx = s["px"] if "px" in s else s.get("um", 10) * k
            dpx = float(np.clip(dpx, s.get("min", 0.5), s.get("max", 1e9)))
            size_pt2 = (max(dpx, 0.6) * 72 / _DPI) ** 2
            edge = L.get("edge")
            ax.scatter(x, y, s=size_pt2, c=rgba[order], marker="s" if L.get("shape") == "square" else "o",
                       linewidths=(edge["px"] * 72 / _DPI) if edge else 0,
                       edgecolors=edge["color"] if edge else "none", zorder=z, rasterized=True)  # fmt: skip
        elif t == "image":
            img = arrays.get(L["id"] + ":rgba")
            if img is None:
                continue
            m = L["m"]
            tr = Affine2D(np.array([[m[0], m[2], m[4]], [m[1], m[3], m[5]], [0, 0, 1]])) + ax.transData
            if not is_map:
                continue
            im = ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0), origin="upper", interpolation="bilinear",
                           transform=tr, zorder=z, alpha=L.get("alpha", 1.0))  # fmt: skip
            cl = L.get("clip")
            if cl and cl in clips:
                im.set_clip_path(clips[cl])
            elif cl:
                src = next((q for q in p["layers"] if q["id"] == cl), None)
                if src is not None:
                    path = _compound(arrays[src["xy"]], arrays[src["off"]], closed=True)
                    patch = PathPatch(path, transform=ax.transData, facecolor="none", edgecolor="none")
                    clips[cl] = patch
                    im.set_clip_path(patch)
        elif t == "paths":
            xy = np.asarray(arrays[L["xy"]], dtype=float).reshape(-1, 2)
            offs = np.asarray(arrays[L["off"]], dtype=np.int64)
            if not is_map:
                xy = np.column_stack(_plot_xy(p, xy[:, 0], xy[:, 1]))
            if L.get("fill"):
                ax.add_patch(PathPatch(_compound(xy.ravel(), offs, closed=True), facecolor=L["fill"],
                                       edgecolor="none", zorder=z, alpha=L.get("fill_alpha", 1.0)))  # fmt: skip
            if L.get("stroke") or L.get("colors"):
                segs = [xy[offs[r] : offs[r + 1]] for r in range(len(offs) - 1)]
                if L.get("closed"):
                    segs = [np.vstack([s_, s_[:1]]) for s_ in segs if len(s_)]
                lw = L.get("width", 1.0) * 72 / _DPI
                cols = L.get("colors") or [L["stroke"]]
                colors = [cols[r % len(cols)] for r in range(len(segs))]
                ls = (0, tuple(v * 72 / _DPI / lw for v in L["dash"])) if L.get("dash") else "solid"
                lc = LineCollection(segs, colors=colors, linewidths=lw, linestyles=ls, zorder=z, capstyle="round",
                                    joinstyle="round", alpha=L.get("alpha", 1.0))  # fmt: skip
                if L.get("halo"):
                    lc.set_path_effects([patheffects.withStroke(linewidth=lw + 1.6, foreground=L["halo"]),
                                         patheffects.Normal()])  # fmt: skip
                ax.add_collection(lc)
        elif t == "segs":
            s4 = np.asarray(arrays[L["xy"]], dtype=float).reshape(-1, 4)
            if not is_map:
                a0 = _plot_xy(p, s4[:, 0], s4[:, 1])
                a1 = _plot_xy(p, s4[:, 2], s4[:, 3])
                s4 = np.column_stack([a0[0], a0[1], a1[0], a1[1]])
            col = L.get("color", "#888888")
            if isinstance(col, dict):
                codes = np.asarray(arrays[col["codes"]], dtype=np.int64)
                colors = [col["colors"][c_] for c_ in codes]
            else:
                colors = col
            ax.add_collection(LineCollection(s4.reshape(-1, 2, 2), colors=colors, linewidths=L.get("width", 1.0) * 0.72,
                                             alpha=L.get("alpha", 1.0), zorder=z))  # fmt: skip
        elif t == "rects":
            r = np.asarray(arrays[L["r"]], dtype=float).reshape(-1, 4)
            fill = L.get("fill")
            labels = d.get("tables", {}).get(L.get("labels"), None) if L.get("labels") else None
            for i, (a0, b0, a1, b1) in enumerate(r):
                if not is_map:
                    (a0, a1), (b0, b1) = _plot_xy(p, np.array([a0, a1]), np.array([b0, b1]))
                col = None
                if isinstance(fill, str):
                    col = fill
                elif isinstance(fill, dict) and "codes" in fill:
                    code = int(arrays[fill["codes"]][i])
                    if code in hidden.get(L.get("hkey", L["id"]), set()):
                        continue
                    col = fill["colors"][code]
                elif isinstance(fill, dict) and "values" in fill:
                    v = float(arrays[fill["values"]][i])
                    lo, hi = fill["domain"]
                    col = _ramp(fill["ramp"])((v - lo) / ((hi - lo) or 1)) if np.isfinite(v) else L.get("nan")
                ax.add_patch(Rectangle((min(a0, a1), min(b0, b1)), abs(a1 - a0), abs(b1 - b0),
                                       facecolor=col if col is not None else "none",
                                       edgecolor=L.get("stroke", "none"), linewidth=L.get("width", 1) * 0.72, zorder=z))  # fmt: skip
                if labels and labels[i]:
                    from spaclone.pl._colors import ink_on

                    ax.text((a0 + a1) / 2, (b0 + b1) / 2, labels[i], ha="center", va="center", fontsize=7.5,
                            color=ink_on(col) if col is not None else ch["ink"], zorder=z + 0.5, clip_on=True)  # fmt: skip
        elif t == "text":
            xy = np.asarray(arrays[L["xy"]], dtype=float).reshape(-1, 2)
            labels = d["tables"][L["text"]]
            for (tx, ty), lab in zip(xy, labels, strict=False):
                if not is_map:
                    tx, ty = (float(v[0]) for v in _plot_xy(p, np.array([tx]), np.array([ty])))
                ha = {"start": "left", "end": "right"}.get(L.get("anchor", "middle"), "center")
                va = {"top": "top", "bottom": "bottom", "alphabetic": "baseline"}.get(
                    L.get("baseline", "middle"), "center"
                )
                txt = ax.text(tx, ty, lab, color=L.get("color", ch["ink"]), fontsize=L.get("size", 12) * 0.72,
                              fontweight="bold" if str(L.get("weight", "600")) >= "600" else "normal", ha=ha, va=va,
                              zorder=z, clip_on=True)  # fmt: skip
                if L.get("halo", "") is not None:
                    txt.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground=L.get("halo") or ch["bg"])])
    if is_map and p.get("scalebar", True):
        _scalebar(ax, mode)
    sub = next((t["text"] for t in p.get("subtitle_when", []) if _matches(t.get("when"), st)), "")
    if title and not is_map:  # a chart: the title above its axes, as the figure draws it
        ax.set_title(title + (f"\n{sub}" if sub else ""), loc="left", color=ch["ink"], fontsize=9, fontweight="bold",
                     pad=5)  # fmt: skip
    elif title or sub:  # a map: title and second line on plates that the data do not cross, wrapped, never cut
        import textwrap

        per = max(int((_axes_px(ax)[0] - 14) / 6.4), 12)  # characters per line at 8.5 pt
        plate = {"boxstyle": "round,pad=0.3", "fc": ch["bg"], "ec": "none", "alpha": 0.74}
        lines_t = textwrap.wrap(title, per) if title else []
        lines_s = textwrap.wrap(sub, int(per * 1.12)) if sub else []
        if lines_t:
            ax.annotate("\n".join(lines_t), (0.012, 0.985), xycoords="axes fraction", va="top", ha="left", fontsize=8.5,
                        fontweight="bold", color=ch["ink"], zorder=70, bbox=plate, linespacing=1.25)  # fmt: skip
        if lines_s:
            ax.annotate("\n".join(lines_s), (0.012, 0.985), xycoords="axes fraction",
                        xytext=(0, -(len(lines_t) * 8.5 * 1.25 + 5) if lines_t else 0), textcoords="offset points",
                        va="top", ha="left", fontsize=7.5, color=ch["ink2"], zorder=70, bbox=plate, linespacing=1.25)  # fmt: skip


def _compound(xy: np.ndarray, offs: np.ndarray, *, closed: bool) -> Any:
    from matplotlib.path import Path

    v = np.asarray(xy, dtype=float).reshape(-1, 2)
    verts, codes = [], []
    for r in range(len(offs) - 1):
        ring = v[offs[r] : offs[r + 1]]
        if len(ring) < 2:
            continue
        verts.append(ring)
        c = np.full(len(ring), Path.LINETO, dtype=np.uint8)
        c[0] = Path.MOVETO
        codes.append(c)
        if closed:
            verts.append(ring[:1])
            codes.append(np.array([Path.CLOSEPOLY], dtype=np.uint8))
    if not verts:
        return Path(np.zeros((1, 2)))
    return Path(np.vstack(verts), np.concatenate(codes))


def _plot_xy(p: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plot-panel coordinates as stored (log axes store log10 values) -> values for matplotlib's scales."""
    ax = p.get("axes", {})
    if ax.get("x", {}).get("scale") == "log":
        x = np.power(10.0, np.asarray(x, dtype=float))
    if ax.get("y", {}).get("scale") == "log":
        y = np.power(10.0, np.asarray(y, dtype=float))
    return x, y


def _plot_axes(ax: Any, p: dict[str, Any], mode: str, st: dict[str, Any] | None = None) -> None:
    ch = CHROME[mode]
    a = dict(p.get("axes", {}))
    for name in ("x", "y"):  # a range that follows a switcher (e.g. the metric): the current one
        for key, ranges in (a.get(f"{name}_by") or {}).items():
            r = ranges.get(str((st or {}).get(key)))
            if r:
                a[name] = {**a.get(name, {}), "range": list(r)}
    for name in ("x", "y"):
        spec = a.get(name, {})
        setter = ax.set_xlim if name == "x" else ax.set_ylim
        scale = spec.get("scale", "lin")
        if scale == "log":
            (ax.set_xscale if name == "x" else ax.set_yscale)("log")
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

            axis = ax.xaxis if name == "x" else ax.yaxis  # plain numbers at 1, 2, 5 x 10^k, as the figure labels them
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        if spec.get("range"):
            lo, hi = spec["range"]
            setter(hi, lo) if (name == "y" and spec.get("flip")) else setter(lo, hi)
        elif scale == "cat":
            n = len(spec.get("cats", []))
            setter(n - 0.5, -0.5) if (name == "y" and spec.get("flip")) else setter(-0.5, n - 0.5)
        if scale == "cat":
            cats = spec.get("cats", [])
            (ax.set_xticks if name == "x" else ax.set_yticks)(range(len(cats)))
            (ax.set_xticklabels if name == "x" else ax.set_yticklabels)(
                cats, rotation=45 if (name == "x" and spec.get("rotate")) else 0,
                ha="right" if (name == "x" and spec.get("rotate")) else "center" if name == "x" else "right")  # fmt: skip
        if spec.get("label"):
            (ax.set_xlabel if name == "x" else ax.set_ylabel)(spec["label"], color=ch["ink"], fontsize=8.5)
        if spec.get("hide"):
            (ax.xaxis if name == "x" else ax.yaxis).set_visible(False)
            ax.spines["bottom" if name == "x" else "left"].set_visible(False)
    if a.get("square"):  # a matrix: square cells, as in the figure
        ax.set_aspect("equal", adjustable="box", anchor="NW")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ch["axis"])
    ax.tick_params(colors=ch["ink2"], labelsize=7.5)
    for r in p.get("refs", []):
        if "x" in r:
            ax.axvline(r["x"], color=r.get("color", ch["ink2"]), lw=r.get("width", 1) * 0.72,
                       ls=(0, tuple(r["dash"])) if r.get("dash") else "-")  # fmt: skip
        else:
            ax.axhline(r["y"], color=r.get("color", ch["ink2"]), lw=r.get("width", 1) * 0.72,
                       ls=(0, tuple(r["dash"])) if r.get("dash") else "-")  # fmt: skip


def _scalebar(ax: Any, mode: str) -> None:
    from matplotlib import patheffects

    ch = CHROME[mode]
    x0, x1 = sorted(ax.get_xlim())
    y_lo, y_hi = ax.get_ylim()
    target = (x1 - x0) / 5
    if target <= 0:
        return
    p10 = 10 ** np.floor(np.log10(target))
    best = p10
    for m in (1, 2, 2.5, 5, 10):
        if m * p10 <= target * 1.15:
            best = m * p10
    xb = x0 + 0.02 * (x1 - x0)
    yb = y_lo + 0.035 * (y_hi - y_lo)
    from matplotlib.patches import Rectangle

    hpx = abs(y_hi - y_lo) / max(_axes_px(ax)[1], 1.0)  # data units per pixel (vertical)
    pad = 5 * hpx
    ax.add_patch(Rectangle((xb - 0.012 * (x1 - x0), min(yb - pad, yb + 22 * hpx * np.sign(y_hi - y_lo))),
                           best + 0.024 * (x1 - x0), 27 * hpx, facecolor=ch["bg"], edgecolor="none", alpha=0.78,
                           zorder=59))  # fmt: skip
    ax.plot([xb, xb + best], [yb, yb], color=ch["ink"], lw=1.6, solid_capstyle="butt", zorder=60)
    label = f"{best / 1000:g} mm" if best >= 1000 else f"{best:g} µm"
    ax.annotate(label, (xb + best / 2, yb), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                color=ch["ink"], fontsize=7.5, zorder=60,
                path_effects=[patheffects.withStroke(linewidth=2, foreground=ch["bg"])])  # fmt: skip


def _legend(fig: Any, legend: list[dict], x: float, top: float, w: float, arrays: dict, st: dict, mode: str, W: float,
            H: float) -> None:  # fmt: skip
    from matplotlib.patches import Rectangle

    ch = CHROME[mode]
    y = top - 0.05 / H
    line = 0.18 / H
    for sec in legend:
        if sec.get("kind") in ("colorbar", "genebar"):
            if sec.get("kind") == "genebar":
                continue
            fig.text(x, y, sec.get("label", ""), color=ch["ink"], fontsize=8, va="top", ha="left")
            y -= line
            ax = fig.add_axes((x, y - 0.1 / H, w * 0.8, 0.09 / H))
            ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=_ramp(sec["ramp"]))
            ax.set_axis_off()
            lo, hi = sec["domain"]
            fmt = sec.get("fmt")
            f = (lambda v: f"{100 * v:.0f}%") if fmt and fmt.startswith("pct") else (lambda v: f"{v:.3g}")
            fig.text(x, y - 0.13 / H, f(lo), color=ch["muted"], fontsize=7, va="top", ha="left")
            fig.text(x + w * 0.8, y - 0.13 / H, f(hi), color=ch["muted"], fontsize=7, va="top", ha="right")
            y -= 0.42 / H
            continue
        if sec.get("title"):
            fig.text(x, y, str(sec["title"]).upper(), color=ch["muted"], fontsize=7, fontweight="bold", va="top")  # fmt: skip
            y -= line
        for it in sec.get("items", [])[:24]:
            shape = it.get("shape", "point")
            ax_y = y - 0.08 / H
            if shape in ("line", "dash", "box"):
                fig.add_artist(Rectangle((x, ax_y), 0.16 / W, 0.012 / H if shape != "box" else 0.09 / H,
                                         transform=fig.transFigure, facecolor="none" if shape == "box" else it["color"],
                                         edgecolor=it["color"]))  # fmt: skip
            else:
                fig.text(x, y, "●" if shape == "point" else "■", color=it["color"], fontsize=8.5, va="top")
            label = str(it.get("label", ""))
            room = int((w * W - 0.2 - (0.45 if it.get("n") is not None else 0.0)) / 0.062)  # characters that fit
            if len(label) > room > 8:
                label = label[: room - 1].rstrip() + "…"
            fig.text(x + 0.2 / W, y, label, color=ch["ink"], fontsize=8, va="top")
            if it.get("n") is not None:
                fig.text(x + w - 0.1 / W, y, f"{int(it['n']):,}", color=ch["muted"], fontsize=7.5, va="top", ha="right")
            y -= line
            if y < 0.05:
                return
        y -= 0.08 / H
