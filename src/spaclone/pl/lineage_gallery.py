"""Lineage trees next to the tissue: one figure per tree, and a gallery writer for a whole lineage workflow.

:func:`lineage_tree` draws one tree of a :class:`spaclone.bcr.LineageWorkflowResult` in the layout of the 2026-10
StereoIR lineage figures (archive L10): the germline-rooted tree on the left (x = mutations from the germline, one row
per tip, tips filled by isotype), a column of squares with each tip's location class, the section on the top right
(long axis horizontal, CSR switched-share map or plain tissue, outlines of the B-rich area, T zones and CSR hotspots)
with the tips at their positions, and below it one row per tip with a dot at the tip's position along the section,
joined to its spot in the map by a thin line. Legends and the clone information are inside the figure; related clone
groups list their member exact clones (CDR3, tips each).

:func:`lineage_gallery` writes the figures of every tree the workflow selected (``lineage_exact_01.png``, ...,
``lineage_related_01.png``, ...) and an index table ``lineage_gallery.csv``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl._colors import ISOTYPE_COLORS, LINEAGE_STYLES, LOCATION_COLORS

__all__ = ["lineage_gallery", "lineage_tree"]

#: Isotype colours of the tips and the figure theme (location colours: :data:`spaclone.pl._colors.LOCATION_COLORS`).
_ISO = {
    mode: {k: ISOTYPE_COLORS[mode][k] for k in ("IgM", "IgD", "IgG", "IgA", "unknown")} for mode in ("dark", "light")
}
_ISO_LABEL = {"IgM": "IgM", "IgD": "IgD", "IgG": "IgG", "IgA": "IgA", "unknown": "unknown"}
_LOC_ORDER = ("CSR hotspot", "follicle", "T zone", "red pulp")
_THEME = LINEAGE_STYLES
SLIDE = (13.333, 7.5)


def _theme(style: str) -> dict[str, Any]:
    if style not in ("dark", "light"):
        raise ValueError(f"style must be 'dark' or 'light', got {style!r}")
    return _THEME[style]


# ------------------------------------------------------------------------------------------------------ tree


class _Tree:
    """A tree hung from its germline tip, with the workflow tips attached to the tree tips that hold them."""

    def __init__(self, tree: Any, tips: pd.DataFrame) -> None:
        n = len(tree.parent)
        nb: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for v, p in enumerate(tree.parent):
            if p >= 0:
                nb[v].append((int(p), float(tree.length[v])))
                nb[int(p)].append((v, float(tree.length[v])))
        labels = list(tree.label)
        start = labels.index("Germline") if "Germline" in labels else tree.root
        self.germ = start
        self.div = {start: 0.0}
        self.kids: dict[int, list[int]] = {}
        stack = [start]
        while stack:
            u = stack.pop()
            for v, w in nb[u]:
                if v not in self.div:
                    self.div[v] = self.div[u] + w
                    self.kids.setdefault(u, []).append(v)
                    stack.append(v)
        node_of = {lab: k for k, lab in enumerate(labels) if lab is not None and lab != "Germline"}
        lv = tips.copy()
        lv["node"] = lv["tree_label"].astype(str).map(node_of)
        lv = lv[lv["node"].notna()].copy()
        lv["node"] = lv["node"].astype(int)
        lv["div"] = lv["node"].map(self.div).astype(float)
        self.leaves = lv.reset_index(drop=True)
        at: dict[int, list[int]] = {}
        for k, u in enumerate(self.leaves["node"]):
            at.setdefault(int(u), []).append(k)
        self.at = at
        size: dict[int, int] = {}
        for u in self._postorder():
            size[u] = len(at.get(u, [])) + sum(size[v] for v in self.kids.get(u, []))
        self.size = size

    def _postorder(self) -> list[int]:
        pre = [self.germ]
        i = 0
        while i < len(pre):
            pre.extend(self.kids.get(pre[i], []))
            i += 1
        return pre[::-1]

    def order(self, x: np.ndarray) -> list[int]:
        """Leaf order of the archive (L_common.LTree): leaves on a node by position, then children, larger first.

        At every node the leaves attached to it come first (by position along the section), then children on
        zero-length branches, then the other children, larger subtrees first.
        """
        out: list[int] = []
        stack: list[int] = [self.germ]
        while stack:
            u = stack.pop()
            here = sorted(self.at.get(u, []), key=lambda k: (x[k], k))
            out.extend(here)
            kids = sorted(
                self.kids.get(u, []),
                key=lambda v: (self.div[v] - self.div[u] > 0.5, -self.size[v], v),
            )
            stack.extend(reversed(kids))
        return out

    def layout(self, order: list[int]) -> tuple[np.ndarray, dict[int, float]]:
        row = np.empty(len(order), dtype=float)
        for r, k in enumerate(order):
            row[k] = r
        ys: dict[int, float] = {}
        for u in self._postorder():
            vals = [row[k] for k in self.at.get(u, [])] + [ys[v] for v in self.kids.get(u, []) if np.isfinite(ys[v])]
            ys[u] = float((min(vals) + max(vals)) / 2) if vals else np.nan
        return row, ys

    def segments(self, row: np.ndarray, ys: dict[int, float]) -> list[list[tuple[float, float]]]:
        segs = []
        for u, kids in self.kids.items():
            yk = [ys[v] for v in kids if np.isfinite(ys.get(v, np.nan))]
            if not yk or not np.isfinite(ys.get(u, np.nan)):
                continue
            segs.append([(self.div[u], min([*yk, ys[u]])), (self.div[u], max([*yk, ys[u]]))])
            for v in kids:
                if np.isfinite(ys.get(v, np.nan)):
                    segs.append([(self.div[u], ys[v]), (self.div[v], ys[v])])
        for k, u in enumerate(self.leaves["node"]):
            if abs(ys.get(int(u), row[k]) - row[k]) > 1e-9:
                segs.append([(self.div[int(u)], ys[int(u)]), (self.div[int(u)], row[k])])
        return segs


# ------------------------------------------------------------------------------------------------------ section


class _Section:
    """Rotation that puts the section's long axis horizontal, the frame, and rotated pixel centres of map layers."""

    def __init__(self, maps: Mapping[str, Any], fallback_xy: np.ndarray) -> None:
        rot = maps.get("rotation") if maps else None
        if rot is not None:
            self.c0, self.angle = np.asarray(rot["centre"], dtype=float), float(rot["angle"])
        else:
            pts = np.asarray(fallback_xy, dtype=float)
            self.c0 = pts.mean(0)
            v = np.linalg.eigh(np.cov((pts - self.c0).T))[1][:, -1] if len(pts) > 2 else np.array([1.0, 0.0])
            self.angle = float(-np.arctan2(v[1], v[0]))
        a = self.angle
        self.R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        if maps and maps.get("frame") is not None:
            self.box = tuple(float(v) for v in maps["frame"])
        else:
            q = np.quantile(self.rot(fallback_xy), [0.001, 0.999], axis=0)
            pad = 0.05 * max(q[1, 0] - q[0, 0], q[1, 1] - q[0, 1], 100.0)
            self.box = (q[0, 0] - pad, q[1, 0] + pad, q[0, 1] - pad, q[1, 1] + pad)
        self._xy: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        self._tri: dict[Any, Any] = {}

    def rot(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, dtype=float) - self.c0) @ self.R.T

    def frame(self, ax: Any) -> None:
        b = self.box
        ax.set_xlim(b[0], b[1])
        ax.set_ylim(b[3], b[2])
        ax.set_aspect("equal")
        ax.set_axis_off()

    def coords(self, grid: Any) -> tuple[np.ndarray, np.ndarray]:
        """Rotated pixel centres of a grid, shape ``grid.shape`` each."""
        key = (grid.origin, grid.pixel_um, grid.shape)
        if key not in self._xy:
            gx, gy = grid.centres()
            gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
            r = self.rot(np.c_[gxx.ravel(), gyy.ravel()])
            self._xy[key] = (r[:, 0].reshape(gxx.shape), r[:, 1].reshape(gxx.shape))
        return self._xy[key]

    def triangulation(self, grid: Any) -> Any:
        """Two triangles per grid cell (a regular-grid triangulation of the rotated pixel centres)."""
        import matplotlib.tri as mtri

        key = (grid.origin, grid.pixel_um, grid.shape)
        if key not in self._tri:
            rx, ry = self.coords(grid)
            nx, ny = rx.shape
            idx = np.arange(nx * ny).reshape(nx, ny)
            a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
            c, d = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
            tri = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])
            self._tri[key] = mtri.Triangulation(rx.ravel(), ry.ravel(), triangles=tri)
        return self._tri[key]

    def contour(self, ax: Any, raster: Any, color: str, lw: float, smooth: float = 1.5, zorder: float = 2,
                values: np.ndarray | None = None) -> None:  # fmt: skip
        from scipy import ndimage as ndi

        if raster is None:
            return
        m = np.asarray(raster.values if values is None else values, dtype=float) > 0
        if not m.any() or m.all():
            return
        rx, ry = self.coords(raster.grid)
        ax.contour(rx, ry, ndi.gaussian_filter(m.astype(float), smooth), levels=[0.5], colors=color, linewidths=lw,
                   zorder=zorder)  # fmt: skip

    def fill(self, ax: Any, raster: Any, color: str, smooth: float = 0.8, zorder: float = 0) -> None:
        from scipy import ndimage as ndi

        if raster is None:
            return
        m = np.asarray(raster.values, dtype=float) > 0
        if not m.any():
            return
        rx, ry = self.coords(raster.grid)
        ax.contourf(rx, ry, ndi.gaussian_filter(m.astype(float), smooth), levels=[0.5, 10], colors=[color],
                    zorder=zorder)  # fmt: skip

    def shade(self, ax: Any, share: Any, eligible: np.ndarray, cmap: Any, norm: Any, zorder: float = 0.5) -> None:
        """Gouraud-shaded map on eligible pixels (triangles with an ineligible corner are not drawn)."""
        v = np.asarray(share.values, dtype=float).ravel()
        e = np.asarray(eligible, dtype=bool).ravel() & np.isfinite(v)
        import matplotlib.tri as mtri

        full = self.triangulation(share.grid)
        # a Triangulation of its own: the collection reads the mask when it is drawn, so it must keep this one
        tri = mtri.Triangulation(full.x, full.y, triangles=full.triangles, mask=~e[full.triangles].all(axis=1))
        ax.tripcolor(tri, np.where(e, v, 0.0), shading="gouraud", cmap=cmap, norm=norm, rasterized=True,
                     zorder=zorder)  # fmt: skip


# ------------------------------------------------------------------------------------------------------ figure


def _member_lines(members: pd.DataFrame, max_letters: int = 30, n_lines: int = 2) -> list[str]:
    """'CDR3 (tips)' of a group's exact clones, largest first, packed into at most ``n_lines`` lines (archive L10)."""
    items = [(str(a), int(n)) for a, n in zip(members[S.JUNCTION_AA], members["n_tips"], strict=True)]
    lines: list[list[tuple[str, int]]] = [[]]
    k = len(items)
    for i, (cd, n) in enumerate(items):
        if lines[-1] and sum(len(c) for c, _ in lines[-1]) + len(cd) > max_letters:
            if len(lines) == n_lines:
                k = i
                break
            lines.append([])
        lines[-1].append((cd, n))
    rest = items[k:]
    if rest and len(lines[-1]) > 1 and sum(len(c) for c, _ in lines[-1]) > 18:
        rest = [lines[-1].pop(), *rest]
    txt = [" · ".join(f"{c} ({n})" for c, n in ln) for ln in lines if ln]
    if rest:
        txt[-1] += f" · +{len(rest)} others ({sum(n for _, n in rest)})"
    return txt


def _split_clone(tree_id: str) -> tuple[str, str, str]:
    parts = str(tree_id).split("|")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2].split("#")[0]
    return str(tree_id), "", ""


def _extent(fig: Any, artist: Any) -> Any:
    """Bounding box of an artist in figure fractions."""
    r = fig.canvas.get_renderer()
    return artist.get_window_extent(r).transformed(fig.transFigure.inverted())


def _occupied(fig: Any, axm: Any, sec: _Section, maps: Mapping[str, Any], fallback_xy: np.ndarray) -> np.ndarray:
    """Figure-fraction positions of the map's content (tissue pixels, else the records), inside the map box."""
    lay = next((maps.get(k) for k in ("tissue", "share_eligible", "b_rich") if maps.get(k) is not None), None)
    if lay is not None:
        rx, ry = sec.coords(lay.grid)
        m = np.asarray(lay.values, dtype=bool)
        pts = np.c_[rx[m], ry[m]]
    else:
        pts = sec.rot(fallback_xy)
    f = fig.transFigure.inverted().transform(axm.transData.transform(pts))
    pm = axm.get_position()
    keep = (f[:, 0] >= pm.x0) & (f[:, 0] <= pm.x1) & (f[:, 1] >= pm.y0) & (f[:, 1] <= pm.y1)
    return f[keep]


def _room(occ: np.ndarray, y0: float, y1: float, limit: float = 0.995) -> float:
    """Right edge available to text spanning figure heights [y0, y1]: left of the map content at those heights."""
    sel = occ[(occ[:, 1] >= y0 - 0.006) & (occ[:, 1] <= y1 + 0.006), 0] if len(occ) else occ
    return min(limit, float(sel.min()) - 0.008) if len(sel) else limit


def _fit(fig: Any, artist: Any, occ: np.ndarray, min_size: float) -> bool:
    """Shrink a text or legend until it ends left of the map content at its height; True when it fits."""
    while True:
        e = _extent(fig, artist)
        if e.x1 <= _room(occ, e.y0, e.y1):
            return True
        texts = [artist] if hasattr(artist, "get_fontsize") else [*artist.get_texts(), artist.get_title()]
        size = texts[0].get_fontsize()
        if size <= min_size:
            return False
        for t_ in texts:
            t_.set_fontsize(t_.get_fontsize() - 0.5)


def lineage_tree(
    result: Any,
    tree_id: str,
    *,
    grouping: str = "exact",
    style: str = "dark",
    background: str = "auto",
    figsize: tuple[float, float] = SLIDE,
    title: str | None = None,
    s_scale: float = 1700.0,
) -> Any:
    """One lineage tree next to the tissue (the archive's annotated lineage figure, all text in English).

    Parameters
    ----------
    result
        A :class:`spaclone.bcr.LineageWorkflowResult`.
    tree_id
        Clone (exact) or group (related) id, as in ``result.selection.tree_id``.
    grouping
        ``"exact"`` or ``"related"``.
    style
        ``"dark"`` (black house style) or ``"light"``.
    background
        Section map: ``"share"`` (the CSR switched-share map of the hotspot result), ``"tissue"`` (plain tissue) or
        ``"auto"`` (share map when the result has one).
    figsize
        Figure size in inches (16:9 slide by default).
    title
        Header text; default ``"V gene  J gene  CDR3"`` (exact) or ``"V gene  J gene  ·  k related exact clones"``.
    s_scale
        Marker area is ``s_scale / tips`` (clipped to 8-48 pt²).

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Left: the germline-rooted tree (x = mutations from the germline along the tree, white diamond = germline, one
    row per tip, tips filled by isotype); squares = the tip's location class; top right: the section with its long
    axis horizontal and the tips at their spots (higher mutation counts on top); bottom right: one row per tip (aligned
    with the tree) with a dot at the tip's position along the section, joined to its spot by a thin line. Header and
    legends are placed by their measured size, so long titles and large counts do not overlap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from spaclone.bcr.lineage_workflow import _grouping

    th = _theme(style)
    if background not in ("auto", "share", "tissue"):
        raise ValueError(f"background must be 'auto', 'share' or 'tissue', got {background!r}")
    g = _grouping(grouping)
    if tree_id not in result.trees[g]:
        raise KeyError(f"lineage_tree: no {g} tree {tree_id!r}")
    tips = result.tree_tips(tree_id, grouping=g)
    if not len(tips):
        raise ValueError(f"lineage_tree: tree {tree_id!r} holds no tips")
    tr = _Tree(result.trees[g][tree_id], tips)
    lv = tr.leaves
    n = len(lv)
    maps = result.maps or {}
    rec_xy = result.records[[S.X, S.Y]].to_numpy(dtype=float) if len(result.records) else lv[[S.X, S.Y]].to_numpy()
    sec = _Section(maps, rec_xy)
    iso_col = _ISO[style]
    loc_col = LOCATION_COLORS[style]
    iso = lv["isotype"].astype(str).where(lv["isotype"].astype(str).isin(list(iso_col)), "unknown")
    cols = np.array([iso_col[i] for i in iso])
    xy = sec.rot(lv[[S.X, S.Y]].to_numpy(dtype=float))
    order = tr.order(xy[:, 0])
    row, ys = tr.layout(order)
    s_pt = float(np.clip(s_scale / n, 8, 48))
    members = result.members[result.members.tree_id == tree_id] if g == "related" else None
    if members is not None and len(members) < 2:
        members = None
    share, elig = maps.get("share"), maps.get("share_eligible")
    use_share = background == "share" or (background == "auto" and share is not None)
    if use_share and share is None:
        raise ValueError("lineage_tree: the result has no switched-share map (pass hotspots= to lineage_workflow)")
    fig = plt.figure(figsize=figsize, facecolor=th["bg"])
    fig.canvas.get_renderer()
    # ---- map (top right)
    axm = fig.add_axes((0.248, 0.615, 0.75, 0.38))
    axm.set_facecolor(th["bg"])
    cmap = LinearSegmentedColormap.from_list("csr", th["ramp"])
    norm = None
    tissue = maps.get("tissue")
    if use_share:
        e = np.asarray(elig.values, dtype=bool) if elig is not None else np.ones(share.grid.shape, dtype=bool)
        e = e & np.isfinite(np.asarray(share.values, dtype=float))
        if tissue is not None:  # shade the tissue only (the share map can be eligible a kernel width beyond it)
            if tissue.grid == share.grid:
                e &= np.asarray(tissue.values, dtype=bool)
            else:
                gx, gy = share.grid.centres()
                gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
                inside = tissue.sample(gxx.ravel(), gyy.ravel(), fill=0).reshape(gxx.shape)
                e &= np.asarray(inside, dtype=bool)
        vals = np.asarray(share.values, dtype=float)[e]
        lo, hi = np.quantile(vals, [0.02, 0.995]) if len(vals) else (0.0, 1.0)
        norm = Normalize(float(lo), float(max(hi, lo + 1e-9)))
        sec.shade(axm, share, e, cmap, norm)
    elif tissue is not None:
        sec.fill(axm, tissue, th["tissue"])
    b_rich, t_rich, hot = maps.get("b_rich"), maps.get("t_rich"), maps.get("hotspots")
    if b_rich is not None:
        b_or_t = np.asarray(b_rich.values, dtype=bool)
        if t_rich is not None and t_rich.grid == b_rich.grid:
            b_or_t = b_or_t | np.asarray(t_rich.values, dtype=bool)
        sec.contour(axm, b_rich, th["fg"], 0.55, values=b_or_t)
    sec.contour(axm, t_rich, th["t_edge"], 0.8)
    if hot is not None:
        sec.contour(axm, hot, th["hot_edge"], 1.5, smooth=0.7, zorder=5)
    sec.frame(axm)
    axm.apply_aspect()
    pm = axm.get_position()
    occ = _occupied(fig, axm, sec, maps, rec_xy)
    # ---- header (top left): measured, shrunk or wrapped where it would run into the map content
    v, j, cdr3 = _split_clone(tree_id)
    if title is None:
        title = f"{v}  {j}  {cdr3}" if members is None else f"{v}  {j}  ·  {len(members)} related exact clones"
    t = fig.text(0.008, 0.985, title, color=th["fg"], fontsize=13, fontweight="bold", va="top")
    _fit(fig, t, occ, 9.5)
    y = _extent(fig, t).y0 - 0.008
    n_loci = lv["locus_id"].nunique()
    mut_txt = f"V mutations per tip: up to {int(lv['mutations'].max())} (median {lv['mutations'].median():g})"
    lines = [f"{n} tips · {n_loci} loci · {mut_txt}"]
    probe = fig.text(0.008, y, lines[0], fontsize=10.5, va="top")
    if not _fit(fig, probe, occ, 10.5):
        lines = [f"{n} tips · {n_loci} loci", mut_txt]
    probe.remove()
    sizes = [10.5] * len(lines)
    if members is not None:
        mem = _member_lines(members)
        lines += mem
        sizes += [10.0] * len(mem)
    for ln, size in zip(lines, sizes, strict=True):
        tt = fig.text(0.008, y, ln, color=th["fg"], fontsize=size, va="top")
        _fit(fig, tt, occ, 8.0)
        y = _extent(fig, tt).y0 - 0.008
    # ---- legends (top left, below the header)
    iso_n = iso.value_counts()
    loc_n = lv["location"].astype(str).value_counts()
    h_iso = [Line2D([], [], color=iso_col[i], lw=0, marker="o", ms=7, label=f"{_ISO_LABEL[i]} ({int(iso_n.get(i, 0))})")
             for i in ("IgM", "IgD", "IgG", "IgA", "unknown")]  # fmt: skip
    # location classes the workflow computed (no "follicle 0" rows when it had no compartments or hotspots)
    computed = ({"CSR hotspot"} if maps.get("hotspots") is not None else set()) | (
        {"follicle", "T zone", "red pulp"} if maps.get("compartments") is not None else set()
    )
    locs = [k for k in _LOC_ORDER if k in computed or loc_n.get(k, 0) > 0]
    locs += [k for k in loc_n.index if k not in _LOC_ORDER]
    h_loc = [Line2D([], [], color=loc_col.get(k, loc_col["unknown"]), lw=0, marker="s", ms=7,
                    label=f"{k} ({int(loc_n.get(k, 0))})") for k in locs]  # fmt: skip
    h_map = [Line2D([], [], color=th["germ"], lw=0, marker="D", ms=6, label="germline")]
    if hot is not None:
        h_map.append(Line2D([0, 1], [0, 0], color=th["hot_edge"], lw=1.6, label="CSR hotspot"))
    if t_rich is not None:
        h_map.append(Line2D([0, 1], [0, 0], color=th["t_edge"], lw=1.3, label="T zone"))
    if b_rich is not None:
        h_map.append(Line2D([0, 1], [0, 0], color=th["fg"], lw=1.0, label="B-rich area"))
    kw = {"frameon": False, "fontsize": 9.5, "labelcolor": th["fg"], "handletextpad": 0.4, "borderaxespad": 0,
          "labelspacing": 0.35, "loc": "upper left"}  # fmt: skip

    def legend(handles: list, name: str, x: float, y_top: float) -> Any:
        lg = fig.legend(handles=handles, title=name, bbox_to_anchor=(x, y_top), **kw)
        lg.get_title().set_color(th["fg"])
        lg.get_title().set_fontsize(10)
        lg.get_title().set_fontweight("bold")
        lg._legend_box.align = "left"
        return lg

    y_leg = min(y - 0.004, 0.905)
    l1 = legend(h_iso, "tip isotype", 0.008, y_leg)
    _fit(fig, l1, occ, 7.5)
    l2 = legend(h_loc, "tip location", max(0.09, _extent(fig, l1).x1 + 0.012), y_leg)
    _fit(fig, l2, occ, 7.5)
    y3 = min(_extent(fig, l1).y0, _extent(fig, l2).y0) - 0.025
    l3 = legend(h_map, "map", 0.008, y3)
    _fit(fig, l3, occ, 7.5)
    e3 = _extent(fig, l3)
    bottom = e3.y0
    if norm is not None:
        cx = max(0.10, e3.x1 + 0.02)
        cw = max(0.06, min(0.12, _room(occ, y3 - 0.08, y3, pm.x0 - 0.006) - cx))
        cax = fig.add_axes((cx, y3 - 0.06, cw, 0.014))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.ax.tick_params(colors=th["fg"], labelsize=8.5)
        cb.outline.set_edgecolor(th["fg"])
        cb.ax.set_title("switched share (IgG+IgA)", color=th["fg"], fontsize=9.5, pad=3)
        cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:.0%}"))
        bottom = min(bottom, _extent(fig, cb.ax).y0 - 0.03)
    top = min(0.595, bottom - 0.03, pm.y0 - 0.02)
    rect = (0.075, max(0.15, top - 0.075))
    # ---- raster (bottom right, x aligned with the map) and tree (bottom left)
    axr = fig.add_axes((pm.x0, rect[0], pm.width, rect[1]))
    axr.set_facecolor(th["bg"])
    x0, x1 = axm.get_xlim()
    axr.set_xlim(x0, x1)
    axr.set_ylim(n - 0.4, -0.6)
    axr.add_collection(LineCollection([[(x0, r), (x1, r)] for r in range(n)], colors=th["row"], lw=0.5, zorder=0))
    axt = fig.add_axes((0.004, rect[0], min(0.232, pm.x0 - 0.016), rect[1]))
    axt.set_facecolor(th["bg"])
    axt.add_collection(LineCollection(tr.segments(row, ys), colors=th["tree"], lw=1.1, capstyle="butt", zorder=2))
    xmax = max(max(tr.div.values()), 1.0)
    axt.scatter([0], [ys[tr.germ]], marker="D", s=s_pt * 1.5, c=th["germ"], lw=0, zorder=5)
    axt.scatter(lv["div"].to_numpy(), row, s=s_pt, c=cols, lw=0, zorder=6)
    xr = xmax * 1.04 + 0.5
    axt.add_collection(LineCollection([[(d, r), (xr, r)] for d, r in zip(lv["div"], row, strict=True)],
                                      colors=th["guide"], lw=0.5, linestyles=(0, (1, 2)), zorder=1))  # fmt: skip
    axt.set_xlim(-0.03 * xmax - 0.8, xr)
    axt.set_ylim(n - 0.4, -0.6)
    # ---- overlay: drop lines, location squares, dots
    ov = fig.add_axes((0, 0, 1, 1), facecolor="none")
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.set_axis_off()
    inv = fig.transFigure.inverted()
    p_map = inv.transform(axm.transData.transform(xy))
    p_row = inv.transform(axr.transData.transform(np.c_[xy[:, 0], row]))
    ov.add_collection(LineCollection([[(a[0], a[1]), (b[0], b[1])] for a, b in zip(p_map, p_row, strict=True)],
                                     colors=cols, lw=0.7, alpha=0.30, zorder=1))  # fmt: skip
    pt = axt.get_position()
    x_sq = (pt.x1 + pm.x0) / 2
    p_sq = inv.transform(axr.transData.transform(np.c_[np.full(n, x0), np.arange(n)]))
    hrow = abs(p_sq[1, 1] - p_sq[0, 1]) if n > 1 else 0.02
    sq_w = min(0.012, max(0.004, pm.x0 - pt.x1 - 0.004))
    loc_of = dict(zip(row.astype(int), lv["location"].astype(str), strict=True))
    for r in range(n):
        ov.add_patch(Rectangle((x_sq - sq_w / 2, p_sq[r, 1] - hrow * 0.4), sq_w, hrow * 0.8,
                               facecolor=loc_col.get(loc_of[r], loc_col["unknown"]), edgecolor="none", zorder=3))  # fmt: skip
    o = np.argsort(lv["mutations"].to_numpy(), kind="stable")
    ov.scatter(p_row[:, 0], p_row[:, 1], s=s_pt, c=cols, lw=0, zorder=4)
    ov.scatter(p_map[o, 0], p_map[o, 1], s=s_pt, c=cols[o], lw=0.4, edgecolors=th["bg"], zorder=5)
    # ---- axes decorations
    txt = {"color": th["fg"], "fontsize": 11}
    for ax in (axt, axr):
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(th["fg"])
        ax.tick_params(axis="x", colors=th["fg"], labelsize=10)
        ax.set_yticks([])
    step = 10 if xmax > 40 else (5 if xmax > 10 else 2)
    axt.set_xticks(np.arange(0, xmax + 0.01, step))
    axt.set_xlabel("tree distance from germline (mutations)", **txt)
    span_mm = (x1 - x0) / 1000
    tick_mm = 1.0 if span_mm >= 2 else (0.5 if span_mm >= 1 else 0.2)
    ticks = np.arange(0, span_mm + 1e-9, tick_mm)
    axr.set_xticks(x0 + ticks * 1000, [f"{v_:g}" for v_ in ticks])
    axr.set_xlabel("position along the section (mm)", **txt)
    return fig


def lineage_gallery(
    result: Any,
    out_dir: str | Path,
    *,
    groupings: Sequence[str] = ("exact", "related"),
    style: str = "dark",
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
    background: str = "auto",
    prefix: str = "lineage",
) -> list[Path]:
    """Write one figure per selected tree (:func:`lineage_tree`) and an index table.

    Parameters
    ----------
    result
        A :class:`spaclone.bcr.LineageWorkflowResult`; the trees with ``drawn`` in ``result.selection`` are written
        in their rank order.
    out_dir
        Folder (created).
    groupings
        Groupings to draw.
    style, background
        As in :func:`lineage_tree`.
    formats
        File formats (``"png"``, ``"pdf"``, ``"svg"``).
    dpi
        Resolution of raster formats.
    prefix
        File name prefix: ``{prefix}_{grouping}_{rank:02d}.{format}``.

    Returns
    -------
    list of pathlib.Path
        The figure files in writing order, then the index ``{prefix}_gallery.csv`` (one row per figure: file,
        grouping, rank, tree id, tips, tree tips, loci, mutations, exact clones, isotype and location counts).
    """
    import matplotlib.pyplot as plt

    from spaclone.bcr.lineage_workflow import _grouping

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    index = []
    for gname in groupings:
        g = _grouping(gname)
        if g not in result.trees:
            raise KeyError(f"lineage_gallery: the result has no {g!r} grouping")
        for r in result.drawn(g).itertuples():
            fig = lineage_tree(result, r.tree_id, grouping=g, style=style, background=background)
            files = []
            for fmt in formats:
                f = out / f"{prefix}_{g}_{int(r.rank):02d}.{fmt}"
                fig.savefig(f, dpi=dpi, facecolor=fig.get_facecolor())
                files.append(f)
                paths.append(f)
            plt.close(fig)
            tp = result.tree_tips(r.tree_id, grouping=g)
            index.append(
                {
                    "file": files[0].name,
                    "grouping": g,
                    "rank": int(r.rank),
                    "tree_id": r.tree_id,
                    "n_tips": int(r.n_tips),
                    "n_tree_tips": int(r.n_tree_tips),
                    "n_loci": int(r.n_loci),
                    "max_mutations": int(r.max_mutations),
                    "median_mutations": float(r.median_mutations),
                    "n_exact_clones": int(r.n_exact_clones),
                    "isotypes": "; ".join(f"{k} {v}" for k, v in tp["isotype"].value_counts().items()),
                    "locations": "; ".join(f"{k} {v}" for k, v in tp["location"].value_counts().items()),
                }
            )
    idx = out / f"{prefix}_gallery.csv"
    pd.DataFrame(index, columns=["file", "grouping", "rank", "tree_id", "n_tips", "n_tree_tips", "n_loci",
                                 "max_mutations", "median_mutations", "n_exact_clones", "isotypes",
                                 "locations"]).to_csv(idx, index=False)  # fmt: skip
    paths.append(idx)
    return paths
