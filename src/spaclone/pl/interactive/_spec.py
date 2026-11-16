"""The figure specification: panels of layers over shared data blocks, controls, legends and selection statistics.

:class:`Spec` collects everything a figure shows and serialises it to the JSON the browser engine reads
(``engine.js``). Layers reference data blocks by id; a block is stored once even when several layers use it (same
bytes, same id). The numpy arrays behind the blocks stay on the Python side (:attr:`Spec.arrays`) for the static
picture (:mod:`spaclone.pl.interactive._png`) and the Python selection statistics
(:meth:`spaclone.pl.interactive.InteractiveFigure.summary`).

Coordinates in layers are display coordinates: for tissue maps, micrometres after the section's rotation
(:class:`spaclone.pl._frame.SectionFrame`), with ``frames`` mapping them back to the original micrometres.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone.pl.interactive import _encode as E

#: Missing category label shown in hover text and legends.
MISSING = "NA"


def _hex(c: Any) -> str:
    """A colour as ``#rrggbb``."""
    from matplotlib.colors import to_hex

    return to_hex(c, keep_alpha=False)


def as_text(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Values as strings (object array) and a mask of missing ones (None, NaN, pandas NA)."""
    s = (
        values.reset_index(drop=True)
        if isinstance(values, pd.Series)
        else pd.Series(
            list(np.asarray(values, dtype=object).ravel()) if not isinstance(values, (list, tuple)) else list(values),
            dtype=object,
        )
    )
    miss = s.isna().to_numpy(dtype=bool)
    txt = s.astype(object).where(~miss, "").map(str).to_numpy(dtype=object)
    return txt, miss


class Spec:
    """Builder of one figure.

    Parameters
    ----------
    style
        ``"dark"`` or ``"light"``.
    title, subtitle
        Header text.
    call
        The Python call that made the figure (written into the saved file).
    """

    def __init__(self, *, style: str = "dark", title: str = "", subtitle: str | None = None, call: str = "") -> None:
        if style not in ("dark", "light"):
            raise ValueError(f"style must be 'dark' or 'light', got {style!r}")
        self.style = style
        self.title = title
        self.subtitle = subtitle
        self.call = call
        self.blobs: dict[str, dict[str, Any]] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.tables: dict[str, list[str]] = {}
        self.panels: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.legend: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.stats: dict[str, dict[str, Any]] = {}
        self.genes: dict[str, dict[str, Any]] = {}
        self.frames: dict[str, list[float]] = {}
        self.layout: dict[str, Any] = {}
        self.hidden: list[dict[str, Any]] = []
        self.layers_off: list[str] = []
        self.extra: dict[str, Any] = {}
        self._seen: dict[str, str] = {}
        self._tseen: dict[str, str] = {}
        self._n = 0
        self._layer_n = 0

    # ---- data blocks
    def blob(self, values: Any, dtype: str, *, compress: bool | None = None, quantise: int | None = None) -> str:
        """Store an array; returns its block id (identical arrays share one block)."""
        arr = np.asarray(values)
        if quantise:
            blk = E.encode_quantised(arr, bits=quantise, compress=compress)
        else:
            blk = E.encode_array(arr, dtype, compress=compress)
        key = hashlib.sha1((repr(sorted((k, v) for k, v in blk.items() if k != "d")) + blk["d"]).encode()).hexdigest()
        if key in self._seen:
            return self._seen[key]
        bid = f"b{self._n}"
        self._n += 1
        self.blobs[bid] = blk
        if quantise:
            self.arrays[bid] = E.decode(blk)
        elif dtype in ("f4", "f8"):
            f = np.asarray(arr, dtype=float).ravel()
            self.arrays[bid] = np.where(np.isfinite(f), f, np.nan).astype(np.float32 if dtype == "f4" else np.float64)
        else:
            self.arrays[bid] = np.asarray(arr).ravel()
        self._seen[key] = bid
        return bid

    def raw_blob(self, blk: dict[str, Any], array: np.ndarray | None = None) -> str:
        """Store a ready-made block (varints, bytes)."""
        bid = f"b{self._n}"
        self._n += 1
        self.blobs[bid] = blk
        if array is not None:
            self.arrays[bid] = array
        return bid

    def table(self, values: Sequence[Any]) -> str:
        """Store a list of strings; returns its id."""
        vals = [str(v) for v in values]
        key = hashlib.sha1("\u0000".join(vals).encode("utf-8", "surrogatepass")).hexdigest() + str(len(vals))
        if key in self._tseen:
            return self._tseen[key]
        tid = f"t{len(self.tables)}"
        self.tables[tid] = vals
        self._tseen[key] = tid
        return tid

    def xy(self, xy: Any) -> tuple[str, str]:
        """Blocks of x and y coordinates (float32)."""
        a = np.asarray(xy, dtype=float).reshape(-1, 2)
        return self.blob(a[:, 0], "f4"), self.blob(a[:, 1], "f4")

    def codes(self, values: Any, names: Sequence[str] | None = None) -> tuple[str, list[str], np.ndarray]:
        """Categorical codes of ``values`` against ``names`` (default: sorted distinct values); missing -> len(names)."""
        txt, miss = as_text(values)
        if names is None:
            names = sorted(set(txt[~miss].tolist()))
        names = [str(n) for n in names]
        codes = np.asarray(pd.Index(names, dtype=object).get_indexer(pd.Index(txt, dtype=object)), dtype=np.int64)
        codes[(codes < 0) | miss] = len(names)
        return self.blob(codes, E.code_dtype(len(names))), names, codes

    def layer_id(self, prefix: str = "L") -> str:
        self._layer_n += 1
        return f"{prefix}{self._layer_n}"

    # ---- structure
    def panel(self, pid: str, kind: str, **kw: Any) -> dict[str, Any]:
        """Add a panel (``kind`` = ``"map"`` or ``"plot"``)."""
        p = {"id": pid, "kind": kind, "layers": []}
        p.update({k: v for k, v in kw.items() if v is not None})
        self.panels.append(p)
        return p

    def control(self, cid: str, kind: str, **kw: Any) -> dict[str, Any]:
        """Add a control (``kind`` = ``"select"`` or ``"search"``)."""
        c = {"id": cid, "kind": kind}
        c.update({k: v for k, v in kw.items() if v is not None})
        self.controls.append(c)
        return c

    def note(self, text: str, *, when: Mapping[str, Any] | None = None, warn: bool = False) -> None:
        """A line under the figure (only in the states matching ``when``)."""
        n: dict[str, Any] = {"text": str(text)}
        if when:
            n["when"] = dict(when)
        if warn:
            n["warn"] = True
        if n not in self.notes:
            self.notes.append(n)

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready specification."""
        from spaclone.pl.interactive._html import package_version

        out: dict[str, Any] = {"v": 1, "version": package_version(), "call": self.call, "style": self.style,
                               "title": self.title}  # fmt: skip
        if self.subtitle:
            out["subtitle"] = self.subtitle
        for key in ("layout", "controls", "panels", "legend", "notes", "stats", "genes", "frames", "hidden",
                    "layers_off"):  # fmt: skip
            val = getattr(self, key)
            if val:
                out[key] = val
        out.update(self.extra)
        out["tables"] = self.tables
        out["blobs"] = self.blobs
        return out

    def nbytes(self) -> int:
        """Approximate size of the data in the saved file (base64 text of the blocks and the string tables)."""
        n = sum(E.block_bytes(b) for b in self.blobs.values())
        n += sum(sum(len(s) + 3 for s in t) for t in self.tables.values())
        return n


# ------------------------------------------------------------------------------------------------ colour encodings


def cat_color(
    spec: Spec,
    values: Any,
    colors: Mapping[str, str],
    *,
    names: Sequence[str] | None = None,
    nan: str | None = None,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    """A categorical colour: codes against ``names`` (default: the keys of ``colors``), each name in its colour."""
    names = list(colors) if names is None else [str(n) for n in names]
    cid, names, codes = spec.codes(values, names)
    enc: dict[str, Any] = {"kind": "cat", "codes": cid, "colors": [_hex(colors.get(n, "#888888")) for n in names],
                           "names": names}  # fmt: skip
    if nan is not None:
        enc["nan"] = _hex(nan)
    return enc, names, codes


def num_color(
    spec: Spec,
    values: Any,
    ramp: Sequence[str],
    *,
    domain: tuple[float, float] | None = None,
    quantiles: tuple[float, float] = (0.02, 0.995),
    scale: str = "lin",
    nan: str | None = None,
    quantise: int | None = 16,
    order: bool = True,
) -> tuple[dict[str, Any], tuple[float, float]]:
    """A continuous colour on ``ramp`` (colour limits ``domain``, default the ``quantiles`` of the finite values)."""
    v = np.asarray(values, dtype=float)
    fin = v[np.isfinite(v) & ((v > 0) if scale == "log" else True)]
    if domain is None:
        if len(fin):
            lo, hi = (float(x) for x in np.quantile(fin, quantiles))
        else:
            lo, hi = (1.0, 10.0) if scale == "log" else (0.0, 1.0)
        if hi <= lo:
            hi = lo * 10 if scale == "log" and lo > 0 else lo + (abs(lo) * 0.1 or 1.0)
        domain = (lo, hi)
    vid = spec.blob(v, "f4", quantise=quantise) if quantise else spec.blob(v, "f4")
    enc: dict[str, Any] = {"kind": "num", "values": vid, "ramp": [_hex(c) for c in ramp], "domain": [float(domain[0]),
                           float(domain[1])], "scale": scale}  # fmt: skip
    if nan is not None:
        enc["nan"] = _hex(nan)
    if not order:
        enc["order"] = False
    return enc, (float(domain[0]), float(domain[1]))


# ------------------------------------------------------------------------------------------------ layers


def points(
    spec: Spec,
    xy: Any,
    *,
    color: dict[str, Any],
    size: Mapping[str, float] | None = None,
    shape: str = "circle",
    alpha: float = 1.0,
    edge: Mapping[str, Any] | None = None,
    hover: Mapping[str, Any] | None = None,
    when: Mapping[str, Any] | None = None,
    name: str | None = None,
    select: str | None = None,
    order: Any = None,
    key: Any = None,
    key_group: str | None = None,
    lid: str | None = None,
    xy_ids: tuple[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A points layer (markers in display coordinates)."""
    if xy_ids is None:
        a = np.asarray(xy, dtype=float).reshape(-1, 2)
        n = len(a)
        xid, yid = spec.xy(a)
    else:
        xid, yid = xy_ids
        n = len(spec.arrays[xid])
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "points", "x": xid, "y": yid, "n": int(n),
                           "color": color, "size": dict(size or {"px": 3.0}), "shape": shape}  # fmt: skip
    if alpha < 1:
        lay["alpha"] = float(alpha)
    if edge:
        lay["edge"] = {"color": _hex(edge["color"]), "px": float(edge.get("px", 0.6))}
    if hover is False:
        lay["hover"] = False
    elif hover:
        lay["hover"] = dict(hover)
    if when:
        lay["when"] = dict(when)
    if name:
        lay["name"] = name
    if select:
        lay["select"] = select
    if order is not None:
        o = np.asarray(order, dtype=np.int64)
        lay["order"] = spec.blob(o, "u4")
    if key is not None:
        lay["key"] = spec.blob(np.asarray(key, dtype=np.int64), E.code_dtype(int(np.max(key)) + 1 if len(key) else 1))
        lay["key_group"] = key_group or "key"
    lay.update({k: v for k, v in extra.items() if v is not None})
    return lay


def rings_of(lines: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flattened vertices and vertex offsets of a list of ``(k, 2)`` polylines."""
    lines = [np.asarray(v, dtype=float).reshape(-1, 2) for v in lines if len(v) >= 2]
    if not lines:
        return np.zeros((0, 2)), np.zeros(1, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum([len(v) for v in lines])]).astype(np.int64)
    return np.vstack(lines), off


def paths(
    spec: Spec,
    lines: Sequence[np.ndarray],
    *,
    closed: bool = False,
    stroke: str | None = None,
    width: float = 1.0,
    fill: str | None = None,
    dash: Sequence[float] | None = None,
    halo: str | None = None,
    alpha: float = 1.0,
    when: Mapping[str, Any] | None = None,
    name: str | None = None,
    lid: str | None = None,
    arrows: Mapping[str, Any] | None = None,
    hover: Mapping[str, Any] | None = None,
    colors: Sequence[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A layer of polylines or filled rings (display coordinates)."""
    v, off = rings_of(lines)
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "paths", "xy": spec.blob(v.ravel(), "f4"),
                           "off": spec.blob(off, "u4"), "closed": bool(closed), "width": float(width)}  # fmt: skip
    if stroke:
        lay["stroke"] = _hex(stroke)
    if colors is not None:
        lay["colors"] = [_hex(c) for c in colors]
    if fill:
        lay["fill"] = _hex(fill)
    if dash:
        lay["dash"] = [float(d) for d in dash]
    if halo:
        lay["halo"] = _hex(halo)
    if alpha < 1:
        lay["alpha"] = float(alpha)
    if when:
        lay["when"] = dict(when)
    if name:
        lay["name"] = name
    if arrows:
        lay["arrows"] = dict(arrows)
    if hover:
        lay["hover"] = dict(hover)
    lay.update({k: v for k, v in extra.items() if v is not None})
    return lay


def segments(
    spec: Spec,
    segs: Any,
    *,
    color: str | dict[str, Any] = "#888888",
    width: float = 1.0,
    alpha: float = 1.0,
    when: Mapping[str, Any] | None = None,
    dash: Sequence[float] | None = None,
    lid: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A layer of straight segments ``(n, 2, 2)`` (display coordinates)."""
    s = np.asarray(segs, dtype=float).reshape(-1, 4)
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "segs", "xy": spec.blob(s.ravel(), "f4"), "n": len(s),
                           "color": color if isinstance(color, dict) else _hex(color), "width": float(width)}  # fmt: skip
    if alpha < 1:
        lay["alpha"] = float(alpha)
    if when:
        lay["when"] = dict(when)
    if dash:
        lay["dash"] = [float(d) for d in dash]
    lay.update({k: v for k, v in extra.items() if v is not None})
    return lay


def rects(
    spec: Spec,
    boxes: Any,
    *,
    fill: Any = None,
    stroke: str | None = None,
    width: float = 1.0,
    labels: Sequence[str] | None = None,
    hover: Mapping[str, Any] | None = None,
    when: Mapping[str, Any] | None = None,
    lid: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A layer of rectangles ``(n, 4)`` = ``x0, y0, x1, y1`` (cells of a heatmap, bars, windows)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "rects", "r": spec.blob(b.ravel(), "f4"), "n": len(b)}
    if fill is not None:
        lay["fill"] = _hex(fill) if isinstance(fill, str) else fill
    if stroke:
        lay["stroke"] = _hex(stroke)
        lay["width"] = float(width)
    if labels is not None:
        lay["labels"] = spec.table(labels)
    if hover:
        lay["hover"] = dict(hover)
    if when:
        lay["when"] = dict(when)
    lay.update({k: v for k, v in extra.items() if v is not None})
    return lay


def text(
    spec: Spec,
    xy: Any,
    labels: Sequence[str],
    *,
    color: str | None = None,
    size: float = 12.0,
    weight: str = "600",
    anchor: str = "middle",
    baseline: str = "middle",
    halo: str | None = "",
    when: Mapping[str, Any] | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
    lid: str | None = None,
) -> dict[str, Any]:
    """A layer of text labels at display coordinates (``halo=""``: the surface colour; ``None``: no halo)."""
    a = np.asarray(xy, dtype=float).reshape(-1, 2)
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "text", "xy": spec.blob(a.ravel(), "f4"),
                           "text": spec.table(labels), "size": float(size), "weight": weight, "anchor": anchor,
                           "baseline": baseline}  # fmt: skip
    if color:
        lay["color"] = _hex(color)
    if halo is None:
        lay["halo"] = None
    elif halo:
        lay["halo"] = _hex(halo)
    if when:
        lay["when"] = dict(when)
    if dx:
        lay["dx"] = float(dx)
    if dy:
        lay["dy"] = float(dy)
    return lay


def image(
    spec: Spec,
    rgba: np.ndarray,
    m: Sequence[float],
    *,
    values: np.ndarray | None = None,
    value_label: str = "value",
    value_fmt: str | None = None,
    clip: str | None = None,
    smooth: bool = True,
    when: Mapping[str, Any] | None = None,
    name: str | None = None,
    lid: str | None = None,
    alpha: float = 1.0,
    hover_title: str | None = None,
) -> dict[str, Any]:
    """A raster layer: ``rgba`` ``(rows, columns, 4)`` placed by the affine ``m`` (image pixels -> display).

    ``m`` follows the canvas convention ``X = m0 u + m2 v + m4``, ``Y = m1 u + m3 v + m5`` for pixel coordinates
    ``(u, v)`` (column, row; pixel corners at integers). ``values`` (same rows x columns) are shown on hover.
    """
    from spaclone.pl.interactive._html import png_data_uri

    img = np.asarray(rgba)
    lay: dict[str, Any] = {"id": lid or spec.layer_id(), "type": "image", "src": png_data_uri(img),
                           "m": [float(v) for v in m], "w": int(img.shape[1]), "h": int(img.shape[0])}  # fmt: skip
    a = np.array([[m[0], m[2]], [m[1], m[3]]], dtype=float)
    ia = np.linalg.inv(a)
    t = -ia @ np.array([m[4], m[5]], dtype=float)
    lay["inv"] = [float(ia[0, 0]), float(ia[1, 0]), float(ia[0, 1]), float(ia[1, 1]), float(t[0]), float(t[1])]
    if values is not None:
        vals = np.asarray(values, dtype=float)
        lay["values"] = {"blob": spec.blob(vals.ravel(), "f4", quantise=16), "w": int(vals.shape[1]),
                         "h": int(vals.shape[0]), "label": value_label}  # fmt: skip
        if value_fmt:
            lay["values"]["fmt"] = value_fmt
    if clip:
        lay["clip"] = clip
    if not smooth:
        lay["smooth"] = False
    if when:
        lay["when"] = dict(when)
    if name:
        lay["name"] = name
    if alpha < 1:
        lay["alpha"] = float(alpha)
    if hover_title:
        lay["hover"] = {"title": hover_title}
    spec.arrays[lay["id"] + ":rgba"] = img
    return lay


# ------------------------------------------------------------------------------------------------ hover fields


def hover_cat(spec: Spec, values: Any, label: str) -> dict[str, Any]:
    """A hover row showing a categorical value (dictionary-encoded)."""
    txt, miss = as_text(values)
    codes_all, uniq = pd.factorize(pd.Series(np.where(miss, None, txt), dtype=object), use_na_sentinel=True)
    names = [str(u) for u in uniq]
    codes = np.asarray(codes_all, dtype=np.int64)
    codes[codes < 0] = len(names)
    return {"label": label, "codes": spec.blob(codes, E.code_dtype(len(names))), "table": spec.table(names),
            "nan": len(names)}  # fmt: skip


def hover_num(spec: Spec, values: Any, label: str, *, fmt: str | None = None, exact: bool = False) -> dict[str, Any]:
    """A hover row showing a number (16-bit quantised unless ``exact``; integers below 2**24 are always exact)."""
    v = np.asarray(values, dtype=float)
    fin = v[np.isfinite(v)]
    integer = len(fin) and np.all(fin == np.round(fin)) and np.abs(fin).max() < 2**24
    vid = spec.blob(v, "f4") if (exact or integer) else spec.blob(v, "f4", quantise=16)
    row: dict[str, Any] = {"label": label, "values": vid}
    if fmt:
        row["fmt"] = fmt
    return row


def legend_cat(
    spec: Spec,
    title: str | None,
    names: Sequence[str],
    colors: Sequence[str],
    counts: Sequence[int] | None,
    *,
    layer: str | None = None,
    shape: str = "point",
    when: Mapping[str, Any] | None = None,
    skip_empty: bool = True,
) -> dict[str, Any]:
    """A legend section of categories (clickable when ``layer`` is given: show / hide a category)."""
    items = []
    for k, (n, c) in enumerate(zip(names, colors, strict=True)):
        cnt = None if counts is None else int(counts[k])
        if skip_empty and cnt == 0:
            continue
        it: dict[str, Any] = {"label": str(n), "color": _hex(c), "shape": shape}
        if cnt is not None:
            it["n"] = cnt
        if layer is not None:
            it["layer"] = layer
            it["cat"] = k
        items.append(it)
    sec: dict[str, Any] = {"title": title, "items": items}
    if when:
        sec["when"] = dict(when)
    spec.legend.append(sec)
    return sec


def legend_colorbar(
    spec: Spec,
    label: str,
    ramp: Sequence[str],
    domain: tuple[float, float],
    *,
    fmt: str | None = None,
    scale: str = "lin",
    note: str | None = None,
    when: Mapping[str, Any] | None = None,
    ticks: Sequence[float] | None = None,
    ticklabels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A colour-bar legend section."""
    sec: dict[str, Any] = {"kind": "colorbar", "label": label, "ramp": [_hex(c) for c in ramp],
                           "domain": [float(domain[0]), float(domain[1])], "scale": scale}  # fmt: skip
    if fmt:
        sec["fmt"] = fmt
    if note:
        sec["note"] = note
    if when:
        sec["when"] = dict(when)
    if ticks is not None:
        sec["ticks"] = [float(t) for t in ticks]
    if ticklabels is not None:
        sec["ticklabels"] = [str(t) for t in ticklabels]
    spec.legend.append(sec)
    return sec


def legend_lines(
    spec: Spec,
    title: str | None,
    items: Sequence[tuple[str, str, str]],
    *,
    when: Mapping[str, Any] | None = None,
    layers: Sequence[str | None] | None = None,
) -> dict[str, Any] | None:
    """A legend section of outlines: ``(label, colour, shape)`` with shape ``line``, ``dash`` or ``box``."""
    if not items:
        return None
    out = []
    for k, (lab, col, shp) in enumerate(items):
        it: dict[str, Any] = {"label": lab, "color": _hex(col), "shape": shp}
        if layers is not None and layers[k]:
            it["layer"] = layers[k]
        out.append(it)
    sec: dict[str, Any] = {"title": title, "items": out}
    if when:
        sec["when"] = dict(when)
    spec.legend.append(sec)
    return sec
