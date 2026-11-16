"""Hotspot layers from a hotspot workflow result: share maps, hotspot outlines, regions, loci dots, module scores.

The images, outlines and colour limits are those of the static atlas (:mod:`spaclone.pl.atlas`, ``_AtlasData``):
the share map covers the tissue and fades into the tissue colour where there were too few loci, CSR hotspots are
outlined in yellow and SHM hotspots in vermilion, colour limits are the 2nd to 99.5th percentiles of the share on the
pixels used for inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from spaclone.pl import _colors as C
from spaclone.pl.interactive import _geo as G
from spaclone.pl.interactive import _spec as P

#: Labels of the two maps.
MAP_LABEL = {"csr": "switched share", "shm": "mutated share"}
OUTLINE_LABEL = {"csr": "CSR hotspot", "shm": "SHM hotspot"}


def atlas_data(result: Any, *, style: str, rotate: bool | float = False, invert_y: bool = True) -> Any:
    from spaclone.pl.atlas import _AtlasData

    return _AtlasData(result, style=style, rotate=rotate, invert_y=invert_y)


class HotspotLayers:
    """Layers of one workflow result in display coordinates given by the 3 x 3 affine ``m`` (original µm -> display)."""

    def __init__(self, spec: P.Spec, panel: dict[str, Any], result: Any, m: np.ndarray, *, style: str = "dark",
                 base_when: Mapping[str, Any] | None = None, key: str = "_", data: Any = None) -> None:  # fmt: skip
        self.spec, self.panel, self.m, self.key = spec, panel, m, key
        self.mode = "dark" if style == "dark" else "light"
        self.pal = C.ATLAS_STYLES[self.mode]
        self.base = dict(base_when or {})
        self.data = data if data is not None else atlas_data(result, style=self.mode, rotate=False)
        self.result = result
        self._tissue_id: str | None = None

    def when(self, **kw: Any) -> dict[str, Any]:
        out = dict(self.base)
        out.update({k: v for k, v in kw.items() if v is not None})
        return out

    def disp(self, lines: list[np.ndarray]) -> list[np.ndarray]:
        return [G.apply(self.m, v) for v in lines]

    # ---- tissue
    def tissue(self, *, when: Mapping[str, Any] | None = None, edge: bool = True) -> str | None:
        """Tissue fill (the clip of the maps) and its outline."""
        rings = G.rings_from_paths(self.data.tissue_fill())
        if not rings:
            return None
        lay = P.paths(self.spec, self.disp(rings), closed=True, fill=self.pal["tissue"], when=self.when(**(when or {})),
                      name="tissue", hover=None)  # fmt: skip
        lay["hover"] = False
        self.panel["layers"].append(lay)
        self._tissue_id = lay["id"]
        if edge:
            el = self.data.tissue_edge()
            if el:
                self.edge_layer = P.paths(self.spec, self.disp(el), stroke=self.pal["tissue_edge"], width=1.0,
                                          when=self.when(**(when or {})), name="tissue edge")  # fmt: skip
                self.edge_layer["hover"] = False
        return lay["id"]

    def add_edge(self) -> None:
        if getattr(self, "edge_layer", None) is not None:
            self.panel["layers"].append(self.edge_layer)

    # ---- share maps
    def share_map(
        self, which: str, *, when: Mapping[str, Any] | None = None, legend: bool = True
    ) -> dict[str, Any] | None:
        d = self.data
        img = d.map_image(which)
        if img is None:
            return None
        sm = d.sm_csr if which == "csr" else d.sm_shm
        norm = d._cache.get(f"norm_{which}")
        vals = np.asarray(sm.share, dtype=float)
        elig = np.asarray(sm.eligible, dtype=bool) & d.tissue & np.isfinite(vals)
        shown = np.where(elig, vals, np.nan)
        lay = P.image(self.spec, G.grid_rows(img), G.grid_image(d.grid, self.m), values=G.grid_rows(shown),
                      value_label=f"{MAP_LABEL[which]} (smoothed)", value_fmt="pct", clip=self._tissue_id,
                      when=self.when(**(when or {})), name=f"{which.upper()} map")  # fmt: skip
        self.panel["layers"].append(lay)
        if legend and norm is not None:
            ramp = self.pal["csr_ramp" if which == "csr" else "shm_ramp"]
            P.legend_colorbar(self.spec, f"{MAP_LABEL[which]} of loci (smoothed)", ramp, (norm.vmin, norm.vmax),
                              fmt="pct0", when=lay.get("when"),
                              note="2nd to 99.5th percentile of the pixels used for inference")  # fmt: skip
        return lay

    # ---- outlines
    def hotspot_outline(self, which: str, *, when: Mapping[str, Any] | None = None, okey: str | None = None,
                        width: float = 1.6) -> dict[str, Any] | None:  # fmt: skip
        lines = self.data.hotspot_lines(which)
        if not lines:
            return None
        lay = P.paths(self.spec, self.disp(lines), stroke=self.pal[which], width=width, halo=self.pal["bg"],
                      when=self.when(**(when or {})), name=OUTLINE_LABEL[which], okey=okey)  # fmt: skip
        lay["hover"] = False
        self.panel["layers"].append(lay)
        return lay

    def regions(self, *, when: Mapping[str, Any] | None = None, okey: str | None = None, labels: bool = True,
                width: float = 2.2, control: str | None = None) -> tuple[dict | None, list[Any]]:  # fmt: skip
        """Region outlines (yellow, the atlas' zoom style) and their numbers; hover / click a region to zoom to it."""
        d = self.data
        ids = list(d.regions.index)
        if not ids:
            return None, []
        rings, hover_names, actions = [], [], []
        boxes = []
        for rid in ids:
            mask = d.region_mask(rid)
            ln = list(self.disp(G.rings_from_paths(_fill(mask, d.grid))))
            if not ln:
                continue
            for ring in ln:
                rings.append(ring)
                hover_names.append(d.region_name(rid))
                lo, hi = ring.min(axis=0), ring.max(axis=0)
                actions.append({"set": [control, str(rid)]} if control else {"zoom": _pad_box(lo, hi)})
            p = np.vstack(ln)
            boxes.append((rid, p.min(axis=0), p.max(axis=0)))
        if not rings:
            return None, []
        lay = P.paths(self.spec, rings, closed=True, stroke=self.pal["csr"], width=width, halo=self.pal["bg"],
                      when=self.when(**(when or {})), name="regions", okey=okey,
                      hover={"title": "region"})  # fmt: skip
        lay["hover"] = {"rows": [{"list": hover_names, "label": "region"}]}
        lay["actions"] = actions
        self.panel["layers"].append(lay)
        out = [lay]
        if labels:
            xy, labs = [], []
            for rid, lo, _hi in boxes:
                xy.append((lo[0], lo[1]))
                labs.append(str(rid))
            tl = P.text(self.spec, xy, labs, color=self.pal["fg"], size=12.5, anchor="end", baseline="bottom",
                        when=self.when(**(when or {})), dx=-2, dy=-2)  # fmt: skip
            tl["okey"] = okey
            self.panel["layers"].append(tl)
            out.append(tl)
        return lay, [(rid, _pad_box(lo, hi)) for rid, lo, hi in boxes]

    def gc_outline(self, *, when: Mapping[str, Any] | None = None, okey: str | None = None) -> dict | None:
        lines = self.data.gc_lines()
        if not lines:
            return None
        lay = P.paths(self.spec, self.disp(lines), stroke=self.pal["gcb"], width=1.2, when=self.when(**(when or {})),
                      name="GC B (GEX)", okey=okey)  # fmt: skip
        lay["hover"] = False
        self.panel["layers"].append(lay)
        return lay

    # ---- loci dots (the atlas' zoom panels)
    def loci_dots(self, which: str, *, when: Mapping[str, Any] | None = None, min_zoom: float | None = None,
                  okey: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
        d = self.data
        lo = d.loci
        if lo is None or not len(lo):
            return []
        xy = G.apply(self.m, d.loci_xy)
        pal = self.pal
        out = []
        hover = _loci_hover(self.spec, lo)
        if which == "csr":
            state, cls = d.isotype_dots()
            groups = [("not switched", state == 0, pal["nomut"], 4.0, None)]
            for c in ("IgG", "IgA", "IgE"):
                groups.append((c, (state == 1) & (cls == c), pal["iso"][c], 9.0, pal["dot_edge"]))
        else:
            mut = d.mutation_dots()
            groups = [("not mutated", mut == 0, pal["nomut"], 4.0, None),
                      (d.mutation_label(), mut == 1, pal["mut"], 9.0, pal["dot_edge"])]  # fmt: skip
        for name, m_, col, um, edge in groups:
            if not m_.any():
                continue
            idx = np.flatnonzero(m_)
            sub_hover = {"title": hover["title"], "rows": hover["rows"]}
            lay = P.points(self.spec, xy[idx], color={"kind": "const", "color": col}, size={"um": um, "min": 3.0 if edge else 2.0,
                           "max": 28.0}, edge={"color": edge, "px": 0.7} if edge else None, when=self.when(**(when or {})),
                           name=name, min_zoom=min_zoom, okey=f"{okey}:{name}" if okey else None)  # fmt: skip
            lay["hover"] = _subset_hover(self.spec, sub_hover, idx)
            self.panel["layers"].append(lay)
            out.append(lay)
        return out

    # ---- module score image of a region's GEX label
    def module_image(self, label: Any, *, when: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        d = self.data
        key = d.module_key(label)
        if key is None:
            return None
        img, _ = d.module_image(key)
        if img is None:
            return None
        lay = P.image(self.spec, G.grid_rows(img), G.grid_image(d.grid, self.m), clip=self._tissue_id,
                      when=self.when(**(when or {})), name=f"{key} score")  # fmt: skip
        self.panel["layers"].append(lay)
        norm = d._cache.get(f"norm_module_{key}")
        if norm is not None:
            P.legend_colorbar(self.spec, f"{key} module score", self.pal["score_ramp"], (norm.vmin, norm.vmax),
                              when=lay.get("when"), note="2nd to 99.5th percentile of the scored units")  # fmt: skip
        return lay


def _fill(mask: np.ndarray, grid: Any) -> list:
    from spaclone.pl.atlas import _mask_fill

    return _mask_fill(mask, grid, 0.7)


def _pad_box(lo: np.ndarray, hi: np.ndarray, pad_um: float = 90.0) -> list[float]:
    return [float(lo[0] - pad_um), float(hi[0] + pad_um), float(lo[1] - pad_um), float(hi[1] + pad_um)]


def _loci_hover(spec: P.Spec, lo: pd.DataFrame) -> dict[str, Any]:
    rows = []
    title = (
        P.hover_cat(spec, "clone " + lo["clone_id"].astype("string").fillna("NA").astype(str), "clone")
        if "clone_id" in lo.columns
        else None
    )
    for col, lab in (("isotype_class", "isotype"), ("n_records", "records"), ("switched_share", "switched share"),
                     ("excess", "excess V mutations"), ("region_id", "region")):  # fmt: skip
        if col not in lo.columns:
            continue
        v = lo[col]
        if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
            rows.append(
                P.hover_num(spec, pd.to_numeric(v, errors="coerce"), lab, fmt="pct" if "share" in col else None)
            )
        else:
            rows.append(P.hover_cat(spec, v.astype("string").to_numpy(dtype=object), lab))
    return {"title": title, "rows": rows}


def _subset_hover(spec: P.Spec, hover: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """The hover fields of all loci, restricted to ``idx`` (new blocks of the subset)."""

    def sub(r: dict[str, Any]) -> dict[str, Any]:
        out = dict(r)
        if "codes" in r:
            out["codes"] = spec.blob(np.asarray(spec.arrays[r["codes"]])[idx], spec.blobs[r["codes"]]["t"])
        if "values" in r:
            v = np.asarray(spec.arrays[r["values"]], dtype=float)[idx]
            out["values"] = spec.blob(v, "f4")
        return out

    return {"title": sub(hover["title"]) if hover.get("title") else None, "rows": [sub(r) for r in hover["rows"]]}
