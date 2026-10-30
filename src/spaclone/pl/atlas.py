"""Hotspot atlas: where class switching (CSR) and somatic hypermutation (SHM) concentrate in a section.

The atlas zooms on the regions where both concentrate and shows what the transcriptome says about each region.

:func:`hotspot_atlas` draws one black 16:9 figure from the result of :func:`spaclone.bcr.hotspot_workflow`:

* top row: the whole section twice, the switched share of loci (CSR map, blue, CSR hotspots outlined in yellow)
  and the mutated share (SHM map, green, SHM hotspots outlined in vermilion), with numbered windows;
* one column per region where a CSR hotspot overlaps an SHM hotspot: the CSR zoom (isotype of every locus: IgG red,
  IgA orange, IgM / IgD grey), the SHM zoom (mutated loci vermilion, the others grey) and the transcriptome zoom
  (the module score of the cell type the region's profile points to, in purple, with the region outline in yellow,
  the germinal-centre B area found from gene expression in cyan, and units without a score in dark grey);
* a legend row.

:func:`hotspot_overview` draws the top row alone and :func:`region_zoom` the zooms of one region. The section is
rotated about its centre so that its long axis is horizontal, and drawn in image convention (y increasing
downwards) unless ``invert_y=False``. Every function returns the Figure (or the Axes for a single panel) and never
calls ``show()`` or ``savefig()``.

The layout reproduces the 2026-10 archive figure ``03-01_QC_StereoIR_hotspot_zoom_GEX.png`` (scripts
``T8e_gc_gex_figures.py``, ``T7b_csr_shm_overlap_map.py``, ``T8_gc_zoom.py``): font sizes and line widths are those of
that 13.33 x 7.5 in slide, scaled with the figure width.
"""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spaclone import _schema as S
from spaclone.pl._colors import ATLAS_STYLES

__all__ = ["hotspot_atlas", "hotspot_overview", "region_zoom"]

logger = logging.getLogger("spaclone")

#: Width (inches) of the reference slide; fonts, dot sizes and line widths scale with ``figure width / _REF_W``.
_REF_W = 13.333

#: House styles (:data:`spaclone.pl._colors.ATLAS_STYLES`). ``dark`` is the archive look (black slide); ``light``
#: the same encodings on white.
_STYLES: dict[str, dict[str, Any]] = ATLAS_STYLES

#: Sizes at the reference width: fonts (pt), dots (pt²), lines (pt).
_FS = {"label": 10.5, "gex": 10.0, "number": 9.5, "legend": 8.5, "legend_row": 9.5, "note": 11.0, "small": 8.0}
_DOT = {"small": 5.0, "big": 23.0, "edge": 0.55}
_LW = {"zoom": 2.8, "gex": 1.6, "top": 1.1, "tissue": 1.0, "box": 0.85, "gcb": 1.0}

_SWITCHED = ("IgG", "IgA", "IgE")
_UNSWITCHED = ("IgM", "IgD")
_GENE_CLASS = {"IGHM": "IgM", "IGHD": "IgD", "IGHG": "IgG", "IGHA": "IgA", "IGHE": "IgE"}
_CHANNELS = ("csr", "shm", "gex")


# ----------------------------------------------------------------------------------------------- small utilities


def _style(style: str | None) -> dict[str, Any]:
    if style not in (None, "dark", "light"):
        raise ValueError(f"style must be 'dark', 'light' or None, got {style!r}")
    return _STYLES["dark" if style == "dark" else "light"]


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or mapping entry ``name`` of a result object (``default`` when absent or None)."""
    if obj is None:
        return default
    val = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    return default if val is None else val


def _cmap(colors: Sequence[str], name: str) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def _stroke(pal: dict[str, Any], width: float) -> list[Any]:
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground=pal["stroke"])]


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of ``values`` with every invalid pixel set to its nearest valid one.

    Keeps colours uniform at mask edges when the image is interpolated.
    """
    if valid.all() or not valid.any():
        return np.where(valid, values, 0.0)
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def _as_bool_array(obj: Any, grid: Any) -> np.ndarray | None:
    """Boolean array on ``grid`` from a Raster or an array of the grid's shape (None otherwise)."""
    if obj is None:
        return None
    vals = getattr(obj, "values", obj)
    if isinstance(vals, pd.DataFrame | pd.Series):
        return None
    arr = np.asarray(vals)
    if arr.ndim != 2 or (grid is not None and arr.shape != tuple(grid.shape)):
        return None
    return arr.astype(bool)


def _labels_array(hs: Any, grid: Any) -> np.ndarray | None:
    """Integer hotspot labels of a HotspotResult (``.labels`` Raster) restricted to hot hotspots."""
    lab = _field(hs, "labels")
    if lab is None:
        return None
    arr = np.asarray(getattr(lab, "values", lab))
    if arr.ndim != 2 or (grid is not None and arr.shape != tuple(grid.shape)):
        return None
    arr = arr.astype(np.int64)
    tab = _field(hs, "table")
    if isinstance(tab, pd.DataFrame) and len(tab) and "direction" in tab.columns and "hotspot" in tab.columns:
        cold = tab.hotspot[tab.direction.astype(str) != "hot"].to_numpy()
        if len(cold):
            arr = np.where(np.isin(arr, cold), 0, arr)
    return arr


def _ids(value: Any) -> list[int]:
    """Hotspot ids of a region cell: a list-like, a scalar or a string such as ``"1,3"`` / ``"[np.int32(4)]"``."""
    if value is None:
        return []
    if isinstance(value, str):
        import re

        return [int(v) for v in re.findall(r"-?\d+", re.sub(r"int\d+\(", "(", value))]
    if isinstance(value, float) and not np.isfinite(value):
        return []
    if np.ndim(value) == 0:
        try:
            return [int(value)]
        except (TypeError, ValueError):
            return []
    out = []
    for v in np.asarray(value, dtype=object).ravel():
        out.extend(_ids(v))
    return out


def _top_set(label: str) -> str:
    """The first cell-type set of a GEX label: ``"plasmablast/ASC / plasma cell (mixed)"`` -> ``"plasmablast/ASC"``."""
    lab = label.strip()
    if lab.endswith("(mixed)"):
        lab = lab[: -len("(mixed)")].strip()
    return lab.split(" / ")[0].strip()


def _short_label(label: str) -> str:
    """Display form of a GEX label: each set by its last name (``"plasmablast/ASC"`` -> ``"ASC"``).

    This is how the archive figure names the sets; a mixed label keeps both sets and its ``(mixed)`` mark.
    """
    lab = label.strip()
    mixed = lab.endswith("(mixed)")
    if mixed:
        lab = lab[: -len("(mixed)")].strip()
    parts = [p.split("/")[-1].strip() for p in lab.split(" / ")]
    return " / ".join(parts) + (" (mixed)" if mixed else "")


def _fmt_p(value: Any, n_perm: Any) -> str | None:
    """A Monte-Carlo p-value with 3 significant digits; at the smallest attainable value, ``"< 1/n_perm"``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    try:
        n = int(n_perm)
    except (TypeError, ValueError):
        n = 0
    if n > 0 and v <= 1.0 / (n + 1) * (1 + 1e-9):
        return f"< {1.0 / n:.3g}"
    return f"{v:.3g}"


def _pct(value: Any) -> str | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return f"{v:.0%}" if np.isfinite(v) else None


# ------------------------------------------------------------------------------------------------ display frame


class _Frame:
    """Display coordinates: the section rotated about its centre so that its long axis is horizontal.

    ``angle`` is the counter-clockwise rotation (radians) applied to the original µm coordinates; ``rotate=True``
    takes it from the principal axis of the tissue pixels (sign fixed so the axis points to +x) when the section is
    elongated (variance along the principal axis at least ``min_ratio`` times that across it; round sections keep
    their orientation), ``False`` / ``0`` keeps the original coordinates, a number is an angle in degrees.
    """

    min_ratio = 1.5

    def __init__(self, grid: Any, tissue: np.ndarray | None, rotate: bool | float | None) -> None:
        gx, gy = grid.centres()
        if tissue is not None and tissue.any():
            ii, jj = np.nonzero(tissue)
        else:
            ii, jj = np.nonzero(np.ones(grid.shape, dtype=bool))
        pts = np.column_stack([gx[ii], gy[jj]])
        self.grid = grid
        angle = 0.0
        if rotate is True:
            if len(pts) >= 3:
                cov = np.cov((pts - pts.mean(0)).T)
                val, vec = np.linalg.eigh(cov)
                v = vec[:, -1]
                if v[0] < 0 or (v[0] == 0 and v[1] < 0):
                    v = -v
                if val[-1] >= self.min_ratio * max(val[0], 1e-12):
                    angle = float(-np.arctan2(v[1], v[0]))
        elif rotate not in (False, None):
            angle = float(np.radians(float(rotate)))
        self.angle = angle
        self.rotated = abs(angle) > 1e-12
        self.centre = pts.mean(0) if self.rotated else np.zeros(2)
        c, s = np.cos(angle), np.sin(angle)
        self.R = np.array([[c, -s], [s, c]])

    def __call__(self, x: Any, y: Any) -> np.ndarray:
        """``(n, 2)`` display coordinates of points."""
        p = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
        if not self.rotated:
            return p
        return (p - self.centre) @ self.R.T

    def points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return self(xy[:, 0], xy[:, 1])

    def affine(self) -> Any:
        """Matplotlib transform from original µm to display µm."""
        from matplotlib.transforms import Affine2D

        if not self.rotated:
            return Affine2D()
        t = -self.R @ self.centre
        return Affine2D(np.array([[self.R[0, 0], self.R[0, 1], t[0]], [self.R[1, 0], self.R[1, 1], t[1]], [0, 0, 1]]))

    def mask_points(self, mask: np.ndarray, *, corners: bool = True) -> np.ndarray:
        """Display coordinates of the pixels of a mask (their corners with ``corners=True``)."""
        gx, gy = self.grid.centres()
        ii, jj = np.nonzero(mask)
        if not len(ii):
            return np.zeros((0, 2))
        x, y = gx[ii], gy[jj]
        if corners:
            h = self.grid.pixel_um / 2
            x = np.concatenate([x - h, x + h, x - h, x + h])
            y = np.concatenate([y - h, y - h, y + h, y + h])
        return self(x, y)

    def section_box(self, tissue: np.ndarray, pad_um: float = 20.0) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the section in display coordinates, plus ``pad_um``.

        Every tissue component holding at least 0.5% of the tissue area counts (specks of debris are left out).
        """
        m = tissue if tissue.any() else np.ones_like(tissue)
        lab, n = ndi.label(m)
        if n > 1:
            area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            keep = 1 + np.flatnonzero(area >= 0.005 * area.sum())
            m = np.isin(lab, keep)
        p = self.mask_points(m, corners=True)
        lo, hi = p.min(0) - pad_um, p.max(0) + pad_um
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _set_view(ax: Any, box: tuple[float, float, float, float], invert_y: bool) -> None:
    ax.set_xlim(box[0], box[1])
    ax.set_ylim((box[3], box[2]) if invert_y else (box[2], box[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


# -------------------------------------------------------------------------------------- vector shapes on the grid


def _grid_axes(grid: Any, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates of the grid, extended by ``pad`` pixels on every side."""
    gx, gy = grid.centres()
    if pad:
        st = grid.pixel_um
        ext = st * np.arange(1, pad + 1)
        gx = np.concatenate([gx[0] - ext[::-1], gx, gx[-1] + ext])
        gy = np.concatenate([gy[0] - ext[::-1], gy, gy[-1] + ext])
    return gx, gy


def _contour_lines(field: np.ndarray, grid: Any, level: float = 0.5, pad: int = 0) -> list[np.ndarray]:
    """Contour lines (original µm) at ``level`` of a field on the grid (padded by ``pad`` pixels per side)."""
    import contourpy

    if not np.isfinite(field).any() or np.nanmax(field) < level:
        return []
    gx, gy = _grid_axes(grid, pad)
    gen = contourpy.contour_generator(x=gx, y=gy, z=np.asarray(field, dtype=float).T, line_type="Separate")
    return [np.asarray(v) for v in gen.lines(level) if len(v) >= 2]


def _filled_paths(
    field: np.ndarray, grid: Any, level: float = 0.5, pad: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Filled regions (points in original µm, path codes) where a field on the (padded) grid exceeds ``level``."""
    import contourpy

    if not np.isfinite(field).any() or np.nanmax(field) < level:
        return []
    gx, gy = _grid_axes(grid, pad)
    gen = contourpy.contour_generator(x=gx, y=gy, z=np.asarray(field, dtype=float).T, fill_type="OuterCode")
    pts, codes = gen.filled(level, np.inf)
    return [(np.asarray(p), np.asarray(c)) for p, c in zip(pts, codes, strict=True) if len(p)]


def _mask_field(mask: np.ndarray, smooth: float) -> tuple[np.ndarray, int]:
    """A mask zero-padded and smoothed: shapes that reach the grid edge still close at level 0.5."""
    pad = int(np.ceil(4 * smooth)) + 1
    return ndi.gaussian_filter(np.pad(np.asarray(mask, dtype=float), pad), smooth), pad


def _opaque_rim(mask: np.ndarray) -> np.ndarray:
    """Alpha of a map image: opaque on the mask and one pixel around it.

    The clip path (the smoothed mask outline), not the interpolated alpha, then draws the edge.
    """
    return ndi.binary_dilation(np.asarray(mask, dtype=bool), iterations=1).astype(float)


def _mask_fill(mask: np.ndarray, grid: Any, smooth: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Filled outline of a mask (Gaussian ``smooth`` px, level 0.5)."""
    if mask is None or not np.asarray(mask).any():
        return []
    field, pad = _mask_field(mask, smooth)
    return _filled_paths(field, grid, pad=pad)


def _ring_area(p: np.ndarray) -> float:
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))))


def _mask_lines(mask: np.ndarray, grid: Any, smooth: float) -> list[np.ndarray]:
    """Outline (original µm) of a boolean mask, drawn as the archive did: Gaussian ``smooth`` px, level 0.5."""
    if mask is None or not mask.any():
        return []
    field, pad = _mask_field(mask, smooth)
    return _contour_lines(field, grid, pad=pad)


def _draw_lines(ax: Any, frame: _Frame, lines: list[np.ndarray], color: str, lw: float, z: float, **kw: Any) -> None:
    from matplotlib.collections import LineCollection

    if not lines:
        return
    segs = [frame.points(v) for v in lines]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=z, capstyle="round", **kw))


def _draw_fill(ax: Any, frame: _Frame, paths: list[tuple[np.ndarray, np.ndarray]], color: str, z: float) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    for p, c in paths:
        ax.add_patch(PathPatch(Path(frame.points(p), c), facecolor=color, edgecolor="none", lw=0, zorder=z))


def _draw_image(
    ax: Any,
    frame: _Frame,
    grid: Any,
    rgba: np.ndarray,
    z: float,
    clip: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Draw an RGBA array on the grid (indexed ``[i, j]``), rotated into display coordinates.

    With ``clip`` the image is cut to the smoothed tissue outline (no pixel staircase at the tissue edge).
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    im = ax.imshow(
        np.ascontiguousarray(rgba.transpose(1, 0, 2)),
        origin="lower",
        extent=grid.extent,
        interpolation="bilinear",
        transform=frame.affine() + ax.transData,
        zorder=z,
    )
    if clip:
        path = Path.make_compound_path(*[Path(frame.points(p), c) for p, c in clip])
        patch = PathPatch(path, facecolor="none", edgecolor="none")
        patch.set_transform(ax.transData)
        im.set_clip_path(patch)


# ------------------------------------------------------------------------------------------------ result access


def _share_map(hs: Any) -> Any:
    """The ShareMap of a HotspotResult (or the object itself when it is a ShareMap)."""
    if hs is None:
        return None
    sm = _field(hs, "map")
    if sm is not None:
        return sm
    return hs if hasattr(hs, "share") and hasattr(hs, "grid") else None


def _loci_xy(loci: pd.DataFrame, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    for xk, yk in ((x_key, y_key), ("x_um", "y_um"), ("x", "y")):
        if xk in loci.columns and yk in loci.columns:
            return loci[xk].to_numpy(dtype=float), loci[yk].to_numpy(dtype=float)
    raise KeyError(f"loci table has no coordinate columns ({x_key!r}, {y_key!r})")


def _nullable_bool(s: pd.Series) -> np.ndarray:
    """``1`` True, ``0`` False, ``-1`` missing."""
    v = pd.array(s, dtype="boolean")
    out = np.full(len(v), -1, dtype=np.int8)
    known = ~np.asarray(v.isna())
    out[known] = np.asarray(v[known], dtype=bool).astype(np.int8)
    return out


def _isotype_column(loci: pd.DataFrame) -> pd.Series | None:
    for c in ("isotype_class", "isotype", "iso_main", "iso"):
        if c in loci.columns:
            s = loci[c].astype("string")
            return s.replace(_GENE_CLASS).astype("string")
    return None


class _AtlasData:
    """Everything the panels need, read from a workflow result once (shapes and images built on demand)."""

    def __init__(
        self,
        result: Any,
        *,
        style: str | None,
        rotate: bool | float | None,
        invert_y: bool,
        x_key: str = S.X,
        y_key: str = S.Y,
        score_mask: Any = None,
    ) -> None:
        self.result = result
        self.pal = _style(style)
        self.invert_y = invert_y
        self.score_mask = score_mask
        self.csr = _field(result, "csr")
        self.shm = _field(result, "shm")
        self.sm_csr = _share_map(self.csr)
        self.sm_shm = _share_map(self.shm)
        grid = _field(result, "grid")
        if grid is None:
            grid = next((sm.grid for sm in (self.sm_csr, self.sm_shm) if sm is not None), None)
        if grid is None:
            raise ValueError("the result has no grid: hotspot_atlas needs result.grid or a share map in result.csr")
        self.grid = grid
        self.tissue = self._tissue()
        self.frame = _Frame(grid, self.tissue, rotate)
        self.csr_labels = _labels_array(self.csr, grid)
        self.shm_labels = _labels_array(self.shm, grid)
        loci = _field(result, "loci")
        self.loci = loci if isinstance(loci, pd.DataFrame) else None
        if self.loci is not None and len(self.loci):
            x, y = _loci_xy(self.loci, x_key, y_key)
            self.loci_xy = self.frame(x, y)
        else:
            self.loci_xy = np.zeros((0, 2))
        self.params = _field(result, "params", {}) or {}
        self._cache: dict[str, Any] = {}
        self.regions = self._regions()

    # ---- tissue, maps, outlines
    def _tissue(self) -> np.ndarray:
        t = _as_bool_array(_field(self.result, "tissue"), self.grid)
        if t is not None and t.any():
            return t
        for sm in (self.sm_csr, self.sm_shm):
            if sm is not None:
                return np.asarray(sm.eligible, dtype=bool) | (np.nan_to_num(np.asarray(sm.n_eff, dtype=float)) > 0)
        return np.ones(self.grid.shape, dtype=bool)

    def tissue_fill(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if "tissue_fill" not in self._cache:
            self._cache["tissue_fill"] = _mask_fill(self.tissue, self.grid, 1.5)
        return self._cache["tissue_fill"]

    def tissue_edge(self) -> list[np.ndarray]:
        if "tissue_edge" not in self._cache:
            self._cache["tissue_edge"] = _mask_lines(self.tissue, self.grid, 1.5)
        return self._cache["tissue_edge"]

    def map_image(self, which: str) -> np.ndarray | None:
        key = f"img_{which}"
        if key in self._cache:
            return self._cache[key]
        sm = self.sm_csr if which == "csr" else self.sm_shm
        if sm is None:
            self._cache[key] = None
            return None
        from matplotlib.colors import Normalize

        vals = np.asarray(sm.share, dtype=float)
        elig = np.asarray(sm.eligible, dtype=bool) & self.tissue & np.isfinite(vals)
        ramp = self.pal["csr_ramp" if which == "csr" else "shm_ramp"]
        cmap = _cmap(ramp, f"spaclone_atlas_{which}")
        fin = vals[elig]
        lo, hi = np.quantile(fin, [0.02, 0.995]) if len(fin) else (0.0, 1.0)
        norm = Normalize(float(lo), float(hi) if hi > lo else float(lo) + 1e-9)
        rgba = cmap(norm(_nearest_fill(vals, elig)))
        rgba[..., 3] = _opaque_rim(elig)
        self._cache[key] = rgba
        self._cache[f"norm_{which}"] = norm
        self._cache[f"clip_{which}"] = _mask_fill(elig, self.grid, 0.7)
        return rgba

    def map_clip(self, which: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Smoothed outline of the eligible pixels of a map (the edge the map is drawn to)."""
        self.map_image(which)
        return self._cache.get(f"clip_{which}") or self.tissue_fill()

    def hotspot_lines(self, which: str) -> list[np.ndarray]:
        key = f"hot_{which}"
        if key not in self._cache:
            lab = self.csr_labels if which == "csr" else self.shm_labels
            self._cache[key] = _mask_lines(lab > 0, self.grid, 0.7) if lab is not None else []
        return self._cache[key]

    def region_lines(self, ids: Sequence[Any] | None = None) -> list[np.ndarray]:
        sel = list(self.regions.index if ids is None else ids)
        key = "reg_" + ",".join(map(str, sel))
        if key not in self._cache:
            m = np.zeros(self.grid.shape, dtype=bool)
            for r in sel:
                m |= self.region_mask(r)
            self._cache[key] = _mask_lines(m, self.grid, 0.7)
        return self._cache[key]

    # ---- regions
    def _regions(self) -> pd.DataFrame:
        tab = _field(self.result, "regions")
        cols = ["order_x", "cx", "cy"]
        if not isinstance(tab, pd.DataFrame) or not len(tab):
            self._masks: dict[Any, np.ndarray] = {}
            return pd.DataFrame(columns=cols)
        tab = tab.copy()
        if "region_id" in tab.columns:
            tab = tab.set_index("region_id", drop=False)
        reg_lab = _field(self.result, "region_labels")
        reg_arr = None
        if reg_lab is not None:
            arr = np.asarray(getattr(reg_lab, "values", reg_lab))
            if arr.shape == tuple(self.grid.shape):
                reg_arr = arr
        masks: dict[Any, np.ndarray] = {}
        for rid, row in tab.iterrows():
            m = None
            if reg_arr is not None:
                try:
                    m = reg_arr == rid
                except TypeError:
                    m = None
                if m is not None and not m.any():
                    m = None
            if m is None and self.csr_labels is not None:
                ids = _ids(row.get("csr_hotspots"))
                if ids:
                    m = np.isin(self.csr_labels, ids)
            if (m is None or not m.any()) and self.shm_labels is not None:
                ids = _ids(row.get("shm_hotspots"))
                if ids:
                    m = np.isin(self.shm_labels, ids)
            if m is None or not m.any():
                m = self._disc_mask(row)
            masks[rid] = m
        self._masks = masks
        cxy = []
        for rid in tab.index:
            p = self.frame.mask_points(masks[rid], corners=False)
            cxy.append(p.mean(0) if len(p) else (np.nan, np.nan))
        cxy = np.asarray(cxy, dtype=float).reshape(-1, 2)
        tab["cx"], tab["cy"] = cxy[:, 0], cxy[:, 1]
        tab["order_x"] = tab["cx"]
        tab = tab.sort_values("order_x", kind="stable")
        if "region_id" not in tab.columns:  # no ids: number the regions 1..n along the section
            new = pd.Index(range(1, len(tab) + 1), name="region_id")
            self._masks = {k: self._masks[old] for k, old in zip(new, tab.index, strict=True)}
            tab.index = new
            tab["region_id"] = new
        return tab

    def _disc_mask(self, row: pd.Series) -> np.ndarray:
        """Fallback region shape: a disc of the region's area around its centre."""
        x, y = row.get("x_um", np.nan), row.get("y_um", np.nan)
        area = row.get("area_um2", np.nan)
        m = np.zeros(self.grid.shape, dtype=bool)
        if not (np.isfinite(x) and np.isfinite(y)):
            return m
        r = math.sqrt(area / math.pi) if np.isfinite(area) and area > 0 else 2 * self.grid.pixel_um
        gx, gy = self.grid.centres()
        return (gx[:, None] - x) ** 2 + (gy[None, :] - y) ** 2 <= r**2

    def region_mask(self, rid: Any) -> np.ndarray:
        return self._masks[rid]

    def window(self, rid: Any, aspect: float, pad_um: float, min_height_um: float) -> tuple[float, float, float, float]:
        """Display window around a region, widened to the panel aspect (width / height).

        The window holds the region and the SHM hotspots it touches, plus ``pad_um`` on every side.
        """
        m = self.region_mask(rid).copy()
        if self.shm_labels is not None:
            touched = np.setdiff1d(np.unique(self.shm_labels[ndi.binary_dilation(m, iterations=1)]), [0])
            if len(touched):
                m |= np.isin(self.shm_labels, touched)
        p = self.frame.mask_points(m)
        if not len(p):
            cx, cy = self.regions.loc[rid, ["cx", "cy"]].to_numpy(dtype=float)
            p = np.array([[cx, cy]])
        lo, hi = p.min(0) - pad_um, p.max(0) + pad_um
        (cx, cy), (w, h) = (lo + hi) / 2, hi - lo
        h = max(h, min_height_um)
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    # ---- loci
    def loci_in(self, box: tuple[float, float, float, float]) -> np.ndarray:
        p = self.loci_xy
        return (p[:, 0] >= box[0]) & (p[:, 0] <= box[1]) & (p[:, 1] >= box[2]) & (p[:, 1] <= box[3])

    def isotype_dots(self) -> tuple[np.ndarray, np.ndarray]:
        """Per locus: 1 switched, 0 unswitched, -1 no isotype; and the switched class used as its colour."""
        if "iso_dots" in self._cache:
            return self._cache["iso_dots"]
        lo = self.loci
        n = 0 if lo is None else len(lo)
        state = np.full(n, -1, dtype=np.int8)
        cls = np.full(n, "", dtype=object)
        if n:
            iso = _isotype_column(lo)
            if "switched" in lo.columns:
                state = _nullable_bool(lo["switched"])
            elif iso is not None:
                isv = iso.fillna("unknown").to_numpy(dtype=object)
                state = np.where(np.isin(isv, _SWITCHED), 1, np.where(np.isin(isv, _UNSWITCHED), 0, -1)).astype(np.int8)
            if iso is not None:
                isv = iso.fillna("").to_numpy(dtype=object)
                cls = np.where(np.isin(isv, _SWITCHED), isv, "").astype(object)
            sw_missing = (state == 1) & (cls == "")
            if sw_missing.any():  # switched by share but main class unswitched: colour by the larger switched count
                g = lo["IgG"].to_numpy(dtype=float) if "IgG" in lo.columns else np.zeros(n)
                a = lo["IgA"].to_numpy(dtype=float) if "IgA" in lo.columns else np.zeros(n)
                if "IGHG" in lo.columns and "IGHA" in lo.columns:
                    g, a = lo["IGHG"].to_numpy(dtype=float), lo["IGHA"].to_numpy(dtype=float)
                cls[sw_missing] = np.where(a[sw_missing] > g[sw_missing], "IgA", "IgG")
        self._cache["iso_dots"] = (state, cls)
        return state, cls

    def mutation_dots(self) -> np.ndarray:
        """Per locus: 1 mutated, 0 evaluable and not mutated, -1 not evaluable."""
        if "mut_dots" in self._cache:
            return self._cache["mut_dots"]
        lo = self.loci
        n = 0 if lo is None else len(lo)
        out = np.full(n, -1, dtype=np.int8)
        if n:
            mcol = next((c for c in ("mutated", "shm_mutated") if c in lo.columns), None)
            if mcol is not None:
                out = _nullable_bool(lo[mcol])
            ecol = next((c for c in ("evaluable", "mut_eval", "shm_evaluable") if c in lo.columns), None)
            if ecol is not None:
                ev = _nullable_bool(lo[ecol]) == 1
                out = np.where(ev, np.maximum(out, 0), -1).astype(np.int8)
        self._cache["mut_dots"] = out
        return out

    def clone_p_line(self, rid: Any) -> str | None:
        """The region's short p line, e.g. ``"clone-level p: CSR = 0.018 · SHM = 0.41"``.

        From ``regions.csr_clone_p`` / ``shm_clone_p``; None when the region has no finite clone-level p-value.
        """
        row = self.regions.loc[rid]
        p = self.params if isinstance(self.params, Mapping) else {}
        n_perm = p.get("n_perm")
        parts = []
        for name, col in (("CSR", "csr_clone_p"), ("SHM", "shm_clone_p")):
            if col in self.regions.columns:
                txt = _fmt_p(row.get(col), n_perm)
                if txt is not None:
                    parts.append(f"{name} {txt}" if txt.startswith("<") else f"{name} = {txt}")
        return "clone-level p: " + " \u00b7 ".join(parts) if parts else None

    def mutation_label(self) -> str:
        p = self.params if isinstance(self.params, Mapping) else {}
        for k in ("min_excess", "min_mutations", "mutated_min_excess", "shm_min_excess"):
            v = p.get(k)
            if v is not None and np.isfinite(float(v)):
                what = "excess V mutations" if "excess" in k else "V mutations"
                return f"mutated (≥ {float(v):g} {what})"
        return "mutated"

    # ---- transcriptome
    def gex_positions(self) -> pd.DataFrame | None:
        if "gex_pos" in self._cache:
            return self._cache["gex_pos"]
        pos = None
        for name in ("gex_units", "units"):  # units: per GEX unit of hotspot_workflow (index = obs_names)
            units = _field(self.result, name)
            if pos is None and isinstance(units, pd.DataFrame) and {"x_um", "y_um"} <= set(units.columns):
                pos = units[["x_um", "y_um"]]
        gex = _field(self.result, "gex")
        if pos is None and gex is not None and hasattr(gex, "obs_names"):
            if "spatial" in getattr(gex, "obsm", {}):
                xy = np.asarray(gex.obsm["spatial"], dtype=float)[:, :2]
                pos = pd.DataFrame(xy, index=pd.Index(gex.obs_names), columns=["x_um", "y_um"])
            elif {"x_um", "y_um"} <= set(gex.obs.columns):
                pos = gex.obs[["x_um", "y_um"]].astype(float)
        self._cache["gex_pos"] = pos
        return pos

    def module_key(self, label: Any) -> Any:
        """Module map of a region's GEX label (the top set of a mixed label ``"<top> / <second> (mixed)"``)."""
        mm = _field(self.result, "module_maps")
        if mm is None or label is None or (isinstance(label, float) and not np.isfinite(label)):
            return None
        keys = list(mm.columns) if isinstance(mm, pd.DataFrame) else list(mm.keys())
        lab = _top_set(str(label))
        for k in keys:
            if str(k) == lab:
                return k
        low = lab.lower()
        for k in keys:
            if str(k).lower() == low:
                return k
        for k in keys:
            parts = [p.strip().lower() for p in str(k).replace("|", "/").split("/")]
            if low in parts:
                return k
        return None

    def module_image(self, key: Any) -> tuple[np.ndarray | None, bool]:
        """RGBA grid image of a module score map and whether any unit was left without a score."""
        ck = f"module_{key}"
        if ck in self._cache:
            return self._cache[ck]
        out: tuple[np.ndarray | None, bool] = (None, False)
        mm = _field(self.result, "module_maps")
        val = None
        if isinstance(mm, pd.DataFrame):
            val = mm
            col = key
        else:
            val = mm.get(key) if isinstance(mm, Mapping) else None
            col = None
        arr = _as_bool_array(val, self.grid) if not isinstance(val, pd.DataFrame | pd.Series) else None
        if arr is not None:  # a raster of scores on the grid
            out = self._score_rgba_grid(np.asarray(getattr(val, "values", val), dtype=float))
        else:
            pts = self._score_points(val, col)
            if pts is not None:
                out = self._score_rgba_points(*pts)
        self._cache[ck] = out
        self._cache[f"norm_{ck}"] = self._cache.pop("last_score_norm", None) if out[0] is not None else None
        return out

    def _score_points(self, val: Any, col: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if isinstance(val, pd.DataFrame):
            xcol = next((c for c in ("x_um", "x") if c in val.columns), None)
            ycol = next((c for c in ("y_um", "y") if c in val.columns), None)
            if col is None:
                num = [c for c in val.columns if c not in (xcol, ycol) and pd.api.types.is_numeric_dtype(val[c])]
                col = "score" if "score" in val.columns else (num[0] if num else None)
            if col is None:
                return None
            if xcol is not None and ycol is not None:
                return (
                    val[xcol].to_numpy(dtype=float),
                    val[ycol].to_numpy(dtype=float),
                    pd.to_numeric(val[col], errors="coerce").to_numpy(dtype=float),
                )
            val = val[col]
        if isinstance(val, pd.Series):
            pos = self.gex_positions()
            if pos is None:
                logger.warning("hotspot_atlas: module map without unit positions (result.gex_units or result.gex)")
                return None
            v = pd.to_numeric(val, errors="coerce").reindex(pos.index).to_numpy(dtype=float)
            keep = self._mask_for(pos.index)
            if keep is not None:
                v = np.where(keep, v, np.nan)
            return pos.x_um.to_numpy(dtype=float), pos.y_um.to_numpy(dtype=float), v
        return None

    def _mask_for(self, index: pd.Index) -> np.ndarray | None:
        """``score_mask`` as booleans aligned with ``index`` (None: no mask)."""
        m = self.score_mask
        if m is None:
            m = _field(self.result, "score_mask")  # set by hotspot_workflow(score_units=...)
            if m is None:
                return None
        if isinstance(m, str):
            units = _field(self.result, "units")
            if units is None:
                units = _field(self.result, "gex_units")
            gex = _field(self.result, "gex")
            if isinstance(units, pd.DataFrame) and m in units.columns:
                m = units[m]
            elif gex is not None and hasattr(gex, "obs") and m in gex.obs.columns:
                m = gex.obs[m]
            else:
                raise KeyError(f"score_mask: column {m!r} not in result.units (or result.gex.obs)")
        if isinstance(m, pd.Series):
            return m.reindex(index).fillna(False).to_numpy(dtype=bool)
        arr = np.asarray(m, dtype=bool)
        if arr.shape != (len(index),):
            raise ValueError("score_mask: a column name, a boolean Series indexed by GEX unit, or one value per unit")
        return arr

    def _score_rgba_points(self, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """RGBA image of per-unit scores, following the archive's T8e recipe.

        Units go on the grid, scores are display-smoothed by a normalised Gaussian convolution over scored units
        only, and units without a score are neutral grey with softened edges.
        """
        from scipy.spatial import cKDTree

        g = self.grid
        i, j, inside = g.index(x, y)
        fin = np.isfinite(v) & inside
        if not fin.any():
            return None, False
        occ = np.zeros(g.shape)
        np.add.at(occ, (i[inside], j[inside]), 1.0)
        ssum = np.zeros(g.shape)
        wsum = np.zeros(g.shape)
        np.add.at(ssum, (i[fin], j[fin]), v[fin])
        np.add.at(wsum, (i[fin], j[fin]), 1.0)
        # unit spacing decides the smoothing: one pixel for bin-sized units, half the spacing for sparse spots
        sub = np.column_stack([x[inside], y[inside]])
        if len(sub) > 20000:
            sub = sub[np.random.default_rng(0).choice(len(sub), 20000, replace=False)]
        spacing = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1])) if len(sub) > 1 else g.pixel_um
        ratio = spacing / g.pixel_um
        sigma = max(1.0, 0.5 * ratio)
        cover = occ > 0
        if ratio > 1.5:
            cover = ndi.binary_dilation(cover, iterations=int(math.ceil(ratio / 2)))
        cover &= self.tissue | (occ > 0)
        mean = np.where(wsum > 0, ssum / np.maximum(wsum, 1e-12), 0.0)
        has = (wsum > 0).astype(float)
        den = ndi.gaussian_filter(has, sigma)
        sm = np.where(den > 1e-3, ndi.gaussian_filter(mean * has, sigma) / np.maximum(den, 1e-12), np.nan)
        ref = sm[has > 0]
        ref = ref[np.isfinite(ref)]
        from matplotlib.colors import Normalize, to_rgba

        lo, hi = np.quantile(ref, [0.02, 0.995]) if len(ref) else (0.0, 1.0)
        norm = Normalize(float(lo), float(hi) if hi > lo else float(lo) + 1e-9)
        self._cache["last_score_norm"] = norm
        cmap = _cmap(self.pal["score_ramp"], "spaclone_atlas_score")
        col = cmap(norm(np.nan_to_num(sm, nan=float(lo))))[..., :3]
        a = np.where(np.isfinite(sm), np.clip(ndi.gaussian_filter(has, 0.6 * sigma) * 1.25, 0, 1), 0.0)[..., None]
        grey = np.array(to_rgba(self.pal["unscored"])[:3])
        rgba = np.zeros((*g.shape, 4))
        rgba[..., :3] = a * col + (1 - a) * grey
        rgba[..., 3] = cover.astype(float)
        unscored = bool((cover & (wsum == 0)).sum() > 0.01 * cover.sum())
        return rgba, unscored

    def _score_rgba_grid(self, vals: np.ndarray) -> tuple[np.ndarray | None, bool]:
        from matplotlib.colors import Normalize, to_rgba

        fin = np.isfinite(vals) & self.tissue
        if not fin.any():
            return None, False
        lo, hi = np.quantile(vals[fin], [0.02, 0.995])
        norm = Normalize(float(lo), float(hi) if hi > lo else float(lo) + 1e-9)
        self._cache["last_score_norm"] = norm
        cmap = _cmap(self.pal["score_ramp"], "spaclone_atlas_score")
        rgba = cmap(norm(_nearest_fill(vals, fin)))
        grey = np.array(to_rgba(self.pal["unscored"]))
        rgba[~fin & self.tissue] = grey
        rgba[..., 3] = self.tissue.astype(float)
        return rgba, bool((self.tissue & ~fin).any())

    def unscored(self, rid: Any, box: tuple[float, float, float, float] | None = None) -> bool:
        """Whether the GEX zoom of a region shows units without a score (they get the grey legend entry)."""
        key = self.module_key(self.regions.loc[rid].get("gex_label"))
        if key is None:
            return False
        img, any_unscored = self.module_image(key)
        if img is None or not any_unscored:
            return False
        if box is None:
            return True
        from matplotlib.colors import to_rgba

        grey = np.array(to_rgba(self.pal["unscored"])[:3])
        sel = self._box_pixels(box)
        px = img[sel]
        return bool(((px[:, 3] > 0.5) & (np.abs(px[:, :3] - grey).sum(1) < 0.02)).any())

    def _box_pixels(self, box: tuple[float, float, float, float]) -> np.ndarray:
        """Boolean grid mask of the pixels whose centres fall in a display window."""
        gx, gy = self.grid.centres()
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        p = self.frame(X.ravel(), Y.ravel())
        inside = (p[:, 0] >= box[0]) & (p[:, 0] <= box[1]) & (p[:, 1] >= box[2]) & (p[:, 1] <= box[3])
        return inside.reshape(self.grid.shape)

    def gc_visible(self, boxes: Sequence[tuple[float, float, float, float]]) -> bool:
        """Whether any GEX germinal-centre B outline crosses one of the windows."""
        for v in self.gc_lines():
            p = self.frame.points(v)
            for b in boxes:
                if ((p[:, 0] >= b[0]) & (p[:, 0] <= b[1]) & (p[:, 1] >= b[2]) & (p[:, 1] <= b[3])).any():
                    return True
        return False

    def gc_lines(self) -> list[np.ndarray]:
        """Germinal-centre B outlines found from gene expression (original µm)."""
        if "gc_lines" in self._cache:
            return self._cache["gc_lines"]
        gc = _field(self.result, "gc_gex")
        lines: list[np.ndarray] = []
        arr = _as_bool_array(gc, self.grid) if not isinstance(gc, pd.DataFrame | pd.Series) else None
        if arr is not None:
            lines = _mask_lines(arr, self.grid, 1.0)
        elif isinstance(gc, pd.DataFrame) and len(gc):
            xcol = next((c for c in ("x_um", "x") if c in gc.columns), None)
            ycol = next((c for c in ("y_um", "y") if c in gc.columns), None)
            flag = next((c for c in ("gc_b", "gcb", "gc", "flag", "is_gc", "gc_gex") if c in gc.columns), None)
            if xcol and ycol and "contour_id" in gc.columns:
                order = "vertex_order" if "vertex_order" in gc.columns else None
                for _, g in gc.groupby("contour_id", sort=False):
                    g = g.sort_values(order) if order else g
                    p = g[[xcol, ycol]].to_numpy(dtype=float)
                    if len(p) >= 3:
                        lines.append(p if np.allclose(p[0], p[-1]) else np.vstack([p, p[:1]]))
            elif xcol and ycol and flag:
                lines = self._flag_lines(
                    gc[xcol].to_numpy(dtype=float),
                    gc[ycol].to_numpy(dtype=float),
                    gc[flag].fillna(False).to_numpy(dtype=bool),
                )
        elif isinstance(gc, pd.Series) and len(gc):
            pos = self.gex_positions()
            if pos is not None:
                f = gc.reindex(pos.index).fillna(False).to_numpy(dtype=bool)
                lines = self._flag_lines(pos.x_um.to_numpy(dtype=float), pos.y_um.to_numpy(dtype=float), f)
        self._cache["gc_lines"] = lines
        return lines

    def _flag_lines(self, x: np.ndarray, y: np.ndarray, flag: np.ndarray, min_area: float = 2000.0) -> list:
        """Archive T8e rule: share of flagged units on the grid (Gaussian 1 px), level 0.5, rings >= 2000 µm²."""
        g = self.grid
        occ = g.rasterize(x, y)
        cnt = g.rasterize(x[flag], y[flag])
        den = ndi.gaussian_filter(occ, 1.0)
        fr = np.where(den >= 0.2, ndi.gaussian_filter(cnt, 1.0) / np.maximum(den, 1e-9), 0.0)
        return [v for v in _contour_lines(np.pad(fr, 2), g, pad=2) if _ring_area(v) >= min_area]

    def unit_word(self) -> str:
        """Word for a GEX unit: ``"bin"`` for bin-sized units, ``"spot"`` for sparse spots (from their spacing)."""
        pos = self.gex_positions()
        if pos is None or len(pos) < 2:
            return "bin"
        from scipy.spatial import cKDTree

        sub = pos.to_numpy(dtype=float)
        if len(sub) > 5000:
            sub = sub[np.random.default_rng(0).choice(len(sub), 5000, replace=False)]
        d = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
        return "spot" if d >= 40 else "bin"


# --------------------------------------------------------------------------------------------------- the panels


def _base(ax: Any, data: _AtlasData, box: tuple[float, float, float, float]) -> None:
    ax.set_facecolor(data.pal["bg"])
    _draw_fill(ax, data.frame, data.tissue_fill(), data.pal["tissue"], z=-1)
    _set_view(ax, box, data.invert_y)


def _mini_bar(ax: Any, norm: Any, ramp: Sequence[str], pal: dict[str, Any], s: float, *, pct: bool, where: str) -> None:
    """A discreet colour bar in a lower corner of a map tile: the ramp with its two end values only."""
    from matplotlib.cm import ScalarMappable

    box = [0.03, 0.05, 0.13, 0.022] if where == "left" else [0.84, 0.05, 0.13, 0.022]
    cax = ax.inset_axes(box, zorder=12)
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=_cmap(ramp, "spaclone_atlas_bar")), cax=cax,
                            orientation="horizontal")  # fmt: skip
    lo, hi = float(norm.vmin), float(norm.vmax)
    fmt = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.2g}")
    cb.set_ticks([lo, hi])
    cb.set_ticklabels([fmt(lo), fmt(hi)])
    cb.ax.tick_params(colors=pal["muted"], labelcolor=pal["muted"], labelsize=_FS["small"] * 0.8 * s, length=0,
                      pad=1.2 * s)  # fmt: skip
    cb.outline.set_visible(False)


def _note(ax: Any, text: str, pal: dict[str, Any], fs: float) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=pal["muted"], fontsize=fs)


def _label(ax: Any, text: str, pal: dict[str, Any], fs: float, s: float) -> Any:
    """Panel text of the archive: bold, top left, with a background-coloured stroke."""
    return ax.text(
        0.015,
        0.965,
        text,
        transform=ax.transAxes,
        color=pal["fg"],
        fontsize=fs,
        fontweight="bold",
        va="top",
        ha="left",
        path_effects=_stroke(pal, 3.5 * s),
        zorder=10,
        clip_on=False,
    )


def _overview_panel(
    ax: Any,
    data: _AtlasData,
    which: str,
    box: tuple[float, float, float, float],
    shown: Sequence[Any],
    windows: Mapping[Any, tuple[float, float, float, float]],
    s: float,
) -> list[Any]:
    """One whole-section map (``which`` = "csr" or "shm") with its hotspots and the numbered windows."""
    pal = data.pal
    _base(ax, data, box)
    img = data.map_image(which)
    texts: list[Any] = []
    if img is None:
        _note(ax, f"no {which.upper()} map in the result", pal, _FS["note"] * s)
    else:
        _draw_image(ax, data.frame, data.grid, img, z=0, clip=data.map_clip(which))
        norm = data._cache.get(f"norm_{which}")
        if norm is not None:
            _mini_bar(ax, norm, pal["csr_ramp" if which == "csr" else "shm_ramp"], pal, s, pct=True, where="left")
    _draw_lines(ax, data.frame, data.tissue_edge(), pal["tissue_edge"], _LW["tissue"] * s, 4)
    _draw_lines(ax, data.frame, data.hotspot_lines(which), pal[which], _LW["top"] * s, 5)
    texts += _draw_windows(ax, data, shown, windows, s)
    _set_view(ax, box, data.invert_y)
    return texts


def _draw_windows(
    ax: Any,
    data: _AtlasData,
    shown: Sequence[Any],
    windows: Mapping[Any, tuple[float, float, float, float]],
    s: float,
) -> list[Any]:
    """Window rectangles with their region numbers, placed so that numbers do not collide."""
    from matplotlib.patches import Rectangle

    pal = data.pal
    fs = _FS["number"] * s
    texts = []
    placed: list[tuple[float, float, float, float]] = []
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    y_lo, y_hi = sorted(ax.get_ylim())
    ax_w_in = ax.get_position().width * ax.figure.get_figwidth()
    um_per_pt = abs(x1 - x0) / (ax_w_in * 72) if ax_w_in > 0 else 1.0
    h_um = fs * 1.15 * um_per_pt
    gap = 2.0 * s * um_per_pt
    up = -1.0 if data.invert_y else 1.0  # data-y direction that points up on screen
    rects = [windows[r] for r in shown]
    for rid in shown:
        b = windows[rid]
        ax.add_patch(
            Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec=pal["frame"], lw=_LW["box"] * s, zorder=7)
        )
        label = str(rid)
        w_um = 0.62 * fs * len(label) * um_per_pt + 2 * um_per_pt
        top = b[2] if data.invert_y else b[3]
        bottom = b[3] if data.invert_y else b[2]
        cands = [
            (b[0], top + up * gap, "above"),
            (b[0], bottom - up * gap, "below"),
            (b[1] - w_um, top + up * gap, "above"),
            (b[1] - w_um, bottom - up * gap, "below"),
        ]
        choice = None
        for cx, cy, where in cands:
            yb = (cy, cy + up * h_um) if where == "above" else (cy - up * h_um, cy)
            r = (cx, cx + w_um, min(yb), max(yb))
            inside = r[2] >= y_lo and r[3] <= y_hi and r[0] >= min(x0, x1) and r[1] <= max(x0, x1)
            hit = any(_overlap(r, q) for q in placed) or any(
                _overlap(r, (q[0], q[1], q[2], q[3])) for q in rects if q is not b
            )
            if inside and not hit:
                choice = (cx, cy, where, r)
                break
        if choice is None:
            cx, cy, where = cands[len(placed) % 2]
            yb = (cy, cy + up * h_um) if where == "above" else (cy - up * h_um, cy)
            choice = (cx, cy, where, (cx, cx + w_um, min(yb), max(yb)))
        cx, cy, where, r = choice
        placed.append(r)
        va = "bottom" if where == "above" else "top"
        texts.append(
            ax.text(
                cx,
                cy,
                label,
                color=pal["fg"],
                fontsize=fs,
                fontweight="bold",
                ha="left",
                va=va,
                zorder=9,
                path_effects=_stroke(pal, 2.5 * s),
            )
        )
    return texts


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _p_line(ax: Any, text: str, pal: dict[str, Any], s: float, *, where: str) -> Any:
    """The short clone-level p line: bottom left of a tile (``"bottom"``) or under its title (``"title"``)."""
    y, va = (0.03, "bottom") if where == "bottom" else (0.965 - 0.05 * 7.5 / max(ax.figure.get_figheight(), 1), "top")
    return ax.text(0.015, y, text, transform=ax.transAxes, color=pal["fg"], fontsize=_FS["small"] * s, va=va,
                   ha="left", path_effects=_stroke(pal, 3.0 * s), zorder=10, clip_on=False)  # fmt: skip


def _zoom_panel(
    ax: Any,
    data: _AtlasData,
    rid: Any,
    channel: str,
    box: tuple[float, float, float, float],
    s: float,
) -> list[Any]:
    """One zoom of a region: ``channel`` = "csr", "shm" or "gex"."""
    pal = data.pal
    _base(ax, data, box)
    row = data.regions.loc[rid]
    texts = []
    if channel in ("csr", "shm"):
        img = data.map_image(channel)
        if img is not None:
            _draw_image(ax, data.frame, data.grid, img, z=0, clip=data.map_clip(channel))
        _draw_lines(ax, data.frame, data.region_lines(), pal["csr"], _LW["zoom"] * s, 6)
        inb = data.loci_in(box)
        p = data.loci_xy
        if channel == "csr":
            state, cls = data.isotype_dots()
            un = inb & (state == 0)
            sw = inb & (state == 1)
            ax.scatter(p[un, 0], p[un, 1], s=_DOT["small"] * s**2, c=pal["nomut"], lw=0, zorder=7)
            cols = [pal["iso"].get(c, pal["iso"]["IgG"]) for c in cls[sw]]
            ax.scatter(
                p[sw, 0],
                p[sw, 1],
                s=_DOT["big"] * s**2,
                c=cols,
                lw=_DOT["edge"] * s,
                edgecolors=pal["dot_edge"],
                zorder=8,
            )
            share = _pct(row.get("switched_share"))
            text = f"CSR · region {rid}" + (f" · {share} switched" if share else "")
        else:
            mut = data.mutation_dots()
            no = inb & (mut == 0)
            yes = inb & (mut == 1)
            ax.scatter(p[no, 0], p[no, 1], s=_DOT["small"] * s**2, c=pal["nomut"], lw=0, zorder=7)
            ax.scatter(
                p[yes, 0],
                p[yes, 1],
                s=_DOT["big"] * s**2,
                c=pal["mut"],
                lw=_DOT["edge"] * s,
                edgecolors=pal["dot_edge"],
                zorder=8,
            )
            share = _pct(row.get("mutated_share"))
            text = "SHM" + (f" · {share} mutated" if share else "")
        if img is None:
            _note(ax, f"no {channel.upper()} map", pal, _FS["note"] * s)
        texts.append(_label(ax, text, pal, _FS["label"] * s, s))
    else:
        label = row.get("gex_label")
        has_label = isinstance(label, str) and label.strip() != ""
        key = data.module_key(label) if has_label else None
        img, _ = data.module_image(key) if key is not None else (None, False)
        if img is not None:
            _draw_image(ax, data.frame, data.grid, img, z=0, clip=data.tissue_fill())
            norm = data._cache.get(f"norm_module_{key}")
            if norm is not None:
                _mini_bar(ax, norm, pal["score_ramp"], pal, s, pct=False, where="right")
        else:
            _note(ax, "no score map for this label" if has_label else "no GEX profile", pal, _FS["small"] * s)
        _draw_lines(ax, data.frame, data.gc_lines(), pal["gcb"], _LW["gcb"] * s, 7)
        _draw_lines(ax, data.frame, data.region_lines(), pal["csr"], _LW["gex"] * s, 8)
        shown = _short_label(label) if has_label else "GEX n/a"
        texts.append(_label(ax, f"Hotspot · {shown}", pal, _FS["gex"] * s, s))
    _set_view(ax, box, data.invert_y)
    return texts


def _dot_legend(ax: Any, data: _AtlasData, channel: str, s: float) -> Any:
    from matplotlib.lines import Line2D

    pal = data.pal
    big, small = 4.0 * s, 2.5 * s
    if channel == "csr":
        _, cls = data.isotype_dots()
        present = [c for c in _SWITCHED if (cls == c).any()] or ["IgG", "IgA"]
        hh = [
            Line2D([], [], ls="", marker="o", ms=big, color=pal["iso"][c], mec=pal["dot_edge"], mew=0.6 * s, label=c)
            for c in ("IgG", "IgA", "IgE")
            if c in present
        ]
        hh.append(Line2D([], [], ls="", marker="o", ms=small, color=pal["nomut"], label="IgM / IgD"))
    else:
        hh = [
            Line2D(
                [], [], ls="", marker="o", ms=big, color=pal["mut"], mec=pal["dot_edge"], mew=0.6 * s,
                label=data.mutation_label(),
            ),
            Line2D([], [], ls="", marker="o", ms=small, color=pal["nomut"], label="not mutated"),
        ]  # fmt: skip
    leg = ax.legend(
        handles=hh,
        loc="lower right",
        frameon=True,
        facecolor=pal["bg"],
        framealpha=0.82,
        edgecolor="none",
        fontsize=_FS["legend"] * s,
        labelcolor=pal["fg"],
        handletextpad=0.45,
        labelspacing=0.25,
        borderpad=0.4,
        borderaxespad=0.35,
    )
    leg.set_zorder(20)
    return leg


def _gex_legend(ax: Any, data: _AtlasData, s: float, *, gcb: bool, unscored: bool) -> Any:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pal = data.pal
    hh = [Line2D([], [], lw=1.6 * s, color=pal["csr"], label="region (CSR hotspot)")]
    if gcb:
        hh.append(Line2D([], [], lw=1.0 * s, color=pal["gcb"], label="GC B (GEX)"))
    if unscored:
        hh.append(Patch(color=pal["unscored"], label=f"non-B {data.unit_word()}"))
    leg = ax.legend(
        handles=hh,
        loc="lower right",
        frameon=True,
        facecolor=pal["bg"],
        framealpha=0.82,
        edgecolor="none",
        fontsize=_FS["legend"] * s,
        labelcolor=pal["fg"],
        handlelength=1.6,
        borderaxespad=0.35,
    )
    leg.set_zorder(20)
    return leg


def _line_legend(
    fig: Any,
    data: _AtlasData,
    s: float,
    *,
    shm: str | None,
    gcb: bool,
    unscored: bool,
    extra_text: str | None,
    y: float,
    csr_label: str | None = "CSR hotspot",
) -> Any:
    """The legend row: outline colours (and the grey of units without a score) of what the figure shows."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pal = data.pal
    hh = [Line2D([], [], lw=1.6 * s, color=pal["csr"], label=csr_label)] if csr_label else []
    if shm:
        hh.append(Line2D([], [], lw=1.6 * s, color=pal["shm"], label=shm))
    if gcb:
        hh.append(Line2D([], [], lw=1.0 * s, color=pal["gcb"], label="GC B (GEX)"))
    if unscored:
        hh.append(Patch(color=pal["unscored"], label=f"non-B {data.unit_word()}"))
    if extra_text:
        fig.text(0.995, y + 0.006, extra_text, ha="right", va="bottom", color=pal["muted"], fontsize=_FS["small"] * s)
    if not hh:
        return None
    leg = fig.legend(
        handles=hh,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(hh),
        frameon=False,
        fontsize=_FS["legend_row"] * s,
        labelcolor=pal["fg"],
        handlelength=1.8,
        columnspacing=2.5,
    )
    return leg


def _fit_texts(fig: Any, texts: Sequence[Any], min_scale: float = 0.6) -> list[str]:
    """Shrink panel texts that do not fit inside their axes (then truncate); return the texts still outside."""
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except (AttributeError, NotImplementedError, RuntimeError):  # a canvas without a renderer: leave texts as they are
        return []
    bad = []
    for t in texts:
        if t.axes is None:
            continue
        ab = t.axes.get_window_extent(renderer)
        fs0 = t.get_fontsize()
        for _ in range(12):
            bb = t.get_window_extent(renderer)
            if bb.x1 <= ab.x1 - 2 and bb.x0 >= ab.x0 - 1:
                break
            if t.get_fontsize() > fs0 * min_scale:
                t.set_fontsize(t.get_fontsize() * 0.92)
            else:
                txt = t.get_text()
                if len(txt) <= 4:
                    break
                t.set_text(txt[:-2].rstrip() + "…")
        bb = t.get_window_extent(renderer)
        if not (bb.x1 <= ab.x1 + 1 and bb.x0 >= ab.x0 - 1 and bb.y0 >= ab.y0 - 1 and bb.y1 <= ab.y1 + 1):
            bad.append(t.get_text())
    return bad


# ------------------------------------------------------------------------------------------------- region choice


def _choose_regions(data: _AtlasData, regions: Any, max_regions: int) -> tuple[list[Any], int]:
    """Region ids to zoom on, in their order along the section, and the number of regions available."""
    order = list(data.regions.index)
    n_all = len(order)
    if regions is None:
        sel = order if n_all <= max_regions else order[:3]
    elif isinstance(regions, int | np.integer) and not isinstance(regions, bool):
        if regions < 0:
            raise ValueError("regions must be >= 0")
        sel = order[: int(regions)]
    else:
        want = list(regions)
        missing = [r for r in want if r not in data.regions.index]
        if missing:
            raise KeyError(f"regions {missing} not in result.regions (available: {order})")
        sel = [r for r in order if r in want]
    if len(sel) > max_regions:
        raise ValueError(f"at most {max_regions} regions fit in the atlas; got {len(sel)} (pass regions=[...])")
    return sel, n_all


# ---------------------------------------------------------------------------------------------- public functions


def _missing_map_reason(result: Any, which: str) -> str:
    """Short reason, from the workflow's messages, why the ``which`` ("SHM" / "CSR") map is missing ("" if unknown)."""
    msgs = [str(m) for m in (getattr(result, "messages", None) or [])]
    if which == "SHM" and any(m.startswith("no V alignments") for m in msgs):
        return "The records have no V alignments; for AIR-SPACE, io.read_airspace(..., alignments=True) keeps MiXCR's"
    hit = next((m for m in msgs if which in m and "skipped" in m), "")
    return hit if len(hit) <= 110 else hit[:108].rstrip() + "…"


def _new_figure(figsize: tuple[float, float], dpi: float, pal: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    return plt.figure(figsize=figsize, dpi=dpi, facecolor=pal["bg"])


def _layout(n: int, sec_aspect: float, W: float, H: float, legend: bool, n_maps: int = 2) -> dict[str, Any]:
    """Axes rectangles (figure fractions) of the atlas: two overview maps on top, then the zooms.

    Three or more regions: one column per region, rows CSR / SHM / GEX (the archive layout). One or two regions:
    one row per region, columns CSR / SHM / GEX, so that the zoom panels keep a reasonable shape. No region and one
    map (e.g. no SHM map): that map alone, the figure as wide as it needs (``width`` in inches).
    """
    legend_h = 0.042 if legend else 0.008
    mx = 0.004
    if n == 0 and n_maps == 1:  # one map: the figure narrows to its width, with a note of up to three lines
        note_h = 0.1
        y0 = legend_h + note_h
        h = 1 - y0 - 0.008
        width = float(np.clip(h * H * sec_aspect + 0.3, 0.5 * W, W))
        w = min(1 - 2 * mx, h * H * sec_aspect / width)
        return {"overview": [((1 - w) / 2, y0, w, h)], "zoom": {}, "aspect": 1.0, "legend_h": legend_h,
                "top_y": 1.0, "note_y": legend_h + note_h / 2, "width": width}  # fmt: skip
    if n == 0:  # overview only: the two maps fill the figure, stacked when the section is long
        note_h = 0.05
        y0 = legend_h + note_h
        if sec_aspect > 2.2:
            h = (1 - y0 - 0.02) / 2
            rects = [(mx, y0 + h + 0.012, 1 - 2 * mx, h), (mx, y0 + 0.004, 1 - 2 * mx, h)]
        else:
            rects = [(mx, y0, 0.5 - 1.5 * mx, 1 - y0 - 0.008), (0.5 + 0.5 * mx, y0, 0.5 - 1.5 * mx, 1 - y0 - 0.008)]
        return {"overview": rects, "zoom": {}, "aspect": 1.0, "legend_h": legend_h, "top_y": 1.0,
                "note_y": legend_h + note_h / 2}  # fmt: skip
    top_w = 0.5 - 1.5 * mx
    natural = (top_w * W / max(sec_aspect, 1e-9)) / H
    cap = {1: 0.36, 2: 0.32}.get(n, 0.30)
    top_h = float(np.clip(natural, 0.16, cap))
    top_y = 1 - top_h - 0.006
    out: dict[str, Any] = {
        "overview": [(mx + k * (top_w + mx), top_y, top_w, top_h) for k in range(2)],
        "zoom": {},
        "aspect": 1.0,
        "legend_h": legend_h,
        "top_y": top_y,
        "note_y": legend_h / 2,
    }
    if n:
        z_top, z_bot, cpad, vgap = top_y - 0.006, legend_h, 0.0015, 0.004
        ncol, nrow = (n, 3) if n >= 3 else (3, n)
        cw = 1 / ncol - 2 * cpad
        rh = (z_top - z_bot - (nrow - 1) * vgap) / nrow
        for k in range(n):
            for c, ch in enumerate(_CHANNELS):
                col, row = (k, c) if n >= 3 else (c, k)
                out["zoom"][(k, ch)] = (col / ncol + cpad, z_top - (row + 1) * rh - row * vgap, cw, rh)
        out["aspect"] = (cw * W) / (rh * H)
    return out


def hotspot_atlas(
    result: Any,
    *,
    regions: Any = None,
    style: str | None = "dark",
    figsize: tuple[float, float] = (16.0, 9.0),
    dpi: float = 200,
    max_regions: int = 4,
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    legend: bool = True,
    score_mask: Any = None,
    clone_p: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> Any:
    """CSR and SHM hotspots of a section with zooms on the regions where they overlap (the archive's 03-01 figure).

    Parameters
    ----------
    result
        A :class:`spaclone.bcr.HotspotWorkflowResult` (or any object or mapping with the same fields). Read:
        ``grid`` and ``tissue``; ``csr`` and ``shm`` (:class:`~spaclone.tl.HotspotResult`: ``map`` for the share
        maps, ``labels`` and ``table`` for the hotspots); ``loci`` (``x_um``, ``y_um``, ``switched``, ``mutated``,
        ``evaluable``, ``isotype_class``); ``regions`` (``region_id``, ``csr_hotspots``, ``shm_hotspots``,
        ``switched_share``, ``mutated_share``, ``gex_label``; hotspot ids as lists or ``";"``-joined strings; a region's
        shape is ``result.region_labels`` when present, else the union of its CSR hotspots); ``module_maps`` (cell
        type -> score per GEX unit, positions from ``result.units``, ``result.gex_units`` or ``result.gex``; units
        without a score are drawn dark grey); ``gc_gex`` (GEX germinal-centre B call: a boolean per GEX unit, a
        boolean raster, or contour vertices ``contour_id, x_um, y_um``);
        ``params`` (``min_excess`` names the mutation threshold in the legend). Missing optional fields leave their
        part of the figure out.
    regions
        Regions to zoom on: None (all when there are at most ``max_regions``, otherwise the first three along the
        section), a number ``k`` (the first ``k`` along the section; 0 draws the overview only) or a list of region
        ids.
    style
        ``"dark"`` (house style: black slide) or ``"light"`` / None.
    figsize, dpi
        Figure size (inches) and resolution; fonts, dots and lines scale with the width (reference 13.33 in).
    max_regions
        Largest number of zoom columns.
    rotate
        True: rotate the section about its centre so that its long axis is horizontal (sections whose variance along
        the principal axis is less than 1.5 times that across it are round enough to keep their orientation);
        False: original orientation; a number: the rotation in degrees (counter-clockwise).
    invert_y
        True (default): image convention, y increases downwards; False: y increases upwards.
    window_pad_um, min_window_um
        Margin around a region (and the SHM hotspots it touches) and the smallest window height; windows are then
        widened to the panel shape.
    legend
        Draw the dot legends and the legend row.
    score_mask
        GEX units whose module score is shown (others are drawn dark grey, e.g. non-B units as in the archive
        figure, where only B-lineage bins were scored): a boolean column of ``result.units`` (or of
        ``result.gex.obs``), a boolean Series indexed by GEX unit, or one value per unit. None: the result's
        ``score_mask`` when present (``hotspot_workflow(score_units=...)``), else every scored unit.
    clone_p
        Write each region's clone-level map p-values (``regions.csr_clone_p`` / ``shm_clone_p``: whole clones
        relabelled) at the bottom left of its GEX tile, e.g. ``"clone-level p: CSR = 0.018 · SHM = 1"``; values at the
        smallest attainable p are shown as ``"< 1/n_perm"``. Nothing is written when the result has no such values.
    x_key, y_key
        Coordinate columns of ``result.loci``.

    Returns
    -------
    matplotlib.figure.Figure
        The atlas. ``fig.axes``: two overview maps (one when the other map is missing), then the zooms region by
        region (CSR, SHM, GEX).

    Notes
    -----
    With no region (no CSR hotspot overlaps an SHM hotspot) the figure shows the overview and a note; when one of
    the two maps is missing (e.g. no SHM map: records without V alignments), it shows the other map alone in a
    narrower figure, with the reason under it. Region numbers
    are the ``region_id`` values; columns follow the order along the (rotated) section. The colour scales are the
    archive's: 2nd to 99.5th percentile of the share on eligible pixels; module scores the same over scored units.
    """
    data = _AtlasData(
        result, style=style, rotate=rotate, invert_y=invert_y, x_key=x_key, y_key=y_key, score_mask=score_mask
    )
    pal = data.pal
    sel, n_all = _choose_regions(data, regions, max_regions)
    W, H = figsize
    s = W / _REF_W
    box = data.frame.section_box(data.tissue)
    sec_aspect = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    n = len(sel)
    maps = [w for w in ("csr", "shm") if data.map_image(w) is not None] if n == 0 else ["csr", "shm"]
    maps = maps if len(maps) == 1 else ["csr", "shm"]
    lay = _layout(n, sec_aspect, W, H, legend, n_maps=len(maps))
    fig = _new_figure((lay.get("width", W), H), dpi, pal)
    texts: list[Any] = []
    ov_axes = [fig.add_axes(r) for r in lay["overview"]]
    zoom_axes: dict[tuple[Any, str], Any] = {}
    windows = {rid: data.window(rid, lay["aspect"], window_pad_um, min_window_um) for rid in sel}
    for k, rid in enumerate(sel):
        for ch in _CHANNELS:
            zoom_axes[(rid, ch)] = fig.add_axes(lay["zoom"][(k, ch)])
    for k, which in enumerate(maps):
        texts += _overview_panel(ov_axes[k], data, which, box, sel, windows, s)
    for (rid, ch), ax in zoom_axes.items():
        texts += _zoom_panel(ax, data, rid, ch, windows[rid], s)
        if clone_p and ch == "gex":
            line = data.clone_p_line(rid)
            if line:
                texts.append(_p_line(ax, line, pal, s, where="bottom"))
    if n == 0:
        n_c = 0 if data.csr_labels is None else int(len(np.setdiff1d(np.unique(data.csr_labels), [0])))
        n_s = 0 if data.shm_labels is None else int(len(np.setdiff1d(np.unique(data.shm_labels), [0])))
        note = (
            "No region where a CSR hotspot overlaps an SHM hotspot "
            f"({n_c} CSR hotspot{'s' if n_c != 1 else ''}, {n_s} SHM hotspot{'s' if n_s != 1 else ''})"
            if n_all == 0
            else f"{n_all} region{'s' if n_all != 1 else ''} not zoomed (choose with regions=)"
        )
        if len(maps) == 1:  # the other map is missing: say which, and why when the workflow said so
            gone = "SHM" if maps == ["csr"] else "CSR"
            why = _missing_map_reason(result, gone)
            note = f"No {gone} map in the result, so no CSR ∩ SHM region ({n_c} CSR hotspot{'s' if n_c != 1 else ''})"
            note += f"\n{why}" if why else ""
            per_line = max(30, int(lay.get("width", W) * 72 / (0.56 * _FS["note"] * s)))  # characters that fit
            note = "\n".join(textwrap.fill(part, per_line) for part in note.split("\n"))
        fig.text(0.5, lay["note_y"], note, ha="center", va="center", color=pal["muted"], fontsize=_FS["note"] * s,
                 linespacing=1.5)  # fmt: skip
    if legend:
        if n:
            last = sel[-1]
            _dot_legend(zoom_axes[(last, "csr")], data, "csr", s)
            _dot_legend(zoom_axes[(last, "shm")], data, "shm", s)
        extra = None
        if n and n < n_all:
            extra = f"{n} of {n_all} regions shown ({', '.join(map(str, sel))}); choose with regions="
        _line_legend(
            fig,
            data,
            s,
            csr_label="CSR hotspot" if (n or data.hotspot_lines("csr")) else None,
            shm=("SHM hotspot (top map)" if n else "SHM hotspot") if data.hotspot_lines("shm") else None,
            gcb=bool(n) and data.gc_visible([windows[r] for r in sel]),
            unscored=any(data.unscored(r, windows[r]) for r in sel),
            extra_text=extra,
            y=-0.004,
        )
    _fit_texts(fig, texts)
    return fig


def hotspot_overview(
    result: Any,
    *,
    regions: Any = None,
    style: str | None = "dark",
    figsize: tuple[float, float] | None = None,
    dpi: float = 200,
    layout: str | None = None,
    boxes: bool = True,
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    max_regions: int = 4,
    legend: bool = True,
    axes: Sequence[Any] | None = None,
) -> Any:
    """The whole-section CSR and SHM maps with their hotspots and the numbered region windows (atlas top row).

    Parameters
    ----------
    result
        Workflow result, as in :func:`hotspot_atlas`.
    regions
        Regions whose windows are drawn (as in :func:`hotspot_atlas`); ``0`` or ``boxes=False`` for none.
    style
        ``"dark"`` or ``"light"`` / None.
    figsize
        Figure size; default 16 in wide and as tall as the section needs.
    dpi
        Resolution of the new figure.
    layout
        ``"row"`` (maps side by side), ``"column"`` (stacked) or None (stacked for sections more than 2.2 times
        longer than wide).
    boxes
        Draw the region windows.
    rotate, invert_y, window_pad_um, min_window_um, max_regions
        As in :func:`hotspot_atlas`; windows have the atlas' zoom shape for the same number of regions.
    legend
        Draw the legend of outline colours.
    axes
        Two Axes to draw on (CSR map, SHM map) instead of a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = _AtlasData(result, style=style, rotate=rotate, invert_y=invert_y)
    pal = data.pal
    sel, _ = _choose_regions(data, regions if boxes else 0, max_regions)
    box = data.frame.section_box(data.tissue)
    sec_aspect = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    if layout is None:
        layout = "column" if sec_aspect > 2.2 else "row"
    if layout not in ("row", "column"):
        raise ValueError("layout must be 'row', 'column' or None")
    W = figsize[0] if figsize is not None else 16.0
    s = W / _REF_W
    leg_in = 0.45 * s if legend else 0.1
    if axes is None:
        if figsize is None:
            map_w = W - 0.2 if layout == "column" else (W - 0.3) / 2
            map_h = map_w / sec_aspect
            Hh = (2 * map_h + 0.15 if layout == "column" else map_h + 0.1) + leg_in
            figsize = (W, float(np.clip(Hh, 2.5, 12.0)))
        fig = _new_figure(figsize, dpi, pal)
        H = figsize[1]
        lb = leg_in / H
        if layout == "column":
            h = (1 - lb - 0.02) / 2
            axes = [fig.add_axes((0.005, lb + h + 0.012, 0.99, h)), fig.add_axes((0.005, lb + 0.004, 0.99, h))]
        else:
            axes = [fig.add_axes((0.004, lb, 0.494, 1 - lb - 0.01)), fig.add_axes((0.502, lb, 0.494, 1 - lb - 0.01))]
    else:
        if len(axes) != 2:
            raise ValueError("axes must be two Axes (CSR map, SHM map)")
        fig = axes[0].figure
    windows = {}
    if sel:  # the windows of the default 16 x 9 atlas with the same regions
        ref_aspect = _layout(len(sel), sec_aspect, 16.0, 9.0, True)["aspect"]
        windows = {rid: data.window(rid, ref_aspect, window_pad_um, min_window_um) for rid in sel}
    texts = []
    for ax, which in zip(axes, ("csr", "shm"), strict=True):
        texts += _overview_panel(ax, data, which, box, sel, windows, s)
        names = {"csr": "CSR · switched share", "shm": "SHM · mutated share"}
        texts.append(_label(ax, names[which], pal, _FS["label"] * s, s))
    if legend:
        from matplotlib.lines import Line2D

        hh = [
            Line2D([], [], lw=1.6 * s, color=pal["csr"], label="CSR hotspot"),
            Line2D([], [], lw=1.6 * s, color=pal["shm"], label="SHM hotspot"),
        ]
        if windows:
            from matplotlib.patches import Patch

            hh.append(Patch(fc="none", ec=pal["frame"], lw=1.2 * s, label="window = region"))
        fig.legend(
            handles=hh,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(hh),
            frameon=False,
            fontsize=_FS["legend_row"] * s,
            labelcolor=pal["fg"],
            handlelength=1.8,
            columnspacing=2.5,
        )
    _fit_texts(fig, texts)
    return fig


def region_zoom(
    result: Any,
    region_id: Any,
    *,
    channels: str | Sequence[str] = _CHANNELS,
    ax: Any = None,
    style: str | None = "dark",
    figsize: tuple[float, float] | None = None,
    dpi: float = 200,
    rotate: bool | float = True,
    invert_y: bool = True,
    window_pad_um: float = 90.0,
    min_window_um: float = 200.0,
    window: tuple[float, float, float, float] | None = None,
    legend: bool = True,
    score_mask: Any = None,
    clone_p: bool = True,
) -> Any:
    """The zooms of one region: CSR (isotype dots), SHM (mutated dots) and GEX (module score) side by side.

    Parameters
    ----------
    result
        Workflow result, as in :func:`hotspot_atlas`.
    region_id
        Region to show (a ``region_id`` of ``result.regions``).
    channels
        ``"csr"``, ``"shm"``, ``"gex"`` or a sequence of them (panel order).
    ax
        Axes (one channel) or a sequence of Axes (one per channel) to draw on; a new figure if None.
    style
        ``"dark"`` or ``"light"`` / None.
    figsize, dpi
        Size and resolution of a new figure (default 16 in wide, 5.4 in per row of panels).
    rotate, invert_y, window_pad_um, min_window_um
        As in :func:`hotspot_atlas`.
    window
        ``(xmin, xmax, ymin, ymax)`` in display coordinates (rotated µm) instead of the automatic window.
    legend
        Draw the dot legends inside the CSR and SHM panels.
    score_mask
        GEX units whose module score is shown, as in :func:`hotspot_atlas`.
    clone_p
        Write the region's clone-level map p-values, as in :func:`hotspot_atlas` (on the GEX panel, else under the
        first panel's title).

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The Axes when a single channel is drawn (on ``ax`` or a new figure), otherwise the Figure.
    """
    data = _AtlasData(result, style=style, rotate=rotate, invert_y=invert_y, score_mask=score_mask)
    if region_id not in data.regions.index:
        raise KeyError(f"region {region_id!r} not in result.regions (available: {list(data.regions.index)})")
    chans = [channels] if isinstance(channels, str) else list(channels)
    bad = [c for c in chans if c not in _CHANNELS]
    if bad or not chans:
        raise ValueError(f"channels must be among {_CHANNELS}, got {channels!r}")
    pal = data.pal
    if ax is None:
        W, H = figsize if figsize is not None else (16.0 if len(chans) > 1 else 6.0, 5.4 if len(chans) > 1 else 4.5)
        fig = _new_figure((W, H), dpi, pal)
        k = len(chans)
        row = legend and k > 1
        y0, h = (0.075, 0.915) if row else (0.01, 0.98)
        axes = [fig.add_axes((c / k + 0.002, y0, 1 / k - 0.004, h)) for c in range(k)]
    else:
        axes = [ax] if not isinstance(ax, Sequence) else list(ax)
        if len(axes) != len(chans):
            raise ValueError("pass one Axes per channel")
        fig = axes[0].figure
    pos = axes[0].get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    s = w_in / (_REF_W / 3)  # the archive's zoom panels are a third of the slide wide
    aspect = w_in / max(h_in, 1e-9)
    box = window if window is not None else data.window(region_id, aspect, window_pad_um, min_window_um)
    texts = []
    gex_shown = "gex" in chans
    gcb = gex_shown and data.gc_visible([box])
    unscored = gex_shown and data.unscored(region_id, box)
    for a, ch in zip(axes, chans, strict=True):
        texts += _zoom_panel(a, data, region_id, ch, box, s)
        if legend and ch in ("csr", "shm"):
            _dot_legend(a, data, ch, s)
        if legend and ch == "gex" and len(chans) == 1:
            _gex_legend(a, data, s, gcb=gcb, unscored=unscored)
    line = data.clone_p_line(region_id) if clone_p else None
    if line:
        on = axes[chans.index("gex")] if "gex" in chans else axes[0]
        texts.append(_p_line(on, line, pal, s, where="bottom" if "gex" in chans else "title"))
    if legend and ax is None and len(chans) > 1:
        _line_legend(fig, data, s, shm=None, gcb=gcb, unscored=unscored, extra_text=None, y=0.0,
                     csr_label="region (CSR hotspot)")  # fmt: skip
    _fit_texts(fig, texts)
    return axes[0] if len(chans) == 1 else fig
