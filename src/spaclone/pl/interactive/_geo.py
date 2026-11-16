"""Geometry shared by the interactive tissue maps: section frames, display coordinates, outlines and raster images.

Tissue maps use the orientation of the static maps (:class:`spaclone.pl._frame.SectionFrame`: the section's long axis
horizontal, y downwards), so a section looks the same in a static and an interactive figure. Layers carry display
coordinates; :func:`inverse` gives the affine map back to the original micrometres (the figure's "copy shape"
button uses it).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from spaclone.pl._frame import SectionFrame


def frame_for(xy: Any, rotate: bool | float = True, crop: bool = True) -> SectionFrame:
    """The frame of a section from its points (finite ones; at least one point needed for a rotation)."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return SectionFrame(np.zeros((0, 2)), False, crop=False)
    return SectionFrame(pts, rotate, crop=crop)


def affine(frame: Any) -> np.ndarray:
    """3 x 3 matrix from original micrometres to display coordinates (SectionFrame or the atlas frame)."""
    m = np.eye(3)
    if frame is None or not getattr(frame, "rotated", False):
        return m
    r = np.asarray(frame.R, dtype=float)
    c = np.asarray(frame.centre, dtype=float)
    off = np.asarray(getattr(frame, "offset", np.zeros(2)), dtype=float)
    m[:2, :2] = r
    m[:2, 2] = -r @ c - off
    return m


def js_affine(m: np.ndarray) -> list[float]:
    """A 3 x 3 affine in the canvas convention ``[a, b, c, d, e, f]`` (``X = a x + c y + e``, ``Y = b x + d y + f``)."""
    return [float(m[0, 0]), float(m[1, 0]), float(m[0, 1]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2])]


def inverse(frame: Any) -> list[float]:
    """Display -> original micrometres, canvas convention."""
    return js_affine(np.linalg.inv(affine(frame)))


def apply(m: np.ndarray, xy: Any) -> np.ndarray:
    """``(n, 2)`` points through the 3 x 3 affine ``m``."""
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    return p @ m[:2, :2].T + m[:2, 2]


def home_box(xy_disp: Any, frame: SectionFrame | None = None) -> list[float]:
    """``[x0, x1, y0, y1]`` of the main tissue in display coordinates (the frame's extent), else of all points."""
    if frame is not None and frame.extent is not None:
        return [float(v) for v in frame.extent]
    p = np.asarray(xy_disp, dtype=float).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        return [0.0, 1.0, 0.0, 1.0]
    lo, hi = p.min(axis=0), p.max(axis=0)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]


def spacing_um(xy: Any, default: float = 10.0) -> float:
    """Median distance from a point to its nearest neighbour (the bin, bead or spot pitch), on up to 20,000 points."""
    from scipy.spatial import cKDTree

    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    p = np.unique(p[np.isfinite(p).all(axis=1)], axis=0)
    if len(p) < 2:
        return default
    sub = p[:: max(1, len(p) // 20000)]
    d, _ = cKDTree(p).query(sub, k=2)
    v = float(np.median(d[:, 1]))
    return v if v > 0 else default


def rings_from_paths(paths: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Rings of filled contour paths (contourpy ``OuterCode``: vertices and Path codes), closing vertices dropped."""
    out = []
    for pts, codes in paths:
        pts = np.asarray(pts, dtype=float)
        codes = np.asarray(codes)
        starts = np.flatnonzero(codes == 1)
        bounds = list(starts) + [len(pts)]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            ring = pts[a:b]
            if len(ring) and codes[b - 1] == 79:
                ring = ring[:-1]
            if len(ring) >= 3:
                out.append(ring)
    return out


def grid_image(grid: Any, m: np.ndarray) -> list[float]:
    """Canvas affine placing the image of a grid (row = y index, column = x index) in display coordinates."""
    x0, y0 = grid.origin
    p = float(grid.pixel_um)
    a = m[:2, :2]
    t = a @ np.array([x0, y0]) + m[:2, 2]
    return [float(a[0, 0] * p), float(a[1, 0] * p), float(a[0, 1] * p), float(a[1, 1] * p), float(t[0]), float(t[1])]


def grid_rows(arr: np.ndarray) -> np.ndarray:
    """An array indexed ``[i, j]`` on a grid (x first) as image rows ``[j, i]`` (row 0 = lowest y)."""
    a = np.asarray(arr)
    return np.ascontiguousarray(np.swapaxes(a, 0, 1))


def thin(n: int, max_points: int | None, *, seed: int = 0) -> np.ndarray | None:
    """Indices of a deterministic random subset of ``max_points`` of ``n`` points (None: keep all), in input order."""
    if max_points is None or n <= max_points:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, int(max_points), replace=False))
