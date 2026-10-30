"""One orientation and one frame for every tissue map: the main tissue, its long axis horizontal, y downwards.

:class:`SectionFrame` applies the rule of :func:`spaclone.pl.hotspot_atlas` to any map: ``rotate=True`` turns the
section about its centre so that the principal axis of its points is horizontal (sign fixed so the axis points to +x),
when the section is elongated (variance along the principal axis at least 1.5 times that across it; round sections
keep their orientation); ``False`` keeps the original coordinates; a number is an angle in degrees. Rotated
coordinates are shifted so that the section starts at 0 µm on both axes. Every map then sets the y axis downwards
with :func:`spaclone.pl._colors.set_y_down` unless ``invert_y=False``.

The frame of a map is the main tissue (:func:`main_tissue`): with ``crop=True`` (the default of every map) a
detached speck of records far from the section, such as a stray cluster on the glass, neither stretches the view nor
turns the section; the points left outside are counted in an INFO message of the ``spaclone`` logger, and
``crop=False`` frames every point.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("spaclone")

#: Axis labels of an unrotated and a rotated map.
LABELS = {False: ("x (µm)", "y (µm)"), True: ("µm along the long axis", "µm across")}
#: Pitch (µm) of the occupancy raster that finds the main tissue.
PITCH_UM = 50.0
#: Share of the points that the kept pieces of tissue hold together.
KEEP = 0.95
#: Fewer points than this are never cropped.
MIN_POINTS = 50
#: Pieces this close to the kept ones (share of their larger side) are kept too: islands of the same section.
NEAR = 0.2


def main_tissue(xy: Any, *, pitch_um: float = PITCH_UM, keep: float = KEEP, near: float = NEAR) -> np.ndarray:
    """Mask of the points that belong to the main tissue (specks far from the section excluded).

    The points are rasterised at ``pitch_um``; the occupied cells are split into 8-connected components and the
    largest components (by their number of points) are kept until they hold at least ``keep`` of the points. When the
    components that hold at least 1% of the points hold less than half of them together (sparse points: a spot
    lattice coarser than the raster, such as Visium's 100 µm, or a thin scatter of records), the pitch is doubled
    until they do, so that sparse tissue is not cut into single points. Components that come within ``near`` times
    the larger side of the kept components' bounding box of a kept cell are kept too (islands of the section next to
    it); only pieces farther away are dropped.

    Parameters
    ----------
    xy
        ``(n, 2)`` positions in µm.
    pitch_um
        Starting pitch of the raster.
    keep
        Share of the points the kept components must hold.
    near
        Distance to the kept cells, as a share of the kept bounding box's larger side, within which other components
        are kept too.

    Returns
    -------
    numpy.ndarray
        Boolean mask (``False`` for detached points and for non-finite positions). Fewer than 50 finite points are
        all kept.
    """
    from scipy import ndimage

    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    ok = np.isfinite(pts).all(axis=1)
    out = ok.copy()
    p = pts[ok]
    n = len(p)
    if n < MIN_POINTS:
        return out
    lo = p.min(axis=0)
    span = float((p.max(axis=0) - lo).max())
    pitch = max(float(pitch_um), span / 1000.0)  # at most ~1000 cells a side
    eight = np.ones((3, 3), dtype=bool)
    while True:
        idx = np.floor((p - lo) / pitch).astype(np.int64)
        shape = tuple(int(v) for v in idx.max(axis=0) + 1)
        occ = np.zeros(shape, dtype=bool)
        occ[idx[:, 0], idx[:, 1]] = True
        lab, k = ndimage.label(occ, structure=eight)
        comp = lab[idx[:, 0], idx[:, 1]] - 1
        counts = np.bincount(comp, minlength=k)
        big = counts >= max(0.01 * n, 10)
        if counts[big].sum() >= 0.5 * n or max(shape) <= 4:
            break
        pitch *= 2.0
    order = np.argsort(-counts, kind="stable")
    n_keep = int(np.searchsorted(np.cumsum(counts[order]), keep * n)) + 1
    kept = np.zeros(k, dtype=bool)
    kept[order[:n_keep]] = True
    if near > 0 and not kept.all():  # islands next to the kept pieces belong to the section
        core = np.r_[False, kept][lab]
        side = float((p[kept[comp]].max(axis=0) - p[kept[comp]].min(axis=0)).max())
        dist = ndimage.distance_transform_edt(~core) * pitch  # µm from each cell to the nearest kept cell
        gap = np.asarray(ndimage.minimum(dist, labels=lab, index=np.arange(1, k + 1)), dtype=float)
        kept |= gap <= near * side + pitch
    out[ok] = kept[comp]
    return out


class SectionFrame:
    """Display coordinates of a section: rotated about its centre so that its long axis is horizontal.

    Parameters
    ----------
    xy
        ``(n, 2)`` points of the section in µm (spots, units, tissue pixels); they fix the angle, the origin and the
        view.
    rotate
        ``True``: the principal axis of ``xy`` when the section is elongated; ``False`` / ``None``: no rotation; a
        number: that counter-clockwise angle in degrees.
    crop
        ``True``: the angle, the origin and the view come from the main tissue (:func:`main_tissue`); ``False``:
        from every point.
    """

    min_ratio = 1.5

    def __init__(self, xy: Any, rotate: bool | float | None = True, *, crop: bool = True) -> None:
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        self.points = pts
        self.crop = bool(crop)
        self.core = main_tissue(pts) if crop else np.ones(len(pts), dtype=bool)
        use = pts[self.core]
        angle = 0.0
        if rotate is True:
            if len(use) >= 3:
                cov = np.cov((use - use.mean(0)).T)
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
        c, s = np.cos(angle), np.sin(angle)
        self.R = np.array([[c, -s], [s, c]])
        self.centre = use.mean(0) if (self.rotated and len(use)) else np.zeros(2)
        self.offset = np.zeros(2)
        if self.rotated and len(use):
            self.offset = ((use - self.centre) @ self.R.T).min(axis=0)

    @property
    def labels(self) -> tuple[str, str]:
        """X and y axis labels."""
        return LABELS[self.rotated]

    def __call__(self, x: Any, y: Any = None) -> np.ndarray:
        """``(n, 2)`` display coordinates of points given as ``(x, y)`` or as one ``(n, 2)`` array."""
        if y is None:
            p = np.asarray(x, dtype=float).reshape(-1, 2)
        else:
            p = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
        if not self.rotated:
            return p
        return (p - self.centre) @ self.R.T - self.offset

    def affine(self) -> Any:
        """Matplotlib transform from original µm to display µm (for images and contours)."""
        from matplotlib.transforms import Affine2D

        if not self.rotated:
            return Affine2D()
        t = -self.R @ self.centre - self.offset
        return Affine2D(np.array([[self.R[0, 0], self.R[0, 1], t[0]], [self.R[1, 0], self.R[1, 1], t[1]], [0, 0, 1]]))

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """``(x0, x1, y0, y1)``: bounding box of the main tissue in display µm (``None`` without points)."""
        if not self.core.any():
            return None
        d = self(self.points[self.core])
        lo, hi = d.min(axis=0), d.max(axis=0)
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    def aspect(self, default: float = 1.0) -> float:
        """Height / width of :attr:`extent` (``default`` without points or width)."""
        e = self.extent
        if e is None or e[1] - e[0] <= 0:
            return default
        return float((e[3] - e[2]) / (e[1] - e[0]))

    def figsize(self, width: float, *, extra_height: float = 0.0, lo: float = 3.0, hi: float = 12.0) -> tuple:
        """Figure size of ``width`` inches whose height follows the section (plus ``extra_height``), within bounds."""
        return float(width), float(np.clip(width * self.aspect() + extra_height, lo, hi))

    def set_view(self, ax: Any, *, invert_y: bool = True, pad_um: float | None = None) -> tuple:
        """Frame ``ax`` on the main tissue (equal aspect, y downwards unless ``invert_y=False``).

        Parameters
        ----------
        ax
            Axes drawn in this frame's display coordinates.
        invert_y
            Draw y increasing downwards.
        pad_um
            Margin around the tissue (default: 2% of its larger side).

        Returns
        -------
        tuple
            ``(x0, x1, y0, y1)``: the limits set, before the y orientation.
        """
        e = self.extent
        if e is None:
            ax.set_aspect("equal")
            return tuple(ax.get_xlim()) + tuple(sorted(ax.get_ylim()))
        x0, x1, y0, y1 = e
        pad = 0.02 * max(x1 - x0, y1 - y0, 1.0) if pad_um is None else float(pad_um)
        box = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim((box[3], box[2]) if invert_y else (box[2], box[3]))
        ax.set_aspect("equal")
        d = self(self.points)
        out = int((~((d[:, 0] >= box[0]) & (d[:, 0] <= box[1]) & (d[:, 1] >= box[2]) & (d[:, 1] <= box[3]))).sum())
        if out and not getattr(self, "_told", False):  # once per frame (a figure of panels shares one)
            self._told = True
            n = len(d)
            logger.info(
                "map frame: %s of %s points (%.2g%%) lie apart from the main tissue, outside the view (crop=False "
                "shows every point)",
                f"{out:,}",
                f"{n:,}",
                100.0 * out / n,
            )
        return box


def frame_of(ax: Any) -> SectionFrame | None:
    """The frame a map drew with (``None`` for an axes drawn by something else)."""
    return getattr(ax, "_spaclone_frame", None)


def attach(ax: Any, frame: SectionFrame) -> None:
    """Remember the frame on the axes, so layers added later use the same orientation."""
    ax._spaclone_frame = frame
