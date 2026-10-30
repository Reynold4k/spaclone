"""Tissue structures on a pixel grid: cell-type-rich masks, zones, signed distance to a border, compartments.

All structures live on a :class:`Grid` of square pixels in micrometres. Arrays on a grid are indexed ``[i, j]`` with
the first axis along x and the second along y (pass ``values.T`` to ``imshow`` with ``origin="lower"``). A
:class:`Raster` couples such an array with its grid and can be sampled at unit positions.

The recipe of :func:`mask_from_types` is the one used for the StereoIR spleen in the 2026-10 archive
(``qc_common.b_mask``): count typed units on a 10-µm grid, smooth unit counts and target counts with a 30-µm Gaussian,
threshold the smoothed share of the target class by Otsu's method among pixels with enough typed units, close
(2 iterations) and open (1 iteration), then remove islands and fill holes smaller than 80 pixels. ``zones`` are the
connected components of such a mask (the archive's T zones), :func:`signed_distance` measures how deep a pixel lies
inside or outside it, and :func:`compartments` classifies every pixel by the first of several masks that claims it.

References
----------
Otsu 1979, doi:10.1109/TSMC.1979.4310076 (threshold); Maurer et al. 2003, doi:10.1109/TPAMI.2003.1177156 (exact
Euclidean distance transform, as in ``scipy.ndimage.distance_transform_edt``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spaclone import _schema as S

__all__ = [
    "Grid",
    "Raster",
    "compartments",
    "mask_from_types",
    "nearest_zone",
    "otsu_threshold",
    "signed_distance",
    "tissue_mask",
    "zones",
]


def _xy(x_um: Any, y_um: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_um, dtype=float).ravel()
    y = np.asarray(y_um, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y have different lengths ({len(x)} and {len(y)})")
    return x, y


@dataclass(frozen=True)
class Grid:
    """A grid of square pixels in micrometres.

    Pixel ``(i, j)`` covers ``x0 + i * pixel_um <= x < x0 + (i + 1) * pixel_um`` and the same along y. Arrays on the
    grid have shape ``shape`` and are indexed ``[i, j]`` (first axis x).

    Parameters
    ----------
    origin
        ``(x0, y0)``, the lower corner of pixel ``(0, 0)`` in µm.
    pixel_um
        Pixel side in µm.
    shape
        ``(nx, ny)``, number of pixels along x and along y.
    """

    origin: tuple[float, float]
    pixel_um: float
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not np.isfinite(self.pixel_um) or self.pixel_um <= 0:
            raise ValueError(f"pixel_um must be positive, got {self.pixel_um}")
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError(f"shape must be two positive integers, got {self.shape}")
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))
        object.__setattr__(self, "pixel_um", float(self.pixel_um))
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))

    @classmethod
    def from_points(cls, x_um: Any, y_um: Any, *, pixel_um: float = 10.0) -> Grid:
        """Smallest grid whose origin is the lowest point and that holds every point.

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm (finite).
        pixel_um
            Pixel side in µm.

        Returns
        -------
        Grid
            Origin ``(min x, min y)``; this is the grid of the archive's StereoIR maps (typed bin20 centres).
        """
        x, y = _xy(x_um, y_um)
        if not len(x) or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Grid.from_points needs at least one point and finite coordinates")
        x0, y0 = float(x.min()), float(y.min())
        i = np.floor_divide(x - x0, pixel_um).astype(np.int64)
        j = np.floor_divide(y - y0, pixel_um).astype(np.int64)
        return cls((x0, y0), pixel_um, (int(i.max()) + 1, int(j.max()) + 1))

    @property
    def pixel_area_um2(self) -> float:
        """Area of one pixel in µm²."""
        return self.pixel_um**2

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the grid in µm (for ``imshow(values.T, origin="lower", extent=...)``)."""
        x0, y0 = self.origin
        return (x0, x0 + self.shape[0] * self.pixel_um, y0, y0 + self.shape[1] * self.pixel_um)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates along x (length ``nx``) and along y (length ``ny``), in µm."""
        x0, y0 = self.origin
        return (
            x0 + (np.arange(self.shape[0]) + 0.5) * self.pixel_um,
            y0 + (np.arange(self.shape[1]) + 0.5) * self.pixel_um,
        )

    def index(self, x_um: Any, y_um: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel indices of points.

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm.

        Returns
        -------
        i, j
            Pixel indices, clipped into the grid (use ``inside`` to know which points really fall on it).
        inside
            Boolean, True for finite points on the grid.
        """
        x, y = _xy(x_um, y_um)
        finite = np.isfinite(x) & np.isfinite(y)
        fi = np.floor_divide(np.where(finite, x, self.origin[0]) - self.origin[0], self.pixel_um)
        fj = np.floor_divide(np.where(finite, y, self.origin[1]) - self.origin[1], self.pixel_um)
        i, j = fi.astype(np.int64), fj.astype(np.int64)
        inside = finite & (i >= 0) & (j >= 0) & (i < self.shape[0]) & (j < self.shape[1])
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1), inside

    def rasterize(self, x_um: Any, y_um: Any, values: Any = None) -> np.ndarray:
        """Sum of ``values`` over the points in each pixel.

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm; points off the grid are ignored.
        values
            One value per point (default 1, i.e. counts).

        Returns
        -------
        numpy.ndarray
            Float array of shape ``shape``.
        """
        i, j, inside = self.index(x_um, y_um)
        out = np.zeros(self.shape, dtype=float)
        v = np.ones(len(i)) if values is None else np.asarray(values, dtype=float).ravel()
        if len(v) != len(i):
            raise ValueError("values must have one entry per point")
        np.add.at(out, (i[inside], j[inside]), v[inside])
        return out


@dataclass(frozen=True, eq=False)
class Raster:
    """An array on a :class:`Grid` (mask, labels, distances, codes).

    Parameters
    ----------
    values
        Array of shape ``grid.shape``.
    grid
        Its grid.
    categories
        For code rasters: the name of each code (code ``k`` is ``categories[k]``).
    name
        Optional name.
    """

    values: np.ndarray
    grid: Grid
    categories: tuple[str, ...] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values)
        if v.shape != self.grid.shape:
            raise ValueError(f"raster values have shape {v.shape}, the grid has {self.grid.shape}")
        object.__setattr__(self, "values", v)
        if self.categories is not None:
            object.__setattr__(self, "categories", tuple(str(c) for c in self.categories))

    def __repr__(self) -> str:
        what = f"{len(self.categories)} categories" if self.categories else str(self.values.dtype)
        nx, ny = self.grid.shape
        return f"Raster({self.name or 'unnamed'}: {nx}×{ny} px of {self.grid.pixel_um:g} µm, {what})"

    def sample(self, x_um: Any, y_um: Any, *, fill: Any = np.nan) -> np.ndarray:
        """Values at the pixels holding the points.

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm.
        fill
            Value for points off the grid.

        Returns
        -------
        numpy.ndarray
            One value per point.
        """
        i, j, inside = self.grid.index(x_um, y_um)
        return np.where(inside, self.values[i, j], fill)

    def sample_categories(self, x_um: Any, y_um: Any, *, fill: str = "outside") -> np.ndarray:
        """Category names at the points (code rasters such as :func:`compartments`).

        Parameters
        ----------
        x_um, y_um
            Point coordinates in µm.
        fill
            Name for points off the grid.

        Returns
        -------
        numpy.ndarray
            Object array of names.
        """
        if self.categories is None:
            raise ValueError("this raster has no categories")
        i, j, inside = self.grid.index(x_um, y_um)
        names = np.asarray(self.categories, dtype=object)[self.values[i, j]]
        return np.where(inside, names, fill)

    @property
    def area_um2(self) -> float:
        """Area of the True (non-zero) pixels in µm²."""
        return float(np.count_nonzero(self.values)) * self.grid.pixel_area_um2

    def _other(self, other: Any) -> np.ndarray:
        if isinstance(other, Raster):
            if other.grid != self.grid:
                raise ValueError("rasters are on different grids")
            return other.values.astype(bool)
        arr = np.asarray(other, dtype=bool)
        if arr.shape != self.grid.shape:
            raise ValueError("array shape differs from the raster's grid")
        return arr

    def __invert__(self) -> Raster:
        return Raster(~self.values.astype(bool), self.grid, name=f"not {self.name}" if self.name else "")

    def __and__(self, other: Any) -> Raster:
        return Raster(self.values.astype(bool) & self._other(other), self.grid)

    def __or__(self, other: Any) -> Raster:
        return Raster(self.values.astype(bool) | self._other(other), self.grid)


def _as_mask(
    mask: Raster | np.ndarray, *, grid: Grid | None = None, pixel_um: float | None = None
) -> tuple[np.ndarray, Grid]:
    """Boolean array and grid of a mask given as a Raster, or as an array with a grid or a pixel size."""
    if isinstance(mask, Raster):
        if pixel_um is not None and not np.isclose(pixel_um, mask.grid.pixel_um):
            raise ValueError(f"pixel_um={pixel_um} differs from the raster's pixel size {mask.grid.pixel_um}")
        return mask.values.astype(bool), mask.grid
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("a mask must be a 2-D array or a Raster")
    if grid is None:
        if pixel_um is None:
            raise ValueError("a plain array mask needs pixel_um (or pass a Raster)")
        grid = Grid((0.0, 0.0), pixel_um, arr.shape)
    elif grid.shape != arr.shape:
        raise ValueError("mask shape differs from the grid")
    return arr, grid


def otsu_threshold(values: Any, *, nbins: int = 256) -> float:
    """Otsu's threshold of a set of values (identical to ``skimage.filters.threshold_otsu`` 0.26 on float input).

    Parameters
    ----------
    values
        Values (any shape; NaN not allowed).
    nbins
        Histogram bins over ``[min, max]``.

    Returns
    -------
    float
        The bin centre that maximises the between-class variance; the single value if all values are equal.

    References
    ----------
    Otsu 1979, doi:10.1109/TSMC.1979.4310076.
    """
    v = np.asarray(values, dtype=float).ravel()
    if not len(v):
        raise ValueError("otsu_threshold needs at least one value")
    if not np.isfinite(v).all():
        raise ValueError("otsu_threshold: values must be finite")
    if np.all(v == v[0]):
        return float(v[0])
    counts, edges = np.histogram(v, bins=nbins, range=(v.min(), v.max()))
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts = counts.astype("float32", copy=False)  # as skimage: float32 class weights
    w1 = np.cumsum(counts)
    w2 = np.cumsum(counts[::-1])[::-1]
    m1 = np.cumsum(counts * centres) / w1
    m2 = (np.cumsum((counts * centres)[::-1]) / w2[::-1])[::-1]
    var12 = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centres[np.argmax(var12)])


def _closing(m: np.ndarray, iterations: int, pad_edges: bool) -> np.ndarray:
    """Binary closing with the 4-connected cross.

    ``pad_edges`` pads with background first, so foreground touching the grid edge is kept (the closing is then
    extensive); without it, scipy's ``border_value=0`` erodes a frame of ``iterations`` pixels wherever the mask
    touches the edge (the 2026-10 archive's behaviour).
    """
    if iterations <= 0:
        return m
    p = int(iterations) + 1 if pad_edges else 0
    out = ndi.binary_closing(np.pad(m, p) if p else m, iterations=int(iterations))
    return out[p:-p, p:-p] if p else out


def _remove_small(m: np.ndarray, min_px: float) -> np.ndarray:
    """Drop foreground islands and fill background holes smaller than ``min_px`` pixels (both found on the input)."""
    out = m.copy()
    for target, value in ((m, False), (~m, True)):
        lab, n = ndi.label(target)
        if n == 0:
            continue
        size = np.bincount(lab.ravel(), minlength=n + 1)
        small = size < min_px
        small[0] = False
        out[small[lab]] = value
    return out


def mask_from_types(
    units: pd.DataFrame,
    *,
    types: str | Sequence[str],
    type_key: str = "cell_type",
    grid: Grid | None = None,
    pixel_um: float = 10.0,
    smooth_um: float = 30.0,
    min_density: float = 0.15,
    threshold: Literal["otsu"] | float = "otsu",
    close_iter: int = 2,
    open_iter: int = 1,
    min_area_um2: float = 8000.0,
    pad_edges: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> Raster:
    """Mask of the pixels where a cell class dominates the typed units (e.g. the B-rich or the T-rich area).

    Parameters
    ----------
    units
        One row per typed unit (spot, bin or cell) with its class in ``type_key``. Every row counts towards the
        denominator, so leave untyped units out (the archive used the cell2location-typed bins only).
    types
        Class name(s) whose share is thresholded.
    type_key
        Column holding the class.
    grid
        Grid to use; by default :meth:`Grid.from_points` of the units with ``pixel_um``.
    pixel_um
        Pixel side when ``grid`` is not given.
    smooth_um
        Gaussian standard deviation (µm) applied to the unit counts and to the class counts.
    min_density
        Only pixels whose smoothed count of typed units exceeds this value can be in the mask (and enter Otsu).
    threshold
        ``"otsu"`` (Otsu's threshold of the smoothed class share among those pixels) or a fixed share.
    close_iter, open_iter
        Iterations of binary closing, then opening (4-connected structuring element); 0 skips the step.
    min_area_um2
        Islands and holes smaller than this area are removed / filled (8,000 µm² = 80 pixels of 10 µm).
    pad_edges
        Treat the space beyond the grid as background during the closing, so mask pixels at the grid edge are kept.
        ``False`` reproduces scipy's default border handling used in the 2026-10 archive, which erodes a frame of
        ``close_iter`` pixels wherever the mask touches the grid edge.
    x_key, y_key
        Coordinate columns (µm).

    Returns
    -------
    Raster
        Boolean mask; ``Raster.name`` records the classes.

    Notes
    -----
    With ``pad_edges=False`` this is exactly the archive's ``qc_common.b_mask`` (B-rich, T-rich and erythrocyte-rich
    areas of the StereoIR spleen); the smoothing uses scipy's ``reflect`` boundary, as there.

    References
    ----------
    Otsu 1979, doi:10.1109/TSMC.1979.4310076.
    """
    for c in (type_key, x_key, y_key):
        if c not in units.columns:
            raise KeyError(f"mask_from_types: column {c!r} not in the table")
    types = [types] if isinstance(types, str) else list(types)
    if not smooth_um > 0:
        raise ValueError("smooth_um must be positive")
    x, y = units[x_key].to_numpy(dtype=float), units[y_key].to_numpy(dtype=float)
    if grid is None:
        grid = Grid.from_points(x, y, pixel_um=pixel_um)
    is_type = units[type_key].isin(types).to_numpy()
    if not is_type.any():
        raise ValueError(f"mask_from_types: no unit of class {types}")
    sg = smooth_um / grid.pixel_um
    n = ndi.gaussian_filter(grid.rasterize(x, y), sg)
    k = ndi.gaussian_filter(grid.rasterize(x, y, is_type), sg)
    share = k / np.maximum(n, 1e-9)
    dense = n > min_density
    if not dense.any():
        raise ValueError("mask_from_types: no pixel reaches min_density")
    thr = otsu_threshold(share[dense]) if threshold == "otsu" else float(threshold)
    m = _closing(dense & (share >= thr), int(close_iter), pad_edges)
    if open_iter > 0:
        m = ndi.binary_opening(m, iterations=int(open_iter))
    if min_area_um2 > 0:
        m = _remove_small(m, min_area_um2 / grid.pixel_area_um2)
    return Raster(m, grid, name="+".join(types))


def tissue_mask(
    units: pd.DataFrame,
    *,
    grid: Grid | None = None,
    pixel_um: float = 10.0,
    close_iter: int = 2,
    fill_holes: bool = True,
    pad_edges: bool = True,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> Raster:
    """Pixels holding at least one unit, closed and with holes filled (the archive's tissue mask).

    Parameters
    ----------
    units
        Units that define the tissue (e.g. every tissue bin, or the typed bins).
    grid
        Grid of the mask; by default :meth:`Grid.from_points` of the units with ``pixel_um``.
    pixel_um
        Pixel side when ``grid`` is not given.
    close_iter
        Iterations of binary closing (0 skips).
    fill_holes
        Fill enclosed holes.
    pad_edges
        Keep tissue pixels at the grid edge during the closing (see :func:`mask_from_types`); ``False`` reproduces
        the 2026-10 archive.
    x_key, y_key
        Coordinate columns (µm).

    Returns
    -------
    Raster
        Boolean tissue mask.
    """
    x, y = units[x_key].to_numpy(dtype=float), units[y_key].to_numpy(dtype=float)
    if grid is None:
        grid = Grid.from_points(x, y, pixel_um=pixel_um)
    m = _closing(grid.rasterize(x, y) > 0, int(close_iter), pad_edges)
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    return Raster(m, grid, name="tissue")


def zones(
    mask: Raster | np.ndarray, *, min_area_um2: float = 0.0, connectivity: int = 1, pixel_um: float | None = None
) -> Raster:
    """Connected components of a mask, labelled 1..n (0 = background).

    Parameters
    ----------
    mask
        Boolean Raster (or array, then with ``pixel_um``).
    min_area_um2
        Components smaller than this are dropped; the rest are relabelled 1..n in their original order.
    connectivity
        1 = edge neighbours (``scipy.ndimage.label``'s default, as in the archive), 2 = also diagonal neighbours.
    pixel_um
        Pixel side of a plain-array mask.

    Returns
    -------
    Raster
        Integer labels in ``scipy.ndimage.label`` order (raster scan, first axis x); the number of zones is
        ``values.max()``.
    """
    m, grid = _as_mask(mask, pixel_um=pixel_um)
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    lab, n = ndi.label(m, structure=ndi.generate_binary_structure(2, connectivity))
    if min_area_um2 > 0 and n:
        size = np.bincount(lab.ravel(), minlength=n + 1) * grid.pixel_area_um2
        keep = size >= min_area_um2
        keep[0] = False
        new = np.zeros(n + 1, dtype=lab.dtype)
        new[keep] = np.arange(1, keep.sum() + 1)
        lab = new[lab]
    return Raster(lab, grid, name="zones")


def signed_distance(mask: Raster | np.ndarray, *, pixel_um: float | None = None) -> Raster:
    """Signed distance (µm) from every pixel centre to the mask border: negative inside, positive outside.

    Inside pixels get minus the distance to the nearest outside pixel, outside pixels the distance to the nearest
    inside pixel (exact Euclidean, pixel centre to pixel centre), so no pixel is at 0 and the smallest magnitude is
    one pixel. ``np.maximum(d, 0)`` is the archive's ``dT`` (distance to the nearest T zone, 0 inside) and ``-d`` its
    depth into the B-rich area (positive inside).

    Parameters
    ----------
    mask
        Boolean Raster (or array, then with ``pixel_um``).
    pixel_um
        Pixel side of a plain-array mask.

    Returns
    -------
    Raster
        Float distances in µm (+inf everywhere if the mask is empty, -inf if it covers the grid).
    """
    m, grid = _as_mask(mask, pixel_um=pixel_um)
    if not m.any():
        return Raster(np.full(m.shape, np.inf), grid, name="signed distance")
    if m.all():
        return Raster(np.full(m.shape, -np.inf), grid, name="signed distance")
    inside = ndi.distance_transform_edt(m) * grid.pixel_um
    outside = ndi.distance_transform_edt(~m) * grid.pixel_um
    return Raster(np.where(m, -inside, outside), grid, name="signed distance")


def nearest_zone(labels: Raster) -> Raster:
    """Label of the nearest zone pixel for every pixel (its own label inside a zone).

    Parameters
    ----------
    labels
        Zone labels from :func:`zones` (0 = background).

    Returns
    -------
    Raster
        Integer labels; ties between equidistant zones are resolved as ``scipy.ndimage.distance_transform_edt``
        resolves them (the archive's ``zone_near``).
    """
    lab = np.asarray(labels.values)
    if not (lab > 0).any():
        raise ValueError("nearest_zone: no zone")
    _, (ii, jj) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    return Raster(lab[ii, jj], labels.grid, name="nearest zone")


def compartments(
    rules: Mapping[str, Raster | np.ndarray] | Sequence[tuple[str, Raster | np.ndarray]],
    *,
    default: str,
    grid: Grid | None = None,
) -> Raster:
    """Classify every pixel by the first rule it satisfies (a priority order), else ``default``.

    Parameters
    ----------
    rules
        ``{name: mask}`` or ``[(name, mask), ...]`` in priority order; a name may repeat (e.g. red pulp =
        erythrocyte-rich, then red pulp = outside the B-rich area).
    default
        Class of the pixels no rule claims.
    grid
        Grid, needed only if every mask is a plain array.

    Returns
    -------
    Raster
        Integer codes with ``categories`` = the distinct names in order of first appearance, then ``default``.

    Examples
    --------
    The archive's StereoIR compartments (T zone > red pulp > follicle)::

        comp = compartments([("T zone", t_rich), ("red pulp", ery_rich), ("red pulp", ~b_rich)], default="follicle")
    """
    items = list(rules.items()) if isinstance(rules, Mapping) else list(rules)
    if not items:
        raise ValueError("compartments needs at least one rule")
    g = grid
    for _, m in items:
        if isinstance(m, Raster):
            if g is not None and m.grid != g:
                raise ValueError("compartment masks are on different grids")
            g = m.grid
    if g is None:
        raise ValueError("compartments: pass grid= when every mask is a plain array")
    masks = [_as_mask(m, grid=g)[0] for _, m in items]
    names: list[str] = []
    for name, _ in items:
        if name not in names:
            names.append(name)
    if default in names:
        raise ValueError(f"default {default!r} is also a rule name")
    names.append(default)
    code = np.full(g.shape, len(names) - 1, dtype=np.int16)
    claimed = np.zeros(g.shape, dtype=bool)
    for (name, _), m in zip(items, masks, strict=True):
        new = m & ~claimed
        code[new] = names.index(name)
        claimed |= m
    return Raster(code, g, categories=tuple(names), name="compartments")
