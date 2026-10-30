"""Coordinates: rigid / affine transforms of sections and concatenation of several sections.

:func:`transform` scales, rotates, flips or applies an affine matrix to the µm coordinates of a receptor table, of a
spot-level AnnData (``obsm["spatial"]``) or of both parts of a :class:`~spaclone.io.SpatialAIR`; the matrix is
recorded so that the step can be traced. :func:`concat_sections` stacks sections, adds the library column and keeps
each section in its own coordinate frame (distances are only meaningful within a section; every function of the
package stratifies by library).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = ["concat_sections", "transform", "transform_matrix"]


def transform_matrix(
    *,
    scale: float | tuple[float, float] = 1.0,
    rotate_deg: float = 0.0,
    flip: str | None = None,
    affine: Any = None,
    origin_um: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """The 3 x 3 homogeneous matrix of :func:`transform`.

    Points are moved to ``origin_um``, flipped, rotated counter-clockwise by ``rotate_deg``, scaled, moved back, and
    then transformed by ``affine`` (a 2 x 3 or 3 x 3 matrix acting on column vectors ``(x, y, 1)``).

    Parameters
    ----------
    scale
        Isotropic factor or ``(sx, sy)``.
    rotate_deg
        Counter-clockwise rotation in degrees (x to the right, y up).
    flip
        ``"x"`` mirrors x (x -> -x), ``"y"`` mirrors y, ``"xy"`` both; ``None`` none.
    affine
        Extra matrix applied last.
    origin_um
        Centre of the flip, rotation and scaling, in µm.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` matrix ``M`` with ``[x', y', 1] = M @ [x, y, 1]``.
    """
    sx, sy = (float(scale), float(scale)) if np.isscalar(scale) else (float(scale[0]), float(scale[1]))
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx == 0 or sy == 0:
        raise ValueError(f"scale must be finite and non-zero, got {scale!r}")
    if flip not in (None, "x", "y", "xy", "yx"):
        raise ValueError(f"flip must be None, 'x', 'y' or 'xy', got {flip!r}")
    ox, oy = float(origin_um[0]), float(origin_um[1])
    T0 = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], dtype=float)
    T1 = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=float)
    fx = -1.0 if flip is not None and "x" in flip else 1.0
    fy = -1.0 if flip is not None and "y" in flip else 1.0
    F = np.diag([fx, fy, 1.0])
    a = np.deg2rad(float(rotate_deg))
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=float)
    Sm = np.diag([sx, sy, 1.0])
    M = T1 @ Sm @ R @ F @ T0
    if affine is not None:
        A = np.asarray(affine, dtype=float)
        if A.shape == (2, 3):
            A = np.vstack([A, [0.0, 0.0, 1.0]])
        if A.shape != (3, 3):
            raise ValueError(f"affine must be 2 x 3 or 3 x 3, got shape {A.shape}")
        if not np.allclose(A[2], [0.0, 0.0, 1.0]):
            raise ValueError("affine must not be projective (last row 0, 0, 1)")
        M = A @ M
    return M


def _apply(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return xy @ M[:2, :2].T + M[:2, 2]


def _record(meta: dict, M: np.ndarray, params: dict) -> dict:
    meta = dict(meta)
    hist = list(meta.get("transforms", []))
    hist.append({"matrix": M.round(12).tolist(), **params})
    meta["transforms"] = hist
    return meta


def transform(
    data: Any,
    *,
    scale: float | tuple[float, float] = 1.0,
    rotate_deg: float = 0.0,
    flip: str | None = None,
    affine: Any = None,
    origin_um: tuple[float, float] = (0.0, 0.0),
    x_key: str = S.X,
    y_key: str = S.Y,
    spatial_key: str = "spatial",
) -> Any:
    """Scale, rotate, flip or affinely transform coordinates (in µm).

    Parameters
    ----------
    data
        Receptor table (``x_key`` / ``y_key``), AnnData (``obsm[spatial_key]``) or :class:`~spaclone.io.SpatialAIR`
        (both, with the same matrix).
    scale, rotate_deg, flip, affine, origin_um
        See :func:`transform_matrix`. A unit conversion is a scale (e.g. DNB -> µm: ``scale=0.5``).
    x_key, y_key, spatial_key
        Coordinate columns of the table and the ``obsm`` key of the AnnData.

    Returns
    -------
    Same type as ``data``
        A transformed copy. The matrix is appended to ``attrs["spaclone"]["transforms"]`` (table) /
        ``uns["spaclone"]["transforms"]`` (AnnData) / ``meta["transforms"]`` (SpatialAIR).

    Examples
    --------
    >>> aligned = transform(air, rotate_deg=90, origin_um=(5000, 5000))  # doctest: +SKIP
    """
    M = transform_matrix(scale=scale, rotate_deg=rotate_deg, flip=flip, affine=affine, origin_um=origin_um)
    params = {
        "scale": scale if np.isscalar(scale) else list(scale),
        "rotate_deg": float(rotate_deg),
        "flip": flip,
        "origin_um": [float(origin_um[0]), float(origin_um[1])],
    }
    from spaclone.io._container import SpatialAIR

    if isinstance(data, SpatialAIR):
        air = (
            transform(
                data.air,
                scale=scale,
                rotate_deg=rotate_deg,
                flip=flip,
                affine=affine,
                origin_um=origin_um,
                x_key=x_key,
                y_key=y_key,
            )
            if data.air is not None
            else None
        )
        gex = (
            transform(
                data.gex,
                scale=scale,
                rotate_deg=rotate_deg,
                flip=flip,
                affine=affine,
                origin_um=origin_um,
                spatial_key=spatial_key,
            )
            if data.gex is not None
            else None
        )
        return SpatialAIR(air, gex, _record(dict(data.meta or {}), M, params))
    if isinstance(data, pd.DataFrame):
        S.validate_air(data, required=(x_key, y_key), numeric=(x_key, y_key))
        out = data.copy()
        xy = _apply(M, S.coords(data, x_key=x_key, y_key=y_key))
        out[x_key], out[y_key] = xy[:, 0], xy[:, 1]
        out.attrs["spaclone"] = _record(dict(data.attrs.get("spaclone", {})), M, params)
        return out
    try:
        import anndata as ad
    except ImportError:  # pragma: no cover - anndata is a dependency
        ad = None
    if ad is not None and isinstance(data, ad.AnnData):
        if spatial_key not in data.obsm:
            raise KeyError(f"transform: AnnData has no obsm[{spatial_key!r}]")
        out = data.copy()
        xy = np.asarray(out.obsm[spatial_key], dtype=float)
        out.obsm[spatial_key] = _apply(M, xy[:, :2])
        out.uns["spaclone"] = _record(dict(out.uns.get("spaclone", {})), M, params)
        return out
    raise TypeError(f"transform: expected a DataFrame, an AnnData or a SpatialAIR, got {type(data).__name__}")


def concat_sections(
    sections: Sequence[Any],
    *,
    keys: Sequence[str] | None = None,
    library_key: str = S.LIBRARY,
    index_unique: str | None = None,
    spot_key: str = S.SPOT,
) -> Any:
    """Stack several sections into one table (or one :class:`~spaclone.io.SpatialAIR`).

    Parameters
    ----------
    sections
        Receptor tables, or SpatialAIR objects (all of the same kind).
    keys
        Library name of each section, written to ``library_key``. Default: each section's single existing library
        value (an error if a section has none or several).
    library_key
        Library column (and ``obs`` column of the concatenated GEX).
    index_unique
        If given, spot ids become ``"<spot><index_unique><key>"`` (anndata's convention), in the receptor table and in
        the GEX ``obs_names`` alike. Needed when GEX spot ids repeat between sections.
    spot_key
        Spot column.

    Returns
    -------
    pandas.DataFrame or SpatialAIR
        Rows of all sections in order with a new ``RangeIndex``. Coordinates are unchanged: each section stays in its
        own frame. Per-section ``attrs`` (or reader ``meta``) are kept under ``attrs["spaclone"]["sections"]``
        (``meta["sections"]``).

    Examples
    --------
    >>> both = concat_sections([air1, air2], keys=["s1", "s2"])  # doctest: +SKIP
    """
    from spaclone.io._container import SpatialAIR

    secs = list(sections)
    if not secs:
        raise ValueError("concat_sections: no sections")
    kinds = {isinstance(s, SpatialAIR) for s in secs}
    if len(kinds) != 1:
        raise TypeError("concat_sections: mix of SpatialAIR and tables")
    is_sp = kinds.pop()
    tables = [s.air if is_sp else s for s in secs]
    for t in tables:
        if not isinstance(t, pd.DataFrame):
            raise TypeError(f"concat_sections: expected receptor tables, got {type(t).__name__}")
    if keys is None:
        keys = []
        for i, t in enumerate(tables):
            vals = t[library_key].dropna().unique() if library_key in t.columns else []
            if len(vals) != 1:
                raise ValueError(f"concat_sections: section {i} has no single {library_key!r} value; pass keys=")
            keys.append(str(vals[0]))
    keys = [str(k) for k in keys]
    if len(keys) != len(secs):
        raise ValueError("concat_sections: one key per section")
    if len(set(keys)) != len(keys):
        raise ValueError(f"concat_sections: keys must be unique, got {keys}")
    parts = []
    for k, t in zip(keys, tables, strict=True):
        p = t.copy()
        p[library_key] = k
        if index_unique is not None and spot_key in p.columns:
            p[spot_key] = p[spot_key].astype(str) + index_unique + k
        parts.append(p)
    air = pd.concat(parts, ignore_index=True, sort=False)
    air.attrs = {
        "spaclone": {"sections": {k: dict(t.attrs.get("spaclone", {})) for k, t in zip(keys, tables, strict=True)}}
    }
    if not is_sp:
        return air
    gexs = [s.gex for s in secs]
    gex = None
    if any(g is not None for g in gexs):
        if any(g is None for g in gexs):
            raise ValueError("concat_sections: some sections have GEX and others do not")
        import anndata as ad

        if index_unique is None:
            names = pd.Index(np.concatenate([np.asarray(g.obs_names) for g in gexs]))
            if names.has_duplicates:
                raise ValueError("concat_sections: GEX spot ids repeat between sections; pass index_unique='-'")
        gex = ad.concat(
            dict(zip(keys, gexs, strict=True)),
            label=library_key,
            index_unique=index_unique,
            merge="same",
            uns_merge="first",
        )
    meta = {"sections": {k: dict(s.meta or {}) for k, s in zip(keys, secs, strict=True)}}
    return SpatialAIR(air, gex, meta)
