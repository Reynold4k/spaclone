"""Aggregation: spot (or bin, or cell) x clone count matrices, and assignment of records to segmented cells.

:func:`aggregate` turns a receptor table into an :class:`~anndata.AnnData` whose observations are spots, square bins
or cells and whose variables are clones (``var`` = clone table), so that clones can be handled like genes by scanpy
and squidpy (``obsm["spatial"]`` in µm). :func:`assign_to_cells` puts each record into the segmented cell that
contains it; records outside every cell are dropped and counted.

Cell assignment cannot validate position. In the archive (chapter 05, the authors' cell-level pipeline), the map of
BCR-positive cells looked like white-pulp follicles also after every DNB's reads had been moved to a random cell:
cell typing and RNA amount drew the map, not the receptor molecules. Test placement first (``spaclone.qc``,
:func:`spaclone.nulls.placebo`).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse

from spaclone import _schema as S
from spaclone._utils import SpacloneWarning
from spaclone.pp.metrics import _as_flag

__all__ = ["CellAssignment", "CellAssignmentWarning", "aggregate", "assign_to_cells"]


class CellAssignmentWarning(SpacloneWarning):
    """Assigning receptor records to segmented cells does not show that they belong to those cells."""


def _mode_by(values: pd.Series, codes: np.ndarray, n: int) -> np.ndarray:
    """Most frequent non-missing value per code (ties: smallest), as an object array (missing when none)."""
    out = np.full(n, None, dtype=object)
    t = pd.DataFrame({"g": codes, "v": values.astype("string").to_numpy()}).dropna()
    if t.empty:
        return out
    c = t.groupby(["g", "v"], sort=True).size().reset_index(name="k")
    c = c.sort_values(["g", "k", "v"], ascending=[True, False, True], kind="stable").drop_duplicates("g")
    out[c.g.to_numpy(np.int64)] = c.v.to_numpy(dtype=object)
    return out


def aggregate(
    air: pd.DataFrame,
    *,
    bin_um: float | None = None,
    unit: str = "spot",
    value: Literal["records", "reads"] = "records",
    min_records: int = 1,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    var_keys: Sequence[str] = (S.V_CALL, S.J_CALL, S.JUNCTION_AA, S.C_CALL, "isotype"),
    obs_keys: Sequence[str] = ("in_tissue",),
):
    """Count records (or reads) of every clone in every spot, bin or cell.

    Parameters
    ----------
    air
        Receptor table with clone ids and coordinates in µm.
    bin_um
        Side of square bins in µm (anchored at 0: bin ``(floor(x / bin_um), floor(y / bin_um))``); overrides
        ``unit``. ``None`` = no binning.
    unit
        Observation unit without binning: ``"spot"`` (``spot_key``), ``"cell"`` (the ``cell_id`` column written by
        :func:`assign_to_cells`) or the name of any id column.
    value
        ``"records"`` (rows: molecules, units or contigs) or ``"reads"`` (sum of ``reads_key``).
    min_records
        Observations with fewer records are dropped.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key
        Column names. With several libraries an observation is a (library, unit) pair and its name is
        ``"<library>:<unit>"``.
    var_keys
        Record columns summarised per clone in ``var`` by their most frequent value (those present).
    obs_keys
        Record columns summarised per observation in ``obs`` by the first record (those present; flags as bool).

    Returns
    -------
    anndata.AnnData
        ``X``: sparse CSR counts (float32 for records, float64 for reads), observations x clones. ``obs``: the unit
        column (and ``bin_x`` / ``bin_y`` indices with binning), the library, ``n_records``, ``n_reads``,
        ``n_clones``, ``n_loci``, ``x_um``, ``y_um`` (spot or cell: first record / mean of its spots; bin: centre) and
        ``obs_keys``. ``var`` (index = clone id): ``locus``, ``var_keys``, ``n_records``, ``n_units`` (observations
        holding the clone), ``n_reads``. ``obsm["spatial"]``: positions in µm. ``uns["spaclone"]``: the
        parameters and the coordinate unit.

    Notes
    -----
    Records with a missing clone are left out of ``X`` but counted in ``obs["n_records"]``. The matrix lets clones be
    treated like genes (e.g. ``squidpy.gr.spatial_autocorr``); remember that a spot or bin is not a cell, and that
    clone counts per spot inherit every placement artefact of the data (test them with ``spaclone.qc``).

    References
    ----------
    Virshup et al. 2024, doi:10.1038/s41587-023-01733-8 (AnnData); Palla et al. 2022,
    doi:10.1038/s41592-021-01358-2 (squidpy); Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy).

    Examples
    --------
    >>> adata = aggregate(air, bin_um=10)  # doctest: +SKIP
    >>> import squidpy as sq  # doctest: +SKIP
    """
    import anndata as ad

    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if value not in ("records", "reads"):
        raise ValueError(f"value must be 'records' or 'reads', got {value!r}")
    if value == "reads" and reads_key not in air.columns:
        raise KeyError(f"aggregate: value='reads' needs column {reads_key!r}")
    if clone_key not in air.columns:
        raise KeyError(f"aggregate: column {clone_key!r} not in the table")
    if bin_um is not None:
        bin_um = float(bin_um)
        if not np.isfinite(bin_um) or bin_um <= 0:
            raise ValueError(f"bin_um must be a positive number, got {bin_um}")
        S.validate_air(air, required=(x_key, y_key), numeric=(x_key, y_key))
        unit_col = None
    else:
        unit_col = spot_key if unit == "spot" else ("cell_id" if unit == "cell" else unit)
        if unit_col not in air.columns:
            raise KeyError(f"aggregate: unit column {unit_col!r} not in the table")
    lib = library_key if library_key in air.columns else None
    n = len(air)
    work = pd.DataFrame(index=pd.RangeIndex(n))
    if lib is not None:
        work["lib"] = air[lib].astype("string").fillna("NA").to_numpy()
    if bin_um is not None:
        bx = np.floor(air[x_key].to_numpy(dtype=float) / bin_um).astype(np.int64)
        by = np.floor(air[y_key].to_numpy(dtype=float) / bin_um).astype(np.int64)
        work["bx"], work["by"] = bx, by
        keys = (["lib"] if lib is not None else []) + ["bx", "by"]
    else:
        work["u"] = air[unit_col].astype("string").to_numpy()
        if work.u.isna().any():
            raise ValueError(f"aggregate: column {unit_col!r} has missing values (drop unassigned records first)")
        keys = (["lib"] if lib is not None else []) + ["u"]
    g = work.groupby(keys, sort=True, observed=True)
    ocode = g.ngroup().to_numpy(np.int64)
    okeys = g.size().reset_index()[keys]
    n_obs = len(okeys)
    ccode, clones = pd.factorize(air[clone_key], sort=True)
    has_c = ccode >= 0
    n_cl = len(clones)
    if value == "reads":
        r = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)
        w = np.where(np.isfinite(r), r, 0.0)
        dtype = np.float64
    else:
        w = np.ones(n)
        dtype = np.float32
    X = sparse.coo_matrix(
        (w[has_c].astype(dtype), (ocode[has_c], ccode[has_c])), shape=(n_obs, n_cl), dtype=dtype
    ).tocsr()
    X.sum_duplicates()
    # obs
    obs = pd.DataFrame(index=pd.RangeIndex(n_obs))
    if bin_um is not None:
        obs["bin_x"] = okeys.bx.to_numpy()
        obs["bin_y"] = okeys.by.to_numpy()
        xy = np.column_stack([(okeys.bx.to_numpy() + 0.5) * bin_um, (okeys.by.to_numpy() + 0.5) * bin_um])
        unit_name = okeys.bx.astype(str) + "_" + okeys.by.astype(str)
        obs["bin_id"] = unit_name.to_numpy()
    else:
        obs[unit_col] = okeys.u.to_numpy()
        unit_name = okeys.u.astype(str)
        if x_key in air.columns and y_key in air.columns:
            if unit_col == spot_key:
                first = np.empty(n_obs, np.int64)
                first[ocode[::-1]] = np.arange(n)[::-1]
                xy = np.column_stack([air[x_key].to_numpy(float)[first], air[y_key].to_numpy(float)[first]])
            else:  # mean of the unit's distinct positions
                pos = pd.DataFrame({"o": ocode, "x": air[x_key].to_numpy(float), "y": air[y_key].to_numpy(float)})
                pos = pos.drop_duplicates().groupby("o")[["x", "y"]].mean().reindex(range(n_obs))
                xy = pos.to_numpy(dtype=float)
        else:
            xy = None
    if lib is not None:
        obs[lib] = okeys.lib.to_numpy()
    obs["n_records"] = np.bincount(ocode, minlength=n_obs).astype(np.int64)
    if reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        rr = air[reads_key].to_numpy(dtype=float)
        obs["n_reads"] = np.bincount(ocode, weights=np.where(np.isfinite(rr), rr, 0.0), minlength=n_obs)
    obs["n_clones"] = np.diff(X.indptr).astype(np.int64)
    if locus_key in air.columns:
        lc = pd.factorize(air[locus_key])[0]
        pairs = np.unique(ocode[lc >= 0] * np.int64(max(lc.max() + 1, 1)) + lc[lc >= 0])
        obs["n_loci"] = np.bincount(pairs // max(lc.max() + 1, 1), minlength=n_obs).astype(np.int64)
    for k in obs_keys:
        if k in air.columns:
            first = np.empty(n_obs, np.int64)
            first[ocode[::-1]] = np.arange(n)[::-1]
            col = air[k]
            obs[k] = _as_flag(col)[first] if k == "in_tissue" else col.to_numpy()[first]
    if xy is not None:
        obs[x_key], obs[y_key] = xy[:, 0], xy[:, 1]
    names = unit_name.to_numpy(dtype=object)
    if lib is not None and okeys.lib.nunique() > 1:
        names = (okeys.lib.astype(str) + ":" + unit_name).to_numpy(dtype=object)
    obs.index = pd.Index(names.astype(str), name=None)
    keep = obs["n_records"].to_numpy() >= min_records
    # var
    var = pd.DataFrame(index=pd.Index(np.asarray(clones).astype(str), name=None))
    cc = ccode[has_c]
    if locus_key in air.columns:
        var[locus_key] = _mode_by(air.loc[has_c, locus_key], cc, n_cl)
    for k in var_keys:
        if k in air.columns and k != clone_key:
            var[k] = _mode_by(air.loc[has_c, k], cc, n_cl)
    var["n_records"] = np.bincount(cc, minlength=n_cl).astype(np.int64)
    if reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        rr = air.loc[has_c, reads_key].to_numpy(dtype=float)
        var["n_reads"] = np.bincount(cc, weights=np.where(np.isfinite(rr), rr, 0.0), minlength=n_cl)
    Xk = X[keep]
    var["n_units"] = np.asarray((Xk > 0).sum(axis=0)).ravel().astype(np.int64)
    obs = obs[keep]
    adata = ad.AnnData(X=Xk, obs=obs, var=var)
    if xy is not None:
        adata.obsm["spatial"] = np.asarray(xy[keep], dtype=float)
    adata.uns["spaclone"] = {
        "aggregate": {
            "bin_um": bin_um,
            "unit": "bin" if bin_um is not None else unit_col,
            "value": value,
            "min_records": int(min_records),
        },
        "coordinates": {"unit": "um"},
    }
    return adata


class CellAssignment(NamedTuple):
    """Result of :func:`assign_to_cells` (unpack as ``air_cells, report = assign_to_cells(air, cells)``).

    Attributes
    ----------
    air
        Records with the cell id column (records outside every cell dropped unless ``drop_outside=False``).
    report
        Records, spots and clones inside / outside cells (one row per library, plus ``"all"``).
    """

    air: pd.DataFrame
    report: pd.DataFrame


def _polygons(cells: Any) -> tuple[list[str], list]:
    """Cell ids and polygon geometries (shapely geometries or (k, 2) vertex arrays)."""
    if hasattr(cells, "geometry") and hasattr(cells, "index"):  # GeoDataFrame / GeoSeries
        geoms = list(cells.geometry)
        ids = [str(i) for i in cells.index]
        return ids, geoms
    if isinstance(cells, Mapping):
        return [str(k) for k in cells], list(cells.values())
    raise TypeError("cells must be a GeoDataFrame, a mapping cell id -> polygon, or a 2-d label image")


def _assign_polygons(x: np.ndarray, y: np.ndarray, ids: list[str], geoms: list) -> np.ndarray:
    """Index into ``ids`` of the first polygon containing each point (boundary included), -1 outside."""
    out = np.full(len(x), -1, np.int64)
    if len(x) == 0 or len(geoms) == 0:
        return out
    try:
        import shapely
    except ImportError:  # pragma: no cover - shapely is optional
        shapely = None
    if shapely is not None:
        polys = [g if hasattr(g, "geom_type") else shapely.Polygon(np.asarray(g, dtype=float)) for g in geoms]
        tree = shapely.STRtree(polys)
        pts = shapely.points(x, y)
        pi, gi = tree.query(pts, predicate="intersects")
        if len(pi):
            o = np.lexsort((gi, pi))
            pi, gi = pi[o], gi[o]
            first = np.r_[True, pi[1:] != pi[:-1]]
            out[pi[first]] = gi[first]
        return out
    from matplotlib.path import Path  # pragma: no cover - fallback without shapely

    verts = [np.asarray(g.exterior.coords if hasattr(g, "exterior") else g, dtype=float) for g in geoms]
    order = np.argsort(x, kind="stable")
    xs = x[order]
    for k in range(len(verts) - 1, -1, -1):  # later polygons first so that earlier ones win overlaps
        v = verts[k]
        lo, hi = np.searchsorted(xs, v[:, 0].min(), "left"), np.searchsorted(xs, v[:, 0].max(), "right")
        cand = order[lo:hi]
        cand = cand[(y[cand] >= v[:, 1].min()) & (y[cand] <= v[:, 1].max())]
        if len(cand):
            inside = Path(v).contains_points(np.column_stack([x[cand], y[cand]]), radius=1e-9)
            out[cand[inside]] = k
    return out


def assign_to_cells(
    air: pd.DataFrame,
    cells: Any,
    *,
    cell_key: str = "cell_id",
    drop_outside: bool = True,
    pixel_um: float | None = None,
    origin_um: tuple[float, float] = (0.0, 0.0),
    per_library: bool = False,
    warn: bool = True,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    x_key: str = S.X,
    y_key: str = S.Y,
    library_key: str = S.LIBRARY,
) -> CellAssignment:
    """Assign each record to the segmented cell that contains its position.

    Parameters
    ----------
    air
        Receptor table with coordinates in µm.
    cells
        Cell segmentation, in µm: a GeoDataFrame / GeoSeries of polygons (index = cell id), a mapping cell id ->
        polygon (shapely geometry or ``(k, 2)`` vertex array), or a 2-d integer label image (0 = background) with
        ``pixel_um`` and ``origin_um`` (row = y, column = x).
    cell_key
        Column written with the cell id (string).
    drop_outside
        Drop records outside every cell (default, as in the authors' pipeline: reads outside cells are dropped).
    pixel_um, origin_um
        Pixel size and the µm position of pixel (0, 0)'s corner, for a label image.
    per_library
        ``cells`` is a mapping library -> segmentation, one per section (each library's records are assigned to its
        own cells).
    warn
        Emit :class:`CellAssignmentWarning` (assignment does not validate position).
    spot_key, clone_key, x_key, y_key, library_key
        Column names.

    Returns
    -------
    CellAssignment
        ``air``: records with ``cell_key``; ``report``: ``records``, ``records_in_cells``, ``records_outside``,
        ``share_in_cells``, ``spots_in_cells``, ``cells_with_records``, ``clones_in_cells`` per library and overall.

    Warns
    -----
    CellAssignmentWarning
        Cell assignment cannot validate position: a map of receptor-positive cells is drawn by cell typing and RNA
        amount and survives random placement (archive chapter 05). Test placement with ``spaclone.qc``.

    Notes
    -----
    A point on a polygon border belongs to the polygon; with overlapping polygons the first one (segmentation order)
    wins.

    References
    ----------
    Zhan et al. 2026, doi:10.1038/s41467-026-75405-x (Stereo-XCR-seq; its cell-level pipeline is examined in chapter
    05 of the archive); Marconato et al. 2025, doi:10.1038/s41592-024-02212-x (SpatialData aggregation of points by
    shapes).

    Examples
    --------
    >>> cells_air, report = assign_to_cells(air, polygons)  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    S.validate_air(air, required=(x_key, y_key), numeric=(x_key, y_key))
    if warn:
        warnings.warn(
            "assign_to_cells: putting receptor records into segmented cells cannot validate their position; in the "
            "archive (chapter 05) the map of BCR+ cells survived random placement because cell typing and RNA amount "
            "drew it. Test placement with spaclone.qc / spaclone.nulls.placebo before interpreting cell-level calls.",
            CellAssignmentWarning,
            stacklevel=2,
        )
    x = air[x_key].to_numpy(dtype=float)
    y = air[y_key].to_numpy(dtype=float)
    lab = np.full(len(air), None, dtype=object)
    lib = library_key if library_key in air.columns else None
    if per_library:
        if lib is None:
            raise KeyError(f"assign_to_cells: per_library=True needs column {library_key!r}")
        if not isinstance(cells, Mapping):
            raise TypeError("assign_to_cells: per_library=True needs a mapping library -> segmentation")
        segs = {str(k): v for k, v in cells.items()}
        lv_all = air[lib].astype(str).to_numpy()
        absent = sorted(set(lv_all) - set(segs))
        if absent:
            raise KeyError(f"assign_to_cells: no segmentation for libraries {absent}")
        groups = [(lv, lv_all == lv, segs[lv]) for lv in sorted(set(lv_all))]
    else:
        groups = [(None, np.ones(len(air), bool), cells)]
    for _, sel, seg in groups:
        idx = np.flatnonzero(sel)
        if isinstance(seg, np.ndarray) and seg.ndim == 2:
            if pixel_um is None or pixel_um <= 0:
                raise ValueError("assign_to_cells: a label image needs a positive pixel_um")
            ix = np.floor((x[idx] - origin_um[0]) / pixel_um).astype(np.int64)
            iy = np.floor((y[idx] - origin_um[1]) / pixel_um).astype(np.int64)
            inside = (ix >= 0) & (iy >= 0) & (ix < seg.shape[1]) & (iy < seg.shape[0])
            v = np.zeros(len(idx), dtype=seg.dtype)
            v[inside] = seg[iy[inside], ix[inside]]
            hit = inside & (v != 0)
            lab[idx[hit]] = v[hit].astype(str)
        else:
            ids, geoms = _polygons(seg)
            k = _assign_polygons(x[idx], y[idx], ids, geoms)
            ids_arr = np.asarray(ids, dtype=object)
            lab[idx[k >= 0]] = ids_arr[k[k >= 0]]
    inside = pd.notna(lab)
    out = air.copy()
    out[cell_key] = pd.Series(lab, index=air.index, dtype=object)
    rows = []
    libs = [("all", np.ones(len(air), bool))]
    if lib is not None:
        libs = [(lv, air[lib].astype(str).to_numpy() == lv) for lv in sorted(air[lib].astype(str).unique())] + libs
    for name, sel in libs:
        m = sel & inside
        rows.append(
            {
                "library": name,
                "records": int(sel.sum()),
                "records_in_cells": int(m.sum()),
                "records_outside": int((sel & ~inside).sum()),
                "share_in_cells": float(m.sum() / sel.sum()) if sel.sum() else float("nan"),
                "spots_in_cells": int(air.loc[m, spot_key].nunique()) if spot_key in air.columns else np.nan,
                "cells_with_records": int(pd.Series(lab[m]).nunique()),
                "clones_in_cells": int(air.loc[m, clone_key].nunique()) if clone_key in air.columns else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    if drop_outside:
        out = out[inside]
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "assign_to_cells": {
            "cell_key": cell_key,
            "drop_outside": drop_outside,
            "records_outside": int((~inside).sum()),
        },
    }
    return CellAssignment(out, report)
