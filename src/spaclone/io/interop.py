"""Interoperability with scirpy (AnnData + awkward ``obsm["airr"]``) and SpatialData.

* :func:`to_scirpy` packs a receptor table into scirpy's data structure: one observation per inferred cell (locus,
  default), spot or segmented cell, with all of its records as AIRR chains in the awkward array ``obsm["airr"]`` and
  its position in ``obsm["spatial"]`` (µm). scirpy then runs unchanged (``pp.index_chains``, ``pp.ir_dist``,
  ``tl.define_clonotypes``, ``tl.define_clonotype_clusters``, ``tl.clonal_expansion``, ``tl.alpha_diversity``).
  Every column of the table travels with the chains, so :func:`from_scirpy` restores the table exactly (values,
  dtypes, row order, index).
* :func:`from_scirpy` turns any scirpy AnnData (or MuData with an ``"airr"`` modality) back into a receptor table.
* :func:`to_spatialdata` builds a :class:`spatialdata.SpatialData` with the records as points, the spots as circles
  and the spot x clone counts (and the GEX) as tables.

scirpy, awkward, mudata and spatialdata are optional (``pip install spaclone[scverse]``) and imported on use.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = ["from_scirpy", "to_scirpy", "to_spatialdata"]

_ROW = "_spaclone_row"
_INDEX = "_spaclone_index"
_RAW = "_spaclone_raw_"
_META = "to_scirpy"


def _need(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{module} is needed for this function; install the optional dependencies with "
            f"'pip install spaclone[scverse]'"
        ) from e


def _dtype_info(s: pd.Series) -> dict:
    info: dict[str, Any] = {"dtype": str(s.dtype)}
    if isinstance(s.dtype, pd.CategoricalDtype):
        info["categories"] = [c.item() if hasattr(c, "item") else c for c in s.dtype.categories]
        info["ordered"] = bool(s.dtype.ordered)
        info["categories_dtype"] = str(s.dtype.categories.dtype)
    return info


def _restore_dtype(s: pd.Series, info: dict) -> pd.Series:
    """Cast a column read back from awkward to its original dtype (best effort for exotic dtypes)."""
    dt = info.get("dtype")
    if dt is None or str(s.dtype) == dt:
        return s
    try:
        if dt == "category":
            cats = pd.Index(info.get("categories", []))
            cdt = info.get("categories_dtype")
            if cdt and str(cats.dtype) != cdt:
                cats = cats.astype(cdt)
            as_text = s.astype(object).where(s.notna(), None)
            codes = pd.Categorical(
                [None if v is None else str(v) for v in as_text], categories=[str(c) for c in cats]
            ).codes
            cat = pd.Categorical.from_codes(codes, categories=cats, ordered=bool(info.get("ordered", False)))
            return pd.Series(cat, index=s.index)
        if dt.startswith("datetime64"):
            return pd.to_datetime(s).astype(dt)
        if dt == "object":
            return s.astype(object).where(s.notna(), None)
        return s.astype(dt)
    except (TypeError, ValueError):
        return s


def _to_ak(s: pd.Series):
    """One awkward 1-d array per column: numbers and booleans as such, everything else as (optional) strings."""
    ak = _need("awkward")
    import pyarrow as pa

    dt = s.dtype
    if pd.api.types.is_bool_dtype(dt) and not isinstance(dt, pd.CategoricalDtype):
        if s.isna().any():
            return ak.from_arrow(pa.array(s.astype(object).where(s.notna(), None).tolist(), type=pa.bool_()))
        return ak.from_numpy(s.to_numpy(dtype=bool))
    if pd.api.types.is_float_dtype(dt):
        return ak.from_numpy(s.to_numpy(dtype=float, na_value=np.nan))
    if pd.api.types.is_integer_dtype(dt):
        if s.isna().any():
            return ak.from_arrow(pa.array(s.astype(object).where(s.notna(), None).tolist(), type=pa.int64()))
        return ak.from_numpy(s.to_numpy(dtype=np.int64))
    vals = [None if (v is None or (not isinstance(v, str) and pd.isna(v))) else str(v) for v in s.astype(object)]
    return ak.from_arrow(pa.array(vals, type=pa.string()))


def _from_ak(arr) -> pd.Series:
    ak = _need("awkward")
    a = ak.to_arrow(arr, extensionarray=False)
    if hasattr(a, "combine_chunks"):
        a = a.combine_chunks()
    return pd.Series(a.to_pandas(), copy=False)


def _flag(values: pd.Series) -> np.ndarray:
    from spaclone.pp.metrics import _as_flag

    return _as_flag(values)


def to_scirpy(
    air: pd.DataFrame,
    *,
    level: Literal["locus", "spot", "cell"] = "locus",
    radius_um: float = 15.0,
    cell_key: str = "cell_id",
    airr_key: str = "airr",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
):
    """Convert a receptor table into scirpy's AnnData (AIRR chains in an awkward ``obsm["airr"]``).

    Parameters
    ----------
    air
        Receptor table.
    level
        Observation unit: ``"locus"`` (inferred cell: same-clone spots within ``radius_um`` merged by
        :func:`spaclone.pp.merge_loci`; one clone per observation, so ``obs["clone_id"]`` is exact and
        ``scirpy.tl.clonal_expansion`` counts loci per clone), ``"spot"`` (all records on a spot; a spot can hold many
        chains, which scirpy flags as multichain) or ``"cell"`` (``cell_key`` from
        :func:`spaclone.pp.assign_to_cells`; unassigned records are not allowed).
    radius_um
        Locus radius for ``level="locus"``.
    cell_key
        Cell column for ``level="cell"``.
    airr_key
        ``obsm`` key of the awkward array (scirpy's default ``"airr"``).
    spot_key, x_key, y_key, clone_key, locus_key, library_key
        Column names.
    productive_key
        Productive column. scirpy filters chains on a boolean ``productive`` field, so a non-boolean column (``"T"`` /
        ``"F"``) is converted (true for ``T``, ``TRUE``, 1, ``Y``) and its original values kept for the way back.
    reads_key
        Reads column (AIRR ``consensus_count``), one of scirpy's sort keys for primary chains.

    Returns
    -------
    anndata.AnnData
        No ``X`` (0 variables). ``obs``: the unit (spot / locus / cell columns, library), ``n_records``, ``n_clones``
        and, for loci, ``clone_id``, ``locus``, ``n_spots``, ``reads_max``. ``obsm[airr_key]``: one list of chains per
        observation, each chain carrying every column of the table. ``obsm["spatial"]``: µm (spot: its position;
        locus: centroid of its spots; cell: mean of its distinct record positions). ``uns["spaclone"]["to_scirpy"]``:
        a JSON string with what :func:`from_scirpy` needs to restore the table exactly.

    Notes
    -----
    scirpy's receptor model keeps at most two VJ and two VDJ chains per observation (``pp.index_chains``); the other
    chains stay in ``obsm["airr"]`` and come back with :func:`from_scirpy`. A locus or spot with more chains is
    flagged multichain, and ``tl.chain_qc`` sets its ``receptor_type`` to ``"multichain"``, so clonotype definitions
    grouped by receptor type (scirpy's default ``within_group``) keep such loci apart (on the StereoIR BCR table:
    4,938 of 103,894 loci). ``scirpy.tl.define_clonotypes`` compares nucleotide junctions (column ``junction``); for
    amino-acid clonotypes use ``scirpy.tl.define_clonotype_clusters(sequence="aa")``.

    References
    ----------
    Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy; since 0.13 the chains are AIRR Rearrangement
    records in an awkward array); Vander Heiden et al. 2018, doi:10.4049/jimmunol.1800669 (AIRR Rearrangement schema).

    Examples
    --------
    >>> adata = to_scirpy(air, level="locus")  # doctest: +SKIP
    >>> import scirpy as ir  # doctest: +SKIP
    >>> ir.tl.clonal_expansion(adata)  # doctest: +SKIP
    """
    ak = _need("awkward")
    import anndata as ad

    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if level not in ("locus", "spot", "cell"):
        raise ValueError(f"level must be 'locus', 'spot' or 'cell', got {level!r}")
    req = [spot_key, x_key, y_key, locus_key] + ([clone_key] if level == "locus" else [])
    req += [cell_key] if level == "cell" else []
    S.validate_air(air, required=req, numeric=(x_key, y_key))
    reserved = [c for c in air.columns if str(c).startswith("_spaclone_")]
    if reserved:
        raise ValueError(f"to_scirpy: column names starting with '_spaclone_' are reserved: {reserved}")
    n = len(air)
    lib = library_key if library_key in air.columns else None
    tab = air.reset_index(drop=True)
    obs: pd.DataFrame
    if level == "locus":
        from spaclone.pp.loci import merge_loci

        loci, member = merge_loci(
            tab,
            radius_um=radius_um,
            clone_key=clone_key,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            locus_key=locus_key,
            library_key=library_key,
            reads_key=reads_key,
        )
        code = member.to_numpy(np.int64)
        obs = loci.copy()
        obs.index = pd.Index([f"locus_{i}" for i in loci[S.LOCUS_ID]], dtype=object)
        xy = loci[[x_key, y_key]].to_numpy(dtype=float)
        obs = obs.drop(columns=[x_key, y_key])
    else:
        ucol = spot_key if level == "spot" else cell_key
        if tab[ucol].isna().any():
            raise ValueError(f"to_scirpy: column {ucol!r} has missing values (drop unassigned records first)")
        keys = ([lib] if lib is not None else []) + [ucol]
        g = tab.groupby(keys, sort=True, observed=True)
        code = g.ngroup().to_numpy(np.int64)
        okeys = g.size().reset_index()[keys]
        obs = okeys.copy()
        names = okeys[ucol].astype(str)
        if lib is not None and okeys[lib].nunique() > 1:
            names = okeys[lib].astype(str) + ":" + names
        obs.index = pd.Index(names.to_numpy(dtype=object))
        if level == "spot":
            first = np.empty(len(okeys), np.int64)
            first[code[::-1]] = np.arange(n)[::-1]
            xy = np.column_stack([tab[x_key].to_numpy(float)[first], tab[y_key].to_numpy(float)[first]])
        else:
            pos = pd.DataFrame({"o": code, "x": tab[x_key].to_numpy(float), "y": tab[y_key].to_numpy(float)})
            xy = pos.drop_duplicates().groupby("o")[["x", "y"]].mean().reindex(range(len(okeys))).to_numpy(float)
    n_obs = len(obs)
    obs["n_records"] = np.bincount(code, minlength=n_obs).astype(np.int64)
    if clone_key in tab.columns:
        cc = pd.factorize(tab[clone_key])[0]
        ok = cc >= 0
        pairs = np.unique(code[ok] * np.int64(max(cc.max() + 1, 1)) + cc[ok]) if ok.any() else np.empty(0, np.int64)
        obs["n_clones"] = np.bincount(pairs // max(cc.max() + 1, 1), minlength=n_obs).astype(np.int64)
    # chains: every column, plus the row position (and the index when it is not the default one)
    order = np.argsort(code, kind="stable")
    counts = np.bincount(code, minlength=n_obs)
    fields: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "level": level,
        "radius_um": radius_um,
        "columns": [str(c) for c in air.columns],
        "dtypes": {str(c): _dtype_info(air[c]) for c in air.columns},
        "raw": [],
        "index": None,
        "attrs": json.dumps(air.attrs, default=str),
        "keys": {"spot": spot_key, "x": x_key, "y": y_key, "clone": clone_key, "locus": locus_key},
    }
    for c in air.columns:
        s = tab[c]
        if c == productive_key and not (pd.api.types.is_bool_dtype(s.dtype) and not s.isna().any()):
            fields[_RAW + str(c)] = _to_ak(s.iloc[order])
            meta["raw"].append(str(c))
            s = pd.Series(_flag(s), index=s.index)
        fields[str(c)] = _to_ak(s.iloc[order])
    fields[_ROW] = ak.from_numpy(np.arange(n, dtype=np.int64)[order])
    if not (isinstance(air.index, pd.RangeIndex) and air.index.start == 0 and air.index.step == 1):
        idx = pd.Series(air.index.to_numpy(), dtype=air.index.dtype)
        fields[_INDEX] = _to_ak(idx.iloc[order])
        meta["index"] = {"name": air.index.name, **_dtype_info(idx)}
    records = ak.zip(fields, depth_limit=1)
    airr = ak.unflatten(records, counts)
    for c in list(obs.columns):
        if obs[c].dtype == object:
            if obs[c].isna().all():
                obs = obs.drop(columns=c)
            else:
                obs[c] = pd.Series(
                    obs[c].where(obs[c].notna(), None).to_numpy(dtype=object), index=obs.index, dtype=object
                )
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Awkward Arrays.*")
        adata = ad.AnnData(obs=obs, obsm={airr_key: airr, "spatial": np.asarray(xy, dtype=float)})
    adata.uns["spaclone"] = {_META: json.dumps(meta, default=str), "coordinates": {"unit": "um"}}
    from importlib.metadata import PackageNotFoundError, version

    try:
        adata.uns["scirpy_version"] = version("scirpy")
    except PackageNotFoundError:  # scirpy is optional; the tag is informative only
        pass
    return adata


def from_scirpy(
    adata: Any,
    *,
    airr_key: str = "airr",
    airr_mod: str = "airr",
    clone_col: str | None = "clone_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    require_spatial: bool = True,
) -> pd.DataFrame:
    """Convert scirpy's AnnData (or MuData) back into a receptor table.

    Parameters
    ----------
    adata
        AnnData with ``obsm[airr_key]``, or MuData whose ``airr_mod`` modality has it.
    airr_key, airr_mod
        Where the awkward array lives.
    clone_col
        ``obs`` column copied to ``clone_key`` for tables that did not come from :func:`to_scirpy` (e.g. after
        ``scirpy.tl.define_clonotypes``); ``None`` skips.
    spot_key, x_key, y_key, clone_key
        Output column names (tables from :func:`to_scirpy` keep their own).
    require_spatial
        For tables not made by :func:`to_scirpy`: raise when there is no ``obsm["spatial"]`` (otherwise the
        coordinates are NaN).

    Returns
    -------
    pandas.DataFrame
        One row per chain. A table made by :func:`to_scirpy` is restored exactly: same columns, values, dtypes, row
        order, index and ``attrs`` (after observations were subset in scirpy: the remaining rows, in their original
        order, labelled by their original index or row number). Otherwise the AIRR fields become columns, ``spot_key`` is the observation name
        (unless the chains carry one), ``x_um`` / ``y_um`` come from ``obsm["spatial"]`` and ``clone_key`` from
        ``obs[clone_col]`` when present.

    Examples
    --------
    >>> air_back = from_scirpy(adata)  # doctest: +SKIP
    """
    ak = _need("awkward")
    if hasattr(adata, "mod"):
        adata = adata.mod[airr_mod]
    if airr_key not in adata.obsm:
        raise KeyError(f"from_scirpy: no obsm[{airr_key!r}]")
    airr = adata.obsm[airr_key]
    counts = ak.to_numpy(ak.num(airr, axis=1)).astype(np.int64)
    flat = ak.flatten(airr, axis=1)
    fields = list(ak.fields(flat))
    cols = {f: _from_ak(flat[f]) for f in fields}
    meta_s = adata.uns.get("spaclone", {}).get(_META) if isinstance(adata.uns.get("spaclone"), dict) else None
    if meta_s is not None and _ROW in cols:
        meta = json.loads(meta_s)
        pos = cols[_ROW].to_numpy(np.int64)
        inv = np.argsort(pos, kind="stable")  # original row order (also after observations were subset)
        complete = bool(np.array_equal(pos[inv], np.arange(len(pos))))
        out = pd.DataFrame(index=pd.RangeIndex(len(pos)) if complete else pd.Index(pos[inv]))
        for c in meta["columns"]:
            if c in meta.get("raw", []) and (_RAW + c) in cols:
                s = cols[_RAW + c]
            else:
                s = cols[c]
            s = s.iloc[inv]
            s.index = out.index
            out[c] = _restore_dtype(s, meta["dtypes"][c])
        if meta.get("index") is not None and _INDEX in cols:
            idx = _restore_dtype(cols[_INDEX].iloc[inv].reset_index(drop=True), meta["index"])
            out.index = pd.Index(idx, name=meta["index"].get("name"))
        attrs = json.loads(meta.get("attrs") or "{}")
        out.attrs = attrs if isinstance(attrs, dict) else {}
        return out
    # a scirpy object from elsewhere
    out = pd.DataFrame({f: s for f, s in cols.items() if not f.startswith("_spaclone_")})
    obs_names = np.repeat(np.asarray(adata.obs_names, dtype=object), counts)
    if spot_key not in out.columns:
        out.insert(0, spot_key, obs_names)
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)
        out[x_key] = np.repeat(xy[:, 0], counts)
        out[y_key] = np.repeat(xy[:, 1], counts)
    elif require_spatial:
        raise ValueError("from_scirpy: no obsm['spatial']; pass require_spatial=False to get NaN coordinates")
    else:
        out[x_key] = np.nan
        out[y_key] = np.nan
    if clone_col is not None and clone_col in adata.obs.columns and clone_key not in out.columns:
        out[clone_key] = np.repeat(adata.obs[clone_col].astype(object).to_numpy(), counts)
    return out


def to_spatialdata(
    data: Any,
    gex: Any = None,
    *,
    spot_radius_um: float | None = None,
    bin_um: float | None = None,
    points: bool = True,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    library_key: str = S.LIBRARY,
):
    """Build a :class:`spatialdata.SpatialData` from a receptor table (and its GEX).

    Parameters
    ----------
    data
        Receptor table or :class:`~spaclone.io.SpatialAIR` (its ``gex`` is used when ``gex`` is None).
    gex
        Spot-level AnnData with ``obsm["spatial"]`` in µm.
    spot_radius_um
        Radius of the spot circles; default half the smallest distance between distinct spot positions (or half of
        ``bin_um``).
    bin_um
        Aggregate the clone table into square bins of this size (see :func:`spaclone.pp.aggregate`).
    points
        Include every record as a point (element ``"air_records"``).
    spot_key, x_key, y_key, clone_key, library_key
        Column names.

    Returns
    -------
    spatialdata.SpatialData
        Points ``"air_records"`` (record columns as features), shapes ``"air_spots"`` (circles of the receptor spots /
        bins) annotated by table ``"air_clones"`` (spot x clone counts, :func:`spaclone.pp.aggregate`); with GEX,
        shapes ``"gex_spots"`` annotated by table ``"gex"``. One coordinate system, ``"global"``, in µm.

    Raises
    ------
    ImportError
        When spatialdata is not installed.

    References
    ----------
    Marconato et al. 2025, doi:10.1038/s41592-024-02212-x (SpatialData).
    """
    _need("spatialdata")
    import warnings

    from spatialdata import SpatialData
    from spatialdata.models import PointsModel, ShapesModel, TableModel

    from spaclone.io._container import SpatialAIR
    from spaclone.pp.aggregate import aggregate

    if isinstance(data, SpatialAIR):
        air, gex = data.air, (gex if gex is not None else data.gex)
    else:
        air = data
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"to_spatialdata: expected a receptor table, got {type(air).__name__}")
    S.validate_air(air, required=(spot_key, x_key, y_key), numeric=(x_key, y_key))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = aggregate(
            air,
            bin_um=bin_um,
            clone_key=clone_key,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            library_key=library_key,
        )
        xy = np.asarray(tab.obsm["spatial"], dtype=float)
        if spot_radius_um is None:
            if bin_um is not None:
                spot_radius_um = bin_um / 2
            else:
                u = np.unique(xy, axis=0)
                if len(u) > 1:
                    from scipy.spatial import cKDTree

                    d, _ = cKDTree(u).query(u, k=2)
                    spot_radius_um = float(np.min(d[:, 1])) / 2
                else:
                    spot_radius_um = 5.0
        names = pd.Index(tab.obs_names.astype(str))
        shapes = {"air_spots": ShapesModel.parse(xy, geometry=0, radius=np.full(len(xy), spot_radius_um), index=names)}
        tab.obs["region"] = pd.Categorical(["air_spots"] * tab.n_obs)
        tab.obs["instance_id"] = names.to_numpy()
        tables = {
            "air_clones": TableModel.parse(tab, region="air_spots", region_key="region", instance_key="instance_id")
        }
        elements: dict[str, Any] = {}
        if points:
            df = air.reset_index(drop=True).copy()
            for c in df.columns:
                if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
                    df[c] = df[c].astype("string").fillna("").astype(str)
            elements["air_records"] = PointsModel.parse(df, coordinates={"x": x_key, "y": y_key})
        if gex is not None:
            if "spatial" not in gex.obsm:
                raise KeyError("to_spatialdata: gex has no obsm['spatial']")
            g = gex.copy()
            gxy = np.asarray(g.obsm["spatial"], dtype=float)[:, :2]
            gnames = pd.Index(g.obs_names.astype(str))
            shapes["gex_spots"] = ShapesModel.parse(
                gxy, geometry=0, radius=np.full(len(gxy), spot_radius_um), index=gnames
            )
            g.obs["region"] = pd.Categorical(["gex_spots"] * g.n_obs)
            g.obs["instance_id"] = gnames.to_numpy()
            tables["gex"] = TableModel.parse(g, region="gex_spots", region_key="region", instance_key="instance_id")
        sd = SpatialData(points=elements, shapes=shapes, tables=tables)
    return sd
