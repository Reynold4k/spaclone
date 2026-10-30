"""Storage of a :class:`~spaclone.io.SpatialAIR` on disk: parquet for the records, h5ad for the GEX, JSON for meta.

``write(data, path)`` creates a directory with ``air.parquet`` (the receptor table, dtypes and index kept by
pyarrow), ``gex.h5ad`` (when there is a GEX) and ``meta.json`` (reader metadata, the table's ``attrs`` and a format
tag); ``read(path)`` returns the SpatialAIR. Values that JSON cannot hold are stored as text (tuples become lists).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spaclone.io._container import SpatialAIR

__all__ = ["read", "write"]

FORMAT = "spaclone-store"
VERSION = 1
_AIR, _GEX, _META = "air.parquet", "gex.h5ad", "meta.json"


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set | frozenset):
        return sorted(o, key=str)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def write(data: SpatialAIR | pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write a SpatialAIR (or a receptor table alone) to a directory.

    Parameters
    ----------
    data
        :class:`~spaclone.io.SpatialAIR` or a receptor table.
    path
        Directory to create (or to fill, if it exists and is empty or ``overwrite``).
    overwrite
        Replace the store's files if they exist.

    Returns
    -------
    pathlib.Path
        The directory.

    Examples
    --------
    >>> write(sp, "spleen.spaclone")  # doctest: +SKIP
    >>> sp2 = read("spleen.spaclone")  # doctest: +SKIP
    """
    if isinstance(data, pd.DataFrame):
        data = SpatialAIR(data, None, {})
    if not isinstance(data, SpatialAIR):
        raise TypeError(f"write: expected a SpatialAIR or a DataFrame, got {type(data).__name__}")
    p = Path(path)
    existing = [p / f for f in (_AIR, _GEX, _META) if (p / f).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"write: {p} already holds a spaclone store; pass overwrite=True")
    p.mkdir(parents=True, exist_ok=True)
    for f in existing:
        f.unlink()
    files = {}
    if data.air is not None:
        if not isinstance(data.air, pd.DataFrame):
            raise TypeError("write: SpatialAIR.air must be a DataFrame")
        air = data.air.copy()
        air.attrs = {}
        air.columns = [str(c) for c in air.columns]
        air.to_parquet(p / _AIR, index=True)
        files["air"] = _AIR
    if data.gex is not None:
        data.gex.write_h5ad(p / _GEX)
        files["gex"] = _GEX
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "files": files,
        "meta": dict(data.meta or {}),
        "air_attrs": dict(data.air.attrs) if data.air is not None else {},
    }
    (p / _META).write_text(json.dumps(meta, default=_jsonable, indent=1), encoding="utf-8")
    return p


def read(path: str | Path) -> SpatialAIR:
    """Read a store written by :func:`write`.

    Parameters
    ----------
    path
        The store directory.

    Returns
    -------
    SpatialAIR
        The receptor table (with its ``attrs``), the GEX (or ``None``) and the metadata.
    """
    import anndata as ad

    p = Path(path)
    mf = p / _META
    if not mf.exists():
        raise FileNotFoundError(f"read: {p} has no {_META}; not a spaclone store")
    meta = json.loads(mf.read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"read: {p} is not a spaclone store (format {meta.get('format')!r})")
    if int(meta.get("version", 0)) > VERSION:
        raise ValueError(f"read: store version {meta.get('version')} is newer than this spaclone ({VERSION})")
    files = meta.get("files", {})
    air = pd.read_parquet(p / files["air"]) if "air" in files else None
    if air is not None:
        air.attrs = dict(meta.get("air_attrs", {}))
    gex = ad.read_h5ad(p / files["gex"]) if "gex" in files else None
    return SpatialAIR(air, gex, dict(meta.get("meta", {})))
