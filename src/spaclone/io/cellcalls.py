"""Per-cell receptor calls stored as AnnData ``obs`` columns (e.g. Stereo-XCR-seq LUAD cell bins, GEO GSE328481).

Processed spatial AIR data are often published as a cell (or bin) AnnData whose ``obs`` holds the final receptor call
of each cell, without reads or molecules. In GSE328481 (Zhan et al., *Nat Commun* 2026) every CellBin cell has
``IGHcdr3aa``, ``IGK/Lcdr3aa`` (light chain; its locus in ``IGKorIGL``), ``TRAcdr3aa`` and ``TRBcdr3aa`` (missing when
there is no call) and ``x`` / ``y`` in DNB units (0.5 µm). :func:`from_cell_calls` turns such columns into a receptor
table, one row per (cell, call).

Such aggregate data cannot carry molecule-level evidence: there are no reads, UMIs or per-molecule positions, and the
calls were made by a pipeline that already used cell types (in GSE328481: a per-CID vote, one clone per chain per
cell, and removal of calls in non-B / non-T cells). The reader therefore sets ``meta["molecule_level"] = False``; the
placement (placebo) tests of :mod:`spaclone.nulls` still apply.

References
----------
Zhan et al. 2026, doi:10.1038/s41467-026-75405-x (Stereo-XCR-seq LUAD, GEO GSE328481).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.io._container import SpatialAIR
from spaclone.io.airr import DNB_UM, as_scale, exact_clone_ids, finish, map_unique, str_values, unit_name

__all__ = ["STEREO_XCR_LUAD_CALLS", "STEREO_XCR_LUAD_LOCI", "from_cell_calls"]

logger = logging.getLogger("spaclone")

#: Call columns of the Stereo-XCR-seq LUAD h5ad files (GEO GSE328481).
STEREO_XCR_LUAD_CALLS: dict[str, str] = {
    "IGH": "IGHcdr3aa",
    "IGK/IGL": "IGK/Lcdr3aa",
    "TRA": "TRAcdr3aa",
    "TRB": "TRBcdr3aa",
}
#: The column that gives the locus of the mixed light-chain call in GSE328481.
STEREO_XCR_LUAD_LOCI: dict[str, str] = {"IGK/IGL": "IGKorIGL"}
_NOTE = (
    "per-cell calls of an upstream pipeline: no reads, UMIs or molecule positions, so molecule-level tests (same-spot "
    "GEX match, UMI match, read floor, swap rate) cannot be run; if the pipeline used cell types to assign calls, "
    "cell-type concordance is circular"
)


def _library(adata_path: Path | None, obs: pd.DataFrame, library_id: str | None) -> np.ndarray:
    if library_id is not None:
        return np.full(len(obs), str(library_id), dtype=object)
    if S.LIBRARY in obs.columns:
        return str_values(obs[S.LIBRARY])
    if adata_path is None:
        return np.full(len(obs), "cells", dtype=object)
    stem = adata_path.name.split(".")[0]
    return np.full(len(obs), stem.split("_", 1)[1] if stem.startswith("GSM") and "_" in stem else stem, dtype=object)


def from_cell_calls(
    adata: ad.AnnData | str | Path,
    *,
    calls: Mapping[str, str] = STEREO_XCR_LUAD_CALLS,
    locus_columns: Mapping[str, str] | None = STEREO_XCR_LUAD_LOCI,
    x_key: str = "x",
    y_key: str = "y",
    spatial_key: str | None = None,
    scale_um: float | tuple[float, float] = DNB_UM,
    cell_type_key: str | None = None,
    obs_columns: Sequence[str] = (),
    record_columns: Mapping[str, Sequence[str]] | None = None,
    keep_x: bool = True,
    library_id: str | None = None,
    technology: str = "cell calls",
) -> SpatialAIR:
    """Build a receptor table from per-cell call columns of an AnnData.

    Parameters
    ----------
    adata
        AnnData, or an h5ad path (read in backed mode when ``keep_x=False``), whose ``obs`` index identifies cells.
    calls
        ``{label: obs column}``; each column holds one junction (CDR3) amino-acid call per cell, missing or empty
        for no call. A label that is a locus (``"IGH"``, ``"TRB"``, ...) is the locus of its calls; any other label
        (e.g. ``"IGK/IGL"``) needs an entry in ``locus_columns``. The default is the GSE328481 layout
        (:data:`STEREO_XCR_LUAD_CALLS`).
    locus_columns
        ``{label: obs column}`` giving the locus of each cell's call for mixed-locus labels (default: GSE328481's
        ``IGKorIGL``).
    x_key, y_key
        ``obs`` columns with the cell coordinates in source units; ignored when ``spatial_key`` is given.
    spatial_key
        Use ``obsm[spatial_key][:, :2]`` as the coordinates instead.
    scale_um
        Micrometres per coordinate unit (0.5 for Stereo-seq DNB coordinates), or ``(x, y)``.
    cell_type_key
        ``obs`` column copied to ``cell_type`` on every record and in ``gex.obs`` (GSE328481: ``"anno"``).
    obs_columns
        Further ``obs`` columns copied onto every record of the cell.
    record_columns
        Per-label ``obs`` columns copied onto that label's records only, e.g. ``{"IGH": ["mutation_percent"]}``
        (other records get missing values).
    keep_x
        Keep the expression matrix in ``gex``; ``False`` returns the cells with ``obs`` and coordinates only.
    library_id
        Library (chip) label; default: ``obs["library_id"]`` if present, else the file stem without its ``GSM..._``
        prefix, else ``"cells"``.
    technology
        Free-text label stored in ``meta``.

    Returns
    -------
    SpatialAIR
        ``air``: one row per (cell, call) with ``spot_id`` (cell id), ``x_um``, ``y_um``, ``locus``, ``clone_id`` (the
        exact key ``locus|||junction_aa``: such data carry no V / J genes), ``junction_aa``, ``consensus_count`` (NaN:
        no read counts), ``library_id``, ``call`` (the label), ``x_src`` / ``y_src`` and the requested columns; no
        ``productive`` column (unknown). ``gex``: every cell (``obs`` copied, ``obsm["spatial"]`` in µm, the count
        matrix shared with ``adata`` when ``keep_x``). ``meta["level"] = "cell_call"`` and
        ``meta["molecule_level"] = False``: molecule-level tests do not apply.

    Notes
    -----
    Cell-call tables come out of a pipeline that has already assigned receptors to cells. A cell-type concordance
    computed on them is circular when the pipeline used cell types; the placebo tests of :mod:`spaclone.nulls` still
    apply.

    Examples
    --------
    >>> sa = from_cell_calls("GSM9684211_D06047A2.h5ad", cell_type_key="anno", keep_x=False)  # doctest: +SKIP
    >>> sa.meta["molecule_level"]  # doctest: +SKIP
    False
    """
    sx, sy = as_scale(scale_um)
    src = None
    if not isinstance(adata, ad.AnnData):
        src = Path(adata)
        adata = ad.read_h5ad(src, backed=None if keep_x else "r")
    try:
        obs = adata.obs
        if not obs.index.is_unique:
            raise ValueError("the AnnData obs index (cell ids) must be unique")
        if spatial_key is not None:
            xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
        else:
            if x_key not in obs.columns or y_key not in obs.columns:
                raise ValueError(f"obs lacks the coordinate columns {x_key!r} / {y_key!r}; pass spatial_key= instead")
            xy = obs[[x_key, y_key]].to_numpy(dtype=float)
        if not np.isfinite(xy).all():
            raise ValueError("cell coordinates contain missing values")
        locus_columns = dict(locus_columns or {})
        record_columns = dict(record_columns or {})
        lib = _library(src, obs, library_id)
        wanted = [*obs_columns, *(c for v in record_columns.values() for c in v)]
        if cell_type_key is not None:
            wanted.append(cell_type_key)
        for c in wanted:
            if c not in obs.columns:
                raise ValueError(f"obs column {c!r} not found")
        for label in record_columns:
            if label not in calls:
                raise ValueError(f"record_columns label {label!r} is not a key of calls")
        cell_ids = str_values(obs.index)
        parts = []
        for label, col in calls.items():
            if col not in obs.columns:
                raise ValueError(f"call column {col!r} not in obs; pass calls={{label: column}} for this data")
            val = map_unique(obs[col].to_numpy(dtype=object), lambda v: str(v).strip() or None)
            idx = np.flatnonzero(~pd.isna(val))
            if label in S.ALL_LOCI:
                locus = np.full(len(idx), label, dtype=object)
            elif label in locus_columns:
                lc = locus_columns[label]
                if lc not in obs.columns:
                    raise ValueError(f"locus column {lc!r} not in obs")
                locus = map_unique(obs[lc].to_numpy(dtype=object)[idx], lambda v: str(v).strip().upper() or None)
                bad = pd.isna(locus) | ~np.isin(locus.astype(str), list(S.ALL_LOCI))
                if bad.any():
                    raise ValueError(
                        f"{int(bad.sum())} calls in {col!r} have no valid locus in {lc!r} (e.g. {str(locus[bad][0])!r})"
                    )
            else:
                raise ValueError(f"call label {label!r} is not a locus; give its locus column in locus_columns")
            aa = val[idx]
            part = pd.DataFrame(
                {
                    S.SPOT: cell_ids[idx],
                    S.X: xy[idx, 0] * sx,
                    S.Y: xy[idx, 1] * sy,
                    S.LOCUS: locus,
                    S.CLONE: exact_clone_ids(locus, np.nan, np.nan, aa),
                    S.JUNCTION_AA: aa,
                    S.READS: np.nan,
                    S.LIBRARY: lib[idx],
                    "call": label,
                    "x_src": xy[idx, 0],
                    "y_src": xy[idx, 1],
                }
            )
            if cell_type_key is not None:
                part["cell_type"] = str_values(obs[cell_type_key].to_numpy(dtype=object)[idx])
            for c in obs_columns:
                part[c] = obs[c].to_numpy()[idx]
            for c in record_columns.get(label, ()):
                part[c] = obs[c].to_numpy()[idx]
            parts.append(part)
        air = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[S.SPOT, S.X, S.Y, S.LOCUS])
        if not parts:
            air = air.assign(**{S.CLONE: [], S.LIBRARY: []})
        for c in {c for v in record_columns.values() for c in v}:
            if c not in air.columns:
                air[c] = np.nan
        cells_obs = obs.copy()
        cells_obs.index = pd.Index(cell_ids, dtype=str)
        cells_obs[S.LIBRARY] = lib
        if cell_type_key is not None:
            cells_obs["cell_type"] = obs[cell_type_key].to_numpy()
        if keep_x:
            X = adata.X[:] if getattr(adata, "isbacked", False) else adata.X
            gex = ad.AnnData(
                X=X,
                obs=cells_obs,
                var=adata.var.copy(),
                obsm={"spatial": np.column_stack([xy[:, 0] * sx, xy[:, 1] * sy])},
            )
        else:
            gex = ad.AnnData(obs=cells_obs, obsm={"spatial": np.column_stack([xy[:, 0] * sx, xy[:, 1] * sy])})
        n_cells = int(adata.n_obs)
    finally:
        if src is not None and getattr(adata, "isbacked", False):
            adata.file.close()
    meta: dict[str, Any] = {
        "reader": "from_cell_calls",
        "technology": technology,
        "files": [str(src)] if src is not None else [],
        "library_id": str(lib[0]) if len(lib) else str(library_id or "cells"),
        "level": "cell_call",
        "molecule_level": False,
        "note": _NOTE,
        "source_unit": unit_name(sx, sy),
        "um_per_unit": sx if sx == sy else [sx, sy],
        "has_read_counts": False,
        "has_molecule_id": False,
        "calls": dict(calls),
        "n_cells": n_cells,
        "n_cells_with_call": int(air[S.SPOT].nunique()) if len(air) else 0,
    }
    meta["io"] = {
        "reader": "from_cell_calls",
        "calls": {str(k): str(v) for k, v in dict(calls).items()},
        "coordinates": f"obsm[{spatial_key!r}]" if spatial_key is not None else f"obs {x_key!r} / {y_key!r}",
        "um_per_unit": {"value": sx if sx == sy else [sx, sy], "rule": f"scale_um = {scale_um}"},
        "clone_rule": "exact key locus|||junction_aa (no V / J genes in cell calls)",
        "level": "cell_call (no reads, UMIs or molecule positions)",
    }
    logger.info("from_cell_calls: %d calls on %d of %d cells", len(air), meta["n_cells_with_call"], n_cells)
    return finish(air, gex, meta)
