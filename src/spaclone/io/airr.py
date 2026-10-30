"""AIRR rearrangement tables with spot coordinates (StereoIR / TRUST4, LongAIRR, barcode-keyed AIRR + positions).

The AIRR Rearrangement schema (1.4 / 2.0) has no coordinate fields, so spatial pipelines put the position in one of
three places, all read by :func:`read_airr`:

* inside ``cell_id`` as ``"x-y"``: the StereoIR demo (TRUST4 on Stereo-seq) writes one contig per row with
  ``cell_id = "8860-6440"``, the DNB (bin1, 0.5 µm) coordinates of its bin20 (10 µm) bin;
* in extra ``x`` / ``y`` columns (LongAIRR spatial output, one consensus molecule per row);
* in a separate positions table keyed by the barcode in ``cell_id`` (TRUST4 barcode mode on Visium or Slide-seq
  barcodes, with the platform's barcode-to-position file).

This module also holds the helpers shared by the other readers of :mod:`spaclone.io`: column typing, gene-call
cleaning, the exact clonotype key, spot-id formatting and the final assembly of a :class:`~spaclone.io.SpatialAIR`.

References
----------
Vander Heiden et al. 2018, doi:10.3389/fimmu.2018.02206 (AIRR Rearrangement schema); Song et al. 2021,
doi:10.1038/s41592-021-01142-2 (TRUST4).
"""

from __future__ import annotations

import logging
import re
import string
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import SpacloneWarning
from spaclone.io._container import SpatialAIR

__all__ = ["read_airr", "read_airr_frame"]

logger = logging.getLogger("spaclone")

#: Stereo-seq DNB pitch: one bin1 coordinate unit is 0.5 µm.
DNB_UM = S.DNB_UM
D_CALL = "d_call"
#: Columns every reader returns, in this order (``productive`` only when the source has the information).
CANONICAL = (
    S.SPOT,
    S.X,
    S.Y,
    S.LOCUS,
    S.CLONE,
    S.JUNCTION_AA,
    S.JUNCTION,
    S.V_CALL,
    D_CALL,
    S.J_CALL,
    S.C_CALL,
    S.PRODUCTIVE,
    S.READS,
    S.UMI,
    S.LIBRARY,
)
_STRING_COLUMNS = (S.SPOT, S.LOCUS, S.CLONE, S.JUNCTION_AA, S.JUNCTION, S.V_CALL, D_CALL, S.J_CALL, S.C_CALL, S.UMI)
#: AIRR fields read by default (more with ``columns=[...]`` or ``columns="all"``).
AIRR_DEFAULT = (
    "sequence_id",
    "cell_id",
    "locus",
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    "junction",
    "junction_aa",
    "productive",
    "consensus_count",
    "duplicate_count",
    "umi_count",
    "clone_id",
    "umi",
)
_NUMERIC_AIRR = ("consensus_count", "duplicate_count", "umi_count")
_MISSING_CALLS = frozenset(("", "*", "NA", "N/A", "None", "nan", "NaN", "region_not_covered"))
_SCORE = re.compile(r"\([^)]*\)$")
_TRUE = ("t", "true", "1", "yes", "y")
_FALSE = ("f", "false", "0", "no", "n")
_PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
#: Meta keys copied to ``air.attrs["spaclone"]`` and ``gex.uns["spaclone"]``.
_LITE = ("reader", "technology", "library_id", "level", "source_unit", "um_per_unit", "molecule_level")


# ------------------------------------------------------------------------------------------------- shared helpers


def str_values(values: Any) -> np.ndarray:
    """Object array of ``str`` with ``nan`` for missing values (pandas infers its native string dtype from it).

    Non-string values are converted with ``str``; the conversion runs once per distinct value, so long columns with
    few distinct values are cheap.
    """
    if isinstance(values, pd.Series | pd.Index):
        arr = values.to_numpy(dtype=object, na_value=np.nan)
    else:
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            arr = arr.reshape(1)
    miss = pd.isna(arr)
    if miss.any():
        arr = arr.copy()
        arr[miss] = np.nan
    if pd.api.types.infer_dtype(arr, skipna=True) not in ("string", "empty"):
        arr = map_unique(arr, str)
    return arr


def map_unique(values: Any, fn: Any) -> np.ndarray:
    """Apply ``fn`` (value → value or ``None``/NaN for missing) once per distinct non-missing value.

    Parameters
    ----------
    values
        Array-like.
    fn
        Python function of one value.

    Returns
    -------
    numpy.ndarray
        Object array, ``nan`` where ``values`` is missing or ``fn`` returned ``None``.
    """
    codes, uniq = pd.factorize(np.asarray(values, dtype=object).ravel(), use_na_sentinel=True)
    mapped = np.array([fn(u) for u in uniq] + [np.nan], dtype=object)
    mapped[pd.isna(mapped)] = np.nan
    return mapped[np.where(codes >= 0, codes, len(uniq))]


def to_bool(values: Any) -> pd.Series:
    """Map ``T`` / ``F`` / ``TRUE`` / ``FALSE`` / ``1`` / ``0`` / ``yes`` / ``no`` (any case) to booleans.

    Parameters
    ----------
    values
        Array-like of flags.

    Returns
    -------
    pandas.Series
        ``bool`` when every value is recognised, else nullable ``boolean`` with ``<NA>`` for missing or unrecognised
        values (``RangeIndex``).
    """
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_bool_dtype(s.dtype) and not s.isna().any():
        return s.astype(bool)

    def flag(v: Any) -> Any:
        t = str(v).strip().lower()
        return True if t in _TRUE else (False if t in _FALSE else None)

    m = map_unique(s.to_numpy(dtype=object, na_value=np.nan), flag)
    out = pd.Series(pd.array([None if pd.isna(v) else bool(v) for v in m], dtype="boolean"))
    return out.astype(bool) if not out.isna().any() else out


def _clean_one(v: Any, first: bool, strip_score: bool) -> Any:
    t = str(v)
    if first:
        t = t.split(",", 1)[0]
    if strip_score:
        t = _SCORE.sub("", t)
    t = t.strip()
    return None if t in _MISSING_CALLS else t


def clean_call(values: Any, *, first: bool = True, strip_score: bool = True) -> np.ndarray:
    """Normalise gene calls; missing markers (``""``, ``"*"``, ``"NA"``, ...) become missing.

    Parameters
    ----------
    values
        Array-like of V / D / J / C calls.
    first
        Keep only the first of several comma-separated calls (``"IGHV1-26*01,IGHV1-18*01"`` → ``"IGHV1-26*01"``).
    strip_score
        Remove a trailing MiXCR hit score (``"IGHV1-26*00(1570)"`` → ``"IGHV1-26*00"``).

    Returns
    -------
    numpy.ndarray
        Object array of ``str`` (``nan`` = no call); alleles are kept.
    """
    return map_unique(values, lambda v: _clean_one(v, first, strip_score))


def gene_of(calls: Any) -> np.ndarray:
    """Gene of a call, allele removed: ``"IGHV1-26*01"`` → ``"IGHV1-26"`` (first call, score stripped)."""

    def gene(v: Any) -> Any:
        t = _clean_one(v, True, True)
        return None if t is None else t.split("*", 1)[0]

    return map_unique(calls, gene)


def locus_from_call(calls: Any) -> np.ndarray:
    """Locus from the first three letters of a V (or J, C) call when they name a locus, else missing."""

    def loc(v: Any) -> Any:
        t = _clean_one(v, True, True)
        return t[:3].upper() if t is not None and t[:3].upper() in S.ALL_LOCI else None

    return map_unique(calls, loc)


def exact_clone_ids(locus: Any, v_call: Any, j_call: Any, junction_aa: Any) -> np.ndarray:
    """Exact clonotype key ``locus|V gene|J gene|junction_aa`` (alleles removed; empty field for a missing gene).

    This is the "exact" rule of the archive (T1 / TR1) and of :func:`spaclone.pp.define_clones` with
    ``method="exact"``; records without a junction get a missing id.

    Parameters
    ----------
    locus, v_call, j_call, junction_aa
        Array-likes of equal length (a scalar is broadcast).

    Returns
    -------
    numpy.ndarray
        Object array of ``str``, ``nan`` where the junction is missing or empty.
    """
    arrs = [np.asarray(a, dtype=object) for a in (locus, v_call, j_call, junction_aa)]
    n = max((a.size for a in arrs if a.ndim), default=1)
    arrs = [np.full(n, a.item(), dtype=object) if a.ndim == 0 else a.ravel() for a in arrs]
    loc = str_values(arrs[0])
    v, j = gene_of(arrs[1]), gene_of(arrs[2])
    aa = str_values(arrs[3])
    out = np.empty(n, dtype=object)
    for i, (a, b, c, d) in enumerate(zip(loc, v, j, aa, strict=True)):
        out[i] = (
            np.nan
            if not isinstance(d, str) or d == ""
            else f"{a if isinstance(a, str) else ''}|{b if isinstance(b, str) else ''}|"
            f"{c if isinstance(c, str) else ''}|{d}"
        )
    return out


def format_spots(x: Any, y: Any, spot_format: str = "{x}_{y}") -> np.ndarray:
    """Spot ids from integer coordinates: a vectorised ``spot_format.format(x=..., y=...)``.

    Parameters
    ----------
    x, y
        Integer-valued coordinates.
    spot_format
        Template with the plain fields ``{x}`` and ``{y}``, e.g. ``"{x}_{y}"`` or ``"{x}-{y}"``.

    Returns
    -------
    numpy.ndarray
        Object array of ``str``.
    """
    fields = [f for _, f, _, _ in string.Formatter().parse(spot_format) if f is not None]
    for _, f, spec, conv in string.Formatter().parse(spot_format):
        if f is not None and (f not in ("x", "y") or spec or conv):
            raise ValueError(f"spot_format may only contain the plain fields {{x}} and {{y}}, got {spot_format!r}")
    if not fields:
        raise ValueError(f"spot_format needs {{x}} and/or {{y}}, got {spot_format!r}")
    xs = np.asarray(x).astype(np.int64).tolist()
    ys = np.asarray(y).astype(np.int64).tolist()
    return np.array([spot_format.format(x=a, y=b) for a, b in zip(xs, ys, strict=True)], dtype=object)


def modal_codes(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Per group, the most frequent value code; ties go to the smallest code; ``-1`` codes are ignored.

    Parameters
    ----------
    groups
        Non-negative group codes, one per row.
    values
        Value codes, one per row (``-1`` = missing). With codes sorted like the values (``pd.factorize(...,
        sort=True)``), ties resolve to the smallest value, as :meth:`pandas.Series.mode` does.
    n_groups
        Number of groups.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length ``n_groups``; ``-1`` for groups without a valid value.
    """
    g = np.asarray(groups, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)
    ok = v >= 0
    g, v = g[ok], v[ok]
    out = np.full(n_groups, -1, dtype=np.int64)
    if len(g) == 0:
        return out
    pair, cnt = np.unique(np.column_stack([g, v]), axis=0, return_counts=True)
    pg, pv = pair[:, 0], pair[:, 1]
    order = np.lexsort((pv, -cnt, pg))  # by group, then count descending, then value ascending
    pg, pv = pg[order], pv[order]
    first = np.r_[True, pg[1:] != pg[:-1]]
    out[pg[first]] = pv[first]
    return out


def _h5_safe(v: Any) -> Any:
    """Recursively drop ``None`` values and cast numpy scalars so that ``uns`` can be written to h5ad."""
    if isinstance(v, dict):
        return {str(k): _h5_safe(x) for k, x in v.items() if x is not None}
    if isinstance(v, list | tuple):
        return [_h5_safe(x) for x in v if x is not None]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, int) and not isinstance(v, bool) and not -(2**63) <= v < 2**63:
        return str(v)  # e.g. a 128-bit seed entropy: HDF5 integers are at most 64-bit
    return v


def lite_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance keys stored in ``air.attrs["spaclone"]`` and ``gex.uns["spaclone"]``."""
    return {k: meta[k] for k in _LITE if k in meta and meta[k] is not None}


def finish(air: pd.DataFrame, gex: ad.AnnData | None, meta: dict[str, Any]) -> SpatialAIR:
    """Order and type the columns, attach the provenance, validate and return a :class:`SpatialAIR`.

    Parameters
    ----------
    air
        Receptor table with at least ``spot_id``, ``x_um``, ``y_um``, ``locus``, ``clone_id`` and ``library_id``.
    gex
        Spot-level AnnData or ``None``.
    meta
        Reader metadata; must contain ``source_unit`` and ``um_per_unit``.

    Returns
    -------
    SpatialAIR
        String columns are pandas strings (missing = ``NaN``), ``x_um`` / ``y_um`` / ``consensus_count`` are
        ``float64``, ``productive`` (when present) is ``bool`` or nullable ``boolean``.

    Raises
    ------
    ValueError
        If ``spot_id``, ``locus`` or ``clone_id`` has missing values or the coordinates are not finite.
    """
    if "source_unit" not in meta or "um_per_unit" not in meta:
        raise ValueError("reader metadata must record source_unit and um_per_unit")
    air = air.reset_index(drop=True)
    for c in CANONICAL:
        if c not in air.columns and c != S.PRODUCTIVE:
            air[c] = np.nan
    for c in (*_STRING_COLUMNS, S.LIBRARY):
        air[c] = str_values(air[c])
        if _PANDAS3 and air[c].dtype == object:  # all missing: keep the string dtype of the other columns
            air[c] = air[c].astype("str")
    air[S.X] = air[S.X].to_numpy(dtype=np.float64)
    air[S.Y] = air[S.Y].to_numpy(dtype=np.float64)
    air[S.READS] = pd.to_numeric(air[S.READS], errors="coerce").astype(np.float64)
    if S.PRODUCTIVE in air.columns and not pd.api.types.is_bool_dtype(air[S.PRODUCTIVE].dtype):
        air[S.PRODUCTIVE] = flag_array(air[S.PRODUCTIVE]) if len(air) else np.zeros(0, bool)
    rest = [c for c in air.columns if c not in CANONICAL]
    air = air[[c for c in CANONICAL if c in air.columns] + rest]
    S.validate_air(air)
    for c in (S.SPOT, S.LOCUS, S.CLONE, S.LIBRARY):
        if air[c].isna().any():
            raise ValueError(f"column {c!r} has {int(air[c].isna().sum())} missing values")
    meta.setdefault("n_records", int(len(air)))
    meta.setdefault("n_spots", int(air[S.SPOT].nunique()))
    meta.setdefault("records_by_locus", {str(k): int(v) for k, v in air[S.LOCUS].value_counts().sort_index().items()})
    air.attrs["spaclone"] = lite_meta(meta)
    if gex is not None:
        gex.uns["spaclone"] = _h5_safe(
            {k: v for k, v in meta.items() if isinstance(v, str | int | float | bool | list | dict)}
        )
    return SpatialAIR(air=air, gex=gex, meta=meta)


def spot_table(
    spot_ids: Sequence[str] | pd.Index | np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    obs: pd.DataFrame | None = None,
) -> ad.AnnData:
    """A spot-level AnnData without genes: ``obs`` index = spot id, ``obsm["spatial"]`` in µm.

    Parameters
    ----------
    spot_ids
        Unique spot ids.
    x_um, y_um
        Spot coordinates in micrometres.
    obs
        Optional per-spot columns (same row order as ``spot_ids``).

    Returns
    -------
    anndata.AnnData
        ``n_obs`` = number of spots, ``n_vars`` = 0.
    """
    idx = pd.Index(str_values(spot_ids), dtype=str, name=None)
    if not idx.is_unique:
        raise ValueError("spot ids of the spot table must be unique")
    o = pd.DataFrame(index=idx) if obs is None else obs.reset_index(drop=True).set_axis(idx, axis=0)
    a = ad.AnnData(obs=o)
    a.obsm["spatial"] = np.column_stack([np.asarray(x_um, float), np.asarray(y_um, float)])
    return a


def split_xy(ids: Any, sep: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``"x<sep>y"`` spot ids into two float arrays; raises ``ValueError`` naming the first bad id."""
    uniq, inv = np.unique(np.asarray(ids, dtype=str), return_inverse=True)
    parts = pd.Series(uniq, dtype=object).str.split(sep, regex=False)
    n = parts.str.len().to_numpy()
    xs = pd.to_numeric(parts.str[0], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(parts.str[1], errors="coerce").to_numpy(dtype=float)
    bad = (n != 2) | ~np.isfinite(xs) | ~np.isfinite(ys)
    if bad.any():
        raise ValueError(
            f"cell_id {str(uniq[np.flatnonzero(bad)[0]])!r} is not of the form 'x{sep}y'; pass coords=('x', 'y') if "
            "the "
            "coordinates are in columns, a positions table, or cell_id_sep= for another separator"
        )
    return xs[inv], ys[inv]


def as_scale(scale_um: float | Sequence[float]) -> tuple[float, float]:
    """Micrometres per source unit along x and y (a scalar applies to both); both must be positive."""
    sx, sy = (scale_um, scale_um) if np.ndim(scale_um) == 0 else tuple(scale_um)  # type: ignore[misc]
    sx, sy = float(sx), float(sy)
    if not (sx > 0 and sy > 0 and np.isfinite(sx) and np.isfinite(sy)):
        raise ValueError(f"scale_um must be positive and finite, got {scale_um!r}")
    return sx, sy


def unit_name(sx: float, sy: float, default: str = "coordinate unit") -> str:
    """Human-readable source unit: Stereo-seq DNB for 0.5 µm, else ``default``."""
    if sx == sy == DNB_UM:
        return "Stereo-seq DNB (bin1, 0.5 µm)"
    return default


def _as_paths(path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(path, str | Path):
        return [Path(path)]
    return [Path(p) for p in path]


def _read_list(source: str | Path | Iterable[str]) -> list[str]:
    if isinstance(source, str | Path):
        return Path(source).read_text().split()
    return [str(s) for s in source]


def _labels(cell_types: Mapping[str, str | Path | Iterable[str]]) -> pd.Series:
    """Spot id → class label from spot lists; a spot in two lists raises ``ValueError``."""
    lab: dict[str, str] = {}
    for label, src in cell_types.items():
        for sid in _read_list(src):
            if lab.get(sid, label) != label:
                raise ValueError(f"spot {sid!r} is listed as both {lab[sid]!r} and {label!r}")
            lab[sid] = str(label)
    return pd.Series(lab, dtype=object)


def flag_array(values: Any) -> Any:
    """Booleans of :func:`to_bool` as a column value: ``bool`` ndarray if complete, else a nullable BooleanArray."""
    b = to_bool(values)
    return b.to_numpy(dtype=bool) if b.dtype == bool else b.array


# ------------------------------------------------------------------------------------------------------- reader


def _read_one_tsv(
    path: Path, columns: Sequence[str] | Literal["all"] | None, extra: set[str], numeric_extra: set[str]
) -> pd.DataFrame:
    head = pd.read_csv(path, sep="\t", nrows=0)
    have = [c for c in head.columns if not str(c).startswith("Unnamed:")]
    if columns == "all":
        use = have
    else:
        want = set(AIRR_DEFAULT if columns is None else columns) | extra
        use = [c for c in have if c in want]
    numeric = set(_NUMERIC_AIRR) | numeric_extra
    dtype = {c: str for c in use if c not in numeric}
    return pd.read_csv(path, sep="\t", usecols=use, dtype=dtype, keep_default_na=False, na_values=[""])


def _positions(positions: pd.DataFrame) -> pd.DataFrame:
    for c in ("x", "y"):
        if c not in positions.columns:
            raise ValueError(f"the positions table lacks column {c!r} (source units)")
    pos = positions.set_axis(pd.Index(str_values(positions.index), dtype=str))
    if not pos.index.is_unique:
        raise ValueError("barcodes of the positions table must be unique")
    if not (np.isfinite(pos["x"].to_numpy(float)).all() and np.isfinite(pos["y"].to_numpy(float)).all()):
        raise ValueError("the positions table has missing coordinates")
    return pos


def read_airr(
    path: str | Path | Sequence[str | Path],
    *,
    coords: Literal["cell_id"] | tuple[str, str] | pd.DataFrame = "cell_id",
    scale_um: float | tuple[float, float] = DNB_UM,
    cell_id_sep: str = "-",
    spot_key: str | None = None,
    cell_types: Mapping[str, str | Path | Iterable[str]] | None = None,
    columns: Sequence[str] | Literal["all"] | None = None,
    clone_key: str | None = None,
    library_id: str | Sequence[str] | None = None,
    technology: str = "AIRR",
) -> SpatialAIR:
    """Read AIRR rearrangement TSV(s) whose records carry a spot position.

    Parameters
    ----------
    path
        One AIRR TSV (optionally gzipped), or several (e.g. the BCR and the TCR table of one section), which are
        concatenated. A leading unnamed index column (as in the StereoIR demo files) is ignored.
    coords
        Where the position is:

        * ``"cell_id"``: inside ``cell_id`` as ``"x<sep>y"`` (StereoIR: ``"8860-6440"``, the DNB coordinates of the
          bin; no half-bin shift is applied);
        * a pair of column names, e.g. ``("x", "y")`` (LongAIRR);
        * a positions table: a ``DataFrame`` indexed by the barcodes found in ``cell_id``, with columns ``x`` and
          ``y`` in source units and optionally ``in_tissue``. Every barcode of the table, with or without receptor
          records, becomes a spot of ``gex``, so off-tissue barcodes are kept; records whose barcode is not in the
          table cannot be placed and are left out (counted in ``meta["n_records_unpositioned"]``).
    scale_um
        Micrometres per source coordinate unit: one value, or ``(x, y)`` for anisotropic grids. The default 0.5 is
        the Stereo-seq DNB pitch (StereoIR, TRUST4 on Stereo-seq CIDs).
    cell_id_sep
        Separator between x and y in ``cell_id`` (``coords="cell_id"``).
    spot_key
        Column with the spot id when ``coords`` is a pair of columns (e.g. LongAIRR's ``"spbcid"``). Default:
        ``cell_id`` if present, else ``"<x>_<y>"``.
    cell_types
        Optional spot lists by class, e.g. ``{"B lineage": "mm_spleen_bcell.tsv", "T": "mm_spleen_tcell.tsv"}``
        (files of whitespace-separated spot ids, or iterables of ids). Adds ``cell_type`` to the receptor table
        (``"untyped"`` for spots in no list). A spot in two lists raises ``ValueError``.
    columns
        AIRR fields to read: ``None`` reads ``sequence_id``, ``cell_id``, ``locus``, the V / D / J / C calls,
        ``junction``, ``junction_aa``, ``productive``, ``consensus_count``, ``duplicate_count``, ``umi_count``,
        ``clone_id`` and ``umi`` when present (``AIRR_DEFAULT`` in ``spaclone.io.airr``), plus the coordinate, spot
        and clone columns; ``"all"`` reads every field; or a list.
    clone_key
        Clone definition. ``None``: the AIRR ``clone_id`` column when present and complete, else the exact clonotype
        ``locus|V gene|J gene|junction_aa`` (a partly filled ``clone_id`` falls back to the exact key with a
        warning). ``"exact"``: always the exact key. Any other value: the name of a column holding clone ids. A
        ``clone_id`` column that is not used is kept as ``clone_id_airr``.
    library_id
        Library (section) label: one for all files, one per file, or ``None`` for the stem of the first file. Spot
        ids are unique within a library only.
    technology
        Free-text technology label stored in ``meta``.

    Returns
    -------
    SpatialAIR
        ``air``: one row per AIRR record with ``spot_id``, ``x_um``, ``y_um``, ``locus``, ``clone_id``,
        ``junction_aa``, ``junction``, ``v_call``, ``d_call``, ``j_call``, ``c_call`` (TRUST4's ``"*"`` = no call
        becomes missing), ``productive`` (only when the files have it), ``consensus_count`` (reads; else
        ``duplicate_count``; NaN when neither exists), ``umi``, ``library_id``, then ``x_src`` / ``y_src`` (source
        units), ``sequence_id``, ``cell_type`` / ``in_tissue`` when known, and any further requested field.
        ``gex``: ``None`` unless ``cell_types`` or a positions table is given; then a spot AnnData without genes
        holding every typed or positioned spot and every receptor-carrying spot (sorted by id), with
        ``obs["n_records"]``, ``obs["library_id"]``, ``obs["cell_type"]`` / ``obs["in_tissue"]`` when known and
        ``obsm["spatial"]`` in µm. ``meta``: technology, files, source unit, ``um_per_unit``, ``level``
        (``"molecule"`` when a ``umi`` column is filled, else ``"contig"``), clone rule and counts.

    Examples
    --------
    >>> sa = read_airr(
    ...     ["mm_spleen_bcr_airr.tsv", "mm_spleen_tcr_airr.tsv"],
    ...     cell_types={"B lineage": "mm_spleen_bcell.tsv", "T": "mm_spleen_tcell.tsv"},
    ...     library_id="mm_spleen",
    ... )  # doctest: +SKIP
    >>> sa.air[["spot_id", "x_um", "y_um", "locus", "clone_id"]].head()  # doctest: +SKIP
    """
    sx, sy = as_scale(scale_um)
    paths = _as_paths(path)
    if not paths:
        raise ValueError("no AIRR file given")
    pos = _positions(coords) if isinstance(coords, pd.DataFrame) else None
    if pos is None and isinstance(coords, str) and coords != "cell_id":
        raise ValueError("coords must be 'cell_id', a pair of column names or a positions DataFrame")
    col_coords = None if isinstance(coords, str | pd.DataFrame) else tuple(coords)
    if col_coords is not None and len(col_coords) != 2:
        raise ValueError("coords must name exactly two columns (x, y)")
    extra = set(col_coords or ()) | {c for c in (spot_key, clone_key) if c not in (None, "exact")}
    if library_id is None or isinstance(library_id, str):
        libs = [library_id or paths[0].name.split(".")[0]] * len(paths)
    else:
        libs = [str(v) for v in library_id]
        if len(libs) != len(paths):
            raise ValueError(f"library_id has {len(libs)} labels for {len(paths)} files")
    frames = [_read_one_tsv(p, columns, extra, set(col_coords or ())) for p in paths]
    lib_col = np.repeat(np.asarray(libs, dtype=object), [len(f) for f in frames])
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return _assemble(df, lib_col, libs, [str(p) for p in paths], pos, col_coords, sx, sy, cell_id_sep, spot_key,
                     cell_types, clone_key, technology)  # fmt: skip


def _frame_columns(
    table: pd.DataFrame, columns: Sequence[str] | Literal["all"] | None, extra: set[str], numeric_extra: set[str]
) -> pd.DataFrame:
    """An in-memory AIRR table typed as :func:`_read_one_tsv` types a file (strings, AIRR counts numeric)."""
    have = [str(c) for c in table.columns if not str(c).startswith("Unnamed:")]
    want = set(have) if columns == "all" else set(AIRR_DEFAULT if columns is None else columns) | extra
    use = [c for c in have if c in want]
    df = table.loc[:, use].copy() if use else pd.DataFrame(index=table.index)
    df.columns = use
    numeric = set(_NUMERIC_AIRR) | numeric_extra
    for c in use:
        if c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            v = df[c].to_numpy(dtype=object)
            miss = pd.isna(v) | (v == "")
            out = np.where(miss, np.nan, v).astype(object)
            df[c] = str_values(out) if (~miss).any() else out
    return df.reset_index(drop=True)


def read_airr_frame(
    table: pd.DataFrame,
    *,
    coords: Literal["cell_id"] | tuple[str, str] | pd.DataFrame = "cell_id",
    scale_um: float | tuple[float, float] = DNB_UM,
    cell_id_sep: str = "-",
    spot_key: str | None = None,
    columns: Sequence[str] | Literal["all"] | None = None,
    clone_key: str | None = None,
    library_id: str = "airr",
    technology: str = "AIRR",
    source: str = "<DataFrame>",
) -> SpatialAIR:
    """:func:`read_airr` on a table already in memory (AIRR column names; ``cell_id`` holds the spot / cell id).

    Parameters
    ----------
    table
        AIRR-named records, e.g. from :func:`spaclone.io.read_receptors`.
    coords, scale_um, cell_id_sep, spot_key, columns, clone_key, technology
        As in :func:`read_airr`.
    library_id
        Library label of every record.
    source
        Text recorded as the file in ``meta["files"]``.

    Returns
    -------
    SpatialAIR
        Exactly what :func:`read_airr` returns for the same table written to a TSV.
    """
    sx, sy = as_scale(scale_um)
    pos = _positions(coords) if isinstance(coords, pd.DataFrame) else None
    if pos is None and isinstance(coords, str) and coords != "cell_id":
        raise ValueError("coords must be 'cell_id', a pair of column names or a positions DataFrame")
    col_coords = None if isinstance(coords, str | pd.DataFrame) else tuple(coords)
    extra = set(col_coords or ()) | {c for c in (spot_key, clone_key) if c not in (None, "exact")}
    df = _frame_columns(table, columns, extra, set(col_coords or ()))
    lib_col = np.repeat(np.asarray([library_id], dtype=object), len(df))
    return _assemble(df, lib_col, [library_id], [source], pos, col_coords, sx, sy, cell_id_sep, spot_key, None,
                     clone_key, technology)  # fmt: skip


def _assemble(
    df: pd.DataFrame,
    lib_col: np.ndarray,
    libs: list[str],
    files: list[str],
    pos: pd.DataFrame | None,
    col_coords: tuple[str, str] | None,
    sx: float,
    sy: float,
    cell_id_sep: str,
    spot_key: str | None,
    cell_types: Mapping[str, str | Path | Iterable[str]] | None,
    clone_key: str | None,
    technology: str,
) -> SpatialAIR:
    """The body of :func:`read_airr` after the table is read (shared with :func:`read_airr_frame`)."""
    paths = files
    n_in = len(df)
    in_tissue = None
    keep = np.ones(n_in, dtype=bool)
    if col_coords is None:
        if "cell_id" not in df.columns:
            raise ValueError("the AIRR table has no cell_id column; pass coords=('x', 'y') or a positions table")
        bc = str_values(df["cell_id"])
        if pd.isna(bc).any():
            raise ValueError("cell_id has empty values")
        if pos is None:
            spot = bc
            xs, ys = split_xy(spot, cell_id_sep)
            coord_rule = f"cell_id 'x{cell_id_sep}y'"
        else:
            row = pos.index.get_indexer(bc)
            keep = row >= 0
            row = row[keep]
            spot = bc[keep]
            xs = pos["x"].to_numpy(dtype=float)[row]
            ys = pos["y"].to_numpy(dtype=float)[row]
            if "in_tissue" in pos.columns:
                in_tissue = to_bool(pos["in_tissue"]).fillna(False).to_numpy(dtype=bool)[row]
            coord_rule = "positions table keyed by cell_id"
    else:
        xk, yk = col_coords
        for c in (xk, yk):
            if c not in df.columns:
                raise ValueError(f"coordinate column {c!r} not in the AIRR table")
        xs = pd.to_numeric(df[xk], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(df[yk], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            raise ValueError(f"coordinate columns {xk!r} / {yk!r} contain missing or non-numeric values")
        sk = spot_key if spot_key is not None else ("cell_id" if "cell_id" in df.columns else None)
        if sk is not None:
            if sk not in df.columns or df[sk].isna().any():
                raise ValueError(f"spot column {sk!r} is missing or has empty values")
            spot = str_values(df[sk])
        elif np.array_equal(xs, np.round(xs)) and np.array_equal(ys, np.round(ys)):
            spot = format_spots(xs, ys)
        else:
            spot = np.array([f"{a:g}_{b:g}" for a, b in zip(xs, ys, strict=True)], dtype=object)
        check = pd.DataFrame({"s": spot, "lib": lib_col, "x": xs, "y": ys}).drop_duplicates()
        dup = check.duplicated(["s", "lib"])
        if dup.any():
            raise ValueError(f"spot {str(check.s[dup].iloc[0])!r} has more than one coordinate pair")
        coord_rule = f"columns {xk!r}, {yk!r}"
    n_unplaced = int((~keep).sum())
    if n_unplaced:
        logger.info("read_airr: %d records with a barcode outside the positions table were left out", n_unplaced)
        df = df.loc[keep].reset_index(drop=True)
        lib_col = lib_col[keep]
    n = len(df)
    air = pd.DataFrame({S.SPOT: spot, S.X: xs * sx, S.Y: ys * sy})
    locus = (
        map_unique(df["locus"], lambda v: str(v).strip().upper() or None)
        if "locus" in df.columns
        else np.full(n, np.nan, dtype=object)
    )
    miss = pd.isna(locus)
    if miss.any() and "v_call" in df.columns:
        locus[miss] = locus_from_call(df["v_call"].to_numpy(dtype=object, na_value=np.nan)[miss])
        miss = pd.isna(locus)
    if miss.any():
        raise ValueError(f"{int(miss.sum())} rows have no locus (no 'locus' field and no V call to infer it)")
    air[S.LOCUS] = locus
    for c in (S.V_CALL, D_CALL, S.J_CALL, S.C_CALL):
        air[c] = clean_call(df[c]) if c in df.columns else np.nan
    for c in (S.JUNCTION_AA, S.JUNCTION):
        air[c] = str_values(df[c]) if c in df.columns else np.nan
    if "productive" in df.columns:
        air[S.PRODUCTIVE] = flag_array(df["productive"])
    reads_from = next((c for c in ("consensus_count", "duplicate_count") if c in df.columns), None)
    air[S.READS] = df[reads_from].to_numpy(dtype=float) if reads_from else np.full(n, np.nan)
    air[S.UMI] = str_values(df["umi"]) if "umi" in df.columns else np.nan
    if clone_key == "exact":
        clone_col = None
    elif clone_key is not None:
        if clone_key not in df.columns:
            raise ValueError(f"clone column {clone_key!r} not in the AIRR table")
        clone_col = clone_key
    elif "clone_id" in df.columns and df["clone_id"].notna().all():
        clone_col = "clone_id"
    else:
        clone_col = None
        if "clone_id" in df.columns and df["clone_id"].notna().any():
            warnings.warn(
                f"AIRR clone_id is empty for {int(df['clone_id'].isna().sum())} of {n} records; using the exact "
                "clonotype key locus|V|J|junction_aa for all records (pass clone_key= to choose)",
                SpacloneWarning,
                stacklevel=2,
            )
    if clone_col is not None:
        clone = str_values(df[clone_col])
        clone_rule = f"column {clone_col!r}"
    else:
        clone = exact_clone_ids(air[S.LOCUS], air[S.V_CALL], air[S.J_CALL], air[S.JUNCTION_AA])
        clone_rule = "exact: locus|V gene|J gene|junction_aa"
    if pd.isna(clone).any():
        why = (
            f"column {clone_col!r} has empty values"
            if clone_col is not None
            else ("the table has no junction_aa column" if "junction_aa" not in df.columns else "junction_aa is empty")
        )
        raise ValueError(
            f"{int(pd.isna(clone).sum())} rows have no clone id ({why}; missing junction_aa or clone column: rename "
            "the CDR3 amino-acid column to junction_aa, or pass clone_key=)"
        )
    air[S.CLONE] = clone
    air[S.LIBRARY] = lib_col
    air["x_src"], air["y_src"] = xs, ys
    if "sequence_id" in df.columns:
        air["sequence_id"] = str_values(df["sequence_id"])
    if in_tissue is not None:
        air["in_tissue"] = in_tissue
    labels = _labels(cell_types) if cell_types is not None else None
    if labels is not None:
        air["cell_type"] = pd.Series(spot, dtype=object).map(labels).fillna("untyped").to_numpy(dtype=object)
    if "clone_id" in df.columns and clone_col != "clone_id":
        air["clone_id_airr"] = str_values(df["clone_id"])
    skip = set(AIRR_DEFAULT) | set(col_coords or ()) | {c for c in (spot_key, clone_key) if c}
    for c in df.columns:
        if c not in air.columns and (c not in skip or (c in ("umi_count", "duplicate_count") and c != reads_from)):
            air[c] = df[c].to_numpy()
    has_umi = bool("umi" in df.columns and df["umi"].notna().any())
    meta: dict[str, Any] = {
        "reader": "read_airr",
        "technology": technology,
        "files": list(paths),
        "library_id": libs[0] if len(set(libs)) == 1 else list(dict.fromkeys(libs)),
        "level": "molecule" if has_umi else "contig",
        "molecule_level": True,
        "coords": coord_rule,
        "source_unit": unit_name(sx, sy),
        "um_per_unit": sx if sx == sy else [sx, sy],
        "clone_rule": clone_rule,
        "reads_from": reads_from,
        "has_read_counts": reads_from is not None,
        "has_molecule_id": has_umi,
        "has_productive": "productive" in df.columns,
        "n_rows_read": int(n_in),
        "n_records_unpositioned": n_unplaced,
    }
    gex = None
    if labels is not None or pos is not None:
        if len(set(libs)) > 1:
            raise ValueError("a spot table (cell_types or a positions table) needs one library per call")
        first = air.drop_duplicates(S.SPOT)
        table = pd.DataFrame(
            {"x": first[S.X].to_numpy(), "y": first[S.Y].to_numpy()}, index=pd.Index(first[S.SPOT], dtype=str)
        )
        if pos is not None:
            add = pos.index[~pos.index.isin(table.index)]
            ptab = pd.DataFrame(
                {"x": pos.loc[add, "x"].to_numpy(float) * sx, "y": pos.loc[add, "y"].to_numpy(float) * sy}, index=add
            )
            table = pd.concat([table, ptab])
        if labels is not None:
            missing = pd.Index(labels.index, dtype=str)
            missing = missing[~missing.isin(table.index)]
            if len(missing):
                if pos is not None or col_coords is not None:
                    raise ValueError(
                        f"{len(missing)} typed spots have no position (e.g. {str(missing[0])!r}); with coordinate "
                        "columns "
                        "or a positions table, typed spots must carry records or be in the positions table"
                    )
                ux, uy = split_xy(missing.to_numpy(), cell_id_sep)
                table = pd.concat([table, pd.DataFrame({"x": ux * sx, "y": uy * sy}, index=missing)])
        table = table.sort_index()
        ids = table.index
        nrec = air[S.SPOT].value_counts().reindex(ids).fillna(0).astype(np.int64).to_numpy()
        obs = pd.DataFrame({"n_records": nrec, S.LIBRARY: libs[0]})
        if labels is not None:
            obs["cell_type"] = pd.Categorical(pd.Series(ids, dtype=object).map(labels).fillna("untyped"))
            meta["cell_type_counts"] = {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}
        if pos is not None and "in_tissue" in pos.columns:
            obs["in_tissue"] = to_bool(pos["in_tissue"]).set_axis(pos.index).reindex(ids).fillna(False).to_numpy(bool)
        gex = spot_table(ids, table["x"].to_numpy(), table["y"].to_numpy(), obs)
        meta["n_spots_table"] = int(gex.n_obs)
    meta["io"] = {
        "reader": "read_airr",
        "coordinates": coord_rule,
        "um_per_unit": {
            "value": sx if sx == sy else [sx, sy],
            "rule": f"given: scale_um = {sx if sx == sy else [sx, sy]}",
        },
        "clone_rule": clone_rule,
        "reads": f"{reads_from!r} (AIRR schema: reads)" if reads_from else "none: no consensus_count / duplicate_count",
        "locus": "column 'locus', else the V call" if "locus" in df.columns else "V call",
        "productive": "column 'productive' (T/F)" if "productive" in df.columns else "absent",
    }
    logger.info("read_airr: %d records on %d spots from %s", n, air[S.SPOT].nunique(), ", ".join(map(str, paths)))
    return finish(air, gex, meta)
