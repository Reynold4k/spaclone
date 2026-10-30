"""Read a spatial gene-expression source and an immune-receptor table of the same section in one call.

:func:`read` is the front door of the end-to-end analyses (:func:`spaclone.bcr.hotspot_workflow`). It takes the two
outputs of a spatial immune-receptor experiment, in any supported format, and returns a
:class:`~spaclone.io.SpatialAIR` in which

* the GEX keeps every unit of the source (on and off the tissue), with raw counts in ``X`` and ``layers["counts"]``
  (where the :mod:`spaclone.gex` functions look for them) and ``obsm["spatial"]`` in micrometres;
* every receptor record carries the micrometre position of its GEX unit (``x_um``, ``y_um``), so the two layers share
  one coordinate system;
* every decision (formats and their evidence, column mapping, count meaning, junction test, coordinate unit, barcode
  matching, annotation candidates) is recorded in ``meta["io"]``.

Receptor formats: :data:`spaclone.io.formats.RECEPTOR_FORMATS` (AIRR TSV of every producer, 10x contig CSV / JSON,
MiXCR, TRUST4, Change-O, immunarch, VDJtools, MiGEC, immunoSEQ, BD Rhapsody, Parse, WAT3R, scirpy and Dandelion
objects, Stereo-XCR-seq meta tables, AIR-SPACE clone tables). GEX sources: :data:`spaclone.io.gexio.GEX_FORMATS`.
Units are automatic only from technology facts (:func:`spaclone.io.gexio.detect_unit`); otherwise ``unit_um=`` is
required (the earlier fallback "assume µm, with a warning" is gone).

References
----------
Vander Heiden et al. 2018, doi:10.3389/fimmu.2018.02206 (AIRR Rearrangement schema); Chen et al. 2022,
doi:10.1016/j.cell.2022.04.003 (Stereo-seq DNB pitch 0.5 µm); Song et al. 2021, doi:10.1038/s41592-021-01142-2
(TRUST4); Virshup et al. 2023, doi:10.1038/s41587-023-01733-8 (scverse / AnnData).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from spaclone._utils import SpacloneWarning
from spaclone.io._container import SpatialAIR
from spaclone.io._hints import candidates, column_hint, examples
from spaclone.io._table import binary_kind
from spaclone.io.airr import AIRR_DEFAULT, _h5_safe, read_airr, read_airr_frame
from spaclone.io.annotations import detect_annotations
from spaclone.io.barcodes import match_barcodes
from spaclone.io.gexio import _unit_check, detect_unit, read_gex, source_coordinates
from spaclone.io.receptors import _junction_decision, read_receptors, receptor_format

__all__ = ["PAIRED_AIRR_FIELDS", "STEREO_BINS", "read"]

logger = logging.getLogger("spaclone")

#: AIRR fields :func:`read` keeps by default: the reader defaults plus the alignments used for somatic hypermutation
#: and lineage trees.
PAIRED_AIRR_FIELDS: tuple[str, ...] = (
    *AIRR_DEFAULT,
    "sequence",
    "sequence_alignment",
    "germline_alignment",
    "v_sequence_alignment",
    "v_germline_alignment",
    "j_sequence_alignment",
    "j_germline_alignment",
    "v_identity",
    "v_cigar",
    "d_cigar",
    "j_cigar",
    "v_germline_start",
    "v_germline_end",
    "np1_length",
    "np2_length",
)
#: Stereo-seq bin sizes (in DNB) of the earlier pitch rule (kept for compatibility; see
#: :data:`spaclone.io.gexio.SAW_BIN_SIZES` for the rule now used).
STEREO_BINS: tuple[int, ...] = (20, 50, 100, 200)
_DNB_UM = 0.5
_SEPS = ("-", "_", ":", ",", "x")
_SPOT_WORDS = ("cell", "spot", "barcode", "bin", "bead", "xy", "coord")


class PairedReadWarning(SpacloneWarning):
    """An assumption of :func:`read`: records dropped, counts not integer, implausible units, unused arguments."""


# ------------------------------------------------------------------------------------------------------- helpers


def _counts(adata: ad.AnnData, counts: str) -> tuple[sparse.csr_matrix, pd.DataFrame, str]:
    """Raw count matrix, its var table and where it came from."""
    if counts == "raw":
        if adata.raw is not None:
            m, var, src = adata.raw.X, adata.raw.var.copy(), "raw.X"
        elif "counts" in adata.layers:
            m, var, src = adata.layers["counts"], adata.var.copy(), "layers['counts'] (no .raw)"
        else:
            m, var, src = adata.X, adata.var.copy(), "X (no .raw, no counts layer)"
    elif counts == "X":
        m, var, src = adata.X, adata.var.copy(), "X"
    else:
        if counts not in adata.layers:
            raise KeyError(f"read: counts={counts!r} is neither 'raw', 'X' nor a layer ({list(adata.layers)})")
        m, var, src = adata.layers[counts], adata.var.copy(), f"layers[{counts!r}]"
    if m is None:
        raise ValueError(f"read: the GEX has no matrix in {src}")
    m = m.tocsr() if sparse.issparse(m) else sparse.csr_matrix(np.asarray(m))
    if m.nnz:
        sample = m.data[: min(m.nnz, 200_000)]
        if not np.allclose(sample, np.round(sample)):
            warnings.warn(
                f"read: the matrix taken as raw counts ({src}) holds non-integer values; pass counts= with raw counts",
                PairedReadWarning,
                stacklevel=3,
            )
    var.index = pd.Index(var.index.astype(str), dtype=str)
    return m, var, src


def _airr_paths(airr: Any) -> list[Path] | None:
    if isinstance(airr, pd.DataFrame):
        return None
    if isinstance(airr, str | Path):
        return [Path(airr)]
    return [Path(p) for p in airr]


def _receptor_ids(airr: Any, spot_key: str) -> np.ndarray:
    """Spot ids of every receptor record (all files), as strings."""
    if isinstance(airr, pd.DataFrame):
        if spot_key not in airr.columns:
            raise KeyError(
                f"read: spot column {spot_key!r} not in the AIRR table; pass spot_key= the column that holds the GEX "
                f"unit ids ({column_hint(spot_key, list(airr.columns), keywords=_SPOT_WORDS)})"
            )
        s = airr[spot_key]
    else:
        parts = []
        for p in _airr_paths(airr):
            head = pd.read_csv(p, sep="\t", nrows=0)
            if spot_key not in head.columns:
                raise KeyError(
                    f"read: spot column {spot_key!r} not in {p.name}; pass spot_key= the column that holds the GEX "
                    f"unit ids ({column_hint(spot_key, list(head), keywords=_SPOT_WORDS)})"
                )
            parts.append(pd.read_csv(p, sep="\t", usecols=[spot_key], dtype=str, keep_default_na=False)[spot_key])
        s = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    s = s.astype("string")
    if s.isna().any() or (s == "").any():
        raise ValueError(f"read: the AIRR spot column {spot_key!r} has empty values")
    return s.to_numpy(dtype=object)


def _parse_ids(ids: np.ndarray, sep: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = pd.Series(ids, dtype=object).astype(str).str.split(sep, regex=False)
    ok = parts.str.len().to_numpy() == 2
    xs = pd.to_numeric(parts.str[0], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(parts.str[1], errors="coerce").to_numpy(dtype=float)
    ok &= np.isfinite(xs) & np.isfinite(ys)
    return xs, ys, ok


def _coordinate_ids(ids: np.ndarray, xy: np.ndarray) -> str | None:
    """Separator with which GEX unit ids spell their own source coordinates ``x<sep>y``, else None."""
    if not len(ids):
        return None
    take = np.linspace(0, len(ids) - 1, min(len(ids), 20_000)).astype(int)
    for sep in _SEPS:
        xs, ys, ok = _parse_ids(ids[take], sep)
        if ok.all() and np.allclose(xs, xy[take, 0], atol=1e-9) and np.allclose(ys, xy[take, 1], atol=1e-9):
            return sep
    return None


def _receptor_in_h5(path: Path) -> str | None:
    """'scirpy' / 'scirpy_legacy' / 'cell_calls' when an .h5ad / .h5mu holds receptor data, else None (h5py only)."""
    if path.suffix.lower() == ".h5mu":
        return "scirpy"
    try:
        with h5py.File(path, "r") as h:
            if "obsm" in h and "airr" in h["obsm"]:
                return "scirpy"
            cols = (
                [str(c) for c in np.asarray(h["obs"].attrs.get("column-order", []), dtype=object)] if "obs" in h else []
            )
    except OSError:
        return None
    if any(c.startswith(("IR_VJ_", "IR_VDJ_")) for c in cols):
        return "scirpy_legacy"
    from spaclone.io.cellcalls import STEREO_XCR_LUAD_CALLS

    if set(STEREO_XCR_LUAD_CALLS.values()) <= set(cols):
        return "cell_calls"
    return None


def _single_source(obj: Any) -> tuple[Any, Any] | None:
    """(gex, receptor source) when one object / file holds both, else None."""
    from spaclone.io.receptors import is_mudata

    if is_mudata(obj):
        mods = list(obj.mod)
        rec = [m for m in mods if "airr" in obj.mod[m].obsm]
        rest = [m for m in mods if m not in rec]
        if len(rec) == 1 and len(rest) == 1:
            return obj.mod[rest[0]], obj
        raise ValueError(f"MuData with modalities {mods}: expected one GEX modality and one with obsm['airr']")
    if isinstance(obj, ad.AnnData):
        ir = any(str(c).startswith(("IR_VJ_", "IR_VDJ_")) for c in obj.obs.columns)
        return (obj, obj) if ("airr" in obj.obsm or ir) else None
    p = Path(obj)
    if p.is_file() and p.suffix.lower() in (".h5ad", ".h5mu"):
        kind = _receptor_in_h5(p)
        if kind in ("scirpy", "scirpy_legacy"):
            if p.suffix.lower() == ".h5mu":
                from spaclone.io.receptors import _read_h5mu

                return _single_source(_read_h5mu(p))
            a = ad.read_h5ad(p)
            return a, a
        if kind == "cell_calls":
            return "cell_calls", p
    return None


def _summary(meta: Mapping[str, Any]) -> str:
    io = meta.get("io", {})
    rec, gx, bc = io.get("receptors", {}), io.get("gex", {}), io.get("barcodes", {})
    lines = [
        f"receptors: {rec.get('format')} ({rec.get('tool', '')})",
        f"  evidence: {'; '.join(map(str, rec.get('evidence', [])))}",
        f"  counts: {rec.get('counts')}",
        f"  junction: {(rec.get('junction') or {}).get('decision')}",
        f"GEX: {gx.get('format')} ({'; '.join(map(str, gx.get('evidence', [])))})",
        f"  coordinates: {meta.get('coordinates_from')}; unit: {meta.get('um_per_unit')} µm per unit ({meta.get('unit_rule')})",
        f"barcodes: {bc.get('rule')} (overlap {bc.get('overlap_before', 0):.1%} -> {bc.get('overlap_after', 0):.1%})",
        f"annotations: {io.get('annotations', {}) or 'none found'}",
        f"records: {meta.get('n_records')} on {meta.get('n_receptor_spots')} spots; dropped {meta.get('n_records_dropped')}",
    ]
    return "\n".join(lines)


def _binary_receptors(paths: list[Path], gex: Any) -> tuple[str, list[str]] | None:
    """(format, evidence) when the receptor files are binary containers that hold receptor data; None for text tables.

    A binary file without receptor data (a GEX ``.h5ad``, a zarr store, ...) raises, saying that gex and airr look
    swapped when the GEX argument is a receptor table.
    """
    kinds = [binary_kind(p) for p in paths]
    if not any(kinds):
        return None
    found: list[tuple[str, str]] = []
    for p, kind in zip(paths, kinds, strict=True):
        suf = p.suffix.lower()
        what = _receptor_in_h5(p) if kind == "HDF5" and suf in (".h5ad", ".h5mu") else None
        if kind == "HDF5" and suf == ".h5ddl":
            found.append(("dandelion", f"{p.name}: Dandelion .h5ddl"))
        elif what in ("scirpy", "scirpy_legacy"):
            found.append(
                (
                    what,
                    f"{p.name}: "
                    + ("obsm['airr'] (scirpy)" if what == "scirpy" else "IR_* obs columns (scirpy < 0.13)"),
                )
            )
        elif what == "cell_calls":
            raise ValueError(
                f"read: {p.name} holds Stereo-XCR-seq cell calls together with their GEX: read it alone, "
                f"spaclone.read({p.name!r})"
            )
        else:
            raise ValueError(_not_receptors(p, kind or "text", gex))
    if len({f for f, _ in found}) > 1:
        raise ValueError(
            f"read: the receptor files mix formats ({sorted({f for f, _ in found})}); read them one at a time"
        )
    return found[0][0], [e for _, e in found]


def _not_receptors(p: Path, kind: str, gex: Any) -> str:
    """Why a binary receptor argument cannot be read, with the swapped-argument hint when it applies."""
    if kind == "Parquet":
        return (
            f"read: the receptor argument {p.name} is a Parquet file: read a Stereo-XCR-seq meta table with "
            "spaclone.io.read_stereo_xcr(path); spaclone.read takes receptor tables as delimited text or AnnData"
        )
    what = "a zarr store" if kind == "zarr" else f"a{'n' if kind == 'HDF5' else ''} {kind} file"
    msg = (
        f"read: the receptor argument {p.name} is {what} without receptor data (no scirpy obsm['airr'] or IR_* columns)"
    )
    g = Path(gex) if isinstance(gex, str | Path) else None
    if g is not None and g.is_file() and binary_kind(g) is None:
        try:
            fmt = receptor_format(g)[0]
        except (ValueError, OSError, UnicodeError):
            fmt = None
        if fmt is not None:
            return (
                f"{msg}, and the GEX argument {g.name} is a receptor table ({fmt.tool}): gex and airr look swapped. "
                f"Give the expression data first: spaclone.read({p.name!r}, {g.name!r})"
            )
    return msg + "; airr must be a receptor table (AIRR TSV, 10x contigs, MiXCR, TRUST4, ...; see spaclone.io.sniff)"


def _read_airspace_pair(
    gex: Any,
    paths: list[Path],
    rec_ev: list[str],
    given: Mapping[str, Any],
    *,
    unit_um: Any,
    library_id: str | None,
    min_overlap: float,
    unmatched: str,
    technology: str,
    verbose: bool,
) -> SpatialAIR:
    """:func:`read` of an AIR-SPACE bead h5ad + clone table: :func:`~spaclone.io.read_airspace` with its arguments."""
    from spaclone.io.airspace import CURIO_UM, read_airspace

    if len(paths) != 1:
        raise ValueError("read: give one AIR-SPACE clone table per bead h5ad")
    if unit_um == "auto":
        scale, rule = CURIO_UM, f"AIR-SPACE / Curio bead unit {CURIO_UM:g} µm (read_airspace)"
    else:
        f = (float(unit_um),) * 2 if np.ndim(unit_um) == 0 else tuple(float(v) for v in unit_um)
        if not all(v > 0 and np.isfinite(v) for v in f):
            raise ValueError(f"unit_um must be positive and finite, got {unit_um!r}")
        if f[0] != f[1]:
            raise ValueError("read: AIR-SPACE bead coordinates take one unit_um for x and y")
        scale, rule = f[0], f"given: unit_um = {unit_um}"
    if unmatched == "coordinates":
        raise ValueError("read: unmatched='coordinates' needs GEX ids that spell their coordinates; AIR-SPACE bead "
                         "barcodes do not (use 'drop' or 'error')")  # fmt: skip
    defaults = {"spot_key": "cell_id", "counts": ("raw", "X"), "spatial_key": "auto", "tissue_key": "in_tissue",
                "columns": None, "clone_key": None, "gex_format": "auto", "count_type": None, "junction": "auto",
                "bin_size": None}  # fmt: skip
    unused = [
        k for k, v in given.items() if not (v in defaults[k] if isinstance(defaults[k], tuple) else v == defaults[k])
    ]
    if unused:
        warnings.warn(
            "read: the AIR-SPACE reader does not use " + ", ".join(f"{k}={given[k]!r}" for k in unused) + " (bead "
            "barcodes, the raw counts in X and the authors' tissue rule are fixed); see spaclone.io.read_airspace",
            PairedReadWarning,
            stacklevel=3,
        )
    sa = read_airspace(gex, paths[0], library_id=library_id, scale_um=scale, min_overlap=min_overlap,
                       unmatched="error" if unmatched == "error" else "drop")  # fmt: skip
    m = sa.meta
    if technology != "spatial GEX + AIRR":
        m["technology"] = technology
    m.update(
        coordinates_from=m["io"]["coordinates"],
        unit_rule=rule,
        n_receptor_spots=int(sa.air["spot_id"].nunique()),
        n_records_dropped=int(m["n_records_unmatched_barcode"]),
    )
    m["io"] = {
        "receptors": {"format": "airspace_ovp", "evidence": rec_ev, "counts": "consensus_count = reads per molecule"},
        "gex": {"format": "AIR-SPACE bead h5ad", "evidence": [m["io"]["coordinates"]]},
        "unit": {"um_per_unit": m["um_per_unit"], "rule": rule, "check": m["unit_check"]},
        "barcodes": {"rule": "exact (bead barcodes)", "overlap_before": m["spot_overlap"],
                     "overlap_after": m["spot_overlap"]},
        "reader": m["io"],
    }  # fmt: skip
    if verbose:
        print(_summary({**m, "n_records": len(sa.air)}))
    return sa


# --------------------------------------------------------------------------------------------------------- read


def read(
    gex: Any,
    airr: Any = None,
    *,
    unit_um: Literal["auto"] | float | tuple[float, float] = "auto",
    spot_key: str = "cell_id",
    counts: str = "raw",
    library_id: str | None = None,
    spatial_key: str = "auto",
    tissue_key: str | None = "in_tissue",
    unmatched: Literal["auto", "coordinates", "drop", "error"] = "auto",
    min_overlap: float = 0.5,
    columns: Sequence[str] | Mapping[str, str] | Literal["all"] | None = None,
    clone_key: str | None = None,
    technology: str = "spatial GEX + AIRR",
    format: str = "auto",
    gex_format: str = "auto",
    count_type: Literal["reads", "umis", "templates", "cells"] | None = None,
    junction: Literal["auto", "junction", "cdr3"] = "auto",
    bin_size: int | None = None,
    verbose: bool = False,
) -> SpatialAIR:
    """Read a spatial GEX source and a receptor table of the same section into one SpatialAIR.

    Parameters
    ----------
    gex
        The expression data: ``.h5ad`` / AnnData (scanpy, squidpy, stereopy / SAW, Seurat-converted, Curio, Giotto),
        a Space Ranger Visium folder, a Visium HD bin folder (``binned_outputs/square_008um``), a Xenium folder, a
        Stereo-seq ``.gef`` / ``.gem`` (with ``bin_size`` for square bins), a Slide-seq folder or a SpatialData object.
        Called alone, with a spaclone store directory (:func:`spaclone.io.write`), a scirpy AnnData / MuData (GEX and
        ``obsm["airr"]`` in one object) or a Stereo-XCR-seq LUAD cell-call h5ad, it reads both from it.
    airr
        The receptor table(s): a file of any format of :data:`spaclone.io.formats.RECEPTOR_FORMATS` (or several, e.g.
        BCR and TCR, concatenated), a ``DataFrame``, an AnnData / MuData with scirpy data or a Dandelion object.
    unit_um
        µm per GEX coordinate unit: ``"auto"`` (technology facts only, see :func:`spaclone.io.gexio.detect_unit`; an
        error names this argument when the unit cannot be established; an AIR-SPACE bead h5ad with its clone table
        uses the Curio bead unit, 0.65 µm), a number, or ``(x, y)`` factors. The result is checked: units closer
        than 0.2 µm or further than 300 µm apart, or a section wider than 5 cm, give a warning naming this argument.
    spot_key
        Column of the receptor table whose values are the GEX unit ids: an AIRR column (TRUST4 on Stereo-seq:
        ``cell_id = "8860-6440"``), a canonical field of :func:`spaclone.io.read_receptors` (``cell_id`` holds the
        10x ``barcode``, MiXCR ``tagValueCELL``, ...) or a source column.
    counts
        Raw GEX counts: ``"raw"`` (``.raw.X``, falls back to ``layers["counts"]``, then ``X``), ``"X"`` or a layer.
    library_id
        Section label of the receptor records (default: the receptor file's stem).
    spatial_key
        ``"auto"`` (``obsm["spatial"]``, else one unambiguous alternative), an ``obsm`` key or ``"obs:xcol,ycol"``.
    tissue_key
        Boolean ``obs`` column of tissue units, copied to the receptor table as ``in_tissue`` when present.
    unmatched
        Receptor records whose spot is not a GEX unit: ``"coordinates"`` places them from their spot id when the GEX
        ids spell their own coordinates (``"x-y"``), ``"drop"`` leaves them out (with a warning), ``"error"`` raises;
        ``"auto"`` = ``"coordinates"`` when possible, else ``"drop"``. AIR-SPACE molecules on barcodes that are not
        beads are dropped (logged at WARNING) or, with ``"error"``, raise.
    min_overlap
        Smallest share of the distinct receptor spot ids that must be GEX units, exactly or after one documented id
        normalisation (:func:`spaclone.io.match_barcodes`); below it :func:`read` raises with examples.
    columns
        A mapping ``{canonical field: source column}`` (wins over the format's synonym table), or, for AIRR TSVs, the
        AIRR fields to keep (default :data:`PAIRED_AIRR_FIELDS`; ``"all"`` = every field).
    clone_key
        Clone definition passed to :func:`spaclone.io.read_airr` (``None``: the source's ``clone_id`` when complete,
        else the exact ``locus|V gene|J gene|junction_aa``).
    technology
        Free-text label stored in ``meta``.
    format
        Receptor format (``"auto"``: documented signature; see :data:`spaclone.io.formats.RECEPTOR_FORMATS`).
    gex_format
        GEX source format (``"auto"`` or a key of :data:`spaclone.io.gexio.GEX_FORMATS`).
    count_type
        What an unlabelled receptor count column counts (``"reads"``, ``"umis"``, ``"templates"``, ``"cells"``).
    junction
        ``"auto"`` (conserved-anchor test), ``"junction"`` or ``"cdr3"``.
    bin_size
        Square-bin side in DNB for Stereo-seq ``.gef`` / ``.gem`` GEX.
    verbose
        Print the decisions (they are always in ``meta["io"]``).

    Returns
    -------
    SpatialAIR
        ``air``: the receptor records with ``spot_id`` (= the GEX unit), ``x_um`` / ``y_um``, ``locus``,
        ``clone_id``, the AIRR fields and ``in_gex``. ``gex``: every GEX unit, raw counts in ``X`` and
        ``layers["counts"]``, the original ``obs`` / ``obsm`` / ``uns``, ``obsm["spatial"]`` in µm and
        ``obsm["spatial_src"]`` in source units. ``meta``: files, ``source_unit``, ``um_per_unit``, ``unit_rule``,
        ``counts_from``, ``n_gex_units``, ``n_receptor_spots``, ``n_receptor_spots_in_gex``, ``spot_overlap``,
        ``record_overlap``, ``unmatched_rule``, ``n_records_placed_from_id``, ``n_records_dropped``, the receptor
        reader's metadata and ``io`` (every decision with its evidence).

    Raises
    ------
    ValueError
        An unrecognised or ambiguous format (``format=`` / ``gex_format=``), an unknown coordinate unit (``unit_um=``),
        an unlabelled count (``count_type=``), mixed junction / CDR3 values (``junction=``), fewer than
        ``min_overlap`` of the receptor spot ids among the GEX units (examples from both sides; for AIR-SPACE, of the
        clone-table barcodes among the beads), a receptor argument that is an HDF5 / zarr container without receptor
        data (the message says when gex and airr look swapped), or records without a junction or clone column (the
        message suggests ``columns=`` / ``clone_key=``).
    KeyError
        ``spot_key`` is not a column of the receptor table (the message lists the columns, or the closest ones).

    Warns
    -----
    PairedReadWarning
        Receptor records left out (spots that are not GEX units), non-integer counts, implausible coordinates in µm
        (check ``unit_um``), or arguments the AIR-SPACE reader does not use.

    References
    ----------
    Vander Heiden et al. 2018, doi:10.3389/fimmu.2018.02206; Chen et al. 2022, doi:10.1016/j.cell.2022.04.003.

    Examples
    --------
    >>> sa = read("mm_spleen_bin20_anno.h5ad", "mm_spleen_bcr_airr.tsv")  # doctest: +SKIP
    >>> sa.meta["unit_rule"], sa.meta["spot_overlap"]  # doctest: +SKIP
    ("obs names spell the integer coordinates as 'x-y' on a 20-unit lattice: ...", 0.991)
    >>> sa = read("outs/binned_outputs/square_008um", "trust4_barcode_airr.tsv")  # doctest: +SKIP
    """
    rec_kind = None
    if airr is None:
        single = _single_source(gex) if not (isinstance(gex, str | Path) and Path(gex).is_dir()) else None
        if single is None:
            if isinstance(gex, str | Path) and not (Path(gex).is_file() and Path(gex).suffix in (".h5ad", ".h5")):
                from spaclone.io.store import read as read_store

                return read_store(gex)  # a store directory (raises FileNotFoundError when there is none)
            raise TypeError("read needs a GEX .h5ad and an AIRR table (or one spaclone store directory)")
        if isinstance(single[0], str) and single[0] == "cell_calls":
            from spaclone.io.cellcalls import from_cell_calls

            sa = from_cell_calls(single[1], library_id=library_id)
            sa.meta["io"] = {"receptors": {"format": "cell_calls", "evidence": ["obs IGHcdr3aa, IGK/Lcdr3aa, TRAcdr3aa, "
                             "TRBcdr3aa (Stereo-XCR-seq LUAD, GEO GSE328481)"]}}  # fmt: skip
            return sa
        gex, airr = single
        rec_kind = "object"
    if unmatched not in ("auto", "coordinates", "drop", "error"):
        raise ValueError("unmatched must be 'auto', 'coordinates', 'drop' or 'error'")
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must lie in [0, 1]")
    mapping = dict(columns) if isinstance(columns, Mapping) else None
    keep_cols = None if isinstance(columns, Mapping) else columns

    # ---- the receptor format (binary containers are checked first, never sniffed as text)
    paths = _airr_paths(airr) if isinstance(airr, str | Path | list | tuple) else None
    binary = _binary_receptors(paths, gex) if paths is not None and rec_kind is None else None
    if rec_kind == "object" or (paths is None and not isinstance(airr, pd.DataFrame)):
        fmt_name, rec_ev = "object", ["AnnData / MuData / Dandelion object"]
    elif binary is not None:
        fmt_name, rec_ev = binary
    else:
        fmt, rec_ev = receptor_format(paths[0] if paths else airr, format=format)
        fmt_name = fmt.name
    if fmt_name == "airspace_ovp":
        given = {"spot_key": spot_key, "counts": counts, "spatial_key": spatial_key, "tissue_key": tissue_key,
                 "columns": columns, "clone_key": clone_key, "gex_format": gex_format, "count_type": count_type,
                 "junction": junction, "bin_size": bin_size}  # fmt: skip
        return _read_airspace_pair(gex, paths, rec_ev, given, unit_um=unit_um, library_id=library_id,
                                   min_overlap=min_overlap, unmatched=unmatched, technology=technology,
                                   verbose=verbose)  # fmt: skip
    if fmt_name == "stereo_xcr_meta":
        raise ValueError(
            "a Stereo-XCR-seq meta table carries its own DNB coordinates: read it with spaclone.io.read_stereo_xcr("
            "path) (records at their DNB, 0.5 µm) and the GEX with spaclone.io.read_gex(...)"
        )

    # ---- the GEX
    adata, gex_info = read_gex(gex, gex_format=gex_format, bin_size=bin_size)
    gex_file = gex_info["file"]
    names = pd.Index(adata.obs_names.astype(str), dtype=str)
    if not names.is_unique:
        raise ValueError("read: GEX obs_names must be unique")
    m, var, counts_from = _counts(adata, counts)
    src_xy, coord_src, coord_others = source_coordinates(adata, spatial_key)
    if unit_um == "auto":
        sx, sy, unit_name, unit_rule = detect_unit(adata, src_xy, coordinates_from=coord_src)
    else:
        sx, sy = (float(unit_um), float(unit_um)) if np.ndim(unit_um) == 0 else tuple(float(v) for v in unit_um)
        if not (sx > 0 and sy > 0 and np.isfinite(sx) and np.isfinite(sy)):
            raise ValueError(f"unit_um must be positive and finite, got {unit_um!r}")
        unit_name = "Stereo-seq DNB" if sx == sy == _DNB_UM else "given"
        unit_rule = f"given: unit_um = {unit_um}"
    xy_um = src_xy * np.array([sx, sy])
    unit_ok = _unit_check(xy_um, argument="unit_um")
    if not unit_ok["plausible"]:
        warnings.warn(f"read: {unit_ok['message']}", PairedReadWarning, stacklevel=2)

    # ---- the GEX: every unit, raw counts in X and layers["counts"], coordinates in µm
    obs = adata.obs.copy()
    obs.index = names
    g = ad.AnnData(X=m, obs=obs, var=var)
    g.layers["counts"] = m
    for k in adata.obsm.keys():
        g.obsm[k] = adata.obsm[k]
    g.obsm["spatial_src"] = src_xy
    g.obsm["spatial"] = xy_um
    for k in adata.obsp.keys():
        g.obsp[k] = adata.obsp[k]
    g.uns = dict(adata.uns)
    tissue = None
    if tissue_key is not None and tissue_key in g.obs.columns:
        from spaclone.io.airr import to_bool

        tissue = to_bool(g.obs[tissue_key]).fillna(False).to_numpy(dtype=bool)

    # ---- the receptor table (AIRR files keep the original reader path)
    airr_file_path = fmt_name == "airr" and paths is not None and mapping is None and junction == "auto"
    if airr_file_path:
        ids = _receptor_ids(airr, spot_key)
        tab = None
        rec_io: dict[str, Any] = {"format": "airr", "tool": "AIRR Rearrangement TSV", "evidence": rec_ev}
    else:
        if paths is not None:
            tab = read_receptors(paths if len(paths) > 1 else paths[0], format=fmt_name, columns=mapping,
                                 count_type=count_type, junction=junction)  # fmt: skip
        elif isinstance(airr, pd.DataFrame) and fmt_name == "airr" and mapping is None:
            tab = airr.copy()
            tab.attrs = {"spaclone": {"io": {"format": "airr", "tool": "AIRR table (DataFrame)", "evidence": rec_ev}}}
        else:
            tab = read_receptors(airr, format=format if fmt_name != "object" else "auto", columns=mapping,
                                 count_type=count_type, junction=junction)  # fmt: skip
        rec_io = dict(tab.attrs.get("spaclone", {}).get("io", {}))
        key = spot_key if spot_key in tab.columns else None
        if key is None:
            have_cells = "cell_id" in tab.columns and tab["cell_id"].notna().any()
            raise KeyError(
                f"read: spot column {spot_key!r} not in the receptor table"
                + (
                    ""
                    if have_cells
                    else f" and the {rec_io.get('format', 'source')} format has no cell / spot barcodes "
                    "(a bulk repertoire cannot be placed on GEX units)"
                )  # fmt: skip
                + f"; pass spot_key= ({column_hint(spot_key, list(tab.columns), keywords=_SPOT_WORDS)})"
            )
        s = tab[key].astype("string")
        if s.isna().any() or (s == "").any():
            bad = int((s.isna() | (s == "")).sum())
            raise ValueError(f"read: {bad:,} receptor records have no spot id in {key!r}; drop them or pass spot_key=")
        ids = s.to_numpy(dtype=object)
    uniq = pd.unique(ids)
    try:
        bm = match_barcodes(uniq, names, min_overlap=min_overlap)
    except ValueError as e:
        raise ValueError(f"read: {e} (AIRR column {spot_key!r})") from None
    if bm.rule != "exact":
        new = pd.Series(ids, dtype=object).map(bm.mapping)
        ids = np.where(new.notna(), new.to_numpy(dtype=object), ids).astype(object)
        uniq = pd.unique(ids)
        if tab is None:  # switch the AIRR files to the in-memory path to rename their spot ids
            frames = [pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_values=[""]) for p in paths]
            frames = [f.loc[:, [c for c in f.columns if not str(c).startswith("Unnamed:")]] for f in frames]
            tab = pd.concat(frames, ignore_index=True)
        tab = tab.copy()
        tab[spot_key] = ids
    hit = pd.Index(uniq).isin(names)
    n_spots, n_hit = len(uniq), int(hit.sum())
    share = n_hit / n_spots if n_spots else 0.0
    rec_hit = float(pd.Index(ids).isin(names).mean()) if len(ids) else 0.0
    miss = uniq[~hit]
    rule = unmatched
    sep = (
        _coordinate_ids(names.to_numpy(dtype=object), src_xy)
        if len(miss) and unmatched in ("auto", "coordinates")
        else None
    )
    if len(miss):
        if unmatched == "error":
            raise ValueError(f"read: {len(miss):,} receptor spot ids are not GEX units (e.g. {examples(miss)})")
        if unmatched == "coordinates" and sep is None:
            raise ValueError("read: unmatched='coordinates' needs GEX ids that spell their coordinates ('x-y')")
        rule = "coordinates" if (unmatched in ("auto", "coordinates") and sep is not None) else "drop"
    else:
        rule = "none unmatched"
    pos = pd.DataFrame({"x": src_xy[:, 0], "y": src_xy[:, 1]}, index=names)
    if tissue is not None:
        pos["in_tissue"] = tissue
    n_placed = n_dropped = 0
    if len(miss) and rule == "coordinates":
        mx, my, ok = _parse_ids(miss, sep)
        extra = pd.DataFrame({"x": mx[ok], "y": my[ok]}, index=pd.Index(miss[ok].astype(str), dtype=str))
        if tissue is not None:
            extra["in_tissue"] = False
        pos = pd.concat([pos, extra])
        placed = set(miss[ok])
        n_placed = int(pd.Series(ids).isin(placed).sum())
        n_dropped = int(len(ids) - pd.Index(ids).isin(pos.index).sum())
        if n_dropped:
            warnings.warn(
                f"read: {n_dropped:,} receptor records on spots that are neither GEX units nor 'x{sep}y' ids were left "
                "out",
                PairedReadWarning,
                stacklevel=2,
            )
    elif len(miss):
        n_dropped = int((~pd.Index(ids).isin(names)).sum())
        warnings.warn(
            f"read: {n_dropped:,} receptor records on {len(miss):,} spots that are not GEX units were left out (their "
            "position is unknown); pass unmatched='error' to stop instead",
            PairedReadWarning,
            stacklevel=2,
        )
    kw = {"coords": pos, "scale_um": (sx, sy), "clone_key": clone_key, "technology": technology}
    try:
        if tab is None:
            cols = list(PAIRED_AIRR_FIELDS) if keep_cols is None else keep_cols
            kw.update(columns=cols, library_id=library_id)
            if spot_key != "cell_id":
                frames = [pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_values=[""]) for p in paths]
                frames = [f.loc[:, [c for c in f.columns if not str(c).startswith("Unnamed:")]] for f in frames]
                kw["library_id"] = library_id or paths[0].name.split(".")[0]
                sa = _frame_read(pd.concat(frames, ignore_index=True), spot_key, str(paths[0]), **kw)
            else:
                sa = read_airr(paths if len(paths) > 1 else paths[0], **kw)
            junction_info = _junction_decision(sa.air["junction_aa"], "auto", "junction_aa", labelled=True)[1]
            rec_io["junction"] = junction_info
            rec_io["counts"] = {"consensus_count": f"{sa.meta.get('reads_from')!r} = reads (AIRR schema)"}
        else:
            if fmt_name == "airr" and isinstance(airr, pd.DataFrame) and mapping is None:
                cols = list(PAIRED_AIRR_FIELDS) if keep_cols is None else keep_cols
            else:
                cols = "all" if keep_cols is None else keep_cols
            lib = library_id or (paths[0].name.split(".")[0] if paths else "airr")
            kw.update(columns=cols, library_id=lib)
            sa = _frame_read(tab, spot_key, str(paths[0]) if paths else "<object>", **kw)
    except ValueError as e:
        if "no clone id" not in str(e):
            raise
        have = list(tab.columns) if tab is not None else [c for p in paths for c in pd.read_csv(p, sep="\t", nrows=0)]
        raise ValueError(f"read: {e}. {_clone_hint(have)}") from None
    air = sa.air
    air["in_gex"] = air["spot_id"].isin(names).to_numpy()
    meta = dict(sa.meta)
    ann = detect_annotations(g)
    meta.update(
        {
            "reader": "read",
            "technology": technology,
            "gex_file": gex_file,
            "airr_files": [str(p) for p in paths] if paths is not None else None,
            "coordinates_from": coord_src,
            "source_unit": unit_name,
            "um_per_unit": sx if sx == sy else [sx, sy],
            "unit_rule": unit_rule,
            "unit_check": {k: v for k, v in unit_ok.items() if k != "message"},
            "counts_from": counts_from,
            "n_gex_units": int(g.n_obs),
            "n_genes": int(g.n_vars),
            "spot_key": spot_key,
            "n_receptor_spots": int(n_spots),
            "n_receptor_spots_in_gex": n_hit,
            "spot_overlap": float(share),
            "record_overlap": rec_hit,
            "unmatched_rule": rule,
            "n_records_placed_from_id": n_placed,
            "n_records_dropped": n_dropped,
        }
    )
    meta.pop("n_spots_table", None)
    meta["io"] = _json_ready(
        {
            "receptors": rec_io,
            "gex": {
                "format": gex_info["format"],
                "evidence": gex_info["evidence"],
                "coordinates": coord_src,
                "other_coordinates": coord_others,
                "unit": {"um_per_unit": meta["um_per_unit"], "rule": unit_rule, "check": unit_ok},
                "counts": counts_from,
            },  # fmt: skip
            "barcodes": bm.record(),
            "annotations": ann.record(),
            "airr_reader": sa.meta.get("io", {}),
        }
    )
    air.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        **{k: meta[k] for k in ("reader", "source_unit", "um_per_unit", "unit_rule") if k in meta},
    }
    g.uns["spaclone"] = _h5_safe(
        {
            **{k: v for k, v in meta.items() if isinstance(v, str | int | float | bool | list) and k != "io"},
            "um_per_unit": meta["um_per_unit"],
        }
    )
    logger.info(
        "read: %d GEX units x %d genes (%s); %d receptor records on %d spots, %.1f%% of them GEX units (%s)",
        g.n_obs,
        g.n_vars,
        unit_rule,
        len(air),
        n_spots,
        100 * share,
        rule,
    )
    if verbose:
        print(_summary({**meta, "n_records": len(air)}))
    return SpatialAIR(air=air, gex=g, meta=meta)


def _clone_hint(columns: Sequence[Any]) -> str:
    """How to name the junction or a clone column when the receptor table has neither under its AIRR name."""
    cdr3 = [c for c in candidates(columns, ("cdr3", "junction")) if c not in ("junction_aa", "junction")]
    cdr3 = [c for c in cdr3 if not c.lower().endswith(("length", "start", "end", "begin", "_len"))]
    clone = [c for c in candidates(columns, ("clone", "clonotype")) if c != "clone_id"]
    out = []
    if cdr3:
        out.append(
            f"columns={{'junction_aa': {cdr3[0]!r}}} (the CDR3 amino-acid column; candidates: "
            f"{', '.join(map(repr, cdr3[:5]))})"
        )
    if clone:
        out.append(f"clone_key={clone[0]!r} (a clone column; candidates: {', '.join(map(repr, clone[:5]))})")
    if not out:
        return (
            f"Pass columns={{'junction_aa': ...}} naming the CDR3 amino-acid column "
            f"({column_hint('junction_aa', columns)})"
        )
    return "Pass " + ", or ".join(out)


def _frame_read(table: pd.DataFrame, spot_key: str, source: str, **kw: Any) -> SpatialAIR:
    """read_airr on an in-memory table whose spot column is ``spot_key``."""
    t = table.copy()
    if spot_key != "cell_id":
        if "cell_id" in t.columns:
            t = t.rename(columns={"cell_id": "cell_id_airr"})
        t = t.rename(columns={spot_key: "cell_id"})
    t.attrs = {}
    return read_airr_frame(t, source=source, **kw)


def _json_ready(v: Any) -> Any:
    """Nested dict / list of str, int, float (no NaN), bool and None only (meta survives JSON and h5ad)."""
    if isinstance(v, Mapping):
        return {str(k): _json_ready(x) for k, x in v.items()}
    if isinstance(v, list | tuple | set):
        return [_json_ready(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if v is None or isinstance(v, str | int | float | bool):
        return v
    return str(v)
