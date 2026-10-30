"""Stereo-XCR-seq MiXCR meta tables (one row per read) → spot-level receptor records.

Two layouts are recognised from the header:

* ``"mixcr_meta"``: the per-read table delivered by the Stereo-XCR-seq pipeline (MiXCR 4.6 ``exportAlignments``
  joined to ``exportClones`` by ``cloneId``, plus the ST_BarcodeMap coordinates), 28 comma-separated columns
  ``readId, targetSequences, allVAlignments, nSeqCDR3, aaSeqCDR3, topChains, cloneId, isotypePrimary, x, y, cid, umi,
  functional, CDR3Begin, CDR3End, sequence_id, sequence, productive, v_call, d_call, j_call, c_call, ..., junction,
  junction_aa, duplicate_count``. ``x`` / ``y`` are bin1 DNB coordinates (0.5 µm); ``topChains`` is the locus; the
  AIRR-like fields (``junction_aa``, ``v_call``, ``c_call``, ``productive``, ``duplicate_count``) are **clone level**
  (copied from ``exportClones``), while ``functional``, ``aaSeqCDR3`` and ``cid`` are read level. In some exports
  ``umi`` is a copy of ``cid`` in every row: there is no molecule identifier, so reads can only be collapsed to
  (DNB, clone) units. Parquet caches of this table with the same column names are read too.
* ``"stereo_xcr_public"``: ``out.meta.gz`` of the public pipeline (github.com/fengyu9481/stereo-xcr-seq,
  ``meta_build/bulid_meta_umi.py``): ``readname, AlignVHits, AlignDHits, AlignJHits, AlignCHits, cloneID, cdr3aa,
  cdr3nt, isotype, priisotype, cid, umi, x, y, readtype, loc, func, cdr3name, addinfo, CombineIsotype``. Here
  ``isotype`` is the locus (``topChains``), ``cdr3aa`` / ``cdr3nt`` are the clone's CDR3, the ``Align*Hits`` are the
  read's MiXCR hits with scores, ``umi`` is the 10-nt MID that follows the 25-nt CID on read 2, and ``readtype`` is
  ``SR`` (short read) or ``LR`` (long read, clone ids ``LR_<chain>_<id>``).

The tables can hold tens of millions of reads: only the needed columns are read, in pyarrow batches.

References
----------
Zhan et al. 2026, doi:10.1038/s41467-026-75405-x (Stereo-XCR-seq); Bolotin et al. 2015, doi:10.1038/nmeth.3364
(MiXCR).
"""

from __future__ import annotations

import gzip
import logging
import re
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from spaclone import _schema as S
from spaclone._utils import SpacloneWarning
from spaclone.io._container import SpatialAIR
from spaclone.io.airr import DNB_UM, as_scale, clean_call, finish, flag_array, format_spots, modal_codes, str_values

__all__ = ["MIXCR_ALIGNMENT_FIELDS", "parse_mixcr_alignment", "read_stereo_xcr"]

logger = logging.getLogger("spaclone")

_FORMATS: dict[str, dict[str, Any]] = {
    "mixcr_meta": {
        "detect": ("readId", "topChains", "cloneId", "x", "y"),
        "map": {
            "x": "x",
            "y": "y",
            "topChains": "locus",
            "cloneId": "clone",
            "cid": "cid",
            "umi": "umi",
            "functional": "functional",
            "aaSeqCDR3": "cdr3_aa",
            "nSeqCDR3": "cdr3_nt",
            "junction_aa": "junction_aa",
            "junction": "junction",
            "v_call": "v_call",
            "d_call": "d_call",
            "j_call": "j_call",
            "c_call": "c_call",
            "productive": "productive",
            "isotypePrimary": "isotype",
        },
        "functional_value": "Functional",
        "productive_from": "productive",
    },
    "stereo_xcr_public": {
        "detect": ("readname", "cloneID", "cdr3aa", "isotype", "x", "y"),
        "map": {
            "x": "x",
            "y": "y",
            "isotype": "locus",
            "cloneID": "clone",
            "cid": "cid",
            "umi": "umi",
            "func": "functional",
            "cdr3aa": "junction_aa",
            "cdr3nt": "junction",
            "AlignVHits": "v_call",
            "AlignDHits": "d_call",
            "AlignJHits": "j_call",
            "AlignCHits": "c_call",
            "CombineIsotype": "isotype",
            "readtype": "read_type",
        },
        "functional_value": "Func",
        "productive_from": "functional",
    },
}
#: Clone-level output fields (taken from the first read of each clone).
_CLONE_FIELDS = ("junction_aa", "junction", "v_call", "d_call", "j_call", "c_call", "productive")
#: Read-level fields returned with ``collapse=None``.
_READ_FIELDS = ("cdr3_aa", "cdr3_nt", "isotype", "read_type", "cid")
_UNASSIGNED = ("", "-1", "NA", "nan", "None")

#: Columns of :func:`parse_mixcr_alignment`, in order.
MIXCR_ALIGNMENT_FIELDS: tuple[str, ...] = (
    "ref_start",
    "ref_end",
    "ref_length",
    "query_start",
    "query_end",
    "score",
    "substitutions",
    "ambiguous_substitutions",
    "deletions",
    "insertions",
    "aligned_nt",
    "mutations",
)
_MUTATION = re.compile(r"S([A-Za-z*])(\d+)([A-Za-z*])|D([A-Za-z*])(\d+)|I(\d+)([A-Za-z*])")
_ACGT = frozenset("ACGT")


def _mutations(muts: str) -> list[tuple[str, int, str, str]]:
    """MiXCR mutation string -> [(kind, reference position, reference base, read base)] in string order."""
    out = []
    for m in _MUTATION.finditer(muts):
        if m.group(1) is not None:
            out.append(("S", int(m.group(2)), m.group(1).upper(), m.group(3).upper()))
        elif m.group(4) is not None:
            out.append(("D", int(m.group(5)), m.group(4).upper(), ""))
        else:
            out.append(("I", int(m.group(6)), "", m.group(7).upper()))
    return out


def _parse_one(text: Any, hit: int, trim: int, skip_ambiguous: bool) -> tuple:
    """Fields of hit ``hit`` of one MiXCR alignment string (NaNs when missing)."""
    nan = (np.nan,) * 11 + (None,)
    if not isinstance(text, str) or not text.strip():
        return nan
    hits = text.split(";")
    if hit >= len(hits):
        return nan
    f = hits[hit].split("|")
    if len(f) < 7 or not f[0].strip().isdigit():
        return nan
    r0, r1, rlen, q0, q1 = (int(v) for v in f[:5])
    muts = f[5]
    score = float(f[6]) if f[6].strip() else np.nan
    ev = _mutations(muts)
    del_pos = np.array(sorted(e[1] for e in ev if e[0] == "D"), dtype=np.int64)
    ins_pos = np.array(sorted(e[1] for e in ev if e[0] == "I"), dtype=np.int64)
    subs = [(e[1], e[2], e[3]) for e in ev if e[0] == "S"]
    keep = np.ones(len(subs), dtype=bool)
    lost = 0
    if trim > 0 and r1 > r0:
        pos = np.arange(r0, r1)
        offset = (pos - r0) + np.searchsorted(ins_pos, pos, side="right") - np.searchsorted(del_pos, pos, side="left")
        aligned_ref = ~np.isin(pos, del_pos)
        lost = int((aligned_ref & (offset < trim)).sum())
        if subs:
            sp = np.array([q for q, _, _ in subs], dtype=np.int64)
            so = (sp - r0) + np.searchsorted(ins_pos, sp, side="right") - np.searchsorted(del_pos, sp, side="left")
            keep = so >= trim
    n_sub = n_amb = 0
    for (_, a, b), k in zip(subs, keep, strict=True):
        if not k:
            continue
        if a in _ACGT and b in _ACGT:
            n_sub += 1
        else:
            n_amb += 1
    aligned = (r1 - r0) - len(del_pos) - lost - (n_amb if skip_ambiguous else 0)
    return r0, r1, rlen, q0, q1, score, n_sub, n_amb, len(del_pos), len(ins_pos), max(aligned, 0), muts


def parse_mixcr_alignment(
    values: Sequence[Any] | pd.Series,
    *,
    hit: int = 0,
    trim_5prime_nt: Any = 0,
    skip_ambiguous: bool = True,
) -> pd.DataFrame:
    """Parse MiXCR alignment strings (``allVAlignments``, ``bestVAlignment``, ...) into SHM counts.

    MiXCR's ``exportAlignments`` / ``exportClones`` write one alignment per hit as
    ``refStart|refEnd|refLength|queryStart|queryEnd|mutations|score`` (hits separated by ``;``), with the mutations
    written against the reference gene: ``SA223G`` = reference A at position 223 read as G, ``DC43`` = reference C
    deleted, ``I329T`` = T inserted before reference position 329 (0-based reference positions).

    Parameters
    ----------
    values
        Alignment strings, one per record (missing or empty = no alignment).
    hit
        Which hit to use: 0 = the first (best) hit, the one named first in ``allVHitsWithScore``.
    trim_5prime_nt
        Sequenced nucleotides to skip at the 5' end of the alignment (template-switch oligo, see
        :func:`spaclone.bcr.tso_trim`), as a scalar or one value per record; substitutions there are not counted and
        the aligned length is reduced.
    skip_ambiguous
        Substitutions from or to a non-ACGT character (``N``) count neither as mismatches nor as aligned positions
        (as :func:`spaclone.bcr.count_mismatches` does); ``False`` counts them as mismatches.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``values``: ``ref_start``, ``ref_end``, ``ref_length``, ``query_start``, ``query_end``,
        ``score``, ``substitutions`` (ACGT -> ACGT; with ``skip_ambiguous=False`` ambiguous ones are added),
        ``ambiguous_substitutions``, ``deletions``, ``insertions``, ``aligned_nt`` (reference positions present in the
        read: ``ref_end - ref_start - deletions``, minus trimmed and ambiguous positions; the number of columns that
        :func:`spaclone.bcr.count_mismatches` would compare in the equivalent AIRR alignment) and ``mutations`` (the
        raw mutation string).

    Notes
    -----
    Insertions and deletions are not counted as mutations (an AIRR alignment comparison skips gap columns, and in
    nanopore or long-read data most indels are homopolymer errors). Pass ``aligned_nt`` and ``substitutions`` to
    :func:`spaclone.bcr.shm` through its ``mixcr_alignment`` argument, which calls this function.

    References
    ----------
    Bolotin et al. 2015, doi:10.1038/nmeth.3364 (MiXCR).

    Examples
    --------
    >>> parse_mixcr_alignment(["223|373|391|0|150|SA223G|1470.0"])[["aligned_nt", "substitutions"]]  # doctest: +SKIP
       aligned_nt  substitutions
    0         150              1
    """
    vals = list(values) if not isinstance(values, pd.Series) else values.tolist()
    index = values.index if isinstance(values, pd.Series) else pd.RangeIndex(len(vals))
    if np.ndim(trim_5prime_nt) == 0:
        trims = np.full(len(vals), 0 if trim_5prime_nt is None else int(trim_5prime_nt), dtype=np.int64)
    else:
        trims = np.nan_to_num(np.asarray(trim_5prime_nt, dtype=float), nan=0.0).astype(np.int64)
        if len(trims) != len(vals):
            raise ValueError(f"trim_5prime_nt: expected {len(vals)} values, got {len(trims)}")
    if (trims < 0).any():
        raise ValueError("trim_5prime_nt must be >= 0")
    if hit < 0:
        raise ValueError("hit must be >= 0")
    cache: dict[tuple, tuple] = {}
    rows = []
    for v, t in zip(vals, trims, strict=True):
        key = (v if isinstance(v, str) else None, int(t))
        r = cache.get(key)
        if r is None:
            r = cache[key] = _parse_one(v, hit, int(t), skip_ambiguous)
        rows.append(r)
    out = pd.DataFrame(rows, columns=list(MIXCR_ALIGNMENT_FIELDS), index=index)
    for c in ("ref_start", "ref_end", "ref_length", "query_start", "query_end"):
        out[c] = out[c].astype("Int64")
    for c in ("substitutions", "ambiguous_substitutions", "deletions", "insertions", "aligned_nt"):
        out[c] = out[c].astype("Int64")
    if not skip_ambiguous:
        out["substitutions"] = out["substitutions"] + out["ambiguous_substitutions"]
    return out


def _header(path: Path) -> tuple[list[str], str]:
    """Column names and delimiter of a (gzipped) CSV / TSV, or the schema names of a Parquet file."""
    if path.suffix == ".parquet":
        return list(pq.ParquetFile(path).schema_arrow.names), ""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        line = fh.readline().rstrip("\r\n")
    sep = "\t" if line.count("\t") > line.count(",") else ","
    return [c.strip().strip('"') for c in line.split(sep)], sep


def _detect(cols: Sequence[str]) -> str:
    have = set(cols)
    for name, spec in _FORMATS.items():
        if all(c in have for c in spec["detect"]):
            return name
    raise ValueError(
        "unrecognised Stereo-XCR-seq meta table: expected the MiXCR per-read export (readId, topChains, cloneId, x, "
        "y, ...) or the public pipeline's out.meta.gz (readname, cloneID, cdr3aa, isotype, x, y, ...); got "
        f"columns {list(cols)[:12]}"
    )


def _batches(path: Path, src: list[str], sep: str, batch_rows: int) -> Iterator[pa.RecordBatch]:
    """Yield pyarrow RecordBatches holding the source columns ``src`` (strings, except numeric ``x`` / ``y``)."""
    if path.suffix == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=src)
        return
    types: dict[str, pa.DataType] = {c: pa.string() for c in src}
    types.update({"x": pa.float64(), "y": pa.float64()})
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter=sep),
        convert_options=pacsv.ConvertOptions(
            include_columns=src,
            column_types=types,
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
            null_values=[""],
        ),
    )
    yield from reader


def _as_str(col: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return col if pa.types.is_string(col.type) or pa.types.is_large_string(col.type) else pc.cast(col, pa.string())


def _np_str(col: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Arrow strings → object ndarray (``None`` for nulls)."""
    return _as_str(col).to_numpy(zero_copy_only=False)


def _hash(col: pa.Array) -> np.ndarray:
    """Stable 64-bit hash (as int64) of a string column; nulls hash like the empty string."""
    arr = pc.fill_null(_as_str(col), "").to_numpy(zero_copy_only=False).astype(object)
    return pd.util.hash_array(arr, categorize=True).view(np.int64)


def _group(*cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Dense group id per row for the combination of integer columns, and the first row (in file order) of each group.

    Groups are numbered in the sort order of the key columns.
    """
    if len(cols[0]) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    order = np.lexsort(cols[::-1])  # stable: the first sorted row of a group is its first row in file order
    new = np.zeros(len(order), dtype=bool)
    new[0] = True
    for c in cols:
        cs = c[order]
        new[1:] |= cs[1:] != cs[:-1]
    gid_sorted = np.cumsum(new) - 1
    gid = np.empty(len(order), dtype=np.int64)
    gid[order] = gid_sorted
    return gid, order[new], int(gid_sorted[-1]) + 1


def _n_distinct(gid: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Number of distinct ``values`` per group."""
    if len(gid) == 0:
        return np.zeros(n, np.int64)
    o = np.lexsort((values, gid))
    g, v = gid[o], values[o]
    new = np.r_[True, (g[1:] != g[:-1]) | (v[1:] != v[:-1])]
    return np.bincount(g[new], minlength=n)


def read_stereo_xcr(
    path: str | Path,
    *,
    collapse: Literal["spot_clone", "molecule", "auto"] | None = "spot_clone",
    loci: Sequence[str] | None = None,
    columns: Sequence[str] = (),
    spot_format: str = "{x}_{y}",
    scale_um: float = DNB_UM,
    library_id: str | None = None,
    batch_rows: int = 1_000_000,
) -> SpatialAIR:
    """Read a Stereo-XCR-seq per-read meta table (``*.meta.gz``, the public ``out.meta.gz``, or a Parquet cache).

    Parameters
    ----------
    path
        The meta table (gzipped or plain CSV / TSV) or a Parquet file with the same column names.
    collapse
        ``"spot_clone"`` (default): one record per (DNB, locus, clone) unit, the smallest PCR-duplicate-free unit
        when there is no molecule identifier. ``"molecule"``: one record per (DNB, MID, locus) molecule, its clone the
        most frequent clone of its reads (ties: the alphabetically first); needs a real ``umi`` column and falls back
        to ``"spot_clone"`` with a warning when ``umi`` is a copy of ``cid``. ``"auto"``: ``"molecule"`` if there is a
        molecule identifier, else ``"spot_clone"``. ``None``: one record per read.
    loci
        Keep only these loci (``topChains`` / ``isotype``), e.g. ``("IGH", "IGK", "IGL")``. Default: every locus.
    columns
        Further source columns to keep. With ``collapse=None`` they are read level; otherwise they are taken from
        each clone's first read (meaningful for clone-level fields such as ``duplicate_count``).
    spot_format
        Spot id template over the integer DNB coordinates (``"{x}_{y}"``, as the public pipeline's ``loc``).
    scale_um
        Micrometres per DNB (0.5 for Stereo-seq).
    library_id
        Library label; default: the file name up to its first dot.
    batch_rows
        Rows per Parquet batch (CSV is read in 64-MB blocks).

    Returns
    -------
    SpatialAIR
        ``air`` (``gex`` is ``None``), one row per record, every DNB of the chip kept (on and off the tissue):
        ``spot_id``, ``x_um``, ``y_um``, ``locus``, ``clone_id`` (MiXCR clone id), the clone-level ``junction_aa``,
        ``junction``, ``v_call``, ``d_call``, ``j_call``, ``c_call`` (first hit, score removed), ``productive``,
        ``consensus_count`` (reads of the record), ``umi`` (the MID for molecules and reads with a real MID, else
        missing), ``library_id``, ``x_src`` / ``y_src`` (DNB), ``f_functional`` (share of the record's reads whose
        CDR3 is in frame and stop-free), ``n_cid`` (distinct raw CID sequences among its reads: barcode errors
        corrected by ST_BarcodeMap) and, with a real MID, ``n_umi`` (distinct MIDs) or ``n_clones_in_molecule``. When
        the table has read-level CDR3s, ``junction_aa_top`` is the clone's most frequent read CDR3
        (ties: the one read first, as :meth:`pandas.Series.value_counts`); it can differ from MiXCR's consensus
        ``junction_aa`` (the archive's validity rule used it). With ``collapse=None``: one row per read with ``functional`` (bool), ``cdr3_aa`` (the read's CDR3), ``cid`` and
        the read-level source columns requested. ``meta``: layout, rows read, reads kept per locus,
        ``umi_equals_cid_fraction``, ``has_molecule_id``, the collapse used (``level``), coordinate unit and factor.

    Notes
    -----
    ``duplicate_count`` of the MiXCR export is the clone's read count copied onto every read; it must never be summed
    over rows. Reads without a clone (``cloneId`` empty or ``-1``) are not records; they are counted in
    ``meta["n_reads_unassigned"]``.

    Examples
    --------
    >>> sa = read_stereo_xcr("chip_read1.fq.gz.meta.gz", loci=("IGH", "IGK", "IGL"))  # doctest: +SKIP
    >>> sa.meta["umi_equals_cid_fraction"], sa.meta["level"]  # doctest: +SKIP
    (1.0, 'spot_clone')
    """
    path = Path(path)
    if collapse not in ("spot_clone", "molecule", "auto", None):
        raise ValueError("collapse must be 'spot_clone', 'molecule', 'auto' or None")
    um, _ = as_scale(scale_um)
    format_spots([0], [0], spot_format)  # validate the template early
    cols, sep = _header(path)
    fmt = _detect(cols)
    spec = _FORMATS[fmt]
    mapping = {s: t for s, t in spec["map"].items() if s in cols}
    for extra in columns:
        if extra not in cols:
            raise ValueError(f"column {extra!r} not in the meta table")
    inv = {t: s for s, t in mapping.items()}
    src = list(dict.fromkeys([*mapping, *columns]))
    has_umi_col = "umi" in inv and "cid" in inv
    keep_reads = collapse is None
    top_aa = collapse is not None and "cdr3_aa" in inv  # read-level CDR3 available
    loci_set = None if loci is None else [str(lc).upper() for lc in loci]

    pair_code: dict[tuple[str, str], int] = {}
    clone_rows: list[pd.DataFrame] = []
    xs, ys, codes, func, cid_h, umi_h = [], [], [], [], [], []
    umi_arrays: list[pa.Array] = []
    read_cols: dict[str, list[Any]] = {}
    aa_counts: list[pd.DataFrame] = []
    n_rows = n_kept = n_eq = n_both = n_unassigned = 0
    for batch in _batches(path, src, sep, batch_rows):
        n_rows += batch.num_rows
        locus_col = pc.utf8_upper(pc.utf8_trim_whitespace(_as_str(batch.column(inv["locus"]))))
        clone_col = pc.utf8_trim_whitespace(_as_str(batch.column(inv["clone"])))
        assigned = pc.invert(pc.fill_null(pc.is_in(clone_col, value_set=pa.array(_UNASSIGNED)), True))
        if loci_set is not None:
            in_loci = pc.fill_null(pc.is_in(locus_col, value_set=pa.array(loci_set)), False)
        else:
            in_loci = pa.array(np.ones(batch.num_rows, dtype=bool))
        n_unassigned += int(pc.sum(pc.and_(in_loci, pc.invert(assigned))).as_py() or 0)
        mask = pc.and_(in_loci, assigned)
        if not pc.all(mask).as_py():
            batch, locus_col, clone_col = batch.filter(mask), locus_col.filter(mask), clone_col.filter(mask)
        if batch.num_rows == 0:
            continue
        n_kept += batch.num_rows
        if has_umi_col:
            u, c = _as_str(batch.column(inv["umi"])), _as_str(batch.column(inv["cid"]))
            n_both += int(pc.sum(pc.and_(pc.is_valid(u), pc.is_valid(c))).as_py() or 0)
            n_eq += int(pc.sum(pc.fill_null(pc.equal(u, c), False)).as_py() or 0)
        x = batch.column("x").to_numpy(zero_copy_only=False).astype(float)
        y = batch.column("y").to_numpy(zero_copy_only=False).astype(float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("the meta table has reads without x / y coordinates")
        loc = locus_col.to_numpy(zero_copy_only=False).astype(object)
        clo = clone_col.to_numpy(zero_copy_only=False).astype(object)
        local, uniq = pd.factorize(pd.Series(loc, dtype=object) + "\x1f" + pd.Series(clo, dtype=object))
        glob = np.empty(len(uniq), dtype=np.int64)
        new_local = []
        for i, k in enumerate(uniq):
            lc, cl = str(k).split("\x1f", 1)
            g = pair_code.get((lc, cl))
            if g is None:
                g = pair_code[(lc, cl)] = len(pair_code)
                new_local.append(i)
            glob[i] = g
        code = glob[local]
        if new_local:
            first_rows = pd.Series(np.arange(len(local))).groupby(local).first().to_numpy()
            take = pa.array(first_rows[np.asarray(new_local)])
            cc: dict[str, Any] = {"_code": glob[np.asarray(new_local)]}
            for t in _CLONE_FIELDS:
                s = inv.get(spec["productive_from"]) if t == "productive" else inv.get(t)
                if s is not None:
                    cc[t] = _np_str(batch.column(s).take(take))
            if not keep_reads:
                for extra in columns:
                    cc[extra] = batch.column(extra).take(take).to_numpy(zero_copy_only=False)
            clone_rows.append(pd.DataFrame(cc))
        if "functional" in inv:
            f = pc.fill_null(pc.equal(_as_str(batch.column(inv["functional"])), spec["functional_value"]), False)
            func.append(f.to_numpy(zero_copy_only=False))
        else:
            func.append(np.ones(batch.num_rows, dtype=bool))
        xs.append(x)
        ys.append(y)
        codes.append(code)
        if "cid" in inv:
            cid_h.append(_hash(batch.column(inv["cid"])))
        if has_umi_col:
            umi_h.append(_hash(batch.column(inv["umi"])))
            if collapse != "spot_clone":
                umi_arrays.append(_as_str(batch.column(inv["umi"])))
        if top_aa:
            aa = _as_str(batch.column(inv["cdr3_aa"])).to_numpy(zero_copy_only=False)
            pos = np.arange(n_kept - batch.num_rows, n_kept)
            aa_counts.append(
                pd.DataFrame({"c": code, "a": aa, "i": pos}).groupby(["c", "a"], sort=False).i.agg(["size", "min"])
            )
        if keep_reads:
            for t in _READ_FIELDS:
                s = inv.get(t)
                if s is not None:
                    read_cols.setdefault(t, []).append(_as_str(batch.column(s)))
            for extra in columns:
                read_cols.setdefault(extra, []).append(batch.column(extra))

    umi_eq = n_eq / n_both if n_both else float("nan")
    has_mid = bool(has_umi_col and n_both > 0 and umi_eq < 0.99)
    used = collapse
    if collapse == "auto":
        used = "molecule" if has_mid else "spot_clone"
    elif collapse == "molecule" and not has_mid:
        warnings.warn(
            "the meta table has no molecule identifier (umi is a copy of cid, or missing); reads are collapsed to "
            "(DNB, clone) units instead of molecules",
            SpacloneWarning,
            stacklevel=2,
        )
        used = "spot_clone"

    x = np.concatenate(xs) if xs else np.zeros(0)
    y = np.concatenate(ys) if ys else np.zeros(0)
    code = np.concatenate(codes) if codes else np.zeros(0, np.int64)
    fun = np.concatenate(func) if func else np.zeros(0, bool)
    if len(x) and not (np.array_equal(x, np.round(x)) and np.array_equal(y, np.round(y))):
        raise ValueError("DNB coordinates must be integers")
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    pairs = sorted(pair_code.items(), key=lambda kv: kv[1])
    pair_locus = np.array([p[0][0] for p in pairs], dtype=object)
    pair_clone = np.array([p[0][1] for p in pairs], dtype=object)
    clones = (
        pd.concat(clone_rows, ignore_index=True).drop_duplicates("_code").set_index("_code").sort_index()
        if clone_rows
        else pd.DataFrame(index=pd.Index([], name="_code"))
    )
    lib = library_id if library_id is not None else path.name.split(".")[0]
    umi_all = pa.chunked_array(umi_arrays, type=pa.string()) if umi_arrays else None

    extra_cols: dict[str, Any] = {}
    if used is None:
        rec_code, first = code, np.arange(len(code))
        reads = np.ones(len(code))
        umi_vals = _np_str(umi_all) if (umi_all is not None and has_mid) else np.full(len(code), np.nan, object)
        extra_cols["functional"] = fun
        for c, arrs in read_cols.items():
            v = pa.chunked_array(arrs).to_numpy(zero_copy_only=False)
            extra_cols[c] = str_values(v) if v.dtype == object else v
        level = "read"
    elif used == "spot_clone":
        gid, first, n = _group(xi, yi, code)
        cnt = np.bincount(gid, minlength=n)
        rec_code = code[first]
        reads = cnt.astype(float)
        umi_vals = np.full(n, np.nan, dtype=object)
        extra_cols["f_functional"] = np.bincount(gid, weights=fun.astype(float), minlength=n) / np.maximum(cnt, 1)
        if cid_h:
            extra_cols["n_cid"] = _n_distinct(gid, np.concatenate(cid_h), n)
        if has_mid and umi_h:
            extra_cols["n_umi"] = _n_distinct(gid, np.concatenate(umi_h), n)
        level = "spot_clone"
    else:  # molecule: (DNB, MID, locus)
        loc_codes = pd.factorize(pair_locus[code])[0].astype(np.int64) if len(code) else np.zeros(0, np.int64)
        gid, first, n = _group(xi, yi, loc_codes, np.concatenate(umi_h))
        cnt = np.bincount(gid, minlength=n)
        # rank pairs alphabetically by clone id so that ties resolve like pandas.Series.mode
        pair_rank = np.argsort(np.argsort(pair_clone.astype(str), kind="stable"), kind="stable")
        best_rank = modal_codes(gid, pair_rank[code], n)
        rec_code = np.argsort(pair_rank)[best_rank]
        reads = cnt.astype(float)
        umi_vals = _np_str(umi_all.take(pa.array(first)))  # type: ignore[union-attr]
        extra_cols["f_functional"] = np.bincount(gid, weights=fun.astype(float), minlength=n) / np.maximum(cnt, 1)
        extra_cols["n_clones_in_molecule"] = _n_distinct(gid, code, n)
        if cid_h:
            extra_cols["n_cid"] = _n_distinct(gid, np.concatenate(cid_h), n)
        level = "molecule"

    air = pd.DataFrame(
        {S.SPOT: format_spots(xi[first], yi[first], spot_format), S.X: x[first] * um, S.Y: y[first] * um}
    )
    air[S.LOCUS] = pair_locus[rec_code] if len(rec_code) else np.zeros(0, object)
    air[S.CLONE] = pair_clone[rec_code] if len(rec_code) else np.zeros(0, object)
    for t in _CLONE_FIELDS:
        if t not in clones.columns:
            continue
        vals = clones[t].reindex(rec_code).to_numpy(dtype=object)
        if t in ("v_call", "d_call", "j_call", "c_call"):
            vals = clean_call(vals)
        elif t == "productive":
            vals = (
                flag_array(vals)
                if fmt == "mixcr_meta"
                else pd.Series(vals, dtype=object).eq(spec["functional_value"]).to_numpy(dtype=bool)
            )
        air[t] = vals
    air[S.READS] = reads
    air[S.UMI] = umi_vals
    air[S.LIBRARY] = lib
    air["x_src"], air["y_src"] = xi[first], yi[first]
    for c, v in extra_cols.items():
        air[c] = v
    if used is not None:
        for extra in columns:
            if extra in clones.columns:
                air[extra] = clones[extra].reindex(rec_code).to_numpy()
    if aa_counts:
        tot = pd.concat(aa_counts).groupby(level=[0, 1], sort=False).agg({"size": "sum", "min": "min"}).reset_index()
        tot = tot.sort_values(["c", "size", "min"], ascending=[True, False, True], kind="stable").drop_duplicates("c")
        air["junction_aa_top"] = tot.set_index("c")["a"].reindex(rec_code).to_numpy(dtype=object)

    by_locus = pd.Series(pair_locus[code] if len(code) else [], dtype=object).value_counts().sort_index()
    meta: dict[str, Any] = {
        "reader": "read_stereo_xcr",
        "technology": "Stereo-XCR-seq",
        "layout": fmt,
        "files": [str(path)],
        "library_id": lib,
        "level": level,
        "molecule_level": True,
        "source_unit": "Stereo-seq DNB (bin1, 0.5 µm)" if um == DNB_UM else "DNB",
        "um_per_unit": float(um),
        "spot_format": spot_format,
        "n_rows_read": int(n_rows),
        "n_reads": int(n_kept),
        "n_reads_unassigned": int(n_unassigned),
        "reads_by_locus": {str(k): int(v) for k, v in by_locus.items()},
        "n_clones": int(len(pair_code)),
        "umi_equals_cid_fraction": float(umi_eq),
        "has_molecule_id": has_mid,
        "has_read_counts": True,
        "collapse_requested": collapse,
        "collapse": used,
        "loci": loci_set,
    }
    meta["io"] = {
        "reader": "read_stereo_xcr",
        "format": fmt,
        "evidence": f"header columns {', '.join(spec['detect'])}",
        "coordinates": "columns x / y: bin1 DNB",
        "um_per_unit": {"value": float(um), "rule": f"scale_um = {um:g} (Stereo-seq DNB 0.5 µm)"},
        "molecule_id": (
            f"umi equals cid in {umi_eq:.1%} of reads: no molecule identifier" if not has_mid else "real MID"
        )
        if n_both
        else "no umi column",
        "collapse": f"{used} (requested {collapse})",
        "reads": "one row per read; consensus_count = reads of the record",
    }
    logger.info(
        "read_stereo_xcr: %d of %d reads kept, %d records (%s); umi == cid in %.1f%% of reads",
        n_kept,
        n_rows,
        len(air),
        used,
        100 * umi_eq if n_both else float("nan"),
    )
    return finish(air, None, meta)
