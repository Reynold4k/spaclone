"""Somatic hypermutation (SHM) per record from AIRR alignments, the allele baseline, locus calls and SHM maps.

V mismatches are counted between the sequenced and the germline V alignment of each record (AIRR fields
``v_sequence_alignment`` / ``v_germline_alignment``), skipping alignment gaps. Two artefacts are removed before a
record is called mutated:

* **germline allele differences**: the subject's alleles can differ from the reference allele the aligner assigned.
  The baseline of an allele is the median mismatch count of that allele over unswitched (IgM / IgD) records, which
  are mostly naive and unmutated; the *excess* of a record is its mismatches minus the baseline of its allele.
* **assembly and PCR errors**: low-read contigs carry spurious mismatches (in the 2026-10 StereoIR data, 20-38% of
  IgD contigs below 30 reads had >= 3 excess mismatches, 3-5% at >= 30 reads), so only records with enough reads and
  enough aligned V nucleotides are *evaluable*.

A record is mutated when it is evaluable and its excess reaches ``min_excess`` (T1: >= 3 V mismatches). Loci and
clones are called from their evaluable records (:func:`call_shm`). :func:`shm_map` and :func:`shm_hotspots` map the
mutated share of loci with an isotype-matched expectation (archive T7), through :func:`spaclone.tl.share_map` and
:func:`spaclone.tl.hotspots`.

For Stereo-XCR-seq, whose template-switch oligo (TSO) sits inside FR3, the first nucleotides of each V alignment are
TSO-derived and must be trimmed (``trim_5prime_nt``, :func:`tso_trim`). :func:`run_igblast` calls an installed
IgBLAST to produce AIRR alignments.

References
----------
Ye et al. 2013, doi:10.1093/nar/gkt382 (IgBLAST); Gadala-Maria et al. 2015, doi:10.1073/pnas.1417683112 (novel
alleles and genotypes, TIgGER); Gupta et al. 2015, doi:10.1093/bioinformatics/btv359 (Change-O); Kelsall & Diggle
1998, doi:10.1111/1467-9876.00128 (kernel share maps).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import RngLike
from spaclone.bcr.isotype import isotype as _isotype
from spaclone.pp.clones import gene_name

__all__ = [
    "allele_baseline",
    "call_shm",
    "count_mismatches",
    "run_igblast",
    "shm",
    "shm_hotspots",
    "shm_map",
    "tso_trim",
]

logger = logging.getLogger("spaclone")

_CHUNK = 20_000
_IS_GAP = np.zeros(256, dtype=bool)
_IS_GAP[[ord("-"), ord("."), 0]] = True
_IS_ACGT = np.zeros(256, dtype=bool)
_IS_ACGT[[ord(c) for c in "ACGT"]] = True

Region = Literal["V", "V_J"] | tuple[int, int]


def _matrix(strings: Sequence[Any]) -> np.ndarray:
    """Upper-cased strings as a uint8 matrix padded with 0 (missing values are empty strings)."""
    s = [x.upper() if isinstance(x, str) else "" for x in strings]
    width = max((len(x) for x in s), default=0)
    if width == 0:
        return np.zeros((len(s), 0), dtype=np.uint8)
    buf = "".join(x.ljust(width, "\0") for x in s).encode("ascii", "replace")
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(s), width)


def _per_record(value: Any, n: int, air: pd.DataFrame | None, what: str) -> np.ndarray:
    """A scalar, a column name or an array with one value per record, as an int array."""
    if isinstance(value, str):
        if air is None or value not in air.columns:
            raise KeyError(f"{what}: column {value!r} not in the table")
        value = air[value]
    if np.ndim(value) == 0:
        arr = np.full(n, 0.0 if value is None else float(value))
    else:
        arr = pd.to_numeric(pd.Series(np.asarray(value, dtype=object)), errors="coerce").to_numpy(dtype=float)
        if len(arr) != n:
            raise ValueError(f"{what}: expected {n} values, got {len(arr)}")
    arr = np.nan_to_num(arr, nan=0.0)
    if (arr < 0).any():
        raise ValueError(f"{what} must be >= 0")
    return arr.astype(np.int64)


def count_mismatches(
    sequence: Sequence[Any] | pd.Series,
    germline: Sequence[Any] | pd.Series,
    *,
    start: int | None = None,
    end: int | None = None,
    trim_5prime_nt: Any = 0,
    skip_ambiguous: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned nucleotides and mismatches between paired sequence and germline alignment strings.

    Parameters
    ----------
    sequence, germline
        Alignment strings of equal register (AIRR ``*_sequence_alignment`` / ``*_germline_alignment``, or the
        IMGT-gapped ``sequence_alignment`` / ``germline_alignment``). Missing values count 0 / 0.
    start, end
        1-based, inclusive alignment columns to use (e.g. ``1, 312`` = IMGT V positions FWR1-FWR3 in IMGT-gapped
        alignments). Default: the whole alignment.
    trim_5prime_nt
        Number of sequenced nucleotides (non-gap characters of ``sequence``) at the 5' end of the used window to
        skip, as a scalar or one value per record (template-switch oligo inside the V gene).
    skip_ambiguous
        Skip columns where either string is not A / C / G / T (``N``, IUPAC codes). ``False`` reproduces the 2026-10
        archive (T1), which skipped only gap characters (``-`` and ``.``) and counted ``N`` against a base as a
        mismatch.

    Returns
    -------
    aligned_nt, mismatches
        Integer arrays: compared columns and differing columns per record. Columns where either string is a gap or
        has ended are never compared; strings are compared in upper case.
    """
    seq = list(sequence)
    ger = list(germline)
    if len(seq) != len(ger):
        raise ValueError("sequence and germline must have the same length")
    n = len(seq)
    trim = _per_record(trim_5prime_nt, n, None, "trim_5prime_nt")
    if start is not None and start < 1:
        raise ValueError("start is 1-based and must be >= 1")
    if start is not None and end is not None and end < start:
        raise ValueError("end must be >= start")
    lo = 0 if start is None else start - 1
    aligned = np.zeros(n, dtype=np.int64)
    mism = np.zeros(n, dtype=np.int64)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        q = _matrix(seq[a:b])[:, lo:end]
        g = _matrix(ger[a:b])[:, lo:end]
        w = min(q.shape[1], g.shape[1])
        q, g = q[:, :w], g[:, :w]
        if w == 0:
            continue
        ok = ~_IS_GAP[q] & ~_IS_GAP[g]
        if skip_ambiguous:
            ok &= _IS_ACGT[q] & _IS_ACGT[g]
        t = trim[a:b]
        if t.any():
            sequenced = np.cumsum(~_IS_GAP[q], axis=1)
            ok &= sequenced > t[:, None]
        aligned[a:b] = ok.sum(1)
        mism[a:b] = (ok & (q != g)).sum(1)
    return aligned, mism


def tso_trim(
    air: pd.DataFrame,
    *,
    pattern: str = r"GAGAGGC?G{2,4}",
    extra_nt: int = 3,
    sequence_alignment: str = "v_sequence_alignment",
) -> pd.Series:
    """Nucleotides to trim at the 5' end of each V alignment because a template-switch oligo (TSO) sits there.

    The TSO of Stereo-seq (``...CTGACGTACTGAGAGGC`` + ``GGG``) lies inside FR3 in many Stereo-XCR-seq reads, so the
    V-aligned stretch 5' of (and including) it is not template. The trim of a record is the end of the last match of
    ``pattern`` in its V sequence alignment (gaps removed) plus ``extra_nt`` non-templated bases; 0 without a match.

    Parameters
    ----------
    air
        Receptor table.
    pattern
        Regular expression of the TSO 3' end (default: the archive's ``GAGAGGC?G{2,4}``, chapter 06).
    extra_nt
        Further bases skipped after the match (the archive skipped 3).
    sequence_alignment
        Column with the V sequence alignment.

    Returns
    -------
    pandas.Series
        Integer trim per record, for ``shm(..., trim_5prime_nt=...)``.
    """
    if sequence_alignment not in air.columns:
        raise KeyError(f"tso_trim: column {sequence_alignment!r} not in the table")
    rx = re.compile(pattern)
    out = np.zeros(len(air), dtype=np.int64)
    for k, s in enumerate(air[sequence_alignment].to_numpy(dtype=object)):
        if not isinstance(s, str):
            continue
        u = s.upper().replace("-", "").replace(".", "")
        last = None
        for m in rx.finditer(u):
            last = m
        if last is not None:
            out[k] = min(len(u), last.end() + extra_nt)
    return pd.Series(out, index=air.index, name="tso_trim")


def _isotypes(air: pd.DataFrame, isotype_key: str, c_key: str) -> pd.Series:
    if isotype_key in air.columns:
        return air[isotype_key].astype("string")
    if c_key in air.columns:
        return _isotype(air, c_key=c_key, species=None)["isotype"]
    return pd.Series(pd.NA, index=air.index, dtype="string")


BaselineFallback = Literal["none", "gene", "pooled", "zero"]


def allele_baseline(
    air: pd.DataFrame,
    mismatches: Any,
    *,
    aligned_nt: Any = None,
    baseline_isotypes: Sequence[str] = ("IgM", "IgD"),
    min_baseline_n: int = 10,
    min_reads: float = 30,
    min_aligned_nt: int = 250,
    fallback: BaselineFallback = "none",
    v_key: str = S.V_CALL,
    reads_key: str = S.READS,
    isotype_key: str = "isotype",
    c_key: str = S.C_CALL,
) -> pd.DataFrame:
    """Median V mismatches of every V allele over high-quality unswitched records, with an explicit fallback.

    Parameters
    ----------
    air
        Receptor records.
    mismatches
        Column name or array: V mismatches per record.
    aligned_nt
        Column name or array: aligned V nucleotides per record (for the ``min_aligned_nt`` floor).
    baseline_isotypes
        Isotype classes whose records define the baseline (IgM and IgD: mostly naive, unmutated B cells).
    min_baseline_n
        An allele (or gene, or the pooled set) needs at least this many such records for its median to be used.
    min_reads, min_aligned_nt
        Floors of the records used (reads >= ``min_reads``, aligned V nucleotides >= ``min_aligned_nt``).
    fallback
        What an allele with fewer than ``min_baseline_n`` baseline records gets. ``"none"`` (default, the archive's
        T1 rule): no baseline, so its records are not evaluable. ``"gene"``: the median over the baseline records of
        all alleles of the same gene (``IGHV1-26*01`` and ``*02`` pooled), when the gene reaches ``min_baseline_n``.
        ``"pooled"``: the gene, else the median over all baseline records of the data set. ``"zero"``: the gene, else
        0 (no germline correction: mismatches count as mutations). Small data sets often have few unswitched records
        per allele (AIR-SPACE D21: 5 of 58 MiXCR V genes reach 10 IgM / IgD molecules), and with a reference that
        matches the strain the allele medians are near 0, so ``"zero"`` or ``"pooled"`` lose little; with a
        mismatched reference keep ``"none"`` or genotype the subject first (TIgGER).
    v_key, reads_key, isotype_key, c_key
        Column names; the allele is the ``v_key`` string as given (one call per record expected).

    Returns
    -------
    pandas.DataFrame
        Indexed by every allele present in ``air`` (sorted): ``baseline`` (the value used; NaN when none applies),
        ``n`` (baseline records of the allele), ``source`` (``"allele"``, ``"gene"``, ``"pooled"``, ``"zero"`` or
        ``"none"``) and ``n_source`` (records behind the value used).
    """
    if fallback not in ("none", "gene", "pooled", "zero"):
        raise ValueError(f"fallback must be 'none', 'gene', 'pooled' or 'zero', got {fallback!r}")
    n = len(air)
    mm = np.asarray(air[mismatches] if isinstance(mismatches, str) else mismatches, dtype=float)
    al = (
        np.full(n, np.inf)
        if aligned_nt is None
        else np.asarray(air[aligned_nt] if isinstance(aligned_nt, str) else aligned_nt, dtype=float)
    )
    if v_key not in air.columns:
        raise KeyError(f"allele_baseline: column {v_key!r} not in the table")
    reads = _reads(air, reads_key, min_reads)
    iso = _isotypes(air, isotype_key, c_key)
    use = (reads >= min_reads) & (al >= min_aligned_nt) & iso.isin(list(baseline_isotypes)).fillna(False).to_numpy()
    use &= air[v_key].notna().to_numpy() & np.isfinite(mm)
    alleles = air[v_key].to_numpy(dtype=object)
    d = pd.DataFrame({"allele": alleles[use], "mm": mm[use]})
    g = d.groupby("allele", sort=True).mm
    present = pd.Index(sorted({a for a in alleles if isinstance(a, str)}), name=v_key)
    tab = pd.DataFrame({"baseline": g.median(), "n": g.size().astype(np.int64)}).reindex(present)
    tab["n"] = tab["n"].fillna(0).astype(np.int64)
    ok = tab.n >= min_baseline_n
    tab.loc[~ok, "baseline"] = np.nan
    tab["source"] = np.where(ok, "allele", "none")
    tab["n_source"] = np.where(ok, tab.n, 0).astype(np.int64)
    if fallback != "none":
        genes = gene_name(pd.Series(tab.index, dtype="string")).to_numpy(dtype=object)
        dg = pd.DataFrame({"gene": gene_name(d.allele).to_numpy(dtype=object), "mm": d.mm.to_numpy()})
        gg = dg.groupby("gene", sort=True).mm
        g_med, g_n = gg.median(), gg.size()
        gene_ok = pd.Series(genes).map(g_n).fillna(0).to_numpy() >= min_baseline_n
        take = ~ok.to_numpy() & gene_ok
        tab.loc[take, "baseline"] = pd.Series(genes[take]).map(g_med).to_numpy(dtype=float)
        tab.loc[take, "source"] = "gene"
        tab.loc[take, "n_source"] = pd.Series(genes[take]).map(g_n).to_numpy(dtype=np.int64)
        rest = ~ok.to_numpy() & ~gene_ok
        if fallback == "pooled" and len(d) >= min_baseline_n:
            tab.loc[rest, "baseline"] = float(d.mm.median())
            tab.loc[rest, "source"] = "pooled"
            tab.loc[rest, "n_source"] = len(d)
        elif fallback == "zero":
            tab.loc[rest, "baseline"] = 0.0
            tab.loc[rest, "source"] = "zero"
    return tab


def _reads(air: pd.DataFrame, reads_key: str, min_reads: float) -> np.ndarray:
    if reads_key in air.columns:
        return pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)
    if min_reads > 0:
        raise KeyError(f"column {reads_key!r} (reads per record) not in the table; pass reads_key or min_reads=0")
    return np.full(len(air), np.inf)


def shm(
    air: pd.DataFrame,
    *,
    sequence_alignment: str = "v_sequence_alignment",
    germline_alignment: str = "v_germline_alignment",
    region: Region = "V",
    j_sequence_alignment: str = "j_sequence_alignment",
    j_germline_alignment: str = "j_germline_alignment",
    baseline: Literal["allele"] | None = "allele",
    baseline_isotypes: Sequence[str] = ("IgM", "IgD"),
    min_baseline_n: int = 10,
    baseline_fallback: BaselineFallback = "none",
    min_reads: float = 30,
    min_aligned_nt: int = 250,
    min_excess: float = 3,
    trim_5prime_nt: Any = 0,
    skip_ambiguous: bool = True,
    mixcr_alignment: str | None = None,
    mixcr_j_alignment: str | None = None,
    v_key: str = S.V_CALL,
    reads_key: str = S.READS,
    isotype_key: str = "isotype",
    c_key: str = S.C_CALL,
    key_added: str = "shm",
    return_baseline: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame | None]:
    """V mismatches against the germline, excess over the allele baseline, evaluability and mutated flag per record.

    Parameters
    ----------
    air
        Receptor records (filter them first, e.g. with :func:`spaclone.pp.filter_valid`; the baseline is computed
        from the records given). Heavy and light chains are handled alike; the baseline is per V allele.
    sequence_alignment, germline_alignment
        Columns with the aligned sequence and germline. Defaults: the AIRR V alignments.
    region
        ``"V"``: the whole V alignment in those columns (archive T1). ``"V_J"``: V plus the J alignment
        (``j_sequence_alignment`` / ``j_germline_alignment``), e.g. for short reads covering the FR3 3' end and J
        (Stereo-XCR-seq). ``(start, end)``: 1-based inclusive columns of ``sequence_alignment`` /
        ``germline_alignment``, e.g. ``(1, 312)`` = IMGT V positions FWR1-FWR3 of IMGT-gapped alignments.
    j_sequence_alignment, j_germline_alignment
        J alignment columns for ``region="V_J"``.
    baseline
        ``"allele"``: subtract the median mismatches of the record's V allele over high-quality IgM / IgD records
        (:func:`allele_baseline`); records whose allele has no baseline are not evaluable. ``None``: excess =
        mismatches.
    baseline_isotypes, min_baseline_n
        See :func:`allele_baseline`.
    baseline_fallback
        Baseline of alleles with fewer than ``min_baseline_n`` baseline records (``"none"``, ``"gene"``,
        ``"pooled"``, ``"zero"``), see :func:`allele_baseline`. Which records got which baseline is in
        ``{key}_baseline_source``.
    min_reads, min_aligned_nt
        Evaluability floors (T1: >= 30 reads and >= 250 aligned V nucleotides); the same floors select the baseline
        records.
    min_excess
        Excess mismatches at which an evaluable record is mutated (T1: 3).
    trim_5prime_nt
        Sequenced nucleotides to skip at the 5' end of the V window: a number, a column name or one value per
        record (see :func:`tso_trim`).
    skip_ambiguous
        Skip columns with non-ACGT characters; ``False`` = the archive's rule (only gaps skipped).
    mixcr_alignment
        Column with MiXCR alignment strings of the V gene (``allVAlignments`` / ``bestVAlignment``, e.g. from
        :func:`spaclone.io.read_airspace` with ``alignments=True``) to use instead of the AIRR alignment strings:
        aligned nucleotides and substitutions come from :func:`spaclone.io.parse_mixcr_alignment` (first hit;
        insertions and deletions not counted, as gap columns are not in an AIRR comparison). ``region`` must then be
        ``"V"`` or ``"V_J"``.
    mixcr_j_alignment
        MiXCR J alignment column for ``region="V_J"`` with ``mixcr_alignment``.
    v_key, reads_key, isotype_key, c_key
        Column names (the isotype class is derived from ``c_key`` when ``isotype_key`` is absent).
    key_added
        Prefix of the added columns.
    return_baseline
        Also return the allele baseline table (:func:`allele_baseline`; ``None`` with ``baseline=None``).

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame or None)
        A copy of ``air`` with ``{key}_aligned_nt``, ``{key}_mismatches``, ``{key}_per100`` (mismatches per 100
        aligned nucleotides), ``{key}_baseline``, ``{key}_baseline_source`` (``"allele"``, ``"gene"``, ``"pooled"``,
        ``"zero"``, ``"none"`` = no baseline, hence not evaluable, or ``"off"`` with ``baseline=None``),
        ``{key}_excess``, ``{key}_evaluable`` and ``{key}_mutated``; with ``return_baseline`` also the allele
        baseline table. ``out.attrs["spaclone"]["shm"]`` holds the parameters and counts only (scalars), so the table
        concatenates and merges under every pandas version; the baseline table is returned, not stored in ``attrs``.

    Notes
    -----
    The baseline is a property of the data set: compute it on all records of a section (or of an animal) at once,
    not per region. For trees and per-site analyses use the IMGT-gapped alignments of IgBLAST + Change-O and a
    genotyped germline (TIgGER) instead of the allele baseline.

    MiXCR tables (AIR-SPACE): read the alignments with ``read_airspace(..., alignments=True)`` and call
    ``shm(air, mixcr_alignment="v_alignment_mixcr", min_reads=1, min_aligned_nt=100, baseline_fallback="zero")``:
    each molecule is one consensus read (no read floor), the reads cover the 3' part of V (median ~150 nt), and few
    V genes have enough unswitched molecules for an allele baseline.

    Examples
    --------
    >>> air = shm(isotype(air))  # doctest: +SKIP
    >>> air.groupby("isotype").shm_mutated.mean()  # doctest: +SKIP
    >>> air, baseline = shm(isotype(air), return_baseline=True)  # doctest: +SKIP
    >>> baseline.source.value_counts()  # doctest: +SKIP
    """
    if isinstance(region, str):
        if region not in ("V", "V_J"):
            raise ValueError(f"region must be 'V', 'V_J' or (start, end), got {region!r}")
        start = end = None
    else:
        try:
            start, end = (int(v) for v in region)
        except (TypeError, ValueError) as err:
            raise ValueError(f"region must be 'V', 'V_J' or (start, end), got {region!r}") from err
    if baseline not in ("allele", None):
        raise ValueError(f"baseline must be 'allele' or None, got {baseline!r}")
    trim = _per_record(trim_5prime_nt, len(air), air, "trim_5prime_nt")
    if mixcr_alignment is not None:
        if not isinstance(region, str):
            raise ValueError("region=(start, end) needs AIRR alignment strings, not MiXCR alignments")
        need = [mixcr_alignment] + ([mixcr_j_alignment] if region == "V_J" else [])
        if region == "V_J" and mixcr_j_alignment is None:
            raise ValueError("region='V_J' with mixcr_alignment needs mixcr_j_alignment")
        missing = [c for c in need if c not in air.columns]
        if missing:
            raise KeyError(f"shm: column(s) {missing} not in the table")
        from spaclone.io.mixcr import parse_mixcr_alignment

        pv = parse_mixcr_alignment(air[mixcr_alignment], trim_5prime_nt=trim, skip_ambiguous=skip_ambiguous)
        al = pv["aligned_nt"].fillna(0).to_numpy(dtype=np.int64)
        mm = pv["substitutions"].fillna(0).to_numpy(dtype=np.int64)
        if region == "V_J":
            pj = parse_mixcr_alignment(air[mixcr_j_alignment], skip_ambiguous=skip_ambiguous)
            al = al + pj["aligned_nt"].fillna(0).to_numpy(dtype=np.int64)
            mm = mm + pj["substitutions"].fillna(0).to_numpy(dtype=np.int64)
    else:
        need = [sequence_alignment, germline_alignment] + (
            [j_sequence_alignment, j_germline_alignment] if region == "V_J" else []
        )
        missing = [c for c in need if c not in air.columns]
        if missing:
            raise KeyError(f"shm: column(s) {missing} not in the table")
        al, mm = count_mismatches(
            air[sequence_alignment],
            air[germline_alignment],
            start=start,
            end=end,
            trim_5prime_nt=trim,
            skip_ambiguous=skip_ambiguous,
        )
        if region == "V_J":
            al_j, mm_j = count_mismatches(
                air[j_sequence_alignment], air[j_germline_alignment], skip_ambiguous=skip_ambiguous
            )
            al, mm = al + al_j, mm + mm_j
    reads = _reads(air, reads_key, min_reads)
    passes = (reads >= min_reads) & (al >= min_aligned_nt)
    if baseline == "allele":
        tab = allele_baseline(
            air,
            mm,
            aligned_nt=al,
            baseline_isotypes=baseline_isotypes,
            min_baseline_n=min_baseline_n,
            min_reads=min_reads,
            min_aligned_nt=min_aligned_nt,
            fallback=baseline_fallback,
            v_key=v_key,
            reads_key=reads_key,
            isotype_key=isotype_key,
            c_key=c_key,
        )
        base = air[v_key].map(tab.baseline).to_numpy(dtype=float)
        source = air[v_key].map(tab.source).fillna("none").to_numpy(dtype=object)
    else:
        tab = None
        base = np.zeros(len(air))
        source = np.full(len(air), "off", dtype=object)
    excess = mm - base
    evaluable = passes & np.isfinite(base)
    out = air.copy()
    k = key_added
    out[f"{k}_aligned_nt"] = al
    out[f"{k}_mismatches"] = mm
    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"{k}_per100"] = np.where(al > 0, 100.0 * mm / np.maximum(al, 1), np.nan)
    out[f"{k}_baseline"] = base if baseline == "allele" else np.nan
    out[f"{k}_baseline_source"] = pd.array(source, dtype="string")
    out[f"{k}_excess"] = excess
    out[f"{k}_evaluable"] = evaluable
    out[f"{k}_mutated"] = evaluable & (excess >= min_excess)
    params = {
        "sequence_alignment": sequence_alignment,
        "germline_alignment": germline_alignment,
        "region": region if isinstance(region, str) else (start, end),
        "baseline": baseline,
        "baseline_isotypes": tuple(baseline_isotypes),
        "min_baseline_n": min_baseline_n,
        "baseline_fallback": baseline_fallback if baseline == "allele" else None,
        "mixcr_alignment": mixcr_alignment,
        "min_reads": min_reads,
        "min_aligned_nt": min_aligned_nt,
        "min_excess": min_excess,
        "skip_ambiguous": skip_ambiguous,
        "trimmed_records": int((trim > 0).sum()),
        "n_alleles_with_baseline": int(tab["baseline"].notna().sum()) if tab is not None else 0,
        "n_evaluable": int(evaluable.sum()),
    }
    # attrs keep scalars only: pandas >= 3 compares attrs in concat / merge, and a DataFrame there cannot be compared
    out.attrs["spaclone"] = {**dict(air.attrs.get("spaclone", {})), "shm": params}
    return (out, tab) if return_baseline else out


def call_shm(
    air: pd.DataFrame,
    *,
    by: str = "locus",
    stat: Literal["median", "mean", "max"] = "median",
    min_excess: float | None = None,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS_ID,
    key: str = "shm",
) -> pd.DataFrame:
    """Mutation call of every locus, clone or other group from its evaluable records.

    Parameters
    ----------
    air
        Output of :func:`shm` with the grouping column.
    by
        ``"locus"`` (``locus_key``), ``"clone"`` (``clone_key``) or any column name.
    stat
        Summary of the evaluable records' excess (T1: the median).
    min_excess
        Excess at which a group is mutated; default: the value used by :func:`shm` (3 in T1).
    clone_key, locus_key, key
        Column names and the prefix given to :func:`shm`.

    Returns
    -------
    pandas.DataFrame
        One row per group (sorted): ``n_records``, ``n_evaluable``, ``excess`` (summary over evaluable records),
        ``per100`` (same summary of mismatches per 100 nt), ``evaluable`` (at least one evaluable record) and
        ``mutated`` (boolean; missing when not evaluable).
    """
    col = {"locus": locus_key, "clone": clone_key}.get(by, by)
    need = [col, f"{key}_excess", f"{key}_evaluable", f"{key}_per100"]
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(f"call_shm: column(s) {missing} not in the table (run bcr.shm first)")
    if stat not in ("median", "mean", "max"):
        raise ValueError(f"stat must be 'median', 'mean' or 'max', got {stat!r}")
    if min_excess is None:
        min_excess = air.attrs.get("spaclone", {}).get("shm", {}).get("min_excess", 3)
    ev = air[f"{key}_evaluable"].fillna(False).to_numpy(dtype=bool)
    groups = air.groupby(col, sort=True)
    out = pd.DataFrame({"n_records": groups.size().astype(np.int64)})
    sub = air.loc[ev, [col, f"{key}_excess", f"{key}_per100"]]
    agg = sub.groupby(col, sort=True).agg(["size", stat])
    out["n_evaluable"] = agg[(f"{key}_excess", "size")].reindex(out.index).fillna(0).astype(np.int64)
    out["excess"] = agg[(f"{key}_excess", stat)].reindex(out.index).astype(float)
    out["per100"] = agg[(f"{key}_per100", stat)].reindex(out.index).astype(float)
    out["evaluable"] = out.n_evaluable > 0
    mut = np.where(out.evaluable, out.excess >= min_excess, None)
    out["mutated"] = pd.array(mut, dtype="boolean")
    out.index.name = col
    return out


# ------------------------------------------------------------------------------------------- maps (via spaclone.tl)


def _tl(name: str):
    """A function of :mod:`spaclone.tl` (resolved at call time, so the maps module can be added later)."""
    from spaclone import tl

    fn = getattr(tl, name, None)
    if fn is None:  # pragma: no cover - depends on the state of spaclone.tl
        raise ImportError(f"spaclone.tl.{name} is not available in this installation (tl.maps not installed yet)")
    return fn


def _map_units(units: pd.DataFrame, mark: str, match_by: str | None, blocks: str | None = None) -> pd.DataFrame:
    if mark not in units.columns:
        raise KeyError(f"column {mark!r} not in the unit table")
    if blocks is not None and blocks not in units.columns:
        raise KeyError(
            f"blocks column {blocks!r} not in the unit table: clone-level shuffles need the clone of every unit "
            "(pass blocks=<clone column>, or blocks=None for unit-level shuffles, valid only when the units of a "
            "clone carry independent marks)"
        )
    keep = units[mark].notna().to_numpy()
    if (~keep).any():
        logger.info("%d of %d units without a %r call are left out of the map", int((~keep).sum()), len(units), mark)
    sub = units.loc[keep].copy()
    if blocks is not None and sub[blocks].isna().any():
        raise ValueError(f"blocks column {blocks!r} has missing values")
    sub[mark] = sub[mark].astype(bool)
    if match_by is not None:
        if match_by not in sub.columns:
            raise KeyError(f"match_by column {match_by!r} not in the unit table")
        sub[match_by] = sub[match_by].astype("string").fillna("unknown").astype(object)
    return sub


def _to_loci(
    sub: pd.DataFrame,
    *,
    mark: str,
    match_by: str | None,
    blocks: str,
    radius_um: float | None,
    x_key: str,
    y_key: str,
) -> tuple[pd.DataFrame, int]:
    """Merge units of one clone within ``radius_um`` (single linkage) into loci; (table, units merged away).

    A table of loci (:func:`spaclone.pp.merge_loci`) has no two units of a clone that close, so it is returned as it
    is. Otherwise each locus takes the centroid, the mark of at least half of its units, the most frequent
    ``match_by`` class (ties: the first in sorted order) and the other columns of its first unit.
    """
    if radius_um is None or not len(sub) or x_key not in sub.columns or y_key not in sub.columns:
        return sub, 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    xy = sub[[x_key, y_key]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        return sub, 0
    blk = pd.factorize(sub[blocks])[0]
    pairs = cKDTree(xy).query_pairs(float(radius_um), output_type="ndarray")
    if len(pairs):
        pairs = pairs[blk[pairs[:, 0]] == blk[pairs[:, 1]]]
    if not len(pairs):
        return sub, 0
    n = len(sub)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(g, directed=False)
    comp = pd.factorize(comp)[0]  # numbered by first appearance: the order of the units is kept
    cnt = np.bincount(comp).astype(float)
    first = np.full(n_comp, -1, dtype=np.int64)
    first[comp[::-1]] = np.arange(n)[::-1]
    out = sub.iloc[first].copy()
    out[x_key] = np.bincount(comp, weights=xy[:, 0]) / cnt
    out[y_key] = np.bincount(comp, weights=xy[:, 1]) / cnt
    share = np.bincount(comp, weights=sub[mark].to_numpy(dtype=float)) / cnt
    out[mark] = share >= 0.5
    if match_by is not None:
        d = pd.DataFrame({"c": comp, "v": sub[match_by].to_numpy(dtype=object)})
        top = (
            d.groupby(["c", "v"])
            .size()
            .rename("k")
            .reset_index()
            .sort_values(["c", "k", "v"], ascending=[True, False, True])
        )
        out[match_by] = top.drop_duplicates("c").set_index("c").v.reindex(range(n_comp)).to_numpy(dtype=object)
    out["n_merged_units"] = cnt.astype(np.int64)
    logger.warning(
        "%d units lay within %g um of another unit of their clone (records or spots of one cell): merged into %d loci "
        "(mark = at least half of the units); build loci with pp.merge_loci and bcr.call_shm / call_isotype "
        "(by='locus') for the archive's calls",
        n,
        radius_um,
        n_comp,
    )
    return out.reset_index(drop=True), n - n_comp


def shm_map(
    units: pd.DataFrame,
    *,
    mark: str = "mutated",
    match_by: str | None = "isotype",
    sigma_um: float = 50.0,
    **kwargs: Any,
):
    """Smoothed share of mutated units (loci) with an isotype-matched expectation (archive T7).

    Parameters
    ----------
    units
        One row per unit (normally loci, see :func:`call_shm`) with coordinates, the mutated call and the isotype.
        Units without a call (not evaluable) are left out.
    mark
        Boolean column (missing = not evaluable).
    match_by
        Covariate of the expectation: each unit's expected probability is the mutated share of its isotype class, so
        a pixel stands out only if it holds more mutated units than its local isotype mix predicts (switched B cells
        are mutated far more often than IgM / IgD ones). ``None``: the global share. Missing classes become
        ``"unknown"``.
    sigma_um
        Gaussian kernel standard deviation in µm.
    **kwargs
        Passed to :func:`spaclone.tl.share_map` (``grid``, ``tissue``, ``pixel_um``, ``min_neff``, ``x_key``, ...).

    Returns
    -------
    spaclone.tl.ShareMap
    """
    sub = _map_units(units, mark, match_by)
    return _tl("share_map")(sub, mark=mark, match_by=match_by, sigma_um=sigma_um, **kwargs)


def shm_hotspots(
    units: pd.DataFrame,
    *,
    mark: str = "mutated",
    match_by: str | None = "isotype",
    blocks: str | None = S.CLONE,
    loci_radius_um: float | None = 15.0,
    sigma_um: float = 50.0,
    n_perm: int = 1000,
    rng: RngLike = None,
    **kwargs: Any,
):
    """SHM hotspots with family-wise error control over the map: clone-level shuffles, isotype-matched expectation.

    Parameters
    ----------
    units
        One row per unit, normally loci (one inferred cell each: :func:`spaclone.pp.merge_loci`, then
        :func:`call_shm` and :func:`spaclone.bcr.call_isotype` with ``by="locus"``), with coordinates, the mutated
        call, the isotype and the clone; units without a call are left out.
    mark, match_by, sigma_um
        As in :func:`shm_map`.
    blocks
        Clone column (default ``"clone_id"``). Mutation is a lineage property: the loci of one clone share most
        mutations and sit together, so the null relabels whole clones (see Notes). ``None``: loci are shuffled one by
        one, the 2026-10 archive's design (T7), valid only when every clone has a single locus.
    loci_radius_um
        With ``blocks``: units of one clone closer than this (records or spots of one cell) are merged into one locus
        before mapping (mark of at least half of its units, most frequent isotype); a loci table from
        :func:`spaclone.pp.merge_loci` is left as it is. Clone-level shuffles are valid on cell-scale units only:
        re-dealing marks among the records of one cell narrows the null (simulated family-wise error 10.5% on records
        against 4.5% on loci). ``None``: no merging.
    n_perm
        Shuffled maps.
    rng
        Seed or generator.
    **kwargs
        Passed to :func:`spaclone.tl.hotspots` (``tissue``, ``grid``, ``min_neff``, ``correction``,
        ``threshold_rule``, ``neff_kernel``, ``block_design``, ``standardise``, ``min_blocks``, ``null_marks``,
        ``min_units``, ``x_key``, ...). The archive's T7 maps: ``blocks=None, neff_kernel="legacy",
        threshold_rule="quantile"``. ``min_units=1`` drops hotspots without an evaluable locus inside
        (:func:`spaclone.bcr.hotspot_workflow` does by default).

    Returns
    -------
    spaclone.tl.HotspotResult

    Notes
    -----
    Null model (default, ``blocks`` = clone, ``block_design="matched"``): positions and isotypes stay in place;
    clones with the same number of evaluable loci of every isotype trade their mutated calls, and inside a clone the
    calls of an isotype are re-dealt among its loci of that isotype. Each isotype keeps its number of mutated loci, so
    the isotype-matched expectation of every pixel is unchanged, and a clone keeps its mutated count unless it trades
    with a clone of the same size and isotype mix: the null allows mutation to depend on isotype and on clone size
    (expanded clones are mutated more often) and asks whether some *place* holds more mutated loci than that. The
    map is studentised by the shuffled maps' pixel means and standard deviations (``standardise="permutation"``).
    In simulations of clustered clones with clone-level mutation the family-wise error of this design stayed at or
    below α, while loci shuffled one by one within isotype gave 99%.

    Null model (``blocks=None``): the mutated labels are permuted among the evaluable loci of the same isotype class
    (``within`` = ``match_by``), positions fixed; with ``match_by=None`` among all evaluable loci.

    References
    ----------
    Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Nichols & Holmes 2002, doi:10.1002/hbm.1058; Westfall &
    Young 1993, Resampling-Based Multiple Testing, Wiley; Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885.
    """
    sub = _map_units(units, mark, match_by, blocks)
    merged = 0
    if blocks is not None and kwargs.get("null_marks") is None:
        x_key, y_key = kwargs.get("x_key", S.X), kwargs.get("y_key", S.Y)
        sub, merged = _to_loci(
            sub, mark=mark, match_by=match_by, blocks=blocks, radius_um=loci_radius_um, x_key=x_key, y_key=y_key
        )
    res = _tl("hotspots")(
        sub, mark=mark, match_by=match_by, blocks=blocks, sigma_um=sigma_um, n_perm=n_perm, rng=rng, **kwargs
    )
    res.params["units_merged_into_loci"] = merged
    return res


# ------------------------------------------------------------------------------------------------------- IgBLAST


def _find_igblast(executable: str | os.PathLike | None, igblast_dir: str | os.PathLike | None) -> Path | None:
    if executable is not None:
        p = Path(executable)
        return p if p.exists() else None
    if igblast_dir is not None:
        p = Path(igblast_dir) / "bin" / "igblastn"
        return p if p.exists() else None
    w = shutil.which("igblastn")
    return Path(w) if w else None


def run_igblast(
    fasta: str | os.PathLike | pd.Series | dict[str, str],
    *,
    organism: str = "mouse",
    igblast_dir: str | os.PathLike | None = None,
    germline_db: str | os.PathLike | dict[str, str | os.PathLike] | None = None,
    auxiliary_data: str | os.PathLike | None = None,
    executable: str | os.PathLike | None = None,
    ig_seqtype: Literal["Ig", "TCR"] = "Ig",
    domain_system: Literal["imgt", "kabat"] = "imgt",
    num_threads: int = 1,
    out: str | os.PathLike | None = None,
    extra_args: Sequence[str] = (),
) -> pd.DataFrame:
    """Align sequences with an installed IgBLAST and return its AIRR rearrangement table.

    Parameters
    ----------
    fasta
        A FASTA file, or sequences as a Series / dict (id -> nucleotide sequence).
    organism
        IgBLAST ``-organism`` (``"mouse"``, ``"human"``, ...).
    igblast_dir
        IgBLAST installation (``bin/igblastn``, ``internal_data/``, ``optional_file/``); used as ``IGDATA``.
    germline_db
        Prefix of the V / D / J BLAST databases: a directory holding ``{organism}_V`` / ``_D`` / ``_J`` (or
        ``imgt_{organism}_ig_v`` / ``_d`` / ``_j``), or a dict ``{"V": prefix, "D": prefix, "J": prefix}``.
        Default: ``{igblast_dir}/database`` and then ``{igblast_dir}/../database``.
    auxiliary_data
        J-gene auxiliary file (default ``{igblast_dir}/optional_file/{organism}_gl.aux`` when present).
    executable
        Path of ``igblastn`` (default: ``{igblast_dir}/bin/igblastn`` or the one on ``PATH``).
    ig_seqtype, domain_system
        IgBLAST ``-ig_seqtype`` and ``-domain_system``.
    num_threads
        IgBLAST threads.
    out
        Where to keep the AIRR TSV (default: a temporary file that is removed).
    extra_args
        Further IgBLAST arguments.

    Returns
    -------
    pandas.DataFrame
        IgBLAST's AIRR table (``-outfmt 19``): ``sequence_id``, ``v_call``, ``sequence_alignment``,
        ``germline_alignment``, ``v_sequence_alignment``, ``v_germline_alignment``, ``j_*``, ``fwr*`` / ``cdr*``,
        ... (not IMGT-gapped; run Change-O ``MakeDb.py`` on IgBLAST's ``-outfmt 7`` output for IMGT gaps).

    Raises
    ------
    FileNotFoundError
        When ``igblastn`` or the germline databases cannot be found (IgBLAST is an optional external tool).

    References
    ----------
    Ye et al. 2013, doi:10.1093/nar/gkt382.
    """
    exe = _find_igblast(executable, igblast_dir)
    if exe is None:
        raise FileNotFoundError(
            "igblastn not found: install IgBLAST (https://ncbi.github.io/igblast/) and pass igblast_dir= or "
            "executable=, or put igblastn on PATH"
        )
    home = Path(igblast_dir) if igblast_dir is not None else exe.parent.parent
    dbs = _germline_dbs(germline_db, home, organism)
    if auxiliary_data is None:
        aux = home / "optional_file" / f"{organism}_gl.aux"
        auxiliary_data = aux if aux.exists() else None
    with tempfile.TemporaryDirectory(prefix="spaclone_igblast_") as tmp:
        if isinstance(fasta, str | os.PathLike) and Path(fasta).exists():
            fa = Path(fasta)
        else:
            items = fasta.items() if isinstance(fasta, dict | pd.Series) else None
            if items is None:
                raise FileNotFoundError(f"FASTA file {fasta!r} not found")
            fa = Path(tmp) / "query.fasta"
            with open(fa, "w") as fh:
                for sid, seq in items:
                    fh.write(f">{sid}\n{seq}\n")
        res = Path(out) if out is not None else Path(tmp) / "igblast_airr.tsv"
        cmd = [
            str(exe),
            "-germline_db_V",
            str(dbs["V"]),
            "-germline_db_D",
            str(dbs["D"]),
            "-germline_db_J",
            str(dbs["J"]),
            "-organism",
            organism,
            "-ig_seqtype",
            ig_seqtype,
            "-domain_system",
            domain_system,
            "-outfmt",
            "19",
            "-num_threads",
            str(int(num_threads)),
            "-query",
            str(fa),
            "-out",
            str(res),
            *([] if auxiliary_data is None else ["-auxiliary_data", str(auxiliary_data)]),
            *extra_args,
        ]
        env = dict(os.environ, IGDATA=str(home))
        env.pop("PYTHONPATH", None)
        logger.info("running %s", " ".join(cmd))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"igblastn failed ({r.returncode}): {r.stderr[-2000:]}")
        return pd.read_csv(res, sep="\t", low_memory=False)


def _germline_dbs(germline_db: Any, home: Path, organism: str) -> dict[str, Path]:
    if isinstance(germline_db, dict):
        dbs = {k.upper(): Path(v) for k, v in germline_db.items()}
        if set(dbs) != {"V", "D", "J"}:
            raise ValueError("germline_db as a dict needs the keys 'V', 'D' and 'J'")
        return dbs
    dirs = [Path(germline_db)] if germline_db is not None else [home / "database", home.parent / "database"]
    for d in dirs:
        for names in (
            {s: f"{organism}_{s}" for s in "VDJ"},
            {s: f"imgt_{organism}_ig_{s.lower()}" for s in "VDJ"},
        ):
            cand = {s: d / n for s, n in names.items()}
            if all(any(p.parent.glob(p.name + ".n*")) for p in cand.values()):
                return cand
    raise FileNotFoundError(f"no {organism} V / D / J germline databases found in {[str(d) for d in dirs]}")
