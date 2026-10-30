"""IgBLAST, Change-O, SHazaM and dowser on spatial receptor records: export, run, import back onto coordinates.

The Immcantation framework (Gupta et al. 2015; Vander Heiden et al. 2014) is a set of command-line programs and R
packages that pass AIRR tables through files. This module runs them on a spatial receptor table with the documented
commands and brings their output back to the records, so that clones, germlines, mutation counts and lineage trees
keep their tissue positions:

======================  ======================================================  ========================================
step                    tool and command                                        back on the records
======================  ======================================================  ========================================
:func:`export_airr`     FASTA of the record sequences + table of the records'   ids ``r<row>`` link every output row to
                        spatial columns                                         its record
:func:`igblast`         ``igblastn -outfmt "7 std qseq sseq btop"`` (as         -
                        ``AssignGenes.py igblast``)
:func:`makedb`          ``MakeDb.py igblast --extended`` (IMGT-gapped AIRR)     V/D/J calls, IMGT-gapped alignments
:func:`define_clones`   ``DefineClones.py --model ham --norm len --dist d``     Change-O clone per record
:func:`create_germlines` ``CreateGermlines.py -g dmask --cloned``               ``germline_alignment_d_mask``
:func:`shazam_threshold` R ``shazam::distToNearest`` + ``findThreshold``        the clonal distance threshold
:func:`observed_mutations` R ``shazam::observedMutations``                      ``mu_count`` / ``mu_freq`` per record
:func:`dowser_trees`    R ``dowser::formatClones`` + ``getTrees``               :class:`spaclone.bcr.LineageTrees` with
                                                                                positioned tips (``bcr.tip_table``)
:func:`run`             the chain above                                         :class:`ImmcantationResult`
======================  ======================================================  ========================================

Tools are found by :func:`find_tools`: explicit arguments, then the environment variables ``SPACLONE_IGBLAST``
(``igblastn`` or an IgBLAST directory), ``SPACLONE_IGDATA`` (IgBLAST data: ``internal_data/``, ``optional_file/``,
``database/``), ``SPACLONE_GERMLINES`` (folder of IMGT-gapped germline FASTA files), ``SPACLONE_CHANGEO`` (folder of
``MakeDb.py`` etc.) and ``SPACLONE_RSCRIPT``, then ``PATH``. Every function takes ``dry_run=True`` to return the
command instead of running it, and every run is logged; :attr:`ImmcantationResult.commands` lists what was run, for
a methods section. Install the tools with ``conda install -c conda-forge -c bioconda igblast changeo r-shazam
r-dowser r-tigger r-alakazam r-scoper`` (or ``pip install changeo`` for the Python programs) and the IMGT references
with Immcantation's ``fetch_imgtdb.sh`` + ``imgt2igblast.sh``.

References
----------
Ye et al. 2013, doi:10.1093/nar/gkt382 (IgBLAST); Gupta et al. 2015, doi:10.1093/bioinformatics/btv359 (Change-O);
Yaari et al. 2013, doi:10.3389/fimmu.2013.00358 (SHazaM targeting models); Nouri & Kleinstein 2018,
doi:10.1093/bioinformatics/bty235 (clonal thresholds); Gadala-Maria et al. 2015, doi:10.1073/pnas.1417683112
(TIgGER); Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885 (dowser); Vander Heiden et al. 2018,
doi:10.4049/jimmunol.1800669 (AIRR Rearrangement schema).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = [
    "ImmcantationResult",
    "Tools",
    "create_germlines",
    "define_clones",
    "dowser_trees",
    "export_airr",
    "find_tools",
    "igblast",
    "makedb",
    "observed_mutations",
    "run",
    "shazam_threshold",
    "to_records",
]

logger = logging.getLogger("spaclone")

Loci = Literal["ig", "tr"]
_ID = "sequence_id"
_ROWCOL = "spaclone_row"


# ------------------------------------------------------------------------------------------------------ tools


@dataclass(frozen=True)
class Tools:
    """Locations of the external programs and references (see :func:`find_tools`).

    Attributes
    ----------
    igblastn
        IgBLAST executable.
    igdata
        IgBLAST data directory (``IGDATA``: ``internal_data/``, ``optional_file/``).
    germline_db
        BLAST database prefixes ``{"V": ..., "D": ..., "J": ...}``.
    auxiliary_data
        IgBLAST J-gene auxiliary file.
    references
        IMGT-gapped germline FASTA files (``MakeDb.py -r``, ``CreateGermlines.py -r``).
    changeo
        ``{"MakeDb.py": path, "DefineClones.py": path, "CreateGermlines.py": path}``.
    rscript
        ``Rscript`` (R with shazam / dowser / alakazam).
    organism, loci
        What the references are for.
    """

    igblastn: Path | None = None
    igdata: Path | None = None
    germline_db: dict[str, Path] | None = None
    auxiliary_data: Path | None = None
    references: tuple[Path, ...] = ()
    changeo: dict[str, Path] = field(default_factory=dict)
    rscript: Path | None = None
    organism: str = "mouse"
    loci: str = "ig"

    def require(self, step: str) -> None:
        """Raise a FileNotFoundError naming what ``step`` needs and does not have."""
        needs = {
            "igblast": [("igblastn", self.igblastn), ("germline databases", self.germline_db)],
            "makedb": [("MakeDb.py", self.changeo.get("MakeDb.py")), ("IMGT-gapped references", self.references)],
            "define_clones": [("DefineClones.py", self.changeo.get("DefineClones.py"))],
            "create_germlines": [
                ("CreateGermlines.py", self.changeo.get("CreateGermlines.py")),
                ("IMGT-gapped references", self.references),
            ],
            "r": [("Rscript", self.rscript)],
        }[step]
        missing = [name for name, v in needs if not v]
        if missing:
            raise FileNotFoundError(
                f"{step} needs {', '.join(missing)}; pass them to spaclone.ext.immcantation.find_tools(...) or set "
                "SPACLONE_IGBLAST / SPACLONE_IGDATA / SPACLONE_GERMLINES / SPACLONE_CHANGEO / SPACLONE_RSCRIPT "
                "(install: conda install -c conda-forge -c bioconda igblast changeo r-shazam r-dowser)"
            )


def _exe(name: str, given: str | os.PathLike | None, env: str) -> Path | None:
    from spaclone._versions import find_tool

    for cand in (given, os.environ.get(env) or None):
        if cand is None:
            continue
        p = Path(cand).expanduser()
        for q in (p, p / name, p / "bin" / name):
            if q.is_file() and os.access(q, os.X_OK):
                return q
    return find_tool(name)


def _blast_db(base: Path, organism: str, loci: str) -> dict[str, Path] | None:
    for d in (base / "database", base.parent / "database", base):
        for names in (
            {s: f"imgt_{organism}_{loci}_{s.lower()}" for s in "VDJ"},
            {s: f"{organism}_{s}" if loci == "ig" else f"{organism}_TR_{s}" for s in "VDJ"},
        ):
            cand = {s: d / n for s, n in names.items()}
            if all(any(p.parent.glob(p.name + ".n*")) for p in cand.values()):
                return cand
    return None


def _references(folder: Path, organism: str, loci: str) -> tuple[Path, ...]:
    pat = ("IGH", "IGK", "IGL") if loci == "ig" else ("TRA", "TRB", "TRD", "TRG")
    files = []
    for f in sorted(folder.glob("*.fasta")):
        stem = f.stem.upper()
        if any(stem.endswith(p + s) for p in pat for s in "VDJ") and (
            organism in f.stem or not f.stem.startswith("imgt")
        ):
            files.append(f)
    return tuple(files)


def _igdata_guess(exe: Path | None) -> Path:
    """IgBLAST's data directory next to an installed ``igblastn`` (bioconda: ``<prefix>/share/igblast``)."""
    if exe is None:
        return Path(".")
    root = exe.resolve().parent.parent
    for cand in (root / "share" / "igblast", root):
        if (cand / "internal_data").is_dir():
            return cand
    return root


def find_tools(
    *,
    igblast: str | os.PathLike | None = None,
    igdata: str | os.PathLike | None = None,
    germline_db: str | os.PathLike | dict[str, str | os.PathLike] | None = None,
    references: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    changeo: str | os.PathLike | None = None,
    rscript: str | os.PathLike | None = None,
    organism: str = "mouse",
    loci: Loci = "ig",
) -> Tools:
    """Locate IgBLAST, its databases, the IMGT-gapped references, Change-O and R.

    Parameters
    ----------
    igblast
        ``igblastn`` or an IgBLAST installation directory (default ``SPACLONE_IGBLAST``, then ``PATH``).
    igdata
        IgBLAST data directory with ``internal_data/`` and ``optional_file/`` (default ``SPACLONE_IGDATA``, then
        IgBLAST's own ``IGDATA``, then the installation: ``<prefix>/share/igblast`` for bioconda's IgBLAST).
    germline_db
        Folder of the BLAST databases (``imgt_{organism}_{loci}_{v,d,j}`` as built by Immcantation's
        ``imgt2igblast.sh``, or NCBI-style ``{organism}_V`` / ``_D`` / ``_J``) or a dict of prefixes (default
        ``{igdata}/database``).
    references
        Folder of IMGT-gapped germline FASTA files (``imgt_{organism}_IGHV.fasta``, ...) or the files (default
        ``SPACLONE_GERMLINES``).
    changeo
        Folder holding ``MakeDb.py``, ``DefineClones.py`` and ``CreateGermlines.py`` (default ``SPACLONE_CHANGEO``,
        then ``PATH``).
    rscript
        ``Rscript`` (default ``SPACLONE_RSCRIPT``, then ``PATH``).
    organism, loci
        ``"mouse"`` / ``"human"`` / ...; ``"ig"`` or ``"tr"``.

    Returns
    -------
    Tools
        Missing pieces are None / empty; each step checks what it needs (:meth:`Tools.require`).
    """
    exe = _exe("igblastn", igblast, "SPACLONE_IGBLAST")
    home = Path(igdata or os.environ.get("SPACLONE_IGDATA") or os.environ.get("IGDATA") or _igdata_guess(exe))
    if isinstance(germline_db, dict):
        db = {k.upper(): Path(v) for k, v in germline_db.items()}
    elif germline_db is not None:
        db = _blast_db(Path(germline_db), organism, loci)
    else:
        db = _blast_db(home, organism, loci) if exe else None
    aux = home / "optional_file" / f"{organism}_gl.aux"
    refs: tuple[Path, ...] = ()
    src = references if references is not None else os.environ.get("SPACLONE_GERMLINES")
    if src is not None:
        if isinstance(src, str | os.PathLike) and Path(src).is_dir():
            refs = _references(Path(src), organism, loci)
        else:
            items = [src] if isinstance(src, str | os.PathLike) else list(src)
            refs = tuple(Path(p) for p in items)
    cdir = changeo
    tools = {n: _exe(n, cdir, "SPACLONE_CHANGEO") for n in ("MakeDb.py", "DefineClones.py", "CreateGermlines.py")}
    return Tools(
        igblastn=exe,
        igdata=home if exe else None,
        germline_db=db,
        auxiliary_data=aux if aux.exists() else None,
        references=refs,
        changeo={k: v for k, v in tools.items() if v is not None},
        rscript=_exe("Rscript", rscript, "SPACLONE_RSCRIPT"),
        organism=organism,
        loci=loci,
    )


def _env(tools: Tools | None = None) -> dict[str, str]:
    from spaclone._versions import external_env

    env = external_env()  # external programs use their own Python environment
    if tools is not None and tools.igdata is not None:
        env["IGDATA"] = str(tools.igdata)
    return env


def _run(cmd: list[str], *, env: dict[str, str], log: list[str] | None, what: str, timeout: float | None = None) -> str:
    text = shlex.join(cmd)
    logger.info("%s: %s", what, text)
    if log is not None:
        log.append(text)
    r = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed ({r.returncode}): {text}\n{(r.stderr or r.stdout)[-3000:]}")
    return r.stdout


# ----------------------------------------------------------------------------------------------- export / import


def export_airr(
    air: pd.DataFrame,
    out_dir: str | os.PathLike,
    *,
    name: str = "spaclone",
    sequence_key: str = "sequence",
    keep: Sequence[str] | None = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    library_key: str = S.LIBRARY,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
) -> dict[str, Path]:
    """Write the record sequences (FASTA) and the records' spatial columns for an Immcantation run.

    Parameters
    ----------
    air
        Receptor table with full contig sequences (``sequence_key``, nucleotides).
    out_dir
        Output folder (created).
    name
        File name stem.
    sequence_key
        Column of the nucleotide sequence given to IgBLAST.
    keep
        Record columns written to the spatial table (default: spot, coordinates, library, clone, locus and the
        AIRR columns ``c_call``, ``consensus_count``, ``umi_count``, ``duplicate_count`` when present).
    spot_key, x_key, y_key, library_key, clone_key, locus_key
        Column names.

    Returns
    -------
    dict
        ``{"fasta": ..., "records": ...}``: the FASTA (ids ``r<row>``, the row position in ``air``; records without a
        sequence are left out) and the TSV of ``sequence_id``, ``spaclone_row`` and the kept columns.
    """
    if sequence_key not in air.columns:
        raise KeyError(f"export_airr: column {sequence_key!r} (nucleotide sequence) not in the table")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab = air.reset_index(drop=True)
    seq = tab[sequence_key].astype("string").str.upper().str.replace(r"[^ACGTN]", "", regex=True)
    ok = seq.notna() & (seq.str.len() > 0)
    rows = np.flatnonzero(ok.to_numpy(dtype=bool))
    ids = [f"r{i}" for i in rows]
    fasta = out / f"{name}.fasta"
    with open(fasta, "w") as fh:
        for sid, s in zip(ids, seq.to_numpy(dtype=object)[rows], strict=True):
            fh.write(f">{sid}\n{s}\n")
    default = [spot_key, x_key, y_key, library_key, clone_key, locus_key, "c_call", S.READS, "umi_count"]
    cols = [c for c in (keep if keep is not None else default) if c in tab.columns]
    rec = tab.iloc[rows][cols].copy()
    rec.insert(0, _ROWCOL, rows)
    rec.insert(0, _ID, ids)
    path = out / f"{name}_records.tsv"
    rec.to_csv(path, sep="\t", index=False)
    return {"fasta": fasta, "records": path}


def _row_of(ids: pd.Series) -> np.ndarray:
    s = ids.astype(str)
    ok = s.str.match(r"^r\d+$")
    out = np.full(len(s), -1, dtype=np.int64)
    out[ok.to_numpy()] = s[ok].str.slice(1).astype(np.int64).to_numpy()
    return out


def to_records(
    db: pd.DataFrame,
    air: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    prefix: str = "imm_",
) -> pd.DataFrame:
    """Copy columns of an Immcantation table back onto the records they came from.

    Parameters
    ----------
    db
        Immcantation output (AIRR table) whose ``sequence_id`` are the ``r<row>`` ids of :func:`export_airr`.
    air
        The receptor table that was exported.
    columns
        ``db`` columns to copy (default: calls, alignments, germlines, clone and mutation columns present).
    prefix
        Prefix of the new columns (``imm_v_call``, ``imm_clone_id``, ...).

    Returns
    -------
    pandas.DataFrame
        A copy of ``air`` with the prefixed columns (missing for records without an Immcantation row).
    """
    default = [
        "v_call",
        "d_call",
        "j_call",
        "junction",
        "junction_aa",
        "productive",
        "sequence_alignment",
        "germline_alignment",
        "germline_alignment_d_mask",
        "v_identity",
        "clone_id",
        "mu_count",
        "mu_freq",
    ]
    cols = [c for c in (columns if columns is not None else default) if c in db.columns]
    rows = _row_of(db[_ID])
    ok = (rows >= 0) & (rows < len(air))
    out = air.copy()
    pos = np.full(len(air), -1, dtype=np.int64)
    pos[rows[ok]] = np.flatnonzero(ok)
    hit = pos >= 0
    for c in cols:
        vals = db[c].to_numpy(dtype=object)
        col = np.full(len(air), None, dtype=object)
        col[hit] = vals[pos[hit]]
        s = pd.Series(col, index=air.index)
        num = pd.to_numeric(s, errors="coerce")
        out[prefix + c] = num if s.notna().any() and num[s.notna()].notna().all() and c != "clone_id" else s
    return out


def _with_space(db: pd.DataFrame, records: pd.DataFrame | None) -> pd.DataFrame:
    if records is None or _ID not in db.columns:
        return db
    extra = [c for c in records.columns if c not in db.columns or c == _ID]
    return db.merge(records[extra], on=_ID, how="left")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False, dtype={_ID: str})


def _to_tsv(db: pd.DataFrame, path: Path) -> Path:
    """Write an AIRR table for the Immcantation programs: integer columns stay integers.

    pandas reads an integer column with missing values (``d_germline_start`` without a D call, ...) as floats, and
    Change-O cannot parse ``"4.0"`` as a position; such columns are written as integers with empty missing values.
    """
    out = db.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s.dtype):
            v = s.dropna().to_numpy()
            if len(v) and np.all(np.isfinite(v)) and np.all(v == np.round(v)) and np.all(np.abs(v) < 2**53):
                out[c] = s.astype("Int64")
    out.to_csv(path, sep="\t", index=False)
    return path


# ------------------------------------------------------------------------------------------------- programs


def igblast(
    fasta: str | os.PathLike,
    *,
    tools: Tools | None = None,
    out: str | os.PathLike | None = None,
    threads: int = 1,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    log: list[str] | None = None,
) -> Path | list[str]:
    """IgBLAST with Immcantation's options (``AssignGenes.py igblast``), output for :func:`makedb`.

    Parameters
    ----------
    fasta
        Query FASTA (e.g. from :func:`export_airr`).
    tools
        From :func:`find_tools` (default: found from the environment).
    out
        Output file (default ``<fasta stem>_igblast.fmt7`` next to the FASTA).
    threads
        IgBLAST threads.
    extra_args
        Further IgBLAST arguments.
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    pathlib.Path or list of str
        The ``-outfmt "7 std qseq sseq btop"`` result, or the command (``dry_run``).
    """
    tools = tools or find_tools()
    tools.require("igblast")
    fa = Path(fasta)
    res = Path(out) if out is not None else fa.with_name(fa.stem + "_igblast.fmt7")
    db = tools.germline_db or {}
    cmd = [
        str(tools.igblastn),
        "-germline_db_V",
        str(db["V"]),
        "-germline_db_D",
        str(db["D"]),
        "-germline_db_J",
        str(db["J"]),
        "-organism",
        tools.organism,
        "-ig_seqtype",
        "Ig" if tools.loci == "ig" else "TCR",
        "-domain_system",
        "imgt",
        "-outfmt",
        "7 std qseq sseq btop",
        "-num_threads",
        str(int(threads)),
        "-query",
        str(fa),
        "-out",
        str(res),
        *(["-auxiliary_data", str(tools.auxiliary_data)] if tools.auxiliary_data else []),
        *extra_args,
    ]
    if dry_run:
        return cmd
    _run(cmd, env=_env(tools), log=log, what="igblastn")
    return res


def makedb(
    fmt7: str | os.PathLike,
    fasta: str | os.PathLike,
    *,
    tools: Tools | None = None,
    records: str | os.PathLike | pd.DataFrame | None = None,
    out_dir: str | os.PathLike | None = None,
    name: str | None = None,
    extended: bool = True,
    partial: bool = False,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> pd.DataFrame | list[str]:
    """Change-O ``MakeDb.py igblast``: IMGT-gapped AIRR table from IgBLAST output.

    Parameters
    ----------
    fmt7, fasta
        IgBLAST output and its query FASTA.
    tools
        From :func:`find_tools`.
    records
        The records table of :func:`export_airr` (path or DataFrame); its spatial columns are joined on
        ``sequence_id``.
    out_dir, name
        Where MakeDb writes ``<name>_db-pass.tsv`` (default: next to ``fmt7``, its stem).
    extended
        ``--extended`` (identities, CDR / FWR regions, alignment positions).
    partial
        ``--partial`` (keep incomplete alignments).
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    pandas.DataFrame or list of str
        One row per sequence that passed (``productive``, ``v_call``, ``sequence_alignment`` with IMGT gaps,
        ``germline_alignment``, ``junction``, ...) with the records' spatial columns; or the command.
    """
    tools = tools or find_tools()
    tools.require("makedb")
    f7 = Path(fmt7)
    od = Path(out_dir) if out_dir is not None else f7.parent
    nm = name or f7.stem
    cmd = [
        str(tools.changeo["MakeDb.py"]),
        "igblast",
        "-i",
        str(f7),
        "-s",
        str(fasta),
        "-r",
        *[str(r) for r in tools.references],
        "--asis-id",
        "--failed",
        "--outdir",
        str(od),
        "--outname",
        nm,
        *(["--extended"] if extended else []),
        *(["--partial"] if partial else []),
    ]
    if dry_run:
        return cmd
    _run(cmd, env=_env(tools), log=log, what="MakeDb.py")
    db = _read_tsv(od / f"{nm}_db-pass.tsv")
    rec = _read_tsv(Path(records)) if isinstance(records, str | os.PathLike) else records
    return _with_space(db, rec)


def _write_db(db: pd.DataFrame, folder: Path, name: str) -> Path:
    return _to_tsv(db, folder / f"{name}.tsv")


def define_clones(
    db: pd.DataFrame,
    *,
    dist: float = 0.16,
    model: str = "ham",
    norm: Literal["len", "mut", "none"] = "len",
    act: Literal["set", "first"] = "set",
    mode: Literal["gene", "allele"] = "gene",
    tools: Tools | None = None,
    out_dir: str | os.PathLike | None = None,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> pd.DataFrame | list[str]:
    """Change-O ``DefineClones.py``: clones by V gene, J gene, junction length and junction distance.

    Parameters
    ----------
    db
        AIRR table from :func:`makedb` (heavy chains; light chains are not used by DefineClones).
    dist
        Distance threshold (normalised by junction length with ``norm="len"``); take it from
        :func:`shazam_threshold`.
    model, norm, act, mode
        ``DefineClones.py`` ``--model``, ``--norm``, ``--act`` and ``--mode``.
    tools
        From :func:`find_tools`.
    out_dir
        Working folder (default a temporary one).
    dry_run
        Return the command (with placeholder paths) without running it.
    log
        List that receives the command line.

    Returns
    -------
    pandas.DataFrame or list of str
        ``db`` with ``clone_id`` (sequences DefineClones could not group are left out by it, and keep their row
        here with a missing clone); or the command.

    References
    ----------
    Gupta et al. 2015, doi:10.1093/bioinformatics/btv359; Gupta et al. 2017, doi:10.4049/jimmunol.1601850.
    """
    tools = tools or find_tools()
    tools.require("define_clones")
    with tempfile.TemporaryDirectory(prefix="spaclone_changeo_") as tmp:
        od = Path(out_dir) if out_dir is not None else Path(tmp)
        od.mkdir(parents=True, exist_ok=True)
        src = od / "dc_input.tsv"
        cmd = [
            str(tools.changeo["DefineClones.py"]),
            "-d",
            str(src),
            "--act",
            act,
            "--model",
            model,
            "--norm",
            norm,
            "--dist",
            str(dist),
            "--mode",
            mode,
            "--outdir",
            str(od),
            "--outname",
            "dc",
            "--failed",
        ]
        if dry_run:
            return cmd
        base = db.drop(columns=["clone_id"], errors="ignore")
        _to_tsv(base, src)
        _run(cmd, env=_env(tools), log=log, what="DefineClones.py")
        res = _read_tsv(od / "dc_clone-pass.tsv")
    clone = pd.Series(res["clone_id"].astype(str).to_numpy(), index=res[_ID].astype(str).to_numpy())
    out = db.copy()
    out["clone_id"] = pd.array(clone.reindex(out[_ID].astype(str).to_numpy()).to_numpy(dtype=object), dtype="string")
    return out


def create_germlines(
    db: pd.DataFrame,
    *,
    germline: Literal["full", "dmask", "vonly", "regions"] = "dmask",
    cloned: bool = True,
    clone_key: str = "clone_id",
    tools: Tools | None = None,
    out_dir: str | os.PathLike | None = None,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> pd.DataFrame | list[str]:
    """Change-O ``CreateGermlines.py``: the germline of every sequence (one per clone with ``cloned=True``).

    Parameters
    ----------
    db
        AIRR table from :func:`makedb` (and :func:`define_clones` for ``cloned=True``).
    germline
        ``-g``: ``"dmask"`` masks the D segment and N/P additions (``germline_alignment_d_mask``).
    cloned
        ``--cloned``: one consensus germline per clone (needs ``clone_key``).
    clone_key
        Clone column (``--cf``).
    tools
        From :func:`find_tools`.
    out_dir
        Working folder (default a temporary one).
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    pandas.DataFrame or list of str
        ``db`` with ``germline_alignment_d_mask`` (or the requested germline column); or the command.
    """
    tools = tools or find_tools()
    tools.require("create_germlines")
    with tempfile.TemporaryDirectory(prefix="spaclone_changeo_") as tmp:
        od = Path(out_dir) if out_dir is not None else Path(tmp)
        od.mkdir(parents=True, exist_ok=True)
        src = od / "cg_input.tsv"
        cmd = [
            str(tools.changeo["CreateGermlines.py"]),
            "-d",
            str(src),
            "-g",
            germline,
            "-r",
            *[str(r) for r in tools.references],
            "--outdir",
            str(od),
            "--outname",
            "cg",
            "--failed",
            *(["--cloned", "--cf", clone_key] if cloned else []),
        ]
        if dry_run:
            return cmd
        base = db if not cloned else db[db[clone_key].notna()]
        _to_tsv(base, src)
        _run(cmd, env=_env(tools), log=log, what="CreateGermlines.py")
        res = _read_tsv(od / "cg_germ-pass.tsv")
    new = [c for c in res.columns if c.startswith("germline_alignment") and c not in db.columns]
    new += [c for c in res.columns if c == "germline_alignment" and c not in new]
    out = db.drop(columns=[c for c in new if c in db.columns])
    return out.merge(res[[_ID, *new]], on=_ID, how="left")


# ------------------------------------------------------------------------------------------------------- R


def _run_r(
    code: str,
    db: pd.DataFrame,
    *,
    tools: Tools | None,
    out_name: str,
    args: Sequence[str] = (),
    log: list[str] | None,
    dry_run: bool,
    timeout: float | None = None,
) -> tuple[Path | None, list[str]]:
    tools = tools or find_tools()
    tools.require("r")
    if dry_run:
        return None, [str(tools.rscript), "--vanilla", "<script.R>", "<input.tsv>", f"<{out_name}>", *args]
    tmp = Path(tempfile.mkdtemp(prefix="spaclone_r_"))
    script = tmp / "script.R"
    script.write_text(code)
    src = _write_db(db, tmp, "input")
    out = tmp / out_name
    cmd = [str(tools.rscript), "--vanilla", str(script), str(src), str(out), *args]
    try:
        _run(cmd, env=_env(tools), log=log, what="Rscript", timeout=timeout)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out, cmd


_R_THRESHOLD = r"""
suppressPackageStartupMessages({library(shazam); library(airr)})
a <- commandArgs(TRUE)
db <- read_rearrangement(a[1])
db <- distToNearest(db, sequenceColumn = "junction", vCallColumn = "v_call", jCallColumn = "j_call",
                    model = a[3], normalize = a[4], nproc = as.integer(a[6]))
th <- findThreshold(db$dist_nearest, method = a[5])
thr <- if (is.null(th)) NA else th@threshold
write.table(data.frame(threshold = thr, n = sum(!is.na(db$dist_nearest))), a[2], sep = "\t", row.names = FALSE,
            quote = FALSE)
write.table(db[, c("sequence_id", "dist_nearest")], paste0(a[2], ".dist"), sep = "\t", row.names = FALSE,
            quote = FALSE)
"""


def shazam_threshold(
    db: pd.DataFrame,
    *,
    model: str = "ham",
    normalize: Literal["len", "none"] = "len",
    method: Literal["density", "gmm"] = "density",
    nproc: int = 1,
    tools: Tools | None = None,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> dict[str, Any] | list[str]:
    """Clonal distance threshold from the distance-to-nearest distribution (SHazaM).

    Parameters
    ----------
    db
        Heavy-chain AIRR table (``junction``, ``v_call``, ``j_call``), e.g. from :func:`makedb`.
    model, normalize
        ``shazam::distToNearest`` distance model and normalisation.
    method
        ``shazam::findThreshold`` method (``"density"`` or ``"gmm"``).
    nproc
        R worker processes.
    tools
        From :func:`find_tools`.
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    dict or list of str
        ``{"threshold": float (NaN when no threshold is found), "n": sequences with a nearest neighbour,
        "dist_nearest": DataFrame (sequence_id, dist_nearest)}``; or the command.

    References
    ----------
    Nouri & Kleinstein 2018, doi:10.1093/bioinformatics/bty235; Gupta et al. 2017, doi:10.4049/jimmunol.1601850.
    """
    out, cmd = _run_r(
        _R_THRESHOLD,
        db,
        tools=tools,
        out_name="threshold.tsv",
        args=(model, normalize, method, str(int(nproc))),
        log=log,
        dry_run=dry_run,
    )
    if out is None:
        return cmd
    t = pd.read_csv(out, sep="\t")
    dist = pd.read_csv(str(out) + ".dist", sep="\t", dtype={_ID: str})
    thr = pd.to_numeric(t["threshold"], errors="coerce").iloc[0]
    shutil.rmtree(out.parent, ignore_errors=True)
    return {"threshold": float(thr), "n": int(t["n"].iloc[0]), "dist_nearest": dist}


_R_MUTATIONS = r"""
suppressPackageStartupMessages({library(shazam); library(airr)})
a <- commandArgs(TRUE)
db <- read_rearrangement(a[1])
reg <- switch(a[4], imgt_v = IMGT_V, imgt_v_by_regions = IMGT_V_BY_REGIONS, NULL)
db <- observedMutations(db, sequenceColumn = "sequence_alignment", germlineColumn = a[3], regionDefinition = reg,
                        frequency = as.logical(a[5]), combine = as.logical(a[6]), nproc = 1)
keep <- c("sequence_id", grep("^mu_(count|freq)", names(db), value = TRUE))
write.table(db[, keep], a[2], sep = "\t", row.names = FALSE, quote = FALSE)
"""


def observed_mutations(
    db: pd.DataFrame,
    *,
    germline_key: str = "germline_alignment_d_mask",
    regions: Literal["imgt_v", "imgt_v_by_regions", "all"] = "imgt_v",
    frequency: bool = True,
    combine: bool = True,
    tools: Tools | None = None,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> pd.DataFrame | list[str]:
    """Observed mutations from the germline per sequence (``shazam::observedMutations``).

    Parameters
    ----------
    db
        AIRR table with IMGT-gapped ``sequence_alignment`` and a germline (:func:`create_germlines`).
    germline_key
        Germline column.
    regions
        ``"imgt_v"`` (IMGT V positions 1-312), ``"imgt_v_by_regions"`` (CDR / FWR separately) or ``"all"``.
    frequency, combine
        Mutation frequencies instead of counts; replacement and silent combined.
    tools
        From :func:`find_tools`.
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    pandas.DataFrame or list of str
        ``db`` with SHazaM's ``mu_freq`` (or ``mu_count``, or per-region / R / S columns); or the command.

    References
    ----------
    Yaari et al. 2013, doi:10.3389/fimmu.2013.00358; Gupta et al. 2015, doi:10.1093/bioinformatics/btv359.
    """
    # SHazaM stops on the whole table when one sequence / germline pair differs in length (clonal germlines of
    # CreateGermlines --cloned can be shorter or longer than a member's alignment): count only the equal pairs
    ok = np.ones(len(db), dtype=bool)
    if "sequence_alignment" in db.columns and germline_key in db.columns:
        n_seq = db["sequence_alignment"].astype("string").str.len()
        n_germ = db[germline_key].astype("string").str.len()
        ok = (n_seq == n_germ).fillna(False).to_numpy(dtype=bool)
    if not dry_run and not ok.all():
        logger.warning(
            "observed_mutations: %d of %d sequences have no germline of their alignment's length; their mutation "
            "columns are NaN",
            int((~ok).sum()),
            len(db),
        )
    out, cmd = _run_r(
        _R_MUTATIONS,
        db if ok.all() else db[ok],
        tools=tools,
        out_name="mutations.tsv",
        args=(germline_key, regions, str(bool(frequency)).upper(), str(bool(combine)).upper()),
        log=log,
        dry_run=dry_run,
    )
    if out is None:
        return cmd
    mu = pd.read_csv(out, sep="\t", dtype={_ID: str})
    shutil.rmtree(out.parent, ignore_errors=True)
    base = db.drop(columns=[c for c in mu.columns if c != _ID and c in db.columns])
    return base.merge(mu, on=_ID, how="left")


_R_TREES = r"""
suppressPackageStartupMessages({library(dowser); library(dplyr); library(ape)})
a <- commandArgs(TRUE)
d <- read.delim(a[1], stringsAsFactors = FALSE)
d$spaclone_ids <- d$sequence_id
traits <- if (nchar(a[3])) strsplit(a[3], ",")[[1]] else NULL
set.seed(as.integer(a[7]))
clones <- formatClones(d, traits = traits, germ = a[8], collapse = TRUE, minseq = as.integer(a[5]),
                       text_fields = "spaclone_ids", nproc = as.integer(a[6]))
if (a[4] == "igphyml") {
  trees <- getTrees(clones, build = "igphyml", exec = a[9], nproc = as.integer(a[6]))
} else {
  trees <- getTrees(clones, build = a[4], nproc = as.integer(a[6]))
}
out <- list()
for (i in seq_len(nrow(trees))) {
  tr <- trees$trees[[i]]
  L <- nchar(trees$data[[i]]@germline)
  nt <- length(tr$tip.label)
  nn <- max(tr$edge)
  parent <- rep(NA_integer_, nn)
  blen <- rep(0, nn)
  parent[tr$edge[, 2]] <- tr$edge[, 1]
  blen[tr$edge[, 2]] <- tr$edge.length * L
  lab <- c(tr$tip.label, rep("", nn - nt))
  dd <- trees$data[[i]]@data
  m <- match(lab, dd$sequence_id)
  out[[i]] <- data.frame(clone_id = trees$clone_id[i], node = seq_len(nn), parent = parent, branch_mut = blen,
                         is_tip = seq_len(nn) <= nt, label = lab,
                         tip_ids = ifelse(is.na(m), "", dd$spaclone_ids[m]), stringsAsFactors = FALSE)
}
write.table(bind_rows(out), a[2], sep = "\t", row.names = FALSE, quote = FALSE)
"""


def dowser_trees(
    db: pd.DataFrame,
    *,
    build: Literal["pratchet", "pml", "dnapars", "igphyml"] = "pratchet",
    traits: Sequence[str] = (),
    germline_key: str = "germline_alignment_d_mask",
    minseq: int = 3,
    nproc: int = 1,
    seed: int = 0,
    igphyml: str | os.PathLike | None = None,
    tools: Tools | None = None,
    dry_run: bool = False,
    log: list[str] | None = None,
) -> Any:
    """Germline-rooted lineage trees with dowser, read back as :class:`spaclone.bcr.LineageTrees`.

    Parameters
    ----------
    db
        Cloned AIRR table with germlines (:func:`create_germlines`), one row per sequence; spatial columns may be
        carried as ``traits``.
    build
        ``dowser::getTrees`` method: ``"pratchet"`` (maximum parsimony, phangorn), ``"pml"`` (maximum likelihood),
        ``"dnapars"`` (PHYLIP) or ``"igphyml"`` (needs ``igphyml``).
    traits
        Columns that must also be identical for sequences to be collapsed into one tip (e.g. isotype, region).
    germline_key
        Germline column (``formatClones(germ=)``).
    minseq
        Smallest clone (distinct sequences) to build a tree for.
    nproc
        R worker processes (dowser's search is then not bit-identical between runs).
    seed
        ``set.seed`` before the search.
    igphyml
        IgPhyML executable for ``build="igphyml"`` (default ``SPACLONE_IGPHYML`` / ``PATH``).
    tools
        From :func:`find_tools`.
    dry_run
        Return the command without running it.
    log
        List that receives the command line.

    Returns
    -------
    spaclone.bcr.LineageTrees or list of str
        Trees read with :func:`spaclone.bcr.read_trees` from dowser's node table (branch lengths in mutations,
        rooted on the germline; ``members`` lists the input sequences collapsed into each tip), so that
        ``spaclone.bcr.tip_table(trees, db)`` gives one row per input sequence with its tissue position. Or the
        command.

    References
    ----------
    Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885 (dowser); Hoehn et al. 2019, doi:10.1073/pnas.1906020116
    (IgPhyML); Schliep 2011, doi:10.1093/bioinformatics/btq706 (phangorn).
    """
    from spaclone._versions import find_tool
    from spaclone.bcr.lineage import read_trees

    exe = ""
    if build == "igphyml":
        p = find_tool("igphyml", igphyml)
        if p is None:
            raise FileNotFoundError("build='igphyml' needs IgPhyML (igphyml= or SPACLONE_IGPHYML)")
        exe = str(p)
    for c in ("clone_id", germline_key, "sequence_alignment", *traits):
        if c not in db.columns:
            raise KeyError(f"dowser_trees: column {c!r} not in the table")
    out, cmd = _run_r(
        _R_TREES,
        db,
        tools=tools,
        out_name="nodes.tsv",
        args=(",".join(traits), build, str(int(minseq)), str(int(nproc)), str(int(seed)), germline_key, exe),
        log=log,
        dry_run=dry_run,
    )
    if out is None:
        return cmd
    nodes = pd.read_csv(out, sep="\t", dtype={"clone_id": str, "label": str}, keep_default_na=False)
    shutil.rmtree(out.parent, ignore_errors=True)
    nodes["parent"] = pd.to_numeric(nodes["parent"].replace("", np.nan), errors="coerce")
    return read_trees(nodes, clone_key="clone_id", length_key="branch_mut")


# ---------------------------------------------------------------------------------------------- pipeline


@dataclass(frozen=True, repr=False)
class ImmcantationResult:
    """Output of :func:`run`.

    Attributes
    ----------
    air
        The receptor table with the Immcantation columns on every record (prefix ``imm_``).
    db
        The Immcantation AIRR table (one row per sequence) with the records' spatial columns.
    threshold
        The SHazaM threshold used for DefineClones (NaN when a fixed ``dist`` was given).
    commands
        Every command that was run, in order.
    out_dir
        Folder with the intermediate files.
    """

    air: pd.DataFrame
    db: pd.DataFrame
    threshold: float
    commands: list[str]
    out_dir: Path

    def __repr__(self) -> str:
        n_clones = self.db["clone_id"].nunique() if "clone_id" in self.db.columns else 0
        return (
            f"ImmcantationResult({len(self.db)} sequences, {n_clones} clones, threshold {self.threshold:.3g}, "
            f"{len(self.commands)} commands; files in {self.out_dir})"
        )


def run(
    air: pd.DataFrame,
    out_dir: str | os.PathLike,
    *,
    tools: Tools | None = None,
    sequence_key: str = "sequence",
    dist: float | Literal["shazam"] = "shazam",
    fallback_dist: float = 0.16,
    mutations: bool = True,
    threads: int = 1,
    locus_key: str = S.LOCUS,
    heavy: Sequence[str] = ("IGH", "TRB", "TRD"),
) -> ImmcantationResult:
    """IgBLAST -> MakeDb -> SHazaM threshold -> DefineClones -> CreateGermlines -> observedMutations, on records.

    Parameters
    ----------
    air
        Receptor table with full contig sequences (``sequence_key``).
    out_dir
        Folder for the intermediate files (kept).
    tools
        From :func:`find_tools`.
    sequence_key
        Nucleotide sequence column.
    dist
        DefineClones threshold, or ``"shazam"`` to take it from :func:`shazam_threshold` (``fallback_dist`` when R
        is missing or no threshold is found).
    fallback_dist
        Threshold when SHazaM gives none.
    mutations
        Add SHazaM's ``mu_freq`` (needs R).
    threads
        IgBLAST threads.
    locus_key
        Chain column; clones and germlines are made for the chains in ``heavy``.
    heavy
        Chains passed to DefineClones / CreateGermlines.

    Returns
    -------
    ImmcantationResult
        Records with ``imm_*`` columns (``imm_v_call``, ``imm_sequence_alignment``, ``imm_clone_id``,
        ``imm_germline_alignment_d_mask``, ``imm_mu_freq``, ...), the AIRR table with spatial columns, the threshold
        and the commands.

    Notes
    -----
    Records keep their coordinates throughout (ids ``r<row>``), so the output feeds :func:`spaclone.pp.merge_loci`,
    :func:`spaclone.bcr.shm`, :func:`spaclone.bcr.build_trees` / :func:`dowser_trees` and the spatial tests directly.
    """
    tools = tools or find_tools()
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    files = export_airr(air, od, sequence_key=sequence_key, locus_key=locus_key)
    f7 = igblast(files["fasta"], tools=tools, threads=threads, log=log)
    db = makedb(f7, files["fasta"], tools=tools, records=files["records"], out_dir=od, name="spaclone", log=log)
    hv = db[db["locus"].astype(str).isin(list(heavy))] if "locus" in db.columns else db
    thr = float("nan")
    if dist == "shazam":
        if tools.rscript is not None:
            try:
                thr = shazam_threshold(hv, tools=tools, log=log)["threshold"]
            except RuntimeError as e:
                logger.warning("SHazaM threshold failed (%s); using %s", str(e).splitlines()[0], fallback_dist)
        d = thr if np.isfinite(thr) else fallback_dist
    else:
        d = float(dist)
    cl = define_clones(hv, dist=d, tools=tools, out_dir=od / "define_clones", log=log)
    cl = create_germlines(cl, tools=tools, out_dir=od / "create_germlines", log=log)
    if mutations and tools.rscript is not None:
        cl = observed_mutations(cl, tools=tools, log=log)
    rest = db[~db[_ID].isin(cl[_ID])]
    full = pd.concat([cl, rest], ignore_index=True)
    full = full.assign(_r=_row_of(full[_ID])).sort_values("_r").drop(columns="_r").reset_index(drop=True)
    full.to_csv(od / "spaclone_immcantation.tsv", sep="\t", index=False)
    (od / "commands.txt").write_text("\n".join(log) + "\n")
    return ImmcantationResult(to_records(full, air), full, thr, log, od)
