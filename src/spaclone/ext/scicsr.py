"""Bridge to the original sciCSR R package (Ng et al. 2023): signatures, definitions, export, runs and agreement.

spaclone re-implements sciCSR's analyses natively (:func:`spaclone.bcr.csr_potential`, :func:`spaclone.bcr.shm_score`,
:func:`spaclone.bcr.ighc_transcripts`, :func:`spaclone.tl.transition_model`, :func:`spaclone.tl.tpt_flux`) and works
on spatial units. This module connects those functions to the original package (sciCSR 0.3.5, GPL-3,
https://github.com/Fraternalilab/sciCSR; no sciCSR code is copied here):

* :func:`nmf_signatures`: sciCSR's isotype NMF signatures (the reference of the reference-based CSR potential),
  downloaded at run time from the pinned sciCSR commit (or read from a file, or exported by an installed sciCSR),
  because they are part of the GPL-licensed package and are not shipped with spaclone;
* :func:`scicsr_definitions`: sciCSR's curated IGHC genomic definitions (via R) for
  :func:`spaclone.bcr.ighc_transcripts`;
* :func:`to_scicsr`: writes spaclone units in the form sciCSR's functions expect (an h5ad for its CellRank /
  deeptime side with ``obs`` columns ``csr_pot``, ``shm``, ``isotype`` and, for spatial graphs,
  ``obsp["connectivities"]``; CSV / Matrix Market tables for its Seurat side);
* :func:`run_r`: runs the original ``getCSRpotential``, ``getSHM``, ``fitTransitionModel`` and ``fitTPT`` through
  ``Rscript`` (with a Python environment holding CellRank 1.x and deeptime for reticulate) and returns the results;
* :func:`compare`: agreement metrics between spaclone's and sciCSR's results;
* :func:`read_rdata`: a small reader of R ``.rda`` / ``.rds`` files (numeric, integer, logical and character vectors,
  lists and matrices with dimnames).

References
----------
Ng et al. 2023, doi:10.1038/s41592-023-02060-1 (sciCSR); R Core Team, "R Internals" (serialization format).
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import logging
import lzma
import os
import shutil
import struct
import subprocess
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

from spaclone import _schema as S

__all__ = [
    "SCICSR_COMMIT",
    "compare",
    "nmf_signatures",
    "read_rdata",
    "run_r",
    "scicsr_definitions",
    "scicsr_isotype_labels",
    "to_scicsr",
]

logger = logging.getLogger("spaclone")

#: sciCSR commit (GitHub Fraternalilab/sciCSR, version 0.3.5) whose data files are fetched.
SCICSR_COMMIT = "13c30f1f444da42d1d38f1f2fb987cf3793fa74e"
_RAW = "https://raw.githubusercontent.com/Fraternalilab/sciCSR/{commit}/data/{name}.rda"
#: SHA-256 of the data files at :data:`SCICSR_COMMIT`.
_SHA256 = {
    "mouse_nmf": "23c0d2778c91eb0aad76f0a94b501a57c89561680f7cc46c1f66949d42e84e3d",
    "human_nmf": "127f3a2fdd45e58b01983cb656ed544fce8498e69a31a0f93a075f406493faec",
}


# ===================================================================================================== R data reader
class _RObject:
    """An R vector with attributes (internal)."""

    def __init__(self, value: Any, attrs: dict[str, Any] | None = None):
        self.value = value
        self.attrs = attrs or {}


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.i = 0
        self.refs: list[Any] = []

    def int(self) -> int:
        (v,) = struct.unpack_from(">i", self.buf, self.i)
        self.i += 4
        return v

    def length(self) -> int:
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def string(self, n: int) -> str:
        s = self.buf[self.i : self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def attrs(self, flags: int) -> dict[str, Any]:
        if not flags & (1 << 9):
            return {}
        pl = self.item()
        return pl if isinstance(pl, dict) else {}

    def item(self) -> Any:
        flags = self.int()
        typ = flags & 0xFF
        if typ == 254:  # NILVALUE_SXP
            return None
        if typ in (242, 241, 253, 251):  # empty / base / global env, missing arg
            return None
        if typ == 255:  # REFSXP
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if typ == 1:  # SYMSXP
            name = self.item()
            self.refs.append(name)
            return name
        if typ == 9:  # CHARSXP
            n = self.int()
            return None if n == -1 else self.string(n)
        if typ in (2, 6):  # LISTSXP / LANGSXP: a pairlist -> ordered dict of tag -> value
            out: dict[str, Any] = {}
            node_flags = flags
            while True:
                attr = self.item() if node_flags & (1 << 9) else None
                del attr
                tag = self.item() if node_flags & (1 << 10) else None
                car = self.item()
                out[tag if isinstance(tag, str) else f"[[{len(out) + 1}]]"] = car
                node_flags = self.int()
                if (node_flags & 0xFF) == 254:
                    break
                if (node_flags & 0xFF) not in (2, 6):
                    raise ValueError("read_rdata: malformed pairlist")
            return out
        if typ in (10, 13):  # LGLSXP / INTSXP (NA = -2^31)
            n = self.length()
            v = np.frombuffer(self.buf, dtype=">i4", count=n, offset=self.i).astype(np.int64)
            self.i += 4 * n
            na = v == -(2**31)
            if na.any():
                v = np.where(na, np.nan, v.astype(float))
            elif typ == 10:
                v = v.astype(bool)
            return _RObject(v, self.attrs(flags))
        if typ == 14:  # REALSXP
            n = self.length()
            v = np.frombuffer(self.buf, dtype=">f8", count=n, offset=self.i).astype(float)
            self.i += 8 * n
            return _RObject(v, self.attrs(flags))
        if typ == 16:  # STRSXP
            n = self.length()
            v = np.array([self.item() for _ in range(n)], dtype=object)
            return _RObject(v, self.attrs(flags))
        if typ in (19, 20):  # VECSXP / EXPRSXP
            n = self.length()
            v = [self.item() for _ in range(n)]
            return _RObject(v, self.attrs(flags))
        if typ == 238:  # ALTREP_SXP: compact sequences and wrappers written by R >= 3.5
            info = self.item()
            state = self.item()
            attr = self.item()
            vals = list(info.values()) if isinstance(info, dict) else []
            cls = vals[0] if vals else None
            if cls in ("compact_intseq", "compact_realseq"):
                n, start, step = (float(x) for x in state.value[:3])
                seq = start + step * np.arange(int(n))
                v = seq.astype(np.int64) if cls == "compact_intseq" else seq
                return _RObject(v, attr if isinstance(attr, dict) else {})
            if isinstance(cls, str) and cls.startswith("wrap_"):
                inner = state.value[0] if isinstance(state, _RObject) else state
                if isinstance(inner, _RObject) and isinstance(attr, dict):
                    inner.attrs = {**inner.attrs, **attr}
                return inner
            if cls == "deferred_string":
                first = next(iter(state.values())) if isinstance(state, dict) else state
                arr = first.value if isinstance(first, _RObject) else np.asarray(first)
                return _RObject(np.array([str(x) for x in arr], dtype=object), attr if isinstance(attr, dict) else {})
            raise ValueError(f"read_rdata: unsupported ALTREP class {cls!r}")
        raise ValueError(f"read_rdata: unsupported R object type {typ} (only vectors, lists and matrices)")


def _to_python(obj: Any) -> Any:
    if not isinstance(obj, _RObject):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        return obj
    val, at = obj.value, obj.attrs
    names = _to_python(at.get("names"))
    dims = _to_python(at.get("dim"))
    dimnames = _to_python(at.get("dimnames"))
    if isinstance(val, list):
        items = [_to_python(v) for v in val]
        if names is not None:
            return dict(zip([str(x) for x in names], items, strict=False))
        return items
    if dims is not None and len(dims) == 2:
        mat = np.asarray(val).reshape(int(dims[1]), int(dims[0])).T  # column-major
        rn = cn = None
        if isinstance(dimnames, list) and len(dimnames) == 2:
            rn, cn = dimnames
        return pd.DataFrame(
            mat,
            index=None if rn is None else [str(x) for x in rn],
            columns=None if cn is None else [str(x) for x in cn],
        )
    if names is not None:
        return pd.Series(val, index=[str(x) for x in names])
    return val


def read_rdata(path_or_bytes: str | Path | bytes) -> dict[str, Any] | Any:
    """Read an R ``.rda`` / ``.RData`` (``save()``) or ``.rds`` (``saveRDS()``) file without R.

    Parameters
    ----------
    path_or_bytes
        File path or the file's bytes (gzip, bzip2, xz or uncompressed; XDR serialization versions 2 and 3).

    Returns
    -------
    dict or object
        For ``.rda``: ``{name: object}``; for ``.rds``: the object. Matrices with dimnames become DataFrames, named
        vectors Series, other vectors numpy arrays, lists Python lists / dicts. S4 objects (e.g. GRanges) and
        environments are not supported.
    """
    raw = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, str | Path) else bytes(path_or_bytes)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif raw[:3] == b"BZh":
        raw = bz2.decompress(raw)
    elif raw[:6] == b"\xfd7zXZ\x00":
        raw = lzma.decompress(raw)
    rda = raw[:5] in (b"RDX2\n", b"RDX3\n")
    if rda:
        raw = raw[5:]
    if raw[:2] != b"X\n":
        raise ValueError("read_rdata: only XDR (binary, big-endian) R serialization is supported")
    r = _Reader(raw[2:])
    version = r.int()
    r.int()
    r.int()
    if version == 3:
        r.string(r.int())  # native encoding
    obj = r.item()
    return _to_python(obj)


# ===================================================================================================== signatures
def _cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        d = Path(cache_dir)
    elif os.environ.get("SPACLONE_CACHE"):
        d = Path(os.environ["SPACLONE_CACHE"]) / "scicsr"
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spaclone" / "scicsr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def nmf_signatures(
    species: Literal["mouse", "human"] = "mouse",
    *,
    path: str | Path | None = None,
    rscript: str | Path | None = None,
    cache_dir: str | Path | None = None,
    commit: str = SCICSR_COMMIT,
) -> pd.DataFrame:
    """Isotype NMF signatures of sciCSR (productive / sterile IGHC feature weights; column 1 = naive signature).

    The matrices (mouse 14 x 2, human 16 x 2, trained by the sciCSR authors on reference B-cell atlases) are part of
    the GPL-3 sciCSR package, so spaclone does not ship them: they are read from ``path``, exported by an installed
    sciCSR (``rscript``), or downloaded once from the pinned commit (:data:`SCICSR_COMMIT`), checked against a
    SHA-256 and cached (``cache_dir``, ``$SPACLONE_CACHE`` or ``~/.cache/spaclone``).

    Parameters
    ----------
    species
        ``"mouse"`` or ``"human"``.
    path
        A local ``{species}_nmf.rda`` (from the sciCSR repository) or a CSV / TSV (features x signatures).
    rscript
        ``Rscript`` of an R with sciCSR installed (used instead of the download).
    cache_dir
        Download cache.
    commit
        sciCSR commit to download from (a non-default commit skips the checksum).

    Returns
    -------
    pandas.DataFrame
        Index = sciCSR feature names (``"Ighm-S"``, ``"IGHG1-P"``, ...), columns ``naive`` and ``switched``.

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1.
    """
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    name = f"{species}_nmf"
    if path is not None:
        p = Path(path)
        if p.suffix.lower() in (".csv", ".tsv", ".txt"):
            df = pd.read_csv(p, sep="\t" if p.suffix.lower() != ".csv" else ",", index_col=0)
        else:
            df = _first_frame(read_rdata(p), name)
    elif rscript is not None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sig.csv"
            code = f'suppressMessages(library(sciCSR)); write.csv(sciCSR::{name}, "{out.as_posix()}")'
            _rscript(rscript, ["-e", code], timeout=600)
            df = pd.read_csv(out, index_col=0)
    else:
        f = _cache_dir(cache_dir) / f"{name}_{commit[:12]}.rda"
        if not f.exists():
            url = _RAW.format(commit=commit, name=name)
            logger.info("nmf_signatures: downloading %s", url)
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            if commit == SCICSR_COMMIT and hashlib.sha256(data).hexdigest() != _SHA256[name]:
                raise RuntimeError(f"nmf_signatures: checksum mismatch for {url}")
            f.write_bytes(data)
        df = _first_frame(read_rdata(f), name)
    df = df.astype(float)
    df.columns = ["naive", "switched"][: df.shape[1]] + [f"signature_{i + 1}" for i in range(2, df.shape[1])]
    df.attrs["spaclone"] = {"nmf_signatures": {"species": species, "commit": commit, "source": "sciCSR (GPL-3)"}}
    return df


def _first_frame(obj: Any, name: str) -> pd.DataFrame:
    if isinstance(obj, dict):
        obj = obj.get(name, next(iter(obj.values())))
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"{name}: expected a matrix with dimnames")
    return obj


# ===================================================================================================== R helpers
def _rscript(rscript: str | Path, args: Sequence[str], *, timeout: float | None, env: Mapping[str, str] | None = None):
    exe = shutil.which(str(rscript)) or str(rscript)
    full_env = dict(os.environ)
    full_env.pop("PYTHONPATH", None)
    full_env["PYTHONNOUSERSITE"] = "1"
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [exe, "--vanilla", *args], capture_output=True, text=True, timeout=timeout, env=full_env, check=False
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-25:])
        raise RuntimeError(f"Rscript failed (exit {proc.returncode}):\n{tail}")
    return proc


def scicsr_definitions(
    species: Literal["mouse", "human"] = "mouse", *, rscript: str | Path = "Rscript", timeout: float = 600
) -> pd.DataFrame:
    """Curated IGHC genomic definitions of sciCSR (``mouse_definitions`` on mm10, ``human_definitions`` on GRCh38).

    Exported through an installed sciCSR, in the table form of :func:`spaclone.bcr.ighc_definitions` (columns
    ``name``, ``kind`` = VDJ / C / sterile, ``chrom``, ``start``, ``end``), for exact replication of sciCSR's counts
    with :func:`spaclone.bcr.ighc_transcripts`.
    """
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "defs.csv"
        code = (
            "suppressMessages({library(sciCSR); library(GenomicRanges)}); "
            f"d <- sciCSR::{species}_definitions; "
            "f <- function(g, k) data.frame(name = if (k == 'VDJ') 'VDJ' else names(g), kind = k, "
            "chrom = as.character(seqnames(g)), start = start(g), end = end(g)); "
            "x <- rbind(f(range(d$VDJ), 'VDJ'), do.call(rbind, lapply(seq_along(d$C), function(i) "
            "rbind(f(d$C[i], 'C'), f(d$sterile[i], 'sterile'))))); "
            f'write.csv(x, "{out.as_posix()}", row.names = FALSE)'
        )
        _rscript(rscript, ["-e", code], timeout=timeout)
        df = pd.read_csv(out, dtype={"chrom": str})
    df["gene"] = df["name"]
    df["strand"] = "*"
    df.attrs["spaclone"] = {"ighc_definitions": {"species": species, "source": "sciCSR curated definitions"}}
    return df


def scicsr_isotype_labels(isotypes: Sequence[Any], *, species: Literal["mouse", "human"] = "mouse") -> list[Any]:
    """Short isotype labels of sciCSR (``"M"``, ``"G1"``, ``"G2b"``, ``"A"``) for spaclone isotype names."""
    out = []
    for v in isotypes:
        if not isinstance(v, str) or v in ("", "unknown"):
            out.append(None)
            continue
        s = v.replace("Ig", "", 1)
        out.append(s[:1].upper() + s[1:].lower())
    return out


# ===================================================================================================== export
def to_scicsr(
    units: pd.DataFrame,
    gex: Any = None,
    *,
    path: str | Path,
    graph: sparse.spmatrix | np.ndarray | None = None,
    sterile: pd.DataFrame | None = None,
    species: Literal["mouse", "human"] = "mouse",
    columns: Sequence[str] = (),
    x_key: str = S.X,
    y_key: str = S.Y,
) -> dict[str, Path]:
    """Write spaclone units in the form sciCSR's R and Python functions read.

    Parameters
    ----------
    units
        One row per unit (index = unit id), e.g. :func:`spaclone.bcr.csr_potential` joined with
        :func:`spaclone.bcr.shm_score`: ``isotype`` (exported as sciCSR's ``c_gene`` gene name and short ``isotype``
        label), and when present ``csr_pot``, ``shm``, ``v_identity``, coordinates, plus any ``columns``.
    gex
        Optional AnnData of the units' transcriptomes (written as the RNA counts; ``obs_names`` = unit ids).
    path
        Output prefix: ``{path}.h5ad``, ``{path}_meta.csv``, ``{path}_rna.mtx`` / ``_genes.txt``,
        ``{path}_ighc.csv`` and ``{path}_graph.mtx`` as applicable.
    graph
        ``n x n`` symmetric neighbour graph of the units (e.g. :func:`spaclone.tl.neighbor_graph`): written as
        ``obsp["connectivities"]`` (CellRank reads it when sciCSR's ``fitTransitionModel`` is called with
        ``do_pca = FALSE, do_neighbors = FALSE``) and as the Seurat ``RNA_nn`` graph used by ``getCSRpotential`` to
        impute isotypes.
    sterile
        IGHC counts per unit (:func:`spaclone.bcr.ighc_transcripts`), written for sciCSR's ``IGHC`` assay with its
        feature names (``Ighm-S``, ...).
    species
        For gene names.
    columns
        Extra ``units`` columns to carry (e.g. ``"region"``).
    x_key, y_key
        Coordinates (written to ``obsm["spatial"]``).

    Returns
    -------
    dict
        Paths written, by role (``"h5ad"``, ``"meta"``, ``"rna"``, ``"genes"``, ``"ighc"``, ``"graph"``).
    """
    import anndata as ad

    from spaclone.bcr.csr_dynamics import _GENE_NAMES

    prefix = Path(path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ids = units.index.astype(str)
    meta = pd.DataFrame(index=pd.Index(ids, name="cell"))
    genes = _GENE_NAMES[species]
    if "isotype" in units.columns:
        iso = units["isotype"].astype(object).to_numpy()
        meta["c_gene"] = [genes.get(v, None) if isinstance(v, str) else None for v in iso]
        meta["c_gene"] = [None if g is None else (g.upper() if species == "mouse" else g) for g in meta["c_gene"]]
        meta["isotype"] = scicsr_isotype_labels(iso, species=species)
    for c in ("csr_pot", "shm", "v_identity", "representative_p", "total_s", *columns):
        if c in units.columns:
            meta[c] = units[c].to_numpy()
    if x_key in units.columns and y_key in units.columns:
        meta[x_key] = units[x_key].to_numpy(dtype=float)
        meta[y_key] = units[y_key].to_numpy(dtype=float)
    paths: dict[str, Path] = {}
    meta.to_csv(prefix.parent / f"{prefix.name}_meta.csv")
    paths["meta"] = prefix.parent / f"{prefix.name}_meta.csv"
    obs = meta.copy()
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype("string").fillna("NA").astype(str)
    if gex is not None:
        g = gex[np.asarray(ids)]
        x = sparse.csr_matrix(g.X, dtype=float)
        var = pd.DataFrame(index=g.var_names.astype(str))
        spio.mmwrite(str(prefix.parent / f"{prefix.name}_rna.mtx"), x.T.tocoo())
        (prefix.parent / f"{prefix.name}_genes.txt").write_text("\n".join(var.index) + "\n")
        paths["rna"] = prefix.parent / f"{prefix.name}_rna.mtx"
        paths["genes"] = prefix.parent / f"{prefix.name}_genes.txt"
    else:
        x = sparse.csr_matrix((len(ids), 0), dtype=float)
        var = pd.DataFrame(index=pd.Index([], dtype=str))
    a = ad.AnnData(X=x, obs=obs, var=var)
    if x_key in meta.columns:
        a.obsm["spatial"] = meta[[x_key, y_key]].to_numpy(dtype=float)
    if graph is not None:
        gm = sparse.csr_matrix(graph, dtype=float)
        if gm.shape != (len(ids), len(ids)):
            raise ValueError(f"graph: expected {len(ids)} x {len(ids)}, got {gm.shape}")
        a.obsp["connectivities"] = gm
        a.obsp["distances"] = gm.copy()
        a.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "params": {"method": "spaclone", "n_neighbors": int(np.diff(gm.indptr).max())},
        }
        spio.mmwrite(str(prefix.parent / f"{prefix.name}_graph.mtx"), gm.tocoo())
        paths["graph"] = prefix.parent / f"{prefix.name}_graph.mtx"
    if sterile is not None:
        st = sterile.reindex(ids).fillna(0.0)
        st = st[[c for c in st.columns if c not in (x_key, y_key)]]
        names = []
        for c in st.columns:
            iso, kind = (c.rsplit("_", 1) + [None])[:2] if "_" in c else (c, "S")
            names.append(f"{genes.get(iso, iso)}-{kind}")
        st.columns = names
        st.to_csv(prefix.parent / f"{prefix.name}_ighc.csv")
        paths["ighc"] = prefix.parent / f"{prefix.name}_ighc.csv"
    a.write_h5ad(prefix.parent / f"{prefix.name}.h5ad")
    paths["h5ad"] = prefix.parent / f"{prefix.name}.h5ad"
    return paths


# ===================================================================================================== run sciCSR
_R_TEMPLATE = r"""
suppressMessages({library(sciCSR); library(Seurat); library(Matrix); library(jsonlite); library(reticulate)})
a <- fromJSON(Sys.getenv("SPACLONE_SCICSR_ARGS"))
out <- a$out
meta <- read.csv(a$meta, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
if (!is.null(a$rna)) {
  rna <- as(readMM(a$rna), "CsparseMatrix")
  dimnames(rna) <- list(readLines(a$genes), rownames(meta))
} else {
  rna <- Matrix(1, nrow = 2, ncol = nrow(meta), sparse = TRUE, dimnames = list(c("g1", "g2"), rownames(meta)))
}
obj <- CreateSeuratObject(counts = rna, meta.data = meta)
iso_genes <- a$ighc_genes
if (!is.null(a$ighc)) {
  ighc <- as.matrix(read.csv(a$ighc, row.names = 1, check.names = FALSE))
} else {
  feats <- unlist(lapply(iso_genes, function(g) paste(g, c("S", "P", "C"), sep = "-")))
  ighc <- matrix(0, nrow = nrow(meta), ncol = length(feats), dimnames = list(rownames(meta), feats))
}
colnames(ighc) <- gsub("_", "-", colnames(ighc))
obj <- mergeIgHCountsToSeurat(igh_counts = ighc, SeuratObj = obj, assay_name = "IGHC")
obj <- NormalizeData(obj, assay = "IGHC", verbose = FALSE)
knn <- FALSE
if (!is.null(a$graph)) {
  g <- as(readMM(a$graph), "CsparseMatrix")
  dimnames(g) <- list(rownames(meta), rownames(meta))
  obj@graphs[["RNA_nn"]] <- as.Graph(g)
  knn <- TRUE
} else if (isTRUE(a$seurat_neighbors)) {
  obj <- NormalizeData(obj, verbose = FALSE); obj <- FindVariableFeatures(obj, verbose = FALSE)
  obj <- ScaleData(obj, verbose = FALSE); obj <- RunPCA(obj, verbose = FALSE, npcs = min(50, ncol(obj) - 1))
  obj <- FindNeighbors(obj, verbose = FALSE)
  knn <- TRUE
}
res <- data.frame(row.names = Cells(obj))
if ("csr_potential" %in% a$steps) {
  vr <- if (is.null(a$library_size)) NULL else a$library_size
  ref <- if (is.null(a$reference)) NULL else a$reference
  obj <- getCSRpotential(obj, c_gene_anno_name = a$c_gene, reference_based = ref, knn_graph = knn,
                         vars.to.regress = vr)
  res$csr_pot <- obj$csr_pot; res$representative_p <- obj$representative_p
  res$total_s <- obj$total_s; res$isotype <- as.character(obj$isotype)
}
if ("shm" %in% a$steps) {
  obj <- getSHM(obj, v_identity_anno_name = "v_identity")
  res$shm <- obj$shm
}
write.csv(res, file.path(out, "scicsr_meta.csv"))
if (any(c("transition", "tpt") %in% a$steps)) {
  use_condaenv(a$python, required = TRUE)
  py_run_string(sprintf(paste0(
    "import pandas as pd, scanpy as sc\nadata = sc.read_h5ad('%s')\nm = pd.read_csv('%s', index_col=0)\n",
    "for c in m.columns:\n    adata.obs[c] = m.loc[adata.obs_names, c].values\nadata.write_h5ad('%s')\n"),
    a$h5ad, file.path(out, "scicsr_meta.csv"), file.path(out, "scicsr.h5ad")))
  h5 <- file.path(out, "scicsr.h5ad")
  fit <- fitTransitionModel(h5, conda_env = a$python, mode = "pseudotime", pseudotime_key = a$time_key,
                            do_pca = isTRUE(a$recompute_graph), do_neighbors = isTRUE(a$recompute_graph))
  writeMM(as(fit$cellrank_obj$transition_matrix, "CsparseMatrix"), file.path(out, "transition_matrix.mtx"))
  writeLines(as.character(fit$CellID), file.path(out, "cells.txt"))
  writeMM(as(py_eval("adata.obsp['connectivities']"), "CsparseMatrix"), file.path(out, "connectivities.mtx"))
  if ("tpt" %in% a$steps) {
    tpt <- fitTPT(h5, fit, group.cells.by = a$states_key, source_state = a$source, target_state = a$target,
                  conda_env = a$python, random_n = a$random_n)
    write.csv(as.matrix(tpt$gross_flux), file.path(out, "tpt_gross_flux.csv"))
    write.csv(as.matrix(tpt$significance), file.path(out, "tpt_significance.csv"))
    write.csv(data.frame(stationary = tpt$stationary_distribution), file.path(out, "tpt_stationary.csv"))
    pw <- tpt$pathways
    write.csv(data.frame(path = sapply(pw$path, function(p) paste(unlist(p), collapse = " > ")),
                         p_path = unlist(pw$p_path)), file.path(out, "tpt_pathways.csv"), row.names = FALSE)
    write(toJSON(list(mfpt = tpt$mfpt, total_gross_flux = tpt$total_gross_flux), auto_unbox = TRUE, digits = NA),
          file.path(out, "tpt_summary.json"))
  }
}
"""


def run_r(
    units: pd.DataFrame,
    gex: Any = None,
    *,
    rscript: str | Path = "Rscript",
    python: str | Path | None = None,
    steps: Sequence[str] = ("csr_potential", "shm"),
    species: Literal["mouse", "human"] = "mouse",
    reference: Literal["mouse", "human"] | None = None,
    graph: sparse.spmatrix | np.ndarray | None = None,
    seurat_neighbors: bool = False,
    sterile: pd.DataFrame | None = None,
    library_size: str | None = None,
    time_key: str = "csr_pot",
    states_key: str = "isotype",
    source: str | None = None,
    target: str | None = None,
    random_n: int = 100,
    recompute_graph: bool | None = None,
    workdir: str | Path | None = None,
    timeout: float | None = 7200,
) -> dict[str, Any]:
    """Run the original sciCSR functions on spaclone units through ``Rscript`` and return their results.

    The units are written with :func:`to_scicsr`; an R script then builds a Seurat object (RNA counts from ``gex``
    or a placeholder assay, the IGHC assay from ``sterile`` or zeros, the neighbour graph), and calls sciCSR's
    ``getCSRpotential`` (``c_gene_anno_name = "c_gene"``), ``getSHM`` (``"v_identity"``), ``fitTransitionModel``
    (pseudotime mode on ``time_key``) and ``fitTPT`` (``group.cells.by = states_key``).

    Parameters
    ----------
    units
        Units table (see :func:`to_scicsr`); needs ``isotype`` for the CSR potential and ``v_identity`` for SHM;
        ``time_key`` / ``states_key`` columns are taken from sciCSR's own results when it computed them.
    gex
        Optional AnnData of the units' transcriptomes.
    rscript
        ``Rscript`` of an R with sciCSR, Seurat and its Bioconductor dependencies.
    python
        Path of the Python environment reticulate should use for the transition and TPT steps (sciCSR needs
        CellRank 1.x, deeptime and scanpy there; e.g. the environment built by sciCSR's ``prepare_sciCSR``).
    steps
        Any of ``"csr_potential"``, ``"shm"``, ``"transition"``, ``"tpt"``.
    species, reference
        Gene names of the IGHC assay; ``reference`` = sciCSR's ``reference_based`` (None: empirical).
    graph
        Neighbour graph of the units, used both for sciCSR's isotype imputation (as ``RNA_nn``) and as the
        connectivities of its transition model (``do_pca = do_neighbors = FALSE``).
    seurat_neighbors
        Without ``graph``: compute Seurat's PCA / ``FindNeighbors`` graph from ``gex`` for the imputation.
    sterile
        IGHC counts per unit.
    library_size
        ``meta`` column regressed out by sciCSR's scaling (sciCSR default ``"nCount_RNA"``; None = no regression).
    time_key, states_key, source, target, random_n
        Transition and TPT settings (sciCSR's ``pseudotime_key``, ``group.cells.by``, ``source_state``,
        ``target_state``, ``random_n``).
    recompute_graph
        Let sciCSR recompute PCA and the scanpy graph from the h5ad ``X`` (its default; needs ``gex``). Default: True
        without ``graph``, False with it.
    workdir
        Directory for the exchange files (default: a temporary directory, removed afterwards).
    timeout
        Seconds before the R run is stopped.

    Returns
    -------
    dict
        ``"meta"`` (DataFrame: ``csr_pot``, ``representative_p``, ``total_s``, ``isotype``, ``shm`` per unit) and, for
        the later steps, ``"transition_matrix"`` (csr), ``"connectivities"`` (csr), ``"cells"``, ``"gross_flux"``,
        ``"significance"`` (DataFrames), ``"stationary"`` (Series), ``"pathways"`` (DataFrame), ``"mfpt"``,
        ``"total_gross_flux"``.
    """
    steps = list(steps)
    bad = [s for s in steps if s not in ("csr_potential", "shm", "transition", "tpt")]
    if bad:
        raise ValueError(f"unknown steps {bad}")
    if any(s in steps for s in ("transition", "tpt")) and python is None:
        raise ValueError("the transition / tpt steps need python= (an environment with CellRank 1.x and deeptime)")
    if "tpt" in steps and (source is None or target is None):
        raise ValueError("the tpt step needs source= and target=")
    from spaclone.bcr.csr_dynamics import _GENE_NAMES, CSR_ISOTYPES

    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="spaclone_scicsr_")
        work = Path(tmp)
    else:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
    try:
        paths = to_scicsr(units, gex, path=work / "units", graph=graph, sterile=sterile, species=species)
        genes = [_GENE_NAMES[species][i] for i in CSR_ISOTYPES[species]]
        args = {
            "out": str(work),
            "meta": str(paths["meta"]),
            "h5ad": str(paths["h5ad"]),
            "rna": str(paths["rna"]) if "rna" in paths else None,
            "genes": str(paths["genes"]) if "genes" in paths else None,
            "ighc": str(paths["ighc"]) if "ighc" in paths else None,
            "graph": str(paths["graph"]) if "graph" in paths else None,
            "ighc_genes": genes,
            "seurat_neighbors": bool(seurat_neighbors),
            "steps": steps,
            "reference": reference,
            "library_size": library_size,
            "c_gene": "c_gene",
            "python": None if python is None else str(python),
            "time_key": time_key,
            "states_key": states_key,
            "source": source,
            "target": target,
            "random_n": int(random_n),
            "recompute_graph": bool(graph is None) if recompute_graph is None else bool(recompute_graph),
        }
        script = work / "run_scicsr.R"
        script.write_text(_R_TEMPLATE)
        _rscript(rscript, [str(script)], timeout=timeout, env={"SPACLONE_SCICSR_ARGS": json.dumps(args)})
        out: dict[str, Any] = {}
        meta = pd.read_csv(work / "scicsr_meta.csv", index_col=0)
        meta.index = meta.index.astype(str)
        out["meta"] = meta
        if (work / "transition_matrix.mtx").exists():
            out["transition_matrix"] = sparse.csr_matrix(spio.mmread(str(work / "transition_matrix.mtx")))
            out["connectivities"] = sparse.csr_matrix(spio.mmread(str(work / "connectivities.mtx")))
            out["cells"] = (work / "cells.txt").read_text().split()
        if (work / "tpt_gross_flux.csv").exists():
            out["gross_flux"] = pd.read_csv(work / "tpt_gross_flux.csv", index_col=0)
            out["significance"] = pd.read_csv(work / "tpt_significance.csv", index_col=0)
            out["stationary"] = pd.read_csv(work / "tpt_stationary.csv", index_col=0)["stationary"]
            out["pathways"] = pd.read_csv(work / "tpt_pathways.csv")
            js = json.loads((work / "tpt_summary.json").read_text())
            out["mfpt"] = float(np.ravel(js["mfpt"])[0])
            out["total_gross_flux"] = float(np.ravel(js["total_gross_flux"])[0])
        out["workdir"] = None if tmp else work
        return out
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


# ===================================================================================================== agreement
def _vec_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": float(len(a))}
    if len(a) == 0:
        return {**out, "pearson": np.nan, "spearman": np.nan, "max_abs_diff": np.nan}
    out["max_abs_diff"] = float(np.max(np.abs(a - b)))
    if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
        out["pearson"] = float(np.corrcoef(a, b)[0, 1])
        out["spearman"] = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    else:
        out["pearson"] = out["spearman"] = np.nan
    return out


def compare(native: Mapping[str, Any], original: Mapping[str, Any]) -> pd.DataFrame:
    """Agreement between spaclone's and sciCSR's results.

    Parameters
    ----------
    native, original
        Dicts with any of: ``"csr_pot"``, ``"shm"``, ``"isotype"``, ``"stationary"`` (Series indexed by unit or
        state), ``"transition_matrix"`` (matrices in one unit order), ``"gross_flux"`` (DataFrames indexed by state)
        and ``"mfpt"`` (float). Series are aligned on their index.

    Returns
    -------
    pandas.DataFrame
        One row per quantity and metric: ``quantity``, ``metric`` (``n``, ``pearson``, ``spearman``,
        ``max_abs_diff``, ``agreement``, ``rel_frobenius``, ``rel_diff``) and ``value``.
    """
    rows = []

    def add(q: str, d: Mapping[str, float]) -> None:
        rows.extend({"quantity": q, "metric": k, "value": float(v)} for k, v in d.items())

    for key in ("csr_pot", "shm", "stationary"):
        if key in native and key in original:
            a, b = pd.Series(native[key]), pd.Series(original[key])
            a.index, b.index = a.index.astype(str), b.index.astype(str)
            common = a.index.intersection(b.index)
            add(key, _vec_metrics(a.loc[common].to_numpy(), b.loc[common].to_numpy()))
    if "isotype" in native and "isotype" in original:
        a, b = pd.Series(native["isotype"]).astype(str), pd.Series(original["isotype"]).astype(str)
        a.index, b.index = a.index.astype(str), b.index.astype(str)
        common = a.index.intersection(b.index)
        add("isotype", {"n": len(common), "agreement": float((a.loc[common] == b.loc[common]).mean())})
    if "transition_matrix" in native and "transition_matrix" in original:
        a = sparse.csr_matrix(native["transition_matrix"], dtype=float)
        b = sparse.csr_matrix(original["transition_matrix"], dtype=float)
        d = abs(a - b)
        add(
            "transition_matrix",
            {
                "n": float(a.shape[0]),
                "max_abs_diff": float(d.max()) if d.nnz else 0.0,
                "rel_frobenius": float(sparse.linalg.norm(a - b) / max(sparse.linalg.norm(b), 1e-300)),
            },
        )
    if "gross_flux" in native and "gross_flux" in original:
        a, b = pd.DataFrame(native["gross_flux"]), pd.DataFrame(original["gross_flux"])
        a.index, a.columns = a.index.astype(str), a.columns.astype(str)
        b.index, b.columns = b.index.astype(str), b.columns.astype(str)
        st = a.index.intersection(b.index)
        add("gross_flux", _vec_metrics(a.loc[st, st].to_numpy(), b.loc[st, st].to_numpy()))
    if "mfpt" in native and "mfpt" in original:
        x, y = float(native["mfpt"]), float(original["mfpt"])
        add("mfpt", {"rel_diff": abs(x - y) / max(abs(y), 1e-300)})
    return pd.DataFrame(rows, columns=["quantity", "metric", "value"])
