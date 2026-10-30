"""CSR potential and SHM score per unit (sciCSR's scores on cells, spots, beads or loci) and IGHC transcript counts.

sciCSR (Ng et al. 2023) orders B cells along class-switch recombination with a *CSR potential* and along somatic
hypermutation with an *SHM score*, then uses either as the time key of a transition model
(:func:`spaclone.tl.transition_model`, :func:`spaclone.tl.tpt_flux`). This module computes both scores on spatial
units, re-implemented from the paper and the package's documented behaviour (sciCSR 0.3.5, GPL-3; no code copied):

* :func:`csr_potential`: the productive isotype of every unit (the most 3' constant gene among its heavy-chain
  records, ``mode="furthest"``, or the most frequent one, ``"highest"``), with missing isotypes imputed by majority
  vote of the unit's **spatial** neighbours (or expression neighbours, as sciCSR does on its kNN graph). The
  *empirical* potential is the min-max scaled norm ``sqrt(p^2 + s^2)`` of the isotype's position ``p`` on the IgH
  locus and the unit's sterile-transcript total ``s`` (0 without sterile counts, e.g. receptor-only spatial data).
  The *reference-based* potential (``reference="mouse"`` / ``"human"``) is ``1 -`` the min-max scaled
  non-negative-least-squares weight of sciCSR's naive NMF signature in the unit's scaled productive / sterile profile
  (the signatures are fetched at run time, :func:`spaclone.ext.scicsr.nmf_signatures`).
* :func:`shm_score`: sciCSR's ``1 - V identity`` scaled to [0, 1] (missing units 0), or the same scaling of
  spaclone's allele-corrected excess V mismatches (:func:`spaclone.bcr.shm`).
* :func:`ighc_transcripts`: productive (VDJ-joined), sterile (germline / I-exon) and C-only IGHC molecules per
  barcode or spot from a BAM file (pysam, or a pure-Python reader), the spatial counterpart of sciCSR's
  ``getIGHmapping`` / ``getIGHreadType`` / ``summariseIGHreads``, with documented genomic definitions per assembly
  (:func:`ighc_definitions`).

The isotype positions follow the switch-competent constant genes in 5' to 3' order (:data:`CSR_ISOTYPES`, the
genes of sciCSR's definitions): IgD has no position, because Cδ is expressed by alternative splicing of the μ–δ
pre-mRNA rather than by switching, so a unit with only IgD records counts as un-isotyped (imputed), as in sciCSR.
Mouse IgG2a (BALB/c) and IgG2c (C57BL/6) are allelic genes at one locus position.

References
----------
Ng et al. 2023, doi:10.1038/s41592-023-02060-1 (sciCSR); Stavnezer, Guikema & Schrader 2008,
doi:10.1146/annurev.immunol.26.021607.090248 (germline transcription and CSR); Lawson & Hanson 1974 / 1995,
doi:10.1137/1.9781611971217 (NNLS); Hao et al. 2021, doi:10.1016/j.cell.2021.04.048 (Seurat scaling);
Li et al. 2009, doi:10.1093/bioinformatics/btp352 (SAM/BAM format).
"""

from __future__ import annotations

import gzip
import logging
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from spaclone import _schema as S
from spaclone.bcr.isotype import isotype as _isotype

__all__ = [
    "CSR_ISOTYPES",
    "IGH_LOCI",
    "csr_potential",
    "ighc_definitions",
    "ighc_transcripts",
    "shm_score",
]

logger = logging.getLogger("spaclone")

#: Switch-competent IgH constant genes in 5' to 3' order (the C genes of sciCSR's mouse / human definitions); the
#: position in this tuple is the isotype index ``p`` of the CSR potential.
CSR_ISOTYPES: dict[str, tuple[str, ...]] = {
    "mouse": ("IgM", "IgG3", "IgG1", "IgG2b", "IgG2c", "IgE", "IgA"),
    "human": ("IgM", "IgG3", "IgG1", "IgA1", "IgG2", "IgG4", "IgE", "IgA2"),
}
#: Allelic genes sharing a locus position.
_ALIASES: dict[str, dict[str, str]] = {"mouse": {"IgG2a": "IgG2c"}, "human": {}}
#: Ensembl gene names of the constant genes (for genomic definitions and BAM output).
_GENE_NAMES: dict[str, dict[str, str]] = {
    "mouse": {
        "IgM": "Ighm",
        "IgD": "Ighd",
        "IgG3": "Ighg3",
        "IgG1": "Ighg1",
        "IgG2b": "Ighg2b",
        "IgG2c": "Ighg2c",
        "IgE": "Ighe",
        "IgA": "Igha",
    },
    "human": {
        "IgM": "IGHM",
        "IgD": "IGHD",
        "IgG3": "IGHG3",
        "IgG1": "IGHG1",
        "IgA1": "IGHA1",
        "IgG2": "IGHG2",
        "IgG4": "IGHG4",
        "IgE": "IGHE",
        "IgA2": "IGHA2",
    },
}

#: Ensembl coordinates (1-based, inclusive; all genes on the minus strand) of the IgH constant genes and of the
#: V(D)J span (first to last IG_V / IG_D / IG_J gene, pseudogenes included) per assembly: GRCm39 and GRCh38 from
#: Ensembl release 115 (rest.ensembl.org, 2026-10-30), GRCm38 = Ensembl 102 (via the Ensembl GRCm39 -> GRCm38
#: assembly map; identical to the gene ranges of sciCSR's mm10 definitions).
IGH_LOCI: dict[str, dict[str, Any]] = {}  # filled below by _register_loci()


def _register_loci() -> None:
    IGH_LOCI.update(_IGH_COORDS)


# ===================================================================================================== isotypes
def _species_order(species: str, isotypes: Sequence[str] | None) -> tuple[tuple[str, ...], dict[str, int]]:
    if isotypes is None:
        if species not in CSR_ISOTYPES:
            raise ValueError(f"species must be one of {sorted(CSR_ISOTYPES)}, got {species!r}")
        isotypes = CSR_ISOTYPES[species]
    iso = tuple(str(i) for i in isotypes)
    if len(set(iso)) != len(iso):
        raise ValueError(f"isotypes must be distinct, got {iso}")
    pos = {s: i for i, s in enumerate(iso)}
    for a, b in _ALIASES.get(species, {}).items():
        if b in pos and a not in pos:
            pos[a] = pos[b]
    return iso, pos


def _class_first(order: tuple[str, ...]) -> dict[str, int]:
    """Position of the 5'-most gene of each class (IgG -> IgG3, human IgA -> IgA1)."""
    out: dict[str, int] = {}
    for i, s in enumerate(order):
        cls = s[:3]
        out.setdefault(cls, i)
    return out


def _positions(
    calls: pd.Series, *, species: str, pos: dict[str, int], order: tuple[str, ...], class_only: str
) -> np.ndarray:
    """CSR position of every constant-gene call (NaN: no call, IgD, a light chain or an unknown gene)."""
    raw = calls.astype(object)
    # sciCSR-style short labels ("M", "G2b", "A1") -> IGHM, IGHG2B, IGHA1
    short = raw.map(lambda v: isinstance(v, str) and re.fullmatch(r"[MDGAEmdgae]\d?[A-Ca-c]?", v.strip()) is not None)
    raw = raw.where(~short, "IGH" + raw.astype(str).str.strip().str.upper())
    parsed = _isotype(raw, species=None)
    cls = parsed["isotype"].astype(object).to_numpy()
    sub = parsed["isotype_subclass"].astype(object).to_numpy()
    out = np.full(len(calls), np.nan)
    first = _class_first(order)
    for i, (c, s) in enumerate(zip(cls, sub, strict=True)):
        if isinstance(s, str) and s in pos:
            out[i] = pos[s]
        elif isinstance(c, str) and c != "unknown":
            if c in pos:  # a class with one gene (mouse IgA, IgE, IgM)
                out[i] = pos[c]
            elif class_only == "lower" and c in first and (not isinstance(s, str) or s == c):
                out[i] = first[c]  # class-only call ("IGHG"): the class's 5'-most gene
    return out


# ===================================================================================================== units
def _unit_column(air: pd.DataFrame, unit_key: str | None) -> str | None:
    if unit_key != "auto":
        if unit_key is not None and unit_key not in air.columns:
            raise KeyError(f"unit_key {unit_key!r} not in the table")
        return unit_key
    for col in (S.LOCUS_ID, S.SPOT, "cell_id"):
        if col in air.columns:
            return col
    return None


def _records_to_units(
    air: pd.DataFrame,
    *,
    unit_key: str | None,
    pos: np.ndarray,
    weight: str,
    mode: str,
    locus_key: str,
    loci: Sequence[str],
    productive_key: str,
    reads_key: str,
    x_key: str,
    y_key: str,
) -> pd.DataFrame:
    """One row per unit: furthest / highest CSR position over its productive heavy-chain records, coordinates."""
    col = _unit_column(air, unit_key)
    ids = air.index.astype(str).to_numpy() if col is None else air[col].astype(str).to_numpy()
    if col is not None and air[col].isna().any():
        raise ValueError(f"unit column {col!r} has missing values")
    is_unit = np.ones(len(air), dtype=bool)
    heavy = np.ones(len(air), dtype=bool)
    if locus_key in air.columns:
        loc = air[locus_key].astype("string").fillna("")
        is_unit = np.asarray(loc.isin(list(loci)).to_numpy(dtype=bool, na_value=False), dtype=bool)
        heavy = np.asarray((loc == "IGH").to_numpy(dtype=bool, na_value=False), dtype=bool)
    if productive_key in air.columns:
        # productive flags as bool, "T" / "F", "True" / "False" or 0 / 1 (any dtype); only explicit False excludes
        flag = air[productive_key].astype("string").str.strip().str.upper()
        heavy &= ~np.asarray(flag.isin(["F", "FALSE", "0"]).to_numpy(dtype=bool, na_value=False), dtype=bool)
    inv, uid = pd.factorize(pd.Series(ids[is_unit], dtype=object), sort=False)
    inv = np.asarray(inv, dtype=np.int64)
    n = len(uid)
    out = pd.DataFrame(index=pd.Index(np.asarray(uid, dtype=object).astype(str), name="unit"))
    if x_key in air.columns and y_key in air.columns:
        xs = air[x_key].to_numpy(dtype=float)[is_unit]
        ys = air[y_key].to_numpy(dtype=float)[is_unit]
        cnt = np.bincount(inv, minlength=n)
        out[S.X] = np.bincount(inv, weights=xs, minlength=n) / cnt
        out[S.Y] = np.bincount(inv, weights=ys, minlength=n) / cnt
    sel = heavy[is_unit] & np.isfinite(pos[is_unit])
    p = pos[is_unit][sel]
    u = inv[sel]
    if weight == "records":
        w = np.ones(len(p))
    elif weight == "reads":
        if reads_key not in air.columns:
            raise KeyError(f"weight='reads' needs the reads column {reads_key!r}")
        w = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)[is_unit][sel]
        w = np.where(np.isfinite(w), w, 0.0)
    else:
        raise ValueError(f"weight must be 'records' or 'reads', got {weight!r}")
    out["n_heavy"] = np.bincount(u, minlength=n).astype(np.int64)
    rep = np.full(n, np.nan)
    if len(p):
        if mode == "furthest":
            best = np.full(n, -np.inf)
            np.maximum.at(best, u, p)
            rep = np.where(np.isfinite(best), best, np.nan)
        elif mode == "highest":
            pi = p.astype(np.int64)
            n_pos = int(pi.max()) + 1
            tot = np.bincount(u * n_pos + pi, weights=w, minlength=n * n_pos).reshape(n, n_pos)
            has = np.bincount(u * n_pos + pi, minlength=n * n_pos).reshape(n, n_pos) > 0
            tot = np.where(has, tot, -np.inf)
            mx = tot.max(axis=1, keepdims=True)
            # ties: the most 3' isotype (sciCSR: max(which(z == max(z))))
            last = n_pos - 1 - np.argmax((tot == mx)[:, ::-1], axis=1)
            rep = np.where(np.isfinite(mx[:, 0]), last.astype(float), np.nan)
        else:
            raise ValueError(f"mode must be 'furthest' or 'highest', got {mode!r}")
    out["position_observed"] = rep
    return out


def _knn_adjacency(xy: np.ndarray, k: int, max_distance_um: float | None) -> sparse.csr_matrix:
    """Unweighted union-symmetrised kNN graph (each unit's k nearest others, ties at the k-th distance included)."""
    from spaclone.tl.transitions import _knn_pairs

    n = len(xy)
    rows, cols, _, _ = _knn_pairs(xy, k, max_distance_um)
    a = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return ((a + a.T) > 0).astype(float).tocsr()


def _impute(pos: np.ndarray, adj: sparse.csr_matrix, fallback: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote of the neighbours' observed positions for units without one (ties: the 5'-most isotype)."""
    out = pos.copy()
    need = np.flatnonzero(~np.isfinite(pos))
    if not len(need):
        return out, np.zeros(len(pos), dtype=bool)
    adj = sparse.csr_matrix(adj)
    sub = adj[need].tocoo()
    nb_pos = pos[sub.col]
    ok = np.isfinite(nb_pos)
    row, val = sub.row[ok], nb_pos[ok].astype(np.int64)
    voted = np.full(len(need), np.nan)
    if len(val):
        n_pos = int(val.max()) + 1
        cnt = np.bincount(row * n_pos + val, minlength=len(need) * n_pos).reshape(len(need), n_pos)
        has = cnt.max(axis=1) > 0
        voted[has] = np.argmax(cnt[has], axis=1)  # first maximum = smallest position (sciCSR's which.max(table))
    if fallback is not None:
        voted = np.where(np.isfinite(voted), voted, float(fallback))
    out[need] = voted
    imputed = np.zeros(len(pos), dtype=bool)
    imputed[need] = np.isfinite(voted)
    return out, imputed


# ===================================================================================================== IGHC tables
_FEATURE = re.compile(r"^(?P<gene>[A-Za-z0-9]+?)(?:[-_.](?P<kind>[SPCIspci]))?$")


def _parse_feature(name: str, species: str, pos: dict[str, int]) -> tuple[str | None, str | None]:
    """(isotype, kind) of an IGHC count column: ``Ighm-S``, ``IGHG1_P``, ``IgG2b_C``, ``IgA`` (kind None)."""
    m = _FEATURE.match(str(name).strip())
    if not m:
        return None, None
    gene, kind = m.group("gene"), m.group("kind")
    parsed = _isotype(pd.Series([gene]), species=None)
    sub = parsed["isotype_subclass"].iat[0]
    cls = parsed["isotype"].iat[0]
    iso = sub if isinstance(sub, str) else cls
    if not isinstance(iso, str) or iso == "unknown":
        return None, None
    iso = _ALIASES.get(species, {}).get(iso, iso)
    kind = None if kind is None else kind.upper().replace("I", "S")
    return iso, kind


def _ighc_frame(table: Any, units: pd.Index, species: str, pos: dict[str, int]) -> pd.DataFrame:
    """Units x features counts with canonical column names ``{isotype}_{S|P|C}`` (missing units are zero rows)."""
    if hasattr(table, "obs_names") and hasattr(table, "var_names"):
        x = table.X.toarray() if sparse.issparse(table.X) else np.asarray(table.X)
        table = pd.DataFrame(x, index=table.obs_names.astype(str), columns=table.var_names.astype(str))
    if not isinstance(table, pd.DataFrame):
        raise TypeError("sterile must be a DataFrame (units x IGHC features) or an AnnData")
    cols, keep = [], []
    for c in table.columns:
        iso, kind = _parse_feature(c, species, pos)
        if iso is None:
            continue
        cols.append(f"{iso}_{kind or 'S'}")
        keep.append(c)
    if not keep:
        raise ValueError("sterile: no column could be read as an IGHC feature (e.g. 'Ighg1-S', 'IgG1_S', 'IgG1')")
    df = table[keep].copy()
    df.columns = cols
    df = df.T.groupby(level=0, sort=False).sum().T  # duplicate names (e.g. IgG2a + IgG2c) are added
    df.index = df.index.astype(str)
    df = df.reindex(units).fillna(0.0).astype(float)
    return df


def _productive_positions(ighc: pd.DataFrame, pos: dict[str, int], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Furthest / highest productive isotype from ``{isotype}_P`` molecule counts (sciCSR's ``getIsotype``)."""
    cols = [c for c in ighc.columns if c.endswith("_P") and c[:-2] in pos]
    n = len(ighc)
    if not cols:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int64)
    p_of = np.array([pos[c[:-2]] for c in cols], dtype=float)
    cnt = ighc[cols].to_numpy(dtype=float)
    has = cnt > 0
    any_ = has.any(axis=1)
    if mode == "furthest":
        rep = np.where(has, p_of[None, :], -np.inf).max(axis=1)
    else:
        mx = cnt.max(axis=1, keepdims=True)
        rep = np.where(has & (cnt == mx), p_of[None, :], -np.inf).max(axis=1)  # ties: the most 3' isotype
    rep = np.where(any_, rep, np.nan)
    return rep, cnt.sum(axis=1).astype(np.int64)


def _seurat_scale(counts: np.ndarray, library_size: np.ndarray | None, scale_max: float = 10.0) -> np.ndarray:
    """Seurat scaling as in sciCSR's default ``scale.data``: LogNormalize (1e4), library-size regression, z, clip."""
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(tot > 0, np.log1p(counts / np.where(tot > 0, tot, 1.0) * 1e4), 0.0)
    if library_size is not None:
        design = np.column_stack([np.ones(len(x)), np.asarray(library_size, dtype=float)])
        q, _ = np.linalg.qr(design)
        x = x - q @ (q.T @ x)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mu) / sd
    z = np.where(z > scale_max, scale_max, z)
    return np.where(np.isfinite(z), z, 0.0)


def _minmax(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    if not ok.any():
        return np.full_like(v, np.nan)
    lo, hi = np.nanmin(v[ok]), np.nanmax(v[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v - lo) / abs(hi - lo)


def csr_potential(
    units: Any,
    gex: Any = None,
    *,
    unit_key: str | None = "auto",
    sterile: Any = None,
    mode: Literal["furthest", "highest"] = "furthest",
    reference: Literal["mouse", "human"] | pd.DataFrame | None = None,
    impute: Literal["spatial", "expression"] | sparse.spmatrix | None = "spatial",
    k: int = 15,
    species: Literal["mouse", "human"] = "mouse",
    isotypes: Sequence[str] | None = None,
    isotype_key: str | None = None,
    isotype_source: Literal["receptor", "transcripts"] = "receptor",
    class_only: Literal["lower", "missing"] = "lower",
    weight: Literal["records", "reads"] = "records",
    sterile_norm: Literal["seurat", "log1p", "counts", "precomputed"] = "seurat",
    library_size: str | Sequence[float] | None = "auto",
    fallback: Literal["IgM"] | None = "IgM",
    max_distance_um: float | None = None,
    connectivities_key: str = "connectivities",
    loci: Sequence[str] = S.BCR_LOCI,
    c_key: str = S.C_CALL,
    locus_key: str = S.LOCUS,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
    spatial_key: str = "spatial",
    x_key: str = S.X,
    y_key: str = S.Y,
) -> pd.DataFrame:
    """CSR potential (sciCSR) of every unit (cell, spot, bead, locus), with spatial imputation of missing isotypes.

    Parameters
    ----------
    units
        Receptor records (an air table: one row per contig / molecule, with ``c_call`` and a unit column), a units
        table (one row per unit with an isotype column, ``isotype_key``) or an AnnData of units (``obs[isotype_key]``,
        ``obsm[spatial_key]``).
    gex
        AnnData of the units' transcriptomes (``obs_names`` = unit ids) for ``impute="expression"`` on a table.
    unit_key
        Column grouping records into units (``"auto"``: ``locus_id``, else ``spot_id``, else ``cell_id``; None: every
        row is a unit). Units are the ids carrying at least one record of ``loci`` (BCR chains by default), so a spot
        with only light-chain records is a unit without an isotype.
    sterile
        Optional IGHC counts per unit (a DataFrame indexed by unit id, or an AnnData), e.g. from
        :func:`ighc_transcripts`: columns ``{gene}{-|_}{S|P|C}`` (``Ighg1-S``, ``IgG1_S``; sciCSR's assay names work)
        or bare isotype names (sterile counts). Units missing from the table have zero counts. Without it the
        sterile total is 0 (state this for receptor-only spatial data: the potential is then the isotype position).
    mode
        Productive isotype of a unit with several heavy-chain isotypes: ``"furthest"`` (the most 3' gene, sciCSR's
        default) or ``"highest"`` (the most frequent; ties to the most 3').
    reference
        None: the empirical potential. ``"mouse"`` / ``"human"`` (or a signature DataFrame: rows = IGHC features,
        first column = the naive signature): ``1 -`` min-max(NNLS weight of the naive signature) on the scaled IGHC
        profile; needs ``sterile`` with productive and sterile columns.
    impute
        Neighbours that vote the isotype of units without one: ``"spatial"`` (the ``k`` nearest units in µm, ties at
        the k-th distance included),
        ``"expression"`` (the support of ``obsp[connectivities_key]`` of the AnnData, or a ``k``-neighbour scanpy
        graph), a precomputed ``n x n`` adjacency in the order of the output units (first appearance in ``units``:
        the row order of a units table or AnnData; e.g. Seurat's ``RNA_nn`` graph for exact sciCSR replication), or
        None (no imputation).
    k
        Neighbours for ``"spatial"`` / a computed expression graph (each unit's ``k`` nearest, union-symmetrised).
    species
        ``"mouse"`` or ``"human"`` (isotype order and aliases).
    isotypes
        Constant genes in 5' to 3' order defining the positions (default :data:`CSR_ISOTYPES`; e.g. add ``"IgD"``
        after ``"IgM"`` for spaclone's full locus order).
    isotype_key
        Column with the isotype or constant-gene call of a units table / AnnData (default: ``c_key`` for records,
        else ``"isotype_subclass"``, ``"isotype"`` or ``c_key``, whichever exists).
    isotype_source
        ``"receptor"`` (default): isotypes from the receptor calls. ``"transcripts"``: from the productive (``_P``)
        IGHC molecules of ``sterile`` (sciCSR's ``c_gene_anno_name = NULL`` path; ``mode`` picks the most 3' or the
        most abundant productive isotype), for transcriptome-only data with IGHC counts from a BAM.
    class_only
        A call naming only the class (``"IGHG"``): ``"lower"`` gives it the class's 5'-most position (a lower bound
        of the switch); ``"missing"`` treats it as un-isotyped (sciCSR's string matching).
    weight
        ``mode="highest"``: count records or sum ``reads_key``.
    sterile_norm
        Sterile total ``s``: ``"seurat"`` (sciCSR's default ``scale.data``: LogNormalize to 1e4 over the IGHC
        features, regression on ``library_size``, z-score, clip at 10, summed over the sterile features; z-scores
        below the mean also raise ``|s|``, as in sciCSR), ``"log1p"`` (``log1p`` of the sterile count total),
        ``"counts"``, or ``"precomputed"``: ``sterile`` already holds normalised values (e.g. the ``scale.data`` an
        object carries; sciCSR uses an existing ``scale.data`` as it is), used as given for ``s`` and the NNLS.
    library_size
        Per-unit library size regressed out by ``"seurat"`` (sciCSR: ``nCount_RNA``): a column name, values, None,
        or ``"auto"`` (``nCount_RNA`` or ``total_counts`` of the table / ``obs`` when present, else no regression).
    fallback
        Isotype of units that cannot be imputed (no isotyped neighbour): ``"IgM"`` (sciCSR) or None (NaN).
    max_distance_um
        Spatial imputation ignores neighbours farther than this.
    connectivities_key, loci, c_key, locus_key, productive_key, reads_key, spatial_key, x_key, y_key
        Column and key names; only productive (or unknown) heavy-chain records give isotypes.

    Returns
    -------
    pandas.DataFrame
        One row per unit (index = unit id): ``x_um`` / ``y_um`` (when known), ``n_heavy``, ``isotype`` (name in
        :data:`CSR_ISOTYPES`), ``representative_p`` (its position), ``imputed``, ``total_s``, ``csr_pot_raw``
        (``sqrt(p^2 + s^2)``, or the naive NNLS weight for ``reference``) and ``csr_pot`` in [0, 1];
        ``naive_weight`` with ``reference``. ``attrs["spaclone"]["csr_potential"]`` holds the settings.

    Notes
    -----
    Differences from sciCSR 0.3.5, by design: (1) imputation votes with the neighbours' isotypes; sciCSR's
    ``getIsotype`` (the count-matrix path, ``c_gene_anno_name = NULL``) passes the positions of a neighbour's
    expressed isotypes to its summary function, so each neighbour votes (number of expressed isotypes - 1); the
    annotation path used in the vignette votes correctly and is reproduced exactly; (2) mouse IgG2a calls take the
    IgG2c position (sciCSR's string match leaves them un-isotyped); (3) class-only calls get the class's 5'-most
    position unless ``class_only="missing"``.

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Lawson & Hanson 1995, doi:10.1137/1.9781611971217.

    Examples
    --------
    >>> pot = csr_potential(air, unit_key="spot_id", impute="spatial")  # doctest: +SKIP
    >>> pot.groupby("isotype").csr_pot.mean()  # doctest: +SKIP
    """
    order, pos_map = _species_order(species, isotypes)
    if mode not in ("furthest", "highest"):
        raise ValueError(f"mode must be 'furthest' or 'highest', got {mode!r}")
    if sterile_norm not in ("seurat", "log1p", "counts", "precomputed"):
        raise ValueError(f"sterile_norm must be 'seurat', 'log1p', 'counts' or 'precomputed', got {sterile_norm!r}")
    if class_only not in ("lower", "missing"):
        raise ValueError(f"class_only must be 'lower' or 'missing', got {class_only!r}")
    if isotype_source not in ("receptor", "transcripts"):
        raise ValueError(f"isotype_source must be 'receptor' or 'transcripts', got {isotype_source!r}")
    from_tx = isotype_source == "transcripts"
    if from_tx and sterile is None:
        raise ValueError("isotype_source='transcripts' needs sterile= (IGHC counts with productive '_P' columns)")
    is_ad = hasattr(units, "obs") and hasattr(units, "obsm")
    if is_ad:
        tab = pd.DataFrame(index=pd.Index(units.obs_names.astype(str), name="unit"))
        if spatial_key in units.obsm:
            xy = np.asarray(units.obsm[spatial_key], dtype=float)
            tab[S.X], tab[S.Y] = xy[:, 0], xy[:, 1]
        if from_tx:
            p = np.full(units.n_obs, np.nan)
        else:
            cols = units.obs.columns
            key = isotype_key or next((c for c in ("isotype_subclass", "isotype", c_key) if c in cols), None)
            if key is None or key not in cols:
                raise KeyError("csr_potential: pass isotype_key= (an obs column with the isotype or c_call)")
            p = _positions(
                units.obs[key].reset_index(drop=True), species=species, pos=pos_map, order=order, class_only=class_only
            )
        tab["n_heavy"] = np.isfinite(p).astype(np.int64)
        tab["position_observed"] = p
        side = units.obs
    elif isinstance(units, pd.DataFrame):
        if from_tx:
            p_all = np.full(len(units), np.nan)
        else:
            records = isotype_key is None and c_key in units.columns
            cols = units.columns
            key = (
                c_key
                if records
                else (isotype_key or next((c for c in ("isotype_subclass", "isotype", c_key) if c in cols), None))
            )
            if key is None or key not in units.columns:
                raise KeyError("csr_potential: the table needs a constant-gene call (c_call) or pass isotype_key=")
            p_all = _positions(
                units[key].reset_index(drop=True), species=species, pos=pos_map, order=order, class_only=class_only
            )
        tab = _records_to_units(
            units,
            unit_key=unit_key,
            pos=p_all,
            weight=weight,
            mode=mode,
            locus_key=locus_key,
            loci=loci,
            productive_key=productive_key,
            reads_key=reads_key,
            x_key=x_key,
            y_key=y_key,
        )
        side = None
        col = _unit_column(units, unit_key)
        if col is None:
            side = units.set_axis(units.index.astype(str))
        else:
            side = units.assign(_u=units[col].astype(str)).groupby("_u", sort=False).first()
    else:
        raise TypeError(f"csr_potential: expected records / units DataFrame or AnnData, got {type(units).__name__}")
    ids = tab.index
    n = len(ids)
    if n == 0:
        raise ValueError("csr_potential: no units (no records of the requested loci)")
    if from_tx:
        tab["position_observed"], tab["n_heavy"] = _productive_positions(
            _ighc_frame(sterile, ids, species, pos_map), pos_map, mode
        )

    # ---------------------------------------------------------------- imputation
    adj = None
    if isinstance(impute, str):
        if impute == "spatial":
            if S.X not in tab.columns:
                raise KeyError("impute='spatial' needs coordinates (x_um / y_um or obsm['spatial'])")
            adj = _knn_adjacency(tab[[S.X, S.Y]].to_numpy(dtype=float), k, max_distance_um)
        elif impute == "expression":
            src = units if is_ad else gex
            if src is None:
                raise ValueError("impute='expression' needs an AnnData (units or gex=)")
            if src is not units:
                missing = ids.difference(pd.Index(src.obs_names.astype(str)))
                if len(missing):
                    raise KeyError(f"{len(missing)} units not in gex.obs_names (e.g. {missing[0]!r})")
                src = src[np.asarray(ids)]
            from spaclone.tl.transitions import _expression_connectivities

            conn = _expression_connectivities(
                src, k=k + 1, use_rep=None, n_pcs=None, connectivities_key=connectivities_key, rng=0
            )
            adj = ((conn + conn.T) > 0).astype(float).tocsr()
        else:
            raise ValueError(f"impute must be 'spatial', 'expression', a matrix or None, got {impute!r}")
    elif impute is not None:
        adj = sparse.csr_matrix(impute, dtype=float)
        if adj.shape != (n, n):
            raise ValueError(f"impute: expected an {n} x {n} adjacency, got {adj.shape}")
        adj = ((adj + adj.T) > 0).astype(float).tocsr()
        adj.setdiag(0.0)
        adj.eliminate_zeros()
    fb = None if fallback is None else 0.0
    if fallback not in (None, "IgM"):
        raise ValueError(f"fallback must be 'IgM' or None, got {fallback!r}")
    p_obs = tab["position_observed"].to_numpy(dtype=float)
    if adj is not None:
        rep, imputed = _impute(p_obs, adj, fb)
    else:
        rep = np.where(np.isfinite(p_obs), p_obs, np.nan if fb is None else fb)
        imputed = ~np.isfinite(p_obs) & np.isfinite(rep)

    # ---------------------------------------------------------------- sterile / IGHC
    total_s = np.zeros(n)
    scaled = feats = None
    if sterile is not None:
        ighc = _ighc_frame(sterile, ids, species, pos_map)
        feats = list(ighc.columns)
        s_cols = [c for c in feats if c.endswith("_S")]
        cnt = ighc.to_numpy(dtype=float)
        lib = None
        if sterile_norm == "precomputed":
            scaled = cnt
        elif sterile_norm == "seurat" or reference is not None:
            lib = _library_size(library_size, side, ids, is_ad)
            scaled = _seurat_scale(cnt, lib)
        if s_cols:
            si = [feats.index(c) for c in s_cols]
            if sterile_norm in ("seurat", "precomputed"):
                total_s = scaled[:, si].sum(axis=1)
            elif sterile_norm == "log1p":
                total_s = np.log1p(cnt[:, si].sum(axis=1))
            else:
                total_s = cnt[:, si].sum(axis=1)
    elif reference is not None:
        raise ValueError("reference-based CSR potential needs sterile= (productive and sterile IGHC counts per unit)")

    out = tab.drop(columns=["position_observed"]).copy()
    names = np.array(order, dtype=object)
    out["isotype"] = pd.array(
        [names[int(v)] if np.isfinite(v) and int(v) < len(names) else None for v in rep], dtype="string"
    )
    out["representative_p"] = rep
    out["imputed"] = imputed
    out["total_s"] = total_s
    naive = None
    if reference is None:
        raw = np.sqrt(rep**2 + total_s**2)
        pot = _minmax(raw)
        pot = np.where(np.isfinite(pot), pot, 0.0) if np.isfinite(raw).any() else pot
        pot = np.where(np.isfinite(raw), pot, np.nan)
    else:
        sig = _signatures(reference, species, pos_map)
        common = [f for f in sig.index if f in feats]
        if len(common) < 2:
            raise ValueError(f"reference: only {len(common)} of the signature's features are in sterile= ({feats})")
        from scipy.optimize import nnls

        w = sig.loc[common].to_numpy(dtype=float)
        y = scaled[:, [feats.index(f) for f in common]]
        naive = np.array([nnls(w, y[i])[0][0] for i in range(n)])
        raw = naive
        pot = 1.0 - _minmax(naive)
    out["csr_pot_raw"] = raw
    out["csr_pot"] = pot
    if naive is not None:
        out["naive_weight"] = naive
    out.attrs["spaclone"] = {
        "csr_potential": {
            "species": species,
            "isotypes": list(order),
            "mode": mode,
            "reference": reference if isinstance(reference, str) or reference is None else "<DataFrame>",
            "impute": impute if isinstance(impute, str) or impute is None else "<adjacency>",
            "k": int(k),
            "sterile": sterile is not None,
            "sterile_norm": sterile_norm,
            "class_only": class_only,
            "fallback": fallback,
            "n_imputed": int(imputed.sum()),
        }
    }
    return out


def _library_size(library_size: Any, side: Any, ids: pd.Index, is_ad: bool) -> np.ndarray | None:
    if library_size is None:
        return None
    if isinstance(library_size, str):
        cols = side.columns if side is not None else []
        if library_size == "auto":
            key = next((c for c in ("nCount_RNA", "total_counts") if c in cols), None)
            if key is None:
                return None
        else:
            key = library_size
            if key not in cols:
                raise KeyError(f"library_size column {key!r} not found")
        vals = side[key]
        vals.index = vals.index.astype(str)
        return pd.to_numeric(vals.reindex(ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vals = np.asarray(library_size, dtype=float)
    if len(vals) != len(ids):
        raise ValueError(f"library_size: {len(vals)} values for {len(ids)} units")
    return vals


def _signatures(reference: Any, species: str, pos: dict[str, int]) -> pd.DataFrame:
    if isinstance(reference, str):
        if reference not in ("mouse", "human"):
            raise ValueError(f"reference must be 'mouse', 'human', a DataFrame or None, got {reference!r}")
        from spaclone.ext.scicsr import nmf_signatures

        sig = nmf_signatures(reference)
    elif isinstance(reference, pd.DataFrame):
        sig = reference
    else:
        raise TypeError("reference must be 'mouse', 'human', a DataFrame or None")
    rows = []
    for r in sig.index:
        iso, kind = _parse_feature(r, species, pos)
        rows.append(f"{iso}_{kind or 'S'}" if iso is not None else None)
    out = sig.copy()
    out.index = rows
    return out[[r is not None for r in rows]]


# ===================================================================================================== SHM score
def shm_score(
    units: pd.DataFrame,
    *,
    method: Literal["identity", "excess"] = "identity",
    unit_key: str | None = "auto",
    agg: Literal["mean", "median", "max", "min", "top_reads"] = "mean",
    fill: float | None = 0.0,
    locus: str | None = "IGH",
    v_identity_key: str = "v_identity",
    excess_key: str = "shm_excess",
    evaluable_key: str = "shm_evaluable",
    locus_key: str = S.LOCUS,
    reads_key: str = S.READS,
    x_key: str = S.X,
    y_key: str = S.Y,
    **shm_kwargs: Any,
) -> pd.DataFrame:
    """SHM score (sciCSR) per unit: min-max scaled ``1 - V identity``, or the same scaling of allele-corrected SHM.

    Parameters
    ----------
    units
        Receptor records with ``v_identity`` (AIRR: a fraction; IMGT / StereoIR percentages are detected and
        divided by 100) or, for ``method="excess"``, the output of :func:`spaclone.bcr.shm` (computed here with
        ``shm_kwargs`` when ``excess_key`` is missing).
    method
        ``"identity"``: sciCSR's ``getSHM`` (``1 - v_identity``). ``"excess"``: spaclone's V mismatches in excess of
        the allele baseline (evaluable records only), which removes germline allele differences and low-read errors.
    unit_key
        Column grouping records into units (``"auto"``: ``locus_id``, ``spot_id`` or ``cell_id``; None: rows).
    agg
        Summary over a unit's records: ``"mean"``, ``"median"``, ``"max"`` / ``"min"`` of the raw score, or the record
        with most reads (``"top_reads"``; sciCSR keeps one heavy chain per cell).
    fill
        Score of units without a usable record (sciCSR: 0); None keeps NaN.
    locus
        Chain used (sciCSR: the heavy chain); None uses every record.
    v_identity_key, excess_key, evaluable_key, locus_key, reads_key, x_key, y_key
        Column names.
    **shm_kwargs
        Passed to :func:`spaclone.bcr.shm` for ``method="excess"``.

    Returns
    -------
    pandas.DataFrame
        One row per unit: ``x_um`` / ``y_um`` when known, ``n_records``, ``shm_raw`` (unscaled) and ``shm`` in [0, 1].

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Gadala-Maria et al. 2015, doi:10.1073/pnas.1417683112.
    """
    if not isinstance(units, pd.DataFrame):
        raise TypeError("shm_score expects a DataFrame of receptor records")
    air = units
    if method == "excess" and excess_key not in air.columns:
        from spaclone.bcr.shm import shm as _shm

        air = _shm(air, **shm_kwargs)
    col = _unit_column(air, unit_key)
    ids = air.index.astype(str).to_numpy() if col is None else air[col].astype(str).to_numpy()
    inv, uid = pd.factorize(pd.Series(ids, dtype=object), sort=False)
    inv = np.asarray(inv, dtype=np.int64)
    n = len(uid)
    out = pd.DataFrame(index=pd.Index(np.asarray(uid, dtype=object).astype(str), name="unit"))
    if x_key in air.columns and y_key in air.columns:
        cnt = np.bincount(inv, minlength=n)
        out[S.X] = np.bincount(inv, weights=air[x_key].to_numpy(dtype=float), minlength=n) / cnt
        out[S.Y] = np.bincount(inv, weights=air[y_key].to_numpy(dtype=float), minlength=n) / cnt
    use = np.ones(len(air), dtype=bool)
    if locus is not None and locus_key in air.columns:
        use &= (air[locus_key].astype("string") == locus).fillna(False).to_numpy(dtype=bool)
    if method == "identity":
        if v_identity_key not in air.columns:
            raise KeyError(f"shm_score: column {v_identity_key!r} not in the table")
        v = pd.to_numeric(air[v_identity_key], errors="coerce").to_numpy(dtype=float)
        if np.nanmax(np.where(use, v, np.nan), initial=-np.inf) > 1.0 + 1e-9:
            v = v / 100.0  # percent identity (IMGT / StereoIR)
        raw = 1.0 - v
    elif method == "excess":
        raw = pd.to_numeric(air[excess_key], errors="coerce").to_numpy(dtype=float)
        if evaluable_key in air.columns:
            use &= air[evaluable_key].astype("boolean").fillna(False).to_numpy(dtype=bool)
    else:
        raise ValueError(f"method must be 'identity' or 'excess', got {method!r}")
    use &= np.isfinite(raw)
    u, r = inv[use], raw[use]
    out["n_records"] = np.bincount(u, minlength=n).astype(np.int64)
    val = np.full(n, np.nan)
    if len(r):
        d = pd.DataFrame({"u": u, "r": r})
        if agg in ("mean", "median", "max", "min"):
            s = d.groupby("u").r.agg(agg)
        elif agg == "top_reads":
            if reads_key not in air.columns:
                raise KeyError(f"agg='top_reads' needs the reads column {reads_key!r}")
            d["w"] = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)[use]
            s = d.sort_values(["u", "w"], ascending=[True, False], kind="stable").drop_duplicates("u").set_index("u").r
        else:
            raise ValueError(f"agg must be 'mean', 'median', 'max', 'min' or 'top_reads', got {agg!r}")
        val[s.index.to_numpy(dtype=np.int64)] = s.to_numpy(dtype=float)
    out["shm_raw"] = val
    sc = _minmax(val)
    out["shm"] = np.where(np.isfinite(sc), sc, np.nan if fill is None else float(fill))
    out.attrs["spaclone"] = {"shm_score": {"method": method, "agg": agg, "fill": fill, "locus": locus}}
    return out


# ===================================================================================================== definitions
def ighc_definitions(
    species: Literal["mouse", "human"] = "mouse",
    *,
    assembly: str | None = None,
    flank_bp: int = 10_000,
    isotypes: Sequence[str] | None = None,
    chrom_prefix: str = "",
) -> pd.DataFrame:
    """Genomic regions used to classify IGHC reads: the V(D)J span, each constant gene and its sterile region.

    Parameters
    ----------
    species
        ``"mouse"`` (default assembly GRCm39; GRCm38 / mm10 also built in) or ``"human"`` (GRCh38).
    assembly
        ``"GRCm39"``, ``"GRCm38"`` or ``"GRCh38"`` (default: the species' current assembly).
    flank_bp
        Length of each sterile region: the ``flank_bp`` nucleotides 5' of the constant gene (towards the V(D)J
        genes), cut at the 3' end of the next constant gene upstream (or, for Cμ, at the J genes). This span holds
        the switch region and the I exon from which germline (sterile) transcripts start; switch regions are
        1-10 kb long, so the default (10 kb) reaches every mouse and human I exon. sciCSR 0.3.5 ships curated
        sterile ranges (4.3-8.9 kb in mm10); load them with :func:`spaclone.ext.scicsr.scicsr_definitions` to
        reproduce its counts exactly.
    isotypes
        Constant genes to include (default :data:`CSR_ISOTYPES`, which leaves out Cδ: it has no switch region and
        no sterile transcript of its own).
    chrom_prefix
        Prefix of the chromosome name in the BAM (``"chr"`` for UCSC-style references); :func:`ighc_transcripts`
        also matches the other style automatically.

    Returns
    -------
    pandas.DataFrame
        Rows ``VDJ``, then for each gene a ``C`` and a ``sterile`` row: ``name`` (isotype), ``kind``, ``gene``
        (Ensembl name), ``chrom``, ``start``, ``end`` (1-based, inclusive), ``strand``.
    """
    if species not in CSR_ISOTYPES:
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    assembly = assembly or {"mouse": "GRCm39", "human": "GRCh38"}[species]
    if assembly not in IGH_LOCI or IGH_LOCI[assembly]["species"] != species:
        ok = [a for a, v in IGH_LOCI.items() if v["species"] == species]
        raise ValueError(f"assembly {assembly!r} is not built in for {species} (choose from {ok})")
    loc = IGH_LOCI[assembly]
    iso = tuple(isotypes) if isotypes is not None else CSR_ISOTYPES[species]
    unknown = [i for i in iso if i not in loc["C"]]
    if unknown:
        raise ValueError(f"no coordinates for {unknown} in {assembly} (have {list(loc['C'])})")
    if flank_bp < 1:
        raise ValueError(f"flank_bp must be >= 1, got {flank_bp}")
    chrom = chrom_prefix + loc["chrom"]
    vdj_s, vdj_e = loc["VDJ"]
    rows = [
        {"name": "VDJ", "kind": "VDJ", "gene": "V(D)J", "chrom": chrom, "start": vdj_s, "end": vdj_e, "strand": "-"}
    ]
    # minus strand: 5' of a gene = its larger coordinate; upstream neighbours have larger coordinates
    genes = sorted(((i, *loc["C"][i]) for i in loc["C"]), key=lambda r: -r[2])
    upstream_start = {g[0]: (genes[j - 1][1] if j > 0 else vdj_s) for j, g in enumerate(genes)}
    for i in iso:
        s, e = loc["C"][i]
        cap = upstream_start[i] - 1
        st_s, st_e = e + 1, min(e + int(flank_bp), cap)
        gene = _GENE_NAMES[species].get(i, i)
        rows.append({"name": i, "kind": "C", "gene": gene, "chrom": chrom, "start": s, "end": e, "strand": "-"})
        rows.append(
            {"name": i, "kind": "sterile", "gene": gene, "chrom": chrom, "start": st_s, "end": st_e, "strand": "-"}
        )
    out = pd.DataFrame(rows)
    out.attrs["spaclone"] = {
        "ighc_definitions": {
            "species": species,
            "assembly": assembly,
            "flank_bp": int(flank_bp),
            "source": loc["source"],
        }
    }
    return out


def _check_definitions(defs: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    need = {"name", "kind", "chrom", "start", "end"}
    if not isinstance(defs, pd.DataFrame) or not need <= set(defs.columns):
        raise ValueError(f"definitions must be a DataFrame with columns {sorted(need)} (see ighc_definitions)")
    kinds = set(defs["kind"])
    if not {"VDJ", "C", "sterile"} <= kinds:
        raise ValueError("definitions need 'VDJ', 'C' and 'sterile' rows")
    genes = list(dict.fromkeys(defs.loc[defs.kind == "C", "name"].astype(str)))
    st = set(defs.loc[defs.kind == "sterile", "name"].astype(str))
    if set(genes) != st:
        raise ValueError("definitions: every constant gene needs one 'C' and one 'sterile' row")
    return genes, {}


# ===================================================================================================== BAM reading
_CIGAR_REF = {0: True, 2: True, 3: True, 7: True, 8: True}  # M D N = X consume the reference
_TAG_SIZE = {"c": 1, "C": 1, "s": 2, "S": 2, "i": 4, "I": 4, "f": 4, "A": 1}
_TAG_FMT = {"c": "<b", "C": "<B", "s": "<h", "S": "<H", "i": "<i", "I": "<I", "f": "<f"}


def _blocks(pos0: int, cigar: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reference blocks (1-based inclusive) of an alignment: runs of M / = / X / D, split at N (GAlignments rglist)."""
    out = []
    cur = pos0
    start = None
    for op, ln in cigar:
        if op in (0, 2, 7, 8):  # M D = X
            if start is None:
                start = cur
            cur += ln
        elif op == 3:  # N: splice gap
            if start is not None:
                out.append((start + 1, cur))
                start = None
            cur += ln
    if start is not None:
        out.append((start + 1, cur))
    return out


def _parse_tags(buf: bytes, i: int, end: int, wanted: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    while i < end and len(out) < len(wanted):
        tag = buf[i : i + 2].decode("ascii")
        typ = chr(buf[i + 2])
        i += 3
        if typ in ("Z", "H"):
            j = buf.index(b"\x00", i)
            val: Any = buf[i:j].decode("ascii", "replace")
            i = j + 1
        elif typ == "A":
            val = chr(buf[i])
            i += 1
        elif typ in _TAG_FMT:
            val = struct.unpack_from(_TAG_FMT[typ], buf, i)[0]
            i += _TAG_SIZE[typ]
        elif typ == "B":
            sub = chr(buf[i])
            cnt = struct.unpack_from("<i", buf, i + 1)[0]
            i += 5 + cnt * _TAG_SIZE[sub]
            val = None
        else:
            raise ValueError(f"unknown BAM tag type {typ!r}")
        if tag in wanted:
            out[tag] = val
    return out


def _read_bam_python(path: Path, chroms: set[str], span: tuple[int, int], tags: set[str], paired: bool):
    """Stream a BAM (BGZF = concatenated gzip members) and yield alignments overlapping the locus span."""
    with gzip.open(path, "rb") as f:
        if f.read(4) != b"BAM\x01":
            raise ValueError(f"{path} is not a BAM file")
        (l_text,) = struct.unpack("<i", f.read(4))
        f.read(l_text)
        (n_ref,) = struct.unpack("<i", f.read(4))
        refs = []
        for _ in range(n_ref):
            (l_name,) = struct.unpack("<i", f.read(4))
            refs.append(f.read(l_name)[:-1].decode("ascii"))
            f.read(4)
        targets = {i for i, r in enumerate(refs) if r in chroms}
        lo, hi = span
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (size,) = struct.unpack("<i", head)
            rec = f.read(size)
            ref_id, pos0, l_name, _mapq, _bin, n_cig, flag, l_seq = struct.unpack_from("<iiBBHHHI", rec, 0)
            if ref_id not in targets or flag & 0x4:
                continue
            if paired and not (flag & 0x1 and flag & 0x2 and flag & 0x40):
                continue
            if pos0 + 1 > hi:
                continue
            o = 32
            qname = rec[o : o + l_name - 1].decode("ascii", "replace")
            o += l_name
            cig = struct.unpack_from(f"<{n_cig}I", rec, o)
            o += 4 * n_cig
            cigar = [(c & 0xF, c >> 4) for c in cig]
            blocks = _blocks(pos0, cigar)
            if not blocks or blocks[-1][1] < lo:
                continue
            o += (l_seq + 1) // 2 + l_seq
            tg = _parse_tags(rec, o, len(rec), tags) if tags else {}
            yield refs[ref_id], qname, flag, blocks, tg


def _read_bam_pysam(path: Path, regions: pd.DataFrame, tags: set[str], paired: bool, chrom_map: dict[str, str]):
    import pysam

    seen: set[tuple[str, int, int, str]] = set()
    with pysam.AlignmentFile(str(path), "rb") as f:
        for chrom, s, e in regions[["chrom", "start", "end"]].itertuples(index=False):
            contig = chrom_map.get(chrom, chrom)
            for r in f.fetch(contig, int(s) - 1, int(e)):
                if r.is_unmapped or r.cigartuples is None:
                    continue
                if paired and not (r.is_paired and r.is_proper_pair and r.is_read1):
                    continue
                key = (r.query_name, r.reference_start, r.flag, r.cigarstring)
                if key in seen:
                    continue
                seen.add(key)
                tg = {t: r.get_tag(t) for t in tags if r.has_tag(t)}
                yield contig, r.query_name, r.flag, _blocks(r.reference_start, r.cigartuples), tg


def _classify(mol: pd.DataFrame, genes: list[str], cls_of: dict[str, str]) -> pd.Series:
    """Classify molecules as sciCSR's getIGHreadType does.

    Productive: V(D)J reads plus C or I reads of one isotype; sterile: I reads, no V(D)J; C-only: C reads of one
    isotype without I or V(D)J reads; conflicting or tied isotypes are left out.
    """
    c = mol[[f"{g}_C" for g in genes]].to_numpy(dtype=float)
    s = mol[[f"{g}_I" for g in genes]].to_numpy(dtype=float)
    vdj = mol["VDJ"].to_numpy(dtype=float) > 0
    cls = np.array([cls_of[g] for g in genes], dtype=object)
    n = len(mol)
    out = np.full(n, None, dtype=object)
    for r in range(n):
        cr, sr = c[r], s[r]
        if len(set(cls[cr > 0])) > 1 or len(set(cls[sr > 0])) > 1:
            continue
        cmax = np.flatnonzero(cr == cr.max()) if cr.max() > 0 else np.zeros(0, dtype=int)
        if not (sr > 0).any():
            c_iso = np.flatnonzero(cr == cr.max())
            if len(c_iso) == 1:
                out[r] = f"{genes[c_iso[0]]}_{'P' if vdj[r] else 'C'}"
            continue
        i_iso = np.flatnonzero(sr == sr.max())
        if not (cr > 0).any():
            if len(i_iso) == 1:
                out[r] = f"{genes[i_iso[0]]}_{'P' if vdj[r] else 'S'}"
            continue
        if len(i_iso) == 1:
            if i_iso[0] in set(cmax.tolist()):
                out[r] = f"{genes[i_iso[0]]}_{'P' if vdj[r] else 'S'}"
        elif len(cmax) == 1 and cmax[0] in set(i_iso.tolist()):
            out[r] = f"{genes[cmax[0]]}_{'P' if vdj[r] else 'S'}"
    return pd.Series(out, index=mol.index, dtype=object)


def ighc_transcripts(
    bam: str | Path | Sequence[str | Path],
    *,
    species: Literal["mouse", "human"] = "mouse",
    assembly: str | None = None,
    definitions: pd.DataFrame | None = None,
    barcode_tag: str | None = "CB",
    umi_tag: str | None = "UB",
    spots: pd.DataFrame | Mapping[str, str] | None = None,
    x_tag: str | None = None,
    y_tag: str | None = None,
    bin_size: int | None = None,
    scale_um: float = S.DNB_UM,
    paired: bool = False,
    flank_bp: int = 10_000,
    backend: Literal["auto", "pysam", "python"] = "auto",
    return_molecules: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Productive, sterile and C-only IGHC molecules per barcode, spot or bin, counted from BAM files.

    Every molecule (barcode + UMI) is classified from the reads that fall in the V(D)J span, in each constant gene's
    coding region (C) and in its sterile region (I), as in sciCSR's ``getIGHreadType``: reads of more than one
    isotype class in the C or in the I regions leave the molecule unclassified; otherwise the isotype is the C (or I)
    region with most reads, and the molecule is productive (``P``) when a read reaches the V(D)J span, sterile
    (``S``) when it has I reads but no V(D)J read, and C-only (``C``, uninformative) when it has only C reads (ties
    and disagreeing C / I isotypes are left out). A read counts in a region when one of its aligned blocks (runs of
    M / D / = / X, split at N) overlaps it; reads are counted once per molecule and region (unique read names);
    secondary and duplicate-flagged alignments are kept, as sciCSR does.

    Parameters
    ----------
    bam
        Coordinate-sorted BAM file(s) (``pysam`` needs the ``.bai`` index; the pure-Python reader streams the file).
        Several files are read in turn; prefix their barcodes through ``spots`` if they overlap.
    species, assembly, flank_bp
        Built-in genomic definitions (:func:`ighc_definitions`), used when ``definitions`` is None.
    definitions
        Custom regions (columns ``name``, ``kind`` in VDJ / C / sterile, ``chrom``, ``start``, ``end``, 1-based
        inclusive), e.g. sciCSR's curated ranges from :func:`spaclone.ext.scicsr.scicsr_definitions`.
    barcode_tag, umi_tag
        Tags of the cell / spot barcode and the UMI (10x / Visium: ``CB`` / ``UB``; Slide-seq / Curio: often ``XC`` /
        ``XM``; Stereo-seq SAW: no barcode tag, ``x_tag="Cx"``, ``y_tag="Cy"``, ``umi_tag="UR"``). Reads lacking a
        tag are skipped. ``barcode_tag=None`` without coordinate tags puts every read of a file in one unit named
        after the file; ``umi_tag=None`` makes every unit one molecule (sciCSR's behaviour).
    spots
        Barcode -> spot mapping: a Series / dict (barcode -> spot id) or a DataFrame indexed by barcode with
        ``x_um`` / ``y_um`` (and optionally ``spot_id``); barcodes absent from it are dropped. Without it the unit
        is the barcode.
    x_tag, y_tag, bin_size, scale_um
        Coordinate tags (Stereo-seq ``Cx`` / ``Cy``): the unit is the bin of ``bin_size`` coordinate units
        (``"x-y"`` of the bin's lower corner, as StereoIR names bins) and ``x_um`` / ``y_um`` = corner x
        ``scale_um`` (0.5 µm per DNB).
    paired
        Paired-end data: keep only the first mate of properly paired reads.
    backend
        ``"pysam"`` (indexed region queries; fast), ``"python"`` (built-in streaming reader; no dependency, slower
        for whole-transcriptome BAMs) or ``"auto"`` (pysam when installed).
    return_molecules
        Also return the molecule table (unit, UMI, reads per region, class).

    Returns
    -------
    pandas.DataFrame or (DataFrame, DataFrame)
        Counts per unit (index = barcode / spot / bin) with columns ``{isotype}_S``, ``{isotype}_P``, ``{isotype}_C``
        in the definitions' gene order (the input of :func:`csr_potential`'s ``sterile=``), plus ``x_um`` / ``y_um``
        when known; ``attrs["spaclone"]["ighc_transcripts"]`` records molecules read and classified.

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Li et al. 2009, doi:10.1093/bioinformatics/btp352.

    Examples
    --------
    >>> counts = ighc_transcripts("possorted_genome_bam.bam", species="mouse", assembly="GRCm38")  # doctest: +SKIP
    >>> pot = csr_potential(air, sterile=counts, unit_key="cell_id")  # doctest: +SKIP
    """
    defs = ighc_definitions(species, assembly=assembly, flank_bp=flank_bp) if definitions is None else definitions
    genes, _ = _check_definitions(defs)
    defs = defs.copy()
    defs["chrom"] = defs["chrom"].astype(str)
    files = [Path(bam)] if isinstance(bam, str | Path) else [Path(b) for b in bam]
    if backend == "auto":
        try:
            import pysam

            backend = "pysam"
        except ImportError:
            backend = "python"
    if backend not in ("pysam", "python"):
        raise ValueError(f"backend must be 'auto', 'pysam' or 'python', got {backend!r}")
    if (x_tag is None) != (y_tag is None):
        raise ValueError("pass both x_tag and y_tag, or neither")
    tags = {t for t in (barcode_tag, umi_tag, x_tag, y_tag) if t}
    regions = defs[["name", "kind", "chrom", "start", "end"]].reset_index(drop=True)
    reg_type = np.where(
        regions.kind == "VDJ", "VDJ", regions.name.astype(str) + np.where(regions.kind == "C", "_C", "_I")
    )
    r_start = regions.start.to_numpy(dtype=np.int64)
    r_end = regions.end.to_numpy(dtype=np.int64)
    span = (int(r_start.min()), int(r_end.max()))
    hits: list[tuple[str, str, str, str]] = []  # (unit, umi, qname, region type)
    n_reads = 0
    r_chrom = regions.chrom.astype(str).to_numpy()
    plain = np.array([c[3:] if c.startswith("chr") else c for c in r_chrom], dtype=object)
    contig_mask: dict[str, np.ndarray] = {}
    for path in files:
        chroms = set(regions.chrom)
        alt = {c[3:] if c.startswith("chr") else "chr" + c for c in chroms}
        if backend == "pysam":
            import pysam

            with pysam.AlignmentFile(str(path), "rb") as fh:
                refs = set(fh.references)
            cmap = {c: (c if c in refs else (c[3:] if c.startswith("chr") else "chr" + c)) for c in chroms}
            reads = _read_bam_pysam(path, regions, tags, paired, cmap)
        else:
            reads = _read_bam_python(path, chroms | alt, span, tags, paired)
        for contig, qname, _flag, blocks, tg in reads:
            n_reads += 1
            if barcode_tag is None and x_tag is None:
                unit = path.name
            elif x_tag is not None:
                if x_tag not in tg or y_tag not in tg:
                    continue
                if bin_size:
                    bx, by = int(tg[x_tag]) // bin_size * bin_size, int(tg[y_tag]) // bin_size * bin_size
                else:
                    bx, by = int(tg[x_tag]), int(tg[y_tag])
                unit = f"{bx}-{by}"
            else:
                if barcode_tag not in tg:
                    continue
                unit = str(tg[barcode_tag])
            if umi_tag is None:
                umi = "None"
            elif umi_tag in tg:
                umi = str(tg[umi_tag])
            else:
                continue
            bs = np.array([b[0] for b in blocks])
            be = np.array([b[1] for b in blocks])
            cm = contig_mask.get(contig)
            if cm is None:
                bare = contig[3:] if contig.startswith("chr") else contig
                cm = contig_mask[contig] = plain == bare
            ov = ((bs[:, None] <= r_end[None, :]) & (be[:, None] >= r_start[None, :])).any(axis=0) & cm
            for j in np.flatnonzero(ov):
                hits.append((unit, umi, qname, reg_type[j]))
    cols = [f"{g}_C" for g in genes] + [f"{g}_I" for g in genes] + ["VDJ"]
    if hits:
        h = pd.DataFrame(hits, columns=["unit", "umi", "qname", "type"]).drop_duplicates()
        mol = h.groupby(["unit", "umi", "type"]).qname.nunique().unstack("type", fill_value=0)
        mol = mol.reindex(columns=cols, fill_value=0)
    else:
        mol = pd.DataFrame(columns=cols, index=pd.MultiIndex.from_arrays([[], []], names=["unit", "umi"]))
    cls_of = {}
    for g in genes:
        parsed = _isotype(pd.Series([g]), species=None)["isotype"].iat[0]
        cls_of[g] = parsed if isinstance(parsed, str) and parsed != "unknown" else g
    anno = _classify(mol, genes, cls_of) if len(mol) else pd.Series([], dtype=object)
    feats = [f"{g}_{k}" for g in genes for k in ("S", "P", "C")]
    if len(mol):
        a = pd.DataFrame({"unit": mol.index.get_level_values(0), "anno": anno.to_numpy()}).dropna()
        counts = a.groupby(["unit", "anno"]).size().unstack("anno", fill_value=0)
        counts = counts.reindex(columns=feats, fill_value=0).astype(np.int64)
    else:
        counts = pd.DataFrame(columns=feats, dtype=np.int64)
    counts.index = counts.index.astype(str)
    counts.index.name = "unit"
    counts.columns.name = None
    if spots is not None:
        counts = _map_spots(counts, spots)
    elif x_tag is not None and len(counts):
        xy = np.array([list(map(float, u.split("-"))) for u in counts.index]) * scale_um
        counts.insert(0, S.Y, xy[:, 1])
        counts.insert(0, S.X, xy[:, 0])
    counts.attrs["spaclone"] = {
        "ighc_transcripts": {
            "files": [str(p) for p in files],
            "backend": backend,
            "reads_in_regions": int(n_reads),
            "molecules": int(len(mol)),
            "classified": int(anno.notna().sum()) if len(mol) else 0,
            "definitions": defs.attrs.get("spaclone", {}).get("ighc_definitions", "custom"),
        }
    }
    if return_molecules:
        mt = mol.reset_index()
        mt["class"] = anno.to_numpy() if len(mol) else []
        return counts, mt
    return counts


def _map_spots(counts: pd.DataFrame, spots: Any) -> pd.DataFrame:
    if isinstance(spots, Mapping):
        spots = pd.Series(spots)
    if isinstance(spots, pd.Series):
        sid = spots.astype(str)
        sid.index = sid.index.astype(str)
        keep = counts.index.intersection(sid.index)
        out = counts.loc[keep].groupby(sid.loc[keep].to_numpy()).sum()
        out.index.name = "unit"
        return out
    if isinstance(spots, pd.DataFrame):
        sp = spots.copy()
        sp.index = sp.index.astype(str)
        keep = counts.index.intersection(sp.index)
        sid = sp.loc[keep, S.SPOT].astype(str).to_numpy() if S.SPOT in sp.columns else keep.to_numpy()
        out = counts.loc[keep].groupby(sid).sum()
        out.index.name = "unit"
        if S.X in sp.columns and S.Y in sp.columns:
            first = sp.loc[keep].assign(_s=sid).groupby("_s")[[S.X, S.Y]].first()
            out.insert(0, S.Y, first[S.Y].reindex(out.index).to_numpy())
            out.insert(0, S.X, first[S.X].reindex(out.index).to_numpy())
        return out
    raise TypeError("spots must be a dict, a Series or a DataFrame indexed by barcode")


_IGH_COORDS: dict[str, dict[str, Any]] = {
    "GRCm39": {
        "species": "mouse",
        "chrom": "12",
        "C": {
            "IgM": (113382178, 113386350),
            "IgD": (113370224, 113379944),
            "IgG3": (113319844, 113324852),
            "IgG1": (113288860, 113294143),
            "IgG2b": (113266585, 113271553),
            "IgG2c": (113248945, 113252552),
            "IgE": (113232880, 113236868),
            "IgA": (113218450, 113223856),
        },
        "VDJ": (113392134, 115973713),
        "source": "Ensembl 115 (GRCm39) gene coordinates; VDJ = Ighj4 start .. last Ighv gene end",
    },
    "GRCm38": {
        "species": "mouse",
        "chrom": "12",
        "C": {
            "IgM": (113418558, 113422730),
            "IgD": (113406604, 113416324),
            "IgG3": (113356224, 113361232),
            "IgG1": (113325240, 113330523),
            "IgG2b": (113302965, 113307933),
            "IgG2c": (113285325, 113288932),
            "IgE": (113269260, 113273248),
            "IgA": (113254830, 113260236),
        },
        "VDJ": (113428514, 116010093),
        "source": "GRCm39 coordinates mapped to GRCm38 / mm10 by the Ensembl assembly map (one block, +36,380 bp)",
    },
    "GRCh38": {
        "species": "human",
        "chrom": "14",
        "C": {
            "IgM": (105851705, 105856218),
            "IgD": (105836765, 105845677),
            "IgG3": (105764503, 105771405),
            "IgG1": (105736343, 105743071),
            "IgA1": (105703995, 105708665),
            "IgG2": (105639559, 105644790),
            "IgG4": (105620506, 105626066),
            "IgE": (105597691, 105601728),
            "IgA2": (105583731, 105588395),
        },
        "VDJ": (105863198, 106879812),
        "source": "Ensembl 115 (GRCh38) gene coordinates; VDJ = IGHJ6 start .. last IGHV gene end",
    },
}
_register_loci()
