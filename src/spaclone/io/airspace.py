"""AIR-SPACE (Jiang et al., PNAS 2026; GEO GSE286452): Curio bead GEX + MiXCR per-read AIR tables.

Per sample, GEO holds the bead AnnData of **all** beads, off-tissue beads included
(``*_adata_<sample>_with_image_raw_counts.h5ad``: raw counts, ``obs`` = ``barcode, x, y, ...``) and the per-read MiXCR
``exportAlignments`` table joined to the demultiplexed bead barcode and UMI (``*_<sample>_ovp_clone_table.txt.gz``:
``readId, allVHitsWithScore, ..., nSeqCDR3, ..., aaSeqCDR3, ..., cloneId, umi, barcode, x, y, len_cDNA,
best_edit_distance, clono_type, receptor_type, CDR3_length``). One AIR molecule is one (bead barcode, UMI, chain);
each read is one R2C2 nanopore consensus read. Bead coordinates are in units of 0.65 µm, as in the AIR-SPACE
re-analysis of this project.

References
----------
Jiang et al. 2026, PNAS (AIR-SPACE; GEO GSE286452); Rodriques et al. 2019, doi:10.1126/science.aaw1219 (Slide-seq
beads, the Curio Seeker chemistry).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone.io._container import SpatialAIR
from spaclone.io._hints import examples
from spaclone.io.airr import clean_call, finish, modal_codes, str_values

__all__ = ["CURIO_UM", "read_airspace", "tissue_beads"]

logger = logging.getLogger("spaclone")

#: Micrometres per Curio bead coordinate unit, as used for AIR-SPACE (GSE286452).
CURIO_UM = S.CURIO_UM

_AIR_COLUMNS = (
    "readId",
    "allVHitsWithScore",
    "allDHitsWithScore",
    "allJHitsWithScore",
    "aaSeqCDR3",
    "nSeqCDR3",
    "cloneId",
    "umi",
    "barcode",
    "clono_type",
    "receptor_type",
)
_REQUIRED = ("barcode", "umi", "receptor_type", "aaSeqCDR3", "cloneId")
#: MiXCR alignment columns kept with ``alignments=True`` and their names in the receptor table.
_ALIGNMENT_COLUMNS = {
    "allVAlignments": "v_alignment_mixcr",
    "allDAlignments": "d_alignment_mixcr",
    "allJAlignments": "j_alignment_mixcr",
    "allCAlignments": "c_alignment_mixcr",
}


def tissue_beads(
    xy: np.ndarray,
    total_counts: np.ndarray,
    *,
    min_counts: float = 100,
    min_neighbours: int = 15,
    radius: float = 100.0,
) -> np.ndarray:
    """Tissue beads by the AIR-SPACE authors' rule.

    A bead is on the tissue when it has at least ``min_counts`` GEX counts and at least ``min_neighbours`` other such
    beads within ``radius`` (in the unit of ``xy``; a bead at exactly ``radius`` counts).

    Parameters
    ----------
    xy
        ``(n, 2)`` bead coordinates.
    total_counts
        GEX counts per bead.
    min_counts, min_neighbours, radius
        The rule's thresholds.

    Returns
    -------
    numpy.ndarray
        Boolean mask of tissue beads.
    """
    xy = np.asarray(xy, dtype=float)
    ok = np.asarray(total_counts, dtype=float) >= min_counts
    out = np.zeros(len(xy), dtype=bool)
    if ok.sum() == 0:
        return out
    pts = xy[ok]
    nn = cKDTree(pts).query_ball_point(pts, r=radius, return_length=True) - 1
    out[np.flatnonzero(ok)[nn >= min_neighbours]] = True
    return out


def _productive_from_cdr3(aa: np.ndarray) -> np.ndarray:
    """CDR3 in frame and stop-free (MiXCR writes ``*`` for a stop codon and ``_`` for an incomplete codon)."""
    s = pd.Series(aa, dtype=object)
    return (s.notna() & ~s.fillna("").str.contains(r"[*_]", regex=True)).to_numpy(dtype=bool)


def _sample_name(path: Path) -> str:
    stem = path.name.split("_ovp_clone_table")[0]
    return stem.split("_", 1)[1] if stem.startswith("GSM") and "_" in stem else stem


def read_airspace(
    h5ad: str | Path | ad.AnnData,
    clone_table: str | Path,
    *,
    loci: Sequence[str] | None = None,
    collapse: Literal["molecule"] | None = "molecule",
    representative: Literal["modal", "first"] = "modal",
    scale_um: float = CURIO_UM,
    tissue_min_counts: float = 100,
    tissue_min_neighbours: int = 15,
    tissue_radius_um: float = 100.0,
    keep_x: bool = True,
    library_id: str | None = None,
    alignments: bool = False,
    min_overlap: float = 0.5,
    unmatched: Literal["drop", "error"] = "drop",
) -> SpatialAIR:
    """Read one AIR-SPACE sample: the bead GEX (every bead, off-tissue beads included) and the per-read AIR table.

    Parameters
    ----------
    h5ad
        Bead AnnData file (or object) with raw counts and ``obs`` columns ``barcode`` (else the index), ``x``, ``y``
        (else ``obsm["spatial"]``), in Curio coordinate units.
    clone_table
        The sample's ``*_ovp_clone_table.txt.gz`` (tab-separated, one row per read).
    loci
        Keep only these chains (``receptor_type``), e.g. ``("IGH", "IGK", "IGL")``. Default: all (BCR and TCR).
    collapse
        ``"molecule"`` (default): one record per (barcode, UMI, chain); ``None``: one record per read.
    representative
        Which read describes a molecule: ``"modal"`` (default) the first read carrying the molecule's most frequent
        CDR3 (ties: the alphabetically first CDR3, as :meth:`pandas.Series.mode`); ``"first"`` the molecule's first
        read in file order (``drop_duplicates(["barcode", "umi", "receptor_type"])``, as in the archive's X1 / X3 /
        X5 scripts). They differ for the ~1% of molecules whose reads disagree.
    scale_um
        Micrometres per bead coordinate unit (0.65).
    tissue_min_counts, tissue_min_neighbours, tissue_radius_um
        The authors' tissue-bead rule (>= 100 GEX counts and >= 15 other such beads within 100 µm), see
        :func:`tissue_beads`; evaluated in source units with radius ``tissue_radius_um / scale_um``.
    keep_x
        Keep the count matrix in ``gex`` (``False`` returns the bead table only).
    library_id
        Library label; default: the sample name from the clone-table file name (``GSM..._<sample>_ovp...``).
    alignments
        Also keep MiXCR's alignment strings of the representative read (``allVAlignments``, ``allDAlignments``,
        ``allJAlignments``, ``allCAlignments`` as ``v_alignment_mixcr``, ``d_alignment_mixcr``,
        ``j_alignment_mixcr``, ``c_alignment_mixcr``): reference start / end / length, read start / end, the
        substitutions and indels against the reference gene, and the score, for every hit. They give somatic
        hypermutation counts without re-alignment: ``bcr.shm(air, mixcr_alignment="v_alignment_mixcr", ...)``
        (parsed by :func:`spaclone.io.parse_mixcr_alignment`). Off by default (they double the table's memory).
    min_overlap
        Smallest share of the clone table's distinct bead barcodes that must be beads of the h5ad; below it the two
        files are taken to come from different samples and a ``ValueError`` gives examples from both.
    unmatched
        Molecules whose barcode is not a bead of the h5ad (their position is unknown): ``"drop"`` leaves them out
        and logs a WARNING with their number, ``"error"`` raises.

    Returns
    -------
    SpatialAIR
        ``air``: one row per molecule with ``spot_id`` (bead barcode), ``x_um``, ``y_um``, ``locus``
        (``receptor_type``), ``clone_id`` (MiXCR ``cloneId``), ``junction_aa`` / ``junction`` (CDR3 of the
        representative read), ``v_call``, ``d_call``, ``j_call`` (its top MiXCR hits), ``c_call`` (``clono_type``,
        the constant gene), ``productive`` (CDR3 in frame and without a stop codon; the rest of V(D)J is not
        checked), ``consensus_count`` (reads of the molecule), ``umi``, ``library_id``, ``x_src`` / ``y_src``,
        ``in_tissue`` and ``n_clones_in_molecule`` (distinct ``cloneId`` among its reads), plus the MiXCR
        alignment columns with ``alignments=True``. Molecules whose barcode is not a bead of the h5ad cannot be
        placed and are left out (``meta["n_records_unmatched_barcode"]``, logged at WARNING); ``meta`` also
        holds the barcode overlap (``spot_overlap``, ``record_overlap``) and the unit check (``unit_check``).
        ``gex``: every bead, ``obs`` index = barcode, ``obsm["spatial"]`` in µm, ``obs["total_counts"]``
        (recomputed from the counts), ``obs["in_tissue"]``, ``obs["x_src"]`` / ``obs["y_src"]``,
        ``obs["library_id"]`` and the h5ad's own ``obs`` columns.

    Raises
    ------
    ValueError
        Fewer than ``min_overlap`` of the clone-table barcodes are beads (files of different samples), or unmatched
        molecules with ``unmatched="error"``.

    Warns
    -----
    spaclone.io.paired.PairedReadWarning
        When the bead positions in µm are implausible (nearest beads outside 0.2-300 µm apart, or a section wider
        than 5 cm): ``scale_um`` is probably wrong.
    """
    if collapse not in ("molecule", None):
        raise ValueError("collapse must be 'molecule' or None")
    if representative not in ("modal", "first"):
        raise ValueError("representative must be 'modal' or 'first'")
    if not (scale_um > 0 and np.isfinite(scale_um)):
        raise ValueError("scale_um must be positive")
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must lie in [0, 1]")
    if unmatched not in ("drop", "error"):
        raise ValueError("unmatched must be 'drop' or 'error'")
    adata = h5ad if isinstance(h5ad, ad.AnnData) else ad.read_h5ad(h5ad)
    obs = adata.obs
    barcodes = str_values(obs["barcode"] if "barcode" in obs.columns else obs.index)
    if pd.isna(barcodes).any() or not pd.Index(barcodes).is_unique:
        raise ValueError("bead barcodes must be unique and non-empty")
    if "x" in obs.columns and "y" in obs.columns:
        xy = obs[["x", "y"]].to_numpy(dtype=float)
    else:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("bead coordinates contain missing values")
    from spaclone.io.gexio import _unit_check
    from spaclone.io.paired import PairedReadWarning

    unit_ok = _unit_check(xy * scale_um, argument="scale_um (unit_um in spaclone.read)")
    if not unit_ok["plausible"]:
        warnings.warn(f"read_airspace: {unit_ok['message']}", PairedReadWarning, stacklevel=2)
    total = np.asarray(adata.X.sum(axis=1), dtype=float).ravel() if adata.n_vars else np.zeros(adata.n_obs)
    in_tissue = tissue_beads(
        xy,
        total,
        min_counts=tissue_min_counts,
        min_neighbours=tissue_min_neighbours,
        radius=tissue_radius_um / scale_um,
    )
    cpath = Path(clone_table)
    lib = library_id if library_id is not None else _sample_name(cpath)
    bead_obs = obs.copy()
    bead_obs.index = pd.Index(barcodes, dtype=str)
    bead_obs["x_src"], bead_obs["y_src"] = xy[:, 0], xy[:, 1]
    bead_obs["total_counts"] = total
    bead_obs["in_tissue"] = in_tissue
    bead_obs[S.LIBRARY] = lib
    gex = ad.AnnData(
        X=adata.X if keep_x else None,
        obs=bead_obs,
        var=adata.var.copy() if keep_x else None,
        obsm={"spatial": xy * scale_um},
    )

    head = pd.read_csv(cpath, sep="\t", nrows=0)
    use = [c for c in _AIR_COLUMNS if c in head.columns]
    aln = [c for c in _ALIGNMENT_COLUMNS if c in head.columns] if alignments else []
    if alignments and "allVAlignments" not in aln:
        raise ValueError("alignments=True: the clone table has no allVAlignments column")
    use += aln
    for need in _REQUIRED:
        if need not in use:
            raise ValueError(f"the clone table lacks column {need!r}")
    M = pd.read_csv(cpath, sep="\t", usecols=use, dtype=str, keep_default_na=False, na_values=[""])
    n_reads = len(M)
    M["receptor_type"] = M["receptor_type"].str.strip().str.upper()
    if loci is not None:
        M = M[M["receptor_type"].isin([str(lc).upper() for lc in loci]).to_numpy(dtype=bool)]
    M = M.reset_index(drop=True)
    n_kept = len(M)
    if collapse == "molecule":
        gid = M.groupby(["barcode", "umi", "receptor_type"], sort=False, dropna=False).ngroup().to_numpy()
        n_mol = int(gid.max()) + 1 if len(gid) else 0
        if representative == "modal":
            cdr3_codes, _ = pd.factorize(M["aaSeqCDR3"], sort=True)
            best = modal_codes(gid, cdr3_codes, n_mol)
            flag = (cdr3_codes == best[gid]) | (best[gid] < 0)
        else:
            flag = np.ones(len(M), dtype=bool)
        order = np.lexsort((np.arange(len(M)), ~flag, gid))
        g_sorted = gid[order]
        firsts = order[np.r_[True, g_sorted[1:] != g_sorted[:-1]]] if len(order) else np.zeros(0, np.int64)
        rep = np.empty(n_mol, dtype=np.int64)
        rep[gid[firsts]] = firsts
        reads = np.bincount(gid, minlength=n_mol).astype(float)
        clone_codes, _ = pd.factorize(M["cloneId"], use_na_sentinel=False)
        pairs = np.unique(np.column_stack([gid, clone_codes]), axis=0) if len(gid) else np.zeros((0, 2), np.int64)
        n_clones = np.bincount(pairs[:, 0], minlength=n_mol) if len(pairs) else np.zeros(n_mol, np.int64)
        R = M.iloc[rep].reset_index(drop=True)
        R["_reads"] = reads
        R["_n_clones"] = n_clones
    else:
        R = M.copy()
        R["_reads"] = 1.0
    pos = pd.Index(barcodes, dtype=str).get_indexer(str_values(R["barcode"]))
    matched = pos >= 0
    n_unmatched = int((~matched).sum())
    bc = pd.unique(str_values(R["barcode"]))
    bc_hit = pd.Index(bc).isin(pd.Index(barcodes, dtype=str))
    share = float(bc_hit.mean()) if len(bc) else 1.0
    if len(bc) and share < min_overlap:
        raise ValueError(
            f"read_airspace: only {share:.1%} of the {len(bc):,} bead barcodes of the clone table {cpath.name} are "
            f"beads of the h5ad (< min_overlap = {min_overlap:.0%}): the two files probably come from different "
            f"samples. Clone-table barcodes: {examples(bc[~bc_hit])}; h5ad beads: {examples(barcodes)}. Give the bead "
            "h5ad of the same sample, or lower min_overlap=."
        )
    if n_unmatched:
        if unmatched == "error":
            raise ValueError(
                f"read_airspace: {n_unmatched:,} molecules on {int((~bc_hit).sum()):,} barcodes that are not beads of "
                "the "
                f"h5ad (e.g. {examples(bc[~bc_hit])}); pass unmatched='drop' to leave them out"
            )
        logger.warning(
            "read_airspace: %d of %d records (%d barcodes) have a barcode that is not a bead of the h5ad and were left "
            "out (their position is unknown)",
            n_unmatched,
            len(R),
            int((~bc_hit).sum()),
        )
    R = R[matched].reset_index(drop=True)
    bead = pos[matched]

    def col(name: str) -> np.ndarray:
        return R[name].to_numpy(dtype=object, na_value=np.nan) if name in R.columns else np.full(len(R), np.nan, object)

    air = pd.DataFrame(
        {
            S.SPOT: barcodes[bead],
            S.X: xy[bead, 0] * scale_um,
            S.Y: xy[bead, 1] * scale_um,
            S.LOCUS: col("receptor_type"),
            S.CLONE: col("cloneId"),
            S.JUNCTION_AA: col("aaSeqCDR3"),
            S.JUNCTION: col("nSeqCDR3"),
            S.V_CALL: clean_call(col("allVHitsWithScore")),
            "d_call": clean_call(col("allDHitsWithScore")),
            S.J_CALL: clean_call(col("allJHitsWithScore")),
            S.C_CALL: clean_call(col("clono_type")),
            S.PRODUCTIVE: _productive_from_cdr3(col("aaSeqCDR3")),
            S.READS: R["_reads"].to_numpy(dtype=float),
            S.UMI: col("umi"),
            S.LIBRARY: lib,
        }
    )
    air["x_src"], air["y_src"] = xy[bead, 0], xy[bead, 1]
    air["in_tissue"] = in_tissue[bead]
    for src_col in aln:
        air[_ALIGNMENT_COLUMNS[src_col]] = col(src_col)
    if collapse == "molecule":
        air["n_clones_in_molecule"] = R["_n_clones"].to_numpy(dtype=np.int64)
    elif "readId" in R.columns:
        air["read_id"] = col("readId")
    meta: dict[str, Any] = {
        "reader": "read_airspace",
        "technology": "AIR-SPACE (Curio Seeker + R2C2 nanopore)",
        "files": [str(h5ad) if not isinstance(h5ad, ad.AnnData) else "<AnnData>", str(cpath)],
        "library_id": lib,
        "level": "molecule" if collapse == "molecule" else "read",
        "molecule_level": True,
        "representative": representative if collapse == "molecule" else None,
        "source_unit": "Curio bead coordinate (0.65 µm)" if scale_um == CURIO_UM else "Curio bead coordinate",
        "um_per_unit": float(scale_um),
        "has_molecule_id": True,
        "has_read_counts": True,
        "n_beads": int(len(barcodes)),
        "n_tissue_beads": int(in_tissue.sum()),
        "tissue_rule": {
            "min_counts": float(tissue_min_counts),
            "min_neighbours": int(tissue_min_neighbours),
            "radius_um": float(tissue_radius_um),
        },
        "n_reads": int(n_reads),
        "n_reads_kept": int(n_kept),
        "collapse": collapse,
        "n_records_unmatched_barcode": n_unmatched,
        "loci": None if loci is None else [str(lc).upper() for lc in loci],
        "alignments": [_ALIGNMENT_COLUMNS[c] for c in aln],
        "spot_overlap": share,
        "record_overlap": float(matched.mean()) if len(matched) else 1.0,
        "unmatched_rule": unmatched if n_unmatched else "none unmatched",
        "unit_check": {k: v for k, v in unit_ok.items() if k != "message"},
    }
    meta["io"] = {
        "reader": "read_airspace",
        "format": "airspace_ovp",
        "evidence": "clone-table columns barcode, umi, receptor_type, cloneId, aaSeqCDR3 (GEO GSE286452)",
        "coordinates": "bead h5ad obs x / y" if "x" in obs.columns else "bead h5ad obsm['spatial']",
        "um_per_unit": {
            "value": float(scale_um),
            "rule": f"scale_um = {scale_um:g} (Curio bead unit used for AIR-SPACE)",
        },
        "molecule": f"(barcode, umi, receptor_type); representative read: {representative}",
        "productive": "derived: CDR3 in frame and stop-free (MiXCR '*' / '_')",
        "tissue": "authors' rule (>= min_counts and >= min_neighbours within radius)",
    }
    logger.info(
        "read_airspace: %d beads (%d on tissue), %d records from %d reads",
        len(barcodes),
        in_tissue.sum(),
        len(air),
        n_kept,
    )
    return finish(air, gex, meta)
