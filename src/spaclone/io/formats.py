"""Registry of the receptor (BCR / TCR) formats that :mod:`spaclone.io` reads, with their documented signatures.

Every format is identified by a header signature that is specific to the tool that writes it: the column names that
must all be present (``requires``), groups of which at least one must be present (``any_of``) and names that must be
absent (``excludes``). A file whose header matches no signature, or the signatures of two unrelated formats, is not
read on a guess: :func:`detect_format` raises and names the candidates and the ``format=`` argument. A format may
``supersede`` a more general one when it is a documented specialisation of it (the Stereo-XCR-seq meta table carries
both MiXCR and AIRR columns; it supersedes both).

Each format also carries its synonym table (``columns``: canonical AIRR field → source column names, first present
wins) and the documented meaning of its count columns (``counts``): ``"reads"``, ``"umis"``, ``"templates"``,
``"cells"`` (barcodes / cells per clone) or ``"unlabelled"`` (the tool writes a bare count whose meaning depends on how
it was run: the user must say what it counts, with ``count_type=``).

References
----------
Vander Heiden et al. 2018, doi:10.3389/fimmu.2018.02206 (AIRR Rearrangement schema; docs.airr-community.org);
10x Genomics, Cell Ranger V(D)J output files (support.10xgenomics.com, "V(D)J annotations"); Bolotin et al. 2015,
doi:10.1038/nmeth.3364 (MiXCR; docs.milaboratories.com); Song et al. 2021, doi:10.1038/s41592-021-01142-2 (TRUST4,
README "Output"); Gupta et al. 2015, doi:10.1093/bioinformatics/btv359 (Change-O, "Data Standards"); Shugay et al.
2015, doi:10.1371/journal.pcbi.1004503 (VDJtools); Shugay et al. 2014, doi:10.1038/nmeth.2960 (MiGEC); Nazarov et
al. 2015, doi:10.1186/s12859-015-0613-1 (tcR / immunarch); Adaptive Biotechnologies, immunoSEQ Analyzer export formats
v2 / v3; BD Rhapsody Sequence Analysis Pipeline user guide (V(D)J outputs); Parse Biosciences Evercode TCR/BCR
(split-pipe ``barcode_report.tsv``); Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy); Suo et al. 2024,
doi:10.1038/s41587-023-01734-7 (Dandelion).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CANONICAL_FIELDS", "COUNT_TYPES", "RECEPTOR_FORMATS", "FormatMatch", "ReceptorFormat", "detect_format"]

#: Canonical receptor fields (AIRR Rearrangement names) a reader can fill, with their meaning.
CANONICAL_FIELDS: dict[str, str] = {
    "cell_id": "cell, spot, bead or bin barcode of the record (the key matched to the GEX units)",
    "sequence_id": "record id of the source (contig, read or clone id)",
    "locus": "IGH / IGK / IGL / TRA / TRB / TRG / TRD",
    "v_call": "V gene call",
    "d_call": "D gene call",
    "j_call": "J gene call",
    "c_call": "constant-region gene call",
    "junction": "IMGT junction nucleotides (CDR3 plus the conserved Cys and Trp/Phe codons)",
    "junction_aa": "IMGT junction amino acids (C ... W/F)",
    "cdr3": "CDR3 nucleotides without the conserved codons",
    "cdr3_aa": "CDR3 amino acids without the conserved residues",
    "productive": "productive rearrangement (bool)",
    "consensus_count": "reads supporting the record",
    "umi_count": "distinct UMIs (molecules) supporting the record",
    "duplicate_count": "AIRR copy number (identical reads, per the AIRR schema)",
    "templates": "estimated templates (immunoSEQ)",
    "cell_count": "cells / barcodes carrying the clone",
    "clone_id": "clone id assigned by the source tool",
    "umi": "molecule identifier (UMI sequence) of a per-molecule or per-read record",
    "sequence": "assembled or read nucleotide sequence",
    "x": "x coordinate written by the source (spatial formats)",
    "y": "y coordinate written by the source (spatial formats)",
}
#: Meanings a count column can have (``count_type=`` values).
COUNT_TYPES: dict[str, str] = {
    "reads": "consensus_count",
    "umis": "umi_count",
    "templates": "templates",
    "cells": "cell_count",
}


@dataclass(frozen=True)
class ReceptorFormat:
    """One receptor format: its signature, synonym table and documented count semantics.

    Attributes
    ----------
    name
        Value of ``format=`` that selects it.
    tool
        Tool (and versions) that write it.
    layout
        ``"table"`` (delimited text), ``"json"``, ``"anndata"``, ``"h5ddl"``, ``"wide"`` (one row per cell, chains in
        column groups), ``"special"`` (a dedicated reader).
    requires, any_of, excludes
        Header signature: every name in ``requires``; at least one name of each ``any_of`` group; none of ``excludes``.
    supersedes
        Formats this one specialises (dropped when both match).
    columns
        Canonical field → source column names in priority order.
    counts
        Source count column → ``"reads"``, ``"umis"``, ``"templates"``, ``"cells"`` or ``"unlabelled"``.
    level
        What one row is: ``"contig"``, ``"clone"``, ``"read"``, ``"molecule"``, ``"cell"`` or ``"chain"``.
    cell_ids
        Whether rows carry a cell / spot / bead barcode (needed to place records on GEX units).
    docs
        Documentation of the fields.
    notes
        Documented semantics worth repeating to the user.
    """

    name: str
    tool: str
    layout: str = "table"
    requires: tuple[str, ...] = ()
    any_of: tuple[tuple[str, ...], ...] = ()
    excludes: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    counts: Mapping[str, str] = field(default_factory=dict)
    level: str = "contig"
    cell_ids: bool = True
    docs: str = ""
    notes: str = ""

    def match(self, cols: Iterable[str]) -> tuple[bool, list[str], int]:
        """Whether a header matches, the evidence, and the number of signature names found (for ranking near misses).

        Parameters
        ----------
        cols
            Header names.
        """
        have = set(cols)
        ev: list[str] = []
        score = 0
        ok = True
        miss = [c for c in self.requires if c not in have]
        score += len(self.requires) - len(miss)
        if miss:
            ok = False
        else:
            ev.append(f"columns {', '.join(self.requires)}")
        for group in self.any_of:
            hit = [c for c in group if c in have]
            if hit:
                score += 1
                ev.append(f"column {hit[0]!r}")
            else:
                ok = False
        bad = [c for c in self.excludes if c in have]
        if bad:
            ok = False
        return ok, ev, score


@dataclass(frozen=True)
class FormatMatch:
    """Result of :func:`detect_format`: the format, why, and the formats it beat or that also came close."""

    format: ReceptorFormat
    evidence: list[str]
    superseded: list[str]
    near: list[str]


def _f(name: str, tool: str, **kw: Any) -> ReceptorFormat:
    for key in ("requires", "excludes", "supersedes"):
        if key in kw:
            kw[key] = tuple(kw[key])
    if "any_of" in kw:
        kw["any_of"] = tuple(tuple(g) for g in kw["any_of"])
    if "columns" in kw:
        kw["columns"] = {k: tuple([v] if isinstance(v, str) else v) for k, v in kw["columns"].items()}
    return ReceptorFormat(name=name, tool=tool, **kw)


_AIRR_COLUMNS = {
    "cell_id": ("cell_id",),
    "sequence_id": ("sequence_id",),
    "locus": ("locus",),
    "v_call": ("v_call",),
    "d_call": ("d_call",),
    "j_call": ("j_call",),
    "c_call": ("c_call",),
    "junction": ("junction",),
    "junction_aa": ("junction_aa",),
    "cdr3": ("cdr3",),
    "cdr3_aa": ("cdr3_aa",),
    "productive": ("productive",),
    "consensus_count": ("consensus_count",),
    "umi_count": ("umi_count",),
    "duplicate_count": ("duplicate_count",),
    "clone_id": ("clone_id",),
    "umi": ("umi",),
    "sequence": ("sequence",),
}
_MIXCR_COLUMNS = {
    "cell_id": ("tagValueCELL", "tagValueCELL0", "tagValueCB", "tagValueSPOT", "tagValueSPATIAL"),
    "umi": ("tagValueUMI", "tagValueUMI0"),
    "sequence_id": ("readId", "Read id", "cloneId", "Clone ID"),
    "clone_id": ("cloneId", "Clone ID"),
    "locus": ("topChains", "chains", "Top chains", "Chains"),
    "v_call": ("allVHitsWithScore", "bestVHit", "allVHits", "All V hits", "Best V hit", "bestVGene"),
    "d_call": ("allDHitsWithScore", "bestDHit", "allDHits", "All D hits", "Best D hit", "bestDGene"),
    "j_call": ("allJHitsWithScore", "bestJHit", "allJHits", "All J hits", "Best J hit", "bestJGene"),
    "c_call": ("allCHitsWithScore", "bestCHit", "allCHits", "All C hits", "Best C hit", "bestCGene"),
    "junction": ("nSeqCDR3", "N. Seq. CDR3"),
    "junction_aa": ("aaSeqCDR3", "AA. Seq. CDR3"),
    "consensus_count": ("readCount", "cloneCount", "Clone count", "Read count"),
    "umi_count": ("uniqueMoleculeCount", "uniqueUMICount", "uniqueTagCountUMI", "Unique UMI count"),
    "sequence": ("targetSequences", "Clonal sequence(s)", "Target sequences"),
}

#: Every receptor format, in the order the documentation lists them.
RECEPTOR_FORMATS: dict[str, ReceptorFormat] = {
    f.name: f
    for f in (
        _f(
            "airr",
            "AIRR Rearrangement TSV 1.x / 2.0 (IgBLAST -outfmt 19, Change-O, Cell Ranger airr_rearrangement.tsv, TRUST4 "
            "*_airr.tsv / *_barcode_airr.tsv, MiXCR exportAirr, Dandelion, scirpy write_airr, BD Rhapsody AIRR, "
            "StereoIR, LongAIRR)",
            requires=("v_call", "j_call"),
            any_of=[("junction_aa", "junction", "cdr3_aa", "cdr3"), ("sequence_id", "cell_id")],
            excludes=("V_CALL", "v_gene", "Chain_Type"),
            columns=_AIRR_COLUMNS,
            counts={"consensus_count": "reads", "duplicate_count": "reads", "umi_count": "umis"},
            level="contig",
            docs="https://docs.airr-community.org/en/stable/datarep/rearrangements.html",
            notes="AIRR schema: consensus_count = reads contributing to the consensus or contig; duplicate_count = "
            "copy number, e.g. identical reads (title 'Read count'); umi_count = distinct UMIs (schema 1.4+).",
        ),
        _f(
            "10x_contig_csv",
            "10x Genomics Cell Ranger V(D)J filtered_contig_annotations.csv / all_contig_annotations.csv (all versions)",
            requires=("barcode", "contig_id", "chain", "v_gene", "j_gene", "cdr3"),
            columns={
                "cell_id": "barcode",
                "sequence_id": "contig_id",
                "locus": "chain",
                "v_call": "v_gene",
                "d_call": "d_gene",
                "j_call": "j_gene",
                "c_call": "c_gene",
                "junction_aa": "cdr3",
                "junction": "cdr3_nt",
                "productive": "productive",
                "consensus_count": "reads",
                "umi_count": "umis",
                "clone_id": "raw_clonotype_id",
            },
            counts={"reads": "reads", "umis": "umis"},
            level="contig",
            docs="https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/"
            "cr-5p-outputs-annotations-vdj",
            notes="reads = reads assigned to the contig; umis = UMIs; cdr3 / cdr3_nt include the conserved C and "
            "W/F (IMGT junction); 'None' marks a missing value; chain 'Multi' = contig spanning several chains.",
        ),
        _f(
            "10x_contig_json",
            "10x Genomics Cell Ranger V(D)J all_contig_annotations.json",
            layout="json",
            requires=("barcode", "contig_name", "annotations"),
            columns={
                "cell_id": "barcode",
                "sequence_id": "contig_name",
                "locus": "chain",
                "v_call": "v_gene",
                "d_call": "d_gene",
                "j_call": "j_gene",
                "c_call": "c_gene",
                "junction_aa": "cdr3",
                "junction": "cdr3_seq",
                "productive": "productive",
                "consensus_count": "read_count",
                "umi_count": "umi_count",
                "clone_id": "raw_clonotype_id",
                "sequence": "sequence",
            },
            counts={"read_count": "reads", "umi_count": "umis"},
            level="contig",
            docs="https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/"
            "cr-5p-outputs-annotations-vdj",
            notes="genes come from 'annotations' (feature.region_type L-REGION+V-REGION / D-REGION / J-REGION / "
            "C-REGION); clonotype from info.raw_clonotype_id.",
        ),
        _f(
            "10x_consensus_csv",
            "10x Genomics Cell Ranger consensus_annotations.csv (one row per clonotype consensus, no barcodes)",
            requires=("clonotype_id", "consensus_id", "chain", "v_gene", "cdr3"),
            excludes=("barcode",),
            columns={
                "sequence_id": "consensus_id",
                "clone_id": "clonotype_id",
                "locus": "chain",
                "v_call": "v_gene",
                "d_call": "d_gene",
                "j_call": "j_gene",
                "c_call": "c_gene",
                "junction_aa": "cdr3",
                "junction": "cdr3_nt",
                "productive": "productive",
                "consensus_count": "reads",
                "umi_count": "umis",
            },
            counts={"reads": "reads", "umis": "umis"},
            level="clone",
            cell_ids=False,
            docs="https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/"
            "cr-5p-outputs-annotations-vdj",
        ),
        _f(
            "10x_clonotypes_csv",
            "10x Genomics Cell Ranger clonotypes.csv (one row per clonotype: frequency = cells, cdr3s_aa = "
            "'IGH:CAR...;IGK:CQQ...'; no barcodes, no genes)",
            layout="special",
            requires=("clonotype_id", "frequency", "proportion", "cdr3s_aa"),
            level="clone",
            cell_ids=False,
            docs="https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/"
            "cr-5p-outputs-annotations-vdj",
            notes="read filtered_contig_annotations.csv (or airr_rearrangement.tsv) of the same run instead: "
            "clonotypes.csv has no barcodes and no V / J genes.",
        ),
        _f(
            "mixcr_clones",
            "MiXCR exportClones (MiXCR 2 human-readable headers, MiXCR 3 cloneCount, MiXCR 4 readCount / "
            "uniqueMoleculeCount; tag columns tagValueCELL / tagValueUMI)",
            any_of=[
                ("cloneId", "Clone ID"),
                ("readCount", "cloneCount", "Clone count", "uniqueMoleculeCount", "uniqueUMICount"),
                ("aaSeqCDR3", "nSeqCDR3", "AA. Seq. CDR3", "N. Seq. CDR3"),
                ("allVHitsWithScore", "bestVHit", "allVHits", "All V hits", "Best V hit", "bestVGene"),
            ],
            excludes=("readId", "Read id", "descrR1"),
            columns=_MIXCR_COLUMNS,
            counts={
                "readCount": "reads",
                "cloneCount": "reads",
                "Clone count": "reads",
                "uniqueMoleculeCount": "umis",
                "uniqueUMICount": "umis",
                "uniqueTagCountUMI": "umis",
            },
            level="clone",
            docs="https://mixcr.com/mixcr/reference/mixcr-export/",
            notes="MiXCR's CDR3 (aaSeqCDR3 / nSeqCDR3) runs from the conserved Cys to the conserved Trp/Phe "
            "(= IMGT junction); hits carry scores in brackets, e.g. IGHV3-23*00(1450); readCount / cloneCount = "
            "reads, uniqueMoleculeCount = UMIs (MiXCR 4).",
        ),
        _f(
            "mixcr_alignments",
            "MiXCR exportAlignments (one row per read; optional cloneId and tag columns)",
            any_of=[
                ("readId", "Read id", "descrR1"),
                ("aaSeqCDR3", "nSeqCDR3", "AA. Seq. CDR3", "N. Seq. CDR3"),
                ("allVHitsWithScore", "bestVHit", "allVHits", "All V hits", "Best V hit", "bestVGene"),
            ],
            excludes=("readCount", "cloneCount", "Clone count", "uniqueMoleculeCount"),
            columns={**_MIXCR_COLUMNS, "sequence_id": ("readId", "Read id", "descrR1")},
            counts={},
            level="read",
            docs="https://mixcr.com/mixcr/reference/mixcr-exportAlignments/",
            notes="one row per read: reads per record = 1; records are collapsed to (barcode, UMI, chain) molecules "
            "when a UMI tag is present.",
        ),
        _f(
            "trust4_report",
            "TRUST4 *_report.tsv (bulk CDR3 report; VDJtools-compatible)",
            requires=("count", "frequency", "CDR3nt", "CDR3aa", "V", "D", "J", "C", "cid"),
            columns={
                "sequence_id": "cid",
                "v_call": "V",
                "d_call": "D",
                "j_call": "J",
                "c_call": "C",
                "junction": "CDR3nt",
                "junction_aa": "CDR3aa",
            },
            counts={"count": "unlabelled"},
            level="clone",
            cell_ids=False,
            docs="https://github.com/liulab-dfci/TRUST4#output",
            notes="'#count' is the read count of a bulk run, but the number of barcodes when TRUST4 ran with "
            "--barcode and UMIs with --UMI (README): the file does not say which.",
        ),
        _f(
            "trust4_barcode_report",
            "TRUST4 *_barcode_report.tsv (one row per barcode; chains as comma-separated fields)",
            layout="wide",
            requires=("barcode", "cell_type", "chain1", "chain2"),
            columns={"cell_id": "barcode"},
            counts={"read_cnt": "reads"},
            level="chain",
            docs="https://github.com/liulab-dfci/TRUST4#10x-genomics-data-and-barcode-based-single-cell-data",
            notes="chain = V,D,J,C,cdr3_nt,cdr3_aa,read_cnt,consensus_id,CDR3_germline_similarity,"
            "consensus_full_length; chain1 = IGH/TRB/TRD, chain2 = IGK/IGL/TRA/TRG; secondary chains are "
            "';'-separated.",
        ),
        _f(
            "changeo",
            "Change-O legacy uppercase table (Change-O < 1.0 or --format changeo)",
            requires=("SEQUENCE_ID", "V_CALL", "J_CALL"),
            any_of=[("JUNCTION", "CDR3_IMGT", "JUNCTION_AA")],
            columns={
                "cell_id": ("CELL", "CELL_ID"),
                "sequence_id": "SEQUENCE_ID",
                "locus": "LOCUS",
                "v_call": "V_CALL",
                "d_call": "D_CALL",
                "j_call": "J_CALL",
                "c_call": ("C_CALL", "CREGION", "ISOTYPE", "PRCONS"),
                "junction": "JUNCTION",
                "junction_aa": "JUNCTION_AA",
                "cdr3": "CDR3_IMGT",
                "productive": ("FUNCTIONAL", "PRODUCTIVE"),
                "consensus_count": "CONSCOUNT",
                "duplicate_count": "DUPCOUNT",
                "umi_count": "UMICOUNT",
                "clone_id": "CLONE",
                "sequence": "SEQUENCE_INPUT",
            },
            counts={"CONSCOUNT": "reads", "DUPCOUNT": "reads", "UMICOUNT": "umis"},
            level="contig",
            docs="https://changeo.readthedocs.io/en/stable/standard.html",
            notes="FUNCTIONAL = productive (T/F); CONSCOUNT = reads in the UMI consensus (pRESTO), DUPCOUNT = "
            "duplicate count, UMICOUNT = UMIs.",
        ),
        _f(
            "immunarch",
            "immunarch export ('# Exported from immunarch'; also tcR)",
            requires=("Clones", "Proportion", "CDR3.nt", "CDR3.aa", "V.name"),
            columns={
                "cell_id": ("Barcode", "barcode"),
                "junction": "CDR3.nt",
                "junction_aa": "CDR3.aa",
                "v_call": "V.name",
                "d_call": "D.name",
                "j_call": "J.name",
                "sequence": "Sequence",
            },
            counts={"Clones": "unlabelled"},
            level="clone",
            cell_ids=False,
            docs="https://immunarch.com/articles/web_only/load_mixcr.html",
            notes="'Clones' is whatever the upstream tool counted (reads, UMIs or cells).",
        ),
        _f(
            "vdjtools",
            "VDJtools clonotype table (also written by MiXCR / MiGEC converters)",
            requires=("count", "cdr3nt", "cdr3aa", "v", "d", "j"),
            any_of=[("freq", "frequency")],
            columns={"junction": "cdr3nt", "junction_aa": "cdr3aa", "v_call": "v", "d_call": "d", "j_call": "j"},
            counts={"count": "unlabelled"},
            level="clone",
            cell_ids=False,
            docs="https://vdjtools-doc.readthedocs.io/en/master/input.html#vdjtools-format",
            notes="'count' is reads or UMI-tagged molecules depending on the upstream data.",
        ),
        _f(
            "migec",
            "MiGEC / CdrBlast clonotype table",
            requires=("Count", "CDR3 nucleotide sequence", "CDR3 amino acid sequence", "V segments", "J segments"),
            any_of=[("Good events", "Total events")],
            columns={
                "junction": "CDR3 nucleotide sequence",
                "junction_aa": "CDR3 amino acid sequence",
                "v_call": "V segments",
                "d_call": "D segments",
                "j_call": "J segments",
                "umi_count": "Good events",
                "consensus_count": "Good reads",
            },
            counts={"Good events": "umis", "Good reads": "reads", "Count": "unlabelled"},
            level="clone",
            cell_ids=False,
            docs="https://migec.readthedocs.io/en/latest/cdrblast.html",
            notes="events = molecules (MIGs) when CdrBlast ran on assembled MIGs, reads otherwise.",
        ),
        _f(
            "mitcr",
            "MiTCR clonotype table",
            requires=("CDR3 nucleotide sequence", "CDR3 amino acid sequence", "V segments", "J segments"),
            any_of=[("Read count", "Read Count", "Count")],
            excludes=("Good events", "Total events"),
            columns={
                "junction": "CDR3 nucleotide sequence",
                "junction_aa": "CDR3 amino acid sequence",
                "v_call": "V segments",
                "d_call": "D segments",
                "j_call": "J segments",
                "consensus_count": ("Read count", "Read Count"),
            },
            counts={"Read count": "reads", "Read Count": "reads", "Count": "unlabelled"},
            level="clone",
            cell_ids=False,
            docs="https://github.com/milaboratory/mitcr",
        ),
        _f(
            "immunoseq_v2",
            "Adaptive immunoSEQ export v1 / v2 (nucleotide, aminoAcid, count (templates/reads), vMaxResolved, ...)",
            requires=("nucleotide", "aminoAcid"),
            any_of=[
                ("count (templates/reads)", "count (reads)", "count (templates)", "count", "estimatedNumberGenomes"),
                ("vMaxResolved", "vGeneName", "vFamilyName"),
            ],
            columns={
                "sequence": "nucleotide",
                "junction_aa": "aminoAcid",
                "v_call": ("vMaxResolved", "vGeneName", "vFamilyName"),
                "d_call": ("dMaxResolved", "dGeneName", "dFamilyName"),
                "j_call": ("jMaxResolved", "jGeneName", "jFamilyName"),
                "productive": "sequenceStatus",
            },
            counts={
                "count (reads)": "reads",
                "count (templates)": "templates",
                "estimatedNumberGenomes": "templates",
                "count (templates/reads)": "unlabelled",
                "count": "unlabelled",
            },
            level="clone",
            cell_ids=False,
            docs="https://clients.adaptivebiotech.com (immunoSEQ Analyzer: export formats v2)",
            notes="aminoAcid is the CDR3 from the conserved Cys to Phe; sequenceStatus In / Out / Stop; gene names in "
            "Adaptive nomenclature (TCRBV05-01*01); 'count (templates/reads)' holds templates when the assay "
            "estimated them, reads otherwise.",
        ),
        _f(
            "immunoseq_v3",
            "Adaptive immunoSEQ export v3 (rearrangement, amino_acid, frame_type, templates, reads, ...)",
            requires=("rearrangement", "amino_acid"),
            any_of=[("templates", "reads"), ("v_resolved", "v_gene", "v_family")],
            columns={
                "sequence": "rearrangement",
                "junction_aa": "amino_acid",
                "v_call": ("v_resolved", "v_gene", "v_family"),
                "d_call": ("d_resolved", "d_gene", "d_family"),
                "j_call": ("j_resolved", "j_gene", "j_family"),
                "productive": "frame_type",
                "templates": "templates",
                "consensus_count": "reads",
            },
            counts={"templates": "templates", "reads": "reads"},
            level="clone",
            cell_ids=False,
            docs="https://clients.adaptivebiotech.com (immunoSEQ Analyzer: export formats v3)",
            notes="frame_type In / Out / Stop; templates may be 'na' when the assay did not estimate them.",
        ),
        _f(
            "bd_rhapsody",
            "BD Rhapsody V(D)J *_VDJ_perCellChain.csv / *_VDJ_Dominant_Contigs.csv / *_VDJ_Unfiltered_Contigs.csv",
            requires=("Chain_Type",),
            any_of=[("V_gene", "V_gene_Dominant"), ("CDR3_Translation", "CDR3_Translation_Dominant")],
            columns={
                "cell_id": ("Cell_Index", ""),
                "locus": "Chain_Type",
                "v_call": ("V_gene", "V_gene_Dominant"),
                "d_call": ("D_gene", "D_gene_Dominant"),
                "j_call": ("J_gene", "J_gene_Dominant"),
                "c_call": ("C_gene", "C_gene_Dominant"),
                "junction": ("CDR3_Nucleotide", "CDR3_Nucleotide_Dominant"),
                "junction_aa": ("CDR3_Translation", "CDR3_Translation_Dominant"),
                "productive": ("Productive", "Productive_Dominant"),
                "consensus_count": ("Read_Count", "Read_Count_Dominant"),
                "umi_count": ("Molecule_Count", "Molecule_Count_Dominant"),
            },
            counts={
                "Read_Count": "reads",
                "Read_Count_Dominant": "reads",
                "Molecule_Count": "umis",
                "Molecule_Count_Dominant": "umis",
            },
            level="chain",
            docs="https://bd-rhapsody-bioinfo-docs.genomics.bd.com/outputs/vdj.html",
            notes="Chain_Type TCR_Alpha / TCR_Beta / TCR_Gamma / TCR_Delta / BCR_Heavy / BCR_Kappa / BCR_Lambda; "
            "Read_Count = reads, Molecule_Count = molecules (UMIs).",
        ),
        _f(
            "parse_barcode_report",
            "Parse Biosciences Evercode TCR / BCR barcode_report.tsv (one row per cell)",
            layout="wide",
            requires=("Barcode",),
            any_of=[("TRA_V", "TRB_V", "IGH_V", "IGK_V", "IGL_V")],
            columns={"cell_id": "Barcode"},
            counts={"_read_count": "reads", "_transcript_count": "umis"},
            level="chain",
            docs="https://support.parsebiosciences.com (Evercode TCR / BCR: barcode_report.tsv)",
            notes="per locus L and rank ('' or 'secondary_'): L_V, L_D, L_J, L_C, L_cdr3_aa, L_read_count (reads), "
            "L_transcript_count (transcripts = UMIs).",
        ),
        _f(
            "wat3r",
            "WAT3R barcode_results.csv (Seq-Well TCR)",
            layout="wide",
            requires=("BC", "TRBV", "TRBJ"),
            any_of=[("TRB_CDR3", "TRA_CDR3")],
            columns={"cell_id": "BC"},
            counts={"_nReads": "reads", "_CDR3_UMIcount": "umis"},
            level="chain",
            docs="https://github.com/mainciburu/WAT3R",
        ),
        _f(
            "sptcr_seq",
            "SPTCR-seq per-section CSV (GEO GSE238071; IgBLAST + UMI-tools on Visium spot barcodes)",
            requires=("barcode", "expression", "Locus", "CDR3_aa", "V", "J"),
            columns={
                "sequence_id": "index",
                "cell_id": "barcode",
                "locus": "Locus",
                "v_call": "V",
                "d_call": "D",
                "j_call": "J",
                "junction": "CDR3",
                "junction_aa": "CDR3_aa",
                "umi_count": "expression",
            },
            counts={"expression": "umis"},
            level="clone",
            docs="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE238071",
            notes="one row per (Visium spot barcode, clonotype); expression = UMI count after UMI-tools collapsing "
            "(GEO series description); CDR3 / CDR3_aa are IgBLAST CDR3s without the conserved anchors.",
        ),
        _f(
            "stereo_xcr_meta",
            "Stereo-XCR-seq per-read meta table (MiXCR + AIRR join, or the public pipeline's out.meta.gz)",
            layout="special",
            any_of=[("readId", "readname"), ("topChains", "isotype"), ("cloneId", "cloneID")],
            requires=("x", "y"),
            supersedes=("airr", "mixcr_alignments", "mixcr_clones"),
            counts={},
            level="read",
            docs="https://github.com/fengyu9481/stereo-xcr-seq",
            notes="read with spaclone.io.read_stereo_xcr: x / y are bin1 DNB coordinates (0.5 µm).",
        ),
        _f(
            "airspace_ovp",
            "AIR-SPACE *_ovp_clone_table.txt.gz (MiXCR exportAlignments + bead barcode / UMI; GEO GSE286452)",
            layout="special",
            requires=("barcode", "umi", "receptor_type", "cloneId", "aaSeqCDR3"),
            supersedes=("mixcr_alignments", "mixcr_clones"),
            counts={},
            level="read",
            docs="https://github.com/ShaowenJCornell/AIR-SPACE",
            notes="read with spaclone.io.read_airspace together with the sample's bead h5ad (0.65 µm per unit).",
        ),
        _f(
            "custom",
            "any delimited table, columns given with columns={canonical: source}",
            columns={},
            counts={},
            level="contig",
            docs="",
        ),
    )
}


def detect_format(
    cols: Sequence[str],
    *,
    layout: str = "table",
    candidates: Iterable[str] | None = None,
) -> FormatMatch:
    """Identify a receptor format from its header names.

    Parameters
    ----------
    cols
        Header names of a table (or the keys of a JSON record).
    layout
        ``"table"`` or ``"json"``: only formats of that layout (and ``"wide"`` / ``"special"`` tables) are considered.
    candidates
        Restrict the search to these format names.

    Returns
    -------
    FormatMatch

    Raises
    ------
    ValueError
        If no signature matches (the message lists the closest formats and what they lack) or two signatures that do
        not supersede one another both match (the message names both and the ``format=`` argument).
    """
    names = list(candidates) if candidates is not None else [n for n in RECEPTOR_FORMATS if n != "custom"]
    pool = [RECEPTOR_FORMATS[n] for n in names]
    if layout == "json":
        pool = [f for f in pool if f.layout == "json"]
    else:
        pool = [f for f in pool if f.layout in ("table", "wide", "special")]
    hits: list[tuple[ReceptorFormat, list[str]]] = []
    scored: list[tuple[int, ReceptorFormat]] = []
    for f in pool:
        ok, ev, score = f.match(cols)
        if ok:
            hits.append((f, ev))
        scored.append((score, f))
    hit_names = {f.name for f, _ in hits}
    superseded = sorted({s for f, _ in hits for s in f.supersedes if s in hit_names})
    hits = [(f, ev) for f, ev in hits if f.name not in superseded]
    scored.sort(key=lambda t: -t[0])
    near = [f.name for s, f in scored if s > 0 and f.name not in hit_names][:3]
    shown = ", ".join(map(repr, list(cols)[:14])) + (" ..." if len(cols) > 14 else "")
    if not hits:
        lack = []
        for name in near:
            f = RECEPTOR_FORMATS[name]
            miss = [c for c in f.requires if c not in set(cols)]
            miss += ["one of " + "/".join(g) for g in f.any_of if not set(g) & set(cols)]
            lack.append(f"{name} (lacks {', '.join(miss[:4]) or 'nothing; an excluded column is present'})")
        raise ValueError(
            f"unrecognised receptor table: its header ({shown}) matches no documented signature. Closest: "
            f"{'; '.join(lack) or 'none'}. Pass format= (one of {', '.join(n for n in RECEPTOR_FORMATS)}) or "
            "format='custom' with columns={'cell_id': ..., 'junction_aa': ..., 'v_call': ..., ...}."
        )
    if len(hits) > 1:
        desc = "; ".join(f"{f.name} ({', '.join(ev)})" for f, ev in hits)
        raise ValueError(
            f"ambiguous receptor table: the header ({shown}) matches the signatures of {len(hits)} formats: {desc}. "
            f"Pass format= with one of {', '.join(repr(f.name) for f, _ in hits)}."
        )
    f, ev = hits[0]
    return FormatMatch(format=f, evidence=ev, superseded=superseded, near=near)
