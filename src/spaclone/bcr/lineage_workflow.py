"""Germline-rooted lineage trees with tissue positions, end to end from spot transcriptomes and AIRR records.

:func:`lineage_workflow` turns heavy-chain AIRR records with spot positions (and, for the tip locations, the spot
transcriptome) into lineage trees per clone and per *related clone group*, one tip per cell-scale locus and cleaned
sequence, each tip with its isotype, its mutations from the clonal germline, its position and its location class. It
is the 2026-10 StereoIR lineage series (archive L1-L8: contig table, alignments, genotype, tips, dowser trees,
figures) in one call, computed from the AIRR fields alone:

1. **Records**: heavy-chain records with a valid junction (IMGT C104 ... W/F118, >= 5 aa), productive, and at least
   ``min_reads`` reads (the archive's S1 / T1 set).
2. **Groupings**: ``"exact"`` clones (same V gene, J gene and junction amino acids) and ``"related"`` groups (same V
   gene, J gene and junction length, single linkage of junction nucleotide identity >= ``identity``; a group can join
   several exact clones).
3. **Loci**: records of one exact clone in spots within ``loci_radius_um`` (15 µm) are one locus, about one cell.
4. **Register**: every record is put in the clone's register from its contig (``sequence``) and its V alignment
   (``v_sequence_alignment`` / ``v_germline_alignment``, gaps ``-``): V germline positions up to the junction
   (insertions relative to the germline dropped, deletions missing data, leading mismatches of the alignment dropped as
   a local aligner would), the junction, then the contig after the junction up to the J end. The records of a clone
   line up as IMGT-gapped alignments do. TRUST4's ``sequence_alignment`` is not used: in the 2026-10 StereoIR data it
   is not a substring of the contig in 25% of records.
5. **Usable, full-length**: a record is usable when its junction lies in its contig and its V alignment is present;
   a full-length record (<= ``max_lead_missing`` V germline nucleotides missing at the 5' end) must also be
   productive by an IgBLAST-like rule (junction in the codon frame of the V start, no stop codon from the V start to
   the J end). Only full-length records become tips; every usable record supports variants (step 7).
6. **Germline**: per clone, the V and J alleles called most often among its full-length records (Change-O
   ``CreateGermlines --cloned``); V germline bases from the V germline alignments of every record of that allele, J
   germline bases as the majority of the sequenced J-templated bases of every record of that J allele; the D segment
   and N regions masked (``N``), the V- and J-templated junction ends taken from one representative record.
7. **Noise**: a base that differs from the reference (the germline where templated, the clone's majority base where
   masked) is kept only when the same base at the same position occurs in at least ``noise_min_spots`` different
   spots and at least one record carrying it has ``noise_min_reads`` reads; otherwise it is an assembly or PCR error
   and is reverted. Missing positions take the reference base. A record whose own V allele fits >= 3 V positions better
   than the clone germline is left out of the clone (a misfit).
8. **Tips**: one tip = one locus x one cleaned full-length sequence; its isotype is the most frequent of IgM / IgD / IgG / IgA among its records (ties in that
   order); its location is that of its locus (CSR hotspot > T zone > red pulp > follicle).
9. **Trees**: a clone gets a tree when it has at least ``min_distinct`` distinct cleaned sequences and at least
   ``min_sites`` mutation sites (V positions up to the conserved cysteine); :func:`spaclone.bcr.build_trees` builds
   germline-rooted parsimony (or neighbour-joining) trees, identical sequences with identical isotype and location
   collapsed into one tip (dowser ``formatClones``).
10. **Selection for figures**: per grouping, trees with at least ``min_tips`` tips, ordered by the largest tip mutation
    count (``order="max_mutation"``), at most ``max_trees``; :func:`spaclone.pl.lineage_gallery` draws them.

Differences to the archive (which used IgBLAST + Change-O + TIgGER + dowser) are listed in the Notes of
:func:`lineage_workflow`.

References
----------
Song et al. 2021, doi:10.1038/s41592-021-01142-2 (TRUST4); Gupta et al. 2015, doi:10.1093/bioinformatics/btv359
(Change-O germlines); Gadala-Maria et al. 2015, doi:10.1073/pnas.1417683112 (TIgGER genotypes); Gupta et al. 2017,
doi:10.4049/jimmunol.1601850 (clonal grouping by V, J, junction length and identity); Hoehn et al. 2022,
doi:10.1371/journal.pcbi.1009885 (dowser); Fitch 1971, doi:10.2307/2412116 (parsimony).
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, as_rng, rng_entropy

__all__ = ["LineageWorkflowResult", "LineageWorkflowWarning", "lineage_workflow"]

logger = logging.getLogger("spaclone")

#: Location classes of tips, in the archive's priority order (first match wins): a CSR hotspot pixel at the locus
#: centre, else the compartment there.
LOCATIONS = ("CSR hotspot", "T zone", "red pulp", "follicle")
#: Isotype classes that name a tip (ties resolved in this order); anything else is ``"unknown"``.
TIP_ISOTYPES = ("IgM", "IgD", "IgG", "IgA")
#: Cell-type labels that define the B-rich, T-rich and erythrocyte-rich masks (cell2location / RCTD style labels).
COMPARTMENT_TYPES = {
    "B": ("B_cells", "B lineage", "B cells", "B", "B cell"),
    "T": ("T_cells", "T", "T cells", "T cell"),
    "erythroid": ("Erythrocyte", "erythrocyte", "Erythrocytes", "erythroid", "Erythroid"),
}
_GROUPINGS = {"exact": "exact", "e": "exact", "E": "exact", "related": "related", "l": "related", "L": "related"}
_ALIGN_COLS = ("v_sequence_alignment", "v_germline_alignment", "v_germline_start")
_N, _GAP = ord("N"), ord("-")
_LUT = np.full(256, 4, dtype=np.int8)  # A C G T -> 0..3, anything else -> 4 (missing)
for _k, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _k
    _LUT[ord(_c.lower())] = _k
_DEC = np.array(list("ACGTN."))  # codes 0-3 bases, 4 missing, 5 legacy 5' padding


class LineageWorkflowWarning(SpacloneWarning):
    """A part of :func:`lineage_workflow` ran with less than it needs (no transcriptome, a harsh read floor)."""


# ------------------------------------------------------------------------------------------------------ result


@dataclass(eq=False)
class LineageWorkflowResult:
    """Output of :func:`lineage_workflow`.

    Attributes
    ----------
    groupings
        The groupings computed (``"exact"``, ``"related"``).
    trees
        ``{grouping: LineageTrees}`` (germline-rooted, branch lengths in mutations).
    tips
        ``{grouping: DataFrame}``, one row per tip (one locus x one cleaned sequence) of the clones that got a tree:
        ``tree_id``, ``tip_id``, ``tree_label`` (the collapsed tree tip holding it; missing when left out, e.g. an
        in-frame stop codon), ``locus_id``, ``isotype``, ``location``, ``mutations`` (V mutations from the clonal
        germline, up to the conserved cysteine), ``depth`` (mutations from the germline along the tree), ``x_um``,
        ``y_um`` (mean of its records' spots), ``n_records``, ``n_spots``, ``reads``, ``exact_clone`` (the exact clone
        of its locus), ``junction_aa``, ``record_ids`` and ``sequence`` (the cleaned sequence in the clone's register).
    clones
        ``{grouping: DataFrame}``, one row per clone / group with at least one full-length usable record: sizes,
        distinct sequences, mutation sites, isotype and location counts, germline alleles, ``selected`` (gets a
        tree).
    selection
        One row per tree: ``grouping``, ``tree_id``, ``n_tips`` (tips held by the tree), ``n_tree_tips`` (collapsed
        tree tips), ``n_loci``, ``max_mutations``, ``median_mutations``, ``n_exact_clones``, ``eligible`` (>=
        ``min_tips`` tips), ``rank`` (order among the eligible trees) and ``drawn`` (rank <= ``max_trees``).
    members
        Member exact clones of every related group with a tree: ``tree_id``, ``exact_clone``, ``junction_aa``,
        ``n_tips`` (largest first).
    records
        One row per record used (heavy chain, valid, productive, read floor): clone ids per grouping, ``locus_id``,
        ``isotype``, ``location``, ``usable``, ``full_length``, ``misfit_<grouping>``, ``mutations_<grouping>`` and
        ``sequence_<grouping>`` (the cleaned register sequence of full-length records of clones with a germline).
    loci
        One row per locus: ``locus_id``, exact clone, centre (``x_um``, ``y_um``), ``n_spots``, ``n_records``,
        ``hotspot`` (label, 0 = none), ``compartment`` and ``location``.
    maps
        Display layers for :mod:`spaclone.pl` (each a :class:`~spaclone.tl.Raster` with its own grid): ``"tissue"``,
        ``"b_rich"``, ``"t_rich"``, ``"compartments"``, ``"hotspots"`` (labels), ``"share"`` / ``"share_eligible"``
        (CSR switched-share map when a hotspot result was given), plus ``"grid"``, ``"rotation"`` (centre and angle
        that put the section's long axis horizontal) and ``"frame"``; missing layers are ``None``.
    params
        Every argument that defines the result, with the generator's seed entropy and the run time;
        ``params["compartments_from"]`` says where the compartments came from (``"gex"``, ``"given"`` or ``None``)
        and ``params["location_classes"]`` which location classes were computed (see :attr:`location_classes`).
    """

    groupings: tuple[str, ...]
    trees: dict[str, Any]
    tips: dict[str, pd.DataFrame]
    clones: dict[str, pd.DataFrame]
    selection: pd.DataFrame
    members: pd.DataFrame
    records: pd.DataFrame
    loci: pd.DataFrame
    maps: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def location_classes(self) -> tuple[str, ...]:
        """Location classes this run could assign, in priority order (a legend lists these and no others).

        ``"CSR hotspot"`` only with hotspots; ``"T zone"``, ``"red pulp"``, ``"follicle"`` only where compartments
        were computed (each only if its cell types were found); ``"unknown"`` when no compartments were computed.
        """
        return tuple(self.params.get("location_classes", LOCATIONS))

    def __repr__(self) -> str:
        parts = []
        for g in self.groupings:
            sel = self.selection[self.selection.grouping == g]
            parts.append(f"{g}: {len(self.trees[g])} trees, {len(self.tips[g])} tips, {int(sel.drawn.sum())} drawn")
        return "LineageWorkflowResult(" + "; ".join(parts) + ")"

    def summary(self) -> pd.DataFrame:
        """One row per grouping: clones with a full-length record, clones with a tree, tips, tree tips, figures."""
        out = []
        for g in self.groupings:
            sel = self.selection[self.selection.grouping == g]
            out.append(
                {
                    "grouping": g,
                    "clones": len(self.clones[g]),
                    "trees": len(self.trees[g]),
                    "tips": len(self.tips[g]),
                    "tips_in_trees": int(self.tips[g].tree_label.notna().sum()) if len(self.tips[g]) else 0,
                    "tree_tips": int(sel.n_tree_tips.sum()),
                    "eligible": int(sel.eligible.sum()),
                    "drawn": int(sel.drawn.sum()),
                }
            )
        return pd.DataFrame(out)

    def newick(self, grouping: str = "exact") -> dict[str, str]:
        """Newick string of every tree of a grouping (branch lengths in mutations)."""
        return self.trees[_grouping(grouping)].newick

    def drawn(self, grouping: str | None = None) -> pd.DataFrame:
        """Trees selected for figures, in figure order (one grouping or all)."""
        sel = self.selection[self.selection.drawn]
        if grouping is not None:
            sel = sel[sel.grouping == _grouping(grouping)]
        return sel.sort_values(["grouping", "rank"]).reset_index(drop=True)

    def tree_tips(self, tree_id: str, *, grouping: str = "exact") -> pd.DataFrame:
        """The tips of one tree (rows of ``tips[grouping]``), only those held by the tree."""
        g = _grouping(grouping)
        t = self.tips[g]
        return t[(t.tree_id == tree_id) & t.tree_label.notna()].reset_index(drop=True)


def _grouping(name: str) -> str:
    try:
        return _GROUPINGS[name]
    except KeyError:
        raise ValueError(f"grouping must be 'exact' or 'related', got {name!r}") from None


# ------------------------------------------------------------------------------------------------------ inputs


def _unpack(data: Any, gex: Any) -> tuple[pd.DataFrame, Any, Any]:
    """Receptor records, transcriptome and (when present) a hotspot workflow result from the input object."""
    workflow = None
    if isinstance(data, pd.DataFrame):
        air = data
    elif hasattr(data, "records") and isinstance(data.records, pd.DataFrame):  # e.g. bcr.hotspot_workflow result
        air, workflow = data.records, data
        gex = gex if gex is not None else getattr(data, "gex", None)
    elif hasattr(data, "air") and isinstance(data.air, pd.DataFrame):  # SpatialAIR
        air = data.air
        gex = gex if gex is not None else getattr(data, "gex", None)
    else:
        raise TypeError(
            "lineage_workflow: data must be a receptor table (DataFrame), a SpatialAIR (spaclone.io.read / read_airr) "
            f"or a hotspot workflow result with .records, got {type(data).__name__}"
        )
    return air, gex, workflow


def _select_records(
    air: pd.DataFrame, *, locus: str, min_reads: float, reads_key: str, locus_key: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Heavy-chain records with a valid junction, productive and at least ``min_reads`` reads (S1 / T1 set)."""
    from spaclone.pp.validity import is_valid_junction

    seq = [] if ("sequence" in air.columns or "sequence_alignment" in air.columns) else ["sequence"]
    no_align = seq + [c for c in _ALIGN_COLS if c not in air.columns]
    if no_align:
        raise ValueError(
            f"lineage_workflow: the receptor table has no {', '.join(no_align)}. Lineage trees are built from the "
            "contig sequences and their V alignments: AIRR rearrangements with `sequence` (or `sequence_alignment`), "
            "`v_sequence_alignment`, `v_germline_alignment` and `v_germline_start`, as IgBLAST writes them "
            "(`igblastn -outfmt 19`). Annotate the contigs with IgBLAST's AIRR output if you have them; clone tables "
            "and other exports without these fields (e.g. AIR-SPACE's clone tables) cannot give lineage trees. If "
            "the AIRR file has these columns, read it with all fields (spaclone.io.read_airr(..., columns='all'))."
        )
    need = [S.SPOT, S.X, S.Y, locus_key, S.V_CALL, S.J_CALL, S.JUNCTION, S.JUNCTION_AA]
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(f"lineage_workflow: the receptor table lacks {missing}")
    n0 = len(air)
    loc = air[locus_key].astype("string").str.upper()
    keep = (loc == locus).fillna(False).to_numpy(dtype=bool)
    n_locus = int(keep.sum())
    keep &= is_valid_junction(air[S.JUNCTION_AA])
    n_valid = int(keep.sum())
    if S.PRODUCTIVE in air.columns:
        prod = air[S.PRODUCTIVE]
        if pd.api.types.is_bool_dtype(prod):
            keep &= prod.fillna(False).to_numpy(dtype=bool)
        else:
            keep &= (
                prod.astype("string").str.strip().str.upper().isin(["T", "TRUE", "1", "Y", "YES"]).to_numpy(dtype=bool)
            )
    n_prod = int(keep.sum())
    if min_reads and min_reads > 0:
        if reads_key not in air.columns:
            raise KeyError(f"lineage_workflow: read column {reads_key!r} not in the table (or pass min_reads=0)")
        keep &= pd.to_numeric(air[reads_key], errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= min_reads
    out = air.loc[keep].copy()
    if "sequence_id" not in out.columns:
        out["sequence_id"] = [f"r{k}" for k in range(len(out))]
    out["sequence_id"] = out["sequence_id"].astype(str)
    if out["sequence_id"].duplicated().any():
        raise ValueError("lineage_workflow: sequence_id values must be unique among the heavy-chain records")
    counts = {
        "records_in": n0,
        f"records_{locus}": n_locus,
        "valid_junction": n_valid,
        "productive": n_prod,
        f"reads_ge_{min_reads:g}": len(out),
    }
    return out.reset_index(drop=True), counts


# ------------------------------------------------------------------------------------------------------ groupings


def _related_names(rec: pd.DataFrame, raw: pd.Series, v: pd.Series, j: pd.Series) -> pd.Series:
    """Archive names of related groups: V gene | J gene | most frequent junction aa (``#k`` for duplicates)."""
    d = pd.DataFrame({"g": raw.to_numpy(), "v": v.to_numpy(), "j": j.to_numpy(), "aa": rec[S.JUNCTION_AA].to_numpy()})
    first = d.groupby("g", sort=True).agg(v=("v", "first"), j=("j", "first"))
    main = d.groupby(["g", "aa"], sort=True).size().reset_index(name="n")
    main = main.sort_values(["g", "n", "aa"], ascending=[True, False, True]).drop_duplicates("g").set_index("g").aa
    name = first.v + "|" + first.j + "|" + main.reindex(first.index)
    dup = name.duplicated(keep=False)
    if dup.any():
        name[dup] = name[dup] + "#" + name[dup].groupby(name[dup]).cumcount().add(1).astype(str)
    return raw.map(name)


def _groupings(rec: pd.DataFrame, identity: float) -> pd.DataFrame:
    from spaclone.pp.clones import define_clones, gene_name

    v = gene_name(rec[S.V_CALL]).astype(object)
    j = gene_name(rec[S.J_CALL]).astype(object)
    base = pd.DataFrame({S.V_CALL: v.to_numpy(), S.J_CALL: j.to_numpy()}, index=rec.index)
    base[S.JUNCTION_AA] = rec[S.JUNCTION_AA].astype(str).to_numpy()
    base[S.JUNCTION] = rec[S.JUNCTION].astype(str).str.upper().to_numpy()
    ex = define_clones(base, method="exact", by=(), key_added="c")["c"]
    rl = define_clones(base, method="vj_length_identity", identity=identity, by=(), junction_key=S.JUNCTION,
                       key_added="c")["c"]  # fmt: skip
    rec["clone_exact"] = ex.astype(object).to_numpy()
    rec["clone_related"] = _related_names(rec, rl.astype(object), v, j).to_numpy()
    return rec


# ------------------------------------------------------------------------------------------------------ register


def _cigar_lead(values: pd.Series) -> np.ndarray:
    """Leading soft clip (``<n>S`` at the start) of CIGAR strings; 0 without one, -1 when missing."""
    s = values.astype("string")
    lead = s.str.extract(r"^(\d+)S", expand=False)
    out = pd.to_numeric(lead, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    out[s.isna().to_numpy()] = -1
    return out


@dataclass
class _Parsed:
    """One record broken into the parts of a clone register (V germline positions, junction, J offsets)."""

    ok: bool
    cys: int = -1  # V germline position of the first junction nucleotide (the conserved cysteine codon)
    vpos: np.ndarray | None = None  # V germline positions upstream of the junction
    vbase: np.ndarray | None = None  # sequence bases there (uint8; N for a deletion)
    junction: bytes = b""
    fbase: np.ndarray | None = None  # sequence bases after the junction (J offsets 0, 1, ... up to the J end)
    vg_pos: np.ndarray | None = None  # V germline positions covered by the V alignment
    vg_base: np.ndarray | None = None  # germline bases there
    n_v: int = 0  # V-templated junction nucleotides
    n_j: int = 0  # J-templated junction nucleotides
    lead_missing: int = 0  # V germline nucleotides missing at the 5' end
    in_frame: bool = True  # V-J in the V germline frame, no stop codon from the V start to the J end


_STOPS = (b"TAA", b"TAG", b"TGA")


def _parse(contig: bytes, v_seq: str, v_germ: str, g0: int, v_start: int, junction: str, j_start: int,
           j_end: int) -> _Parsed:  # fmt: skip
    """Break one record into register parts from its contig and its V alignment.

    ``v_seq`` / ``v_germ``: the V alignment (``-`` = gap) starting at contig position ``v_start`` (0-based) and at V
    germline position ``g0`` (0-based); ``j_start`` / ``j_end``: contig positions of the J alignment (-1 unknown).
    """
    if not v_seq or len(v_seq) != len(v_germ) or not junction or v_start < 0:
        return _Parsed(False)
    jx = junction.upper().encode("ascii", "replace")
    lj = len(jx)
    js = contig.find(jx, max(0, v_start - lj))
    if js < 0:
        js = contig.find(jx)
    if js < 0:
        return _Parsed(False)
    s = np.frombuffer(v_seq.upper().encode("ascii", "replace"), np.uint8)
    g = np.frombuffer(v_germ.upper().encode("ascii", "replace"), np.uint8)
    seq_char, germ_char = s != _GAP, g != _GAP
    # a local alignment starts with a match: drop leading mismatches and gaps (aligner overhang into the leader)
    match = seq_char & germ_char & (s == g)
    k = int(np.argmax(match)) if match.any() else len(s)
    if k:
        v_start += int(seq_char[:k].sum())
        g0 += int(germ_char[:k].sum())
        s, g, seq_char, germ_char = s[k:], g[k:], seq_char[k:], germ_char[k:]
    if not len(s):
        return _Parsed(False)
    cpos = v_start + np.cumsum(seq_char) - seq_char  # contig position of each column (next base for deletions)
    gpos = g0 + np.cumsum(germ_char) - germ_char  # germline position of each column
    v_end = v_start + int(seq_char.sum())
    g_end = g0 + int(germ_char.sum())
    up = germ_char & (cpos < js)  # germline columns upstream of the junction
    vpos = gpos[up]
    vbase = np.where(seq_char[up], s[up], _N).astype(np.uint8)
    if v_end < js:  # V alignment ends before the junction: extend in germline coordinates without gaps
        ext = np.arange(v_end, js)
        vpos = np.concatenate([vpos, g_end + (ext - v_end)])
        vbase = np.concatenate([vbase, np.frombuffer(contig[v_end:js], np.uint8)])
        cys = g_end + (js - v_end)
    else:
        at = np.flatnonzero(germ_char & (cpos >= js))
        cys = int(gpos[at[0]]) if len(at) else g_end
    vgc = np.flatnonzero(germ_char)
    n_v = max(0, min(v_end, js + lj) - js)
    n_j = max(0, min(js + lj - j_start, lj)) if j_start >= 0 else 0
    f_hi = j_end if j_end > js + lj else len(contig)
    fbase = np.frombuffer(contig[js + lj : f_hi], np.uint8).copy()
    # productive (IgBLAST-like): junction in the V germline codon frame, no stop codon from the V start to the J end
    frame0 = (v_start - g0) % 3
    in_frame = (js - frame0) % 3 == 0
    if in_frame:
        lo = v_start + ((frame0 - v_start) % 3)
        hi = f_hi if j_end > 0 else js + lj
        in_frame = not any(contig[i : i + 3] in _STOPS for i in range(lo, hi - 2, 3))
    return _Parsed(
        True,
        cys=int(cys),
        vpos=vpos.astype(np.int64),
        vbase=vbase,
        junction=jx,
        fbase=fbase,
        vg_pos=gpos[vgc].astype(np.int64),
        vg_base=g[vgc],
        n_v=int(n_v),
        n_j=int(n_j),
        lead_missing=int(g0),
        in_frame=bool(in_frame),
    )


_J_SHIFT = 128  # J germline offsets from the junction end start at -_J_SHIFT (inside the junction)


def _vote(keys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Majority base per (allele, position) from packed keys; ties -> the lowest base code (A < C < G < T)."""
    uk, n = np.unique(np.concatenate(keys), return_counts=True)
    site = uk >> 3
    order = np.lexsort((uk & 7, -n, site))
    first = np.r_[True, site[order][1:] != site[order][:-1]]
    best = uk[order][first]
    return best >> 24, ((best >> 3) & ((1 << 21) - 1)) - 4096, (best & 7).astype(np.int8)


def _germline_tables(
    parsed: list[_Parsed], v_alleles: np.ndarray, j_alleles: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Germline codes per V allele and per J allele, and the junction-start position per V allele.

    V: by V germline position, from the V germline alignments. J: by offset from the junction end (shifted by
    ``_J_SHIFT``), the majority of the sequenced J-templated bases (SHM rarely reaches FWR4). Each value is the
    majority over all records of the allele.
    """
    vnames = pd.Index(sorted({a for a in v_alleles if isinstance(a, str)}))
    jnames = pd.Index(sorted({a for a in j_alleles if isinstance(a, str)}))
    vc = vnames.get_indexer([a if isinstance(a, str) else None for a in v_alleles])
    jc = jnames.get_indexer([a if isinstance(a, str) else None for a in j_alleles])
    vkeys, jkeys, cys_rows = [], [], []
    for p, a, b in zip(parsed, vc, jc, strict=True):
        if not p.ok:
            continue
        if a >= 0:
            code = _LUT[p.vg_base].astype(np.int64)
            ok = code < 4
            if ok.any():
                vkeys.append((np.int64(a) << 24) | ((p.vg_pos[ok].astype(np.int64) + 4096) << 3) | code[ok])
            cys_rows.append((a, p.cys))
        if b >= 0:  # J germline = majority of the sequenced J-templated bases (junction end and FWR4)
            jb = np.frombuffer(p.junction, np.uint8)[len(p.junction) - p.n_j :]
            off = np.concatenate([np.arange(-p.n_j, 0), np.arange(len(p.fbase))]).astype(np.int64)
            code = _LUT[np.concatenate([jb, p.fbase])].astype(np.int64)
            ok = (code < 4) & (off >= -_J_SHIFT)
            if ok.any():
                jkeys.append((np.int64(b) << 24) | ((off[ok] + 4096) << 3) | code[ok])
    vtab: dict[str, np.ndarray] = {}
    if vkeys:
        al, pos, base = _vote(vkeys)
        for k in np.unique(al):
            m = al == k
            arr = np.full(int(pos[m].max()) + 1, 4, dtype=np.int8)
            arr[pos[m]] = base[m]
            vtab[str(vnames[int(k)])] = arr
    jtab: dict[str, np.ndarray] = {}
    if jkeys:
        al, pos, base = _vote(jkeys)
        for k in np.unique(al):
            m = al == k
            arr = np.full(int(pos[m].max()) + _J_SHIFT + 1, 4, dtype=np.int8)
            arr[pos[m] + _J_SHIFT] = base[m]
            jtab[str(jnames[int(k)])] = arr
    cys: dict[str, int] = {}
    if cys_rows:
        d = pd.DataFrame(cys_rows, columns=["a", "c"])
        n = d.groupby(["a", "c"], sort=True).size().reset_index(name="n")
        n = n.sort_values(["a", "n", "c"], ascending=[True, False, True]).drop_duplicates("a")
        cys = {str(vnames[int(a)]): int(c) for a, c in zip(n.a, n.c, strict=True)}
    return vtab, jtab, cys


def _own_mismatches(parsed: list[_Parsed], v_alleles: np.ndarray, vtab: dict[str, np.ndarray]) -> np.ndarray:
    """V mismatches (up to the conserved cysteine codon) of every record against its own allele's germline."""
    out = np.full(len(parsed), np.nan)
    for k, (p, a) in enumerate(zip(parsed, v_alleles, strict=True)):
        g = vtab.get(a) if isinstance(a, str) else None
        if not p.ok or g is None:
            continue
        pos = np.concatenate([p.vpos, p.cys + np.arange(min(3, len(p.junction)))])
        base = _LUT[np.concatenate([p.vbase, np.frombuffer(p.junction[:3], np.uint8)])]
        keep = (pos < p.cys + 3) & (pos >= 0) & (pos < len(g))
        gb = g[pos[keep]]
        out[k] = float(((base[keep] != gb) & (base[keep] < 4) & (gb < 4)).sum())
    return out


def _tied_top(codes: np.ndarray) -> np.ndarray:
    """Codes tied for the most occurrences (Change-O's consensus allele: every allele with the top count)."""
    c = codes[codes >= 0]
    if not len(c):
        return np.empty(0, dtype=codes.dtype)
    u, n = np.unique(c, return_counts=True)
    return u[n == n.max()]


# ------------------------------------------------------------------------------------------------------ cleaning


def _clean(seqs: np.ndarray, spots: np.ndarray, reads: np.ndarray, germ: np.ndarray, min_spots: int,
           min_reads: float) -> np.ndarray:  # fmt: skip
    """Archive L4 noise rule on a clone's register codes (0..3 = ACGT, 4 = missing); returns the cleaned codes.

    Reference = germline where templated, else the clone's majority base (ties -> lowest code). A base differing
    from the reference is kept when it occurs in >= ``min_spots`` distinct spots and one record carrying it has >=
    ``min_reads`` reads; otherwise it is reverted. Missing positions take the reference.
    """
    ref = germ.copy()
    masked = np.flatnonzero(germ == 4)
    if len(masked):
        col = seqs[:, masked]
        cnt = np.stack([(col == b).sum(0) for b in range(4)])
        ref[masked] = np.where(cnt.sum(0) > 0, cnt.argmax(0), 4)
    var = (seqs < 4) & (ref[None, :] < 4) & (seqs != ref[None, :])
    keep = np.zeros_like(var)
    ii, pp = np.nonzero(var)
    if len(ii):
        key = pp.astype(np.int64) * 4 + seqs[ii, pp]
        pair = np.unique(key * (int(spots.max()) + 1) + spots[ii])
        uk, nsp = np.unique(pair // (int(spots.max()) + 1), return_counts=True)
        hi = np.unique(key[reads[ii] >= min_reads])
        ok = uk[(nsp >= min_spots) & np.isin(uk, hi)]
        good = np.isin(key, ok)
        keep[ii[good], pp[good]] = True
    out = np.where(var & ~keep, ref[None, :], seqs)
    return np.where((out == 4) & (ref[None, :] < 4), ref[None, :], out).astype(np.int8)


def _tip_isotype(iso: pd.Series) -> str:
    cnt = iso.value_counts()
    top = max((int(cnt.get(i, 0)) for i in TIP_ISOTYPES), default=0)
    if top == 0:
        return "unknown"
    return next(i for i in TIP_ISOTYPES if int(cnt.get(i, 0)) == top)


# ------------------------------------------------------------------------------------------------------ tissue


def _raster_from(obj: Any) -> Any:
    """Label raster of a hotspot input: a HotspotResult (``.labels``), an object with ``.csr``, or a Raster."""
    from spaclone.tl.structures import Raster

    if obj is None:
        return None
    if isinstance(obj, Raster):
        return obj
    for attr in ("labels",):
        r = getattr(obj, attr, None)
        if isinstance(r, Raster):
            return r
    csr = getattr(obj, "csr", None)
    if csr is not None:
        return _raster_from(csr)
    raise TypeError(
        "hotspots must be a spaclone.tl.Raster of hotspot labels, a HotspotResult (bcr.csr_hotspots / tl.hotspots) "
        f"or a hotspot workflow result with .csr, got {type(obj).__name__}"
    )


def _share_layers(obj: Any) -> tuple[Any, Any]:
    """CSR switched-share map and its eligible pixels of a hotspot input, as Rasters (or ``None``)."""
    from spaclone.tl.structures import Raster

    if obj is None:
        return None, None
    m = getattr(obj, "map", None)
    if m is None and getattr(obj, "csr", None) is not None:
        m = getattr(obj.csr, "map", None)
    if m is None or getattr(m, "grid", None) is None:
        return None, None
    return Raster(np.asarray(m.share, dtype=float), m.grid, name="switched share"), Raster(
        np.asarray(m.eligible, dtype=bool), m.grid, name="eligible"
    )


def _gex_units(gex: Any, cell_type_key: str | None, types: Mapping[str, Sequence[str]]) -> tuple[pd.DataFrame, str]:
    """Spots of the transcriptome with µm coordinates and their compartment class (B / T / erythroid / None)."""
    obs = gex.obs
    key = cell_type_key
    if key is None:
        key = next((k for k in ("cell2loc_anno", "cell_type", "first_type", "anno") if k in obs.columns), None)
    if key is None or key not in obs.columns:
        raise KeyError(
            "lineage_workflow: the transcriptome has no cell-type column for the compartments (pass "
            "cell_type_key=, or compartments= / compartments=False)"
        )
    if "spatial" in getattr(gex, "obsm", {}):
        xy = np.asarray(gex.obsm["spatial"], dtype=float)[:, :2]
    elif {S.X, S.Y} <= set(obs.columns):
        xy = obs[[S.X, S.Y]].to_numpy(dtype=float)
    else:
        raise KeyError("lineage_workflow: the transcriptome has no obsm['spatial'] (µm) and no x_um / y_um columns")
    lab = obs[key].astype("string").to_numpy(dtype=object, na_value=None)
    cls = np.full(len(lab), None, dtype=object)
    for c, names in types.items():
        cls[np.isin(lab, list(names))] = c
    return pd.DataFrame({S.X: xy[:, 0], S.Y: xy[:, 1], "class": cls}), key


def _structures(gex: Any, *, cell_type_key: str | None, types: Mapping[str, Sequence[str]], pixel_um: float,
                structure_kw: Mapping[str, Any], receptor_xy: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """B-rich, T-rich and erythrocyte-rich masks, compartments (T zone > red pulp > follicle) and the tissue mask.

    The tissue (for display) is made of the typed spots and the receptor spots, closed and filled, as the archive's
    T_common.Tissue: spots of other types can lie beyond the section (StereoIR: a straight-edged band of
    macrophage- and neutrophil-labelled bins below it).
    """
    from spaclone.tl.structures import Grid, compartments, mask_from_types, tissue_mask

    units, key = _gex_units(gex, cell_type_key, types)
    typed = units[units["class"].notna()]
    if not len(typed):
        raise ValueError(f"lineage_workflow: no spot of the transcriptome has a cell type of {dict(types)} in {key!r}")
    grid = Grid.from_points(typed[S.X], typed[S.Y], pixel_um=pixel_um)
    kw = dict(structure_kw)
    masks = {}
    for c in ("B", "T", "erythroid"):
        if (typed["class"] == c).any():
            masks[c] = mask_from_types(typed, types=c, type_key="class", grid=grid, **kw)
        else:
            masks[c] = None
    rules = []
    if masks["T"] is not None:
        rules.append(("T zone", masks["T"]))
    if masks["erythroid"] is not None:
        rules.append(("red pulp", masks["erythroid"]))
    if masks["B"] is not None:
        rules.append(("red pulp", ~masks["B"]))
    comp = compartments(rules, default="follicle") if rules else None
    pad = {k: v for k, v in kw.items() if k == "pad_edges"}
    pts = typed[[S.X, S.Y]]
    if receptor_xy is not None and len(receptor_xy):
        pts = pd.concat([pts, pd.DataFrame(np.asarray(receptor_xy, dtype=float), columns=[S.X, S.Y])])
    tissue = tissue_mask(pts, grid=grid, **pad)
    return {
        "grid": grid,
        "b_rich": masks["B"],
        "t_rich": masks["T"],
        "erythroid_rich": masks["erythroid"],
        "compartments": comp,
        "tissue": tissue,
        "cell_type_key": key,
        **_orientation(typed[[S.X, S.Y]].to_numpy(dtype=float)),
    }


def _orientation(xy: np.ndarray, pad_um: float = 10.0) -> dict[str, Any]:
    """Rotation that puts the long axis of a point set horizontal, and the display frame (archive T_common.Tissue).

    The long axis is the first principal axis; the frame spans the 0.1-99.9% quantiles of the rotated points plus
    ``pad_um``.
    """
    c0 = xy.mean(0)
    v = np.linalg.eigh(np.cov((xy - c0).T))[1][:, -1]
    a = float(-np.arctan2(v[1], v[0]))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = np.quantile((xy - c0) @ rot.T, [0.001, 0.999], axis=0)
    frame = (q[0, 0] - pad_um, q[1, 0] + pad_um, q[0, 1] - pad_um, q[1, 1] + pad_um)
    return {"rotation": {"centre": (float(c0[0]), float(c0[1])), "angle": a}, "frame": tuple(float(f) for f in frame)}


# ------------------------------------------------------------------------------------------------------ main


def lineage_workflow(
    data: Any,
    gex: Any = None,
    *,
    groupings: Sequence[str] = ("exact", "related"),
    identity: float = 0.85,
    min_tips: int = 10,
    max_trees: int | None = 16,
    order: Literal["max_mutation", "tips"] = "max_mutation",
    method: Literal["parsimony", "nj"] = "parsimony",
    hotspots: Any = None,
    compartments: Any = None,
    min_reads: float | Literal["auto"] = "auto",
    locus: str = "IGH",
    loci_radius_um: float = 15.0,
    max_lead_missing: int = 3,
    noise_min_spots: int = 2,
    noise_min_reads: float = 30,
    misfit_mismatches: int | None = 3,
    min_distinct: int = 5,
    min_sites: int = 1,
    traits: Sequence[str] = ("isotype", "location"),
    cell_type_key: str | None = None,
    compartment_types: Mapping[str, Sequence[str]] | None = None,
    structure_kw: Mapping[str, Any] | None = None,
    pixel_um: float = 10.0,
    check_frame: bool = True,
    germline_start_base: Literal[0, 1, "auto"] = "auto",
    legacy: bool = False,
    igblast: bool | Mapping[str, Any] = False,
    n_starts: int = 4,
    n_ratchet: int = 8,
    n_jobs: int = 1,
    rng: RngLike = 0,
    reads_key: str = S.READS,
    locus_key: str = S.LOCUS,
) -> LineageWorkflowResult:
    """Lineage trees per clone and per related clone group, with every tip's isotype, mutations and tissue location.

    Parameters
    ----------
    data
        Heavy-chain AIRR records with spot positions: a :class:`~spaclone.io.SpatialAIR` (``spaclone.io.read`` /
        ``spaclone.io.read_airr(..., columns="all")``), a receptor table, or a hotspot workflow result with
        ``.records`` (:func:`spaclone.bcr.hotspot_workflow`; its ``.csr`` is used as ``hotspots`` when that is not
        given). The records need the contig ``sequence`` (else the ungapped ``sequence_alignment`` stands in), the V
        alignment ``v_sequence_alignment`` / ``v_germline_alignment`` / ``v_germline_start`` (TRUST4 writes it 0-based),
        the contig positions of the V and J alignments (AIRR ``v_sequence_start``, ``j_sequence_start``,
        ``j_sequence_end``, or TRUST4's ``v_cigar`` / ``j_cigar``), and ``junction``, ``junction_aa``, ``v_call``,
        ``j_call``, ``c_call``, ``productive`` and the reads. Without the contig and its V alignment (e.g. a clone
        table) a ``ValueError`` names what is missing: IgBLAST's AIRR output (``igblastn -outfmt 19``) has every
        field.
    gex
        Spot transcriptome (AnnData, ``obsm["spatial"]`` in µm) with a cell-type column (``cell_type_key``) for the
        compartments; default the ``gex`` of ``data`` (a :class:`~spaclone.io.SpatialAIR`, or a hotspot workflow
        result, which keeps the transcriptome it ran on). Not needed when ``compartments`` is given or ``False``;
        without it (and with ``compartments=None``) a :class:`LineageWorkflowWarning` says that locations are only
        "CSR hotspot" or "unknown".
    groupings
        ``"exact"`` (same V gene, J gene and junction amino acids) and / or ``"related"`` (same V gene, J gene and
        junction length, junction nucleotide identity >= ``identity``, single linkage).
    identity
        Junction nucleotide identity of related groups (0.85).
    min_tips
        Trees with at least this many tips (one locus x one cleaned sequence, those left out of the tree not counted)
        are eligible for figures.
    max_trees
        At most this many trees per grouping are marked ``drawn`` (``None``: all eligible trees).
    order
        Figure order of eligible trees: ``"max_mutation"`` (largest tip mutation count first; ties: more collapsed
        tree tips, then more tips, then the tree id) or ``"tips"`` (most tips first).
    method
        Tree inference of :func:`spaclone.bcr.build_trees`: ``"parsimony"`` or ``"nj"``.
    hotspots
        CSR hotspots for the location class "CSR hotspot": a :class:`~spaclone.tl.HotspotResult` (e.g.
        :func:`spaclone.bcr.csr_hotspots`; its switched-share map is kept for figures), a label
        :class:`~spaclone.tl.Raster` (0 = none), or ``None`` (no hotspot class).
    compartments
        Compartment :class:`~spaclone.tl.Raster` with categories (``"T zone"``, ``"red pulp"``, ``"follicle"``), or
        ``None`` to derive it from ``gex`` with :func:`spaclone.tl.mask_from_types` (B-rich, T-rich and
        erythrocyte-rich masks of the typed spots; T zone > red pulp (erythrocyte-rich or outside the B-rich area) >
        follicle, the archive's T_common rule), or ``False`` (no compartments: locations are ``"unknown"`` outside
        hotspots).
    min_reads
        Read floor of the records (S1 / T1: 2). ``"auto"`` (default): 1 when every record is one UMI molecule or one
        read (the reader's ``meta["level"]`` ``"molecule"`` / ``"read"``), else 2, as in
        :func:`spaclone.bcr.hotspot_workflow`; a :class:`LineageWorkflowWarning` names the floor when it removes more
        than half of the productive heavy-chain records.
    locus
        Chain (heavy chain ``"IGH"``).
    loci_radius_um
        Records of one exact clone in spots within this distance form one locus (T1: 15 µm).
    max_lead_missing
        A record is full-length when at most this many V germline nucleotides are missing at its 5' end (L4: 3).
    noise_min_spots, noise_min_reads
        A base that differs from the reference is kept when it occurs in at least ``noise_min_spots`` spots of the
        clone and one record carrying it has at least ``noise_min_reads`` reads (L4: 2 and 30).
    misfit_mismatches
        A record whose own V allele fits at least this many V positions better than the clone germline is left out of
        the clone (L4: 3); ``None`` keeps every record.
    min_distinct, min_sites
        A clone gets a tree with at least ``min_distinct`` distinct cleaned full-length sequences and ``min_sites``
        mutation sites (L4: 5 and 1).
    traits
        Tip columns that must also agree for identical sequences to collapse into one tree tip (dowser:
        ``("isotype", "location")``).
    cell_type_key
        Cell-type column of ``gex.obs`` (default the first of ``cell2loc_anno``, ``cell_type``, ``first_type``,
        ``anno``).
    compartment_types
        Cell-type labels of the B, T and erythroid classes (default :data:`COMPARTMENT_TYPES`); spots with other
        labels are left out of the masks, as in the archive.
    structure_kw
        Passed to :func:`spaclone.tl.mask_from_types` (e.g. ``{"pad_edges": False}`` reproduces the 2026-10 archive's
        edge handling exactly).
    pixel_um
        Pixel size of the structure grid (10 µm).
    check_frame
        A full-length record is usable only when its junction lies in the codon frame of its V germline start and its
        contig has no stop codon from the V start to the J end in that frame (an IgBLAST-like productive call; the
        archive required IgBLAST's productive call). Truncated records are not checked: they only support variants.
    germline_start_base
        Whether ``v_germline_start`` counts from 0 (TRUST4) or 1 (AIRR standard); ``"auto"``: 0 when any value is 0.
    legacy
        Reproduce two quirks of the 2026-10 archive: V nucleotides missing in front of a record's first sequenced
        nucleotide stay as padding (IMGT ``.``) instead of taking the reference base, so sequences that differ only by
        1-3 missing 5' nucleotides count as distinct sequences and tips (the archive's L4 docstring says they take the
        reference; its code kept the padding); and the structure masks keep scipy's edge erosion (``pad_edges=False``).
    igblast
        ``True`` (or a dict of :func:`spaclone.bcr.run_igblast` arguments): re-align every record's ``sequence`` with
        an installed IgBLAST and use its V calls and alignments (records whose IgBLAST junction differs from the
        input junction, or that IgBLAST calls unproductive, are not usable, as in the archive). Default: the input
        alignments (TRUST4).
    n_starts, n_ratchet, n_jobs
        Tree search of :func:`spaclone.bcr.build_trees`: random addition orders, ratchet iterations per start, worker
        processes (a more thorough search than ``build_trees``' default; results do not depend on ``n_jobs``).
    rng
        Seed or generator; trees are reproducible for a given seed.
    reads_key, locus_key
        Column names.

    Returns
    -------
    LineageWorkflowResult

    Notes
    -----
    Differences to the archive's L1-L8 (IgBLAST 1.22 + Change-O MakeDb / CreateGermlines + TIgGER + dowser):

    * The register comes from the input V alignment and the contig (TRUST4), not from IMGT-gapped IgBLAST
      alignments; the columns line up the same way (V germline positions, junction, J positions), only the IMGT gap
      columns (missing in every sequence) are absent, so parsimony lengths are unchanged. Insertions relative to the
      germline are dropped (the archive kept a codon insertion that fell into IMGT gap columns as a distinct
      sequence).
    * Alleles are the input V / J calls (TRUST4); the archive corrected V alleles to the animal's genotype (TIgGER)
      and used IgBLAST's calls for the germline. Where they disagree (StereoIR: 10% of the clones with a tree, e.g.
      ``IGHV5-17*03`` by TRUST4 and IgBLAST vs ``*01`` by the genotype), a clone's mutation counts differ by the polymorphic
      positions; on records with the same allele the counts agree for 99%.
    * Productive full-length records follow an IgBLAST-like rule on the contig (99.9% agreement with the archive's
      IgBLAST call on StereoIR); truncated records are not checked (they only support variants; the archive required
      IgBLAST's productive call for them too, 1% of records).
    * A misfit is a record whose own V allele (input call) fits the clone better than the consensus allele; the
      archive compared IgBLAST alleles, which can also name another V gene.
    * Missing 5' nucleotides take the reference base, as the archive's L4 documents; its code kept them as padding,
      which made sequences that differ only there distinct (``legacy=True`` reproduces this).
    * Trees: :func:`spaclone.bcr.build_trees` parsimony (stepwise addition, SPR, ratchet) instead of phangorn's
      ``pratchet``; equally short trees can differ (see validation ``v3_lineage``).

    StereoIR (37,690 heavy-chain records): about 35 s on one core.

    Examples
    --------
    >>> sa = spaclone.io.read_airr("mm_spleen_bcr_airr.tsv", columns="all")  # doctest: +SKIP
    >>> res = spaclone.bcr.lineage_workflow(sa, gex, hotspots=csr)  # doctest: +SKIP
    >>> spaclone.pl.lineage_gallery(res, "figures/")  # doctest: +SKIP
    """
    t_start = time.time()
    gens = [_grouping(g) for g in groupings]
    if not gens:
        raise ValueError("lineage_workflow: groupings is empty")
    gens = list(dict.fromkeys(gens))
    if not 0 < identity <= 1:
        raise ValueError(f"identity must be in (0, 1], got {identity}")
    if order not in ("max_mutation", "tips"):
        raise ValueError(f"order must be 'max_mutation' or 'tips', got {order!r}")
    if method not in ("parsimony", "nj"):
        raise ValueError(f"method must be 'parsimony' or 'nj', got {method!r}")
    gen = as_rng(rng)
    seed = int(gen.integers(0, 2**63 - 1))
    if legacy:
        structure_kw = {"pad_edges": False, **dict(structure_kw or {})}
    air, gex, workflow = _unpack(data, gex)
    from spaclone.bcr.workflow import _auto_floor, _filter_counts, _floor_message, _record_level

    meta = getattr(data, "meta", None)
    if workflow is not None:
        wp = getattr(workflow, "params", None)
        meta = {"level": wp.get("record_level")} if isinstance(wp, Mapping) else None
    level = _record_level(meta if isinstance(meta, Mapping) else None, air)
    min_reads, min_reads_rule = _auto_floor(min_reads, level, per_unit=1, contig=2, name="min_reads")
    min_reads = 0 if min_reads is None else min_reads
    if hotspots is None and workflow is not None:
        hotspots = getattr(workflow, "csr", None)
    rec, counts = _select_records(air, locus=locus.upper(), min_reads=min_reads, reads_key=reads_key,
                                  locus_key=locus_key)  # fmt: skip
    if not len(rec):
        steps = [("input", counts["records_in"]), (locus.upper(), counts[f"records_{locus.upper()}"]),
                 ("valid junction", counts["valid_junction"]), ("productive", counts["productive"]),
                 (f"reads >= {min_reads:g}", len(rec))]  # fmt: skip
        raise ValueError(
            "lineage_workflow: no heavy-chain record passes the junction, productive and read filters (records left: "
            f"{_filter_counts(steps)}; min_reads={min_reads:g}, {min_reads_rule}); check locus=, or lower "
            "min_reads (1 for molecule-level records)"
        )
    if min_reads:
        msg = _floor_message("lineage_workflow", min_reads, counts["productive"] - len(rec), counts["productive"],
                             air[reads_key] if reads_key in air.columns else None, level)  # fmt: skip
        if msg is not None:
            warnings.warn(msg, LineageWorkflowWarning, stacklevel=2)
    rec = _groupings(rec, identity)  # clones always from the input calls (the archive's TRUST4 groupings)
    if igblast:
        rec = _apply_igblast(rec, igblast)
        germline_start_base = 1
    # ---- isotypes
    from spaclone.bcr.isotype import isotype as _isotype

    if S.C_CALL in rec.columns:
        iso = _isotype(rec, species=None, locus_key=locus_key)["isotype"].astype("string").fillna("unknown")
        rec["isotype"] = iso.to_numpy(dtype=object)
    else:
        rec["isotype"] = "unknown"
    # ---- loci (exact clones, cell scale)
    from spaclone.pp.loci import merge_loci

    loci, member = merge_loci(rec.assign(**{S.CLONE: rec["clone_exact"]}), radius_um=loci_radius_um,
                              reads_key=reads_key, locus_key=locus_key)  # fmt: skip
    rec[S.LOCUS_ID] = member.to_numpy()
    loci = loci.rename(columns={S.CLONE: "clone_exact"})
    # ---- tissue structures and locations
    maps: dict[str, Any] = dict.fromkeys(
        ("grid", "tissue", "b_rich", "t_rich", "compartments", "hotspots", "share", "share_eligible", "rotation",
         "frame")
    )  # fmt: skip
    types = dict(COMPARTMENT_TYPES if compartment_types is None else compartment_types)
    comp = None
    comp_from = None
    if compartments is None and gex is None:
        warnings.warn(
            "lineage_workflow: no transcriptome, so no compartments: tip locations are only 'CSR hotspot' or "
            "'unknown' (no T zone, red pulp or follicle). Pass gex=<AnnData with a cell-type column> (or "
            "compartments=<Raster>), or compartments=False to run without them",
            LineageWorkflowWarning,
            stacklevel=2,
        )
    if compartments is None and gex is not None:
        st = _structures(gex, cell_type_key=cell_type_key, types=types, pixel_um=pixel_um,
                         structure_kw=structure_kw or {}, receptor_xy=rec[[S.X, S.Y]].to_numpy(dtype=float))  # fmt: skip
        maps.update({k: st[k] for k in ("grid", "tissue", "b_rich", "t_rich", "compartments", "rotation", "frame")})
        comp = st["compartments"]
        comp_from = "gex"
    elif compartments is not None and compartments is not False:
        comp = compartments
        maps["compartments"] = comp
        maps["grid"] = comp.grid
        comp_from = "given"
    hot = _raster_from(hotspots)
    maps["hotspots"] = hot
    maps["share"], maps["share_eligible"] = _share_layers(hotspots)
    if maps["grid"] is None and hot is not None:
        maps["grid"] = hot.grid
    lx, ly = loci[S.X].to_numpy(dtype=float), loci[S.Y].to_numpy(dtype=float)
    hot_lab = hot.sample(lx, ly, fill=0) if hot is not None else np.zeros(len(loci))
    loci["hotspot"] = np.nan_to_num(np.asarray(hot_lab, dtype=float), nan=0.0).astype(np.int64)
    if comp is not None:
        loci["compartment"] = comp.sample_categories(lx, ly, fill="red pulp")  # off the grid: outside the B-rich area
    else:
        loci["compartment"] = "unknown"
    loci["location"] = np.where(loci["hotspot"] > 0, "CSR hotspot", loci["compartment"].astype(object))
    rec["location"] = loci["location"].to_numpy()[rec[S.LOCUS_ID].to_numpy()]
    loc_classes = ["CSR hotspot"] if hot is not None else []
    if comp is not None:
        cats = [str(c) for c in (getattr(comp, "categories", None) or ())]
        loc_classes += [c for c in LOCATIONS[1:] if c in cats or c == "red pulp"]  # off the grid = red pulp
        loc_classes += [c for c in cats if c not in loc_classes]
    else:
        loc_classes.append("unknown")
    # ---- register parts of every record
    parsed = _parse_all(rec, germline_start_base)
    ok = (
        np.array([p.ok for p in parsed], dtype=bool)
        & rec[S.V_CALL].notna().to_numpy()
        & rec[S.J_CALL].notna().to_numpy()
    )
    if "_igblast_ok" in rec.columns:
        ok &= rec["_igblast_ok"].to_numpy(dtype=bool)
    lead = np.array([p.lead_missing if p.ok else -1 for p in parsed], dtype=np.int64)
    whole = ok & (lead >= 0) & (lead <= max_lead_missing)
    frame_ok = np.array([p.in_frame for p in parsed], dtype=bool)
    # productive check (IgBLAST-like) for full-length records; truncated records only support variants
    rec["usable"] = ok & (frame_ok | ~whole if check_frame else True)
    rec["lead_missing"] = lead
    rec["full_length"] = rec["usable"].to_numpy() & whole
    rec["in_frame"] = frame_ok
    vk = "v_call_igblast" if "v_call_igblast" in rec.columns else S.V_CALL
    jk = "j_call_igblast" if "j_call_igblast" in rec.columns else S.J_CALL
    v_all = rec[vk].astype("string").str.split(",").str[0].str.strip().to_numpy(dtype=object, na_value=None)
    j_all = rec[jk].astype("string").str.split(",").str[0].str.strip().to_numpy(dtype=object, na_value=None)
    rec["v_allele"], rec["j_allele"] = v_all, j_all
    vtab, jtab, cys_of = _germline_tables(parsed, v_all, j_all)
    own_mm = _own_mismatches(parsed, v_all, vtab)
    arr = _Arrays(rec, parsed, reads_key)
    # ---- per grouping
    trees, tips_out, clones_out, sel_rows, mem_rows = {}, {}, {}, [], []
    from spaclone.bcr.lineage import build_trees, tip_table

    for gk in gens:
        cl, tp = _grouping_tips(
            rec,
            arr,
            ckey=f"clone_{gk}",
            vtab=vtab,
            jtab=jtab,
            cys_of=cys_of,
            own_mm=own_mm,
            misfit_mismatches=misfit_mismatches,
            noise_min_spots=noise_min_spots,
            noise_min_reads=noise_min_reads,
            min_distinct=min_distinct,
            min_sites=min_sites,
            reads_key=reads_key,
            tag=gk,
            legacy=legacy,
        )
        clones_out[gk] = cl
        if len(tp):
            tr = build_trees(tp, clone_key="tree_id", id_key="tip_id", sequence_key="sequence",
                             germline_key="germline", traits=tuple(traits), method=method, n_starts=n_starts,
                             n_ratchet=n_ratchet, rng=np.random.SeedSequence([seed, 1 if gk == "exact" else 2]), n_jobs=n_jobs)  # fmt: skip
            tt = tip_table(tr, clone_key="tree_id", id_key="tip_id", sequence_key="sequence", germline_key="germline",
                           keep=[])  # fmt: skip
            depth = tt.set_index("tip_id")["depth"]
            lab = tr.tips.set_index("tip_id")["tree_label"] if tr.tips is not None else pd.Series(dtype="string")
            tp["tree_label"] = tp["tip_id"].map(lab).astype("string")
            tp["depth"] = tp["tip_id"].map(depth).astype(float)
        else:
            from spaclone.bcr.lineage import LineageTrees

            tr = LineageTrees({})
            tp["tree_label"] = pd.Series(dtype="string")
            tp["depth"] = pd.Series(dtype=float)
        trees[gk] = tr
        tips_out[gk] = tp
        s_rows, m_rows = _selection(tp, tr, gk, min_tips=min_tips, max_trees=max_trees, order=order)
        sel_rows.extend(s_rows)
        mem_rows.extend(m_rows)
    selection = pd.DataFrame(
        sel_rows,
        columns=[
            "grouping",
            "tree_id",
            "n_tips",
            "n_tree_tips",
            "n_loci",
            "max_mutations",
            "median_mutations",
            "n_exact_clones",
            "eligible",
            "rank",
            "drawn",
        ],  # fmt: skip
    )
    if len(selection):
        selection["rank"] = pd.array(selection["rank"], dtype="Int64")
    members = pd.DataFrame(mem_rows, columns=["tree_id", "exact_clone", "junction_aa", "n_tips"])
    keep_rec = [c for c in rec.columns if c not in ("sequence", "sequence_alignment", "germline_alignment",
                                                     "v_sequence_alignment", "v_germline_alignment")]  # fmt: skip
    params = {
        "groupings": tuple(gens),
        "identity": identity,
        "min_tips": min_tips,
        "max_trees": max_trees,
        "order": order,
        "method": method,
        "min_reads": min_reads,
        "locus": locus,
        "loci_radius_um": loci_radius_um,
        "max_lead_missing": max_lead_missing,
        "noise_min_spots": noise_min_spots,
        "noise_min_reads": noise_min_reads,
        "misfit_mismatches": misfit_mismatches,
        "min_distinct": min_distinct,
        "min_sites": min_sites,
        "traits": tuple(traits),
        "compartment_types": types,
        "structure_kw": dict(structure_kw or {}),
        "igblast": bool(igblast),
        "check_frame": check_frame,
        "legacy": legacy,
        "n_starts": n_starts,
        "n_ratchet": n_ratchet,
        "rng": rng_entropy(gen),
        "tree_seed": seed,
        "counts": counts,
        "min_reads_rule": min_reads_rule,
        "record_level": level,
        "compartments_from": comp_from,
        "location_classes": tuple(loc_classes),
        "hotspots": type(hotspots).__name__ if hotspots is not None else None,
        "runtime_s": round(time.time() - t_start, 2),
    }
    return LineageWorkflowResult(
        groupings=tuple(gens),
        trees=trees,
        tips=tips_out,
        clones=clones_out,
        selection=selection,
        members=members,
        records=rec[keep_rec],
        loci=loci,
        maps=maps,
        params=params,
    )


def _apply_igblast(rec: pd.DataFrame, igblast: bool | Mapping[str, Any]) -> pd.DataFrame:
    """Replace the V alignments by IgBLAST's and add its V / J allele calls (``v_call_igblast``, ``j_call_igblast``).

    Records IgBLAST calls unproductive, or whose junction differs from the input junction, keep no alignment and are
    not usable, as in the archive. Clones stay defined by the input calls.
    """
    from spaclone.bcr.shm import run_igblast

    if "sequence" not in rec.columns:
        raise KeyError("lineage_workflow(igblast=True) needs the full contig sequence in column 'sequence'")
    kw = dict(igblast) if isinstance(igblast, Mapping) else {}
    seqs = dict(zip(rec["sequence_id"].astype(str), rec["sequence"].astype(str), strict=True))
    ig = run_igblast(seqs, **kw)
    ig["sequence_id"] = ig["sequence_id"].astype(str)
    ig = ig.drop_duplicates("sequence_id").set_index("sequence_id").reindex(rec["sequence_id"].astype(str))
    prod = ig["productive"].astype("string").str.upper().isin(["T", "TRUE"]).fillna(False).to_numpy(dtype=bool)
    same = ig["junction"].astype("string").str.upper() == rec[S.JUNCTION].astype("string").str.upper().to_numpy()
    ok = prod & same.fillna(False).to_numpy(dtype=bool)
    out = rec.drop(columns=[c for c in ("v_cigar", "j_cigar") if c in rec.columns]).copy()
    for c in ("v_sequence_alignment", "v_germline_alignment"):
        out[c] = np.where(ok, ig[c].astype("string").to_numpy(dtype=object, na_value=None), None)
    for c in ("v_germline_start", "v_sequence_start", "j_sequence_start", "j_sequence_end"):
        out[c] = np.where(ok, pd.to_numeric(ig[c], errors="coerce").to_numpy(dtype=float), np.nan)
    first = lambda c: ig[c].astype("string").str.split(",").str[0].to_numpy(dtype=object, na_value=None)
    out["v_call_igblast"] = np.where(ok, first("v_call"), None)
    out["j_call_igblast"] = np.where(ok, first("j_call"), None)
    out["_igblast_ok"] = ok
    out.attrs = dict(rec.attrs)
    return out


def _parse_all(rec: pd.DataFrame, germline_start_base: Literal[0, 1, "auto"]) -> list[_Parsed]:
    """Register parts of every record (see :func:`_parse`)."""
    n = len(rec)
    cols = set(rec.columns)
    v_seq = rec["v_sequence_alignment"].astype("string").to_numpy(dtype=object, na_value=None)
    v_germ = rec["v_germline_alignment"].astype("string").to_numpy(dtype=object, na_value=None)
    g0 = pd.to_numeric(rec["v_germline_start"], errors="coerce").to_numpy(dtype=float)
    base = germline_start_base
    if base == "auto":  # AIRR is 1-based; TRUST4 writes 0-based starts (complete V genes start at 0)
        base = 0 if (np.nan_to_num(g0, nan=1.0) == 0).any() else 1
    g0 = g0 - float(base)
    jx = rec[S.JUNCTION].astype("string").to_numpy(dtype=object, na_value=None)
    if "sequence" in cols:
        contigs = rec["sequence"].astype("string").str.upper().to_numpy(dtype=object, na_value=None)
    else:  # no contig: the ungapped aligned sequence stands in for it
        contigs = rec["sequence_alignment"].astype("string").str.replace("-", "", regex=False).str.upper()
        contigs = contigs.to_numpy(dtype=object, na_value=None)
    if {"v_sequence_start", "j_sequence_start", "j_sequence_end"} <= cols and "sequence" in cols:  # AIRR, 1-based
        v_start = pd.to_numeric(rec["v_sequence_start"], errors="coerce").fillna(0).to_numpy() - 1
        j_start = pd.to_numeric(rec["j_sequence_start"], errors="coerce").fillna(0).to_numpy() - 1
        j_end = pd.to_numeric(rec["j_sequence_end"], errors="coerce").fillna(-1).to_numpy()
    elif {"v_cigar", "j_cigar"} <= cols and "sequence" in cols:  # TRUST4: CIGARs on the contig
        v_start = _cigar_lead(rec["v_cigar"])
        j_start = _cigar_lead(rec["j_cigar"])
        j_len = (
            rec["j_cigar"].astype("string").str.findall(r"(\d+)[MI]")
            .map(lambda v: sum(int(x) for x in v) if isinstance(v, list) else 0, na_action="ignore")
            .fillna(0).to_numpy(dtype=np.int64)
        )  # fmt: skip
        j_end = np.where(j_start >= 0, j_start + j_len, -1)
    else:
        v_start = np.zeros(n, dtype=np.int64)
        j_start = np.full(n, -1, dtype=np.int64)
        j_end = np.full(n, -1, dtype=np.int64)
    out = []
    for k in range(n):
        if contigs[k] is None or v_seq[k] is None or v_germ[k] is None or jx[k] is None or not np.isfinite(g0[k]):
            out.append(_Parsed(False))
            continue
        out.append(
            _parse(
                contigs[k].encode("ascii", "replace"),
                v_seq[k],
                v_germ[k],
                int(g0[k]),
                int(v_start[k]),
                jx[k],
                int(j_start[k]),
                int(j_end[k]),
            )  # fmt: skip
        )
    return out


class _Arrays:
    """Per-record arrays shared by the groupings (computed once)."""

    def __init__(self, rec: pd.DataFrame, parsed: list[_Parsed], reads_key: str) -> None:
        self.parsed = parsed
        self.n = len(rec)
        self.usable = rec["usable"].to_numpy(dtype=bool)
        self.full = rec["full_length"].to_numpy(dtype=bool)
        self.reads = pd.to_numeric(rec[reads_key], errors="coerce").fillna(0).to_numpy(dtype=float)
        self.spot = pd.factorize(rec[S.SPOT].astype(str))[0].astype(np.int64)
        self.locus = rec[S.LOCUS_ID].to_numpy(dtype=np.int64)
        self.iso = pd.Index([*TIP_ISOTYPES, "unknown"]).get_indexer(rec["isotype"].astype(str))
        self.iso[self.iso < 0] = len(TIP_ISOTYPES)
        self.v = rec["v_allele"].to_numpy(dtype=object)
        self.j = rec["j_allele"].to_numpy(dtype=object)
        self.vcode = pd.factorize(pd.Series(self.v, dtype=object), sort=True)[0]
        self.jcode = pd.factorize(pd.Series(self.j, dtype=object), sort=True)[0]
        self.sid = rec["sequence_id"].astype(str).to_numpy()
        self.jaa_code, self.jaa_names = pd.factorize(rec[S.JUNCTION_AA].astype(str).to_numpy(), sort=True)
        self.exact_code = pd.factorize(rec["clone_exact"].astype(str).to_numpy(), sort=True)[0]
        self.reg_len = np.array([_reg_len(p) for p in parsed], dtype=np.int64)
        self.location = rec["location"].astype(str).to_numpy()


def _grouping_tips(
    rec: pd.DataFrame,
    arr: _Arrays,
    *,
    ckey: str,
    vtab: dict[str, np.ndarray],
    jtab: dict[str, np.ndarray],
    cys_of: dict[str, int],
    own_mm: np.ndarray,
    misfit_mismatches: int | None,
    noise_min_spots: int,
    noise_min_reads: float,
    min_distinct: int,
    min_sites: int,
    reads_key: str,
    tag: str,
    legacy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clone table and tips of the selected clones of one grouping (archive L4 ``run_grouping``)."""
    parsed = arr.parsed
    misfit_all = np.zeros(arr.n, dtype=bool)
    mut_all = np.full(arr.n, np.nan)
    seq_all = np.full(arr.n, None, dtype=object)
    codes_c, names = pd.factorize(rec[ckey].astype(object), sort=True)
    use = np.flatnonzero(arr.usable & (codes_c >= 0))
    order = use[np.argsort(codes_c[use], kind="stable")]
    bounds = np.flatnonzero(np.diff(codes_c[order])) + 1
    rows, tips = [], []
    for idx in np.split(order, bounds) if len(order) else []:
        full = arr.full[idx]
        if not full.any():
            continue
        clone = names[codes_c[idx[0]]]
        v_top = _tied_top(arr.vcode[idx[full]])
        j_top = _tied_top(arr.jcode[idx[full]])
        cand = idx[full & np.isin(arr.vcode[idx], v_top) & np.isin(arr.jcode[idx], j_top)]
        if not len(cand):
            continue
        # representative (Change-O): the longest alignment, then the alphabetically first record id
        rep = int(min(cand, key=lambda i: (-arr.reg_len[i], arr.sid[i])))
        pr = parsed[rep]
        va, ja = str(arr.v[rep]), str(arr.j[rep])
        cys = cys_of.get(va, pr.cys)
        lj = len(pr.junction)
        lf = int(max(len(parsed[i].fbase) for i in idx))
        germ = _germline(vtab.get(va), jtab.get(ja), cys, lj, lf, pr.n_v, pr.n_j)
        vwin = cys + 3  # V positions up to and including the conserved cysteine (IMGT 1-312)
        codes = np.stack([_register(parsed[i], cys, lj, lf, legacy) for i in idx])
        misfit = np.zeros(len(idx), dtype=bool)
        if misfit_mismatches is not None:
            mc = _mismatch(codes[:, :vwin], germ[None, :vwin])
            mo = own_mm[idx]
            misfit = np.isfinite(mo) & (mo <= mc - misfit_mismatches)
        misfit_all[idx] = misfit
        keep = ~misfit
        if not (full & keep).any():
            continue
        idx_k, full_k = idx[keep], full[keep]
        cleaned = _clean(codes[keep], arr.spot[idx_k], arr.reads[idx_k], germ, noise_min_spots, noise_min_reads)
        cf = cleaned[full_k]
        gv = germ[:vwin]
        mv = (cf[:, :vwin] != gv[None, :]) & (gv[None, :] < 4) & (cf[:, :vwin] < 4)
        mut = mv.sum(1)
        ii, pp = np.nonzero(mv)
        n_sites = int(np.unique(pp.astype(np.int64) * 4 + cf[ii, pp]).size)
        seqs = np.array(["".join(_DEC[r]) for r in cf], dtype=object)
        n_distinct = int(len(set(seqs)))
        fidx = idx_k[full_k]
        seq_all[fidx] = seqs
        mut_all[fidx] = mut
        iso_n = np.bincount(arr.iso[fidx], minlength=len(TIP_ISOTYPES) + 1)
        _, first_of_locus = np.unique(arr.locus[fidx], return_index=True)
        loc_u, loc_c = np.unique(arr.location[fidx][first_of_locus], return_counts=True)
        loc_n = dict(zip(loc_u.tolist(), loc_c.tolist(), strict=True))
        selected = n_distinct >= min_distinct and n_sites >= min_sites
        jaa, jn = np.unique(arr.jaa_code[idx], return_counts=True)
        rows.append(
            {
                "clone_id": clone,
                "v_gene": str(va.split("*")[0]),
                "j_gene": str(ja.split("*")[0]),
                "germline_v": va,
                "germline_j": ja,
                "junction_aa_main": str(arr.jaa_names[jaa[np.argmax(jn)]]),
                "junction_aa_n": len(jaa),
                "exact_clones": int(np.unique(arr.exact_code[idx]).size),
                "records_usable": len(idx),
                "records_full": int(full.sum()),
                "misfits_removed": int(misfit.sum()),
                "spots": int(np.unique(arr.spot[fidx]).size),
                "loci": int(len(first_of_locus)),
                "distinct_seqs": n_distinct,
                "mutation_sites": n_sites,
                "max_mutations": int(mut.max()),
                "median_mutations": float(np.median(mut)),
                **{f"n_{i}": int(iso_n[k]) for k, i in enumerate((*TIP_ISOTYPES, "unknown"))},
                **{f"loci_{k}": int(loc_n.get(k, 0)) for k in LOCATIONS},
                "selected": bool(selected),
            }
        )
        if not selected:
            continue
        germ_str = "".join(_DEC[germ])
        gf = rec.iloc[fidx].assign(_seq=seqs, _mut=mut)
        for (locus_id, sq), t in gf.groupby([S.LOCUS_ID, "_seq"], sort=True):
            r = t.sort_values(reads_key, ascending=False, kind="stable").iloc[0]
            tips.append(
                {
                    "tree_id": clone,
                    "tip_id": f"{tag[0]}{len(tips) + 1}",
                    "locus_id": int(locus_id),
                    "isotype": _tip_isotype(t["isotype"]),
                    "location": r["location"],
                    "mutations": int(t["_mut"].iloc[0]),
                    S.X: float(t[S.X].mean()),
                    S.Y: float(t[S.Y].mean()),
                    "n_records": len(t),
                    "n_spots": int(t[S.SPOT].nunique()),
                    "reads": float(pd.to_numeric(t[reads_key], errors="coerce").sum()),
                    "exact_clone": r["clone_exact"],
                    S.JUNCTION_AA: r[S.JUNCTION_AA],
                    "record_ids": ",".join(t["sequence_id"].astype(str)),
                    "sequence": sq,
                    "germline": germ_str,
                }
            )
    rec[f"misfit_{tag}"] = misfit_all
    rec[f"mutations_{tag}"] = mut_all
    rec[f"sequence_{tag}"] = seq_all
    cols = ["clone_id", "v_gene", "j_gene", "germline_v", "germline_j", "junction_aa_main", "junction_aa_n",
            "exact_clones", "records_usable", "records_full", "misfits_removed", "spots", "loci", "distinct_seqs",
            "mutation_sites", "max_mutations", "median_mutations", *[f"n_{i}" for i in (*TIP_ISOTYPES, "unknown")],
            *[f"loci_{k}" for k in LOCATIONS], "selected"]  # fmt: skip
    cl = pd.DataFrame(rows, columns=cols)
    if len(cl):
        cl = cl.sort_values(["selected", "distinct_seqs", "clone_id"], ascending=[False, False, True])
        cl = cl.reset_index(drop=True)
    tip_cols = ["tree_id", "tip_id", "locus_id", "isotype", "location", "mutations", S.X, S.Y, "n_records", "n_spots",
                "reads", "exact_clone", S.JUNCTION_AA, "record_ids", "sequence", "germline"]  # fmt: skip
    return cl, pd.DataFrame(tips, columns=tip_cols)


def _reg_len(p: _Parsed) -> int:
    return len(p.vbase) + len(p.junction) + len(p.fbase) if p.ok else -1


def _germline(vg: np.ndarray | None, jg: np.ndarray | None, cys: int, lj: int, lf: int, n_v: int,
              n_j: int) -> np.ndarray:  # fmt: skip
    """Clonal germline codes: V germline, V- and J-templated junction ends, D / N masked, J germline (FWR4)."""
    out = np.full(cys + lj + lf, 4, dtype=np.int8)
    nv = min(max(n_v, 0), lj)
    if vg is not None:
        m = min(cys + nv, len(vg))
        out[:m] = vg[:m]
        if m < cys + nv:
            out[m : cys + nv] = 4
    if jg is not None:
        q = np.arange(max(lj - max(n_j, 0), nv), lj)
        k = q - lj + _J_SHIFT
        ok = (k >= 0) & (k < len(jg))
        out[cys + q[ok]] = jg[k[ok]]
        f = np.arange(lf)
        k = f + _J_SHIFT
        ok = k < len(jg)
        out[cys + lj + f[ok]] = jg[k[ok]]
    return out


def _register(p: _Parsed, cys: int, lj: int, lf: int, pad: bool = False) -> np.ndarray:
    """One record's codes in the clone register (V right-anchored at the junction start).

    With ``pad``, the V positions in front of the first sequenced one get code 5 (the archive's IMGT ``.`` padding,
    never filled).
    """
    out = np.full(cys + lj + lf, 4, dtype=np.int8)
    if not p.ok:
        return out
    idx = p.vpos - p.cys + cys
    ok = (idx >= 0) & (idx < cys)
    out[idx[ok]] = _LUT[p.vbase[ok]]
    if pad and ok.any():
        out[: max(0, int(idx[ok].min()))] = 5
    jx = np.frombuffer(p.junction, np.uint8)
    n = min(lj, len(jx))
    out[cys : cys + n] = _LUT[jx[:n]]
    nf = min(lf, len(p.fbase))
    out[cys + lj : cys + lj + nf] = _LUT[p.fbase[:nf]]
    return out


def _mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a != b) & (a < 4) & (b < 4)).sum(-1).astype(float)


def _selection(tp: pd.DataFrame, tr: Any, gk: str, *, min_tips: int, max_trees: int | None,
               order: str) -> tuple[list[dict], list[dict]]:  # fmt: skip
    """Per-tree summary, eligibility (>= min_tips tips), figure order and member exact clones."""
    rows, mem = [], []
    held = tp[tp["tree_label"].notna()] if len(tp) else tp
    for tid, t in held.groupby("tree_id", sort=True):
        if tid not in tr:
            continue
        rows.append(
            {
                "grouping": gk,
                "tree_id": tid,
                "n_tips": len(t),
                "n_tree_tips": int(tr[tid].n_tips - 1),
                "n_loci": int(t["locus_id"].nunique()),
                "max_mutations": int(t["mutations"].max()),
                "median_mutations": float(t["mutations"].median()),
                "n_exact_clones": int(t["exact_clone"].nunique()),
                "eligible": len(t) >= min_tips,
                "rank": None,
                "drawn": False,
            }
        )
        if gk == "related":
            m = (
                t.groupby(["exact_clone", S.JUNCTION_AA], sort=False)
                .size()
                .reset_index(name="n_tips")
                .sort_values(["n_tips", "exact_clone"], ascending=[False, True])
            )
            mem.extend({"tree_id": tid, **r} for r in m.to_dict("records"))
    el = [r for r in rows if r["eligible"]]
    if order == "max_mutation":
        el.sort(key=lambda r: (-r["max_mutations"], -r["n_tree_tips"], -r["n_tips"], str(r["tree_id"])))
    else:
        el.sort(key=lambda r: (-r["n_tips"], -r["max_mutations"], str(r["tree_id"])))
    for k, r in enumerate(el, 1):
        r["rank"] = k
        r["drawn"] = max_trees is None or k <= max_trees
    return rows, mem
