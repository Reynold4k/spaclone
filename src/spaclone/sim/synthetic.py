"""Synthetic spatial receptor data with known truth for every record.

Two generators:

* :func:`synthetic_chip`: a tissue section with cells (follicular B, germinal-centre B, plasma cells, T cells, other
  cells) in structures (a T zone, follicles with germinal centres, plasma-cell foci, red pulp). Every B-lineage cell
  carries one heavy and one light chain clone of its lineage (T cells one TRB and one TRA clone); germinal centres
  hold a few expanded lineages, plasma-cell foci are dominated by one or two (some shared with germinal centres).
  Each receptor molecule is either **genuine** (it sits on a spot inside its cell), **swapped** (its barcode was
  exchanged in library PCR, so it sits on the spot of another captured cDNA molecule anywhere on the chip, on or off
  the tissue) or a **sink** record (one of many chimeric copies collected by a single barcode-sink spot).

  The default (``legacy=False``) is the realistic chip of spaclone 1.0: captured cDNA per spot depends on the cell
  under it (B / plasma cells and the white pulp capture 1.5–1.8× more), constant-region transcripts sit only on the
  spots of the cells that express them (bursty, cell-to-cell heterogeneous), swapped molecules land on spots with
  probability proportional to the spot's cDNA (so receptor-rich cells attract more swaps, and some swaps land on a
  spot that holds their chain's constant-region RNA by chance), and an optional share of swaps is
  constant-region-homologous ("right isotype, wrong clone"). ``legacy=True`` is the 1.0.0.dev0 chip draw for draw.
* :func:`legacy_demo_chip`: the synthetic chip of the archived ``spaclone demo`` (``03_BCR_final_2026-10``),
  ported draw for draw: genuine IGL, swapped IGK and IGH, one genuine IGK clone ``"9999"`` in a 40-DNB disc and four
  barcode sinks. With ``rng=0`` it reproduces the archived demo input exactly and can write the same files.

Both return a :class:`~spaclone.io.SpatialAIR` whose ``air`` has the truth columns ``truth``
(``"genuine"`` / ``"swapped"`` / ``"sink"``), ``true_cell``, ``true_x_um`` / ``true_y_um`` and the source cell's type
and compartment, and whose ``meta`` holds the truth tables (cells, lineages, clones, structures, sinks).

References
----------
Morris et al. 2019, doi:10.1002/sim.8086 (simulation studies with known truth); Young & Behjati 2020,
doi:10.1093/gigascience/giaa151 (ambient RNA in droplets, the analogue of barcode swapping); Raj et al. 2006,
doi:10.1371/journal.pbio.0040309 (transcriptional bursts); Kebschull & Zador 2015, doi:10.1093/nar/gkv717 (PCR
template switching between homologous templates).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone._utils import RngLike, as_rng, rng_entropy
from spaclone.io._container import SpatialAIR
from spaclone.io.airr import DNB_UM, finish, format_spots

__all__ = [
    "CDNA_BY_COMPARTMENT",
    "CDNA_BY_TYPE",
    "CONSTANT_PER_CELL",
    "MOUSE_CONSTANT_GENES",
    "legacy_demo_chip",
    "synthetic_chip",
]

logger = logging.getLogger("spaclone")

#: Constant-region call → mouse GEX gene symbol used by the generators.
MOUSE_CONSTANT_GENES: dict[str, str] = {
    "IGKC": "Igkc",
    "IGLC1": "Iglc1",
    "IGLC2": "Iglc2",
    "IGLC3": "Iglc3",
    "IGHM": "Ighm",
    "IGHD": "Ighd",
    "IGHG1": "Ighg1",
    "IGHG2B": "Ighg2b",
    "IGHG2C": "Ighg2c",
    "IGHG3": "Ighg3",
    "IGHA": "Igha",
    "TRAC": "Trac",
    "TRBC1": "Trbc1",
    "TRBC2": "Trbc2",
}

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
_CODON = {
    "A": "GCC",
    "C": "TGC",
    "D": "GAC",
    "E": "GAG",
    "F": "TTC",
    "G": "GGC",
    "H": "CAC",
    "I": "ATC",
    "K": "AAG",
    "L": "CTG",
    "M": "ATG",
    "N": "AAC",
    "P": "CCC",
    "Q": "CAG",
    "R": "CGC",
    "S": "AGC",
    "T": "ACC",
    "V": "GTG",
    "W": "TGG",
    "Y": "TAC",
    "*": "TAA",
}
#: Background GEX composition of a spot by the type of the cell covering it ("none" = no cell); legacy chip.
_GEX = {
    "none": {"Actb": 0.30, "Malat1": 0.25, "Hbb-bs": 0.20, "Lyz2": 0.10, "Cd74": 0.10, "Igkc": 0.03, "Jchain": 0.02},
    "other": {"Actb": 0.30, "Malat1": 0.25, "Hbb-bs": 0.20, "Lyz2": 0.12, "Cd74": 0.08, "Igkc": 0.03, "Jchain": 0.02},
    "B": {"Actb": 0.25, "Malat1": 0.20, "Cd74": 0.20, "Cd79a": 0.10, "Ms4a1": 0.08, "Igkc": 0.10, "Ighm": 0.05,
          "Ighd": 0.02},
    "GC B": {"Actb": 0.25, "Malat1": 0.20, "Cd74": 0.15, "Cd79a": 0.08, "Aicda": 0.07, "S1pr2": 0.05, "Igkc": 0.10,
             "Ighg1": 0.05, "Ighm": 0.05},
    "plasma": {"Jchain": 0.15, "Xbp1": 0.10, "Igkc": 0.35, "Igha": 0.10, "Ighg1": 0.10, "Ighm": 0.05, "Actb": 0.10,
               "Malat1": 0.05},
    "T": {"Actb": 0.30, "Malat1": 0.25, "Cd3e": 0.15, "Cd3d": 0.10, "Trac": 0.05, "Trbc2": 0.05, "Cd74": 0.05,
          "Igkc": 0.05},
}  # fmt: skip
#: Background GEX composition of the realistic chip: no constant-region gene (those come from the cells that express
#: them, see ``constant_per_cell``).
_GEX_V2 = {
    "none": {"Actb": 0.30, "Malat1": 0.28, "Hbb-bs": 0.25, "Lyz2": 0.10, "Cd74": 0.07},
    "other": {"Actb": 0.30, "Malat1": 0.25, "Hbb-bs": 0.22, "Lyz2": 0.15, "Cd74": 0.08},
    "B": {"Actb": 0.28, "Malat1": 0.22, "Cd74": 0.25, "Cd79a": 0.13, "Ms4a1": 0.12},
    "GC B": {"Actb": 0.28, "Malat1": 0.22, "Cd74": 0.18, "Cd79a": 0.10, "Aicda": 0.12, "S1pr2": 0.10},
    "plasma": {"Jchain": 0.35, "Xbp1": 0.25, "Actb": 0.20, "Malat1": 0.10, "Sdc1": 0.10},
    "T": {"Actb": 0.32, "Malat1": 0.28, "Cd3e": 0.20, "Cd3d": 0.15, "Cd74": 0.05},
}  # fmt: skip
#: Captured background cDNA per spot by tissue compartment, relative to the red pulp (realistic chip): the white pulp
#: (T zone, follicles, germinal centres) and plasma-cell foci capture 1.5–1.7× more (survey S3 §B, archive v2).
CDNA_BY_COMPARTMENT: dict[str, float] = {
    "T zone": 1.5,
    "follicle": 1.6,
    "germinal centre": 1.7,
    "plasma-cell focus": 1.6,
    "red pulp": 1.0,
}
#: Captured background cDNA per spot by the type of the cell covering it, on top of the compartment factor
#: (realistic chip): B-lineage and plasma cells capture more than T and other cells, spots without a cell less.
CDNA_BY_TYPE: dict[str, float] = {"plasma": 1.3, "GC B": 1.15, "B": 1.1, "T": 1.0, "other": 1.0, "none": 0.7}
#: Mean captured constant-region transcripts per cell and chain besides the receptor-library molecules (realistic
#: chip): the cell's other Ig / TCR mRNA, which the GEX library sees and the receptor library does not.
CONSTANT_PER_CELL: dict[str, float] = {"plasma": 30.0, "GC B": 8.0, "B": 4.0, "T": 2.0}
#: Cell-type probabilities by compartment.
_TYPES = ("B", "GC B", "plasma", "T", "other")
_TYPE_P = {
    "T zone": (0.10, 0.00, 0.00, 0.85, 0.05),
    "follicle": (0.80, 0.00, 0.00, 0.10, 0.10),
    "germinal centre": (0.00, 0.85, 0.00, 0.05, 0.10),
    "plasma-cell focus": (0.00, 0.00, 0.75, 0.00, 0.25),
    "red pulp": (0.07, 0.00, 0.03, 0.05, 0.85),
}
#: Heavy-chain isotype probabilities by B-lineage cell type.
_ISOTYPES = ("IGHM", "IGHD", "IGHG1", "IGHG2B", "IGHA")
_ISO_P = {
    "B": (0.70, 0.30, 0.0, 0.0, 0.0),
    "GC B": (0.30, 0.0, 0.50, 0.10, 0.10),
    "plasma": (0.30, 0.0, 0.30, 0.0, 0.40),
}
#: Mean receptor molecules per cell and chain.
_MOLECULES = {"plasma": 20.0, "GC B": 4.0, "B": 2.0, "T": 1.0, "other": 0.0}
SwapWeight = Literal["gex", "captured", "background"]


# ------------------------------------------------------------------------------------------------------- helpers


def _junction(rng: np.random.Generator, n: int, ends: np.ndarray, lo: int = 8, hi: int = 16) -> np.ndarray:
    """Random junctions ``C...W`` / ``C...F`` with ``lo`` to ``hi`` amino acids."""
    lengths = rng.integers(lo, hi + 1, n)
    out = np.empty(n, dtype=object)
    for i, (k, e) in enumerate(zip(lengths, ends, strict=True)):
        out[i] = "C" + "".join(_AA[rng.integers(0, 20, k - 2)]) + e
    return out


def _nt(aa: np.ndarray) -> np.ndarray:
    return np.array(["".join(_CODON[a] for a in s) for s in aa], dtype=object)


def _disc(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """``n`` points uniform in a disc of ``radius`` around the origin."""
    r = radius * np.sqrt(rng.random(n))
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _choice_rows(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """One category per row of the probability matrix ``p`` (rows sum to 1)."""
    c = np.cumsum(p, axis=1)
    u = rng.random(len(p))[:, None] * c[:, -1:]
    return (u > c).sum(axis=1)


def _draw_weighted(rng: np.random.Generator, weights: np.ndarray, n: int) -> np.ndarray:
    """``n`` indices drawn with replacement with probability proportional to ``weights`` (non-negative)."""
    cw = np.cumsum(np.asarray(weights, dtype=float))
    if n == 0:
        return np.zeros(0, np.int64)
    if not cw[-1] > 0:
        raise ValueError("no cDNA to place swapped molecules on")
    u = rng.random(n) * cw[-1]
    return np.minimum(np.searchsorted(cw, u, side="right"), len(cw) - 1).astype(np.int64)


# --------------------------------------------------------------------------------------------- synthetic_chip


def synthetic_chip(
    *,
    size_um: float = 600.0,
    tissue_radius_um: float = 250.0,
    spot_um: float = 2.0,
    cell_density: float = 0.01,
    cell_radius_um: float = 4.0,
    t_zone_radius_um: float = 70.0,
    follicle_width_um: float = 90.0,
    n_gc: int = 3,
    gc_radius_um: float = 40.0,
    gc_lineages: int = 4,
    n_pc_foci: int = 4,
    pc_radius_um: float = 25.0,
    pc_lineages: int = 2,
    gc_to_pc: float = 0.5,
    t_expanded: int = 3,
    t_expanded_share: float = 0.1,
    molecules_per_cell: Mapping[str, float] | None = None,
    genuine_fraction: float | Mapping[str, float] = 0.3,
    n_sinks: int = 4,
    sink_size: int = 150,
    p_det: float = 0.5,
    occupancy_in: float = 0.3,
    occupancy_out: float = 0.01,
    cdna_per_spot: float = 1.5,
    reads_genuine: float = 4.0,
    reads_swapped: float = 1.0,
    nonproductive_fraction: float = 0.02,
    light_lambda: float = 0.1,
    tcr: bool = True,
    umi: bool = True,
    library_id: str = "synthetic",
    cdna_by_type: Mapping[str, float] | None = None,
    cdna_by_compartment: Mapping[str, float] | None = None,
    cdna_shape: float = 0.4,
    constant_per_cell: Mapping[str, float] | None = None,
    constant_burst: float = 2.0,
    constant_cell_shape: float = 2.0,
    constant_detection: float | None = None,
    swap_weight: SwapWeight = "gex",
    homologous_switch: float = 0.0,
    swap_family_mean: float = 0.0,
    legacy: bool = False,
    rng: RngLike = None,
) -> SpatialAIR:
    """Simulate a spatial BCR / TCR section with known truth for every receptor record.

    Parameters
    ----------
    size_um
        Side of the square chip (µm); the spot grid covers it, on and off the tissue.
    tissue_radius_um
        Radius of the disc of tissue at the chip centre.
    spot_um
        Spot pitch (0.5 for Stereo-seq DNBs, 10 for beads). Positions are snapped to the grid; ``x_um = index ·
        spot_um``. GEX amounts are per spot, so they do not scale with the pitch.
    cell_density
        Cells per µm² of tissue (a Poisson number of cells, uniform in the disc).
    cell_radius_um
        Radius of a cell: a genuine molecule sits uniformly within it, and a spot belongs to the nearest cell centre
        within this radius (its *covering cell*; ``"none"`` if there is none).
    t_zone_radius_um, follicle_width_um
        The T zone (disc at the centre) and the follicle ring around it.
    n_gc, gc_radius_um, gc_lineages
        Germinal centres (discs in the follicle ring) and the lineages each one holds (Zipf weights 1/k).
    n_pc_foci, pc_radius_um, pc_lineages, gc_to_pc
        Plasma-cell foci near the tissue edge, the lineages of each, and the probability that a focus lineage is one
        of the germinal-centre lineages (the same clone in a germinal centre and a focus).
    t_expanded, t_expanded_share
        Expanded T-cell lineages and the share of T cells that belong to them (the rest are unique).
    molecules_per_cell
        Mean receptor-library molecules per cell and chain by cell type (Poisson); default plasma 20, GC B 4, B 2,
        T 1.
    genuine_fraction
        Probability that a molecule sits on its own cell (one value, or by locus). The rest are swapped.
    n_sinks, sink_size
        Barcode sinks (half on the tissue, half far off it) and the chimeric records each collects (clones drawn from
        all molecules). A sink spot holds no other cDNA and carries one GEX molecule of a constant gene (the trap of
        counting per record).
    p_det
        Probability that the GEX library detects a receptor molecule's own constant-region transcript (on the
        molecule's true spot, whether the record is genuine or swapped).
    occupancy_in, occupancy_out, cdna_per_spot
        Background (non-constant-region) GEX. Legacy chip: a spot holds cDNA with probability ``occupancy_in`` on
        the tissue (``occupancy_out`` off it), ``1 + Poisson(cdna_per_spot)`` molecules. Realistic chip: a tissue
        spot holds ``NegBin(mean = occupancy_in · (1 + cdna_per_spot) · cdna_by_type[type], shape = cdna_shape)``
        background molecules (``occupancy_out · (1 + cdna_per_spot)`` off the tissue), so the mean per spot equals
        the legacy one for a red-pulp cell; ``type`` and ``compartment`` factors from ``cdna_by_type`` and
        ``cdna_by_compartment``.
    reads_genuine, reads_swapped
        Reads per record: ``1 + Poisson(reads_genuine)`` for genuine, ``1 + Poisson(reads_swapped)`` for swapped and
        sink records (chimeras form late in PCR). Equal values make reads uninformative.
    nonproductive_fraction
        Share of clones whose junction carries a stop codon (``productive`` False).
    light_lambda
        Share of B lineages with a lambda (IGL) instead of a kappa (IGK) light chain.
    tcr
        Simulate TRA / TRB records of T cells.
    umi
        Give every molecule a random 10-nt UMI (``False`` leaves ``umi`` missing, as in meta tables without a molecule identifier).
    library_id
        Library label.
    cdna_by_type, cdna_by_compartment
        Realistic chip: factors of the captured background cDNA of a tissue spot, by the type of its covering cell
        (default :data:`CDNA_BY_TYPE`: plasma 1.3, GC B 1.15, B 1.1, T and other 1.0, no cell 0.7) and by its
        compartment (default :data:`CDNA_BY_COMPARTMENT`: T zone 1.5, follicle 1.6, germinal centre 1.7, plasma-cell
        focus 1.6, red pulp 1.0), so the white pulp holds 1.5–1.8× more cDNA per area than the red pulp; keys not
        given keep their default, and all-1 mappings remove the dependence.
    cdna_shape
        Realistic chip: gamma shape of the per-spot background amount (smaller = more spots without cDNA and more
        molecules on the occupied ones).
    constant_per_cell
        Realistic chip: mean captured constant-region transcripts per cell and chain besides its receptor-library
        molecules (default :data:`CONSTANT_PER_CELL`: plasma 40, GC B 8, B 4, T 2), of the cell's own constant genes
        (its isotype and light-chain class; Trac and the cell's Trbc gene for T cells). They lie only on the spots
        covered by the cell (a cell whose spots all belong to nearer cell centres carries none). Zeros remove them.
    constant_burst
        Realistic chip: mean transcripts per burst; a cell's constant-region transcripts come in
        ``Poisson(total / constant_burst)`` bursts of ``1 + Poisson(constant_burst - 1)`` molecules, each burst on
        one random spot of the cell (clustered within spots).
    constant_cell_shape
        Realistic chip: gamma shape of the cell-to-cell variation of constant-region expression (mean 1; larger =
        more uniform cells).
    constant_detection
        Realistic chip: probability that the GEX library detects one of those constant-region transcripts
        (default ``p_det``).
    swap_weight
        Realistic chip: where a swapped molecule lands. ``"gex"`` (default): on a spot drawn with probability
        proportional to its GEX molecules (background, detected constant-region transcripts and detected own
        transcripts of receptor molecules), i.e. on a random captured cDNA molecule as the GEX library sees it.
        ``"captured"``: proportional to every captured molecule, including those the GEX library missed (a stress
        test: the landing weight is then not fully observable, which no GEX-based chance model can reproduce).
        ``"background"``: proportional to background molecules only (the legacy behaviour).
    homologous_switch
        Realistic chip: share of swapped molecules whose template switch happened inside the constant region: the
        record keeps the acceptor's constant region and barcode and the donor's V(D)J, i.e. it lands on a spot drawn
        with probability proportional to the captured constant-region transcripts of its own locus there (receptor
        molecules and other constant-region transcripts, detected or not) and takes the acceptor's ``c_call``
        ("right isotype, wrong clone"). Molecule-level tests (same spot, own cell type) cannot see these swaps.
    swap_family_mean
        Realistic chip: swapped molecules of one clone come in chimera families of ``1 + Poisson(swap_family_mean)``
        molecules that share one landing spot (0: every swap lands independently).
    legacy
        Reproduce the spaclone 1.0.0.dev0 chip draw for draw (uniform cDNA, constant-region genes in the background
        composition of every spot, swaps on background molecules only); the realistic-chip arguments are ignored.
    rng
        Seed or generator (same seed ⇒ same chip).

    Returns
    -------
    SpatialAIR
        ``air``: one row per receptor molecule with the reader columns (``spot_id``, ``x_um``, ``y_um``, ``locus``,
        ``clone_id`` = exact ``locus|V|J|junction_aa``, ``junction_aa``, ``junction``, ``v_call``, ``j_call``,
        ``c_call``, ``productive``, ``consensus_count``, ``umi``, ``library_id``, ``x_src`` / ``y_src``), the spot's
        ``cell_type`` and ``in_tissue``, and the truth: ``truth`` (``"genuine"`` / ``"swapped"`` / ``"sink"``),
        ``lineage``, ``true_cell``, ``true_cell_type``, ``true_compartment``, ``true_structure``, ``true_x_um``,
        ``true_y_um``; the realistic chip adds ``swap_kind`` (``""`` for genuine and sink records, ``"cdna"`` for a
        swap onto a random cDNA molecule, ``"homologous"`` for a constant-region-homologous switch), ``family``
        (chimera family id of a swap, ``-1`` otherwise) and ``own_seen`` (the GEX library detected the molecule's own
        transcript at its true spot). ``gex``: every spot with GEX (on and off the tissue), counts in ``X``
        (cell-type genes and constant-region genes), ``obs`` ``in_tissue``, ``total_counts``, ``cell_type``,
        ``compartment``, ``library_id`` (realistic chip also ``captured_cdna``: every captured molecule of the spot,
        detected or not, a truth column); ``obsm["spatial"]`` in µm. ``meta``: ``params``, ``model``
        (``"realistic"`` / ``"legacy"``), the truth tables ``cells``, ``lineages``, ``clones``, ``structures``,
        ``sinks`` (DataFrames), ``truth_counts`` and ``p_det_given_occupied`` (share of genuine records whose spot
        is GEX-occupied that hold their own constant gene).

    Notes
    -----
    Realistic chip, in order: cells and receptor molecules as in the legacy chip; every grid spot gets its covering
    cell and background cDNA (by cell type); every receptor-carrying cell gets bursts of its constant-region
    transcripts on its own spots; the GEX library detects each constant-region transcript with
    ``constant_detection`` and each receptor molecule's own transcript with ``p_det``; swapped molecules then land
    by ``swap_weight`` (or ``homologous_switch``), and sinks are placed on spots without cDNA.

    Examples
    --------
    >>> sa = synthetic_chip(rng=0)
    >>> sorted(sa.air.truth.unique())
    ['genuine', 'sink', 'swapped']
    >>> old = synthetic_chip(rng=0, legacy=True)  # the 1.0.0.dev0 chip
    """
    if legacy:
        out = _chip_v1(
            size_um=size_um, tissue_radius_um=tissue_radius_um, spot_um=spot_um, cell_density=cell_density,
            cell_radius_um=cell_radius_um, t_zone_radius_um=t_zone_radius_um, follicle_width_um=follicle_width_um,
            n_gc=n_gc, gc_radius_um=gc_radius_um, gc_lineages=gc_lineages, n_pc_foci=n_pc_foci,
            pc_radius_um=pc_radius_um, pc_lineages=pc_lineages, gc_to_pc=gc_to_pc, t_expanded=t_expanded,
            t_expanded_share=t_expanded_share, molecules_per_cell=molecules_per_cell,
            genuine_fraction=genuine_fraction, n_sinks=n_sinks, sink_size=sink_size, p_det=p_det,
            occupancy_in=occupancy_in, occupancy_out=occupancy_out, cdna_per_spot=cdna_per_spot,
            reads_genuine=reads_genuine, reads_swapped=reads_swapped, nonproductive_fraction=nonproductive_fraction,
            light_lambda=light_lambda, tcr=tcr, umi=umi, library_id=library_id, rng=rng,
        )  # fmt: skip
        out.meta["model"] = "legacy"
        return out
    return _chip_v2(**{k: v for k, v in locals().items() if k != "legacy"})


def _chip_v1(
    *,
    size_um: float,
    tissue_radius_um: float,
    spot_um: float,
    cell_density: float,
    cell_radius_um: float,
    t_zone_radius_um: float,
    follicle_width_um: float,
    n_gc: int,
    gc_radius_um: float,
    gc_lineages: int,
    n_pc_foci: int,
    pc_radius_um: float,
    pc_lineages: int,
    gc_to_pc: float,
    t_expanded: int,
    t_expanded_share: float,
    molecules_per_cell: Mapping[str, float] | None,
    genuine_fraction: float | Mapping[str, float],
    n_sinks: int,
    sink_size: int,
    p_det: float,
    occupancy_in: float,
    occupancy_out: float,
    cdna_per_spot: float,
    reads_genuine: float,
    reads_swapped: float,
    nonproductive_fraction: float,
    light_lambda: float,
    tcr: bool,
    umi: bool,
    library_id: str,
    rng: RngLike,
) -> SpatialAIR:
    """The spaclone 1.0.0.dev0 chip (``synthetic_chip(legacy=True)``), unchanged draw for draw."""
    params = {k: v for k, v in locals().items() if k != "rng"}
    gen = as_rng(rng)
    for name, v in (("size_um", size_um), ("tissue_radius_um", tissue_radius_um), ("spot_um", spot_um)):
        if not (v > 0 and np.isfinite(v)):
            raise ValueError(f"{name} must be positive")
    if 2 * tissue_radius_um >= size_um:
        raise ValueError("the tissue disc must fit inside the chip (2 * tissue_radius_um < size_um)")
    if not 0 <= p_det <= 1:
        raise ValueError("p_det must be in [0, 1]")
    mpc = dict(_MOLECULES)
    mpc.update(molecules_per_cell or {})
    params["molecules_per_cell"] = mpc
    params["genuine_fraction"] = dict(genuine_fraction) if isinstance(genuine_fraction, Mapping) else genuine_fraction
    params["rng_entropy"] = rng_entropy(gen)
    c0 = size_um / 2
    nx = int(np.floor(size_um / spot_um)) + 1

    # ---- structures
    structures = [{"kind": "T zone", "structure": "T0", "x_um": c0, "y_um": c0, "radius_um": t_zone_radius_um}]
    ring = t_zone_radius_um + follicle_width_um / 2
    a0 = gen.uniform(0, 2 * np.pi)
    for k in range(n_gc):
        a = a0 + 2 * np.pi * k / max(n_gc, 1)
        structures.append(
            {"kind": "germinal centre", "structure": f"GC{k}", "x_um": c0 + ring * np.cos(a),
             "y_um": c0 + ring * np.sin(a), "radius_um": gc_radius_um}
        )  # fmt: skip
    edge = tissue_radius_um - pc_radius_um - 5
    for k in range(n_pc_foci):
        a = a0 + np.pi / max(n_gc, 1) + 2 * np.pi * k / max(n_pc_foci, 1)
        structures.append(
            {"kind": "plasma-cell focus", "structure": f"PC{k}", "x_um": c0 + edge * np.cos(a),
             "y_um": c0 + edge * np.sin(a), "radius_um": pc_radius_um}
        )  # fmt: skip
    st = pd.DataFrame(structures)

    def compartment_of(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d0 = np.hypot(xy[:, 0] - c0, xy[:, 1] - c0)
        comp = np.where(
            d0 <= t_zone_radius_um, "T zone", np.where(d0 <= ring + follicle_width_um / 2, "follicle", "red pulp")
        )
        comp = comp.astype(object)
        sid = np.full(len(xy), "", dtype=object)
        for r in st.itertuples():
            if r.kind == "T zone":
                continue
            inside = np.hypot(xy[:, 0] - r.x_um, xy[:, 1] - r.y_um) <= r.radius_um
            comp[inside] = r.kind
            sid[inside] = r.structure
        sid[comp == "T zone"] = "T0"
        return comp, sid

    # ---- cells
    n_cells = int(gen.poisson(cell_density * np.pi * tissue_radius_um**2))
    if n_cells == 0:
        raise ValueError("no cells: increase cell_density or tissue_radius_um")
    cxy = _disc(gen, n_cells, tissue_radius_um) + c0
    ccomp, cstruct = compartment_of(cxy)
    p = np.array([_TYPE_P[c] for c in ccomp]) if n_cells else np.zeros((0, len(_TYPES)))
    ctype = np.array(_TYPES, dtype=object)[_choice_rows(gen, p)] if n_cells else np.zeros(0, object)
    cell_ids = np.array([f"cell{i:05d}" for i in range(n_cells)], dtype=object)

    # ---- lineages (one heavy + one light clone each)
    lineages: list[dict[str, Any]] = []

    def new_lineage(kind: str, comp: str, structure: str) -> int:
        lineages.append({"lineage": f"L{len(lineages):05d}", "kind": kind, "compartment": comp, "structure": structure})
        return len(lineages) - 1

    gc_lin = {r.structure: [new_lineage("BCR", "germinal centre", r.structure) for _ in range(gc_lineages)]
              for r in st.itertuples() if r.kind == "germinal centre"}  # fmt: skip
    all_gc = [i for v in gc_lin.values() for i in v]
    pc_lin = {}
    for r in st.itertuples():
        if r.kind != "plasma-cell focus":
            continue
        ids = []
        for _ in range(pc_lineages):
            if all_gc and gen.random() < gc_to_pc:
                ids.append(int(gen.choice(all_gc)))
            else:
                ids.append(new_lineage("BCR", "plasma-cell focus", r.structure))
        pc_lin[r.structure] = ids
    t_lin = [new_lineage("TCR", "T zone", "T0") for _ in range(t_expanded if tcr else 0)]
    cell_lin = np.full(n_cells, -1, dtype=np.int64)

    def zipf_pick(options: list[int], n: int) -> np.ndarray:
        w = 1.0 / np.arange(1, len(options) + 1)
        return np.asarray(options)[gen.choice(len(options), n, p=w / w.sum())]

    for sid_, opts in list(gc_lin.items()) + list(pc_lin.items()):
        want = "GC B" if sid_.startswith("GC") else "plasma"
        m = (cstruct == sid_) & (ctype == want)
        if m.any() and opts:
            cell_lin[m] = zipf_pick(opts, int(m.sum()))
    if t_lin:
        m = np.flatnonzero(ctype == "T")
        exp = m[gen.random(len(m)) < t_expanded_share]
        cell_lin[exp] = zipf_pick(t_lin, len(exp))
    for i in np.flatnonzero((cell_lin < 0) & np.isin(ctype, ["B", "GC B", "plasma"] + (["T"] if tcr else []))):
        cell_lin[i] = new_lineage("TCR" if ctype[i] == "T" else "BCR", str(ccomp[i]), str(cstruct[i]))
    L = pd.DataFrame(lineages)
    n_lin = len(L)
    is_t = (L.kind == "TCR").to_numpy() if n_lin else np.zeros(0, bool)
    light_is_l = gen.random(n_lin) < light_lambda
    heavy_locus = np.where(is_t, "TRB", "IGH").astype(object)
    light_locus = np.where(is_t, "TRA", np.where(light_is_l, "IGL", "IGK")).astype(object)

    # clone sequences: one heavy and one light clone per lineage
    def calls(locus: np.ndarray, seg: str, n_genes: int) -> np.ndarray:
        g = gen.integers(1, n_genes + 1, len(locus))
        return np.array([f"{lc}{seg}{k}*01" for lc, k in zip(locus, g, strict=True)], dtype=object)

    loci = np.r_[heavy_locus, light_locus]
    ends = np.where(loci == "IGH", "W", "F")
    jaa = _junction(gen, len(loci), ends)
    bad = gen.random(len(loci)) < nonproductive_fraction
    for i in np.flatnonzero(bad):
        s = list(jaa[i])
        s[len(s) // 2] = "*"
        jaa[i] = "".join(s)
    vcall, jcall = calls(loci, "V", 30), calls(loci, "J", 4)
    vgene = np.array([v.split("*")[0] for v in vcall], dtype=object)
    jgene = np.array([j.split("*")[0] for j in jcall], dtype=object)
    clone_id = np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(loci, vgene, jgene, jaa, strict=True)], dtype=object)
    clones = pd.DataFrame(
        {
            "clone_id": clone_id,
            "locus": loci,
            "chain": np.r_[np.full(n_lin, "heavy"), np.full(n_lin, "light")],
            "lineage": np.r_[L.lineage.to_numpy(dtype=object), L.lineage.to_numpy(dtype=object)] if n_lin else [],
            "junction_aa": jaa,
            "junction": _nt(jaa),
            "v_call": vcall,
            "j_call": jcall,
            "productive": ~bad,
        }
    )

    # ---- molecules of every cell carrying a lineage
    carriers = np.flatnonzero(cell_lin >= 0)
    lam = np.array([mpc.get(str(t), 0.0) for t in ctype[carriers]])
    n_h, n_l = gen.poisson(lam), gen.poisson(lam)
    mol_cell = np.r_[np.repeat(carriers, n_h), np.repeat(carriers, n_l)]
    mol_clone = np.r_[np.repeat(cell_lin[carriers], n_h), n_lin + np.repeat(cell_lin[carriers], n_l)]
    n_mol = len(mol_cell)
    true_xy = cxy[mol_cell] + _disc(gen, n_mol, cell_radius_um) if n_mol else np.zeros((0, 2))
    iso = np.full(n_cells, "", dtype=object)
    for t, pr in _ISO_P.items():
        m = ctype == t
        iso[m] = np.array(_ISOTYPES, dtype=object)[gen.choice(len(_ISOTYPES), int(m.sum()), p=pr)]
    mol_locus = loci[mol_clone]
    c_call = np.empty(n_mol, dtype=object)
    c_call[mol_locus == "IGH"] = iso[mol_cell[mol_locus == "IGH"]]
    c_call[mol_locus == "IGK"] = "IGKC"
    c_call[mol_locus == "IGL"] = "IGLC1"
    c_call[mol_locus == "TRA"] = "TRAC"
    trb = np.flatnonzero(mol_locus == "TRB")
    c_call[trb] = np.where(gen.random(len(trb)) < 0.5, "TRBC1", "TRBC2")
    gf = genuine_fraction if isinstance(genuine_fraction, Mapping) else dict.fromkeys(S.ALL_LOCI, genuine_fraction)
    rho = np.array([float(gf.get(str(lc), 0.0)) for lc in mol_locus])
    genuine = gen.random(n_mol) < rho

    # ---- background GEX on the spot grid
    def snap(xy: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(xy / spot_um), 0, nx - 1).astype(np.int64)

    grid = np.arange(nx) * spot_um
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    lin_all = np.arange(nx * nx)
    in_t_all = np.hypot(gx.ravel() - c0, gy.ravel() - c0) <= tissue_radius_um
    occ = gen.random(nx * nx) < np.where(in_t_all, occupancy_in, occupancy_out)
    occ_lin = lin_all[occ]
    tree = cKDTree(cxy) if n_cells else None

    def type_at(lin: np.ndarray) -> np.ndarray:
        """Type of the cell covering each spot ('none' without a cell within the cell radius)."""
        out = np.full(len(lin), "none", dtype=object)
        if tree is None or len(lin) == 0:
            return out
        pts = np.column_stack([(lin % nx) * spot_um, (lin // nx) * spot_um])
        d, j = tree.query(pts, distance_upper_bound=cell_radius_um)
        ok = np.isfinite(d)
        out[ok] = ctype[j[ok]]
        return out

    genes = sorted({g for comp in _GEX.values() for g in comp} | set(MOUSE_CONSTANT_GENES.values()))
    gidx = {g: i for i, g in enumerate(genes)}
    occ_type = type_at(occ_lin)
    k_bg = 1 + gen.poisson(cdna_per_spot, len(occ_lin))
    bg_lin = np.repeat(occ_lin, k_bg)
    bg_type = np.repeat(occ_type, k_bg)
    bg_gene = np.empty(len(bg_lin), dtype=np.int64)
    for t, comp in _GEX.items():
        m = bg_type == t
        if m.any():
            names = list(comp)
            pr = np.array([comp[g] for g in names], float)
            bg_gene[m] = np.array([gidx[g] for g in names])[gen.choice(len(names), int(m.sum()), p=pr / pr.sum())]

    # ---- positions: genuine molecules on their cell, swapped ones on a random captured cDNA
    true_ij = snap(true_xy) if n_mol else np.zeros((0, 2), np.int64)
    true_lin = true_ij[:, 1] * nx + true_ij[:, 0]
    rec_lin = true_lin.copy()
    n_sw = int((~genuine).sum())
    if n_sw:
        if len(bg_lin) == 0:
            raise ValueError("no GEX spot to place swapped molecules on (occupancy is 0)")
        rec_lin[~genuine] = bg_lin[gen.integers(0, len(bg_lin), n_sw)]
    truth = np.where(genuine, "genuine", "swapped").astype(object)

    # ---- barcode sinks: chimeric copies of random molecules on single spots
    sink_rows = []
    if n_sinks and n_mol:
        d_all = np.hypot((lin_all % nx) * spot_um - c0, (lin_all // nx) * spot_um - c0)
        core = lin_all[(d_all <= 0.6 * tissue_radius_um) & ~occ]
        far = lin_all[(d_all >= tissue_radius_um + 0.25 * (c0 - tissue_radius_um)) & ~occ]
        n_on = (n_sinks + 1) // 2
        picks = [*gen.choice(core, n_on, replace=False), *gen.choice(far, n_sinks - n_on, replace=False)]
        for k, sl in enumerate(picks):
            src = gen.integers(0, n_mol, sink_size)
            sink_rows.append((k, int(sl), src))
    sink_src = np.concatenate([s for _, _, s in sink_rows]) if sink_rows else np.zeros(0, np.int64)
    sink_lin = np.concatenate([np.full(len(s), sl) for _, sl, s in sink_rows]) if sink_rows else np.zeros(0, np.int64)
    rec_src = np.r_[np.arange(n_mol), sink_src]
    rec_lin = np.r_[rec_lin, sink_lin]
    truth = np.r_[truth, np.full(len(sink_src), "sink", dtype=object)]
    n_rec = len(rec_src)

    # ---- own constant-region transcripts (at the source cell) and the sinks' trap molecules
    seen = gen.random(n_mol) < p_det
    own_lin = true_lin[seen]
    own_gene = np.array([gidx[MOUSE_CONSTANT_GENES[c]] for c in c_call[seen]], dtype=np.int64)
    trap_lin = np.array([sl for _, sl, _ in sink_rows], dtype=np.int64)
    trap_gene = np.array(
        [gidx[MOUSE_CONSTANT_GENES[c_call[s[0]]]] for _, _, s in sink_rows], dtype=np.int64
    ) if sink_rows else np.zeros(0, np.int64)  # fmt: skip
    all_lin = np.r_[bg_lin, own_lin, trap_lin]
    all_gene = np.r_[bg_gene, own_gene, trap_gene]
    key, cnt = np.unique(all_lin * len(genes) + all_gene, return_counts=True)
    s_lin, s_gene = key // len(genes), key % len(genes)
    spots, s_row = np.unique(s_lin, return_inverse=True)
    X = sp.csr_matrix((cnt.astype(np.float32), (s_row, s_gene)), shape=(len(spots), len(genes)))
    sx_, sy_ = spots % nx, spots // nx
    s_xy = np.column_stack([sx_ * spot_um, sy_ * spot_um])
    s_comp, _ = compartment_of(s_xy)
    s_in = np.hypot(s_xy[:, 0] - c0, s_xy[:, 1] - c0) <= tissue_radius_um
    s_comp = np.where(s_in, s_comp, "off tissue").astype(object)
    sids = format_spots(sx_, sy_)
    obs = pd.DataFrame(
        {
            "in_tissue": s_in,
            "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(float),
            "cell_type": pd.Categorical(type_at(spots)),
            "compartment": pd.Categorical(s_comp),
            S.LIBRARY: library_id,
        },
        index=pd.Index(sids, dtype=str),
    )
    gex = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    gex.obsm["spatial"] = s_xy

    # ---- receptor records
    src_clone = mol_clone[rec_src] if n_rec else np.zeros(0, np.int64)
    ix, iy = rec_lin % nx, rec_lin // nx
    rxy = np.column_stack([ix * spot_um, iy * spot_um])
    reads = 1 + gen.poisson(np.where(truth == "genuine", reads_genuine, reads_swapped))
    umis = (
        np.array(["".join(r) for r in np.array(list("ACGT"))[gen.integers(0, 4, (n_rec, 10))]], dtype=object)
        if umi and n_rec
        else np.full(n_rec, np.nan, dtype=object)
    )
    src_cell = mol_cell[rec_src] if n_rec else np.zeros(0, np.int64)
    cl = clones.iloc[src_clone]
    air = pd.DataFrame(
        {
            S.SPOT: format_spots(ix, iy),
            S.X: rxy[:, 0],
            S.Y: rxy[:, 1],
            S.LOCUS: cl["locus"].to_numpy(dtype=object),
            S.CLONE: cl["clone_id"].to_numpy(dtype=object),
            S.JUNCTION_AA: cl["junction_aa"].to_numpy(dtype=object),
            S.JUNCTION: cl["junction"].to_numpy(dtype=object),
            S.V_CALL: cl["v_call"].to_numpy(dtype=object),
            S.J_CALL: cl["j_call"].to_numpy(dtype=object),
            S.C_CALL: c_call[rec_src] if n_rec else np.zeros(0, object),
            S.PRODUCTIVE: cl["productive"].to_numpy(dtype=bool),
            S.READS: reads.astype(float),
            S.UMI: umis,
            S.LIBRARY: library_id,
            "x_src": ix,
            "y_src": iy,
            "cell_type": type_at(rec_lin),
            "in_tissue": np.hypot(rxy[:, 0] - c0, rxy[:, 1] - c0) <= tissue_radius_um,
            "truth": truth,
            "lineage": cl["lineage"].to_numpy(dtype=object),
            "true_cell": cell_ids[src_cell] if n_rec else np.zeros(0, object),
            "true_cell_type": ctype[src_cell] if n_rec else np.zeros(0, object),
            "true_compartment": ccomp[src_cell] if n_rec else np.zeros(0, object),
            "true_structure": cstruct[src_cell] if n_rec else np.zeros(0, object),
            "true_x_um": true_xy[rec_src, 0] if n_rec else np.zeros(0),
            "true_y_um": true_xy[rec_src, 1] if n_rec else np.zeros(0),
        }
    )

    # ---- truth tables
    cells = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_um": cxy[:, 0],
            "y_um": cxy[:, 1],
            "cell_type": ctype,
            "compartment": ccomp,
            "structure": cstruct,
            "lineage": np.where(cell_lin >= 0, L.lineage.to_numpy(dtype=object)[np.maximum(cell_lin, 0)], None)
            if n_lin
            else np.full(n_cells, None),
        }
    )
    L["heavy_clone"] = clone_id[:n_lin]
    L["light_clone"] = clone_id[n_lin:]
    L["n_cells"] = np.bincount(cell_lin[cell_lin >= 0], minlength=n_lin) if n_lin else []
    by_clone = air.groupby(S.CLONE).truth.value_counts().unstack(fill_value=0)
    clones = clones.join(by_clone.reindex(columns=["genuine", "swapped", "sink"], fill_value=0), on="clone_id")
    clones[["genuine", "swapped", "sink"]] = clones[["genuine", "swapped", "sink"]].fillna(0).astype(np.int64)
    clones["n_cells"] = np.r_[L["n_cells"].to_numpy(), L["n_cells"].to_numpy()] if n_lin else []
    sinks = pd.DataFrame(
        {
            S.SPOT: format_spots([sl % nx for _, sl, _ in sink_rows], [sl // nx for _, sl, _ in sink_rows]),
            S.X: [(sl % nx) * spot_um for _, sl, _ in sink_rows],
            S.Y: [(sl // nx) * spot_um for _, sl, _ in sink_rows],
            "in_tissue": [k < (n_sinks + 1) // 2 for k, _, _ in sink_rows],
            "n_records": [len(s) for _, _, s in sink_rows],
        }
    )
    gen_rec = air[air.truth == "genuine"]
    occ_spot = gen_rec[S.SPOT].isin(gex.obs_names).to_numpy()
    own = np.zeros(len(gen_rec), bool)
    if occ_spot.any():
        rows_ = gex.obs_names.get_indexer(gen_rec[S.SPOT].to_numpy()[occ_spot])
        cols_ = np.array([gidx[MOUSE_CONSTANT_GENES[c]] for c in gen_rec[S.C_CALL].to_numpy()[occ_spot]])
        own[occ_spot] = np.asarray(gex.X[rows_, cols_]).ravel() > 0
    meta: dict[str, Any] = {
        "reader": "sim.synthetic_chip",
        "technology": "synthetic",
        "library_id": library_id,
        "level": "molecule" if umi else "spot_clone",
        "molecule_level": True,
        "source_unit": f"synthetic spot ({spot_um:g} µm)",
        "um_per_unit": float(spot_um),
        "has_molecule_id": bool(umi),
        "has_read_counts": True,
        "params": params,
        "cells": cells,
        "lineages": L,
        "clones": clones,
        "structures": st,
        "sinks": sinks,
        "truth_counts": {f"{t}|{lc}": int(n) for (t, lc), n in air.groupby(["truth", S.LOCUS]).size().items()},
        "p_det": float(p_det),
        "p_det_given_occupied": float(own[occ_spot].mean()) if occ_spot.any() else float("nan"),
    }
    logger.info("synthetic_chip: %d cells, %d lineages, %d records, %d GEX spots", n_cells, n_lin, len(air), gex.n_obs)
    return finish(air, gex, meta)


def _chip_v2(
    *,
    size_um: float,
    tissue_radius_um: float,
    spot_um: float,
    cell_density: float,
    cell_radius_um: float,
    t_zone_radius_um: float,
    follicle_width_um: float,
    n_gc: int,
    gc_radius_um: float,
    gc_lineages: int,
    n_pc_foci: int,
    pc_radius_um: float,
    pc_lineages: int,
    gc_to_pc: float,
    t_expanded: int,
    t_expanded_share: float,
    molecules_per_cell: Mapping[str, float] | None,
    genuine_fraction: float | Mapping[str, float],
    n_sinks: int,
    sink_size: int,
    p_det: float,
    occupancy_in: float,
    occupancy_out: float,
    cdna_per_spot: float,
    reads_genuine: float,
    reads_swapped: float,
    nonproductive_fraction: float,
    light_lambda: float,
    tcr: bool,
    umi: bool,
    library_id: str,
    cdna_by_type: Mapping[str, float] | None,
    cdna_by_compartment: Mapping[str, float] | None,
    cdna_shape: float,
    constant_per_cell: Mapping[str, float] | None,
    constant_burst: float,
    constant_cell_shape: float,
    constant_detection: float | None,
    swap_weight: SwapWeight,
    homologous_switch: float,
    swap_family_mean: float,
    rng: RngLike,
) -> SpatialAIR:
    """The realistic chip of :func:`synthetic_chip` (``legacy=False``)."""
    params = {k: v for k, v in locals().items() if k != "rng"}
    gen = as_rng(rng)
    for name, v in (("size_um", size_um), ("tissue_radius_um", tissue_radius_um), ("spot_um", spot_um)):
        if not (v > 0 and np.isfinite(v)):
            raise ValueError(f"{name} must be positive")
    if 2 * tissue_radius_um >= size_um:
        raise ValueError("the tissue disc must fit inside the chip (2 * tissue_radius_um < size_um)")
    if not 0 <= p_det <= 1:
        raise ValueError("p_det must be in [0, 1]")
    p_const = p_det if constant_detection is None else float(constant_detection)
    if not 0 <= p_const <= 1:
        raise ValueError("constant_detection must be in [0, 1]")
    if swap_weight not in ("gex", "captured", "background"):
        raise ValueError(f"swap_weight must be 'gex', 'captured' or 'background', got {swap_weight!r}")
    if not 0 <= homologous_switch <= 1:
        raise ValueError("homologous_switch must be in [0, 1]")
    if not (cdna_shape > 0 and constant_burst >= 1 and constant_cell_shape > 0 and swap_family_mean >= 0):
        raise ValueError("cdna_shape and constant_cell_shape must be > 0, constant_burst >= 1, swap_family_mean >= 0")
    mpc = dict(_MOLECULES)
    mpc.update(molecules_per_cell or {})
    cmult = dict(CDNA_BY_TYPE)
    cmult.update(cdna_by_type or {})
    kmult = dict(CDNA_BY_COMPARTMENT)
    kmult.update(cdna_by_compartment or {})
    cpc = dict(CONSTANT_PER_CELL)
    cpc.update(constant_per_cell or {})
    params.update(
        molecules_per_cell=mpc,
        cdna_by_type=cmult,
        cdna_by_compartment=kmult,
        constant_per_cell=cpc,
        constant_detection=p_const,
    )
    params["genuine_fraction"] = dict(genuine_fraction) if isinstance(genuine_fraction, Mapping) else genuine_fraction
    params["rng_entropy"] = rng_entropy(gen)
    c0 = size_um / 2
    nx = int(np.floor(size_um / spot_um)) + 1

    # ---- structures
    structures = [{"kind": "T zone", "structure": "T0", "x_um": c0, "y_um": c0, "radius_um": t_zone_radius_um}]
    ring = t_zone_radius_um + follicle_width_um / 2
    a0 = gen.uniform(0, 2 * np.pi)
    for k in range(n_gc):
        a = a0 + 2 * np.pi * k / max(n_gc, 1)
        structures.append(
            {"kind": "germinal centre", "structure": f"GC{k}", "x_um": c0 + ring * np.cos(a),
             "y_um": c0 + ring * np.sin(a), "radius_um": gc_radius_um}
        )  # fmt: skip
    edge = tissue_radius_um - pc_radius_um - 5
    for k in range(n_pc_foci):
        a = a0 + np.pi / max(n_gc, 1) + 2 * np.pi * k / max(n_pc_foci, 1)
        structures.append(
            {"kind": "plasma-cell focus", "structure": f"PC{k}", "x_um": c0 + edge * np.cos(a),
             "y_um": c0 + edge * np.sin(a), "radius_um": pc_radius_um}
        )  # fmt: skip
    st = pd.DataFrame(structures)

    def compartment_of(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d0 = np.hypot(xy[:, 0] - c0, xy[:, 1] - c0)
        comp = np.where(
            d0 <= t_zone_radius_um, "T zone", np.where(d0 <= ring + follicle_width_um / 2, "follicle", "red pulp")
        )
        comp = comp.astype(object)
        sid = np.full(len(xy), "", dtype=object)
        for r in st.itertuples():
            if r.kind == "T zone":
                continue
            inside = np.hypot(xy[:, 0] - r.x_um, xy[:, 1] - r.y_um) <= r.radius_um
            comp[inside] = r.kind
            sid[inside] = r.structure
        sid[comp == "T zone"] = "T0"
        return comp, sid

    # ---- cells
    n_cells = int(gen.poisson(cell_density * np.pi * tissue_radius_um**2))
    if n_cells == 0:
        raise ValueError("no cells: increase cell_density or tissue_radius_um")
    cxy = _disc(gen, n_cells, tissue_radius_um) + c0
    ccomp, cstruct = compartment_of(cxy)
    p = np.array([_TYPE_P[c] for c in ccomp])
    ctype = np.array(_TYPES, dtype=object)[_choice_rows(gen, p)]
    cell_ids = np.array([f"cell{i:05d}" for i in range(n_cells)], dtype=object)

    # ---- lineages (one heavy + one light clone each)
    lineages: list[dict[str, Any]] = []

    def new_lineage(kind: str, comp: str, structure: str) -> int:
        lineages.append({"lineage": f"L{len(lineages):05d}", "kind": kind, "compartment": comp, "structure": structure})
        return len(lineages) - 1

    gc_lin = {r.structure: [new_lineage("BCR", "germinal centre", r.structure) for _ in range(gc_lineages)]
              for r in st.itertuples() if r.kind == "germinal centre"}  # fmt: skip
    all_gc = [i for v in gc_lin.values() for i in v]
    pc_lin = {}
    for r in st.itertuples():
        if r.kind != "plasma-cell focus":
            continue
        ids = []
        for _ in range(pc_lineages):
            if all_gc and gen.random() < gc_to_pc:
                ids.append(int(gen.choice(all_gc)))
            else:
                ids.append(new_lineage("BCR", "plasma-cell focus", r.structure))
        pc_lin[r.structure] = ids
    t_lin = [new_lineage("TCR", "T zone", "T0") for _ in range(t_expanded if tcr else 0)]
    cell_lin = np.full(n_cells, -1, dtype=np.int64)

    def zipf_pick(options: list[int], n: int) -> np.ndarray:
        w = 1.0 / np.arange(1, len(options) + 1)
        return np.asarray(options)[gen.choice(len(options), n, p=w / w.sum())]

    for sid_, opts in list(gc_lin.items()) + list(pc_lin.items()):
        want = "GC B" if sid_.startswith("GC") else "plasma"
        m = (cstruct == sid_) & (ctype == want)
        if m.any() and opts:
            cell_lin[m] = zipf_pick(opts, int(m.sum()))
    if t_lin:
        m = np.flatnonzero(ctype == "T")
        exp = m[gen.random(len(m)) < t_expanded_share]
        cell_lin[exp] = zipf_pick(t_lin, len(exp))
    for i in np.flatnonzero((cell_lin < 0) & np.isin(ctype, ["B", "GC B", "plasma"] + (["T"] if tcr else []))):
        cell_lin[i] = new_lineage("TCR" if ctype[i] == "T" else "BCR", str(ccomp[i]), str(cstruct[i]))
    L = pd.DataFrame(lineages, columns=["lineage", "kind", "compartment", "structure"])
    n_lin = len(L)
    is_t = (L.kind == "TCR").to_numpy() if n_lin else np.zeros(0, bool)
    light_is_l = gen.random(n_lin) < light_lambda
    heavy_locus = np.where(is_t, "TRB", "IGH").astype(object)
    light_locus = np.where(is_t, "TRA", np.where(light_is_l, "IGL", "IGK")).astype(object)

    def calls(locus: np.ndarray, seg: str, n_genes: int) -> np.ndarray:
        g = gen.integers(1, n_genes + 1, len(locus))
        return np.array([f"{lc}{seg}{k}*01" for lc, k in zip(locus, g, strict=True)], dtype=object)

    loci = np.r_[heavy_locus, light_locus]
    ends = np.where(loci == "IGH", "W", "F")
    jaa = _junction(gen, len(loci), ends)
    bad = gen.random(len(loci)) < nonproductive_fraction
    for i in np.flatnonzero(bad):
        s = list(jaa[i])
        s[len(s) // 2] = "*"
        jaa[i] = "".join(s)
    vcall, jcall = calls(loci, "V", 30), calls(loci, "J", 4)
    vgene = np.array([v.split("*")[0] for v in vcall], dtype=object)
    jgene = np.array([j.split("*")[0] for j in jcall], dtype=object)
    clone_id = np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(loci, vgene, jgene, jaa, strict=True)], dtype=object)
    clones = pd.DataFrame(
        {
            "clone_id": clone_id,
            "locus": loci,
            "chain": np.r_[np.full(n_lin, "heavy"), np.full(n_lin, "light")],
            "lineage": np.r_[L.lineage.to_numpy(dtype=object), L.lineage.to_numpy(dtype=object)] if n_lin else [],
            "junction_aa": jaa,
            "junction": _nt(jaa),
            "v_call": vcall,
            "j_call": jcall,
            "productive": ~bad,
        }
    )

    # ---- per-cell constant regions (isotype, light-chain class, TRBC gene)
    iso = np.full(n_cells, "", dtype=object)
    for t, pr in _ISO_P.items():
        m = ctype == t
        iso[m] = np.array(_ISOTYPES, dtype=object)[gen.choice(len(_ISOTYPES), int(m.sum()), p=pr)]
    trbc = np.where(gen.random(n_cells) < 0.5, "TRBC1", "TRBC2").astype(object)
    lin_of = np.maximum(cell_lin, 0)
    has_lin = cell_lin >= 0
    cell_heavy = np.full(n_cells, "", dtype=object)
    cell_light = np.full(n_cells, "", dtype=object)
    if n_lin:
        hl, ll = heavy_locus[lin_of], light_locus[lin_of]
        cell_heavy = np.where(has_lin & (hl == "IGH"), iso, np.where(has_lin & (hl == "TRB"), trbc, "")).astype(object)
        light_c = np.where(ll == "IGK", "IGKC", np.where(ll == "IGL", "IGLC1", "TRAC"))
        cell_light = np.where(has_lin, light_c, "").astype(object)

    # ---- receptor-library molecules of every cell carrying a lineage
    carriers = np.flatnonzero(has_lin)
    lam = np.array([mpc.get(str(t), 0.0) for t in ctype[carriers]])
    n_h, n_l = gen.poisson(lam), gen.poisson(lam)
    mol_cell = np.r_[np.repeat(carriers, n_h), np.repeat(carriers, n_l)]
    mol_clone = np.r_[np.repeat(cell_lin[carriers], n_h), n_lin + np.repeat(cell_lin[carriers], n_l)]
    mol_heavy = np.r_[np.ones(int(n_h.sum()), bool), np.zeros(int(n_l.sum()), bool)]
    n_mol = len(mol_cell)
    true_xy = cxy[mol_cell] + _disc(gen, n_mol, cell_radius_um) if n_mol else np.zeros((0, 2))
    mol_locus = loci[mol_clone] if n_mol else np.zeros(0, object)
    c_call = np.where(mol_heavy, cell_heavy[mol_cell], cell_light[mol_cell]).astype(object) if n_mol else np.zeros(
        0, object
    )  # fmt: skip
    gf = genuine_fraction if isinstance(genuine_fraction, Mapping) else dict.fromkeys(S.ALL_LOCI, genuine_fraction)
    rho = np.array([float(gf.get(str(lc), 0.0)) for lc in mol_locus])
    genuine = gen.random(n_mol) < rho

    # ---- spot grid: covering cell, tissue, background cDNA by cell type
    def snap(xy: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(xy / spot_um), 0, nx - 1).astype(np.int64)

    n_grid = nx * nx
    lin_all = np.arange(n_grid, dtype=np.int64)
    gx_all, gy_all = (lin_all % nx) * spot_um, (lin_all // nx) * spot_um
    d_centre = np.hypot(gx_all - c0, gy_all - c0)
    in_t_all = d_centre <= tissue_radius_um
    d_cell, j_cell = cKDTree(cxy).query(np.column_stack([gx_all, gy_all]), distance_upper_bound=cell_radius_um)
    cover = np.where(np.isfinite(d_cell), j_cell, -1).astype(np.int64)
    stype = np.where(cover >= 0, ctype[np.maximum(cover, 0)], "none").astype(object)
    grid_comp, _ = compartment_of(np.column_stack([gx_all, gy_all]))
    mult = pd.Series(stype).map(lambda t: float(cmult.get(str(t), 1.0))).to_numpy(dtype=float)
    mult = mult * pd.Series(grid_comp).map(lambda c: float(kmult.get(str(c), 1.0))).to_numpy(dtype=float)
    mean_in = occupancy_in * (1.0 + cdna_per_spot)
    mean_out = occupancy_out * (1.0 + cdna_per_spot)
    mu = np.where(in_t_all, mean_in * mult, mean_out)
    n_bg = gen.negative_binomial(cdna_shape, cdna_shape / (cdna_shape + mu)).astype(np.int64)

    genes = sorted({g for comp in _GEX_V2.values() for g in comp} | set(MOUSE_CONSTANT_GENES.values()))
    gidx = {g: i for i, g in enumerate(genes)}
    bg_lin = np.repeat(lin_all, n_bg)
    bg_type = np.where(in_t_all[bg_lin], stype[bg_lin], "none")
    bg_gene = np.empty(len(bg_lin), dtype=np.int64)
    for t, comp in _GEX_V2.items():
        m = bg_type == t
        if m.any():
            names = list(comp)
            pr = np.array([comp[g] for g in names], float)
            bg_gene[m] = np.array([gidx[g] for g in names])[gen.choice(len(names), int(m.sum()), p=pr / pr.sum())]

    # ---- constant-region transcripts of every lineage cell: bursts on the cell's own spots
    foot = np.flatnonzero(cover >= 0)
    foot = foot[np.argsort(cover[foot], kind="stable")]
    f_start = np.searchsorted(cover[foot], np.arange(n_cells), side="left")
    f_count = np.searchsorted(cover[foot], np.arange(n_cells), side="right") - f_start
    cij = snap(cxy)
    centre_lin = cij[:, 1] * nx + cij[:, 0]
    expr = gen.gamma(constant_cell_shape, 1.0 / constant_cell_shape, n_cells)
    k_lin, k_cell, k_ccall = [], [], []
    for chain_c in (cell_heavy, cell_light):
        mean_k = np.array([float(cpc.get(str(t), 0.0)) for t in ctype[carriers]]) * expr[carriers]
        n_burst = gen.poisson(mean_k / constant_burst) * (f_count[carriers] > 0)  # a cell hidden under others has none
        b_cell = np.repeat(carriers, n_burst)
        size = 1 + gen.poisson(constant_burst - 1.0, len(b_cell))
        u = gen.random(len(b_cell))
        cnt = f_count[b_cell]
        pick = foot[np.minimum(f_start[b_cell] + (u * cnt).astype(np.int64), max(len(foot) - 1, 0))] if len(
            foot
        ) else np.zeros(len(b_cell), np.int64)  # fmt: skip
        b_lin = np.where(cnt > 0, pick, centre_lin[b_cell])
        k_lin.append(np.repeat(b_lin, size))
        k_cell.append(np.repeat(b_cell, size))
        k_ccall.append(np.repeat(chain_c[b_cell], size))
    const_lin = np.concatenate(k_lin) if k_lin else np.zeros(0, np.int64)
    const_cell = np.concatenate(k_cell) if k_cell else np.zeros(0, np.int64)
    const_ccall = np.concatenate(k_ccall) if k_ccall else np.zeros(0, object)
    const_gene = np.array([gidx[MOUSE_CONSTANT_GENES[c]] for c in const_ccall], dtype=np.int64)
    const_seen = gen.random(len(const_lin)) < p_const

    # ---- the GEX library sees each receptor molecule's own transcript at its true spot with p_det
    true_ij = snap(true_xy) if n_mol else np.zeros((0, 2), np.int64)
    true_lin = true_ij[:, 1] * nx + true_ij[:, 0]
    seen = gen.random(n_mol) < p_det
    own_gene_all = np.array([gidx[MOUSE_CONSTANT_GENES[c]] for c in c_call], dtype=np.int64)
    w_gex = (
        n_bg + np.bincount(const_lin[const_seen], minlength=n_grid) + np.bincount(true_lin[seen], minlength=n_grid)
    ).astype(float)
    w_cap = (n_bg + np.bincount(const_lin, minlength=n_grid) + np.bincount(true_lin, minlength=n_grid)).astype(float)
    weight = {"gex": w_gex, "captured": w_cap, "background": n_bg.astype(float)}[swap_weight]

    # ---- swapped molecules: random cDNA (optionally in chimera families) or constant-region-homologous
    rec_lin = true_lin.copy()
    rec_ccall = c_call.copy()
    swap_kind = np.full(n_mol, "", dtype=object)
    family = np.full(n_mol, -1, dtype=np.int64)
    sw = np.flatnonzero(~genuine)
    homo = sw[gen.random(len(sw)) < homologous_switch]
    rand = np.setdiff1d(sw, homo, assume_unique=True)
    if len(rand):
        if swap_family_mean > 0:
            order = rand[np.lexsort((gen.random(len(rand)), mol_clone[rand]))]
            fam = np.empty(len(order), np.int64)
            clone_o = mol_clone[order]
            i, f = 0, 0
            sizes = 1 + gen.poisson(swap_family_mean, len(order))
            while i < len(order):
                j = i + int(sizes[f])
                j = min(j, int(np.searchsorted(clone_o, clone_o[i], side="right")))
                fam[i:j] = f
                i, f = j, f + 1
            land = _draw_weighted(gen, weight, f)
            rec_lin[order] = land[fam]
            family[order] = fam
        else:
            rec_lin[rand] = _draw_weighted(gen, weight, len(rand))
        swap_kind[rand] = "cdna"
    if len(homo):
        # acceptor pool per locus: captured constant-region transcripts (cells' other transcripts and receptor
        # molecules at their true spots), detected or not
        acc_lin = np.r_[const_lin, true_lin]
        acc_ccall = np.r_[const_ccall, c_call]
        acc_locus = np.array([_LOCUS_OF_C.get(str(c), "") for c in acc_ccall], dtype=object)
        for lc in pd.unique(mol_locus[homo]):
            mine = homo[mol_locus[homo] == lc]
            pool = np.flatnonzero(acc_locus == lc)
            if len(pool) == 0:
                rec_lin[mine] = _draw_weighted(gen, weight, len(mine))
                swap_kind[mine] = "cdna"
                continue
            k = pool[gen.integers(0, len(pool), len(mine))]
            rec_lin[mine] = acc_lin[k]
            rec_ccall[mine] = acc_ccall[k]
            swap_kind[mine] = "homologous"
    truth = np.where(genuine, "genuine", "swapped").astype(object)

    # ---- barcode sinks: chimeric copies of random molecules on single spots without cDNA
    sink_rows = []
    free = w_cap == 0
    if n_sinks and n_mol:
        core = lin_all[(d_centre <= 0.6 * tissue_radius_um) & free]
        far = lin_all[(d_centre >= tissue_radius_um + 0.25 * (c0 - tissue_radius_um)) & free]
        n_on = (n_sinks + 1) // 2
        if len(core) < n_on or len(far) < n_sinks - n_on:
            raise ValueError("not enough cDNA-free spots for the barcode sinks (lower the cDNA or n_sinks)")
        picks = [*gen.choice(core, n_on, replace=False), *gen.choice(far, n_sinks - n_on, replace=False)]
        for k, sl in enumerate(picks):
            src = gen.integers(0, n_mol, sink_size)
            sink_rows.append((k, int(sl), src))
    sink_src = np.concatenate([s for _, _, s in sink_rows]) if sink_rows else np.zeros(0, np.int64)
    sink_lin = np.concatenate([np.full(len(s), sl) for _, sl, s in sink_rows]) if sink_rows else np.zeros(0, np.int64)
    rec_src = np.r_[np.arange(n_mol), sink_src].astype(np.int64)
    rec_lin = np.r_[rec_lin, sink_lin].astype(np.int64)
    rec_ccall = np.r_[rec_ccall, c_call[sink_src]].astype(object)
    truth = np.r_[truth, np.full(len(sink_src), "sink", dtype=object)]
    swap_kind = np.r_[swap_kind, np.full(len(sink_src), "", dtype=object)]
    family = np.r_[family, np.full(len(sink_src), -1, dtype=np.int64)]
    n_rec = len(rec_src)

    # ---- GEX: background, detected constant-region transcripts, detected own transcripts, the sinks' traps
    trap_lin = np.array([sl for _, sl, _ in sink_rows], dtype=np.int64)
    trap_gene = np.array([own_gene_all[s[0]] for _, _, s in sink_rows], dtype=np.int64)
    all_lin = np.r_[bg_lin, const_lin[const_seen], true_lin[seen], trap_lin].astype(np.int64)
    all_gene = np.r_[bg_gene, const_gene[const_seen], own_gene_all[seen], trap_gene].astype(np.int64)
    key, cnt = np.unique(all_lin * len(genes) + all_gene, return_counts=True)
    s_lin, s_gene = key // len(genes), key % len(genes)
    spots, s_row = np.unique(s_lin, return_inverse=True)
    X = sp.csr_matrix((cnt.astype(np.float32), (s_row, s_gene)), shape=(len(spots), len(genes)))
    sx_, sy_ = spots % nx, spots // nx
    s_xy = np.column_stack([sx_ * spot_um, sy_ * spot_um])
    s_comp, _ = compartment_of(s_xy)
    s_in = in_t_all[spots]
    s_comp = np.where(s_in, s_comp, "off tissue").astype(object)
    obs = pd.DataFrame(
        {
            "in_tissue": s_in,
            "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(float),
            "cell_type": pd.Categorical(stype[spots]),
            "compartment": pd.Categorical(s_comp),
            S.LIBRARY: library_id,
            "captured_cdna": w_cap[spots] + np.isin(spots, trap_lin),
        },
        index=pd.Index(format_spots(sx_, sy_), dtype=str),
    )
    gex = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    gex.obsm["spatial"] = s_xy

    # ---- receptor records
    src_clone = mol_clone[rec_src] if n_rec else np.zeros(0, np.int64)
    ix, iy = rec_lin % nx, rec_lin // nx
    rxy = np.column_stack([ix * spot_um, iy * spot_um])
    reads = 1 + gen.poisson(np.where(truth == "genuine", reads_genuine, reads_swapped))
    umis = (
        np.array(["".join(r) for r in np.array(list("ACGT"))[gen.integers(0, 4, (n_rec, 10))]], dtype=object)
        if umi and n_rec
        else np.full(n_rec, np.nan, dtype=object)
    )
    src_cell = mol_cell[rec_src] if n_rec else np.zeros(0, np.int64)
    cl = clones.iloc[src_clone]
    air = pd.DataFrame(
        {
            S.SPOT: format_spots(ix, iy),
            S.X: rxy[:, 0],
            S.Y: rxy[:, 1],
            S.LOCUS: cl["locus"].to_numpy(dtype=object),
            S.CLONE: cl["clone_id"].to_numpy(dtype=object),
            S.JUNCTION_AA: cl["junction_aa"].to_numpy(dtype=object),
            S.JUNCTION: cl["junction"].to_numpy(dtype=object),
            S.V_CALL: cl["v_call"].to_numpy(dtype=object),
            S.J_CALL: cl["j_call"].to_numpy(dtype=object),
            S.C_CALL: rec_ccall,
            S.PRODUCTIVE: cl["productive"].to_numpy(dtype=bool),
            S.READS: reads.astype(float),
            S.UMI: umis,
            S.LIBRARY: library_id,
            "x_src": ix,
            "y_src": iy,
            "cell_type": stype[rec_lin],
            "in_tissue": in_t_all[rec_lin],
            "truth": truth,
            "swap_kind": swap_kind,
            "family": family,
            "own_seen": seen[rec_src] if n_rec else np.zeros(0, bool),
            "lineage": cl["lineage"].to_numpy(dtype=object),
            "true_cell": cell_ids[src_cell] if n_rec else np.zeros(0, object),
            "true_cell_type": ctype[src_cell] if n_rec else np.zeros(0, object),
            "true_compartment": ccomp[src_cell] if n_rec else np.zeros(0, object),
            "true_structure": cstruct[src_cell] if n_rec else np.zeros(0, object),
            "true_x_um": true_xy[rec_src, 0] if n_rec else np.zeros(0),
            "true_y_um": true_xy[rec_src, 1] if n_rec else np.zeros(0),
        }
    )

    # ---- truth tables
    cells = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_um": cxy[:, 0],
            "y_um": cxy[:, 1],
            "cell_type": ctype,
            "compartment": ccomp,
            "structure": cstruct,
            "lineage": np.where(has_lin, L.lineage.to_numpy(dtype=object)[lin_of], None)
            if n_lin
            else np.full(n_cells, None),
            "heavy_c_call": cell_heavy,
            "light_c_call": cell_light,
            "n_spots": f_count,
            "constant_transcripts": np.bincount(const_cell, minlength=n_cells),
        }
    )
    L["heavy_clone"] = clone_id[:n_lin]
    L["light_clone"] = clone_id[n_lin:]
    L["n_cells"] = np.bincount(cell_lin[has_lin], minlength=n_lin) if n_lin else []
    by_clone = air.groupby(S.CLONE).truth.value_counts().unstack(fill_value=0)
    clones = clones.join(by_clone.reindex(columns=["genuine", "swapped", "sink"], fill_value=0), on="clone_id")
    clones[["genuine", "swapped", "sink"]] = clones[["genuine", "swapped", "sink"]].fillna(0).astype(np.int64)
    clones["n_cells"] = np.r_[L["n_cells"].to_numpy(), L["n_cells"].to_numpy()] if n_lin else []
    sinks = pd.DataFrame(
        {
            S.SPOT: format_spots([sl % nx for _, sl, _ in sink_rows], [sl // nx for _, sl, _ in sink_rows]),
            S.X: [(sl % nx) * spot_um for _, sl, _ in sink_rows],
            S.Y: [(sl // nx) * spot_um for _, sl, _ in sink_rows],
            "in_tissue": [k < (n_sinks + 1) // 2 for k, _, _ in sink_rows],
            "n_records": [len(s) for _, _, s in sink_rows],
        }
    )
    gen_rec = air[air.truth == "genuine"]
    occ_spot = gen_rec[S.SPOT].isin(gex.obs_names).to_numpy()
    own = np.zeros(len(gen_rec), bool)
    if occ_spot.any():
        rows_ = gex.obs_names.get_indexer(gen_rec[S.SPOT].to_numpy()[occ_spot])
        cols_ = np.array([gidx[MOUSE_CONSTANT_GENES[c]] for c in gen_rec[S.C_CALL].to_numpy()[occ_spot]])
        own[occ_spot] = np.asarray(gex.X[rows_, cols_]).ravel() > 0
    meta: dict[str, Any] = {
        "reader": "sim.synthetic_chip",
        "technology": "synthetic",
        "library_id": library_id,
        "level": "molecule" if umi else "spot_clone",
        "molecule_level": True,
        "source_unit": f"synthetic spot ({spot_um:g} µm)",
        "um_per_unit": float(spot_um),
        "has_molecule_id": bool(umi),
        "has_read_counts": True,
        "model": "realistic",
        "params": params,
        "cells": cells,
        "lineages": L,
        "clones": clones,
        "structures": st,
        "sinks": sinks,
        "truth_counts": {f"{t}|{lc}": int(n) for (t, lc), n in air.groupby(["truth", S.LOCUS]).size().items()},
        "swap_counts": {str(k): int(v) for k, v in air.loc[air.truth == "swapped", "swap_kind"].value_counts().items()},
        "p_det": float(p_det),
        "p_det_given_occupied": float(own[occ_spot].mean()) if occ_spot.any() else float("nan"),
    }
    logger.info("synthetic_chip: %d cells, %d lineages, %d records, %d GEX spots", n_cells, n_lin, len(air), gex.n_obs)
    return finish(air, gex, meta)


#: Locus of each constant-region call (for homologous switches).
_LOCUS_OF_C = {
    "IGKC": "IGK", "IGLC1": "IGL", "IGLC2": "IGL", "IGLC3": "IGL", "IGHM": "IGH", "IGHD": "IGH", "IGHG1": "IGH",
    "IGHG2B": "IGH", "IGHG2C": "IGH", "IGHG3": "IGH", "IGHA": "IGH", "TRAC": "TRA", "TRBC1": "TRB", "TRBC2": "TRB",
}  # fmt: skip


# ------------------------------------------------------------------------------------------- legacy demo port

_SYN_GENES = ["Actb", "Malat1", "Cd74", "Igkc", "Ighm", "Cd79a", "Iglc1", "Hbb-bs"]
_SYN_SHARE = [0.40, 0.25, 0.15, 0.10, 0.005, 0.03, 0.01, 0.055]


def _cid(lin: np.ndarray) -> np.ndarray:
    """Deterministic 25-nt CID per DNB (the legacy encoding of ``y * size + x`` in base 4)."""
    digits = (np.asarray(lin, np.int64)[:, None] // (4 ** np.arange(24, -1, -1, dtype=np.int64))) % 4
    return np.array(["".join(r) for r in np.array(list("ACGT"))[digits]])


def legacy_demo_chip(
    *,
    size: int = 3000,
    radius: int = 700,
    occ_in: float = 0.3,
    occ_out: float = 0.01,
    n_igl: int = 3000,
    n_igk: int = 8000,
    n_igh: int = 4000,
    n_hot: int = 150,
    hot_radius: int = 40,
    sink_size: int = 300,
    p_det: float = 0.5,
    rng: RngLike = 0,
    write_dir: str | Path | None = None,
) -> SpatialAIR:
    """The synthetic Stereo-XCR-seq BCR chip of the archived ``spaclone demo``, with per-molecule truth.

    A faithful port of ``03_BCR_final_2026-10/spaclone/src/spaclone/simulate.py::synthetic_chip``: the random draws
    are made in the same order, so ``rng=0`` reproduces the archived demo input (meta table, GEF, tissue mask, truth)
    exactly. Coordinates are DNB indices (0.5 µm) on a ``size × size`` chip with a tissue disc of ``radius`` DNB:

    * GEX: DNBs in the tissue are occupied with probability ``occ_in`` (``occ_out`` off it), each with
      ``1 + Poisson(1.5)`` MIDs of a fixed gene composition;
    * IGL molecules are **genuine**: each on its own DNB, whose Iglc1 cDNA GEX sees with probability ``p_det``;
    * IGK and IGH molecules are **swapped**: each takes the DNB of a random captured cDNA and its clone from one
      global Zipf pool;
    * IGK clone ``"9999"`` also has ``n_hot`` genuine molecules in a disc of ``hot_radius`` DNB (its Igkc seen with
      ``p_det``), the only spatially real clone, plus ``n_hot`` swapped copies;
    * four **barcode sinks** collect ``sink_size`` swapped molecules each and carry one GEX MID of their chain's
      constant gene (two on the tissue: IGH / Ighm, IGK / Igkc; two far off it: IGK / Igkc, IGH / Ighm).

    Every molecule has ``1 + Poisson(2)`` reads; ``umi`` is a copy of ``cid`` (no molecule identifier).

    Parameters
    ----------
    size, radius, occ_in, occ_out, n_igl, n_igk, n_igh, n_hot, hot_radius, sink_size, p_det
        The legacy generator's parameters (lengths in DNB).
    rng
        Seed or generator; ``0`` is the archived demo.
    write_dir
        If given, write the legacy files there: ``synthetic.meta.parquet`` (one row per read, Stereo-XCR-seq meta
        columns), ``synthetic.raw.gef`` (SAW bin1 layout), ``synthetic_tissue_cut.tif`` (needs ``tifffile``) and
        ``truth.json``.

    Returns
    -------
    SpatialAIR
        ``air``: one row per molecule (``consensus_count`` = its reads; two molecules of one clone on one DNB stay
        two records) with ``truth`` (``"genuine"`` / ``"swapped"`` / ``"sink"``), ``x_src`` / ``y_src`` (DNB) and
        ``in_tissue``. ``gex``: every GEX-occupied DNB with the eight genes. ``meta``: ``truth`` (the legacy
        ``truth.json`` content), ``reads`` (the per-read meta table), ``files`` when written, ``params``.

    References
    ----------
    The archived ``spaclone demo`` (03 README, Validation: synthetic chip with known truth).
    """
    gen = as_rng(rng)
    c = size / 2
    yy, xx = np.ogrid[0:size, 0:size]
    mask = (xx - c) ** 2 + (yy - c) ** 2 <= radius**2  # rows = y, as in SAW tissue-cut images
    lin_in, lin_out = np.flatnonzero(mask.ravel()), np.flatnonzero(~mask.ravel())  # lin = y * size + x
    occ = np.r_[
        gen.choice(lin_in, int(occ_in * len(lin_in)), replace=False),
        gen.choice(lin_out, int(occ_out * len(lin_out)), replace=False),
    ]
    dnb = np.repeat(occ, 1 + gen.poisson(1.5, len(occ)))  # one entry per captured background cDNA
    gene = gen.choice(len(_SYN_GENES), len(dnb), p=_SYN_SHARE)
    igl = gen.choice(lin_in, n_igl, replace=False)
    hc = np.array([c - radius / 2, c])
    hot_pool = lin_in[(lin_in % size - hc[0]) ** 2 + (lin_in // size - hc[1]) ** 2 <= hot_radius**2]
    hot = gen.choice(hot_pool, n_hot, replace=False)
    seen_l, seen_h = igl[gen.random(n_igl) < p_det], hot[gen.random(n_hot) < p_det]
    occ_set = np.zeros(size * size, bool)
    occ_set[occ] = True
    far = lin_out[(lin_out % size - c) ** 2 + (lin_out // size - c) ** 2 >= (radius + 800) ** 2]
    core = lin_in[(lin_in % size - c) ** 2 + (lin_in // size - c) ** 2 <= (radius - 150) ** 2]
    if (~occ_set[far]).sum() < 2:
        raise ValueError("the chip is too small for the legacy layout (sinks need DNBs >= radius + 800 from centre)")
    sink_lin = np.r_[
        gen.choice(core[~occ_set[core]], 2, replace=False), gen.choice(far[~occ_set[far]], 2, replace=False)
    ]
    sink_chain, sink_gene = ["IGH", "IGK", "IGK", "IGH"], ["Ighm", "Igkc", "Igkc", "Ighm"]
    g_all = np.r_[
        gene,
        np.full(len(seen_l), _SYN_GENES.index("Iglc1")),
        np.full(len(seen_h), _SYN_GENES.index("Igkc")),
        [_SYN_GENES.index(g) for g in sink_gene],
    ]
    d_all = np.r_[dnb, seen_l, seen_h, sink_lin]
    key, cnt = np.unique(g_all.astype(np.int64) * size * size + d_all, return_counts=True)
    g_of, d_of = key // (size * size), key % (size * size)
    expr = np.zeros(len(key), dtype=[("x", "<i4"), ("y", "<i4"), ("count", "<u2")])
    expr["x"], expr["y"], expr["count"] = d_of % size, d_of // size, cnt
    per_gene = np.bincount(g_of, minlength=len(_SYN_GENES))
    gtab = np.zeros(len(_SYN_GENES), dtype=[("geneName", "S32"), ("offset", "<u4"), ("count", "<u4")])
    gtab["geneName"] = [g.encode() for g in _SYN_GENES]
    gtab["offset"], gtab["count"] = np.r_[0, np.cumsum(per_gene)[:-1]], per_gene

    def zipf(n_clones: int, n: int, first: int) -> np.ndarray:
        w = 1.0 / np.arange(1, n_clones + 1) ** 1.1
        return (first + gen.choice(n_clones, n, p=w / w.sum())).astype(str)

    pools = {"IGL": (300, 3000), "IGK": (400, 1000), "IGH": (400, 2000)}
    p_igl = ("IGL", igl, zipf(300, n_igl, 3000), "genuine")
    p_igk = ("IGK", dnb[gen.integers(0, len(dnb), n_igk)], zipf(400, n_igk, 1000), "swapped")
    p_hot = ("IGK", np.r_[hot, dnb[gen.integers(0, len(dnb), n_hot)]], np.full(2 * n_hot, "9999"), "hot")
    p_igh = ("IGH", dnb[gen.integers(0, len(dnb), n_igh)], zipf(400, n_igh, 2000), "swapped")
    parts = [p_igl, p_igk, p_hot, p_igh]
    parts += [
        (ch, np.full(sink_size, sl), zipf(pools[ch][0], sink_size, pools[ch][1]), "sink")
        for sl, ch in zip(sink_lin, sink_chain, strict=True)
    ]
    rows, mols = [], []
    for chain, lin, clone, kind in parts:
        nr = 1 + gen.poisson(2.0, len(lin))
        rows.append(
            pd.DataFrame(
                {"x": np.repeat(lin % size, nr), "y": np.repeat(lin // size, nr), "topChains": chain,
                 "cloneId": np.repeat(clone, nr)}
            )
        )  # fmt: skip
        if kind == "hot":
            tr = np.r_[np.full(n_hot, "genuine"), np.full(n_hot, "swapped")]
        else:
            tr = np.full(len(lin), kind)
        mols.append(pd.DataFrame({"lin": lin, "locus": chain, "clone": clone, "reads": nr, "truth": tr}))
    R = pd.concat(rows, ignore_index=True)
    R["readId"] = [f"SYN0000000L1C001R001{i:08d}" for i in range(len(R))]
    R["cid"] = _cid(R.y.to_numpy() * size + R.x.to_numpy())
    R["umi"] = R["cid"]
    k = R.cloneId.astype(int).to_numpy()
    cdr3 = {
        cl: "CA" + "".join(_AA[np.random.default_rng(int(cl)).integers(0, 20, 9)]) + "F" for cl in R.cloneId.unique()
    }
    R["aaSeqCDR3"] = R.cloneId.map(cdr3)
    R["nSeqCDR3"] = R.aaSeqCDR3.str.len().map(lambda n: "TGT" * n)
    R["functional"], R["productive"] = "Functional", "T"
    R["v_call"] = R.topChains + "V1-" + (k % 20 + 1).astype(str) + "*01"
    R["j_call"] = R.topChains + "J" + (k % 4 + 1).astype(str) + "*01"
    R["c_call"] = R.topChains.map({"IGH": "IGHM", "IGK": "IGKC", "IGL": "IGLC1"})
    R["isotypePrimary"] = R["c_call"]
    R["CDR3Begin"] = 100.0
    truth = {
        "genuine_chains": ["IGL"],
        "swapped_chains": ["IGK", "IGH"],
        "hot_clone": "9999",
        "hot_chain": "IGK",
        "hot_center": [float(hc[0]), float(hc[1])],
        "hot_radius": hot_radius,
        "tissue_radius": radius,
        "chip_size": size,
        "sinks": [
            [int(sl % size), int(sl // size), ch, g] for sl, ch, g in zip(sink_lin, sink_chain, sink_gene, strict=True)
        ],
        "p_det": p_det,
        "p_det_given_occupied": p_det / (p_det + (1 - p_det) * occ_in),
    }
    files: dict[str, str] = {}
    if write_dir is not None:
        files = _write_legacy(Path(write_dir), R, expr, gtab, mask, truth, size)

    # molecule records with truth
    Mo = pd.concat(mols, ignore_index=True)
    mx, my = Mo.lin.to_numpy() % size, Mo.lin.to_numpy() // size
    k2 = Mo.clone.astype(int).to_numpy()
    air = pd.DataFrame(
        {
            S.SPOT: format_spots(mx, my),
            S.X: mx * DNB_UM,
            S.Y: my * DNB_UM,
            S.LOCUS: Mo.locus.to_numpy(dtype=object),
            S.CLONE: Mo.clone.to_numpy(dtype=object),
            S.JUNCTION_AA: Mo.clone.map(cdr3).to_numpy(dtype=object),
            S.V_CALL: (Mo.locus + "V1-" + (k2 % 20 + 1).astype(str) + "*01").to_numpy(dtype=object),
            S.J_CALL: (Mo.locus + "J" + (k2 % 4 + 1).astype(str) + "*01").to_numpy(dtype=object),
            S.C_CALL: Mo.locus.map({"IGH": "IGHM", "IGK": "IGKC", "IGL": "IGLC1"}).to_numpy(dtype=object),
            S.PRODUCTIVE: True,
            S.READS: Mo.reads.to_numpy(dtype=float),
            S.LIBRARY: "synthetic",
            "x_src": mx,
            "y_src": my,
            "in_tissue": mask[my, mx],
            "truth": Mo.truth.to_numpy(dtype=object),
        }
    )
    air[S.JUNCTION] = air[S.JUNCTION_AA].str.len().map(lambda n: "TGT" * n).to_numpy(dtype=object)
    spots, s_row = np.unique(d_of, return_inverse=True)
    X = sp.csr_matrix((cnt.astype(np.float32), (s_row, g_of)), shape=(len(spots), len(_SYN_GENES)))
    sx_, sy_ = spots % size, spots // size
    obs = pd.DataFrame(
        {"in_tissue": mask[sy_, sx_], "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(float),
         S.LIBRARY: "synthetic"},
        index=pd.Index(format_spots(sx_, sy_), dtype=str),
    )  # fmt: skip
    gex = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(_SYN_GENES, dtype=str)))
    gex.obsm["spatial"] = np.column_stack([sx_ * DNB_UM, sy_ * DNB_UM])
    params = {
        "size": size,
        "radius": radius,
        "occ_in": occ_in,
        "occ_out": occ_out,
        "n_igl": n_igl,
        "n_igk": n_igk,
        "n_igh": n_igh,
        "n_hot": n_hot,
        "hot_radius": hot_radius,
        "sink_size": sink_size,
        "p_det": p_det,
        "rng_entropy": rng_entropy(gen),
    }
    meta: dict[str, Any] = {
        "reader": "sim.legacy_demo_chip",
        "technology": "synthetic (legacy spaclone demo)",
        "library_id": "synthetic",
        "level": "molecule",
        "molecule_level": True,
        "source_unit": "Stereo-seq DNB (bin1, 0.5 µm)",
        "um_per_unit": DNB_UM,
        "has_molecule_id": False,
        "has_read_counts": True,
        "truth": truth,
        "reads": R,
        "files": files,
        "params": params,
    }
    return finish(air, gex, meta)


def _write_legacy(
    out: Path, R: pd.DataFrame, expr: np.ndarray, gtab: np.ndarray, mask: np.ndarray, truth: dict, size: int
) -> dict[str, str]:
    """Write the legacy demo files (meta Parquet, SAW-layout GEF, tissue-cut TIFF, truth.json)."""
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    meta, gef, tif = out / "synthetic.meta.parquet", out / "synthetic.raw.gef", out / "synthetic_tissue_cut.tif"
    with h5py.File(gef, "w") as h:
        grp = h.create_group("geneExp/bin1")
        e = grp.create_dataset("expression", data=expr)
        for k, v in (("minX", 0), ("minY", 0), ("maxX", size - 1), ("maxY", size - 1)):
            e.attrs[k] = np.array([v], dtype=np.int32)
        grp.create_dataset("gene", data=gtab)
    files = {"meta": str(meta), "raw_gef": str(gef), "truth": str(out / "truth.json")}
    try:
        import tifffile
    except ImportError:  # pragma: no cover - optional extra
        logger.info("tifffile is not installed; the tissue mask TIFF was not written (pip install spaclone[images])")
    else:
        tifffile.imwrite(tif, mask.astype(np.uint8) * 255)
        files["tissue_mask"] = str(tif)
    R.to_parquet(meta, index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return files
