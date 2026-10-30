"""``spaclone.run``: QC, filtering and every applicable location-trust test in one call."""

from __future__ import annotations

import math
import re
import warnings
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone._utils import LowPowerWarning, RngLike, SpacloneWarning, as_rng
from spaclone.qc.report import TrustReport, evaluate_trust

__all__ = ["run"]

#: Own-type vocabulary of ``run(own_types="auto")``: a cell-type label is the receptor's own type when one of these
#: phrases occurs in it as whole words. Matching is case-insensitive on word tokens (the label is split at every
#: character that is not a letter or a digit), so "B_cells", "GC B", "Memory B cells" and "Plasma cells" are B lineage
#: but "Plasmacytoid DC", "Fibroblast" and "NK cells" are not. A label that matches both lineages is left out of both.
OWN_TYPE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "BCR": ("B", "B cell", "B cells", "B_cells", "naive B", "memory B", "GC B", "germinal centre B",
            "germinal center B", "follicular B", "plasma", "plasma cell", "plasma cells", "plasmablast",
            "plasmablasts", "ASC", "ASCs", "antibody-secreting", "Bcell", "Bcells", "Plasmacell", "Plasmacells", "GCB"),
    "TCR": ("T", "T cell", "T cells", "T_cells", "CD4", "CD8", "Treg", "Tregs", "Tfh", "NKT", "gamma-delta", "Tcell", "Tcells", "CD4T", "CD8T"),
}  # fmt: skip
#: Name of each receptor's lineage in messages.
_LINEAGE = {"BCR": "B-lineage", "TCR": "T-lineage"}
ALL_STEPS = (
    "qc",
    "filter",
    "barcode_sinks",
    "same_spot",
    "umi_match",
    "colocalisation",
    "own_type_share",
    "pairing",
    "read_floor",
)
#: Heavy and light loci of the pairing test, per receptor.
CHAIN_PAIRS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "BCR": (("IGH",), ("IGK", "IGL")),
    "TCR": (("TRB",), ("TRA",)),
    "TCRgd": (("TRD",), ("TRG",)),
}
#: Focal (clone) chain and partner chain of the partner-sharing test.
PARTNERS = (("IGH", "IGK"), ("IGH", "IGL"), ("TRB", "TRA"), ("TRD", "TRG"))
#: Read floors of the partner-sharing test (reads of a locus' best record; 20 is the archive's TR1/TR4 floor).
PARTNER_FLOORS = (2.0, 5.0, 20.0)
#: Spots spaced closer than this are sub-cellular (a cell covers several of them).
SUBCELLULAR_UM = 5.0
#: Radius of the local Ig RNA sum for sub-cellular spots (about one cell).
IG_RADIUS_UM = 5.0
#: Locus radius: same-clone spots within this distance form one cell-scale unit.
LOCI_RADIUS_UM = 15.0
#: Clones per spot from which a spot is a barcode sink.
SINK_MIN_CLONES = 20
#: Share of receptor spot ids that must be found in the GEX before GEX-derived strata and cell types are used.
MIN_ID_OVERLAP = 0.5
#: Share of records that must fall in a depth stratum for co-localisation to count as depth-controlled.
MIN_STRATA_COVERAGE = 0.5
ALPHA = 0.05


class RunWarning(SpacloneWarning):
    """A step of :func:`spaclone.run` could not be run, or an input check failed."""


def _matcher(pattern: str | Sequence[str] | Callable[[pd.Series], Any]) -> Callable[[pd.Series], Any]:
    if callable(pattern):
        return pattern
    if isinstance(pattern, str):
        return lambda s: s.astype(str).str.contains(pattern, regex=True, na=False)
    allowed = {str(p) for p in pattern}
    return lambda s: s.astype(str).isin(allowed)


def _step_rngs(rng: RngLike) -> Callable[[str], np.random.Generator]:
    """One generator per step name, derived from the run seed and the name only (not from the order of steps)."""
    if isinstance(rng, np.random.SeedSequence):
        entropy, key = rng.entropy, tuple(rng.spawn_key)
    elif isinstance(rng, int | np.integer) and not isinstance(rng, bool):
        entropy, key = int(rng), ()
    else:  # a Generator (consumed once) or None (fresh entropy)
        entropy, key = int(as_rng(rng).integers(2**63)), ()

    def get(name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(*key, zlib.crc32(name.encode()))))

    return get


def _n_perm(n_perm: int, m: int, k: int = 1) -> int:
    """Permutations B >= n_perm with 1 / (B + 1) <= alpha / (k m): one row can pass BH in a family of m rows."""
    return max(int(n_perm), math.ceil(k * max(int(m), 1) / ALPHA) - 1)


def _keys(spot: pd.Series | pd.Index, lib: pd.Series | np.ndarray | None) -> np.ndarray:
    s = np.asarray(spot, dtype=object).astype(str)
    if lib is None:
        return s
    return np.char.add(np.char.add(np.asarray(lib, dtype=object).astype(str), "\x1f"), s)


def _spacing(xy: np.ndarray, groups: np.ndarray | None = None, n_max: int = 20000, q: float = 0.1) -> float:
    """Spot spacing: a low quantile of nearest-neighbour distances between distinct positions.

    Computed per group (library), then the median over groups. A low quantile recovers the grid pitch of sparse
    receptor tables (on 2-um spots holding 20% of the records the median is 5.7 um, the 10th percentile 2 um) and stays
    near the bead size on beads (AIR-SPACE D14: 7.5 um).
    """
    groups = np.zeros(len(xy), int) if groups is None else pd.factorize(groups)[0]
    out = []
    for g in np.unique(groups):
        p = np.unique(xy[groups == g], axis=0)
        if len(p) < 2:
            continue
        pts = p if len(p) <= n_max else p[np.linspace(0, len(p) - 1, n_max).astype(int)]
        d, _ = cKDTree(p).query(pts, k=2)
        out.append(float(np.quantile(d[:, 1], q)))
    return float(np.median(out)) if out else np.nan


def _on_grid(xy: np.ndarray, pitch: float, tol: float = 1e-3) -> bool:
    """Whether positions lie on a regular grid of ``pitch`` µm (the rule of ``GexLookup.from_anndata``)."""
    if not len(xy) or not pitch > 0:
        return False
    origin = xy[0] - pitch * np.floor(xy[0] / pitch)
    f = (xy - origin) / pitch
    return bool(np.abs(f - np.rint(f)).max() <= tol)


def run(
    data: Any,
    gex: ad.AnnData | None = None,
    *,
    species: Literal["mouse", "human"] = "mouse",
    min_reads: float | str | None = None,
    on_tissue: bool | str | None = None,
    cell_type_key: str | None = "auto",
    own_types: Mapping[str, Any] | Literal["auto"] | None = "auto",
    spot_um: float | None = None,
    gex_reads: pd.DataFrame | None = None,
    steps: Sequence[str] | None = None,
    options: Mapping[str, Mapping[str, Any]] | None = None,
    n_perm: int = 200,
    rng: RngLike = 0,
) -> Any:
    """Run QC, filtering and every location-trust test the data allow, and return a :class:`~spaclone.qc.TrustReport`.

    The report gives three verdicts. **Molecule-level placement** (same-spot test, own-cell-type share, UMI match):
    do receptor molecules sit on their own transcript / cell type? **Clone-level placement** (co-localisation of heavy
    chains, heavy/light pairing, partner-chain sharing): does clone identity follow position? Only the clone level
    licenses clone maps and spatial clone statistics (``tl``, ``bcr`` maps). The **overall** verdict follows the
    pre-stated rules :data:`spaclone.qc.OVERALL_RULES`:

    1. any clone-level signal: "Clone positions supported: clone identity follows position; clone maps and spatial
       clone statistics are licensed." (the molecule level is stated too);
    2. molecule-level signal and a clone level tested by at least two tests without signal: "Cell-type placement only:
       molecules sit on their own cell type / transcript, but clone-level placement was not detected at the power of
       these data (as expected under template switching between constant regions, or when too few molecules are in
       place); do not use clone maps until a clone-level test detects placement.";
    3. no signal at either level and at least two tests with testable rows: "No evidence of positional signal at the
       power of these data", with a power statement (testable rows, permutations, the observed / chance a single row
       would have needed, same-spot bounds);
    4. otherwise "Limited evidence" (one testable test, mostly untestable or inconclusive rows, or a molecule-level
       signal whose clone level could not be tested).

    Parameters
    ----------
    data
        A :class:`spaclone.io.SpatialAIR` from any reader, or a receptor table.
    gex
        Spot-level GEX (obs index = ``spot_id``, ``obsm["spatial"]`` in µm; optional ``obs["library_id"]``,
        ``obs["total_counts"]``, cell types). Taken from ``data.gex`` when omitted. Needed for the same-spot test, the
        depth (Ig RNA) strata of co-localisation and the cell types of the own-type share. When fewer than half of the
        receptor spot ids are found in ``gex.obs_names`` (per library when both carry ``library_id``), the strata and
        cell types are not used and the report says so.
    species
        ``"mouse"`` or ``"human"`` (constant-region and Ig gene names). When the GEX gene names clearly belong to the
        other species (gene-name case, or Ensembl prefixes) a :class:`RunWarning` says so.
    min_reads
        Read floor for the filter (``None``: no floor; ``"auto"``: the floor of :func:`spaclone.qc.read_floor` on
        the chains present — TRB/TRA, else IGH/IGK — with the run's generator).
    on_tissue
        Keep only records on tissue: ``True`` uses the receptor table's ``in_tissue`` column, a string names the
        boolean column; ``None`` keeps all records (the same-spot test still reads the tissue zone).
    cell_type_key
        Column of ``gex.obs`` with cell types. ``"auto"`` (default) uses a column only when the choice is certain:
        ``obs["cell_type"]`` if it exists, else the single standard cell-type label column found by
        :func:`spaclone.io.detect_annotations` (obs label columns only, e.g. ``cell2loc_anno``; deconvolution
        proportion tables are never used); otherwise the own-type share is skipped with a note that lists the
        candidates. A column name that is not in ``gex.obs`` raises. Run :func:`spaclone.gex.score_cell_types` first
        if there is no label column.
    own_types
        ``"auto"`` (default) or ``{"BCR": ..., "TCR": ...}`` with labels (list), a regex, or a function of the label
        Series that marks a unit as the receptor's own cell type. ``"auto"`` matches the label categories against
        :data:`OWN_TYPE_VOCABULARY` (case-insensitive, whole words): B lineage for BCR ("B", "B cells", "B_cells",
        naive / memory / GC / germinal-centre / follicular B, "plasma", "plasma cells", "plasmablasts", "ASC",
        "antibody-secreting", one-word forms such as "Bcells", "GCB") and T lineage for TCR ("T", "T cells", "T_cells",
        "CD4", "CD8", "Treg", "Tfh", "NKT", "gamma-delta", "Tcells", "CD8T"); a label matching both is left out, and a receptor whose lineage matches no category is skipped
        with a note listing the categories. A mapping that names one receptor leaves the other on ``"auto"``. The
        column, the own labels and their evidence are in ``params["cell_types"]``, the test notes and ``methods()``.
    spot_um
        Barcode-spot pitch in µm for the same-spot test (defaults to ``data.meta["um_per_unit"]``).
    gex_reads
        Per-read GEX table (barcode, UMI, gene) for :func:`spaclone.qc.umi_match` (receptor and GEX libraries made
        from the same barcoded cDNA); the UMI match runs only when it is given.
    steps
        Subset of ``("qc", "filter", "barcode_sinks", "same_spot", "umi_match", "colocalisation", "own_type_share",
        "pairing", "read_floor")``; all applicable steps by default. Each step's result depends only on the data, its
        options and the run seed, not on which other steps ran (q-values and verdicts are computed over the steps
        that ran).
    options
        Per-step keyword arguments that override the defaults, e.g. ``{"pairing": {"min_units": 10}}`` or
        ``{"colocalisation": {"bands_um": ((0, 20), (20, 100))}}``. An ``n_perm`` given here is a minimum like the
        argument below. Seeds cannot be set per step.
    n_perm
        Minimum number of permutations of every Monte-Carlo test. It is raised to ``ceil(m / 0.05) - 1`` for a family
        of ``m`` rows (``ceil(k m / 0.05) - 1`` for partner sharing, Bonferroni over ``k`` read floors), so that one
        positive row can pass Benjamini-Hochberg. The cost grows linearly with ``m``: one library (about 7 clone-level
        rows) keeps 200; 12 libraries (about 60 rows) use about 1,200, roughly six times the single-library time per
        row.
    rng
        Seed (or ``SeedSequence`` / generator). Every step draws from its own generator, derived from the seed and the
        step name.

    Returns
    -------
    spaclone.qc.TrustReport
        ``.verdict``, ``.verdict_molecule``, ``.verdict_clone``, ``.power``, ``.status``, ``.tests()``, ``.plot()``,
        ``.methods()``, ``.to_markdown()``, and every underlying result in ``.results``.

    Raises
    ------
    ValueError
        ``species`` is not ``"mouse"`` / ``"human"``; ``cell_type_key`` names a column that ``gex.obs`` lacks (the
        message lists the label columns); ``own_types`` is malformed; or no record passed the filters (the message
        gives ``min_reads`` / ``on_tissue`` and what each filter step removed).

    Warns
    -----
    RunWarning
        When a step fails (it is listed as "not run" with the error), fewer than half of the receptor spot ids are
        found in the GEX, or the GEX gene names belong to the other species.

    Notes
    -----
    Choices fixed before looking at the data (all in the report's ``params``):

    * Sub-cellular spots (molecule-level data whose spot spacing, the 10th percentile of nearest-neighbour distances
      between spots, is < 5 µm): co-localisation uses cell-scale loci (same-clone spots within 15 µm merged), local
      Ig RNA is summed within 5 µm, and pairing needs 50 units per clone. Cell-sized units (beads, bins >= 5 µm, cell
      calls): spot units, the unit's own Ig RNA and 10 units per clone.
    * Barcode sinks (spots with >= 20 clones, per library) are removed before every clone-level test and the
      own-type share.
    * The same-spot test is skipped on beads (no regular grid) and on spots >= 5 µm (a spot holds many transcripts),
      and is untestable when the median chance exceeds 0.5 (test at ceiling).
    * The own-type share places whole spots with probability proportional to GEX total counts (captured cDNA).
    * Partner-chain sharing is tested at loci with >= 2, 5 and 20 reads (Bonferroni over the three floors).
    * Light-chain co-localisation is reported as information; untestable rows (NaN p-values, too few units) are
      excluded from the BH families and from the verdicts.

    Steps that do not apply (no GEX, no cell types, no heavy/light pairs, cell-level calls without molecules) are
    listed as "not run" with the reason.
    """
    from spaclone import __version__, pp, qc
    from spaclone import gex as gexmod
    from spaclone.io.gexio import _species_of_genes

    meta: dict[str, Any] = {}
    if hasattr(data, "air") and hasattr(data, "meta"):
        air, meta = data.air, dict(data.meta or {})
        gex = gex if gex is not None else data.gex
    elif isinstance(data, pd.DataFrame):
        air = data
    else:
        raise TypeError("data must be a SpatialAIR (from a spaclone reader) or a receptor table")
    S.validate_air(air)
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    todo = set(ALL_STEPS if steps is None else steps)
    unknown = todo - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {ALL_STEPS}")
    opts: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (options or {}).items()}
    bad = set(opts) - set(ALL_STEPS)
    if bad:
        raise ValueError(f"options for unknown steps {sorted(bad)}; choose from {ALL_STEPS}")
    if any("rng" in v for v in opts.values()):
        raise ValueError("seeds come from run's rng; do not pass 'rng' in options")
    step_rng = _step_rngs(rng)
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    notes: list[str] = []
    molecule_level = bool(meta.get("molecule_level", True))
    # cell types: the column and each receptor's own labels, decided only when certain (before any step runs)
    ct_key, ct_rule = _cell_type_key(gex, cell_type_key) if gex is not None else (None, "no GEX given")
    own = _own_types(gex.obs[ct_key] if ct_key is not None else None, own_types, ct_key)

    def attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            with warnings.catch_warnings():
                # a step's own BH family (e.g. colocalisation's q-values) is not used here: run forms its families and
                # sets B for them, so that step's "cannot pass Benjamini-Hochberg" warning does not apply
                warnings.filterwarnings("ignore", message=".*Benjamini", category=LowPowerWarning)
                return fn()
        except Exception as e:  # noqa: BLE001 - a failing step is reported, the others still run
            skipped[name] = f"error: {type(e).__name__}: {e}"
            warnings.warn(f"spaclone.run: step {name!r} failed ({type(e).__name__}: {e})", RunWarning, stacklevel=3)
            return None

    def note(msg: str, warn: bool = False) -> None:
        notes.append(msg)
        if warn:
            warnings.warn(f"spaclone.run: {msg}", RunWarning, stacklevel=3)

    sp_found, sp_evidence = _species_of_genes(gex.var_names) if gex is not None else (None, "no GEX given")
    if sp_found is not None and sp_found != species:
        note(
            f"species={species!r} but the GEX gene names look {sp_found} ({sp_evidence}); constant-region and Ig genes "
            f"are looked up by {species} names: pass species={sp_found!r}",
            warn=True,
        )
    if ct_key is not None:
        note(f"cell types: obs[{ct_key!r}] ({ct_rule})")
        for r, o in own.items():
            note(f"own cell types {r}: {o['rule']}")
    elif gex is not None:
        note(f"cell types: none ({ct_rule})")

    # ---------------------------------------------------------------------------------------- QC and filtering
    qc_table = None
    if "qc" in todo:
        m = attempt("qc", lambda: pp.qc_metrics(air))
        qc_table = getattr(m, "table", None)

    air_f, filt = air, None
    if "filter" in todo:
        loci0 = set(air[S.LOCUS].astype(str).unique())
        rf_kw = None
        if min_reads == "auto":
            pair = next(((a, b) for a, b in (("TRB", "TRA"), ("IGH", "IGK")) if {a, b} <= loci0), None)
            rf_kw = {"rng": step_rng("filter:read_floor"), "n_perm": int(n_perm)}
            if pair:
                rf_kw.update(locus=pair[0], partner_locus=pair[1])
        kw = {"valid_junction": True, "productive": True, "min_reads": min_reads, "on_tissue": on_tissue}
        kw.update(opts.get("filter", {}))
        kw.setdefault("read_floor_kwargs", rf_kw)
        fr = attempt("filter", lambda: pp.filter_records(air, **kw))
        if fr is not None:
            air_f, filt = fr.air, fr.report
    elif on_tissue:
        col = on_tissue if isinstance(on_tissue, str) else "in_tissue"
        if col in air.columns:
            air_f = air[air[col].astype(bool).to_numpy()]
    if len(air) and not len(air_f):
        raise ValueError(_no_records_message(air, filt, min_reads, on_tissue))
    air_f = air_f.reset_index(drop=True)
    loci_present = set(air_f[S.LOCUS].astype(str).unique())
    multi_lib = S.LIBRARY in air_f.columns and air_f[S.LIBRARY].nunique() > 1
    lib_air = air_f[S.LIBRARY].astype(str).to_numpy() if S.LIBRARY in air_f.columns else None

    # ---------------------------------------------------------------------------------------- barcode sinks
    air_ns = air_f
    if molecule_level and len(air_f):
        sink_kw = {"min_clones": SINK_MIN_CLONES, **opts.get("barcode_sinks", {})}

        def _sinks():
            if not multi_lib:
                r = pp.barcode_sinks(air_f, **sink_kw)
                return r, np.asarray(r.mask, bool)
            by, mask = {}, np.zeros(len(air_f), bool)
            for lib_v, idx in air_f.groupby(S.LIBRARY, sort=True).indices.items():
                r = pp.barcode_sinks(air_f.iloc[idx], **sink_kw)
                by[str(lib_v)] = r
                mask[idx] = np.asarray(r.mask, bool)
            return by, mask

        out = attempt("barcode_sinks", _sinks)
        if out is not None:
            if "barcode_sinks" in todo:
                results["barcode_sinks"] = out[0]
            air_ns = air_f[~out[1]].reset_index(drop=True)
    elif "barcode_sinks" in todo:
        skipped["barcode_sinks"] = "cell-level calls: spots are cells, sinks do not apply"

    # ---------------------------------------------------------------------------------------- GEX checks
    gex_ids_ok, gex_lib = False, False
    overlap = np.nan
    if gex is not None:
        gex_lib = S.LIBRARY in gex.obs.columns and lib_air is not None
        k_gex = _keys(gex.obs_names, gex.obs[S.LIBRARY].to_numpy() if gex_lib else None)
        k_air = pd.unique(_keys(air_f[S.SPOT], lib_air if gex_lib else None))
        overlap = float(np.isin(k_air, k_gex).mean()) if len(k_air) else np.nan
        if multi_lib and not gex_lib:
            note("the receptor table has several libraries but gex.obs has no 'library_id': spots matched by id only")
        if np.isfinite(overlap) and overlap < MIN_ID_OVERLAP:
            note(
                f"only {overlap:.0%} of receptor spot ids are found in gex.obs_names (< {MIN_ID_OVERLAP:.0%}): "
                "GEX-derived depth strata and cell types are not used (check the spot id format)",
                warn=True,
            )
        else:
            gex_ids_ok = True
            if np.isfinite(overlap) and overlap < 0.9:
                note(f"{overlap:.0%} of receptor spot ids are found in gex.obs_names")
    # unit scale, always from the data (spot_um is only the same-spot lookup pitch): sub-cellular spots (a cell
    # covers several) or cell-sized units (beads, large bins, cells)
    if gex is not None and gex.n_obs > 1:
        spacing = _spacing(
            np.asarray(gex.obsm["spatial"], float)[:, :2], gex.obs[S.LIBRARY].to_numpy() if gex_lib else None
        )
    else:
        spacing = _spacing(S.coords(air_f), lib_air) if len(air_f) > 1 else np.nan
    subcellular = bool(molecule_level and np.isfinite(spacing) and spacing < SUBCELLULAR_UM)

    # depth strata: local Ig RNA of every unit (planned from what the GEX allows, computed only for co-localisation)
    depth_controlled, depth_note = False, "no GEX given"
    ig_list = gexmod.ig_genes(gex.var_names, genes=species) if gex is not None else []
    strata_possible = bool(gex is not None and gex_ids_ok and ig_list)
    if gex is not None and not gex_ids_ok:
        depth_note = "GEX spot ids do not match"
    elif gex is not None and not ig_list:
        depth_note = "no Ig genes in the GEX"
    elif strata_possible and "colocalisation" in todo:
        try:
            genes = ig_list
            if genes:
                sub = gex[:, genes].copy()
                gexmod.ig_rna(
                    sub,
                    genes=genes,
                    radius_um=IG_RADIUS_UM if subcellular else 0.0,
                    library_key=S.LIBRARY if gex_lib else None,
                )
                bins = pd.Series(
                    sub.obs["ig_rna_bin"].astype(str).to_numpy(),
                    index=_keys(sub.obs_names, (sub.obs[S.LIBRARY].to_numpy() if gex_lib else None)),
                )
                lab = pd.Series(_keys(air_ns[S.SPOT], air_ns[S.LIBRARY].to_numpy() if gex_lib else None)).map(bins)
                coverage = float(lab.notna().mean()) if len(lab) else 0.0
                air_ns = air_ns.assign(ig_bin=lab.fillna("NA").to_numpy())
                if coverage >= MIN_STRATA_COVERAGE:
                    depth_controlled = True
                    depth_note = f"{coverage:.0%} of records in Ig RNA strata"
                else:
                    depth_note = f"only {coverage:.0%} of records have an Ig RNA stratum"
                    note(f"depth strata cover only {coverage:.0%} of records: co-localisation is not depth-controlled")
        except Exception as e:  # noqa: BLE001
            skipped["ig_strata"] = f"error: {type(e).__name__}: {e}"
            depth_note = "Ig RNA strata failed"

    # cell types and cDNA weights of the typed units
    unit_types = None
    if gex is not None and gex_ids_ok and ct_key is not None:
        ut = {S.SPOT: np.asarray(gex.obs_names, dtype=object), "cell_type": gex.obs[ct_key].to_numpy()}
        if gex_lib:
            ut[S.LIBRARY] = gex.obs[S.LIBRARY].astype(str).to_numpy()
        if "total_counts" in gex.obs.columns:
            tc = pd.to_numeric(gex.obs["total_counts"], errors="coerce").to_numpy(float)
            ut["total_counts"] = np.where(np.isfinite(tc) & (tc > 0), tc, 0.0)
        unit_types = pd.DataFrame(ut)
        unit_types = unit_types[unit_types["cell_type"].notna()].reset_index(drop=True)

    # ---------------------------------------------------------------------------------------- same-spot checks
    ss_reason, pitch = None, spot_um or meta.get("um_per_unit")
    if isinstance(pitch, list | tuple):
        pitch = pitch[0] if len(set(pitch)) == 1 else None
    if not molecule_level:
        ss_reason = "cell-level calls carry no molecule positions"
    elif gex is None:
        ss_reason = "needs spot-level GEX"
    elif not pitch:
        ss_reason = "spot pitch unknown: pass spot_um"
    elif multi_lib and not gex_lib:
        ss_reason = "several libraries: gex.obs needs a 'library_id' column"
    else:
        xy_gex = np.asarray(gex.obsm["spatial"], float)[:, :2]
        libs_gex = gex.obs[S.LIBRARY].astype(str).to_numpy() if gex_lib else np.zeros(gex.n_obs, int)
        on = all(_on_grid(xy_gex[libs_gex == v], float(pitch)) for v in np.unique(libs_gex))
        if not on:
            ss_reason = (
                f"spots are not on a regular grid of pitch {float(pitch):g} um (beads, e.g. AIR-SPACE or Slide-seq): "
                "the same-spot test needs gridded spots"
            )
        elif max(float(pitch), spacing if np.isfinite(spacing) else 0.0) >= SUBCELLULAR_UM:
            ss_reason = (
                f"spots are cell-sized or larger ({max(float(pitch), spacing):.3g} um): every spot holds many "
                "transcripts and the test is at its ceiling"
            )

    # ---------------------------------------------------------------------------------------- family sizes
    has_reads = S.READS in air_ns.columns and air_ns[S.READS].notna().any()
    libs_ns = air_ns[S.LIBRARY].astype(str) if S.LIBRARY in air_ns.columns else pd.Series("", index=air_ns.index)
    lib_loci = set(zip(libs_ns, air_ns[S.LOCUS].astype(str), strict=True))
    from spaclone.qc.same_spot import HUMAN_CONSTANT_GENES, MOUSE_CONSTANT_GENES

    const = HUMAN_CONSTANT_GENES if species == "human" else MOUSE_CONSTANT_GENES
    present_genes = set(map(str, gex.var_names)) if gex is not None else set()
    ss_genes = {k: [g for g in v if g in present_genes] for k, v in const.items()}
    ss_genes = {k: v for k, v in ss_genes.items() if v}
    m_mol = 0
    if ss_reason is None:
        m_mol += sum(1 for _, lc in lib_loci if lc in ss_genes)
    receptors = {"BCR": S.BCR_LOCI, "TCR": S.TCR_LOCI}
    n_rec = {r: int(air_ns[S.LOCUS].isin(loci).sum()) for r, loci in receptors.items()}
    if unit_types is not None:
        m_mol += sum(1 for r in receptors if n_rec[r] >= 20 and own[r]["matcher"] is not None)
    if gex_reads is not None:
        m_mol += len(loci_present)
    m_clone = 0
    if strata_possible:  # what the data allow, so that B does not depend on the steps that run
        m_clone += sum(1 for _, lc in lib_loci if lc in S.HEAVY_LOCI)
    for heavy, light in CHAIN_PAIRS.values():
        for lib_v in set(libs_ns):
            if any((lib_v, h) in lib_loci for h in heavy):
                m_clone += sum(1 for lt in light if (lib_v, lt) in lib_loci)
    partners = [(a, b) for a, b in PARTNERS if {a, b} <= loci_present] if has_reads else []
    m_clone += len(partners)
    floors = tuple(opts.get("read_floor", {}).get("floors", PARTNER_FLOORS))
    b_mol, b_clone = _n_perm(n_perm, m_mol), _n_perm(n_perm, m_clone)
    b_partner = _n_perm(n_perm, m_clone, k=len(floors))

    def perms(step: str, b: int) -> int:
        return max(int(b), int(opts.get(step, {}).get("n_perm", 0)))

    def kwargs(step: str, **defaults: Any) -> dict[str, Any]:
        out = {**defaults, **opts.get(step, {})}
        out.pop("n_perm", None)
        return out

    # ---------------------------------------------------------------------------------------- molecule level
    if "same_spot" in todo:
        if ss_reason is not None:
            skipped["same_spot"] = ss_reason
        elif not ss_genes:
            skipped["same_spot"] = "no constant-region genes of the chains in the GEX"
        else:
            tile = max(1, round(25.0 / float(pitch))) * float(pitch)

            def _same(a: pd.DataFrame, g: ad.AnnData):
                look = qc.GexLookup.from_anndata(g, spot_um=float(pitch))
                zones = None
                if "in_tissue" in a.columns:
                    a = a.assign(zone=np.where(a["in_tissue"].astype(bool), "tissue", "off tissue"))
                    zones = "zone"
                kw = kwargs("same_spot", constant_genes=ss_genes, tile_um=tile, zones=zones)
                return qc.same_spot(a, look, **kw)

            if gex_lib and multi_lib:
                by = {}
                for lib_v, idx in air_f.groupby(S.LIBRARY, sort=True).indices.items():
                    g = gex[gex.obs[S.LIBRARY].astype(str).to_numpy() == str(lib_v)]
                    if g.n_obs:
                        r = attempt(f"same_spot:{lib_v}", lambda a=air_f.iloc[idx], g=g: _same(a, g))
                        if r is not None:
                            by[str(lib_v)] = r
                if by:
                    results["same_spot"] = by
            else:
                res = attempt("same_spot", lambda: _same(air_f, gex))
                if res is not None:
                    results["same_spot"] = res

    if "own_type_share" in todo:
        if unit_types is None:
            if gex is None:
                skipped["own_type_share"] = "needs spot-level GEX with cell types"
            elif not gex_ids_ok:
                skipped["own_type_share"] = "GEX spot ids do not match the receptor spots"
            else:
                skipped["own_type_share"] = ct_rule
        else:
            typed = pd.Index(_keys(unit_types[S.SPOT], unit_types[S.LIBRARY] if gex_lib else None))
            for receptor, loci in receptors.items():
                sub_air = air_ns[air_ns[S.LOCUS].isin(loci)]
                if len(sub_air) < 20:
                    if len(sub_air):
                        skipped[f"own_type_share:{receptor}"] = f"{len(sub_air)} records (< 20)"
                    continue
                if own[receptor]["matcher"] is None:
                    skipped[f"own_type_share:{receptor}"] = own[receptor]["reason"]
                    continue
                on_typed = typed.isin(pd.unique(_keys(sub_air[S.SPOT], sub_air[S.LIBRARY] if gex_lib else None)))
                if not on_typed.any():
                    skipped[f"own_type_share:{receptor}"] = "no receptor record sits on a typed GEX unit"
                    continue
                kw = kwargs("own_type_share", own_types=own[receptor]["matcher"], read_floors=None, by_spot=True)
                if "total_counts" in unit_types.columns:  # placement proportional to captured cDNA
                    kw.setdefault("weights", "total_counts")
                elif receptor == "BCR":
                    note("own_type_share: gex.obs has no 'total_counts'; units are drawn uniformly")
                ut = unit_types if gex_lib else unit_types.drop(columns=[S.LIBRARY], errors="ignore")
                res = attempt(
                    f"own_type_share:{receptor}",
                    lambda sub_air=sub_air, kw=kw, ut=ut, receptor=receptor: qc.own_type_share(
                        sub_air,
                        unit_types=ut,
                        n_perm=perms("own_type_share", b_mol),
                        rng=step_rng(f"own_type_share:{receptor}"),
                        **kw,
                    ),
                )
                if res is not None:
                    results[f"own_type_share_{receptor}"] = res

    if "umi_match" in todo and gex_reads is not None:
        res = attempt("umi_match", lambda: qc.umi_match(air_f, gex_reads, rng=step_rng("umi_match"),
                                                        **kwargs("umi_match")))  # fmt: skip
        if res is not None:
            results["umi_match"] = res

    # ---------------------------------------------------------------------------------------- clone level
    if "colocalisation" in todo:
        fn = qc.colocalisation
        # cell-scale loci on sub-cellular spots (a bead or cell is already one unit); sinks are already removed per
        # library, and removed again inside for safety
        kw = kwargs(
            "colocalisation",
            strata="ig_bin" if depth_controlled else None,
            units="loci" if subcellular else "records",
            loci_radius_um=LOCI_RADIUS_UM,
            sink_min_clones=SINK_MIN_CLONES if molecule_level else None,
        )
        heavy = air_ns[air_ns[S.LOCUS].isin(S.HEAVY_LOCI)]
        light = air_ns[~air_ns[S.LOCUS].isin(S.HEAVY_LOCI)]
        for key, part, b in (
            ("colocalisation", heavy, b_clone if depth_controlled else n_perm),
            ("colocalisation_light", light, n_perm),
        ):
            if not len(part):
                continue
            nm = "colocalisation:heavy" if key == "colocalisation" else "colocalisation:light"
            kw_k = {**kw, "per_clone": kw.get("per_clone", key == "colocalisation")}  # per-clone table (heavy)
            res = attempt(
                nm,
                lambda part=part, b=b, nm=nm, kw_k=kw_k: fn(
                    part, n_perm=perms("colocalisation", b), rng=step_rng(nm), **kw_k
                ),
            )
            if res is not None:
                results[key] = res
        if not len(heavy) and not len(light):
            skipped["colocalisation"] = "no records"

    if "pairing" in todo:
        any_pair = False
        for receptor, (heavy, light) in CHAIN_PAIRS.items():
            hs = tuple(h for h in heavy if h in loci_present)
            ls = tuple(lt for lt in light if lt in loci_present)
            if not hs or not ls:
                continue
            any_pair = True
            kw = kwargs("pairing", heavy=hs, light=ls, min_units=50 if subcellular else 10)
            nm = f"pairing:{receptor}"
            res = attempt(
                nm,
                lambda kw=kw, nm=nm: qc.pairing(air_ns, n_perm=perms("pairing", b_clone), rng=step_rng(nm), **kw),
            )
            if res is not None:
                results[f"pairing_{receptor}"] = res
        if not any_pair:
            skipped["pairing"] = "needs a heavy and a light chain of one receptor (IGH + IGK/IGL, TRB + TRA, TRD + TRG)"

    if "read_floor" in todo:
        if not has_reads:
            skipped["read_floor"] = "no read counts"
        elif not partners:
            skipped["read_floor"] = "needs a heavy chain and its partner chain (IGH + IGK/IGL, TRB + TRA, TRD + TRG)"
        else:
            for focal, partner in partners:
                nm = f"read_floor:{focal}-{partner}"
                kw = kwargs("read_floor", locus=focal, partner_locus=partner, floors=floors, sides=("above",))
                res = attempt(
                    nm,
                    lambda kw=kw, nm=nm: qc.read_floor(
                        air_ns, n_perm=perms("read_floor", b_partner), rng=step_rng(nm), **kw
                    ),
                )
                if res is not None:
                    results[f"read_floor_{focal}_{partner}"] = res

    # ---------------------------------------------------------------------------------------- verdicts
    results["_skipped"] = skipped
    table, v = evaluate_trust(results, depth_controlled=depth_controlled, depth_note=depth_note)
    results.pop("_skipped")
    results["skipped"] = skipped
    ran = [r for r in own if f"own_type_share_{r}" in results]
    for r in ran:  # the cell types each own-type row rests on
        lab = ", ".join(own[r]["labels"][:6]) + (" ..." if len(own[r]["labels"]) > 6 else "")
        add = f"cell types obs[{ct_key!r}]; own {r} types: {lab or own[r]['rule']}"
        rows = ((table.test == "own_type_share") & (table.locus == r) & (table.verdict != "not run")).to_numpy()
        table.loc[rows, "note"] = [f"{n}; {add}" if isinstance(n, str) and n else add for n in table.loc[rows, "note"]]
    cell_types = {
        "key": ct_key,
        "key_rule": ct_rule,
        "own_types": {r: {"labels": o["labels"], "rule": o["rule"]} for r, o in own.items()},
        "methods": (
            f"Cell types were taken from gex.obs['{ct_key}'] ({ct_rule}); own cell types: "
            + "; ".join(
                f"{r} {', '.join(own[r]['labels']) or 'as given'} ({own[r]['rule'].split(':')[0]})" for r in ran
            )
            + "."
            if ran
            else ""
        ),
    }
    params = {
        "version": __version__,
        "species": species,
        "species_detected": sp_found,
        "species_evidence": sp_evidence,
        "cell_types": cell_types,
        "min_reads": min_reads,
        "on_tissue": on_tissue,
        "n_perm": n_perm,
        "n_perm_used": {"molecule": b_mol, "clone": b_clone, "partner_sharing": b_partner},
        "family_planned": {"molecule": m_mol, "clone": m_clone},
        "depth_controlled": depth_controlled,
        "depth_note": depth_note,
        "stratified": depth_controlled,
        "subcellular": subcellular,
        "spot_spacing_um": spacing,
        "ig_radius_um": IG_RADIUS_UM if subcellular else 0.0,
        "id_overlap": overlap,
        "molecule_level": molecule_level,
        "n_records_in": int(len(air)),
        "n_records": int(len(air_f)),
        "n_records_without_sinks": int(len(air_ns)),
        "steps": sorted(todo),
        "options": {k: sorted(v) for k, v in opts.items()},
        "notes": notes,
        "meta": {k: val for k, val in meta.items() if isinstance(val, str | int | float | bool)},
    }
    summary = {
        "verdict": v["verdict"],
        "overall": v["overall"],
        "molecule": v["molecule"],
        "clone": v["clone"],
        **table.verdict.value_counts().to_dict(),
    }
    return _RunReport(
        name="trust_report",
        table=table,
        summary=summary,
        params=params,
        verdict=v["verdict"],
        verdict_molecule=v["verdict_molecule"],
        verdict_clone=v["verdict_clone"],
        power=v["power"],
        results=results,
        qc=qc_table,
        filtering=filt,
    )


def _tokens(label: Any) -> tuple[str, ...]:
    """Lower-case word tokens of a label: split at every character that is not a letter or a digit."""
    return tuple(re.findall(r"[^\W_]+", str(label).casefold()))


#: Vocabulary phrases as word tokens, longest first (the evidence names the most specific phrase found).
_VOCABULARY_TOKENS = {
    r: sorted(((p, _tokens(p)) for p in ps), key=lambda pt: -len(pt[1])) for r, ps in OWN_TYPE_VOCABULARY.items()
}


def _vocabulary_hit(label: Any, receptor: str) -> str | None:
    """The first phrase of the receptor's vocabulary that occurs in ``label`` as whole words, else None."""
    words = _tokens(label)
    for phrase, toks in _VOCABULARY_TOKENS[receptor]:
        k = len(toks)
        if any(words[i : i + k] == toks for i in range(len(words) - k + 1)):
            return phrase
    return None


def _is_text(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s.dtype)


def _label_columns(gex: ad.AnnData) -> str:
    """The obs columns that can hold cell-type labels: standard label columns first, then other text columns."""
    from spaclone.io.annotations import detect_annotations

    found = detect_annotations(gex)
    other = [str(c) for c in gex.obs.columns if str(c) not in found.labels and _is_text(gex.obs[c])]
    out = "standard label columns: " + (", ".join(map(repr, found.labels)) or "none")
    if other:
        out += "; other text columns: " + ", ".join(map(repr, other[:20])) + (" ..." if len(other) > 20 else "")
    return out


def _cell_type_key(gex: ad.AnnData, cell_type_key: str | None) -> tuple[str | None, str]:
    """The ``gex.obs`` column of cell types and the evidence, or ``None`` with the reason (``cell_type_key="auto"``).

    ``"auto"`` takes ``obs["cell_type"]`` when it exists, else the single standard label column of
    :func:`spaclone.io.detect_annotations` (obs label columns only; proportion tables are never used), else none.
    """
    from spaclone.io.annotations import detect_annotations

    if cell_type_key is None:
        return None, "cell_type_key=None: no cell types"
    if cell_type_key != "auto":
        if cell_type_key in gex.obs.columns:
            return cell_type_key, f"given: cell_type_key={cell_type_key!r}"
        raise ValueError(f"cell_type_key={cell_type_key!r} is not a column of gex.obs ({_label_columns(gex)})")
    if "cell_type" in gex.obs.columns:
        return "cell_type", "cell_type_key='auto': gex.obs['cell_type'] exists"
    found = detect_annotations(gex)
    if found.label is not None:
        return found.label, f"cell_type_key='auto': obs[{found.label!r}] is the only standard cell-type label column"
    if found.labels:
        return None, (
            f"cell_type_key='auto': {len(found.labels)} standard cell-type label columns "
            f"({', '.join(map(repr, found.labels))}), none chosen; pass cell_type_key= one of them"
        )
    clusters = (
        f" (cluster columns are not cell types: {', '.join(map(repr, found.clusters))})" if found.clusters else ""
    )
    return None, (
        f"cell_type_key='auto': no cell-type label column in gex.obs{clusters}; pass cell_type_key= or label the "
        "units with spaclone.gex.score_cell_types"
    )


def _own_types(labels: pd.Series | None, own_types: Any, key: str | None) -> dict[str, dict[str, Any]]:
    """Per receptor: the matcher of its own cell types (``None`` unless certain), the labels it marks, the rule."""
    if not (own_types is None or own_types == "auto" or isinstance(own_types, Mapping)):
        raise ValueError("own_types must be 'auto' or a mapping {'BCR': labels / regex / function, 'TCR': ...}")
    given = dict(own_types) if isinstance(own_types, Mapping) else {}
    bad = set(given) - set(OWN_TYPE_VOCABULARY)
    if bad:
        raise ValueError(f"own_types keys must be 'BCR' or 'TCR', got {sorted(map(str, bad))}")
    cats = [] if labels is None else sorted({str(v) for v in pd.unique(labels.dropna())})
    hits = {r: {c: _vocabulary_hit(c, r) for c in cats} for r in OWN_TYPE_VOCABULARY}
    both = [c for c in cats if all(hits[r][c] for r in OWN_TYPE_VOCABULARY)]
    shown = ", ".join(map(repr, cats[:25])) + (" ..." if len(cats) > 25 else "")
    out: dict[str, dict[str, Any]] = {}
    for r in OWN_TYPE_VOCABULARY:
        if r in given:
            m = _matcher(given[r])
            try:
                marked = [
                    c for c, ok in zip(cats, np.asarray(m(pd.Series(cats, dtype=object)), bool), strict=True) if ok
                ]
            except Exception:  # noqa: BLE001 - a user function that only works on the full Series
                marked = []
            out[r] = {"matcher": m, "labels": marked, "rule": f"given: own_types[{r!r}]", "reason": ""}
            continue
        own = [c for c in cats if hits[r][c] and c not in both]
        if labels is None:
            out[r] = {"matcher": None, "labels": [], "rule": "no cell-type labels", "reason": "no cell-type labels"}
        elif own:
            rule = (
                f"own_types='auto': categories matching the {_LINEAGE[r]} vocabulary: "
                + ", ".join(f"{c!r} ({hits[r][c]!r})" for c in own)
                + (f"; left out as both B and T: {', '.join(map(repr, both))}" if both else "")
            )
            out[r] = {"matcher": _matcher(own), "labels": own, "rule": rule, "reason": ""}
        else:
            reason = (
                f"own_types='auto': no category of obs[{key!r}] matches the {_LINEAGE[r]} vocabulary "
                f"(spaclone._run.OWN_TYPE_VOCABULARY; categories: {shown or 'none'}); pass "
                f"own_types={{{r!r}: [labels]}}"
            )
            out[r] = {"matcher": None, "labels": [], "rule": reason, "reason": reason}
    return out


def _no_records_message(air: pd.DataFrame, filt: pd.DataFrame | None, min_reads: Any, on_tissue: Any) -> str:
    """Why ``run`` stops when no record passed the filters: the arguments and what each filter step removed."""
    removed = ""
    if filt is not None and {"rule", "removed"} <= set(filt.columns):
        steps = filt[pd.to_numeric(filt["removed"], errors="coerce").fillna(0) > 0]
        removed = (
            "; removed: " + ", ".join(f"{r.rule} {int(r.removed):,}" for r in steps.itertuples()) if len(steps) else ""
        )
    return (
        f"spaclone.run: no record passed the filters (min_reads={min_reads!r}, on_tissue={on_tissue!r}, valid "
        f"junction, productive): {len(air):,} records in, 0 left{removed}. Lower min_reads= or check the input"
    )


class _RunReport(TrustReport):
    """:class:`~spaclone.qc.TrustReport` whose methods paragraph also states the cell types used."""

    def methods(self) -> str:
        """A methods paragraph describing what was run, with parameters and references."""
        text = super().methods()
        extra = (self.params.get("cell_types") or {}).get("methods")
        if not extra or not any(k.startswith("own_type_share") for k in self.results):
            return text
        anchor = "(Gel et al. 2016)."
        return text.replace(anchor, f"{anchor} {extra}", 1) if anchor in text else f"{text} {extra}"
