"""Class-switch (CSR) and somatic-hypermutation (SHM) hotspots of a spatial BCR section, from two files to a table.

:func:`hotspot_workflow` runs the whole analysis of the 2026-10 archive (StereoIR spleen, workflow T1 → T3 → T7 → T7b
→ T8f) on a :class:`~spaclone.io.SpatialAIR` from :func:`spaclone.io.read`:

1. **records** — valid, productive heavy-chain (IGH) records with at least ``min_reads`` reads
   (:func:`spaclone.pp.filter_valid`; ``"auto"``: 2 for contigs, 1 for UMI molecules and single reads); clones = V
   gene + J gene + junction amino acids (:func:`spaclone.pp.define_clones`); isotype class, subclass and switched
   flag (:func:`spaclone.bcr.isotype`); V mismatches, excess over the allele baseline and the mutated call
   (:func:`spaclone.bcr.shm` on the AIRR V alignments: evaluable with >= 30 reads and >= 250 aligned V nt, or on
   MiXCR's ``v_alignment_mixcr`` of :func:`spaclone.io.read_airspace` with ``alignments=True``; mutated at >= 3
   excess mismatches).
2. **loci** — same-clone spots within ``loci_radius_um`` merged into one inferred cell (:func:`spaclone.pp.merge_loci`),
   switched when >= 50% of its isotyped records are IgG / IgA / IgE, mutated when the median excess of its evaluable
   records is >= 3 (:func:`spaclone.bcr.call_isotype`, :func:`spaclone.bcr.call_shm`).
3. **maps and hotspots** — Gaussian (``sigma_um``) switched-share map of the isotyped loci and mutated-share maps of
   the evaluable loci, raw and isotype-matched, on a ``pixel_um`` grid over the tissue; hotspots = connected pixels
   above the 95th percentile of the shuffled maps' maxima (family-wise over the section; loci shuffled one by one,
   within isotype class for the isotype-matched map) (:func:`spaclone.bcr.csr_hotspots`,
   :func:`spaclone.bcr.shm_hotspots`). Every CSR hotspot is also tested for more mutated loci than its isotype mix
   predicts (``csr_shm_test``, loci shuffled within isotype; and ``csr_shm_clone``, whole clones relabelled: the
   confirmatory SHM test, as CSR hotspots are selected on switching only; :func:`spaclone.bcr.region_test`).
4. **CSR ∩ SHM regions** — a region is a CSR hotspot whose drawn outline (indicator smoothed by a Gaussian of 0.7
   pixel, level 0.5) intersects the outline of an SHM hotspot; CSR hotspots linked through one SHM hotspot form one
   region; regions are numbered along the section's long axis. Each region gets its switched and mutated shares,
   the isotype-matched SHM excess (loci shuffled within isotype, as the archive) and, with ``clone_null``, the
   **clone-level** p-values (whole clones relabelled, :func:`spaclone.tl.hotspots` with ``blocks``): the stringent
   test, because the loci of one expanded clone share their isotype and mutations.
5. **GEX** — what each region is made of, from the transcriptome only (no Ig gene): marker pseudobulk inside vs the
   rest of a window vs the whole tissue (:func:`spaclone.gex.profile_regions`), smoothed module maps for drawing
   (:func:`spaclone.gex.module_scores`) and an optional GC-B call (:func:`spaclone.gex.call_regions`).

Reading the p-values. The hotspot maps are calibrated against unit-level shuffles: they control the chance of a false
hotspot anywhere on the map *if loci were independent*. Loci of one clone are not, so ``csr_clone_p`` and
``shm_clone_p`` ask the same question against shuffles of whole clones (clones trade marks only with clones of the same
size and isotype mix) and take the region's peak against the maxima of those shuffled maps (map-level, so the
selection of the region is accounted for). ``p`` (isotype-matched excess of mutated loci inside the region) is the
archive's T7 test and ``shm_clone_region_p`` its clone-level version; the region was selected partly on mutation (it
touches an SHM hotspot), so both describe the excess rather than test it. The confirmatory SHM number is
``csr_shm_clone``: every CSR hotspot (selected on switching alone) tested for more mutated loci than its isotype mix
and clone sizes predict, whole clones relabelled, BH over the CSR hotspots.

References
----------
Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128 (kernel share maps); Nichols & Holmes 2002, doi:10.1002/hbm.1058
(max-T map thresholds); Westfall & Young 1993, Resampling-Based Multiple Testing, Wiley; Hoehn et al. 2022,
doi:10.1371/journal.pcbi.1009885 (lineage-respecting permutations); Phipson & Smyth 2010, doi:10.2202/1544-6115.1585
(Monte-Carlo p-values); Victora & Nussenzweig 2022, doi:10.1146/annurev-immunol-120419-022408 (germinal centres:
switching and mutating in one place); Squair et al. 2021, doi:10.1038/s41467-021-25960-2 (pseudobulk).
"""

from __future__ import annotations

import copy
import json
import logging
import warnings
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, mc_pvalue
from spaclone.bcr.csr import csr_hotspots, region_test
from spaclone.bcr.isotype import call_isotype, isotype
from spaclone.bcr.shm import call_shm, shm, shm_hotspots
from spaclone.pp.clones import define_clones
from spaclone.pp.loci import merge_loci
from spaclone.pp.validity import filter_valid

if TYPE_CHECKING:  # spaclone.tl is imported when the workflow runs (no import-time dependency of bcr on tl)
    from spaclone.tl.maps import HotspotResult
    from spaclone.tl.structures import Grid, Raster

__all__ = ["HotspotWorkflowResult", "hotspot_workflow"]

logger = logging.getLogger("spaclone")

_ALIGN = ("v_sequence_alignment", "v_germline_alignment")
_LONG_COLUMNS = (
    "sequence",
    "sequence_alignment",
    "germline_alignment",
    "v_sequence_alignment",
    "v_germline_alignment",
    "j_sequence_alignment",
    "j_germline_alignment",
)
_ISO_CLASSES = ("IgM", "IgD", "IgG", "IgA", "IgE", "unknown")
_POLY_SIGMA_PX = 0.7


class WorkflowWarning(SpacloneWarning):
    """An optional step of :func:`hotspot_workflow` could not run; the result says which and why."""


# ------------------------------------------------------------------------------------------------------- result


@dataclass(frozen=True, repr=False)
class HotspotWorkflowResult:
    """Everything :func:`hotspot_workflow` computed, from the filtered records to the GEX label of each region.

    Attributes
    ----------
    records
        Filtered heavy-chain records (one row per contig / molecule): ``spot_id``, ``x_um``, ``y_um``, ``locus``,
        ``clone_id``, ``isotype`` (subclass when the C call names one, e.g. ``"IgG2b"``, else the class),
        ``isotype_class`` (IgM / IgD / IgG / IgA / IgE / unknown), ``isotype_subclass``, ``isotype_position`` (5' to 3'
        on the IgH locus), ``switched``, ``shm_mutations`` (V mismatches), ``shm_excess`` (over the allele baseline),
        ``mutated``, ``evaluable``, ``locus_id``, ``region_id`` (0 = none), the AIRR fields read (alignments included)
        and the other ``shm_*`` columns.
    loci
        One row per locus (inferred cell): ``locus_id``, ``clone_id``, ``x_um``, ``y_um``, ``n_spots``, ``n_records``,
        ``reads_max``, isotype counts (``IgM`` ... ``unknown``), ``n_isotyped``, ``switched_share``, ``isotype_class``
        (main class), ``switched`` (missing without isotype), ``n_evaluable``, ``excess`` (median), ``evaluable``,
        ``mutated`` (missing when not evaluable), ``on_grid``, ``csr_hotspot``, ``shm_hotspot``,
        ``shm_iso_hotspot`` and ``region_id`` (0 = none).
    grid, tissue
        The pixel grid of every map and the tissue mask (:class:`~spaclone.tl.Grid`, :class:`~spaclone.tl.Raster`).
    csr
        Class-switch hotspots (:class:`~spaclone.tl.HotspotResult`; switched share of isotyped loci, loci shuffled one
        by one: the archive's T3).
    shm, shm_iso
        SHM hotspots with the raw and with the isotype-matched expectation (archive T7), or ``None`` without
        evaluable loci.
    regions
        One row per CSR ∩ SHM region (empty when there is none), numbered 1, 2, ... along the section's long axis:
        ``region_id``, ``csr_hotspots`` / ``shm_hotspots`` (``";"``-joined hotspot ids), ``x_um``, ``y_um``
        (centroid), ``area_um2``, ``n_loci``, ``n_isotyped``, ``n_switched``, ``switched_share``, ``n_evaluable``,
        ``n_mutated``, ``mutated_share``, ``shm_expected`` (isotype-matched), ``shm_excess_ratio``, ``p`` and ``q``
        (isotype-matched excess, loci shuffled within isotype: the archive's T7 test; BH over regions),
        ``csr_peak_z`` / ``shm_peak_z`` (peaks on the unit-level maps), ``csr_p_map`` / ``shm_p_map`` (map-level p of
        the region's hotspots, unit-level null), the **clone-level** columns ``csr_clone_p``, ``shm_clone_p`` (peak of
        the region on the clone-relabelled, studentised map against the maxima of the shuffled maps: map-level,
        selection-proof) and ``shm_clone_region_p`` (excess of mutated loci given isotype and clone size inside the
        region; **descriptive**: the region was selected on SHM, so this p is not a test; the confirmatory number is
        ``csr_shm_clone``), and the GEX answer ``gex_label``, ``gex_ratio_window``, ``gex_second_label``,
        ``gex_second_ratio``, ``gex_mixed``, ``gex_ratio_tissue``, ``n_gex_units``, ``n_gex_window_units``.
    profiles
        Region × marker set (:attr:`spaclone.gex.RegionProfiles.table`): UMIs per 10k inside / window / tissue and
        the ratios.
    module_maps
        Marker set → per-GEX-unit score (:func:`spaclone.gex.module_scores`, index = GEX ``obs_names``), for drawing.
    windows
        ``region_id``, ``x0``, ``x1``, ``y0``, ``y1`` (µm): the box around each region whose other units are the
        region's surroundings in ``profiles``.
    gc_gex
        Boolean per GEX unit: germinal-centre B call from the GEX alone (or ``None``).
    score_mask
        Boolean per GEX unit (index = ``obs_names``): the units whose ``module_maps`` score was computed (tissue units
        selected by ``score_units``; every tissue unit by default). Plots grey the others. ``None`` without GEX.
    params
        Every argument, the generator seeds per step, the null designs and the messages.
    region_labels
        Region id per pixel (the raw pixels of the region's CSR hotspots; 0 = none).
    units
        Per GEX unit (index = ``obs_names``): ``x_um``, ``y_um``, ``tissue``, ``scored`` (= ``score_mask``),
        ``region_id``, ``in_window``, ``gc_gex``.
    csr_shm_test
        Every CSR hotspot (>= 5 evaluable loci) and all of them pooled: mutated loci vs the isotype-matched
        expectation, loci shuffled one by one within isotype class (archive T7).
    csr_shm_clone
        The **confirmatory** SHM test (``clone_null``): every CSR hotspot (>= 5 evaluable loci; column
        ``csr_hotspot`` = the CSR hotspot id) and all of them pooled (``csr_hotspot`` = ``"all"``), mutated loci
        against whole clones relabelled (clones trade calls only with clones of the same size and isotype mix;
        hotspots fixed); ``null_mean`` is the expectation given isotype and
        clone size, ``q_value`` the BH adjustment over the CSR hotspots. CSR hotspots are found from the switched
        calls alone, so this p tests SHM; ``None`` without CSR hotspots or evaluable loci.
    csr_clone, shm_clone
        The clone-level maps behind ``csr_clone_p`` / ``shm_clone_p`` (or ``None``).
    qc
        Records and units kept at each step.
    messages
        What was skipped or found missing, in words.
    gex
        The transcriptome the workflow ran on (the AnnData given, not copied; ``None`` without GEX), so that
        :func:`spaclone.bcr.lineage_workflow` can place tips in compartments from the result alone. Not in ``repr``,
        ``to_csv`` or pickles.
    """

    records: pd.DataFrame
    loci: pd.DataFrame
    grid: Grid | None
    tissue: Raster | None
    csr: HotspotResult | None
    shm: HotspotResult | None
    shm_iso: HotspotResult | None
    regions: pd.DataFrame
    profiles: pd.DataFrame
    module_maps: dict[str, pd.Series]
    windows: pd.DataFrame
    gc_gex: pd.Series | None
    params: dict[str, Any]
    region_labels: Raster | None = None
    units: pd.DataFrame | None = None
    csr_shm_test: TestResult | None = None
    csr_clone: HotspotResult | None = None
    shm_clone: HotspotResult | None = None
    qc: pd.DataFrame = field(default_factory=pd.DataFrame)
    messages: tuple[str, ...] = ()
    score_mask: pd.Series | None = None
    csr_shm_clone: TestResult | None = None
    gex: Any = field(default=None, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle state without the transcriptome (``gex`` is a reference, often gigabytes)."""
        state = dict(self.__dict__)
        state.pop("gex", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for k, v in {"gex": None, "csr_shm_clone": None, **dict(state)}.items():
            object.__setattr__(self, k, v)

    def __repr__(self) -> str:
        n_c = self.csr.summary.get("n_hotspots", 0) if self.csr is not None else 0
        n_s = self.shm.summary.get("n_hotspots", 0) if self.shm is not None else 0
        reg = "; ".join(
            f"{int(r.region_id)}: {r.gex_label} (clone-level p CSR {r.csr_clone_p:.3g}, SHM {r.shm_clone_p:.3g})"
            for r in self.regions.itertuples()
        )
        return (
            f"HotspotWorkflowResult({len(self.records):,} IGH records, {len(self.loci):,} loci; {n_c} CSR and {n_s} SHM "
            f"hotspots; {len(self.regions)} CSR∩SHM regions{': ' + reg if reg else ''})"
        )

    def summary(self) -> pd.DataFrame:
        """One tidy table of the headline numbers: ``step``, ``item``, ``value``, ``note``."""
        rows: list[dict[str, Any]] = []

        def add(step: str, item: str, value: Any, note: str = "") -> None:
            rows.append({"step": step, "item": item, "value": value, "note": note})

        for r in self.qc.itertuples():
            add("records", str(r.step), int(r.records), str(getattr(r, "note", "")))
        add("loci", "loci", len(self.loci))
        if len(self.loci):
            add("loci", "isotyped loci on the grid", int((self.loci.switched.notna() & self.loci.on_grid).sum()))
            add("loci", "evaluable loci on the grid", int((self.loci.evaluable & self.loci.on_grid).sum()))
        unit_nulls = {
            "CSR": "null: loci shuffled one by one, max-T over the map",
            "SHM raw": "null: loci shuffled one by one, max-T over the map",
            "SHM isotype-matched": "null: loci shuffled one by one within isotype class, max-T over the map",
        }
        for name, res in (("CSR", self.csr), ("SHM raw", self.shm), ("SHM isotype-matched", self.shm_iso)):
            if res is None:
                add(name, "map", "not computed")
                continue
            s = res.summary
            add(name, "marked share p0", s.get("p0"))
            add(name, "units on the map", s.get("n_units"))
            add(name, "z threshold", res.threshold.get("hot"), f"{res.params.get('threshold_rule')} rule")
            add(name, "hotspots", s.get("n_hotspots"), f"unit-level {unit_nulls[name]}")
            add(name, "hotspot area mm2", float(res.table.area_um2.sum() / 1e6) if len(res.table) else 0.0)
            add(name, "max z", s.get("max_z"))
        if self.csr_shm_test is not None and len(self.csr_shm_test.table):
            t = self.csr_shm_test.table
            pooled = t[t.region.astype(str) == "all"]
            if len(pooled):
                p = pooled.iloc[0]
                add("CSR hotspots: SHM", "mutated loci (pooled)", int(p.marked))
                add("CSR hotspots: SHM", "isotype-matched expectation (pooled)", float(p.expected))
                add("CSR hotspots: SHM", "p (pooled; loci shuffled within isotype)", float(p.p_value),
                    "unit-level: loci of one clone count as independent")  # fmt: skip
        if self.csr_shm_clone is not None and len(self.csr_shm_clone.table):
            t = self.csr_shm_clone.table
            st = "CSR hotspots: SHM, clone-level (confirmatory)"
            per = t[t.csr_hotspot.astype(str) != "all"]
            add(st, "null", "whole clones relabelled",
                f"CSR hotspots selected on switching only, fixed; clones trade calls only with clones of the same size "
                f"and isotype mix; BH over the {len(per)} tested CSR hotspots")  # fmt: skip
            for r in per.itertuples():
                add(st, f"CSR hotspot {r.csr_hotspot}: q (BH)", float(r.q_value),
                    f"p {float(r.p_value):.3g}; {int(r.marked)} of {int(r.units)} evaluable loci mutated vs "
                    f"{float(r.null_mean):.1f} expected from isotype and clone size")  # fmt: skip
            pooled = t[t.csr_hotspot.astype(str) == "all"]
            if len(pooled):
                p = pooled.iloc[0]
                note = f"{int(p.marked)} of {int(p.units)} evaluable loci mutated vs {float(p.null_mean):.1f} expected"
                add(st, "all CSR hotspots pooled: p", float(p.p_value), note)
        for name, res in (("CSR clone-level", self.csr_clone), ("SHM clone-level", self.shm_clone)):
            if res is not None:
                add(name, "hotspots", res.summary.get("n_hotspots"), "whole clones relabelled, studentised map")
                add(name, "map-level p of the maximum", res.summary.get("p_map_hot"))
        add("regions", "CSR ∩ SHM regions", len(self.regions))
        for r in self.regions.itertuples():
            st = f"region {int(r.region_id)}"
            add(st, "CSR hotspots", r.csr_hotspots)
            add(st, "SHM hotspots", r.shm_hotspots)
            add(st, "area um2", r.area_um2)
            add(st, "switched share", r.switched_share, f"{int(r.n_switched)} of {int(r.n_isotyped)} isotyped loci")
            add(st, "mutated share", r.mutated_share, f"{int(r.n_mutated)} of {int(r.n_evaluable)} evaluable loci")
            add(st, "SHM excess ratio (isotype-matched)", r.shm_excess_ratio)
            add(st, "p (isotype-matched, loci shuffled)", r.p,
                "descriptive: the region was selected on SHM (it touches an SHM hotspot)")  # fmt: skip
            add(st, "clone-level p, CSR (map-level)", r.csr_clone_p, "stringent: whole clones relabelled, max-T")
            add(st, "clone-level p, SHM (map-level)", r.shm_clone_p, "stringent: whole clones relabelled, max-T")
            add(st, "clone-level p, SHM excess in region (descriptive)", r.shm_clone_region_p,
                "descriptive: the region was selected on SHM; the confirmatory SHM test is per CSR hotspot")  # fmt: skip
            add(st, "GEX label", r.gex_label, f"ratio to window {r.gex_ratio_window:.2f}")
            add(st, "GEX second", r.gex_second_label, f"ratio to window {r.gex_second_ratio:.2f}")
        for m in self.messages:
            add("messages", "note", m)
        return pd.DataFrame(rows, columns=["step", "item", "value", "note"])

    def to_csv(self, folder: str | Path, *, alignments: bool = False) -> list[Path]:
        """Write the tables to ``folder`` (created) and return the paths.

        Parameters
        ----------
        folder
            Output directory.
        alignments
            Also write the long sequence / alignment columns of ``records``.

        Returns
        -------
        list of pathlib.Path
            ``summary.csv``, ``qc.csv``, ``records.csv``, ``loci.csv``, ``regions.csv``, ``profiles.csv``,
            ``windows.csv``, ``units.csv`` (with the module scores and the GC call), the hotspot tables
            (``csr_hotspots.csv``, ``shm_hotspots.csv``, ``shm_iso_hotspots.csv``, ``csr_clone_hotspots.csv``,
            ``shm_clone_hotspots.csv``), ``csr_shm_test.csv``, ``csr_shm_clone.csv`` and ``params.json``.
        """
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        def put(name: str, df: pd.DataFrame | None, index: bool = False) -> None:
            if df is None:
                return
            p = out / name
            df.to_csv(p, index=index)
            paths.append(p)

        put("summary.csv", self.summary())
        put("qc.csv", self.qc)
        rec = self.records if alignments else self.records.drop(columns=[c for c in _LONG_COLUMNS if c in self.records])
        put("records.csv", rec)
        put("loci.csv", self.loci)
        put("regions.csv", self.regions)
        put("profiles.csv", self.profiles)
        put("windows.csv", self.windows)
        if self.units is not None:
            u = self.units.copy()
            for k, v in self.module_maps.items():
                u[f"score_{k}"] = v.reindex(u.index).to_numpy()
            put("units.csv", u.rename_axis("spot_id").reset_index())
        for name, res in (
            ("csr_hotspots.csv", self.csr),
            ("shm_hotspots.csv", self.shm),
            ("shm_iso_hotspots.csv", self.shm_iso),
            ("csr_clone_hotspots.csv", self.csr_clone),
            ("shm_clone_hotspots.csv", self.shm_clone),
            ("csr_shm_test.csv", self.csr_shm_test),
            ("csr_shm_clone.csv", self.csr_shm_clone),
        ):
            if res is not None:
                put(name, res.table)
        p = out / "params.json"
        p.write_text(json.dumps(self.params, default=_jsonable, indent=1), encoding="utf-8")
        paths.append(p)
        return paths


def _jsonable(o: Any) -> Any:
    from spaclone.tl.structures import Grid

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Grid):
        return {"origin": list(o.origin), "pixel_um": o.pixel_um, "shape": list(o.shape)}
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="list")
    if isinstance(o, set | frozenset | tuple):
        return list(o)
    return str(o)


# ------------------------------------------------------------------------------------------------------ helpers


def _step_rngs(rng: RngLike) -> Callable[[str], np.random.Generator]:
    """One generator per step name, derived from the seed and the name only (as :func:`spaclone.run`)."""
    if isinstance(rng, np.random.SeedSequence):
        entropy, key = rng.entropy, tuple(rng.spawn_key)
    elif isinstance(rng, int | np.integer) and not isinstance(rng, bool):
        entropy, key = int(rng), ()
    else:
        entropy, key = int(as_rng(rng).integers(2**63)), ()

    def get(name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(*key, zlib.crc32(name.encode()))))

    get.entropy = entropy  # type: ignore[attr-defined]
    return get


def _unpack(data: Any) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    from spaclone.io._container import SpatialAIR

    if isinstance(data, SpatialAIR):
        return data.air, data.gex, dict(data.meta or {})
    if isinstance(data, pd.DataFrame):
        return data, None, {}
    if isinstance(data, tuple) and len(data) >= 2 and isinstance(data[0], pd.DataFrame):
        return data[0], data[1], dict(data[2]) if len(data) > 2 and isinstance(data[2], Mapping) else {}
    raise TypeError("data must be a SpatialAIR (spaclone.io.read), a receptor table, or (air, gex)")


def _poly(mask: np.ndarray) -> np.ndarray:
    """The drawn outline of a pixel set: indicator smoothed by a Gaussian of 0.7 pixel, level 0.5 (archive T7b)."""
    return ndi.gaussian_filter(mask.astype(float), _POLY_SIGMA_PX) >= 0.5


def _long_axis(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and unit vector of the first principal axis, oriented so its larger component is positive."""
    c = xy.mean(axis=0)
    if len(xy) < 3:
        return c, np.array([1.0, 0.0])
    _, v = np.linalg.eigh(np.cov((xy - c).T))
    a = v[:, -1]
    k = int(np.argmax(np.abs(a)))
    return c, a * (1.0 if a[k] >= 0 else -1.0)


def _hot_ids(res: HotspotResult | None) -> list[int]:
    if res is None or not len(res.table):
        return []
    t = res.table
    return [int(h) for h in t.hotspot[t.direction == "hot"]]


def _region_peak(res: HotspotResult | None, pix: np.ndarray) -> tuple[float, float]:
    """Peak of a region on a map's thresholded z and its map-level Monte-Carlo p (against the per-map maxima)."""
    if res is None or res.z_map is None:
        return np.nan, np.nan
    z = res.z_map.values[pix]
    z = z[np.isfinite(z)]
    if not len(z):
        return np.nan, np.nan
    peak = float(z.max())
    null = np.asarray(res.null, dtype=float)
    null = null if null.ndim == 1 else null[:, 0]
    return peak, float(mc_pvalue(null, peak, alternative="greater")[0])


def _legacy_t7_nulls(mark: np.ndarray, classes: np.ndarray, n_perm: int, gen: np.random.Generator):
    """The archive's T7 draw order: per shuffle one raw permutation, then one permutation per isotype class."""
    groups = [np.flatnonzero(classes == c) for c in sorted(pd.unique(classes))]
    raw = np.empty((n_perm, len(mark)), dtype=bool)
    iso = np.empty((n_perm, len(mark)), dtype=bool)
    for b in range(n_perm):
        raw[b] = mark[gen.permutation(len(mark))] > 0.5
        out = mark.copy()
        for idx in groups:
            out[idx] = mark[idx][gen.permutation(len(idx))]
        iso[b] = out > 0.5
    return raw, iso


#: Record levels (the reader's ``meta["level"]``) at which every record is one UMI molecule or one sequencing read.
_ONE_UNIT_LEVELS = ("molecule", "read")
_MIXCR_V = "v_alignment_mixcr"


def _record_level(meta: Mapping[str, Any] | None, air: pd.DataFrame) -> str | None:
    """The reader's record level: ``meta["level"]``, else ``air.attrs["spaclone"]["level"]``, else ``None``."""
    lvl = meta.get("level") if isinstance(meta, Mapping) else None
    if lvl is None:
        sp = air.attrs.get("spaclone")
        lvl = sp.get("level") if isinstance(sp, Mapping) else None
    return lvl if isinstance(lvl, str) and lvl else None


def _auto_floor(value: Any, level: str | None, *, per_unit: float, contig: float, name: str) -> tuple[Any, str]:
    """An ``"auto"`` read floor resolved from the record level: (floor, the rule in words)."""
    if not isinstance(value, str):
        return value, "given"
    if value != "auto":
        raise ValueError(f"{name} must be a number, None or 'auto', got {value!r}")
    if level in _ONE_UNIT_LEVELS:
        return per_unit, f"auto: {per_unit:g}, one {level} per record (reader level {level!r})"
    why = f"reader level {level!r}" if level else "record level unknown"
    return contig, f"auto: {contig:g}, contigs or read-counted clones ({why})"


def _floor_message(caller: str, floor: Any, removed: int, before: int, reads: Any, level: str | None) -> str | None:
    """Warning text when the read floor removes more than half of the records (``None`` otherwise)."""
    if floor is None or before <= 0 or removed <= 0.5 * before:
        return None
    r = pd.to_numeric(pd.Series(reads), errors="coerce") if reads is not None else pd.Series(dtype=float)
    med = f"; median {float(r.median()):g} reads per record" if r.notna().any() else ""
    hint = (
        f"the records are single {level}s: min_reads='auto' uses 1"
        if level in _ONE_UNIT_LEVELS
        else "for molecule-level records (one row per UMI molecule) pass min_reads=1 or min_reads=None"
    )
    return f"{caller}: the read floor min_reads={floor:g} removed {removed:,} of {before:,} records ({removed / before:.0%}{med}); {hint}"


def _filter_counts(steps: list[tuple[str, int]]) -> str:
    """``"input 120; valid junction 110; ..."``: the records left after every filter, for error messages."""
    return "; ".join(f"{name} {n:,}" for name, n in steps)


def _species_hint(gex: Any, markers: Any, species: str, n_undetermined: int, n_regions: int) -> str | None:
    """A message naming ``species=`` when the marker genes are in ``gex.var_names`` in another letter case only.

    Mouse genes are written ``Cd19``, human genes ``CD19``: when fewer than half of the marker genes are found as
    written but at least twice as many in another case, the species does not match the data. ``None`` otherwise.
    """
    from spaclone.gex.profile import profile_markers

    try:
        with warnings.catch_warnings():  # the profiles already warned about these sets
            warnings.simplefilter("ignore")
            cells, _ = profile_markers(markers, None)
    except (ValueError, TypeError):
        return None
    genes = sorted({str(g) for v in cells.values() for g in v})
    if not genes:
        return None
    names = pd.Index(gex.var_names).astype(str)
    found = int(np.isin(genes, names).sum())
    upper = set(names.str.upper())
    found_ci = sum(g.upper() in upper for g in genes)
    if found >= 0.5 * len(genes) or found_ci < max(2 * found, 3):
        return None
    lab = f"; {n_undetermined} of {n_regions} region GEX labels are 'undetermined'" if n_regions else ""
    if isinstance(markers, str):
        other = "human" if species == "mouse" else "mouse"
        what, fix = f"markers={markers!r}", f"the gene names look {other}: pass species={other!r}"
    else:
        what, fix = "the markers given", "pass markers= with the gene names spelled as in gex.var_names"
    return (
        f"hotspot_workflow: only {found} of {len(genes)} marker genes of {what} (species={species!r}) are in "
        f"gex.var_names, {found_ci} in another letter case{lab}; {fix}"
    )


# ------------------------------------------------------------------------------------------------ the workflow


def _csr_hotspot_column(res: TestResult) -> TestResult:
    """Name the tested ids ``csr_hotspot`` (CSR hotspot ids; ``region_test`` calls them ``region``)."""
    return replace(res, table=res.table.rename(columns={"region": "csr_hotspot"}))


def hotspot_workflow(
    data: Any,
    *,
    species: Literal["mouse", "human"] = "mouse",
    min_reads: float | Literal["auto"] | None = "auto",
    loci_radius_um: float = 15.0,
    sigma_um: float = 50.0,
    pixel_um: float = 10.0,
    n_perm: int = 1000,
    alpha: float = 0.05,
    clone_null: bool = True,
    markers: str | Mapping[str, Any] = "immune_mouse_no_ig",
    rng: RngLike = 0,
    n_perm_region: int = 10_000,
    min_neff: float = 10.0,
    shm_from: Literal["raw", "isotype"] = "raw",
    overlap: Literal["polygon", "pixel"] = "polygon",
    window_pad_um: float = 100.0,
    tissue_key: str | None = "in_tissue",
    gc_call: bool = True,
    gc_options: Mapping[str, Any] | None = None,
    module_smooth_um: float = 22.5,
    score_units: Any = None,
    mixed_log2: float = 0.3,
    clone_method: Literal["exact", "vj_length_identity", "given"] = "exact",
    shm_min_reads: float | Literal["auto"] = "auto",
    shm_min_aligned_nt: int | Literal["auto"] = "auto",
    min_excess: float = 3,
    shm_options: Mapping[str, Any] | None = None,
    legacy: bool = False,
) -> HotspotWorkflowResult:
    """Run the class-switch / hypermutation hotspot analysis of one section end to end (see the module notes).

    Parameters
    ----------
    data
        A :class:`~spaclone.io.SpatialAIR` from :func:`spaclone.io.read` (receptor records with ``spot_id``,
        ``x_um``, ``y_um``, ``locus``, the V / J / C calls, ``junction_aa``, ``productive``, ``consensus_count`` and,
        for SHM, ``v_sequence_alignment`` / ``v_germline_alignment`` or MiXCR's ``v_alignment_mixcr``, e.g. from
        :func:`spaclone.io.read_airspace` with ``alignments=True``; GEX with raw counts and ``obsm["spatial"]`` in
        µm). A receptor table alone runs steps 1–4 (no GEX characterisation). One section at a time.
    species
        ``"mouse"`` or ``"human"``: IgH constant-locus order and the default marker sets.
    min_reads
        Read floor of the records; ``None`` = no floor. ``"auto"`` (default) decides from the record level the
        reader wrote (``meta["level"]`` of the :class:`~spaclone.io.SpatialAIR`, else ``attrs["spaclone"]["level"]``
        of the table): **1** when every record is one UMI molecule or one sequencing read (``"molecule"``,
        ``"read"``: AIR-SPACE, Stereo-XCR-seq molecules, AIRR tables with a ``umi`` column), whose read count only
        counts PCR copies of one molecule (AIR-SPACE: median 1); **2** (the archive's floor) for contigs and per-spot
        clone tables (``"contig"``, ``"spot_clone"``, ...: StereoIR / TRUST4, MiXCR clone tables) and when the level
        is unknown. A warning names the floor whenever it removes more than half of the records.
    loci_radius_um
        Same-clone spots within this distance are one locus (archive: 15 µm, adjacent 10-µm bins incl. diagonal).
    sigma_um
        Gaussian standard deviation of the maps (archive: 50 µm).
    pixel_um
        Map pixel (archive: 10 µm).
    n_perm
        Shuffled maps for every hotspot map (archive: 1,000).
    alpha
        Family-wise error rate of each map.
    clone_null
        Also compute the clone-level maps and the clone-level p-values of each region (doubles the run time).
    markers
        GEX marker sets for the region profiles and module maps (:data:`spaclone.gex.PROFILE_MARKERS`; with
        ``species="human"`` the default becomes ``"immune_human_no_ig"``) or ``{set: genes}`` without Ig genes.
    rng
        Seed. Every step draws from its own generator derived from the seed and the step's name. With ``legacy`` it
        must be an integer: the CSR map and the SHM maps then each start ``numpy.random.default_rng(rng)`` as the
        archive did.
    n_perm_region
        Shuffles of the region tests (archive: 10,000).
    min_neff
        Pixels with fewer effective loci within the kernel are not used (archive: 10).
    shm_from
        SHM hotspots that define the regions: ``"raw"`` (the archive's T7b / T8 choice) or ``"isotype"``
        (isotype-matched map: only places with more mutation than their isotype mix predicts).
    overlap
        ``"polygon"`` (drawn outlines intersect, archive T7b) or ``"pixel"`` (hotspot pixels overlap, archive T8).
    window_pad_um
        Pad (µm) around a region's bounding box: the box is the region's window, whose other GEX units are its
        surroundings in the profile.
    tissue_key
        Boolean GEX ``obs`` column of tissue units (Visium ``in_tissue``); absent column = every GEX unit is tissue.
        The tissue units set the grid and the tissue mask of the maps and the reference of the profiles.
    gc_call
        Call germinal-centre B units from the GEX alone (:func:`spaclone.gex.call_regions`, GC-B marker share within
        30 µm in the top 1% of tissue units, DBSCAN clusters of >= 30 such units), for drawing; descriptive.
    gc_options
        Arguments of :func:`spaclone.gex.call_regions` that replace those defaults (``top_frac``, ``min_units``,
        ``smooth_um``, ``dbscan_eps_um``, ``min_samples``, ``widen_um``, ``markers``).
    module_smooth_um
        Neighbourhood radius of the module maps (archive: 22.5 µm).
    score_units
        GEX units that get a module-map score (:func:`spaclone.gex.unit_mask`): ``None`` (default) = every tissue
        unit; a boolean ``obs`` column name; ``(column, values)``, e.g. ``("cell2loc_anno", ["B_cells"])`` to score
        B-lineage units only as the archive's figure did; or one boolean per GEX unit (array, or Series indexed by
        ``obs_names``). Only the selected units are z-scored and smoothed together; the others get NaN and are listed
        as unscored in ``score_mask``. The region profiles and labels always use every unit.
    mixed_log2
        Profile label rule: "mixed" when the second set is within this log2 ratio of the first (archive 0.3).
    clone_method
        ``"exact"`` (V gene + J gene + junction aa, the archive), ``"vj_length_identity"`` (85% junction identity),
        or ``"given"`` (keep ``clone_id``).
    shm_min_reads, shm_min_aligned_nt, min_excess
        SHM evaluability floors and the mutated threshold (archive: 30 reads, 250 aligned V nt, >= 3 excess).
        ``"auto"``: 30 reads for contigs, 1 for molecules and single reads (as ``min_reads``); 250 aligned V nt for
        AIRR alignments, 100 for MiXCR alignments (long reads covering the 3' part of V, median ~150 nt).
    shm_options
        Arguments of :func:`spaclone.bcr.shm` that replace the workflow's choices, e.g. ``{"baseline_fallback":
        "pooled"}``. With MiXCR alignments the workflow uses ``baseline_fallback="zero"`` (few V genes have enough
        unswitched molecules for an allele baseline), with AIRR alignments the archive's ``"none"``.
    legacy
        Reproduce the 2026-10 archive bit for bit: Kish ``n_eff`` approximated as in T3 / T7, thresholds by
        ``numpy.quantile``, the archive's random draw order, tissue closing without edge padding and only gap
        characters skipped in alignments. Default ``False``: exact ``n_eff``, rank thresholds (exact family-wise
        error), padded tissue edge, non-ACGT skipped.

    Returns
    -------
    HotspotWorkflowResult
        Records, loci, grid, tissue, the CSR / SHM hotspot results, the regions with their tests and GEX labels,
        profiles, module maps, windows, the GC call, ``params``; ``summary()`` and ``to_csv(folder)``.

    Raises
    ------
    ValueError
        No heavy-chain record passes the filters (the message gives the records left after every filter and the
        ``min_reads`` used), or the records come from more than one section.

    Notes
    -----
    Null model (CSR and SHM hotspots, ``csr`` / ``shm`` / ``shm_iso``): random labelling of loci. Locus positions and
    the eligible pixels are fixed; the switched (mutated) calls are permuted among the isotyped (evaluable) loci, for
    ``shm_iso`` within isotype class; a hotspot is a connected set of pixels above the (1 − α) quantile of the shuffled
    maps' maxima.

    Null model (clone-level, ``csr_clone`` / ``shm_clone`` and the ``*_clone_p`` columns): random relabelling of whole
    clones. Positions, isotype classes and eligible pixels are fixed; clones with the same number of loci of every
    isotype trade their calls, and inside a clone the calls are re-dealt among its loci of each isotype; the map is
    studentised by the shuffled maps (Westfall–Young max-T). ``csr_clone_p`` / ``shm_clone_p`` = share of shuffled maps
    whose maximum anywhere reaches the region's peak.

    Null model (``p``, ``csr_shm_test``): mutated calls permuted among evaluable loci of the same isotype class,
    positions and regions fixed (archive T7). ``csr_shm_clone`` and ``shm_clone_region_p``: the clone-level design
    above with the CSR hotspots / the region fixed. ``csr_shm_clone`` is the confirmatory SHM test (CSR hotspots are
    selected on switching alone; BH over the hotspots); ``p`` and ``shm_clone_region_p`` are descriptive, because
    a region is a CSR hotspot that touches an SHM hotspot.

    No overlap region: ``regions`` is empty and ``messages`` says why; the hotspots are still returned.

    References
    ----------
    Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Nichols & Holmes 2002, doi:10.1002/hbm.1058; Westfall &
    Young 1993, Resampling-Based Multiple Testing, Wiley; Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885;
    Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x; Squair et al. 2021, doi:10.1038/s41467-021-25960-2.

    Examples
    --------
    >>> data = spaclone.io.read("mm_spleen_bin20_anno.h5ad", "mm_spleen_bcr_airr.tsv")  # doctest: +SKIP
    >>> res = spaclone.bcr.hotspot_workflow(data, rng=0)  # doctest: +SKIP
    >>> res.regions[["region_id", "switched_share", "mutated_share", "gex_label", "csr_clone_p"]]  # doctest: +SKIP
    """
    from spaclone import gex as gexmod
    from spaclone.tl.structures import Grid, Raster, tissue_mask

    if shm_from not in ("raw", "isotype"):
        raise ValueError("shm_from must be 'raw' or 'isotype'")
    if overlap not in ("polygon", "pixel"):
        raise ValueError("overlap must be 'polygon' or 'pixel'")
    if clone_method not in ("exact", "vj_length_identity", "given"):
        raise ValueError("clone_method must be 'exact', 'vj_length_identity' or 'given'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if n_perm < 1 or n_perm_region < 1:
        raise ValueError("n_perm and n_perm_region must be >= 1")
    if legacy and not (isinstance(rng, int | np.integer) and not isinstance(rng, bool)):
        raise ValueError("legacy=True replays the archive's draws and needs an integer seed (rng=0 in the archive)")
    if species == "human" and markers == "immune_mouse_no_ig":
        markers = "immune_human_no_ig"
    air, gex, meta = _unpack(data)
    level = _record_level(meta, air)
    min_reads, min_reads_rule = _auto_floor(min_reads, level, per_unit=1, contig=2, name="min_reads")
    bad = {"key_added", "return_baseline"} & set(shm_options or {})
    if bad:
        raise ValueError(f"shm_options cannot set {sorted(bad)}")
    step_rng = _step_rngs(rng)
    messages: list[str] = []
    warn_log: list[str] = []
    seeds: dict[str, Any] = {}

    def note(msg: str) -> None:
        messages.append(msg)
        logger.info("hotspot_workflow: %s", msg)

    def gen(name: str) -> np.random.Generator:
        seeds[name] = f"SeedSequence({step_rng.entropy}, crc32({name!r}))"  # type: ignore[attr-defined]
        return step_rng(name)

    def optional(name: str, fn: Callable[[], Any]) -> Any:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                out = fn()
            for w in caught:
                warn_log.append(f"{name}: {w.message}")
                logger.warning("hotspot_workflow %s: %s", name, w.message)
            return out
        except Exception as e:  # noqa: BLE001 - an optional step is reported, the rest still runs
            msg = f"{name} skipped ({type(e).__name__}: {e})"
            messages.append(msg)
            warnings.warn(f"hotspot_workflow: {msg}", WorkflowWarning, stacklevel=3)
            return None

    legacy_kw = {"neff_kernel": "legacy", "threshold_rule": "quantile"} if legacy else {}

    # ============================================================================================ 1. records
    qc_rows = [{"step": "receptor records", "records": len(air), "note": "input"}]
    for c in (S.SPOT, S.X, S.Y, S.LOCUS, S.JUNCTION_AA):
        if c not in air.columns:
            raise KeyError(f"hotspot_workflow: the receptor table lacks {c!r} (read the data with spaclone.io.read)")
    ok, report = filter_valid(air, require_productive=True, min_reads=min_reads)
    report = report.reset_index(drop=True)
    for k, r in enumerate(report.itertuples()):
        if r.step == 0:
            continue
        is_floor = str(r.rule).startswith("reads >=")
        qc_rows.append(
            {
                "step": str(r.rule),
                "records": int(r.records),
                "note": ("" if r.applied else "not applied") or (f"min_reads {min_reads_rule}" if is_floor else ""),
            }
        )
        if is_floor:
            reads_in = air[S.READS] if S.READS in air.columns else None
            msg = _floor_message(
                "hotspot_workflow", min_reads, int(r.removed), int(report.records.iloc[k - 1]), reads_in, level
            )
            if msg is not None:
                messages.append(msg)
                warnings.warn(msg, WorkflowWarning, stacklevel=2)
    H = ok[ok[S.LOCUS].astype("string") == "IGH"].copy()
    qc_rows.append({"step": "IGH (heavy chain)", "records": len(H), "note": "units of the CSR / SHM analysis"})
    if not len(H):
        mr = "None" if min_reads is None else f"{min_reads:g} ({min_reads_rule})"
        raise ValueError(
            "hotspot_workflow: no heavy-chain (IGH) record passes the filters (records left: "
            f"{_filter_counts([(q['step'], int(q['records'])) for q in qc_rows])}; min_reads={mr}); CSR / SHM "
            "hotspots need BCR heavy chains: check the locus column, or lower min_reads (1 or None for molecule-level "
            "records)"
        )
    if S.LIBRARY in H.columns and H[S.LIBRARY].nunique(dropna=True) > 1:
        raise ValueError("hotspot_workflow: the records come from several sections (library_id); run one at a time")
    H = H.reset_index(drop=True)
    if clone_method != "given":
        H = define_clones(H, method=clone_method, by=(S.LOCUS,))
    if H[S.CLONE].isna().any():
        n_na = int(H[S.CLONE].isna().sum())
        H = H[H[S.CLONE].notna()].reset_index(drop=True)
        qc_rows.append({"step": "clone id defined", "records": len(H), "note": f"{n_na} without V / J gene"})
    H = isotype(H, species=species)
    has_airr = all(c in H.columns for c in _ALIGN) and H[_ALIGN[0]].notna().any()
    has_mixcr = not has_airr and _MIXCR_V in H.columns and H[_MIXCR_V].notna().any()
    has_align = has_airr or has_mixcr
    shm_kw: dict[str, Any] = {}
    shm_rule = None
    shm_source = "AIRR v_sequence_alignment / v_germline_alignment" if has_airr else None
    if has_mixcr:
        shm_source = f"MiXCR {_MIXCR_V}"
    if has_align:
        shm_mr, shm_rule = _auto_floor(shm_min_reads, level, per_unit=1, contig=30, name="shm_min_reads")
        if isinstance(shm_min_aligned_nt, str) and shm_min_aligned_nt != "auto":
            raise ValueError(f"shm_min_aligned_nt must be an integer or 'auto', got {shm_min_aligned_nt!r}")
        shm_nt = (100 if has_mixcr else 250) if shm_min_aligned_nt == "auto" else shm_min_aligned_nt
        shm_kw = {"min_reads": shm_mr, "min_aligned_nt": shm_nt, "min_excess": min_excess, "skip_ambiguous": not legacy}
        if has_mixcr:
            shm_kw.update(mixcr_alignment=_MIXCR_V, baseline_fallback="zero")
        shm_kw.update(dict(shm_options or {}))
        H, baseline_table = shm(H, return_baseline=True, **shm_kw)
    else:
        baseline_table = None
        note(
            "no V alignments (v_sequence_alignment / v_germline_alignment, or MiXCR's v_alignment_mixcr from "
            "io.read_airspace(..., alignments=True)): SHM calls, SHM maps and regions skipped"
        )
        for c, v in (("shm_mismatches", np.nan), ("shm_excess", np.nan), ("shm_evaluable", False)):
            H[c] = v
        H["shm_mutated"] = False
    # loci, numbered by clone and first appearance of a record (the archive's order, so its draws can be replayed)
    loci, member = merge_loci(H, radius_um=loci_radius_um)
    first = pd.Series(np.arange(len(H))).groupby(member.to_numpy()).min()
    loci["_first"] = loci[S.LOCUS_ID].map(first).to_numpy()
    loci = loci.sort_values([S.CLONE, "_first"], kind="stable").reset_index(drop=True)
    new_id = pd.Series(np.arange(len(loci), dtype=np.int64), index=loci[S.LOCUS_ID].to_numpy())
    H[S.LOCUS_ID] = member.map(new_id).to_numpy()
    loci[S.LOCUS_ID] = np.arange(len(loci), dtype=np.int64)
    loci = loci.drop(columns="_first")
    ic = call_isotype(H, by="locus")
    sc = call_shm(H, by="locus") if has_align else None
    keep_ic = ["IgM", "IgD", "IgG", "IgA", "IgE", "unknown", "n_isotyped", "switched_share", "isotype", "switched"]
    L = loci.join(ic[keep_ic].rename(columns={"isotype": "isotype_class"}), on=S.LOCUS_ID)
    if sc is not None:
        L = L.join(sc[["n_evaluable", "excess", "evaluable", "mutated"]], on=S.LOCUS_ID)
    else:
        L["n_evaluable"], L["excess"], L["evaluable"] = 0, np.nan, False
        L["mutated"] = pd.array([pd.NA] * len(L), dtype="boolean")
    L["evaluable"] = L["evaluable"].fillna(False).astype(bool)
    L["isotype_class"] = L["isotype_class"].astype("string").fillna("unknown")
    qc_rows.append({"step": "loci", "records": len(L), "note": f"same-clone spots within {loci_radius_um:g} um merged"})

    # ============================================================================================ 2. grid, tissue
    units = None
    if gex is not None:
        from spaclone.gex.qc import _bool_obs, get_xy

        gxy = get_xy(gex)
        tis = (
            _bool_obs(gex, tissue_key)
            if (tissue_key is not None and tissue_key in gex.obs)
            else np.ones(gex.n_obs, bool)
        )
        units = pd.DataFrame({S.X: gxy[:, 0], S.Y: gxy[:, 1], "tissue": tis}, index=gex.obs_names.copy())
        T = units.loc[tis, [S.X, S.Y]]
        if not len(T):
            raise ValueError(f"hotspot_workflow: no GEX unit is flagged as tissue in obs[{tissue_key!r}]")
        tissue_src = f"GEX units ({'obs[' + repr(tissue_key) + ']' if tissue_key in gex.obs else 'all'})"
    else:
        T = H.drop_duplicates(S.SPOT)[[S.X, S.Y]]
        tissue_src = "receptor spots (no GEX)"
        note("no GEX: the tissue mask is built from the receptor spots and the regions are not characterised")
    grid = Grid.from_points(T[S.X].to_numpy(), T[S.Y].to_numpy(), pixel_um=pixel_um)
    tissue = tissue_mask(T, grid=grid, pad_edges=not legacy)
    on = grid.index(L[S.X].to_numpy(), L[S.Y].to_numpy())[2]
    L["on_grid"] = on
    qc_rows.append({"step": "loci on the map grid", "records": int(on.sum()), "note": tissue_src})

    # ============================================================================================ 3. CSR
    map_kw = {"sigma_um": sigma_um, "tissue": tissue, "grid": grid, "min_neff": min_neff, "alpha": alpha}
    I = L[L["switched"].notna().to_numpy() & on].reset_index(drop=True)
    I["switched"] = I["switched"].astype(bool)
    csr = None
    if len(I) and I["switched"].nunique() == 2:
        g3 = np.random.default_rng(int(rng)) if legacy else gen("csr")
        csr = csr_hotspots(I, blocks=None, n_perm=n_perm, rng=g3, **map_kw, **legacy_kw)
    else:
        note("CSR map skipped: fewer than two isotyped loci classes (switched and unswitched) on the grid")

    # ============================================================================================ 4. SHM
    E = L[L["evaluable"].to_numpy() & on].reset_index(drop=True)
    E["mutated"] = E["mutated"].fillna(False).astype(bool)
    E["isotype_class"] = E["isotype_class"].astype(str)
    shm_raw = shm_iso = csr_test = csr_clone_test = None
    g_test = g_region = None
    if len(E) and E["mutated"].nunique() == 2:
        if legacy:
            g7 = np.random.default_rng(int(rng))
            null_raw, null_iso = _legacy_t7_nulls(
                E["mutated"].to_numpy(float), E["isotype_class"].to_numpy(), n_perm, g7
            )
            shm_raw = shm_hotspots(E, match_by=None, blocks=None, null_marks=null_raw, **map_kw, **legacy_kw)
            shm_iso = shm_hotspots(E, match_by="isotype_class", blocks=None, null_marks=null_iso, **map_kw, **legacy_kw)
            g_region = copy.deepcopy(g7)
            g_test = g7
        else:
            shm_raw = shm_hotspots(E, match_by=None, blocks=None, n_perm=n_perm, rng=gen("shm_raw"), **map_kw)
            shm_iso = shm_hotspots(
                E, match_by="isotype_class", blocks=None, n_perm=n_perm, rng=gen("shm_iso"), **map_kw
            )
            g_test, g_region = gen("csr_shm_test"), gen("region_test")
        if csr is not None and _hot_ids(csr):
            csr_test = region_test(
                E,
                regions=csr.labels,
                match_by="isotype_class",
                blocks=None,
                n_perm=n_perm_region,
                min_units=5,
                rng=g_test,
            )
            if clone_null:  # confirmatory: CSR hotspots are selected on switching alone, whole clones relabelled
                csr_clone_test = optional(
                    "clone-level SHM test of the CSR hotspots",
                    lambda: _csr_hotspot_column(
                        region_test(
                            E,
                            regions=csr.labels,
                            match_by="isotype_class",
                            blocks=S.CLONE,
                            n_perm=n_perm_region,
                            min_units=5,
                            rng=gen("csr_shm_clone"),
                        )
                    ),
                )
    elif has_align:
        note("SHM maps skipped: the evaluable loci on the grid are all mutated or all unmutated (or none)")

    # ============================================================================================ 5. regions
    lab_csr = csr.labels.values if csr is not None else None
    shm_src = shm_raw if shm_from == "raw" else shm_iso
    lab_shm = shm_src.labels.values if shm_src is not None else None
    csr_ids, shm_ids = _hot_ids(csr), _hot_ids(shm_src)
    groups: list[tuple[list[int], list[int]]] = []
    if csr_ids and shm_ids:
        shape = (lambda m: _poly(m)) if overlap == "polygon" else (lambda m: m)
        cpoly = {h: shape(lab_csr == h) for h in csr_ids}
        spoly = {s: shape(lab_shm == s) for s in shm_ids}
        union = shape(np.isin(lab_shm, shm_ids))
        links: dict[int, set[int]] = {}
        for h in csr_ids:
            if (cpoly[h] & union).any():
                linked = {s for s in shm_ids if (cpoly[h] & spoly[s]).any()}
                if not linked:  # the union outline bridged two SHM hotspots: take those within one pixel
                    near = ndi.binary_dilation(cpoly[h])
                    linked = {int(s) for s in np.unique(lab_shm[near]) if s in shm_ids}
                links[h] = linked
        # CSR hotspots linked through a shared SHM hotspot form one region (connected components)
        todo = sorted(links)
        while todo:
            seed = todo.pop(0)
            comp_c, comp_s, stack = {seed}, set(links[seed]), [seed]
            while stack:
                h = stack.pop()
                for h2 in list(todo):
                    if links[h2] & (links[h] | comp_s):
                        todo.remove(h2)
                        comp_c.add(h2)
                        comp_s |= links[h2]
                        stack.append(h2)
            groups.append((sorted(comp_c), sorted(comp_s)))
    region_labels = Raster(np.zeros(grid.shape, dtype=np.int32), grid, name="regions")
    rows: list[dict[str, Any]] = []
    if groups:
        axis_xy = T[[S.X, S.Y]].to_numpy(dtype=float)
        centre, axis = _long_axis(axis_xy)
        gx, gy = grid.centres()
        info = []
        for cs, ss in groups:
            pix = np.isin(lab_csr, cs)
            ii, jj = np.nonzero(pix)
            cx, cy = float(gx[ii].mean()), float(gy[jj].mean())
            info.append((float((np.array([cx, cy]) - centre) @ axis), cs, ss, pix, cx, cy))
        info.sort(key=lambda t: t[0])
        lab = np.zeros(grid.shape, dtype=np.int32)
        for k, (_, cs, ss, pix, cx, cy) in enumerate(info, start=1):
            lab[pix & (lab == 0)] = k
            rows.append({"region_id": k, "csr_hotspots": ";".join(map(str, cs)), "shm_hotspots": ";".join(map(str, ss)),
                         "x_um": cx, "y_um": cy, "area_um2": float(pix.sum() * grid.pixel_area_um2),
                         "_csr": cs, "_shm": ss})  # fmt: skip
        region_labels = Raster(lab, grid, name="regions")
    elif not csr_ids:
        note("no CSR ∩ SHM region: no CSR hotspot on the map")
    elif not shm_ids:
        note(f"no CSR ∩ SHM region: no {shm_from} SHM hotspot on the map; the CSR hotspots are returned")
    else:
        note(
            f"no CSR ∩ SHM region: none of the {len(csr_ids)} CSR hotspots overlaps one of the {len(shm_ids)} "
            f"{shm_from} SHM hotspots ({overlap} rule); the hotspots are returned"
        )

    # loci and records: hotspots and regions
    xs, ys = L[S.X].to_numpy(), L[S.Y].to_numpy()

    def sample(res: HotspotResult | None) -> np.ndarray:
        if res is None:
            return np.zeros(len(L), dtype=np.int64)
        return np.where(on, res.labels.sample(xs, ys, fill=0), 0).astype(np.int64)

    L["csr_hotspot"], L["shm_hotspot"], L["shm_iso_hotspot"] = sample(csr), sample(shm_raw), sample(shm_iso)
    L["region_id"] = np.where(on, region_labels.sample(xs, ys, fill=0), 0).astype(np.int64)
    I_reg = I.assign(region_id=np.asarray(region_labels.sample(I[S.X].to_numpy(), I[S.Y].to_numpy(), fill=0)))
    E_reg = E.assign(region_id=np.asarray(region_labels.sample(E[S.X].to_numpy(), E[S.Y].to_numpy(), fill=0)))

    # ============================================================================================ 6. region tests
    rt = rt_clone = csr_clone = shm_clone = None
    if rows:
        if len(E) and E["mutated"].nunique() == 2:
            rt = region_test(
                E,
                regions=region_labels,
                match_by="isotype_class",
                blocks=None,
                n_perm=n_perm_region,
                min_units=5,
                rng=g_region,
            )
        if clone_null:
            clone_kw = dict(map_kw)
            csr_clone = optional(
                "clone-level CSR map",
                lambda: csr_hotspots(
                    I, blocks=S.CLONE, loci_radius_um=loci_radius_um, n_perm=n_perm, rng=gen("csr_clone"), **clone_kw
                ),
            )
            if len(E) and E["mutated"].nunique() == 2:
                shm_clone = optional(
                    "clone-level SHM map",
                    lambda: shm_hotspots(
                        E,
                        match_by="isotype_class",
                        blocks=S.CLONE,
                        loci_radius_um=loci_radius_um,
                        n_perm=n_perm,
                        rng=gen("shm_clone"),
                        **clone_kw,
                    ),
                )
                rt_clone = optional(
                    "clone-level region test",
                    lambda: region_test(
                        E,
                        regions=region_labels,
                        match_by="isotype_class",
                        blocks=S.CLONE,
                        n_perm=n_perm_region,
                        min_units=5,
                        rng=gen("region_clone"),
                    ),
                )
    rtab = rt.table.assign(key=rt.table.region.astype(str)).set_index("key") if rt is not None else None
    ctab = (
        rt_clone.table.assign(key=rt_clone.table.region.astype(str)).set_index("key") if rt_clone is not None else None
    )
    for r in rows:
        k = r["region_id"]
        pix = region_labels.values == k
        gi, ge = I_reg[I_reg.region_id == k], E_reg[E_reg.region_id == k]
        r["n_loci"] = int((L.region_id == k).sum())
        r["n_isotyped"], r["n_switched"] = len(gi), int(gi["switched"].sum())
        r["switched_share"] = r["n_switched"] / r["n_isotyped"] if r["n_isotyped"] else np.nan
        r["n_evaluable"], r["n_mutated"] = len(ge), int(ge["mutated"].sum())
        r["mutated_share"] = r["n_mutated"] / r["n_evaluable"] if r["n_evaluable"] else np.nan
        row = rtab.loc[str(k)] if (rtab is not None and str(k) in rtab.index) else None
        r["shm_expected"] = float(row.expected) if row is not None else np.nan
        r["shm_excess_ratio"] = float(row.ratio) if row is not None else np.nan
        r["p"] = float(row.p_value) if row is not None else np.nan
        r["q"] = float(row.q_value) if row is not None else np.nan
        r["csr_peak_z"], _ = _region_peak(csr, pix)
        r["shm_peak_z"], _ = _region_peak(shm_src, np.isin(lab_shm, r["_shm"]) if lab_shm is not None else pix)
        ct = csr.table.set_index("hotspot") if csr is not None else None
        st = shm_src.table.set_index("hotspot") if shm_src is not None else None
        r["csr_p_map"] = float(ct.loc[r["_csr"], "p_value"].min()) if ct is not None else np.nan
        r["shm_p_map"] = float(st.loc[r["_shm"], "p_value"].min()) if (st is not None and r["_shm"]) else np.nan
        r["csr_clone_peak_z"], r["csr_clone_p"] = _region_peak(csr_clone, pix)
        r["shm_clone_peak_z"], r["shm_clone_p"] = _region_peak(shm_clone, pix)
        crow = ctab.loc[str(k)] if (ctab is not None and str(k) in ctab.index) else None
        r["shm_clone_region_p"] = float(crow.p_value) if crow is not None else np.nan
        r["shm_clone_null_mean"] = float(crow.null_mean) if crow is not None else np.nan

    # ============================================================================================ 7. GEX
    module_maps: dict[str, pd.Series] = {}
    profiles = pd.DataFrame()
    windows = pd.DataFrame(columns=["region_id", "x0", "x1", "y0", "y1"])
    gc_gex = None
    prof = None
    if rows:
        wrows = []
        for r in rows:
            ii, jj = np.nonzero(region_labels.values == r["region_id"])
            x0, y0 = grid.origin
            wrows.append(
                {
                    "region_id": r["region_id"],
                    "x0": x0 + ii.min() * grid.pixel_um - window_pad_um,
                    "x1": x0 + (ii.max() + 1) * grid.pixel_um + window_pad_um,
                    "y0": y0 + jj.min() * grid.pixel_um - window_pad_um,
                    "y1": y0 + (jj.max() + 1) * grid.pixel_um + window_pad_um,
                }
            )
        windows = pd.DataFrame(wrows, columns=["region_id", "x0", "x1", "y0", "y1"])
    score_mask = None
    if gex is not None:
        tk = tissue_key if (tissue_key is not None and tissue_key in gex.obs) else None
        sel = gexmod.unit_mask(gex, score_units)  # a wrong column or length is the caller's error: raise here
        score_mask = pd.Series(sel & units["tissue"].to_numpy(dtype=bool), index=gex.obs_names.copy(), name="scored")
        mm = optional(
            "module maps",
            lambda: gexmod.module_scores(
                gex, markers, smooth_um=module_smooth_um, tissue_key=tk, units=score_mask.to_numpy()
            ),
        )
        if mm is not None:
            module_maps = {c: mm[c] for c in mm.columns}
        if rows:
            prof = optional(
                "region profiles",
                lambda: gexmod.profile_regions(
                    gex, region_labels, markers=markers, window=windows, tissue_key=tk, mixed_log2=mixed_log2
                ),
            )
        if gc_call:
            gc_gex = optional("GC-B call", lambda: _gc_call(gex, tk, species, gc_options))
        units["scored"] = score_mask.to_numpy()
        units["region_id"] = np.asarray(region_labels.sample(units[S.X].to_numpy(), units[S.Y].to_numpy(), fill=0))
        units["in_window"] = prof.units["in_window"].to_numpy() if prof is not None else False
        units["gc_gex"] = gc_gex.to_numpy() if gc_gex is not None else False
    if prof is not None:
        profiles = prof.table
        lab = prof.labels.set_index("region_id")
        for r in rows:
            a = lab.loc[r["region_id"]] if r["region_id"] in lab.index else None
            r["gex_label"] = a.label if a is not None else None
            r["gex_ratio_window"] = float(a.top_ratio_window) if a is not None else np.nan
            r["gex_second_label"] = a.second if a is not None else None
            r["gex_second_ratio"] = float(a.second_ratio_window) if a is not None else np.nan
            r["gex_mixed"] = bool(a.mixed) if a is not None else False
            r["gex_ratio_tissue"] = float(a.top_ratio_tissue) if a is not None else np.nan
            r["n_gex_units"] = int(a.n_units_inside) if a is not None else 0
            r["n_gex_window_units"] = int(a.n_units_window) if a is not None else 0
    if gex is not None:  # marker genes missing in their case but present in another: the species does not match
        n_und = sum(1 for r in rows if r.get("gex_label") == "undetermined")
        hint = _species_hint(gex, markers, species, n_und, len(rows))
        if hint is not None:
            messages.append(hint)
            warnings.warn(hint, WorkflowWarning, stacklevel=2)
    reg_cols = [
        "region_id", "csr_hotspots", "shm_hotspots", "x_um", "y_um", "area_um2", "n_loci", "n_isotyped",
        "n_switched", "switched_share", "n_evaluable", "n_mutated", "mutated_share", "shm_expected",
        "shm_excess_ratio", "p", "q", "csr_peak_z", "shm_peak_z", "csr_p_map", "shm_p_map", "csr_clone_peak_z",
        "csr_clone_p", "shm_clone_peak_z", "shm_clone_p", "shm_clone_region_p", "shm_clone_null_mean", "gex_label",
        "gex_ratio_window", "gex_second_label", "gex_second_ratio", "gex_mixed", "gex_ratio_tissue", "n_gex_units",
        "n_gex_window_units",
    ]  # fmt: skip
    regions = pd.DataFrame([{c: r.get(c, np.nan) for c in reg_cols} for r in rows], columns=reg_cols)
    if len(regions):
        regions["region_id"] = regions["region_id"].astype(np.int64)
        if gex is None:
            regions["gex_label"] = None

    # ============================================================================================ 8. assemble
    rec = H.rename(columns={"shm_mismatches": "shm_mutations"})
    rec["isotype_class"] = rec["isotype"]
    rec["isotype"] = rec["isotype_subclass"].astype("string").fillna(rec["isotype_class"].astype("string"))
    rec["mutated"] = rec["shm_mutated"].astype(bool)
    rec["evaluable"] = rec["shm_evaluable"].astype(bool)
    rec["region_id"] = rec[S.LOCUS_ID].map(L.set_index(S.LOCUS_ID)["region_id"]).fillna(0).astype(np.int64).to_numpy()
    lead = [S.SPOT, S.X, S.Y, S.LOCUS, S.CLONE, "isotype", "isotype_class", "isotype_subclass", "isotype_position",
            "switched", "shm_mutations", "shm_excess", "mutated", "evaluable", S.LOCUS_ID, "region_id"]  # fmt: skip
    rec = rec[[c for c in lead if c in rec.columns] + [c for c in rec.columns if c not in lead]]
    loci_cols = [S.LOCUS_ID, S.CLONE, S.X, S.Y, "n_spots", "n_records", "reads_max", S.SPOT, *_ISO_CLASSES,
                 "n_isotyped", "switched_share", "isotype_class", "switched", "n_evaluable", "excess", "evaluable",
                 "mutated", "on_grid", "csr_hotspot", "shm_hotspot", "shm_iso_hotspot", "region_id"]  # fmt: skip
    L = L[[c for c in loci_cols if c in L.columns] + [c for c in L.columns if c not in loci_cols]]
    qc = pd.DataFrame(qc_rows, columns=["step", "records", "note"])
    if not rows:
        messages.insert(0, next((m for m in messages if m.startswith("no CSR ∩ SHM")), "no CSR ∩ SHM region"))
        messages = list(dict.fromkeys(messages))
    params = {
        "species": species,
        "min_reads": min_reads,
        "min_reads_rule": min_reads_rule,
        "record_level": level,
        "loci_radius_um": loci_radius_um,
        "sigma_um": sigma_um,
        "pixel_um": pixel_um,
        "n_perm": n_perm,
        "n_perm_region": n_perm_region,
        "alpha": alpha,
        "min_neff": min_neff,
        "clone_null": clone_null,
        "markers": markers if isinstance(markers, str) else {k: list(v) for k, v in markers.items()},
        "shm_from": shm_from,
        "overlap": overlap,
        "window_pad_um": window_pad_um,
        "tissue_key": tissue_key,
        "tissue_from": tissue_src,
        "gc_call": gc_call,
        "gc_options": {"top_frac": 0.01, "widen_um": 0.0, **dict(gc_options or {})} if gc_call else None,
        "module_smooth_um": module_smooth_um,
        "score_units": (
            score_units if (score_units is None or isinstance(score_units, str | tuple)) else "flags per unit"
        ),
        "n_scored_units": int(score_mask.sum()) if score_mask is not None else 0,
        "mixed_log2": mixed_log2,
        "clone_method": clone_method,
        "shm_alignments": shm_source,
        "shm_min_reads": shm_kw.get("min_reads", shm_min_reads),
        "shm_min_reads_rule": shm_rule,
        "shm_min_aligned_nt": shm_kw.get("min_aligned_nt", shm_min_aligned_nt),
        "shm_baseline_fallback": shm_kw.get("baseline_fallback", "none") if has_align else None,
        "shm_options": dict(shm_options or {}),
        "min_excess": min_excess,
        "legacy": legacy,
        "rng": rng if isinstance(rng, int | np.integer) else str(rng),
        "seeds": seeds
        if not legacy
        else {
            **seeds,
            "csr": f"default_rng({rng})",
            "shm_raw, shm_iso, csr_shm_test, region_test": f"default_rng({rng}), the archive's T7 draw order",
        },
        "neff_kernel": "legacy" if legacy else "exact",
        "threshold_rule": "quantile" if legacy else "rank",
        "grid": grid,
        "null_designs": {
            "csr, shm, shm_iso": "loci shuffled one by one (shm_iso: within isotype class), positions fixed",
            "p, csr_shm_test": "mutated calls shuffled among evaluable loci within isotype class (p: descriptive)",
            "csr_clone_p, shm_clone_p": "whole clones relabelled (matched sizes and isotype mix), studentised max-T",
            "shm_clone_region_p": "whole clones relabelled, region fixed (descriptive: regions are selected on SHM)",
            "csr_shm_clone": "whole clones relabelled, CSR hotspots fixed (confirmatory: hotspots selected on CSR "
            "only); BH over the CSR hotspots",
        },
        "shm_baseline": baseline_table,
        "reader_meta": {k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)},
        "warnings": warn_log,
        "messages": messages,
    }
    return HotspotWorkflowResult(
        records=rec,
        loci=L,
        grid=grid,
        tissue=tissue,
        csr=csr,
        shm=shm_raw,
        shm_iso=shm_iso,
        regions=regions,
        profiles=profiles,
        module_maps=module_maps,
        windows=windows,
        gc_gex=gc_gex,
        params=params,
        region_labels=region_labels,
        units=units,
        csr_shm_test=csr_test,
        csr_clone=csr_clone,
        shm_clone=shm_clone,
        qc=qc,
        messages=tuple(messages),
        score_mask=score_mask,
        csr_shm_clone=csr_clone_test,
        gex=gex,
    )


def _gc_call(gex: Any, tissue_key: str | None, species: str, options: Mapping[str, Any] | None) -> pd.Series:
    """Germinal-centre B units from the GEX alone (call_regions on a light copy; the user's AnnData is not changed)."""
    import anndata as ad

    from spaclone.gex.qc import _bool_obs, get_counts, get_xy
    from spaclone.gex.regions import call_regions

    tis = _bool_obs(gex, tissue_key) if tissue_key is not None else np.ones(gex.n_obs, bool)
    light = ad.AnnData(
        X=get_counts(gex, None, check=False),
        obs=pd.DataFrame({"in_tissue": tis}, index=gex.obs_names.copy()),
        var=pd.DataFrame(index=pd.Index(gex.var_names).astype(str)),
    )
    light.obsm["spatial"] = get_xy(gex)
    kw = {"top_frac": 0.01, "widen_um": 0.0, **dict(options or {})}
    call_regions(light, kind="germinal_centre", species=species, tissue_key="in_tissue", **kw)
    return pd.Series((light.obs["region"].astype(str) == "GC").to_numpy(), index=gex.obs_names.copy(), name="gc_gex")
