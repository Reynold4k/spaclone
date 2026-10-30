# Why spaclone

## The problem

Sequencing-based spatial immune-receptor methods (Spatial VDJ, Slide-TCR-seq, SPTCR-seq, Stereo-XCR-seq, AIR-SPACE,
circVDJ-seq, SpaCir-VDJ, …) attach a spatial barcode to each receptor molecule through capture, PCR and often
circularisation. A template switch during amplification joins a receptor's CDR3 to the barcode of an unrelated
captured cDNA. The receptor sequence is still correct; its position is not. Nothing on the molecule records the
swap, and one of these methods documents barcode switching in its own amplification step (Slide-TCR-seq,
replaced in its 2023 protocol).

How do the published methods show that positions are right? A survey of 13 technology classes (2026-10) found the same
pattern in all of them: receptor density agrees with B- or T-cell density, receptor-positive cells express Ig or TCR
genes, chain counts correlate between spots. None re-ran these checks after moving receptors to random places;
the tutorial {doc}`tutorials/02_is_my_data_spatial` re-runs such metrics under random placement on public data.

## What exists

| Need | Closest existing tool | What is missing |
|---|---|---|
| Is a receptor molecule on its own cell? | Density / cell-level concordance in each method paper | A chance model, a molecule-level test, a bound on the fraction correctly placed |
| Barcode + UMI of a receptor found in GEX | Slide-TCR-seq *rescues* reads that match a GEX read | A rate with decoys; a per-molecule flag |
| Barcode sinks, off-tissue receptors | Discarded as noise in several pipelines | Used as *estimates* of mis-location |
| Re-run a metric under random placement | spatstat `envelope(simulate=)` for point-pattern summaries; one-claim shuffles in single papers | A general engine for any metric or claim, with stratified and local designs |
| Same-clone co-localisation | squidpy `nhood_enrichment` (z only, no strata), scirpy `clonotype_modularity` (unstratified) | Depth-stratified test that separates working from broken assays |
| Read floor against chimeras | Fixed or knee-plot floors | A floor derived from partner-chain sharing above chance |
| Clonal-feature hotspots | Getis–Ord / KDE maps without matched expectation or map-level error control | Covariate-matched expectation (e.g. isotype), family-wise hotspots |
| Clone concentrated in space | Kulldorff scans in R (one pattern per call, fixed simulations) | A scan calibrated across thousands of clones, sequential Monte-Carlo p-values |
| Heavy/light pairing from co-occurrence | repair, SPaCeD | A null that does not mistake chimeric spots for cells |
| Niches along a structure border | Signed distances (sosta, imcRtools, SPIAT) without a null | Permutation bands and a global test |
| Lineage trees with positioned tips | dowser trait tests on discrete labels | Covariate-matched, within-clone tests; tissue distance |

No spatial receptor package was on PyPI, CRAN or Bioconductor in October 2026; the only spatial-AIR toolkits found
(SpaXCR 0.1.1, a Stereo-seq TCR command-line wrapper) offer smoothing and unstratified pairing, no location trust.

## What spaclone offers

1. **A location-trust report** (`spaclone.run`, `qc`): same-spot test against the GEX library with a located-fraction
   bound; UMI match with decoys; barcode-sink and off-tissue diagnostics with a swap-rate estimate; share on own cell
   type against random placement; clone composition by cell type; depth-stratified same-clone co-localisation;
   chimera-aware heavy/light pairing; a data-driven read floor. Two verdicts, one per level: molecules on their own
   cell, and clone identity following position.
2. **A placebo engine** (`nulls`) that re-runs any metric, including a published one, under declared null designs.
3. **Calibrated concentration statistics** (`tl`): covariate-matched share maps with map-level hotspots, a calibrated
   per-clone scan, niche profiles, region sharing against placement, lineage tests, and repertoire statistics against
   random placement of intact clones.
4. **B-cell biology on tissue** (`bcr`, `pl`): from two files to class-switch and mutation hotspots, the regions where
   they overlap and what those regions are made of (`bcr.hotspot_workflow`, `pl.hotspot_atlas`, `pl.csr_map`), with a
   confirmatory clone-level mutation test; germline-rooted lineage trees with every tip on the tissue
   (`bcr.lineage_workflow`, `pl.lineage_tree`); sciCSR's class-switch potential and transition path theory on spatial
   graphs (`bcr.csr_potential`, `tl.transition_model`, `tl.tpt_flux`).
5. **Readers and bridges**: `spaclone.read` for the mainstream spatial and receptor formats, and bridges to scirpy,
   squidpy, Dandelion, Immcantation, sciCSR, Threadfin and SpatialData ({doc}`ecosystem`).

Each statistic is validated by reproducing the numbers of the analyses it came from (see {doc}`validation`) and by
calibration tests under its null in the test suite.
