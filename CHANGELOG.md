# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/).

## 1.0.2 (2026-11-16)

- **Every documented function runs on data anyone can get.** `datasets.synthetic_example(sequences=True)` (and
  `sim.synthetic_chip(sequences=True, shm_mutations=, seq_error_rate=, truncated_fraction=)`) gives every record of the
  simulated section its contig and V and J alignments, in the AIRR fields IgBLAST writes, with somatic hypermutation
  along planted B-cell lineage trees (synthetic germline genes; every other draw of the section unchanged). Lineage
  trees (`bcr.lineage_workflow`, `pl.lineage_tree`, `pl.lineage_gallery`) therefore run without a download, and the
  new `sim.score_lineages` checks each tree against the planted one (mutations recovered, Robinson-Foulds distance,
  tips in place). New tutorial `08_lineages_and_clone_states`: these trees, then `pl.region_zoom`,
  `pl.stationary_distribution`, `pl.clone_axis_profile` and `pl.spanning_clones` on the public AIR-SPACE day-21 node.
  Tutorial 06 is marked everywhere as a showcase on BGI data that are not yet public and cannot be re-run until they
  are released.
- **A first run that works.** The README quick start reads a public AIR-SPACE node (`spaclone.datasets.airspace`), its
  biology block names the data each part runs on, and a new page documents the command line with a worked example
  (`docs/cli.md`). `check_install(require=[...])` checks the extras a script needs and stops with an `ImportError` that
  names the missing ones and the `pip install "spaclone[...]"` command; every tutorial starts with it and states its
  install line (Jupyter included). The install hints for the IgBLAST/R environment script say to clone the repository
  first (the script is not in the pip package).
- **AIR-SPACE γδ chains.** The GEO clone tables write `receptor_type` "NA" for TRG reads; `io.read_airspace` (and
  `spaclone.read`) now take their locus from the V, J or C call, so they are TRG records instead of a locus "NA"
  (Mock 12, D3 11, D7 2, D10 2 molecules).
- **Lineage noise floor for molecules.** `bcr.lineage_workflow(noise_min_reads="auto")`: 30 reads for assembled
  contigs (the archive's rule, unchanged), 1 for records that are single molecules or reads, whose reads are copies of
  one molecule (a 30-read floor reverted every mutation of such data).
- **No red pulp on a lymph node.** The compartment outside the T zones and the B-rich area is called the red pulp
  only where it is erythroid-rich, else the "non-B/T area" (`bcr.lineage_workflow(compartment_names="auto")`,
  `gex.call_regions(kind="red_pulp", names="auto")`), as `ext.threadfin` already did; spleen results are unchanged.
- **Plots.** The trust-report title wraps to the figure width, so `spaclone run` no longer cuts long verdicts in
  `report.png`; `pl.gex_volcano` puts its legend above the plot and places gene labels without overlaps, with leader
  lines; `ext.scirpy.clonotype_modularity().plot()` draws every unit grey, with "no clonotype at q < 0.05", when no
  clonotype is significant (it coloured the largest ones); the lineage-figure legend lists the location classes the
  run named.
- **Dandelion on Python 3.13.** The `dandelion` extra, and with it `full`, pins `scanpy<1.13`: sc-dandelion 1.0.1
  does not import with scanpy 1.13.0, which pip installs on Python 3.13 (the core install keeps scanpy 1.13). The pin
  goes when Dandelion supports scanpy 1.13.

## 1.0.1 (2026-10-31)

- `bcr.hotspot_workflow(region_source="auto")`: without a CSR ∩ SHM region, each CSR hotspot becomes a region (`regions.source`), with its shares, clone-level CSR p, confirmatory SHM q (`shm_clone_q`), GEX profile and zoom ("CSR hotspot N" in `pl.hotspot_atlas` / `pl.region_zoom`); `min_hotspot_loci` (default 1, `tl.hotspots(min_units=)`) drops hotspots that hold no locus.
- `spaclone.read`, `io.read_airspace` and `datasets.load_airspace` keep MiXCR's V alignments by default (`alignments="auto"`), so SHM runs from the main entry point.
- Plots: share maps (`pl.share_map`, the atlas, `pl.csr_map`) cover the tissue with a smooth outline; the legend says why a unit has no score, and the atlas note wraps instead of being cut. `pl.spatial_records` colours highlighted values before the category cut, so `ext.scirpy.clonotype_modularity().plot()` shows its q < 0.05 clonotypes. `pl.csr_flow(kind="grid")` arrows fit the grid pitch. `pl.same_spot_multiplicity` no longer warns on classes without an expectation. The `pl.tree_tissue` legend sits above the map. `ext.threadfin` compartment names follow the labels (no "red pulp" on a lymph node; numbers unchanged). `pl.spot_qc_map` gains `style="dark"` and a visible low end. `pl.clone_map` uses the README hero settings by default. `pl.diversity` draws the rarefied interval even when the full-sample statistic lies outside it.

## 1.0.0 (2026-10-31)

First public release.

### Highlights

- **Read any format.** `spaclone.read(gex, airr)` reads the spatial transcriptome and the receptor table of a section
  into one `SpatialAIR`, with coordinates in micrometres, and recognises both files from their contents: h5ad
  (scanpy, squidpy, Stereopy / SAW, Curio), Space Ranger Visium and Visium HD, Stereo-seq GEF / GEM, Xenium,
  Slide-seq and SpatialData; AIRR rearrangement tables (Cell Ranger, TRUST4, IgBLAST / Change-O, MiXCR, scirpy,
  Dandelion), 10x contigs, MiXCR, TRUST4 reports and Change-O tables, and other tables with a column mapping.
  `io.sniff(path)` says what a file is and what must still be given; `io.read_airspace` keeps MiXCR's alignments
  (`alignments=True`) and checks that the two files belong together (`min_overlap`, `unmatched`).
- **QC and filters.** Receptor QC metrics and read-support curves (`pp.qc_metrics`); one front door for filtering
  (`pp.filter_records`: IMGT junction rules, productive flag, read floor, tissue, chain, barcode sinks); UMI and
  error-variant collapsing, contamination flags, clones, and loci (same-clone spots within 15 µm, about one cell).
  The transcriptome side (`gex`) covers QC on and off the tissue, clustering, cell types, Ig constant-gene RNA,
  regions with instances and pseudobulk DE with region instances as replicates.
- **The trust report.** `spaclone.run` runs every location-trust test the data allow and gives two verdicts: at the
  molecule level (is a molecule on its own constant-region transcript, on its own cell type?) and at the clone level
  (do clones co-localise beyond their local Ig RNA, do heavy and light chains pair?). The criteria are stated before
  the data are seen (`qc.CRITERIA`, `qc.OVERALL_RULES`), q-values are corrected within each level, and the report
  comes with a figure (`pl.trust_report`, also `compact=True`) and a methods paragraph. The cell-type column and the
  own cell types are detected only when the case is unambiguous (`cell_type_key="auto"`, `own_types="auto"`);
  otherwise the report names the candidates. The `spaclone run` command takes the same choices (`--cell-type-key`,
  `--own-types`).
- **CSR/SHM hotspots.** `bcr.hotspot_workflow` goes from the two files to class-switch (CSR) and hypermutation (SHM)
  hotspots, with family-wise thresholds over the map and clone-level nulls that relabel whole clones, the regions
  where they overlap, and what each region is made of in the transcriptome. The read floor follows the data
  (`min_reads="auto"`: 1 for molecule- or read-level records, else 2), and SHM comes from MiXCR alignments when they
  are present. `result.csr_shm_clone` is the confirmatory SHM test: every CSR hotspot (table column `csr_hotspot`),
  selected on switching alone, tested for more mutated loci than its isotypes and clone sizes predict, with BH over
  the hotspots (`shm_clone_region_p` is descriptive). `pl.hotspot_atlas` draws the atlas with zooms on the regions, `pl.csr_map`
  the isotypes and class switches along the IgH locus. Somatic hypermutation on its own: `bcr.shm` (allele baseline
  with `return_baseline=True`).
- **Lineages.** `bcr.lineage_workflow` builds germline-rooted trees per clone and per group of related clones, with
  every tip's isotype, mutations and tissue location; `pl.lineage_tree` and `pl.lineage_gallery` draw them next to
  the section. Trees need the contig sequences and their V alignments (IgBLAST's AIRR output has them); a clone
  table without them raises a `ValueError` that says so.
- **sciCSR on tissue.** CSR potential and SHM score on cells, spots, beads or loci (`bcr.csr_potential`,
  `bcr.shm_score`), Markov transition models on spatial, expression or joint graphs (`tl.transition_model`) and
  transition path theory with randomisation tests (`tl.tpt_flux`), drawn as flows on the tissue (`pl.csr_flow`,
  streamlines kept on the tissue) and as stationary distributions (`pl.stationary_distribution`, from a flux result
  or a transition model).
  `ext.scicsr` runs the original R package; 103 of 103 parity checks match sciCSR 0.3.5.
- **Threadfin in space.** `ext.threadfin.run` profiles clone states on spatial units and maps the clone axes and
  programmes back to the tissue (`pl.clone_state_map`); the tissue context (compartments, distances) is found from
  the cell-type labels (`context="auto"`, explained in `result.context_note`).
- **Bridges.** scirpy, squidpy, Dandelion, Immcantation (IgBLAST, Change-O, SHazaM, dowser) and SpatialData
  (`spaclone.ext`, `io.to_scirpy`, `io.from_scirpy`, `io.to_spatialdata`). BCR clonotypes and receptor labels use
  the heavy chain by default (`ext.scirpy` `receptor_arms="auto"`, with a `LargeClonotypeWarning` for implausibly
  large clonotypes; `ext.squidpy.receptor_labels(chains=)`; neighbourhood enrichment adds normal-approximation `p_norm` / `q_norm`).
- **Placebo engine.** Declarative null designs, random labelling (`nulls.Permute`) and random placement
  (`nulls.Relocate`), and `nulls.placebo`, which re-runs any statistic, a published metric included, on randomised
  data. Every test in the package states its null model.
- **Also:** spatial repertoire statistics (rarefied diversity and overlap against placebos, gene usage, clonotype
  networks, public clonotypes, per-clone dispersion, invariant TCR chains), share maps and the clonal scan, niche
  profiles, simulated sections with known truth (`sim`), the six AIR-SPACE lymph nodes of GEO GSE286452 as example
  data (`datasets`), and the `spaclone run` command. Every tissue map is turned to the section's long axis, draws
  y downwards, like an image of the section, and frames the main tissue: a detached speck of records far from the
  section is left out of the view and counted in a log message (`crop=False` shows every point).

### Requirements

Python 3.11–3.13 and numpy ≥ 2. Optional extras: `datasets`, `scicsr`, `threadfin`, `scverse` (scirpy, squidpy,
SpatialData), `dandelion`, `immcantation` and `tcr`, or `full` for all of them.
