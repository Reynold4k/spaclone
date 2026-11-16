# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/).

## 1.2.0 (2026-11-16)

- **Antibody-secreting cells instead of plasma cells.** Everything spaclone writes and shows now calls
  antibody-secreting cells (ASC: plasmablasts and plasma cells) "ASC". The marker genes cannot tell plasmablasts from
  plasma cells, so "plasma cells" overstated the resolution. Only labels, names and text changed: every score, test,
  call, p-value, coordinate, simulated value and figure is the same as in 1.1.0, number for number (checked on all nine
  tutorials, the synthetic example and the AIR-SPACE day-21 node). The one exception is a test that you stratify by the
  synthetic section's renamed labels (see "Orders").

  | Where | 1.1.0 | 1.2.0 |
  |---|---|---|
  | GEX marker set (`gex.PROFILE_MARKERS`; `profile_regions` sets, labels, `top`, `second`; `module_scores` columns; `bcr.hotspot_workflow` `module_maps` keys, `gex_label`, `gex_second_label`, `profiles`, the `score_` columns of `units.csv`) | `plasmablast/ASC` | `ASC` |
  | The second ASC set (Sdc1, Tnfrsf17, Slamf7), in the same places | `plasma cell` | `ASC (Sdc1+)` |
  | Label of a region whose two top sets are these two, within 1.23× | `plasmablast/ASC / plasma cell (mixed)` (drawn as "ASC / plasma cell (mixed)") | `ASC` (`top`, `second` and `mixed` still name both sets) |
  | Legend of the module map in the interactive atlas | `plasmablast/ASC module score` | `ASC module score` |
  | Region kind of `gex.call_regions` (keys of `REGION_MARKERS` and `REGION_LABELS`; `kind` in the instance table and in `uns["spaclone"]`) | `plasma_cell` | `asc` |
  | Cell type of the synthetic section (`sim.synthetic_chip`, `datasets.synthetic_example`: `cell_type`, `true_cell_type`, `meta["cells"]`; keys of `CDNA_BY_TYPE`, `CONSTANT_PER_CELL` and `meta["params"]`) | `plasma` | `ASC` |
  | Compartment of the synthetic section (`compartment`, `true_compartment`, `meta["structures"]`, `meta["lineages"]`; keys of `CDNA_BY_COMPARTMENT`) | `plasma-cell focus` | `ASC focus` |
  | Structures of the synthetic section | `PC0`, `PC1`, ... | `ASC0`, `ASC1`, ... |

- **Old spellings still work as input.** `gex.call_regions(kind=...)` takes `"plasma_cell"`, `"pc"` and `"plasma"` as
  before and gives exactly the result of `kind="asc"`. The mappings of `sim.synthetic_chip` by cell type or compartment
  (`molecules_per_cell`, `cdna_by_type`, `constant_per_cell`, `cdna_by_compartment`) accept the keys `"plasma"` and
  `"plasma-cell focus"`. The vocabularies that recognise your own labels ("Plasma cells", "plasmablasts", "PC",
  "Plasmacell", "plasma", ...: `run(own_types="auto")`, the region masks, `ext.threadfin`, the plot colours) are
  unchanged, a marker dictionary with your own set names keeps your names, and results saved with 1.1.0 still draw
  (the atlas finds their module maps). Constants and results are read with their new keys
  (`gex.REGION_MARKERS["asc"]`, `module_maps["ASC"]`).
- **Orders.** The categories of the synthetic `gex.obs["cell_type"]` and `["compartment"]` keep their 1.1.0 order
  ("ASC" where "plasma" sorted), so colours, legends and category codes are those of 1.1.0. Where spaclone sorts your
  labels by name, "ASC" now comes first: the notes and methods text of `spaclone.run` on the synthetic section list
  the own B-lineage types as "ASC, B, GC B" (before: "B, GC B, plasma"), and a static plot coloured by the synthetic
  receptor table's `cell_type` lists ASC first in its legend (same colours). Tests that visit their strata in the
  sorted order of the stratum names (`qc.colocalisation` and `qc.composition` with `within=`, `tl.niche_profile` with
  `compartment=`) therefore draw differently from 1.1.0 when you stratify them by the synthetic section's cell types,
  compartments or structures: the tests are unchanged, the names they sort are not. The arguments `n_pc_foci`,
  `pc_radius_um`, `pc_lineages` and `gc_to_pc` of `sim.synthetic_chip` keep their names and now describe the ASC
  foci.
- **Documentation.** Docstrings, the docs pages, the README and the tutorials say ASC. The tutorials were re-executed
  with 1.2.0 in the environment of their 1.0.3 run (the same package versions, Threadfin 4.6.2, one BLAS thread):
  labels and version lines are the only changes in their outputs. The docstring of
  `sim.synthetic_chip(constant_per_cell=)` gave 40 for the ASC default; the value is 30, as it was.

## 1.1.0 (2026-11-16)

- **Interactive figures.** The spatial figures (all but the Threadfin clone-state maps) and the main other ones have an
  interactive form: one HTML file with its data inside that opens by double-click in a browser, works offline, can be
  e-mailed, and is what a notebook shows inline. Zoom into a region (drag a box, scroll, pinch; panels of one section
  zoom together, the µm scale bar follows), switch to another sample or layer of the same kind, pick a gene from a
  searchable list, search clones, read any point on hover, and count what lies inside a box or a free shape (records,
  clones, isotype and chain shares, mean reads; the same numbers in Python with `fig.summary`). The colours, orientation and styles are those of the static figures.
- **`spaclone.pl.explore`** shows one or several sections in one figure: a sample switcher (e.g. the six AIR-SPACE
  lymph nodes), a layer switcher (isotype class, switched state, BCR / TCR, chain, clones, reads, the CSR and SHM share
  maps with their hotspots and regions, cell type, transcriptome regions, gene expression, UMIs), a gene picker over the
  embedded genes (`genes=300` by default: the most variable genes plus the immune marker and immunoglobulin genes;
  integer counts stored exactly, log1p(counts per 10k) computed in the figure) and a clone search.
- **`spaclone.pl.interactive`** holds the interactive forms of `spatial_records`, `isotype_map`, `clone_map`,
  `gex_map`, `gex_regions`, `spot_qc_map`, `share_map`, `csr_share_map`, `shm_share_map`, `hotspot_overview`,
  `hotspot_atlas`, `region_zoom`, `csr_map`, `csr_flow`, `lineage_tree`, `tree_tissue`, `gex_volcano`,
  `csr_transitions`, `flux_matrix`, `overlap`, `diversity`, `clonal_expansion`, `stationary_distribution`, `placebo`,
  `clonal_scan`, `read_support`, `niche_profile` and `trust_report`, with the data arguments of the static functions;
  the static functions are unchanged.
- **Dependable by design.** The drawing code is spaclone's own (no WebGL, no third-party script, nothing to install;
  `pip install "spaclone[interactive]"` is accepted and adds nothing): figures work where WebGL is off or exhausted, with
  any number of figures on a page, offline from `file://` (Chrome / Chromium and Firefox tested), and in JupyterLab 4
  and Notebook 7 (VS Code and Colab also run scripts in outputs; not tested). A notebook output also carries a PNG of
  the first view, shown where scripts do not run (an untrusted notebook, nbconvert to PDF). Large sections are drawn
  whole up to `max_points` (500,000 per sample; above it a random subset, with a note); saving prints the file size and
  warns above 50 MB. Each file carries the spaclone version, the call that made it and its licence notice; the same
  input gives the same bytes.
- New tutorial `09_interactive_figures` and docs page "Interactive figures" (where they work, sizes, precision,
  fallbacks, which plotting functions have an interactive form); `check_install()` lists the feature.

## 1.0.3 (2026-11-16)

- **Tutorials say what data you must fetch yourself.** Tutorials 02 and 05 stop at the top with one sentence that
  names the Stereo-XCR-seq LUAD chip they need (`GSM9684211_D06047A2.h5ad`, 0.98 GB, GEO GSM9684211 of GSE328481),
  its download link and `SPACLONE_LUAD_H5AD`, instead of an h5py error halfway; the README's tutorial table and the
  install page say the same. Tutorial 06, the showcase on BGI's StereoIR spleen, stops at its first cell with the
  sentence that these data are not public yet and that tutorials 07 and 08 run the same functions on public data.
  Tutorial 01's next steps lead to 07 and 08 and mark 06 as the showcase, and the documentation examples that used
  the StereoIR files (`docs/ecosystem.md`, `docs/reading_data.md`, the examples of `spaclone.read`,
  `bcr.hotspot_workflow` and `bcr.lineage_workflow`) start from public or synthetic data or say that they need them.
- **Dandelion `.h5ddl` files are detected.** `io.sniff`, `io.receptor_format` and `spaclone sniff` recognise the
  layout that Dandelion writes (an HDF5 dataset `data` holding the AIRR table, or the older PyTables layout) as
  `io.read_receptors` reads it, with that layout as evidence; any other HDF5 file named `.h5ddl` is reported, not
  guessed at. Reading one no longer raises a pandas `FutureWarning`.
- **SHM read floor by record level, and no silent zero.** `bcr.shm(min_reads="auto")` (and `bcr.allele_baseline`):
  30 reads for assembled contigs (as before, and whenever the record level is unknown), 1 when every record is one
  UMI molecule or read, the rule `bcr.lineage_workflow` and `bcr.hotspot_workflow` already used. When no record is
  evaluable, `bcr.shm` warns (`bcr.SHMWarning`) with the number each floor removed, and `bcr.hotspot_workflow` says
  that its SHM maps are empty: an empty evaluable set no longer looks like "no mutation". On the synthetic section
  (`synthetic_example(sequences=True)`, UMI molecules with at most 14 reads) 3,663 of 7,954 records are now
  evaluable instead of none. Contig data (StereoIR) and every call with an explicit floor are unchanged, tutorial
  numbers included; a direct `bcr.shm` call on AIR-SPACE molecules without `min_reads` now uses 1 instead of 30.
- **The same numbers under pandas 2 and 3.** Null designs with several strata (`nulls.Permute`, `nulls.Relocate`
  with `within=[...]` or `window_um`, the placements of `qc.own_type_share`) draw their strata in the order in which
  they first appear, whatever order pandas returns the groups in (pandas 2, which pip installs on Python 3.11, sorted
  them otherwise): tutorial 05's 300-µm design gave 1.54 instead of 1.49 on Python 3.11 and now gives the published
  numbers there too. `pl.clone_map` breaks ties between clones of the same size by first appearance (tutorial 07
  showed another day-14 clone on Python 3.11), and the rankings that choose which clones, genes or arcs a figure
  labels (`pl.spanning_clones`, `pl.csr_map`, `pl.gene_usage`, the compact `pl.trust_report`) and
  the clone ranks of `qc.composition` break ties by table order instead of by the CPU's sorting routine.
- **Clusters and dependency versions.** `gex.cluster` no longer claims that its labels do not depend on the last bits
  of the PCs: another scipy (1.16 against 1.17) changes the PCs in their sixth digit, and one changed neighbour of one
  bead gives tutorial 04 ten clusters of other sizes. The docstring, the transcriptome module and tutorial 04 say
  plainly that cluster labels can differ between dependency versions.
- **Trust report.** `report.md` of `spaclone run` heads each verdict once (no "**Power.** Power: ..."). The rows,
  criteria, methods text and figure name the chains VDJ (IGH, TRB, TRD) and VJ (IGK, IGL, TRA, TRG), terms that are
  right for B and T cells (TRA and TRG are not light chains). `print()` of every `qc` result (`ColocalisationResult`,
  `PairingResult`, `SameSpotResult`, `UmiMatchResult`, `ReadFloorResult`, `CompositionResult`, `SwapRateResult`)
  gives the one-line summary instead of every table and array.
- **Documentation.** The examples in the docstrings run with the test suite (`--doctest-modules`; the
  `tl.patristic_distances` example works under numpy 2). A GitHub Actions workflow builds the documentation site
  strictly on every push and deploys it to GitHub Pages once Pages is switched on. The Markdown pages read better on
  GitHub: every link has visible text, and the API page lists every name with the first line of its docstring as
  plain Markdown. The install page describes the `scicsr` extra correctly (CellRank kernels in
  `tl.transition_model`; deeptime's transition path theory in `tl.tpt_flux`, which gives the native backend's numbers
  80 to 90 times more slowly) and lists the versions pip installs now.
- **Tutorials re-executed** with spaclone 1.0.3 on Python 3.12 (one BLAS thread, Threadfin 4.6.2): every number is
  unchanged; the trust-report tables, methods text and figures use the new chain names. Tutorial 07 now says that its
  germinal-centre and plasma-cell calls are relative to each node (so even the uninfected node has some), what an
  "erythroid" region label means on a lymph node, and that the Mock node's flux test leaves out 683 beads that its
  bead graph does not connect to the rest; tutorials 05, 07 and 08 say which digits move with the random draws, the
  BLAS threads or the numpy and scipy versions.

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
