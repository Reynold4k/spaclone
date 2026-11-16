# Tutorials

Executed Jupyter notebooks, in reading order. Notebook 01 needs no download, and notebook 08 starts on a synthetic
section; the others use public data, except notebook 06, a showcase on data that are not yet public:

* the AIR-SPACE mouse lymph nodes (Jiang et al., *PNAS* 2026; GEO GSE286452): all six nodes, Mock and days 3–21, are
  built into {mod}`spaclone.datasets`, downloaded once and verified by sha256 (`pip install "spaclone[datasets]"`;
  notebook 07 shows every analysis on all six);
* a Stereo-XCR-seq lung adenocarcinoma chip (Zhan et al., *Nat Commun* 2026; GEO GSE328481, sample GSM9684211), which
  you download yourself: the file `GSM9684211_D06047A2.h5ad` (0.98 GB,
  [GEO link](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9684nnn/GSM9684211/suppl/GSM9684211_D06047A2.h5ad)), next to
  the notebook or at the path in the environment variable `SPACLONE_LUAD_H5AD` (notebooks 02 and 05);
* a synthetic section with receptor sequences and planted B-cell lineages,
  `spaclone.datasets.synthetic_example(sequences=True)` (simulated, no download; notebook 08);
* the StereoIR mouse spleen (Stereo-seq 5′ immune-repertoire files from BGI: bin20 transcriptome and TRUST4 AIRR
  contigs), read from the folder in `SPACLONE_STEREOIR` (notebook 06). These data are not yet public, so notebook 06
  is a showcase that cannot be re-run until BGI releases them; notebook 08 runs the functions that only notebook 06
  used on data anyone can get.

Running a notebook needs Jupyter and the extras it names in its first cell; `pip install "spaclone[full]" jupyter`
covers every notebook. Each notebook says at the top where its files come from.

| Notebook | What it shows |
|---|---|
| {doc}`01_quickstart` | One `spaclone.run` call on two synthetic sections (30% and 0% of molecules in their own cell): the two verdicts, the test table, the report figure, the methods text and the receptor QC figures (read support, valid shares, clones per spot, a per-spot QC map, same-spot hits by clones per spot). Runs in under a minute. |
| {doc}`02_is_my_data_spatial` | The core use case on real data, AIR-SPACE vs Stereo-XCR-seq LUAD: reading, receptor QC and filtering, records against the distance to the tissue edge, cell types, the trust report, and published-style quality metrics re-run under placebos (random placement and random labelling within Ig RNA strata). |
| {doc}`03_bcr_class_switching_and_mutation` | BCR biology in tissue: isotypes and inferred cells, the isotype mix by region, switched-share maps with clone-level hotspots, isotype transitions within clones, somatic mutation from MiXCR alignments (and what `bcr.shm` needs), germinal-centre region tests and lineage trees with tissue positions. |
| {doc}`04_transcriptome_and_repertoire` | The GEX workflow (QC, filtering, clustering, cell types, germinal centres, pseudobulk DE with GCs as replicates, neighbourhood enrichment) and the spatial repertoire toolkit (expansion, rarefied diversity and overlap against random placement of intact clones, gene usage, CDR3 lengths, clonotype networks, clone dispersion, a calibrated clonal scan, clone maps, scirpy clonotypes, expansion and modularity on the tissue, public clonotypes across six mice, a niche profile along the GC border and T-cell share hotspots). |
| {doc}`05_custom_placebo` | Writing your own statistic and testing it with `nulls.Permute`, `nulls.Relocate`, `nulls.placebo` (drawn by `pl.placebo`) and `nulls.compare_designs`; why the null must keep the confounders (Ig RNA depth strata, whole units with their read counts). |
| {doc}`06_hotspots_and_lineages` | Showcase on data that are not yet public (BGI; cannot be re-run until they are released). The flagship workflow on a Stereo-seq mouse spleen, from two files: the trust report, class-switch and mutation hotspots with their overlap regions and GEX identity (`bcr.hotspot_workflow`, `pl.hotspot_atlas`, `pl.region_zoom`, `pl.csr_map`), the confirmatory clone-level SHM test, lineage trees on the tissue (`bcr.lineage_workflow`, `pl.lineage_tree`, `pl.lineage_gallery`), sciCSR transitions over tissue regions (`pl.csr_flow`, `pl.flux_matrix`, `pl.stationary_distribution`) and Threadfin clone states (`pl.clone_state_map`, `pl.clone_axis_profile`, `pl.spanning_clones`). |
| {doc}`07_airspace_time_course` | Every AIR-SPACE analysis on all six public lymph nodes (Mock and days 3, 7, 10, 14 and 21) with the package defaults: the trust report of each node, CSR and SHM hotspots with the hotspot atlas, isotype maps and switches within clones, isotype and share maps, sciCSR's class-switch potential and flow with the flux test, transcriptome regions, clone and scirpy maps, the repertoire across the days, Threadfin clone states, and one table of the key numbers per day. |
| {doc}`08_lineages_and_clone_states` | The functions that only the showcase used, on data anyone can get: lineage trees on a synthetic section with planted lineages (`bcr.lineage_workflow`, `pl.lineage_tree`, `pl.lineage_gallery`), checked against the planted trees (`sim.score_lineages`), then on the public AIR-SPACE day-21 node a region zoom (`pl.region_zoom`), the stationary distribution of sciCSR's transition model (`pl.stationary_distribution`) and Threadfin clone states along the clone axis (`pl.clone_axis_profile`, `pl.spanning_clones`). |
| {doc}`09_interactive_figures` | Interactive figures: the explorer on all six AIR-SPACE lymph nodes in one file (sample switcher, layers, gene picker, clone search, selections), the numbers of a selection in Python (`fig.summary`), one figure inline and others saved as files, how many genes a file carries, and how the files behave offline, in notebooks and on GitHub. |

```{toctree}
:maxdepth: 1
:hidden:

01_quickstart
02_is_my_data_spatial
03_bcr_class_switching_and_mutation
04_transcriptome_and_repertoire
05_custom_placebo
06_hotspots_and_lineages
07_airspace_time_course
08_lineages_and_clone_states
09_interactive_figures
```
