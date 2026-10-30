# Tutorials

Executed Jupyter notebooks, in reading order. The first needs no download; the others use public data:

* the AIR-SPACE mouse lymph nodes (Jiang et al., *PNAS* 2026; GEO GSE286452): all six nodes, Mock and days 3–21, are
  built into {mod}`spaclone.datasets`, downloaded once and verified by sha256 (`pip install "spaclone[datasets]"`);
* a Stereo-XCR-seq lung adenocarcinoma chip (Zhan et al., *Nat Commun* 2026; GEO GSE328481, sample GSM9684211), read
  from the path in the environment variable `SPACLONE_LUAD_H5AD` (notebooks 02 and 05);
* the StereoIR mouse spleen (Stereo-seq 5′ immune-repertoire demonstration files: bin20 transcriptome and TRUST4 AIRR
  contigs), read from the folder in `SPACLONE_STEREOIR` (notebook 06).

Each notebook says at the top where its files come from.

| Notebook | What it shows |
|---|---|
| {doc}`01_quickstart` | One `spaclone.run` call on two synthetic sections (30% and 0% of molecules in their own cell): the two verdicts, the test table, the report figure and the methods text. Runs in under a minute. |
| {doc}`02_is_my_data_spatial` | The core use case on real data, AIR-SPACE vs Stereo-XCR-seq LUAD: reading, receptor QC and filtering, cell types, the trust report, and published-style quality metrics re-run under placebos (random placement and random labelling within Ig RNA strata). |
| {doc}`03_bcr_class_switching_and_mutation` | BCR biology in tissue: isotypes and inferred cells, switched-share maps with clone-level hotspots, isotype transitions within clones, somatic mutation from MiXCR alignments (and what `bcr.shm` needs), germinal-centre region tests and lineage trees with tissue positions. |
| {doc}`04_transcriptome_and_repertoire` | The GEX workflow (QC, filtering, clustering, cell types, germinal centres, pseudobulk DE with GCs as replicates, neighbourhood enrichment) and the spatial repertoire toolkit (expansion, rarefied diversity and overlap against random placement of intact clones, gene usage, clonotype networks, clone dispersion, public clonotypes across six mice, a niche profile along the GC border). |
| {doc}`05_custom_placebo` | Writing your own statistic and testing it with `nulls.Permute`, `nulls.Relocate`, `nulls.placebo` and `nulls.compare_designs`; why the null must keep the confounders (Ig RNA depth strata, whole units with their read counts). |
| {doc}`06_hotspots_and_lineages` | The flagship workflow on a Stereo-seq mouse spleen, from two files: the trust report, class-switch and mutation hotspots with their overlap regions and GEX identity (`bcr.hotspot_workflow`, `pl.hotspot_atlas`, `pl.csr_map`), the confirmatory clone-level SHM test, lineage trees on the tissue (`bcr.lineage_workflow`, `pl.lineage_tree`), sciCSR transitions over tissue regions and Threadfin clone states. |

```{toctree}
:maxdepth: 1
:hidden:

01_quickstart
02_is_my_data_spatial
03_bcr_class_switching_and_mutation
04_transcriptome_and_repertoire
05_custom_placebo
06_hotspots_and_lineages
```
