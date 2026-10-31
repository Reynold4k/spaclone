Claude (Anthropic) develops, tests and pushes all code in this repository, and keeps it up to date as its dependencies change.

<h1 align="center">spaclone</h1>

<p align="center"><b>Spatial BCR and TCR analysis, from raw tables to biology, with every position checked against random placement</b></p>

<p align="center">
  <a href="https://pypi.org/project/spaclone/"><img src="https://img.shields.io/pypi/v/spaclone?style=flat-square&color=2a78d6&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/spaclone/"><img src="https://img.shields.io/pypi/pyversions/spaclone?style=flat-square&color=2a78d6" alt="Python versions"></a>
  <a href="https://github.com/Reynold4k/spaclone/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-MIT-2a78d6?style=flat-square" alt="MIT licence"></a>
  <a href="https://github.com/Reynold4k/spaclone/actions/workflows/test.yaml"><img src="https://img.shields.io/github/actions/workflow/status/Reynold4k/spaclone/test.yaml?branch=main&style=flat-square&label=tests" alt="tests"></a>
  <a href="https://doi.org/10.5281/zenodo.23826886"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23826886-2a78d6?style=flat-square" alt="DOI 10.5281/zenodo.23826886"></a>
</p>

<p align="center"><sub>Each figure below is the spaclone function named in its header, on an AIR-SPACE mouse lymph node (public, GEO GSE286452) and, as a showcase, the StereoIR mouse spleen (Stereo-seq, BGI; data not yet public), drawn by <a href="https://github.com/Reynold4k/spaclone/blob/main/docs/_static/homepage/make_homepage.py">make_homepage.py</a>.<br>Every other figure, with the code that draws it, is in the <a href="https://github.com/Reynold4k/spaclone/tree/main/docs/tutorials">tutorials</a>.</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/01_airspace_hotspot_overview.png" width="100%" alt="spaclone.pl.hotspot_overview on an AIR-SPACE mouse lymph node, day 21: the switched share (CSR) and the mutated share (SHM) over the node, with the CSR hotspots outlined and numbered"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/02_stereoir_hotspot_overview.png" width="100%" alt="spaclone.pl.hotspot_overview on the StereoIR mouse spleen: switched-share and mutated-share maps with the CSR and SHM hotspots and the numbered regions where they overlap"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/03_airspace_csr_map.png" width="100%" alt="spaclone.pl.csr_map on the AIR-SPACE lymph node: switched share, dominant switched class and one map per isotype in IgH locus order, arcs for the class switches within clones and the isotype mix of each region"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/04_stereoir_csr_map.png" width="100%" alt="spaclone.pl.csr_map on the StereoIR spleen: switched share, dominant switched class and one map per isotype in IgH locus order, arcs for the class switches within clones and the isotype mix of each region"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/05_airspace_csr_flow.png" width="100%" alt="spaclone.pl.csr_flow on the AIR-SPACE lymph node: sciCSR class-switch potential of every bead, with the transition flow of the chain as streamlines"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/06_stereoir_csr_flow.png" width="100%" alt="spaclone.pl.csr_flow on the StereoIR spleen: sciCSR class-switch potential of every bin, with the transition flow of the chain as streamlines"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/07_stereoir_region_zoom.png" width="100%" alt="spaclone.pl.region_zoom on the StereoIR spleen: one region where CSR and SHM hotspots overlap, with its isotypes, mutated loci and transcriptome identity"></p>

<p align="center"><img src="https://raw.githubusercontent.com/Reynold4k/spaclone/main/docs/_static/homepage/08_stereoir_lineage_tree.png" width="100%" alt="spaclone.pl.lineage_tree on the StereoIR spleen: a germline-rooted lineage tree with every tip joined to its position on the section"></p>

## Related tools

Spatial immune-repertoire analysis is young. The closest public work (October 2026):

- [Stereo-XCR-seq analysis code](https://github.com/fengyu9481/Stereo-XCR-seq-LUAD) (Zhan et al., *Nat Commun* 2026, [doi](https://doi.org/10.1038/s41467-026-75405-x)): Stereo-seq BCR maps, a Gi* SHM hotspot map and per-cell class switching.
- [AIR-SPACE](https://github.com/ShaowenJCornell/AIR-SPACE) (Jiang et al., *PNAS* 2026, [doi](https://doi.org/10.1073/pnas.2504742123)): bead-level BCR maps, regional SHM and IgPhyML trees with tips on beads.
- [Spatial VDJ](https://doi.org/10.5281/zenodo.7326539) (Engblom et al., *Science* 2023): Visium BCR/TCR, isotype regions, putative class-switch events and trees split by follicle.
- [BAITS](https://github.com/zenglab-pku/BAITS) (2026): a Python package for per-location QC, clone families and B-cell aggregates.
- [McEnany, Good & Cvijović](https://github.com/icvijovic/tonsil-spatial) (bioRxiv 2025, [doi](https://doi.org/10.1101/2025.10.31.685876)): lineage trees with follicle-labelled tips and a migration model.
- [Platypus](https://cran.r-project.org/package=Platypus) had a Visium spatial module in 2022 (isotype and SHM on the image, lineage arrows); it was removed in version 3.6.0.
- Single-cell building blocks: [sciCSR](https://github.com/Fraternalilab/sciCSR) and [Dowser](https://doi.org/10.1371/journal.pcbi.1009885) (class-switch models and tests on trees), [scirpy](https://github.com/scverse/scirpy), [Dandelion](https://github.com/tuonglab/dandelion) and [Immcantation](https://immcantation.readthedocs.io).

spaclone adds the following, and the public record shows none of it elsewhere:

- location-trust tests against depth-matched random placement;
- CSR and SHM hotspots with family-wise thresholds and clone-level nulls, and tested overlap regions with their transcriptome identity;
- class-switch potential, transitions and fluxes on spatial graphs;
- Threadfin clone states on tissue;
- one package across sequencing technologies.

## Get started

```bash
pip install spaclone            # core: reading, QC, trust tests, BCR and TCR analysis, plots
pip install "spaclone[full]"    # + every package in "Integrates" below
```

```python
import spaclone as spc

report = spc.run(spc.read("section.h5ad", "airr_rearrangement.tsv"))
```

Cell-type columns and own types are detected when unambiguous (`cell_type_key="auto"`, `own_types="auto"`) and
otherwise named in the report.

`spc.read` recognises both files from their contents: the transcriptome (h5ad from scanpy, squidpy or Stereopy, Space
Ranger Visium SD / HD, Stereo-seq GEF / GEM, Xenium, Slide-seq, SpatialData) and the receptor table (AIRR from Cell
Ranger, TRUST4, IgBLAST / Change-O, MiXCR, scirpy or Dandelion; 10x contigs; MiXCR 3 / 4; TRUST4 reports; Change-O;
Dandelion; scirpy). Other tables (immunoSEQ, BD, Parse, VDJtools, immunarch …) read with a column mapping;
`spc.io.sniff(path)` shows what was detected. `spc.run` filters the records and runs every location-trust test the
data allow:

| You get back | |
|---|---|
| `report.verdict` | Can clone positions be trusted? With `report.verdict_molecule` and `report.verdict_clone`, one per level |
| `report.tests()` | One row per test and locus: observed, chance, ratio, p, BH q, verdict |
| `report.plot()` | The one-page trust report |
| `report.methods()` | A methods paragraph with every parameter and reference |

Then the biology, on the same data:

```python
data = spc.read("section.h5ad", "airr_rearrangement.tsv")
hot = spc.bcr.hotspot_workflow(data)                # CSR and SHM hotspots, their overlap regions, GEX identity
spc.pl.hotspot_atlas(hot)                           # the flagship figure
spc.pl.csr_map(hot)                                 # isotypes and class switches along the IgH locus
trees = spc.bcr.lineage_workflow(hot)               # lineage trees with every tip on the tissue
spc.pl.lineage_gallery(trees, "lineages")           # one figure per tree and an index table
```

Lineage trees need each record's contig and its V alignment (the AIRR fields `sequence`, `v_sequence_alignment`, `v_germline_alignment` and `v_germline_start`, as IgBLAST (`-outfmt 19`) and TRUST4 write them); clone tables without sequences cannot give trees, and `bcr.lineage_workflow` says so.

## What it does

| | | |
|---|---|---|
| **Read anything**<br>Spatial transcriptome and receptor files, detected automatically | **Trust first**<br>Molecule- and clone-level tests against random placement that keeps local Ig RNA depth | **Class switching and hypermutation**<br>Switched- and mutated-share maps with family-wise hotspots |
| **Hotspot atlas**<br>Where CSR and SHM hotspots overlap, and the transcriptome identity of each region | **CSR dynamics**<br>sciCSR's CSR potential, transition models and fluxes, on tissue | **Lineages**<br>Germline-rooted trees with every tip on the section |
| **Repertoire in space**<br>Expansion, diversity, overlap, networks and public clonotypes by region | **Where clones gather**<br>Calibrated clonal scans, niche profiles and clone states | **Your own test**<br>Any statistic against random placement with `nulls.placebo` |

Every statistic reproduces the numbers of the analyses it came from (AIR-SPACE and Stereo-XCR-seq LUAD, public; StereoIR, BGI data not yet public): of 4,001 recorded values, 3,302 are equal, 310 agree within Monte-Carlo tolerance, 17 have no archived value, 7 differ for documented reasons and 365 are new or informational ([validation](https://github.com/Reynold4k/spaclone/blob/main/docs/validation.md)).

## Integrates

| Package | What spaclone does with it, spatially | Install |
|---|---|---|
| [scanpy](https://scanpy.readthedocs.io) · [anndata](https://anndata.readthedocs.io) | Transcriptome workflow on spots: QC, clusters, regions (germinal centres, ASC foci, T zones), DE with regions as replicates; one container for spots and receptor records | core |
| [squidpy](https://squidpy.readthedocs.io) | Neighbourhood enrichment, co-occurrence and spatial autocorrelation of receptor-derived labels (`ext.squidpy`) | `[squidpy]` |
| [scirpy](https://scirpy.scverse.org) | Clonotypes, expansion, diversity, overlap and V(D)J usage per spatial unit, mapped back to tissue (`ext.scirpy`) | `[scirpy]` |
| [SpatialData](https://spatialdata.scverse.org) | SpatialData in and out: records as points, spots as shapes (`io.to_spatialdata`) | `[spatialdata]` |
| [Dandelion](https://sc-dandelion.readthedocs.io) | Clone calls written back to records and loci, and compared with spaclone's (`ext.dandelion`) | `[dandelion]` |
| [Immcantation](https://immcantation.readthedocs.io): IgBLAST · Change-O · SHazaM · dowser · IgPhyML | Germlines, mutation counts and lineage trees that keep their tissue positions (`ext.immcantation`) | `[immcantation]` + IgBLAST, R |
| [sciCSR](https://github.com/Fraternalilab/sciCSR) | CSR potential, SHM score, transition models and transition-path fluxes on tissue (`bcr.csr_potential`, `tl.tpt_flux`, `pl.csr_flow`), and a bridge to the R package (`ext.scicsr`; 103 of 103 parity checks match sciCSR 0.3.5) | `[scicsr]` |
| [CellRank](https://cellrank.readthedocs.io) · [deeptime](https://deeptime-ml.github.io) | Transition kernels on spatial neighbour graphs (`tl.transition_model`) | `[scicsr]` |
| [Threadfin](https://github.com/Reynold4k/Threadfin) | Clone-state landscapes on spots and loci, with calibrated spatial readouts (`ext.threadfin`) | `[threadfin]` |
| [OLGA](https://github.com/statbiophys/OLGA) | Generation probabilities of public TCR clonotypes (`tl.public`) | `[tcr]` |

`spc.check_install()` shows which of these work on your machine and the command that enables each missing one.

## Tutorials

| | |
|---|---|
| [Quick start](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/01_quickstart.ipynb) | One call on data with and without real positions |
| [Is my data spatial?](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/02_is_my_data_spatial.ipynb) | AIR-SPACE against Stereo-XCR-seq LUAD; a published QC metric under placebo |
| [Class switching and mutation](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/03_bcr_class_switching_and_mutation.ipynb) | CSR maps and hotspots, SHM, lineages in germinal centres |
| [Transcriptome and repertoire](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/04_transcriptome_and_repertoire.ipynb) | GEX workflow, regions, DE, diversity, networks, niches |
| [Your own placebo](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/05_custom_placebo.ipynb) | Test any statistic against random placement |
| [Hotspots and lineages](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/06_hotspots_and_lineages.ipynb) | The flagship on a Stereo-seq spleen: trust report, CSR/SHM hotspot atlas, lineage trees, sciCSR flow, Threadfin states |
| [AIR-SPACE time course](https://github.com/Reynold4k/spaclone/blob/main/docs/tutorials/07_airspace_time_course.ipynb) | Every analysis on all six public AIR-SPACE lymph nodes, Mock to day 21, with the package defaults and one table of the key numbers |

Reading data, concepts and the full API: [docs](https://github.com/Reynold4k/spaclone/tree/main/docs).

## How to cite

Please cite spaclone as in [CITATION.cff](https://github.com/Reynold4k/spaclone/blob/main/CITATION.cff) (GitHub: "Cite this repository") and also cite the packages you use through it; [docs/citing.md](https://github.com/Reynold4k/spaclone/blob/main/docs/citing.md) lists which function uses which package.
spaclone is MIT-licensed; optional packages and programs keep their own licences ([install](https://github.com/Reynold4k/spaclone/blob/main/docs/install.md#licences)).

## Credits

Chen Satoshi (@Reynold4k) and Claude Opus 5.5 (Anthropic)

## Licence

MIT, see [LICENSE](https://github.com/Reynold4k/spaclone/blob/main/LICENSE).
