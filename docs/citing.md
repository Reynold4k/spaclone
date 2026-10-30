
# Citing

## spaclone

spaclone has no paper. Please cite the version you used:

> Satoshi, C. spaclone (version 1.0.0). https://github.com/Reynold4k/spaclone

```bibtex
@software{spaclone,
  author  = {Satoshi, Chen},
  title   = {spaclone},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/Reynold4k/spaclone}
}
```

GitHub's "Cite this repository" button gives the same entry. It reads
[CITATION.cff](https://github.com/Reynold4k/spaclone/blob/main/CITATION.cff), which also lists every reference below.

## Please also cite the packages you use through spaclone

spaclone runs these packages unchanged, or re-implements their published method (sciCSR). When a result comes from one
of them, cite it as its authors ask.

| spaclone function or extra | Package | Please cite |
|---|---|---|
| {mod}`spaclone.gex`, AnnData input and output (core) | scanpy, anndata | [Wolf et al. 2018](https://doi.org/10.1186/s13059-017-1382-0); [Virshup et al. 2024](https://doi.org/10.21105/joss.04371); scverse: [Virshup et al. 2023](https://doi.org/10.1038/s41587-023-01733-8) |
| {func}`~spaclone.gex.cluster`, {func}`~spaclone.gex.domains` (core; `[leiden]`) | python-igraph; leidenalg | Csárdi & Nepusz 2006, *InterJournal Complex Systems* 1695; [Traag et al. 2019](https://doi.org/10.1038/s41598-019-41695-z) |
| {mod}`spaclone.ext.scirpy`, {func}`~spaclone.io.to_scirpy`, {func}`~spaclone.io.from_scirpy`, `backend="scirpy"` of {func}`~spaclone.tl.clonotype_network` (`[scirpy]`) | scirpy; mudata | [Sturm et al. 2020](https://doi.org/10.1093/bioinformatics/btaa611); for `.h5mu` files [Bredikhin et al. 2022](https://doi.org/10.1186/s13059-021-02577-8) |
| {mod}`spaclone.ext.squidpy`, {func}`~spaclone.gex.spatially_variable_genes` (`[squidpy]`) | squidpy | [Palla et al. 2022](https://doi.org/10.1038/s41592-021-01358-2) |
| {func}`~spaclone.io.to_spatialdata` (`[spatialdata]`) | SpatialData | [Marconato et al. 2025](https://doi.org/10.1038/s41592-024-02212-x) |
| {mod}`spaclone.ext.dandelion` (`[dandelion]`) | Dandelion | [Suo et al. 2024](https://doi.org/10.1038/s41587-023-01734-7) |
| {mod}`spaclone.ext.immcantation`: `makedb`, `define_clones`, `create_germlines` (`[immcantation]`) | Change-O | [Gupta et al. 2015](https://doi.org/10.1093/bioinformatics/btv359) |
| `observed_mutations`, `shazam_threshold` | SHazaM | [Gupta et al. 2015](https://doi.org/10.1093/bioinformatics/btv359); threshold: [Nouri & Kleinstein 2018](https://doi.org/10.3389/fimmu.2018.01687) |
| `dowser_trees` | dowser, alakazam | [Hoehn et al. 2022](https://doi.org/10.1371/journal.pcbi.1009885); alakazam: [Gupta et al. 2015](https://doi.org/10.1093/bioinformatics/btv359) |
| `dowser_trees(build="igphyml")` | IgPhyML | [Hoehn et al. 2019](https://doi.org/10.1073/pnas.1906020116); [Hoehn et al. 2017](https://doi.org/10.1534/genetics.116.196303) |
| {func}`~spaclone.bcr.run_igblast`, `igblast` | IgBLAST | [Ye et al. 2013](https://doi.org/10.1093/nar/gkt382) |
| TIgGER and SCOPer (installed by `scripts/install_spaclone_env.sh`) | TIgGER; SCOPer | [Gadala-Maria et al. 2015](https://doi.org/10.1073/pnas.1417683112); [Nouri & Kleinstein 2018](https://doi.org/10.1093/bioinformatics/bty235) |
| {func}`~spaclone.bcr.csr_potential`, {func}`~spaclone.bcr.shm_score`, {func}`~spaclone.tl.transition_model`, {func}`~spaclone.tl.tpt_flux`, {mod}`spaclone.ext.scicsr` | sciCSR (method re-implemented; the original through `ext.scicsr`) | [Ng et al. 2024](https://doi.org/10.1038/s41592-023-02060-1) |
| `backend="cellrank"` of {func}`~spaclone.tl.transition_model` (`[scicsr]`) | CellRank | [Lange et al. 2022](https://doi.org/10.1038/s41592-021-01346-6); [Weiler et al. 2024](https://doi.org/10.1038/s41592-024-02303-9) |
| `backend="deeptime"` of {func}`~spaclone.tl.tpt_flux` (`[scicsr]`) | deeptime | [Hoffmann et al. 2022](https://doi.org/10.1088/2632-2153/ac3de0) |
| BAM input of {func}`~spaclone.bcr.ighc_transcripts` (`[scicsr]`) | pysam | the [pysam](https://github.com/pysam-developers/pysam) software |
| {func}`~spaclone.tl.pgen`, `pgen=True` of {func}`~spaclone.tl.public_clonotypes` (`[tcr]`) | OLGA | [Sethna et al. 2019](https://doi.org/10.1093/bioinformatics/btz035) |
| {mod}`spaclone.ext.threadfin` (`[threadfin]`) | Threadfin | Satoshi, C. Threadfin (version 4.6.0), https://github.com/Reynold4k/Threadfin |
| {mod}`spaclone.datasets` (`[datasets]`) | Pooch; the AIR-SPACE data | [Uieda et al. 2020](https://doi.org/10.21105/joss.01943); data: [Jiang et al. 2026](https://doi.org/10.1073/pnas.2504742123) (GEO GSE286452) |
| {func}`~spaclone.io.read_tissue_mask` (`[images]`) | tifffile | [Gohlke, tifffile](https://doi.org/10.5281/zenodo.6795860) |

NumPy, SciPy, pandas, Matplotlib, scikit-learn, h5py and pyarrow are used throughout; their references are in
CITATION.cff.

## Data from other programs

- The receptor tables that {mod}`spaclone.io` reads come from other programs. Cite the program that produced your
  data, for example TRUST4 ([Song et al. 2021](https://doi.org/10.1038/s41592-021-01142-2)), MiXCR
  ([Bolotin et al. 2015](https://doi.org/10.1038/nmeth.3364)) or 10x Genomics Cell Ranger.
- IgBLAST databases built from IMGT germlines (`--references` of the install script): cite IMGT
  ([Lefranc et al. 2015](https://doi.org/10.1093/nar/gku1056)). IMGT data are distributed under IMGT's own terms
  (CC BY-NC-ND 4.0), so spaclone does not ship them.
