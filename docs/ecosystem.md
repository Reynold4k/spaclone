---
orphan: true
---

# Working with established packages

spaclone does not re-implement the single-cell repertoire and transcriptome tools. It runs them, unchanged, on
spatial units and brings their results back onto tissue coordinates, where spaclone's own maps, scans and calibrated
tests take over. Each bridge lives in {mod}`spaclone.ext`, imports its package only when called, and names the extra
that installs it when the package is missing ([](install.md)).

| Package | What spaclone runs | Spatial counterpart | Module | Extra |
|---|---|---|---|---|
| scirpy | `pp.ir_dist`, `tl.define_clonotype_clusters`, `tl.clonal_expansion`, `tl.alpha_diversity`, `tl.repertoire_overlap`, `tl.clonotype_modularity`, `get.airr` + `tl.group_abundance` | units are inferred cells (loci), spots or segmented cells; labels on every unit and record with positions; per-region tables; modularity on the spatial neighbour graph | {mod}`spaclone.ext.scirpy` | `scirpy` |
| squidpy | `gr.spatial_neighbors_knn` / `_radius` (`gr.spatial_neighbors` before 1.8), `gr.nhood_enrichment`, `gr.co_occurrence`, `gr.spatial_autocorr` | receptor-derived labels (isotype, switched, mutated, expanded, clone) on the GEX spots | {mod}`spaclone.ext.squidpy` | `squidpy` |
| Dandelion | `tl.find_clones` | clone of every spatial record, reproducible ids | {mod}`spaclone.ext.dandelion` | `dandelion` |
| IgBLAST, Change-O, SHazaM, dowser | `igblastn`, `MakeDb.py`, `DefineClones.py`, `CreateGermlines.py`, `distToNearest` / `findThreshold`, `observedMutations`, `formatClones` / `getTrees` | alignments, clones, germlines, mutation frequencies and trees with tip positions | {mod}`spaclone.ext.immcantation` | `immcantation` (+ IgBLAST and R) |
| sciCSR | CSR potential, SHM score, transition models and fluxes; the original R package for parity | the same scores on cells, spots, beads or loci; transitions on spatial graphs | {mod}`spaclone.ext.scicsr`, {mod}`spaclone.bcr.csr_dynamics`, {mod}`spaclone.tl.transitions` | `scicsr` |
| Threadfin | clone-state axes and programmes | clone axes and programmes positioned on tissue | {mod}`spaclone.ext.threadfin` | `threadfin` |
| scanpy, anndata | the transcriptome workflow ({mod}`spaclone.gex`) | GEX regions, domains, pseudobulk DE with regions as replicates | core | |
| SpatialData | export | records as points, spots as shapes, clone tables | {func}`~spaclone.io.to_spatialdata` | `spatialdata` |

## scirpy

{func}`spaclone.ext.scirpy.prepare` builds scirpy's AnnData from a receptor table with {func}`spaclone.io.to_scirpy`
and runs `pp.index_chains` and `tl.chain_qc`. A spatial unit often holds several identical chains (molecules or
spots of one cell); they count once in scirpy's receptor model (`collapse=True`), otherwise most loci would be
"multichain" and scirpy's default grouping would keep them apart. On the StereoIR spleen, collapsing leaves one
multichain locus out of 31,516.

B-cell clonotypes and clones are heavy-chain ones by default. `receptor_arms="auto"` runs `"VDJ"` when the units hold
B-cell chains and scirpy's `"all"` otherwise, in {func}`~spaclone.ext.scirpy.clonotypes` and in every function that
counts clones ({func}`~spaclone.ext.scirpy.clonal_expansion`, {func}`~spaclone.ext.scirpy.expansion_map`,
{func}`~spaclone.ext.scirpy.diversity_by_region`, {func}`~spaclone.ext.scirpy.repertoire_overlap`,
{func}`~spaclone.ext.scirpy.clonotype_modularity`); units without a heavy chain get no clonotype. Light-chain
junctions (IGK, IGL) are shared by unrelated B cells: on StereoIR, `receptor_arms="all"` puts 32,379 loci (29% of the
112,451 loci; all IGK, from 2,250 exact clones spread over the whole section) into one clonotype, and `expansion_map`
then calls 79% of the loci "expanded". A clonotype or clone that holds more than 5% of the units (and at least 50 of
them) raises {class}`~spaclone.ext.scirpy.LargeClonotypeWarning`, which names the argument to change.

```python
import spaclone
from spaclone.ext import scirpy as sx

sa = spaclone.read("mm_spleen_bin20_anno.h5ad", "mm_spleen_bcr_airr.tsv.tsv")  # the StereoIR spleen
res = sx.clonotypes(sa.air, metric="hamming", cutoff=1)  # heavy-chain loci -> clonotype clusters
res.summary  # 37,291 heavy-chain loci in 13,586 clonotypes; the largest holds 1,252 (3.4%)
res.air["scirpy_clone_id"]  # on every record, with x_um / y_um (light-chain records: none)
res.plot(highlight=res.table.clonotype.head(5))  # the five largest clonotypes on the tissue
wf = spaclone.bcr.hotspot_workflow(sa)  # CSR / SHM hotspot regions: region_id on every record (0 = none)
sx.diversity_by_region(wf.records, groupby="region_id").table  # scirpy's alpha diversity per region
sx.expansion_map(sa.air).plot()  # where expanded clones sit: 17,263 of the 37,291 heavy-chain loci (46%)
```

Every number is scirpy's: on StereoIR the bridge and plain scirpy calls give identical clonotype labels, expansion
classes, diversities, overlaps and modularity scores (validation `s2_scirpy_stereoir`, on the IGH contigs). scirpy's
expansion classes equal {func}`spaclone.tl.clone_sizes` and its normalised Shannon entropy equals the `evenness` of
{func}`spaclone.tl.diversity` on the same units. When no unit has a chain of one arm, `"all"` and `"any"` are run as
the arm that exists (scirpy 0.25 fails otherwise; the result is the same). Maps follow the convention of
{mod}`spaclone.pl`: the section rotated so that its long axis is horizontal (`rotate=True`) and y pointing down
(`invert_y=True`). These are descriptions: compare regions with {func}`spaclone.tl.diversity` (rarefied, against
random placement of intact clones), and test concentration with {func}`spaclone.tl.hotspots`.

## squidpy

{func}`spaclone.ext.squidpy.receptor_labels` puts the receptor records onto the GEX spots (by spot id, or nearest
spot), then {func}`~spaclone.ext.squidpy.nhood_enrichment`, {func}`~spaclone.ext.squidpy.co_occurrence` and
{func}`~spaclone.ext.squidpy.spatial_autocorr` run squidpy on those labels with squidpy's own graph and return tidy
tables. The numbers equal the plain squidpy calls (validation `s2_squidpy_stereoir`). The `expanded` and `clone`
labels count heavy-chain clones (`chains=("IGH", "TRB", "TRD")`; `None` counts every locus). squidpy's neighbourhood
enrichment gives z-scores against its permutations but keeps no permutation distribution, so
{func}`~spaclone.ext.squidpy.nhood_enrichment` reports `p_norm`, the two-sided normal approximation of z, and `q_norm`
(Benjamini-Hochberg), not a `p_value`.

Every call warns ({class}`~spaclone.ext.squidpy.DepthNullWarning`): squidpy's nulls relabel spots at random, so any
receptor label that follows B or T cells looks enriched, also when receptor positions are random. Whether positions
carry clonal information is the question of {func}`spaclone.qc.colocalisation`, which shuffles clone labels within
depth strata.

## Dandelion

{func}`spaclone.ext.dandelion.find_clones` runs `dandelion.tl.find_clones` on the records and writes each record's
clone back. Two things make Dandelion's clone calls reproducible here:

- **Version.** sc-dandelion 1.0 groups junctions by connected components within V gene, J gene and junction length;
  up to 0.5 a junction joined only its direct neighbours and a record took the last of its overlapping
  neighbourhoods, so a chain A ~ B ~ C (A and C too far apart) was split. 1.0 therefore calls fewer, larger clones.
  The version is read from the package metadata (`dandelion.__version__` reports `0.1.dev9` for both) and recorded
  in `attrs["spaclone"]["dandelion"]`; 0.5 raises {class}`~spaclone.ext.dandelion.DandelionVersionWarning`.
- **Hash seed.** Clone numbering follows Python's set order, which changes between sessions; Dandelion runs in a
  child process with `PYTHONHASHSEED=0`, so the same table always gives the same ids.

With records as units (the default: spots and beads are not cells), sc-dandelion 1.0 gives exactly the partition of
{func}`spaclone.pp.define_clones` with `method="vj_length_identity"` and of the archive's Dandelion rule, on StereoIR
and AIR-SPACE (validation `s2_dandelion_stereoir`). {func}`~spaclone.ext.dandelion.compare_clones` reports the
agreement of any two clone calls (adjusted Rand index, split and merged clones).

## Immcantation

{mod}`spaclone.ext.immcantation` exports the record sequences with ids that point back to their rows, runs IgBLAST,
Change-O and the R packages with the documented commands, and imports the results onto the records:

```python
import spaclone
from spaclone.ext import immcantation as im

sa = spaclone.read("mm_spleen_bin20_anno.h5ad", "mm_spleen_bcr_airr.tsv.tsv")  # the StereoIR spleen
air = sa.air[sa.air.locus == "IGH"]  # heavy-chain contigs with their sequences and positions
res = im.run(air, "immcantation_out", threads=8)  # IgBLAST, MakeDb, SHazaM, DefineClones, germlines, mutations
res.air[["x_um", "y_um", "imm_v_call", "imm_clone_id", "imm_mu_freq"]]
trees = im.dowser_trees(res.db, nproc=8)  # dowser trees as spaclone LineageTrees
tips = spaclone.bcr.tip_table(trees, res.db, id_key="sequence_id")  # one row per sequence, with its position
res.commands  # what was run, for the methods
```

The programs are found on `PATH` or through `SPACLONE_IGBLAST`, `SPACLONE_IGDATA`, `SPACLONE_GERMLINES`,
`SPACLONE_CHANGEO` and `SPACLONE_RSCRIPT` ({func}`~spaclone.ext.immcantation.find_tools`, which also takes the paths).
On StereoIR (43,517 IGH contigs) the block gives 13,394 clones among the 43,250 annotated sequences, V-region
mutation frequencies for 39,063 of them (the other 4,187 have no clonal germline of their alignment's length and
are left missing) and 1,214 dowser trees with 24,082 tips, in about 30 minutes on 8 CPUs.

Each step is also a function ({func}`~spaclone.ext.immcantation.igblast`, {func}`~spaclone.ext.immcantation.makedb`,
{func}`~spaclone.ext.immcantation.define_clones`, {func}`~spaclone.ext.immcantation.create_germlines`,
{func}`~spaclone.ext.immcantation.shazam_threshold`, {func}`~spaclone.ext.immcantation.observed_mutations`), takes
`dry_run=True` to return its command, and finds the programs through {func}`~spaclone.ext.immcantation.find_tools`.
With the archive's programs and references the IgBLAST + MakeDb table of the 37,690 StereoIR IGH contigs is
reproduced contig by contig, and dowser gives trees for the same 71 clones with the same tips (validation
`s2_immcantation_stereoir`).

## sciCSR, CellRank and deeptime

The sciCSR scores and transition models are computed natively by {mod}`spaclone.bcr.csr_dynamics` and
{mod}`spaclone.tl.transitions`; `backend="cellrank"` and `backend="deeptime"` reproduce the kernels and fluxes with an
installed CellRank (2.x) and deeptime (`spaclone[scicsr]`). The original sciCSR package needs Python 3.8 with
CellRank 1.5, which cannot share spaclone's Python; {mod}`spaclone.ext.scicsr` reaches it through R in its own
environment (`bash scripts/install_spaclone_env.sh --with-scicsr`).

## Threadfin

{mod}`spaclone.ext.threadfin` runs Threadfin (`spaclone[threadfin]`) and positions its clone axes and programmes on
the tissue. {func}`~spaclone.ext.threadfin.run` builds the spatial context (T zone, follicle and red pulp, and the
signed distances to them) itself when that is certain: exactly one standard cell-type label column in `gex.obs`
whose classes name T, B and erythroid cells (on StereoIR `cell2loc_anno`: T_cells, B_cells, Erythrocyte). Otherwise
`result.context` is None, `result.units` has no `compartment`, and `result.context_note` says why; pass
`context=spatial_context(gex, cell_type_key=...)` to choose. The maps of a result are drawn by
{func}`spaclone.pl.clone_state_map` and {func}`spaclone.pl.clone_axis_profile`.

## Versions tested

See [the tested installations](install.md#tested-installations): Python 3.11, 3.12 and 3.13 with and without
`[full]`, the oldest supported core versions, and the conda environment.
