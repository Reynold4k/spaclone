# Format fixtures for the io tests

Small excerpts of public example files, or synthetic files, in the same layout as the original output of each tool
(same header lines, delimiter, compression, HDF5 groups / datasets / attributes and dtypes). The excerpts were built
by `fixes/IO/build_fixtures.py` of the development tree. "Synthetic" marks files whose values are made up (plausible,
not real): either no public example exists, or the example could not be redistributed under spaclone's terms; those
were rebuilt by `fixes/AUDIT/scripts/synth_fixtures.py` (licence audit, 2026-10-31). Total size about 2.1 MB.

| folder / file | source | written by (version) | licence | how cut |
|---|---|---|---|---|
| `10x/cr2.2.0_*`, `cr3.0.0_*`, `cr3.1.0_*`, `cr4.0.0_*`, `cr5.0.0_*`, `cr6.1.0_*`, `cr9.0.0_*` | 10x Genomics public datasets `https://cf.10xgenomics.com/samples/cell-vdj/<version>/<sample>/<sample>_<file>`: vdj_v1_hs_pbmc_t (2.2.0), vdj_v1_hs_pbmc2_t (3.0.0), vdj_v1_hs_pbmc3_b (3.1.0), sc5p_v1p1_hs_PBMC_1k_b (4.0.0), sc5p_v2_hs_PBMC_10k_b (5.0.0), 10k_PBMC_5pv2_nextgem_Chromium_Controller (6.1.0), 5k_Mouse_Splenocytes_5p_gem-x (9.0.0) | Cell Ranger of that version | CC BY 4.0 (10x Genomics) | first barcodes; `cr5.0.0`: the same 20 cell barcodes (all their contigs) in filtered CSV, all CSV, all JSON and airr_rearrangement.tsv, plus 3 non-cell barcodes in the all files; clonotypes / consensus rows of those clonotypes |
| `airr/airr_standards_*` | github.com/airr-community/airr-standards `lang/python/tests/data` (commit 6f0ac9b, 2026-09-17) | AIRR Community test files | CC BY 4.0 (AIRR Community) | copied |
| `airr/scirpy_rearrangement_ig.tsv` | github.com/scverse/scirpy `src/scirpy/tests/data/airr` (commit a82b793) | scirpy test data | BSD-3-Clause, [LICENSES/scirpy-BSD-3-Clause.txt](LICENSES/scirpy-BSD-3-Clause.txt) | copied |
| `trust4/TRUST_example_report.tsv`, `TRUST_example_airr.tsv` | github.com/liulab-dfci/TRUST4 `example/` (commit a3fedd4, 2026-09-25) | TRUST4 example run | MIT, [LICENSES/TRUST4-MIT.txt](LICENSES/TRUST4-MIT.txt) | first 40 rows |
| `trust4/TRUST_barcode_report.tsv` | synthetic, README "barcode report" format (`#barcode cell_type chain1 chain2 secondary_chain1 secondary_chain2`, chain = `V,D,J,C,cdr3_nt,cdr3_aa,read_cnt,consensus_id,CDR3_germline_similarity,consensus_full_length`) | — | — | 3 barcodes |
| `mixcr/mixcr3_P15-T0-TIGIT.TRA.tsv` | synthetic: the column header of a MiXCR 3 `exportClones` table (cloneCount, uniqueTagCountUMI, no cell tag); the file name is kept for the tests | — | — | 30 TRA clones |
| `mixcr/mixcr4_sc_MIXCR_contigs.csv` | synthetic: the column header of a MiXCR 4 single-cell `exportClones` CSV (tagValueCELL, readCount, uniqueMoleculeCount, topChains) | — | — | 40 TRA / TRB clones on 20 cell barcodes |
| `mixcr/spatialTCR_clones.clonotypes.ALL.txt` | synthetic: the column header of a MiXCR `exportClones` `clones.clonotypes.ALL.txt` table | — | — | 30 IGK clones |
| `mixcr/mixcr2_human_readable.txt` | synthetic, MiXCR 2 human-readable header (immunarch parser) | — | — | 2 rows |
| `changeo/bracer_changeodb.tab` | scirpy `tests/data/bracer/changeodb.tab` | BraCeR (Change-O legacy uppercase) | BSD-3-Clause (scirpy), [LICENSES/scirpy-BSD-3-Clause.txt](LICENSES/scirpy-BSD-3-Clause.txt) | copied |
| `dandelion/vdj_processed_tiny.h5ddl` | synthetic: the h5py layout of a Dandelion `.h5ddl` (compound datasets `data` and `metadata`, `metadata_names`; Dandelion's AIRR + metadata columns and dtypes) | — | — | 60 contigs of 31 cells |
| `scirpy/wu2020_40_v0_6.h5ad`, `wu2020_40_v0_11.h5ad`, `wu2020_25.h5mu` | scirpy `tests/data/wu2020_200_v0_6.h5ad`, `_v0_11.h5ad`, `wu2020_200.h5mu` (from Wu et al. 2020, GEO GSE139555) | scirpy (old IR_* schema 0.6 / 0.11; awkward obsm["airr"]) | BSD-3-Clause (scirpy), [LICENSES/scirpy-BSD-3-Clause.txt](LICENSES/scirpy-BSD-3-Clause.txt) | first 40 (25) cells, first 10 genes, uns / obsm dropped |
| `immunoseq/immunoseq_1..4.txt` | immunarch `tests/test_parsing_data` (commit 274afb2) | Adaptive immunoSEQ export v2 / v3 | Apache-2.0, [LICENSES/immunarch-Apache-2.0.txt](LICENSES/immunarch-Apache-2.0.txt) | first 30 rows (2, 3: 19 selected rows) |
| `bd/test_per_cell_chain.csv`, `test_unfiltered_contigs.csv` | scirpy `tests/data/bd` | BD Rhapsody pipeline 1.9 / 1.10 | BSD-3-Clause (scirpy), [LICENSES/scirpy-BSD-3-Clause.txt](LICENSES/scirpy-BSD-3-Clause.txt) | copied |
| `sptcr/GSM7658227_SPTCR14_IGB_TRABDG.csv.gz` | GEO GSM7658227 (GSE238071) supplementary file; Benotmane et al. 2023, doi:10.1038/s41467-023-43201-6 | SPTCR-seq pipeline | GEO public data | first 60 rows |
| `clone_tables/immunarch_Sample1.tsv.gz` | immunarch `inst/extdata/io/Sample1.tsv.gz` (commit 274afb2) | immunarch 0.5.6 export | Apache-2.0, [LICENSES/immunarch-Apache-2.0.txt](LICENSES/immunarch-Apache-2.0.txt) | first 30 rows |
| `clone_tables/vdjtools.txt`, `migec_cdrblast.txt`, `parse_barcode_report.tsv`, `wat3r_barcode_results.csv` | synthetic, documented headers (VDJtools docs; MiGEC CdrBlast docs; Parse and WAT3R columns as parsed by scRepertoire `loadContigs.R`) | — | — | 1–2 rows |
| `visium_sd_sr1/` | 10x `spatial-exp/1.0.0/V1_Human_Lymph_Node` (filtered_feature_bc_matrix.h5, spatial.tar.gz) | Space Ranger 1.x (headerless `tissue_positions_list.csv`) | CC BY 4.0 (10x Genomics) | 120 tissue spots of a 10 × 24 array window + 5 off-tissue positions, 40 genes; scale factors copied |
| `visium_sd_sr2/` | 10x `spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Tonsil` | Space Ranger 2.1 (`tissue_positions.csv` with header; Gene Expression + Antibody Capture features) | CC BY 4.0 (10x Genomics) | 114 spots, 40 genes + 2 antibody features |
| `visium_hd_square_008um/` | Space Ranger 3.1.2 output of the SpatialBench Visium HD mouse section KO167 (BioStudies S-BSST2361), `binned_outputs/square_008um` | Space Ranger 3.1.2 | CC0 (BioStudies S-BSST2361) | 144 bins of a 12 × 12 window, 40 genes; parquet positions subset; scale factors copied |
| `xenium/` | 10x `xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1` (cells.parquet, cell_feature_matrix.h5, experiment.xenium) | Xenium Onboard Analysis 1.0.1 | CC BY 4.0 (10x Genomics) | 100 cells, 30 genes + 2 control features |
| `stereo/chipA_window.raw.gef` | synthetic: the layout of a SAW square-bin GEF v4 (`geneExp/bin1` expression / exon / gene, root and dataset attributes; neutral chip id `CHIPA000`) | — | — | 80 × 80 DNB window, 200 genes, random counts |
| `stereo/chipA_subset.cellbin.gef` | synthetic: the layout of a SAW cell-bin GEF (`cellBin` cell / cellExp / cellExpExon / gene / geneExp / cellBorder / cellExon) | — | — | 60 cells with octagonal borders |
| `stereo/chipA_window.gem.gz` | the synthetic GEF window above written as SAW GEM v0.1 (documented `#` header and `geneID x y MIDCount ExonCount`) | — | — | same rows as the GEF window |
| `stereo/chipA_window_cellbin_synthetic.gem.gz` | as above with a synthetic `CellID` (10 × 10 DNB blocks) | — | — | — |
| `stereo/chipA_bin20_stereopy_subset.h5ad` | synthetic: the layout of a stereopy-written bin20 AnnData (packed integer `obs_names`, `obs` x / y / total_counts / leiden, `uns` bin_type / bin_size / resolution / sn / omics) | — | — | 120 bins, 40 genes, random counts |
| `slideseq/`, `seurat/object.rds` | synthetic (Slide-seq `BeadLocationsForR.csv` + `MappedDGEForR.csv` layout; RDS magic bytes only) | — | — | — |

Gene names and identifiers in the synthetic files are public Ensembl identifiers; no other value comes from a real data
set.

## Attribution and licences

The excerpts are redistributed under the terms of their sources; each was shortened as stated in the table ("how
cut") and is otherwise unchanged.

- **10x Genomics datasets** (`10x/`, `visium_sd_sr1/`, `visium_sd_sr2/`, `xenium/`): © 10x Genomics, licensed under the
  Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/);
  dataset pages at https://www.10xgenomics.com/datasets.
- **AIRR Community standards test files** (`airr/airr_standards_*`): © AIRR Community, CC BY 4.0
  (https://github.com/airr-community/airr-standards).
- **scirpy test data** (`airr/scirpy_rearrangement_ig.tsv`, `changeo/`, `bd/`, `scirpy/`): BSD 3-Clause licence of
  scirpy, copyright notices in [LICENSES/scirpy-BSD-3-Clause.txt](LICENSES/scirpy-BSD-3-Clause.txt).
- **TRUST4 example output** (`trust4/TRUST_example_*`): MIT licence, Copyright (c) 2018 Li Song, X. Shirley Liu,
  [LICENSES/TRUST4-MIT.txt](LICENSES/TRUST4-MIT.txt).
- **immunarch test and example data** (`immunoseq/`, `clone_tables/immunarch_Sample1.tsv.gz`): Apache License 2.0,
  [LICENSES/immunarch-Apache-2.0.txt](LICENSES/immunarch-Apache-2.0.txt); the files were shortened (first rows only).
- **SPTCR-seq** (`sptcr/`): public GEO data (GSE238071; Benotmane et al. 2023, Nat Commun,
  doi:10.1038/s41467-023-43201-6).
- **SpatialBench Visium HD** (`visium_hd_square_008um/`): BioStudies S-BSST2361, released under CC0.
