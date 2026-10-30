# Reading your data

spaclone reads the output of the mainstream receptor and spatial-transcriptomics tools directly. One call reads a
spatial expression source and a receptor table of the same section:

```python
import spaclone

data = spaclone.read("outs/", "filtered_contig_annotations.csv")   # Space Ranger folder + 10x contigs
# StereoIR demo files: Stereo-seq bin20 + TRUST4 AIRR (the AIRR file name ends in .tsv.tsv as distributed)
data = spaclone.read("mm_spleen_bin20_anno.h5ad", "mm_spleen_bcr_airr.tsv.tsv")
# AIR-SPACE D14 (GEO GSE286452, spaclone.datasets.airspace("D14")): bead h5ad + clone table
data = spaclone.read("GSM8727833_adata_D14_with_image_raw_counts.h5ad", "GSM8727839_D14_ovp_clone_table.txt.gz")
print(data)              # one line: records per locus, library, GEX shape, unit, reader
print(data.meta["io"])   # every decision and its evidence
```

Not sure what a file is, or what spaclone still needs? Ask first:

```python
print(spaclone.io.sniff("TRUST_report.tsv"))
```

```bash
spaclone sniff TRUST_report.tsv outs/binned_outputs/square_008um
spaclone sniff GSM8727839_D14_ovp_clone_table.txt.gz GSM8727833_adata_D14_with_image_raw_counts.h5ad
```

Files given together are judged as `spaclone.read` reads them: an AIR-SPACE bead h5ad alone needs `unit_um=`, but
with its clone table the bead unit (0.65 µm) is known.

## The rule: detect what is certain, ask for the rest

spaclone identifies a file only from documented signatures (column names, file layout, metadata written by the tool).
When the evidence is not certain, it stops and tells you which argument to pass. It never guesses silently.

| Decision | Automatic when | Otherwise you pass |
|---|---|---|
| Receptor format | exactly one tool signature matches the header | `format=` (the error lists the candidates) |
| Column mapping | the tool's documented column names | `columns={"junction_aa": "CDR3.aa", ...}` (always wins) |
| Count meaning | the tool documents it (reads, UMIs, templates, cells) | `count_type="reads"` / `"umis"` / `"templates"` / `"cells"` for a bare `count` column |
| Junction vs CDR3 | ≥ 80% of the amino-acid CDR3s carry C … W/F (junction); ≤ 20% (CDR3 without anchors, stored as `cdr3_aa`) | `junction="junction"` / `"cdr3"` for a mix |
| Locus | a locus / chain column, else the first letters of the V, J or C call | — |
| Productive | T/F, True/False, 1/0, yes/no, In/Out/Stop; anything else is unknown | — |
| Spot ↔ GEX barcodes | exact ids; else one documented normalisation that turns a low overlap into a high one (10x `-1` suffix, anndata / Seurat sample keys, Visium HD bin sizes, Stereo-seq `x-y` / `x_y` / packed ids) | rename the ids, or `spot_key=` |
| µm per coordinate unit | a technology fact: spaclone-written AnnData; Visium HD `microns_per_pixel`; Visium SD 100 µm spot array; Stereo-seq DNB 0.5 µm from GEF / GEM / stereopy metadata / coordinate ids; Xenium centroids in µm; AIR-SPACE beads with their clone table 0.65 µm. Every result is checked: units 0.2–300 µm apart and a section < 5 cm wide, else a warning names `unit_um` | `unit_um=` |
| Coordinates | `obsm["spatial"]`, else one unambiguous alternative | `spatial_key=` |
| Square-bin size | — | `bin_size=` for Stereo-seq `.gef` / `.gem` (e.g. 20) |
| Cell types (`spaclone.run`, `spaclone run`) | `obs["cell_type"]`, else exactly one standard label column such as `cell2loc_anno` (obs labels only; proportion tables are listed in `meta["io"]` but never used as labels); each receptor's own types from the B / T lineage vocabulary, whole words only (`B_cells`, `GC B`, `Plasma cells`; not `Plasmacytoid DC`) | `cell_type_key=` / `own_types=` (`--cell-type-key`, `--own-types`); otherwise the own-type share is skipped with the candidates listed |

## Your tool → your file → what spaclone detects → what you must give

### Receptor tables

| Your tool | Your file | spaclone detects | You must give |
|---|---|---|---|
| Any AIRR producer (IgBLAST `-outfmt 19`, Change-O / Immcantation, MiXCR `exportAirr`, Dandelion, scirpy `write_airr`, BD Rhapsody AIRR, StereoIR, LongAIRR) | AIRR Rearrangement TSV 1.x / 2.0 | format `airr`; `consensus_count` = reads, `duplicate_count` per AIRR schema, `umi_count` = UMIs | — |
| 10x Cell Ranger V(D)J 2.x – 9.x | `filtered_contig_annotations.csv`, `all_contig_annotations.csv` | `reads` = reads, `umis` = UMIs, `cdr3` = junction, `raw_clonotype_id` = clone | — |
| 10x Cell Ranger | `all_contig_annotations.json` | genes from the annotations, `read_count`, `umi_count` | — |
| 10x Cell Ranger 4+ | `airr_rearrangement.tsv` | writer: `consensus_count` = reads, `duplicate_count` = UMIs (checked against the CSV) | — |
| 10x Cell Ranger | `clonotypes.csv`, `consensus_annotations.csv` | recognised; no barcodes | use the contig file of the same run |
| TRUST4 | `*_airr.tsv`, `*_barcode_airr.tsv` | AIRR; `consensus_count` = reads | `count_type="umis"` if TRUST4 ran with `--UMI` |
| TRUST4 | `*_barcode_report.tsv` | chains split into records; `read_cnt` = reads | — |
| TRUST4 | `*_report.tsv` (bulk) | recognised | `count_type=` (reads, barcodes = cells or UMIs depend on the run); no barcodes |
| MiXCR 2 / 3 / 4 | `exportClones` (default or custom columns, human-readable MiXCR 2 headers) | `readCount` / `cloneCount` = reads, `uniqueMoleculeCount` = UMIs, `tagValueCELL` = barcode, hit scores stripped, productive from the CDR3 | — |
| MiXCR | `exportAlignments` (one read per row; `tagValueCELL`, `tagValueUMI`) | one record per read | collapse molecules with `spaclone.pp.collapse_umis` |
| Change-O < 1.0 / `--format changeo` | uppercase table (`SEQUENCE_ID`, `FUNCTIONAL`, `V_CALL`, `JUNCTION`, `CLONE`, `CONSCOUNT`, `DUPCOUNT`, `UMICOUNT`, `CELL`) | `FUNCTIONAL` = productive, `CONSCOUNT` = reads, `UMICOUNT` = UMIs, junction translated | — |
| Dandelion | `.h5ddl`, its TSV, a `Dandelion` object | AIRR table (h5py; no Dandelion install needed for current files) | — |
| scirpy ≥ 0.13 | AnnData / MuData with `obsm["airr"]` (also `.h5ad`, `.h5mu`) | chains as records; the GEX can be the same object | — |
| scirpy < 0.13 | AnnData with `IR_VJ_1_*` / `IR_VDJ_1_*` obs columns | v0.6 names mapped (`cdr3` → junction_aa, `expr` → UMIs) | — |
| BD Rhapsody | `*_perCellChain.csv`, `*_Dominant_Contigs.csv`, `*_Unfiltered_Contigs.csv` | `Chain_Type` → locus, `Read_Count` = reads, `Molecule_Count` = UMIs; `CDR3_Translation` lacks the anchors → `cdr3_aa` | — |
| Parse Evercode TCR / BCR | `barcode_report.tsv` | chains split; `*_read_count` = reads, `*_transcript_count` = UMIs | — |
| WAT3R | `barcode_results.csv` | TRA / TRB chains; reads and UMIs | — |
| Adaptive immunoSEQ v3 | `rearrangement, amino_acid, frame_type, templates, reads` | templates and reads; `frame_type` = productive; Adaptive gene names → IMGT (`TCRBV05-01` → `TRBV5-1`) | no barcodes (bulk) |
| Adaptive immunoSEQ v1 / v2 | `nucleotide, aminoAcid, count (templates/reads), ...` | `sequenceStatus` = productive | `count_type=` for `count (templates/reads)`; no barcodes |
| immunarch | `# Exported from immunarch` | recognised | `count_type=` for `Clones`; no barcodes |
| VDJtools | `count freq cdr3nt cdr3aa v d j ...` | recognised | `count_type=`; no barcodes |
| MiGEC / CdrBlast, MiTCR | clonotype tables | `Good events` = UMIs, `Good reads` = reads | no barcodes |
| SPTCR-seq (GEO GSE238071) | per-section CSV | Visium barcodes, `expression` = UMIs, CDR3 without anchors | — |
| Stereo-XCR-seq | MiXCR meta table / public `out.meta.gz` | recognised | read with `spaclone.io.read_stereo_xcr` (own DNB coordinates) |
| AIR-SPACE (GSE286452) | `*_ovp_clone_table.txt.gz` + bead h5ad | `spaclone.read(h5ad, table)` runs `read_airspace` | — |
| Stereo-XCR-seq LUAD (GSE328481) | cell h5ad with `IGHcdr3aa`, ... | `spaclone.read(h5ad)` runs `from_cell_calls` | — |
| Anything else | any delimited table | — | `format="custom", columns={"cell_id": ..., "junction_aa": ..., "v_call": ...}` |

Bulk tables (no cell or spot barcode) are read by `spaclone.io.read_receptors`, but cannot be placed on spots.

### Spatial expression

| Your tool | Your file | spaclone detects | You must give |
|---|---|---|---|
| Space Ranger (Visium SD) 1.x – 3.x | `outs/` folder (`filtered_feature_bc_matrix.h5`, `spatial/tissue_positions_list.csv` or `tissue_positions.csv`) | full-resolution pixels; µm per pixel measured on the 100 µm spot array | — |
| Space Ranger (Visium HD) | `binned_outputs/square_008um/` (parquet positions) | `microns_per_pixel` | pick one bin folder (an `outs/` with several bins raises) |
| scanpy / squidpy | Visium `.h5ad` (`uns["spatial"]` scale factors, `array_row` / `array_col`) | as Space Ranger | — |
| SAW / stereopy | `.h5ad` (`uns["bin_type"]`, `uns["resolution"]`) | DNB, resolution nm | — |
| SAW | `.gef` (square bins), `.cellbin.gef` | DNB (0.5 µm), stereopy packed ids | `bin_size=` for square bins |
| SAW | `.gem` / `.gem.gz` (with or without `CellID`) | DNB (0.5 µm) | `bin_size=` unless `CellID` |
| StereoIR / TRUST4 on Stereo-seq | `.h5ad` with `x-y` obs names on a bin lattice | DNB (0.5 µm) | — |
| Xenium | output folder (`cell_feature_matrix.h5`, `cells.parquet` or `cells.csv.gz`) | centroids in µm, controls dropped | — |
| SpatialData | `.zarr` or object with one table | the table | `unit_um=` unless a rule above applies |
| Seurat | `.rds` | recognised | convert to `.h5ad` (the error prints the snippet) |
| Slide-seq | `BeadLocationsForR.csv` + `MappedDGEForR.csv` | bead pixels | `unit_um=` (Slide-seqV2 papers: 0.65) |
| AIR-SPACE (GSE286452) | bead `.h5ad` read together with its clone table | bead coordinates, 0.65 µm per unit | — (`unit_um=` overrides) |
| Curio Seeker, Giotto- or Seurat-converted `.h5ad` | `.h5ad` | coordinates | `unit_um=` (no unit stored) |
| stereopy's own StereoExpData `.h5` | — | recognised | convert with `st.io.stereo_to_anndata` (snippet in the error) |

## Mistakes spaclone stops

| Mistake | What you see |
|---|---|
| `spaclone.read(airr, gex)`: the arguments swapped | an error saying gex and airr look swapped, with the call to make |
| a wrong unit (`unit_um=500` for 0.5) | a warning naming `unit_um` with the spacing and width it implies |
| a receptor table of another section or sample | an error with ids from both files (`min_overlap=`; AIR-SPACE: the clone-table barcodes that are beads) |
| a renamed spot, junction or clone column | an error listing the columns (or the closest ones) and the `spot_key=` / `columns={"junction_aa": ...}` / `clone_key=` to pass |
| `.h5mu` or a tissue image without the optional packages | an `ImportError` naming the extra: `pip install 'spaclone[scirpy]'` / `'spaclone[images]'` |
| `spaclone.run(species=...)` that does not fit the gene names | a warning (mouse `Cd19` vs human `CD19`); a misspelt species is an error |

## What is recorded

`data.meta["io"]` holds the receptor format with its evidence, the column mapping, the meaning of every count
column, the junction test, the locus and productive rules, the GEX format, the coordinate source and the unit rule,
the barcode rule with the overlap before and after, and the annotation candidates. `spaclone.read(..., verbose=True)`
prints the same summary.
