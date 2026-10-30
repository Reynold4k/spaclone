# Validation

spaclone was developed from a 2026 analysis of spatial BCR and TCR data that also used unpublished data; those data and
their numbers are not distributed. The validation below uses public data only, plus simulated data where stated. The
analysis scripts and their result tables (the *archive*) are the reference: for every analysis that became a package
function, a validation script recomputes the archived numbers with the package and compares them, number by number.

## How the numbers were recomputed

Each script loads the original inputs with pandas, anndata or a {mod}`spaclone.io` reader, computes every number with
spaclone functions only, reads the archived value from the archive's result tables, and writes one row per number with
the columns `quantity`, `archived`, `package`, `match` and `note`. The scripts ran as SLURM jobs on the HPC cluster
where the data are stored. Three data sets were used, plus simulated data (the calibration of the clonal scan and the
benchmark):

| Data | Receptors | Unit | Source |
|---|---|---|---|
| StereoIR, mouse spleen | BCR and TCR (AIRR contigs) | bin20 (10 µm) | Stereo-seq 5′ immune-repertoire demonstration files |
| AIR-SPACE, mouse lymph nodes, 6 time points | BCR molecules | Curio bead (10 µm) | GEO GSE286452 |
| Stereo-XCR-seq, human lung adenocarcinoma, 11 chips | BCR cell calls | segmented cell | GEO GSE328481 |

The comparisons follow three rules:

- **Same method, same numbers.** Where the package keeps the archived method, counts and observed statistics must be
  identical.
- **Random draws.** Where the archive's generator and order of draws can be replayed, the package is called with the
  same generator in the same order ("legacy-equivalent" runs), so its null draws are identical draw for draw and
  every null summary and p-value must match. Where the package draws differently (its own generator, or an
  equivalent null built another way, such as fixed-margin tables instead of pooled shuffles), Monte-Carlo quantities
  must agree within the tolerance stated in the row's note, usually 4.5 Monte-Carlo or binomial standard errors,
  and observed statistics must still be identical.
- **Deliberate improvements.** Where the package improves a method, the legacy-equivalent setting is checked against
  the archive, and the improved result is reported in the same table with the reason for the difference.

The `match` column takes these values:

| `match` | Meaning |
|---|---|
| `True` | equal to the archived value, or within the tolerance or decision rule given in the note |
| `False` | different; the note gives the reason (the 7 rows are explained [below](#the-7-differences)) |
| `new` (only `m1_*`) | a number the archive did not compute |
| `improved` (`m1_*`, `m2_*`) | the number of a deliberately improved method, next to the archived legacy number (after the review: the own-gene rule of the UMI match, co-localisation units and sinks, own-type placement) |
| empty | an information row without a pass/fail: improved settings, method variants, descriptive numbers |

## Summary

| Scripts | What they validate | Checks | Match | Differences | No pass/fail |
|---|---|---:|---:|---:|---:|
| `m1_*` | molecule-level location trust: UMI match with decoys | 132 | 96 | 0 | 36 |
| `m2_*` | cell- and clone-level trust: co-localisation, share on own cell type, read floor | 1,217 | 1,138 | 0 | 79 |
| `m3_*` | share maps and hotspots, tissue structures, niche profiles, sharing between regions | 703 | 578 | 0 | 125 |
| `m4_*` | sequence validity, clone definitions, loci, calibration of the clonal scan, lineage tests | 197 | 197 | 0 | 0 |
| `m5_*` | readers, simulators and the benchmark harness | 239 | 239 | 0 | 0 |
| `m6_*` | filtering and interoperability | 81 | 81 | 0 | 0 |
| `m7_*` | transcriptome: Ig RNA, regions, cell types, workflow | 746 | 730 | 7 | 9 |
| `m8_*` | BCR: isotypes, somatic hypermutation, CSR and SHM hotspots, lineage trees, spectrum and selection | 250 | 194 | 0 | 56 |
| `m9_*` | repertoire statistics, public clonotypes, clone dispersion, invariant TCR chains, Pgen | 436 | 376 | 0 | 60 |
| **Total** | | **4,001** | **3,629** | **7** | **365** |

The counts are rows of the result tables, recounted from the `match` column after the review fixes (2026-10-30) and
after the rows on unpublished data were removed (2026-10-31). `m4_scan.csv` holds the 20 rows of
`m4_scan_part_calibration.csv`, the calibration job of the clonal-scan validation; they are counted once. The 365 rows
without a pass/fail are 24 `new`, 52 `improved` and 289 information rows. The review fixes added rows (the new
defaults next to the legacy settings); the 7 differences are the same rows as before.

Four tables keep only their rows on public or simulated data: `m2_own_type_share.csv`, `m4_scan.csv` (with
`m4_scan_part_calibration.csv`), `m5_readers.csv` and `m6_filtering.csv`. The scripts that wrote them also read the
unpublished data and are not distributed.

The 3,629 matches are of three kinds:

- 3,302 rows where the package value equals the archived value (to floating-point rounding);
- 310 rows where the two values differ by less than the tolerance given in the note, or meet the criterion it states
  (for example the same decision at FDR 5%). Most of these are Monte-Carlo quantities computed with the package's
  own draws (for example the co-localisation nulls of `m2_colocalisation.csv` and the shuffle p-values of
  `m4_lineage.csv`); others are the calibration rows of the clonal scan (`m4_scan.csv`: false-positive rates below
  the bound in the note), binomial checks of the benchmark (`m5_benchmark.csv`) and values within a numerical
  tolerance;
- 17 rows without an archived value: 10 pre-stated criteria that passed (9 false-positive rates of the clonal scan
  under exact random labelling, for the archive's, the 1.0.dev0 and the current design; one cluster rule of
  `m7_workflow.csv`), 6 information rows of `m5_readers.csv` that the script marked `True`, and one row of
  `m5_benchmark.csv` marked `True` for a check that the script does not run (the same-spot numbers of the archived
  synthetic demo; its note says that the functions it needs were stubs when the script was written).

The package's own test suite adds unit tests with known answers, calibration tests under each null (false-positive rate
at most α plus binomial tolerance), power tests on planted signal and determinism tests: 920 tests, 878 run by default
and 42 slow calibration and power tests (`pytest -m slow`).

(the-7-differences)=
## The 7 differences

| File | Quantity | Archived | Package | Cause |
|---|---|---:|---:|---|
| `m7_regions.csv` | AIR-SPACE D7: distinct germinal centres (X3b rule) | 9 | 3 | germinal-centre count of the archive |
| `m7_regions.csv` | AIR-SPACE D10: distinct germinal centres (X3b rule) | 11 | 8 | as above |
| `m7_regions.csv` | AIR-SPACE D14: distinct germinal centres (X3b rule) | 11 | 9 | as above |
| `m7_regions.csv` | AIR-SPACE D21: distinct germinal centres (X3b rule) | 16 | 9 | as above |
| `m7_workflow.csv` | top cluster of the B module | – | B 1.14 | pre-stated workflow check failed |
| `m7_workflow.csv` | top cluster of the T module | – | T 1.91 | as above |
| `m7_workflow.csv` | separation rule passed | 1 | 0 | as above |

**Germinal-centre count of the archive (4 rows).** For the X3b germinal centres of the AIR-SPACE lymph nodes, the
archive reported the number of germinal centres as the largest DBSCAN label plus one. DBSCAN labels are not renumbered
after clusters with fewer than 30 beads are dropped, so that number includes the dropped clusters: 9, 11, 11 and 16
instead of 3, 8, 9 and 9 distinct germinal centres. The germinal-centre label of each bead was right, and the sharing
statistics computed from those labels match exactly (`m3_region_sharing.csv`). With `relabel=False`,
{func}`spaclone.gex.call_regions` keeps the DBSCAN labels and reproduces the archive's count (rows "archive count =
largest DBSCAN label + 1", which match); by default it numbers the germinal centres that are kept.

**Pre-stated workflow check (3 rows).** The transcriptome workflow (`call_tissue` → `qc_metrics` → `filter` →
`preprocess` → `cluster` → `embed` → `score_cell_types` → `call_regions` → `domains` → `rank_genes_regions` →
`nhood_enrichment`) was run end to end on the AIR-SPACE D14 lymph node. No archived clustering exists, so the check was
a rule fixed before the run: the Leiden clusters with the highest mean B, T and ASC module enrichment are three
different clusters, and in each of them its own module's mean enrichment is at least 2 (twice the tissue mean) and
higher than the other two modules'. The three clusters were different and the ASC cluster passed (enrichment 3.96), but
the B and T clusters reached 1.14 and 1.91, so the rule failed. The result is reported as it came out (re-run on
2026-10-30 after {func}`spaclone.gex.cluster` was made independent of the number of threads; before, 1.14 and 1.72).

## By script

Each table lists the checks (rows), matches, differences and rows without a pass/fail of one group of scripts. Archive
step codes (A5, B2, X5, T3, ...) are given so that each number can be traced to the archived analysis.

### Molecule-level location trust (`m1_*`)

`m1_airspace_umi_match.py` rebuilds the BCR molecule keys of AIR-SPACE D10 and D14 as the archive's UMI match (X4) did
and recomputes its per-chain summary with {func}`~spaclone.qc.umi_match`. The `new` rows give the package's exact
stratified test of the BCR match rate against decoys; the `improved` rows give its default own-gene rule (a key
without an isotype falls back to its locus).

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m1_airspace_umi_match.csv` | 132 | 96 | 0 | 36 |

### Cell- and clone-level location trust (`m2_*`)

Same-clone co-localisation on the lung-cancer chips and the AIR-SPACE nodes (X5 T3), the share of receptor records on
their own cell type against random placement on the three data sets (Q5) and partner-chain sharing by read floor on
the StereoIR TCR data (TR1, TR4). The co-localisation and own-type comparisons and the bin-level part of the read-floor
script replay the archive's draws, so their nulls match draw for draw. The co-localisation and own-type comparisons
also run the package with its own draws; those nulls are compared within Monte-Carlo error, the observed statistics
exactly.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m2_colocalisation.csv` | 986 | 952 | 0 | 34 |
| `m2_own_type_share.csv` | 128 | 122 | 0 | 6 |
| `m2_read_floor.csv` | 103 | 64 | 0 | 39 |

The `improved` rows give the review's designs next to the archived ones: co-localisation on 15-µm loci with barcode
sinks removed, and the own-type share against whole spots placed in proportion to their cDNA. The co-localisation script
also writes per-unit tables for figures (`m2_banner_units_*.parquet`).

### Maps, structures, niches and sharing (`m3_*`)

On the StereoIR spleen: the cell-type masks, the 19 T zones, the compartments and every locus's distances to them
(T_common, T1, TR1); the class-switch share map with its 13 hotspots and the isotype-matched SHM test of those
hotspots (T3, T7); the niche profiles of iNKT and expanded T-cell clones by compartment and across the white-pulp and
T-zone borders (TR3, TR7). On AIR-SPACE: clone sharing between germinal centres against two placebos (X3b), with the
archive's draws replayed. The legacy settings (the archive's approximation of the effective sample size, its quantile
threshold rule) are checked exactly; the package defaults are reported next to them.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m3_csr_shm.csv` | 269 | 253 | 0 | 16 |
| `m3_niche.csv` | 202 | 182 | 0 | 20 |
| `m3_region_sharing.csv` | 200 | 116 | 0 | 84 |
| `m3_structures.csv` | 32 | 27 | 0 | 5 |

Information rows added after the review report the clone-level defaults on the same data: hotspots under whole-clone
relabelling, niche profiles with clone blocks, and sharing between germinal centres counted in cells instead of
records.

### Validity, clones, loci, scan and lineage tests (`m4_*`)

The sequence-validity filter counts, exact and Dandelion-like clones and the 15-µm loci of the StereoIR and AIR-SPACE
data (QC filter table, T1, TR1, A4), with the locus partitions compared record by record; the calibration of the
clonal scan under exact random labelling on the archive's simulated point patterns (B6b); and the within-lineage tests
of the StereoIR lineage trees (L7, L11).

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m4_validity_loci.csv` | 42 | 42 | 0 | 0 |
| `m4_scan.csv` | 20 | 20 | 0 | 0 |
| `m4_lineage.csv` | 135 | 135 | 0 | 0 |

After the review, the calibration rows of the current default (10-µm grid cells, 25-µm block relabelling) were added:
the exact null gives no false positive. The lineage
script adds the package's p-value convention (doubled one-sided p) next to the archive's centred p.

### Readers, simulators and benchmarks (`m5_*`)

`m5_readers.csv` compares the record, unit and bin counts of the readers with the archive: StereoIR AIRR contigs, the
AIR-SPACE bead files and the lung-cancer cell calls, plus the checksums of the cached GEO files. `m5_benchmark.py`
checks that {func}`~spaclone.sim.legacy_demo_chip` reproduces the archived synthetic demo input file for file, the truth
tables of {func}`~spaclone.sim.synthetic_chip`, and {func}`~spaclone.sim.plant` / {func}`~spaclone.sim.score` on a real
AIR-SPACE background.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m5_readers.csv` | 176 | 176 | 0 | 0 |
| `m5_benchmark.csv` | 63 | 63 | 0 | 0 |

### Filtering and interoperability (`m6_*`)

The QC filter table of the three data sets (units with receptors before and after the validity filter) and, on the
StereoIR BCR table, lossless round trips through scirpy and the package's own storage and the conversion to
SpatialData.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m6_filtering.csv` | 65 | 65 | 0 | 0 |
| `m6_interop.csv` | 16 | 16 | 0 | 0 |

### Transcriptome (`m7_*`)

The Ig constant-gene RNA strata (units and receptor calls per stratum) on AIR-SPACE and the lung-cancer chips (X5), the
AIR-SPACE tissue beads, germinal-centre and plasma-cell regions bead by bead (X3, X3b) and the bead cell types (X6), and
the end-to-end workflow on one lymph node with its pre-stated check.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m7_ig.csv` | 346 | 346 | 0 | 0 |
| `m7_regions.csv` | 386 | 382 | 4 | 0 |
| `m7_workflow.csv` | 14 | 2 | 3 | 9 |

### B-cell receptor biology (`m8_*`)

The StereoIR BCR isotypes and somatic-hypermutation calls recomputed from the raw AIRR file, contig by contig and locus
by locus (T1); the class-switch hotspots (T3) and the SHM test of those hotspots (T7: 90 mutated loci against an
isotype-matched expectation of 56.17); and lineage trees built by {func}`~spaclone.bcr.build_trees` compared with the
archived dowser trees (tips, collapsed tips and mutations exactly; parsimony length and tree distances reported).
`m8_spectrum_selection.csv` is information only: the archive has no mutation spectrum or selection analysis to compare
with.

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m8_airspace_shm.csv` | 27 | 11 | 0 | 16 |
| `m8_isotype_shm.csv` | 172 | 163 | 0 | 9 |
| `m8_lineage.csv` | 34 | 20 | 0 | 14 |
| `m8_spectrum_selection.csv` | 17 | 0 | 0 | 17 |

`m8_airspace_shm.py` (added after the review) reads the MiXCR alignments of AIR-SPACE D21
({func}`~spaclone.io.parse_mixcr_alignment`) and checks the package's somatic-hypermutation calls against a direct
parse of the same alignment strings and the tutorial's calculation; it has no archived counterpart.

### Repertoire, dispersion and TCR (`m9_*`)

AIR-SPACE clone dynamics with rarefaction (A1), public clones across lymph nodes (A4) and the germinal-centre versus
plasma-cell Gini contrast (X3, X3b); the "clustered clone" calls of the StereoIR and AIR-SPACE data (S1), replayed with
the archive's draws; and the StereoIR iNKT and MAIT chains (TR1, TR3) and OLGA generation probabilities (TR2).

| File | Checks | Match | Differences | No pass/fail |
|---|---:|---:|---:|---:|
| `m9_airspace_repertoire.csv` | 295 | 249 | 0 | 46 |
| `m9_dispersion.csv` | 117 | 105 | 0 | 12 |
| `m9_stereoir_tcr.csv` | 24 | 22 | 0 | 2 |

## Files

The scripts, their SLURM files and the result tables are in the
[`validation/`](https://github.com/Reynold4k/spaclone/tree/main/validation) folder of the repository, as run, except
that their fixed paths on the project's HPC storage were replaced by environment variables (`SPACLONE_ARCHIVE`, the
archived analysis of the public data sets; `SPACLONE_DEMO`, the archived synthetic demo; `SPACLONE_ROOT` in the SLURM
files). They need the archived results and do not run without them. The StereoIR files are not included. The
AIR-SPACE and lung-cancer data are on GEO (GSE286452, GSE328481).
