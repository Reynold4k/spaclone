# Command line

Two commands come with the package: `spaclone run` writes the location-trust report of one section, the same report
as {func}`spaclone.run`, and `spaclone sniff` says what a file is and what spaclone still needs to read it.

## A worked example

The public AIR-SPACE lymph node of day 21 (Jiang et al., *PNAS* 2026; GEO GSE286452) is built into
{mod}`spaclone.datasets`. Download it once (78 MB, checked by sha256) and keep its two paths:

```bash
pip install "spaclone[datasets]"
read -r H5AD TABLE < <(python -c "import spaclone; print(*spaclone.datasets.airspace('D21'))")
```

`spaclone sniff` recognises both files, and that they belong together:

```bash
spaclone sniff "$TABLE" "$H5AD"
```

```text
~/.cache/spaclone/GSM8727840_D21_ovp_clone_table.txt.gz
  kind: receptor; format: airspace_ovp (AIR-SPACE *_ovp_clone_table.txt.gz (MiXCR exportAlignments + bead barcode / UMI; GEO GSE286452))
  evidence: tab-delimited, 42 columns
  evidence: gzip-compressed (magic bytes)
  evidence: signature: columns barcode, umi, receptor_type, cloneId, aaSeqCDR3
  evidence: specialises mixcr_alignments
  note: read with spaclone.io.read_airspace together with the sample's bead h5ad (0.65 µm per unit).
  nothing else required
~/.cache/spaclone/GSM8727834_adata_D21_with_image_raw_counts.h5ad
  kind: gex; format: anndata
  evidence: h5ad (AnnData)
  coordinates: obsm['spatial'] (also: obs['x'], obs['y'])
  unit: 0.65 µm per unit (AIR-SPACE / Curio bead unit, applied with the clone table)
  note: read together with the AIR-SPACE clone table: spaclone.read(bead_h5ad, clone_table)
  nothing else required
```

`spaclone run` reads the two files with {func}`spaclone.read`, filters the records and runs every location-trust test
the data allow:

```bash
spaclone run --gex "$H5AD" --input "$TABLE" --out d21_report
```

```text
Verdict: Clone positions supported: clone identity follows position; clone maps and spatial clone statistics are licensed. Molecule level: not tested (same_spot: spots are not on a regular grid of pitch 0.65 um (beads, e.g. AIR-SPACE or Slide-seq): the same-spot test needs gridded spots; own_type_share: cell_type_key='auto': no cell-type label column in gex.obs; pass cell_type_key= or label the units with spaclone.gex.score_cell_types).
Molecule level: not tested (same_spot: spots are not on a regular grid of pitch 0.65 um (beads, e.g. AIR-SPACE or Slide-seq): the same-spot test needs gridded spots; own_type_share: cell_type_key='auto': no cell-type label column in gex.obs; pass cell_type_key= or label the units with spaclone.gex.score_cell_types).
Clone level: signal — clone identity follows position (colocalisation (IGH D21); pairing (IGH/IGK D21)).
Power: 5 testable row(s) in 3 tests (colocalisation 2, pairing 2, partner_sharing 1); clone-level BH family m = 5. Monte-Carlo tests used B >= 200 permutations (smallest p 0.005; alpha/m >= 0.01). A single positive row would have needed observed/chance >= 1.85 (colocalisation), 1.14 (pairing), 3.42 (partner_sharing) (median over rows). 3 decisive row(s) were untestable or inconclusive (see the table).
written: d21_report/report.md, tests.csv, methods.txt, report.png
```

The folder holds four files:

| File | Content |
|---|---|
| `report.md` | the verdicts, the table of every test and the methods paragraph, in Markdown |
| `tests.csv` | one row per test and locus: observed, chance, ratio, p, BH q, verdict, notes |
| `methods.txt` | the methods paragraph, with every parameter and reference |
| `report.png` | the one-page trust report ({func}`spaclone.pl.trust_report`) |

On this node the clone level shows a signal, so clone maps and spatial clone statistics are licensed; the
molecule-level tests are not run, because beads are not a regular grid (same spot) and the bead table has no
cell-type labels (own cell type: label the beads with {func}`spaclone.gex.score_cell_types`, save the h5ad and pass
`--cell-type-key cell_type` to add that test).

## Your own section

`--input` is the receptor table and `--gex` the spatial transcriptome of the same section, in any format that
{func}`spaclone.read` reads ([](reading_data.md)):

```bash
spaclone run --gex section.h5ad --input airr_rearrangement.tsv --out report
```

| Option | |
|---|---|
| `--format` | `auto` (default: detected from the files), or `airr`, `stereo-xcr`, `airspace` (with `--clone-table`), `cell-calls`, `spatialair` |
| `--species` | `mouse` (default) or `human`: the constant-region genes of the same-spot test |
| `--cell-type-key`, `--own-types` | the cell-type column of the transcriptome and each receptor's own types (default `auto`: used when the choice is unambiguous, otherwise the report lists the candidates) |
| `--unit-um`, `--spot-key`, `--bin-size`, `--count-type` | what spaclone cannot detect for some formats (the error says which one to give) |
| `--min-reads`, `--on-tissue` | read floor and tissue filter of the records |
| `--options` | per-test options as JSON, e.g. `'{"pairing": {"min_units": 10}}'` |
| `--n-perm`, `--seed` | permutations of every Monte-Carlo test (at least) and the seed |

`spaclone run --help` lists every option.
