# Concepts

## Two questions, in this order

1. **Is the receptor signal where it is drawn?** (`spaclone.qc`) A spatial receptor library can give real sequences
   and random positions. PCR template switching joins a CDR3 to the barcode of an unrelated captured cDNA: the
   molecule's identity survives, its position becomes a random cDNA-bearing spot. Nothing on the molecule records the
   swap, and the usual quality metrics (receptor density follows B cells, receptor-positive cells express Ig genes)
   pass just as well after every receptor has been moved to a random place.
2. **Where do clonal features concentrate?** (`spaclone.tl`) Only for data that pass the first question: class
   switching, somatic mutation and clonal expansion maps, clones concentrated in one region, niches along the border of
   a structure, lineage trees whose tips sit in tissue.

## Units

| Unit | What it is | Why it matters |
|---|---|---|
| record | one row of the receptor table: a molecule, a contig or a cell-level call | the input |
| spot | the barcode unit: a DNB (0.5 µm), a bin, a bead (10 µm) or a segmented cell | the unit that shares barcodes with GEX; molecule-level tests count spots, not records |
| singleton spot | a spot carrying records of one clone of the chain | a spot with tens to hundreds of clones is a *barcode sink* (a barcode shared by chimeric molecules), not a cell |
| locus | same-clone records within ~15 µm merged (`pp.merge_loci`) | one cell's mRNA spans several bins; concentration tests count loci to avoid pseudoreplication |

All coordinates are micrometres. Readers convert from each technology's native unit and record the factor.

## Null designs

Every test in spaclone compares an observed statistic with the same statistic on randomised data. The randomisation
is an explicit object (`spaclone.nulls`):

- {class}`~spaclone.nulls.Permute` — *random labelling*: exchange labels (clone, isotype, mutated, …) between records
  with positions fixed, optionally only inside strata (an Ig RNA depth bin, a compartment, an isotype class) and
  optionally moving whole blocks (all records of one spot, all loci of one clone).
- {class}`~spaclone.nulls.Relocate` — *random placement*: move whole units (with their read counts and every other
  attribute) to random slots (spots, beads or cells), inside strata (crowding, depth) or local windows, optionally
  weighted by RNA amount.

What a design keeps fixed decides what a test can conclude. Two rules came out of the work that produced this package:

1. **Keep the confounders in the null.** Receptor calls follow local Ig RNA in a working assay and in a broken one, so a
   co-localisation test must shuffle clone labels *within* Ig RNA depth bins; a pairing test must keep how many clones
   share each spot; a placement null must move molecules *with* their read counts, or larger clones win majority votes
   by construction.
2. **The statistic must contain the whole pipeline.** {func}`~spaclone.nulls.placebo` re-runs the statistic from scratch
   on every randomised data set, including deduplication, gating, voting and thresholds. A partial placebo is biased.

## Reading a result

Every test returns a {class}`~spaclone.TestResult`: a tidy `table` (one row per tested unit), a `summary` of headline
numbers, the `params` that define the test (including the null design and the seed) and, optionally, the `null` draws.
Monte-Carlo p-values are `(1 + #{null ≥ observed}) / (B + 1)` and are never 0; their floor is `1 / (B + 1)`, so the
number of permutations must leave room for the multiple-testing threshold (for example ≥ 20 × the number of clones
tested, or a family-wise max statistic).

A metric that a placebo reproduces is not evidence that receptors sit where their cells were. Report the placebo
distribution next to the observed value, and check that the conclusion holds under at least two null designs
({func}`~spaclone.nulls.compare_designs`).
