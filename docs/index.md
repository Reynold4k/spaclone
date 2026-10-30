# spaclone

Is spatial immune-receptor data spatial? spaclone tests whether BCR and TCR molecules or calls sit where their cells
were, and maps where clonal features concentrate, with every statistic compared against an explicit null design.

Around these tests it covers the whole analysis: readers for the mainstream spatial and receptor formats, receptor QC
and filtering, the transcriptome, B-cell receptor biology (class-switch and mutation hotspots with the regions where
they overlap, lineage trees on the tissue, sciCSR transitions, Threadfin clone states) and repertoire statistics.
{doc}`tutorials/06_hotspots_and_lineages` shows the whole workflow on a mouse spleen.

```{toctree}
:maxdepth: 2

why
install
reading_data
concepts
tutorials/index
ecosystem
api
validation
citing
changelog
```
