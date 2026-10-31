# spaclone

```{image} _static/overview_light.png
:class: only-light
:width: 100%
:alt: spaclone overview: reading spatial transcriptome and receptor files, QC, the trust report, CSR and SHM hotspot maps, the hotspot atlas with each region's transcriptome identity, CSR potential and flow, lineage trees on tissue, clone states, clone maps, isotypes by region, a placebo test, and the integrated packages.
```

```{image} _static/overview_dark.png
:class: only-dark
:width: 100%
:alt: spaclone overview: reading spatial transcriptome and receptor files, QC, the trust report, CSR and SHM hotspot maps, the hotspot atlas with each region's transcriptome identity, CSR potential and flow, lineage trees on tissue, clone states, clone maps, isotypes by region, a placebo test, and the integrated packages.
```

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
