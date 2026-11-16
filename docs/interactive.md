# Interactive figures

Every interactive figure of spaclone is **one HTML file with its data inside**. It opens by double-click in a current
browser, works offline, can be e-mailed, and is what a notebook shows inline. Nothing runs behind it: no Python kernel,
no server, no network access, no browser extension. The drawing code is spaclone's own (a small script inside the file,
MIT licence), so there is nothing extra to install: `pip install spaclone` is enough (`pip install
"spaclone[interactive]"` is accepted and installs nothing more).

```python
import spaclone
from spaclone.pl import interactive as I

fig = spaclone.pl.explore(sa)            # one section: records, transcriptome, layers, genes, clones
fig = spaclone.pl.explore({"D7": sa7, "D21": sa21}, hotspots={"D21": hot21})   # a sample switcher
fig = I.hotspot_atlas(hot)               # the interactive form of spaclone.pl.hotspot_atlas
fig.save("atlas.html")                   # one file, opens anywhere
fig                                       # in a notebook: the figure itself
fig.show()                                # outside a notebook: the default browser
```

## What you can do in a figure

| Action | How |
|---|---|
| Zoom | scroll or pinch; drag a box with the zoom tool (or Shift + drag); `+` / `−` keys |
| Pan | drag; arrow keys |
| Reset | double-click, the reset button, or the `0` key |
| Switch | the selectors above the figure: sample, layer (isotype, chain, clones, share maps, cell type, regions, gene, ...), region, tree, metric |
| Pick a gene | type in the gene box: matches are listed as you type (any case); Enter takes the exact name or the first match; a name the file does not hold is reported, not drawn |
| Find clones | type a clone id: up to eight clones, each in its own colour, joined along their minimum spanning tree |
| Hover | a record, unit, pixel, cell or tip: its clone, isotype, V / J, CDR3, reads, cell type, value |
| Show or hide | click a legend entry; double-click to show only that one |
| Count what is inside | the box or free-shape selection tool: records, clones, isotype and chain shares, mean reads (or units by cell type); "copy shape (µm)" copies the shape in the section's original micrometres |
| Save a picture | the download button saves the current view as PNG |

In a notebook the mouse wheel scrolls the notebook until you click the figure (or hold Ctrl / ⌘); a standalone file
zooms with the wheel directly. Panels of one figure that show the same section (the CSR and SHM maps of the atlas)
zoom and pan together; the µm scale bar follows the zoom.

Tissue maps have the orientation of the static figures of the same function (the section's long axis horizontal, y
downwards) and the same colours: isotypes, chains, regions, hotspot outlines (CSR yellow, SHM vermilion), the house
ramps, in the dark and the light style.

## Where they work

| Where | Behaviour | Checked |
|---|---|---|
| A saved file in Chrome, Edge, Firefox | interactive, offline (`file://`) | Chromium 159 and Firefox 159 (headless), in the browser tests |
| A saved file in Safari | interactive | not tested yet: the browser tests include WebKit (Safari's engine), which cannot start on the test machine and runs in the CI browser job, not yet run; the figure needs only the 2D canvas, pointer events and `ResizeObserver` (Safari 13.1 and later) |
| JupyterLab 4 and Notebook 7: cells run in the session | interactive | JupyterLab 4.0.11 and 4.6.4, Notebook 7.0.8 and 7.6.3 (Chromium) |
| JupyterLab 4 and Notebook 7: a notebook you ran and saved (trusted) | interactive | the same versions |
| JupyterLab 4 and Notebook 7: a notebook from elsewhere, not trusted (`jupyter trust notebook.ipynb` trusts it) | the picture of the first view | the same versions |
| VS Code notebooks, Google Colab | interactive | not run here; both run scripts in HTML outputs. Where the figure's own frame cannot start, it starts inside the output after 9 s |
| GitHub, and other viewers that remove scripts | the picture, if the viewer keeps images inside HTML outputs | not run here. For a notebook meant to be read on GitHub, also show the picture as an image output: `IPython.display.Image(fig.to_png())` |
| nbconvert to HTML | interactive (the exported page runs the figure) | nbconvert 7.17.2, the page opened in Chromium |
| nbconvert to HTML with `--sanitize-html` | the figure's HTML source as text: that option escapes every HTML output, tables included | nbconvert 7.10.0 and 7.17.2 |
| nbconvert to PDF / LaTeX | the picture (`image/png`) | nbconvert 7.10.0 and 7.17.2 |
| These docs (Sphinx, MyST-NB) | interactive | |

A notebook output carries the interactive figure **and** a PNG of its first view: a frontend that cannot run the
figure shows the picture, and the picture is replaced only once the figure has started. A figure larger than
`spaclone.pl.interactive.options.notebook_max_mb` (20 MB) is shown as its picture in the notebook, with a line saying
to save it: notebooks stay small, and Jupyter's output-rate limit is never reached.

The drawing does not use WebGL: it works where WebGL is off or unavailable, after a lost graphics context, and with any
number of figures on one page (browsers allow only about 16 WebGL contexts at a time). If the figure cannot start at all
(a very old browser, a damaged file), the saved file and the notebook output show the static picture and a line saying
why, never an empty box.

## Size, genes and precision

| Data | How it is stored | Precision |
|---|---|---|
| Positions | 32-bit floats, after the section's rotation | 0.0002 µm on a 3 mm section |
| Categories (chain, isotype, clone, cell type, ...) | codes into a table of names | exact |
| Counts and reads | 32-bit floats | exact below 16.7 million |
| Other numbers on hover | 16-bit levels between their minimum and maximum | within 1/131,070 of the range |
| Gene counts (integer data) | per gene, the units where it is detected, compressed | exact; log1p(counts per 10,000) is computed in the browser from the counts and each unit's total |
| Gene values that are not counts | 8-bit levels of each gene's range [0, max] | within max / 510; the colour bar says so |
| Share maps | the image of the static figure, and the share of each pixel (16-bit) for hover | as the static figure |

Large blocks are compressed with zlib. `genes=` sets the genes of the gene picker: a number `k` (default 300: the most
variable genes of the drawn units, plus the immune marker and immunoglobulin genes found in the data), a list of names,
`"all"`, or `None`. `max_points=` (default 500,000 per sample) caps the records and units drawn: above it a random subset
is drawn, counted in selections, and a note in the figure says so; `max_points=None` draws everything. Saving a file
larger than `options.warn_mb` (50 MB) warns. `fig.nbytes` gives the size before saving.

Measured on public and synthetic data (file sizes include the picture of the first view; times in headless Chromium
159 on a server CPU without a graphics card, 1440 × 900 window: opening the file until the figure is drawn, and one
full redraw after a zoom step):

| Figure | File | Opens in | Redraw | Gene pick |
|---|---|---|---|---|
| explorer, AIR-SPACE day 21 with hotspots: 17,173 records, 23,040 beads, 348 genes (default) | 3.4 MB | 0.26 s | 7 ms | 9 ms (first 36 ms) |
| the same with all 31,053 genes | 12.2 MB | 0.34 s | 7 ms | 6 ms (first 0.27 s) |
| explorer, the six AIR-SPACE nodes with hotspots, 348 genes; sample switch 25 ms | 14.7 MB | 0.38 s | 5 ms | 6 ms (first 29 ms) |
| the same with all 31,053 genes; sample switch 0.32 s | 76.9 MB | 1.0 s | 5 ms | 5 ms (first 0.25 s) |
| explorer, `datasets.synthetic_example()` (7,954 records, 19,345 units, 27 genes) | 1.6 MB | 0.16 s | 6 ms | 6 ms |
| `gex_map`, a synthetic section of 150,000 units (7 cell types, 20 genes; no picture) | 3.8 MB | 0.18 s | 10 ms | |
| `spatial_records`, 500,000 synthetic records (no picture) | 12.7 MB | 0.43 s | 47 ms | |

## The numbers behind a selection

The side panel computes its counts in the browser from the data in the file. The same numbers in Python:

```python
fig.summary(box=(x0, x1, y0, y1))                 # display coordinates of the figure (µm after rotation)
fig.summary(polygon=[(x, y), ...], state={"sample": "D21"})
```

A record on the edge of a box is inside it; a free shape follows the even-odd rule. The "copy shape (µm)" button gives
the same shape in the original micrometres of the section, e.g. for `matplotlib.path.Path(shape).contains_points(xy)`.

## Sharing

The file holds the data it shows (positions, clone ids, CDR3s, gene counts): share it as you would share the data.
`include_js="cdn"` loads the drawing code of this spaclone release from jsDelivr instead of embedding it (about 120 kB
smaller; needs the network). `fig.save("x.html", include_js="spaclone-figure.js")` writes the drawing code once next to
the figures of a folder.

## Which functions have an interactive form

`spaclone.pl.explore` and the functions of {mod}`spaclone.pl.interactive` (same names and data arguments as the static
ones):

| Static function | Interactive form |
|---|---|
| (none) | {func}`~spaclone.pl.interactive.explore`: sample, layer, gene and clone switchers, selections |
| {func}`~spaclone.pl.spatial_records` | records with a colouring switcher (BCR / TCR, chain, isotype, truth, your column) |
| {func}`~spaclone.pl.isotype_map` | classes shown or hidden from the legend |
| {func}`~spaclone.pl.clone_map` | searchable clones with their spanning trees |
| {func}`~spaclone.pl.gex_map` | every `obs` column and a gene picker |
| {func}`~spaclone.pl.gex_regions` | regions with their instance numbers |
| {func}`~spaclone.pl.spot_qc_map` | a switcher over the spot metrics |
| {func}`~spaclone.pl.share_map`, {func}`~spaclone.pl.csr_share_map`, {func}`~spaclone.pl.shm_share_map` | a switcher over the map's layers, pixel values on hover |
| {func}`~spaclone.pl.hotspot_overview`, {func}`~spaclone.pl.hotspot_atlas`, {func}`~spaclone.pl.region_zoom` | the CSR and SHM maps zoomed together; pick or click a region to zoom to it (the atlas and the region zoom then add the region's transcriptome tile, as the static zooms); loci appear when zoomed in |
| {func}`~spaclone.pl.csr_map` | a switcher over the isotype maps, the transitions matrix next to it |
| {func}`~spaclone.pl.csr_flow` | streamlines, grid arrows or units alone |
| {func}`~spaclone.pl.lineage_tree`, {func}`~spaclone.pl.tree_tissue` | every drawn tree; a tip hovered in the tree is ringed on the section, and the other way round |
| {func}`~spaclone.pl.gex_volcano` | every class, a gene search |
| {func}`~spaclone.pl.csr_transitions`, {func}`~spaclone.pl.flux_matrix`, {func}`~spaclone.pl.overlap` | a value switcher, every cell on hover |
| {func}`~spaclone.pl.diversity`, {func}`~spaclone.pl.clonal_expansion`, {func}`~spaclone.pl.stationary_distribution`, {func}`~spaclone.pl.placebo`, {func}`~spaclone.pl.clonal_scan`, {func}`~spaclone.pl.read_support`, {func}`~spaclone.pl.niche_profile`, {func}`~spaclone.pl.trust_report` | zoom, hover, switchers where the result has several metrics |

Static only in 1.1.0 (this release converted the spatial figures and the main other ones first; the figure engine can
draw the rest, which may follow):

- the Threadfin clone-state figures {func}`~spaclone.pl.clone_state_map`, {func}`~spaclone.pl.spanning_clones` and
  {func}`~spaclone.pl.clone_axis_profile` (the `spaclone[threadfin]` bridge);
- the summary charts {func}`~spaclone.pl.isotype_composition`, {func}`~spaclone.pl.qc_panel`,
  {func}`~spaclone.pl.valid_shares`, {func}`~spaclone.pl.clones_per_spot`, {func}`~spaclone.pl.gene_usage`,
  {func}`~spaclone.pl.spectratype`, {func}`~spaclone.pl.clone_dispersion`, {func}`~spaclone.pl.density_profile`,
  {func}`~spaclone.pl.same_spot_multiplicity`, {func}`~spaclone.pl.gex_dotplot` and {func}`~spaclone.pl.placement`;
- {func}`~spaclone.pl.lineage_gallery` (writes a folder of figure files; each of its trees is
  {func}`spaclone.pl.interactive.lineage_tree`) and {func}`~spaclone.pl.mst_edges` (returns edges, draws nothing).
