"""Interactive figures: one self-contained HTML file per figure, with its data inside.

Every function here returns an :class:`InteractiveFigure`. It opens offline in any current browser (double-click the
saved file), needs no Python kernel or server, and is what a notebook shows inline; frontends that do not run scripts
(GitHub's notebook viewer, nbconvert to PDF, an untrusted notebook) show its static picture instead. The drawing code
is spaclone's own (no WebGL, no third-party script), so there is nothing extra to install.

* :func:`explore`: sections with a sample switcher, a layer switcher, a gene picker, a clone search and region
  selections (also ``spaclone.pl.explore``);
* the interactive forms of the ``spaclone.pl`` functions, with the same names and data arguments.

Controls: drag to pan, scroll or pinch to zoom (in a notebook, click the figure first or hold Ctrl / Cmd), box zoom,
box and free-shape selection (counts in the side panel, with a button that copies the shape in the section's original
micrometres), double-click to reset; legend entries show or hide a category (double-click: only that one).
"""

from spaclone.pl.interactive._atlas import (
    csr_share_map,
    hotspot_atlas,
    hotspot_overview,
    region_zoom,
    share_map,
    shm_share_map,
)
from spaclone.pl.interactive._explore import explore
from spaclone.pl.interactive._figure import FigureSizeWarning, InteractiveFigure, Options, options
from spaclone.pl.interactive._flow import csr_flow, csr_map
from spaclone.pl.interactive._lineage import lineage_tree, tree_tissue
from spaclone.pl.interactive._maps import clone_map, gex_map, gex_regions, isotype_map, spatial_records, spot_qc_map
from spaclone.pl.interactive._plots import (
    clonal_expansion,
    clonal_scan,
    csr_transitions,
    diversity,
    flux_matrix,
    gex_volcano,
    niche_profile,
    overlap,
    placebo,
    read_support,
    stationary_distribution,
    trust_report,
)

__all__ = [
    "FigureSizeWarning",
    "InteractiveFigure",
    "Options",
    "clonal_expansion",
    "clonal_scan",
    "clone_map",
    "csr_flow",
    "csr_map",
    "csr_share_map",
    "csr_transitions",
    "diversity",
    "explore",
    "flux_matrix",
    "gex_map",
    "gex_regions",
    "gex_volcano",
    "hotspot_atlas",
    "hotspot_overview",
    "isotype_map",
    "lineage_tree",
    "niche_profile",
    "options",
    "overlap",
    "placebo",
    "read_support",
    "region_zoom",
    "share_map",
    "shm_share_map",
    "spatial_records",
    "spot_qc_map",
    "stationary_distribution",
    "tree_tissue",
    "trust_report",
]
