"""The palette of spaclone: every colour the plots use, for the black house style and for white.

This module is the single source of the package's colours. Plot modules take every surface, ink, categorical,
region, cell-class, isotype, chain, receptor, verdict and outline colour and every sequential or diverging ramp from
here; none defines its own.

A colour follows the entity, never its rank: a chain, a truth label, a region or an isotype keeps its colour whatever
else is on the plot, and a name without a fixed colour takes a slot by its position in the column's full list of
categories (the declared categories of a categorical column, else its sorted values), not among the categories drawn.
Slots are never cycled: names beyond the free slots are drawn in a neutral grey.

Every categorical set was checked with the dataviz palette validator (``validate_palette.py``; OKLab Delta E x 100,
Machado 2009 colour-vision simulation) on its own surface, black ``#000000`` (dark) and white ``#ffffff`` (light):

* :data:`SLOTS`: adjacent pairs Delta E >= 19 with full colour vision and >= 8.4 under protanopia and deuteranopia;
  the first three slots pass every pair. Three light slots (aqua, yellow, magenta) sit below 3:1 on white, so they
  are always drawn with a legend or direct labels.
* :data:`REGION_COLORS` (5), :data:`ISOTYPE_COLORS` classes (5), :data:`MARKS` hotspot outlines (3),
  :data:`RECEPTOR_COLORS` (2), :data:`LOCATION_COLORS` (4) and :data:`OUTLINE_SLOTS` (5): every pair >= 15 with full
  colour vision and >= 8 under protanopia and deuteranopia, in both modes; every colour >= 3:1 on its surface.
* :data:`CELL_COLORS`: nine classes cannot keep every pair apart; read cell-class maps with the legend, or fold minor
  classes (``groups=`` of :func:`spaclone.pl.gex_map`).
* Sequential ramps (:data:`RAMPS`) are one hue with monotone lightness; on black they rise from the surface to a
  luminous end, on white they fall from near-white to a deep end. :data:`DIVERGING` is blue - grey - red with equal
  lightness steps per arm.

Luminous by design: on black, the isotype classes IgM, IgE and IgA, the hotspot outlines (yellow, vermilion, cyan;
the archive look of :func:`spaclone.pl.hotspot_atlas`) and the lineage location colours sit above the validator's dark
lightness band, which is calibrated for a ``#1a1a19`` surface; the outline slots include the ink colour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S

# ------------------------------------------------------------------------------------------------ surfaces and ink

#: Surfaces and ink of the two house styles: pure black (the house style) and white. ``tissue`` fills a section
#: behind data, ``tissue_strong`` draws tissue spots under receptor dots, ``glass`` the slide outside the tissue,
#: ``unassigned`` units without a class, ``null`` the placebo (null) draws, ``panel`` bands and cards, ``floor`` the
#: zero of a colour-mixed map.
CHROME: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#000000",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#8a8984",
        "grid": "#2b2b2a",
        "axis": "#4a4946",
        "tissue": "#1f1f1e",
        "tissue_strong": "#3a3a38",
        "glass": "#141413",
        "unassigned": "#3a3a39",
        "background": "#262625",
        "null": "#6b6a66",
        "null_edge": "#8a8984",
        "panel": "#111110",
        "floor": "#101010",
    },
    "light": {
        "bg": "#ffffff",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#6b6a66",
        "grid": "#e4e3df",
        "axis": "#c3c2b7",
        "tissue": "#e9e8e4",
        "tissue_strong": "#d6d5cf",
        "glass": "#efeee9",
        "unassigned": "#d4d3cf",
        "background": "#e9e8e4",
        "null": "#c3c2b7",
        "null_edge": "#898781",
        "panel": "#f5f4f1",
        "floor": "#f4f4f2",
    },
}

# ------------------------------------------------------------------------------------------------- categorical

#: The eight categorical slots (blue, orange, aqua, yellow, magenta, green, violet, red) per surface.
SLOTS: dict[str, tuple[str, ...]] = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}
#: Neutral colour of names beyond the slots and of pooled "other" categories.
NEUTRAL = {"light": "#a3a29c", "dark": "#6b6a66"}
#: Fixed slot of every chain (the same chain has the same colour in every plot).
LOCUS_SLOT = {locus: i for i, locus in enumerate(S.ALL_LOCI)}
#: Fixed slot of the simulator's truth labels.
TRUTH_SLOT = {"genuine": 0, "swapped": 1, "sink": 2}
#: Receptor families: BCR blue, TCR magenta.
RECEPTOR_COLORS: dict[str, dict[str, str]] = {
    "dark": {"BCR": "#3987e5", "TCR": "#e8559b"},
    "light": {"BCR": "#2a78d6", "TCR": "#c2399b"},
}
#: Fixed colours of tissue regions: follicle blue, germinal centre (GC) orange, T zone magenta, ASC area violet, red
#: pulp green (every pair passes the all-pairs floors in both modes).
REGION_COLORS: dict[str, dict[str, str]] = {
    "dark": {"follicle": "#3987e5", "GC": "#d95926", "T zone": "#ed49ab", "ASC": "#8645b2", "red pulp": "#179467"},
    "light": {"follicle": "#2a78d6", "GC": "#eb6834", "T zone": "#b7489b", "ASC": "#682dac", "red pulp": "#04a97a"},
}
#: Other spellings of the region labels (``"PC"`` is the plasma-cell / ASC area).
REGION_ALIASES = {
    "PC": "ASC",
    "germinal centre": "GC",
    "germinal center": "GC",
    "T_zone": "T zone",
    "t_zone": "T zone",
    "red_pulp": "red pulp",
}
#: Fixed colours of the built-in cell classes (dark, light): B blue, T pink, ASC amber, GC B aqua, myeloid orange,
#: stromal green, Tfh violet, erythroid red. Nine classes cannot keep every pair apart (see the module notes).
CELL_COLORS: dict[str, tuple[str, str]] = {
    "B": ("#3987e5", "#2d7bd9"),
    "T": ("#ce5199", "#e77ba7"),
    "ASC": ("#c98500", "#e3a71f"),
    "plasma": ("#c98500", "#e3a71f"),
    "GC_B": ("#199e70", "#13ac77"),
    "myeloid": ("#cf4f18", "#d38158"),
    "stromal": ("#008300", "#008300"),
    "Tfh": ("#958ed5", "#5f4ad8"),
    "erythroid": ("#e6676b", "#df444a"),
}
#: Heavy-chain isotypes: the hue names the class (IgM light blue, IgD blue, IgG red, IgE violet, IgA orange); inside
#: IgG (and human IgA) the lightness names the subclass along the locus, light to dark (an ordinal ramp, always read
#: with labels). Classes pass every pair; neighbouring subclasses differ by Delta E >= 8 under simulated colour-vision
#: deficiency.
ISOTYPE_COLORS: dict[str, dict[str, str]] = {
    "dark": {
        "IgM": "#7ecfff",
        "IgD": "#3b6cff",
        "IgG": "#ff3b3b",
        "IgG3": "#ffaaa7",
        "IgG1": "#f66d67",
        "IgG2b": "#de3c37",
        "IgG2": "#de3c37",
        "IgG2a": "#ac2724",
        "IgG2c": "#ac2724",
        "IgG2a/c": "#ac2724",
        "IgG4": "#ac2724",
        "IgE": "#c77dff",
        "IgA": "#ffa31a",
        "IgA1": "#fecb78",
        "IgA2": "#ffa31a",
        "unknown": "#8a8a8a",
    },
    "light": {
        "IgM": "#3d9be0",
        "IgD": "#2c57cd",
        "IgG": "#d02c2a",
        "IgG3": "#f28885",
        "IgG1": "#e54c48",
        "IgG2b": "#c22826",
        "IgG2": "#c22826",
        "IgG2a": "#901114",
        "IgG2c": "#901114",
        "IgG2a/c": "#901114",
        "IgG4": "#901114",
        "IgE": "#6b2f8a",
        "IgA": "#dc7d12",
        "IgA1": "#f5ae39",
        "IgA2": "#dc7d12",
        "unknown": "#8a8a8a",
    },
}
#: The five isotype classes in locus order of the mouse IgH constant genes.
ISOTYPE_CLASSES = ("IgM", "IgD", "IgG", "IgE", "IgA")
#: Hotspot and structure outlines: CSR hotspots yellow, SHM hotspots vermilion, the germinal-centre B area cyan;
#: ``hot`` / ``cold`` outline concentration hotspots and cold spots of any share map; ``nomut`` draws unmutated or
#: unswitched loci.
MARKS: dict[str, dict[str, str]] = {
    "dark": {
        "csr": "#ffd23f",
        "shm": "#ff5a36",
        "gcb": "#2ee6ff",
        "hot": "#ffd23f",
        "cold": "#2ee6ff",
        "nomut": "#8c8c8c",
    },
    "light": {
        "csr": "#c98500",
        "shm": "#c52e1a",
        "gcb": "#0b8fb3",
        "hot": "#c98500",
        "cold": "#0b8fb3",
        "nomut": "#9a9993",
    },
}
#: Location classes of lineage tips (the 2026-10 lineage figures): CSR hotspot yellow, follicle blue, T zone pink,
#: red pulp brown.
LOCATION_COLORS: dict[str, dict[str, str]] = {
    "dark": {"CSR hotspot": "#ffd23f", "follicle": "#2864dd", "T zone": "#ff4fd8", "red pulp": "#c2691f",
             "unknown": "#8a8a8a"},
    "light": {"CSR hotspot": "#c98500", "follicle": "#2f6fe4", "T zone": "#d6409f", "red pulp": "#8a4b12",
              "unknown": "#8a8a8a"},
}  # fmt: skip
#: Colours of extra outlines on a map, by their position among the outlines drawn without an explicit colour
#: (T zone magenta first, then ink, green, orange, violet).
OUTLINE_SLOTS: dict[str, tuple[str, ...]] = {
    "dark": ("#ed49ab", "#f0efec", "#3dfc8f", "#ff8c1a", "#c77dff"),
    "light": ("#b7489b", "#0b0b0b", "#1d9a6c", "#d9661f", "#6250d6"),
}
#: Verdicts of a trust report: signal blue, inconclusive / limited orange, the rest neutral.
VERDICT_COLORS: dict[str, dict[str, str]] = {
    "dark": {
        "signal": SLOTS["dark"][0],
        "not detected": "#8a8984",
        "inconclusive": SLOTS["dark"][1],
        "untestable": "#4a4946",
        "info": "#8a8984",
        "limited": SLOTS["dark"][1],
        "not tested": "#8a8984",
    },
    "light": {
        "signal": SLOTS["light"][0],
        "not detected": "#898781",
        "inconclusive": SLOTS["light"][1],
        "untestable": "#c3c2b7",
        "info": "#898781",
        "limited": SLOTS["light"][1],
        "not tested": "#898781",
    },
}
#: Labels drawn in a neutral colour (no class).
GREY_LABELS = frozenset({"unassigned", "other", "nan", "None", "none", "off_tissue", "background", "outside regions"})
_REGION_ONLY = {"GC", "follicle", "T zone", "red pulp", *REGION_ALIASES} - set(CELL_COLORS)

# ------------------------------------------------------------------------------------------------------- ramps

#: Sequential ramps, low to high value, per surface: on black they rise from the surface to a luminous end, on white
#: they fall from near-white to a deep end. ``csr`` (switched share, blue to ice), ``shm`` (mutated share, green),
#: ``gex`` (module scores, violet), ``tcr`` (magenta), ``blue`` (any other magnitude and heatmaps), ``blue_ordinal``
#: (ordered bins; the light end still clears 2:1 on white), ``lineage`` (switched share behind lineage trees).
RAMPS: dict[str, dict[str, tuple[str, ...]]] = {
    "csr": {
        "dark": ("#08122e", "#173a8a", "#2f6fe4", "#2ec4ff", "#9ff4ff"),
        "light": ("#f4f8fd", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"),
    },
    "shm": {
        "dark": ("#020d06", "#0a3d22", "#138a4a", "#35d27a", "#d2ffe6"),
        "light": ("#f3faf5", "#c7ebd3", "#5cc287", "#16834a", "#0a3d22"),
    },
    "gex": {
        "dark": ("#0a0612", "#2a1550", "#5b2ca0", "#9d6cf0", "#e8dcff"),
        "light": ("#ebe2fa", "#cdb7f2", "#a07ae3", "#6a3bbf", "#2a1550"),
    },
    "tcr": {
        "dark": ("#0d0410", "#3a0d3a", "#8e1f72", "#e0399f", "#ff8fd0", "#ffe3f4"),
        "light": ("#fdf3f8", "#f8c8dc", "#e887b6", "#c2428d", "#821d5e", "#480c34"),
    },
    "blue": {
        "dark": ("#0e1d33", "#16314f", "#1f4f8f", "#256abf", "#3987e5", "#6da7ec", "#cde2fb"),
        "light": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    },
    "blue_ordinal": {
        "dark": ("#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"),
        "light": ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"),
    },
    "lineage": {
        "dark": ("#050b1f", "#10306e", "#2556c8", "#2ea8ff", "#a6f6ff"),
        "light": ("#f4f8fd", "#c9dcf2", "#7fb0e0", "#2f74c0", "#0b3d7a"),
    },
}
#: The default sequential ramp (the house blue: :data:`RAMPS` ``csr``), for gene expression and any other magnitude
#: drawn on tissue.
RAMPS["seq"] = RAMPS["csr"]
#: Diverging maps (blue - neutral grey - red) for standardised values, equal lightness steps per arm. On black the
#: arms brighten away from a dark grey centre; on white they deepen away from a light grey centre.
DIVERGING: dict[str, tuple[str, ...]] = {
    "dark": ("#afd5fe", "#4c94ec", "#2e5794", "#2e2e2c", "#8f3734", "#e36462", "#fcc0bc"),
    "light": ("#11458c", "#2c79d9", "#a4c8f1", "#f0efec", "#eeb3ad", "#d33a3c", "#831a18"),
}

# ------------------------------------------------------------------------------------- the two reference figures

#: Colours of :func:`spaclone.pl.hotspot_atlas` (the 2026-10 archive figure ``03-01``): black slide, blue CSR ramp,
#: green SHM ramp, violet module-score ramp, CSR hotspots yellow, SHM hotspots vermilion, GC-B area cyan.
ATLAS_STYLES: dict[str, dict[str, Any]] = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "fg": {"dark": "#ffffff", "light": "#0b0b0b"}[mode],
        "muted": {"dark": "#a6a6a6", "light": "#52514e"}[mode],
        "tissue": {"dark": "#1c1c1c", "light": "#ececea"}[mode],
        "tissue_edge": {"dark": "#a6a6a6", "light": "#898781"}[mode],
        "frame": {"dark": "#d9d9d9", "light": "#383835"}[mode],
        "unscored": {"dark": "#3d3d3d", "light": "#c9c8c2"}[mode],
        "nomut": MARKS[mode]["nomut"],
        "dot_edge": "#ffffff",
        "stroke": {"dark": "#000000", "light": "#ffffff"}[mode],
        "csr": MARKS[mode]["csr"],
        "shm": MARKS[mode]["shm"],
        "gcb": MARKS[mode]["gcb"],
        "mut": MARKS[mode]["shm"],
        "iso": {k: ISOTYPE_COLORS[mode][k] for k in ("IgG", "IgA", "IgE")},
        "csr_ramp": RAMPS["csr"][mode],
        "shm_ramp": RAMPS["shm"][mode],
        "score_ramp": RAMPS["gex"][mode],
    }
    for mode in ("dark", "light")
}
#: Colours of :func:`spaclone.pl.lineage_tree` (the 2026-10 lineage figures, archive L10).
LINEAGE_STYLES: dict[str, dict[str, Any]] = {
    mode: {
        "bg": CHROME[mode]["bg"],
        "fg": {"dark": "#ffffff", "light": "#111111"}[mode],
        "muted": {"dark": "#b3b3b3", "light": "#555555"}[mode],
        "row": {"dark": "#1c1c1c", "light": "#ececec"}[mode],
        "guide": {"dark": "#3a3a3a", "light": "#c8c8c8"}[mode],
        "tissue": {"dark": "#2b2b2b", "light": "#e3e3e3"}[mode],
        "tree": {"dark": "#b3b3b3", "light": "#4d4d4d"}[mode],
        "germ": {"dark": "#ffffff", "light": "#111111"}[mode],
        "t_edge": LOCATION_COLORS[mode]["T zone"],
        "hot_edge": LOCATION_COLORS[mode]["CSR hotspot"],
        "ramp": list(RAMPS["lineage"][mode]),
    }
    for mode in ("dark", "light")
}


# ----------------------------------------------------------------------------------------------------- helpers


def mode_of(style: str | None) -> str:
    """``"dark"`` for the dark house style, ``"light"`` otherwise."""
    return "dark" if style == "dark" else "light"


def theme(mode: str = "light") -> dict[str, str]:
    """Surface and ink colours (:data:`CHROME`) of a light or dark surface."""
    return dict(CHROME[mode])


def ramp(name: str, mode: str = "light") -> tuple[str, ...]:
    """Stops of the sequential ramp ``name`` (:data:`RAMPS`), low to high value, on a light or dark surface."""
    return RAMPS[name][mode]


def cmap(name: str, mode: str = "light", *, n: int = 256) -> Any:
    """Matplotlib colormap of a ramp of :data:`RAMPS`, or of ``"diverging"`` (:data:`DIVERGING`)."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = DIVERGING[mode] if name == "diverging" else RAMPS[name][mode]
    return LinearSegmentedColormap.from_list(f"spaclone_{name}_{mode}", list(stops), N=n)


def ink_on(fill: Any) -> str:
    """Label colour inside a filled mark: near-black on light fills (relative luminance above 0.3), white on the rest."""
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(fill))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return CHROME["light"]["ink"] if float(lin @ [0.2126, 0.7152, 0.0722]) > 0.3 else CHROME["light"]["bg"]


def isotype_color(name: str, mode: str = "light") -> str:
    """Fixed colour of an isotype name: its own, else its class colour, else the colour of ``"unknown"``."""
    pal = ISOTYPE_COLORS[mode]
    if name in pal:
        return pal[name]
    base = str(name).split(" ")[0]
    for cls in ISOTYPE_CLASSES:
        if base.startswith(cls):
            return pal[cls]
    return pal["unknown"]


def region_colors(mode: str = "light") -> dict[str, str]:
    """Fixed colour of every region label and alias on a light or dark surface."""
    base = dict(REGION_COLORS[mode])
    return base | {alias: base[name] for alias, name in REGION_ALIASES.items()}


def is_region_vocabulary(names: Iterable[str], *, column: str | None = None) -> bool:
    """True when ``names`` are tissue regions.

    The column is named like one (``region``, ``compartment``), or every name that is not neutral is a region label
    and at least one of them is not also a cell class (``"ASC"`` alone reads as the cell class).
    """
    if column is not None and any(k in str(column).lower() for k in ("region", "compartment")):
        return True
    named = {str(n) for n in names} - GREY_LABELS
    regions = set(region_colors("light"))
    return bool(named) and named <= regions and bool(named & _REGION_ONLY)


def named_colors(names: Iterable[str], mode: str = "light", *, column: str | None = None) -> dict[str, str]:
    """Fixed colours of the known names among ``names``.

    The region set for a region vocabulary (see :func:`is_region_vocabulary`), else the cell classes, with region
    labels that are not cell classes in their region colours.
    """
    names = [str(n) for n in names]
    k = 0 if mode == "dark" else 1
    regions = region_colors(mode)
    if is_region_vocabulary(names, column=column):
        return {n: regions[n] for n in names if n in regions}
    out = {n: CELL_COLORS[n][k] for n in names if n in CELL_COLORS}
    return out | {n: regions[n] for n in names if n in regions and n not in out}


def locus_colors(mode: str = "light") -> dict[str, str]:
    """Fixed colour of every chain on a light or dark surface."""
    return {locus: SLOTS[mode][i] for locus, i in LOCUS_SLOT.items()}


def vocabulary(values: Any, *, order: Iterable[str] | None = None) -> list[str]:
    """The full, fixed list of categories of a column.

    ``order`` first, then the declared categories of a categorical, else the sorted distinct values (as strings,
    missing values left out).
    """
    s = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values, dtype=object))
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in s.cat.categories]
    else:
        cats = sorted({str(v) for v in s.dropna().tolist()})
    head = [str(c) for c in (order or [])]
    return list(dict.fromkeys([*head, *cats]))


def fixed_colors(
    vocab: Iterable[str],
    *,
    mode: str = "light",
    named: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | None = None,
    neutral: Mapping[str, str] | None = None,
    slots: tuple[str, ...] | None = None,
    beyond: str | None = None,
) -> dict[str, str]:
    """Colour of every name of ``vocab``.

    Parameters
    ----------
    vocab
        The full list of categories (fixed for the column, not only the ones drawn).
    mode
        ``"light"`` or ``"dark"`` surface.
    named
        Fixed colours by name (house colours, chains, isotypes...); their hues are not given to other names.
    palette
        User overrides by name (they win over everything and do not shift the slots of other names).
    neutral
        Names drawn in a given neutral colour (``"unassigned"``, ``"other"``...), no slot used.
    slots
        Slot colours (default: :data:`SLOTS` of ``mode``).
    beyond
        Colour of names past the last free slot (default :data:`NEUTRAL` of ``mode``): no generated hue.

    Returns
    -------
    dict
        ``{name: colour}`` for every name of ``vocab``.
    """
    vocab = list(dict.fromkeys(str(v) for v in vocab))
    named = dict(named or {})
    neutral = dict(neutral or {})
    slots = SLOTS[mode] if slots is None else slots
    beyond = NEUTRAL[mode] if beyond is None else beyond
    taken = {named[v].lower() for v in vocab if v in named}
    free = [c for c in slots if c.lower() not in taken]
    unnamed = [v for v in vocab if v not in named and v not in neutral]
    rank = {v: i for i, v in enumerate(unnamed)}
    out = {}
    for v in vocab:
        if v in named:
            out[v] = named[v]
        elif v in neutral:
            out[v] = neutral[v]
        else:
            out[v] = free[rank[v]] if rank[v] < len(free) else beyond
    for k, c in (palette or {}).items():
        out[str(k)] = c
    return out


def set_y_down(ax: Any, invert_y: bool) -> None:
    """Orient the y axis: ``invert_y=True`` draws y increasing downwards (image convention), else upwards.

    Sets the limits explicitly (never toggles), so calling it twice or on an axes that is already inverted is safe.
    """
    lo, hi = sorted(ax.get_ylim())
    ax.set_ylim((hi, lo) if invert_y else (lo, hi))
