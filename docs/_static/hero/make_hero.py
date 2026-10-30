"""Build the README hero, ``docs/_static/overview_dark.png`` and ``docs/_static/overview_light.png``.

Every picture in the hero is a spaclone figure drawn on real data (StereoIR mouse spleen, AIR-SPACE mouse lymph node):
``panels.py`` reads the data with ``spaclone.read`` / ``spaclone.datasets``, runs the workflows and renders each panel
with ``spaclone.pl`` in the dark and the light house style. This script runs it, then lays the panels out on one
bento grid with the package's typography (Inter), cards, a fine grain and the "Integrates" band.

usage (from the repository root)::

    python -I docs/_static/hero/make_hero.py --cache CACHE --stereoir DIR --airspace DIR --fonts DIR
    python -I docs/_static/hero/make_hero.py --cache CACHE --fonts DIR --compose-only

``--fonts`` is a folder holding the Inter and Inter Display TTF files (https://rsms.me/inter, SIL OFL 1.1); without it
the hero falls back to DejaVu Sans. The panels are cached in ``CACHE/panels/{dark,light}/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent

# ------------------------------------------------------------------------------------------------- design tokens

W = 2400  # canvas width (px); everything below is in px at this width
M, G = 104, 32  # outer margin, gutter
NCOL = 12
COLW = (W - 2 * M - (NCOL - 1) * G) / NCOL
PAD = 40  # card padding
HEAD = 150  # card header height (overline, title, call)
RADIUS = 26

THEMES = {
    "dark": {
        "page": (0, 0, 0),
        "glow": [((0.10, 0.03), (22, 52, 110), 0.55, 0.42), ((0.95, 0.55), (58, 22, 92), 0.28, 0.38),
                 ((0.20, 0.97), (8, 70, 60), 0.20, 0.30)],
        "card_top": (14, 14, 16),
        "card_bottom": (6, 6, 7),
        "edge": (40, 40, 46),
        "highlight": (70, 70, 80),
        "shadow": None,
        "grain": 2.2,
        "ink": "#f2f1ec",
        "ink2": "#b9b8b1",
        "muted": "#7d7c77",
        "accent": "#8cc4ff",
        "rule": ("#2f6fe4", "#9d6cf0", "#ed49ab"),
        "chip": (22, 22, 25),
        "chip_edge": (52, 52, 58),
    },
    "light": {
        "page": (244, 243, 239),
        "glow": [((0.10, 0.03), (205, 226, 251), 0.55, 0.45), ((0.95, 0.55), (235, 226, 250), 0.35, 0.40),
                 ((0.20, 0.97), (220, 240, 228), 0.25, 0.30)],
        "card_top": (255, 255, 255),
        "card_bottom": (255, 255, 255),
        "edge": (228, 226, 220),
        "highlight": None,
        "shadow": (0, 0, 0, 22),
        "grain": 1.4,
        "ink": "#0b0b0b",
        "ink2": "#4a4946",
        "muted": "#8a8984",
        "accent": "#1f5fbf",
        "rule": ("#2a78d6", "#6a3bbf", "#b7489b"),
        "chip": (247, 246, 243),
        "chip_edge": (226, 224, 218),
    },
}  # fmt: skip

# ------------------------------------------------------------------------------------------------------ content

TITLE = "spaclone"
TAGLINE = "Spatial BCR and TCR analysis, from raw tables to biology"
SUBLINE = "with every position checked against random placement"
PIPELINE = ["Read", "QC & filter", "Trust", "CSR · SHM maps", "Hotspot atlas", "CSR flow", "Lineages", "Repertoire",
            "Clone states"]  # fmt: skip

GEX_FORMATS = ["h5ad · scanpy · squidpy · Stereopy", "Visium SD / HD", "Stereo-seq GEF / GEM", "Xenium", "Slide-seq",
               "SpatialData"]  # fmt: skip
AIR_FORMATS = ["AIRR · Cell Ranger · TRUST4 · IgBLAST · MiXCR", "10x contigs", "MiXCR 3 / 4", "TRUST4 report",
               "Change-O", "Dandelion", "scirpy", "immunoSEQ · BD · Parse"]  # fmt: skip

#: (package, its spatial counterpart in spaclone, extra to install)
INTEGRATES = [
    ("scanpy", "transcriptome workflow, regions and DE on spots", "core"),
    ("anndata", "one container: spots + receptor records in µm", "core"),
    ("squidpy", "neighbourhood enrichment and autocorrelation of receptor labels", "[squidpy]"),
    ("scirpy", "clonotypes, expansion and diversity by region, back on tissue", "[scirpy]"),
    ("SpatialData", "read and write SpatialData objects", "[spatialdata]"),
    ("Dandelion", "clone calls mapped back to records and loci", "[dandelion]"),
    ("Immcantation", "IgBLAST · Change-O · SHazaM · dowser: germlines, mutations, trees on tissue", "[immcantation]"),
    ("sciCSR", "CSR potential, transitions and flux on tissue (R-parity verified)", "[scicsr]"),
    ("CellRank", "transition models on spatial neighbour graphs", "[scicsr]"),
    ("Threadfin", "clone-state landscapes mapped onto tissue", "[threadfin]"),
    ("OLGA", "TCR generation probabilities of public clonotypes", "[tcr]"),
]

#: Cards: (key, overline, title, call, panels stacked top to bottom, (row, col0, col1)).
CARDS = [
    ("read", "01  READ", "Any spatial transcriptome, any receptor table", "spaclone.read(gex, receptors)", [], (1, 0, 12)),
    ("records", "02  QC", "Every receptor record on its spot", "pp.qc_metrics · pp.filter_records", ["records"],
     (2, 0, 8)),
    ("trust", "03  TRUST", "Can these positions be trusted?", "spaclone.run(data)", ["trust"], (2, 8, 12)),
    ("atlas", "04  FLAGSHIP", "Where class switching and hypermutation meet, and what the transcriptome says there",
     "bcr.hotspot_workflow  →  pl.hotspot_atlas", ["atlas"], (3, 0, 12)),
    ("csr_map", "05  CLASS SWITCHING", "Isotypes and switches along the IgH locus", "pl.csr_map", ["csr_map"],
     (4, 0, 6)),
    ("csr_flow", "06  CSR DYNAMICS", "CSR potential, flow and flux on tissue",
     "bcr.csr_potential  →  tl.tpt_flux  →  pl.csr_flow", ["csr_flow", "flux"], (4, 6, 12)),
    ("lineage", "07  LINEAGES", "Lineage trees placed on the tissue", "bcr.lineage_workflow  →  pl.lineage_gallery",
     ["lineage"], (5, 0, 8)),
    ("clone_map", "08  CLONES", "Where clones gather", "tl.clonal_scan · pl.clone_map", ["clone_map"], (5, 8, 12)),
    ("clone_states", "09  CLONE STATES", "Clone states on tissue", "ext.threadfin.run  →  pl.clone_state_map",
     ["clone_states"], (6, 0, 8)),
    ("placebo", "10  YOUR OWN TEST", "Any statistic against random placement", "nulls.placebo", ["placebo"],
     (6, 8, 12)),
    ("composition", "11  REPERTOIRE", "Isotypes by region", "pl.isotype_composition", ["composition"], (7, 0, 6)),
    ("code", "12  IN CODE", "Two files in, every figure out", "pip install spaclone", [], (7, 6, 12)),
]  # fmt: skip
#: Python shown on the code card.
CODE = [
    "import spaclone as spc",
    "",
    'data = spc.read("section.h5ad", "airr.tsv")',
    "report = spc.run(data)",
    "hot = spc.bcr.hotspot_workflow(data)",
    "spc.pl.hotspot_atlas(hot)",
    "trees = spc.bcr.lineage_workflow(hot)",
    'spc.pl.lineage_gallery(trees, "lineages")',
]
#: Row heights (px); the panel rows are refitted to the panels (:func:`fit_rows`), the last row is the Integrates band.
ROW_H = {0: 480, 1: 478, 2: 700, 3: 1400, 4: 900, 5: 980, 6: 640, 7: 620, 8: 736}
ART_X = 1250  # left edge of the header art (the CSR / SHM overview)
INTEGRATES_ROW = 8
STACK_GAP = 26  # px between panels stacked in one card
SECOND_CAP = 0.38  # a stacked panel below the first is at most this fraction of the card's inner width tall

# ---------------------------------------------------------------------------------------------------- utilities


def span(c0: int, c1: int) -> tuple[float, float]:
    """The x position and width of the grid columns [c0, c1)."""
    return M + c0 * (COLW + G), (c1 - c0) * COLW + (c1 - c0 - 1) * G


def row_y(row: int) -> float:
    """Top of a grid row."""
    return sum(ROW_H[r] + (G if r > 0 else 0) for r in range(row)) + (G if row > 0 else 0)


def stack(names: list[str], inner: float, panels: Path, mode: str) -> list[tuple[str, float]]:
    """Heights of the panels stacked in a card at its inner width (missing panels get a placeholder height)."""
    out = []
    for k, name in enumerate(names):
        p = panels / mode / f"{name}.png"
        hh = 0.4 * inner
        if p.exists():
            with Image.open(p) as im:
                hh = inner * im.height / im.width
        cap = 1.2 * inner if k == 0 else SECOND_CAP * inner
        out.append((name, min(hh, cap)))
    return out


def fit_rows(panels: Path, mode: str = "dark") -> None:
    """Heights of the panel rows from the panels' aspect ratios, so that the tallest card sets its row."""
    need: dict[int, float] = {}
    for key, _ov, _ti, _call, names, (row, c0, c1) in CARDS:
        if key in ("read", "code"):
            continue
        inner = span(c0, c1)[1] - 2 * PAD
        parts = stack(names, inner, panels, mode)
        hh = sum(h for _, h in parts) + STACK_GAP * max(len(parts) - 1, 0)
        need[row] = max(need.get(row, 0.0), HEAD + 1.5 * PAD + hh)
    for row, v in need.items():
        ROW_H[row] = int(np.ceil(v))


def height() -> int:
    """Canvas height."""
    return int(row_y(INTEGRATES_ROW) + ROW_H[INTEGRATES_ROW] + 128)


class Fonts:
    """Inter / Inter Display when available, else DejaVu Sans; mono = DejaVu Sans Mono."""

    def __init__(self, folder: Path | None):
        from matplotlib import font_manager as fm

        self.fm = fm
        self.files: dict[str, str | None] = {}
        names = {"display": "InterDisplay-Bold.ttf", "display_semi": "InterDisplay-SemiBold.ttf",
                 "semi": "Inter-SemiBold.ttf", "medium": "Inter-Medium.ttf", "regular": "Inter-Regular.ttf",
                 "light": "InterDisplay-Light.ttf"}  # fmt: skip
        for key, name in names.items():
            hit = next(iter(sorted(folder.rglob(name))), None) if folder else None
            self.files[key] = str(hit) if hit else None

    def __call__(self, key: str, size_px: float):
        """Font properties of a role at a size in px."""
        from matplotlib.font_manager import FontProperties

        if key == "mono":
            return FontProperties(family="DejaVu Sans Mono", size=size_px * 0.72)
        f = self.files.get(key)
        if f:
            return FontProperties(fname=f, size=size_px * 0.72)
        weight = {"display": "bold", "display_semi": "semibold", "semi": "semibold", "medium": "medium"}.get(key)
        return FontProperties(family="DejaVu Sans", weight=weight or "normal", size=size_px * 0.72)


# ---------------------------------------------------------------------------------------------------- base layer


def _rounded_mask(w: int, h: int, radius: float, *, outline: float = 0.0, ss: int = 3) -> Image.Image:
    """Anti-aliased rounded-rectangle mask of a w x h card (filled, or a hairline ``outline`` px wide)."""
    big = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(big)
    box = (0, 0, w * ss - 1, h * ss - 1)
    if outline:
        d.rounded_rectangle(box, radius=radius * ss, outline=255, width=max(1, int(round(outline * ss))))
    else:
        d.rounded_rectangle(box, radius=radius * ss, fill=255)
    return big.resize((w, h), Image.LANCZOS)


def card_fill(theme: dict, w: int, h: int) -> np.ndarray:
    """Vertical gradient fill of a card (float32, h x w x 3)."""
    top, bot = np.array(theme["card_top"], np.float32), np.array(theme["card_bottom"], np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return (top * (1 - t) + bot * t) * np.ones((h, w, 1), np.float32)


def base_layer(theme: dict, h: int, cards: list[tuple[float, float, float, float]], rng: np.random.Generator):
    """Page (glows + grain), card shadows, gradient card fills, hairline edges and top highlights."""
    yy, xx = np.mgrid[0:h, 0:W].astype(np.float32)
    page = np.ones((h, W, 3), np.float32) * np.array(theme["page"], np.float32)
    for (cx, cy), col, strength, radius in theme["glow"]:
        d = np.hypot((xx - cx * W) / (radius * W), (yy - cy * h) / (radius * W * 1.2))
        a = strength * np.exp(-(d**2) * 2.2)
        page = page * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]
    del yy, xx
    img = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))
    if theme["shadow"] is not None:
        sh = Image.new("L", (W, h), 0)
        d = ImageDraw.Draw(sh)
        for x0, y0, x1, y1 in cards:
            d.rounded_rectangle((x0, y0 + 10, x1, y1 + 14), radius=RADIUS, fill=theme["shadow"][3])
        sh = sh.filter(ImageFilter.GaussianBlur(22))
        img = Image.composite(Image.new("RGB", (W, h), theme["shadow"][:3]), img, sh)
    for x0, y0, x1, y1 in cards:
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        w, ch = x1 - x0, y1 - y0
        local = img.crop((x0, y0, x1, y1))
        fill = Image.fromarray(card_fill(theme, w, ch).astype(np.uint8))
        local = Image.composite(fill, local, _rounded_mask(w, ch, RADIUS))
        local = Image.composite(Image.new("RGB", (w, ch), theme["edge"]), local,
                                _rounded_mask(w, ch, RADIUS, outline=1.4))  # fmt: skip
        if theme["highlight"] is not None:  # light catching the top edge of the card
            hl = Image.new("L", (w, ch), 0)
            ImageDraw.Draw(hl).line((RADIUS, 1, w - RADIUS, 1), fill=140, width=1)
            local = Image.composite(Image.new("RGB", (w, ch), theme["highlight"]), local, hl)
        img.paste(local, (x0, y0))
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0.0, theme["grain"], arr.shape[:2]).astype(np.float32)[..., None]  # monochrome film grain
    return np.clip(arr, 0, 255).astype(np.uint8)


def fit_panel(path: Path, box_w: float, box_h: float) -> np.ndarray:
    """The panel scaled (Lanczos) to fit the box, as float32 RGB."""
    im = Image.open(path).convert("RGB")
    s = min(box_w / im.width, box_h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)


def key_white(panel: np.ndarray, under: np.ndarray) -> np.ndarray:
    """Colour-to-alpha against white, composited on what lies under: out = panel + (under - 255) * min(panel) / 255."""
    return np.clip(panel + (under - 255.0) * (panel.min(axis=2, keepdims=True) / 255.0), 0, 255).astype(np.uint8)


def key_black(panel: np.ndarray, under: np.ndarray) -> np.ndarray:
    """Colour-to-alpha against black, composited on the card: out = panel + card * (1 - max(panel) / 255)."""
    alpha = panel.max(axis=2, keepdims=True) / 255.0
    return np.clip(panel + under * (1.0 - alpha), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------ compose


def compose(mode: str, panels: Path, out: Path, fonts: Fonts, seed: int = 7) -> Path:
    """Lay out the cards, panels, text and the Integrates band of one mode and save the PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[mode]
    h = height()
    rng = np.random.default_rng(seed)
    boxes = {}
    for key, _ov, _ti, _call, _pn, (row, c0, c1) in CARDS:
        x, w = span(c0, c1)
        y = row_y(row)
        boxes[key] = (x, y, x + w, y + ROW_H[row])
    ix, iw = span(0, 12)
    iy = row_y(INTEGRATES_ROW)
    boxes["integrates"] = (ix, iy, ix + iw, iy + ROW_H[INTEGRATES_ROW])
    base = base_layer(th, h, list(boxes.values()), rng)

    fig = plt.figure(figsize=(W / 100, h / 100), dpi=100)
    fig.figimage(base, 0, 0, origin="upper", zorder=0)

    def text(x, y, s, key, size, color, **kw):
        return fig.text(x / W, 1 - y / h, s, fontproperties=fonts(key, size), color=color, **kw)

    # ---- header
    text(M, 128, TITLE, "display", 156, th["ink"], va="baseline")
    text(M + 4, 214, TAGLINE, "medium", 42, th["ink"], va="baseline")
    text(M + 4, 270, SUBLINE, "regular", 36, th["ink2"], va="baseline")
    rule = np.linspace(0, 1, 640)
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list("rule", th["rule"])
    bar = (cm(rule)[None, :, :3] * 255).astype(np.uint8).repeat(5, axis=0)
    fig.figimage(bar, M + 4, h - 316, origin="upper", zorder=3)
    # pipeline chips under the rule
    from matplotlib.patches import FancyBboxPatch

    renderer = fig.canvas.get_renderer()
    x, y = M + 4, 372
    for i, step in enumerate(PIPELINE):
        t = text(x + 22, y, step, "medium", 27, th["ink2"], va="center", zorder=5)
        wpx = t.get_window_extent(renderer=renderer).width + 44
        if x + wpx > ART_X - 40:  # chips wrap before the header art
            x, y = M + 4, y + 64
            t.set_position(((x + 22) / W, 1 - y / h))
        fig.patches.append(FancyBboxPatch(((x) / W, 1 - (y + 25) / h), wpx / W, 50 / h,
                                          boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                          fc=np.array(th["chip"]) / 255, ec=np.array(th["chip_edge"]) / 255, lw=1.2,
                                          zorder=4))  # fmt: skip
        x += wpx + 14
        if i < len(PIPELINE) - 1:
            text(x - 2, y, "›", "regular", 30, th["muted"], va="center", ha="left", zorder=5)
            x += 26

    # ---- header art: the CSR / SHM overview of the section, fading into the page on its left
    art = panels / mode / "overview_noboxes.png"
    art = art if art.exists() else panels / mode / "overview.png"
    if art.exists():
        by0, bw_, bh_ = 26, W - M - ART_X, 420
        arr = fit_panel(art, bw_, bh_)
        px, py = int(W - M - arr.shape[1]), int(by0 + (bh_ - arr.shape[0]) / 2)
        under = base[py : py + arr.shape[0], px : px + arr.shape[1]].astype(np.float32)
        arr = (key_black if mode == "dark" else key_white)(arr, under).astype(np.float32)
        fade = np.clip(np.linspace(0.0, 1.0, arr.shape[1]) / 0.12, 0, 1)[None, :, None]
        arr = arr * fade + under * (1 - fade)
        fig.figimage(arr.astype(np.uint8), px, h - py - arr.shape[0], origin="upper", zorder=1)

    # ---- cards
    for key, overline, title, call, names, _pos in CARDS:
        x0, y0, x1, y1 = boxes[key]
        inner = x1 - x0 - 2 * PAD
        text(x0 + PAD, y0 + PAD + 16, overline, "semi", 24, th["muted"], va="baseline")
        _fit_width(text(x0 + PAD, y0 + PAD + 66, title, "display_semi", 38, th["ink"], va="baseline"), inner, renderer)
        _fit_width(text(x0 + PAD, y0 + PAD + 112, call, "mono", 24, th["accent"], va="baseline"), inner, renderer)
        if key == "read":
            _read_card(fig, text, th, x0, y0, x1, y1)
            continue
        if key == "code":
            for k, line in enumerate(CODE):
                text(x0 + PAD, y0 + HEAD + PAD + 12 + 44 * k, line, "mono", 25, th["ink2"], va="baseline")
            continue
        fill = card_fill(th, int(x1 - x0), int(y1 - y0))
        parts = stack(names, inner, panels, mode)
        used = sum(hh for _, hh in parts) + STACK_GAP * max(len(parts) - 1, 0)
        room = y1 - y0 - HEAD - 1.5 * PAD
        py0 = y0 + HEAD + PAD + max(room - used, 0) / 2  # the stack centred in the card
        for name, hh in parts:
            p = panels / mode / f"{name}.png"
            if not p.exists():
                text((x0 + x1) / 2, py0 + hh / 2, f"[{name}]", "regular", 24, th["muted"], ha="center")
                py0 += hh + STACK_GAP
                continue
            arr = fit_panel(p, inner, hh)
            px = int(x0 + PAD + (inner - arr.shape[1]) / 2)
            py = int(py0 + (hh - arr.shape[0]) / 2)
            if mode == "dark":
                under = fill[py - int(y0) : py - int(y0) + arr.shape[0], px - int(x0) : px - int(x0) + arr.shape[1]]
                arr = key_black(arr, under)
            fig.figimage(arr.astype(np.uint8), px, h - py - arr.shape[0], origin="upper", zorder=2)
            py0 += hh + STACK_GAP

    # ---- integrates band
    x0, y0, x1, y1 = boxes["integrates"]
    text(x0 + PAD, y0 + PAD + 16, "13  INTEGRATES", "semi", 24, th["muted"], va="baseline")
    text(x0 + PAD, y0 + PAD + 66, "The single-cell immune-repertoire toolbox, each tool with a spatial counterpart",
         "display_semi", 38, th["ink"], va="baseline")  # fmt: skip
    text(x0 + PAD, y0 + PAD + 112, 'pip install "spaclone[full]"', "mono", 24, th["accent"], va="baseline")
    ncol, gap = 4, 22
    bw = (x1 - x0 - 2 * PAD - (ncol - 1) * gap) / ncol
    bh = 146
    from matplotlib.patches import FancyBboxPatch

    for i, (name, what, extra) in enumerate(INTEGRATES):
        r, c = divmod(i, ncol)
        bx = x0 + PAD + c * (bw + gap)
        by = y0 + HEAD + PAD + r * (bh + gap)
        fig.patches.append(FancyBboxPatch((bx / W, 1 - (by + bh) / h), bw / W, bh / h,
                                          boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
                                          fc=np.array(th["chip"]) / 255, ec=np.array(th["chip_edge"]) / 255, lw=1.3,
                                          zorder=3))  # fmt: skip
        text(bx + 26, by + 50, name, "semi", 34, th["ink"], va="baseline", zorder=4)
        text(bx + bw - 24, by + 48, extra, "mono", 19, th["accent"], va="baseline", ha="right", zorder=4)
        _wrap(text, bx + 26, by + 90, what, bw - 52, "regular", 23, th["ink2"], fig)
    # the twelfth slot: the call that starts it all
    r, c = divmod(len(INTEGRATES), ncol)
    bx = x0 + PAD + c * (bw + gap)
    by = y0 + HEAD + PAD + r * (bh + gap)
    text(bx + 26, by + 50, "spaclone.check_install()", "mono", 24, th["ink"], va="baseline", zorder=4)
    _wrap(text, bx + 26, by + 90, "what works here, and the command for what does not", bw - 52, "regular", 23,
          th["ink2"], fig)  # fmt: skip

    text(W / 2, h - 48, "MIT licence  ·  Python 3.11 – 3.13  ·  github.com/Reynold4k/spaclone", "regular", 23,
         th["muted"], ha="center", va="baseline")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=np.array(th["page"]) / 255)
    plt.close(fig)
    _optimise(out)
    return out


def _fit_width(t, max_px: float, renderer, min_scale: float = 0.72) -> None:
    """Shrink a text until it fits ``max_px`` (down to ``min_scale`` of its size)."""
    size = t.get_fontsize()
    for _ in range(30):
        if t.get_window_extent(renderer=renderer).width <= max_px or t.get_fontsize() <= size * min_scale:
            return
        t.set_fontsize(t.get_fontsize() * 0.96)


def _read_card(fig, text, th, x0, y0, x1, y1):
    """Format chips in two rows: transcriptome sources and receptor tables, detected automatically."""
    from matplotlib.patches import FancyBboxPatch

    h = fig.get_size_inches()[1] * 100
    renderer = fig.canvas.get_renderer()
    y = y0 + HEAD + PAD + 4
    label_w = 290
    for label, items in (("Transcriptome", GEX_FORMATS), ("Receptors", AIR_FORMATS)):
        text(x0 + PAD, y + 30, label, "semi", 27, th["ink"], va="center")
        x = x0 + PAD + label_w
        for item in items:
            t = text(x + 16, y + 30, item, "regular", 23, th["ink2"], va="center", zorder=5)
            wpx = t.get_window_extent(renderer=renderer).width + 32
            if x + wpx > x1 - PAD:
                x, y = x0 + PAD + label_w, y + 64
                t.set_position(((x + 16) / W, 1 - (y + 30) / h))
            fig.patches.append(FancyBboxPatch((x / W, 1 - (y + 55) / h), wpx / W, 50 / h,
                                              boxstyle="round,pad=0,rounding_size=0.005", transform=fig.transFigure,
                                              fc=np.array(th["chip"]) / 255, ec=np.array(th["chip_edge"]) / 255,
                                              lw=1.2, zorder=4))  # fmt: skip
            x += wpx + 12
        y += 64 + 14
    text(x0 + PAD + label_w, y + 26, "Detected from the file contents; spaclone.io.sniff(path) shows what was found.",
         "regular", 23, th["muted"], va="baseline")  # fmt: skip


def _wrap(text, x, y, s, width_px, key, size, color, fig):
    """Greedy word wrap at a pixel width (measured with the renderer)."""
    renderer = fig.canvas.get_renderer()
    words, line, lines = s.split(), "", []
    for wd in words:
        trial = f"{line} {wd}".strip()
        t = text(-1000, -1000, trial, key, size, color)
        wpx = t.get_window_extent(renderer=renderer).width
        t.remove()
        if wpx > width_px and line:
            lines.append(line)
            line = wd
        else:
            line = trial
    lines.append(line)
    for k, ln in enumerate(lines[:2]):
        text(x, y + k * (size * 1.3), ln, key, size, color, va="baseline", zorder=4)


def _optimise(path: Path, limit_mb: float = 5.0) -> None:
    """Re-save with PNG optimisation; quantise to 256 colours only if the file is above the limit."""
    im = Image.open(path).convert("RGB")
    im.save(path, optimize=True)
    if path.stat().st_size > limit_mb * 1e6:
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
            path, optimize=True
        )


# ---------------------------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    """Command line: render the panels (panels.py), then compose both modes."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True, help="folder for results and rendered panels")
    ap.add_argument(
        "--stereoir", type=Path, help="folder with mm_spleen_bin20_anno.h5ad and mm_spleen_bcr_airr.tsv.tsv"
    )
    ap.add_argument("--airspace", type=Path, help="spaclone.datasets cache folder holding the AIR-SPACE D21 files")
    ap.add_argument("--fonts", type=Path, help="folder with the Inter / Inter Display TTF files")
    ap.add_argument("--modes", nargs="+", default=["dark", "light"])
    ap.add_argument("--compose-only", action="store_true", help="use the panels already in CACHE/panels")
    ap.add_argument("--out", type=Path, default=STATIC)
    a = ap.parse_args(argv)
    if not a.compose_only:
        cmd = [sys.executable, "-I", str(HERE / "panels.py"), "--cache", str(a.cache), "--modes", *a.modes]
        if a.stereoir:
            cmd += ["--stereoir", str(a.stereoir)]
        if a.airspace:
            cmd += ["--airspace", str(a.airspace)]
        subprocess.run(cmd, check=True)
    fonts = Fonts(a.fonts)
    fit_rows(a.cache / "panels")
    for mode in a.modes:
        p = compose(mode, a.cache / "panels", a.out / f"overview_{mode}.png", fonts)
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
