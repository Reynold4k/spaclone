"""Draw the README figures, ``docs/_static/homepage/NN_<dataset>_<function>.png``.

Each figure is one spaclone plot function on real data, under one header line that names the function, says what it
does and names the data set:

* AIR-SPACE mouse lymph node (Jiang et al., PNAS 2026; GEO GSE286452), day 21 after vaccinia infection: the day with
  the most heavy-chain loci among the days on which every figure below can be drawn (a CSR flow test at p < 0.05 and at
  least one CSR hotspot); ``--airspace DIR`` holds its two GEO files, else they are downloaded
  (:func:`spaclone.datasets.airspace`);
* StereoIR mouse spleen (Stereo-seq bin20 transcriptome + BCR AIRR contigs from BGI, not yet public; a showcase):
  ``--stereoir DIR`` holds ``mm_spleen_bin20_anno.h5ad`` and ``mm_spleen_bcr_airr.tsv.tsv``.

What is drawn (package defaults; seeds fixed, so a second run draws the same pixels):

* ``pl.hotspot_overview`` and ``pl.csr_map`` of :func:`spaclone.bcr.hotspot_workflow` (AIR-SPACE read with MiXCR's V
  alignments, ``io.read_airspace(..., alignments=True)``, for the mutation map);
* ``pl.csr_flow`` of :func:`spaclone.tl.transition_model` over the regions called from the transcriptome, with the CSR
  potential of :func:`spaclone.bcr.csr_potential` as its time;
* ``pl.region_zoom`` (StereoIR) of the region with the smallest clone-level CSR p (ties: most loci);
* ``pl.lineage_tree`` (StereoIR) of :func:`spaclone.bcr.lineage_workflow`: the drawn tree with >= 3 isotypes and >= 5
  tips in CSR hotspots first, then the most switched tips, tips in hotspots and tips. AIR-SPACE gives no tree: its
  nanopore reads rarely reach the 5' end of V, and the clones whose reads do are almost unmutated.

usage (from the repository root)::

    python -I docs/_static/homepage/make_homepage.py --out docs/_static/homepage --stereoir DIR [--airspace DIR]
        [--fonts DIR] [--cache DIR] [--candidates]

``--fonts`` is a folder with the Inter TTF files (https://rsms.me/inter, SIL OFL 1.1), else DejaVu Sans is used.
``--candidates`` draws every function for both data sets (``cand_<dataset>_<function>.png``) instead of the README
set. ``manifest.csv`` lists what was drawn, the picks and the file sizes.
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

import spaclone as spc

Image.MAX_IMAGE_PIXELS = None
DAY = "D21"
DPI = 240  # matplotlib figures are rasterised at this resolution
MW = 2700  # width of the composed figure before the final resize (px)
RW = 1600  # width of the README files (px)
CW = 1800  # reference width of the header geometry (px)
MAX_BYTES = 1_450_000
BG, INK, INK2, MUTED, ACCENT = (0, 0, 0), (240, 239, 236), (195, 194, 183), (138, 137, 132), (57, 135, 229)
#: The README set, in display order (AIR-SPACE first within each function).
README = [("hotspot_overview", "airspace"), ("hotspot_overview", "stereoir"), ("csr_map", "airspace"),
          ("csr_map", "stereoir"), ("csr_flow", "airspace"), ("csr_flow", "stereoir"), ("region_zoom", "stereoir"),
          ("lineage_tree", "stereoir")]  # fmt: skip
WHAT = {
    "hotspot_overview": "class-switch (CSR) and mutation (SHM) maps with their family-wise hotspots",
    "csr_map": "isotypes along the IgH locus, and the class switches within clones",
    "csr_flow": "sciCSR class-switch potential on the tissue, its transition flow as streamlines",
    "region_zoom": "one hotspot region up close: isotypes, mutated loci, transcriptome identity",
    "lineage_tree": "a germline-rooted lineage tree with every tip on the tissue",
}
LABEL = {"airspace": f"AIR-SPACE lymph node {DAY}", "stereoir": "StereoIR spleen"}

T0 = time.time()
log = logging.getLogger("homepage")


def say(msg: str) -> None:
    """Log a message with the time since the start."""
    log.info(f"[{time.time() - T0:7.1f}s] {msg}")


# ================================================================================================== results
class Results:
    """Results computed once per run (pickled in ``--cache`` when given)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache = Path(args.cache) if args.cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.mem: dict = {}
        self.picks: dict = {}

    def __call__(self, name: str):
        """The result ``name``, computed on first use."""
        if name in self.mem:
            return self.mem[name]
        path = self.cache / f"{name}.pkl" if self.cache and not name.endswith("_sa") else None
        if path is not None and path.exists():
            with open(path, "rb") as fh:
                self.mem[name] = pickle.load(fh)
            say(f"{name}: loaded from the cache")
            return self.mem[name]
        t = time.time()
        self.mem[name] = COMPUTE[name](self)
        say(f"{name}: computed in {time.time() - t:.0f}s")
        if path is not None:
            with open(path, "wb") as fh:
                pickle.dump(self.mem[name], fh, protocol=pickle.HIGHEST_PROTOCOL)
        return self.mem[name]


def c_as_sa(r: Results):
    """AIR-SPACE: bead transcriptome and molecules, with MiXCR's V alignments (for the mutation calls)."""
    if r.args.airspace:
        d = Path(r.args.airspace)
        h5ad = next(d.glob(f"*_adata_{DAY}_with_image_raw_counts.h5ad"))
        clones = next(d.glob(f"*_{DAY}_ovp_clone_table.txt.gz"))
    else:
        h5ad, clones = spc.datasets.airspace(DAY)
    sa = spc.io.read_airspace(h5ad, clones, alignments=True)
    say(f"AIR-SPACE {DAY}: {sa!r}")
    return sa


def c_si_sa(r: Results):
    """StereoIR: bin20 transcriptome and BCR contigs, formats detected by spaclone.read."""
    d = Path(r.args.stereoir)
    sa = spc.read(d / "mm_spleen_bin20_anno.h5ad", d / "mm_spleen_bcr_airr.tsv.tsv")
    say(f"StereoIR: {sa!r}")
    return sa


def _flow(records: pd.DataFrame, region: pd.Series):
    units = spc.bcr.csr_potential(records, unit_key="spot_id", impute="spatial", k=15, species="mouse")
    units["region"] = region.reindex(units.index).fillna("other").to_numpy()
    return spc.tl.transition_model(units, time_key="csr_pot", graph="spatial", k=30, states="region", rng=0)


def c_as_flow(r: Results):
    """CSR potential per bead (spatially imputed) and its transition model over GC / ASC / other beads."""
    sa = r("as_sa")
    g = sa.gex.copy()
    spc.gex.call_regions(g, kind="germinal_centre")
    spc.gex.call_regions(g, kind="plasma_cell")
    g = g[g.obs["in_tissue"].to_numpy(bool)]
    ok = spc.pp.filter_records(sa.air, valid_junction=True, productive=True, on_tissue=True).air
    ok = ok[ok["spot_id"].astype(str).isin(g.obs_names)].reset_index(drop=True)
    return _flow(ok, g.obs["region"].astype(str).replace({"nan": "other"}))


def c_si_flow(r: Results):
    """CSR potential per bin and its transition model over the regions of the bin20 transcriptome."""
    import anndata as ad
    from scipy import sparse

    sa = r("si_sa")
    names = sa.gex.obs_names.astype(str)
    counts = sparse.csr_matrix(sa.gex.layers["counts"] if "counts" in sa.gex.layers else sa.gex.X)
    obs = pd.DataFrame(
        {"cell_type": sa.gex.obs["cell2loc_anno"].astype(str).to_numpy(), "in_tissue": True}, index=names
    )
    g = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=sa.gex.var_names.astype(str)))
    g.obsm["spatial"] = np.asarray(sa.gex.obsm["spatial"], dtype=float)
    for kind in ("germinal_centre", "plasma_cell", "t_zone", "red_pulp", "follicle"):
        spc.gex.call_regions(g, kind=kind)
    ok = spc.pp.filter_records(sa.air, valid_junction=True, productive=True).air
    ok = ok[ok["spot_id"].astype(str).isin(names)].reset_index(drop=True)
    return _flow(ok, g.obs["region"].astype(str).replace({"nan": "other"}))


COMPUTE = {
    "as_sa": c_as_sa,
    "si_sa": c_si_sa,
    "as_hot": lambda r: spc.bcr.hotspot_workflow(r("as_sa")),
    "si_hot": lambda r: spc.bcr.hotspot_workflow(r("si_sa"), rng=0),
    "as_flow": c_as_flow,
    "si_flow": c_si_flow,
    "si_lin": lambda r: spc.bcr.lineage_workflow(r("si_hot"), r("si_sa").gex, rng=0, n_jobs=r.args.n_jobs),
}


# ================================================================================================== figures
def _hot(r: Results, ds: str):
    return r("as_hot") if ds == "airspace" else r("si_hot")


def pick_region(hot) -> int:
    """The region with the smallest clone-level CSR p (ties: most loci, then the lowest id)."""
    reg = hot.regions.sort_values(["csr_clone_p", "n_loci", "region_id"], ascending=[True, False, True])
    return reg["region_id"].iloc[0]


def pick_tree(lin) -> pd.Series:
    """Drawn tree with >= 3 isotypes and >= 5 tips in CSR hotspots first, then by switched, hotspot and all tips."""
    rows = []
    for g in lin.groupings:
        sel = lin.selection[lin.selection["grouping"] == g]
        drawn = set(sel.loc[sel["drawn"].astype(bool), "tree_id"].astype(str))
        for tid, t in lin.tips[g].groupby("tree_id"):
            if str(tid) in drawn:
                iso = t["isotype"].astype(str)
                known = iso[~iso.isin(["", "nan", "None", "unknown"])]
                rows.append({"grouping": g, "tree_id": tid, "n_tips": len(t), "n_isotypes": known.nunique(),
                             "n_switched_tips": int((~known.isin(["IgM", "IgD"])).sum()),
                             "n_hotspot_tips": int(t["location"].astype(str).eq("CSR hotspot").sum())})  # fmt: skip
    cand = pd.DataFrame(rows)
    cand["qualifies"] = (cand["n_isotypes"] >= 3) & (cand["n_hotspot_tips"] >= 5)
    keys = ["qualifies", "n_switched_tips", "n_hotspot_tips", "n_tips"]
    return cand.sort_values(keys, ascending=False, ignore_index=True).iloc[0]


def f_hotspot_overview(r: Results, ds: str):
    """Whole-section CSR and SHM maps with hotspots."""
    return spc.pl.hotspot_overview(_hot(r, ds))


def f_csr_map(r: Results, ds: str):
    """Isotype maps along the IgH locus and the switches within clones (AIR-SPACE: a shorter frame for the round node)."""
    return spc.pl.csr_map(_hot(r, ds), figsize=(16.0, 7.2) if ds == "airspace" else (16.0, 9.0))


def f_csr_flow(r: Results, ds: str):
    """CSR potential on the tissue with the transition flow (AIR-SPACE: a square frame, larger dots)."""
    if ds == "stereoir":
        return spc.pl.csr_flow(r("si_flow"), label="CSR potential")
    fig, ax = plt.subplots(figsize=(10, 10))
    spc.pl.csr_flow(r("as_flow"), label="CSR potential", ax=ax, s=9.0, density=1.0)
    return fig


def f_region_zoom(r: Results, ds: str):
    """One region up close."""
    hot = _hot(r, ds)
    rid = pick_region(hot)
    r.picks[(ds, "region_zoom")] = f"region {rid}"
    return spc.pl.region_zoom(hot, rid)


def f_lineage_tree(r: Results, ds: str):
    """One lineage tree beside the section (StereoIR)."""
    if ds != "stereoir":
        raise ValueError("AIR-SPACE has no lineage tree (see the module notes)")
    lin = r("si_lin")
    best = pick_tree(lin)
    r.picks[(ds, "lineage_tree")] = f"{best.grouping} {best.tree_id} ({best.n_tips} tips)"
    return spc.pl.lineage_tree(lin, best.tree_id, grouping=best.grouping)


DRAW = {"hotspot_overview": f_hotspot_overview, "csr_map": f_csr_map, "csr_flow": f_csr_flow,
        "region_zoom": f_region_zoom, "lineage_tree": f_lineage_tree}  # fmt: skip


# ================================================================================================== header band
def figure_of(obj) -> Figure:
    """The Figure of a Figure, an Axes or an array of Axes."""
    if isinstance(obj, Figure):
        return obj
    if isinstance(obj, (list, tuple, np.ndarray)):
        return figure_of(np.ravel(obj)[0])
    return obj.figure


def raster(obj) -> Image.Image:
    """The figure as an RGB image at DPI (tight bounding box, its own surface colour)."""
    fig = figure_of(obj)
    face = fig.get_facecolor()
    face = (0, 0, 0, 1) if face[3] == 0 else face
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.18, facecolor=face)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fit_width(im: Image.Image, w: int) -> Image.Image:
    """Resize to width ``w`` (Lanczos), keeping the aspect."""
    return im if im.width == w else im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)


class Fonts:
    """Inter from ``--fonts`` when present, else matplotlib's DejaVu Sans; DejaVu Sans Mono Bold for the function."""

    def __init__(self, folder: str | None):
        self.dir = Path(folder) if folder else None
        self.ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

    def text(self, size: float) -> ImageFont.FreeTypeFont:
        """Regular text face."""
        hits = sorted(self.dir.rglob("Inter-Regular.ttf")) if self.dir else []
        return ImageFont.truetype(str(hits[0] if hits else self.ttf / "DejaVuSans.ttf"), max(6, round(size)))

    def mono(self, size: float) -> ImageFont.FreeTypeFont:
        """Monospaced bold face."""
        return ImageFont.truetype(str(self.ttf / "DejaVuSansMono-Bold.ttf"), max(6, round(size)))


def with_header(fig_img: Image.Image, func: str, what: str, fonts: Fonts, width: int = MW) -> Image.Image:
    """The figure scaled to ``width`` under one header line: the function (bold mono) and what it does."""
    s = width / CW
    fig_img = fit_width(fig_img, width)
    pad, bar_x, text_x = round(44 * s), round(44 * s), round(66 * s)
    size_f, size_d = 30.0, 27.0
    dash = "  —  "
    while True:
        ff, fd = fonts.mono(size_f * s), fonts.text(size_d * s)
        wf, wdash, wd = ff.getlength(func), fd.getlength(dash), fd.getlength(what)
        if wf + wdash + wd <= width - text_x - pad or size_d < 16:
            break
        size_f, size_d = size_f * 0.96, size_d * 0.96
    top, band, below, bottom = round(42 * s), round(size_f * s * 1.5), round(26 * s), round(40 * s)
    canvas = Image.new("RGB", (width, top + band + below + fig_img.height + bottom), BG)
    d = ImageDraw.Draw(canvas)
    cy = top + band / 2
    d.rectangle([bar_x, round(cy - band * 0.42), bar_x + round(6 * s), round(cy + band * 0.42)], fill=ACCENT)
    d.text((text_x, cy), func, font=ff, fill=INK, anchor="lm")
    d.text((text_x + wf, cy), dash, font=fd, fill=MUTED, anchor="lm")
    d.text((text_x + wf + wdash, cy), what, font=fd, fill=INK2, anchor="lm")
    canvas.paste(fig_img, (0, top + band + below))
    return canvas


def save_capped(img: Image.Image, path: Path) -> str:
    """PNG of at most MAX_BYTES, always RGB: full depth when it fits, else each channel rounded to 6, 5 or 4 bits."""
    img.save(path, optimize=True)
    if path.stat().st_size <= MAX_BYTES:
        return "RGB"
    for bits in (6, 5, 4):
        q = 1 << (8 - bits)
        lut = [min(255, (v + q // 2) // q * q) for v in range(256)]
        img.point(lut * 3).save(path, optimize=True)
        if path.stat().st_size <= MAX_BYTES:
            return f"RGB, {bits} bits per channel"
    return "RGB, over the size cap"


# ================================================================================================== main
def main() -> None:
    """Draw the README set (or every candidate) into ``--out``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stereoir", required=True, help="folder with the StereoIR h5ad and BCR AIRR files")
    ap.add_argument("--airspace", default=None, help="folder with the AIR-SPACE GEO files (default: download)")
    ap.add_argument("--fonts", default=None, help="folder with the Inter TTF files")
    ap.add_argument("--cache", default=None, help="folder for result pickles (default: none)")
    ap.add_argument("--candidates", action="store_true", help="draw every function for both data sets")
    ap.add_argument("--only", nargs="+", default=[], help="file stems to draw, e.g. airspace_csr_map")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("matplotlib", "PIL", "numba", "fontTools", "h5py", "anndata"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fonts = Fonts(args.fonts)
    say(f"spaclone {spc.__version__}; out {out}")
    if args.candidates:
        todo = [(f"cand_{ds}_{fn}", fn, ds) for fn in DRAW for ds in ("airspace", "stereoir")]
    else:
        todo = [(f"{k:02d}_{ds}_{fn}", fn, ds) for k, (fn, ds) in enumerate(README, 1)]
    r = Results(args)
    rows = []
    for stem, fn, ds in todo:
        if args.only and not any(o in stem for o in args.only):
            continue
        t = time.time()
        row = {"file": f"{stem}.png", "function": f"spaclone.pl.{fn}", "dataset": LABEL[ds]}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = raster(DRAW[fn](r, ds))
            master = with_header(img, f"spaclone.pl.{fn}", f"{WHAT[fn]}  ·  {LABEL[ds]}", fonts)
            how = save_capped(fit_width(master, RW), out / f"{stem}.png")
            with Image.open(out / f"{stem}.png") as im:
                size = im.size
            row.update({"pick": r.picks.get((ds, fn), ""), "width": size[0], "height": size[1],
                        "bytes": (out / f"{stem}.png").stat().st_size, "storage": how, "status": "ok"})  # fmt: skip
            say(f"{stem}: {size[0]} x {size[1]}, {row['bytes'] / 1e6:.2f} MB ({how}) in {time.time() - t:.0f}s")
        except Exception:  # noqa: BLE001  (one failed figure does not stop the others)
            row["status"] = "failed"
            say(f"{stem} FAILED\n{traceback.format_exc()}")
        finally:
            plt.close("all")
        rows.append(row)
    man = pd.DataFrame(rows)
    path = out / "manifest.csv"
    if args.only and path.exists():
        old = pd.read_csv(path)
        man = pd.concat([old[~old["file"].isin(man["file"])], man], ignore_index=True)
    man.to_csv(path, index=False)
    say("done")


if __name__ == "__main__":
    main()
