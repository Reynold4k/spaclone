"""Real-data panels for the spaclone README hero image.

Two public data sets, package functions only:

* StereoIR mouse spleen (Stereo-seq bin20 transcriptome + 5' BCR records); ``--stereoir DIR`` holds
  ``mm_spleen_bin20_anno.h5ad`` and ``mm_spleen_bcr_airr.tsv.tsv``;
* AIR-SPACE mouse lymph node, day 21 (GEO GSE286452); ``--airspace DIR`` holds the two GEO files of D21
  (:func:`spaclone.datasets.load_airspace` with ``download=False``).

Stage 1 computes every result once and pickles it in ``CACHE/results/`` (an existing pickle is reused unless
``--recompute`` names it); stage 2 draws every panel in the dark and the light style into
``CACHE/panels/{dark,light}/<panel>.png`` (dpi 200, tight bounding box). Seeds are fixed, so a re-run draws the same
pictures.

Usage::

    python -I panels.py --cache DIR --stereoir DIR --airspace DIR [--only PANEL ...] [--modes dark light]
                        [--recompute NAME ...] [--stage all|compute|render] [--tree GROUPING:ID]
                        [--n-clones N] [--max-link-um UM] [--tpt-random N] [--n-jobs N]

Panels: records, trust, atlas, overview, overview_noboxes, csr_map, lineage, clone_states, clone_map, composition,
csr_flow, flux, placebo. Results: records, hotspot, lineage, threadfin, csr_flow, airspace (``--recompute`` takes
either kind of name; a panel name recomputes the results it draws).
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import sparse

import spaclone as sc

BCR_LOCI = ["IGH", "IGK", "IGL"]
TYPES = {"T": ["T_cells"], "B": ["B_cells"], "erythroid": ["Erythrocyte"]}  # StereoIR cell2location labels
REGION_ORDER = ["follicle", "GC", "ASC", "T zone", "red pulp", "other"]
PANELS = {  # panel -> results it draws
    "records": ("records",),
    "trust": ("trust",),
    "atlas": ("hotspot",),
    "overview": ("hotspot",),
    "overview_noboxes": ("hotspot",),
    "csr_map": ("hotspot",),
    "lineage": ("lineage",),
    "clone_states": ("threadfin",),
    "clone_map": ("airspace",),
    "composition": ("hotspot",),
    "csr_flow": ("csr_flow",),
    "flux": ("csr_flow",),
    "placebo": ("airspace",),
}
RESULTS = ("records", "hotspot", "lineage", "threadfin", "csr_flow", "airspace", "trust")
T0 = time.time()


def log(msg):
    """Print a time-stamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================================================== stage 1: results
class Cache:
    """Results computed once, pickled in ``DIR/results/`` and held in memory for the drawing stage."""

    def __init__(self, args):
        self.dir = Path(args.cache) / "results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.recompute = set(args.recompute)
        self.mem = {}
        self.timing = []
        self._sa = None

    def stereoir(self):
        """StereoIR section read once per run with the package reader (format auto-detected)."""
        if self._sa is None:
            t = time.time()
            d = Path(self.args.stereoir)
            self._sa = sc.read(d / "mm_spleen_bin20_anno.h5ad", d / "mm_spleen_bcr_airr.tsv.tsv")
            span = np.ptp(np.asarray(self._sa.gex.obsm["spatial"], dtype=float), axis=0)
            log(f"StereoIR read in {time.time() - t:.0f}s: {self._sa.gex.n_obs:,} bins, {len(self._sa.air):,} records, "
                f"section {span[0]:.0f} x {span[1]:.0f} um, layers {list(self._sa.gex.layers)}")  # fmt: skip
            self.timing.append({"step": "StereoIR read", "seconds": round(time.time() - t, 1)})
        return self._sa

    def get(self, name):
        """Return a result: from memory, from its pickle, or computed (and pickled) now."""
        if name in self.mem:
            return self.mem[name]
        path = self.dir / f"{name}.pkl"
        t = time.time()
        if path.exists() and name not in self.recompute:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            log(f"{name}: loaded {path.name} in {time.time() - t:.0f}s")
        else:
            log(f"{name}: computing")
            obj = COMPUTE[name](self)
            dt = time.time() - t
            with open(path, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self.recompute.discard(name)
            self.timing.append({"step": f"compute {name}", "seconds": round(dt, 1)})
            log(f"{name}: computed in {dt:.0f}s -> {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        self.mem[name] = obj
        return obj


def _valid(air, min_reads=2):
    """Valid, productive BCR records with at least ``min_reads`` reads."""
    ok = sc.pp.filter_valid(air, require_productive=True, min_reads=min_reads).air
    return ok[ok["locus"].isin(BCR_LOCI)].reset_index(drop=True)


def compute_records(c):
    """Valid productive StereoIR BCR records (>= 2 reads) on the hotspot workflow's tissue mask, and its bins."""
    sa, wf = c.stereoir(), c.get("hotspot")
    names = sa.gex.obs_names.astype(str)
    xy = np.asarray(sa.gex.obsm["spatial"], dtype=float)
    on_mask = np.nan_to_num(np.asarray(wf.tissue.sample(xy[:, 0], xy[:, 1], fill=0), dtype=float)) > 0
    tissue = on_mask & wf.units["tissue"].reindex(names).fillna(False).to_numpy(dtype=bool)  # no debris specks
    air = _valid(sa.air)
    air = air.loc[air["spot_id"].astype(str).isin(names[tissue]), ["spot_id", "x_um", "y_um", "locus"]]
    bg = ad.AnnData(obs=pd.DataFrame(index=names[tissue]))
    bg.obsm["spatial"] = np.asarray(sa.gex.obsm["spatial"], dtype=float)[tissue]
    log(f"records: {len(air):,} valid productive BCR records on {tissue.sum():,} tissue bins "
        f"({(~tissue).sum():,} bins off the tissue mask); {air.locus.value_counts().to_dict()}")  # fmt: skip
    return {"air": air.reset_index(drop=True), "gex": bg}


def compute_hotspot(c):
    """CSR / SHM hotspot workflow on StereoIR with the package defaults."""
    return sc.bcr.hotspot_workflow(c.stereoir(), rng=0)


def tree_candidates(lin):
    """Drawn trees ranked by the pre-stated rule.

    Trees with >= 3 isotypes and >= 5 tips in CSR hotspots come first, ranked by switched tips (IgG / IgA / IgE),
    then tips in hotspots, then tips; the other trees follow in the same order.
    """
    rows = []
    for g in lin.groupings:
        sel = lin.selection[lin.selection["grouping"] == g]
        drawn = set(sel.loc[sel["drawn"].astype(bool), "tree_id"].astype(str))
        for tid, t in lin.tips[g].groupby("tree_id"):
            if str(tid) not in drawn:
                continue
            iso = t["isotype"].astype(str)
            known = iso[~iso.isin(["", "nan", "None", "unknown"])]
            rows.append({"grouping": g, "tree_id": str(tid), "n_tips": len(t), "n_isotypes": known.nunique(),
                         "isotypes": ";".join(sorted(known.unique())),
                         "n_switched_tips": int((~known.isin(["IgM", "IgD"])).sum()),
                         "n_hotspot_tips": int(t["location"].astype(str).eq("CSR hotspot").sum()),
                         "max_mutations": float(t["mutations"].max())})  # fmt: skip
    cand = pd.DataFrame(rows)
    cand["qualifies"] = (cand["n_isotypes"] >= 3) & (cand["n_hotspot_tips"] >= 5)
    keys = ["qualifies", "n_switched_tips", "n_hotspot_tips", "n_tips"]
    return cand.sort_values(keys, ascending=False, ignore_index=True)


def compute_lineage(c):
    """Lineage trees (exact clones and related groups) on top of the hotspot workflow."""
    lin = sc.bcr.lineage_workflow(c.get("hotspot"), c.stereoir().gex, rng=0, n_jobs=c.args.n_jobs)
    cand = tree_candidates(lin)
    cand.to_csv(c.dir / "lineage_candidates.csv", index=False)
    log(f"lineage: {lin!r}; best trees\n{cand.head(8).to_string()}")
    return lin


def compute_threadfin(c):
    """Threadfin clone states of StereoIR IGH clones (spot units, expression features)."""
    from spaclone.ext import threadfin as tfx

    sa, wf = c.stereoir(), c.get("hotspot")
    heavy = _valid(sa.air)
    heavy = sc.pp.define_clones(heavy[heavy["locus"] == "IGH"].reset_index(drop=True), method="exact", by=())
    g = sa.gex.copy()
    g.obs["donor"], g.obs["sample"] = "mm_spleen", "StereoIR_bin20"
    extra = g.obs["cell2loc_anno"].astype(str).eq("B_cells").to_numpy()  # B-context bins, as Threadfin allows
    lab = wf.region_labels
    regions = None
    if lab is not None and np.any(np.asarray(getattr(lab, "values", lab)) > 0):
        ras = lab if isinstance(lab, sc.tl.Raster) else sc.tl.Raster(np.asarray(lab), wf.grid, name="CSR/SHM regions")
        regions = {"CSR/SHM regions": ras}
    ctx = tfx.spatial_context(g, cell_type_key="cell2loc_anno", types=TYPES, regions=regions)
    res = tfx.run((heavy, g), units="spot", features="expression", clone_key="clone_id", donor_key="donor",
                  sample_key="sample", extra_spots=extra, context=ctx, threadfin_kwargs={"n_jobs": c.args.n_jobs},
                  rng=0)  # fmt: skip
    log(f"threadfin: {res!r}; unit columns {list(res.units.columns)}")
    return dataclasses.replace(res, threadfin=None)  # tables, axes and context only (the drawing needs no more)


def compute_csr_flow(c):
    """CSR potential of BCR bins, spatial transition model over GEX regions and TPT flux."""
    sa = c.stereoir()
    counts = sa.gex.layers["counts"] if "counts" in sa.gex.layers else sa.gex.X
    g = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"cell_type": sa.gex.obs["cell2loc_anno"].astype(str).to_numpy(), "in_tissue": True},
                         index=sa.gex.obs_names.astype(str)),
        var=pd.DataFrame(index=sa.gex.var_names.astype(str)),
    )  # fmt: skip
    g.obsm["spatial"] = np.asarray(sa.gex.obsm["spatial"], dtype=float)
    for kind in ("germinal_centre", "plasma_cell", "t_zone", "red_pulp", "follicle"):
        sc.gex.call_regions(g, kind=kind)
    region = g.obs["region"].astype(str).replace({"nan": "other"})
    ok = _valid(sa.air)
    ok = ok[ok["spot_id"].astype(str).isin(g.obs_names)].reset_index(drop=True)
    units = sc.bcr.csr_potential(ok, unit_key="spot_id", impute="spatial", k=15, species="mouse")
    units["region"] = region.reindex(units.index).fillna("other").to_numpy()
    model = sc.tl.transition_model(units, time_key="csr_pot", graph="spatial", k=30, states="region")
    order = [s for s in REGION_ORDER if s in set(model.states.categories)]
    target = "red pulp" if "red pulp" in order else "ASC"
    t = time.time()
    flux = sc.tl.tpt_flux(model, source="follicle", target=target, null="time_within_states",
                          n_random=c.args.tpt_random, rng=1)  # fmt: skip
    log(f"csr_flow: {model.n_units:,} units, regions {units.region.value_counts().to_dict()}; TPT follicle -> "
        f"{target} with {c.args.tpt_random} null draws in {time.time() - t:.0f}s")  # fmt: skip
    return {"model": model, "flux": flux, "order": order, "target": target,
            "span_um": np.ptp(g.obsm["spatial"], axis=0)}  # fmt: skip


def compute_airspace(c):
    """AIR-SPACE D21: trust report, filtered on-tissue records and the co-localisation test."""
    t = time.time()
    sp = sc.datasets.load_airspace("D21", data_dir=c.args.airspace, download=False)
    log(f"AIR-SPACE D21 loaded in {time.time() - t:.0f}s: {sp!r}")
    sc.gex.score_cell_types(sp.gex, markers="immune_mouse")  # bead cell types: lets run() test own_type_share
    t = time.time()
    rep = sc.run(sp, species="mouse", rng=0)
    log(f"AIR-SPACE D21 run in {time.time() - t:.0f}s: {rep.verdict}\n{rep.tests().to_string()}")
    filt = sc.pp.filter_records(sp.air, valid_junction=True, productive=True, on_tissue=True).air
    keep = [k for k in ("spot_id", "x_um", "y_um", "locus", "clone_id", "consensus_count") if k in filt]
    coloc = rep.results.get("colocalisation") if isinstance(rep.results, dict) else None
    if coloc is None:
        coloc = sc.qc.colocalisation(filt, n_perm=200, rng=0)
    coloc.table.to_csv(c.dir / "airspace_colocalisation.csv", index=False)
    return {"report": rep, "coloc": coloc, "records": filt[keep].reset_index(drop=True)}


def compute_trust(c):
    """StereoIR trust report with the defaults of spaclone.run (cell types detected when unambiguous)."""
    t = time.time()
    rep = sc.run(c.stereoir(), species="mouse", rng=0)
    log(f"StereoIR run in {time.time() - t:.0f}s: {rep.verdict}\n{rep.tests().to_string()}")
    return {"report": rep}


COMPUTE = {"records": compute_records, "hotspot": compute_hotspot, "lineage": compute_lineage,
           "threadfin": compute_threadfin, "csr_flow": compute_csr_flow, "airspace": compute_airspace,
           "trust": compute_trust}  # fmt: skip


# ============================================================================================== stage 2: panels
def _axes(figsize):
    """A new figure with one Axes."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _surface(fig, ax, style):
    """A function drawing into a supplied Axes may leave the figure surface alone: give it the Axes surface."""
    face = ax.get_facecolor()
    fig.set_facecolor(face if face[3] > 0 else ("black" if style == "dark" else "white"))
    return fig


def p_records(c, style):
    """StereoIR BCR records on the tissue, coloured by chain."""
    d = c.get("records")
    fig, ax = _axes((12.0, 4.6))  # every map turns the section's long axis horizontal
    sc.pl.spatial_records(d["air"], d["gex"], color="locus", style=style, ax=ax, background="none", size=1.2)
    return _surface(fig, ax, style)


def p_trust(c, style):
    """Card-sized trust report of StereoIR (both levels tested with run()'s defaults)."""
    return sc.pl.trust_report(c.get("trust")["report"], style=style, compact=True, width=7.0)


def p_atlas(c, style):
    """Hotspot atlas of StereoIR (flagship)."""
    return sc.pl.hotspot_atlas(c.get("hotspot"), style=style, clone_p=False)


def p_overview(c, style):
    """Whole-section CSR and SHM maps with the region windows (no legend row)."""
    return sc.pl.hotspot_overview(c.get("hotspot"), style=style, legend=False)


def p_overview_noboxes(c, style):
    """Whole-section CSR and SHM maps without region windows or legend row."""
    return sc.pl.hotspot_overview(c.get("hotspot"), style=style, legend=False, boxes=False)


def p_csr_map(c, style):
    """Class-switch map of StereoIR."""
    return sc.pl.csr_map(c.get("hotspot"), style=style, notes=False)


def p_lineage(c, style):
    """The lineage tree chosen by ``--tree`` or by :func:`tree_candidates`."""
    lin = c.get("lineage")
    if c.args.tree:
        grouping, tree_id = c.args.tree.split(":", 1)
    else:
        best = tree_candidates(lin).iloc[0]
        grouping, tree_id = best.grouping, best.tree_id
    log(f"lineage tree: {grouping} {tree_id}")
    return sc.pl.lineage_tree(lin, tree_id, grouping=grouping, style=style)


def _end_labels(res):
    """Name the clone-axis-1 ends by where their units sit (mean in red pulp vs B-rich / follicle), else None."""
    u = res.units
    if "compartment" not in u or "clone_axis_1" not in u:
        return None
    m = u.groupby(u["compartment"].astype(str).str.lower())["clone_axis_1"].mean().dropna()
    red = m[m.index.str.contains("red pulp")]
    white = m[m.index.str.contains("follicle|white pulp|b-rich|b rich")]
    log(f"clone axis 1 mean by compartment: {m.round(4).to_dict()}")
    if red.empty or white.empty:
        return None
    low_red = red.mean() < white.mean()
    return ("red-pulp end", "white-pulp end") if low_red else ("white-pulp end", "red-pulp end")


def p_clone_states(c, style):
    """Threadfin clone axis 1 of StereoIR clones on the tissue."""
    res = c.get("threadfin")
    fig, ax = _axes((9.0, 3.9))  # small inches, so the legend and colour bar read at the card's width
    sc.pl.clone_state_map(res, color="clone_axis", style=style, ax=ax, end_labels=_end_labels(res), scalebar_um=None)
    return _surface(fig, ax, style)


def p_clone_map(c, style):
    """The largest AIR-SPACE D21 IGH clones on the node, with their spanning-tree links."""
    d = c.get("airspace")["records"]
    fig, ax = _axes((6.0, 6.0))
    sc.pl.clone_map(d[d["locus"] == "IGH"], n_top=c.args.n_clones, links=True, max_link_um=c.args.max_link_um,
                    background_size=2.0, style=style, ax=ax, scalebar_um=500)  # fmt: skip
    return _surface(fig, ax, style)


def p_composition(c, style):
    """Isotype composition of StereoIR by region."""
    out = sc.pl.isotype_composition(c.get("hotspot"), by="region", style=style)
    return out if isinstance(out, Figure) else out.figure


def p_csr_flow(c, style):
    """Class-switch potential and its expected displacement as streamlines on the tissue."""
    d = c.get("csr_flow")
    ax = sc.pl.csr_flow(d["model"], kind="stream", style=style, s=2.0, density=1.0,
                        grid_um=float(max(d["span_um"])) / 50, label="CSR potential")  # fmt: skip
    return _surface(ax.figure, ax, style)


def p_flux(c, style):
    """Net TPT flux between tissue regions (percent of the total)."""
    d = c.get("csr_flow")
    fig, ax = _axes((5.0, 5.0))
    sc.pl.flux_matrix(d["flux"], value="net", order=d["order"], style=style, ax=ax)
    return _surface(fig, ax, style)


def p_placebo(c, style):
    """AIR-SPACE D21 heavy-chain co-localisation: observed share against its placebo interval."""
    res = c.get("airspace")["coloc"]
    tab = res.table
    query = "design == 'stratified'" if "design" in tab and tab["design"].eq("stratified").any() else None
    x = "locus" if "locus" in tab and tab["locus"].nunique() > 1 else None
    hue = "band" if "band" in tab and tab["band"].nunique() > 1 else None
    fig, ax = _axes((6.0, 4.5))
    sc.pl.placement(res, x=x, hue=hue, query=query, style=style, ax=ax)
    return _surface(fig, ax, style)


DRAW = {name: globals()[f"p_{name}"] for name in PANELS}


def main():
    """Command line: compute the results the chosen panels need, then draw them."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="cache folder (results/, panels/, runtime.csv)")
    ap.add_argument("--stereoir", required=True, help="folder with the two StereoIR files")
    ap.add_argument("--airspace", required=True, help="folder with the AIR-SPACE GSE286452 D21 files")
    ap.add_argument("--only", nargs="+", default=list(PANELS), choices=list(PANELS), metavar="PANEL")
    ap.add_argument("--modes", nargs="+", default=["dark", "light"], choices=["dark", "light"])
    ap.add_argument("--recompute", nargs="+", default=[], choices=list(PANELS) + list(RESULTS), metavar="NAME")
    ap.add_argument("--stage", default="all", choices=["all", "compute", "render"])
    ap.add_argument("--tree", default=None, help="lineage tree as GROUPING:ID (default: the rule in tree_candidates)")
    ap.add_argument("--n-clones", type=int, default=3, help="largest AIR-SPACE IGH clones drawn on the clone map")
    ap.add_argument("--max-link-um", type=float, default=250.0, help="longest clone-map link drawn (um)")
    ap.add_argument("--tpt-random", type=int, default=0, help="TPT null draws (0: flux without a test)")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    args.recompute = sorted({r for n in args.recompute for r in (PANELS.get(n) or (n,))})
    cache = Cache(args)
    log(f"spaclone {sc.__version__}; panels {args.only}; recompute {args.recompute}")
    needed = [r for r in RESULTS if any(r in PANELS[p] for p in args.only)]
    for name in needed:
        cache.get(name)
    if args.stage != "compute":
        for style in args.modes:
            out = Path(args.cache) / "panels" / style
            out.mkdir(parents=True, exist_ok=True)
            for name in args.only:
                t = time.time()
                fig = DRAW[name](cache, style)
                fig.savefig(out / f"{name}.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight",
                            pad_inches=0.05)  # fmt: skip
                plt.close("all")
                cache.timing.append({"step": f"draw {style} {name}", "seconds": round(time.time() - t, 1)})
                log(f"drew {style}/{name}.png in {time.time() - t:.0f}s")
    rt = Path(args.cache) / "runtime.csv"
    pd.DataFrame(cache.timing).to_csv(rt, index=False, mode="a", header=not rt.exists())
    log("done")


if __name__ == "__main__":
    main()
