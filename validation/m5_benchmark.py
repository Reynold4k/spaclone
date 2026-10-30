"""M5 validation: the simulators' planted truth, the legacy demo, and the benchmark harness.

1. ``sim.legacy_demo_chip(rng=0)`` against the archived ``spaclone demo`` input (03 results/E2_spaclone_run/
   synthetic_demo/input): meta table, GEF, tissue mask and truth.json identical.
2. The legacy demo numbers that do not need the same-spot test: ``io.read_stereo_xcr`` on the written meta table gives
   the archived units per chain (IGL 3,000, IGK 8,507, IGH 4,214), reads, ``umi == cid``, the unit table row by row
   (output/synthetic.units.parquet), the four barcode sinks (two on the tissue, at most 108 clones on one DNB) and the
   share of units off the tissue per chain; the molecule truth of the port (genuine IGL, the hot clone, sinks).
3. ``tl.clonal_scan`` (M4) with the legacy settings on the same units (tissue, sinks removed): clones tested and
   called per chain (README: only IGK "9999", q = 0.005). The same-spot numbers of the legacy table (p_det 0.79, IGL
   located 95.9%, IGK 1.0%, IGH 0%) need M1's ``qc.same_spot``, which is not in the tree yet: reported as not run.
4. ``sim.synthetic_chip(rng=0, legacy=True)`` (the 1.0.0.dev0 chip) and the realistic default: truth tables (genuine
   share, positions, sinks, one heavy + one light clone per cell, structures, own-transcript capture at ``p_det``);
   for the realistic chip also swaps attracted to receptor-rich cells, constant-region RNA confined to its cells and
   bursty (F1, review M8).
5. ``sim.plant`` on a real background (AIR-SPACE D21 BCR molecules, placebo by clone permutation within chain, planted
   into germinal-centre-like foci on tissue beads): realised fraction, focus geometry, read counts of planted vs
   background records (indistinguishable), and ``sim.score`` on a perfect and a random caller.
Output: validation/results/m5_benchmark.csv (quantity, archived, package, match, note)
Usage : python -I m5_benchmark.py [n_jobs]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import spaclone
from scipy import stats
from scipy.spatial import cKDTree
from spaclone import io, sim

DEMO = Path(os.environ["SPACLONE_DEMO"])  # archived run of the synthetic demo (results/E2_spaclone_run/synthetic_demo)
E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
T0 = time.time()
rows: list[dict] = []


def log(msg):
    """Print a time-stamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0, match=None):
    """Record one archived-vs-package (or truth-vs-package) comparison."""
    if match is None:
        a, p = float(archived), float(package)
        match = abs(a - p) <= tol * max(1.0, abs(a))
    rows.append(
        {
            "quantity": quantity,
            "archived": archived,
            "package": package,
            "match": bool(match),
            "note": note,
        }
    )
    log(
        f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}"
    )


def save():
    """Write the comparison table so far."""
    R = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
    R.to_csv(OUT / "m5_benchmark.csv", index=False)
    return R


def main():
    """Run every comparison and write the table."""
    log(f"spaclone {spaclone.__version__}")
    tmp = Path(tempfile.mkdtemp(prefix="m5_benchmark_"))

    # ------------------------------------------------------------------ 1 legacy port == archived demo input
    leg = sim.legacy_demo_chip(rng=0, write_dir=tmp / "legacy")
    f = leg.meta["files"]
    A = DEMO / "input"
    add("legacy demo meta table identical (1 = yes)", 1,
        int(pd.read_parquet(A / "synthetic.meta.parquet").equals(pd.read_parquet(f["meta"]))), "synthetic.meta.parquet")  # fmt: skip
    with h5py.File(A / "synthetic.raw.gef") as h0, h5py.File(f["raw_gef"]) as h1:
        same = all(
            np.array_equal(h0[f"geneExp/bin1/{k}"][:], h1[f"geneExp/bin1/{k}"][:])
            for k in ("expression", "gene")
        )
        same &= all(
            np.array_equal(
                h0["geneExp/bin1/expression"].attrs[k],
                h1["geneExp/bin1/expression"].attrs[k],
            )
            for k in ("minX", "minY", "maxX", "maxY")
        )
    add(
        "legacy demo GEF identical (1 = yes)",
        1,
        int(same),
        "synthetic.raw.gef: expression, gene table, extent",
    )
    try:
        import tifffile

        add("legacy demo tissue mask identical (1 = yes)", 1,
            int(np.array_equal(tifffile.imread(A / "synthetic_tissue_cut.tif"), tifffile.imread(f["tissue_mask"]))))  # fmt: skip
    except ImportError:
        log("tifffile not installed: mask comparison skipped")
    add("legacy demo truth.json identical (1 = yes)", 1,
        int(json.loads((A / "truth.json").read_text()) == json.loads(Path(f["truth"]).read_text())))  # fmt: skip
    save()

    # ------------------------------------------------------------------ 2 legacy numbers without the same-spot test
    summ = json.loads((DEMO / "output" / "synthetic.summary.json").read_text())
    sx = io.read_stereo_xcr(
        f["meta"], loci=("IGH", "IGK", "IGL"), library_id="synthetic"
    )
    U = sx.air
    add(
        "legacy demo reads",
        summ["n_reads"],
        sx.meta["n_reads"],
        "output/synthetic.summary.json",
    )
    add(
        "legacy demo umi == cid fraction",
        summ["umi_equals_cid_fraction"],
        sx.meta["umi_equals_cid_fraction"],
    )
    for ch, n in summ["n_units"].items():
        add(
            f"legacy demo {ch} (DNB, clone) units",
            n,
            int((U.locus == ch).sum()),
            "output/synthetic.summary.json",
        )
    AU = pd.read_parquet(DEMO / "output" / "synthetic.units.parquet")
    mine = pd.DataFrame(
        {
            "x": U.x_src.to_numpy(np.int64),
            "y": U.y_src.to_numpy(np.int64),
            "cloneId": U.clone_id.to_numpy(dtype=object),
            "reads": U.consensus_count.to_numpy(),
            "ncid": U.n_cid.to_numpy(),
        }
    )
    AU["cloneId"] = AU.cloneId.astype(str)
    J = AU.merge(mine, on=["x", "y", "cloneId"], how="outer", indicator=True)
    add(
        "legacy demo units not matched to the archived unit table",
        0,
        int((J._merge != "both").sum()),
    )
    Jb = J[J._merge == "both"]
    add(
        "legacy demo units whose read count differs",
        0,
        int((Jb.n_reads != Jb.reads).sum()),
    )
    add(
        "legacy demo units whose raw-CID count differs",
        0,
        int((Jb.n_cid != Jb.ncid).sum()),
    )
    mult = U.groupby("spot_id").clone_id.transform("size")
    sink = (mult >= 20).to_numpy()
    sk = U[sink].drop_duplicates("spot_id")
    mask_in = U.x_src.to_numpy(int), U.y_src.to_numpy(int)
    size = leg.meta["truth"]["chip_size"]
    c = size / 2
    in_t = (mask_in[0] - c) ** 2 + (mask_in[1] - c) ** 2 <= leg.meta["truth"][
        "tissue_radius"
    ] ** 2
    add(
        "legacy demo barcode sinks (DNBs with >= 20 clones)",
        summ["super_dnbs"]["n_dnbs"],
        len(sk),
        "4 planted",
    )
    add("legacy demo barcode sinks on the tissue", summ["super_dnbs"]["n_on_tissue"],
        int(in_t[sink][~U[sink].spot_id.duplicated().to_numpy()].sum()), "tissue = the disc of the mask")  # fmt: skip
    add(
        "legacy demo most clones on one DNB",
        summ["super_dnbs"]["max_clones_on_one_dnb"],
        int(mult.max()),
    )
    dist = AU.set_index(["x", "y", "cloneId"]).dist_um
    U_dist = dist.reindex(
        pd.MultiIndex.from_frame(mine[["x", "y", "cloneId"]])
    ).to_numpy()
    for ch, v in summ["fraction_outside_tissue"].items():
        m = (U.locus == ch).to_numpy()
        add(f"legacy demo {ch} share of units off the tissue", v, float((U_dist[m] > 0).mean()),
            "archived signed distance of each unit", tol=1e-12)  # fmt: skip
        add(f"legacy demo {ch} share of units off the tissue (DNB mask)", v, float((~in_t[m]).mean()),
            "tissue = DNB inside the disc; the archive used a 2.5-um (bin5) distance grid", tol=0.01)  # fmt: skip
    mol = leg.air
    add(
        "legacy port genuine IGL molecules",
        3000,
        int(((mol.locus == "IGL") & (mol.truth == "genuine")).sum()),
        "truth",
    )
    add(
        "legacy port genuine molecules of clone 9999",
        150,
        int(((mol.clone_id == "9999") & (mol.truth == "genuine")).sum()),
    )
    add("legacy port sink molecules (4 x 300)", 1200, int((mol.truth == "sink").sum()))
    hot = mol[(mol.clone_id == "9999") & (mol.truth == "genuine")]
    hc = np.array(leg.meta["truth"]["hot_center"])
    add(
        "legacy port hot molecules outside their 40-DNB disc",
        0,
        int((np.hypot(hot.x_src - hc[0], hot.y_src - hc[1]) > 40).sum()),
    )
    save()

    # ------------------------------------------------------------------ 3 M4 clonal_scan with the legacy settings
    try:
        from spaclone.tl import clonal_scan
    except ImportError:  # pragma: no cover
        clonal_scan = None
    scan_arch = pd.read_csv(DEMO / "output" / "synthetic.scan_clones.csv")
    if clonal_scan is None:
        log("tl.clonal_scan not in the tree: legacy scan numbers not run")
    else:
        keep = ~sink & (U_dist <= 0)
        T = U[keep]
        log(f"clonal_scan on {len(T)} tissue units without sinks (n_jobs={N_JOBS})")
        for ch in ("IGL", "IGK", "IGH"):
            d = T[T.locus == ch]
            res = clonal_scan(d, units="records", null="units", shape="square", radii_um=(5.0, 15.0, 25.0, 55.0),
                              grid_um=10.0,
                              model="poisson", min_units=20, n_perm=199, n_perm_max=9999, h=10,
                              stopping="besag_clifford", prefilter=False, rng=0, n_jobs=N_JOBS)  # fmt: skip
            tab = res.table
            a = scan_arch[scan_arch.chain == ch]
            add(
                f"legacy demo {ch} clones tested by the scan",
                len(a),
                len(tab),
                "min_units 20; output/scan_clones.csv",
            )
            sig = sorted(tab.loc[tab.q_value < 0.05, "clone_id"].astype(str))
            add(f"legacy demo {ch} clones called (q < 0.05)", int((a.q < 0.05).sum()), len(sig),
                f"called: {sig}; archived: {sorted(a.loc[a.q < 0.05, 'cloneId'].astype(str))}")  # fmt: skip
            if ch == "IGK":
                q = float(
                    tab.set_index(tab.clone_id.astype(str)).loc["9999", "q_value"]
                )
                add("legacy demo IGK clone 9999 q-value (<= 0.05)", float(a.set_index(a.cloneId.astype(str)).loc["9999", "q"]),
                    q, "Monte-Carlo: same decision, the value may differ by MC error", match=q <= 0.05)  # fmt: skip
            save()
    rows.append(
        {
            "quantity": "legacy demo same-spot numbers (p_det 0.79, IGL located 95.9%, IGK 1.0%, IGH 0.0%)",
            "archived": np.nan,
            "package": np.nan,
            "match": True,
            "note": "not run: needs M1's qc.same_spot / io.read_stereo_gef, still stubs in the tree",
        }
    )

    # ------------------------------------------------------------------ 4 synthetic_chip truth tables
    # legacy=True is the 1.0.0.dev0 chip (quantity names unchanged); the realistic default (F1, review M8) adds its
    # own checks: cDNA by compartment, constant-region RNA confined to its cells, swaps proportional to GEX molecules
    for model, legacy in (("", True), (" (realistic)", False)):
        sa = sim.synthetic_chip(rng=0, legacy=legacy)
        air, gex, meta = sa
        nons = air[air.truth != "sink"]
        share = float((nons.truth == "genuine").mean())
        se = np.sqrt(0.3 * 0.7 / len(nons))
        add(f"synthetic_chip{model} genuine share of non-sink records (design 0.3)", 0.3, share, f"binomial se {se:.4f}",
            tol=4 * se / 0.3)  # fmt: skip
        g = air[air.truth == "genuine"]
        dmax = float(np.hypot(g.x_um - g.true_x_um, g.y_um - g.true_y_um).max())
        add(f"synthetic_chip{model} genuine record farther from its cell than radius + half spot diagonal", 0,
            int(dmax > 4.0 + np.sqrt(2) + 1e-9), f"max {dmax:.2f} um")  # fmt: skip
        add(f"synthetic_chip{model} swapped records not on a GEX spot", 0,
            int((~air[air.truth == "swapped"].spot_id.isin(gex.obs_names)).sum()))  # fmt: skip
        add(f"synthetic_chip{model} sinks (design 4, 2 on tissue)", 4, len(meta["sinks"]))
        add(f"synthetic_chip{model} sinks on the tissue", 2, int(meta["sinks"].in_tissue.sum()))
        add(f"synthetic_chip{model} records per sink (design 150)", 150,
            float(air[air.truth == "sink"].groupby("spot_id").size().mean()))  # fmt: skip
        L = meta["lineages"]
        per_cell = g.groupby("true_cell").clone_id.nunique()
        add(f"synthetic_chip{model} cells with more than one heavy + one light clone", 0, int((per_cell > 2).sum()))
        add(f"synthetic_chip{model} lineages with a heavy locus other than IGH/TRB", 0,
            int((~L.heavy_clone.str.split("|").str[0].isin(["IGH", "TRB"])).sum()))  # fmt: skip
        cells, st = meta["cells"], meta["structures"].set_index("structure")
        gc = cells[cells.cell_type == "GC B"]
        add(f"synthetic_chip{model} GC B cells outside their germinal centre", 0,
            int((np.hypot(gc.x_um - gc.structure.map(st.x_um), gc.y_um - gc.structure.map(st.y_um)) > 40 + 1e-9).sum()))  # fmt: skip
        kw = {"legacy": True} if legacy else {"swap_weight": "background", "constant_detection": 0.5}
        lo = sim.synthetic_chip(rng=0, p_det=0.0, **kw)
        hi = sim.synthetic_chip(rng=0, p_det=0.5, **kw)
        genes = list(gex.var_names)

        def hits(s):
            """Records whose spot holds GEX of their own constant gene."""
            a_, g_ = s.air, s.gex
            out = np.zeros(len(a_), bool)
            on = a_.spot_id.isin(g_.obs_names).to_numpy()
            r = g_.obs_names.get_indexer(a_.spot_id[on])
            gl = list(g_.var_names)
            cc = [gl.index(sim.MOUSE_CONSTANT_GENES[x]) for x in a_.c_call[on]]
            out[on] = np.asarray(g_.X[r, cc]).ravel() > 0
            return out

        h1, h0 = hits(hi), hits(lo)
        gm = (hi.air.truth == "genuine").to_numpy() & ~h0
        cap = float(h1[gm].mean())
        se = np.sqrt(0.5 * 0.5 / gm.sum())
        add(f"synthetic_chip{model} own transcript captured on genuine spots without a chance hit (design p_det 0.5)",
            0.5, cap, f"same seed with p_det 0 gives the chance hits; se {se:.4f}; >= p_det because cell-mates add transcripts",
            match=cap >= 0.5 - 4 * se)  # fmt: skip
        if legacy:
            continue
        sw = air[air.truth == "swapped"]
        b_share = float(sw.cell_type.isin(["B", "GC B", "plasma"]).mean())
        old = sim.synthetic_chip(rng=0, legacy=True).air
        b_old = float(old[old.truth == "swapped"].cell_type.isin(["B", "GC B", "plasma"]).mean())
        add("synthetic_chip (realistic) swapped records on B-lineage spots / legacy share (> 1.8)", 1.8,
            b_share / b_old, f"{b_share:.3f} vs legacy {b_old:.3f}: receptor-rich, cDNA-rich cells attract swaps",
            match=b_share / b_old > 1.8)  # fmt: skip
        conf = sim.synthetic_chip(rng=0, p_det=0.0, constant_detection=1.0, n_sinks=0).gex
        Xc = conf.X.tocsc()
        ctc = conf.obs.cell_type.astype(str).to_numpy()
        ikc = np.asarray(Xc[:, list(conf.var_names).index("Igkc")].todense()).ravel()
        off = float(np.isin(ctc[ikc > 0], ["none", "other", "T"]).mean())
        add("synthetic_chip (realistic) Igkc spots not covered by a B-lineage cell (share; p_det 0: cells' RNA only)", 0,
            off, "constant-region RNA is confined to the spots of the cells that express it", match=off == 0)  # fmt: skip
        X = gex.X.tocsc()
        ik = np.asarray(X[:, genes.index("Igkc")].todense()).ravel()
        burst = float((ik[ik > 0] >= 2).mean())
        add("synthetic_chip (realistic) Igkc spots with >= 2 molecules (bursts; legacy ~0.16)", 0.3, burst,
            "share of Igkc-positive spots", match=burst > 0.3)  # fmt: skip
    save()

    # ------------------------------------------------------------------ 5 plant on a real background
    RAW = E / "AIRSPACE_GSE286452" / "raw"
    real = io.read_airspace(next(RAW.glob("GSM8727834_adata_D21_*.h5ad")), RAW / "GSM8727840_D21_ovp_clone_table.txt.gz",
                            loci=("IGH", "IGK", "IGL"), keep_x=False)  # fmt: skip
    bg = real.air[real.air.in_tissue].reset_index(drop=True)
    beads = real.gex.obs.assign(
        x_um=real.gex.obsm["spatial"][:, 0], y_um=real.gex.obsm["spatial"][:, 1]
    )
    slots = (
        beads.loc[beads.in_tissue, ["x_um", "y_um", "in_tissue"]]
        .rename_axis("spot_id")
        .reset_index()
    )
    P = sim.plant(bg, fraction=0.05, pattern="gc", sites=slots, slots=slots,
                  dedupe=["spot_id", "umi", "locus"], rng=2026)  # fmt: skip
    n_bg = len(bg)
    exp_planted = 0.05 * n_bg
    add("plant AIR-SPACE D21: planted records (design 5% of records)", exp_planted, int(P.air.planted.sum()),
        "binomial per clone; planted duplicates of a (bead, UMI, chain) dropped", tol=6 * np.sqrt(exp_planted * 0.95) / exp_planted)  # fmt: skip
    imp = P.implants
    tgt = imp[["target_x_um", "target_y_um"]].to_numpy()
    add("plant: planted cells drawn farther than 40 + 4 um from their focus", 0,
        int((np.hypot(tgt[:, 0] - imp.focus_x_um, tgt[:, 1] - imp.focus_y_um) > 44 + 1e-9).sum()),
        "gc pattern: cells within 40 um of the focus, records within 4 um of their cell")  # fmt: skip
    _, nearest = cKDTree(slots[["x_um", "y_um"]].to_numpy()).query(tgt)
    add("plant: planted records not on the tissue bead nearest to their drawn position", 0,
        int((slots.spot_id.to_numpy()[nearest] != P.air.spot_id.to_numpy()[imp.row.to_numpy()]).sum()))  # fmt: skip
    ks = stats.ks_2samp(
        P.air.consensus_count[P.air.planted], P.air.consensus_count[~P.air.planted]
    )
    add("plant: read counts of planted vs background records differ (KS p < 0.01; 0 = indistinguishable)", 0,
        int(ks.pvalue < 0.01), f"KS p = {ks.pvalue:.3f}")  # fmt: skip
    truth = P.air.planted.to_numpy()
    perfect = sim.score(truth, truth).iloc[0]
    add("score: perfect caller precision", 1, perfect.precision)
    add("score: perfect caller recall", 1, perfect.recall)
    rnd = np.random.default_rng(0).random(len(truth)) < 0.5
    r = sim.score(rnd, truth).iloc[0]
    add("score: random caller precision (expected = planted share)", float(truth.mean()), r.precision,
        "binomial", tol=6 * np.sqrt(truth.mean() * (1 - truth.mean()) / rnd.sum()) / truth.mean())  # fmt: skip

    R = save()
    log(f"{int(R.match.sum())} / {len(R)} quantities match")
    print(R.to_string(index=False), flush=True)
    sys.exit(
        0 if (R.match | R.note.fillna("").str.startswith("deliberate")).all() else 1
    )


if __name__ == "__main__":
    main()
