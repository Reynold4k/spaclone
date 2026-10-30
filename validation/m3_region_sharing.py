"""M3 validation: clone sharing between germinal centres in AIR-SPACE (archive X3b_airspace_gc_sharing.py) with
spaclone.tl.region_sharing.

Data preparation as in X3b (plain numpy / pandas / scipy, not package code): germinal centres = tissue beads whose
GC-B transcript share is in the top 8% (and > 0), DBSCAN clusters (eps 25 um, min_samples 8; re-implemented below with
scipy's cKDTree in scikit-learn's labelling order, because the package environment has no scikit-learn) of >= 30
beads, widened by 25 um; IGH and IGK molecules deduplicated by (barcode, UMI, chain) and joined to the beads.
Test: tl.region_sharing(design=("regions", "tissue"), n_perm=500, min_records=2) per chain: share of clones (>= 2 GC
molecules) in >= 2 germinal centres and mean share of a clone's GC molecules in its main germinal centre, against
clone labels shuffled among GC molecules and among all the chain's molecules.
The archive drew everything from one generator (np.random.default_rng(0)) across D7, D10, D14, D21: per chain the two
shuffles interleaved (as region_sharing draws them), then per sample 500 Gini shuffles and 500 within-clone region
shuffles. Those extra draws are replayed (same sizes, same order) between the region_sharing calls, so the nulls must
match exactly. Null SDs: the archive used ddof = 0, the package reports ddof = 1; the archive's value is recomputed
from the package's null draws.
Output: validation/results/m3_region_sharing.csv (quantity, archived, package, match, note)
Usage : python -I m3_region_sharing.py
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import spaclone
from spaclone import tl

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RAW = E / "AIRSPACE_GSE286452" / "raw"
RES = E / "results" / "AIRSPACE"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
GSM = {"D7": "GSM8727837", "D10": "GSM8727838", "D14": "GSM8727839", "D21": "GSM8727840"}
UM, N_PERM = 0.65, 500
SUB = re.compile(r"S[ACGTN](\d+)[ACGTN]")
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol * max(1.0, abs(a)))
    match = match or (np.isnan(a) and np.isnan(p))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}")


def info(quantity, archived, package, note):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def dbscan(xy, eps, min_samples):
    """scikit-learn's DBSCAN labelling (dbscan_inner: clusters grown depth-first from core points in index order)."""
    nb = cKDTree(xy).query_ball_point(xy, r=eps)
    core = np.array([len(v) >= min_samples for v in nb])
    labels = np.full(len(xy), -1, dtype=np.int64)
    label = 0
    for i0 in range(len(xy)):
        if labels[i0] != -1 or not core[i0]:
            continue
        stack, i = [], i0
        while True:
            if labels[i] == -1:
                labels[i] = label
                if core[i]:
                    stack.extend(v for v in nb[i] if labels[v] == -1)
            if not stack:
                break
            i = stack.pop()
        label += 1
    return labels


def v_mut(s):
    """X3b: % V mismatches of the first V alignment (substitutions per aligned length >= 50)."""
    if not isinstance(s, str) or not s:
        return np.nan
    f = s.split(";", 1)[0].split("|")
    try:
        n = int(f[1]) - int(f[0])
    except (ValueError, IndexError):
        return np.nan
    return 100 * len(SUB.findall(f[5])) / n if n >= 50 else np.nan


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
A = pd.read_csv(RES / "gc_sharing.csv")
g = np.random.default_rng(0)
for s, gsm in GSM.items():
    T = pd.read_parquet(RES / f"region_maps_{s}.parquet")
    gc = (T.gc_share >= np.quantile(T.gc_share, 0.92)) & (T.gc_share > 0)
    cl = dbscan(T.loc[gc, ["x", "y"]].to_numpy(), 25 / UM, 8)
    sizes = pd.Series(cl[cl >= 0]).value_counts()
    core = T.loc[gc].assign(c=cl)
    core = core[core.c.isin(sizes[sizes >= 30].index)]
    d, j = cKDTree(core[["x", "y"]].to_numpy()).query(T[["x", "y"]].to_numpy())
    T["gid"] = np.where(d <= 25 / UM, core.c.to_numpy()[j], -1)
    T["reg2"] = np.where(T.gid >= 0, "GC", np.where(T.region == "PC", "PC", "other"))
    M = pd.read_csv(RAW / f"{gsm}_{s}_ovp_clone_table.txt.gz", sep="\t", low_memory=False,
                    usecols=["barcode", "umi", "receptor_type", "cloneId", "allVAlignments"])
    M = M[M.receptor_type.isin(["IGH", "IGK"])].drop_duplicates(["barcode", "umi", "receptor_type"])
    M = M.merge(T[["barcode", "gid", "reg2"]], on="barcode", how="inner")
    a = A[A["sample"] == s]
    add(f"{s} germinal centres (largest DBSCAN id + 1, as X3b counts them)", a.germinal_centres.iloc[0],
        int(T.gid.max() + 1) if (T.gid >= 0).any() else 0)
    info(f"{s} germinal centres (distinct ids)", a.germinal_centres.iloc[0], int(T.gid[T.gid >= 0].nunique()),
         "X3b counted the largest DBSCAN label + 1")

    # legacy (archive X3b): every molecule a unit, two-sided p
    r = tl.region_sharing(M, region_key="gid", clone_key="cloneId", chain_key="receptor_type", min_records=2,
                          design=("regions", "tissue"), n_perm=N_PERM, background=-1, rng=g, keep_null=True,
                          unit_key=None, alternative="two-sided")
    tab = r.table.reset_index(drop=True)
    for ch in ("IGH", "IGK"):
        for stat, measure in (("frac_clones_in_2plus_GCs", "multi_region_share"),
                              ("mean_share_in_main_GC", "main_region_share")):
            ar = a[(a.chain == ch) & (a.stat == stat)].iloc[0]
            tag = f"{s} {ch} {stat}"
            k_reg = int(np.flatnonzero((tab.chain == ch) & (tab.design == "regions") & (tab.measure == measure))[0])
            k_tis = int(np.flatnonzero((tab.chain == ch) & (tab.design == "tissue") & (tab.measure == measure))[0])
            pr, pt = tab.iloc[k_reg], tab.iloc[k_tis]
            add(f"{tag}: observed", ar.observed, pr.statistic, "", tol=1e-12)
            add(f"{tag}: clones", ar.clones, pr.clones)
            add(f"{tag}: null mean, GC shuffle", ar.null_gc_shuffle_mean, pr.null_mean, "", tol=1e-12)
            add(f"{tag}: null mean, random placement", ar.null_random_place_mean, pt.null_mean, "", tol=1e-12)
            sd_reg = float(np.nanstd(r.null[:, k_reg]))
            sd_tis = float(np.nanstd(r.null[:, k_tis]))
            add(f"{tag}: null sd (ddof 0), GC shuffle", ar.null_gc_shuffle_sd, sd_reg, "from the package's draws",
                tol=1e-12)
            add(f"{tag}: null sd (ddof 0), random placement", ar.null_random_place_sd, sd_tis,
                "from the package's draws", tol=1e-12)
            add(f"{tag}: z (GC shuffle, ddof 0)", ar.z_gc_shuffle, (pr.statistic - pr.null_mean) / sd_reg, "",
                tol=1e-9)
            info(f"{tag}: Monte-Carlo p, random placement (two-sided)", np.nan, pt.p_value,
                 f"package z (ddof 1) = {pt.z:.3g}; not in the archive")
    # ---- improved (F4a defaults): cell-scale units, confinement direction; a separate generator (no replay shift)
    Mx = M.merge(T[["barcode", "x", "y"]], on="barcode", how="left").assign(x_um=lambda d: d.x * UM,
                                                                              y_um=lambda d: d.y * UM)
    for uk, label in (("auto", "loci 15 um"), ("spot", "(bead, clone)")):
        ri = tl.region_sharing(Mx, region_key="gid", clone_key="cloneId", chain_key="receptor_type", min_records=2,
                               design=("regions", "tissue"), n_perm=N_PERM, background=-1,
                               rng=np.random.default_rng([2026, 10, 30]), unit_key=uk, spot_key="barcode")
        for row in ri.table.itertuples():
            leg = tab[(tab.chain == row.chain) & (tab.design == row.design) & (tab.measure == row.measure)].iloc[0]
            info(f"{s} {row.chain} {row.measure} [improved: {label} units, {row.design} design]: observed / null mean",
                 f"{leg.statistic:.4g} / {leg.null_mean:.4g}", f"{row.statistic:.4g} / {row.null_mean:.4g}",
                 f"{row.clones} clones ({leg.clones} with molecules as units); p ({row.alternative}, confined) = "
                 f"{row.p_value:.4g}; units in GCs {row.records_in_regions} (molecules {leg.records_in_regions})")
    # ---- replay the archive's other draws of this sample (Gini, then within-clone region shuffles)
    H = M[M.receptor_type == "IGH"].reset_index(drop=True)
    for _ in range(N_PERM):
        g.permutation(len(H))
    H["vmut"] = H.allVAlignments.map(v_mut)
    D = H[H.reg2.isin(["GC", "PC"]) & H.vmut.notna()]
    both = D.groupby("cloneId").reg2.nunique()
    D = D[D.cloneId.isin(both[both == 2].index)].reset_index(drop=True)
    if D.cloneId.nunique() >= 5:
        sizes_c = [len(idx) for _, idx in D.groupby("cloneId").indices.items()]
        for _ in range(N_PERM):
            for n in sizes_c:
                g.permutation(n)
    log(f"{s} done")

res = pd.DataFrame(rows)
res.to_csv(OUT / "m3_region_sharing.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm3_region_sharing.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")
