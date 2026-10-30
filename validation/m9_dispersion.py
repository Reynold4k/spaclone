"""M9 validation: S1 "clustered" clone calls (StereoIR and the six AIR-SPACE nodes) with tl.clone_dispersion.

The archive's S1 rule (workflow/qc/S1_select_clones.py): IGH clones on >= 5 units (bins / beads); mean nearest-neighbour
distance of the clone's units vs the same number of units drawn from all IGH units of the sample (200 draws, null
shared by clones of one size), z = (observed - null mean) / null sd, one-sided normal p, BH, clustered = q < 0.01;
selected = clustered and (>= 50% IgG / IgA or median V identity < 98%); chance = the whole rule on IGH clone labels
shuffled between records (5 shuffles per sample). One generator (default_rng(0)) served StereoIR and then Mock, D3,
D7, D10, D14, D21 in turn, each followed by its 5 shuffles.

Legacy replay: tl.clone_dispersion(unit="spot", measures=("mean_nn",), p_method="normal", sequential=False,
n_perm=200) called in the same order with the same generator (the shuffles drawn with rng.permutation as in S1) ->
per-clone statistics, z, p, q, the clustered and selected counts and the shuffled-label selections must match exactly
(results/QC/clones/<dataset>__<sample>.clones.csv and the S1 log).

Improved setting (reported): Monte-Carlo p-values with Besag-Clifford sequential stopping and max_perm = m / alpha
(alpha 0.01), the floor that lets a single clone pass BH; on spot units (the S1 unit) and on loci (cell-scale units,
draws by sequential inhibition at 15 um).
Output: validation/results/m9_dispersion.csv (quantity, archived, package, match, note)
Usage : python -I m9_dispersion.py   (SPACLONE_DEV_SRC=<src dir> runs a development copy)
"""

import os
import re
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_DEV_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_DEV_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone import tl  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
C = E / "results" / "QC" / "clones"
LOG = E / "logs" / "qc_clones_33827606.log"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = [("StereoIR", "mm_spleen")] + [("AIRSPACE", d) for d in ["Mock", "D3", "D7", "D10", "D14", "D21"]]
KEYS = {"spot_key": "unit", "x_key": "x", "y_key": "y", "clone_key": "clone", "locus_key": "locus"}
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol * max(1.0, abs(a)))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}")


def info(quantity, archived, package, note):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def legacy(units, gen):
    return tl.clone_dispersion(
        units,
        unit="spot",
        min_units=5,
        n_perm=200,
        measures=("mean_nn",),
        p_method="normal",
        sequential=False,
        rng=gen,
        **KEYS,
    ).table


def selected(units, table, alpha=0.01):
    """S1's selection on top of the clustered call: >= 50% IgG / IgA records or median V identity < 98%."""
    g = units.groupby("clone")
    switched = g["iso"].agg(lambda v: float(v.isin(["IGHG", "IGHA"]).mean()))
    vid = g["vid"].median()
    t = table.set_index("clone")
    clustered = t["q_value"] < alpha
    return clustered & ((switched.reindex(t.index) >= 0.5) | (vid.reindex(t.index) < 98)), clustered


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
logtxt = LOG.read_text()
gen = np.random.default_rng(0)
tables = {}
for ds, sample in DATASETS:
    u = pd.read_parquet(C / f"{ds}__{sample}.units.parquet")
    if "locus" not in u.columns:
        u["locus"] = "IGH"
    arch = pd.read_csv(C / f"{ds}__{sample}.clones.csv")
    t1 = time.time()
    tab = legacy(u, gen)
    tables[(ds, sample)] = (u, tab)
    sel, clus = selected(u, tab)
    chance = []
    for _ in range(5):
        s = u.copy()
        s["clone"] = s["clone"].to_numpy()[gen.permutation(len(s))]
        ts = legacy(s, gen)
        chance.append(int(selected(s, ts)[0].sum()) if len(ts) else 0)
    log(f"{ds} {sample}: replay in {time.time() - t1:.1f} s")
    m = re.search(rf"{ds} {sample}: .*? (\d+) candidate clones .*?, (\d+) clustered, (\d+) selected; "
                  rf"shuffled labels select \[([0-9, ]+)\]", logtxt)
    tag = f"S1 {ds} {sample}"
    add(f"{tag} candidate clones (>= 5 units)", len(arch), len(tab))
    add(f"{tag} clustered (q < 0.01)", int((arch["q_clustered"] < 0.01).sum()), int(clus.sum()))
    add(f"{tag} selected (clustered and switched or mutated)", int(arch["selected"].sum()), int(sel.sum()))
    if m:
        arch_chance = [int(x) for x in m.group(4).split(",")]
        for i, (a, b) in enumerate(zip(arch_chance, chance, strict=True)):
            add(f"{tag} shuffle {i + 1}: selected", a, b, "S1 log; whole rule on shuffled IGH clone labels")
    mm = arch.merge(tab, on="clone", how="outer", indicator=True)
    add(f"{tag} clones in both tables", len(arch), int((mm["_merge"] == "both").sum()))
    both = mm[mm["_merge"] == "both"]
    for a_col, p_col, name in (("units", "n_units", "units"), ("nn_um", "statistic", "mean NN distance"),
                               ("nn_random_um", "null_mean", "null mean NN distance"), ("z_clustered", "z", "z"),
                               ("p_clustered", "p_value", "p"), ("q_clustered", "q_value", "q")):
        d = np.abs(both[a_col].to_numpy(dtype=float) - both[p_col].to_numpy(dtype=float))
        scale = np.maximum(1.0, np.abs(both[a_col].to_numpy(dtype=float)))
        add(f"{tag} max |difference| of {name} over clones", 0.0, float(np.max(d / scale)) if len(d) else 0.0,
            "relative to max(1, |archived|); floating point only", tol=1e-9)

# ---------------------------------------------------------------- improved: Monte-Carlo p with sequential stopping
for ds, sample in [("StereoIR", "mm_spleen"), ("AIRSPACE", "D14")]:
    u, leg = tables[(ds, sample)]
    arch = pd.read_csv(C / f"{ds}__{sample}.clones.csv")
    for unit in ("spot", "locus"):
        t1 = time.time()
        r = tl.clone_dispersion(u, unit=unit, min_units=5, n_perm=200, measures=("mean_nn",), p_method="mc",
                                sequential=True, alpha=0.01, rng=np.random.default_rng(1), **KEYS)
        tab = r.table
        info(
            f"S1 {ds} {sample} improved ({unit} units): clones tested",
            len(arch),
            len(tab),
            "loci = same-clone units within 15 um merged" if unit == "locus" else "S1 unit",
        )
        info(
            f"S1 {ds} {sample} improved ({unit} units): clustered at q < 0.01",
            int((arch["q_clustered"] < 0.01).sum()),
            int(tab["significant"].sum()),
            f"Monte-Carlo p, Besag-Clifford h=10, max_perm={r.params['max_perm']}; {time.time() - t1:.0f} s",
        )
        if unit == "spot":
            both = set(arch.loc[arch["q_clustered"] < 0.01, "clone"]) & set(tab.loc[tab["significant"], "clone"])
            info(f"S1 {ds} {sample} improved: clustered in both the legacy and the improved call", np.nan, len(both),
                 "overlap of the two calls")
            sq = tab.loc[tab["significant"], "p_value"]
            info(f"S1 {ds} {sample} improved: largest p among the clustered", np.nan, float(sq.max()) if len(sq) else np.nan,
                 "Monte-Carlo p reaches below the BH threshold")

res = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
res.to_csv(OUT / "m9_dispersion.csv", index=False)
req = res[res["match"].notna()]
log(f"done: {int(req['match'].astype(bool).sum())}/{len(req)} required quantities match; {res['match'].isna().sum()} reported")
