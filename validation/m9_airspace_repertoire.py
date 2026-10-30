"""M9 validation: AIR-SPACE clone dynamics (A1), public clones (A4) and the GC vs plasma-cell Gini contrast (X3, X3b).

Recomputes with spaclone functions only (tl.clone_units, tl.clone_sizes, tl.diversity, tl.public_clonotypes,
tl.public_top_clones, pp.define_clones):
  A1  results/QC/AIRSPACE_clones/expansion.csv: per lymph node occurrences (bead x clone = spot units), sites (beads of
      a clone <= 15 um apart merged = locus units), beads, molecules, clones, expanded clones (>= 5 beads), the share of
      the 10 largest clones and the Gini of clone sizes, unrarefied and rarefied to the smallest node (1,000 draws;
      beads and sites). Legacy replay: one generator default_rng(0) passed to tl.diversity node by node, beads then
      sites, as A1 drew -> must match exactly (means, 2.5% and 97.5% of the draws).
  A4  results/QC/AIRSPACE_clones/public_sensitivity.csv, public_clones.csv, top10_sharing.csv: clones by number of
      nodes for Dandelion-like clones (archived dclone, and recomputed with pp.define_clones), exact V|J|CDR3 and CDR3
      alone; beads per node of the clones in >= 3 nodes; top-10 clones found in another mouse vs 10 random clones
      (10,000 draws, default_rng(0), nodes in A4 order) -> exact.
  X3b results/AIRSPACE/gc_vs_pc.csv (the report's "GC Gini 0.13-0.44 vs plasma-cell area 0.46-0.57"): Gini of IGH
      molecule counts per clone in GC and PC beads (GC = X3b rule recomputed from the archived region maps) -> exact;
      GC - PC under clone labels shuffled over all IGH molecules on tissue (500) -> Monte-Carlo comparison of the null
      mean (|difference| <= 4 Monte-Carlo SE).  X3 results/AIRSPACE/gc_findings.csv (P2, X3 regions, 200 shuffles):
      the same.  Improved setting reported alongside: both regions rarefied to the smaller one, with the placebo.
      Clone-level test (fix F4b, 2026-10-30; reported): placebo="clones" (random placement of intact clones,
      tl.CloneRelocation) instead of the label placebo, whose p-values test whether a clone's molecules are scattered
      at random (they never are) rather than a difference in diversity.
Output: validation/results/m9_airspace_repertoire.csv (quantity, archived, package, match, note)
Usage : python -I m9_airspace_repertoire.py   (SPACLONE_DEV_SRC=<src dir> runs a development copy)
"""

import os
import sys
import time
import warnings
from pathlib import Path

if os.environ.get("SPACLONE_DEV_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_DEV_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

import spaclone  # noqa: E402
from spaclone import tl  # noqa: E402
from spaclone.pp import define_clones  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
QC = E / "results" / "QC"
AIRS = E / "results" / "AIRSPACE"
RAW = E / "AIRSPACE_GSE286452" / "raw"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = ["Mock", "D3", "D7", "D10", "D14", "D21"]
GSM = {"D7": "GSM8727837", "D10": "GSM8727838", "D14": "GSM8727839", "D21": "GSM8727840"}
UM = 0.65
T0 = time.time()
rows = []
KEYS = {"spot_key": "spot_id", "x_key": "x_um", "y_key": "y_um", "locus_key": "locus", "library_key": "day"}


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


def mc(quantity, archived_mean, archived_sd, n_arch, package_mean, package_sd, n_pkg, note):
    """Monte-Carlo comparison of two null means: |difference| <= 4 SE of the difference."""
    se = np.sqrt(archived_sd**2 / n_arch + package_sd**2 / n_pkg)
    ok = abs(archived_mean - package_mean) <= 4 * se
    rows.append(
        {
            "quantity": quantity,
            "archived": archived_mean,
            "package": package_mean,
            "match": bool(ok),
            "note": f"{note}; Monte-Carlo: |diff| {abs(archived_mean - package_mean):.4g} vs 4 SE {4 * se:.4g}",
        }
    )
    log(f"{quantity}: archived {archived_mean:.5g} package {package_mean:.5g} (4 SE {4 * se:.3g}) {'OK' if ok else 'MISMATCH'}")


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")

# ------------------------------------------------------------------------------------------------ A1 expansion
U = pd.read_parquet(QC / "clones" / "dandelion_AIRSPACE.units.parquet")
air = pd.DataFrame(
    {
        "spot_id": U["unit"].to_numpy(),
        "x_um": U["x"].to_numpy(dtype=float),
        "y_um": U["y"].to_numpy(dtype=float),
        "locus": U["receptor_type"].to_numpy(),
        "clone_id": U["dclone"].to_numpy(),
        "clone_exact": U["clone"].to_numpy(),
        "cdr3": U["clone"].str.split("|").str[2].to_numpy(),
        "day": pd.Categorical(U["day"], categories=DAYS, ordered=True),
    }
)
X = pd.read_csv(QC / "AIRSPACE_clones" / "expansion.csv").set_index("day")
per_day = {d: air[air["day"] == d] for d in DAYS}
occ = {d: len(tl.clone_units(per_day[d], unit="spot", **KEYS).units) for d in DAYS}
site = {d: len(tl.clone_units(per_day[d], unit="locus", radius_um=15.0, **KEYS).units) for d in DAYS}
nb, ns = min(occ.values()), min(site.values())
gen = np.random.default_rng(0)
for d in DAYS:
    a = per_day[d]
    add(f"A1 {d} occurrences (spot units)", X.loc[d, "occurrences"], occ[d])
    add(f"A1 {d} sites (locus units, 15 um)", X.loc[d, "sites"], site[d])
    add(f"A1 {d} beads", X.loc[d, "beads"], a["spot_id"].nunique())
    add(f"A1 {d} molecules", X.loc[d, "molecules"], len(a))
    cs = tl.clone_sizes(a, unit="spot", **KEYS)
    add(f"A1 {d} clones", X.loc[d, "clones"], len(cs))
    add(f"A1 {d} expanded clones (>= 5 beads)", X.loc[d, "clones_expanded"], int((cs["size"] >= 5).sum()))
    rb = tl.diversity(a, by=None, metrics=("top10", "gini"), unit="spot", rarefy_to=nb, n_rarefy=1000, rng=gen, **KEYS)
    rs = tl.diversity(a, by=None, metrics=("top10", "gini"), unit="locus", rarefy_to=ns, n_rarefy=1000, rng=gen, **KEYS)
    tb = rb.table.set_index("metric")
    ts = rs.table.set_index("metric")
    note = f"rarefied to {nb} beads, 1,000 draws, rng replay"
    add(f"A1 {d} top-10 share, rarefied", X.loc[d, "top10_share"], tb.loc["top10", "rarefied"], note)
    add(f"A1 {d} top-10 share, 2.5%", X.loc[d, "top10_lo"], tb.loc["top10", "rarefied_lo"], note)
    add(f"A1 {d} top-10 share, 97.5%", X.loc[d, "top10_hi"], tb.loc["top10", "rarefied_hi"], note)
    add(f"A1 {d} Gini, rarefied", X.loc[d, "gini"], tb.loc["gini", "rarefied"], note)
    add(f"A1 {d} Gini, 2.5%", X.loc[d, "gini_lo"], tb.loc["gini", "rarefied_lo"], note)
    add(f"A1 {d} Gini, 97.5%", X.loc[d, "gini_hi"], tb.loc["gini", "rarefied_hi"], note)
    note = f"sites rarefied to {ns}, 1,000 draws, rng replay"
    add(f"A1 {d} top-10 share of sites, rarefied", X.loc[d, "top10_share_sites"], ts.loc["top10", "rarefied"], note)
    add(f"A1 {d} top-10 share of sites, 2.5%", X.loc[d, "top10_sites_lo"], ts.loc["top10", "rarefied_lo"], note)
    add(f"A1 {d} top-10 share of sites, 97.5%", X.loc[d, "top10_sites_hi"], ts.loc["top10", "rarefied_hi"], note)
    add(f"A1 {d} Gini of sites, rarefied", X.loc[d, "gini_sites"], ts.loc["gini", "rarefied"], note)
    add(f"A1 {d} top-10 share, unrarefied", X.loc[d, "top10_share_unrarefied"], tb.loc["top10", "value"])
    add(f"A1 {d} Gini, unrarefied", X.loc[d, "gini_unrarefied"], tb.loc["gini", "value"])
    cl = tl.clone_sizes(a, unit="locus", radius_um=15.0, **KEYS)
    e = cl[cl["n_spots"] >= 5]
    add(
        f"A1 {d} beads per site of expanded clones",
        X.loc[d, "beads_per_site_expanded"],
        (e["n_spots"] / e["size"]).mean(),
        "mean over clones; clone order differs, so floating-point summation order differs",
        tol=1e-12,
    )

# ------------------------------------------------------------------------------------------------ A4 public clones
S = pd.read_csv(QC / "AIRSPACE_clones" / "public_sensitivity.csv").set_index("definition")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", spaclone._utils.SpacloneWarning)
    for name, key in (("Dandelion", "clone_id"), ("exact V|J|CDR3", "clone_exact"), ("CDR3 aa only", "cdr3")):
        pub = tl.public_clonotypes(air, sample_key="day", clone_key=key, unit="spot", **KEYS)
        add(f"A4 {name}: clones", S.loc[name, "clones"], len(pub))
        add(f"A4 {name}: max nodes", S.loc[name, "max_nodes"], int(pub["n_samples"].max()))
        for k in range(2, 7):
            add(f"A4 {name}: clones in {k} nodes", S.loc[name, f"in_{k}_nodes"], int((pub["n_samples"] == k).sum()))
    # the same chain from pp.define_clones (Dandelion-like rule, nodes pooled)
    parts = U["clone"].str.split("|", expand=True)
    rec = air.assign(v_call=parts[0].to_numpy(), j_call=parts[1].to_numpy(), junction_aa=parts[2].to_numpy())
    rec = define_clones(rec, method="vj_length_identity", identity=0.85, by=(), key_added="clone_pp")
    pub = tl.public_clonotypes(rec, sample_key="day", clone_key="clone_pp", unit="spot", **KEYS)
    for k in (2, 3, 4):
        add(
            f"A4 Dandelion via pp.define_clones: clones in {k} nodes",
            S.loc["Dandelion", f"in_{k}_nodes"],
            int((pub["n_samples"] == k).sum()),
        )
P = pd.read_csv(QC / "AIRSPACE_clones" / "public_clones.csv", index_col=0)
sizes = tl.clone_sizes(air, by="day", unit="spot", **KEYS)
piv = sizes.pivot_table(index="clone_id", columns="day", values="size", aggfunc="sum", observed=False).reindex(
    columns=DAYS
)
piv = piv.fillna(0).astype(int)
for c in P.index:
    for d in DAYS:
        add(f"A4 beads of {c} in {d}", P.loc[c, d], piv.loc[c, d] if c in piv.index else 0)
T = pd.read_csv(QC / "AIRSPACE_clones" / "top10_sharing.csv").set_index("day")
r = tl.public_top_clones(air, sample_key="day", top_k=10, n_draws=10_000, unit="spot", rng=np.random.default_rng(0), **KEYS)
tt = r.table.set_index("sample")
for d in [*DAYS, "all"]:
    add(f"A4 {d} top-10 clones found in another mouse", T.loc[d, "top10_shared"], tt.loc[d, "statistic"])
    add(f"A4 {d} 10 random clones: mean", T.loc[d, "null_mean"], tt.loc[d, "null_mean"], "10,000 draws, rng replay")
    add(f"A4 {d} 10 random clones: 2.5%", T.loc[d, "null_lo"], tt.loc[d, "null_lo"], "10,000 draws, rng replay")
    add(f"A4 {d} 10 random clones: 97.5%", T.loc[d, "null_hi"], tt.loc[d, "null_hi"], "10,000 draws, rng replay")
    add(f"A4 {d} p (more sharing)", T.loc[d, "p_more"], tt.loc[d, "p_value"], "10,000 draws, rng replay")
    if d != "all":
        info(f"A4 {d} exact hypergeometric p", T.loc[d, "p_more"], tt.loc[d, "p_hypergeom"], "improved: exact null")


# ------------------------------------------------------------------------------------------------ X3 / X3b Gini
def molecules(s):
    m = pd.read_csv(
        RAW / f"{GSM[s]}_{s}_ovp_clone_table.txt.gz",
        sep="\t",
        low_memory=False,
        usecols=["barcode", "umi", "receptor_type", "cloneId"],
    )
    m = m[m["receptor_type"].isin(["IGH", "IGK"])].drop_duplicates(["barcode", "umi", "receptor_type"])
    return m


def gc_x3b(t):
    """X3b germinal centres from the X3 region map: GC share top 8%, DBSCAN 25 um / 8, >= 30 beads, widened by 25 um."""
    gc = (t["gc_share"] >= np.quantile(t["gc_share"], 0.92)) & (t["gc_share"] > 0)
    cl = DBSCAN(eps=25 / UM, min_samples=8).fit_predict(t.loc[gc, ["x", "y"]].to_numpy())
    sizes_ = pd.Series(cl[cl >= 0]).value_counts()
    core = t.loc[gc].assign(c=cl)
    core = core[core["c"].isin(sizes_[sizes_ >= 30].index)]
    dist, j = cKDTree(core[["x", "y"]].to_numpy()).query(t[["x", "y"]].to_numpy())
    gid = np.where(dist <= 25 / UM, core["c"].to_numpy()[j], -1)
    return np.where(gid >= 0, "GC", np.where(t["region"] == "PC", "PC", "other"))


V = pd.read_csv(AIRS / "gc_vs_pc.csv")
V = V[V["stat"] == "gini_GC_minus_PC"].set_index("sample")
F = pd.read_csv(AIRS / "gc_findings.csv")
F = F[F["finding"] == "P2_gini_GC_minus_PC"].set_index("sample")
for s in GSM:
    t = pd.read_parquet(AIRS / f"region_maps_{s}.parquet")
    t["reg_x3b"] = gc_x3b(t)
    m = molecules(s).merge(t[["barcode", "x", "y", "region", "reg_x3b"]], on="barcode", how="inner")
    h = m[m["receptor_type"] == "IGH"].reset_index(drop=True)
    hair = pd.DataFrame(
        {
            "spot_id": h["barcode"].to_numpy(),
            "x_um": h["x"].to_numpy(dtype=float) * UM,
            "y_um": h["y"].to_numpy(dtype=float) * UM,
            "locus": "IGH",
            "clone_id": h["cloneId"].astype(str).to_numpy(),
            "region_x3": h["region"].to_numpy(),
            "region_x3b": h["reg_x3b"].to_numpy(),
        }
    )
    for tag, col, arch, n_perm, seed in (("X3b", "region_x3b", V, 500, 0), ("X3", "region_x3", F, 200, 1)):
        if s not in arch.index:
            continue
        g_gc, g_pc = ("gini_GC", "gini_PC")
        res = tl.diversity(hair, by=col, metrics=("gini",), unit="record", rarefy=False, placebo="labels", n_perm=n_perm, rng=seed)
        tab = res.table.set_index("group")
        add(f"{tag} {s} Gini GC (IGH molecules)", arch.loc[s, g_gc], tab.loc["GC", "value"], "exact")
        add(f"{tag} {s} Gini PC (IGH molecules)", arch.loc[s, g_pc], tab.loc["PC", "value"], "exact")
        add(f"{tag} {s} Gini GC - PC", arch.loc[s, "observed"], tab.loc["GC - PC", "value"], "exact")
        if tag == "X3b":
            add(f"{tag} {s} IGH molecules in GC", arch.loc[s, "molecules_GC"], tab.loc["GC", "n_units"])
            add(f"{tag} {s} IGH molecules in PC", arch.loc[s, "molecules_PC"], tab.loc["PC", "n_units"])
        mean_col, sd_col = ("null_mean", "null_sd") if tag == "X3b" else ("random_mean", "random_sd")
        mc(
            f"{tag} {s} Gini GC - PC under shuffled clone labels: mean",
            arch.loc[s, mean_col],
            arch.loc[s, sd_col],
            n_perm,
            tab.loc["GC - PC", "null_mean"],
            tab.loc["GC - PC", "null_sd"],
            n_perm,
            f"placebo='labels' (clone labels permuted over all IGH molecules on tissue), {n_perm} draws",
        )
        info(
            f"{tag} {s} Gini GC - PC: p (two-sided) vs shuffled labels",
            np.nan,
            tab.loc["GC - PC", "p_value"],
            "archive reported z only",
        )
        imp = tl.diversity(hair, by=col, metrics=("gini",), unit="record", rarefy=True, n_rarefy=200, placebo="labels", n_perm=n_perm, rng=seed + 10)
        it = imp.table.set_index("group")
        info(
            f"{tag} {s} improved: Gini GC - PC rarefied to {it.loc['GC', 'rarefied_to']} molecules",
            arch.loc[s, "observed"],
            it.loc["GC - PC", "rarefied"],
            f"null mean {it.loc['GC - PC', 'null_mean']:.4f}, p {it.loc['GC - PC', 'p_value']:.3g} (both regions "
            "rarefied to the smaller one, same placebo)",
        )
        info(f"{tag} {s} improved: Gini GC rarefied", arch.loc[s, g_gc], it.loc["GC", "rarefied"], "rarefied mean")
        info(f"{tag} {s} improved: Gini PC rarefied", arch.loc[s, g_pc], it.loc["PC", "rarefied"], "rarefied mean")
        cl = tl.diversity(hair, by=col, metrics=("gini",), unit="record", rarefy=True, n_rarefy=200, placebo="clones",
                          n_perm=n_perm, rng=seed + 20)  # fmt: skip
        ct = cl.table.set_index("group")
        info(
            f"{tag} {s} clone-level test: Gini GC - PC rarefied, random placement of intact clones",
            arch.loc[s, "observed"],
            ct.loc["GC - PC", "rarefied"],
            f"null mean {ct.loc['GC - PC', 'null_mean']:.4f}, sd {ct.loc['GC - PC', 'null_sd']:.4f}, p "
            f"{ct.loc['GC - PC', 'p_value']:.3g} (placebo='clones', {n_perm} relocations)",
        )

res = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
res.to_csv(OUT / "m9_airspace_repertoire.csv", index=False)
req = res[res["match"].notna()]
log(f"done: {int(req['match'].astype(bool).sum())}/{len(req)} required quantities match; {res['match'].isna().sum()} reported")
