"""M9 validation: StereoIR TCR invariant chains (archive TR1 / TR3) and their generation probability (TR2).

Recomputes with spaclone functions only (pp.filter_valid, pp.define_clones, pp.merge_loci, tl.invariant_tcr, tl.pgen):
  TR1 / TR3  raw TCR AIRR (read-only) -> TRA / TRB contigs passing the validity rules (valid junction, productive,
             >= 2 reads) -> exact clonotypes -> iNKT alpha (TRAV11 / 11D / 11N + TRAJ18 + CVVGDRGSALGRLHF) and MAIT
             alpha (TRAV1 + TRAJ33 + CAVRDSNYQLIW) contigs and loci (same-clonotype bins within 15 um merged):
             310 / 295 iNKT, 4 / 4 MAIT (results/QC/StereoIR_TCR/TR1_summary.csv, TR3_summary.csv) -> exact.
             The iNKT loci's T-zone / follicle / red-pulp counts (173 / 104 / 18, TR3_compartments.csv) using the
             archived compartment of each locus (matched by clonotype and centroid), and the TR3 beta-chain identity
             check (TRB contigs using TRBV13-2 / 29 / 1 in the bins of iNKT loci: 128 of 453) -> exact.
             Also reported: the 'genes' rule (V and J only), which the archive did not use.
  TR2        OLGA Pgen of every TR1 clonotype (mouse_T_alpha / mouse_T_beta, V and J given when the model knows them)
             against results/QC/StereoIR_TCR/TR2_pgen.tsv (written with 7 significant digits): relative error
             <= 1e-6, zeros and known-gene flags identical.
Output: validation/results/m9_stereoir_tcr.csv (quantity, archived, package, match, note)
Usage : python -I m9_stereoir_tcr.py [n_workers]   (SPACLONE_DEV_SRC=<src dir> runs a development copy)
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_DEV_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_DEV_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone import tl  # noqa: E402
from spaclone.pp import define_clones, filter_valid, merge_loci  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RES = E / "results" / "QC" / "StereoIR_TCR"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
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


def main():
    log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
    summary1 = pd.read_csv(RES / "TR1_summary.csv").set_index("item")["value"]
    summary3 = pd.read_csv(RES / "TR3_summary.csv").set_index("item")["value"]
    cols = ["cell_id", "locus", "v_call", "d_call", "j_call", "c_call", "junction", "junction_aa", "productive"]
    raw = pd.read_csv(E / "StereoIR" / "mm_spleen_tcr_airr.tsv.tsv", sep="\t", usecols=[*cols, "consensus_count"])
    add("TR1 raw TCR contigs", summary1["raw TCR contigs"], len(raw))
    raw = raw[raw["locus"].isin(["TRA", "TRB"])].reset_index(drop=True)
    xy = raw["cell_id"].str.split("-", expand=True).astype(float) * 0.5
    raw = raw.assign(spot_id=raw["cell_id"], x_um=xy[0].to_numpy(), y_um=xy[1].to_numpy())
    fv = filter_valid(raw, require_productive=True, min_reads=2)
    air = fv.air.reset_index(drop=True)
    add("TR1 valid TRA/TRB contigs", summary1["valid contigs"], len(air), "pp.filter_valid (junction, productive, >= 2 reads)")
    air = define_clones(air, method="exact", by=())
    air = tl.invariant_tcr(air, species="mouse", rule="canonical")
    add("TR1 iNKT alpha contigs", summary1["iNKT alpha contigs"], int(air["inkt"].sum()), "tl.invariant_tcr canonical")
    add("TR1 MAIT alpha contigs", summary1["MAIT alpha contigs"], int(air["mait"].sum()), "tl.invariant_tcr canonical")
    loci, member = merge_loci(air, radius_um=15.0)
    air["locus_id"] = member.to_numpy()
    inkt_loci = set(air.loc[air["inkt"], "locus_id"])
    mait_loci = set(air.loc[air["mait"], "locus_id"])
    add("TR1 iNKT alpha loci", summary1["iNKT alpha loci"], len(inkt_loci), "pp.merge_loci 15 um")
    add("TR3 iNKT loci (on the typed-bin grid)", summary3["iNKT loci"], len(inkt_loci), "no iNKT locus is off the grid")
    add("TR1 MAIT alpha loci", summary1["MAIT alpha loci"], len(mait_loci))
    add("TR3 MAIT loci", summary3["MAIT loci"], len(mait_loci))
    g = tl.invariant_tcr(air, species="mouse", rule="genes")
    info("iNKT alpha contigs, 'genes' rule (V + J only)", summary1["iNKT alpha contigs"], int(g["inkt"].sum()), "not used by the archive")
    info("MAIT alpha contigs, 'genes' rule (V + J only)", summary1["MAIT alpha contigs"], int(g["mait"].sum()), "not used by the archive")

    # compartments of the iNKT loci (archived locus compartments, matched by clonotype and centroid)
    lo_arch = pd.read_parquet(RES / "TR1_tcr_loci.parquet")
    lo_arch["key"] = lo_arch["clone"] + "|" + lo_arch["x"].round(4).astype(str) + "|" + lo_arch["y"].round(4).astype(str)
    mine = loci[loci["locus_id"].isin(inkt_loci | mait_loci)].copy()
    mine["clone_plain"] = mine["clone_id"].str.split("|", n=0).str[-3:].str.join("|")
    mine["key"] = mine["clone_plain"] + "|" + mine["x_um"].round(4).astype(str) + "|" + mine["y_um"].round(4).astype(str)
    mm = mine.merge(lo_arch[["key", "comp", "on_grid", "inkt", "mait", "bin_ids"]], on="key", how="left")
    add("iNKT/MAIT loci matched to the archived TR1 loci", len(inkt_loci) + len(mait_loci), int(mm["comp"].notna().sum()))
    cp = pd.read_csv(RES / "TR3_compartments.csv").set_index("compartment")
    ink = mm[mm["locus_id"].isin(inkt_loci)]
    for c in cp.index:
        add(f"TR3 iNKT loci in {c}", cp.loc[c, "iNKT_loci"], int((ink["comp"] == c).sum()), "archived compartment per locus")

    # TR3 beta-chain identity check: TRB contigs (and iNKT-type TRB V genes) in the bins of each iNKT locus, summed
    beta = pd.read_csv(RES / "TR3_beta_check.csv").set_index("group")
    trb = air[air["locus"] == "TRB"]
    n_all = trb.groupby("spot_id").size()
    n_nkt = trb[trb["inkt_beta_v"]].groupby("spot_id").size()
    bins = air[air["locus_id"].isin(inkt_loci)].groupby("locus_id")["spot_id"].agg(lambda s: sorted(set(s)))
    tot = sum(int(n_all.reindex(b).fillna(0).sum()) for b in bins)
    nkt = sum(int(n_nkt.reindex(b).fillna(0).sum()) for b in bins)
    add("TR3 TRB contigs in the bins of iNKT loci", beta.loc["iNKT loci", "TRB_contigs_in_bins"], tot)
    add("TR3 of which TRBV13-2 / 29 / 1 (inkt_beta_v)", beta.loc["iNKT loci", "iNKT_type_V"], nkt)

    # TR2 Pgen
    tr2 = pd.read_csv(RES / "TR2_pgen.tsv", sep="\t")
    cl = air.drop_duplicates("clone_id")[["locus", "v_call", "j_call", "junction_aa", "clone_id"]].copy()
    cl["clone"] = cl["clone_id"].str.split("|").str[-3:].str.join("|")
    add("TR1 clonotypes (TRA + TRB)", len(tr2), len(cl))
    t1 = time.time()
    p = tl.pgen(cl, species="mouse", n_jobs=N_WORKERS)
    log(f"pgen of {len(cl)} clonotypes in {time.time() - t1:.0f} s with {N_WORKERS} workers")
    cmp = cl.assign(pgen_vj=p["pgen_vj"].to_numpy(), pgen_aa=p["pgen_aa"].to_numpy(), v_known=p["v_known"].to_numpy(),
                    j_known=p["j_known"].to_numpy()).merge(tr2, on="clone", suffixes=("", "_arch"))
    add("TR2 clonotypes compared", len(tr2), len(cmp))
    for col in ("pgen_vj", "pgen_aa"):
        a, b = cmp[f"{col}_arch"].to_numpy(dtype=float), cmp[col].to_numpy(dtype=float)
        pos = a > 0
        rel = np.abs(a[pos] - b[pos]) / a[pos]
        add(f"TR2 {col}: max relative error (archive: 7 significant digits)", 0.0, float(rel.max()), "tolerance 1e-6", tol=1e-6)
        add(f"TR2 {col}: zeros", int((a == 0).sum()), int((b == 0).sum()))
    for col in ("v_known", "j_known"):
        add(f"TR2 {col} identical", len(cmp), int((cmp[col].to_numpy() == cmp[f"{col}_arch"].to_numpy()).sum()))

    res = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
    res.to_csv(OUT / "m9_stereoir_tcr.csv", index=False)
    req = res[res["match"].notna()]
    log(f"done: {int(req['match'].astype(bool).sum())}/{len(req)} required quantities match; {res['match'].isna().sum()} reported")


if __name__ == "__main__":
    main()
