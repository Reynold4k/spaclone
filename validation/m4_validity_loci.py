"""M4 validation: sequence validity, clone definitions and locus merging against the archive.

Recomputes with spaclone functions only (pp.filter_valid, pp.define_clones, pp.merge_loci):
  1. QC filter counts (the archive's figures/QC/.../QC_filter_summary.csv): StereoIR BCR / TCR bins with any contig before
     and after the validity filter (valid junction, productive, >= 2 reads), from the raw AIRR files.
  2. T1 (results/QC/StereoIR_clones/T1_summary.csv): IGH contigs, exact clones, bins, loci (same-clone bins within
     15 um, single linkage), loci with >= 2 bins; plus the exact locus partition against T1_igh_contigs.parquet.
  3. TR1 (results/QC/StereoIR_TCR/TR1_summary.csv): per chain contigs, bins, clonotypes, loci, member loci (best
     contig >= 20 reads) and clonotypes with >= 2 / 3 / 5 member loci; locus partition against TR1_tcr_contigs.parquet.
  4. S4 / A4 (logs/air_A14_*.log, results/QC/clones/dandelion_AIRSPACE.units.parquet): AIR-SPACE Dandelion-like
     clones (same V, J, junction length, identity >= 85 %, single linkage, six nodes pooled) by number of nodes, and
     the partition against the archived dclone; StereoIR exact -> Dandelion-like clone counts.
Output: validation/results/m4_validity_loci.csv (quantity, archived, package, match, note)
Usage : python -I m4_validity_loci.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import spaclone
from spaclone.pp import define_clones, filter_valid, merge_loci

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = abs(a - p) <= tol * max(1.0, abs(a))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}")


def same_partition(a, b):
    """True if label arrays a and b define the same partition of the rows."""
    t = pd.DataFrame({"a": np.asarray(a), "b": np.asarray(b)})
    return bool(t.groupby("a").b.nunique().max() == 1 and t.groupby("b").a.nunique().max() == 1)


def stereoir(kind):
    cols = [
        "cell_id",
        "locus",
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
        "productive",
        "consensus_count",
    ]
    a = pd.read_csv(E / "StereoIR" / f"mm_spleen_{kind}_airr.tsv.tsv", sep="\t", usecols=cols, low_memory=False)
    xy = a.cell_id.str.split("-", expand=True).astype(float) * 0.5
    a = a.rename(columns={"cell_id": "spot_id"}).assign(x_um=xy[0].to_numpy(), y_um=xy[1].to_numpy())
    return a


log(f"spaclone {spaclone.__version__}")
summary = pd.read_csv(E / "figures" / "QC" / "00_\u6807\u51c6\u4e0e\u793a\u610f" / "QC_filter_summary.csv", encoding="utf-8-sig")
arch = {r.Dataset: (int(r.Before.replace(",", "")), int(r.After.replace(",", ""))) for r in summary.itertuples()}

# ------------------------------------------------------------------ 1 filter counts (StereoIR BCR / TCR)
raw, ok = {}, {}
for kind in ("bcr", "tcr"):
    raw[kind] = stereoir(kind)
    ok[kind], rep = filter_valid(raw[kind], require_productive=True, min_reads=2)
    b, a = arch[f"StereoIR (5' demo) {kind.upper()}"]
    add(
        f"StereoIR {kind.upper()} bins with a contig, before filter",
        b,
        int(rep.spots.iloc[0]),
        "QC_filter_summary.csv Before",
    )
    add(
        f"StereoIR {kind.upper()} bins after filter (valid junction, productive, >= 2 reads)",
        a,
        int(rep.spots.iloc[-1]),
        "QC_filter_summary.csv After",
    )
    log(rep.to_string(index=False))

# ------------------------------------------------------------------ 2 T1 (IGH exact clones and loci)
t1 = pd.read_csv(E / "results" / "QC" / "StereoIR_clones" / "T1_summary.csv").set_index("item").value
H = ok["bcr"][ok["bcr"].locus == "IGH"].copy()
H = define_clones(H, method="exact", by=())
loci, member = merge_loci(H, radius_um=15.0)
add("T1 IGH contigs", t1["IGH contigs"], len(H))
add("T1 clones (exact V|J|junction)", t1["clones"], H.clone_id.nunique())
add("T1 IGH bins", t1["IGH bins"], H.spot_id.nunique())
add("T1 loci (15 um, single linkage)", t1["loci"], len(loci))
add("T1 bins per locus, mean", t1["bins per locus mean"], loci.n_spots.mean(), tol=1e-9)
add("T1 loci with >= 2 bins", t1["loci with >= 2 bins"], int((loci.n_spots >= 2).sum()))
T1c = pd.read_parquet(
    E / "results" / "QC" / "StereoIR_clones" / "T1_igh_contigs.parquet", columns=["cell_id", "clone", "loc_id"]
)
mine = pd.DataFrame(
    {"cell_id": H.spot_id.to_numpy(), "clone": H.clone_id.astype(str).to_numpy(), "lid": member.to_numpy()}
).drop_duplicates(["cell_id", "clone"])
m = T1c.drop_duplicates(["cell_id", "clone"]).merge(mine, on=["cell_id", "clone"], how="outer", indicator=True)
add("T1 (bin, clone) pairs not matched to the archive", 0, int((m._merge != "both").sum()))
add(
    "T1 locus partition identical to T1_igh_contigs.loc_id (1 = yes)",
    1,
    int(same_partition(m.loc_id[m._merge == "both"], m.lid[m._merge == "both"])),
)

# ------------------------------------------------------------------ 3 TR1 (TRA / TRB exact clonotypes and loci)
tr1 = pd.read_csv(E / "results" / "QC" / "StereoIR_TCR" / "TR1_summary.csv").set_index("item").value
A = ok["tcr"][ok["tcr"].locus.isin(["TRA", "TRB"])].copy()
add("TR1 raw TCR contigs", tr1["raw TCR contigs"], len(raw["tcr"]))
add("TR1 valid TRA + TRB contigs", tr1["valid contigs"], len(A))
A = define_clones(A, method="exact", by=())
lt, mt = merge_loci(A, radius_um=15.0)
for ch in ("TRA", "TRB"):
    s = A[A.locus == ch]
    L = lt[lt.locus == ch]
    M = L[L.reads_max >= 20]
    sz = M.groupby("clone_id").size()
    add(f"TR1 {ch} contigs", tr1[f"{ch} contigs"], len(s))
    add(f"TR1 {ch} bins", tr1[f"{ch} bins"], s.spot_id.nunique())
    add(f"TR1 {ch} clonotypes", tr1[f"{ch} clonotypes"], s.clone_id.nunique())
    add(f"TR1 {ch} loci", tr1[f"{ch} loci"], len(L))
    add(f"TR1 {ch} member loci (>= 20 reads)", tr1[f"{ch} member loci (>= 20 reads)"], len(M))
    for k in (2, 3, 5):
        add(
            f"TR1 {ch} clonotypes with >= {k} member loci",
            tr1[f"{ch} clonotypes with >= {k} member loci"],
            int((sz >= k).sum()),
        )
TRc = pd.read_parquet(
    E / "results" / "QC" / "StereoIR_TCR" / "TR1_tcr_contigs.parquet", columns=["cell_id", "clone", "loc_id"]
)
mine = pd.DataFrame(
    {"cell_id": A.spot_id.to_numpy(), "clone": A.clone_id.astype(str).to_numpy(), "lid": mt.to_numpy()}
).drop_duplicates(["cell_id", "clone"])
m = TRc.drop_duplicates(["cell_id", "clone"]).merge(mine, on=["cell_id", "clone"], how="outer", indicator=True)
add("TR1 (bin, clonotype) pairs not matched to the archive", 0, int((m._merge != "both").sum()))
add(
    "TR1 locus partition identical to TR1_tcr_contigs.loc_id (1 = yes)",
    1,
    int(same_partition(m.loc_id[m._merge == "both"], m.lid[m._merge == "both"])),
)

# ------------------------------------------------------------------ 4 S4 / A4: AIR-SPACE Dandelion-like clone sharing
DAYS = ["Mock", "D3", "D7", "D10", "D14", "D21"]
C = E / "results" / "QC" / "clones"
U = pd.concat([pd.read_parquet(C / f"AIRSPACE__{d}.units.parquet").assign(day=d) for d in DAYS], ignore_index=True)
p = U.clone.str.split("|", expand=True)
U = U.assign(v_call=p[0], j_call=p[1], junction_aa=p[2], locus=U.receptor_type)
U = define_clones(U, method="vj_length_identity", identity=0.85, by=())
nodes = U.drop_duplicates(["day", "unit", "clone_id"]).groupby("clone_id").day.nunique().value_counts()
arch_nodes = {1: 4568, 2: 51, 3: 5, 4: 1}  # logs/air_A14_33830212.log, line 69
for k, v in arch_nodes.items():
    add(f"S4/A4 AIR-SPACE Dandelion-like clones in {k} node(s)", v, int(nodes.get(k, 0)), "logs/air_A14_33830212.log")
add("S4/A4 AIR-SPACE clones in >= 5 nodes", 0, int(nodes[nodes.index >= 5].sum()))
D = pd.read_parquet(C / "dandelion_AIRSPACE.units.parquet", columns=["day", "umi", "barcode", "clone", "dclone"])
add("S4 AIR-SPACE molecules (rows)", len(D), len(U))
add(
    "S4 AIR-SPACE partition identical to archived dclone (1 = yes)",
    1,
    int(len(D) == len(U) and (D.clone.to_numpy() == U.clone.to_numpy()).all() and same_partition(D.dclone, U.clone_id)),
    "row order of the units files",
)
IR = pd.read_parquet(C / "dandelion_StereoIR.units.parquet")
p = IR.clone.str.split("|", expand=True)
IR = IR.assign(v_call=p[0], j_call=p[1], junction_aa=p[2])
IR2 = define_clones(IR, method="vj_length_identity", by=())
add(
    "S4 StereoIR exact clones",
    IR.clone.nunique(),
    IR.assign(c=IR.v_call + "|" + IR.j_call + "|" + IR.junction_aa).c.nunique(),
    "dandelion_StereoIR.units.parquet",
)
add(
    "S4 StereoIR Dandelion-like clones", IR.dclone.nunique(), IR2.clone_id.nunique(), "dandelion_StereoIR.units.parquet"
)
add("S4 StereoIR partition identical to archived dclone (1 = yes)", 1, int(same_partition(IR.dclone, IR2.clone_id)))

R = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
R.to_csv(OUT / "m4_validity_loci.csv", index=False)
log(f"{int(R.match.sum())} / {len(R)} quantities match")
print(R.to_string(index=False), flush=True)
sys.exit(0 if R.match.all() else 1)
