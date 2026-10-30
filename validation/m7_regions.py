"""M7 validation: AIR-SPACE tissue beads, germinal-centre / plasma-cell regions and bead cell types from package functions.

Recomputes with spaclone.gex only (call_tissue, call_regions, score_cell_types, celltype_composition):
  1. tissue beads (>= 100 counts and >= 15 such beads within 100 um): regions_summary.csv tissue_beads and the bead
     set of region_maps_<sample>.parquet;
  2. the archived region maps (X3 rule: GC-B share within 30 um in the top 4%, DBSCAN 30 um / 10 beads, >= 40 beads;
     PC = plasma share in the top 8% and not GC): per-bead gc_share, pc_share (float32 as archived), gc_id and region,
     and regions_summary.csv germinal_centres / gc_beads / pc_beads;
  3. the X3b germinal centres (top 8%, DBSCAN 25 um / 8 beads, >= 30 beads, widened by 25 um; the package default):
     gc_sharing.csv germinal_centres (D7-D21);
  4. the X6 bead typing (module enrichment, >= 2 marker UMIs; package setting min_module_counts=1, the legacy rule):
     per-bead class against bead_celltype_<sample>.parquet,
     and bcr_bead_celltype.csv (% of each class and of B lineage among beads with >= 1 IGH / >= 1 BCR / >= 3 BCR
     molecules and all tissue beads; depth-matched % from 20 depth bins).
Output: validation/results/m7_regions.csv (quantity, archived, package, match, note)
Usage : python -I m7_regions.py            (SPACLONE_DEV=<src dir> puts a development copy first on sys.path)
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_DEV"):
    sys.path.insert(0, os.environ["SPACLONE_DEV"])

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

import spaclone  # noqa: E402
from spaclone import gex  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RAW = E / "AIRSPACE_GSE286452" / "raw"
RES = E / "results" / "AIRSPACE"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = {"Mock": ("GSM8727829", "GSM8727835"), "D3": ("GSM8727830", "GSM8727836"),
           "D7": ("GSM8727831", "GSM8727837"), "D10": ("GSM8727832", "GSM8727838"),
           "D14": ("GSM8727833", "GSM8727839"), "D21": ("GSM8727834", "GSM8727840")}
UM = 0.65  # Curio pixel -> um (as in the archive)
X6_MODULES = {"B": ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Pax5", "Cr2", "Fcer2a"],
              "plasma": ["Jchain", "Xbp1", "Prdm1", "Sdc1", "Mzb1"],
              "T": ["Cd3e", "Cd3d", "Cd3g", "Cd4", "Cd8a", "Cd8b1"],
              "myeloid": ["Lyz2", "Cd68", "C1qa", "C1qb", "Csf1r"],
              "stromal": ["Col1a1", "Col3a1", "Pdpn", "Ccl21a", "Pecam1", "Cdh5"]}
CLASSES = [*X6_MODULES, "unassigned"]
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isnan(a) and np.isnan(p)) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol)
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    if not match:
        log(f"MISMATCH {quantity}: archived {archived} package {package}")


def load(sample, gsm):
    """Bead AnnData with raw counts and um coordinates (plain anndata; the archive's GEO h5ad)."""
    a = ad.read_h5ad(next(RAW.glob(f"{gsm}_adata_{sample}_*.h5ad")))
    g = ad.AnnData(X=sparse.csr_matrix(a.X), obs=pd.DataFrame(index=a.obs["barcode"].astype(str).to_numpy()),
                   var=pd.DataFrame(index=a.var_names.astype(str)))
    g.obsm["spatial"] = np.column_stack([a.obs["x"].to_numpy(float), a.obs["y"].to_numpy(float)]) * UM
    return g


def id_number(ids):
    s = pd.Series(ids, dtype=object)
    return s.map(lambda v: -1 if v is None or (isinstance(v, float) and np.isnan(v)) else int(str(v).rsplit("_", 1)[1]))


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
summ = pd.read_csv(RES / "regions_summary.csv").set_index("sample")
shar = pd.read_csv(RES / "gc_sharing.csv").drop_duplicates("sample").set_index("sample").germinal_centres
bct = pd.read_csv(RES / "bcr_bead_celltype.csv")
for s, (g_gex, g_air) in SAMPLES.items():
    a = load(s, g_gex)
    log(f"{s}: {a.n_obs} beads x {a.n_vars} genes")
    # ---- 1. tissue beads
    gex.call_tissue(a, min_counts=100, radius_um=100.0, min_neighbors=15)
    R = pd.read_parquet(RES / f"region_maps_{s}.parquet")
    tis = a.obs_names[a.obs.in_tissue.to_numpy()]
    add(f"{s}: tissue beads", summ.loc[s, "tissue_beads"], len(tis), "regions_summary.csv")
    add(f"{s}: tissue beads identical to region_maps (same barcodes, same order; 1 = yes)", 1,
        int(len(tis) == len(R) and (tis.to_numpy() == R.barcode.to_numpy()).all()))
    # ---- 2. X3 region maps
    gex.call_regions(a, kind="germinal_centre", top_frac=0.04, smooth_um=30.0, dbscan_eps_um=30.0, min_samples=10,
                     min_units=40, widen_um=0.0, relabel=False, region_key="region_x3", instance_key="id_x3",
                     share_key="gc_share", label="GC")
    gex.call_regions(a, kind="plasma_cell", top_frac=0.08, smooth_um=30.0, region_key="region_x3",
                     instance_key="id_x3", share_key="pc_share", label="PC")
    P = a.obs.loc[R.barcode]
    for col in ("gc_share", "pc_share"):
        pk = P[col].to_numpy(float).astype(np.float32)
        ar = R[col].to_numpy(np.float32)
        add(f"{s}: beads whose {col} differs (float32)", 0, int((pk != ar).sum()),
            f"{len(R):,} beads; max |diff| {np.max(np.abs(pk.astype(float) - ar.astype(float))):.3g}")
    gid = id_number(P["id_x3"].astype(object).to_numpy()).to_numpy()
    add(f"{s}: beads whose gc_id differs", 0, int((gid != R.gc_id.to_numpy()).sum()), f"{len(R):,} beads")
    add(f"{s}: beads whose region differs", 0, int((P.region_x3.astype(str).to_numpy() != R.region.to_numpy()).sum()))
    add(f"{s}: germinal centres (X3 rule)", summ.loc[s, "germinal_centres"], len(set(gid[gid >= 0])),
        "regions_summary.csv")
    add(f"{s}: GC beads (X3 rule)", summ.loc[s, "gc_beads"], int((P.region_x3 == "GC").sum()))
    add(f"{s}: PC beads (X3 rule)", summ.loc[s, "pc_beads"], int((P.region_x3 == "PC").sum()))
    # ---- 3. X3b germinal centres (package defaults)
    t_new = gex.call_regions(a, kind="germinal_centre", region_key="region_x3b", instance_key="id_x3b")
    t_raw = gex.call_regions(a, kind="germinal_centre", region_key="region_raw", instance_key="id_raw", relabel=False)
    mx = int(id_number(t_raw.region_id.to_numpy()).max()) + 1 if len(t_raw) else 0
    if s in shar.index:
        add(f"{s}: germinal centres X3b, archive count = largest DBSCAN label + 1", shar[s], mx,
            "gc_sharing.csv (T.gid.max() + 1); package relabel=False")
        add(f"{s}: germinal centres X3b, distinct instances", shar[s], len(t_new),
            "distinct GCs; differs from the archive's count when DBSCAN clusters < 30 beads were dropped")
    log(f"{s}: X3 GCs {len(set(gid[gid >= 0]))}, X3b GCs {len(t_new)} (max label + 1 = {mx})")
    # ---- 4. X6 bead typing
    gex.score_cell_types(a, markers=X6_MODULES, method="enrichment", min_marker_counts=2, min_module_counts=1,
                         tissue_key="in_tissue", key_added="cls")  # min_module_counts=1 = the X6 rule
    B = pd.read_parquet(RES / f"bead_celltype_{s}.parquet")
    pc = a.obs.loc[B.barcode, "cls"].astype(str).to_numpy()
    add(f"{s}: beads whose X6 class differs", 0, int((pc != B.cls.to_numpy()).sum()), f"{len(B):,} beads")
    M = pd.read_csv(next(RAW.glob(f"{g_air}_{s}_ovp_clone_table.txt.gz")), sep="\t", low_memory=False,
                    usecols=["barcode", "umi", "receptor_type"])
    M = M[M.receptor_type.isin(["IGH", "IGK", "IGL"])].drop_duplicates(["barcode", "umi", "receptor_type"])
    nb = M.barcode.value_counts()
    nh = M[M.receptor_type == "IGH"].barcode.value_counts()
    bcr = a.obs_names.map(nb).fillna(0).to_numpy(dtype=int)
    igh = a.obs_names.map(nh).fillna(0).to_numpy(dtype=int)
    tm = a.obs.in_tissue.to_numpy()
    sets = {"IGH >= 1": igh >= 1, "BCR >= 1": bcr >= 1, "BCR >= 3": bcr >= 3, "all tissue beads": tm}
    for name, sel in sets.items():
        comp = gex.celltype_composition(a, sel, type_key="cls", n_depth_bins=20, tissue_key="in_tissue",
                                        groups={"B_lineage": ["B", "plasma"]}).set_index("cell_type")
        ref = bct[(bct["sample"] == s) & (bct.beads == name)].iloc[0]
        add(f"{s} [{name}]: beads", ref["n"], int((sel & tm).sum()), "bcr_bead_celltype.csv")
        for c in [*CLASSES, "B_lineage"]:
            add(f"{s} [{name}]: % {c}", ref[f"pct_{c}"], comp.loc[c, "observed_pct"], tol=1e-9)
            if name != "all tissue beads":
                add(f"{s} [{name}]: depth-matched % {c}", ref[f"depth_matched_pct_{c}"],
                    comp.loc[c, "depth_matched_pct"], tol=1e-9)
    log(f"{s}: typing done")
    del a

res = pd.DataFrame(rows)
res.to_csv(OUT / "m7_regions.csv", index=False)
log(f"wrote {OUT / 'm7_regions.csv'}: {int(res.match.sum())}/{len(res)} match")
