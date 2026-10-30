"""M7 validation: Ig constant-gene RNA per unit (the depth strata of the trust tests) from package functions.

Recomputes the units (and calls) per Ig RNA stratum of the archive's X5 depth curve (results/DEPTH/t1_depth_curve.csv)
with spaclone.gex (call_tissue, ig_rna) and spaclone.nulls.IG_RNA_BINS:
  * AIR-SPACE (GSE286452, 6 lymph nodes): units = tissue beads (X3 rule), Ig RNA = Igkc, Iglc1-3 and the heavy
    constant genes; call = >= 1 IGH molecule (bead x UMI) in the ovp clone table;
  * Stereo-XCR-seq LUAD (GSE328481, 11 chips): units = CellBin cells, Ig RNA = IGKC, IGLC* (not IGLCOR) and the heavy
    constant genes; groups B/plasma (anno b_cell / plasma_cell) and non-B; call = IGHcdr3aa present.
Output: validation/results/m7_ig.csv (quantity, archived, package, match, note)
Usage : python -I m7_ig.py            (SPACLONE_DEV=<src dir> puts a development copy first on sys.path)
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_DEV"):
    sys.path.insert(0, os.environ["SPACLONE_DEV"])

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

import spaclone  # noqa: E402
from spaclone import gex  # noqa: E402

try:  # anndata >= 0.11
    from anndata.io import read_elem  # noqa: E402
except ImportError:  # pragma: no cover
    from anndata.experimental import read_elem  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RAW = E / "AIRSPACE_GSE286452" / "raw"
H5L = E / "StereoXCR_LUAD_GSE328481" / "raw"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = {"Mock": ("GSM8727829", "GSM8727835"), "D3": ("GSM8727830", "GSM8727836"),
           "D7": ("GSM8727831", "GSM8727837"), "D10": ("GSM8727832", "GSM8727838"),
           "D14": ("GSM8727833", "GSM8727839"), "D21": ("GSM8727834", "GSM8727840")}
IG_MOUSE_X5 = ["Igkc", "Iglc1", "Iglc2", "Iglc3", "Ighm", "Ighd", "Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha", "Ighe"]
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note=""):
    match = float(archived) == float(package)
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    if not match:
        log(f"MISMATCH {quantity}: archived {archived} package {package}")


def compare(dataset, sample, group, obs, t1):
    """Units and calls per Ig stratum (obs: ig_rna_bin, call) against the archive's t1 rows."""
    got = obs.groupby("ig_rna_bin", observed=True).agg(units=("call", "size"), calls=("call", "sum"))
    ref = t1[(t1.dataset == dataset) & (t1["sample"] == sample) & (t1.group == group)].set_index("ig_bin")
    for b in sorted(set(ref.index) | set(got.index.astype(str)), key=lambda v: list(spaclone.nulls.bin_labels(
            np.asarray(spaclone.nulls.IG_RNA_BINS[:-1]), spaclone.nulls.IG_RNA_BINS)).index(v)):
        ru = int(ref.units.get(b, 0))
        rc = int(ref.calls.get(b, 0))
        gu = int(got.units.get(b, 0)) if b in got.index else 0
        gc = int(got.calls.get(b, 0)) if b in got.index else 0
        add(f"{dataset} {sample} {group} Ig {b}: units", ru, gu, "t1_depth_curve.csv")
        add(f"{dataset} {sample} {group} Ig {b}: units with a call", rc, gc, "t1_depth_curve.csv")


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
t1 = pd.read_csv(E / "results" / "DEPTH" / "t1_depth_curve.csv")
add("mouse Ig constant genes identical to X5 (1 = yes)", 1, int(gex.IG_GENES["mouse"] == IG_MOUSE_X5))
add("Ig RNA strata edges identical to X5 [0,1,3,10,30,100,300,inf) (1 = yes)", 1,
    int(tuple(spaclone.nulls.IG_RNA_BINS) == (0, 1, 3, 10, 30, 100, 300, np.inf)))

# ---------------------------------------------------------------- AIR-SPACE
for s, (g_gex, g_air) in SAMPLES.items():
    a = ad.read_h5ad(next(RAW.glob(f"{g_gex}_adata_{s}_*.h5ad")))
    g = ad.AnnData(X=sparse.csr_matrix(a.X), obs=pd.DataFrame(index=a.obs["barcode"].astype(str).to_numpy()),
                   var=pd.DataFrame(index=a.var_names.astype(str)))
    g.obsm["spatial"] = np.column_stack([a.obs["x"].to_numpy(float), a.obs["y"].to_numpy(float)]) * 0.65
    del a
    gex.call_tissue(g, min_counts=100, radius_um=100.0, min_neighbors=15)
    t = g[g.obs.in_tissue.to_numpy()].copy()
    del g
    gex.ig_rna(t, genes="mouse")
    M = pd.read_csv(next(RAW.glob(f"{g_air}_{s}_ovp_clone_table.txt.gz")), sep="\t", low_memory=False,
                    usecols=["barcode", "umi", "receptor_type"])
    M = M[M.receptor_type == "IGH"].drop_duplicates(["barcode", "umi"])
    obs = pd.DataFrame({"ig_rna_bin": t.obs.ig_rna_bin.astype(str).to_numpy(),
                        "call": t.obs_names.isin(M.barcode.astype(str))})
    compare("AIR-SPACE", s, "all beads", obs, t1)
    log(f"AIR-SPACE {s}: {t.n_obs} tissue beads")

# ---------------------------------------------------------------- LUAD
for f in sorted(H5L.glob("GSM*.h5ad")):
    chip = f.stem.split("_")[1]
    with h5py.File(f, "r") as h:
        x = read_elem(h["X"])
        o = read_elem(h["obs"])
        v = read_elem(h["var"])
    g = ad.AnnData(X=sparse.csr_matrix(x), obs=pd.DataFrame(index=o.index.astype(str)),
                   var=pd.DataFrame(index=v.index.astype(str)))
    del x
    gex.ig_rna(g, genes="human")
    grp = np.where(o["anno"].astype(str).isin(["b_cell", "plasma_cell"]).to_numpy(), "B/plasma", "non-B")
    obs = pd.DataFrame({"ig_rna_bin": g.obs.ig_rna_bin.astype(str).to_numpy(),
                        "call": o["IGHcdr3aa"].notna().to_numpy(), "grp": grp})
    for grp_name, d in obs.groupby("grp"):
        compare("LUAD", chip, grp_name, d, t1)
    log(f"LUAD {chip}: {g.n_obs} cells, Ig genes {gex.ig_genes(g.var_names, 'human')}")
    del g

res = pd.DataFrame(rows)
res.to_csv(OUT / "m7_ig.csv", index=False)
log(f"wrote {OUT / 'm7_ig.csv'}: {int(res.match.sum())}/{len(res)} match")
