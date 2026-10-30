"""M6 validation: interoperability (io.to_scirpy / from_scirpy, io.write / read, io.to_spatialdata, pp.aggregate).

On the StereoIR BCR table ($SPACLONE_ARCHIVE/StereoIR/mm_spleen_bcr_airr.tsv.tsv, every AIRR column plus the
bin20 position in µm):
  1. SpatialAIR -> scirpy AnnData (spot level and locus level) -> back is lossless: same columns, values, dtypes and
     row order (pandas.testing.assert_frame_equal), in memory and after writing the AnnData to h5ad;
  2. scirpy runs on it: pp.index_chains, tl.clonal_expansion (locus level: loci per clone), pp.ir_dist +
     tl.define_clonotypes;
  3. io.write / io.read of the SpatialAIR is lossless;
  4. pp.aggregate gives one observation per bin with a valid BCR (63,468 = QC_filter_summary.csv "After"), and
     io.to_spatialdata builds the SpatialData object.
Output: validation/results/m6_interop.csv (quantity, archived, package, match, note)
Usage : python -I m6_interop.py      (SPACLONE_DEV_SRC=<src> runs a development copy of the package)
"""

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

if os.environ.get("SPACLONE_DEV_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_DEV_SRC"])

import anndata as ad
import pandas as pd
import spaclone
from spaclone import io
from spaclone.pp import aggregate, define_clones, filter_records

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
rows = []
warnings.filterwarnings("ignore", message=".*Awkward Arrays.*")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
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


def same(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, check_exact=True)
        return 1
    except AssertionError as e:
        log(f"  frames differ: {str(e)[:400]}")
        return 0


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
a = pd.read_csv(
    E / "StereoIR" / "mm_spleen_bcr_airr.tsv.tsv", sep="\t", low_memory=False
)
xy = a.cell_id.str.split("-", expand=True).astype(float) * 0.5
air = a.assign(
    spot_id=a.cell_id.to_numpy(),
    x_um=xy[0].to_numpy(),
    y_um=xy[1].to_numpy(),
    library_id="mm_spleen",
)
log(
    f"StereoIR BCR: {len(air):,} contigs x {air.shape[1]} columns; dtypes {air.dtypes.astype(str).value_counts().to_dict()}"
)
sp = io.SpatialAIR(
    air,
    None,
    {
        "technology": "StereoIR",
        "unit_factor_um": 0.5,
        "source": "mm_spleen_bcr_airr.tsv",
    },
)

# ------------------------------------------------------------------ 1 scirpy round trips
tmp = Path(tempfile.mkdtemp(prefix="m6_interop_", dir=os.environ.get("TMPDIR", None)))
A_spot = io.to_scirpy(sp.air, level="spot")
add(
    "to_scirpy spot level: observations (bins with a contig)",
    65905,
    A_spot.n_obs,
    "QC_filter_summary.csv Before",
)
add(
    "spot level: SpatialAIR -> scirpy -> back lossless (1 = yes)",
    1,
    same(io.from_scirpy(A_spot), sp.air),
)
A_spot.write_h5ad(tmp / "spot.h5ad")
A_spot2 = ad.read_h5ad(tmp / "spot.h5ad")
add(
    "spot level: lossless after writing the AnnData to h5ad (1 = yes)",
    1,
    same(io.from_scirpy(A_spot2), sp.air),
)

ok, rep = filter_records(sp.air, valid_junction=True, productive=True, min_reads=2)
ok = define_clones(ok, method="exact")
ok = ok[ok.clone_id.notna()].copy()
log(f"valid contigs with a clone: {len(ok):,}")
A_loc = io.to_scirpy(ok, level="locus", radius_um=15)
add(
    "locus level: SpatialAIR -> scirpy -> back lossless (1 = yes)",
    1,
    same(io.from_scirpy(A_loc), ok),
)
A_loc.write_h5ad(tmp / "locus.h5ad")
add(
    "locus level: lossless after h5ad (1 = yes)",
    1,
    same(io.from_scirpy(ad.read_h5ad(tmp / "locus.h5ad")), ok),
)

# ------------------------------------------------------------------ 2 scirpy functions
import scirpy as ir

ir.pp.index_chains(A_loc)
ir.tl.clonal_expansion(A_loc, target_col="clone_id", breakpoints=(1, 2, 5))
ce = A_loc.obs.clonal_expansion.value_counts().sort_index()
log(f"scirpy clonal_expansion (loci per clone):\n{ce.to_string()}")
n_loc_clones = A_loc.obs.clone_id.nunique()
add(
    "scirpy tl.clonal_expansion ran on the locus-level AnnData: loci labelled",
    A_loc.n_obs,
    int(ce.sum()),
    "every locus gets a size class",
)
sizes = A_loc.obs.groupby("clone_id", observed=True).size()
add(
    "loci in clones of size 1 (scirpy '<= 1' class)",
    int((sizes == 1).sum()),
    int(ce.get("<= 1", 0)),
    "pandas count of loci per clone_id",
)
ir.pp.index_chains(A_spot)
ir.tl.chain_qc(A_loc)
log(
    "scirpy chain_qc on loci: "
    + str(A_loc.obs.chain_pairing.value_counts().to_dict())
    + "; receptor_type "
    + str(A_loc.obs.receptor_type.value_counts().to_dict())
)


def primary(adata, field):
    """Primary VJ / VDJ value of every observation, as scirpy indexes it."""
    vals = {
        arm: ir.get.airr(adata, field, f"{arm}_1").astype(object)
        for arm in ("VJ", "VDJ")
    }
    return pd.DataFrame(vals).where(pd.DataFrame(vals).notna(), None)


ir.tl.define_clonotypes(
    A_loc, key_added="scirpy_clone_id", receptor_arms="all", dual_ir="primary_only"
)
pn = primary(A_loc, "junction")
exp_nt = pn.assign(
    t=pn.receptor_type if "receptor_type" in pn else A_loc.obs.receptor_type.to_numpy()
).drop_duplicates()
add(
    "scirpy tl.define_clonotypes ran: clonotypes = distinct primary (VJ, VDJ) CDR3 nt per receptor type",
    len(exp_nt.dropna(how="all", subset=["VJ", "VDJ"])),
    A_loc.obs.scirpy_clone_id.nunique(),
    "scirpy clonotypes = identical nucleotide junctions of the primary chains",
)
ir.pp.ir_dist(A_loc, metric="identity", sequence="aa")
ir.tl.define_clonotype_clusters(
    A_loc,
    sequence="aa",
    metric="identity",
    receptor_arms="all",
    dual_ir="primary_only",
    key_added="scirpy_cc_aa",
)
pa = (
    primary(A_loc, "junction_aa")
    .assign(t=A_loc.obs.receptor_type.to_numpy())
    .drop_duplicates()
)
add(
    "scirpy tl.define_clonotype_clusters (aa identity) ran: clusters = distinct primary (VJ, VDJ) CDR3 aa",
    len(pa.dropna(how="all", subset=["VJ", "VDJ"])),
    A_loc.obs.scirpy_cc_aa.nunique(),
    "identity metric on CDR3 aa; V gene not used",
)
two = int((primary(A_loc, "junction_aa").notna().sum(axis=1) == 2).sum())
log(
    f"loci with both a VJ and a VDJ primary chain: {two:,}; package exact clones {n_loc_clones:,}"
)
multichain_spots = int(A_spot.obsm["chain_indices"]["multichain"].to_numpy().sum())
log(
    f"spot level: {multichain_spots:,} of {A_spot.n_obs:,} bins flagged multichain by scirpy"
)

# ------------------------------------------------------------------ 3 store round trip
io.write(sp, tmp / "store", overwrite=True)
back = io.read(tmp / "store")
add("io.write / io.read: receptor table lossless (1 = yes)", 1, same(back.air, sp.air))
add("io.write / io.read: meta kept (1 = yes)", 1, int(back.meta == sp.meta))

# ------------------------------------------------------------------ 4 aggregate and SpatialData
X = aggregate(ok)
add(
    "pp.aggregate: bins with a valid BCR contig (observations)",
    63468,
    X.n_obs,
    "QC_filter_summary.csv After",
)
add("pp.aggregate: clones (variables) = exact clones", ok.clone_id.nunique(), X.n_vars)
add("pp.aggregate: matrix total = valid contigs", len(ok), float(X.X.sum()))
sd = io.to_spatialdata(ok)
add(
    "io.to_spatialdata: points = valid contigs",
    len(ok),
    int(sd.points["air_records"].shape[0].compute()),
)
add("io.to_spatialdata: spot shapes = bins", 63468, len(sd.shapes["air_spots"]))

R = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
R.to_csv(OUT / "m6_interop.csv", index=False)
log(f"{int(R.match.sum())} / {len(R)} quantities match")
print(R.to_string(index=False), flush=True)
sys.exit(0 if R.match.all() else 1)
