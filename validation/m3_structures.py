"""M3 validation: tissue structures of the StereoIR spleen (archive T_common.py / qc_common.py) from package functions.

Recomputes with spaclone.tl only (Grid, mask_from_types, zones, compartments, signed_distance, nearest_zone,
tissue_mask, Raster.sample):
  1. masks: B-rich, T-rich and erythrocyte-rich areas of the cell2location-typed bin20 (StereoIR/mm_spleen_*.tsv),
     pixel by pixel against the archive's own qc_common.b_mask run on the same bins (the archive saved no masks);
  2. T zones = connected components of the T-rich mask: T1_summary.csv "T zones" (19);
  3. compartments (T zone > red pulp (erythrocyte-rich or outside the B-rich area) > follicle): tissue area of each
     compartment (T1_summary.csv) and the compartment of every T1 IGH locus and TR1 TCR locus;
  4. per locus: nearest T zone, distance to the T-zone border (dT, 0 inside), signed depth in the B-rich area
     (depthB), the T zone holding the locus (zone_in, TR1), the signed white-pulp depth (wp_depth, TR1) and on_grid,
     against T1_igh_loci.parquet and TR1_tcr_loci.parquet (31,516 and 28,958 loci).
Output: validation/results/m3_structures.csv (quantity, archived, package, match, note)
Usage : python -I m3_structures.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import spaclone
from spaclone import tl

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
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
    """A reported (not required to match) number: the package default where it deliberately differs."""
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def typed_bins():
    """cell2location-typed bin20 of the StereoIR section (later lists override earlier ones, as in TR1 / Q0)."""
    lab = {}
    for f, c in (("bcell", "B lineage"), ("tcell", "T"), ("erythrocyte", "erythrocyte")):
        for b in open(E / "StereoIR" / f"mm_spleen_{f}.tsv").read().split():
            lab[b] = c
    ids = sorted(lab)
    xy = np.array([list(map(int, i.split("-"))) for i in ids], dtype=float) * 0.5  # DNB -> um
    return pd.DataFrame({"bin": ids, "x_um": xy[:, 0], "y_um": xy[:, 1], "cell_type": [lab[i] for i in ids]})


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
bins = typed_bins()
M = pd.read_parquet(E / "results" / "QC" / "maps" / "StereoIR__mm_spleen.parquet", columns=["x_um", "y_um", "cls"])
Mt = M[M.cls != "untyped"]
key = lambda d, c: set(zip(d.x_um.round(1), d.y_um.round(1), d[c], strict=True))  # noqa: E731
add("typed bins (TSV lists)", len(Mt), len(bins), "StereoIR__mm_spleen.parquet cls != untyped")
add("typed bins with the same position and class as the archive map table (1 = all)", 1,
    int(key(Mt, "cls") == key(bins, "cell_type")))

grid = tl.Grid.from_points(bins.x_um, bins.y_um, pixel_um=10.0)
CLASSES = ("B lineage", "T", "erythrocyte")
# legacy-equivalent: scipy's border handling in the closing (pad_edges=False), as qc_common.b_mask
masks = {c: tl.mask_from_types(bins, types=c, type_key="cell_type", grid=grid, pad_edges=False) for c in CLASSES}
log(f"grid {grid}")

# ---- 1. masks against the archive's own b_mask on the same bins
sys.dont_write_bytecode = True  # leave no __pycache__ behind in the archive's workflow folder
sys.path.insert(0, str(E / "workflow" / "qc"))
from qc_common import b_mask  # noqa: E402  (archive reference implementation, read-only)

xy = bins[["x_um", "y_um"]].to_numpy()
for c, m in masks.items():
    ref, lo = b_mask(xy, (bins.cell_type == c).to_numpy())
    same_grid = ref.shape == grid.shape and np.allclose(lo, grid.origin, rtol=0, atol=0)
    add(f"{c}-rich mask: grid identical to b_mask (1 = yes)", 1, int(same_grid))
    add(f"{c}-rich mask: pixels differing from qc_common.b_mask", 0, int((ref != m.values).sum()),
        f"{int(ref.sum()):,} mask pixels")
arch = np.load(E / "results" / "QC" / "StereoIR_clones" / "T3_csr_maps.npz")
add("grid origin x (um) = T3_csr_maps.npz lo", float(arch["lo"][0]), grid.origin[0])
add("grid origin y (um) = T3_csr_maps.npz lo", float(arch["lo"][1]), grid.origin[1])
add("grid pixels along x = T3_csr_maps.npz", arch["hotspot"].shape[0], grid.shape[0])
add("grid pixels along y = T3_csr_maps.npz", arch["hotspot"].shape[1], grid.shape[1])

mB, mT, mE = masks["B lineage"], masks["T"], masks["erythrocyte"]
zones = tl.zones(mT)
near = tl.nearest_zone(zones)
comp = tl.compartments([("T zone", mT), ("red pulp", mE), ("red pulp", ~mB)], default="follicle")
dT = np.maximum(tl.signed_distance(mT).values, 0.0)
dT_r = tl.Raster(dT, grid)
depthB = tl.Raster(-tl.signed_distance(mB).values, grid)
wpd = tl.Raster(-tl.signed_distance(mB | mT).values, grid)

# ---- 2. T zones
T1s = pd.read_csv(E / "results" / "QC" / "StereoIR_clones" / "T1_summary.csv").set_index("item").value
add("T zones (connected components of the T-rich mask)", T1s["T zones"], int(zones.values.max()), "T1_summary.csv")

# ---- 3. compartment areas inside the typed-bin pixels
pix = tl.tissue_mask(bins, grid=grid, close_iter=0, fill_holes=False).values
for c in ("T zone", "follicle", "red pulp"):
    k = comp.categories.index(c)
    add(f"tissue area mm2 {c}", T1s[f"tissue area mm2 {c}"], float(((comp.values == k) & pix).sum() * 1e-4),
        "T1_summary.csv; typed-bin pixels", tol=1e-12)


# ---- 4. per-locus lookups
def compare_loci(name, L, cols):
    x, y = L.x.to_numpy(), L.y.to_numpy()
    got = {
        "comp": comp.sample_categories(x, y, fill="red pulp"),
        "zone": near.sample(x, y, fill=0),
        "dT": dT_r.sample(x, y, fill=np.inf),
        "depthB": depthB.sample(x, y, fill=-np.inf),
        "zone_in": np.where(mT.sample(x, y, fill=False), zones.sample(x, y, fill=0), 0),
        "wp_depth": wpd.sample(x, y, fill=-np.inf),
        "on_grid": grid.index(x, y)[2],
    }
    for c in cols:
        a, p = L[c].to_numpy(), got[c]
        if a.dtype.kind == "f":
            diff = int((~((a == p) | (np.isclose(a, p, rtol=1e-12, atol=1e-9)))).sum())
        else:
            diff = int((a.astype(str) != p.astype(str)).sum())
        add(f"{name}: loci whose {c} differs from the archive", 0, diff, f"{len(L):,} loci")


Lo = pd.read_parquet(E / "results" / "QC" / "StereoIR_clones" / "T1_igh_loci.parquet")
compare_loci("T1 IGH", Lo, ["comp", "zone", "dT", "depthB"])
TR = pd.read_parquet(E / "results" / "QC" / "StereoIR_TCR" / "TR1_tcr_loci.parquet")
compare_loci("TR1 TCR", TR, ["comp", "zone_in", "zone", "dT", "depthB", "wp_depth", "on_grid"])
for c in ("T zone", "follicle", "red pulp"):
    log(f"T1 loci in {c}: archive {(Lo.comp == c).sum():,}")

# ---- 5. the package default (pad_edges=True: mask pixels on the grid edge survive the closing)
for c in CLASSES:
    d = tl.mask_from_types(bins, types=c, type_key="cell_type", grid=grid)
    gained = int((d.values & ~masks[c].values).sum())
    lost = int((~d.values & masks[c].values).sum())
    info(f"{c}-rich mask, package default pad_edges=True: pixels gained / lost vs the archive", 0, gained + lost,
         f"{gained} gained, {lost} lost of {int(masks[c].values.sum()):,}")
dT_def = tl.mask_from_types(bins, types="T", type_key="cell_type", grid=grid)
info("T zones with the package default pad_edges=True", T1s["T zones"], int(tl.zones(dT_def).values.max()),
     "edge pixels only; the archive numbers use pad_edges=False")
t_def = tl.tissue_mask(bins, grid=grid)
t_leg = tl.tissue_mask(bins, grid=grid, pad_edges=False)
info("typed-bin tissue mask (closed, filled), pad_edges=True vs False: pixels differing", 0,
     int((t_def.values != t_leg.values).sum()), f"{int(t_leg.values.sum()):,} tissue pixels (legacy)")

res = pd.DataFrame(rows)
res.to_csv(OUT / "m3_structures.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm3_structures.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")
