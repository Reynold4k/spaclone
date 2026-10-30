"""M3 validation: StereoIR CSR hotspots (archive T3) and the isotype-matched SHM test of those hotspots (archive T7).

Recomputes with spaclone.tl only (Grid, tissue_mask, mask_from_types, compartments, zones, nearest_zone,
signed_distance, hotspots, region_excess):
  T3  isotyped T1 IGH loci on the typed-bin grid; switched share map (sigma 50 um, n_eff >= 10 on the typed-bin tissue);
      1,000 shuffles of the switched label; hotspots above the 95th percentile of the shuffled map maxima.
      Legacy-equivalent settings: neff_kernel="legacy" (the archive's sigma/sqrt(2) approximation of sum w^2),
      threshold_rule="quantile" (numpy.quantile of the maxima), rng = np.random.default_rng(0) with the archive's
      draw order (one permutation per shuffle) -> must match exactly: p0, threshold, 13 hotspots with areas, loci,
      maxima, compartment shares, the saved maps (T3_csr_maps.npz) and the sigma 30 / 100 sensitivity runs (after
      replaying the 100 clone-label shuffles the archive drew in between).
      Improved settings (package defaults: exact n_eff kernel, rank threshold rule; same seed) reported alongside.
      Clone-level null (fix F4b, 2026-10-30; reported): blocks="clone" with the matched block design (clones trade
      switched / mutated calls only with clones of the same size and isotype mix, calls re-dealt inside a clone) and
      the permutation-studentised map; the archive's unit-level shuffles count the loci of one clone as independent.
  T7  evaluable T1 loci; SHM share map with a raw and an isotype-matched expectation; 1,000 shuffles each; then the
      CSR test (mutated loci in each CSR hotspot vs the isotype-matched expectation, 10,000 within-isotype shuffles).
      The archive drew the raw and the within-isotype shuffles interleaved from one generator, so the exact replay
      passes those draws as null_marks (same generator, same order) and continues the generator into region_excess;
      the package's own draws (rng = default_rng(0), no replay) are reported as well, with a Monte-Carlo check.
Output: validation/results/m3_csr_shm.csv (quantity, archived, package, match, note)
Usage : python -I m3_csr_shm.py   (SPACLONE_SRC=<dev src> to test a scratch copy)
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

import spaclone  # noqa: E402
from spaclone import tl  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RES = E / "results" / "QC" / "StereoIR_clones"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
rows = []
FP = 1e-9  # relative tolerance for floating-point reassociation (z = (S - E) / sqrt(V) vs (share - p0) / se)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol * max(1.0, abs(a)))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}")


def info(quantity, archived, package, note):
    """A reported (not required to match) number: improved settings or Monte-Carlo comparisons."""
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def typed_bins():
    lab = {}
    for f, c in (("bcell", "B lineage"), ("tcell", "T"), ("erythrocyte", "erythrocyte")):
        for b in open(E / "StereoIR" / f"mm_spleen_{f}.tsv").read().split():
            lab[b] = c
    ids = sorted(lab)
    xy = np.array([list(map(int, i.split("-"))) for i in ids], dtype=float) * 0.5
    return pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "cell_type": [lab[i] for i in ids]})


def same_map(a, b):
    """Max |a - b| / max(1, |a|) on pixels finite in both, and whether the NaN patterns agree."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not (fa == fb).all():
        return np.inf, False
    d = np.abs(a[fa] - b[fa]) / np.maximum(1.0, np.abs(a[fa]))
    return (float(d.max()) if d.size else 0.0), True


def quantile_se(null, q=0.95):
    """Monte-Carlo SE of the q-quantile of the null draws, from the binomial order-statistic 95% CI (half-width / 1.96)."""
    v = np.sort(np.asarray(null, float))
    n = len(v)
    h = 1.959964 * np.sqrt(n * q * (1 - q))
    return (v[min(int(np.ceil(n * q + h)), n - 1)] - v[max(int(np.floor(n * q - h)), 0)]) / (2 * 1.959964)


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
bins = typed_bins()
grid = tl.Grid.from_points(bins.x_um, bins.y_um, pixel_um=10.0)
# legacy-equivalent structures (pad_edges=False: scipy's closing at the grid edge, as the archive)
tissue = tl.tissue_mask(bins, grid=grid, pad_edges=False)  # typed-bin pixels, closed twice, holes filled (T3 / T7)
mB, mT, mE = (tl.mask_from_types(bins, types=c, grid=grid, pad_edges=False) for c in ("B lineage", "T", "erythrocyte"))
tissue_default = tl.tissue_mask(bins, grid=grid)  # package default, for the improved runs
comp = tl.compartments([("T zone", mT), ("red pulp", mE), ("red pulp", ~mB)], default="follicle")
near = tl.nearest_zone(tl.zones(mT))
dT = np.maximum(tl.signed_distance(mT).values, 0.0)
Lo = pd.read_parquet(RES / "T1_igh_loci.parquet")
on = lambda d: grid.index(d.x.to_numpy(), d.y.to_numpy())[2]  # noqa: E731

# =============================================================================================== T3 legacy-equivalent
I = Lo[Lo.isotyped > 0]
info("T3 isotyped loci off the typed-bin grid (left out, as in T3)", np.nan, int((~on(I)).sum()),
     "the archive printed this count only; the isotyped-loci row below checks the remainder")
I = I[on(I)].reset_index(drop=True)
S3 = pd.read_csv(RES / "T3_csr_summary.csv").set_index("item").value
H3 = pd.read_csv(RES / "T3_csr_hotspots.csv")
npz3 = np.load(RES / "T3_csr_maps.npz")
g = np.random.default_rng(0)
kw = {"sigma_um": 50.0, "tissue": tissue, "min_neff": 10.0, "x_key": "x", "y_key": "y"}
legacy = {"neff_kernel": "legacy", "threshold_rule": "quantile"}
hs = tl.hotspots(I, mark="switched", n_perm=1000, rng=g, **kw, **legacy)
log(repr(hs))
add("T3 isotyped loci", S3["isotyped loci"], hs.summary["n_units"])
add("T3 switched share p0", S3["switched share p0"], hs.summary["p0"])
add("T3 z threshold (95th pct of shuffled map maxima)", S3["z threshold"], hs.threshold["hot"], "", tol=FP)
add("T3 hotspots", S3["hotspots"], hs.summary["n_hotspots"])
add("T3 hotspot area mm2", S3["hotspot area mm2"], hs.table.area_um2.sum() / 1e6, "", tol=1e-12)
add("T3 shuffled maps with any hotspot", S3["shuffled maps with any hotspot"], hs.summary["null_any_hot"])
d, nanok = same_map(npz3["neff"], hs.map.n_eff)
add("T3 n_eff map: max relative difference to T3_csr_maps.npz", 0.0, d if nanok else np.inf, "0 = bitwise", tol=1e-12)
d, nanok = same_map(npz3["share"], hs.map.share)
add("T3 share map: max relative difference (finite pixels; NaN pattern equal)", 0.0, d if nanok else np.inf, "", tol=1e-12)
elig = hs.map.eligible
dz = float(np.max(np.abs(npz3["z"][elig] - hs.map.z[elig]) / np.maximum(1.0, np.abs(npz3["z"][elig]))))
add("T3 z map on eligible pixels: max relative difference", 0.0, dz, "floating-point reassociation", tol=FP)
add("T3 hotspot label raster identical to T3_csr_maps.npz (1 = yes)", 1, int(np.array_equal(npz3["hotspot"], hs.labels.values)))
lab = hs.labels.values
for r in H3.itertuples():
    p = hs.table.set_index("hotspot").loc[r.hotspot]
    m = lab == r.hotspot
    k = r.hotspot
    add(f"T3 hotspot {k} area mm2", r.area_mm2, p.area_um2 / 1e6, "", tol=1e-12)
    add(f"T3 hotspot {k} isotyped loci", r.isotyped_loci, p.units)
    add(f"T3 hotspot {k} switched loci", r.switched_loci, p.marked)
    add(f"T3 hotspot {k} max z", r.max_z, p.peak_z, "", tol=FP)
    for c in ("T zone", "follicle", "red pulp"):
        share_c = float((comp.values[m] == comp.categories.index(c)).mean())
        add(f"T3 hotspot {k} share {c}", getattr(r, f"share_{c.replace(' ', '_')}"), share_c, "", tol=1e-12)
    add(f"T3 hotspot {k} median distance to T zone (um)", r.median_dist_to_T_zone_um, float(np.median(dT[m])), "",
        tol=1e-12)
    add(f"T3 hotspot {k} nearest T zone", r.nearest_T_zone, int(pd.Series(near.values[m]).mode().iat[0]))

# ---- sensitivity (sigma 30 / 100, 200 shuffles); the archive drew 100 clone-label shuffles before them
n_iso_all = int((Lo.isotyped > 0).sum())
for _ in range(100):
    g.permutation(n_iso_all)
for sg in (30.0, 100.0):
    hs_s = tl.hotspots(I, mark="switched", n_perm=200, rng=g, **{**kw, "sigma_um": sg}, **legacy)
    tag = f"sensitivity sigma {int(sg)}"
    add(f"T3 {tag} threshold_z", S3[f"{tag} threshold_z"], hs_s.threshold["hot"], "", tol=FP)
    add(f"T3 {tag} hotspots", S3[f"{tag} hotspots"], hs_s.summary["n_hotspots"])
    add(f"T3 {tag} area_mm2", S3[f"{tag} area_mm2"], hs_s.table.area_um2.sum() / 1e6, "", tol=1e-12)
    cov = float(((hs_s.labels.values > 0) & (lab > 0)).sum() / max((lab > 0).sum(), 1))
    add(f"T3 {tag} share of main hotspot area covered", S3[f"{tag} share_of_main_hotspot_area_covered"], cov, "",
        tol=1e-12)

# ---- improved settings (same seed and draws): which change moves what
for name, opts in (
    ("exact n_eff kernel, quantile rule", {"neff_kernel": "exact", "threshold_rule": "quantile"}),
    ("legacy n_eff kernel, rank rule", {"neff_kernel": "legacy", "threshold_rule": "rank"}),
    ("package defaults (exact kernel, rank rule, padded tissue edge)", {"tissue": tissue_default}),
):
    h2 = tl.hotspots(I, mark="switched", n_perm=1000, rng=0, **{**kw, **opts})
    ov = float(((h2.labels.values > 0) & (lab > 0)).sum() / max((lab > 0).sum(), 1))
    info(f"T3 improved [{name}]: hotspots", S3["hotspots"], h2.summary["n_hotspots"],
         f"threshold {h2.threshold['hot']:.4f}, eligible pixels {h2.summary['n_eligible']} (legacy "
         f"{hs.summary['n_eligible']}), hotspot area {h2.table.area_um2.sum() / 1e6:.4f} mm2, "
         f"{ov:.1%} of the archive's hotspot area kept")

# =============================================================================================== T7
S7 = pd.read_csv(RES / "T7_summary.csv").iloc[0]
H7 = pd.read_csv(RES / "T7_shm_hotspots.csv")
C7 = pd.read_csv(RES / "T7_csr_hotspot_shm_test.csv")
npz7 = np.load(RES / "T7_shm_maps.npz")
L = Lo[Lo.mut_eval.astype(bool)]
L = L[on(L)].reset_index(drop=True)
L["iso"] = L.iso_main.fillna("unknown")
m = L.mutated.to_numpy().astype(float)
classes = [np.flatnonzero(L.iso.to_numpy() == c) for c in sorted(L.iso.unique())]
add("T7 evaluable loci", S7.evaluable_loci, len(L))
add("T7 mutated share p0", S7.p0, m.mean())

# ---- exact replay of the archive's interleaved draws (raw permutation, then one permutation per isotype class)
g = np.random.default_rng(0)
null_raw = np.empty((1000, len(L)), dtype=bool)
null_iso = np.empty((1000, len(L)), dtype=bool)
for b in range(1000):
    null_raw[b] = m[g.permutation(len(m))] > 0.5
    out = m.copy()
    for idx in classes:
        out[idx] = m[idx][g.permutation(len(idx))]
    null_iso[b] = out > 0.5
raw = tl.hotspots(L, mark="mutated", null_marks=null_raw, **kw, **legacy)
iso = tl.hotspots(L, mark="mutated", match_by="iso", null_marks=null_iso, **kw, **legacy)
log(repr(raw))
log(repr(iso))
add("T7 raw threshold", S7.thr_raw, raw.threshold["hot"], "replayed draws", tol=FP)
add("T7 isotype-matched threshold", S7.thr_iso, iso.threshold["hot"], "replayed draws", tol=FP)
add("T7 SHM hotspots (raw)", S7.shm_hotspots_raw, raw.summary["n_hotspots"], "replayed draws")
add("T7 SHM hotspots (isotype-matched)", S7.shm_hotspots_iso, iso.summary["n_hotspots"], "replayed draws")
add("T7 max z (raw)", S7.max_z_raw, raw.summary["max_z"], "", tol=FP)
add("T7 max z (isotype-matched)", S7.max_z_iso, iso.summary["max_z"], "", tol=FP)
for key, arr in (("neff", raw.map.n_eff), ("share", raw.map.share)):
    d, nanok = same_map(npz7[key], arr)
    add(f"T7 {key} map: max relative difference to T7_shm_maps.npz", 0.0, d if nanok else np.inf, "", tol=1e-12)
for key, hres in (("z_raw", raw), ("z_iso", iso)):
    el = hres.map.eligible
    dz = float(np.max(np.abs(npz7[key][el] - hres.map.z[el]) / np.maximum(1.0, np.abs(npz7[key][el]))))
    add(f"T7 {key} map on eligible pixels: max relative difference", 0.0, dz, "", tol=FP)
add("T7 SHM hotspot labels (raw) identical to T7_shm_maps.npz (1 = yes)", 1,
    int(np.array_equal(npz7["hotspot"], raw.labels.values)))
add("T7 SHM hotspot labels (isotype-matched) identical (1 = yes)", 1,
    int(np.array_equal(npz7["hotspot_iso"], iso.labels.values)))
for r in H7.itertuples():
    hres = raw if r.kind == "raw" else iso
    p = hres.table.set_index("hotspot").loc[r.hotspot]
    mk = hres.labels.values == r.hotspot
    tag = f"T7 SHM hotspot {r.kind} {r.hotspot}"
    add(f"{tag} area mm2", r.area_mm2, p.area_um2 / 1e6, "", tol=1e-12)
    add(f"{tag} evaluable loci", r.evaluable_loci, p.units)
    add(f"{tag} mutated loci", r.mutated_loci, p.marked)
    add(f"{tag} share of pixels in CSR hotspots", r.share_in_CSR_hotspots, float((lab[mk] > 0).mean()), "", tol=1e-12)

# ---- CSR test (continues the archive's generator after the 2 x 1,000 map shuffles)
csr = tl.region_excess(L, mark="mutated", regions=hs.labels, match_by="iso", n_perm=10_000, min_units=5, rng=g,
                       x_key="x", y_key="y")
log(repr(csr))
P = csr.table.assign(key=csr.table.region.astype(str)).set_index("key")
add("T7 CSR test rows (hotspots with >= 5 evaluable loci + pooled)", len(C7), len(P))
for r in C7.itertuples():
    p = P.loc[str(r.csr_hotspot)]
    tag = f"T7 CSR test {r.csr_hotspot}"
    add(f"{tag} evaluable loci", r.evaluable_loci, p.units)
    add(f"{tag} mutated loci", r.mutated_loci, p.marked)
    add(f"{tag} isotype-matched expectation", r.expected_isotype_matched, p.expected, "", tol=1e-12)
    add(f"{tag} raw expectation", r.expected_raw, p.expected_raw, "", tol=1e-12)
    add(f"{tag} ratio to the isotype-matched expectation", r.ratio_isotype_matched, p.ratio, "", tol=1e-12)
    add(f"{tag} null 2.5%", r.null_lo, p.null_lo, "replayed draws")
    add(f"{tag} null 97.5%", r.null_hi, p.null_hi, "replayed draws")
    add(f"{tag} p (10,000 within-isotype shuffles)", r.p_isotype_matched, p.p_value, "replayed draws", tol=1e-12)
    if np.isfinite(r.q):
        add(f"{tag} q (BH over hotspots)", r.q, p.q_value, "replayed draws", tol=1e-12)

# ---- the package's own draws (no replay): differences must be Monte-Carlo error only
g0 = np.random.default_rng(0)  # one generator, used in the package's natural order: raw map, iso map, CSR test
raw_n = tl.hotspots(L, mark="mutated", n_perm=1000, rng=g0, **kw, **legacy)
iso_n = tl.hotspots(L, mark="mutated", match_by="iso", n_perm=1000, rng=g0, **kw, **legacy)
for nm, own, rep, thr in (("raw", raw_n, raw, S7.thr_raw), ("isotype-matched", iso_n, iso, S7.thr_iso)):
    # two independent estimates of the same 95th percentile: their difference has SE sqrt(2) x SE(quantile)
    zq = (own.threshold["hot"] - thr) / (np.sqrt(2) * quantile_se(rep.null))
    ks = ks_2samp(own.null, rep.null).pvalue
    add(f"T7 {nm} threshold, own draws, within 3 Monte-Carlo SE of the archive's (1 = yes)", 1, int(abs(zq) <= 3),
        f"own {own.threshold['hot']:.4f}, archive {thr:.4f}, difference {zq:+.2f} SE; shuffled maxima, own vs "
        f"archive: two-sample KS p = {ks:.2f}")
    info(f"T7 {nm} SHM hotspots, own draws", rep.summary["n_hotspots"], own.summary["n_hotspots"],
         "hotspots near the threshold can flip with the Monte-Carlo threshold")
csr_n = tl.region_excess(L, mark="mutated", regions=hs.labels, match_by="iso", n_perm=10_000, min_units=5, rng=g0,
                         x_key="x", y_key="y")
Pn = csr_n.table.assign(key=csr_n.table.region.astype(str)).set_index("key")
for r in C7.itertuples():
    p = Pn.loc[str(r.csr_hotspot)]
    se = np.sqrt(max(r.p_isotype_matched * (1 - r.p_isotype_matched), 1e-4 * (1 - 1e-4)) / 10_000)
    zdiff = abs(p.p_value - r.p_isotype_matched) / (np.sqrt(2) * se)
    add(f"T7 CSR test {r.csr_hotspot} p, own draws, within 3 Monte-Carlo SE of the archive (1 = yes)", 1,
        int(zdiff <= 3), f"own p {p.p_value:.4g}, archive {r.p_isotype_matched:.4g}, |diff| = {zdiff:.2f} SE")

# =============================================================================== clone-level null (F4b, reported)
# the archived loci are cell-scale (no two loci of a clone within 15 um), so the clone-level design applies as it is
kw_c = {"sigma_um": 50.0, "tissue": tissue_default, "min_neff": 10.0, "x_key": "x", "y_key": "y"}
hc = tl.hotspots(I, mark="switched", blocks="clone", n_perm=1000, rng=0, **kw_c)
ov = float(((hc.labels.values > 0) & (lab > 0)).sum() / max((lab > 0).sum(), 1))
info("T3 clone-level null [blocks='clone', matched, studentised; exact kernel, rank rule]: hotspots", S3["hotspots"],
     hc.summary["n_hotspots"],
     f"threshold {hc.threshold['hot']:.3f} on the studentised map (max {hc.summary['max_z']:.2f}); hotspot area "
     f"{hc.table.area_um2.sum() / 1e6:.4f} mm2, {ov:.1%} of the archive's hotspot area; "
     f"{hc.params['design']['n_exchangeable_blocks']} of {I.clone.nunique()} clones tradeable")  # fmt: skip
if len(hc.table):
    t1 = hc.table.iloc[0]
    info("T3 clone-level null: largest hotspot loci / switched / p", np.nan, int(t1.units),
         f"{int(t1.marked)} switched, map-level p {t1.p_value:.4f}")  # fmt: skip
hi = tl.hotspots(L, mark="mutated", match_by="iso", blocks="clone", n_perm=1000, rng=0, **kw_c)
info("T7 clone-level null [isotype strata, matched, studentised]: SHM hotspots", S7.shm_hotspots_iso,
     hi.summary["n_hotspots"],
     f"threshold {hi.threshold['hot']:.3f}, max studentised z {hi.summary['max_z']:.2f}, map-level p "
     f"{hi.summary['p_map_hot']:.4f} (archive, loci shuffled within isotype: 1 hotspot)")  # fmt: skip
cc = tl.region_excess(L, mark="mutated", regions=hs.labels, match_by="iso", blocks="clone", n_perm=10_000,
                      min_units=5, rng=0, x_key="x", y_key="y")  # fmt: skip
pc = cc.table.assign(key=cc.table.region.astype(str)).set_index("key").loc["all"]
info("T7 CSR test pooled, clone-level null: mutated loci", 90, pc.marked,
     f"isotype-matched expectation {pc.expected:.2f} (archive 56.17); null mean {pc.null_mean:.2f} (isotype and "
     f"clone size matched), sd {pc.null_sd:.2f}, ratio to null mean {pc.marked / pc.null_mean:.2f}, p "
     f"{pc.p_value:.4g} (archive, loci shuffled: 1e-4)")  # fmt: skip
for r in cc.table.itertuples():
    if str(r.region) != "all":
        info(f"T7 CSR test {r.region}, clone-level null: p", np.nan, r.p_value,
             f"{int(r.marked)} mutated of {int(r.units)}; null mean {r.null_mean:.2f}; q {r.q_value:.3g}")  # fmt: skip

res = pd.DataFrame(rows)
res.to_csv(OUT / "m3_csr_shm.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm3_csr_shm.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")
