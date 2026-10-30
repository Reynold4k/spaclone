"""M3 validation: StereoIR TCR niche profiles, archive TR3 (iNKT alpha loci by compartment and across the white-pulp
border) and TR7 (iNKT and expanded-clone profiles along the T-zone and white-pulp borders).

Recomputes with spaclone.tl only (mask_from_types, compartments, signed_distance, niche_profile):
  TR3  TRA loci of TR1 on the typed-bin grid; iNKT locus (TRAV11-TRAJ18) vs every other non-MAIT TRA locus;
       iNKT share in T zone / follicle / red pulp against 10,000 random draws of the same number of TRA loci from
       iNKT + reference (pool="union"), the profile across the white-pulp border (25-um bins, -150 to +250 um), and
       the same compartment test on loci with >= 20 reads (the archive continued the same generator).
       Draws: rng = np.random.default_rng(0) in the archive's order (one choice without replacement per draw), so
       shares, chance means and quantiles must match exactly. The archive's two-sided p counted draws at least as
       far from the draw mean ("legacy p"); the package's p doubles the smaller one-sided p. Both are reported;
       the legacy p is recomputed from the package's null draws.
  TR7  profiles along the signed T-zone distance (negative inside) and the white-pulp depth, 25-um bins from -150 to
       +300 um, for iNKT (all / >= 20 reads, baseline = other alpha loci) and the TRB member loci of expanded
       clonotypes (one / several rearrangements, baseline = all TRB member loci); band = 2.5-97.5% of 5,000 draws of
       the group size from the baseline (pool="baseline"), all eight profiles from one generator in the archive's
       order.
Output: validation/results/m3_niche.csv (quantity, archived, package, match, note)
Usage : python -I m3_niche.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import spaclone
from spaclone import tl

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RES = E / "results" / "QC" / "StereoIR_TCR"
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
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def legacy_p(obs, draws):
    """TR3's two-sided p: draws at least as far from the draw mean as the observed value."""
    mu = draws.mean()
    return (1 + np.sum(np.abs(draws - mu) >= abs(obs - mu) - 1e-12)) / (1 + len(draws))


def typed_bins():
    lab = {}
    for f, c in (("bcell", "B lineage"), ("tcell", "T"), ("erythrocyte", "erythrocyte")):
        for b in open(E / "StereoIR" / f"mm_spleen_{f}.tsv").read().split():
            lab[b] = c
    ids = sorted(lab)
    xy = np.array([list(map(int, i.split("-"))) for i in ids], dtype=float) * 0.5
    return pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "cell_type": [lab[i] for i in ids]})


TOL = 1e-12
log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
bins = typed_bins()
grid = tl.Grid.from_points(bins.x_um, bins.y_um, pixel_um=10.0)
# legacy-equivalent structures (pad_edges=False: scipy's closing at the grid edge, as the archive's T_common)
mB, mT, mE = (tl.mask_from_types(bins, types=c, grid=grid, pad_edges=False) for c in ("B lineage", "T", "erythrocyte"))
comp = tl.compartments([("T zone", mT), ("red pulp", mE), ("red pulp", ~mB)], default="follicle")
tz = tl.signed_distance(mT)  # negative inside a T zone
wpd = tl.Raster(-tl.signed_distance(mB | mT).values, grid)  # + inside the white pulp (B-rich or T-rich)

Lo = pd.read_parquet(RES / "TR1_tcr_loci.parquet")
Lo = Lo[Lo.on_grid].reset_index(drop=True)
x, y = Lo.x.to_numpy(), Lo.y.to_numpy()
Lo["comp_pkg"] = comp.sample_categories(x, y, fill="red pulp")
Lo["wp_pkg"] = wpd.sample(x, y, fill=-np.inf)
Lo["tz_pkg"] = tz.sample(x, y, fill=np.inf)
add("TR1 loci on the grid whose package compartment differs from TR1 comp", 0, int((Lo.comp_pkg != Lo.comp).sum()))
add("TR1 loci on the grid whose package wp_depth differs from TR1 wp_depth", 0,
    int((~np.isclose(Lo.wp_pkg, Lo.wp_depth, rtol=TOL, atol=1e-9)).sum()))

# =============================================================================================== TR3
A = Lo[Lo.chain == "TRA"].reset_index(drop=True)
iN, iM = A.inkt.to_numpy(), A.mait.to_numpy()
ref = ~iN & ~iM
S3 = pd.read_csv(RES / "TR3_summary.csv").set_index("item").value
C3 = pd.read_csv(RES / "TR3_compartments.csv").set_index("compartment")
P3 = pd.read_csv(RES / "TR3_wp_profile.csv")
F3 = pd.read_csv(RES / "TR3_sensitivity_floor20.csv").set_index("readout")
g = np.random.default_rng(0)
# legacy (archive TR3): loci exchanged one by one; iNKT alpha loci are independent cells sharing an invariant chain
r = tl.niche_profile(A, group=iN, baseline=ref, compartment="comp_pkg", pool="union", n_perm=10_000, rng=g,
                     blocks=None)
log(repr(r))
add("TR3 iNKT loci", S3["iNKT loci"], r.summary["n_group"])
add("TR3 MAIT loci", S3["MAIT loci"], int(iM.sum()))
add("TR3 reference TRA loci", S3["reference TRA loci"], r.summary["n_baseline"])
t = r.table.set_index("label")
for c, a in C3.iterrows():
    p = t.loc[c]
    k = int(p.band)
    add(f"TR3 {c}: iNKT loci", a.iNKT_loci, p.group_units)
    add(f"TR3 {c}: iNKT share", a.iNKT_share, p.statistic, "", tol=TOL)
    add(f"TR3 {c}: reference share", a.reference_share, p.expected, "", tol=TOL)
    add(f"TR3 {c}: chance mean (10,000 draws)", a.chance_mean, p.null_mean, "", tol=TOL)
    add(f"TR3 {c}: chance 2.5%", a.chance_lo, p.null_lo, "", tol=TOL)
    add(f"TR3 {c}: chance 97.5%", a.chance_hi, p.null_hi, "", tol=TOL)
    add(f"TR3 {c}: ratio to chance", a.ratio_to_chance, p.statistic / p.null_mean, "", tol=TOL)
    add(f"TR3 {c}: two-sided p (legacy rule, from the package's null draws)", a.p_two_sided,
        legacy_p(p.statistic, r.null[:, k]), "", tol=TOL)
    info(f"TR3 {c}: two-sided p (package rule: twice the smaller one-sided p)", a.p_two_sided, p.p_value,
         f"family-wise over the 3 compartments p_maxT = {p.p_maxT:.4g}")
info("TR3 global test (max-T over compartments)", np.nan, r.summary["global_p"], "not in the archive")

# white-pulp profile: same 10,000 draws (fresh generator with the same seed)
rw = tl.niche_profile(A, group=iN, baseline=ref, distance_um="wp_pkg", bands_um=np.arange(-150, 251, 25.0),
                      n_perm=10_000, rng=np.random.default_rng(0), blocks=None)
tw = rw.table
add("TR3 white-pulp profile bins", len(P3), len(tw))
for (_, a), (_, p) in zip(P3.iterrows(), tw.iterrows(), strict=True):
    tag = f"TR3 white-pulp bin {a.wp_depth_mid:g} um"
    add(f"{tag}: iNKT loci", a.iNKT_loci, p.group_units)
    add(f"{tag}: reference loci", a.ref_loci, p.baseline_units)
    add(f"{tag}: enrichment", a.enrichment, p.ratio, "", tol=TOL)
    add(f"{tag}: band 2.5%", a.band_lo, p.ratio_lo, "", tol=TOL)
    add(f"{tag}: band 97.5%", a.band_hi, p.ratio_hi, "", tol=TOL)
info("TR3 white-pulp profile, global test (max-T over 17 bins)", np.nan, rw.summary["global_p"], "not in the archive")

# >= 20 reads: the archive continued the same generator after the first 10,000 draws
hi = (A.max_reads >= 20).to_numpy()
r20 = tl.niche_profile(A, group=iN & hi, baseline=ref & hi, compartment="comp_pkg", n_perm=10_000, rng=g,
                       blocks=None)
t20 = r20.table.set_index("label")
for c in ("T zone", "follicle", "red pulp"):
    a, p = F3.loc[f"{c} share"], t20.loc[c]
    k = int(p.band)
    tag = f"TR3 >= 20 reads {c}"
    add(f"{tag}: iNKT loci (all compartments)", a.iNKT_loci, r20.summary["n_group"])
    add(f"{tag}: iNKT share", a.iNKT, p.statistic, "", tol=TOL)
    add(f"{tag}: reference share", a.reference, p.expected, "", tol=TOL)
    add(f"{tag}: chance mean", a.chance_mean, p.null_mean, "", tol=TOL)
    add(f"{tag}: chance 2.5%", a.chance_lo, p.null_lo, "", tol=TOL)
    add(f"{tag}: chance 97.5%", a.chance_hi, p.null_hi, "", tol=TOL)
    add(f"{tag}: two-sided p (legacy rule)", a.p_two_sided, legacy_p(p.statistic, r20.null[:, k]), "", tol=TOL)
    info(f"{tag}: two-sided p (package rule)", a.p_two_sided, p.p_value, f"p_maxT = {p.p_maxT:.4g}")

# =============================================================================================== TR7
P7 = pd.read_csv(RES / "TR7_profiles.csv")
t4 = pd.read_csv(RES / "TR4_clonotypes.csv", index_col=0)
exp_cl = set(t4[t4.expanded & (t4.cls == "clonal")].index)
exp_cv = set(t4[t4.expanded & (t4.cls == "convergent")].index)
Bm = Lo[(Lo.chain == "TRB") & Lo.member].reset_index(drop=True)
a_ref = ~A.inkt & ~A.mait
a_hi = A.max_reads >= 20
groups = [
    ("iNKT, all loci", A, A.inkt, a_ref),
    ("iNKT, ≥ 20 reads", A, A.inkt & a_hi, a_ref & a_hi),
    ("expanded, one rearrangement", Bm, Bm.clone.isin(exp_cl), np.ones(len(Bm), bool)),
    ("expanded, several rearrangements", Bm, Bm.clone.isin(exp_cv), np.ones(len(Bm), bool)),
]
edges = np.arange(-150, 301, 25.0)
g = np.random.default_rng(0)
n_rows = n_skip = 0
for axis, col in (("tz", "tz_pkg"), ("wp_depth", "wp_pkg")):
    for name, tab, grp, base in groups:
        rr = tl.niche_profile(tab, group=np.asarray(grp, bool), baseline=np.asarray(base, bool), distance_um=col,
                              bands_um=edges, pool="baseline", n_perm=5000, rng=g, blocks=None)
        a = P7[(P7.axis == axis) & (P7.group == name)].reset_index(drop=True)
        p = rr.table
        if len(a) != len(p):
            add(f"TR7 {axis} {name}: bins", len(a), len(p))
            continue
        ok_ref = (a.ref_loci > 0).to_numpy()
        n_skip += int((~ok_ref).sum())
        checks = {
            "group loci": (a.group_loci.to_numpy(), p.group_units.to_numpy(), 0.0),
            "reference loci": (a.ref_loci.to_numpy(), p.baseline_units.to_numpy(), 0.0),
            "enrichment": (a.enrichment.to_numpy()[ok_ref], p.ratio.to_numpy()[ok_ref], TOL),
            "band 2.5%": (a.band_lo.to_numpy()[ok_ref], p.ratio_lo.to_numpy()[ok_ref], TOL),
            "band 97.5%": (a.band_hi.to_numpy()[ok_ref], p.ratio_hi.to_numpy()[ok_ref], TOL),
        }
        for what, (av, pv, tol) in checks.items():
            bad = int((~np.isclose(av.astype(float), pv.astype(float), rtol=tol, atol=0)).sum())
            add(f"TR7 {axis} / {name}: bins whose {what} differs", 0, bad, f"{len(av)} bins")
        n_rows += len(a)
        info(f"TR7 {axis} / {name}: global test (max-T over bins)", np.nan, rr.summary["global_p"],
             f"{rr.summary['n_group']} group loci; most extreme bin {rr.summary['most_extreme']}")
        if name.startswith("expanded"):  # a clone-level label: whole clones are the exchangeable blocks (F4a default)
            rb = tl.niche_profile(tab, group=np.asarray(grp, bool), baseline=np.asarray(base, bool), distance_um=col,
                                  bands_um=edges, pool="baseline", n_perm=5000, rng=np.random.default_rng([7, n_rows]),
                                  blocks="auto", clone_key="clone")
            info(f"TR7 {axis} / {name} [improved: clone blocks, the F4a default for clone-level labels]: global test",
                 np.nan, rb.summary["global_p"],
                 f"{rb.summary['n_group_blocks']} clones ({rb.summary['n_group']} loci); blocks {rb.summary['blocks']}; "
                 f"loci shuffled one by one: {rr.summary['global_p']:.4g}")
add("TR7 profile rows compared", len(P7), n_rows, f"{n_skip} bins without reference loci compared on counts only")

res = pd.DataFrame(rows)
res.to_csv(OUT / "m3_niche.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm3_niche.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")
