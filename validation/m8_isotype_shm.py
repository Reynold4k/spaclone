"""M8 validation: StereoIR BCR isotypes and SHM (archive T1), CSR hotspots (T3) and the SHM test of CSR hotspots (T7),
recomputed from the raw StereoIR BCR AIRR with spaclone functions only.

Steps (package functions; data loading with pandas):
  T1  raw AIRR -> pp.filter_valid (valid junction, productive, >= 2 reads) -> IGH -> pp.define_clones (exact) ->
      bcr.isotype (class / subclass from c_call) -> bcr.shm (V mismatches of v_sequence_alignment vs
      v_germline_alignment, gaps skipped; allele baseline = median over IgM + IgD contigs with >= 30 reads and >= 250
      aligned V nt, alleles with >= 10 such contigs; evaluable = >= 30 reads, >= 250 nt, baseline; mutated = excess
      >= 3) -> pp.merge_loci (15 um) -> bcr.call_isotype(by="locus") and bcr.call_shm(by="locus").
      Compared contig by contig with T1_igh_contigs.parquet (iso, sub, vlen, vmm, base, excess, evaluable, mutated)
      and locus by locus with T1_igh_loci.parquet, plus T1_summary.csv. skip_ambiguous=False is the archive's exact
      mismatch rule (only '-' / '.' skipped); the package default (skip non-ACGT) is reported alongside.
  T3  isotyped loci (in the archive's loc_id order) -> bcr.csr_hotspots (thin wrapper of tl.hotspots; sigma 50 um,
      n_eff >= 10 on the typed-bin tissue, 1,000 shuffles, rng default_rng(0)); legacy settings blocks=None (loci
      shuffled one by one), neff_kernel="legacy", threshold_rule="quantile" must give the 13 archived hotspots
      exactly; blocks=None with the exact kernel and the rank rule gives 12; the clone-level default (blocks=
      "clone_id", matched design, studentised map; fix F4b 2026-10-30) is reported.
      Also checked: identical output to tl.hotspots called directly on the archived T1 loci (M3's validation input).
  T7  evaluable loci -> bcr.shm_hotspots (raw and isotype-matched; the archive's interleaved shuffles replayed as
      null_marks, the generator continued) -> bcr.region_test(blocks=None) (CSR hotspots: mutated loci vs the
      isotype-matched expectation, 10,000 within-isotype shuffles): pooled 90 vs 56.17, p = 1e-4. The clone-level
      defaults of bcr.shm_hotspots / bcr.region_test are reported alongside.
Output: validation/results/m8_isotype_shm.csv (quantity, archived, package, match, note)
Usage : python -I m8_isotype_shm.py   (SPACLONE_SRC=<dev src> to test a scratch copy; logs the module checksums)
"""

import hashlib
import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone import bcr, tl  # noqa: E402
from spaclone.pp import define_clones, filter_valid, merge_loci  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RES = E / "results" / "QC" / "StereoIR_clones"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
NP = int(os.environ.get("M8_NPERM", "0")) or None  # debugging only: fewer shuffles
rows = []
FP = 1e-9


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


def n_diff(a, b, tol=0.0):
    """Number of positions where two arrays differ (NaN == NaN), with an absolute tolerance for floats."""
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    na, nb = a.isna(), b.isna()
    both = ~na & ~nb
    if tol and pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        d = (a[both].astype(float) - b[both].astype(float)).abs() > tol
    else:
        d = a[both].astype(object).to_numpy() != b[both].astype(object).to_numpy()
    return int(np.sum(d)) + int((na != nb).sum())


def typed_bins():
    lab = {}
    for f, c in (("bcell", "B lineage"), ("tcell", "T"), ("erythrocyte", "erythrocyte")):
        for b in open(E / "StereoIR" / f"mm_spleen_{f}.tsv").read().split():
            lab[b] = c
    ids = sorted(lab)
    xy = np.array([list(map(int, i.split("-"))) for i in ids], dtype=float) * 0.5
    return pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "cell_type": [lab[i] for i in ids]})


src = Path(spaclone.__file__).resolve().parent
log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
for f in ("isotype", "shm", "csr"):
    log(f"md5 bcr/{f}.py {hashlib.md5((src / 'bcr' / f'{f}.py').read_bytes()).hexdigest()}")

# ============================================================================================================= T1
cols = ["cell_id", "locus", "v_call", "d_call", "j_call", "c_call", "junction", "junction_aa", "productive",
        "consensus_count", "v_sequence_alignment", "v_germline_alignment"]  # fmt: skip
NROWS = int(os.environ.get("M8_NROWS", "0")) or None  # debugging only: first rows of the raw table
raw = pd.read_csv(E / "StereoIR" / "mm_spleen_bcr_airr.tsv.tsv", sep="\t", usecols=cols, low_memory=False,
                  nrows=NROWS)
xy = raw.cell_id.str.split("-", expand=True).astype(float) * 0.5
raw = raw.rename(columns={"cell_id": "spot_id"}).assign(x_um=xy[0].to_numpy(), y_um=xy[1].to_numpy())
ok, _ = filter_valid(raw, require_productive=True, min_reads=2)
H = ok[ok.locus == "IGH"].reset_index(drop=True)
H = define_clones(H, method="exact", by=())
H = bcr.isotype(H, species="mouse")
H_def = bcr.shm(H)  # package default: non-ACGT characters skipped
H, tab = bcr.shm(H, skip_ambiguous=False, return_baseline=True)  # the archive's rule: only gap characters skipped
log(f"IGH contigs {len(H):,}")
T1 = pd.read_parquet(RES / "T1_igh_contigs.parquet")
if NROWS:
    T1 = T1.iloc[: len(H)].reset_index(drop=True)
S1 = pd.read_csv(RES / "T1_summary.csv").set_index("item").value
same_rows = (
    len(T1) == len(H)
    and (T1.cell_id.to_numpy() == H.spot_id.to_numpy()).all()
    and (T1.junction.to_numpy() == H.junction.to_numpy()).all()
    and (T1.reads.to_numpy() == H.consensus_count.to_numpy()).all()
)
add("T1 contigs in the archive's row order (1 = yes)", 1, int(same_rows), "same filter, same order")
add("T1 clone ids identical (exact V|J|junction aa) (1 = yes)", 1,
    int((T1.clone.to_numpy() == H.clone_id.astype(str).to_numpy()).all()))  # fmt: skip
iso_map = {"IgM": "IGHM", "IgD": "IGHD", "IgG": "IGHG", "IgA": "IGHA", "IgE": "IGHE", "unknown": "unknown"}
my_iso = H.isotype.astype(str).map(iso_map)
add("T1 per-contig isotype class differing from T1 'iso'", 0, n_diff(my_iso, T1.iso))
my_sub = H.isotype_subclass.astype("string").str.upper().str.replace("IG", "IGH", n=1, regex=False).fillna("unknown")
add("T1 per-contig subclass differing from T1 'sub'", 0, n_diff(my_sub, T1["sub"]),
    "T1 'sub' = c_call gene; mouse IGHG2A kept as IgG2a")  # fmt: skip
for k, v in my_iso.value_counts().items():
    add(f"T1 contigs with isotype {k}", int((T1.iso == k).sum()), int(v))
add("T1 per-contig aligned V nt (vlen) differing", 0, n_diff(H.shm_aligned_nt, T1.vlen))
add("T1 per-contig V mismatches (vmm) differing", 0, n_diff(H.shm_mismatches, T1.vmm))
add("T1 per-contig allele baseline differing (NaN = NaN)", 0, n_diff(H.shm_baseline, T1.base, tol=1e-12))
add("T1 per-contig excess differing (NaN = NaN)", 0, n_diff(H.shm_excess, T1.excess, tol=1e-12))
add("T1 per-contig evaluable flag differing", 0, n_diff(H.shm_evaluable, T1.evaluable))
add("T1 per-contig mutated flag differing", 0, n_diff(H.shm_mutated, T1.mutated))
add("T1 evaluable contigs (mutation)", S1["evaluable contigs (mutation)"], int(H.shm_evaluable.sum()))
add("T1 alleles with a baseline", S1["alleles with a baseline"], int(tab.baseline.notna().sum()))
add("T1 alleles with baseline >= 3", S1["alleles with baseline >= 3"], int((tab.baseline >= 3).sum()))
for k in ("IgA", "IgD", "IgG", "IgM", "unknown"):
    g = H[H.isotype.astype(str) == k]
    a = iso_map[k]
    add(f"T1 {a} evaluable share", S1[f"{a} evaluable share"], g.shm_evaluable.mean(), "", tol=1e-12)
    add(f"T1 {a} mutated share of evaluable", S1[f"{a} mutated share of evaluable"],
        g.loc[g.shm_evaluable, "shm_mutated"].mean(), "", tol=1e-12)  # fmt: skip
info("T1 contigs whose V mismatch count changes when non-ACGT characters are skipped (package default)", 0,
     n_diff(H_def.shm_mismatches, H.shm_mismatches),
     f"aligned nt changes in {n_diff(H_def.shm_aligned_nt, H.shm_aligned_nt)} contigs; mutated flag changes in "
     f"{n_diff(H_def.shm_mutated, H.shm_mutated)}; evaluable changes in "
     f"{n_diff(H_def.shm_evaluable, H.shm_evaluable)}")  # fmt: skip
info("T1 contigs with lower-case letters in the V alignments", 0,
     int(H.v_sequence_alignment.fillna("").str.contains("[a-z]").sum()
         + H.v_germline_alignment.fillna("").str.contains("[a-z]").sum()),
     "count against an upper-cased comparison; 0 = the upper-casing changes nothing")  # fmt: skip
info("T1 V calls with several comma-separated alleles", 0, int(H.v_call.fillna("").str.contains(",").sum()),
     "the baseline groups the V call string as given")  # fmt: skip

# ---- loci
loci, member = merge_loci(H, radius_um=15.0)
H["locus_id"] = member.to_numpy()
ic = bcr.call_isotype(H, by="locus")
sc = bcr.call_shm(H, by="locus")
L = loci.join(ic.drop(columns="n_records"), on="locus_id").join(
    sc[["n_evaluable", "excess", "evaluable", "mutated"]], on="locus_id"
)
# the archive numbers loci by clone and first appearance of a bin in the contig table: same order by construction
first = pd.Series(np.arange(len(H))).groupby(H.locus_id.to_numpy()).min()
L["first_row"] = L.locus_id.map(first).to_numpy()
L = L.sort_values(["clone_id", "first_row"], kind="stable").reset_index(drop=True)
Lo = pd.read_parquet(RES / "T1_igh_loci.parquet")
pair = pd.DataFrame({"mine": H.locus_id.to_numpy(), "t1": T1.loc_id.to_numpy()}).drop_duplicates()
bij = pair.groupby("mine").t1.nunique().max() == 1 and pair.groupby("t1").mine.nunique().max() == 1
add("T1 locus partition identical (contig -> locus) (1 = yes)", 1, int(bij))
look = pair.set_index("mine").t1
L["loc_id"] = L.locus_id.map(look).to_numpy()
add("T1 loci in the archive's loc_id order after sorting by (clone, first contig) (1 = yes)", 1,
    int(len(L) == len(Lo) and (L.loc_id.to_numpy() == Lo.loc_id.to_numpy()).all()))  # fmt: skip
A = Lo.set_index("loc_id").loc[L.loc_id.to_numpy()].reset_index()
add("T1 loci", len(Lo), len(L))
add("T1 loci: bins per locus differing", 0, n_diff(L.n_spots, A.bins))
add("T1 loci: contigs per locus differing", 0, n_diff(L.n_records, A.contigs))
add("T1 loci: centroid x differing (> 1e-9 um)", 0, n_diff(L.x_um, A.x, tol=1e-9))
add("T1 loci: centroid y differing (> 1e-9 um)", 0, n_diff(L.y_um, A.y, tol=1e-9))
add("T1 loci: max reads differing", 0, n_diff(L.reads_max, A.max_reads))
for k in ("IgM", "IgD", "IgG", "IgA", "unknown"):
    add(f"T1 loci: {iso_map[k]} contig counts differing", 0, n_diff(L[k], A[iso_map[k]]))
add("T1 loci: isotyped contigs differing", 0, n_diff(L.n_isotyped, A.isotyped))
add("T1 loci: switched share differing (NaN = NaN)", 0, n_diff(L.switched_share, A.switched_share, tol=1e-12))
add("T1 loci: main isotype differing", 0, n_diff(L.isotype.astype(str).map(iso_map), A.iso_main))
iso_ok = (A.isotyped > 0).to_numpy()
add("T1 loci: switched call differing (isotyped loci)", 0,
    n_diff(L.switched[iso_ok].fillna(False).astype(bool), A.switched[iso_ok]))  # fmt: skip
add("T1 loci without isotype: switched missing in the package (archive: False)", int((~iso_ok).sum()),
    int(L.switched.isna().sum()))  # fmt: skip
add("T1 loci: median excess differing (NaN = NaN)", 0, n_diff(L.excess, A.excess, tol=1e-12))
add("T1 loci: evaluable (mut_eval) differing", 0, n_diff(L.evaluable, A.mut_eval))
ev = A.mut_eval.to_numpy(dtype=bool)
add("T1 loci: mutated call differing (evaluable loci)", 0, n_diff(L.mutated[ev].fillna(False).astype(bool), A.mutated[ev]))
for name, mask in (("switched (IgG / IgA)", A.switched.to_numpy(dtype=bool)),
                   ("unswitched (IgM / IgD)", (A.iso_main.isin(["IGHM", "IGHD"])).to_numpy())):  # fmt: skip
    sel = ev & mask
    add(f"T1 mutated share of evaluable {name} loci", A.mutated[sel].mean(),
        L.mutated[sel].fillna(False).astype(bool).mean(), "", tol=1e-12)  # fmt: skip

# ============================================================================================================= T3
bins = typed_bins()
grid = tl.Grid.from_points(bins.x_um, bins.y_um, pixel_um=10.0)
tissue = tl.tissue_mask(bins, grid=grid)
on = grid.index(L.x_um.to_numpy(), L.y_um.to_numpy())[2]
I = L[L.switched.notna().to_numpy() & on].reset_index(drop=True)
S3 = pd.read_csv(RES / "T3_csr_summary.csv").set_index("item").value
H3 = pd.read_csv(RES / "T3_csr_hotspots.csv")
legacy = {"neff_kernel": "legacy", "threshold_rule": "quantile"}
kw = {"sigma_um": 50.0, "tissue": tissue, "min_neff": 10.0, "grid": grid}
hs = bcr.csr_hotspots(I, blocks=None, n_perm=NP or 1000, rng=np.random.default_rng(0), **kw, **legacy)
log(repr(hs))
add("T3 isotyped loci", S3["isotyped loci"], hs.summary["n_units"])
add("T3 switched share p0", S3["switched share p0"], hs.summary["p0"], "", tol=1e-12)
add("T3 z threshold (95th pct of shuffled map maxima)", S3["z threshold"], hs.threshold["hot"], "", tol=FP)
add("T3 hotspots", S3["hotspots"], hs.summary["n_hotspots"])
add("T3 hotspot area mm2", S3["hotspot area mm2"], hs.table.area_um2.sum() / 1e6, "", tol=1e-12)
add("T3 shuffled maps with any hotspot", S3["shuffled maps with any hotspot"], hs.summary["null_any_hot"])
npz3 = np.load(RES / "T3_csr_maps.npz")
add("T3 hotspot label raster identical to T3_csr_maps.npz (1 = yes)", 1,
    int(np.array_equal(npz3["hotspot"], hs.labels.values)))  # fmt: skip
P = hs.table.set_index("hotspot")
NA_ROW = pd.Series({"units": np.nan, "marked": np.nan, "peak_z": np.nan, "expected": np.nan, "p_value": np.nan,
                    "q_value": np.nan})  # fmt: skip
for r in H3.itertuples():
    p = P.loc[r.hotspot] if r.hotspot in P.index else NA_ROW
    add(f"T3 hotspot {r.hotspot} isotyped loci", r.isotyped_loci, p.units)
    add(f"T3 hotspot {r.hotspot} switched loci", r.switched_loci, p.marked)
    add(f"T3 hotspot {r.hotspot} max z", r.max_z, p.peak_z, "", tol=FP)
# same result as tl.hotspots on the archived T1 loci (M3's validation input)
Lo_on = Lo[(Lo.isotyped > 0).to_numpy() & grid.index(Lo.x.to_numpy(), Lo.y.to_numpy())[2]].reset_index(drop=True)
direct = tl.hotspots(Lo_on, mark="switched", n_perm=NP or 1000, rng=np.random.default_rng(0), x_key="x", y_key="y",
                     **kw, **legacy)  # fmt: skip
add("T3 bcr.csr_hotspots (raw AIRR) = tl.hotspots (archived T1 loci): labels identical (1 = yes)", 1,
    int(np.array_equal(direct.labels.values, hs.labels.values)))  # fmt: skip
add("T3 bcr.csr_hotspots = tl.hotspots: threshold", direct.threshold["hot"], hs.threshold["hot"], "", tol=1e-12)
cm = bcr.csr_map(I, tissue=tissue, min_neff=10.0, grid=grid, neff_kernel="legacy")
add("T3 bcr.csr_map share = hotspot map share (max abs difference)", 0.0,
    float(np.nanmax(np.abs(np.where(cm.eligible, cm.share - hs.map.share, 0.0)))), "", tol=1e-12)  # fmt: skip
hs_def = bcr.csr_hotspots(I, blocks=None, n_perm=NP or 1000, rng=0, **kw)
ov = float(((hs_def.labels.values > 0) & (hs.labels.values > 0)).sum() / max((hs.labels.values > 0).sum(), 1))
info("T3 loci shuffled one by one (blocks=None), exact n_eff kernel, rank threshold rule: hotspots", S3["hotspots"],
     hs_def.summary["n_hotspots"],
     f"threshold {hs_def.threshold['hot']:.4f}; eligible pixels {hs_def.summary['n_eligible']} (legacy "
     f"{hs.summary['n_eligible']}); {ov:.1%} of the archive's hotspot area kept")  # fmt: skip
hs_clone = bcr.csr_hotspots(I, n_perm=NP or 1000, rng=0, **kw)
ov = float(((hs_clone.labels.values > 0) & (hs.labels.values > 0)).sum() / max((hs.labels.values > 0).sum(), 1))
info("T3 package defaults (clone-level matched design, studentised map, exact kernel, rank rule): hotspots",
     S3["hotspots"], hs_clone.summary["n_hotspots"],
     f"threshold {hs_clone.threshold['hot']:.3f} (studentised), max {hs_clone.summary['max_z']:.2f}; "
     f"{hs_clone.table.area_um2.sum() / 1e6:.4f} mm2, {ov:.1%} of the archive's hotspot area; loci merged "
     f"{hs_clone.params['units_merged_into_loci']}")  # fmt: skip

# ============================================================================================================= T7
S7 = pd.read_csv(RES / "T7_summary.csv").iloc[0]
H7 = pd.read_csv(RES / "T7_shm_hotspots.csv")
C7 = pd.read_csv(RES / "T7_csr_hotspot_shm_test.csv")
Ev = L[L.evaluable.to_numpy(dtype=bool) & on].reset_index(drop=True)
Ev["mutated"] = Ev.mutated.fillna(False).astype(bool)
Ev["isotype"] = Ev.isotype.astype(str)
m = Ev.mutated.to_numpy(dtype=float)
add("T7 evaluable loci", S7.evaluable_loci, len(Ev))
add("T7 mutated share p0", S7.p0, m.mean(), "", tol=1e-12)
classes = [np.flatnonzero(Ev.isotype.to_numpy() == c) for c in sorted(Ev.isotype.unique())]
g = np.random.default_rng(0)
null_raw = np.empty((NP or 1000, len(Ev)), dtype=bool)
null_iso = np.empty((NP or 1000, len(Ev)), dtype=bool)
for b in range(NP or 1000):  # the archive's draw order: a raw permutation, then one per isotype class
    null_raw[b] = m[g.permutation(len(m))] > 0.5
    o = m.copy()
    for idx in classes:
        o[idx] = m[idx][g.permutation(len(idx))]
    null_iso[b] = o > 0.5
kw7 = {"sigma_um": 50.0, "tissue": tissue, "min_neff": 10.0, "grid": grid, **legacy}
raw7 = bcr.shm_hotspots(Ev, match_by=None, blocks=None, null_marks=null_raw, **kw7)
iso7 = bcr.shm_hotspots(Ev, match_by="isotype", blocks=None, null_marks=null_iso, **kw7)
add("T7 raw threshold", S7.thr_raw, raw7.threshold["hot"], "replayed draws", tol=FP)
add("T7 isotype-matched threshold", S7.thr_iso, iso7.threshold["hot"], "replayed draws", tol=FP)
add("T7 SHM hotspots (raw)", S7.shm_hotspots_raw, raw7.summary["n_hotspots"], "replayed draws")
add("T7 SHM hotspots (isotype-matched)", S7.shm_hotspots_iso, iso7.summary["n_hotspots"], "replayed draws")
add("T7 max z (raw)", S7.max_z_raw, raw7.summary["max_z"], "", tol=FP)
add("T7 max z (isotype-matched)", S7.max_z_iso, iso7.summary["max_z"], "", tol=FP)
npz7 = np.load(RES / "T7_shm_maps.npz")
add("T7 SHM hotspot labels (raw) identical to T7_shm_maps.npz (1 = yes)", 1,
    int(np.array_equal(npz7["hotspot"], raw7.labels.values)))  # fmt: skip
add("T7 SHM hotspot labels (isotype-matched) identical (1 = yes)", 1,
    int(np.array_equal(npz7["hotspot_iso"], iso7.labels.values)))  # fmt: skip
for r in H7.itertuples():
    hres = raw7 if r.kind == "raw" else iso7
    tb = hres.table.set_index("hotspot")
    p = tb.loc[r.hotspot] if r.hotspot in tb.index else NA_ROW
    add(f"T7 SHM hotspot {r.kind} {r.hotspot} evaluable loci", r.evaluable_loci, p.units)
    add(f"T7 SHM hotspot {r.kind} {r.hotspot} mutated loci", r.mutated_loci, p.marked)
csr = bcr.region_test(Ev, regions=hs, blocks=None, n_perm=10 * (NP or 1000), min_units=5, rng=g)
log(repr(csr))
Pc = csr.table.assign(key=csr.table.region.astype(str)).set_index("key")
for r in C7.itertuples():
    p = Pc.loc[str(r.csr_hotspot)] if str(r.csr_hotspot) in Pc.index else NA_ROW
    tag = f"T7 CSR test {r.csr_hotspot}"
    add(f"{tag} evaluable loci", r.evaluable_loci, p.units)
    add(f"{tag} mutated loci", r.mutated_loci, p.marked)
    add(f"{tag} isotype-matched expectation", r.expected_isotype_matched, p.expected, "", tol=1e-12)
    add(f"{tag} p (10,000 within-isotype shuffles)", r.p_isotype_matched, p.p_value, "replayed draws", tol=1e-12)
    if np.isfinite(r.q):
        add(f"{tag} q (BH over hotspots)", r.q, p.q_value, "replayed draws", tol=1e-12)
# the package's own draws (no replay): Monte-Carlo error only
g0 = np.random.default_rng(1)
own = bcr.region_test(Ev, regions=hs, blocks=None, n_perm=10 * (NP or 1000), min_units=5, rng=g0)
po_t = own.table.assign(key=own.table.region.astype(str)).set_index("key")
po = po_t.loc["all"] if "all" in po_t.index else pd.Series({"marked": np.nan, "p_value": np.nan})
add("T7 CSR test pooled, own draws: mutated vs isotype-matched expectation", 90.0, po.marked)
info("T7 CSR test pooled, own draws: p", 1e-4, po.p_value, "smallest attainable p = 1/(n_perm + 1)")
own_iso = bcr.shm_hotspots(Ev, blocks=None, n_perm=NP or 1000, rng=np.random.default_rng(2), **kw7)
info("T7 isotype-matched SHM hotspots, own draws", S7.shm_hotspots_iso, own_iso.summary["n_hotspots"],
     f"threshold {own_iso.threshold['hot']:.3f} (archive {S7.thr_iso:.3f}); Monte-Carlo threshold")  # fmt: skip
clone_iso = bcr.shm_hotspots(Ev, n_perm=NP or 1000, rng=0, sigma_um=50.0, tissue=tissue, min_neff=10.0, grid=grid)
info("T7 SHM hotspots, package defaults (clone-level matched design, isotype strata, studentised)",
     S7.shm_hotspots_iso, clone_iso.summary["n_hotspots"],
     f"max studentised z {clone_iso.summary['max_z']:.2f}, threshold {clone_iso.threshold['hot']:.3f}, map-level p "
     f"{clone_iso.summary['p_map_hot']:.4f}")  # fmt: skip
cl_rt = bcr.region_test(Ev, regions=hs, n_perm=10 * (NP or 1000), min_units=5, rng=0)
pc = cl_rt.table.assign(key=cl_rt.table.region.astype(str)).set_index("key")
pcr = pc.loc["all"] if "all" in pc.index else pd.Series({"marked": np.nan, "expected": np.nan, "null_mean": np.nan,
                                                         "null_sd": np.nan, "p_value": np.nan})  # fmt: skip
add("T7 CSR test pooled, clone-level default: mutated loci (design-free count)", 90.0, pcr.marked)
add("T7 CSR test pooled, clone-level default: isotype-matched expectation (design-free)", 56.1683960518621,
    pcr.expected, "", tol=1e-12)  # fmt: skip
info("T7 CSR test pooled, clone-level default: p", 1e-4, pcr.p_value,
     f"null mean {pcr.null_mean:.2f} (isotype and clone size matched), sd {pcr.null_sd:.2f}; the archive shuffled loci "
     "one by one")  # fmt: skip

res = pd.DataFrame(rows)
res.to_csv(OUT / "m8_isotype_shm.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm8_isotype_shm.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")  # fmt: skip
