"""M4 validation: within-lineage tests (tl.tip_region_test, tl.tip_contrast, tl.tree_tissue_distance) against L7 / L11.

Recomputes, with package functions only, the StereoIR lineage statistics of the archive
($SPACLONE_ARCHIVE/workflow/qc/L7_tree_stats.py, L11_hotspot_lineages.py) from the archived tables in
results/QC/StereoIR_lineage (L4 tips and loci, L5 trees as edge tables, L1 locus locations, T1 loci):

  L11  per tree: tips in a CSR hotspot vs the isotype-matched expectation (hotspot share of all T1 IGH loci of the
       tip's isotype), one-sided p from 100,000 Bernoulli draws per tree (the archive's rule; the generator is replayed
       in the archive's order, so the p-values must be identical), BH over trees, called = >= 3 hotspot tips and
       q < 0.05; and the exact Poisson-binomial p-values (improved).
  L7 B within-tree label shuffles (10,000): B1 mutations hotspot - other tips, B2 switched - unswitched, B3a/B3b share
       in hotspot / follicle by isotype class, B4a/B4b by mutation status.
  L7 A loci (not trees): A1 / A1r per isotype hotspot - other loci SHM (shuffles within isotype), A2 within clone.
  L7 C within-tree Spearman correlation of patristic and tissue distance (pairs of tips in different loci), shuffles
       of tip positions within the tree (1,000; the archive's rule) and of locus positions (improved).
Monte-Carlo quantities (null summaries, p) may differ by Monte-Carlo error; observed statistics must be identical.

Output: validation/results/m4_lineage.csv (quantity, archived, package, match, note)
Usage : python -I m4_lineage.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import spaclone
from spaclone.tl import tip_contrast, tip_region_test, tree_tissue_distance

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
R = E / "results" / "QC"
LR = R / "StereoIR_lineage"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOC_FROM_CODE = {"hotspot": "CSR hotspot", "follicle": "follicle", "Tzone": "T zone", "redpulp": "red pulp"}
ISO4 = ["IGHM", "IGHD", "IGHG", "IGHA"]
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, match, note=""):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'} {note}")


def mc_close(p_arch, p_pkg, n_perm, z=3.5):
    """Two Monte-Carlo p-values agree within z standard errors (both from n_perm draws)."""
    p = max(min((p_arch + p_pkg) / 2, 1 - 1e-9), 1.0 / (n_perm + 1))
    return abs(p_arch - p_pkg) <= z * np.sqrt(2 * p * (1 - p) / n_perm) + 2.0 / (n_perm + 1)


def iso_class(iso):
    iso = np.asarray(iso, dtype=object)
    return np.where(np.isin(iso, ["IGHG", "IGHA"]), "switched",
                    np.where(np.isin(iso, ["IGHM", "IGHD"]), "unswitched", "unknown"))


def tree_tips(tag):
    """L4 tips attached to the L5 tree tips that hold them (LTree.leaves) and the trees as edge tables."""
    nodes = pd.read_csv(LR / f"L5_nodes_{tag}.tsv", sep="\t", keep_default_na=False, na_values={"parent": ["NA", ""]},
                        dtype={"tip_ids": str, "label": str},
                        usecols=["clone_id", "node", "parent", "branch_mut", "label", "tip_ids"])
    t4 = pd.read_csv(LR / f"L4_tips_{tag}.tsv", sep="\t")
    t4["location"] = t4.location.map(LOC_FROM_CODE)
    hold = []
    for c, df in nodes.groupby("clone_id", sort=False):
        for lab, ids in zip(df.label, df.tip_ids, strict=True):
            if lab and lab != "Germline":
                hold.extend((c, s.strip(), lab) for s in str(ids).split(","))
    hold = pd.DataFrame(hold, columns=["clone_id", "sequence_id", "node_label"]).drop_duplicates(
        ["clone_id", "sequence_id"], keep="last")
    tips = t4.merge(hold, on=["clone_id", "sequence_id"], how="inner")
    edges = {c: df[["node", "parent", "branch_mut", "label"]].reset_index(drop=True)
             for c, df in nodes.groupby("clone_id", sort=False)}
    return tips, edges


log(f"spaclone {spaclone.__version__}")
L7 = pd.read_csv(LR / "L7_tests.csv")


def arch_row(test, grouping):
    r = L7[(L7.test == test) & (L7.grouping == grouping)]
    return r.iloc[0] if len(r) == 1 else r


# ================================================================================================= L11
loci = pd.read_parquet(R / "StereoIR_clones" / "T1_igh_loci.parquet", columns=["loc_id", "iso_main"])
hot = pd.read_parquet(LR / "L1_loci_location.parquet", columns=["loc_id", "hot"])
ref = loci.merge(hot, on="loc_id")
ref["iso"] = ref.iso_main.fillna("unknown")
ref["inhot"] = ref.hot > 0
share = ref.groupby("iso").inhot.mean()
L11 = pd.read_csv(LR / "L11_hotspot_lineages.csv")
gen = np.random.default_rng(0)  # the archive's generator, replayed in its order (E test, 20 chance sets, L test)
for g in "EL":
    t = pd.read_csv(LR / f"L4_tips_{g}.tsv", sep="\t", usecols=["clone_id", "loc_id", "isotype", "location", "mut"])
    t = t.reset_index(drop=True).merge(hot, on="loc_id", how="left")
    t["inhot"] = t.hot.fillna(0) > 0
    iso = t.isotype.fillna("unknown")
    t["iso"] = iso.where(iso.isin(share.index), "unknown")
    res = tip_region_test(t, region_key="inhot", covariate_key="iso", reference=ref, method="mc", n_perm=100_000,
                          min_region_tips=3, alpha=0.05, rng=gen, unit_key=None)  # legacy: every tip a unit
    pvec = t.iso.map(share).to_numpy(dtype=float)
    for _ in range(20):  # the archive's chance-level data sets (consume the generator as the archive did)
        f = gen.random(len(t)) < pvec
        tip_region_test(t.assign(inhot=f), region_key="inhot", covariate_key="iso", reference=ref, method="mc",
                        n_perm=5_000, rng=gen, unit_key=None)
    a = L11[L11.grouping == g].set_index("clone_id")
    p = res.table.set_index("tree")
    j = a.join(p, how="inner", rsuffix="_pkg")
    add(f"L11 {g}: trees", len(a), len(p), len(a) == len(p) == len(j))
    add(f"L11 {g}: tips, hotspot tips and expected identical (trees)", len(a),
        int(((j.tips == j.n_tips) & (j.hotspot_tips == j.statistic) & np.isclose(j.expected, j.expected_pkg)).sum()),
        bool(((j.tips == j.n_tips) & (j.hotspot_tips == j.statistic) & np.isclose(j.expected, j.expected_pkg)).all()))
    dp = float(np.abs(j.p - j.p_value).max())
    add(f"L11 {g}: max |p(package) - p(archive)| (100,000 draws, replayed generator)", 0.0, dp, dp == 0.0,
        "identical draws")
    add(f"L11 {g}: max |q(package) - q(archive)|", 0.0, float(np.abs(j.q - j.q_value).max()),
        float(np.abs(j.q - j.q_value).max()) < 1e-12)
    add(f"L11 {g}: trees called (>= 3 hotspot tips, q < 0.05)", int(a.passes.sum()), int(res.summary["n_significant"]),
        int(a.passes.sum()) == int(res.summary["n_significant"]))
    ex = tip_region_test(t, region_key="inhot", covariate_key="iso", reference=ref, method="exact", unit_key=None)
    e = ex.table.set_index("tree")
    de = float(np.abs(e.p_value.reindex(a.index) - a.p).max())
    add(f"L11 {g} [improved: exact Poisson-binomial p]: max |p(exact) - p(archive)|", 0.0, de,
        de <= 4 * np.sqrt(0.25 / 100_000) + 2e-5, "Monte-Carlo error of the archive's 100,000 draws")
    add(f"L11 {g} [improved: exact p]: trees called", int(a.passes.sum()), int(ex.summary["n_significant"]),
        int(ex.summary["n_significant"]) == int(a.passes.sum()))
    ul = tip_region_test(t, region_key="inhot", covariate_key="iso", reference=ref, method="exact", unit_key="loc_id")
    add(f"L11 {g} [improved: one unit per locus, exact p]: trees called", int(a.passes.sum()),
        int(ul.summary["n_significant"]), True,
        f"several sequences of one locus count once; tips {len(t)} -> units {int(ul.table.n_tips.sum())}")
    add(f"L11 {g} [new default (F4a): one unit per locus]: smallest q", float(a.q.min()), float(ul.table.q_value.min()),
        True, "package default since F4a: tips collapsed to cells (locus_id / 15-um loci); legacy = unit_key=None")

# ================================================================================================= L7 B (trees)
for tag in ("E", "L"):
    tips, edges = tree_tips(tag)
    tips["cls"] = iso_class(tips.isotype)
    tips["hot"] = tips.location == "CSR hotspot"
    tips["fol"] = tips.location == "follicle"
    tips["mutated"] = tips.mut > 0
    tips["unmutated"] = ~tips.mutated
    K = tips[tips.cls != "unknown"].assign(switched=lambda d: d.cls == "switched",
                                            unswitched=lambda d: d.cls == "unswitched")
    log(f"[{tag}] trees {tips.clone_id.nunique()}; tips {len(tips)}")
    tests = [
        ("B1", tips, "mut", "hot", "contrast"),
        ("B2", K, "mut", "switched", "contrast"),
        ("B3a", K, "hot", "switched", "region"),
        ("B3b", K, "fol", "unswitched", "region"),
        ("B4a", tips, "fol", "unmutated", "region"),
        ("B4b", tips, "hot", "mutated", "region"),
    ]
    for i, (name, d, value, group, kind) in enumerate(tests):
        rng = np.random.default_rng([7, i, ord(tag)])
        if kind == "contrast":
            r = tip_contrast(d, value_key=value, group_key=group, tree_key="clone_id", n_perm=10_000,
                             alternative="greater", rng=rng, unit_key=None)
            u = tip_contrast(d, value_key=value, group_key=group, tree_key="clone_id", n_perm=10_000,
                             alternative="greater", rng=np.random.default_rng([8, i, ord(tag)]), unit_key="loc_id")
        else:
            r = tip_region_test(d, region_key=value, covariate_key=group, null="within_tree", tree_key="clone_id",
                                n_perm=10_000, alternative="greater", rng=rng, unit_key=None)
            u = tip_region_test(d, region_key=value, covariate_key=group, null="within_tree", tree_key="clone_id",
                                n_perm=10_000, alternative="greater", rng=np.random.default_rng([8, i, ord(tag)]),
                                unit_key="loc_id")
        a = arch_row(name, tag)
        s = r.summary
        add(f"L7 {name} [{tag}] {a.description}: trees", int(a.clones), s["n_trees"], int(a.clones) == s["n_trees"])
        add(f"L7 {name} [{tag}]: observed", float(a.observed), s["statistic"],
            abs(float(a.observed) - s["statistic"]) < 1e-12)
        add(f"L7 {name} [{tag}]: chance 95% range", f"[{a.null_lo:.3f}, {a.null_hi:.3f}]",
            f"[{s['null_lo']:.3f}, {s['null_hi']:.3f}]",
            abs(a.null_lo - s["null_lo"]) <= 0.1 * (a.null_hi - a.null_lo) + 1e-9
            and abs(a.null_hi - s["null_hi"]) <= 0.1 * (a.null_hi - a.null_lo) + 1e-9, "Monte Carlo")
        add(f"L7 {name} [{tag}]: p (greater)", float(a.p), s["p_value"], mc_close(float(a.p), s["p_value"], 10_000),
            "Monte Carlo, 10,000 shuffles")
        add(f"L7 {name} [{tag}] [new default (F4a): one unit per locus]: p (greater)", float(a.p),
            u.summary["p_value"], True,
            f"trees {u.summary['n_trees']}; observed {u.summary['statistic']:.4g}; sequences of one locus count once")

    # ============================================================================================ L7 C (trees)
    pt = pd.read_csv(LR / f"L7_C_space_per_tree_{tag}.csv").set_index("clone")
    for shuffle in ("tips", "units"):
        r = tree_tissue_distance(tips, edges, tree_key="clone_id", tip_key="node_label", unit_key="loc_id", x_key="x",
                                 y_key="y", min_tips=5, min_units=2, min_pairs=3, shuffle=shuffle, n_perm=1_000,
                                 rng=np.random.default_rng([11, ord(tag)]), length_key="branch_mut")
        a = arch_row("C", tag)
        s = r.summary
        tb = r.table.set_index("tree")
        if shuffle == "tips":
            add(f"L7 C [{tag}]: trees tested (>= 5 tips in >= 2 loci)", int(a.clones), s["n_trees"],
                int(a.clones) == s["n_trees"])
            jj = pt.join(tb, how="inner")
            add(f"L7 C [{tag}]: max |rho(package) - rho(archive)| per tree", 0.0,
                float(np.abs(jj.rho - jj.statistic).max()), float(np.abs(jj.rho - jj.statistic).max()) < 1e-9,
                f"{len(jj)} trees joined")
            add(f"L7 C [{tag}]: pairs and median tissue distances identical (trees)", len(pt),
                int(((jj.pairs == jj.n_pairs) & np.isclose(jj.median_um_other_pairs, jj.median_um_other,
                                                           equal_nan=True)).sum()),
                bool(((jj.pairs == jj.n_pairs) & np.isclose(jj.median_um_other_pairs, jj.median_um_other,
                                                            equal_nan=True)).all()))
            add(f"L7 C [{tag}]: mean rho over trees", float(a.observed), s["statistic"],
                abs(float(a.observed) - s["statistic"]) < 1e-9)
            add(f"L7 C [{tag}]: chance 95% range", f"[{a.null_lo:.4f}, {a.null_hi:.4f}]",
                f"[{s['null_lo']:.4f}, {s['null_hi']:.4f}]",
                abs(a.null_lo - s["null_lo"]) <= 0.15 * (a.null_hi - a.null_lo)
                and abs(a.null_hi - s["null_hi"]) <= 0.15 * (a.null_hi - a.null_lo), "Monte Carlo")
            add(f"L7 C [{tag}]: p (greater)", float(a.p), s["p_value"], mc_close(float(a.p), s["p_value"], 1_000),
                "Monte Carlo, 1,000 shuffles")
            add(f"L7 C [{tag}]: per-tree p within Monte-Carlo error (share of trees)", 1.0,
                float(np.mean([mc_close(x, y, 1_000) for x, y in zip(jj.p_greater, jj.p_value, strict=True)])),
                np.mean([mc_close(x, y, 1_000) for x, y in zip(jj.p_greater, jj.p_value, strict=True)]) >= 0.97)
        else:
            add(f"L7 C [{tag}] [improved: locus positions shuffled, tips of a locus move together]: p (greater)",
                float(a.p), s["p_value"], True,
                f"mean rho {s['statistic']:.4f}; chance [{s['null_lo']:.4f}, {s['null_hi']:.4f}]")

# ================================================================================================= L7 A (loci)
Lo = pd.read_parquet(LR / "L4_loci_shm.parquet")
A = Lo[Lo.shm.notna() & Lo.iso_main.isin(ISO4)].copy()
A["hot"] = A.location == "CSR hotspot"
def centred_p(obs, null):
    """The archive's two-sided convention: share of |null - mean| >= |obs - mean| (from the package's null draws)."""
    mu = null.mean()
    return (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (1 + len(null))


LAB = {"IGHM": "IgM", "IGHD": "IgD", "IGHG": "IgG", "IGHA": "IgA"}
for name, sub in (("A1", A), ("A1r", A[A.shm <= 30])):
    p_leg = {}
    for i, iso in enumerate(ISO4):
        part = sub[sub.iso_main == iso]
        r = tip_contrast(part, value_key="shm", group_key="hot", tree_key="iso_main", n_perm=10_000,
                         alternative="two-sided", rng=np.random.default_rng([13, len(name), i]), keep_null=True)
        a = L7[(L7.test == name) & L7.description.str.startswith(LAB[iso])].iloc[0]
        st = r.table.iloc[0]
        p_leg[iso] = centred_p(st.statistic, r.null)
        add(f"L7 {name} {iso}: hotspot loci", int(a.clones), int(st.n_group), int(a.clones) == int(st.n_group))
        add(f"L7 {name} {iso}: observed SHM difference", float(a.observed), float(st.statistic),
            abs(float(a.observed) - float(st.statistic)) < 1e-12)
        add(f"L7 {name} {iso}: p (two-sided, the archive's centred convention)", float(a.p), p_leg[iso],
            mc_close(float(a.p), p_leg[iso], 10_000), "from the package's 10,000 within-isotype shuffles")
        add(f"L7 {name} {iso} [package convention: doubled one-sided p]", float(a.p), float(st.p_value), True,
            "differs from the centred p when the null is skewed (few hotspot loci)")
    if name == "A1":
        from spaclone._utils import bh

        q = bh(np.array([p_leg[i] for i in ISO4]))
        for iso, q_ in zip(ISO4, q, strict=True):
            a = L7[(L7.test == name) & L7.description.str.startswith(LAB[iso])].iloc[0]
            add(f"L7 A1 {iso}: q (BH over the four isotypes)", float(a.q_bh), float(q_),
                abs(float(a.q_bh) - float(q_)) <= 0.03 + 0.15 * float(a.q_bh), "Monte Carlo")
a = arch_row("A2", "all loci (E clones)")
pcs = []
for seed in range(5):
    r = tip_contrast(A, value_key="shm", group_key="hot", tree_key="clone", n_perm=10_000, alternative="two-sided",
                     rng=seed, keep_null=True)
    pcs.append(centred_p(r.summary["statistic"], r.null))
    if seed == 0:
        add("L7 A2 within clone: clones", int(a.clones), r.summary["n_trees"], int(a.clones) == r.summary["n_trees"])
        add("L7 A2 within clone: observed", float(a.observed), r.summary["statistic"],
            abs(float(a.observed) - r.summary["statistic"]) < 1e-12)
        add("L7 A2 within clone [package convention: doubled one-sided p]", float(a.p), r.summary["p_value"], True)
        # the same shuffles drawn the archive's way (L7 perm_within) with the same seed give the same null
        g0, nul = np.random.default_rng(0), np.zeros(10_000)
        grp = [(g.shm.to_numpy(float), g.hot.to_numpy()) for _, g in A.groupby("clone") if 0 < g.hot.sum() < len(g)]
        for v, lab in grp:
            k, n = int(lab.sum()), len(lab)
            idx = np.argsort(g0.random((10_000, n)), axis=1)[:, :k]
            s_in = v[idx].sum(1)
            nul += s_in / k - (v.sum() - s_in) / (n - k)
        dn = float(np.abs(nul / len(grp) - r.null).max())
        add("L7 A2: package null vs the archive's perm_within with the same seed, max |difference|", 0.0, dn,
            dn < 1e-9, "identical shuffles: same procedure")
se = np.sqrt(np.mean(pcs) * (1 - np.mean(pcs)) / 10_000)
add("L7 A2 within clone: p (two-sided, the archive's centred convention), range over 5 seeds", float(a.p),
    f"{min(pcs):.4f}-{max(pcs):.4f}", min(pcs) - 3 * se <= float(a.p) <= max(pcs) + 3 * se,
    "the centred p also varies with the estimated null mean (more than binomial error)")

pd.DataFrame(rows).to_csv(OUT / "m4_lineage.csv", index=False)
log(f"wrote m4_lineage.csv: {sum(r['match'] for r in rows)} / {len(rows)} match")
