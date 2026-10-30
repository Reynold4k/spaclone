import numpy as np
import pandas as pd
import pytest

from spaclone._utils import SpacloneWarning
from spaclone.tl import patristic_distances, tip_contrast, tip_region_test, tree_tissue_distance

TOL = 3.1  # binomial 99.9 % one-sided tolerance in standard errors


def test_newick_patristic_known_values():
    d = patristic_distances("((a:1,b:2)x:1,(c:0.5,'d e':1.5):2,Germline:0)root;")
    assert list(d.index) == ["root", "x", "a", "b", "c", "d e", "Germline"] or set(d.index) >= {"a", "b", "c"}
    assert d.loc["a", "b"] == pytest.approx(3.0)
    assert d.loc["a", "c"] == pytest.approx(1 + 1 + 2 + 0.5)
    assert d.loc["c", "d e"] == pytest.approx(2.0)
    assert d.loc["a", "Germline"] == pytest.approx(2.0)
    np.testing.assert_allclose(d.to_numpy(), d.to_numpy().T)
    assert (np.diag(d.to_numpy()) == 0).all()
    # comments, missing lengths and no internal labels
    e = patristic_distances("((a,b)[&comment],c:1);")
    assert e.loc["a", "b"] == 0.0 and e.loc["a", "c"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        patristic_distances("((a:1,b:2);")


def test_edge_table_matches_newick():
    edges = pd.DataFrame(
        {
            "node": [1, 2, 3, 4, 5, 6],
            "parent": [np.nan, 1, 1, 2, 2, 3],
            "length": [0.0, 1.0, 2.0, 1.0, 2.0, 0.5],
            "label": ["", "", "", "a", "b", "c"],
        }
    )
    d = patristic_distances(edges)
    n = patristic_distances("((a:1,b:2):1,(c:0.5):2);")
    for p, q in (("a", "b"), ("a", "c"), ("b", "c")):
        assert d.loc[p, q] == pytest.approx(n.loc[p, q])
    with pytest.raises(ValueError, match="one root"):
        patristic_distances(edges.assign(parent=np.nan))


def _tips(seed=0, n_trees=30, n_tips=12):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trees):
        for i in range(n_tips):
            rows.append(
                {
                    "clone_id": f"T{t}",
                    "sequence_id": f"T{t}_{i}",
                    "mut": int(rng.poisson(3)),
                    "switched": bool(rng.random() < 0.4),
                    "iso": rng.choice(["IGHM", "IGHG", "IGHA"]),
                    "x_um": rng.uniform(0, 1000),
                    "y_um": rng.uniform(0, 1000),
                }
            )
    d = pd.DataFrame(rows)
    d["hot"] = rng.random(len(d)) < np.where(d.iso == "IGHM", 0.1, 0.4)
    return d


def test_tip_contrast_known_answer_and_null():
    d = pd.DataFrame(
        {
            "clone_id": ["A"] * 4 + ["B"] * 4,
            "mut": [5, 5, 1, 1, 2, 4, 2, 0],
            "hot": [True, True, False, False, True, False, False, False],
        }
    )
    r = tip_contrast(d, value_key="mut", group_key="hot", n_perm=2000, rng=0)
    # A: 5 - 1 = 4; B: 2 - 2 = 0; mean 2
    assert r.summary["statistic"] == pytest.approx(2.0)
    assert list(r.table.statistic) == [4.0, 0.0]
    assert r.summary["null_mean"] == pytest.approx(0.0, abs=0.15)
    # trees without both groups are left out
    d2 = pd.concat([d, pd.DataFrame({"clone_id": ["C"] * 3, "mut": [1, 2, 3], "hot": [False] * 3})])
    assert tip_contrast(d2, value_key="mut", group_key="hot", n_perm=50, rng=0).summary["n_trees"] == 2


def test_tip_contrast_calibration_and_power():
    hits = 0
    for seed in range(60):
        d = _tips(seed=seed, n_trees=15, n_tips=8)
        p = tip_contrast(d, value_key="mut", group_key="switched", n_perm=199, rng=seed).summary["p_value"]
        hits += p <= 0.05
    assert hits / 60 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 60)
    d = _tips(seed=99)
    d.loc[d.switched, "mut"] += 2
    assert tip_contrast(d, value_key="mut", group_key="switched", n_perm=999, rng=1).summary["p_value"] <= 0.002


def test_tip_contrast_deterministic():
    d = _tips(seed=3)
    a = tip_contrast(d, value_key="mut", group_key="hot", n_perm=300, rng=7)
    b = tip_contrast(d, value_key="mut", group_key="hot", n_perm=300, rng=7)
    c = tip_contrast(d, value_key="mut", group_key="hot", n_perm=300, rng=8)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.summary["p_value"] == b.summary["p_value"]
    assert a.summary["null_mean"] != c.summary["null_mean"]


def test_tip_region_test_expected_exact_and_mc_agree():
    d = _tips(seed=4)
    ex = tip_region_test(d, region_key="hot", covariate_key="iso", unit_key=None)
    mc = tip_region_test(d, region_key="hot", covariate_key="iso", method="mc", n_perm=20_000, rng=0, unit_key=None)
    np.testing.assert_allclose(ex.table.expected, mc.table.expected)
    np.testing.assert_allclose(ex.table.p_value, mc.table.p_value, atol=0.02)
    # expectation = the tips' own isotype shares when no reference is given
    share = d.groupby("iso").hot.mean()
    first = d[d.clone_id == "T0"]
    assert ex.table.set_index("tree").loc["T0", "expected"] == pytest.approx(first.iso.map(share).sum())
    # a reference with other shares changes the expectation; unseen classes are an error
    ref = pd.DataFrame({"iso": ["IGHM", "IGHG", "IGHA"] * 10, "hot": [False, True, False] * 10})
    r = tip_region_test(d, region_key="hot", covariate_key="iso", reference=ref, unit_key=None)
    assert r.table.set_index("tree").loc["T0", "expected"] == pytest.approx((first.iso == "IGHG").sum())
    with pytest.raises(ValueError, match="do not occur"):
        tip_region_test(d, region_key="hot", covariate_key="iso", reference=ref[ref.iso != "IGHA"], unit_key=None)


def test_tip_region_test_power_and_calibration():
    d = _tips(seed=5, n_trees=40)
    d.loc[d.clone_id == "T3", "hot"] = True  # one tree entirely in the region
    r = tip_region_test(d, region_key="hot", covariate_key="iso", unit_key=None)
    t = r.table.set_index("tree")
    assert t.loc["T3", "significant"]
    assert t.drop(index="T3").significant.sum() <= 1
    # calibration: region flags redrawn from the expectation
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(20):
        p_iso = d.iso.map({"IGHM": 0.1, "IGHG": 0.4, "IGHA": 0.4})
        dd = d.assign(hot=rng.random(len(d)) < p_iso)
        ref = dd.copy()
        ps.extend(
            tip_region_test(dd, region_key="hot", covariate_key="iso", reference=ref, unit_key=None).table.p_value
        )
    ps = np.asarray(ps)
    assert (ps <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(ps))


def test_tip_region_test_labels_units_and_within_tree():
    d = _tips(seed=6)
    d["location"] = np.where(d.hot, "hotspot", "follicle")
    a = tip_region_test(d, region_key="location", region="hotspot", covariate_key="iso", unit_key=None)
    b = tip_region_test(d, region_key="hot", covariate_key="iso", unit_key=None)
    pd.testing.assert_frame_equal(a.table, b.table)
    # two tips per locus count once with unit_key
    dd = pd.concat([d, d.assign(sequence_id=d.sequence_id + "b")], ignore_index=True)
    dd["locus_id"] = np.r_[np.arange(len(d)), np.arange(len(d))]
    u = tip_region_test(dd, region_key="hot", covariate_key="iso", unit_key="locus_id")
    np.testing.assert_allclose(u.table.statistic, b.table.statistic)
    w = tip_region_test(
        d, region_key="hot", covariate_key="switched", null="within_tree", n_perm=500, rng=0, unit_key=None
    )
    c = tip_contrast(d.assign(h=d.hot.astype(float)), value_key="h", group_key="switched", n_perm=500, rng=0)
    assert w.summary["statistic"] == pytest.approx(c.summary["statistic"])
    assert w.summary["p_value"] == pytest.approx(c.summary["p_value"])
    with pytest.raises(ValueError, match="covariate_key"):
        tip_region_test(d, region_key="hot", null="within_tree", unit_key=None)


def _lineage_tree(rng, n_tips=10, clustered=True):
    """A ladder-like tree; tip i sits at x = 30 i (tree neighbours are tissue neighbours) when clustered."""
    newick = "Germline:0"
    labels = [f"t{i}" for i in range(n_tips)]
    sub = f"{labels[0]}:1"
    for lab in labels[1:]:
        sub = f"({sub},{lab}:1):1"
    newick = f"({sub},Germline:0);"
    x = np.arange(n_tips) * 30.0 if clustered else rng.permutation(np.arange(n_tips)) * 30.0
    return newick, labels, x


def test_tree_tissue_distance_power_null_and_units():
    rng = np.random.default_rng(0)
    rows, trees = [], {}
    for t in range(12):
        nwk, labs, x = _lineage_tree(rng, clustered=True)
        trees[f"C{t}"] = nwk
        for i, lab in enumerate(labs):
            rows.append({"clone_id": f"C{t}", "sequence_id": lab, "x_um": x[i], "y_um": 0.0, "locus_id": f"C{t}_{i}"})
    tips = pd.DataFrame(rows)
    r = tree_tissue_distance(tips, trees, unit_key="locus_id", n_perm=199, rng=0)
    assert r.summary["n_trees"] == 12
    assert r.summary["statistic"] > 0.5 and r.summary["p_value"] <= 0.01
    # positions unrelated to the tree: calibrated
    hits = 0
    for seed in range(40):
        g = np.random.default_rng(seed)
        tt = tips.assign(x_um=g.uniform(0, 1000, len(tips)), y_um=g.uniform(0, 1000, len(tips)))
        hits += tree_tissue_distance(tt, trees, unit_key="locus_id", n_perm=99, rng=seed).summary["p_value"] <= 0.05
    assert hits / 40 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 40)
    # tips sharing a label (collapsed sequences) and pairs inside one unit are handled
    dup = pd.concat([tips, tips[tips.sequence_id == "t3"].assign(locus_id=lambda d: d.locus_id)], ignore_index=True)
    r2 = tree_tissue_distance(dup, trees, unit_key="locus_id", n_perm=49, rng=0, shuffle="tips")
    assert (r2.table.n_tips == 11).all()
    # unknown labels are dropped and counted
    bad = pd.concat(
        [
            tips,
            pd.DataFrame({"clone_id": ["C0"], "sequence_id": ["zz"], "x_um": [0.0], "y_um": [0.0], "locus_id": ["zz"]}),
        ],
        ignore_index=True,
    )
    r3 = tree_tissue_distance(bad, trees, unit_key="locus_id", n_perm=19, rng=0)
    assert r3.table.set_index("tree").loc["C0", "n_dropped"] == 1


def test_tree_tissue_distance_matrix_input_and_determinism():
    rng = np.random.default_rng(1)
    nwk, labs, x = _lineage_tree(rng, n_tips=8)
    tips = pd.DataFrame({"clone_id": "A", "sequence_id": labs, "x_um": x, "y_um": 0.0})
    mat = patristic_distances(nwk)
    a = tree_tissue_distance(tips, {"A": nwk}, n_perm=99, rng=3)
    b = tree_tissue_distance(tips, {"A": mat}, n_perm=99, rng=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    c = tree_tissue_distance(tips, {"A": nwk}, n_perm=99, rng=4)
    assert a.table.statistic.iloc[0] == c.table.statistic.iloc[0]
    assert a.table.null_mean.iloc[0] != c.table.null_mean.iloc[0]
    assert "TestResult(tree_tissue_distance" in repr(a)


# ------------------------------------------------------------------------------- cell-scale units by default (F4a)


def _cell_tips(rng, n_trees=150, boost=0, dnb=False, iso_indep=False):
    """Trees of 3-8 cells with 1 + Poisson(2) tips (sequences) per cell at the cell's spot (bead data) or on 0.5-um
    spots within 4 um (``dnb``); a cell lies in the region with the reference share of its isotype (or 0.25 for both
    isotypes, ``iso_indep``); the first ``boost`` trees have every cell in the region with probability 0.6."""
    rows = []
    for tr in range(n_trees):
        for c in range(int(rng.integers(3, 9))):
            iso = rng.choice(["IgM", "IgG"], p=[0.6, 0.4])
            p_in = 0.6 if tr < boost else (0.25 if iso_indep else (0.15 if iso == "IgM" else 0.35))
            inside = bool(rng.random() < p_in)
            cx, cy = rng.uniform(0, 3000, 2)
            k = 1 + int(rng.poisson(2))
            if dnb:
                r, a = 4 * np.sqrt(rng.random(k)), rng.uniform(0, 2 * np.pi, k)
                px, py = np.round((cx + r * np.cos(a)) * 2) / 2, np.round((cy + r * np.sin(a)) * 2) / 2
            else:
                px, py = np.full(k, np.round(cx / 10) * 10), np.full(k, np.round(cy / 10) * 10)
            rows += [(f"T{tr}", f"T{tr}c{c}", f"{a:g}_{b:g}", a, b, iso, inside) for a, b in zip(px, py, strict=True)]
    return pd.DataFrame(rows, columns=["clone_id", "cell", "spot_id", "x_um", "y_um", "iso", "gc"])


REF = pd.DataFrame({"iso": ["IgM"] * 20 + ["IgG"] * 20, "gc": [True] * 3 + [False] * 17 + [True] * 7 + [False] * 13})


def test_unit_rules():
    rng = np.random.default_rng(20)
    t = _cell_tips(rng, n_trees=30, dnb=True)
    cells = t.groupby("clone_id").cell.nunique()
    auto = tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF)
    assert auto.params["units"].startswith("loci (15 um") and auto.params["units"].endswith("(auto)")
    assert (auto.table.set_index("tree").n_tips <= cells).all()
    oracle = tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF, unit_key="cell")
    assert oracle.table.n_tips.sum() == t.cell.nunique()
    with_ids = t.assign(locus_id=pd.factorize(t.cell)[0])
    li = tip_region_test(with_ids, region_key="gc", covariate_key="iso", reference=REF)
    assert li.params["units"] == "locus_id (auto)"
    pd.testing.assert_frame_equal(li.table, oracle.table)
    spot = tip_region_test(t.drop(columns=["x_um", "y_um"]), region_key="gc", covariate_key="iso", reference=REF)
    assert spot.params["units"] == "spot_id (auto)" and spot.table.n_tips.sum() > oracle.table.n_tips.sum()
    with pytest.warns(SpacloneWarning, match="independent unit"):
        rows = tip_region_test(t[["clone_id", "iso", "gc"]], region_key="gc", covariate_key="iso", reference=REF)
    assert rows.params["units"] == "rows (no unit column)" and rows.table.n_tips.sum() == len(t)
    rows2 = tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF, unit_key=None)
    pd.testing.assert_frame_equal(rows.table, rows2.table)
    with pytest.raises(KeyError):
        tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF, unit_key="nope")
    with pytest.raises(KeyError, match="coordinates"):
        tip_region_test(t.drop(columns="x_um"), region_key="gc", covariate_key="iso", unit_key="loci")


def test_cells_not_sequences_are_the_unit():
    """Several sequences per cell: tips as units reject far more than alpha; the default (cells) does not."""
    rng = np.random.default_rng(21)
    p_auto, p_rows = [], []
    for i in range(6):
        t = _cell_tips(rng, n_trees=100, dnb=bool(i % 2))
        p_auto += tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF).table.p_value.tolist()
        p_rows += tip_region_test(
            t, region_key="gc", covariate_key="iso", reference=REF, unit_key=None
        ).table.p_value.tolist()
    p_auto, p_rows = np.asarray(p_auto), np.asarray(p_rows)
    assert (p_auto <= 0.05).mean() <= 0.05 + TOL * np.sqrt(0.0475 / len(p_auto))
    assert (p_rows <= 0.05).mean() > 0.10


@pytest.mark.slow
def test_cells_not_sequences_within_tree_slow():
    rng = np.random.default_rng(24)
    p_auto, p_rows = [], []
    for i in range(150):
        t = _cell_tips(rng, n_trees=60, iso_indep=True, dnb=bool(i % 2))
        kw = {"region_key": "gc", "covariate_key": "iso", "covariate": "IgG", "null": "within_tree", "n_perm": 99}
        p_auto.append(tip_region_test(t, rng=i, **kw).summary["p_value"])
        p_rows.append(tip_region_test(t, rng=i, unit_key=None, **kw).summary["p_value"])
    p_auto, p_rows = np.asarray(p_auto), np.asarray(p_rows)
    assert (p_auto <= 0.05).mean() <= 0.05 + TOL * np.sqrt(0.0475 / len(p_auto))
    assert (p_rows <= 0.05).mean() > 0.10


def test_power_of_cell_units():
    rng = np.random.default_rng(22)
    t = _cell_tips(rng, boost=30)
    r = tip_region_test(t, region_key="gc", covariate_key="iso", reference=REF).table
    boosted = r.tree.str[1:].astype(int) < 30
    assert (r.p_value[boosted] <= 0.05).mean() >= 0.3 and (r.p_value[~boosted] <= 0.05).mean() <= 0.06


def test_tip_contrast_units_and_tree_tissue_distance_auto():
    rng = np.random.default_rng(23)
    t = _cell_tips(rng, n_trees=20)
    t["mut"] = rng.poisson(3, len(t)).astype(float)
    rows = tip_contrast(t, value_key="mut", group_key="gc", n_perm=99, rng=0)
    assert rows.params["units"] == "rows"
    cell = tip_contrast(t, value_key="mut", group_key="gc", n_perm=99, rng=0, unit_key="cell")
    both = rows.table.merge(cell.table, on="tree", suffixes=("", "_cell"))
    assert len(both) and (both.n_tips_cell <= both.n_tips).all() and (both.n_tips_cell < both.n_tips).any()
    assert cell.params["units"] == "cell"
    nwk = {
        c: "(" + ",".join(f"{s}:1" for s in g.seq) + ",Germline:0);"
        for c, g in t.assign(seq=[f"s{i}" for i in range(len(t))]).groupby("clone_id")
    }
    tips = t.assign(sequence_id=[f"s{i}" for i in range(len(t))])
    r = tree_tissue_distance(tips, nwk, n_perm=19, rng=0, min_tips=3)
    assert r.params["units"].startswith("loci") and (r.table.n_units <= r.table.n_tips).all()
