import itertools
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from spaclone import nulls, pl, qc
from spaclone._utils import LowPowerWarning
from spaclone.qc.colocalisation import _PairShare, _same_group_pairs, _shuffle_within, _strata_groups

LEGACY_BANDS = [(0, 100), (0, 20), (20, 100)]
LEGACY = {"units": "records", "sink_min_clones": None}  # the archive's T3 (X5_depth_tests.py)


def _legacy_t3(xy, clone, strata, bead, n_perm, rng):
    """The pair-share test of X5_depth_tests.t3 (archive: external-data validation, workflow), verbatim apart from returning."""
    pairs = cKDTree(xy).query_pairs(100.0, output_type="ndarray")
    dist = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    same_bead = (bead[pairs[:, 0]] == bead[pairs[:, 1]]) if bead is not None else np.zeros(len(pairs), bool)
    if bead is not None:
        sb = pd.DataFrame({"b": bead, "i": np.arange(len(bead))})
        sbp = sb.merge(sb, on="b")
        sbp = sbp[sbp.i_x < sbp.i_y][["i_x", "i_y"]].to_numpy()
    else:
        sbp = np.empty((0, 2), int)

    def share(lab):
        s = pd.Series(lab).map(pd.Series(lab).value_counts()).to_numpy()
        ok = (s >= 2) & (s <= 50)
        tot = sum(n * (n - 1) / 2 for n in pd.Series(lab[ok]).value_counts().to_numpy())
        if len(sbp):
            tot -= int(((lab[sbp[:, 0]] == lab[sbp[:, 1]]) & ok[sbp[:, 0]]).sum())
        a, b = pairs[:, 0], pairs[:, 1]
        hit = (lab[a] == lab[b]) & ok[a] & ~same_bead
        return np.array([hit[(dist >= lo) & (dist <= hi)].sum() / tot if tot else np.nan for lo, hi in LEGACY_BANDS])

    obs = share(clone)
    out = {}
    for kind, st in (("stratified", strata), ("whole_sample", np.zeros(len(clone), int))):
        null = []
        idx_by = pd.Series(np.arange(len(clone))).groupby(st).indices
        for _ in range(n_perm):
            lab = clone.copy()
            for _, ix in idx_by.items():
                lab[ix] = clone[rng.permutation(ix)]
            null.append(share(lab))
        out[kind] = np.array(null)
    return obs, out


def _units(n_beads=150, seed=0, clustered=False, extent=600.0):
    """(bead, clone) units: several clones per bead, clone sizes 1..60, integer depth strata per bead."""
    rng = np.random.default_rng(seed)
    bx, by = rng.uniform(0, extent, n_beads), rng.uniform(0, extent, n_beads)
    k = rng.integers(1, 4, n_beads)
    bead = np.repeat(np.arange(n_beads), k)
    n = len(bead)
    clone = rng.choice(np.r_[np.arange(12), np.arange(40)], n)  # clones 0-11 are larger
    if clustered:  # beads holding a clone move next to that clone's centre
        cx, cy = rng.uniform(0, extent, 40), rng.uniform(0, extent, 40)
        x = cx[clone] + rng.normal(0, 15, n)
        y = cy[clone] + rng.normal(0, 15, n)
        bead = np.arange(n)
    else:
        x, y = bx[bead], by[bead]
    df = pd.DataFrame({"spot_id": [f"b{b}" for b in bead], "x_um": x, "y_um": y, "locus": "IGH"})
    df["clone_id"] = [f"c{c}" for c in clone]
    df = df.drop_duplicates(["spot_id", "clone_id"]).reset_index(drop=True)
    ig = rng.integers(0, 4, df.spot_id.nunique())
    df["ig_bin"] = df.spot_id.map(dict(zip(df.spot_id.unique(), ig, strict=True)))
    return df


def test_known_answer_and_size_window():
    df = pd.DataFrame(
        {
            "spot_id": list("abcdef"),
            "x_um": [0.0, 10.0, 150.0, 0.0, 0.0, 300.0],
            "y_um": [0.0, 0.0, 0.0, 50.0, 55.0, 300.0],
            "locus": "IGH",
            "clone_id": ["A", "A", "A", "B", "B", "C"],
        }
    )
    r = qc.colocalisation(df, n_perm=5, rng=0, **LEGACY)
    t = r.table[r.table.design == "whole_sample"].set_index("band")
    assert t.loc["0-20", "pairs_total"] == 4
    assert t.loc["0-20", "pairs_within"] == 2
    assert t.loc["0-20", "statistic"] == pytest.approx(0.5)
    assert t.loc["20-100", "statistic"] == 0.0
    assert t.loc["0-20", "n_clones_in_window"] == 2
    r2 = qc.colocalisation(df, max_clone_size=2, n_perm=5, rng=0, **LEGACY)
    t2 = r2.table.set_index("band")
    assert t2.loc["0-20", "pairs_total"] == 1
    assert t2.loc["0-20", "statistic"] == pytest.approx(1.0)
    u = r2.units.set_index("spot_id")
    assert u.loc["a", "clone_size"] == 3 and not u.loc["a", "in_window"]
    assert u.loc["d", "in_window"] and not u.loc["f", "in_window"]


def test_band_edges():
    df = pd.DataFrame(
        {"spot_id": ["a", "b"], "x_um": [0.0, 20.0], "y_um": [0.0, 0.0], "locus": "IGH", "clone_id": ["A", "A"]}
    )
    for closed, expect in (("both", (1, 1)), ("left", (0, 1)), ("right", (1, 0))):
        t = qc.colocalisation(df, closed=closed, n_perm=3, rng=0, **LEGACY).table
        assert tuple(t.pairs_within) == expect


def test_same_group_pairs_enumerates_all_pairs():
    g = np.array([3, 1, 3, 3, 2, 1, 7, 3])
    got = {tuple(p) for p in _same_group_pairs(g)}
    want = {(i, j) for i, j in itertools.combinations(range(len(g)), 2) if g[i] == g[j]}
    assert got == want
    assert len(_same_group_pairs(np.arange(5))) == 0


def test_same_spot_pairs_leave_numerator_and_denominator():
    # spot s1 holds A and B; under the labelling (A, A, B, B) s1 holds A twice
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    spot = np.array([0, 0, 1, 2])
    kw = {"min_size": 2, "max_size": 50, "closed": "both"}
    ps = _PairShare(xy, _same_group_pairs(spot), np.array([0, 1, 0, 1]), [(0, 20)], **kw)
    num, den, _ = ps(np.array([0, 1, 0, 1]))
    assert (num[0], den) == (2, 2.0)
    num, den, _ = ps(np.array([0, 0, 1, 1]))  # s1 gets A twice: that pair counts nowhere; B's pair still counts
    assert (num[0], den) == (1, 1.0)
    ps_in = _PairShare(xy, np.empty((0, 2), int), np.array([0, 1, 0, 1]), [(0, 20)], **kw)
    num, den, _ = ps_in(np.array([0, 0, 1, 1]))
    assert (num[0], den) == (2, 2.0)
    # a distance below which pairs count nowhere (loci closer than the merge radius would have been one locus)
    ps_r = _PairShare(xy, np.empty((0, 2), int), np.array([0, 1, 0, 1]), [(0, 20)], excl_um=5.0, **kw)
    num, den, _ = ps_r(np.array([0, 1, 0, 1]))
    assert (num[0], den) == (0, 0.0)  # every pair is within 5 um


def test_shuffler_matches_nulls_permute_draw_for_draw():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 9, 300)
    st = rng.integers(0, 4, 300)
    order = np.argsort(st, kind="stable")
    frame = pd.DataFrame({"_code": codes[order], "_stratum": st[order]})
    design = nulls.Permute("_code", within="_stratum")
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    groups = _strata_groups(st)
    for _ in range(5):
        a = _shuffle_within(codes, groups, g1)
        perm = design(frame, g2)["_code"].to_numpy()
        b = np.empty_like(perm)
        b[order] = perm
        np.testing.assert_array_equal(a, b)
        for s in range(4):  # multisets per stratum are kept
            np.testing.assert_array_equal(np.sort(a[st == s]), np.sort(codes[st == s]))


def test_matches_legacy_t3_draw_for_draw():
    df = _units(seed=3)
    # molecule-level input: duplicate some units; the function collapses them back to (spot, clone) units
    mol = pd.concat([df, df.sample(frac=0.5, random_state=1)]).sort_index(kind="stable").reset_index(drop=True)
    res = qc.colocalisation(
        mol, strata="ig_bin", bands_um=LEGACY_BANDS, n_perm=25, rng=np.random.default_rng(7), **LEGACY
    )
    obs, null = _legacy_t3(
        df[["x_um", "y_um"]].to_numpy(),
        df.clone_id.to_numpy(object),
        df.ig_bin.to_numpy(),
        df.spot_id.to_numpy(object),
        25,
        np.random.default_rng(7),
    )
    t = res.table
    for design in ("stratified", "whole_sample"):
        sub = t[t.design == design]
        np.testing.assert_array_equal(sub.statistic.to_numpy(), obs)
        np.testing.assert_allclose(sub.null_mean.to_numpy(), null[design].mean(0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(sub.null_lo.to_numpy(), np.quantile(null[design], 0.025, axis=0), atol=1e-15)
        p_legacy = (1 + (null[design] >= obs).sum(0)) / (1 + len(null[design]))
        np.testing.assert_allclose(sub.p_value.to_numpy(), p_legacy)
    assert (t.n_units == len(df)).all()


def test_expected_matches_monte_carlo_mean():
    # one unit per spot: the analytic expectation is exact, so the MC mean must be within its own error
    df = _units(seed=4, clustered=True).assign(spot_id=lambda d: np.arange(len(d)).astype(str))
    t = qc.colocalisation(df, strata="ig_bin", n_perm=300, rng=2, **LEGACY).table
    se = t.null_sd / np.sqrt(t.n_perm)
    assert (np.abs(t.null_mean - t.expected) <= 4.5 * se + 1e-12).all()
    # with multi-clone beads (or loci closer than the merge radius) the ratio of expectations stays close
    t2 = qc.colocalisation(_units(seed=4), strata="ig_bin", n_perm=300, rng=2, **LEGACY).table
    assert (np.abs(t2.null_mean - t2.expected) <= 4.5 * t2.null_sd / np.sqrt(300) + 0.02 * t2.expected).all()
    t3 = qc.colocalisation(_units(seed=4), strata="ig_bin", n_perm=300, rng=2).table
    t3 = t3[t3.lo_um >= 15]  # loci: pairs below the merge radius count nowhere
    assert (np.abs(t3.null_mean - t3.expected) <= 4.5 * t3.null_sd / np.sqrt(300) + 0.03 * t3.expected).all()


def test_calibration_under_random_labelling():
    """Labels exchangeable within strata: the stratified test rejects at <= alpha + binomial tolerance."""
    reps, n = 100, 160
    rng = np.random.default_rng(11)
    tabs = []
    for r in range(reps):  # each replicate is one library of the same table
        stratum = rng.integers(0, 2, n)
        x = np.where(stratum == 0, rng.uniform(0, 300, n), rng.uniform(300, 600, n))  # territories differ
        y = rng.uniform(0, 300, n)
        clone = np.where(stratum == 0, rng.integers(0, 12, n), 12 + rng.integers(0, 12, n))
        tabs.append(
            pd.DataFrame(
                {
                    "spot_id": np.arange(n).astype(str),
                    "x_um": x,
                    "y_um": y,
                    "locus": "IGH",
                    "clone_id": clone.astype(str),
                    "st": stratum,
                    "library_id": f"r{r:03d}",
                }
            )
        )
    for mode in ("records", "loci"):
        with pytest.warns(LowPowerWarning, match="cannot pass"):  # 100 libraries, 39 shuffles
            t = qc.colocalisation(pd.concat(tabs), strata="st", bands_um=((0, 100),), n_perm=39, rng=rng,
                                  units=mode).table  # fmt: skip
        assert t.library_id.nunique() == reps
        rate = (t[t.design == "stratified"].p_value <= 0.05).mean()
        assert rate <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
        # the whole-sample null ignores the territories that the strata explain, so it rejects often
        assert (t[t.design == "whole_sample"].p_value <= 0.05).mean() > 0.5


def test_power_on_planted_clusters():
    df = _units(seed=5, clustered=True)  # beads: a (spot, clone) unit is a cell
    t = qc.colocalisation(df, strata="ig_bin", n_perm=99, rng=1, units="records").table
    assert (t.p_value <= 0.01).all()
    assert (t.ratio > 3).all()
    assert (t.p_maxT <= 0.01).all()
    assert (t.q_value <= 0.01).all()


def test_determinism_and_invariance():
    df = _units(seed=8)
    r1 = qc.colocalisation(df, strata="ig_bin", n_perm=20, rng=4)
    r2 = qc.colocalisation(df, strata="ig_bin", n_perm=20, rng=4)
    r3 = qc.colocalisation(df, strata="ig_bin", n_perm=20, rng=5)
    pd.testing.assert_frame_equal(r1.table, r2.table)
    np.testing.assert_array_equal(r1.null, r2.null)
    assert not np.array_equal(r1.null, r3.null)
    assert r1.null.shape == (20, len(r1.table))
    shifted = df.assign(x_um=df.x_um + 1e4, y_um=df.y_um - 3e3)
    r4 = qc.colocalisation(shifted, strata="ig_bin", n_perm=20, rng=4)
    np.testing.assert_allclose(r4.table.statistic, r1.table.statistic)
    np.testing.assert_array_equal(r4.null, r1.null)
    shuffled = df.sample(frac=1.0, random_state=2)
    r5 = qc.colocalisation(shuffled, strata="ig_bin", n_perm=20, rng=4)
    np.testing.assert_allclose(r5.table.statistic, r1.table.statistic)
    np.testing.assert_allclose(r5.table.expected, r1.table.expected)
    np.testing.assert_array_equal(r5.null, r1.null)  # loci come in a canonical order: the draws do not move either
    r5r = qc.colocalisation(shuffled, strata="ig_bin", n_perm=20, rng=4, units="records")
    np.testing.assert_allclose(
        r5r.table.statistic, qc.colocalisation(df, strata="ig_bin", n_perm=2, units="records").table.statistic
    )
    # strata given as labels instead of codes: the same strata, so the same observed values and expectations
    r6 = qc.colocalisation(df.assign(ig_bin=nulls.bin_labels(df.ig_bin, (0, 1, 2, 3, np.inf))), strata="ig_bin")
    np.testing.assert_allclose(r6.table.expected, r1.table.expected)


def test_per_clone_contributions_and_libraries():
    df = pd.concat([_units(seed=1).assign(library_id="L1"), _units(seed=2).assign(library_id="L2")])
    with pytest.warns(LowPowerWarning):  # two libraries, ten shuffles: p >= 1/11 cannot pass BH alone
        r = qc.colocalisation(df, strata="ig_bin", n_perm=10, per_clone=True, rng=0, units="records")
    assert set(r.table.library_id) == {"L1", "L2"}
    assert len(r.table) == 2 * 2 * 2
    obs = r.table[r.table.design == "stratified"].set_index(["library_id", "band"]).pairs_within
    pooled = r.clones.groupby(["library_id", "band"]).pairs_within.sum()
    pd.testing.assert_series_equal(pooled.sort_index(), obs.sort_index(), check_names=False)
    contrib = r.clones.groupby(["library_id", "band"]).contribution.sum()
    np.testing.assert_allclose(contrib.to_numpy(), 1.0)
    assert {"null_mean_stratified", "null_mean_whole_sample"} <= set(r.clones.columns)
    assert len(r.units) == len(df.drop_duplicates(["library_id", "spot_id", "clone_id"]))
    # q-values are BH over the two libraries of each design and band
    for _, t in r.table.groupby(["design", "band"]):
        assert (t.q_value >= t.p_value - 1e-12).all()


def test_input_checks():
    df = _units(seed=0)
    bad = df.copy()
    bad.loc[bad.index[0], "ig_bin"] = 99
    dup = pd.concat([bad.iloc[[0]].assign(ig_bin=1), bad])
    with pytest.raises(ValueError, match="vary inside"):
        qc.colocalisation(dup, strata="ig_bin", n_perm=2)
    with pytest.raises(KeyError):
        qc.colocalisation(df, strata="nope", n_perm=2)
    with pytest.raises(ValueError):
        qc.colocalisation(df, bands_um=((20, 10),), n_perm=2)
    with pytest.raises(ValueError):
        qc.colocalisation(df.drop(columns="x_um"), n_perm=2)
    with pytest.raises(ValueError):
        qc.colocalisation(df, min_clone_size=1, n_perm=2)
    with pytest.raises(ValueError, match="aliases"):
        qc.colocalisation(df, strata="ig_bin", within="spot_id", n_perm=2)
    with pytest.raises(ValueError):
        qc.colocalisation(df, units="cells", n_perm=2)
    with pytest.raises(ValueError):
        qc.colocalisation(df, sink_min_clones=0, n_perm=2)
    a = qc.colocalisation(df, strata="ig_bin", n_perm=5, rng=1)
    b = qc.colocalisation(df, within="ig_bin", n_perm=5, rng=1)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_loci_known_answer_window_and_exclusion():
    # clone A: two cells, each covering two spots 2 um apart, 50 um between the cells; clone B: one cell (2 spots) and
    # a second cell 30 um away; clone C: one spot 8 um from A's first cell (closer than the merge radius)
    rows = [
        ("a1", 0.0, 0.0, "A"), ("a2", 2.0, 0.0, "A"), ("a3", 50.0, 0.0, "A"), ("a4", 52.0, 0.0, "A"),
        ("b1", 0.0, 100.0, "B"), ("b2", 0.0, 102.0, "B"), ("b3", 30.0, 100.0, "B"),
        ("c1", 8.0, 0.0, "C"),
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["spot_id", "x_um", "y_um", "clone_id"]).assign(locus="IGH")
    r = qc.colocalisation(df, loci_radius_um=5.0, n_perm=5, rng=0, per_clone=True)
    t = r.table.set_index("band")
    assert (t.n_units == 5).all() and (t.n_spot_clone == 8).all()  # loci A1, A2, B1, B2, C
    assert t.loc["20-100", "pairs_total"] == 2 and t.loc["20-100", "pairs_within"] == 2  # A1-A2 (50 um), B1-B2 (29 um)
    assert t.loc["0-20", "pairs_within"] == 0
    u = r.units.set_index("clone_id")
    assert u.loc["C", "clone_size"] == 1 and not u.loc["C", "in_window"]
    assert u.n_spots.sum() == 8
    # with a 10 um radius C (8 um from A's first cell) is a neighbour of A1: had they carried one clone they would
    # be one locus, so that pair counts nowhere; at 60 um the two cells of A are one locus and A leaves the window
    r60 = qc.colocalisation(df, loci_radius_um=60.0, n_perm=5, rng=0)
    t60 = r60.table.set_index("band")
    assert t60.loc["20-100", "n_units"] == 3 and t60.loc["20-100", "n_clones_in_window"] == 0


def test_sinks_are_removed_first():
    df = _units(seed=8)
    sink = pd.DataFrame({"spot_id": "sink", "x_um": 1.0, "y_um": 1.0, "locus": "IGH", "ig_bin": 0,
                         "clone_id": [f"c{i}" for i in range(25)]})  # fmt: skip
    both = pd.concat([df, sink], ignore_index=True)
    r = qc.colocalisation(both, strata="ig_bin", n_perm=5, rng=0)
    assert r.summary["n_sink_spots"] == 1 and r.summary["n_sink_records"] == 25
    assert "sink" not in set(r.units.spot_id)
    kept = qc.colocalisation(both, strata="ig_bin", n_perm=5, rng=0, sink_min_clones=None)
    assert kept.summary["n_sink_spots"] == 0 and kept.table.n_spot_clone.iloc[0] == r.table.n_spot_clone.iloc[0] + 25


def _cell_chip(seed, n_cells=500, extent=400.0, clustered=False, n_clones=40, spacing=18.0, n_libs=1):
    """Cells on a jittered lattice (18 um apart), each covering 2-6 spots of a 2-um grid within 3 um of its centre.

    ``clustered=False``: clone labels random over cells (a working assay without clonal structure, the null of
    co-localisation beyond one cell). ``clustered=True``: each clone holds the cells of one 60-um patch.
    """
    rng = np.random.default_rng(seed)
    out = []
    for lib in range(n_libs):
        g = np.arange(0, extent, spacing)
        cx, cy = np.meshgrid(g, g)
        cx = (cx.ravel() + rng.uniform(-2, 2, cx.size))[:n_cells]
        cy = (cy.ravel() + rng.uniform(-2, 2, cy.size))[:n_cells]
        n = len(cx)
        if clustered:
            centres = rng.uniform(0, extent, (n_clones, 2))
            lab = np.argmin(np.hypot(cx[:, None] - centres[:, 0], cy[:, None] - centres[:, 1]), axis=1)
            near = np.hypot(cx - centres[lab, 0], cy - centres[lab, 1]) <= 60
            lab = np.where(near, lab, n_clones + np.arange(n))
        else:
            w = 1 / np.arange(1, n_clones + 1)
            lab = np.where(rng.random(n) < 0.6, rng.choice(n_clones, n, p=w / w.sum()), n_clones + np.arange(n))
        k = rng.integers(2, 7, n)
        cell = np.repeat(np.arange(n), k)
        r, t = 3.0 * np.sqrt(rng.random(len(cell))), rng.uniform(0, 2 * np.pi, len(cell))
        x, y = np.rint((cx[cell] + r * np.cos(t)) / 2) * 2, np.rint((cy[cell] + r * np.sin(t)) / 2) * 2
        out.append(pd.DataFrame({"spot_id": [f"{a:g}_{b:g}" for a, b in zip(x, y, strict=True)], "x_um": x, "y_um": y,
                                 "locus": "IGK", "clone_id": [f"c{v}" for v in lab[cell]], "library_id": f"L{lib:03d}"}))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def test_loci_calibrated_when_cells_cover_several_spots():
    """Clone labels random over cells that cover several spots: loci are calibrated at 20-100 um."""
    reps = 60
    air = _cell_chip(0, n_cells=300, extent=320.0, n_libs=reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        t = qc.colocalisation(air, bands_um=((20, 100),), n_perm=39, rng=1, loci_radius_um=8.0).table
    assert len(t) == reps
    assert (t.p_value <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_loci_power_on_clustered_cells():
    air = _cell_chip(1, clustered=True)
    t = qc.colocalisation(air, bands_um=((20, 100),), n_perm=99, rng=2, loci_radius_um=8.0).table.iloc[0]
    assert t.p_value <= 0.01 and t.ratio > 1.5


def test_placement_plot_returns_axes():
    import matplotlib.pyplot as plt

    df = _units(seed=8)
    r = qc.colocalisation(df, strata="ig_bin", n_perm=10, rng=0)
    ax = pl.placement(r)
    assert ax.get_ylabel()
    plt.close(ax.figure)
    two = qc.colocalisation(pd.concat([df.assign(library_id="a"), df.assign(library_id="b")]), n_perm=39, rng=0)
    ax = pl.placement(two, style="dark")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    plt.close(ax.figure)
