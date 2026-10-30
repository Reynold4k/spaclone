import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from spaclone import pl, tl
from spaclone._utils import LowPowerWarning
from spaclone.tl import maps as maps_module
from spaclone.tl.maps import _critical, _MatchedShuffler, _Shuffler

TOL = 3.1  # binomial 99.9% one-sided tolerance in standard errors


def _field(n=1500, size=300.0, p=0.1, seed=0, hot=None, cold=None):
    """Uniform units; inside ``hot`` / ``cold`` = (cx, cy, r, p) discs the mark probability is changed."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, size, n), rng.uniform(0, size, n)
    prob = np.full(n, p)
    for disc in (hot, cold):
        if disc is not None:
            cx, cy, r, q = disc
            prob[(x - cx) ** 2 + (y - cy) ** 2 < r**2] = q
    return pd.DataFrame({"x_um": x, "y_um": y, "mark": rng.random(n) < prob})


def _brute_force(t, mark, p, grid, sigma_um, truncate=4.0):
    """Explicit sums over units with the discrete Gaussian weights (truncated, zero outside the grid)."""
    sg = sigma_um / grid.pixel_um
    r = int(truncate * sg + 0.5)
    k = np.arange(-r, r + 1)
    phi = np.exp(-0.5 * k**2 / sg**2)
    phi /= phi.sum()
    i, j, inside = grid.index(t.x_um, t.y_um)
    out = {key: np.zeros(grid.shape) for key in ("n1", "n2", "s1", "e", "v")}
    for iu, ju, mu, pu in zip(i[inside], j[inside], mark[inside], p[inside], strict=True):
        for a in k:
            for b in k:
                ii, jj = iu + a, ju + b
                if 0 <= ii < grid.shape[0] and 0 <= jj < grid.shape[1]:
                    w = phi[a + r] * phi[b + r]
                    out["n1"][ii, jj] += w
                    out["n2"][ii, jj] += w * w
                    out["s1"][ii, jj] += w * mu
                    out["e"][ii, jj] += w * pu
                    out["v"][ii, jj] += w * w * pu * (1 - pu)
    return out


def test_share_map_equals_brute_force_sums():
    t = _field(n=60, size=150.0, p=0.3, seed=2)
    t["cls"] = np.where(np.arange(60) % 3 == 0, "a", "b")
    grid = tl.Grid((0.0, 0.0), 10.0, (15, 15))
    m = t.mark.to_numpy(float)
    cls_share = t.groupby("cls").mark.mean()
    rng = np.random.default_rng(0)
    given = rng.uniform(0.05, 0.6, len(t))
    for kw, p in (
        ({}, np.full(len(t), m.mean())),
        ({"match_by": "cls"}, t.cls.map(cls_share).to_numpy(float)),
        ({"expected": given}, given),
    ):
        sm = tl.share_map(t, mark="mark", sigma_um=20.0, grid=grid, min_neff=0.0, **kw)
        bf = _brute_force(t, m, p, grid, 20.0)
        ok = bf["n1"] > 1e-12
        np.testing.assert_allclose(sm.share[ok], bf["s1"][ok] / bf["n1"][ok], rtol=1e-10)
        np.testing.assert_allclose(sm.expected[ok], bf["e"][ok] / bf["n1"][ok], rtol=1e-10)
        np.testing.assert_allclose(sm.n_eff[ok], bf["n1"][ok] ** 2 / bf["n2"][ok], rtol=1e-10)
        v = bf["v"] > 0
        np.testing.assert_allclose(sm.z[v], (bf["s1"][v] - bf["e"][v]) / np.sqrt(bf["v"][v]), rtol=1e-8, atol=1e-10)
        assert np.isnan(sm.share[~ok]).all()
    assert sm.p0 == pytest.approx(m.mean()) and sm.n_units == 60 and sm.n_marked == int(m.sum())


def test_neff_counts_the_units_of_an_isolated_pixel():
    # 7 units in one pixel and 3 in another far beyond the kernel window: n_eff is exactly 7 and 3 there
    x = np.r_[np.full(7, 15.0), np.full(3, 985.0)]
    t = pd.DataFrame({"x_um": x, "y_um": np.full(10, 15.0), "mark": np.arange(10) % 2 == 0})
    sm = tl.share_map(t, mark="mark", sigma_um=20.0, pixel_um=10.0, min_neff=0.0)
    assert sm.n_eff[0, 0] == pytest.approx(7.0, rel=1e-12)
    assert sm.n_eff[-1, 0] == pytest.approx(3.0, rel=1e-12)
    assert sm.share[0, 0] == pytest.approx(4 / 7) and sm.share[-1, 0] == pytest.approx(1 / 3)


def test_legacy_neff_kernel_is_the_archive_formula():
    t = _field(n=300, size=200.0, seed=4)
    sm = tl.share_map(t, mark="mark", sigma_um=30.0, neff_kernel="legacy", min_neff=0.0)
    g = sm.grid
    i, j, _ = g.index(t.x_um, t.y_um)
    cnt = np.zeros(g.shape)
    np.add.at(cnt, (i, j), 1.0)
    sg = 3.0
    n1 = ndi.gaussian_filter(cnt, sg, mode="constant")
    n2 = ndi.gaussian_filter(cnt, sg / np.sqrt(2), mode="constant") / (4 * np.pi * sg**2)
    np.testing.assert_array_equal(sm.n_eff, np.where(n2 > 0, n1**2 / np.maximum(n2, 1e-300), 0))
    exact = tl.share_map(t, mark="mark", sigma_um=30.0, min_neff=0.0)
    # far from units the legacy window (2.83 sigma) inflates n_eff; near them the two agree
    dense = exact.n_eff > 20
    np.testing.assert_allclose(sm.n_eff[dense], exact.n_eff[dense], rtol=0.02)


def test_share_map_sample_raster_and_off_grid_units(caplog):
    t = _field(n=400, size=200.0, seed=5)
    grid = tl.Grid((0.0, 0.0), 10.0, (10, 20))  # covers x < 100 only
    with caplog.at_level("WARNING", logger="spaclone"):
        sm = tl.share_map(t, mark="mark", grid=grid, sigma_um=20.0)
    assert "fall off the grid" in caplog.text
    assert sm.n_units == int((t.x_um < 100).sum()) and sm.params["n_off_grid"] == int((t.x_um >= 100).sum())
    v = sm.sample([5.0, 55.0, 150.0], [5.0, 105.0, 5.0], layer="n_eff", fill=-1.0)
    assert v[0] == sm.n_eff[0, 0] and v[1] == sm.n_eff[5, 10] and v[2] == -1.0
    assert sm.raster("z").values is sm.z
    with pytest.raises(ValueError):
        sm.raster("nope")
    assert "ShareMap(" in repr(sm)


def test_share_map_is_invariant_to_row_order_and_translation():
    t = _field(n=500, seed=6)
    a = tl.share_map(t, mark="mark")
    b = tl.share_map(t.sample(frac=1.0, random_state=1), mark="mark")
    c = tl.share_map(t.assign(x_um=t.x_um + 5000.0, y_um=t.y_um - 300.0), mark="mark")
    for key in ("share", "z", "n_eff", "eligible"):
        np.testing.assert_allclose(getattr(a, key), getattr(b, key), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(getattr(a, key), getattr(c, key), rtol=1e-12, equal_nan=True)


def test_tissue_raster_sets_the_grid_and_the_eligible_pixels():
    t = _field(n=800, size=200.0, seed=7)
    grid = tl.Grid((-50.0, -50.0), 10.0, (30, 30))
    tissue = np.zeros(grid.shape, bool)
    tissue[5:20, 5:25] = True
    sm = tl.share_map(t, mark="mark", tissue=tl.Raster(tissue, grid), sigma_um=20.0, min_neff=5.0)
    assert sm.grid == grid
    assert (sm.eligible <= tissue).all() and sm.eligible.any()
    with pytest.raises(ValueError, match="different grid"):
        tl.share_map(t, mark="mark", grid=tl.Grid((0.0, 0.0), 10.0, (30, 30)), tissue=tl.Raster(tissue, grid))


def test_input_errors():
    t = _field(n=50, seed=8)
    with pytest.raises(ValueError, match="binary"):
        tl.share_map(t.assign(mark=np.arange(50)), mark="mark")
    with pytest.raises(ValueError, match="either"):
        tl.share_map(t.assign(c="a"), mark="mark", match_by="c", expected=np.full(50, 0.1))
    with pytest.raises(KeyError):
        tl.share_map(t, mark="mark", x_key="x")
    with pytest.raises(ValueError, match="variance is 0"):
        tl.share_map(t.assign(mark=False), mark="mark")


def test_critical_values_rank_and_quantile_rules():
    null = np.arange(1.0, 100.0)  # B = 99
    assert _critical(null, 0.05, "rank") == 95.0  # 5th largest: obs > 95 <=> p = (1 + #{>= obs}) / 100 <= 0.05
    assert _critical(null, 0.05, "quantile") == pytest.approx(np.quantile(null, 0.95))
    assert _critical(null, 0.05, "quantile") < _critical(null, 0.05, "rank")
    with pytest.warns(LowPowerWarning):
        assert _critical(np.arange(10.0), 0.05, "rank") == np.inf


def test_shuffler_keeps_strata_and_blocks():
    rng = np.random.default_rng(0)
    mark = np.r_[np.ones(10), np.zeros(20), np.ones(3), np.zeros(7)]
    strata = np.r_[np.zeros(30, int), np.ones(10, int)]
    sh = _Shuffler(mark, strata, None)
    for _ in range(20):
        d = sh.draw(rng)
        assert d[:30].sum() == 10 and d[30:].sum() == 3
    blocks = np.repeat(np.arange(10), 4)
    bmark = np.repeat(rng.random(10) < 0.5, 4).astype(float)
    sb = _Shuffler(bmark, None, blocks)
    for _ in range(20):
        d = sb.draw(rng)
        assert (d.reshape(10, 4).std(axis=1) == 0).all() and d.sum() == bmark.sum()
    with pytest.raises(ValueError, match="varies inside a block"):
        _Shuffler(np.r_[1.0, 0.0, 0.0, 0.0], None, np.zeros(4, int))
    # without strata a draw is one permutation of the whole mark (the archive's draw order)
    a = _Shuffler(mark, None, None).draw(np.random.default_rng(3))
    np.testing.assert_array_equal(a, mark[np.random.default_rng(3).permutation(len(mark))])


def test_hotspots_finds_a_planted_hotspot():
    t = _field(hot=(150.0, 150.0, 40.0, 0.7), seed=9)
    r = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=99, rng=1)
    assert r.summary["n_hotspots"] >= 1
    top = r.table.iloc[0]
    assert np.hypot(top.x_um - 150.0, top.y_um - 150.0) < 40.0
    assert top.p_value <= 0.05 and top.marked > top.expected and top.ratio > 2
    assert r.labels.sample([150.0], [150.0])[0] == 1
    assert r.threshold["hot"] == r.summary["threshold_hot"] and r.null.shape == (99,)
    assert "TestResult(hotspots" in repr(r)


@pytest.mark.parametrize("reps", [100, pytest.param(1000, marks=pytest.mark.slow)])
def test_hotspots_family_wise_error_under_the_null(reps):
    """Random labelling of an exchangeable mark: the share of maps with any hotspot is at most alpha."""
    rng = np.random.default_rng(10)
    hits_rank, hits_quantile = 0, 0
    for _ in range(reps):
        t = _field(n=300, size=200.0, p=0.2, seed=int(rng.integers(2**31)))
        seed = int(rng.integers(2**31))
        kw = {"mark": "mark", "sigma_um": 20.0, "min_neff": 5.0, "n_perm": 19}
        hits_rank += tl.hotspots(t, rng=seed, **kw).summary["n_hotspots"] > 0
        hits_quantile += tl.hotspots(t, rng=seed, threshold_rule="quantile", **kw).summary["n_hotspots"] > 0
    assert hits_rank / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_quantile >= hits_rank  # the legacy rule's threshold is never above the rank rule's


@pytest.mark.parametrize("reps", [80, pytest.param(600, marks=pytest.mark.slow)])
def test_hotspots_block_shuffles_keep_clone_level_marks_calibrated(reps):
    """A clone-level mark on clones whose units sit together: shuffling whole clones keeps the error rate."""
    rng = np.random.default_rng(11)
    hits_blocks, hits_units = 0, 0
    for _ in range(reps):
        n_cl = 60
        cx, cy = rng.uniform(0, 200, n_cl), rng.uniform(0, 200, n_cl)
        size = rng.integers(1, 9, n_cl)
        clone = np.repeat(np.arange(n_cl), size)
        t = pd.DataFrame(
            {
                "x_um": cx[clone] + rng.normal(0, 3, len(clone)),
                "y_um": cy[clone] + rng.normal(0, 3, len(clone)),
                "clone": clone,
                "mark": (rng.random(n_cl) < 0.3)[clone],
            }
        )
        kw = {"mark": "mark", "sigma_um": 20.0, "min_neff": 3.0, "n_perm": 19, "rng": int(rng.integers(2**31))}
        hits_blocks += tl.hotspots(t, blocks="clone", **kw).summary["n_hotspots"] > 0
        hits_units += tl.hotspots(t, **kw).summary["n_hotspots"] > 0
    assert hits_blocks / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_units > hits_blocks  # unit-level shuffles of a clone-level mark are anti-conservative


def test_hotspots_is_deterministic_given_the_seed():
    t = _field(hot=(150.0, 150.0, 40.0, 0.5), seed=12)
    a = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=39, rng=5)
    b = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=39, rng=np.random.default_rng(5))
    c = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=39, rng=6)
    np.testing.assert_array_equal(a.null, b.null)
    np.testing.assert_array_equal(a.labels.values, b.labels.values)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.null, c.null)


def test_coldspots_two_sided_and_cluster_mass():
    t = _field(p=0.4, cold=(150.0, 150.0, 45.0, 0.0), seed=13)
    r = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=99, alternative="less", rng=2)
    assert r.summary["n_coldspots"] >= 1 and r.summary["n_hotspots"] == 0
    assert r.table.iloc[0].direction == "cold" and r.table.iloc[0].peak_z < r.threshold["cold"] < 0
    both = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=99, alternative="two-sided", rng=2)
    assert both.null.shape == (99, 2) and set(both.threshold) == {"hot", "cold"}
    assert both.summary["n_coldspots"] >= 1
    h = _field(hot=(150.0, 150.0, 40.0, 0.7), seed=9)
    cm = tl.hotspots(h, mark="mark", sigma_um=20.0, n_perm=99, correction="cluster_mass", cluster_z=2.0, rng=3)
    assert cm.summary["n_hotspots"] >= 1 and cm.table.statistic.iloc[0] > cm.threshold["hot"]
    pw = tl.hotspots(h, mark="mark", sigma_um=20.0, n_perm=99, correction=None, rng=3)
    assert pw.summary["n_hotspots"] >= 1 and pw.threshold["hot"] == 0.05


def test_null_marks_replay_equals_the_internal_draws():
    t = _field(hot=(150.0, 150.0, 40.0, 0.6), seed=14)
    m = t.mark.to_numpy()
    g = np.random.default_rng(7)
    replay = np.array([m[g.permutation(len(m))] for _ in range(49)])
    a = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=49, rng=7)
    b = tl.hotspots(t, mark="mark", sigma_um=20.0, null_marks=replay)
    np.testing.assert_array_equal(a.null, b.null)
    assert a.threshold == b.threshold
    with pytest.raises(ValueError, match="null_marks"):
        tl.hotspots(t, mark="mark", null_marks=replay[:, :10])


def test_match_by_shuffles_within_class():
    t = _field(n=1200, seed=15)
    t["iso"] = np.where(t.x_um < 150, "IgG", "IgM")  # class confounded with position
    t["mark"] = np.random.default_rng(1).random(len(t)) < np.where(t.iso == "IgG", 0.5, 0.05)
    raw = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=49, rng=1)
    iso = tl.hotspots(t, mark="mark", match_by="iso", sigma_um=20.0, n_perm=49, rng=1)
    assert raw.summary["n_hotspots"] >= 1  # the raw map redraws the class map
    assert iso.summary["n_hotspots"] == 0 and iso.params["within"] == "iso"
    # 1.0 block design ("free"): no strata by default, expectation recomputed on every shuffled map
    t["clone"] = np.arange(len(t)) // 3
    t["mark"] = t.groupby("clone").mark.transform("max")
    kw = {"mark": "mark", "match_by": "iso", "blocks": "clone", "sigma_um": 20.0, "n_perm": 19, "rng": 1}
    blk = tl.hotspots(t, block_design="free", **kw)
    assert blk.params["within"] is None and blk.params["design"]["exchangeable"] == "blocks"
    assert blk.params["standardise"] == "binomial"
    with pytest.raises(ValueError, match="stratum varies inside a block"):
        tl.hotspots(t, block_design="free", within="iso", **kw)
    # matched design (default with blocks): strata = match_by even when they vary inside a clone, studentised map
    mat = tl.hotspots(t, **kw)
    assert mat.params["within"] == "iso" and mat.params["block_design"] == "matched"
    assert mat.params["standardise"] == "permutation" and mat.z_map is not None


def test_region_excess_known_answer():
    cls = np.r_[["A"] * 20, ["B"] * 20]
    mark = np.r_[np.ones(10), np.zeros(10), np.ones(2), np.zeros(18)].astype(bool)
    region = np.zeros(40, int)
    region[[0, 1, 2, 3, 4, 10, 11, 12, 20, 21]] = 1  # 8 A (5 marked) + 2 B (2 marked)
    t = pd.DataFrame({"x_um": 0.0, "y_um": 0.0, "cls": cls, "mark": mark, "region": region})
    r = tl.region_excess(t, mark="mark", regions="region", match_by="cls", n_perm=4000, min_units=5, rng=0)
    row = r.table.iloc[0]
    assert row.region == 1 and row.units == 10 and row.marked == 7 and row.statistic == 7
    assert row.expected == pytest.approx(8 * 0.5 + 2 * 0.1)
    assert row.expected_raw == pytest.approx(0.3 * 10)
    assert row.ratio == pytest.approx(7 / 4.2)
    assert row.null_mean == pytest.approx(4.2, abs=4 * row.null_sd / np.sqrt(4000))
    assert r.params["within"] == "cls" and list(r.table.region) == [1, "all"]
    assert np.isnan(r.table.q_value.iloc[-1]) and r.summary["pooled_marked"] == 7


@pytest.mark.parametrize("reps", [120, pytest.param(1000, marks=pytest.mark.slow)])
def test_region_excess_calibration_and_power(reps):
    rng = np.random.default_rng(16)
    hits = 0
    region = np.repeat(np.arange(4), 25)
    for _ in range(reps):
        t = pd.DataFrame({"x_um": 0.0, "y_um": 0.0, "mark": rng.random(100) < 0.2, "region": region})
        r = tl.region_excess(t, mark="mark", regions="region", n_perm=99, rng=int(rng.integers(2**31)))
        hits += r.table.p_value.iloc[0] <= 0.05
    assert hits / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    mark = np.random.default_rng(17).random(100) < np.where(region == 2, 0.7, 0.2)  # planted excess in region 2
    t = pd.DataFrame({"x_um": 0.0, "y_um": 0.0, "mark": mark, "region": region})
    r = tl.region_excess(t, mark="mark", regions="region", n_perm=999, rng=1)
    top = r.table.set_index("region").loc[2]
    assert top.p_value <= 0.01 and top.q_value <= 0.05 and top.ratio > 1.5


def test_region_excess_accepts_rasters_hotspot_results_and_string_ids():
    t = _field(hot=(150.0, 150.0, 40.0, 0.7), seed=9)
    hs = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=99, rng=1)
    t["other"] = np.random.default_rng(2).random(len(t)) < 0.3
    a = tl.region_excess(t, mark="other", regions=hs, n_perm=99, rng=3)
    b = tl.region_excess(t, mark="other", regions=hs.labels, n_perm=99, rng=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    lab = hs.labels.sample(t.x_um, t.y_um, fill=0)
    s = tl.region_excess(t, mark="other", regions=np.where(lab > 0, "hs" + lab.astype(str), ""), n_perm=99, rng=3)
    np.testing.assert_array_equal(s.table.marked.to_numpy(), a.table.marked.to_numpy())


def test_plots_return_axes():
    t = _field(hot=(150.0, 150.0, 40.0, 0.7), seed=9)
    hs = tl.hotspots(t, mark="mark", sigma_um=20.0, n_perm=19, rng=1)
    mask = tl.Raster(hs.map.n_eff > 30, hs.map.grid)
    ax = pl.share_map(hs, style="dark", outlines={"dense": mask})
    assert ax.figure.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    ax2 = pl.share_map(hs.map, layer="z", receptor="TCR", colorbar=False)
    fig, ax3 = plt.subplots()
    assert pl.share_map(hs, layer="n_eff", ax=ax3) is ax3
    t["d"] = t.x_um - 150.0
    prof = tl.niche_profile(t, group="mark", distance_um="d", bands_um=[-100, -50, 0, 50, 100], n_perm=39, rng=0)
    ax4 = pl.niche_profile(prof, style="dark")
    t["c"] = np.where(t.x_um < 150, "left", "right")
    ax5 = pl.niche_profile(tl.niche_profile(t, group="mark", compartment="c", n_perm=39, rng=0), ratio=False)
    assert all(hasattr(a, "plot") for a in (ax, ax2, ax4, ax5))
    with pytest.raises(ValueError):
        pl.share_map(hs, layer="bad")
    plt.close("all")


# ------------------------------------------------------------------------------- clone-level (matched) block design


def test_matched_shuffler_trades_only_within_signatures():
    """Known answer: two 2-unit clones swap; two (1 + 2)-unit clones swap and re-deal stratum 1; a unique clone stays."""
    blocks = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4])
    strata = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
    mark = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], float)
    sh = _MatchedShuffler(mark, strata, blocks)
    rng = np.random.default_rng(0)
    seen = {"ab": set(), "cd": set()}
    for _ in range(400):
        d = sh.draw(rng)
        for s_ in (0, 1):  # every stratum keeps its marked count
            assert d[strata == s_].sum() == mark[strata == s_].sum()
        assert d[10] == 1.0  # unique signature, one unit: fixed
        seen["ab"].add(tuple(d[:4]))
        seen["cd"].add(tuple(d[4:10]))
        # the receiving clone of the (1 IgM-like + 2) signature holds the donor's stratum-0 mark on its stratum-0 unit
        assert d[4] + d[7] == 1.0 and d[5] + d[6] + d[8] + d[9] == 1.0
    assert seen["ab"] == {(1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)}
    assert len(seen["cd"]) == 4
    assert sh.movable.tolist() == [True] * 10 + [False]
    # without peers and with a constant mark nothing moves
    fixed = _MatchedShuffler(np.array([1.0, 1.0, 0.0]), None, np.array([0, 0, 1]))
    assert not fixed.movable.any()
    np.testing.assert_array_equal(fixed.draw(rng), [1.0, 1.0, 0.0])


def _clone_field(rng, n_cl=70, size_effect=True, mixed=0.3, place=0.0, side=200.0):
    """Clones of 1-8 loci within ~4 um of their centre; isotype per locus (mixed inside some clones); a clone-level
    mark that depends on isotype and clone size (and, with ``place``, on a disc at the centre)."""
    cx, cy = rng.uniform(0, side, n_cl), rng.uniform(0, side, n_cl)
    size = rng.integers(1, 9, n_cl)
    clone = np.repeat(np.arange(n_cl), size)
    c_igg = rng.random(n_cl) < np.where(cx < side / 3, 0.7, 0.2)
    mix = rng.random(n_cl) < mixed
    igg = np.where(mix[clone], rng.random(len(clone)) < 0.5, c_igg[clone])
    lp = -1.8 + (0.35 * size if size_effect else 0.0) + rng.normal(0, 1.0, n_cl)
    lp = lp + place * ((cx - side / 2) ** 2 + (cy - side / 2) ** 2 < (side / 5) ** 2)
    p = 1 / (1 + np.exp(-(lp[clone] + 1.2 * igg)))
    clone_draw = rng.random(n_cl)
    mark = np.where(mix[clone], rng.random(len(clone)) < p, clone_draw[clone] < p)
    return pd.DataFrame(
        {
            "x_um": cx[clone] + rng.normal(0, 4, len(clone)),
            "y_um": cy[clone] + rng.normal(0, 4, len(clone)),
            "clone": clone,
            "iso": np.where(igg, "IgG", "IgM"),
            "mark": mark,
        }
    )


@pytest.mark.parametrize("reps", [40, pytest.param(400, marks=pytest.mark.slow)])
def test_matched_block_design_controls_family_wise_error(reps):
    """Null: the mark depends on clone size and isotype (also inside mixed clones), clones sit together, no place
    effect. Matched clone relabelling with the isotype-matched expectation keeps any-hotspot rates at <= alpha;
    shuffling loci one by one does not."""
    rng = np.random.default_rng(21)
    hits_m = hits_u = 0
    for _ in range(reps):
        t = _clone_field(rng)
        kw = {"mark": "mark", "match_by": "iso", "sigma_um": 20.0, "min_neff": 3.0, "n_perm": 19}
        seed = int(rng.integers(2**31))
        hits_m += tl.hotspots(t, blocks="clone", rng=seed, **kw).summary["n_hotspots"] > 0
        hits_u += tl.hotspots(t, rng=seed, **kw).summary["n_hotspots"] > 0
    assert hits_m / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_u > hits_m


def test_matched_block_design_finds_a_place_effect():
    rng = np.random.default_rng(22)
    t = _clone_field(rng, n_cl=400, size_effect=False, mixed=0.0, place=3.0, side=400.0)
    r = tl.hotspots(t, mark="mark", match_by="iso", blocks="clone", sigma_um=20.0, min_neff=3.0, n_perm=99, rng=3)
    assert r.summary["n_hotspots"] >= 1
    top = r.table.iloc[0]
    assert np.hypot(top.x_um - 200.0, top.y_um - 200.0) < 80.0 and top.p_value <= 0.05


def test_permutation_studentised_map_is_deterministic_and_memory_independent(monkeypatch):
    t = _clone_field(np.random.default_rng(23), n_cl=120)
    kw = {"mark": "mark", "match_by": "iso", "blocks": "clone", "sigma_um": 20.0, "min_neff": 3.0, "n_perm": 29}
    a = tl.hotspots(t, rng=4, **kw)
    b = tl.hotspots(t, rng=np.random.default_rng(4), **kw)
    np.testing.assert_array_equal(a.null, b.null)
    np.testing.assert_array_equal(a.z_map.values, b.z_map.values)
    monkeypatch.setattr(maps_module, "_STORE_LIMIT", 0)  # second pass recomputes the shuffled maps
    c = tl.hotspots(t, rng=4, **kw)
    np.testing.assert_array_equal(a.null, c.null)
    np.testing.assert_array_equal(a.labels.values, c.labels.values)
    # the studentised observed map has the shuffled maps' scale: |z| of a few units, NaN where nothing moves
    z = a.z_map.values[np.isfinite(a.z_map.values)]
    assert len(z) and np.abs(z).max() < 10
    pw = tl.hotspots(t, rng=4, correction=None, **kw)  # pointwise p from the same draws
    assert pw.threshold["hot"] == 0.05
    kw2 = {**kw, "n_perm": 39}
    two = tl.hotspots(t, rng=4, alternative="two-sided", correction="cluster_mass", cluster_z=2.0, **kw2)
    assert two.null.shape == (39, 2)


def test_mark_that_is_a_function_of_clone_size_cannot_be_relabelled():
    rng = np.random.default_rng(24)
    t = _clone_field(rng, n_cl=80)
    t["expanded"] = t.groupby("clone").clone.transform("size") >= 4
    with pytest.warns(LowPowerWarning, match="moves the mark nowhere"):
        r = tl.hotspots(t, mark="expanded", blocks="clone", sigma_um=20.0, min_neff=3.0, n_perm=19, rng=0)
    assert r.summary["n_hotspots"] == 0 and r.summary["n_eligible"] == 0


def test_region_excess_matched_blocks_null_mean_and_calibration():
    # clones A, B (2 loci each) trade; clone C (3 loci) is alone: region 1 = A's loci + C's loci
    t = pd.DataFrame(
        {
            "clone": ["A", "A", "B", "B", "C", "C", "C"],
            "mark": [1, 1, 0, 0, 1, 0, 1],
            "region": [1, 1, 0, 0, 1, 1, 1],
        }
    )
    t["mark"] = t.mark.astype(bool)
    r = tl.region_excess(t, mark="mark", regions="region", blocks="clone", n_perm=2000, min_units=1, rng=0)
    row = r.table.iloc[0]
    assert row.marked == 4 and row.null_mean == pytest.approx(3.0, abs=0.1)  # 2 (C kept) + 2 x 1/2 (A <-> B)
    assert r.params["design"]["block_design"] == "matched"
    rng = np.random.default_rng(25)
    hits_m = hits_u = 0
    reps = 50
    for _ in range(reps):
        f = _clone_field(rng, n_cl=60)
        f["region"] = (f.x_um > 100).astype(int)
        kw = {"mark": "mark", "regions": "region", "match_by": "iso", "n_perm": 99, "rng": int(rng.integers(2**31))}
        hits_m += tl.region_excess(f, blocks="clone", **kw).table.p_value.iloc[0] <= 0.05
        hits_u += tl.region_excess(f, **kw).table.p_value.iloc[0] <= 0.05
    assert hits_m / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_u >= hits_m
