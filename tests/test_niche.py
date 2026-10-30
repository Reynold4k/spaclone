import numpy as np
import pandas as pd
import pytest

from spaclone import tl

TOL = 3.1  # binomial 99.9% one-sided tolerance in standard errors


def _profile_table(n=400, n_group=40, shift=None, seed=0):
    """Units with a signed distance; the group is random, or drawn near ``shift`` µm when given."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(-150, 250, n)
    if shift is None:
        g = np.zeros(n, bool)
        g[rng.choice(n, n_group, replace=False)] = True
    else:
        w = np.exp(-0.5 * ((d - shift) / 30.0) ** 2)
        g = np.zeros(n, bool)
        g[rng.choice(n, n_group, replace=False, p=w / w.sum())] = True
    return pd.DataFrame({"d": d, "group": g, "comp": np.where(d < 0, "inside", "outside")})


def test_niche_profile_known_values():
    t = pd.DataFrame(
        {
            "d": [-30.0, -10, 5, 15, 25, 60, 70, 80, 90, 100],
            "g": [True, True, True, False, False, False, False, False, False, False],
        }
    )
    r = tl.niche_profile(t, group="g", distance_um="d", bands_um=[0, 50], n_perm=99, rng=0)
    tab = r.table
    assert list(tab.label) == ["<0", "[0, 50)", ">=50"]
    np.testing.assert_array_equal(tab.group_units, [2, 1, 0])
    np.testing.assert_array_equal(tab.baseline_units, [2, 3, 5])
    np.testing.assert_allclose(tab.statistic, [2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(tab.expected, [0.2, 0.3, 0.5])
    np.testing.assert_allclose(tab.ratio, [10 / 3, 10 / 9, 0])
    np.testing.assert_allclose(tab.mid_um, [-25, 25, 75])
    assert np.isneginf(tab.lo_um[0]) and np.isposinf(tab.hi_um[2])
    assert r.null.shape == (99, 3) and np.allclose(r.null.sum(axis=1), 1.0)
    assert (tab.p_value > 0).all() and (tab.p_maxT > 0).all()
    assert r.summary["n_group"] == 3 and r.summary["n_baseline"] == 10 and r.summary["bands"] == 3
    g = tl.niche_profile(t, group="g", distance_um="d", bands_um=[0, 50], n_perm=99, alternative="greater", rng=0)
    assert (g.table.p_maxT >= g.table.p_value - 1e-12).all()  # one-sided: family-wise p never below pointwise p
    assert np.isneginf(g.table.env_lo).all() and (g.table.env_hi >= g.table.null_mean).all()


def test_compartment_order_and_missing_values():
    t = pd.DataFrame({"c": ["b", "a", "c", "a", None, "b"], "g": [1, 0, 1, 0, 1, 0]})
    r = tl.niche_profile(t, group="g", compartment="c", n_perm=19, rng=0)
    assert list(r.table.label) == ["a", "b", "c"] and r.summary["n_missing"] == 1
    np.testing.assert_allclose(r.table.statistic, [0, 0.5, 0.5])
    t["c"] = pd.Categorical(t.c, categories=["c", "b", "a"])
    assert list(tl.niche_profile(t, group="g", compartment="c", n_perm=19, rng=0).table.label) == ["c", "b", "a"]


def test_null_mean_is_the_pool_share():
    t = _profile_table(n=600, n_group=60, seed=1)
    base = ~t.group.to_numpy()
    union = tl.niche_profile(t, group="group", distance_um="d", bands_um=[0, 100], baseline=base, n_perm=2000, rng=1)
    pool_share = np.bincount(np.digitize(t.d, [0, 100]), minlength=3) / len(t)
    np.testing.assert_allclose(union.table.null_mean, pool_share, atol=4 * union.table.null_sd.max() / np.sqrt(2000))
    only = tl.niche_profile(
        t, group="group", distance_um="d", bands_um=[0, 100], baseline=base, pool="baseline", n_perm=2000, rng=1
    )
    np.testing.assert_allclose(only.table.null_mean, only.table.expected, atol=4 * only.table.null_sd.max() / 45)


@pytest.mark.parametrize("reps", [150, pytest.param(1000, marks=pytest.mark.slow)])
def test_global_test_is_calibrated_under_random_labelling(reps):
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(reps):
        t = _profile_table(n=300, n_group=30, seed=int(rng.integers(2**31)))
        r = tl.niche_profile(
            t,
            group="group",
            distance_um="d",
            bands_um=[-100, -50, 0, 50, 100, 150],
            n_perm=99,
            rng=int(rng.integers(2**31)),
        )
        hits += r.summary["global_p"] <= 0.05
    assert hits / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)


def test_power_on_a_group_at_the_border():
    t = _profile_table(n=800, n_group=80, shift=0.0, seed=3)
    r = tl.niche_profile(t, group="group", distance_um="d", bands_um=[-100, -50, 0, 50, 100, 150], n_perm=999, rng=4)
    assert r.summary["global_p"] <= 0.01
    assert r.summary["most_extreme"] in ("[-50, 0)", "[0, 50)")
    hi = r.table.set_index("label").loc["[0, 50)"]
    assert hi.ratio > 1.5 and hi.statistic > hi.env_hi


def test_determinism():
    t = _profile_table(seed=5)
    a = tl.niche_profile(t, group="group", distance_um="d", bands_um=[0, 100], n_perm=49, rng=9)
    b = tl.niche_profile(t, group="group", distance_um="d", bands_um=[0, 100], n_perm=49, rng=np.random.default_rng(9))
    c = tl.niche_profile(t, group="group", distance_um="d", bands_um=[0, 100], n_perm=49, rng=10)
    np.testing.assert_array_equal(a.null, b.null)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.null, c.null)


def _structures(rng, n_struct=40, per=8):
    """Units grouped in structures that sit at one distance each; the label belongs to the structure."""
    d0 = rng.uniform(-150, 250, n_struct)
    sid = np.repeat(np.arange(n_struct), per)
    g = (rng.random(n_struct) < 0.2)[sid]
    return pd.DataFrame({"d": d0[sid] + rng.normal(0, 5, len(sid)), "sid": sid, "group": g})


@pytest.mark.parametrize("reps", [80, pytest.param(600, marks=pytest.mark.slow)])
def test_structure_level_resampling_avoids_pseudoreplication(reps):
    rng = np.random.default_rng(6)
    hits_blocks = hits_units = 0
    bands = [-100, -50, 0, 50, 100, 150]
    for _ in range(reps):
        t = _structures(rng)
        if not t.group.any() or t.group.all():
            continue
        seed = int(rng.integers(2**31))
        hits_blocks += (
            tl.niche_profile(
                t, group="group", distance_um="d", bands_um=bands, blocks="sid", n_perm=99, rng=seed
            ).summary["global_p"]
            <= 0.05
        )
        hits_units += (
            tl.niche_profile(t, group="group", distance_um="d", bands_um=bands, n_perm=99, rng=seed).summary["global_p"]
            <= 0.05
        )
    assert hits_blocks / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_units > 2 * hits_blocks  # shuffling units of whole-structure labels is anti-conservative


def test_blocks_keep_structures_together_and_strata():
    rng = np.random.default_rng(7)
    t = _structures(rng, n_struct=30, per=5)
    comp = np.where(t.sid % 2 == 0, "even", "odd")  # every structure lies in one compartment
    r = tl.niche_profile(t, group="group", compartment=comp, blocks="sid", n_perm=99, rng=1)
    # each null draw takes whole structures of 5 units: drawn counts per compartment are multiples of 5
    k = int(t.group.sum())
    assert np.allclose(np.round(r.null * k) % 5, 0) and r.summary["n_group"] == k
    with pytest.raises(ValueError, match="varies inside a block"):
        tl.niche_profile(
            t.assign(group=np.arange(len(t)) == 0), group="group", compartment="sid", blocks="sid", n_perm=19, rng=0
        )
    # strata: the label is redistributed only inside its stratum
    s = pd.DataFrame({"d": np.r_[np.full(50, -10.0), np.full(50, 10.0)], "st": np.repeat(["a", "b"], 50)})
    s["group"] = np.r_[np.ones(5, bool), np.zeros(95, bool)]
    rs = tl.niche_profile(s, group="group", distance_um="d", bands_um=[0], within="st", n_perm=49, rng=2)
    np.testing.assert_allclose(rs.table.null_mean, [1.0, 0.0])


def test_cluster_bootstrap_interval():
    rng = np.random.default_rng(8)
    t = _structures(rng, n_struct=60, per=6)
    r = tl.niche_profile(
        t, group="group", distance_um="d", bands_um=[0, 100], structure_key="sid", n_boot=400, n_perm=99, rng=3
    )
    ok = np.isfinite(r.table.ratio) & (r.table.group_units > 0)
    assert ((r.table.ci_lo[ok] <= r.table.ratio[ok]) & (r.table.ratio[ok] <= r.table.ci_hi[ok])).all()


def test_errors():
    t = _profile_table(seed=9)
    with pytest.raises(ValueError, match="exactly one"):
        tl.niche_profile(t, group="group", distance_um="d", bands_um=[0], compartment="comp")
    with pytest.raises(ValueError, match="bands_um"):
        tl.niche_profile(t, group="group", distance_um="d")
    with pytest.raises(ValueError, match="increasing"):
        tl.niche_profile(t, group="group", distance_um="d", bands_um=[10, 0])
    with pytest.raises(ValueError, match="structure_key"):
        tl.niche_profile(t, group="group", compartment="comp", n_boot=10)
    with pytest.raises(ValueError, match="need group"):
        tl.niche_profile(t.assign(group=False), group="group", compartment="comp")


def _clone_table(rng, on_border=False, n_bg=1500, sections=1):
    """Singleton loci plus 6 group clones and 10 other clones of 20 loci each (sd 15 um), centres uniform over the
    field (a true null for the clone-level label) or, for the group, on the border at radius 100 um."""
    x, y = rng.uniform(0, 500, n_bg), rng.uniform(0, 500, n_bg)
    clone = np.array([f"s{i}" for i in range(n_bg)], dtype=object)
    cx, cy = rng.uniform(0, 500, 16), rng.uniform(0, 500, 16)
    if on_border:
        a = rng.uniform(0, 2 * np.pi, 6)
        cx[:6], cy[:6] = 250 + 100 * np.cos(a), 250 + 100 * np.sin(a)
    xs = np.concatenate([np.clip(c + rng.normal(0, 15, 20), 0, 500) for c in cx])
    ys = np.concatenate([np.clip(c + rng.normal(0, 15, 20), 0, 500) for c in cy])
    t = pd.DataFrame(
        {
            "x_um": np.r_[x, xs],
            "y_um": np.r_[y, ys],
            "clone_id": np.r_[clone, np.repeat([f"E{k}" for k in range(16)], 20)],
            "group": np.r_[np.zeros(n_bg, bool), np.repeat(np.arange(16) < 6, 20)],
            "section": np.r_[rng.integers(0, sections, n_bg), np.repeat(rng.integers(0, sections, 16), 20)],
        }
    )
    t["d"] = np.hypot(t.x_um - 250, t.y_um - 250) - 100
    return t


def test_auto_blocks_follow_the_label_level():
    rng = np.random.default_rng(10)
    t = _clone_table(rng)
    r = tl.niche_profile(t, group="group", distance_um="d", bands_um=[-50, 0, 50, 100], n_perm=49, rng=0)
    assert r.summary["blocks"] == "clone_id (auto)" and r.summary["n_group_blocks"] == 6
    # auto = explicit clone blocks
    same = tl.niche_profile(
        t, group="group", distance_um="d", bands_um=[-50, 0, 50, 100], n_perm=49, rng=0, blocks="clone_id"
    )
    np.testing.assert_array_equal(r.null, same.null)
    unit = tl.niche_profile(
        t.assign(group=rng.random(len(t)) < 0.1),
        group="group",
        distance_um="d",
        bands_um=[-50, 0, 50, 100],
        n_perm=19,
        rng=0,
    )
    assert unit.summary["blocks"].startswith("units (the group label varies")
    none = tl.niche_profile(t, group="group", distance_um="d", bands_um=[0], n_perm=19, rng=0, blocks=None)
    assert none.summary["blocks"] == "units" and none.summary["n_group_blocks"] == none.summary["n_group"]
    other = tl.niche_profile(
        t.rename(columns={"clone_id": "clone"}), group="group", distance_um="d", bands_um=[0], n_perm=19, rng=0
    )
    assert other.summary["blocks"] == "units (no clone column)"
    two = tl.niche_profile(
        _clone_table(rng, sections=2),
        group="group",
        distance_um="d",
        bands_um=[0, 50],
        within="section",
        n_perm=19,
        rng=0,
    )
    assert two.summary["blocks"] == "clone_id (auto)"


@pytest.mark.parametrize("reps", [60, pytest.param(300, marks=pytest.mark.slow)])
def test_clone_level_label_calibrated_by_default(reps):
    """Loci of expanded clones placed independently of the border: unit shuffles call most tables, clone blocks not."""
    rng = np.random.default_rng(11)
    hits_auto = hits_units = 0
    bands = [-60, -30, 0, 30, 60, 120]
    for _ in range(reps):
        t = _clone_table(rng)
        seed = int(rng.integers(2**31))
        hits_auto += (
            tl.niche_profile(t, group="group", distance_um="d", bands_um=bands, n_perm=99, rng=seed).summary["global_p"]
            <= 0.05
        )
        hits_units += (
            tl.niche_profile(
                t, group="group", distance_um="d", bands_um=bands, n_perm=99, rng=seed, blocks=None
            ).summary["global_p"]
            <= 0.05
        )
    assert hits_auto / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)
    assert hits_units / reps > 0.5


def test_clone_level_label_power_on_the_border():
    rng = np.random.default_rng(12)
    hits = 0
    for _ in range(10):
        t = _clone_table(rng, on_border=True)
        r = tl.niche_profile(
            t,
            group="group",
            distance_um="d",
            bands_um=[-60, -30, 0, 30, 60, 120],
            n_perm=199,
            rng=int(rng.integers(2**31)),
        )
        hits += r.summary["global_p"] <= 0.05
    assert hits >= 6
