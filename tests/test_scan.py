import warnings

import numpy as np
import pandas as pd
import pytest

from spaclone._utils import LowPowerWarning, SpacloneWarning
from spaclone.tl import clonal_scan, scan_llr
from spaclone.tl import scan as scan_mod

LEGACY = {  # the archive's ClonalScan (03 B5 / B6)
    "units": "records",
    "null": "units",
    "shape": "square",
    "radii_um": (5, 15, 25, 55),
    "stopping": "besag_clifford",
    "prefilter": False,
}
TOL = 3.1  # binomial 99.9 % one-sided tolerance in standard errors


def _cell_chip(
    seed, n_cells=700, side=320.0, pitch=2.0, mol=6.0, n_exp=6, big=0.15, clumped=False, swap=0.0, plant=None
):
    """Cells of radius 4 um emitting Poisson(mol) molecules on a ``pitch`` grid (several spots per cell); clone labels
    random over cells (``swap``: share of molecules whose labels are then permuted). ``plant=(n, k)``: a clone "P" of
    n cells, k of them the cells nearest to a random point."""
    rng = np.random.default_rng(seed)
    if clumped:
        c = rng.uniform(60, side - 60, (4, 2))
        k = int(0.6 * n_cells)
        cxy = np.vstack([c[rng.integers(0, 4, k)] + rng.normal(0, 30, (k, 2)), rng.uniform(0, side, (n_cells - k, 2))])
    else:
        cxy = rng.uniform(0, side, (n_cells, 2))
    lab = np.full(n_cells, -1)
    free = np.ones(n_cells, bool)
    names, nxt = {}, 0
    if plant is not None:
        n_p, k_p = plant
        near = np.argsort(np.hypot(*(cxy - rng.uniform(40, side - 40, 2)).T))[:k_p]
        rest = rng.choice(np.setdiff1d(np.arange(n_cells), near), n_p - k_p, replace=False)
        lab[np.r_[near, rest]], free[np.r_[near, rest]] = nxt, False
        names[nxt], nxt = "P", nxt + 1
    perm = rng.permutation(np.flatnonzero(free))
    sizes = ([int(big * n_cells)] if big else []) + list(rng.integers(20, 41, n_exp))
    pos = 0
    for j, sz in enumerate(sizes):
        lab[perm[pos : pos + sz]] = nxt
        names[nxt] = "big" if (big and j == 0) else f"e{j}"
        nxt, pos = nxt + 1, pos + sz
    single = lab < 0
    lab[single] = nxt + np.arange(int(single.sum()))
    cell = np.repeat(np.arange(n_cells), rng.poisson(mol, n_cells))
    r, a = 4.0 * np.sqrt(rng.random(len(cell))), rng.uniform(0, 2 * np.pi, len(cell))
    x = np.rint((cxy[cell, 0] + r * np.cos(a)) / pitch) * pitch
    y = np.rint((cxy[cell, 1] + r * np.sin(a)) / pitch) * pitch
    clone = lab[cell]
    if swap:
        sel = np.flatnonzero(rng.random(len(clone)) < swap)
        clone[sel] = clone[sel][rng.permutation(len(sel))]
    return pd.DataFrame(
        {
            "spot_id": [f"{a:g}_{b:g}" for a, b in zip(x, y, strict=True)],
            "x_um": x,
            "y_um": y,
            "locus": "IGK",
            "clone_id": [names.get(int(c), f"s{int(c)}") for c in clone],
        }
    )


def _null_pvalues(seeds, clones=("big", "e1", "e2", "e3", "e4", "e5", "e6"), n_perm=99, **chip):
    """Plain Monte-Carlo p-values of the named null clones under the default design."""
    ps = []
    for seed in seeds:
        air = _cell_chip(seed, **chip)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)  # fixed 99 draws: only the per-clone p is used
            t = clonal_scan(
                air,
                clones=list(clones),
                min_units=1,
                n_perm=n_perm,
                n_perm_max=n_perm,
                h=10**6,
                stopping="besag_clifford",
                prefilter=False,
                radii_um=(10, 20, 40, 80),
                rng=seed,
            ).table
        ps.extend(t.p_plain.tolist())
    return np.asarray(ps)


def _pool(n=1500, n_clones=30, side=1000.0, seed=0, plant=None, chain="IGK"):
    """Uniform units on distinct 0.5-um spots, Zipf-like clone labels (random labelling holds by construction)."""
    rng = np.random.default_rng(seed)
    lin = rng.choice(int(side * 2) ** 2, n, replace=False)
    x, y = (lin % int(side * 2)) * 0.5, (lin // int(side * 2)) * 0.5
    w = 1.0 / np.arange(1, n_clones + 1)
    lab = rng.choice(n_clones, n, p=w / w.sum()).astype(str)
    if plant is not None:
        k, cx, cy, r = plant
        ang, rad = rng.uniform(0, 2 * np.pi, k), r * np.sqrt(rng.uniform(0, 1, k))
        x[:k] = np.round((cx + rad * np.cos(ang)) * 2) / 2
        y[:k] = np.round((cy + rad * np.sin(ang)) * 2) / 2
        lab[:k] = "planted"
    df = pd.DataFrame(
        {
            "spot_id": [f"{a:g}_{b:g}" for a, b in zip(x, y, strict=True)],
            "x_um": x,
            "y_um": y,
            "locus": chain,
            "clone_id": lab,
        }
    )
    return df.drop_duplicates(["spot_id", "clone_id"]).reset_index(drop=True)


def test_llr_known_values():
    # W = 3 of N_c = 10 in a window with T = 6 of N = 100: mu = 0.6
    v = scan_llr([3.0], [6.0], 10, 100)[0]
    assert v == pytest.approx(3 * np.log(3 / 0.6) + 7 * np.log(7 / 9.4))
    assert scan_llr([0.5], [6.0], 10, 100)[0] == 0.0  # no excess
    b = scan_llr([3.0], [6.0], 10, 100, model="bernoulli")[0]
    ll = lambda c, n: c * np.log(c / n) + (n - c) * np.log(1 - c / n)
    assert b == pytest.approx(ll(3, 6) + ll(7, 94) - ll(10, 100))
    with pytest.raises(ValueError):
        scan_llr([1.0], [1.0], 1, 2, model="gauss")


def test_llr_bernoulli_excess_rule():
    assert scan_llr([1.0], [6.0], 10, 100, model="bernoulli")[0] > 0  # 1/6 inside > 9/94 outside
    assert scan_llr([1.0], [20.0], 10, 100, model="bernoulli")[0] == 0.0  # 1/20 < 9/80


def test_windows_geometry():
    w = scan_mod._make_windows((5, 15, 25, 55), 10.0, "square")
    assert [len(o) for o in w.offsets] == [1, 9, 25, 121]
    c = scan_mod._make_windows((10, 20), 10.0, "circle")
    assert [len(o) for o in c.offsets] == [5, 13]
    counts = np.zeros((7, 9), dtype=np.int64)
    counts[3, 4] = 2
    counts[0, 0] = 1
    for s in range(c.n_sizes):
        got = scan_mod._window_sums(counts, c, s)
        exp = np.zeros_like(got)
        for dy, dx in c.offsets[s]:
            sh = np.zeros_like(counts)
            ys, xs = np.nonzero(counts)
            for yy, xx in zip(ys, xs, strict=True):
                cy, cx = yy - dy, xx - dx
                if 0 <= cy < 7 and 0 <= cx < 9:
                    sh[cy, cx] += counts[yy, xx]
            exp += sh
        np.testing.assert_array_equal(got, exp)


def test_known_cluster_and_best_window():
    """A clone of 6 units in one 10-um cell among 400 spread units: the best window is that cell's window."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 2000, 400).round() + 0.5
    y = rng.uniform(0, 2000, 400).round() + 0.5
    lab = np.array(["bg"] * 400, dtype=object)
    x[:6] = 1003.0 + np.arange(6)
    y[:6] = 1005.0
    lab[:6] = "A"
    lab[6:40] = "B"
    air = pd.DataFrame({"spot_id": np.arange(400).astype(str), "x_um": x, "y_um": y, "locus": "IGH", "clone_id": lab})
    r = clonal_scan(air, min_units=6, n_perm=99, rng=0, **LEGACY)
    a = r.table.set_index("clone_id").loc["A"]
    assert a.best_observed == 6
    assert a.p_value <= 0.02
    # the best window (largest among equally good ones) contains the cluster
    assert abs(a.best_x_um - 1005.5) <= a.best_radius_um + 10 and abs(a.best_y_um - 1005.0) <= a.best_radius_um + 10
    # the fast null path agrees with a full evaluation of every window
    idx = np.arange(6)
    win = scan_mod._make_windows((5, 15, 25, 55), 10.0, "square")
    pool, _, _ = scan_mod._build_pool(x.astype(float), y.astype(float), None, win, "poisson")
    full = scan_mod._observed(idx, pool)["statistic"]
    b, m = np.zeros(6, np.int64), np.array([6])
    fast = scan_mod._max_sparse(idx, b, m, pool)[0][0]
    dense = scan_mod._max_dense(idx, b, m, pool)[0]
    assert full == pytest.approx(a.statistic) and dense == pytest.approx(full) and fast == pytest.approx(full)


@pytest.mark.parametrize("shape", ["square", "circle"])
@pytest.mark.parametrize("model", ["poisson", "bernoulli"])
def test_fast_null_maximum_equals_full_evaluation(shape, model):
    """The shortcut used for null draws (t_min bound + clustered scatter) equals the full window maximum."""
    df = _pool(n=900, seed=3)
    x, y = df.x_um.to_numpy(), df.y_um.to_numpy()
    win = scan_mod._make_windows((5, 15, 25, 55) if shape == "square" else (10, 20, 30, 60), 10.0, shape)
    pool, _, _ = scan_mod._build_pool(x, y, None, win, model)
    rng = np.random.default_rng(0)
    for n_c in (2, 5, 30, 200):
        idx = np.stack([rng.choice(pool.n, n_c, replace=False) for _ in range(6)])
        for merge in (None, 15.0):
            units, b, m = scan_mod._merge(pool, idx, merge)
            fast = scan_mod._max_sparse(units, b, m, pool)[0]
            dense = scan_mod._max_dense(units, b, m, pool)
            full = [scan_mod._observed(units[b == i], pool)["statistic"] for i in range(len(idx))]
            np.testing.assert_allclose(fast, full, rtol=1e-12)
            np.testing.assert_allclose(dense, full, rtol=1e-12)


def test_merge_matches_merge_loci():
    from spaclone.pp import merge_loci

    df = _pool(n=2000, n_clones=8, seed=14)
    u = df.sort_values(["x_um", "y_um", "spot_id", "clone_id"], ignore_index=True)
    win = scan_mod._make_windows((10,), 10.0, "circle")
    pool, _, _ = scan_mod._build_pool(u.x_um.to_numpy(), u.y_um.to_numpy(), None, win, "poisson")
    loci, _ = merge_loci(u, radius_um=15.0, by=())
    for c, g in u.groupby("clone_id"):
        _, _, m = scan_mod._merge(pool, g.index.to_numpy()[None, :], 15.0)
        assert m[0] == (loci.clone_id == c).sum()


def test_power_planted_clone_found_and_null_clones_not():
    df = _pool(n=1500, n_clones=15, seed=4, plant=(14, 500.0, 500.0, 20.0))
    r = clonal_scan(df, min_units=10, n_perm=99, n_perm_max=999, rng=1, **LEGACY)
    t = r.table.set_index("clone_id")
    assert t.loc["planted", "p_value"] <= 0.01
    assert t.loc["planted", "significant"]
    assert (t.drop(index="planted").q_value > 0.05).all()


def test_bh_stopping_reaches_significance_where_fixed_cap_cannot():
    """One genuine clone among many: a fixed 199-draw cap cannot pass BH alone, the BH-driven cap can."""
    df = _pool(n=3000, n_clones=60, seed=5, plant=(10, 300.0, 300.0, 15.0))
    common = {
        "units": "records",
        "null": "units",
        "shape": "square",
        "radii_um": (5, 15, 25),
        "min_units": 10,
        "n_perm": 99,
        "rng": 2,
    }
    with pytest.warns(LowPowerWarning, match="only together"):
        fixed = clonal_scan(df, stopping="besag_clifford", n_perm_max=199, prefilter=False, **common)
    t_fix = fixed.table.set_index("clone_id")
    assert t_fix.loc["planted", "p_value"] == pytest.approx(1 / 200)  # at the floor, yet not called
    assert not t_fix.loc["planted", "significant"]
    bh = clonal_scan(df, stopping="bh", **common)
    t = bh.table.set_index("clone_id")
    m = int(t.testable.sum())
    assert t.loc["planted", "significant"] and t.loc["planted", "stop"] == "rejected"
    assert t.loc["planted", "n_perm"] >= 10 * m / 0.05 / 2  # needed about h m / alpha draws
    others = t.drop(index="planted")
    assert others.significant.sum() <= 1  # FDR control allows an occasional extreme null clone
    assert others.n_drawn.median() < 400  # clones with an open decision only were extended


def test_bh_pvalues_are_conservative_versions_of_besag_clifford():
    """With the same seed, BH mode reports p >= h / L for clones stopped before the h-th exceedance."""
    df = _pool(n=3000, n_clones=40, seed=6)
    r = clonal_scan(df, min_units=20, n_perm=99, rng=3, units="records", shape="square", radii_um=(5, 15, 25))
    t = r.table
    run = t[t.stop.isin(["decided", "cap", "rejected"])]
    assert np.allclose(run.p_value, 10 / (run.n_drawn + 1))
    ex = t[t.stop == "exceedances"]
    assert np.allclose(ex.p_value, 10 / ex.n_perm)


def test_calibration_under_random_labelling():
    """Exact random labelling by construction: P(p <= 0.05) <= 0.05 + binomial tolerance."""
    ps = []
    for seed in range(6):
        df = _pool(n=1200, n_clones=12, seed=100 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)  # the 99-draw floor cannot pass BH; only p is used
            r = clonal_scan(
                df,
                min_units=15,
                n_perm=99,
                n_perm_max=99,
                rng=seed,
                units="records",
                null="units",
                shape="square",
                radii_um=(5, 15, 25),
                stopping="besag_clifford",
                prefilter=False,
            )
        ps.extend(r.table.p_value.tolist())
    ps = np.asarray(ps)
    assert len(ps) >= 40
    assert (ps <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(ps))


@pytest.mark.slow
def test_calibration_large_bh_and_loci():
    """Units exchangeable by construction: every design (default, legacy records, 1.0.dev0 loci) is calibrated."""
    for kw in ({}, {"units": "records", "null": "units"}, {"units": "loci", "null": "units"}):
        ps = []
        for seed in range(40):
            df = _pool(n=2000, n_clones=15, seed=300 + seed)
            r = clonal_scan(df, min_units=20, n_perm=199, rng=seed, **kw)
            ps.extend(r.table.p_value.tolist())
        ps = np.asarray(ps)
        assert (ps <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(ps))


def test_one_cell_is_not_a_cluster():
    """One cell spilling 8 molecules over adjacent spots: a cluster for the archive's design, not for the default
    (the cell occupies one or two grid cells and is one block) nor for loci (one locus). Counted in records, the
    block null is still valid but the one cell weighs 8 units: the clone looks concentrated because it owns the only
    big cell of the pool (p about the chance of drawing that block, ~0.03)."""
    kw = {
        "clones": ["spill"],
        "min_units": 5,
        "n_perm": 99,
        "n_perm_max": 99,
        "shape": "square",
        "radii_um": (5, 15, 25, 55),
        "stopping": "besag_clifford",
        "prefilter": False,
    }
    p_rec, p_loc, p_def, p_recb = [], [], [], []
    for seed in range(4):
        df = _pool(n=1200, n_clones=10, seed=20 + seed)
        rng = np.random.default_rng(seed)
        cell = pd.DataFrame(
            {
                "spot_id": [f"c{i}" for i in range(8)],
                "x_um": 700.0 + np.arange(8) * 0.5,
                "y_um": np.full(8, 700.0),
                "locus": "IGK",
                "clone_id": "spill",
            }
        )
        rest = pd.DataFrame(
            {
                "spot_id": [f"r{i}" for i in range(20)],
                "x_um": rng.uniform(0, 1000, 20).round(),
                "y_um": rng.uniform(0, 1000, 20).round(),
                "locus": "IGK",
                "clone_id": "spill",
            }
        )
        air = pd.concat([df, cell, rest], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)
            rec = clonal_scan(air, units="records", null="units", rng=seed, **kw).table.iloc[0]
            loc = clonal_scan(air, units="loci", null="units", rng=seed, **kw).table.iloc[0]
            dft = clonal_scan(air, rng=seed, **{**kw, "min_units": 2}).table.iloc[0]
            recb = clonal_scan(air, units="records", rng=seed, **{**kw, "min_units": 2}).table.iloc[0]
        assert loc.n_records == rec.n_units and loc.n_units <= rec.n_units - 7  # the cell counts once
        assert recb.n_blocks <= rec.n_units - 7 and dft.n_records < rec.n_units  # one block; few grid cells
        p_rec.append(rec.p_value)
        p_loc.append(loc.p_value)
        p_def.append(dft.p_value)
        p_recb.append(recb.p_value)
    assert max(p_rec) <= 0.02
    assert np.mean(p_loc) > 0.2 and np.mean(p_def) > 0.2
    assert 0.01 < np.mean(p_recb) < 0.2


def test_distinct_spot_draws():
    rng = np.random.default_rng(0)
    spot = np.repeat(np.arange(50), 4)
    x = np.repeat(np.arange(50) * 20.0, 4)
    win = scan_mod._make_windows((5,), 10.0, "square")
    pool, _, _ = scan_mod._build_pool(x, np.zeros(200), spot, win, "poisson")
    for _ in range(20):
        idx = scan_mod._draw_one(pool, 30, rng)
        assert len(np.unique(spot[idx])) == 30
    batch = scan_mod._draw_batch(pool, 10, 50, rng)
    assert all(len(np.unique(spot[r])) == 10 for r in batch)


def test_large_clone_in_a_dense_pool_is_not_called():
    """A clone holding a fifth of a clumped pool of independent units (the broken-assay null): its loci merge more where
    spots are dense, so relabelling ready-made loci would call it; the default and the loci designs must not."""
    rng = np.random.default_rng(15)
    n = 2500
    centres = rng.uniform(100, 400, (6, 2))
    pts = np.vstack(
        [c + rng.normal(0, 15, (n // 6, 2)) for c in centres] + [rng.uniform(0, 500, (n - 6 * (n // 6), 2))]
    )
    pts = np.round(pts * 2) / 2
    lab = np.where(rng.random(len(pts)) < 0.2, "big", rng.choice([f"c{i}" for i in range(20)], len(pts)))
    air = pd.DataFrame(
        {"spot_id": [f"{a}_{b}" for a, b in pts], "x_um": pts[:, 0], "y_um": pts[:, 1], "locus": "IGK", "clone_id": lab}
    ).drop_duplicates(["spot_id", "clone_id"])
    for kw in ({"units": "loci"}, {}):
        tabs = [
            clonal_scan(air, clones=["big"], min_units=3, n_perm=99, rng=s, radii_um=(10, 30, 60), **kw).table
            for s in range(3)
        ]
        if kw:
            assert all(len(t) == 1 and t.n_units.iloc[0] < t.n_records.iloc[0] / 4 for t in tabs)  # it merges a lot
        assert min(t.p_value.iloc[0] for t in tabs) > 0.05
    empty = clonal_scan(air, clones=["big"], min_units=400, n_perm=9, rng=0).table
    assert len(empty) == 0 and "p_value" in empty.columns


def test_prefilter_drops_untestable_singletons():
    df = _pool(n=800, n_clones=10, seed=9)
    extra = pd.DataFrame(
        {
            "spot_id": [f"z{i}" for i in range(40)],
            "x_um": np.arange(40) * 23.0 + 0.5,
            "y_um": np.full(40, 3.0),
            "locus": "IGK",
            "clone_id": [f"single{i}" for i in range(40)],
        }
    )
    air = pd.concat([df, extra], ignore_index=True)
    r = clonal_scan(air, min_units=1, n_perm=19, rng=0, units="records", null="units", shape="square", radii_um=(5, 15))
    t = r.table.set_index("clone_id")
    assert not t.loc[[f"single{i}" for i in range(40)], "testable"].any()
    assert t.loc[t.n_units >= 2, "testable"].all()
    assert t.loc["single0", "stop"] == "untestable" and np.isnan(t.loc["single0", "p_value"])


@pytest.mark.parametrize("units", ["bins", "records"])
def test_deterministic_and_order_and_translation_invariant(units):
    df = _cell_chip(10, n_cells=400, side=250.0)
    kw = {"min_units": 15, "n_perm": 49, "units": units, "shape": "circle", "radii_um": (10, 30)}
    a = clonal_scan(df, rng=5, **kw).table
    assert len(a) >= 3
    b = clonal_scan(df.sample(frac=1.0, random_state=3), rng=5, **kw).table
    pd.testing.assert_frame_equal(a, b)
    c = clonal_scan(df, rng=6, **kw).table
    assert not np.allclose(a.null_mean, c.null_mean)
    shifted = df.assign(x_um=df.x_um + 1000.0, y_um=df.y_um - 250.0)
    s = clonal_scan(shifted, rng=5, **kw).table
    np.testing.assert_allclose(s.statistic, a.statistic)
    np.testing.assert_allclose(s.p_value, a.p_value)
    np.testing.assert_allclose(s.best_x_um - 1000.0, a.best_x_um)


@pytest.mark.slow  # worker start-up (spawn / forkserver) dominates
def test_n_jobs_gives_same_result():
    df = _pool(n=1500, n_clones=10, seed=11)
    for extra in ({"units": "loci", "null": "units"}, {}):
        kw = {"min_units": 20, "n_perm": 49, "radii_um": (10, 30), "rng": 4, **extra}
        a = clonal_scan(df, **kw).table
        b = clonal_scan(df, n_jobs=2, **kw).table
        pd.testing.assert_frame_equal(a, b)


def test_strata_and_inputs():
    a = _pool(n=600, n_clones=6, seed=12, chain="IGK")
    b = _pool(n=600, n_clones=6, seed=13, chain="IGH")
    air = pd.concat([a, b], ignore_index=True)
    r = clonal_scan(air, min_units=20, n_perm=19, rng=0, units="records", radii_um=(10,))
    assert set(r.table.locus) == {"IGK", "IGH"}
    only = clonal_scan(air, locus="IGH", min_units=20, n_perm=19, rng=0, units="records", radii_um=(10,))
    assert set(only.table.locus) == {"IGH"}
    lib = air.assign(library_id=np.where(np.arange(len(air)) % 2 == 0, "L1", "L2"))
    rl = clonal_scan(lib, min_units=10, n_perm=19, rng=0, units="records", radii_um=(10,))
    assert set(rl.table.library_id) == {"L1", "L2"}
    with pytest.raises(ValueError, match="units"):
        clonal_scan(air, units="cells")
    with pytest.raises(ValueError, match="null"):
        clonal_scan(air, null="spots")
    with pytest.raises(ValueError, match="block_radius_um"):
        clonal_scan(air, block_radius_um=-1.0)
    with pytest.raises(ValueError, match="lacks column"):
        clonal_scan(air.drop(columns="clone_id"))
    with pytest.raises(KeyError):
        clonal_scan(air, by="sample")
    assert "TestResult(clonal_scan" in repr(r)


def test_plots_return_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    from spaclone import pl

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (40, 2))
    e = pl.mst_edges(xy)
    assert len(e) == 39
    length = np.hypot(*(xy[e[:, 0]] - xy[e[:, 1]]).T).sum()
    assert length == pytest.approx(minimum_spanning_tree(squareform(pdist(xy))).sum())
    assert len(pl.mst_edges(np.vstack([xy[:3], xy[:1]]))) == 3  # a duplicated point joins at length 0
    df = _pool(n=800, n_clones=8, seed=16, plant=(12, 300.0, 300.0, 20.0))
    for style in ("light", "dark"):
        ax = pl.clone_map(df, ["planted", "0"], style=style, max_link_um=200, scalebar_um=100)
        assert ax.collections and ax.get_aspect() == 1.0
        plt.close(ax.figure)
    r = clonal_scan(df, min_units=10, n_perm=19, rng=0, units="records", radii_um=(10, 30))
    for kind in ("size", "qq"):
        ax = pl.clonal_scan(r, kind=kind, style="dark")
        assert ax.get_ylabel()
        plt.close(ax.figure)
    with pytest.raises(ValueError):
        pl.clonal_scan(r, kind="map")


# ------------------------------------------------------------------------------------- cell-scale block null (F4a)


def test_leader_blocks_known_answer():
    # clone 0 on a line: the seed at 0 takes 0, 5, 10 (<= 15); 20 seeds the next block (40 is 20 away); 40 is alone
    xy = np.array([[0.0, 0], [5, 0], [10, 0], [20, 0], [40, 0], [100, 100], [103, 100]])
    codes = np.array([0, 0, 0, 0, 0, 1, 1])
    ids = scan_mod._leader_blocks(xy, codes, 15.0)
    assert len(set(ids[:3])) == 1 and len(set(ids)) == 4
    assert ids[3] != ids[0] and ids[4] not in (ids[0], ids[3]) and ids[5] == ids[6]  # clone 1: one block
    b = scan_mod._Blocks.from_ids(ids, 15.0)
    assert b.n == 4 and sorted(b.size.tolist()) == [1, 1, 2, 3]
    assert all(len(set(codes[b.members[b.ptr[k] : b.ptr[k + 1]]])) == 1 for k in range(b.n))  # one clone per block
    # blocks never reach beyond the radius around their seed, however dense the clone
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (400, 2))
    ids = scan_mod._leader_blocks(pts, np.zeros(400, np.int64), 10.0)
    for k in np.unique(ids):
        m = pts[ids == k]
        assert np.hypot(*(m - m[0]).T).max() <= 10.0 + 1e-9


def test_block_draws_take_whole_blocks_on_distinct_spots():
    rng = np.random.default_rng(1)
    # 60 blocks of 1-6 units (one per clone), some spots shared between clones
    sizes = rng.integers(1, 7, 60)
    cx, cy = rng.uniform(0, 2000, 60), rng.uniform(0, 2000, 60)
    x = np.concatenate([np.full(s, a) + np.arange(s) * 0.5 for s, a in zip(sizes, cx, strict=True)])
    y = np.concatenate([np.full(s, b) for s, b in zip(sizes, cy, strict=True)])
    codes = np.repeat(np.arange(60), sizes)
    x[-1], y[-1] = x[0], y[0]  # one shared spot
    order = np.lexsort((codes, y, x))
    x, y, codes = x[order], y[order], codes[order]
    spot = pd.factorize(pd.Series([f"{a}_{b}" for a, b in zip(x, y, strict=True)]))[0].astype(np.int64)
    win = scan_mod._make_windows((10,), 10.0, "circle")
    pool, _, _ = scan_mod._build_pool(x, y, spot, win, "poisson")
    blocks = scan_mod._Blocks.from_ids(scan_mod._leader_blocks(pool.xy, codes, 15.0), 15.0)
    of_unit = np.empty(pool.n, np.int64)
    for k in range(blocks.n):
        of_unit[blocks.members[blocks.ptr[k] : blocks.ptr[k + 1]]] = k
    draws = scan_mod._draw_blocks(pool, blocks, 25, 40, np.random.default_rng(2))
    assert draws.shape == (40, 25)
    for row in draws:
        assert len(np.unique(spot[row])) == 25  # never two units on one spot
        used = np.unique(of_unit[row])
        partial = [k for k in used if np.isin(blocks.members[blocks.ptr[k] : blocks.ptr[k + 1]], row).mean() < 1]
        assert len(partial) <= 2  # only the last block is cut (plus a block that lost a unit to a shared spot)
    again = scan_mod._draw_blocks(pool, blocks, 25, 40, np.random.default_rng(2))
    np.testing.assert_array_equal(draws, again)
    big = scan_mod._draw_blocks(pool, blocks, int(sizes.sum()) - 5, 3, np.random.default_rng(3))  # permutation path
    assert all(len(np.unique(spot[r])) == big.shape[1] for r in big)


def test_bins_count_each_grid_cell_once():
    air = _cell_chip(4, n_cells=300, side=200.0, pitch=0.5)
    t = clonal_scan(air, clones=["big", "e0"], min_units=1, n_perm=9, rng=0).table.set_index("clone_id")
    x0, y0 = air.x_um.min(), air.y_um.min()
    cells = air.assign(b=np.floor((air.x_um - x0) / 10).astype(int) * 100000 + np.floor((air.y_um - y0) / 10))
    expect = cells[cells.clone_id.isin(["big", "e0"])].groupby("clone_id").b.nunique()
    assert t.n_records.to_dict() == expect.to_dict()
    rec = clonal_scan(air, clones=["big"], units="records", min_units=1, n_perm=9, rng=0).table
    assert rec.n_records.iloc[0] == air[air.clone_id == "big"].spot_id.nunique() > t.loc["big", "n_records"]


def test_default_is_calibrated_when_cells_span_several_spots():
    """Clone labels random over cells covering ~5 spots each: the archive's unit relabelling calls most null clones
    concentrated (and warns); the default block null does not."""
    ps = _null_pvalues(range(8), n_cells=500, side=270.0)
    assert len(ps) == 56
    assert (ps <= 0.05).mean() <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / len(ps))
    air = _cell_chip(0, n_cells=500, side=270.0)
    with pytest.warns(SpacloneWarning, match="clusters of a median"), warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        leg = clonal_scan(
            air,
            clones=["e1", "e2", "e3"],
            min_units=1,
            n_perm=99,
            n_perm_max=99,
            rng=0,
            stopping="besag_clifford",
            units="records",
            null="units",
            radii_um=(10, 20, 40, 80),
        ).table
    assert (leg.p_plain <= 0.05).all()


def test_power_for_a_packed_clone():
    hits = []
    for seed in range(4):
        air = _cell_chip(100 + seed, plant=(30, 15))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)
            t = clonal_scan(
                air,
                clones=["P"],
                min_units=1,
                n_perm=99,
                n_perm_max=99,
                stopping="besag_clifford",
                radii_um=(10, 20, 40, 80),
                rng=seed,
            ).table
        hits.append(t.p_value.iloc[0] <= 0.05)
    assert sum(hits) >= 3


@pytest.mark.slow
@pytest.mark.parametrize(
    "chip",
    [
        {"clumped": True},  # labels random over cells, clumped pool, large clone
        {"clumped": True, "swap": 1.0},  # every molecule misplaced (the broken-assay null)
        {"clumped": True, "swap": 0.5},  # half of the molecules misplaced
        {"pitch": 10.0},  # about two spots per cell
    ],
)
def test_default_calibration_slow(chip):
    ps = _null_pvalues(range(200, 260), **chip)
    assert (ps <= 0.05).mean() <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / len(ps))


@pytest.mark.slow
def test_power_for_a_germinal_centre_clone_and_bh_family():
    """Half of 150 cells packed in a clump: found by the default; all-null families rarely give a BH discovery."""
    found, families = 0, 0
    for seed in range(10):
        air = _cell_chip(300 + seed, n_cells=900, clumped=True, plant=(150, 75))
        t = clonal_scan(air, min_units=10, n_perm=99, radii_um=(10, 20, 40, 80), rng=seed).table.set_index("clone_id")
        found += bool(t.loc["P", "significant"])
        families += bool(t.drop(index="P").significant.any())
    assert found >= 9 and families <= 2
