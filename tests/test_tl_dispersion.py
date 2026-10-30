import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from scipy.stats import norm

from spaclone import nulls, tl
from spaclone._utils import LowPowerWarning, bh
from spaclone.tl.dispersion import _batch_measures, _besag_clifford, _Frame, _mst_length, _ssi


def _grid_air(n=1500, n_spots=1200, n_clones=150, seed=3, planted=True):
    rng = np.random.default_rng(seed)
    spot = rng.integers(0, n_spots, n)
    df = pd.DataFrame(
        {
            "spot_id": spot.astype(str),
            "x_um": (spot % 40) * 20.0,
            "y_um": (spot // 40) * 20.0,
            "locus": "IGH",
            "clone_id": rng.choice([f"c{i:03d}" for i in range(n_clones)], n),
        }
    )
    if planted:
        box = (df["x_um"] < 160) & (df["y_um"] < 160)
        df.loc[box & (rng.random(n) < 0.6), "clone_id"] = "planted"
    return df


def test_measures_known_values():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = _batch_measures(square[None], ["mean_nn", "mst"])[0]
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(3.0)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (20, 25, 2))
    b = _batch_measures(pts, ["mean_nn", "mst"])
    for i in range(20):
        d, _ = cKDTree(pts[i]).query(pts[i], k=2)
        assert b[i, 0] == d[:, 1].mean()  # bitwise equal to the archive's mean_nn
        assert b[i, 1] == pytest.approx(_mst_length(pts[i]))
    line = np.column_stack([np.arange(2500.0), 2 * np.arange(2500.0)])  # collinear, above the dense limit
    assert _mst_length(line) == pytest.approx(2499 * np.sqrt(5))
    big = rng.uniform(0, 1000, (2050, 2))
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    assert _mst_length(big) == pytest.approx(minimum_spanning_tree(squareform(pdist(big))).sum())


def test_legacy_rule_replays_the_archive_s1_draws():
    air = _grid_air()

    def mean_nn(xy):
        d, _ = cKDTree(xy).query(xy, k=2)
        return d[:, 1].mean()

    rng = np.random.default_rng(0)
    allxy = air.drop_duplicates("spot_id")[["x_um", "y_um"]].to_numpy()
    cache, rows = {}, []
    for c, g in air.groupby("clone_id"):
        g1 = g.drop_duplicates("spot_id")
        n = len(g1)
        if n < 5:
            continue
        obs = mean_nn(g1[["x_um", "y_um"]].to_numpy())
        if n not in cache:
            cache[n] = np.array([mean_nn(allxy[rng.choice(len(allxy), n, replace=False)]) for _ in range(50)])
        z = (obs - cache[n].mean()) / cache[n].std(ddof=1)
        rows.append({"clone_id": c, "z": z, "p": norm.cdf(z)})
    ref = pd.DataFrame(rows)
    ref["q"] = bh(ref["p"].to_numpy())
    r = tl.clone_dispersion(
        air, unit="spot", n_perm=50, measures=("mean_nn",), p_method="normal", sequential=False, rng=0
    )
    m = ref.merge(r.table, on="clone_id")
    assert len(m) == len(ref) == len(r.table)
    np.testing.assert_array_equal(m["z_x"].to_numpy(), m["z_y"].to_numpy())
    np.testing.assert_array_equal(m["q"].to_numpy(), m["q_value"].to_numpy())


def test_besag_clifford_p_value():
    pool = np.array([5.0, 1.0, 6.0, 0.5, 7.0, 0.2, 8.0])  # observed 1.0, alternative "less": exceedances at 2, 4, 6
    p, used = _besag_clifford(pool, 1.0, "less", h=2)
    assert (p, used) == (2 / 4, 4)
    p, used = _besag_clifford(pool, 0.1, "less", h=2)
    assert (p, used) == (1 / 8, 7)


def test_sequential_inhibition_keeps_the_minimum_separation():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 300, (400, 2))
    frame = _Frame(xy, None, None)
    for _ in range(20):
        idx = _ssi(rng, frame, 30, None, 15.0)
        d = cKDTree(xy[idx]).query(xy[idx], k=2)[0][:, 1]
        assert len(set(idx)) == 30 and d.min() > 15.0
    strata = np.repeat([0, 1], 200)
    frame2 = _Frame(xy, strata, [np.flatnonzero(strata == 0), np.flatnonzero(strata == 1)])
    idx = _ssi(rng, frame2, 10, np.array([7, 3]), 15.0)
    assert np.bincount(strata[idx], minlength=2).tolist() == [7, 3]


def test_power_on_a_planted_clustered_clone():
    r = tl.clone_dispersion(_grid_air(), unit="spot", rng=1)
    t = r.table.set_index(["clone_id", "measure"])
    assert t.loc[("planted", "mean_nn"), "significant"]
    assert t.loc[("planted", "mst"), "q_value"] < 0.05
    assert t.loc[("planted", "mean_nn"), "ratio"] < 0.5
    assert r.params["max_perm"] >= len(r.table) // 2 / 0.05
    loci = tl.clone_dispersion(_grid_air(), unit="locus", measures=("mean_nn",), rng=1)
    assert loci.table.set_index("clone_id").loc["planted", "significant"]


def _calibration(reps, seed):
    rng = np.random.default_rng(seed)
    p = []
    for _ in range(reps):
        air = _grid_air(n=400, n_spots=600, n_clones=40, seed=int(rng.integers(1 << 31)), planted=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)
            r = tl.clone_dispersion(
                air, unit="spot", n_perm=99, measures=("mean_nn",), sequential=False, min_units=6, rng=rng
            )
        p.extend(r.table["p_value"].tolist())
    return np.asarray(p)


def test_calibration_under_random_labels():
    """Clone labels placed at random: per-clone Monte-Carlo p-values are uniform (P(p <= 0.05) <= 0.05 + tolerance)."""
    p = _calibration(12, 7)
    assert len(p) > 40
    assert (p <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / len(p))


@pytest.mark.slow
def test_calibration_under_random_labels_many_reps():
    p = _calibration(150, 8)
    assert (p <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / len(p))
    assert (p <= 0.5).mean() == pytest.approx(0.5, abs=0.06)


def test_design_placebo_and_within_strata():
    air = _grid_air(n=800, n_spots=600, n_clones=60)
    air["half"] = np.where(air["y_um"] < 300, "low", "high")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        r = tl.clone_dispersion(
            air, unit="record", design=nulls.Permute("clone_id"), n_perm=49, measures=("mean_nn",), rng=2
        )
    t = r.table.set_index("clone_id")
    assert t.loc["planted", "p_value"] <= 0.04
    w = tl.clone_dispersion(air, unit="spot", within="half", measures=("mean_nn",), rng=2)
    assert w.table.set_index("clone_id").loc["planted", "significant"]


def test_determinism_and_errors():
    air = _grid_air(n=500, n_spots=400, n_clones=40)
    a = tl.clone_dispersion(air, unit="spot", measures=("mean_nn",), rng=5)
    b = tl.clone_dispersion(air, unit="spot", measures=("mean_nn",), rng=5)
    pd.testing.assert_frame_equal(a.table, b.table)
    c = tl.clone_dispersion(air, unit="spot", measures=("mean_nn",), rng=6)
    assert not np.array_equal(a.table["null_mean"].to_numpy(), c.table["null_mean"].to_numpy())
    with pytest.raises(ValueError, match="measure"):
        tl.clone_dispersion(air, measures=("hull",))
    with pytest.raises(ValueError, match="min_units"):
        tl.clone_dispersion(air, min_units=1)
    with pytest.raises(ValueError, match="p_method"):
        tl.clone_dispersion(air, p_method="exact")
    with pytest.warns(LowPowerWarning):
        tl.clone_dispersion(air, unit="spot", measures=("mean_nn",), sequential=False, n_perm=20, rng=1)
