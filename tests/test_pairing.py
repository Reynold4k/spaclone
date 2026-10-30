import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import qc
from spaclone.qc.pairing import _curveball_sweep


def _legacy_b5c(h, lu, size, n_perm, rng, strat):
    """The inner test of B5c_pairing.py (archive: BCR analysis, workflow) on DNB coordinates, verbatim but returning."""

    def bin_matrix(keys, labels, bidx, cidx):
        r, c = bidx.get_indexer(keys), cidx.get_indexer(labels)
        m = (r >= 0) & (c >= 0)
        return sparse.csr_matrix((np.ones(m.sum(), np.float32), (r[m], c[m])), shape=(len(bidx), len(cidx)))

    hk = (h.x // size).to_numpy(np.int64) * 100000 + (h.y // size).to_numpy(np.int64)
    lk = (lu.x // size).to_numpy(np.int64) * 100000 + (lu.y // size).to_numpy(np.int64)
    bidx = pd.Index(np.union1d(hk, lk))
    hc, lc = h.cloneId.value_counts(), lu.cloneId.value_counts()
    hkeep, lkeep = pd.Index(hc.index[hc >= 50]), pd.Index(lc.index[lc >= 50])
    L = bin_matrix(lk, lu.cloneId.to_numpy(), bidx, lkeep).tocsc()
    rows_h = bidx.get_indexer(hk)
    codes, uniq = pd.factorize(h.cloneId.to_numpy())
    col_h = hkeep.get_indexer(uniq)[codes]

    def heavy_matrix(col):
        m = col >= 0
        return sparse.csr_matrix((np.ones(m.sum(), np.float32), (rows_h[m], col[m])), shape=(len(bidx), len(hkeep)))

    groups = [np.flatnonzero(h.stratum.to_numpy() == k) for k in range(4)] if strat else [np.arange(len(col_h))]

    def shuffled():
        c = col_h.copy()
        for g in groups:
            c[g] = col_h[rng.permutation(g)]
        return c

    O = (heavy_matrix(col_h).T @ L).toarray()
    Ob = np.empty((n_perm,) + O.shape, np.float32)
    for b in range(n_perm):
        Ob[b] = (heavy_matrix(shuffled()).T @ L).toarray()
    mu, sd = Ob.mean(0), Ob.std(0)
    sd = np.maximum(sd, np.sqrt(np.maximum(mu, 0.25)))
    Z, Zb = (O - mu) / sd, (Ob - mu) / sd
    zmax, zbmax = Z.max(1), Zb.max(2)
    p_max = (1 + (zbmax >= zmax).sum(0)) / (n_perm + 1)
    g_obs, g_null = zmax.mean(), zbmax.mean(1)
    return {
        "heavy": hkeep,
        "z": zmax,
        "best": lkeep[Z.argmax(1)],
        "p_max": p_max,
        "g_obs": float(g_obs),
        "g_null_mean": float(g_null.mean()),
        "p_global": float((1 + (g_null >= g_obs).sum()) / (n_perm + 1)),
    }


_H = np.array([f"H{i}" for i in range(40)])
_K = np.array([f"K{i}" for i in range(40)])


def _chip(seed=0, n_spots=6000, extent=1500, n_h=12, n_l=15, planted=0.0, crowd=0.25, lib="L1"):
    """(spot, clone) units in DNB coordinates (0.5 um), IGH and IGK, some spots holding several units."""
    rng = np.random.default_rng(seed)
    sx, sy = rng.integers(0, extent * 2, n_spots), rng.integers(0, extent * 2, n_spots)
    k = np.where(rng.random(n_spots) < crowd, rng.integers(2, 6, n_spots), 1)
    spot = np.repeat(np.arange(n_spots), k)
    heavy = rng.random(len(spot)) < 0.45
    wh, wl = 1 / np.arange(1, n_h + 1) ** 0.5, 1 / np.arange(1, n_l + 1) ** 0.5
    clone = np.where(
        heavy, _H[rng.choice(n_h, len(spot), p=wh / wh.sum())], _K[rng.choice(n_l, len(spot), p=wl / wl.sum())]
    )
    x, y, chain = sx[spot], sy[spot], np.where(heavy, "IGH", "IGK")
    if planted:  # cells: heavy clone Hi and its partner Ki put one unit each into the same 10 um bin
        n_cells = int(planted * n_spots)
        cx, cy, ci = (
            rng.integers(0, extent * 2, n_cells),
            rng.integers(0, extent * 2, n_cells),
            rng.integers(0, n_h, n_cells),
        )
        cell_spot = n_spots + np.arange(2 * n_cells)
        spot = np.r_[spot, cell_spot]
        x = np.r_[x, np.tile((cx // 20) * 20, 2) + rng.integers(0, 20, 2 * n_cells)]
        y = np.r_[y, np.tile((cy // 20) * 20, 2) + rng.integers(0, 20, 2 * n_cells)]
        chain = np.r_[chain, np.repeat(["IGH", "IGK"], n_cells)]
        clone = np.r_[clone, _H[ci], _K[ci]]
    df = pd.DataFrame({"spot": spot, "x": x, "y": y, "chain": chain, "cloneId": clone})
    df = df.drop_duplicates(["spot", "chain", "cloneId"]).reset_index(drop=True)
    df["mult_all"] = df.groupby("spot").spot.transform("size")
    df["stratum"] = pd.cut(df.mult_all, [0, 1, 2, 5, 19], labels=False)
    return df.assign(
        spot_id=df.spot.astype(np.int64) + 10_000_000 * int(lib[1:]) if lib[1:].isdigit() else df.spot,
        x_um=df.x * 0.5,
        y_um=df.y * 0.5,
        locus=df.chain,
        clone_id=df.cloneId,
        library_id=lib,
    )


def test_cooccurrence_known_answer():
    # two heavy and two light clones, 60 units each, in two 10-um bins; H0 meets K0 only, H1 meets K1 only
    rows = []
    for b, (hc_, lc_) in enumerate((("H0", "K0"), ("H1", "K1"))):
        for i in range(60):
            rows.append((f"h{b}_{i}", b * 50.0 + 1, 1.0, "IGH", hc_))
            rows.append((f"l{b}_{i}", b * 50.0 + 2, 2.0, "IGK", lc_))
    df = pd.DataFrame(rows, columns=["spot_id", "x_um", "y_um", "locus", "clone_id"])
    r = qc.pairing(df, light="IGK", n_perm=49, rng=0)
    c = r.clones.set_index("heavy")
    assert c.loc["H0", "best_light"] == "K0" and c.loc["H1", "best_light"] == "K1"
    assert c.loc["H0", "O"] == 60 * 60
    assert c.loc["H0", "O_same_spot"] == 0
    t = r.table.iloc[0]
    assert t.n_heavy == 2 and t.n_light == 2 and t.design == "stratified"
    assert r.null.shape == (49, 1)


def test_matches_legacy_b5c_draw_for_draw():
    df = _chip(seed=1, planted=0.05)
    h, lu = df[df.chain == "IGH"], df[df.chain == "IGK"]
    for strat, bins in ((True, (1, 2, 3, 6, np.inf)), (False, None)):
        leg = _legacy_b5c(h, lu, 20, 99, np.random.default_rng(11), strat)
        r = qc.pairing(df, light="IGK", strata_bins=bins, n_perm=99, rng=np.random.default_rng(11))
        t = r.table.iloc[0]
        assert t.statistic == pytest.approx(leg["g_obs"], rel=1e-5)
        assert t.null_mean == pytest.approx(leg["g_null_mean"], rel=1e-5)
        assert t.p_value == pytest.approx(leg["p_global"])
        c = r.clones.set_index("heavy").loc[list(leg["heavy"])]
        np.testing.assert_allclose(c.z, leg["z"], rtol=1e-5)
        np.testing.assert_allclose(c.p_value, leg["p_max"])
        assert list(c.best_light) == list(leg["best"])


def _calibration(reps, null, n_perm=39, **kw):
    df = pd.concat([_chip(seed=100 + s, n_spots=900, extent=300, lib=f"L{s:03d}") for s in range(reps)])
    t = qc.pairing(df, light="IGK", null=null, n_perm=n_perm, rng=5, **kw).table
    assert len(t) == reps and t.statistic.notna().all()
    assert (t.p_value <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_calibration_independent_chains():
    """Heavy and light clones placed independently: the global test rejects at <= alpha + tolerance."""
    _calibration(40, "stratified")


@pytest.mark.slow
def test_calibration_independent_chains_many():
    _calibration(300, "stratified")


def test_power_on_planted_cells():
    df = _chip(seed=2, planted=0.08)
    r = qc.pairing(df, light="IGK", n_perm=99, rng=3)
    t = r.table.iloc[0]
    assert t.p_value <= 0.01
    c = r.clones
    assert (c.best_light.str[1:] == c.heavy.str[1:]).mean() >= 0.75
    assert (c.q_value < 0.05).sum() >= 6
    assert (c.z > c.z_second).all()


def _crowded_chip(seed=4, n_single=4000, n_crowded=1000, extent=1200):
    """One heavy unit per spot. Crowded spots (1 heavy + 3 light units) carry the heavy clones H0-H2 and the light
    clone K0; singleton spots carry the other clones. No heavy clone pairs with any light clone."""
    rng = np.random.default_rng(seed)
    n = n_single + n_crowded
    sx, sy = rng.integers(0, extent * 2, n), rng.integers(0, extent * 2, n)
    single_h = rng.random(n_single) < 0.5
    rows = [
        pd.DataFrame(
            {
                "spot": np.arange(n_single),
                "chain": np.where(single_h, "IGH", "IGK"),
                "cloneId": np.where(single_h, _H[rng.integers(3, 12, n_single)], _K[rng.integers(1, 15, n_single)]),
            }
        )
    ]
    cs = n_single + np.arange(n_crowded)
    rows.append(pd.DataFrame({"spot": cs, "chain": "IGH", "cloneId": _H[rng.integers(0, 3, n_crowded)]}))
    rows.append(pd.DataFrame({"spot": cs, "chain": "IGK", "cloneId": "K0"}))
    others = np.argsort(rng.random((n_crowded, 14)), axis=1)[:, :2] + 1  # two distinct light clones from K1-K14
    rows.append(pd.DataFrame({"spot": np.repeat(cs, 2), "chain": "IGK", "cloneId": _K[others.ravel()]}))
    df = pd.concat(rows, ignore_index=True)
    return df.assign(
        spot_id=df.spot, x_um=sx[df.spot] * 0.5, y_um=sy[df.spot] * 0.5, locus=df.chain, clone_id=df.cloneId
    )


def test_stratified_and_curveball_nulls_absorb_clone_specific_crowding():
    """Clones that favour crowded spots meet there; only the whole-sample null mistakes that for pairing."""
    df = _crowded_chip()
    whole = qc.pairing(df, light="IGK", strata_bins=None, n_perm=59, rng=1).table.iloc[0]
    strat = qc.pairing(df, light="IGK", n_perm=59, rng=1).table.iloc[0]
    curve = qc.pairing(df, light="IGK", null="curveball", n_perm=29, n_sweeps=4, rng=1).table.iloc[0]
    assert whole.p_value <= 0.02 and whole.max_heavy_per_best_light >= 3  # K0 is the "hub" partner of H0-H2
    assert strat.p_value > 0.05 and curve.p_value > 0.05
    assert whole.design == "whole_sample" and curve.design == "curveball"


def test_curveball_sweep_preserves_margins_and_binariness():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 400, 30
    m = rng.random((n_rows, n_cols)) < 0.08
    m[np.arange(n_rows), rng.integers(0, n_cols, n_rows)] = True  # every row holds a column
    row, col = np.nonzero(m)
    st = rng.integers(0, 3, n_rows)
    cur = row.copy()
    changed = False
    for _ in range(25):
        cur = _curveball_sweep(cur, col, n_cols, st, rng)
        new = np.zeros_like(m)
        np.add.at(new, (cur, col), 1)
        assert new.max() == 1  # binary
        np.testing.assert_array_equal(new.sum(1), m.sum(1))  # row sums
        for s in range(3):  # column sums inside every row stratum
            np.testing.assert_array_equal(new[st == s].sum(0), m[st == s].sum(0))
        changed |= not np.array_equal(new, m)
    assert changed


def test_curveball_equals_permutation_on_singleton_spots():
    # one unit per spot: a binary matrix with fixed margins is a label permutation, so the two nulls agree
    df = _chip(seed=6, n_spots=3000, crowd=0.0)
    perm = qc.pairing(df, light="IGK", n_perm=59, rng=0).clones.set_index("heavy")
    curve = qc.pairing(df, light="IGK", null="curveball", n_perm=59, n_sweeps=6, rng=0).clones.set_index("heavy")
    curve = curve.loc[perm.index]
    np.testing.assert_array_equal(curve.n_heavy, perm.n_heavy)
    # the best pairs may differ; compare the null means of each heavy clone's best permutation partner
    assert (np.abs(curve.mu - perm.mu) <= 5 * np.sqrt(perm.mu / 59) + 0.5)[perm.best_light == curve.best_light].all()
    assert abs(curve.z.mean() - perm.z.mean()) < 0.6


def test_curveball_power_on_planted_cells():
    df = _chip(seed=2, planted=0.08)
    t = qc.pairing(df, light="IGK", null="curveball", n_perm=29, n_sweeps=4, rng=3).table.iloc[0]
    assert t.p_value <= 0.05 and t.n_q05 >= 4


@pytest.mark.slow
def test_curveball_calibration_independent_chains():
    _calibration(100, "curveball", n_sweeps=8)


def test_singleton_only_determinism_and_untested():
    df = _chip(seed=7, planted=0.03)
    a = qc.pairing(df, light=("IGK", "IGL"), singleton_only=True, n_perm=29, rng=2)
    b = qc.pairing(df, light=("IGK", "IGL"), singleton_only=True, n_perm=29, rng=2)
    pd.testing.assert_frame_equal(a.table, b.table)
    pd.testing.assert_frame_equal(a.clones, b.clones)
    t = a.table.set_index("light_locus")
    assert t.loc["IGK", "n_heavy_units"] == int(((df.mult_all == 1) & (df.locus == "IGH")).sum())
    assert (a.clones.O_same_spot == 0).all()  # singleton spots never hold two chains
    assert np.isnan(t.loc["IGL", "statistic"])  # no IGL units
    assert not t.loc["IGL", "testable"] and t.loc["IGL", "note"].startswith("untested")
    assert t.loc["IGK", "testable"] and t.loc["IGK", "q_floor"] == pytest.approx(t.loc["IGK", "n_heavy"] / 30)
    assert a.summary["n_untestable"] == 1
    assert a.null.shape == (29, 2) and np.isnan(a.null[:, 1]).all()
    with pytest.raises(ValueError):
        qc.pairing(df, heavy="IGH", light="IGH")
    with pytest.raises(ValueError):
        qc.pairing(df, null="exact")


def test_placement_plot_of_pairing():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spaclone import pl

    df = pd.concat([_chip(seed=8, n_spots=1500, lib="L1"), _chip(seed=9, n_spots=1500, lib="L2")])
    r = qc.pairing(df, light="IGK", n_perm=9, rng=0)
    ax = pl.placement(r)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["L1", "L2"]
    plt.close(ax.figure)
