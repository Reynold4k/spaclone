"""Tests of spaclone.tl.transitions: transition models (sciCSR / CellRank kernels) and transition path theory."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import tl
from spaclone._utils import LowPowerWarning, SpacloneWarning
from spaclone.tl import transitions as tr


def _grid_units(n_side=12, step=10.0, *, gradient=1.0, noise=0.05, seed=0):
    """Units on a square grid (µm), three vertical bands as states, time = gradient * x + noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side) * step, np.arange(n_side) * step)
    x, y = xs.ravel(), ys.ravel()
    span = x.max()
    band = np.where(x < span / 3, "L", np.where(x < 2 * span / 3, "M", "R"))
    t = gradient * x / span + rng.normal(0, noise, len(x))
    return pd.DataFrame({"x_um": x, "y_um": y, "t": t, "state": band}, index=[f"u{i}" for i in range(len(x))])


# ================================================================================================ unit tests
def test_stationary_two_state_chain_hand_computed():
    p, q = 0.3, 0.1
    t = np.array([[1 - p, p], [q, 1 - q]])
    pi = tl.stationary_distribution(t)
    np.testing.assert_allclose(pi, [q / (p + q), p / (p + q)], atol=1e-14)


def test_committors_three_state_chain_hand_computed():
    # deeptime's documented example: q+ = (0, 1/2, 1), q- = (1, 5/11, 0)
    t = sparse.csr_matrix(np.array([[0.89, 0.1, 0.01], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]]))
    pi = tl.stationary_distribution(t)
    qp, qm = tr._committors(t, pi, np.array([True, False, False]), np.array([False, False, True]))
    np.testing.assert_allclose(qp, [0.0, 0.5, 1.0], atol=1e-14)
    np.testing.assert_allclose(qm, [1.0, 5.0 / 11.0, 0.0], atol=1e-14)


def test_birth_death_chain_committor_and_flux():
    # 0 <-> 1 <-> 2 with A = {0}, B = {2}: q+_1 = c / (b + c); the reactive flux runs 0 -> 1 -> 2 only
    a, b, c, d = 0.2, 0.3, 0.1, 0.4
    t = sparse.csr_matrix(np.array([[1 - a, a, 0], [b, 1 - b - c, c], [0, d, 1 - d]]))
    pi = tl.stationary_distribution(t)
    qp, qm = tr._committors(t, pi, np.array([True, False, False]), np.array([False, False, True]))
    assert qp[1] == pytest.approx(c / (b + c), abs=1e-14)
    assert qm[1] == pytest.approx(1 - c / (b + c), abs=1e-14)  # reversible chain: q- = 1 - q+
    f = tr._gross_flux(t, pi, qm, qp).toarray()
    assert f[0, 1] == pytest.approx(pi[0] * 1.0 * a * qp[1], abs=1e-15)
    assert f[1, 2] == pytest.approx(pi[1] * qm[1] * c * 1.0, abs=1e-15)
    assert f[0, 1] == pytest.approx(f[1, 2], rel=1e-12)  # flux conservation through the intermediate
    assert f[1, 0] == f[2, 1] == 0.0


def test_hard_threshold_scheme_hand_computed():
    # unit 0 with 10 neighbours of distinct weight: the floor(10 * 0.3) = 3 strongest are kept whatever their time,
    # the others only when their time is not earlier than unit 0's
    w = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])
    t = np.r_[0.5, [0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.6, 0.4, 0.5, 0.0]]
    conn = sparse.csr_matrix((w, (np.zeros(10, int), np.arange(1, 11))), shape=(11, 11))
    keep = np.zeros(10, bool)
    keep[:3] = True  # three strongest
    keep |= t[1:] >= 0.5  # not in the past
    for ties in ("keep", "cellrank"):  # distinct weights: both tie rules agree
        b = tr._biased_connectivities(conn, t, scheme="hard", frac_to_keep=0.3, b=10.0, nu=0.5, ties=ties)
        np.testing.assert_array_equal(b.toarray()[0, 1:], np.where(keep, w, 0.0))


def test_hard_threshold_ties():
    # 10 neighbours of equal weight, all in the past: "keep" keeps every tied neighbour, "cellrank" keeps the
    # floor(10 * 0.3) = 3 listed last (np.flip of a stable argsort)
    w = np.ones(10)
    t = np.r_[1.0, np.zeros(10)]
    conn = sparse.csr_matrix((w, (np.zeros(10, int), np.arange(1, 11))), shape=(11, 11))
    keep = tr._biased_connectivities(conn, t, scheme="hard", frac_to_keep=0.3, b=10.0, nu=0.5, ties="keep")
    cr = tr._biased_connectivities(conn, t, scheme="hard", frac_to_keep=0.3, b=10.0, nu=0.5, ties="cellrank")
    assert keep.toarray()[0, 1:].sum() == 10
    np.testing.assert_array_equal(cr.toarray()[0, 1:], np.r_[np.zeros(7), np.ones(3)])


def test_soft_threshold_scheme_hand_computed():
    w = np.array([1.0, 0.5, 0.25])
    t = np.array([0.5, 0.2, 0.7, 0.5])
    conn = sparse.csr_matrix((w, (np.zeros(3, int), np.array([1, 2, 3]))), shape=(4, 4))
    b = tr._biased_connectivities(conn, t, scheme="soft", frac_to_keep=0.3, b=10.0, nu=0.5).toarray()[0, 1:]
    expect = np.array([1.0 * 2.0 / (1 + np.exp(10 * 0.3)) ** 2, 0.5, 0.25])
    np.testing.assert_allclose(b, expect, rtol=1e-14)


def test_connectivity_kernel_density_normalisation():
    c = sparse.csr_matrix(np.array([[0, 1.0, 0.5], [1.0, 0, 0.2], [0.5, 0.2, 0]]))
    q = np.asarray(c.sum(axis=0)).ravel()
    m = np.diag(1 / q) @ c.toarray() @ np.diag(1 / q)
    expect = m / m.sum(1, keepdims=True)
    got = tr._connectivity_kernel(c, density_normalize=True).toarray()
    np.testing.assert_allclose(got, expect, rtol=1e-14)


def test_transition_model_combination_rows_and_weight():
    u = _grid_units(8)
    m = tl.transition_model(u, time_key="t", graph="spatial", k=8, weight=0.8, states="state")
    np.testing.assert_allclose(np.asarray(m.transition_matrix.sum(1)).ravel(), 1.0, atol=1e-12)
    d = m.transition_matrix - (0.8 * m.pseudotime_matrix + 0.2 * m.connectivity_matrix)
    assert abs(d).max() < 1e-15
    assert list(m.states.categories) == ["L", "M", "R"]
    assert m.coords.shape == (64, 2)


def test_spatial_graph_union_and_bandwidth():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    g = tl.neighbor_graph(pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1]}), graph="spatial", k=1, bandwidth_um=10)
    a = g.toarray()
    assert a[0, 1] == pytest.approx(np.exp(-0.5)) and a[1, 0] == a[0, 1]
    assert a[2, 1] == pytest.approx(np.exp(-400 / 200))  # unit 2's nearest is 1: union keeps the edge both ways
    assert a[0, 2] == 0.0 and np.all(np.diag(a) == 0)


def test_coarse_grain_lumping_is_stochastic_and_stationary():
    m = tl.transition_model(_grid_units(9), time_key="t", graph="spatial", k=8, states="state")
    p = m.coarse_grain()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    pi_c = pd.Series(m.pi).groupby(np.asarray(m.states)).sum().loc[p.index].to_numpy()
    np.testing.assert_allclose(pi_c @ p.to_numpy(), pi_c, atol=1e-12)


def test_tpt_flux_conservation_and_tables():
    m = tl.transition_model(_grid_units(10), time_key="t", graph="spatial", k=8, states="state")
    res = tl.tpt_flux(m, source="L", target="R", n_random=0)
    net = res.net.to_numpy()
    assert res.gross.to_numpy().sum() == pytest.approx(100.0)
    # reactive flux leaves L, passes M, enters R: net flux out of L equals net flux into R
    raw = res.table.pivot(index="from_state", columns="to_state", values="net_flux_raw").fillna(0)
    assert raw.loc["L"].sum() == pytest.approx(raw["R"].sum(), rel=1e-9)
    assert res.summary["total_flux"] == pytest.approx(raw.loc["L"].sum(), rel=1e-12)
    assert res.pathways.percent.sum() == pytest.approx(100.0, abs=1e-6)
    assert res.pathways.path.iloc[0] == "L > M > R"
    assert net[2].sum() == 0  # nothing leaves the target
    assert set(res.states.role) == {"source", "intermediate", "target"}
    assert res.states.loc["L", "q_plus"] == 0 and res.states.loc["R", "q_plus"] == 1
    assert {"x_um", "y_um", "q_plus", "q_minus", "stationary"} <= set(res.units.columns)


def test_labels_null_replays_given_permutations():
    m = tl.transition_model(_grid_units(7), time_key="t", graph="spatial", k=6, states="state")
    rng = np.random.default_rng(5)
    perms = np.array([rng.permutation(m.n_units) for _ in range(5)])
    res = tl.tpt_flux(
        m, source="L", target="R", null="labels", n_random=5, permutations=perms, statistic="gross_percent"
    )
    sizes = np.bincount(np.asarray(m.states.codes), minlength=3)
    codes = np.empty(m.n_units, int)
    codes[perms[2]] = np.repeat(np.arange(3), sizes)
    manual = tr._tpt_native(m.transition_matrix, m.pi, codes, 3, np.array([0]), np.array([2]))["gross"]
    np.testing.assert_allclose(res.null[2], tr._percent(manual), atol=1e-12)


def test_displacement_points_up_the_gradient_and_flow_grid():
    m = tl.transition_model(_grid_units(12, gradient=1.0, noise=0.02), time_key="t", graph="spatial", k=8)
    d = tl.transition_displacement(m)
    assert d.dx_um.mean() > 0.5 and abs(d.dy_um.mean()) < 0.1 * d.dx_um.mean()
    flat = tl.transition_model(_grid_units(12, gradient=0.0, noise=0.0), time_key="t", graph="spatial", k=8)
    d0 = tl.transition_displacement(flat)
    assert abs(d0.dx_um.mean()) < 0.2 * d.dx_um.mean()
    u = tl.transition_displacement(m, method="unit")
    assert u.dx_um.mean() > 0
    g = tl.flow_grid(pd.DataFrame({"x_um": d.x_um, "y_um": d.y_um, "dx_um": 1.0, "dy_um": -2.0}), grid_um=20)
    np.testing.assert_allclose(g.dx_um, 1.0)
    np.testing.assert_allclose(g.dy_um, -2.0)
    assert g.attrs["spaclone"]["flow_grid"]["grid_um"] == 20


def test_disconnected_graph_keeps_largest_component():
    u = _grid_units(6)
    far = u.iloc[:3].copy()
    far.index = ["f0", "f1", "f2"]
    far["x_um"] += 1e6
    uu = pd.concat([u, far])
    with pytest.warns(SpacloneWarning, match="connected components"):
        m = tl.transition_model(uu, time_key="t", graph="spatial", k=4, max_distance_um=50)
    assert m.n_units == len(u) and set(m.dropped) == {"f0", "f1", "f2"}


def test_input_errors():
    u = _grid_units(5)
    with pytest.raises(ValueError, match="weight"):
        tl.transition_model(u, time_key="t", weight=1.5)
    bad = u.assign(t=np.where(np.arange(len(u)) == 0, np.nan, u.t))
    with pytest.raises(ValueError, match="time"):
        tl.transition_model(bad, time_key="t")
    m = tl.transition_model(u, time_key="t", k=4, states="state")
    with pytest.raises(ValueError, match="disjoint"):
        tl.tpt_flux(m, source="L", target=["L", "R"], n_random=0)
    with pytest.raises(ValueError, match="not among"):
        tl.tpt_flux(m, source="X", target="R", n_random=0)
    with pytest.raises(ValueError, match="expression graph"):
        tl.transition_model(u, time_key="t", graph="expression")


def test_constant_time_within_states_warns():
    u = _grid_units(6)
    u["t"] = u.state.map({"L": 0.0, "M": 0.5, "R": 1.0})
    m = tl.transition_model(u, time_key="t", k=4, states="state")
    with pytest.warns(LowPowerWarning, match="constant within every state"):
        res = tl.tpt_flux(m, source="L", target="R", n_random=5, rng=0)
    assert np.all(res.table.p_value == 1.0)


# ================================================================================================ calibration / power
def _calibration(null, alpha, n_sim, n_rand):
    hits = 0
    for s in range(n_sim):
        u = _grid_units(7, gradient=0.0, noise=1.0, seed=100 + s)
        m = tl.transition_model(u, time_key="t", graph="spatial", k=6, states="state")
        r = tl.tpt_flux(m, source="L", target="R", null=null, n_random=n_rand, rng=s, n_bootstrap=0)
        hits += r.summary["rate_p_value"] <= alpha
    tol = 2.5 * np.sqrt(alpha * (1 - alpha) / n_sim)
    assert hits / n_sim <= alpha + tol, (null, hits / n_sim)


def test_null_calibration_without_time_gradient():
    """Rate test under H0 (time i.i.d.): the false-positive rate stays within alpha + binomial tolerance.

    Null model
    ----------
    The time key is permuted among the units of each state (``null="time_within_states"``, the default); graph,
    states, source and target stay fixed. Data are simulated with an i.i.d. time key, so observed and permuted time
    keys are exchangeable and the Monte-Carlo p-value is valid.

    References
    ----------
    Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Metzner, Schütte & Vanden-Eijnden 2009,
    doi:10.1137/070699500.
    """
    _calibration("time_within_states", alpha=0.1, n_sim=60, n_rand=19)


@pytest.mark.slow
@pytest.mark.parametrize("null", ["time_within_states", "time"])
def test_null_calibration_without_time_gradient_slow(null):
    """Larger calibration run of both time nulls at alpha = 0.05.

    Null model
    ----------
    As :func:`test_null_calibration_without_time_gradient`, for the within-state and the global time shuffle.

    References
    ----------
    Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    _calibration(null, alpha=0.05, n_sim=300, n_rand=39)


def test_power_with_planted_gradient():
    """A time gradient from the source band to the target band is detected by the rate test.

    Null model
    ----------
    The time key is permuted over all units (``null="time"``); the graph and the band states stay fixed.

    References
    ----------
    Metzner, Schütte & Vanden-Eijnden 2009, doi:10.1137/070699500.
    """
    sig = 0
    for s in range(8):
        u = _grid_units(8, gradient=1.0, noise=0.1, seed=s)
        m = tl.transition_model(u, time_key="t", graph="spatial", k=6, states="state")
        r = tl.tpt_flux(m, source="L", target="R", null="time", n_random=39, rng=s, n_bootstrap=0)
        sig += r.summary["rate_p_value"] <= 0.05
        assert r.summary["rate"] > r.summary["rate_null_mean"]
    assert sig >= 7


def test_determinism():
    m = tl.transition_model(_grid_units(7), time_key="t", graph="spatial", k=6, states="state")
    a = tl.tpt_flux(m, source="L", target="R", n_random=10, rng=3)
    b = tl.tpt_flux(m, source="L", target="R", n_random=10, rng=3)
    c = tl.tpt_flux(m, source="L", target="R", n_random=10, rng=4)
    np.testing.assert_array_equal(a.null, b.null)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.null, c.null)
    m2 = tl.transition_model(_grid_units(7), time_key="t", graph="spatial", k=6, states="state")
    assert abs(m.transition_matrix - m2.transition_matrix).max() == 0


def test_translation_and_row_order_invariance():
    u = _grid_units(7)
    m = tl.transition_model(u, time_key="t", k=6, states="state")
    v = u.assign(x_um=u.x_um + 1000.0, y_um=u.y_um - 50.0)
    m2 = tl.transition_model(v, time_key="t", k=6, states="state")
    assert abs(m.transition_matrix - m2.transition_matrix).max() < 1e-12
    perm = np.random.default_rng(0).permutation(len(u))
    m3 = tl.transition_model(u.iloc[perm], time_key="t", k=6, states="state")  # grid: many tied distances
    t1 = m.transition_matrix.toarray()[np.ix_(perm, perm)]
    np.testing.assert_allclose(m3.transition_matrix.toarray(), t1, atol=1e-12)
    r1 = tl.tpt_flux(m, source="L", target="R", n_random=0)
    r3 = tl.tpt_flux(m3, source="L", target="R", n_random=0)
    assert r1.summary["rate"] == pytest.approx(r3.summary["rate"], rel=1e-8)


def test_large_irregular_section_runs_in_seconds():
    """20,000 scattered units (a Stereo-seq-like section): model, TPT and two null draws well under a minute.

    The nested-dissection ordering of the spatial units keeps the sparse LU fill small; SuperLU's own orderings took
    minutes per factorisation on a real 63,000-unit section.
    """
    import time

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 2000, (20_000, 2))
    u = pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1]})
    u["t"] = np.round(u.x_um / 2000 * 6) / 6  # isotype-like tied levels
    u["s"] = np.where(u.x_um < 666, "L", np.where(u.x_um < 1333, "M", "R"))
    t0 = time.time()
    m = tl.transition_model(u, time_key="t", k=30, states="s")
    r = tl.tpt_flux(m, source="L", target="R", n_random=2, rng=0, n_bootstrap=10)
    assert time.time() - t0 < 60
    assert m.order is not None and sorted(m.order) == list(range(m.n_units))
    np.testing.assert_allclose(m.pi @ m.transition_matrix, m.pi, atol=1e-12)
    assert r.summary["rate"] > 0 and r.pathways.percent.sum() == pytest.approx(100.0, abs=1e-6)


# ================================================================================================ optional backends
def test_cellrank_backend_parity():
    pytest.importorskip("cellrank")
    u = _grid_units(8)
    jit = np.random.default_rng(7).uniform(-1, 1, (len(u), 2))  # distinct distances: no tied weights, which
    u = u.assign(x_um=u.x_um + jit[:, 0], y_um=u.y_um + jit[:, 1])  # CellRank breaks by its own sort order
    a = tl.transition_model(u, time_key="t", k=8, states="state")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = tl.transition_model(u, time_key="t", k=8, states="state", backend="cellrank")
    assert abs(a.transition_matrix - b.transition_matrix).max() < 1e-12


def test_deeptime_backend_parity():
    pytest.importorskip("deeptime")
    m = tl.transition_model(_grid_units(8), time_key="t", k=8, states="state")
    a = tl.tpt_flux(m, source="L", target="R", n_random=0)
    b = tl.tpt_flux(m, source="L", target="R", n_random=0, backend="deeptime")
    np.testing.assert_allclose(a.gross.to_numpy(), b.gross.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(a.states.stationary, b.states.stationary, atol=1e-10)
