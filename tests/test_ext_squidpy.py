"""Tests of spaclone.ext.squidpy: receptor-derived labels on spots and squidpy statistics on them."""

import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sq = pytest.importorskip("squidpy")

from spaclone.ext import squidpy as sqx  # noqa: E402
from spaclone.ext.squidpy import DepthNullWarning  # noqa: E402


def _gex(n_side: int = 4, step: float = 20.0) -> ad.AnnData:
    xy = np.array([(i * step, j * step) for i in range(n_side) for j in range(n_side)], dtype=float)
    obs = pd.DataFrame(index=[f"s{k}" for k in range(len(xy))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.AnnData(obs=obs, obsm={"spatial": xy})


def _rec(spot, xy, c_call, clone, locus="IGH", mutated=None):
    return {
        "spot_id": spot,
        "x_um": xy[0],
        "y_um": xy[1],
        "locus": locus,
        "c_call": c_call,
        "clone_id": clone,
        "mutated": mutated,
        "library_id": "s1",
    }


def _toy():
    g = _gex()
    xy = dict(zip(g.obs_names, map(tuple, g.obsm["spatial"]), strict=True))
    rows = [
        _rec("s0", xy["s0"], "IGHG2B*01", "X", mutated=True),
        _rec("s0", xy["s0"], "IGHG1", "X", mutated=True),
        _rec("s0", xy["s0"], "IGHM", "X", mutated=False),
        _rec("s0", xy["s0"], "IGKC", "K", locus="IGK"),
        _rec("s5", xy["s5"], "IGHM", "X", mutated=False),
        _rec("s5", xy["s5"], "IGHD", "Z", mutated=False),
        _rec("s10", xy["s10"], "IGHA", "Y", mutated=True),
        _rec("s10", xy["s10"], "IGHM", "Y", mutated=False),
    ]
    air = pd.DataFrame(rows)
    air["mutated"] = air["mutated"].astype("boolean")
    return air, g


def test_receptor_labels_known_answer():
    air, g = _toy()
    out = sqx.receptor_labels(air, g, n_top_clones=1)
    o = out.obs
    assert o.loc["s0", "air_n_records"] == 4 and o.loc["s15", "air_n_records"] == 0
    assert o.loc["s0", "air_isotype"] == "IgG"
    assert o.loc["s5", "air_isotype"] == "IgM"  # IgM / IgD tie: first class on the locus
    assert o.loc["s0", "air_switched_share"] == pytest.approx(2 / 3)
    assert (o.loc["s0", "air_switched"], o.loc["s5", "air_switched"], o.loc["s10", "air_switched"]) == (
        "switched",
        "unswitched",
        "mixed",
    )
    assert o.loc["s10", "air_mutated_share"] == pytest.approx(0.5)
    assert np.isnan(o.loc["s15", "air_switched_share"]) and pd.isna(o.loc["s15", "air_isotype"])
    # clone X sits on s0 and s5 (28 µm apart: two loci) -> expanded; Y, Z, K are single loci
    assert (o.loc["s0", "air_expanded"], o.loc["s10", "air_expanded"]) == ("expanded", "not expanded")
    assert o.loc["s0", "air_expanded_share"] == pytest.approx(1.0)  # X expanded; the light chain K is not a clone
    assert (o.loc["s0", "air_clone"], o.loc["s5", "air_clone"], o.loc["s10", "air_clone"]) == ("X", "X", "other")
    assert out.uns["spaclone"]["receptor_labels"]["match"] == "spot_id"
    every = sqx.receptor_labels(air, g, n_top_clones=1, chains=None).obs
    assert every.loc["s0", "air_expanded_share"] == pytest.approx(1 / 2)  # X expanded, K not


def test_receptor_labels_nearest_and_without_gex():
    air, g = _toy()
    moved = air.assign(spot_id="dnb_" + air.spot_id, x_um=air.x_um + 3.0)
    out = sqx.receptor_labels(moved, g)
    assert out.uns["spaclone"]["receptor_labels"]["match"] == "nearest"
    assert out.obs.loc["s0", "air_isotype"] == "IgG"
    far = sqx.receptor_labels(moved, g, match="nearest", max_distance_um=1.0)
    assert far.obs["air_n_records"].sum() == 0
    spots = sqx.receptor_labels(air)
    assert spots.n_obs == air.spot_id.nunique()
    with pytest.raises(KeyError, match="mutated"):
        sqx.receptor_labels(air.drop(columns="mutated"), g, labels=["mutated"])


def _random_labels(seed: int, n_side: int = 20, gradient: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    g = _gex(n_side)
    x = g.obsm["spatial"][:, 0]
    g.obs["lab"] = pd.Categorical(rng.choice(["IgM", "IgG", "IgA"], g.n_obs))
    g.obs["share"] = x / x.max() + rng.normal(0, 0.1, g.n_obs) if gradient else rng.uniform(0, 1, g.n_obs)
    return g


def test_nhood_enrichment_equals_squidpy_and_is_deterministic():
    """Null model: squidpy's random relabelling of units over the fixed graph.

    References: Palla et al. 2022, doi:10.1038/s41592-021-01358-2.
    """
    g = _random_labels(0)
    with pytest.warns(DepthNullWarning):
        res = sqx.nhood_enrichment(g, cluster_key="lab", n_neighbors=6, n_perms=200, rng=3)
    direct = g.copy()
    sq.gr.spatial_neighbors(direct, coord_type="generic", n_neighs=6)
    seed = res.params["seed"]
    kw = {**sqx._seed_kw(sq.gr.nhood_enrichment, seed), **sqx._quiet_kw(sq.gr.nhood_enrichment)}
    z, count = sq.gr.nhood_enrichment(direct, cluster_key="lab", n_perms=200, copy=True, **kw)[:2]
    iu = np.triu_indices(3)
    assert np.allclose(res.table.z.to_numpy(), np.asarray(z)[iu])
    assert np.array_equal(res.table.statistic.to_numpy(), np.asarray(count, dtype=float)[iu])
    with pytest.warns(DepthNullWarning):
        again = sqx.nhood_enrichment(g, cluster_key="lab", n_neighbors=6, n_perms=200, rng=3)
    pd.testing.assert_frame_equal(res.table, again.table)
    assert set(res.table.columns) >= {"cluster_a", "cluster_b", "statistic", "z", "p_norm", "q_norm", "n_perm"}
    assert "p_value" not in res.table.columns  # a normal approximation, not a permutation p-value
    from scipy import stats

    assert np.allclose(res.table.p_norm, 2 * stats.norm.sf(np.abs(res.table.z)))


def test_spatial_autocorr_calibration_and_power():
    """Null model: values permuted over units (squidpy), positions and graph fixed; power: an x gradient.

    References: Moran 1950, doi:10.2307/2332142; Palla et al. 2022, doi:10.1038/s41592-021-01358-2.
    """
    rng = np.random.default_rng(5)
    g = _gex(16)
    keys = [f"r{k}" for k in range(40)]
    for k in keys:
        g.obs[k] = rng.uniform(0, 1, g.n_obs)
    with pytest.warns(DepthNullWarning):
        null = sqx.spatial_autocorr(g, keys=keys, n_perms=199, rng=1)
    alpha, n = 0.05, len(keys)
    assert (null.table.p_value < alpha).mean() <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n)
    gr = _random_labels(2, n_side=16, gradient=True)
    with pytest.warns(DepthNullWarning):
        pw = sqx.spatial_autocorr(gr, keys=["share", "lab"], n_perms=199, rng=1)
    row = pw.table.set_index("label").loc["share"]
    assert row.statistic > 0.5 and row.p_value <= 0.01
    assert {"lab=IgA", "lab=IgG", "lab=IgM"} <= set(pw.table.label)


def test_co_occurrence_equals_squidpy():
    g = _random_labels(4, n_side=10)
    with pytest.warns(DepthNullWarning):
        res = sqx.co_occurrence(g, cluster_key="lab", distances_um=(0, 30, 60))
    occ, _ = sq.gr.co_occurrence(g.copy(), cluster_key="lab", interval=np.array([0.0, 30.0, 60.0]), copy=True)
    cats = list(g.obs["lab"].cat.categories)
    t = res.table.set_index(["cluster", "neighbour", "distance_um"])
    for c in range(3):
        for i in range(3):
            assert t.loc[(cats[c], cats[i], 60.0), "statistic"] == pytest.approx(occ[i, c, 1])


def test_missing_labels_dropped_or_filled():
    air, g = _toy()
    out = sqx.receptor_labels(air, g)
    with pytest.warns(DepthNullWarning):
        dropped = sqx.nhood_enrichment(out, cluster_key="air_switched", n_neighbors=2, n_perms=50, rng=0)
    assert dropped.summary["n_units"] == 3
    with pytest.warns(DepthNullWarning):
        filled = sqx.nhood_enrichment(out, cluster_key="air_switched", missing="none", n_neighbors=4, n_perms=50, rng=0)
    assert filled.summary["n_units"] == out.n_obs
    assert "none" in set(filled.table.cluster_a) | set(filled.table.cluster_b)


def test_missing_squidpy_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "squidpy", None)
    with pytest.raises(ImportError, match=r"spaclone\[squidpy\]"):
        sqx.nhood_enrichment(_random_labels(0, n_side=4), cluster_key="lab")
