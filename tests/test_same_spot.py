import warnings
from fractions import Fraction

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from spaclone.qc import GexLookup, located_fraction, same_spot, shift_test
from spaclone.qc.same_spot import ZeroChanceWarning, poisson_binomial_sf


def _chip(seed=0, n=120, n_rec=600, rho=0.0, p_det=0.5, lam=(0.4, 1.2), q=(0.01, 0.06), tile_px=20, sink=False):
    """Synthetic Stereo-seq-like chip: n x n DNBs (0.5 um), GEX molecules i.i.d. within tiles; IGK singletons."""
    rng = np.random.default_rng(seed)
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    tile = (ix // tile_px) * 10_000 + iy // tile_px
    tiles, inv = np.unique(tile, return_inverse=True)
    lam_s = rng.uniform(*lam, len(tiles))[inv]
    q_s = rng.uniform(*q, len(tiles))[inv]
    m = rng.poisson(lam_s)
    k = rng.binomial(m, q_s)
    occ = np.flatnonzero(m > 0)
    recs = rng.choice(occ, n_rec, replace=False)
    genuine = rng.random(n_rec) < rho
    detected = genuine & (rng.random(n_rec) < p_det)
    k[recs[detected]] += 1
    m[recs[detected]] += 1
    rows = []
    for gene, cnt in (("Igkc", k), ("Other", m - k)):
        nz = np.flatnonzero(cnt > 0)
        rows.append(pd.DataFrame({"x_um": ix[nz] * 0.5, "y_um": iy[nz] * 0.5, "gene": gene, "count": cnt[nz]}))
    long = pd.concat(rows, ignore_index=True)
    air = pd.DataFrame(
        {
            "spot_id": [f"{a}_{b}" for a, b in zip(ix[recs], iy[recs], strict=True)],
            "x_um": ix[recs] * 0.5,
            "y_um": iy[recs] * 0.5,
            "locus": "IGK",
            "clone_id": [f"c{i}" for i in range(n_rec)],
            "consensus_count": rng.integers(1, 20, n_rec).astype(float),
        }
    )
    if sink:  # one barcode sink with 30 clones on a DNB whose GEX holds Igkc
        j = int(np.flatnonzero(k > 0)[0])
        extra = pd.DataFrame(
            {
                "spot_id": f"{ix[j]}_{iy[j]}",
                "x_um": ix[j] * 0.5,
                "y_um": iy[j] * 0.5,
                "locus": "IGK",
                "clone_id": [f"s{i}" for i in range(30)],
                "consensus_count": 1.0,
            }
        )
        air = pd.concat([air, extra], ignore_index=True)
    gex = GexLookup.from_arrays(long.x_um, long.y_um, long.gene, long["count"], spot_um=0.5, genes=["Igkc"])
    return gex, air, {"genuine": genuine, "detected": detected, "long": long}


GENES = {"IGK": ["Igkc"]}


# ------------------------------------------------------------------------------------------------- GexLookup


def test_lookup_known_answer():
    # 2 x 2 DNB tile (tile_um = 1.0) at the origin plus one DNB in the next tile
    long = pd.DataFrame(
        {
            "x_um": [0.0, 0.0, 0.5, 0.5, 0.0, 1.0],
            "y_um": [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            "gene": ["Igkc", "Other", "Other", "Igkc", "Other", "Igkc"],
            "count": [1, 2, 1, 2, 3, 1],
        }
    )
    gex = GexLookup.from_arrays(long.x_um, long.y_um, long.gene, long["count"], spot_um=0.5, genes=["Igkc"])
    assert gex.n_spots == 5
    m, own = gex.spot_counts([0.0, 0.5, 0.0, 2.0], [0.0, 0.5, 0.5, 2.0], ["Igkc"])
    assert m.tolist() == [3, 2, 3, 0] and own.tolist() == [1, 2, 0, 0]
    # tile at origin: T = 3 + 1 + 2 + 3 = 9 molecules, S = 1 + 2 = 3 Igkc
    ev = gex.evaluate([0.0, 0.0], [0.0, 0.5], ["Igkc"], tile_um=1.0, chance="binomial")
    np.testing.assert_allclose(ev["q"], [3 / 9, 3 / 9])
    np.testing.assert_allclose(ev["chance"], [1 - (6 / 9) ** 3, 1 - (6 / 9) ** 3])
    ev = gex.evaluate([0.0, 0.0], [0.0, 0.5], ["Igkc"], tile_um=1.0, chance="binomial", leave_own_out=True)
    np.testing.assert_allclose(ev["q"], [2 / 6, 3 / 6])
    ev = gex.evaluate([0.0, 0.0], [0.0, 0.5], ["Igkc"], tile_um=1.0, chance="conditional")
    hyper = 1 - stats.hypergeom(9, 3, 3).pmf(0)
    np.testing.assert_allclose(ev["chance"], [hyper, hyper])
    ev = gex.evaluate([0.0], [0.5], ["Igkc"], tile_um=1.0, chance="conditional", leave_own_out=True)
    np.testing.assert_allclose(ev["chance"], [1 - stats.hypergeom(6, 3, 3).pmf(0)])
    # a lone DNB: leave-own-out leaves nothing in its tile
    ev = gex.evaluate([1.0], [0.0], ["Igkc"], tile_um=1.0, chance="conditional", leave_own_out=True)
    assert np.isnan(ev["chance"][0])
    ev = gex.evaluate([1.0, 3.0], [0.0, 3.0], ["Igkc"], tile_um=1.0)
    assert ev["chance"].tolist() == [1.0, 0.0]  # every molecule of the tile is Igkc; empty spot
    with pytest.raises(ValueError, match="multiple"):
        gex.evaluate([0.0], [0.0], ["Igkc"], tile_um=0.7)
    with pytest.raises(KeyError, match="not loaded"):
        gex.evaluate([0.0], [0.0], ["Ighm"])


def test_hypergeometric_chance_matches_scipy_for_large_counts():
    from spaclone.qc.same_spot import _log_hyper_none

    t = np.array([50.0, 900.0, 5000.0, 30.0, 400.0])
    s = np.array([3.0, 20.0, 70.0, 29.0, 0.0])
    m = np.array([4, 70, 120, 2, 5])
    ref = np.array([stats.hypergeom(int(a), int(b), int(c)).pmf(0) for a, b, c in zip(t, s, m, strict=True)])
    np.testing.assert_allclose(np.exp(_log_hyper_none(t, s, m)), ref, rtol=1e-9, atol=1e-300)


def test_from_anndata_matches_long_format():
    rng = np.random.default_rng(3)
    xy = np.array([(i, j) for i in range(6) for j in range(5)], dtype=float) * 2.0 + 1.0  # bin centres, 2 um
    counts = rng.poisson(1.5, (len(xy), 3)).astype(float)
    adata = ad.AnnData(sparse.csr_matrix(counts), var=pd.DataFrame(index=["Igkc", "Ighm", "Actb"]))
    adata.obsm["spatial"] = xy
    a = GexLookup.from_anndata(adata, spot_um=2.0)
    assert set(a.genes) == {"Igkc", "Ighm"}
    rr, cc = np.nonzero(counts)
    names = np.array(["Igkc", "Ighm", "Actb"])
    b = GexLookup.from_arrays(
        xy[rr, 0], xy[rr, 1], names[cc], counts[rr, cc], spot_um=2.0, origin_um=(1.0, 1.0), genes=["Igkc", "Ighm"]
    )
    ea = a.evaluate(xy[:, 0], xy[:, 1], ["Igkc"], tile_um=4.0)
    eb = b.evaluate(xy[:, 0], xy[:, 1], ["Igkc"], tile_um=4.0)
    for key in ("m", "own", "tile_total", "tile_set"):
        np.testing.assert_array_equal(ea[key], eb[key])
    adata.obsm["spatial"] = xy + 0.3
    adata.obsm["spatial"][0] += 0.4
    with pytest.raises(ValueError, match="not on a grid"):
        GexLookup.from_anndata(adata, spot_um=2.0)


# --------------------------------------------------------------------------------------------- Poisson-binomial


def test_poisson_binomial_tail_is_exact():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.001, 0.3, 18)
    pmf = [Fraction(1)]
    for q in (Fraction(float(v)) for v in p):
        pmf = [a * (1 - q) + b * q for a, b in zip([*pmf, Fraction(0)], [Fraction(0), *pmf], strict=True)]
    for k in range(0, 20):
        exact = float(sum(pmf[k:])) if k <= 18 else 0.0
        assert poisson_binomial_sf(p, k) == pytest.approx(exact, rel=1e-10, abs=0 if exact else 1e-300)
    for n, pp, k in [(4000, 0.02, 80), (4000, 0.02, 260), (20000, 0.03, 1000), (300, 0.5, 290)]:
        assert poisson_binomial_sf(np.full(n, pp), k) == pytest.approx(stats.binom.sf(k - 1, n, pp), rel=1e-8)
    # degenerate probabilities and bounds
    assert poisson_binomial_sf([1.0, 1.0, 0.0, 0.5], 2) == 1.0
    assert poisson_binomial_sf([1.0, 1.0, 0.0, 0.5], 3) == pytest.approx(0.5)
    assert poisson_binomial_sf([1.0, 1.0, 0.0, 0.5], 4) == 0.0
    assert poisson_binomial_sf([0.2], 0) == 1.0
    assert poisson_binomial_sf(np.full(2000, 0.3), 1900) > 0  # floored, never 0
    assert np.isfinite(poisson_binomial_sf(np.full(2000, 0.3), 1900, log=True))


# ------------------------------------------------------------------------------------------- located fraction


def _legacy_real_fraction(n_occupied, concordance, chance, p_det=0.34, alpha=0.05):
    """Copy of the archived spaclone 0.x real_fraction (03 concordance.py)."""
    k = int(round(concordance * n_occupied))
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n_occupied - k)
    lo = stats.beta.ppf(alpha / 2, k, n_occupied - k + 1) if k > 0 else 0.0
    d = p_det * (1 - chance)
    return {
        "rho_hat": max(0.0, (concordance - chance) / d),
        "rho_lower": max(0.0, (lo - chance) / d),
        "rho_upper": max(0.0, (hi - chance) / d),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plugin_estimate_reproduces_archived_formula(seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.005, 0.06, 4000)
    y = rng.random(4000) < c * (1.0 + 0.5 * seed)
    for p_det in (1.0, 0.5, 0.1):
        ref = _legacy_real_fraction(len(y), y.mean(), c.mean(), p_det=p_det)
        got = located_fraction(y, c, p_det=None if p_det == 1.0 else p_det, method="plugin")
        key = "rho_pdet" if p_det == 1.0 else "rho"
        assert got[key] == pytest.approx(ref["rho_hat"], rel=1e-12, abs=1e-15)
        assert got[f"{key}_lo"] == pytest.approx(ref["rho_lower"], rel=1e-12, abs=1e-15)
        assert got[f"{key}_hi"] == pytest.approx(ref["rho_upper"], rel=1e-12, abs=1e-15)
    assert got["hits"] == int(y.sum()) and got["expected"] == pytest.approx(c.sum())


def test_boundary_deficit_and_p_det_propagation():
    rng = np.random.default_rng(5)
    c = rng.uniform(0.01, 0.05, 5000)
    y = np.zeros(5000, bool)
    y[rng.choice(5000, int(round(c.sum() * 0.97)), replace=False, p=c / c.sum())] = True  # hits just below chance
    for method in ("ipw", "exact", "ml"):
        r = located_fraction(y, c, method=method)
        assert r["rho_pdet"] == 0.0 and r["rho_pdet_lo"] == 0.0 and 0 < r["rho_pdet_hi"] < 0.01
        assert r["p_value"] > 0.5 and not r["deficit"]
        fixed = located_fraction(y, c, p_det=0.5, method=method)
        assert fixed["rho_hi"] == pytest.approx(r["rho_pdet_hi"] / 0.5)
        wide = located_fraction(y, c, p_det={"p_det": 0.5, "se": 0.1}, method=method)
        assert wide["rho_hi"] > fixed["rho_hi"] and wide["rho_lo"] == 0.0
        assert located_fraction(y, c, p_det=(0.5, 0.1), method=method)["rho_hi"] == pytest.approx(wide["rho_hi"])
    # far fewer hits than chance: the chance model over-states the chance, so no bound is given (review B1)
    few = rng.random(5000) < c * 0.5
    for method in ("ipw", "exact", "ml"):
        d = located_fraction(few, c, method=method, p_det=0.5)
        assert d["deficit"] and d["deficit_p"] < 0.025 and not d["bound_reliable"]
        assert np.isnan(d["rho_pdet"]) and np.isnan(d["rho_pdet_hi"]) and np.isnan(d["rho_hi"])
        assert "over-states the chance" in d["bound_note"]
    leg = located_fraction(few, c, method="plugin")  # the archive's numbers are kept, but flagged
    assert leg["deficit"] and leg["rho_pdet_hi"] == 0.0 and "not reliable" in leg["bound_note"]
    with pytest.raises(ValueError):
        located_fraction(y, c, p_det=1.5)
    empty = located_fraction(np.zeros(0, bool), np.zeros(0))
    assert empty["n"] == 0 and np.isnan(empty["rho_pdet"])


def test_exact_interval_matches_clopper_pearson_and_ipw_matches_wilson_for_constant_chance():
    # with c_i = 0 the model is binomial(theta): the exact interval is Clopper-Pearson, the IPW score interval Wilson
    y = np.r_[np.ones(17, bool), np.zeros(483, bool)]
    r = located_fraction(y, np.zeros(500), method="exact")
    lo, hi = stats.beta.ppf(0.025, 17, 484), stats.beta.ppf(0.975, 18, 483)
    assert r["rho_pdet_lo"] == pytest.approx(lo, rel=1e-5) and r["rho_pdet_hi"] == pytest.approx(hi, rel=1e-5)
    assert r["rho_pdet"] == pytest.approx(17 / 500)
    w = located_fraction(y, np.zeros(500))
    z, p, n = stats.norm.ppf(0.975), 17 / 500, 500
    centre, half = (
        (p + z**2 / (2 * n)) / (1 + z**2 / n),
        z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)),
    )
    assert w["rho_pdet"] == pytest.approx(p) and w["rho_pdet_lo"] == pytest.approx(centre - half, rel=1e-6)
    assert w["rho_pdet_hi"] == pytest.approx(centre + half, rel=1e-6)


@pytest.mark.parametrize("method", ["ipw", "exact"])
def test_interval_coverage_and_estimate(method):
    # calibration of the estimator: intervals cover the true theta at least at about the nominal rate
    rng = np.random.default_rng(11)
    theta, cover, est = 0.02, 0, []
    for _ in range(40):
        c = rng.beta(1.0, 40.0, 1500)
        y = rng.random(1500) < c + theta * (1 - c)
        r = located_fraction(y, c, method=method, blocks=rng.integers(0, 30, 1500))
        cover += r["rho_pdet_lo"] <= theta <= r["rho_pdet_hi"]
        est.append(r["rho_pdet"])
    assert cover / 40 >= 0.85
    assert abs(np.mean(est) - theta) < 0.004


def test_p_value_is_calibrated_under_the_null():
    rng = np.random.default_rng(7)
    p = []
    for _ in range(400):
        c = rng.beta(0.8, 30.0, 800)
        y = rng.random(800) < c
        p.append(poisson_binomial_sf(c, int(y.sum())))
    p = np.asarray(p)
    for a in (0.01, 0.05, 0.1):
        assert (p <= a).mean() <= a + 3 * np.sqrt(a * (1 - a) / len(p))


# ------------------------------------------------------------------------------------------------- same_spot


def test_same_spot_calibration_on_null_chips():
    pvals, ratios = [], []
    for seed in range(40):
        gex, air, _ = _chip(seed=seed, n=80, n_rec=500)
        t = same_spot(air, gex, constant_genes=GENES, tile_um=10.0).table
        pvals.append(float(t.p_value.iloc[0]))
        ratios.append(float(t.ratio.iloc[0]))
    pvals = np.asarray(pvals)
    assert (pvals <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(pvals))
    assert abs(np.mean(ratios) - 1.0) < 0.06


def test_same_spot_power_and_estimate_on_planted_signal():
    gex, air, truth = _chip(seed=3, n=160, n_rec=3000, rho=0.3, p_det=0.5)
    res = same_spot(air, gex, constant_genes=GENES, tile_um=10.0, p_det=0.5)
    row = res.table.iloc[0]
    assert row.p_value < 1e-10 and row.ratio > 2
    # the moment estimate is close to the truth (theta = 0.15); a detected own transcript raises its spot's chance,
    # which biases it down by about the chance itself (~5% here)
    assert 0.125 <= row.rho_pdet <= 0.17 and row.rho_pdet_lo < 0.16 and row.rho_pdet_hi > 0.14
    assert row.rho_lo < 0.32 and row.rho_hi > 0.28
    ml = same_spot(air, gex, constant_genes=GENES, tile_um=10.0, estimator="ml").table.iloc[0]
    assert ml.rho_pdet < row.rho_pdet  # maximum likelihood is pulled further down
    assert row.statistic == row.hits
    assert set(res.table.columns) >= {"statistic", "expected", "null_sd", "z", "p_value", "q_value"}
    assert "Null" not in res.params["null"] and "rho = 0" in res.params["null"]


def test_sinks_count_once_and_naive_contrast():
    gex, air, _ = _chip(seed=4, n=100, n_rec=800, sink=True)
    res = same_spot(air, gex, constant_genes=GENES, tile_um=10.0, context_um=None)  # every record gets a chance
    row = res.table.iloc[0]
    rec = res.records
    sink = rec[rec.n_clones == 30]
    assert len(sink) == 1 and bool(sink.hit.iloc[0]) and not bool(sink.singleton.iloc[0])
    assert row.n_spots == 800 and row.n_no_chance == 0  # the sink spot is not a singleton
    assert row.naive_records == 830 and row.naive_hits >= row.hits - 0 + 30 - 30  # sink counted 30 times
    assert row.naive_hits == int((rec.n_records * rec.hit).sum())
    bm = res.by_multiplicity
    assert set(bm.level) == {"spot", "record"}
    top = bm[(bm.level == "record") & (bm.multiplicity == ">=21")]
    assert int(top.n.iloc[0]) == 30 and int(top.hits.iloc[0]) == 30
    allspots = same_spot(air, gex, constant_genes=GENES, tile_um=10.0, singleton_only=False, context_um=None)
    assert allspots.table.iloc[0].n_spots == 801


def _legacy_b2(long, units, tile=20, chain="IGK", genes=("Igkc",)):
    """Inline copy of the archived MIDAwareConcordance + B2 singleton-DNB records (DNB integer coordinates)."""
    km = np.int64(30000)
    x = np.rint(long.x_um.to_numpy() / 0.5).astype(np.int64)
    y = np.rint(long.y_um.to_numpy() / 0.5).astype(np.int64)
    key = x * km + y
    mid = long["count"].to_numpy().astype(np.int64)
    occ_keys, inv = np.unique(key, return_inverse=True)
    occ_mid = np.bincount(inv, weights=mid)
    tk = (x // tile) * 100000 + (y // tile)
    tile_tot = pd.Series(mid).groupby(tk).sum()
    rows = np.flatnonzero(np.isin(long.gene.to_numpy(), list(genes)))
    set_keys = np.unique(key[rows])
    tile_g = pd.Series(mid[rows]).groupby(tk[rows]).sum()
    ux = np.rint(units.x_um.to_numpy() / 0.5).astype(np.int64)
    uy = np.rint(units.y_um.to_numpy() / 0.5).astype(np.int64)
    k = ux * km + uy
    i = np.clip(np.searchsorted(set_keys, k), 0, len(set_keys) - 1)
    hit = set_keys[i] == k
    t = (ux // tile) * 100000 + (uy // tile)
    q = tile_g.reindex(t).fillna(0).to_numpy(float) / tile_tot.reindex(t).to_numpy(float)
    j = np.clip(np.searchsorted(occ_keys, k), 0, len(occ_keys) - 1)
    m = np.where(occ_keys[j] == k, occ_mid[j], 0.0)
    chance = 1 - (1 - np.nan_to_num(q)) ** m
    u = units.assign(x=ux, y=uy, hit=hit, chance=chance, m=m)
    u["mult_all"] = u.groupby(["x", "y"]).clone_id.transform("size")
    u["mult_chain"] = u.groupby(["x", "y", "locus"]).clone_id.transform("size")
    occ = u[u.m > 0]
    d = (
        occ.groupby(["locus", "x", "y"])
        .agg(
            mult_chain=("mult_chain", "first"),
            mult_all=("mult_all", "first"),
            hit=("hit", "first"),
            chance=("chance", "first"),
        )
        .reset_index()
    )
    single = d[(d.mult_chain == 1) & (d.mult_all < 20)]
    return occ, single


def test_legacy_mode_reproduces_archived_b2_exactly():
    gex, air, truth = _chip(seed=8, n=100, n_rec=900, rho=0.2, sink=True)
    # a few multi-clone DNBs
    extra = air.iloc[:40].copy()
    extra["clone_id"] = [f"m{i}" for i in range(40)]
    air = pd.concat([air, extra], ignore_index=True)
    occ, single = _legacy_b2(truth["long"], air, tile=50)
    res = same_spot(air, gex, constant_genes=GENES, tile_um=25.0, legacy=True, p_det=0.5)
    row = res.table.iloc[0]
    assert row.n_spots == len(single)
    assert row.hits == int(single.hit.sum())
    assert row.expected == pytest.approx(single.chance.sum(), rel=1e-12)
    ref = _legacy_real_fraction(len(single), single.hit.mean(), single.chance.mean(), p_det=1.0)
    assert row.rho_pdet_hi == pytest.approx(ref["rho_upper"], rel=1e-10)
    ref = _legacy_real_fraction(len(single), single.hit.mean(), single.chance.mean(), p_det=0.5)
    assert row.rho_hi == pytest.approx(ref["rho_upper"], rel=1e-10)
    assert row.naive_records == len(occ) and row.naive_hits == int(occ.hit.sum())
    assert row.naive_expected == pytest.approx(occ.chance.sum(), rel=1e-12)
    assert res.params["chance"] == "binomial" and res.params["estimator"] == "plugin"
    assert res.params["variance"] == "binomial" and np.isnan(row.p_value_block)
    # the 1.0.0.dev0 default is reachable with explicit settings and unchanged by the new columns
    dev0 = same_spot(air, gex, constant_genes=GENES, tile_um=25.0, chance="conditional", estimator="exact",
                     variance="binomial").table.iloc[0]  # fmt: skip
    rec = same_spot(air, gex, constant_genes=GENES, tile_um=25.0, chance="conditional").records
    pick = rec[rec.tested & rec.occupied & rec.singleton]
    ref = located_fraction(pick.hit.to_numpy(), pick.chance.to_numpy(), method="exact", variance="binomial")
    assert dev0.rho_pdet_hi == pytest.approx(ref["rho_pdet_hi"], rel=1e-12) and dev0.n_blocks == 0


def test_zones_auto_genes_and_errors():
    gex, air, _ = _chip(seed=9, n=80, n_rec=400)
    air["zone"] = np.where(air.x_um < 20, "left", "right")
    res = same_spot(air, gex, tile_um=10.0, zones="zone")  # constant genes picked from the mouse table
    assert res.params["constant_genes"] == {"IGK": ["Igkc"]} and res.params["constant_genes_source"] == "mouse"
    assert list(res.table.zone) == ["left", "right"]
    assert res.table.n_spots.sum() == 400
    bad = air.copy()
    bad.loc[bad.index[1], "spot_id"] = bad.spot_id.iloc[0]
    bad.loc[bad.index[1], "zone"] = "left" if bad.zone.iloc[0] == "right" else "right"
    with pytest.raises(ValueError, match="varies inside a spot"):
        same_spot(bad, gex, tile_um=10.0, zones="zone")
    with pytest.raises(ValueError, match="clone_id"):
        same_spot(air.drop(columns="clone_id"), gex)
    with pytest.raises(ValueError, match="chance"):
        same_spot(air, gex, chance="poisson")
    human = air.assign(locus="TRB")
    with pytest.raises(ValueError, match="constant_genes"):
        same_spot(human, gex)


def test_zero_chance_hits_warn_with_leave_own_out():
    gex, air, _ = _chip(seed=2, n=120, n_rec=500, lam=(0.05, 0.1), q=(0.02, 0.04), tile_px=8)
    with pytest.warns(ZeroChanceWarning):
        res = same_spot(air, gex, constant_genes=GENES, tile_um=4.0, chance="binomial", leave_own_out=True)
    assert res.table.n_zero_chance_hits.iloc[0] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = same_spot(air, gex, constant_genes=GENES, tile_um=4.0, chance="conditional")
        same_spot(air, gex, constant_genes=GENES, tile_um=4.0)  # the empirical default never warns
    assert ok.table.n_zero_chance_hits.iloc[0] == 0


def test_row_order_invariance_and_shift_test():
    gex, air, _ = _chip(seed=6, n=120, n_rec=1500, rho=0.5, p_det=0.6)
    a = same_spot(air, gex, constant_genes=GENES, tile_um=10.0).table
    b = same_spot(air.sample(frac=1.0, random_state=1), gex, constant_genes=GENES, tile_um=10.0).table
    pd.testing.assert_frame_equal(a, b)
    sh = shift_test(air, gex, constant_genes=GENES, tile_um=10.0, max_shift=2)
    t = sh.table.set_index(["dx", "dy"])
    assert t.loc[(0, 0), "ratio"] > t.drop(index=(0, 0)).ratio.max()
    assert sh.summary["p_value[IGK]"] == pytest.approx(1 / 25)
    s1 = shift_test(air, gex, constant_genes=GENES, tile_um=10.0, max_shift=1, max_records=300, rng=4)
    s2 = shift_test(air, gex, constant_genes=GENES, tile_um=10.0, max_shift=1, max_records=300, rng=4)
    s3 = shift_test(air, gex, constant_genes=GENES, tile_um=10.0, max_shift=1, max_records=300, rng=5)
    pd.testing.assert_frame_equal(s1.table, s2.table)
    assert not s1.table.hits.equals(s3.table.hits)


def test_multiplicity_plot_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spaclone.pl import same_spot_multiplicity

    gex, air, _ = _chip(seed=4, n=100, n_rec=800, sink=True)
    air = pd.concat([air, air.iloc[:60].assign(clone_id=[f"z{i}" for i in range(60)], locus="IGH")], ignore_index=True)
    res = same_spot(air, gex, constant_genes={"IGK": ["Igkc"], "IGH": ["Igkc"]}, tile_um=10.0)
    ax = same_spot_multiplicity(res, p_det=0.5)
    assert ax.get_ylabel().startswith("hits") and len(ax.get_legend().get_texts()) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3-5", "6-20", ">=21"]
    fig, ax2 = plt.subplots()
    assert same_spot_multiplicity(res.by_multiplicity, level="record", loci=["IGK"], ax=ax2, style="dark") is ax2
    with pytest.raises(ValueError):
        same_spot_multiplicity(res, zone="nowhere")
    plt.close("all")


def test_multiplicity_plot_is_silent_on_classes_without_an_expectation():
    """A multiplicity class with no chance expectation (0 expected hits) has NaN intervals: no point, no warning."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spaclone.pl import same_spot_multiplicity

    cats = pd.Categorical(["1", "2", "3-5"], categories=["1", "2", "3-5"])
    bm = pd.DataFrame({"level": "spot", "locus": "IGK", "multiplicity": cats, "n": [50.0, 10.0, 4.0],
                       "hits": [5.0, 1.0, 0.0], "expected": [4.0, 0.5, 0.0]})  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = same_spot_multiplicity(bm, p_det=0.5)
    y = ax.lines[0].get_ydata() if ax.lines else ax.containers[0][0].get_ydata()
    assert np.isfinite(np.asarray(y, dtype=float)[:2]).all() and np.isnan(np.asarray(y, dtype=float)[2])
    plt.close("all")


# ------------------------------------------------------------------------------- empirical chance (review B1)


def _tiny():
    """0.5-um DNBs. Tile (0, 0) of 1 um: A (m 1, Igkc), B (m 1), C (m 1, the receptor spot), D (m 2, Igkc);
    tile (1, 0): E (m 1), F (m 1, Igkc), G (m 3)."""
    long = pd.DataFrame(
        [(0.0, 0.0, "Igkc", 1), (0.0, 0.5, "Other", 1), (0.5, 0.0, "Other", 1), (0.5, 0.5, "Igkc", 1),
         (0.5, 0.5, "Other", 1), (1.0, 0.0, "Other", 1), (1.5, 0.0, "Igkc", 1), (1.0, 0.5, "Other", 3)],
        columns=["x_um", "y_um", "gene", "count"],
    )  # fmt: skip
    return GexLookup.from_arrays(long.x_um, long.y_um, long.gene, long["count"], spot_um=0.5, genes=["Igkc"])


def test_empirical_chance_known_answer():
    gex = _tiny()
    excl = ([0.5], [0.0])  # C carries a record of the locus: never a reference spot
    ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, min_reference=2)
    assert ev["m"][0] == 1 and not ev["hit"][0]
    assert ev["chance"][0] == pytest.approx(0.5) and ev["ref_n"][0] == 2 and ev["pool_um"][0] == 0.5  # A, B
    ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, min_reference=3)
    assert ev["chance"][0] == pytest.approx(0.5) and ev["ref_n"][0] == 4 and ev["pool_um"][0] == 1.5  # + E, F
    # one class for m in [1, 3): A, B, D; cDNA weights count D twice
    for weights, want in (("cdna", 3 / 4), ("spots", 2 / 3)):
        ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, m_bins=(1, 3, np.inf),
                                    min_reference=1, weights=weights)  # fmt: skip
        assert ev["chance"][0] == pytest.approx(want)
    # D (m 2) with C and D excluded: no reference spot with m = 2 anywhere -> no chance model
    ev = gex.evaluate_empirical([0.5], [0.5], ["Igkc"], exclude=([0.5, 0.5], [0.0, 0.5]), tile_um=1.0,
                                min_reference=2)  # fmt: skip
    assert np.isnan(ev["chance"][0]) and ev["hit"][0] and ev["own"][0] == 1
    empty = gex.evaluate_empirical([3.0], [3.0], ["Igkc"], tile_um=1.0)
    assert empty["m"][0] == 0 and empty["chance"][0] == 0.0
    with pytest.raises(ValueError, match="m_bins"):
        gex.evaluate_empirical([0.5], [0.0], ["Igkc"], m_bins=(2, 5, np.inf), tile_um=1.0)
    with pytest.raises(KeyError, match="label set"):
        gex.evaluate_empirical([0.5], [0.0], ["Igkc"], strata="cell_type", tile_um=1.0)


def test_labels_strata_and_context_known_answer():
    gex = _tiny()
    xs, ys = [0.0, 0.0, 0.5, 0.5, 1.0, 1.5, 1.0], [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5]
    gex.add_labels("cell", xs, ys, ["B", "other", "B", "B", "other", "B", "other"])
    assert gex.labels == ("cell",)
    excl = ([0.5], [0.0])
    ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, strata="cell", min_reference=1)
    assert ev["chance"][0] == pytest.approx(1.0)  # the only B spot with m 1 in the tile is A (a hit)
    ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, strata="cell", min_reference=2)
    assert ev["chance"][0] == pytest.approx(1.0) and ev["ref_n"][0] == 2  # A and F
    # bin labels: every DNB takes the label of its 1-um bin
    gex.add_labels("bin", [0.0, 1.0], [0.0, 0.0], ["white", "red"], bin_um=1.0)
    lab, levels = gex._labels["bin"]
    x, y, _ = gex.spot_arrays()
    assert [levels[k] for k in lab] == ["white" if xi < 1.0 else "red" for xi in x]
    # constant-gene molecules on the other DNBs of the 3 x 3 window (0.5 um), occupied spots in key order
    np.testing.assert_array_equal(gex.context_counts(["Igkc"], 0.5), [1, 2, 2, 1, 2, 2, 0])
    ev = gex.evaluate_empirical([0.5], [0.0], ["Igkc"], exclude=excl, tile_um=1.0, context_um=0.5,
                                context_bins=(0, 2, np.inf), min_reference=1)  # fmt: skip
    assert ev["chance"][0] == pytest.approx(0.0)  # same class (context >= 2) and m 1 in the tile: B only
    rng = np.random.default_rng(3)
    xy = np.array([(i, j) for i in range(6) for j in range(5)], dtype=float) * 2.0
    adata = ad.AnnData(sparse.csr_matrix(rng.poisson(1.5, (len(xy), 2)).astype(float)),
                       var=pd.DataFrame(index=["Igkc", "Actb"]))  # fmt: skip
    adata.obsm["spatial"] = xy
    adata.obs["cell_type"] = np.where(xy[:, 0] < 6, "B", "T")
    lk = GexLookup.from_anndata(adata, spot_um=2.0, labels="cell_type")
    lab, levels = lk._labels["cell_type"]
    lx, _, _ = lk.spot_arrays()
    assert levels == ["B", "T"] and ([levels[k] for k in lab] == np.where(lx < 6, "B", "T")).all()
    with pytest.raises(KeyError, match="adata.obs"):
        GexLookup.from_anndata(adata, spot_um=2.0, labels="zone")


def test_ipw_estimates_the_plain_mean_and_exact_a_weighted_one():
    # located molecules sit on spots rich in constant-region RNA (high chance): the moment estimate weights them by
    # (1 - c) and under-states the fraction located; the IPW estimate does not
    rng = np.random.default_rng(3)
    n = 20000
    c = np.where(rng.random(n) < 0.5, 0.05, 0.6)
    theta = np.where(c > 0.5, 0.4, 0.0)
    y = rng.random(n) < c + theta * (1 - c)
    plain, weighted = theta.mean(), (theta * (1 - c)).sum() / (1 - c).sum()
    ipw, ex = located_fraction(y, c), located_fraction(y, c, method="exact")
    assert abs(ipw["rho_pdet"] - plain) < 0.02 and ipw["rho_pdet_lo"] <= plain <= ipw["rho_pdet_hi"]
    assert abs(ex["rho_pdet"] - weighted) < 0.02 and ex["rho_pdet_hi"] < plain


def test_ipw_is_not_biased_by_the_own_transcript_raising_its_spot_chance():
    # a detected own transcript adds one molecule to its spot, and the chance rises with the molecules: the moment
    # estimate scores such a hit against a higher chance and is biased low; in the IPW estimate a hit counts 1
    rng = np.random.default_rng(4)
    n, rho, p = 30000, 0.4, 0.5

    def chance(m):
        return 1 - 0.8 ** np.asarray(m, float)

    m0 = 1 + rng.poisson(1.0, n)
    own = (rng.random(n) < rho) & (rng.random(n) < p)
    m = m0 + own
    y = own | (rng.random(n) < chance(m0))
    ipw, ex = located_fraction(y, chance(m)), located_fraction(y, chance(m), method="exact")
    assert abs(ipw["rho_pdet"] - rho * p) < 0.01 and ipw["rho_pdet_lo"] <= rho * p <= ipw["rho_pdet_hi"]
    assert ex["rho_pdet"] < rho * p - 0.02
    null = located_fraction(rng.random(n) < chance(m0), chance(m0))
    assert null["p_value"] > 0.01 and null["rho_pdet_lo"] == 0.0
    capped = located_fraction(y, np.where(np.arange(n) < 10, 0.95, chance(m)))
    assert capped["n_capped"] == 10


def test_block_variance_is_calibrated_under_overdispersion():
    """Null model: hits Bernoulli(c_b) with a block-level chance c_b around the analyst's chance c (over-dispersion);
    the exact Poisson-binomial test treats records as independent, the block test only blocks."""
    rng = np.random.default_rng(5)
    reps, n_b, per = 150, 40, 100
    blocks = np.repeat(np.arange(n_b), per)
    p_exact, p_block, phis = [], [], []
    for _ in range(reps):
        cb = rng.beta(2.0, 18.0, n_b)  # mean 0.1
        y = rng.random(n_b * per) < cb[blocks]
        r = located_fraction(y, np.full(n_b * per, 0.1), blocks=blocks)
        p_exact.append(r["p_value_exact"])
        p_block.append(r["p_value"])
        phis.append(r["phi"])
    p_exact, p_block = np.asarray(p_exact), np.asarray(p_block)
    assert (p_exact <= 0.05).mean() > 0.1  # records treated as independent: anti-conservative
    assert (p_block <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / reps)
    assert np.median(phis) > 3
    binom = located_fraction(y, np.full(n_b * per, 0.1), variance="binomial")
    assert binom["rho_pdet_hi"] < r["rho_pdet_hi"] and np.isnan(binom["phi"])


REAL = {"size_um": 400.0, "tissue_radius_um": 170.0, "follicle_width_um": 60.0, "t_zone_radius_um": 50.0}


def _real_chip(gf, seed, **kw):
    from spaclone import sim

    sa = sim.synthetic_chip(genuine_fraction=gf, rng=seed, **REAL, **kw)
    look = GexLookup.from_anndata(sa.gex, spot_um=2.0, labels="cell_type")
    return sa, look, sa.air.assign(zone=np.where(sa.air.in_tissue, "tissue", "off tissue"))


def test_deficit_suppresses_the_bound_and_the_empirical_chance_has_none():
    """The 1.0.0.dev0 tile chance over-states the chance of swapped molecules on the realistic chip (review B1): a
    deficit, which must not become "no signal, located <= 0%"."""
    sa, look, a = _real_chip(0.0, 1)
    dev0 = same_spot(a, look, tile_um=24.0, zones="zone", chance="conditional", estimator="exact",
                     variance="binomial").table  # fmt: skip
    new = same_spot(a, look, tile_um=24.0, zones="zone").table
    for t in (dev0, new):
        t.set_index(["locus", "zone"], inplace=True)
    for lc in ("IGH", "IGK"):
        old, now = dev0.loc[(lc, "tissue")], new.loc[(lc, "tissue")]
        assert old.ratio < 0.85 and old.deficit and np.isnan(old.rho_pdet_hi) and not old.bound_reliable
        assert 0.8 < now.ratio < 1.2 and not now.deficit and now.bound_reliable and 0 < now.rho_pdet_hi < 0.15
        assert now.p_value > 0.01 and now.n_blocks >= 10


def test_empirical_same_spot_power_strata_and_shift_test():
    sa, look, a = _real_chip(0.3, 2)
    res = same_spot(a, look, tile_um=24.0, zones="zone", strata="cell_type")
    t = res.table.set_index(["locus", "zone"])
    for lc in ("IGH", "IGK"):
        r = t.loc[(lc, "tissue")]
        assert r.p_value < 0.01 and r.ratio > 1.2 and 0.05 < r.rho_pdet < 0.3 and r.rho_pdet_lo > 0
    assert {"ref_n", "pool_um", "stratum", "block"} <= set(res.records.columns)
    assert res.params["strata"] == "cell_type" and "proportional to its GEX molecules" in res.params["null"]
    sh = shift_test(a, look, constant_genes={"IGK": ["Igkc"]}, tile_um=24.0, zones="zone", zone="tissue", max_shift=1,
                    strata="cell_type")  # fmt: skip
    assert sh.summary["ratio_at_0[IGK]"] == pytest.approx(t.loc[("IGK", "tissue"), "ratio"], rel=1e-12)
    assert np.isfinite(sh.table.ratio).all() and len(sh.table) == 9
    again = same_spot(a, look, tile_um=24.0, zones="zone", strata="cell_type").table
    pd.testing.assert_frame_equal(res.table, again)  # deterministic
    with pytest.raises(ValueError, match="empirical"):
        same_spot(a, look, chance="conditional", strata="cell_type")
    assert same_spot(a, look, tile_um=24.0, chance="conditional").params["context_um"] is None


@pytest.mark.slow
def test_empirical_same_spot_calibration_and_coverage_on_realistic_chips():
    """Null model: realistic chips with every molecule swapped onto a random cDNA molecule (swaps proportional to the
    spot's GEX molecules, constant-region RNA bursty and confined to its cells); false-positive rate <= alpha + tol.
    Coverage: at 30% genuine the interval (with cell-type strata) covers the occupancy-conditional theta."""
    pvals, deficits, cover = [], [], {"IGH": [], "IGK": []}
    for seed in range(20):
        for gf in (0.0, 0.3):
            sa, look, a = _real_chip(gf, 100 + seed)
            res = same_spot(a, look, tile_um=24.0, zones="zone", strata="cell_type" if gf else None)
            t = res.table[res.table.zone == "tissue"].set_index("locus")
            if gf == 0:
                pvals += t.p_value.tolist()
                deficits += t.deficit.tolist()
                continue
            rec = res.records
            g = sa.air[sa.air.truth == "genuine"]
            kgen, kseen = g.groupby(["spot_id", "locus"]).size(), g.groupby(["spot_id", "locus"]).own_seen.sum()
            for lc in cover:
                u = rec[(rec.locus == lc) & rec.tested & rec.occupied & rec.singleton & (rec.zone == "tissue")]
                u = u[np.isfinite(u.chance)]
                key = pd.MultiIndex.from_arrays([u.spot_id.astype(str), u.locus.astype(str)])
                k, seen = kgen.reindex(key).fillna(0).to_numpy(), kseen.reindex(key).fillna(0).to_numpy()
                th_i = np.where(k == 0, 0.0, np.where(u.m.to_numpy() - seen >= 1, 1 - 0.5**k, 1.0))
                cover[lc].append(t.loc[lc, "rho_pdet_lo"] <= th_i.mean() <= t.loc[lc, "rho_pdet_hi"])
    pvals = np.asarray(pvals)
    assert (pvals <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(pvals))
    assert np.mean(deficits) <= 0.05
    for lc, v in cover.items():
        assert np.mean(v) >= 0.85, lc
