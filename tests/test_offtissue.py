import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spaclone.io import signed_distance_map
from spaclone.qc import GexLookup, composition_on_off, density_profile, swap_rate, tissue_zones


def _grid(n=400, tissue_frac=0.5):
    """n x n DNB chip (0.5 um); tissue = the left part; distance grid of 2.5 um."""
    mask = np.zeros((n, n), dtype=bool)
    mask[:, : int(n * tissue_frac)] = True  # mask[y, x]
    return signed_distance_map(mask, pixel_um=0.5, downsample=5)


def test_tissue_zones_are_right_closed():
    d = np.array([-60, -50, -49.9, 0, 0.1, 50, 50.1, 300, 300.1, np.nan])
    z = tissue_zones(d)
    assert list(z[:-1]) == [
        "in_core", "in_core", "in_edge", "in_edge", "out_near", "out_near", "out_mid", "out_mid", "out_far"
    ]  # fmt: skip
    assert pd.isna(z[-1])
    assert z.ordered
    with pytest.raises(ValueError):
        tissue_zones(d, labels=("a", "b"))


def test_density_profile_known_answer_uniform_chip():
    grid = _grid()
    n = 400
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    air = pd.DataFrame({"x_um": ix * 0.5, "y_um": iy * 0.5, "locus": "IGK"})
    gex = GexLookup.from_arrays(ix * 0.5, iy * 0.5, np.full(len(ix), "Actb"), np.full(len(ix), 2), genes=[])
    edges = (-100, -50, -10, 0, 10, 50, 100)
    res = density_profile(air, grid, edges_um=edges, gex=gex, core_max_um=-20)
    t = res.table
    air_t = t[t.source == "air"]
    # one record per DNB = 4 per um2 in every shell
    np.testing.assert_allclose(air_t.density, 4.0)
    np.testing.assert_allclose(air_t.relative, 1.0)
    g = t[t.group == "gex:total"]
    np.testing.assert_allclose(g.density[g.n > 0], 8.0)
    assert res.summary["core_density[IGK]"] == pytest.approx(4.0)
    assert air_t.area_um2.sum() == pytest.approx(grid.area_um2(edges).sum())


def _legacy_profile(d_records, signed5, edges, groups):
    """Copy of the archived anatomy.distance_profile (area by np.histogram, DNB units)."""
    shell = pd.cut(d_records, edges, right=False)
    h, _ = np.histogram(signed5.ravel(), bins=edges)
    area = pd.Series(h * 25, index=shell.categories)
    tab = pd.DataFrame({"shell": shell, "g": groups, "w": 1}).groupby(["shell", "g"], observed=False)["w"].sum()
    return tab.unstack(fill_value=0).div(area, axis=0)


def test_density_profile_matches_archived_definition():
    rng = np.random.default_rng(2)
    grid = _grid(n=600, tissue_frac=0.4)
    m = 5000
    air = pd.DataFrame(
        {
            "x_um": rng.integers(0, 600, m) * 0.5,
            "y_um": rng.integers(0, 600, m) * 0.5,
            "locus": rng.choice(["IGH", "IGK"], m),
        }
    )
    # the archived area used np.histogram, whose last bin is closed (records: open); an edge that no grid distance
    # (2.5 um x sqrt(integer)) can equal makes both definitions agree
    edges = (-150, -100, -50, -25, 0, 25, 50, 100, 151)
    res = density_profile(air, grid, edges_um=edges)
    d = grid(air.x_um, air.y_um)
    ref = _legacy_profile(d, grid.values, np.asarray(edges, float), air.locus.to_numpy())
    for lc in ("IGH", "IGK"):
        got = res.table[res.table.group == lc].density.to_numpy() * 0.25  # per DNB, as archived
        np.testing.assert_allclose(got, ref[lc].to_numpy(), rtol=1e-12)


def _legacy_similarity(a, b, min_total=20):
    """Copy of the archived anatomy.composition_similarity."""
    t = pd.DataFrame({"a": a, "b": b})
    s = t[(t.a + t.b) >= min_total]
    fa = (s.a + 0.5) / (s.a.sum() + 0.5 * len(s))
    fb = (s.b + 0.5) / (s.b.sum() + 0.5 * len(s))
    p, q = fa.to_numpy() / fa.sum(), fb.to_numpy() / fb.sum()
    mm = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log2(p / mm)) + np.sum(q * np.log2(q / mm)))
    return {
        "n_clones": len(s),
        "spearman": stats.spearmanr(fa, fb).statistic,
        "pearson_log": stats.pearsonr(np.log10(fa), np.log10(fb)).statistic,
        "jsd": jsd,
    }


def _composition_air(seed, n=6000, n_clones=150, shift=False):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(n_clones, 0.3))
    off = rng.random(n) < 0.4
    w_off = rng.dirichlet(np.full(n_clones, 0.3)) if shift else w
    clone = np.where(off, rng.choice(n_clones, n, p=w_off), rng.choice(n_clones, n, p=w))
    # on tissue x < 100 um, far off tissue x > 180 um (grid of 200 um, tissue = left half)
    x = np.where(off, rng.uniform(181, 199, n), rng.uniform(0, 99, n))
    return pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n)],
            "x_um": np.round(x * 2) / 2,
            "y_um": np.round(rng.uniform(0, 199, n) * 2) / 2,
            "locus": "IGK",
            "clone_id": [f"c{c}" for c in clone],
        }
    )


def test_composition_matches_archived_similarity_and_is_deterministic():
    grid = _grid()
    air = _composition_air(1)
    res = composition_on_off(air, grid, far_um=60.0, n_perm=99, rng=3)
    row = res.table.iloc[0]
    d = grid(air.x_um, air.y_um)
    z = np.where(d <= 0, "in", np.where(d > 60, "far", "x"))
    ct = pd.crosstab(air.clone_id, z)
    ref = _legacy_similarity(ct["in"].to_numpy(), ct["far"].to_numpy())
    for k in ("n_clones", "spearman", "pearson_log", "jsd"):
        assert row[k] == pytest.approx(ref[k], rel=1e-12)
    assert row.n_a + row.n_b == len(air)
    again = composition_on_off(air, grid, far_um=60.0, n_perm=99, rng=3)
    pd.testing.assert_frame_equal(res.table, again.table)
    other = composition_on_off(air, grid, far_um=60.0, n_perm=99, rng=4)
    assert not np.array_equal(res.null, other.null)
    # a precomputed distance column gives the same answer
    col = composition_on_off(air.assign(d=d), "d", far_um=60.0, n_perm=99, rng=3)
    pd.testing.assert_frame_equal(res.table, col.table)


def test_composition_g_test_calibration_and_power():
    grid = _grid()
    p = [composition_on_off(_composition_air(s, n=1500, n_clones=40), grid, far_um=60.0, n_perm=99,
                            rng=s).table.p_value.iloc[0] for s in range(60)]  # fmt: skip
    p = np.asarray(p)
    assert (p <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(p))
    shifted = composition_on_off(_composition_air(7, n=1500, n_clones=40, shift=True), grid, far_um=60.0,
                                 n_perm=99, rng=1).table.iloc[0]  # fmt: skip
    assert shifted.p_value <= 0.01 and shifted.pearson_log < 0.9


def _swap_chip(seed, rho=0.02, genuine=0, n=400, lam_in=0.5, lam_out=0.04):
    rng = np.random.default_rng(seed)
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    lam = np.where(ix < n // 2, lam_in, lam_out) * rng.gamma(4.0, 0.25, len(ix))
    m = rng.poisson(lam)
    nz = m > 0
    gex = GexLookup.from_arrays(ix[nz] * 0.5, iy[nz] * 0.5, np.full(nz.sum(), "Actb"), m[nz], genes=[])
    # swap records: each GEX molecule spawns a record with probability rho, on its own DNB
    k = rng.binomial(m, rho)
    rows = np.repeat(np.arange(len(ix)), k)
    tissue = np.flatnonzero(ix < n // 2)
    g = rng.choice(tissue, genuine, replace=True)
    allr = np.concatenate([rows, g])
    air = pd.DataFrame(
        {
            "spot_id": [f"{a}_{b}" for a, b in zip(ix[allr], iy[allr], strict=True)],
            "x_um": ix[allr] * 0.5,
            "y_um": iy[allr] * 0.5,
            "locus": "IGK",
            "clone_id": [f"c{i}" for i in range(len(allr))],
        }
    )
    return air, gex, _grid(n=n), len(rows[ix[rows] < n // 2])


def test_swap_rate_recovers_planted_rate_and_genuine_share():
    cover_rho = cover_share = 0
    for seed in range(25):
        air, gex, grid, n_swap_in = _swap_chip(seed, rho=0.02, genuine=400)
        res = swap_rate(air, gex, grid, bin_um=10.0, far_um=40.0, sink_min_clones=None)
        r = res.table.iloc[0]
        cover_rho += r.rho_lo <= 0.02 <= r.rho_hi
        true_share = 400 / (400 + n_swap_in)
        cover_share += r.genuine_share_lo <= true_share <= r.genuine_share_hi
    assert cover_rho >= 21 and cover_share >= 21
    assert r.slope_lo <= 1.0 <= r.slope_hi
    bd = res.by_depth
    far = res.bins[(res.bins.region == "far") & (res.bins.gex > 0)]
    assert len(bd) >= 3 and bd.records.sum() == far.records.sum()
    assert bd.expected.sum() == pytest.approx(r.rho * far.gex.sum())
    assert set(res.bins.region) == {"far", "tissue", "other"}
    assert r.statistic == r.rho


def test_swap_rate_tissue_test_calibration_and_power():
    p = []
    for seed in range(40):
        air, gex, grid, _ = _swap_chip(100 + seed, rho=0.02, genuine=0)
        p.append(swap_rate(air, gex, grid, bin_um=10.0, far_um=40.0, sink_min_clones=None).table.p_value.iloc[0])
    p = np.asarray(p)
    assert (p <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(p))
    air, gex, grid, _ = _swap_chip(9, rho=0.02, genuine=1500)
    r = swap_rate(air, gex, grid, bin_um=10.0, far_um=40.0, sink_min_clones=None).table.iloc[0]
    assert r.p_value < 1e-4 and r.genuine_share > 0.2
    with pytest.raises(ValueError):
        swap_rate(air, gex, grid, far_um=0.0, tissue_max_um=0.0)


def test_density_profile_plot_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spaclone.pl import density_profile as plot_profile

    grid = _grid()
    rng = np.random.default_rng(0)
    air = pd.DataFrame({"x_um": rng.uniform(0, 199, 3000), "y_um": rng.uniform(0, 199, 3000),
                        "locus": rng.choice(["IGK", "IGH"], 3000)})  # fmt: skip
    gex = GexLookup.from_arrays(air.x_um.round(0), air.y_um.round(0), np.full(3000, "Actb"), genes=[])
    res = density_profile(air, grid, edges_um=(-100, -50, 0, 50, 100), gex=gex, core_max_um=-20)
    ax = plot_profile(res)
    assert ax.get_yscale() == "log" and len(ax.get_legend().get_texts()) == 3
    colours = {ln.get_label(): ln.get_color() for ln in ax.get_lines() if not ln.get_label().startswith("_")}
    fig, ax2 = plt.subplots()
    assert plot_profile(res.table, relative=False, groups=["IGK"], ax=ax2, style="dark") is ax2
    assert colours["IGK"] != colours["IGH"]
    plt.close("all")
