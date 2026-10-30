import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from spaclone import tl
from spaclone.tl.structures import _closing, _remove_small


def _halves(n=6000, size=400.0, noise=0.1, seed=0):
    """Typed units: B on the left half, T on the right half, a share ``noise`` of labels flipped."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, size, n), rng.uniform(0, size, n)
    t = np.where(x < size / 2, "B", "T")
    flip = rng.random(n) < noise
    t = np.where(flip, np.where(t == "B", "T", "B"), t)
    return pd.DataFrame({"x_um": x, "y_um": y, "cell_type": t})


def _b_mask_reference(xy, is_b, threshold):
    """The 2026-10 archive recipe (qc_common.b_mask), with the Otsu function passed in."""
    px = 10.0
    lo = xy.min(0)
    ij = ((xy - lo) // px).astype(int)
    shape = tuple(ij.max(0) + 1)
    n, k = np.zeros(shape), np.zeros(shape)
    np.add.at(n, (ij[:, 0], ij[:, 1]), 1)
    np.add.at(k, (ij[:, 0], ij[:, 1]), is_b)
    n, k = ndi.gaussian_filter(n, 3), ndi.gaussian_filter(k, 3)
    share = k / np.maximum(n, 1e-9)
    m = (n > 0.15) & (share >= threshold(share[n > 0.15]))
    m = ndi.binary_opening(ndi.binary_closing(m, iterations=2), iterations=1)
    for target in (m, ~m):
        lab, nl = ndi.label(target)
        small = np.isin(lab, 1 + np.where(ndi.sum(target, lab, range(1, nl + 1)) < 80)[0])
        m = m & ~small if target is m else m | small
    return m, lo


def test_grid_from_points_index_and_rasterize():
    g = tl.Grid.from_points([5.0, 25.0, 39.9], [100.0, 100.0, 119.0], pixel_um=10.0)
    assert g.origin == (5.0, 100.0)
    assert g.shape == (4, 2)  # (39.9 - 5) // 10 = 3 ; (119 - 100) // 10 = 1
    i, j, inside = g.index([5.0, 25.0, 4.9, np.nan, 44.9], [100.0, 109.9, 100.0, 100.0, 100.0])
    np.testing.assert_array_equal(inside, [True, True, False, False, True])
    np.testing.assert_array_equal(i[inside], [0, 2, 3])
    np.testing.assert_array_equal(j[inside], [0, 0, 0])
    r = g.rasterize([5.0, 6.0, 25.0], [100.0, 101.0, 100.0], values=[1.0, 2.0, 5.0])
    assert r[0, 0] == 3.0 and r[2, 0] == 5.0 and r.sum() == 8.0
    assert g.extent == (5.0, 45.0, 100.0, 120.0)
    gx, gy = g.centres()
    np.testing.assert_allclose(gx, [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(gy, [105.0, 115.0])
    with pytest.raises(ValueError):
        tl.Grid((0, 0), 0.0, (2, 2))


def test_raster_sample_categories_and_logic():
    g = tl.Grid((0.0, 0.0), 10.0, (3, 2))
    r = tl.Raster(np.array([[0, 1], [2, 0], [1, 1]]), g, categories=("a", "b", "c"))
    np.testing.assert_array_equal(r.sample([5, 15, 99], [15, 5, 5], fill=-1), [1, 2, -1])
    np.testing.assert_array_equal(r.sample_categories([5, 15, 99], [15, 5, 5], fill="out"), ["b", "c", "out"])
    m = tl.Raster(np.array([[True, False], [False, False], [True, True]]), g, name="m")
    assert (~m).values.sum() == 3 and (m & ~m).values.sum() == 0 and (m | ~m).values.all()
    assert m.area_um2 == 300.0
    with pytest.raises(ValueError):
        tl.Raster(np.zeros((2, 2)), g)


def test_otsu_threshold_separates_two_modes_and_matches_skimage():
    rng = np.random.default_rng(1)
    v = np.r_[rng.normal(0.2, 0.05, 3000), rng.normal(0.8, 0.05, 1000)]
    t = tl.otsu_threshold(v)
    assert 0.35 < t < 0.65
    assert tl.otsu_threshold(np.full(5, 0.3)) == 0.3
    skf = pytest.importorskip("skimage.filters")
    for seed in range(5):
        w = np.random.default_rng(seed).beta(0.5 + seed, 2.0, 5000)
        assert tl.otsu_threshold(w) == pytest.approx(float(skf.threshold_otsu(w)), abs=0, rel=0)


def test_mask_from_types_matches_the_archive_recipe():
    units = _halves()
    xy = units[["x_um", "y_um"]].to_numpy()
    ref, lo = _b_mask_reference(xy, (units.cell_type == "B").to_numpy(), tl.otsu_threshold)
    m = tl.mask_from_types(units, types="B", type_key="cell_type", pad_edges=False)
    assert m.grid.origin == (lo[0], lo[1])
    np.testing.assert_array_equal(m.values, ref)
    # the default keeps mask pixels on the grid edge, which scipy's closing erodes: a superset here
    padded = tl.mask_from_types(units, types="B", type_key="cell_type")
    assert (padded.values >= m.values).all() and padded.values.sum() > m.values.sum()
    skf = pytest.importorskip("skimage.filters")
    ref2, _ = _b_mask_reference(xy, (units.cell_type == "B").to_numpy(), skf.threshold_otsu)
    np.testing.assert_array_equal(m.values, ref2)


def test_closing_keeps_edge_pixels_only_with_padding():
    m = np.zeros((12, 12), bool)
    m[:6, :] = True  # touches three grid edges
    np.testing.assert_array_equal(_closing(m, 2, True), m)
    legacy = _closing(m, 2, False)
    np.testing.assert_array_equal(legacy, ndi.binary_closing(m, iterations=2))
    assert not legacy[0].any() and legacy[2:6, 2:-2].all()


def test_mask_from_types_recovers_the_typed_half():
    units = _halves()
    m = tl.mask_from_types(units, types="B")
    gx, _ = m.grid.centres()
    truth = np.broadcast_to((gx < 200)[:, None], m.grid.shape)
    assert (m.values == truth).mean() > 0.98
    t = tl.mask_from_types(units, types=["T"])
    assert (m.values & t.values).mean() < 0.01  # complementary classes: Otsu thresholds near 0.5 in both


def test_mask_from_types_is_translation_invariant():
    units = _halves(n=3000)
    a = tl.mask_from_types(units, types="B")
    b = tl.mask_from_types(units.assign(x_um=units.x_um + 1234.0, y_um=units.y_um - 77.0), types="B")
    np.testing.assert_array_equal(a.values, b.values)
    assert b.grid.origin == pytest.approx((a.grid.origin[0] + 1234.0, a.grid.origin[1] - 77.0))
    c = tl.mask_from_types(units.sample(frac=1.0, random_state=3), types="B")
    np.testing.assert_array_equal(a.values, c.values)


def test_remove_small_islands_and_holes():
    m = np.zeros((30, 30), bool)
    m[2:20, 2:20] = True  # big square, 324 px
    m[8:10, 8:10] = False  # 4-px hole
    m[25:27, 25:27] = True  # 4-px island
    out = _remove_small(m, 80)
    assert out[8:10, 8:10].all() and not out[25:27, 25:27].any() and out[2:20, 2:20].all()


def test_zones_and_min_area():
    g = tl.Grid((0.0, 0.0), 10.0, (20, 10))
    m = np.zeros(g.shape, bool)
    m[1:4, 1:4] = True  # 9 px = 900 um2
    m[10:18, 2:9] = True  # 56 px
    m[5, 5] = m[6, 6] = True  # diagonal pair: two zones with connectivity 1, one with 2
    z = tl.zones(tl.Raster(m, g))
    assert z.values.max() == 4
    assert tl.zones(tl.Raster(m, g), connectivity=2).values.max() == 3
    big = tl.zones(tl.Raster(m, g), min_area_um2=1000.0)
    assert big.values.max() == 1 and (big.values > 0).sum() == 56
    with pytest.raises(ValueError, match="pixel_um"):
        tl.zones(m)
    assert tl.zones(m, pixel_um=10.0).values.max() == 4


def test_signed_distance_known_values():
    g = tl.Grid((0.0, 0.0), 10.0, (9, 1))
    m = np.zeros(g.shape, bool)
    m[3:6, 0] = True
    d = tl.signed_distance(tl.Raster(m, g)).values[:, 0]
    np.testing.assert_allclose(d, [30, 20, 10, -10, -20, -10, 10, 20, 30])
    assert np.isposinf(tl.signed_distance(tl.Raster(np.zeros(g.shape, bool), g)).values).all()
    assert np.isneginf(tl.signed_distance(tl.Raster(np.ones(g.shape, bool), g)).values).all()
    # 2-D: the diagonal neighbour of a single pixel is sqrt(2) pixels away
    g2 = tl.Grid((0.0, 0.0), 5.0, (3, 3))
    one = np.zeros((3, 3), bool)
    one[1, 1] = True
    d2 = tl.signed_distance(tl.Raster(one, g2)).values
    assert d2[1, 1] == -5.0 and d2[0, 0] == pytest.approx(5.0 * np.sqrt(2))


def test_nearest_zone_and_compartments_priority():
    g = tl.Grid((0.0, 0.0), 10.0, (10, 3))
    a = np.zeros(g.shape, bool)
    a[0:2, :] = True
    b = np.zeros(g.shape, bool)
    b[8:10, :] = True
    z = tl.zones(tl.Raster(a | b, g))
    near = tl.nearest_zone(z).values[:, 1]
    np.testing.assert_array_equal(near, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    t_rich = tl.Raster(a, g)
    ery = tl.Raster(b | a, g)  # overlaps the T-rich pixels: priority decides
    b_rich = tl.Raster(np.zeros(g.shape, bool) | (np.arange(10)[:, None] < 6), g)
    comp = tl.compartments([("T zone", t_rich), ("red pulp", ery), ("red pulp", ~b_rich)], default="follicle")
    assert comp.categories == ("T zone", "red pulp", "follicle")
    names = comp.sample_categories(np.arange(10) * 10 + 5, np.full(10, 15.0))
    np.testing.assert_array_equal(names, ["T zone"] * 2 + ["follicle"] * 4 + ["red pulp"] * 4)
    with pytest.raises(ValueError, match="default"):
        tl.compartments({"a": t_rich}, default="a")
    with pytest.raises(ValueError, match="grid"):
        tl.compartments({"a": a}, default="b")


def test_tissue_mask_closes_and_fills():
    xs, ys = np.meshgrid(np.arange(0, 200, 10.0) + 5, np.arange(0, 200, 10.0) + 5, indexing="ij")
    keep = ~((np.abs(xs - 100) < 30) & (np.abs(ys - 100) < 30))  # a 6x6-pixel hole
    units = pd.DataFrame({"x_um": xs[keep], "y_um": ys[keep]})
    t = tl.tissue_mask(units, pixel_um=10.0)
    assert t.values.all()
    legacy = tl.tissue_mask(units, pixel_um=10.0, pad_edges=False)  # scipy's closing erodes a 2-pixel frame
    assert legacy.values[2:-2, 2:-2].all() and not legacy.values[0].any()
    raw = tl.tissue_mask(units, pixel_um=10.0, close_iter=0, fill_holes=False)
    assert raw.values.sum() == keep.sum()
