import json

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from scipy.optimize import curve_fit

from spaclone.io import (
    DistanceGrid,
    p_det_from_saturation,
    read_gef_bin1,
    read_stereo_gef,
    read_tissue_mask,
    signed_distance_map,
)
from spaclone.qc import GexLookup

tifffile = pytest.importorskip("tifffile")


def _write_gef(path, entries, max_x=99, max_y=79):
    """Minimal SAW bin1 .gef: entries = list of (gene_id, gene_name, [(x, y, count), ...]) sorted by gene."""
    dt = np.dtype([("x", "<u4"), ("y", "<u4"), ("count", "u1")])
    rows, genes, off = [], [], 0
    for gid, name, cells in entries:
        arr = np.array(cells, dtype=dt)
        rows.append(arr)
        genes.append((gid.encode(), name.encode(), off, len(arr)))
        off += len(arr)
    expr = np.concatenate(rows)
    gtab = np.array(genes, dtype=[("geneID", "S64"), ("geneName", "S64"), ("offset", "<u4"), ("count", "<u4")])
    with h5py.File(path, "w") as h:
        h.attrs["sn"] = "TEST01"
        e = h.create_dataset("geneExp/bin1/expression", data=expr)
        for k, v in (("maxX", max_x), ("maxY", max_y), ("minX", 0), ("minY", 0), ("resolution", 500)):
            e.attrs[k] = np.array([v], dtype=np.uint32)
        h.create_dataset("geneExp/bin1/gene", data=gtab)
    return expr


@pytest.fixture
def gef(tmp_path):
    rng = np.random.default_rng(0)
    entries = []
    for gi, name in enumerate(["Actb", "Igkc", "Ighm", "Igkc"]):  # Igkc listed under two gene ids
        cells = {(int(rng.integers(0, 100)), int(rng.integers(0, 80))) for _ in range(300)}
        cl = sorted(cells)
        entries.append((f"G{gi}", name, [(x, y, int(rng.integers(1, 5))) for x, y in cl]))
    entries[0][2][0] = (*entries[0][2][0][:2], 0)  # one row with 0 MIDs
    p = tmp_path / "chip.raw.gef"
    _write_gef(p, entries)
    return p, entries


def test_read_gef_bin1_totals_genes_and_meta(gef):
    path, entries = gef
    g = read_gef_bin1(path, ["Igkc", "Ighm", "Nope"])
    long = pd.DataFrame([(n, x, y, c) for _, n, cells in entries for x, y, c in cells], columns=["g", "x", "y", "c"])
    long = long[long.c > 0]
    tot = long.groupby(["x", "y"]).c.sum()
    got = pd.Series(g.total, index=pd.MultiIndex.from_arrays([g.x, g.y]))
    pd.testing.assert_series_equal(got.sort_index(), tot.sort_index(), check_names=False, check_dtype=False)
    igkc = long[long.g == "Igkc"].groupby(["x", "y"]).c.sum()  # both gene ids summed
    gx, gy, gc = g.genes["Igkc"]
    assert dict(zip(zip(gx, gy, strict=True), gc, strict=True)) == igkc.to_dict()
    assert len(g.genes["Nope"][0]) == 0 and g.meta["genes_missing"] == ["Nope"]
    assert g.meta["n_zero_entries"] == 1 and g.meta["um_per_dnb"] == 0.5 and g.meta["sn"] == "TEST01"
    assert (g.meta["max_x"], g.meta["max_y"]) == (99, 79)
    small = read_stereo_gef(path, ["Igkc"], chunk_rows=7)  # chunked reading gives the same result
    np.testing.assert_array_equal(small.total, g.total)
    np.testing.assert_array_equal(small.genes["Igkc"][2], g.genes["Igkc"][2])
    lk = GexLookup.from_gef(path, ["Igkc"])
    m, own = lk.spot_counts(gx * 0.5, gy * 0.5, ["Igkc"])
    np.testing.assert_array_equal(own, gc)
    assert lk.extent == (0, 99, 0, 79)


def _legacy_geometry(m, b5=5):
    """Copy of the archived gexfield.tissue_geometry after reading the image."""
    m5 = m[: (m.shape[0] // b5) * b5, : (m.shape[1] // b5) * b5]
    m5 = m5.reshape(m5.shape[0] // b5, b5, m5.shape[1] // b5, b5).mean(axis=(1, 3)) >= 0.5
    return (np.where(m5, -ndimage.distance_transform_edt(m5), ndimage.distance_transform_edt(~m5)) * b5 * 0.5).astype(
        np.float32
    )


def test_tissue_mask_without_tifffile_names_the_extra(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tifffile", None)  # as in a core install
    with pytest.raises(ImportError, match=r"needs tifffile: pip install 'spaclone\[images\]'"):
        read_tissue_mask(tmp_path / "tissue_cut.tif")


def test_tissue_mask_reader_and_signed_distance(tmp_path):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:203, 0:157]
    mask = ((xx - 80) ** 2 + (yy - 100) ** 2 < 60**2) | (rng.random((203, 157)) < 0.02)
    f = tmp_path / "tissue_cut.tif"
    tifffile.imwrite(f, (mask * 255).astype(np.uint8))
    m = read_tissue_mask(f, shape=(203, 157))
    np.testing.assert_array_equal(m, mask)
    np.testing.assert_array_equal(read_tissue_mask(f, shape=(157, 203)), mask.T)  # stored transposed
    with pytest.raises(ValueError, match="transpose"):
        read_tissue_mask(f, shape=(100, 100))
    grid = signed_distance_map(m)
    np.testing.assert_array_equal(grid.values, _legacy_geometry(m))
    assert grid.pixel_um == 2.5 and grid.shape == (40, 31)
    # half plane, no reduction: distances are exact multiples of the pixel
    half = np.zeros((20, 30), bool)
    half[:, :10] = True
    g1 = signed_distance_map(half, pixel_um=0.5, downsample=1)
    assert g1(4.75, 3.0) == pytest.approx(-0.5) and g1(5.25, 3.0) == pytest.approx(0.5)
    assert g1(0.0, 0.0) == pytest.approx(-0.5 * 5) or g1(0.0, 0.0) < 0
    assert g1(1e6, 1e6) == g1.values[-1, -1]  # off the grid: nearest edge cell
    assert g1.in_tissue(np.array([1.0, 10.0]), np.array([1.0, 1.0])).tolist() == [True, False]
    assert g1.area_inside_um2 == pytest.approx(200 * 0.25) and g1.area_outside_um2 == pytest.approx(400 * 0.25)
    a = g1.area_um2([-1.0, 0.0, 0.75, 100.0])
    assert a[0] == pytest.approx(20 * 2 * 0.25) and a[1] == pytest.approx(20 * 0.25)
    with pytest.raises(ValueError, match="no edge"):
        signed_distance_map(np.ones((10, 10), bool), downsample=1)
    assert isinstance(grid, DistanceGrid)


def _legacy_a4(f, u):
    """Copy of the archived A4 Michaelis-Menten and Gamma-Poisson fits."""

    def nb(f, m, mu, k):
        return m * (1 - (1 + f * mu / k) ** (-k))

    (m, mu, k), _ = curve_fit(nb, f, u, p0=(u[-1] * 1.3, 3.0, 1.0), bounds=([u[-1], 1e-3, 1e-3],
                              [u[-1] * 100, 1e3, 1e3]), maxfev=200000)  # fmt: skip
    (um, kk), _ = curve_fit(lambda f, um, kk: um * f / (kk + f), f, u, p0=(u[-1] * 1.5, 0.5), maxfev=200000)
    return {"p_seen_mm": u[-1] / um, "um": um, "k": kk, "nb_m": m}


def test_p_det_from_saturation(tmp_path):
    f = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    rng = np.random.default_rng(2)
    u = 1e8 * f / (0.7 + f) * (1 + rng.normal(0, 0.002, len(f)))
    p = tmp_path / "saturation.json"
    p.write_text(json.dumps({"All": {"sample": f.tolist(), "bar_umi": u.tolist(), "bar_y1": (f * 0.8).tolist()}}))
    r = p_det_from_saturation(p)
    ref = _legacy_a4(f, u)
    assert r["p_det"] == ref["p_seen_mm"] and r["mm_umax"] == ref["um"] and r["mm_k"] == ref["k"]
    assert r["nb_m"] == ref["nb_m"]
    assert r["p_det"] == pytest.approx(1 / 1.7, rel=0.01) and 0 < r["se"] < 0.01
    assert r["sequencing_saturation"] == pytest.approx(0.8)
    assert r["unique_full_depth"] == u[-1]
