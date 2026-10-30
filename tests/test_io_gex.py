"""GEX sources on real excerpts (tests/_data/formats): layouts, coordinates and the µm unit from technology facts."""

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import io
from spaclone.io.gexio import detect_unit, gex_format, source_coordinates

F = Path(__file__).parent / "_data" / "formats"


def _unit(a):
    xy, src, _ = source_coordinates(a)
    return detect_unit(a, xy, coordinates_from=src)


@pytest.mark.parametrize(("folder", "positions"), [("visium_sd_sr1", "headerless"), ("visium_sd_sr2", "header")])
def test_space_ranger_sd_folders(folder, positions):
    assert gex_format(F / folder)[0] == "spaceranger"
    a, info = io.read_gex(F / folder)
    sf = json.loads((F / folder / "spatial/scalefactors_json.json").read_text())
    pos_file = "tissue_positions_list.csv" if positions == "headerless" else "tissue_positions.csv"
    pos = pd.read_csv(F / folder / "spatial" / pos_file, header=None if positions == "headerless" else 0)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = pos.set_index("barcode").loc[a.obs_names]
    np.testing.assert_allclose(a.obsm["spatial"], pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy())
    sx, sy, _, rule = _unit(a)
    assert "100 µm apart" in rule and (a.var["feature_types"] == "Gene Expression").all() and a.obs.in_tissue.all()
    # neighbouring spots of one array row are 100 µm apart (Visium SD centre-to-centre); spot_diameter_fullres would
    # have said otherwise: it spans a "theoretical 65 µm spot" that measures 65-70 µm here
    p = pos.assign(x=a.obsm["spatial"][:, 0] * sx, y=a.obsm["spatial"][:, 1] * sy)
    r = p[p.array_row == p.array_row.iloc[0]].sort_values("array_col")
    steps = np.hypot(np.diff(r.x), np.diff(r.y)) / np.diff(r.array_col) * 2
    assert np.allclose(steps, 100.0, rtol=0.01)
    assert 60 < sf["spot_diameter_fullres"] * sx < 75


def test_visium_hd_bins_are_8_um_apart(tmp_path):
    a, info = io.read_gex(F / "visium_hd_square_008um")
    assert info["format"] == "visium_hd" and a.obs_names.str.match(r"^s_008um_\d{5}_\d{5}-1$").all()
    sx, _, name, rule = _unit(a)
    assert "microns_per_pixel" in rule and name.startswith("Visium HD")
    xy = a.obsm["spatial"] * sx
    o = a.obs.assign(x=xy[:, 0], y=xy[:, 1]).sort_values(["array_row", "array_col"])
    same_row = o.groupby("array_row").filter(lambda g: len(g) > 1)
    d = same_row.groupby("array_row").apply(lambda g: np.diff(g.x.to_numpy()) / np.diff(g.array_col.to_numpy()))
    assert np.allclose(np.abs(np.concatenate(d.to_list())), 8.0, rtol=1e-3)
    for b in ("square_002um", "square_008um"):
        (tmp_path / "outs" / "binned_outputs" / b).mkdir(parents=True)
    with pytest.raises(ValueError, match="binned_outputs"):  # several bin sizes: the user picks one
        gex_format(tmp_path / "outs")


def test_xenium_folder_is_um_and_drops_controls():
    a, info = io.read_gex(F / "xenium")
    assert info["format"] == "xenium" and a.n_vars == 30
    assert _unit(a)[:2] == (1.0, 1.0)
    np.testing.assert_allclose(a.obsm["spatial"], a.obs[["x_centroid", "y_centroid"]].to_numpy())
    b = ad.AnnData(obs=a.obs.copy())  # a Xenium cell table converted elsewhere: centroids are µm
    b.obsm["spatial"] = a.obsm["spatial"]
    assert "Xenium" in _unit(b)[3]


def test_stereo_gef_gem_and_cellbins():
    gef, _ = io.read_gex(F / "stereo/chipA_window.raw.gef", bin_size=20)
    gem, _ = io.read_gex(F / "stereo/chipA_window.gem.gz", bin_size=20)
    assert gef.n_obs == 16 and list(gef.obs_names) == list(gem.obs_names)
    common = gef.var["gene_ids"].to_numpy()
    gem_by_id = gem[:, common].X.toarray()  # GEM names genes by geneID
    np.testing.assert_array_equal(gef.X.toarray(), gem_by_id)
    x, y = gef.obs.x.to_numpy(), gef.obs.y.to_numpy()
    assert (x % 20 == 0).all() and (gef.obs_names.astype(np.int64) == (x << 32 | y)).all()
    assert _unit(gef)[:2] == (0.5, 0.5)
    with pytest.raises(ValueError, match="bin_size="):
        io.read_gex(F / "stereo/chipA_window.raw.gef")
    with pytest.raises(ValueError, match="bin_size="):
        io.read_gex(F / "stereo/chipA_window.gem.gz")
    with h5py.File(F / "stereo/chipA_window.raw.gef") as h:
        assert int(h["geneExp/bin1/expression"]["count"].sum()) == int(gef.X.sum())
    cb, info = io.read_gex(F / "stereo/chipA_subset.cellbin.gef")
    assert info["format"] == "stereo_cellbin_gef" and cb.n_obs == 60 and cb.uns["bin_type"] == "cell_bins"
    with h5py.File(F / "stereo/chipA_subset.cellbin.gef") as h:
        np.testing.assert_array_equal(np.asarray(cb.X.sum(axis=1)).ravel(), h["cellBin/cell"]["expCount"])
    cg, _ = io.read_gex(F / "stereo/chipA_window_cellbin_synthetic.gem.gz")
    assert cg.uns["bin_type"] == "cell_bins" and cg.X.sum() == gem.X.sum()


def test_stereopy_h5ad_and_stereo_ids():
    a, info = io.read_gex(F / "stereo/chipA_bin20_stereopy_subset.h5ad")
    assert info["format"] == "anndata"
    sx, _, name, rule = _unit(a)
    assert sx == 0.5 and "uns['bin_type'] = 'bins'" in rule
    b = ad.AnnData(obs=pd.DataFrame(index=a.obs_names.copy()))  # packed ids without the uns markers
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"])
    assert "packed integer" in _unit(b)[3]
    c = b.copy()
    c.obs_names = [f"{int(u)}_{int(v)}" for u, v in c.obsm["spatial"]]
    assert "'x_y'" in _unit(c)[3] and "bin20" in _unit(c)[3]
    d = b.copy()
    d.obs_names = [f"bin{i}" for i in range(d.n_obs)]
    with pytest.raises(ValueError, match="unit_um=.*20-unit lattice"):
        _unit(d)


def test_coordinates_and_unit_rules_raise_without_facts():
    a = ad.AnnData(obs=pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "imagecol": [9.0, 8.0], "imagerow": [7.0, 6.0]},
                                    index=["a", "b"]))  # fmt: skip
    with pytest.raises(ValueError, match="spatial_key="):
        source_coordinates(a)
    xy, src, _ = source_coordinates(a, "obs:imagecol,imagerow")
    assert src.startswith("obs['imagecol']") and xy[0, 0] == 9.0
    a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy()
    with pytest.raises(ValueError, match="unit_um="):
        _unit(a)
    v = a.copy()
    v.uns["spatial"] = {"s1": {"scalefactors": {"microns_per_pixel": 0.27}},
                        "s2": {"scalefactors": {"microns_per_pixel": 0.29}}}  # fmt: skip
    with pytest.raises(ValueError, match="different scale factors"):
        _unit(v)


def test_slideseq_needs_unit_and_seurat_gives_the_snippet(tmp_path):
    a, info = io.read_gex(F / "slideseq")
    assert info["format"] == "slideseq" and a.shape == (2, 3)
    with pytest.raises(ValueError, match="unit_um="):
        _unit(a)
    with pytest.raises(ValueError, match="SaveH5Seurat|sceasy"):
        io.read_gex(F / "seurat/object.rds")
    with h5py.File(tmp_path / "native.h5ad", "w") as h:  # stereopy's own StereoExpData h5 layout
        h.create_group("exp_matrix")
        h.create_group("position")
        h.create_group("cells")
        h.create_group("genes")
    with pytest.raises(ValueError, match="stereo_to_anndata"):
        io.read_gex(tmp_path / "native.h5ad")


def test_spatialdata_table():
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    a = ad.AnnData(X=sparse.csr_matrix(np.ones((3, 2))), obs=pd.DataFrame({"region": "cells", "id": [0, 1, 2]},
                                                                          index=["c0", "c1", "c2"]))  # fmt: skip
    a.obs["region"] = a.obs["region"].astype("category")
    a.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sdata = sd.SpatialData(
        tables={"table": TableModel.parse(a, region="cells", region_key="region", instance_key="id")}
    )
    g, info = io.read_gex(sdata)
    assert info["format"] == "spatialdata" and g.n_obs == 3
