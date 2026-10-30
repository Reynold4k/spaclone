from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import io
from spaclone.io.paired import PairedReadWarning, detect_unit

GENES = ["Jchain", "Xbp1", "Cd3e", "Actb", "Ighg1"]


def _gex(xy, ids=None, raw=True):
    """Units at xy (source units) with integer counts; X keeps 3 log genes, .raw the 5 count genes."""
    rng = np.random.default_rng(0)
    n = len(xy)
    counts = rng.poisson(3.0, (n, len(GENES))).astype(np.float32)
    ids = ids if ids is not None else [f"{int(x)}-{int(y)}" for x, y in xy]
    a = ad.AnnData(X=sparse.csr_matrix(counts), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=GENES))
    a.obsm["spatial"] = np.asarray(xy)
    a.obs["x"], a.obs["y"] = np.asarray(xy)[:, 0], np.asarray(xy)[:, 1]
    if raw:
        a.raw = a
        a = a[:, GENES[:3]].copy()
        a.X = sparse.csr_matrix(np.log1p(a.X.toarray()))
    return a, counts


def _airr(spots, n_extra=0):
    rows = []
    for k, s in enumerate(spots):
        rows.append(
            {
                "sequence_id": f"r{k}",
                "cell_id": s,
                "locus": "IGH",
                "v_call": "IGHV1-26*01",
                "j_call": "IGHJ4*01",
                "c_call": "IGHG1*01" if k % 2 else "IGHM*01",
                "junction": "TGTGCAAGATGG",
                "junction_aa": "CARDYW" if k % 3 else "CARGGYW",
                "productive": "T",
                "consensus_count": 40,
                "v_sequence_alignment": "ACGTACGTAC",
                "v_germline_alignment": "ACGTACGTAA",
            }
        )
    return pd.DataFrame(rows)


def _grid(nx=4, ny=3, pitch=20, x0=1000, y0=2000):
    return np.array([[x0 + pitch * i, y0 + pitch * j] for i in range(nx) for j in range(ny)], dtype=np.int64)


def _write(tmp_path: Path, a: ad.AnnData, air: pd.DataFrame, index_col=True):
    g, t = tmp_path / "gex.h5ad", tmp_path / "airr.tsv"
    a.write_h5ad(g)
    air.to_csv(t, sep="\t", index=index_col)
    return g, t


def test_read_stereo_bins_units_counts_and_positions(tmp_path):
    xy = _grid()
    a, counts = _gex(xy)
    spots = [a.obs_names[0], a.obs_names[0], a.obs_names[5], a.obs_names[7]]
    g, t = _write(tmp_path, a, _airr(spots))
    sa = io.read(g, t)
    m = sa.meta
    assert m["um_per_unit"] == 0.5 and "bin20" in m["unit_rule"]
    assert m["counts_from"] == "raw.X" and m["n_gex_units"] == 12
    assert sa.gex.n_obs == 12 and list(sa.gex.var_names) == GENES  # every unit, the raw genes
    np.testing.assert_array_equal(sa.gex.X.toarray(), counts)
    assert sa.gex.layers["counts"] is sa.gex.X
    np.testing.assert_allclose(sa.gex.obsm["spatial"], 0.5 * xy)
    np.testing.assert_allclose(sa.gex.obsm["spatial_src"], xy)
    lookup = pd.DataFrame(0.5 * xy, index=a.obs_names, columns=["x", "y"])
    np.testing.assert_allclose(sa.air[["x_um", "y_um"]].to_numpy(), lookup.loc[sa.air.spot_id].to_numpy())
    assert sa.air.spot_id.tolist() == spots and sa.air.in_gex.all()
    assert m["spot_overlap"] == 1.0 and m["record_overlap"] == 1.0 and m["unmatched_rule"] == "none unmatched"
    assert "v_sequence_alignment" in sa.air.columns and sa.air.clone_id.str.startswith("IGH|").all()
    assert sa.gex.uns["spaclone"]["um_per_unit"] == 0.5
    # the spaclone GEX is recognised as already in µm when it is read again
    sx, sy, _, rule = detect_unit(sa.gex, sa.gex.obsm["spatial"])
    assert (sx, sy) == (1.0, 1.0) and "already" in rule
    # determinism
    sa2 = io.read(g, t)
    pd.testing.assert_frame_equal(sa.air, sa2.air)


def test_unit_detection_rules_and_override(tmp_path):
    xy = _grid(pitch=7, x0=3, y0=5)  # not a Stereo-seq bin grid
    a, _ = _gex(xy)
    g, t = _write(tmp_path, a, _airr([a.obs_names[1]]))
    with pytest.raises(ValueError, match=r"unit_um=.*7-unit lattice"):  # no technology fact: never assumed
        io.read(g, t)
    sa = io.read(g, t, unit_um=1.0)
    assert sa.meta["um_per_unit"] == 1.0 and sa.meta["unit_rule"].startswith("given")
    sa = io.read(g, t, unit_um=0.65)
    np.testing.assert_allclose(sa.gex.obsm["spatial"], 0.65 * xy)
    assert sa.meta["unit_rule"].startswith("given")
    sa = io.read(g, t, unit_um=(0.5, 2.0))
    np.testing.assert_allclose(sa.gex.obsm["spatial"], xy * [0.5, 2.0])
    assert sa.meta["um_per_unit"] == [0.5, 2.0]
    # bin50 / bin100 grids are DNB too; Visium scale factors win
    for pitch in (50, 100):
        b, _ = _gex(_grid(pitch=pitch))
        assert detect_unit(b, b.obsm["spatial"].astype(float))[0] == 0.5
    v, _ = _gex(xy.astype(float) * 13.3)
    v.uns["spatial"] = {"lib": {"scalefactors": {"spot_diameter_fullres": 110.0}}}
    with pytest.raises(ValueError, match="array_row"):  # SD pixel size is measured on the 100 µm spot array
        detect_unit(v, v.obsm["spatial"])
    v.obs["array_row"] = (xy[:, 1] - xy[:, 1].min()) // 7
    v.obs["array_col"] = (xy[:, 0] - xy[:, 0].min()) // 7 * 2
    assert detect_unit(v, v.obsm["spatial"])[0] == pytest.approx(100 / (7 * 13.3))
    v.uns["spatial"] = {"lib": {"scalefactors": {"microns_per_pixel": 0.2737}}}
    assert detect_unit(v, v.obsm["spatial"])[0] == pytest.approx(0.2737)
    with pytest.raises(ValueError, match="unit_um"):
        io.read(g, t, unit_um=-1)


def test_id_mismatch_raises_a_clear_error(tmp_path):
    a, _ = _gex(_grid())
    air = _airr([f"{s}-1" for s in a.obs_names[:4]])  # a barcode suffix the GEX does not have
    g, t = _write(tmp_path, a, air)
    with pytest.raises(ValueError, match=r"only 0\.0% .*spot_key") as e:
        io.read(g, t)
    assert "1000-2000-1" in str(e.value) and "1000-2000" in str(e.value)
    with pytest.raises(KeyError, match="spot_key"):
        io.read(g, t, spot_key="barcode")


def test_unmatched_spots_placed_from_their_ids_or_dropped(tmp_path):
    a, _ = _gex(_grid())
    spots = [a.obs_names[0], a.obs_names[1], a.obs_names[2], "1100-2100"]  # the last bin is not a GEX unit
    g, t = _write(tmp_path, a, _airr(spots))
    sa = io.read(g, t)
    assert sa.meta["unmatched_rule"] == "coordinates" and sa.meta["n_records_placed_from_id"] == 1
    last = sa.air.iloc[-1]
    assert (last.x_um, last.y_um) == (550.0, 1050.0) and not last.in_gex
    assert sa.meta["spot_overlap"] == pytest.approx(0.75) and sa.gex.n_obs == 12
    with pytest.warns(PairedReadWarning, match="left out"):
        sd = io.read(g, t, unmatched="drop")
    assert len(sd.air) == 3 and sd.meta["n_records_dropped"] == 1
    with pytest.raises(ValueError, match="not GEX units"):
        io.read(g, t, unmatched="error")
    with pytest.raises(ValueError, match="min_overlap"):
        io.read(g, t, min_overlap=0.9)
    # an unmatched id that does not spell a position cannot be placed: left out with a warning
    g2, t2 = _write(tmp_path, a, _airr([*spots, "unplaceable"]))
    with pytest.warns(PairedReadWarning, match="neither GEX units"):
        s2 = io.read(g2, t2)
    assert len(s2.air) == 4 and s2.meta["n_records_dropped"] == 1 and s2.meta["n_records_placed_from_id"] == 1


def test_barcode_ids_dataframe_input_and_spot_key(tmp_path):
    xy = _grid().astype(float) * 0.5
    bcs = [f"AAAC{i:04d}-1" for i in range(len(xy))]
    a, _ = _gex(xy, ids=bcs, raw=False)
    a.obs["in_tissue"] = np.arange(len(xy)) < 10
    a.uns["spaclone"] = {"um_per_unit": 1.0}  # already µm
    air = _airr([bcs[0], bcs[3], "GGGG0001-1"]).rename(columns={"cell_id": "barcode"})
    with pytest.warns(PairedReadWarning, match="left out"):
        sa = io.read(a, air, spot_key="barcode", counts="X")
    assert sa.meta["unmatched_rule"] == "drop" and len(sa.air) == 2
    np.testing.assert_allclose(sa.air[["x_um", "y_um"]].to_numpy(), xy[[0, 3]])
    assert sa.air.in_tissue.tolist() == [True, True]
    assert sa.meta["counts_from"] == "X" and sa.meta["um_per_unit"] == 1.0
    with pytest.raises(ValueError, match="'coordinates' needs"):
        io.read(a, air, spot_key="barcode", unmatched="coordinates")
    a.layers["counts"] = a.X.copy()
    sb = io.read(a, air[air.barcode != "GGGG0001-1"], spot_key="barcode", counts="counts")
    assert sb.meta["counts_from"] == "layers['counts']"
    with pytest.raises(KeyError, match="neither"):
        io.read(a, air, spot_key="barcode", counts="nope")


def test_store_directories_are_still_read(tmp_path):
    a, _ = _gex(_grid())
    g, t = _write(tmp_path, a, _airr([a.obs_names[0], a.obs_names[4]]))
    sa = io.read(g, t)
    io.write(sa, tmp_path / "store")
    back = io.read(tmp_path / "store")
    assert back.gex.n_obs == 12 and len(back.air) == 2
    with pytest.raises(TypeError, match="needs a GEX"):
        io.read(g)


def test_non_integer_counts_warn(tmp_path):
    a, _ = _gex(_grid(), raw=False)
    a.X = sparse.csr_matrix(a.X.toarray() + 0.5)
    with pytest.warns(PairedReadWarning, match="non-integer"):
        io.read(a, _airr([a.obs_names[0]]), counts="X")


def test_swapped_arguments_are_named(tmp_path):
    a, _ = _gex(_grid())
    g, t = _write(tmp_path, a, _airr([a.obs_names[0], a.obs_names[1]]))
    with pytest.raises(ValueError, match="gex and airr look swapped") as e:
        io.read(t, g)  # receptor table first, the GEX h5ad second
    assert "spaclone.read('gex.h5ad', 'airr.tsv')" in str(e.value)
    with pytest.raises(ValueError, match="HDF5 file without receptor data") as e:
        io.read(g, g)  # a GEX h5ad is not a receptor table either way
    assert "swapped" not in str(e.value)
    (tmp_path / "store.zarr").mkdir()
    (tmp_path / "store.zarr" / ".zgroup").write_text("{}")
    with pytest.raises(ValueError, match="zarr store without receptor data"):
        io.read(g, tmp_path / "store.zarr")
    _airr([a.obs_names[0]]).to_parquet(tmp_path / "meta.parquet")
    with pytest.raises(ValueError, match=r"Parquet file: read a Stereo-XCR-seq meta table with .*read_stereo_xcr"):
        io.read(g, tmp_path / "meta.parquet")


def test_scirpy_h5ad_file_as_receptor_argument(tmp_path):
    src = Path(__file__).parent / "_data" / "formats" / "scirpy" / "wu2020_40_v0_6.h5ad"
    a = ad.read_h5ad(src)
    a.obsm["spatial"] = np.column_stack([np.arange(a.n_obs), np.zeros(a.n_obs)]).astype(float) * 10.0
    a.write_h5ad(tmp_path / "gex.h5ad")
    sa = io.read(tmp_path / "gex.h5ad", src, unit_um=1.0, counts="X")  # an HDF5 receptor file is not sniffed as text
    assert sa.meta["io"]["receptors"]["format"] == "scirpy_legacy" and sa.air.spot_id.isin(a.obs_names).all()


def test_implausible_units_warn_and_name_unit_um(tmp_path):
    a, _ = _gex(_grid())
    g, t = _write(tmp_path, a, _airr([a.obs_names[0]]))
    with pytest.warns(PairedReadWarning, match=r"units 1e\+04 µm apart.*check unit_um"):
        sa = io.read(g, t, unit_um=500)  # nm given instead of µm per DNB
    assert not sa.meta["unit_check"]["plausible"] and sa.meta["unit_check"]["spacing_um"] == 10_000
    two = np.array([[0, 0], [10, 0], [60_000, 0], [60_010, 0]], dtype=float)  # neighbours 10 µm apart, 6 cm span
    b, _ = _gex(two, ids=[f"bin{i}" for i in range(len(two))])
    with pytest.warns(PairedReadWarning, match=r"give a section 6 cm wide \(expected < 5 cm\): check unit_um"):
        io.read(b, _airr(["bin0"]), unit_um=1.0)
    ok = io.read(g, t)  # the Stereo-seq bin20 fixture: 10 µm apart, no warning
    assert ok.meta["unit_check"] == {"spacing_um": 10.0, "width_um": 30.0, "plausible": True}


def test_missing_columns_are_named(tmp_path):
    a, _ = _gex(_grid())
    air = _airr([a.obs_names[0], a.obs_names[1]])
    g, t = _write(tmp_path, a, air.rename(columns={"cell_id": "bin_xy"}), index_col=False)
    with pytest.raises(KeyError, match="spot_key") as e:
        io.read(g, t)
    assert "'bin_xy'" in str(e.value) and "'v_germline_alignment'" in str(e.value)  # every column of a small table
    _, t2 = _write(tmp_path, a, air.rename(columns={"junction_aa": "cdr3_aa_seq"}), index_col=False)
    with pytest.raises(ValueError, match=r"no junction_aa column.*columns=\{'junction_aa': 'cdr3_aa_seq'\}"):
        io.read(g, t2)
    sa = io.read(g, t2, columns={"junction_aa": "cdr3_aa_seq"})  # the suggested fix works
    assert sa.air.junction_aa.tolist() == air.junction_aa.tolist()
    with pytest.raises(ValueError, match="not GEX units") as e:
        io.read(g, _write(tmp_path, a, _airr([a.obs_names[0], "1100-2100"]))[1], unmatched="error")
    assert "['1100-2100']" in str(e.value) and "np.str_" not in str(e.value)


def test_spatialair_repr_is_one_short_line(tmp_path):
    a, _ = _gex(_grid())
    g, t = _write(tmp_path, a, _airr([a.obs_names[0], a.obs_names[0], a.obs_names[5]]))
    r = repr(io.read(g, t))
    assert r.startswith("SpatialAIR(3 records (IGH 3); library airr; GEX 12 units x 5 genes; 0.5 µm per ")
    assert "reader read" in r and "\n" not in r and len(r) < 300 and "io" not in r.split("reader")[0]
