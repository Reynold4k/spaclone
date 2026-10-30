import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("awkward")

from spaclone import io
from spaclone.io import from_scirpy, to_scirpy

warnings.filterwarnings("ignore", message=".*Awkward Arrays.*")


def _air():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e", "e"],
            "x_um": [0.0, 0, 10, 20, 20, 30, 40, 40],
            "y_um": [0.0] * 8,
            "locus": ["IGH", "IGK", "IGH", "TRA", "TRB", "IGH", "IGK", "IGK"],
            "clone_id": ["h1", "k1", "h1", "a1", "b1", "h2", "k1", "k2"],
            "junction_aa": ["CARDYW", "CQQYF", None, "CAVRW", "CASSF", "CARDYW", "CQQYF", "CQQLF"],
            "productive": ["T", "T", "F", "T", "T", "T", "T", "F"],
            "consensus_count": [5, 1, 9, 3, 2, np.nan, 2, 7],
            "umis": pd.array([1, 2, None, 4, 5, 6, 7, 8], dtype="Int64"),
            "n": np.arange(8),
            "flag": [True, False, True, True, False, True, True, True],
            "isotype": pd.Categorical(["IgM", None, "IgG", None, None, "IgM", None, None]),
        }
    )


@pytest.mark.parametrize("level", ["spot", "locus", "cell"])
def test_round_trip_is_lossless(level, tmp_path):
    air = _air().assign(cell_id=["c1", "c1", "c2", "c3", "c3", "c4", "c5", "c5"])
    air.index = pd.Index([f"r{i}" for i in range(8)], name="rec")
    air.attrs["spaclone"] = {"technology": "toy"}
    adata = to_scirpy(air, level=level)
    back = from_scirpy(adata)
    pd.testing.assert_frame_equal(back, air)
    assert back.attrs == air.attrs
    adata.write_h5ad(tmp_path / "x.h5ad")
    import anndata as ad

    pd.testing.assert_frame_equal(from_scirpy(ad.read_h5ad(tmp_path / "x.h5ad")), air)


def test_round_trip_after_subsetting_observations():
    air = _air()
    adata = to_scirpy(air, level="spot")
    sub = adata[adata.obs.n_records == 2].copy()
    back = from_scirpy(sub)
    kept = air[air.spot_id.isin(["a", "c", "e"])]
    pd.testing.assert_frame_equal(back, kept)


def test_observations_positions_and_productive_flag():
    air = _air()
    s = to_scirpy(air, level="spot")
    assert s.n_obs == 5 and s.obs.n_records.tolist() == [2, 1, 2, 1, 2]
    np.testing.assert_allclose(s.obsm["spatial"][:, 0], [0, 10, 20, 30, 40])
    import awkward as ak

    assert str(ak.type(s.obsm["airr"]["productive"])).endswith("bool")
    loc = to_scirpy(air, level="locus", radius_um=15)
    assert set(loc.obs.clone_id) == set(air.clone_id) and loc.obs.n_records.sum() == 8
    two = pd.concat([air.assign(library_id="L1"), air.assign(library_id="L2")], ignore_index=True)
    assert to_scirpy(two, level="spot").n_obs == 10
    with pytest.raises(ValueError):
        to_scirpy(air, level="bead")
    with pytest.raises(ValueError):
        to_scirpy(air.assign(_spaclone_row=1), level="spot")


def _valid_air():
    return _air().assign(
        productive="T", junction_aa=["CARDYW", "CQQYF", "CARDYW", "CAVRW", "CASSF", "CARDYW", "CQQYF", "CQQLF"]
    )


def test_scirpy_clonal_expansion_runs():
    ir = pytest.importorskip("scirpy")
    a = to_scirpy(_valid_air(), level="locus")
    ir.tl.clonal_expansion(a)
    sizes = a.obs.groupby("clone_id", observed=True).size()
    assert (a.obs.clonal_expansion.astype(str) == "<= 1").sum() == (sizes == 1).sum()


@pytest.mark.slow
def test_scirpy_chain_indexing_and_clonotype_clusters_run():
    ir = pytest.importorskip("scirpy")
    a = to_scirpy(_valid_air(), level="locus")
    ir.pp.index_chains(a)
    ir.tl.chain_qc(a)
    ir.pp.ir_dist(a, metric="identity", sequence="aa")
    ir.tl.define_clonotype_clusters(a, sequence="aa", metric="identity", key_added="cc")
    assert a.obs.cc.notna().all()
    assert set(a.obs.chain_pairing) <= {"orphan VJ", "orphan VDJ"}


def test_from_scirpy_on_a_foreign_object():
    pytest.importorskip("scirpy")
    from scirpy.io import AirrCell, from_airr_cells

    cells = []
    for i, (locus, cdr3) in enumerate([("TRB", "CASSF"), ("TRA", "CAVRW"), ("TRB", "CASTF")]):
        c = AirrCell(cell_id=f"bead{i}")
        ch = AirrCell.empty_chain_dict()
        ch.update(locus=locus, junction_aa=cdr3, productive=True, consensus_count=3)
        c.add_chain(ch)
        cells.append(c)
    a = from_airr_cells(cells)
    with pytest.raises(ValueError):
        from_scirpy(a)
    a.obsm["spatial"] = np.array([[0.0, 1], [2, 3], [4, 5]])
    t = from_scirpy(a)
    assert t.spot_id.tolist() == ["bead0", "bead1", "bead2"] and t.y_um.tolist() == [1, 3, 5]


@pytest.mark.slow
def test_to_spatialdata():
    pytest.importorskip("spatialdata")
    sd = io.to_spatialdata(_air())
    assert len(sd.shapes["air_spots"]) == 5 and sd.tables["air_clones"].n_obs == 5
    assert len(sd.points["air_records"].compute()) == 8
    import anndata as ad

    g = ad.AnnData(np.ones((2, 3)), obs=pd.DataFrame(index=["a", "b"]), obsm={"spatial": np.array([[0.0, 0], [10, 0]])})
    sd2 = io.to_spatialdata(io.SpatialAIR(_air(), g, {}), bin_um=20)
    assert "gex" in sd2.tables and len(sd2.shapes["air_spots"]) == 3
