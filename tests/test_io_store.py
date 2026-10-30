from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spaclone.io import SpatialAIR, read, write


def _sp():
    air = pd.DataFrame(
        {
            "spot_id": ["a", "b"],
            "x_um": [0.0, 1.5],
            "y_um": [2.0, 3.0],
            "locus": ["IGH", "TRB"],
            "clone_id": ["x", None],
            "consensus_count": [3.0, np.nan],
            "umis": pd.array([1, None], dtype="Int64"),
            "iso": pd.Categorical(["IgM", "IgG"]),
        },
        index=pd.Index([10, 20], name="rec"),
    )
    air.attrs["spaclone"] = {"filter_records": {"min_reads": 2}}
    gex = ad.AnnData(
        np.arange(6, dtype=float).reshape(2, 3),
        obs=pd.DataFrame(index=["a", "b"]),
        obsm={"spatial": np.array([[0.0, 2.0], [1.5, 3.0]])},
    )
    meta = {"technology": "toy", "factor_um": np.float64(0.5), "path": Path("/x/y"), "n": np.int64(3)}
    return SpatialAIR(air, gex, meta)


def test_write_read_round_trip(tmp_path):
    sp = _sp()
    p = write(sp, tmp_path / "s")
    assert sorted(f.name for f in p.iterdir()) == ["air.parquet", "gex.h5ad", "meta.json"]
    back = read(p)
    pd.testing.assert_frame_equal(back.air, sp.air)
    assert back.air.attrs == sp.air.attrs
    np.testing.assert_allclose(back.gex.X, sp.gex.X)
    np.testing.assert_allclose(back.gex.obsm["spatial"], sp.gex.obsm["spatial"])
    assert back.meta == {"technology": "toy", "factor_um": 0.5, "path": "/x/y", "n": 3}


def test_table_only_overwrite_and_errors(tmp_path):
    sp = _sp()
    write(sp.air, tmp_path / "t")
    back = read(tmp_path / "t")
    assert back.gex is None and back.meta == {}
    with pytest.raises(FileExistsError):
        write(sp, tmp_path / "t")
    write(sp, tmp_path / "t", overwrite=True)
    assert read(tmp_path / "t").gex is not None
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "nothing")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "meta.json").write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        read(tmp_path / "bad")
    with pytest.raises(TypeError):
        write([1], tmp_path / "u")
