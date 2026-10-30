import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spaclone import io, validate_air
from spaclone.pp import filter_valid


def _cells():
    obs = pd.DataFrame(
        {
            "x": np.array([10, 20, 30, 40, 50], np.uint32),
            "y": np.array([1, 2, 3, 4, 5], np.uint32),
            "IGHcdr3aa": pd.Categorical(["CARDYW", None, "CARDYW", None, "CAR*W"]),
            "IGK/Lcdr3aa": pd.Categorical([None, "CQQYF", "CQQYF", None, None]),
            "IGKorIGL": pd.Categorical([None, "IGK", "IGL", None, None]),
            "TRAcdr3aa": pd.Categorical([None, None, None, "CAVRDSNYQLIW", None]),
            "TRBcdr3aa": pd.Categorical([None] * 5),
            "anno": pd.Categorical(["b_cell", "b_cell", "plasma_cell", "Tcell", "b_cell"]),
            "mutation_percent": [1.5, np.nan, 7.0, np.nan, 0.0],
        },
        index=[str(i) for i in range(5)],
    )
    return ad.AnnData(X=np.eye(5, 3, dtype=np.float32), obs=obs)


def _write_h5ad(a, path):
    """GEO's LUAD files have '/' in obs names (IGK/Lcdr3aa); recent anndata refuses to write them by default."""
    override = getattr(getattr(ad, "settings", None), "override", None)
    if override is None:
        a.write_h5ad(path)
        return
    try:
        ctx = override(disallow_forward_slash_in_h5ad=False)
    except (TypeError, ValueError, AttributeError):
        a.write_h5ad(path)
        return
    with ctx:
        a.write_h5ad(path)


def test_luad_layout_records_and_flags():
    sa = io.from_cell_calls(_cells(), cell_type_key="anno", record_columns={"IGH": ["mutation_percent"]})
    air = sa.air
    validate_air(air)
    assert len(air) == 6
    assert air.locus.tolist() == ["IGH", "IGH", "IGH", "IGK", "IGL", "TRA"]
    assert air.clone_id.iloc[0] == "IGH|||CARDYW"
    assert air.x_um.tolist()[:2] == [5.0, 15.0]
    assert air.cell_type.tolist()[:3] == ["b_cell", "plasma_cell", "b_cell"]
    assert air.consensus_count.isna().all() and "productive" not in air.columns
    assert air.mutation_percent.isna().tolist() == [False, False, False, True, True, True]
    m = sa.meta
    assert m["level"] == "cell_call" and m["molecule_level"] is False and "molecule-level tests" in m["note"]
    assert m["n_cells"] == 5 and m["n_cells_with_call"] == 5
    # BCR cells (heavy or light call), as counted in the archive's QC table
    assert air[air.locus.str.startswith("IG")].spot_id.nunique() == 4
    assert sa.gex.n_obs == 5 and sa.gex.obs.cell_type.tolist()[2] == "plasma_cell"


def test_validity_filter_skips_the_unknown_productive_flag():
    ok, rep = filter_valid(io.from_cell_calls(_cells()).air, require_productive=True)
    assert len(ok) == 5 and not rep.applied.iloc[-1]


def test_path_backed_and_spatial_key(tmp_path):
    a = _cells()
    a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy(float) * 2
    p = tmp_path / "GSM9999999_CHIP42.h5ad"
    _write_h5ad(a, p)
    sa = io.from_cell_calls(p, keep_x=False, spatial_key="spatial", scale_um=0.25)
    assert sa.gex.n_vars == 0 and sa.meta["library_id"] == "CHIP42"
    np.testing.assert_allclose(sa.air.x_um.to_numpy()[:2], [5.0, 15.0])
    full = io.from_cell_calls(p, library_id="L1")
    assert full.gex.n_vars == 3 and full.air.library_id.eq("L1").all()


def test_custom_calls_and_errors():
    a = _cells()
    sa = io.from_cell_calls(a, calls={"TRA": "TRAcdr3aa"}, locus_columns=None)
    assert len(sa.air) == 1 and sa.air.locus.iloc[0] == "TRA"
    with pytest.raises(ValueError, match="not a locus"):
        io.from_cell_calls(a, calls={"light": "IGK/Lcdr3aa"}, locus_columns=None)
    with pytest.raises(ValueError, match="not in obs"):
        io.from_cell_calls(a, calls={"IGH": "nope"})
    b = a.copy()
    b.obs["IGKorIGL"] = pd.Categorical([None, "IGX", "IGL", None, None])
    with pytest.raises(ValueError, match="no valid locus"):
        io.from_cell_calls(b)
    with pytest.raises(ValueError, match="coordinate columns"):
        io.from_cell_calls(a, x_key="cx")
