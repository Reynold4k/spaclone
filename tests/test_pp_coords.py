import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spaclone.io import SpatialAIR
from spaclone.pp import concat_sections, transform, transform_matrix


def _air():
    return pd.DataFrame(
        {
            "spot_id": ["a", "b", "c"],
            "x_um": [1.0, 0, 3],
            "y_um": [0.0, 2, 4],
            "locus": "IGH",
            "clone_id": ["x", "y", "x"],
        }
    )


def test_rotation_flip_scale_affine():
    t = transform(_air(), rotate_deg=90)
    np.testing.assert_allclose(t[["x_um", "y_um"]].to_numpy(), [[0, 1], [-2, 0], [-4, 3]], atol=1e-12)
    np.testing.assert_allclose(transform(_air(), flip="x").x_um, [-1, 0, -3])
    np.testing.assert_allclose(transform(_air(), scale=0.5).y_um, [0, 1, 2])
    np.testing.assert_allclose(transform(_air(), scale=(2, 1)).x_um, [2, 0, 6])
    np.testing.assert_allclose(transform(_air(), affine=[[1, 0, 10], [0, 1, -1]]).y_um, [-1, 1, 3])
    o = transform(_air(), rotate_deg=180, origin_um=(1, 0))
    np.testing.assert_allclose(o[["x_um", "y_um"]].to_numpy(), [[1, 0], [2, -2], [-1, -4]], atol=1e-12)
    M = transform_matrix(scale=2, rotate_deg=30, flip="y", affine=np.eye(3))
    assert M.shape == (3, 3) and np.allclose(M[2], [0, 0, 1])
    hist = transform(transform(_air(), scale=2), rotate_deg=10).attrs["spaclone"]["transforms"]
    assert len(hist) == 2 and hist[0]["scale"] == 2


def test_rigid_transforms_keep_distances():
    rng = np.random.default_rng(0)
    air = pd.DataFrame(
        {"spot_id": np.arange(50).astype(str), "x_um": rng.uniform(0, 100, 50), "y_um": rng.uniform(0, 100, 50)}
    )
    t = transform(air, rotate_deg=37, flip="x", origin_um=(50, 50))
    d0 = np.hypot(*(air[["x_um", "y_um"]].to_numpy()[:, None] - air[["x_um", "y_um"]].to_numpy()[None]).T)
    d1 = np.hypot(*(t[["x_um", "y_um"]].to_numpy()[:, None] - t[["x_um", "y_um"]].to_numpy()[None]).T)
    np.testing.assert_allclose(d0, d1, atol=1e-9)


def test_transform_anndata_and_spatialair():
    g = ad.AnnData(obs=pd.DataFrame(index=["a", "b"]), obsm={"spatial": np.array([[1.0, 0], [0, 2]])})
    tg = transform(g, scale=3)
    np.testing.assert_allclose(tg.obsm["spatial"], [[3, 0], [0, 6]])
    assert "transforms" in tg.uns["spaclone"]
    sp = transform(SpatialAIR(_air(), g, {"technology": "toy"}), rotate_deg=90)
    np.testing.assert_allclose(sp.gex.obsm["spatial"], [[0, 1], [-2, 0]], atol=1e-12)
    assert sp.meta["technology"] == "toy" and len(sp.meta["transforms"]) == 1
    with pytest.raises(ValueError):
        transform(_air(), scale=0)
    with pytest.raises(ValueError):
        transform(_air(), affine=np.ones((2, 2)))
    with pytest.raises(TypeError):
        transform([1, 2])


def test_concat_sections():
    c = concat_sections([_air(), _air()], keys=["s1", "s2"])
    assert len(c) == 6 and c.library_id.tolist() == ["s1"] * 3 + ["s2"] * 3
    assert c.index.equals(pd.RangeIndex(6)) and (c.x_um.to_numpy()[:3] == c.x_um.to_numpy()[3:]).all()
    u = concat_sections([_air(), _air()], keys=["s1", "s2"], index_unique="-")
    assert u.spot_id.tolist()[0] == "a-s1"
    k = concat_sections([_air().assign(library_id="L1"), _air().assign(library_id="L2")])
    assert set(k.library_id) == {"L1", "L2"}
    with pytest.raises(ValueError):
        concat_sections([_air(), _air()])
    g = ad.AnnData(obs=pd.DataFrame(index=["a", "b", "c"]), obsm={"spatial": np.zeros((3, 2))})
    sp = concat_sections(
        [SpatialAIR(_air(), g, {}), SpatialAIR(_air(), g.copy(), {})], keys=["s1", "s2"], index_unique="-"
    )
    assert sp.gex.n_obs == 6 and "a-s1" in sp.gex.obs_names and set(sp.gex.obs.library_id) == {"s1", "s2"}
    assert set(sp.air.spot_id) == set(sp.gex.obs_names)
    with pytest.raises(ValueError):
        concat_sections([SpatialAIR(_air(), g, {}), SpatialAIR(_air(), g.copy(), {})], keys=["s1", "s2"])
