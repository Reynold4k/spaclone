import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex
from spaclone.gex.qc import CountsWarning

matplotlib.use("Agg")


def _adata(x, xy, genes=None, obs=None):
    x = np.asarray(x, dtype=np.float32)
    genes = genes or [f"g{i}" for i in range(x.shape[1])]
    a = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=obs if obs is not None else pd.DataFrame(index=[f"u{i}" for i in range(len(x))]),
        var=pd.DataFrame(index=genes),
    )
    a.obsm["spatial"] = np.asarray(xy, dtype=float)
    return a


def test_qc_metrics_known_values_and_tissue_split():
    x = [[10, 0, 0], [4, 4, 2], [0, 1, 0], [0, 0, 0]]
    a = _adata(x, [[0, 0], [10, 0], [20, 0], [30, 0]], genes=["Cd3e", "Ms4a1", "mt-Co1"])
    a.obs["in_tissue"] = [True, True, False, False]
    s = gex.qc_metrics(a)
    np.testing.assert_array_equal(a.obs.total_counts.to_numpy(), [10, 10, 1, 0])
    np.testing.assert_array_equal(a.obs.n_genes_by_counts.to_numpy(), [1, 3, 1, 0])
    np.testing.assert_allclose(a.obs.pct_counts_mt.to_numpy()[:2], [0.0, 20.0])
    assert a.var.mt.tolist() == [False, False, True]
    s = s.set_index("tissue")
    assert s.loc["in_tissue", "n_units"] == 2 and s.loc["off_tissue", "n_units"] == 2
    assert s.loc["in_tissue", "share_of_counts"] == pytest.approx(20 / 21)


def test_call_tissue_counts_and_neighbours():
    # a 5 x 5 block of rich units (10 um apart), one rich unit far away, one poor unit inside the block
    gx, gy = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    xy = np.column_stack([gx.ravel(), gy.ravel()]) * 10.0
    xy = np.vstack([xy, [[500.0, 500.0]]])
    x = np.full((len(xy), 1), 200.0)
    x[12] = 5  # centre unit is poor
    a = _adata(x, xy)
    tab = gex.call_tissue(a, min_counts=100, radius_um=20.0, min_neighbors=3)
    t = a.obs.in_tissue.to_numpy()
    assert not t[12] and not t[-1]
    # corner (0,0) has rich neighbours at 10, 10, 14.1, 20, 20 um -> 5 >= 3
    assert t[0]
    assert tab.n_tissue.iloc[0] == 24 - 0 and tab.n_min_counts.iloc[0] == 25
    # inclusive radius: exactly 3 other rich units within 10 um of the corner when min_neighbors=3 and radius 10
    gex.call_tissue(a, min_counts=100, radius_um=10.0, min_neighbors=3, key_added="t10")
    assert not a.obs.t10.to_numpy()[0]  # corner has only 2 units at 10 um
    assert a.obs.t10.to_numpy()[6]  # interior unit (1,1) has 4 at 10 um (one is the poor centre? no: (1,2),(2,1) rich)


def test_call_tissue_is_row_order_and_translation_invariant():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 300, size=(300, 2))
    x = rng.poisson(150, size=(300, 2)).astype(float)
    a = _adata(x, xy)
    gex.call_tissue(a, min_counts=290, radius_um=40.0, min_neighbors=5)
    perm = rng.permutation(300)
    b = _adata(x[perm], xy[perm] + 1234.5)
    b.obs_names = a.obs_names[perm]
    gex.call_tissue(b, min_counts=290, radius_um=40.0, min_neighbors=5)
    assert (b.obs.in_tissue == a.obs.in_tissue.loc[b.obs_names]).all()


def test_filter_keeps_off_tissue_units_unless_asked():
    x = [[200, 0], [5, 0], [5, 0], [300, 1]]
    a = _adata(x, [[0, 0], [1, 0], [2, 0], [3, 0]])
    a.obs["in_tissue"] = [True, True, False, False]
    keep = gex.filter(a, min_counts=50)
    assert keep.obs_names.tolist() == ["u0", "u2", "u3"]
    rep = keep.uns["spaclone"]["filter"]
    assert rep["removed_min_counts"] == 1 and rep["n_off_tissue_kept"] == 2
    filt = gex.filter(a, min_counts=50, off_tissue="filter")
    assert filt.obs_names.tolist() == ["u0", "u3"]
    drop = gex.filter(a, min_counts=50, off_tissue="drop")
    assert drop.obs_names.tolist() == ["u0"]
    genes = gex.filter(a, min_units_per_gene=2)
    assert genes.n_vars == 1
    with pytest.raises(ValueError):
        gex.filter(a, off_tissue="maybe")


def test_bin_sums_counts_with_stereo_convention():
    # DNB coordinates (0.5 um pitch): bin20 = 10 um, so DNB x 0..19 -> bin 0, 20..39 -> bin 1
    dnb = np.array([[0, 0], [19, 19], [20, 0], [39, 5], [45, 45]])
    xy = dnb * 0.5
    x = np.array([[1, 0], [2, 1], [3, 0], [4, 2], [5, 5]])
    a = _adata(x, xy)
    a.obs["in_tissue"] = [True, True, False, False, True]
    b = gex.bin(a, bin_um=10.0)
    ix, iy = gex.bin_ids(xy[:, 0], xy[:, 1], bin_um=10.0)
    np.testing.assert_array_equal(ix, dnb[:, 0] // 20)
    np.testing.assert_array_equal(iy, dnb[:, 1] // 20)
    assert b.obs_names.tolist() == ["0_0", "1_0", "2_2"]
    np.testing.assert_array_equal(b.X.toarray(), [[3, 1], [7, 2], [5, 5]])
    np.testing.assert_array_equal(b.obs.n_units, [2, 2, 1])
    np.testing.assert_allclose(b.obsm["spatial"], [[5, 5], [15, 5], [25, 25]])
    assert b.obs.in_tissue.tolist() == [True, False, True]
    assert a.obs.bin_id.astype(str).tolist() == ["0_0", "0_0", "1_0", "1_0", "2_2"]
    m = gex.bin(a, bin_um=10.0, position="mean", key_added=None)
    np.testing.assert_allclose(m.obsm["spatial"][0], xy[:2].mean(axis=0))


def test_bin_never_mixes_libraries():
    xy = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
    a = _adata(np.ones((3, 1)), xy)
    a.obs["lib"] = ["s1", "s1", "s2"]
    b = gex.bin(a, bin_um=10.0, library_key="lib")
    assert b.obs_names.tolist() == ["s1|0_0", "s2|0_0"]
    np.testing.assert_array_equal(b.X.toarray().ravel(), [2, 1])


def test_non_integer_counts_warn():
    a = _adata([[0.5, 1.2], [1.0, 0.0]], [[0, 0], [1, 1]])
    with pytest.warns(CountsWarning):
        gex.call_tissue(a, min_counts=0, radius_um=5, min_neighbors=0)


def test_gex_map_draws_and_returns_axes():
    import matplotlib.pyplot as plt

    from spaclone import pl

    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 200, size=(200, 2))
    a = _adata(rng.poisson(3, size=(200, 3)), xy, genes=["Cd3e", "Ms4a1", "Jchain"])
    a.obs["cell_type"] = pd.Categorical(rng.choice(["B", "T", "ASC", "unassigned"], 200))
    ax = pl.gex_map(a, "cell_type", style="dark")
    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("B (") for t in labels)
    ax2 = pl.gex_map(a, "Ms4a1")
    assert ax2.collections
    plt.close("all")
