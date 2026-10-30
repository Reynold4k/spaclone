import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex


def _grid(labels, n_side, spacing=10.0, n_genes=3):
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij"), -1).reshape(-1, 2) * spacing
    a = ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(g), n_genes), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"u{i}" for i in range(len(g))]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    a.obsm["spatial"] = g
    a.obs["cl"] = pd.Categorical(labels)
    return a


def test_spatial_graph_radius_and_knn():
    a = _grid(["a"] * 9, 3)
    r = gex.spatial_graph(a, radius_um=10.0)
    assert r.nnz == 24  # 12 edges of a 3 x 3 lattice, both directions
    assert (r != r.T).nnz == 0 and r.diagonal().sum() == 0
    k = gex.spatial_graph(a, n_neighbors=1)
    assert (k != k.T).nnz == 0 and k.diagonal().sum() == 0
    a.obs["lib"] = ["s1"] * 4 + ["s2"] * 5
    rl = gex.spatial_graph(a, radius_um=10.0, library_key="lib")
    lib = a.obs.lib.to_numpy()
    rows, cols = rl.nonzero()
    assert (lib[rows] == lib[cols]).all()


def test_nhood_counts_known_answer():
    # a 1 x 4 strip a a b b: edges a-a 1, a-b 1, b-b 1 (each counted in both directions as squidpy does)
    a = ad.AnnData(
        X=sparse.csr_matrix(np.ones((4, 1), dtype=np.float32)),
        obs=pd.DataFrame({"cl": pd.Categorical(list("aabb"))}, index=list("wxyz")),
        var=pd.DataFrame(index=["G"]),
    )
    a.obsm["spatial"] = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    r = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, n_perm=50, rng=0)
    t = r.table.set_index(["cluster_a", "cluster_b"])
    assert t.loc[("a", "a"), "statistic"] == 2 and t.loc[("a", "b"), "statistic"] == 1
    assert t.loc[("b", "b"), "statistic"] == 2
    assert (r.table.p_value > 0).all() and (r.table.p_value <= 1).all()


def test_nhood_detects_segregation_and_is_deterministic():
    n = 20
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    lab = np.where(g[:, 0] < n // 2, "B", "T")
    a = _grid(lab, n)
    r = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, n_perm=199, rng=1, keep_null=True)
    bt = r.table[(r.table.cluster_a == "B") & (r.table.cluster_b == "T")].iloc[0]
    assert bt.z < -5 and bt.p_less == pytest.approx(1 / 200)
    assert r.null.shape == (199, 3)
    r2 = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, n_perm=199, rng=1)
    pd.testing.assert_frame_equal(r.table, r2.table)
    r3 = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, n_perm=199, rng=2)
    assert not np.allclose(
        r.table.null_mean.to_numpy(), r3.table.null_mean.to_numpy(), rtol=0, atol=0
    ) or not np.allclose(r.table.null_sd.to_numpy(), r3.table.null_sd.to_numpy(), rtol=0, atol=0)


def test_nhood_calibrated_under_random_labelling():
    """Null model: labels drawn independently of position; the false-positive rate of between-class pairs must stay
    near alpha."""
    rng = np.random.default_rng(5)
    n_side, hits, tests = 16, 0, 0
    for _ in range(40):
        lab = rng.choice(["A", "B", "C"], n_side * n_side)
        a = _grid(lab, n_side)
        r = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, n_perm=99, rng=rng)
        between = r.table[r.table.cluster_a != r.table.cluster_b]
        hits += int((between.p_value < 0.05).sum())
        tests += len(between)
    alpha = 0.05
    assert hits / tests <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / tests)


def test_nhood_respects_libraries():
    # two sections, each internally segregated in opposite directions; permutation stays within a section
    n = 12
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    lab1 = np.where(g[:, 0] < n // 2, "B", "T")
    lab2 = np.where(g[:, 0] < n // 2, "T", "B")
    a = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2 * len(g), 1), dtype=np.float32)),
        obs=pd.DataFrame(
            {"cl": pd.Categorical(np.concatenate([lab1, lab2])), "lib": ["s1"] * len(g) + ["s2"] * len(g)},
            index=[f"u{i}" for i in range(2 * len(g))],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    a.obsm["spatial"] = np.vstack([g, g]) * 10.0
    r = gex.nhood_enrichment(a, cluster_key="cl", radius_um=10.0, library_key="lib", n_perm=99, rng=0)
    assert r.params["n_edges"] == 2 * 2 * 2 * n * (n - 1)


@pytest.mark.slow
def test_nhood_zscore_matches_squidpy():
    sq = pytest.importorskip("squidpy")
    n = 20
    rng = np.random.default_rng(3)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    lab = np.where(g[:, 0] < n // 2, "B", "T")
    lab[rng.random(len(lab)) < 0.3] = "M"
    a = _grid(lab, n)
    sq.gr.spatial_neighbors(a, coord_type="generic", radius=10.0)
    zs = sq.gr.nhood_enrichment(a, cluster_key="cl", n_perms=2000, rng=0, copy=True, show_progress_bar=False)
    z_sq = np.asarray(zs.zscore if hasattr(zs, "zscore") else zs[0], dtype=float)
    r = gex.nhood_enrichment(a, cluster_key="cl", connectivity_key="spatial_connectivities", n_perm=2000, rng=0)
    cats = list(a.obs.cl.cat.categories)
    for row in r.table.itertuples():
        i, j = cats.index(row.cluster_a), cats.index(row.cluster_b)
        assert row.z == pytest.approx(z_sq[i, j], abs=0.25)


@pytest.mark.slow
def test_spatially_variable_genes_finds_a_gradient():
    pytest.importorskip("squidpy")
    n = 20
    rng = np.random.default_rng(4)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2) * 10.0
    x = np.column_stack([g[:, 0] / 10 + rng.normal(0, 1, len(g)), rng.normal(0, 1, len(g))])
    a = ad.AnnData(
        X=x.astype(np.float32),
        obs=pd.DataFrame(index=[f"u{i}" for i in range(len(g))]),
        var=pd.DataFrame(index=["gradient", "noise"]),
    )
    a.obsm["spatial"] = g
    r = gex.spatially_variable_genes(a, genes=["gradient", "noise"], n_neighbors=6, n_perm=99, rng=0)
    t = r.table.set_index("gene")
    assert t.loc["gradient", "statistic"] > 0.5 and t.loc["gradient", "p_value"] <= 0.02
    assert t.loc["noise", "p_value"] > 0.01
