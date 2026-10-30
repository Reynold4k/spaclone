import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex


def _planted(n_per=120, n_genes=60, seed=0):
    """Three planted cell types (each with its own 10-gene programme) plus Ig and mitochondrial genes."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)] + ["Ighm", "Igkc", "Ighv1-2", "Trbv1", "mt-Co1", "Rpl13a", "Hbb-bs"]
    types = np.repeat(["A", "B", "C"], n_per)
    lam = np.full((len(types), len(genes)), 0.4)
    for k, t in enumerate("ABC"):
        lam[types == t, k * 10 : (k + 1) * 10] = 6.0
    lam[:, n_genes:] = rng.gamma(1.0, 8.0, size=(len(types), len(genes) - n_genes))  # noisy, very variable genes
    x = rng.poisson(lam).astype(np.float32)
    a = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=pd.DataFrame({"truth": types}, index=[f"u{i}" for i in range(len(types))]),
        var=pd.DataFrame(index=genes),
    )
    a.obsm["spatial"] = rng.uniform(0, 500, size=(len(types), 2))
    return a


def test_gene_groups_match_ig_tr_and_housekeeping_names():
    names = [
        "Ighm",
        "Ighd",
        "IGHD3-10",
        "Ighv1-2",
        "Ighj1",
        "Igkc",
        "IGLC2",
        "Igll1",
        "Ighg2b",
        "IGHA1",
        "Trbv13-1",
        "TRAJ12",
        "Trbd1",
        "Trac",
        "mt-Co1",
        "MT-ND1",
        "Rpl13a",
        "RPS4X",
        "Mrpl15",
        "Hbb-bs",
        "HBA1",
        "Hbegf",
        "IGHMBP2",
        "Cd79a",
        "Traf1",
        "Igfbp7",
    ]
    m = gex.gene_group_mask(names, ("ig", "tr", "mt", "ribo", "hb"))
    hit = {n for n, v in zip(names, m, strict=True) if v}
    assert hit == {
        "Ighm",
        "Ighd",
        "IGHD3-10",
        "Ighv1-2",
        "Ighj1",
        "Igkc",
        "IGLC2",
        "Igll1",
        "Ighg2b",
        "IGHA1",
        "Trbv13-1",
        "TRAJ12",
        "Trbd1",
        "mt-Co1",
        "MT-ND1",
        "Rpl13a",
        "RPS4X",
        "Mrpl15",
        "Hbb-bs",
        "HBA1",
    }
    assert not gex.gene_group_mask(names, None).any()
    assert gex.gene_group_mask(names, [r"^Cd79.*$"]).sum() == 1


def test_preprocess_keeps_counts_excludes_groups_and_is_idempotent():
    a = _planted()
    raw = a.X.copy()
    gex.preprocess(a, n_hvg=20, n_pcs=5, min_gene_frac=0.0)
    assert (a.layers["counts"] != raw).nnz == 0
    np.testing.assert_allclose(np.expm1(a.X.toarray()).sum(axis=1), 1e4, rtol=1e-4)
    hv = a.var.highly_variable
    assert hv.sum() == 20
    assert not hv[["Ighm", "Igkc", "Ighv1-2", "Trbv1", "mt-Co1", "Rpl13a", "Hbb-bs"]].any()
    p1 = a.obsm["X_pca"].copy()
    gex.preprocess(a, n_hvg=20, n_pcs=5, min_gene_frac=0.0)
    np.testing.assert_allclose(np.abs(a.obsm["X_pca"]), np.abs(p1), rtol=1e-4, atol=1e-4)
    rec = a.uns["spaclone"]["preprocess"]
    assert rec["n_hvg_selected"] == 20 and rec["n_pcs"] == 5


def test_preprocess_gene_floor_removes_rare_genes_from_hvgs():
    a = _planted()
    x = a.X.toarray()
    x[:, 0] = 0
    x[0, 0] = 50  # detected in a single unit (< 0.5%)
    a.X = sparse.csr_matrix(x)
    gex.preprocess(a, n_hvg=10, n_pcs=5)
    assert not a.var.loc["G0", "hvg_eligible"]


def test_cluster_recovers_planted_types_and_is_deterministic():
    from sklearn.metrics import adjusted_rand_score

    a = _planted()
    gex.preprocess(a, n_hvg=30, n_pcs=10, min_gene_frac=0.0)
    sizes = gex.cluster(a, resolution=0.3, rng=0)
    assert sizes.n_units.sum() == a.n_obs
    assert adjusted_rand_score(a.obs.truth, a.obs.leiden) > 0.9
    first = a.obs.leiden.astype(str).to_numpy().copy()
    gex.cluster(a, resolution=0.3, rng=0, recompute_neighbors=True)
    assert (a.obs.leiden.astype(str).to_numpy() == first).all()
    gex.embed(a, rng=0)
    assert a.obsm["X_umap"].shape == (a.n_obs, 2)
    with pytest.raises(ValueError):
        gex.cluster(a, method="kmeans")


def test_preprocess_cluster_embed_identical_whatever_the_thread_count():
    """Same rng => bit-identical PCs, neighbour graph, Leiden labels and UMAP with 1 or 3 BLAS threads set by the
    caller (multi-threaded sums made AIR-SPACE D21 give 11 Leiden clusters with 4 threads and 9 with 2); the steps
    run with one thread and give the caller's thread settings back; knn="auto" builds the exact graph."""
    import numba
    from threadpoolctl import threadpool_info, threadpool_limits

    base = _planted(n_per=300, seed=5)
    n_numba = numba.get_num_threads()
    runs = []
    for limit in (1, 3):
        a = base.copy()
        with threadpool_limits(limits=limit):
            gex.preprocess(a, n_hvg=30, n_pcs=10, min_gene_frac=0.0, rng=0)
            gex.cluster(a, resolution=0.5, rng=0)
            gex.embed(a, rng=0)
            blas = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
        assert blas and all(t == limit for t in blas)  # the caller's limit is back after the calls
        runs.append(a)
    a, b = runs
    np.testing.assert_array_equal(a.obsm["X_pca"], b.obsm["X_pca"])
    assert (a.obsp["connectivities"] != b.obsp["connectivities"]).nnz == 0
    np.testing.assert_array_equal(a.obs.leiden.astype(str).to_numpy(), b.obs.leiden.astype(str).to_numpy())
    np.testing.assert_array_equal(a.obsm["X_umap"], b.obsm["X_umap"])
    assert a.uns["spaclone"]["neighbors"]["knn"] == "exact" and a.uns["spaclone"]["cluster_leiden"]["knn"] == "exact"
    assert numba.get_num_threads() == n_numba  # numba threads given back too
    with pytest.raises(ValueError, match="knn"):
        gex.cluster(a, knn="fast")


@pytest.mark.slow
def test_approximate_neighbour_graph_is_seeded_and_rebuilt_on_request():
    """knn="approximate" (nearest-neighbour descent, one thread, seeded) gives the same labels twice, and a stored
    exact graph is rebuilt rather than reused when another kNN method is asked for (slow: compiles pynndescent)."""
    a = _planted(n_per=300, seed=5)
    gex.preprocess(a, n_hvg=30, n_pcs=10, min_gene_frac=0.0, rng=0)
    gex.cluster(a, resolution=0.5, rng=0)
    gex.cluster(a, resolution=0.5, rng=0, knn="approximate")  # the stored graph is exact: rebuilt
    assert a.uns["spaclone"]["neighbors"]["knn"] == "approximate"
    first = a.obs.leiden.astype(str).to_numpy().copy()
    gex.cluster(a, resolution=0.5, rng=0, knn="approximate", recompute_neighbors=True)
    np.testing.assert_array_equal(a.obs.leiden.astype(str).to_numpy(), first)
