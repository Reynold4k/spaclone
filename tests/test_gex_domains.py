import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex


def test_neighbour_mean_known_answer():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [100.0, 0.0]])
    v = np.array([[1.0], [2.0], [4.0], [8.0]])
    m, deg = gex.neighbour_mean(xy, v, radius_um=10.0)
    np.testing.assert_allclose(m.ravel(), [2.0, 2.5, 2.0, 8.0])
    np.testing.assert_array_equal(deg, [1, 2, 1, 0])
    xy2 = np.array([[0.0, 0.0], [10.0, 0.0], [25.0, 0.0], [100.0, 0.0]])  # no tied nearest neighbours
    k, _ = gex.neighbour_mean(xy2, v, radius_um=None, n_neighbors=1)
    np.testing.assert_allclose(k.ravel(), [2.0, 1.0, 2.0, 4.0])
    g, _ = gex.neighbour_mean(xy, v, radius_um=10.0, groups=np.array([0, 1, 1, 1]))
    np.testing.assert_allclose(g.ravel(), [1.0, 4.0, 2.0, 8.0])
    with pytest.raises(ValueError):
        gex.neighbour_mean(xy, v, radius_um=None)


def _two_domains(seed=0, n_side=36):
    """Left and right halves differ only in the *mixture* of two cell programmes, so single units are ambiguous but
    their neighbourhoods are not."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij"), -1).reshape(-1, 2) * 10.0
    left = g[:, 0] < n_side * 5
    p_a = np.where(left, 0.7, 0.3)
    is_a = rng.random(len(g)) < p_a
    n_genes = 40
    lam = np.full((len(g), n_genes), 1.0)
    lam[is_a, :10] += 3.0
    lam[~is_a, 10:20] += 3.0
    x = rng.poisson(lam).astype(np.float32)
    a = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=pd.DataFrame({"truth": np.where(left, "L", "R")}, index=[f"b{i}" for i in range(len(g))]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    a.obsm["spatial"] = g
    a.obs["in_tissue"] = True
    return a


def test_domains_recover_planted_halves_better_than_expression_alone():
    from sklearn.metrics import adjusted_rand_score

    a = _two_domains()
    out = gex.domains(a, radius_um=22.5, lambda_=0.8, resolution=0.1, n_hvg=20, n_pcs=8, min_gene_frac=0.0, rng=0)
    ari_spatial = adjusted_rand_score(a.obs.truth, a.obs.domain.astype(str))
    gex.domains(a, lambda_=0.0, resolution=0.1, n_hvg=20, n_pcs=8, min_gene_frac=0.0, rng=0, key_added="expr")
    ari_expr = adjusted_rand_score(a.obs.truth, a.obs.expr.astype(str))
    assert ari_spatial > 0.6
    assert ari_spatial > ari_expr + 0.3
    assert out.n_units.sum() == a.n_obs
    assert np.average(out.agreement, weights=out.n_units) > 0.6
    assert a.obsm["X_domain"].shape[0] == a.n_obs


def test_domains_are_deterministic_and_skip_off_tissue_units():
    a = _two_domains(seed=1, n_side=24)
    a.obs["in_tissue"] = a.obsm["spatial"][:, 1] < 200
    gex.domains(a, resolution=0.2, n_hvg=20, n_pcs=6, min_gene_frac=0.0, rng=3)
    first = a.obs.domain.astype(object).where(a.obs.domain.notna(), "nan").tolist()
    gex.domains(a, resolution=0.2, n_hvg=20, n_pcs=6, min_gene_frac=0.0, rng=3)
    assert a.obs.domain.astype(object).where(a.obs.domain.notna(), "nan").tolist() == first
    assert a.obs.domain[~a.obs.in_tissue.to_numpy()].isna().all()
    with pytest.raises(ValueError):
        gex.domains(a, lambda_=1.5)
