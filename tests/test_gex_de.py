import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse, special, stats

from spaclone import gex
from spaclone._utils import LowPowerWarning
from spaclone.gex.de import PseudoreplicationWarning, _fit_f_dist, _tmm, _trigamma_inverse

matplotlib.use("Agg")


def _regions(seed, n_gc=6, n_bg=8, units=25, n_genes=300, de_genes=0, lfc=1.0, inst_sd=0.3):
    """Units grouped in region instances (germinal centres and background patches); instance-level random effects
    make units of one instance correlated, as in tissue."""
    rng = np.random.default_rng(seed)
    n_inst = n_gc + n_bg
    mu = np.exp(rng.normal(0.0, 1.0, n_genes)) * 0.5
    eff = np.exp(rng.normal(0.0, inst_sd, size=(n_inst, n_genes)))
    is_gc = np.arange(n_inst) < n_gc
    eff[np.ix_(is_gc, np.arange(de_genes))] *= 2.0**lfc
    inst = np.repeat(np.arange(n_inst), units)
    size = rng.gamma(5.0, 0.2, len(inst))
    x = rng.poisson(mu[None, :] * eff[inst] * size[:, None]).astype(np.float32)
    region = np.where(is_gc[inst], "GC", "other")
    rid = np.array([f"{'GC' if is_gc[i] else 'BG'}_{i}" for i in inst])
    a = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=pd.DataFrame({"region": region, "region_id": rid}, index=[f"u{i}" for i in range(len(inst))]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]),
    )
    a.obsm["spatial"] = np.column_stack(
        [inst * 1050.0 + rng.uniform(0, 100, len(inst)), rng.uniform(0, 100, len(inst))]
    )
    a.obs["in_tissue"] = True
    return a


def test_pseudobulk_sums_units_of_each_instance():
    a = _regions(0, n_gc=2, n_bg=2, units=3, n_genes=4)
    pb = gex.pseudobulk(a, groupby=["region", "region_id"])
    assert pb.n_obs == 4
    first = pb.obs.index[0]
    reg, rid = first.split("|")
    m = (a.obs.region == reg) & (a.obs.region_id == rid)
    np.testing.assert_allclose(pb.X[0].toarray().ravel(), a.X[m.to_numpy()].toarray().sum(axis=0))
    assert (pb.obs.n_units == 3).all()
    assert gex.pseudobulk(a, groupby="region_id", min_units=4).n_obs == 0


def test_empirical_bayes_machinery():
    for x in (1e-3, 0.3, 2.0, 50.0):
        assert special.polygamma(1, _trigamma_inverse(x)) == pytest.approx(x, rel=1e-6)
    rng = np.random.default_rng(1)
    d0, s0, d = 8.0, 0.5, 4.0
    sigma2 = s0 * d0 / rng.chisquare(d0, 20000)
    s2 = sigma2 * rng.chisquare(d, 20000) / d
    s20, df2 = _fit_f_dist(s2, d)
    assert df2 == pytest.approx(d0, rel=0.15)
    assert s20 == pytest.approx(s0, rel=0.1)


def test_tmm_corrects_a_composition_shift():
    rng = np.random.default_rng(2)
    base = rng.poisson(50, 200).astype(float) + 1
    other = base.copy()
    other[:20] *= 10  # 10% of genes up tenfold in sample 2
    y = np.column_stack([base, other])
    lib = y.sum(axis=0)
    f = _tmm(y, lib)
    assert np.exp(np.mean(np.log(f))) == pytest.approx(1.0)
    eff = lib * f
    assert eff[1] / eff[0] == pytest.approx(1.0, rel=0.03)  # the unchanged genes get equal normalised levels


@pytest.mark.parametrize("test", ["limma", "welch", "nb"])
def test_calibrated_under_the_null(test):
    """Null model: no gene differs between GC and background instances (random instance effects only)."""
    hits, n = 0, 0
    for seed in range(3 if test != "nb" else 2):
        a = _regions(100 + seed, n_genes=300 if test != "nb" else 200)
        r = gex.rank_genes_regions(a, groups="GC", reference="other", test=test)
        hits += int((r.table.p_value < 0.05).sum())
        n += len(r.table)
    alpha = 0.05
    assert hits / n <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n)


@pytest.mark.parametrize("test", ["limma", "nb"])
def test_power_on_planted_differences(test):
    a = _regions(7, de_genes=30, lfc=1.5, n_genes=300 if test == "limma" else 200)
    r = gex.rank_genes_regions(a, groups="GC", reference="other", test=test)
    t = r.table
    called = set(t.gene[t.q_value < 0.05])
    planted = {f"g{j}" for j in range(30)} & set(t.gene)
    assert len(called & planted) / len(planted) >= 0.7
    assert len(called - planted) / max(len(called), 1) <= 0.15
    up = t.set_index("gene").loc[sorted(planted), "log2fc"]
    assert (up > 0).mean() > 0.9
    assert r.summary["n_instances[GC]"] == 6
    assert r.params["test"] == test


def test_missing_instances_become_tiles_and_low_power_warns():
    a = _regions(3, n_genes=50)
    a.obs.loc[a.obs.region == "other", "region_id"] = None
    r = gex.rank_genes_regions(a, groups="GC", reference="other", tile_um=150.0)
    assert r.table.n_instances_reference.iloc[0] == 8
    with pytest.raises(ValueError, match="no instance id"):
        gex.rank_genes_regions(a, groups="GC", reference="other", missing_instances="error")
    b = _regions(4, n_gc=1, n_genes=50)
    with pytest.warns(LowPowerWarning):
        r1 = gex.rank_genes_regions(b, groups="GC", reference="other")
    assert r1.table.empty


def test_blocking_factor_enters_the_design():
    a = _regions(5, n_gc=6, n_bg=6, n_genes=80)
    inst = a.obs.region_id.str.split("_").str[1].astype(int)
    a.obs["section"] = np.where(inst % 2 == 0, "s1", "s2")
    r = gex.rank_genes_regions(a, groups="GC", reference="other", sample_key="section")
    assert len(r.table) > 0 and r.table.p_value.between(0, 1).all()
    a.obs["section"] = np.where(np.arange(a.n_obs) % 2 == 0, "s1", "s2")  # instances span sections
    with pytest.raises(ValueError, match="span"):
        gex.rank_genes_regions(a, groups="GC", reference="other", sample_key="section")


def test_volcano_plot():
    import matplotlib.pyplot as plt

    from spaclone import pl

    a = _regions(8, de_genes=20, lfc=2.0, n_genes=100)
    r = gex.rank_genes_regions(a, groups="GC", reference="other")
    ax = pl.gex_volcano(r, style="dark", label_top=5)
    assert len(ax.texts) == 5
    plt.close("all")


@pytest.mark.slow
def test_units_as_replicates_are_anticonservative():
    """The archive lesson: per-unit tests call many null genes significant; pseudobulk does not."""
    a = _regions(11, n_genes=200, inst_sd=0.4)
    with pytest.warns(PseudoreplicationWarning):
        r = gex.rank_genes_regions(a, groups="GC", reference="other", method="scanpy")
    fpr_units = float((r.table.p_value < 0.05).mean())
    pb = gex.rank_genes_regions(a, groups="GC", reference="other")
    fpr_pb = float((pb.table.p_value < 0.05).mean())
    assert fpr_units > 0.2
    assert fpr_pb < 0.1
    assert stats.binomtest(int(fpr_units * len(r.table)), len(r.table), 0.05).pvalue < 1e-6
