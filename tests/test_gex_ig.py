import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex, nulls


def _adata(x, genes, xy=None):
    a = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(x, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"u{i}" for i in range(len(x))]),
        var=pd.DataFrame(index=genes),
    )
    a.obsm["spatial"] = np.asarray(
        xy if xy is not None else np.column_stack([np.arange(len(x)) * 10.0, np.zeros(len(x))]), dtype=float
    )
    return a


def test_ig_gene_lists():
    mouse = ["Xkr4", "Igkc", "Iglc1", "Iglc4", "Ighm", "Ighg2c", "Igha", "Jchain"]
    assert gex.ig_genes(mouse, "mouse") == ["Igkc", "Iglc1", "Ighm", "Ighg2c", "Igha"]
    human = ["IGKC", "IGLC2", "IGLC3", "IGLCOR22-1", "IGHM", "IGHG1", "IGHA2", "IGHV3-23", "JCHAIN"]
    assert gex.ig_genes(human, "human") == ["IGKC", "IGLC2", "IGLC3", "IGHM", "IGHG1", "IGHA2"]
    assert gex.ig_genes(human, ["IGHM", "NOPE"]) == ["IGHM"]
    with pytest.raises(ValueError):
        gex.ig_genes(human, "fish")


def test_ig_rna_per_unit_and_strata_labels():
    genes = ["Igkc", "Ighm", "Ighg1", "Actb"]
    own = [0, 1, 2, 3, 9, 10, 29, 30, 99, 100, 299, 300, 1000]
    x = np.array([[v, 0, 0, 5] for v in own])
    a = _adata(x, genes)
    tab = gex.ig_rna(a, genes="mouse")
    np.testing.assert_array_equal(a.obs.ig_rna.to_numpy(), own)
    assert a.obs.ig_rna_bin.astype(str).tolist() == [
        "0",
        "1-2",
        "1-2",
        "3-9",
        "3-9",
        "10-29",
        "10-29",
        "30-99",
        "30-99",
        "100-299",
        "100-299",
        ">=300",
        ">=300",
    ]
    assert tab.bin.tolist() == ["0", "1-2", "3-9", "10-29", "30-99", "100-299", ">=300"]
    assert tab.n_units.tolist() == [1, 2, 2, 2, 2, 2, 2]
    # the labels are those of the core strata helper
    assert (a.obs.ig_rna_bin.astype(str).to_numpy() == nulls.bin_labels(own, nulls.IG_RNA_BINS)).all()


def test_ig_rna_radius_sums_tissue_units_within_sections():
    x = np.array([[1, 0], [2, 0], [4, 0], [8, 0]])
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [10.0, 0.0]])
    a = _adata(x, ["Igkc", "Actb"], xy=xy)
    a.obs["in_tissue"] = [True, True, True, False]
    a.obs["lib"] = ["s1", "s1", "s1", "s2"]
    gex.ig_rna(a, radius_um=10.0, tissue_key="in_tissue", library_key="lib", key_added="ig10")
    v = a.obs.ig10.to_numpy()
    np.testing.assert_allclose(v[:3], [3, 7, 6])
    assert np.isnan(v[3])
    assert a.obs.ig10_bin.astype(str).tolist()[3] == "NA"


def test_ig_class_composition_per_unit_and_group():
    genes = ["Ighm", "Ighd", "Ighg1", "Ighg2b", "Igha", "Ighe", "Igkc"]
    x = np.array([[6, 2, 1, 1, 0, 0, 50], [0, 0, 3, 3, 4, 0, 5], [0, 0, 0, 0, 0, 0, 9]])
    a = _adata(x, genes)
    a.obs["region"] = ["GC", "GC", "other"]
    c = gex.ig_class_composition(a)
    np.testing.assert_allclose(
        c.loc["u0", ["IgM", "IgD", "IgG", "IgA", "IgE"]].to_numpy(dtype=float), [0.6, 0.2, 0.2, 0.0, 0.0]
    )
    assert c.loc["u1", "switched"] == pytest.approx(1.0)
    assert np.isnan(c.loc["u2", "IgM"]) and c.loc["u2", "n_heavy"] == 0
    s = gex.ig_class_composition(a, level="subclass")
    assert s.loc["u1", "IgG2b"] == pytest.approx(0.3)
    g = gex.ig_class_composition(a, groupby="region")
    assert g.loc["GC", "n_heavy"] == 20 and g.loc["GC", "IgG"] == pytest.approx(0.4)
    h = gex.ig_class_composition(_adata([[1, 1, 2]], ["IGHA1", "IGHA2", "IGHG4"]), species="human")
    assert h.loc["u0", "IgA"] == pytest.approx(0.5) and h.loc["u0", "IgG"] == pytest.approx(0.5)
