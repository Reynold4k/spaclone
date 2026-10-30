import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex
from spaclone.gex.annotate import JoinWarning, MarkerWarning

matplotlib.use("Agg")

MODS = {"B": ["Cd79a", "Ms4a1"], "T": ["Cd3e", "Cd3d"], "ASC": ["Jchain", "Xbp1"]}


def _str(series):
    """Labels as strings with missing values as "nan" (pandas 3 keeps NaN through astype(str))."""
    return series.astype(object).where(series.notna(), "nan").astype(str).tolist()


def _adata(x, genes, xy=None, tissue=None):
    a = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(x, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"u{i}" for i in range(len(x))]),
        var=pd.DataFrame(index=genes),
    )
    a.obsm["spatial"] = np.asarray(
        xy if xy is not None else np.column_stack([np.arange(len(x)) * 10.0, np.zeros(len(x))]), dtype=float
    )
    if tissue is not None:
        a.obs["in_tissue"] = tissue
    return a


def _x6_reference(x, genes, mods, floor=2):
    """The archive's X6 typing written as there (pandas share / mean share, idxmax, >= 2 marker UMIs)."""
    t = pd.DataFrame(x, columns=genes)
    total = t.sum(axis=1)
    m = pd.DataFrame({k: t[v].sum(axis=1) for k, v in mods.items()})
    share = m.div(total.clip(lower=1), axis=0)
    enr = share / share.mean()
    return np.where(m.sum(axis=1) >= floor, enr.idxmax(axis=1), "unassigned")


def test_enrichment_typing_known_answer_and_floor():
    genes = ["Cd79a", "Ms4a1", "Cd3e", "Cd3d", "Jchain", "Xbp1", "Actb"]
    x = [
        [5, 3, 0, 0, 0, 0, 10],  # B
        [0, 0, 4, 4, 0, 0, 10],  # T
        [0, 0, 0, 0, 20, 5, 10],  # ASC
        [1, 0, 0, 0, 0, 0, 30],  # one marker UMI -> unassigned
        [1, 0, 1, 0, 0, 0, 30],
    ]  # two markers, one B and one T molecule
    a = _adata(x, genes)
    tab = gex.score_cell_types(a, markers=MODS)
    lab = a.obs.cell_type.astype(str).tolist()
    # default: the winning module needs two of its own molecules, so one B + one T molecule stays unassigned
    assert lab == ["B", "T", "ASC", "unassigned", "unassigned"]
    assert list(a.obs.cell_type.cat.categories) == ["B", "T", "ASC", "unassigned"]
    assert tab.set_index("cell_type").loc["unassigned", "n_units"] == 2
    # the archive rule (X6): two marker molecules in total suffice and the higher enrichment wins
    gex.score_cell_types(a, markers=MODS, min_module_counts=1, key_added="x6")
    enr = a.obsm["x6_enrichment"]
    assert a.obs.x6.astype(str).iloc[4] == ("B" if enr.loc["u4", "B"] >= enr.loc["u4", "T"] else "T")


def test_module_floor_removes_the_rare_module_bias():
    # a rare module (one gene, mean share tiny) wins every unit holding a single molecule of it under the X6 rule
    rng = np.random.default_rng(9)
    genes = ["Cd79a", "Ms4a1", "Cd3e", "Cd3d", "Aicda", "Actb"]
    n = 2000
    lam = np.column_stack(
        [np.full(n, 2.0), np.full(n, 2.0), np.full(n, 2.0), np.full(n, 2.0), np.full(n, 0.03), np.full(n, 40.0)]
    )
    a = _adata(rng.poisson(lam), genes)
    mods = {"B": ["Cd79a", "Ms4a1"], "T": ["Cd3e", "Cd3d"], "GC_B": ["Aicda"]}
    gex.score_cell_types(a, markers=mods, min_module_counts=1, key_added="x6", tissue_key=None)
    gex.score_cell_types(a, markers=mods, tissue_key=None)
    share_x6 = (a.obs.x6.astype(str) == "GC_B").mean()
    share_new = (a.obs.cell_type.astype(str) == "GC_B").mean()
    assert share_x6 > 0.02  # ~3% of units hold one Aicda molecule by chance and are all called GC B
    assert share_new < 0.002


def test_enrichment_typing_equals_archive_x6_formula():
    rng = np.random.default_rng(3)
    genes = [g for v in MODS.values() for g in v] + [f"G{i}" for i in range(20)]
    x = rng.poisson(rng.gamma(0.5, 2.0, size=(400, len(genes)))).astype(float)
    a = _adata(x, genes)
    gex.score_cell_types(a, markers=MODS, tissue_key=None, min_module_counts=1)
    ref = _x6_reference(x, genes, MODS)
    assert (a.obs.cell_type.astype(str).to_numpy() == ref).all()


def test_reference_mean_uses_tissue_units_only():
    genes = ["Cd79a", "Ms4a1", "Cd3e", "Cd3d", "Actb"]
    x = np.array([[2, 0, 2, 0, 10]] * 4 + [[50, 50, 0, 0, 0]] * 4, dtype=float)
    mods = {"B": MODS["B"], "T": MODS["T"]}
    a = _adata(x, genes, tissue=[True] * 4 + [False] * 4)
    gex.score_cell_types(a, markers=mods)
    # tissue-only mean: B and T shares are equal and tie -> B (first module); with the B-rich off-tissue units in
    # the mean the B enrichment would drop below T
    assert a.obs.cell_type.astype(str).tolist()[:4] == ["B"] * 4
    gex.score_cell_types(a, markers=mods, tissue_key=None, key_added="all_units")
    assert a.obs.all_units.astype(str).tolist()[:4] == ["T"] * 4


def test_missing_module_warns_and_is_dropped():
    a = _adata([[3, 3]], ["Cd79a", "Ms4a1"])
    with pytest.warns(MarkerWarning):
        tab = gex.score_cell_types(a, markers=MODS, tissue_key=None)
    assert "T" not in list(a.obs.cell_type.cat.categories)
    assert set(tab.cell_type) == {"B", "unassigned"}
    with pytest.raises(ValueError):
        gex.score_cell_types(a, markers="immune_fish")


def test_builtin_sets_have_mouse_and_human_names():
    assert set(gex.MARKERS["immune_mouse"]) == set(gex.MARKERS["immune_human"])
    assert {"B", "GC_B", "ASC", "T", "Tfh", "myeloid", "stromal", "erythroid"} <= set(gex.MARKERS["immune_mouse"])
    assert gex.MARKERS["immune_human"]["B"][0] == gex.MARKERS["immune_mouse"]["B"][0].upper()


def test_celltype_composition_depth_matched_known_answer():
    # depth bins: units 0-3 shallow, 4-7 deep; shallow are half B, deep are all T; select two shallow units
    genes = ["Cd79a", "Cd3e", "Actb"]
    x = np.array([[5, 0, 5], [5, 0, 5], [0, 5, 5], [0, 5, 5], [0, 9, 91], [0, 9, 91], [0, 9, 91], [0, 9, 91]])
    a = _adata(x, genes)
    a.obs["cell_type"] = pd.Categorical(["B", "B", "T", "T", "T", "T", "T", "T"])
    sel = np.array([True, False, True, False, False, False, False, False])
    c = gex.celltype_composition(a, sel, n_depth_bins=2, tissue_key=None, groups={"all": ["B", "T"]}).set_index(
        "cell_type"
    )
    assert c.loc["B", "observed_pct"] == pytest.approx(50.0)
    assert c.loc["B", "depth_matched_pct"] == pytest.approx(50.0)
    assert c.loc["T", "baseline_pct"] == pytest.approx(75.0)
    assert c.loc["all", "observed_pct"] == pytest.approx(100.0)


def test_import_deconvolution_by_index_never_by_row_order():
    a = _adata(np.ones((4, 1)), ["Actb"])
    w = pd.DataFrame(
        {"B": [3.0, 0.0, 1.0], "T": [1.0, 2.0, 1.0], "first_type": ["B", "T", "B"]}, index=["u2", "u0", "u3"]
    )
    rep = gex.import_deconvolution(a, w, kind="rctd")
    got = a.obsm["rctd"]
    np.testing.assert_allclose(got.loc["u2"].to_numpy(), [0.75, 0.25])
    np.testing.assert_allclose(got.loc["u0"].to_numpy(), [0.0, 1.0])
    assert got.loc["u1"].isna().all()
    assert _str(a.obs.rctd_top) == ["T", "nan", "B", "B"]
    assert _str(a.obs.rctd_first_type) == ["T", "nan", "B", "B"]
    assert rep.n_matched.iloc[0] == 3
    with pytest.raises(ValueError, match="row order"):
        gex.import_deconvolution(a, w.reset_index(drop=True), kind="rctd")


def test_import_deconvolution_by_coordinates_and_cell2location_names():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    a = _adata(np.ones((3, 1)), ["Actb"], xy=xy)
    t = pd.DataFrame(
        {
            "x": [20.4, 0.2],
            "y": [0.1, -0.3],
            "q05cell_abundance_w_sf_B_cells": [5.0, 1.0],
            "q05cell_abundance_w_sf_T_cells": [1.0, 2.0],
        },
        index=["r1", "r2"],
    )
    gex.import_deconvolution(a, t, kind="cell2location", on="coordinates", key_added="c2l")
    assert list(a.obsm["c2l"].columns) == ["B_cells", "T_cells"]
    np.testing.assert_allclose(a.obsm["c2l"].loc["u2"].to_numpy(), [5.0, 1.0])
    assert _str(a.obs.c2l_top) == ["T_cells", "nan", "B_cells"]
    t2 = t.copy()
    t2["x"] = [0.1, 0.3]  # two rows claim unit u0
    with pytest.raises(ValueError, match="one to one"):
        gex.import_deconvolution(a, t2, kind="cell2location", on="coordinates")


def test_import_deconvolution_warns_when_few_units_match():
    a = _adata(np.ones((4, 1)), ["Actb"])
    w = pd.DataFrame({"B": [1.0, 1.0, 1.0], "T": [0.0, 1.0, 1.0]}, index=["u0", "zz1", "zz2"])
    with pytest.warns(JoinWarning):
        gex.import_deconvolution(a, w, kind="generic")


def test_scanpy_score_method_and_dotplot():
    import matplotlib.pyplot as plt

    from spaclone import pl

    rng = np.random.default_rng(0)
    genes = [g for v in MODS.values() for g in v] + [f"G{i}" for i in range(200)]
    n = 150
    lam = np.full((n, len(genes)), 1.0)
    lam[:50, 0:2] = 8
    lam[50:100, 2:4] = 8
    lam[100:, 4:6] = 8
    a = _adata(rng.poisson(lam), genes)
    gex.preprocess(a, n_hvg=20, n_pcs=5, min_gene_frac=0.0)
    gex.score_cell_types(a, markers=MODS, method="scanpy_score", tissue_key=None)
    lab = a.obs.cell_type.astype(str).to_numpy()
    assert (lab[:50] == "B").mean() > 0.8 and (lab[50:100] == "T").mean() > 0.8 and (lab[100:] == "ASC").mean() > 0.8
    ax = pl.gex_dotplot(a, MODS, groupby="cell_type", style="dark")
    assert len(ax.get_xticklabels()) == 6
    plt.close("all")
