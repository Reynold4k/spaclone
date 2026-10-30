import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import gex, tl
from spaclone.gex.profile import ProfileWarning, is_ig_gene

SETS = {"a": ["A1", "A2"], "b": ["B1"]}


def _tiny():
    """Six units on a line; units 0-1 = region 1, units 2-3 = its window; known pseudobulk answers."""
    genes = ["A1", "A2", "B1", "H"]
    x = np.array(
        [
            [5, 5, 0, 90],
            [3, 1, 1, 95],
            [0, 1, 4, 195],
            [1, 0, 1, 98],
            [0, 0, 2, 98],
            [0, 0, 0, 100],
        ],
        dtype=np.float32,
    )
    a = ad.AnnData(
        X=sparse.csr_matrix(x), obs=pd.DataFrame(index=[f"u{i}" for i in range(6)]), var=pd.DataFrame(index=genes)
    )
    a.obsm["spatial"] = np.column_stack([np.arange(6) * 10.0, np.zeros(6)])
    return a


def test_profile_known_answer():
    a = _tiny()
    region = np.array([1, 1, 0, 0, 0, 0])
    window = {1: np.array([0, 0, 1, 1, 0, 0], bool)}
    prof = gex.profile_regions(a, region, markers=SETS, readouts=None, window=window, tissue_key=None)
    t = prof.table.set_index("set")
    assert t.loc["a", "cp10k_inside"] == pytest.approx(14 / 200 * 1e4)
    assert t.loc["a", "cp10k_window"] == pytest.approx(2 / 300 * 1e4)
    assert t.loc["a", "cp10k_tissue"] == pytest.approx(16 / 700 * 1e4)
    assert t.loc["a", "ratio_window"] == pytest.approx(10.5)
    assert t.loc["a", "ratio_tissue"] == pytest.approx(3.0625)
    assert t.loc["b", "ratio_window"] == pytest.approx((1 / 200) / (5 / 300))
    assert t.loc["b", "ratio_tissue"] == pytest.approx((1 / 200) / (8 / 700))
    assert t.loc["a", "n_units_inside"] == 2 and t.loc["a", "n_units_window"] == 2
    assert t.loc["b", "frac_units_detected_inside"] == pytest.approx(0.5)
    lab = prof.labels.iloc[0]
    assert lab.label == "a" and lab.top == "a" and lab.second == "b" and not lab.mixed
    assert lab.ranking.startswith("a=10.50;b=0.30")
    assert prof.units.region_id.tolist() == [1, 1, 0, 0, 0, 0]
    assert prof.units.in_window.tolist() == [False, False, True, True, False, False]


def test_mixed_label_and_threshold():
    a = _tiny()
    region = np.array([1, 1, 0, 0, 0, 0])
    window = {1: np.array([0, 0, 1, 1, 0, 0], bool)}
    sets = {"a": ["A1", "A2"], "a_copy": ["A1"]}
    prof = gex.profile_regions(a, region, markers=sets, readouts=None, window=window, tissue_key=None)
    lab = prof.labels.iloc[0]
    # a: 14/200 vs 2/300 -> 10.5; a_copy: 8/200 vs 1/300 -> 12.0; log2(12 / 10.5) = 0.19 < 0.3 -> mixed
    assert lab.top == "a_copy" and lab.mixed and lab.label == "a_copy / a (mixed)"
    prof0 = gex.profile_regions(a, region, markers=sets, readouts=None, window=window, tissue_key=None, mixed_log2=0.1)
    assert prof0.labels.iloc[0].label == "a_copy"


def test_region_and_window_inputs_agree():
    a = _tiny()
    labels = np.array([2, 2, 0, 0, 0, 0])
    grid = tl.Grid((-5.0, -5.0), 10.0, (6, 1))
    raster = tl.Raster(np.array([[2], [2], [0], [0], [0], [0]]), grid)
    masks = pd.DataFrame({2: labels == 2}, index=a.obs_names)
    boxes = pd.DataFrame({"region_id": [2], "x0": [-1.0], "x1": [31.0], "y0": [-1.0], "y1": [1.0]})
    kw = {"markers": SETS, "readouts": None, "tissue_key": None}
    t1 = gex.profile_regions(a, labels, window=boxes, **kw).table
    t2 = gex.profile_regions(a, raster, window=boxes, **kw).table
    t3 = gex.profile_regions(a, masks, window={2: np.array([0, 0, 1, 1, 0, 0], bool)}, **kw).table
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(t1, t3)
    # a pad of 20 um around units 0-1 (x 0..10) reaches units 2 and 3 (x 20, 30 -> 30 > 30? no: 0-20 .. 10+20 = 30)
    t4 = gex.profile_regions(a, labels, window=20.0, **kw)
    assert t4.table.n_units_window.iloc[0] == 2
    assert t4.params["boxes"].x1.iloc[0] == pytest.approx(30.0)
    # obs column input, string region ids
    a.obs["reg"] = np.where(labels == 2, "R", "")
    t5 = gex.profile_regions(a, "reg", window={"R": np.array([0, 0, 1, 1, 0, 0], bool)}, **kw).table
    assert t5.region_id.unique().tolist() == ["R"]
    np.testing.assert_allclose(t5.ratio_window, t1.ratio_window)


def test_exclude_other_regions_from_windows_and_tissue_key():
    a = _tiny()
    labels = np.array([1, 1, 2, 0, 0, 0])
    win = {1: np.array([0, 0, 1, 1, 0, 0], bool), 2: np.array([0, 0, 0, 1, 1, 0], bool)}
    kw = {"markers": SETS, "readouts": None, "tissue_key": None}
    t = gex.profile_regions(a, labels, window=win, **kw).table.set_index(["region_id", "set"])
    assert t.loc[(1, "a"), "n_units_window"] == 1  # unit 2 is region 2: excluded
    t_own = gex.profile_regions(a, labels, window=win, exclude="own", **kw).table.set_index(["region_id", "set"])
    assert t_own.loc[(1, "a"), "n_units_window"] == 2
    a.obs["in_tissue"] = [True, True, True, True, True, False]
    tt = gex.profile_regions(a, labels, window=win, markers=SETS, readouts=None).table.set_index(["region_id", "set"])
    assert tt.loc[(1, "a"), "n_units_tissue"] == 5
    assert tt.loc[(1, "a"), "cp10k_tissue"] == pytest.approx(16 / 600 * 1e4)


def test_ig_genes_are_removed_from_cell_type_sets():
    assert is_ig_gene(["Ighg2b", "IGKC", "Ighv1-26", "Iglc1", "Igll1", "IGHA1", "Ighj3"]).all()
    assert not is_ig_gene(["Jchain", "Ighmbp2", "Igf1", "Irf4", "Mzb1", "IGLON5"]).any()
    with pytest.warns(ProfileWarning, match="immunoglobulin genes"):
        cells, _ = gex.profile_markers({"ASC": ["Jchain", "Xbp1", "Ighg1"]}, readouts=None)
    assert cells == {"ASC": ["Jchain", "Xbp1"]}
    with pytest.raises(ValueError, match="no gene left"), pytest.warns(ProfileWarning):
        gex.profile_markers({"Ig": ["Igkc"]})
    builtin, reads = gex.profile_markers("immune_mouse_no_ig")
    assert not any(is_ig_gene(g).any() for g in builtin.values())
    assert "switched Ig" in reads and "plasmablast/ASC" in builtin
    assert set(gex.PROFILE_MARKERS) == {"immune_mouse_no_ig", "immune_human_no_ig"}


def test_region_without_window_units_warns():
    a = _tiny()
    with pytest.warns(ProfileWarning, match="no unit in its window"):
        prof = gex.profile_regions(
            a, np.array([1, 0, 0, 0, 0, 0]), markers=SETS, readouts=None, window={1: np.zeros(6, bool)}, tissue_key=None
        )
    assert np.isnan(prof.table.ratio_window).all()
    assert prof.labels.label.iloc[0] == "undetermined"


def test_module_scores_known_answer():
    genes = ["A1", "A2", "B1"]
    x = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 0]], dtype=np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(x), obs=pd.DataFrame(index=["p", "q", "r"]), var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    tot = x.sum(1)
    lg = np.log1p(x * (1e4 / tot)[:, None])
    z = (lg - lg.mean(0)) / lg.std(0)
    raw_a = z[:, :2].mean(1)
    s = gex.module_scores(a, {"a": ["A1", "A2", "missing"]}, readouts=None, smooth_um=15.0)
    np.testing.assert_allclose(s["a"], [raw_a[:2].mean(), raw_a[:2].mean(), raw_a[2]], atol=1e-6)
    s0 = gex.module_scores(a, {"a": ["A1", "A2"]}, readouts=None, smooth_um=None)
    np.testing.assert_allclose(s0["a"], raw_a, atol=1e-6)
    a.obs["in_tissue"] = [True, True, False]
    st = gex.module_scores(a, {"a": ["A1", "A2"]}, readouts=None, smooth_um=None, tissue_key="in_tissue")
    assert np.isnan(st["a"].iloc[2]) and np.isfinite(st["a"].iloc[:2]).all()
    with pytest.warns(ProfileWarning, match="no gene"):
        sn = gex.module_scores(a, {"none": ["Zz1"]}, readouts=None)
    assert (sn["none"] == 0).all()
    gex.module_scores(a, {"a": ["A1"]}, readouts=None, key_added="mod")
    assert "mod_a" in a.obs


def test_profile_is_deterministic_and_invariant_to_unit_order():
    rng = np.random.default_rng(3)
    n = 200
    xy = rng.uniform(0, 300, (n, 2))
    lam = np.full((n, 4), 2.0)
    hot = np.hypot(xy[:, 0] - 150, xy[:, 1] - 150) < 50
    lam[hot, 0] += 6
    x = rng.poisson(lam).astype(np.float32)
    a = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=list("WXYZ")),
    )
    a.obsm["spatial"] = xy
    sets = {"w": ["W"], "x": ["X"], "y": ["Y"]}
    p1 = gex.profile_regions(a, hot.astype(int), markers=sets, readouts=None, window=60.0, tissue_key=None)
    p2 = gex.profile_regions(a, hot.astype(int), markers=sets, readouts=None, window=60.0, tissue_key=None)
    pd.testing.assert_frame_equal(p1.table, p2.table)
    perm = rng.permutation(n)
    b = a[perm].copy()
    p3 = gex.profile_regions(b, hot[perm].astype(int), markers=sets, readouts=None, window=60.0, tissue_key=None)
    pd.testing.assert_frame_equal(p1.table, p3.table)
    assert p1.labels.label.iloc[0] == "w" and p1.labels.top_ratio_window.iloc[0] > 2


def test_profile_input_errors():
    a = _tiny()
    with pytest.raises(ValueError, match="exclude"):
        gex.profile_regions(a, np.zeros(6), markers=SETS, exclude="none")
    with pytest.raises(KeyError, match="no mask"):
        gex.profile_regions(a, np.array([1, 0, 0, 0, 0, 0]), markers=SETS, window={2: np.ones(6, bool)})
    with pytest.raises(ValueError, match="unknown marker set"):
        gex.profile_regions(a, np.zeros(6), markers="nope")
    with pytest.raises(ValueError, match="window"):
        gex.profile_regions(a, np.zeros(6), markers=SETS, window="wide")


def test_unit_mask_forms_and_errors():
    a = _tiny()
    a.obs["kind"] = pd.Categorical(["B", "B", "T", "B", "x", "T"])
    a.obs["flag"] = [True, False, True, False, True, False]
    a.obs["n"] = [1, 0, 1, 0, 1, 0]
    assert gex.unit_mask(a, None).all()
    np.testing.assert_array_equal(gex.unit_mask(a, ("kind", ["B"])), [1, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(gex.unit_mask(a, ("kind", "T")), [0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(gex.unit_mask(a, "flag"), [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(gex.unit_mask(a, "n"), [1, 0, 1, 0, 1, 0])
    s = pd.Series([True, False, False, False, False, True], index=a.obs_names)[::-1]
    np.testing.assert_array_equal(gex.unit_mask(a, s), [1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(gex.unit_mask(a, np.array([0, 1, 0, 1, 0, 1])), [0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="not boolean"):
        gex.unit_mask(a, "kind")
    with pytest.raises(KeyError, match="not found"):
        gex.unit_mask(a, ("annotation", ["B"]))
    with pytest.raises(ValueError, match="no unit has"):
        gex.unit_mask(a, ("kind", ["Plasma"]))
    with pytest.raises(ValueError, match="one flag per unit"):
        gex.unit_mask(a, np.ones(4, bool))
    with pytest.raises(ValueError, match="lacks"):
        gex.unit_mask(a, s.iloc[:3])


def test_module_scores_on_selected_units_only():
    genes = ["A1", "A2", "B1"]
    x = np.array([[2, 0, 1], [0, 2, 1], [5, 5, 0], [1, 1, 0]], dtype=np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(x), obs=pd.DataFrame(index=["p", "q", "r", "s"]), var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [100.0, 0.0]])
    a.obs["cell"] = ["B", "B", "T", "B"]
    sel = np.array([True, True, False, True])
    lg = np.log1p(x[sel] * (1e4 / x[sel].sum(1))[:, None])
    raw = ((lg - lg.mean(0)) / lg.std(0))[:, :2].mean(1)  # z over the selected units only
    sc = gex.module_scores(a, {"a": ["A1", "A2"]}, readouts=None, smooth_um=15.0, units=("cell", ["B"]))
    # q's neighbours within 15 um are p and r; r is not selected, so q averages p and q only
    np.testing.assert_allclose(sc["a"].to_numpy()[[0, 1, 3]], [raw[:2].mean(), raw[:2].mean(), raw[2]], atol=1e-6)
    assert np.isnan(sc["a"].iloc[2])
    assert sc.attrs["spaclone"]["module_scores"]["n_scored"] == 3
    a.obs["in_tissue"] = [True, False, True, True]
    st = gex.module_scores(a, {"a": ["A1"]}, readouts=None, tissue_key="in_tissue", units=("cell", ["B"]))
    assert st["a"].isna().tolist() == [False, True, True, False]
