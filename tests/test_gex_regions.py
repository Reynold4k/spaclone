import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial import cKDTree

from spaclone import gex, tl

matplotlib.use("Agg")

GC = ["Aicda", "S1pr2", "Gcsam", "Rgs13", "Mef2b"]
PC = ["Jchain", "Xbp1", "Prdm1", "Sdc1"]


def _tissue(seed=0, n_side=40, centres=((100.0, 100.0), (290.0, 280.0)), asc=(300.0, 90.0), jitter=True):
    """Beads on a jittered 10-um lattice with planted germinal centres (r = 45 um) and an ASC patch (r = 40 um)."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij"), -1).reshape(-1, 2) * 10.0
    xy = g + (rng.uniform(-3, 3, size=g.shape) if jitter else 0)
    genes = GC + PC + ["Cd79a", "Cd3e", "Actb"] + [f"G{i}" for i in range(20)]
    lam = np.full((len(xy), len(genes)), 0.3)
    lam[:, genes.index("Actb")] = 20
    for cx, cy in centres:
        m = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < 45
        lam[np.ix_(m, range(5))] += 2.0
    m = np.hypot(xy[:, 0] - asc[0], xy[:, 1] - asc[1]) < 40
    lam[np.ix_(m, range(5, 9))] += 4.0
    x = rng.poisson(lam).astype(np.float32)
    a = ad.AnnData(
        X=sparse.csr_matrix(x), obs=pd.DataFrame(index=[f"b{i}" for i in range(len(xy))]), var=pd.DataFrame(index=genes)
    )
    a.obsm["spatial"] = xy
    a.obs["in_tissue"] = True
    return a


def _x3_reference(a, um=0.65, top=0.04, eps=30, ms=10, keep=40):
    """The archive's X3 germinal-centre rule written as in X3_airspace_claims.py (pixel units)."""
    from sklearn.cluster import DBSCAN

    x = a.X.tocsc()
    names = list(a.var_names)
    xy = a.obsm["spatial"] / um
    tot = np.asarray(x.sum(1)).ravel()
    gc = np.asarray(x[:, [names.index(g) for g in GC]].sum(1)).ravel()
    nb = cKDTree(xy).query_ball_point(xy, r=30 / um)
    share = np.array([gc[v].sum() for v in nb]) / np.maximum(np.array([tot[v].sum() for v in nb]), 1)
    hot = (share >= np.quantile(share, 1 - top)) & (share > 0)
    lab = np.full(len(xy), -1)
    if hot.sum() >= ms:
        cl = DBSCAN(eps=eps / um, min_samples=ms).fit_predict(xy[hot])
        sizes = pd.Series(cl[cl >= 0]).value_counts()
        k = set(sizes[sizes >= keep].index)
        lab[np.where(hot)[0]] = np.where(np.isin(cl, list(k)), cl, -1)
    return share, lab


def _ids(a, key="region_id"):
    v = a.obs[key].astype(object).to_numpy()
    return np.array(
        [-1 if (x is None or (isinstance(x, float) and np.isnan(x))) else int(str(x).rsplit("_", 1)[1]) for x in v]
    )


def test_marker_share_known_answer():
    x = np.array([[1, 9], [0, 10], [5, 5]], dtype=np.float32)
    a = ad.AnnData(
        X=sparse.csr_matrix(x), obs=pd.DataFrame(index=["a", "b", "c"]), var=pd.DataFrame(index=["Aicda", "Actb"])
    )
    a.obsm["spatial"] = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    a.obs["in_tissue"] = [True, True, False]
    s = gex.marker_share(a, ["Aicda"], radius_um=10.0)
    np.testing.assert_allclose(s[:2], [1 / 20, 1 / 20])
    assert np.isnan(s[2])
    s0 = gex.marker_share(a, ["Aicda"], radius_um=0.0, tissue_key=None)
    np.testing.assert_allclose(s0, [0.1, 0.0, 0.5])


def test_legacy_rule_matches_the_archive_implementation_exactly():
    a = _tissue(seed=1)
    share, ref = _x3_reference(a)
    gex.call_regions(
        a,
        kind="germinal_centre",
        top_frac=0.04,
        smooth_um=30.0,
        dbscan_eps_um=30.0,
        min_samples=10,
        min_units=40,
        widen_um=0.0,
        relabel=False,
    )
    # the archive computed shares in float32; the package's float64 shares round to exactly those values
    np.testing.assert_array_equal(a.obs.gc_share.to_numpy().astype(np.float32), share.astype(np.float32))
    np.testing.assert_array_equal(_ids(a), ref)


def test_default_rule_recovers_planted_germinal_centres_and_asc_area():
    a = _tissue(seed=2)
    tab = gex.call_regions(a, kind="germinal_centre")
    assert len(tab) == 2
    cen = tab[["x_um", "y_um"]].to_numpy()
    for cx, cy in ((100.0, 100.0), (290.0, 280.0)):
        assert np.min(np.hypot(cen[:, 0] - cx, cen[:, 1] - cy)) < 15
    assert (tab.n_units >= tab.n_core).all()
    gex.call_regions(a, kind="plasma_cell")
    reg = a.obs.region.astype(str).to_numpy()
    xy = a.obsm["spatial"]
    in_asc = np.hypot(xy[:, 0] - 300, xy[:, 1] - 90) < 30
    assert (reg[in_asc] == "ASC").mean() > 0.8
    assert set(reg) == {"GC", "ASC", "other"}
    assert a.obs.region_id.astype(object)[reg == "ASC"].isna().all()


def test_priority_overwrite_and_relabel():
    a = _tissue(seed=3)
    gex.call_regions(a, kind="germinal_centre")
    gc_units = a.obs.region.astype(str) == "GC"
    # ASC called with every marker of the GC set: without overwrite the GC units keep their label
    gex.call_regions(a, kind="plasma_cell", markers=GC, top_frac=0.2)
    assert (a.obs.region.astype(str)[gc_units] == "GC").all()
    gex.call_regions(a, kind="plasma_cell", markers=GC, top_frac=0.2, overwrite=True)
    assert (a.obs.region.astype(str)[gc_units] == "ASC").any()
    # calling the same kind again replaces the earlier call instead of adding to it
    b = _tissue(seed=3)
    gex.call_regions(b, kind="germinal_centre")
    n1 = b.obs.region_id.notna().sum()
    gex.call_regions(b, kind="germinal_centre")
    assert b.obs.region_id.notna().sum() == n1
    t_raw = gex.call_regions(b, kind="germinal_centre", relabel=False, region_key="r2", instance_key="i2")
    t_new = gex.call_regions(b, kind="germinal_centre", region_key="r3", instance_key="i3")
    assert sorted(t_new.region_id) == [f"GC_{i}" for i in range(len(t_new))]
    assert len(t_raw) == len(t_new)


def test_sections_are_called_separately_and_translation_invariant():
    a = _tissue(seed=4)
    b = _tissue(seed=5)
    both = ad.concat([a, b], index_unique="-", keys=["s1", "s2"], label="lib")
    both.obsm["spatial"] = np.vstack([a.obsm["spatial"], b.obsm["spatial"]])  # same coordinates, two sections
    tab = gex.call_regions(both, kind="germinal_centre", library_key="lib")
    assert sorted(tab.library.unique()) == ["s1", "s2"]
    assert all(r.startswith(("s1|", "s2|")) for r in tab.region_id)
    gex.call_regions(a, kind="germinal_centre")
    shifted = a.copy()
    shifted.obsm["spatial"] = a.obsm["spatial"] + np.array([5000.0, -300.0])
    gex.call_regions(shifted, kind="germinal_centre")
    same = shifted.obs.region_id.astype(object).fillna("none") == a.obs.region_id.astype(object).fillna("none")
    assert same.all()


def test_off_tissue_units_are_never_called():
    a = _tissue(seed=6)
    a.obs["in_tissue"] = a.obsm["spatial"][:, 0] < 200
    gex.call_regions(a, kind="germinal_centre", min_units=10)
    off = ~a.obs.in_tissue.to_numpy()
    assert a.obs.region[off].isna().all()
    assert np.isnan(a.obs.gc_share.to_numpy()[off]).all()


@pytest.mark.skipif(not hasattr(tl, "mask_from_types"), reason="needs spaclone.tl.structures")
def test_t_zone_and_compartments_from_typed_units():
    rng = np.random.default_rng(7)
    g = np.stack(np.meshgrid(np.arange(60), np.arange(60), indexing="ij"), -1).reshape(-1, 2) * 10.0
    a = ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(g), 1), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"b{i}" for i in range(len(g))]),
        var=pd.DataFrame(index=["Actb"]),
    )
    a.obsm["spatial"] = g
    a.obs["in_tissue"] = True
    t_zone = np.hypot(g[:, 0] - 300, g[:, 1] - 300) < 100
    follicle = (np.hypot(g[:, 0] - 300, g[:, 1] - 300) < 220) & ~t_zone
    lab = np.where(t_zone, "T", np.where(follicle, "B", "erythroid"))
    noise = rng.random(len(g)) < 0.1
    lab[noise] = rng.choice(["T", "B", "erythroid"], noise.sum())
    a.obs["cell_type"] = lab
    tz = gex.call_regions(a, kind="t_zone")
    assert len(tz) == 1
    assert (a.obs.region.astype(str).to_numpy()[np.hypot(g[:, 0] - 300, g[:, 1] - 300) < 60] == "T zone").all()
    gex.call_regions(a, kind="follicle")
    gex.call_regions(a, kind="red_pulp")
    reg = a.obs.region.astype(str).to_numpy()
    r = np.hypot(g[:, 0] - 300, g[:, 1] - 300)
    assert (reg[(r > 140) & (r < 190)] == "follicle").mean() > 0.9
    assert (reg[r > 260] == "red pulp").mean() > 0.9


def test_gex_regions_plot_labels_instances():
    import matplotlib.pyplot as plt

    from spaclone import pl

    a = _tissue(seed=8)
    gex.call_regions(a, kind="germinal_centre")
    ax = pl.gex_regions(a, style="dark")
    texts = {t.get_text() for t in ax.texts}
    assert {"0", "1"} <= texts
    plt.close("all")
