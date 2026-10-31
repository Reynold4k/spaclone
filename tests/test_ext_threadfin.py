import dataclasses
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

threadfin = pytest.importorskip("threadfin")

from spaclone.ext import threadfin as tfx  # noqa: E402
from spaclone.tl import Grid, Raster  # noqa: E402

# Threadfin settings that keep a run on ~800 units at about a second (no clone-map search, few permutations)
KW = {"n_perm": 9, "n_boot": 5, "map_n_neighbors": 10, "map_min_dist": 0.3}
COMP = ["B_cells", "T_cells", "Erythrocyte"]


def _toy(*, side=30, n_clones=120, signal="none", seed=0):
    """Spleen-like section: white pulp (left, with a T zone) and red pulp (right) on a 10-um grid.

    Expression = niche genes (red / white) + state genes (a unit's own state); ``signal``: "none" (states iid),
    "state" (clone members share a state, independent of place) or "niche" (clone members in one compartment).
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    x, y = ii.ravel() * 10.0, jj.ravel() * 10.0
    n = x.size
    red = x >= side * 5.0
    tz = (~red) & (x < 0.3 * side * 10) & (y < 0.35 * side * 10)
    ctype = np.where(red, "Erythrocyte", np.where(tz, "T_cells", "B_cells")).astype(object)
    flip = rng.random(n) < 0.08
    ctype[flip] = rng.choice(["Erythrocyte", "T_cells", "B_cells", "Monocytes"], flip.sum())
    ab = {k: rng.gamma(0.5, 0.3, n) for k in COMP}
    for k in ab:
        ab[k][ctype == k] += rng.gamma(4.0, 0.5, (ctype == k).sum())
    sizes = np.minimum(rng.geometric(0.35, n_clones), 25)
    sizes[:12] = rng.integers(6, 14, 12)
    clone = np.full(n, None, dtype=object)
    state = rng.integers(0, 2, n).astype(float)
    free = np.ones(n, bool)
    for c, m in enumerate(sizes):
        cand = np.flatnonzero(free)
        if signal == "niche":
            cand = cand[red[cand] == (c % 2 == 0)]
        pick = rng.choice(cand, size=min(m, len(cand)), replace=False)
        free[pick] = False
        clone[pick] = f"c{c:03d}"
        if signal == "state":
            state[pick] = c % 2
    lam = np.ones((n, 60))
    lam[:, :15] += 4.0 * red[:, None]
    lam[:, 15:30] += 4.0 * (~red)[:, None]
    lam[:, 30:40] += 4.0 * state[:, None]
    X = np.hstack([rng.poisson(lam), rng.poisson(np.where(red, 8.0, 1.0)[:, None] * np.ones((1, 3)))])
    genes = [f"G{i}" for i in range(60)] + ["Ighm", "Igkc", "Ighg1"]
    ids = [f"{int(a)}-{int(b)}" for a, b in zip(x * 2, y * 2, strict=True)]
    obs = pd.DataFrame({"cell_type": pd.Categorical(ctype), **ab}, index=ids)
    gex = ad.AnnData(X=sparse.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=genes))
    gex.obsm["spatial"] = np.column_stack([x, y])
    cl = np.flatnonzero(clone != None)  # noqa: E711
    air = pd.DataFrame(
        {
            "spot_id": np.asarray(ids)[cl],
            "x_um": x[cl],
            "y_um": y[cl],
            "locus": "IGH",
            "clone_id": clone[cl].astype(str),
            "consensus_count": rng.integers(2, 30, cl.size).astype(float),
        }
    )
    return air, gex, {"red": red, "state": state}


def _run(air, gex, **kw):
    ctx = kw.pop("context", None) or tfx.spatial_context(gex, cell_type_key="cell_type")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Threadfin's notes on unstable programmes in tiny data
        return tfx.run(
            (air, gex), extra_spots=gex.obs.cell_type.eq("B_cells").to_numpy(), context=ctx, threadfin_kwargs=KW, **kw
        )


@pytest.fixture(scope="module")
def state_data():
    air, gex, truth = _toy(signal="state", seed=1)
    return air, gex, truth, _run(air, gex, rng=0)


def test_run_tables_and_reproduction(state_data):
    air, gex, _, res = state_data
    u, c = res.units, res.clones
    assert res.params["reproduction_max_abs_diff"] < 1e-9
    assert int(u.clonal.sum()) == air.spot_id.nunique()
    assert u.clone_id.nunique() == air.clone_id.nunique()
    sizes = air.clone_id.value_counts()
    assert set(c.index) == set(sizes.index[sizes >= 2])
    assert (c.n_units == sizes.reindex(c.index)).all()
    for col in ("axis_1", "clone_axis_1", "compartment", "ig_bin", "dist_t_zone_um", "depth_b_rich_um"):
        assert col in u.columns
    assert u.profiled.sum() == c.n_units.sum()
    # the mean of a clone's unit positions follows its (shrunken) profile value
    from scipy.stats import spearmanr

    assert spearmanr(c.axis_1, c.unit_mean_axis_1)[0] > 0.8
    sh = res.axes.share.to_numpy()
    assert (np.diff(sh) <= 1e-12).all() and sh.sum() <= 1 + 1e-12
    # Threadfin's sign rule: the most extreme clone is positive on every axis
    for j in (1, 2, 3):
        v = c[f"axis_{j}"].to_numpy()
        assert v[np.argmax(np.abs(v))] > 0
    assert repr(res).startswith("ThreadfinSpatialResult(spot/expression")
    assert res.summary()["n_profiled_clones"] == len(c)


def test_dominant_clone_rule():
    air = pd.DataFrame(
        {
            "spot_id": ["a", "a", "a", "b", "b", "c"],
            "clone_id": ["x", "y", "y", "z", "w", "x"],
            "locus": ["IGH", "IGH", "IGH", "IGH", "IGH", "IGK"],
            "consensus_count": [5.0, 3.0, 1.0, 4.0, 4.0, 9.0],
        }
    )
    d = tfx._dominant_clones(
        air, chains=("IGH",), clone_key="clone_id", spot_key="spot_id", locus_key="locus", reads_key="consensus_count"
    )
    assert d.loc["a", "clone"] == "x"  # 5 reads vs 3 + 1 = 4
    assert d.loc["b", "clone"] == "w"  # tie: smallest clone id
    assert "c" not in d.index  # light chains do not define the clone
    assert d.loc["a", "n_clones"] == 2


def test_locus_units_merge_adjacent_spots():
    air, gex, _ = _toy(signal="none", seed=3)
    # give the large clones a second spot next to their first one, so that loci really merge spots
    xy = gex.obsm["spatial"]
    where = pd.Series(np.arange(gex.n_obs), index=gex.obs_names)
    taken = set(air.spot_id)
    extra_rows = []
    for cid in [f"c{k:03d}" for k in range(12)]:
        first = air[air.clone_id == cid].iloc[0]
        p = xy[where[first.spot_id]]
        for dx, dy in ((10, 0), (-10, 0), (0, 10), (0, -10), (10, 10)):
            hit = np.flatnonzero((xy[:, 0] == p[0] + dx) & (xy[:, 1] == p[1] + dy))
            if len(hit) and gex.obs_names[hit[0]] not in taken:
                r = first.copy()
                r["spot_id"], r["x_um"], r["y_um"] = gex.obs_names[hit[0]], p[0] + dx, p[1] + dy
                extra_rows.append(r)
                taken.add(r["spot_id"])
                break
    a2 = pd.concat([air, pd.DataFrame(extra_rows)], ignore_index=True)
    res = _run(a2, gex, units="locus", rng=0)
    u = res.units
    # loci are exactly pp.merge_loci's connected components of same-clone spots within 15 um
    from spaclone.pp import merge_loci

    loci, _ = merge_loci(a2[["spot_id", "x_um", "y_um", "clone_id"]], radius_um=15.0, by=())
    assert int(u.clonal.sum()) == len(loci)
    assert sorted(u.loc[u.clonal, "n_spots"]) == sorted(loci.n_spots)
    assert int(u.loc[u.clonal, "n_spots"].max()) >= 2
    # counts are summed over the locus' spots; unclonal context spots stay single units
    spot_total = pd.Series(np.asarray(gex.X.sum(axis=1)).ravel(), index=gex.obs_names)
    m = res.membership
    exp = spot_total.reindex(m.index).groupby(m.to_numpy()).sum()
    np.testing.assert_allclose(u.loc[exp.index, "total_counts"].to_numpy(), exp.to_numpy())
    assert (u.loc[~u.clonal, "n_spots"] == 1).all()
    sizes = u.loc[u.clonal].groupby("clone_id").size()
    assert set(res.clones.index) == set(sizes.index[sizes >= 2])


def test_niche_features_known_answer():
    # 5 x 5 grid, 10 um; classes by column: B B T E E
    ii, jj = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    x, y = ii.ravel() * 10.0, jj.ravel() * 10.0
    cls = np.array(["B", "B", "T", "E", "E"])[ii.ravel()]
    gex = ad.AnnData(
        X=sparse.csr_matrix(np.ones((25, 2), np.float32)),
        obs=pd.DataFrame({"ct": pd.Categorical(cls)}, index=[f"s{i}" for i in range(25)]),
    )
    gex.obsm["spatial"] = np.c_[x, y]
    centre = np.array([[20.0, 20.0]])  # the T spot in the middle
    spec = tfx.NicheSpec(cell_type_key="ct", radius_um=10.0, exclude_self=True, distances=False)
    f = tfx.niche_features(gex, centre, spec=spec)
    # neighbours within 10 um, itself excluded: (10,20)=B, (30,20)=E, (20,10)=T, (20,30)=T
    assert f.loc[0, "comp_b"] == pytest.approx(0.25)
    assert f.loc[0, "comp_e"] == pytest.approx(0.25)
    assert f.loc[0, "comp_t"] == pytest.approx(0.5)
    g = tfx.niche_features(gex, centre, spec=dataclasses.replace(spec, exclude_self=False))
    assert g.loc[0, "comp_t"] == pytest.approx(0.6)
    # abundances (composition_keys) are summed, then made proportions
    gex.obs["a"] = np.where(cls == "B", 2.0, 0.0)
    gex.obs["b"] = 1.0
    h = tfx.niche_features(
        gex, centre, spec=tfx.NicheSpec(composition_keys=["a", "b"], radius_um=10.0, distances=False)
    )
    assert h.loc[0, "comp_a"] == pytest.approx(2.0 / 6.0)
    with pytest.raises(ValueError, match="nothing to describe"):
        tfx.niche_features(gex, centre, spec=tfx.NicheSpec(distances=False))


def test_spatial_context_and_signed_distances():
    _, gex, truth = _toy(seed=2)
    ctx = tfx.spatial_context(gex, cell_type_key="cell_type")
    xy = gex.obsm["spatial"]
    smp = ctx.sample(xy[:, 0], xy[:, 1])
    red = truth["red"]
    # the right half is red pulp, away from its border
    core = red & (xy[:, 0] >= xy[:, 0].max() * 0.65)
    assert (smp.compartment[core] == "red pulp").mean() > 0.95
    left = (~red) & (xy[:, 0] <= xy[:, 0].max() * 0.35) & (xy[:, 1] > xy[:, 1].max() * 0.6)
    assert (smp.compartment[left] == "follicle").mean() > 0.9
    # signed distance: negative inside the B-rich area, positive outside; depth = minus distance
    assert (smp.dist_b_rich_um[left] < 0).all()
    assert (smp.dist_b_rich_um[core] > 0).all()
    np.testing.assert_allclose(smp.depth_b_rich_um, -smp.dist_b_rich_um)
    with pytest.raises(KeyError):
        tfx.spatial_context(gex, cell_type_key="nope")


def _node_like(seed=2):
    """The toy section with a lymph node's right half: sparse typed spots (B, T and blood erythrocytes) among untyped
    ones (medulla, sinus), so the area outside the B and T zones holds no erythroid-rich red pulp."""
    _, gex, truth = _toy(seed=seed)
    rng = np.random.default_rng(seed)
    red = truth["red"]
    ct = gex.obs["cell_type"].astype(str).to_numpy().copy()
    ct[red] = rng.choice(["Monocytes", "B_cells", "T_cells", "Erythrocyte"], red.sum(), p=[0.91, 0.03, 0.03, 0.03])
    gex.obs["cell_type"] = pd.Categorical(ct)
    return gex, truth


def test_spatial_context_names_follow_the_labels():
    """Red pulp only where the labels show one; tissue-neutral names elsewhere; names never move a pixel."""
    _, gex, truth = _toy(seed=2)
    spleen = tfx.spatial_context(gex, cell_type_key="cell_type")
    assert spleen.params["naming"] == "spleen" and spleen.params["outside_erythroid_share"] >= 0.8
    assert spleen.compartments.categories == tfx.COMPARTMENT_NAMES["spleen"]
    assert list(spleen.masks) == ["T zone", "B-rich", "red pulp", "erythroid-rich"]
    node, truth = _node_like()
    ctx = tfx.spatial_context(node, cell_type_key="cell_type")
    assert ctx.params["naming"] == "neutral" and ctx.params["names"] == "auto"
    assert ctx.params["outside_erythroid_share"] < tfx.RED_PULP_MIN_ERYTHROID
    assert ctx.compartments.categories == ("T zone", "non-B/T area", "follicle")
    assert list(ctx.masks) == ["T zone", "B-rich", "non-B/T area", "erythroid-rich"]
    xy = node.obsm["spatial"]
    smp = ctx.sample(xy[:, 0], xy[:, 1])
    core = truth["red"] & (xy[:, 0] >= xy[:, 0].max() * 0.65)
    assert (smp.compartment[core] == "non-B/T area").mean() > 0.95
    assert "dist_non_b_t_area_um" in smp.columns and not smp.columns.str.contains("pulp").any()
    forced = tfx.spatial_context(node, cell_type_key="cell_type", names="spleen")
    assert forced.params["naming"] == "spleen" and "red pulp" in forced.compartments.categories
    np.testing.assert_array_equal(forced.compartments.values, ctx.compartments.values)  # same pixels, other names
    for a, b in zip(forced.masks.values(), ctx.masks.values(), strict=True):
        np.testing.assert_array_equal(a.values, b.values)
    smp2 = forced.sample(xy[:, 0], xy[:, 1])
    np.testing.assert_array_equal(smp2["dist_red_pulp_um"].to_numpy(), smp["dist_non_b_t_area_um"].to_numpy())
    neutral = tfx.spatial_context(gex, cell_type_key="cell_type", names="neutral")
    assert "red pulp" not in neutral.compartments.categories
    with pytest.raises(ValueError, match="names"):
        tfx.spatial_context(gex, cell_type_key="cell_type", names="lymph node")


def test_compartment_names_never_change_a_number(state_data):
    """Calling the red pulp the non-B/T area leaves every readout unchanged (the default red covers both names)."""
    *_, res = state_data
    u = res.units.copy()
    u["compartment"] = u["compartment"].astype(str).replace({"red pulp": "non-B/T area"})
    renamed = dataclasses.replace(res, units=u)
    a = tfx.spanning_clones(res, n_perm=19, rng=3)
    b = tfx.spanning_clones(renamed, n_perm=19, rng=3)
    assert a.table["white_and_red"].astype(float).sum() > 0  # the red / white columns are in play
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.summary.keys() == b.summary.keys()
    for k, v in a.summary.items():
        assert v == b.summary[k] or (isinstance(v, float) and np.isnan(v) and np.isnan(b.summary[k])), k


@pytest.mark.slow
def test_compartment_names_never_change_niche_by_axis(state_data):
    """The niche readout under either name: the same table, row for row, but for the category label."""
    *_, res = state_data
    u = res.units.copy()
    u["compartment"] = u["compartment"].astype(str).replace({"red pulp": "non-B/T area"})
    renamed = dataclasses.replace(res, units=u)
    n1 = tfx.niche_by_axis(res, n_perm=9, rng=2, n_bins=4, within=None)
    n2 = tfx.niche_by_axis(renamed, n_perm=9, rng=2, n_bins=4, within=None)
    t2 = n2.table.assign(category=n2.table.category.replace({"non-B/T area": "red pulp"}))
    pd.testing.assert_frame_equal(n1.table, t2)


def test_region_inputs_agree():
    _, gex, truth = _toy(seed=2)
    mask = truth["red"] & (gex.obsm["spatial"][:, 1] < 60)
    gex.obs["reg"] = mask
    c1 = tfx.spatial_context(gex, cell_type_key="cell_type", regions={"r": "reg"})
    lab = c1.regions["r"]
    assert lab.values.max() == 1
    c2 = tfx.spatial_context(gex, cell_type_key="cell_type", regions={"r": Raster(lab.values > 0, lab.grid)})

    class FakeWorkflow:  # duck-typed spaclone.bcr.HotspotWorkflowResult
        region_labels = Raster(lab.values.astype(np.int64), lab.grid)

    c3 = tfx.spatial_context(gex, cell_type_key="cell_type", regions={"r": FakeWorkflow()})
    for c in (c2, c3):
        np.testing.assert_array_equal(c.regions["r"].values, lab.values)
    xy = gex.obsm["spatial"]
    s = c1.sample(xy[:, 0], xy[:, 1])
    assert s.in_r[mask].all() and not s.in_r[~mask & (xy[:, 1] > 100)].any()
    assert (s.dist_r_um[mask] < 0).all()


def test_determinism_and_rng(state_data):
    air, gex, _, res = state_data
    res2 = _run(air, gex, rng=0)
    pd.testing.assert_frame_equal(res.clones, res2.clones)
    np.testing.assert_array_equal(res.units.axis_1.to_numpy(), res2.units.axis_1.to_numpy())
    a = tfx.spatial_coherence(res, n_perm=19, rng=5)
    b = tfx.spatial_coherence(res, n_perm=19, rng=5)
    c = tfx.spatial_coherence(res, n_perm=19, rng=6)
    np.testing.assert_array_equal(a.null, b.null)
    assert not np.array_equal(a.null, c.null)
    s1 = tfx.spanning_clones(res, n_perm=19, rng=7)
    s2 = tfx.spanning_clones(res, n_perm=19, rng=7)
    np.testing.assert_array_equal(s1.null, s2.null)
    pd.testing.assert_frame_equal(s1.table, s2.table)


def test_coherence_power_on_planted_clone_state(state_data):
    *_, res = state_data
    r = tfx.spatial_coherence(res, n_perm=19, rng=1, within=("ig_bin", "compartment"))
    t = r.table.set_index("statistic_name")
    assert t.loc["unit_vs_rest_of_clone", "p_value"] == pytest.approx(1 / 20)
    assert t.loc["unit_vs_rest_of_clone", "statistic"] > t.loc["unit_vs_rest_of_clone", "null_hi"]
    # self-inclusion: a unit against its own clone is far from 0 even under the null
    assert t.loc["unit_vs_own_clone", "expected"] > 0.2
    for col in ("statistic", "expected", "null_mean", "null_sd", "z", "p_value", "q_value", "n_perm"):
        assert col in r.table.columns


def _relabel(res, gen):
    """The same units with clone labels permuted among clonal units (clone sizes kept): a null data set."""
    u = res.units.copy()
    m = u.clonal.to_numpy()
    u.loc[m, "clone_id"] = gen.permutation(u.loc[m, "clone_id"].to_numpy())
    return dataclasses.replace(res, units=u)


def _calibration(res, reps, seed):
    gen = np.random.default_rng(seed)
    hits = total = 0
    for _ in range(reps):
        r0 = _relabel(res, gen)
        # one-sided p with 19 and two-sided p with 39 permutations both reject with probability exactly 0.05
        coh = tfx.spatial_coherence(r0, n_perm=19, rng=gen)
        nba = tfx.niche_by_axis(r0, n_perm=39, rng=gen, n_bins=4)
        p = np.r_[coh.table.p_value.to_numpy(), nba.table[nba.table.kind == "trend"].p_value.to_numpy()]
        hits += int((p <= 0.05).sum())
        total += len(p)
    return hits, total


def test_calibration_under_random_clone_labels(state_data):
    """Clone labels independent of the units: P(p <= 0.05) <= 0.05 + binomial tolerance (exchangeability)."""
    *_, res = state_data
    hits, total = _calibration(res, 15, 123)
    assert hits / total <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / total)


@pytest.mark.slow
def test_calibration_under_random_clone_labels_long(state_data):
    *_, res = state_data
    hits, total = _calibration(res, 200, 321)
    assert hits / total <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / total)
    assert hits / total >= 0.05 - 3.1 * np.sqrt(0.05 * 0.95 / total)  # not conservative either


def test_niche_by_axis_power_on_place_coherent_clones():
    air, gex, truth = _toy(signal="niche", seed=4)
    res = _run(air, gex, rng=0)
    # no strata: in this toy, Ig RNA (the default depth strata) is itself a red-pulp marker
    r = tfx.niche_by_axis(res, n_perm=39, rng=2, n_bins=5, within=None)
    tr = r.table[(r.table.kind == "trend") & (r.table.category == "red pulp")].iloc[0]
    assert tr.p_value == pytest.approx(2 / 40)  # two-sided floor
    assert abs(tr.statistic) > max(abs(tr.null_lo), abs(tr.null_hi))
    # null axes are oriented like the observed one: the null of the end bins keeps its side
    sh = r.table[(r.table.kind == "share") & (r.table.category == "red pulp")]
    lo, hi = (sh.iloc[0], sh.iloc[-1]) if tr.statistic > 0 else (sh.iloc[-1], sh.iloc[0])
    assert lo.null_mean < hi.null_mean
    sh = r.table[(r.table.kind == "share") & (r.table.category == "red pulp")]
    assert len(sh) == 5 and sh.n_units.sum() == int(res.units.profiled.sum())
    assert {"red pulp D1", "red pulp D5", "trend[red pulp]", "p[red pulp]"} <= set(r.summary)


def test_rigid_landings_keep_shape():
    gen = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(0, 400, 2.0), np.arange(0, 400, 2.0))
    slots = np.c_[gx.ravel(), gy.ravel()]
    from scipy.spatial import cKDTree

    tree = cKDTree(slots)
    pts = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 40.0], [25.0, 25.0]])
    idx, got = tfx._rigid_landings(pts, slots, tree, n_draws=50, snap_um=2.0, max_tries=20, gen=gen)
    assert got == 50 and (idx >= 0).all()
    d0 = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    for b in range(got):
        q = slots[idx[b]]
        d = np.sqrt(((q[:, None] - q[None]) ** 2).sum(-1))
        assert np.abs(d - d0).max() <= 2 * 2.0 * np.sqrt(2) + 1e-9
    # a shape larger than the slot area never fits
    big = pts * 100
    _, got = tfx._rigid_landings(big, slots, tree, n_draws=5, snap_um=2.0, max_tries=3, gen=gen)
    assert got == 0


def test_spanning_clones_relocation_and_permutation(state_data):
    *_, res = state_data
    s = tfx.spanning_clones(res, n_perm=19, rng=3)
    t = s.table
    assert len(t) == int((res.clones.n_units >= 5).sum())
    for col in (
        "spanning",
        "axis_range",
        "cross_um",
        "white_and_red",
        "median_pairwise_um",
        "p_range",
        "p_cross",
        "n_valid",
        "axis_range_null",
    ):
        assert col in t.columns
    ok = t.p_range.dropna()
    assert ((ok > 0) & (ok <= 1)).all()
    # planted clone states: members are more alike than their relocated copies
    assert s.summary["spanning_share"] < s.summary["spanning_share_null"]
    assert s.summary["median_axis_range"] < s.summary["median_axis_range_null"]
    # relocation keeps the footprint, so spread is unchanged under it
    np.testing.assert_allclose(t.median_pairwise_um_null, t.median_pairwise_um)
    sp = tfx.spanning_clones(res, n_perm=9, rng=3, null="permute")
    assert "p_spread" in sp.table.columns and sp.params["design"]["design"] == "Permute"
    with pytest.raises(ValueError):
        tfx.spanning_clones(res, null="shuffle")


def test_relocation_readouts_from_saved_tables(state_data):
    *_, res = state_data
    assert res.params["projection_max_abs_diff"] < 1e-9  # same positions as threadfin.profiles.project_features
    light = dataclasses.replace(res, engine=None, threadfin=None)  # e.g. rebuilt from saved tables
    a = tfx.spanning_clones(res, n_perm=19, rng=8)
    b = tfx.spanning_clones(light, n_perm=19, rng=8)
    pd.testing.assert_frame_equal(a.table, b.table)
    xy = res.units[["x_um", "y_um"]].to_numpy()
    grid = Grid.from_points(xy[:, 0], xy[:, 1], pixel_um=10.0)
    lab = np.zeros(grid.shape, dtype=np.int64)
    lab[3:8, 3:8] = 1
    pd.testing.assert_frame_equal(
        tfx.region_axis(res, Raster(lab, grid), n_perm=9, rng=1).table,
        tfx.region_axis(light, Raster(lab, grid), n_perm=9, rng=1).table,
    )
    with pytest.raises(TypeError, match="feature matrix"):
        tfx.niche_by_axis(light)
    with pytest.raises(TypeError, match="feature matrix"):
        tfx.spanning_clones(light, null="permute")


def test_region_axis(state_data):
    air, gex, _, res = state_data
    xy = res.units[["x_um", "y_um"]].to_numpy()
    grid = Grid.from_points(xy[:, 0], xy[:, 1], pixel_um=10.0)
    lab = np.zeros(grid.shape, dtype=np.int64)
    lab[2:6, 2:6] = 1
    lab[20:24, 20:25] = 2
    r = tfx.region_axis(res, Raster(lab, grid), n_perm=19, rng=4)
    t = r.table
    assert set(t.region) == {"1", "2", "all"}
    assert set(t.statistic_name) == {"unit_percentile", "clone_percentile"}
    assert ((t.p_value > 0) & (t.p_value <= 1)).all()
    assert (t.statistic.between(0, 1) | t.statistic.isna()).all()

    class FakeWorkflow:
        region_labels = Raster(lab, grid)

    r2 = tfx.region_axis(res, FakeWorkflow(), n_perm=19, rng=4)
    pd.testing.assert_frame_equal(r.table, r2.table)
    with pytest.raises(ValueError, match="no unit"):
        tfx.region_axis(res, Raster(np.zeros(grid.shape, dtype=np.int64), grid), n_perm=5)


def test_niche_and_joint_features_and_compare(state_data):
    air, gex, _, res = state_data
    spec = tfx.NicheSpec(composition_keys=COMP, radius_um=30.0)
    resn = _run(air, gex, features="niche", niche=spec, rng=0)
    assert resn.params["basis"] == "X_niche" and resn.niche is not None
    assert {"comp_b_cells", "comp_t_cells", "comp_erythrocyte", "dist_t_zone_um"} <= set(resn.niche.columns)
    np.testing.assert_allclose(resn.niche.filter(like="comp_").sum(axis=1), 1.0)
    # the niche variant knows nothing of the planted clone states: its axes follow place, not state
    from scipy.stats import spearmanr

    red = resn.units.compartment.eq("red pulp").to_numpy()
    assert abs(spearmanr(resn.units.axis_1, red)[0]) > 0.5
    c = tfx.compare(res, resn, n_perm=19, rng=1)
    assert 0 <= c.summary["R2(a|b)"] <= 1
    same = tfx.compare(res, res, n_perm=9, rng=1)
    assert same.summary["R2(a|b)"] == pytest.approx(1.0)
    assert same.summary["rho_axis_1"] == pytest.approx(1.0)
    resj = _run(air, gex, features="joint", niche=spec, rng=0)
    assert resj.params["basis"] == "X_joint"
    assert (
        resj.adata.obsm["X_joint"].shape[1]
        == res.adata.obsm["X_threadfin"].shape[1] + resn.adata.obsm["X_niche"].shape[1]
    )


def test_anndata_input_matches_receptor_table(state_data):
    """Clone ids already in gex.obs give the same run as the receptor table."""
    air, gex, _, res = state_data
    g3 = gex.copy()
    g3.obs["clone_id"] = res.units.clone_id.reindex(g3.obs_names).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r3 = tfx.run(
            g3, extra_spots=g3.obs.cell_type.eq("B_cells").to_numpy(), context=res.context, threadfin_kwargs=KW, rng=0
        )
    pd.testing.assert_frame_equal(r3.clones, res.clones)
    np.testing.assert_array_equal(r3.units.axis_1.to_numpy(), res.units.axis_1.to_numpy())


def test_input_errors(state_data):
    air, gex, _, res = state_data
    with pytest.raises(ValueError, match="units"):
        tfx.run((air, gex), units="cell")
    with pytest.raises(ValueError, match="features"):
        tfx.run((air, gex), features="both")
    with pytest.raises(ValueError, match="needs niche"):
        tfx.run((air, gex), features="niche", context=None)
    with pytest.raises(ValueError, match="context must be"):
        tfx.run((air, gex), context="spleen")
    with pytest.raises(ValueError, match="rng"):
        tfx.run((air, gex), threadfin_kwargs={"random_state": 1})
    with pytest.raises(TypeError):
        tfx.run(air)
    g2 = gex.copy()
    g2.obs["library_id"] = np.where(np.arange(g2.n_obs) % 2 == 0, "a", "b")
    with pytest.raises(ValueError, match="one section"):
        tfx.run((air, g2))

    with pytest.raises(ValueError, match="axis"):
        tfx.niche_by_axis(res, axis=9)
    with pytest.raises(KeyError):
        tfx.niche_by_axis(res, compartment_key="nope")
    with pytest.raises(KeyError):
        tfx.spatial_coherence(res, within=("nope",))
    with pytest.raises(TypeError):
        tfx.spatial_coherence(res.units)


def test_context_built_automatically_only_when_certain(state_data):
    """One standard label column naming T, B and erythroid classes -> the same context as spatial_context."""
    air, gex, _, res = state_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        auto = tfx.run((air, gex), extra_spots=gex.obs.cell_type.eq("B_cells").to_numpy(), threadfin_kwargs=KW, rng=0)
    assert auto.context_note.startswith("auto: ") and "'cell_type'" in auto.context_note
    assert auto.params["context_note"] == auto.context_note
    pd.testing.assert_series_equal(auto.units.compartment, res.units.compartment)
    np.testing.assert_array_equal(auto.units.axis_1.to_numpy(), res.units.axis_1.to_numpy())
    assert res.context_note == "given"

    two = gex.copy()
    two.obs["annotation"] = two.obs["cell_type"].astype(str)  # two standard label columns: ambiguous
    ctx, note = tfx._auto_context(two, spatial_key="spatial")
    assert ctx is None and note.startswith("none: ") and "cell_type" in note and "annotation" in note
    node = gex.copy()
    node.obs["cell_type"] = node.obs["cell_type"].astype(str).replace({"Erythrocyte": "Monocytes"})  # no red pulp
    ctx, note = tfx._auto_context(node, spatial_key="spatial")
    assert ctx is None and "erythroid" in note
    bare = gex.copy()
    bare.obs = bare.obs.drop(columns="cell_type")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = tfx.run((air, bare), threadfin_kwargs=KW, rng=0)
    assert r.context is None and r.context_note.startswith("none: ")
    assert "compartment" not in r.units.columns
