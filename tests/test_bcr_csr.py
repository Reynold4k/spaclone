import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, pl, tl


def _toy():
    return pd.DataFrame(
        {
            "clone_id": ["A", "A", "B", "B", "B", "C", "C"],
            "isotype_subclass": ["IgM", "IgG1", "IgA", "IgM", "IgG1", "IgG2b", "IgG2b"],
            "mut": [0, 4, 9, 1, 5, 2, 3],
        }
    )


def test_transitions_known_counts_and_orientation():
    r = bcr.csr_transitions(_toy(), n_perm=19, rng=0)
    t = r.table.set_index(["upstream", "downstream"])
    assert t.loc[("IgM", "IgG1"), "n_clones_both"] == 2
    assert t.loc[("IgM", "IgA"), "n_clones_both"] == 1
    assert t.loc[("IgG1", "IgA"), "n_clones_both"] == 1
    assert t.loc[("IgM", "IgG2b"), "n_clones_both"] == 0
    assert ("IgA", "IgM") not in t.index  # always oriented 5' -> 3'
    assert (r.table.upstream_position < r.table.downstream_position).all()
    assert t.loc[("IgM", "IgG1"), "share"] == pytest.approx(1.0)
    assert r.summary["n_clones_multi"] == 2
    assert {"statistic", "null_mean", "null_sd", "z", "p_value", "q_value", "n_perm"} <= set(r.table.columns)


def test_transitions_within_clone_direction():
    r = bcr.csr_transitions(_toy(), null="within", order_key="mut", n_perm=99, rng=0)
    t = r.table.set_index(["upstream", "downstream"])
    # IgM -> IgG1: clone A 4 - 0, clone B 5 - 1 -> mean 4
    assert t.loc[("IgM", "IgG1"), "statistic"] == pytest.approx(4.0)
    assert np.isnan(t.loc[("IgM", "IgG2b"), "p_value"])  # never in one clone: not tested
    with pytest.raises(ValueError, match="order_key"):
        bcr.csr_transitions(_toy(), null="within")


def _clones(rng, n_clones=120, planted=False, coherent=True):
    rows = []
    for c in range(n_clones):
        n = int(rng.integers(2, 7))
        if coherent:
            labs = [rng.choice(["IgM", "IgG1", "IgG2b", "IgA"], p=[0.55, 0.15, 0.15, 0.15])] * n
            if rng.random() < 0.3:  # some clones carry two isotypes
                labs[: n // 2] = ["IgM"] * (n // 2)
        else:
            labs = list(rng.choice(["IgM", "IgG1", "IgG2b", "IgA"], n, p=[0.55, 0.15, 0.15, 0.15]))
        for lab in labs:
            m = rng.poisson(2.0 + (3.0 if planted and lab != "IgM" else 0.0))
            rows.append({"clone_id": f"c{c}", "isotype_subclass": lab, "mut": m, "library_id": f"L{c % 2}"})
    return pd.DataFrame(rows)


def test_transitions_within_power_on_planted_direction():
    rng = np.random.default_rng(5)
    r = bcr.csr_transitions(
        _clones(rng, n_clones=400, planted=True), null="within", order_key="mut", unit_key=None, n_perm=199, rng=1
    )
    t = r.table.set_index(["upstream", "downstream"])
    assert t.loc[("IgM", "IgG1"), "p_value"] <= 0.01
    assert t.loc[("IgM", "IgG1"), "statistic"] > 2


def test_transitions_calibration_under_the_null():
    """Labels independent of clones (global null) and of mutations (within null): P(p <= 0.05) <= 0.05 + tol."""
    rng = np.random.default_rng(11)
    reps = 100
    hits_g = hits_w = 0
    for _ in range(reps):
        d = _clones(rng, n_clones=60, coherent=False)
        g = bcr.csr_transitions(d, n_perm=39, rng=rng, alternative="two-sided", unit_key=None)
        hits_g += g.table.set_index(["upstream", "downstream"]).loc[("IgM", "IgG1"), "p_value"] <= 0.05
        d2 = _clones(rng, n_clones=60, coherent=True)
        w = bcr.csr_transitions(d2, null="within", order_key="mut", unit_key=None, n_perm=39, rng=rng)
        p = w.table.set_index(["upstream", "downstream"]).loc[("IgM", "IgG1"), "p_value"]
        hits_w += p <= 0.05
    tol = 3.1 * np.sqrt(0.05 * 0.95 / reps)
    assert hits_g / reps <= 0.05 + tol
    assert hits_w / reps <= 0.05 + tol


def test_transitions_coherent_clones_fall_below_the_global_null():
    rng = np.random.default_rng(2)
    r = bcr.csr_transitions(_clones(rng), n_perm=99, rng=3, alternative="less", unit_key=None)
    t = r.table.set_index(["upstream", "downstream"])
    assert t.loc[("IgG1", "IgA"), "ratio"] < 0.5  # isotype-coherent clones mix far less than random labelling
    assert t.loc[("IgG1", "IgA"), "p_value"] <= 0.05


def test_transitions_determinism_strata_and_units():
    rng = np.random.default_rng(4)
    d = _clones(rng)
    a = bcr.csr_transitions(d, n_perm=29, rng=7, keep_null=True, unit_key=None)
    b = bcr.csr_transitions(d, n_perm=29, rng=7, keep_null=True, unit_key=None)
    c = bcr.csr_transitions(d, n_perm=29, rng=8, keep_null=True, unit_key=None)
    np.testing.assert_array_equal(a.null, b.null)
    assert not np.array_equal(a.null, c.null)
    s = bcr.csr_transitions(d, n_perm=29, rng=7, within="library_id", unit_key=None)
    assert s.params["within"] == "library_id"
    rec = pd.DataFrame(
        {
            "clone_id": ["A"] * 5,
            "locus_id": [0, 0, 0, 1, 1],
            "isotype_subclass": ["IgM", "IgM", "IgG1", "IgG1", "IgG1"],
        }
    )
    u = bcr.csr_transitions(rec, unit_key="locus_id", n_perm=9, rng=0)
    assert u.summary["n_units"] == 2
    assert u.table.set_index(["upstream", "downstream"]).loc[("IgM", "IgG1"), "n_clones_both"] == 1
    cls = bcr.csr_transitions(
        d.assign(isotype=d.isotype_subclass.str[:3]), level="class", n_perm=9, rng=0, unit_key=None
    )
    assert set(cls.table.upstream) <= {"IgM", "IgG"}


def _grid_loci(seed=0, planted=True):
    rng = np.random.default_rng(seed)
    n = 2000
    x, y = rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)
    p = np.where(planted & ((x - 300) ** 2 + (y - 300) ** 2 < 80**2), 0.7, 0.15)
    sw = pd.array(rng.random(n) < p, dtype="boolean")
    sw[::40] = pd.NA  # no isotype: left out of the map
    return pd.DataFrame({"x_um": x, "y_um": y, "switched": sw, "clone_id": np.arange(n).astype(str)})


@pytest.mark.skipif(not hasattr(tl, "hotspots"), reason="spaclone.tl.hotspots not installed")
def test_csr_hotspots_find_a_planted_cluster_and_wrap_tl():
    loci = _grid_loci()
    hs = bcr.csr_hotspots(loci, n_perm=49, rng=0, min_neff=5)
    assert hs.summary["n_units"] == int(loci.switched.notna().sum())
    assert hs.summary["n_hotspots"] >= 1
    top = hs.table.iloc[0]
    assert abs(top.x_um - 300) < 100 and abs(top.y_um - 300) < 100
    sub = loci[loci.switched.notna()].assign(switched=lambda d: d.switched.astype(bool))
    legacy = bcr.csr_hotspots(loci, blocks=None, n_perm=49, rng=0, min_neff=5)
    direct = tl.hotspots(sub, mark="switched", n_perm=49, rng=0, min_neff=5)
    np.testing.assert_array_equal(direct.labels.values, legacy.labels.values)
    assert hs.params["block_design"] == "matched" and hs.params["standardise"] == "permutation"
    assert legacy.params["blocks"] is None and legacy.params["standardise"] == "binomial"
    m = bcr.csr_map(loci, min_neff=5)
    assert m.p0 == pytest.approx(float(loci.switched.dropna().astype(float).mean()))
    null_hits = sum(
        bcr.csr_hotspots(_grid_loci(seed=s, planted=False), n_perm=39, rng=s, min_neff=5).summary["n_hotspots"] > 0
        for s in range(10, 20)
    )
    assert null_hits <= 3
    ax = pl.csr_share_map(hs, units=loci.assign(isotype="IgG"))
    assert ax is not None


@pytest.mark.skipif(not hasattr(tl, "region_excess"), reason="spaclone.tl.region_excess not installed")
def test_region_test_isotype_matched_expectation():
    """A region rich in IgG holds more mutated loci only because IgG mutates more: matched test is calibrated."""
    reps = 40
    hits_raw = hits_iso = 0
    for s in range(reps):
        rng = np.random.default_rng(s)
        n = 600
        region = rng.random(n) < 0.15
        iso = np.where(rng.random(n) < np.where(region, 0.7, 0.2), "IgG", "IgM")
        mut = rng.random(n) < np.where(iso == "IgG", 0.35, 0.05)
        loci = pd.DataFrame({"isotype": iso, "mutated": mut, "region": region.astype(int), "clone_id": np.arange(n)})
        raw = bcr.region_test(loci, regions="region", match_by=None, blocks=None, n_perm=199, rng=s)
        iso_t = bcr.region_test(loci, regions="region", blocks=None, n_perm=199, rng=s)
        hits_raw += raw.summary["pooled_p"] <= 0.05
        hits_iso += iso_t.summary["pooled_p"] <= 0.05
    assert hits_raw / reps > 0.5
    assert hits_iso / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_csr_transitions_plot():
    r = bcr.csr_transitions(_toy(), n_perm=9, rng=0)
    ax = pl.csr_transitions(r, value="n_clones_both", style="dark")
    assert ax.get_xlabel().startswith("downstream")
    with pytest.raises(KeyError):
        pl.csr_transitions(r, value="nope")


def _records_per_cell(rng, n_clones=120, spatial=True):
    """The reviewer's null: isotype per cell independent of its mutations; 1 + Poisson(6) records per cell, the cells
    of a clone 30-300 um apart (records within 2 um of their cell)."""
    rows = []
    cell = 0
    for c in range(n_clones):
        px, py = rng.uniform(0, 1e4, 2)
        for _ in range(rng.integers(2, 7)):
            iso = rng.choice(["IGHM", "IGHG1", "IGHA"], p=[0.5, 0.3, 0.2])
            mut = rng.poisson(5)
            th = rng.uniform(0, 2 * np.pi)
            px, py = px + rng.uniform(30, 300) * np.cos(th), py + rng.uniform(30, 300) * np.sin(th)
            for _ in range(1 + rng.poisson(6)):
                rows.append((f"c{c}", f"cell{cell}", iso, mut, px + rng.uniform(-2, 2), py + rng.uniform(-2, 2)))
            cell += 1
    t = pd.DataFrame(rows, columns=["clone_id", "cell", "c_call", "mut", "x_um", "y_um"])
    t["spot_id"] = (t.x_um.round().astype(int).astype(str) + "_" + t.y_um.round().astype(int).astype(str)).to_numpy()
    return t if spatial else t.drop(columns=["x_um", "y_um", "spot_id"])


def test_transitions_units_are_cells_by_default():
    rng = np.random.default_rng(31)
    t = _records_per_cell(rng, n_clones=40)
    auto = bcr.csr_transitions(t, null="within", order_key="mut", n_perm=19, rng=0)
    cell = bcr.csr_transitions(t, null="within", order_key="mut", unit_key="cell", n_perm=19, rng=0)
    assert auto.summary["n_units"] == cell.summary["n_units"] == t.cell.nunique()
    assert auto.params["units"].startswith("loci built") and cell.params["units"] == "column 'cell'"
    np.testing.assert_allclose(auto.table.statistic.to_numpy(), cell.table.statistic.to_numpy(), equal_nan=True)
    rows = bcr.csr_transitions(t, null="within", order_key="mut", unit_key=None, n_perm=19, rng=0)
    assert rows.summary["n_units"] == len(t) and rows.params["units"] == "rows"
    by_locus = bcr.csr_transitions(t.assign(locus_id=pd.factorize(t.cell)[0]), null="within", order_key="mut",
                                   n_perm=19, rng=0)  # fmt: skip
    assert by_locus.params["unit_key"] == "locus_id" and by_locus.summary["n_units"] == t.cell.nunique()
    by_cell_id = bcr.csr_transitions(t.rename(columns={"cell": "cell_id"}), n_perm=9, rng=0)
    assert by_cell_id.params["unit_key"] == "cell_id"
    with pytest.warns(spc_warning(), match="exact copies"):
        bcr.csr_transitions(_records_per_cell(rng, n_clones=20, spatial=False).drop(columns="cell"), n_perm=9, rng=0)


def spc_warning():
    from spaclone._utils import SpacloneWarning

    return SpacloneWarning


@pytest.mark.parametrize(("reps", "units"), [(30, "cell"), pytest.param(300, "auto", marks=pytest.mark.slow)])
def test_within_null_is_calibrated_on_cells_not_records(reps, units):
    """Records of one cell share its isotype and mutations: records as units reject far too often (22-32% in the
    review), cells (the default: loci from the spot coordinates) keep P(p <= 0.05) <= 0.05 + tolerance."""
    rng = np.random.default_rng(32)
    hits_cell = hits_rows = 0
    for _ in range(reps):
        t = _records_per_cell(rng, n_clones=50)
        seed = int(rng.integers(2**31))
        a = bcr.csr_transitions(t, null="within", order_key="mut", unit_key=units, n_perm=39, rng=seed).table
        r = bcr.csr_transitions(t, null="within", order_key="mut", unit_key=None, n_perm=39, rng=seed).table
        pick = lambda tab: tab.set_index(["upstream", "downstream"]).loc[("IgM", "IgG1"), "p_value"]
        hits_cell += pick(a) <= 0.05
        hits_rows += pick(r) <= 0.05
    assert hits_cell / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    assert hits_rows > hits_cell


def test_csr_hotspots_merge_records_into_loci_first():
    rng = np.random.default_rng(33)
    loci = _grid_loci(seed=3)
    loci = loci[loci.switched.notna()].reset_index(drop=True)
    k = 1 + rng.poisson(2, len(loci))
    rec = loci.loc[loci.index.repeat(k)].reset_index(drop=True)
    rec["x_um"] = rec.x_um + rng.uniform(-2, 2, len(rec))
    rec["y_um"] = rec.y_um + rng.uniform(-2, 2, len(rec))
    grid = tl.Grid((-50.0, -50.0), 10.0, (110, 110))
    a = bcr.csr_hotspots(rec, n_perm=19, rng=0, min_neff=5, grid=grid)
    b = bcr.csr_hotspots(loci, n_perm=19, rng=0, min_neff=5, grid=grid)
    assert a.params["units_merged_into_loci"] == len(rec) - len(loci) and b.params["units_merged_into_loci"] == 0
    assert a.summary["n_units"] == b.summary["n_units"] == len(loci) and a.summary["n_marked"] == b.summary["n_marked"]
    ok = a.map.eligible & b.map.eligible
    d = np.abs(a.map.share[ok] - b.map.share[ok])  # loci centroids differ by < 2 um
    assert np.median(d) < 0.005 and d.max() < 0.05
