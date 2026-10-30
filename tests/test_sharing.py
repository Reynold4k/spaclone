import numpy as np
import pandas as pd
import pytest

from spaclone import nulls, tl
from spaclone._utils import SpacloneWarning
from spaclone.tl.sharing import _stats

TOL = 3.1  # binomial 99.9% one-sided tolerance in standard errors


def _records(n_clones=40, n_regions=6, confined=False, seed=0):
    """Molecules of clones in regions (-1 = outside); ``confined`` puts each clone's region records in one region."""
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 7, n_clones)
    clone = np.repeat(np.arange(n_clones), size)
    if confined:
        home = rng.integers(0, n_regions, n_clones)[clone]
        region = np.where(rng.random(len(clone)) < 0.8, home, -1)
    else:
        region = np.where(rng.random(len(clone)) < 0.8, rng.integers(0, n_regions, len(clone)), -1)
    return pd.DataFrame(
        {"clone_id": clone.astype(str), "gid": region, "locus": np.where(np.arange(len(clone)) % 2, "IGH", "IGK")}
    )


def test_measures_known_answer():
    clone = np.array([0, 0, 0, 0, 1, 1, 2])
    region = np.array([0, 0, 0, 1, 2, 2, 1])
    m, s, n = _stats(clone, region, 3, 2)
    assert (m, n) == (0.5, 2) and s == pytest.approx((0.75 + 1.0) / 2)
    t = pd.DataFrame({"clone_id": ["a", "a", "a", "a", "b", "b", "c", "d"], "gid": [0, 0, 0, 1, 2, 2, 1, -1]})
    r = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=19, rng=0, unit_key=None)
    obs = r.table[r.table.design == "regions"].set_index("measure").statistic
    assert obs["multi_region_share"] == 0.5 and obs["main_region_share"] == pytest.approx(0.875)
    row = r.table.iloc[0]
    assert row.clones == 2 and row.regions == 3 and row.records_in_regions == 7 and row.chain == "all"
    assert _stats(np.array([], int), np.array([], int), 1, 2)[2] == 0


def test_dense_and_sparse_counting_agree(monkeypatch):
    rng = np.random.default_rng(9)
    for n, k, g in ((150, 40, 6), (3000, 2000, 16), (12, 4, 3)):
        c, r = rng.integers(0, k, n), rng.integers(0, g, n)
        dense = _stats(c, r, g, 2)
        monkeypatch.setattr("spaclone.tl.sharing._DENSE_MAX", 0)
        sparse = _stats(c, r, g, 2)
        monkeypatch.undo()
        assert dense == sparse


def test_outside_values_and_region_zero():
    t = pd.DataFrame({"clone_id": list("aaaabb"), "gid": [0, 0, -1, np.nan, 5, 5]})
    r = tl.region_sharing(t, region_key="gid", chain_key=None, design="regions", n_perm=19, rng=0, unit_key=None)
    assert r.table.records_in_regions.iloc[0] == 4 and r.table.regions.iloc[0] == 2
    s = pd.DataFrame({"clone_id": list("aaab"), "gid": ["GC1", "none", "GC2", "GC1"]})
    rs = tl.region_sharing(
        s, region_key="gid", chain_key=None, background="none", design="regions", n_perm=19, rng=0, unit_key=None
    )
    assert rs.table.records_in_regions.iloc[0] == 3 and rs.table.statistic.iloc[0] == 1.0  # clone a in GC1 and GC2


def test_confined_clones_are_detected():
    t = _records(n_clones=80, confined=True, seed=1)
    r = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=199, rng=2, unit_key=None)
    for design in ("regions", "tissue"):
        row = r.table[(r.table.design == design) & (r.table.measure == "multi_region_share")].iloc[0]
        assert row.statistic < row.null_mean and row.p_value <= 0.01 and row.z < -3
        main = r.table[(r.table.design == design) & (r.table.measure == "main_region_share")].iloc[0]
        assert main.statistic == 1.0 and main.p_value <= 0.01


@pytest.mark.parametrize("reps", [120, pytest.param(1000, marks=pytest.mark.slow)])
def test_calibrated_when_clone_labels_are_random(reps):
    rng = np.random.default_rng(3)
    hits = {"regions": 0, "tissue": 0}
    for _ in range(reps):
        t = _records(seed=int(rng.integers(2**31)))
        t["clone_id"] = t.clone_id.to_numpy()[rng.permutation(len(t))]
        r = tl.region_sharing(
            t, region_key="gid", chain_key=None, n_perm=39, rng=int(rng.integers(2**31)), unit_key=None
        )
        for d in hits:
            hits[d] += (
                r.table[(r.table.design == d) & (r.table.measure == "multi_region_share")].p_value.iloc[0] <= 0.05
            )
    for d in hits:
        assert hits[d] / reps <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / reps)


def test_determinism_chains_and_custom_designs():
    t = _records(seed=4)
    t["library_id"] = np.where(np.arange(len(t)) % 3 == 0, "L1", "L2")
    kw = {"region_key": "gid", "n_perm": 49, "keep_null": True, "unit_key": None}
    a = tl.region_sharing(t, rng=5, **kw)
    b = tl.region_sharing(t, rng=np.random.default_rng(5), **kw)
    c = tl.region_sharing(t, rng=6, **kw)
    pd.testing.assert_frame_equal(a.table, b.table)
    np.testing.assert_array_equal(a.null, b.null)
    assert not np.array_equal(a.null, c.null, equal_nan=True)
    assert list(a.table.chain.unique()) == ["IGH", "IGK"] and len(a.table) == 2 * 2 * 2
    assert a.null.shape == (49, len(a.table)) and "IGH multi_region_share" in a.summary
    design = nulls.Permute("clone_id", within="library_id")
    d = tl.region_sharing(t, region_key="gid", design=["regions", design], n_perm=19, rng=7, unit_key=None)
    assert set(d.table.design) == {"regions", repr(design)}
    assert d.params["design"][1]["within"] == ["library_id"]
    with pytest.raises(ValueError, match="design"):
        tl.region_sharing(t, region_key="gid", design="anywhere", unit_key=None)
    with pytest.raises(KeyError):
        tl.region_sharing(t, region_key="gid", chain_key="chain")
    with pytest.raises(KeyError):
        tl.region_sharing(t, region_key="region")


# ------------------------------------------------------------------------------- cell-scale units by default (F4a)


def _cells(rng, confined=False, dnb=False, n_clones=200):
    """Clones of 1-12 cells; a cell lies in one of 8 germinal centres (radius 60 um) with probability 0.7 (the clone's
    home centre when ``confined``) and carries 1 + Poisson(3) molecules on its 10-um bead or on 0.5-um spots."""
    gx, gy = rng.uniform(100, 900, 8), rng.uniform(100, 900, 8)
    rows = []
    for c in range(n_clones):
        home = rng.integers(0, 8)
        for _ in range(int(rng.integers(1, 13))):
            if rng.random() < 0.7:
                k = home if confined else rng.integers(0, 8)
                r, a = 60 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
                cx, cy, reg = gx[k] + r * np.cos(a), gy[k] + r * np.sin(a), int(k)
            else:
                cx, cy, reg = rng.uniform(0, 1000), rng.uniform(0, 1000), -1
            m = 1 + int(rng.poisson(3))
            if dnb:
                rr, tt = 4 * np.sqrt(rng.random(m)), rng.uniform(0, 2 * np.pi, m)
                px, py = np.round((cx + rr * np.cos(tt)) * 2) / 2, np.round((cy + rr * np.sin(tt)) * 2) / 2
            else:
                px, py = np.full(m, np.round(cx / 10) * 10), np.full(m, np.round(cy / 10) * 10)
            rows += [(f"c{c}", f"{a:g}_{b:g}", a, b, reg) for a, b in zip(px, py, strict=True)]
    return pd.DataFrame(rows, columns=["clone_id", "spot_id", "x_um", "y_um", "gid"])


def test_unit_rules_and_record_weights():
    rng = np.random.default_rng(30)
    t = _cells(rng, dnb=True, n_clones=60)
    auto = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=9, rng=0)
    assert set(auto.table.units) == {"loci (15 um)"}
    from spaclone.pp import merge_loci

    n_loci = len(merge_loci(t.assign(r=t.gid.astype(str)), by="r").loci.query("r != '-1'"))
    assert (auto.table.records_in_regions == n_loci).all()
    spot = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=9, rng=0, unit_key="spot")
    assert set(spot.table.units) == {"(spot_id, clone)"} and (spot.table.records_in_regions > n_loci).all()
    ids = tl.region_sharing(t.assign(locus_id=t.spot_id), region_key="gid", chain_key=None, n_perm=9, rng=0)
    assert set(ids.table.units) == {"locus_id"}
    with pytest.warns(SpacloneWarning, match="one by one"):
        rec = tl.region_sharing(t[["clone_id", "gid"]], region_key="gid", chain_key=None, n_perm=9, rng=0)
    assert (
        set(rec.table.units) == {"records (no unit column)"}
        and (rec.table.records_in_regions == (t.gid >= 0).sum()).all()
    )
    # presence counts units, the main-region share weighs them by records
    clone, region = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0])
    assert _stats(clone, region, 2, 2, np.array([3.0, 1.0, 1.0, 1.0])) == (0.5, pytest.approx((0.75 + 1.0) / 2), 2)
    assert set(auto.table.alternative) == {"less", "greater"}
    two = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=9, rng=0, alternative="two-sided")
    assert set(two.table.alternative) == {"two-sided"}


@pytest.mark.parametrize("reps", [16, pytest.param(300, marks=pytest.mark.slow)])
def test_cells_not_molecules_are_the_unit(reps):
    """Random placement of cells with several molecules each: records as units call clones confined in nearly every
    chip; cell-scale units (default, confinement direction) do not."""
    rng = np.random.default_rng(31)
    hits = {"auto": 0, "records": 0}
    n = 0
    for i in range(reps):
        t = _cells(rng, dnb=bool(i % 2))
        for name, uk in (("auto", "auto"), ("records", None)):
            r = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=49, rng=i, unit_key=uk)
            hits[name] += int((r.table.p_value <= 0.05).sum())
        n += 4
    assert hits["auto"] / n <= 0.05 + TOL * np.sqrt(0.05 * 0.95 / n)
    assert hits["records"] / n > 0.5


def test_confined_cells_are_detected_with_cell_units():
    t = _cells(np.random.default_rng(32), confined=True, dnb=True)
    r = tl.region_sharing(t, region_key="gid", chain_key=None, n_perm=199, rng=1)
    assert (r.table.p_value <= 0.01).all()
