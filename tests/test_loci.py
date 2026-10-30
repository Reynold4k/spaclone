import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from spaclone.pp import merge_loci


def _air(rows):
    df = pd.DataFrame(rows, columns=["spot_id", "x_um", "y_um", "locus", "clone_id", "consensus_count"])
    return df


def _partition(loci, member, air):
    """Set of frozensets of record indices per locus."""
    return {frozenset(air.index[member.to_numpy() == k]) for k in loci.locus_id}


def test_known_answer_grid():
    air = _air(
        [
            ("a", 0.0, 0.0, "IGH", "A", 5),
            ("a", 0.0, 0.0, "IGH", "A", 9),  # second record of A on the same spot
            ("b", 10.0, 0.0, "IGH", "A", 2),
            ("c", 20.0, 10.0, "IGH", "A", 1),  # diagonal to b: 14.1 um
            ("d", 50.0, 50.0, "IGH", "A", 3),
            ("a", 0.0, 0.0, "IGH", "B", 4),
        ]
    )
    loci, member = merge_loci(air)
    assert len(loci) == 3
    a = loci[loci.clone_id == "A"].sort_values("n_spots")
    assert a.n_spots.tolist() == [1, 3]
    big = a.iloc[1]
    assert big.n_records == 4
    assert big.x_um == pytest.approx(10.0)
    assert big.y_um == pytest.approx(10 / 3)
    assert big.reads_max == 9
    assert big.reads_sum == 17
    assert big.spot_id == "a"  # spot of the record with most reads
    assert member.iloc[0] == member.iloc[1] == member.iloc[2] == member.iloc[3]
    assert member.iloc[5] != member.iloc[0]
    assert set(loci.columns) >= {"locus_id", "clone_id", "locus", "x_um", "y_um", "n_spots", "n_records", "reads_max"}


def test_radius_is_inclusive_and_single_linkage():
    # 0, 15, 30 um along x: exactly at the radius, chained
    air = _air([(f"s{i}", 15.0 * i, 0.0, "IGK", "C", 1) for i in range(3)])
    loci, _ = merge_loci(air, radius_um=15)
    assert len(loci) == 1 and loci.n_spots.iloc[0] == 3
    loci, _ = merge_loci(air, radius_um=14.99)
    assert len(loci) == 3
    # 10-um grid: diagonal (14.1) linked at 15, a 20-um gap not
    air = _air([("p", 0.0, 0.0, "IGK", "C", 1), ("q", 10.0, 10.0, "IGK", "C", 1), ("r", 30.0, 10.0, "IGK", "C", 1)])
    loci, _ = merge_loci(air, radius_um=15)
    assert sorted(loci.n_spots) == [1, 2]
    loci, _ = merge_loci(air, radius_um=20)
    assert loci.n_spots.tolist() == [3]


def test_radius_zero_merges_only_same_position():
    air = _air([("p", 0.0, 0.0, "IGK", "C", 1), ("p", 0.0, 0.0, "IGK", "C", 1), ("q", 10.0, 0.0, "IGK", "C", 1)])
    loci, member = merge_loci(air, radius_um=0)
    assert len(loci) == 2
    assert member.iloc[0] == member.iloc[1]


def test_strata_keep_libraries_and_chains_apart():
    air = _air([("p", 0.0, 0.0, "IGH", "C", 1), ("q", 5.0, 0.0, "IGK", "C", 1)])
    air["library_id"] = ["L1", "L1"]
    loci, _ = merge_loci(air)
    assert len(loci) == 2  # same clone id on two chains: separate by default
    loci, _ = merge_loci(air, by=())
    assert len(loci) == 1
    air["locus"] = "IGH"
    air["library_id"] = ["L1", "L2"]
    loci, _ = merge_loci(air)
    assert len(loci) == 2
    assert set(loci.library_id) == {"L1", "L2"}
    with pytest.raises(KeyError):
        merge_loci(air, by="sample")


def test_matches_brute_force_and_is_order_and_translation_invariant():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.integers(0, 60, n) * 10.0
    y = rng.integers(0, 40, n) * 10.0
    air = pd.DataFrame(
        {
            "spot_id": [f"{a:.0f}_{b:.0f}" for a, b in zip(x, y, strict=True)],
            "x_um": x,
            "y_um": y,
            "locus": rng.choice(["IGH", "IGK"], n),
            "clone_id": rng.choice([f"c{i}" for i in range(25)], n),
            "consensus_count": rng.integers(1, 100, n).astype(float),
        }
    )
    loci, member = merge_loci(air)
    # brute force: per (locus, clone), components of distinct spots within 15 um
    expected = set()
    for _, g in air.groupby(["locus", "clone_id"]):
        sites = g.drop_duplicates("spot_id")
        d = squareform(pdist(sites[["x_um", "y_um"]].to_numpy())) if len(sites) > 1 else np.zeros((1, 1))
        _, lab = connected_components(d <= 15.0 + 1e-9, directed=False)
        spot_lab = dict(zip(sites.spot_id, lab, strict=True))
        for k in np.unique(lab):
            expected.add(frozenset(g.index[g.spot_id.map(spot_lab).to_numpy() == k]))
    assert _partition(loci, member, air) == expected
    # row order
    perm = rng.permutation(n)
    loci2, member2 = merge_loci(air.iloc[perm])
    pd.testing.assert_frame_equal(loci.reset_index(drop=True), loci2.reset_index(drop=True))
    pd.testing.assert_series_equal(member.sort_index(), member2.sort_index())
    # translation by a non-dyadic amount keeps the partition
    shifted = air.assign(x_um=air.x_um + 1234.567, y_um=air.y_um - 89.25)
    loci3, member3 = merge_loci(shifted)
    assert _partition(loci3, member3, shifted) == expected
    np.testing.assert_allclose(loci3.x_um - 1234.567, loci.x_um, atol=1e-6)


def test_custom_keys_and_missing_reads():
    air = pd.DataFrame({"bead": ["a", "b"], "px": [0.0, 3.0], "py": [0.0, 0.0], "ct": ["k", "k"]})
    loci, member = merge_loci(air, clone_key="ct", spot_key="bead", x_key="px", y_key="py")
    assert len(loci) == 1
    assert {"ct", "bead", "px", "py"} <= set(loci.columns)
    assert np.isnan(loci.reads_max.iloc[0])
    assert loci["locus"].isna().all()
    assert member.tolist() == [0, 0]


def test_empty_and_invalid_input():
    air = _air([])
    loci, member = merge_loci(air.astype({"x_um": float, "y_um": float}))
    assert len(loci) == 0 and len(member) == 0
    assert "locus_id" in loci.columns
    bad = _air([("p", 0.0, 0.0, "IGH", None, 1)])
    with pytest.raises(ValueError, match="missing"):
        merge_loci(bad)
    ok = _air([("p", 0.0, 0.0, "IGH", "A", 1)])
    with pytest.raises(ValueError, match="radius_um"):
        merge_loci(ok, radius_um=-1)
    with pytest.raises(ValueError, match="lacks column"):
        merge_loci(ok.drop(columns="x_um"))
