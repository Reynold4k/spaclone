import numpy as np
import pandas as pd
import pytest

from spaclone.pp import barcode_sinks, spot_multiplicity


def _air():
    rows = []
    # spot a: 22 IGK clones + 3 IGH clones (a sink at 20), clone k0 twice
    rows += [("a", 0.0, 0.0, "IGK", f"k{i}", 1.0) for i in range(22)] + [("a", 0.0, 0.0, "IGK", "k0", 5.0)]
    rows += [("a", 0.0, 0.0, "IGH", f"h{i}", 2.0) for i in range(3)]
    # spot b: one clone, three records
    rows += [("b", 1.0, 0.0, "IGK", "k1", 3.0)] * 3
    # spot c: the same clone id under two loci counts twice
    rows += [("c", 2.0, 0.0, "IGK", "x", 1.0), ("c", 2.0, 0.0, "IGH", "x", 1.0)]
    return pd.DataFrame(rows, columns=["spot_id", "x_um", "y_um", "locus", "clone_id", "consensus_count"])


def test_spot_multiplicity_known_answer():
    air = _air()
    rec = spot_multiplicity(air)
    assert list(rec.columns[-2:]) == ["mult_all", "mult_locus"]
    a = rec[rec.spot_id == "a"]
    assert (a.mult_all == 25).all()
    assert (a[a.locus == "IGK"].mult_locus == 22).all() and (a[a.locus == "IGH"].mult_locus == 3).all()
    assert (rec[rec.spot_id == "b"].mult_all == 1).all()
    assert (rec[rec.spot_id == "c"].mult_all == 2).all() and (rec[rec.spot_id == "c"].mult_locus == 1).all()
    sp = spot_multiplicity(air, level="spot").set_index("spot_id")
    assert sp.loc["a", "n_records"] == 26 and sp.loc["a", "n_clones"] == 25
    assert sp.loc["a", "n_clones_IGK"] == 22 and sp.loc["a", "n_clones_IGH"] == 3
    assert sp.loc["b", "n_records"] == 3 and sp.loc["b", "n_clones"] == 1
    assert sp.loc["c", "x_um"] == 2.0
    with pytest.raises(ValueError):
        spot_multiplicity(air, level="cell")


def test_barcode_sinks_known_answer():
    air = _air()
    s = barcode_sinks(air, min_clones=20)
    assert len(s.table) == 1
    row = s.table.iloc[0]
    assert row.spot_id == "a" and row.n_clones == 25 and row.n_records == 26
    assert row.dominant_locus == "IGK" and row.dominant_share == pytest.approx(22 / 25) and row.n_loci == 2
    assert row.n_reads == pytest.approx(22 + 5 + 6)
    assert s.mask.sum() == 26 and s.mask[:26].all() and not s.mask[26:].any()
    assert s.summary == {
        "n_sinks": 1,
        "max_clones": 25,
        "n_records_on_sinks": 26,
        "share_records_on_sinks": pytest.approx(26 / 31),
        "min_clones": 20,
    }
    none = barcode_sinks(air, min_clones=30)
    assert len(none.table) == 0 and not none.mask.any() and none.summary["max_clones"] == 0
    two = barcode_sinks(air, min_clones=2)
    assert list(two.table.spot_id) == ["a", "c"]
    with pytest.raises(ValueError):
        barcode_sinks(air, min_clones=0)
    bad = air.copy()
    bad.loc[0, "clone_id"] = None
    with pytest.raises(ValueError, match="missing clone"):
        barcode_sinks(bad)


def test_invariance_and_custom_keys():
    air = _air()
    shuffled = air.sample(frac=1.0, random_state=3)
    a = barcode_sinks(air).table
    b = barcode_sinks(shuffled).table
    pd.testing.assert_frame_equal(a, b)
    m = barcode_sinks(shuffled).mask
    assert (shuffled.spot_id.to_numpy()[m] == "a").all()
    renamed = air.rename(columns={"spot_id": "bead", "clone_id": "ct", "locus": "chain"})
    c = barcode_sinks(renamed, spot_key="bead", clone_key="ct", locus_key="chain", reads_key=None).table
    assert c.bead.tolist() == ["a"] and "n_reads" not in c.columns


def test_random_placement_makes_no_sinks_and_planted_sinks_are_found():
    rng = np.random.default_rng(0)
    n = 20000
    air = pd.DataFrame(
        {
            "spot_id": rng.integers(0, 200_000, n).astype(str),
            "x_um": 0.0,
            "y_um": 0.0,
            "locus": rng.choice(["IGH", "IGK"], n),
            "clone_id": rng.integers(0, 3000, n).astype(str),
        }
    )
    assert len(barcode_sinks(air).table) == 0
    planted = pd.DataFrame(
        {
            "spot_id": np.repeat(["S1", "S2"], [40, 25]),
            "x_um": 0.0,
            "y_um": 0.0,
            "locus": "IGK",
            "clone_id": [f"p{i}" for i in range(65)],
        }  # fmt: skip
    )
    s = barcode_sinks(pd.concat([air, planted], ignore_index=True))
    assert s.table.spot_id.tolist() == ["S1", "S2"] and s.table.n_clones.tolist() == [40, 25]
