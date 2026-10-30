import numpy as np
import pandas as pd
import pytest

from spaclone.pp import filter_records, filter_valid, resolve_min_reads


def _toy():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e", "e"],
            "x_um": [0.0, 0, 10, 20, 20, 30, 40, 40],
            "y_um": [0.0] * 8,
            "locus": ["IGH", "IGK", "IGH", "TRA", "TRB", "IGH", "IGK", "IGK"],
            "clone_id": ["h1", "k1", "h1", "a1", "b1", "h2", "k1", "k2"],
            "junction_aa": ["CARDYW", "CQQYF", "CAR*YW", "CAVRDSNYQLIW", "CASSF", "CARDYW", "CQQYF", "CQQLF"],
            "productive": ["T", "T", "T", "F", "T", "TRUE", "1", "T"],
            "consensus_count": [5, 1, 9, 3, 2, np.nan, 2, 7],
            "in_tissue": [True, True, True, True, False, True, True, True],
        }
    )


def test_steps_and_removal_report():
    ok, rep = filter_records(_toy(), min_reads=2, on_tissue=True, loci=("IGH", "IGK", "TRB"))
    # valid: row 2 fails; productive: row 3 (TRA "F"); reads: row 1 (1 read), row 5 (NaN); tissue: row 4
    assert ok.index.tolist() == [0, 6, 7]
    assert rep.rule.iloc[0] == "input"
    assert rep.records.tolist() == [8, 7, 6, 4, 3, 3]
    assert rep.removed.tolist() == [0, 1, 1, 2, 1, 0]
    assert rep.spots.tolist() == [5, 4, 4, 3, 2, 2]
    assert rep.clones.tolist() == [6, 6, 5, 4, 3, 3]
    assert rep.clones_removed.tolist() == [0, 0, 1, 1, 1, 0]
    assert rep.failing.tolist() == [0, 1, 1, 2, 1, 1]
    assert ok.attrs["spaclone"]["filter_records"]["min_reads"] == 2


def test_junction_step_is_m4_filter_valid():
    air = _toy()
    ok, _ = filter_records(air, productive=False)
    ref, _ = filter_valid(air, require_productive=False)
    assert ok.index.equals(ref.index)
    ok2, _ = filter_records(air, min_reads=2)
    ref2, _ = filter_valid(air, require_productive=True, min_reads=2)
    assert ok2.index.equals(ref2.index)


def test_in_frame_rule_and_skipped_steps():
    air = _toy().assign(junction_aa=["CARDYW", "CQQ_F", "AR*Y", "CAVR", None, "XXXX", "CQQYF", "CQQLF"])
    ok, rep = filter_records(air, valid_junction="in_frame", productive=False)
    # no stop / frame shift: rows 1 (_), 2 (*) and the missing junction fail; anchors and length are not checked
    assert ok.index.tolist() == [0, 3, 5, 6, 7]
    ok, rep = filter_records(air, valid_junction=False, productive=False)
    assert len(ok) == len(air) and not rep.applied.iloc[1]
    ok, rep = filter_records(_toy().drop(columns="productive"))
    assert not rep.applied.iloc[2] and "no 'productive' column" in rep.rule.iloc[2]


def test_per_locus_floor_and_sinks():
    air = _toy()
    ok, rep = filter_records(air, valid_junction=False, productive=False, min_reads={"IGK": 2, "TRB": 3})
    # IGK row 1 (1 read) and TRB row 4 (2 reads) go; IGH keeps its NaN-read record (no floor for IGH)
    assert ok.index.tolist() == [0, 2, 3, 5, 6, 7]
    sink = pd.DataFrame(
        {
            "spot_id": "z",
            "x_um": 1.0,
            "y_um": 1.0,
            "locus": "IGK",
            "clone_id": [f"s{i}" for i in range(20)],
            "junction_aa": "CQQYF",
            "productive": "T",
        }
    )
    both = pd.concat([air, sink], ignore_index=True)
    ok, rep = filter_records(both, valid_junction=False, productive=False, sink_min_clones=20)
    assert len(ok) == len(air) and rep.rule.iloc[-1].startswith("not on a barcode sink")
    assert rep.spots_removed.iloc[-1] == 1


def test_auto_floor_uses_read_floor_and_falls_back_to_the_inflection(monkeypatch):
    import importlib

    rng = np.random.default_rng(2)
    reads = np.r_[rng.geometric(0.8, 3000), np.round(rng.lognormal(np.log(50), 0.3, 1000))]
    air = pd.DataFrame(
        {
            "spot_id": np.arange(4000).astype(str),
            "locus": np.where(np.arange(4000) % 2, "TRB", "TRA"),
            "consensus_count": reads.astype(float),
            "junction_aa": "CASSF",
            "clone_id": "c",
        }
    )
    rf = importlib.import_module("spaclone.qc.read_floor")
    state = {"floor": 20.0, "fail": False}
    calls = []

    class _Res:
        @property
        def summary(self):
            return {"suggested_floor": state["floor"]}

    def fake(a, *, locus="TRB", partner_locus="TRA", **kw):
        calls.append((locus, partner_locus, kw.get("rng")))
        if state["fail"]:
            raise ValueError("cannot test this table")
        return _Res()

    monkeypatch.setattr(rf, "read_floor", fake, raising=False)
    floor, src = resolve_min_reads(air, "auto")
    assert floor == {"TRB": 20.0, "TRA": 20.0} and src.startswith("auto: qc.read_floor") and "TRB/TRA >= 20" in src
    assert calls == [("TRB", "TRA", 0)]  # only the pair present in the table, seeded
    ok, _ = filter_records(air, min_reads="auto")
    assert ok.consensus_count.min() >= 20
    calls.clear()
    floor, _ = resolve_min_reads(air, "auto", read_floor_kwargs={"locus": "IGH", "partner_locus": "IGK", "rng": 5})
    assert floor == {"IGH": 20.0, "IGK": 20.0} and calls == [("IGH", "IGK", 5)]
    both = pd.concat([air, air.assign(locus=air.locus.map({"TRB": "IGH", "TRA": "IGK"}))], ignore_index=True)
    calls.clear()
    floor, src = resolve_min_reads(both, "auto")
    assert [c[:2] for c in calls] == [("TRB", "TRA"), ("IGH", "IGK")] and set(floor) == {"TRB", "TRA", "IGH", "IGK"}
    # a tested pair without a suggestion gets no floor (not the inflection)
    state.update(floor=float("nan"))
    floor, src = resolve_min_reads(air, "auto")
    assert floor == {} and "no floor" in src
    ok, rep = filter_records(air, min_reads="auto")
    assert len(ok) == len(air) and rep.rule.iloc[3].startswith("read floor: none") and not rep.applied.iloc[3]
    # no pair can be tested: the inflection of the read-support curve, and the source says why
    for st, table in (({"floor": 20.0, "fail": True}, air), ({"floor": 20.0, "fail": False}, air.assign(locus="TRB"))):
        state.update(st)
        floor, src = resolve_min_reads(table, "auto")
        assert src.startswith("auto: inflection of the read-support curve") and 4 <= floor["TRB"] <= 30
        ok, rep = filter_records(table, min_reads="auto")
        assert "inflection" in rep.rule.iloc[3] and (ok.consensus_count[ok.locus == "TRB"] >= floor["TRB"]).all()


def test_auto_floor_is_deterministic_on_real_read_floor():
    from spaclone.sim import synthetic_chip

    air = synthetic_chip(genuine_fraction=0.3, rng=1, n_sinks=0).air
    a = resolve_min_reads(air, "auto", read_floor_kwargs={"n_perm": 39, "floors": (2, 5)})
    b = resolve_min_reads(air, "auto", read_floor_kwargs={"n_perm": 39, "floors": (2, 5)})
    assert a == b and a[1].startswith("auto: qc.read_floor") and "IGH/IGK" in a[1]


def test_strata_order_invariance_and_input_untouched():
    air = _toy()
    before = air.copy()
    ok, rep = filter_records(air, min_reads=2, by="locus")
    pd.testing.assert_frame_equal(air, before)
    last = rep[rep.step == rep.step.max()].set_index("locus")
    assert last.records.sum() == len(ok) and last.loc["TRA", "records"] == 0
    perm = air.sample(frac=1.0, random_state=1)
    ok2, rep2 = filter_records(perm, min_reads=2, by="locus")
    assert sorted(ok2.index) == sorted(ok.index)
    pd.testing.assert_frame_equal(rep, rep2)


def test_errors():
    air = _toy()
    with pytest.raises(ValueError):
        filter_records(air, valid_junction="strict")
    with pytest.raises(KeyError):
        filter_records(air, on_tissue="tissue")
    with pytest.raises(KeyError):
        filter_records(air.drop(columns="consensus_count"), min_reads=2)
    with pytest.raises(KeyError):
        filter_records(air.drop(columns="junction_aa"))
    with pytest.raises(ValueError):
        filter_records(air, min_reads="max")
