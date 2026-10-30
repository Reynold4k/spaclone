import numpy as np
import pandas as pd
import pytest

from spaclone.pp import filter_valid, is_valid_junction


def test_is_valid_junction_examples_of_the_standard():
    good = ["CARNSHYYGSRYPFAYW", "CQQYWSTPLTF", "CAVRDSNYQLIW", "CASSW", "CVVGDRGSALGRLHF"]
    bad = ["CARDY*YW", "ARNSHYYG", "CAR_DYW", "CARXDYW", "CAW", "CARDYV", "carnshyyw", "", None, np.nan, 5]
    assert is_valid_junction(good).all()
    assert not is_valid_junction(bad).any()
    assert is_valid_junction("CQQYWSTPLTF") is True
    assert is_valid_junction("CQQ") is False
    s = pd.Series(good + bad, index=np.arange(100, 100 + len(good) + len(bad)))
    np.testing.assert_array_equal(is_valid_junction(s), [True] * len(good) + [False] * len(bad))


def test_is_valid_junction_length_and_anchor_options():
    assert not is_valid_junction(["CAAAF"], min_length=6)[0]
    assert is_valid_junction(["CAAAAF"], min_length=6)[0]
    assert not is_valid_junction(["CAAAW"], last="F")[0]
    with pytest.raises(ValueError):
        is_valid_junction(["CF"], min_length=1)


def _toy():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e"],
            "locus": ["IGH", "IGK", "IGH", "TRA", "TRB", "IGH", "IGK"],
            "junction_aa": ["CARDYW", "CQQYF", "CAR*YW", "CAVRDSNYQLIW", "CASSF", "CARDYW", "CQQLF"],
            "productive": ["T", "T", "T", "F", True, "TRUE", 1],
            "consensus_count": [5, 1, 9, 3, 2, np.nan, 2],
        }
    )


def test_filter_valid_rules_and_report():
    air = _toy()
    ok, rep = filter_valid(air, min_reads=2)
    # a/IGH ok; a/IGK fails reads; b fails junction; c/TRA not productive; c/TRB ok; d unknown reads; e ok
    assert ok.index.tolist() == [0, 4, 6]
    assert rep.rule.tolist()[0] == "input"
    assert rep.records.tolist() == [7, 6, 5, 3]
    assert rep.removed.tolist() == [0, 1, 1, 2]
    assert rep.spots.tolist() == [5, 4, 4, 3]
    assert rep.failing.tolist() == [0, 1, 1, 2]
    # without the read floor
    ok2, rep2 = filter_valid(air)
    assert len(rep2) == 3 and len(ok2) == 5
    # productive column absent: rule reported as not applied
    ok3, rep3 = filter_valid(air.drop(columns="productive"))
    assert not rep3.applied.iloc[2] and len(ok3) == 6


def test_filter_valid_tissue_and_strata():
    air = _toy().assign(in_tissue=[True, True, True, True, False, True, True])
    ok, rep = filter_valid(air, tissue_key="in_tissue", by="locus")
    assert 4 not in ok.index
    assert set(rep.locus) == {"IGH", "IGK", "TRA", "TRB"}
    last = rep[rep.step == rep.step.max()].set_index("locus")
    assert last.loc["TRB", "records"] == 0
    assert last.records.sum() == len(ok)
    with pytest.raises(KeyError):
        filter_valid(air, min_reads=2, reads_key="reads")
    with pytest.raises(KeyError):
        filter_valid(air, tissue_key="tissue")


def test_filter_valid_keeps_index_and_does_not_modify_input():
    air = _toy().set_index(pd.Index(list("pqrstuv")))
    before = air.copy()
    ok, _ = filter_valid(air)
    pd.testing.assert_frame_equal(air, before)
    assert set(ok.index) <= set(air.index)
    assert ok.attrs["spaclone"]["filter_valid"]["min_reads"] is None
