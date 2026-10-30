import pandas as pd
import pytest

from spaclone import tl


def _mouse():
    return pd.DataFrame(
        {
            "locus": ["TRA", "TRA", "TRA", "TRA", "TRB", "TRA", "TRA"],
            "v_call": ["TRAV11*01", "TRAV11D*01,TRAV11*02", "TRAV1*01", "TRAV11*01", "TRBV13-2*01", "TRAV11N", "TRAV2"],
            "j_call": ["TRAJ18*01", "TRAJ18*01", "TRAJ33*01", "TRAJ18*01", "TRBJ2-7*01", "TRAJ18", "TRAJ18"],
            "junction_aa": [
                "CVVGDRGSALGRLHF",
                "CVVGDRGSALGRLHF",
                "CAVRDSNYQLIW",
                "CVVGDRGSALGRLYF",  # one substitution: not canonical
                "CASSQDRGYEQYF",
                "CVVGDRGSALGRLHF",
                "CVVGDRGSALGRLHF",  # wrong V gene
            ],
        }
    )


def test_mouse_canonical_rule():
    out = tl.invariant_tcr(_mouse())
    assert out["inkt"].tolist() == [True, True, False, False, False, True, False]
    assert out["mait"].tolist() == [False, False, True, False, False, False, False]
    assert out["invariant"].iloc[2] == "MAIT" and pd.isna(out["invariant"].iloc[3])
    assert out["inkt_beta_v"].tolist() == [False, False, False, False, True, False, False]
    assert "inkt" not in _mouse().columns  # input untouched


def test_genes_rule_and_locus_inference():
    out = tl.invariant_tcr(_mouse(), rule="genes")
    assert out["inkt"].tolist() == [True, True, False, True, False, True, False]
    no_locus = tl.invariant_tcr(_mouse().drop(columns="locus"))
    assert no_locus["inkt"].tolist() == tl.invariant_tcr(_mouse())["inkt"].tolist()


def test_human_rules():
    t = pd.DataFrame(
        {
            "locus": "TRA",
            "v_call": ["TRAV10*01", "TRAV1-2*01", "TRAV1-2*01", "TRAV1-2*01", "TRAV1-2*01"],
            "j_call": ["TRAJ18*01", "TRAJ33*01", "TRAJ12*01", "TRAJ20*01", "TRAJ33*01"],
            "junction_aa": ["CVVSDRGSTLGRLYF", "CAVMDSNYQLIW", "CAVMDSSYKLIF", "CAVSDNDYKLSF", "CAVRMDSNYQLIW"],
        }
    )
    out = tl.invariant_tcr(t, species="human")
    assert out["inkt"].tolist() == [True, False, False, False, False]
    assert out["mait"].tolist() == [False, True, True, True, False]  # 13-residue CDR3 is not canonical
    assert tl.invariant_tcr(t, species="human", rule="genes")["mait"].iloc[4]


def test_custom_rules_and_errors():
    rules = {"Vg9Vd2": {"locus": "TRG", "v": ("TRGV9",), "j": ("TRGJP",), "junction_length": (14, 15)}}
    t = pd.DataFrame(
        {"locus": ["TRG"], "v_call": ["TRGV9*01"], "j_call": ["TRGJP*01"], "junction_aa": ["CALWEVQELGKKIKF"]}
    )
    assert tl.invariant_tcr(t, rules=rules)["vg9vd2"].iloc[0]
    with pytest.raises(ValueError, match="species"):
        tl.invariant_tcr(t, species="rat")
    with pytest.raises(KeyError):
        tl.invariant_tcr(t.drop(columns="j_call"))
