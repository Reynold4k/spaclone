import numpy as np
import pandas as pd
import pytest

from spaclone.pp import clonotype_key, flag_carryover, flag_public


def _rec(sample, locus, jx, n, v="IGHV1-1*01", j="IGHJ1*01", nadd=10.0):
    return pd.DataFrame(
        {
            "library_id": sample,
            "spot_id": [f"{sample}{jx}{i}" for i in range(n)],
            "locus": locus,
            "junction_aa": jx,
            "v_call": v,
            "j_call": j,
            "n_additions": nadd,
        }
    )


def _two_mice():
    return pd.concat(
        [
            _rec("A", "IGH", "CARCARRYW", 1000),
            _rec("B", "IGH", "CARCARRYW", 5),  # 200x: carry-over in B
            _rec("A", "IGH", "CARSHAREW", 100),
            _rec("B", "IGH", "CARSHAREW", 50),  # comparable heavy: unexplained
            _rec("A", "IGK", "CQQSHAREF", 100, v="IGKV1", j="IGKJ1"),
            _rec("B", "IGK", "CQQSHAREF", 50, v="IGKV1", j="IGKJ1"),  # comparable light: convergent -> public
            _rec("A", "IGH", "CARPUBW", 1000, nadd=1.0),
            _rec("B", "IGH", "CARPUBW", 1, nadd=1.0),  # germline-like
            _rec("A", "IGH", "CARONLYAW", 30),
            _rec("B", "IGH", "CARONLYBW", 30),  # not shared
        ],
        ignore_index=True,
    )


def test_clonotype_key():
    k = clonotype_key(_rec("A", "IGH", "CARW", 1, v="IGHV1-15*01,IGHV14-2*01|IGHV1-2"))
    assert k.iloc[0] == "IGH|IGHJ1|CARW|IGHV1,IGHV14"
    assert clonotype_key(_rec("A", "IGH", "CARW", 1), v_level=None).iloc[0] == "IGH|IGHJ1|CARW|"
    assert clonotype_key(_rec("A", "IGH", "CARW", 1), v_level="gene").iloc[0] == "IGH|IGHJ1|CARW|IGHV1-1"
    k = clonotype_key(_rec("A", "IGH", "CARW", 1).assign(junction_aa=None))
    assert k.isna().all()


def test_flag_public_reference_individuals_and_germline_like():
    air = _two_mice()
    ref = pd.DataFrame(
        {
            "locus": "IGH",
            "v_call": ["IGHV1-5*01", "IGHV1-9", "IGHV2-1", "IGHV1-1"],
            "j_call": "IGHJ1",
            "junction_aa": ["CARSHAREW", "CARSHAREW", "CARCARRYW", "CARCARRYW"],
            "individual": ["m1", "m2", "m1", "m2"],
        }
    )
    flagged, t = flag_public(air, reference=ref, n_key="n_additions", max_n=2)
    t = t.set_index("junction_aa")
    assert t.loc["CARSHAREW", "n_reference_individuals"] == 2 and t.loc["CARSHAREW", "public"]  # V family IGHV1
    assert t.loc["CARCARRYW", "n_reference_individuals"] == 1  # the IGHV2 cell has another V family
    assert t.loc["CARPUBW", "germline_like"] and t.loc["CARPUBW", "reason"] == "germline-like"
    assert not t.loc["CQQSHAREF", "germline_like"]  # light chains are not judged by N
    assert flagged.public.sum() == (flagged.junction_aa.isin(["CARSHAREW", "CARPUBW"])).sum()
    # individuals counted in air only when asked
    _, t2 = flag_public(air, individual_key="library_id")
    assert t2.set_index("junction_aa").loc["CARSHAREW", "public"]
    assert not t2.set_index("junction_aa").loc["CARONLYAW", "public"]
    with pytest.raises(KeyError):
        flag_public(air, reference=ref.drop(columns="individual"))


def test_flag_carryover_classes_and_record_flags():
    air = _two_mice()
    pub = flag_public(air, n_key="n_additions")
    flagged, t = flag_carryover(air, sample_key="library_id", public=pub)
    t = t.set_index("junction_aa")
    assert len(t) == 4
    assert t.loc["CARCARRYW", "class"] == "carryover" and t.loc["CARCARRYW", "source"] == "A"
    assert t.loc["CARCARRYW", "minor_over_major"] == pytest.approx(5 / 1000)
    assert t.loc["CARSHAREW", "class"] == "unexplained"
    assert t.loc["CQQSHAREF", "class"] == "public" and t.loc["CQQSHAREF", "reason"] == "convergent locus"
    assert t.loc["CARPUBW", "class"] == "public" and t.loc["CARPUBW", "reason"] == "public"
    carry = flagged[flagged.carryover]
    assert set(carry.library_id) == {"B"} and set(carry.junction_aa) == {"CARCARRYW"} and len(carry) == 5
    assert flagged.shared_class[flagged.junction_aa == "CARONLYAW"].isna().all()
    # without public calls the germline-like 1000:1 clonotype is carry-over too; the boundary is >= min_ratio
    _, t0 = flag_carryover(air)
    assert t0.set_index("junction_aa").loc["CARPUBW", "class"] == "carryover"
    _, t1 = flag_carryover(air, min_ratio=200)
    assert t1.set_index("junction_aa").loc["CARCARRYW", "class"] == "carryover"
    _, t2 = flag_carryover(air, min_ratio=201)
    assert t2.set_index("junction_aa").loc["CARCARRYW", "class"] == "unexplained"
    # convergent light chains can be switched off
    _, t3 = flag_carryover(air, convergent_loci=())
    assert t3.set_index("junction_aa").loc["CQQSHAREF", "class"] == "unexplained"


def test_carryover_with_three_samples_and_other_values():
    air = pd.concat(
        [_rec("A", "IGH", "CARXW", 500), _rec("B", "IGH", "CARXW", 400), _rec("C", "IGH", "CARXW", 2)],
        ignore_index=True,
    )
    flagged, t = flag_carryover(air)
    assert t["class"].iloc[0] == "unexplained" and t.n_samples.iloc[0] == 3
    assert set(flagged.library_id[flagged.carryover]) == {"C"}
    reads = air.assign(consensus_count=np.where(air.library_id == "C", 1000.0, 1.0))
    _, tr = flag_carryover(reads, value="reads")
    assert tr.source.iloc[0] == "C"
    with pytest.raises(KeyError):
        flag_carryover(air, sample_key="mouse")
    with pytest.raises(ValueError):
        flag_carryover(air, min_ratio=0.5)
