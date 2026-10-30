import numpy as np
import pandas as pd
import pytest

from spaclone.pp import find_error_variants, merge_error_variants, v_families


def _units(spec):
    """spec: list of (clone, locus, junction, v, j, n_records); one record per spot s<clone><i>."""
    rows = []
    for clone, locus, jx, v, j, n in spec:
        rows += [
            {
                "spot_id": f"{clone}{i}",
                "locus": locus,
                "clone_id": clone,
                "junction_aa": jx,
                "v_call": v,
                "j_call": j,
                "consensus_count": 1.0,
            }
            for i in range(n)
        ]
    return pd.DataFrame(rows)


def test_satellite_rule_known_answers():
    air = _units(
        [
            ("P", "IGH", "CARDYW", "IGHV1-1*01", "IGHJ1*01", 20),
            ("S1", "IGH", "CARDFW", "IGHV1-2*01", "IGHJ1*01", 4),  # 1 aa, same J, V family IGHV1: 20% -> satellite
            ("S2", "IGH", "CARDFW", "IGHV1-5", "IGHJ1", 5),  # 25% > 20% -> not a satellite of P
            ("S3", "IGH", "CARDYW", "IGHV1-1", "IGHJ2", 1),  # other J
            ("S4", "IGH", "CARDYW", "IGHV2-1", "IGHJ1", 1),  # other V family
            ("S5", "IGH", "CAREFW", "IGHV1-1", "IGHJ1", 1),  # 2 aa from P (and 1 from S1, a satellite: no chains)
            ("S6", "IGH", "CARDYYW", "IGHV1-1", "IGHJ1", 1),  # other length
            ("K", "IGK", "CQQYF", "IGKV1", "IGKJ1", 40),
            ("K1", "IGK", "CQQLF", "IGKV1", "IGKJ1", 2),  # 5% -> satellite
            ("K2", "IGK", "CQQSF", "IGKV1", "IGKJ1", 3),  # 7.5% > 5% light share
            ("K3", "IGK", "CQQYF", "IGKV1", "IGKJ1", 1),  # synonymous (same aa), different clone id
        ]
    )
    sat = find_error_variants(air)
    pairs = dict(zip(sat.satellite, sat.parent, strict=True))
    # S5 (CAREFW) differs from S2 (CARDFW) by one residue and S2 is no satellite, but 1/5 = 20%: S5 joins S2
    assert pairs == {"S1": "P", "S5": "S2", "K1": "K", "K3": "K"}
    assert set(sat.columns) >= {"satellite", "parent", "locus", "aa_diff", "size_share"}
    assert sat.set_index("satellite").loc["K3", "aa_diff"] == 0
    # options: light share 0.1 admits K2; no V check admits S4; synonymous-only
    assert "K2" in set(find_error_variants(air, light_max_share=0.1).satellite)
    assert "S4" in set(find_error_variants(air, match_v=None).satellite)
    assert set(find_error_variants(air, max_aa_diff=0).satellite) == {"K3"}
    # TCR chains use the light share by default
    tcr = _units([("T", "TRB", "CASSF", "TRBV1", "TRBJ1", 20), ("T1", "TRB", "CASTF", "TRBV1", "TRBJ1", 2)])
    assert find_error_variants(tcr).empty
    assert len(find_error_variants(tcr, heavy_loci=("TRB",))) == 1


def test_ties_and_order_option_and_row_order_invariance():
    air = _units(
        [
            ("A", "IGH", "CARDYW", "IGHV1", "IGHJ1", 10),
            ("B", "IGH", "CARDFW", "IGHV1", "IGHJ1", 10),  # tied with A, both 1 aa from S
            ("S", "IGH", "CARDHW", "IGHV1", "IGHJ1", 1),
        ]
    )
    air.loc[air.clone_id == "B", "consensus_count"] = 2.0  # B has more reads: visited first
    sat = find_error_variants(air)
    assert dict(zip(sat.satellite, sat.parent, strict=True)) == {"S": "B"}
    sat = find_error_variants(air, order=["A", "B", "S"])
    assert dict(zip(sat.satellite, sat.parent, strict=True)) == {"S": "A"}
    perm = air.sample(frac=1, random_state=4)
    pd.testing.assert_frame_equal(find_error_variants(air), find_error_variants(perm), check_like=True)


def test_merge_relabels_takes_parent_sequence_and_recollapses():
    air = _units([("P", "IGH", "CARDYW", "IGHV1-1", "IGHJ1", 10), ("S", "IGH", "CARDFW", "IGHV1-1", "IGHJ1", 2)])
    air.loc[air.clone_id == "S", "spot_id"] = ["P0", "Z"]  # one satellite record shares a spot with the parent
    merged, sat = merge_error_variants(air)
    assert len(sat) == 1 and (merged.clone_id == "P").all()
    assert len(merged) == 11  # the copy on spot P0 is the parent's molecule
    assert merged.set_index("spot_id").loc["P0", "consensus_count"] == 2.0
    assert (merged.junction_aa == "CARDYW").all()
    assert set(merged.clone_id_premerge) == {"P", "S"}
    st = merged.attrs["spaclone"]["merge_error_variants"]
    assert st["records_relabelled"] == 2 and st["records_absorbed"] == 1
    assert st["absorbed_share_of_relabelled"] == pytest.approx(0.5)
    kept, _ = merge_error_variants(air, recollapse=False, premerge_key=None)
    assert len(kept) == 12 and "clone_id_premerge" not in kept.columns


def test_merge_with_a_precomputed_map_from_another_table():
    big = _units([("P", "IGH", "CARDYW", "IGHV1", "IGHJ1", 10), ("S", "IGH", "CARDFW", "IGHV1", "IGHJ1", 2)])
    sat = find_error_variants(big)
    sub = big[big.clone_id == "S"]  # the parent has no record here
    merged, used = merge_error_variants(sub, variants=sat)
    assert (merged.clone_id == "P").all() and (merged.junction_aa == "CARDYW").all()
    with pytest.raises(ValueError):
        merge_error_variants(sub, variants=pd.DataFrame({"a": [1]}))


def test_independent_clones_are_not_merged_and_planted_errors_are():
    """Null and power: random 13-aa junctions have no 1-aa neighbours; planted 1-aa error copies are all found."""
    rng = np.random.default_rng(3)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    spec = [
        (f"c{i}", "IGK", "C" + "".join(rng.choice(aa, 11)) + "F", "IGKV1", "IGKJ1", int(rng.integers(1, 50)))
        for i in range(300)
    ]
    assert find_error_variants(_units(spec)).empty
    big = spec[0]
    errs = []
    for k in range(5):
        s = list(big[2])
        p = 1 + k
        s[p] = "W" if s[p] != "W" else "Y"
        errs.append((f"e{k}", "IGK", "".join(s), "IGKV1", "IGKJ1", 1))
    big = (big[0], big[1], big[2], big[3], big[4], 100)
    sat = find_error_variants(_units([big, *spec[1:], *errs]))
    assert set(sat.satellite) == {f"e{k}" for k in range(5)} and set(sat.parent) == {"c0"}


def test_v_families_and_errors():
    f = v_families(pd.Series(["IGHV1-15*01,IGHV14-2*01", "TRBV12-1*01|TRBV12-2", None, "IGKV4S2", ""]))
    assert f.iloc[0] == frozenset({"IGHV1", "IGHV14"}) and f.iloc[1] == frozenset({"TRBV12"})
    assert f.iloc[2] == frozenset() and f.iloc[4] == frozenset() and f.iloc[3] == frozenset({"IGKV4"})
    assert v_families(["IGHV1-15*01"], level="gene").iloc[0] == frozenset({"IGHV1-15"})
    air = _units([("P", "IGH", "CARDYW", "IGHV1", "IGHJ1", 2)])
    with pytest.raises(ValueError):
        find_error_variants(air, heavy_max_share=2)
    with pytest.raises(KeyError):
        find_error_variants(air.drop(columns="v_call"))
    assert find_error_variants(air.drop(columns="v_call"), match_v=None).empty
