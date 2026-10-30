"""Tests of spaclone.ext.dandelion: Dandelion clone calls on spatial records, version handling, determinism."""

import numpy as np
import pandas as pd
import pytest

from spaclone.ext import dandelion as dx
from spaclone.pp import define_clones

# A ~ B ~ C (one aa each, length 13: Dandelion's threshold floor(13 * 0.15) = 1), A and C two aa apart; D far away
A, B, C, D = "CAREEFLGGFDYW", "CAREEFLGGFDHW", "CAREEFLGGFNHW", "CARDDDLGGFNHW"


def _toy() -> pd.DataFrame:
    rows = []
    for k, (jaa, x) in enumerate([(A, 0), (A, 40), (B, 80), (C, 120), (D, 160)]):
        rows.append(
            {
                "spot_id": f"s{k}",
                "x_um": float(x),
                "y_um": 0.0,
                "locus": "IGH",
                "v_call": "IGHV1-7*01",
                "d_call": "",
                "j_call": "IGHJ2*01",
                "c_call": "IGHM",
                "junction_aa": jaa,
                "junction": "TGT" * len(jaa),
                "productive": "T",
                "consensus_count": 3.0,
                "library_id": "s1",
            }
        )
    rows.append(
        {**rows[0], "locus": "IGK", "v_call": "IGKV1-110*01", "j_call": "IGKJ1*01", "junction_aa": "CSQSTHVPFTF"}
    )
    rows.append({**rows[0], "spot_id": "s9", "v_call": None})  # unusable: no V call
    air = pd.DataFrame(rows)
    air.index = [f"rec{i}" for i in range(len(air))]
    return air


def _families(seed: int = 0, n: int = 120) -> pd.DataFrame:
    """Random heavy chains in a few V/J/length groups, junctions one or two aa from a few bases."""
    rng = np.random.default_rng(seed)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    bases = ["CAR" + "".join(rng.choice(aa, 9)) + "W" for _ in range(6)]
    rows = []
    for i in range(n):
        s = np.array(list(bases[rng.integers(0, len(bases))]))
        pos = rng.choice(np.arange(3, len(s) - 1), size=rng.integers(0, 3), replace=False)
        s[pos] = rng.choice(aa, len(pos))
        jaa = "".join(s)
        rows.append(
            {
                "spot_id": f"b{i}",
                "x_um": float(rng.uniform(0, 500)),
                "y_um": float(rng.uniform(0, 500)),
                "locus": "IGH",
                "v_call": f"IGHV{1 + i % 2}-1*01",
                "j_call": "IGHJ1*01",
                "c_call": "IGHM",
                "junction_aa": jaa,
                "junction": "TGT" * len(jaa),
                "productive": True,
                "consensus_count": float(rng.integers(1, 20)),
            }
        )
    return pd.DataFrame(rows)


def test_to_dandelion_table_units_and_filters():
    air = _toy()
    t = dx.to_dandelion_table(air)
    assert len(t) == len(air) - 1  # the record without a V call is left out
    assert t.sequence_id.is_unique and (t.sequence_id == t.cell_id).all()
    assert set(t.productive) == {"T"} and t.umi_count.eq(3.0).all()
    assert t.attrs["spaclone_rows"] == [0, 1, 2, 3, 4, 5]
    spot = dx.to_dandelion_table(air, unit="spot")
    assert spot.loc[spot.locus == "IGK", "cell_id"].iloc[0] == spot.cell_id.iloc[0] == "s1:s0"
    loc = dx.to_dandelion_table(define_clones(air.dropna(subset=["v_call"]), method="exact"), unit="locus")
    assert loc.cell_id.str.startswith("locus_").all()
    with pytest.raises(KeyError, match="not a column"):
        dx.to_dandelion_table(air, unit="cell_segmentation")


def test_compare_clones_known_answer():
    air = pd.DataFrame(
        {
            "locus": ["IGH"] * 6,
            "a": ["x", "x", "x", "y", "z", "z"],
            "b": ["1", "1", "2", "3", "4", "4"],
        }
    )
    t = dx.compare_clones(air, a="a", b="b").iloc[0]
    assert (t.n_records, t.n_clones_a, t.n_clones_b, t.n_split, t.n_merged) == (6, 3, 4, 0, 1)
    assert not t.identical and t.largest_a == 3 and t.largest_b == 2
    same = dx.compare_clones(air.assign(c=air.b.map({"1": "p", "2": "q", "3": "r", "4": "s"})), a="c", b="b").iloc[0]
    assert same.identical and same.adjusted_rand == pytest.approx(1.0)


def _dandelion_major() -> int:
    pytest.importorskip("dandelion")
    return int(dx.dandelion_version().split(".")[0])


def test_find_clones_versions_and_mapping():
    major = _dandelion_major()
    air = _toy()
    if major >= 1:
        out = dx.find_clones(air)
    else:
        with pytest.warns(dx.DandelionVersionWarning):
            out = dx.find_clones(air)
    assert list(out.index) == list(air.index)
    assert pd.isna(out.loc["rec6", "dandelion_clone_id"])
    heavy = out[out.locus == "IGH"].dropna(subset=["dandelion_clone_id"])["dandelion_clone_id"]
    info = out.attrs["spaclone"]["dandelion"]
    assert info["version"] == dx.dandelion_version() and info["PYTHONHASHSEED"] == "0"
    assert heavy.loc["rec0"] == heavy.loc["rec1"]  # identical junctions
    assert heavy.loc["rec4"] != heavy.loc["rec0"]  # D is far from everything
    if major >= 1:  # connected components: the chain A ~ B ~ C is one clone, as single linkage gives
        assert info["algorithm"] == "connected components"
        assert heavy.loc["rec0"] == heavy.loc["rec2"] == heavy.loc["rec3"]
    else:  # overlapping neighbourhoods: B ends up with C, A alone
        assert info["algorithm"] == "overlapping neighbourhoods"
        assert heavy.loc["rec2"] == heavy.loc["rec3"] != heavy.loc["rec0"]


def test_find_clones_equal_define_clones_and_deterministic():
    """Dandelion >= 1.0 on records = single linkage within V, J and length; a second session gives the same ids."""
    import warnings

    major = _dandelion_major()
    air = _families(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dx.DandelionVersionWarning)
        out = dx.find_clones(air)
        again = dx.find_clones(air)
    assert out["dandelion_clone_id"].equals(again["dandelion_clone_id"])  # PYTHONHASHSEED=0 child processes
    ref = define_clones(air, method="vj_length_identity", identity=0.85)
    t = dx.compare_clones(out.assign(sp=ref["clone_id"].to_numpy()), a="dandelion_clone_id", b="sp")
    if major >= 1:
        assert t.identical.all() and (t.adjusted_rand == 1.0).all()
        assert t.n_clones_a.iloc[0] < air.junction_aa.nunique()  # families were merged
    else:
        assert (t.n_clones_a >= t.n_clones_b).all()  # neighbourhoods never merge more than single linkage


def test_missing_dandelion_names_the_extra(monkeypatch):
    monkeypatch.setattr(dx, "dandelion_version", lambda python=None: None)
    with pytest.raises(ImportError, match=r"spaclone\[dandelion\]"):
        dx.find_clones(_toy())
