import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, pl


def test_isotype_parsing_mouse():
    calls = pd.Series(["IGHG2B*01", "IGHM", "*", "IGKC", None, "IGHA*09", "IGHG2C", "IGHGP", "IgG1", "IGHG", "IGHE*01"])
    out = bcr.isotype(calls)
    assert out.isotype.tolist()[:3] == ["IgG", "IgM", "unknown"]
    assert pd.isna(out.isotype.iloc[3])  # light chain: not an isotype
    assert out.isotype.iloc[4] == "unknown"
    assert out.isotype_subclass.tolist()[5:7] == ["IgA", "IgG2c"]
    assert out.isotype.iloc[7] == "unknown"  # pseudogene
    assert out.isotype_subclass.iloc[8] == "IgG1"
    assert out.isotype.iloc[9] == "IgG" and pd.isna(out.isotype_subclass.iloc[9])  # class-only call
    assert (
        out.isotype_position.iloc[0] == 4 and out.isotype_position.iloc[5] == 7 and out.isotype_position.iloc[10] == 6
    )
    assert out.switched.tolist()[:2] == [True, False]
    assert pd.isna(out.switched.iloc[2])


def test_isotype_human_and_species_none():
    calls = pd.Series(["IGHA1", "IGHA2", "IGHG4", "IGHG2B", "IGHA"])
    h = bcr.isotype(calls, species="human")
    assert h.isotype_subclass.tolist()[:3] == ["IgA1", "IgA2", "IgG4"]
    assert pd.isna(h.isotype_subclass.iloc[3]) and h.isotype.iloc[3] == "IgG"  # no IgG2b in human
    assert pd.isna(h.isotype_subclass.iloc[4])  # human IgA needs a subclass
    assert h.isotype_position.tolist()[:3] == [4, 8, 6]
    n = bcr.isotype(calls, species=None)
    assert n.isotype_subclass.iloc[3] == "IgG2b"
    assert n.isotype_position.isna().all()
    with pytest.raises(ValueError):
        bcr.isotype(calls, species="rat")


def test_isotype_on_table_uses_locus_and_agreeing_calls():
    air = pd.DataFrame(
        {
            "c_call": ["IGHG1*01,IGHG1*02", "IGHG1,IGHG2B", "IGHM,IGHA", "IGHM", "IGKC"],
            "locus": ["IGH", "IGH", "IGH", "IGK", "IGK"],
        }
    )
    out = bcr.isotype(air)
    assert out.isotype_subclass.iloc[0] == "IgG1"
    assert out.isotype.iloc[1] == "IgG" and pd.isna(out.isotype_subclass.iloc[1])
    assert out.isotype.iloc[2] == "unknown"  # calls name two classes
    assert out.isotype.iloc[3:].isna().all()  # other chains
    assert list(air.columns) == ["c_call", "locus"]  # input untouched
    with pytest.raises(KeyError):
        bcr.isotype(air, c_key="nope")


def test_locus_order_is_monotone_along_the_locus():
    o = bcr.isotype_order("mouse")
    assert o["IgM"] < o["IgD"] < o["IgG3"] < o["IgG1"] < o["IgG2b"] < o["IgG2c"] < o["IgE"] < o["IgA"]
    assert o["IgG2a"] == o["IgG2c"]
    c = bcr.isotype_order("human", level="class")
    assert c["IgM"] == 0 and c["IgG"] == 2 and c["IgA"] == 4 and c["IgE"] == 7


def _clone_class(contigs):
    """The archive's T_common.clone_class (reference implementation)."""
    ct = pd.crosstab(contigs.clone, contigs.iso).reindex(columns=["IGHM", "IGHD", "IGHG", "IGHA"], fill_value=0)
    ct = ct[ct.sum(axis=1) > 0]
    at_top = ct.eq(ct.max(axis=1), axis=0)
    s = at_top[["IGHG", "IGHA"]].any(axis=1)
    u = at_top[["IGHM", "IGHD"]].any(axis=1)
    return pd.Series(np.where(s & u, "tied", np.where(s, "switched", "unswitched")), index=ct.index)


def test_call_isotype_clone_rule_matches_archive_clone_class():
    rng = np.random.default_rng(0)
    n = 600
    iso = rng.choice(["IGHM", "IGHD", "IGHG", "IGHA", "unknown"], n, p=[0.4, 0.15, 0.15, 0.15, 0.15])
    d = pd.DataFrame({"clone": rng.integers(0, 150, n).astype(str), "iso": iso})
    ref = _clone_class(d)
    air = pd.DataFrame({"clone_id": d.clone, "isotype": d.iso.map(lambda x: "Ig" + x[3] if x != "unknown" else x)})
    calls = bcr.call_isotype(air, by="clone")
    sub = calls.loc[calls.n_isotyped > 0]
    assert sorted(sub.index) == sorted(ref.index)
    assert (sub.call.astype(str).reindex(ref.index) == ref).all()
    assert sub.switched[sub.call == "tied"].isna().all()
    forced = bcr.call_isotype(air, by="clone", tie="switched")
    assert forced.switched[forced.call == "tied"].all()


def test_call_isotype_locus_share_rule_and_main_isotype():
    air = pd.DataFrame(
        {
            "locus_id": [0, 0, 1, 1, 1, 2, 3],
            "isotype": ["IgM", "IgG", "IgM", "IgM", "IgA", "unknown", "IgD"],
        }
    )
    calls = bcr.call_isotype(air, by="locus")
    assert calls.switched.iloc[0]  # 1 of 2 isotyped = 50 % -> switched (T3 rule)
    assert calls.isotype.iloc[0] == "IgM"  # tie broken in the order IgM, IgD, IgG, IgA, IgE
    assert not calls.switched.iloc[1]
    assert calls.switched_share.iloc[1] == pytest.approx(1 / 3)
    assert pd.isna(calls.switched.iloc[2]) and calls.call.iloc[2] == "unknown" and calls.unknown.iloc[2] == 1
    assert calls.isotype.iloc[3] == "IgD"
    strict = bcr.call_isotype(air, by="locus", min_share=0.6)
    assert not strict.switched.iloc[0]
    with pytest.raises(KeyError, match="merge_loci"):
        bcr.call_isotype(air.rename(columns={"locus_id": "x"}), by="locus")
    with pytest.raises(ValueError):
        bcr.call_isotype(air, by="locus", rule="vote")


def test_call_isotype_derives_classes_from_c_call():
    air = pd.DataFrame({"clone_id": ["a", "a", "b"], "c_call": ["IGHG1*01", "IGHM", "IGHM"]})
    calls = bcr.call_isotype(air)
    assert calls.loc["a", "call"] == "tied" and pd.isna(calls.loc["a", "switched"])
    assert calls.loc["b", "call"] == "unswitched"


def test_isotype_map_returns_axes():
    rng = np.random.default_rng(1)
    units = pd.DataFrame(
        {"x_um": rng.uniform(0, 100, 50), "y_um": rng.uniform(0, 100, 50), "isotype": rng.choice(["IgM", "IgG"], 50)}
    )
    ax = pl.isotype_map(units, style="dark", background=units)
    assert ax.get_xlabel() == "x (µm)"
    assert len(ax.collections) == 3
    with pytest.raises(KeyError):
        pl.isotype_map(units, isotype_key="nope")
