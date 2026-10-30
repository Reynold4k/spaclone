import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import connected_components

from spaclone.pp import define_clones, gene_name


def _air():
    return pd.DataFrame(
        {
            "locus": ["IGH"] * 6 + ["TRB"],
            "v_call": ["IGHV1-15*01", "IGHV1-15*02", "IGHV1-15*01", "IGHV1-15*01", "IGHV2-2*01", None, "TRBV1*01"],
            "j_call": ["IGHJ1*01", "IGHJ1*01", "IGHJ1*01", "IGHJ1*01", "IGHJ1*01", "IGHJ1*01", "TRBJ2-7*01"],
            "junction_aa": [
                "CARDYYAMDYW",  # 11 aa
                "CARDYYAMDYW",  # same, other allele
                "CARDYYGMDYW",  # 1 mismatch
                "CAREYYGMDFW",  # 2 more mismatches from the previous one, 3 from the first
                "CARDYYAMDYW",  # other V gene
                "CARDYYAMDYW",  # no V call
                "CASSQDRGYEQYF",
            ],
        }
    )


def test_gene_name():
    g = gene_name(pd.Series(["IGHV1-15*01", "TRBV12-1*01,TRBV12-2*01", "IGKJ1", None, ""]))
    assert g.tolist()[:3] == ["IGHV1-15", "TRBV12-1", "IGKJ1"]
    assert g.isna().tolist()[3:] == [True, True]


def test_exact_clones():
    out = define_clones(_air())
    c = out.clone_id
    assert c.iloc[0] == "IGH|IGHV1-15|IGHJ1|CARDYYAMDYW"
    assert c.iloc[0] == c.iloc[1]  # alleles dropped
    assert c.iloc[0] != c.iloc[2] != c.iloc[4]
    assert pd.isna(c.iloc[5])
    plain = define_clones(_air(), by=())
    assert plain.clone_id.iloc[0] == "IGHV1-15|IGHJ1|CARDYYAMDYW"
    assert plain.clone_id.iloc[6] == "TRBV1|TRBJ2-7|CASSQDRGYEQYF"
    assert "clone_id" not in _air().columns  # input untouched


def test_vj_length_identity_single_linkage():
    air = _air()
    out = define_clones(air, method="vj_length_identity", identity=0.85)
    c = out.clone_id
    # 11 aa: floor(11 * 0.15) = 1 mismatch allowed: rows 0-2 join, row 3 (2 from row 2) stays apart
    assert c.iloc[0] == c.iloc[1] == c.iloc[2]
    assert c.iloc[3] != c.iloc[2]
    assert c.iloc[0].startswith("IGH|IGHV1-15|IGHJ1|L11|")
    out = define_clones(air, method="vj_length_identity", identity=0.8)  # 2 mismatches allowed: chain 0-2-3
    assert out.clone_id.iloc[0] == out.clone_id.iloc[3]
    assert out.clone_id.iloc[4] != out.clone_id.iloc[0]


def test_identity_matches_brute_force_and_is_order_invariant():
    rng = np.random.default_rng(3)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    base = ["".join(rng.choice(alphabet, 12)) for _ in range(6)]
    seqs = []
    for _ in range(300):
        s = list(base[rng.integers(6)])
        for p in rng.choice(12, rng.integers(0, 3), replace=False):
            s[p] = rng.choice(alphabet)
        seqs.append("C" + "".join(s) + "W")
    air = pd.DataFrame({"locus": "IGH", "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01", "junction_aa": seqs})
    out = define_clones(air, method="vj_length_identity")
    u = np.array(sorted(set(seqs)))
    a = np.array([list(s) for s in u])
    d = (a[:, None, :] != a[None, :, :]).sum(-1) / a.shape[1]
    _, lab = connected_components(d <= 0.15 + 1e-9, directed=False)
    truth = dict(zip(u, lab, strict=True))
    got = out.groupby("junction_aa").clone_id.first()
    # same partition of the unique junctions
    pairs = pd.DataFrame({"t": [truth[s] for s in got.index], "g": got.to_numpy()})
    assert pairs.groupby("t").g.nunique().max() == 1 and pairs.groupby("g").t.nunique().max() == 1
    perm = rng.permutation(len(air))
    out2 = define_clones(air.iloc[perm], method="vj_length_identity")
    pd.testing.assert_series_equal(out.clone_id, out2.clone_id.sort_index())


def test_errors():
    with pytest.raises(ValueError):
        define_clones(_air(), method="cdr3")
    with pytest.raises(ValueError):
        define_clones(_air(), identity=0)
    with pytest.raises(KeyError, match="by=()"):
        define_clones(_air().drop(columns="locus"))
