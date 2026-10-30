import numpy as np
import pandas as pd
import pytest

from spaclone.pp import collapse_spot_clone, collapse_umis


def _reads(umis, counts, spot="s", locus="IGH", clone="x"):
    rows = []
    for u, n in zip(umis, counts, strict=True):
        rows += [{"spot_id": spot, "locus": locus, "umi": u, "clone_id": clone, "junction_aa": "CARW"}] * n
    return pd.DataFrame(rows)


def test_directional_hamming_and_unique_on_a_known_graph():
    # A (100 reads) absorbs B (10, 1 mismatch) which absorbs C (3, 1 mismatch from B, 2 from A); E (60) is 1
    # mismatch from A but too abundant to be its error (100 < 2 * 60 - 1); D (50) is far from all.
    r = _reads(["AAAAAA", "AAAAAT", "AAAATT", "TTTTTT", "AAAAAG"], [100, 10, 3, 50, 60])
    u = collapse_umis(r, method="unique")
    assert len(u) == 5
    h = collapse_umis(r, method="hamming")  # connected components: A-B-C-E and D
    assert sorted(h.consensus_count) == [50, 173]
    d = collapse_umis(r, method="directional")
    assert sorted(d.consensus_count) == [50, 60, 113]
    top = d.set_index("umi")
    assert top.loc["AAAAAA", "n_umis"] == 3 and top.loc["AAAAAA", "n_rows"] == 113
    assert top.loc["AAAAAG", "n_umis"] == 1


def test_singletons_one_mismatch_apart_merge_into_the_smaller_umi_and_strata_are_respected():
    r = _reads(["ACGT", "ACGA"], [1, 1])
    d = collapse_umis(r)
    assert len(d) == 1 and d.umi.iloc[0] == "ACGA" and d.consensus_count.iloc[0] == 2
    # the same UMI on two spots or two chains is two molecules
    two = pd.concat(
        [
            _reads(["ACGT"], [3], spot="s1"),
            _reads(["ACGT"], [3], spot="s2"),
            _reads(["ACGT"], [3], spot="s1", locus="IGK"),
        ],
        ignore_index=True,
    )
    assert len(collapse_umis(two)) == 3
    # UMIs of different length never join
    assert len(collapse_umis(_reads(["ACGT", "ACGTA"], [5, 1]))) == 2


def test_majority_vote_by_reads_and_missing_values_abstain():
    r = pd.DataFrame(
        {
            "spot_id": "s",
            "locus": "IGH",
            "umi": ["A"] * 5 + ["B"] * 3,
            "clone_id": ["x", "x", "y", None, None, None, None, "z"],
            "consensus_count": [1, 1, 3, 9, 9, 1, 1, 1.0],
        }
    )
    m = collapse_umis(r, method="unique").set_index("umi")
    assert m.loc["A", "clone_id"] == "y"  # 3 reads beat 2 reads; the 18 reads without a clone do not vote
    assert m.loc["A", "consensus_count"] == 23 and m.loc["A", "vote_share"] == pytest.approx(3 / 23)
    assert m.loc["B", "clone_id"] == "z" and m.loc["B", "n_rows"] == 3


def test_collapse_umis_recovers_planted_molecules():
    """Power: 300 molecules with random 10-nt UMIs, reads carry a substitution error with probability 0.05."""
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACGT"))
    true = ["".join(rng.choice(alphabet, 10)) for _ in range(300)]
    rows = []
    for t in true:
        for _ in range(rng.poisson(15) + 1):
            u = list(t)
            if rng.random() < 0.05:
                p = rng.integers(10)
                u[p] = rng.choice([a for a in "ACGT" if a != u[p]])
            rows.append("".join(u))
    r = pd.DataFrame({"spot_id": "s", "locus": "IGK", "umi": rows, "clone_id": "c"})
    n_unique = len(collapse_umis(r, method="unique"))
    n_dir = len(collapse_umis(r, method="directional"))
    assert n_unique > 330  # errors inflate exact counting
    assert abs(n_dir - 300) <= 3
    # determinism / row-order invariance
    a = collapse_umis(r).sort_values("umi").reset_index(drop=True)
    b = collapse_umis(r.sample(frac=1, random_state=2)).sort_values("umi").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["umi", "consensus_count", "n_umis"]], b[["umi", "consensus_count", "n_umis"]])


def test_collapse_umis_without_errors_keeps_distinct_molecules():
    """Null: distinct random UMIs without errors are almost never joined (expected 1-mismatch pairs < 1)."""
    rng = np.random.default_rng(11)
    umis = pd.unique(np.array(["".join(rng.choice(list("ACGT"), 10)) for _ in range(200)]))
    r = pd.DataFrame({"spot_id": "s", "locus": "IGH", "umi": np.repeat(umis, 5)})
    assert len(collapse_umis(r)) >= len(umis) - 2


def test_collapse_umis_errors_and_empty():
    r = _reads(["A"], [1])
    with pytest.raises(ValueError):
        collapse_umis(r, method="adjacency")
    with pytest.raises(KeyError):
        collapse_umis(r, umi_key="mid")
    with pytest.raises(TypeError):
        collapse_umis(r.assign(umi=1), method="directional")
    assert len(collapse_umis(r.iloc[:0])) == 0


def test_collapse_spot_clone_units():
    r = pd.DataFrame(
        {
            "spot_id": ["a", "a", "a", "b", "a"],
            "clone_id": ["x", "x", "y", "x", None],
            "consensus_count": [2.0, 3.0, 1.0, np.nan, 4.0],
            "v": [1, 2, 3, 4, 5],
        }
    )
    u = collapse_spot_clone(r)
    assert u[["spot_id", "clone_id"]].astype(object).where(
        u[["spot_id", "clone_id"]].notna(), None
    ).values.tolist() == [["a", "x"], ["a", "y"], ["b", "x"], ["a", None]]
    assert u.consensus_count.tolist()[:2] == [5.0, 1.0] and np.isnan(u.consensus_count.iloc[2])
    assert u.n_rows.tolist() == [2, 1, 1, 1] and u.v.tolist() == [1, 3, 4, 5]  # first row's values
    lib = pd.concat([r.assign(library_id="L1"), r.assign(library_id="L2")], ignore_index=True)
    assert len(collapse_spot_clone(lib)) == 8
    assert len(collapse_spot_clone(lib, by=["spot_id", "clone_id"])) == 4
    with pytest.raises(KeyError):
        collapse_spot_clone(r, by=["spot", "clone_id"])
