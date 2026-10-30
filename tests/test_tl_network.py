from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from spaclone import tl
from spaclone.tl.network import _levenshtein_batch, _seq_matrix

ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ham(a, b):
    return sum(x != y for x, y in zip(a, b, strict=True)) if len(a) == len(b) else 10**6


def _random_air(n=300, seed=1):
    rng = np.random.default_rng(seed)
    base = ["".join(rng.choice(ALPHABET, rng.integers(6, 10))) for _ in range(25)]
    seqs = []
    for _ in range(n):
        s = list(base[rng.integers(25)])
        for _ in range(rng.integers(0, 3)):
            op, p = rng.integers(3), rng.integers(len(s))
            if op == 0:
                s[p] = rng.choice(ALPHABET)
            elif op == 1:
                s.insert(p, rng.choice(ALPHABET))
            elif len(s) > 4:
                del s[p]
        seqs.append("C" + "".join(s) + "W")
    spot = rng.integers(0, 200, n)
    return pd.DataFrame(
        {
            "junction_aa": seqs,
            "locus": rng.choice(["IGH", "IGK"], n),
            "v_call": rng.choice(["V1*01", "V2*01"], n),
            "j_call": "J1*01",
            "spot_id": spot.astype(str),
            "x_um": (spot % 20) * 10.0,
            "y_um": (spot // 20) * 10.0,
            "clone_id": "x",
            "region": np.where(spot % 20 < 10, "A", "B"),
        }
    )


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
@pytest.mark.parametrize("cutoff", [1, 2])
@pytest.mark.parametrize("same_v", [True, False])
def test_numpy_edges_match_brute_force(metric, cutoff, same_v):
    air = _random_air()
    net = tl.clonotype_network(air, metric=metric, cutoff=cutoff, same_v=same_v, backend="numpy", unit="spot")
    fn = _ham if metric == "hamming" else _lev
    key = ["locus"] + (["v_gene"] if same_v else []) + ["j_gene"]
    truth = set()
    for _, g in net.nodes.groupby(key):
        for (a, sa), (b, sb) in combinations(zip(g["node_id"], g["junction"], strict=True), 2):
            d = fn(sa, sb)
            if d <= cutoff:
                truth.add((min(a, b), max(a, b), d))
    assert set(zip(net.edges["node_a"], net.edges["node_b"], net.edges["distance"], strict=True)) == truth


def test_levenshtein_batch_matches_dynamic_programming():
    rng = np.random.default_rng(0)
    for la, lb in ((5, 5), (5, 7), (9, 8), (1, 3)):
        a = ["".join(rng.choice(ALPHABET[:4], la)) for _ in range(40)]
        b = ["".join(rng.choice(ALPHABET[:4], lb)) for _ in range(40)]
        got = _levenshtein_batch(_seq_matrix(np.array(a, dtype=object)), _seq_matrix(np.array(b, dtype=object)))
        assert got.tolist() == [_lev(x, y) for x, y in zip(a, b, strict=True)]


def test_components_membership_and_spatial_summary():
    air = pd.DataFrame(
        {
            "junction_aa": ["CASSLW", "CASSFW", "CATSFW", "CQQYW", "CASSLW"],
            "locus": "TRB",
            "v_call": ["V1", "V1", "V1", "V1", "V2"],
            "j_call": "J1",
            "spot_id": ["a", "b", "c", "d", "e"],
            "x_um": [0.0, 100.0, 200.0, 0.0, 50.0],
            "y_um": [0.0, 0.0, 0.0, 300.0, 0.0],
            "region": ["R1", "R1", "R2", "R1", "R2"],
        }
    )
    net = tl.clonotype_network(air, backend="numpy", unit="spot", region_key="region")
    comp = net.membership.tolist()
    assert comp[0] == comp[1] == comp[2]  # CASSLW - CASSFW - CATSFW chain
    assert comp[3] != comp[0] and comp[4] != comp[0]  # other junction; same junction but another V gene
    c0 = net.components.set_index("component").loc[comp[0]]
    assert c0["n_nodes"] == 3 and c0["n_units"] == 3
    assert c0["mst_um"] == pytest.approx(200.0)
    assert c0["rg_um"] == pytest.approx(np.sqrt((100**2 + 0 + 100**2) / 3))
    assert c0["regions"] == "R1;R2" and c0["main_region_share"] == pytest.approx(2 / 3)
    free = tl.clonotype_network(air, backend="numpy", same_v=False, unit="spot")
    assert free.membership.iloc[4] == free.membership.iloc[0]
    assert net.nodes["degree"].sum() == 2 * len(net.edges)
    missing = air.assign(junction_aa=[None, "CASSFW", "CATSFW", "CQQYW", "CASSLW"])
    assert pd.isna(tl.clonotype_network(missing, backend="numpy", unit="spot").membership.iloc[0])


def test_errors():
    air = _random_air(20)
    with pytest.raises(ValueError, match="metric"):
        tl.clonotype_network(air, metric="tcrdist", backend="numpy")
    with pytest.raises(ValueError, match="backend"):
        tl.clonotype_network(air, backend="gpu")
    with pytest.raises(KeyError):
        tl.clonotype_network(air.drop(columns="v_call"), backend="numpy")


@pytest.mark.slow
@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
def test_scirpy_backend_gives_the_same_edges(metric):
    pytest.importorskip("scirpy")
    air = _random_air(250, seed=4)
    a = tl.clonotype_network(air, metric=metric, cutoff=2, backend="numpy", unit="spot")
    b = tl.clonotype_network(air, metric=metric, cutoff=2, backend="scirpy", unit="spot")
    pd.testing.assert_frame_equal(a.edges, b.edges)
    pd.testing.assert_frame_equal(a.components, b.components)
