import numpy as np
import pandas as pd
import pytest

from spaclone.qc import umi_collisions, umi_decoys, umi_match


def _rand(rng, n, length):
    return _umis(rng, n, length).tolist()


def test_known_answer_categories_and_tie_breaking():
    air = pd.DataFrame(
        {
            "spot_id": ["B1", "B1", "B1", "B2", "B2", "B2", "B3", "B3"],
            "umi": ["AAAAAAA", "CCCCCCC", "GGGGGGG", "AAAAAAA", "TTTTTTT", "ACACACA", "AAAAAAA", "AAAAAAA"],
            "locus": ["IGK", "IGK", "IGK", "IGK", "IGK", "IGH", "IGH", "IGH"],
            "c_call": ["IGKC", "IGKC", "IGKC", "IGKC", None, "IGHG1", "IGHG2C", "IGHG2C"],
        }
    )
    reads = pd.DataFrame(
        {
            "spot_id": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2", "B2", "B3", "B3"],
            "umi": ["AAAAAAA", "CCCCCCC", "CCCCCCC", "GGGGGGG", "AAAAAAA", "AAAAAAA", "TTTTTTT", "ACACACA", "ACACACA",
                    "AAAAAAA", "AAAAAAA"],
            "gene": ["Igkc", "Actb", "Igkc", "multi", "Igkv1-117", "Igkv1-117", "Igkc", "Ighm", "unmapped", "Ighg2c",
                     "multi"],
        }
    )  # fmt: skip
    res = umi_match(air, reads, decoys="random", n_decoys=1, rng=0)
    k = res.keys[res.keys.kind == "air"].set_index(["barcode", "umi"])
    assert k.loc[("B1", "AAAAAAA"), "category"] == "own_constant"
    assert k.loc[("B1", "CCCCCCC"), "top_gene"] == "Actb"  # tie 1:1, the gene read first wins (X4 / Counter)
    assert k.loc[("B1", "CCCCCCC"), "category"] == "other_gene"
    assert k.loc[("B1", "GGGGGGG"), "category"] == "undetermined"
    assert k.loc[("B2", "AAAAAAA"), "category"] == "v_or_j"
    assert k.loc[("B2", "TTTTTTT"), "category"] == "own_constant"  # no c_call: falls back to the locus genes
    assert k.loc[("B2", "ACACACA"), "category"] == "other_constant"  # IGHG1 molecule meets Ighm
    assert len(k) == 7  # duplicate (barcode, UMI, locus) collapsed
    assert k.loc[("B3", "AAAAAAA"), "category"] == "own_constant"
    strict = umi_match(air, reads, decoys="random", n_decoys=1, rng=0, own_genes="isotype")
    ks = strict.keys[strict.keys.kind == "air"].set_index(["barcode", "umi"])
    assert ks.loc[("B2", "TTTTTTT"), "category"] == "other_constant"  # strict: a key without isotype is never own
    cat = res.categories.set_index(["kind", "locus"])
    assert cat.loc[("air", "IGK"), "keys"] == 5 and cat.loc[("air", "IGK"), "matched"] == 5
    assert cat.loc[("air", "IGK"), "of_matched_own_constant"] == pytest.approx(2 / 5)
    assert cat.loc[("air", "IGK"), "frac_matched_own"] == pytest.approx(2 / 5)
    row = res.table.set_index(["locus", "outcome"]).loc[("IGK", "own_constant")]
    assert row.statistic == 2 and row.n_keys == 5


def test_decoys_rules_and_determinism():
    rng = np.random.default_rng(1)
    bcs = _rand(rng, 50, 14)
    air = pd.DataFrame(
        {
            "spot_id": np.repeat(bcs, 4),
            "umi": _rand(rng, 200, 7),
            "locus": rng.choice(["IGK", "IGH"], 200),
            "c_call": "IGKC",
        }
    )
    taken = set(zip(air.spot_id, air.umi, strict=True))
    for method in ("shuffle", "random"):
        d = umi_decoys(air, n_decoys=3, method=method, rng=5)
        assert len(d) == 600 and (d.kind == "decoy").all()
        assert not any((b, u) in taken for b, u in zip(d.barcode, d.umi, strict=True))
        assert not d.duplicated(["barcode", "umi"]).any()
        keys = air.drop_duplicates(["spot_id", "umi", "locus"]).reset_index(drop=True)
        assert (keys.spot_id.to_numpy()[d.parent] == d.barcode.to_numpy()).all()
        assert (keys.locus.to_numpy()[d.parent] == d.locus.to_numpy()).all()
        pd.testing.assert_frame_equal(d, umi_decoys(air, n_decoys=3, method=method, rng=5))
        assert not d.umi.equals(umi_decoys(air, n_decoys=3, method=method, rng=6).umi)
    sh = umi_decoys(air, n_decoys=2, method="shuffle", rng=0)
    assert set(sh.umi) <= set(air.umi)  # shuffled decoys reuse real UMIs
    with pytest.raises(ValueError):
        umi_decoys(air, method="poisson")


def test_collision_expectation_matches_simulation():
    rng = np.random.default_rng(2)
    m, length = 400, 6
    pairs = []
    for _ in range(400):
        u = rng.integers(0, 4**length, m)
        _, c = np.unique(u, return_counts=True)
        pairs.append((c * (c - 1) / 2).sum())
    exp = umi_collisions(m, length)
    assert np.mean(pairs) == pytest.approx(float(exp["pairs"]), rel=0.05)
    hit = np.mean([np.isin(rng.integers(0, 4**length), rng.integers(0, 4**length, m)) for _ in range(4000)])
    assert hit == pytest.approx(float(exp["p_key"]), abs=0.015)


def _umis(rng, n, length):
    codes = rng.integers(0, 4, (n, length), dtype=np.uint8)
    raw = np.frombuffer(np.array(list(b"ACGT"), dtype=np.uint8)[codes].tobytes(), dtype=f"S{length}")
    return raw.astype(str)


def _beads(seed, n_beads=300, depth=(200, 600), n_air=3, genuine=0.0, p_det=0.6, length=7):
    """Synthetic barcoded beads: GEX molecules with random UMIs; receptor molecules genuine (own transcript in GEX
    with p_det) or with a new random UMI."""
    rng = np.random.default_rng(seed)
    bcs = _umis(rng, n_beads, 14)
    genes = np.array(["Igkc", "Ighg1", "Actb", "Cd74", "Igkv3-2"])
    m = rng.integers(*depth, n_beads)
    gb, gu = np.repeat(bcs, m), _umis(rng, int(m.sum()), length)
    gg = rng.choice(genes, int(m.sum()), p=[0.04, 0.02, 0.5, 0.4, 0.04])
    na = n_beads * n_air
    ab, au = np.repeat(bcs, n_air), _umis(rng, na, length)
    igk = rng.random(na) < 0.7
    seen = (rng.random(na) < genuine) & (rng.random(na) < p_det)
    gex = pd.DataFrame(
        {
            "spot_id": np.r_[gb, ab[seen]],
            "umi": np.r_[gu, au[seen]],
            "gene": np.r_[gg, np.where(igk[seen], "Igkc", "Ighg1")],
        }
    )
    reads = gex.loc[gex.index.repeat(rng.integers(1, 4, len(gex)))].reset_index(drop=True)
    reads.loc[rng.random(len(reads)) < 0.3, "gene"] = "multi"
    air = pd.DataFrame(
        {"spot_id": ab, "umi": au, "locus": np.where(igk, "IGK", "IGH"), "c_call": np.where(igk, "IGKC", "IGHG1")}
    )
    return air, reads, gex.groupby("spot_id").size()


def _null_pvalues(n_sims, n_beads, depth=(100, 300)):
    p = {"matched": [], "own_constant": []}
    for seed in range(n_sims):
        air, reads, _ = _beads(seed, n_beads=n_beads, depth=depth)
        t = umi_match(air, reads, decoys="shuffle", rng=seed).table
        for outcome in p:
            p[outcome].append(t[(t.locus == "IGK") & (t.outcome == outcome)].p_value.iloc[0])
    return {k: np.asarray(v) for k, v in p.items()}


def test_calibration_under_the_null():
    for outcome, v in _null_pvalues(40, 60).items():
        assert (v <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(v)), outcome
        assert (v <= 0.2).mean() <= 0.2 + 3 * np.sqrt(0.2 * 0.8 / len(v)), outcome


@pytest.mark.slow
def test_calibration_under_the_null_many_sims():
    for outcome, v in _null_pvalues(400, 150, depth=(200, 600)).items():
        for a in (0.01, 0.05, 0.1):
            assert (v <= a).mean() <= a + 3 * np.sqrt(a * (1 - a) / len(v)), outcome


def test_power_and_collision_chance_on_planted_genuine_molecules():
    air, reads, depth = _beads(3, genuine=0.5, p_det=0.6)
    res = umi_match(air, reads, rng=1, barcode_umis=depth)
    t = res.table.set_index(["locus", "outcome"])
    own = t.loc[("IGK", "own_constant")]
    # 0.5 genuine x 0.6 detected, minus keys whose 1-3 reads were all relabelled "multi" (~14%): ~0.26
    assert own.p_value < 1e-30 and own.excess_rate == pytest.approx(0.26, abs=0.06)
    cat = res.categories.set_index(["kind", "locus"])
    dec = cat.loc[("decoy", "IGK")]
    assert dec.frac_matched == pytest.approx(dec.expected_chance_match, abs=0.02)  # decoys ~ UMI-collision chance
    # supplied decoys without parent: strata are barcodes; the result stays significant
    d = res.keys[res.keys.kind == "decoy"][["barcode", "umi", "locus", "c_call"]]
    sup = umi_match(air, reads, decoys=d).table.set_index(["locus", "outcome"])
    assert sup.loc[("IGK", "own_constant"), "p_value"] < 1e-30
    assert sup.loc[("IGK", "own_constant"), "statistic"] == own.statistic
