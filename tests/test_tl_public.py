import numpy as np
import pandas as pd
import pytest

from spaclone import tl
from spaclone._utils import SpacloneWarning


def _air():
    return pd.DataFrame(
        {
            "clone_id": ["a", "a", "b", "b", "c", "d", "a", "e"],
            "mouse": pd.Categorical(["m2", "m1", "m2", "m2", "m1", "m3", "m3", "m2"], categories=["m2", "m1", "m3"]),
            "spot_id": list("pqrstuvw"),
            "locus": "IGH",
            "junction_aa": ["CARW", "CARW", "CAKW", "CAKW", "CTTW", "CAAW", "CARW", "CQQW"],
            "junction": ["TGT", "TGC", "AAA", "AAA", "CCC", "GGG", "TGT", "TTT"],
            "v_call": "IGHV1-1*01",
            "j_call": "IGHJ1*01",
        }
    )


def test_public_clonotypes_counts_and_order():
    pub = tl.public_clonotypes(_air(), sample_key="mouse", unit="record")
    a = pub.iloc[0]
    assert a["clone_id"] == "a" and a["n_samples"] == 3 and a["public"]
    assert a["samples"] == "m2;m1;m3"  # categorical order
    assert a["n_junction_nt"] == 2  # TGT / TGC encode the same junction: convergent recombination
    assert pub["public"].sum() == 1
    assert pub.set_index("clone_id").loc["b", "max_units"] == 2
    assert pub["v_gene"].iloc[0] == "IGHV1-1"


def test_public_clonotypes_warns_about_light_chains_and_single_linkage():
    air = _air()
    with pytest.warns(SpacloneWarning, match="light-chain"):
        tl.public_clonotypes(air.assign(locus="IGK"), sample_key="mouse", unit="record")
    linked = air.copy()
    linked.loc[6, "junction_aa"] = "CAKW"  # clone a now holds two junctions and is public
    with pytest.warns(SpacloneWarning, match="single-linkage"):
        tl.public_clonotypes(linked, sample_key="mouse", unit="record")
    with pytest.raises(ValueError, match="min_samples"):
        tl.public_clonotypes(air, sample_key="mouse", min_samples=1)


def _samples(rng, n_samples=4, n_clones=60, public_bias=0.0):
    rows = []
    for s in range(n_samples):
        sizes = rng.geometric(0.4, n_clones)
        for c, k in enumerate(sizes):
            rows += [{"clone_id": f"s{s}_c{c}", "mouse": f"m{s}", "spot_id": f"m{s}_{c}_{i}"} for i in range(k)]
    df = pd.DataFrame(rows)
    # public clones: relabel some clones as one shared id in two samples
    sizes = df.groupby("clone_id").size()
    cand = sizes.index.to_numpy()
    w = np.exp(public_bias * np.log(sizes.to_numpy(dtype=float)))
    shared = rng.choice(cand, 30, replace=False, p=w / w.sum())
    for i, c in enumerate(shared):
        other = f"m{(int(c[1]) + 1) % n_samples}"
        df.loc[df["clone_id"] == c, "clone_id"] = f"pub{i}"
        df = pd.concat([df, pd.DataFrame([{"clone_id": f"pub{i}", "mouse": other, "spot_id": f"x{i}"}])])
    return df.reset_index(drop=True)


def test_top_clones_null_is_hypergeometric_and_replays_the_loop():
    rng = np.random.default_rng(0)
    air = _samples(rng)
    r = tl.public_top_clones(air, sample_key="mouse", top_k=10, n_draws=4000, unit="record", rng=1)
    t = r.table.set_index("sample")
    for s in ("m0", "m1"):
        se = t.loc[s, "null_sd"] / np.sqrt(4000)
        assert abs(t.loc[s, "null_mean"] - t.loc[s, "expected"]) < 4 * se
    # exact replay of the archive's loop (A4): per sample, n_draws calls of rng.choice(n_clones, k, replace=False)
    sizes = tl.clone_sizes(air, by="mouse", unit="record")
    nsamp = sizes.groupby("clone_id")["mouse"].nunique()
    gen = np.random.default_rng(1)
    first = sizes[sizes["mouse"] == "m0"]
    sh = (first["clone_id"].map(nsamp) >= 2).to_numpy().astype(int)
    nl = np.array([sh[gen.choice(len(sh), 10, replace=False)].sum() for _ in range(4000)])
    assert t.loc["m0", "null_mean"] == nl.mean()
    assert t.loc["m0", "statistic"] == sh[:10].sum()


def test_top_clones_calibration_and_power():
    rng = np.random.default_rng(3)
    hits, reps = 0, 20
    for _ in range(reps):
        r = tl.public_top_clones(_samples(rng), sample_key="mouse", top_k=10, n_draws=199, unit="record", rng=rng)
        hits += r.table.set_index("sample").loc["all", "p_value"] <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    r = tl.public_top_clones(
        _samples(rng, public_bias=3.0), sample_key="mouse", top_k=10, n_draws=999, unit="record", rng=1
    )
    assert r.table.set_index("sample").loc["all", "p_value"] <= 0.01


def test_pgen_matches_the_archived_olga_values():
    pytest.importorskip("olga")
    table = pd.DataFrame(
        {
            "locus": ["TRA", "TRA", "IGQ"],
            "v_call": ["TRAV10D*01", "TRAV10D", "X"],
            "j_call": ["TRAJ12*01", "TRAJ12", "Y"],
            "junction_aa": ["CAADTGGYKVVF", "CAASKAGGYKVVF", "CARW"],
        }
    )
    p = tl.pgen(table, species="mouse")
    # results/QC/StereoIR_TCR/TR2_pgen.tsv
    assert p["pgen_vj"].iloc[0] == pytest.approx(2.373522e-08, rel=1e-6)
    assert p["pgen_aa"].iloc[0] == pytest.approx(1.405835e-06, rel=1e-6)
    assert p["pgen_vj"].iloc[1] == pytest.approx(1.884868e-07, rel=1e-6)
    assert np.isnan(p["pgen_vj"].iloc[2])
