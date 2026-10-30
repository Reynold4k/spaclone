import numpy as np
import pandas as pd
import pytest

from spaclone import nulls
from spaclone._utils import as_rng, bh, max_t_pvalue, mc_pvalue


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    spot = np.repeat(np.arange(n // 3), 3)
    return pd.DataFrame(
        {
            "spot_id": spot.astype(str),
            "x_um": (spot % 10) * 10.0,
            "y_um": (spot // 10) * 10.0,
            "locus": rng.choice(["IGH", "IGK"], n),
            "clone_id": rng.choice(list("abcdef"), n),
            "consensus_count": rng.integers(1, 50, n).astype(float),
            "stratum": np.where(spot % 2 == 0, "even", "odd"),
        }
    )


def test_mc_pvalue_convention():
    null = np.arange(99, dtype=float)
    assert mc_pvalue(null, 1000.0, alternative="greater")[0] == pytest.approx(1 / 100)
    assert mc_pvalue(null, -1.0, alternative="greater")[0] == pytest.approx(1.0)
    assert mc_pvalue(null, -1.0, alternative="less")[0] == pytest.approx(1 / 100)
    assert mc_pvalue(null, 49.0, alternative="two-sided")[0] <= 1.0
    with pytest.raises(ValueError):
        mc_pvalue(null, 1.0, alternative="directed")


def test_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    q = bh(p)
    assert np.isnan(q[-1])
    # R: p.adjust(c(.01,.04,.03,.2), "BH") = 0.04 0.0533 0.0533 0.2
    np.testing.assert_allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_max_t_is_family_wise():
    rng = np.random.default_rng(1)
    null = rng.normal(size=(2000, 20))
    obs = rng.normal(size=20)
    p = max_t_pvalue(null, obs)
    # under the null the smallest adjusted p is rarely small
    assert p.min() > 0.01


def test_permute_preserves_strata_multisets():
    df = _toy()
    out = nulls.Permute("clone_id", within="stratum")(df, as_rng(3))
    for s in ("even", "odd"):
        a = sorted(df.loc[df.stratum == s, "clone_id"])
        b = sorted(out.loc[out.stratum == s, "clone_id"])
        assert a == b
    assert (out[["spot_id", "x_um", "y_um"]] == df[["spot_id", "x_um", "y_um"]]).all().all()
    assert not (out.clone_id == df.clone_id).all()


def test_permute_blocks_moves_whole_spots():
    df = _toy()
    out = nulls.Permute(["x_um", "y_um"], blocks="spot_id", within="stratum")(df, as_rng(5))
    per_spot = out.groupby("spot_id")[["x_um", "y_um"]].nunique()
    assert (per_spot == 1).all().all()
    # the multiset of spot positions is unchanged
    a = df.drop_duplicates("spot_id")[["x_um", "y_um"]].apply(tuple, axis=1).sort_values().tolist()
    b = out.drop_duplicates("spot_id")[["x_um", "y_um"]].apply(tuple, axis=1).sort_values().tolist()
    assert a == b


def test_permute_blocks_rejects_varying_columns():
    df = _toy()
    with pytest.raises(ValueError, match="varies inside a block"):
        nulls.Permute("clone_id", blocks="spot_id")(df, as_rng(0))


def test_relocate_moves_units_with_their_attributes():
    df = _toy()
    slots = pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(40)],
            "x_um": np.arange(40) * 5.0,
            "y_um": np.zeros(40),
            "stratum": np.where(np.arange(40) % 2 == 0, "even", "odd"),
            "w": np.ones(40),
        }
    )
    out = nulls.Relocate(slots, within="stratum", weights="w")(df, as_rng(2))
    assert out.spot_id.str.startswith("s").all()
    # rows of one original spot stay together, with their reads and clones
    m = df.assign(new=out.spot_id.to_numpy())
    assert (m.groupby("spot_id").new.nunique() == 1).all()
    assert (out.consensus_count.to_numpy() == df.consensus_count.to_numpy()).all()
    # strata are respected
    st = slots.set_index("spot_id").stratum
    assert (out.spot_id.map(st).to_numpy() == df.stratum.to_numpy()).all()


def test_relocate_without_replacement_needs_enough_slots():
    df = _toy()
    slots = pd.DataFrame({"spot_id": ["a", "b"], "x_um": [0.0, 1.0], "y_um": [0.0, 0.0]})
    with pytest.raises(ValueError, match="replace=True"):
        nulls.Relocate(slots)(df, as_rng(0))
    out = nulls.Relocate(slots, replace=True)(df, as_rng(0))
    assert set(out.spot_id) <= {"a", "b"}


def test_relocate_window_keeps_units_local():
    df = _toy(n=300)
    slots = df.drop_duplicates("spot_id")[["spot_id", "x_um", "y_um"]]
    out = nulls.Relocate(slots, window_um=50.0)(df, as_rng(4))
    assert (np.floor(out.x_um / 50) == np.floor(df.x_um / 50)).all()
    assert (np.floor(out.y_um / 50) == np.floor(df.y_um / 50)).all()


def test_placebo_detects_planted_signal_and_is_reproducible():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 1000, n)
    df = pd.DataFrame(
        {
            "spot_id": np.arange(n).astype(str),
            "x_um": x,
            "y_um": rng.uniform(0, 1000, n),
            "locus": "IGH",
            "clone_id": np.where(x < 200, "A", "B"),
        }
    )

    def mean_x_of_a(d):
        return d.loc[d.clone_id == "A", "x_um"].mean()

    design = nulls.Permute("clone_id")
    r1 = nulls.placebo(mean_x_of_a, df, design, n_perm=99, alternative="less", rng=7)
    r2 = nulls.placebo(mean_x_of_a, df, design, n_perm=99, alternative="less", rng=7)
    assert r1.p_value == pytest.approx(0.01)
    np.testing.assert_array_equal(r1.null, r2.null)
    assert r1.table.loc[0, "ratio"] < 0.5


def test_placebo_vector_statistic_and_compare_designs():
    df = _toy(n=120)

    def stat(d):
        return {"IGH": float((d.locus == "IGH").mean()), "n_clones": float(d.clone_id.nunique())}

    r = nulls.placebo(stat, df, nulls.Permute("clone_id"), n_perm=30, rng=1, alternative="two-sided")
    assert list(r.table.component) == ["IGH", "n_clones"]
    assert (r.table.p_value > 0).all()
    c = nulls.compare_designs(
        stat,
        df,
        {"global": nulls.Permute("clone_id"), "strata": nulls.Permute("clone_id", within="stratum")},
        n_perm=10,
        rng=2,
        alternative="two-sided",
    )
    assert set(c.table.design) == {"global", "strata"}


def test_placebo_calibration_under_null():
    """Under exchangeability the placebo p-value is uniform: P(p <= 0.05) <= 0.05 + tolerance."""
    rng = np.random.default_rng(11)
    hits = 0
    reps = 200
    for _ in range(reps):
        n = 80
        df = pd.DataFrame(
            {
                "spot_id": np.arange(n).astype(str),
                "x_um": rng.uniform(0, 100, n),
                "y_um": rng.uniform(0, 100, n),
                "locus": "IGK",
                "clone_id": rng.choice(["A", "B"], n),
            }
        )

        def stat(d):
            return d.loc[d.clone_id == "A", "x_um"].mean()

        r = nulls.placebo(stat, df, nulls.Permute("clone_id"), n_perm=39, rng=rng, alternative="greater")
        hits += r.p_value <= 0.05
    # binomial 99.9% upper bound for 200 trials at 5%
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
