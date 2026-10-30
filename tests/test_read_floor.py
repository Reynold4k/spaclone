import numpy as np
import pandas as pd
import pytest

from spaclone import qc
from spaclone._utils import mc_pvalue
from spaclone.pp import merge_loci
from spaclone.qc.read_floor import _Sharing


def _legacy_shared_partner(df, bin_a):
    """shared_partner of diag/stereoir_TR_chimera.py (archive: external-data validation, workflow), verbatim."""
    out = {}
    for c, g in df.groupby("clone"):
        if len(g) < 2:
            continue
        cnt = {}
        for b in g.cell_id:
            for t in bin_a.get(b, ()):
                cnt[t] = cnt.get(t, 0) + 1
        n_a = sum(1 for b in g.cell_id if b in bin_a)
        out[c] = (len(g), n_a, max(cnt.values()) if cnt else 0)
    return pd.DataFrame(out, index=["bins", "bins_with_TRA", "max_shared"]).T


def _legacy_floor(b, bin_a, floor, side, n_shuf, rng):
    sub = b[b.consensus_count >= floor] if side == "above" else b[b.consensus_count < floor]
    s = _legacy_shared_partner(sub, bin_a)
    s = s[s.bins_with_TRA >= 2]
    obs = (s.max_shared >= 2).mean()
    ch = []
    for _ in range(n_shuf):
        sb = sub.copy()
        sb["clone"] = sb.clone.to_numpy()[rng.permutation(len(sb))]
        sbs = _legacy_shared_partner(sb, bin_a)
        sbs = sbs[sbs.bins_with_TRA >= 2]
        ch.append((sbs.max_shared >= 2).mean())
    return len(s), obs, float(np.mean(ch))


def _tcr(seed=0, n_bg=1500, n_clones=40, side=60, chimeras=3, alpha_det=0.6, genuine_reads=200.0, n_bg_alpha=None):
    """StereoIR-like TCR contigs on a grid of 10-um bins.

    Expanded clones: cells (one bin each, high reads) carry the clone's own alpha chain with probability
    ``alpha_det``; chimeric copies of the clone's beta CDR3 (2-9 reads) sit on random bins of other cells, which carry
    those cells' alpha chains. Background bins hold one random beta and (with probability ``alpha_det``) one alpha,
    unique or, with ``n_bg_alpha``, drawn from that many public alpha chains (so that chance sharing happens).
    """
    rng = np.random.default_rng(seed)
    rows = []
    cells = rng.choice(side * side, n_bg, replace=False)
    host_alpha = {}
    for i, b in enumerate(cells):  # background cells
        rows.append((b, "TRB", f"bgB{i}", float(np.ceil(rng.lognormal(4, 1)))))
        if rng.random() < alpha_det:
            a = f"bgA{i}" if n_bg_alpha is None else f"pubA{rng.integers(n_bg_alpha)}"
            host_alpha[b] = a
            rows.append((b, "TRA", a, 50.0))
    free = np.setdiff1d(np.arange(side * side), cells)
    rng.shuffle(free)
    pos = 0
    for c in range(n_clones):
        k = int(rng.integers(2, 6))
        for b in free[pos : pos + k]:  # genuine cells of the clone
            rows.append((b, "TRB", f"clone{c}", float(np.ceil(rng.lognormal(np.log(genuine_reads), 0.5)))))
            if rng.random() < alpha_det:
                rows.append((b, "TRA", f"alpha{c}", 50.0))
        pos += k
        for b in rng.choice(cells, chimeras, replace=False):  # chimeric copies on other cells' bins
            rows.append((b, "TRB", f"clone{c}", float(rng.integers(2, 10))))
    df = pd.DataFrame(rows, columns=["bin", "locus", "clone_id", "consensus_count"])
    return df.assign(
        spot_id=df.bin.astype(str), x_um=(df.bin % side) * 30.0, y_um=(df.bin // side) * 30.0, library_id="S1"
    )


def test_known_answer():
    rows = [  # spot, locus, clone, reads, x
        ("s1", "TRB", "A", 100.0, 0.0),
        ("s1", "TRA", "a1", 9.0, 0.0),
        ("s2", "TRB", "A", 50.0, 100.0),
        ("s2", "TRA", "a1", 9.0, 100.0),
        ("s3", "TRB", "A", 3.0, 200.0),
        ("s3", "TRA", "zz", 9.0, 200.0),
        ("s4", "TRB", "B", 80.0, 300.0),
        ("s4", "TRA", "b1", 9.0, 300.0),
        ("s5", "TRB", "B", 60.0, 400.0),
        ("s5", "TRA", "b2", 9.0, 400.0),
        ("s6", "TRB", "C", 70.0, 500.0),
        ("s7", "TRB", "C", 2.0, 505.0),  # 5 um from s6: one locus with it at 15 um
    ]
    df = pd.DataFrame(rows, columns=["spot_id", "locus", "clone_id", "consensus_count", "x_um"]).assign(y_um=0.0)
    r = qc.read_floor(df, floors=(20,), n_perm=20, rng=0)
    t = r.table.set_index("side")
    assert t.loc["above", "n_loci"] == 5  # A:s1, A:s2, B:s4, B:s5, C:s6+s7 (best record 70 reads)
    assert t.loc["above", "n_tested"] == 2 and t.loc["above", "n_shared"] == 1  # A shares a1; C has no partner
    assert t.loc["above", "statistic"] == pytest.approx(0.5)
    assert t.loc["below", "n_loci"] == 1 and np.isnan(t.loc["below", "statistic"])
    loci = r.loci.set_index("clone_id")
    assert loci.loc["C", "n_spots"] == 2 and loci.loc["C", "reads_max"] == 70
    # without merging, s7 is a unit of its own below the floor
    r0 = qc.read_floor(df, floors=(20,), radius_um=None, n_perm=5, rng=0)
    assert r0.table.set_index("side").loc["below", "n_loci"] == 2
    # TR4 definition: every clonotype with >= 2 loci counts, sharing or not (C has a single locus)
    r4 = qc.read_floor(df.assign(locus=df.locus.replace({"TRA": "TRX"})), partner_locus="TRX", floors=(20,),
                       sides=("above",), min_partner_loci=0, n_perm=5, rng=0)  # fmt: skip
    assert r4.table.n_tested.iloc[0] == 2 and r4.table.n_shared.iloc[0] == 1
    no_partner = df[df.spot_id != "s2"]  # A keeps one partner-bearing locus above the floor
    r5 = qc.read_floor(no_partner, floors=(20,), sides=("above",), min_partner_loci=0, n_perm=5, rng=0)
    assert r5.table.n_tested.iloc[0] == 1 and r5.table.n_shared.iloc[0] == 0


def test_matches_legacy_diag_draw_for_draw():
    df = _tcr(seed=1, n_bg=600, n_clones=25, side=40, n_bg_alpha=40)
    b = df[df.locus == "TRB"].rename(columns={"spot_id": "cell_id", "clone_id": "clone"})
    b = b.sort_values("consensus_count", ascending=False).drop_duplicates(["cell_id", "clone"])
    a = df[df.locus == "TRA"]
    bin_a = a.groupby("spot_id").clone_id.apply(set).to_dict()
    leg_rng, pkg_rng = np.random.default_rng(0), np.random.default_rng(0)
    inp = pd.concat([b.rename(columns={"cell_id": "spot_id", "clone": "clone_id"}), a])  # TRB in the legacy order
    for floor, side in ((2, "above"), (20, "above"), (20, "below")):
        n_t, obs, sh = _legacy_floor(b, bin_a, floor, side, 8, leg_rng)
        r = qc.read_floor(inp, floors=(floor,), sides=(side,), radius_um=None, n_perm=8, rng=pkg_rng)
        t = r.table.iloc[0]
        assert t.n_tested == n_t
        assert t.statistic == pytest.approx(obs, rel=1e-12)
        assert t.null_mean == pytest.approx(sh, rel=1e-12)


def test_remerge_reproduces_merge_loci_partition():
    df = _tcr(seed=2)
    trb = df[df.locus == "TRB"].drop_duplicates(["spot_id", "clone_id"]).reset_index(drop=True)
    loci, member = merge_loci(trb, radius_um=45.0, by=())
    lab = pd.factorize(loci.clone_id)[0]
    eng = _Sharing(member.to_numpy(), trb[["x_um", "y_um"]].to_numpy(), np.zeros(len(trb)), (np.empty(0, int),) * 2,
                   1, 45.0, True)  # fmt: skip
    comp, unit_lab = eng.units(lab)
    t = pd.DataFrame({"a": comp, "b": member.to_numpy()})
    assert t.groupby("a").b.nunique().max() == 1 and t.groupby("b").a.nunique().max() == 1
    np.testing.assert_array_equal(np.sort(unit_lab), np.sort(lab))


def test_power_floor_and_chance_below():
    df = _tcr(seed=3, n_clones=60)
    r = qc.read_floor(df, floors=(2, 10, 20, 50), n_perm=99, rng=1)
    t = r.table.set_index(["side", "floor"])
    for f in (10, 20, 50):  # genuine cells (about 200 reads) share their alpha chain
        assert t.loc[("above", f), "p_value"] <= 0.02 and t.loc[("above", f), "ratio"] > 3
    assert t.loc[("below", 10), "p_value"] > 0.05  # chimeric copies (2-9 reads) share at chance level
    # every floor between the chimeras (< 10 reads) and the cells (about 200) separates them; the rule takes the
    # largest floor whose 'below' side is still at chance
    assert r.summary["suggested_floor"] in (10.0, 20.0, 50.0)


def test_q_values_and_the_suggestion_rule():
    """The suggestion is the largest floor with BH q <= alpha above and p > alpha below (raw p with multiplicity=None)."""
    from spaclone._utils import bh

    df = _tcr(seed=6, n_clones=50, n_bg_alpha=60)
    r = qc.read_floor(df, floors=(2, 5, 10, 20, 50, 100), n_perm=119, rng=3)
    t = r.table
    for side in ("above", "below"):
        s = t[t.side == side]
        np.testing.assert_allclose(s.q_value, bh(s.p_value.to_numpy(float)), equal_nan=True)
    ti = t.set_index(["side", "floor"])

    def rule(col):
        ok = [f for f in (2, 5, 10, 20, 50, 100)
              if ti.loc[("above", f), col] <= 0.05 and ti.loc[("below", f), "n_tested"] > 0
              and ti.loc[("below", f), "p_value"] > 0.05]  # fmt: skip
        return float(max(ok)) if ok else np.nan

    assert r.summary["suggested_floor"] == pytest.approx(rule("q_value"), nan_ok=True)
    assert r.summary["suggested_floor_unadjusted"] == pytest.approx(rule("p_value"), nan_ok=True)
    legacy = qc.read_floor(df, floors=(2, 5, 10, 20, 50, 100), n_perm=119, rng=3, multiplicity=None)
    assert legacy.summary["suggested_floor"] == pytest.approx(r.summary["suggested_floor_unadjusted"], nan_ok=True)
    pd.testing.assert_frame_equal(legacy.table, t)  # the correction changes the suggestion, not the table
    again = qc.read_floor(df, floors=(2, 5, 10, 20, 50, 100), n_perm=119, rng=3)
    assert again.summary["suggested_floor"] == pytest.approx(r.summary["suggested_floor"], nan_ok=True)
    with pytest.raises(ValueError):
        qc.read_floor(df, multiplicity="holm")


def test_calibration_partner_independent_of_clonotype():
    """Labels independent of partners: the test rejects at <= alpha + binomial tolerance (with and without re-merging)."""
    rng = np.random.default_rng(0)
    reps, hits = 100, 0
    for r in range(reps):
        n = 250
        xy = rng.uniform(0, 600, (n, 2))
        has = np.flatnonzero(rng.random(n) < 0.6)
        part = (has, rng.integers(0, 25, len(has)))  # 25 public partner chains: chance sharing is common
        eng = _Sharing(np.arange(n), xy, np.zeros(n), part, 25, 15.0, remerge=bool(r % 2))
        lab = pd.factorize(rng.integers(0, 100, n))[0]
        t0, s0, _ = eng(lab, 2, 2)
        null = []
        for _ in range(39):
            t, s_, _ = eng(lab[rng.permutation(n)], 2, 2)
            null.append(s_ / t if t else np.nan)
        hits += mc_pvalue(np.array(null), s0 / t0, alternative="greater")[0] <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_determinism_and_checks():
    df = _tcr(seed=4, n_bg=800, side=40, n_bg_alpha=30)
    a = qc.read_floor(df, floors=(20,), n_perm=10, rng=3)
    b = qc.read_floor(df, floors=(20,), n_perm=10, rng=3)
    c = qc.read_floor(df, floors=(20,), n_perm=10, rng=4)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.null, c.null, equal_nan=True)
    assert a.null.shape == (10, 2)
    shuffled = df.sample(frac=1.0, random_state=0)
    d = qc.read_floor(shuffled, floors=(20,), n_perm=10, rng=3)
    np.testing.assert_array_equal(d.table.statistic, a.table.statistic)  # row order does not change the loci
    with pytest.raises(ValueError):
        qc.read_floor(df, sides=("over",))
    with pytest.raises(ValueError):
        qc.read_floor(df, locus="TRA", partner_locus="TRA")
    with pytest.raises(ValueError):
        qc.read_floor(df.drop(columns="consensus_count"))


def test_placement_plot_of_read_floor_and_placebo():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spaclone import nulls, pl

    df = _tcr(seed=5, n_bg=500, side=40, n_bg_alpha=30)
    r = qc.read_floor(df, floors=(2, 20), n_perm=39, rng=0)
    ax = pl.placement(r)
    assert ax.get_xscale() == "log" and len(ax.get_legend().get_texts()) == 4  # two sides + observed + null
    plt.close(ax.figure)
    trb = df[df.locus == "TRB"].assign(rich=lambda d: d.consensus_count >= 20)
    pb = nulls.placebo(lambda t: t.rich.mean(), trb, nulls.Permute("rich"), n_perm=9, rng=0)
    ax = pl.placement(pb, style="dark")
    assert ax.get_xticklabels()[0].get_text() == "value"
    plt.close(ax.figure)
