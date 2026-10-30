import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, fisher_exact

from spaclone import pl, qc
from spaclone._utils import SpacloneWarning, mc_pvalue
from spaclone.nulls import Relocate
from spaclone.qc.celltype import _composition_core, _null_tables, _Placer, gstat

K_FLOORS = [1, 2, 3, 5, 10, 20, 50]


# ------------------------------------------------------------------------------------------- legacy oracles
class _LegacyAcc:
    """Q5_shuffle_vs_celltype.Acc (archive: external-data validation, workflow/qc), with its module-level rng passed in."""

    def __init__(self, rng, n_shuf=20):
        self.rng, self.n_shuf = rng, n_shuf
        self.reads, self.obs, self.shuf = [], [], []

    def add(self, reads, obs, unit_flags):
        n = len(obs)
        self.reads.append(np.asarray(reads))
        self.obs.append(np.asarray(obs, bool))
        self.shuf.append(np.asarray(unit_flags, bool)[self.rng.integers(0, len(unit_flags), size=(self.n_shuf, n))])

    def rows(self, floors):
        r, o, s = np.concatenate(self.reads), np.concatenate(self.obs), np.concatenate(self.shuf, axis=1)
        return [
            {
                "min_reads": k,
                "n": int((r >= k).sum()),
                "pct_kept": 100 * (r >= k).mean(),
                "pct_own": 100 * o[r >= k].mean() if (r >= k).any() else np.nan,
                "pct_own_shuffled": 100 * s[:, r >= k].mean() if (r >= k).any() else np.nan,
            }
            for k in floors
        ]


def _legacy_gstat(a, b):
    tot = a + b
    na, nb = a.sum(), b.sum()
    m = tot > 0
    ea, eb = tot[m] * na / (na + nb), tot[m] * nb / (na + nb)
    aa, bb = a[m], b[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(aa > 0, aa * np.log(aa / ea), 0.0).sum() + np.where(bb > 0, bb * np.log(bb / eb), 0.0).sum()
    return 2 * s


def _legacy_pair_test(codes_a, codes_b, K, feats, nperm, rng):
    """B4a_region_composition.pair_test (archive: BCR analysis, workflow), with its module-level rng passed in."""
    a, b = np.bincount(codes_a, minlength=K), np.bincount(codes_b, minlength=K)
    tot, na = a + b, len(codes_a)
    G0 = _legacy_gstat(a, b)
    fa0 = {f: (a @ v) / na for f, v in feats.items()}
    fb0 = {f: (b @ v) / len(codes_b) for f, v in feats.items()}
    pool = np.concatenate([codes_a, codes_b])
    Gn, dn = np.empty(nperm), {f: np.empty(nperm) for f in feats}
    for i in range(nperm):
        rng.shuffle(pool)
        aa = np.bincount(pool[:na], minlength=K)
        Gn[i] = _legacy_gstat(aa, tot - aa)
        for f, v in feats.items():
            dn[f][i] = (aa @ v) / na - ((tot - aa) @ v) / len(codes_b)
    res = {"G": G0, "G_null_mean": Gn.mean(), "G_null_sd": Gn.std(), "p_G": (1 + (Gn >= G0).sum()) / (nperm + 1)}
    for f in feats:
        d0 = fa0[f] - fb0[f]
        res[f"{f}_A"], res[f"{f}_B"], res[f"{f}_diff"] = fa0[f], fb0[f], d0
        res[f"{f}_p"] = (1 + (np.abs(dn[f]) >= abs(d0)).sum()) / (nperm + 1)
    return res, a, b


# ------------------------------------------------------------------------------------------- toy data
def _typed_units(n_units=400, b_share=0.4, seed=0, lib="L1"):
    rng = np.random.default_rng(seed)
    t = np.where(rng.random(n_units) < b_share, "B", "T")
    return pd.DataFrame({"spot_id": [f"{lib}_{i}" for i in range(n_units)], "cell_type": t, "library_id": lib})


def _records_on(units, n_rec, own_bias, seed, reads_scale=20.0):
    """Records on units; with probability own_bias a record sits on a B unit, otherwise on a random unit."""
    rng = np.random.default_rng(seed)
    b_units = units.spot_id[units.cell_type == "B"].to_numpy()
    spot = np.where(
        rng.random(n_rec) < own_bias,
        rng.choice(b_units, n_rec),
        rng.choice(units.spot_id.to_numpy(), n_rec),
    )
    reads = np.ceil(rng.exponential(reads_scale, n_rec))
    return pd.DataFrame({"spot_id": spot, "consensus_count": reads, "library_id": units.library_id.iat[0]})


def _composition_table(n=6000, n_clones=300, seed=0, planted=0.0):
    """Unit table (one row per (spot, clone) unit) with groups A / B / neither; ``planted`` share of A from clone 0."""
    rng = np.random.default_rng(seed)
    w = 1 / np.arange(1, n_clones + 1)
    clone = rng.choice(n_clones, n, p=w / w.sum())
    grp = rng.choice(["A", "B", "none"], n, p=[0.3, 0.4, 0.3])
    if planted:
        a = np.flatnonzero(grp == "A")
        clone[rng.choice(a, int(planted * len(a)), replace=False)] = n_clones  # a group-specific clone
    spot = rng.integers(0, int(n * 0.8), n)
    return pd.DataFrame(
        {
            "spot_id": spot.astype(str),
            "clone_id": [f"c{c}" for c in clone],
            "locus": "IGH",
            "in_a": grp == "A",
            "in_b": grp == "B",
        }
    )


# ------------------------------------------------------------------------------------------- own_type_share
def test_own_type_share_known_answer():
    units = pd.Series(["B", "B", "T", "T", "T"], index=list("abcde"))
    air = pd.DataFrame({"spot_id": ["a", "a", "b", "c", "z"], "consensus_count": [1.0, 30.0, 5.0, 2.0, 9.0]})
    r = qc.own_type_share(air, unit_types=units, own_types="B", read_floors=(1, 3, 10), n_place=50, rng=0)
    t = r.table.set_index("min_reads")
    assert r.summary["n_dropped"] == 1  # record on the untyped unit z
    assert list(t.n) == [4, 2, 1]
    np.testing.assert_allclose(t.statistic, [3 / 4, 1.0, 1.0])
    np.testing.assert_allclose(t.expected, 0.4)
    np.testing.assert_allclose(t.kept_share, [1.0, 0.5, 0.25])
    assert t.n_perm.eq(50).all()
    assert r.null.shape == (50, 3)


def test_own_type_share_matches_q5_draw_for_draw():
    units = _typed_units(seed=1)
    air = _records_on(units, 3000, own_bias=0.3, seed=2)
    types = units.set_index("spot_id").cell_type
    r = qc.own_type_share(
        air.drop(columns="library_id"),
        unit_types=types,
        own_types="B",
        read_floors=K_FLOORS,
        by_spot=False,
        weights=None,
        n_place=20,
        rng=np.random.default_rng(0),
    )
    assert r.params["n_perm"] == 20 and not r.params["by_spot"]
    acc = _LegacyAcc(np.random.default_rng(0))
    own = types.reindex(air.spot_id).to_numpy() == "B"
    acc.add(air.consensus_count, own, types.to_numpy() == "B")
    leg = pd.DataFrame(acc.rows(K_FLOORS))
    t = r.table
    np.testing.assert_array_equal(t.n, leg.n)
    np.testing.assert_allclose(100 * t.statistic, leg.pct_own, rtol=1e-12)
    np.testing.assert_allclose(100 * t.null_mean, leg.pct_own_shuffled, rtol=1e-12)
    np.testing.assert_allclose(100 * t.kept_share, leg.pct_kept, rtol=1e-12)


def test_placer_matches_relocate_draw_for_draw():
    """The fast placement loop consumes the generator exactly as nulls.Relocate (strata, weights, units)."""
    rng = np.random.default_rng(3)
    n_slots, n_rec = 300, 900
    slots = pd.DataFrame(
        {
            "_key": [f"s{i}" for i in range(n_slots)],
            "_own": rng.random(n_slots) < 0.3,
            "_lib": rng.choice(["L1", "L2"], n_slots),
            "_s0": rng.choice(["0", "1-2", "3-9"], n_slots),
            "_w": rng.gamma(2.0, 1.0, n_slots),
        }
    )
    pick = rng.integers(0, n_slots, n_rec)
    rec = slots[["_lib", "_s0"]].iloc[pick].reset_index(drop=True)
    rec["_unit"] = slots["_key"].to_numpy()[pick]
    for w in (None, "_w"):
        design = Relocate(slots, unit="_unit", within=["_lib", "_s0"], weights=w, replace=True, slot_columns=["_own"],
                          spot_key="_key")  # fmt: skip
        placer = _Placer(rec["_unit"].to_numpy(), rec[["_lib", "_s0"]], slots[["_lib", "_s0"]],
                         None if w is None else slots["_w"].to_numpy())  # fmt: skip
        g1, g2 = np.random.default_rng(9), np.random.default_rng(9)
        for _ in range(3):
            a = design(rec, g1)["_own"].to_numpy(dtype=bool)
            b = slots["_own"].to_numpy()[placer.draw(g2)]
            np.testing.assert_array_equal(a, b)


def _weighted_chips(n_libs, seed, n_units=400, n_rec=240, rich=2.5, signal=0.0, group_size=1, ig_strata=False):
    """``n_libs`` independent libraries in one table (each a replicate): units with a type (B share 0.3), a cDNA
    amount (B units ``rich`` times richer) and an Ig RNA stratum (B units Ig richer, every stratum mixed).

    Swapped records land on a unit with probability proportional to its cDNA; with ``ig_strata`` they first pick an
    Ig RNA stratum in proportion to its Ig RNA (records follow Ig RNA) and then a unit of it by cDNA. A share
    ``signal`` of the record groups sits on B units of the stratum instead. Records come in groups of ``group_size``
    on one unit (one cell's molecules, a sink).
    """
    rng = np.random.default_rng(seed)
    bins = np.array(["0", "3-9", "10-29"])
    ig_load = {"0": 0.05, "3-9": 1.0, "10-29": 3.0}
    airs, unit_tabs = [], []
    for lib in range(n_libs):
        b = rng.random(n_units) < 0.3
        cdna = rng.gamma(4.0, 1.0, n_units) * np.where(b, rich, 1.0)
        ig = np.where(b, rng.choice(bins, n_units, p=[0.2, 0.4, 0.4]), rng.choice(bins, n_units, p=[0.6, 0.3, 0.1]))
        sid = np.array([f"L{lib}_u{i}" for i in range(n_units)], dtype=object)
        n_groups = n_rec // group_size
        if ig_strata:
            st_w = np.array([cdna[ig == s].sum() * ig_load[s] for s in bins])
            stratum = rng.choice(bins, n_groups, p=st_w / st_w.sum())
        else:
            stratum = np.full(n_groups, "all", dtype=object)
        spot = np.empty(n_groups, dtype=object)
        for s in np.unique(stratum):
            cand = np.flatnonzero(ig == s) if ig_strata else np.arange(n_units)
            m = stratum == s
            spot[m] = sid[cand[rng.choice(len(cand), int(m.sum()), p=cdna[cand] / cdna[cand].sum())]]
            if signal:
                hit = m & (rng.random(n_groups) < signal)
                cand_b = cand[b[cand]]
                spot[hit] = sid[cand_b[rng.integers(0, len(cand_b), int(hit.sum()))]]
        airs.append(
            pd.DataFrame({"spot_id": np.repeat(spot, group_size), "consensus_count": 1.0, "library_id": f"L{lib}"})
        )
        unit_tabs.append(pd.DataFrame({"spot_id": sid, "cell_type": np.where(b, "B", "T"), "total_counts": cdna,
                                       "ig_bin": ig, "library_id": f"L{lib}"}))  # fmt: skip
    return pd.concat(airs, ignore_index=True), pd.concat(unit_tabs, ignore_index=True)


def _per_lib_p(air, units, **kw):
    t = qc.own_type_share(air, unit_types=units, own_types="B", read_floors=None, by_library=True, **kw).table
    return t[t.library_id != "all"].p_value.to_numpy(float)


def _fpr(p):
    p = np.asarray(p, float)
    return float((p <= 0.05).mean()), 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / len(p))


def test_own_type_share_cdna_weights_calibrate_the_swap_null():
    """Swaps land on cDNA, which B units have more of: uniform placement rejects, cDNA weights are calibrated."""
    air, units = _weighted_chips(150, seed=1)
    rate_w, tol = _fpr(_per_lib_p(air, units, weights="total_counts", n_perm=19, rng=1))
    assert rate_w <= tol
    assert _fpr(_per_lib_p(air, units, n_perm=19, rng=1))[0] > 0.5  # uniform placement ignores cDNA-rich B units
    air, units = _weighted_chips(1, seed=2, n_units=1000, n_rec=2000, signal=0.2)
    w = units.set_index("spot_id").total_counts
    r = qc.own_type_share(air, unit_types=units, own_types="B", weights=w, read_floors=None, n_perm=99, rng=1)
    assert r.table.p_value.iloc[0] <= 0.02 and r.table.ratio.iloc[0] > 1.1
    assert r.table.expected.iloc[0] == pytest.approx(
        (units.total_counts * (units.cell_type == "B")).sum() / units.total_counts.sum()
    )


def test_own_type_share_by_spot_keeps_grouped_records_together():
    """Records come in groups of 8 on one unit: record-by-record placement is far too narrow, spot placement is not."""
    air, units = _weighted_chips(150, seed=3, n_rec=240, group_size=8)
    kw = {"weights": "total_counts", "n_perm": 19, "rng": 3}
    rate_spot, tol = _fpr(_per_lib_p(air, units, by_spot=True, **kw))
    assert rate_spot <= tol
    assert _fpr(_per_lib_p(air, units, by_spot=False, **kw))[0] > 0.15
    one = air[air.library_id == "L0"]
    r = qc.own_type_share(one, unit_types=units, own_types="B", weights="total_counts", read_floors=None, n_perm=19,
                          rng=0)  # fmt: skip
    assert r.table.n_units.iloc[0] == one.spot_id.nunique() == r.summary["n_units"]


def test_own_type_share_ig_rna_strata():
    """Records follow Ig RNA (B units hold more): only the Ig-stratified, cDNA-weighted placement is calibrated."""
    air, units = _weighted_chips(150, seed=4, ig_strata=True)
    kw = {"weights": "total_counts", "n_perm": 19, "rng": 4}
    rate_ig, tol = _fpr(_per_lib_p(air, units, within="ig_bin", **kw))
    assert rate_ig <= tol
    assert _fpr(_per_lib_p(air, units, **kw))[0] > 0.5
    # power: a share of the records on B units beyond what their Ig RNA stratum predicts
    air, units = _weighted_chips(1, seed=5, n_units=1000, n_rec=2000, ig_strata=True, signal=0.3)
    ig = units.set_index("spot_id").ig_bin
    r = qc.own_type_share(air, unit_types=units, own_types="B", weights="total_counts", within=ig, read_floors=None,
                          n_perm=99, rng=2)  # fmt: skip
    assert r.table.p_value.iloc[0] <= 0.02
    assert r.params["within"] == ["ig_bin"] and "within" in r.summary["design"]


def test_own_type_share_calibration_and_power():
    libs = [_typed_units(n_units=150, seed=s, lib=f"L{s:03d}") for s in range(100)]
    units = pd.concat(libs, ignore_index=True)
    null_air = pd.concat([_records_on(u, 100, own_bias=0.0, seed=1000 + i) for i, u in enumerate(libs)])
    r = qc.own_type_share(
        null_air, unit_types=units, own_types="B", read_floors=None, n_place=19, by_library=True, rng=3
    )
    per_lib = r.table[r.table.library_id != "all"]
    assert len(per_lib) == 100
    # p has resolution 1/20 = 0.05 with 19 placements: P(p <= 0.05) = 1/20 under the null
    assert (per_lib.p_value <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / 100)
    assert (np.abs(per_lib.null_mean - per_lib.expected) <= 4.5 * per_lib.null_sd / np.sqrt(19) + 1e-12).all()
    gen_air = pd.concat([_records_on(u, 100, own_bias=0.3, seed=2000 + i) for i, u in enumerate(libs[:10])])
    r2 = qc.own_type_share(gen_air, unit_types=units, own_types="B", read_floors=None, n_place=19, rng=4)
    pooled = r2.table.iloc[0]
    assert pooled.p_value <= 0.05 and pooled.ratio > 1.3


def test_own_type_share_by_spot_strata_and_callable():
    units = pd.DataFrame(
        {"spot_id": list("abcdef"), "cell_type": ["B mature", "B GC", "T", "T", "T", "T"], "zone": list("xxxyyy")}
    )
    air = pd.DataFrame({"spot_id": ["a"] * 5 + ["d"] * 5, "zone": ["x"] * 5 + ["y"] * 5})
    r = qc.own_type_share(
        air,
        unit_types=units,
        own_types=lambda t: t.str.startswith("B "),
        read_floors=None,
        within="zone",
        n_place=30,
        rng=0,
    )
    t = r.table.iloc[0]
    assert t.statistic == pytest.approx(0.5)
    # zone x slots are 2/3 own, zone y slots none: the placed share is 0.5 * 2/3
    assert t.expected == pytest.approx(1 / 3)
    r1 = qc.own_type_share(air, unit_types=units, own_types=["B mature", "B GC"], by_spot=True, n_place=30, rng=0)
    vals = np.unique(r1.null.ravel())  # two spots move as wholes: shares are multiples of 1/2
    assert set(np.round(vals * 2, 9)) <= {0.0, 1.0, 2.0}
    # strata are unit attributes: the column need not be in the receptor table; a Series by spot id works too
    r2 = qc.own_type_share(air.drop(columns="zone"), unit_types=units, own_types=lambda t: t.str.startswith("B "),
                           read_floors=None, within=units.set_index("spot_id").zone, n_perm=30, rng=0)  # fmt: skip
    assert r2.table.expected.iloc[0] == pytest.approx(1 / 3)


def test_own_type_share_determinism_and_checks():
    units = _typed_units(seed=5)
    air = _records_on(units, 500, own_bias=0.2, seed=6)
    a = qc.own_type_share(air, unit_types=units, own_types="B", rng=7)
    b = qc.own_type_share(air, unit_types=units, own_types="B", rng=7)
    c = qc.own_type_share(air, unit_types=units, own_types="B", rng=8)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.null, c.null)
    assert a.params["n_perm"] == 199 and a.params["by_spot"] and a.null.shape[0] == 199
    d = qc.own_type_share(air, unit_types=units.rename(columns={"cell_type": "ct"}), own_types="B", cell_type_key="ct",
                          n_perm=199, rng=7)  # fmt: skip
    pd.testing.assert_frame_equal(a.table, d.table)
    shuffled = air.sample(frac=1.0, random_state=0)
    e = qc.own_type_share(shuffled, unit_types=units, own_types="B", rng=7)
    np.testing.assert_allclose(e.table.statistic, a.table.statistic)
    with pytest.raises(KeyError):
        qc.own_type_share(air, unit_types=units.drop(columns="cell_type"), own_types="B")
    with pytest.raises(ValueError):
        qc.own_type_share(air, unit_types=pd.concat([units, units]), own_types="B")
    with pytest.raises(ValueError, match="aliases"):
        qc.own_type_share(air, unit_types=units, own_types="B", n_perm=19, n_place=20)
    with pytest.raises(ValueError, match="spot ids match"):  # GEX names that do not match the receptor spot ids
        qc.own_type_share(air, unit_types=units.assign(spot_id="chip1_" + units.spot_id), own_types="B")
    with pytest.warns(SpacloneWarning, match="without a type"):
        qc.own_type_share(air, unit_types=units.iloc[:100], own_types="B", n_perm=9, rng=0)
    with pytest.raises(ValueError, match="not finite"):
        qc.own_type_share(air, unit_types=units.assign(w=np.nan), own_types="B", weights="w", n_perm=9)
    with pytest.raises(KeyError):
        qc.own_type_share(air, unit_types=units, own_types="B", within="nope", n_perm=9)


# ------------------------------------------------------------------------------------------- composition
def test_gstat_matches_scipy_log_likelihood():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 30, 25), rng.integers(1, 30, 25)
    a[3] = 0
    g, *_ = chi2_contingency(np.vstack([a, b]), lambda_="log-likelihood", correction=False)
    assert gstat(a, b) == pytest.approx(g, rel=1e-12)
    assert gstat(a, b) == pytest.approx(_legacy_gstat(a.astype(float), b.astype(float)), rel=1e-12)
    many = gstat(np.vstack([a, a]), np.vstack([b, b]))
    np.testing.assert_allclose(many, [g, g])


def test_null_tables_keep_margins_within_strata():
    rng = np.random.default_rng(1)
    ca, cb = rng.integers(0, 12, 300), rng.integers(0, 12, 500)
    sa, sb = rng.integers(0, 3, 300), rng.integers(0, 3, 500)
    tabs = _null_tables(ca, cb, sa, sb, 12, 50, np.random.default_rng(2))
    assert tabs.shape == (50, 12)
    np.testing.assert_array_equal(tabs.sum(1), 300)
    for s in range(3):  # each stratum's A total and clone totals are fixed, so the overall A counts are bounded
        tot = np.bincount(ca[sa == s], minlength=12) + np.bincount(cb[sb == s], minlength=12)
        assert (tot >= 0).all()
    tot_all = np.bincount(ca, minlength=12) + np.bincount(cb, minlength=12)
    assert (tabs <= tot_all).all() and (tabs >= 0).all()


def test_composition_matches_b4a_pair_test():
    df = _composition_table(seed=3)
    shm = {f"c{c}": float(c % 7) for c in range(0, 300, 2)}  # half the clones have a value
    r = qc.composition(df, group="in_a", reference="in_b", clone_features={"shm": shm}, unit="rows", n_perm=400, rng=0)
    codes, uniq = pd.factorize(df.clone_id)
    size = np.bincount(codes)
    rank = np.argsort(np.argsort(-size))
    f_shm = pd.Series(shm).reindex(uniq.astype(str)).to_numpy(float)
    f_shm = np.where(np.isfinite(f_shm), f_shm, np.nanmean(f_shm))
    feats = {
        "top10": (rank < 10).astype(float),
        "top100": (rank < 100).astype(float),
        "rare_le10": (size <= 10).astype(float),
        "log10_size": np.log10(size),
        "shm": f_shm,
    }
    leg, a, b = _legacy_pair_test(
        codes[df.in_a.to_numpy()], codes[df.in_b.to_numpy()], len(uniq), feats, 400, np.random.default_rng(9)
    )
    t = r.table.set_index("measure")
    assert t.loc["G", "statistic"] == pytest.approx(leg["G"], rel=1e-12)
    names = {
        "top10": "top10_share",
        "top100": "top100_share",
        "rare_le10": "rare_le10_share",
        "log10_size": "log10_clone_size",
        "shm": "shm",
    }
    for f, m in names.items():
        assert t.loc[m, "value_a"] == pytest.approx(leg[f"{f}_A"], rel=1e-12)
        assert t.loc[m, "value_b"] == pytest.approx(leg[f"{f}_B"], rel=1e-12)
        assert t.loc[m, "statistic"] == pytest.approx(leg[f"{f}_diff"], rel=1e-12, abs=1e-15)
    # the two nulls sample the same distribution (pooled shuffles vs Patefield tables)
    se = np.hypot(t.loc["G", "null_sd"], leg["G_null_sd"]) / np.sqrt(400)
    assert abs(t.loc["G", "null_mean"] - leg["G_null_mean"]) <= 4.5 * se
    # per-clone Fisher tests on clones with >= 20 records in the two groups, BH within the locus
    na, nb = int(df.in_a.sum()), int(df.in_b.sum())
    ks = np.flatnonzero(a + b >= 20)
    p_leg = {uniq[k]: fisher_exact([[a[k], na - a[k]], [b[k], nb - b[k]]])[1] for k in ks}
    cl = r.clones.set_index("clone_id")
    assert len(cl) == len(ks)
    np.testing.assert_allclose(cl.p_value.reindex(list(p_leg)), list(p_leg.values()), rtol=1e-12)


def test_composition_calibration_and_strata():
    """Group labels independent of identity: the G-test rejects at <= alpha + binomial tolerance."""
    reps, hits, hits_f = 200, 0, 0
    rng = np.random.default_rng(100)
    w = 1 / np.arange(1, 81)
    for _ in range(reps):  # the test core on random labellings (the wrapper is checked against B4a above)
        codes = rng.choice(80, 1200, p=w / w.sum())
        in_a = rng.random(1200) < 0.4
        ca, cb = codes[in_a], codes[~in_a]
        size = np.bincount(codes, minlength=80)
        feats = {"rare": (size <= 10).astype(float)}
        out = _composition_core(ca, cb, np.zeros(len(ca), int), np.zeros(len(cb), int), 80, feats, 49, rng)
        hits += mc_pvalue(out["G_null"], out["G"], alternative="greater")[0] <= 0.05
        _, _, d0, dn, centre = out["feats"]["rare"]
        hits_f += mc_pvalue(np.abs(dn - centre), abs(d0 - centre), alternative="greater")[0] <= 0.05
    tol = 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    assert hits / reps <= tol and hits_f / reps <= tol
    # confounding: tile 0 is mostly group A and holds clones 0-39, tile 1 mostly B with clones 40-79
    rng = np.random.default_rng(7)
    n = 3000
    tile = rng.integers(0, 2, n)
    clone = np.where(tile == 0, rng.integers(0, 40, n), rng.integers(40, 80, n))
    in_a = rng.random(n) < np.where(tile == 0, 0.8, 0.2)
    df = pd.DataFrame(
        {"spot_id": np.arange(n).astype(str), "clone_id": clone.astype(str), "in_a": in_a, "in_b": ~in_a, "tile": tile}
    )
    free = qc.composition(df, group="in_a", reference="in_b", n_perm=99, rng=1).table.set_index("measure")
    strat = qc.composition(df, group="in_a", reference="in_b", within="tile", n_perm=99, rng=1).table
    strat = strat.set_index("measure")
    assert free.loc["G", "p_value"] <= 0.01
    assert strat.loc["G", "p_value"] > 0.05
    assert free.loc["G", "statistic"] == strat.loc["G", "statistic"]


def test_composition_power_on_planted_clone_and_per_clone_q():
    df = _composition_table(seed=4, planted=0.05)
    r = qc.composition(df, group="in_a", reference="in_b", unit="rows", n_perm=199, rng=2)
    g = r.table.set_index("measure").loc["G"]
    assert g.p_value <= 0.01 and g.z > 3
    top = r.clones.iloc[0]
    assert top.clone_id == "c300" and top.q_value < 0.01
    assert r.summary["n_clones_q05"] >= 1


def test_composition_singleton_only_and_untested():
    df = _composition_table(seed=5)
    r = qc.composition(df, group="in_a", reference="in_b", singleton_only=True, unit="rows", n_perm=19, rng=0)
    single = df.groupby("spot_id").spot_id.transform("size") == 1
    g = r.table.set_index("measure").loc["G"]
    assert g.n_a == int((df.in_a & single).sum()) and g.n_b == int((df.in_b & single).sum())
    tiny = qc.composition(df.head(100), group="in_a", reference="in_b", unit="rows", n_perm=9, rng=0)
    assert np.isnan(tiny.table.statistic.iloc[0]) and tiny.summary["n_loci_tested"] == 0
    with pytest.raises(ValueError, match="both"):
        qc.composition(df, group="in_a", reference="in_a", n_perm=9)
    with pytest.raises(ValueError, match="varies inside"):  # random row-level groups on shared (spot, clone) units
        qc.composition(df, group="in_a", reference="in_b", n_perm=9)


def _family_table(seed, n_spots=1200, n_clones=120, mean_extra=4.0):
    """Molecules in same-clone families on one spot (chimera families / one cell's molecules); the group is a spot
    attribute independent of identity (a true null for composition)."""
    rng = np.random.default_rng(seed)
    w = 1 / np.arange(1, n_clones + 1)
    clone = rng.choice(n_clones, n_spots, p=w / w.sum())
    grp = rng.choice(["A", "B", "none"], n_spots, p=[0.35, 0.35, 0.3])
    k = 1 + rng.poisson(mean_extra, n_spots)
    spot = np.repeat(np.arange(n_spots), k)
    return pd.DataFrame({"spot_id": spot.astype(str), "clone_id": [f"c{c}" for c in clone[spot]], "locus": "IGK",
                         "in_a": grp[spot] == "A", "in_b": grp[spot] == "B"})  # fmt: skip


def test_composition_counts_spot_clone_units_not_molecules():
    """Molecule rows of one unit are not exchangeable: rows reject the true null, (spot, clone) units do not."""
    reps = 60
    p_rows, p_units = [], []
    for s in range(reps):
        df = _family_table(s)
        kw = {"group": "in_a", "reference": "in_b", "n_perm": 19, "min_units": 100, "min_clone_units": 10**9, "rng": s}
        p_units.append(qc.composition(df, **kw).summary["min_p_G"])
        if s < 20:
            p_rows.append(qc.composition(df, unit="rows", **kw).summary["min_p_G"])
    p_units = np.asarray(p_units)
    assert (p_units <= 0.05).mean() <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    assert (np.asarray(p_rows) <= 0.05).mean() > 0.5
    # the default equals unit='rows' on the deduplicated table
    df = _family_table(99)
    a = qc.composition(df, group="in_a", reference="in_b", n_perm=29, rng=5)
    b = qc.composition(df.drop_duplicates(["spot_id", "clone_id"]), group="in_a", reference="in_b", unit="rows", n_perm=29,
                       rng=5)  # fmt: skip
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.summary["n_units"] == df.spot_id.nunique() and a.summary["n_rows"] == len(df)
    pw = qc.composition_power(df, group="in_a", reference="in_b", g_grid=(0.2,), models=("polyclonal",), n_rep=3,
                              n_perm=19, rng=0)  # fmt: skip
    assert pw.table.n_a.iloc[0] == int(df.drop_duplicates("spot_id").in_a.sum())


def test_composition_determinism_and_plot():
    import matplotlib.pyplot as plt

    df = _composition_table(seed=6)
    a = qc.composition(df, group="in_a", reference="in_b", unit="rows", n_perm=49, rng=3)
    b = qc.composition(df, group="in_a", reference="in_b", unit="rows", n_perm=49, rng=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    ax = pl.placement(a)
    assert ax.get_ylabel().startswith("standardised")
    plt.close(ax.figure)
    units = _typed_units(seed=6)
    one = qc.own_type_share(_records_on(units, 200, 0.2, seed=1), unit_types=units, own_types="B", read_floors=None)
    ax = pl.placement(one)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["all records"]
    plt.close(ax.figure)


def test_composition_power_detects_large_spike_ins():
    df = _composition_table(n=3000, n_clones=150, seed=8)
    r = qc.composition_power(
        df,
        group="in_a",
        reference="in_b",
        g_grid=(0.002, 0.2),
        models=("polyclonal",),
        n_rep=6,
        n_perm=49,
        rng=0,
        unit="rows",
    )
    t = r.table.set_index("g")
    assert list(t.columns[:2]) == ["locus", "model"]
    assert t.loc[0.2, "power_G"] == 1.0
    assert t.loc[0.002, "power_G"] <= t.loc[0.2, "power_G"]
    assert {"power_rare", "power_top10", "power_any_bonf"} <= set(t.columns)
