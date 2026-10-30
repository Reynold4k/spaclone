import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from spaclone import pl, tl
from spaclone._utils import LowPowerWarning, SpacloneWarning


def _tiny():
    # clone A: spots s1 (2 records) and s2 10 um apart -> one locus; s3 far away -> second locus; clone B: one record
    return pd.DataFrame(
        {
            "spot_id": ["s1", "s1", "s2", "s3", "s1"],
            "x_um": [0.0, 0.0, 10.0, 500.0, 0.0],
            "y_um": [0.0, 0.0, 0.0, 0.0, 0.0],
            "locus": "IGH",
            "clone_id": ["A", "A", "A", "A", "B"],
            "region": ["GC", "GC", "PC", "PC", "GC"],
            "consensus_count": [5.0, 3.0, 2.0, 1.0, 4.0],
        }
    )


def _random_air(n=600, n_spots=300, n_clones=60, seed=0, p=None, region_split=150.0):
    rng = np.random.default_rng(seed)
    spot = rng.integers(0, n_spots, n)
    df = pd.DataFrame(
        {
            "spot_id": spot.astype(str),
            "x_um": (spot % 30) * 10.0,
            "y_um": (spot // 30) * 10.0,
            "locus": "IGH",
            "clone_id": rng.choice([f"c{i}" for i in range(n_clones)], n, p=p),
        }
    )
    df["region"] = np.where(df["x_um"] < region_split, "A", "B")
    return df


def test_clone_units_record_spot_locus():
    air = _tiny()
    rec = tl.clone_units(air, unit="record").units
    assert len(rec) == 5
    spot = tl.clone_units(air, unit="spot").units
    assert spot["clone_id"].tolist() == ["A", "A", "A", "B"]  # first-seen order
    assert spot["n_records"].tolist() == [2, 1, 1, 1]
    loc = tl.clone_units(air, unit="locus", radius_um=15).units
    assert len(loc) == 3
    a = loc[loc["clone_id"] == "A"].sort_values("x_um")
    assert a["n_spots"].tolist() == [2, 1]
    assert a["x_um"].tolist() == [5.0, 500.0]  # centroid of the two distinct spots
    assert a["reads_max"].tolist() == [5.0, 1.0]
    # loci never span groups
    split = tl.clone_units(air, unit="locus", by="region", radius_um=15).units
    assert len(split) == 4
    # an existing locus_id column is used as given
    with_ids = air.assign(locus_id=[0, 0, 0, 0, 1])
    assert len(tl.clone_units(with_ids, unit="locus").units) == 2
    cu = tl.clone_units(air.assign(clone_id=["A", None, "A", "A", "B"]), unit="spot")
    assert len(cu.membership) == 4


def test_clone_sizes_and_expansion_classes():
    air = _tiny()
    cs = tl.clone_sizes(air, unit="spot", breakpoints=(1, 2))
    assert cs["clone_id"].tolist() == ["A", "B"]
    assert cs["size"].tolist() == [3, 1]
    assert cs["n_spots"].tolist() == [3, 1]
    assert cs["expansion"].astype(str).tolist() == ["> 2", "<= 1"]
    assert cs["frequency"].sum() == pytest.approx(1.0)
    rec = tl.clone_sizes(air, unit="record")
    assert rec.set_index("clone_id").loc["A", "n_spots"] == 3
    ce = tl.clonal_expansion(air, by="region", unit="spot", breakpoints=(1,))
    gc = ce[ce["region"] == "GC"].set_index("expansion")
    # GC: A on s1 (A has 3 spots overall) and B (1 spot): one unit in each class
    assert gc.loc["<= 1", "n_units"] == 1 and gc.loc["> 1", "n_units"] == 1
    assert ce.groupby("region")["share_units"].sum().tolist() == pytest.approx([1.0, 1.0])
    within = tl.clonal_expansion(air, by="region", unit="spot", breakpoints=(1,), expanded_in="region")
    pc = within[within["region"] == "PC"].set_index("expansion")
    assert pc.loc["> 1", "n_units"] == 2  # A has two spots inside PC


def test_metric_values_known_answer():
    air = pd.DataFrame({"clone_id": list("AAABCD"), "g": "x"})
    r = tl.diversity(
        air,
        by="g",
        unit="record",
        rarefy=False,
        metrics=("richness", "shannon", "simpson", "inverse_simpson", "evenness", "gini", "chao1", "top1", "expanded"),
    )
    v = r.table.set_index("metric")["value"]
    p = np.array([0.5, 1 / 6, 1 / 6, 1 / 6])
    h = -(p * np.log(p)).sum()
    assert v["richness"] == 4
    assert v["shannon"] == pytest.approx(h)
    assert v["simpson"] == pytest.approx(1 - (p**2).sum())
    assert v["inverse_simpson"] == pytest.approx(1 / (p**2).sum())
    assert v["evenness"] == pytest.approx(h / np.log(4))
    assert v["gini"] == pytest.approx(0.25)  # sorted 1,1,1,3: (-3 - 1 + 1 + 9) / (4 * 6)
    assert v["chao1"] == pytest.approx(7.0)  # 4 + 3 * 2 / (2 * 1)
    assert v["top1"] == pytest.approx(0.5)
    assert v["expanded"] == pytest.approx(0.5)
    custom = tl.diversity(air, by="g", unit="record", rarefy=False, metrics=[lambda c: float(c.max())])
    assert custom.table["value"].iloc[0] == 3.0
    with pytest.raises(ValueError, match="unknown diversity metric"):
        tl.diversity(air, by="g", unit="record", metrics=("hill7",))


def test_rarefaction_replays_the_archive_loop():
    rng0 = np.random.default_rng(4)
    labels = rng0.choice(list("abcdefghijklmnop"), 300, p=np.r_[0.4, np.full(15, 0.04)])
    air = pd.DataFrame({"clone_id": labels})
    gen = np.random.default_rng(0)
    r = tl.diversity(air, by=None, unit="record", metrics=("top10", "gini"), rarefy_to=120, n_rarefy=50, rng=gen)

    def gini(c):
        x = np.sort(np.asarray(c, float))
        n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

    ref = np.random.default_rng(0)
    codes = pd.factorize(pd.Series(labels))[0]
    top, gi = np.empty(50), np.empty(50)
    for i in range(50):
        c = np.bincount(codes[ref.choice(len(codes), 120, replace=False)], minlength=codes.max() + 1)
        c = np.sort(c[c > 0])[::-1]
        top[i], gi[i] = c[:10].sum() / 120, gini(c)
    t = r.table.set_index("metric")
    assert t.loc["top10", "rarefied"] == top.mean()
    assert t.loc["gini", "rarefied"] == gi.mean()
    assert t.loc["gini", "rarefied_lo"] == np.quantile(gi, 0.025)
    assert t.loc["top10", "rarefied_hi"] == np.quantile(top, 0.975)
    # a group of exactly rarefy_to units is used whole and draws nothing
    state = gen.bit_generator.state
    whole = tl.diversity(air, by=None, unit="record", metrics=("gini",), rarefy_to=300, rng=gen)
    assert gen.bit_generator.state == state
    assert whole.table["rarefied"].iloc[0] == pytest.approx(whole.table["value"].iloc[0])
    with pytest.warns(LowPowerWarning):
        tl.diversity(air, by=None, unit="record", metrics=("gini",), rarefy_to=301, rng=1)


def _calibration(reps, n_perm, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        n = 160
        air = pd.DataFrame(
            {
                "clone_id": rng.choice([f"c{i}" for i in range(30)], n, p=np.r_[np.full(5, 0.1), np.full(25, 0.02)]),
                "region": np.where(rng.random(n) < 0.25, "GC", "PC"),
            }
        )
        with pytest.warns(SpacloneWarning, match="placebo='labels'"):
            r = tl.diversity(
                air, by="region", unit="record", metrics=("gini",), rarefy=False, placebo="labels", n_perm=n_perm,
                rng=rng,
            )  # fmt: skip
        hits += r.table.set_index("group").loc["GC - PC", "p_value"] <= 0.05
    return hits / reps


def test_placebo_is_calibrated_under_random_labels():
    """Two regions of different size, clone labels exchangeable: P(p <= 0.05) <= 0.05 + binomial tolerance."""
    reps = 20
    assert _calibration(reps, 19, 11) <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


@pytest.mark.slow
def test_placebo_is_calibrated_under_random_labels_many_reps():
    reps = 300
    assert _calibration(reps, 39, 12) <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_placebo_shows_a_built_in_gini_difference_and_detects_a_real_one():
    rng = np.random.default_rng(2)
    # one heavy-tailed clone distribution, the GC holds 10x fewer molecules: its raw Gini is lower (mostly singletons)
    # and the placebo reproduces the difference, as in AIR-SPACE (archive X3b)
    freq = 1 / np.arange(1, 301)
    labels = rng.choice([f"c{i}" for i in range(300)], 1100, p=freq / freq.sum())
    air = pd.DataFrame({"clone_id": labels, "region": np.r_[np.full(100, "GC"), np.full(1000, "PC")]})
    with pytest.warns(SpacloneWarning):
        r = tl.diversity(
            air, by="region", unit="record", metrics=("gini",), rarefy=False, placebo="labels", n_perm=99, rng=3
        )
    c = r.table.set_index("group").loc["GC - PC"]
    assert c["value"] < -0.1 and c["null_mean"] < -0.1 and c["p_value"] > 0.05
    # planted: one clone dominates the GC -> lower diversity than random placement gives
    planted = air.copy()
    planted.loc[:59, "clone_id"] = "dominant"
    with pytest.warns(SpacloneWarning):
        r2 = tl.diversity(
            planted, by="region", unit="record", metrics=("shannon",), rarefy=True, placebo="labels", n_perm=99, rng=3
        )
    assert r2.table.set_index("group").loc["GC - PC", "p_value"] <= 0.02


def test_diversity_is_deterministic_and_order_invariant():
    air = _random_air(seed=5)
    a = tl.diversity(air, by="region", unit="spot", placebo="clones", n_perm=20, n_rarefy=20, rng=7)
    b = tl.diversity(air, by="region", unit="spot", placebo="clones", n_perm=20, n_rarefy=20, rng=7)
    pd.testing.assert_frame_equal(a.table, b.table)
    np.testing.assert_array_equal(a.null, b.null)
    c = tl.diversity(air, by="region", unit="spot", placebo="clones", n_perm=20, n_rarefy=20, rng=8)
    assert not np.array_equal(a.null, c.null)
    shuffled = air.sample(frac=1.0, random_state=1)
    x = tl.diversity(air, by="region", unit="locus", rarefy=False).table
    y = tl.diversity(shuffled, by="region", unit="locus", rarefy=False).table
    np.testing.assert_allclose(x["value"].to_numpy(), y["value"].to_numpy())
    moved = air.assign(x_um=air["x_um"] + 1000.0)
    z = tl.diversity(moved, by="region", unit="locus", rarefy=False).table
    np.testing.assert_allclose(x["value"].to_numpy(), z["value"].to_numpy())


def test_overlap_known_values_and_placebo():
    air = pd.DataFrame(
        {
            "clone_id": ["a", "b", "a", "b", "c", "d", "a", "c"],
            "region": ["X", "X", "Y", "Y", "Z", "Z", "W", "W"],
        }
    )
    r = tl.overlap(air, by="region", unit="record", metric=list(tl.OVERLAP_METRICS))
    t = r.table.set_index(["group_a", "group_b", "metric"])["value"]
    assert t[("X", "Y", "morisita_horn")] == pytest.approx(1.0)
    assert t[("X", "Y", "jaccard")] == pytest.approx(1.0)
    assert t[("X", "Z", "jaccard")] == 0.0 and t[("X", "Z", "bhattacharyya")] == 0.0
    # W = {a, c}, X = {a, b}: Jaccard 1/3, Sorensen 1/2, Bhattacharyya 0.5, Morisita-Horn 2*0.25 / (0.5 + 0.5) = 0.5
    assert t[("W", "X", "jaccard")] == pytest.approx(1 / 3)
    assert t[("W", "X", "sorensen")] == pytest.approx(0.5)
    assert t[("W", "X", "bhattacharyya")] == pytest.approx(0.5)
    assert t[("W", "X", "morisita_horn")] == pytest.approx(0.5)
    pres = pd.crosstab(air["region"], air["clone_id"]) > 0
    jd = 1 - pdist(pres.to_numpy(), "jaccard")
    assert sorted(np.round(jd, 9)) == sorted(np.round(r.table.loc[r.table["metric"] == "jaccard", "value"], 9))
    big = _random_air(seed=3)
    p = tl.overlap(big, by="region", unit="spot", metric="jaccard", placebo="labels", n_perm=30, rng=1)
    assert p.table["n_perm"].iloc[0] == 30 and 0 < p.table["p_value"].iloc[0] <= 1


def test_gene_usage_and_spectratype():
    air = pd.DataFrame(
        {
            "clone_id": ["a", "a", "b", "c"],
            "region": ["X", "X", "X", "Y"],
            "spot_id": ["s1", "s2", "s3", "s4"],
            "v_call": ["IGHV1-1*01", "IGHV1-1*02", "IGHV2-2*01", "IGHV2-2*01"],
            "junction_aa": ["CARW", "CARW", "CAKKW", "CAW"],
        }
    )
    gu = tl.gene_usage(air, by="region", unit="spot", weight="clones")
    x = gu[gu["region"] == "X"].set_index("gene")
    assert x.loc["IGHV1-1", "count"] == 1 and x.loc["IGHV2-2", "count"] == 1
    gr = tl.gene_usage(air, by="region", unit="spot", weight="records").set_index(["region", "gene"])
    assert gr.loc[("X", "IGHV1-1"), "count"] == 2
    sp = tl.cdr3_length(air, by="region", unit="spot", weight="records")
    assert set(sp["length"]) == {3, 4, 5}
    assert sp.groupby("region")["share"].sum().tolist() == pytest.approx([1.0, 1.0])
    cdr3 = tl.cdr3_length(air, unit="spot", weight="records", anchors=False)
    assert cdr3["length"].min() == 1


def test_errors():
    air = _tiny()
    with pytest.raises(ValueError, match="unit must be"):
        tl.clone_units(air, unit="cell")
    with pytest.raises(KeyError):
        tl.clone_units(air.drop(columns="spot_id"), unit="spot")
    with pytest.raises(ValueError, match="at least two groups"):
        tl.overlap(air.assign(region="one"), by="region", unit="record")
    with pytest.raises(ValueError, match="contrast"):
        tl.diversity(air, by="region", unit="record", contrasts=[("GC", "nope")])
    with pytest.raises(TypeError, match="placebo"):
        tl.diversity(air, by="region", unit="record", placebo=3)


def test_plots_return_axes():
    air = _random_air(seed=9)
    air["v_call"] = np.where(air["clone_id"].str[-1].isin(list("0123")), "IGHV1-1*01", "IGHV2-2*01")
    air["junction_aa"] = np.where(air["clone_id"].str.len() > 2, "CARDW", "CARW")
    d = tl.diversity(air, by="region", unit="spot", placebo="clones", n_perm=10, n_rarefy=10, rng=1)
    for ax in (
        pl.diversity(d),
        pl.diversity(d, kind="contrast"),
        pl.overlap(tl.overlap(air, by="region", unit="spot")),
        pl.gene_usage(tl.gene_usage(air, by="region", unit="spot")),
        pl.gene_usage(tl.gene_usage(air, unit="spot")),
        pl.spectratype(tl.cdr3_length(air, by="region", unit="spot")),
        pl.clonal_expansion(tl.clonal_expansion(air, by="region", unit="spot")),
    ):
        assert isinstance(ax, plt.Axes)
    fig, ax = plt.subplots()
    assert pl.diversity(d, ax=ax) is ax
    plt.close("all")


# ------------------------------------------------------------------------------------- clone relocation placebo


def _clustered(rng, n_cells=900, n_clones=90, sd=20.0, split=150.0, top_in_a=0, width=600.0, height=300.0):
    """Cells with Zipf clone sizes clustered around clone centres; region A = x < split. ``top_in_a`` largest clones
    are centred inside A (a truly more clonal region)."""
    w = 1.0 / np.arange(1, n_clones + 1) ** 1.1
    lab = rng.choice(n_clones, n_cells, p=w / w.sum())
    cx, cy = rng.uniform(0, width, n_clones), rng.uniform(0, height, n_clones)
    if top_in_a:
        top = np.argsort(-np.bincount(lab, minlength=n_clones))[:top_in_a]
        cx[top] = rng.uniform(20, split - 20, top_in_a)
    x = np.clip(cx[lab] + rng.normal(0, sd, n_cells), 0, width)
    y = np.clip(cy[lab] + rng.normal(0, sd, n_cells), 0, height)
    return pd.DataFrame(
        {
            "spot_id": np.arange(n_cells).astype(str),
            "x_um": x,
            "y_um": y,
            "locus": "IGH",
            "clone_id": [f"c{v}" for v in lab],
            "region": np.where(x < split, "A", "B"),
        }
    )


def test_clone_relocation_keeps_footprints_and_identity():
    rng = np.random.default_rng(51)
    u = tl.clone_units(_clustered(rng), unit="spot", by="region").units
    d = tl.CloneRelocation(
        u, by=["region"], within=["locus"], section=None, clone_key="clone_id", x_key="x_um", y_key="y_um"
    )
    out = d(u, np.random.default_rng(0))
    assert len(out) == len(u) and (out.clone_id == u.clone_id).all() and set(out.region) <= {"A", "B"}
    assert (out.region != u.region).any()
    # a clone whose centroid is given back to itself (the identity) keeps every unit in its group
    d._prepare(u)
    ident = d._group[np.arange(len(u))]
    np.testing.assert_array_equal(d._group_values.iloc[ident].region.to_numpy(), u.region.to_numpy())
    assert d.describe()["design"] == "CloneRelocation"
    with pytest.raises(KeyError, match="coordinates"):
        tl.CloneRelocation(u.drop(columns="x_um"), by=["region"], within=[], section=None, clone_key="clone_id",
                           x_key="x_um", y_key="y_um")  # fmt: skip
    with pytest.raises(ValueError, match="needs by"):
        tl.diversity(_clustered(rng), by=None, unit="spot", placebo="clones", n_perm=5)


@pytest.mark.parametrize(
    ("reps", "placebos"), [(12, ("clones",)), pytest.param(200, ("clones", "labels"), marks=pytest.mark.slow)]
)
def test_clone_relocation_contrasts_are_calibrated_with_clustered_clones(reps, placebos):
    """Equal clone-size distributions, clones clustered, regions of unequal size (a quarter vs the rest): the
    contrasts of placebo='clones' reject at <= alpha + tolerance; the label placebo rejects most of the time."""
    rng = np.random.default_rng(52)
    hits = dict.fromkeys(placebos, 0)
    for _ in range(reps):
        air = _clustered(rng)
        seed = int(rng.integers(2**31))
        for placebo in placebos:
            # two-sided: with 39 placebo data sets the smallest p is 2 / 40 = 0.05 (with 19 it would be 0.1)
            kw = {"by": "region", "unit": "spot", "metrics": ("shannon",), "placebo": placebo, "n_perm": 39,
                  "n_rarefy": 3, "rng": seed}  # fmt: skip
            if placebo == "labels":
                with pytest.warns(SpacloneWarning):
                    r = tl.diversity(air, **kw)
            else:
                r = tl.diversity(air, **kw)
            hits[placebo] += r.table.set_index("group").loc["A - B", "p_value"] <= 0.05
    assert hits["clones"] / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    if "labels" in hits:
        assert hits["labels"] > hits["clones"]


def test_clone_relocation_detects_a_more_clonal_region():
    """40 expanded clones (15 cells each) centred in region A (a quarter), 600 singletons anywhere: random placement
    of intact clones puts about 10 expanded clones in A, so A is far less diverse than the placebo."""
    rng = np.random.default_rng(53)
    n_exp, n_single = 40, 600
    cx = np.r_[rng.uniform(20, 130, n_exp), rng.uniform(0, 600, n_single)]
    cy = rng.uniform(0, 300, n_exp + n_single)
    lab = np.r_[np.repeat(np.arange(n_exp), 15), np.arange(n_exp, n_exp + n_single)]
    x = np.clip(cx[lab] + rng.normal(0, 12, len(lab)) * (lab < n_exp), 0, 600)
    y = np.clip(cy[lab] + rng.normal(0, 12, len(lab)) * (lab < n_exp), 0, 300)
    air = pd.DataFrame({"spot_id": np.arange(len(lab)).astype(str), "x_um": x, "y_um": y, "locus": "IGH",
                        "clone_id": lab.astype(str), "region": np.where(x < 150, "A", "B")})  # fmt: skip
    r = tl.diversity(air, by="region", unit="spot", metrics=("simpson",), placebo="clones", n_perm=99, n_rarefy=5,
                     rng=1)  # fmt: skip
    c = r.table.set_index("group").loc["A - B"]
    assert c.statistic < c.null_mean and c.p_value <= 0.05


def test_overlap_with_clone_relocation():
    rng = np.random.default_rng(54)
    air = _clustered(rng, n_cells=600, n_clones=60)
    r = tl.overlap(air, by="region", unit="spot", metric=("jaccard", "morisita_horn"), placebo="clones", n_perm=19,
                   rng=2)  # fmt: skip
    assert r.params["placebo"]["design"] == "CloneRelocation" and (r.table.n_perm == 19).all()
    assert r.table.p_value.between(0, 1).all() and np.isfinite(r.table.null_mean).all()
