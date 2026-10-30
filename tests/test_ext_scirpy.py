"""Tests of spaclone.ext.scirpy: scirpy functions on spatial units, identical numbers, mapping back to records."""

import sys

import numpy as np
import pandas as pd
import pytest

ir = pytest.importorskip("scirpy")

from spaclone import io, tl  # noqa: E402
from spaclone.ext import scirpy as sx  # noqa: E402
from spaclone.ext.scirpy import LargeClonotypeWarning  # noqa: E402
from spaclone.ext.squidpy import DepthNullWarning  # noqa: E402
from spaclone.pp import define_clones  # noqa: E402

A = ("IGHV1-7*01", "IGHJ2*01", "CAREEFLGGFDYW", "TGTGCAAGAGAGGAGTTTCTGGGTGGCTTTGACTACTGG")
B = ("IGHV1-7*01", "IGHJ2*01", "CAREEFLGGFDHW", "TGTGCAAGAGAGGAGTTTCTGGGTGGCTTTGACCACTGG")
C = ("IGHV2-2*01", "IGHJ3*01", "CARNSHYYGSRYPFAYW", "TGTGCCAGAAATTCCCATTACTACGGTAGTAGGTACCCCTTTGCTTACTGG")
K = ("IGKV1-110*01", "IGKJ1*01", "CSQSTHVPFTF", "TGCTCTCAAAGTACACATGTTCCGTTCACGTTC")


def _rec(spot, x, y, locus, chain, reads=5.0, region="A", c="IGHM"):
    v, j, jaa, jnt = chain
    return {
        "spot_id": spot,
        "x_um": float(x),
        "y_um": float(y),
        "locus": locus,
        "v_call": v,
        "j_call": j,
        "junction_aa": jaa,
        "junction": jnt,
        "productive": True,
        "consensus_count": float(reads),
        "c_call": c,
        "library_id": "s1",
        "region": region,
    }


def _toy() -> pd.DataFrame:
    """Clone A: 3 molecules on s0 + s1 (5 µm away: one locus) and s2 (another locus); B one aa from A; C; light K."""
    rows = [_rec("s0", 0, 0, "IGH", A, reads=5 + k) for k in range(3)]
    rows += [
        _rec("s1", 5, 0, "IGH", A),
        _rec("s2", 100, 0, "IGH", A, region="B"),
        _rec("s3", 200, 0, "IGH", B, region="B"),
        _rec("s4", 300, 0, "IGH", C, region="B", c="IGHG2B"),
        _rec("s0", 0, 0, "IGK", K, c="IGKC"),
    ]
    return define_clones(pd.DataFrame(rows), method="exact")


def _grid_table(seed: int = 0, n_clones: int = 25, n_spots: int = 160) -> pd.DataFrame:
    """One IGH record per spot on a 20-µm grid (loci = spots), random clones of a few sequence families, 2 regions."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_spots)))
    ij = np.array([(i, j) for i in range(side) for j in range(side)])[:n_spots]
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    base = ["CAR" + "".join(rng.choice(aa, 8)) + "W" for _ in range(n_clones // 3 + 1)]
    juncs = []
    for k in range(n_clones):  # clones in families of three: one or two aa from a family base
        s = list(base[k // 3])
        for pos in rng.choice(np.arange(3, len(s) - 1), size=k % 3, replace=False):
            s[pos] = aa[(aa.index(s[pos]) + 1) % 20]
        juncs.append("".join(s))
    clone = rng.integers(0, n_clones, n_spots)
    rows = []
    for s, ((i, j), c) in enumerate(zip(ij, clone, strict=True)):
        jaa = juncs[c]
        rows.append(
            {
                "spot_id": f"b{s}",
                "x_um": 20.0 * i,
                "y_um": 20.0 * j,
                "locus": "IGH",
                "v_call": f"IGHV{1 + c % 4}-1*01",
                "j_call": "IGHJ1*01",
                "junction_aa": jaa,
                "junction": "".join("TGT" for _ in jaa),
                "productive": True,
                "consensus_count": float(rng.integers(1, 50)),
                "c_call": "IGHM",
                "library_id": "s1",
                "region": "left" if i < side / 2 else "right",
            }
        )
    return define_clones(pd.DataFrame(rows), method="exact")


def test_prepare_collapses_identical_chains():
    air = _toy()
    a1 = sx.prepare(air)
    a0 = sx.prepare(air, collapse=False)
    big = int(np.flatnonzero(a1.obs["n_records"].to_numpy() == 4)[0])  # the locus of s0 + s1
    assert a1.obs["chain_pairing"].iloc[big] == "orphan VDJ"
    assert a0.obs["chain_pairing"].iloc[big] == "multichain"
    assert (a1.obs["receptor_type"] == "BCR").all()


def test_clonotypes_hamming_known_answer():
    air = _toy()
    res = sx.clonotypes(air, metric="hamming", cutoff=1)
    lab = res.air.set_index("spot_id")
    heavy = res.air[res.air.locus == "IGH"]
    by_spot = heavy.groupby("spot_id")["scirpy_clone_id"].first()
    assert by_spot["s0"] == by_spot["s1"] == by_spot["s2"] == by_spot["s3"]  # A and B one aa apart
    assert by_spot["s4"] != by_spot["s0"]
    # B-cell units: heavy-chain clonotypes by default; the light-chain locus of s0 gets none
    assert res.params["receptor_arms_used"] == "VDJ"
    assert lab.loc["s0", "scirpy_clone_id"].nunique() == 1
    assert res.air.loc[res.air.locus == "IGK", "scirpy_clone_id"].isna().all()
    top = res.table.iloc[0]
    assert (top.n_units, top.n_records, top.n_clones) == (3, 6, 2)
    assert (res.summary["n_clonotypes"], res.summary["n_clustered"], res.summary["n_units"]) == (2, 4, 5)
    assert len(res.air) == len(air)
    both = sx.clonotypes(air, metric="hamming", cutoff=1, receptor_arms="all")
    assert both.air.set_index("spot_id").loc["s0", "scirpy_clone_id"].nunique() == 2  # heavy and light units
    assert both.summary["n_clonotypes"] == 3


def test_clonotypes_identical_to_direct_scirpy():
    air = _grid_table(1)
    res = sx.clonotypes(air, metric="levenshtein", cutoff=1, key_added="cc")
    a = io.to_scirpy(air, level="locus")
    ir.pp.index_chains(a, filter=["productive", "require_junction_aa", sx._distinct_chains()])
    ir.tl.chain_qc(a)
    ir.pp.ir_dist(a, metric="levenshtein", cutoff=1, sequence="aa", n_jobs=1)
    assert res.params["receptor_arms_used"] == "VDJ"  # heavy chains only: scirpy fails with "all" here
    ir.tl.define_clonotype_clusters(
        a, sequence="aa", metric="levenshtein", receptor_arms="VDJ", dual_ir="primary_only", key_added="cc", n_jobs=1
    )
    assert np.array_equal(res.adata.obs["cc"].astype(str).to_numpy(), a.obs["cc"].astype(str).to_numpy())
    # every record carries its unit's clonotype
    unit_cc = dict(zip(a.obs_names, a.obs["cc"].astype(str), strict=True))
    assert (res.air["cc"].astype(str).to_numpy() == res.air["scirpy_unit"].map(unit_cc).to_numpy()).all()
    assert res.air["cc"].nunique() < air["clone_id"].nunique()  # families merged


def test_to_records_restores_the_table():
    air = _toy()
    adata = sx.prepare(air)
    back = sx.to_records(adata, columns=[]).drop(columns="scirpy_unit")
    pd.testing.assert_frame_equal(back, air)


def test_clonal_expansion_equals_spaclone_clone_sizes():
    air = _grid_table(2)
    res = sx.clonal_expansion(air, breakpoints=(1, 2, 5), groupby="region")
    sizes = tl.clone_sizes(air, unit="locus", breakpoints=(1, 2, 5))
    exp = sizes.set_index("clone_id")["expansion"].astype(str)
    got = res.units.set_index("clone_id")["clonal_expansion"].astype(str)
    assert (got.to_numpy() == exp.reindex(got.index).to_numpy()).all()
    t = res.table
    assert set(t.columns) == {"region", "clonal_expansion", "n_units", "share"}
    assert np.allclose(t.groupby("region").share.sum(), 1.0)


def test_diversity_by_region_matches_scirpy_and_evenness():
    air = _grid_table(3)
    res = sx.diversity_by_region(air, groupby="region")
    a = io.to_scirpy(air, level="locus")
    a.obs["region"] = res.adata.obs["region"].to_numpy()
    direct = ir.tl.alpha_diversity(a, "region", target_col="clone_id", inplace=False)
    assert np.allclose(res.table.set_index("region")["value"], direct.iloc[:, 0].reindex(res.table.region))
    even = tl.diversity(air, by="region", metrics=["evenness"], unit="locus", rarefy=False, contrasts=None).table
    even = even.set_index("region")["value"]
    assert np.allclose(res.table.set_index("region")["value"], even.reindex(res.table.region))
    assert "normalized_shannon_entropy_clone_id" in res.units.columns


def test_repertoire_overlap_known_answer():
    res = sx.repertoire_overlap(_toy(), groupby="region")  # heavy-chain units: A | A, B, C
    row = res.table.iloc[0]
    assert (row.group_a, row.group_b, row.n_shared, row.n_a, row.n_b) == ("A", "B", 1, 1, 3)
    assert row.distance == pytest.approx(1 - 1 / 3)
    assert res.params["n_units_left_out"] == 1  # the light-chain locus
    row = sx.repertoire_overlap(_toy(), groupby="region", receptor_arms="all").table.iloc[0]
    assert (row.n_shared, row.n_a, row.n_b) == (1, 2, 3)
    assert row.distance == pytest.approx(1 - 1 / 4)


def test_vdj_usage_known_answer():
    t = sx.vdj_usage(_toy(), groupby="region").table.set_index(["region", "gene"])
    assert t.loc[("A", "IGHV1-7"), "n_units"] == 1
    assert t.loc[("B", "IGHV1-7"), "n_units"] == 2
    assert t.loc[("B", "IGHV2-2"), "share"] == pytest.approx(1 / 3)
    assert ("A", "IGKV1-110") not in t.index  # VDJ_1 only


def test_expansion_map_share_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    res = sx.expansion_map(_toy(), sigma_um=50, pixel_um=10, min_neff=0.5)
    assert res.summary["n_expanded"] == 2  # the two loci of clone A
    assert res.summary["n_counted"] == 4  # heavy-chain loci; the light-chain locus is left out
    assert res.share_map.p0 == pytest.approx(2 / 4)
    assert res.units["expanded"].sum() == 2 and res.units["expanded"].isna().sum() == 1
    assert res.plot() is not None
    assert res.plot(kind="units") is not None
    assert res.plot(rotate=False) is not None  # spaclone.pl.share_map
    allc = sx.expansion_map(_toy(), sigma_um=50, pixel_um=10, min_neff=0.5, receptor_arms="all")
    assert allc.share_map.p0 == pytest.approx(2 / 5)


def _modularity_table(seed: int, clustered: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cl, size = 30, 8
    if clustered:
        centres = rng.uniform(100, 900, (n_cl, 2))
        xy = np.concatenate([c + rng.normal(0, 15, (size, 2)) for c in centres])
    else:
        xy = rng.uniform(0, 1000, (n_cl * size, 2))
    clone = np.repeat([f"c{k}" for k in range(n_cl)], size)
    return pd.DataFrame(
        {
            "spot_id": [f"u{i}" for i in range(len(xy))],
            "x_um": xy[:, 0],
            "y_um": xy[:, 1],
            "locus": "IGH",
            "clone_id": clone,
            "junction_aa": [f"CAR{c}W" for c in clone],
            "v_call": "IGHV1-1*01",
            "j_call": "IGHJ1*01",
            "productive": True,
            "consensus_count": 3.0,
        }
    )


def test_clonotype_modularity_calibration_power_determinism():
    """Null model: clone labels placed at random over the units (positions fixed); power: compact planted clones.

    References: Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy clonotype modularity).
    """
    kw = {"level": "spot", "carry": "clone_id", "graph_radius_um": 60.0}
    p_null = []
    for seed in range(4):
        with pytest.warns(DepthNullWarning):
            r = sx.clonotype_modularity(_modularity_table(seed, False), rng=seed, **kw)
        p_null.append(r.table.p_value.to_numpy())
    p_null = np.concatenate(p_null)
    alpha, n = 0.05, len(p_null)
    assert (p_null < alpha).mean() <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n)
    with pytest.warns(DepthNullWarning):
        pw = sx.clonotype_modularity(_modularity_table(9, True), rng=1, **kw)
    assert (pw.table.q_value < 0.05).mean() >= 0.9
    assert pw.table.statistic.min() > 0
    with pytest.warns(DepthNullWarning):
        again = sx.clonotype_modularity(_modularity_table(9, True), rng=1, **kw)
    pd.testing.assert_frame_equal(pw.table, again.table)
    assert pw.params["random_state"] == again.params["random_state"]
    assert {"clonotype_modularity", "clonotype_modularity_fdr"} <= set(pw.units.columns)


def test_spot_level_needs_a_unit_clone_column():
    air = _toy()
    with pytest.raises(KeyError, match="clonotypes first"):
        sx.clonal_expansion(air, level="spot")
    res = sx.clonotypes(air, level="spot", key_added="cc")
    ce = sx.clonal_expansion(res.adata, target_col="cc")
    assert ce.adata is res.adata


def test_missing_scirpy_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "scirpy", None)
    with pytest.raises(ImportError, match=r"spaclone\[scirpy\]"):
        sx.clonotypes(_toy())


def _heavy_light_table(seed: int = 0, n_heavy: int = 120, n_light: int = 120) -> pd.DataFrame:
    """Loci 20 µm apart: heavy chains in 40 clones of two loci and 40 singletons (random CDR3s); light chains with
    two public IGK junctions shared by 60 unrelated loci each."""
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    clones = [f"CAR{''.join(rng.choice(aa, 10))}W" for _ in range(80)]
    heavy = [clones[k // 2] for k in range(80)] + clones[40:80]
    light = ["CQQYNSYPLTF", "CLQHGESPYTF"] * (n_light // 2)
    rows = []
    for k, (locus, jaa) in enumerate([("IGH", j) for j in heavy[:n_heavy]] + [("IGK", j) for j in light]):
        rows.append(
            {
                "spot_id": f"b{k}",
                "x_um": 20.0 * (k % 20),
                "y_um": 20.0 * (k // 20),
                "locus": locus,
                "v_call": "IGHV1-1*01" if locus == "IGH" else "IGKV1-1*01",
                "j_call": "IGHJ1*01" if locus == "IGH" else "IGKJ1*01",
                "junction_aa": jaa,
                "junction": "TGT" * len(jaa),
                "productive": True,
                "consensus_count": 5.0,
                "library_id": "s1",
            }
        )
    return define_clones(pd.DataFrame(rows), method="exact")


def test_bcr_defaults_to_heavy_chains_and_warns_on_light_chain_clonotypes():
    air = _heavy_light_table()
    res = sx.clonotypes(air, metric="hamming", cutoff=1)  # no warning: heavy chains only
    assert res.params["receptor_arms_used"] == "VDJ"
    assert res.summary["n_clustered"] == 120 and res.summary["largest"] == 2
    assert res.air.loc[res.air.locus == "IGK", "scirpy_clone_id"].isna().all()
    with pytest.warns(LargeClonotypeWarning, match="receptor_arms='VDJ'"):
        both = sx.clonotypes(air, metric="hamming", cutoff=1, receptor_arms="all")
    assert both.summary["largest"] == 60
    em = sx.expansion_map(air, sigma_um=50, pixel_um=10, min_neff=0.5)
    assert (em.summary["n_counted"], em.summary["n_expanded"]) == (120, 80)
    with pytest.warns(LargeClonotypeWarning, match="receptor_arms='VDJ'"):
        em_all = sx.expansion_map(air, sigma_um=50, pixel_um=10, min_neff=0.5, receptor_arms="all")
    assert (em_all.summary["n_counted"], em_all.summary["n_expanded"]) == (240, 200)
    with pytest.raises(ValueError, match="receptor_arms"):
        sx.clonal_expansion(air, receptor_arms="any")


def test_loose_heavy_chain_clonotype_warns_with_the_cutoff():
    air = _heavy_light_table(n_light=0)
    base = air.junction_aa.iloc[0]
    variants = [base[:k] + ("A" if base[k] != "A" else "C") + base[k + 1 :] for k in range(3, len(base) - 1)]
    idx = np.arange(1, 1 + 6 * len(variants))  # 61 loci within one aa of the base junction
    air.loc[idx, "junction_aa"] = [variants[i % len(variants)] for i in range(len(idx))]
    air = define_clones(air.drop(columns="clone_id"), method="exact")
    with pytest.warns(LargeClonotypeWarning, match="cutoff"):
        res = sx.clonotypes(air, metric="hamming", cutoff=1)
    assert res.summary["largest_share"] > 0.05
    assert sx.clonotypes(air, metric="identity").summary["largest_share"] <= 0.05


def test_tcr_units_keep_both_arms():
    air = _heavy_light_table()
    air["locus"] = np.where(air.locus == "IGH", "TRB", "TRA")
    air["v_call"] = np.where(air.locus == "TRB", "TRBV1*01", "TRAV1*01")
    air["j_call"] = np.where(air.locus == "TRB", "TRBJ1-1*01", "TRAJ1*01")
    with pytest.warns(LargeClonotypeWarning):  # the shared alpha junctions: named, not hidden
        res = sx.clonotypes(air)
    assert res.params["receptor_arms_used"] == "all"


def test_table_attrs_holding_a_dataframe_do_not_break_scirpy():
    air = _grid_table(3)
    air.attrs["spaclone"] = {"shm": {"baseline_table": pd.DataFrame({"a": [1, 2]})}}
    assert len(sx.diversity_by_region(air, groupby="region").table) == 2
    air["region_id"] = np.where(air.region == "left", 0, 1).astype(np.int64)  # e.g. hotspot_workflow records
    t = sx.diversity_by_region(air, groupby="region_id").table
    assert t.region_id.tolist() == [0, 1] and pd.api.types.is_integer_dtype(t.region_id)


def _principal_angle(xy: np.ndarray) -> float:
    v = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, -1]
    return float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))


def test_maps_rotate_the_long_axis_horizontal_and_draw_y_down():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4)
    n = 300
    t, w = rng.uniform(0, 2000, n), rng.uniform(-100, 100, n)
    c, s_ = np.cos(np.radians(40)), np.sin(np.radians(40))
    air = pd.DataFrame(
        {
            "spot_id": [f"u{i}" for i in range(n)],
            "x_um": t * c - w * s_,
            "y_um": t * s_ + w * c,
            "locus": "IGH",
            "clone_id": [f"c{i // 2}" if i < n // 2 else f"s{i}" for i in range(n)],
            "junction_aa": [f"CAR{i // 2}W" if i < n // 2 else f"CARS{i}W" for i in range(n)],
            "v_call": "IGHV1-1*01",
            "j_call": "IGHJ1*01",
            "productive": True,
            "consensus_count": 3.0,
        }
    )
    em = sx.expansion_map(air, sigma_um=50, pixel_um=10, min_neff=2)

    def drawn(ax):
        return np.vstack([col.get_offsets() for col in ax.collections if len(col.get_offsets())])

    ax = em.plot(kind="units")
    assert _principal_angle(drawn(ax)) < 5 and ax.get_ylim()[0] > ax.get_ylim()[1]
    ax = em.plot(kind="units", rotate=False)
    assert abs(_principal_angle(drawn(ax)) - 40) < 5
    ax = em.plot()  # spaclone.pl.share_map in the same frame
    assert len(ax.images) >= 1 and ax.get_ylim()[0] > ax.get_ylim()[1]
    assert ax.get_xlabel() == "µm along the long axis"
    assert em.plot(rotate=False).get_xlabel() == "x (µm)"
    plt.close("all")
