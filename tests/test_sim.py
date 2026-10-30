import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from spaclone import nulls, sim, validate_air
from spaclone.pp import filter_valid

#: Folder of the archived legacy demo (``synthetic.meta.parquet``, ``truth.json``); its test is skipped when the
#: variable is unset or the folder is missing.
ARCHIVE = Path(os.environ.get("SPACLONE_TEST_LEGACY_DEMO", "") or "/nonexistent-legacy-demo")
SMALL = {
    "size_um": 300.0,
    "tissue_radius_um": 120.0,
    "spot_um": 4.0,
    "t_zone_radius_um": 35.0,
    "follicle_width_um": 50.0,
    "n_gc": 2,
    "gc_radius_um": 22.0,
    "n_pc_foci": 2,
    "pc_radius_um": 18.0,
    "sink_size": 40,
}


def _small(**kw):
    return sim.synthetic_chip(**{**SMALL, **kw})


def nn_same_clone(d, locus="IGH"):
    """Median distance from a record to the nearest other record of its clone (clones with >= 2 records)."""
    d = d[d.locus == locus]
    code = pd.factorize(d.clone_id)[0]
    keep = np.bincount(code)[code] >= 2
    pts = np.column_stack([d.x_um.to_numpy(), d.y_um.to_numpy(), code * 1e7])[keep]
    if len(pts) < 2:
        return np.nan
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dist[:, 1]))


# ------------------------------------------------------------------------------------------------ synthetic_chip


@pytest.mark.parametrize("legacy", [False, True])
def test_synthetic_chip_truth_schema_and_positions(legacy):
    sa = sim.synthetic_chip(rng=0, legacy=legacy)
    air, gex, meta = sa
    validate_air(air)
    assert set(air.truth) == {"genuine", "swapped", "sink"}
    assert meta["model"] == ("legacy" if legacy else "realistic")
    nonsink = air[air.truth != "sink"]
    share = (nonsink.truth == "genuine").mean()
    assert abs(share - 0.3) < 4 * np.sqrt(0.3 * 0.7 / len(nonsink))
    g = air[air.truth == "genuine"]
    d = np.hypot(g.x_um - g.true_x_um, g.y_um - g.true_y_um)
    assert d.max() <= 4.0 + 2.0 * np.sqrt(2) / 2 + 1e-9  # cell radius + half a spot diagonal
    sw = air[air.truth == "swapped"]
    assert sw.spot_id.isin(gex.obs_names).all()  # a swapped record sits on a captured cDNA
    sinks = meta["sinks"]
    assert len(sinks) == 4 and sinks.in_tissue.sum() == 2
    assert (air[air.truth == "sink"].groupby("spot_id").size() == 150).all()
    assert air.groupby("spot_id").clone_id.nunique().max() >= 50  # a sink is not a cell
    assert meta["um_per_unit"] == 2.0 and (air.x_um == air.x_src * 2.0).all()
    assert {"cells", "lineages", "clones", "structures"} <= set(meta)
    assert gex.n_obs > 1000 and (~gex.obs.in_tissue).sum() > 0  # off-tissue spots are kept
    assert air.umi.str.len().eq(10).all() and meta["level"] == "molecule"
    if not legacy:
        assert set(sw.swap_kind) == {"cdna"} and (air.loc[air.truth != "swapped", "swap_kind"] == "").all()
        assert "captured_cdna" in gex.obs and (gex.obs.captured_cdna >= gex.obs.total_counts).all()


def test_legacy_flag_reproduces_the_dev0_chip():
    """Fingerprint of spaclone 1.0.0.dev0 ``synthetic_chip(rng=0)`` (recorded 2026-10-30 from the original code)."""
    sa = sim.synthetic_chip(rng=0, legacy=True)
    a = sa.air
    assert len(a) == 7962 and sa.gex.n_obs == 17043 and float(sa.gex.X.sum()) == 41868.0
    assert float(a.x_um.sum()) == 2362076.0 and float(a.y_um.sum()) == 2387678.0
    assert float(a.consensus_count.sum()) == 22796.0
    assert a.clone_id.iloc[[0, 100, 5000]].tolist() == [
        "TRB|TRBV5|TRBJ2|CIEKMAHISQCF", "IGH|IGHV5|IGHJ4|CFGKDISPGNGNVW", "IGL|IGLV16|IGLJ1|CHTAVAISCVCYPTSF"
    ]  # fmt: skip
    assert sa.meta["truth_counts"]["swapped|IGK"] == 2070 and sa.meta["truth_counts"]["sink|IGH"] == 283
    assert sa.meta["p_det_given_occupied"] == pytest.approx(0.8135405105438401, rel=1e-12)


def test_one_heavy_and_one_light_clone_per_cell():
    sa = sim.synthetic_chip(rng=1)
    L = sa.meta["lineages"]
    assert L.heavy_clone.str.split("|").str[0].isin(["IGH", "TRB"]).all()
    assert L.light_clone.str.split("|").str[0].isin(["IGK", "IGL", "TRA"]).all()
    g = sa.air[sa.air.truth == "genuine"]
    per_cell = g.groupby("true_cell").clone_id.nunique()
    assert per_cell.max() == 2
    lin = L.set_index("lineage")
    rec = g.drop_duplicates(["true_cell", "clone_id"])
    allowed = rec.lineage.map(lin.heavy_clone).eq(rec.clone_id) | rec.lineage.map(lin.light_clone).eq(rec.clone_id)
    assert allowed.all()
    # every molecule of a cell carries that cell's constant region (isotype, light class, TRBC gene)
    cells = sa.meta["cells"].set_index("cell_id")
    mol = sa.air[sa.air.truth != "sink"]
    heavy = mol.locus.isin(["IGH", "TRB"])
    want = np.where(heavy, mol.true_cell.map(cells.heavy_c_call), mol.true_cell.map(cells.light_c_call))
    assert (mol.c_call.to_numpy() == want).all()


def test_structures_hold_their_lineages():
    sa = sim.synthetic_chip(rng=2, gc_to_pc=1.0)
    cells, st, L = sa.meta["cells"], sa.meta["structures"].set_index("structure"), sa.meta["lineages"]
    gc = cells[cells.cell_type == "GC B"]
    d = np.hypot(gc.x_um - gc.structure.map(st.x_um), gc.y_um - gc.structure.map(st.y_um))
    assert (d <= 40.0 + 1e-9).all()
    gc_lin = set(L[L.compartment == "germinal centre"].lineage)
    pc = cells[(cells.cell_type == "plasma") & cells.structure.str.startswith("PC")]
    assert set(pc.lineage) <= gc_lin  # every focus lineage comes from a germinal centre
    # a germinal centre holds a few expanded lineages
    assert gc.groupby("structure").lineage.nunique().max() <= 4


def _hits(sa):
    air, gex = sa.air, sa.gex
    genes = list(gex.var_names)
    out = np.zeros(len(air), bool)
    on = air.spot_id.isin(gex.obs_names).to_numpy()
    r = gex.obs_names.get_indexer(air.spot_id[on])
    c = [genes.index(sim.MOUSE_CONSTANT_GENES[x]) for x in air.c_call[on]]
    out[on] = np.asarray(gex.X[r, c]).ravel() > 0
    return out


@pytest.mark.parametrize("legacy", [False, True])
def test_own_constant_transcript_is_captured_with_p_det(legacy):
    """Same seed: p_det only adds own transcripts, so records and background GEX are identical."""
    kw = {"legacy": True} if legacy else {"swap_weight": "background", "constant_detection": 0.5}
    hi, lo = sim.synthetic_chip(rng=3, p_det=0.6, **kw), sim.synthetic_chip(rng=3, p_det=0.0, **kw)
    drop = ["cell_type"] if legacy else ["cell_type", "own_seen"]
    pd.testing.assert_frame_equal(hi.air.drop(columns=drop), lo.air.drop(columns=drop))
    n_mol = int((hi.air.truth != "sink").sum())
    added = hi.gex.X.sum() - lo.gex.X.sum()
    assert abs(added - 0.6 * n_mol) < 4 * np.sqrt(0.6 * 0.4 * n_mol)
    h_hi, h_lo = _hits(hi), _hits(lo)
    assert (h_hi >= h_lo).all()
    miss = ~h_lo
    gen_, swp = miss & (hi.air.truth == "genuine").to_numpy(), miss & (hi.air.truth == "swapped").to_numpy()
    # a genuine record meets its own transcript with probability >= p_det; a swapped one only by chance
    assert h_hi[gen_].mean() >= 0.6 - 4 * np.sqrt(0.24 / gen_.sum())
    assert h_hi[swp].mean() < 0.3
    assert hi.meta["p_det_given_occupied"] > lo.meta["p_det_given_occupied"]
    if not legacy:
        own = hi.air.own_seen.to_numpy() & (hi.air.truth == "genuine").to_numpy()
        assert h_hi[own].all()  # a detected own transcript is a hit by construction


def _area_density(sa):
    """GEX molecules per grid spot in the white pulp, red pulp and plasma-cell foci (empty spots included)."""
    p = sa.meta["params"]
    spot_um, size_um, radius_um = p["spot_um"], p["size_um"], p["tissue_radius_um"]
    nx = int(np.floor(size_um / spot_um)) + 1
    lin = np.arange(nx * nx)
    xy = np.column_stack([(lin % nx) * spot_um, (lin // nx) * spot_um])
    c0 = size_um / 2
    d0 = np.hypot(xy[:, 0] - c0, xy[:, 1] - c0)
    comp = np.where(d0 <= p["t_zone_radius_um"] + p["follicle_width_um"], "white", "red").astype(object)
    st = sa.meta["structures"]
    for r in st[st.kind == "plasma-cell focus"].itertuples():
        comp[np.hypot(xy[:, 0] - r.x_um, xy[:, 1] - r.y_um) <= r.radius_um] = "focus"
    comp[d0 > radius_um] = "off"
    area = pd.Series(comp).value_counts()
    ij = sa.gex.obs_names.str.split("_", expand=True)
    where = comp[ij.get_level_values(1).astype(int) * nx + ij.get_level_values(0).astype(int)]
    mol = pd.Series(sa.gex.obs.total_counts.to_numpy()).groupby(where).sum()
    return (mol / area).to_dict()


WIDE = {**SMALL, "size_um": 400.0, "tissue_radius_um": 170.0, "follicle_width_um": 60.0, "n_sinks": 0, "spot_um": 2.0}


def test_realistic_cdna_depends_on_compartment_and_cell_type():
    d = _area_density(sim.synthetic_chip(rng=0, genuine_fraction=0.0, **WIDE))
    assert 1.4 <= d["white"] / d["red"] <= 2.0  # white pulp captures 1.5-1.8x more cDNA
    old = _area_density(sim.synthetic_chip(rng=0, genuine_fraction=0.0, legacy=True, **WIDE))
    assert old["white"] / old["red"] < 1.15
    flat = sim.synthetic_chip(rng=0, genuine_fraction=0.0, cdna_by_type=dict.fromkeys(sim.CDNA_BY_TYPE, 1.0),
                              cdna_by_compartment=dict.fromkeys(sim.CDNA_BY_COMPARTMENT, 1.0),
                              constant_per_cell=dict.fromkeys(sim.CONSTANT_PER_CELL, 0.0), p_det=0.0, **WIDE)  # fmt: skip
    d = _area_density(flat)
    assert abs(d["white"] / d["red"] - 1.0) < 0.08
    const = flat.gex.var_names.isin(list(sim.MOUSE_CONSTANT_GENES.values()))
    assert float(flat.gex.X[:, np.flatnonzero(const)].sum()) == 0.0  # no constant-region RNA at all


def test_constant_region_rna_is_confined_to_its_cells_and_bursty():
    sa = sim.synthetic_chip(rng=4, p_det=0.0, constant_detection=1.0, **WIDE)
    gex = sa.gex
    names = list(gex.var_names)
    X = gex.X.tocsc()
    ct = gex.obs.cell_type.astype(str).to_numpy()
    for gene, types in (("Igkc", {"B", "GC B", "plasma"}), ("Ighm", {"B", "GC B", "plasma"}), ("Trac", {"T"})):
        v = np.asarray(X[:, names.index(gene)].todense()).ravel()
        assert v.sum() > 0 and set(ct[v > 0]) <= types  # only on the spots of cells that express it
    ik = np.asarray(X[:, names.index("Igkc")].todense()).ravel()
    pos = ik[ik > 0]
    assert (pos >= 2).mean() > 0.3  # several molecules per spot: bursts, not a uniform sprinkle
    legacy = sim.synthetic_chip(rng=4, legacy=True, **WIDE).gex
    lk = np.asarray(legacy.X.tocsc()[:, list(legacy.var_names).index("Igkc")].todense()).ravel()
    assert set(legacy.obs.cell_type.astype(str).to_numpy()[lk > 0]) >= {"none", "other"}  # legacy: everywhere
    per = sa.meta["cells"].groupby("cell_type").constant_transcripts.mean()
    assert per["plasma"] > per["GC B"] > per["B"] > 0 and per["T"] > 0 and per["other"] == 0


def test_swaps_land_on_cdna_rich_receptor_rich_cells():
    def b_share(sa):
        sw = sa.air[sa.air.truth == "swapped"]
        return sw.cell_type.isin(["B", "GC B", "plasma"]).mean()

    real = [sim.synthetic_chip(rng=s, genuine_fraction=0.0, **WIDE) for s in range(2)]
    old = [sim.synthetic_chip(rng=s, genuine_fraction=0.0, legacy=True, **WIDE) for s in range(2)]
    assert np.mean([b_share(s) for s in real]) > 1.8 * np.mean([b_share(s) for s in old])
    # swaps land in proportion to the spot's GEX molecules: per-spot swap counts track total counts
    sa = real[0]
    n_sw = sa.air[sa.air.truth == "swapped"].spot_id.value_counts()
    tc = sa.gex.obs.total_counts
    per = n_sw.reindex(sa.gex.obs_names).fillna(0).groupby(np.minimum(tc.to_numpy(), 6)).mean()
    assert per.iloc[-1] > 3 * per.iloc[0] and np.corrcoef(per.index, per.to_numpy())[0, 1] > 0.9
    # some swaps meet their chain's constant-region RNA by chance (the legacy chip never placed them on it)
    assert _hits(sa)[(sa.air.truth == "swapped").to_numpy()].mean() > 0.01


def test_homologous_switch_keeps_the_acceptor_constant_region():
    sa = sim.synthetic_chip(rng=5, genuine_fraction=0.0, homologous_switch=1.0, p_det=0.5, **WIDE)
    air = sa.air
    sw = air[air.truth == "swapped"]
    assert (sw.swap_kind == "homologous").all() and sa.meta["swap_counts"] == {"homologous": len(sw)}
    hit = _hits(sa)[(air.truth == "swapped").to_numpy()]
    base = sim.synthetic_chip(rng=5, genuine_fraction=0.0, p_det=0.5, **WIDE)
    hit0 = _hits(base)[(base.air.truth == "swapped").to_numpy()]
    assert hit.mean() > 0.45 and hit.mean() > 2 * hit0.mean()  # "right isotype": the acceptor's transcript
    cells = sa.meta["cells"].set_index("cell_id")
    igh = sw[sw.locus == "IGH"]
    donor_iso = igh.true_cell.map(cells.heavy_c_call)
    assert (igh.c_call != donor_iso).mean() > 0.2  # the record carries the acceptor's isotype, not the donor's
    assert (sw.c_call.str[:3] == sw.locus).all()  # the acceptor's constant region is of the same locus
    half = sim.synthetic_chip(rng=5, genuine_fraction=0.0, homologous_switch=0.5, **WIDE)
    share = (half.air.loc[half.air.truth == "swapped", "swap_kind"] == "homologous").mean()
    assert abs(share - 0.5) < 0.05


def test_chimera_families_share_one_spot():
    sa = sim.synthetic_chip(rng=6, genuine_fraction=0.0, swap_family_mean=2.0, **WIDE)
    sw = sa.air[sa.air.truth == "swapped"]
    fam = sw.groupby("family").agg(n=("spot_id", "size"), spots=("spot_id", "nunique"), clones=("clone_id", "nunique"))
    assert (fam.spots == 1).all() and (fam.clones == 1).all()
    assert 2.0 < fam.n.mean() < 3.2  # 1 + Poisson(2), truncated at the clone's last molecule
    assert (sa.air.loc[sa.air.truth != "swapped", "family"] == -1).all()


def test_reads_options_umi_and_tcr_switches():
    sa = _small(rng=4, tcr=False, umi=False)
    assert not sa.air.locus.isin(["TRA", "TRB"]).any() and sa.air.umi.isna().all()
    assert sa.meta["level"] == "spot_clone"
    m = sa.air.groupby("truth").consensus_count.mean()
    assert m["genuine"] > 4.0 > m["swapped"]
    eq = _small(rng=4, reads_genuine=1.0, reads_swapped=1.0).air.groupby("truth").consensus_count.mean()
    assert abs(eq["genuine"] - eq["swapped"]) < 0.4


def test_nonproductive_clones_are_flagged_and_filtered():
    sa = sim.synthetic_chip(rng=5, nonproductive_fraction=0.2)
    bad = ~sa.air.productive
    assert bad.any() and sa.air.junction_aa[bad].str.contains(r"\*").all()
    ok, rep = filter_valid(sa.air, require_productive=True)
    assert len(ok) == int((~bad).sum())


@pytest.mark.parametrize("kw", [{}, {"legacy": True}, {"homologous_switch": 0.4, "swap_family_mean": 1.0}])
def test_synthetic_chip_is_deterministic(kw):
    a, b, c = _small(rng=7, **kw), _small(rng=7, **kw), _small(rng=8, **kw)
    pd.testing.assert_frame_equal(a.air, b.air)
    assert (a.gex.X != b.gex.X).nnz == 0
    pd.testing.assert_frame_equal(a.gex.obs, b.gex.obs)
    assert not a.air.equals(c.air)


def test_synthetic_chip_errors():
    with pytest.raises(ValueError, match="fit inside"):
        sim.synthetic_chip(size_um=100, tissue_radius_um=60)
    with pytest.raises(ValueError, match="p_det"):
        sim.synthetic_chip(p_det=2)
    with pytest.raises(ValueError, match="swap_weight"):
        sim.synthetic_chip(swap_weight="uniform")
    with pytest.raises(ValueError, match="homologous_switch"):
        sim.synthetic_chip(homologous_switch=1.5)
    with pytest.raises(ValueError, match="constant_detection"):
        sim.synthetic_chip(constant_detection=-0.1)


def test_clone_clustering_placebo_calibrated_when_all_swapped():
    """All molecules swapped: clone and position are independent, so P(p <= 0.05) <= 0.05 + tolerance."""
    rng = np.random.default_rng(11)
    reps, hits = 30, 0
    design = nulls.Permute("clone_id", within="locus")
    for _ in range(reps):
        sa = _small(genuine_fraction=0.0, n_sinks=0, rng=rng)
        r = nulls.placebo(nn_same_clone, sa.air, design, n_perm=19, alternative="less", rng=rng)
        hits += r.p_value <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_clone_clustering_placebo_has_power_when_genuine():
    rng = np.random.default_rng(12)
    design = nulls.Permute("clone_id", within="locus")
    ps = [
        nulls.placebo(nn_same_clone, _small(genuine_fraction=1.0, rng=rng).air, design, n_perm=19, rng=rng,
                      alternative="less").p_value
        for _ in range(5)
    ]  # fmt: skip
    assert np.mean(np.array(ps) <= 0.05) >= 0.8


# ----------------------------------------------------------------------------------------------- legacy demo port


def test_legacy_demo_chip_layout_and_truth(tmp_path):
    sa = sim.legacy_demo_chip(size=2400, radius=300, n_igl=500, n_igk=1200, n_igh=600, n_hot=60, sink_size=80, rng=0,
                              write_dir=tmp_path)  # fmt: skip
    air, meta = sa.air, sa.meta
    validate_air(air)
    assert (air[air.locus == "IGL"].truth == "genuine").all()
    hot = air[(air.clone_id == "9999") & (air.truth == "genuine")]
    hc = np.array(meta["truth"]["hot_center"])
    assert len(hot) == 60 and (np.hypot(hot.x_src - hc[0], hot.y_src - hc[1]) <= 40).all()
    assert (air.truth == "sink").sum() == 4 * 80 and air[air.truth == "sink"].spot_id.nunique() == 4
    assert air.consensus_count.sum() == len(meta["reads"])
    assert not air.umi.notna().any() and (meta["reads"].umi == meta["reads"].cid).all()
    assert set(meta["files"]) >= {"meta", "raw_gef", "truth"}
    assert json.loads(Path(meta["files"]["truth"]).read_text())["chip_size"] == 2400
    assert sa.gex.obs.total_counts.sum() > 0 and sa.gex.n_vars == 8


@pytest.mark.slow
@pytest.mark.skipif(not ARCHIVE.exists(), reason="set SPACLONE_TEST_LEGACY_DEMO to the archived demo")
def test_legacy_demo_chip_reproduces_the_archived_demo(tmp_path):
    sa = sim.legacy_demo_chip(rng=0, write_dir=tmp_path)
    old = pd.read_parquet(ARCHIVE / "synthetic.meta.parquet")
    pd.testing.assert_frame_equal(old, pd.read_parquet(sa.meta["files"]["meta"]))
    assert json.loads((ARCHIVE / "truth.json").read_text()) == sa.meta["truth"]


# ------------------------------------------------------------------------------------------------------- plant


def _background(rng=0):
    sa = _small(genuine_fraction=1.0, n_sinks=0, rng=rng)
    return sa.air[sa.air.locus.isin(["IGH", "IGK"])].reset_index(drop=True)


def test_plant_background_is_a_placebo():
    bg = _background(0)
    P = sim.plant(bg, fraction=0.0, rng=1)
    assert not P.air.planted.any() and len(P.implants) == 0
    # the real clustering is gone: clone multisets per locus kept, positions and reads fixed
    assert nn_same_clone(P.air) > 2 * nn_same_clone(bg)
    for lc in ("IGH", "IGK"):
        assert sorted(P.air[P.air.locus == lc].clone_id) == sorted(bg[bg.locus == lc].clone_id)
    np.testing.assert_array_equal(P.air.x_um.to_numpy(), bg.x_um.to_numpy())
    np.testing.assert_array_equal(P.air.consensus_count.to_numpy(), bg.consensus_count.to_numpy())
    # sequence fields travel with the clone id
    assert (P.air.clone_id.str.split("|").str[3] == P.air.junction_aa).all()


def test_plant_moves_records_into_cells_and_keeps_reads():
    bg = _background(1)
    P = sim.plant(bg, fraction=0.3, pattern="focus", rng=2)
    imp, air = P.implants, P.air
    assert air.planted.sum() == len(imp) > 0
    d_focus = np.hypot(imp.x_um - imp.focus_x_um, imp.y_um - imp.focus_y_um)
    assert d_focus.max() <= 25.0 + 4.0 + 1e-9
    d_cell = np.hypot(imp.x_um - imp.cell_x_um, imp.y_um - imp.cell_y_um)
    assert d_cell.max() <= 4.0 + 1e-9
    assert np.allclose(imp[["target_x_um", "target_y_um"]].to_numpy(), imp[["x_um", "y_um"]].to_numpy())
    # paired heavy and light clones share cells
    shared = imp.groupby("cell").locus.nunique()
    assert (shared == 2).any()
    lin = imp.groupby("lineage").locus.agg(lambda s: tuple(sorted(set(s))))
    assert (lin == ("IGH", "IGK")).any()
    # planted records are background records: per clone the read counts are unchanged
    before = sim.plant(bg, fraction=0.0, rng=2).air.groupby("clone_id").consensus_count.apply(sorted)
    after = air.groupby("clone_id").consensus_count.apply(sorted)
    assert before.equals(after)


def test_plant_patterns_sites_grids_slots_and_dedupe():
    bg = _background(2)
    sites = np.array([[150.0, 150.0]])
    P = sim.plant(bg, fraction=0.2, pattern="scattered", sites=sites, grid_um=4.0, rng=3)
    imp = P.implants
    assert imp.focus.isna().all() and np.allclose(imp[["cell_x_um", "cell_y_um"]].to_numpy(), 150.0)
    moved = P.air[P.air.planted]
    assert (moved.x_um % 4.0 == 0).all() and (moved.spot_id == (moved.x_um / 4).astype(int).astype(str) + "_"
                                              + (moved.y_um / 4).astype(int).astype(str)).all()  # fmt: skip
    assert (moved.x_src * 4.0 == moved.x_um).all()
    slots = pd.DataFrame({"spot_id": ["s0", "s1"], "x_um": [140.0, 400.0], "y_um": [140.0, 400.0]})
    Q = sim.plant(bg, fraction=0.2, pattern="gc", sites=sites, slots=slots, rng=3)
    assert set(Q.air[Q.air.planted].spot_id) == {"s0"}
    t = Q.implants[["target_x_um", "target_y_um"]].to_numpy()
    assert (np.hypot(t[:, 0] - 150.0, t[:, 1] - 150.0) <= 40.0 + 4.0 + 1e-9).all()
    keys = ["spot_id", "locus", "clone_id"]
    D = sim.plant(bg, fraction=0.2, pattern="gc", sites=sites, slots=slots, dedupe=keys, rng=3)
    assert not D.air.duplicated(keys).any()
    assert D.params["n_planted"] == D.air.planted.sum() == len(D.implants)
    assert (D.air.iloc[D.implants.row].planted).all()
    one = bg.clone_id.value_counts().index[0]
    C = sim.plant(bg, fraction=1.0, pattern={one: "gc"}, clones=[one], rng=4)
    assert set(C.implants.clone_id) == {one} and set(C.implants.pattern) == {"gc"}
    with pytest.raises(ValueError, match="pattern"):
        sim.plant(bg, fraction=0.1, pattern="ring")
    with pytest.raises(ValueError, match="fraction"):
        sim.plant(bg, fraction=1.5)


def test_plant_is_deterministic():
    bg = _background(3)
    a, b, c = (sim.plant(bg, fraction=0.2, rng=s) for s in (5, 5, 6))
    pd.testing.assert_frame_equal(a.air, b.air)
    pd.testing.assert_frame_equal(a.implants, b.implants)
    assert not a.air.equals(c.air)


def test_placebo_on_planted_background_is_calibrated_and_powered():
    """fraction = 0: the background is a null, P(p <= 0.05) <= 0.05 + tol; planted foci are found."""
    rng = np.random.default_rng(21)
    design = nulls.Permute("clone_id", within="locus")
    reps, hits = 30, 0
    bg = _background(4)
    for _ in range(reps):
        P = sim.plant(bg, fraction=0.0, rng=rng)
        hits += nulls.placebo(nn_same_clone, P.air, design, n_perm=19, alternative="less", rng=rng).p_value <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    found = [
        nulls.placebo(nn_same_clone, sim.plant(bg, fraction=0.3, pattern="gc", rng=rng).air, design, n_perm=19,
                      alternative="less", rng=rng).p_value <= 0.05
        for _ in range(5)
    ]  # fmt: skip
    assert np.mean(found) >= 0.8


# ------------------------------------------------------------------------------------------------------- score


def test_score_known_values():
    truth = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    calls = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    s = sim.score(calls, truth).iloc[0]
    assert (s.TP, s.FP, s.FN, s.calls, s.n_true) == (2, 1, 1, 3, 3)
    assert s.precision == pytest.approx(2 / 3) and s.recall == pytest.approx(2 / 3)
    assert s.F1 == pytest.approx(2 / 3) and s.FDR == pytest.approx(1 / 3)
    scores = np.array([0.9, 0.8, 0.1, 0.85, 0.2, 0.3, 0.0, 0.05])
    # ranking: T(0.9) F(0.85) T(0.8) F(0.3) F(0.2) T(0.1): AP = (1/1 + 2/3 + 3/6) / 3
    ap = sim.score({"m": calls}, truth, scores={"m": scores}).AUPRC.iloc[0]
    assert ap == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)
    by = sim.score({"m": calls, "none": np.zeros(8, bool)}, truth, by=np.r_[["a"] * 4, ["b"] * 4])
    assert set(by.columns) >= {"method", "stratum"} and len(by) == 4
    assert np.isnan(by[(by.method == "none") & (by.stratum == "a")].precision.iloc[0])
    with pytest.raises(ValueError, match="rows"):
        sim.score(calls[:3], truth)
