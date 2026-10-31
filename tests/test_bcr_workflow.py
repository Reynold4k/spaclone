import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import bcr, io
from spaclone.bcr.workflow import WorkflowWarning
from spaclone.io import SpatialAIR

MARKERS = {
    "GC B": ["Aicda", "S1pr2", "Rgs13", "Gcsam", "Mef2b", "Bcl6", "Mybl1", "Lmo2"],
    "plasmablast/ASC": ["Jchain", "Xbp1", "Prdm1", "Irf4", "Mzb1"],
    "plasma cell": ["Sdc1", "Tnfrsf17", "Slamf7"],
    "follicular B": ["Fcer2a", "Sell", "Klf2", "Ms4a1"],
    "T": ["Cd3e", "Cd3g", "Cd3d", "Cd4", "Cd8b1"],
    "myeloid": ["Lyz2", "C1qa", "C1qb", "Cd68", "Csf1r"],
}
GENES = [g for v in MARKERS.values() for g in v] + ["Actb", "Malat1", "Ighm", "Ighg1", "Igha", "Igkc"]
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
FAST = {"sigma_um": 30.0, "n_perm": 99, "n_perm_region": 199, "gc_options": {"top_frac": 0.05, "min_units": 10}}


def _section(seed=0, nx=48, ny=32, both=(160.0, 160.0, 50.0), csr_only=(380.0, 160.0, 50.0), null=False):
    """bin20 grid (DNB units) with GEX counts and IGH contigs (+ light chains).

    ``both``: switched-rich and mutation-rich disc with ASC markers (the planted CSR ∩ SHM region); ``csr_only``:
    switched-rich disc with background mutation and GC-B markers. ``null``: no planted signal at all.
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    xd, yd = 1000 + 20 * ii, 2000 + 20 * jj
    xu, yu = 10.0 * ii, 10.0 * jj
    ids = np.array([f"{a}-{b}" for a, b in zip(xd, yd, strict=True)], dtype=object)
    n = len(ids)

    def disc(c):
        if c is None or null:
            return np.zeros(n, bool)
        cs = [c] if np.ndim(c[0]) == 0 else c
        return np.any([(xu - d[0]) ** 2 + (yu - d[1]) ** 2 < d[2] ** 2 for d in cs], axis=0)

    ina, inb = disc(both), disc(csr_only)
    gi = {g: k for k, g in enumerate(GENES)}
    lam = np.full((n, len(GENES)), 0.15)
    lam[:, gi["Actb"]], lam[:, gi["Malat1"]] = 15.0, 10.0
    for g in MARKERS["plasmablast/ASC"]:
        lam[ina, gi[g]] += 3.0
    for g in MARKERS["GC B"]:
        lam[inb, gi[g]] += 2.0
    counts = rng.poisson(lam).astype(np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(counts), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=GENES))
    a.obsm["spatial"] = np.column_stack([xd, yd]).astype(np.int64)
    nrec = rng.poisson(np.where(ina | inb, 2.5, 0.6))
    rb = np.repeat(np.arange(n), nrec)
    m = len(rb)
    hot = ina[rb] | inb[rb]
    u = rng.random(m)
    bg = np.select([u < 0.72, u < 0.88, u < 0.96], ["IGHM*01", "IGHD*01", "IGHG1*01"], "IGHA*01")
    sw = np.select([u < 0.3, u < 0.7, u < 0.8], ["IGHM*01", "IGHG1*01", "IGHG2B*01"], "IGHA*01")
    c_call = np.where(hot, sw, bg)
    mutated = rng.random(m) < np.where(ina[rb], 0.65, 0.05)
    alleles = [f"IGHV1-{k}*01" for k in (26, 31, 53, 64)]
    germ = {al: "".join(rng.choice(list("ACGT"), 260)) for al in alleles}
    v = rng.choice(alleles, m)
    k_mm = np.where(mutated, 3 + rng.poisson(3, m), rng.choice([0, 0, 0, 1], m))
    seq = []
    for al, k in zip(v, k_mm, strict=True):
        s = np.array(list(germ[al]))
        for p in rng.choice(260, size=k, replace=False):
            s[p] = "A" if s[p] != "A" else "C"
        seq.append("".join(s))
    jaa = ["C" + "".join(rng.choice(AA, rng.integers(8, 13))) + "W" for _ in range(m)]
    for r in np.flatnonzero(rng.random(m) < 0.15):  # some clones with several loci
        r2 = rng.integers(m)
        jaa[r], v[r] = jaa[r2], v[r2]
    air = pd.DataFrame(
        {
            "cell_id": ids[rb],
            "locus": "IGH",
            "v_call": v,
            "j_call": "IGHJ4*01",
            "c_call": c_call,
            "junction_aa": jaa,
            "productive": "T",
            "consensus_count": rng.choice([1, 12, 40, 80], m, p=[0.05, 0.1, 0.45, 0.4]),
            "v_sequence_alignment": seq,
            "v_germline_alignment": [germ[al] for al in v],
        }
    )
    light = air.iloc[: min(80, m)].assign(locus="IGK", v_call="IGKV8-19*01", c_call="IGKC*01", junction_aa="CQNDYW")
    return a, pd.concat([air, light], ignore_index=True)


@pytest.fixture(scope="module")
def planted():
    a, air = _section(seed=0)
    return io.read(a, air)


@pytest.fixture(scope="module")
def planted_result(planted):
    return bcr.hotspot_workflow(planted, rng=0, **FAST)


def test_workflow_finds_and_labels_the_planted_region(planted_result):
    res = planted_result
    assert res.csr.summary["n_hotspots"] >= 2 and res.shm.summary["n_hotspots"] >= 1
    assert len(res.regions) == 1
    r = res.regions.iloc[0]
    # the planted CSR ∩ SHM disc is centred at (160, 160) um from the origin (500, 1000) um
    assert abs(r.x_um - 660) < 30 and abs(r.y_um - 1160) < 30
    assert r.switched_share > 0.5 and r.mutated_share > 0.4 and r.shm_excess_ratio > 1.3
    assert r.p < 0.01 and r.csr_clone_p < 0.05 and r.shm_clone_p < 0.05 and r.shm_clone_region_p < 0.05
    assert r.gex_label == "plasmablast/ASC" and r.gex_ratio_window > 3
    # the CSR-only disc is a CSR hotspot that is not a region
    csr_only = res.csr.table[(res.csr.table.x_um - 880).abs() < 40]
    assert len(csr_only) == 1 and int(csr_only.hotspot.iloc[0]) not in [int(h) for h in r.csr_hotspots.split(";")]
    # the GEX-only GC-B call sits on the CSR-only disc (GC-B markers there), not on the ASC region
    gc = res.gc_gex
    u = res.units
    assert gc.sum() > 0
    assert (u.loc[gc.to_numpy(), "x_um"] > 800).mean() > 0.8
    assert res.units.region_id.max() == 1 and res.units.in_window.any()


def test_result_tables_have_the_documented_columns(planted_result):
    res = planted_result
    for c in ("spot_id", "x_um", "y_um", "locus", "clone_id", "isotype", "isotype_class", "switched", "shm_mutations",
              "shm_excess", "mutated", "evaluable", "locus_id"):  # fmt: skip
        assert c in res.records.columns
    assert set(res.records.locus) == {"IGH"}
    assert res.records.isotype.isin(["IgM", "IgD", "IgG1", "IgG2b", "IgA"]).all()
    for c in ("locus_id", "clone_id", "x_um", "y_um", "n_spots", "isotype_class", "switched", "mutated", "evaluable"):
        assert c in res.loci.columns
    for c in ("region_id", "csr_hotspots", "shm_hotspots", "x_um", "y_um", "area_um2", "n_loci", "switched_share",
              "mutated_share", "shm_excess_ratio", "p", "csr_clone_p", "shm_clone_p", "gex_label", "gex_ratio_window",
              "gex_second_label", "gex_second_ratio"):  # fmt: skip
        assert c in res.regions.columns
    assert list(res.windows.columns) == ["region_id", "x0", "x1", "y0", "y1"]
    assert set(res.profiles.region_id) == {1} and "ratio_window" in res.profiles
    assert set(res.module_maps) >= {"GC B", "plasmablast/ASC", "cycling", "switched Ig"}
    assert res.module_maps["plasmablast/ASC"].index.equals(res.units.index)
    assert res.grid.pixel_um == 10.0 and res.tissue.values.dtype == bool
    assert res.region_labels.values.max() == 1 and res.csr_shm_test is not None
    # records and loci agree: a locus's region is the region of its records
    m = res.records.groupby("locus_id").region_id.first()
    assert (res.loci.set_index("locus_id").region_id.loc[m.index] == m).all()
    s = res.summary()
    assert list(s.columns) == ["step", "item", "value", "note"]
    assert s[(s.step == "region 1") & (s.item == "GEX label")].value.iloc[0] == "plasmablast/ASC"
    assert "1: plasmablast/ASC" in repr(res)


def test_to_csv_writes_every_table(planted_result, tmp_path):
    paths = planted_result.to_csv(tmp_path / "out")
    names = {p.name for p in paths}
    assert {"regions.csv", "records.csv", "loci.csv", "profiles.csv", "units.csv", "params.json"} <= names
    assert "csr_shm_clone.csv" in names
    back = pd.read_csv(tmp_path / "out" / "regions.csv")
    assert back.gex_label.iloc[0] == "plasmablast/ASC"
    assert "v_sequence_alignment" not in pd.read_csv(tmp_path / "out" / "records.csv", nrows=2).columns


def test_same_seed_same_result_other_seed_other_null(planted):
    kw = {**FAST, "clone_null": False, "gc_call": False}
    r1 = bcr.hotspot_workflow(planted, rng=7, **kw)
    r2 = bcr.hotspot_workflow(planted, rng=7, **kw)
    pd.testing.assert_frame_equal(r1.regions, r2.regions)
    np.testing.assert_array_equal(r1.csr.null, r2.csr.null)
    pd.testing.assert_frame_equal(r1.csr_shm_test.table, r2.csr_shm_test.table)
    r3 = bcr.hotspot_workflow(planted, rng=8, **kw)
    assert not np.array_equal(r1.csr.null, r3.csr.null)
    assert r1.params["seeds"]["csr"] != r3.params["seeds"]["csr"]


def test_no_overlap_returns_the_hotspots_and_says_why():
    a, air = _section(seed=1, both=None)
    sa = io.read(a, air)
    res = bcr.hotspot_workflow(sa, rng=0, region_source="overlap", **{**FAST, "clone_null": False})
    assert res.csr.summary["n_hotspots"] >= 1
    assert len(res.regions) == 0 and list(res.regions.columns)[:3] == ["region_id", "csr_hotspots", "shm_hotspots"]
    assert res.messages[0].startswith("no CSR ∩ SHM region") and res.messages[0].endswith("returned")
    assert res.profiles.empty and res.windows.empty and len(res.module_maps) > 0
    assert (res.units.region_id == 0).all() and res.params["region_source_used"] is None
    assert "0 CSR∩SHM regions" in repr(res)
    s = res.summary()
    assert s[s.step == "messages"].value.str.startswith("no CSR ∩ SHM").any()


def test_without_an_overlap_region_the_csr_hotspots_are_the_regions():
    """region_source="auto": no CSR ∩ SHM region, so each CSR hotspot is a region with the numbers of a region.

    Null model: a CSR hotspot's SHM numbers are its own tests: mutated calls shuffled within isotype class
    (``p``, ``csr_shm_test``) and whole clones relabelled within clone size and isotype mix with the hotspot fixed
    (``shm_clone_region_p`` / ``shm_clone_q``, ``csr_shm_clone``, BH over the hotspots); ``csr_clone_p`` is the
    hotspot's peak on the clone-relabelled CSR map against the shuffled maps' maxima. References: Hoehn et al. 2022,
    doi:10.1371/journal.pcbi.1009885; Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    a, air = _section(seed=1, both=None)
    res = bcr.hotspot_workflow(io.read(a, air), rng=0, **{**FAST, "gc_call": False})
    ids = [int(h) for h in res.csr.table.hotspot]
    R = res.regions
    assert ids and R.region_id.tolist() == ids and (R.source == "csr").all()
    assert R.csr_hotspots.tolist() == [str(h) for h in ids]
    assert res.params["region_source"] == "auto" and res.params["region_source_used"] == "csr"
    assert res.messages[0].startswith("no CSR ∩ SHM region") and "(region_source='auto')" in res.messages[0]
    disc = R[(R.x_um - 880).abs() < 40].iloc[0]  # the switched disc with GC-B markers and background mutation
    hid = str(int(disc.region_id))
    assert disc.switched_share > 0.5 and disc.gex_label.startswith("GC B")
    assert np.isfinite(disc.csr_clone_p) and np.isfinite(disc.shm_clone_q)
    conf = res.csr_shm_clone.table.set_index(res.csr_shm_clone.table.csr_hotspot.astype(str))
    unit = res.csr_shm_test.table.set_index(res.csr_shm_test.table.region.astype(str))
    assert disc.shm_clone_q == conf.loc[hid, "q_value"] and disc.shm_clone_region_p == conf.loc[hid, "p_value"]
    assert disc.p == unit.loc[hid, "p_value"] and disc.q == unit.loc[hid, "q_value"]
    assert res.windows.region_id.tolist() == ids and set(res.profiles.region_id) == set(ids)
    assert (res.units.region_id > 0).any() and res.region_labels.values.max() == max(ids)
    s = res.summary()
    assert s[(s.step == "regions") & (s.item == "regions from CSR hotspots")].value.iloc[0] == len(ids)
    assert s[(s.step == f"region {hid}") & s.item.str.startswith("p (")].note.iloc[0].startswith("unit-level")
    assert f"{len(ids)} regions from CSR hotspots" in repr(res)


def test_region_source_csr_takes_every_csr_hotspot(planted):
    res = bcr.hotspot_workflow(planted, rng=0, region_source="csr", clone_null=False, gc_call=False, **FAST)
    R = res.regions
    assert (R.source == "csr").all() and R.region_id.tolist() == [int(h) for h in res.csr.table.hotspot]
    assert (R.shm_hotspots != "").sum() >= 1  # the planted CSR ∩ SHM disc keeps its SHM hotspots
    assert res.messages[0].startswith("region_source='csr'") and res.params["region_source_used"] == "csr"


def test_from_files_without_gex_or_alignments(tmp_path):
    a, air = _section(seed=2)
    a.write_h5ad(tmp_path / "g.h5ad")
    air.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    sa = io.read(tmp_path / "g.h5ad", tmp_path / "a.tsv")
    assert sa.meta["um_per_unit"] == 0.5
    res = bcr.hotspot_workflow(sa.air, rng=0, **{**FAST, "clone_null": False})
    assert res.units is None and res.module_maps == {} and len(res.regions) == 1
    assert res.regions.gex_label.isna().all() and any("no GEX" in m for m in res.messages)
    noalign = sa.air.drop(columns=["v_sequence_alignment", "v_germline_alignment"])
    res2 = bcr.hotspot_workflow(SpatialAIR(noalign, sa.gex, {}), rng=0, **{**FAST, "clone_null": False})
    assert res2.shm is None and res2.csr is not None
    assert len(res2.regions) == res2.csr.summary["n_hotspots"] and (res2.regions.source == "csr").all()
    assert any("no V alignments" in m for m in res2.messages)
    assert not res2.records.evaluable.any()


def test_legacy_replays_the_archive_draws(planted):
    res = bcr.hotspot_workflow(planted, rng=0, legacy=True, clone_null=False, gc_call=False, **FAST)
    assert res.csr.params["threshold_rule"] == "quantile" and res.csr.params["neff_kernel"] == "legacy"
    assert res.params["min_hotspot_loci"] == 0 and res.csr.params["min_units"] == 0  # the archive kept every set
    iso = res.loci[res.loci.switched.notna() & res.loci.on_grid].reset_index(drop=True)
    iso["switched"] = iso.switched.astype(bool)
    direct = bcr.csr_hotspots(iso, blocks=None, n_perm=99, rng=np.random.default_rng(0), sigma_um=30.0,
                              tissue=res.tissue, grid=res.grid, neff_kernel="legacy", threshold_rule="quantile")  # fmt: skip
    np.testing.assert_array_equal(direct.labels.values, res.csr.labels.values)
    assert direct.threshold["hot"] == res.csr.threshold["hot"]
    with pytest.raises(ValueError, match="integer seed"):
        bcr.hotspot_workflow(planted, rng=np.random.default_rng(0), legacy=True)


def test_isotype_matched_regions_and_pixel_overlap(planted):
    res = bcr.hotspot_workflow(planted, rng=0, shm_from="isotype", overlap="pixel", clone_null=False, **FAST)
    assert res.params["shm_from"] == "isotype" and len(res.regions) == 1
    assert res.regions.gex_label.iloc[0] == "plasmablast/ASC"


def test_two_regions_are_numbered_along_the_long_axis():
    a, air = _section(seed=3, nx=60, ny=24, both=[(470.0, 120.0, 45.0), (120.0, 120.0, 45.0)], csr_only=None)
    res = bcr.hotspot_workflow(io.read(a, air), rng=0, clone_null=False, **FAST)
    assert len(res.regions) == 2
    assert res.regions.region_id.tolist() == [1, 2] and res.regions.x_um.is_monotonic_increasing
    assert (res.regions.gex_label == "plasmablast/ASC").all()


def test_score_units_restricts_the_module_maps_only(planted, planted_result):
    assert planted_result.score_mask.all() and planted_result.score_mask.index.equals(planted.gex.obs_names)
    g = planted.gex.copy()
    g.obs["kind"] = np.where(g.obsm["spatial"][:, 0] < 1200, "B_cells", "other")  # DNB x < 1200: the left part
    b = g.obs["kind"].eq("B_cells").to_numpy()
    sa = SpatialAIR(planted.air, g, {})
    kw = {**FAST, "clone_null": False, "gc_call": False}
    res = bcr.hotspot_workflow(sa, rng=0, score_units=("kind", ["B_cells"]), **kw)
    np.testing.assert_array_equal(res.score_mask.to_numpy(), b)
    assert res.score_mask.index.equals(g.obs_names) and (res.units.scored == res.score_mask).all()
    m = res.module_maps["plasmablast/ASC"]
    assert m[b].notna().all() and m[~b].isna().all()
    assert res.params["score_units"] == ("kind", ["B_cells"]) and res.params["n_scored_units"] == int(b.sum())
    # the regions, their tests and their GEX labels do not depend on which units are scored for the maps
    cols = ["region_id", "csr_hotspots", "switched_share", "mutated_share", "gex_label", "gex_ratio_window"]
    pd.testing.assert_frame_equal(res.regions[cols], planted_result.regions[cols])
    g.obs["is_b"] = b
    r2 = bcr.hotspot_workflow(SpatialAIR(planted.air, g, {}), rng=0, score_units="is_b", **kw)
    r3 = bcr.hotspot_workflow(SpatialAIR(planted.air, g, {}), rng=0, score_units=b, **kw)
    pd.testing.assert_series_equal(r2.score_mask, res.score_mask)
    pd.testing.assert_series_equal(r3.score_mask, res.score_mask)
    with pytest.raises(KeyError, match="annotation"):
        bcr.hotspot_workflow(sa, rng=0, score_units=("annotation", ["B"]), **kw)
    with pytest.raises(ValueError, match="not boolean"):
        bcr.hotspot_workflow(sa, rng=0, score_units="kind", **kw)
    no_gex = bcr.hotspot_workflow(planted.air, rng=0, score_units=("kind", ["B_cells"]), **kw)
    assert no_gex.score_mask is None


def test_input_errors(planted):
    tcr = planted.air.assign(locus="TRB")
    with pytest.raises(ValueError, match=r"no heavy-chain.*IGH \(heavy chain\) 0; min_reads=2 \(auto"):
        bcr.hotspot_workflow(SpatialAIR(tcr, planted.gex, {}), **FAST)
    with pytest.raises(ValueError, match="shm_options"):
        bcr.hotspot_workflow(planted, shm_options={"key_added": "x"}, **FAST)
    with pytest.raises(ValueError, match="min_reads must be"):
        bcr.hotspot_workflow(planted, min_reads="two", **FAST)
    two = pd.concat([planted.air.assign(library_id="a"), planted.air.assign(library_id="b")])
    with pytest.raises(ValueError, match="several sections"):
        bcr.hotspot_workflow(two, **FAST)
    with pytest.raises(ValueError, match="shm_from"):
        bcr.hotspot_workflow(planted, shm_from="both")
    with pytest.raises(ValueError, match="region_source"):
        bcr.hotspot_workflow(planted, region_source="both")
    with pytest.raises(ValueError, match="min_hotspot_loci"):
        bcr.hotspot_workflow(planted, min_hotspot_loci=-1)
    with pytest.raises(TypeError, match="SpatialAIR"):
        bcr.hotspot_workflow("file.tsv")


def _plain(value, where="attrs"):
    """Assert that ``value`` holds only scalars and plain containers (no DataFrame, Series or array)."""
    if isinstance(value, dict):
        for k, v in value.items():
            _plain(v, f"{where}[{k!r}]")
    elif isinstance(value, list | tuple | set | frozenset):
        for k, v in enumerate(value):
            _plain(v, f"{where}[{k}]")
    else:
        assert isinstance(value, type(None) | bool | int | float | str | np.generic), f"{where}: {type(value)}"


def test_tables_concatenate_and_merge_and_the_result_keeps_its_gex(planted, planted_result):
    """pandas >= 3 compares ``attrs`` in concat / merge; the records carried the SHM baseline table there."""
    res = planted_result
    for tab in (res.records, res.loci, res.regions, res.qc):
        _plain(tab.attrs)
        assert len(pd.concat([tab, tab])) == 2 * len(tab)
        assert len(tab.merge(tab, left_index=True, right_index=True, suffixes=("", "_r"))) == len(tab)
    assert len(res.records.merge(res.loci[["locus_id", "n_spots"]], on="locus_id")) == len(res.records)
    assert isinstance(res.params["shm_baseline"], pd.DataFrame) and "IGHV1-26*01" in res.params["shm_baseline"].index
    assert res.gex is planted.gex and "gex" not in repr(res)
    back = pickle.loads(pickle.dumps(res))
    assert back.gex is None and back.csr_shm_clone is not None
    pd.testing.assert_frame_equal(back.regions, res.regions)


def test_confirmatory_clone_level_shm_test_of_the_csr_hotspots(planted_result):
    """Null model: whole clones relabelled within clone size and isotype mix, the CSR hotspots (found from the switched
    calls alone) fixed; BH over the hotspots. References: Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885;
    Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x."""
    res = planted_result
    t = res.csr_shm_clone.table
    assert "region" not in t.columns  # its ids are CSR hotspots, not the CSR ∩ SHM regions
    per = t[t.csr_hotspot.astype(str) != "all"].set_index("csr_hotspot")
    assert {"marked", "null_mean", "p_value", "q_value", "n_perm"} <= set(t.columns) and (t.n_perm == 199).all()
    both = int(res.regions.csr_hotspots.iloc[0].split(";")[0])
    csr_only = res.csr.table[(res.csr.table.x_um - 880).abs() < 40].hotspot.iloc[0]
    assert per.loc[both, "q_value"] < 0.05 and per.loc[both, "p_value"] <= per.loc[both, "q_value"]
    assert per.loc[csr_only, "p_value"] > 0.05  # switched, but mutation at background
    assert res.params["seeds"]["csr_shm_clone"] and "confirmatory" in res.params["null_designs"]["csr_shm_clone"]
    s = res.summary()
    conf = s[s.step == "CSR hotspots: SHM, clone-level (confirmatory)"]
    assert conf.item.str.startswith(f"CSR hotspot {both}: q").any() and (conf.item == "null").any()
    desc = s[s.item == "clone-level p, SHM excess in region (descriptive)"]
    assert len(desc) == 1 and desc.note.iloc[0].startswith("descriptive")
    assert s[(s.step == "region 1") & s.item.str.startswith("p (")].note.iloc[0].startswith("descriptive")
    unit = s[(s.item == "hotspots") & s.step.isin(["CSR", "SHM raw", "SHM isotype-matched"])]
    assert len(unit) == 3 and unit.note.str.contains("loci shuffled one by one").all()
    assert unit.note.str.contains("max-T over the map").all() and unit.note.iloc[2].count("within isotype") == 1


def test_min_reads_auto_follows_the_record_level(planted, planted_result):
    assert planted.meta["level"] == "contig" and planted_result.params["min_reads"] == 2
    assert planted_result.params["shm_min_reads"] == 30 and planted_result.params["shm_min_aligned_nt"] == 250
    kw = {**FAST, "clone_null": False, "gc_call": False}
    mol = SpatialAIR(planted.air, planted.gex, {**planted.meta, "level": "molecule"})
    res = bcr.hotspot_workflow(mol, rng=0, **kw)
    assert res.params["min_reads"] == 1 and res.params["shm_min_reads"] == 1
    assert "reads >= 1" in set(res.qc.step) and res.params["record_level"] == "molecule"
    assert len(res.records) > len(planted_result.records)  # the 1-read records are kept
    # a floor that removes most records says so, and an empty result names min_reads and the counts
    one = planted.air.assign(consensus_count=np.where(np.arange(len(planted.air)) % 5 == 0, 40, 1))
    with pytest.warns(WorkflowWarning, match=r"min_reads=2 removed .* \(80%; median 1 reads per record\)"):
        r2 = bcr.hotspot_workflow(SpatialAIR(one, planted.gex, planted.meta), rng=0, **kw)
    assert any("min_reads=2 removed" in m for m in r2.messages)
    with pytest.warns(WorkflowWarning), pytest.raises(ValueError, match=r"reads >= 2 0; IGH \(heavy chain\) 0"):
        bcr.hotspot_workflow(SpatialAIR(planted.air.assign(consensus_count=1), planted.gex, planted.meta), **kw)


def test_shm_from_mixcr_alignments(planted):
    """AIR-SPACE style input: MiXCR V alignments (150 nt), one molecule per record."""
    air = planted.air.copy()
    al, mm = bcr.count_mismatches(air.v_sequence_alignment, air.v_germline_alignment)
    air["v_alignment_mixcr"] = [
        "100|250|300|0|150|" + "".join(f"SA{100 + 7 * k}G" for k in range(m)) + "|500.0" for m in mm
    ]
    air = air.drop(columns=["v_sequence_alignment", "v_germline_alignment"])
    sa = SpatialAIR(air, planted.gex, {**planted.meta, "level": "molecule"})
    res = bcr.hotspot_workflow(sa, rng=0, **{**FAST, "clone_null": False, "gc_call": False})
    p = res.params
    assert p["shm_alignments"] == "MiXCR v_alignment_mixcr" and p["shm_min_aligned_nt"] == 100
    assert p["shm_min_reads"] == 1 and p["shm_baseline_fallback"] == "zero"
    assert res.records.evaluable.mean() > 0.95 and (res.records.shm_aligned_nt == 150).all()
    np.testing.assert_array_equal(res.records.shm_mutations, res.records.shm_mutations.astype(int))
    assert res.shm is not None and len(res.regions) == 1
    assert res.regions.mutated_share.iloc[0] > 0.4 and res.regions.gex_label.iloc[0] == "plasmablast/ASC"
    strict = bcr.hotspot_workflow(sa, rng=0, shm_min_aligned_nt=250, **{**FAST, "clone_null": False, "gc_call": False})
    assert not strict.records.evaluable.any() and strict.shm is None


def test_species_mismatch_warns_and_names_species(planted):
    g = planted.gex.copy()
    g.var_names = [v.upper() for v in g.var_names]  # human-style gene names, mouse marker sets
    with pytest.warns(WorkflowWarning, match=r"1 of 1 region GEX labels are 'undetermined'.*species='human'"):
        res = bcr.hotspot_workflow(SpatialAIR(planted.air, g, planted.meta), rng=0, clone_null=False, gc_call=False,
                                   **FAST)  # fmt: skip
    assert res.regions.gex_label.iloc[0] == "undetermined" and any("species='human'" in m for m in res.messages)


def test_min_hotspot_loci_drops_hotspots_and_says_so(planted, planted_result):
    """Hotspots with fewer loci of their map than ``min_hotspot_loci`` are dropped, counted and named in messages."""
    assert planted_result.params["min_hotspot_loci"] == 1 and planted_result.csr.params["min_units"] == 1
    assert (planted_result.csr.table.units >= 1).all()
    kw = {**FAST, "clone_null": False, "gc_call": False}
    base = bcr.hotspot_workflow(planted, rng=0, min_hotspot_loci=0, **kw)
    n = int(base.csr.table.units.min())
    res = bcr.hotspot_workflow(planted, rng=0, min_hotspot_loci=n + 1, **kw)
    gone = int((base.csr.table.units <= n).sum())
    assert res.csr.summary["n_hotspots"] == base.csr.summary["n_hotspots"] - gone >= 0
    assert res.params["hotspots_dropped_min_loci"]["csr"] == gone and res.params["min_hotspot_loci"] == n + 1
    assert any(m.startswith(f"{gone} CSR hotspot") and m.endswith(f"dropped (min_hotspot_loci={n + 1})")
               for m in res.messages)  # fmt: skip


@pytest.mark.slow
def test_null_sections_rarely_give_a_region():
    """Calibration: without planted signal a CSR hotspot appears on at most alpha + tolerance of the sections."""
    n_rep, any_csr, any_region = 60, 0, 0
    for seed in range(n_rep):
        a, air = _section(seed=100 + seed, null=True)
        res = bcr.hotspot_workflow(io.read(a, air), rng=seed, clone_null=False, gc_call=False, **FAST)
        any_csr += res.csr.summary["n_hotspots"] > 0
        any_region += len(res.regions) > 0
    tol = 3.1 * np.sqrt(0.05 * 0.95 / n_rep)
    assert any_csr / n_rep <= 0.05 + tol
    assert any_region / n_rep <= 0.05 + tol
