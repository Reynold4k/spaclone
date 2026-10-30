import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import spaclone as spc
from spaclone._run import RunWarning, _n_perm, _step_rngs
from spaclone._utils import TestResult
from spaclone.qc import CRITERIA, OVERALL_RULES, TrustReport, evaluate_trust
from spaclone.qc.same_spot import SameSpotResult

CHIP = {"size_um": 440.0, "tissue_radius_um": 200.0}  # a smaller section: every structure, ~65% of the cells
COLS = ["test", "library_id", "locus", "p_value"]


def _chip(gf, seed, **kw):
    return spc.sim.synthetic_chip(genuine_fraction=gf, rng=seed, **{**CHIP, **kw})


@pytest.fixture(scope="module")
def chips():
    # homologous_switch: half of the swaps keep the acceptor's constant region and barcode and the donor's V(D)J
    # (template switching inside the shared constant region): right transcript and cell type, wrong clone
    return {"planted": _chip(0.3, 0), "swapped": _chip(0.0, 1), "homologous": _chip(0.0, 2, homologous_switch=0.5)}


@pytest.fixture(scope="module")
def reports(chips):
    return {k: spc.run(v, n_perm=99, rng=0) for k, v in chips.items()}


# --------------------------------------------------------------------------------------------- verdicts


def test_planted_positions_support_clone_maps(reports):
    rep = reports["planted"]
    assert isinstance(rep, TrustReport)
    assert rep.verdict.startswith("Clone positions supported: clone identity follows position")
    assert rep.status == {"overall": "clone_positions", "molecule": "signal", "clone": "signal"}
    assert rep.clone_positions_supported
    assert "Molecule level: signal" in rep.verdict  # the molecule level is stated too
    t = rep.tests()
    assert (t.loc[t.test == "same_spot", "verdict"] == "signal").any()  # the small TCR loci may lack power
    assert set(t.test) >= {"same_spot", "own_type_share", "colocalisation", "pairing", "partner_sharing"}


def test_swapped_data_never_license_clone_maps(reports):
    """Every molecule swapped: no clone-level signal (molecule-level calibration is the job of the qc tests)."""
    rep = reports["swapped"]
    assert rep.status["overall"] != "clone_positions" and rep.status["clone"] != "signal"
    assert not rep.clone_positions_supported
    t = rep.tests()
    assert not ((t.verdict == "signal") & (t.level == "clone")).any()
    if rep.status["overall"] == "no_evidence":
        assert rep.verdict.startswith("No evidence of positional signal at the power of these data")
        assert "Power:" in rep.verdict and "testable row" in rep.verdict


def test_homologous_switching_is_cell_type_placement_only(reports):
    """Molecule-level tests see molecules on their own transcript / type; clone-level tests see no clone placement."""
    rep = reports["homologous"]
    assert rep.status == {"overall": "cell_type_only", "molecule": "signal", "clone": "not detected"}
    assert rep.verdict.startswith("Cell-type placement only: molecules sit on their own cell type / transcript")
    assert "do not use clone maps" in rep.verdict


def test_light_chains_are_information_only(reports):
    t = reports["planted"].tests()
    co = t[t.test == "colocalisation"]
    light = co[~co.locus.isin(["IGH", "TRB", "TRD"])]
    assert len(light) and (light.verdict == "info").all() and (light.family == "").all()
    assert light.q_value.isna().all()
    heavy = co[co.locus.isin(["IGH", "TRB", "TRD"])]
    assert (heavy.family.isin(["clone", ""])).all() and not heavy.note.str.contains("light chain").any()


def test_bh_runs_within_levels_over_testable_rows_only(reports):
    t = reports["swapped"].tests()
    fam = t[t.family != ""]
    assert set(fam.family) <= {"molecule", "clone"}
    assert fam.q_value.notna().all() and (fam.q_value >= fam.p_value - 1e-12).all()
    assert t.loc[t.verdict == "untestable", "q_value"].isna().all()
    assert (t.loc[t.family == "molecule", "test"].isin(["same_spot", "own_type_share", "umi_match"])).all()
    assert (t.loc[t.family == "clone", "test"].isin(["colocalisation", "pairing", "partner_sharing"])).all()


def test_permutations_reach_the_bh_floor(reports):
    rep = reports["planted"]
    used, m = rep.params["n_perm_used"], rep.params["family_planned"]
    assert used["clone"] >= 99 and 1 / (used["clone"] + 1) <= 0.05 / max(m["clone"], 1) + 1e-12
    assert 1 / (used["molecule"] + 1) <= 0.05 / max(m["molecule"], 1) + 1e-12
    assert 1 / (used["partner_sharing"] + 1) <= 0.05 / (3 * max(m["clone"], 1)) + 1e-12
    t = rep.tests()
    assert (t.loc[t.test == "pairing", "n_perm"] == used["clone"]).all()
    assert _n_perm(200, 60) == 1199 and _n_perm(200, 5) == 200 and _n_perm(10, 5, k=3) == 299


def test_one_positive_library_survives_many_null_libraries():
    """B5: with the permutation floor below alpha/m, one genuine library among null ones is still detected."""
    parts = [_chip(0.3 if k == 0 else 0.0, 50 + k, library_id=f"lib{k}", tcr=False).air for k in range(4)]
    rep = spc.run(pd.concat(parts, ignore_index=True), n_perm=49, rng=0, steps=("filter", "barcode_sinks", "pairing"))
    t = rep.tests()
    pr = t[(t.test == "pairing") & (t.verdict != "not run")]
    assert set(pr.library_id) == {f"lib{k}" for k in range(4)}
    b = rep.params["n_perm_used"]["clone"]
    assert 1 / (b + 1) <= 0.05 / rep.params["family_planned"]["clone"] + 1e-12
    assert pr.loc[(pr.library_id == "lib0") & (pr.locus == "IGH/IGK"), "verdict"].item() == "signal"
    assert rep.status["clone"] == "signal"


# --------------------------------------------------------------------------------------------- input checks


def test_mismatched_gex_ids_drop_strata_and_types(chips):
    sa = chips["swapped"]
    gex = sa.gex.copy()
    gex.obs_names = "chip1_" + gex.obs_names
    with pytest.warns(RunWarning, match="spot ids"):
        rep = spc.run(sa.air, gex, n_perm=19, rng=0, steps=("colocalisation", "own_type_share"))
    assert rep.params["depth_controlled"] is False and rep.params["id_overlap"] == 0
    t = rep.tests()
    co = t[(t.test == "colocalisation") & t.locus.isin(["IGH", "TRB"])]
    assert len(co) and not co.note.str.contains("stratified").any()
    assert (co.verdict.isin(["inconclusive", "untestable"])).all()
    assert co.loc[co.verdict == "inconclusive", "note"].str.contains("depth not controlled").all()
    own = t[t.test == "own_type_share"]
    assert (own.verdict == "not run").all() and own.note.str.contains("do not match").all()
    assert any("spot ids" in n for n in rep.params["notes"])
    assert "Input checks" in rep.to_markdown()


def test_without_gex_colocalisation_is_not_decisive(chips):
    rep = spc.run(chips["planted"].air, n_perm=19, rng=0, steps=("barcode_sinks", "colocalisation"))
    t = rep.tests()
    heavy = t[(t.test == "colocalisation") & (t.locus == "IGH")]
    assert heavy.verdict.item() == "inconclusive" and "depth not controlled" in heavy.note.item()
    assert rep.status["clone"] == "limited" and rep.status["overall"] == "limited"
    assert "Mixed" not in rep.verdict


def test_beads_skip_same_spot_with_a_note():
    rng = np.random.default_rng(0)
    n = 400
    xy = rng.uniform(0, 300, size=(n, 2))  # bead centres: no regular grid
    obs = pd.DataFrame({"in_tissue": True}, index=[f"b{i}" for i in range(n)])
    x = rng.poisson(2.0, size=(n, 3)).astype(np.float32)
    gex = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=["Igkc", "Ighm", "Actb"]))
    gex.obsm["spatial"] = xy
    pick = rng.integers(0, n, 300)
    air = pd.DataFrame(
        {
            "spot_id": obs.index.to_numpy()[pick],
            "x_um": xy[pick, 0],
            "y_um": xy[pick, 1],
            "locus": rng.choice(["IGH", "IGK"], 300),
            "clone_id": rng.choice([f"c{i}" for i in range(30)], 300),
        }
    )
    sa = spc.io.SpatialAIR(air, gex, {"molecule_level": True, "um_per_unit": 0.65})
    rep = spc.run(sa, n_perm=19, rng=0, steps=("same_spot",))
    note = rep.tests().set_index("test").loc["same_spot", "note"]
    assert "grid" in note and "beads" in note


def test_same_spot_at_ceiling_is_untestable():
    tab = pd.DataFrame(
        {"locus": ["IGK"], "zone": ["tissue"], "n_spots": [500], "hits": [480], "statistic": [480],
         "expected": [470.0], "null_sd": [5.0], "ratio": [480 / 470], "p_value": [0.04], "rho_pdet_hi": [0.2],
         "deficit": [False]}
    )  # fmt: skip
    rec = pd.DataFrame(
        {"locus": "IGK", "zone": "tissue", "chance": np.full(500, 0.94), "tested": True, "occupied": True,
         "singleton": True}
    )  # fmt: skip
    ss = SameSpotResult(name="same_spot", table=tab, params={"singleton_only": True}, records=rec)
    table, v = evaluate_trust({"same_spot": ss})
    row = table[table.test == "same_spot"].iloc[0]
    assert row.verdict == "untestable" and "ceiling" in row.note and np.isnan(row.q_value)


def _pairing_result(p, stat=2.0):
    tab = pd.DataFrame(
        {"library_id": ["L"], "heavy_locus": ["IGH"], "light_locus": ["IGK"], "design": ["stratified"],
         "n_heavy_units": [500], "n_heavy": [5 if np.isfinite(p) else 1], "n_light": [5], "statistic": [stat],
         "null_mean": [1.0], "p_value": [p], "n_perm": [199]}
    )  # fmt: skip
    return TestResult(name="pairing", table=tab, params={"min_units": 50}, null=np.linspace(0.5, 1.5, 199)[:, None])


def test_nan_p_values_are_untestable_not_no_signal():
    """M1: an untested combination is never counted as evidence of absence."""
    table, v = evaluate_trust({"pairing_BCR": _pairing_result(np.nan, stat=np.nan)})
    row = table[table.test == "pairing"].iloc[0]
    assert row.verdict == "untestable" and row.family == "" and np.isnan(row.q_value)
    assert v["clone"] == "limited" and v["overall"] == "limited"
    assert v["verdict"].startswith("Limited evidence")


def test_a_single_test_gives_limited_evidence_never_mixed():
    table, v = evaluate_trust({"pairing_BCR": _pairing_result(0.4, stat=1.1)})
    assert table.loc[table.test == "pairing", "verdict"].item() == "not detected"
    assert v["overall"] == "limited" and "only 1 test" in v["verdict"] and "Mixed" not in v["verdict"]
    table, v = evaluate_trust({"pairing_BCR": _pairing_result(0.005)})
    assert v["overall"] == "clone_positions" and v["clone"] == "signal"


def test_mostly_untestable_rows_give_limited_evidence():
    """Two tests with testable rows, but most decisive rows untestable: never "No evidence"."""
    pr = _pairing_result(0.4, stat=1.1)
    nan_rows = pr.table.assign(statistic=np.nan, p_value=np.nan, n_heavy=1)
    tab = pd.concat([pr.table, nan_rows, nan_rows, nan_rows], ignore_index=True)
    pairing = TestResult(name="pairing", table=tab, params={"min_units": 50}, null=np.tile(pr.null, (1, 4)))
    co = pd.DataFrame(
        {"library_id": ["L"], "locus": ["IGH"], "design": ["stratified"], "lo_um": [20.0], "hi_um": [100.0],
         "n_units": [400], "n_clones_in_window": [30], "statistic": [0.11], "null_mean": [0.1], "ratio": [1.1],
         "p_value": [0.3], "n_perm": [199], "top_clone_share": [0.2]}
    )  # fmt: skip
    coloc = TestResult(name="colocalisation", table=co, params={"units": "loci"}, null=pr.null)
    table, v = evaluate_trust({"pairing_BCR": pairing, "colocalisation": coloc}, depth_controlled=True)
    assert (table.verdict == "not detected").sum() == 2 and (table.verdict == "untestable").sum() == 3
    assert v["clone"] == "limited" and v["overall"] == "limited"
    assert "2 of 5 decisive rows" in v["verdict"]


# --------------------------------------------------------------------------------------------- determinism and API


def test_steps_do_not_change_each_others_results(chips, reports):
    full = reports["planted"].tests()
    sub = spc.run(chips["planted"], n_perm=99, rng=0, steps=("pairing",)).tests()
    a = full[full.test == "pairing"][COLS].reset_index(drop=True)
    b = sub[sub.test == "pairing"][COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    g = _step_rngs(5)
    assert g("pairing:BCR").integers(1 << 30) == _step_rngs(5)("pairing:BCR").integers(1 << 30)
    assert g("pairing:BCR").integers(1 << 30) != g("pairing:TCR").integers(1 << 30)


def test_same_seed_same_report(chips, reports):
    again = spc.run(chips["swapped"], n_perm=99, rng=0)
    pd.testing.assert_frame_equal(again.table, reports["swapped"].table)


def test_tcr_pairing_and_options(chips):
    air = chips["planted"].air
    tcr = air[air.locus.isin(["TRA", "TRB"])]
    rep = spc.run(tcr, n_perm=19, rng=0, steps=("pairing",), options={"pairing": {"min_units": 3}})
    assert "pairing_TCR" in rep.results and rep.results["pairing_TCR"].params["min_units"] == 3
    assert rep.tests().loc[rep.tests().test == "pairing", "locus"].eq("TRB/TRA").all()
    with pytest.raises(ValueError, match="unknown steps"):
        spc.run(air, options={"bogus": {}})
    with pytest.raises(ValueError, match="rng"):
        spc.run(air, options={"pairing": {"rng": 1}})


def test_on_tissue_filter(chips):
    air = chips["planted"].air
    rep = spc.run(air, n_perm=19, rng=0, steps=("filter",), on_tissue=True)
    assert rep.params["n_records"] == int(spc.pp.filter_records(air, on_tissue=True).air.shape[0]) and rep.params[
        "n_records"
    ] < len(air)


def test_run_skips_what_does_not_apply(chips):
    rep = spc.run(chips["swapped"].air, n_perm=19, rng=1, steps=("filter", "same_spot", "own_type_share"))
    notes = dict(zip(rep.tests().test, rep.tests().note, strict=False))
    assert "GEX" in notes["same_spot"] and "cell types" in notes["own_type_share"]
    assert rep.status["overall"] == "limited"


def test_run_rejects_unknown_steps(chips):
    with pytest.raises(ValueError, match="unknown steps"):
        spc.run(chips["swapped"], steps=("bogus",))


# --------------------------------------------------------------------------------------------- cell types


def _relabel(sa, rename=None, **extra):
    """The chip with its GEX obs columns renamed / added (the receptor table is shared)."""
    g = sa.gex.copy()
    g.obs = g.obs.rename(columns=rename or {})
    for k, v in extra.items():
        g.obs[k] = v
    return spc.io.SpatialAIR(sa.air, g, dict(sa.meta))


OWN = ("filter", "own_type_share")


def test_auto_cell_type_key_uses_the_only_standard_label_column(chips):
    d = _relabel(chips["planted"], {"cell_type": "cell2loc_anno"})
    d.gex.obsm["q05_cell_abundance_w_sf"] = np.ones((d.gex.n_obs, 3))  # proportion tables never count as labels
    auto = spc.run(d, n_perm=19, rng=0, steps=OWN)
    given = spc.run(d, n_perm=19, rng=0, steps=OWN, cell_type_key="cell2loc_anno")
    ct = auto.params["cell_types"]
    assert ct["key"] == "cell2loc_anno" and "only standard cell-type label column" in ct["key_rule"]
    assert ct["own_types"]["BCR"]["labels"] == ["B", "GC B", "plasma"] and ct["own_types"]["TCR"]["labels"] == ["T"]
    num = ["observed", "chance", "p_value", "n"]
    pd.testing.assert_frame_equal(auto.tests()[num], given.tests()[num])  # the same test as naming the column
    notes = auto.tests().set_index("locus").note
    assert "cell types obs['cell2loc_anno']; own BCR types: B, GC B, plasma" in notes["BCR"]
    assert "gex.obs['cell2loc_anno'] (cell_type_key='auto'" in auto.methods()
    assert any("cell types: obs['cell2loc_anno']" in n for n in auto.params["notes"])


def test_auto_cell_type_key_skips_when_not_certain(chips):
    labels = chips["planted"].gex.obs.cell_type.astype(str)
    two = _relabel(chips["planted"], {"cell_type": "cell2loc_anno"}, annotation=labels)
    rep = spc.run(two, n_perm=19, rng=0, steps=OWN)
    note = dict(zip(rep.tests().test, rep.tests().note, strict=False))["own_type_share"]
    assert "2 standard cell-type label columns ('cell2loc_anno', 'annotation')" in note and "cell_type_key=" in note
    none = _relabel(chips["planted"], {"cell_type": "leiden"})
    none.gex.obsm["q05_cell_abundance_w_sf"] = np.ones((none.gex.n_obs, 3))
    note = dict(zip(*spc.run(none, n_perm=19, rng=0, steps=OWN).tests()[["test", "note"]].T.values, strict=False))
    assert "no cell-type label column" in note["own_type_share"] and "q05" not in note["own_type_share"]
    assert "cluster columns are not cell types: 'leiden'" in note["own_type_share"]


def test_explicit_missing_cell_type_key_raises(chips):
    with pytest.raises(
        ValueError, match=r"'celltype' is not a column of gex.obs \(standard label columns: 'cell_type'"
    ):
        spc.run(chips["planted"], cell_type_key="celltype", steps=OWN)


def test_own_type_vocabulary_matches_whole_words_only():
    from spaclone._run import _own_types, _vocabulary_hit

    for lab in ("B", "B cells", "B_cells", "GC B", "Germinal centre B", "Memory B cells", "naive B", "Follicular B",
                "Plasma cells", "plasmablasts", "ASC", "Antibody-secreting cells", "Bcells", "GCB"):  # fmt: skip
        assert _vocabulary_hit(lab, "BCR") and not _vocabulary_hit(lab, "TCR"), lab
    for lab in ("T", "T_cells", "CD4+ T", "CD8 T cells", "Treg", "Tfh", "NKT", "gamma-delta T", "Tcell", "CD8T"):
        assert _vocabulary_hit(lab, "TCR") and not _vocabulary_hit(lab, "BCR"), lab
    for lab in (
        "Plasmacytoid DC",
        "Fibroblast",
        "NK cells",
        "Monocytes_Macrophages",
        "Erythrocyte",
        "Bcellular",
        "other",
    ):
        assert not _vocabulary_hit(lab, "BCR") and not _vocabulary_hit(lab, "TCR"), lab
    own = _own_types(pd.Series(["B_cells", "T_cells", "B/T doublet", "NK_cells"]), "auto", "cell2loc_anno")
    assert own["BCR"]["labels"] == ["B_cells"] and own["TCR"]["labels"] == ["T_cells"]
    assert "left out as both B and T: 'B/T doublet'" in own["BCR"]["rule"]
    given = _own_types(pd.Series(["Bc", "Tc"]), {"BCR": ["Bc"]}, "x")  # one receptor given, the other stays auto
    assert given["BCR"]["labels"] == ["Bc"] and given["TCR"]["matcher"] is None
    with pytest.raises(ValueError, match="'BCR' or 'TCR'"):
        _own_types(None, {"IG": ["B"]}, None)
    assert _own_types(None, "auto", None)["BCR"] == {"matcher": None, "labels": [], "rule": "no cell-type labels",
                                                      "reason": "no cell-type labels"}  # fmt: skip


def test_receptor_without_matching_categories_is_skipped_with_the_categories(chips):
    d = _relabel(chips["planted"])
    d.gex.obs["cell_type"] = d.gex.obs.cell_type.astype(str).replace({"B": "Bc", "GC B": "GCBc", "plasma": "PC"})
    rep = spc.run(d, n_perm=19, rng=0, steps=OWN)
    t = rep.tests().set_index("locus")
    assert t.loc["BCR", "verdict"] == "not run" and t.loc["TCR", "verdict"] != "not run"
    assert "no category of obs['cell_type'] matches the B-lineage vocabulary" in t.loc["BCR", "note"]
    assert "'Bc'" in t.loc["BCR", "note"] and "own_types={'BCR': [labels]}" in t.loc["BCR", "note"]
    rep2 = spc.run(d, n_perm=19, rng=0, steps=OWN, own_types={"BCR": ["Bc", "GCBc", "PC"]})
    assert rep2.tests().set_index("locus").loc["BCR", "verdict"] != "not run"


def test_species_is_checked(chips):
    from spaclone.io.gexio import _species_of_genes

    with pytest.raises(ValueError, match=r"species must be 'mouse' or 'human', got 'mm10'"):
        spc.run(chips["planted"], species="mm10", steps=("qc",))
    assert _species_of_genes([f"CD{i}" for i in range(60)])[0] == "human"
    assert _species_of_genes([f"Cd{i}" for i in range(60)])[0] == "mouse"
    assert _species_of_genes([f"ENSMUSG{i:011d}" for i in range(60)])[0] == "mouse"
    assert _species_of_genes(["Cd19", "CD19"] * 30)[0] is None  # mixed: no decision
    assert _species_of_genes([f"CD{i}" for i in range(20)])[0] is None  # too few genes
    d = chips["planted"]
    g = ad.AnnData(np.zeros((d.gex.n_obs, 60), np.float32), obs=d.gex.obs.copy())
    g.obsm["spatial"] = d.gex.obsm["spatial"]
    g.var_names = [f"CD{i}" for i in range(60)]
    with pytest.warns(RunWarning, match=r"species='mouse' but the GEX gene names look human .* pass species='human'"):
        rep = spc.run(d.air, g, steps=("filter",))
    assert rep.params["species_detected"] == "human"
    spc.run(d.air, g, species="human", steps=("filter",))  # no warning when they agree


def test_no_record_passes_the_filters(chips):
    with pytest.raises(ValueError, match=r"no record passed the filters \(min_reads=1000000\.0, .*removed: .*min"):
        spc.run(chips["planted"], min_reads=1e6, steps=("filter",))


# --------------------------------------------------------------------------------------------- outputs


def test_report_outputs(reports, tmp_path):
    rep = reports["homologous"]
    md = rep.to_markdown(tmp_path / "r.md")
    for s in ("**Verdict.**", "**Molecule-level placement.**", "**Clone-level placement.**", "**Power.**"):
        assert s in md
    assert all(k in md for k in CRITERIA) and all(r in md for r in OVERALL_RULES)
    assert "Monte-Carlo p-values" in rep.methods() and "light chains" in rep.methods()
    fig = rep.plot()
    assert len(fig.axes) == 1
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "Molecule-level placement" in texts and "Clone-level placement" in texts
    assert "TrustReport(cell_type_only" in repr(rep)


def test_plot_without_decisive_rows(chips):
    rep = spc.run(chips["swapped"].air, n_perm=19, rng=0, steps=("filter",))
    assert rep.status == {"overall": "limited", "molecule": "not tested", "clone": "not tested"}
    for style in ("light", "dark"):
        fig = rep.plot(style=style)
        assert "no test could be run" in " ".join(t.get_text() for t in fig.texts)
        plt.close(fig)


def test_cli_prints_three_verdicts(tmp_path, capsys):
    from spaclone.cli import main

    d = spc.sim.synthetic_chip(genuine_fraction=0.3, rng=3, size_um=320.0, tissue_radius_um=150.0)
    spc.io.write(d, tmp_path / "data")
    args = ["run", "--format", "spatialair", "--input", str(tmp_path / "data"), "--out", str(tmp_path / "out")]
    rc = main([*args, "--n-perm", "19", "--options", '{"pairing": {"min_units": 20}}'])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out and "Molecule level:" in out and "Clone level:" in out
    assert (tmp_path / "out" / "report.md").exists()
    t = pd.read_csv(tmp_path / "out" / "tests.csv")
    assert {"test", "level", "family", "verdict", "q_value", "n_perm", "detectable"} <= set(t.columns)
    with pytest.raises(SystemExit):
        main([*args, "--options", "{not json"])
    rc = main([*args, "--n-perm", "19", "--cell-type-key", "cell_type", "--own-types", '{"BCR": ["B", "GC B"]}'])
    assert rc == 0 and "own BCR types: B, GC B" in pd.read_csv(tmp_path / "out" / "tests.csv").note.str.cat()
    with pytest.raises(SystemExit, match="--own-types must be auto or a JSON object"):
        main([*args, "--own-types", '["B"]'])
    with pytest.raises(SystemExit, match="spaclone run: cell_type_key='nope' is not a column"):
        main([*args, "--cell-type-key", "nope"])


# --------------------------------------------------------------------------------------------- calibration


@pytest.mark.slow
def test_null_chips_rarely_license_clone_maps():
    """Calibration: with every molecule swapped, "Clone positions supported" in <= alpha + binomial tolerance.

    Null model: all receptor molecules sit on random captured-cDNA spots (package simulator), so clone identity is
    independent of position; the clone-level BH family controls the family-wise error at 0.05 under this null.
    """
    n = 20
    hits = sum(spc.run(_chip(0.0, 100 + s), n_perm=99, rng=s).clone_positions_supported for s in range(n))
    assert hits <= 3  # P(Binomial(20, 0.05) > 3) < 0.02
