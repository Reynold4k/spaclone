"""sniff, barcode matching, annotation detection, spaclone.read on new formats, and the CLI."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import io
from spaclone.cli import main

F = Path(__file__).parent / "_data" / "formats"


def test_sniff_receptor_tables():
    r = io.sniff(F / "10x/cr5.0.0_filtered_contig_annotations.csv")
    assert r.ok and r.format == "10x_contig_csv" and r.columns["cell_id"] == "barcode"
    assert "consensus_count" in r.counts and "junction" in r.junction["decision"]
    assert "10x_contig_csv" in str(r) and "nothing else required" in str(r)
    t = io.sniff(F / "trust4/TRUST_example_report.tsv")
    assert not t.ok and "count_type" in t.required and "REQUIRED count_type=" in str(t)
    assert any("bulk repertoire" in n for n in t.notes)
    b = io.sniff(F / "bd/test_per_cell_chain.csv")
    assert b.ok and b.junction["decision"].startswith("CDR3 without")
    u = io.sniff(pd.DataFrame({"a": [1], "b": [2]}))
    assert not u.ok and "format" in u.required


def test_sniff_gex_sources():
    v = io.sniff(F / "visium_hd_square_008um")
    assert v.ok and v.kind == "gex" and "microns_per_pixel" in v.units["unit"]
    g = io.sniff(F / "stereo/chipA_window.raw.gef")
    assert "bin_size" in g.required
    s = io.sniff(F / "slideseq")
    assert "unit_um" in s.required
    h = io.sniff(F / "stereo/chipA_bin20_stereopy_subset.h5ad")
    assert h.ok and "0.5 µm" in h.units["unit"]
    sc = io.sniff(F / "scirpy/wu2020_40_v0_6.h5ad")
    assert sc.kind == "receptor+gex" and sc.format == "scirpy_legacy"


def test_barcode_rules_exact_then_documented_normalisations():
    gex = [f"{b}-1" for b in ("AAACCTGAGATCTGAA", "AAACCTGAGCTAACTC", "AAACCTGCATGCTGGC")]
    m = io.match_barcodes(gex[:2], gex)
    assert m.rule == "exact" and m.overlap_before == 1.0
    m = io.match_barcodes([g[:-2] for g in gex], gex)  # receptor ids without the 10x suffix
    assert m.rule == "10x_suffix" and m.overlap_before == 0.0 and m.overlap_after == 1.0
    concat = [f"{g}-sec1" for g in gex] + [f"{g}-sec1" for g in ("GGGCCTGAGATCTGAA-1", "TTTCCTGAGCTAACTC-1")] * 0
    m = io.match_barcodes(gex, concat)
    assert m.rule.startswith("concat_key suffix") and m.mapping[gex[0]] == concat[0]
    seurat = [f"S1_{g}" for g in gex]
    assert io.match_barcodes(gex, seurat).rule.startswith("concat_key prefix")
    hd2 = ["s_002um_00400_00800-1", "s_002um_00403_00801-1", "s_002um_00404_00800-1"]
    hd8 = ["s_008um_00100_00200-1", "s_008um_00101_00200-1"]
    m = io.match_barcodes(hd2, hd8)
    assert m.rule == "visium_hd_bins" and m.mapping["s_002um_00404_00800-1"] == "s_008um_00101_00200-1"
    st = io.match_barcodes(["8860-6440", "8880-6440"], [str(8860 << 32 | 6440), str(8880 << 32 | 6440)])
    assert st.rule == "stereo_coordinates" and st.mapping["8880-6440"] == str(8880 << 32 | 6440)
    assert io.match_barcodes(["8860-6440"], ["8860_6440", "1_2"]).rule == "stereo_coordinates"
    with pytest.raises(ValueError, match=r"only 0\.0% .*Receptor ids: \['ZZZ'\]"):
        io.match_barcodes(["ZZZ"], gex)
    with pytest.raises(ValueError, match="no documented id normalisation"):  # a collision is never resolved
        io.match_barcodes([g[:-2] for g in gex], [f"{g[:-2]}-1" for g in gex] + [f"{g[:-2]}-2" for g in gex])


def test_annotation_candidates():
    a = ad.AnnData(obs=pd.DataFrame({"cell2loc_anno": pd.Categorical(["B", "T", "B"]), "leiden": ["0", "1", "0"]},
                                    index=list("abc")))  # fmt: skip
    for k in ("means_cell_abundance_w_sf", "q05_cell_abundance_w_sf", "q95_cell_abundance_w_sf"):
        a.obsm[k] = pd.DataFrame(np.abs(np.random.default_rng(0).normal(size=(3, 2))), index=a.obs_names,
                                 columns=["B", "T"])  # fmt: skip
    r = io.detect_annotations(a)
    assert r.label == "cell2loc_anno" and r.clusters == ["leiden"] and r.proportion == "q05_cell_abundance_w_sf"
    a.obs["cell_type"] = ["x", "y", "x"]
    r = io.detect_annotations(a)
    assert r.label is None and sorted(r.labels) == ["cell2loc_anno", "cell_type"]


def _visium_pair(tmp_path, *, suffix=True):
    a, _ = io.read_gex(F / "visium_sd_sr1")
    bcs = a.obs_names[:5]
    rec = pd.read_csv(F / "10x/cr5.0.0_filtered_contig_annotations.csv").head(10)
    rec["barcode"] = [bcs[i % 5] if suffix else bcs[i % 5][:-2] for i in range(10)]  # TRUST4 / 10x on spot barcodes
    p = tmp_path / "contigs.csv"
    rec.to_csv(p, index=False)
    return a, p, rec


def test_read_places_10x_contigs_on_visium_spots(tmp_path, capsys):
    a, p, rec = _visium_pair(tmp_path)
    sa = io.read(F / "visium_sd_sr1", p)
    assert capsys.readouterr().out == ""
    m = sa.meta
    assert m["io"]["receptors"]["format"] == "10x_contig_csv" and m["io"]["gex"]["format"] == "spaceranger"
    assert m["io"]["barcodes"]["rule"] == "exact" and m["spot_overlap"] == 1.0
    sx = m["um_per_unit"]
    want = a.obsm["spatial"][a.obs_names.get_indexer(rec.barcode)] * sx
    np.testing.assert_allclose(sa.air[["x_um", "y_um"]].to_numpy(), want)
    assert sa.air.consensus_count.tolist() == rec.reads.astype(float).tolist()
    assert "umi_count" in sa.air and (sa.air.clone_id == rec.raw_clonotype_id).all()  # 10x clonotypes, as in AIRR
    _, p2, _ = _visium_pair(tmp_path, suffix=False)
    sb = io.read(F / "visium_sd_sr1", p2, verbose=True)
    out = capsys.readouterr().out
    assert sb.meta["io"]["barcodes"]["rule"] == "10x_suffix" and "barcodes: 10x_suffix" in out
    np.testing.assert_allclose(sb.air[["x_um", "y_um"]].to_numpy(), sa.air[["x_um", "y_um"]].to_numpy())


def test_read_mixcr_cells_and_bulk_errors(tmp_path):
    mx = pd.read_csv(F / "mixcr/mixcr4_sc_MIXCR_contigs.csv")
    cells = pd.unique(mx.tagValueCELL)
    g = ad.AnnData(X=sparse.csr_matrix(np.ones((len(cells), 2), np.float32)), obs=pd.DataFrame(index=cells))
    g.obsm["spatial"] = np.column_stack([np.arange(len(cells)), np.zeros(len(cells))]).astype(float)
    sa = io.read(g, F / "mixcr/mixcr4_sc_MIXCR_contigs.csv", unit_um=10.0)
    assert len(sa.air) == len(mx) and sa.meta["io"]["receptors"]["format"] == "mixcr_clones"
    assert sa.air.v_call.str.match(r"^TR[AB]V").all() and not sa.air.v_call.str.contains(r"\(").any()
    with pytest.raises(KeyError, match="no cell / spot barcodes"):
        io.read(g, F / "mixcr/mixcr3_P15-T0-TIGIT.TRA.tsv", unit_um=10.0)
    with pytest.raises(ValueError, match="count_type="):
        io.read(g, F / "trust4/TRUST_example_report.tsv", unit_um=10.0)


def test_read_single_scirpy_object_and_unit_error(tmp_path):
    a = ad.read_h5ad(F / "scirpy/wu2020_40_v0_6.h5ad")
    a.obsm["spatial"] = np.column_stack([np.arange(a.n_obs), np.arange(a.n_obs)]).astype(float)
    with pytest.raises(ValueError, match="unit_um="):
        io.read(a)
    sa = io.read(a, unit_um=1.0, counts="X")
    assert sa.meta["io"]["receptors"]["format"] == "scirpy_legacy" and sa.air.spot_id.isin(a.obs_names).all()


def test_cli_sniff(capsys):
    assert main(["sniff", str(F / "10x/cr5.0.0_filtered_contig_annotations.csv")]) == 0
    assert "10x_contig_csv" in capsys.readouterr().out
    assert main(["sniff", str(F / "trust4/TRUST_example_report.tsv")]) == 1
    assert "REQUIRED count_type=" in capsys.readouterr().out


def test_cli_sniff_judges_the_airspace_pair_together(tmp_path, capsys):
    bc = [f"BEAD{i:03d}" for i in range(36)] + ["BEAD098", "BEAD099"]
    grid = np.stack(np.meshgrid(np.arange(6) * 10.0, np.arange(6) * 10.0), -1).reshape(-1, 2)
    xy = np.r_[grid, [[25, 25], [500, 500]]]
    a = ad.AnnData(X=sparse.csr_matrix(np.full((38, 3), 50.0, np.float32)), obs=pd.DataFrame({"barcode": bc}, index=bc))
    a.obs["x"], a.obs["y"] = xy[:, 0] + 700, xy[:, 1] + 700
    a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy()
    h5 = tmp_path / "GSM0000001_adata_D99_with_image_raw_counts.h5ad"
    a.write_h5ad(h5)
    ct = tmp_path / "GSM0000002_D99_ovp_clone_table.txt"
    ct.write_text((Path(__file__).parent / "_data" / "airspace_mini_ovp_clone_table.txt").read_text())
    assert main(["sniff", str(h5)]) == 1  # the bead h5ad alone: no unit
    assert "REQUIRED unit_um=" in capsys.readouterr().out
    assert main(["sniff", str(ct), str(h5)]) == 0  # with its clone table, as spaclone.read reads the pair
    out = capsys.readouterr().out
    assert "0.65 µm per unit (AIR-SPACE / Curio bead unit" in out and "REQUIRED" not in out
    assert len(io.read(h5, ct).air) == 6  # and read agrees
