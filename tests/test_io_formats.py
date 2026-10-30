"""Receptor formats: detection on real excerpts (tests/_data/formats, see its README), mapping, counts, anchors."""

import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spaclone import io
from spaclone.io.airr import gene_of

F = Path(__file__).parent / "_data" / "formats"

FIXTURES = [
    ("10x/cr2.2.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr3.0.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr3.1.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr4.0.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr5.0.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr5.0.0_all_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr6.1.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr9.0.0_filtered_contig_annotations.csv", "10x_contig_csv"),
    ("10x/cr5.0.0_all_contig_annotations.json", "10x_contig_json"),
    ("10x/cr5.0.0_consensus_annotations.csv", "10x_consensus_csv"),
    ("10x/cr5.0.0_clonotypes.csv", "10x_clonotypes_csv"),
    ("10x/cr4.0.0_airr_rearrangement.tsv", "airr"),
    ("10x/cr5.0.0_airr_rearrangement.tsv", "airr"),
    ("10x/cr6.1.0_airr_rearrangement.tsv", "airr"),
    ("10x/cr9.0.0_airr_rearrangement.tsv", "airr"),
    ("airr/airr_standards_good_rearrangement.tsv", "airr"),
    ("airr/airr_standards_extra_rearrangement.tsv", "airr"),
    ("airr/scirpy_rearrangement_ig.tsv", "airr"),
    ("trust4/TRUST_example_airr.tsv", "airr"),
    ("trust4/TRUST_example_report.tsv", "trust4_report"),
    ("trust4/TRUST_barcode_report.tsv", "trust4_barcode_report"),
    ("mixcr/mixcr2_human_readable.txt", "mixcr_clones"),
    ("mixcr/mixcr3_P15-T0-TIGIT.TRA.tsv", "mixcr_clones"),
    ("mixcr/mixcr4_sc_MIXCR_contigs.csv", "mixcr_clones"),
    ("mixcr/spatialTCR_clones.clonotypes.ALL.txt", "mixcr_clones"),
    ("changeo/bracer_changeodb.tab", "changeo"),
    ("immunoseq/immunoseq_1.txt", "immunoseq_v3"),
    ("immunoseq/immunoseq_2.txt", "immunoseq_v3"),
    ("immunoseq/immunoseq_3.txt", "immunoseq_v2"),
    ("immunoseq/immunoseq_4.txt", "immunoseq_v3"),
    ("bd/test_per_cell_chain.csv", "bd_rhapsody"),
    ("bd/test_unfiltered_contigs.csv", "bd_rhapsody"),
    ("sptcr/GSM7658227_SPTCR14_IGB_TRABDG.csv.gz", "sptcr_seq"),
    ("clone_tables/immunarch_Sample1.tsv.gz", "immunarch"),
    ("clone_tables/vdjtools.txt", "vdjtools"),
    ("clone_tables/migec_cdrblast.txt", "migec"),
    ("clone_tables/parse_barcode_report.tsv", "parse_barcode_report"),
    ("clone_tables/wat3r_barcode_results.csv", "wat3r"),
]
LEGACY = Path(__file__).parent / "_data"


@pytest.mark.parametrize(("rel", "expected"), FIXTURES)
def test_detection_is_never_wrong_on_the_fixtures(rel, expected):
    fmt, ev = io.receptor_format(F / rel)
    assert fmt.name == expected, ev
    assert ev


def test_existing_spatial_tables_are_detected():
    assert io.receptor_format(LEGACY / "xcr_meta_v12.csv")[0].name == "stereo_xcr_meta"
    assert io.receptor_format(LEGACY / "xcr_public_meta.csv")[0].name == "stereo_xcr_meta"
    assert io.receptor_format(LEGACY / "airspace_mini_ovp_clone_table.txt")[0].name == "airspace_ovp"
    assert io.receptor_format(LEGACY / "airr_stereoir_bcr.tsv")[0].name == "airr"
    with pytest.raises(ValueError, match="read_stereo_xcr"):
        io.read_receptors(LEGACY / "xcr_meta_v12.csv")


def test_10x_same_sample_agrees_across_csv_json_and_airr():
    csv_all = io.read_receptors(F / "10x/cr5.0.0_all_contig_annotations.csv")
    js = io.read_receptors(F / "10x/cr5.0.0_all_contig_annotations.json")
    filt = io.read_receptors(F / "10x/cr5.0.0_filtered_contig_annotations.csv")
    airr = io.read_receptors(F / "10x/cr5.0.0_airr_rearrangement.tsv")
    a, j = csv_all.set_index("sequence_id"), js.set_index("sequence_id")
    assert sorted(a.index) == sorted(j.index)
    j = j.loc[a.index]
    for c in ("cell_id", "locus", "junction_aa", "junction", "consensus_count", "umi_count"):
        pd.testing.assert_series_equal(a[c].astype(object), j[c].astype(object), check_names=False)
    for c in ("v_call", "j_call", "c_call"):
        np.testing.assert_array_equal(
            gene_of(a[c].to_numpy(object)).astype(str), gene_of(j[c].to_numpy(object)).astype(str)
        )
    assert (a["productive"].astype(object) == j["productive"].astype(object)).all()
    # the filtered file = the cell, high-confidence, productive contigs of the all file (Cell Ranger 5)
    keep = a[
        (a["is_cell"].str.lower() == "true")
        & (a["high_confidence"].str.lower() == "true")
        & a["productive"].fillna(False)
    ]
    assert sorted(filt.sequence_id) == sorted(keep.index)
    # AIRR: same contigs; consensus_count = reads and duplicate_count = UMIs (measured on this sample, recorded as the
    # writer rule for Cell Ranger airr_rearrangement.tsv)
    f, r = filt.set_index("sequence_id"), airr.set_index("sequence_id")
    assert set(r.index) <= set(a.index) and len(r) == len(f)
    r = r.loc[f.index]
    np.testing.assert_array_equal(r.consensus_count.to_numpy(), f.consensus_count.to_numpy())
    np.testing.assert_array_equal(r.umi_count.to_numpy(), f.umi_count.to_numpy())
    assert "writer: 10x Cell Ranger" in " ".join(airr.attrs["spaclone"]["io"]["evidence"])
    assert (r.junction_aa.astype(object) == f.junction_aa.astype(object)).all()
    np.testing.assert_array_equal(gene_of(r.v_call.to_numpy(object)), gene_of(f.v_call.to_numpy(object)))

    # same exact clonotypes
    def clones(t):
        from spaclone.io.airr import exact_clone_ids

        return sorted(pd.unique(exact_clone_ids(t.locus, t.v_call, t.j_call, t.junction_aa)).astype(str))

    assert clones(r) == clones(f)


def test_count_semantics_per_tool():
    def io_counts(rel, **kw):
        return io.read_receptors(F / rel, **kw).attrs["spaclone"]["io"]["counts"]

    c = io_counts("10x/cr5.0.0_filtered_contig_annotations.csv")
    assert c["consensus_count"].startswith("'reads' = reads") and c["umi_count"].startswith("'umis' = umis")
    assert io_counts("mixcr/mixcr3_P15-T0-TIGIT.TRA.tsv")["consensus_count"].startswith("'cloneCount' = reads")
    m4 = io_counts("mixcr/mixcr4_sc_MIXCR_contigs.csv")
    assert m4["consensus_count"].startswith("'readCount'") and m4["umi_count"].startswith(
        "'uniqueMoleculeCount' = umis"
    )
    bd = io_counts("bd/test_per_cell_chain.csv")
    assert "Read_Count" in bd["consensus_count"] and "Molecule_Count" in bd["umi_count"]
    v3 = io.read_receptors(F / "immunoseq/immunoseq_1.txt")
    assert "templates" in v3.attrs["spaclone"]["io"]["counts"] and v3.templates.notna().all()
    assert io_counts("sptcr/GSM7658227_SPTCR14_IGB_TRABDG.csv.gz")["umi_count"].startswith("'expression' = umis")
    mg = io_counts("clone_tables/migec_cdrblast.txt")
    assert "Good events" in mg["umi_count"] and "Good reads" in mg["consensus_count"]
    p = io.read_receptors(F / "clone_tables/parse_barcode_report.tsv")
    assert sorted(p.locus) == ["TRA", "TRA", "TRB", "TRB"] and p.umi_count.sum() == 17


@pytest.mark.parametrize(
    "rel", ["trust4/TRUST_example_report.tsv", "clone_tables/vdjtools.txt", "clone_tables/immunarch_Sample1.tsv.gz",
            "immunoseq/immunoseq_3.txt"]
)  # fmt: skip
def test_unlabelled_counts_must_be_named(rel):
    with pytest.raises(ValueError, match="count_type="):
        io.read_receptors(F / rel)
    t = io.read_receptors(F / rel, count_type="umis")
    assert t.umi_count.notna().any()
    assert "given: count_type='umis'" in t.attrs["spaclone"]["io"]["counts"]["umi_count"]
    t2 = io.read_receptors(F / rel, columns={"consensus_count": io.read_receptors(F / rel, count_type="reads")
                                             .attrs["spaclone"]["io"]["columns"]["consensus_count"]})  # fmt: skip
    assert t2.consensus_count.notna().any()


def test_junction_or_cdr3_is_decided_by_the_anchors():
    bd = io.read_receptors(F / "bd/test_per_cell_chain.csv")
    assert "junction_aa" not in bd and bd.cdr3_aa.notna().all()
    assert bd.attrs["spaclone"]["io"]["junction"]["decision"].startswith("CDR3 without the conserved anchors")
    sp = io.read_receptors(F / "sptcr/GSM7658227_SPTCR14_IGB_TRABDG.csv.gz")
    assert "cdr3_aa" in sp and "junction_aa" not in sp
    tx = io.read_receptors(F / "10x/cr5.0.0_filtered_contig_annotations.csv")
    assert tx.attrs["spaclone"]["io"]["junction"]["anchored_fraction"] > 0.9 and "junction_aa" in tx
    mix = pd.DataFrame({"barcode": list("abcdef"), "contig_id": list("123456"), "chain": "TRB", "v_gene": "TRBV20-1",
                        "j_gene": "TRBJ2-7", "cdr3": ["CASSF", "CASSW", "CASSF", "ASSL", "ASSY", "ASSQ"]})  # fmt: skip
    with pytest.raises(ValueError, match="junction="):
        io.read_receptors(mix)
    assert "cdr3_aa" in io.read_receptors(mix, junction="cdr3")
    std = io.read_receptors(F / "airr/airr_standards_good_rearrangement.tsv")  # labelled AIRR junction is kept
    assert "junction_aa" in std and "labelled" in std.attrs["spaclone"]["io"]["junction"]["decision"]


def test_locus_productive_and_gene_names():
    v3 = io.read_receptors(F / "immunoseq/immunoseq_2.txt")
    assert set(v3.locus.dropna()) == {"TRB"} and v3.attrs["spaclone"]["io"]["locus"].startswith("gene calls")
    assert (
        v3.v_call.dropna().str.match(r"^TRBV\d+(-\d+)?").all()
        and v3.v_call_source.dropna().str.startswith("TCRBV").all()
    )
    assert set(v3.productive.dropna()) <= {True, False} and (~v3.productive.dropna()).any()  # In / Out / Stop
    bd = io.read_receptors(F / "bd/test_per_cell_chain.csv")
    assert set(bd.locus) <= {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"} and "TRG" in set(bd.locus)
    tx = io.read_receptors(F / "10x/cr2.2.0_filtered_contig_annotations.csv")
    assert tx.productive.dtype == "boolean"
    ch = io.read_receptors(F / "changeo/bracer_changeodb.tab")
    assert set(ch.locus) <= {"IGH", "IGK", "IGL"} and ch.productive.all()  # LOCUS = H/K/L → from the calls


def test_columns_override_custom_format_and_errors():
    df = pd.DataFrame({"bc": ["s1", "s2"], "CDR3.aa": ["CASSLGQAYEQYF", "CAVNSGNTGKLIF"], "V": ["TRBV20-1", "TRAV12-2"],
                       "reads": [5, 7]})  # fmt: skip
    with pytest.raises(ValueError, match="matches no documented signature"):
        io.read_receptors(df)
    with pytest.raises(ValueError, match="format='custom' needs columns"):
        io.read_receptors(df, format="custom")
    t = io.read_receptors(df, format="custom", columns={"cell_id": "bc", "junction_aa": "CDR3.aa", "v_call": "V",
                                                         "consensus_count": "reads"})  # fmt: skip
    assert t.locus.tolist() == ["TRB", "TRA"] and t.consensus_count.tolist() == [5.0, 7.0]
    with pytest.raises(ValueError, match="columns="):
        io.read_receptors(df, format="custom", columns={"junction_aa": "nope"})
    with pytest.raises(ValueError, match="canonical fields"):
        io.read_receptors(df, format="custom", columns={"cdr3aa": "CDR3.aa"})
    t2 = io.read_receptors(F / "10x/cr5.0.0_filtered_contig_annotations.csv", columns={"cell_id": "contig_id"})
    assert t2.cell_id.str.contains("_contig_").all()


def test_ambiguous_header_raises_with_both_candidates():
    df = pd.DataFrame({"barcode": ["a"], "contig_id": ["1"], "chain": ["TRB"], "v_gene": ["TRBV1"], "j_gene": ["TRBJ1"],
                       "cdr3": ["CASSF"], "count": [1], "freq": [1.0], "cdr3nt": ["TGT"], "cdr3aa": ["CASSF"], "v": ["TRBV1"],
                       "d": ["."], "j": ["TRBJ1"]})  # fmt: skip
    with pytest.raises(ValueError, match="ambiguous receptor table.*10x_contig_csv.*vdjtools|ambiguous.*vdjtools.*10x"):
        io.read_receptors(df)


def _write_variant(tmp_path, name, text, *, gz=False, bom=False, crlf=False):
    if crlf:
        text = text.replace("\n", "\r\n")
    data = (("﻿" if bom else "") + text).encode()
    p = tmp_path / name
    if gz:
        with gzip.open(p, "wb") as fh:
            fh.write(data)
    else:
        p.write_bytes(data)
    return p


@pytest.mark.parametrize(
    "variant", [{}, {"gz": True}, {"bom": True}, {"crlf": True}, {"gz": True, "bom": True, "crlf": True}]
)
def test_text_variants_read_identically(tmp_path, variant):
    src = (F / "10x/cr5.0.0_filtered_contig_annotations.csv").read_text()
    ref = io.read_receptors(F / "10x/cr5.0.0_filtered_contig_annotations.csv")
    p = _write_variant(tmp_path, "x.csv" + (".gz" if variant.get("gz") else ""), src, **variant)
    t = io.read_receptors(p)
    pd.testing.assert_frame_equal(t.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False)
    # an extra pandas index column is ignored
    df = pd.read_csv(F / "10x/cr5.0.0_filtered_contig_annotations.csv", dtype=str, keep_default_na=False)
    df.to_csv(tmp_path / "idx.csv")
    assert len(io.read_receptors(tmp_path / "idx.csv")) == len(ref)


def test_renamed_missing_and_case_changed_columns(tmp_path):
    df = pd.read_csv(F / "10x/cr5.0.0_filtered_contig_annotations.csv", dtype=str, keep_default_na=False)
    df.drop(columns=["umis"]).to_csv(tmp_path / "nou.csv", index=False)
    t = io.read_receptors(tmp_path / "nou.csv")
    assert "umi_count" not in t and t.consensus_count.notna().all()
    df.rename(columns=str.upper).to_csv(tmp_path / "upper.csv", index=False)
    with pytest.raises(ValueError, match="no documented signature"):  # never mapped by a guess
        io.read_receptors(tmp_path / "upper.csv")
    up = io.read_receptors(tmp_path / "upper.csv", format="custom", columns={"cell_id": "BARCODE", "junction_aa": "CDR3",
                                                                             "v_call": "V_GENE", "j_call": "J_GENE"})  # fmt: skip
    assert len(up) == len(df)


def test_mixed_loci_and_several_files(tmp_path):
    a = io.read_receptors(
        [F / "10x/cr2.2.0_filtered_contig_annotations.csv", F / "10x/cr3.1.0_filtered_contig_annotations.csv"]
    )
    assert {"TRA", "TRB"} <= set(a.locus) and {"IGH"} & set(a.locus)
    assert sorted(a.source_file.unique()) == [
        "cr2.2.0_filtered_contig_annotations.csv",
        "cr3.1.0_filtered_contig_annotations.csv",
    ]


def test_trust4_barcode_report_melts_every_chain():
    t = io.read_receptors(F / "trust4/TRUST_barcode_report.tsv")
    assert len(t) == 6 and set(t.locus) == {"IGH", "IGK", "IGL", "TRA", "TRB"}
    assert (t["rank"] == "secondary").sum() == 1 and t.consensus_count.tolist()[0] == 12.0
    assert t.cell_id.str.endswith("-1").all()


def test_scirpy_dandelion_and_their_objects():
    import anndata as ad

    for n in ("wu2020_40_v0_6.h5ad", "wu2020_40_v0_11.h5ad"):
        t = io.read_receptors(F / "scirpy" / n)
        assert t.attrs["spaclone"]["io"]["format"] == "scirpy_legacy" and t.junction_aa.notna().any()
        assert t.cell_id.isin(ad.read_h5ad(F / "scirpy" / n).obs_names).all()
    dd = io.read_receptors(F / "dandelion/vdj_processed_tiny.h5ddl")
    assert (
        len(dd) == 60 and "umi_count" in dd and "writer: Dandelion" in " ".join(dd.attrs["spaclone"]["io"]["evidence"])
    )
    fake = SimpleNamespace(data=pd.read_csv(F / "airr/scirpy_rearrangement_ig.tsv", sep="\t"), metadata=pd.DataFrame())
    assert len(io.read_receptors(fake)) == 4


def test_scirpy_mudata_files_and_objects():
    md = pytest.importorskip("mudata")  # optional: spaclone[scirpy]
    pytest.importorskip("awkward")
    mu = io.read_receptors(F / "scirpy/wu2020_25.h5mu")
    assert mu.cell_id.nunique() <= 25 and "junction_aa" in mu
    obj = md.read_h5mu(F / "scirpy/wu2020_25.h5mu")
    pd.testing.assert_frame_equal(io.read_receptors(obj), mu)
    pd.testing.assert_frame_equal(io.read_receptors(obj.mod["airr"]), mu)


def test_h5mu_without_mudata_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mudata", None)  # as in a core install: import mudata fails
    with pytest.raises(ImportError, match=r"needs mudata: pip install 'spaclone\[scirpy\]'"):
        io.read_receptors(F / "scirpy/wu2020_25.h5mu")


def test_per_tool_readers_restrict_the_family():
    assert (
        io.read_10x_vdj(F / "10x/cr5.0.0_all_contig_annotations.json").attrs["spaclone"]["io"]["format"]
        == "10x_contig_json"
    )
    assert io.read_mixcr(F / "mixcr/mixcr4_sc_MIXCR_contigs.csv").cell_id.notna().all()
    assert io.read_trust4(F / "trust4/TRUST_example_report.tsv", count_type="reads").consensus_count.sum() > 0
    assert io.read_changeo(F / "changeo/bracer_changeodb.tab").clone_id.notna().all()
    assert io.read_bd_rhapsody(F / "bd/test_unfiltered_contigs.csv").umi_count.notna().all()
    assert len(io.read_parse(F / "clone_tables/parse_barcode_report.tsv")) == 4
    with pytest.raises(ValueError, match="no documented signature"):
        io.read_mixcr(F / "10x/cr5.0.0_filtered_contig_annotations.csv")
