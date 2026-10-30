"""Tests of spaclone.ext.immcantation: export / import of spatial records and the IgBLAST -> Change-O -> R round trip.

The round-trip tests run when the tools are found (``SPACLONE_IGBLAST``, ``SPACLONE_IGDATA``, ``SPACLONE_GERMLINES``,
``SPACLONE_CHANGEO``, ``SPACLONE_RSCRIPT`` or ``PATH``); they build a rearranged mouse heavy chain from the IMGT
references with two planted V mutations and check that the tools recover it.
"""

import os
import shlex
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spaclone.ext import immcantation as im

MUT = {"A": "G", "G": "A", "C": "T", "T": "C"}


def test_export_airr_and_to_records(tmp_path):
    air = pd.DataFrame(
        {
            "spot_id": ["a", "b", "c"],
            "x_um": [0.0, 10.0, 20.0],
            "y_um": [5.0, 5.0, 5.0],
            "locus": ["IGH", "IGH", "IGK"],
            "sequence": ["acgtNNac", None, "ttgg-ca"],
            "library_id": "s1",
        },
        index=["p", "q", "r"],
    )
    files = im.export_airr(air, tmp_path)
    fa = files["fasta"].read_text().split()
    assert fa == [">r0", "ACGTNNAC", ">r2", "TTGGCA"]
    rec = pd.read_csv(files["records"], sep="\t")
    assert rec.sequence_id.tolist() == ["r0", "r2"] and rec.spaclone_row.tolist() == [0, 2]
    assert {"spot_id", "x_um", "y_um", "library_id", "locus"} <= set(rec.columns)
    db = pd.DataFrame({"sequence_id": ["r2", "r0"], "v_call": ["IGKV1-110*01", "IGHV1-7*01"], "mu_freq": [0.0, 0.02]})
    back = im.to_records(db, air)
    assert back.loc["p", "imm_v_call"] == "IGHV1-7*01" and pd.isna(back.loc["q", "imm_v_call"])
    assert back.loc["r", "imm_mu_freq"] == 0.0 and back["imm_mu_freq"].dtype == float
    assert list(back.index) == list(air.index)


def _fake_exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\necho fake\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_tools_from_environment_and_dry_runs(tmp_path, monkeypatch):
    for d in ("bin", "igdata/database", "igdata/optional_file", "germ", "changeo"):
        (tmp_path / d).mkdir(parents=True)
    _fake_exe(tmp_path / "bin" / "igblastn")
    for s in "vdj":
        (tmp_path / "igdata" / "database" / f"imgt_mouse_ig_{s}.nsq").write_text("")
    (tmp_path / "igdata" / "optional_file" / "mouse_gl.aux").write_text("")
    for g in ("IGHV", "IGHD", "IGHJ", "IGKV", "TRBV"):
        (tmp_path / "germ" / f"imgt_mouse_{g}.fasta").write_text(">x\nacgt\n")
    for n in ("MakeDb.py", "DefineClones.py", "CreateGermlines.py"):
        _fake_exe(tmp_path / "changeo" / n)
    monkeypatch.setenv("SPACLONE_IGBLAST", str(tmp_path / "bin"))
    monkeypatch.setenv("SPACLONE_IGDATA", str(tmp_path / "igdata"))
    monkeypatch.setenv("SPACLONE_GERMLINES", str(tmp_path / "germ"))
    monkeypatch.setenv("SPACLONE_CHANGEO", str(tmp_path / "changeo"))
    t = im.find_tools()
    assert t.igblastn == tmp_path / "bin" / "igblastn"
    assert t.germline_db["V"].name == "imgt_mouse_ig_v" and t.auxiliary_data is not None
    assert [p.name for p in t.references] == [f"imgt_mouse_{g}.fasta" for g in ("IGHD", "IGHJ", "IGHV", "IGKV")]
    cmd = im.igblast(tmp_path / "q.fasta", tools=t, threads=4, dry_run=True)
    assert cmd[0].endswith("igblastn") and "7 std qseq sseq btop" in cmd and "-auxiliary_data" in cmd
    assert cmd[cmd.index("-num_threads") + 1] == "4" and cmd[cmd.index("-organism") + 1] == "mouse"
    mk = im.makedb(tmp_path / "q.fmt7", tmp_path / "q.fasta", tools=t, dry_run=True)
    assert mk[1] == "igblast" and "--extended" in mk and "--asis-id" in mk and len(t.references) == 4
    dc = im.define_clones(pd.DataFrame(), dist=0.12, tools=t, dry_run=True)
    assert dc[dc.index("--dist") + 1] == "0.12" and dc[dc.index("--model") + 1] == "ham"
    cg = im.create_germlines(pd.DataFrame(), tools=t, dry_run=True)
    assert "--cloned" in cg and cg[cg.index("-g") + 1] == "dmask"
    with pytest.raises(FileNotFoundError, match="Rscript"):
        im.Tools(rscript=None).require("r")


def _tools_or_skip(*need: str) -> im.Tools:
    from spaclone._versions import _tool_check

    t = im.find_tools()
    try:
        for step in need:
            t.require(step)
    except FileNotFoundError as e:
        pytest.skip(f"Immcantation tools not found: {e}")
    for name, exe in [("igblastn", t.igblastn), ("Rscript", t.rscript), *t.changeo.items()]:
        if exe is not None and not _tool_check(name, exe)[0]:
            pytest.skip(f"{exe} is installed but does not run")
    return t


def _fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            out[name] = ""
        elif name is not None:
            out[name] += line.strip()
    return out


def _stop_free(seq: str) -> bool:
    return not any(seq[i : i + 3] in ("TAA", "TAG", "TGA") for i in range(0, len(seq) - 2, 3))


def _mutate(seq: str, candidates, k: int) -> tuple[str, list[int]]:
    """Plant k transitions at the first candidate positions that keep the reading frame free of stop codons."""
    s, done = list(seq), []
    for p in candidates:
        if len(done) == k:
            break
        trial = s.copy()
        trial[p] = MUT[trial[p]]
        if _stop_free("".join(trial)):
            s, done = trial, [*done, int(p)]
    return "".join(s), done


def _planted(t: im.Tools) -> tuple[pd.DataFrame, str, list[int]]:
    ref = {p.stem.upper()[-4:]: _fasta(p) for p in t.references}
    v_gapped = ref["IGHV"]["IGHV1-7*01"].upper()
    d, j = ref["IGHD"]["IGHD1-1*01"].upper(), ref["IGHJ"]["IGHJ2*01"].upper()
    v = v_gapped.replace(".", "")  # 294 nt, 98 codons; J's codons start after its first 2 nt
    joins = [(n1, n2) for n1 in ("", "G", "GG", "C", "CC", "GGC") for n2 in ("", "C", "CC", "G", "GG", "CCG")]
    rearranged = next(
        r
        for n1, n2 in joins
        if (len(v) + len(n1) + len(d) + len(n2) + 2) % 3 == 0 and _stop_free(r := v + n1 + d + n2 + j)
    )
    seq, sites = _mutate(rearranged, [60, 150, 61, 151, 62, 152], 2)
    air = pd.DataFrame(
        {
            "spot_id": ["s1", "s2"],
            "x_um": [12.0, 300.0],
            "y_um": [7.0, 40.0],
            "locus": ["IGH", "IGH"],
            "sequence": [seq, seq],
            "library_id": "lib",
        }
    )
    return air, v_gapped, sites


def test_igblast_makedb_germlines_round_trip(tmp_path):
    t = _tools_or_skip("igblast", "makedb", "create_germlines")
    air, v_gapped, sites = _planted(t)
    log: list[str] = []
    files = im.export_airr(air, tmp_path)
    f7 = im.igblast(files["fasta"], tools=t, log=log)
    db = im.makedb(f7, files["fasta"], tools=t, records=files["records"], log=log)
    assert len(db) == 2 and db.v_call.str.contains("IGHV1-7").all() and db.j_call.str.contains("IGHJ2").all()
    assert db.sequence_alignment.str.contains(r"\.").all()  # IMGT gaps
    assert db.x_um.tolist() == [12.0, 300.0]  # the spatial columns travel with the sequences
    g = im.create_germlines(db, cloned=False, tools=t, log=log)
    germ = g.germline_alignment_d_mask.iloc[0][:312]
    seq = g.sequence_alignment.iloc[0][:312]
    assert germ.replace(".", "") == v_gapped[:312].replace(".", "")[: len(germ.replace(".", ""))]
    mism = sum(1 for a, b in zip(seq, germ, strict=False) if a in "ACGT" and b in "ACGT" and a != b)
    assert mism == len(sites)
    back = im.to_records(g, air)
    assert back.imm_v_call.str.contains("IGHV1-7").all() and back.imm_germline_alignment_d_mask.notna().all()
    assert len(log) == 3 and shlex.split(log[0])[0].endswith("igblastn")


@pytest.mark.slow
def test_run_with_r_threshold_mutations_and_trees(tmp_path):
    """Null model: none (a deterministic pipeline); checks the R round trip on planted sequences.

    References: Gupta et al. 2015, doi:10.1093/bioinformatics/btv359; Hoehn et al. 2022,
    doi:10.1371/journal.pcbi.1009885.
    """
    t = _tools_or_skip("igblast", "makedb", "define_clones", "create_germlines", "r")
    air, _, _ = _planted(t)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(6):  # one clone: the planted chain with 0-5 further V mutations (no stop codons)
        s, _ = _mutate(air.sequence.iloc[0], rng.permutation(np.arange(20, 280)), k)
        rows.append({**air.iloc[0].to_dict(), "spot_id": f"s{k}", "x_um": 10.0 * k, "sequence": s})
    clone = pd.DataFrame(rows)
    res = im.run(clone, tmp_path / "run", tools=t, dist=0.2)
    assert res.db.clone_id.nunique() == 1 and res.air.imm_clone_id.notna().all()
    assert (res.air.imm_mu_freq.fillna(0) >= 0).all() and res.air.imm_mu_freq.max() > 0
    assert len(res.commands) == 5 and shlex.split(res.commands[-1])[0].endswith("Rscript")  # observedMutations
    trees = im.dowser_trees(res.db, tools=t, minseq=3)
    assert len(trees) == 1
    from spaclone.bcr import tip_table

    tips = tip_table(trees, res.db, id_key="sequence_id")
    assert len(tips) == len(clone) and tips.x_um.notna().all()  # every input sequence has its tree tip and position
    assert sorted(tips.x_um) == sorted(clone.x_um)
    assert os.path.exists(res.out_dir / "commands.txt")


def test_airr_tables_keep_integer_positions(tmp_path):
    """pandas reads integer AIRR columns with gaps as floats; Change-O needs "4", not "4.0"."""
    db = pd.DataFrame(
        {"sequence_id": ["r0", "r1"], "d_germline_start": [4.0, np.nan], "v_identity": [0.98, 1.0], "x": [1.0, 2.0]}
    )
    text = im._to_tsv(db, tmp_path / "db.tsv").read_text().splitlines()
    assert text[1].split("\t") == ["r0", "4", "0.98", "1"] and text[2].split("\t") == ["r1", "", "1.0", "2"]
