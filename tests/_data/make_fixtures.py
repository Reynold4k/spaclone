"""Write the tiny synthetic reader fixtures of ``tests/_data`` (hand-made rows, no collaborator data).

Run from anywhere: ``python make_fixtures.py``. The files mimic the column layouts of the real formats:

* ``airr_stereoir_{bcr,tcr}.tsv`` + ``cells_{b,t}.tsv``: TRUST4 AIRR as in the StereoIR demo (leading unnamed index
  column, ``cell_id = "x-y"`` on the bin20 grid, ``"*"`` = no call) and cell-type bin lists;
* ``airr_longairr.tsv``: LongAIRR-style AIRR with ``spbcid``, ``x``, ``y``, ``umi``;
* ``xcr_meta_v12.csv``: the 28-column Stereo-XCR-seq v1.2 meta table (one row per read, ``umi`` = ``cid``);
* ``xcr_public_meta.csv``: the public pipeline's ``out.meta.gz`` layout with a real 10-nt MID;
* ``airspace_mini_ovp_clone_table.txt``: the AIR-SPACE per-read clone table (subset of its columns).
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

AIRR_COLS = [
    "",
    "sequence_id",
    "sequence",
    "rev_comp",
    "productive",
    "locus",
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    "junction",
    "junction_aa",
    "consensus_count",
    "cell_id",
    "evaluate_reads",
]
BCR = [
    # cell_id, locus, v, d, j, c, junction_aa, productive, reads
    ("100-200", "IGH", "IGHV2-2*01", "IGHD1-1*01", "IGHJ3*01", "IGHA*09", "CARNSHYYGSRYPFAYW", "T", 120),
    ("100-200", "IGK", "IGKV8-19*01", "*", "IGKJ2*01", "IGKC*01", "CQNDYSYPLTF", "T", 80),
    ("120-200", "IGH", "IGHV2-2*02", "IGHD1-1*01", "IGHJ3*01", "*", "CARNSHYYGSRYPFAYW", "T", 1),
    ("140-220", "IGL", "IGLV1*01", "*", "IGLJ3*01", "IGLC3*01", "CALWYSNQFIF", "T", 15),
    ("140-220", "IGK", "IGKV6-15*01", "*", "IGKJ5*01", "IGKC*01", "CQQYWSTPLTF", "F", 3),
    ("160-240", "IGH", "IGHV1-26*01,IGHV1-18*01", "IGHD2-3*01", "IGHJ4*01", "IGHM*01", "CARDY*YW", "T", 2),
]
TCR = [
    ("200-300", "TRB", "TRBV13-1*01", "TRBD1*01", "TRBJ1-4*01", "TRBC1*01", "CASSLGQNTLYF", "T", 40),
    ("100-200", "TRA", "TRAV13-4/DV7*01", "*", "TRAJ33*01", "TRAC*01", "CAMREGSNYQLIW", "T", 2),
]


def nt(aa: str) -> str:
    """A fake in-frame nucleotide junction of the right length."""
    return "TGT" + "GCN" * (len(aa) - 2) + "TGG"


def write_airr(path: Path, rows: list[tuple]) -> None:
    lines = ["\t".join(AIRR_COLS)]
    for i, (cell, locus, v, d, j, c, aa, prod, reads) in enumerate(rows):
        seq = "ACGT" * 10
        lines.append(
            "\t".join(
                map(str, [i, f"{cell}_{i}", seq, "F", prod, locus, v, d, j, c, nt(aa), aa, reads, cell, 50000 + i])
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_longairr(path: Path) -> None:
    cols = [
        "sequence_id",
        "locus",
        "productive",
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
        "consensus_count",
        "umi",
        "spbcid",
        "x",
        "y",
    ]
    rows = [
        (
            "m1",
            "IGH",
            "T",
            "IGHV1-69*01",
            "IGHD3-10*01",
            "IGHJ4*02",
            "IGHG1",
            "CARDYW",
            7,
            "AACCGGTTAA",
            "s_002um_01",
            10,
            20,
        ),
        (
            "m2",
            "IGH",
            "T",
            "IGHV1-69*01",
            "IGHD3-10*01",
            "IGHJ4*02",
            "IGHG1",
            "CARDYW",
            3,
            "AACCGGTTCC",
            "s_002um_01",
            10,
            20,
        ),
        ("m3", "IGK", "F", "IGKV1-39*01", "", "IGKJ1*01", "IGKC", "CQQSYSTPWTF", 5, "GGTTAACCAA", "s_002um_02", 11, 20),
        (
            "m4",
            "TRB",
            "T",
            "TRBV20-1*01",
            "TRBD1*01",
            "TRBJ2-7*01",
            "TRBC2",
            "CSARDYEQYF",
            2,
            "TTTTAAAACC",
            "s_002um_03",
            40,
            5,
        ),
    ]
    lines = ["\t".join(cols)]
    for r in rows:
        r = list(r)
        r.insert(7, nt(r[7]))
        lines.append("\t".join(map(str, r)))
    path.write_text("\n".join(lines) + "\n")


META_COLS = (
    "readId,targetSequences,allVAlignments,nSeqCDR3,aaSeqCDR3,topChains,cloneId,isotypePrimary,x,y,cid,umi,functional,"
    "CDR3Begin,CDR3End,sequence_id,sequence,productive,v_call,d_call,j_call,c_call,sequence_alignment,"
    "germline_alignment,complete_vdj,junction,junction_aa,duplicate_count"
).split(",")
CLONES = {
    # cloneId: (chain, junction_aa, v, d, j, c, productive, duplicate_count)
    "0": ("IGK", "CQQYSSYPLTF", "IGKV8-19*00", "", "IGKJ2*00", "IGKC", "T", 4),
    "1": ("IGH", "CARGITVVANPDYW", "IGHV1-26*00", "IGHD1-1*00", "IGHJ2*00", "IGHM", "T", 3),
    "2": ("TRB", "CASSQDWGYEQYF", "TRBV13-1*00", "TRBD1*00", "TRBJ2-7*00", "TRBC2", "T", 2),
}
READS = [
    # cloneId, x, y, cid, read CDR3 (aa), functional
    ("0", 100, 200, "AAAAAAAAAAAAAAAAAAAAAAAAA", "CQQYSSYPLTF", "Functional"),
    ("0", 100, 200, "AAAAAAAAAAAAAAAAAAAAAAAAA", "CQQYSSYPLTF", "Functional"),
    ("0", 100, 200, "AAAAAAAAAAAAAAAAAAAAAAAAC", "CQQYSSYPLTF", "Functional"),
    ("0", 100, 201, "AAAAAAAAAAAAAAAAAAAAAAAAG", "CQQYSSYPLTF", "Functional"),
    ("1", 100, 200, "AAAAAAAAAAAAAAAAAAAAAAAAA", "CARGITVVANPDYW", "Functional"),
    ("1", 100, 200, "AAAAAAAAAAAAAAAAAAAAAAAAA", "CARGITVV*NPDYW", "NonFunctional"),
    ("1", 5000, 5000, "CCCCCCCCCCCCCCCCCCCCCCCCC", "CARGITVVANPDYW", "Functional"),
    ("2", 150, 220, "GGGGGGGGGGGGGGGGGGGGGGGGG", "CASSQDWGYEQYF", "Functional"),
    ("2", 150, 220, "GGGGGGGGGGGGGGGGGGGGGGGGG", "CASSQDWGYEQYF", "Functional"),
]


def write_meta_v12(path: Path) -> None:
    lines = [",".join(META_COLS)]
    for i, (cl, x, y, cid, aa, func) in enumerate(READS):
        chain, jaa, v, d, j, c, prod, dup = CLONES[cl]
        row = {
            "readId": f"V000000001L1C001R00100{i:06d}",
            "targetSequences": "ACGT" * 30,
            "allVAlignments": "272|362|385|0|92|ST276G|328.0",
            "nSeqCDR3": nt(aa),
            "aaSeqCDR3": aa,
            "topChains": chain,
            "cloneId": cl,
            "isotypePrimary": "",
            "x": x,
            "y": y,
            "cid": cid,
            "umi": cid,
            "functional": func,
            "CDR3Begin": 69,
            "CDR3End": 102,
            "sequence_id": f"clone.{cl}",
            "sequence": nt(jaa),
            "productive": prod,
            "v_call": v,
            "d_call": d,
            "j_call": j,
            "c_call": c,
            "sequence_alignment": nt(jaa),
            "germline_alignment": nt(jaa),
            "complete_vdj": "F",
            "junction": nt(jaa),
            "junction_aa": jaa,
            "duplicate_count": dup,
        }
        lines.append(",".join(str(row[c]) for c in META_COLS))
    # one read without a clone (MiXCR cloneId -1): not a record
    lines.append(lines[1].replace(",IGK,0,", ",IGK,-1,", 1))
    path.write_text("\n".join(lines) + "\n")


PUB_COLS = (
    "readname,AlignVHits,AlignDHits,AlignJHits,AlignCHits,cloneID,cdr3aa,cdr3nt,isotype,priisotype,cid,umi,x,y,"
    "readtype,loc,func,cdr3name,addinfo,CombineIsotype"
).split(",")
PUB = [
    # readname, clone, cdr3aa, locus, cid, umi, x, y, readtype
    ("r0", "SR_0", "CQQYSSYPLTF", "IGK", "A" * 25, "TTTTTTTTTT", 10, 20, "SR"),
    ("r1", "SR_0", "CQQYSSYPLTF", "IGK", "A" * 25, "TTTTTTTTTT", 10, 20, "SR"),
    ("r2", "SR_0", "CQQYSSYPLTF", "IGK", "A" * 25, "GGGGGGGGGG", 10, 20, "SR"),
    ("r3", "SR_5", "CQQHYSTPFTF", "IGK", "A" * 25, "TTTTTTTTTT", 10, 20, "SR"),
    ("r4", "SR_1", "CARGITVV*NPDYW", "IGH", "C" * 25, "AAAAAAAAAA", 30, 40, "SR"),
    ("r5", "LR_IGH_7", "CARGITVVANPDYW", "IGH", "C" * 25, "CCCCCCCCCC", 30, 40, "LR"),
]


def write_public(path: Path) -> None:
    """Comma-separated with quoting, as ``pandas.DataFrame.to_csv`` writes ``out.meta.gz``."""
    rows = [PUB_COLS]
    for rn, cl, aa, locus, cid, umi, x, y, rt in PUB:
        func = "None" if ("*" in aa or "_" in aa) else "Func"
        v = "IGKV8-19*00(1570),IGKV8-24*00(1500)" if locus == "IGK" else "IGHV1-26*00(1490)"
        row = [rn, v, "", f"{locus}J2*00(300)", f"{locus}C*00(800)" if locus == "IGH" else "IGKC*00(800)", cl, aa]
        row += [nt(aa), locus, "IGHM" if locus == "IGH" else locus, cid, umi, x, y, rt, f"{x}_{y}", func]
        row += [f"{aa}@{locus}", "", locus]
        rows.append(row)
    with path.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


AIRSPACE_COLS = [
    "readId",
    "allVHitsWithScore",
    "allDHitsWithScore",
    "allJHitsWithScore",
    "allCHitsWithScore",
    "aaSeqCDR3",
    "nSeqCDR3",
    "cloneId",
    "umi",
    "barcode",
    "x",
    "y",
    "clono_type",
    "receptor_type",
    "CDR3_length",
]
AIRSPACE = [
    # readId, V, cloneId, CDR3, umi, barcode, clono_type, receptor_type
    (3, "IGKV8-24*00(1500)", "11", "CQQYSSYPLTA", "AAAAAAA", "BEAD000", "IGKC", "IGK"),
    (1, "IGKV8-19*00(1570)", "10", "CQQYSSYPLTF", "AAAAAAA", "BEAD000", "IGKC", "IGK"),
    (2, "IGKV8-19*00(1570)", "10", "CQQYSSYPLTF", "AAAAAAA", "BEAD000", "IGKC", "IGK"),
    (4, "IGHV1-26*00(1490)", "20", "CARGITVVANPDYW", "AAAAAAA", "BEAD000", "IGHM", "IGH"),
    (5, "IGKV6-15*00(1400)", "12", "CQQYWSTPLTF", "CCCCCCC", "BEAD001", "IGKC", "IGK"),
    (6, "IGKV6-15*00(1400)", "13", "CQQYWSTPLT*", "GGGGGGG", "BEAD001", "IGKC", "IGK"),
    (7, "TRBV13-1*00(1570)", "30", "CASKTGPNERLFF", "TTTTTTT", "BEAD002", "TRBC1", "TRB"),
    (8, "IGKV8-19*00(1570)", "10", "CQQYSSYPLTF", "ACACACA", "BEAD099", "IGKC", "IGK"),
    (9, "IGKV8-19*00(1570)", "10", "CQQYSSYPLTF", "ACACACA", "NOTABEAD", "IGKC", "IGK"),
]


def write_airspace(path: Path) -> None:
    lines = ["\t".join(AIRSPACE_COLS)]
    for rid, v, clone, aa, umi, bc, ctype, rtype in AIRSPACE:
        row = [rid, v, "", f"{rtype}J1*00(300)", f"{ctype}*00(1800)", aa, nt(aa), clone, umi, bc, 0.0, 0.0]
        row += [ctype, rtype, len(aa)]
        lines.append("\t".join(map(str, row)))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    write_airr(HERE / "airr_stereoir_bcr.tsv", BCR)
    write_airr(HERE / "airr_stereoir_tcr.tsv", TCR)
    (HERE / "cells_b.tsv").write_text("100-200\n120-200\n180-260\n")
    (HERE / "cells_t.tsv").write_text("200-300\n")
    write_longairr(HERE / "airr_longairr.tsv")
    write_meta_v12(HERE / "xcr_meta_v12.csv")
    write_public(HERE / "xcr_public_meta.csv")
    write_airspace(HERE / "airspace_mini_ovp_clone_table.txt")


if __name__ == "__main__":
    main()
