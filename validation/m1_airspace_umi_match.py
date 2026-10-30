"""M1 validation of qc.umi_match against the archived AIR-SPACE UMI match (04 workflow X4, samples D10 and D14).

X4 streamed the raw GEX read pairs from SRA, kept the reads whose (bead barcode, UMI) was a BCR molecule key or one of
two random decoy keys per molecule, mapped them to the transcriptome and saved the matched reads with their gene
(results/AIRSPACE/umi_match_<sample>.matches.tsv.gz) and one row per key (.molecules.csv). Here the BCR molecule keys
are rebuilt from the raw ovp clone table exactly as X4 did (BCR loci, one key per (barcode, UMI, chain), 7-nt UMIs;
X4 stored keys in a dict, so a (barcode, UMI) carried by two chains keeps the last one), the decoys are X4's own
(archived key table; they carry no parent molecule, so the package stratifies by barcode), the matched reads are the
archived ones, and spaclone.qc.umi_match recomputes the per-chain summary (umi_match_summary.csv).
Writes validation/results/m1_airspace_umi_match.csv (quantity, archived, package, match, note).
Usage: python -I m1_airspace_umi_match.py [<package src to put first on sys.path>]
"""

import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone.qc import umi_match  # noqa: E402

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
RES = E / "results" / "AIRSPACE"
RAW = E / "AIRSPACE_GSE286452" / "raw"
GSM = {"D10": "GSM8727838", "D14": "GSM8727839"}
OUT = Path(__file__).resolve().parent / "results"
T0 = time.time()
rows = []


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def add(quantity, archived, package, note="", exact=False, rel=1e-12, improved=False):
    if archived is None:
        match = "new"
    elif improved:
        match = "improved"  # a deliberately different method; archived = the legacy-equivalent value
    elif exact:
        match = bool(archived == package)
    else:
        match = bool(abs(float(archived) - float(package)) <= rel * max(1.0, abs(float(archived))))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": match, "note": note})


arch = pd.read_csv(RES / "umi_match_summary.csv")
for sample, gsm in GSM.items():
    M = pd.read_csv(RAW / f"{gsm}_{sample}_ovp_clone_table.txt.gz", sep="\t", low_memory=False,
                    usecols=["barcode", "umi", "receptor_type", "clono_type"])  # fmt: skip
    M = M[M.receptor_type.isin(["IGH", "IGK", "IGL"])].drop_duplicates(["barcode", "umi", "receptor_type"])
    M = M[M.umi.astype(str).str.len() == 7]
    air = M.drop_duplicates(["barcode", "umi"], keep="last").rename(
        columns={"barcode": "spot_id", "receptor_type": "locus", "clono_type": "c_call"}
    )
    K = pd.read_csv(RES / f"umi_match_{sample}.molecules.csv", usecols=["key", "kind", "chain", "isotype"])
    dec = K[K.kind == "decoy"].copy()
    dec[["barcode", "umi"]] = dec.key.str.split("_", n=1, expand=True)
    dec = dec.rename(columns={"chain": "locus", "isotype": "c_call"})[["barcode", "umi", "locus", "c_call"]]
    H = pd.read_csv(RES / f"umi_match_{sample}.matches.tsv.gz", sep="\t", usecols=["key", "gene"])
    H[["spot_id", "umi"]] = H.key.str.split("_", n=1, expand=True)
    log(f"{sample}: {len(air):,} BCR keys, {len(dec):,} archived decoys, {len(H):,} matched GEX reads")
    for own_mode, note in (("isotype", "X4 definition: a key without isotype never matches its own gene"),
                           ("isotype_or_locus", "package default: a key without isotype falls back to its locus")):  # fmt: skip
        res = umi_match(air, H, decoys=dec, own_genes=own_mode)
        cat = res.categories.set_index(["kind", "locus"])
        for _, a in arch[arch["sample"] == sample].iterrows():
            c = cat.loc[(a.kind, a.chain)]
            key = f"{sample} {a.kind} {a.chain}"
            if own_mode == "isotype":
                add(f"{key} keys", int(a["keys"]), int(c["keys"]), "X4 umi_match_summary", exact=True)
                add(f"{key} matched", int(a.matched), int(c.matched), "X4 umi_match_summary", exact=True)
                add(f"{key} share matched", a.frac_matched, c.frac_matched, "X4 umi_match_summary")
                pairs = (
                    ("frac_matched_own_gene", "frac_matched_own", "own gene among all keys"),
                    ("of_matched_own_gene", "of_matched_own_constant", "own constant gene among matched"),
                    ("of_matched_other_ig_constant", "of_matched_other_constant", "other Ig constant among matched"),
                    ("of_matched_non_ig_gene", "of_matched_other_gene", "non-Ig gene among matched"),
                    ("of_matched_unmapped", "of_matched_undetermined", "undetermined among matched"),
                )
                for ac, pc, label in pairs:
                    add(f"{key} {label}", float(a[ac]), float(c[pc]), note)
            else:
                add(f"{key} own constant gene among matched (default own_genes)", float(a.of_matched_own_gene),
                    float(c.of_matched_own_constant), note, improved=True)  # fmt: skip
        for _, r in res.table.iterrows():
            add(f"{sample} {r.locus} {r.outcome}: BCR vs decoy rate (own_genes={own_mode})", None,
                f"{r.air_rate:.4f} vs {r.decoy_rate:.4f}",
                f"new: exact stratified test (barcode strata), p = {r.p_value:.3g}, z = {r.z:.1f}")  # fmt: skip
    log(f"{sample} done")

pd.DataFrame(rows).to_csv(OUT / "m1_airspace_umi_match.csv", index=False)
res = pd.DataFrame(rows)
log(f"spaclone {spaclone.__version__} from {spaclone.__file__}; match counts {res['match'].astype(str).value_counts().to_dict()}")
bad = res[res["match"].astype(str) == "False"]
if len(bad):
    log("NOT MATCHING:\n" + bad.to_string())
