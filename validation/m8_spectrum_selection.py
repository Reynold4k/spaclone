"""M8 validation (reported, no archived counterpart): SHM spectrum, AID motif targeting and R / S selection on the
StereoIR IGH contigs, switched vs unswitched clones.

Input: the IMGT-gapped IgBLAST + Change-O alignments of the lineage series (L3_igblast_genotyped.tsv: sequence and
IgBLAST germline alignment) joined to L4_contigs.parquet (reads, full-length and usable flags, isotype, clone). One
record per clone (the full-length usable contig with most reads, >= 30 reads), so a lineage's shared mutations count
once. Groups: switched (main contig isotype IgG / IgA) and unswitched (IgM / IgD).
Rows:
  * V mismatches at IMGT 1-312 of every full-length usable contig by bcr.shm(region=(1, 312)) against the IgBLAST
    germline vs L4 'vmm_own' (against the TIgGER-genotyped germline): share identical (the genotyped allele differs
    for some contigs, so this is not an exact target);
  * bcr.spectrum: transition share, WRCY/RGYW, SYC/GRS and WA/TW targeting ratios with Monte-Carlo p-values;
  * bcr.selection: CDR and FWR selection strength (sigma) and p-values, kappa = 1 and kappa = the observed
    transition / transversion bias.
Expected biology (not a pass / fail criterion): hotspot ratio > 1 and coldspot ratio < 1 in mutated (switched)
clones; FWR R share below its expectation (negative selection).
Output: validation/results/m8_spectrum_selection.csv (quantity, archived, package, match, note)
Usage : python -I m8_spectrum_selection.py
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone import bcr  # noqa: E402

LRES = Path(os.environ["SPACLONE_ARCHIVE"]) / "results" / "QC" / "StereoIR_lineage"
OUT = Path(__file__).resolve().parent / "results"
NROWS = int(os.environ.get("M8_NROWS", "0")) or None  # debugging only
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def info(quantity, archived, package, note):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
g = pd.read_csv(LRES / "L3_igblast_genotyped.tsv", sep="\t", low_memory=False, nrows=NROWS,
                usecols=["sequence_id", "sequence_alignment", "germline_alignment", "productive"])  # fmt: skip
c = pd.read_parquet(LRES / "L4_contigs.parquet",
                    columns=["sequence_id", "clone", "iso", "reads", "full", "usable", "vmm_own", "shm_eval"])  # fmt: skip
d = c.merge(g, on="sequence_id", how="inner")
d = d[d.usable & d.full & d.germline_alignment.notna()].reset_index(drop=True)
log(f"full-length usable contigs with an IgBLAST germline: {len(d):,}")
s = bcr.shm(d, sequence_alignment="sequence_alignment", germline_alignment="germline_alignment", region=(1, 312),
            baseline=None, min_reads=0, min_aligned_nt=0, reads_key="reads")  # fmt: skip
ev = d.shm_eval.to_numpy(dtype=bool)
same = float((s.shm_mismatches.to_numpy()[ev] == d.vmm_own.to_numpy()[ev]).mean())
info("IMGT 1-312 V mismatches equal to L4 vmm_own (SHM-evaluable contigs)", 1.0, same,
     "bcr.shm against the IgBLAST germline; L4 used the TIgGER-genotyped germline, so a few alleles differ")  # fmt: skip
d["group"] = np.where(d.iso.isin(["IGHG", "IGHA"]), "switched", np.where(d.iso.isin(["IGHM", "IGHD"]),
                      "unswitched", "unknown"))  # fmt: skip
one = (
    d[(d.reads >= 30) & (d.group != "unknown")]
    .sort_values(["clone", "reads"], ascending=[True, False])
    .drop_duplicates("clone")
    .reset_index(drop=True)
)
log(f"one record per clone: {len(one):,} ({one.group.value_counts().to_dict()})")
sp = bcr.spectrum(one, region="V", by="group", n_perm=999, rng=20261030)
log(repr(sp))
ti = sp.substitutions.groupby(["group", "transition"])["count"].sum().unstack()
for grp, r in ti.iterrows():
    tot = r.sum()
    info(f"spectrum {grp}: transition share of V mutations", 1 / 3, r[True] / tot,
         f"{int(tot):,} mutations; 1/3 = uniform substitution")  # fmt: skip
for r in sp.table.itertuples():
    info(f"spectrum {r.group}: {r.motif} targeting ratio", 1.0, r.ratio,
         f"{int(r.statistic)} of {r.mutations} mutations in the motif vs {r.expected:.1f} expected; p = {r.p_value:.3g}, "
         f"q = {r.q_value:.3g}")  # fmt: skip
n_ti = int(sp.substitutions.loc[sp.substitutions.transition, "count"].sum())
n_tv = int(sp.substitutions.loc[~sp.substitutions.transition, "count"].sum())
kappa = 2.0 * n_ti / n_tv
for k in (1.0, kappa):
    se = bcr.selection(one, by="group", kappa=k, n_perm=1999, rng=20261030)
    for r in se.table.itertuples():
        info(f"selection {r.group} {r.region} (kappa {k:.2f}): sigma", 0.0, r.sigma,
             f"R {r.R} / S {r.S}; R share {r.share_R:.3f} vs expected {r.expected_share_R:.3f}; p = {r.p_value:.3g}, "
             f"q = {r.q_value:.3g}")  # fmt: skip
pd.DataFrame(rows).to_csv(OUT / "m8_spectrum_selection.csv", index=False)
log(f"wrote {OUT / 'm8_spectrum_selection.csv'}: {len(rows)} reported rows")
