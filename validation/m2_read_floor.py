"""M2 validation: partner-chain sharing by read floor (spaclone.qc.read_floor) on the StereoIR TCR data.

Input: $SPACLONE_ARCHIVE/StereoIR/mm_spleen_tcr_airr.tsv.tsv, filtered as TR1 (TRA / TRB, CDR3 C...[FW] of standard
amino acids, >= 5 aa, productive, consensus_count >= 2); clonotype = V gene | J gene | CDR3 aa; positions = bin20
centres (cell_id "x-y" in DNB * 0.5 um).

1. Bin-level diagnostic (workflow/diag/stereoir_TR_chimera.py, printed in logs/TR_chimera_33861267.log): TRB
   (bin, clonotype) units, partner = TRA clonotypes of the same bin; shares printed with three decimals. Package with
   radius_um=None, units in the script's order and its generator np.random.default_rng(0) consumed in its order, so
   the shuffles are identical draw for draw: 'bins >= 20 reads 8.2% vs 0.9%', 'bins < 20 reads 8.6% vs 7.2%', and the
   floor table (2, 5, 10, 20, 50, 100).
2. TR4 matching alpha (results/QC/StereoIR_TCR/TR4_partner_test.csv, 'all expanded'): TRB member loci (TR1 loci of
   15 um, best contig >= 20 reads), clonotypes with >= 3 member loci, matching alpha = one TRA clonotype at >= 2 of
   them; chance = the same number of random member loci (TR4: 1,000 draws per clonotype, no re-merging). Package
   with radius_um=15, floors=(20,), min_loci=3, min_partner_loci=0, remerge=False (legacy-equivalent), then
   remerge=True (improved: shuffled loci that meet within 15 um are merged as real loci would be).
3. TR1 counts (results/QC/StereoIR_TCR/TR1_summary.csv): TRB loci and member loci, clonotypes with >= 2 / 3 / 5
   member loci, from the package's locus table.
4. Improved (information): the locus-level read-floor table (radius 15 um, re-merging, both sides of every floor);
   the archive has no locus-level numbers, so these rows carry match = NaN. The package default since the
   2026-10-30 review (fix M6) removes barcode sinks first (sink_min_clones=20) and picks the suggested floor after a
   Benjamini-Hochberg correction across floors; both suggestions (corrected and the legacy raw-p rule) are reported.
Legacy settings for blocks 1-3: sink_min_clones=None (multiplicity changes only the suggestion, not the table).
Output: validation/results/m2_read_floor.csv
Usage : python -I m2_read_floor.py
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spaclone
from spaclone import qc

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
AA = re.compile(r"^C[ACDEFGHIKLMNPQRSTVWY]{3,}[FW]$")
T0 = time.time()
rows = []
LEGACY = {"sink_min_clones": None}
# logs/TR_chimera_33861267.log (three decimals): (floor, side, shuffles) -> (tested, share, shuffled mean)
DIAG = {
    ("all", 2, "above", 50): (811, 0.067, 0.027),
    ("first", 20, "above", 50): (269, 0.082, 0.009),
    ("first", 20, "below", 50): (304, 0.086, 0.072),
    ("floor", 2, "above", 30): (811, 0.067, 0.027),
    ("floor", 5, "above", 30): (409, 0.078, 0.016),
    ("floor", 10, "above", 30): (315, 0.079, 0.013),
    ("floor", 20, "above", 30): (269, 0.082, 0.010),
    ("floor", 50, "above", 30): (220, 0.077, 0.006),
    ("floor", 100, "above", 30): (186, 0.081, 0.007),
}
# same log: floor -> (contigs, clonotypes with >= 2, >= 3, >= 5 bins)
DIAG_SIZES = {2: (15728, 2373, 805, 190), 5: (12383, 1680, 431, 81), 10: (11151, 1430, 332, 57),
              20: (10201, 1217, 268, 44), 50: (9014, 957, 182, 32), 100: (8049, 779, 147, 25)}  # fmt: skip


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", atol=0.0, rtol=0.0, info=False):
    a, p = float(archived), float(package)
    match = np.nan if info else bool((np.isnan(a) and np.isnan(p)) or abs(a - p) <= atol + rtol * abs(a))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": match, "note": note})


def main():
    log(f"spaclone {spaclone.__version__}")
    cols = ["cell_id", "locus", "v_call", "j_call", "junction_aa", "productive", "consensus_count"]
    a = pd.read_csv(E / "StereoIR" / "mm_spleen_tcr_airr.tsv.tsv", sep="\t", usecols=cols, low_memory=False)
    a = a[a.locus.isin(["TRA", "TRB"])
          & a.junction_aa.fillna("").astype(str).map(lambda s: bool(AA.match(s)))
          & a.productive.astype(str).str.upper().str.startswith("T") & (a.consensus_count >= 2)].copy()  # fmt: skip
    a["clone"] = a.v_call.str.split("*").str[0] + "|" + a.j_call.str.split("*").str[0] + "|" + a.junction_aa
    xy = a.cell_id.str.split("-", expand=True).astype(float) * 0.5
    a["x_um"], a["y_um"] = xy[0].to_numpy(), xy[1].to_numpy()
    tr1 = pd.read_csv(E / "results" / "QC" / "StereoIR_TCR" / "TR1_summary.csv").set_index("item").value
    add("TR1 valid TRA + TRB contigs", tr1["valid contigs"], len(a))
    # the diag script's TRB order: by reads, descending, one row per (bin, clonotype)
    b = a[a.locus == "TRB"].sort_values("consensus_count", ascending=False).drop_duplicates(["cell_id", "clone"])
    tra = a[a.locus == "TRA"].drop_duplicates(["cell_id", "clone"])
    air = pd.concat([b, tra]).rename(columns={"cell_id": "spot_id", "clone": "clone_id"})
    air = air[["spot_id", "x_um", "y_um", "locus", "clone_id", "consensus_count"]]

    # ---------------------------------------------------------------- 1 bin-level diagnostic, same draws
    gen = np.random.default_rng(0)
    exact = "legacy-equivalent (radius_um=None, same draws); archive printed 3 decimals"
    calls = [("all", (2,), "above", 50), ("first", (20,), "above", 50), ("first", (20,), "below", 50),
             ("floor", (2, 5, 10, 20, 50, 100), "above", 30)]  # fmt: skip
    for block, floors, side, n_perm in calls:
        r = qc.read_floor(air, floors=floors, sides=(side,), radius_um=None, n_perm=n_perm, rng=gen, **LEGACY)
        for t in r.table.itertuples():
            n_t, share, shuf = DIAG[(block, int(t.floor), side, n_perm)]
            tag = f"diag bins {'>=' if side == 'above' else '<'} {t.floor:g} reads ({n_perm} shuffles)"
            add(f"{tag}: clonotypes tested", n_t, t.n_tested, exact)
            add(f"{tag}: share with a shared TRA clonotype", share, round(t.statistic, 3), exact)
            add(f"{tag}: shuffled mean", shuf, round(t.null_mean, 3), exact)
            if block == "floor":
                lo = r.loci[r.loci.reads_max >= t.floor]
                sz = lo.groupby("clone_id").size()
                arch = DIAG_SIZES[int(t.floor)]
                add(f"diag floor {t.floor:g}: TRB (bin, clonotype) units", arch[0], len(lo), exact)
                for k, v in zip((2, 3, 5), arch[1:], strict=True):
                    add(f"diag floor {t.floor:g}: clonotypes with >= {k} bins", v, int((sz >= k).sum()), exact)
        log(
            r.table[["floor", "side", "n_tested", "n_shared", "statistic", "null_mean", "p_value"]].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------- 2 TR4 matching alpha among expanded clonotypes
    part = pd.read_csv(E / "results" / "QC" / "StereoIR_TCR" / "TR4_partner_test.csv").set_index("class")
    ae = part.loc["all expanded"]
    for remerge, note in ((False, "legacy-equivalent (remerge=False, TR4 chance model)"),
                          (True, "improved (remerge=True): shuffled loci within 15 um merged")):  # fmt: skip
        r = qc.read_floor(air, floors=(20,), sides=("above",), radius_um=15.0, min_loci=3, min_partner_loci=0,
                          remerge=remerge, n_perm=1000, rng=np.random.default_rng(4), **LEGACY)  # fmt: skip
        t = r.table.iloc[0]
        add(f"TR4 expanded TRB clonotypes (>= 3 member loci), remerge={remerge}", ae.clonotypes, t.n_tested, note)
        add(
            f"TR4 expanded clonotypes with a matching alpha, remerge={remerge}",
            ae.with_matching_alpha,
            t.n_shared,
            note,
        )
        se = np.sqrt(ae.expected * (1 - ae.expected / ae.clonotypes) / 1000) * 2  # MC s.e. of both chance estimates
        add(f"TR4 expected matching alphas by chance, remerge={remerge}", ae.expected, t.null_n_shared,
            f"{note}; tolerance 4.5 MC s.e. = {4.5 * se:.2g}", atol=4.5 * se, info=remerge)  # fmt: skip
        add(f"TR4 one-sided p at the Monte-Carlo floor (1 = yes), remerge={remerge}", 1,
            int(t.p_value <= 1.5 / (t.n_perm + 1)), f"{note}; archive p = {ae.p_one_sided:.1e} (10,000 draws)")  # fmt: skip
        log(f"TR4 remerge={remerge}: {t.n_shared} / {t.n_tested} vs chance {t.null_n_shared:.3f}, p {t.p_value:.4g}")

    # ---------------------------------------------------------------- 3 TR1 locus counts and 4 locus-level floors
    r = qc.read_floor(air, floors=(2, 5, 10, 20, 50, 100), radius_um=15.0, n_perm=200, rng=np.random.default_rng(5),
                      **LEGACY)  # fmt: skip
    lo = r.loci
    member = lo[lo.reads_max >= 20]
    sz = member.groupby("clone_id").size()
    add("TR1 TRB loci (15 um)", tr1["TRB loci"], len(lo))
    add("TR1 TRB member loci (>= 20 reads)", tr1["TRB member loci (>= 20 reads)"], len(member))
    for k in (2, 3, 5):
        add(f"TR1 TRB clonotypes with >= {k} member loci", tr1[f"TRB clonotypes with >= {k} member loci"],
            int((sz >= k).sum()))  # fmt: skip
    info = "improved (loci of 15 um, re-merged shuffles, 200 shuffles); archived value is the bin-level number"
    for t in r.table.itertuples():
        side = "above" if t.side == "above" else "below"
        key = ("floor", int(t.floor), side, 30) if side == "above" else ("first", int(t.floor), side, 50)
        n_t, share, shuf = DIAG.get(key, (np.nan, np.nan, np.nan))
        tag = f"locus-level {'>=' if side == 'above' else '<'} {t.floor:g} reads"
        add(f"{tag}: clonotypes tested", n_t, t.n_tested, info, info=True)
        add(f"{tag}: share sharing a TRA clonotype", share, t.statistic, info, info=True)
        add(f"{tag}: shuffled mean", shuf, t.null_mean, f"{info}; p = {t.p_value:.3g}", info=True)
    add("locus-level suggested floor (largest floor with chance below and excess above)", 20,
        r.summary["suggested_floor_unadjusted"], "archive chose 20 reads (TR1 docstring); legacy rule (raw p), sinks kept",
        info=True)  # fmt: skip
    log(r.table[["floor", "side", "n_tested", "n_shared", "statistic", "null_mean", "p_value"]].to_string(index=False))
    log(f"suggested floor (legacy rule) {r.summary['suggested_floor_unadjusted']}")
    rd = qc.read_floor(air, floors=(2, 5, 10, 20, 50, 100), radius_um=15.0, n_perm=200, rng=np.random.default_rng(5))
    add("locus-level suggested floor, package default (sinks removed, BH across floors)", 20, rd.summary["suggested_floor"],
        f"improved (review M6): {rd.summary['n_sink_spots']} sink bins removed; above-side q = "
        + ", ".join(f"{q:.3g}" for q in rd.table[rd.table.side == "above"].q_value), info=True)  # fmt: skip
    log(f"package default: suggested floor {rd.summary['suggested_floor']} ({rd.summary['n_sink_spots']} sinks removed)")

    R = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
    R.to_csv(OUT / "m2_read_floor.csv", index=False)
    checked = R[R.match.notna()]
    log(
        f"{int(checked.match.astype(bool).sum())} / {len(checked)} archived quantities match ({len(R) - len(checked)} information rows)"
    )
    print(checked[~checked.match.astype(bool)].to_string(index=False), flush=True)
    return 0 if checked.match.astype(bool).all() else 1


if __name__ == "__main__":
    sys.exit(main())
