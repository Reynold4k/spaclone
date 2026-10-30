"""M8 validation: somatic hypermutation on AIR-SPACE D21 from MiXCR alignments, with package functions only.

Reference numbers: tutorial 03 (docs/tutorials/03_bcr_class_switching_and_mutation.ipynb, executed 2026-10-30), which
parsed MiXCR's allVAlignments by hand (first read of each IGH molecule with a V alignment; aligned V = germline end -
start; mismatches = substitutions A/C/G/T -> A/C/G/T; no allele baseline; evaluable from 100 aligned nt; mutated =
excess >= 3) and found the GEX-called germinal centres 1.32x richer in mutated loci than their isotype mix predicts
(58 of 129 evaluable GC loci vs 44.1 expected, p = 0.0057, 10,000 within-isotype shuffles, rng 0).

Package route (no hand parsing):
  io.read_airspace(alignments=True) -> pp.filter_records(on_tissue, IGH) -> bcr.isotype -> bcr.shm(mixcr_alignment=
  "v_alignment_mixcr", min_reads=1, min_aligned_nt=100; baseline=None as the tutorial, and baseline="allele" with
  baseline_fallback="zero" / "pooled") -> pp.merge_loci (15 um) -> bcr.call_isotype / bcr.call_shm (by locus) ->
  bcr.region_test (blocks=None: the tutorial's unit design; and the new clone-level default).
Differences from the tutorial are reported, not hidden: the package keeps the alignment of each molecule's
representative read (modal CDR3; the tutorial took the first read with a V alignment) and counts aligned positions as
germline span minus deletions (the AIRR definition; the tutorial used the span).
Output: validation/results/m8_airspace_shm.csv (quantity, archived, package, match, note)
Usage : python -I m8_airspace_shm.py   (SPACLONE_SRC=<dev src> to test a scratch copy)
"""

import os
import re
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_SRC"):
    sys.path.insert(0, os.environ["SPACLONE_SRC"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import spaclone  # noqa: E402
from spaclone import bcr, gex, io, pp  # noqa: E402

DATA = Path(os.environ["SPACLONE_ARCHIVE"]) / "AIRSPACE_GSE286452" / "raw"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
rows = []
TUT = {"loci_called": 1852, "loci_mutated": 635, "gc_units": 129, "gc_marked": 58, "gc_expected": 44.105538,
       "gc_ratio": 1.315028, "gc_p": 0.005699, "n_baseline_genes": 5, "n_genes": 58}  # fmt: skip


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol * max(1.0, abs(a)))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: tutorial {archived} package {package} {'OK' if match else 'DIFFERENT'}")


def info(quantity, archived, package, note):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: tutorial {archived} package {package} ({note})")


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
h5ad, clone_table = spaclone.datasets.airspace("D21", data_dir=DATA, download=False)
sa = io.read_airspace(h5ad, clone_table, alignments=True)
G = sa.gex
gex.call_regions(G, kind="germinal_centre")
gex.call_regions(G, kind="plasma_cell")
igh = pp.filter_records(sa.air, on_tissue=True, loci="IGH").air.reset_index(drop=True)
igh = bcr.isotype(igh, species="mouse")
igh["region"] = igh["spot_id"].map(G.obs["region"].astype(str)).to_numpy()
add("IGH molecules on tissue (valid)", 2542, len(igh), "same reader and filter as the tutorial")
info("IGH molecules with a MiXCR V alignment", np.nan, int(igh["v_alignment_mixcr"].notna().sum()),
     "alignment of the molecule's representative read")  # fmt: skip

# ---- molecule-level SHM: package parse vs the tutorial's hand parse of the same strings
pv = io.parse_mixcr_alignment(igh["v_alignment_mixcr"])
sub_rx = re.compile(r"S([ACGT])(\d+)([ACGT])")


def hand(s):
    if not isinstance(s, str) or not s:
        return np.nan, np.nan
    f = s.split(";", 1)[0].split("|")
    return int(f[1]) - int(f[0]), len(sub_rx.findall(f[5]))


hp = np.array([hand(s) for s in igh["v_alignment_mixcr"]], dtype=float)
ok = np.isfinite(hp[:, 0])
add("substitutions per molecule: package parse = hand parse (molecules differing)", 0,
    int((pv["substitutions"].to_numpy(dtype=float)[ok] != hp[ok, 1]).sum()), "same alignment strings")  # fmt: skip
span_minus_del = (pv["ref_end"] - pv["ref_start"] - pv["deletions"]).to_numpy(dtype=float)
add("aligned nt = germline span - deletions (molecules differing)", 0,
    int((pv["aligned_nt"].to_numpy(dtype=float)[ok] != span_minus_del[ok]).sum()), "AIRR definition")  # fmt: skip
info("molecules whose evaluability (>= 100 nt) differs between span and span - deletions", 0,
     int(((hp[ok, 0] >= 100) != (span_minus_del[ok] >= 100)).sum()),
     f"of {int(ok.sum())} molecules with an alignment; deletions are nanopore homopolymer errors mostly")  # fmt: skip

for name, kw in (
    ("tutorial-equivalent (baseline=None)", {"baseline": None}),
    ("allele baseline, fallback 'zero'", {"baseline": "allele", "baseline_fallback": "zero"}),
    ("allele baseline, fallback 'pooled'", {"baseline": "allele", "baseline_fallback": "pooled"}),
    ("allele baseline, no fallback (T1 rule)", {"baseline": "allele", "baseline_fallback": "none"}),
):
    d, tab_d = bcr.shm(
        igh, mixcr_alignment="v_alignment_mixcr", min_reads=1, min_aligned_nt=100, min_excess=3, return_baseline=True, **kw
    )
    if kw.get("baseline") == "allele":
        tab = tab_d
        src = d["shm_baseline_source"].value_counts().to_dict()
        if kw["baseline_fallback"] == "none":
            add("V genes with >= 10 evaluable IgM / IgD molecules (allele baseline)", TUT["n_baseline_genes"],
                int((tab.source == "allele").sum()), f"of {len(tab)} V genes present; tutorial: 5 of 58")  # fmt: skip
        info(f"[{name}] molecules by baseline source", np.nan, sum(src.values()),
             ", ".join(f"{k} {v}" for k, v in sorted(src.items())))  # fmt: skip
    loci, member = pp.merge_loci(d, radius_um=15)
    d["locus_id"] = member.to_numpy()
    units = loci.join(bcr.call_isotype(d, by="locus").drop(columns="n_records"), on="locus_id")
    gc_id = units["spot_id"].map(G.obs["region_id"]).astype("string")
    units["gc_number"] = gc_id.str.removeprefix("GC_").astype("float").add(1).fillna(0).astype(int)
    calls = bcr.call_shm(d, by="locus", min_excess=3)
    units = units.join(calls[["n_evaluable", "per100", "mutated"]], on="locus_id")
    tag = f"[{name}]"
    called = int(units["mutated"].notna().sum())
    mutated = int(units["mutated"].fillna(False).astype(bool).sum())
    if kw.get("baseline") is None:
        note = "within 1-5%: the package uses each molecule's modal-CDR3 read, the tutorial its first read"
        add(f"{tag} loci with a mutation call", TUT["loci_called"], called, note, tol=0.01)
        add(f"{tag} mutated loci", TUT["loci_mutated"], mutated, note, tol=0.02)
    else:
        info(f"{tag} loci with a mutation call / mutated", TUT["loci_called"], called, f"mutated {mutated}")
    rt = bcr.region_test(units, regions=units["gc_number"].to_numpy(), mark="mutated", match_by="isotype",
                         blocks=None, rng=0)  # fmt: skip
    pooled = rt.table.set_index(rt.table.region.astype(str)).loc["all"]
    rt_c = bcr.region_test(units, regions=units["gc_number"].to_numpy(), mark="mutated", match_by="isotype", rng=0)
    pc = rt_c.table.set_index(rt_c.table.region.astype(str)).loc["all"]
    if kw.get("baseline") is None:
        add(f"{tag} GC loci evaluable (pooled)", TUT["gc_units"], pooled.units, note, tol=0.03)
        add(f"{tag} GC loci mutated (pooled)", TUT["gc_marked"], pooled.marked, note, tol=0.05)
        add(f"{tag} GC isotype-matched expectation", TUT["gc_expected"], pooled.expected, note, tol=0.05)
        add(f"{tag} GC ratio to the isotype-matched expectation", TUT["gc_ratio"], pooled.ratio, note, tol=0.05)
        info(f"{tag} GC excess p (blocks=None: loci shuffled within isotype, 10,000)", TUT["gc_p"], pooled.p_value,
             "same design and seed as the tutorial; the inputs differ slightly (representative read of a molecule)")
        add(f"{tag} GC excess significant at 0.05 with loci shuffled (1 = yes)", 1, int(pooled.p_value <= 0.05), "")
    else:
        info(f"{tag} GC pooled mutated / expected / ratio / p (loci shuffled)", TUT["gc_ratio"], pooled.ratio,
             f"{int(pooled.marked)} of {int(pooled.units)} vs {pooled.expected:.2f}, p {pooled.p_value:.4f}")  # fmt: skip
    info(f"{tag} GC pooled p, clone-level shuffles (new default)", TUT["gc_p"], pc.p_value,
         f"null mean {pc.null_mean:.2f} (isotype- and clone-size-matched), sd {pc.null_sd:.2f}; observed "
         f"{int(pc.marked)}, isotype-matched expectation {pc.expected:.2f}")  # fmt: skip

res = pd.DataFrame(rows)
res.to_csv(OUT / "m8_airspace_shm.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm8_airspace_shm.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")  # fmt: skip
