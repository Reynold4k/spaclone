"""M7 validation: the transcriptome workflow end to end on one AIR-SPACE lymph node (GSE286452, D14).

Package functions only: call_tissue -> qc_metrics -> filter -> preprocess -> cluster -> embed -> score_cell_types
-> call_regions (GC, ASC) -> domains -> rank_genes_regions (GC vs other, GC instances and background tiles as
replicates) -> nhood_enrichment.  No archived number exists for the clustering, so the check is a rule fixed before
the run (2026-10-30):
  separation: the Leiden clusters with the highest mean B, T and ASC module enrichment (from score_cell_types) are
  three different clusters, and in each of them its own module's mean enrichment is >= 2 (twice the tissue mean)
  and higher than the other two modules' mean enrichments.
Descriptive rows (no pass/fail): adjusted Rand index of Leiden clusters vs marker classes, number of clusters /
domains, GC instances and DE hits.
Output: validation/results/m7_workflow.csv (quantity, archived, package, match, note)
Usage : python -I m7_workflow.py            (SPACLONE_DEV=<src dir> puts a development copy first on sys.path)
"""

import os
import sys
import time
from pathlib import Path

if os.environ.get("SPACLONE_DEV"):
    sys.path.insert(0, os.environ["SPACLONE_DEV"])

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import spaclone  # noqa: E402
from spaclone import gex  # noqa: E402

RAW = Path(os.environ["SPACLONE_ARCHIVE"]) / "AIRSPACE_GSE286452" / "raw"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE, GSM = "D14", "GSM8727833"
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, match, note=""):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": match, "note": note})
    log(f"{quantity}: {package} ({'pass' if match is True else 'FAIL' if match is False else 'descriptive'})")


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
a = ad.read_h5ad(next(RAW.glob(f"{GSM}_adata_{SAMPLE}_*.h5ad")))
g = ad.AnnData(X=sparse.csr_matrix(a.X), obs=pd.DataFrame(index=a.obs["barcode"].astype(str).to_numpy()),
               var=pd.DataFrame(index=a.var_names.astype(str)))
g.obsm["spatial"] = np.column_stack([a.obs["x"].to_numpy(float), a.obs["y"].to_numpy(float)]) * 0.65
del a
gex.call_tissue(g)
qc = gex.qc_metrics(g)
log("QC on/off tissue:\n" + qc.to_string(index=False))
t = gex.filter(g, min_counts=100, off_tissue="drop")
add("tissue beads after filter", np.nan, t.n_obs, None, "call_tissue + filter(off_tissue='drop')")

gex.preprocess(t, n_hvg=2000, n_pcs=30)
cl = gex.cluster(t, resolution=1.0)
gex.embed(t)
add("Leiden clusters (resolution 1)", np.nan, len(cl), None)
types = gex.score_cell_types(t, markers="immune_mouse")
legacy = gex.score_cell_types(t, markers="immune_mouse", min_module_counts=1, key_added="cell_type_x6rule")
log("cell types (default; X6 rule = min_module_counts=1):\n" + types[["cell_type", "n_units", "pct_units"]].assign(
    pct_x6rule=legacy.pct_units.to_numpy()).to_string(index=False))
for c in ("GC_B", "Tfh"):
    add(f"% tissue beads typed {c}: default (min_module_counts=2) vs X6 rule (=1)", np.nan,
        f"{types.set_index('cell_type').loc[c, 'pct_units']:.2f} vs {legacy.set_index('cell_type').loc[c, 'pct_units']:.2f}",
        None, "rare-gene modules win single-molecule beads under the X6 rule")

enr = t.obsm["cell_type_enrichment"][["B", "T", "ASC"]]
means = enr.groupby(t.obs["leiden"].astype(str).to_numpy()).mean()
top = means.idxmax(axis=0)
distinct = top.nunique() == 3
own_ok = True
for m in ("B", "T", "ASC"):
    c = top[m]
    row = means.loc[c]
    ok = row[m] >= 2 and row[m] > row.drop(m).max()
    own_ok &= bool(ok)
    add(f"top cluster of the {m} module", np.nan, f"cluster {c}: {m} {row[m]:.2f}, others "
        f"{', '.join(f'{k} {v:.2f}' for k, v in row.drop(m).items())}", bool(ok), "own enrichment >= 2 and highest")
add("separation rule: B, T and ASC peak in three different clusters (1 = yes)", 1, int(distinct), bool(distinct))
add("separation rule passed (1 = yes)", 1, int(distinct and own_ok), bool(distinct and own_ok))
assigned = t.obs["cell_type"].astype(str) != "unassigned"
add("adjusted Rand index, Leiden vs marker classes (assigned beads)", np.nan,
    round(adjusted_rand_score(t.obs.loc[assigned, "leiden"], t.obs.loc[assigned, "cell_type"]), 4), None)

gc = gex.call_regions(t, kind="germinal_centre")
gex.call_regions(t, kind="plasma_cell")
add("germinal centres (package default rule)", np.nan, len(gc), None)
dom = gex.domains(t, resolution=0.5)
add("spatial domains (BANKSY-style, lambda 0.8, resolution 0.5)", np.nan, len(dom), None,
    f"mean neighbour agreement {np.average(dom.agreement, weights=dom.n_units):.3f}")
de = gex.rank_genes_regions(t, groups="GC", reference="other", test="limma")
top_genes = de.table.head(15).gene.tolist()
add("DE GC vs other (limma, instances as replicates): genes q < 0.05", np.nan, de.summary.get("n_q05[GC]"), None,
    f"top 15: {', '.join(top_genes)}; GC markers define the regions, so their own enrichment is circular")
ne = gex.nhood_enrichment(t, cluster_key="cell_type", radius_um=15.0, n_perm=200, rng=0)
bt = ne.table[(ne.table.cluster_a == "B") & (ne.table.cluster_b == "T")]
add("nhood enrichment B-T z (15 um, 200 relabellings)", np.nan, round(float(bt.z.iloc[0]), 2), None)
res = pd.DataFrame(rows)
res.to_csv(OUT / "m7_workflow.csv", index=False)
log(f"wrote {OUT / 'm7_workflow.csv'}")
