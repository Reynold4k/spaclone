"""M8 validation: germline-rooted lineage trees for the StereoIR L4 tip sets, against the dowser trees of L5.

For both clone groupings (E = exact V|J|junction aa, L = Dandelion-like) the L4 tips (one tip = one locus x one
noise-cleaned full-length sequence, IMGT-gapped, clonal germline with the D / N region masked) are given to
bcr.build_trees (pure Python parsimony: stepwise addition + SPR + ratchet; and neighbour joining) with dowser's tip
handling: tips with an in-frame stop codon left out, identical sequences collapsed only when isotype and location are
also identical (traits), N / gap characters matching any base (alakazam collapseDuplicates).
The dowser parsimony trees (phangorn pratchet, ACCTRAN lengths in mutations) are read from L5_nodes_*.tsv with
bcr.read_trees.

Exact (must match): clones with a tree, collapsed tree tips, the partition of L4 tips into tree tips, every tip's
mutations from the germline (IMGT V 1-312, = L4 'mut') through bcr.tip_table, determinism.
Reported honestly (no exact target; dowser is the reference): parsimony length of our tree vs the dowser tree scored
on the same sequences (bcr.parsimony_score), normalised Robinson-Foulds distance after collapsing zero-length
branches and removing the germline (as L6 compared IgPhyML with parsimony), and the correlation of germline-to-tip
divergence; the IgPhyML-vs-dowser numbers of L5 are shown as the scale of disagreement between two good methods.
Output: validation/results/m8_lineage.csv (quantity, archived, package, match, note)
Usage : python -I m8_lineage.py [n_jobs]   (SPACLONE_SRC=<dev src> to test a scratch copy)
"""

import hashlib
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
OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_CLONES = int(os.environ.get("M8_MAX_CLONES", "0")) or None  # debugging only
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", tol=0.0):
    a, p = float(archived), float(package)
    match = (a == p) or (np.isfinite(a) and np.isfinite(p) and abs(a - p) <= tol * max(1.0, abs(a)))
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})
    log(f"{quantity}: archived {archived} package {package} {'OK' if match else 'MISMATCH'}")


def info(quantity, archived, package, note):
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": np.nan, "note": note})
    log(f"{quantity}: archived {archived} package {package} ({note})")


def canonical(tree, members, rank):
    """Copy of a tree whose tip labels are the first member (in L4 order) of each collapsed tip."""
    lab = []
    for x in tree.label:
        if x is None or x == bcr.GERMLINE:
            lab.append(x)
        else:
            ms = members.get(x, [x])
            lab.append(min(ms, key=lambda t: rank.get(t, 10**9)))
    return bcr.LineageTree(tree.clone_id, tree.parent.copy(), tree.length.copy(), lab, method=tree.method)


log(f"spaclone {spaclone.__version__} from {spaclone.__file__}")
src = Path(spaclone.__file__).resolve().parent
log(f"md5 bcr/lineage.py {hashlib.md5((src / 'bcr' / 'lineage.py').read_bytes()).hexdigest()}")
for tag in ("E", "L"):
    tips = pd.read_csv(LRES / f"L4_tips_{tag}.tsv", sep="\t")
    if MAX_CLONES:
        keep = sorted(tips.clone_id.unique())[:MAX_CLONES]
        tips = tips[tips.clone_id.isin(keep)].reset_index(drop=True)
    nodes = pd.read_csv(LRES / f"L5_nodes_{tag}.tsv", sep="\t")
    nodes = nodes[nodes.clone_id.isin(tips.clone_id.unique())]
    t5 = pd.read_csv(LRES / f"L5_tips_{tag}.tsv", sep="\t")
    t5 = t5[t5.clone_id.isin(tips.clone_id.unique())]
    rank = {t: k for k, t in enumerate(tips.sequence_id.astype(str))}
    kw = dict(clone_key="clone_id", id_key="sequence_id", sequence_key="sequence_alignment",
              germline_key="germline_alignment_d_mask", traits=("isotype", "location"))  # fmt: skip
    t1 = time.time()
    mine = bcr.build_trees(tips, method="parsimony", rng=20261030, n_jobs=N_JOBS, **kw)
    log(f"[{tag}] parsimony trees for {len(mine)} clones in {time.time() - t1:.1f}s")
    t1 = time.time()
    nj = bcr.build_trees(tips, method="nj", rng=20261030, n_jobs=N_JOBS, **kw)
    log(f"[{tag}] NJ trees in {time.time() - t1:.1f}s")
    dow = bcr.read_trees(nodes, clone_key="clone_id", length_key="branch_mut", members_key="tip_ids")
    add(f"L5 {tag}: clones with a tree", nodes.clone_id.nunique(), len(mine), "dowser minseq 2 after collapsing")
    n_tips_mine = sum(t.n_tips - 1 for t in mine.values())
    add(f"L5 {tag}: collapsed tree tips (germline excluded)", len(t5), n_tips_mine,
        "identical sequence + identical isotype and location")  # fmt: skip
    same_part = 0
    for c, t in mine.items():
        a = {frozenset(v) for v in t.members.values()}
        b = {frozenset(v) for v in dow[c].members.values()} if c in dow else set()
        same_part += a == b
    add(f"L5 {tag}: clones whose L4 tips collapse into the same tree tips as dowser", len(mine), same_part)
    tt = bcr.tip_table(mine, positions=(1, 312), x_key="x", y_key="y")
    m4 = tips.set_index("sequence_id").mut
    in_dowser = {x for v in t5.tip_ids for x in str(v).split(",")}
    add(f"L4 {tag}: tips in the tip table (= L4 tips in the dowser trees)", len(in_dowser), len(tt),
        f"{len(tips) - len(in_dowser)} L4 tips left out by dowser (in-frame stop codon), here too")  # fmt: skip
    add(f"L4 {tag}: tip-table tips identical to the L4 tips held by the dowser trees (1 = yes)", 1,
        int(set(tt.tip_id) == in_dowser))  # fmt: skip
    add(f"L4 {tag}: tips whose mutations from the germline (IMGT 1-312) differ from L4 'mut'", 0,
        int((tt.mutations.to_numpy() != m4.loc[tt.tip_id].to_numpy()).sum()))  # fmt: skip
    info(f"L4 {tag}: tip table columns", np.nan, len(tt.columns), ", ".join(tt.columns))
    again = bcr.build_trees(tips, method="parsimony", rng=20261030, n_jobs=1 if N_JOBS > 1 else 2, **kw)
    add(f"L5 {tag}: same trees with the same seed and another n_jobs (1 = yes)", 1,
        int(all(again[c].to_newick() == mine[c].to_newick() for c in mine)))  # fmt: skip
    # ---- topology and length against dowser
    recs = []
    seqs = dict(zip(tips.sequence_id.astype(str), tips.sequence_alignment, strict=True))
    for c, t in mine.items():
        if c not in dow:
            continue
        g = tips.loc[tips.clone_id == c, "germline_alignment_d_mask"].iloc[0]
        s_m = {lab: seqs[lab] for lab in t.tips() if lab != bcr.GERMLINE}
        s_m[bcr.GERMLINE] = g
        d = dow[c]
        s_d = {}
        for lab in d.tips():
            if lab == bcr.GERMLINE:
                s_d[lab] = g
            else:
                ms = d.members.get(lab, [lab])
                s_d[lab] = seqs[ms[0]]
        a = canonical(t, t.members, rank)
        b = canonical(d, d.members, rank)
        aj = canonical(nj[c], nj[c].members, rank) if c in nj else None
        cm = bcr.compare_trees(a, b)
        cj = bcr.compare_trees(aj, b) if aj is not None else {"rf_norm": np.nan, "div_cor": np.nan}
        recs.append({"clone": c, "tips": cm["tips"], "len_mine": bcr.parsimony_score(t, s_m),
                     "len_dowser": bcr.parsimony_score(d, s_d), "len_dowser_acctran": d.score,
                     "len_nj": bcr.parsimony_score(nj[c], s_m) if c in nj else np.nan,
                     "rf_norm": cm["rf_norm"], "div_cor": cm["div_cor"], "rf_norm_nj": cj["rf_norm"],
                     "div_cor_nj": cj["div_cor"]})  # fmt: skip
    R = pd.DataFrame(recs)
    R.to_csv(OUT / f"m8_lineage_per_tree_{tag}.csv", index=False)
    add(f"L5 {tag}: our parsimony length = sum of our branch lengths (all trees) (1 = yes)", 1,
        int(np.allclose(R.len_mine.to_numpy(), [mine[c].score for c in R.clone])))  # fmt: skip
    info(f"L5 {tag}: trees where our parsimony tree is as short or shorter than dowser's", len(R),
         int((R.len_mine <= R.len_dowser).sum()),
         f"strictly shorter {int((R.len_mine < R.len_dowser).sum())}, longer {int((R.len_mine > R.len_dowser).sum())}; "
         f"total length ours {R.len_mine.sum():.0f} vs dowser {R.len_dowser.sum():.0f} (scored on the same "
         f"sequences; dowser's own ACCTRAN lengths sum to {R.len_dowser_acctran.sum():.1f})")  # fmt: skip
    ig = pd.read_csv(LRES / f"L5_igphyml_compare_{tag}.tsv", sep="\t")
    info(f"L5 {tag}: median normalised RF, ours vs dowser parsimony (archived: IgPhyML vs dowser)",
         float(ig.rf_norm.median()), float(R.rf_norm.median()),
         f"mean ours {R.rf_norm.mean():.3f} (IgPhyML {ig.rf_norm.mean():.3f}); identical topology after collapsing "
         f"zero-length branches in {int((R.rf_norm == 0).sum())} of {len(R)} trees (IgPhyML "
         f"{int((ig.rf_norm == 0).sum())} of {int(ig.rf_norm.notna().sum())})")  # fmt: skip
    info(f"L5 {tag}: median correlation of germline-to-tip divergence, ours vs dowser (archived: IgPhyML vs dowser)",
         float(ig.div_cor.median()), float(R.div_cor.median()),
         f"mean ours {R.div_cor.mean():.3f} (IgPhyML {ig.div_cor.mean():.3f})")  # fmt: skip
    # ---- a more thorough search on the trees the default search left longer than dowser's
    longer = R.loc[R.len_mine > R.len_dowser, "clone"].tolist()
    if longer:
        sub = tips[tips.clone_id.isin(longer)]
        deep = bcr.build_trees(sub, method="parsimony", rng=20261030, n_jobs=N_JOBS, n_starts=8, n_ratchet=30, **kw)
        dl = {c: bcr.parsimony_score(deep[c], {**seqs, bcr.GERMLINE: sub.loc[sub.clone_id == c,
              "germline_alignment_d_mask"].iloc[0]}) for c in deep}  # fmt: skip
        rl = R.set_index("clone").loc[longer]
        info(f"L5 {tag}: trees longer than dowser's after a thorough search (8 starts x 30 ratchet iterations)",
             0, int(sum(dl[c] > rl.loc[c, "len_dowser"] for c in longer)),
             f"of the {len(longer)} trees longer with the defaults; their total length "
             f"{rl.len_mine.sum():.0f} -> {sum(dl.values()):.0f} (dowser {rl.len_dowser.sum():.0f})")  # fmt: skip
    # ---- downstream: the within-tree Spearman test of L7 (spaclone.tl.tree_tissue_distance) on both tree sets
    from spaclone import tl

    if hasattr(tl, "tree_tissue_distance"):
        L7 = pd.read_csv(LRES / "L7_tests.csv")
        a7 = L7[(L7.test == "C") & (L7.grouping == tag)].iloc[0]
        kwc = dict(tree_key="clone_id", tip_key="sequence_id", unit_key="loc_id", min_tips=5, min_units=2,
                   min_pairs=3, shuffle="tips", n_perm=1000)  # fmt: skip
        td = bcr.tip_table(dow, tips, x_key="x", y_key="y")
        rd = tl.tree_tissue_distance(td, dow.newick, rng=np.random.default_rng([11, ord(tag)]), **kwc)
        add(f"L7 C {tag}: trees tested, dowser trees through bcr.read_trees + bcr.tip_table", int(a7.clones),
            rd.summary["n_trees"])  # fmt: skip
        add(f"L7 C {tag}: mean Spearman(patristic, tissue distance), dowser trees through bcr.read_trees + "
            f"bcr.tip_table", float(a7.observed), rd.summary["statistic"], "tl.tree_tissue_distance", tol=1e-9)  # fmt: skip
        rm = tl.tree_tissue_distance(tt, mine.newick, rng=np.random.default_rng([11, ord(tag)]), **kwc)
        info(f"L7 C {tag}: mean Spearman(patristic, tissue distance) with our parsimony trees", float(a7.observed),
             rm.summary["statistic"],
             f"p = {rm.summary['p_value']:.3g} (dowser trees: archived p = {float(a7.p):.3g}); trees tested "
             f"{rm.summary['n_trees']}")  # fmt: skip
    info(f"L5 {tag}: NJ trees: median normalised RF vs dowser", float(ig.rf_norm.median()),
         float(R.rf_norm_nj.median()),
         f"NJ length >= parsimony length in {int((R.len_nj >= R.len_mine).sum())} of {len(R)}; median divergence "
         f"correlation {R.div_cor_nj.median():.3f}")  # fmt: skip

res = pd.DataFrame(rows)
res.to_csv(OUT / "m8_lineage.csv", index=False)
req = res[res.match.notna()]
log(f"wrote {OUT / 'm8_lineage.csv'}: {int(req.match.astype(bool).sum())}/{len(req)} required rows match, "
    f"{int(res.match.isna().sum())} reported rows")  # fmt: skip
