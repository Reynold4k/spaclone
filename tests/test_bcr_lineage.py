import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, pl

_SENSE = [c for c, aa in bcr.CODON_TABLE.items() if aa != "*"]


def _simulate(rng, n_tips=20, length=240, rate=1.5, clone="c1"):
    """Random binary genealogy below a germline of sense codons; Poisson mutations per branch (never creating a stop
    codon). Returns the tips, the true tree and its number of mutations."""
    germ = "".join(rng.choice(_SENSE, length // 3))
    parent, seqs = [-1], [germ]
    tips = []

    def mutate(s, k):
        s = list(s)
        for p in rng.choice(length, size=k, replace=False):
            c0 = p - p % 3
            alts = [
                b for b in "ACGT" if b != s[p] and bcr.CODON_TABLE["".join(s[c0:p] + [b] + s[p + 1 : c0 + 3])] != "*"
            ]
            if alts:
                s[p] = rng.choice(alts)
        return "".join(s)

    stack = [(0, n_tips)]
    while stack:
        par, n = stack.pop()
        node = len(parent)
        parent.append(par)
        seqs.append(mutate(seqs[par], int(rng.poisson(rate))))
        if n == 1:
            tips.append(node)
        else:
            a = int(rng.integers(1, n))
            stack.extend([(node, a), (node, n - a)])
    labels = [None] * len(parent)
    for k, v in enumerate(tips):
        labels[v] = f"{clone}_t{k}"
    length_ = [0.0] + [
        float(sum(x != y for x, y in zip(seqs[v], seqs[parent[v]], strict=True))) for v in range(1, len(parent))
    ]
    true = bcr.LineageTree(clone, parent + [0], length_ + [0.0], labels + [bcr.GERMLINE])
    df = pd.DataFrame(
        {
            "sequence_id": [labels[v] for v in tips],
            "clone_id": clone,
            "sequence_alignment": [seqs[v] for v in tips],
            "germline_alignment_d_mask": germ,
            "x_um": rng.uniform(0, 500, len(tips)),
            "y_um": rng.uniform(0, 500, len(tips)),
            "isotype": rng.choice(["IgM", "IgG"], len(tips)),
        }
    )
    return df, true, sum(length_)


def test_parsimony_recovers_simulated_trees():
    rng = np.random.default_rng(3)
    frames, truth = [], {}
    for k, n in enumerate((8, 25, 45)):
        df, true, total = _simulate(rng, n_tips=n, clone=f"c{k}")
        frames.append(df)
        truth[f"c{k}"] = (true, total)
    tips = pd.concat(frames, ignore_index=True)
    trees = bcr.build_trees(tips, rng=1)
    assert sorted(trees) == ["c0", "c1", "c2"]
    for c, t in trees.items():
        true, total = truth[c]
        assert t.score <= total  # never longer than the true number of mutations
        cmp = bcr.compare_trees(t, true)
        assert cmp["rf_norm"] <= 0.2 and cmp["div_cor"] > 0.9
        assert t.label[1] == bcr.GERMLINE and t.length[1] == 0 and t.parent[1] == 0
    nj = bcr.build_trees(tips, method="nj", rng=1)
    for c, t in nj.items():
        assert bcr.compare_trees(t, truth[c][0])["rf_norm"] <= 0.35
        assert t.score >= trees[c].score  # NJ is never shorter than the parsimony search
    s = trees.summary()
    distinct = tips.groupby("clone_id").sequence_alignment.nunique()
    assert list(s.clone_id) == ["c0", "c1", "c2"] and (s.tips.to_numpy() == distinct.loc[s.clone_id].to_numpy()).all()


def test_tip_table_mutations_depth_and_positions():
    rng = np.random.default_rng(4)
    df, true, _ = _simulate(rng, n_tips=12)
    trees = bcr.build_trees(df, rng=0)
    tt = bcr.tip_table(trees)
    assert len(tt) == 12 and set(tt.columns) >= {"clone_id", "sequence_id", "tip_id", "depth", "mutations", "x_um"}
    germ = df.germline_alignment_d_mask.iloc[0]
    expect = [sum(a != b for a, b in zip(s, germ, strict=True)) for s in df.sequence_alignment]
    assert tt.set_index("tip_id").mutations.loc[df.sequence_id].tolist() == expect
    assert (tt.depth >= tt.mutations).all()  # path length >= Hamming distance (homoplasy adds)
    win = bcr.tip_table(trees, positions=(1, 60))
    assert (win.mutations <= tt.mutations).all()
    assert tt.isotype.notna().all()


def test_collapse_rules_traits_ambiguity_and_stop_codons():
    germ = "ATGGCTGCTGCTGCTGCTGCT"
    rows = [
        ("t1", "ATGGCAGCTGCTGCTGCTGCT", "IgM"),
        ("t2", ".TGGCAGCTGCTGCTGCTGCT", "IgM"),  # 5'-padded copy of t1: same tip under the dowser rule
        ("t3", "ATGGCAGCTGCTGCTGCTGCT", "IgG"),  # identical to t1 but another isotype: own tip
        ("t4", "ATGGCTGCAGCTGCTGCTGCT", "IgM"),
        ("t5", "ATGGCTGCTTAAGCTGCTGCT", "IgM"),  # in-frame stop codon: left out (dowser filterstop)
        ("t6", "ATGGCTGCTGCTGCTGCAGCT", "IgM"),
    ]
    tips = pd.DataFrame(rows, columns=["sequence_id", "sequence_alignment", "isotype"]).assign(
        clone_id="c", germline_alignment_d_mask=germ
    )
    t = bcr.build_trees(tips, traits=("isotype",), rng=0)["c"]
    members = {frozenset(v) for v in t.members.values()}
    assert frozenset({"t1", "t2"}) in members and frozenset({"t3"}) in members
    assert "t5" not in {m for v in t.members.values() for m in v}
    assert t.n_tips - 1 == 4
    tr = bcr.build_trees(tips, traits=("isotype",), rng=0)
    assert pd.isna(tr.tips.set_index("sequence_id").loc["t5", "tree_label"])
    exact = bcr.build_trees(tips, traits=("isotype",), collapse="exact", rng=0)["c"]
    assert exact.n_tips - 1 == 5
    keep = bcr.build_trees(tips, collapse="none", filter_stop=False, rng=0)["c"]
    assert keep.n_tips - 1 == 6
    anyt = bcr.build_trees(tips, rng=0)["c"]  # no traits: t1, t2 and t3 collapse
    assert frozenset({"t1", "t2", "t3"}) in {frozenset(v) for v in anyt.members.values()}
    with pytest.raises(ValueError):
        bcr.build_trees(tips, collapse="maybe")


def test_ambiguous_sequence_compatible_with_two_others_is_discarded():
    germ = "ATGGCTGCTGCT"
    tips = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "x"],
            "sequence_alignment": ["ATGGCTGCAGCT", "ATGGCTGCCGCT", "ATGGCTGCNGCT"],
            "clone_id": "c",
            "germline_alignment_d_mask": germ,
        }
    )
    tr = bcr.build_trees(tips, rng=0)
    assert pd.isna(tr.tips.set_index("sequence_id").loc["x", "tree_label"])
    assert tr["c"].n_tips - 1 == 2


def test_newick_round_trip_node_tables_and_scoring():
    rng = np.random.default_rng(5)
    df, _, _ = _simulate(rng, n_tips=15)
    trees = bcr.build_trees(df, rng=2)
    t = trees["c1"]
    back = bcr.read_trees(trees.newick)["c1"]
    cmp = bcr.compare_trees(t, back, min_length=0.0)
    assert cmp["rf"] == 0 and cmp["div_cor"] == pytest.approx(1.0)
    np.testing.assert_allclose(np.sort(back.depths()[[k for k, x in enumerate(back.label) if x]]),
                               np.sort(t.depths()[[k for k, x in enumerate(t.label) if x]]))  # fmt: skip
    seqs = dict(zip(df.sequence_id, df.sequence_alignment, strict=True))
    seqs[bcr.GERMLINE] = df.germline_alignment_d_mask.iloc[0]
    assert bcr.parsimony_score(back, seqs) == t.score
    edges = trees.edges()
    tab = edges.rename(columns={"length": "branch_mut"})
    tab["tip_ids"] = tab.label.map(lambda x: ",".join(t.members.get(x, [])) if isinstance(x, str) else "")
    rt = bcr.read_trees(tab, clone_key="clone_id")
    assert rt["c1"].members == dict(t.members)
    assert bcr.compare_trees(rt["c1"], t)["rf"] == 0
    tt = bcr.tip_table(rt, df, positions=None)
    assert len(tt) == 15
    single = tt.n_collapsed == 1
    assert (tt.sequence_id[single] == tt.tip_id[single]).all()


def test_parsimony_score_known_answer_and_polytomy():
    # ((a,b),(c,d)) with a=b=A, c=d=C at one site: 1 change; star (a,b,c,d): 2 changes (Hartigan)
    t = bcr.read_trees("((a:1,b:1):1,(c:1,d:1):1);")["tree"]
    seqs = {"a": "A", "b": "A", "c": "C", "d": "C"}
    assert bcr.parsimony_score(t, seqs) == 1
    star = bcr.read_trees("(a,b,c,d);")["tree"]
    assert bcr.parsimony_score(star, seqs) == 2
    assert bcr.parsimony_score(t, {"a": "A", "b": "N", "c": "C", "d": "C"}) == 1


def test_determinism_and_clone_independence():
    rng = np.random.default_rng(6)
    a, _, _ = _simulate(rng, n_tips=20, clone="a")
    b, _, _ = _simulate(rng, n_tips=20, clone="b")
    both = bcr.build_trees(pd.concat([a, b], ignore_index=True), rng=9)
    alone = bcr.build_trees(a, rng=9)
    assert both["a"].to_newick() == alone["a"].to_newick()  # seeds derive from the clone id
    assert bcr.build_trees(a, rng=9)["a"].to_newick() == alone["a"].to_newick()
    with pytest.raises(ValueError, match="unique"):
        bcr.build_trees(pd.concat([a, a]))
    with pytest.raises(KeyError):
        bcr.build_trees(a.drop(columns="germline_alignment_d_mask"))


def test_tree_tissue_plot():
    rng = np.random.default_rng(7)
    df, _, _ = _simulate(rng, n_tips=10)
    trees = bcr.build_trees(df, rng=0)
    tt = bcr.tip_table(trees)
    fig = pl.tree_tissue(trees["c1"], tt, color_key="isotype", background=tt, style="dark")
    assert len(fig.axes) == 2
    with pytest.raises(KeyError):
        pl.tree_tissue(trees["c1"], tt.drop(columns="x_um"))
