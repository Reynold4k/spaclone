"""Receptor sequences of the synthetic chip with planted B-cell lineages, and lineage trees scored against them.

:func:`spaclone.sim.synthetic_chip` with ``sequences=True`` gives every receptor record the contig it was read from
and its V and J alignments, in the AIRR fields that IgBLAST (``-outfmt 19``) writes, so that
:func:`spaclone.bcr.lineage_workflow` can build lineage trees from it. Everything here is synthetic:

* **Germline genes.** A fixed synthetic repertoire (the same in every simulated section): V genes 1-30 and J genes
  1-4 of every locus, each V gene 285-297 nt of sense codons ending in the conserved cysteine codon ``TGC``, a 30-nt
  FR4 after each J gene's last junction codon, an 18-nt leader before each V gene and 24 nt of every constant region.
  They are not real immunoglobulin genes.
* **Planted lineages.** The cells of a germinal-centre or plasma-cell-focus lineage (and a lone plasma cell) descend
  from one naive B cell along a Kingman coalescent tree. Somatic hypermutation adds Poisson numbers of substitutions on
  every branch: a quarter of ``shm_mutations`` (the mean V mutations of a cell) on the trunk shared by every cell of the
  lineage, the rest spread over the tree in proportion to branch length; light chains mutate at half the heavy-chain
  rate. Mutations hit V positions from the 13th up to the cysteine codon, never create a stop codon and
  leave the junction unchanged, so a lineage stays one exact clone. Naive B cells (follicles, T zone, red pulp) and T
  cells carry their germline.
* **Records.** Every record (genuine, swapped or sink) carries the sequence of the cell its molecule came from, plus
  its own sequencing errors (``seq_error_rate`` per V base, never a stop codon); a ``truncated_fraction`` of the
  contigs start inside the V gene, as 5' truncated assemblies do.

The truth sits next to the records (``true_genotype``, ``true_mutations``, ``seq_errors``, ``seq_error_sites``) and in
the metadata (``meta["genotypes"]``, and ``meta["lineage_trees"]``: the planted heavy-chain tree of every lineage with
at least two cells, in Newick with tips named by genotype). :func:`score_lineages` compares a lineage-workflow result
with it, which real data cannot offer.

References
----------
Kingman 1982, doi:10.1016/0304-4149(82)90011-4 (the coalescent); Robinson & Foulds 1981,
doi:10.1016/0025-5564(81)90043-2 (tree distance); Morris et al. 2019, doi:10.1002/sim.8086 (simulation studies with
known truth).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["score_lineages"]

#: First V position (0-based) that the simulated hypermutation reaches: the 5' end of FR1 stays germline, so every
#: full-length contig starts its V alignment with a match, as a local aligner reports it.
_SHM_START = 12
_LEADER_NT, _FWR4_CODONS, _CONST_NT = 18, 10, 24
_LOCI = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
_STOPS = frozenset({b"TAA", b"TAG", b"TGA"})
_SENSE = np.array(
    [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in ("TAA", "TAG", "TGA")], dtype=object
)
_ACGT = np.frombuffer(b"ACGT", np.uint8)
#: Lineage compartments whose B cells carry somatic hypermutation (lone plasma cells too).
_MUTATED_COMPARTMENTS = ("germinal centre", "plasma-cell focus")


@functools.lru_cache(maxsize=1)
def _germline_set() -> dict[str, dict[str, str]]:
    """The fixed synthetic germline repertoire: V genes (ending in TGC) with leaders, and the FR4 of every J gene."""
    g = np.random.default_rng(20261116)
    v, leader, fr4 = {}, {}, {}
    for locus in _LOCI:
        for k in range(1, 31):
            v[f"{locus}V{k}"] = "".join(g.choice(_SENSE, int(g.integers(95, 100)))) + "TGC"
            leader[f"{locus}V{k}"] = "".join(g.choice(list("ACGT"), _LEADER_NT))
        for k in range(1, 5):
            fr4[f"{locus}J{k}"] = "".join(g.choice(_SENSE, _FWR4_CODONS))
    return {"v": v, "leader": leader, "fr4": fr4}


@functools.lru_cache(maxsize=64)
def _constant_start(c_call: str) -> str:
    """24 nt standing in for the start of a constant region (deterministic per name)."""
    g = np.random.default_rng(list(str(c_call).encode()) or [0])
    return "".join(g.choice(list("ACGT"), _CONST_NT))


def _gene(call: Any) -> str:
    return str(call).split("*", 1)[0]


def _substitute(seq: bytearray, n: int, gen: np.random.Generator, lo: int, hi: int) -> list[tuple[int, int]]:
    """``n`` substitutions at positions in ``[lo, hi)`` that create no stop codon (frame of position 0), in place."""
    out: list[tuple[int, int]] = []
    tries = 0
    while len(out) < n and tries < 50 * n + 50 and hi > lo:
        tries += 1
        p = int(gen.integers(lo, hi))
        choices = _ACGT[_ACGT != seq[p]]
        new = int(choices[gen.integers(0, len(choices))])
        c0 = p - p % 3
        codon = bytearray(seq[c0 : c0 + 3])
        codon[p - c0] = new
        if bytes(codon) in _STOPS:
            continue
        seq[p] = new
        out.append((p, new))
    return out


def _coalescent(n: int, gen: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Kingman coalescent of ``n`` leaves: parent of every node and branch lengths in coalescent time.

    Leaves are ``0..n-1`` and the root (parent -1) comes last; a parent's index is always larger than its children's.
    """
    if n == 1:
        return np.array([-1]), np.array([0.0])
    parent = np.full(2 * n - 1, -1, dtype=np.int64)
    when = np.zeros(2 * n - 1)
    active = list(range(n))
    nxt, t = n, 0.0
    while len(active) > 1:
        k = len(active)
        t += gen.exponential(2.0 / (k * (k - 1)))
        i, j = sorted(int(x) for x in gen.choice(k, 2, replace=False))
        a, b = active[i], active[j]
        parent[a] = parent[b] = nxt
        when[nxt] = t
        del active[j]
        del active[i]
        active.append(nxt)
        nxt += 1
    length = np.where(parent >= 0, when[np.maximum(parent, 0)] - when, 0.0)
    return parent, length


def _newick(labels: list[str | None], parent: list[int], length: list[float]) -> str:
    from spaclone.bcr.lineage import LineageTree

    return LineageTree("planted", np.asarray(parent), np.asarray(length, dtype=float), labels).to_newick()


def _plant_sequences(
    gen: np.random.Generator,
    *,
    rec_cell: np.ndarray,
    rec_clone: np.ndarray,
    rec_ccall: np.ndarray,
    clones: pd.DataFrame,
    lineages: pd.DataFrame,
    cell_lin: np.ndarray,
    cell_type: np.ndarray,
    shm_mutations: float,
    seq_error_rate: float,
    truncated_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sequences of every record of a synthetic chip (used by :func:`spaclone.sim.synthetic_chip`).

    ``rec_cell`` / ``rec_clone``: source cell and clone (row of ``clones``: heavy chains first, then light chains) of
    every record; ``rec_ccall``: its constant-region call; ``cell_lin``: lineage of every cell (-1 none). Returns the
    record columns (AIRR fields and truth) and the metadata (``genotypes``, ``lineage_trees``).
    """
    germ = _germline_set()
    n_lin = len(lineages)
    lin_ids = lineages["lineage"].astype(str).to_numpy()
    kinds = lineages["kind"].astype(str).to_numpy()
    comps = lineages["compartment"].astype(str).to_numpy()
    v_of = clones["v_call"].map(_gene).to_numpy(dtype=object)
    j_of = clones["j_call"].map(_gene).to_numpy(dtype=object)
    loc_of = clones["locus"].astype(str).to_numpy()
    # ---- planted genealogies: sequence of every (cell, chain) of the mutated B lineages
    cell_seq: dict[tuple[int, int], bytes] = {}  # (cell, 0 heavy / 1 light) -> V sequence
    trees: dict[str, str] = {}
    members = pd.Series(np.arange(len(cell_lin))).groupby(cell_lin).apply(list).to_dict()
    for lin in range(n_lin):
        cells = np.asarray(members.get(lin, []), dtype=np.int64)
        if kinds[lin] != "BCR" or not len(cells):
            continue
        mutated = comps[lin] in _MUTATED_COMPARTMENTS or (len(cells) == 1 and cell_type[cells[0]] == "plasma")
        if not mutated:
            continue
        n = len(cells)
        parent, tlen = _coalescent(n, gen)
        leaf_cell = cells[gen.permutation(n)]
        height = 2.0 * (1.0 - 1.0 / n) if n > 1 else 1.0
        for chain in (0, 1):
            k = lin if chain == 0 else n_lin + lin
            g0 = germ["v"][v_of[k]].encode()
            cys = len(g0) - 3
            mean = shm_mutations * (1.0 if chain == 0 else 0.5)
            root = bytearray(g0)
            trunk = len(_substitute(root, int(gen.poisson(0.25 * mean)), gen, _SHM_START, cys))
            rate = 0.75 * mean / height
            seqs: dict[int, bytearray] = {}
            n_mut = np.zeros(len(parent), dtype=np.int64)
            for v in range(len(parent) - 1, -1, -1):  # parents before children
                s = bytearray(root) if parent[v] < 0 else bytearray(seqs[int(parent[v])])
                n_mut[v] = len(_substitute(s, int(gen.poisson(rate * (tlen[v] if n > 1 else 1.0))), gen, _SHM_START,
                                           cys))  # fmt: skip
                seqs[v] = s
            for leaf in range(n):
                cell_seq[(int(leaf_cell[leaf]), chain)] = bytes(seqs[leaf])
            if chain == 0 and n > 1:  # the planted heavy-chain tree, tips named after their genotype below
                trees[lin_ids[lin]] = (parent, n_mut, trunk, leaf_cell)
    # ---- genotypes: distinct sequences per lineage and chain, in cell order
    geno_rows: list[dict[str, Any]] = []
    geno_of: dict[tuple[int, int], str] = {}
    seen: dict[tuple[int, int, bytes], str] = {}
    for (cell, chain), s in sorted(cell_seq.items()):
        lin = int(cell_lin[cell])
        key = (lin, chain, s)
        if key not in seen:
            k = lin if chain == 0 else n_lin + lin
            g0 = germ["v"][v_of[k]].encode()
            cys = len(g0) - 3
            sites = [(p, chr(s[p])) for p in range(cys) if s[p] != g0[p]]
            gid = f"{lin_ids[lin]}_{'H' if chain == 0 else 'L'}{sum(1 for x in seen if x[:2] == (lin, chain)) + 1}"
            seen[key] = gid
            geno_rows.append({"genotype": gid, "lineage": lin_ids[lin], "locus": loc_of[k],
                              "chain": "heavy" if chain == 0 else "light", "clone_id": clones["clone_id"].iloc[k],
                              "v_gene": v_of[k], "v_length": len(g0), "mutations": len(sites),
                              "sites": ";".join(f"{p}:{b}" for p, b in sites), "n_cells": 0})  # fmt: skip
        geno_of[(cell, chain)] = seen[key]
    genotypes = pd.DataFrame(geno_rows, columns=["genotype", "lineage", "locus", "chain", "clone_id", "v_gene",
                                                 "v_length", "mutations", "sites", "n_cells"])  # fmt: skip
    if len(genotypes):
        counts = pd.Series(list(geno_of.values())).value_counts()
        genotypes["n_cells"] = genotypes["genotype"].map(counts).fillna(0).astype(np.int64)
    newick: dict[str, str] = {}
    for lid, (parent, n_mut, trunk, leaf_cell) in trees.items():
        n_leaf = len(leaf_cell)
        m = len(parent)  # coalescent nodes; then the germline root and the "Germline" tip
        labels: list[str | None] = [geno_of[(int(leaf_cell[i]), 0)] for i in range(n_leaf)] + [None] * (m - n_leaf)
        par = [int(p) for p in parent]
        lens = [float(x) for x in n_mut]
        root = int(np.flatnonzero(parent < 0)[0])
        par[root] = m
        lens[root] = float(trunk + n_mut[root])
        labels += [None, "Germline"]
        par += [-1, m]
        lens += [0.0, 0.0]
        newick[lid] = _newick(labels, par, lens)
    # ---- records: contig, alignments, errors, truncation
    n_rec = len(rec_cell)
    out = {c: np.empty(n_rec, dtype=object) for c in ("sequence", "v_sequence_alignment", "v_germline_alignment",
                                                      "true_genotype", "seq_error_sites")}  # fmt: skip
    ints = {c: np.zeros(n_rec, dtype=np.int64) for c in ("v_sequence_start", "v_sequence_end", "v_germline_start",
                                                         "v_germline_end", "j_sequence_start", "j_sequence_end",
                                                         "true_mutations", "seq_errors")}  # fmt: skip
    truncated = gen.random(n_rec) < truncated_fraction
    lead_all = np.where(truncated, gen.integers(30, 150, n_rec), 0)
    err_rate = min(max(float(seq_error_rate), 0.0), 1.0)
    junctions = clones["junction"].astype(str).to_numpy()
    geno_mut = dict(zip(genotypes["genotype"], genotypes["mutations"], strict=True)) if len(genotypes) else {}
    for r in range(n_rec):
        k = int(rec_clone[r])
        cell = int(rec_cell[r])
        chain = 0 if k < n_lin else 1
        lin = k % n_lin
        vg = germ["v"][v_of[k]]
        g0 = vg.encode()
        cys = len(g0) - 3
        s = bytearray(cell_seq.get((cell, chain), g0))
        gid = geno_of.get((cell, chain))
        out["true_genotype"][r] = gid if gid is not None else f"{lin_ids[lin]}_{'H' if chain == 0 else 'L'}0"
        ints["true_mutations"][r] = geno_mut.get(gid, 0)
        lead = int(lead_all[r])
        errs = _substitute(s, int(gen.binomial(cys - lead, err_rate)), gen, lead, cys)  # in the sequenced V only
        ints["seq_errors"][r] = len(errs)
        out["seq_error_sites"][r] = ";".join(f"{p}:{chr(b)}" for p, b in errs)
        prefix = "" if lead else germ["leader"][v_of[k]]
        jx = junctions[k]
        v_part = s[lead:cys].decode()
        fr4 = germ["fr4"].get(j_of[k], "")
        out["sequence"][r] = prefix + v_part + jx + fr4 + _constant_start(str(rec_ccall[r]))
        out["v_sequence_alignment"][r] = s[lead:].decode()
        out["v_germline_alignment"][r] = vg[lead:]
        js = len(prefix) + len(v_part)  # 0-based start of the junction
        ints["v_sequence_start"][r] = len(prefix) + 1
        ints["v_sequence_end"][r] = len(prefix) + len(g0) - lead
        ints["v_germline_start"][r] = lead + 1
        ints["v_germline_end"][r] = len(g0)
        ints["j_sequence_start"][r] = js + len(jx) - 3 + 1  # the J alignment starts at the last junction codon
        ints["j_sequence_end"][r] = js + len(jx) + len(fr4)
    cols = pd.DataFrame(
        {
            "sequence_id": [f"rec{i:06d}" for i in range(n_rec)],
            "sequence": out["sequence"],
            "v_sequence_start": ints["v_sequence_start"],
            "v_sequence_end": ints["v_sequence_end"],
            "v_germline_start": ints["v_germline_start"],
            "v_germline_end": ints["v_germline_end"],
            "v_sequence_alignment": out["v_sequence_alignment"],
            "v_germline_alignment": out["v_germline_alignment"],
            "j_sequence_start": ints["j_sequence_start"],
            "j_sequence_end": ints["j_sequence_end"],
            "true_genotype": out["true_genotype"],
            "true_mutations": ints["true_mutations"],
            "seq_errors": ints["seq_errors"],
            "seq_error_sites": out["seq_error_sites"],
        }
    )
    return cols, {"genotypes": genotypes, "lineage_trees": newick}


# ------------------------------------------------------------------------------------------------- scoring


def _sites(text: Any) -> set[tuple[int, str]]:
    if not isinstance(text, str) or not text:
        return set()
    return {(int(p), b) for p, b in (x.split(":") for x in text.split(";"))}


def score_lineages(result: Any, truth: Any, *, grouping: str = "exact") -> pd.DataFrame:
    """Lineage trees of :func:`spaclone.bcr.lineage_workflow` against the lineages planted by the simulator.

    Parameters
    ----------
    result
        :class:`~spaclone.bcr.LineageWorkflowResult` of a section simulated with
        ``spaclone.sim.synthetic_chip(sequences=True)`` (or ``spaclone.datasets.synthetic_example(sequences=True)``).
    truth
        That section (the :class:`~spaclone.io.SpatialAIR`; its records carry ``sequence_id``, ``lineage``,
        ``true_genotype`` and ``truth``, its ``meta`` the planted genotypes and trees).
    grouping
        ``"exact"`` or ``"related"``.

    Returns
    -------
    pandas.DataFrame
        One row per tree: ``tree_id``, ``lineage`` (the planted lineage of most of its records), ``purity`` (share of
        its records from that lineage), ``drawn``, ``tips``; ``tips_exact`` (tips whose cleaned sequence carries
        exactly the planted mutations of their cells' genotype, no more, no fewer), ``tips_mixed`` (tips joining
        records of different genotypes, which the workflow could not tell apart), ``planted_mutations``,
        ``recovered_mutations`` and ``false_mutations`` (summed over tips with one genotype); ``tips_in_place``
        (tips whose records all sit on their own cell, ``truth == "genuine"``: what the position of a tip is worth in
        this section); ``rf``, ``rf_norm`` (Robinson-Foulds distance to the planted tree, on the genotypes the tree
        shows; 0 = same topology) and ``depth_cor`` (correlation of each tip's distance from the germline in the two
        trees). ``attrs["summary"]`` holds the totals and the record-level numbers: full-length records, the share
        whose cleaned mutation count equals the planted one, and the planted sequencing errors kept and reverted.

    Notes
    -----
    A planted mutation is recovered when the cleaned sequence of the tip has its base at its position; a false
    mutation is a cleaned difference from the germline that the tip's genotype does not carry. The workflow reverts
    a variant seen in fewer than ``noise_min_spots`` spots, so the private mutations of a cell read from a single
    molecule are lost by design: they are counted as not recovered.

    References
    ----------
    Robinson & Foulds 1981, doi:10.1016/0025-5564(81)90043-2.
    """
    from spaclone.bcr.lineage import LineageTree, compare_trees, read_trees
    from spaclone.bcr.lineage_workflow import _grouping

    g = _grouping(grouping)
    air = truth.air if hasattr(truth, "air") else truth
    meta = getattr(truth, "meta", None) or {}
    need = {"sequence_id", "lineage", "true_genotype", "truth"}
    if not need <= set(air.columns) or "genotypes" not in meta:
        raise ValueError(
            "score_lineages needs a section simulated with sequences (spaclone.sim.synthetic_chip(sequences=True) or "
            "spaclone.datasets.synthetic_example(sequences=True)): its records carry the planted truth"
        )
    rec = air.set_index(air["sequence_id"].astype(str))
    geno = meta["genotypes"].set_index("genotype")
    sites_of = {k: _sites(v) for k, v in geno["sites"].items()}
    planted = read_trees(dict(meta.get("lineage_trees", {})))
    tips = result.tips[g]
    sel = result.selection[result.selection.grouping == g].set_index("tree_id")
    rows = []
    for tid, t in tips[tips.tree_label.notna()].groupby("tree_id", sort=True):
        ids = [i for s in t["record_ids"].astype(str) for i in s.split(",")]
        lin = rec.loc[ids, "lineage"].astype(str)
        main = lin.value_counts().index[0]
        n_exact = n_mixed = planted_n = recovered = false = in_place = 0
        tip_geno: dict[str, str | None] = {}
        for tip in t.itertuples(index=False):
            r_ids = str(tip.record_ids).split(",")
            gs = set(rec.loc[r_ids, "true_genotype"].astype(str))
            in_place += bool((rec.loc[r_ids, "truth"].astype(str) == "genuine").all())
            if len(gs) != 1:
                n_mixed += 1
                tip_geno[str(tip.tree_label)] = None
                continue
            gid = next(iter(gs))
            want = sites_of.get(gid, set())
            cys = int(geno.loc[gid, "v_length"]) - 3 if gid in geno.index else 0
            seq, germ = str(tip.sequence), str(tip.germline)
            got = {(p, seq[p]) for p in range(min(cys, len(seq), len(germ)))
                   if seq[p] in "ACGT" and germ[p] in "ACGT" and seq[p] != germ[p]}  # fmt: skip
            planted_n += len(want)
            recovered += len(want & got)
            false += len(got - want)
            n_exact += got == want
            label = str(tip.tree_label)
            tip_geno[label] = gid if tip_geno.get(label, gid) == gid else None
        rf = rf_norm = cor = np.nan
        if (lin == main).all() and main in planted:
            tree = result.trees[g][tid]
            labels = [lab if lab in (None, "Germline") else tip_geno.get(lab) for lab in tree.label]
            ours = LineageTree(tid, tree.parent, tree.length, labels, method="relabelled")
            cmp = compare_trees(ours, planted[main])
            rf, rf_norm, cor = cmp["rf"], cmp["rf_norm"], cmp["div_cor"]
        rows.append({"tree_id": tid, "lineage": main, "purity": float((lin == main).mean()),
                     "drawn": bool(sel.loc[tid, "drawn"]) if tid in sel.index else False, "tips": len(t),
                     "tips_exact": int(n_exact), "tips_mixed": int(n_mixed), "planted_mutations": int(planted_n),
                     "recovered_mutations": int(recovered), "false_mutations": int(false),
                     "tips_in_place": int(in_place), "rf": rf, "rf_norm": rf_norm, "depth_cor": cor})  # fmt: skip
    cols = ["tree_id", "lineage", "purity", "drawn", "tips", "tips_exact", "tips_mixed", "planted_mutations",
            "recovered_mutations", "false_mutations", "tips_in_place", "rf", "rf_norm", "depth_cor"]  # fmt: skip
    out = pd.DataFrame(rows, columns=cols)
    # ---- record level: cleaned mutation counts and the planted errors
    r = result.records
    full = r[r["full_length"].astype(bool) & r[f"mutations_{g}"].notna()]
    truth_r = rec.reindex(full["sequence_id"].astype(str))
    exact = full[f"mutations_{g}"].to_numpy(dtype=float) == truth_r["true_mutations"].to_numpy(dtype=float)
    kept = reverted = 0
    if "seq_error_sites" in truth_r.columns:
        for sid, s_clean in zip(full["sequence_id"].astype(str), full[f"sequence_{g}"], strict=True):
            for p, b in _sites(rec.at[sid, "seq_error_sites"]):
                if isinstance(s_clean, str) and p < len(s_clean):
                    kept += s_clean[p] == b
                    reverted += s_clean[p] != b
    out.attrs["summary"] = {
        "trees": len(out),
        "tips": int(out["tips"].sum()),
        "tips_exact": int(out["tips_exact"].sum()),
        "planted_mutations": int(out["planted_mutations"].sum()),
        "recovered_mutations": int(out["recovered_mutations"].sum()),
        "false_mutations": int(out["false_mutations"].sum()),
        "median_rf_norm": float(out["rf_norm"].median()) if out["rf_norm"].notna().any() else float("nan"),
        "records_full_length": len(full),
        "records_exact_mutations": float(exact.mean()) if len(full) else float("nan"),
        "errors_in_full_length_records": int(kept + reverted),
        "errors_reverted": int(reverted),
    }
    return out
