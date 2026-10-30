"""Germline-rooted B-cell lineage trees for small clones, Newick input / output, and tip tables with tissue positions.

:func:`build_trees` infers one tree per clone in pure Python / numpy:

* ``method="parsimony"``: maximum parsimony (Fitch) with the clonal germline as a fixed taxon. Random-order stepwise
  addition, then subtree pruning and regrafting (SPR) until no move shortens the tree, then a parsimony ratchet
  (SPR on bootstrap-reweighted sites, then on the original sites); the shortest tree over ``n_starts`` starts is kept.
  Insertion costs are exact for Fitch parsimony (a subtree with state set X joined to an edge with set F adds one
  step per site where X and F are disjoint), so every SPR neighbour is scored without rebuilding the tree.
* ``method="nj"``: neighbour joining on pairwise differences at sites both sequences cover.

Both trees are hung from the germline (the root's children are the ``Germline`` tip, branch length 0, and the most
recent common ancestor's lineage, as dowser exports them). Branch lengths are mutations under one most-parsimonious
reconstruction (ACCTRAN: changes as close to the root as allowed). Identical sequences of a clone are collapsed into
one tip (only when their ``traits``, e.g. isotype and location, are also identical, as dowser's ``formatClones``).

:func:`read_trees` imports Newick strings or files, IgPhyML / Change-O tables with a Newick column, and node tables
exported from dowser (``clone_id``, ``node``, ``parent``, branch length, ``label``). :func:`tip_table` returns one row
per tip with the tree, the tip label, its tissue position, region and isotype and its mutations from the germline:
the input of :func:`spaclone.tl.tip_region_test` / :func:`spaclone.tl.tree_tissue_distance`. :func:`compare_trees`
gives the Robinson-Foulds distance and the correlation of germline-to-tip divergence between two trees.

These trees are meant for small clones (up to a few hundred distinct sequences) and for spatial statistics on tips.
For likelihood trees with SHM-aware models use IgPhyML (Hoehn et al. 2019) through dowser (Hoehn et al. 2022).

References
----------
Fitch 1971, doi:10.2307/2412116 (small parsimony); Swofford & Maddison 1987, doi:10.1016/0025-5564(87)90074-5
(ACCTRAN); Nixon 1999, doi:10.1111/j.1096-0031.1999.tb00277.x (parsimony ratchet); Saitou & Nei 1987,
doi:10.1093/oxfordjournals.molbev.a040454 (neighbour joining); Robinson & Foulds 1981,
doi:10.1016/0025-5564(81)90043-2; Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885 (dowser); Hoehn et al. 2019,
doi:10.1073/pnas.1906020116 (IgPhyML).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import RngLike, as_rng

__all__ = [
    "GERMLINE",
    "LineageTree",
    "LineageTrees",
    "build_trees",
    "compare_trees",
    "parsimony_score",
    "read_trees",
    "tip_table",
]

#: Label of the germline tip (dowser convention).
GERMLINE = "Germline"

_BIT = np.full(256, 15, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _BIT[ord(_c)] = 1 << _i
    _BIT[ord(_c.lower())] = 1 << _i
_LETTER = np.array(list("NACNGNNNTNNNNNNN"))  # bit -> letter (single bits only; anything else N)


# ------------------------------------------------------------------------------------------------- tree objects


@dataclass(eq=False)
class LineageTree:
    """One rooted tree: parent pointers, branch lengths (mutations) and tip labels.

    Attributes
    ----------
    clone_id
        Tree (clone) id.
    parent
        Parent index of every node (-1 for the root). Node 0 is the root for trees built here.
    length
        Branch length to the parent (mutations); 0 for the root.
    label
        Tip label of every node (``None`` for internal nodes); the germline tip is ``"Germline"``.
    sequences
        Node sequences (tips: as given; internal nodes: one most-parsimonious reconstruction), or ``None``.
    members
        Tip label -> ids of the input tips collapsed into it.
    method
        How the tree was made (``"parsimony"``, ``"nj"``, ``"newick"``, ``"node table"``).
    score
        Parsimony length (sum of branch lengths for trees built here).
    """

    clone_id: Any
    parent: np.ndarray
    length: np.ndarray
    label: list[str | None]
    sequences: list[str] | None = None
    members: dict[str, list[str]] = field(default_factory=dict)
    method: str = ""
    score: float = np.nan

    def __post_init__(self) -> None:
        self.parent = np.asarray(self.parent, dtype=np.int64)
        self.length = np.asarray(self.length, dtype=float)
        if len(self.parent) != len(self.length) or len(self.label) != len(self.parent):
            raise ValueError("parent, length and label must have one entry per node")
        if (self.parent < 0).sum() != 1:
            raise ValueError("a tree has exactly one root (parent -1)")

    def __repr__(self) -> str:
        return (
            f"LineageTree({self.clone_id!r}: {self.n_tips} tips, {len(self.parent)} nodes, method {self.method}, "
            f"length {self.score:g})"
        )

    @property
    def n_tips(self) -> int:
        """Number of labelled tips, the germline included."""
        return sum(lab is not None for lab in self.label)

    @property
    def root(self) -> int:
        """Index of the root."""
        return int(np.flatnonzero(self.parent < 0)[0])

    def children(self) -> list[list[int]]:
        """Children of every node."""
        out: list[list[int]] = [[] for _ in range(len(self.parent))]
        for v, p in enumerate(self.parent):
            if p >= 0:
                out[p].append(v)
        return out

    def tips(self) -> list[str]:
        """Tip labels (germline included), in node order."""
        return [lab for lab in self.label if lab is not None]

    def depths(self, *, from_label: str | None = GERMLINE) -> np.ndarray:
        """Path length (mutations) from the tip ``from_label`` to every node, along the unrooted tree.

        ``from_label`` defaults to the germline; without such a tip, depths are measured from the root.
        """
        n = len(self.parent)
        nb: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for v, p in enumerate(self.parent):
            if p >= 0:
                nb[v].append((int(p), float(self.length[v])))
                nb[p].append((v, float(self.length[v])))
        start = self.root
        if from_label is not None and from_label in self.label:
            start = self.label.index(from_label)
        d = np.full(n, np.nan)
        d[start] = 0.0
        stack = [start]
        while stack:
            u = stack.pop()
            for v, w in nb[u]:
                if np.isnan(d[v]):
                    d[v] = d[u] + w
                    stack.append(v)
        return d

    def edges(self) -> pd.DataFrame:
        """Node table: ``node``, ``parent`` (missing for the root), ``length``, ``label``, ``is_tip``."""
        ch = self.children()
        return pd.DataFrame(
            {
                "node": np.arange(len(self.parent)),
                "parent": pd.array([int(p) if p >= 0 else None for p in self.parent], dtype="Int64"),
                "length": self.length,
                "label": self.label,
                "is_tip": [len(c) == 0 for c in ch],
            }
        )

    def to_newick(self, *, digits: int = 6) -> str:
        """Newick string (rooted; branch lengths in mutations)."""
        ch = self.children()

        def name(v: int) -> str:
            lab = self.label[v]
            if lab is None:
                return ""
            return lab if re.fullmatch(r"[^\s(),:;\[\]']+", lab) else "'" + lab.replace("'", "''") + "'"

        def bl(v: int) -> str:
            x = round(float(self.length[v]), digits)
            return f":{x:g}"

        out: dict[int, str] = {}
        order = self._postorder(ch)
        for v in order:
            if ch[v]:
                out[v] = "(" + ",".join(out[c] + bl(c) for c in ch[v]) + ")" + name(v)
            else:
                out[v] = name(v)
        return out[self.root] + ";"

    def _postorder(self, ch: list[list[int]]) -> list[int]:
        pre = [self.root]
        i = 0
        while i < len(pre):
            pre.extend(ch[pre[i]])
            i += 1
        return pre[::-1]

    def splits(self, *, drop: Sequence[str] = (GERMLINE,), min_length: float = 0.5) -> tuple[set[frozenset], set]:
        """Non-trivial bipartitions of the unrooted tree on the labelled tips minus ``drop``.

        Edges shorter than ``min_length`` are collapsed first (their split is not counted). Returns the splits (each
        as the side without the smallest label) and the label set.
        """
        ch = self.children()
        keep = {lab for lab in self.label if lab is not None and lab not in drop}
        below: list[frozenset] = [frozenset()] * len(self.parent)
        for v in self._postorder(ch):
            s = {self.label[v]} if self.label[v] in keep else set()
            for c in ch[v]:
                s |= below[c]
            below[v] = frozenset(s)
        if not keep:
            return set(), keep
        ref = min(keep)
        out = set()
        n = len(keep)
        for v in range(len(self.parent)):
            if self.parent[v] < 0 or not ch[v] or self.length[v] < min_length:
                continue
            side = below[v]
            if ref in side:
                side = frozenset(keep - side)
            if 2 <= len(side) <= n - 2:
                out.add(side)
        return out, keep


class LineageTrees(Mapping):
    """Trees by clone id (a read-only mapping), with the tips they were built from.

    Attributes
    ----------
    tips
        For trees made by :func:`build_trees`: the input rows of the clones with a tree, with the tree's tip label in
        ``tree_label`` and ``n_collapsed``. ``None`` for trees read from files.
    """

    def __init__(self, trees: Mapping[Any, LineageTree], tips: pd.DataFrame | None = None) -> None:
        self._trees = dict(trees)
        self.tips = tips

    def __getitem__(self, key: Any) -> LineageTree:
        return self._trees[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __repr__(self) -> str:
        return f"LineageTrees({len(self)} trees)"

    @property
    def newick(self) -> dict[Any, str]:
        """Newick string per clone (for :func:`spaclone.tl.tree_tissue_distance`)."""
        return {k: t.to_newick() for k, t in self._trees.items()}

    def edges(self, *, clone_key: str = S.CLONE) -> pd.DataFrame:
        """All node tables stacked, with the clone id."""
        parts = [t.edges().assign(**{clone_key: k}) for k, t in self._trees.items()]
        if not parts:
            return pd.DataFrame(columns=[clone_key, "node", "parent", "length", "label", "is_tip"])
        d = pd.concat(parts, ignore_index=True)
        return d[[clone_key] + [c for c in d.columns if c != clone_key]]

    def summary(self) -> pd.DataFrame:
        """One row per tree: tips (germline excluded), nodes, method, parsimony length."""
        return pd.DataFrame(
            [
                {
                    "clone_id": k,
                    "tips": t.n_tips - (GERMLINE in t.label),
                    "nodes": len(t.parent),
                    "method": t.method,
                    "length": t.score,
                }
                for k, t in self._trees.items()
            ]
        )

    def write_newick(self, path: str | Path) -> None:
        """Write ``clone_id <TAB> newick`` lines."""
        with open(path, "w") as fh:
            for k, t in self._trees.items():
                fh.write(f"{k}\t{t.to_newick()}\n")


# ------------------------------------------------------------------------------------------------ Fitch search


def _pack(codes: np.ndarray) -> list[int]:
    """Rows of 4-bit state sets (uint8 1..15) as Python integers, 4 bits per site (site j at bits 4j..4j+3)."""
    n, p = codes.shape
    if p % 2:
        codes = np.column_stack([codes, np.full(n, 0, dtype=np.uint8)])
    by = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return [int.from_bytes(row.tobytes(), "little") for row in by]


class _Fitch:
    """Rooted binary tree below the germline; tips 0..n-1, internal nodes n..2n-2; sets as packed integers."""

    def __init__(self, codes: np.ndarray, germ: np.ndarray) -> None:
        self.n = codes.shape[0]
        self.set_data(codes, germ)
        size = 2 * self.n - 1
        self.left = [-1] * size
        self.right = [-1] * size
        self.par = [-1] * size
        self.top = -1
        self.cost = [0] * size
        self.U = [0] * size
        self.F = [0] * size

    def set_data(self, codes: np.ndarray, germ: np.ndarray) -> None:
        p = codes.shape[1]
        self.M = sum(1 << (4 * j) for j in range(p))
        tips = _pack(codes)
        self.G = _pack(germ[None, :])[0]
        size = 2 * self.n - 1
        if not hasattr(self, "D") or len(self.D) != size:
            self.D = [0] * size
        self.D[: self.n] = tips

    def _nonempty(self, x: int) -> int:
        y = x | (x >> 1)
        y |= y >> 2
        return y & self.M

    def preorder(self) -> list[int]:
        out = [self.top]
        i = 0
        left, right = self.left, self.right
        while i < len(out):
            v = out[i]
            i += 1
            if left[v] >= 0:
                out.append(left[v])
                out.append(right[v])
        return out

    def downpass(self, pre: list[int]) -> int:
        D, M, cost, left, right = self.D, self.M, self.cost, self.left, self.right
        total = 0
        for v in reversed(pre):
            lv = left[v]
            if lv < 0:
                continue
            a, b = D[lv], D[right[v]]
            i = a & b
            y = i | (i >> 1)
            y |= y >> 2
            e = M ^ (y & M)
            D[v] = i | ((a | b) & (e * 15))
            c = e.bit_count()
            cost[v] = c
            total += c
        i = D[self.top] & self.G
        y = i | (i >> 1)
        y |= y >> 2
        return total + (M ^ (y & M)).bit_count()

    def uppass(self, pre: list[int]) -> None:
        D, U, F, M, left, right = self.D, self.U, self.F, self.M, self.left, self.right
        U[self.top] = self.G
        for v in pre:
            u = U[v]
            d = D[v]
            i = u & d
            y = i | (i >> 1)
            y |= y >> 2
            e = M ^ (y & M)
            F[v] = i | ((u | d) & (e * 15))
            lv = left[v]
            if lv < 0:
                continue
            rv = right[v]
            for c, s in ((lv, rv), (rv, lv)):
                ds = D[s]
                i = u & ds
                y = i | (i >> 1)
                y |= y >> 2
                e = M ^ (y & M)
                U[c] = i | ((u | ds) & (e * 15))

    def insertion_costs(self, pre: list[int], x: int) -> list[int]:
        F, M = self.F, self.M
        out = []
        for v in pre:
            i = F[v] & x
            y = i | (i >> 1)
            y |= y >> 2
            out.append((M ^ (y & M)).bit_count())
        return out

    def attach(self, x: int, e: int, t: int) -> None:
        """New internal node x on the edge above e, with children e and t."""
        p = self.par[e]
        self.par[x] = p
        self.left[x], self.right[x] = e, t
        self.par[e] = x
        self.par[t] = x
        if p < 0:
            self.top = x
        elif self.left[p] == e:
            self.left[p] = x
        else:
            self.right[p] = x

    def detach(self, v: int) -> tuple[int, int]:
        """Remove v's subtree and v's parent; v's sibling takes the parent's place. Returns (parent, sibling)."""
        p = self.par[v]
        s = self.right[p] if self.left[p] == v else self.left[p]
        gp = self.par[p]
        self.par[s] = gp
        if gp < 0:
            self.top = s
        elif self.left[gp] == p:
            self.left[gp] = s
        else:
            self.right[gp] = s
        self.par[v] = -1
        self.par[p] = -1
        self.left[p] = self.right[p] = -1
        return p, s

    def subtree(self, v: int) -> list[int]:
        out = [v]
        i = 0
        while i < len(out):
            u = out[i]
            i += 1
            if self.left[u] >= 0:
                out.append(self.left[u])
                out.append(self.right[u])
        return out

    def stepwise(self, order: np.ndarray, rng: np.random.Generator) -> int:
        n = self.n
        free = list(range(2 * n - 2, n - 1, -1))
        x = free.pop()
        self.top = x
        self.left[x], self.right[x] = int(order[0]), int(order[1])
        self.par[order[0]] = self.par[order[1]] = x
        for t in order[2:]:
            pre = self.preorder()
            self.downpass(pre)
            self.uppass(pre)
            costs = self.insertion_costs(pre, self.D[t])
            m = min(costs)
            cand = [pre[k] for k, c in enumerate(costs) if c == m]
            e = cand[int(rng.integers(len(cand)))] if len(cand) > 1 else cand[0]
            self.attach(free.pop(), e, int(t))
        return self.downpass(self.preorder())

    def repair(self, v: int) -> None:
        """Recompute the Fitch sets and costs of v and its ancestors (after a subtree was put back)."""
        D, M, cost, left, right, par = self.D, self.M, self.cost, self.left, self.right, self.par
        while v >= 0:
            a, b = D[left[v]], D[right[v]]
            i = a & b
            y = i | (i >> 1)
            y |= y >> 2
            e = M ^ (y & M)
            D[v] = i | ((a | b) & (e * 15))
            cost[v] = e.bit_count()
            v = par[v]

    def spr(self, rng: np.random.Generator, max_rounds: int = 100) -> int:
        """Hill-climb by subtree pruning and regrafting until no move shortens the tree; returns the length."""
        pre = self.preorder()
        length = self.downpass(pre)
        for _ in range(max_rounds):
            improved = False
            cand = [v for v in pre if v != self.top]
            rng.shuffle(cand)
            for v in cand:
                if v == self.top:
                    continue
                l_sub = sum(self.cost[u] for u in self.subtree(v) if self.left[u] >= 0)
                x = self.D[v]
                p, s = self.detach(v)
                pre_r = self.preorder()
                l_rem = self.downpass(pre_r)
                self.uppass(pre_r)
                costs = self.insertion_costs(pre_r, x)
                m = min(costs)
                if l_rem + l_sub + m < length:
                    cand_e = [pre_r[k] for k, c in enumerate(costs) if c == m]
                    e = cand_e[int(rng.integers(len(cand_e)))] if len(cand_e) > 1 else cand_e[0]
                    self.attach(p, e, v)
                    length = self.downpass(self.preorder())
                    improved = True
                else:
                    self.attach(p, s, v)  # back in place: only p and its ancestors have stale sets
                    self.repair(p)
            pre = self.preorder()
            if not improved:
                break
        return length

    def state(self) -> tuple[list[int], list[int], list[int], int]:
        return list(self.left), list(self.right), list(self.par), self.top

    def restore(self, st: tuple[list[int], list[int], list[int], int]) -> None:
        self.left, self.right, self.par, self.top = list(st[0]), list(st[1]), list(st[2]), st[3]


def _search(codes: np.ndarray, germ: np.ndarray, n_starts: int, n_ratchet: int, rng: np.random.Generator) -> _Fitch:
    """Best Fitch tree over several starts (stepwise addition + SPR + ratchet)."""
    best, best_len = None, np.inf
    p = codes.shape[1]
    for _ in range(max(1, n_starts)):
        f = _Fitch(codes, germ)
        f.stepwise(rng.permutation(f.n), rng)
        length = f.spr(rng)
        for _ in range(n_ratchet):
            keep = f.state()
            w = rng.multinomial(p, np.full(p, 1.0 / p))
            cols = np.repeat(np.arange(p), w)
            f.set_data(codes[:, cols], germ[cols])
            f.spr(rng)
            f.set_data(codes, germ)
            new = f.spr(rng)
            if new > length:
                f.restore(keep)
                f.downpass(f.preorder())
            else:
                length = new
        if length < best_len:
            best, best_len = f, length
    assert best is not None
    best.score_ = best_len
    return best


def _nj_tree(codes: np.ndarray, germ: np.ndarray) -> _Fitch:
    """Neighbour-joining topology (germline as a taxon), hung from the germline as a binary tree."""
    allc = np.vstack([codes, germ[None, :]])
    m = len(allc)
    single = np.isin(allc, (1, 2, 4, 8))
    dist = np.zeros((m, m))
    for i in range(m):
        both = single[i][None, :] & single
        dist[i] = (both & (allc[i][None, :] != allc)).sum(1)
    size = 2 * m
    dm = np.zeros((size, size))
    dm[:m, :m] = dist
    adj: dict[int, dict[int, float]] = {i: {} for i in range(size)}
    active = list(range(m))
    nxt = m

    def link(u: int, x: int, w: float) -> None:
        adj[u][x] = adj[x][u] = max(float(w), 0.0)

    while len(active) > 3:
        act = np.array(active)
        k = len(act)
        sub = dm[np.ix_(act, act)]
        r = sub.sum(1)
        q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        ia, ib = divmod(int(np.argmin(q)), k)
        i, j = int(act[ia]), int(act[ib])
        li = 0.5 * sub[ia, ib] + (r[ia] - r[ib]) / (2 * (k - 2))
        u = nxt
        nxt += 1
        link(u, i, li)
        link(u, j, sub[ia, ib] - li)
        new = 0.5 * (dm[i, act] + dm[j, act] - sub[ia, ib])
        dm[u, act] = new
        dm[act, u] = new
        active = [c for c in active if c not in (i, j)] + [u]
    if len(active) == 3:
        x, y, z = active
        u = nxt
        nxt += 1
        lx = 0.5 * (dm[x, y] + dm[x, z] - dm[y, z])
        link(u, x, lx)
        link(u, y, dm[x, y] - lx)
        link(u, z, dm[x, z] - lx)
    else:  # pragma: no cover - build_trees needs >= 2 tips, so >= 3 taxa
        link(active[0], active[1], dm[active[0], active[1]])
    g = m - 1
    n = codes.shape[0]
    f = _Fitch(codes, germ)
    nid: dict[int, int] = {}
    free = list(range(2 * n - 2, n - 1, -1))

    def node_id(x: int) -> int:
        if x < n:
            return x
        if x not in nid:
            nid[x] = free.pop()
        return nid[x]

    (start,) = adj[g].keys()
    f.top = node_id(start)
    stack = [(start, g)]
    while stack:
        x, frm = stack.pop()
        kids = [y for y in adj[x] if y != frm]
        if not kids:
            continue
        if len(kids) != 2:  # pragma: no cover - NJ trees are binary
            raise RuntimeError("unexpected node degree in the NJ tree")
        xi, ka, kb = node_id(x), node_id(kids[0]), node_id(kids[1])
        f.left[xi], f.right[xi] = ka, kb
        f.par[ka] = f.par[kb] = xi
        stack.extend([(kids[0], x), (kids[1], x)])
    f.score_ = f.downpass(f.preorder())
    return f


def _reconstruct(f: _Fitch, codes: np.ndarray, germ: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """ACCTRAN-style most-parsimonious states (one bit per site) of all nodes and the germline (numpy)."""
    pre = f.preorder()
    size = 2 * f.n - 1
    D = [None] * size
    for t in range(f.n):
        D[t] = codes[t]
    for v in reversed(pre):
        lv = f.left[v]
        if lv < 0:
            continue
        a, b = D[lv], D[f.right[v]]
        i = a & b
        D[v] = np.where(i == 0, a | b, i)

    def lowest(x: np.ndarray) -> np.ndarray:
        return (x & (-x.astype(np.int16)).astype(np.uint8)).astype(np.uint8)

    g_state = np.where(np.isin(germ, (1, 2, 4, 8)), germ, 0).astype(np.uint8)
    top_d = D[f.top]
    g_state = np.where(g_state == 0, np.where(top_d & germ, lowest(top_d & germ), lowest(top_d)), g_state)
    state: list[np.ndarray | None] = [None] * size
    for v in pre:
        ps = g_state if f.par[v] < 0 else state[f.par[v]]
        dv = D[v]
        state[v] = np.where(dv & ps, ps, lowest(dv)).astype(np.uint8)
    return state, g_state


def _to_lineage_tree(
    f: _Fitch,
    codes_full: np.ndarray,
    germ_full: np.ndarray,
    labels: list[str],
    sequences: list[str],
    germline_seq: str,
    clone_id: Any,
    method: str,
    members: dict[str, list[str]],
) -> LineageTree:
    state, g_state = _reconstruct(f, codes_full, germ_full)
    pre = f.preorder()
    # export order: 0 = root (germline state), 1 = Germline tip, then the search tree in preorder
    index = {v: k + 2 for k, v in enumerate(pre)}
    n_nodes = len(pre) + 2
    parent = np.full(n_nodes, -1, dtype=np.int64)
    length = np.zeros(n_nodes)
    lab: list[str | None] = [None] * n_nodes
    seqs: list[str] = [""] * n_nodes
    parent[1] = 0
    lab[1] = GERMLINE
    seqs[1] = germline_seq
    seqs[0] = "".join(_LETTER[g_state])
    for v in pre:
        k = index[v]
        pv = f.par[v]
        parent[k] = 0 if pv < 0 else index[pv]
        ps = g_state if pv < 0 else state[pv]
        length[k] = float((state[v] != ps).sum())
        if v < f.n:
            lab[k] = labels[v]
            seqs[k] = sequences[v]
        else:
            seqs[k] = "".join(_LETTER[state[v]])
    return LineageTree(
        clone_id=clone_id,
        parent=parent,
        length=length,
        label=lab,
        sequences=seqs,
        members=members,
        method=method,
        score=float(length.sum()),
    )


def _one_clone(args: tuple) -> LineageTree | None:
    (clone_id, labels, seqs, germ, members, method, n_starts, n_ratchet, seed) = args
    rng = np.random.default_rng(seed)
    width = max(max(len(s) for s in seqs), len(germ))
    codes_full = _BIT[np.frombuffer("".join(s.ljust(width, "N") for s in seqs).encode("ascii", "replace"), np.uint8)]
    codes_full = codes_full.reshape(len(seqs), width)
    germ_full = _BIT[np.frombuffer(germ.ljust(width, "N").encode("ascii", "replace"), np.uint8)]
    # parsimony-informative pattern of sites: at least two different known states among sequences + germline
    allc = np.vstack([codes_full, germ_full[None, :]])
    single = np.isin(allc, (1, 2, 4, 8))
    known = np.where(single, allc, 0)
    nstates = sum(((known & b) > 0).any(0) for b in (1, 2, 4, 8))
    var = nstates >= 2
    if len(seqs) < 2:
        return None
    codes = codes_full[:, var] if var.any() else codes_full[:, :1]
    gv = germ_full[var] if var.any() else germ_full[:1]
    f = _search(codes, gv, n_starts, n_ratchet, rng) if method == "parsimony" else _nj_tree(codes, gv)
    return _to_lineage_tree(f, codes_full, germ_full, labels, seqs, germ, clone_id, method, members)


def _in_frame_stop(seqs: Sequence[str]) -> np.ndarray:
    """True for sequences with TAA / TAG / TGA in frame from the first character."""
    if not len(seqs):
        return np.zeros(0, dtype=bool)
    width = max(len(x) for x in seqs)
    width += (-width) % 3
    a = np.frombuffer("".join(x.upper().ljust(width, "N") for x in seqs).encode("ascii", "replace"), np.uint8)
    cod = a.reshape(len(seqs), width // 3, 3)
    t, aa, g = ord("T"), ord("A"), ord("G")
    stop = (cod[..., 0] == t) & (
        ((cod[..., 1] == aa) & ((cod[..., 2] == aa) | (cod[..., 2] == g))) | ((cod[..., 1] == g) & (cod[..., 2] == aa))
    )
    return stop.any(1)


_IGNORE = np.frombuffer(b"N-.?", np.uint8)


def _collapse_group(ids: list[str], seqs: list[str], mode: str) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    """Collapse identical sequences of one trait group.

    Returns ``[(label, sequence, member ids), ...]`` in order of first appearance and the discarded ids.
    ``mode="exact"``: identical strings. ``mode="ambiguous"``: the rule of alakazam ``collapseDuplicates`` (used by
    dowser ``formatClones``): ``N``, ``-``, ``.`` and ``?`` match any character; a sequence compatible with two
    incompatible sequences is ambiguous and discarded; the tip keeps the member with most informative characters
    (first in input order among ties).
    """
    if mode == "exact" or len(ids) == 1:
        first: dict[str, int] = {}
        out: list[tuple[str, str, list[str]]] = []
        for tid, sq in zip(ids, seqs, strict=True):
            if sq not in first:
                first[sq] = len(out)
                out.append((tid, sq, [tid]))
            else:
                out[first[sq]][2].append(tid)
        return out, []
    uniq = list(dict.fromkeys(seqs))
    width = max(len(x) for x in uniq)
    a = np.frombuffer("".join(x.ljust(width, "N") for x in uniq).encode("ascii", "replace"), np.uint8)
    a = a.reshape(len(uniq), width)
    ign = np.isin(a, _IGNORE)
    eq = ((a[:, None, :] == a[None, :, :]) | ign[:, None, :] | ign[None, :, :]).all(-1)
    ambiguous = np.array([not eq[np.ix_(np.flatnonzero(r), np.flatnonzero(r))].all() for r in eq])
    info_len = (~ign).sum(1)
    row_u = [uniq.index(x) for x in seqs]
    discarded = [tid for tid, u in zip(ids, row_u, strict=True) if ambiguous[u]]
    keep_rows = [k for k, u in enumerate(row_u) if not ambiguous[u]]
    if len(keep_rows) <= 1:  # alakazam: one representative per connected group of compatible sequences
        from scipy.sparse.csgraph import connected_components

        _, comp = connected_components(eq.astype(np.int8), directed=False)
        groups: dict[int, list[int]] = {}
        for k, u in enumerate(row_u):
            groups.setdefault(int(comp[u]), []).append(k)
        out = []
        for rows in groups.values():
            best = max(rows, key=lambda k: (info_len[row_u[k]], -k))
            out.append((ids[best], seqs[best], [ids[k] for k in rows]))
        return out, []
    done: set[int] = set()
    out = []
    for u in dict.fromkeys(row_u[k] for k in keep_rows):
        if u in done:
            continue
        idx = {int(v) for v in np.flatnonzero(eq[u]) if not ambiguous[v]}
        done |= idx
        rows = [k for k in keep_rows if row_u[k] in idx]
        best = max(rows, key=lambda k: (info_len[row_u[k]], -k))
        out.append((ids[best], seqs[best], [ids[k] for k in rows]))
    return out, discarded


def build_trees(
    tips: pd.DataFrame,
    *,
    clone_key: str = S.CLONE,
    method: Literal["parsimony", "nj"] = "parsimony",
    germline_key: str = "germline_alignment_d_mask",
    sequence_key: str = "sequence_alignment",
    id_key: str = "sequence_id",
    traits: Sequence[str] = (),
    collapse: Literal["ambiguous", "exact", "none"] = "ambiguous",
    filter_stop: bool = True,
    min_seqs: int = 2,
    n_starts: int = 2,
    n_ratchet: int = 4,
    rng: RngLike = None,
    n_jobs: int = 1,
) -> LineageTrees:
    """Germline-rooted lineage trees, one per clone, in pure Python.

    Parameters
    ----------
    tips
        One row per tip (e.g. one locus x one sequence) with the clone id, a tip id, the aligned sequence and the
        clone's germline (aligned in the same register, e.g. IMGT-gapped ``sequence_alignment`` and Change-O's
        ``germline_alignment_d_mask``). Characters other than A / C / G / T (gaps, ``N``) are missing data.
    clone_key
        Tree (clone) column.
    method
        ``"parsimony"`` (maximum parsimony search) or ``"nj"`` (neighbour joining).
    germline_key
        Germline column (constant within a clone); falls back to ``"germline_alignment"`` when absent.
    sequence_key, id_key
        Sequence and tip id columns.
    traits
        Columns that must also be identical for identical sequences to collapse into one tip (e.g.
        ``("isotype", "location")``); with ``()`` identical sequences always collapse.
    collapse
        How identical sequences (with identical traits) become one tip: ``"ambiguous"`` (default, as dowser's
        ``formatClones`` / alakazam ``collapseDuplicates``): ``N``, ``-``, ``.`` and ``?`` match any character, so a
        5'-padded copy collapses with the full sequence; a sequence compatible with two incompatible ones is ambiguous
        and left out (``tree_label`` missing in ``tips``); the tip keeps the member with most informative characters.
        ``"exact"``: identical strings only. ``"none"``: every row is a tip. All member ids are in ``tree.members``.
    filter_stop
        Leave out tips whose aligned sequence has an in-frame stop codon (codons read from the first alignment
        column, as IMGT-gapped alignments are in frame), as dowser's ``formatClones(filterstop=TRUE)``; they get a
        missing ``tree_label`` in ``tips``.
    min_seqs
        Clones with fewer distinct tips get no tree.
    n_starts, n_ratchet
        Parsimony search: random addition orders and ratchet iterations per start.
    rng
        Seed or generator; each clone's search uses a seed derived from it and from the clone id, so results do not
        depend on ``n_jobs`` or on which other clones are in the table.
    n_jobs
        Worker processes.

    Returns
    -------
    LineageTrees
        Mapping clone id -> :class:`LineageTree` (germline-rooted, branch lengths in mutations), with ``tips``: the
        input rows of the clones with a tree plus ``tree_label`` (the tree tip that holds the row) and
        ``n_collapsed``.

    Notes
    -----
    Parsimony length is the number of changes on the tree with the germline as an observed taxon; ties between
    equally short trees are broken at random (seeded), so other most-parsimonious trees may exist. Clone order
    (sorted ids) and per-clone seeds make the result reproducible.
    """
    if method not in ("parsimony", "nj"):
        raise ValueError(f"method must be 'parsimony' or 'nj', got {method!r}")
    if germline_key not in tips.columns and "germline_alignment" in tips.columns:
        germline_key = "germline_alignment"
    need = [clone_key, sequence_key, germline_key, id_key, *traits]
    missing = [c for c in need if c not in tips.columns]
    if missing:
        raise KeyError(f"build_trees: column(s) {missing} not in the tip table")
    if tips[id_key].duplicated().any():
        raise ValueError(f"build_trees: tip ids ({id_key!r}) must be unique")
    gen = as_rng(rng)
    d = tips.copy()
    d["_seq"] = d[sequence_key].astype("string").str.upper().fillna("")
    clones = sorted(d[clone_key].dropna().unique(), key=str)
    root = getattr(gen.bit_generator, "seed_seq", None)
    if root is None:  # pragma: no cover - generators built from a SeedSequence always carry one
        root = np.random.SeedSequence(int(gen.integers(0, 2**63 - 1)))
    seeds = [
        np.random.SeedSequence(
            entropy=root.entropy,
            spawn_key=(*root.spawn_key, int.from_bytes(hashlib.md5(str(c).encode()).digest()[:8], "little")),
        )
        for c in clones
    ]
    if collapse not in ("ambiguous", "exact", "none"):
        raise ValueError(f"collapse must be 'ambiguous', 'exact' or 'none', got {collapse!r}")
    jobs, label_of, n_coll, discarded = [], {}, {}, []
    if filter_stop:
        has_stop = _in_frame_stop(d["_seq"].tolist())
        discarded.extend(d.loc[has_stop, id_key].astype(str).tolist())
        d_tree = d.loc[~has_stop]
    else:
        d_tree = d
    by_clone = d_tree.groupby(clone_key, sort=False).indices
    clones = [c for c in clones if c in by_clone]
    seed_of = dict(zip(sorted(d[clone_key].dropna().unique(), key=str), seeds, strict=True))
    for c in clones:
        ss = seed_of[c]
        g = d_tree.iloc[by_clone[c]]
        germs = g[germline_key].dropna().astype(str).unique()
        if len(germs) != 1:
            raise ValueError(f"clone {c!r}: expected one germline, found {len(germs)}")
        mem: dict[str, list[str]] = {}
        labs, seqs = [], []
        if collapse == "none":
            labs = g[id_key].astype(str).tolist()
            seqs = g["_seq"].tolist()
            mem = {t: [t] for t in labs}
        else:
            tgroups = (
                g.groupby(list(traits), sort=True, dropna=False).indices.values() if traits else [np.arange(len(g))]
            )
            ids_all = g[id_key].astype(str).to_numpy()
            seq_all = g["_seq"].to_numpy(dtype=object)
            row_of = {t: k for k, t in enumerate(ids_all)}
            found = []
            for pos in tgroups:
                pos = np.sort(pos)
                tips_c, dropped = _collapse_group(list(ids_all[pos]), list(seq_all[pos]), collapse)
                found.extend((min(row_of[m] for m in ms), lab, sq, ms) for lab, sq, ms in tips_c)
                discarded.extend(dropped)
            for _, lab, sq, ms in sorted(found, key=lambda r: r[0]):
                labs.append(lab)
                seqs.append(sq)
                mem[lab] = ms
        for lab, ms in mem.items():
            for x in ms:
                label_of[x] = lab
        for ms in mem.values():
            for x in ms:
                n_coll[x] = len(ms)
        if len(labs) < min_seqs or len(labs) < 2:
            continue
        jobs.append((c, labs, seqs, germs[0].upper(), mem, method, n_starts, n_ratchet, ss))
    if n_jobs == 1:
        built = [_one_clone(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            built = list(ex.map(_one_clone, jobs))
    trees = {j[0]: t for j, t in zip(jobs, built, strict=True) if t is not None}
    out = d.loc[d[clone_key].isin(list(trees))].drop(columns="_seq").copy()
    out["tree_label"] = out[id_key].astype(str).map(label_of).astype("string")
    out.loc[out[id_key].astype(str).isin(discarded), "tree_label"] = pd.NA
    out["n_collapsed"] = pd.array(out[id_key].astype(str).map(n_coll), dtype="Int64")
    return LineageTrees(trees, tips=out)


# ------------------------------------------------------------------------------------------------ reading trees


def _parse_newick(text: str) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """Parent index, branch length and label of every node of one Newick tree (node 0 = root)."""
    s = re.sub(r"\[[^\]]*\]", "", text).strip()
    tokens = re.findall(r"'(?:[^']|'')*'|[(),:;]|[^(),:;'\s]+", s)
    parent, length, label = [-1], [0.0], [None]
    cur, stack, want_len = 0, [], False
    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            parent.append(cur)
            length.append(0.0)
            label.append(None)
            cur = len(parent) - 1
        elif tok == ",":
            if not stack:
                raise ValueError("Newick: ',' outside parentheses")
            parent.append(stack[-1])
            length.append(0.0)
            label.append(None)
            cur = len(parent) - 1
        elif tok == ")":
            if not stack:
                raise ValueError("Newick: unbalanced ')'")
            cur = stack.pop()
        elif tok == ":":
            want_len = True
        elif tok == ";":
            break
        elif want_len:
            length[cur] = float(tok)
            want_len = False
        else:
            label[cur] = tok[1:-1].replace("''", "'") if tok.startswith("'") else tok
    if stack:
        raise ValueError("Newick: unbalanced '('")
    par = np.asarray(parent, dtype=np.int64)
    ch_count = np.bincount(par[par >= 0], minlength=len(par))
    lab = [lb if ch_count[k] == 0 else None for k, lb in enumerate(label)]  # internal labels (support) dropped
    return par, np.asarray(length, dtype=float), lab


def _from_node_table(
    g: pd.DataFrame, clone_id: Any, node_key: str, parent_key: str, length_key: str, label_key: str, members_key: str
) -> LineageTree:
    ids = g[node_key].to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    par = g[parent_key]
    parent = np.array([pos.get(p, -1) if pd.notna(p) else -1 for p in par], dtype=np.int64)
    if pd.api.types.is_float_dtype(par):
        look = {float(k): v for k, v in pos.items()}
        parent = np.array([look.get(float(p), -1) if pd.notna(p) else -1 for p in par], dtype=np.int64)
    length = pd.to_numeric(g[length_key], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ch_count = np.bincount(parent[parent >= 0], minlength=len(parent))
    labels = [
        str(v) if isinstance(v, str) and v != "" and ch_count[k] == 0 else None
        for k, v in enumerate(g[label_key] if label_key in g.columns else [None] * len(g))
    ]
    members: dict[str, list[str]] = {}
    if members_key in g.columns:
        for lab, m in zip(labels, g[members_key], strict=True):
            if lab is not None and isinstance(m, str) and m:
                members[lab] = [x.strip() for x in m.split(",") if x.strip()]
    return LineageTree(
        clone_id, parent, length, labels, members=members, method="node table", score=float(length.sum())
    )


def read_trees(
    source: Any,
    *,
    clone_key: str = S.CLONE,
    node_key: str = "node",
    parent_key: str = "parent",
    length_key: str | None = None,
    label_key: str = "label",
    members_key: str = "tip_ids",
    newick_key: str | None = None,
) -> LineageTrees:
    """Read lineage trees from Newick strings or files, IgPhyML / Change-O tables, or dowser node tables.

    Parameters
    ----------
    source
        * a Newick string (one tree, clone id ``"tree"``) or a dict ``{clone_id: Newick}``;
        * a path: a Newick file (one tree per line, optionally ``clone_id <TAB> Newick``), or a TSV / CSV table;
        * a DataFrame: a node table with ``node_key``, ``parent_key`` (missing for the root), a branch-length
          column and ``label_key`` (one row per node, ``clone_key`` for several trees; e.g. the node export of
          dowser trees), or a table with a clone column and a Newick column (IgPhyML / Change-O ``TREE``).
    clone_key
        Clone column of tables (``CLONE`` is accepted for IgPhyML tables).
    node_key, parent_key, label_key
        Node-table columns.
    length_key
        Branch-length column of node tables; default the first of ``length``, ``branch_mut``, ``branch_length``.
    members_key
        Node-table column listing (comma-separated) the input tips collapsed into a tree tip (dowser ``tip_ids``).
    newick_key
        Newick column of tables; default the first of ``TREE``, ``tree``, ``newick``.

    Returns
    -------
    LineageTrees
        Branch lengths as given (dowser: mutations; IgPhyML: substitutions per site unless rescaled).
    """
    if isinstance(source, str) and source.strip().startswith("(") and source.strip().endswith(";"):
        source = {"tree": source}
    if isinstance(source, Mapping) and not isinstance(source, pd.DataFrame):
        return LineageTrees({k: _newick_tree(k, v) for k, v in source.items()})
    if isinstance(source, str | Path):
        path = Path(source)
        text = path.read_text()
        first = text.lstrip()[:1]
        if first == "(":
            trees = {}
            for k, line in enumerate(x for x in text.splitlines() if x.strip()):
                trees[f"tree{k + 1}"] = _newick_tree(f"tree{k + 1}", line)
            return LineageTrees(trees)
        lines = [x for x in text.splitlines() if x.strip()]
        if lines and "\t" in lines[0] and lines[0].split("\t", 1)[1].lstrip().startswith("("):
            return LineageTrees(
                {ln.split("\t", 1)[0]: _newick_tree(ln.split("\t", 1)[0], ln.split("\t", 1)[1]) for ln in lines}
            )
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        source = pd.read_csv(path, sep=sep, low_memory=False)
    if not isinstance(source, pd.DataFrame):
        raise TypeError(f"cannot read trees from {type(source).__name__}")
    tab = source
    ck = clone_key if clone_key in tab.columns else ("CLONE" if "CLONE" in tab.columns else None)
    nk = newick_key or next((c for c in ("TREE", "tree", "newick") if c in tab.columns), None)
    if nk is not None and node_key not in tab.columns:
        keys = tab[ck].tolist() if ck else [f"tree{k + 1}" for k in range(len(tab))]
        return LineageTrees({k: _newick_tree(k, t) for k, t in zip(keys, tab[nk], strict=True)})
    lk = length_key or next((c for c in ("length", "branch_mut", "branch_length") if c in tab.columns), None)
    for c in (node_key, parent_key):
        if c not in tab.columns:
            raise KeyError(f"read_trees: node table lacks column {c!r}")
    if lk is None:
        raise KeyError("read_trees: node table has no branch-length column (length, branch_mut or branch_length)")
    if ck is None:
        return LineageTrees({"tree": _from_node_table(tab, "tree", node_key, parent_key, lk, label_key, members_key)})
    trees = {
        k: _from_node_table(g, k, node_key, parent_key, lk, label_key, members_key)
        for k, g in tab.groupby(ck, sort=False)
    }
    return LineageTrees(trees)


def _newick_tree(clone_id: Any, text: str) -> LineageTree:
    par, length, lab = _parse_newick(text)
    return LineageTree(clone_id, par, length, lab, method="newick", score=float(length.sum()))


# ------------------------------------------------------------------------------------------------- tip tables


def _hamming_to_germline(seqs: Sequence[str], germ: str, positions: tuple[int, int] | None) -> np.ndarray:
    width = max(max((len(s) for s in seqs), default=0), len(germ))
    a = np.frombuffer("".join(s.upper().ljust(width, "N") for s in seqs).encode("ascii", "replace"), np.uint8)
    a = a.reshape(len(seqs), width)
    g = np.frombuffer(germ.upper().ljust(width, "N").encode("ascii", "replace"), np.uint8)
    lo, hi = (0, width) if positions is None else (positions[0] - 1, positions[1])
    a, g = a[:, lo:hi], g[lo:hi]
    ka = np.isin(a, np.frombuffer(b"ACGT", np.uint8))
    kg = np.isin(g, np.frombuffer(b"ACGT", np.uint8))
    return ((a != g[None, :]) & ka & kg[None, :]).sum(1)


def tip_table(
    trees: LineageTrees,
    tips: pd.DataFrame | None = None,
    *,
    clone_key: str = S.CLONE,
    id_key: str = "sequence_id",
    sequence_key: str = "sequence_alignment",
    germline_key: str = "germline_alignment_d_mask",
    positions: tuple[int, int] | None = None,
    x_key: str = S.X,
    y_key: str = S.Y,
    keep: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per input tip of every tree, ready for the lineage tests of :mod:`spaclone.tl`.

    Parameters
    ----------
    trees
        Output of :func:`build_trees` or :func:`read_trees`.
    tips
        The tip table (one row per input tip). Default: ``trees.tips`` (trees built here). For trees read from
        files, tips are matched to tree tips through the tree's ``members`` (e.g. dowser ``tip_ids``) or, without
        them, by ``id_key`` = tree tip label.
    clone_key, id_key, sequence_key, germline_key
        Columns of ``tips``.
    positions
        1-based inclusive alignment columns over which mutations from the germline are counted (e.g. ``(1, 312)``:
        IMGT V positions without the junction); default all columns. Only columns where both are A / C / G / T
        count.
    x_key, y_key
        Coordinate columns of ``tips`` (written as ``x_um`` / ``y_um``).
    keep
        Further columns of ``tips`` to carry (default: all except sequences).

    Returns
    -------
    pandas.DataFrame
        ``clone_id`` (the tree), ``sequence_id`` (the tree tip holding the row; shared by collapsed tips),
        ``tip_id`` (the input id), ``n_collapsed``, ``depth`` (mutations from the germline along the tree),
        ``mutations`` (differences to the germline, when sequences are available), ``x_um``, ``y_um`` and the kept
        columns (region, isotype, locus, ...). Tips that are in no tree are left out.
    """
    if tips is None:
        tips = trees.tips
        if tips is None:
            raise ValueError("tip_table: pass the tip table (trees read from files carry no tips)")
    for c in (clone_key, id_key):
        if c not in tips.columns:
            raise KeyError(f"tip_table: column {c!r} not in the tip table")
    rows = []
    built = "tree_label" in tips.columns
    for c, g in tips.groupby(clone_key, sort=False):
        if c not in trees:
            continue
        t = trees[c]
        depth = t.depths()
        lab_depth = {lab: depth[k] for k, lab in enumerate(t.label) if lab is not None}
        if built:
            labs = g["tree_label"].astype(str).to_numpy()
        else:
            inv = {m: lab for lab, ms in t.members.items() for m in ms}
            labs = np.array([inv.get(str(x), str(x)) for x in g[id_key]], dtype=object)
        ok = np.array([lab in lab_depth for lab in labs])
        sub = g.loc[ok]
        labs = labs[ok]
        n_coll = (
            sub["n_collapsed"].to_numpy()
            if built
            else np.array([len(t.members.get(lab, [lab])) for lab in labs], dtype=np.int64)
        )
        r = pd.DataFrame(
            {
                "clone_id": c,
                "sequence_id": labs,
                "tip_id": sub[id_key].astype(str).to_numpy(),
                "n_collapsed": n_coll,
                "depth": [lab_depth[lab] for lab in labs],
            },
            index=sub.index,
        )
        gk = germline_key if germline_key in sub.columns else "germline_alignment"
        if sequence_key in sub.columns and gk in sub.columns and len(sub):
            germ = str(sub[gk].dropna().iloc[0]) if sub[gk].notna().any() else ""
            r["mutations"] = _hamming_to_germline(sub[sequence_key].fillna("").astype(str).tolist(), germ, positions)
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["clone_id", "sequence_id", "tip_id", "n_collapsed", "depth"])
    out = pd.concat(rows)
    skip = {sequence_key, germline_key, "germline_alignment", clone_key, id_key, "tree_label", "n_collapsed"}
    extra = [c for c in (keep if keep is not None else tips.columns) if c not in skip and c not in out.columns]
    for c in extra:
        name = {x_key: S.X, y_key: S.Y}.get(c, c)
        out[name] = tips.loc[out.index, c].to_numpy()
    return out.reset_index(drop=True)


def parsimony_score(tree: LineageTree, sequences: Mapping[str, str]) -> float:
    """Parsimony length of a given tree topology on given tip sequences (e.g. to score a dowser tree).

    Parameters
    ----------
    tree
        Any rooted tree; multifurcations are allowed (Hartigan's rule: a node with k children costs k minus the
        largest number of children sharing a state).
    sequences
        Aligned sequence of every tip label (the germline under ``"Germline"``); characters other than A / C / G /
        T are missing data. Tips without a sequence are missing data everywhere.

    Returns
    -------
    float
        Minimum number of changes on the tree (unrooted, so the root position does not matter).

    References
    ----------
    Fitch 1971, doi:10.2307/2412116; Hartigan 1973, doi:10.2307/2529676.
    """
    width = max((len(v) for v in sequences.values()), default=0)
    ch = tree.children()
    sets: list[np.ndarray | None] = [None] * len(tree.parent)
    total = 0
    for v in tree._postorder(ch):
        if not ch[v]:
            seq = sequences.get(tree.label[v], "") if tree.label[v] is not None else ""
            sets[v] = _BIT[np.frombuffer(seq.upper().ljust(width, "N").encode("ascii", "replace"), np.uint8)]
            continue
        kids = np.vstack([sets[c] for c in ch[v]])
        counts = np.stack([((kids & b) > 0).sum(0) for b in (1, 2, 4, 8)])
        mx = counts.max(0)
        total += int((len(ch[v]) - mx).sum())
        sets[v] = sum(np.where(counts[k] == mx, np.uint8(1 << k), np.uint8(0)) for k in range(4)).astype(np.uint8)
    return float(total)


# ----------------------------------------------------------------------------------------------- tree comparison


def compare_trees(
    a: LineageTree, b: LineageTree, *, drop: Sequence[str] = (GERMLINE,), min_length: float = 0.5
) -> dict[str, float]:
    """Robinson-Foulds distance and divergence correlation between two trees of the same tips.

    Parameters
    ----------
    a, b
        Trees whose tip labels overlap (only common labels are compared).
    drop
        Labels removed before comparing splits (the germline: the comparison is then unrooted).
    min_length
        Edges shorter than this (mutations) are collapsed in both trees first (0.5: zero-length edges).

    Returns
    -------
    dict
        ``rf`` (splits in one tree only), ``rf_norm`` (rf / total splits of both trees; 0 = same topology, 1 =
        nothing shared), ``splits_a``, ``splits_b``, ``tips`` (common labels) and ``div_cor`` (Pearson correlation
        of the germline-to-tip path lengths over the common tips).

    References
    ----------
    Robinson & Foulds 1981, doi:10.1016/0025-5564(81)90043-2.
    """
    common = (set(a.tips()) & set(b.tips())) - set(drop)

    def restricted(t: LineageTree) -> set[frozenset]:
        sp, keep = t.splits(drop=drop, min_length=min_length)
        out = set()
        ref = min(common) if common else None
        for s in sp:
            s2 = frozenset(s & common)
            if ref is not None and ref in s2:
                s2 = frozenset(common - s2)
            if 2 <= len(s2) <= len(common) - 2:
                out.add(s2)
        return out

    sa, sb = restricted(a), restricted(b)
    rf = len(sa ^ sb)
    denom = len(sa) + len(sb)
    da, db = a.depths(), b.depths()
    ia = {lab: da[k] for k, lab in enumerate(a.label) if lab in common}
    ib = {lab: db[k] for k, lab in enumerate(b.label) if lab in common}
    keys = sorted(common)
    x = np.array([ia[k] for k in keys])
    y = np.array([ib[k] for k in keys])
    cor = float(np.corrcoef(x, y)[0, 1]) if len(keys) > 2 and x.std() > 0 and y.std() > 0 else np.nan
    return {
        "rf": float(rf),
        "rf_norm": rf / denom if denom else 0.0,
        "splits_a": float(len(sa)),
        "splits_b": float(len(sb)),
        "tips": float(len(common)),
        "div_cor": cor,
    }
