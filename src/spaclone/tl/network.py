"""Clonotype similarity networks: clonotypes linked by junction distance, their components, and where they sit.

Nodes are clonotypes (chain, V gene, J gene and junction; the genes only when ``same_v`` / ``same_j``); two nodes are
linked when their junctions are within ``cutoff`` (Hamming: same length, substitutions only; Levenshtein: edit
distance). Connected components are families of related receptors: somatic variants of one B-cell lineage, or, across
unrelated cells, convergent receptors of similar specificity. The spatial layer summarises each component: how many
units, sections and regions it covers, its centroid, its radius of gyration and the length of its minimum spanning
tree.

The distances are computed by scirpy's distance calculators when scirpy is installed (``backend="scirpy"``, the
reference), and otherwise by a numpy implementation that gives the same edges: candidate pairs from exact matches of
``cutoff + 1`` junction segments (Hamming, pigeonhole principle) or of deletion neighbourhoods (Levenshtein, symmetric
deletion), each verified exactly.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from collections.abc import Iterator
from itertools import combinations
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spaclone import _schema as S
from spaclone.pp.clones import gene_name
from spaclone.tl.dispersion import _mst_length
from spaclone.tl.repertoire import _first_seen_codes, _keys, clone_units

__all__ = ["ClonotypeNetwork", "clonotype_network"]

_CHUNK = 2_000_000


class ClonotypeNetwork(NamedTuple):
    """Result of :func:`clonotype_network`.

    Attributes
    ----------
    nodes
        One row per clonotype: ``node_id``, chain, V and J gene (when part of the node), ``junction``, ``component``,
        ``n_records``, ``n_units``, ``degree``.
    edges
        One row per link: ``node_a``, ``node_b`` (``node_a < node_b``) and ``distance``.
    components
        One row per connected component, with its size and spatial summary (see :func:`clonotype_network`).
    membership
        Component of every record (``<NA>`` for records without a junction or a required gene), indexed like the input.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    components: pd.DataFrame
    membership: pd.Series


# ------------------------------------------------------------------------------------------- candidate pairs


def _group_pairs(inv: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """All pairs (i, j), i != j, of rows that share a value of ``inv``, in chunks (each unordered pair once)."""
    if len(inv) < 2:
        return
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    cut = np.flatnonzero(np.diff(sinv)) + 1
    starts = np.r_[0, cut]
    sizes = np.r_[cut, len(sinv)] - starts
    small = (sizes >= 2) & (sizes <= 64)
    for m in np.unique(sizes[small]):
        a, b = np.triu_indices(int(m), 1)
        sel = starts[small & (sizes == m)]
        step = max(1, _CHUNK // len(a))
        for s in range(0, len(sel), step):
            base = sel[s : s + step, None]
            yield order[(base + a).ravel()], order[(base + b).ravel()]
    for st, m in zip(starts[sizes > 64], sizes[sizes > 64], strict=True):
        members = order[st : st + m]
        r0 = 0
        while r0 < m - 1:
            lens = m - 1 - np.arange(r0, m - 1)
            r1 = r0 + max(1, int(np.searchsorted(np.cumsum(lens), _CHUNK)))
            k = np.arange(r0, min(r1, m - 1))
            ln = m - 1 - k
            tot = int(ln.sum())
            first = np.cumsum(ln) - ln
            i = np.repeat(k, ln)
            j = i + 1 + (np.arange(tot) - np.repeat(first, ln))
            yield members[i], members[j]
            r0 = int(k[-1]) + 1


def _seq_matrix(seqs: np.ndarray) -> np.ndarray:
    length = len(seqs[0]) if len(seqs) else 0
    return np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(len(seqs), length)


def _hamming_edges(seqs: np.ndarray, strata: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    out_i, out_j, out_d = [], [], []
    for length in np.unique(lengths):
        idx = np.flatnonzero(lengths == length)
        if len(idx) < 2:
            continue
        arr = _seq_matrix(seqs[idx])
        g = strata[idx].astype(np.int64)
        found_i, found_j = [], []
        if length <= cutoff:
            for i, j in _group_pairs(g):
                found_i.append(i)
                found_j.append(j)
        else:
            bounds = np.linspace(0, length, cutoff + 2).astype(np.int64)
            gb = g.view(np.uint8).reshape(-1, 8)
            for s in range(cutoff + 1):
                seg = np.ascontiguousarray(np.hstack([gb, arr[:, bounds[s] : bounds[s + 1]]]))
                key = seg.view(np.dtype((np.void, seg.shape[1]))).ravel()
                inv = np.unique(key, return_inverse=True)[1].ravel()
                for i, j in _group_pairs(inv):
                    d = (arr[i] != arr[j]).sum(1)
                    keep = d <= cutoff
                    found_i.append(i[keep])
                    found_j.append(j[keep])
        if not found_i:
            continue
        i = np.concatenate(found_i)
        j = np.concatenate(found_j)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        u = np.unique(lo * len(idx) + hi)
        lo, hi = u // len(idx), u % len(idx)
        out_i.append(idx[lo])
        out_j.append(idx[hi])
        out_d.append((arr[lo] != arr[hi]).sum(1))
    if not out_i:
        e = np.empty(0, np.int64)
        return e, e, e
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d).astype(np.int64)


def _levenshtein_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Edit distances of the rows of two uint8 matrices (P x La) and (P x Lb), vectorised over rows."""
    p, la = a.shape
    lb = b.shape[1]
    j = np.arange(lb + 1)
    prev = np.broadcast_to(j, (p, lb + 1)).copy()
    t = np.empty((p, lb + 1), np.int64)
    for i in range(1, la + 1):
        cost = (a[:, i - 1 : i] != b).astype(np.int64)
        t[:, 0] = i
        t[:, 1:] = np.minimum(prev[:, 1:] + 1, prev[:, :-1] + cost)
        prev = np.minimum.accumulate(t - j, axis=1) + j
    return prev[:, lb]


def _levenshtein_pairs(seqs: np.ndarray, i: np.ndarray, j: np.ndarray, cutoff: int) -> np.ndarray:
    """Edit distance of each candidate pair (cutoff + 1 when the lengths alone exceed the cutoff)."""
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    swap = lengths[i] > lengths[j]
    a = np.where(swap, j, i)
    b = np.where(swap, i, j)
    la, lb = lengths[a], lengths[b]
    d = np.full(len(a), cutoff + 1, np.int64)
    ok = (lb - la) <= cutoff
    pair = pd.DataFrame({"la": la[ok], "lb": lb[ok], "k": np.flatnonzero(ok)})
    for (x, y), grp in pair.groupby(["la", "lb"]).indices.items():
        k = pair["k"].to_numpy()[grp]
        if x == 0:
            d[k] = y
            continue
        d[k] = _levenshtein_batch(_seq_matrix(seqs[a[k]]), _seq_matrix(seqs[b[k]]))
    return d


def _deletion_variants(s: str, k: int) -> set[str]:
    out = {s}
    for r in range(1, min(k, len(s)) + 1):
        for pos in combinations(range(len(s)), r):
            out.add("".join(c for q, c in enumerate(s) if q not in pos))
    return out


def _levenshtein_edges(seqs: np.ndarray, strata: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cutoff <= 2:
        rows, keys = [], []
        for r, (s, g) in enumerate(zip(seqs, strata, strict=True)):
            for v in _deletion_variants(s, cutoff):
                rows.append(r)
                keys.append(f"{g}\x00{v}")
        rows_a = np.asarray(rows, np.int64)
        inv = pd.factorize(pd.Series(keys, dtype=object))[0]
        cand = [(rows_a[i], rows_a[j]) for i, j in _group_pairs(inv)]
    else:
        cand = list(_group_pairs(np.asarray(strata, np.int64)))
    if not cand:
        e = np.empty(0, np.int64)
        return e, e, e
    i = np.concatenate([c[0] for c in cand])
    j = np.concatenate([c[1] for c in cand])
    keep = i != j
    lo, hi = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    u = np.unique(lo * len(seqs) + hi)
    lo, hi = u // len(seqs), u % len(seqs)
    d = _levenshtein_pairs(seqs, lo, hi, cutoff)
    sel = d <= cutoff
    return lo[sel], hi[sel], d[sel]


def _edges_numpy(seqs: np.ndarray, strata: np.ndarray, metric: str, cutoff: int):
    if metric == "hamming":
        return _hamming_edges(seqs, strata, cutoff)
    return _levenshtein_edges(seqs, strata, cutoff)


def _edges_scirpy(seqs: np.ndarray, strata: np.ndarray, metric: str, cutoff: int, n_jobs: int):
    from scirpy.ir_dist.metrics import HammingDistanceCalculator, LevenshteinDistanceCalculator

    out_i, out_j, out_d = [], [], []
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        if metric == "hamming":
            useq, inv = np.unique(seqs, return_inverse=True)
            if len(useq) >= 2:
                mat = HammingDistanceCalculator(n_jobs=n_jobs, cutoff=cutoff).calc_dist_mat(np.asarray(useq, dtype=str))
                mat = mat.tocoo()
                up = mat.row < mat.col
                sp = pd.DataFrame({"a": mat.row[up], "b": mat.col[up], "d": mat.data[up].astype(np.int64) - 1})
                nd = pd.DataFrame({"s": inv.ravel(), "g": strata, "node": np.arange(len(seqs))})
                e = sp.merge(nd.rename(columns={"s": "a", "g": "ga", "node": "na"}), on="a")
                e = e.merge(nd.rename(columns={"s": "b", "g": "gb", "node": "nb"}), on="b")
                e = e[e["ga"] == e["gb"]]
                out_i.append(e["na"].to_numpy())
                out_j.append(e["nb"].to_numpy())
                out_d.append(e["d"].to_numpy())
        else:
            calc = LevenshteinDistanceCalculator(cutoff=cutoff, n_jobs=n_jobs)
            for _, pos in pd.Series(np.arange(len(seqs))).groupby(strata).indices.items():
                if len(pos) < 2:
                    continue
                mat = calc.calc_dist_mat(np.asarray(seqs[pos], dtype=str)).tocoo()
                up = mat.row < mat.col
                out_i.append(pos[mat.row[up]])
                out_j.append(pos[mat.col[up]])
                out_d.append(mat.data[up].astype(np.int64) - 1)
    if not out_i:
        e = np.empty(0, np.int64)
        return e, e, e
    i, j, d = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return lo, hi, d


def _backend(backend: str) -> str:
    if backend == "auto":
        return "scirpy" if importlib.util.find_spec("scirpy") is not None else "numpy"
    if backend not in ("numpy", "scirpy"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'scirpy', got {backend!r}")
    if backend == "scirpy" and importlib.util.find_spec("scirpy") is None:
        raise ImportError("backend='scirpy' needs scirpy (pip install 'spaclone[scverse]')")
    return backend


# ----------------------------------------------------------------------------------------------- the network


def clonotype_network(
    air: pd.DataFrame,
    *,
    metric: Literal["hamming", "levenshtein"] = "hamming",
    cutoff: int = 1,
    same_v: bool = True,
    same_j: bool = True,
    sequence_key: str = S.JUNCTION_AA,
    backend: Literal["auto", "numpy", "scirpy"] = "auto",
    unit: Literal["locus", "spot", "record"] = "locus",
    region_key: str | None = None,
    radius_um: float = 15.0,
    n_jobs: int = 1,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> ClonotypeNetwork:
    """Sequence-similarity network of clonotypes, its connected components and their spatial summary.

    Parameters
    ----------
    air
        Receptor table with junctions (and V / J calls when ``same_v`` / ``same_j``).
    metric
        ``"hamming"`` (substitutions; junctions of different length are never linked, as in scirpy) or
        ``"levenshtein"`` (insertions, deletions and substitutions).
    cutoff
        Largest distance of a link.
    same_v, same_j
        Link only clonotypes with the same V (J) gene; the gene is then part of the node. Nodes of different chains are
        never linked.
    sequence_key
        Junction column (amino acids by default; ``"junction"`` for nucleotides).
    backend
        ``"scirpy"`` (scirpy's ``HammingDistanceCalculator`` / ``LevenshteinDistanceCalculator``, the reference),
        ``"numpy"`` (no extra dependency, same edges) or ``"auto"`` (scirpy when installed).
    unit
        Unit of the spatial summaries (see :func:`spaclone.tl.clone_units`; units are counted per node).
    region_key
        Optional record-level region column; components then report the regions they cover.
    radius_um
        Linkage distance of locus units in µm.
    n_jobs
        Threads / workers handed to scirpy.
    v_key, j_key, clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names (``clone_key`` is not used to build nodes; it is reported per component when present).

    Returns
    -------
    ClonotypeNetwork
        ``components`` has ``component``, ``n_nodes``, ``n_records``, ``n_units``, ``n_spots``, the chain, the most
        common V and J gene, ``n_clones`` (distinct values of ``clone_key`` when present), ``n_sections``, and with
        coordinates ``x_um`` / ``y_um`` (centroid of its units), ``rg_um`` (radius of gyration of its units) and
        ``mst_um`` (length of the minimum spanning tree of its units); with ``region_key`` also ``n_regions``,
        ``regions`` and ``main_region_share``. Components are numbered by their first node in sorted node order.

    Notes
    -----
    A component of a clonotype network is not a clone: single linkage chains similar junctions, so unrelated
    receptors can join through intermediates, and light chains (IGK / IGL) and short public TCR junctions are shared
    by unrelated cells. To ask whether a component's members sit together, pass ``membership`` as the clone column of
    :func:`spaclone.tl.clone_dispersion`.

    References
    ----------
    Hamming 1950, doi:10.1002/j.1538-7305.1950.tb00463.x; Levenshtein 1966 (Sov Phys Dokl 10:707); Sturm et al. 2020,
    doi:10.1093/bioinformatics/btaa611 (scirpy); Bashford-Rogers et al. 2013, doi:10.1101/gr.154815.113 (BCR
    networks); Garbe et al. 2005, doi:10.1007/11564096_32 (symmetric deletion candidates).
    """
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")
    cutoff = int(cutoff)
    if cutoff < 0:
        raise ValueError("cutoff must be >= 0")
    be = _backend(backend)
    need = [sequence_key] + ([v_key] if same_v else []) + ([j_key] if same_j else [])
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(f"clonotype_network: column(s) {missing} not in the table")
    seq = air[sequence_key].astype("string")
    ok = (seq.notna() & (seq != "")).to_numpy(dtype=bool)
    parts = {"locus": air[locus_key].astype("string") if locus_key in air.columns else pd.Series("NA", index=air.index)}
    if same_v:
        parts["v"] = gene_name(air[v_key])
        ok &= parts["v"].notna().to_numpy(dtype=bool)
    if same_j:
        parts["j"] = gene_name(air[j_key])
        ok &= parts["j"].notna().to_numpy(dtype=bool)
    parts["locus"] = parts["locus"].fillna("NA")
    key = pd.DataFrame({k: v.to_numpy(dtype=object, na_value=None) for k, v in parts.items()}, index=air.index)
    key["junction"] = seq.to_numpy(dtype=object, na_value=None)
    key = key.loc[ok]
    kcols = list(key.columns)
    nodes = key.drop_duplicates().sort_values(kcols).reset_index(drop=True)
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    rec_node = key.merge(nodes, on=kcols, how="left")["node_id"].to_numpy(dtype=np.int64)
    scode = _first_seen_codes(nodes, kcols[:-1])
    seqs = nodes["junction"].to_numpy(dtype=object)
    if be == "scirpy":
        ei, ej, ed = _edges_scirpy(seqs, scode, metric, cutoff, n_jobs)
    else:
        ei, ej, ed = _edges_numpy(seqs, scode, metric, cutoff)
    order = np.lexsort((ej, ei))
    edges = pd.DataFrame({"node_a": ei[order], "node_b": ej[order], "distance": ed[order]})
    n_nodes = len(nodes)
    adj = coo_matrix((np.ones(len(ei), np.int8), (ei, ej)), shape=(n_nodes, n_nodes))
    _, lab = connected_components(adj, directed=False)
    if n_nodes:
        first = np.full(lab.max() + 1, n_nodes, np.int64)
        np.minimum.at(first, lab, np.arange(n_nodes))
        rank = np.empty(len(first), np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        comp = rank[lab]
    else:
        comp = np.empty(0, np.int64)
    nodes["component"] = comp
    nodes["n_records"] = np.bincount(rec_node, minlength=n_nodes)
    nodes["degree"] = np.bincount(np.r_[ei, ej], minlength=n_nodes) if n_nodes else np.empty(0, np.int64)
    nodes = nodes.rename(columns={"locus": locus_key, "v": "v_gene", "j": "j_gene"})
    membership = pd.Series(pd.array([pd.NA] * len(air), dtype="Int64"), index=air.index, name="component")
    membership.loc[key.index] = comp[rec_node]
    # units and spatial summaries
    rec = air.loc[key.index].copy()
    rec["_node"] = rec_node
    keys = _keys("_node", spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    has_xy = x_key in air.columns and y_key in air.columns
    can_units = unit == "record" or (spot_key in air.columns and (unit == "spot" or has_xy))
    comps = pd.DataFrame({"component": np.arange(int(comp.max()) + 1 if n_nodes else 0, dtype=np.int64)})
    comps["n_nodes"] = np.bincount(comp, minlength=len(comps)) if n_nodes else []
    comps["n_records"] = np.bincount(comp[rec_node], minlength=len(comps)) if n_nodes else []
    comps[locus_key] = nodes.groupby("component")[locus_key].first().reindex(comps["component"]).to_numpy()
    for g in ("v_gene", "j_gene"):
        if g in nodes.columns:
            w = nodes.groupby(["component", g])["n_records"].sum().reset_index()
            w = w.sort_values(["component", "n_records", g], ascending=[True, False, True]).drop_duplicates("component")
            comps[g] = w.set_index("component")[g].reindex(comps["component"]).to_numpy()
    if clone_key in rec.columns:
        comps["n_clones"] = (
            pd.Series(rec[clone_key].to_numpy())
            .groupby(comp[rec_node])
            .nunique()
            .reindex(comps["component"])
            .to_numpy()
        )
    if library_key in rec.columns:
        comps["n_sections"] = (
            pd.Series(rec[library_key].to_numpy()).groupby(comp[rec_node]).nunique().reindex(comps["component"])
        ).to_numpy()
    if spot_key in rec.columns:
        uid = _first_seen_codes(rec, ([library_key] if library_key in rec.columns else []) + [spot_key])
        comps["n_spots"] = pd.Series(uid).groupby(comp[rec_node]).nunique().reindex(comps["component"]).to_numpy()
    if can_units and len(rec):
        carry = [region_key] if region_key else []
        u = clone_units(rec, unit=unit, carry=carry, radius_um=radius_um, **keys).units
        ucomp = comp[u["_node"].to_numpy(dtype=np.int64)]
        nodes["n_units"] = np.bincount(u["_node"].to_numpy(dtype=np.int64), minlength=n_nodes)
        comps["n_units"] = np.bincount(ucomp, minlength=len(comps))
        if has_xy:
            xy = u[[x_key, y_key]].to_numpy(dtype=float)
            cx = np.bincount(ucomp, weights=xy[:, 0], minlength=len(comps)) / np.maximum(comps["n_units"], 1)
            cy = np.bincount(ucomp, weights=xy[:, 1], minlength=len(comps)) / np.maximum(comps["n_units"], 1)
            comps[x_key], comps[y_key] = cx, cy
            d2 = (xy[:, 0] - cx[ucomp]) ** 2 + (xy[:, 1] - cy[ucomp]) ** 2
            comps["rg_um"] = np.sqrt(
                np.bincount(ucomp, weights=d2, minlength=len(comps)) / np.maximum(comps["n_units"], 1)
            )
            mst = np.zeros(len(comps))
            for c, pos in pd.Series(np.arange(len(u))).groupby(ucomp).indices.items():
                if len(pos) >= 2:
                    mst[c] = _mst_length(xy[pos])
            comps["mst_um"] = mst
        if region_key:
            reg = pd.DataFrame({"c": ucomp, "r": u[region_key].to_numpy(dtype=object)}).dropna()
            cnt = reg.groupby(["c", "r"]).size().rename("n").reset_index()
            comps["n_regions"] = cnt.groupby("c").size().reindex(comps["component"]).fillna(0).astype(int).to_numpy()
            lists = cnt.groupby("c")["r"].agg(lambda v: ";".join(sorted(map(str, v))))
            comps["regions"] = lists.reindex(comps["component"]).to_numpy()
            top = cnt.groupby("c")["n"].max() / cnt.groupby("c")["n"].sum()
            comps["main_region_share"] = top.reindex(comps["component"]).to_numpy()
    else:
        nodes["n_units"] = np.nan
    nodes = nodes[[c for c in nodes.columns if c != "degree"] + ["degree"]]
    return ClonotypeNetwork(nodes, edges, comps, membership)
