"""Deduplication: UMI error correction and (spot, clone) units.

* :func:`collapse_umis` groups the reads (or records) of one spot and chain whose UMIs are identical or one
  sequencing error apart into molecules, as UMI-tools does (``"directional"`` adjacency by default, connected
  components with ``"hamming"``, exact UMIs with ``"unique"``). Reads are summed; the molecule's clone and sequence
  are the majority (by reads) of its reads.
* :func:`collapse_spot_clone` makes one unit per (spot, clone) when the data carry no molecule identifier: in
  Stereo-XCR-seq meta tables that export the coordinate barcode (CID) as the UMI, PCR duplicates can only be
  collapsed to spot-molecule units (archive step A5: one unit = one (DNB, clone) pair).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spaclone import _schema as S
from spaclone.pp.metrics import _cols

__all__ = ["collapse_spot_clone", "collapse_umis"]


def _resolve(air: pd.DataFrame, by, defaults, fn: str) -> list[str]:
    if by is None:
        return [c for c in defaults if c in air.columns]
    cols = _cols(by)
    missing = [c for c in cols if c not in air.columns]
    if missing:
        raise KeyError(f"{fn}: column(s) {missing} not in the table")
    return cols


def _first_rows(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Position of the first row of each group code (codes numbered by first appearance)."""
    first = np.empty(n_groups, np.int64)
    first[codes[::-1]] = np.arange(len(codes))[::-1]
    return first


def collapse_spot_clone(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    reads_key: str = S.READS,
    library_key: str = S.LIBRARY,
    count_key: str | None = "n_rows",
) -> pd.DataFrame:
    """One unit per (spot, clone): collapse the records of a clone on a spot, summing their reads.

    Use it when the table has no molecule identifier, e.g. Stereo-XCR-seq meta tables whose UMI column is a
    copy of the coordinate barcode: all reads of one clone on one DNB then count as one spot-molecule unit (PCR
    duplicates collapsed), as in the archive (A5 ``make_units``; C1 after satellite merging).

    Parameters
    ----------
    air
        Receptor table (reads or records).
    by
        Columns defining a unit. ``None`` (default) = ``library_key`` (when present), ``spot_key`` and ``clone_key``.
    spot_key, clone_key, reads_key, library_key
        Column names. Records with a missing clone form one unit per spot.
    count_key
        Column added with the number of input rows per unit (the read count of a read-level table); ``None`` skips.

    Returns
    -------
    pandas.DataFrame
        One row per unit, in the order of first appearance, with every column of the unit's first row, ``reads_key``
        replaced by the sum over the unit (NaN when all are unknown) and ``count_key``. The index is a new
        ``RangeIndex``.

    Examples
    --------
    >>> units = collapse_spot_clone(reads)  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    keys = _resolve(air, by, (library_key, spot_key, clone_key), "collapse_spot_clone")
    if not keys:
        raise ValueError("collapse_spot_clone: no unit columns")
    for k in keys:
        if k not in air.columns:
            raise KeyError(f"collapse_spot_clone: column {k!r} not in the table")
    n = len(air)
    if n == 0:
        out = air.copy().reset_index(drop=True)
        if count_key is not None:
            out[count_key] = pd.Series(dtype=np.int64)
        return out
    codes = air.groupby(keys, sort=False, dropna=False, observed=True).ngroup().to_numpy(np.int64)
    n_u = int(codes.max()) + 1
    first = _first_rows(codes, n_u)
    out = air.iloc[first].reset_index(drop=True)
    if reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        r = air[reads_key].to_numpy(dtype=float)
        fin = np.isfinite(r)
        tot = np.bincount(codes, weights=np.where(fin, r, 0.0), minlength=n_u)
        any_fin = np.bincount(codes, weights=fin.astype(float), minlength=n_u) > 0
        vals = np.where(any_fin, tot, np.nan)
        if pd.api.types.is_integer_dtype(air[reads_key]):
            out[reads_key] = vals.astype(air[reads_key].dtype)
        else:
            out[reads_key] = vals
    if count_key is not None:
        out[count_key] = np.bincount(codes, minlength=n_u).astype(np.int64)
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "collapse_spot_clone": {"by": keys, "n_in": int(n), "n_out": int(n_u)},
    }
    return out


# ------------------------------------------------------------------------------------------------ UMIs
def _umi_pairs(group: np.ndarray, umis: np.ndarray, max_dist: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (i < j) of UMI-table rows in the same group, of equal length, with Hamming distance <= max_dist.

    Candidates come from the pigeonhole principle: two strings of equal length within ``max_dist`` substitutions share
    at least one of ``max_dist + 1`` aligned segments exactly (the substring index of UMI-tools).
    """
    n = len(umis)
    if n < 2 or max_dist < 1:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    lengths = np.fromiter((len(u) for u in umis), dtype=np.int64, count=n)
    size = np.bincount(group)
    multi = size[group] >= 2
    ii, jj = [], []
    for L in np.unique(lengths[multi]):
        idx = np.flatnonzero(multi & (lengths == L))
        if len(idx) < 2 or L == 0:
            continue
        k = min(max_dist + 1, int(L))
        bounds = np.linspace(0, L, k + 1).round().astype(int)
        arr = np.frombuffer("".join(umis[idx]).encode("ascii", "replace"), dtype=np.uint8).reshape(len(idx), L)
        cand = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            seg = np.ascontiguousarray(arr[:, a:b])
            seg_code = np.unique(seg.view(np.dtype((np.void, b - a))).ravel(), return_inverse=True)[1]
            seg_code = seg_code.astype(np.int64).ravel()
            key = pd.DataFrame({"g": group[idx], "s": seg_code, "p": np.arange(len(idx))})
            m = key.merge(key, on=["g", "s"], suffixes=("_a", "_b"))
            m = m[m.p_a < m.p_b]
            cand.append(m[["p_a", "p_b"]].to_numpy())
        if not cand:
            continue
        c = np.unique(np.concatenate(cand), axis=0) if len(cand) > 1 else cand[0]
        if len(c) == 0:
            continue
        d = (arr[c[:, 0]] != arr[c[:, 1]]).sum(axis=1)
        keep = d <= max_dist
        ii.append(idx[c[keep, 0]])
        jj.append(idx[c[keep, 1]])
    if not ii:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ii).astype(np.int64), np.concatenate(jj).astype(np.int64)


def _directional(n: int, order_rank: np.ndarray, counts: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """UMI-tools ``directional`` grouping; returns the representative node of every node.

    Edge a -> b when ``count(a) >= 2 count(b) - 1``. Nodes are visited by decreasing count (ties by UMI); each
    unvisited node starts a breadth-first search along directed edges, and every node goes to the first search that
    reaches it (UMI-tools ``_get_connected_components_adjacency`` followed by ``_group_directional``).
    """
    rep = np.arange(n)
    if len(i) == 0:
        return rep
    a = np.concatenate([i, j])
    b = np.concatenate([j, i])
    ok = counts[a] >= 2 * counts[b] - 1
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return rep
    o = np.lexsort((order_rank[b], a))
    a, b = a[o], b[o]
    start = np.searchsorted(a, np.arange(n), side="left")
    stop = np.searchsorted(a, np.arange(n), side="right")
    touched = np.unique(np.concatenate([a, b]))
    nodes = touched[np.argsort(order_rank[touched], kind="stable")]
    assigned = np.zeros(n, bool)
    for s in nodes:
        if assigned[s]:
            continue
        # every node reachable from s; nodes already taken by an earlier search keep their group
        seen = {int(s)}
        queue = [int(s)]
        while queue:
            u = queue.pop()
            for v in b[start[u] : stop[u]]:
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        for v in seen:
            if not assigned[v]:
                assigned[v] = True
                rep[v] = s
    return rep


def collapse_umis(
    air: pd.DataFrame,
    *,
    umi_key: str = S.UMI,
    method: Literal["directional", "hamming", "unique"] = "directional",
    max_dist: int = 1,
    by: str | Sequence[str] | None = None,
    vote_key: str | None = None,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    clone_key: str = S.CLONE,
    junction_key: str = S.JUNCTION_AA,
) -> pd.DataFrame:
    """Collapse reads into molecules by UMI, correcting UMI sequencing errors.

    Within each molecule stratum (by default the library, the spot and the chain), UMIs are grouped:

    * ``"directional"`` (UMI-tools default): UMI ``a`` absorbs ``b`` when they are at most ``max_dist`` substitutions
      apart and ``count(a) >= 2 count(b) - 1``; groups grow from the most abundant UMI along such edges;
    * ``"hamming"``: connected components of UMIs within ``max_dist`` (UMI-tools ``cluster``);
    * ``"unique"``: identical UMIs only (no error correction).

    Each group is one molecule. Its UMI is the group's most abundant UMI (ties: the smallest string), its reads are
    the sum over all of its rows, and its clone and sequence come from the majority value of ``vote_key`` by reads
    (ties: the smallest value); the molecule's other columns are those of the first row carrying that value.

    Parameters
    ----------
    air
        Receptor table at read level (one row per read) or with reads per row in ``reads_key``.
    umi_key
        Column with the UMI sequence. Rows with a missing UMI are each kept as their own molecule.
    method
        ``"directional"``, ``"hamming"`` or ``"unique"``.
    max_dist
        Maximum Hamming distance between UMIs of one molecule (UMIs of unequal length are never joined).
    by
        Columns that define the molecule stratum besides the UMI. ``None`` (default) = whichever of ``library_key``,
        ``spot_key`` and ``locus_key`` are present.
    vote_key
        Column whose majority value (by reads) names the molecule: default ``clone_key`` when present, else
        ``junction_key``; ``None`` with neither column present = the first row.
    reads_key
        Reads per row; without this column every row counts as one read. Missing values count as one read.
    spot_key, locus_key, library_key, clone_key, junction_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per molecule (order of the first row of each molecule) with the input columns, ``umi_key`` set to the
        molecule's UMI, ``reads_key`` to its reads, and ``n_umis`` (distinct raw UMIs merged), ``n_rows`` (input rows)
        and ``vote_share`` (share of the molecule's reads carrying the majority value). The index is a new
        ``RangeIndex``; the method is recorded in ``attrs["spaclone"]["collapse_umis"]``.

    Notes
    -----
    The directional rule assumes that an erroneous UMI is seen in fewer than half as many reads as its parent; a
    molecule therefore needs read counts per UMI, which is why the rule acts on reads, not on records. Spatial
    barcodes are part of the stratum, so UMIs on different spots are never joined.

    References
    ----------
    Smith, Heger & Sudbery 2017, doi:10.1101/gr.209601.116 (UMI-tools: unique, cluster, adjacency, directional);
    Kivioja et al. 2012, doi:10.1038/nmeth.1778 (UMIs); Shugay et al. 2014, doi:10.1038/nmeth.2960 (UMI consensus of
    immune-receptor molecules).

    Examples
    --------
    >>> molecules = collapse_umis(reads, umi_key="umi", method="directional")  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if method not in ("directional", "hamming", "unique"):
        raise ValueError(f"method must be 'directional', 'hamming' or 'unique', got {method!r}")
    if int(max_dist) != max_dist or max_dist < 0:
        raise ValueError(f"max_dist must be a non-negative integer, got {max_dist!r}")
    max_dist = int(max_dist)
    if umi_key not in air.columns:
        raise KeyError(f"collapse_umis: column {umi_key!r} not in the table")
    numeric_umi = pd.api.types.is_numeric_dtype(air[umi_key])
    if numeric_umi and method != "unique" and max_dist > 0:
        raise TypeError("collapse_umis: numeric UMI codes can only be collapsed with method='unique'")
    strata = _resolve(air, by, (library_key, spot_key, locus_key), "collapse_umis")
    if vote_key is None:
        vote_key = clone_key if clone_key in air.columns else (junction_key if junction_key in air.columns else None)
    elif vote_key not in air.columns:
        raise KeyError(f"collapse_umis: column {vote_key!r} not in the table")
    n = len(air)
    if n == 0:
        out = air.copy().reset_index(drop=True)
        for c, dt in (("n_umis", np.int64), ("n_rows", np.int64), ("vote_share", float)):
            out[c] = pd.Series(dtype=dt)
        return out

    if reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        w = air[reads_key].to_numpy(dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
    else:
        w = np.ones(n)
    gcode = (
        air.groupby(strata, sort=False, dropna=False, observed=True).ngroup().to_numpy(np.int64)
        if strata
        else np.zeros(n, np.int64)
    )
    umi = air[umi_key].astype("string")
    missing = umi.isna().to_numpy()
    # distinct (stratum, UMI) nodes
    node_key = pd.DataFrame({"g": gcode, "u": umi.fillna("").to_numpy(dtype=object)})
    node_key.loc[missing, "u"] = [f"\x00{i}" for i in np.flatnonzero(missing)]  # missing UMIs never merge
    node = node_key.groupby(["g", "u"], sort=True).ngroup().to_numpy(np.int64)
    n_node = int(node.max()) + 1
    nodes = node_key.groupby(["g", "u"], sort=True).size().reset_index()[["g", "u"]]
    ng = nodes.g.to_numpy(np.int64)
    nu = nodes.u.to_numpy(dtype=object)
    counts = np.bincount(node, weights=w, minlength=n_node)
    # order: by stratum, decreasing count, then UMI string (nodes are sorted by (g, u) already)
    order = np.lexsort((np.arange(n_node), -counts, ng))
    rank = np.empty(n_node, np.int64)
    rank[order] = np.arange(n_node)
    is_missing = np.array([u.startswith("\x00") for u in nu], dtype=bool)
    if method == "unique" or max_dist == 0:
        rep = np.arange(n_node)
    else:
        sel = np.flatnonzero(~is_missing)
        pi, pj = _umi_pairs(ng[sel], nu[sel].astype(str), max_dist)
        pi, pj = sel[pi], sel[pj]
        if method == "hamming":
            adj = coo_matrix((np.ones(len(pi), np.int8), (pi, pj)), shape=(n_node, n_node))
            _, comp = connected_components(adj, directed=False)
            best = pd.Series(rank).groupby(comp).idxmin()  # most abundant node of each component
            rep = best.reindex(comp).to_numpy(np.int64)
        else:
            rep = _directional(n_node, rank, counts, pi, pj)
    mol_node = rep[node]  # representative node of every row
    mol, mol_ids = pd.factorize(mol_node)  # molecule code per row, numbered by first appearance
    n_mol = len(mol_ids)
    # majority vote by reads within each molecule; missing values vote only when nothing else does
    first = _first_rows(mol, n_mol)
    win_w = None
    if vote_key is not None:
        vcode, vvals = pd.factorize(air[vote_key].astype("string"), sort=True)  # sorted: ties -> smallest value
        has = vcode >= 0
        win_w = np.zeros(n_mol)
        if has.any():
            nv = np.int64(len(vvals))
            pair = mol[has].astype(np.int64) * nv + vcode[has]
            pw = pd.Series(w[has]).groupby(pair).sum()
            pm = pw.index.to_numpy() // nv
            pv = pw.index.to_numpy() % nv
            tab = pd.DataFrame({"m": pm, "v": pv, "w": pw.to_numpy()}).sort_values(
                ["m", "w", "v"], ascending=[True, False, True], kind="stable"
            )
            win = tab.drop_duplicates("m")
            win_v = np.full(n_mol, -1, np.int64)
            win_v[win.m.to_numpy(np.int64)] = win.v.to_numpy(np.int64)
            win_w[win.m.to_numpy(np.int64)] = win.w.to_numpy(dtype=float)
            cand = np.flatnonzero(has & (vcode == win_v[mol]))
            first = first.copy()
            first[mol[cand][::-1]] = cand[::-1]
        no_vote = win_w == 0  # molecules whose rows all lack the value: share of the missing value
        if no_vote.any():
            win_w[no_vote] = np.bincount(mol, weights=w, minlength=n_mol)[no_vote]
    out = air.iloc[first].reset_index(drop=True)
    tot = np.bincount(mol, weights=w, minlength=n_mol)
    rep_umi = [None if nu[i].startswith("\x00") else nu[i] for i in mol_ids]
    if not numeric_umi:  # numeric UMI codes are only collapsed exactly: every row of a molecule has its UMI
        out[umi_key] = pd.array(rep_umi, dtype=air[umi_key].dtype)
    out[reads_key] = tot
    out["n_umis"] = pd.Series(node).groupby(mol).nunique().reindex(range(n_mol)).to_numpy(np.int64)
    out["n_rows"] = np.bincount(mol, minlength=n_mol).astype(np.int64)
    out["vote_share"] = (win_w / tot) if win_w is not None else 1.0
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "collapse_umis": {
            "method": method,
            "max_dist": max_dist,
            "by": strata,
            "vote_key": vote_key,
            "n_rows": int(n),
            "n_umis": int((~is_missing).sum()),
            "n_molecules": int(n_mol),
        },
    }
    return out
