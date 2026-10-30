"""Clone (clonotype) definitions from V gene, J gene and junction.

Two rules, both from the archive:

* ``"exact"``: the same V gene, the same J gene (alleles dropped) and the same junction. This is the T1 / TR1 rule; T
  cells do not hypermutate, so for TCRs it is the only sensible rule.
* ``"vj_length_identity"``: the Dandelion-like BCR rule of S4: the same V gene, J gene and junction length, and a
  junction identity of at least ``identity`` (Hamming), joined by single linkage. This groups somatically mutated
  members of one B-cell lineage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spaclone import _schema as S

__all__ = ["define_clones", "gene_name"]


def gene_name(calls: pd.Series | Sequence[str]) -> pd.Series:
    """Gene of a V / D / J / C call: the first of several comma-separated calls, without the allele.

    ``"IGHV1-15*01"`` -> ``"IGHV1-15"``; ``"TRBV12-1*01,TRBV12-2*01"`` -> ``"TRBV12-1"``; missing stays missing.
    """
    s = pd.Series(calls, dtype="string") if not isinstance(calls, pd.Series) else calls.astype("string")
    g = s.str.split(",").str[0].str.split("*").str[0].str.strip()
    return g.mask(g == "", pd.NA)


def define_clones(
    air: pd.DataFrame,
    *,
    method: Literal["exact", "vj_length_identity"] = "exact",
    identity: float = 0.85,
    by: str | Sequence[str] = (S.LOCUS,),
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
    key_added: str = S.CLONE,
) -> pd.DataFrame:
    """Assign a clone id to every record.

    Parameters
    ----------
    air
        Receptor table with V calls, J calls and junctions.
    method
        ``"exact"``: same V gene, J gene and junction. ``"vj_length_identity"``: same V gene, J gene and junction
        length, junction identity >= ``identity`` (Hamming distance / length <= 1 - ``identity``), single linkage
        (the Dandelion-like rule of the archive's S4).
    identity
        Minimum junction identity for ``"vj_length_identity"`` (0.85 = Dandelion's default).
    by
        Strata clustered separately (default: the chain). Their values are the first fields of the id, e.g.
        ``"IGH|IGHV2-2|IGHJ3|CARNSHYYGSRYPFAYW"``; pass ``by=()`` for ids without them (V gene names already name the
        chain). For a clone shared between sections (public clones), keep the library out of ``by``.
    v_key, j_key, junction_key
        Columns with the V call, the J call and the junction (amino acids by default; ``"junction"`` for nucleotides).
    key_added
        Column written with the clone id.

    Returns
    -------
    pandas.DataFrame
        A copy of ``air`` with ``key_added``: ``"<by>|V|J|junction"`` (exact) or ``"<by>|V|J|L<length>|<k>"`` with
        component number ``k`` (vj_length_identity; components numbered in the sorted order of their first junction).
        Records missing a V gene, J gene or junction get ``<NA>``.

    Notes
    -----
    Genes are taken from the first call with the allele dropped (:func:`gene_name`). Ids do not depend on the row
    order. Single linkage means two junctions can share a clone through a chain of close neighbours although they
    differ by more than the identity threshold themselves.

    References
    ----------
    Gupta et al. 2017, doi:10.4049/jimmunol.1601850 (clonal grouping by V, J, junction length and junction
    similarity); Stephenson et al. 2021, doi:10.1038/s41591-021-01329-2 (Dandelion).
    """
    if method not in ("exact", "vj_length_identity"):
        raise ValueError(f"method must be 'exact' or 'vj_length_identity', got {method!r}")
    if not 0 < identity <= 1:
        raise ValueError(f"identity must be in (0, 1], got {identity}")
    by_cols = [by] if isinstance(by, str) else list(by)
    need = [v_key, j_key, junction_key, *by_cols]
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(
            f"define_clones: column(s) {missing} not in the table"
            + (" (pass by=())" if any(c in by_cols for c in missing) else "")
        )
    out = air.copy()
    v = gene_name(air[v_key])
    j = gene_name(air[j_key])
    jx = air[junction_key].astype("string")
    jx = jx.mask(jx == "", pd.NA)
    ok = (v.notna() & j.notna() & jx.notna()).to_numpy(dtype=bool)
    if by_cols:
        prefix = air[by_cols[0]].astype("string").fillna("NA")
        for c in by_cols[1:]:
            prefix = prefix + "|" + air[c].astype("string").fillna("NA")
        prefix = prefix + "|"
    else:
        prefix = pd.Series("", index=air.index, dtype="string")
    if method == "exact":
        ids = (prefix + v + "|" + j + "|" + jx).to_numpy(dtype=object, na_value=None)
    else:
        ids = np.full(len(air), None, dtype=object)
        ids[ok] = _identity_clusters(
            *(s_.to_numpy(dtype=object, na_value=None)[ok] for s_ in (prefix, v, j, jx)), identity
        )
    out[key_added] = pd.array(ids, dtype="string")
    return out


def _identity_clusters(prefix, v, j, junction, identity: float) -> np.ndarray:
    """Single-linkage components of junctions within (prefix, V, J, length) groups; returns one id per input."""
    lengths = np.fromiter((len(s) for s in junction), dtype=np.int64, count=len(junction))
    tab = pd.DataFrame({"p": prefix, "v": v, "j": j, "L": lengths, "s": junction})
    uniq = tab.drop_duplicates().sort_values(["p", "v", "j", "L", "s"]).reset_index(drop=True)
    comp = np.zeros(len(uniq), dtype=np.int64)
    for _, pos in uniq.groupby(["p", "v", "j", "L"], sort=False).indices.items():
        if len(pos) < 2:
            continue
        seqs = uniq.s.to_numpy()[pos]
        length = int(uniq.L.iat[pos[0]])
        max_mm = int(np.floor(length * (1.0 - identity) + 1e-6))
        comp[pos] = _hamming_components(seqs, length, max_mm)
    uniq["id"] = [f"{p}{a}|{b}|L{n}|{k}" for p, a, b, n, k in zip(uniq.p, uniq.v, uniq.j, uniq.L, comp, strict=True)]
    key = pd.MultiIndex.from_frame(uniq[["p", "v", "j", "L", "s"]])
    look = pd.Series(uniq["id"].to_numpy(), index=key)
    return look.reindex(pd.MultiIndex.from_frame(tab[["p", "v", "j", "L", "s"]])).to_numpy(dtype=object)


def _hamming_components(seqs: np.ndarray, length: int, max_mm: int) -> np.ndarray:
    """Components (numbered by first member, in input order) of sequences within ``max_mm`` mismatches."""
    n = len(seqs)
    if max_mm <= 0 or length == 0:
        return np.arange(n)
    arr = np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(n, length)
    rows, cols = [], []
    block = max(1, int(2e7 // max(1, n * length)))
    for start in range(0, n, block):
        sub = arr[start : start + block]
        d = (sub[:, None, :] != arr[None, :, :]).sum(-1)
        r, c = np.nonzero(d <= max_mm)
        keep = c > r + start
        rows.append(r[keep] + start)
        cols.append(c[keep])
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    adj = coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    first = np.full(lab.max() + 1, n, np.int64)
    np.minimum.at(first, lab, np.arange(n))
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[lab]
