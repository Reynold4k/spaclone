"""Error variants: satellite clones one amino acid (or a synonymous change) away from a larger parent.

PCR and sequencing errors, and for heavy chains somatic hypermutation, turn one rearrangement into a cloud of small
"satellite" clones whose junctions differ from the parent's by one residue (or not at all at the amino-acid level).
:func:`find_error_variants` finds them with the rule of the archive's denoising ladder (C1 step 4):

* same chain, same J gene, same junction length and at least one shared V family;
* at most ``max_aa_diff`` amino-acid differences between the junctions;
* the satellite has at most ``heavy_max_share`` (IGH: 20%) or ``light_max_share`` (light chains and TCR: 5%) of the
  parent's size. Heavy chains get the larger share because independent heavy rearrangements almost never differ by
  one residue, while one lineage carries somatic variants; distinct kappa / lambda (and TCR) rearrangements often
  differ by one junctional residue, so only error-sized satellites are merged there;
* clones are visited from the largest down; a clone that became a satellite is never a parent, and each satellite
  goes to the first (largest) parent that claims it. No clone is removed for being small.

:func:`merge_error_variants` relabels the satellites' records to their parent and then re-collapses to distinct
(spot, clone) units, because an error copy keeps the barcode of the molecule it came from: a satellite record on a
spot that also holds its parent is the same molecule (21-23% of satellite units in the archive).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pp.dedup import collapse_spot_clone

__all__ = ["VariantMerge", "find_error_variants", "merge_error_variants", "v_families"]

_SPLIT = re.compile(r"[,|;]")
_FAMILY = re.compile(r"^((?:IG[HKL]|TR[ABGD])V\d+)")


def _genes(call) -> list[str]:
    """Genes of a V call: every comma / bar separated call, allele dropped (``"IGHV1-15*01,IGHV1-2*02"``)."""
    if call is None or (isinstance(call, float) and np.isnan(call)) or call is pd.NA:
        return []
    out = []
    for c in _SPLIT.split(str(call)):
        g = c.split("*")[0].strip()
        if g and g.lower() not in ("nan", "none", "<na>"):
            out.append(g)
    return out


def v_families(calls: pd.Series | Sequence, *, level: Literal["family", "gene"] = "family") -> pd.Series:
    """Set of V families (or genes) named in each V call.

    ``"IGHV1-15*01,IGHV14-2*01"`` -> ``frozenset({"IGHV1", "IGHV14"})``; a gene whose name has no family number
    (``IG[HKL]V<n>`` / ``TR[ABGD]V<n>``) is its own family; a missing call gives the empty set.

    Parameters
    ----------
    calls
        V calls (AIRR ``v_call``), possibly with several comma-separated genes.
    level
        ``"family"`` or ``"gene"``.

    Returns
    -------
    pandas.Series
        One ``frozenset`` per call (index kept for a Series).
    """
    if level not in ("family", "gene"):
        raise ValueError(f"level must be 'family' or 'gene', got {level!r}")
    s = calls if isinstance(calls, pd.Series) else pd.Series(list(calls))
    uniq = pd.unique(s.astype(object).to_numpy())
    look = {}
    for u in uniq:
        gs = _genes(u)
        if level == "family":
            gs = [(m.group(1) if (m := _FAMILY.match(g)) else g) for g in gs]
        look[u] = frozenset(gs)
    return pd.Series([look[u] for u in s.astype(object).to_numpy()], index=s.index, dtype=object)


def _mode(values: pd.Series, groups: np.ndarray) -> pd.Series:
    """Most frequent non-missing value per group (ties: smallest value); missing when a group has none."""
    t = pd.DataFrame({"g": groups, "v": values.astype("string").to_numpy()}).dropna()
    if t.empty:
        return pd.Series(dtype=object)
    c = t.groupby(["g", "v"], sort=True).size().reset_index(name="n")
    c = c.sort_values(["g", "n", "v"], ascending=[True, False, True], kind="stable").drop_duplicates("g")
    return c.set_index("g").v


def _clone_table(
    air: pd.DataFrame,
    *,
    size: str,
    clone_key: str,
    locus_key: str,
    junction_key: str,
    v_key: str | None,
    j_key: str | None,
    reads_key: str,
    spot_key: str,
    v_level: str,
) -> pd.DataFrame:
    from spaclone.pp.clones import gene_name

    codes, ids = pd.factorize(air[clone_key])
    ok = codes >= 0
    sub = air[ok]
    c = codes[ok]
    n_c = len(ids)
    t = pd.DataFrame({"clone": np.asarray(ids, dtype=object)})
    t["locus"] = _mode(sub[locus_key], c).reindex(range(n_c)).to_numpy(dtype=object)
    t["junction"] = _mode(sub[junction_key], c).reindex(range(n_c)).to_numpy(dtype=object)
    t["j_gene"] = (
        _mode(gene_name(sub[j_key]), c).reindex(range(n_c)).to_numpy(dtype=object) if j_key is not None else None
    )
    if v_key is not None:
        fam = v_families(sub[v_key], level=v_level)
        fams = pd.Series(fam.to_numpy(), index=c).groupby(level=0).agg(lambda x: frozenset().union(*x))
        t["v_families"] = fams.reindex(range(n_c)).to_numpy(dtype=object)
    else:
        t["v_families"] = [frozenset()] * n_c
    rec = np.bincount(c, minlength=n_c).astype(float)
    if reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        r = sub[reads_key].to_numpy(dtype=float)
        reads = np.bincount(c, weights=np.where(np.isfinite(r), r, 0.0), minlength=n_c)
    else:
        reads = rec.copy()
    if size == "records":
        t["size"] = rec
    elif size == "reads":
        t["size"] = reads
    else:
        sp = pd.factorize(sub[spot_key])[0]
        t["size"] = pd.Series(sp).groupby(c).nunique().reindex(range(n_c)).to_numpy(dtype=float)
    t["reads"] = reads
    return t


def _hamming_pairs(seqs: np.ndarray, group: np.ndarray, max_diff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (i, j), i != j unordered, of equal-length strings in the same group within ``max_diff`` mismatches."""
    n = len(seqs)
    if n < 2:
        e = np.empty(0, np.int64)
        return e, e, e
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=n)
    gl = pd.factorize(pd.Series(group.astype(np.int64) * 10_000 + lengths))[0]
    size = np.bincount(gl)
    ii, jj, dd = [], [], []
    for L in np.unique(lengths[size[gl] >= 2]):
        idx = np.flatnonzero((lengths == L) & (size[gl] >= 2))
        if len(idx) < 2 or L == 0:
            continue
        arr = np.frombuffer("".join(seqs[idx]).encode("ascii", "replace"), dtype=np.uint8).reshape(len(idx), L)
        g = gl[idx]
        if max_diff == 0:
            k = 1
        else:
            k = min(max_diff + 1, int(L))
        bounds = np.linspace(0, L, k + 1).round().astype(int)
        cand = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            seg = np.ascontiguousarray(arr[:, a:b])
            code = np.unique(seg.view(np.dtype((np.void, max(b - a, 1)))).ravel(), return_inverse=True)[1].ravel()
            key = pd.DataFrame({"g": g, "s": code.astype(np.int64), "p": np.arange(len(idx))})
            m = key.merge(key, on=["g", "s"], suffixes=("_a", "_b"))
            m = m[m.p_a < m.p_b]
            cand.append(m[["p_a", "p_b"]].to_numpy())
        c = np.unique(np.concatenate(cand), axis=0) if len(cand) > 1 else cand[0]
        if len(c) == 0:
            continue
        d = (arr[c[:, 0]] != arr[c[:, 1]]).sum(axis=1)
        keep = d <= max_diff
        ii.append(idx[c[keep, 0]])
        jj.append(idx[c[keep, 1]])
        dd.append(d[keep])
    if not ii:
        e = np.empty(0, np.int64)
        return e, e, e
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(dd)


def find_error_variants(
    air: pd.DataFrame,
    *,
    heavy_max_share: float = 0.20,
    light_max_share: float = 0.05,
    max_aa_diff: int = 1,
    heavy_loci: Sequence[str] = ("IGH",),
    size: Literal["records", "reads", "spots"] = "records",
    match_j: bool = True,
    match_v: Literal["family", "gene"] | None = "family",
    order: Sequence | None = None,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    junction_key: str = S.JUNCTION_AA,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
) -> pd.DataFrame:
    """Find satellite clones one amino acid (or synonymous) away from a larger parent clone.

    Parameters
    ----------
    air
        Receptor table with clone ids and, per record, the chain, junction (amino acids), V and J calls. The clone's
        junction and J gene are its most frequent values; its V families are the union over its records.
    heavy_max_share, light_max_share
        Largest satellite size, as a share of the parent's size, for chains in ``heavy_loci`` and for all other
        chains.
    max_aa_diff
        Largest number of amino-acid differences between the junctions (equal length; 0 = synonymous variants only).
    heavy_loci
        Chains that use ``heavy_max_share``. Only IGH by default: T-cell receptors do not hypermutate, so their 1-aa
        neighbours are treated like light chains.
    size
        Clone size: ``"records"`` (rows, e.g. spot-molecule units; the archive), ``"reads"`` or ``"spots"``.
    match_j
        Require the same J gene.
    match_v
        Require at least one shared V ``"family"`` (default; robust to the ambiguous V calls of 5'-truncated reads) or
        ``"gene"``; ``None`` ignores V.
    order
        Clone ids in the order in which clones are visited (first = visited first), replacing the default order
        within each chain; clones not listed follow in the default order. Use it to replay another tool's order
        (e.g. the archive's, whose ties in size were broken by an unstable sort). The size rule still applies.
    clone_key, locus_key, junction_key, v_key, j_key, reads_key, spot_key
        Column names (``v_key`` / ``j_key`` may be absent when ``match_v=None`` / ``match_j=False``).

    Returns
    -------
    pandas.DataFrame
        One row per satellite: ``satellite``, ``parent``, ``locus``, ``satellite_junction``, ``parent_junction``,
        ``aa_diff``, ``satellite_size``, ``parent_size``, ``size_share`` (satellite / parent), in the order the parents
        were visited. Clone ids keep their input type.

    Notes
    -----
    Clones are visited by decreasing size; ties are broken by decreasing reads and then by clone id, so the result
    does not depend on the row order. (The archive broke ties by the order of an unstable sort; ``order`` replays
    any given order.) A satellite of a satellite is not merged (no chains), as in the archive.

    References
    ----------
    Shugay et al. 2014, doi:10.1038/nmeth.2960 (MiTCR / MIGEC error correction: low-abundance 1-mismatch neighbours
    of a large clonotype are merged into it); Bolotin et al. 2015, doi:10.1038/nmeth.3364 (MiXCR clustering of
    erroneous clonotypes); Gupta et al. 2017, doi:10.4049/jimmunol.1601850 (V, J, junction length and similarity
    define B-cell clones).

    Examples
    --------
    >>> sat = find_error_variants(units)  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    for name, v in (("heavy_max_share", heavy_max_share), ("light_max_share", light_max_share)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must lie in [0, 1], got {v}")
    if int(max_aa_diff) != max_aa_diff or max_aa_diff < 0:
        raise ValueError(f"max_aa_diff must be a non-negative integer, got {max_aa_diff!r}")
    if size not in ("records", "reads", "spots"):
        raise ValueError(f"size must be 'records', 'reads' or 'spots', got {size!r}")
    if match_v not in ("family", "gene", None):
        raise ValueError(f"match_v must be 'family', 'gene' or None, got {match_v!r}")
    need = [clone_key, locus_key, junction_key]
    if match_v is not None:
        need.append(v_key)
    if match_j:
        need.append(j_key)
    if size == "spots":
        need.append(spot_key)
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(f"find_error_variants: column(s) {missing} not in the table")
    cols = ["satellite", "parent", "locus", "satellite_junction", "parent_junction", "aa_diff"]
    cols += ["satellite_size", "parent_size", "size_share"]
    if len(air) == 0:
        return pd.DataFrame(columns=cols)
    t = _clone_table(
        air,
        size=size,
        clone_key=clone_key,
        locus_key=locus_key,
        junction_key=junction_key,
        v_key=v_key if match_v is not None else None,
        j_key=j_key if match_j else None,
        reads_key=reads_key,
        spot_key=spot_key,
        v_level=match_v or "family",
    )
    t = t[t.junction.notna() & t.locus.notna() & (t["size"] > 0)].reset_index(drop=True)
    if len(t) < 2:
        return pd.DataFrame(columns=cols)
    # visiting order: by chain, decreasing size, decreasing reads, then clone id (as a string)
    cid = t.clone.astype(str).to_numpy()
    keys = [cid, -t.reads.to_numpy(), -t["size"].to_numpy()]
    if order is not None:
        pos = {str(c): i for i, c in enumerate(order)}
        given = np.array([pos.get(c, len(pos)) for c in cid], dtype=np.int64)  # unlisted clones after listed ones
        keys.append(given)
    keys.append(t.locus.astype(str).to_numpy())
    visit = np.lexsort(tuple(keys))
    rank = np.empty(len(t), np.int64)
    rank[visit] = np.arange(len(t))
    lcode = pd.factorize(t.locus.astype(str))[0]
    i, j, d = _hamming_pairs(t.junction.astype(str).to_numpy(), lcode, int(max_aa_diff))
    if len(i) == 0:
        return pd.DataFrame(columns=cols)
    # orient each pair parent (earlier) -> satellite (later)
    swap = rank[i] > rank[j]
    p = np.where(swap, j, i)
    s = np.where(swap, i, j)
    share = np.where(t.locus.astype(str).isin(list(heavy_loci)).to_numpy(), heavy_max_share, light_max_share)
    ok = t["size"].to_numpy()[s] <= share[p] * t["size"].to_numpy()[p] + 1e-12
    if match_j:
        jg = t.j_gene.astype("string")
        ok &= (jg.to_numpy(dtype=object)[s] == jg.to_numpy(dtype=object)[p]) & jg.notna().to_numpy()[p]
    if match_v is not None:
        vf = t.v_families.to_numpy()
        ok &= np.fromiter((bool(vf[a] & vf[b]) for a, b in zip(p, s, strict=True)), dtype=bool, count=len(p))
    p, s, d = p[ok], s[ok], d[ok]
    if len(p) == 0:
        return pd.DataFrame(columns=cols)
    o = np.lexsort((rank[s], rank[p]))
    p, s, d = p[o], s[o], d[o]
    is_sat = np.zeros(len(t), bool)
    parent = np.full(len(t), -1, np.int64)
    dist = np.zeros(len(t), np.int64)
    sat_order = []
    bounds = np.flatnonzero(np.r_[True, p[1:] != p[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        pp = p[a]
        if is_sat[pp]:
            continue
        for k in range(a, b):
            ss = s[k]
            if not is_sat[ss]:
                is_sat[ss] = True
                parent[ss] = pp
                dist[ss] = d[k]
                sat_order.append(ss)
    sat = np.asarray(sat_order, dtype=np.int64)
    par = parent[sat]
    out = pd.DataFrame(
        {
            "satellite": t.clone.to_numpy()[sat],
            "parent": t.clone.to_numpy()[par],
            "locus": t.locus.to_numpy()[sat],
            "satellite_junction": t.junction.to_numpy()[sat],
            "parent_junction": t.junction.to_numpy()[par],
            "aa_diff": dist[sat],
            "satellite_size": t["size"].to_numpy()[sat],
            "parent_size": t["size"].to_numpy()[par],
        }
    )
    out["size_share"] = out.satellite_size / out.parent_size
    out.attrs["spaclone"] = {
        "find_error_variants": {
            "heavy_max_share": heavy_max_share,
            "light_max_share": light_max_share,
            "max_aa_diff": int(max_aa_diff),
            "heavy_loci": list(heavy_loci),
            "size": size,
            "match_j": match_j,
            "match_v": match_v,
            "order": "given" if order is not None else "size, reads, clone id",
        }
    }
    return out


class VariantMerge(NamedTuple):
    """Result of :func:`merge_error_variants` (unpack as ``merged, variants = merge_error_variants(air)``).

    Attributes
    ----------
    air
        The table with satellites relabelled to their parent (and re-collapsed to (spot, clone) units).
    variants
        The satellite map used (see :func:`find_error_variants`).
    """

    air: pd.DataFrame
    variants: pd.DataFrame


def merge_error_variants(
    air: pd.DataFrame,
    *,
    heavy_max_share: float = 0.20,
    light_max_share: float = 0.05,
    max_aa_diff: int = 1,
    variants: pd.DataFrame | None = None,
    recollapse: bool = True,
    premerge_key: str | None = "clone_id_premerge",
    heavy_loci: Sequence[str] = ("IGH",),
    size: Literal["records", "reads", "spots"] = "records",
    match_j: bool = True,
    match_v: Literal["family", "gene"] | None = "family",
    order: Sequence | None = None,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    junction_key: str = S.JUNCTION_AA,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
    library_key: str = S.LIBRARY,
) -> VariantMerge:
    """Merge satellite clones into their parent and re-collapse to distinct (spot, clone) units.

    Parameters
    ----------
    air
        Receptor table (e.g. spot-molecule units).
    heavy_max_share, light_max_share, max_aa_diff, heavy_loci, size, match_j, match_v, order
        The satellite rule, see :func:`find_error_variants` (ignored when ``variants`` is given).
    variants
        A satellite map (``satellite`` and ``parent`` columns) to apply instead of computing one from ``air``, e.g.
        one computed on the whole table before sinks and off-tissue records were removed, as the archive did.
    recollapse
        After relabelling, collapse records of the same clone on the same spot into one unit (reads summed,
        :func:`collapse_spot_clone`). An error copy keeps the barcode of the molecule it came from, so a satellite
        record on a spot that also holds its parent is that molecule.
    premerge_key
        Column added with each record's clone id before merging (``None`` skips it).
    clone_key, locus_key, junction_key, v_key, j_key, reads_key, spot_key, library_key
        Column names. Relabelled records also take the parent's junction, V and J calls (the parent's most frequent
        values in ``air``; the map's ``parent_junction`` when the parent has no record in ``air``).

    Returns
    -------
    VariantMerge
        ``air``: the merged table (``attrs["spaclone"]["merge_error_variants"]`` holds the counts: satellites,
        parents, relabelled records, records absorbed by the re-collapse and their share of the relabelled records).
        ``variants``: the map used.

    References
    ----------
    Shugay et al. 2014, doi:10.1038/nmeth.2960; Bolotin et al. 2015, doi:10.1038/nmeth.3364.

    Examples
    --------
    >>> merged, sat = merge_error_variants(units)  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if clone_key not in air.columns:
        raise KeyError(f"merge_error_variants: column {clone_key!r} not in the table")
    if variants is None:
        variants = find_error_variants(
            air,
            heavy_max_share=heavy_max_share,
            light_max_share=light_max_share,
            max_aa_diff=max_aa_diff,
            heavy_loci=heavy_loci,
            size=size,
            match_j=match_j,
            match_v=match_v,
            order=order,
            clone_key=clone_key,
            locus_key=locus_key,
            junction_key=junction_key,
            v_key=v_key,
            j_key=j_key,
            reads_key=reads_key,
            spot_key=spot_key,
        )
    elif not {"satellite", "parent"} <= set(variants.columns):
        raise ValueError("variants must have 'satellite' and 'parent' columns")
    out = air.copy()
    if premerge_key is not None:
        out[premerge_key] = air[clone_key].to_numpy()
    sat = pd.Series(variants.parent.to_numpy(), index=pd.Index(variants.satellite.to_numpy()))
    sat = sat[~sat.index.duplicated()]
    cur = air[clone_key]
    mapped = cur.map(sat)
    moved = mapped.notna().to_numpy() & cur.notna().to_numpy()
    n_moved = int(moved.sum())
    if n_moved:
        new = cur.astype(object).to_numpy().copy()
        new[moved] = mapped.to_numpy(dtype=object)[moved]
        out[clone_key] = pd.array(new, dtype=air[clone_key].dtype)
        # parent's representative sequence and calls
        stay = air[~moved]
        for col in (junction_key, v_key, j_key):
            if col not in air.columns:
                continue
            look = _mode(stay[col], pd.factorize(stay[clone_key])[0]) if len(stay) else pd.Series(dtype=object)
            ids = pd.factorize(stay[clone_key])[1] if len(stay) else []
            rep = pd.Series(look.reindex(range(len(ids))).to_numpy(dtype=object), index=pd.Index(ids))
            if col == junction_key and "parent_junction" in variants.columns:
                fill = pd.Series(variants.parent_junction.to_numpy(), index=pd.Index(variants.parent.to_numpy()))
                fill = fill[~fill.index.duplicated()]
                rep = rep.combine_first(fill)
            vals = out[col].astype(object).to_numpy().copy()
            par = pd.Series(new[moved]).map(rep).to_numpy(dtype=object)
            has = pd.notna(par)
            idx = np.flatnonzero(moved)
            vals[idx[has]] = par[has]
            out[col] = pd.array(vals, dtype=air[col].dtype) if not pd.api.types.is_numeric_dtype(air[col]) else vals
    n_before = len(out)
    absorbed = 0
    if recollapse:
        if spot_key not in out.columns:
            raise KeyError(f"merge_error_variants: recollapse needs column {spot_key!r}")
        keys = [c for c in (library_key, spot_key) if c in out.columns] + [clone_key]
        out = collapse_spot_clone(out, by=keys, reads_key=reads_key, count_key=None)
        absorbed = n_before - len(out)
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "merge_error_variants": {
            "satellites": int(len(sat)),
            "parents": int(sat.nunique()),
            "records_relabelled": n_moved,
            "records_absorbed": int(absorbed),
            "absorbed_share_of_relabelled": float(absorbed / n_moved) if n_moved else float("nan"),
            "recollapse": recollapse,
        },
    }
    return VariantMerge(out, variants)
