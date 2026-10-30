"""Cross-sample contamination: public (convergent) clonotypes and carry-over between libraries.

A clonotype seen in two samples (sections, mice, libraries) is either

* **public / convergent**: the same receptor arose independently, so it is genuine in both. Evidence: it is found
  in at least ``min_individuals`` individuals of an independent reference (e.g. a single-cell atlas of the same model),
  its junction is germline-like (few non-templated N nucleotides) on a chain where that is informative, or it is a
  light chain shared at comparable abundance (light chains join V to J with few or no N additions, so the same kappa /
  lambda junction arises in many individuals);
* **carry-over**: a few molecules of a clonotype that is abundant in one library appear in another (index hopping,
  template switching or cross-contamination during library preparation). A non-public clonotype with at least
  ``min_ratio`` (100) times more molecules in one sample is carry-over in the others;
* **unexplained**: shared at comparable abundance without evidence of convergence.

The rules reproduce the archive's bulk biology analysis (D1 section E). The archive warns
that IgBLAST N counts of light chains are unreliable for 5'-truncated reads, which is why germline-likeness is used
for heavy chains only by default.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pp.variants import v_families

__all__ = ["CarryoverResult", "PublicResult", "clonotype_key", "flag_carryover", "flag_public"]


def _key_parts(
    air: pd.DataFrame,
    *,
    v_level: Literal["family", "gene"] | None,
    locus_key: str,
    v_key: str,
    j_key: str,
    junction_key: str,
) -> pd.DataFrame:
    from spaclone.pp.clones import gene_name

    need = [locus_key, junction_key, j_key] + ([v_key] if v_level is not None else [])
    missing = [c for c in need if c not in air.columns]
    if missing:
        raise KeyError(f"clonotype_key: column(s) {missing} not in the table (v_level=None drops V)")
    parts = pd.DataFrame(index=air.index)
    parts["locus"] = air[locus_key].astype("string")
    parts["j_gene"] = gene_name(air[j_key])
    parts["junction_aa"] = air[junction_key].astype("string")
    if v_level is not None:
        fam = v_families(air[v_key], level=v_level)
        parts["v_families"] = fam
        parts["v_text"] = fam.map(lambda f: ",".join(sorted(f)))
    else:
        parts["v_families"] = [frozenset()] * len(air)
        parts["v_text"] = ""
    return parts


def clonotype_key(
    air: pd.DataFrame,
    *,
    v_level: Literal["family", "gene"] | None = "family",
    locus_key: str = S.LOCUS,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
) -> pd.Series:
    """Clonotype key comparable between samples: ``"<locus>|<J gene>|<junction>|<V families>"``.

    Parameters
    ----------
    air
        Receptor table.
    v_level
        ``"family"`` (default: the sorted set of V families named in the V call, robust to the ambiguous V calls of
        5'-truncated reads, as in the archive), ``"gene"``, or ``None`` (no V).
    locus_key, v_key, j_key, junction_key
        Column names.

    Returns
    -------
    pandas.Series
        String key per record (missing when the chain, J gene or junction is missing).
    """
    p = _key_parts(air, v_level=v_level, locus_key=locus_key, v_key=v_key, j_key=j_key, junction_key=junction_key)
    key = p.locus + "|" + p.j_gene + "|" + p.junction_aa + "|" + p.v_text.astype("string")
    return pd.Series(
        key.astype(object).where(key.notna(), None).to_numpy(dtype=object),
        index=air.index,
        name="clonotype_key",
        dtype=object,
    )


class PublicResult(NamedTuple):
    """Result of :func:`flag_public` (unpack as ``air_flagged, table = flag_public(air)``).

    Attributes
    ----------
    air
        Copy of the input with ``clonotype_key`` and ``public`` (bool) columns.
    table
        One row per clonotype, see :func:`flag_public`.
    """

    air: pd.DataFrame
    table: pd.DataFrame


def _reference_individuals(
    clon: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    v_level: Literal["family", "gene"] | None,
    individual_key: str,
    locus_key: str,
    v_key: str,
    j_key: str,
    junction_key: str,
) -> pd.Series:
    """Distinct reference individuals per clonotype: same locus, J gene and junction, and a shared V family."""
    use_v = v_level is not None and v_key in reference.columns
    rp = _key_parts(
        reference,
        v_level=v_level if use_v else None,
        locus_key=locus_key,
        v_key=v_key,
        j_key=j_key,
        junction_key=junction_key,
    )
    rp["ind"] = reference[individual_key].astype("string").to_numpy()
    rp = rp.dropna(subset=["locus", "j_gene", "junction_aa", "ind"])
    left = clon[["key", "locus", "j_gene", "junction_aa", "v_families"]]
    if use_v:
        le = left.explode("v_families").dropna(subset=["v_families"]).rename(columns={"v_families": "fam"})
        re_ = rp.explode("v_families").dropna(subset=["v_families"]).rename(columns={"v_families": "fam"})
        m = le.merge(
            re_[["locus", "j_gene", "junction_aa", "fam", "ind"]], on=["locus", "j_gene", "junction_aa", "fam"]
        )
    else:
        m = left.merge(rp[["locus", "j_gene", "junction_aa", "ind"]], on=["locus", "j_gene", "junction_aa"])
    return m.groupby("key").ind.nunique()


def flag_public(
    air: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    min_individuals: int = 2,
    individual_key: str | None = None,
    reference_individual_key: str = "individual",
    n_key: str | None = None,
    max_n: float = 2,
    n_loci: Sequence[str] = ("IGH",),
    n_by: str | None = None,
    v_level: Literal["family", "gene"] | None = "family",
    locus_key: str = S.LOCUS,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
) -> PublicResult:
    """Flag public (convergent) clonotypes.

    A clonotype (see :func:`clonotype_key`) is public when

    * it is found in at least ``min_individuals`` distinct individuals, counted in ``reference`` (same chain, J gene
      and junction and a shared V family) plus, when ``individual_key`` is given, in ``air``; or
    * ``n_key`` is given, its chain is in ``n_loci`` and its junction carries at most ``max_n`` non-templated (N)
      nucleotides (germline-like junctions arise independently in many individuals).

    Parameters
    ----------
    air
        Receptor table.
    reference
        Independent repertoire with ``locus``, ``j_call``, ``junction_aa``, optionally ``v_call``, and
        ``reference_individual_key`` (e.g. a 10x scBCR atlas of the same model, one row per chain).
    min_individuals
        Individuals needed.
    individual_key
        Column of ``air`` naming the individual. Leave ``None`` when the samples of ``air`` are the ones being checked
        for carry-over: a clonotype carried over from one library to another would otherwise count as seen in two
        individuals.
    reference_individual_key
        Individual column of ``reference``.
    n_key
        Column with the number of N nucleotides of each record's junction (e.g. IgBLAST ``np1_length + np2_length``).
    max_n
        Largest N count of a germline-like junction.
    n_loci
        Chains for which germline-likeness counts (IGH by default: IgBLAST N counts of light chains from
        5'-truncated reads are unreliable, and light chains have few N additions anyway).
    n_by
        Column whose groups (samples) each count once when averaging N: the clonotype's N is the mean over groups of
        the group medians. Default: ``individual_key`` or the library column when present, else all records pooled.
    v_level, locus_key, v_key, j_key, junction_key
        See :func:`clonotype_key`.

    Returns
    -------
    PublicResult
        ``air`` with ``clonotype_key`` and ``public``; ``table`` with one row per clonotype: ``key``, ``locus``,
        ``j_gene``, ``junction_aa``, ``v_families`` (text), ``n_records``, ``n_individuals`` (in ``air``; NaN without
        ``individual_key``), ``n_reference_individuals``, ``n_additions``, ``germline_like``, ``public`` and
        ``reason`` (``"individuals"``, ``"germline-like"``, both joined by ``+``, or empty).

    Notes
    -----
    This is a rule, not a test; it has no null model.

    References
    ----------
    Venturi et al. 2008, doi:10.1038/nri2260 (convergent recombination and public receptors); Greiff et al. 2017,
    doi:10.1016/j.celrep.2017.04.054 (public antibody repertoires in mice); Elhanati et al. 2018,
    doi:10.1111/imr.12665 (sharing predicted by generation probability).

    Examples
    --------
    >>> flagged, table = flag_public(air, reference=atlas, n_key="n_additions")  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if min_individuals < 1:
        raise ValueError("min_individuals must be >= 1")
    if individual_key is not None and individual_key not in air.columns:
        raise KeyError(f"flag_public: column {individual_key!r} not in the table")
    if n_key is not None and n_key not in air.columns:
        raise KeyError(f"flag_public: column {n_key!r} not in the table")
    if reference is not None and reference_individual_key not in reference.columns:
        raise KeyError(f"flag_public: reference lacks column {reference_individual_key!r}")
    parts = _key_parts(air, v_level=v_level, locus_key=locus_key, v_key=v_key, j_key=j_key, junction_key=junction_key)
    key = (parts.locus + "|" + parts.j_gene + "|" + parts.junction_aa + "|" + parts.v_text.astype("string")).rename(
        "clonotype_key"
    )
    ok = key.notna().to_numpy()
    kp = parts[ok].assign(key=key[ok].to_numpy())
    clon = kp.drop_duplicates("key")[["key", "locus", "j_gene", "junction_aa", "v_families", "v_text"]]
    clon = clon.sort_values("key", kind="stable").reset_index(drop=True)
    clon["n_records"] = kp.groupby("key").size().reindex(clon.key).to_numpy(np.int64)
    if individual_key is not None:
        ind = air.loc[ok, individual_key].astype("string").to_numpy()
        n_ind = pd.Series(ind).groupby(kp.key.to_numpy()).nunique()
        clon["n_individuals"] = n_ind.reindex(clon.key).fillna(0).to_numpy(dtype=float)
    else:
        clon["n_individuals"] = np.nan
    if reference is not None and len(clon):
        nref = _reference_individuals(
            clon,
            reference,
            v_level=v_level,
            individual_key=reference_individual_key,
            locus_key=locus_key,
            v_key=v_key,
            j_key=j_key,
            junction_key=junction_key,
        )
        clon["n_reference_individuals"] = nref.reindex(clon.key).fillna(0).to_numpy(np.int64)
    else:
        clon["n_reference_individuals"] = 0
    total = clon.n_reference_individuals.to_numpy(dtype=float) + np.nan_to_num(clon.n_individuals.to_numpy(float))
    by_ind = total >= min_individuals
    if n_key is not None:
        nb = n_by if n_by is not None else (individual_key if individual_key is not None else S.LIBRARY)
        nv = pd.to_numeric(air.loc[ok, n_key], errors="coerce").to_numpy(dtype=float)
        grp = air.loc[ok, nb].astype("string").fillna("").to_numpy() if nb in air.columns else np.zeros(int(ok.sum()))
        per = pd.DataFrame({"k": kp.key.to_numpy(), "g": grp, "n": nv}).groupby(["k", "g"]).n.median()
        n_add = per.groupby(level="k").mean()
        clon["n_additions"] = n_add.reindex(clon.key).to_numpy(dtype=float)
        germ = clon.locus.isin(list(n_loci)).to_numpy() & (clon.n_additions.to_numpy(dtype=float) <= max_n)
    else:
        clon["n_additions"] = np.nan
        germ = np.zeros(len(clon), bool)
    clon["germline_like"] = germ
    clon["public"] = by_ind | germ
    reason = np.where(by_ind, "individuals", "")
    reason = np.where(germ, np.where(by_ind, "individuals+germline-like", "germline-like"), reason)
    clon["reason"] = reason
    table = clon.drop(columns="v_families").rename(columns={"v_text": "v_families"})
    out = air.copy()
    out["clonotype_key"] = pd.Series(
        key.astype(object).where(key.notna(), None).to_numpy(dtype=object), index=air.index, dtype=object
    )
    pub = pd.Series(table.public.to_numpy(), index=table.key.to_numpy())
    out["public"] = key.map(pub).fillna(False).to_numpy(dtype=bool)
    return PublicResult(out, table)


class CarryoverResult(NamedTuple):
    """Result of :func:`flag_carryover` (unpack as ``air_flagged, table = flag_carryover(air)``).

    Attributes
    ----------
    air
        Copy of the input with ``clonotype_key``, ``shared_class`` and ``carryover`` columns.
    table
        One row per shared clonotype, see :func:`flag_carryover`.
    """

    air: pd.DataFrame
    table: pd.DataFrame


def _public_keys(public, air: pd.DataFrame, key: pd.Series) -> set:
    if public is None:
        return set()
    if isinstance(public, PublicResult):
        public = public.table
    if isinstance(public, pd.DataFrame):
        if not {"key", "public"} <= set(public.columns):
            raise ValueError("public table needs 'key' and 'public' columns (the table of flag_public)")
        return set(public.key[public.public.astype(bool)])
    if isinstance(public, str):
        if public not in air.columns:
            raise KeyError(f"flag_carryover: column {public!r} not in the table")
        flag = air[public].fillna(False).to_numpy(dtype=bool)
        return set(key[flag].dropna())
    if isinstance(public, Collection):
        return set(public)
    raise TypeError("public must be None, a flag_public result or table, a column name or a collection of keys")


def flag_carryover(
    air: pd.DataFrame,
    *,
    sample_key: str = S.LIBRARY,
    min_ratio: float = 100,
    public: PublicResult | pd.DataFrame | str | Collection | None = None,
    value: Literal["records", "reads", "spots"] = "records",
    convergent_loci: Sequence[str] = ("IGK", "IGL"),
    v_level: Literal["family", "gene"] | None = "family",
    locus_key: str = S.LOCUS,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
) -> CarryoverResult:
    """Flag cross-sample carry-over of clonotypes shared between samples.

    For every clonotype (see :func:`clonotype_key`) present in at least two samples, the sample holding most of it is
    its source. The copies in another sample are **carry-over** when the clonotype is not public and the source holds
    at least ``min_ratio`` times more of it. A non-public clonotype shared at comparable abundance is public when its
    chain is in ``convergent_loci`` (convergent light chains) and **unexplained** otherwise.

    Parameters
    ----------
    air
        Receptor table of several samples.
    sample_key
        Column naming the sample (library, section, mouse).
    min_ratio
        Abundance ratio source / copy at or above which a non-public copy is carry-over.
    public
        Public clonotypes, never called carry-over: a :func:`flag_public` result or its table, a boolean column of
        ``air``, or a collection of keys. ``None`` = no clonotype is public.
    value
        Abundance measure: ``"records"`` (rows; spot-molecule units in the archive), ``"reads"`` or ``"spots"``.
    convergent_loci
        Chains whose non-public clonotypes shared at comparable abundance count as convergent (public). Kappa and
        lambda by default; ``()`` disables.
    v_level, locus_key, v_key, j_key, junction_key
        See :func:`clonotype_key`.
    reads_key, spot_key
        Column names for ``value="reads"`` / ``"spots"``.

    Returns
    -------
    CarryoverResult
        ``air`` with ``clonotype_key``, ``shared_class`` (``"public"``, ``"carryover"``, ``"unexplained"``, missing
        for clonotypes in one sample) and ``carryover`` (True for the records of a carry-over copy, i.e. in a sample
        other than the source). ``table``: one row per shared clonotype with ``key``, ``locus``, ``junction_aa``,
        ``abundance_<sample>`` columns, ``n_samples``, ``source`` (most abundant sample; ties: first in sorted order),
        ``minor_over_major`` (smallest over largest abundance), ``public``, ``class`` and ``reason``
        (``"public"``, ``"convergent locus"``, ``"carry-over"``, ``"unexplained"``).

    Notes
    -----
    This is a rule, not a test; it has no null model. With more than two samples, the clonotype is carry-over when
    every non-source copy satisfies the ratio; otherwise copies that do are still flagged in ``carryover`` and the
    clonotype is classed by its comparable copies.

    References
    ----------
    Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x (index hopping between multiplexed libraries: a molecule
    is assigned to the sample holding most of its reads); Farouni et al. 2020, doi:10.1038/s41467-020-16522-z;
    Wang et al. 2021, doi:10.3389/fimmu.2021.778298 (inter-sample artefacts in repertoire sequencing).

    Examples
    --------
    >>> pub = flag_public(air, reference=atlas, n_key="n_additions")  # doctest: +SKIP
    >>> flagged, shared = flag_carryover(air, sample_key="library_id", public=pub)  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if sample_key not in air.columns:
        raise KeyError(f"flag_carryover: column {sample_key!r} not in the table")
    if not min_ratio >= 1:
        raise ValueError("min_ratio must be >= 1")
    if value not in ("records", "reads", "spots"):
        raise ValueError(f"value must be 'records', 'reads' or 'spots', got {value!r}")
    key = clonotype_key(air, v_level=v_level, locus_key=locus_key, v_key=v_key, j_key=j_key, junction_key=junction_key)
    pub_keys = _public_keys(public, air, key)
    ok = key.notna().to_numpy() & air[sample_key].notna().to_numpy()
    k = key[ok].to_numpy(dtype=object)
    smp = air.loc[ok, sample_key].astype("string").to_numpy(dtype=object)
    if value == "records":
        w = np.ones(int(ok.sum()))
        ab = pd.Series(w).groupby([k, smp]).sum()
    elif value == "reads":
        if reads_key not in air.columns:
            raise KeyError(f"flag_carryover: column {reads_key!r} not in the table")
        w = pd.to_numeric(air.loc[ok, reads_key], errors="coerce").fillna(0).to_numpy(dtype=float)
        ab = pd.Series(w).groupby([k, smp]).sum()
    else:
        if spot_key not in air.columns:
            raise KeyError(f"flag_carryover: column {spot_key!r} not in the table")
        sp = air.loc[ok, spot_key].astype("string").to_numpy(dtype=object)
        ab = pd.Series(sp).groupby([k, smp]).nunique()
    wide = ab.unstack(fill_value=0)
    wide = wide[sorted(wide.columns)]
    wide = wide[(wide > 0).sum(axis=1) >= 2]
    out = air.copy()
    out["clonotype_key"] = key.to_numpy()
    out["shared_class"] = pd.Series([None] * len(air), index=air.index, dtype=object)
    out["carryover"] = False
    cols = ["key", "locus", "junction_aa", "n_samples", "source", "minor_over_major", "public", "class", "reason"]
    if wide.empty:
        return CarryoverResult(out, pd.DataFrame(columns=cols))
    arr = wide.to_numpy(dtype=float)
    src = arr.argmax(axis=1)  # ties -> first sample in sorted order
    mx = arr[np.arange(len(arr)), src]
    present = arr > 0
    mn = np.where(present, arr, np.inf).min(axis=1)
    copy_is_carry = present & (mx[:, None] >= min_ratio * arr)
    copy_is_carry[np.arange(len(arr)), src] = False
    comparable = present & ~copy_is_carry
    comparable[np.arange(len(arr)), src] = False
    keys = wide.index.to_numpy(dtype=object)
    is_pub = np.array([kk in pub_keys for kk in keys], dtype=bool)
    loc = np.array([str(kk).split("|", 1)[0] for kk in keys], dtype=object)
    conv = np.isin(loc, list(convergent_loci)) & comparable.any(axis=1)
    cls = np.where(is_pub | conv, "public", np.where(comparable.any(axis=1), "unexplained", "carryover"))
    reason = np.where(
        is_pub, "public", np.where(conv, "convergent locus", np.where(cls == "carryover", "carry-over", "unexplained"))
    )
    table = pd.DataFrame(
        {
            "key": keys,
            "locus": loc,
            "junction_aa": [str(kk).split("|")[2] for kk in keys],
        }
    )
    for i, c in enumerate(wide.columns):
        table[f"abundance_{c}"] = arr[:, i]
    table["n_samples"] = present.sum(axis=1)
    table["source"] = np.asarray(wide.columns, dtype=object)[src]
    table["minor_over_major"] = mn / mx
    table["public"] = is_pub
    table["class"] = cls
    table["reason"] = reason
    # per record flags
    cls_map = pd.Series(cls, index=keys)
    sc = key.map(cls_map)
    out["shared_class"] = pd.Series(sc.where(sc.notna(), None).to_numpy(dtype=object), index=air.index, dtype=object)
    carry_pairs = {
        (keys[r], wide.columns[c]) for r, c in zip(*np.nonzero(copy_is_carry & ~is_pub[:, None]), strict=True)
    }
    if carry_pairs:
        rec_k = key.to_numpy(dtype=object)
        rec_s = air[sample_key].astype("string").to_numpy(dtype=object)
        out["carryover"] = np.fromiter(
            ((a, b) in carry_pairs for a, b in zip(rec_k, rec_s, strict=True)), dtype=bool, count=len(air)
        )
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "flag_carryover": {"sample_key": sample_key, "min_ratio": min_ratio, "value": value},
    }
    return CarryoverResult(out, table[[*cols[:3], *[f"abundance_{c}" for c in wide.columns], *cols[3:]]])
