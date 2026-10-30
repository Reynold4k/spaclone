"""Isotype of BCR heavy chains (class, subclass, position on the constant locus) and switched calls per clone or locus.

The heavy-chain constant genes lie in a fixed 5' to 3' order on the IgH locus. Class-switch recombination (CSR)
deletes the DNA between the switch region upstream of Cμ and the switch region of a downstream gene, so a B cell can
only move downstream, never back:

* mouse: Cμ, Cδ, Cγ3, Cγ1, Cγ2b, Cγ2a / Cγ2c (allelic genes at one position: BALB/c IGHG2A, C57BL/6 IGHG2C), Cε, Cα;
* human: Cμ, Cδ, Cγ3, Cγ1, (ψε), Cα1, (ψγ), Cγ2, Cγ4, Cε, Cα2 (pseudogenes in brackets are never expressed).

IgM and IgD are the unswitched classes (co-expressed by alternative splicing in naive B cells); IgG, IgA and IgE are
switched.

References
----------
Stavnezer, Guikema & Schrader 2008, doi:10.1146/annurev.immunol.26.021607.090248 (CSR mechanism and locus order);
Horns et al. 2016, doi:10.7554/eLife.16578 (isotypes within human B-cell lineages).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = [
    "CONSTANT_ORDER",
    "ISOTYPE_CLASSES",
    "SWITCHED_CLASSES",
    "UNSWITCHED_CLASSES",
    "call_isotype",
    "isotype",
    "isotype_order",
]

#: Isotype classes, in the order used to break ties of the main isotype (as the 2026-10 archive: IgM, IgD, IgG, IgA).
ISOTYPE_CLASSES: tuple[str, ...] = ("IgM", "IgD", "IgG", "IgA", "IgE")
#: Classes produced by class-switch recombination.
SWITCHED_CLASSES: frozenset[str] = frozenset({"IgG", "IgA", "IgE"})
#: Classes of B cells that have not switched (IgD is co-expressed with IgM by alternative splicing).
UNSWITCHED_CLASSES: frozenset[str] = frozenset({"IgM", "IgD"})

#: Position (0 = most 5') of every expressed constant gene on the IgH locus, by species; allelic genes share one.
CONSTANT_ORDER: dict[str, dict[str, int]] = {
    "mouse": {"IgM": 0, "IgD": 1, "IgG3": 2, "IgG1": 3, "IgG2b": 4, "IgG2a": 5, "IgG2c": 5, "IgE": 6, "IgA": 7},
    "human": {"IgM": 0, "IgD": 1, "IgG3": 2, "IgG1": 3, "IgA1": 4, "IgG2": 5, "IgG4": 6, "IgE": 7, "IgA2": 8},
}

# class-level order: position of the class's most 5' gene (exact for mouse; for human IgG / IgA / IgE interleave)
_CLASS_ORDER = {
    sp: {c: min(p for s, p in order.items() if s.startswith(c)) for c in ISOTYPE_CLASSES}
    for sp, order in CONSTANT_ORDER.items()
}

_GENE = re.compile(r"^IG(?:H)?([MDGAE])(P?)(\d?)([A-C]?)$")
_NON_HEAVY = re.compile(r"^(IG[KL]|TR[ABGD])")


def isotype_order(species: Literal["mouse", "human"] = "mouse", *, level: Literal["subclass", "class"] = "subclass"):
    """Position of every isotype on the IgH constant locus (0 = most 5').

    Parameters
    ----------
    species
        ``"mouse"`` or ``"human"``.
    level
        ``"subclass"`` (one entry per expressed constant gene) or ``"class"`` (the position of the class's most 5'
        gene; for human, IgG, IgA and IgE genes interleave, so class-level order is only a convention there).

    Returns
    -------
    dict
        ``{isotype: position}``.
    """
    _check_species(species)
    if level == "subclass":
        return dict(CONSTANT_ORDER[species])
    if level == "class":
        return dict(_CLASS_ORDER[species])
    raise ValueError(f"level must be 'subclass' or 'class', got {level!r}")


def _check_species(species: str | None) -> None:
    if species is not None and species not in CONSTANT_ORDER:
        raise ValueError(f"species must be one of {sorted(CONSTANT_ORDER)} (or None), got {species!r}")


def _parse_gene(gene: str, species: str | None) -> tuple[str | None, str | None]:
    """Class and subclass of one constant gene name (allele already removed); (None, None) if not a heavy gene."""
    m = _GENE.match(gene)
    if not m:
        return None, None
    letter, pseudo, num, sub_letter = m.groups()
    if pseudo:  # IGHGP, IGHEP1: pseudogenes are never expressed
        return None, None
    cls = "Ig" + letter
    sub = cls + num + sub_letter.lower()
    if species is not None and sub not in CONSTANT_ORDER[species]:
        sub = None  # class-only call (e.g. mouse "IGHG") or a subclass that this species does not have
    return cls, sub


def _parse_call(call: Any, species: str | None) -> tuple[str | None, str | None, bool]:
    """(class, subclass, heavy) of one c_call; several comma-separated calls must agree."""
    if not isinstance(call, str):
        return None, None, True
    text = call.strip().upper()
    if not text or text in ("*", "NA", "NAN", "NONE"):
        return None, None, True
    genes = [g.strip().split("*")[0] for g in re.split(r"[,;|]", text) if g.strip()]
    genes = [g for g in genes if g]
    if not genes:
        return None, None, True
    if all(_NON_HEAVY.match(g) for g in genes):
        return None, None, False
    parsed = [_parse_gene(g, species) for g in genes]
    classes = {c for c, _ in parsed}
    if len(classes) != 1 or None in classes:
        return None, None, True
    subs = {s for _, s in parsed}
    return classes.pop(), (subs.pop() if len(subs) == 1 else None), True


def isotype(
    air: pd.DataFrame | pd.Series | Sequence[str],
    *,
    species: Literal["mouse", "human"] | None = "mouse",
    c_key: str = S.C_CALL,
    locus_key: str = S.LOCUS,
    key_added: str = "isotype",
    switched_key: str = "switched",
) -> pd.DataFrame:
    """Isotype class, subclass, position on the constant locus and switched flag of every heavy-chain record.

    Parameters
    ----------
    air
        Receptor table with constant-gene calls (``c_call``), or the calls themselves (a Series or list).
    species
        ``"mouse"`` or ``"human"``: subclasses and locus positions follow that species' IgH locus (see
        :data:`CONSTANT_ORDER`). ``None`` keeps any subclass name and gives no positions.
    c_key
        Column with the constant-gene call, e.g. ``"IGHG2B*01"``, ``"IGHA"``, ``"IgG1"``. Alleles (``*01``) are
        dropped; several comma-separated calls are accepted when they name one class (the subclass is kept only when
        they also agree on it). Missing values, ``""`` and TRUST4's ``"*"`` (the contig does not reach the constant
        region) give ``"unknown"``.
    locus_key
        Chain column; records of other chains than IGH get missing values. Without it, light-chain and TCR constant
        calls (``IGKC``, ``IGLC2``, ``TRBC1``, ...) are recognised from the call.
    key_added
        Name of the class column; the subclass and position columns are ``{key_added}_subclass`` and
        ``{key_added}_position``.
    switched_key
        Name of the switched column.

    Returns
    -------
    pandas.DataFrame
        A copy of ``air`` (or a new table for Series input) with:

        * ``isotype``: ``"IgM"``, ``"IgD"``, ``"IgG"``, ``"IgA"``, ``"IgE"``, ``"unknown"`` (heavy chain without a
          usable constant call) or missing (not a heavy chain);
        * ``isotype_subclass``: e.g. ``"IgG2b"``, ``"IgA1"``; missing when the call names only the class;
        * ``isotype_position``: 0-based position of the subclass on the IgH locus (5' to 3'), missing without a
          subclass or a species;
        * ``switched``: True for IgG / IgA / IgE, False for IgM / IgD, missing otherwise.

    Notes
    -----
    Pseudogenes (IGHGP, IGHEP1) give ``"unknown"``. A mouse ``"IGHG2A"`` call is kept as IgG2a: IMGT carries both
    allelic genes, and their position on the locus is the same.

    Examples
    --------
    >>> isotype(pd.Series(["IGHG2B*01", "IGHM", "*", "IGKC"]))["isotype"].tolist()  # doctest: +SKIP
    ['IgG', 'IgM', 'unknown', <NA>]
    """
    _check_species(species)
    if isinstance(air, pd.DataFrame):
        if c_key not in air.columns:
            raise KeyError(f"isotype: column {c_key!r} not in the table; pass c_key")
        calls = air[c_key]
        out = air.copy()
        locus = air[locus_key] if locus_key in air.columns else None
    else:
        calls = air if isinstance(air, pd.Series) else pd.Series(list(air), dtype=object)
        out = pd.DataFrame(index=calls.index)
        locus = None
    uniq = pd.unique(calls.astype(object).to_numpy())
    look = {u if isinstance(u, str) else None: _parse_call(u, species) for u in uniq}
    keys = [c if isinstance(c, str) else None for c in calls.astype(object).to_numpy()]
    parsed = [look[k] for k in keys]
    cls = np.array([p[0] if p[0] is not None else ("unknown" if p[2] else None) for p in parsed], dtype=object)
    sub = np.array([p[1] for p in parsed], dtype=object)
    if locus is not None:
        heavy = (locus.astype("string") == "IGH").fillna(False).to_numpy(dtype=bool)
        cls[~heavy] = None
        sub[~heavy] = None
    order = CONSTANT_ORDER.get(species, {}) if species is not None else {}
    pos = pd.array([order.get(s) if s is not None else None for s in sub], dtype="Int64")
    sw = pd.array(
        [None if c is None or c == "unknown" else (c in SWITCHED_CLASSES) for c in cls],
        dtype="boolean",
    )
    out[key_added] = pd.array(cls, dtype="string")
    out[f"{key_added}_subclass"] = pd.array(sub, dtype="string")
    out[f"{key_added}_position"] = pos
    out[switched_key] = sw
    return out


def _group_column(air: pd.DataFrame, by: str, clone_key: str, locus_key: str) -> str:
    col = {"clone": clone_key, "locus": locus_key}.get(by, by)
    if col not in air.columns:
        hint = " (add the merge_loci membership: air['locus_id'] = membership)" if by == "locus" else ""
        raise KeyError(f"call_isotype: grouping column {col!r} not in the table{hint}")
    return col


def call_isotype(
    air: pd.DataFrame,
    *,
    by: str = "clone",
    rule: Literal["plurality", "share"] | None = None,
    tie: Literal["drop", "switched", "unswitched"] = "drop",
    min_share: float = 0.5,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS_ID,
    isotype_key: str = "isotype",
    c_key: str = S.C_CALL,
) -> pd.DataFrame:
    """Isotype and switched call of every clone, locus or other group of records.

    Parameters
    ----------
    air
        Records (contigs, molecules) with an isotype class column (from :func:`isotype`; computed from ``c_key``
        when absent) and the grouping column.
    by
        ``"clone"`` (``clone_key``), ``"locus"`` (``locus_key``, e.g. the membership returned by
        :func:`spaclone.pp.merge_loci`) or the name of any other column (spot, cell).
    rule
        ``"plurality"``: the most frequent isotype class decides; the group is switched when every class at the top
        count is switched, unswitched when every one is unswitched, and tied when a switched and an unswitched class
        share the top count (the archive's clone rule, ``T_common.clone_class``). ``"share"``: switched when at least
        ``min_share`` of the isotyped records are IgG / IgA / IgE (the archive's locus rule, T1 / T3: >= 50%).
        ``None`` (default): ``"plurality"`` for ``by="clone"``, ``"share"`` otherwise.
    tie
        ``rule="plurality"`` only: ``"drop"`` leaves tied groups out of both classes (``switched`` missing),
        ``"switched"`` / ``"unswitched"`` assign them.
    min_share
        Threshold of ``rule="share"``.
    clone_key, locus_key, isotype_key, c_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per group (index named after the grouping column, sorted): ``n_records``, one count column per
        class (``IgM``, ``IgD``, ``IgG``, ``IgA``, ``IgE``) and ``unknown`` (heavy records without a class),
        ``n_isotyped``, ``switched_share`` ((IgG + IgA + IgE) / isotyped), ``isotype`` (main class: most frequent,
        ties broken in the order IgM, IgD, IgG, IgA, IgE; ``"unknown"`` without isotyped records), ``switched``
        (boolean, missing for dropped ties and groups without isotype) and ``call`` (``"switched"``,
        ``"unswitched"``, ``"tied"`` or ``"unknown"``).

    Notes
    -----
    Records of other chains (isotype missing) count in ``n_records`` only. IgE counts as switched; the archive's
    StereoIR tables had no IgE, so its rules (IgG / IgA only) give the same calls there.

    Examples
    --------
    >>> calls = call_isotype(air, by="clone")  # doctest: +SKIP
    >>> switched_clones = calls.index[calls.switched.fillna(False)]  # doctest: +SKIP
    """
    if rule is None:
        rule = "plurality" if by == "clone" else "share"
    if rule not in ("plurality", "share"):
        raise ValueError(f"rule must be 'plurality' or 'share', got {rule!r}")
    if tie not in ("drop", "switched", "unswitched"):
        raise ValueError(f"tie must be 'drop', 'switched' or 'unswitched', got {tie!r}")
    if not 0 < min_share <= 1:
        raise ValueError(f"min_share must lie in (0, 1], got {min_share}")
    col = _group_column(air, by, clone_key, locus_key)
    if air[col].isna().any():
        raise ValueError(f"call_isotype: grouping column {col!r} has missing values")
    iso = air[isotype_key] if isotype_key in air.columns else isotype(air, c_key=c_key, species=None)["isotype"]
    iso = iso.astype("string")
    classes = list(ISOTYPE_CLASSES)
    n_rec = air.groupby(col, sort=True).size()
    known = iso.notna().to_numpy()
    tab = (
        pd.DataFrame({"g": air[col].to_numpy()[known], "iso": iso.to_numpy(dtype=object)[known]})
        .groupby(["g", "iso"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    tab = tab.reindex(index=n_rec.index, columns=[*classes, "unknown"], fill_value=0).fillna(0).astype(np.int64)
    out = pd.DataFrame({"n_records": n_rec.astype(np.int64)})
    for c in [*classes, "unknown"]:
        out[c] = tab[c].to_numpy()
    counts = out[classes].to_numpy()
    n_iso = counts.sum(1)
    sw_cols = [i for i, c in enumerate(classes) if c in SWITCHED_CLASSES]
    un_cols = [i for i, c in enumerate(classes) if c in UNSWITCHED_CLASSES]
    n_sw = counts[:, sw_cols].sum(1)
    out["n_isotyped"] = n_iso
    with np.errstate(invalid="ignore", divide="ignore"):
        out["switched_share"] = np.where(n_iso > 0, n_sw / np.maximum(n_iso, 1), np.nan)
    has = n_iso > 0
    main = np.array(classes, dtype=object)[counts.argmax(1)]  # argmax = first maximum -> class order breaks ties
    out["isotype"] = pd.array(np.where(has, main, "unknown"), dtype="string")
    switched = np.full(len(out), None, dtype=object)
    call = np.where(has, "", "unknown").astype(object)
    if rule == "plurality":
        top = counts == counts.max(1, keepdims=True)
        s = top[:, sw_cols].any(1) & has
        u = top[:, un_cols].any(1) & has
        tied = s & u
        call[has & s & ~u] = "switched"
        call[has & u & ~s] = "unswitched"
        call[tied] = "tied"
        switched[has & s & ~u] = True
        switched[has & u & ~s] = False
        if tie != "drop":
            switched[tied] = tie == "switched"
    else:
        share = out["switched_share"].to_numpy()
        is_sw = has & (share >= min_share)
        call[has] = np.where(is_sw[has], "switched", "unswitched")
        switched[has] = is_sw[has]
    out["switched"] = pd.array(switched, dtype="boolean")
    out["call"] = pd.array(call, dtype="string")
    out.index.name = col
    out.attrs["spaclone"] = {"call_isotype": {"by": col, "rule": rule, "tie": tie, "min_share": min_share}}
    return out
