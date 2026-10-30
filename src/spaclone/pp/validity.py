"""Sequence validity: IMGT junction rules, the productive flag, read support and tissue position.

The rules are the project standard (the archive's sequence-validity standard, implemented there as ``valid_cdr3``):

1. the junction (IMGT CDR3 plus the two anchors) starts with C (2nd-CYS 104);
2. it ends with W or F (J-TRP / J-PHE 118) on every chain: mouse TRAJ33, TRAJ38, TRAJ44 and some IGKJ1 junctions end
   in W, so the rule does not depend on the chain;
3. it holds only the 20 standard amino acids (no stop ``*``, frame-shift ``_`` or unknown ``X``);
4. it is at least 5 amino acids long (C + >= 3 + W/F);
5. where the data carry a productive / functional flag, the flag is true;
6. optionally, a read floor (short-read technologies: >= 2 reads, as in pRESTO's consensus step);
7. optionally, records off the tissue are removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = ["STANDARD_AA", "FilterResult", "filter_valid", "is_valid_junction"]

#: The 20 standard amino acids (one-letter codes).
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_TRUE = {"T", "TRUE", "1", "Y", "YES"}


def _pattern(min_length: int, first: str, last: str) -> str:
    if min_length < 2:
        raise ValueError("min_length must be >= 2 (the two anchor residues)")
    return f"[{first}][{STANDARD_AA}]{{{min_length - 2},}}[{last}]"


def is_valid_junction(
    junction_aa: Any, *, min_length: int = 5, first: str = "C", last: str = "WF"
) -> np.ndarray | bool:
    """Apply the IMGT junction rules to amino-acid junctions.

    Parameters
    ----------
    junction_aa
        One junction (str) or an array-like of junctions (list, ndarray, Series). Missing values are invalid.
    min_length
        Minimum length in amino acids, anchors included.
    first, last
        Allowed first and last residues (IMGT: C104 and W/F118).

    Returns
    -------
    numpy.ndarray or bool
        Boolean array (one value per input, order kept), or a single bool for a single string.

    Examples
    --------
    >>> is_valid_junction(["CARNSHYYGSRYPFAYW", "CQQYWSTPLTF", "CAVRDSNYQLIW", "CARDY*YW", "ARNSHYYG"]).tolist()
    [True, True, True, False, False]
    """
    pat = _pattern(min_length, first, last)
    if isinstance(junction_aa, str):
        return bool(pd.Series([junction_aa], dtype="string").str.fullmatch(pat).fillna(False).iloc[0])
    values = junction_aa.to_numpy() if isinstance(junction_aa, pd.Series | pd.Index) else junction_aa
    s = pd.Series(np.asarray(values, dtype=object)).astype("string")  # missing -> <NA> -> invalid
    return s.str.fullmatch(pat).fillna(False).to_numpy(dtype=bool)


def _as_flag(values: pd.Series) -> np.ndarray:
    """Interpret a productive / functional / in-tissue column as booleans (missing = False)."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        v = values.to_numpy(dtype=float)
        return np.isfinite(v) & (v != 0)
    s = values.astype("string").str.strip().str.upper()
    return s.isin(_TRUE).fillna(False).to_numpy(dtype=bool)


class FilterResult(NamedTuple):
    """Result of :func:`filter_valid` (unpack as ``air_ok, report = filter_valid(air)``).

    Attributes
    ----------
    air
        The records that pass every applied rule (index kept).
    report
        One row per rule (and stratum, with ``by``): see :func:`filter_valid`.
    """

    air: pd.DataFrame
    report: pd.DataFrame


def filter_valid(
    air: pd.DataFrame,
    *,
    require_productive: bool = True,
    min_reads: float | None = None,
    tissue_key: str | None = None,
    min_length: int = 5,
    by: str | Sequence[str] | None = None,
    junction_key: str = S.JUNCTION_AA,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
) -> FilterResult:
    """Keep the records whose sequence is valid, and report what each rule removed.

    Rules are applied in this order: valid junction (:func:`is_valid_junction`), productive flag (if
    ``require_productive`` and the column exists), read floor (if ``min_reads``), on tissue (if ``tissue_key``).

    Parameters
    ----------
    air
        Receptor table.
    require_productive
        Require a true productive / functional flag (``T``, ``TRUE``, ``True``, 1, ``Y``). Skipped, and reported as
        not applied, when the table has no ``productive_key`` column.
    min_reads
        Read floor: keep records with ``reads_key >= min_reads`` (records with unknown reads fail). ``None`` = no
        floor (long-read data such as AIR-SPACE, or data without read counts).
    tissue_key
        Boolean column, true on tissue; records off the tissue are removed.
    min_length
        Minimum junction length in amino acids.
    by
        Column(s) for a stratified report (e.g. ``"locus"``); the filter itself is the same for every stratum.
    junction_key, productive_key, reads_key, spot_key
        Column names.

    Returns
    -------
    FilterResult
        ``air``: the passing records. ``report``: columns ``step`` (0 = input), ``rule``, ``applied``, ``failing``
        (input records failing this rule, regardless of order), ``removed`` (records removed at this step),
        ``records`` and ``spots`` (records and distinct spots left after the step), plus the ``by`` columns.

    Examples
    --------
    >>> ok, report = filter_valid(air, min_reads=2)  # doctest: +SKIP
    >>> report[["rule", "records", "spots"]]  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if junction_key not in air.columns:
        raise KeyError(f"filter_valid: column {junction_key!r} not in the table")
    if min_reads is not None and reads_key not in air.columns:
        raise KeyError(f"filter_valid: min_reads given but column {reads_key!r} not in the table")
    if tissue_key is not None and tissue_key not in air.columns:
        raise KeyError(f"filter_valid: column {tissue_key!r} not in the table")
    by_cols = [] if by is None else ([by] if isinstance(by, str) else list(by))
    missing = [c for c in by_cols if c not in air.columns]
    if missing:
        raise KeyError(f"filter_valid: column(s) {missing} not in the table")

    rules: list[tuple[str, bool, np.ndarray]] = [
        (
            f"valid junction (C...W/F, 20 aa, >= {min_length} aa)",
            True,
            is_valid_junction(air[junction_key], min_length=min_length),
        ),
    ]
    if require_productive:
        if productive_key in air.columns:
            rules.append(("productive", True, _as_flag(air[productive_key])))
        else:
            rules.append((f"productive (no {productive_key!r} column)", False, np.ones(len(air), bool)))
    if min_reads is not None:
        r = air[reads_key].to_numpy(dtype=float)
        rules.append((f"reads >= {min_reads:g}", True, np.isfinite(r) & (r >= min_reads)))
    if tissue_key is not None:
        rules.append(("on tissue", True, _as_flag(air[tissue_key])))

    keep = np.ones(len(air), bool)
    steps = [("input", True, keep.copy(), np.zeros(len(air), bool), np.zeros(len(air), bool))]
    for name, applied, ok in rules:
        removed = keep & ~ok
        keep = keep & ok
        steps.append((name, applied, keep.copy(), ~ok, removed))

    spots = air[spot_key] if spot_key in air.columns else pd.Series(np.arange(len(air)), index=air.index)
    strata = [((), np.ones(len(air), bool))]
    if by_cols:
        strata = [
            ((k if isinstance(k, tuple) else (k,)), np.isin(np.arange(len(air)), pos))
            for k, pos in air.groupby(by_cols, sort=True, dropna=False, observed=True).indices.items()
        ]
    rows = []
    for key, sel in strata:
        for i, (name, applied, left, failing, removed) in enumerate(steps):
            row = dict(zip(by_cols, key, strict=True))
            row.update(
                {
                    "step": i,
                    "rule": name,
                    "applied": applied,
                    "failing": int((failing & sel).sum()),
                    "removed": int((removed & sel).sum()),
                    "records": int((left & sel).sum()),
                    "spots": int(spots[left & sel].nunique()),
                }
            )
            rows.append(row)
    report = pd.DataFrame(rows)[[*by_cols, "step", "rule", "applied", "failing", "removed", "records", "spots"]]
    out = air.loc[keep].copy()
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "filter_valid": {
            "require_productive": require_productive,
            "min_reads": min_reads,
            "tissue_key": tissue_key,
            "min_length": min_length,
        },
    }
    return FilterResult(out, report)
