"""Clones per spot and barcode sinks.

A Stereo-seq DNB is 0.5 µm wide and captures a handful of molecules, so it can hold the receptor sequences of at most
one or two cells. A spot that carries tens to hundreds of distinct clones is a *barcode sink*: a spot barcode that
many chimeric molecules inherited during library PCR, never a cell. Sinks must be flagged before any spot-level test:
one sink whose GEX happens to hold an immunoglobulin transcript turns hundreds of swapped molecules into "hits".

References
----------
Kebschull & Zador 2015, doi:10.1093/nar/gkv717 (template switching during PCR joins sequences to foreign barcodes).
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = ["BarcodeSinks", "barcode_sinks", "spot_multiplicity"]

#: Column added by :func:`spot_multiplicity`: distinct clones (any locus) on the record's spot.
MULT_ALL = "mult_all"
#: Column added by :func:`spot_multiplicity`: distinct clones of the record's locus on the record's spot.
MULT_LOCUS = "mult_locus"


def _codes(
    air: pd.DataFrame, spot_key: str, locus_key: str, clone_key: str
) -> tuple[np.ndarray, pd.Index, np.ndarray, pd.Index, np.ndarray, int]:
    """Integer codes of spots (order of appearance), loci (sorted) and clones; raises on missing clone ids."""
    if air[clone_key].isna().any():
        raise ValueError(
            f"column {clone_key!r} has missing clone ids; define clones first (spaclone.pp.define_clones) or drop "
            "the unassigned records"
        )
    s, spots = pd.factorize(air[spot_key], sort=False, use_na_sentinel=False)
    lo, loci = pd.factorize(air[locus_key].astype(object), sort=True, use_na_sentinel=False)
    c, clones = pd.factorize(air[clone_key], sort=False, use_na_sentinel=False)
    return s.astype(np.int64), spots, lo.astype(np.int64), loci, c.astype(np.int64), len(clones)


def _distinct(s: np.ndarray, lo: np.ndarray, c: np.ndarray, n_loci: int, n_clones: int) -> tuple[np.ndarray, ...]:
    """Spot and locus code of every distinct (spot, locus, clone) triple; a clone id counts once per locus."""
    n_spots = int(s.max()) + 1 if len(s) else 0
    if n_spots * max(n_loci, 1) * max(n_clones, 1) < 2**62:
        key = pd.unique((s * n_loci + lo) * n_clones + c)
        return key // (n_loci * n_clones), (key // n_clones) % n_loci
    t = pd.DataFrame({"s": s, "l": lo, "c": c}).drop_duplicates(ignore_index=True)  # pragma: no cover - huge tables
    return t["s"].to_numpy(), t["l"].to_numpy()  # pragma: no cover


def _first_rows(s: np.ndarray, n_spots: int) -> np.ndarray:
    first = np.full(n_spots, -1, dtype=np.int64)
    first[s[::-1]] = np.arange(len(s))[::-1]
    return first


def spot_multiplicity(
    air: pd.DataFrame,
    *,
    level: Literal["record", "spot"] = "record",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> pd.DataFrame:
    """Count the distinct clones on each spot, over all loci and per locus.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, (spot, clone) unit, read or call).
    level
        ``"record"`` returns a copy of ``air`` with two added columns, ``mult_all`` (distinct clones of any locus on
        the record's spot) and ``mult_locus`` (distinct clones of the record's locus on that spot). ``"spot"`` returns
        one row per spot: ``spot_id``, ``x_um``, ``y_um`` (first record), ``n_records``, ``n_clones`` and
        ``n_clones_<LOCUS>`` for every locus present.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    Clones are counted as distinct ``(locus, clone_id)`` pairs, so several molecules or reads of one clone on a spot
    count once, and a clone id reused by two loci counts twice. On a table of (DNB, clone) units this is the
    ``mult_all`` / ``mult_chain`` of the archived Stereo-XCR-seq analysis.
    """
    if level not in ("record", "spot"):
        raise ValueError(f"level must be 'record' or 'spot', got {level!r}")
    S.validate_air(air, required=(spot_key, locus_key, clone_key), numeric=())
    s, spots, lo, loci, c, n_clones = _codes(air, spot_key, locus_key, clone_key)
    n_spots, n_loci = len(spots), len(loci)
    ts, tl = _distinct(s, lo, c, n_loci, n_clones)
    per_spot = np.bincount(ts, minlength=n_spots)
    per_spot_locus = np.bincount(ts * n_loci + tl, minlength=n_spots * n_loci)
    if level == "record":
        out = air.copy()
        out[MULT_ALL] = per_spot[s]
        out[MULT_LOCUS] = per_spot_locus[s * n_loci + lo]
        return out
    first = _first_rows(s, n_spots)
    tab = pd.DataFrame({spot_key: np.asarray(spots, dtype=object)})
    for k in (x_key, y_key):
        if k in air.columns:
            tab[k] = air[k].to_numpy()[first]
    tab["n_records"] = np.bincount(s, minlength=n_spots)
    tab["n_clones"] = per_spot
    psl = per_spot_locus.reshape(n_spots, n_loci)
    for j, name in enumerate(loci):
        tab[f"n_clones_{name}"] = psl[:, j]
    return tab


class BarcodeSinks(NamedTuple):
    """Result of :func:`barcode_sinks`.

    Attributes
    ----------
    table
        One row per sink spot, most clones first: ``spot_id``, ``x_um``, ``y_um``, ``n_clones`` (distinct clones, all
        loci), ``n_records``, ``n_reads`` (when a read column is present), ``dominant_locus`` (locus holding most of
        the spot's clones), ``dominant_share``, ``n_loci``.
    mask
        Boolean array aligned with the rows of the receptor table: True for records on a sink.
    min_clones
        The threshold used.
    """

    table: pd.DataFrame
    mask: np.ndarray
    min_clones: int

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers (sinks, the largest sink, records on sinks and their share)."""
        return {
            "n_sinks": len(self.table),
            "max_clones": int(self.table["n_clones"].max()) if len(self.table) else 0,
            "n_records_on_sinks": int(self.mask.sum()),
            "share_records_on_sinks": float(self.mask.mean()) if len(self.mask) else float("nan"),
            "min_clones": self.min_clones,
        }


def barcode_sinks(
    air: pd.DataFrame,
    *,
    min_clones: int = 20,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
    reads_key: str | None = S.READS,
) -> BarcodeSinks:
    """Flag barcode sinks: spots that carry at least ``min_clones`` distinct clones.

    Parameters
    ----------
    air
        Receptor table.
    min_clones
        A spot with this many distinct clones (all loci together) or more is a sink. The default 20 is the threshold
        of the archived Stereo-XCR-seq analyses: a 0.5 µm DNB cannot hold 20 cells.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.
    reads_key
        Column of reads per record, summed per sink when present (``None`` to skip).

    Returns
    -------
    BarcodeSinks
        ``table`` (one row per sink), ``mask`` (records on sinks; drop them with ``air[~sinks.mask]``) and a
        ``summary`` property.

    Notes
    -----
    Sinks are flagged, not tested: the threshold is physical (spot size), not statistical. Choose ``min_clones`` for
    other spot sizes accordingly (a 55 µm Visium spot holds tens of cells). Clones are distinct ``(locus, clone_id)``
    pairs as in :func:`spot_multiplicity`.

    References
    ----------
    Kebschull & Zador 2015, doi:10.1093/nar/gkv717.
    """
    if min_clones < 1:
        raise ValueError("min_clones must be >= 1")
    S.validate_air(air, required=(spot_key, locus_key, clone_key), numeric=())
    s, spots, lo, loci, c, n_clones_total = _codes(air, spot_key, locus_key, clone_key)
    n_spots, n_loci = len(spots), len(loci)
    ts, tl = _distinct(s, lo, c, n_loci, n_clones_total)
    n_clones = np.bincount(ts, minlength=n_spots)
    is_sink = n_clones >= min_clones
    mask = is_sink[s] if len(s) else np.zeros(0, dtype=bool)
    sink_idx = np.flatnonzero(is_sink)
    cols = [spot_key, x_key, y_key, "n_clones", "n_records", "dominant_locus", "dominant_share", "n_loci"]
    if len(sink_idx) == 0:
        return BarcodeSinks(table=pd.DataFrame(columns=cols), mask=mask, min_clones=int(min_clones))
    by_locus = np.bincount(ts * n_loci + tl, minlength=n_spots * n_loci).reshape(n_spots, n_loci)[sink_idx]
    rows = _first_rows(s, n_spots)[sink_idx]
    tab = pd.DataFrame({spot_key: np.asarray(spots, dtype=object)[sink_idx]})
    for k in (x_key, y_key):
        if k in air.columns:
            tab[k] = air[k].to_numpy()[rows]
    tab["n_clones"] = n_clones[sink_idx]
    tab["n_records"] = np.bincount(s, minlength=n_spots)[sink_idx]
    if reads_key is not None and reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key]):
        reads = np.nan_to_num(air[reads_key].to_numpy(dtype=float))
        tab["n_reads"] = np.bincount(s, weights=reads, minlength=n_spots)[sink_idx]
    top = by_locus.argmax(axis=1)
    tab["dominant_locus"] = np.asarray(loci, dtype=object)[top]
    tab["dominant_share"] = by_locus[np.arange(len(sink_idx)), top] / tab["n_clones"].to_numpy()
    tab["n_loci"] = (by_locus > 0).sum(axis=1)
    tab = tab.sort_values(["n_clones", spot_key], ascending=[False, True], kind="stable", ignore_index=True)
    return BarcodeSinks(table=tab, mask=mask, min_clones=int(min_clones))
