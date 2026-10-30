"""Quality-control metrics of a receptor table (scanpy-style), read-support curves and their knee.

:func:`qc_metrics` summarises a receptor ("air") table per library and chain and per spot: records, reads, reads per
record, the shares of valid junctions and productive records, chain composition, clones per spot and spots per clone,
the share off the tissue and the share of spots that are barcode sinks. :func:`read_support_curve` counts the records
(and spots) that a read floor keeps, and :func:`knee_point` finds the knee of that curve with the barcode-rank rule of
DropletUtils: the floor where the deep, well-supported records end and the drop to the low-read cloud begins.

These are descriptive summaries, not tests: they carry no null model. Whether a read floor separates genuine from
misplaced molecules is a test of location trust (``spaclone.qc.read_floor``); the knee only describes the read
distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from spaclone import _schema as S

__all__ = ["Knee", "QCMetrics", "ReadSupport", "knee_point", "qc_metrics", "read_support_curve"]

_TRUE = {"T", "TRUE", "1", "Y", "YES"}
_DEFAULT_TISSUE = "in_tissue"


# ----------------------------------------------------------------------------------------------- shared helpers
def _as_flag(values: pd.Series) -> np.ndarray:
    """Booleans from a flag column (bool, numeric != 0, or T / TRUE / 1 / Y / YES strings); missing = False."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        v = values.to_numpy(dtype=float)
        return np.isfinite(v) & (v != 0)
    s = values.astype("string").str.strip().str.upper()
    return s.isin(_TRUE).fillna(False).to_numpy(dtype=bool)


def _cols(by: str | Sequence[str] | None) -> list[str]:
    if by is None:
        return []
    return [by] if isinstance(by, str) else list(by)


def _resolve_by(air: pd.DataFrame, by: str | Sequence[str] | None, defaults: Sequence[str], fn: str) -> list[str]:
    """``None`` = the default columns that are present; otherwise exactly the given columns (KeyError if absent)."""
    if by is None:
        return [c for c in defaults if c in air.columns]
    cols = _cols(by)
    missing = [c for c in cols if c not in air.columns]
    if missing:
        raise KeyError(f"{fn}: column(s) {missing} not in the table")
    return cols


def _resolve_tissue(air: pd.DataFrame, tissue_key: str | None, fn: str) -> str | None:
    """The default ``"in_tissue"`` is used only when present; any other name must exist."""
    if tissue_key is None:
        return None
    if tissue_key in air.columns:
        return tissue_key
    if tissue_key == _DEFAULT_TISSUE:
        return None
    raise KeyError(f"{fn}: column {tissue_key!r} not in the table")


def _group_codes(air: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Integer code per row for the groups defined by ``cols`` (sorted, NA kept) and the table of group keys."""
    if not cols:
        return np.zeros(len(air), np.int64), pd.DataFrame(index=pd.RangeIndex(1 if len(air) else 0))
    g = air.groupby(list(cols), sort=True, dropna=False, observed=True)
    code = g.ngroup().to_numpy(np.int64)
    keys = g.size().reset_index()[list(cols)]
    return code, keys


def _spot_codes(air: pd.DataFrame, spot_key: str, library_key: str | None) -> tuple[np.ndarray, pd.DataFrame]:
    """Code per row for a spot (a spot is (library, spot id) when the table has a library column)."""
    cols = [library_key, spot_key] if library_key is not None and library_key in air.columns else [spot_key]
    return _group_codes(air, cols)


def _n_distinct(a: np.ndarray, b: np.ndarray, n_a: int) -> np.ndarray:
    """Number of distinct ``b`` per ``a`` code (``b < 0`` = missing, not counted)."""
    ok = b >= 0
    if not ok.any():
        return np.zeros(n_a, np.int64)
    key = np.unique(a[ok].astype(np.int64) * (np.int64(b.max()) + 1) + b[ok].astype(np.int64))
    return np.bincount((key // (np.int64(b.max()) + 1)).astype(np.int64), minlength=n_a)


def _sink_spots(
    air: pd.DataFrame,
    spot_code: np.ndarray,
    n_spots: int,
    *,
    min_clones: int,
    spot_key: str,
    locus_key: str,
    clone_key: str,
    library_key: str | None,
) -> np.ndarray:
    """Boolean per spot code: the spot holds >= ``min_clones`` distinct clones (all loci) - a barcode sink.

    Uses M1's :func:`spaclone.pp.barcode_sinks` when the package provides it (one definition everywhere); otherwise
    counts distinct clones per spot here. Both apply the same rule.
    """
    mask = None
    try:  # reuse M1's implementation when present
        import importlib

        fn = getattr(importlib.import_module("spaclone.pp.sinks"), "barcode_sinks", None)
    except ImportError:  # pragma: no cover - module always exists in the package tree
        fn = None
    if fn is not None and locus_key in air.columns and clone_key in air.columns:
        tab = air
        if library_key is not None and library_key in air.columns:  # spots are (library, spot) pairs
            tab = air.assign(**{spot_key: air[library_key].astype("string") + "\x1f" + air[spot_key].astype("string")})
        try:
            res = fn(tab, min_clones=min_clones, spot_key=spot_key, locus_key=locus_key, clone_key=clone_key)
            mask = np.asarray(getattr(res, "mask", res), dtype=bool)
            if mask.shape != (len(air),):
                mask = None
        except (TypeError, ValueError, KeyError):
            mask = None
    if mask is None:
        clone = pd.factorize(air[clone_key])[0] if clone_key in air.columns else np.full(len(air), -1)
        n_clones = _n_distinct(spot_code, clone, n_spots)
        return n_clones >= min_clones
    out = np.zeros(n_spots, bool)
    np.logical_or.at(out, spot_code, mask)
    return out


# ----------------------------------------------------------------------------------------------- qc_metrics
class QCMetrics(NamedTuple):
    """Result of :func:`qc_metrics` (unpack as ``table, spots = qc_metrics(air)``).

    Attributes
    ----------
    table
        One row per group (``by``), see :func:`qc_metrics`.
    spots
        One row per spot (per library), see :func:`qc_metrics`.
    """

    table: pd.DataFrame
    spots: pd.DataFrame


def qc_metrics(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    sink_min_clones: int | None = 20,
    tissue_key: str | None = _DEFAULT_TISSUE,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    reads_key: str = S.READS,
    junction_key: str = S.JUNCTION_AA,
    productive_key: str = S.PRODUCTIVE,
    library_key: str = S.LIBRARY,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> QCMetrics:
    """Quality-control metrics of a receptor table, per group and per spot.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or cell call). Only ``spot_key`` is required; every metric whose
        columns are missing is reported as NaN.
    by
        Columns defining the groups of the summary table. ``None`` (default) uses whichever of ``library_key`` and
        ``locus_key`` are present; ``()`` gives one row for the whole table.
    quantiles
        Quantiles of reads per record reported as ``reads_q<100 q>`` (e.g. ``reads_q50``).
    sink_min_clones
        A spot with at least this many distinct clones (all loci together, within its library) is a barcode sink
        (the archive's 20 for 0.5-µm DNBs; M1's :func:`spaclone.pp.barcode_sinks` when available). ``None`` skips
        the sink metrics.
    tissue_key
        Boolean column, true on the tissue. The default ``"in_tissue"`` is used when present and skipped otherwise.
    spot_key, clone_key, locus_key, reads_key, junction_key, productive_key, library_key, x_key, y_key
        Column names. A spot is a (library, spot id) pair when the table has ``library_key``.

    Returns
    -------
    QCMetrics
        ``table``: the ``by`` columns and

        * ``n_records``, ``n_spots``, ``n_clones`` (distinct non-missing clones), ``reads_total``;
        * ``reads_mean`` and ``reads_q*`` (reads per record), ``share_single_read`` (records with exactly 1 read);
        * ``valid_share`` (IMGT junction rule, :func:`spaclone.pp.is_valid_junction`), ``productive_share``;
        * ``locus_share`` (records of the group / records of the group without the chain) when the chain is a ``by``
          column, otherwise ``share_<locus>`` columns (chain composition);
        * ``clones_per_spot_mean`` / ``_median`` / ``_max``, ``share_spots_one_clone`` (distinct clones of the group on
          each spot), ``spots_per_clone_mean`` / ``_median`` / ``_max``;
        * ``off_tissue_share`` (records off the tissue, when known);
        * ``sink_spot_share`` (spots of the group that are sinks) and ``sink_record_share``.

        ``spots``: one row per spot with the library and spot columns, ``x_um``, ``y_um`` (first record), ``n_records``,
        ``n_reads``, ``n_clones``, ``n_loci``, ``n_<locus>`` (records per chain), ``valid_share``,
        ``productive_share``, the tissue flag (when known) and ``is_sink``.

    Notes
    -----
    Reads per record are taken from ``reads_key`` (AIRR ``consensus_count``); records with unknown reads are ignored in
    the read metrics. Clones are counted as given in ``clone_key`` (define them first with
    :func:`spaclone.pp.define_clones` if needed).

    References
    ----------
    Wolf et al. 2018, doi:10.1186/s13059-017-1382-0 (scanpy ``calculate_qc_metrics``); McCarthy et al. 2017,
    doi:10.1093/bioinformatics/btw777 (scater per-cell QC); Lefranc et al. 2003, doi:10.1016/S0145-305X(02)00039-3
    (IMGT junction).

    Examples
    --------
    >>> table, spots = qc_metrics(air)  # doctest: +SKIP
    >>> table[["library_id", "locus", "n_records", "valid_share", "clones_per_spot_mean"]]  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if spot_key not in air.columns:
        raise KeyError(f"qc_metrics: column {spot_key!r} not in the table")
    by_cols = _resolve_by(air, by, (library_key, locus_key), "qc_metrics")
    tissue = _resolve_tissue(air, tissue_key, "qc_metrics")
    qs = [float(q) for q in quantiles]
    if any(not 0 <= q <= 1 for q in qs):
        raise ValueError("quantiles must lie in [0, 1]")
    lib = library_key if library_key in air.columns else None

    n = len(air)
    spot, spot_tab = _spot_codes(air, spot_key, lib)
    n_spot = len(spot_tab)
    gcode, groups = _group_codes(air, by_cols)
    n_g = len(groups)
    clone = pd.factorize(air[clone_key])[0] if clone_key in air.columns else np.full(n, -1)
    has_reads = reads_key in air.columns and pd.api.types.is_numeric_dtype(air[reads_key])
    reads = air[reads_key].to_numpy(dtype=float) if has_reads else np.full(n, np.nan)
    valid = None
    if junction_key in air.columns:
        from spaclone.pp.validity import is_valid_junction

        valid = np.asarray(is_valid_junction(air[junction_key]), dtype=bool)
    productive = _as_flag(air[productive_key]) if productive_key in air.columns else None
    in_tissue = _as_flag(air[tissue]) if tissue is not None else None
    sink_spot = None
    if sink_min_clones is not None:
        if sink_min_clones < 1:
            raise ValueError("sink_min_clones must be >= 1")
        sink_spot = _sink_spots(
            air,
            spot,
            n_spot,
            min_clones=int(sink_min_clones),
            spot_key=spot_key,
            locus_key=locus_key,
            clone_key=clone_key,
            library_key=lib,
        )

    # ---------------------------------------------------------------- group table
    t = groups.copy()
    t["n_records"] = np.bincount(gcode, minlength=n_g)
    t["n_spots"] = _n_distinct(gcode, spot, n_g)
    t["n_clones"] = _n_distinct(gcode, clone, n_g)
    finite = np.isfinite(reads)
    t["reads_total"] = np.bincount(gcode, weights=np.where(finite, reads, 0.0), minlength=n_g) if has_reads else np.nan
    rs = pd.Series(reads)
    if has_reads and finite.any():
        gr = rs.groupby(gcode)
        t["reads_mean"] = gr.mean().reindex(range(n_g)).to_numpy()
        qv = gr.quantile(qs).unstack().reindex(range(n_g))
        for q in qs:
            t[f"reads_q{100 * q:g}"] = qv[q].to_numpy()
        n_fin = np.bincount(gcode, weights=finite.astype(float), minlength=n_g)
        with np.errstate(invalid="ignore", divide="ignore"):
            t["share_single_read"] = np.bincount(gcode, weights=(reads == 1).astype(float), minlength=n_g) / n_fin
    else:
        t["reads_mean"] = np.nan
        for q in qs:
            t[f"reads_q{100 * q:g}"] = np.nan
        t["share_single_read"] = np.nan
    nrec = t["n_records"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["valid_share"] = (
            np.bincount(gcode, weights=valid.astype(float), minlength=n_g) / nrec if valid is not None else np.nan
        )
        t["productive_share"] = (
            np.bincount(gcode, weights=productive.astype(float), minlength=n_g) / nrec
            if productive is not None
            else np.nan
        )
    # chain composition
    if locus_key in air.columns:
        if locus_key in by_cols:
            parent = [c for c in by_cols if c != locus_key]
            if parent:
                tot = t.groupby(parent, dropna=False, observed=True)["n_records"].transform("sum")
            else:
                tot = pd.Series(float(n), index=t.index)
            t["locus_share"] = t["n_records"] / tot.to_numpy(dtype=float)
        else:
            loc = air[locus_key].astype("string").fillna("NA").to_numpy(dtype=object)
            for lv in sorted(set(loc)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    t[f"share_{lv}"] = np.bincount(gcode, weights=(loc == lv).astype(float), minlength=n_g) / nrec
    # clones per spot and spots per clone, within each group
    cps = _per_pair_counts(gcode, spot, clone, n_g)
    t["clones_per_spot_mean"], t["clones_per_spot_median"], t["clones_per_spot_max"], one = cps
    t["share_spots_one_clone"] = one
    spc = _per_pair_counts(gcode, np.where(clone >= 0, clone, -1), np.where(clone >= 0, spot, -1), n_g)
    t["spots_per_clone_mean"], t["spots_per_clone_median"], t["spots_per_clone_max"], _ = spc
    if in_tissue is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            t["off_tissue_share"] = np.bincount(gcode, weights=(~in_tissue).astype(float), minlength=n_g) / nrec
    if sink_spot is not None:
        on_sink = sink_spot[spot]
        pairs = np.unique(gcode.astype(np.int64) * max(n_spot, 1) + spot)
        pg, ps = pairs // max(n_spot, 1), pairs % max(n_spot, 1)
        n_sink_spots = np.bincount(pg, weights=sink_spot[ps].astype(float), minlength=n_g)
        with np.errstate(invalid="ignore", divide="ignore"):
            t["sink_spot_share"] = n_sink_spots / t["n_spots"].to_numpy(dtype=float)
            t["sink_record_share"] = np.bincount(gcode, weights=on_sink.astype(float), minlength=n_g) / nrec

    # ---------------------------------------------------------------- spot table
    sp = spot_tab.copy()
    first = np.full(n_spot, -1, np.int64)
    first[spot[::-1]] = np.arange(n)[::-1]
    for k in (x_key, y_key):
        if k in air.columns:
            sp[k] = air[k].to_numpy()[first]
    sp["n_records"] = np.bincount(spot, minlength=n_spot)
    sp["n_reads"] = np.bincount(spot, weights=np.where(finite, reads, 0.0), minlength=n_spot) if has_reads else np.nan
    sp["n_clones"] = _n_distinct(spot, clone, n_spot)
    if locus_key in air.columns:
        locus_code, locus_vals = pd.factorize(air[locus_key].astype("string").fillna("NA"), sort=True)
        sp["n_loci"] = _n_distinct(spot, locus_code, n_spot)
        for i, lv in enumerate(locus_vals):
            sp[f"n_{lv}"] = np.bincount(spot, weights=(locus_code == i).astype(float), minlength=n_spot).astype(
                np.int64
            )
    srec = sp["n_records"].to_numpy(dtype=float)
    if valid is not None:
        sp["valid_share"] = np.bincount(spot, weights=valid.astype(float), minlength=n_spot) / srec
    if productive is not None:
        sp["productive_share"] = np.bincount(spot, weights=productive.astype(float), minlength=n_spot) / srec
    if in_tissue is not None:
        sp[tissue] = in_tissue[first]
    if sink_spot is not None:
        sp["is_sink"] = sink_spot
    return QCMetrics(t.reset_index(drop=True), sp.reset_index(drop=True))


def _per_pair_counts(gcode: np.ndarray, a: np.ndarray, b: np.ndarray, n_g: int):
    """Per group: mean / median / max over ``a`` units of the number of distinct ``b`` (b < 0 ignored).

    Returns the three summaries and the share of ``a`` units with exactly one distinct ``b``.
    """
    ok = (a >= 0) & (b >= 0)
    if not ok.any():
        nan = np.full(n_g, np.nan)
        return nan, nan, nan, nan
    tab = pd.DataFrame({"g": gcode[ok], "a": a[ok], "b": b[ok]}).drop_duplicates()
    per = tab.groupby(["g", "a"], sort=False).size()
    by_g = per.groupby(level="g")
    mean = by_g.mean().reindex(range(n_g)).to_numpy(dtype=float)
    med = by_g.median().reindex(range(n_g)).to_numpy(dtype=float)
    mx = by_g.max().reindex(range(n_g)).to_numpy(dtype=float)
    one = (per == 1).groupby(level="g").mean().reindex(range(n_g)).to_numpy(dtype=float)
    return mean, med, mx, one


# ----------------------------------------------------------------------------------------------- knee
class Knee(NamedTuple):
    """Result of :func:`knee_point`: the knee and the inflection of a read-count distribution (NaN when undefined)."""

    knee: float
    inflection: float


def knee_point(reads: Any, *, exclude_from: int = 50, lower: float = 0.0) -> Knee:
    """Knee and inflection of the distribution of reads per record (the barcode-rank rule of DropletUtils).

    Records are ranked by reads (descending; tied records share their mid-rank) and the curve log10(reads) against
    log10(rank) is differentiated numerically. The **inflection** is the steepest drop (after the ``exclude_from``
    highest-ranked records); the flattest point before it is the left edge; the **knee** is the point between the two
    edges farthest above the straight line joining them. Records-kept-against-read-floor (:func:`read_support_curve`)
    is the same curve with its axes swapped, so the knee is the floor where the deep, well-supported records end and
    the cliff down to the low-read cloud begins.

    Parameters
    ----------
    reads
        Reads per record (non-finite values ignored).
    exclude_from
        Number of highest-ranked records excluded from the search for the steepest drop (DropletUtils default 50: the
        few deepest records give steep but meaningless slopes).
    lower
        Only records with more than ``lower`` reads are used.

    Returns
    -------
    Knee
        ``knee`` and ``inflection`` in reads; NaN when fewer than three distinct read values remain.

    Notes
    -----
    A description of the read distribution, not a test: it has no null model, and a floor at the knee does not show
    that the records above it are correctly placed (``spaclone.qc.read_floor`` tests that).

    References
    ----------
    Lun et al. 2019, doi:10.1186/s13059-019-1662-y (DropletUtils ``barcodeRanks``: knee = the point farthest from the
    line between the curve bounds; inflection = the steepest drop); Satopää et al. 2011, doi:10.1109/ICDCSW.2011.20
    (Kneedle).
    """
    v = np.asarray(reads, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > lower)]
    if v.size == 0:
        return Knee(float("nan"), float("nan"))
    u, cnt = np.unique(v, return_counts=True)
    totals = u[::-1]
    lengths = cnt[::-1].astype(float)
    if len(totals) < 3:
        return Knee(float("nan"), float("nan"))
    rank = np.cumsum(lengths) - (lengths - 1) / 2  # mid-rank of each run of ties
    x = np.log10(rank)
    y = np.log10(totals)
    d1 = np.diff(y) / np.diff(x)
    skip = int(min(len(d1) - 1, np.sum(x <= np.log10(max(exclude_from, 1)))))
    right = skip + int(np.argmin(d1[skip:]))
    left = skip + int(np.argmax(d1[skip : right + 1]))
    inflection = float(totals[right])
    xs, ys = x[left : right + 1], y[left : right + 1]
    if len(xs) < 2 or xs[-1] == xs[0]:
        return Knee(float(totals[left]), inflection)
    grad = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    icpt = ys[0] - xs[0] * grad
    above = np.flatnonzero(ys >= xs * grad + icpt)
    if len(above) == 0:
        return Knee(float(10 ** ys[0]), inflection)
    dist = np.abs(grad * xs[above] - ys[above] + icpt) / np.sqrt(grad**2 + 1)
    knee = float(totals[left + above[int(np.argmax(dist))]])
    return Knee(knee, inflection)


# ----------------------------------------------------------------------------------------------- read support
class ReadSupport(NamedTuple):
    """Result of :func:`read_support_curve` (unpack as ``curve, knee = read_support_curve(air)``).

    Attributes
    ----------
    curve
        One row per group and read floor, see :func:`read_support_curve`.
    knee
        One row per group: ``knee_reads`` and the records / spots kept there.
    """

    curve: pd.DataFrame
    knee: pd.DataFrame


def read_support_curve(
    air: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = None,
    floors: Sequence[float] | None = None,
    exclude_from: int = 50,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
    library_key: str = S.LIBRARY,
) -> ReadSupport:
    """Records and spots kept by each read floor, and the knee of that curve.

    Parameters
    ----------
    air
        Receptor table with reads per record (``reads_key``, AIRR ``consensus_count``).
    by
        Groups with their own curve (e.g. ``"locus"``). ``None`` (default) = whichever of ``library_key`` and
        ``"locus"`` are present; ``()`` = one curve.
    floors
        Read floors to evaluate (records with ``reads >= floor`` are kept). Default: every distinct read value of the
        group (the curve only changes there).
    exclude_from
        Passed to :func:`knee_point` (deepest records ignored when looking for the steepest drop).
    reads_key, spot_key, library_key
        Column names. Records with unknown reads are never kept.

    Returns
    -------
    ReadSupport
        ``curve``: the ``by`` columns, ``min_reads``, ``records``, ``share_records``, ``spots`` (spots keeping >= 1
        record), ``share_spots``. ``knee``: the ``by`` columns, ``knee_reads`` and ``inflection_reads``
        (:func:`knee_point`; NaN with fewer than three distinct read values), and the ``records``, ``share_records``,
        ``spots`` and ``share_spots`` kept by a floor at the knee.

    Notes
    -----
    The knee marks where the cloud of low-read records ends; it is a description of the read distribution, not
    evidence that records above it are correctly placed (``spaclone.qc.read_floor`` tests that).

    References
    ----------
    Lun et al. 2019, doi:10.1186/s13059-019-1662-y (DropletUtils barcode ranks); Stern et al. 2014,
    doi:10.1126/scitranslmed.3008879 (pRESTO: keep sequences supported by >= 2 reads).

    Examples
    --------
    >>> curve, knee = read_support_curve(air, by="locus")  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if reads_key not in air.columns:
        raise KeyError(f"read_support_curve: column {reads_key!r} not in the table")
    if spot_key not in air.columns:
        raise KeyError(f"read_support_curve: column {spot_key!r} not in the table")
    by_cols = _resolve_by(air, by, (library_key, S.LOCUS), "read_support_curve")
    lib = library_key if library_key in air.columns else None
    spot, _ = _spot_codes(air, spot_key, lib)
    gcode, groups = _group_codes(air, by_cols)
    reads = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)
    fl = None if floors is None else np.unique(np.asarray(floors, dtype=float))
    curves, knees = [], []
    for g in range(len(groups)):
        sel = gcode == g
        r = reads[sel]
        n_rec = int(sel.sum())
        fin = np.isfinite(r)
        rs = np.sort(r[fin])
        smax = pd.Series(r).groupby(spot[sel]).max()
        sm = np.sort(smax.to_numpy(dtype=float)[np.isfinite(smax.to_numpy(dtype=float))])
        n_sp = int(len(smax))
        f = fl if fl is not None else np.unique(rs)
        if f is None or len(f) == 0:
            f = np.array([1.0])
        rec = len(rs) - np.searchsorted(rs, f, side="left")
        spt = len(sm) - np.searchsorted(sm, f, side="left")
        c = pd.DataFrame(
            {
                "min_reads": f,
                "records": rec.astype(np.int64),
                "share_records": rec / max(n_rec, 1),
                "spots": spt.astype(np.int64),
                "share_spots": spt / max(n_sp, 1),
            }
        )
        for col in reversed(by_cols):
            c.insert(0, col, groups[col].iloc[g])
        curves.append(c)
        kn = knee_point(rs, exclude_from=exclude_from)
        row = {col: groups[col].iloc[g] for col in by_cols}
        row.update(knee_reads=kn.knee, inflection_reads=kn.inflection)
        if np.isfinite(kn.knee):
            kept = int((rs >= kn.knee).sum())
            ksp = int((sm >= kn.knee).sum())
            row.update(records=kept, share_records=kept / max(n_rec, 1), spots=ksp, share_spots=ksp / max(n_sp, 1))
        else:
            row.update(records=np.nan, share_records=np.nan, spots=np.nan, share_spots=np.nan)
        knees.append(row)
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    knee = pd.DataFrame(
        knees, columns=[*by_cols, "knee_reads", "inflection_reads", "records", "share_records", "spots", "share_spots"]
    )
    return ReadSupport(curve, knee)
