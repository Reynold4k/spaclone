"""One front door for record filtering: valid junction, productive flag, read floor, tissue, chain and sinks.

:func:`filter_records` applies, in order,

1. the IMGT junction rules of the project standard through :func:`spaclone.pp.filter_valid` (C104 ... W/F118 on every
   chain, the 20 standard amino acids, >= 5 aa; the archive's sequence-validity standard), or the looser legacy rule
   "no stop codon or frame shift" (``valid_junction="in_frame"``, step 5 of the archived denoising ladder);
2. the productive / functional flag;
3. a read floor: fixed, per chain, or ``"auto"`` (the data-driven floor of :func:`spaclone.qc.read_floor` for every
   chain pair present, seeded, so the same table always gets the same floor);
4. on-tissue position;
5. a chain (locus) selection;
6. barcode sinks (spots with at least ``sink_min_clones`` distinct clones),

and reports, for every step, the records, spots and clones it removed and those left. It reproduces the archived QC
filter table (``QC_filter_summary.csv``; ``validation/results/m6_filtering.csv``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pp.metrics import _as_flag, _cols, _group_codes, _sink_spots, _spot_codes, read_support_curve
from spaclone.pp.validity import FilterResult, filter_valid

__all__ = ["filter_records", "resolve_min_reads"]

logger = logging.getLogger("spaclone")

#: Chain pairs (focal, partner) that ``min_reads="auto"`` tests with :func:`spaclone.qc.read_floor`, in this order.
AUTO_PAIRS: tuple[tuple[str, str], ...] = (("TRB", "TRA"), ("IGH", "IGK"))
#: Seed of :func:`spaclone.qc.read_floor` inside ``min_reads="auto"`` (unless ``read_floor_kwargs`` gives ``rng``).
AUTO_SEED = 0


def _floor_from_read_floor(
    air: pd.DataFrame, kwargs: Mapping[str, Any] | None, *, reads_key: str, locus_key: str
) -> tuple[dict[str, float] | None, str]:
    """Floors per chain from ``spaclone.qc.read_floor`` on every testable chain pair.

    Returns ``(floors, how)``: ``floors`` maps the focal and partner chain of every pair with a suggested floor to it
    (pairs tested without a suggestion contribute nothing); ``None`` when no pair could be tested.
    """
    # the module, not a same-named function re-exported by the namespace
    fn = getattr(importlib.import_module("spaclone.qc.read_floor"), "read_floor", None)
    if fn is None:  # pragma: no cover - the module always defines it
        return None, "qc.read_floor not available"
    kw = dict(kwargs or {})
    kw.setdefault("rng", AUTO_SEED)
    if "locus" in kw or "partner_locus" in kw:
        defaults = inspect.signature(fn).parameters
        pairs = [(str(kw.pop("locus", defaults["locus"].default)), str(kw.pop("partner_locus",
                                                                              defaults["partner_locus"].default)))]  # fmt: skip
    else:
        present: set[str] = set()
        if locus_key in air.columns and reads_key in air.columns:
            r = pd.to_numeric(air[reads_key], errors="coerce")
            present = set(air.loc[r.notna(), locus_key].astype(str))
        pairs = [p for p in AUTO_PAIRS if set(p) <= present]
    floors: dict[str, float] = {}
    notes: list[str] = []
    n_tested = 0
    for focal, partner in pairs:
        try:
            res = fn(air, locus=focal, partner_locus=partner, **kw)
        except (ValueError, KeyError) as e:  # the pair cannot be tested (too few records, columns missing)
            notes.append(f"{focal}/{partner} not testable ({e})")
            continue
        n_tested += 1
        v = getattr(res, "summary", {}).get("suggested_floor")
        if isinstance(v, Mapping):
            floors.update({str(a): float(b) for a, b in v.items() if np.isfinite(float(b))})
            notes.append(f"{focal}/{partner}: per chain")
        elif v is not None and np.isfinite(float(v)):
            floors[focal] = floors[partner] = float(v)
            notes.append(f"{focal}/{partner} >= {float(v):g}")
        else:
            notes.append(f"{focal}/{partner}: no floor")
    if not pairs:
        notes.append("no chain pair of " + ", ".join("/".join(p) for p in AUTO_PAIRS) + " in the table")
    return (floors if n_tested else None), "; ".join(notes)


def resolve_min_reads(
    air: pd.DataFrame,
    min_reads: float | Mapping[str, float] | Literal["auto"] | None,
    *,
    read_floor_kwargs: Mapping[str, Any] | None = None,
    reads_key: str = S.READS,
    locus_key: str = S.LOCUS,
    spot_key: str = S.SPOT,
    library_key: str = S.LIBRARY,
) -> tuple[float | dict[str, float] | None, str]:
    """Turn a ``min_reads`` argument into a floor (number or per-chain dict) and say where it came from.

    Parameters
    ----------
    air
        Receptor table.
    min_reads
        ``None`` (no floor), a number, a mapping chain -> floor, or ``"auto"``, which does exactly this:

        1. every chain pair of ``spaclone.pp.filtering.AUTO_PAIRS`` (TRB/TRA, then IGH/IGK) whose two chains have
           records with read counts is tested with :func:`spaclone.qc.read_floor` (partner-chain sharing of
           same-clonotype loci against shuffles at the floors 2, 5, 10, 20, 50, 100, the suggestion corrected for
           testing several floors), seeded with ``spaclone.pp.filtering.AUTO_SEED`` (0) unless ``read_floor_kwargs``
           gives ``rng``, so the same table always gets the same floor; ``read_floor_kwargs`` with ``locus`` /
           ``partner_locus`` tests that one pair instead;
        2. a pair with a suggested floor gives it to both of its chains; a pair tested without a suggestion gives no
           floor ("no floor": the data show no read support at which partner sharing departs from chance); chains in
           no tested pair (e.g. IGL) get no floor;
        3. only when no pair can be tested at all (no partner chain, too few records) does it fall back to the
           inflection of each chain's read-support curve (:func:`spaclone.pp.knee_point`: the steepest drop between
           the low-read cloud and the deep records), which only describes the read distribution and says so in the
           source.
    read_floor_kwargs
        Keyword arguments passed to :func:`spaclone.qc.read_floor` (e.g. ``rng``, ``n_perm``, ``floors``, or
        ``locus="IGH"``, ``partner_locus="IGL"`` to test one chosen pair).
    reads_key, locus_key, spot_key, library_key
        Column names.

    Returns
    -------
    tuple
        ``(floor, source)``. ``floor`` is ``None``, a number or a dict chain -> floor (empty when ``"auto"`` tested a
        pair but found no floor); ``source`` is ``"none"``, ``"fixed"``, ``"per locus"``, ``"auto: qc.read_floor
        (<pair results>)"`` or ``"auto: inflection of the read-support curve (<why qc.read_floor could not run>)"``.
    """
    if min_reads is None:
        return None, "none"
    if isinstance(min_reads, str):
        if min_reads != "auto":
            raise ValueError(f"min_reads must be a number, a mapping, 'auto' or None, got {min_reads!r}")
        floor, how = _floor_from_read_floor(air, read_floor_kwargs, reads_key=reads_key, locus_key=locus_key)
        if floor is not None:
            return floor, f"auto: qc.read_floor ({how})"
        logger.info(
            "filter_records: qc.read_floor could not test the table (%s); using the read-support inflection", how
        )
        by = [locus_key] if locus_key in air.columns else []
        _, knee = read_support_curve(air, by=by, reads_key=reads_key, spot_key=spot_key, library_key=library_key)
        src = f"auto: inflection of the read-support curve ({how})"
        infl = knee.inflection_reads.to_numpy(dtype=float)
        if by:
            out = {str(k): float(v) for k, v in zip(knee[locus_key], infl, strict=True) if np.isfinite(v)}
            for k in knee[locus_key][~np.isfinite(infl)]:
                out[str(k)] = 1.0
                logger.info("filter_records: no inflection in the read-support curve of %s; floor 1 used", k)
            return out, src
        kv = float(infl[0]) if len(infl) else np.nan
        return (kv if np.isfinite(kv) else 1.0), src
    if isinstance(min_reads, Mapping):
        return {str(k): float(v) for k, v in min_reads.items()}, "per locus"
    v = float(min_reads)
    if not np.isfinite(v):
        raise ValueError(f"min_reads must be finite, got {min_reads!r}")
    return v, "fixed"


def filter_records(
    air: pd.DataFrame,
    *,
    valid_junction: bool | Literal["imgt", "in_frame"] = True,
    productive: bool = True,
    min_reads: float | Mapping[str, float] | Literal["auto"] | None = None,
    on_tissue: bool | str | None = None,
    loci: str | Sequence[str] | None = None,
    sink_min_clones: int | None = None,
    min_length: int = 5,
    by: str | Sequence[str] | None = None,
    read_floor_kwargs: Mapping[str, Any] | None = None,
    junction_key: str = S.JUNCTION_AA,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> FilterResult:
    """Filter receptor records step by step and report what each step removed.

    Parameters
    ----------
    air
        Receptor table.
    valid_junction
        ``True`` / ``"imgt"``: the IMGT junction rules applied by :func:`spaclone.pp.filter_valid` (starts with C,
        ends with W or F on every chain, only the 20 standard amino acids, at least ``min_length`` aa).
        ``"in_frame"``: only "no stop codon ``*`` and no frame shift ``_``" (the legacy "functional CDR3" rule of the
        archive's denoising ladder). ``False``: skip. Missing junctions fail both rules.
    productive
        Require a true productive / functional flag (``T``, ``TRUE``, ``True``, 1, ``Y``). Reported as not applied
        when the table has no ``productive_key`` column.
    min_reads
        Read floor (records with ``reads_key >= floor`` are kept; unknown reads fail): ``None`` (no floor; long-read
        data or data without reads), a number, a mapping chain -> floor (chains not in the mapping get no floor), or
        ``"auto"`` (see :func:`resolve_min_reads`: :func:`spaclone.qc.read_floor`, seeded, on every chain pair present
        (TRB/TRA, IGH/IGK); its multiplicity-corrected suggestion, or no floor for a pair without one; the inflection
        of each chain's read-support curve only when no pair can be tested).
    on_tissue
        ``True`` keeps records whose ``"in_tissue"`` column is true; a string names the boolean column to use;
        ``None`` / ``False`` skips the step.
    loci
        Chain(s) to keep (e.g. ``"IGH"`` or ``("TRA", "TRB")``).
    sink_min_clones
        Remove records on barcode sinks: spots holding at least this many distinct clones (all loci, judged on the
        input table, as in the archive; :func:`spaclone.pp.barcode_sinks`). ``None`` keeps them.
    min_length
        Minimum junction length in amino acids for the IMGT rule.
    by
        Column(s) for a stratified report (e.g. ``"locus"``); the filter is the same for every stratum.
    read_floor_kwargs
        Passed to :func:`spaclone.qc.read_floor` when ``min_reads="auto"`` (e.g. ``{"rng": 1}`` for another seed).
    junction_key, productive_key, reads_key, spot_key, clone_key, locus_key, library_key
        Column names. A spot is a (library, spot id) pair when the table has ``library_key``.

    Returns
    -------
    FilterResult
        ``air``: the records passing every applied step (index and columns kept; the parameters are recorded in
        ``attrs["spaclone"]["filter_records"]``). ``report``: one row per step (and stratum) with ``step`` (0 =
        input), ``rule``, ``applied``, ``failing`` (input records failing the rule, regardless of order), ``removed``,
        ``spots_removed``, ``clones_removed`` (removed at this step) and ``records``, ``spots``, ``clones`` (left after
        the step).

    Notes
    -----
    Every step is a record-level mask, so the final table does not depend on the order; the order only decides
    which step a record is attributed to. Spots and clones are counted as distinct (library, spot) pairs and distinct
    non-missing clone ids.

    References
    ----------
    Lefranc et al. 2003, doi:10.1016/S0145-305X(02)00039-3 (IMGT junction, C104 to W/F118); Vander Heiden et al.
    2018, doi:10.4049/jimmunol.1800669 (AIRR Rearrangement schema, ``productive``); Stern et al. 2014,
    doi:10.1126/scitranslmed.3008879 (pRESTO: sequences supported by >= 2 reads).

    Examples
    --------
    >>> ok, report = filter_records(air, min_reads=2)  # doctest: +SKIP
    >>> report[["rule", "records", "spots", "clones"]]  # doctest: +SKIP
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    if spot_key not in air.columns:
        raise KeyError(f"filter_records: column {spot_key!r} not in the table")
    if valid_junction not in (True, False, "imgt", "in_frame"):
        raise ValueError(f"valid_junction must be True, False, 'imgt' or 'in_frame', got {valid_junction!r}")
    if valid_junction is not False and junction_key not in air.columns:
        raise KeyError(f"filter_records: column {junction_key!r} not in the table (pass valid_junction=False)")
    tissue_col = None
    if on_tissue is True:
        tissue_col = "in_tissue"
    elif isinstance(on_tissue, str):
        tissue_col = on_tissue
    elif on_tissue not in (None, False):
        raise ValueError(f"on_tissue must be True, False, None or a column name, got {on_tissue!r}")
    if tissue_col is not None and tissue_col not in air.columns:
        raise KeyError(f"filter_records: tissue column {tissue_col!r} not in the table")
    by_cols = _cols(by)
    missing = [c for c in by_cols if c not in air.columns]
    if missing:
        raise KeyError(f"filter_records: column(s) {missing} not in the table")
    loci_keep = None if loci is None else ([loci] if isinstance(loci, str) else list(loci))
    if loci_keep is not None and locus_key not in air.columns:
        raise KeyError(f"filter_records: loci given but column {locus_key!r} not in the table")
    floor, floor_source = resolve_min_reads(
        air,
        min_reads,
        read_floor_kwargs=read_floor_kwargs,
        reads_key=reads_key,
        locus_key=locus_key,
        spot_key=spot_key,
        library_key=library_key,
    )
    if floor is not None and reads_key not in air.columns:
        raise KeyError(f"filter_records: min_reads given but column {reads_key!r} not in the table")
    if isinstance(floor, dict) and locus_key not in air.columns:
        raise KeyError(f"filter_records: per-locus min_reads needs column {locus_key!r}")

    n = len(air)
    rules: list[tuple[str, bool, np.ndarray]] = []
    # 1 valid junction (M4's filter_valid applies the IMGT rules)
    if valid_junction in (True, "imgt"):
        tmp = pd.DataFrame({junction_key: air[junction_key].to_numpy()}, index=pd.RangeIndex(n))
        ok_tab, _ = filter_valid(tmp, require_productive=False, min_length=min_length, junction_key=junction_key)
        mask = np.zeros(n, bool)
        mask[ok_tab.index.to_numpy()] = True
        rules.append((f"valid junction (C...W/F, 20 aa, >= {min_length} aa)", True, mask))
    elif valid_junction == "in_frame":
        jx = air[junction_key].astype("string")
        mask = (jx.notna() & (jx.str.len() > 0) & ~jx.str.contains(r"[*_]", regex=True)).fillna(False)
        rules.append(("in-frame junction (no stop codon or frame shift)", True, mask.to_numpy(dtype=bool)))
    else:
        rules.append(("valid junction (not applied)", False, np.ones(n, bool)))
    # 2 productive
    if productive:
        if productive_key in air.columns:
            rules.append(("productive", True, _as_flag(air[productive_key])))
        else:
            rules.append((f"productive (no {productive_key!r} column)", False, np.ones(n, bool)))
    # 3 read floor
    if isinstance(floor, dict) and not floor:  # "auto" tested a chain pair and found no floor
        rules.append((f"read floor: none ({floor_source})", False, np.ones(n, bool)))
    elif floor is not None:
        r = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(dtype=float)
        if isinstance(floor, dict):
            loc = air[locus_key].astype("string").fillna("NA").to_numpy(dtype=object)
            thr = np.array([floor.get(str(v), -np.inf) for v in loc], dtype=float)
            label = ", ".join(f"{k} >= {v:g}" for k, v in sorted(floor.items()))
            ok_r = np.isneginf(thr) | (np.isfinite(r) & (r >= thr))  # chains without a floor keep unknown reads
            rules.append((f"reads per locus ({label}; {floor_source})", True, ok_r))
        else:
            src = "" if floor_source == "fixed" else f" ({floor_source})"
            rules.append((f"reads >= {floor:g}{src}", True, np.isfinite(r) & (r >= floor)))
    # 4 tissue
    if tissue_col is not None:
        rules.append((f"on tissue ({tissue_col})", True, _as_flag(air[tissue_col])))
    # 5 loci
    if loci_keep is not None:
        lk = air[locus_key].astype("string").isin(loci_keep).fillna(False).to_numpy(dtype=bool)
        rules.append((f"loci {'/'.join(loci_keep)}", True, lk))
    lib = library_key if library_key in air.columns else None
    spot, spot_tab = _spot_codes(air, spot_key, lib)
    # 6 sinks
    if sink_min_clones is not None:
        if sink_min_clones < 1:
            raise ValueError("sink_min_clones must be >= 1")
        if clone_key not in air.columns:
            raise KeyError(f"filter_records: sink_min_clones needs column {clone_key!r}")
        sinks = _sink_spots(
            air,
            spot,
            len(spot_tab),
            min_clones=int(sink_min_clones),
            spot_key=spot_key,
            locus_key=locus_key,
            clone_key=clone_key,
            library_key=lib,
        )
        rules.append((f"not on a barcode sink (< {sink_min_clones} clones per spot)", True, ~sinks[spot]))

    keep = np.ones(n, bool)
    steps = [("input", True, keep.copy(), np.zeros(n, bool))]
    for name, applied, ok in rules:
        keep = keep & ok
        steps.append((name, applied, keep.copy(), ~ok))
    clone = pd.factorize(air[clone_key])[0] if clone_key in air.columns else np.full(n, -1)

    if by_cols:
        gcode, keys = _group_codes(air, by_cols)
        strata = [(tuple(keys.iloc[g]), gcode == g) for g in range(len(keys))]
    else:
        strata = [((), np.ones(n, bool))]
    rows = []
    for key, sel in strata:
        prev = None
        for i, (name, applied, left, failing) in enumerate(steps):
            m = left & sel
            n_sp = int(len(np.unique(spot[m])))
            cl = clone[m]
            n_cl = int(len(np.unique(cl[cl >= 0])))
            rec = int(m.sum())
            row = dict(zip(by_cols, key, strict=True))
            row.update(
                step=i,
                rule=name,
                applied=applied,
                failing=int((failing & sel).sum()),
                removed=0 if prev is None else prev[0] - rec,
                spots_removed=0 if prev is None else prev[1] - n_sp,
                clones_removed=0 if prev is None else prev[2] - n_cl,
                records=rec,
                spots=n_sp,
                clones=n_cl,
            )
            rows.append(row)
            prev = (rec, n_sp, n_cl)
    cols = [*by_cols, "step", "rule", "applied", "failing", "removed", "spots_removed", "clones_removed"]
    report = pd.DataFrame(rows, columns=[*cols, "records", "spots", "clones"])
    out = air[keep].copy()
    out.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "filter_records": {
            "valid_junction": valid_junction,
            "productive": productive,
            "min_reads": floor,
            "min_reads_source": floor_source,
            "on_tissue": tissue_col,
            "loci": loci_keep,
            "sink_min_clones": sink_min_clones,
            "min_length": min_length,
        },
    }
    return FilterResult(out, report)
