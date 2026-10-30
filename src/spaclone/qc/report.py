"""The location-trust report: pre-stated criteria that turn test results into two verdicts and an overall one.

Two levels of evidence are kept apart, because they answer different questions:

* **molecule-level placement** (:func:`~spaclone.qc.same_spot`, :func:`~spaclone.qc.own_type_share`,
  :func:`~spaclone.qc.umi_match`): does a receptor molecule sit on a spot with its own constant-region transcript, or
  on a unit of its own cell type? Any molecule whose *barcode* is right passes, even when its CDR3 is not: PCR template
  switching inside the shared constant region keeps the acceptor's constant region and barcode (a spot that does hold
  that transcript) and the donor's V(D)J.
* **clone-level placement** (:func:`~spaclone.qc.colocalisation` of heavy chains, heavy/light
  :func:`~spaclone.qc.pairing`, partner-chain sharing from :func:`~spaclone.qc.read_floor`): does clone identity
  follow position? Only this level licenses clone maps and spatial clone statistics. Light chains (IGK, IGL, TRA, TRG)
  are not clone markers — their CDR3s are shared by unrelated cells (public, convergent junctions) and the light-chain
  type can be local — so light-chain co-localisation is reported as information only.

Every criterion is fixed here, before any data are seen, and printed with the verdicts (:data:`CRITERIA`,
:data:`OVERALL_RULES`). A row is **signal** when it beats random placement in the direction a genuine positional
signal predicts with Benjamini–Hochberg q < 0.05 inside its level's family, **not detected** when it was tested and did
not (absence of evidence, never evidence of absence), **inconclusive** when the design cannot decide (no depth strata;
fewer hits than the chance model allows), and **untestable** when it could not be computed or the data are too few
(NaN p-values included). Untestable and inconclusive rows never count as "no signal".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spaclone import _schema as S
from spaclone._utils import TestResult, bh

__all__ = ["CRITERIA", "OVERALL_RULES", "TrustReport", "evaluate_trust"]

#: Level of every decision (one-sided tests, BH within a level's family).
ALPHA = 0.05
#: Minimum singleton spots for the same-spot test, records for the own-type share, clones with pairs in the band for
#: co-localisation, tested clonotypes for partner sharing and receptor keys for the UMI match.
MIN_SPOTS, MIN_RECORDS, MIN_CLONES, MIN_TESTED, MIN_KEYS = 100, 20, 5, 5, 20
#: Median per-spot chance above which the same-spot test is at its ceiling (every spot holds the transcript).
CEILING_CHANCE = 0.5
#: Share of the decisive rows that ran which must be testable before "not detected" / "no evidence" (else limited).
MIN_TESTABLE_SHARE = 0.5
#: Loci whose clone ids mark clones (heavy chains).
CLONE_MARKER_LOCI = S.HEAVY_LOCI

MOLECULE_TESTS = ("same_spot", "own_type_share", "umi_match")
CLONE_TESTS = ("colocalisation", "pairing", "partner_sharing")
DECISIVE = MOLECULE_TESTS + CLONE_TESTS
LEVEL = {**dict.fromkeys(MOLECULE_TESTS, "molecule"), **dict.fromkeys(CLONE_TESTS, "clone")}
_ORDER = (*DECISIVE, "barcode_sinks", "read_floor")
#: Level of every step of :func:`spaclone.run` (for the reasons a level was not tested).
STEP_LEVEL = {**LEVEL, "read_floor": "clone"}
_TESTED = ("signal", "not detected")
_RUN = ("signal", "not detected", "inconclusive", "untestable")

#: The pre-stated criteria, one entry per test and for the multiplicity rule (printed in every report).
CRITERIA: dict[str, str] = {
    "same_spot": (
        "Molecule level. Singleton spots (one clone of the chain) in tissue against the per-spot chance model (exact, "
        "one-sided): signal if hits exceed chance with BH q < 0.05 in the molecule-level family; inconclusive if hits "
        "fall below chance (deficit: the chance model over-states chance, so no bound is claimed); untestable with "
        f"< {MIN_SPOTS} spots, on spots that are not a regular grid (beads) or cell-sized or larger, or when the "
        f"median chance exceeds {CEILING_CHANCE:g} (test at ceiling); otherwise not detected, with the 95% upper bound "
        "of located fraction x detection probability."
    ),
    "own_type_share": (
        "Molecule level. Share of receptor records on units of their own cell type (B lineage for BCR, T for TCR), no "
        "read floor, barcode sinks removed, against whole spots relocated to typed units of the same library with "
        "probability proportional to captured cDNA (GEX total counts): signal if BH q < 0.05 in the molecule-level "
        f"family; untestable with < {MIN_RECORDS} records on typed units."
    ),
    "umi_match": (
        "Molecule level. Receptor (barcode, UMI) keys whose GEX reads call their own constant gene, against decoy UMIs "
        "on the same barcode (exact): signal if BH q < 0.05 in the molecule-level family; untestable with "
        f"< {MIN_KEYS} keys."
    ),
    "colocalisation": (
        "Clone level, heavy chains only (IGH, TRB, TRD). Same-clone pairs 20-100 um apart among cell-scale units "
        "(sub-cellular spots: loci of same-clone spots within 15 um; barcode sinks removed) against clone labels "
        "permuted within strata of local Ig RNA: signal if BH q < 0.05 in the clone-level family; untestable with "
        f"< {MIN_CLONES} clones contributing pairs; inconclusive without depth strata (whole-sample null: depth not "
        "controlled). Light chains (IGK, IGL, TRA, TRG) are information only: their CDR3s are shared by unrelated "
        "cells."
    ),
    "pairing": (
        "Clone level. Heavy/light co-occurrence in 10-um bins (IGH with IGK and IGL, TRB with TRA, TRD with TRG; "
        "barcode sinks removed) against heavy labels permuted within clones-per-spot strata: signal if BH q < 0.05 in "
        "the clone-level family; untestable when fewer than two heavy or two light clones have min_units units."
    ),
    "partner_sharing": (
        "Clone level. Loci of one heavy clonotype (IGH, TRB, TRD) sharing a partner-chain clonotype (IGK / IGL, TRA, "
        "TRG) at loci with at least f = 2, 5, 20 reads, against clonotype labels permuted among the loci of the same "
        "read support (spaclone.qc.read_floor, 'above' side); the p-value is Bonferroni over the floors tested: signal "
        f"if BH q < 0.05 in the clone-level family; untestable when no floor has >= {MIN_TESTED} tested clonotypes."
    ),
    "families": (
        "Multiplicity: Benjamini-Hochberg within each level, over every testable row of that level's tests "
        "(libraries x loci x tests), never selected on outcomes. Monte-Carlo tests use B >= max(n_perm, "
        "ceil(m / 0.05) - 1) permutations for a family of m rows, so the smallest attainable p, 1 / (B + 1), is "
        "<= 0.05 / m."
    ),
    "barcode_sinks": "Information. Spots carrying >= 20 clones are reported and removed before the clone-level tests; "
    "> 1% of records on sinks is flagged.",
    "read_floor": "Information. Read floor at which same-clonotype loci share their partner chain above chance.",
}

#: The overall verdict, in this order (printed in every report).
OVERALL_RULES: tuple[str, ...] = (
    "1. Any clone-level signal -> 'Clone positions supported'.",
    (
        "2. Molecule-level signal, and the clone level adequately tested without signal (>= 2 tests with testable "
        "rows, >= half of its decisive rows testable) -> 'Cell-type placement only'."
    ),
    (
        "3. No signal at either level, >= 2 tests with testable rows and >= half of the decisive rows testable -> "
        "'No evidence of positional signal at the power of these data', with the power statement."
    ),
    (
        "4. Otherwise (a single testable test, mostly untestable or inconclusive rows, or a molecule-level signal "
        "whose clone level could not be tested adequately) -> 'Limited evidence'."
    ),
)

_TEXT = {
    "clone": (
        "Clone positions supported: clone identity follows position; clone maps and spatial clone statistics are "
        "licensed."
    ),
    "cell_type": (
        "Cell-type placement only: molecules sit on their own cell type / transcript, but clone-level placement was "
        "not detected at the power of these data (as expected under template switching between constant regions, or "
        "when too few molecules are in place); do not use clone maps until a clone-level test detects placement."
    ),
    "none": "No evidence of positional signal at the power of these data",
    "limited": "Limited evidence",
}


def _row(test: str, **kw: Any) -> dict[str, Any]:
    base = {
        "test": test,
        "level": LEVEL.get(test, "info"),
        "family": "",
        "library_id": "all",
        "locus": "",
        "n": np.nan,
        "observed": np.nan,
        "chance": np.nan,
        "ratio": np.nan,
        "p_value": np.nan,
        "q_value": np.nan,
        "n_perm": np.nan,
        "detectable": np.nan,
        "verdict": "not run",
        "note": "",
    }
    base.update(kw)
    return base


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _ratio(obs: float, chance: float) -> float:
    return obs / chance if np.isfinite(obs) and np.isfinite(chance) and chance > 0 else np.nan


def _null_col(res: TestResult, j: int) -> np.ndarray | None:
    null = getattr(res, "null", None)
    if null is None:
        return None
    null = np.asarray(null, dtype=float)
    if null.ndim == 1:
        null = null[:, None]
    if j >= null.shape[1]:
        return None
    col = null[:, j]
    col = col[np.isfinite(col)]
    return col if len(col) else None


# ------------------------------------------------------------------------------------------------ row builders
# Each builder returns (row, aux) pairs; aux holds what the detectable-effect computation needs: ("mc", null draws)
# for Monte-Carlo tests, ("normal", expected, sd) for exact ones, plus "deficit" and "k" (Bonferroni factor).


def _same_spot_rows(res: TestResult, library_id: str = "all") -> list[tuple[dict, dict]]:
    t = res.table
    if not len(t):
        return []
    zone_used = None
    if "zone" in t.columns:
        z = t.zone.astype(str)
        tissue = z.str.lower().isin(["tissue", "core", "tissue core", "in"])
        if tissue.any():
            t, zone_used = t[tissue], set(z[tissue])
        elif (z == "all").any():
            t, zone_used = t[z == "all"], {"all"}
    rec = getattr(res, "records", None)
    med_chance: dict[str, float] = {}
    if isinstance(rec, pd.DataFrame) and {"chance", "tested", "occupied"} <= set(rec.columns):
        use = rec["tested"].to_numpy(bool) & rec["occupied"].to_numpy(bool) & np.isfinite(rec["chance"].to_numpy())
        if bool(res.params.get("singleton_only", True)) and "singleton" in rec.columns:
            use &= rec["singleton"].to_numpy(bool)
        if zone_used is not None and "zone" in rec.columns:
            use &= rec["zone"].astype(str).isin(zone_used).to_numpy()
        r = rec[use]
        med_chance = r.groupby(S.LOCUS)["chance"].median().to_dict() if len(r) else {}
    out = []
    for r in t.itertuples(index=False):
        locus = str(getattr(r, S.LOCUS, getattr(r, "locus", "")))
        n = int(_f(getattr(r, "n_spots", 0)) or 0)
        p = _f(getattr(r, "p_value", np.nan))
        hits, exp = _f(getattr(r, "hits", getattr(r, "statistic", np.nan))), _f(getattr(r, "expected", np.nan))
        hi = _f(getattr(r, "rho_pdet_hi", np.nan))
        deficit = bool(getattr(r, "deficit", False))
        if not bool(getattr(r, "bound_reliable", True)):  # same_spot says its interval cannot be trusted here
            hi = np.nan
        mc = _f(med_chance.get(locus, np.nan))
        row = _row(
            "same_spot", library_id=library_id, locus=locus, n=n, observed=hits, chance=exp, ratio=_ratio(hits, exp)
        )
        row["p_value"] = p
        aux = {"kind": "normal", "expected": exp, "sd": _f(getattr(r, "null_sd", np.nan)), "deficit": deficit}
        if n < MIN_SPOTS:
            row.update(verdict="untestable", note=f"{n} singleton spots (< {MIN_SPOTS})")
        elif np.isfinite(mc) and mc > CEILING_CHANCE:
            row.update(verdict="untestable", note=f"test at ceiling: median chance {mc:.1%} > {CEILING_CHANCE:.0%}")
        elif not np.isfinite(p):
            row.update(verdict="untestable", note="no p-value (no chance model)")
        else:
            bound_note = str(getattr(r, "bound_note", "") or "")
            row.update(family="molecule", verdict="tested", note=bound_note if bound_note != "nan" else "")
            row["_bound"] = hi
        out.append((row, aux))
    return out


def _own_type_rows(res: TestResult, receptor: str) -> list[tuple[dict, dict]]:
    t = res.table.reset_index(drop=True)
    if not len(t):
        return []
    if "min_reads" in t.columns and t["min_reads"].notna().any():
        # the lowest read floor (all records) is the pre-stated row
        t = t.assign(tbl_pos=np.arange(len(t))).sort_values("min_reads", kind="stable", na_position="first")
    else:
        t = t.assign(tbl_pos=np.arange(len(t)))
    n_perm = _f(res.params.get("n_perm", res.params.get("n_place", np.nan)))
    out = []
    for lib, g in t.groupby("library_id", sort=False):
        r = g.iloc[0]
        n, p = int(_f(r.get("n", 0)) or 0), _f(r.get("p_value"))
        obs, mu = _f(r.get("statistic")), _f(r.get("null_mean"))
        row = _row(
            "own_type_share",
            library_id=str(lib),
            locus=receptor,
            n=n,
            observed=obs,
            chance=mu,
            ratio=_ratio(obs, mu),
            p_value=p,
            n_perm=_f(r.get("n_perm", n_perm)),
        )
        aux = {"kind": "mc", "null": _null_col(res, int(r["tbl_pos"]))}
        if n < MIN_RECORDS or not np.isfinite(p):
            row.update(verdict="untestable", note=f"{n} records on typed units (< {MIN_RECORDS})")
        else:
            w = res.params.get("weights")
            how = "cDNA-weighted" if w is not None and w is not False else "uniform"
            unit = "spot" if res.params.get("by_spot", True) else "record"
            row.update(family="molecule", verdict="tested", note=f"{how} {unit} placement")
        out.append((row, aux))
    return out


def _umi_rows(res: TestResult) -> list[tuple[dict, dict]]:
    t = res.table
    if not len(t) or "outcome" not in t.columns:
        return []
    out = []
    for r in t[t.outcome.astype(str) == "own_constant"].itertuples(index=False):
        n, p = int(_f(getattr(r, "n_keys", 0)) or 0), _f(getattr(r, "p_value", np.nan))
        obs, exp = _f(getattr(r, "statistic", np.nan)), _f(getattr(r, "expected", np.nan))
        row = _row("umi_match", locus=str(getattr(r, "locus", "")), n=n, observed=obs, chance=exp, p_value=p)
        row["ratio"] = _ratio(obs, exp)
        aux = {"kind": "normal", "expected": exp, "sd": _f(getattr(r, "null_sd", np.nan))}
        if n < MIN_KEYS or not np.isfinite(p):
            row.update(verdict="untestable", note=f"{n} receptor keys (< {MIN_KEYS})")
        else:
            row.update(family="molecule", verdict="tested", note="own constant gene vs decoy UMIs")
        out.append((row, aux))
    return out


def _coloc_rows(res: TestResult, depth_controlled: bool, depth_note: str = "") -> list[tuple[dict, dict]]:
    t = res.table.reset_index(drop=True).assign(tbl_pos=lambda d: np.arange(len(d)))
    if not len(t):
        return []
    designs = set(t.design.astype(str)) if "design" in t.columns else {"whole_sample"}
    stratified = depth_controlled and "stratified" in designs
    design = "stratified" if stratified else "whole_sample"
    if "design" in t.columns:
        t = t[t.design.astype(str) == design]
    t = t[(t.lo_um == 20) & (t.hi_um == 100)]
    units = str(res.params.get("units", "spots"))
    out = []
    for r in t.itertuples(index=False):
        locus = str(r.locus)
        n_cl = int(_f(getattr(r, "n_clones_in_window", 0)) or 0)
        p, obs, mu = _f(r.p_value), _f(r.statistic), _f(r.null_mean)
        n_units = _f(getattr(r, "n_loci", getattr(r, "n_units", np.nan)))
        row = _row(
            "colocalisation",
            library_id=str(r.library_id),
            locus=locus,
            n=n_units,
            observed=obs,
            chance=mu,
            ratio=_ratio(obs, mu),
            p_value=p,
            n_perm=_f(getattr(r, "n_perm", np.nan)),
        )
        aux = {"kind": "mc", "null": _null_col(res, int(r.tbl_pos))}
        top = _f(getattr(r, "top_clone_share", np.nan))
        where = f"{design.replace('_', '-')} null, {units} units, {n_cl} clones" + (
            f", top clone {top:.0%} of pairs" if np.isfinite(top) else ""
        )
        if locus not in CLONE_MARKER_LOCI:
            row.update(
                verdict="info", note=f"light chain, not a clone marker (shared CDR3s): information only; {where}"
            )
        elif n_cl < MIN_CLONES or not np.isfinite(p):
            row.update(verdict="untestable", note=f"{n_cl} clones with pairs in the band (< {MIN_CLONES})")
        elif not stratified:
            row.update(
                verdict="inconclusive",
                note="whole-sample null: depth not controlled" + (f" ({depth_note})" if depth_note else ""),
            )
        else:
            row.update(family="clone", verdict="tested", note=where)
        out.append((row, aux))
    return out


def _pairing_rows(res: TestResult) -> list[tuple[dict, dict]]:
    t = res.table.reset_index(drop=True).assign(tbl_pos=lambda d: np.arange(len(d)))
    min_units = res.params.get("min_units", "")
    out = []
    for r in t.itertuples(index=False):
        p, obs, mu = _f(r.p_value), _f(r.statistic), _f(r.null_mean)
        row = _row(
            "pairing",
            library_id=str(r.library_id),
            locus=f"{r.heavy_locus}/{r.light_locus}",
            n=_f(getattr(r, "n_heavy_units", np.nan)),
            observed=obs,
            chance=mu,
            ratio=_ratio(obs, mu),
            p_value=p,
            n_perm=_f(getattr(r, "n_perm", np.nan)),
        )
        aux = {"kind": "mc", "null": _null_col(res, int(r.tbl_pos))}
        if not (np.isfinite(p) and np.isfinite(obs)):
            nh, nl = int(_f(getattr(r, "n_heavy", 0)) or 0), int(_f(getattr(r, "n_light", 0)) or 0)
            row.update(
                verdict="untestable",
                note=f"{nh} heavy / {nl} light clones with >= {min_units} units (two of each needed)",
            )
        else:
            row.update(family="clone", verdict="tested", note=f"{getattr(r, 'design', '')} null")
        out.append((row, aux))
    return out


def _partner_rows(res: TestResult) -> list[tuple[dict, dict]]:
    """Partner-chain sharing: Bonferroni over every floor of the "above" side (a pre-stated omnibus)."""
    t = res.table.reset_index(drop=True).assign(tbl_pos=lambda d: np.arange(len(d)))
    locus = f"{res.params.get('locus', '')}>{res.params.get('partner_locus', '')}"
    above = t[(t.side.astype(str) == "above")] if "side" in t.columns else t
    ok = above[(above.n_tested.fillna(0) >= MIN_TESTED) & above.p_value.notna()]
    row = _row("partner_sharing", locus=locus, n_perm=_f(res.params.get("n_perm", np.nan)))
    if ok.empty:
        best = int(above.n_tested.fillna(0).max()) if len(above) else 0
        row.update(verdict="untestable", note=f"no read floor with >= {MIN_TESTED} tested clonotypes (most: {best})")
        return [(row, {"kind": None})]
    k = max(int(above.floor.nunique()) if "floor" in above.columns else len(above), 1)  # every floor attempted
    b = ok.loc[ok.p_value.idxmin()]
    obs, mu = _f(b.statistic), _f(b.null_mean)
    row.update(
        family="clone",
        verdict="tested",
        n=_f(b.n_tested),
        observed=obs,
        chance=mu,
        ratio=_ratio(obs, mu),
        p_value=min(1.0, k * _f(b.p_value)),
        note=f"Bonferroni over {k} read floors ({len(ok)} testable); smallest p at >= {_f(b.floor):g} reads",
    )
    return [(row, {"kind": "mc", "null": _null_col(res, int(b["tbl_pos"])), "k": k})]


def _sink_rows(res: Any) -> list[tuple[dict, dict]]:
    items = res.items() if isinstance(res, Mapping) else [("all", res)]
    n_sinks, n_rec, n_on, max_cl = 0, 0, 0, 0
    for _, r in items:
        s = r.summary
        n_sinks += int(s.get("n_sinks", 0))
        mask = getattr(r, "mask", None)
        if mask is not None:
            n_rec += len(mask)
            n_on += int(np.asarray(mask).sum())
        max_cl = max(max_cl, int(s.get("max_clones", 0) or 0))
    share = n_on / n_rec if n_rec else 0.0
    note = f"{n_sinks} sinks, max {max_cl} clones; {share:.2%} of records" + (" (flag: > 1%)" if share > 0.01 else "")
    return [(_row("barcode_sinks", n=n_sinks, observed=share, verdict="info", note=note), {"kind": None})]


def _floor_rows(res: TestResult) -> list[tuple[dict, dict]]:
    sides = set(res.table.side.astype(str)) if "side" in res.table.columns else set()
    if not {"above", "below"} <= sides:  # a floor is suggested only when both sides were tested
        return []
    fl = _f(res.summary.get("suggested_floor"))
    locus = f"{res.params.get('locus', '')}>{res.params.get('partner_locus', '')}"
    note = f"suggested read floor >= {fl:g}" if np.isfinite(fl) else "no floor suggested"
    return [(_row("read_floor", locus=locus, observed=fl, verdict="info", note=note), {"kind": None})]


# ------------------------------------------------------------------------------------------------ evaluation


def _detectable(aux: dict, chance: float, alpha_row: float) -> float:
    """Observed / chance a single row needed for BH q < alpha in its family (``alpha_row`` = alpha / m)."""
    if not (np.isfinite(chance) and chance > 0 and 0 < alpha_row < 1):
        return np.nan
    kind = aux.get("kind")
    if kind == "mc" and aux.get("null") is not None:
        a = alpha_row / aux.get("k", 1)
        null = aux["null"]
        if 1.0 / (len(null) + 1) > a:
            return np.nan  # not reachable with these permutations
        return float(np.quantile(null, 1 - a, method="higher")) / chance
    if kind == "normal":
        sd = aux.get("sd", np.nan)
        if np.isfinite(sd) and sd >= 0:
            return 1 + float(stats.norm.isf(alpha_row)) * sd / chance
    return np.nan


def _names(rows: pd.DataFrame, max_items: int = 6) -> str:
    parts = []
    for test, g in rows.groupby("test", sort=False):
        where = g.locus.astype(str) + np.where(g.library_id != "all", " " + g.library_id, "")
        loci = list(dict.fromkeys(where))
        shown = ", ".join(loci[:max_items]) + (f", +{len(loci) - max_items} more" if len(loci) > max_items else "")
        parts.append(f"{test} ({shown})" if any(loci) else test)
    return "; ".join(parts)


def _adequate(rows: pd.DataFrame) -> bool:
    """At least two tests with testable rows, and at least half of the decisive rows that ran are testable."""
    ok = rows.verdict.isin(_TESTED)
    return rows.loc[ok, "test"].nunique() >= 2 and ok.sum() >= MIN_TESTABLE_SHARE * len(rows)


def _level_status(t: pd.DataFrame, level: str) -> str:
    rows = t[(t.level == level) & t.test.isin(DECISIVE) & t.verdict.isin(_RUN)]
    if rows.empty:
        return "not tested"
    if (rows.verdict == "signal").any():
        return "signal"
    return "not detected" if _adequate(rows) else "limited"


def _level_text(t: pd.DataFrame, level: str, status: str, skipped: Mapping[str, str]) -> str:
    name = "Molecule level" if level == "molecule" else "Clone level"
    rows = t[(t.level == level) & t.test.isin(DECISIVE) & t.verdict.isin(_RUN)]
    tested = rows[rows.verdict.isin(_TESTED)]
    if status == "signal":
        sig = rows[rows.verdict == "signal"]
        what = (
            "molecules sit on their own transcript / cell type more often than random placement"
            if level == "molecule"
            else "clone identity follows position"
        )
        return f"{name}: signal — {what} ({_names(sig)})."
    if status == "not tested":
        why = [f"{k}: {v}" for k, v in skipped.items() if STEP_LEVEL.get(k.split(":")[0]) == level]
        return f"{name}: not tested" + (f" ({'; '.join(why)})." if why else ".")
    bad = rows[~rows.verdict.isin(_TESTED)]
    extra = f"; {len(bad)} row(s) untestable or inconclusive" if len(bad) else ""
    if status == "not detected":
        det = tested[np.isfinite(tested.detectable.to_numpy(float))]
        need = ""
        if len(det):
            med = det.groupby("test", sort=False).detectable.median()
            need = "; a single positive row would have needed observed/chance >= " + ", ".join(
                f"{v:.3g} ({k})" for k, v in med.items()
            )
        return (
            f"{name}: not detected — none of {len(tested)} testable row(s) in {tested.test.nunique()} tests reached "
            f"BH q < {ALPHA:g} ({_names(tested)}){need}{extra}."
        )
    if len(tested) and tested.test.nunique() < 2:
        return f"{name}: limited — only 1 test with testable rows ({_names(tested)}), no signal{extra}."
    if len(tested):
        return (
            f"{name}: limited — only {len(tested)} of {len(rows)} decisive rows testable ({_names(tested)}), no "
            f"signal{extra}."
        )
    return f"{name}: limited — no row could be tested ({len(bad)} row(s) untestable or inconclusive)."


def _power_text(t: pd.DataFrame, fam: Mapping[str, int]) -> str:
    tested = t[t.verdict.isin(_TESTED) & (t.family != "")]
    if tested.empty:
        return "Power: no decisive row could be tested."
    parts = [
        f"Power: {len(tested)} testable row(s) in {tested.test.nunique()} tests "
        f"({', '.join(f'{k} {v}' for k, v in tested.test.value_counts(sort=False).items())}); "
        + ", ".join(f"{lv}-level BH family m = {m}" for lv, m in fam.items() if m)
        + "."
    ]
    perm = tested[tested.n_perm.notna()]
    if len(perm):
        b = int(perm.n_perm.min())
        mmax = max(fam.values()) if fam else 1
        parts.append(
            f"Monte-Carlo tests used B >= {b} permutations (smallest p {1 / (b + 1):.2g}; alpha/m >= {ALPHA / mmax:.2g})."
        )
    det = tested[np.isfinite(tested.detectable.to_numpy(float))]
    if len(det):
        need = det.groupby("test", sort=False).detectable.median()
        parts.append(
            "A single positive row would have needed observed/chance >= "
            + ", ".join(f"{v:.3g} ({k})" for k, v in need.items())
            + " (median over rows)."
        )
    if "_bound" in t.columns:
        b = t[(t.test == "same_spot") & (t.verdict == "not detected") & np.isfinite(t["_bound"].to_numpy(float))]
        if len(b):
            parts.append(
                "Same-spot 95% upper bounds of located fraction x detection probability: "
                + ", ".join(f"{lc} {v:.2%}" for lc, v in zip(b.locus, b["_bound"], strict=True))
                + "."
            )
    bad = t[t.test.isin(DECISIVE) & t.verdict.isin(["untestable", "inconclusive"])]
    if len(bad):
        parts.append(f"{len(bad)} decisive row(s) were untestable or inconclusive (see the table).")
    return " ".join(parts)


def evaluate_trust(
    results: Mapping[str, Any],
    *,
    depth_controlled: bool | None = None,
    depth_note: str = "",
    stratified: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply :data:`CRITERIA` and :data:`OVERALL_RULES` to the results of :func:`spaclone.run`.

    Parameters
    ----------
    results
        Step results keyed as :func:`spaclone.run` stores them (``"same_spot"`` — one result or a dict per library —,
        ``"own_type_share_BCR"`` / ``"_TCR"``, ``"umi_match"``, ``"colocalisation"`` and ``"colocalisation_light"``,
        ``"pairing_BCR"`` / ``"_TCR"`` / ``"_TCRgd"``, ``"read_floor_<focal>_<partner>"``, ``"barcode_sinks"``) and
        ``"_skipped"``: step -> reason.
    depth_controlled
        Whether co-localisation was run with depth (Ig RNA) strata; without them it is not decisive.
    depth_note
        Why depth is not controlled (printed in the rows).
    stratified
        Old name of ``depth_controlled``.

    Returns
    -------
    tuple
        The verdict table (one row per test x library x locus) and a dict with ``verdict``, ``verdict_molecule``,
        ``verdict_clone``, ``power`` (sentences) and ``overall``, ``molecule``, ``clone`` (status codes).
    """
    if depth_controlled is None:
        depth_controlled = True if stratified is None else bool(stratified)
    skipped = dict(results.get("_skipped", {}))
    pairs: list[tuple[dict, dict]] = []
    ss = results.get("same_spot")
    if isinstance(ss, Mapping):
        for lib, r in ss.items():
            pairs += _same_spot_rows(r, str(lib))
    elif ss is not None:
        pairs += _same_spot_rows(ss)
    for key in ("own_type_share_BCR", "own_type_share_TCR"):
        if key in results:
            pairs += _own_type_rows(results[key], key.rsplit("_", 1)[1])
    if "umi_match" in results:
        pairs += _umi_rows(results["umi_match"])
    for key in ("colocalisation", "colocalisation_light"):
        if key in results:
            pairs += _coloc_rows(results[key], bool(depth_controlled), depth_note)
    for key in sorted(k for k in results if k == "pairing" or k.startswith("pairing_")):
        pairs += _pairing_rows(results[key])
    floors = sorted(k for k in results if k.startswith("read_floor"))
    for key in floors:
        pairs += _partner_rows(results[key])
    if "barcode_sinks" in results:
        pairs += _sink_rows(results["barcode_sinks"])
    for key in floors:
        pairs += _floor_rows(results[key])
    for key, msg in skipped.items():
        test = key.split(":")[0]
        pairs.append((_row(test, locus=key[len(test) + 1 :], verdict="not run", note=msg), {"kind": None}))
    rows = [r for r, _ in pairs]
    aux = [a for _, a in pairs]
    table = pd.DataFrame(rows, columns=list(_row("x"))) if not rows else pd.DataFrame(rows)
    if "_bound" not in table.columns:
        table["_bound"] = np.nan
    fam: dict[str, int] = {}
    for level in ("molecule", "clone"):
        m = (table.family == level).to_numpy() & np.isfinite(table.p_value.to_numpy(float))
        fam[level] = int(m.sum())
        if not m.any():
            continue
        q = bh(table.loc[m, "p_value"].to_numpy(float))
        table.loc[m, "q_value"] = q
        for i in np.flatnonzero(m):
            r = table.iloc[i]
            up = np.isfinite(r.observed) and np.isfinite(r.chance) and r.observed > r.chance
            if r.q_value < ALPHA and up:
                verdict, note = "signal", r.note
                if r.test == "same_spot" and np.isfinite(r["_bound"]):
                    note = f"located x p_det up to {r['_bound']:.2%}"
            elif aux[i].get("deficit"):
                verdict = "inconclusive"
                note = "fewer hits than chance: the chance model over-states chance here (see qc.shift_test)"
            else:
                verdict = "not detected"
                note = r.note + (f" (p < {ALPHA:g} but BH q >= {ALPHA:g})" if r.p_value < ALPHA and up else "")
                if r.test == "same_spot" and np.isfinite(r["_bound"]):
                    note = f"located x p_det <= {r['_bound']:.2%} (95% bound)"
                elif r.test == "same_spot":
                    note = "no reliable bound" + (f": {r.note}" if r.note else "")
            table.iat[i, table.columns.get_loc("verdict")] = verdict
            table.iat[i, table.columns.get_loc("note")] = note
            table.iat[i, table.columns.get_loc("detectable")] = _detectable(aux[i], r.chance, ALPHA / fam[level])
    table.loc[table.verdict == "tested", "verdict"] = "not detected"  # pragma: no cover - defensive
    st_m, st_c = _level_status(table, "molecule"), _level_status(table, "clone")
    v_m, v_c = _level_text(table, "molecule", st_m, skipped), _level_text(table, "clone", st_c, skipped)
    power = _power_text(table, fam)
    run = table[table.test.isin(DECISIVE) & table.verdict.isin(_RUN)]
    n_tests = run.loc[run.verdict.isin(_TESTED), "test"].nunique()
    if st_c == "signal":
        overall, verdict = "clone_positions", f"{_TEXT['clone']} {v_m}"
    elif st_m == "signal" and st_c == "not detected":
        overall, verdict = "cell_type_only", f"{_TEXT['cell_type']} {v_c}"
    elif st_m != "signal" and _adequate(run):
        overall = "no_evidence"
        verdict = (
            f"{_TEXT['none']}: no row reached BH q < {ALPHA:g} (absence of evidence, not evidence of absence). {power}"
        )
    else:
        overall = "limited"
        if st_m == "signal":
            why = (
                "molecules sit on their own cell type / transcript, but clone-level placement could not be tested "
                "adequately, so clone identity has not been shown to follow position; do not use clone maps."
            )
            verdict = f"{_TEXT['limited']}: {why} {v_c}"
        elif run.empty:
            verdict = f"{_TEXT['limited']}: no decisive test could be run (see the table)."
        else:
            n_ok = int(run.verdict.isin(_TESTED).sum())
            if n_tests == 1:
                why = "only 1 test with testable rows"
            elif n_tests == 0:
                why = "no decisive row could be tested"
            else:
                why = f"only {n_ok} of {len(run)} decisive rows were testable (the rest untestable or inconclusive)"
            verdict = f"{_TEXT['limited']}: {why}; no row shows a signal and clone maps are not licensed. {power}"
    table = table.drop(columns=["_bound"])
    out = {
        "verdict": verdict,
        "verdict_molecule": v_m,
        "verdict_clone": v_c,
        "power": power,
        "overall": overall,
        "molecule": st_m,
        "clone": st_c,
    }
    return table, out


@dataclass(frozen=True, repr=False)
class TrustReport(TestResult):
    """Result of :func:`spaclone.run`.

    Attributes
    ----------
    table
        One row per test x library x locus: ``level`` (molecule / clone / info), ``family`` (the BH family the row
        belongs to, empty if none), observed, chance, ratio, p, q, ``n_perm``, ``detectable`` (observed / chance a
        single row needed for BH q < 0.05), the verdict and a note.
    verdict
        The overall verdict in one or a few sentences (rules in :data:`OVERALL_RULES`).
    verdict_molecule, verdict_clone
        The verdict of each level ("signal", "not detected", "limited" or "not tested", with the rows it rests on).
    power
        What the decisive tests could have detected (testable rows, permutations, detectable effects, bounds).
    results
        Every underlying result object, keyed by step.
    qc
        Receptor QC metrics before filtering.
    filtering
        Step-by-step removal report of the filter.
    """

    verdict: str = ""
    verdict_molecule: str = ""
    verdict_clone: str = ""
    power: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    qc: pd.DataFrame | None = None
    filtering: pd.DataFrame | None = None

    @property
    def status(self) -> dict[str, str]:
        """Status codes of the three verdicts.

        ``overall``: clone_positions / cell_type_only / no_evidence / limited; ``molecule`` and ``clone``: signal /
        not detected / limited / not tested.
        """
        return {k: str(self.summary.get(k, "")) for k in ("overall", "molecule", "clone")}

    @property
    def clone_positions_supported(self) -> bool:
        """True when the clone-level verdict licenses clone maps and spatial clone statistics."""
        return self.summary.get("clone") == "signal"

    def __repr__(self) -> str:
        counts = self.table.verdict.value_counts().to_dict() if len(self.table) else {}
        st = self.status
        return f"TrustReport({st['overall']} | molecule: {st['molecule']}, clone: {st['clone']} | {counts})"

    def tests(self) -> pd.DataFrame:
        """The verdict table: molecule-level tests, clone-level tests, then information and steps not run."""
        order = {t: i for i, t in enumerate(_ORDER)}
        key = self.table.test.map(order).fillna(99) + np.where(self.table.verdict == "not run", 100, 0)
        return self.table.iloc[np.argsort(key.to_numpy(), kind="stable")].reset_index(drop=True)

    def plot(self, **kwargs: Any):
        """One-page figure of the report (see :func:`spaclone.pl.trust_report`)."""
        from spaclone.pl.report import trust_report

        return trust_report(self, **kwargs)

    def methods(self) -> str:
        """A methods paragraph describing what was run, with parameters and references."""
        p = self.params
        parts = [
            f"Location trust was assessed with spaclone {p.get('version', '')} on {p.get('n_records_in', 0):,} "
            f"receptor records ({p.get('n_records', 0):,} after filtering: valid IMGT junction, productive"
            + (f", >= {p['min_reads']} reads" if p.get("min_reads") else "")
            + (", on tissue" if p.get("on_tissue") else "")
            + "). Molecule-level placement (is a molecule on its own transcript / cell type?) and clone-level "
            "placement (does clone identity follow position?) were judged separately."
        ]
        r = self.results
        if "same_spot" in r:
            parts.append(
                "Molecule-level concordance with the GEX library was tested on spots carrying a single clone of the "
                "chain, against a per-spot chance model from the local transcript composition, and the located "
                "fraction was bounded with a Rogan-Gladen-type estimator (Rogan & Gladen 1978)."
            )
        if any(k.startswith("own_type_share") for k in r):
            parts.append(
                "The share of records on their own cell type was compared with random placement of whole spots on "
                "typed units of the same library, drawn with probability proportional to captured cDNA, after "
                "barcode sinks were removed (Gel et al. 2016)."
            )
        if "umi_match" in r:
            parts.append(
                "Receptor (barcode, UMI) keys were looked up in the GEX reads beside decoy UMIs of the same barcode."
            )
        if "colocalisation" in r:
            parts.append(
                "Same-clone co-localisation of heavy chains (pairs 20-100 um apart; "
                + ("loci of same-clone spots within 15 um as units" if p.get("subcellular") else "spot units")
                + ") was compared with permutations of clone labels"
                + (" within strata of local Ig RNA" if p.get("depth_controlled") else " over the whole sample")
                + "; light chains were reported but not used for the verdict (Baddeley et al. 2015)."
            )
        if any(k.startswith("pairing") for k in r):
            parts.append(
                "Heavy/light co-occurrence in 10-um bins (BCR and TCR) was compared with heavy labels shuffled within "
                "clones-per-spot strata, a chimera-aware null (Howie et al. 2015)."
            )
        if any(k.startswith("read_floor") for k in r):
            parts.append(
                "Partner-chain sharing of the loci of one heavy clonotype was compared with clonotype labels permuted "
                "among loci of the same read support, Bonferroni over read floors (Kebschull & Zador 2015)."
            )
        parts.append(
            "Monte-Carlo p-values are (1 + b)/(B + 1) (Davison & Hinkley 1997) with B chosen so that 1/(B + 1) <= "
            "0.05/m for the m rows of a family; Benjamini-Hochberg q-values were computed within each level "
            "(Benjamini & Hochberg 1995). Criteria: "
            + " ".join(f"{k}: {v}" for k, v in CRITERIA.items() if k in DECISIVE or k == "families")
            + " Overall: "
            + " ".join(OVERALL_RULES)
        )
        return " ".join(parts)

    def to_markdown(self, path: str | Path | None = None) -> str:
        """The report as Markdown (written to ``path`` when given): three verdicts, power, tests, criteria, methods."""
        cols = [
            "test",
            "level",
            "library_id",
            "locus",
            "n",
            "observed",
            "chance",
            "ratio",
            "p_value",
            "q_value",
            "n_perm",
            "detectable",
            "verdict",
            "note",
        ]
        t = self.tests()[cols].copy()
        for c in ("observed", "chance", "ratio", "p_value", "q_value", "detectable"):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
        for c in ("n", "n_perm"):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = "".join(
            "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |\n" for row in t.itertuples(index=False)
        )
        notes = list(self.params.get("notes", []) or [])
        text = (
            "# spaclone location-trust report\n\n"
            f"**Verdict.** {self.verdict}\n\n"
            f"**Molecule-level placement.** {self.verdict_molecule}\n\n"
            f"**Clone-level placement.** {self.verdict_clone}\n\n"
            f"**Power.** {self.power}\n\n"
            + ("## Input checks\n\n" + "".join(f"- {n}\n" for n in notes) + "\n" if notes else "")
            + f"## Tests\n\n{header}{body}\n"
            "## Criteria (fixed before looking at the data)\n\n"
            + "".join(f"- **{k}**: {v}\n" for k, v in CRITERIA.items())
            + "\n**Overall verdict**, in this order:\n\n"
            + "".join(f"{r}\n" for r in OVERALL_RULES)
            + f"\n## Methods\n\n{self.methods()}\n"
        )
        if self.filtering is not None and _has_tabulate():
            text += "\n## Filtering\n\n" + self.filtering.to_markdown(index=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True
