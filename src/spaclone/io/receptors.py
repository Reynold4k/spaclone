"""Read immune-receptor tables of any mainstream tool into one AIRR-named record table.

:func:`read_receptors` identifies the format from documented header signatures (:mod:`spaclone.io.formats`), maps the
tool's columns to AIRR names (``cell_id``, ``locus``, ``v_call`` ... ``junction_aa``, ``productive``,
``consensus_count`` = reads, ``umi_count`` = UMIs, ``templates``, ``cell_count``, ``clone_id``), and records every
decision with its evidence in ``table.attrs["spaclone"]["io"]``. Nothing is guessed:

* **format**: exactly one signature must match, else the error lists the candidates (``format=``);
* **columns**: the format's synonym table, overridden by ``columns={canonical: source}``;
* **counts**: each tool's documented meaning (reads, UMIs, templates, cells); a bare unlabelled count must be named
  with ``count_type=``;
* **junction vs CDR3**: tested on the data with the conserved anchors (C ... W/F): mostly anchored = junction, almost
  never = CDR3 without anchors (stored as ``cdr3_aa`` / ``cdr3``), a mix = error (``junction=``);
* **locus**: the locus / chain column, else the first three letters of the V, J or C call;
* **productive**: T/F, True/False, 1/0, yes/no, In/Out/Stop; ``None`` / empty = unknown.

References
----------
See :mod:`spaclone.io.formats` for every format's documentation.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.io._table import peek_table, read_text_table
from spaclone.io.airr import clean_call, locus_from_call, map_unique
from spaclone.io.formats import CANONICAL_FIELDS, COUNT_TYPES, RECEPTOR_FORMATS, ReceptorFormat, detect_format

__all__ = ["ANCHOR_THRESHOLDS", "anchored_fraction", "read_receptors", "receptor_format"]

logger = logging.getLogger("spaclone")

#: Junction decision on the share of amino-acid values that start with C and end with W or F: at least the first
#: value means junction, at most the second means CDR3 without the anchors; in between the reader asks.
ANCHOR_THRESHOLDS: tuple[float, float] = (0.8, 0.2)
_MISSING = frozenset(("", "None", "none", "NA", "N/A", "na", "nan", "NaN", "null", "NULL", "*", ".", "-"))
_TRUE = frozenset(("t", "true", "1", "yes", "y", "in", "functional", "func", "productive"))
_FALSE = frozenset(("f", "false", "0", "no", "n", "out", "stop", "nonfunctional", "unproductive"))
_LOCUS_NAMES = {
    "TCR_ALPHA": "TRA",
    "TCR_BETA": "TRB",
    "TCR_GAMMA": "TRG",
    "TCR_DELTA": "TRD",
    "BCR_HEAVY": "IGH",
    "BCR_KAPPA": "IGK",
    "BCR_LAMBDA": "IGL",
    "TCRA": "TRA",
    "TCRB": "TRB",
    "TCRG": "TRG",
    "TCRD": "TRD",
}
_CODON = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        [(x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"],
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        strict=True,
    )
}
_ADAPTIVE = re.compile(r"^(TCR|IG)([ABGDHKL])([VDJC])(\d+)(?:-(\d+))?(\*\d+)?$")
_PATH_LIKE = (str, Path)


# ------------------------------------------------------------------------------------------------- value helpers


def _clean_text(values: Any, extra_missing: Sequence[str] = ()) -> np.ndarray:
    miss = _MISSING | set(extra_missing)

    def one(v: Any) -> Any:
        t = str(v).strip()
        return None if t in miss else t

    return map_unique(np.asarray(values, dtype=object), one)


def _flag(values: Any) -> pd.arrays.BooleanArray:
    def one(v: Any) -> Any:
        t = str(v).strip().lower()
        return True if t in _TRUE else (False if t in _FALSE else None)

    m = map_unique(np.asarray(values, dtype=object), one)
    return pd.array([None if pd.isna(v) else bool(v) for v in m], dtype="boolean")


def _translate(nt: Any) -> Any:
    if not isinstance(nt, str) or not nt or len(nt) % 3:
        return None
    s = nt.upper()
    return "".join(_CODON.get(s[i : i + 3], "X") for i in range(0, len(s), 3))


def anchored_fraction(aa: Any) -> tuple[float | None, int]:
    """Share of amino-acid CDR3 / junction values that carry both conserved anchors (start C, end W or F).

    Parameters
    ----------
    aa
        Amino-acid sequences (missing values are ignored).

    Returns
    -------
    tuple
        ``(fraction, n)``: the share among the ``n`` non-missing values (``None`` when ``n == 0``).
    """
    s = pd.Series(np.asarray(aa, dtype=object)).dropna().astype(str).str.strip()
    s = s[s.str.len() >= 3]
    if not len(s):
        return None, 0
    ok = s.str.startswith("C") & s.str[-1].isin(["W", "F"])
    return float(ok.mean()), int(len(s))


def _adaptive_gene(v: Any) -> Any:
    """Adaptive gene names (``TCRBV05-01*01``) in IMGT form (``TRBV5-1*01``); other names unchanged."""
    if not isinstance(v, str):
        return v
    m = _ADAPTIVE.match(v.strip())
    if not m:
        return v.strip()
    pre, loc, seg, fam, sub, allele = m.groups()
    name = f"{'TR' if pre == 'TCR' else 'IG'}{loc}{seg}{int(fam)}" + (f"-{int(sub)}" if sub else "")
    return name + (allele or "")


def _locus_values(values: Any) -> np.ndarray:
    def one(v: Any) -> Any:
        t = str(v).strip().upper().replace(" ", "_")
        if t in _MISSING or t in ("MULTI", "NONE", ""):
            return None
        t = _LOCUS_NAMES.get(t, t)
        return t if t in S.ALL_LOCI else None

    return map_unique(np.asarray(values, dtype=object), one)


# ------------------------------------------------------------------------------------------------- standardising


def _resolve(cols: Sequence[str], fmt: ReceptorFormat, columns: Mapping[str, str] | None) -> dict[str, str]:
    """Canonical field → source column: the user's mapping, then the format's synonyms (first present wins)."""
    have = list(cols)
    out: dict[str, str] = {}
    user = dict(columns or {})
    bad = [k for k in user if k not in CANONICAL_FIELDS]
    if bad:
        raise ValueError(
            f"columns= keys must be canonical fields ({', '.join(CANONICAL_FIELDS)}); got {bad}. The mapping is "
            "{canonical: source column}."
        )
    for k, v in user.items():
        if v not in have:
            raise ValueError(f"columns={{{k!r}: {v!r}}}: no column {v!r} in the table (columns: {have[:20]})")
        out[k] = v
    taken = set(out.values())
    for k, srcs in fmt.columns.items():
        if k in out:
            continue
        hit = [s for s in srcs if s in have and s not in taken]
        if hit:
            out[k] = hit[0]
    return out


_BY_NAME = {"consensus_count": "reads", "umi_count": "umis", "templates": "templates", "cell_count": "cells"}


def _counts(
    df: pd.DataFrame,
    fmt: ReceptorFormat,
    mapping: dict[str, str],
    count_type: str | None,
    overrides: dict[str, str],
    user: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Source count columns → canonical count fields (reads / UMIs / templates / cells), with the evidence.

    Precedence: the user's ``columns=`` mapping; the documented meaning of the writer (``overrides``, e.g. 10x AIRR
    ``duplicate_count`` = UMIs); the format's documented meaning; for AIRR ``duplicate_count`` the schema (reads, used
    only when there is no ``consensus_count``). Unlabelled counts need ``count_type``.
    """
    if count_type is not None and count_type not in COUNT_TYPES:
        raise ValueError(f"count_type must be one of {list(COUNT_TYPES)}, got {count_type!r}")
    meaning = {**dict(fmt.counts), **overrides}
    srcs: list[tuple[str, str | None]] = []
    for canon in ("consensus_count", "umi_count", "templates", "cell_count", "duplicate_count"):
        s = mapping.get(canon)
        if s is not None and s not in [x for x, _ in srcs]:
            srcs.append((s, canon))
    srcs += [(s, None) for s in meaning if s in df.columns and s not in [x for x, _ in srcs]]

    def empty(col: str) -> bool:
        v = pd.Series(df[col].to_numpy(dtype=object)).astype(str).str.strip()
        return bool(v.isin(list(_MISSING)).all() or df[col].isna().all())

    srcs = [(s, c) for s, c in srcs if not empty(s)]
    assign: dict[str, str] = {}
    note: dict[str, str] = {}
    bare: list[str] = []
    for s, canon in srcs:
        if canon is not None and user.get(canon) == s:
            m, why = (_BY_NAME.get(canon, "reads"), f"given: columns={{{canon!r}: {s!r}}}")
        elif s in meaning:
            m, why = meaning[s], ("writer" if s in overrides else fmt.name)
        else:
            m, why = (_BY_NAME.get(canon or "", "reads"), f"{fmt.name}: column name")
        if m == "unlabelled":
            bare.append(s)
            continue
        target = COUNT_TYPES[m]
        if s == "duplicate_count" and m == "reads" and (mapping.get("consensus_count") or "consensus_count" in assign):
            target, m = "duplicate_count", "reads (AIRR copy number; consensus_count is the read count)"
        if target in assign:
            continue
        assign[target] = s
        note[target] = f"{s!r} = {m} ({why})"
    have = any(k in assign for k in _BY_NAME)
    if bare and count_type is None and not have:
        raise ValueError(
            f"{fmt.name}: the count column {bare[0]!r} is unlabelled ({fmt.notes or 'its meaning depends on the run'})"
            ". Say what it counts with count_type='reads', 'umis', 'templates' or 'cells'."
        )
    if bare and count_type is not None:
        canon = COUNT_TYPES[count_type]
        assign[canon] = bare[0]
        note[canon] = f"{bare[0]!r} = {count_type} (given: count_type={count_type!r})"
    elif count_type is not None:
        note["count_type"] = f"count_type={count_type!r} not used: the count columns are labelled"
    return assign, note


def _junction_decision(aa: Any, junction: str, label: str, *, labelled: bool = False) -> tuple[str, dict[str, Any]]:
    frac, n = anchored_fraction(aa)
    info: dict[str, Any] = {"column": label, "anchored_fraction": frac, "n_tested": n, "rule": "C ... W/F anchors"}
    if junction in ("junction", "cdr3"):
        info["decision"] = f"{junction} (given: junction={junction!r})"
        return junction, info
    if frac is None:
        info["decision"] = "junction (no amino-acid values to test)"
        return "junction", info
    hi, lo = ANCHOR_THRESHOLDS
    if frac >= hi:
        info["decision"] = f"junction: {frac:.1%} of {n:,} values start with C and end with W/F"
        return "junction", info
    if frac <= lo:
        info["decision"] = (
            f"CDR3 without the conserved anchors: only {frac:.1%} of {n:,} values start with C and end with W/F; "
            "stored as cdr3_aa / cdr3 (junction rules such as pp.filter_valid cannot apply)"
        )
        return "cdr3", info
    if labelled:
        info["decision"] = (
            f"junction (labelled by the AIRR schema); note: only {frac:.1%} of {n:,} values carry both anchors"
        )
        return "junction", info
    raise ValueError(
        f"{label}: {frac:.1%} of {n:,} CDR3 values carry the conserved anchors (C ... W/F), a mix of junctions and "
        "anchor-less CDR3s. Say which it is with junction='junction' or junction='cdr3'."
    )


def _standardise(
    df: pd.DataFrame,
    fmt: ReceptorFormat,
    *,
    columns: Mapping[str, str] | None,
    count_type: str | None,
    junction: str,
    count_overrides: Mapping[str, str] | None = None,
    evidence: Sequence[str] = (),
    extra_missing: Sequence[str] = (),
) -> pd.DataFrame:
    """Source table → canonical record table with ``attrs["spaclone"]["io"]``."""
    if fmt.name == "custom" and not columns:
        raise ValueError("format='custom' needs columns={canonical: source}, e.g. {'cell_id': 'barcode', "
                         "'junction_aa': 'cdr3', 'v_call': 'v'}")  # fmt: skip
    mapping = _resolve(df.columns, fmt, columns)
    n = len(df)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    io: dict[str, Any] = {
        "format": fmt.name,
        "tool": fmt.tool,
        "evidence": list(evidence),
        "columns": {},
        "level": fmt.level,
        "docs": fmt.docs,
    }
    if fmt.notes:
        io["notes"] = fmt.notes
    cnt_assign, cnt_note = _counts(df, fmt, mapping, count_type, dict(count_overrides or {}), dict(columns or {}))
    count_fields = {"consensus_count", "umi_count", "templates", "cell_count", "duplicate_count"}
    for canon, src in mapping.items():
        if canon in count_fields:
            continue
        io["columns"][canon] = src
        v = df[src].to_numpy(dtype=object, na_value=np.nan)
        if canon in ("v_call", "d_call", "j_call", "c_call"):
            vals = clean_call(_clean_text(v, extra_missing))
            if fmt.name.startswith("immunoseq"):
                out[f"{canon}_source"] = vals
                vals = map_unique(vals, _adaptive_gene)
            out[canon] = vals
        elif canon == "productive":
            out[canon] = _flag(v)
        elif canon == "locus":
            out[canon] = _locus_values(v)
        elif canon in ("x", "y"):
            out[canon] = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)
        else:
            out[canon] = _clean_text(v, extra_missing)
    for canon, src in cnt_assign.items():
        out[canon] = pd.to_numeric(pd.Series(df[src].to_numpy(dtype=object)).replace(list(_MISSING), np.nan),
                                   errors="coerce").to_numpy(dtype=float)  # fmt: skip
        io["columns"][canon] = src
    io["counts"] = cnt_note
    # junction or CDR3 (anchors tested on the data)
    if "junction_aa" in out:
        labelled = mapping["junction_aa"] == "junction_aa"
        decision, info = _junction_decision(out["junction_aa"], junction, mapping["junction_aa"], labelled=labelled)
        if decision == "cdr3":
            out["cdr3_aa"] = out.pop("junction_aa")
            if "junction" in out:
                out["cdr3"] = out.pop("junction")
        io["junction"] = info
    elif "junction" in out:
        out["junction_aa"] = map_unique(out["junction"], _translate)
        decision, info = _junction_decision(out["junction_aa"], junction, "translated " + mapping["junction"])
        info["derived"] = "junction_aa translated from the junction nucleotides (standard genetic code)"
        io["junction"] = info
        if decision == "cdr3":
            out["cdr3_aa"] = out.pop("junction_aa")
            out["cdr3"] = out.pop("junction")
    elif "cdr3_aa" in out:
        decision, info = _junction_decision(out["cdr3_aa"], junction, mapping["cdr3_aa"])
        if decision == "junction":
            out["junction_aa"] = out.pop("cdr3_aa")
            if "cdr3" in out:
                out["junction"] = out.pop("cdr3")
            info["decision"] += f" (column {mapping['cdr3_aa']!r} holds anchored sequences: junctions)"
        io["junction"] = info
    # locus
    if "locus" in out and not pd.isna(out["locus"]).all():
        io["locus"] = f"column {mapping['locus']!r}"
        miss = pd.isna(out["locus"].to_numpy(dtype=object))
    else:
        out["locus"] = np.full(n, np.nan, dtype=object)
        miss = np.ones(n, dtype=bool)
        io["locus"] = "gene calls (V, then J, then C)"
    if miss.any():
        loc = out["locus"].to_numpy(dtype=object).copy()
        for c in ("v_call", "j_call", "c_call"):
            if c in out and miss.any():
                fill = locus_from_call(out[c].to_numpy(dtype=object)[miss])
                loc[np.flatnonzero(miss)] = np.where(pd.isna(fill), loc[miss], fill)
                miss = pd.isna(loc)
        out["locus"] = loc
        if "column" in io["locus"] and (~pd.isna(loc)).sum() > 0:
            io["locus"] += "; missing values from the V / J / C calls"
    io["n_records_without_locus"] = int(pd.isna(out["locus"]).sum())
    if "productive" in out:
        io["productive"] = f"column {mapping['productive']!r} (T/F, True/False, 1/0, In/Out/Stop; others unknown)"
    # remaining source columns keep their names (a name that is also a canonical field gets a suffix)
    used = set(mapping.values()) | set(cnt_assign.values())
    for c in df.columns:
        if c in used or c == "" or str(c).startswith("_col"):
            continue
        name = f"{c}_src" if c in CANONICAL_FIELDS or c in out.columns else c
        out[name] = df[c].to_numpy(dtype=object, na_value=np.nan)
    order = [c for c in CANONICAL_FIELDS if c in out.columns]
    out = out[order + [c for c in out.columns if c not in order]]
    io["n_records"] = int(n)
    io["cell_ids"] = bool("cell_id" in out.columns and fmt.cell_ids)
    out.attrs["spaclone"] = {"io": io}
    return out


# ------------------------------------------------------------------------------------------------- special layouts


def _read_10x_json(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of contigs (10x all_contig_annotations.json)")
    rows = []
    region = {"L-REGION+V-REGION": "v_gene", "V-REGION": "v_gene", "D-REGION": "d_gene", "J-REGION": "j_gene",
              "C-REGION": "c_gene"}  # fmt: skip
    for c in data:
        r = {k: c.get(k) for k in ("barcode", "contig_name", "sequence", "cdr3", "cdr3_seq", "productive", "is_cell",
                                   "high_confidence", "full_length", "read_count", "umi_count", "filtered")}  # fmt: skip
        chains = []
        for a in c.get("annotations") or []:
            feat = a.get("feature") or {}
            key = region.get(feat.get("region_type"))
            if key and r.get(key) is None:
                r[key] = feat.get("gene_name")
            if feat.get("chain"):
                chains.append(feat["chain"])
        uniq = sorted(set(chains))
        r["chain"] = uniq[0] if len(uniq) == 1 else ("Multi" if uniq else None)
        info = c.get("info") or {}
        r["raw_clonotype_id"] = info.get("raw_clonotype_id")
        r["raw_consensus_id"] = info.get("raw_consensus_id")
        rows.append(r)
    out = pd.DataFrame(rows)
    for k in ("productive", "is_cell", "high_confidence", "full_length", "filtered"):
        if k in out:
            out[k] = out[k].map(lambda v: "None" if v is None else str(bool(v)))
    return out.astype(object).where(out.notna(), None)


def _melt_trust4_barcode(df: pd.DataFrame) -> pd.DataFrame:
    fields = ["V", "D", "J", "C", "cdr3_nt", "cdr3_aa", "read_cnt", "consensus_id", "CDR3_germline_similarity",
              "consensus_full_length"]  # fmt: skip
    rows = []
    for col, rank in (("chain1", "primary"), ("chain2", "primary"), ("secondary_chain1", "secondary"),
                      ("secondary_chain2", "secondary")):  # fmt: skip
        if col not in df.columns:
            continue
        for bc, ct, val in zip(df["barcode"], df["cell_type"], df[col], strict=True):
            if not isinstance(val, str) or val in ("*", ""):
                continue
            for part in val.split(";"):
                f = part.split(",")
                if len(f) < 7 or all(x in ("*", "") for x in f[:6]):
                    continue
                rows.append({"barcode": bc, "cell_type": ct, "slot": col, "rank": rank,
                             **dict(zip(fields, f + [None] * (len(fields) - len(f)), strict=False))})  # fmt: skip
    return pd.DataFrame(rows, columns=["barcode", "cell_type", "slot", "rank", *fields])


def _melt_parse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loc in ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"):
        for rank in ("", "secondary_"):
            pre = f"{rank}{loc}_"
            if f"{pre}V" not in df.columns and f"{pre}cdr3_aa" not in df.columns:
                continue
            part = pd.DataFrame(
                {"Barcode": df["Barcode"].to_numpy(), "locus": loc, "rank": rank.rstrip("_") or "primary"}
            )
            for f in ("V", "D", "J", "C", "cdr3_aa", "cdr3_nt", "read_count", "transcript_count"):
                part[f] = df[f"{pre}{f}"].to_numpy() if f"{pre}{f}" in df.columns else None
            keep = part[["V", "J", "cdr3_aa"]].notna().any(axis=1) & ~part["cdr3_aa"].isin(["", "nan", "NaN"])
            rows.append(part[keep.to_numpy()])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Barcode", "locus"])


def _melt_wat3r(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loc, v, d, j, aa, nt, rd, um in (
        ("TRB", "TRBV", "TRBD", "TRBJ", "TRB_CDR3", "TRB_CDR3nuc", "TRB_nReads", "TRB_CDR3_UMIcount"),
        ("TRA", "TRAV", None, "TRAJ", "TRA_CDR3", "TRA_CDR3nuc", "TRA_nReads", "TRA_CDR3_UMIcount"),
    ):
        part = pd.DataFrame({"BC": df["BC"].to_numpy(), "locus": loc})
        for out, src in (("V", v), ("D", d), ("J", j), ("cdr3_aa", aa), ("cdr3_nt", nt), ("nReads", rd), ("UMIs", um)):
            part[out] = df[src].to_numpy() if src is not None and src in df.columns else None
        rows.append(part[part[["V", "cdr3_aa"]].notna().any(axis=1).to_numpy()])
    return pd.concat(rows, ignore_index=True)


_SPECIAL_MAPS: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    "trust4_barcode_report": (
        {"cell_id": "barcode", "sequence_id": "consensus_id", "v_call": "V", "d_call": "D", "j_call": "J",
         "c_call": "C", "junction": "cdr3_nt", "junction_aa": "cdr3_aa", "consensus_count": "read_cnt"},
        {"read_cnt": "reads"},
    ),
    "parse_barcode_report": (
        {"cell_id": "Barcode", "locus": "locus", "v_call": "V", "d_call": "D", "j_call": "J", "c_call": "C",
         "junction_aa": "cdr3_aa", "junction": "cdr3_nt", "consensus_count": "read_count",
         "umi_count": "transcript_count"},
        {"read_count": "reads", "transcript_count": "umis"},
    ),
    "wat3r": (
        {"cell_id": "BC", "locus": "locus", "v_call": "V", "d_call": "D", "j_call": "J", "junction_aa": "cdr3_aa",
         "junction": "cdr3_nt", "consensus_count": "nReads", "umi_count": "UMIs"},
        {"nReads": "reads", "UMIs": "umis"},
    ),
}  # fmt: skip


def _wide_format(fmt: ReceptorFormat, df: pd.DataFrame) -> tuple[pd.DataFrame, ReceptorFormat]:
    melt = {"trust4_barcode_report": _melt_trust4_barcode, "parse_barcode_report": _melt_parse, "wat3r": _melt_wat3r}
    long = melt[fmt.name](df)
    cols, counts = _SPECIAL_MAPS[fmt.name]
    spec = ReceptorFormat(
        name=fmt.name, tool=fmt.tool, layout="table", columns={k: (v,) for k, v in cols.items()}, counts=counts,
        level=fmt.level, cell_ids=fmt.cell_ids, docs=fmt.docs, notes=fmt.notes,
    )  # fmt: skip
    return long, spec


# ------------------------------------------------------------------------------------------------- objects


def _flatten_scirpy(adata: Any, airr_key: str = "airr") -> pd.DataFrame:
    from spaclone.io.interop import _from_ak, _need

    ak = _need("awkward")
    arr = adata.obsm[airr_key]
    counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
    flat = ak.flatten(arr, axis=1)
    cols = {f: _from_ak(flat[f]).to_numpy(dtype=object) for f in ak.fields(flat)}
    out = pd.DataFrame(cols)
    if "cell_id" in out.columns:
        out = out.rename(columns={"cell_id": "cell_id_airr"})
    out.insert(0, "cell_id", np.repeat(np.asarray(adata.obs_names, dtype=object), counts))
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = [None if v is None or (isinstance(v, float) and np.isnan(v)) else (str(v) if not isinstance(v, str) else v)
                      for v in out[c]]  # fmt: skip
    return out


_IR = re.compile(r"^IR_(VJ|VDJ)_([12])_(.+)$")
#: scirpy < 0.13 IR_* field names (schema v0.6 and v0.11) → AIRR names.
_IR_FIELDS = {
    "cdr3": "junction_aa",
    "cdr3_nt": "junction",
    "v_gene": "v_call",
    "d_gene": "d_call",
    "j_gene": "j_call",
    "c_gene": "c_call",
    "expr": "umi_count",
    "expr_raw": "consensus_count",
}


def _legacy_scirpy(obs: pd.DataFrame) -> pd.DataFrame:
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for c in obs.columns:
        m = _IR.match(str(c))
        if m:
            groups.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = c
    parts = []
    for (arm, slot), fields in sorted(groups.items()):
        part = pd.DataFrame({"cell_id": np.asarray(obs.index, dtype=object), "ir_arm": arm, "ir_slot": slot})
        for f, c in fields.items():
            part[_IR_FIELDS.get(f, f)] = obs[c].astype(object).to_numpy()
        keyset = [k for k in ("junction_aa", "junction", "v_call", "j_call") if k in part]
        keep = part[keyset].notna().any(axis=1) & ~part[keyset].astype(str).isin(["None", "nan", ""]).all(axis=1)
        parts.append(part[keep.to_numpy()])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cell_id"])


def _read_h5mu(path: str | Path) -> Any:
    """Read a MuData ``.h5mu`` file; mudata is an optional dependency (``spaclone[scirpy]``)."""
    try:
        import mudata as md
    except ImportError as e:
        raise ImportError("reading .h5mu needs mudata: pip install 'spaclone[scirpy]'") from e
    return md.read_h5mu(path)


def is_mudata(x: Any) -> bool:
    """MuData-like object (``.mod`` mapping of modalities), not a DataFrame (whose ``.mod`` is a method)."""
    return not isinstance(x, pd.DataFrame) and hasattr(x, "obs") and hasattr(getattr(x, "mod", None), "items")


def is_anndata(x: Any) -> bool:
    """AnnData-like object (``obs`` and ``obsm``), not a DataFrame."""
    return not isinstance(x, pd.DataFrame) and hasattr(x, "obsm") and hasattr(x, "obs") and not is_mudata(x)


def _anndata_source(adata: Any) -> tuple[pd.DataFrame, str, list[str]]:
    """Receptor rows held by an AnnData / MuData (scirpy) and the format name."""
    if is_mudata(adata):
        mods = [m for m, a in adata.mod.items() if "airr" in a.obsm]
        if len(mods) != 1:
            raise ValueError(f"MuData: expected one modality with obsm['airr'], found {mods or 'none'}")
        return _flatten_scirpy(adata.mod[mods[0]]), "scirpy", [f"MuData modality {mods[0]!r} with obsm['airr']"]
    if "airr" in adata.obsm:
        return _flatten_scirpy(adata), "scirpy", ["AnnData obsm['airr'] (scirpy >= 0.13 awkward array)"]
    ir = [c for c in adata.obs.columns if _IR.match(str(c))]
    if ir:
        return _legacy_scirpy(adata.obs), "scirpy_legacy", [f"{len(ir)} obs columns IR_VJ_* / IR_VDJ_* (scirpy < 0.13)"]
    raise ValueError(
        "this AnnData holds no receptor data: no obsm['airr'] (scirpy), no IR_* obs columns (old scirpy). For "
        "per-cell call columns (e.g. Stereo-XCR-seq LUAD IGHcdr3aa) use spaclone.io.from_cell_calls."
    )


def _dandelion_source(obj: Any) -> pd.DataFrame:
    data = getattr(obj, "data", None)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("expected a Dandelion object with a .data AIRR table")
    df = data.copy()
    if df.index.name == "sequence_id" or "sequence_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sequence_id"})
    return df


def _read_h5ddl(path: Path) -> pd.DataFrame:
    """The AIRR table of a Dandelion ``.h5ddl``.

    The compound dataset ``data`` (read with h5py), else the pandas HDFStore key ``data`` of older files (needs
    PyTables).
    """
    import h5py

    with h5py.File(path, "r") as h:
        node = h.get("data")
        if isinstance(node, h5py.Dataset) and node.dtype.names:
            arr = node[:]
            df = pd.DataFrame({n: arr[n] for n in node.dtype.names})
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = [v.decode() if isinstance(v, bytes) else v for v in df[c]]
            return df.astype(object).where(df.notna(), None)
    try:
        return pd.read_hdf(path, key="data")
    except (ImportError, KeyError, ValueError, TypeError) as e:
        raise ImportError(
            f"{path}: this .h5ddl stores its table as a pandas HDFStore; install PyTables or Dandelion "
            "('pip install spaclone[dandelion]'), or export the table: ddl.read_h5ddl(path).write_airr('contigs.tsv')"
        ) from e


# ------------------------------------------------------------------------------------------------- front door


def receptor_format(source: Any, *, format: str = "auto") -> tuple[ReceptorFormat, list[str]]:
    """Identify the receptor format of a file, table or object (see :func:`read_receptors`).

    Parameters
    ----------
    source
        Path, ``DataFrame``, AnnData / MuData (scirpy) or Dandelion object.
    format
        ``"auto"`` or a format name (checked to exist).

    Returns
    -------
    tuple
        The :class:`~spaclone.io.formats.ReceptorFormat` and the evidence.
    """
    if format != "auto":
        if format not in RECEPTOR_FORMATS and format not in ("scirpy", "scirpy_legacy", "dandelion"):
            raise ValueError(
                f"unknown format {format!r}; known: {', '.join([*RECEPTOR_FORMATS, 'scirpy', 'dandelion'])}"
            )
        return RECEPTOR_FORMATS.get(format, RECEPTOR_FORMATS["airr"]), [f"given: format={format!r}"]
    if isinstance(source, pd.DataFrame):
        m = detect_format([str(c) for c in source.columns])
        return m.format, ["DataFrame columns: " + "; ".join(m.evidence)]
    p = Path(source)
    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as fh:
            head = fh.read(4096)
        keys = re.findall(r'"([A-Za-z_]+)"\s*:', head)
        m = detect_format(keys, layout="json")
        return m.format, ["JSON keys: " + "; ".join(m.evidence)]
    head = peek_table(p)
    m = detect_format(list(head.columns))
    ev = head.evidence() + ["signature: " + "; ".join(m.evidence)]
    if m.superseded:
        ev.append(f"specialises {', '.join(m.superseded)}")
    return m.format, ev


def read_receptors(
    source: Any,
    *,
    format: str = "auto",
    columns: Mapping[str, str] | None = None,
    count_type: Literal["reads", "umis", "templates", "cells"] | None = None,
    junction: Literal["auto", "junction", "cdr3"] = "auto",
) -> pd.DataFrame:
    """Read a receptor table of any supported tool into AIRR-named records.

    Parameters
    ----------
    source
        A file (CSV / TSV, optionally compressed; 10x ``.json``; scirpy ``.h5ad`` / ``.h5mu``; Dandelion ``.h5ddl``),
        several files of the same format (concatenated, ``source_file`` column added), a ``DataFrame``, an AnnData or
        MuData with scirpy data (``obsm["airr"]`` or the old ``IR_*`` columns) or a Dandelion object.
    format
        ``"auto"`` (documented signature, see :data:`spaclone.io.formats.RECEPTOR_FORMATS`), a format name, or
        ``"custom"`` with ``columns``.
    columns
        ``{canonical field: source column}``; wins over the format's synonym table. Canonical fields:
        :data:`spaclone.io.formats.CANONICAL_FIELDS`.
    count_type
        What an unlabelled count column counts (``"reads"``, ``"umis"``, ``"templates"``, ``"cells"``); required
        when the format has only such a column (TRUST4 report, VDJtools, immunarch, MiGEC ``Count``).
    junction
        ``"auto"`` tests the conserved anchors (:data:`ANCHOR_THRESHOLDS`); ``"junction"`` / ``"cdr3"`` overrides.

    Returns
    -------
    pandas.DataFrame
        One row per record (contig, chain, read or clone, see ``attrs["spaclone"]["io"]["level"]``) with the
        canonical fields the source has, in :data:`~spaclone.io.formats.CANONICAL_FIELDS` order, then the other
        source columns under their own names (a source name that is also a canonical field gets the suffix
        ``_src``). ``attrs["spaclone"]["io"]`` records the format, its evidence, the column mapping, the meaning of
        every count column, the junction test and the locus rule.

    Raises
    ------
    ValueError
        Unrecognised or ambiguous format (``format=``), an unlabelled count (``count_type=``), a mix of anchored and
        anchor-less CDR3s (``junction=``), or a ``columns=`` entry naming a missing column.

    Examples
    --------
    >>> tab = read_receptors("filtered_contig_annotations.csv")  # doctest: +SKIP
    >>> tab.attrs["spaclone"]["io"]["counts"]  # doctest: +SKIP
    {'consensus_count': "'reads' = reads (10x_contig_csv)", 'umi_count': "'umis' = umis (10x_contig_csv)"}
    """
    if junction not in ("auto", "junction", "cdr3"):
        raise ValueError("junction must be 'auto', 'junction' or 'cdr3'")
    kw = {"columns": columns, "count_type": count_type, "junction": junction}
    if isinstance(source, list | tuple) and source and isinstance(source[0], _PATH_LIKE):
        parts = []
        for p in source:
            t = read_receptors(p, format=format, **kw)
            t["source_file"] = Path(p).name
            parts.append(t)
        out = pd.concat(parts, ignore_index=True)
        out.attrs["spaclone"] = {"io": {**parts[0].attrs["spaclone"]["io"], "files": [str(p) for p in source],
                                        "n_records": int(len(out))}}  # fmt: skip
        return out
    # objects
    if is_anndata(source) or is_mudata(source):
        df, name, ev = _anndata_source(source)
        return _standardise(df, RECEPTOR_FORMATS["airr"] if name == "scirpy" else _scirpy_legacy_format(),
                            evidence=ev, **kw)  # fmt: skip
    if hasattr(source, "data") and hasattr(source, "metadata") and not isinstance(source, pd.DataFrame):
        return _finish_table(_dandelion_source(source), RECEPTOR_FORMATS["airr"], ["Dandelion object .data"], **kw)
    if isinstance(source, pd.DataFrame):
        fmt, ev = receptor_format(source, format=format)
        df = source.copy()
        df.columns = [str(c) for c in df.columns]
        return _finish_table(df.astype(object).where(df.notna(), None), fmt, ev, **kw)
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"read_receptors: {p} not found")
    if p.is_dir():
        raise ValueError(
            f"{p} is a directory; pass the receptor file itself (e.g. filtered_contig_annotations.csv, "
            "airr_rearrangement.tsv, *_barcode_airr.tsv or clones.tsv)"
        )
    suf = "".join(p.suffixes[-2:]).lower()
    if p.suffix.lower() in (".h5ad", ".h5mu"):
        import anndata as ad

        if p.suffix.lower() == ".h5mu":
            obj = _read_h5mu(p)
        else:
            obj = ad.read_h5ad(p)
        df, name, ev = _anndata_source(obj)
        fmt = RECEPTOR_FORMATS["airr"] if name == "scirpy" else _scirpy_legacy_format()
        return _standardise(df, fmt, evidence=[f"{p.name}: " + e for e in ev], **kw)
    if p.suffix.lower() == ".h5ddl":
        return _finish_table(_read_h5ddl(p), RECEPTOR_FORMATS["airr"], [f"Dandelion {p.name}: compound dataset 'data'"],
                             **kw)  # fmt: skip
    if p.suffix.lower() == ".json" or suf == ".json.gz":
        fmt, ev = receptor_format(p, format=format)
        return _finish_table(_read_10x_json(p), fmt, ev, **kw)
    fmt, ev = receptor_format(p, format=format)
    if fmt.layout == "special":
        how = {
            "stereo_xcr_meta": "read it with spaclone.io.read_stereo_xcr(path)",
            "airspace_ovp": "read it with spaclone.io.read_airspace(bead_h5ad, path) or spaclone.read(bead_h5ad, path)",
        }.get(fmt.name, fmt.notes)
        raise ValueError(f"{p.name} is a {fmt.tool}: {how}")
    head = peek_table(p)
    df = read_text_table(head)
    return _finish_table(df, fmt, ev, **kw)


def _scirpy_legacy_format() -> ReceptorFormat:
    base = RECEPTOR_FORMATS["airr"]
    return ReceptorFormat(
        name="scirpy_legacy",
        tool="scirpy < 0.13 AnnData (IR_VJ_1_* / IR_VDJ_1_* obs columns)",
        columns={**dict(base.columns), "umi_count": ("umi_count",), "consensus_count": ("consensus_count",)},
        counts={"consensus_count": "reads", "umi_count": "umis", "duplicate_count": "umis"},
        level="chain",
        docs="https://scirpy.scverse.org/en/latest/data-structure.html (old IR_* columns: scirpy.io.upgrade_schema)",
        notes="v0.6 names (cdr3, cdr3_nt, v_gene, expr = UMIs, expr_raw = reads) are mapped to AIRR names.",
    )


def _finish_table(
    df: pd.DataFrame, fmt: ReceptorFormat, ev: list[str], *, columns: Any, count_type: Any, junction: str
) -> pd.DataFrame:
    extra_missing: tuple[str, ...] = ()
    count_overrides: dict[str, str] = {}
    if fmt.layout == "wide":
        df, fmt = _wide_format(fmt, df)
    if fmt.name.startswith("immunoseq"):
        extra_missing = ("unresolved", "no data", "na", "null")
    if fmt.name in ("trust4_report", "trust4_barcode_report"):
        extra_missing = ("out_of_frame", "partial")
        if "CDR3aa" in df.columns:
            df = df.assign(productive=np.where(df["CDR3aa"].astype(str) == "out_of_frame", "F", None))
            fmt = ReceptorFormat(**{**fmt.__dict__, "columns": {**dict(fmt.columns), "productive": ("productive",)}})
    if fmt.name == "airr":
        writer, wev, count_overrides = _airr_writer(df)
        ev = [*ev, f"writer: {writer} ({wev})"] if writer else ev
    if fmt.name in ("mixcr_clones", "mixcr_alignments") and "junction_aa" not in (columns or {}):
        aa = next((c for c in ("aaSeqCDR3", "AA. Seq. CDR3") if c in df.columns), None)
        if aa is not None and "productive" not in df.columns:
            s = df[aa].astype(object)
            df = df.assign(productive=np.where(s.isna(), None, np.where(s.astype(str).str.contains(r"[*_]"), "F", "T")))
            fmt = ReceptorFormat(**{**fmt.__dict__, "columns": {**dict(fmt.columns), "productive": ("productive",)},
                                    "notes": fmt.notes + " productive derived: CDR3 in frame and stop-free."})  # fmt: skip
    if fmt.name == "bd_rhapsody" and "Cell_Index" not in df.columns and "" in df.columns:
        df = df.rename(columns={"": "Cell_Index"})
    return _standardise(df, fmt, columns=columns, count_type=count_type, junction=junction, evidence=ev,
                        count_overrides=count_overrides, extra_missing=extra_missing)  # fmt: skip


#: AIRR writers whose count columns deviate from the schema, identified by documented header / id signatures.
_AIRR_WRITERS: tuple[tuple[str, Any, str, dict[str, str]], ...] = (
    (
        "10x Cell Ranger airr_rearrangement.tsv",
        lambda df: (
            {"sequence_id", "duplicate_count", "is_cell", "sequence_aa"} <= set(df.columns)
            and not {"locus", "umi_count", "rearrangement_status"} & set(df.columns)
            and df["sequence_id"].astype(str).str.contains(r"_contig_\d+$").mean() > 0.95
        ),
        "Cell Ranger 4+ AIRR header (sequence_aa, duplicate_count, is_cell; no locus / umi_count) and "
        "sequence_id '<barcode>_contig_<n>'",
        {"consensus_count": "reads", "duplicate_count": "umis"},
    ),
    (
        "TRUST4 *_airr.tsv",
        lambda df: {"complete_vdj", "cdr1", "cdr2", "consensus_count"} <= set(df.columns),
        "columns complete_vdj, cdr1, cdr2, consensus_count",
        {"consensus_count": "reads"},
    ),
    (
        "Dandelion",
        lambda df: bool({"rearrangement_status", "j_call_multimappers", "mu_count"} & set(df.columns)),
        "Dandelion columns (rearrangement_status / j_call_multimappers / mu_count)",
        {"consensus_count": "reads", "umi_count": "umis", "duplicate_count": "umis"},
    ),
)


def _airr_writer(df: pd.DataFrame) -> tuple[str | None, str, dict[str, str]]:
    for name, test, ev, counts in _AIRR_WRITERS:
        try:
            if test(df):
                return name, ev, counts
        except (KeyError, TypeError, AttributeError):
            continue
    return None, "", {}


# ------------------------------------------------------------------------------------------------- per-tool readers


def _family_reader(name: str, family: tuple[str, ...], tool: str) -> Any:
    def reader(
        source: Any,
        *,
        columns: Mapping[str, str] | None = None,
        count_type: Literal["reads", "umis", "templates", "cells"] | None = None,
        junction: Literal["auto", "junction", "cdr3"] = "auto",
    ) -> pd.DataFrame:
        if isinstance(source, str | Path) and Path(source).suffix.lower() not in (".h5ad", ".h5mu", ".h5ddl"):
            p = Path(source)
            if p.suffix.lower() == ".json":
                fmt_name = "10x_contig_json"
            else:
                m = detect_format(list(peek_table(p).columns), candidates=family)
                fmt_name = m.format.name
            return read_receptors(source, format=fmt_name, columns=columns, count_type=count_type, junction=junction)
        return read_receptors(source, columns=columns, count_type=count_type, junction=junction)

    reader.__name__ = name
    reader.__qualname__ = name
    reader.__doc__ = f"""Read {tool} into AIRR-named records (:func:`read_receptors` restricted to {", ".join(family)}).

    Parameters
    ----------
    source
        The file (or object).
    columns, count_type, junction
        As in :func:`read_receptors`.

    Returns
    -------
    pandas.DataFrame
        Records with ``attrs["spaclone"]["io"]``.
    """
    return reader


read_10x_vdj = _family_reader(
    "read_10x_vdj", ("10x_contig_csv", "10x_contig_json", "10x_consensus_csv", "airr"),
    "10x Cell Ranger V(D)J contig annotations (CSV, JSON, airr_rearrangement.tsv)",
)  # fmt: skip
read_mixcr = _family_reader("read_mixcr", ("mixcr_clones", "mixcr_alignments"), "MiXCR exportClones / exportAlignments")
read_trust4 = _family_reader(
    "read_trust4", ("trust4_report", "trust4_barcode_report", "airr"), "TRUST4 reports and AIRR tables"
)
read_changeo = _family_reader("read_changeo", ("changeo", "airr"), "Change-O tables (legacy uppercase or AIRR)")
read_clone_table = _family_reader(
    "read_clone_table", ("immunarch", "vdjtools", "migec", "mitcr", "immunoseq_v2", "immunoseq_v3"),
    "bulk clone tables (immunarch, VDJtools, MiGEC, MiTCR, Adaptive immunoSEQ v1-v3)",
)  # fmt: skip
read_bd_rhapsody = _family_reader("read_bd_rhapsody", ("bd_rhapsody", "airr"), "BD Rhapsody V(D)J tables")
read_parse = _family_reader("read_parse", ("parse_barcode_report", "airr"), "Parse Evercode TCR / BCR barcode_report")
__all__ += [
    "read_10x_vdj",
    "read_bd_rhapsody",
    "read_changeo",
    "read_clone_table",
    "read_mixcr",
    "read_parse",
    "read_trust4",
]
