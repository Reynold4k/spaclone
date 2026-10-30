"""Match receptor spot / cell barcodes to GEX unit ids: exact first, then only documented normalisations.

A normalisation is tried only when the exact overlap is below ``min_overlap``, and kept only when it turns that low
overlap into one of at least ``min_overlap``. Two normalisations that both succeed must give the same mapping, else
the match is ambiguous and raises. The rules (each a documented id convention, never a fuzzy match):

* ``10x_suffix``: Cell Ranger / Space Ranger barcodes end in ``-<GEM well>`` (``AAACCTGAGATCTGAA-1``); one side may
  have dropped it;
* ``concat_key``: ``anndata.concat(..., index_unique=sep)`` appends ``<sep><key>`` to every obs name, Seurat's
  ``RenameCells(add.cell.id=)`` / ``merge`` prepends ``<key>_``; the key is a section label shared by many ids;
* ``visium_hd_bins``: Visium HD bin ids ``s_002um_RRRRR_CCCCC-1``; a finer receptor bin maps to the coarser GEX bin
  ``(row // f, col // f)`` with ``f`` the ratio of the bin sizes;
* ``stereo_coordinates``: Stereo-seq ids that spell DNB / bin coordinates in another convention (``"x-y"``,
  ``"x_y"``, ``"sample:x_y"``, stereopy's packed ``x << 32 | y``).

References
----------
10x Genomics, "What is the -1 at the end of the barcodes?" and Visium HD binned outputs; anndata ``concat``
(index_unique); Seurat ``RenameCells``; stereopy ``parse_bin_coor_no_offset`` (packed bin ids).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["BarcodeMatch", "match_barcodes"]

_SUFFIX = re.compile(r"-\d+$")
_HD = re.compile(r"^s_(\d{3})um_(\d{5})_(\d{5})(?:-\d+)?$")


@dataclass
class BarcodeMatch:
    """How receptor spot ids were matched to GEX unit ids.

    Attributes
    ----------
    mapping
        Receptor id → GEX id for every matched distinct receptor id.
    rule
        ``"exact"`` or the normalisation used.
    overlap_before, overlap_after
        Share of distinct receptor ids that are GEX ids exactly / after the rule.
    tried
        ``{rule: overlap}`` of every normalisation tried.
    examples
        A few unmatched ids of each side (for error messages).
    """

    mapping: pd.Series
    rule: str
    overlap_before: float
    overlap_after: float
    tried: dict[str, float] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """JSON-ready summary for ``meta["io"]``."""
        return {
            "rule": self.rule,
            "overlap_before": round(self.overlap_before, 6),
            "overlap_after": round(self.overlap_after, 6),
            "tried": {k: round(v, 6) for k, v in self.tried.items()},
            "n_matched_ids": int(len(self.mapping)),
        }


_NT = re.compile(r"^[ACGTN]{6,}$")


def _strip_suffix(ids: pd.Index) -> pd.Index | None:
    """Ids without the 10x ``-<GEM well>`` suffix, or None when the cores are not nucleotide barcodes."""
    core = pd.Index(pd.Series(ids, dtype=str).str.replace(_SUFFIX, "", regex=True))
    return core if pd.Series(core, dtype=str).str.match(_NT.pattern).all() else None


def _suffix_rule(rec: pd.Index, gex: pd.Index) -> pd.Series | None:
    rk, gk = _strip_suffix(rec), _strip_suffix(gex)
    return None if rk is None or gk is None else _via_keys(rec, gex, rk, gk)


def _concat_keys(ids: pd.Index) -> list[tuple[str, pd.Index]]:
    """Candidate cores of GEX ids built by anndata.concat (suffix) or Seurat (prefix): (label, core ids)."""
    out = []
    s = pd.Series(ids, dtype=str)
    for sep in ("-", "_", ":"):
        last = s.str.rsplit(sep, n=1)
        ok = last.str.len() == 2
        if ok.all():
            keys = last.str[1]
            if keys.nunique() <= max(1, len(s) // 50) and keys.nunique() <= 1000:
                out.append((f"suffix '{sep}<key>' ({keys.nunique()} keys)", pd.Index(last.str[0])))
        first = s.str.split(sep, n=1)
        ok = first.str.len() == 2
        if ok.all():
            keys = first.str[0]
            if keys.nunique() <= max(1, len(s) // 50) and keys.nunique() <= 1000:
                out.append((f"prefix '<key>{sep}' ({keys.nunique()} keys)", pd.Index(first.str[1])))
    return out


def _stereo_xy(ids: pd.Index) -> pd.Series | None:
    """(x, y) packed as one int64 per id for Stereo-seq coordinate ids of any convention, else None."""
    s = pd.Series(ids, dtype=str)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and (num >= 0).all() and (num.astype("int64") >= (1 << 32)).mean() > 0.5:
        return pd.Series(s.astype("int64").to_numpy(), index=ids)
    core = s.str.replace(r"^[^:]*:", "", regex=True)
    for sep in ("-", "_"):
        parts = core.str.split(sep, regex=False)
        if (parts.str.len() == 2).all():
            x = pd.to_numeric(parts.str[0], errors="coerce")
            y = pd.to_numeric(parts.str[1], errors="coerce")
            if x.notna().all() and y.notna().all() and (x >= 0).all() and (y >= 0).all():
                xi, yi = x.to_numpy(dtype=np.int64), y.to_numpy(dtype=np.int64)
                return pd.Series((xi << 32) | yi, index=ids)
    return None


def _via_keys(rec: pd.Index, gex: pd.Index, rk: pd.Index | np.ndarray, gk: pd.Index | np.ndarray) -> pd.Series | None:
    """Mapping receptor id → GEX id through shared keys; None when the GEX keys are not unique (collision)."""
    gk = pd.Index(gk)
    if not gk.is_unique:
        return None
    pos = gk.get_indexer(pd.Index(rk))
    ok = pos >= 0
    return pd.Series(np.asarray(gex, dtype=object)[pos[ok]], index=pd.Index(np.asarray(rec, dtype=object)[ok]))


def _hd(rec: pd.Index, gex: pd.Index) -> pd.Series | None:
    rm = pd.Series(rec, dtype=str).str.extract(_HD)
    gm = pd.Series(gex, dtype=str).str.extract(_HD)
    if rm[0].isna().any() or gm[0].isna().any() or rm[0].nunique() != 1 or gm[0].nunique() != 1:
        return None
    rs, gs = int(rm[0].iloc[0]), int(gm[0].iloc[0])
    if gs % rs:
        return None
    f = gs // rs
    rk = (rm[1].astype(int) // f).astype(str).str.zfill(5) + "_" + (rm[2].astype(int) // f).astype(str).str.zfill(5)
    gk = gm[1].str.zfill(5) + "_" + gm[2].str.zfill(5)
    return _via_keys(rec, gex, rk.to_numpy(dtype=object), gk.to_numpy(dtype=object))


def match_barcodes(receptor_ids: Any, gex_ids: Any, *, min_overlap: float = 0.5) -> BarcodeMatch:
    """Match distinct receptor spot ids to GEX unit ids (exact, else one documented normalisation).

    Parameters
    ----------
    receptor_ids
        Spot / cell ids of the receptor records (duplicates allowed).
    gex_ids
        GEX obs names (unique).
    min_overlap
        Share of distinct receptor ids that must match.

    Returns
    -------
    BarcodeMatch

    Raises
    ------
    ValueError
        When no rule reaches ``min_overlap`` (examples from both sides and every rule's overlap are given), or two
        rules reach it with different mappings.
    """
    rec = pd.Index(pd.unique(np.asarray(receptor_ids, dtype=object)).astype(str))
    gex = pd.Index(np.asarray(gex_ids, dtype=object).astype(str))
    if not gex.is_unique:
        raise ValueError("GEX obs names must be unique")
    hit = rec.isin(gex)
    before = float(hit.mean()) if len(rec) else 0.0
    ex = {"receptor": [str(v) for v in rec[~hit][:3]], "gex": [str(v) for v in gex[:3]]}
    if before >= min_overlap or not len(rec):
        return BarcodeMatch(pd.Series(rec[hit], index=rec[hit]), "exact", before, before, {}, ex)
    rules: list[tuple[str, Callable[[], pd.Series | None]]] = [
        ("10x_suffix", lambda: _suffix_rule(rec, gex)),
        ("visium_hd_bins", lambda: _hd(rec, gex)),
    ]
    for label, core in _concat_keys(gex):
        rules.append((f"concat_key {label}", lambda core=core: _via_keys(rec, gex, rec, core)))
        rules.append((f"concat_key {label} + 10x_suffix",
                      lambda core=core: (None if _strip_suffix(rec) is None or _strip_suffix(core) is None
                                         else _via_keys(rec, gex, _strip_suffix(rec), _strip_suffix(core)))))  # fmt: skip
    rs, gs = _stereo_xy(rec), _stereo_xy(gex)
    if rs is not None and gs is not None:
        rules.append(("stereo_coordinates", lambda: _via_keys(rec, gex, rs.to_numpy(), gs.to_numpy())))
    tried: dict[str, float] = {}
    good: list[tuple[str, pd.Series]] = []
    for name, fn in rules:
        m = fn()
        share = 0.0 if m is None else float(len(m) / len(rec))
        tried[name] = share
        if m is not None and share >= min_overlap:
            good.append((name, m))
    if not good:
        raise ValueError(
            f"only {before:.1%} of the {len(rec):,} receptor spot ids are GEX units (< min_overlap = {min_overlap:.0%})"
            f", and no documented id normalisation helps ({', '.join(f'{k}: {v:.1%}' for k, v in tried.items()) or 'none applicable'}). "
            f"Receptor ids: {ex['receptor']}; GEX obs names: {ex['gex']}. Check spot_key=, whether both files come from "
            "the same section, or rename the ids yourself."
        )
    name, m = good[0]
    for other, m2 in good[1:]:
        both = m.index.intersection(m2.index)
        if len(m) != len(m2) or not (m.loc[both] == m2.loc[both]).all():
            raise ValueError(
                f"ambiguous barcode normalisation: {name} and {other} both match >= {min_overlap:.0%} of the receptor ids "
                "but map them differently; rename the ids to the GEX convention yourself"
            )
    return BarcodeMatch(m, name, before, float(len(m) / len(rec)), tried, ex)
