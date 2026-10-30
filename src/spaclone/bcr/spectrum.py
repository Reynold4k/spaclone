"""Somatic-hypermutation spectrum: substitution types, codon positions and AID hotspot / coldspot targeting.

Every mutation is a column where the sequenced base and the germline base are both A / C / G / T and differ. The
spectrum counts germline -> observed substitutions (transitions A <-> G, C <-> T; transversions otherwise) overall
and by codon position. Targeting compares the share of mutations that fall in a germline sequence motif with the
share of mutable germline positions in that motif:

* ``WRCY/RGYW``: AID hotspot (mutated C of W-R-C-Y, or G of R-G-Y-W; W = A/T, R = A/G, Y = C/T);
* ``SYC/GRS``: AID coldspot (mutated C of S-Y-C, or G of G-R-S; S = C/G);
* ``WA/TW``: polymerase-η hotspot of A / T (mutated A of W-A, or T of T-W).

Motif contexts are read on the ungapped germline (IMGT gap columns are skipped), so a motif can span a gap.

References
----------
Rogozin & Kolchanov 1992, doi:10.1016/0167-4781(92)90134-L (RGYW / WRCY hotspots); Rogozin et al. 2001,
doi:10.1038/88732 (WA / TW and polymerase η); Shapiro et al. 1999, doi:10.4049/jimmunol.163.1.259 (SYC coldspots);
Yaari et al. 2013, doi:10.3389/fimmu.2013.00358 (S5F targeting model, the full method); Kimura 1980,
doi:10.1007/BF01731581 (transitions and transversions).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from spaclone._utils import (
    Alternative,
    RngLike,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    mc_pvalue,
    rng_entropy,
)
from spaclone.bcr.shm import _matrix

__all__ = ["IMGT_REGIONS", "SpectrumResult", "mutations", "spectrum"]

#: IMGT V regions, 1-based inclusive nucleotide positions of IMGT-gapped alignments (CDR3 starts at 313).
IMGT_REGIONS: dict[str, tuple[int, int]] = {
    "FWR1": (1, 78),
    "CDR1": (79, 114),
    "FWR2": (115, 165),
    "CDR2": (166, 195),
    "FWR3": (196, 312),
}

_BASES = "ACGT"
_CODE = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate(_BASES):
    _CODE[ord(_c)] = _i
_GAP = np.zeros(256, dtype=bool)
_GAP[[ord("-"), ord("."), 0]] = True
_TRANSITION = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
MOTIFS = ("WRCY/RGYW", "SYC/GRS", "WA/TW")
_MOTIF_BASES = {"WRCY/RGYW": (1, 2), "SYC/GRS": (1, 2), "WA/TW": (0, 3)}  # C, G / A, T


def _in(codes: np.ndarray, letters: str) -> np.ndarray:
    return np.isin(codes, [_BASES.index(ch) for ch in letters])


@dataclass
class _Sites:
    """Per-column information of a block of records (all arrays (n, w))."""

    g: np.ndarray  # germline base code 0..3, -1 unknown
    q: np.ndarray  # observed base code
    observable: np.ndarray  # both bases known and inside the window
    mutated: np.ndarray
    codon_pos: np.ndarray  # 1..3
    motif: dict[str, np.ndarray]  # column in motif (at its mutable base)
    classifiable: dict[str, np.ndarray]  # context needed by the motif class is known
    column: np.ndarray  # 1-based alignment column


def _sites(seq: Sequence[Any], ger: Sequence[Any], start: int | None, end: int | None, frame: np.ndarray) -> _Sites:
    q8 = _matrix(seq)
    g8 = _matrix(ger)
    w = max(q8.shape[1], g8.shape[1])
    q8 = np.pad(q8, ((0, 0), (0, w - q8.shape[1])))
    g8 = np.pad(g8, ((0, 0), (0, w - g8.shape[1])))
    n = len(q8)
    col = np.broadcast_to(np.arange(1, w + 1), (n, w))
    window = np.ones((n, w), dtype=bool)
    if start is not None:
        window &= col >= start
    if end is not None:
        window &= col <= end
    g = _CODE[g8]
    q = _CODE[q8]
    # germline coordinate: every column except insertions ('-' in the germline) and padding advances it
    adv = (g8 != ord("-")) & (g8 != 0)
    coord = np.cumsum(adv, axis=1) - 1
    codon_pos = (coord + frame[:, None]) % 3 + 1
    # ungapped germline context: neighbours among the germline's non-gap columns
    nongap = ~_GAP[g8]
    rank = np.cumsum(nongap, axis=1) - 1
    idx = np.argsort(~nongap, axis=1, kind="stable")
    comp = np.take_along_axis(g, idx, axis=1)
    comp = np.where(np.take_along_axis(nongap, idx, axis=1), comp, -1)
    ncomp = nongap.sum(1)

    def nb(offset: int) -> np.ndarray:
        j = rank + offset
        ok = nongap & (j >= 0) & (j < ncomp[:, None])
        jj = np.clip(j, 0, max(w - 1, 0))
        return np.where(ok, np.take_along_axis(comp, jj, axis=1), -1)

    l2, l1, r1, r2 = nb(-2), nb(-1), nb(1), nb(2)
    known = lambda a: a >= 0
    is_c, is_g, is_a, is_t = g == 1, g == 2, g == 0, g == 3
    motif = {
        "WRCY/RGYW": (is_c & _in(l2, "AT") & _in(l1, "AG") & _in(r1, "CT"))
        | (is_g & _in(l1, "AG") & _in(r1, "CT") & _in(r2, "AT")),
        "SYC/GRS": (is_c & _in(l2, "CG") & _in(l1, "CT")) | (is_g & _in(r1, "AG") & _in(r2, "CG")),
        "WA/TW": (is_a & _in(l1, "AT")) | (is_t & _in(r1, "AT")),
    }
    cg_ok = (is_c & known(l2) & known(l1) & known(r1)) | (is_g & known(l1) & known(r1) & known(r2))
    at_ok = (is_a & known(l1)) | (is_t & known(r1))
    classifiable = {"WRCY/RGYW": cg_ok, "SYC/GRS": cg_ok, "WA/TW": at_ok}
    observable = window & (g >= 0) & (q >= 0)
    mutated = observable & (g != q)
    return _Sites(g, q, observable, mutated, codon_pos, motif, classifiable, col)


def _frame(air: pd.DataFrame, frame_offset: Any) -> np.ndarray:
    if isinstance(frame_offset, str):
        if frame_offset not in air.columns:
            raise KeyError(f"frame_offset: column {frame_offset!r} not in the table")
        v = pd.to_numeric(air[frame_offset], errors="coerce").fillna(0).to_numpy()
    else:
        v = np.broadcast_to(np.asarray(frame_offset), (len(air),))
    return np.asarray(v, dtype=np.int64) % 3


def _window(region: Any) -> tuple[int | None, int | None]:
    if region is None:
        return None, None
    if isinstance(region, str):
        if region == "V":
            return 1, 312
        if region in IMGT_REGIONS:
            return IMGT_REGIONS[region]
        raise ValueError(f"region must be None, 'V', an IMGT region name or (start, end), got {region!r}")
    start, end = (int(v) for v in region)
    if start < 1 or end < start:
        raise ValueError("region (start, end) is 1-based with end >= start")
    return start, end


def _imgt_region(col: np.ndarray) -> np.ndarray:
    out = np.full(col.shape, "", dtype=object)
    for name, (a, b) in IMGT_REGIONS.items():
        out[(col >= a) & (col <= b)] = name
    out[col > 312] = "CDR3+"
    return out


def mutations(
    air: pd.DataFrame,
    *,
    sequence_alignment: str = "sequence_alignment",
    germline_alignment: str = "germline_alignment",
    region: Any = None,
    frame_offset: Any = 0,
) -> pd.DataFrame:
    """One row per mutation (germline and sequenced base both A / C / G / T and different).

    Parameters
    ----------
    air
        Records with aligned sequence and germline (IMGT-gapped AIRR ``sequence_alignment`` /
        ``germline_alignment`` from IgBLAST + Change-O by default; AIRR ``v_sequence_alignment`` /
        ``v_germline_alignment`` work too).
    sequence_alignment, germline_alignment
        Column names.
    region
        Alignment window: ``None`` (all columns), ``"V"`` (IMGT 1-312), an IMGT region name (``"CDR1"``, ...) or
        ``(start, end)`` (1-based inclusive columns).
    frame_offset
        Codon phase of the first alignment column (0 = first codon position), a number or a column; 0 for
        IMGT-gapped alignments. For alignments that start inside the germline gene (e.g. TRUST4's
        ``v_germline_start``, 0-based), pass that column.

    Returns
    -------
    pandas.DataFrame
        ``record`` (row position in ``air``), ``index`` (row label), ``column`` (1-based), ``imgt_region``,
        ``germline``, ``observed``, ``substitution`` (e.g. ``"C>T"``), ``transition`` (bool), ``codon_position``
        (1-3) and one boolean column per motif class (at the mutated base: ``WRCY/RGYW``, ``SYC/GRS``, ``WA/TW``).
    """
    for c in (sequence_alignment, germline_alignment):
        if c not in air.columns:
            raise KeyError(f"mutations: column {c!r} not in the table")
    start, end = _window(region)
    frame = _frame(air, frame_offset)
    parts = []
    seqs = air[sequence_alignment].to_numpy(dtype=object)
    gers = air[germline_alignment].to_numpy(dtype=object)
    for a in range(0, len(air), 5000):
        b = min(len(air), a + 5000)
        st = _sites(seqs[a:b], gers[a:b], start, end, frame[a:b])
        r, c = np.nonzero(st.mutated)
        gb = np.array(list(_BASES))[st.g[r, c]]
        ob = np.array(list(_BASES))[st.q[r, c]]
        d = {
            "record": r + a,
            "column": st.column[r, c],
            "germline": gb,
            "observed": ob,
            "codon_position": st.codon_pos[r, c],
        }
        for m in MOTIFS:
            d[m] = st.motif[m][r, c] & st.classifiable[m][r, c]
        parts.append(pd.DataFrame(d))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["record", "column"])
    if not len(out):
        out = pd.DataFrame(
            {
                "record": pd.Series(dtype=np.int64),
                "column": pd.Series(dtype=np.int64),
                "germline": pd.Series(dtype=object),
                "observed": pd.Series(dtype=object),
                "codon_position": pd.Series(dtype=np.int64),
                **{m: pd.Series(dtype=bool) for m in MOTIFS},
            }
        )
    out.insert(1, "index", air.index.to_numpy()[out.record.to_numpy(dtype=np.int64)])
    out.insert(3, "imgt_region", _imgt_region(out.column.to_numpy(dtype=np.int64)))
    out.insert(6, "substitution", out.germline.astype(str) + ">" + out.observed.astype(str))
    out.insert(7, "transition", [(x, y) in _TRANSITION for x, y in zip(out.germline, out.observed, strict=True)])
    return out


@dataclass(frozen=True, repr=False)
class SpectrumResult(TestResult):
    """Result of :func:`spectrum`: motif targeting tests plus the substitution spectrum.

    Attributes
    ----------
    substitutions
        Long table: ``group``, ``germline``, ``observed``, ``count``, ``transition``.
    codon_positions
        ``group``, ``codon_position``, ``mutations``, ``transitions``, ``transversions``, ``transition_share``.
    """

    substitutions: pd.DataFrame = field(default_factory=pd.DataFrame)
    codon_positions: pd.DataFrame = field(default_factory=pd.DataFrame)


def spectrum(
    air: pd.DataFrame,
    *,
    sequence_alignment: str = "sequence_alignment",
    germline_alignment: str = "germline_alignment",
    region: Any = "V",
    frame_offset: Any = 0,
    by: str | None = None,
    n_perm: int = 1000,
    alternative: Alternative = "two-sided",
    rng: RngLike = None,
    keep_null: bool = False,
) -> SpectrumResult:
    """Substitution spectrum and targeting of AID hotspot / coldspot motifs against the germline motif background.

    Parameters
    ----------
    air
        Records with aligned sequence and germline (see :func:`mutations`). Inherited mutations are counted once per
        record, so pass one record per clone (or per cell) to avoid counting a lineage's shared mutations many
        times.
    sequence_alignment, germline_alignment, region, frame_offset
        See :func:`mutations` (default window: IMGT V positions 1-312).
    by
        Optional grouping column (e.g. switched / unswitched, inside / outside a hotspot); rows per group.
    n_perm
        Monte-Carlo draws of the null.
    alternative
        Direction of the targeting tests (``"greater"`` = enriched in the motif).
    rng
        Seed or generator.
    keep_null
        Keep the null draws, shape ``(n_perm, n_rows)``.

    Returns
    -------
    SpectrumResult
        ``table``: one row per (group, motif class): ``group``, ``motif``, ``mutations`` (mutations at the motif's
        bases with known context), ``statistic`` (those in the motif), ``share`` (statistic / mutations),
        ``expected`` (sum over records and bases of mutations × motif share of the mutable positions),
        ``expected_share``, ``ratio`` (statistic / expected), ``null_mean``, ``null_sd``, ``z``, ``p_value``,
        ``q_value`` (BH over rows), ``n_perm``, ``n_records``. ``substitutions`` and ``codon_positions`` as
        attributes; ``summary`` holds the overall transition share and the transition / transversion ratio.

    Notes
    -----
    Null model: in every record, the mutations at each germline base (C, G, A or T) are placed uniformly at random,
    without replacement, on the record's positions of the same germline base that are observed (sequenced base
    known) and whose motif context is known; the number of mutations per record and base is fixed. The count in a
    motif is then a sum of independent hypergeometric variables, drawn exactly. Under uniform targeting (no motif
    preference) the share of transitions is 1/3 (one transition and two transversions per base).

    References
    ----------
    Rogozin & Kolchanov 1992, doi:10.1016/0167-4781(92)90134-L; Rogozin et al. 2001, doi:10.1038/88732; Shapiro et
    al. 1999, doi:10.4049/jimmunol.163.1.259; Yaari et al. 2013, doi:10.3389/fimmu.2013.00358 (full targeting model,
    SHazaM); Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    check_alternative(alternative)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    for c in [sequence_alignment, germline_alignment] + ([by] if by else []):
        if c not in air.columns:
            raise KeyError(f"spectrum: column {c!r} not in the table")
    gen = as_rng(rng)
    start, end = _window(region)
    frame = _frame(air, frame_offset)
    grp = (
        pd.Series(air[by].to_numpy(dtype=object)).astype("string").fillna("NA").to_numpy(dtype=object)
        if by
        else np.full(len(air), "all", dtype=object)
    )
    groups = sorted(pd.unique(grp))
    seqs = air[sequence_alignment].to_numpy(dtype=object)
    gers = air[germline_alignment].to_numpy(dtype=object)
    # per (record, base, motif): N eligible, K in motif, k mutated, O mutated in motif
    acc: dict[str, list[np.ndarray]] = {m: [] for m in MOTIFS}
    subs = np.zeros((len(groups), 4, 4), dtype=np.int64)
    cpos = np.zeros((len(groups), 3, 2), dtype=np.int64)  # (transitions, transversions) by codon position
    gidx = {g: i for i, g in enumerate(groups)}
    gcode = np.array([gidx[g] for g in grp], dtype=np.int64)
    is_ti = np.zeros((4, 4), dtype=bool)
    for x, y in _TRANSITION:
        is_ti[_BASES.index(x), _BASES.index(y)] = True
    for a in range(0, len(air), 5000):
        b = min(len(air), a + 5000)
        st = _sites(seqs[a:b], gers[a:b], start, end, frame[a:b])
        r, c = np.nonzero(st.mutated)
        gg = gcode[a:b][r]
        np.add.at(subs, (gg, st.g[r, c], st.q[r, c]), 1)
        ti = is_ti[st.g[r, c], st.q[r, c]]
        np.add.at(cpos, (gg, st.codon_pos[r, c] - 1, (~ti).astype(np.int64)), 1)
        for m in MOTIFS:
            elig = st.observable & st.classifiable[m]
            for base in _MOTIF_BASES[m]:
                eb = elig & (st.g == base)
                acc[m].append(
                    np.column_stack(
                        [
                            gcode[a:b],
                            eb.sum(1),
                            (eb & st.motif[m]).sum(1),
                            (eb & st.mutated).sum(1),
                            (eb & st.mutated & st.motif[m]).sum(1),
                        ]
                    )
                )
    rows, nulls = [], []
    n_rec = np.bincount(gcode, minlength=len(groups))
    for m in MOTIFS:
        arr = np.vstack(acc[m]) if acc[m] else np.zeros((0, 5), dtype=np.int64)
        for gi, g in enumerate(groups):
            sub = arr[(arr[:, 0] == gi) & (arr[:, 3] > 0)]
            nn, kk_, kmut, obs_in = sub[:, 1], sub[:, 2], sub[:, 3], sub[:, 4]
            n_mut = int(kmut.sum())
            stat = float(obs_in.sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                exp = float((kmut * kk_ / np.maximum(nn, 1)).sum())
            draws = np.zeros(n_perm)
            if len(sub):
                for c0 in range(0, len(sub), 20_000):
                    s_ = sub[c0 : c0 + 20_000]
                    draws += gen.hypergeometric(s_[:, 2], s_[:, 1] - s_[:, 2], s_[:, 3], size=(n_perm, len(s_))).sum(1)
            nulls.append(draws)
            mu = float(draws.mean())
            sd = float(draws.std(ddof=1)) if n_perm > 1 else np.nan
            rows.append(
                {
                    "group": g,
                    "motif": m,
                    "mutations": n_mut,
                    "statistic": stat,
                    "share": stat / n_mut if n_mut else np.nan,
                    "expected": exp,
                    "expected_share": exp / n_mut if n_mut else np.nan,
                    "ratio": stat / exp if exp > 0 else np.nan,
                    "null_mean": mu,
                    "null_sd": sd,
                    "z": (stat - mu) / sd if sd and sd > 0 else np.nan,
                    "p_value": float(mc_pvalue(draws, stat, alternative=alternative)[0]) if n_mut else np.nan,
                    "n_perm": n_perm,
                    "n_records": int(n_rec[gi]),
                }
            )
    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("n_perm"), "q_value", bh(table.p_value.to_numpy()))
    order = {m: i for i, m in enumerate(MOTIFS)}
    table = table.sort_values(["group", "motif"], key=lambda s: s.map(order) if s.name == "motif" else s)
    null = np.column_stack(nulls)[:, table.index.to_numpy()] if keep_null and nulls else None
    table = table.reset_index(drop=True)
    st_rows = []
    for gi, g in enumerate(groups):
        for i, x in enumerate(_BASES):
            for j, y in enumerate(_BASES):
                if i != j:
                    st_rows.append(
                        {
                            "group": g,
                            "germline": x,
                            "observed": y,
                            "count": int(subs[gi, i, j]),
                            "transition": is_ti[i, j],
                        }
                    )
    substitutions = pd.DataFrame(st_rows)
    cp_rows = []
    for gi, g in enumerate(groups):
        for k in range(3):
            t_, v_ = int(cpos[gi, k, 0]), int(cpos[gi, k, 1])
            cp_rows.append(
                {
                    "group": g,
                    "codon_position": k + 1,
                    "mutations": t_ + v_,
                    "transitions": t_,
                    "transversions": v_,
                    "transition_share": t_ / (t_ + v_) if t_ + v_ else np.nan,
                }
            )
    codon_positions = pd.DataFrame(cp_rows)
    n_ti = int(substitutions.loc[substitutions.transition, "count"].sum())
    n_tv = int(substitutions.loc[~substitutions.transition, "count"].sum())
    summary = {
        "mutations": n_ti + n_tv,
        "transition_share": n_ti / (n_ti + n_tv) if n_ti + n_tv else np.nan,
        "ti_tv_ratio": n_ti / n_tv if n_tv else np.nan,
        "records": len(air),
    }
    for _, r in table[table.group == groups[0]].iterrows():
        summary[f"ratio[{r.motif}]"] = r.ratio
    params = {
        "sequence_alignment": sequence_alignment,
        "germline_alignment": germline_alignment,
        "region": region,
        "frame_offset": frame_offset if isinstance(frame_offset, str | int) else "per record",
        "by": by,
        "n_perm": n_perm,
        "alternative": alternative,
        "design": "mutations placed uniformly on same-base observable positions of each record (hypergeometric)",
        "rng_entropy": rng_entropy(gen),
    }
    return SpectrumResult(
        name="spectrum",
        table=table,
        summary=summary,
        params=params,
        null=null,
        substitutions=substitutions,
        codon_positions=codon_positions,
    )
