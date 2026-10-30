"""Antigen selection from replacement (R) and silent (S) mutations in CDRs and FWRs.

Affinity maturation favours amino-acid replacements in the complementarity-determining regions (CDRs, antigen
contact) and removes those that destabilise the framework (FWRs). A region under positive selection therefore
carries more R mutations than the germline codons would produce by chance, and a region under negative selection
fewer.

The model here is deliberately simple and fully stated: every mutation is classified by changing only its own base
in the germline codon (other codon positions germline); mutations creating a stop codon are reported but left out;
the expected R share of a region comes from all single-nucleotide substitutions of the record's germline codons in
that region, with transitions weighted ``kappa`` times a transversion (``kappa = 1``: uniform substitution). Only
codons fully sequenced and fully templated in the germline count. Observed R counts are compared with the sum of
per-record binomial draws. BASELINe / SHazaM (Yaari et al. 2012, 2013) implement the full method with
context-dependent targeting and substitution models (S5F) and a Bayesian selection strength; use them for published
estimates.

References
----------
Shlomchik et al. 1987, doi:10.1038/328805a0 (R / S in CDRs); Hershberg et al. 2008, doi:10.1093/intimm/dxn026;
Uduman et al. 2011, doi:10.1093/nar/gkr413; Yaari et al. 2012, doi:10.1093/nar/gks457 (BASELINe); Yaari et al.
2013, doi:10.3389/fimmu.2013.00358 (S5F); Lefranc et al. 2003, doi:10.1016/S0145-305X(02)00039-3 (IMGT numbering).
"""

from __future__ import annotations

from itertools import product
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
from spaclone.bcr.spectrum import IMGT_REGIONS

__all__ = ["CODON_TABLE", "classify_mutation", "selection"]

_BASES = "ACGT"
_AA = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
#: Standard genetic code, codon (upper-case DNA) -> one-letter amino acid (``*`` = stop).
CODON_TABLE: dict[str, str] = {"".join(c): _AA[i] for i, c in enumerate(product(_BASES, repeat=3))}
_AA_ARR = np.array([CODON_TABLE["".join(c)] for c in product(_BASES, repeat=3)])
_CODE = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate(_BASES):
    _CODE[ord(_c)] = _i
_TI = {(0, 2), (2, 0), (1, 3), (3, 1)}

#: IMGT V regions grouped into CDR and FWR.
REGION_CLASS = {"FWR1": "FWR", "CDR1": "CDR", "FWR2": "FWR", "CDR2": "CDR", "FWR3": "FWR"}


def _expected_tables(kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Per germline codon (index 0..63): summed weight of R and of S single-nucleotide substitutions (no stops)."""
    w_r = np.zeros(64)
    w_s = np.zeros(64)
    for ci in range(64):
        if _AA_ARR[ci] == "*":
            continue
        bases = [ci // 16, (ci // 4) % 4, ci % 4]
        for p in range(3):
            for alt in range(4):
                if alt == bases[p]:
                    continue
                nb = list(bases)
                nb[p] = alt
                aa = _AA_ARR[nb[0] * 16 + nb[1] * 4 + nb[2]]
                if aa == "*":
                    continue
                w = kappa if (bases[p], alt) in _TI else 1.0
                if aa == _AA_ARR[ci]:
                    w_s[ci] += w
                else:
                    w_r[ci] += w
    return w_r, w_s


def classify_mutation(germline_codon: str, position: int, base: str) -> str:
    """``"R"`` (replacement), ``"S"`` (silent) or ``"stop"`` for one base change in a germline codon.

    Parameters
    ----------
    germline_codon
        Three germline nucleotides.
    position
        Codon position of the change (1, 2 or 3).
    base
        The new nucleotide.

    Examples
    --------
    >>> classify_mutation("TTA", 3, "G")  # Leu -> Leu
    'S'
    """
    g = germline_codon.upper()
    if len(g) != 3 or any(ch not in _BASES for ch in g) or base.upper() not in _BASES or position not in (1, 2, 3):
        raise ValueError("need a germline codon of A/C/G/T, a position 1-3 and a base A/C/G/T")
    m = g[: position - 1] + base.upper() + g[position:]
    if CODON_TABLE[m] == "*":
        return "stop"
    return "S" if CODON_TABLE[m] == CODON_TABLE[g] else "R"


def selection(
    air: pd.DataFrame,
    *,
    sequence_alignment: str = "sequence_alignment",
    germline_alignment: str = "germline_alignment",
    kappa: float = 1.0,
    by: str | None = None,
    n_perm: int = 2000,
    alternative: Alternative = "two-sided",
    rng: RngLike = None,
    keep_null: bool = False,
) -> TestResult:
    """Replacement mutations in CDRs and FWRs against the expectation from the germline codons.

    Parameters
    ----------
    air
        Records with IMGT-gapped alignments (IgBLAST + Change-O ``MakeDb``: IMGT position = alignment column).
        Inherited mutations count once per record; pass one record per clone (or cell) for lineage-level
        statements.
    sequence_alignment, germline_alignment
        Column names.
    kappa
        Transition / transversion weight of the expected substitutions (1 = uniform; SHM has a transition bias,
        e.g. the observed ratio from :func:`spaclone.bcr.spectrum`).
    by
        Optional grouping column (isotype, switched, inside / outside a hotspot); rows per group.
    n_perm
        Monte-Carlo draws of the null.
    alternative
        ``"greater"`` tests positive selection (more R than expected), ``"less"`` negative selection.
    rng
        Seed or generator.
    keep_null
        Keep the null draws.

    Returns
    -------
    TestResult
        ``table``: one row per (group, region class) with ``group``, ``region`` (``CDR`` = CDR1 + CDR2, ``FWR`` =
        FWR1-3, IMGT positions 1-312), ``n_records``, ``mutations`` (R + S), ``R``, ``S``, ``stop`` (left out),
        ``share_R``, ``expected_share_R``, ``statistic`` (= R), ``expected``, ``null_mean``, ``null_sd``, ``z``,
        ``sigma`` (log of the observed over the expected R / S odds, 0.5 added to R and S), ``p_value``,
        ``q_value`` (BH over rows), ``n_perm``.

    Notes
    -----
    Null model: neutral mutation with uniform targeting. The number of R + S mutations of each record in each region
    is fixed; each is R independently with the record's expected R probability in that region (from its germline
    codons and ``kappa``), so the R count of a group is a sum of independent binomials, drawn by Monte Carlo.
    Context-dependent targeting (hotspots) changes the expectation; the full method (BASELINe) models it.

    References
    ----------
    Hershberg et al. 2008, doi:10.1093/intimm/dxn026; Uduman et al. 2011, doi:10.1093/nar/gkr413; Yaari et al. 2012,
    doi:10.1093/nar/gks457; Yaari et al. 2013, doi:10.3389/fimmu.2013.00358; Phipson & Smyth 2010,
    doi:10.2202/1544-6115.1585.
    """
    check_alternative(alternative)
    if not kappa > 0:
        raise ValueError("kappa must be positive")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    for c in [sequence_alignment, germline_alignment] + ([by] if by else []):
        if c not in air.columns:
            raise KeyError(f"selection: column {c!r} not in the table")
    gen = as_rng(rng)
    w_r, w_s = _expected_tables(kappa)
    ncod = 104  # IMGT codons 1-104 = nucleotides 1-312
    cls = np.empty(ncod, dtype=object)
    for name, (a, b) in IMGT_REGIONS.items():
        cls[(a - 1) // 3 : b // 3] = REGION_CLASS[name]
    is_cdr = cls == "CDR"
    grp = (
        pd.Series(air[by].to_numpy(dtype=object)).astype("string").fillna("NA").to_numpy(dtype=object)
        if by
        else np.full(len(air), "all", dtype=object)
    )
    groups = sorted(pd.unique(grp))
    gi_of = {g: i for i, g in enumerate(groups)}
    gcode = np.array([gi_of[g] for g in grp], dtype=np.int64)
    n = len(air)
    # per record and region class (0 = FWR, 1 = CDR): R, S, stop counts and expected R probability
    cnt = np.zeros((n, 2, 3), dtype=np.int64)
    p_r = np.full((n, 2), np.nan)
    seqs = air[sequence_alignment].to_numpy(dtype=object)
    gers = air[germline_alignment].to_numpy(dtype=object)
    for a in range(0, n, 5000):
        b = min(n, a + 5000)
        q8 = _matrix(seqs[a:b])
        g8 = _matrix(gers[a:b])
        q8 = np.pad(q8, ((0, 0), (0, max(0, 3 * ncod - q8.shape[1]))))[:, : 3 * ncod]
        g8 = np.pad(g8, ((0, 0), (0, max(0, 3 * ncod - g8.shape[1]))))[:, : 3 * ncod]
        q = _CODE[q8].reshape(b - a, ncod, 3).astype(np.int64)
        g = _CODE[g8].reshape(b - a, ncod, 3).astype(np.int64)
        full = (q >= 0).all(2) & (g >= 0).all(2)
        gi = np.where(full, g[..., 0] * 16 + g[..., 1] * 4 + g[..., 2], 0)
        full &= _AA_ARR[gi] != "*"
        er = np.where(full, w_r[gi], 0.0)
        es = np.where(full, w_s[gi], 0.0)
        for k, mask in ((0, ~is_cdr), (1, is_cdr)):
            tot = (er[:, mask] + es[:, mask]).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                p_r[a:b, k] = np.where(tot > 0, er[:, mask].sum(1) / tot, np.nan)
        # each mutated base: germline codon with only that base changed
        for pos in range(3):
            mut = full & (q[..., pos] != g[..., pos])
            mc = g.copy()
            mc[..., pos] = np.where(mut, q[..., pos], g[..., pos])
            mi = mc[..., 0] * 16 + mc[..., 1] * 4 + mc[..., 2]
            aa_m, aa_g = _AA_ARR[mi], _AA_ARR[gi]
            stop = mut & (aa_m == "*")
            rep = mut & ~stop & (aa_m != aa_g)
            sil = mut & ~stop & (aa_m == aa_g)
            for k, mask in ((0, ~is_cdr), (1, is_cdr)):
                cnt[a:b, k, 0] += rep[:, mask].sum(1)
                cnt[a:b, k, 1] += sil[:, mask].sum(1)
                cnt[a:b, k, 2] += stop[:, mask].sum(1)
    rows, nulls = [], []
    for gi_, g in enumerate(groups):
        sel = gcode == gi_
        for k, region in ((1, "CDR"), (0, "FWR")):
            r_ = cnt[sel, k, 0]
            s_ = cnt[sel, k, 1]
            stop_ = cnt[sel, k, 2]
            m_ = r_ + s_
            p_ = np.nan_to_num(p_r[sel, k], nan=0.0)
            use = m_ > 0
            R, S_, M = int(r_.sum()), int(s_.sum()), int(m_.sum())
            exp_r = float((m_[use] * p_[use]).sum())
            exp_s = M - exp_r
            draws = np.zeros(n_perm)
            mm, pp = m_[use], p_[use]
            for c0 in range(0, len(mm), 20_000):
                draws += gen.binomial(
                    mm[c0 : c0 + 20_000], pp[c0 : c0 + 20_000], size=(n_perm, len(mm[c0 : c0 + 20_000]))
                ).sum(1)
            nulls.append(draws)
            mu = float(draws.mean())
            sd = float(draws.std(ddof=1)) if n_perm > 1 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma = (
                    float(np.log((R + 0.5) / (S_ + 0.5)) - np.log(exp_r / exp_s)) if exp_r > 0 and exp_s > 0 else np.nan
                )
            rows.append(
                {
                    "group": g,
                    "region": region,
                    "n_records": int((sel & (cnt[:, k, :2].sum(1) >= 0)).sum()),
                    "mutations": M,
                    "R": R,
                    "S": S_,
                    "stop": int(stop_.sum()),
                    "share_R": R / M if M else np.nan,
                    "expected_share_R": exp_r / M if M else np.nan,
                    "statistic": float(R),
                    "expected": exp_r,
                    "null_mean": mu,
                    "null_sd": sd,
                    "z": (R - mu) / sd if sd and sd > 0 else np.nan,
                    "sigma": sigma,
                    "p_value": float(mc_pvalue(draws, float(R), alternative=alternative)[0]) if M else np.nan,
                    "n_perm": n_perm,
                }
            )
    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("n_perm"), "q_value", bh(table.p_value.to_numpy()))
    summary: dict[str, Any] = {}
    for _, r in table.iterrows():
        tag = r.region if len(groups) == 1 else f"{r.group}:{r.region}"
        summary[f"sigma[{tag}]"] = r.sigma
        summary[f"p[{tag}]"] = r.p_value
    params = {
        "sequence_alignment": sequence_alignment,
        "germline_alignment": germline_alignment,
        "kappa": kappa,
        "by": by,
        "n_perm": n_perm,
        "alternative": alternative,
        "design": "R ~ sum of per-record binomials (expected R share from germline codons, uniform targeting)",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(
        name="selection",
        table=table,
        summary=summary,
        params=params,
        null=np.column_stack(nulls) if keep_null and nulls else None,
    )
