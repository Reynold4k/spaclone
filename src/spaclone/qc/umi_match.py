"""UMI match: is each receptor molecule's (spot barcode, UMI) found in the GEX library, as its own constant gene?

When the receptor library and the GEX library are made from the same barcoded cDNA (Curio / AIR-SPACE beads,
Stereo-seq with molecule identifiers), a genuine receptor molecule keeps the barcode and UMI of its own mRNA. If that
cDNA was also sequenced in GEX, the GEX read with the same (barcode, UMI) maps to the molecule's own constant-region
gene (Igkc for an IGK molecule, Ighg2c for an IGHG2C one). A molecule that took over the barcode and UMI of another
cDNA in a PCR chimera matches whatever gene that cDNA was; a chimera with a new (barcode, UMI) matches nothing more
often than a random key. *Decoy* keys — UMIs that no receptor molecule carries, on the same barcode — measure the
chance of a match given the bead's GEX depth and make-up, and pair each molecule with its own null.

References
----------
Dixit 2016, doi:10.1101/093237 and Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x (molecule keys shared
between libraries); Shugay et al. 2014, doi:10.1038/nmeth.2960 (UMI-tagged repertoires); Fu et al. 2011,
doi:10.1073/pnas.1017621108 (UMI collisions); Jiang et al. 2026, doi:10.1073/pnas.2504742123 (AIR-SPACE: receptor
and GEX libraries from the same barcoded beads).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from spaclone import _schema as S
from spaclone._utils import RngLike, TestResult, as_rng, bh, rng_entropy
from spaclone.qc.same_spot import HUMAN_CONSTANT_GENES, MOUSE_CONSTANT_GENES, discrete_sum_sf

__all__ = ["UNDETERMINED_GENES", "UmiMatchResult", "umi_collisions", "umi_decoys", "umi_match"]

#: Gene labels of GEX reads whose gene could not be called (not used to pick a key's gene).
UNDETERMINED_GENES: tuple[str, ...] = ("unmapped", "multi", "low_complexity")
_ACGT = np.array(list("ACGT"))
_VJ = {
    "BCR": re.compile(r"^IG[HKL][VJ]", re.IGNORECASE),
    "TCR": re.compile(r"^(TR[ABGD]|TCR[ABGD]-?)[VJ]", re.IGNORECASE),
}
_CATEGORIES = ("own_constant", "other_constant", "v_or_j", "other_gene", "undetermined")


def umi_collisions(n_molecules: Any, umi_length: int) -> dict[str, np.ndarray]:
    """Expected UMI collisions among ``M`` molecules of one barcode with random UMIs of length ``L``.

    Parameters
    ----------
    n_molecules
        Molecules per barcode (scalar or array).
    umi_length
        UMI length in nucleotides.

    Returns
    -------
    dict
        ``pairs``: expected pairs of molecules sharing a UMI, ``M (M − 1) / (2 · 4^L)`` (≈ ``M² / (2 · 4^L)``);
        ``p_key``: probability that a given foreign key on the barcode meets at least one of the ``M`` molecules,
        ``1 − (1 − 4^−L)^M`` (the chance match rate of a decoy).

    References
    ----------
    Fu et al. 2011, doi:10.1073/pnas.1017621108.
    """
    m = np.asarray(n_molecules, dtype=float)
    space = 4.0 ** int(umi_length)
    return {"pairs": m * (m - 1) / (2 * space), "p_key": -np.expm1(m * np.log1p(-1.0 / space))}


def _valid_umi(u: pd.Series, length: int) -> np.ndarray:
    s = u.astype("string").str.upper()
    return (s.str.len() == length).fillna(False).to_numpy(bool) & s.str.fullmatch("[ACGT]+").fillna(False).to_numpy(
        bool
    )


def _air_keys(
    air: pd.DataFrame, umi_length: int | None, barcode_key: str, umi_key: str, locus_key: str, c_call_key: str
) -> tuple[pd.DataFrame, int, int]:
    S.validate_air(air, required=(barcode_key, umi_key, locus_key), numeric=())
    if umi_length is None:
        lens = air[umi_key].dropna().astype(str).str.len()
        if len(lens) == 0:
            raise ValueError(f"column {umi_key!r} has no UMIs")
        umi_length = int(lens.mode().iloc[0])
    ok = _valid_umi(air[umi_key], umi_length)
    cols = [barcode_key, umi_key, locus_key] + ([c_call_key] if c_call_key in air.columns else [])
    k = air.loc[ok, cols].copy()
    k[umi_key] = k[umi_key].astype(str).str.upper()
    k[barcode_key] = k[barcode_key].astype(str)
    k = k.drop_duplicates([barcode_key, umi_key, locus_key], keep="first", ignore_index=True)
    if c_call_key not in k.columns:
        k[c_call_key] = pd.NA
    k = k.rename(columns={barcode_key: "barcode", umi_key: "umi", locus_key: "locus", c_call_key: "c_call"})
    k["locus"] = k["locus"].astype(str)
    return k, int(umi_length), int((~ok).sum())


def umi_decoys(
    air: pd.DataFrame,
    *,
    n_decoys: int = 2,
    method: Literal["shuffle", "random"] = "shuffle",
    umi_length: int | None = None,
    rng: RngLike = None,
    barcode_key: str = S.SPOT,
    umi_key: str = S.UMI,
    locus_key: str = S.LOCUS,
    c_call_key: str = S.C_CALL,
) -> pd.DataFrame:
    """Decoy (barcode, UMI) keys: UMIs that no receptor molecule carries, on the barcodes of the receptor molecules.

    Parameters
    ----------
    air
        Receptor table with barcode (``spot_id``), ``umi``, ``locus`` and optionally ``c_call`` columns.
    n_decoys
        Decoys per receptor key.
    method
        ``"shuffle"`` (default): UMIs of receptor molecules on other barcodes, so decoys share the real UMIs' base
        composition and biases (poly-T and the like), which keeps them exchangeable with the receptor keys under the
        null. ``"random"``: uniform random UMIs of the same length (the archived AIR-SPACE analysis).
    umi_length
        UMI length; default the most common length. Keys with other lengths or non-ACGT characters are dropped.
    rng
        Seed or generator.
    barcode_key, umi_key, locus_key, c_call_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        ``barcode``, ``umi``, ``locus``, ``c_call`` (inherited from the receptor key), ``parent`` (row of the receptor
        key in the deduplicated key table, which defines the matched stratum of the test), ``kind = "decoy"``.
        A decoy never equals a receptor key or another decoy of its barcode; a key whose decoys cannot all be drawn
        uniquely within 50 attempts keeps fewer.
    """
    if n_decoys < 1:
        raise ValueError("n_decoys must be >= 1")
    if method not in ("shuffle", "random"):
        raise ValueError(f"method must be 'shuffle' or 'random', got {method!r}")
    gen = as_rng(rng)
    keys, length, _ = _air_keys(air, umi_length, barcode_key, umi_key, locus_key, c_call_key)
    n = len(keys)
    if n == 0:
        return pd.DataFrame(columns=["barcode", "umi", "locus", "c_call", "parent", "kind"])
    taken = set(zip(keys["barcode"], keys["umi"], strict=True))
    bc = keys["barcode"].to_numpy()
    pool = keys["umi"].to_numpy()
    parent = np.repeat(np.arange(n), n_decoys)
    want = np.ones(len(parent), dtype=bool)
    umis = np.empty(len(parent), dtype=object)
    for _ in range(50):
        idx = np.flatnonzero(want)
        if len(idx) == 0:
            break
        if method == "random":
            draw = ["".join(r) for r in _ACGT[gen.integers(0, 4, (len(idx), length))]]
        else:
            draw = pool[gen.integers(0, n, len(idx))]
        for i, u in zip(idx, draw, strict=True):
            key = (bc[parent[i]], u)
            if key not in taken:
                taken.add(key)
                umis[i] = u
                want[i] = False
    got = ~want
    out = keys.iloc[parent[got]][["barcode", "locus", "c_call"]].reset_index(drop=True)
    out.insert(1, "umi", umis[got])
    out["parent"] = parent[got]
    out["kind"] = "decoy"
    return out


@dataclass(frozen=True)
class UmiMatchResult(TestResult):
    """Result of :func:`umi_match`.

    Attributes
    ----------
    categories
        One row per kind (``"air"`` / ``"decoy"``) × locus: ``keys``, ``matched``, ``frac_matched``,
        ``frac_matched_own`` (own constant gene among all keys) and, among matched keys, ``of_matched_own_constant``,
        ``of_matched_other_constant``, ``of_matched_v_or_j``, ``of_matched_other_gene``, ``of_matched_undetermined``;
        ``expected_chance_match`` (mean UMI-collision probability of the keys, when barcode depths are given).
    keys
        One row per key: ``barcode``, ``umi``, ``locus``, ``c_call``, ``kind``, ``stratum``, ``gex_reads``,
        ``top_gene``, ``matched``, ``category``, ``own`` and ``p_chance`` (when barcode depths are given).
    """

    categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    keys: pd.DataFrame = field(default_factory=pd.DataFrame)


def _gene_tables(
    constant_genes: Mapping[str, Sequence[str]] | None, genes_seen: set[str], loci: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    if constant_genes is not None:
        return {str(k): tuple(v) for k, v in constant_genes.items()}
    mouse = sum(g in genes_seen for lc in loci for g in MOUSE_CONSTANT_GENES.get(lc, ()))
    human = sum(g in genes_seen for lc in loci for g in HUMAN_CONSTANT_GENES.get(lc, ()))
    return dict(HUMAN_CONSTANT_GENES if human > mouse else MOUSE_CONSTANT_GENES)


def _receptor_class(locus: str) -> str:
    return "TCR" if str(locus).upper().startswith("TR") else "BCR"


def umi_match(
    air: pd.DataFrame,
    gex_reads: pd.DataFrame,
    *,
    constant_genes: Mapping[str, Sequence[str]] | None = None,
    isotype_genes: Mapping[str, Sequence[str]] | None = None,
    own_genes: Literal["isotype_or_locus", "isotype", "locus"] = "isotype_or_locus",
    n_decoys: int = 2,
    decoys: Literal["shuffle", "random"] | pd.DataFrame = "shuffle",
    umi_length: int | None = None,
    undetermined: Sequence[str] = UNDETERMINED_GENES,
    barcode_umis: Mapping[str, float] | pd.Series | None = None,
    rng: RngLike = None,
    barcode_key: str = S.SPOT,
    umi_key: str = S.UMI,
    locus_key: str = S.LOCUS,
    c_call_key: str = S.C_CALL,
    gene_key: str = "gene",
    count_key: str | None = None,
) -> UmiMatchResult:
    """Look every receptor molecule's (barcode, UMI) up in the GEX reads, beside decoy UMIs on the same barcode.

    Parameters
    ----------
    air
        Receptor table, one row per molecule: barcode (``spot_id``), ``umi``, ``locus`` and ``c_call`` (isotype or
        constant gene, e.g. ``IGHG2C``; optional). Molecules are deduplicated by (barcode, UMI, locus).
    gex_reads
        GEX reads with columns ``barcode_key``, ``umi_key`` and ``gene_key`` (the gene each read maps to; labels in
        ``undetermined`` for reads without a gene call), optionally ``count_key`` (reads per row). May hold every
        GEX read or only those whose (barcode, UMI) is a receptor or decoy key; row order breaks ties when picking a
        key's gene (the gene seen first wins, as in the archived analysis).
    constant_genes
        ``{locus: genes}``; default the mouse or human table of :func:`spaclone.qc.same_spot`, whichever matches more
        genes in ``gex_reads``.
    isotype_genes
        ``{c_call: genes}``; default: a ``c_call`` (allele suffix removed) names the constant gene of its locus with
        the same symbol, ignoring case (``IGHG2C`` → ``Ighg2c``, ``IGKC`` → ``Igkc``).
    own_genes
        A key's *own* constant gene: ``"isotype_or_locus"`` (default) the gene of its ``c_call``, or any constant gene
        of its locus when the call is missing or unknown; ``"isotype"`` only the ``c_call`` gene (keys without a call
        cannot match their own gene; the archived analysis); ``"locus"`` any constant gene of the locus.
    n_decoys
        Decoys per receptor key when generated.
    decoys
        ``"shuffle"`` / ``"random"`` generate decoys with :func:`umi_decoys`; a DataFrame with ``barcode``, ``umi``,
        ``locus`` (and optionally ``c_call``, ``parent``) supplies them.
    umi_length
        UMI length; default the most common length among the receptor UMIs.
    undetermined
        Gene labels that do not count as a gene call.
    barcode_umis
        GEX molecules (distinct UMIs) per barcode, e.g. ``gex.obs["total_counts"]``; adds the UMI-collision chance
        ``1 − (1 − 4^−L)^M`` of every key (:func:`umi_collisions`).
    rng
        Seed or generator for the decoys.
    barcode_key, umi_key, locus_key, c_call_key, gene_key, count_key
        Column names.

    Returns
    -------
    UmiMatchResult
        ``table``: one row per locus × outcome (``"matched"``: the key is found in GEX; ``"own_constant"``: its GEX
        reads call the key's own constant gene) with ``n_keys``, ``statistic`` (receptor keys with the outcome),
        ``expected`` and ``null_sd`` (under the stratified null), ``z``, ``p_value`` (exact, one-sided), ``q_value``,
        ``air_rate``, ``decoy_rate``, ``excess_rate``. ``categories`` and ``keys`` as in :class:`UmiMatchResult`.

    Notes
    -----
    Null model: within each stratum — a receptor key and its own decoys (``parent``), or all keys of one barcode
    when supplied decoys carry no parent — the labels "receptor" and "decoy" are exchangeable: a receptor UMI is no
    more likely to meet a GEX molecule (or one of its own constant gene) than a decoy UMI on the same barcode. The
    number of receptor keys with the outcome is then a sum of independent hypergeometric variables, one per stratum,
    whose exact distribution gives the p-value; positions, GEX reads and outcomes are fixed. Uniform random decoys
    are exchangeable with real UMIs only if UMIs are uniform; shuffled decoys keep the real UMI composition.

    References
    ----------
    Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x; Mantel & Haenszel 1959, doi:10.1093/jnci/22.4.719
    (stratified 2 × 2 tables); Fu et al. 2011, doi:10.1073/pnas.1017621108.
    """
    if own_genes not in ("isotype_or_locus", "isotype", "locus"):
        raise ValueError("own_genes must be 'isotype_or_locus', 'isotype' or 'locus'")
    gen = as_rng(rng)
    keys, length, n_bad = _air_keys(air, umi_length, barcode_key, umi_key, locus_key, c_call_key)
    keys["kind"] = "air"
    keys["parent"] = np.arange(len(keys))
    if isinstance(decoys, pd.DataFrame):
        dec = decoys.rename(columns={barcode_key: "barcode", umi_key: "umi", locus_key: "locus", c_call_key: "c_call"})
        missing = [c for c in ("barcode", "umi", "locus") if c not in dec.columns]
        if missing:
            raise ValueError(f"decoys lack column(s) {missing}")
        dec = dec.copy()
        if "c_call" not in dec.columns:
            dec["c_call"] = pd.NA
        has_parent = "parent" in dec.columns
        if not has_parent:
            dec["parent"] = -1
        dec["kind"] = "decoy"
        dec["barcode"] = dec["barcode"].astype(str)
        dec["umi"] = dec["umi"].astype(str).str.upper()
        dec["locus"] = dec["locus"].astype(str)
        decoy_source = "supplied"
    else:
        dec = umi_decoys(
            air, n_decoys=n_decoys, method=decoys, umi_length=length, rng=gen, barcode_key=barcode_key,
            umi_key=umi_key, locus_key=locus_key, c_call_key=c_call_key,
        )  # fmt: skip
        has_parent = True
        decoy_source = decoys
    allk = pd.concat([keys, dec[["barcode", "umi", "locus", "c_call", "kind", "parent"]]], ignore_index=True)
    # ------------------------------------------------------------------ reads per key and gene
    for c in (barcode_key, umi_key, gene_key):
        if c not in gex_reads.columns:
            raise ValueError(f"gex_reads lacks column {c!r}")
    rd = pd.DataFrame(
        {
            "barcode": gex_reads[barcode_key].astype(str).to_numpy(),
            "umi": gex_reads[umi_key].astype(str).str.upper().to_numpy(),
            "gene": gex_reads[gene_key].astype(str).to_numpy(),
            "w": gex_reads[count_key].to_numpy(dtype=float) if count_key else np.ones(len(gex_reads)),
            "pos": np.arange(len(gex_reads)),
        }
    )
    kk = allk[["barcode", "umi"]].drop_duplicates()
    rd = rd.merge(kk, on=["barcode", "umi"], how="inner")
    per_gene = (
        rd.groupby(["barcode", "umi", "gene"], sort=False).agg(n=("w", "sum"), first=("pos", "min")).reset_index()
    )
    reads = per_gene.groupby(["barcode", "umi"], sort=False)["n"].sum().rename("gex_reads")
    det = per_gene[~per_gene["gene"].isin(list(undetermined))]
    top = (
        det.sort_values(["barcode", "umi", "n", "first"], ascending=[True, True, False, True], kind="stable")
        .drop_duplicates(["barcode", "umi"])
        .set_index(["barcode", "umi"])["gene"]
        .rename("top_gene")
    )
    allk = allk.merge(reads, left_on=["barcode", "umi"], right_index=True, how="left")
    allk = allk.merge(top, left_on=["barcode", "umi"], right_index=True, how="left")
    allk["matched"] = allk["gex_reads"].fillna(0) > 0
    allk["gex_reads"] = allk["gex_reads"].fillna(0.0)
    # ------------------------------------------------------------------ categories
    loci = sorted(pd.unique(allk["locus"]), key=str)
    genes_seen = set(per_gene["gene"].unique())
    cgenes = _gene_tables(constant_genes, genes_seen, loci)
    by_class: dict[str, set[str]] = {"BCR": set(), "TCR": set()}
    for lc, gs in cgenes.items():
        by_class[_receptor_class(lc)].update(gs)
    lookup: dict[str, tuple[str, ...]] = {}
    if isotype_genes is not None:
        lookup = {str(k).upper(): tuple(v) for k, v in isotype_genes.items()}
    else:
        for gs in cgenes.values():
            for g in gs:
                lookup.setdefault(g.upper(), (g,))
    own = np.zeros(len(allk), dtype=bool)
    cat = np.full(len(allk), "unmatched", dtype=object)
    tg = allk["top_gene"].to_numpy(dtype=object)
    for i, (lc, cc, m, g) in enumerate(zip(allk["locus"], allk["c_call"], allk["matched"], tg, strict=True)):
        if not m:
            continue
        if not isinstance(g, str):
            cat[i] = "undetermined"
            continue
        locus_set = set(cgenes.get(lc, ()))
        call = str(cc).split("*")[0].strip().upper() if isinstance(cc, str) and cc else ""
        iso = set(lookup.get(call, ())) if call else set()
        if own_genes == "locus":
            mine = locus_set
        elif own_genes == "isotype":
            mine = iso
        else:
            mine = iso if iso else locus_set
        cls = _receptor_class(lc)
        if g in mine:
            own[i] = True
            cat[i] = "own_constant"
        elif g in by_class[cls]:
            cat[i] = "other_constant"
        elif _VJ[cls].match(g):
            cat[i] = "v_or_j"
        else:
            cat[i] = "other_gene"
    allk["category"] = cat
    allk["own"] = own
    allk["top_gene"] = allk["top_gene"].where(allk["matched"], None)
    if barcode_umis is not None:
        depth = pd.Series(barcode_umis, dtype=float)
        depth.index = depth.index.astype(str)
        m_b = allk["barcode"].map(depth).to_numpy(dtype=float)
        allk["p_chance"] = umi_collisions(m_b, length)["p_key"]
    # ------------------------------------------------------------------ strata and exact tests
    if has_parent:
        stratum = allk["parent"].to_numpy()
    else:
        stratum = pd.factorize(allk["barcode"].astype(str) + "\x00" + allk["locus"].astype(str))[0]
    allk["stratum"] = stratum
    rows = []
    for lc in loci:
        sub = allk[allk["locus"] == lc]
        is_air = (sub["kind"] == "air").to_numpy()
        for outcome, col in (("matched", "matched"), ("own_constant", "own")):
            y = sub[col].to_numpy(dtype=bool)
            st = pd.factorize(sub["stratum"])[0]
            n_s = np.bincount(st)
            a_s = np.bincount(st, weights=is_air).astype(np.int64)
            k_s = np.bincount(st, weights=y).astype(np.int64)
            t_obs = int((y & is_air).sum())
            use = a_s > 0
            n_s, a_s, k_s = n_s[use], a_s[use], k_s[use]
            mean = float((a_s * k_s / n_s).sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                var_s = np.where(
                    n_s > 1, a_s * (k_s / n_s) * (1 - k_s / n_s) * (n_s - a_s) / np.maximum(n_s - 1, 1), 0.0
                )
            sd = float(np.sqrt(var_s.sum()))
            width = int(a_s.max()) + 1 if len(a_s) else 1
            jj = np.arange(width)
            pmf = stats.hypergeom.pmf(jj[None, :], n_s[:, None], k_s[:, None], a_s[:, None]) if len(a_s) else None
            p = discrete_sum_sf(pmf, t_obs) if pmf is not None else np.nan
            n_air = int(is_air.sum())
            n_dec = int((~is_air).sum())
            air_rate = t_obs / n_air if n_air else np.nan
            dec_rate = float((y & ~is_air).sum()) / n_dec if n_dec else np.nan
            rows.append(
                {
                    "locus": lc,
                    "outcome": outcome,
                    "n_keys": n_air,
                    "n_decoys": n_dec,
                    "statistic": t_obs,
                    "expected": mean,
                    "null_sd": sd,
                    "z": (t_obs - mean) / sd if sd > 0 else np.nan,
                    "p_value": p,
                    "air_rate": air_rate,
                    "decoy_rate": dec_rate,
                    "excess_rate": air_rate - dec_rate,
                }
            )
    table = pd.DataFrame(rows)
    if len(table):
        table.insert(table.columns.get_loc("p_value") + 1, "q_value", bh(table["p_value"].to_numpy()))
    # ------------------------------------------------------------------ X4-style summary
    crows = []
    for (kind, lc), g in allk.groupby(["kind", "locus"], sort=True):
        m = g[g["matched"]]
        row = {
            "kind": kind,
            "locus": lc,
            "keys": len(g),
            "matched": len(m),
            "frac_matched": len(m) / len(g) if len(g) else np.nan,
            "frac_matched_own": float(g["own"].mean()) if len(g) else np.nan,
        }
        for c in _CATEGORIES:
            row[f"of_matched_{c}"] = float((m["category"] == c).mean()) if len(m) else np.nan
        if "p_chance" in g.columns:
            row["expected_chance_match"] = float(np.nanmean(g["p_chance"])) if g["p_chance"].notna().any() else np.nan
        crows.append(row)
    categories = pd.DataFrame(crows)
    summary: dict[str, Any] = {}
    for _, r in table.iterrows():
        summary[f"{r.outcome}[{r.locus}]"] = f"{r.air_rate:.3f} vs decoy {r.decoy_rate:.3f}"
    summary["n_dropped_umi"] = n_bad
    params = {
        "constant_genes": {k: list(v) for k, v in cgenes.items()},
        "own_genes": own_genes,
        "n_decoys": n_decoys,
        "decoys": decoy_source,
        "strata": "parent key" if has_parent else "barcode x locus",
        "umi_length": length,
        "undetermined": list(undetermined),
        "rng_entropy": rng_entropy(gen),
        "null": "receptor and decoy labels exchangeable within strata (a key and its decoys, or one barcode); "
        "receptor keys with the outcome ~ sum of independent hypergeometric variables",
    }
    return UmiMatchResult(
        name="umi_match", table=table, summary=summary, params=params, categories=categories, keys=allk
    )
