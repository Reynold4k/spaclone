"""Off-tissue diagnostics: where do receptor molecules sit relative to the tissue, and what are the off-tissue ones?

Off-tissue spots hold no cells, so like empty droplets they show what the noise looks like:

* :func:`density_profile` — records per µm² against the signed distance to the tissue edge, beside the GEX (cDNA)
  profile. mRNA diffusion decays within tens of µm; barcode-swap chimeras follow captured cDNA wherever it is.
* :func:`composition_on_off` — clone composition on tissue against far off tissue. Diffusion carries the clones of
  the adjacent tissue, chimeras the global pool; a log clone-frequency correlation near 1 with the far field means the
  off-tissue records are a sample of the whole library.
* :func:`swap_rate` — the rate law ``E[records(b)] = ρ · D_b`` (records of a locus in bin ``b`` against its GEX
  molecules ``D_b``) fitted on far off-tissue bins, where every record is an artefact, with the excess on tissue that
  the rate leaves for genuine records.
* :func:`tissue_zones` — the zone labels of the archived analysis (tissue core, edge, near / mid / far off tissue).

References
----------
Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x (swapped molecules estimated from impossible combinations);
Kebschull & Zador 2015, doi:10.1093/nar/gkv717 (PCR template switching); Fleming et al. 2023,
doi:10.1038/s41592-023-01943-7 (swap term proportional to the barcode's molecules); Ni et al. 2022,
doi:10.1038/s41467-022-30587-y (spot swapping in spatial transcriptomics).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spaclone import _schema as S
from spaclone._utils import RngLike, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy

__all__ = [
    "DEFAULT_SHELLS_UM",
    "TISSUE_ZONE_EDGES_UM",
    "TISSUE_ZONE_LABELS",
    "SwapRateResult",
    "composition_on_off",
    "density_profile",
    "swap_rate",
    "tissue_zones",
]

#: Signed-distance shells (µm, half-open ``[lo, hi)``) of the archived density profile (03 B1).
DEFAULT_SHELLS_UM: tuple[float, ...] = (
    -1500, -500, -200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000,
    3000, 6000,
)  # fmt: skip
#: Upper edges (µm, right-closed) of the archived tissue zones (03 B2).
TISSUE_ZONE_EDGES_UM: tuple[float, ...] = (-50.0, 0.0, 50.0, 300.0)
#: Labels of the archived tissue zones: ``d ≤ −50``, ``−50 < d ≤ 0``, ``0 < d ≤ 50``, ``50 < d ≤ 300``, ``d > 300``.
TISSUE_ZONE_LABELS: tuple[str, ...] = ("in_core", "in_edge", "out_near", "out_mid", "out_far")


def tissue_zones(
    distance_um: Any,
    *,
    edges_um: Sequence[float] = TISSUE_ZONE_EDGES_UM,
    labels: Sequence[str] = TISSUE_ZONE_LABELS,
) -> pd.Categorical:
    """Label signed distances to the tissue edge (negative inside) by zone.

    Parameters
    ----------
    distance_um
        Signed distances in µm (e.g. from :class:`spaclone.io.DistanceGrid`).
    edges_um
        Increasing upper edges; zone ``k`` holds ``edges[k − 1] < d ≤ edges[k]`` (right-closed, as archived), the first
        zone ``d ≤ edges[0]`` and the last ``d > edges[-1]``.
    labels
        One label more than edges.

    Returns
    -------
    pandas.Categorical
        Ordered zone labels; NaN distances give missing labels.
    """
    e = np.asarray(edges_um, dtype=float)
    if len(labels) != len(e) + 1:
        raise ValueError("labels must have one entry more than edges_um")
    if np.any(np.diff(e) <= 0):
        raise ValueError("edges_um must be strictly increasing")
    d = np.asarray(distance_um, dtype=float)
    codes = np.searchsorted(e, d, side="left").astype(np.int64)
    codes[~np.isfinite(d)] = -1
    return pd.Categorical.from_codes(codes, categories=list(labels), ordered=True)


def _distance_of(air: pd.DataFrame, distance: Any, x_key: str, y_key: str) -> np.ndarray:
    if isinstance(distance, str):
        if distance not in air.columns:
            raise KeyError(f"distance column {distance!r} not in the table")
        return air[distance].to_numpy(dtype=float)
    if callable(distance):
        return np.asarray(distance(air[x_key].to_numpy(dtype=float), air[y_key].to_numpy(dtype=float)), dtype=float)
    raise TypeError("distance must be a DistanceGrid (or any callable of x_um, y_um) or the name of a column")


def _shell_index(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, d, side="right") - 1
    idx[(idx < 0) | (idx >= len(edges) - 1) | ~np.isfinite(d)] = -1
    return idx


def density_profile(
    air: pd.DataFrame,
    distance: Any,
    *,
    edges_um: Sequence[float] = DEFAULT_SHELLS_UM,
    group: str | None = S.LOCUS,
    weight: str | None = None,
    gex: Any = None,
    gex_genes: Mapping[str, Sequence[str]] | None = None,
    gex_bin_um: float = 10.0,
    core_max_um: float = -100.0,
    report_at_um: Sequence[float] = (1750.0,),
    x_key: str = S.X,
    y_key: str = S.Y,
) -> TestResult:
    """Records per µm² against the signed distance to the tissue edge, optionally beside the GEX (cDNA) profile.

    Parameters
    ----------
    air
        Receptor table (one row per molecule or (spot, clone) unit; drop barcode sinks first).
    distance
        :class:`spaclone.io.DistanceGrid` of the chip: gives each record's signed distance and the area of every
        shell (the whole measured area, on and off tissue).
    edges_um
        Shell edges (µm); shells are half-open ``[lo, hi)``. Records and area outside the edges are ignored.
    group
        Column splitting the records (default the locus); ``None`` for one group ``"all"``.
    weight
        Optional column summed instead of counting records (e.g. ``"consensus_count"`` for read density).
    gex
        Optional :class:`spaclone.qc.GexLookup` of the whole chip: GEX molecules per µm² by shell (``group``
        ``"gex:total"``), counted on bins of ``gex_bin_um`` placed by their centre, as archived.
    gex_genes
        ``{name: genes}`` for further GEX profiles (``group`` ``"gex:<name>"``; genes must be loaded in ``gex``).
    gex_bin_um
        Bin side for the GEX profile (10 µm = Stereo-seq bin20, as archived).
    core_max_um
        Shells whose midpoint lies below this distance form the tissue core; ``relative`` is the density divided by
        the core density of the same group (area-weighted mean over the core shells).
    report_at_um
        Distances whose shell's relative density goes into the summary (e.g. 1750 µm off tissue).
    x_key, y_key
        Coordinate columns.

    Returns
    -------
    TestResult
        ``table`` (tidy, one row per source × group × shell): ``source`` (``"air"`` / ``"gex"``), ``group``,
        ``shell_lo``, ``shell_hi``, ``shell_mid``, ``n`` (records, weights or molecules), ``area_um2``, ``density``
        (per µm²), ``relative``. ``summary``: core density per group and the relative density at ``report_at_um``.

    Notes
    -----
    Descriptive, no null model. Barcode-swap artefacts follow captured cDNA, so their profile parallels the GEX
    profile; diffusion of transcripts out of the section decays within tens of µm (Slide-DNA-seq measures ~25 µm).

    References
    ----------
    Ni et al. 2022, doi:10.1038/s41467-022-30587-y; Zhao et al. 2022, doi:10.1038/s41586-021-04217-4.
    """
    e = np.asarray(edges_um, dtype=float)
    if len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("edges_um must hold at least two strictly increasing values")
    if not hasattr(distance, "area_um2"):
        raise TypeError("distance must be a DistanceGrid (it provides the area of each shell)")
    S.validate_air(air, required=(x_key, y_key) + ((group,) if group else ()), numeric=(x_key, y_key))
    d = _distance_of(air, distance, x_key, y_key)
    sh = _shell_index(d, e)
    area = np.asarray(distance.area_um2(e), dtype=float)
    lo, hi = e[:-1], e[1:]
    mid = (lo + hi) / 2
    w = air[weight].to_numpy(dtype=float) if weight else np.ones(len(air))
    groups = air[group].astype(object).to_numpy() if group else np.full(len(air), "all", dtype=object)
    rows = []
    ok = sh >= 0
    for gname in sorted(pd.unique(groups[ok]), key=str):
        sel = ok & (groups == gname)
        n = np.bincount(sh[sel], weights=w[sel], minlength=len(lo))
        rows.append(pd.DataFrame({"source": "air", "group": str(gname), "shell_lo": lo, "shell_hi": hi,
                                  "shell_mid": mid, "n": n, "area_um2": area}))  # fmt: skip
    if gex is not None:
        sets = dict(gex_genes or {})
        gc = gex.grid_counts(gex_bin_um, sets)
        dc = np.asarray(distance(gc["x_um"].to_numpy() + gex_bin_um / 2, gc["y_um"].to_numpy() + gex_bin_um / 2))
        gsh = _shell_index(dc.astype(float), e)
        gok = gsh >= 0
        bins_per_shell = np.bincount(gsh[gok], minlength=len(lo)).astype(float)
        for name, col in [("total", "total"), *[(k, k) for k in sets]]:
            n = np.bincount(gsh[gok], weights=gc[col].to_numpy(dtype=float)[gok], minlength=len(lo))
            rows.append(pd.DataFrame({"source": "gex", "group": f"gex:{name}", "shell_lo": lo, "shell_hi": hi,
                                      "shell_mid": mid, "n": n, "area_um2": bins_per_shell * gex_bin_um**2}))  # fmt: skip
    table = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["source", "group", "shell_lo", "shell_hi", "shell_mid", "n", "area_um2"])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        table["density"] = table["n"] / table["area_um2"].where(table["area_um2"] > 0)
    core = table["shell_mid"] < core_max_um
    agg = table[core].groupby(["source", "group"], sort=False)[["n", "area_um2"]].sum()
    core_density = (agg["n"] / agg["area_um2"].where(agg["area_um2"] > 0)).to_dict()
    table["relative"] = table["density"] / table.apply(lambda r: core_density.get((r.source, r.group), np.nan), axis=1)
    summary: dict[str, Any] = {f"core_density[{g}]": v for (_, g), v in core_density.items()}
    for at in report_at_um:
        k = int(np.searchsorted(e, at, side="right") - 1)
        if 0 <= k < len(lo):
            for (_src, g), sub in table.groupby(["source", "group"], sort=False):
                summary[f"relative[{g}]@{at:g}um"] = float(sub["relative"].iloc[k])
    params = {
        "edges_um": e.tolist(),
        "group": group,
        "weight": weight,
        "gex_genes": {k: list(v) for k, v in (gex_genes or {}).items()},
        "gex_bin_um": gex_bin_um if gex is not None else None,
        "core_max_um": core_max_um,
        "pixel_um": getattr(distance, "pixel_um", None),
    }
    return TestResult(name="density_profile", table=table, summary=summary, params=params)


# ----------------------------------------------------------------------------------------------- composition


def _drop_sinks(
    air: pd.DataFrame, sink_min_clones: int | None, spot_key: str, locus_key: str, clone_key: str
) -> pd.DataFrame:
    if sink_min_clones is None:
        return air
    from spaclone.pp.sinks import barcode_sinks  # imported here: pp may import qc

    mask = barcode_sinks(
        air, min_clones=sink_min_clones, spot_key=spot_key, locus_key=locus_key, clone_key=clone_key, reads_key=None
    ).mask
    return air[~mask]


def _offtissue_zone(d: np.ndarray, near_um: float, far_um: float) -> np.ndarray:
    z = np.full(len(d), "mid", dtype=object)
    z[d <= 0] = "in"
    z[(d > 0) & (d <= near_um)] = "near"
    z[d > far_um] = "far"
    z[~np.isfinite(d)] = "na"
    return z


def _similarity(na: np.ndarray, nb: np.ndarray, min_total: int, pseudocount: float) -> dict[str, float]:
    """Archived composition similarity: clones with na + nb >= min_total, pseudocounted frequencies."""
    keep = (na + nb) >= min_total
    a, b = na[keep].astype(float), nb[keep].astype(float)
    out = {"n_clones": int(keep.sum()), "pearson_log": np.nan, "spearman": np.nan, "jsd": np.nan}
    if keep.sum() < 3:
        return out
    fa = (a + pseudocount) / (a.sum() + pseudocount * len(a))
    fb = (b + pseudocount) / (b.sum() + pseudocount * len(b))
    out["spearman"] = float(stats.spearmanr(fa, fb).statistic)
    out["pearson_log"] = float(stats.pearsonr(np.log10(fa), np.log10(fb)).statistic)
    pa, pb = fa / fa.sum(), fb / fb.sum()
    m = 0.5 * (pa + pb)
    out["jsd"] = float(0.5 * (np.sum(pa * np.log2(pa / m)) + np.sum(pb * np.log2(pb / m))))
    return out


def _g_stat(tab: np.ndarray) -> np.ndarray:
    """G statistic 2 Σ O log(O / E) of one table (r × 2) or a stack of tables (B × r × 2)."""
    t = np.asarray(tab, dtype=float)
    rows = t.sum(axis=-1, keepdims=True)
    cols = t.sum(axis=-2, keepdims=True)
    tot = t.sum(axis=(-2, -1), keepdims=True)
    exp = rows * cols / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(t > 0, t * np.log(t / exp), 0.0)
    return 2.0 * term.sum(axis=(-2, -1))


def composition_on_off(
    air: pd.DataFrame,
    distance: Any,
    *,
    near_um: float = 50.0,
    far_um: float = 300.0,
    pairs: Sequence[tuple[str, str]] = (("in", "far"),),
    min_total: int = 20,
    pseudocount: float = 0.5,
    n_perm: int = 999,
    sink_min_clones: int | None = 20,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> TestResult:
    """Clone composition on tissue against off tissue: log clone-frequency correlation and a permutation G-test.

    Parameters
    ----------
    air
        Receptor table, one row per molecule or (spot, clone) unit (the exchangeable unit of the test; never reads).
    distance
        :class:`spaclone.io.DistanceGrid` or the name of a column with signed distances (µm, negative inside).
    near_um, far_um
        Zones: ``"in"`` (``d ≤ 0``), ``"near"`` (``0 < d ≤ near_um``), ``"mid"`` and ``"far"`` (``d > far_um``).
    pairs
        Zone pairs compared, e.g. ``(("in", "far"), ("in", "near"))``.
    min_total
        Clones with at least this many records in the two zones enter the correlation (as archived).
    pseudocount
        Added to each clone's count before frequencies are logged (as archived).
    n_perm
        Random tables for the G-test.
    sink_min_clones
        Records on spots with at least this many distinct clones (barcode sinks) are dropped first; ``None`` keeps
        them.
    rng
        Seed or generator.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per locus × pair with ``n_a``, ``n_b`` (records), ``n_clones`` (clones in the
        correlation), ``pearson_log`` (Pearson correlation of log10 frequencies, the archived similarity),
        ``spearman``, ``jsd`` (Jensen–Shannon divergence, bits), ``statistic`` (G over every clone present),
        ``df``, ``null_mean``, ``null_sd``, ``z``, ``p_value`` (Monte-Carlo), ``q_value`` (BH), ``n_perm``.

    Notes
    -----
    Null model: random labelling of the records of a locus between the two zones, conditional on each clone's total in
    the two zones and on the zone totals (Patefield's algorithm draws tables with both margins fixed; positions and
    clones fixed). A small G p-value says the compositions differ beyond multinomial sampling; the correlation says
    by how much. Records of one cell are not independent on tissue, which makes the test anti-conservative there, so
    a non-significant G (off-tissue molecules indistinguishable from on-tissue ones) is the informative outcome.

    References
    ----------
    Patefield 1981, doi:10.2307/2346669; Hothorn et al. 2008, doi:10.18637/jss.v028.i08 (conditional permutation
    tests); Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if near_um <= 0 or far_um < near_um:
        raise ValueError("need 0 < near_um <= far_um")
    zones_ok = {"in", "near", "mid", "far"}
    for a, b in pairs:
        if a not in zones_ok or b not in zones_ok or a == b:
            raise ValueError(f"pairs must name two different zones of {sorted(zones_ok)}, got {(a, b)}")
    req = [spot_key, locus_key, clone_key] + ([] if isinstance(distance, str) else [x_key, y_key])
    S.validate_air(air, required=req, numeric=() if isinstance(distance, str) else (x_key, y_key))
    gen = as_rng(rng)
    data = _drop_sinks(air, sink_min_clones, spot_key, locus_key, clone_key)
    zone = _offtissue_zone(_distance_of(data, distance, x_key, y_key), near_um, far_um)
    loci = sorted(pd.unique(data[locus_key].astype(object)), key=str)
    jobs = [(lc, a, b) for lc in loci for a, b in pairs]
    children = gen.spawn(len(jobs)) if jobs else []
    rows, nulls = [], []
    for (lc, a, b), child in zip(jobs, children, strict=True):
        sel = (data[locus_key].astype(object).to_numpy() == lc) & np.isin(zone, [a, b])
        clones = data[clone_key].to_numpy()[sel]
        za = zone[sel] == a
        codes, _ = pd.factorize(pd.Series(clones), sort=False)
        n_a = np.bincount(codes[za], minlength=codes.max() + 1 if len(codes) else 0)
        n_b = np.bincount(codes[~za], minlength=codes.max() + 1 if len(codes) else 0)
        sim = _similarity(n_a, n_b, min_total, pseudocount)
        row = {locus_key: lc, "zone_a": a, "zone_b": b, "n_a": int(n_a.sum()), "n_b": int(n_b.sum()), **sim}
        if n_a.sum() == 0 or n_b.sum() == 0 or len(n_a) < 2:
            row.update({"statistic": np.nan, "df": 0, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan,
                        "p_value": np.nan, "n_perm": 0})  # fmt: skip
            nulls.append(np.full(n_perm, np.nan))
            rows.append(row)
            continue
        tab = np.column_stack([n_a, n_b])
        g_obs = float(_g_stat(tab))
        rt = stats.random_table(tab.sum(axis=1), tab.sum(axis=0), seed=child)
        null = _g_stat(rt.rvs(size=n_perm, method="patefield"))
        mu, sd, _, _ = null_summary(null)
        p = float(mc_pvalue(null, g_obs, alternative="greater")[0])
        row.update({"statistic": g_obs, "df": int(len(n_a) - 1), "null_mean": float(mu[0]), "null_sd": float(sd[0]),
                    "z": float((g_obs - mu[0]) / sd[0]) if sd[0] > 0 else np.nan, "p_value": p,
                    "n_perm": n_perm})  # fmt: skip
        nulls.append(null)
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table.insert(table.columns.get_loc("p_value") + 1, "q_value", bh(table["p_value"].to_numpy()))
    summary = {f"pearson_log[{r[locus_key]}/{r.zone_a}~{r.zone_b}]": r.pearson_log for _, r in table.iterrows()} | {
        f"p_value[{r[locus_key]}/{r.zone_a}~{r.zone_b}]": r.p_value for _, r in table.iterrows()
    }
    params = {
        "near_um": near_um,
        "far_um": far_um,
        "pairs": [list(p) for p in pairs],
        "min_total": min_total,
        "pseudocount": pseudocount,
        "n_perm": n_perm,
        "sink_min_clones": sink_min_clones,
        "rng_entropy": rng_entropy(gen),
        "null": "records relabelled between the two zones within locus, clone totals and zone totals fixed "
        "(Patefield random tables)",
    }
    null_arr = np.column_stack(nulls) if nulls else None
    return TestResult(name="composition_on_off", table=table, summary=summary, params=params, null=null_arr)


# -------------------------------------------------------------------------------------------------- swap rate


@dataclass(frozen=True)
class SwapRateResult(TestResult):
    """Result of :func:`swap_rate`.

    Attributes
    ----------
    by_depth
        Far off-tissue bins grouped by GEX molecules per bin: records observed against ``ρ̂ · D`` (the rate law
        predicts a ratio of 1 in every class).
    bins
        One row per bin and locus: ``x_um``, ``y_um`` (lower-left corner), ``distance_um`` (centre), ``region``
        (``"far"``, ``"tissue"`` or ``"other"``), ``gex`` (molecules ``D_b``), ``records``.
    """

    by_depth: pd.DataFrame = field(default_factory=pd.DataFrame)
    bins: pd.DataFrame = field(default_factory=pd.DataFrame)


def _poisson_slope(n: np.ndarray, d: np.ndarray) -> tuple[float, float, float, float]:
    """Poisson regression log E[n] = a + b log d by Newton's method; returns (a, b, quasi-Poisson se(b), dispersion)."""
    x = np.column_stack([np.ones(len(d)), np.log(d)])
    beta = np.array([np.log(n.sum() / d.sum()), 1.0])
    for _ in range(100):
        mu = np.exp(x @ beta)
        grad = x.T @ (n - mu)
        hess = (x * mu[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(x @ beta)
    phi = float(np.sum((n - mu) ** 2 / mu) / max(len(n) - 2, 1))
    cov = np.linalg.inv((x * mu[:, None]).T @ x) * max(phi, 1.0)
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[1, 1])), phi


def swap_rate(
    air: pd.DataFrame,
    gex: Any,
    distance: Any,
    *,
    bin_um: float = 25.0,
    far_um: float = 300.0,
    tissue_max_um: float = 0.0,
    sink_min_clones: int | None = 20,
    n_depth_classes: int = 6,
    conf_level: float = 0.95,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> SwapRateResult:
    """Barcode-swap rate from far off-tissue bins, ``E[records(b)] = ρ · D_b``, and the excess it leaves on tissue.

    Parameters
    ----------
    air
        Receptor table, one row per molecule or (spot, clone) unit.
    gex
        :class:`spaclone.qc.GexLookup` of the whole chip (on and off tissue).
    distance
        :class:`spaclone.io.DistanceGrid` of the chip.
    bin_um
        Bin side (a multiple of the GEX spot pitch); bins are aligned at the GEX grid origin and placed by their
        centre.
    far_um
        Bins whose centre lies further than this off tissue fit the rate (no cell, no diffusion from the section).
    tissue_max_um
        Bins whose centre has a signed distance at most this are tissue (0: the whole section; −50: the core).
    sink_min_clones
        Records on spots with at least this many distinct clones are dropped first (``None`` keeps them).
    n_depth_classes
        Quantile classes of GEX molecules per far bin in ``by_depth``.
    conf_level
        Two-sided level of the intervals.
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    SwapRateResult
        ``table``: one row per locus with ``statistic`` (= ``rho``, records per GEX molecule in far bins), ``rho_lo``,
        ``rho_hi`` (quasi-Poisson interval), ``dispersion`` (Pearson χ² / df over far bins), ``n_bins_far``,
        ``records_far``, ``gex_far``, ``records_far_no_gex`` (records in far bins without GEX), ``slope`` with
        ``slope_lo`` / ``slope_hi`` (exponent ``β`` of ``E[records] ∝ D^β``; the rate law needs ``β = 1``),
        ``records_tissue``, ``gex_tissue``, ``expected`` (swap records predicted on tissue, ``ρ̂ · D_tissue``),
        ``excess``, ``genuine_share`` with ``genuine_share_lo`` / ``genuine_share_hi`` (``excess / records``, the
        upper bound clipped at 1), ``ratio_tissue`` (records per GEX molecule on tissue ÷ far), ``expected_power`` and
        ``genuine_share_power`` (the same with the power law ``E = e^α D^β`` fitted on far bins, a sensitivity check
        when ``β ≠ 1``; no interval), ``z``, ``p_value`` (one-sided, more records on tissue than the linear rate
        predicts), ``q_value``. ``by_depth`` and ``bins`` as in :class:`SwapRateResult`.

    Notes
    -----
    Null model: every record is a swap artefact whose expected number in a bin is ``ρ`` times the bin's GEX molecules
    (bimolecular template switching between the receptor pool and the captured cDNA); far off tissue this holds by
    construction, and under the null it holds on tissue too. ``ρ̂ = Σ records / Σ D`` is the Poisson maximum-likelihood
    estimate over far bins, unchanged by the bin size; its interval uses the quasi-Poisson dispersion (at least 1) of
    the bin counts, and the tissue test adds that dispersion to the Poisson variance of the tissue records. If the
    slope ``β`` differs from 1 the rate law is wrong for these data and extrapolating ``ρ̂`` to the cDNA-rich tissue
    is not justified; ``expected_power`` extrapolates the fitted power law instead, which is itself an extrapolation
    beyond the GEX depths seen off tissue.

    References
    ----------
    Kebschull & Zador 2015, doi:10.1093/nar/gkv717; Griffiths et al. 2018, doi:10.1038/s41467-018-05083-x; Fleming
    et al. 2023, doi:10.1038/s41592-023-01943-7; McCullagh & Nelder 1989, doi:10.1007/978-1-4899-3242-6.
    """
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must lie in (0, 1)")
    if far_um <= tissue_max_um:
        raise ValueError("far_um must exceed tissue_max_um")
    S.validate_air(air, required=(spot_key, x_key, y_key, locus_key, clone_key), numeric=(x_key, y_key))
    data = _drop_sinks(air, sink_min_clones, spot_key, locus_key, clone_key)
    grid = gex.grid_counts(bin_um)
    bpx = gex._tile_px(bin_um)
    bx0, by0 = gex.extent[0] // bpx, gex.extent[2] // bpx
    nby = gex.extent[3] // bpx - by0 + 1
    nbx = gex.extent[1] // bpx - bx0 + 1
    dc = np.asarray(distance(grid["x_um"].to_numpy() + bin_um / 2, grid["y_um"].to_numpy() + bin_um / 2), dtype=float)
    region = np.where(dc > far_um, "far", np.where(dc <= tissue_max_um, "tissue", "other"))
    dgex = grid["total"].to_numpy(dtype=float)
    ix, iy = gex._grid(data[x_key].to_numpy(dtype=float), data[y_key].to_numpy(dtype=float))
    bx, by = ix // bpx - bx0, iy // bpx - by0
    inside = (bx >= 0) & (bx < nbx) & (by >= 0) & (by < nby)
    lin = np.where(inside, bx * nby + by, -1)
    z = float(stats.norm.ppf(0.5 + conf_level / 2))
    loci = sorted(pd.unique(data[locus_key].astype(object)), key=str)
    far, tis = region == "far", region == "tissue"
    rows, depth_rows, bin_parts = [], [], []
    for lc in loci:
        sel = (data[locus_key].astype(object).to_numpy() == lc) & (lin >= 0)
        n_b = np.bincount(lin[sel], minlength=len(grid)).astype(float)
        bin_parts.append(
            pd.DataFrame(
                {
                    locus_key: lc,
                    "x_um": grid["x_um"],
                    "y_um": grid["y_um"],
                    "distance_um": dc,
                    "region": region,
                    "gex": dgex,
                    "records": n_b,
                }
            )  # fmt: skip
        )
        n_far, d_far = n_b[far].sum(), dgex[far].sum()
        row: dict[str, Any] = {locus_key: lc}
        if d_far <= 0 or n_far <= 0:
            row.update({"statistic": np.nan, "rho_lo": np.nan, "rho_hi": np.nan, "n_bins_far": int(far.sum()),
                        "records_far": n_far, "gex_far": d_far})  # fmt: skip
            rows.append(row)
            continue
        rho = n_far / d_far
        pos = far & (dgex > 0)
        resid = (n_b[pos] - rho * dgex[pos]) ** 2 / (rho * dgex[pos])
        phi = float(resid.sum() / max(pos.sum() - 1, 1))
        se_log = np.sqrt(max(phi, 1.0) / n_far)
        rho_lo, rho_hi = rho * np.exp(-z * se_log), rho * np.exp(z * se_log)
        icpt, slope, slope_se, _ = _poisson_slope(n_b[pos], dgex[pos]) if pos.sum() > 2 else (np.nan,) * 4
        n_in, d_in = n_b[tis].sum(), dgex[tis].sum()
        expected = rho * d_in
        tpos = tis & (dgex > 0)
        expected_power = float(np.exp(icpt + slope * np.log(dgex[tpos])).sum()) if np.isfinite(slope) else np.nan
        var = max(phi, 1.0) * (expected + d_in**2 * rho / d_far)
        zz = (n_in - expected) / np.sqrt(var) if var > 0 else np.nan
        row.update(
            {
                "statistic": rho,
                "rho_lo": rho_lo,
                "rho_hi": rho_hi,
                "dispersion": phi,
                "n_bins_far": int(far.sum()),
                "records_far": n_far,
                "gex_far": d_far,
                "records_far_no_gex": float(n_b[far & (dgex <= 0)].sum()),
                "slope": slope,
                "slope_lo": slope - z * slope_se,
                "slope_hi": slope + z * slope_se,
                "records_tissue": n_in,
                "gex_tissue": d_in,
                "expected": expected,
                "excess": n_in - expected,
                "genuine_share": (n_in - expected) / n_in if n_in > 0 else np.nan,
                "genuine_share_lo": 1 - rho_hi * d_in / n_in if n_in > 0 else np.nan,
                "genuine_share_hi": min(1.0, 1 - rho_lo * d_in / n_in) if n_in > 0 else np.nan,
                "ratio_tissue": (n_in / d_in) / rho if d_in > 0 else np.nan,
                "expected_power": expected_power,
                "genuine_share_power": (n_in - expected_power) / n_in if n_in > 0 else np.nan,
                "z": zz,
                "p_value": float(stats.norm.sf(zz)) if np.isfinite(zz) else np.nan,
            }
        )
        rows.append(row)
        dpos = dgex[pos]
        qs = np.unique(np.quantile(dpos, np.linspace(0, 1, n_depth_classes + 1)))
        cls = np.clip(np.searchsorted(qs, dpos, side="right") - 1, 0, len(qs) - 2)
        for k in range(len(qs) - 1):
            m = cls == k
            obs, ex = float(n_b[pos][m].sum()), float(rho * dpos[m].sum())
            depth_rows.append(
                {
                    locus_key: lc,
                    "gex_lo": qs[k],
                    "gex_hi": qs[k + 1],
                    "n_bins": int(m.sum()),
                    "records": obs,
                    "expected": ex,
                    "ratio": obs / ex if ex > 0 else np.nan,
                }  # fmt: skip
            )
    table = pd.DataFrame(rows).rename(columns={"statistic": "statistic"})
    if len(table) and "statistic" in table:
        table.insert(table.columns.get_loc("statistic") + 1, "rho", table["statistic"])
        if "p_value" in table:
            table.insert(table.columns.get_loc("p_value") + 1, "q_value", bh(table["p_value"].to_numpy()))
    summary = {}
    for _, r in table.iterrows():
        summary[f"rho[{r[locus_key]}]"] = r.get("rho", np.nan)
        summary[f"genuine_share[{r[locus_key]}]"] = r.get("genuine_share", np.nan)
    params = {
        "bin_um": bin_um,
        "far_um": far_um,
        "tissue_max_um": tissue_max_um,
        "sink_min_clones": sink_min_clones,
        "conf_level": conf_level,
        "null": "every record is a swap artefact: E[records in bin b] = rho * GEX molecules in b, rho fitted on bins "
        f"more than {far_um:g} um off tissue",
    }
    return SwapRateResult(
        name="swap_rate",
        table=table,
        summary=summary,
        params=params,
        by_depth=pd.DataFrame(depth_rows),
        bins=pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame(),
    )
