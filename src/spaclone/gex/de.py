"""Differential expression between region classes with region instances as replicates (pseudobulk).

Bins of one germinal centre are not independent replicates: they share the centre's history, its depth and its
neighbourhood. A per-bin test therefore mistakes between-instance variation for evidence (the archive found that
per-bin tests called 60% of shuffled GC labels significant). Here the replicate is the region *instance* (an
individual germinal centre, T zone, tile of background tissue): counts are summed over the units of each instance,
and genes are tested between the instances of two region classes.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, special, stats

from spaclone._utils import LowPowerWarning, SpacloneWarning, TestResult, bh
from spaclone.gex.qc import _bool_obs, _groups, get_counts, get_xy

__all__ = ["pseudobulk", "rank_genes_regions"]

logger = logging.getLogger("spaclone")


class PseudoreplicationWarning(SpacloneWarning):
    """Units (bins, spots) are used as replicates; p-values are not calibrated."""


# ---------------------------------------------------------------------------------------------------- pseudobulk


def pseudobulk(
    adata: ad.AnnData,
    *,
    groupby: str | Sequence[str],
    layer: str | None = None,
    min_units: int = 1,
) -> ad.AnnData:
    """Sum raw counts over the units of each group (e.g. each region instance).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts.
    groupby
        ``obs`` column(s) defining the groups; units with a missing value in any of them are left out.
    layer
        Layer holding raw counts.
    min_units
        Groups with fewer units are dropped.

    Returns
    -------
    anndata.AnnData
        One observation per group: ``X`` = summed counts (CSR), ``obs`` = the group columns, ``n_units`` and
        ``lib_size`` (total counts). Equivalent to ``scanpy.get.aggregate(..., func="sum")`` plus unit counts.
    """
    keys = [groupby] if isinstance(groupby, str) else list(groupby)
    counts = get_counts(adata, layer)
    frame = adata.obs[keys].copy()
    ok = ~frame.isna().any(axis=1).to_numpy()
    idx = np.flatnonzero(ok)
    sub = frame.iloc[idx].astype(str)
    grouped = sub.groupby(keys, sort=True)
    codes = grouped.ngroup().to_numpy()
    uniq = grouped.size().reset_index()[keys]
    member = sparse.csr_matrix((np.ones(len(idx)), (codes, idx)), shape=(len(uniq), adata.n_obs))
    x = (member @ counts).tocsr()
    obs = uniq.copy()
    obs["n_units"] = np.asarray(member.sum(axis=1)).ravel().astype(int)
    obs["lib_size"] = np.asarray(x.sum(axis=1)).ravel()
    obs.index = pd.Index(["|".join(map(str, r)) for r in uniq.itertuples(index=False)], dtype=str)
    keep = obs.n_units.to_numpy() >= min_units
    return ad.AnnData(X=x[keep], obs=obs[keep], var=pd.DataFrame(index=adata.var_names.copy()))


# ------------------------------------------------------------------------------------------------ normalisation


def _tmm(counts: np.ndarray, lib: np.ndarray, *, logratio_trim: float = 0.3, sum_trim: float = 0.05) -> np.ndarray:
    """TMM normalisation factors (edgeR ``calcNormFactors(method="TMM")``); ``counts`` is genes × samples."""
    n = counts.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f75 = np.array([np.quantile(counts[:, j] / lib[j], 0.75) for j in range(n)])
    ref = (
        int(np.argmax(np.sqrt(counts).sum(axis=0)))
        if np.median(f75) < 1e-20
        else int(np.argmin(np.abs(f75 - f75.mean())))
    )
    out = np.ones(n)
    r, nr = counts[:, ref].astype(float), lib[ref]
    for j in range(n):
        o, no = counts[:, j].astype(float), lib[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.log2((o / no) / (r / nr))
            abs_e = (np.log2(o / no) + np.log2(r / nr)) / 2
            v = (no - o) / no / o + (nr - r) / nr / r
        fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > -1e10)
        log_r, abs_e, v = log_r[fin], abs_e[fin], v[fin]
        if not len(log_r) or np.max(np.abs(log_r)) < 1e-6:
            continue
        m = len(log_r)
        lo_l = np.floor(m * logratio_trim) + 1
        hi_l = m + 1 - lo_l
        lo_s = np.floor(m * sum_trim) + 1
        hi_s = m + 1 - lo_s
        rl, ra = stats.rankdata(log_r), stats.rankdata(abs_e)
        keep = (rl >= lo_l) & (rl <= hi_l) & (ra >= lo_s) & (ra <= hi_s)
        f = np.sum(log_r[keep] / v[keep]) / np.sum(1 / v[keep]) if keep.any() else 0.0
        out[j] = 2.0 ** (f if np.isfinite(f) else 0.0)
    return out / np.exp(np.mean(np.log(out)))


def _log_cpm(counts: np.ndarray, lib: np.ndarray, prior_count: float) -> np.ndarray:
    """Log2 CPM as edgeR ``cpm(log=TRUE, prior.count=)`` (genes × samples; prior count scaled by library size)."""
    prior = prior_count * lib / lib.mean()
    return np.log2((counts + prior[None, :]) / (lib + 2 * prior)[None, :] * 1e6)


# -------------------------------------------------------------------------------------------- empirical Bayes


def _trigamma_inverse(x: float) -> float:
    """Inverse of the trigamma function by Newton's method (limma ``trigammaInverse``)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _fit_f_dist(s2: np.ndarray, df1: float) -> tuple[float, float]:
    """Prior scale and degrees of freedom of the sample variances (limma ``fitFDist``, no covariate)."""
    x = np.asarray(s2, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or df1 <= 0:
        return float(np.nanmean(x)) if len(x) else np.nan, 0.0
    x = np.maximum(x, 0)
    m = np.median(x)
    m = 1.0 if m == 0 else m
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) - special.digamma(df1 / 2) + np.log(df1 / 2)
    emean = e.mean()
    evar = np.sum((e - emean) ** 2) / (len(e) - 1) - special.polygamma(1, df1 / 2)
    if evar > 0:
        df2 = 2 * _trigamma_inverse(float(evar))
        s20 = np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        s20 = np.exp(emean)
    return float(s20), float(df2)


def _squeeze(s2: np.ndarray, df: float) -> tuple[np.ndarray, float]:
    """Posterior variances and prior degrees of freedom (limma ``squeezeVar``)."""
    s20, d0 = _fit_f_dist(s2, df)
    if not np.isfinite(d0):
        return np.full_like(s2, s20), np.inf
    return (d0 * s20 + df * s2) / (d0 + df), d0


# ------------------------------------------------------------------------------------------------------ tests


def _limma(y: np.ndarray, x: np.ndarray, coef: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moderated t (limma lmFit + eBayes, no weights): returns t, df_total, p. ``y`` is genes × samples."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = y @ x @ xtx_inv
    resid = y - beta @ x.T
    df_res = x.shape[0] - np.linalg.matrix_rank(x)
    s2 = (resid**2).sum(axis=1) / df_res
    s2_post, d0 = _squeeze(s2, df_res)
    se = np.sqrt(xtx_inv[coef, coef] * s2_post)
    t = beta[:, coef] / se
    df_tot = min(df_res + d0, df_res * len(y))
    p = 2 * stats.t.sf(np.abs(t), df_tot)
    return t, np.full(len(t), df_tot), p


def _welch(y: np.ndarray, is_g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = y[:, is_g], y[:, ~is_g]
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(axis=1, ddof=1) / na, b.var(axis=1, ddof=1) / nb
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    return t, df, p


def _nb_fit(
    y: np.ndarray, x: np.ndarray, offset: np.ndarray, phi: float, maxit: int = 50, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Negative-binomial GLM (log link, fixed dispersion) for every gene by Fisher scoring with step halving.

    Returns the fitted means (genes × samples) and deviances.
    """
    g, n = y.shape
    z0 = np.log((y + 0.5) / np.exp(offset)[None, :])
    beta = np.linalg.lstsq(x, z0.T, rcond=None)[0].T
    eta_cap = 30.0

    def mean(b):
        return np.exp(np.clip(b @ x.T + offset[None, :], -eta_cap, eta_cap))

    def deviance(mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            if phi > 0:
                t2 = (y + 1 / phi) * np.log((1 + phi * y) / (1 + phi * mu))
            else:
                t2 = y - mu
        return 2 * np.sum(t1 - t2, axis=1)

    mu = mean(beta)
    dev = deviance(mu)
    for _ in range(maxit):
        w = mu / (1 + phi * mu)
        z = np.log(mu) - offset[None, :] + (y - mu) / mu
        xtwx = np.einsum("ni,gn,nj->gij", x, w, x)
        xtwz = np.einsum("ni,gn->gi", x, w * z)
        xtwx += 1e-10 * np.eye(x.shape[1])[None]
        new = np.linalg.solve(xtwx, xtwz[..., None])[..., 0]
        step = new - beta
        done = np.zeros(g, dtype=bool)
        cand_dev = dev.copy()
        cand = beta.copy()
        for _h in range(10):
            b = beta + step
            d = deviance(mean(b))
            better = (d <= dev + 1e-10 * np.abs(dev)) & ~done
            cand[better], cand_dev[better] = b[better], d[better]
            done |= better
            if done.all():
                break
            step = np.where(done[:, None], step, step / 2)
        change = np.abs(dev - cand_dev) / (np.abs(cand_dev) + 0.1)
        beta, dev = cand, cand_dev
        mu = mean(beta)
        if np.all(change < tol):
            break
    return mu, dev


def _nb_apl(y, x, offset, phi):
    """Cox–Reid adjusted profile log-likelihood summed over genes at dispersion ``phi``."""
    mu, _ = _nb_fit(y, x, offset, phi, maxit=25, tol=1e-6)
    r = 1 / phi
    ll = (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + y * np.log(phi * mu / (1 + phi * mu))
        - r * np.log1p(phi * mu)
    ).sum(axis=1)
    w = mu / (1 + phi * mu)
    xtwx = np.einsum("ni,gn,nj->gij", x, w, x)
    sign, logdet = np.linalg.slogdet(xtwx + 1e-10 * np.eye(x.shape[1])[None])
    return float(np.sum(ll - 0.5 * logdet))


def _nb_ql(
    y: np.ndarray, x: np.ndarray, offset: np.ndarray, coef: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """NB GLM with a common Cox–Reid dispersion and the quasi-likelihood F-test of one coefficient."""
    grid = np.logspace(-4, 1, 26)
    apl = np.array([_nb_apl(y, x, offset, p) for p in grid])
    i = int(np.argmax(apl))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.geomspace(lo, hi, 11)
    phi = float(fine[int(np.argmax([_nb_apl(y, x, offset, p) for p in fine]))])
    _, dev = _nb_fit(y, x, offset, phi)
    x0 = np.delete(x, coef, axis=1)
    _, dev0 = _nb_fit(y, x0, offset, phi)
    df_res = x.shape[0] - np.linalg.matrix_rank(x)
    s2 = np.maximum(dev, 0) / df_res
    s2_post, d0 = _squeeze(s2, df_res)
    df_tot = min(df_res + d0, df_res * len(y))
    f = np.maximum(dev0 - dev, 0) / s2_post
    p = stats.f.sf(f, 1, df_tot)
    return f, np.full(len(f), df_tot), p, phi


# ------------------------------------------------------------------------------------------------- front door


def _instances(adata, region_key, instance_key, labels, missing, tile_um, library_key, spatial_key):
    reg = adata.obs[region_key].astype(object).to_numpy()
    if instance_key is not None and instance_key in adata.obs:
        inst = adata.obs[instance_key].astype(object).to_numpy().copy()
        inst = np.where(pd.isna(inst), None, inst)
    else:
        inst = np.full(adata.n_obs, None, dtype=object)
    use = np.isin(reg.astype(str), [str(lab) for lab in labels]) & ~pd.isna(reg)
    need = use & np.array([v is None for v in inst], dtype=bool)
    if need.any():
        if missing == "error":
            bad = sorted(set(reg[need].astype(str)))
            raise ValueError(
                f"units of region(s) {bad} have no instance id in obs[{instance_key!r}]; pass "
                "missing_instances='tiles' or 'drop', or give instance ids"
            )
        if missing == "tiles":
            if not tile_um > 0:
                raise ValueError("tile_um must be positive")
            xy = get_xy(adata, spatial_key)
            lib = np.full(adata.n_obs, "", dtype=object)
            for k, idx in _groups(adata, library_key).items():
                lib[idx] = f"{k}|" if library_key is not None else ""
            ix = pd.Series(np.floor(xy[need, 0] / tile_um).astype(np.int64)).astype(str)
            iy = pd.Series(np.floor(xy[need, 1] / tile_um).astype(np.int64)).astype(str)
            name = pd.Series(lib[need]).astype(str) + pd.Series(reg[need]).astype(str) + ":tile_" + ix + "_" + iy
            inst[need] = name.to_numpy(dtype=object)
        elif missing == "drop":
            use &= ~need
        else:
            raise ValueError(f"missing_instances must be 'tiles', 'drop' or 'error', got {missing!r}")
    return use, reg, inst


def rank_genes_regions(
    adata: ad.AnnData,
    *,
    region_key: str = "region",
    instance_key: str | None = "region_id",
    groups: str | Sequence[str] | None = None,
    reference: str = "rest",
    method: Literal["pseudobulk", "scanpy"] = "pseudobulk",
    test: Literal["limma", "welch", "nb"] = "limma",
    sample_key: str | None = None,
    missing_instances: Literal["tiles", "drop", "error"] = "tiles",
    tile_um: float = 200.0,
    min_units: int = 10,
    min_instances: int = 2,
    min_count: float = 10,
    min_total_count: float = 15,
    norm: Literal["tmm", "none"] = "tmm",
    prior_count: float = 2.0,
    tissue_key: str | None = "in_tissue",
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    genes: Sequence[str] | None = None,
) -> TestResult:
    """Genes that differ between region classes, with region instances (not units) as replicates.

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts, a region label column and an instance column (from
        :func:`~spaclone.gex.call_regions`).
    region_key
        ``obs`` column of region classes (e.g. ``"GC"``, ``"ASC"``, ``"other"``).
    instance_key
        ``obs`` column of region instances (e.g. one id per germinal centre).
    groups
        Region class(es) to test, each against ``reference`` (default: every class except the reference).
    reference
        A region class, or ``"rest"`` (all other classes).
    method
        ``"pseudobulk"`` (default). ``"scanpy"`` runs ``scanpy.tl.rank_genes_groups`` (Wilcoxon) with units as
        replicates; it raises a :class:`PseudoreplicationWarning` because its p-values are not calibrated.
    test
        ``"limma"``: moderated t on TMM log-CPM (limma-trend without trend; empirical Bayes variance shrinkage);
        ``"welch"``: Welch's t on log-CPM (ignores ``sample_key``); ``"nb"``: negative-binomial GLM with a common
        Cox–Reid dispersion and the
        quasi-likelihood F-test (edgeR glmQLFit / glmQLFTest style).
    sample_key
        ``obs`` column of a blocking factor (section, animal) added to the design; every instance must lie in one
        sample.
    missing_instances
        Units of a tested class without an instance id: ``"tiles"`` (default) splits them into square tiles of
        ``tile_um`` that act as instances (e.g. background tissue), ``"drop"`` leaves them out, ``"error"`` raises.
    tile_um
        Tile side for ``missing_instances="tiles"`` (about the size of a germinal centre).
    min_units
        Instances with fewer units are dropped.
    min_instances
        Minimum instances per side; otherwise the comparison is skipped with a :class:`LowPowerWarning`.
    min_count, min_total_count
        Gene filter (edgeR ``filterByExpr``): CPM ≥ ``min_count`` / median library size × 10⁶ in at least as many
        instances as the smaller side, and at least ``min_total_count`` counts in total.
    norm
        ``"tmm"`` (trimmed mean of M values) or ``"none"`` (library sizes only).
    prior_count
        Prior count of the log-CPM values.
    tissue_key
        Boolean ``obs`` column; off-tissue units are never used.
    library_key
        ``obs`` column naming the section (tiles never cross sections).
    layer
        Layer holding raw counts.
    spatial_key
        ``obsm`` key of the coordinates (tiles).
    genes
        Restrict the test to these genes.

    Returns
    -------
    TestResult
        ``table``: one row per (group, gene) with ``group``, ``reference``, ``gene``, ``log2fc`` (difference of mean
        log2 CPM), ``logcpm_group``, ``logcpm_reference``, ``statistic`` (t or F), ``df``, ``p_value``,
        ``q_value`` (BH within the group), ``n_instances_group``, ``n_instances_reference``; ``summary``: genes
        tested and significant (q < 0.05) per group; ``params``: every argument; ``null``: None.

    Notes
    -----
    Null model: the instances of the two classes are exchangeable — the expected log expression of a gene is the
    same in both classes (given the blocking factor), with between-instance variation estimated from the instances
    themselves (moderated by empirical Bayes across genes). Units within an instance are never treated as
    independent. With ``missing_instances="tiles"`` background tiles are spatially autocorrelated neighbours, so the
    reference variance can be slightly underestimated; a larger ``tile_um`` is more conservative.

    References
    ----------
    Smyth 2004, doi:10.2202/1544-6115.1027 (moderated t); Lund et al. 2012, doi:10.1515/1544-6115.1826 and Chen, Lun
    & Smyth 2016, doi:10.12688/f1000research.8987.2 (quasi-likelihood F-test); McCarthy, Chen & Smyth 2012,
    doi:10.1093/nar/gks042 (NB GLM, Cox–Reid dispersion); Robinson & Oshlack 2010, doi:10.1186/gb-2010-11-3-r25
    (TMM); Squair et al. 2021, doi:10.1038/s41467-021-25960-2 and Crowell et al. 2020,
    doi:10.1038/s41467-020-19894-4 (pseudobulk); Zimmerman, Espeland & Langefeld 2021,
    doi:10.1038/s41467-021-21038-1 (pseudoreplication); Welch 1947, doi:10.1093/biomet/34.1-2.28.
    """
    if region_key not in adata.obs:
        raise KeyError(f"obs[{region_key!r}] is missing; run spaclone.gex.call_regions first")
    tissue = _bool_obs(adata, tissue_key)
    reg_all = adata.obs[region_key].astype(object)
    present = [str(c) for c in pd.unique(reg_all[tissue & ~reg_all.isna().to_numpy()])]
    if reference != "rest" and reference not in present:
        raise ValueError(f"reference {reference!r} is not a region class ({present})")
    if groups is None:
        groups = [c for c in present if c != reference]
    groups = [groups] if isinstance(groups, str) else list(groups)
    params: dict[str, Any] = {
        "region_key": region_key,
        "instance_key": instance_key,
        "groups": groups,
        "reference": reference,
        "method": method,
        "test": test,
        "sample_key": sample_key,
        "missing_instances": missing_instances,
        "tile_um": tile_um,
        "min_units": min_units,
        "min_instances": min_instances,
        "min_count": min_count,
        "min_total_count": min_total_count,
        "norm": norm,
        "prior_count": prior_count,
    }
    if method == "scanpy":
        return _scanpy_fallback(adata, region_key, groups, reference, tissue, layer, params)
    if method != "pseudobulk":
        raise ValueError(f"method must be 'pseudobulk' or 'scanpy', got {method!r}")
    if test not in ("limma", "welch", "nb"):
        raise ValueError(f"test must be 'limma', 'welch' or 'nb', got {test!r}")
    labels = present if reference == "rest" else sorted(set(groups) | {reference})
    use, reg, inst = _instances(
        adata, region_key, instance_key, labels, missing_instances, tile_um, library_key, spatial_key
    )
    use &= tissue
    obs = pd.DataFrame({"_reg": reg.astype(str), "_inst": np.where(use, inst, None)}, index=adata.obs_names)
    if sample_key is not None:
        obs["_sample"] = adata.obs[sample_key].astype(str).to_numpy()
        per = obs[use].groupby("_inst")["_sample"].nunique()
        if (per > 1).any():
            raise ValueError(f"instances {list(per[per > 1].index[:5])} span several {sample_key!r} values")
    tmp = ad.AnnData(X=get_counts(adata, layer), obs=obs, var=pd.DataFrame(index=adata.var_names.copy()))
    pb = pseudobulk(tmp, groupby=["_reg", "_inst"] + (["_sample"] if sample_key else []), min_units=min_units)
    gene_names = pb.var_names.astype(str)
    if genes is not None:
        gsel = gene_names.get_indexer(list(genes))
        gsel = gsel[gsel >= 0]
    else:
        gsel = np.arange(len(gene_names))
    y_all = np.asarray(pb.X[:, gsel].todense(), dtype=float).T  # genes x instances
    lib_all = np.asarray(pb.X.sum(axis=1), dtype=float).ravel()
    tables, summary = [], {}
    for g in groups:
        is_g = (pb.obs["_reg"] == g).to_numpy()
        is_r = (~is_g) if reference == "rest" else (pb.obs["_reg"] == reference).to_numpy()
        sel = is_g | is_r
        ng, nr = int(is_g.sum()), int(is_r.sum())
        summary[f"n_instances[{g}]"] = ng
        if (
            ng < min_instances
            or nr < min_instances
            or ng + nr - (2 + (pb.obs.loc[sel, "_sample"].nunique() - 1 if sample_key else 0)) < 1
        ):
            warnings.warn(
                f"rank_genes_regions: {g!r} has {ng} and the reference {nr} instance(s) with >= "
                f"{min_units} units; comparison skipped",
                LowPowerWarning,
                stacklevel=2,
            )
            continue
        y = y_all[:, sel]
        lib = lib_all[sel]
        grp = is_g[sel]
        cpm = y / lib[None, :] * 1e6
        cut = min_count / np.median(lib) * 1e6
        keep = ((cpm >= cut).sum(axis=1) >= min(ng, nr)) & (y.sum(axis=1) >= min_total_count)
        if keep.sum() < 3:
            warnings.warn(
                f"rank_genes_regions: fewer than 3 genes pass the expression filter for {g!r}",
                LowPowerWarning,
                stacklevel=2,
            )
            continue
        y = y[keep]
        nf = _tmm(y, lib) if norm == "tmm" else np.ones(len(lib))
        eff = lib * nf
        logcpm = _log_cpm(y, eff, prior_count)
        cols = [np.ones(len(grp)), grp.astype(float)]
        if sample_key is not None:
            smp = pb.obs.loc[sel, "_sample"].to_numpy()
            for s in pd.unique(smp)[1:]:
                cols.append((smp == s).astype(float))
        x = np.column_stack(cols)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("the design is not of full rank (is every class present in one sample only?)")
        if test == "limma":
            stat, df, p = _limma(logcpm, x, 1)
        elif test == "welch":
            stat, df, p = _welch(logcpm, grp)
        else:
            stat, df, p, phi = _nb_ql(y, x, np.log(eff), 1)
            summary[f"dispersion[{g}]"] = phi
        lg, lr = logcpm[:, grp].mean(axis=1), logcpm[:, ~grp].mean(axis=1)
        t = pd.DataFrame(
            {
                "group": g,
                "reference": reference,
                "gene": gene_names[gsel][keep],
                "log2fc": lg - lr,
                "logcpm_group": lg,
                "logcpm_reference": lr,
                "statistic": stat,
                "df": df,
                "p_value": p,
                "q_value": bh(p),
                "n_instances_group": ng,
                "n_instances_reference": nr,
            }
        )
        tables.append(t.sort_values("p_value", kind="stable").reset_index(drop=True))
        summary[f"n_genes[{g}]"] = int(len(t))
        summary[f"n_q05[{g}]"] = int((t.q_value < 0.05).sum())
    cols = [
        "group",
        "reference",
        "gene",
        "log2fc",
        "logcpm_group",
        "logcpm_reference",
        "statistic",
        "df",
        "p_value",
        "q_value",
        "n_instances_group",
        "n_instances_reference",
    ]
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=cols)
    return TestResult(name="rank_genes_regions", table=table, summary=summary, params=params)


def _scanpy_fallback(adata, region_key, groups, reference, tissue, layer, params):
    import scanpy as sc

    warnings.warn(
        "rank_genes_regions(method='scanpy') treats every unit as a replicate; units of one region instance are not "
        "independent and the p-values are anti-conservative (per-bin tests called 60% of shuffled labels "
        "significant in the archive). Use method='pseudobulk' for inference.",
        PseudoreplicationWarning,
        stacklevel=3,
    )
    counts = get_counts(adata, layer)[tissue]
    tmp = ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame({"_reg": adata.obs[region_key].astype(str).to_numpy()[tissue]}, index=adata.obs_names[tissue]),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    tmp.obs["_reg"] = tmp.obs["_reg"].astype("category")
    sc.tl.rank_genes_groups(tmp, "_reg", groups=groups, reference=reference, method="wilcoxon", use_raw=False)
    df = sc.get.rank_genes_groups_df(tmp, group=groups)
    if "group" not in df.columns:
        df.insert(0, "group", groups[0])
    table = pd.DataFrame(
        {
            "group": df["group"].astype(str),
            "reference": reference,
            "gene": df["names"].astype(str),
            "log2fc": df["logfoldchanges"],
            "statistic": df["scores"],
            "p_value": df["pvals"],
            "q_value": df["pvals_adj"],
        }
    )
    return TestResult(
        name="rank_genes_regions", table=table, summary={"replicate": "unit (not calibrated)"}, params=params
    )
