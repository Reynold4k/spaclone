"""Spatial statistics of cell types and regions: neighbourhood enrichment and spatially variable genes.

:func:`nhood_enrichment` is squidpy's neighbourhood-enrichment statistic (edges between two classes on the spatial
neighbour graph, against random relabelling of the units within each section), returned as a
:class:`~spaclone.TestResult` with Monte-Carlo p-values and BH q-values, which squidpy does not report. It uses a
graph built by squidpy (``connectivity_key``) or builds a radius / k-nearest-neighbour graph itself, so it runs
without squidpy. :func:`spatially_variable_genes` wraps squidpy's Moran's I and needs squidpy (``pip install
spaclone[scverse]``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from spaclone._utils import RngLike, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy
from spaclone.gex.qc import _groups, get_xy, radius_graph

__all__ = ["nhood_enrichment", "spatial_graph", "spatially_variable_genes"]

logger = logging.getLogger("spaclone")


def spatial_graph(
    adata: ad.AnnData,
    *,
    radius_um: float | None = None,
    n_neighbors: int | None = None,
    library_key: str | None = None,
    spatial_key: str = "spatial",
) -> sparse.csr_matrix:
    """Symmetric 0/1 spatial neighbour graph (self excluded; edges only within a section).

    Parameters
    ----------
    adata
        AnnData with ``obsm[spatial_key]`` in µm.
    radius_um
        Units within this distance (inclusive) are neighbours.
    n_neighbors
        Instead of a radius: each unit's ``n_neighbors`` nearest units (union of both directions). Default 6 when
        neither is given (squidpy's default for generic coordinates).
    library_key
        ``obs`` column naming the section.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``n × n`` adjacency.
    """
    xy = get_xy(adata, spatial_key)
    lib = np.zeros(adata.n_obs, dtype=np.int64)
    for k, (_, idx) in enumerate(_groups(adata, library_key).items()):
        lib[idx] = k
    if radius_um is not None:
        return radius_graph(xy, radius_um, groups=lib)
    k = 6 if n_neighbors is None else int(n_neighbors)
    rows, cols = [], []
    for idx in pd.Series(np.arange(adata.n_obs)).groupby(lib).indices.values():
        kk = min(k, len(idx) - 1)
        if kk < 1:
            continue
        _, j = cKDTree(xy[idx]).query(xy[idx], k=kk + 1)
        rows.append(np.repeat(idx, kk))
        cols.append(idx[j[:, 1:]].ravel())
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    keep = r != c  # duplicate positions can return a unit as its own neighbour
    a = sparse.csr_matrix((np.ones(int(keep.sum())), (r[keep], c[keep])), shape=(adata.n_obs, adata.n_obs))
    return ((a + a.T) > 0).astype(float).tocsr()


def nhood_enrichment(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    radius_um: float | None = None,
    n_neighbors: int | None = None,
    connectivity_key: str | None = None,
    library_key: str | None = None,
    n_perm: int = 1000,
    rng: RngLike = None,
    spatial_key: str = "spatial",
    keep_null: bool = False,
) -> TestResult:
    """Are units of two classes neighbours more (or less) often than random labelling gives?

    Parameters
    ----------
    adata
        AnnData with a categorical ``obs[cluster_key]`` (cell types, regions, domains) and coordinates in µm.
    cluster_key
        ``obs`` column of the classes; units with a missing class are left out.
    radius_um, n_neighbors
        Graph built by :func:`spatial_graph` when ``connectivity_key`` is not given.
    connectivity_key
        ``obsp`` key of an existing graph (e.g. squidpy's ``"spatial_connectivities"``); non-zero entries are
        edges.
    library_key
        ``obs`` column naming the section; labels are permuted within a section and edges never cross sections.
    n_perm
        Number of random relabellings.
    rng
        Seed or generator.
    spatial_key
        ``obsm`` key of the coordinates.
    keep_null
        Keep the null draws (``n_perm × n_pairs``) in ``result.null``.

    Returns
    -------
    TestResult
        ``table``: one row per unordered class pair with ``cluster_a``, ``cluster_b``, ``statistic`` (directed edge
        count from a to b, squidpy's count), ``null_mean``, ``null_sd``, ``z`` (squidpy's z-score), ``ratio``
        (statistic / null mean), ``p_value`` (two-sided Monte-Carlo), ``p_greater``, ``p_less``, ``q_value`` (BH
        over pairs of the two-sided p), ``n_perm``.

    Notes
    -----
    Null model: the class labels are permuted among the units of each section (random labelling); the positions,
    the neighbour graph and the number of units of each class per section stay fixed. Neighbouring units are
    spatially autocorrelated, so contiguous classes (regions, domains) are always "enriched" with themselves;
    the test is informative for pairs of different classes.

    References
    ----------
    Palla et al. 2022, doi:10.1038/s41592-021-01358-2 (squidpy ``nhood_enrichment``); Schapiro et al. 2017,
    doi:10.1038/nmeth.4391 (histoCAT neighbourhood analysis); Phipson & Smyth 2010, doi:10.2202/1544-6115.1585
    (Monte-Carlo p-values).
    """
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    gen = as_rng(rng)
    lab = adata.obs[cluster_key]
    ok = ~lab.isna().to_numpy()
    if connectivity_key is not None:
        if connectivity_key not in adata.obsp:
            raise KeyError(f"adata.obsp[{connectivity_key!r}] is missing")
        a = sparse.csr_matrix(adata.obsp[connectivity_key])
        a = (a != 0).astype(float)
    else:
        a = spatial_graph(
            adata, radius_um=radius_um, n_neighbors=n_neighbors, library_key=library_key, spatial_key=spatial_key
        )
    keep = np.flatnonzero(ok)
    a = a[keep][:, keep].tocoo()
    cats = (
        list(lab.cat.categories)
        if isinstance(lab.dtype, pd.CategoricalDtype)
        else sorted(pd.unique(lab[ok].astype(str)))
    )
    codes = pd.Categorical(lab.to_numpy()[keep].astype(str), categories=[str(c) for c in cats]).codes.astype(np.int64)
    k = len(cats)
    rows, cols = a.row, a.col
    libs = (
        pd.Series(np.zeros(len(keep), dtype=int))
        if library_key is None
        else pd.Series(adata.obs[library_key].astype(str).to_numpy()[keep])
    )
    strata = list(libs.groupby(libs.to_numpy()).indices.values())

    def count(c: np.ndarray) -> np.ndarray:
        return np.bincount(c[rows] * k + c[cols], minlength=k * k).reshape(k, k).astype(float)

    iu = np.triu_indices(k)
    obs = count(codes)[iu]
    null = np.empty((n_perm, len(obs)))
    perm = codes.copy()
    for b in range(n_perm):
        for idx in strata:
            perm[idx] = codes[idx][gen.permutation(len(idx))]
        null[b] = count(perm)[iu]
    mu, sd, _, _ = null_summary(null)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - mu) / sd
        ratio = obs / mu
    p2 = mc_pvalue(null, obs, alternative="two-sided")
    table = pd.DataFrame(
        {
            "cluster_a": [str(cats[i]) for i in iu[0]],
            "cluster_b": [str(cats[j]) for j in iu[1]],
            "statistic": obs,
            "null_mean": mu,
            "null_sd": sd,
            "z": z,
            "ratio": ratio,
            "p_value": p2,
            "p_greater": mc_pvalue(null, obs, alternative="greater"),
            "p_less": mc_pvalue(null, obs, alternative="less"),
            "q_value": bh(p2),
            "n_perm": n_perm,
        }
    )
    params = {
        "cluster_key": cluster_key,
        "radius_um": radius_um,
        "n_neighbors": n_neighbors,
        "connectivity_key": connectivity_key,
        "library_key": library_key,
        "n_perm": n_perm,
        "design": "Permute(cluster labels, within=library)",
        "rng_entropy": rng_entropy(gen),
        "n_edges": int(len(rows)),
    }
    diff = table[table.cluster_a != table.cluster_b]
    summary = {
        "n_classes": k,
        "n_pairs": len(table),
        "n_q05_between": int((diff.q_value < 0.05).sum()),
        "n_edges": int(len(rows)),
    }
    return TestResult(
        name="nhood_enrichment", table=table, summary=summary, params=params, null=null if keep_null else None
    )


def spatially_variable_genes(
    adata: ad.AnnData,
    *,
    genes: Sequence[str] | None = None,
    radius_um: float | None = None,
    n_neighbors: int | None = 6,
    library_key: str | None = None,
    n_perm: int | None = None,
    layer: str | None = None,
    rng: RngLike = None,
    spatial_key: str = "spatial",
) -> TestResult:
    """Global Moran's I of genes on the spatial neighbour graph (squidpy ``spatial_autocorr``; needs squidpy).

    Parameters
    ----------
    adata
        AnnData with log-normalised ``X`` (or ``layer``) and coordinates in µm.
    genes
        Genes to test (default: the highly variable genes if marked, else all).
    radius_um, n_neighbors
        Neighbour graph (radius in µm, or k nearest units).
    library_key
        ``obs`` column naming the section (edges never cross sections).
    n_perm
        Permutations for Monte-Carlo p-values (``None``: analytic normal approximation only).
    layer
        Expression layer (default ``X``).
    rng
        Seed of the permutations.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    TestResult
        ``table``: one row per gene with ``gene``, ``statistic`` (Moran's I), ``expected`` (−1 / (n − 1)),
        ``p_value`` (Monte-Carlo with ``n_perm``, else analytic), ``p_norm``, ``q_value`` (BH), ``n_perm``.

    Raises
    ------
    ImportError
        If squidpy is not installed.

    Notes
    -----
    Null model: expression values are permuted over all units of the graph (complete spatial randomness of the
    values; positions and graph fixed), or the analytic normal approximation of that randomisation. Spatially
    smooth confounders (depth, tissue density) also give a high I; read I as "spatially structured", not as
    "biologically variable".

    References
    ----------
    Moran 1950, doi:10.2307/2332142; Palla et al. 2022, doi:10.1038/s41592-021-01358-2 (squidpy).
    """
    try:
        import squidpy as sq
    except ImportError as e:  # pragma: no cover - squidpy is installed in the dev environment
        raise ImportError("spatially_variable_genes needs squidpy: pip install 'spaclone[scverse]'") from e
    gen = as_rng(rng)
    a = spatial_graph(
        adata,
        radius_um=radius_um,
        n_neighbors=n_neighbors if radius_um is None else None,
        library_key=library_key,
        spatial_key=spatial_key,
    )
    if genes is None:
        genes = (
            list(adata.var_names[adata.var["highly_variable"].to_numpy(bool)])
            if "highly_variable" in adata.var
            else list(adata.var_names)
        )
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        raise ValueError("no gene to test")
    tmp = ad.AnnData(
        X=adata.layers[layer] if layer is not None else adata.X,
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    tmp.obsp["spaclone_connectivities"] = a
    seed = int(gen.integers(0, 2**31 - 1))
    df = sq.gr.spatial_autocorr(
        tmp,
        connectivity_key="spaclone_connectivities",
        genes=genes,
        mode="moran",
        n_perms=n_perm,
        rng=seed,
        copy=True,
        show_progress_bar=False,
    )
    p = df["pval_sim"].to_numpy(dtype=float) if n_perm else df["pval_norm"].to_numpy(dtype=float)
    table = (
        pd.DataFrame(
            {
                "gene": df.index.astype(str),
                "statistic": df["I"].to_numpy(dtype=float),
                "expected": -1.0 / (adata.n_obs - 1),
                "p_value": p,
                "p_norm": df["pval_norm"].to_numpy(dtype=float),
                "q_value": bh(p),
                "n_perm": n_perm or 0,
            }
        )
        .sort_values("p_value", kind="stable")
        .reset_index(drop=True)
    )
    params = {
        "radius_um": radius_um,
        "n_neighbors": n_neighbors,
        "library_key": library_key,
        "n_perm": n_perm,
        "layer": layer,
        "rng_entropy": rng_entropy(gen),
        "n_genes": len(genes),
    }
    summary = {"n_genes": len(table), "n_q05": int((table.q_value < 0.05).sum())}
    return TestResult(name="spatially_variable_genes", table=table, summary=summary, params=params)
