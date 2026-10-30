"""Spatial domains: clustering of neighbourhood-augmented expression (BANKSY-style), then Leiden.

Each unit is described by its own log-expression of the highly variable genes and by the mean expression of the
units around it (within ``radius_um``, itself excluded). Both blocks are z-scored per gene and weighted
``sqrt(1 - lambda_)`` and ``sqrt(lambda_)``; ``lambda_ = 0.8`` favours tissue domains (follicle, T zone, red pulp)
over cell types. PCA, an exact k-nearest-neighbour graph and Leiden follow. This is the archive's recipe for the
StereoIR spleen (``04_external_validation_2026-10/workflow/qc/T8d_gc_gex_leiden.py`` / ``T8h_domains.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from spaclone._utils import RngLike
from spaclone.gex.qc import _bool_obs, _groups, _record, get_counts, get_xy, radius_graph
from spaclone.gex.workflow import _seed, gene_group_mask

__all__ = ["domains", "neighbour_mean"]

logger = logging.getLogger("spaclone")


def neighbour_mean(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    radius_um: float | None = 22.5,
    n_neighbors: int | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` over each unit's spatial neighbours (the unit itself excluded).

    Parameters
    ----------
    xy
        ``(n, 2)`` coordinates in µm.
    values
        ``(n, p)`` array (rows = units).
    radius_um
        Neighbours within this distance (inclusive).
    n_neighbors
        Instead of a radius: the ``n_neighbors`` nearest units.
    groups
        Section labels; neighbours are taken within a section.

    Returns
    -------
    mean
        ``(n, p)`` neighbour means; a unit without neighbours keeps its own values.
    degree
        Number of neighbours of each unit.
    """
    n = len(xy)
    if n_neighbors is not None:
        rows, cols = [], []
        parts = pd.Series(np.arange(n)).groupby(np.zeros(n) if groups is None else np.asarray(groups)).indices
        for idx in parts.values():
            k = min(int(n_neighbors), len(idx) - 1)
            if k < 1:
                continue
            _, j = cKDTree(xy[idx]).query(xy[idx], k=k + 1)
            rows.append(np.repeat(idx, k))
            cols.append(idx[j[:, 1:]].ravel())
        r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
        keep = r != c
        w = sparse.csr_matrix((np.ones(int(keep.sum())), (r[keep], c[keep])), shape=(n, n))
    elif radius_um is not None and radius_um > 0:
        w = radius_graph(xy, radius_um, groups=groups)
    else:
        raise ValueError("give radius_um > 0 or n_neighbors")
    deg = np.asarray(w.sum(axis=1)).ravel()
    v = np.asarray(values, dtype=np.float32)
    mean = (sparse.diags(1.0 / np.maximum(deg, 1)).astype(np.float32) @ w.astype(np.float32)) @ v
    mean = np.asarray(mean, dtype=np.float32)
    mean[deg == 0] = v[deg == 0]
    return mean, deg


def _zscore(m: np.ndarray) -> np.ndarray:
    mu, sd = m.mean(axis=0), m.std(axis=0)
    sd[sd == 0] = 1
    z = (m - mu) / sd
    np.clip(z, -10, 10, out=z)
    return z.astype(np.float32)


def domains(
    adata: ad.AnnData,
    *,
    radius_um: float | None = 22.5,
    n_neighbors: int | None = None,
    lambda_: float = 0.8,
    resolution: float = 0.5,
    n_hvg: int = 3000,
    genes: Sequence[str] | None = None,
    min_gene_frac: float = 0.005,
    exclude: str | Sequence[str] | None = ("ig", "tr", "mt", "ribo", "hb"),
    n_pcs: int = 30,
    graph_neighbors: int = 15,
    target_sum: float = 1e4,
    tissue_key: str | None = "in_tissue",
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    key_added: str = "domain",
    rng: RngLike = 0,
) -> pd.DataFrame:
    """Spatial domains by Leiden clustering of own and neighbourhood-mean expression (in place).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts and ``obsm[spatial_key]`` in µm.
    radius_um
        Spatial neighbourhood radius (µm; 22.5 µm = the 20 nearest bins of a 10-µm grid).
    n_neighbors
        Use the ``n_neighbors`` nearest units instead of a radius.
    lambda_
        Weight of the neighbourhood block (0 = expression only, 1 = neighbourhood only).
    resolution
        Leiden resolution.
    n_hvg
        Highly variable genes (scanpy ``"seurat"`` flavour on log CP10k) among the eligible genes.
    genes
        Use these genes instead of HVGs.
    min_gene_frac
        Genes detected in fewer units are not eligible.
    exclude
        Gene groups / patterns that are not eligible (see :data:`spaclone.gex.GENE_GROUPS`).
    n_pcs
        Principal components of the augmented matrix.
    graph_neighbors
        Neighbours of the exact kNN graph in PC space.
    target_sum
        Counts per unit of the normalisation.
    tissue_key
        Boolean ``obs`` column; only tissue units are clustered (others get NaN).
    library_key
        ``obs`` column naming the section; spatial neighbours stay within a section.
    layer
        Layer holding raw counts.
    spatial_key
        ``obsm`` key of the coordinates.
    key_added
        ``obs`` column of the domains; PCs go to ``obsm[f"X_{key_added}"]``.
    rng
        Seed of PCA and Leiden.

    Returns
    -------
    pandas.DataFrame
        Per domain: ``n_units`` and ``agreement`` (mean share of a unit's spatial neighbours in the same domain).

    References
    ----------
    Singhal et al. 2024, doi:10.1038/s41588-024-01664-3 (BANKSY neighbourhood-augmented features); Traag, Waltman &
    van Eck 2019, doi:10.1038/s41598-019-41695-z (Leiden).
    """
    import scanpy as sc
    from sklearn.decomposition import PCA

    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    seed = _seed(rng)
    tissue = _bool_obs(adata, tissue_key)
    t = np.flatnonzero(tissue)
    if len(t) < 3:
        raise ValueError("fewer than 3 tissue units")
    counts = get_counts(adata, layer)[t]
    tot = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    lg = (sparse.diags(target_sum / np.maximum(tot, 1)) @ counts).tocsr().astype(np.float32)
    lg.data = np.log1p(lg.data)
    names = adata.var_names.astype(str)
    if genes is not None:
        sel = names.get_indexer(list(genes))
        sel = sel[sel >= 0]
        if not len(sel):
            raise ValueError("none of the given genes is in the data")
    else:
        det = np.asarray((counts > 0).sum(axis=0)).ravel()
        elig = np.flatnonzero((det >= min_gene_frac * len(t)) & (det > 0) & ~gene_group_mask(names, exclude))
        if len(elig) < 2:
            raise ValueError("fewer than 2 eligible genes")
        sub = ad.AnnData(lg[:, elig], var=pd.DataFrame(index=names[elig]))
        hv = sc.pp.highly_variable_genes(sub, flavor="seurat", n_top_genes=int(min(n_hvg, len(elig))), inplace=False)
        sel = elig[np.asarray(hv["highly_variable"], dtype=bool)]
    xo = lg[:, sel].toarray()
    lib = np.zeros(adata.n_obs, dtype=np.int64)
    for k, (_, idx) in enumerate(_groups(adata, library_key).items()):
        lib[idx] = k
    xy = get_xy(adata, spatial_key)[t]
    xn, deg = neighbour_mean(xy, xo, radius_um=radius_um, n_neighbors=n_neighbors, groups=lib[t])
    zo = _zscore(xo)
    del xo
    blocks = [np.sqrt(1 - lambda_) * zo] if lambda_ < 1 else []
    if lambda_ > 0:
        blocks.append(np.sqrt(lambda_) * _zscore(xn))
    del xn
    feat = np.hstack(blocks).astype(np.float32)
    n_comp = int(min(n_pcs, feat.shape[0] - 1, feat.shape[1] - 1))
    pcs = PCA(n_comp, svd_solver="randomized", random_state=seed).fit_transform(feat).astype(np.float32)
    del feat
    g = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names[t]))
    g.obsm["X"] = pcs
    sc.pp.neighbors(
        g, use_rep="X", n_neighbors=int(min(graph_neighbors, len(t) - 1)), transformer="sklearn", random_state=seed
    )
    sc.tl.leiden(
        g, resolution=resolution, flavor="igraph", n_iterations=2, directed=False, random_state=seed, key_added="domain"
    )
    lab = g.obs["domain"].astype(str).to_numpy()
    full = np.full(adata.n_obs, None, dtype=object)
    full[t] = lab
    cats = sorted(set(lab), key=lambda s: (len(s), s))
    adata.obs[key_added] = pd.Categorical(full, categories=cats)
    emb = np.full((adata.n_obs, pcs.shape[1]), np.nan, dtype=np.float32)
    emb[t] = pcs
    adata.obsm[f"X_{key_added}"] = emb
    w = radius_graph(xy, radius_um, groups=lib[t]) if n_neighbors is None else None
    same = _agreement(w, lab) if w is not None and w.nnz else np.full(len(t), np.nan)
    _record(
        adata,
        f"domains_{key_added}",
        {
            "radius_um": radius_um,
            "n_neighbors": n_neighbors,
            "lambda": lambda_,
            "resolution": resolution,
            "n_genes": len(sel),
            "n_pcs": n_comp,
            "graph_neighbors": graph_neighbors,
            "seed": seed,
            "median_spatial_degree": float(np.median(deg)),
        },
    )
    df = pd.DataFrame({"domain": lab, "agree": same})
    out = df.groupby("domain", sort=False).agg(n_units=("agree", "size"), agreement=("agree", "mean"))
    return out.reindex(cats).reset_index()


def _agreement(w: sparse.csr_matrix, lab: np.ndarray) -> np.ndarray:
    """Share of each unit's neighbours (rows of ``w``) carrying the unit's own label (NaN without neighbours)."""
    codes, _ = pd.factorize(lab)
    rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    same = codes[rows] == codes[w.indices]
    num = np.bincount(rows, weights=same.astype(float), minlength=w.shape[0])
    deg = np.diff(w.indptr).astype(float)
    return np.where(deg > 0, num / np.maximum(deg, 1), np.nan)
