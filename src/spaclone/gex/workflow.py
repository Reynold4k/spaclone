"""Normalisation, highly variable genes, PCA, neighbours, Leiden and UMAP: thin scanpy wrappers with spatial-immune defaults.

Results go to the standard scanpy keys (``X`` log-normalised, ``layers["counts"]`` raw counts,
``var["highly_variable"]``, ``obsm["X_pca"]``, ``obsp["connectivities"]``, ``obs["leiden"]``, ``obsm["X_umap"]``), so
every scanpy / squidpy function works afterwards; the parameters of each step are recorded in
``uns["spaclone"]``. The defaults that differ from plain scanpy are the ones spatial immune data need:

* a gene detection floor before HVG selection (genes seen in < 0.5% of units are background; without the floor the
  HVGs of sparse spatial data can be pure noise);
* immunoglobulin and T-cell-receptor genes, mitochondrial, ribosomal and haemoglobin genes are not eligible as HVGs
  (Ig RNA spills over between neighbouring units and V genes track clonotype, not cell state). They stay in the
  matrix for marker scoring, Ig RNA and DE;
* reproducible results: the same ``rng`` gives the same HVGs, PCs, neighbour graph, clusters and embedding whatever the
  number of CPUs. Every step runs with one BLAS / OpenMP / numba thread (multi-threaded sums of gene means and of the
  PCA iterations differ in the last bits with the thread count), and the neighbour graph is exact up to
  50,000 units (approximate nearest-neighbour descent turns last-bit differences into different neighbours: on
  AIR-SPACE D21, 23,040 tissue beads, Leiden gave 11 clusters with 4 threads and 9 with 2 before this was fixed).
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterator, Sequence
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from spaclone._utils import RngLike, as_rng
from spaclone.gex.qc import _record, get_counts

__all__ = ["EXACT_KNN_MAX", "GENE_GROUPS", "cluster", "embed", "gene_group_mask", "preprocess"]

logger = logging.getLogger("spaclone")

#: Case-insensitive name patterns of gene groups excluded from HVG selection (mouse and human symbols).
GENE_GROUPS: dict[str, str] = {
    "ig": (
        r"^(IGH[MDE]|IGHG\d*[A-C]?|IGHA\d?|IGHV.*|IGHD\d.*|IGHJ\d*|IGKC|IGKV.*|IGKJ\d*|IGLC\d*|IGLV.*|IGLJ\d*|"
        r"IGLL\d*)$"
    ),
    "tr": r"^(TR[ABDG][VJ].*|TR[BD]D\d.*)$",
    "mt": r"^MT-.*$",
    "ribo": r"^(RP[SL]\d+[A-Z]?\d*|RPLP\d|RPSA|MRP[SL]\d+[A-Z]?)$",
    "hb": r"^(HB[AB](-.*|\d*)|HB[DEGMQZ]\d*)$",
}


def gene_group_mask(var_names: Sequence[str] | pd.Index, groups: str | Sequence[str] | None) -> np.ndarray:
    """Boolean mask of the genes that belong to the named groups of :data:`GENE_GROUPS` (or regular expressions).

    Parameters
    ----------
    var_names
        Gene names.
    groups
        Group names (``"ig"``, ``"tr"``, ``"mt"``, ``"ribo"``, ``"hb"``) and/or regular expressions, matched
        case-insensitively against the whole name. ``None`` or empty: no gene.

    Returns
    -------
    numpy.ndarray
        One boolean per gene.
    """
    names = pd.Index(var_names).astype(str)
    if not groups:
        return np.zeros(len(names), dtype=bool)
    groups = [groups] if isinstance(groups, str) else list(groups)
    out = np.zeros(len(names), dtype=bool)
    for g in groups:
        pat = re.compile(GENE_GROUPS.get(g, g), flags=re.IGNORECASE)
        out |= np.fromiter((bool(pat.match(n)) for n in names), dtype=bool, count=len(names))
    return out


#: Largest number of units for which ``knn="auto"`` builds the exact neighbour graph.
EXACT_KNN_MAX = 50_000


@contextlib.contextmanager
def _one_thread() -> Iterator[None]:
    """Run the enclosed scanpy steps with one BLAS / OpenMP thread and one numba thread.

    Multi-threaded reductions (gene means and variances in numba, the BLAS products of the PCA iterations) add in an
    order that depends on the number of threads, so their last bits change with the CPUs of the machine; the
    approximate neighbour search and Leiden can turn those bits into different clusters.
    """
    from threadpoolctl import threadpool_limits

    try:
        import numba
    except ImportError:  # pragma: no cover - numba is a scanpy dependency
        numba = None
    old = numba.get_num_threads() if numba is not None else None
    try:
        if numba is not None:
            numba.set_num_threads(1)
        with threadpool_limits(limits=1):
            yield
    finally:
        if numba is not None:
            numba.set_num_threads(old)


def _seed(rng: RngLike) -> int:
    """Integer seed for scanpy / igraph / umap from an int, a generator or a seed sequence."""
    if isinstance(rng, int | np.integer) and not isinstance(rng, bool):
        return int(rng)
    return int(as_rng(rng).integers(0, 2**31 - 1))


def preprocess(
    adata: ad.AnnData,
    *,
    target_sum: float = 1e4,
    n_hvg: int = 2000,
    n_pcs: int = 30,
    batch_key: str | None = None,
    hvg_flavor: str = "seurat",
    min_gene_frac: float = 0.005,
    exclude_hvg: str | Sequence[str] | None = ("ig", "tr", "mt", "ribo", "hb"),
    counts_layer: str = "counts",
    rng: RngLike = 0,
) -> pd.DataFrame:
    """Normalise, log-transform, select highly variable genes and run PCA (in place).

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts in ``layers[counts_layer]`` or, the first time, in ``X``.
    target_sum
        Counts per unit after ``scanpy.pp.normalize_total`` (CP10k by default).
    n_hvg
        Number of highly variable genes.
    n_pcs
        Number of principal components (capped by the data size).
    batch_key
        ``obs`` column passed to ``scanpy.pp.highly_variable_genes`` (HVGs chosen within batches and merged).
    hvg_flavor
        scanpy HVG flavour; ``"seurat"`` and ``"cell_ranger"`` use the log data, ``"seurat_v3"`` the counts.
    min_gene_frac
        Only genes detected (count > 0) in at least this share of units can be HVGs.
    exclude_hvg
        Gene groups (see :data:`GENE_GROUPS`) or patterns that cannot be HVGs.
    counts_layer
        Layer that keeps (or already holds) the raw counts. ``X`` is always rebuilt from it, so calling
        ``preprocess`` twice gives the same result.
    rng
        Seed for the PCA solver.

    Returns
    -------
    pandas.DataFrame
        The HVG table of the eligible genes (scanpy columns, plus ``eligible``).

    Notes
    -----
    Writes ``X`` (log1p of normalised counts), ``layers[counts_layer]``, ``var["highly_variable"]``,
    ``var["hvg_eligible"]``, ``obsm["X_pca"]``, ``varm["PCs"]``, ``uns["pca"]`` and
    ``uns["spaclone"]["preprocess"]``. Runs with one thread (BLAS, OpenMP, numba), so the same ``rng`` gives the same
    PCs bit for bit whatever the number of CPUs.

    References
    ----------
    Wolf, Angerer & Theis 2018, doi:10.1186/s13059-017-1382-0 (scanpy); Satija et al. 2015, doi:10.1038/nbt.3192
    (Seurat HVG flavour); Luecken & Theis 2019, doi:10.15252/msb.20188746 (best practices).
    """
    with _one_thread():
        return _preprocess(
            adata,
            target_sum=target_sum,
            n_hvg=n_hvg,
            n_pcs=n_pcs,
            batch_key=batch_key,
            hvg_flavor=hvg_flavor,
            min_gene_frac=min_gene_frac,
            exclude_hvg=exclude_hvg,
            counts_layer=counts_layer,
            rng=rng,
        )


def _preprocess(
    adata: ad.AnnData,
    *,
    target_sum: float,
    n_hvg: int,
    n_pcs: int,
    batch_key: str | None,
    hvg_flavor: str,
    min_gene_frac: float,
    exclude_hvg: str | Sequence[str] | None,
    counts_layer: str,
    rng: RngLike,
) -> pd.DataFrame:
    import scanpy as sc

    if counts_layer not in adata.layers:
        adata.layers[counts_layer] = get_counts(adata, None).astype(np.float32)
    counts = get_counts(adata, counts_layer)
    adata.X = counts.astype(np.float32, copy=True)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    det = np.asarray((counts > 0).sum(axis=0)).ravel()
    excluded = gene_group_mask(adata.var_names, exclude_hvg)
    eligible = (det >= min_gene_frac * adata.n_obs) & (det > 0) & ~excluded
    if eligible.sum() < 2:
        raise ValueError(f"only {int(eligible.sum())} genes pass the detection floor and exclusions")
    sub = adata[:, eligible].copy()
    n_top = int(min(n_hvg, eligible.sum()))
    kw = {"layer": counts_layer} if hvg_flavor.startswith("seurat_v3") else {}
    hv = sc.pp.highly_variable_genes(
        sub, flavor=hvg_flavor, n_top_genes=n_top, batch_key=batch_key, inplace=False, **kw
    )
    hv.index = sub.var_names
    adata.var["hvg_eligible"] = eligible
    adata.var["highly_variable"] = False
    adata.var.loc[hv.index[hv["highly_variable"].to_numpy(bool)], "highly_variable"] = True
    n_sel = int(adata.var["highly_variable"].sum())
    n_comps = int(min(n_pcs, adata.n_obs - 1, n_sel - 1))
    if n_comps < 1:
        raise ValueError("too few units or HVGs for PCA")
    sc.pp.pca(adata, n_comps=n_comps, mask_var="highly_variable", random_state=_seed(rng))
    _record(
        adata,
        "preprocess",
        {
            "target_sum": target_sum,
            "n_hvg": n_hvg,
            "n_hvg_selected": n_sel,
            "n_pcs": n_comps,
            "batch_key": batch_key,
            "hvg_flavor": hvg_flavor,
            "min_gene_frac": min_gene_frac,
            "exclude_hvg": [exclude_hvg] if isinstance(exclude_hvg, str) else list(exclude_hvg or []),
            "n_excluded": int(excluded.sum()),
            "n_eligible": int(eligible.sum()),
            "counts_layer": counts_layer,
            "seed": _seed(rng),
            "threads": 1,
        },
    )
    logger.info("gex.preprocess: %d eligible genes, %d HVGs, %d PCs", eligible.sum(), n_sel, n_comps)
    return hv.assign(eligible=True)


Knn = Literal["auto", "exact", "approximate"]


def _knn_method(adata: ad.AnnData, knn: Knn) -> str:
    if knn not in ("auto", "exact", "approximate"):
        raise ValueError(f"knn must be 'auto', 'exact' or 'approximate', got {knn!r}")
    if knn == "auto":
        return "exact" if adata.n_obs <= EXACT_KNN_MAX else "approximate"
    return knn


def _ensure_neighbors(
    adata: ad.AnnData,
    *,
    n_neighbors: int,
    use_rep: str,
    n_pcs: int | None,
    seed: int,
    force: bool,
    knn: Knn = "auto",
) -> str:
    """Build the neighbour graph unless a matching one exists; returns the kNN method of the graph in use.

    ``"exact"``: brute-force Euclidean neighbours (scikit-learn), the same neighbours whatever the last bits of the
    representation. ``"approximate"``: scanpy's nearest-neighbour descent (pynndescent) with one thread and the seed.
    A graph from :func:`scanpy.pp.neighbors` itself (no spaclone record) is reused when its parameters match.
    """
    import scanpy as sc

    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] is missing; run spaclone.gex.preprocess first")
    method = _knn_method(adata, knn)
    p = adata.uns.get("neighbors", {}).get("params", {})
    rec = (adata.uns.get("spaclone") or {}).get("neighbors") if isinstance(adata.uns.get("spaclone"), dict) else None
    same = (
        "connectivities" in adata.obsp
        and int(p.get("n_neighbors", -1)) == n_neighbors
        and p.get("use_rep", use_rep) == use_rep
        and (n_pcs is None or int(p.get("n_pcs", -1)) == n_pcs)
        and (rec is None or (rec.get("knn") == method and int(rec.get("seed", seed)) == seed))
    )
    if force or not same:
        with _one_thread():
            if method == "exact":
                sc.pp.neighbors(
                    adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, random_state=seed,
                    transformer="sklearn",
                )  # fmt: skip
            else:
                from pynndescent import PyNNDescentTransformer

                n = adata.n_obs
                tr = PyNNDescentTransformer(
                    n_neighbors=n_neighbors,
                    metric="euclidean",
                    random_state=seed,
                    n_jobs=1,
                    n_trees=min(64, 5 + round(n**0.5 / 20.0)),
                    n_iters=max(5, round(np.log2(n))),
                )  # scanpy's own settings, with one thread
                sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep=use_rep, random_state=seed, transformer=tr)
        _record(adata, "neighbors", {"knn": method, "n_neighbors": n_neighbors, "use_rep": use_rep, "seed": seed})
        return method
    return rec.get("knn", "scanpy") if rec else "scanpy"


def cluster(
    adata: ad.AnnData,
    *,
    resolution: float = 1.0,
    method: str = "leiden",
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str = "X_pca",
    key_added: str | None = None,
    recompute_neighbors: bool = False,
    knn: Knn = "auto",
    rng: RngLike = 0,
) -> pd.DataFrame:
    """Leiden (or Louvain) clustering on the expression neighbour graph (in place).

    Parameters
    ----------
    adata
        AnnData after :func:`preprocess` (or with any representation in ``obsm[use_rep]``).
    resolution
        Resolution of the modularity (higher = more clusters).
    method
        ``"leiden"`` (igraph implementation, 2 iterations, undirected) or ``"louvain"`` (needs ``louvain``).
    n_neighbors, n_pcs, use_rep
        Neighbour graph (``scanpy.pp.neighbors``); recomputed only if absent or built with other values.
    key_added
        ``obs`` column of the labels (default the method name).
    recompute_neighbors
        Rebuild the neighbour graph even if a matching one exists.
    knn
        Neighbour search: ``"exact"`` (brute force), ``"approximate"`` (nearest-neighbour descent, scanpy's default
        above 8,192 units) or ``"auto"`` (exact up to :data:`EXACT_KNN_MAX` units, approximate above).
    rng
        Seed of the graph construction and the community search.

    Returns
    -------
    pandas.DataFrame
        Cluster sizes (``cluster``, ``n_units``).

    Notes
    -----
    Deterministic: the same ``rng`` gives the same labels on any number of CPUs. The graph and the community search
    run with one thread, the exact graph does not depend on the last bits of the PCs, and Leiden (igraph) is seeded.
    The kNN method is recorded in ``uns["spaclone"]["neighbors"]``.

    References
    ----------
    Traag, Waltman & van Eck 2019, doi:10.1038/s41598-019-41695-z (Leiden); Blondel et al. 2008,
    doi:10.1088/1742-5468/2008/10/P10008 (Louvain).
    """
    import scanpy as sc

    if method not in ("leiden", "louvain"):
        raise ValueError(f"method must be 'leiden' or 'louvain', got {method!r}")
    seed = _seed(rng)
    key = key_added or method
    used = _ensure_neighbors(
        adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs=n_pcs, seed=seed, force=recompute_neighbors, knn=knn
    )
    with _one_thread():
        if method == "leiden":
            sc.tl.leiden(
                adata,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=seed,
                key_added=key,
            )
        else:
            sc.tl.louvain(adata, resolution=resolution, random_state=seed, key_added=key)
    _record(
        adata,
        f"cluster_{key}",
        {
            "method": method,
            "resolution": resolution,
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
            "n_pcs": n_pcs,
            "knn": used,
            "seed": seed,
        },
    )
    vc = adata.obs[key].value_counts(sort=False)
    return pd.DataFrame({"cluster": vc.index.astype(str), "n_units": vc.to_numpy()})


def embed(
    adata: ad.AnnData,
    *,
    method: str = "umap",
    min_dist: float = 0.5,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    knn: Knn = "auto",
    rng: RngLike = 0,
) -> None:
    """Two-dimensional embedding of the expression neighbour graph (UMAP; in place, ``obsm["X_umap"]``).

    Parameters
    ----------
    adata
        AnnData after :func:`preprocess`.
    method
        Only ``"umap"``.
    min_dist
        UMAP minimum distance.
    n_neighbors, use_rep, knn
        Neighbour graph (built if absent, see :func:`cluster`).
    rng
        Seed (UMAP runs with one thread, so the embedding does not depend on the number of CPUs).

    References
    ----------
    McInnes, Healy & Melville 2018, doi:10.48550/arXiv.1802.03426.
    """
    import scanpy as sc

    if method != "umap":
        raise ValueError(f"method must be 'umap', got {method!r}")
    seed = _seed(rng)
    _ensure_neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs=None, seed=seed, force=False, knn=knn)
    with _one_thread():
        sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
    _record(adata, "embed", {"method": method, "min_dist": min_dist, "n_neighbors": n_neighbors, "seed": seed})
