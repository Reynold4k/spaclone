"""squidpy on receptor-derived labels: neighbourhood enrichment, co-occurrence and spatial autocorrelation.

:func:`receptor_labels` puts what the receptor records say about each spot onto the GEX AnnData (or onto a spot
table built from the records): the main isotype class, switched and mutated shares, whether the spot carries an
expanded clone, and which of the largest clones it carries. :func:`nhood_enrichment`, :func:`co_occurrence` and
:func:`spatial_autocorr` then run squidpy's own functions on those labels, with squidpy's own neighbour graph
(``squidpy.gr.spatial_neighbors_knn`` / ``_radius``; ``spatial_neighbors`` before squidpy 1.8), and return tidy
:class:`~spaclone.TestResult` tables.

.. warning::

   squidpy's nulls relabel spots at random over the tissue. They keep neither the local Ig / TCR RNA depth nor cell
   density, so any receptor label that follows where B or T cells are (every one of them does) comes out enriched,
   co-occurring and autocorrelated, also when the receptor positions themselves are random (barcode swapping).
   Every function here warns (:class:`DepthNullWarning`). Whether receptor positions carry information beyond
   depth is tested by :func:`spaclone.qc.colocalisation` (same-clone pairs against clone labels shuffled within
   depth strata); covariate-matched maps and calibrated hotspots are in :mod:`spaclone.tl`.

squidpy is optional: ``pip install 'squidpy>=1.6'`` (the ``spaclone[squidpy]`` extra).

References
----------
Palla et al. 2022, doi:10.1038/s41592-021-01358-2 (squidpy); Moran 1950, doi:10.2307/2332142; Geary 1954,
doi:10.2307/2986645.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, bh, rng_entropy

__all__ = [
    "LABELS",
    "DepthNullWarning",
    "co_occurrence",
    "nhood_enrichment",
    "receptor_labels",
    "spatial_autocorr",
]

logger = logging.getLogger("spaclone")

#: Labels that :func:`receptor_labels` can compute.
LABELS: tuple[str, ...] = ("isotype", "switched", "mutated", "expanded", "clone")
_GRAPH_KEY = "spaclone_squidpy"


class DepthNullWarning(SpacloneWarning):
    """A spatial statistic was compared with random relabelling of units, which controls neither depth nor density."""


def _depth_warning(what: str) -> None:
    warnings.warn(
        f"{what}: the null relabels units at random over the tissue and keeps neither the local receptor (Ig / TCR) "
        "RNA depth nor cell density, so receptor labels that follow B- or T-cell density look structured even when "
        "receptor positions are random. Test whether positions carry clonal information with "
        "spaclone.qc.colocalisation (clone labels shuffled within depth strata).",
        DepthNullWarning,
        stacklevel=3,
    )


def _sq():
    try:
        import squidpy as sq
    except ImportError as e:
        raise ImportError("spaclone.ext.squidpy needs squidpy: pip install 'squidpy>=1.6' (spaclone[squidpy])") from e
    return sq


def _seed_kw(func: Any, seed: int) -> dict[str, int]:
    """Seed keyword of a squidpy function: ``rng=`` in squidpy >= 1.7, ``seed=`` before."""
    names = inspect.signature(func).parameters
    return {"rng": seed} if "rng" in names else ({"seed": seed} if "seed" in names else {})


def _quiet_kw(func: Any) -> dict[str, bool]:
    return {"show_progress_bar": False} if "show_progress_bar" in inspect.signature(func).parameters else {}


# ------------------------------------------------------------------------------------------------- labels


def _spot_frame(air: pd.DataFrame, spot_key: str, x_key: str, y_key: str, library_key: str):
    """One AnnData observation per distinct (library, spot) of the records, at the spot's first position."""
    import anndata as ad

    lib = library_key if library_key in air.columns else None
    keys = ([lib] if lib else []) + [spot_key]
    first = air.drop_duplicates(keys)[[*keys, x_key, y_key]]
    names = first[spot_key].astype(str)
    if lib is not None and first[lib].nunique() > 1:
        names = first[lib].astype(str) + ":" + names
    obs = pd.DataFrame(index=pd.Index(names.to_numpy(dtype=object)))
    if lib is not None:
        obs[library_key] = pd.Categorical(first[lib].astype(str).to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.AnnData(obs=obs, obsm={"spatial": first[[x_key, y_key]].to_numpy(dtype=float)})
    return a


def _match(
    air: pd.DataFrame,
    gex: Any,
    match: str,
    max_distance_um: float | None,
    spot_key: str,
    x_key: str,
    y_key: str,
    library_key: str,
) -> tuple[np.ndarray, str]:
    """Observation position of every record in ``gex`` (-1 when unmatched) and the rule used."""
    from scipy.spatial import cKDTree

    names = pd.Index(np.asarray(gex.obs_names, dtype=object).astype(str))
    sid = air[spot_key].astype(str)
    multi_lib = library_key in air.columns and air[library_key].nunique() > 1
    if multi_lib:
        sid = air[library_key].astype(str) + ":" + sid
    if match == "auto":
        hit = sid.isin(names).mean() if len(sid) else 0.0
        match = "spot_id" if hit >= 0.5 else "nearest"
    if match == "spot_id":
        return names.get_indexer(sid.to_numpy()), "spot_id"
    if match != "nearest":
        raise ValueError(f"match must be 'auto', 'spot_id' or 'nearest', got {match!r}")
    gxy = np.asarray(gex.obsm["spatial"], dtype=float)[:, :2]
    tree = cKDTree(gxy)
    if max_distance_um is None:
        d, _ = tree.query(gxy, k=2) if len(gxy) > 1 else (np.zeros((1, 2)), None)
        spacing = float(np.median(d[:, 1])) if len(gxy) > 1 else 0.0
        max_distance_um = 0.75 * spacing
    rxy = air[[x_key, y_key]].to_numpy(dtype=float)
    pos = np.full(len(air), -1, dtype=np.int64)
    glib = gex.obs[library_key].astype(str).to_numpy() if library_key in gex.obs.columns else None
    if glib is not None and library_key in air.columns:
        alib = air[library_key].astype(str).to_numpy()
        for lv in np.unique(alib):
            gi = np.flatnonzero(glib == lv)
            ri = np.flatnonzero(alib == lv)
            if len(gi) == 0 or len(ri) == 0:
                continue
            d, j = cKDTree(gxy[gi]).query(rxy[ri], k=1)
            ok = d <= max_distance_um + 1e-9
            pos[ri[ok]] = gi[j[ok]]
    else:
        d, j = tree.query(rxy, k=1)
        ok = d <= max_distance_um + 1e-9
        pos[ok] = j[ok]
    return pos, "nearest"


def _share_label(share: np.ndarray, yes: str, no: str) -> pd.Categorical:
    lab = np.where(np.isnan(share), None, np.where(share > 0.5, yes, np.where(share < 0.5, no, "mixed")))
    return pd.Categorical(lab, categories=[c for c in (no, "mixed", yes) if (lab == c).any()])


def receptor_labels(
    air: pd.DataFrame,
    gex: Any = None,
    *,
    labels: str | Sequence[str] = "auto",
    match: Literal["auto", "spot_id", "nearest"] = "auto",
    max_distance_um: float | None = None,
    n_top_clones: int = 7,
    min_size: int = 2,
    radius_um: float = 15.0,
    species: Literal["mouse", "human"] | None = "mouse",
    prefix: str = "air_",
    isotype_key: str = "isotype",
    switched_key: str = "switched",
    mutated_key: str | None = None,
    chains: Sequence[str] | None = S.HEAVY_LOCI,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    c_key: str = S.C_CALL,
):
    """Receptor-derived labels on every spot of the GEX AnnData (or of a spot table built from the records).

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or cell call).
    gex
        Spot-level AnnData with ``obsm["spatial"]`` in µm. ``None`` builds one observation per receptor spot.
    labels
        Which labels: any of :data:`LABELS`, or ``"auto"`` (every label the table supports; ``mutated`` needs a
        mutated column, ``isotype`` / ``switched`` need heavy-chain constant calls or an isotype column).
    match
        How records find their spot: ``"spot_id"`` (record ``spot_key`` = ``gex.obs_names``; ``"library:spot"`` with
        several sections), ``"nearest"`` (nearest GEX spot within ``max_distance_um``) or ``"auto"`` (``spot_id``
        when at least half of the record spots are GEX spot names).
    max_distance_um
        Largest record-to-spot distance for ``match="nearest"`` (default 0.75 × the median spot spacing).
    n_top_clones
        The ``clone`` label names the largest clones (by loci); other clones are ``"other"``.
    min_size
        A clone is expanded when it has at least this many loci (:func:`spaclone.tl.clone_sizes`).
    radius_um
        Locus radius for clone sizes.
    species
        Species for :func:`spaclone.bcr.isotype` when the isotype is computed from ``c_key``.
    prefix
        Prefix of the new ``obs`` columns.
    isotype_key, switched_key
        Isotype class and switched columns of ``air`` (computed with :func:`spaclone.bcr.isotype` when missing).
    mutated_key
        Boolean mutated column (default: the first of ``"mutated"`` and ``"shm_mutated"`` present).
    chains
        Loci whose records define clone sizes and the ``expanded`` / ``clone`` labels (default the heavy chains IGH,
        TRB, TRD: light-chain junctions are shared by unrelated cells, so light-chain "clones" span the section and
        make most spots look expanded); ``None`` = every locus.
    spot_key, x_key, y_key, clone_key, locus_key, library_key, c_key
        Column names.

    Returns
    -------
    anndata.AnnData
        A copy of ``gex`` (or the new spot AnnData) with ``obs`` columns ``{prefix}n_records`` and, per label:
        ``{prefix}isotype`` (main class of the spot's heavy-chain records with a known class), ``{prefix}switched``
        (``"switched"`` / ``"unswitched"`` / ``"mixed"`` by majority) and ``{prefix}switched_share``,
        ``{prefix}mutated`` and ``{prefix}mutated_share``, ``{prefix}expanded`` (``"expanded"`` when the spot carries
        a record of an expanded clone of ``chains``, else ``"not expanded"``) and ``{prefix}expanded_share`` (share of
        the spot's clones that are expanded), ``{prefix}clone`` (the top clone with most records on the spot,
        ``"other"``).
        Spots without a usable record have missing values. ``uns["spaclone"]["receptor_labels"]`` records the rule.

    Notes
    -----
    Labels follow the records, so they also follow everything that moves records: depth of receptor RNA, barcode
    swapping, sink spots. Use the label maps to describe; test with :func:`spaclone.qc.colocalisation` and the
    covariate-matched maps of :mod:`spaclone.tl`.
    """
    from spaclone.bcr.isotype import ISOTYPE_CLASSES
    from spaclone.bcr.isotype import isotype as call_isotype
    from spaclone.tl.repertoire import clone_sizes

    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a receptor table, got {type(air).__name__}")
    S.validate_air(air, required=(spot_key, x_key, y_key), numeric=(x_key, y_key))
    out = _spot_frame(air, spot_key, x_key, y_key, library_key) if gex is None else gex.copy()
    if "spatial" not in out.obsm:
        raise KeyError("receptor_labels: gex has no obsm['spatial'] (µm)")
    pos, rule = _match(air, out, match, max_distance_um, spot_key, x_key, y_key, library_key)
    ok = pos >= 0
    n_obs = out.n_obs
    a = air.loc[ok].copy()
    p = pos[ok]
    out.obs[f"{prefix}n_records"] = np.bincount(p, minlength=n_obs).astype(np.int64)
    if mutated_key is None:
        mutated_key = next((c for c in ("mutated", "shm_mutated") if c in a.columns), "mutated")
    have = {
        "isotype": isotype_key in a.columns or c_key in a.columns,
        "switched": switched_key in a.columns or c_key in a.columns,
        "mutated": mutated_key in a.columns,
        "expanded": clone_key in a.columns,
        "clone": clone_key in a.columns,
    }
    if labels == "auto":
        want = [lab for lab in LABELS if have[lab]]
    else:
        want = [labels] if isinstance(labels, str) else list(labels)
        bad = [lab for lab in want if lab not in LABELS]
        if bad:
            raise ValueError(f"unknown label(s) {bad}; choose from {LABELS}")
        lacking = [lab for lab in want if not have[lab]]
        if lacking:
            hints = {
                "isotype": f"a {c_key!r} or {isotype_key!r} column",
                "switched": f"a {c_key!r} or {switched_key!r} column",
                "mutated": "a boolean mutated column (spaclone.bcr.shm + call_shm, or mutated_key=)",
                "expanded": f"a {clone_key!r} column",
                "clone": f"a {clone_key!r} column",
            }
            raise KeyError("receptor_labels: " + "; ".join(f"{lab} needs {hints[lab]}" for lab in lacking))
    if ("isotype" in want or "switched" in want) and (isotype_key not in a.columns or switched_key not in a.columns):
        iso = call_isotype(a, species=species, c_key=c_key, locus_key=locus_key)
        if isotype_key not in a.columns:
            a[isotype_key] = iso["isotype"]
        if switched_key not in a.columns:
            a[switched_key] = iso["switched"]
    if "isotype" in want:
        cls = a[isotype_key].astype("string")
        known = cls.isin(list(ISOTYPE_CLASSES)).fillna(False).to_numpy(dtype=bool)
        tab = pd.crosstab(p[known], cls[known].to_numpy())
        order = [c for c in ISOTYPE_CLASSES if c in tab.columns]
        tab = tab[order]
        main = pd.Series([None] * n_obs, dtype=object)
        if len(tab):
            main.iloc[tab.index.to_numpy()] = [order[i] for i in tab.to_numpy().argmax(axis=1)]
        out.obs[f"{prefix}isotype"] = pd.Categorical(main.to_numpy(dtype=object), categories=order)
    for lab, col, yes, no in (
        ("switched", switched_key, "switched", "unswitched"),
        ("mutated", mutated_key, "mutated", "unmutated"),
    ):
        if lab not in want:
            continue
        v = pd.array(a[col], dtype="boolean")
        known = ~np.asarray(v.isna(), dtype=bool)
        num = np.bincount(p[known], weights=np.asarray(v[known], dtype=float), minlength=n_obs)
        den = np.bincount(p[known], minlength=n_obs).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        out.obs[f"{prefix}{lab}_share"] = share
        out.obs[f"{prefix}{lab}"] = _share_label(share, yes, no)
    if "expanded" in want or "clone" in want:
        if chains is not None and locus_key not in air.columns:
            raise KeyError(f"receptor_labels: chains= needs a {locus_key!r} column (or chains=None)")
        loc = {str(c).upper() for c in chains} if chains is not None else None

        def on_chains(t: pd.DataFrame) -> np.ndarray:
            has = t[clone_key].notna().to_numpy(dtype=bool)
            if loc is not None:
                has = has & t[locus_key].astype("string").str.upper().isin(loc).fillna(False).to_numpy(dtype=bool)
            return has

        full = air[on_chains(air)]
        sizes = clone_sizes(
            full,
            unit="locus",
            radius_um=radius_um,
            clone_key=clone_key,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            locus_key=locus_key,
            library_key=library_key,
        )
        size_of = sizes.groupby(clone_key, observed=True)["size"].sum()
        has_clone = on_chains(a)
        ac = a.loc[has_clone, clone_key].astype(object)
        pc = p[has_clone]
        if "expanded" in want:
            big = (ac.map(size_of).fillna(0).to_numpy(dtype=float) >= min_size).astype(float)
            pairs = pd.DataFrame({"o": pc, "c": ac.to_numpy(), "b": big}).drop_duplicates(["o", "c"])
            n_cl = np.bincount(pairs.o, minlength=n_obs).astype(float)
            n_big = np.bincount(pairs.o, weights=pairs.b, minlength=n_obs)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(n_cl > 0, n_big / np.where(n_cl > 0, n_cl, 1), np.nan)
            out.obs[f"{prefix}expanded_share"] = share
            lab = np.where(n_cl == 0, None, np.where(n_big > 0, "expanded", "not expanded"))
            out.obs[f"{prefix}expanded"] = pd.Categorical(lab, categories=["not expanded", "expanded"])
        if "clone" in want:
            top = list(size_of.sort_values(ascending=False, kind="stable").index[: max(int(n_top_clones), 0)])
            cnt = pd.DataFrame({"o": pc, "c": ac.to_numpy()})
            cnt = cnt[cnt.c.isin(top)].groupby(["o", "c"], sort=True).size().reset_index(name="n")
            cnt = cnt.sort_values(["o", "n"], ascending=[True, False], kind="stable").drop_duplicates("o")
            lab = pd.Series([None] * n_obs, dtype=object)
            any_clone = np.bincount(pc, minlength=n_obs) > 0
            lab[any_clone] = "other"
            lab.iloc[cnt.o.to_numpy()] = cnt.c.astype(str).to_numpy()
            cats = [str(c) for c in top if (lab == str(c)).any()] + (["other"] if (lab == "other").any() else [])
            out.obs[f"{prefix}clone"] = pd.Categorical(lab.to_numpy(dtype=object), categories=cats)
    out.uns.setdefault("spaclone", {})
    if not isinstance(out.uns["spaclone"], dict):
        out.uns["spaclone"] = {}
    out.uns["spaclone"]["receptor_labels"] = {
        "labels": list(want),
        "match": rule,
        "matched_records": int(ok.sum()),
        "records": len(air),
        "min_size": int(min_size),
        "chains": list(chains) if chains is not None else None,
        "n_top_clones": int(n_top_clones),
        "prefix": prefix,
    }
    logger.info("receptor_labels: %d of %d records matched to spots (%s)", int(ok.sum()), len(air), rule)
    return out


# ------------------------------------------------------------------------------------------------- squidpy


def _subset(adata: Any, keys: Sequence[str], missing: str | None) -> Any:
    """Units with a value in every key (or missing values filled with ``missing``), as a copy."""
    sub = adata.copy()
    for k in keys:
        if k not in sub.obs.columns:
            raise KeyError(f"obs has no column {k!r}")
    if missing is not None and missing != "drop":
        for k in keys:
            s = sub.obs[k]
            if isinstance(s.dtype, pd.CategoricalDtype):
                if missing not in s.cat.categories:
                    s = s.cat.add_categories([missing])
                sub.obs[k] = s.fillna(missing)
            elif not pd.api.types.is_numeric_dtype(s.dtype):
                sub.obs[k] = s.astype(object).where(s.notna(), missing)
        return sub
    keep = np.ones(sub.n_obs, dtype=bool)
    for k in keys:
        keep &= sub.obs[k].notna().to_numpy(dtype=bool)
    return sub[keep].copy()


def _categorical(adata: Any, key: str) -> None:
    s = adata.obs[key]
    if isinstance(s.dtype, pd.CategoricalDtype):
        adata.obs[key] = s.cat.remove_unused_categories()
    else:
        adata.obs[key] = pd.Categorical(s.astype(str))


def _graph(
    adata: Any,
    *,
    connectivity_key: str | None,
    radius_um: float | None,
    n_neighbors: int | None,
    library_key: str | None,
    spatial_key: str,
) -> str:
    if connectivity_key is not None:
        if connectivity_key not in adata.obsp:
            raise KeyError(f"adata.obsp[{connectivity_key!r}] is missing")
        return connectivity_key
    sq = _sq()
    kw: dict[str, Any] = {"spatial_key": spatial_key, "key_added": _GRAPH_KEY}
    if library_key is not None:
        _categorical(adata, library_key)
        kw["library_key"] = library_key
    # squidpy >= 1.8 has one function per graph (its spatial_neighbors is deprecated and goes in 1.9): the same
    # generic radius / k-nearest graphs; older squidpy builds them with spatial_neighbors(coord_type="generic")
    by_radius = getattr(sq.gr, "spatial_neighbors_radius", None)
    by_knn = getattr(sq.gr, "spatial_neighbors_knn", None)
    if radius_um is not None:
        if by_radius is not None:
            by_radius(adata, radius=float(radius_um), **kw)
        else:
            sq.gr.spatial_neighbors(adata, coord_type="generic", radius=float(radius_um), **kw)
    elif by_knn is not None:
        by_knn(adata, n_neighs=int(n_neighbors or 6), **kw)
    else:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=int(n_neighbors or 6), **kw)
    return f"{_GRAPH_KEY}_connectivities"


def nhood_enrichment(
    adata: Any,
    *,
    cluster_key: str,
    radius_um: float | None = None,
    n_neighbors: int | None = 6,
    connectivity_key: str | None = None,
    library_key: str | None = None,
    n_perms: int = 1000,
    missing: str | None = "drop",
    rng: RngLike = None,
    spatial_key: str = "spatial",
) -> TestResult:
    """Neighbourhood enrichment of a receptor label with squidpy (``squidpy.gr.nhood_enrichment``).

    Parameters
    ----------
    adata
        AnnData with the label (e.g. from :func:`receptor_labels`) and ``obsm[spatial_key]`` in µm.
    cluster_key
        Categorical ``obs`` column (``"air_isotype"``, ``"air_switched"``, ``"air_expanded"``, ``"air_clone"``, ...).
    radius_um, n_neighbors
        squidpy's generic graph (``spatial_neighbors_radius`` / ``spatial_neighbors_knn``; before squidpy 1.8
        ``spatial_neighbors(coord_type="generic")``): units within ``radius_um``, else the
        ``n_neighbors`` nearest.
    connectivity_key
        ``obsp`` key of an existing graph instead.
    library_key
        Section column (graph within sections; squidpy permutes within sections).
    n_perms
        squidpy's permutations.
    missing
        ``"drop"``: units without a label are left out (the graph is built on the labelled units); a string: they
        become that category (e.g. ``"none"`` for spots without receptor records).
    rng
        Seed or generator (squidpy receives an integer seed drawn from it).
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    TestResult
        ``table``: one row per unordered pair of categories with ``cluster_a``, ``cluster_b``, ``statistic``
        (squidpy's edge count), ``z`` (squidpy's z-score against its permutations), ``p_norm`` (two-sided normal
        approximation of ``z``: squidpy keeps no permutation distribution, so there is no Monte-Carlo p-value),
        ``q_norm`` (BH over ``p_norm``), ``n_perm``. ``summary["n_q05_between"]`` counts pairs of different
        categories with ``q_norm < 0.05``.

    Warns
    -----
    DepthNullWarning
        Always (see the module note).

    Null model
    ----------
    squidpy permutes the labels over the units (within sections when ``library_key`` is given); the
    positions and the graph are fixed. Neither receptor depth nor cell density is kept, so labels carried by B-cell
    areas are enriched with each other whatever the receptor positions mean. Monte-Carlo p-values of the same
    statistic are given by :func:`spaclone.gex.nhood_enrichment`.

    References
    ----------
    Palla et al. 2022, doi:10.1038/s41592-021-01358-2; Schapiro et al. 2017, doi:10.1038/nmeth.4391.
    """
    sq = _sq()
    gen = as_rng(rng)
    sub = _subset(adata, [cluster_key], missing)
    _categorical(sub, cluster_key)
    key = _graph(
        sub,
        connectivity_key=connectivity_key,
        radius_um=radius_um,
        n_neighbors=n_neighbors,
        library_key=library_key,
        spatial_key=spatial_key,
    )
    seed = int(gen.integers(0, 2**31 - 1))
    _depth_warning("ext.squidpy.nhood_enrichment")
    kw: dict[str, Any] = {"cluster_key": cluster_key, "connectivity_key": key, "n_perms": int(n_perms), "copy": True}
    if library_key is not None:
        kw["library_key"] = library_key
    res = sq.gr.nhood_enrichment(
        sub, **kw, **_seed_kw(sq.gr.nhood_enrichment, seed), **_quiet_kw(sq.gr.nhood_enrichment)
    )
    zscore, count = np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
    cats = [str(c) for c in sub.obs[cluster_key].cat.categories]
    iu = np.triu_indices(len(cats))
    z = zscore[iu]
    p = np.where(np.isfinite(z), 2 * stats.norm.sf(np.abs(z)), np.nan)
    table = pd.DataFrame(
        {
            "cluster_a": [cats[i] for i in iu[0]],
            "cluster_b": [cats[j] for j in iu[1]],
            "statistic": count[iu],
            "z": z,
            "p_norm": p,
            "q_norm": bh(p),
            "n_perm": int(n_perms),
        }
    )
    params = {
        "cluster_key": cluster_key,
        "radius_um": radius_um,
        "n_neighbors": n_neighbors if radius_um is None else None,
        "connectivity_key": connectivity_key,
        "library_key": library_key,
        "n_perms": int(n_perms),
        "missing": missing,
        "seed": seed,
        "rng_entropy": rng_entropy(gen),
        "design": "squidpy: labels permuted over units (within library), graph fixed",
        "squidpy_version": getattr(sq, "__version__", None),
    }
    off = table[table.cluster_a != table.cluster_b]
    summary = {
        "n_units": int(sub.n_obs),
        "n_classes": len(cats),
        "n_edges": int(sub.obsp[key].nnz),
        "n_q05_between": int((off.q_norm < 0.05).sum()),
    }
    return TestResult(name="squidpy_nhood_enrichment", table=table, summary=summary, params=params)


def co_occurrence(
    adata: Any,
    *,
    cluster_key: str,
    distances_um: Sequence[float] = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0),
    missing: str | None = "drop",
    spatial_key: str = "spatial",
) -> TestResult:
    """Co-occurrence of receptor labels with distance, with squidpy (``squidpy.gr.co_occurrence``).

    Parameters
    ----------
    adata
        AnnData with the label and ``obsm[spatial_key]`` in µm.
    cluster_key
        Categorical ``obs`` column.
    distances_um
        squidpy's ``interval``: increasing distances in µm; the ratios are reported at each distance after the first.
    missing
        ``"drop"`` or a fill category (see :func:`nhood_enrichment`).
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    TestResult
        ``table``: one row per (cluster, neighbour, distance) with ``cluster`` (the conditioning label),
        ``neighbour``, ``distance_um`` and ``statistic`` = squidpy's ratio p(neighbour | within ``distance_um`` of
        ``cluster``) / p(neighbour) (1 = what random labelling gives).

    Warns
    -----
    DepthNullWarning
        Always: the reference value 1 is random labelling of the units.

    Null model
    ----------
    None is simulated; the ratio is relative to random labelling of the units (positions fixed), which
    keeps neither receptor depth nor cell density. squidpy computes all pairwise distances (n² time); subset large
    sections.

    References
    ----------
    Palla et al. 2022, doi:10.1038/s41592-021-01358-2.
    """
    sq = _sq()
    sub = _subset(adata, [cluster_key], missing)
    _categorical(sub, cluster_key)
    interval = np.asarray(sorted(float(d) for d in distances_um), dtype=float)
    if len(interval) < 2:
        raise ValueError("distances_um needs at least two values")
    _depth_warning("ext.squidpy.co_occurrence")
    occ, interval = sq.gr.co_occurrence(
        sub, cluster_key=cluster_key, spatial_key=spatial_key, interval=interval, copy=True
    )
    occ = np.asarray(occ, dtype=float)
    cats = [str(c) for c in sub.obs[cluster_key].cat.categories]
    k, _, n_d = occ.shape
    ii, cc, rr = np.meshgrid(np.arange(k), np.arange(k), np.arange(n_d), indexing="ij")
    table = pd.DataFrame(
        {
            "cluster": [cats[c] for c in cc.ravel()],
            "neighbour": [cats[i] for i in ii.ravel()],
            "distance_um": np.asarray(interval, dtype=float)[1:][rr.ravel()],
            "statistic": occ.ravel(),
        }
    ).sort_values(["cluster", "neighbour", "distance_um"], kind="stable")
    params = {
        "cluster_key": cluster_key,
        "distances_um": [float(d) for d in interval],
        "missing": missing,
        "squidpy_version": getattr(sq, "__version__", None),
    }
    summary = {"n_units": int(sub.n_obs), "n_classes": len(cats), "n_distances": int(n_d)}
    return TestResult(name="squidpy_co_occurrence", table=table.reset_index(drop=True), summary=summary, params=params)


def spatial_autocorr(
    adata: Any,
    *,
    keys: str | Sequence[str],
    mode: Literal["moran", "geary"] = "moran",
    radius_um: float | None = None,
    n_neighbors: int | None = 6,
    connectivity_key: str | None = None,
    library_key: str | None = None,
    n_perms: int | None = 1000,
    two_tailed: bool = False,
    missing: Literal["drop"] = "drop",
    rng: RngLike = None,
    spatial_key: str = "spatial",
) -> TestResult:
    """Global Moran's I or Geary's C of receptor labels with squidpy (``squidpy.gr.spatial_autocorr``).

    Parameters
    ----------
    adata
        AnnData with the labels and ``obsm[spatial_key]`` in µm.
    keys
        ``obs`` columns: numeric (``"air_switched_share"``, ``"air_expanded_share"``, ...), boolean, or categorical
        (expanded into one 0/1 indicator per category, named ``"{key}={category}"``).
    mode
        ``"moran"`` or ``"geary"``.
    radius_um, n_neighbors, connectivity_key, library_key
        squidpy's graph (see :func:`nhood_enrichment`), built on the units that have every key.
    n_perms
        squidpy's permutations (``None``: analytic normal approximation only).
    two_tailed
        squidpy's two-tailed p-values.
    missing
        Units missing any key are left out.
    rng
        Seed or generator.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    TestResult
        ``table``: one row per label with ``label``, ``statistic`` (I or C), ``expected`` (−1 / (n − 1) for I, 1 for
        C), ``p_value`` (squidpy's permutation p, or its normal p without permutations), ``p_norm``, ``q_value``
        (BH over labels), ``n_perm``.

    Warns
    -----
    DepthNullWarning
        Always (see the module note).

    Null model
    ----------
    squidpy permutes the values over the units (complete spatial randomness of the values; positions and
    graph fixed), or uses the analytic normal approximation of that randomisation. Shares computed from few records
    are noisy, and every label that follows B- or T-cell density is autocorrelated under this null.

    References
    ----------
    Moran 1950, doi:10.2307/2332142; Geary 1954, doi:10.2307/2986645; Palla et al. 2022,
    doi:10.1038/s41592-021-01358-2.
    """
    sq = _sq()
    gen = as_rng(rng)
    keys = [keys] if isinstance(keys, str) else list(keys)
    if missing != "drop":
        raise ValueError("spatial_autocorr: only missing='drop' is supported")
    sub = _subset(adata, keys, "drop")
    names: list[str] = []
    for k in keys:
        s = sub.obs[k]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            for c in pd.Categorical(s.astype(str)).categories:
                name = f"{k}={c}"
                sub.obs[name] = (s.astype(str) == c).to_numpy(dtype=float)
                names.append(name)
        elif pd.api.types.is_bool_dtype(s.dtype):
            sub.obs[f"{k}_float"] = s.to_numpy(dtype=float)
            names.append(f"{k}_float")
        else:
            names.append(k)
    key = _graph(
        sub,
        connectivity_key=connectivity_key,
        radius_um=radius_um,
        n_neighbors=n_neighbors,
        library_key=library_key,
        spatial_key=spatial_key,
    )
    seed = int(gen.integers(0, 2**31 - 1))
    _depth_warning("ext.squidpy.spatial_autocorr")
    fn = sq.gr.spatial_autocorr
    df = fn(
        sub,
        connectivity_key=key,
        genes=names,
        mode=mode,
        attr="obs",
        n_perms=n_perms,
        two_tailed=two_tailed,
        copy=True,
        **_seed_kw(fn, seed),
        **_quiet_kw(fn),
    )
    stat_col = "I" if mode == "moran" else "C"
    p = df["pval_sim"].to_numpy(dtype=float) if n_perms else df["pval_norm"].to_numpy(dtype=float)
    n = sub.n_obs
    table = pd.DataFrame(
        {
            "label": df.index.astype(str),
            "statistic": df[stat_col].to_numpy(dtype=float),
            "expected": -1.0 / (n - 1) if mode == "moran" else 1.0,
            "p_value": p,
            "p_norm": df["pval_norm"].to_numpy(dtype=float),
            "q_value": bh(p),
            "n_perm": int(n_perms or 0),
        }
    )
    table = table.set_index("label").reindex(names).reset_index()
    params = {
        "keys": keys,
        "labels": names,
        "mode": mode,
        "radius_um": radius_um,
        "n_neighbors": n_neighbors if radius_um is None else None,
        "connectivity_key": connectivity_key,
        "library_key": library_key,
        "n_perms": n_perms,
        "two_tailed": two_tailed,
        "seed": seed,
        "rng_entropy": rng_entropy(gen),
        "design": "squidpy: values permuted over units, graph fixed",
        "squidpy_version": getattr(sq, "__version__", None),
    }
    summary = {"n_units": int(n), "n_labels": len(table), "n_q05": int((table.q_value < 0.05).sum())}
    return TestResult(name=f"squidpy_spatial_autocorr_{mode}", table=table, summary=summary, params=params)
