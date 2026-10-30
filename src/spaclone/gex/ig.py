"""Immunoglobulin constant-gene RNA per unit: depth strata for the location-trust tests and Ig class composition.

A spatial BCR call is partly a readout of how much Ig RNA a unit holds (plasma cells hold far more than B cells, and
Ig RNA spills over to neighbours). The trust tests therefore compare like with like inside strata of local Ig RNA
(:data:`spaclone.nulls.IG_RNA_BINS`). :func:`ig_rna` gives that amount per unit, optionally summed within a radius,
with the gene lists of the archive (``04_external_validation_2026-10/workflow/X5_depth_tests.py``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from spaclone.gex.qc import _bool_obs, _groups, _record, gene_index, get_counts, get_xy, radius_sum
from spaclone.nulls import IG_RNA_BINS, bin_labels

__all__ = ["IG_CLASSES", "IG_GENES", "ig_class_composition", "ig_genes", "ig_rna"]

logger = logging.getLogger("spaclone")

#: Ig constant-region genes counted as "Ig RNA" (archive X5). Mouse: kappa, lambda 1–3 and every heavy constant gene;
#: human: IGKC, every IGLC gene (not IGLCOR pseudogenes) and every heavy constant gene.
IG_GENES: dict[str, list[str]] = {
    "mouse": ["Igkc", "Iglc1", "Iglc2", "Iglc3", "Ighm", "Ighd", "Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha", "Ighe"],
    "human": ["IGKC", "IGLC*", "IGHM", "IGHD", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHE"],
}

#: Heavy-chain constant genes by class and subclass, in locus order (mouse Ighg2a is the BALB/c allele of Ighg2c).
IG_CLASSES: dict[str, dict[str, list[str]]] = {
    "mouse": {
        "IgM": ["Ighm"],
        "IgD": ["Ighd"],
        "IgG3": ["Ighg3"],
        "IgG1": ["Ighg1"],
        "IgG2b": ["Ighg2b"],
        "IgG2c": ["Ighg2c", "Ighg2a"],
        "IgE": ["Ighe"],
        "IgA": ["Igha"],
    },
    "human": {
        "IgM": ["IGHM"],
        "IgD": ["IGHD"],
        "IgG3": ["IGHG3"],
        "IgG1": ["IGHG1"],
        "IgA1": ["IGHA1"],
        "IgG2": ["IGHG2"],
        "IgG4": ["IGHG4"],
        "IgE": ["IGHE"],
        "IgA2": ["IGHA2"],
    },
}
_CLASS_OF = {"IgM": "IgM", "IgD": "IgD", "IgE": "IgE"}
_SWITCHED = ("IgG", "IgA", "IgE")


def ig_genes(var_names: Sequence[str] | pd.Index, genes: str | Sequence[str] = "mouse") -> list[str]:
    """Ig constant genes present in ``var_names``.

    Parameters
    ----------
    var_names
        Gene names of the data.
    genes
        ``"mouse"``, ``"human"`` (see :data:`IG_GENES`) or a list of names; a trailing ``*`` is a prefix wildcard
        (``"IGLC*"`` = every gene starting with IGLC except the IGLCOR pseudogenes).

    Returns
    -------
    list of str
        Present genes, in the order of ``var_names`` for wildcards and of ``genes`` otherwise.
    """
    names = pd.Index(var_names).astype(str)
    spec = IG_GENES[genes] if isinstance(genes, str) and genes in IG_GENES else genes
    if isinstance(spec, str):
        raise ValueError(f"genes must be 'mouse', 'human' or a list of gene names, got {spec!r}")
    out: list[str] = []
    for g in spec:
        if g.endswith("*"):
            pre = g[:-1]
            hits = [n for n in names if n.startswith(pre) and not re.match(r"^IGLCOR", n, flags=re.IGNORECASE)]
            out.extend(h for h in hits if h not in out)
        elif g in names and g not in out:
            out.append(g)
    return out


def ig_rna(
    adata: ad.AnnData,
    *,
    genes: str | Sequence[str] = "mouse",
    radius_um: float = 0.0,
    bins: Sequence[float] = IG_RNA_BINS,
    tissue_key: str | None = None,
    library_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    key_added: str = "ig_rna",
) -> pd.DataFrame:
    """Molecules of Ig constant genes per unit (or summed within a radius), with depth-stratum labels.

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts.
    genes
        ``"mouse"``, ``"human"`` or a gene list (see :func:`ig_genes`).
    radius_um
        0: the unit's own Ig RNA. > 0: the sum over the units within this radius (the unit included; same library;
        only tissue units when ``tissue_key`` is given).
    bins
        Bin edges of the strata (default :data:`spaclone.nulls.IG_RNA_BINS`: 0, 1–2, 3–9, 10–29, 30–99, 100–299,
        ≥ 300).
    tissue_key
        Boolean ``obs`` column; with ``radius_um > 0`` only tissue units contribute to the sums.
    library_key
        ``obs`` column naming the section (radius sums stay within a section).
    layer
        Layer holding raw counts.
    spatial_key
        ``obsm`` key of the coordinates (``radius_um > 0``).
    key_added
        ``obs[key_added]`` receives the amount and ``obs[f"{key_added}_bin"]`` its stratum label.

    Returns
    -------
    pandas.DataFrame
        Units per stratum (``bin``, ``n_units``, ``ig_rna_sum``), in bin order.

    References
    ----------
    Archive X5 (depth tests): the share of units with a BCR call rises with local Ig RNA, so a call is compared with
    units of the same Ig RNA stratum.
    """
    counts = get_counts(adata, layer)
    used = ig_genes(adata.var_names, genes)
    if not used:
        raise ValueError("no Ig constant gene of the list is in the data")
    pos, _ = gene_index(adata.var_names, used)
    own = np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel()
    if radius_um > 0:
        tissue = _bool_obs(adata, tissue_key)
        xy = get_xy(adata, spatial_key)
        lib = np.zeros(adata.n_obs, dtype=np.int64)
        for k, (_, idx) in enumerate(_groups(adata, library_key).items()):
            lib[idx] = k
        amount = np.full(adata.n_obs, np.nan)
        t = np.flatnonzero(tissue)
        amount[t] = radius_sum(xy[t], own[t], radius_um, groups=lib[t])
    else:
        amount = own
    labels = bin_labels(amount, bins)
    order = list(dict.fromkeys(bin_labels(np.asarray(bins[:-1], dtype=float), bins)))
    cats = order + (["NA"] if (labels == "NA").any() else [])
    adata.obs[key_added] = amount
    adata.obs[f"{key_added}_bin"] = pd.Categorical(labels, categories=cats, ordered=True)
    _record(adata, f"ig_rna_{key_added}", {"genes": used, "radius_um": radius_um, "bins": [float(b) for b in bins]})
    t = pd.DataFrame({"bin": pd.Categorical(labels, categories=cats, ordered=True), "ig": amount})
    out = t.groupby("bin", observed=False).agg(n_units=("ig", "size"), ig_rna_sum=("ig", "sum")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def ig_class_composition(
    adata: ad.AnnData,
    *,
    species: Literal["mouse", "human"] | Mapping[str, Sequence[str]] = "mouse",
    groupby: str | None = None,
    level: Literal["class", "subclass"] = "class",
    layer: str | None = None,
    min_molecules: float = 1,
) -> pd.DataFrame:
    """Shares of IgM / IgD / IgG (subclasses) / IgA / IgE among heavy-chain constant molecules.

    Parameters
    ----------
    adata
        Spot-level AnnData with raw counts.
    species
        ``"mouse"``, ``"human"`` (see :data:`IG_CLASSES`) or a ``{subclass: [genes]}`` mapping.
    groupby
        ``obs`` column (e.g. a region label or a region instance); counts are summed per group. ``None``: per unit.
    level
        ``"class"`` merges subclasses (IgG1 + IgG2b + ... = IgG; IgA1 + IgA2 = IgA); ``"subclass"`` keeps them.
    layer
        Layer holding raw counts.
    min_molecules
        Rows with fewer heavy-chain constant molecules get NaN shares.

    Returns
    -------
    pandas.DataFrame
        One row per unit or group: molecule counts per class (``n_<class>``), ``n_heavy``, shares (``<class>``,
        summing to 1) and ``switched`` (share of IgG + IgA + IgE).
    """
    sub = IG_CLASSES[species] if isinstance(species, str) else {str(k): list(v) for k, v in species.items()}
    counts = get_counts(adata, layer)
    mat = {}
    for name, genes in sub.items():
        pos, _ = gene_index(adata.var_names, genes)
        mat[name] = np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel() if len(pos) else np.zeros(adata.n_obs)
    df = pd.DataFrame(mat, index=adata.obs_names)
    if level == "class":
        merged: dict[str, pd.Series] = {}
        for name in df.columns:
            cls = _CLASS_OF.get(name, re.sub(r"\d.*$", "", name))
            merged[cls] = merged.get(cls, 0) + df[name]
        df = pd.DataFrame(merged)
    elif level != "subclass":
        raise ValueError(f"level must be 'class' or 'subclass', got {level!r}")
    if groupby is not None:
        df = df.groupby(adata.obs[groupby].astype(str).to_numpy(), sort=True).sum()
        df.index.name = groupby
    n_heavy = df.sum(axis=1)
    out = df.add_prefix("n_")
    out["n_heavy"] = n_heavy
    shares = df.div(n_heavy.where(n_heavy >= max(min_molecules, 1e-12)), axis=0)
    switched = shares[[c for c in shares.columns if c.startswith(_SWITCHED)]].sum(axis=1, min_count=1)
    out = pd.concat([out, shares], axis=1)
    out["switched"] = switched
    return out
