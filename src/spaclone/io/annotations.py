"""Find cell-type labels and deconvolution proportions in a GEX AnnData, and use one only when it is unambiguous.

Label candidates are categorical / text ``obs`` columns whose name is a standard cell-type name (``cell_type``,
``celltype``, ``annotation``, ``anno``, ``cell2loc_anno``, ``predicted.id``, ``majority_voting``, ``first_type``,
...). Cluster columns (``leiden``, ``louvain``, ``seurat_clusters``, ``clusters``) are listed separately: they are not
cell types. Proportion candidates are ``obsm`` tables or obs column groups of non-negative numbers per cell type
(cell2location ``*_cell_abundance_w_sf``, RCTD weights, Tangram, ``*_prop`` columns). A candidate is chosen
automatically only when it is the single standard one; cell2location's 5% quantile (``q05_cell_abundance_w_sf``) is
its documented point estimate and wins over the mean / quantile / sd tables written next to it.

References
----------
Kleshchevnikov et al. 2022, doi:10.1038/s41587-021-01139-4 (cell2location: "we recommend the 5% quantile");
Cable et al. 2022, doi:10.1038/s41587-021-00830-w (RCTD).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["Annotations", "detect_annotations"]

_LABEL = re.compile(
    r"^(cell[_. ]?types?|cell[_. ]?type[_. ]?(major|minor|fine|coarse|l\d)|annotations?|annot|anno|cell2loc_anno|"
    r"predicted[_.]?(id|cell[_.]?type|labels?)|majority_voting|celltypist[_.]?\w*|labels?|cell[_.]?labels?|"
    r"first_type|cell_ontology_class|sctype_classification|singler[_.]?labels?)$",
    re.IGNORECASE,
)
_CLUSTER = re.compile(r"^(leiden|louvain|seurat_clusters|clusters?|kmeans\w*|gene_exp_leiden)$", re.IGNORECASE)
_PROP_OBSM = re.compile(r"(cell_abundance|proportion|prop$|weights?$|deconv|tangram|rctd|spotlight|stereoscope)",
                        re.IGNORECASE)  # fmt: skip


@dataclass
class Annotations:
    """Cell-type label and proportion candidates of a GEX AnnData (:func:`detect_annotations`).

    Attributes
    ----------
    labels
        Standard cell-type label columns of ``obs``.
    clusters
        Cluster columns (not cell types).
    proportions
        ``obsm`` keys (or ``"obs:<prefix>"`` groups) holding per-type abundances or proportions.
    label
        The chosen label column (``None`` unless exactly one standard candidate exists).
    proportion
        The chosen proportion table (``None`` unless unambiguous).
    evidence
        Why.
    """

    labels: list[str] = field(default_factory=list)
    clusters: list[str] = field(default_factory=list)
    proportions: list[str] = field(default_factory=list)
    label: str | None = None
    proportion: str | None = None
    evidence: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        """JSON-ready summary for ``meta["io"]``."""
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}

    def __repr__(self) -> str:
        lab = self.label or ("ambiguous: " + ", ".join(self.labels) if self.labels else "none")
        prop = self.proportion or ("ambiguous: " + ", ".join(self.proportions) if self.proportions else "none")
        return f"Annotations(label={lab}; proportions={prop})"


def detect_annotations(adata: Any, *, max_categories: int = 200) -> Annotations:
    """List cell-type label and deconvolution candidates of a GEX AnnData and choose one only when unambiguous.

    Parameters
    ----------
    adata
        AnnData (only ``obs`` / ``obsm`` are read).
    max_categories
        A label column has at most this many distinct values.

    Returns
    -------
    Annotations
        ``label`` / ``proportion`` are set only when exactly one standard candidate exists (cell2location: the
        ``q05`` table); otherwise the candidates are listed and the user passes the one to use (e.g.
        ``spaclone.run(..., cell_type_key=...)``).
    """
    out = Annotations()
    obs = adata.obs
    for c in obs.columns:
        s = obs[c]
        name = str(c)
        if _CLUSTER.match(name):
            out.clusters.append(name)
            continue
        if not _LABEL.match(name):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            n = s.nunique(dropna=True)
            if 1 <= n <= max_categories:
                out.labels.append(name)
    if len(out.labels) == 1:
        out.label = out.labels[0]
        out.evidence.append(f"obs[{out.label!r}] is the only standard cell-type label column")
    elif out.labels:
        out.evidence.append(f"{len(out.labels)} label columns ({', '.join(out.labels)}): none chosen")
    for k in adata.obsm.keys():
        v = adata.obsm[k]
        if not _PROP_OBSM.search(str(k)):
            continue
        arr = v.to_numpy(dtype=float) if isinstance(v, pd.DataFrame) else np.asarray(v, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and np.nanmin(arr) >= 0:
            out.proportions.append(str(k))
    prefixes: dict[str, list[str]] = {}
    for c in obs.columns:
        m = re.match(r"^(.*?)(_prop|_proportion|_weight)$", str(c), re.IGNORECASE)
        if m and pd.api.types.is_numeric_dtype(obs[c]):
            prefixes.setdefault(m.group(2), []).append(str(c))
    for suf, cols in prefixes.items():
        if len(cols) >= 2:
            out.proportions.append(f"obs:*{suf}")
    c2l = [p for p in out.proportions if p.startswith("q05") and "cell_abundance" in p]
    if len(c2l) == 1:
        out.proportion = c2l[0]
        out.evidence.append(f"obsm[{c2l[0]!r}]: cell2location's documented point estimate (5% quantile)")
    elif len(out.proportions) == 1:
        out.proportion = out.proportions[0]
        out.evidence.append(f"{out.proportion!r} is the only proportion table")
    elif out.proportions:
        out.evidence.append(f"{len(out.proportions)} proportion tables ({', '.join(out.proportions)}): none chosen")
    return out
