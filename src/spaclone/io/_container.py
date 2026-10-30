"""The value returned by every reader."""

from __future__ import annotations

from typing import Any, NamedTuple

import anndata as ad
import pandas as pd

__all__ = ["SpatialAIR"]


class SpatialAIR(NamedTuple):
    """Receptor records, the matching spot-level GEX (optional) and reader metadata.

    Attributes
    ----------
    air
        Receptor table (one row per molecule, contig or cell call) with ``spot_id``, ``x_um``, ``y_um``, ``locus``,
        ``clone_id`` and AIRR fields; see :func:`spaclone.validate_air`.
    gex
        Spot-level AnnData whose ``obs`` index is ``spot_id`` and whose ``obsm["spatial"]`` is in micrometres, or
        ``None`` when the reader has no expression data.
    meta
        Reader metadata: technology, sample, source coordinate unit and the factor to micrometres, file paths.
    """

    air: pd.DataFrame
    gex: ad.AnnData | None
    meta: dict[str, Any]

    def __repr__(self) -> str:
        """One line: records per locus, library, GEX shape, unit and reader (``meta`` itself is not printed)."""
        air, meta = self.air, self.meta if isinstance(self.meta, dict) else {}
        n = len(air) if isinstance(air, pd.DataFrame) else 0
        parts = [f"{n:,} records"]
        if n and "locus" in air.columns:
            loci = air["locus"].astype(str).value_counts()
            more = f", +{len(loci) - 7} loci" if len(loci) > 7 else ""
            parts[0] += " (" + ", ".join(f"{k} {v:,}" for k, v in loci.head(7).items()) + more + ")"
        libs = pd.unique(air["library_id"].astype(str)) if n and "library_id" in air.columns else []
        lib = (
            meta.get("library_id")
            if not len(libs)
            else ", ".join(libs[:3]) + (f" +{len(libs) - 3}" if len(libs) > 3 else "")
        )
        if lib:
            parts.append(f"library {lib}")
        g = self.gex
        parts.append(f"GEX {g.n_obs:,} units x {g.n_vars:,} genes" if g is not None else "no GEX")
        unit = meta.get("um_per_unit")
        if unit is not None:
            parts.append(f"{unit} µm per {meta.get('source_unit') or 'coordinate unit'}")
        for k in ("reader", "technology"):
            if meta.get(k):
                parts.append(f"{k} {meta[k]}")
        return "SpatialAIR(" + "; ".join(parts) + ")"
