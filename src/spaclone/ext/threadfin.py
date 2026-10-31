"""Threadfin clone-state profiles in space: clone axes, programmes and per-unit positions on tissue, with calibrated nulls.

Threadfin (``pip install threadfin``; https://github.com/Reynold4k/Threadfin) describes every clone by the
distribution of its cells' transcriptional states: each cell's expression embedding is mapped to random Fourier
features of a Gaussian kernel, a clone is the shrunken mean of its cells' features (a kernel mean embedding), and the
clone profiles are reduced to principal components (the *clone landscape*), drawn as a clone map and grouped into
programmes. On spatial data a "cell" is a spatial unit: a **spot** (bin, bead) or a **locus** (same-clone spots
within 15 µm merged, about one cell; :func:`spaclone.pp.merge_loci`). :func:`run` builds the unit-level AnnData, runs
:func:`threadfin.run` and adds the spatial layer:

* per clone: the clone-profile axes (axis 1 holds most of the variance between clone profiles) with Threadfin's sign
  rule (the most extreme clone is positive, as in ``threadfin.tl.landscape_genes``), the clone-map position and the
  programme;
* per unit: its own position on the same axes, from Threadfin's fitted feature map
  (``threadfin.profiles.model_from_adata``) projected with the profile components (``project_features``), unshrunk;
  the mean of a clone's unit positions is its unshrunk profile position;
* per unit: coordinates, Ig RNA depth stratum and, with a :class:`SpatialContext`, the compartment (T zone /
  follicle / red pulp; the last is the ``"non-B/T area"`` when the labels show no red pulp, as in a lymph node),
  signed distances to the structures and region memberships. :func:`run` builds the context
  itself when that is certain: one standard cell-type label column of the GEX
  (:func:`spaclone.io.annotations.detect_annotations`) whose classes name T, B and erythroid cells in
  :data:`MASK_TYPES`; otherwise ``result.context`` is None and ``result.context_note`` says why.

``features`` chooses what a unit's state is: its own expression (Threadfin's embedding), its spatial context
(:func:`niche_features`: local cell-type composition and signed distances to tissue structures, without the unit's own
expression) or both. On 10-µm bins a unit's expression mixes the B cell with its neighbours, so an expression axis can
be a niche axis; the niche variant and :func:`compare` separate "state" from "place".

Readouts, each with a stated null:

* :func:`niche_by_axis` — compartment shares of clone members by clone-axis decile, against clone labels permuted
  among units within depth strata (the clone axis is re-fitted on every permutation);
* :func:`spatial_coherence` — a unit against its own clone (and against the rest of its clone), and a clone against
  other clones' units within ``radius_um``, against the same label permutation;
* :func:`spanning_clones` — clones with members at both ends of the axis: spread, nearest cross-niche distance, against
  whole-clone rigid relocation (rotation, reflection and translation of the clone's footprint);
* :func:`region_axis` — where regions (e.g. CSR ∩ SHM overlap regions of :func:`spaclone.bcr.hotspot_workflow`) sit
  on the axes, against rigid relocation of each region;
* :func:`compare` — how much of one feature space's clone profiles another explains (e.g. expression vs niche).

The permutation nulls re-fit Threadfin's clone profiles from the same unit features on every permuted labelling
(variance components, shrinkage, principal components and sign rule), so self-inclusion of a unit in its own clone,
clone size and shrinkage are part of the null. The embedding, the neighbour smoothing and the random features do not
depend on the clone labels and are kept fixed.

References
----------
Rahimi & Recht 2007, NeurIPS 20 (random Fourier features); Muandet et al. 2017, doi:10.1561/2200000060 (kernel mean
embeddings); Searle, Casella & McCulloch 1992, doi:10.1002/9780470316856 (BLUP shrinkage); Besag & Diggle 1977,
doi:10.2307/2346974 and Phipson & Smyth 2010, doi:10.2202/1544-6115.1585 (Monte-Carlo p-values); Lotwick & Silverman
1982, doi:10.1111/j.2517-6161.1982.tb01221.x (displacement of a whole pattern); Alexander-Bloch et al. 2018,
doi:10.1016/j.neuroimage.2018.05.070 (rigid "spin" nulls of maps); Markello & Misic 2021,
doi:10.1016/j.neuroimage.2021.118052 (the null drives false positives); Hurlbert 1984, doi:10.2307/1942661
(pseudoreplication).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from spaclone import _schema as S
from spaclone._utils import (
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    bh,
    mc_pvalue,
    null_summary,
    rng_entropy,
)
from spaclone.nulls import IG_RNA_BINS, Permute, bin_labels, placebo

__all__ = [
    "NicheSpec",
    "SpatialContext",
    "ThreadfinSpatialResult",
    "compare",
    "niche_by_axis",
    "niche_features",
    "region_axis",
    "run",
    "spanning_clones",
    "spatial_coherence",
    "spatial_context",
]

logger = logging.getLogger("spaclone")

#: Oldest Threadfin release with the fitted-feature-map API used here (``model_from_adata``, ``project_features``).
MIN_THREADFIN = (4, 5)
_EXTRA_HINT = "pip install 'spaclone[threadfin]'  (or: pip install 'threadfin>=4.5')"

#: Default cell classes behind the three tissue masks of :func:`spatial_context` (built-in marker labels of
#: :data:`spaclone.gex.MARKERS` and common deconvolution names).
MASK_TYPES: dict[str, tuple[str, ...]] = {
    "T": ("T", "Tfh", "T lineage", "T_cells", "T cells", "T cell"),
    "B": ("B", "GC_B", "ASC", "B lineage", "B_cells", "B cells", "B cell", "plasma"),
    "erythroid": ("erythroid", "erythrocyte", "Erythrocyte", "Erythrocytes", "Ery"),
}
#: Compartment names of :func:`spatial_context` (T-rich, outside the B-rich area and the T zones, B-rich) under each
#: naming: the spleen's, and tissue-neutral ones for a section without a red pulp (e.g. a lymph node, whose area outside
#: its B and T zones is mostly medulla and sinus). Only the middle name differs.
COMPARTMENT_NAMES: dict[str, tuple[str, str, str]] = {
    "spleen": ("T zone", "red pulp", "follicle"),
    "neutral": ("T zone", "non-B/T area", "follicle"),
}
#: Share of the typed spots outside the B-rich area and the T zones that must lie in the erythroid-rich mask for
#: ``spatial_context(names="auto")`` to name that compartment red pulp.
RED_PULP_MIN_ERYTHROID = 0.5
#: Compartments counted as white pulp (B-rich follicles and T zones) by :func:`spanning_clones`, under either naming.
WHITE_PULP = ("T zone", "follicle")
#: Compartments counted as red pulp (outside the B-rich area and the T zones) by :func:`spanning_clones`: its name under
#: either naming of :func:`spatial_context`.
RED_PULP = (COMPARTMENT_NAMES["spleen"][1], COMPARTMENT_NAMES["neutral"][1])


class ThreadfinWarning(SpacloneWarning):
    """The Threadfin bridge noticed something the user should know (e.g. a reproduction check that did not pass)."""


# ======================================================================================================== threadfin


def _import_threadfin() -> Any:
    """Import Threadfin lazily, with a clear message naming the extra."""
    try:
        import threadfin
        import threadfin.profiles
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"spaclone.ext.threadfin needs the Threadfin package: {_EXTRA_HINT}") from e
    ver = str(getattr(threadfin, "__version__", "0"))
    try:
        parts = tuple(int(p) for p in ver.split(".")[:2])
    except ValueError:  # pragma: no cover - development versions
        parts = MIN_THREADFIN
    if parts < MIN_THREADFIN:  # pragma: no cover - old installations only
        raise ImportError(f"spaclone.ext.threadfin needs threadfin >= 4.5, found {ver}: {_EXTRA_HINT}")
    return threadfin


def _seed(rng: RngLike) -> int:
    """Threadfin's ``random_state``: an integer ``rng`` unchanged, otherwise one draw from the generator."""
    if isinstance(rng, int | np.integer) and not isinstance(rng, bool):
        return int(rng)
    return int(as_rng(rng).integers(0, 2**31 - 1))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); NaN with fewer than 3 finite pairs or a constant input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# =================================================================================================== spatial context


def _xy_of(adata: ad.AnnData, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] (coordinates in µm) is missing")
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError(f"obsm[{spatial_key!r}] contains NaN or infinite coordinates")
    return xy


def _mask_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name).lower()).strip("_")


@dataclass(frozen=True, repr=False)
class SpatialContext:
    """Tissue structures used as places: compartments, masks to measure signed distances to, and regions.

    Attributes
    ----------
    compartments
        Code raster (:func:`spaclone.tl.compartments`) whose categories name the compartments, or None.
    masks
        Boolean rasters (``{name: Raster}``) whose signed distance (µm, negative inside) describes a unit's place,
        e.g. ``"T zone"``, ``"B-rich"``, ``"red pulp"`` (``"non-B/T area"`` without a red pulp).
    regions
        Integer label rasters of region instances (0 = none), e.g. ``{"CSR/SHM overlap": labels}``.
    params
        How the context was built.
    """

    compartments: Any = None
    masks: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        comp = list(self.compartments.categories) if self.compartments is not None else []
        return f"SpatialContext(compartments={comp}, masks={list(self.masks)}, regions={list(self.regions)})"

    def sample(self, x_um: Any, y_um: Any, *, fill_compartment: str = "outside") -> pd.DataFrame:
        """Compartment, signed distances and region labels at points.

        Parameters
        ----------
        x_um, y_um
            Point coordinates (µm).
        fill_compartment
            Compartment of points off the grid.

        Returns
        -------
        pandas.DataFrame
            ``compartment``; ``dist_<mask>_um`` for every mask (signed, negative inside; NaN off the grid);
            ``depth_b_rich_um`` (= minus the B-rich distance, positive inside, the archive's ``depthB``) when a
            ``"B-rich"`` mask exists; per region ``<region>_id`` (0 = outside), ``in_<region>`` and
            ``dist_<region>_um``.
        """
        from spaclone.tl.structures import Raster, signed_distance

        x = np.asarray(x_um, dtype=float)
        y = np.asarray(y_um, dtype=float)
        out = pd.DataFrame(index=np.arange(len(x)))
        if self.compartments is not None:
            out["compartment"] = self.compartments.sample_categories(x, y, fill=fill_compartment)
        for name, m in self.masks.items():
            d = signed_distance(m).sample(x, y, fill=np.nan)
            out[f"dist_{_mask_name(name)}_um"] = d
            if name == "B-rich":
                out["depth_b_rich_um"] = -d
        for name, lab in self.regions.items():
            key = _mask_name(name)
            ids = lab.sample(x, y, fill=0).astype(np.int64)
            out[f"{key}_id"] = ids
            out[f"in_{key}"] = ids > 0
            if (lab.values > 0).any():
                out[f"dist_{key}_um"] = signed_distance(Raster(lab.values > 0, lab.grid)).sample(x, y, fill=np.nan)
            else:
                out[f"dist_{key}_um"] = np.inf
        return out


def _as_label_raster(region: Any, gex: ad.AnnData | None, grid: Any, spatial_key: str) -> Any:
    """A region given as a Raster, a workflow result (``.region_labels``) or per-spot labels -> integer Raster."""
    from spaclone.tl.structures import Raster, zones

    if hasattr(region, "region_labels") and getattr(region, "region_labels", None) is not None:
        region = region.region_labels
    if isinstance(region, Raster):
        v = np.asarray(region.values)
        if v.dtype == bool:
            return zones(region)
        return Raster(v.astype(np.int64), region.grid, name=region.name)
    if gex is None or grid is None:
        raise ValueError("per-spot region labels need the GEX AnnData and a grid")
    if isinstance(region, str):
        if region not in gex.obs:
            raise KeyError(f"region column {region!r} not in gex.obs")
        vals = gex.obs[region]
    else:
        vals = pd.Series(np.asarray(region), index=gex.obs_names) if not isinstance(region, pd.Series) else region
        vals = vals.reindex(gex.obs_names)
    if vals.dtype == bool:
        lab_spot = vals.fillna(False).to_numpy().astype(np.int64)
        bool_input = True
    else:
        codes, _ = pd.factorize(vals.where(~vals.isin([0, "0", "", "none", "None"])), use_na_sentinel=True)
        lab_spot = codes.astype(np.int64) + 1
        bool_input = False
    xy = _xy_of(gex, spatial_key)
    i, j, inside = grid.index(xy[:, 0], xy[:, 1])
    arr = np.zeros(grid.shape, dtype=np.int64)
    sel = inside & (lab_spot > 0)
    np.maximum.at(arr, (i[sel], j[sel]), lab_spot[sel])
    r = Raster(arr, grid, name="region")
    return zones(Raster(arr > 0, grid)) if bool_input else r


def spatial_context(
    gex: ad.AnnData,
    *,
    cell_type_key: str,
    types: Mapping[str, Sequence[str]] | None = None,
    regions: Mapping[str, Any] | None = None,
    pixel_um: float = 10.0,
    smooth_um: float = 30.0,
    min_density: float = 0.15,
    min_area_um2: float = 8000.0,
    pad_edges: bool = True,
    spatial_key: str = "spatial",
    names: Literal["auto", "spleen", "neutral"] = "auto",
) -> SpatialContext:
    """Compartments, masks and regions of a section from cell-typed spots (the archive's StereoIR T-series rule).

    Typed spots (class in one of ``types``) are counted on a ``pixel_um`` grid; the T-rich, B-rich and
    erythroid-rich masks are :func:`spaclone.tl.mask_from_types` (30-µm Gaussian, Otsu threshold, closing / opening,
    small islands and holes removed); compartments are T zone > red pulp (erythroid-rich, or outside the B-rich area)
    > follicle (:func:`spaclone.tl.compartments`). The middle compartment is called red pulp only where the labels show
    one (see ``names``); otherwise it is the ``"non-B/T area"``, with the same pixels.

    Parameters
    ----------
    gex
        Spot-level AnnData with ``obsm[spatial_key]`` in µm and a class column.
    cell_type_key
        ``obs`` column of spot classes (a deconvolution top class or :func:`spaclone.gex.score_cell_types`).
    types
        ``{"T": [...], "B": [...], "erythroid": [...]}``: classes behind each mask (default :data:`MASK_TYPES`).
        Spots whose class is in none of them are left out of the masks (the archive used typed bins only).
    regions
        ``{name: region}`` where a region is a :class:`spaclone.tl.Raster` (boolean mask or instance labels), a
        :class:`spaclone.bcr.HotspotWorkflowResult` (its ``region_labels``), an ``obs`` column name or per-spot
        labels (boolean or instance ids; 0 / empty = none), rasterised on the context grid.
    pixel_um, smooth_um, min_density, min_area_um2
        Mask recipe (:func:`spaclone.tl.mask_from_types`).
    pad_edges
        Keep mask pixels at the grid edge during the closing; ``False`` reproduces the 2026-10 archive.
    spatial_key
        ``obsm`` key of the coordinates.
    names
        Names of the compartments (:data:`COMPARTMENT_NAMES`), derived from the labels by default (``"auto"``): the
        spleen's (``T zone`` / ``red pulp`` / ``follicle``) when most typed spots outside the B-rich area and the T
        zones lie in the erythroid-rich mask (share at least :data:`RED_PULP_MIN_ERYTHROID`; 0.95 on the StereoIR
        spleen), else tissue-neutral (``T zone`` / ``non-B/T area`` / ``follicle``; 0.00 to 0.18 on the AIR-SPACE
        lymph nodes, whose area outside the B and T zones is mostly medulla and sinus). ``"spleen"`` or ``"neutral"``
        forces either set. Names never change a pixel.

    Returns
    -------
    SpatialContext
        Compartments ``T zone`` / ``red pulp`` (or ``non-B/T area``) / ``follicle``; masks ``"T zone"``,
        ``"B-rich"``, the middle compartment under its name, and ``"erythroid-rich"``; the regions. ``params`` records
        the naming (``names``, ``naming``) and the erythroid share it was decided on (``outside_erythroid_share``).

    References
    ----------
    Otsu 1979, doi:10.1109/TSMC.1979.4310076 (threshold); Maurer et al. 2003, doi:10.1109/TPAMI.2003.1177156
    (distance transform).
    """
    from spaclone.tl.structures import Grid, Raster, compartments, mask_from_types

    if cell_type_key not in gex.obs:
        raise KeyError(f"cell_type_key {cell_type_key!r} not in gex.obs")
    if names not in ("auto", *COMPARTMENT_NAMES):
        raise ValueError(f"names must be 'auto', 'spleen' or 'neutral', got {names!r}")
    groups = {k: tuple(v) for k, v in (types or MASK_TYPES).items()}
    missing = [k for k in ("T", "B", "erythroid") if k not in groups]
    if missing:
        raise ValueError(f"types needs the groups 'T', 'B' and 'erythroid' (missing {missing})")
    group_of = {str(c): g for g, classes in groups.items() for c in classes}
    lab = gex.obs[cell_type_key].astype("object").map(lambda v: group_of.get(str(v)) if pd.notna(v) else None)
    typed = lab.notna().to_numpy()
    if not typed.any():
        raise ValueError(f"no spot of {cell_type_key!r} belongs to the mask classes {groups}")
    xy = _xy_of(gex, spatial_key)
    units = pd.DataFrame({S.X: xy[typed, 0], S.Y: xy[typed, 1], "cell_type": lab.to_numpy()[typed]})
    grid = Grid.from_points(units[S.X], units[S.Y], pixel_um=pixel_um)
    kw = {"smooth_um": smooth_um, "min_density": min_density, "min_area_um2": min_area_um2, "pad_edges": pad_edges}
    m_t = mask_from_types(units, types="T", type_key="cell_type", grid=grid, **kw)
    m_b = mask_from_types(units, types="B", type_key="cell_type", grid=grid, **kw)
    m_e = mask_from_types(units, types="erythroid", type_key="cell_type", grid=grid, **kw)
    # label-derived names: the area outside the B-rich area and the T zones is the red pulp when it is erythroid-rich
    ux, uy = units[S.X].to_numpy(dtype=float), units[S.Y].to_numpy(dtype=float)
    in_t, in_b, in_e = (np.asarray(m.sample(ux, uy, fill=0), dtype=bool) for m in (m_t, m_b, m_e))
    outside = ~in_t & (in_e | ~in_b)  # the typed spots of the middle compartment (same rule as the raster below)
    ery_share = float(in_e[outside].mean()) if outside.any() else float("nan")
    naming = names if names != "auto" else ("spleen" if ery_share >= RED_PULP_MIN_ERYTHROID else "neutral")
    t_name, out_name, b_name = COMPARTMENT_NAMES[naming]
    comp = compartments([(t_name, m_t), (out_name, m_e), (out_name, ~m_b)], default=b_name)
    red = Raster(comp.values == comp.categories.index(out_name), grid, name=out_name)
    regs = {str(k): _as_label_raster(v, gex, grid, spatial_key) for k, v in (regions or {}).items()}
    params = {
        "cell_type_key": cell_type_key,
        "types": {k: list(v) for k, v in groups.items()},
        "n_typed_spots": int(typed.sum()),
        "grid": {"origin": grid.origin, "pixel_um": grid.pixel_um, "shape": grid.shape},
        **kw,
        "names": names,
        "naming": naming,
        "outside_erythroid_share": ery_share,
    }
    return SpatialContext(
        compartments=comp,
        masks={t_name: m_t, "B-rich": m_b, out_name: red, "erythroid-rich": m_e},
        regions=regs,
        params=params,
    )


# ====================================================================================================== niche states


@dataclass(frozen=True)
class NicheSpec:
    """How :func:`niche_features` describes a unit's place.

    Attributes
    ----------
    composition_keys
        ``obs`` columns of per-spot abundances (e.g. cell2location cell abundances) summed over the neighbourhood and
        turned into proportions.
    cell_type_key
        Alternative to ``composition_keys``: a categorical ``obs`` column (one-hot per spot); classes in
        ``ignore_types`` are left out.
    radius_um
        Neighbourhood radius (µm) of the composition.
    exclude_self
        Leave the unit's own spot(s) out of its neighbourhood, so a unit's place does not contain its own expression.
    distances
        Add the signed distances (µm) to the masks and regions of the :class:`SpatialContext`.
    clip_um
        Distances are clipped to ``[-clip_um, clip_um]`` (far away is far away).
    ignore_types
        Classes of ``cell_type_key`` that are not counted.
    """

    composition_keys: Sequence[str] | None = None
    cell_type_key: str | None = None
    radius_um: float = 50.0
    exclude_self: bool = True
    distances: bool = True
    clip_um: float = 300.0
    ignore_types: Sequence[str] = ("unassigned",)


def _composition_values(gex: ad.AnnData, spec: NicheSpec) -> tuple[np.ndarray, list[str]]:
    if spec.composition_keys:
        keys = list(spec.composition_keys)
        miss = [k for k in keys if k not in gex.obs]
        if miss:
            raise KeyError(f"composition_keys {miss} not in gex.obs")
        v = gex.obs[keys].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (v < 0).any():
            raise ValueError("composition values must be non-negative (abundances or proportions)")
        return v, keys
    if spec.cell_type_key:
        if spec.cell_type_key not in gex.obs:
            raise KeyError(f"cell_type_key {spec.cell_type_key!r} not in gex.obs")
        s = gex.obs[spec.cell_type_key].astype("object")
        cats = sorted({str(c) for c in s.dropna()} - {str(c) for c in spec.ignore_types})
        if not cats:
            raise ValueError(f"no class of {spec.cell_type_key!r} to count")
        idx = pd.Index(cats).get_indexer(s.astype(str))
        v = np.zeros((gex.n_obs, len(cats)))
        ok = idx >= 0
        v[np.flatnonzero(ok), idx[ok]] = 1.0
        return v, cats
    return np.zeros((gex.n_obs, 0)), []


def niche_features(
    gex: ad.AnnData,
    unit_xy: np.ndarray,
    *,
    spec: NicheSpec | None = None,
    context: SpatialContext | None = None,
    members: sparse.spmatrix | None = None,
    spatial_key: str = "spatial",
) -> pd.DataFrame:
    """Describe each unit by its place: neighbourhood cell-type composition and signed distances to structures.

    Parameters
    ----------
    gex
        Spot-level AnnData (all spots of the section, so the neighbourhood is complete) with ``obsm[spatial_key]``.
    unit_xy
        ``(n_units, 2)`` unit positions in µm (spot centres or locus centroids).
    spec
        :class:`NicheSpec` (default: distances only, which needs ``context``).
    context
        :class:`SpatialContext` for the signed distances.
    members
        Sparse ``(n_units, n_spots)`` 0/1 membership of units in spots; with ``spec.exclude_self`` these spots are left
        out of the unit's neighbourhood. ``None``: a unit's own spot is the spot at its position, if any.
    spatial_key
        ``obsm`` key of the coordinates.

    Returns
    -------
    pandas.DataFrame
        One row per unit: ``comp_<class>`` (proportions of the neighbourhood's summed abundances or typed spots; NaN
        when the neighbourhood is empty) and ``dist_<structure>_um`` (signed, negative inside, clipped).
    """
    spec = spec or NicheSpec()
    xy_s = _xy_of(gex, spatial_key)
    unit_xy = np.asarray(unit_xy, dtype=float)
    out = pd.DataFrame(index=np.arange(len(unit_xy)))
    vals, names = _composition_values(gex, spec)
    if names:
        tree_s = cKDTree(xy_s)
        tree_u = cKDTree(unit_xy)
        nb = tree_u.sparse_distance_matrix(tree_s, spec.radius_um + 1e-9, output_type="coo_matrix").tocsr()
        nb.data[:] = 1.0
        # distance 0 pairs are dropped by sparse_distance_matrix: add them back (a unit on a spot)
        d0, j0 = tree_s.query(unit_xy, distance_upper_bound=1e-6)
        on = np.isfinite(d0)
        nb = nb + sparse.csr_matrix((np.ones(on.sum()), (np.flatnonzero(on), j0[on])), shape=nb.shape)
        nb.data[:] = 1.0
        if spec.exclude_self:
            own = (
                members
                if members is not None
                else sparse.csr_matrix((np.ones(on.sum()), (np.flatnonzero(on), j0[on])), shape=nb.shape)
            )
            own = sparse.csr_matrix(own, dtype=float)
            if own.shape != nb.shape:
                raise ValueError(f"members has shape {own.shape}, expected {nb.shape}")
            nb = nb - nb.multiply(own > 0)
            nb.eliminate_zeros()
        sums = np.asarray(nb @ vals)
        tot = sums.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            prop = np.where(tot > 0, sums / tot, np.nan)
        for k, nm in enumerate(names):
            out[f"comp_{_mask_name(nm)}"] = prop[:, k]
    if spec.distances and context is not None:
        smp = context.sample(unit_xy[:, 0], unit_xy[:, 1])
        for c in smp.columns:
            if c.startswith("dist_"):
                out[c] = np.clip(smp[c].to_numpy(dtype=float), -spec.clip_um, spec.clip_um)
    if out.shape[1] == 0:
        raise ValueError("niche_features: nothing to describe; give composition_keys / cell_type_key or a context")
    return out


def _niche_basis(niche: pd.DataFrame, clip_um: float) -> np.ndarray:
    """Standardised niche features, each block (composition, distances) weighted to the same total variance."""
    blocks = []
    for prefix in ("comp_", "dist_"):
        cols = [c for c in niche.columns if c.startswith(prefix)]
        if not cols:
            continue
        v = niche[cols].to_numpy(dtype=float)
        fill = np.nanmean(v, axis=0) if prefix == "comp_" else np.full(v.shape[1], clip_um)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        v = np.where(np.isfinite(v), v, fill[None, :])
        sd = v.std(axis=0)
        keep = sd > 0
        if not keep.any():
            continue
        z = (v[:, keep] - v[:, keep].mean(axis=0)) / sd[keep]
        blocks.append(z / np.sqrt(keep.sum()))
    if not blocks:
        raise ValueError("niche features have no variation")
    basis = np.hstack(blocks) / np.sqrt(len(blocks))
    return np.ascontiguousarray(basis, dtype=np.float32)


# ===================================================================================================== profile engine


@dataclass
class _Fit:
    """Clone profiles and axes for one clone labelling (see :class:`_ProfileEngine`)."""

    clone_rows: np.ndarray  # codes (into the engine's clone index) of the profiled clones
    sizes: np.ndarray  # units per code
    values: np.ndarray  # (n_profiled, n_axes) clone axis values
    positions: np.ndarray  # (n_units, n_axes) unit positions
    share: np.ndarray  # variance share of each axis among the stored components
    share_all: np.ndarray  # ... among all components
    reliability: np.ndarray  # per profiled clone
    components: np.ndarray  # (n_axes, n_features) axis directions in feature space


class _ProfileEngine:
    """Re-fit Threadfin's clone profiles from fixed unit features for any clone labelling.

    ``residuals`` are Threadfin's context-centred, smoothed kernel features of every unit (``model.residuals``). For a
    labelling, the variance components, the shrunken clone means (BLUPs) of the clones with at least ``min_cells``
    units, their principal components and Threadfin's sign rule are recomputed exactly as
    ``threadfin.tl.clone_profiles`` does; unit positions are the unshrunk projections
    ``(residual - grand mean - mean profile) @ components``. With ``reference`` (the components of the observed
    labelling), each axis is oriented to point the same way as the reference axis (positive dot product), so that
    null labellings do not flip an axis by the arbitrary sign rule. Null re-fits (``exact=False``, the default with a
    reference) take the components from the eigenvectors of the Gram matrix of the profiles instead of their SVD
    (the same components, faster).
    """

    def __init__(self, residuals: np.ndarray, *, min_cells: int, n_components: int, n_axes: int) -> None:
        tf = _import_threadfin()
        self._vc = tf.profiles.variance_components
        self.R = np.ascontiguousarray(residuals, dtype=float)
        self.min_cells = int(min_cells)
        self.n_components = int(n_components)
        self.n_axes = int(n_axes)

    def fit(self, codes: np.ndarray, reference: np.ndarray | None = None, *, exact: bool | None = None) -> _Fit:
        codes = np.asarray(codes, dtype=np.int64)
        n_codes = int(codes.max()) + 1 if (codes >= 0).any() else 0
        exact = reference is None if exact is None else exact
        vc = self._vc(self.R, codes)
        ok = np.flatnonzero(codes >= 0)
        sizes = np.bincount(codes[ok], minlength=n_codes).astype(float)
        keep = np.flatnonzero(sizes >= self.min_cells)
        # shrunken means (BLUPs) of the profiled clones only
        remap = np.full(n_codes, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        g = remap[codes[ok]]
        sel = g >= 0
        member = sparse.csr_matrix((np.ones(int(sel.sum())), (g[sel], ok[sel])), shape=(len(keep), codes.size))
        means = np.asarray(member @ self.R) / sizes[keep][:, None] - vc.grand_mean[None, :]
        b = vc.shrinkage(sizes[keep]) * means
        raw_mean = b.mean(axis=0)
        bc = b - raw_mean
        if exact:  # the SVD Threadfin uses (reproduces the stored profiles to rounding error)
            _, s, vt = np.linalg.svd(bc, full_matrices=False)
            ev = s**2
        else:  # permutations: eigenvectors of the small Gram matrix, the same components ~5x faster
            w, v = np.linalg.eigh(bc.T @ bc)
            ev, vt = np.clip(w[::-1], 0.0, None), v[:, ::-1].T
        k = int(min(self.n_components, int((ev > 1e-24).sum())))
        na = min(self.n_axes, k)
        comps = np.array(vt[:na])
        values = bc @ comps.T
        for j in range(na):  # Threadfin's sign rule (landscape_genes): the most extreme clone is positive
            flip = values[np.argmax(np.abs(values[:, j])), j] < 0
            if reference is not None and j < len(reference):  # null labellings: orient like the observed axis
                flip = float(comps[j] @ reference[j]) < 0
            if flip:
                values[:, j] *= -1
                comps[j] *= -1
        positions = self.R @ comps.T - (vc.grand_mean + raw_mean) @ comps.T
        return _Fit(
            clone_rows=keep,
            sizes=sizes,
            values=values,
            positions=positions,
            share=ev[:na] / ev[:k].sum(),
            share_all=ev[:na] / ev.sum(),
            reliability=vc.reliability(sizes[keep]),
            components=comps.copy(),
        )


# ============================================================================================================ result


@dataclass(frozen=True, repr=False)
class ThreadfinSpatialResult:
    """Threadfin's result on spatial units plus the spatial layer.

    Attributes
    ----------
    threadfin
        The :class:`threadfin.ThreadfinResult` (``.adata`` is the unit-level AnnData Threadfin ran on; ``.clones``,
        ``.programmes``, ``.coherence``, ``.summary()``, ``.methods()``).
    units
        One row per unit (index ``unit_id``): ``x_um``, ``y_um``, ``clone_id`` (NaN = no clone), ``clonal``,
        ``profiled``, ``n_spots``, ``total_counts``, ``ig_rna``, ``ig_bin`` (Ig RNA stratum,
        :data:`spaclone.nulls.IG_RNA_BINS`), ``count_bin`` (total-count quintile), ``axis_1`` ... (the unit's own
        position), ``clone_axis_1`` ... (its clone's value), ``programme`` and, with a context, ``compartment``,
        ``dist_*_um``, ``depth_b_rich_um`` and region columns.
    clones
        One row per profiled clone (index ``clone_id``): ``n_units``, ``reliability``, ``map_x``, ``map_y``,
        ``programme``, ``confidence``, ``axis_1`` ..., ``unit_mean_axis_1`` ..., ``n_spots``.
    axes
        One row per axis: ``variance`` (of the clone values), ``share`` (of the variance of the stored profile
        components), ``share_all`` (of all components).
    params
        Every argument, the Threadfin version and settings, and the reproduction check of the axes.
    niche
        Niche features per unit (unscaled; ``features`` ``"niche"`` / ``"joint"``), else None.
    context
        The :class:`SpatialContext` used, or None (then ``units`` has no ``compartment`` / ``dist_*_um`` columns).
    context_note
        Where the context came from (``"given"``, ``"auto: ..."``: the label column and its classes behind each mask)
        or why there is none (``"none: ..."``).
    membership
        ``units="locus"``: unit id of every clonal spot (index = spot id).
    landscape
        ``threadfin.tl.landscape_genes`` of the run (all genes with FDR < 0.05), when requested.
    """

    threadfin: Any
    units: pd.DataFrame
    clones: pd.DataFrame
    axes: pd.DataFrame
    params: dict[str, Any]
    niche: pd.DataFrame | None = None
    context: SpatialContext | None = None
    membership: pd.Series | None = None
    landscape: pd.DataFrame | None = None
    engine: Any = field(default=None, repr=False, compare=False)
    context_note: str = ""

    @property
    def adata(self) -> ad.AnnData:
        """The unit-level AnnData Threadfin ran on."""
        return self.threadfin.adata

    def __repr__(self) -> str:
        u = self.units
        share = ", ".join(f"axis {int(r.axis)} {100 * r.share:.1f}%" for r in self.axes.itertuples())
        return (
            f"ThreadfinSpatialResult({self.params.get('units')}/{self.params.get('features')}: {len(u):,} units, "
            f"{int(u.clonal.sum()):,} with a clone, {u.clone_id.nunique():,} clones, {len(self.clones):,} profiled; "
            f"{share})"
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers: units, clones, axis variance shares, Threadfin's coherence and programmes."""
        u = self.units
        coh = getattr(self.threadfin, "coherence", {}) or {}
        prog = getattr(self.threadfin, "programmes", pd.DataFrame())
        out = {
            "units": self.params.get("units"),
            "features": self.params.get("features"),
            "n_units": int(len(u)),
            "n_clonal_units": int(u.clonal.sum()),
            "n_clones": int(u.clone_id.nunique()),
            "n_profiled_clones": int(len(self.clones)),
            "n_profiled_units": int(u.profiled.sum()),
            "icc": float(coh.get("icc", np.nan)),
            "icc_p": float(coh.get("p_value", np.nan)),
            "n_programmes": int(len(prog)),
        }
        for r in self.axes.itertuples():
            out[f"axis_{int(r.axis)}_share"] = float(r.share)
        return out


# =============================================================================================================== run


def _unpack(data: Any) -> tuple[pd.DataFrame | None, ad.AnnData]:
    if isinstance(data, ad.AnnData):
        return None, data
    if isinstance(data, tuple) and len(data) >= 2:  # SpatialAIR is a NamedTuple
        air, gex = data[0], data[1]
        if air is not None and not isinstance(air, pd.DataFrame):
            raise TypeError(f"the receptor table must be a DataFrame, got {type(air).__name__}")
        if not isinstance(gex, ad.AnnData):
            raise TypeError("run needs the spot-level GEX AnnData (with counts) next to the receptor table")
        return air, gex
    raise TypeError(f"data must be a SpatialAIR, an (air, gex) tuple or an AnnData, got {type(data).__name__}")


def _dominant_clones(
    air: pd.DataFrame,
    *,
    chains: Sequence[str] | None,
    clone_key: str,
    spot_key: str,
    locus_key: str,
    reads_key: str,
) -> pd.DataFrame:
    """One clone per spot: the clone with most reads among the spot's records (ties: smallest clone id)."""
    for c in (spot_key, clone_key):
        if c not in air.columns:
            raise KeyError(f"receptor table lacks column {c!r}")
    a = air
    if chains is not None and locus_key in a.columns:
        a = a[a[locus_key].isin(list(chains))]
    a = a[a[clone_key].notna()]
    if a.empty:
        raise ValueError("no receptor record with a clone id on the requested chains")
    reads = pd.to_numeric(a[reads_key], errors="coerce").fillna(1.0) if reads_key in a.columns else 1.0
    t = pd.DataFrame({"spot": a[spot_key].astype(str).to_numpy(), "clone": a[clone_key].astype(str).to_numpy()})
    t["reads"] = np.asarray(reads, dtype=float) if not np.isscalar(reads) else 1.0
    g = t.groupby(["spot", "clone"], sort=False).reads.sum().reset_index()
    g = g.sort_values(["spot", "reads", "clone"], ascending=[True, False, True], kind="mergesort")
    n_cl = g.groupby("spot", sort=False).size()
    dom = g.drop_duplicates("spot").set_index("spot")
    dom["n_clones"] = n_cl.reindex(dom.index).to_numpy()
    return dom


def _spot_mask(sel: Any, gex: ad.AnnData) -> np.ndarray:
    """Boolean mask over gex spots from None, an obs column, a boolean array or spot names."""
    n = gex.n_obs
    if sel is None:
        return np.zeros(n, dtype=bool)
    if isinstance(sel, str):
        if sel not in gex.obs:
            raise KeyError(f"extra_spots column {sel!r} not in gex.obs")
        v = gex.obs[sel]
        if v.dtype != bool:
            raise ValueError(f"extra_spots column {sel!r} must be boolean")
        return v.to_numpy()
    arr = np.asarray(sel)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"extra_spots mask has length {arr.size}, gex has {n} spots")
        return arr
    return gex.obs_names.isin([str(s) for s in arr])


def _ig_counts(counts: sparse.csr_matrix, var_names: pd.Index, species: str | None) -> np.ndarray:
    from spaclone.gex.ig import ig_genes

    genes: list[str] = []
    for sp in [species] if species else ["mouse", "human"]:
        genes = ig_genes(var_names, sp)
        if genes:
            break
    if not genes:
        return np.full(counts.shape[0], np.nan)
    pos = pd.Index(var_names).get_indexer(genes)
    return np.asarray(counts[:, pos].sum(axis=1), dtype=float).ravel()


def _auto_context(gex: ad.AnnData, *, spatial_key: str) -> tuple[SpatialContext | None, str]:
    """The spatial context when it is certain, and how it was decided.

    Certain means: exactly one standard cell-type label column (:func:`spaclone.io.annotations.detect_annotations`)
    whose classes include T, B and erythroid classes of :data:`MASK_TYPES`, and three non-empty masks from it.
    """
    from spaclone.io.annotations import detect_annotations

    hint = "pass context=spatial_context(gex, cell_type_key=...) (types= maps other class names)"
    ann = detect_annotations(gex)
    if ann.label is None:
        why = "; ".join(ann.evidence) if ann.evidence else "no standard cell-type label column in gex.obs"
        return None, f"none: {why}; {hint}"
    key = ann.label
    present = set(gex.obs[key].dropna().astype(str).unique())
    found = {g: sorted(present.intersection(str(c) for c in MASK_TYPES[g])) for g in ("T", "B", "erythroid")}
    missing = [g for g, v in found.items() if not v]
    if missing:
        return None, f"none: obs[{key!r}] has no {' / '.join(missing)} class named in MASK_TYPES; {hint}"
    try:
        ctx = spatial_context(gex, cell_type_key=key, spatial_key=spatial_key)
    except (KeyError, ValueError) as e:
        return None, f"none: spatial_context(cell_type_key={key!r}) failed ({e}); {hint}"
    empty = [m for m in ("T zone", "B-rich", "erythroid-rich") if not np.asarray(ctx.masks[m].values, dtype=bool).any()]
    if empty:
        return None, f"none: obs[{key!r}] gives an empty {' / '.join(empty)} mask; {hint}"
    classes = "; ".join(f"{g}: {', '.join(v)}" for g, v in found.items())
    return ctx, f"auto: spatial_context(gex, cell_type_key={key!r}), the only standard label column ({classes})"


def run(
    data: Any,
    *,
    units: Literal["spot", "locus"] = "spot",
    features: Literal["expression", "niche", "joint"] = "expression",
    clone_key: str = S.CLONE,
    donor_key: str | None = None,
    sample_key: str | None = None,
    state_key: str | None = None,
    chains: Sequence[str] | None = S.HEAVY_LOCI,
    extra_spots: Any = None,
    loci_radius_um: float = 15.0,
    context: SpatialContext | Literal["auto"] | None = "auto",
    niche: NicheSpec | Mapping[str, Any] | None = None,
    n_axes: int = 3,
    landscape: bool = False,
    species: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    spot_key: str = S.SPOT,
    locus_key: str = S.LOCUS,
    reads_key: str = S.READS,
    library_key: str = S.LIBRARY,
    threadfin_kwargs: Mapping[str, Any] | None = None,
    rng: RngLike = 0,
) -> ThreadfinSpatialResult:
    """Run Threadfin on spatial units and map its clone landscape back to the tissue.

    Parameters
    ----------
    data
        A :class:`spaclone.io.SpatialAIR` (receptor records + spot-level GEX), an ``(air, gex)`` tuple, or a GEX
        AnnData whose ``obs[clone_key]`` already holds one clone per spot (NaN = none). GEX must hold raw counts
        (``layer``, else ``layers["counts"]``, else ``X``) and ``obsm[spatial_key]`` in µm; its ``obs_names`` are the
        spot ids of the receptor table. One section at a time (one coordinate frame).
    units
        ``"spot"``: every spot is one Threadfin "cell" (one clone per spot: the clone with most reads among the spot's
        ``chains`` records, ties to the smallest clone id). ``"locus"``: same-clone spots within ``loci_radius_um``
        are merged into one unit (:func:`spaclone.pp.merge_loci`; counts summed, centroid position), because one
        cell's transcripts spill over adjacent spots and would otherwise count as several cells of its clone.
    features
        ``"expression"``: Threadfin's own embedding of the unit's counts (``threadfin.pp.prepare_embedding``).
        ``"niche"``: the unit's place (:func:`niche_features` with ``niche`` and ``context``), not its own
        expression. ``"joint"``: both, each block scaled to the same total variance.
    clone_key
        Clone-id column of the receptor table (or of ``gex.obs``).
    donor_key, sample_key
        ``gex.obs`` columns passed to Threadfin (``sample_key`` is the context whose mean is removed and within which
        the coherence test shuffles). Without ``sample_key`` the section (``library_key`` if present, else one
        constant) is the context.
    state_key
        Optional ``gex.obs`` state annotation for Threadfin's programme composition (loci take the most common
        value of their spots).
    chains
        Receptor chains whose records define the spot's clone (default heavy chains); None = all.
    extra_spots
        Spots without a clone to include as context units (they shape the embedding, the context mean and the
        smoothing, as Threadfin's docs allow): a boolean ``obs`` column, a boolean array over spots or spot ids.
    loci_radius_um
        Locus linkage distance (``units="locus"``).
    context
        :class:`SpatialContext` (:func:`spatial_context`): compartments, masks and regions sampled at every unit and,
        for ``"niche"`` / ``"joint"``, the signed distances of the niche features. ``"auto"`` (default) builds it with
        :func:`spatial_context` only when that is certain: exactly one standard cell-type label column in
        ``gex.obs`` (:func:`spaclone.io.annotations.detect_annotations`) whose classes include T, B and erythroid
        classes of :data:`MASK_TYPES` (the spleen recipe: T zone, follicle, red pulp, the last named the non-B/T area
        when the labels show no red pulp); otherwise no context.
        ``None``: no context. ``result.context_note`` records which.
    niche
        :class:`NicheSpec` (or its fields as a dict) for ``"niche"`` / ``"joint"``.
    n_axes
        Number of clone-profile axes kept.
    landscape
        Also run ``threadfin.tl.landscape_genes`` (genes that follow the clone landscape; ``corr_axis1`` names the
        ends of axis 1).
    species
        ``"mouse"`` / ``"human"`` Ig genes for the Ig RNA strata (default: whichever is present).
    layer
        GEX layer with raw counts.
    spatial_key, spot_key, locus_key, reads_key, library_key
        Column / key names.
    threadfin_kwargs
        Further arguments of :func:`threadfin.run` (e.g. ``n_jobs``, ``n_perm``, ``n_boot``, ``map_n_neighbors``,
        ``map_min_dist``, ``verbose``); ``random_state`` comes from ``rng``.
    rng
        Seed: an integer is Threadfin's ``random_state`` unchanged (0 = Threadfin's default); a generator gives one
        draw.

    Returns
    -------
    ThreadfinSpatialResult
        Threadfin's result plus per-unit and per-clone tables (see the class).

    Notes
    -----
    The clone axes are the principal components of Threadfin's clone profiles, re-derived from the fitted feature map
    and checked against the stored profiles and against ``threadfin.profiles.project_features`` of every unit
    (``params["reproduction_max_abs_diff"]``; a :class:`ThreadfinWarning` if they differ by more than 1e-6).

    References
    ----------
    Rahimi & Recht 2007, NeurIPS 20; Muandet et al. 2017, doi:10.1561/2200000060; Searle, Casella & McCulloch 1992,
    doi:10.1002/9780470316856; Traag et al. 2019, doi:10.1038/s41598-019-41695-z (Leiden programmes); McInnes et al.
    2018, doi:10.21105/joss.00861 (clone map).
    """
    tf = _import_threadfin()
    from spaclone.gex.qc import get_counts

    if units not in ("spot", "locus"):
        raise ValueError(f"units must be 'spot' or 'locus', got {units!r}")
    if features not in ("expression", "niche", "joint"):
        raise ValueError(f"features must be 'expression', 'niche' or 'joint', got {features!r}")
    if not isinstance(n_axes, int | np.integer) or n_axes < 1:
        raise ValueError("n_axes must be a positive integer")
    kw = dict(threadfin_kwargs or {})
    if "random_state" in kw:
        raise ValueError("pass the seed as rng=, not threadfin_kwargs['random_state']")
    for k in ("donor_key", "sample_key", "state_key", "clone_key", "basis", "bcr"):
        if k in kw:
            raise ValueError(f"pass {k} as an argument of run(), not in threadfin_kwargs")
    kw.setdefault("verbose", False)
    seed = _seed(rng)
    spec = niche if isinstance(niche, NicheSpec) or niche is None else NicheSpec(**dict(niche))
    if isinstance(context, str) and context != "auto":
        raise ValueError(f"context must be a SpatialContext, 'auto' or None, got {context!r}")
    if features != "expression" and spec is None and context is None:
        raise ValueError(f"features={features!r} needs niche= (composition) and/or context= (distances)")

    air, gex = _unpack(data)
    if library_key in gex.obs and gex.obs[library_key].nunique() > 1:
        raise ValueError(
            f"gex holds {gex.obs[library_key].nunique()} sections ({library_key!r}); the spatial readouts need one "
            "coordinate frame: run one section at a time"
        )
    if isinstance(context, str):
        context, context_note = _auto_context(gex, spatial_key=spatial_key)
        logger.info("ext.threadfin.run: context %s", context_note)
    else:
        context_note = "none: context=None" if context is None else "given"
    if features != "expression" and spec is None and context is None:
        raise ValueError(
            f"features={features!r} needs niche= (composition) and/or context= (distances); no automatic context "
            f"({context_note})"
        )
    spec = spec or NicheSpec()
    spots = pd.Index(gex.obs_names.astype(str))
    if not spots.is_unique:
        raise ValueError("gex.obs_names (spot ids) must be unique")
    xy_spot = _xy_of(gex, spatial_key)
    counts_all = get_counts(gex, layer)

    # ---- one clone per spot
    clone_of_spot = pd.Series(np.nan, index=spots, dtype=object)
    n_without_gex = 0
    n_multi = 0
    if air is not None:
        dom = _dominant_clones(
            air, chains=chains, clone_key=clone_key, spot_key=spot_key, locus_key=locus_key, reads_key=reads_key
        )
        hit = dom.index.isin(spots)
        n_without_gex = int((~hit).sum())
        dom = dom[hit]
        clone_of_spot.loc[dom.index] = dom.clone.to_numpy()
        n_multi = int((dom.n_clones > 1).sum())
    else:
        if clone_key not in gex.obs:
            raise KeyError(f"gex.obs has no {clone_key!r} column and no receptor table was given")
        v = gex.obs[clone_key]
        clone_of_spot[:] = v.astype("object").map(lambda c: np.nan if pd.isna(c) else str(c)).to_numpy()
    clonal_spot = clone_of_spot.notna().to_numpy()
    if not clonal_spot.any():
        raise ValueError("no spot carries a clone")
    extra = _spot_mask(extra_spots, gex)

    # ---- units
    if units == "spot":
        rows = np.flatnonzero(clonal_spot | extra)
        unit_ids = spots[rows]
        members = sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), rows)), shape=(len(rows), len(spots)))
        u_xy = xy_spot[rows]
        u_clone = clone_of_spot.to_numpy()[rows]
        n_spots = np.ones(len(rows), dtype=np.int64)
        membership = None
    else:
        from spaclone.pp.loci import merge_loci

        cr = np.flatnonzero(clonal_spot)
        t = pd.DataFrame(
            {spot_key: spots[cr], S.X: xy_spot[cr, 0], S.Y: xy_spot[cr, 1], clone_key: clone_of_spot.to_numpy()[cr]}
        )
        loci, member = merge_loci(t, radius_um=loci_radius_um, clone_key=clone_key, by=(), spot_key=spot_key)
        lid = member.to_numpy()
        first = pd.Series(cr).groupby(lid).min()  # earliest gex row of every locus: keeps the gex order
        ex = np.flatnonzero(extra & ~clonal_spot)
        order_key = np.r_[first.reindex(loci.locus_id).to_numpy(), ex]
        ids = np.r_[[f"L{i}" for i in loci.locus_id], spots[ex].to_numpy()].astype(object)
        o = np.argsort(order_key, kind="stable")
        pos_of_locus = pd.Series(np.arange(len(loci)), index=loci.locus_id)
        rr = np.r_[pos_of_locus.reindex(lid).to_numpy(), len(loci) + np.arange(len(ex))]
        cc = np.r_[cr, ex]
        m0 = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(loci) + len(ex), len(spots)))
        members = m0[o]
        unit_ids = pd.Index(ids[o])
        u_xy = np.r_[loci[[S.X, S.Y]].to_numpy(dtype=float), xy_spot[ex]][o]
        u_clone = np.r_[loci[clone_key].astype(str).to_numpy(), np.full(len(ex), np.nan, dtype=object)][o]
        n_spots = np.asarray(members.sum(axis=1)).ravel().astype(np.int64)
        membership = pd.Series(unit_ids[np.argsort(o)][rr[: len(cr)]], index=spots[cr], name="unit_id")
    counts = sparse.csr_matrix(members @ counts_all, dtype=np.float32)

    # ---- the unit-level AnnData Threadfin runs on
    obs = pd.DataFrame(index=pd.Index(unit_ids.astype(str), name=None))
    obs[clone_key] = pd.Series(u_clone, index=obs.index, dtype=object)

    def per_unit(col: str) -> np.ndarray:
        v = gex.obs[col]
        if units == "spot":
            return v.to_numpy()[rows]
        first_spot = np.asarray(members.argmax(axis=1)).ravel()
        if col == state_key:  # most common state of the unit's spots
            codes, cats = pd.factorize(v.astype("object"))
            c = sparse.csr_matrix(
                (np.ones(len(codes)), (np.arange(len(codes)), np.maximum(codes, 0))), shape=(len(codes), len(cats))
            )
            best = np.asarray((members @ c).argmax(axis=1)).ravel()
            return np.asarray(cats, dtype=object)[best]
        return v.to_numpy()[first_spot]

    if sample_key is None:
        tf_sample = "spaclone_sample"
        obs[tf_sample] = per_unit(library_key).astype(str) if library_key in gex.obs else "section"
    else:
        if sample_key not in gex.obs:
            raise KeyError(f"sample_key {sample_key!r} not in gex.obs")
        tf_sample = sample_key
        obs[tf_sample] = per_unit(sample_key)
    if donor_key is not None:
        if donor_key not in gex.obs:
            raise KeyError(f"donor_key {donor_key!r} not in gex.obs")
        obs[donor_key] = per_unit(donor_key)
    if state_key is not None:
        if state_key not in gex.obs:
            raise KeyError(f"state_key {state_key!r} not in gex.obs")
        obs[state_key] = pd.Categorical(per_unit(state_key))
    tfa = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(gex.var_names.astype(str))))
    tfa.obsm["spatial"] = u_xy.astype(float)

    # ---- features
    niche_tab = None
    basis = None
    if features in ("niche", "joint"):
        niche_tab = niche_features(gex, u_xy, spec=spec, context=context, members=members, spatial_key=spatial_key)
        niche_tab.index = obs.index
        nb = _niche_basis(niche_tab, spec.clip_um)
        if features == "niche":
            tfa.obsm["X_niche"] = nb
            basis = "X_niche"
            # log-normalised counts for landscape genes and markers (Threadfin's prepare_embedding is not run)
            tot = np.maximum(np.asarray(tfa.X.sum(axis=1), dtype=float).ravel(), 1.0)
            ln = sparse.csr_matrix(sparse.diags(1e4 / tot) @ tfa.X)
            ln.data = np.log1p(ln.data)
            tfa.layers["log_norm"] = ln.astype(np.float32)
        else:
            tf.pp.prepare_embedding(tfa, random_state=seed, verbose=False)
            e = np.asarray(tfa.obsm["X_threadfin"], dtype=float)
            e = (e - e.mean(axis=0)) / np.sqrt(max(float(e.var(axis=0).sum()), 1e-12))
            n = nb.astype(float)
            n = n / np.sqrt(max(float(n.var(axis=0).sum()), 1e-12))
            tfa.obsm["X_joint"] = np.ascontiguousarray(np.hstack([e, n]) / np.sqrt(2.0), dtype=np.float32)
            basis = "X_joint"
    result = tf.run(
        tfa,
        donor_key=donor_key,
        sample_key=tf_sample,
        state_key=state_key,
        clone_key=clone_key,
        basis=basis,
        random_state=seed,
        **kw,
    )

    # ---- spatial layer: axes re-derived from the fitted feature map
    prof = tf.profiles.get_profiles(tfa)
    pp_ = prof["params"]
    model = tf.profiles.model_from_adata(tfa)
    engine = _ProfileEngine(
        model.residuals,
        min_cells=int(pp_.get("min_cells", 2)),
        n_components=int(pp_.get("n_components", 30)),
        n_axes=n_axes,
    )
    # Threadfin's own projection of every unit (as for bootstrap replicates), for the check below
    tf_pos = np.asarray(tf.profiles.project_features(tfa, model.residuals - model.vc.grand_mean[None, :]))
    del model
    clone_index = pd.Index(pd.unique(pd.Series(u_clone).dropna().astype(str)))
    codes = clone_index.get_indexer(pd.Series(u_clone).astype(str).where(pd.notna(u_clone)))
    fit = engine.fit(codes)
    prof_ids = clone_index[fit.clone_rows]
    stored = prof["features"].reindex(prof_ids.astype(str))
    diff = np.nan
    if stored.notna().all().all() and len(stored):
        st = stored.to_numpy(dtype=float)[:, : fit.values.shape[1]]
        diff = float(
            max(
                np.min([np.abs(fit.values[:, j] - st[:, j]).max(), np.abs(fit.values[:, j] + st[:, j]).max()])
                for j in range(st.shape[1])
            )
        )
    na_ = fit.values.shape[1]
    pdiff = float(
        max(
            min(np.abs(fit.positions[:, j] - tf_pos[:, j]).max(), np.abs(fit.positions[:, j] + tf_pos[:, j]).max())
            for j in range(min(na_, tf_pos.shape[1]))
        )
    )
    del tf_pos
    diff = max(diff, pdiff) if np.isfinite(diff) else diff
    if not np.isfinite(diff) or diff > 1e-6:
        warnings.warn(
            f"clone axes re-derived from Threadfin's feature map differ from the stored profiles (max |diff| {diff:.3g})",
            ThreadfinWarning,
            stacklevel=2,
        )
    na = fit.values.shape[1]
    ax_cols = [f"axis_{j + 1}" for j in range(na)]

    # ---- per-unit table
    U = pd.DataFrame(index=pd.Index(obs.index, name="unit_id"))
    U[S.X] = u_xy[:, 0]
    U[S.Y] = u_xy[:, 1]
    U[clone_key] = obs[clone_key].to_numpy()
    U["clonal"] = codes >= 0
    profiled_code = np.zeros(len(clone_index), dtype=bool)
    profiled_code[fit.clone_rows] = True
    U["profiled"] = (codes >= 0) & profiled_code[np.maximum(codes, 0)]
    U["n_spots"] = n_spots
    U["total_counts"] = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    ig = _ig_counts(counts, tfa.var_names, species)
    U["ig_rna"] = ig
    U["ig_bin"] = bin_labels(ig, IG_RNA_BINS) if np.isfinite(ig).any() else "NA"
    q = np.unique(np.quantile(U.total_counts, np.linspace(0, 1, 6)))
    U["count_bin"] = np.clip(np.searchsorted(q, U.total_counts, side="right") - 1, 0, max(len(q) - 2, 0)).astype(str)
    for j, c in enumerate(ax_cols):
        U[c] = fit.positions[:, j]
    val_of_code = np.full((len(clone_index), na), np.nan)
    val_of_code[fit.clone_rows] = fit.values
    for j in range(na):
        U[f"clone_{ax_cols[j]}"] = np.where(codes >= 0, val_of_code[np.maximum(codes, 0), j], np.nan)
    if "clone_programme" in tfa.obs:
        U["programme"] = tfa.obs["clone_programme"].astype("object").to_numpy()
    if context is not None:
        smp = context.sample(u_xy[:, 0], u_xy[:, 1])
        for c in smp.columns:
            U[c] = smp[c].to_numpy()

    # ---- per-clone table
    ct = result.clones
    C = pd.DataFrame(index=pd.Index(prof_ids.astype(str), name=clone_key))
    C["n_units"] = fit.sizes[fit.clone_rows].astype(int)
    C["reliability"] = fit.reliability
    for src, dst in (("x", "map_x"), ("y", "map_y"), ("clone_programme", "programme"), ("confidence", "confidence")):
        if src in ct.columns:
            C[dst] = ct[src].reindex(C.index).to_numpy()
    for j, c in enumerate(ax_cols):
        C[c] = fit.values[:, j]
    pr = U[U.profiled]
    g = pr.groupby(clone_key, sort=False)
    for c in ax_cols:
        C[f"unit_mean_{c}"] = g[c].mean().reindex(C.index).to_numpy()
    C["n_spots"] = g.n_spots.sum().reindex(C.index).to_numpy()
    A = pd.DataFrame(
        {
            "axis": np.arange(1, na + 1),
            "variance": np.var(fit.values, axis=0),
            "share": fit.share,
            "share_all": fit.share_all,
        }
    )

    land = None
    if landscape:
        land = tf.tl.landscape_genes(tfa, n_top=None, verbose=False)
    params = {
        "units": units,
        "features": features,
        "clone_key": clone_key,
        "donor_key": donor_key,
        "sample_key": tf_sample,
        "state_key": state_key,
        "chains": list(chains) if chains is not None else None,
        "extra_spots": int(extra.sum()),
        "loci_radius_um": loci_radius_um if units == "locus" else None,
        "n_axes": int(na),
        "niche": None if features == "expression" else spec.__dict__,
        "context": context.params if context is not None else None,
        "context_note": context_note,
        "threadfin_version": str(tf.__version__),
        "threadfin_kwargs": kw,
        "random_state": seed,
        "rng_entropy": seed,
        "n_spots_without_gex": n_without_gex,
        "n_multi_clone_spots": n_multi,
        "reproduction_max_abs_diff": diff,
        "projection_max_abs_diff": pdiff,
        "min_cells": engine.min_cells,
        "n_components": engine.n_components,
        "basis": result.settings.get("basis"),
    }
    return ThreadfinSpatialResult(
        threadfin=result,
        units=U,
        clones=C,
        axes=A,
        params=params,
        niche=niche_tab,
        context=context,
        membership=membership,
        landscape=land,
        engine=engine,
        context_note=context_note,
    )


# ========================================================================================================= readouts


def _check_result(result: Any, *, engine: bool = True) -> None:
    if not isinstance(result, ThreadfinSpatialResult) or (engine and result.engine is None):
        raise TypeError(
            "expected the ThreadfinSpatialResult of spaclone.ext.threadfin.run"
            + (" (with its feature matrix; permutation nulls re-fit the profiles)" if engine else "")
        )


def _table_fit(result: ThreadfinSpatialResult, codes: np.ndarray) -> _Fit:
    """The observed fit read back from ``result.units`` (unit positions and clone values of every kept axis)."""
    u = result.units
    na = int(result.params["n_axes"])
    n_codes = int(codes.max()) + 1 if (codes >= 0).any() else 0
    sizes = np.bincount(codes[codes >= 0], minlength=n_codes).astype(float)
    cv = u[[f"clone_axis_{j + 1}" for j in range(na)]].to_numpy(dtype=float)
    pos = u[[f"axis_{j + 1}" for j in range(na)]].to_numpy(dtype=float)
    prof = np.isfinite(cv[:, 0]) & (codes >= 0)
    rows = np.unique(codes[prof])
    first = pd.Series(np.flatnonzero(prof)).groupby(codes[prof]).first()
    values = cv[first.reindex(rows).to_numpy()]
    return _Fit(
        clone_rows=rows,
        sizes=sizes,
        values=values,
        positions=pos,
        share=np.full(na, np.nan),
        share_all=np.full(na, np.nan),
        reliability=np.full(len(rows), np.nan),
        components=np.empty((na, 0)),
    )


def _axis_index(result: ThreadfinSpatialResult, axis: int) -> int:
    na = int(result.params.get("n_axes", 0))
    if not 1 <= int(axis) <= na:
        raise ValueError(f"axis must be between 1 and {na} (run(n_axes=...) keeps {na} axes)")
    return int(axis) - 1


def _codes(result: ThreadfinSpatialResult) -> tuple[np.ndarray, pd.Index]:
    ck = result.params["clone_key"]
    lab = result.units[ck]
    index = pd.Index(pd.unique(lab.dropna().astype(str)))
    return index.get_indexer(lab.astype(str).where(lab.notna())), index


def _strata(result: ThreadfinSpatialResult, within: str | Sequence[str] | None) -> list[str]:
    cols = [within] if isinstance(within, str) else list(within or [])
    miss = [c for c in cols if c not in result.units.columns]
    if miss:
        raise KeyError(f"strata column(s) {miss} not in result.units (columns: {list(result.units.columns)})")
    return cols


def _label_placebo(
    result: ThreadfinSpatialResult,
    statistic: Callable[[_Fit, np.ndarray], dict[str, float]],
    *,
    within: list[str],
    n_perm: int,
    alternative: str,
    rng: RngLike,
    name: str,
) -> Any:
    """Run ``statistic(fit, codes)`` on the observed labels and on clone labels permuted among clonal units."""
    ck = result.params["clone_key"]
    codes, index = _codes(result)
    clonal = np.flatnonzero(codes >= 0)
    tab = result.units.iloc[clonal][[ck, *within]].reset_index(drop=True).copy()
    tab[ck] = tab[ck].astype(str)
    for c in within:
        tab[c] = tab[c].astype("object").where(tab[c].notna(), "NA").astype(str)
    tab["_row"] = clonal
    n_units = len(result.units)
    engine = result.engine
    ref = engine.fit(codes).components  # null axes are oriented like the observed ones

    def stat(df: pd.DataFrame) -> dict[str, float]:
        cd = np.full(n_units, -1, dtype=np.int64)
        cd[df["_row"].to_numpy()] = index.get_indexer(df[ck].to_numpy())
        return statistic(engine.fit(cd, reference=ref), cd)

    stat.__name__ = name
    design = Permute(ck, within=within or None)
    return placebo(stat, tab, design, n_perm=n_perm, alternative=alternative, rng=rng, name=name)


def _std_table(pr: Any, alternative: str, extra: pd.DataFrame | None = None, q_groups: Sequence[str] | None = None):
    """Standard columns from a PlaceboResult table: statistic, expected, null_mean, null_sd, z, p_value, q_value."""
    t = pr.table
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (t.observed - t.null_mean) / t.null_sd.where(t.null_sd > 0)
    out = pd.DataFrame(
        {
            "component": t.component.to_numpy(),
            "statistic": t.observed.to_numpy(),
            "expected": t.null_mean.to_numpy(),
            "null_mean": t.null_mean.to_numpy(),
            "null_sd": t.null_sd.to_numpy(),
            "null_lo": t.null_lo.to_numpy(),
            "null_hi": t.null_hi.to_numpy(),
            "z": z.to_numpy(),
            "p_value": t.p_value.to_numpy(),
            "p_maxT": t.p_maxT.to_numpy(),
            "n_perm": t.n_perm.to_numpy(),
        }
    )
    if extra is not None:
        for c in extra.columns:
            out[c] = extra[c].to_numpy()
    out["q_value"] = np.nan
    groups = list(q_groups) if q_groups is not None else [None]
    for g in groups:
        sel = np.ones(len(out), bool) if g is None else (out["kind"] == g).to_numpy()
        out.loc[sel, "q_value"] = bh(out.loc[sel, "p_value"].to_numpy())
    out.attrs["alternative"] = alternative
    return out


def _unit_clone_values(fit: _Fit, codes: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Per unit: profiled flag and the clone's value on axis j (NaN when not profiled)."""
    n_codes = len(fit.sizes)
    val = np.full(n_codes, np.nan)
    val[fit.clone_rows] = fit.values[:, j]
    ok = codes >= 0
    v = np.where(ok, val[np.maximum(codes, 0)], np.nan)
    return np.isfinite(v), v


def niche_by_axis(
    result: ThreadfinSpatialResult,
    *,
    axis: int = 1,
    n_bins: int = 10,
    compartment_key: str = "compartment",
    flags: Sequence[str] | None = None,
    within: str | Sequence[str] | None = ("ig_bin",),
    n_perm: int = 199,
    rng: RngLike = None,
) -> TestResult:
    """Compartment shares of clone members by clone-axis decile, against permuted clone labels.

    Profiled clones are cut into ``n_bins`` equal-count bins of their axis value (ranks, ties by order); for every
    bin and compartment the share of the bin's member units in that compartment is the statistic, and per
    compartment the trend is the Spearman correlation between a member unit's clone value and its being in the
    compartment.

    Parameters
    ----------
    result
        :func:`run` result with a compartment column (run with ``context=``).
    axis
        Clone-profile axis (1 = the main one).
    n_bins
        Number of clone bins (10 = deciles).
    compartment_key
        ``result.units`` column of compartments.
    flags
        Further boolean ``result.units`` columns reported the same way (e.g. ``"in_csr_shm_overlap"``).
    within
        ``result.units`` columns inside which clone labels are permuted (default the Ig RNA depth stratum).
    n_perm
        Permutations.
    rng
        Seed or generator.

    Returns
    -------
    TestResult
        ``table``: one row per (``kind`` = ``"share"``, ``bin``, ``category``) and per (``"trend"``, ``category``):
        ``statistic``, ``expected`` / ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``, ``z``, ``p_value``
        (two-sided), ``q_value`` (BH within kind), ``n_perm``, ``n_units``. ``summary``: shares of the first and
        last bin and the trend with its p-value per category.

    Notes
    -----
    Null model: clone labels are permuted among the units that carry a clone, within strata of ``within``; unit
    positions, expression, compartments and clone sizes stay fixed. Each permuted labelling re-fits Threadfin's
    clone profiles from the fixed unit features (variance components, shrinkage, principal components), with each
    null axis oriented like the observed one, so the null contains what any grouping of units of the same sizes
    produces: a clone's value is computed from its own members, so members' compartments follow the clone value even
    for random groups. The observed excess over this null is the part of the gradient due to clone members sharing
    places.

    References
    ----------
    Besag & Diggle 1977, doi:10.2307/2346974; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Benjamini &
    Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    _check_result(result)
    j = _axis_index(result, axis)
    if compartment_key not in result.units.columns:
        raise KeyError(f"result.units has no {compartment_key!r}: run(..., context=spatial_context(...))")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    comp = result.units[compartment_key].astype(str).to_numpy()
    cats = [c for c in pd.unique(comp) if c != "nan"]
    first = (*WHITE_PULP, *RED_PULP)  # T zone, follicle, then red pulp under either name
    order = [c for c in first if c in cats] + sorted(c for c in cats if c not in first)
    ind = {c: (comp == c).astype(float) for c in order}
    for f in flags or []:
        if f not in result.units.columns:
            raise KeyError(f"flag column {f!r} not in result.units")
        ind[f] = result.units[f].fillna(False).astype(bool).to_numpy().astype(float)
    names = list(ind)
    within_ = _strata(result, within)
    labels = [f"D{b + 1}" for b in range(n_bins)]

    def bins_of(fit: _Fit, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = pd.Series(fit.values[:, j]).rank(method="first").to_numpy()
        b = pd.qcut(r, n_bins, labels=False)
        bin_code = np.full(len(fit.sizes), -1, dtype=np.int64)
        bin_code[fit.clone_rows] = b
        ub = np.where(codes >= 0, bin_code[np.maximum(codes, 0)], -1)
        return ub, b

    def statistic(fit: _Fit, codes: np.ndarray) -> dict[str, float]:
        ub, _ = bins_of(fit, codes)
        sel = ub >= 0
        _, cv = _unit_clone_values(fit, codes, j)
        nb = np.bincount(ub[sel], minlength=n_bins).astype(float)
        out: dict[str, float] = {}
        for k in names:
            sh = np.bincount(ub[sel], weights=ind[k][sel], minlength=n_bins) / np.maximum(nb, 1)
            for b in range(n_bins):
                out[f"share|{labels[b]}|{k}"] = float(sh[b])
        for k in names:
            out[f"trend||{k}"] = _spearman(cv[sel], ind[k][sel])
        return out

    pr = _label_placebo(
        result, statistic, within=within_, n_perm=n_perm, alternative="two-sided", rng=rng, name="niche_by_axis"
    )
    parts = pr.table.component.str.split("|", expand=True)
    codes, _ = _codes(result)
    fit0 = result.engine.fit(codes)
    ub, _ = bins_of(fit0, codes)
    n_units = []
    for kind, b, _k in parts.itertuples(index=False):
        n_units.append(int((ub == labels.index(b)).sum()) if kind == "share" else int((ub >= 0).sum()))
    extra = pd.DataFrame(
        {"kind": parts[0], "bin": parts[1].replace("", np.nan), "category": parts[2], "n_units": n_units}
    )
    table = _std_table(pr, "two-sided", extra, q_groups=["share", "trend"])
    table = table[
        [
            "kind",
            "bin",
            "category",
            "n_units",
            *[c for c in table.columns if c not in ("kind", "bin", "category", "n_units", "component")],
        ]
    ]
    summ: dict[str, Any] = {
        "axis": int(axis),
        "n_profiled_clones": int(len(fit0.clone_rows)),
        "n_units": int((ub >= 0).sum()),
    }
    for k in names:
        sh = table[(table.kind == "share") & (table.category == k)]
        tr = table[(table.kind == "trend") & (table.category == k)].iloc[0]
        summ[f"{k} {labels[0]}"] = float(sh.statistic.iloc[0])
        summ[f"{k} {labels[-1]}"] = float(sh.statistic.iloc[-1])
        summ[f"trend[{k}]"] = float(tr.statistic)
        summ[f"trend_null[{k}]"] = float(tr.null_mean)
        summ[f"p[{k}]"] = float(tr.p_value)
    params = {
        "axis": int(axis),
        "n_bins": int(n_bins),
        "compartment_key": compartment_key,
        "flags": list(flags or []),
        "design": {
            "design": "Permute",
            "columns": [result.params["clone_key"]],
            "within": within_,
            "refit": "Threadfin clone profiles re-fitted per permutation",
        },
        "n_perm": int(n_perm),
        "alternative": "two-sided",
        "rng_entropy": pr.params.get("rng_entropy"),
    }
    return TestResult(
        name="niche_by_axis", table=table.reset_index(drop=True), summary=summ, params=params, null=pr.null
    )


def spatial_coherence(
    result: ThreadfinSpatialResult,
    *,
    axis: int = 1,
    radius_um: float = 50.0,
    within: str | Sequence[str] | None = ("ig_bin",),
    n_perm: int = 199,
    rng: RngLike = None,
) -> TestResult:
    """Is a unit's state its clone's state, and are neighbouring clones alike, beyond chance groupings?

    Three statistics over the units of profiled clones (Spearman correlations):

    * ``unit_vs_own_clone``: a unit's own axis position against its clone's value (the clone value contains the
      unit itself);
    * ``unit_vs_rest_of_clone``: a unit's position against the mean position of the *other* units of its clone;
    * ``clone_vs_other_clones``: a unit's clone value against the mean clone value of other clones' units within
      ``radius_um`` (units with at least one such neighbour).

    Parameters
    ----------
    result
        :func:`run` result.
    axis
        Clone-profile axis.
    radius_um
        Neighbourhood radius (inclusive) of ``clone_vs_other_clones``.
    within
        ``result.units`` columns inside which clone labels are permuted (default the Ig RNA depth stratum; add
        ``"compartment"`` to ask whether clones are coherent beyond their compartment mix).
    n_perm
        Permutations.
    rng
        Seed or generator.

    Returns
    -------
    TestResult
        ``table``: one row per statistic with ``statistic``, ``expected`` (null mean), ``null_sd``, ``z``,
        ``p_value`` (one-sided, greater), ``q_value``, ``n_perm``, ``n_units``.

    Notes
    -----
    Null model: clone labels are permuted among the units that carry a clone, within strata of ``within``
    (``spaclone.nulls.Permute``); positions, expression and clone sizes stay fixed, and Threadfin's clone profiles are
    re-fitted on every permuted labelling. Under this null a unit still sits in its (random) clone, and small clones'
    values are pulled towards their own members, so the expected correlations are not 0; the comparison with
    permuted labels is what calibrates them. With 10-µm bins the units of one cell spill over adjacent bins; use
    ``run(units="locus")`` to keep one cell from counting as several members.

    References
    ----------
    Besag & Diggle 1977, doi:10.2307/2346974; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Hurlbert 1984,
    doi:10.2307/1942661.
    """
    _check_result(result)
    j = _axis_index(result, axis)
    if not radius_um > 0:
        raise ValueError("radius_um must be positive")
    within_ = _strata(result, within)
    codes0, _ = _codes(result)
    clonal = np.flatnonzero(codes0 >= 0)
    xy = result.units[[S.X, S.Y]].to_numpy(dtype=float)
    pairs = cKDTree(xy[clonal]).query_pairs(radius_um + 1e-9, output_type="ndarray")
    pi, pj = clonal[pairs[:, 0]], clonal[pairs[:, 1]]

    def statistic(fit: _Fit, codes: np.ndarray) -> dict[str, float]:
        prof, cv = _unit_clone_values(fit, codes, j)
        pos = fit.positions[:, j]
        sums = np.bincount(codes[prof], weights=pos[prof], minlength=len(fit.sizes))
        n = fit.sizes[np.maximum(codes, 0)]
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = np.where(prof, (sums[np.maximum(codes, 0)] - pos) / (n - 1), np.nan)
        keep = prof[pi] & prof[pj] & (codes[pi] != codes[pj])
        a, b = pi[keep], pj[keep]
        ends = np.r_[a, b]
        s = np.bincount(ends, weights=np.r_[cv[b], cv[a]], minlength=len(codes))
        m = np.bincount(ends, minlength=len(codes))
        nbr = np.where(m > 0, s / np.maximum(m, 1), np.nan)
        has = m > 0
        return {
            "unit_vs_own_clone": _spearman(pos[prof], cv[prof]),
            "unit_vs_rest_of_clone": _spearman(pos[prof], loo[prof]),
            "clone_vs_other_clones": _spearman(cv[has], nbr[has]),
        }

    pr = _label_placebo(
        result, statistic, within=within_, n_perm=n_perm, alternative="greater", rng=rng, name="spatial_coherence"
    )
    fit0 = result.engine.fit(codes0)
    prof0, _ = _unit_clone_values(fit0, codes0, j)
    keep = prof0[pi] & prof0[pj] & (codes0[pi] != codes0[pj])
    n_nb = int(len(np.unique(np.r_[pi[keep], pj[keep]])))
    extra = pd.DataFrame({"n_units": [int(prof0.sum()), int(prof0.sum()), n_nb]})
    table = _std_table(pr, "greater", extra)
    table = table.rename(columns={"component": "statistic_name"})
    summ = {"axis": int(axis), "radius_um": float(radius_um)}
    for r in table.itertuples():
        summ[r.statistic_name] = float(r.statistic)
        summ[f"{r.statistic_name}_null"] = float(r.expected)
        summ[f"p[{r.statistic_name}]"] = float(r.p_value)
    params = {
        "axis": int(axis),
        "radius_um": float(radius_um),
        "design": {
            "design": "Permute",
            "columns": [result.params["clone_key"]],
            "within": within_,
            "refit": "Threadfin clone profiles re-fitted per permutation",
        },
        "n_perm": int(n_perm),
        "alternative": "greater",
        "rng_entropy": pr.params.get("rng_entropy"),
    }
    return TestResult(name="spatial_coherence", table=table, summary=summ, params=params, null=pr.null)


# ------------------------------------------------------------------------------------------- rigid relocation helpers


def _rigid_landings(
    pts: np.ndarray,
    slot_xy: np.ndarray,
    tree: cKDTree,
    *,
    n_draws: int,
    snap_um: float,
    max_tries: int,
    gen: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Slot indices under random rigid copies of a point set (rotation, reflection, translation onto a slot).

    Returns ``(idx, n_valid)``: ``idx`` has shape ``(n_draws, m)``; rows beyond ``n_valid`` are -1 (a copy is valid
    when every point has a slot within ``snap_um``; at most ``max_tries * n_draws`` copies are tried).
    """
    m = len(pts)
    rel = pts - pts.mean(axis=0)
    out = np.full((n_draws, m), -1, dtype=np.int64)
    got, tried, budget = 0, 0, int(max_tries) * int(n_draws)
    while got < n_draws and tried < budget:
        k = int(min(max(4 * (n_draws - got), 16), budget - tried, max(16, 2_000_000 // max(m, 1))))
        theta = gen.uniform(0.0, 2 * np.pi, k)
        flip = gen.random(k) < 0.5
        cen = slot_xy[gen.integers(0, len(slot_xy), size=k)]
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        rx = np.where(flip[:, None], -rel[None, :, 0], rel[None, :, 0])
        ry = np.broadcast_to(rel[None, :, 1], rx.shape)
        nx = c * rx - s * ry + cen[:, 0, None]
        ny = s * rx + c * ry + cen[:, 1, None]
        d, idx = tree.query(np.column_stack([nx.ravel(), ny.ravel()]), distance_upper_bound=snap_um)
        ok = np.isfinite(d).reshape(k, m).all(axis=1)
        acc = idx.reshape(k, m)[ok]
        take = min(len(acc), n_draws - got)
        out[got : got + take] = acc[:take]
        got += take
        tried += k
    return out, got


def _min_dist_median(d0: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Median over ``src`` points of the distance to the nearest ``dst`` point (NaN if either set is empty)."""
    if not src.any() or not dst.any():
        return float("nan")
    return float(np.median(d0[np.ix_(src, dst)].min(axis=1)))


def spanning_clones(
    result: ThreadfinSpatialResult,
    *,
    axis: int = 1,
    min_units: int = 5,
    null: Literal["relocate", "permute"] = "relocate",
    n_perm: int = 199,
    within: str | Sequence[str] | None = ("ig_bin",),
    snap_um: float = 20.0,
    slots: Literal["input", "clonal"] = "input",
    max_tries: int = 50,
    compartment_key: str = "compartment",
    white: Sequence[str] = WHITE_PULP,
    red: Sequence[str] = RED_PULP,
    rng: RngLike = None,
) -> TestResult:
    """Clones whose members sit at both ends of a clone axis: how spread they are and how far apart the two ends lie.

    Units of profiled clones are cut into tertiles of their own axis position; an *expanded* clone has at least
    ``min_units`` units and *spans* when it has at least one unit in the bottom and one in the top tertile. Per
    expanded clone: ``spanning``, ``axis_range`` (max − min of its units' positions), ``cross_um`` (median, over its
    bottom-tertile units, of the distance to the nearest top-tertile unit of the clone), ``white_and_red`` (units in
    the ``white`` and in the ``red`` compartments: white pulp and red pulp on a spleen, B / T zones and the non-B/T
    area elsewhere), ``red_to_white_um`` (median distance from a ``red`` unit to the nearest ``white`` unit of the
    clone), ``median_pairwise_um`` and ``mst_um`` (spread).

    Parameters
    ----------
    result
        :func:`run` result (a compartment column is needed for the white / red columns). The relocation null reads
        positions from ``result.units`` only, so a result rebuilt from saved tables works too.
    axis
        Clone-profile axis.
    min_units
        Units of an expanded clone.
    null
        ``"relocate"`` (default): every expanded clone is moved as a rigid body; ``"permute"``: clone labels permuted
        within strata (members scattered), the axis re-fitted.
    n_perm
        Null draws.
    within
        Strata of the label permutation (``null="permute"``).
    snap_um
        Relocation: a relocated unit lands on the nearest slot unit within this distance; a copy with a unit that
        finds none is rejected and redrawn.
    slots
        Relocation landing units: ``"input"`` (every unit Threadfin ran on) or ``"clonal"`` (units with a clone).
    max_tries
        Relocation: at most ``max_tries * n_perm`` copies are tried per clone (``n_valid`` reports how many were
        accepted; a clone too large for the tissue gets NaN p-values).
    compartment_key, white, red
        Compartment column and the compartments counted as white (default :data:`WHITE_PULP`: T zone and follicle)
        and as red (default :data:`RED_PULP`: the red pulp or, under tissue-neutral names, the non-B/T area; see
        :func:`spatial_context`).
    rng
        Seed or generator.

    Returns
    -------
    TestResult
        ``table``: one row per expanded clone with the statistics above, their null means and one-sided p-values:
        ``p_range`` (``axis_range`` smaller than its null: members alike beyond their places) and ``p_cross``
        (``cross_um`` smaller: the two ends lie closer than the clone's shape implies) under relocation; ``p_spread``
        (``median_pairwise_um`` smaller) under permutation. ``summary``: ``n_expanded``, ``n_spanning``,
        ``spanning_share`` and the medians, each with its null mean and a two-sided p-value of the aggregate.

    Notes
    -----
    Null model (``"relocate"``): each expanded clone is rotated by a uniform angle, reflected with probability 1/2 and
    translated so that its centroid falls on a random slot unit; every unit lands on the nearest slot unit within
    ``snap_um``, whose axis position and compartment it takes. The clone's size and internal geometry are kept
    (distances are those of the observed footprint); what changes is where in the tissue it lies. States belong to
    places here, so a clone that spans more or less than its relocated copies spans for reasons other than its shape.
    Null model (``"permute"``): clone labels permuted among clonal units within ``within`` strata, profiles re-fitted;
    members are scattered, so spread is tested against random groups of the same sizes.

    References
    ----------
    Lotwick & Silverman 1982, doi:10.1111/j.2517-6161.1982.tb01221.x; Alexander-Bloch et al. 2018,
    doi:10.1016/j.neuroimage.2018.05.070; Besag & Diggle 1977, doi:10.2307/2346974; Phipson & Smyth 2010,
    doi:10.2202/1544-6115.1585.
    """
    if null not in ("relocate", "permute"):
        raise ValueError("null must be 'relocate' or 'permute'")
    _check_result(result, engine=null == "permute")
    j = _axis_index(result, axis)
    if min_units < 2:
        raise ValueError("min_units must be at least 2")
    U = result.units
    ck = result.params["clone_key"]
    gen = as_rng(rng)
    codes, index = _codes(result)
    fit = result.engine.fit(codes) if null == "permute" else _table_fit(result, codes)
    prof, _ = _unit_clone_values(fit, codes, j)
    pos = fit.positions[:, j]
    q1, q2 = np.quantile(pos[prof], [1 / 3, 2 / 3])
    xy = U[[S.X, S.Y]].to_numpy(dtype=float)
    has_comp = compartment_key in U.columns
    comp = U[compartment_key].astype(str).to_numpy() if has_comp else np.full(len(U), "")
    is_w = np.isin(comp, list(white))
    is_r = np.isin(comp, list(red))
    sizes = fit.sizes
    exp_codes = np.intersect1d(np.flatnonzero(sizes >= min_units), fit.clone_rows)
    rows = {c: np.flatnonzero(codes == c) for c in exp_codes}

    def clone_stats(r: np.ndarray, d0: np.ndarray, land: np.ndarray) -> tuple[float, float, float, float, float]:
        """Statistics of a clone whose units ``r`` (geometry d0) take the states of the units ``land``."""
        a = pos[land]
        lo, hi = a <= q1, a >= q2
        sp = float(lo.any() and hi.any())
        cross = _min_dist_median(d0, lo, hi) if sp else np.nan
        w, rd = is_w[land], is_r[land]
        wr = float(w.any() and rd.any()) if has_comp else np.nan
        r2w = _min_dist_median(d0, rd, w) if has_comp and wr else np.nan
        return sp, float(a.max() - a.min()), cross, wr, r2w

    obs_rows = []
    geo = {}
    for c in exp_codes:
        r = rows[c]
        d0 = np.sqrt(((xy[r, None, :] - xy[None, r, :]) ** 2).sum(-1))
        geo[c] = d0
        sp, rg, cross, wr, r2w = clone_stats(r, d0, r)
        iu = np.triu_indices(len(r), 1)
        from scipy.sparse.csgraph import minimum_spanning_tree

        mst = float(minimum_spanning_tree(sparse.csr_matrix(d0 + 1e-9 * (d0 == 0) * (1 - np.eye(len(r))))).sum())
        obs_rows.append(
            {
                ck: index[c],
                "n_units": int(len(r)),
                "clone_axis": float(fit.values[np.searchsorted(fit.clone_rows, c), j]),
                "spanning": bool(sp),
                "axis_range": rg,
                "cross_um": cross,
                "white_and_red": (bool(wr) if np.isfinite(wr) else np.nan),
                "red_to_white_um": r2w,
                "median_pairwise_um": float(np.median(d0[iu])),
                "mst_um": mst,
                "share_red": float(is_r[r].mean()) if has_comp else np.nan,
            }
        )
    T = pd.DataFrame(obs_rows)
    n_exp = len(exp_codes)
    agg_names = [
        "spanning_share",
        "median_axis_range",
        "median_cross_um",
        "white_and_red_share",
        "median_red_to_white_um",
        "median_pairwise_um",
    ]

    def aggregates(sp, rg, cross, wr, r2w, spread) -> np.ndarray:
        sp = np.asarray(sp, float)
        spanning = sp == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.array(
                [
                    np.nanmean(sp),
                    np.nanmedian(rg),
                    np.nanmedian(np.where(spanning, cross, np.nan)) if spanning.any() else np.nan,
                    np.nanmean(np.where(spanning, wr, np.nan)) if spanning.any() else np.nan,
                    np.nanmedian(np.where(spanning, r2w, np.nan)) if spanning.any() else np.nan,
                    np.nanmedian(spread),
                ]
            )

    obs_agg = aggregates(
        T.spanning.astype(float),
        T.axis_range,
        T.cross_um,
        T.white_and_red.astype(float),
        T.red_to_white_um,
        T.median_pairwise_um,
    )
    null_cl = {k: np.full((n_perm, n_exp), np.nan) for k in ("sp", "rg", "cross", "wr", "r2w", "spread")}
    n_valid = np.zeros(n_exp, dtype=np.int64)
    if null == "relocate":
        slot_rows = np.arange(len(U)) if slots == "input" else np.flatnonzero(codes >= 0)
        if slots not in ("input", "clonal"):
            raise ValueError("slots must be 'input' or 'clonal'")
        tree = cKDTree(xy[slot_rows])
        for k, c in enumerate(exp_codes):
            r = rows[c]
            idx, got = _rigid_landings(
                xy[r], xy[slot_rows], tree, n_draws=n_perm, snap_um=snap_um, max_tries=max_tries, gen=gen
            )
            n_valid[k] = got
            d0 = geo[c]
            for b in range(got):
                land = slot_rows[idx[b]]
                sp, rg, cross, wr, r2w = clone_stats(r, d0, land)
                null_cl["sp"][b, k], null_cl["rg"][b, k], null_cl["cross"][b, k] = sp, rg, cross
                null_cl["wr"][b, k], null_cl["r2w"][b, k] = wr, r2w
            null_cl["spread"][:got, k] = T.median_pairwise_um.iloc[k]
        design = {
            "design": "rigid relocation of whole clones",
            "snap_um": float(snap_um),
            "slots": slots,
            "max_tries": int(max_tries),
            "rotation": "uniform",
            "reflection": 0.5,
            "translation": "centroid onto a random slot unit",
        }
    else:
        within_ = _strata(result, within)
        tab = U.iloc[np.flatnonzero(codes >= 0)][[ck, *within_]].reset_index(drop=True).copy()
        tab[ck] = tab[ck].astype(str)
        for cc in within_:
            tab[cc] = tab[cc].astype("object").where(tab[cc].notna(), "NA").astype(str)
        rows_cl = np.flatnonzero(codes >= 0)
        perm = Permute(ck, within=within_ or None)
        children = gen.spawn(n_perm)
        for b in range(n_perm):
            t2 = perm(tab, children[b])
            cd = np.full(len(U), -1, dtype=np.int64)
            cd[rows_cl] = index.get_indexer(t2[ck].to_numpy())
            f2 = result.engine.fit(cd, reference=fit.components)
            pr2, _ = _unit_clone_values(f2, cd, j)
            p2 = f2.positions[:, j]
            qq1, qq2 = np.quantile(p2[pr2], [1 / 3, 2 / 3])
            for k, c in enumerate(exp_codes):
                r = np.flatnonzero(cd == c)
                d0 = np.sqrt(((xy[r, None, :] - xy[None, r, :]) ** 2).sum(-1))
                a = p2[r]
                lo, hi = a <= qq1, a >= qq2
                sp = float(lo.any() and hi.any())
                null_cl["sp"][b, k] = sp
                null_cl["rg"][b, k] = a.max() - a.min()
                null_cl["cross"][b, k] = _min_dist_median(d0, lo, hi) if sp else np.nan
                w, rd = is_w[r], is_r[r]
                wr = float(w.any() and rd.any()) if has_comp else np.nan
                null_cl["wr"][b, k] = wr
                null_cl["r2w"][b, k] = _min_dist_median(d0, rd, w) if has_comp and wr else np.nan
                null_cl["spread"][b, k] = np.median(d0[np.triu_indices(len(r), 1)])
        n_valid[:] = n_perm
        design = {
            "design": "Permute",
            "columns": [ck],
            "within": within_,
            "refit": "Threadfin clone profiles re-fitted per permutation",
        }

    # per-clone p-values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        T["n_valid"] = n_valid
        T["spanning_null"] = np.nanmean(null_cl["sp"], axis=0)
        T["axis_range_null"] = np.nanmean(null_cl["rg"], axis=0)
        T["cross_um_null"] = np.nanmean(null_cl["cross"], axis=0)
        T["median_pairwise_um_null"] = np.nanmean(null_cl["spread"], axis=0)
    if null == "relocate":
        T["p_range"] = mc_pvalue(null_cl["rg"], T.axis_range.to_numpy(), alternative="less")
        T["p_cross"] = mc_pvalue(null_cl["cross"], T.cross_um.to_numpy(), alternative="less")
        T.loc[T.n_valid == 0, ["p_range", "p_cross"]] = np.nan
        T["q_range"] = bh(T.p_range.to_numpy())
        T["q_cross"] = bh(T.p_cross.to_numpy())
    else:
        T["p_spread"] = mc_pvalue(null_cl["spread"], T.median_pairwise_um.to_numpy(), alternative="less")
        T["p_range"] = mc_pvalue(null_cl["rg"], T.axis_range.to_numpy(), alternative="less")
        T["q_spread"] = bh(T.p_spread.to_numpy())
        T["q_range"] = bh(T.p_range.to_numpy())
    # aggregate null: per draw over clones with a valid draw
    null_agg = np.vstack(
        [
            aggregates(
                null_cl["sp"][b],
                null_cl["rg"][b],
                null_cl["cross"][b],
                null_cl["wr"][b],
                null_cl["r2w"][b],
                null_cl["spread"][b],
            )
            for b in range(n_perm)
        ]
    )
    mu, sd, lo95, hi95 = null_summary(null_agg)
    p_agg = mc_pvalue(null_agg, obs_agg, alternative="two-sided")
    summ: dict[str, Any] = {
        "axis": int(axis),
        "null": null,
        "n_expanded": int(n_exp),
        "n_spanning": int(T.spanning.sum()) if n_exp else 0,
        "tertile_cuts": [float(q1), float(q2)],
        "n_clones_without_valid_relocation": int((n_valid == 0).sum()),
    }
    for i, nm in enumerate(agg_names):
        summ[nm] = float(obs_agg[i])
        summ[f"{nm}_null"] = float(mu[i])
        summ[f"{nm}_null_lo"] = float(lo95[i])
        summ[f"{nm}_null_hi"] = float(hi95[i])
        summ[f"p[{nm}]"] = float(p_agg[i])
    params = {
        "axis": int(axis),
        "min_units": int(min_units),
        "null": null,
        "design": design,
        "n_perm": int(n_perm),
        "compartment_key": compartment_key if has_comp else None,
        "white": list(white),
        "red": list(red),
        "rng_entropy": rng_entropy(gen),
        "aggregates": agg_names,
    }
    return TestResult(name="spanning_clones", table=T, summary=summ, params=params, null=null_agg)


def region_axis(
    result: ThreadfinSpatialResult,
    regions: Any = None,
    *,
    axis: int = 1,
    n_perm: int = 199,
    snap_um: float = 20.0,
    max_tries: int = 50,
    rng: RngLike = None,
) -> TestResult:
    """Where regions sit on a clone axis: their units' positions and the values of the clones they hold.

    For every region instance (and all instances pooled): ``unit_percentile`` = median percentile of the positions of
    the units inside it among all units, and ``clone_percentile`` = median percentile, among profiled clones, of the
    values of the profiled clones with at least one unit inside it.

    Parameters
    ----------
    result
        :func:`run` result.
    regions
        A ``result.units`` column of instance labels (0 / NaN / empty = none; e.g. ``"csr_shm_overlap_id"`` from a
        context region), a :class:`spaclone.tl.Raster` (mask or labels), or a
        :class:`spaclone.bcr.HotspotWorkflowResult` (its ``region_labels``). ``None``: the first ``*_id`` region
        column of ``result.units``.
    axis
        Clone-profile axis.
    n_perm
        Relocations per region.
    snap_um, max_tries
        Rigid relocation settings (see :func:`spanning_clones`).
    rng
        Seed or generator.

    Returns
    -------
    TestResult
        ``table``: one row per (region, statistic) with ``statistic``, ``expected`` (null mean), ``null_sd``, ``z``,
        ``p_value`` (two-sided), ``q_value`` (BH over instances, per statistic), ``n_perm`` (valid relocations),
        ``n_units``.

    Notes
    -----
    Null model: each region instance's footprint (the positions of the units inside it) is rotated by a uniform
    angle, reflected with probability 1/2 and translated so that its centroid falls on a random unit; footprint
    points land on the nearest unit within ``snap_um`` (copies with a point that finds none are redrawn). Region
    shape and size are kept; the units and clones under the copy give the null values. The pooled row relocates
    every instance independently in each draw.

    References
    ----------
    Alexander-Bloch et al. 2018, doi:10.1016/j.neuroimage.2018.05.070; Gel et al. 2016,
    doi:10.1093/bioinformatics/btv562; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    _check_result(result, engine=False)
    j = _axis_index(result, axis)
    U = result.units
    xy = U[[S.X, S.Y]].to_numpy(dtype=float)
    if regions is None:
        cols = [c for c in U.columns if c.endswith("_id") and U[c].dtype.kind in "iu"]
        if not cols:
            raise ValueError("no region column in result.units; pass regions=")
        regions = cols[0]
    if isinstance(regions, str):
        if regions not in U.columns:
            raise KeyError(f"region column {regions!r} not in result.units")
        v = U[regions]
        lab = (pd.factorize(v.where(~v.isin([0, "0", "", "none"])), use_na_sentinel=True)[0] + 1).astype(np.int64)
        names = {
            k + 1: str(n)
            for k, n in enumerate(pd.factorize(v.where(~v.isin([0, "0", "", "none"])), use_na_sentinel=True)[1])
        }
        rname = regions
    else:
        rast = _as_label_raster(regions, None, None, "spatial")
        lab = rast.sample(xy[:, 0], xy[:, 1], fill=0).astype(np.int64)
        names = {int(k): str(int(k)) for k in np.unique(lab[lab > 0])}
        rname = "regions"
    inst = sorted(int(k) for k in np.unique(lab[lab > 0]))
    if not inst:
        raise ValueError("no unit lies inside a region")
    gen = as_rng(rng)
    codes, _ = _codes(result)
    fit = _table_fit(result, codes)
    pos = fit.positions[:, j]
    pct_u = rankdata(pos) / len(pos)
    cval = fit.values[:, j]
    pct_c = rankdata(cval) / len(cval)
    row_of_code = np.full(len(fit.sizes), -1, dtype=np.int64)
    row_of_code[fit.clone_rows] = np.arange(len(fit.clone_rows))
    prof_row = np.where(codes >= 0, row_of_code[np.maximum(codes, 0)], -1)

    def stats(units_idx: np.ndarray) -> tuple[float, float]:
        if len(units_idx) == 0:
            return np.nan, np.nan
        cl = np.unique(prof_row[units_idx])
        cl = cl[cl >= 0]
        return float(np.median(pct_u[units_idx])), (float(np.median(pct_c[cl])) if len(cl) else np.nan)

    tree = cKDTree(xy)
    obs = {}
    nulls_ = {}
    valid = {}
    draws = {}
    for r in inst:
        idx_r = np.flatnonzero(lab == r)
        obs[r] = stats(idx_r)
        land, got = _rigid_landings(xy[idx_r], xy, tree, n_draws=n_perm, snap_um=snap_um, max_tries=max_tries, gen=gen)
        draws[r], valid[r] = land, got
        nulls_[r] = np.array([stats(np.unique(land[b])) for b in range(got)] + [(np.nan, np.nan)] * (n_perm - got))
    rows_ = []
    all_idx = np.flatnonzero(lab > 0)
    obs_pool = stats(all_idx)
    pool_null = []
    for b in range(n_perm):
        parts = [draws[r][b] for r in inst if b < valid[r]]
        pool_null.append(stats(np.unique(np.concatenate(parts))) if parts else (np.nan, np.nan))
    pool_null = np.array(pool_null)
    for key, o, nl, n_u, nv in [
        (names.get(r, str(r)), obs[r], nulls_[r], int((lab == r).sum()), valid[r]) for r in inst
    ] + [("all", obs_pool, pool_null, int(len(all_idx)), n_perm)]:
        mu, sd, lo, hi = null_summary(nl)
        p = mc_pvalue(nl, np.asarray(o), alternative="two-sided")
        for i, st in enumerate(("unit_percentile", "clone_percentile")):
            rows_.append(
                {
                    "region": key,
                    "statistic_name": st,
                    "statistic": o[i],
                    "expected": mu[i],
                    "null_mean": mu[i],
                    "null_sd": sd[i],
                    "null_lo": lo[i],
                    "null_hi": hi[i],
                    "z": (o[i] - mu[i]) / sd[i] if sd[i] > 0 else np.nan,
                    "p_value": p[i],
                    "n_perm": int(nv),
                    "n_units": n_u,
                }
            )
    table = pd.DataFrame(rows_)
    table["q_value"] = np.nan
    for st in ("unit_percentile", "clone_percentile"):
        sel = (table.statistic_name == st) & (table.region != "all")
        table.loc[sel, "q_value"] = bh(table.loc[sel, "p_value"].to_numpy())
    pooled = table[table.region == "all"].set_index("statistic_name")
    summ = {"axis": int(axis), "regions": rname, "n_instances": len(inst)}
    for st in ("unit_percentile", "clone_percentile"):
        summ[st] = float(pooled.loc[st, "statistic"])
        summ[f"{st}_null"] = float(pooled.loc[st, "expected"])
        summ[f"p[{st}]"] = float(pooled.loc[st, "p_value"])
    params = {
        "axis": int(axis),
        "regions": rname,
        "n_perm": int(n_perm),
        "snap_um": float(snap_um),
        "max_tries": int(max_tries),
        "design": "rigid relocation of each region instance",
        "alternative": "two-sided",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="region_axis", table=table, summary=summ, params=params)


def compare(
    a: ThreadfinSpatialResult,
    b: ThreadfinSpatialResult,
    *,
    n_axes: int | None = None,
    n_perm: int = 199,
    size_bins: Sequence[int] = (2, 3, 5, 10),
    rng: RngLike = None,
) -> TestResult:
    """How much of one run's clone landscape another run explains (e.g. expression profiles from niche profiles).

    Over the clones profiled in both runs: ``R2(a|b)`` is the share of the variance of ``a``'s clone profiles (all its
    stored profile components) explained by a least-squares linear map from ``b``'s profiles, and ``R2(b|a)`` the
    reverse; ``rho(a.axis_i, b.axis_k)`` are Spearman correlations of the clone axis values.

    Parameters
    ----------
    a, b
        Two :func:`run` results on the same clones (e.g. ``features="expression"`` and ``features="niche"``).
    n_axes
        Axes compared pairwise (default: the smaller number kept).
    n_perm
        Permutations.
    size_bins
        Lower edges of the clone-size strata of the permutation (units: 2, 3–4, 5–9, ≥ 10, as Threadfin's label
        tests).
    rng
        Seed or generator.

    Returns
    -------
    TestResult
        ``table``: one row per statistic with ``statistic``, ``expected`` (null mean), ``null_sd``, ``z``,
        ``p_value`` (R²: greater; correlations: two-sided), ``q_value``, ``n_perm``. ``summary``: R² both ways with
        their null means and p-values and ``rho_axis_1``.

    Notes
    -----
    Null model: ``b``'s clone profiles are permuted among clones of the same size stratum (a clone's profile is
    shrunk according to its size in both runs, so sizes stay matched); ``a`` is fixed. Under this null the R² is
    about the number of components over the number of clones.

    References
    ----------
    Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Anderson 2001, doi:10.1111/j.1442-9993.2001.01070.pp.x
    (permutation of multivariate profiles).
    """
    _check_result(a)
    _check_result(b)
    fa = a.threadfin.adata.uns["threadfin"]["profiles"]["features"]
    fb = b.threadfin.adata.uns["threadfin"]["profiles"]["features"]
    shared = a.clones.index.intersection(b.clones.index)
    if len(shared) < 10:
        raise ValueError(f"only {len(shared)} clones are profiled in both runs")
    A = np.array(fa.reindex(shared.astype(str)), dtype=float)
    B = np.array(fb.reindex(shared.astype(str)), dtype=float)
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    na = int(n_axes or min(a.params["n_axes"], b.params["n_axes"]))
    xa = a.clones.loc[shared, [f"axis_{i + 1}" for i in range(na)]].to_numpy(dtype=float)
    xb = b.clones.loc[shared, [f"axis_{i + 1}" for i in range(na)]].to_numpy(dtype=float)
    size = a.clones.loc[shared, "n_units"].to_numpy()
    edges = np.asarray(sorted(size_bins), dtype=float)
    stratum = np.searchsorted(edges, size, side="right")
    gen = as_rng(rng)

    def r2(y: np.ndarray, x: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        res = y - x @ coef
        return float(1.0 - (res**2).sum() / (y**2).sum())

    def stats(perm: np.ndarray) -> np.ndarray:
        Bp, xbp = B[perm], xb[perm]
        out = [r2(A, Bp), r2(Bp, A)]
        out += [_spearman(xa[:, i], xbp[:, k]) for i in range(na) for k in range(na)]
        return np.array(out)

    names = ["R2(a|b)", "R2(b|a)"] + [f"rho(a.axis_{i + 1}, b.axis_{k + 1})" for i in range(na) for k in range(na)]
    ident = np.arange(len(shared))
    obs = stats(ident)
    groups = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
    nulls_ = np.empty((n_perm, len(obs)))
    for t in range(n_perm):
        perm = ident.copy()
        for g in groups:
            perm[g] = g[gen.permutation(len(g))]
        nulls_[t] = stats(perm)
    mu, sd, lo, hi = null_summary(nulls_)
    p = np.r_[
        mc_pvalue(nulls_[:, :2], obs[:2], alternative="greater"),
        mc_pvalue(nulls_[:, 2:], obs[2:], alternative="two-sided"),
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - mu) / np.where(sd > 0, sd, np.nan)
    table = pd.DataFrame(
        {
            "statistic_name": names,
            "statistic": obs,
            "expected": mu,
            "null_mean": mu,
            "null_sd": sd,
            "null_lo": lo,
            "null_hi": hi,
            "z": z,
            "p_value": p,
            "n_perm": n_perm,
            "n_clones": len(shared),
        }
    )
    table["q_value"] = bh(table.p_value.to_numpy())
    summ = {
        "n_clones": int(len(shared)),
        "R2(a|b)": float(obs[0]),
        "R2(a|b)_null": float(mu[0]),
        "p[R2(a|b)]": float(p[0]),
        "R2(b|a)": float(obs[1]),
        "R2(b|a)_null": float(mu[1]),
        "p[R2(b|a)]": float(p[1]),
        "rho_axis_1": float(obs[2]),
    }
    params = {
        "a": f"{a.params['units']}/{a.params['features']}",
        "b": f"{b.params['units']}/{b.params['features']}",
        "n_axes": na,
        "n_perm": int(n_perm),
        "size_bins": list(size_bins),
        "design": "b's clone profiles permuted among clones of the same size stratum",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="compare", table=table, summary=summ, params=params, null=nulls_)
