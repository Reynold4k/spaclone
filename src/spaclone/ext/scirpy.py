"""scirpy on spatial units: clonotypes, expansion, diversity, overlap, modularity and V(D)J usage, mapped back to tissue.

scirpy (Sturm et al. 2020) works on one observation per cell. In spatial receptor data the observation is a spatial
unit, built by :func:`spaclone.io.to_scirpy`: an inferred cell (``level="locus"``: same-clone spots within
``radius_um`` merged), a spot (``"spot"``) or a segmented cell (``"cell"``). Every function here runs scirpy's own
function on that AnnData, unchanged, and returns its output with the spatial counterpart:

* ``result.units``: one row per unit with the new labels and its position (µm), ready for :mod:`spaclone.tl` (share
  maps, hotspots) and :func:`spaclone.pl.spatial_records`;
* ``result.air``: the receptor table with the unit's labels on every record;
* ``result.table``: the per-clonotype, per-region or per-pair table;
* ``result.plot()``: the labels on the tissue, through :mod:`spaclone.pl`.

=====================================  ==================================================  ===========================
spaclone.ext.scirpy                    scirpy                                              spatial output
=====================================  ==================================================  ===========================
:func:`clonotypes`                     ``pp.ir_dist`` + ``tl.define_clonotype_clusters``  clonotype per unit / record,
                                                                                           centroid and spread
:func:`clonal_expansion`               ``tl.clonal_expansion``                             size class per unit and
                                                                                           per region
:func:`expansion_map`                  ``tl.clonal_expansion``                             share map of expanded units
:func:`diversity_by_region`            ``tl.alpha_diversity``                              one value per region
:func:`repertoire_overlap`             ``tl.repertoire_overlap``                           region × region distances
:func:`clonotype_modularity`           ``tl.clonotype_modularity``                         on the spatial neighbour
                                                                                           graph of the units
:func:`vdj_usage`                      ``get.airr`` + ``tl.group_abundance``               gene shares per region
=====================================  ==================================================  ===========================

Identical chains of one unit (the same locus, V gene, J gene and junction: several molecules, contigs or spots of one
cell) count once in scirpy's receptor model (:func:`prepare`, ``collapse=True``): the copy with most reads is
indexed and the others stay in ``obsm["airr"]``. Without this, a locus holding three molecules of one heavy chain is
"multichain" and is kept apart from every other unit by scirpy's default ``within_group="receptor_type"``.

B-cell clonotypes and clones are heavy-chain ones by default (``receptor_arms="auto"``: ``"VDJ"`` when the units hold
B-cell chains, scirpy's ``"all"`` otherwise). Light-chain junctions (IGK, IGL) are shared by unrelated B cells: with
both arms, the light-chain loci of a spleen form clonotypes of thousands of loci spread over the whole section, and
most loci look "expanded". Every function that groups units by clonotype or clone warns
(:class:`LargeClonotypeWarning`) when one of them holds more than 5% of the units (and at least 50 units), and names
the argument to change.

Maps (``result.plot()``) follow the convention of :mod:`spaclone.pl`: the section rotated about its centre so that
its long axis is horizontal (``rotate=True``) and y increasing downwards (``invert_y=True``).

These are descriptions. Raw diversity and expansion depend on how many units a region holds, and scirpy's nulls
(clonotype modularity) relabel units at random. For calibrated comparisons use :func:`spaclone.tl.diversity`
(rarefied, random placement of intact clones), :func:`spaclone.tl.hotspots` and :func:`spaclone.qc.colocalisation`.

scirpy, awkward and mudata are optional: ``pip install 'scirpy>=0.20' 'awkward>=2' 'mudata>=0.3'`` (the
``spaclone[scirpy]`` extra).

References
----------
Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy); Vander Heiden et al. 2018,
doi:10.4049/jimmunol.1800669 (AIRR Rearrangement schema).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, bh, rng_entropy

__all__ = [
    "ExpansionMap",
    "LargeClonotypeWarning",
    "ModularityResult",
    "ScirpyResult",
    "clonal_expansion",
    "clonotype_modularity",
    "clonotypes",
    "diversity_by_region",
    "expansion_map",
    "prepare",
    "region_table",
    "repertoire_overlap",
    "to_records",
    "to_units",
    "vdj_usage",
]

Level = Literal["locus", "spot", "cell"]
_ROW = "_spaclone_row"
_GRAPH = "spaclone_spatial_connectivities"
#: One clonotype (or clone) holding more than this share of the units with one raises :class:`LargeClonotypeWarning`...
LARGE_CLONOTYPE_SHARE = 0.05
#: ... when it also holds at least this many units (in a small table one expanded clone passes 5%).
LARGE_CLONOTYPE_MIN_SIZE = 50
Arms = Literal["auto", "VJ", "VDJ", "all", "any"]
UnitArms = Literal["auto", "VJ", "VDJ", "all"]
_CLONOTYPE_ARMS = ("auto", "VJ", "VDJ", "all", "any")
_UNIT_ARMS = ("auto", "VJ", "VDJ", "all")


class LargeClonotypeWarning(SpacloneWarning):
    """One clonotype or clone holds more than 5% of the units: light chains or a loose definition merge cells."""


def _ir():
    try:
        import scirpy as ir
    except ImportError as e:
        raise ImportError(
            "spaclone.ext.scirpy needs scirpy: pip install 'scirpy>=0.20' 'awkward>=2' 'mudata>=0.3' "
            "(the spaclone[scirpy] extra)"
        ) from e
    return ir


def _ak():
    try:
        import awkward as ak
    except ImportError as e:
        raise ImportError("spaclone.ext.scirpy needs awkward: pip install 'awkward>=2' (spaclone[scirpy])") from e
    return ak


def _is_adata(x: Any) -> bool:
    return hasattr(x, "obs") and hasattr(x, "obsm") and hasattr(x, "n_obs")


def _flat(adata: Any, airr_key: str):
    """(flattened chains, obs position of every chain, chains per obs)."""
    ak = _ak()
    arr = adata.obsm[airr_key]
    counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
    obs = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return ak.flatten(arr, axis=1), obs, counts


def _field(flat: Any, name: str) -> pd.Series:
    ak = _ak()
    a = ak.to_arrow(flat[name], extensionarray=False)
    if hasattr(a, "combine_chunks"):
        a = a.combine_chunks()
    return pd.Series(a.to_pandas(), copy=False)


def _distinct_chains(fields: Sequence[str] = (S.LOCUS, S.V_CALL, S.J_CALL, S.JUNCTION, S.JUNCTION_AA)) -> Callable:
    """A scirpy chain filter that keeps one copy (the most-read one) of identical chains within each observation."""

    def keep_distinct(airr: Any) -> Any:
        from spaclone.pp.clones import gene_name

        ak = _ak()
        counts = ak.to_numpy(ak.num(airr, axis=1)).astype(np.int64)
        flat = ak.flatten(airr, axis=1)
        n = len(flat)
        names = set(ak.fields(flat))
        cols: dict[str, Any] = {"_o": np.repeat(np.arange(len(counts), dtype=np.int64), counts)}
        key = []
        for f in fields:
            if f not in names:
                continue
            v = _field(flat, f).astype(object)
            if f in (S.V_CALL, S.J_CALL):
                v = gene_name(v).astype(object)
            cols[f] = v.where(v.notna(), "").astype(str).to_numpy()
            key.append(f)
        if S.JUNCTION in key and S.JUNCTION_AA in key:
            key.remove(S.JUNCTION_AA)  # identical nucleotides imply identical amino acids
        reads = np.zeros(n)
        for r in ("umi_count", "duplicate_count", S.READS):
            if r in names:
                reads = np.nan_to_num(_field(flat, r).to_numpy(dtype=float, na_value=np.nan), nan=0.0)
                break
        cols["_r"] = -reads
        cols["_i"] = np.arange(n, dtype=np.int64)
        df = pd.DataFrame(cols).sort_values(["_o", "_r", "_i"], kind="stable")
        mask = np.zeros(n, dtype=bool)
        mask[df.drop_duplicates(["_o", *key])["_i"].to_numpy()] = True
        return ak.unflatten(mask, counts)

    return keep_distinct


def _mode_per_obs(obs: np.ndarray, values: pd.Series, n_obs: int) -> pd.Series:
    """Most common non-missing value per observation (ties: the first in sorted order); missing when none."""
    df = pd.DataFrame({"o": obs, "v": values.to_numpy(dtype=object)})
    df = df[pd.notna(df["v"])]
    out = pd.Series([None] * n_obs, dtype=object)
    if len(df):
        cnt = df.groupby(["o", "v"], sort=True).size().reset_index(name="n")
        cnt = cnt.sort_values(["o", "n"], ascending=[True, False], kind="stable").drop_duplicates("o")
        out.iloc[cnt["o"].to_numpy()] = cnt["v"].to_numpy(dtype=object)
    return out


def _carry(adata: Any, columns: Sequence[str], airr_key: str) -> None:
    """Copy record columns to ``obs``: the most common value of each unit's records."""
    need = [c for c in dict.fromkeys(columns) if c and c not in adata.obs.columns]
    if not need:
        return
    ak = _ak()
    flat, obs, _ = _flat(adata, airr_key)
    names = set(ak.fields(flat))
    missing = [c for c in need if c not in names]
    if missing:
        raise KeyError(f"column(s) {missing} are neither in obs nor in the receptor records")
    for c in need:
        vals = _field(flat, c)
        m = _mode_per_obs(obs, vals, adata.n_obs)
        num = pd.to_numeric(m, errors="coerce")
        if m.notna().any() and num[m.notna()].notna().all() and not pd.api.types.is_bool_dtype(vals.dtype):
            if pd.api.types.is_integer_dtype(vals.dtype):  # integer ids (region_id, ...) stay integers
                adata.obs[c] = num.astype("Int64").to_numpy() if m.isna().any() else num.to_numpy(dtype=np.int64)
            else:
                adata.obs[c] = num.to_numpy(dtype=float)
        else:
            adata.obs[c] = pd.Categorical(m.where(m.notna(), None).to_numpy(dtype=object))


def prepare(
    data: Any,
    *,
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    index_chains: bool = True,
    chain_qc: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
):
    """Spatial units as a scirpy AnnData, with scirpy's chain indices and chain QC.

    Parameters
    ----------
    data
        Receptor table, :class:`~spaclone.io.SpatialAIR`, or an AnnData from :func:`spaclone.io.to_scirpy` (used in
        place, as scirpy functions do).
    level
        Observation unit of :func:`spaclone.io.to_scirpy`: ``"locus"`` (inferred cell, one clone each; needs
        ``clone_key``), ``"spot"`` or ``"cell"`` (``cell_key``).
    radius_um
        Locus radius in µm.
    carry
        Record columns copied to ``obs`` (region, section, cell type, ...): the most common value among the unit's
        records (ties: the first in sorted order).
    collapse
        Count identical chains of a unit once (same locus, V gene, J gene and junction; the copy with most reads is
        indexed), through scirpy's ``index_chains(filter=...)``. Filtered chains stay in ``obsm[airr_key]`` and do not
        make a unit multichain.
    index_chains, chain_qc
        Run ``scirpy.pp.index_chains`` and ``scirpy.tl.chain_qc`` when their output is not there yet.
    airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Keys passed to :func:`spaclone.io.to_scirpy`.

    Returns
    -------
    anndata.AnnData
        scirpy's data structure: ``obsm[airr_key]`` (chains), ``obsm["spatial"]`` (µm), ``obsm["chain_indices"]``,
        ``obs["receptor_type"]``, ``obs["chain_pairing"]`` and the ``carry`` columns.

    Raises
    ------
    ImportError
        If scirpy or awkward is not installed.
    """
    ir = _ir()
    from spaclone.io._container import SpatialAIR
    from spaclone.io.interop import to_scirpy

    if isinstance(data, SpatialAIR):
        data = data.air
    if isinstance(data, pd.DataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Awkward Arrays.*")
            adata = to_scirpy(
                data,
                level=level,
                radius_um=radius_um,
                cell_key=cell_key,
                airr_key=airr_key,
                spot_key=spot_key,
                x_key=x_key,
                y_key=y_key,
                clone_key=clone_key,
                locus_key=locus_key,
                library_key=library_key,
                productive_key=productive_key,
                reads_key=reads_key,
            )
        # the table's attrs travel into obs; a non-scalar entry (a DataFrame) breaks pandas' concat inside scirpy.
        # from_scirpy restores them from uns["spaclone"]["to_scirpy"].
        adata.obs.attrs = {}
    elif _is_adata(data):
        adata = data
        if airr_key not in adata.obsm:
            raise KeyError(f"prepare: the AnnData has no obsm[{airr_key!r}] (build it with spaclone.io.to_scirpy)")
    else:
        raise TypeError(f"expected a receptor table, a SpatialAIR or a scirpy AnnData, got {type(data).__name__}")
    if "spatial" not in adata.obsm:
        raise KeyError("prepare: the AnnData has no obsm['spatial'] (positions in µm)")
    _carry(adata, [carry] if isinstance(carry, str) else list(carry), airr_key)
    if index_chains and "chain_indices" not in adata.obsm:
        filt: list[Any] = ["productive", "require_junction_aa"]
        if collapse:
            filt.append(_distinct_chains())
        ir.pp.index_chains(adata, airr_key=airr_key, filter=filt)
        adata.uns.setdefault("spaclone", {})["collapse_identical_chains"] = bool(collapse)
    if chain_qc and "receptor_type" not in adata.obs.columns:
        ir.tl.chain_qc(adata, airr_key=airr_key)
    return adata


def _record_obs(adata: Any, airr_key: str) -> np.ndarray:
    """Observation position of every record, in the row order of :func:`spaclone.io.from_scirpy`."""
    ak = _ak()
    flat, obs, _ = _flat(adata, airr_key)
    meta = adata.uns.get("spaclone", {}).get("to_scirpy") if isinstance(adata.uns.get("spaclone"), dict) else None
    if meta is not None and _ROW in ak.fields(flat):
        pos = ak.to_numpy(flat[_ROW]).astype(np.int64)
        return obs[np.argsort(pos, kind="stable")]
    return obs


def to_units(
    adata: Any, columns: Sequence[str] | None = None, *, x_key: str = S.X, y_key: str = S.Y, unit_key: str = "unit_id"
) -> pd.DataFrame:
    """One row per unit: ``unit_key`` (the observation name), ``obs`` columns and the position in µm.

    Parameters
    ----------
    adata
        scirpy AnnData with ``obsm["spatial"]``.
    columns
        ``obs`` columns to keep (default all).
    x_key, y_key, unit_key
        Output column names.

    Returns
    -------
    pandas.DataFrame
        A plain table (RangeIndex) that :mod:`spaclone.tl` and :func:`spaclone.pl.spatial_records` accept.
    """
    obs = adata.obs if columns is None else adata.obs[[c for c in columns if c in adata.obs.columns]]
    u = obs.copy()
    u.insert(0, unit_key, np.asarray(adata.obs_names, dtype=object))
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    u[x_key] = xy[:, 0]
    u[y_key] = xy[:, 1]
    return u.reset_index(drop=True)


def to_records(
    adata: Any, columns: Sequence[str] | None = None, *, airr_key: str = "airr", unit_key: str = "scirpy_unit"
) -> pd.DataFrame:
    """The receptor table with unit labels on every record.

    Parameters
    ----------
    adata
        scirpy AnnData (from :func:`spaclone.io.to_scirpy`, or any scirpy AnnData with ``obsm["spatial"]``).
    columns
        ``obs`` columns to copy to the records (default: every ``obs`` column that is not already a record column).
        A requested column that is also a record column is overwritten.
    airr_key
        ``obsm`` key of the chains.
    unit_key
        Column naming the unit (observation) of each record.

    Returns
    -------
    pandas.DataFrame
        :func:`spaclone.io.from_scirpy` (the original table, restored exactly, when it came from
        :func:`~spaclone.io.to_scirpy`) plus ``unit_key`` and the label columns.
    """
    from spaclone.io.interop import from_scirpy

    air = from_scirpy(adata, airr_key=airr_key, require_spatial=False).copy()
    o = _record_obs(adata, airr_key)
    air[unit_key] = np.asarray(adata.obs_names, dtype=object)[o]
    cols = [c for c in adata.obs.columns if c not in air.columns] if columns is None else list(columns)
    for c in cols:
        s = adata.obs[c]
        vals = s.iloc[o]
        air[c] = pd.Series(vals.to_numpy(), index=air.index, dtype=s.dtype)
    return air


@dataclass(frozen=True, repr=False)
class ScirpyResult:
    """Output of a scirpy function run on spatial units.

    Attributes
    ----------
    name
        The scirpy function.
    adata
        The scirpy AnnData; ``obs`` carries the new columns, ``obsm["spatial"]`` the positions (µm).
    columns
        The ``obs`` columns this call added.
    table
        The tidy result table (per clonotype, region, pair, ...).
    summary
        Headline numbers.
    params
        Arguments of the call and the scirpy version.

    Notes
    -----
    ``units`` (one row per unit with position) and ``air`` (records with the unit labels) are computed on first use.
    """

    name: str
    adata: Any
    columns: tuple[str, ...]
    table: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @functools.cached_property
    def units(self) -> pd.DataFrame:
        """One row per unit: name, ``obs`` columns, ``x_um``, ``y_um``."""
        return to_units(self.adata)

    @functools.cached_property
    def air(self) -> pd.DataFrame:
        """The receptor table with this call's labels on every record."""
        return to_records(self.adata, list(self.columns), airr_key=self.params.get("airr_key", "airr"))

    def plot(
        self,
        color: str | None = None,
        *,
        highlight: Sequence[Any] | None = None,
        ax: Any = None,
        style: Literal["light", "dark"] = "dark",
        rotate: bool | float = True,
        invert_y: bool = True,
        **kwargs: Any,
    ):
        """Draw the units at their positions, coloured by a label (default: this call's first column).

        Parameters
        ----------
        color
            ``obs`` column (categorical; see :func:`spaclone.pl.spatial_records`).
        highlight
            Values of ``color`` drawn in colour over grey.
        ax
            Axes to draw on.
        style
            ``"dark"`` (the black house style of every spaclone map, default) or ``"light"``.
        rotate
            ``True`` (the convention of spaclone's maps): the section rotated about its centre so that its long axis
            is horizontal (round sections keep their orientation); an angle in degrees; ``False``: the original
            coordinates.
        invert_y
            Draw y increasing downwards (image convention, the default of every spaclone map).
        **kwargs
            Passed to :func:`spaclone.pl.spatial_records` (e.g. ``crop=False`` to frame every unit, not only the
            main tissue).

        Returns
        -------
        matplotlib.axes.Axes
        """
        col = color or (self.columns[0] if self.columns else None)
        return _plot_units(
            self.units, col, highlight=highlight, ax=ax, rotate=rotate, invert_y=invert_y, style=style, **kwargs
        )

    def __repr__(self) -> str:
        head = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.summary.items())
        return (
            f"ScirpyResult({self.name}: {head}; {self.adata.n_obs} units; columns {list(self.columns)}; "
            f"table {self.table.shape[0]}×{self.table.shape[1]})"
        )


@dataclass(frozen=True, repr=False)
class ExpansionMap(ScirpyResult):
    """:class:`ScirpyResult` of :func:`expansion_map` plus the share map of expanded units (``share_map``)."""

    share_map: Any = None

    def plot(
        self,
        color: str | None = None,
        *,
        kind: Literal["share", "units"] = "share",
        ax: Any = None,
        rotate: bool | float = True,
        invert_y: bool = True,
        **kwargs: Any,
    ):
        """Draw the share map of expanded units (``kind="share"``) or the unit labels (``"units"``).

        Parameters
        ----------
        color
            Label column for ``kind="units"``.
        kind
            ``"share"``: the smoothed share of expanded units on the eligible pixels (the rest of the section in the
            tissue colour); ``"units"``: the units coloured by a label.
        ax
            Axes to draw on.
        rotate
            ``True`` (the convention of spaclone's maps): the section rotated about its centre so that its long axis
            is horizontal; an angle in degrees; ``False``: the original coordinates.
        invert_y
            Draw y increasing downwards (image convention).
        **kwargs
            Passed to :func:`spaclone.pl.share_map` (``kind="share"``: ``layer``, ``style``, ``cmap``, ``vmin``,
            ``vmax``, ...) or to :func:`spaclone.pl.spatial_records` (``kind="units"``).

        Returns
        -------
        matplotlib.axes.Axes
        """
        if kind == "units":
            return super().plot(color, ax=ax, rotate=rotate, invert_y=invert_y, **kwargs)
        if kind != "share":
            raise ValueError(f"kind must be 'share' or 'units', got {kind!r}")
        from spaclone.pl.maps import share_map

        return share_map(self.share_map, ax=ax, rotate=rotate, invert_y=invert_y, **kwargs)


@dataclass(frozen=True, repr=False)
class ModularityResult(TestResult):
    """:class:`~spaclone.TestResult` of :func:`clonotype_modularity`; ``adata`` holds the per-unit scores."""

    adata: Any = None

    @functools.cached_property
    def units(self) -> pd.DataFrame:
        """One row per unit with its clonotype's modularity, p-value and q-value."""
        return to_units(self.adata)

    def plot(
        self,
        color: str | None = None,
        *,
        highlight: Sequence[Any] | None = None,
        ax: Any = None,
        rotate: bool | float = True,
        invert_y: bool = True,
        **kwargs: Any,
    ):
        """Draw the units, by default coloured by clonotype with the clonotypes at q < 0.05 highlighted.

        Parameters
        ----------
        color
            ``obs`` column (default: the clonotype column).
        highlight
            Values to highlight (default: clonotypes with q < 0.05, at most 8).
        ax
            Axes to draw on.
        rotate, invert_y
            The map convention of :meth:`ScirpyResult.plot` (long axis horizontal, y downwards).
        **kwargs
            Passed to :func:`spaclone.pl.spatial_records`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        col = color or self.params["target_col"]
        if highlight is None and color is None:
            sig = self.table[self.table["q_value"] < 0.05].sort_values("statistic", ascending=False)
            highlight = sig["clonotype"].astype(str).head(8).tolist() or None
        return _plot_units(self.units, col, highlight=highlight, ax=ax, rotate=rotate, invert_y=invert_y, **kwargs)


def _params(ir: Any, **kw: Any) -> dict[str, Any]:
    return {**kw, "scirpy_version": getattr(ir, "__version__", None)}


def _need_obs(adata: Any, col: str, what: str) -> None:
    if col not in adata.obs.columns:
        hint = (
            "build loci with a clone column (level='locus'), run spaclone.ext.scirpy.clonotypes first, or pass a "
            "record column in carry="
        )
        raise KeyError(f"{what}: obs has no column {col!r}; {hint}")


def _prep(data: Any, kw: dict[str, Any], extra_carry: Sequence[str | None] = ()) -> Any:
    carry = kw.pop("carry", ())
    carry = [carry] if isinstance(carry, str) else list(carry)
    carry += [c for c in extra_carry if c]
    return prepare(data, carry=carry, **kw)


def _keys(**kw: Any) -> dict[str, Any]:
    return kw


def _plot_units(units: pd.DataFrame, color: str | None, *, rotate: bool | float, invert_y: bool, **kwargs: Any):
    """The units drawn by :func:`spaclone.pl.spatial_records`, in the house frame of spaclone's maps."""
    from spaclone.pl.records import spatial_records

    kwargs.setdefault("background", "none")
    return spatial_records(units, color=color, rotate=rotate, invert_y=invert_y, **kwargs)


def _chains(ir: Any, adata: Any, airr_key: str) -> pd.DataFrame:
    """Locus of every unit's primary VJ and VDJ chain (scirpy's chain indices; missing = no chain of that arm)."""
    out = pd.DataFrame(index=adata.obs_names)
    for arm in ("VJ", "VDJ"):
        try:
            v = ir.get.airr(adata, "locus", f"{arm}_1", airr_key=airr_key)
            out[arm] = np.asarray(v, dtype=object)
        except (KeyError, ValueError):  # no locus field: presence only
            v = np.asarray(ir.get.airr(adata, "junction_aa", f"{arm}_1", airr_key=airr_key), dtype=object)
            out[arm] = np.where(pd.notna(v), "", None)
    return out


def _resolve_arms(receptor_arms: str, ch: pd.DataFrame, allowed: Sequence[str]) -> str:
    """``"auto"``: ``"VDJ"`` when the units hold B-cell chains (heavy chains define B-cell clones), else ``"all"``."""
    if receptor_arms not in allowed:
        raise ValueError(f"receptor_arms must be one of {list(allowed)}, got {receptor_arms!r}")
    if receptor_arms != "auto":
        return receptor_arms
    loci = pd.unique(pd.concat([ch["VJ"], ch["VDJ"]]).dropna().astype(str).str.upper())
    return "VDJ" if np.isin(loci, S.BCR_LOCI).any() else "all"


def _arm_target(adata: Any, target_col: str, arms: str, ch: pd.DataFrame) -> tuple[str, int]:
    """``target_col`` kept on the units with a chain of ``arms`` (``"VJ"`` / ``"VDJ"``; ``"all"`` keeps every unit).

    Returns the ``obs`` column scirpy runs on (a temporary copy, missing on the other units, when units are left out)
    and the number of units with a value that are left out.
    """
    if arms not in ("VJ", "VDJ"):
        return target_col, 0
    keep = pd.notna(ch[arms]).to_numpy()
    has = adata.obs[target_col].notna().to_numpy()
    n_out = int((has & ~keep).sum())
    if n_out == 0:
        return target_col, 0
    tmp = f"_spaclone_{target_col}_{arms}"
    adata.obs[tmp] = adata.obs[target_col].where(keep)
    return tmp, n_out


def _warn_large(labels: pd.Series, ch: pd.DataFrame, *, what: str, arms: str, other: str, stacklevel: int = 3) -> None:
    """:class:`LargeClonotypeWarning` when one label holds more than :data:`LARGE_CLONOTYPE_SHARE` of the units."""
    lab = pd.Series(np.asarray(labels, dtype=object))
    ok = lab.notna().to_numpy()
    n = int(ok.sum())
    if n == 0:
        return
    vc = lab[ok].value_counts()
    top, n_top = vc.index[0], int(vc.iloc[0])
    if n_top <= LARGE_CLONOTYPE_SHARE * n or n_top < LARGE_CLONOTYPE_MIN_SIZE:
        return
    in_top = ok & (lab == top).to_numpy()
    light = float(pd.isna(ch["VDJ"].to_numpy()[in_top]).mean())
    if light > 0.5 and arms != "VDJ":
        fix = (
            f"{light:.0%} of its units carry only a VJ (light / alpha) chain, whose junctions unrelated cells share: "
            "pass receptor_arms='VDJ' (heavy / beta chains; the default for B cells)"
        )
    else:
        fix = other
    warnings.warn(
        f"{what}: {top!r} holds {n_top:,} of the {n:,} units with a value ({n_top / n:.0%}, receptor_arms used "
        f"{arms!r}); {fix}.",
        LargeClonotypeWarning,
        stacklevel=stacklevel,
    )


def _drop_tmp(adata: Any, tcol: str, target_col: str) -> None:
    if tcol != target_col and tcol in adata.obs.columns:
        del adata.obs[tcol]


def _no_nan_class(adata: Any, key: str) -> None:
    """Turn the ``"nan"`` expansion class of scirpy (units without a clone) into a missing value."""
    s = adata.obs[key]
    if isinstance(s.dtype, pd.CategoricalDtype) and "nan" in list(s.cat.categories):
        adata.obs[key] = s.cat.remove_categories("nan")


def _arms_used(ir: Any, adata: Any, receptor_arms: str, sequence: str, airr_key: str = "airr") -> str:
    """``receptor_arms`` restricted to the arm that exists when no unit has a chain of the other arm.

    scirpy (0.25) fails in ``define_clonotype_clusters`` with ``"all"`` or ``"any"`` when one arm is absent from every
    observation (e.g. heavy-chain-only spatial tables); an arm that is missing everywhere cannot change the result.
    """
    if receptor_arms not in ("all", "any"):
        return receptor_arms
    field = "junction_aa" if sequence == "aa" else "junction"
    present = []
    for arm in ("VJ", "VDJ"):
        try:
            if pd.Series(ir.get.airr(adata, field, f"{arm}_1", airr_key=airr_key)).notna().any():
                present.append(arm)
        except (KeyError, ValueError):
            continue
    return present[0] if len(present) == 1 else receptor_arms


def clonotypes(
    data: Any,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: str = "identity",
    cutoff: int | None = None,
    receptor_arms: Arms = "auto",
    dual_ir: Literal["primary_only", "all", "any"] = "primary_only",
    same_v_gene: bool = False,
    same_j_gene: bool = False,
    within_group: str | Sequence[str] | None = "receptor_type",
    partitions: Literal["connected", "leiden", "fastgreedy"] = "connected",
    resolution: float = 1.0,
    n_jobs: int = 1,
    key_added: str = "scirpy_clone_id",
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ScirpyResult:
    """Clonotype clusters of scirpy on spatial units (``pp.ir_dist`` + ``tl.define_clonotype_clusters``).

    Parameters
    ----------
    data
        Receptor table, :class:`~spaclone.io.SpatialAIR` or scirpy AnnData (see :func:`prepare`).
    sequence, metric, cutoff
        Passed to ``scirpy.pp.ir_dist`` and ``scirpy.tl.define_clonotype_clusters``: compare amino-acid (``"aa"``) or
        nucleotide junctions with ``"identity"``, ``"hamming"``, ``"levenshtein"``, ``"normalized_hamming"``,
        ``"alignment"``, ``"tcrdist"``, ... within ``cutoff``. ``sequence="nt", metric="identity"`` is scirpy's
        ``define_clonotypes``.
    receptor_arms
        Chains that define a clonotype, passed to ``scirpy.tl.define_clonotype_clusters``: ``"VDJ"`` (heavy / beta /
        delta), ``"VJ"`` (light / alpha / gamma), ``"all"`` (both must match), ``"any"``. Default ``"auto"``:
        ``"VDJ"`` when the units hold B-cell chains (IGH, IGK, IGL), scirpy's ``"all"`` otherwise. Light-chain
        junctions are shared by unrelated B cells, so light-chain units would form clonotypes of thousands of loci;
        with ``"VDJ"`` units without a heavy chain get no clonotype. When no unit has a chain of one arm,
        ``"all"`` and ``"any"`` are run as the arm that exists (the same result; scirpy 0.25 fails otherwise).
        ``params["receptor_arms_used"]`` records the arm run.
    dual_ir, same_v_gene, same_j_gene, within_group, partitions, resolution
        Passed to ``scirpy.tl.define_clonotype_clusters`` (``dual_ir`` default ``"primary_only"``: spots and loci can
        carry extra chains).
    n_jobs
        Worker processes for scirpy.
    key_added
        ``obs`` column of the clonotype (scirpy also writes ``{key_added}_size``).
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`). ``level="locus"`` needs a clone column (the loci are merged per clone,
        for example the exact clones of :func:`spaclone.pp.define_clones`); scirpy then groups loci into clonotype
        clusters.

    Returns
    -------
    ScirpyResult
        ``table``: one row per clonotype with ``clonotype``, ``n_units``, ``n_records``, ``n_clones`` (distinct
        ``clone_key`` values among its units, when present), ``n_sections``, ``x_um``, ``y_um`` (centroid of its
        units) and ``spread_um`` (root-mean-square distance of its units to the centroid); sorted by size. ``summary``:
        ``n_units``, ``n_clustered`` (units with a clonotype), ``n_clonotypes``, ``n_expanded``, ``largest`` and
        ``largest_share`` (of the clustered units). ``air`` / ``units`` carry ``key_added`` and ``{key_added}_size``.

    Warns
    -----
    LargeClonotypeWarning
        When one clonotype holds more than 5% of the clustered units (and at least 50): naming ``receptor_arms`` when
        light chains make it, else ``metric`` / ``cutoff``.

    References
    ----------
    Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611; Gupta et al. 2017, doi:10.4049/jimmunol.1601850.

    Examples
    --------
    >>> res = clonotypes(air, metric="hamming", cutoff=1)  # doctest: +SKIP
    >>> res.air["scirpy_clone_id"]  # doctest: +SKIP
    """
    ir = _ir()
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw)
    ch = _chains(ir, adata, airr_key)
    arms = _arms_used(ir, adata, _resolve_arms(receptor_arms, ch, _CLONOTYPE_ARMS), sequence, airr_key)
    ir.pp.ir_dist(adata, metric=metric, cutoff=cutoff, sequence=sequence, n_jobs=n_jobs, airr_key=airr_key)
    ir.tl.define_clonotype_clusters(
        adata,
        sequence=sequence,
        metric=metric,
        receptor_arms=arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
        same_j_gene=same_j_gene,
        within_group=within_group,
        key_added=key_added,
        partitions=partitions,
        resolution=resolution,
        n_jobs=n_jobs,
        airr_key=airr_key,
    )
    u = to_units(adata, x_key=x_key, y_key=y_key)
    u = u[u[key_added].notna()]
    g = u.groupby(key_added, observed=True, sort=False)
    tab = pd.DataFrame(
        {
            "n_units": g.size(),
            "n_records": g["n_records"].sum() if "n_records" in u.columns else np.nan,
            "x_um": g[x_key].mean(),
            "y_um": g[y_key].mean(),
        }
    )
    if clone_key in u.columns:
        tab["n_clones"] = g[clone_key].nunique()
    if library_key in u.columns:
        tab["n_sections"] = g[library_key].nunique()
    d2 = (u[x_key] - u[key_added].map(tab["x_um"]).astype(float)) ** 2 + (
        u[y_key] - u[key_added].map(tab["y_um"]).astype(float)
    ) ** 2
    tab["spread_um"] = np.sqrt(d2.groupby(u[key_added].astype(object)).mean()).reindex(tab.index).to_numpy()
    tab.index.name = "clonotype"
    tab = tab.reset_index().sort_values(["n_units", "clonotype"], ascending=[False, True], kind="stable")
    tab["clonotype"] = tab["clonotype"].astype(str)
    tab = tab.reset_index(drop=True)
    summary = {
        "n_units": int(adata.n_obs),
        "n_clustered": int(len(u)),
        "n_clonotypes": len(tab),
        "n_expanded": int((tab.n_units >= 2).sum()),
        "largest": int(tab.n_units.max()) if len(tab) else 0,
        "largest_share": float(tab.n_units.max() / len(u)) if len(tab) else 0.0,
    }
    _warn_large(
        adata.obs[key_added],
        ch,
        what="clonotypes",
        arms=arms,
        other="a loose definition merges unrelated cells: tighten metric= / cutoff= (or same_v_gene=True)",
    )
    params = _params(
        ir,
        sequence=sequence,
        metric=metric,
        cutoff=cutoff,
        receptor_arms=receptor_arms,
        receptor_arms_used=arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
        same_j_gene=same_j_gene,
        within_group=within_group,
        partitions=partitions,
        key_added=key_added,
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    return ScirpyResult("define_clonotype_clusters", adata, (key_added, f"{key_added}_size"), tab, summary, params)


def region_table(units: pd.DataFrame, *, column: str, groupby: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Per-region summary of a unit label.

    Parameters
    ----------
    units
        One row per unit (``result.units``).
    column
        The label: categorical / boolean / text (counts and shares of each value) or numeric (count, mean, median,
        sd).
    groupby
        Region column(s); ``None`` summarises all units.

    Returns
    -------
    pandas.DataFrame
        Categorical labels: one row per (region, value) with ``n_units`` and ``share`` (of the region's units with a
        value). Numeric labels: one row per region with ``n_units``, ``mean``, ``median`` and ``sd``.
    """
    by = [] if groupby is None else ([groupby] if isinstance(groupby, str) else list(groupby))
    for c in [column, *by]:
        if c not in units.columns:
            raise KeyError(f"region_table: column {c!r} not in the units table")
    u = units[units[column].notna()]
    s = u[column]
    numeric = pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
    if numeric and not isinstance(s.dtype, pd.CategoricalDtype):
        if by:
            g = u.groupby(by, observed=True, sort=True)[column]
            return g.agg(n_units="size", mean="mean", median="median", sd="std").reset_index()
        return pd.DataFrame(
            {"n_units": [len(s)], "mean": [s.mean()], "median": [s.median()], "sd": [s.std()]}, index=[0]
        )
    keys = [*by, column]
    t = u.groupby(keys, observed=True, sort=True).size().rename("n_units").reset_index()
    tot = t.groupby(by, observed=True)["n_units"].transform("sum") if by else t["n_units"].sum()
    t["share"] = t["n_units"] / tot
    return t


def clonal_expansion(
    data: Any,
    *,
    target_col: str = S.CLONE,
    breakpoints: Sequence[int] = (1, 2),
    clip_at: int | None = None,
    expanded_in: str | None = None,
    groupby: str | None = None,
    receptor_arms: UnitArms = "auto",
    key_added: str = "clonal_expansion",
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ScirpyResult:
    """Clonal-expansion classes of scirpy on spatial units (``tl.clonal_expansion``), with per-region shares.

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    target_col
        ``obs`` column of the clone or clonotype (default the clone of each locus; on spots and cells run
        :func:`clonotypes` first and pass its ``key_added``).
    breakpoints, clip_at, expanded_in
        Passed to ``scirpy.tl.clonal_expansion``: size classes ``"<= 1"``, ``"<= 2"``, ``"> 2"`` for ``(1, 2)``, sizes
        counted in units (loci: about cells) over all units or within ``expanded_in``.
    groupby
        Region column (a record or ``obs`` column) of the per-region table.
    receptor_arms
        Units counted: ``"VDJ"`` (units with a heavy / beta / delta chain), ``"VJ"``, ``"all"``. Default ``"auto"``:
        ``"VDJ"`` for B-cell units (light-chain clones are shared by unrelated B cells and would make most loci
        "expanded"), ``"all"`` otherwise. The other units keep no value (``params["n_units_left_out"]``).
    key_added
        ``obs`` column of the class (missing on units not counted).
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ScirpyResult
        ``table`` (:func:`region_table`): one row per (region, class) with ``n_units`` and ``share``.

    Warns
    -----
    LargeClonotypeWarning
        When one clone holds more than 5% of the counted units (and at least 50).

    Notes
    -----
    The same classes counted by :func:`spaclone.tl.clonal_expansion` (``unit="locus"``) agree with scirpy's when the
    units are the same. Shares per region are descriptive: compare regions with :func:`spaclone.tl.diversity`
    (``"expanded"``), which rarefies.
    """
    ir = _ir()
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw, (groupby, expanded_in))
    _need_obs(adata, target_col, "clonal_expansion")
    ch = _chains(ir, adata, airr_key)
    arms = _resolve_arms(receptor_arms, ch, _UNIT_ARMS)
    tcol, n_out = _arm_target(adata, target_col, arms, ch)
    extra = {} if clip_at is None else {"clip_at": clip_at}
    try:
        _warn_large(
            adata.obs[tcol],
            ch,
            what=f"clonal_expansion ({target_col})",
            arms=arms,
            other=f"one clone rarely holds that many units: check target_col={target_col!r}",
        )
        ir.tl.clonal_expansion(
            adata,
            target_col=tcol,
            expanded_in=expanded_in,
            breakpoints=tuple(breakpoints),
            key_added=key_added,
            **extra,
        )
    finally:
        _drop_tmp(adata, tcol, target_col)
    _no_nan_class(adata, key_added)
    u = to_units(adata, x_key=x_key, y_key=y_key)
    tab = region_table(u, column=key_added, groupby=groupby)
    cls = adata.obs[key_added]
    counted = cls.notna().to_numpy()
    single = (cls[counted].astype(str) == f"<= {breakpoints[0]}").to_numpy()
    summary = {
        "n_units": int(adata.n_obs),
        "n_counted": int(counted.sum()),
        "n_classes": int(cls.nunique()),
        "share_singletons": float(single.mean()) if len(single) else 0.0,
    }
    params = _params(
        ir,
        target_col=target_col,
        breakpoints=tuple(breakpoints),
        expanded_in=expanded_in,
        groupby=groupby,
        receptor_arms=receptor_arms,
        receptor_arms_used=arms,
        n_units_left_out=n_out,
        key_added=key_added,
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    return ScirpyResult("clonal_expansion", adata, (key_added,), tab, summary, params)


def expansion_map(
    data: Any,
    *,
    target_col: str = S.CLONE,
    min_size: int = 2,
    expanded_in: str | None = None,
    sigma_um: float = 50.0,
    pixel_um: float = 10.0,
    tissue: Any = None,
    match_by: str | None = None,
    min_neff: float = 10.0,
    receptor_arms: UnitArms = "auto",
    key_added: str = "clonal_expansion",
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ExpansionMap:
    """Where expanded clones sit: scirpy's expansion classes on units, smoothed into a share map.

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    target_col
        Clone or clonotype column of the units.
    min_size
        A unit is expanded when its clone has at least this many units (scirpy ``breakpoints=(min_size - 1,)``: the
        class ``"> min_size - 1"``).
    expanded_in
        Count clone sizes within this column (scirpy's ``expanded_in``), e.g. the section.
    sigma_um, pixel_um, tissue, match_by, min_neff
        Passed to :func:`spaclone.tl.share_map` (Gaussian kernel, grid, tissue mask, covariate-matched expectation,
        eligibility).
    receptor_arms
        Units counted: ``"VDJ"`` (units with a heavy / beta / delta chain), ``"VJ"``, ``"all"``. Default ``"auto"``:
        ``"VDJ"`` for B-cell units (light-chain clones are shared by unrelated B cells and would make most loci
        "expanded"), ``"all"`` otherwise. The other units keep no value (``params["n_units_left_out"]``).
    key_added
        ``obs`` column of the scirpy class; the boolean ``"expanded"`` is added next to it (missing on units not
        counted, which are left out of the map).
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ExpansionMap
        ``share_map`` (:class:`spaclone.tl.ShareMap` of the share of expanded units), ``table`` (:func:`region_table`
        of ``"expanded"``), ``plot(kind="share" | "units")``.

    Notes
    -----
    The map is descriptive (its ``z`` is a standardised residual, not a test). Whether expanded clones concentrate
    beyond chance is tested by :func:`spaclone.tl.hotspots` on ``result.units`` with ``mark="expanded"`` and
    clone-level shuffles.
    """
    if min_size < 2:
        raise ValueError("min_size must be >= 2")
    res = clonal_expansion(
        data,
        target_col=target_col,
        breakpoints=(min_size - 1,),
        expanded_in=expanded_in,
        receptor_arms=receptor_arms,
        key_added=key_added,
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    from spaclone.tl.maps import share_map

    adata = res.adata
    cls = adata.obs[key_added]
    counted = cls.notna().to_numpy()
    expanded = pd.array((cls.astype(str) == f"> {min_size - 1}").to_numpy(dtype=bool), dtype="boolean")
    expanded[~counted] = pd.NA
    adata.obs["expanded"] = expanded
    u = to_units(adata, x_key=x_key, y_key=y_key)
    u = u[counted].reset_index(drop=True)
    u["expanded"] = u["expanded"].astype(bool)
    sm = share_map(
        u,
        mark="expanded",
        sigma_um=sigma_um,
        pixel_um=pixel_um,
        tissue=tissue,
        match_by=match_by,
        min_neff=min_neff,
        x_key=x_key,
        y_key=y_key,
    )
    tab = region_table(u, column="expanded")
    summary = {
        "n_units": int(adata.n_obs),
        "n_counted": int(counted.sum()),
        "n_expanded": int(u["expanded"].sum()),
        "p0": float(sm.p0),
    }
    params = {**res.params, "min_size": min_size, "sigma_um": sigma_um, "pixel_um": pixel_um, "match_by": match_by}
    return ExpansionMap("clonal_expansion", adata, (key_added, "expanded"), tab, summary, params, share_map=sm)


def diversity_by_region(
    data: Any,
    *,
    groupby: str,
    target_col: str = S.CLONE,
    metric: str | Callable[[np.ndarray], float] = "normalized_shannon_entropy",
    metric_kwargs: dict[str, Any] | None = None,
    receptor_arms: UnitArms = "auto",
    key_added: str | None = None,
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ScirpyResult:
    """Alpha diversity of clones per region with scirpy (``tl.alpha_diversity``), the value on every unit.

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    groupby
        Region column (record or ``obs`` column: germinal centre, compartment, niche, section).
    target_col
        Clone or clonotype column of the units.
    metric, metric_kwargs
        Passed to ``scirpy.tl.alpha_diversity``: ``"normalized_shannon_entropy"`` (Shannon entropy / log of the
        number of clones; 0 for one clone), ``"D50"``, ``"DXX"`` (``metric_kwargs={"percentage": 80}``), any
        scikit-bio alpha metric (needs scikit-bio), or a function of the clone-size vector.
    receptor_arms
        Units counted: ``"VDJ"`` (units with a heavy / beta / delta chain), ``"VJ"``, ``"all"``. Default ``"auto"``:
        ``"VDJ"`` for B-cell units (light-chain clones are shared by unrelated B cells and would make most loci
        "expanded"), ``"all"`` otherwise. The other units keep no value (``params["n_units_left_out"]``).
    key_added
        ``obs`` column receiving each unit's region value (scirpy's default ``"{metric}_{target_col}"``).
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ScirpyResult
        ``table``: one row per region with ``groupby``, ``metric``, ``value`` (scirpy's number), ``n_units`` and
        ``n_clones`` (units and distinct clones with a clone value in the region).

    Notes
    -----
    The values are scirpy's, unrarefied: regions with more units have more clones, and the normalised entropy and D50
    move with the number of units. To compare regions, use :func:`spaclone.tl.diversity`, which rarefies to a common
    number of units and tests the difference against random placement of intact clones. scirpy's normalised Shannon
    entropy equals the ``"evenness"`` of :func:`spaclone.tl.diversity` (Pielou's J), except for one-clone regions (0
    here, missing there).
    """
    ir = _ir()
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw, (groupby,))
    _need_obs(adata, target_col, "diversity_by_region")
    ch = _chains(ir, adata, airr_key)
    arms = _resolve_arms(receptor_arms, ch, _UNIT_ARMS)
    tcol, n_out = _arm_target(adata, target_col, arms, ch)
    try:
        _warn_large(
            adata.obs[tcol],
            ch,
            what=f"diversity_by_region ({target_col})",
            arms=arms,
            other=f"one clone rarely holds that many units: check target_col={target_col!r}",
        )
        df = ir.tl.alpha_diversity(
            adata, groupby, target_col=tcol, metric=metric, inplace=False, **(metric_kwargs or {})
        )
        obs = adata.obs[[groupby, tcol]].copy()
    finally:
        _drop_tmp(adata, tcol, target_col)
    name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
    key = key_added or f"{name}_{target_col}"
    values = df.iloc[:, 0]
    adata.obs[key] = adata.obs[groupby].astype(object).map(values.to_dict()).astype(float)
    obs = obs[obs[tcol].notna() & obs[groupby].notna()]
    g = obs.groupby(groupby, observed=True)
    tab = pd.DataFrame(
        {
            groupby: values.index.to_numpy(),
            "metric": name,
            "value": values.to_numpy(dtype=float),
        }
    )
    tab["n_units"] = tab[groupby].map(g.size()).fillna(0).astype(np.int64).to_numpy()
    tab["n_clones"] = tab[groupby].map(g[tcol].nunique()).fillna(0).astype(np.int64).to_numpy()
    summary = {"n_regions": len(tab), "metric": name, "min": float(tab.value.min()), "max": float(tab.value.max())}
    params = _params(
        ir,
        groupby=groupby,
        target_col=target_col,
        metric=name,
        metric_kwargs=metric_kwargs,
        receptor_arms=receptor_arms,
        receptor_arms_used=arms,
        n_units_left_out=n_out,
        key_added=key,
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    return ScirpyResult("alpha_diversity", adata, (key, groupby), tab, summary, params)


def repertoire_overlap(
    data: Any,
    *,
    groupby: str,
    target_col: str = S.CLONE,
    overlap_measure: str = "jaccard",
    overlap_threshold: float | None = None,
    fraction: str | bool = False,
    receptor_arms: UnitArms = "auto",
    added_key: str = "repertoire_overlap",
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ScirpyResult:
    """Clonotype overlap between regions with scirpy (``tl.repertoire_overlap``), as a tidy pair table.

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    groupby
        Region column (record or ``obs`` column).
    target_col
        Clone or clonotype column of the units.
    overlap_measure, overlap_threshold, fraction
        Passed to ``scirpy.tl.repertoire_overlap`` (a ``scipy.spatial.distance`` measure on clone presence; a clone
        is present in a region when its weight exceeds ``overlap_threshold``).
    receptor_arms
        Units counted: ``"VDJ"`` (units with a heavy / beta / delta chain), ``"VJ"``, ``"all"``. Default ``"auto"``:
        ``"VDJ"`` for B-cell units (light-chain clones are shared by unrelated B cells and would make most loci
        "expanded"), ``"all"`` otherwise. The other units keep no value (``params["n_units_left_out"]``).
    added_key
        ``uns`` key where scirpy stores the weighted matrix, the distances and the linkage.
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ScirpyResult
        ``table``: one row per pair of regions with ``group_a``, ``group_b``, ``distance`` (scirpy's), ``n_shared``
        (clones present in both), ``n_a``, ``n_b`` (clones present in each).

    Notes
    -----
    Neighbouring regions share clones because cells of a clone sit together; how much sharing random placement gives
    is tested by :func:`spaclone.tl.region_sharing` and :func:`spaclone.tl.overlap` (``placebo=``).
    """
    from scipy.spatial.distance import squareform

    ir = _ir()
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw, (groupby,))
    _need_obs(adata, target_col, "repertoire_overlap")
    ch = _chains(ir, adata, airr_key)
    arms = _resolve_arms(receptor_arms, ch, _UNIT_ARMS)
    tcol, n_out = _arm_target(adata, target_col, arms, ch)
    try:
        _warn_large(
            adata.obs[tcol],
            ch,
            what=f"repertoire_overlap ({target_col})",
            arms=arms,
            other=f"one clone rarely holds that many units: check target_col={target_col!r}",
        )
        ir.tl.repertoire_overlap(
            adata,
            groupby,
            target_col=tcol,
            overlap_measure=overlap_measure,
            overlap_threshold=overlap_threshold,
            fraction=fraction,
            inplace=True,
            added_key=added_key,
        )
    finally:
        _drop_tmp(adata, tcol, target_col)
    res = adata.uns[added_key]
    w = res["weighted"]
    dist = squareform(np.asarray(res["distance"], dtype=float))
    pres = (w > (overlap_threshold or 0)).to_numpy(dtype=bool)
    groups = list(w.index)
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            rows.append(
                {
                    "group_a": groups[i],
                    "group_b": groups[j],
                    "distance": float(dist[i, j]),
                    "n_shared": int((pres[i] & pres[j]).sum()),
                    "n_a": int(pres[i].sum()),
                    "n_b": int(pres[j].sum()),
                }
            )
    tab = pd.DataFrame(rows, columns=["group_a", "group_b", "distance", "n_shared", "n_a", "n_b"])
    summary = {"n_regions": len(groups), "n_pairs": len(tab), "measure": overlap_measure}
    params = _params(
        ir,
        groupby=groupby,
        target_col=target_col,
        overlap_measure=overlap_measure,
        overlap_threshold=overlap_threshold,
        fraction=fraction,
        receptor_arms=receptor_arms,
        receptor_arms_used=arms,
        n_units_left_out=n_out,
        added_key=added_key,
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    return ScirpyResult("repertoire_overlap", adata, (groupby,), tab, summary, params)


def clonotype_modularity(
    data: Any,
    *,
    target_col: str = S.CLONE,
    graph_radius_um: float | None = 50.0,
    n_neighbors: int | None = None,
    connectivity_key: str | None = None,
    permutation_test: Literal["approx", "exact"] = "approx",
    n_permutations: int | None = None,
    rng: RngLike = None,
    receptor_arms: UnitArms = "auto",
    key_added: str = "clonotype_modularity",
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ModularityResult:
    """Are the units of a clonotype more connected on the spatial neighbour graph than random units?

    scirpy's clonotype modularity (``tl.clonotype_modularity``) on the spatial graph of the units instead of the
    transcriptome graph: for each clonotype, log2((edges among its units + 1) / (expected edges + 1)).

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    target_col
        Clone or clonotype column of the units.
    graph_radius_um
        Units within this distance are neighbours (:func:`spaclone.gex.spatial_graph`; edges never cross sections).
        Keep it well above the locus radius: two loci of one clone are always more than ``radius_um`` apart.
    n_neighbors
        Use the k nearest units instead of a radius.
    connectivity_key
        ``obsp`` key of an existing graph (overrides the spatial graph).
    permutation_test, n_permutations
        scirpy's null: ``"approx"`` (negative binomial fitted to 1,000 random subgraphs per clonotype size) or
        ``"exact"`` (10,000 random subgraphs).
    rng
        Seed or generator; scirpy receives an integer ``random_state`` drawn from it.
    receptor_arms
        Units counted: ``"VDJ"`` (units with a heavy / beta / delta chain), ``"VJ"``, ``"all"``. Default ``"auto"``:
        ``"VDJ"`` for B-cell units (light-chain clones are shared by unrelated B cells and would make most loci
        "expanded"), ``"all"`` otherwise. The other units keep no value (``params["n_units_left_out"]``).
    key_added
        ``obs`` columns ``key_added``, ``{key_added}_pvalue`` and ``{key_added}_fdr``.
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ModularityResult
        ``table``: one row per clonotype with ``clonotype``, ``n_units``, ``statistic`` (modularity), ``p_value``
        (scirpy's), ``q_value`` (BH over clonotypes, = scirpy's FDR), ``n_perm``.

    Warns
    -----
    spaclone.ext.squidpy.DepthNullWarning
        Always: the null keeps neither unit density nor receptor depth.

    Null model
    ----------
    For each clonotype size k, k units are drawn uniformly at random from all units with a clonotype and
    the edges among them counted (random subgraphs; positions and graph fixed). Units are not exchangeable in
    tissue: dense B-cell areas hold more units and more edges, so clonotypes in follicles look modular under this
    null whatever their positions mean. Use it to rank clonotypes, and :func:`spaclone.qc.colocalisation`
    (same-clone pairs against clone labels shuffled within depth strata) to test whether positions carry clonal
    information.

    References
    ----------
    Sturm et al. 2020, doi:10.1093/bioinformatics/btaa611 (scirpy); Schattgen et al. 2022,
    doi:10.1038/s41587-021-00989-2 (CoNGA, the graph-overlap idea).
    """
    from spaclone.ext.squidpy import _depth_warning
    from spaclone.gex.spatial import spatial_graph

    ir = _ir()
    gen = as_rng(rng)
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw)
    _need_obs(adata, target_col, "clonotype_modularity")
    key = connectivity_key
    if key is None:
        lib = library_key if library_key in adata.obs.columns else None
        adata.obsp[_GRAPH] = spatial_graph(
            adata,
            radius_um=graph_radius_um if n_neighbors is None else None,
            n_neighbors=n_neighbors,
            library_key=lib,
        )
        key = _GRAPH
    elif key not in adata.obsp:
        raise KeyError(f"clonotype_modularity: adata.obsp[{key!r}] is missing")
    seed = int(gen.integers(0, 2**31 - 1))
    _depth_warning("clonotype_modularity")
    ch = _chains(ir, adata, airr_key)
    arms = _resolve_arms(receptor_arms, ch, _UNIT_ARMS)
    tcol, n_out = _arm_target(adata, target_col, arms, ch)
    try:
        _warn_large(
            adata.obs[tcol],
            ch,
            what=f"clonotype_modularity ({target_col})",
            arms=arms,
            other=f"one clone rarely holds that many units: check target_col={target_col!r}",
        )
        scores, pvals = ir.tl.clonotype_modularity(
            adata,
            target_col=tcol,
            connectivity_key=key,
            permutation_test=permutation_test,
            n_permutations=n_permutations,
            inplace=False,
            fdr_correction=False,
            random_state=seed,
        )
        lab = adata.obs[tcol].astype(object)
    finally:
        _drop_tmp(adata, tcol, target_col)
    clon = list(scores)
    p = np.array([pvals[c] for c in clon], dtype=float)
    q = bh(p)
    sizes = lab.value_counts()
    n_perm = n_permutations or (1000 if permutation_test == "approx" else 10000)
    tab = pd.DataFrame(
        {
            "clonotype": [str(c) for c in clon],
            "n_units": [int(sizes.get(c, 0)) for c in clon],
            "statistic": np.array([scores[c] for c in clon], dtype=float),
            "p_value": p,
            "q_value": q,
            "n_perm": n_perm,
        }
    )
    tab = tab.sort_values(["p_value", "statistic"], ascending=[True, False], kind="stable").reset_index(drop=True)
    qd = dict(zip(clon, q, strict=True))
    adata.obs[key_added] = lab.map(scores).astype(float).to_numpy()
    adata.obs[f"{key_added}_pvalue"] = lab.map(pvals).astype(float).to_numpy()
    adata.obs[f"{key_added}_fdr"] = lab.map(qd).astype(float).to_numpy()
    params = _params(
        ir,
        target_col=target_col,
        graph_radius_um=graph_radius_um if connectivity_key is None and n_neighbors is None else None,
        n_neighbors=n_neighbors,
        connectivity_key=connectivity_key,
        permutation_test=permutation_test,
        n_permutations=n_perm,
        random_state=seed,
        rng_entropy=rng_entropy(gen),
        receptor_arms=receptor_arms,
        receptor_arms_used=arms,
        n_units_left_out=n_out,
        design="scirpy random subgraphs of each clonotype size (units relabelled at random, graph fixed)",
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    multi = tab[tab.n_units >= 2]
    summary = {
        "n_clonotypes": len(tab),
        "n_tested_ge2": len(multi),
        "n_q05": int((multi.q_value < 0.05).sum()),
        "n_edges": int(adata.obsp[key].nnz),
    }
    return ModularityResult(name="clonotype_modularity", table=tab, summary=summary, params=params, adata=adata)


def vdj_usage(
    data: Any,
    *,
    gene: Literal["v_call", "d_call", "j_call", "c_call"] = "v_call",
    chain: Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"] = "VDJ_1",
    groupby: str | None = None,
    allele: bool = False,
    key_added: str | None = None,
    level: Level = "locus",
    radius_um: float = 15.0,
    carry: str | Sequence[str] = (),
    collapse: bool = True,
    airr_key: str = "airr",
    cell_key: str = "cell_id",
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> ScirpyResult:
    """Gene usage of one chain per region (scirpy ``get.airr`` + ``tl.group_abundance``), with the gene on every unit.

    Parameters
    ----------
    data
        Receptor table, SpatialAIR or scirpy AnnData.
    gene
        AIRR gene field of the chain.
    chain
        scirpy chain: ``"VDJ_1"`` (primary heavy / beta), ``"VJ_1"`` (primary light / alpha), or the secondary ones.
    groupby
        Region column; ``None`` counts over all units.
    allele
        Keep alleles (``IGHV1-7*01``); default the gene of the first call (:func:`spaclone.pp.gene_name`).
    key_added
        ``obs`` column of the gene (default ``"{chain}_{gene}"``).
    level, radius_um, carry, collapse, airr_key, cell_key, spot_key, x_key, y_key, clone_key, locus_key, library_key, productive_key, reads_key
        Units and keys (see :func:`prepare`).

    Returns
    -------
    ScirpyResult
        ``table``: one row per (region, gene) with ``n_units`` (scirpy ``group_abundance`` counts) and ``share``
        (scirpy ``group_abundance`` with ``fraction=groupby``: the share of the region's units with that chain).

    Notes
    -----
    Units are counted once each (loci: about cells), not records, so one cell's many molecules do not inflate a
    gene. Spatial differences in usage between regions are tested against random placement by
    :func:`spaclone.tl.gene_usage`.
    """
    from spaclone.pp.clones import gene_name

    ir = _ir()
    kw = _keys(
        level=level,
        radius_um=radius_um,
        carry=carry,
        collapse=collapse,
        airr_key=airr_key,
        cell_key=cell_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    adata = _prep(data, kw, (groupby,))
    vals = ir.get.airr(adata, gene, chain)
    vals = pd.Series(vals, index=adata.obs_names).astype(object)
    if not allele:
        vals = gene_name(vals).astype(object)
    key = key_added or f"{chain}_{gene}"
    adata.obs[key] = pd.Categorical(vals.where(vals.notna(), None).to_numpy(dtype=object))
    grp = groupby
    if grp is None:
        grp = "_spaclone_all"
        adata.obs[grp] = pd.Categorical(["all"] * adata.n_obs)
    sub = adata[adata.obs[key].notna().to_numpy() & adata.obs[grp].notna().to_numpy()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        counts = ir.tl.group_abundance(sub, groupby=key, target_col=grp)
        shares = ir.tl.group_abundance(sub, groupby=key, target_col=grp, fraction=grp)
    if groupby is None:
        del adata.obs[grp]
    c = counts.reset_index().melt(id_vars=key, var_name=grp, value_name="n_units")
    s = shares.reset_index().melt(id_vars=key, var_name=grp, value_name="share")
    tab = c.merge(s, on=[key, grp], how="left").rename(columns={key: "gene"})
    tab = tab[tab["n_units"] > 0]
    if groupby is None:
        tab = tab.drop(columns=grp)
        tab = tab.sort_values(["n_units", "gene"], ascending=[False, True], kind="stable")
    else:
        tab = tab[[grp, "gene", "n_units", "share"]].sort_values(
            [grp, "n_units", "gene"], ascending=[True, False, True], kind="stable"
        )
    tab["gene"] = tab["gene"].astype(str)
    tab["n_units"] = tab["n_units"].astype(np.int64)
    tab = tab.reset_index(drop=True)
    summary = {"n_units": int(sub.n_obs), "n_genes": int(tab["gene"].nunique())}
    params = _params(
        ir,
        gene=gene,
        chain=chain,
        groupby=groupby,
        allele=allele,
        key_added=key,
        level=level,
        radius_um=radius_um,
        collapse=collapse,
        airr_key=airr_key,
    )
    return ScirpyResult("group_abundance", adata, (key,), tab, summary, params)
