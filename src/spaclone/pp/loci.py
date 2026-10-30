"""Locus merging: same-clone spots within a radius form one sampling unit (about one cell).

One cell can spill its receptor molecules over several neighbouring spots (bins, beads), and one spot can hold
several cells. Counting spots therefore over-counts cells and makes clones look clustered. A *locus* is a connected
component of the spots that carry the same clone and lie within ``radius_um`` of each other (single linkage); it is the
cell-scale unit used by the concentration tests and by the read-floor diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from spaclone import _schema as S

__all__ = ["Loci", "merge_loci"]


class Loci(NamedTuple):
    """Result of :func:`merge_loci` (unpack as ``loci, membership = merge_loci(air)``).

    Attributes
    ----------
    loci
        One row per locus, see :func:`merge_loci`.
    membership
        ``locus_id`` of every record, indexed like the input table.
    """

    loci: pd.DataFrame
    membership: pd.Series


def _resolve_by(air: pd.DataFrame, by: str | Sequence[str] | None, defaults: Sequence[str]) -> list[str]:
    """Strata columns: ``None`` = the default columns that are present; otherwise exactly the given columns."""
    if by is None:
        return [c for c in defaults if c in air.columns]
    cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in cols if c not in air.columns]
    if missing:
        raise KeyError(f"merge_loci: strata column(s) {missing} not in the table")
    return cols


def merge_loci(
    air: pd.DataFrame,
    *,
    radius_um: float = 15.0,
    clone_key: str = S.CLONE,
    by: str | Sequence[str] | None = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
) -> Loci:
    """Merge the spots of one clone that lie within ``radius_um`` of each other into loci.

    Within each stratum (``by``) and clone, the distinct spots carrying the clone are linked when their centres are at
    most ``radius_um`` apart; a locus is a connected component of that graph (single linkage). With 10-µm bins and the
    default 15 µm, adjacent bins including diagonal ones are linked, which is the T1 / TR1 rule of the archive.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig or call) with spot ids, coordinates in µm and clone ids. Spot
        coordinates are assumed constant per spot (the first record's position is used).
    radius_um
        Linkage distance in µm (inclusive: spots exactly ``radius_um`` apart are linked). 0 merges only records on the
        same spot (or on spots at identical positions).
    clone_key
        Column with the clone (clonotype) id; missing values are not allowed.
    by
        Strata in which clones are merged separately, e.g. the chain and the library (section). ``None`` (default)
        uses whichever of ``locus_key`` and ``library_key`` are present; ``()`` merges across all records.
    spot_key, x_key, y_key, locus_key, library_key, reads_key
        Column names. ``reads_key`` (reads per record, e.g. ``consensus_count``) is optional; without it the read
        columns of the result are NaN.

    Returns
    -------
    Loci
        ``loci``: one row per locus with columns ``locus_id`` (0, 1, ... ordered by stratum, clone and first spot),
        the clone, chain and library columns (named as in the input; chain / library NA when absent), ``x_um``,
        ``y_um`` (centroid of the locus' distinct spots), ``n_spots``, ``n_records``, ``reads_max`` and ``reads_sum``
        (over the locus' records), the spot column (the spot of the locus' record with most reads, so that the loci
        table is itself a valid receptor table) and any further ``by`` columns. ``membership``: the ``locus_id`` of
        every input record (int64 Series with the input's index).

    Notes
    -----
    Locus ids do not depend on the row order of ``air``. Coordinates on a regular grid (such as Stereo-seq bins at
    multiples of 0.5 µm) are handled exactly, so distances equal to ``radius_um`` are always linked.

    Examples
    --------
    >>> loci, member = merge_loci(air, radius_um=15)  # doctest: +SKIP
    >>> air["locus_id"] = member  # doctest: +SKIP
    """
    S.validate_air(air, required=(spot_key, x_key, y_key, clone_key), numeric=(x_key, y_key))
    radius_um = float(radius_um)
    if not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError(f"radius_um must be a finite number >= 0, got {radius_um}")
    if air[clone_key].isna().any():
        raise ValueError(f"column {clone_key!r} has missing values; drop or label those records first")
    if air[spot_key].isna().any():
        raise ValueError(f"column {spot_key!r} has missing values")
    strata = _resolve_by(air, by, (locus_key, library_key))
    strata = [c for c in strata if c != clone_key]
    out_cols = _output_columns(strata, clone_key, locus_key, library_key, x_key, y_key, spot_key)
    n = len(air)
    if n == 0:
        empty = pd.DataFrame({c: pd.Series(dtype=float if c in (x_key, y_key) else object) for c in out_cols})
        empty[S.LOCUS_ID] = empty[S.LOCUS_ID].astype(np.int64)
        return Loci(empty, pd.Series(np.empty(0, np.int64), index=air.index, name=S.LOCUS_ID))

    keys = [*strata, clone_key]
    gcode = air.groupby(keys, sort=True, dropna=False, observed=True).ngroup().to_numpy(np.int64)
    scode, spots = pd.factorize(air[spot_key], sort=True)
    site_key = gcode * np.int64(len(spots)) + scode.astype(np.int64)
    site, _ = pd.factorize(site_key, sort=True)
    n_site = int(site.max()) + 1
    first = np.empty(n_site, np.int64)
    first[site[::-1]] = np.arange(n)[::-1]
    x = air[x_key].to_numpy(dtype=float)
    y = air[y_key].to_numpy(dtype=float)
    sx, sy, sg = x[first], y[first], gcode[first]

    comp = _components(sx, sy, sg, radius_um)
    n_loc = int(comp.max()) + 1
    site_locus = comp
    rec_locus = site_locus[site]

    # one representative record per locus: most reads, ties -> first site (sorted spot order)
    rec_site = site
    reads = air[reads_key].to_numpy(dtype=float) if reads_key in air.columns else np.full(n, np.nan)
    order = np.lexsort((rec_site, -np.nan_to_num(reads, nan=-np.inf), rec_locus))
    is_first = np.r_[True, rec_locus[order][1:] != rec_locus[order][:-1]]
    rep = order[is_first]  # record index per locus, in locus order

    n_spots = np.bincount(site_locus, minlength=n_loc)
    cx = np.bincount(site_locus, weights=sx, minlength=n_loc) / n_spots
    cy = np.bincount(site_locus, weights=sy, minlength=n_loc) / n_spots
    rs = pd.Series(reads)
    g = rs.groupby(rec_locus)
    table = {
        S.LOCUS_ID: np.arange(n_loc, dtype=np.int64),
        clone_key: air[clone_key].to_numpy()[rep],
        locus_key: air[locus_key].to_numpy()[rep] if locus_key in air.columns else pd.NA,
        library_key: air[library_key].to_numpy()[rep] if library_key in air.columns else pd.NA,
        x_key: cx,
        y_key: cy,
        "n_spots": n_spots.astype(np.int64),
        "n_records": np.bincount(rec_locus, minlength=n_loc).astype(np.int64),
        "reads_max": g.max().reindex(range(n_loc)).to_numpy(dtype=float),
        "reads_sum": g.sum(min_count=1).reindex(range(n_loc)).to_numpy(dtype=float),
        spot_key: air[spot_key].to_numpy()[rep],
    }
    for c in strata:
        if c not in table:
            table[c] = air[c].to_numpy()[rep]
    loci = pd.DataFrame(table)[out_cols]
    loci.attrs["spaclone"] = {
        **dict(air.attrs.get("spaclone", {})),
        "merge_loci": {"radius_um": radius_um, "by": strata, "clone_key": clone_key},
    }
    membership = pd.Series(rec_locus.astype(np.int64), index=air.index, name=S.LOCUS_ID)
    return Loci(loci, membership)


def _output_columns(strata, clone_key, locus_key, library_key, x_key, y_key, spot_key) -> list[str]:
    base = [S.LOCUS_ID, clone_key, locus_key, library_key, x_key, y_key, "n_spots", "n_records", "reads_max"]
    base += ["reads_sum", spot_key]
    return base + [c for c in strata if c not in base]


def _components(sx: np.ndarray, sy: np.ndarray, group: np.ndarray, radius_um: float) -> np.ndarray:
    """Single-linkage components of sites within ``radius_um`` inside each group.

    Groups are separated in space by a power-of-two offset along x, so one KD-tree finds all within-group pairs and no
    pair across groups; grid coordinates (multiples of a power of two) stay exact. Components are labelled in the order
    of their smallest site index.
    """
    n_site = len(sx)
    size = np.bincount(group)
    idx = np.flatnonzero(size[group] >= 2)
    rows = cols = np.empty(0, np.int64)
    if len(idx):
        x0, y0 = sx[idx].min(), sy[idx].min()
        span = float(sx[idx].max() - x0)
        step = 2.0 ** np.ceil(np.log2(span + 2.0 * radius_um + 2.0))
        gid = pd.factorize(group[idx])[0].astype(float)
        pts = np.column_stack([(sx[idx] - x0) + gid * step, sy[idx] - y0])
        pairs = cKDTree(pts).query_pairs(radius_um, output_type="ndarray")
        if len(pairs):
            rows, cols = idx[pairs[:, 0]], idx[pairs[:, 1]]
    adj = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_site, n_site))
    n_comp, lab = connected_components(adj, directed=False)
    first = np.full(n_comp, n_site, np.int64)
    np.minimum.at(first, lab, np.arange(n_site))
    rank = np.empty(n_comp, np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(n_comp)
    return rank[lab]
