"""Class-switch recombination (CSR): isotype transitions within clones, switched-share maps and region tests.

* :func:`csr_transitions` asks which downstream isotypes appear in the same clones as which upstream ones. Because
  CSR only moves 5' to 3' on the IgH locus, an upstream and a downstream isotype in one clone mean that the lineage
  switched between them (or both descend from a cell upstream of both). Two calibrated questions: do clones carry a
  pair of isotypes more often than random labelling of the units predicts (``null="global"``), and, within the
  clones that carry both, do the downstream units sit later in the lineage (more mutated, deeper in the tree) or
  elsewhere in the tissue (any unit attribute) than the upstream units (``null="within"``)?
* :func:`csr_map` / :func:`csr_hotspots` map the switched share of loci (one unit per inferred cell, from
  :func:`spaclone.pp.merge_loci` and :func:`spaclone.bcr.call_isotype`) and call family-wise hotspots, through
  :func:`spaclone.tl.share_map` and :func:`spaclone.tl.hotspots` (archive T3).
* :func:`region_test` tests given regions (e.g. CSR hotspots) for an excess of switched or mutated units beyond
  their covariate-matched (isotype-matched) expectation, through :func:`spaclone.tl.region_excess` (archive T7).

References
----------
Stavnezer, Guikema & Schrader 2008, doi:10.1146/annurev.immunol.26.021607.090248; Horns et al. 2016,
doi:10.7554/eLife.16578 (class-switching networks from isotype co-occurrence within lineages); Hoehn et al. 2022,
doi:10.1371/journal.pcbi.1009885 (within-lineage permutation tests); Phipson & Smyth 2010,
doi:10.2202/1544-6115.1585.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    LowPowerWarning,
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    mc_pvalue,
    null_summary,
    rng_entropy,
)
from spaclone.bcr.isotype import isotype as _isotype
from spaclone.bcr.isotype import isotype_order
from spaclone.bcr.shm import _map_units, _tl, _to_loci
from spaclone.pp.loci import merge_loci

__all__ = ["csr_hotspots", "csr_map", "csr_transitions", "region_test"]

logger = logging.getLogger("spaclone")


def _unit_labels(
    air: pd.DataFrame,
    *,
    label_key: str,
    clone_key: str,
    unit_key: str | None,
    order: dict[str, int],
    order_key: str | None,
    strata: list[str],
) -> pd.DataFrame:
    """One row per unit: clone, isotype label (most frequent, ties to the 5'-most), order value, stratum code."""
    lab = air[label_key].astype("string")
    ok = lab.isin(list(order)).fillna(False).to_numpy() & air[clone_key].notna().to_numpy()
    cols = {"clone": air[clone_key].to_numpy(dtype=object)[ok], "label": lab.to_numpy(dtype=object)[ok]}
    if order_key is not None:
        cols["order"] = pd.to_numeric(air[order_key], errors="coerce").to_numpy(dtype=float)[ok]
    if strata:
        key = air[strata].astype(object).agg(tuple, axis=1) if len(strata) > 1 else air[strata[0]]
        cols["stratum"] = pd.factorize(pd.Series(key.to_numpy(dtype=object)[ok]), sort=True, use_na_sentinel=False)[0]
    if unit_key is None:
        d = pd.DataFrame(cols)
    else:
        cols["unit"] = air[unit_key].to_numpy(dtype=object)[ok]
        d = pd.DataFrame(cols)
        if strata and (d.groupby(["clone", "unit"]).stratum.nunique() > 1).any():
            raise ValueError(f"within strata {strata} vary inside a unit ({unit_key!r})")
        d["pos"] = d.label.map(order)
        cnt = d.groupby(["clone", "unit", "label"], sort=True).agg(n=("pos", "size"), pos=("pos", "first"))
        cnt = cnt.reset_index().sort_values(["clone", "unit", "n", "pos"], ascending=[True, True, False, True])
        top = cnt.drop_duplicates(["clone", "unit"])[["clone", "unit", "label"]]
        extra = {}
        if order_key is not None:
            extra["order"] = ("order", "mean")
        if strata:
            extra["stratum"] = ("stratum", "first")
        if extra:
            agg = d.groupby(["clone", "unit"], sort=True).agg(**extra).reset_index()
            top = top.merge(agg, on=["clone", "unit"], how="left")
        d = top.reset_index(drop=True)
    if order_key is not None:
        d = d[np.isfinite(d.order.to_numpy(dtype=float))].reset_index(drop=True)
    return d


def _resolve_units(
    air: pd.DataFrame,
    *,
    unit_key: str | None,
    order_key: str | None,
    isotype_key: str | None,
    radius_um: float,
    clone_key: str,
    spot_key: str,
    x_key: str,
    y_key: str,
    locus_key: str,
    library_key: str,
) -> tuple[pd.DataFrame, str | None, str]:
    """The table, the unit column and how the units were found (``unit_key="auto"`` resolution)."""
    if unit_key != "auto":
        return air, unit_key, "rows" if unit_key is None else f"column {unit_key!r}"
    for col in (S.LOCUS_ID, "cell_id"):
        if col in air.columns:
            return air, col, f"column {col!r}"
    if all(c in air.columns for c in (spot_key, x_key, y_key)):
        ok = (air[clone_key].notna() & air[spot_key].notna() & air[x_key].notna() & air[y_key].notna()).to_numpy()
        if ok.any():
            _, member = merge_loci(
                air.loc[ok],
                radius_um=radius_um,
                clone_key=clone_key,
                spot_key=spot_key,
                x_key=x_key,
                y_key=y_key,
                locus_key=locus_key,
                library_key=library_key,
            )
            unit = pd.Series(pd.NA, index=air.index, dtype="Int64")
            unit.loc[member.index] = member.to_numpy()
            return air.assign(_csr_unit=unit), "_csr_unit", f"loci built with pp.merge_loci ({radius_um:g} um)"
    n_dup = int(air.duplicated().sum())
    if n_dup:
        warnings.warn(
            f"{n_dup} rows are exact copies of another row and the table has no cell, locus or spot column: if they "
            "are several records of one cell, pass unit_key=<cell column> (the cell is the exchangeable unit: records "
            "of one cell share its isotype and mutations); pass unit_key=None if every row is a cell",
            SpacloneWarning,
            stacklevel=3,
        )
    return air, None, "rows"


def _within_perm(codes: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = codes.copy()
    for pos in groups:
        out[pos] = codes[pos][rng.permutation(len(pos))]
    return out


def csr_transitions(
    air: pd.DataFrame,
    *,
    clone_key: str = S.CLONE,
    species: Literal["mouse", "human"] = "mouse",
    level: Literal["subclass", "class"] = "subclass",
    isotype_key: str | None = None,
    unit_key: str | None = "auto",
    loci_radius_um: float = 15.0,
    null: Literal["global", "within"] = "global",
    order_key: str | None = None,
    within: str | Sequence[str] | None = None,
    n_perm: int = 1000,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    keep_null: bool = False,
    c_key: str = S.C_CALL,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> TestResult:
    """Pairs of upstream and downstream isotypes within clones, against random labelling.

    For every pair of isotypes ``(u, d)`` with ``u`` 5' of ``d`` on the IgH locus, the clones holding at least one
    unit of each are counted. ``null="global"`` tests whether clones carry the pair more often than when the
    isotype labels of all units are shuffled; ``null="within"`` tests, within the clones that carry both, whether the
    downstream units have larger values of ``order_key`` (V mutations, tree depth, distance to a structure) than the
    upstream units.

    Parameters
    ----------
    air
        Records (contigs) or units (loci) with a clone id and an isotype column.
    clone_key
        Clone column.
    species
        IgH locus order (see :data:`spaclone.bcr.CONSTANT_ORDER`).
    level
        ``"subclass"`` (constant genes; units without a subclass are left out) or ``"class"`` (IgM, IgD, IgG, IgE,
        IgA ordered by each class's most 5' gene; exact for mouse, a convention for human where IgG, IgA and IgE
        genes interleave).
    isotype_key
        Column with the isotype labels; default ``"isotype_subclass"`` or ``"isotype"`` (from
        :func:`spaclone.bcr.isotype`, computed from ``c_key`` when absent).
    unit_key
        Column of the unit that is exchanged: one cell. Records are collapsed to one label per unit before testing
        (the most frequent label of the unit, ties to the 5'-most; ``order_key`` is averaged). ``"auto"`` (default):
        the ``locus_id`` column when present (e.g. the membership of :func:`spaclone.pp.merge_loci`), else a
        ``cell_id`` column (AIRR single-cell tables, :func:`spaclone.pp.assign_to_cells`); otherwise, when the table
        has spot ids and coordinates, loci are built here (same-clone spots within ``loci_radius_um`` merged, about
        one cell); otherwise every row is a unit, with a warning when rows are exact copies of each other (records of
        one cell without a cell column). Rows without a unit are left out. ``None``: every row is a unit (one row per
        cell).
    loci_radius_um
        Linkage distance of the loci built by ``unit_key="auto"``.
    null
        ``"global"`` or ``"within"`` (see Notes).
    order_key
        Numeric unit attribute for ``null="within"``.
    within
        ``null="global"`` only: strata (columns, constant within units) inside which labels are shuffled, e.g. the
        library (section).
    n_perm
        Shuffles; the smallest attainable p-value is ``1 / (n_perm + 1)``.
    alternative
        ``"greater"`` (default): more co-occurrence than chance (``global``) or downstream units larger
        (``within``).
    rng
        Seed or generator.
    keep_null
        Keep the ``(n_perm, n_pairs)`` null draws.
    c_key
        Constant-call column used when no isotype column exists.
    spot_key, x_key, y_key, locus_key, library_key
        Columns used to build loci for ``unit_key="auto"`` (loci never span chains or sections).

    Returns
    -------
    TestResult
        ``table``: one row per pair with ``upstream``, ``downstream``, their locus positions, ``n_clones_upstream``,
        ``n_clones_downstream``, ``n_clones_both``, ``share`` (both / upstream), ``statistic`` (``global``: clones
        with both; ``within``: mean over those clones of mean(order | downstream) − mean(order | upstream)),
        ``null_mean``, ``null_sd``, ``z``, ``ratio`` (statistic / null mean, ``global`` only), ``p_value``,
        ``q_value`` (BH over pairs), ``n_perm``. Pairs that never co-occur keep ``NaN`` statistics for
        ``within``.

    Notes
    -----
    The exchangeable unit is the **cell**. The records of one cell (molecules, spots, contigs) share the cell's
    isotype and its mutations, so they are not exchangeable one by one: shuffling them as separate units splits one
    cell's label over several draws, which both narrows the null of ``"within"`` (in simulations with 1 + Poisson(6)
    records per cell and no association, 22-32% of pairs had p <= 0.05) and inflates the co-occurrence expected under
    ``"global"``. Hence the default ``unit_key="auto"``, which collapses records to loci (cells) first.

    Null model (``"global"``): random labelling of cells. The isotype labels of all cells are permuted among the
    cells (within ``within`` strata); clone membership, the number of cells per clone and the overall label counts
    are fixed. Clones are normally isotype-coherent, so most pairs co-occur less often than under this null; a pair
    at or above it is mixed within lineages beyond chance.

    Null model (``"within"``): within-clone random labelling of cells. In every clone the cells' labels are permuted
    among the clone's cells while the cells' ``order_key`` values stay in place; each clone keeps its multiset of
    isotypes, so the clones that carry a pair are the same in every shuffle. The null is exact when, within a clone,
    which cells carry which isotype is unrelated to their ``order_key`` values (every assignment of the clone's labels
    to its cells is then equally likely). A positive statistic beyond the null means that downstream isotypes occupy
    later (or other) positions of their lineages, as expected when switching follows mutation. When a switch is
    inherited by a whole subclone, the cells of that subclone share both isotype and mutations, and one switching
    event counts as several cells; for an event-level test use the lineage trees (:func:`spaclone.bcr.build_trees`,
    :func:`spaclone.tl.tip_region_test` with ``null="within_tree"``).

    References
    ----------
    Horns et al. 2016, doi:10.7554/eLife.16578; Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885; Phipson &
    Smyth 2010, doi:10.2202/1544-6115.1585; Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    if null not in ("global", "within"):
        raise ValueError(f"null must be 'global' or 'within', got {null!r}")
    if null == "within" and order_key is None:
        raise ValueError("null='within' needs order_key (the unit attribute compared between downstream and upstream)")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    for c in [clone_key, *([unit_key] if unit_key and unit_key != "auto" else []), *([order_key] if order_key else [])]:
        if c not in air.columns:
            raise KeyError(f"csr_transitions: column {c!r} not in the table")
    gen = as_rng(rng)
    order = isotype_order(species, level=level)
    if isotype_key is None:
        isotype_key = "isotype_subclass" if level == "subclass" else "isotype"
        if isotype_key not in air.columns:
            air = _isotype(air, species=species, c_key=c_key)
    elif isotype_key not in air.columns:
        raise KeyError(f"csr_transitions: column {isotype_key!r} not in the table")
    air, unit_key, unit_source = _resolve_units(
        air,
        unit_key=unit_key,
        order_key=order_key if null == "within" else None,
        isotype_key=isotype_key,
        radius_um=loci_radius_um,
        clone_key=clone_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        locus_key=locus_key,
        library_key=library_key,
    )
    wcols = [] if within is None else ([within] if isinstance(within, str) else list(within))
    if null == "within" and wcols:
        raise ValueError("within strata apply to null='global' only (null='within' shuffles inside clones)")
    missing = [c for c in wcols if c not in air.columns]
    if missing:
        raise KeyError(f"csr_transitions: strata column(s) {missing} not in the table")
    units = _unit_labels(
        air,
        label_key=isotype_key,
        clone_key=clone_key,
        unit_key=unit_key,
        order=order,
        order_key=order_key if null == "within" else None,
        strata=wcols,
    )
    names = sorted(set(units.label.unique()), key=lambda k: (order[k], k))
    if len(names) < 2:
        raise ValueError(f"fewer than two isotypes on the locus order in {isotype_key!r}: {names}")
    k_idx = {n: i for i, n in enumerate(names)}
    lab = units.label.map(k_idx).to_numpy(dtype=np.int64)
    clo, clones = pd.factorize(units.clone, sort=True)
    nc, kk = len(clones), len(names)
    pairs = [(a, b) for a, b in combinations(range(kk), 2) if order[names[a]] < order[names[b]]]
    pa = np.array([p[0] for p in pairs], dtype=np.int64)
    pb = np.array([p[1] for p in pairs], dtype=np.int64)

    def presence(codes: np.ndarray) -> np.ndarray:
        return np.bincount(clo * kk + codes, minlength=nc * kk).reshape(nc, kk) > 0

    pres = presence(lab)
    n_up = pres[:, pa].sum(0)
    n_down = pres[:, pb].sum(0)
    both = pres[:, pa] & pres[:, pb]
    n_both = both.sum(0)

    if null == "global":
        if not wcols:
            groups = [np.arange(len(lab))]
        else:
            groups = list(pd.Series(np.arange(len(lab))).groupby(units.stratum.to_numpy()).indices.values())

        def stat(codes: np.ndarray) -> np.ndarray:
            p = presence(codes)
            return (p[:, pa] & p[:, pb]).sum(0).astype(float)

        obs = n_both.astype(float)
        null_draws = np.empty((n_perm, len(pairs)))
        for b in range(n_perm):
            null_draws[b] = stat(_within_perm(lab, groups, gen))
    else:
        val = units.order.to_numpy(dtype=float)
        groups = [g for g in pd.Series(np.arange(len(lab))).groupby(clo).indices.values() if len(g) > 1]

        def stat(codes: np.ndarray) -> np.ndarray:
            key = clo * kk + codes
            s = np.bincount(key, weights=val, minlength=nc * kk).reshape(nc, kk)
            n = np.bincount(key, minlength=nc * kk).reshape(nc, kk)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = s / n
            d = m[:, pb] - m[:, pa]
            ok = both  # clones carrying the pair (fixed under within-clone shuffles)
            cnt = ok.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(cnt > 0, np.where(ok, d, 0.0).sum(0) / np.maximum(cnt, 1), np.nan)

        obs = stat(lab)
        null_draws = np.empty((n_perm, len(pairs)))
        for b in range(n_perm):
            null_draws[b] = stat(_within_perm(lab, groups, gen))

    with warnings.catch_warnings():  # pairs that never co-occur have all-NaN nulls under null="within"
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd, _, _ = null_summary(null_draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (obs - mu) / sd, np.nan)
        ratio = obs / mu if null == "global" else np.full(len(pairs), np.nan)
    p = mc_pvalue(null_draws, obs, alternative=alternative)
    testable = n_both > 0 if null == "within" else np.ones(len(pairs), dtype=bool)
    p = np.where(testable, p, np.nan)
    table = pd.DataFrame(
        {
            "upstream": [names[a] for a in pa],
            "downstream": [names[b] for b in pb],
            "upstream_position": [order[names[a]] for a in pa],
            "downstream_position": [order[names[b]] for b in pb],
            "n_clones_upstream": n_up.astype(np.int64),
            "n_clones_downstream": n_down.astype(np.int64),
            "n_clones_both": n_both.astype(np.int64),
            "share": np.where(n_up > 0, n_both / np.maximum(n_up, 1), np.nan),
            "statistic": obs,
            "null_mean": mu,
            "null_sd": sd,
            "z": z,
            "ratio": ratio,
            "p_value": p,
            "q_value": bh(p),
            "n_perm": n_perm,
        }
    )
    if null == "within" and int((n_both > 0).sum()) == 0:
        warnings.warn(
            "no clone carries two isotypes on the locus order: nothing to test", LowPowerWarning, stacklevel=2
        )
    summary = {
        "n_units": len(lab),
        "n_clones": nc,
        "n_clones_multi": int((pres.sum(1) >= 2).sum()),
        "n_pairs": len(pairs),
        "n_pairs_q05": int(np.nansum(table.q_value.to_numpy() <= 0.05)),
    }
    params = {
        "species": species,
        "level": level,
        "isotype_key": isotype_key,
        "unit_key": unit_key,
        "units": unit_source,
        "null": null,
        "order_key": order_key,
        "within": within,
        "n_perm": n_perm,
        "alternative": alternative,
        "design": {
            "design": "Permute (positions fixed)",
            "columns": isotype_key,
            "within": "clone" if null == "within" else within,
            "exchangeable": f"cells ({unit_source})",
        },
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(
        name="csr_transitions", table=table, summary=summary, params=params, null=null_draws if keep_null else None
    )


def csr_map(units: pd.DataFrame, *, mark: str = "switched", sigma_um: float = 50.0, **kwargs: Any):
    """Smoothed share of class-switched units (loci) over the tissue (archive T3 map).

    Parameters
    ----------
    units
        One row per unit (normally loci with the locus calls of :func:`spaclone.bcr.call_isotype`, ``by="locus"``)
        with coordinates and the switched call; units without an isotype (missing call) are left out.
    mark
        Boolean column (missing = no isotype).
    sigma_um
        Gaussian kernel standard deviation in µm (T3: 50).
    **kwargs
        Passed to :func:`spaclone.tl.share_map` (``grid``, ``tissue``, ``pixel_um``, ``min_neff``, ``match_by``,
        ``x_key``, ...).

    Returns
    -------
    spaclone.tl.ShareMap
    """
    sub = _map_units(units, mark, kwargs.get("match_by"))
    return _tl("share_map")(sub, mark=mark, sigma_um=sigma_um, **kwargs)


def csr_hotspots(
    units: pd.DataFrame,
    *,
    mark: str = "switched",
    blocks: str | None = S.CLONE,
    loci_radius_um: float | None = 15.0,
    sigma_um: float = 50.0,
    n_perm: int = 1000,
    rng: RngLike = None,
    **kwargs: Any,
):
    """Class-switch hotspots: pixels whose switched share exceeds the family-wise threshold of shuffled maps (T3).

    Parameters
    ----------
    units
        One row per unit, normally loci (one inferred cell each: :func:`spaclone.pp.merge_loci`, then
        :func:`spaclone.bcr.call_isotype` with ``by="locus"``), with coordinates, the switched call and the clone;
        units without an isotype are left out.
    mark, sigma_um
        As in :func:`csr_map`.
    blocks
        Clone column (default ``"clone_id"``): the switched call belongs to a lineage (a clone expands after
        switching), so the null relabels whole clones (see Notes). ``None``: loci are shuffled one by one, the
        2026-10 archive's design (T3), valid only when every clone has a single locus.
    loci_radius_um
        With ``blocks``: units of one clone closer than this (records or spots of one cell) are merged into one locus
        first (switched when at least half of its units are); a loci table is left as it is. ``None``: no merging.
    n_perm
        Shuffled maps (T3: 1,000).
    rng
        Seed or generator.
    **kwargs
        Passed to :func:`spaclone.tl.hotspots` (``tissue``, ``grid``, ``min_neff``, ``correction``,
        ``threshold_rule``, ``neff_kernel``, ``match_by``, ``within``, ``block_design``, ``standardise``,
        ``min_blocks``, ``min_units``, ``x_key``, ...). The archive's settings are ``blocks=None,
        neff_kernel="legacy", threshold_rule="quantile"`` (13 hotspots in T3; 12 with the exact kernel and the rank
        rule). ``min_units=1`` drops hotspots without an isotyped locus inside (:func:`spaclone.bcr.hotspot_workflow`
        does by default).

    Returns
    -------
    spaclone.tl.HotspotResult

    Notes
    -----
    Null model (default, ``blocks`` = clone, ``block_design="matched"``): positions stay in place; clones with the
    same number of isotyped loci trade their switched calls (within ``within`` / ``match_by`` strata when given), and
    inside a clone the calls are re-dealt among its loci. The switched count of the section is fixed and a clone keeps
    its own switched count unless it trades with a clone of the same size, so the null lets switching depend on clone
    size (expanded clones are switched more often) and asks whether some *place* holds more switched loci than that;
    a single expanded switched clone is one observation, not a hotspot. The map is studentised by the shuffled maps'
    pixel means and standard deviations (``standardise="permutation"``).

    Null model (``blocks=None``): the switched labels are permuted among the isotyped units, positions fixed (within
    ``within`` / ``match_by`` strata when given); a hotspot is a connected set of eligible pixels whose standardised
    share exceeds the (1 − α) quantile of the shuffled maps' maxima, so the chance of any false hotspot on the map is
    α *if the loci of a clone switch independently*, which they do not.

    References
    ----------
    Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Nichols & Holmes 2002, doi:10.1002/hbm.1058; Westfall &
    Young 1993, Resampling-Based Multiple Testing, Wiley; Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885.
    """
    sub = _map_units(units, mark, kwargs.get("match_by"), blocks)
    merged = 0
    if blocks is not None and kwargs.get("null_marks") is None:
        sub, merged = _to_loci(
            sub,
            mark=mark,
            match_by=kwargs.get("match_by"),
            blocks=blocks,
            radius_um=loci_radius_um,
            x_key=kwargs.get("x_key", S.X),
            y_key=kwargs.get("y_key", S.Y),
        )
    res = _tl("hotspots")(sub, mark=mark, blocks=blocks, sigma_um=sigma_um, n_perm=n_perm, rng=rng, **kwargs)
    res.params["units_merged_into_loci"] = merged
    return res


def region_test(
    units: pd.DataFrame,
    *,
    regions: Any,
    mark: str = "mutated",
    match_by: str | None = "isotype",
    blocks: str | None = S.CLONE,
    n_perm: int = 10_000,
    rng: RngLike = None,
    **kwargs: Any,
) -> TestResult:
    """Excess of mutated (or switched) units in given regions beyond their isotype-matched expectation (T7).

    Typical use: CSR hotspots (found with the switched mark) tested for more mutated loci than their isotype mix
    predicts, the germinal-centre signature of switching and mutating in one place.

    Parameters
    ----------
    units
        One row per unit (loci) with coordinates, the mark, the covariate and the clone; units without a call are
        left out.
    regions
        Region of each unit: a label raster or a :class:`spaclone.tl.HotspotResult` (sampled at the unit positions),
        or a column / array of region ids (0 or missing = outside).
    mark
        Boolean column tested (``"mutated"`` or ``"switched"``).
    match_by
        Covariate of the expectation and strata of the shuffles (default: the isotype class); ``None`` = global
        share. Missing classes become ``"unknown"``.
    blocks
        Clone column (default ``"clone_id"``): whole clones are relabelled (see Notes). ``None``: units are shuffled
        one by one within ``match_by`` classes, the archive's T7 design.
    n_perm
        Shuffles (T7: 10,000).
    rng
        Seed or generator.
    **kwargs
        Passed to :func:`spaclone.tl.region_excess` (``min_units``, ``alternative``, ``pooled``, ``block_design``,
        ``x_key``, ...).

    Returns
    -------
    TestResult
        One row per region (and a pooled row): ``units``, ``marked``, ``expected`` (sum of the units'
        isotype-matched probabilities), ``ratio``, ``null_mean``, ``null_sd``, ``z``, ``p_value``, ``q_value``,
        ``n_perm``.

    Notes
    -----
    Null model (default, ``blocks`` = clone, ``block_design="matched"``): positions, regions and isotypes are fixed;
    clones with the same number of loci of every isotype trade their calls, and inside a clone the calls of an
    isotype are re-dealt among its loci of that isotype. The null mean of a region is then its expected count given
    the isotypes *and* the clone sizes of its loci; ``expected`` is the isotype-matched expectation alone.

    Null model (``blocks=None``): the mark is permuted among the units of the same ``match_by`` class (positions and
    regions fixed), so the expected count of a region equals its null mean and an excess means more marked units
    than the region's isotype mix predicts; loci of one clone count as independent units.

    The regions must not have been selected with the same mark.

    References
    ----------
    Davison & Hinkley 1997, doi:10.1017/CBO9780511802843; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Hoehn
    et al. 2022, doi:10.1371/journal.pcbi.1009885.
    """
    sub = _map_units(units, mark, match_by, blocks)
    if not isinstance(regions, str) and np.ndim(regions) == 1 and len(regions) == len(units):
        regions = np.asarray(regions)[units[mark].notna().to_numpy()]
    return _tl("region_excess")(
        sub, mark=mark, regions=regions, match_by=match_by, blocks=blocks, n_perm=n_perm, rng=rng, **kwargs
    )
