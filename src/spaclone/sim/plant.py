"""Planted-truth benchmark harness: plant genuine clonal structure into a placebo background, then score callers.

:func:`plant` turns a real receptor table into a benchmark with known truth, as the archive's F1 benchmark did:

1. **Placebo background.** A null design of :mod:`spaclone.nulls` removes every clone–position link from the real
   data (default: clone identity permuted among the records of the same locus, positions and read counts fixed), so
   whatever a method calls on the background alone is false by construction.
2. **Plants.** A fraction of every clone's records is moved into cells of that clone, placed by biology: plasma-cell
   clones as foci of cells around one site (``"focus"``), germinal-centre clones inside one germinal centre
   (``"gc"``), follicular clones as single cells (``"scattered"``). Heavy and light clones are paired and share their
   cells (one heavy and one light chain per cell). Planted records are background records, so their read counts come
   from the same distribution as everything else: a caller cannot find them by read support alone.
3. The planted records are the truth (``planted`` column); :func:`score` compares calls with it.

References
----------
Weber et al. 2019, doi:10.1186/s13059-019-1738-8 (benchmarking with ground truth); Sun et al. 2020,
doi:10.1038/s41592-019-0701-7 (SPARK: planted spatial patterns); Markello & Misic 2021,
doi:10.1016/j.neuroimage.2021.118052 (the null design drives false positives).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone._utils import RngLike, as_rng, rng_entropy
from spaclone.io._container import SpatialAIR
from spaclone.io.airr import format_spots
from spaclone.nulls import NullDesign, Permute

__all__ = ["PATTERNS", "Planted", "plant", "score"]

logger = logging.getLogger("spaclone")

#: Planting patterns: records per cell and the radius (µm) of the clone's focus (``None`` = single cells anywhere).
PATTERNS: dict[str, dict[str, float | None]] = {
    "focus": {"records_per_cell": 20.0, "spread_um": 25.0},
    "gc": {"records_per_cell": 4.0, "spread_um": 40.0},
    "scattered": {"records_per_cell": 2.0, "spread_um": None},
}
#: Clone-level columns permuted together with the clone id by the default placebo design.
CLONE_FIELDS = (S.JUNCTION_AA, S.JUNCTION, S.V_CALL, "d_call", S.J_CALL, S.C_CALL, S.PRODUCTIVE)


class Planted(NamedTuple):
    """Result of :func:`plant`.

    Attributes
    ----------
    air
        The placebo background with the planted records moved into their cells; boolean column ``planted`` (the
        truth) and, for planted rows, ``implant_cell`` and ``implant_focus``.
    implants
        One row per planted record: ``row`` (position in ``air``), clone, locus, ``lineage`` (the heavy/light pair
        that shares cells), ``pattern``, ``focus`` / ``focus_x_um`` / ``focus_y_um``, ``cell`` / ``cell_x_um`` /
        ``cell_y_um``, the position before (``from_x_um`` / ``from_y_um``), the drawn position in its cell
        (``target_x_um`` / ``target_y_um``) and the final position after snapping (``x_um`` / ``y_um``).
    params
        Every argument that defines the benchmark, the background design and the generator's seed entropy.
    """

    air: pd.DataFrame
    implants: pd.DataFrame
    params: dict[str, Any]


def _sites(sites: Any, background: pd.DataFrame, x_key: str, y_key: str) -> np.ndarray:
    if sites is None:
        return background[[x_key, y_key]].drop_duplicates().to_numpy(dtype=float)
    if isinstance(sites, pd.DataFrame):
        return sites[[x_key, y_key]].to_numpy(dtype=float)
    arr = np.asarray(sites, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("sites must be a DataFrame with x / y columns or an (n, 2) array (µm)")
    return arr


def _pairs(sizes: pd.Series, loci: pd.Series, pattern_of: pd.Series, pair: bool) -> pd.Series:
    """Lineage label per clone: heavy and light clones paired by rank of size within receptor family and pattern."""
    lineage = pd.Series([f"single:{c}" for c in sizes.index], index=sizes.index, dtype=object)
    if not pair:
        return lineage
    fam = loci.map(lambda lc: "BCR" if lc in S.BCR_LOCI else ("TCR_ab" if lc in ("TRA", "TRB") else "TCR_gd"))
    heavy = loci.isin(S.HEAVY_LOCI)
    for (f, pat), idx in pd.DataFrame({"f": fam, "p": pattern_of}).groupby(["f", "p"], sort=True).groups.items():
        hs = sizes.loc[idx][heavy.loc[idx]].sort_values(ascending=False, kind="stable").index
        ls = sizes.loc[idx][~heavy.loc[idx]].sort_values(ascending=False, kind="stable").index
        for k in range(min(len(hs), len(ls))):
            lineage[hs[k]] = lineage[ls[k]] = f"{f}:{pat}:{k}"
    return lineage


def plant(
    background: pd.DataFrame | SpatialAIR,
    *,
    fraction: float,
    pattern: Literal["focus", "gc", "scattered"] | Mapping[str, str] = "focus",
    sites: pd.DataFrame | np.ndarray | Mapping[str, pd.DataFrame | np.ndarray] | None = None,
    design: NullDesign | Literal["default"] | None = "default",
    clones: Sequence[str] | None = None,
    pair: bool = True,
    records_per_cell: float | None = None,
    spread_um: float | None = None,
    cell_radius_um: float = 4.0,
    grid_um: float | None = None,
    spot_format: str = "{x}_{y}",
    slots: pd.DataFrame | None = None,
    dedupe: Sequence[str] | None = None,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> Planted:
    """Plant clonal structure with known truth into a placebo version of a receptor table.

    Parameters
    ----------
    background
        Receptor table (or a :class:`~spaclone.io.SpatialAIR`): usually real data, whose density, read counts and
        clone sizes the benchmark keeps.
    fraction
        Share of each clone's records that is planted (a binomial draw per clone), e.g. 0.001–0.1.
    pattern
        ``"focus"`` (plasma-cell focus: 20 records per cell, cells within 25 µm of one site), ``"gc"`` (germinal
        centre: 4 per cell, within 40 µm), ``"scattered"`` (follicular B cells: 2 per cell, each cell at its own
        site), or ``{clone_id: pattern}`` (clones not listed use ``"scattered"``).
    sites
        Candidate centres in µm (``DataFrame`` with ``x_key`` / ``y_key`` or an ``(n, 2)`` array), or
        ``{pattern: sites}``. Default: the background's record positions. Pass the places where the biology happens
        (plasma-cell areas, germinal centres, mature-B areas) for a realistic benchmark.
    design
        Placebo design applied to the background before planting: ``"default"`` permutes the clone id together with
        its sequence fields (those of ``junction_aa``, ``junction``, ``v_call``, ``d_call``, ``j_call``, ``c_call``
        and ``productive`` present in the table) within each locus; any
        :class:`~spaclone.nulls.NullDesign`; ``None`` keeps the background as it is (already a null).
    clones
        Plant only these clones (default: every clone).
    pair
        Pair heavy and light clones (by rank of size within receptor family and pattern) so that they share cells.
    records_per_cell, spread_um
        Override the pattern's records per cell and focus radius.
    cell_radius_um
        A planted record lands uniformly within this radius of its cell's centre.
    grid_um, spot_format
        Snap planted positions to a grid of this pitch (e.g. 0.5 for Stereo-seq DNBs; ``spot_id`` =
        ``spot_format`` over the integer grid indices, and ``x_src`` / ``y_src`` are updated when present).
    slots
        Snap planted positions to the nearest of these spots instead (``spot_key``, ``x_key``, ``y_key``; e.g. all
        beads); any other slot columns present in the receptor table (``in_tissue``, ``cell_type``, ...) are copied.
        Without ``grid_um`` or ``slots`` the positions stay continuous and every planted record gets its own spot id.
    dedupe
        Columns that identify one unit (e.g. ``("spot_id", "locus", "clone_id")`` for (DNB, clone) units without
        UMIs): after shuffling and after planting, later duplicates are dropped (a planted record landing on a spot
        that already holds its clone is the same unit).
    rng
        Seed or generator (same seed ⇒ same benchmark).
    spot_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    Planted
        ``air`` (with the boolean truth column ``planted``), ``implants`` and ``params``.

    Notes
    -----
    Null model of the background: with the default design, clone identity (the clone id and its sequence fields) is
    exchanged at random between records of the same locus, positions, read counts, UMIs and every spot attribute
    fixed (random labelling within chain, :class:`~spaclone.nulls.Permute`). This keeps the data's spatial density,
    read distribution and clone-size distribution and removes every link between clone and position, so anything a
    method calls on the background is a false positive.

    The planted records keep all their attributes except position; GEX evidence at the new positions (same-spot
    hits) is not planted. To benchmark same-spot tests use :func:`spaclone.sim.synthetic_chip`, or recompute the GEX
    lookup at the planted positions.

    References
    ----------
    Besag & Diggle 1977, doi:10.2307/2346974 (Monte-Carlo tests by random labelling); Weber et al. 2019,
    doi:10.1186/s13059-019-1738-8 (benchmarks with ground truth).
    """
    air0 = background.air if isinstance(background, SpatialAIR) else background
    if not isinstance(air0, pd.DataFrame):
        raise TypeError("background must be a receptor table or a SpatialAIR")
    for c in (spot_key, x_key, y_key, locus_key, clone_key):
        if c not in air0.columns:
            raise KeyError(f"plant: column {c!r} not in the background")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if isinstance(pattern, str) and pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)} or a mapping, got {pattern!r}")
    if grid_um is not None and slots is not None:
        raise ValueError("give grid_um or slots, not both")
    gen = as_rng(rng)
    air = air0.reset_index(drop=True).copy()
    if design == "default":
        cols = [clone_key, *[c for c in CLONE_FIELDS if c in air.columns]]
        design = Permute(cols, within=locus_key)
    if design is not None:
        air = design(air, gen).reset_index(drop=True)
    if dedupe is not None:
        air = air.drop_duplicates(list(dedupe), keep="first").reset_index(drop=True)
    air["planted"] = False
    air["implant_cell"] = pd.Series(np.full(len(air), np.nan, dtype=object), dtype=object)
    air["implant_focus"] = pd.Series(np.full(len(air), np.nan, dtype=object), dtype=object)

    by_clone = air.groupby(clone_key, sort=True).indices
    if clones is not None:
        wanted = {str(c) for c in clones}
        by_clone = {k: v for k, v in by_clone.items() if str(k) in wanted}
    sizes = pd.Series({k: len(v) for k, v in by_clone.items()}, dtype=np.int64)
    loci = pd.Series({k: air.at[int(v[0]), locus_key] for k, v in by_clone.items()}, dtype=object)
    if isinstance(pattern, Mapping):
        pattern_of = pd.Series({k: str(pattern.get(k, "scattered")) for k in by_clone}, dtype=object)
        bad = set(pattern_of) - set(PATTERNS)
        if bad:
            raise ValueError(f"unknown pattern(s) {sorted(bad)}")
    else:
        pattern_of = pd.Series(dict.fromkeys(by_clone, pattern), dtype=object)
    lineage = _pairs(sizes, loci, pattern_of, pair) if len(sizes) else pd.Series(dtype=object)
    site_sets = {}
    for pat in set(pattern_of):
        src = sites.get(pat) if isinstance(sites, Mapping) else sites
        site_sets[pat] = _sites(src, air0, x_key, y_key)
        if len(site_sets[pat]) == 0:
            raise ValueError(f"no sites for pattern {pat!r}")

    # draw the planted records of every clone (binomial share), grouped by lineage
    chosen: dict[str, list[tuple[str, np.ndarray]]] = {}
    for k, idx in by_clone.items():
        n = int(gen.binomial(len(idx), fraction))
        if n:
            chosen.setdefault(str(lineage[k]), []).append((k, gen.choice(idx, n, replace=False)))
    x0, y0 = air[x_key].to_numpy(dtype=float), air[y_key].to_numpy(dtype=float)
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("row", "clone", "lineage", "pattern", "cell", "focus", "fxy", "cxy")
    }
    new_xy = np.zeros((len(air), 2))
    for lin, members in sorted(chosen.items()):
        pat = str(pattern_of[members[0][0]])
        rpc = float(records_per_cell if records_per_cell is not None else PATTERNS[pat]["records_per_cell"])
        spr = spread_um if spread_um is not None else PATTERNS[pat]["spread_um"]
        n_rec = sum(len(i) for _, i in members)
        n_cells = int(np.ceil(n_rec / max(rpc, 1e-9)))
        sset = site_sets[pat]
        if spr is None:  # single cells at their own sites
            centres = sset[gen.integers(0, len(sset), n_cells)]
            has_focus = False
            fxy = np.full(2, np.nan)
        else:
            fxy = sset[gen.integers(0, len(sset))]
            r = float(spr) * np.sqrt(gen.random(n_cells))
            a = gen.uniform(0, 2 * np.pi, n_cells)
            centres = fxy + np.column_stack([r * np.cos(a), r * np.sin(a)])
            has_focus = True
        for k, idx in members:
            cell = gen.integers(0, n_cells, len(idx))
            r = cell_radius_um * np.sqrt(gen.random(len(idx)))
            a = gen.uniform(0, 2 * np.pi, len(idx))
            new_xy[idx] = centres[cell] + np.column_stack([r * np.cos(a), r * np.sin(a)])
            acc["row"].append(idx)
            acc["clone"].append(np.full(len(idx), k, dtype=object))
            acc["lineage"].append(np.full(len(idx), lin, dtype=object))
            acc["pattern"].append(np.full(len(idx), pat, dtype=object))
            acc["cell"].append(np.char.add(f"{lin}:cell", cell.astype(str)).astype(object))
            acc["focus"].append(np.full(len(idx), f"{lin}:focus" if has_focus else np.nan, dtype=object))
            acc["fxy"].append(np.repeat(np.asarray(fxy, float)[None, :], len(idx), axis=0))
            acc["cxy"].append(centres[cell])
    if acc["row"]:
        row = np.concatenate(acc["row"])
        fxy_all, cxy_all = np.vstack(acc["fxy"]), np.vstack(acc["cxy"])
        clone_all = np.concatenate(acc["clone"])
        imp = pd.DataFrame(
            {
                "row": row,
                clone_key: clone_all,
                locus_key: loci.reindex(clone_all).to_numpy(dtype=object),
                "lineage": np.concatenate(acc["lineage"]),
                "pattern": np.concatenate(acc["pattern"]),
                "focus": np.concatenate(acc["focus"]),
                "focus_x_um": fxy_all[:, 0],
                "focus_y_um": fxy_all[:, 1],
                "cell": np.concatenate(acc["cell"]),
                "cell_x_um": cxy_all[:, 0],
                "cell_y_um": cxy_all[:, 1],
                "from_x_um": x0[row],
                "from_y_um": y0[row],
            }
        )
    else:
        cols = ["row", clone_key, locus_key, "lineage", "pattern", "focus", "focus_x_um", "focus_y_um", "cell"]
        imp = pd.DataFrame(columns=[*cols, "cell_x_um", "cell_y_um", "from_x_um", "from_y_um"])
    if len(imp):
        idx = imp["row"].to_numpy(dtype=np.int64)
        xy = new_xy[idx]
        imp["target_x_um"], imp["target_y_um"] = xy[:, 0], xy[:, 1]
        if slots is not None:
            st = slots.reset_index(drop=True)
            _, j = cKDTree(st[[x_key, y_key]].to_numpy(dtype=float)).query(xy)
            air.loc[idx, spot_key] = st[spot_key].to_numpy(dtype=object)[j]
            air.loc[idx, x_key] = st[x_key].to_numpy(dtype=float)[j]
            air.loc[idx, y_key] = st[y_key].to_numpy(dtype=float)[j]
            for c in st.columns:
                if c not in (spot_key, x_key, y_key) and c in air.columns:
                    air.loc[idx, c] = st[c].to_numpy()[j]
        elif grid_um is not None:
            ij = np.rint(xy / grid_um).astype(np.int64)
            air.loc[idx, spot_key] = format_spots(ij[:, 0], ij[:, 1], spot_format)
            air.loc[idx, x_key] = ij[:, 0] * grid_um
            air.loc[idx, y_key] = ij[:, 1] * grid_um
            for c, k in (("x_src", 0), ("y_src", 1)):
                if c in air.columns:
                    air.loc[idx, c] = ij[:, k]
        else:
            air.loc[idx, x_key] = xy[:, 0]
            air.loc[idx, y_key] = xy[:, 1]
            air.loc[idx, spot_key] = np.array([f"planted:{i}" for i in idx], dtype=object)
        air.loc[idx, "planted"] = True
        air.loc[idx, "implant_cell"] = imp["cell"].to_numpy(dtype=object)
        air.loc[idx, "implant_focus"] = imp["focus"].to_numpy(dtype=object)
        imp["x_um"] = air.loc[idx, x_key].to_numpy(dtype=float)
        imp["y_um"] = air.loc[idx, y_key].to_numpy(dtype=float)
    n_dropped = 0
    if dedupe is not None and len(imp):
        dup = air.duplicated(list(dedupe), keep="first").to_numpy()
        n_dropped = int(dup[imp["row"].to_numpy(dtype=np.int64)].sum())
        keep_rows = np.flatnonzero(~dup)
        remap = pd.Series(np.arange(len(keep_rows)), index=keep_rows)
        imp = imp[~dup[imp["row"].to_numpy(dtype=np.int64)]].copy()
        air = air.iloc[keep_rows].reset_index(drop=True)
        imp["row"] = remap.reindex(imp["row"].to_numpy()).to_numpy(dtype=np.int64)
    air["planted"] = air["planted"].astype(bool)
    params = {
        "fraction": float(fraction),
        "pattern": pattern if isinstance(pattern, str) else dict(pattern),
        "design": design.describe() if design is not None else None,
        "pair": pair,
        "records_per_cell": records_per_cell,
        "spread_um": spread_um,
        "cell_radius_um": cell_radius_um,
        "grid_um": grid_um,
        "slots": None if slots is None else len(slots),
        "dedupe": None if dedupe is None else list(dedupe),
        "clones": None if clones is None else len(clones),
        "n_records": int(len(air)),
        "n_planted": int(air["planted"].sum()),
        "n_planted_dropped_as_duplicates": n_dropped,
        "rng_entropy": rng_entropy(gen),
    }
    logger.info("plant: %d of %d records planted (%s)", params["n_planted"], len(air), params["pattern"])
    return Planted(air=air, implants=imp.reset_index(drop=True), params=params)


def _auprc(score: np.ndarray, truth: np.ndarray) -> float:
    """Average precision: mean precision at the ranks of the true records (ties broken by input order)."""
    if truth.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    t = truth[o]
    prec = np.cumsum(t) / np.arange(1, len(t) + 1)
    return float((prec * t).sum() / t.sum())


def _as_frame(x: Any, n: int, name: str) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        df = x.reset_index(drop=True)
    elif isinstance(x, Mapping):
        df = pd.DataFrame({str(k): np.asarray(v) for k, v in x.items()})
    else:
        df = pd.DataFrame({name: np.asarray(x)})
    if len(df) != n:
        raise ValueError(f"{name}: {len(df)} rows for {n} truth values")
    return df


def score(
    calls: Any,
    truth: Any,
    *,
    scores: Any = None,
    by: Any = None,
) -> pd.DataFrame:
    """Score the calls of one or several methods against the planted (or simulated) truth.

    Parameters
    ----------
    calls
        Boolean calls, one per record: an array, or a ``DataFrame`` / ``dict`` with one column per method.
    truth
        Boolean truth, one per record (e.g. ``planted.air["planted"]`` or ``sa.air["truth"] == "genuine"``).
    scores
        Optional continuous scores (higher = more likely true) with the same layout as ``calls``, for the area
        under the precision–recall curve.
    by
        Optional strata (one label per record, e.g. the locus); metrics are reported per stratum.

    Returns
    -------
    pandas.DataFrame
        One row per method (and stratum): ``n``, ``n_true``, ``calls``, ``TP``, ``FP``, ``FN``, ``precision``
        (NaN without calls), ``recall`` (NaN without truth), ``F1``, ``FDR``, ``AUPRC`` (average precision; NaN
        without scores).

    Examples
    --------
    >>> score([True, True, False, False], [True, False, True, False])[["TP", "FP", "FN", "precision", "recall"]]
       TP  FP  FN  precision  recall
    0   1   1   1        0.5     0.5
    """
    t = np.asarray(truth)
    if t.dtype != bool:
        t = pd.Series(t).fillna(False).astype(bool).to_numpy()
    n = len(t)
    C = _as_frame(calls, n, "calls")
    Sc = _as_frame(scores, n, "calls") if scores is not None else None
    if Sc is not None and isinstance(calls, np.ndarray | list | pd.Series) and Sc.columns.tolist() == ["calls"]:
        Sc.columns = C.columns
    strata = [("all", np.ones(n, bool))] if by is None else [
        (k, np.asarray(by) == k) for k in pd.unique(np.asarray(by))
    ]  # fmt: skip
    out = []
    for meth in C.columns:
        c = C[meth].fillna(False).to_numpy(dtype=bool)
        s = Sc[meth].to_numpy(dtype=float) if Sc is not None and meth in Sc.columns else None
        for k, m in strata:
            cm, tm = c[m], t[m]
            tp, fp, fn = int((cm & tm).sum()), int((cm & ~tm).sum()), int((~cm & tm).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
            row = {"method": meth}
            if by is not None:
                row["stratum"] = k
            row.update(
                {
                    "n": int(m.sum()),
                    "n_true": int(tm.sum()),
                    "calls": int(cm.sum()),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": prec,
                    "recall": rec,
                    "F1": f1,
                    "FDR": fp / (tp + fp) if tp + fp else np.nan,
                    "AUPRC": _auprc(s[m], tm) if s is not None else np.nan,
                }
            )
            out.append(row)
    res = pd.DataFrame(out)
    return res.drop(columns="method") if (C.columns.tolist() == ["calls"] and by is None) else res
