"""Per-clone spatial dispersion: are a clone's units closer together than the same number of random units?

For every clone with enough units, two statistics of its unit positions are compared with the same statistics of the
same number of units drawn at random from all units of its section and chain (the archive's S1 "clustered" rule):

* ``mean_nn``: the mean distance from each unit to its nearest other unit of the clone (Clark & Evans 1954), sensitive
  to local clumping;
* ``mst``: the total length of the Euclidean minimum spanning tree of the units, sensitive to the overall extent (a
  clone split between two distant places has one long edge).

Two pitfalls are handled. (1) With hundreds of clones, a Monte-Carlo p-value from a few hundred draws has a floor of
1 / (B + 1) that can never pass Benjamini-Hochberg (BH) at level alpha once there are more than about alpha * B clones; the
default therefore samples sequentially (Besag & Clifford 1991): a clone stops once ``h`` draws are at least as extreme as
the observed value, and clones that stay extreme are sampled up to ``max_perm`` = m / alpha draws, the smallest number
at which a single clone can pass BH. (2) Loci (merged same-clone spots) of one clone are by construction more than
``radius_um`` apart; random loci of different clones are not, so the default null for loci draws random units by
sequential inhibition with that minimum separation.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

from spaclone import _schema as S
from spaclone._utils import LowPowerWarning, RngLike, TestResult, as_rng, bh, check_alternative, mc_pvalue, rng_entropy
from spaclone.nulls import NullDesign
from spaclone.nulls import placebo as run_placebo
from spaclone.tl.repertoire import _as_list, _groups, _keys, clone_units

__all__ = ["DISPERSION_MEASURES", "clone_dispersion"]

#: Dispersion statistics of :func:`clone_dispersion`.
DISPERSION_MEASURES: tuple[str, ...] = ("mean_nn", "mst")
_DENSE_MST = 2000


def _mean_nn(xy: np.ndarray) -> float:
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1].mean()


def _mst_length(xy: np.ndarray) -> float:
    pts = np.unique(np.asarray(xy, dtype=float), axis=0)
    n = len(pts)
    if n < 2:
        return 0.0
    if n <= _DENSE_MST:
        return float(minimum_spanning_tree(squareform(pdist(pts))).sum())
    try:
        tri = Delaunay(pts)
    except QhullError:  # all points on one line: the tree is the path along the line
        axis = pts - pts.mean(0)
        _, _, vt = np.linalg.svd(axis, full_matrices=False)
        o = np.argsort(axis @ vt[0])
        return float(np.sqrt((np.diff(pts[o], axis=0) ** 2).sum(1)).sum())
    s = tri.simplices
    e = np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), axis=1)
    e = np.unique(e, axis=0)
    w = np.sqrt(((pts[e[:, 0]] - pts[e[:, 1]]) ** 2).sum(1))
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return float(minimum_spanning_tree(g).sum())


_MEASURE_FUNCS = {"mean_nn": _mean_nn, "mst": _mst_length}
_BATCH_N = 256  # point sets up to this size are measured in vectorised batches


def _prim(d: np.ndarray) -> np.ndarray:
    """Minimum-spanning-tree length of each (n x n) distance matrix in a (B, n, n) stack (Prim's algorithm)."""
    b, n, _ = d.shape
    rows = np.arange(b)
    in_tree = np.zeros((b, n), bool)
    in_tree[:, 0] = True
    key = d[:, 0, :].copy()
    total = np.zeros(b)
    for _ in range(n - 1):
        j = np.argmin(np.where(in_tree, np.inf, key), axis=1)
        total += key[rows, j]
        in_tree[rows, j] = True
        np.minimum(key, d[rows, j, :], out=key)
    return total


def _batch_measures(pts: np.ndarray, meas: list[str]) -> np.ndarray:
    """Measures of a (B, n, 2) stack of point sets; returns (B, len(meas))."""
    b, n, _ = pts.shape
    out = np.empty((b, len(meas)))
    if n > _BATCH_N:
        for r in range(b):
            out[r] = [_MEASURE_FUNCS[m](pts[r]) for m in meas]
        return out
    chunk = max(1, int(4e6 // (n * n)))
    eye = np.eye(n, dtype=bool)
    for s in range(0, b, chunk):
        q = pts[s : s + chunk]
        d = np.sqrt((np.square(q[:, :, None, :] - q[:, None, :, :])).sum(-1))
        for k, m in enumerate(meas):
            if m == "mean_nn":
                out[s : s + chunk, k] = np.where(eye, np.inf, d).min(axis=2).mean(axis=1)
            else:
                out[s : s + chunk, k] = _prim(d)
    return out


def _measures(xy: np.ndarray, meas: list[str]) -> np.ndarray:
    return _batch_measures(np.asarray(xy, dtype=float)[None], meas)[0]


class _Frame(NamedTuple):
    xy: np.ndarray
    within: np.ndarray | None
    pos_by_within: list[np.ndarray] | None


def _greedy(cand: np.ndarray, frame: _Frame, n: int, quota: np.ndarray | None, min_sep: float) -> np.ndarray | None:
    """Sequential inhibition over candidates in the given (random) order; None if fewer than needed are accepted."""
    pairs = cKDTree(frame.xy[cand]).query_pairs(min_sep, output_type="ndarray")
    if quota is None:
        rejected = np.zeros(len(cand), bool)
        if len(pairs):
            pairs = np.sort(pairs, axis=1)
            for i, j in pairs[np.argsort(pairs[:, 1], kind="stable")]:
                if not rejected[i]:
                    rejected[j] = True
        acc = np.flatnonzero(~rejected)
        return cand[acc[:n]] if len(acc) >= n else None
    nbrs: list[list[int]] = [[] for _ in range(len(cand))]
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)
    left = np.array(quota, dtype=np.int64)
    blocked = np.zeros(len(cand), bool)
    out = []
    w_of = frame.within[cand]
    for c in range(len(cand)):
        w = w_of[c]
        if blocked[c] or left[w] == 0:
            continue
        out.append(c)
        left[w] -= 1
        blocked[nbrs[c]] = True
        if left.sum() == 0:
            return cand[np.array(out)]
    return None


def _ssi(
    gen: np.random.Generator, frame: _Frame, n: int, quota: np.ndarray | None, min_sep: float
) -> np.ndarray | None:
    f = len(frame.xy)
    if quota is None:
        # the first n candidates of the random order; when none of them conflict, inhibition accepts exactly these
        cand = gen.choice(f, n, replace=False)
        if not len(cKDTree(frame.xy[cand]).query_pairs(min_sep, output_type="ndarray")):
            return cand
    else:
        cand = gen.choice(f, min(f, 2 * n + 16), replace=False)
    m = len(cand)
    while True:
        acc = _greedy(cand, frame, n, quota, min_sep)
        if acc is not None:
            return acc
        if m >= f:
            return None
        rest = np.setdiff1d(np.arange(f), cand)
        cand = np.concatenate([cand, gen.choice(rest, min(len(rest), max(m, n)), replace=False)])
        m = len(cand)


def _draw(
    gen: np.random.Generator, frame: _Frame, n: int, quota: np.ndarray | None, min_sep: float
) -> np.ndarray | None:
    if min_sep > 0:
        return _ssi(gen, frame, n, quota, min_sep)
    if quota is None:
        return gen.choice(len(frame.xy), n, replace=False)
    return np.concatenate([gen.choice(frame.pos_by_within[w], q, replace=False) for w, q in enumerate(quota) if q > 0])


def _pool(
    gen: np.random.Generator, frame: _Frame, n: int, quota: np.ndarray | None, min_sep: float, b: int, meas: list[str]
) -> np.ndarray:
    idx = np.zeros((b, n), np.int64)
    ok = np.ones(b, bool)
    for r in range(b):
        d = _draw(gen, frame, n, quota, min_sep)
        if d is None:
            ok[r] = False
        else:
            idx[r] = d
    out = np.full((b, len(meas)), np.nan)
    if ok.any():
        out[ok] = _batch_measures(frame.xy[idx[ok]], meas)
    return out


def _exceed(pool: np.ndarray, obs: float, alternative: str) -> np.ndarray:
    tol = 1e-12 * abs(obs)
    with np.errstate(invalid="ignore"):
        return (pool <= obs + tol) if alternative == "less" else (pool >= obs - tol)


def _besag_clifford(pool: np.ndarray, obs: float, alternative: str, h: int) -> tuple[float, int]:
    """Sequential p-value: h / L at the h-th exceedance, else (g + 1) / (B + 1)."""
    finite = pool[np.isfinite(pool)]
    e = _exceed(finite, obs, alternative)
    hits = np.flatnonzero(e)
    if len(hits) >= h:
        big_l = int(hits[h - 1]) + 1
        return h / big_l, big_l
    return (len(hits) + 1) / (len(finite) + 1), len(finite)


def _dispersion_statistic(
    df: pd.DataFrame,
    *,
    cands: list[tuple[tuple, str]],
    unit: str,
    strata: list[str],
    meas: list[str],
    radius_um: float,
    keys: dict[str, str],
) -> np.ndarray:
    u = clone_units(df, unit=unit, radius_um=radius_um, **keys).units
    xy = u[[keys["x_key"], keys["y_key"]]].to_numpy(dtype=float)
    clone = u[keys["clone_key"]].to_numpy()
    at: dict[tuple, np.ndarray] = {}
    for sk, sp in zip(*_groups(u, strata), strict=True):
        for cl, cpos in pd.Series(sp).groupby(clone[sp]).indices.items():
            at[(sk, cl)] = sp[cpos]
    out = np.full((len(cands), len(meas)), np.nan)
    for i, c in enumerate(cands):
        p = at.get(c)
        if p is not None and len(p) >= 2:
            out[i] = _measures(xy[p], meas)
    return out.ravel()


def clone_dispersion(
    air: pd.DataFrame,
    *,
    unit: Literal["locus", "spot", "record"] = "locus",
    min_units: int = 5,
    n_perm: int = 200,
    design: NullDesign | None = None,
    within: str | Sequence[str] | None = None,
    measures: Sequence[str] = ("mean_nn", "mst"),
    alternative: Literal["less", "greater", "two-sided"] = "less",
    p_method: Literal["mc", "normal"] = "mc",
    sequential: bool = True,
    max_perm: int | None = None,
    h: int = 10,
    alpha: float = 0.05,
    min_separation_um: float | None = None,
    radius_um: float = 15.0,
    n_jobs: int = 1,
    rng: RngLike = None,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> TestResult:
    """Test every clone with at least ``min_units`` units for spatial clustering of its units.

    Parameters
    ----------
    air
        Receptor table with spot ids, coordinates in µm and clone ids.
    unit
        ``"locus"`` (cell-scale units, default), ``"spot"`` (one clone on one spot counts once; the archive's S1 unit)
        or ``"record"`` (see :func:`spaclone.tl.clone_units`).
    min_units
        Smallest number of units a clone needs to be tested (>= 2).
    n_perm
        Random draws per clone size before any sequential extension.
    design
        ``None``: for a clone with n units, n units are drawn without replacement from all units of its section and
        chain (for spot units: from the distinct spots of the section and chain); see ``within`` and
        ``min_separation_um``. Otherwise any :class:`~spaclone.nulls.NullDesign`, applied to the receptor records; the
        units and the statistics of the same clones are then rebuilt on each randomised table (no sequential stopping).
    within
        Column(s) carried by the units (for example a depth stratum or a region): a clone with n_s units in stratum s is
        compared with n_s random units of stratum s, for every s (stratified random labelling).
    measures
        Statistics among :data:`DISPERSION_MEASURES`.
    alternative
        ``"less"`` (default): clustered, the clone's statistic is smaller than at random; ``"greater"``: dispersed.
    p_method
        ``"mc"``: Monte-Carlo p-value (Besag-Clifford when ``sequential``); ``"normal"``: one-sided normal p-value of
        z = (observed - null mean) / null sd (ddof 1), the archive's S1 rule, which assumes a normal null.
    sequential
        Besag-Clifford sequential sampling (one-sided alternatives and ``p_method="mc"`` only): the draws of a clone
        size are doubled until every clone of that size has ``h`` exceedances or ``max_perm`` draws are reached.
    max_perm
        Largest number of draws per clone size; ``None`` = ceil(m / alpha) for m tested clones (at least ``n_perm``),
        the smallest number at which a single clone with no exceedance can pass BH at ``alpha``.
    h
        Exceedances that stop the sampling of a clone.
    alpha
        Level of the ``significant`` flag (BH q-value < alpha) and of the default ``max_perm``.
    min_separation_um
        Minimum distance between the random units of one draw (sequential inhibition); ``None`` = ``radius_um`` for
        locus units (same-clone loci are more than ``radius_um`` apart by construction) and 0 otherwise.
    radius_um
        Linkage distance of locus units in µm.
    n_jobs
        Worker processes for a ``design`` placebo.
    rng
        Seed or generator. Without ``design``, draws are taken in a fixed order: sections and chains in sorted order,
        clones in sorted id order, and ``n_perm`` draws for a clone size when that size is first met; sequential
        extensions follow in rounds.
    clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per (clone, measure) with the section / chain columns, the clone, ``measure``,
        ``n_units``, ``n_records``, ``x_um`` / ``y_um`` (centroid of the clone's units), ``statistic`` (µm),
        ``expected`` / ``null_mean``, ``null_sd``, ``ratio`` (statistic / null mean), ``z``, ``p_value``,
        ``q_value`` (BH over the clones, per measure), ``n_perm`` (draws used: the stopping point of the clone under
        sequential sampling) and ``significant`` (q < alpha).

    Notes
    -----
    Null model (``design=None``): conditional on the units of each section and chain and on the clone's number of units
    (per ``within`` stratum), the clone's units are a uniform random subset of the section's units of that chain -- or,
    with a minimum separation, a random sequential-inhibition sample of them. Positions are fixed; only which units
    carry the clone label is random (random labelling). Null draws of one clone size are shared by all clones of that
    size, as in the archive.

    The normal p-value of ``p_method="normal"`` is accurate only when the null is close to normal (many units); for
    small clones the null of the mean nearest-neighbour distance is skewed. Monte-Carlo p-values with sequential
    stopping are exact under the null at any size.

    References
    ----------
    Clark & Evans 1954, doi:10.2307/1931034 (nearest-neighbour distance); Kruskal 1956,
    doi:10.1090/S0002-9939-1956-0078686-7 (minimum spanning tree); Diggle 2013, doi:10.1201/b15326 (random
    labelling, sequential inhibition); Besag & Clifford 1991, doi:10.1093/biomet/78.2.301 (sequential Monte-Carlo
    p-values); Gandy & Hahn 2014, doi:10.1111/sjos.12085 (Monte-Carlo multiple testing); Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    if p_method not in ("mc", "normal"):
        raise ValueError(f"p_method must be 'mc' or 'normal', got {p_method!r}")
    meas = list(dict.fromkeys(_as_list(measures)))
    bad = [m for m in meas if m not in DISPERSION_MEASURES]
    if bad or not meas:
        raise ValueError(f"unknown measure(s) {bad}; use {DISPERSION_MEASURES}")
    if min_units < 2:
        raise ValueError("min_units must be >= 2")
    if n_perm < 2:
        raise ValueError("n_perm must be >= 2")
    if h < 1:
        raise ValueError("h must be >= 1")
    within_cols = _as_list(within)
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    gen = as_rng(rng)
    u = clone_units(air, unit=unit, carry=within_cols, radius_um=radius_um, **keys).units
    if x_key not in u.columns or y_key not in u.columns:
        raise KeyError(f"clone_dispersion needs coordinates ({x_key!r}, {y_key!r})")
    strata = [c for c in (library_key, locus_key) if c in u.columns]
    min_sep = (radius_um if unit == "locus" else 0.0) if min_separation_um is None else float(min_separation_um)
    xy_all = u[[x_key, y_key]].to_numpy(dtype=float)
    clone_all = u[clone_key].to_numpy()
    wcode_all = (
        pd.factorize(pd.Series(list(map(tuple, u[within_cols].to_numpy(dtype=object))), dtype=object))[0]
        if within_cols
        else None
    )
    n_w = int(wcode_all.max()) + 1 if wcode_all is not None and len(wcode_all) else 0
    skeys, spos = _groups(u, strata)
    cands: list[dict[str, Any]] = []
    frames: list[_Frame] = []
    pools: dict[tuple, np.ndarray] = {}
    pool_frame: dict[tuple, tuple[int, int, np.ndarray | None]] = {}
    for si, (sk, sp) in enumerate(zip(skeys, spos, strict=True)):
        if unit == "spot":
            spots = u[spot_key].to_numpy()[sp]
            fpos = sp[~pd.Series(spots).duplicated().to_numpy()]
        else:
            fpos = sp
        fw = wcode_all[fpos] if wcode_all is not None else None
        frame = _Frame(
            xy_all[fpos],
            fw,
            [np.flatnonzero(fw == w) for w in range(n_w)] if fw is not None else None,
        )
        frames.append(frame)
        cl = pd.Series(clone_all[sp])
        for clone, cpos in sorted(cl.groupby(cl.to_numpy(), sort=False).indices.items(), key=lambda kv: kv[0]):
            n = len(cpos)
            if n < min_units:
                continue
            p = sp[cpos]
            quota = np.bincount(wcode_all[p], minlength=n_w) if wcode_all is not None else None
            pkey = (si, n) if quota is None else (si, tuple(quota))
            obs = _measures(xy_all[p], meas)
            if design is None and pkey not in pools:
                pools[pkey] = _pool(gen, frame, n, quota, min_sep, n_perm, meas)
                pool_frame[pkey] = (si, n, quota)
            cands.append(
                {
                    "stratum": sk,
                    "clone": clone,
                    "pkey": pkey,
                    "obs": obs,
                    "n_units": n,
                    "n_records": int(u["n_records"].to_numpy()[p].sum()),
                    "x": float(xy_all[p, 0].mean()),
                    "y": float(xy_all[p, 1].mean()),
                }
            )
    m_tests = len(cands)
    cap = max(n_perm, math.ceil(m_tests / alpha)) if max_perm is None else int(max_perm)
    do_seq = design is None and p_method == "mc" and sequential and alternative != "two-sided"
    if design is None and p_method == "mc" and m_tests and not do_seq and 1 / (n_perm + 1) > alpha / m_tests:
        warnings.warn(
            f"with {m_tests} clones the smallest Monte-Carlo p-value 1/{n_perm + 1} cannot pass BH at {alpha}; "
            "use sequential=True or n_perm >= m / alpha",
            LowPowerWarning,
            stacklevel=2,
        )
    if do_seq:
        by_key: dict[tuple, list[int]] = {}
        for i, c in enumerate(cands):
            by_key.setdefault(c["pkey"], []).append(i)
        while True:
            todo = []
            for pkey, members in by_key.items():
                pool = pools[pkey]
                if len(pool) >= cap:
                    continue
                if any(
                    _exceed(pool[:, k], cands[i]["obs"][k], alternative).sum() < h
                    for i in members
                    for k in range(len(meas))
                ):
                    todo.append(pkey)
            if not todo:
                break
            for pkey in todo:
                si, n, quota = pool_frame[pkey]
                add = min(cap, 2 * len(pools[pkey])) - len(pools[pkey])
                pools[pkey] = np.vstack([pools[pkey], _pool(gen, frames[si], n, quota, min_sep, add, meas)])
    rows = []
    null_arr = None
    if design is not None and m_tests:
        cand_keys = [(c["stratum"], c["clone"]) for c in cands]
        stat = functools.partial(
            _dispersion_statistic, cands=cand_keys, unit=unit, strata=strata, meas=meas, radius_um=radius_um, keys=keys
        )
        res = run_placebo(
            stat, air, design, n_perm=n_perm, alternative=alternative, rng=gen, n_jobs=n_jobs, name="clone_dispersion"
        )
        null_arr = res.null
    for i, c in enumerate(cands):
        for k, m in enumerate(meas):
            if design is not None:
                pool = null_arr[:, i * len(meas) + k]
            else:
                pool = pools[c["pkey"]][:, k]
            obs = c["obs"][k]
            fin = pool[np.isfinite(pool)]
            mu = float(fin.mean()) if len(fin) else np.nan
            sd = float(fin.std(ddof=1)) if len(fin) > 1 else np.nan
            z = (obs - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
            if p_method == "normal":
                if alternative == "less":
                    p = float(norm.cdf(z))
                elif alternative == "greater":
                    p = float(norm.sf(z))
                else:
                    p = float(min(1.0, 2 * min(norm.cdf(z), norm.sf(z))))
                used = len(fin)
            elif do_seq:
                p, used = _besag_clifford(pool, obs, alternative, h)
            else:
                p = float(mc_pvalue(pool, obs, alternative=alternative)[0])
                used = len(fin)
            row = dict(zip(strata, c["stratum"], strict=True))
            row.update(
                {
                    clone_key: c["clone"],
                    "measure": m,
                    "n_units": c["n_units"],
                    "n_records": c["n_records"],
                    x_key: c["x"],
                    y_key: c["y"],
                    "statistic": obs,
                    "expected": mu,
                    "null_mean": mu,
                    "null_sd": sd,
                    "ratio": obs / mu if mu else np.nan,
                    "z": z,
                    "p_value": p,
                    "n_perm": used,
                }
            )
            rows.append(row)
    cols = [*strata, clone_key, "measure", "n_units", "n_records", x_key, y_key, "statistic", "expected", "null_mean"]
    cols += ["null_sd", "ratio", "z", "p_value", "q_value", "n_perm", "significant"]
    table = pd.DataFrame(rows, columns=[c for c in cols if c not in ("q_value", "significant")])
    table["q_value"] = np.nan
    for m in meas:
        sel = (table["measure"] == m).to_numpy()
        table.loc[sel, "q_value"] = bh(table.loc[sel, "p_value"].to_numpy(dtype=float))
    table["significant"] = table["q_value"] < alpha
    table = table[cols]
    first = table[table["measure"] == meas[0]]
    summary = {
        "clones_tested": m_tests,
        "significant": int(first["significant"].sum()),
        "measure": meas[0],
        "alpha": alpha,
        "p_method": p_method + ("/sequential" if do_seq else ""),
        "max_perm": cap if do_seq else n_perm,
    }
    params = {
        "unit": unit,
        "min_units": min_units,
        "n_perm": n_perm,
        "max_perm": cap if do_seq else n_perm,
        "h": h,
        "sequential": do_seq,
        "p_method": p_method,
        "alternative": alternative,
        "alpha": alpha,
        "measures": meas,
        "within": within_cols,
        "min_separation_um": min_sep,
        "radius_um": radius_um,
        "design": "random units of the section and chain" if design is None else design.describe(),
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="clone_dispersion", table=table, summary=summary, params=params)
