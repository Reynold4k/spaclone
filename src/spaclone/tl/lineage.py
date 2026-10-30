"""Within-lineage tests on trees whose tips have tissue positions.

A B-cell lineage tree (germline-rooted, branch lengths in mutations) whose tips sit at known places lets three
questions be asked without any model of the tissue:

* :func:`tip_region_test`: do the tips of a tree lie in a region (a germinal centre, a CSR hotspot) more often than
  their covariate (isotype) predicts? Expected shares come from a reference population (all loci of the same isotype);
  alternatively, within trees, do covariate-positive tips (switched, mutated) lie in the region more often than the
  tree's other tips?
* :func:`tip_contrast`: within trees, does a tip value (mutations) differ between two groups of tips (in a hotspot or
  not, switched or not)? Labels are shuffled within each tree, so every tree is its own control.
* :func:`tree_tissue_distance`: are lineage relatives (short patristic distance) also tissue neighbours? Positions are
  shuffled within each tree, by cell (locus) so that the tips of one cell move together.

Patristic distances come from Newick strings (a minimal parser is included), edge tables or ready matrices
(:func:`patristic_distances`).

References
----------
Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885 (dowser: within-lineage permutation tests of tip traits);
Phipson & Smyth 2010, doi:10.2202/1544-6115.1585 (Monte-Carlo p-values); Benjamini & Hochberg 1995,
doi:10.1111/j.2517-6161.1995.tb02031.x; Hong 2013, doi:10.1016/j.csda.2012.10.006 (Poisson-binomial distribution).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from spaclone import _schema as S
from spaclone._utils import (
    Alternative,
    RngLike,
    SpacloneWarning,
    TestResult,
    as_rng,
    bh,
    check_alternative,
    mc_pvalue,
    rng_entropy,
)

__all__ = ["patristic_distances", "tip_contrast", "tip_region_test", "tree_tissue_distance"]


# ------------------------------------------------------------------------------------------------------- trees


def _parse_newick(text: str) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """Parent index, branch length and label of every node of a Newick tree (node 0 is the root)."""
    s = re.sub(r"\[[^\]]*\]", "", text).strip()
    tokens = re.findall(r"'(?:[^']|'')*'|[(),:;]|[^(),:;'\s]+", s)
    parent, length, label = [-1], [0.0], [None]
    cur, stack, want_len = 0, [], False
    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            parent.append(cur)
            length.append(0.0)
            label.append(None)
            cur = len(parent) - 1
        elif tok == ",":
            if not stack:
                raise ValueError("Newick: ',' outside parentheses")
            parent.append(stack[-1])
            length.append(0.0)
            label.append(None)
            cur = len(parent) - 1
        elif tok == ")":
            if not stack:
                raise ValueError("Newick: unbalanced ')'")
            cur = stack.pop()
        elif tok == ":":
            want_len = True
        elif tok == ";":
            break
        elif want_len:
            length[cur] = float(tok)
            want_len = False
        else:
            label[cur] = tok[1:-1].replace("''", "'") if tok.startswith("'") else tok
    if stack:
        raise ValueError("Newick: unbalanced '('")
    return np.asarray(parent, dtype=np.int64), np.asarray(length, dtype=float), label


def _edges_to_arrays(
    edges: pd.DataFrame, node_key: str, parent_key: str, length_key: str, label_key: str
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    missing = [c for c in (node_key, parent_key, length_key) if c not in edges.columns]
    if missing:
        raise KeyError(f"edge table lacks column(s) {missing}")
    ids = edges[node_key].to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    if len(pos) != len(ids):
        raise ValueError("edge table: node ids are not unique")
    par = edges[parent_key]
    parent = np.array([pos[p] if pd.notna(p) and p in pos else -1 for p in par], dtype=np.int64)
    if pd.api.types.is_float_dtype(par):  # integer ids read as floats because of the root's NaN
        look = {float(k): v for k, v in pos.items() if isinstance(k, int | np.integer | float)}
        parent = np.array([look.get(float(p), -1) if pd.notna(p) else -1 for p in par], dtype=np.int64)
    if (parent < 0).sum() != 1:
        raise ValueError(f"edge table: expected one root (a node without parent), found {(parent < 0).sum()}")
    length = edges[length_key].to_numpy(dtype=float)
    length = np.where(np.isfinite(length), length, 0.0)
    lab = edges[label_key] if label_key in edges.columns else pd.Series([None] * len(edges))
    labels = [str(v) if isinstance(v, str) and v != "" else None for v in lab]
    # re-root the arrays so that the root comes first is not needed; depths are computed from the parent pointers
    return parent, length, labels


def _patristic(parent: np.ndarray, length: np.ndarray, labels: list[str | None]) -> pd.DataFrame:
    """Patristic distances between all labelled nodes (labels may repeat only across distinct nodes)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    root = int(np.flatnonzero(parent < 0)[0])
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
    depth = np.zeros(n)
    order = [root]
    for v in order:  # breadth-first; appends while iterating
        for c in children[v]:
            depth[c] = depth[v] + length[c]
            order.append(c)
    if len(order) != n:
        raise ValueError("tree: the parent pointers do not form one connected tree")
    tip_nodes = [v for v in range(n) if labels[v] is not None]
    index = {v: i for i, v in enumerate(tip_nodes)}
    m = len(tip_nodes)
    dist = np.zeros((m, m))
    below: list[list[int] | None] = [None] * n
    for v in reversed(order):  # children before parents
        groups = [below[c] for c in children[v] if below[c]]
        if v in index:
            groups.append([index[v]])
        for a in range(len(groups)):
            ga = np.asarray(groups[a])
            for b in range(a + 1, len(groups)):
                gb = np.asarray(groups[b])
                d = depth[np.asarray(tip_nodes)[ga]][:, None] + depth[np.asarray(tip_nodes)[gb]][None, :] - 2 * depth[v]
                dist[np.ix_(ga, gb)] = d
                dist[np.ix_(gb, ga)] = d.T
        below[v] = [i for g in groups for i in g]
        for c in children[v]:
            below[c] = None
    names = [labels[v] for v in tip_nodes]
    return pd.DataFrame(dist, index=names, columns=names)


def patristic_distances(
    tree: str | pd.DataFrame,
    *,
    node_key: str = "node",
    parent_key: str = "parent",
    length_key: str = "length",
    label_key: str = "label",
) -> pd.DataFrame:
    """Patristic distances (sum of branch lengths on the path) between the labelled nodes of a tree.

    Parameters
    ----------
    tree
        A Newick string (``"((a:1,b:2):1,Germline:0);"``; quoted labels, internal labels and ``[...]`` comments
        allowed; missing branch lengths count 0), or an edge table with one row per node: its id, its parent's id
        (missing for the root) and the length of the branch to the parent; nodes with a label are reported.
    node_key, parent_key, length_key, label_key
        Columns of the edge table.

    Returns
    -------
    pandas.DataFrame
        Square, symmetric matrix indexed by node label (in node order), zero diagonal.

    Examples
    --------
    >>> patristic_distances("((a:1,b:2):1,c:0.5);").loc["a", "c"]
    2.5
    """
    if isinstance(tree, str):
        parent, length, labels = _parse_newick(tree)
    elif isinstance(tree, pd.DataFrame):
        parent, length, labels = _edges_to_arrays(tree, node_key, parent_key, length_key, label_key)
    else:
        raise TypeError(f"tree must be a Newick string or an edge table, got {type(tree).__name__}")
    out = _patristic(parent, length, labels)
    if out.index.duplicated().any():
        dup = out.index[out.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"tree labels are not unique: {dup}")
    return out


def _distance_matrix(tree: Any, keys: dict[str, str]) -> pd.DataFrame:
    if isinstance(tree, pd.DataFrame) and keys["parent_key"] not in tree.columns:
        if not tree.index.equals(tree.columns):
            raise ValueError("a distance matrix must have the same labels as index and columns")
        return tree
    return patristic_distances(tree, **keys)


# --------------------------------------------------------------------------------------------- shared helpers


def _factor(values: Any) -> np.ndarray:
    """Integer codes in sorted order; missing values get codes of their own (each its own unit)."""
    codes = pd.factorize(pd.Series(values), sort=True)[0].astype(np.int64)
    miss = codes < 0
    if miss.any():
        codes[miss] = codes.max(initial=-1) + 1 + np.arange(int(miss.sum()))
    return codes


def _resolve_units(
    tips: pd.DataFrame,
    tree_key: str,
    unit_key: str | None,
    *,
    spot_key: str,
    x_key: str,
    y_key: str,
    loci_radius_um: float,
    fn: str,
) -> tuple[np.ndarray | None, str]:
    """Cell-scale unit of every row (codes; units are always nested in trees) or None (every row its own unit).

    ``"auto"``: the ``locus_id`` column, else rows of one tree within ``loci_radius_um`` merged by single linkage
    (:func:`spaclone.pp.merge_loci`; needs coordinates), else the spot column, else every row with a warning.
    ``"loci"``: the merge, required. Any other string: that column.
    """
    if unit_key is None:
        return None, "rows"
    if unit_key == "auto" and S.LOCUS_ID not in tips.columns and x_key in tips.columns and y_key in tips.columns:
        unit_key = "loci"
        auto = " (auto)"
    else:
        auto = ""
    if unit_key == "loci":
        missing = [c for c in (x_key, y_key) if c not in tips.columns]
        if missing:
            raise KeyError(f"{fn}: unit_key='loci' needs coordinates {missing}")
        from spaclone.pp.loci import merge_loci

        xy = tips[[x_key, y_key]].to_numpy(dtype=float)
        ok = np.isfinite(xy).all(axis=1)
        codes = np.full(len(tips), -1, np.int64)
        if ok.any():
            t = pd.DataFrame(
                {
                    S.SPOT: [f"{a!r}_{b!r}" for a, b in xy[ok]],
                    S.X: xy[ok, 0],
                    S.Y: xy[ok, 1],
                    S.CLONE: tips[tree_key].to_numpy()[ok],
                }
            )
            codes[ok] = merge_loci(t, radius_um=loci_radius_um, by=()).membership.to_numpy()
        if (~ok).any():
            codes[~ok] = codes.max(initial=-1) + 1 + np.arange(int((~ok).sum()))
        return codes, f"loci ({loci_radius_um:g} um, rows of one tree merged){auto}"
    if unit_key != "auto":
        if unit_key not in tips.columns:
            raise KeyError(f"{fn}: column {unit_key!r} not in the table")
        return _factor(tips[unit_key].to_numpy()), unit_key
    if S.LOCUS_ID in tips.columns:
        return _factor(tips[S.LOCUS_ID].to_numpy()), f"{S.LOCUS_ID} (auto)"
    if spot_key in tips.columns:
        return _factor(tips[spot_key].to_numpy()), f"{spot_key} (auto)"
    warnings.warn(
        f"{fn}: no locus, spot or position column to tell which rows come from one cell; every row is treated as an "
        "independent unit, which is anti-conservative when cells contribute several sequences. Pass unit_key= (e.g. "
        "the locus_id of spaclone.pp.merge_loci) or unit_key=None if every row is a cell.",
        SpacloneWarning,
        stacklevel=3,
    )
    return None, "rows (no unit column)"


def _strata(tips: pd.DataFrame, tree_key: str) -> list[tuple[Any, np.ndarray]]:
    return [(k, pos) for k, pos in tips.groupby(tree_key, sort=True, observed=True).indices.items()]


def _flag(tips: pd.DataFrame, key: str, value: Any) -> pd.Series:
    """Boolean membership: the column itself when ``value`` is None (must be boolean-like), else ``column == value``."""
    col = tips[key]
    if value is None:
        if not (pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col)):
            raise ValueError(f"column {key!r} is not boolean; pass the label to test (e.g. region='GC')")
        out = col.astype("boolean")
    elif isinstance(value, list | tuple | set | frozenset | np.ndarray):
        out = col.isin(list(value)).astype("boolean").mask(col.isna())
    else:
        out = (col == value).astype("boolean").mask(col.isna())
    return out


def _poisson_binomial_sf(p: np.ndarray, k: int) -> float:
    """P(X >= k) for X a sum of independent Bernoulli(p_i) (exact recursion)."""
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for i, pi in enumerate(p, start=1):
        pmf[1 : i + 1] = pmf[1 : i + 1] * (1 - pi) + pmf[0:i] * pi
        pmf[0] *= 1 - pi
    return float(min(1.0, max(0.0, pmf[k:].sum()))) if k > 0 else 1.0


def _poisson_binomial_cdf(p: np.ndarray, k: int) -> float:
    """P(X <= k)."""
    return 1.0 - _poisson_binomial_sf(p, k + 1) if k + 1 <= len(p) else 1.0


def _pb_pvalue(p: np.ndarray, obs: int, alternative: str) -> float:
    if alternative == "greater":
        return _poisson_binomial_sf(p, obs)
    if alternative == "less":
        return _poisson_binomial_cdf(p, obs)
    return min(1.0, 2 * min(_poisson_binomial_sf(p, obs), _poisson_binomial_cdf(p, obs)))


# ------------------------------------------------------------------------------------------------ tip_contrast


def tip_contrast(
    tips: pd.DataFrame,
    *,
    value_key: str,
    group_key: str,
    group: Any = None,
    tree_key: str = S.CLONE,
    n_perm: int = 10_000,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    keep_null: bool = False,
    unit_key: str | None = None,
    loci_radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> TestResult:
    """Within-tree difference of a tip value between two groups of tips, against within-tree label shuffles.

    For every tree that holds tips of both groups, ``d = mean(value | in group) - mean(value | not in group)``; the
    statistic is the mean of ``d`` over these trees (each tree weighs the same, whatever its size). Examples: mutations
    of hotspot tips minus the tree's other tips; share of follicle tips among unmutated minus mutated tips (``value``
    0 / 1).

    Parameters
    ----------
    tips
        One row per tip (or per unit of analysis, e.g. a locus) with the tree id, the value and the group.
    value_key
        Numeric column compared between the groups (booleans count 0 / 1). Rows with a missing value are dropped.
    group_key
        Column defining the group: boolean, or compared with ``group``. Rows with a missing group are dropped.
    group
        Value (or list of values) of ``group_key`` that form the group; ``None`` when ``group_key`` is boolean.
    tree_key
        Tree (clone) id; any grouping column works (e.g. the isotype, for a test across all loci of an isotype).
    n_perm
        Shuffles; the smallest attainable p-value is ``1 / (n_perm + 1)``.
    alternative
        Direction of the alternative for the mean difference (``"greater"``: the group's values are higher).
    rng
        Seed or generator (used in tree order, so equal seeds give equal results).
    keep_null
        Keep the null draws of the mean difference in ``result.null``.
    unit_key
        ``None`` (default): every row is a unit (pass one row per cell or locus, or a unit rule). Otherwise rows are
        collapsed to one unit per (tree, unit) first (mean value, group of the majority of its rows), with the rules
        of :func:`tip_region_test` (``"auto"``, ``"loci"`` or a column name).
    loci_radius_um, spot_key, x_key, y_key
        Merge radius and columns of the unit rules.

    Returns
    -------
    TestResult
        ``table``: one row per eligible tree: ``tree``, ``n_tips`` (units), ``n_group``, ``statistic`` (``d``),
        ``null_mean``, ``null_sd``, ``z``, ``p_value`` (per tree), ``q_value`` (BH over trees), ``n_perm``.
        ``summary``: ``statistic`` (mean of ``d``), ``null_mean``, ``null_sd``, ``null_lo`` / ``null_hi`` (2.5 % /
        97.5 %), ``z``, ``p_value`` and ``n_trees``.

    Notes
    -----
    Null model: within each tree, the group labels are permuted among the tree's units (rows, or the collapsed units
    of ``unit_key``); the number of group units per tree, the values and the tree membership are fixed; trees are
    shuffled independently and the statistic is recomputed as the mean over the same trees. Several rows of one cell
    must be collapsed (``unit_key``): permuted one by one they are pseudoreplicates.

    References
    ----------
    Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    check_alternative(alternative)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    for c in (tree_key, value_key, group_key):
        if c not in tips.columns:
            raise KeyError(f"tip_contrast: column {c!r} not in the table")
    gen = as_rng(rng)
    lab = _flag(tips, group_key, group)
    val = pd.to_numeric(
        tips[value_key].astype("Float64") if tips[value_key].dtype == bool else tips[value_key], errors="coerce"
    )
    ucodes, unit_rule = _resolve_units(
        tips,
        tree_key,
        unit_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        loci_radius_um=loci_radius_um,
        fn="tip_contrast",
    )
    ok = lab.notna().to_numpy() & val.notna().to_numpy()
    d = tips.loc[ok, [tree_key]].assign(_v=val[ok].to_numpy(dtype=float), _g=lab[ok].to_numpy(dtype=bool))
    if ucodes is not None:  # one row per (tree, unit): mean value, majority group (ties: in the group)
        d["_u"] = ucodes[ok]
        agg = d.groupby([tree_key, "_u"], sort=True, observed=True)
        d = pd.DataFrame({"_v": agg["_v"].mean(), "_g": agg["_g"].mean() >= 0.5}).reset_index()
    rows, nulls = [], []
    for tree, pos in _strata(d, tree_key):
        v = d["_v"].to_numpy()[pos]
        g = d["_g"].to_numpy()[pos]
        k, n = int(g.sum()), len(g)
        if k == 0 or k == n:
            continue
        obs = v[g].mean() - v[~g].mean()
        null = np.empty(n_perm)
        chunk = max(1, 2_000_000 // n)
        for a in range(0, n_perm, chunk):  # k random members per shuffle (the k smallest of n uniform keys)
            bsz = min(chunk, n_perm - a)
            pick = np.argpartition(gen.random((bsz, n)), k, axis=1)[:, :k] if k < n else np.arange(n)[None, :]
            s_in = v[pick].sum(axis=1)
            null[a : a + bsz] = s_in / k - (v.sum() - s_in) / (n - k)
        mu, sd = float(null.mean()), float(null.std(ddof=1)) if n_perm > 1 else float("nan")
        rows.append(
            {
                "tree": tree,
                "n_tips": n,
                "n_group": k,
                "statistic": float(obs),
                "null_mean": mu,
                "null_sd": sd,
                "z": (obs - mu) / sd if sd > 0 else np.nan,
                "p_value": float(mc_pvalue(null, obs, alternative=alternative)[0]),
                "n_perm": n_perm,
            }
        )
        nulls.append(null)
    table = pd.DataFrame(
        rows, columns=["tree", "n_tips", "n_group", "statistic", "null_mean", "null_sd", "z", "p_value", "n_perm"]
    )
    table["q_value"] = bh(table["p_value"].to_numpy(dtype=float))
    summary: dict[str, Any] = {"n_trees": len(rows)}
    gnull = np.mean(nulls, axis=0) if nulls else None
    if gnull is not None:
        obs = float(table["statistic"].mean())
        mu, sd = float(gnull.mean()), float(gnull.std(ddof=1)) if n_perm > 1 else float("nan")
        summary.update(
            {
                "statistic": obs,
                "null_mean": mu,
                "null_sd": sd,
                "null_lo": float(np.quantile(gnull, 0.025)),
                "null_hi": float(np.quantile(gnull, 0.975)),
                "z": (obs - mu) / sd if sd > 0 else float("nan"),
                "p_value": float(mc_pvalue(gnull, obs, alternative=alternative)[0]),
            }
        )
    else:
        summary.update({"statistic": float("nan"), "p_value": float("nan")})
    params = {
        "value_key": value_key,
        "group_key": group_key,
        "group": group,
        "tree_key": tree_key,
        "unit_key": unit_key,
        "units": unit_rule,
        "n_perm": n_perm,
        "alternative": alternative,
        "null": "group labels permuted among the units (rows) of each tree",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(
        name="tip_contrast", table=table, summary=summary, params=params, null=gnull if keep_null else None
    )


# --------------------------------------------------------------------------------------------- tip_region_test


def _collapse_units(d: pd.DataFrame, tree_key: str, unit_key: str, cov: str | None) -> pd.DataFrame:
    """One row per (tree, unit): the region flag of its first tip, its most frequent covariate class."""
    agg = d.groupby([tree_key, unit_key], sort=True, observed=True, dropna=False)
    out = agg["_in"].first().to_frame()
    if cov is not None:
        out["_cov"] = agg["_cov"].agg(lambda s: s.value_counts().sort_index(kind="stable").idxmax())
    return out.reset_index()


def tip_region_test(
    tips: pd.DataFrame,
    *,
    region_key: str,
    region: Any = None,
    covariate_key: str | None = None,
    covariate: Any = None,
    reference: pd.DataFrame | None = None,
    null: Literal["expected", "within_tree"] = "expected",
    tree_key: str = S.CLONE,
    unit_key: str | None = "auto",
    method: Literal["exact", "mc"] = "exact",
    n_perm: int = 10_000,
    min_region_tips: int = 3,
    alpha: float = 0.05,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    loci_radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> TestResult:
    """Do the tips of lineage trees lie in a region more often than their covariate predicts?

    ``null="expected"`` (per tree): each tip's chance of lying in the region is the region share of the reference units
    with the same covariate class (e.g. the hotspot share of all IGH loci of the tip's isotype); the tree's observed
    region tips are compared with the sum of these chances. Because a region defined by a covariate (a CSR hotspot by
    switched isotypes) attracts covariate-positive tips anyway, the matched expectation is what makes a tree's
    concentration lineage-specific. ``null="within_tree"`` (one statistic over trees): within trees, do
    covariate-positive tips (switched, mutated) lie in the region more often than the tree's other tips?

    Parameters
    ----------
    tips
        One row per tip with the tree id, the region membership and the covariate.
    region_key
        Region membership: a boolean column, or a label column compared with ``region``.
    region
        Region label (or list of labels) when ``region_key`` holds labels.
    covariate_key
        Covariate column (e.g. the isotype). ``null="expected"``: classes of the matched expectation (``None``: one
        class). ``null="within_tree"``: required, boolean or compared with ``covariate``.
    covariate
        Covariate value(s) forming the positive class for ``null="within_tree"``.
    reference
        Units that define the expected region share per covariate class (``null="expected"``), with the same
        ``region_key`` and ``covariate_key`` columns, e.g. all loci of the section. ``None``: the tips themselves.
        Covariate classes of the tips must occur in the reference.
    null
        ``"expected"`` or ``"within_tree"`` (see above).
    tree_key
        Tree (clone) id.
    unit_key
        Unit of analysis, so that several sequences from one cell count once: tips are collapsed to one unit per
        (tree, unit) before testing (region of the unit's first tip, its most frequent covariate class). ``"auto"``
        (default): the ``locus_id`` column when present (the membership of :func:`spaclone.pp.merge_loci`), else the
        tips of one tree within ``loci_radius_um`` merged by single linkage (needs ``x_key`` / ``y_key``), else the
        ``spot_key`` column, else every tip, with a :class:`~spaclone._utils.SpacloneWarning`. ``"loci"``: the merge.
        A column name: that column. ``None``: every tip is a unit (the archive's L11 rule).
    method
        ``null="expected"``: ``"exact"`` Poisson-binomial tail (default) or ``"mc"``, ``n_perm`` Bernoulli draws of
        every tip per tree (the archive's L11 rule).
    n_perm
        Draws for ``method="mc"`` and shuffles for ``null="within_tree"``.
    min_region_tips
        ``null="expected"``: a tree is called concentrated when it has at least this many region tips and
        ``q <= alpha``.
    alpha
        FDR level (BH over trees).
    alternative
        ``"greater"`` (default): more region tips than expected.
    rng
        Seed or generator (used for ``method="mc"`` and ``null="within_tree"``, in tree order).
    loci_radius_um
        Merge radius of the ``"loci"`` unit rule (µm).
    spot_key, x_key, y_key
        Columns used by the unit rules.

    Returns
    -------
    TestResult
        ``null="expected"``: ``table`` with one row per tree: ``tree``, ``n_tips`` (units after collapsing),
        ``statistic`` (region units),
        ``expected``, ``ratio``, ``p_value``, ``q_value``, ``n_perm`` (0 for exact), ``significant``; ``summary``
        totals (``statistic``, ``expected``) with a global exact p-value over all tips and the number of trees
        called.
        ``null="within_tree"``: as :func:`tip_contrast` with the region membership as the value.

    Notes
    -----
    Null model (``"expected"``): every unit (cell or locus, its tips collapsed) lies in the region independently, with
    the probability of its covariate class in the reference (random placement of the units among the reference units
    of the same class); tree membership and covariates are fixed. Null model (``"within_tree"``): covariate labels
    permuted among the units of each tree, region memberships fixed. Give the reference in the same units (one row per
    locus); without a reference the collapsed units themselves define the shares.

    The tips of one cell share its position, so counted as separate tips they are pseudoreplicates: with
    1 + Poisson(2) sequences per cell, testing tips (``unit_key=None``) rejected 15 % of null trees at
    alpha = 0.05, cells (``"auto"``) 2.2 % (the exact test is discrete with 3-8 cells per tree); within trees 17 % vs
    3-6 % (F4a calibration study, 300 replicates of 200 trees; review 2026-10-30, M2).

    References
    ----------
    Hoehn et al. 2022, doi:10.1371/journal.pcbi.1009885; Hong 2013, doi:10.1016/j.csda.2012.10.006; Benjamini &
    Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    check_alternative(alternative)
    if null not in ("expected", "within_tree"):
        raise ValueError(f"null must be 'expected' or 'within_tree', got {null!r}")
    if method not in ("exact", "mc"):
        raise ValueError(f"method must be 'exact' or 'mc', got {method!r}")
    for c in [tree_key, region_key] + ([covariate_key] if covariate_key else []):
        if c not in tips.columns:
            raise KeyError(f"tip_region_test: column {c!r} not in the table")
    gen = as_rng(rng)
    ucodes, unit_rule = _resolve_units(
        tips,
        tree_key,
        unit_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        loci_radius_um=loci_radius_um,
        fn="tip_region_test",
    )
    d = tips[[c for c in dict.fromkeys([tree_key, covariate_key]) if c]].copy()
    if ucodes is not None:
        d["_unit"] = ucodes
    d["_in"] = _flag(tips, region_key, region)
    if null == "within_tree":
        if covariate_key is None:
            raise ValueError("null='within_tree' needs covariate_key (the two classes compared within trees)")
        d["_cov"] = _flag(tips, covariate_key, covariate)
        d = d[d["_in"].notna() & d["_cov"].notna()]
        if ucodes is not None:
            d = _collapse_units(d, tree_key, "_unit", "_cov")
        res = tip_contrast(
            d.assign(_in=d["_in"].astype(float)),
            value_key="_in",
            group_key="_cov",
            tree_key=tree_key,
            n_perm=n_perm,
            alternative=alternative,
            rng=gen,
            unit_key=None,
        )
        params = dict(res.params)
        params.update(
            {
                "region_key": region_key,
                "region": region,
                "covariate_key": covariate_key,
                "covariate": covariate,
                "unit_key": unit_key,
                "units": unit_rule,
                "null": "covariate labels permuted among the units of each tree",
            }
        )
        return TestResult(name="tip_region_test", table=res.table, summary=res.summary, params=params)
    if covariate_key is not None:
        d["_cov"] = tips[covariate_key]
    d = d[d["_in"].notna()]
    if ucodes is not None:
        d = _collapse_units(d, tree_key, "_unit", "_cov" if covariate_key else None)
    if reference is None:  # the tips themselves (their units after collapsing)
        ref_in = d["_in"].reset_index(drop=True)
        ref_cov = d["_cov"].reset_index(drop=True) if covariate_key else None
    else:
        for c in [region_key] + ([covariate_key] if covariate_key else []):
            if c not in reference.columns:
                raise KeyError(f"tip_region_test: reference lacks column {c!r}")
        ref_in = _flag(reference, region_key, region)
        ref_cov = reference[covariate_key] if covariate_key else None
    okr = ref_in.notna().to_numpy()
    if covariate_key is None:
        share = pd.Series({"_all": float(ref_in[okr].astype(float).mean())})
        prob = np.full(len(d), share["_all"])
    else:
        share = ref_in[okr].astype(float).groupby(ref_cov[okr].to_numpy(), dropna=False).mean()
        unseen = sorted(set(d["_cov"].dropna().unique()) - set(share.index.dropna()), key=str)
        if unseen or (d["_cov"].isna().any() and not share.index.isna().any()):
            raise ValueError(f"covariate class(es) {unseen or ['<NA>']} of the tips do not occur in the reference")
        prob = d["_cov"].map(share).to_numpy(dtype=float)
        if d["_cov"].isna().any():
            prob[d["_cov"].isna().to_numpy()] = float(share[share.index.isna()].iloc[0])
    d = d.assign(_p=prob)
    rows = []
    for tree, pos in _strata(d, tree_key):
        p = d["_p"].to_numpy()[pos]
        obs = int(d["_in"].to_numpy(dtype=bool)[pos].sum())
        if method == "exact":
            pv, used = _pb_pvalue(p, obs, alternative), 0
        else:
            sims = (gen.random((n_perm, len(p))) < p).sum(axis=1)
            pv, used = float(mc_pvalue(sims, obs, alternative=alternative)[0]), n_perm
        exp = float(p.sum())
        rows.append(
            {
                "tree": tree,
                "n_tips": len(p),
                "statistic": obs,
                "expected": exp,
                "ratio": obs / exp if exp > 0 else np.nan,
                "p_value": pv,
                "n_perm": used,
            }
        )
    table = pd.DataFrame(rows, columns=["tree", "n_tips", "statistic", "expected", "ratio", "p_value", "n_perm"])
    table["q_value"] = bh(table["p_value"].to_numpy(dtype=float))
    table["significant"] = (table["statistic"] >= min_region_tips) & (table["q_value"] <= alpha)
    allp = d["_p"].to_numpy()
    tot = int(d["_in"].to_numpy(dtype=bool).sum())
    summary = {
        "n_trees": len(table),
        "n_significant": int(table["significant"].sum()),
        "statistic": tot,
        "expected": float(allp.sum()),
        "p_value": _pb_pvalue(allp, tot, alternative) if len(allp) else np.nan,
    }
    params = {
        "region_key": region_key,
        "region": region,
        "covariate_key": covariate_key,
        "tree_key": tree_key,
        "unit_key": unit_key,
        "units": unit_rule,
        "method": method,
        "n_perm": n_perm,
        "min_region_tips": min_region_tips,
        "alpha": alpha,
        "alternative": alternative,
        "reference_share": share.to_dict(),
        "null": "each tip in the region independently with the reference share of its covariate class",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="tip_region_test", table=table, summary=summary, params=params)


# ----------------------------------------------------------------------------------------- tree_tissue_distance


def _shuffled_spearman(
    dmat: np.ndarray, pi: np.ndarray, pj: np.ndarray, rp: np.ndarray, size: int, n_perm: int, gen: np.random.Generator
) -> np.ndarray:
    """Spearman correlations of ``rp`` with the pair distances under ``n_perm`` random permutations of the positions."""
    out = np.empty(n_perm)
    chunk = max(1, 2_000_000 // max(1, len(pi)))
    for a in range(0, n_perm, chunk):
        bsz = min(chunk, n_perm - a)
        perm = np.argsort(gen.random((bsz, size)), axis=1)
        r = rankdata(dmat[perm[:, pi], perm[:, pj]], axis=1)
        sd = r.std(axis=1, keepdims=True)
        z = (r - r.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
        val = (z * rp).mean(axis=1)
        val[sd[:, 0] == 0] = 0.0
        out[a : a + bsz] = val
    return out


def _spearman_fixed(pat_rank_std: np.ndarray, dist: np.ndarray) -> float:
    rd = rankdata(dist)
    sd = rd.std()
    return float(((rd - rd.mean()) / sd * pat_rank_std).mean()) if sd > 0 else 0.0


def tree_tissue_distance(
    tips: pd.DataFrame,
    distances: Mapping[Any, Any],
    *,
    tree_key: str = S.CLONE,
    tip_key: str = "sequence_id",
    unit_key: str | None = "auto",
    x_key: str = S.X,
    y_key: str = S.Y,
    min_tips: int = 5,
    min_units: int = 2,
    min_pairs: int = 3,
    shuffle: Literal["units", "tips"] = "units",
    n_perm: int = 1_000,
    alternative: Alternative = "greater",
    rng: RngLike = None,
    node_key: str = "node",
    parent_key: str = "parent",
    length_key: str = "length",
    label_key: str = "label",
    loci_radius_um: float = 15.0,
    spot_key: str = S.SPOT,
) -> TestResult:
    """Are lineage relatives also tissue neighbours? Within-tree correlation of patristic and tissue distance.

    For every tree with at least ``min_tips`` tips in at least ``min_units`` units (loci), the Spearman correlation
    between the patristic distance (mutations along the tree) and the distance in the tissue is computed over all pairs
    of tips in different units (pairs inside one unit are one cell's sequences and are left out). The statistic is
    the mean correlation over trees; a positive value means that closer relatives sit closer together.

    Parameters
    ----------
    tips
        One row per tip with the tree id, the tip label in its tree, the unit and the position in µm.
    distances
        Per tree id: a Newick string, an edge table (see :func:`patristic_distances`) or a square distance matrix
        indexed by tip label. Tips whose label is not in their tree are left out (column ``n_dropped``); several tips
        may share a label (collapsed identical sequences: patristic distance 0).
    tree_key, tip_key, x_key, y_key
        Columns of ``tips``.
    unit_key
        Unit whose tips move together under the shuffles (pairs inside one unit are left out). ``"auto"`` (default):
        the ``locus_id`` column, else the tips of one tree within ``loci_radius_um`` merged by single linkage, else
        the spot column. ``"loci"``: the merge; a column name: that column; ``None``: every tip is its own unit.
    min_tips, min_units, min_pairs
        Trees with fewer tips, units or between-unit pairs (or constant patristic distances) are not tested.
    shuffle
        ``"units"`` (default): unit positions are permuted among the tree's units and the tips of a unit move together
        (the cell is the exchangeable unit). ``"tips"``: positions are permuted among tips (the archive's L7 C rule).
    n_perm
        Shuffles per tree.
    alternative
        ``"greater"`` (default): relatives closer than chance.
    rng
        Seed or generator (used in tree order).
    node_key, parent_key, length_key, label_key
        Columns of edge tables.
    loci_radius_um, spot_key
        Merge radius and spot column of the unit rules.

    Returns
    -------
    TestResult
        ``table``: one row per tested tree: ``tree``, ``n_tips``, ``n_units``, ``n_pairs``, ``n_dropped``,
        ``statistic`` (Spearman rho), ``null_mean``, ``null_sd``, ``z``, ``p_value``, ``q_value``, ``n_perm``,
        ``median_um_identical`` / ``median_um_other`` (tissue distance of pairs with patristic distance < 0.5 and of
        the other pairs). ``summary``: mean rho over trees, its null mean, sd, 2.5 / 97.5 % quantiles, ``z``,
        ``p_value``, ``n_trees``.

    Notes
    -----
    Null model: within each tree, tissue positions are permuted among the tree's units (``shuffle="units"``: a unit's
    tips move together, at the position of its first tip) or tips (``"tips"``); the tree, the patristic distances and
    the set of between-unit pairs are fixed. Trees are shuffled independently and the statistic is recomputed as the
    mean over the same trees.

    References
    ----------
    Mantel 1967 (doi:10.1158/0008-5472.CAN-67-0026; matrix correlation against permutations); Hoehn et al. 2022,
    doi:10.1371/journal.pcbi.1009885; Phipson & Smyth 2010, doi:10.2202/1544-6115.1585.
    """
    check_alternative(alternative)
    if shuffle not in ("units", "tips"):
        raise ValueError(f"shuffle must be 'units' or 'tips', got {shuffle!r}")
    need = [tree_key, tip_key, x_key, y_key]
    missing = [c for c in need if c not in tips.columns]
    if missing:
        raise KeyError(f"tree_tissue_distance: column(s) {missing} not in the table")
    gen = as_rng(rng)
    ucodes, unit_rule = _resolve_units(
        tips,
        tree_key,
        unit_key,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        loci_radius_um=loci_radius_um,
        fn="tree_tissue_distance",
    )
    keys = {"node_key": node_key, "parent_key": parent_key, "length_key": length_key, "label_key": label_key}
    rows, nulls = [], []
    for tree, pos in _strata(tips, tree_key):
        if tree not in distances:
            continue
        lv = tips.iloc[pos]
        dm = _distance_matrix(distances[tree], keys)
        known = lv[tip_key].astype(str).isin(dm.index).to_numpy()
        n_drop = int((~known).sum())
        lv = lv[known]
        n = len(lv)
        unit = ucodes[pos][known] if ucodes is not None else np.arange(n)
        if n < min_tips or len(pd.unique(unit)) < min_units:
            continue
        iu = np.triu_indices(n, 1)
        lab = dm.index.get_indexer(lv[tip_key].astype(str))
        pat = dm.to_numpy()[lab[iu[0]], lab[iu[1]]]
        diff = unit[iu[0]] != unit[iu[1]]
        if diff.sum() < min_pairs or np.nanstd(pat[diff]) == 0:
            continue
        xy = lv[[x_key, y_key]].to_numpy(dtype=float)
        rp = rankdata(pat[diff])
        rp = (rp - rp.mean()) / rp.std()
        obs = _spearman_fixed(rp, pdist(xy)[diff])
        pi, pj = iu[0][diff], iu[1][diff]
        if shuffle == "tips":
            dmat, size = squareform(pdist(xy)), n
        else:
            ucode, uidx = np.unique(unit, return_inverse=True)
            first = np.full(len(ucode), -1)
            first[uidx[::-1]] = np.arange(n)[::-1]
            dmat, size = squareform(pdist(xy[first])), len(ucode)
            pi, pj = uidx[pi], uidx[pj]
        null = _shuffled_spearman(dmat, pi, pj, rp, size, n_perm, gen)
        dd = pdist(xy)[diff]
        same = pat[diff] < 0.5
        mu, sd = float(null.mean()), float(null.std(ddof=1)) if n_perm > 1 else float("nan")
        rows.append(
            {
                "tree": tree,
                "n_tips": n,
                "n_units": int(len(pd.unique(unit))),
                "n_pairs": int(diff.sum()),
                "n_dropped": n_drop,
                "statistic": obs,
                "null_mean": mu,
                "null_sd": sd,
                "z": (obs - mu) / sd if sd > 0 else np.nan,
                "p_value": float(mc_pvalue(null, obs, alternative=alternative)[0]),
                "n_perm": n_perm,
                "median_um_identical": float(np.median(dd[same])) if same.any() else np.nan,
                "median_um_other": float(np.median(dd[~same])) if (~same).any() else np.nan,
            }
        )
        nulls.append(null)
    cols = [
        "tree",
        "n_tips",
        "n_units",
        "n_pairs",
        "n_dropped",
        "statistic",
        "null_mean",
        "null_sd",
        "z",
        "p_value",
        "n_perm",
        "median_um_identical",
        "median_um_other",
    ]
    table = pd.DataFrame(rows, columns=cols)
    table.insert(10, "q_value", bh(table["p_value"].to_numpy(dtype=float)))
    summary: dict[str, Any] = {"n_trees": len(rows)}
    if nulls:
        gnull = np.mean(nulls, axis=0)
        obs = float(table["statistic"].mean())
        mu, sd = float(gnull.mean()), float(gnull.std(ddof=1)) if n_perm > 1 else float("nan")
        summary.update(
            {
                "statistic": obs,
                "null_mean": mu,
                "null_sd": sd,
                "null_lo": float(np.quantile(gnull, 0.025)),
                "null_hi": float(np.quantile(gnull, 0.975)),
                "z": (obs - mu) / sd if sd > 0 else float("nan"),
                "p_value": float(mc_pvalue(gnull, obs, alternative=alternative)[0]),
            }
        )
    else:
        gnull = None
        summary.update({"statistic": float("nan"), "p_value": float("nan")})
    params = {
        "tree_key": tree_key,
        "tip_key": tip_key,
        "unit_key": unit_key,
        "units": unit_rule,
        "min_tips": min_tips,
        "min_units": min_units,
        "min_pairs": min_pairs,
        "shuffle": shuffle,
        "n_perm": n_perm,
        "alternative": alternative,
        "null": f"tissue positions permuted among the {shuffle} of each tree",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="tree_tissue_distance", table=table, summary=summary, params=params, null=gnull)
