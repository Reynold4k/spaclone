"""Cell-type trust tests: are records on their own cell type, and does identity depend on the type under them?

* :func:`own_type_share` - share of receptor records sitting on units of their own cell type (B lineage for BCR, T
  cells for TCR) by read floor, against random placement of whole spots on units of the same sample (weighted by the
  captured cDNA and matched for local Ig RNA when those are given). A molecule-level test: it cannot tell a molecule
  on its own cell from one on another cell of the same type.
* :func:`composition` - G-test of clone composition between two groups of (spot, clone) units (e.g. units on
  RCTD-certain B-cell bins vs units on certain non-B bins) against a fixed-margin null, with group differences of
  clone-level features (top-clone share, rare-clone share, clone size) and per-clone exact tests.
* :func:`composition_power` - how large a genuine fraction the composition test would see (spike-ins of group-biased
  identities into the real table).

Under barcode swapping a molecule keeps its identity but takes the barcode of an unrelated captured cDNA molecule, so
its position is a draw from the captured cDNA, whatever its identity: the share on own type equals the cDNA-weighted
share of own-type units (inside every stratum of local Ig RNA, when swaps follow Ig cDNA), and every group of units
holds the same clone composition, however many records it holds.

References
----------
Hope 1968, doi:10.1111/j.2517-6161.1968.tb00759.x (Monte-Carlo tests); Patefield 1981, doi:10.2307/2346669
(contingency tables with fixed margins); Hothorn et al. 2008, doi:10.18637/jss.v028.i08 (conditional permutation
framework, strata); Gel et al. 2016, doi:10.1093/bioinformatics/btv562 (randomise-and-evaluate placement tests).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import fisher_exact, random_table

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy
from spaclone.nulls import Relocate

__all__ = ["CompositionResult", "composition", "composition_power", "own_type_share"]

logger = logging.getLogger("spaclone")

#: Placements of :func:`own_type_share` when neither ``n_perm`` nor ``n_place`` is given.
N_PLACE_DEFAULT = 199
#: Share of records on units without a type above which :func:`own_type_share` warns.
DROPPED_WARN_SHARE = 0.5


def _as_list(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _resolve_n_perm(n_perm: int | None, n_place: int | None, default: int) -> int:
    """``n_perm`` and ``n_place`` are aliases; at most one of them (or both with the same value) may be given."""
    if n_perm is not None and n_place is not None and int(n_perm) != int(n_place):
        raise ValueError(f"n_perm ({n_perm}) and n_place ({n_place}) are aliases; give one of them")
    n = n_perm if n_perm is not None else n_place
    n = default if n is None else int(n)
    if n < 1:
        raise ValueError("n_perm (n_place) must be >= 1")
    return n


def _by_spot_id(values: pd.Series, spot_ids: pd.Series, what: str) -> np.ndarray:
    """Values of a Series indexed by spot id, aligned with the units' spot ids (ids must be unique)."""
    if spot_ids.duplicated().any():
        raise ValueError(
            f"{what} given as a Series needs unique spot ids in unit_types; with several libraries pass a column of "
            "the unit_types DataFrame instead"
        )
    v = pd.Series(values)
    if v.index.duplicated().any():
        raise ValueError(f"{what} has duplicated spot ids in its index")
    return v.reindex(pd.Index(spot_ids.to_numpy())).to_numpy()


def _unit_values(spec: str | pd.Series, units: pd.DataFrame, spot_key: str, what: str) -> tuple[np.ndarray, str]:
    """One value per unit from a column name of ``units`` or a Series indexed by spot id; returns (values, name)."""
    if isinstance(spec, str):
        if spec not in units.columns:
            raise KeyError(f"{what} column {spec!r} not in unit_types (pass a DataFrame with that column)")
        return units[spec].to_numpy(), spec
    if isinstance(spec, pd.Series):
        return _by_spot_id(spec, units[spot_key], what), str(spec.name) if spec.name is not None else what
    raise TypeError(f"{what} must be a column name of unit_types or a Series indexed by spot id")


def _strata_labels(values: np.ndarray) -> np.ndarray:
    """Stratum labels as strings, missing values as ``"NA"`` (so that they form one stratum of their own)."""
    s = pd.Series(values, dtype=object)
    return s.where(s.notna(), "NA").astype(str).to_numpy(dtype=object)


class _Placer:
    """Draws of :class:`spaclone.nulls.Relocate` (with replacement) without copying a table per draw.

    The generator is consumed exactly as ``Relocate.__call__`` consumes it on the same inputs: units in order of first
    appearance, strata in order of first appearance among the units, one ``choice`` per stratum.
    """

    def __init__(self, unit: np.ndarray, rec_strata: pd.DataFrame, slot_strata: pd.DataFrame, w: np.ndarray | None):
        codes, _ = pd.factorize(pd.Series(unit, dtype=object), use_na_sentinel=False)
        self.codes = codes.astype(np.int64)
        n_units = int(self.codes.max()) + 1 if len(self.codes) else 0
        first = np.full(n_units, -1, dtype=np.int64)
        first[self.codes[::-1]] = np.arange(len(self.codes))[::-1]
        self.first = first
        cols = list(rec_strata.columns)
        heads = rec_strata.iloc[first].reset_index(drop=True)
        self.unit_groups = _tuple_groups(heads, cols)
        self.slot_groups = _tuple_groups(slot_strata.reset_index(drop=True), cols)
        self.p: dict[tuple, np.ndarray | None] = {}
        for key, upos in self.unit_groups.items():
            spos = self.slot_groups.get(key)
            if spos is None or len(spos) == 0:
                raise ValueError(f"own_type_share: no unit of stratum {key!r} to place {len(upos)} unit(s) on")
            if w is None:
                self.p[key] = None
            else:
                ws = w[spos]
                if not ws.sum() > 0:
                    raise ValueError(f"own_type_share: placement weights sum to 0 in stratum {key!r}")
                self.p[key] = ws / ws.sum()
        self.n_units = n_units

    def draw(self, gen: np.random.Generator) -> np.ndarray:
        """Slot of every record in one random placement."""
        target = np.empty(self.n_units, dtype=np.int64)
        for key, upos in self.unit_groups.items():
            spos = self.slot_groups[key]
            target[upos] = spos[gen.choice(len(spos), size=len(upos), replace=True, p=self.p[key])]
        return target[self.codes]


def _tuple_groups(df: pd.DataFrame, cols: list[str]) -> dict[tuple, np.ndarray]:
    if not cols:
        return {(): np.arange(len(df))}
    g = df[cols].groupby(cols, sort=False, observed=True, dropna=False).indices
    return {(k if isinstance(k, tuple) else (k,)): v for k, v in g.items()}


# ------------------------------------------------------------------------------------------------- own-type share


def own_type_share(
    air: pd.DataFrame,
    *,
    unit_types: pd.Series | pd.DataFrame,
    own_types: str | Sequence[str] | Callable[[pd.Series], Any],
    weights: str | pd.Series | None = None,
    by_spot: bool = True,
    n_perm: int | None = None,
    n_place: int | None = None,
    read_floors: Sequence[float] | None = (1, 2, 3, 5, 10, 20),
    within: str | Sequence[str] | pd.Series | None = None,
    by_library: bool = False,
    rng: RngLike = None,
    type_key: str = "cell_type",
    cell_type_key: str | None = None,
    reads_key: str = S.READS,
    spot_key: str = S.SPOT,
    library_key: str = S.LIBRARY,
) -> TestResult:
    """Share of receptor records on units of their own cell type, by read floor, vs random placement.

    Parameters
    ----------
    air
        Receptor table, one row per record (molecule, contig, (bin, clone) unit or cell call), with the spot id of the
        unit it sits on and, for read floors, its read count. Barcode sinks are best removed first
        (:func:`spaclone.pp.filter_records` with ``sink_min_clones``): with ``by_spot=True`` the test stays valid with
        them, but a sink moves as one heavy unit and costs power.
    unit_types
        Cell type of every unit of the sample(s): a ``Series`` indexed by spot id, or a ``DataFrame`` with
        ``spot_key`` and ``type_key`` columns (plus ``library_key`` for several samples and any column named by
        ``weights`` or ``within``). These units are also the slots of the random placement, so pass all units (e.g.
        all GEX spots or all typed tissue bins), not only those holding records. Records on units missing from this
        table are dropped (a :class:`~spaclone._utils.SpacloneWarning` when that is half of them or more; an error
        when it is all of them, e.g. spot ids that do not match).
    own_types
        Label(s) of the own cell type (e.g. ``"B lineage"`` or the set of T-cell types), or a function mapping the
        type column to a boolean array (e.g. ``lambda t: t.str.startswith("B ")``).
    weights
        Placement weight of every unit: the captured cDNA (GEX ``total_counts``), because a swapped molecule takes
        the barcode of a captured cDNA molecule and so lands on a spot with probability proportional to its cDNA. A
        column name of the ``unit_types`` DataFrame or a ``Series`` indexed by spot id (e.g.
        ``gex.obs["total_counts"]``); non-negative, finite, one value per unit. ``None``: uniform over the units
        (only right when every unit captures about the same amount of cDNA; own-type units are usually richer).
    by_spot
        ``True`` (default): every spot moves as a whole, with all its records and their read counts, so records that
        came together (one cell's molecules, a chimera family, a barcode sink) stay together and the null
        distribution has the right width. ``False``: every record is placed independently (the archive's Q5; the null
        is far too narrow when records are grouped on spots).
    n_perm, n_place
        Number of random placements (aliases; give one). Default 199 (smallest p-value 0.005).
    read_floors
        Keep records with at least this many reads (``reads_key``). ``None``, or a table without finite read counts,
        gives one level with all records.
    within
        Slot strata: a unit moves only to units of the same stratum. Column name(s) of the ``unit_types`` DataFrame or
        a ``Series`` indexed by spot id; a record's stratum is the stratum of the unit it sits on. Recommended for BCR:
        the local Ig constant-gene RNA bin of every unit (:func:`spaclone.gex.ig_rna` with a radius of about one cell
        on sub-cellular spots, the unit's own RNA on beads or cells; bins :data:`spaclone.nulls.IG_RNA_BINS`), because
        receptor calls follow Ig RNA in working and broken assays alike, and cell types partly do too. For TCR use the
        TR constant-gene RNA (``genes=["Trac", "Trbc1", "Trbc2"]``) or no strata, not Ig RNA. Other uses: a tissue
        compartment or a depth bin.
    by_library
        Also report every library separately (rows with their ``library_id``; pooled rows have ``"all"``).
    rng
        Seed or generator.
    type_key, cell_type_key
        Column of the cell type in a ``unit_types`` DataFrame (``cell_type_key`` is an alias, the name used by
        :func:`spaclone.run`).
    reads_key, spot_key, library_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per (library, read floor) with ``min_reads`` (NaN for the single level without reads),
        ``n`` (records kept), ``n_units`` (distinct spots holding them), ``kept_share``, ``statistic`` (share on own
        type), ``expected`` (own-type share of the eligible slots under the placement weights, averaged over the kept
        records), ``null_mean``, ``null_sd``, ``null_lo``, ``null_hi``, ``ratio``, ``z``, ``p_value`` (one-sided,
        greater) and ``n_perm``. ``null`` holds the placement shares, ``(n_perm, n_rows)``. ``summary["design"]``
        describes the placement.

    Null model
    ----------
    Random placement (:class:`spaclone.nulls.Relocate`, with replacement): every spot (``by_spot=True``; every
    record with ``by_spot=False``) keeps its records, their read counts and clones, and moves to a unit drawn from the
    units of the same library and ``within`` stratum with probability proportional to ``weights`` (uniformly without
    them). The placed share is the weighted own-type share of the eligible units, whatever the read floor; this is
    what barcode swapping onto captured cDNA gives, and a genuine assay exceeds it, increasingly so at higher read
    floors. Fixed: the records, their grouping on spots, the unit types, weights and strata.

    Notes
    -----
    Recommended design (the one :func:`spaclone.run` uses when it has the GEX): ``by_spot=True``,
    ``weights=gex.obs["total_counts"]`` and, for BCR, ``within`` = the local Ig RNA bin of every unit. Without Ig RNA
    strata a broken assay whose receptor calls follow Ig cDNA (B-lineage and plasma units hold most of it) looks like
    a positional signal: on AIR-SPACE D14 the BCR molecules sat on B-lineage beads 1.80 times as often as uniformly
    placed spots, 1.37 times as often as cDNA-weighted ones, and no more often than beads of the same Ig RNA (0.58 vs
    0.59; ``validation/results/m2_own_type_share.csv``). The strata cost power: in the package simulator (2-µm spots, swaps on
    captured cDNA) a 5% genuine fraction was detected in 93% of BCR chips with cDNA weights alone and in 17% with Ig
    strata (radius 5 µm); 30% genuine in 100% and 98%. With the spot's own Ig RNA (radius 0) as the stratum on
    2-µm spots the test lost all power (genuine molecules are what puts Ig RNA on their spots).

    False-positive rates at alpha = 0.05 (200 simulated chips without genuine molecules, ``fixes/F2``): swaps
    weighted by cell-type cDNA (the 2026-10-30 review's scenario) 100% with the legacy design, 1.5% with cDNA weights,
    0% with weights and Ig strata; package simulator 100% / 2.5% / 5.0%; chimera families 4.0-4.5% with weights.

    Blind spot: a molecule that takes the barcode of a cell of the right type (template switching inside the
    constant region keeps the acceptor's constant region and barcode) counts as on its own type; this test and
    :func:`spaclone.qc.same_spot` say "signal" then (100% of such simulated chips in every design). Only clone-level
    tests (:func:`spaclone.qc.colocalisation`, :func:`spaclone.qc.pairing`) see that the clone is wrong.

    Legacy setting: ``by_spot=False, weights=None, n_place=20`` reproduces
    ``04_external_validation_2026-10/workflow/qc/Q5_shuffle_vs_celltype.py`` (``QC_shuffle.csv``); for one library and the
    same generator the placements are identical draw for draw.

    References
    ----------
    Gel et al. 2016, doi:10.1093/bioinformatics/btv562; Besag & Diggle 1977, doi:10.2307/2346974; Hothorn et al.
    2008, doi:10.18637/jss.v028.i08 (strata).
    """
    S.validate_air(air, required=(spot_key,), numeric=())
    n_draws = _resolve_n_perm(n_perm, n_place, N_PLACE_DEFAULT)
    if cell_type_key is not None:
        if type_key not in ("cell_type", cell_type_key):
            raise ValueError("type_key and cell_type_key are aliases; give one of them")
        type_key = cell_type_key
    gen = as_rng(rng)
    if isinstance(unit_types, pd.Series):
        units = pd.DataFrame({spot_key: unit_types.index.to_numpy(), type_key: unit_types.to_numpy()})
    else:
        units = pd.DataFrame(unit_types).reset_index(drop=True)
    missing = [c for c in (spot_key, type_key) if c not in units.columns]
    if missing:
        raise KeyError(f"unit_types lacks column(s) {missing}")
    use_lib = library_key in air.columns and library_key in units.columns
    if library_key in air.columns and not use_lib and air[library_key].nunique() > 1:
        raise ValueError(f"the receptor table has several {library_key!r} values; unit_types needs that column too")
    if callable(own_types):
        own = np.asarray(own_types(units[type_key]), dtype=bool)
    else:
        own = units[type_key].isin(_as_list(own_types)).to_numpy()
    lib_units = units[library_key].astype(str).to_numpy() if use_lib else np.full(len(units), "", dtype=object)
    slots = pd.DataFrame({"_key": _unit_key(lib_units, units[spot_key]), "_own": own, "_lib": lib_units})
    if slots["_key"].duplicated().any():
        raise ValueError("unit_types has duplicated unit ids (per library)")
    # slot strata (unit attributes)
    specs = [within] if isinstance(within, pd.Series) else _as_list(within)
    strata_names: list[str] = []
    for i, spec in enumerate(specs):
        vals, name = _unit_values(spec, units, spot_key, "within")
        slots[f"_s{i}"] = _strata_labels(vals)
        strata_names.append(name)
    st_cols = ["_lib", *[f"_s{i}" for i in range(len(specs))]]
    # placement weights
    w = None
    weights_name = None
    if weights is not None:
        wv, weights_name = _unit_values(weights, units, spot_key, "weights")
        w = pd.to_numeric(pd.Series(wv), errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(w)
        if bad.any():
            raise ValueError(f"weights missing or not finite for {int(bad.sum())} of {len(w)} units")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        slots["_w"] = w
    # records on typed units
    lib_rec = air[library_key].astype(str).to_numpy() if use_lib else np.full(len(air), "", dtype=object)
    pos = pd.Index(slots["_key"]).get_indexer(_unit_key(lib_rec, air[spot_key]))
    keep = pos >= 0
    n_drop = int((~keep).sum())
    if not keep.any():
        raise ValueError(
            f"none of the {len(air)} records sits on a unit of unit_types: check that the spot ids match (e.g. GEX obs "
            "names with a prefix) and, for several libraries, the library column"
        )
    drop_share = n_drop / len(air)
    if n_drop:
        msg = f"own_type_share: {n_drop} of {len(air)} records ({drop_share:.1%}) sit on units without a type; dropped"
        if drop_share >= DROPPED_WARN_SHARE:
            warnings.warn(msg, SpacloneWarning, stacklevel=2)
        else:
            logger.info(msg)
    kept_pos = pos[keep]
    rec_strata = slots[st_cols].iloc[kept_pos].reset_index(drop=True)
    unit_of_rec = slots["_key"].to_numpy()[kept_pos]
    obs_own = own[kept_pos]
    reads = None
    if read_floors is not None and reads_key in air.columns:
        r = pd.to_numeric(air[reads_key], errors="coerce").to_numpy(float)[keep]
        if np.isfinite(r).any():
            reads = r
            if not np.isfinite(r).all():
                logger.info(
                    "own_type_share: %d records without a read count count at no floor", int((~np.isfinite(r)).sum())
                )
    design = Relocate(
        slots,
        unit="_unit",
        within=st_cols,
        weights="_w" if w is not None else None,
        replace=True,
        slot_columns=["_own"],
        spot_key="_key",
    )
    move = unit_of_rec if by_spot else np.arange(len(kept_pos))
    placer = _Placer(move, rec_strata, slots[st_cols], w)
    placed = np.empty((n_draws, len(kept_pos)), dtype=bool)
    for j in range(n_draws):
        placed[j] = own[placer.draw(gen)]
    # analytic expectation: (weighted) own-type share of the eligible slots of every record
    sw = slots.assign(_wt=w if w is not None else 1.0)
    sw["_wo"] = sw["_wt"] * sw["_own"]
    agg = sw.groupby(st_cols, sort=False, dropna=False)[["_wo", "_wt"]].sum()
    share_by = agg["_wo"] / agg["_wt"]
    exp_rec = (
        share_by.reindex(pd.MultiIndex.from_frame(rec_strata))
        if len(st_cols) > 1
        else share_by.reindex(rec_strata["_lib"])
    )
    exp_rec = np.asarray(exp_rec, dtype=float)

    levels = [np.nan] if reads is None else [float(f) for f in read_floors]
    groups = [("all", np.ones(len(kept_pos), bool))]
    if by_library:
        groups += [(lv, (rec_strata["_lib"] == lv).to_numpy()) for lv in pd.unique(rec_strata["_lib"])]
    rows, nulls = [], []
    for lib_v, gm in groups:
        n_lib = int(gm.sum())
        for f in levels:
            m = gm.copy() if reads is None else gm & (reads >= f)
            n = int(m.sum())
            stat = float(obs_own[m].mean()) if n else np.nan
            null = placed[:, m].mean(axis=1) if n else np.full(n_draws, np.nan)
            mu, sd, lo, hi = (v[0] for v in null_summary(null)) if n else (np.nan,) * 4  # no record at this floor
            p = float(mc_pvalue(null, stat, alternative="greater")[0]) if n else np.nan
            rows.append(
                {
                    "library_id": lib_v,
                    "min_reads": f,
                    "n": n,
                    "n_units": int(len(pd.unique(unit_of_rec[m]))) if n else 0,
                    "kept_share": n / n_lib if (reads is not None and n_lib) else np.nan,
                    "statistic": stat,
                    "expected": float(exp_rec[m].mean()) if n else np.nan,
                    "null_mean": mu,
                    "null_sd": sd,
                    "null_lo": lo,
                    "null_hi": hi,
                    "ratio": stat / mu if n and mu > 0 else np.nan,
                    "z": (stat - mu) / sd if n and sd > 0 else np.nan,
                    "p_value": p,
                    "n_perm": n_draws,
                }
            )
            nulls.append(null)
    table = pd.DataFrame(rows)
    pooled = table[table.library_id == "all"]
    desc = "spots move as wholes" if by_spot else "records placed one by one"
    desc += f", weights {weights_name}" if weights_name else ", uniform weights"
    if strata_names:
        desc += f", within {strata_names}"
    summary = {
        "n_records": int(len(kept_pos)),
        "n_units": int(len(pd.unique(unit_of_rec))),
        "n_dropped": n_drop,
        "dropped_share": float(drop_share),
        "share_observed": float(pooled.statistic.iloc[0]),
        "share_placed": float(pooled.null_mean.iloc[0]),
        "design": desc,
    }
    if reads is not None:
        summary["share_observed_top_floor"] = float(pooled.statistic.iloc[-1])
    params = {
        "own_types": own_types if not callable(own_types) else getattr(own_types, "__name__", repr(own_types)),
        "read_floors": None if reads is None else levels,
        "n_perm": n_draws,
        "n_place": n_draws,
        "by_spot": by_spot,
        "weights": weights_name,
        "within": strata_names,
        "design": design.describe(),
        "alternative": "greater",
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="own_type_share", table=table, summary=summary, params=params, null=np.column_stack(nulls))


def _unit_key(lib: np.ndarray, spot: pd.Series) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(lib, dtype=str), "\x1f"), spot.astype(str).to_numpy().astype(str))


# ---------------------------------------------------------------------------------------------- clone composition


@dataclass(frozen=True)
class CompositionResult(TestResult):
    """Result of :func:`composition`; ``clones`` holds the per-clone exact tests (BH ``q_value`` within locus)."""

    clones: pd.DataFrame | None = None


def gstat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Likelihood-ratio G statistic of 2 × K tables given as count vectors (last axis = clones).

    ``a`` and ``b`` may be ``(K,)`` or ``(B, K)`` with the same column totals in every row.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    tot = a + b
    na, nb = a.sum(axis=-1, keepdims=True), b.sum(axis=-1, keepdims=True)
    n = na + nb
    with np.errstate(divide="ignore", invalid="ignore"):
        ea, eb = tot * na / n, tot * nb / n
        g = xlogy(a, np.where(a > 0, a / ea, 1.0)) + xlogy(b, np.where(b > 0, b / eb, 1.0))
    return 2 * g.sum(axis=-1)


def _mask(air: pd.DataFrame, m: str | np.ndarray | pd.Series, what: str) -> np.ndarray:
    if isinstance(m, str):
        if m not in air.columns:
            raise KeyError(f"{what} column {m!r} not in the receptor table")
        v = air[m].to_numpy()
    elif isinstance(m, pd.Series) and not m.index.equals(air.index):
        v = m.reindex(air.index).to_numpy()
    else:
        v = np.asarray(m)
    if len(v) != len(air):
        raise ValueError(f"{what} mask must have one value per record")
    return np.asarray(pd.array(v, dtype="boolean").fillna(False), dtype=bool)


def _jsd(pa: np.ndarray, pb: np.ndarray) -> float:
    """Jensen–Shannon divergence (bits) between two composition vectors."""
    mid = (pa + pb) / 2

    def kl(p: np.ndarray) -> float:
        nz = p > 0
        return float((p[nz] * np.log2(p[nz] / mid[nz])).sum())

    return 0.5 * kl(pa) + 0.5 * kl(pb)


def _strata_codes(air: pd.DataFrame, within: list[str]) -> np.ndarray:
    if not within:
        return np.zeros(len(air), dtype=np.int64)
    return air.groupby(within, sort=True, dropna=False).ngroup().to_numpy()


def _null_tables(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    st_a: np.ndarray,
    st_b: np.ndarray,
    n_codes: int,
    n_perm: int,
    gen: np.random.Generator,
) -> np.ndarray:
    """Group-A clone counts under fixed margins within strata: ``(n_perm, n_codes)`` (Patefield sampling)."""
    out = np.zeros((n_perm, n_codes), dtype=np.int64)
    for s in np.union1d(np.unique(st_a), np.unique(st_b)):
        a = np.bincount(codes_a[st_a == s], minlength=n_codes)
        b = np.bincount(codes_b[st_b == s], minlength=n_codes)
        tot = a + b
        cols = np.flatnonzero(tot > 0)
        na, nb = int(a.sum()), int(b.sum())
        if na == 0 or nb == 0 or len(cols) < 2:
            out[:, cols] += a[cols]  # nothing can move inside this stratum
            continue
        tab = random_table([na, nb], tot[cols]).rvs(size=n_perm, method="patefield", random_state=gen)
        out[:, cols] += tab[:, 0, :]
    return out


def _composition_core(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    st_a: np.ndarray,
    st_b: np.ndarray,
    n_codes: int,
    feats: Mapping[str, np.ndarray],
    n_perm: int,
    gen: np.random.Generator,
) -> dict[str, Any]:
    a = np.bincount(codes_a, minlength=n_codes)
    b = np.bincount(codes_b, minlength=n_codes)
    tot = a + b
    na, nb = int(a.sum()), int(b.sum())
    g0 = float(gstat(a, b))
    aa = _null_tables(codes_a, codes_b, st_a, st_b, n_codes, n_perm, gen)
    gn = gstat(aa, tot[None, :] - aa)
    # E[A counts] under the stratified fixed-margin null (the centre of the two-sided feature tests)
    ea = np.zeros(n_codes)
    for s in np.union1d(np.unique(st_a), np.unique(st_b)):
        ca, cb = codes_a[st_a == s], codes_b[st_b == s]
        ns = len(ca) + len(cb)
        if ns:
            ea += (np.bincount(ca, minlength=n_codes) + np.bincount(cb, minlength=n_codes)) * len(ca) / ns
    out: dict[str, Any] = {"n_a": na, "n_b": nb, "a": a, "b": b, "G": g0, "G_null": gn, "feats": {}}
    for f, v in feats.items():
        va, vb = float(a @ v) / na, float(b @ v) / nb
        dn = (aa @ v) / na - ((tot[None, :] - aa) @ v) / nb
        centre = float(ea @ v) / na - float((tot - ea) @ v) / nb
        out["feats"][f] = (va, vb, va - vb, dn, centre)
    return out


def _clone_features(
    size: np.ndarray,
    uniq: np.ndarray,
    top: Sequence[int],
    rare_max: int,
    clone_features: Mapping[str, Mapping | pd.Series] | None,
) -> dict[str, np.ndarray]:
    rank = np.argsort(np.argsort(-size))
    feats: dict[str, np.ndarray] = {f"top{t}_share": (rank < t).astype(float) for t in top}
    feats[f"rare_le{rare_max}_share"] = (size <= rare_max).astype(float)
    feats["log10_clone_size"] = np.log10(np.maximum(size, 1))
    for name, values in (clone_features or {}).items():
        v = pd.Series(values).reindex(pd.Index(uniq).astype(str)).to_numpy(dtype=float)
        if np.isfinite(v).any():
            v = np.where(np.isfinite(v), v, np.nanmean(v))
        else:
            v = np.zeros(len(uniq))
        feats[name] = v
    return feats


def _locus_records(air: pd.DataFrame, loci: str | Sequence[str] | None, locus_key: str) -> list[tuple[str, np.ndarray]]:
    if locus_key in air.columns:
        loc = air[locus_key].astype(str).to_numpy()
        levels = _as_list(loci) if loci is not None else sorted(pd.unique(loc))
        return [(lv, loc == lv) for lv in levels]
    return [("", np.ones(len(air), bool))]


Unit = Literal["spot_clone", "rows"]


def _collapse_units(
    air: pd.DataFrame,
    in_a: np.ndarray,
    in_b: np.ndarray,
    within: list[str],
    *,
    unit: str,
    spot_key: str,
    clone_key: str,
    locus_key: str,
    library_key: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows -> one row per distinct (library, locus, spot, clone) unit, with unit-level group masks.

    Records without a clone id are dropped (they are ignored by the tests anyway). Group membership and ``within``
    strata must be constant inside a unit.
    """
    if unit == "rows":
        return air, in_a, in_b
    if unit != "spot_clone":
        raise ValueError(f"unit must be 'spot_clone' or 'rows', got {unit!r}")
    if spot_key not in air.columns:
        raise KeyError(
            f"unit='spot_clone' needs the spot column {spot_key!r}; pass unit='rows' for a table that already has one "
            "row per unit"
        )
    has = air[clone_key].notna().to_numpy()
    sub = air.loc[has]
    keys = [c for c in (library_key, locus_key) if c in sub.columns] + [spot_key, clone_key]
    code = sub.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
    n_u = int(code.max()) + 1 if len(code) else 0
    first = np.full(n_u, -1, dtype=np.int64)
    first[code[::-1]] = np.arange(len(code))[::-1]
    out_a, out_b = [], []
    for name, m in (("group", in_a[has]), ("reference", in_b[has])):
        lo = np.ones(n_u, bool)
        hi = np.zeros(n_u, bool)
        np.logical_and.at(lo, code, m)
        np.logical_or.at(hi, code, m)
        if (lo != hi).any():
            raise ValueError(
                f"the {name} mask varies inside {int((lo != hi).sum())} (spot, clone) unit(s); define groups per unit "
                "(e.g. by the spot's cell type) or pass unit='rows'"
            )
        (out_a if name == "group" else out_b).append(lo)
    if within:
        nun = sub.groupby(code, sort=False)[within].nunique(dropna=False)
        bad = [c for c in within if (nun[c] > 1).any()]
        if bad:
            raise ValueError(f"within column(s) {bad} vary inside a (spot, clone) unit; strata must be unit-level")
    return sub.iloc[first], out_a[0], out_b[0]


def composition(
    air: pd.DataFrame,
    *,
    group: str | np.ndarray | pd.Series,
    reference: str | np.ndarray | pd.Series,
    loci: str | Sequence[str] | None = None,
    unit: Unit = "spot_clone",
    singleton_only: bool = False,
    within: str | Sequence[str] | None = None,
    n_perm: int = 999,
    top: Sequence[int] = (10, 100),
    rare_max: int = 10,
    clone_features: Mapping[str, Mapping | pd.Series] | None = None,
    min_clone_units: int = 20,
    min_units: int = 200,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> CompositionResult:
    """G-test of clone composition between two groups of (spot, clone) units against a fixed-margin null.

    Parameters
    ----------
    air
        Receptor table (one row per molecule, contig, (spot, clone) unit or call). Barcode sinks should already be
        removed. Records with a missing clone id are ignored.
    group, reference
        Boolean masks (or names of boolean columns of ``air``) selecting the two groups, e.g. records on RCTD-certain
        B-cell bins and records on certain non-B bins; they must not overlap. Records in neither group still define
        the clone-level features. With ``unit="spot_clone"`` the masks must be constant within a (spot, clone) unit
        (spot-level attributes such as the cell type of the spot are).
    loci
        Locus or loci to test (each separately); default every locus in the table.
    unit
        The exchangeable unit. ``"spot_clone"`` (default): records are first collapsed to distinct (library, locus,
        spot, clone) units, so the molecules of one clone on one spot (one cell's transcripts, a chimera family copied
        onto several UMIs) count once. ``"rows"``: every row is a unit, for tables that already have one row per unit
        (cells, beads, (bin, clone) units); on molecule rows this treats dependent molecules as independent and
        rejects far too often (100% of null chips with chimera families in the review's simulation).
    singleton_only
        Keep only units on spots holding a single unit (of any locus), where multi-clone spots cannot mix identities
        (with ``unit="rows"``: rows on spots holding a single row).
    within
        Stratum column(s): the null fixes the group sizes and clone counts inside every stratum (e.g. a spatial tile,
        to respect contiguous groups). Must be constant within a unit.
    n_perm
        Number of null tables.
    top, rare_max
        Clone-level features: share of records from the ``top`` largest clones of the locus (global size over all
        records of the locus), share from rare clones (``<= rare_max`` records) and mean log10 clone size.
    clone_features
        Further per-clone features ``{name: {clone_id: value}}`` (e.g. SHM); clones without a value get the mean.
    min_clone_units
        Clones with at least this many units in the two groups get a per-clone exact test.
    min_units
        Both groups need at least this many units, else the locus is reported untested.
    rng
        Seed or generator.
    spot_key, clone_key, locus_key, library_key
        Column names (the library column, when present, keeps equal spot ids of different libraries apart).

    Returns
    -------
    CompositionResult
        ``table``: one row per (locus, measure). ``measure = "G"``: ``statistic`` (G), ``null_mean``, ``null_sd``,
        ``null_lo``, ``null_hi``, ``z``, ``p_value`` (one-sided), ``tvd`` (total variation distance), ``jsd``
        (Jensen–Shannon divergence, bits) and ``chi2_divergence_excess`` ((G − E[G]) / N). Feature rows:
        ``value_a``, ``value_b``, ``statistic`` (difference A − B), null summaries of the difference and a
        two-sided ``p_value``. ``clones``: per-clone two-sided Fisher exact tests (``odds_ratio``, ``p_value``,
        ``q_value`` by BH within locus; pooled over strata).

    Null model
    ----------
    Random labelling of group membership of (spot, clone) units with both margins fixed: conditional on the number of
    units of each group and of each clone (inside every ``within`` stratum), every assignment of units to groups is
    equally likely (multivariate hypergeometric 2 × K tables, drawn with Patefield's algorithm). Under barcode
    swapping identity is independent of position, so this holds for units; genuine records pull a group's composition
    towards its cells. Molecules of one unit are not exchangeable with each other (they share a spot and a clone),
    which is why the unit, not the molecule, is permuted.

    Notes
    -----
    Reproduces the clone-composition tests of ``03_BCR_final_2026-10/workflow/B4a_region_composition.py`` (observed
    G, feature means and per-clone Fisher tests exactly; the null by Monte-Carlo, B4a shuffled the pooled labels,
    which samples the same distribution). B4a's input already had one row per (DNB, clone) unit, so ``unit="rows"``
    and the default give the same result there.

    References
    ----------
    Patefield 1981, doi:10.2307/2346669; Hope 1968, doi:10.1111/j.2517-6161.1968.tb00759.x; Hothorn et al. 2008,
    doi:10.18637/jss.v028.i08; Benjamini & Hochberg 1995, doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    S.validate_air(air, required=(clone_key,), numeric=())
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    gen = as_rng(rng)
    within = _as_list(within)
    in_a, in_b = _mask(air, group, "group"), _mask(air, reference, "reference")
    if (in_a & in_b).any():
        raise ValueError(f"{int((in_a & in_b).sum())} records are in both group and reference")
    n_rows = len(air)
    air, in_a, in_b = _collapse_units(
        air, in_a, in_b, within, unit=unit, spot_key=spot_key, clone_key=clone_key, locus_key=locus_key,
        library_key=library_key,
    )  # fmt: skip
    has_clone = air[clone_key].notna().to_numpy()
    single = np.ones(len(air), bool)
    if singleton_only:
        if spot_key not in air.columns:
            raise KeyError(f"singleton_only needs the spot column {spot_key!r}")
        skeys = [c for c in (library_key, spot_key) if c in air.columns] if unit == "spot_clone" else [spot_key]
        single = (air.groupby(skeys, sort=False, dropna=False)[spot_key].transform("size") == 1).to_numpy()
    st = _strata_codes(air, within)
    rows, clone_tabs = [], []
    for locus, lm in _locus_records(air, loci, locus_key):
        sel = lm & has_clone
        codes, uniq = pd.factorize(air[clone_key].to_numpy()[sel], sort=False)
        k = len(uniq)
        size = np.bincount(codes, minlength=k)
        feats = _clone_features(size, np.asarray(uniq), top, rare_max, clone_features)
        ia, ib = (in_a & single)[sel], (in_b & single)[sel]
        base = {"locus": locus, "n_a": int(ia.sum()), "n_b": int(ib.sum()), "n_clones": k}
        if ia.sum() < min_units or ib.sum() < min_units:
            logger.info("composition: locus %s untested (%d vs %d records)", locus, ia.sum(), ib.sum())
            rows.append({**base, "measure": "G", "statistic": np.nan, "p_value": np.nan, "n_perm": n_perm})
            continue
        r = _composition_core(codes[ia], codes[ib], st[sel][ia], st[sel][ib], k, feats, n_perm, gen)
        mu, sd, lo, hi = (v[0] for v in null_summary(r["G_null"]))
        pa, pb = r["a"] / r["n_a"], r["b"] / r["n_b"]
        rows.append(
            {
                **base,
                "measure": "G",
                "value_a": np.nan,
                "value_b": np.nan,
                "statistic": r["G"],
                "null_mean": mu,
                "null_sd": sd,
                "null_lo": lo,
                "null_hi": hi,
                "z": (r["G"] - mu) / sd if sd > 0 else np.nan,
                "p_value": float(mc_pvalue(r["G_null"], r["G"], alternative="greater")[0]),
                "tvd": 0.5 * float(np.abs(pa - pb).sum()),
                "jsd": _jsd(pa, pb),
                "chi2_divergence_excess": (r["G"] - mu) / (r["n_a"] + r["n_b"]),
                "n_perm": n_perm,
            }
        )
        for f, (va, vb, d0, dn, centre) in r["feats"].items():
            fm, fs, flo, fhi = (v[0] for v in null_summary(dn))
            rows.append(
                {
                    **base,
                    "measure": f,
                    "value_a": va,
                    "value_b": vb,
                    "statistic": d0,
                    "null_mean": fm,
                    "null_sd": fs,
                    "null_lo": flo,
                    "null_hi": fhi,
                    "z": (d0 - fm) / fs if fs > 0 else np.nan,
                    "p_value": float(mc_pvalue(np.abs(dn - centre), abs(d0 - centre), alternative="greater")[0]),
                    "n_perm": n_perm,
                }
            )
        a, b, na, nb = r["a"], r["b"], r["n_a"], r["n_b"]
        test = np.flatnonzero(a + b >= min_clone_units)
        if len(test):
            rank = np.argsort(np.argsort(-size))
            res = [fisher_exact([[a[j], na - a[j]], [b[j], nb - b[j]]]) for j in test]
            ct = pd.DataFrame(
                {
                    "locus": locus,
                    "clone_id": np.asarray(uniq)[test],
                    "global_units": size[test],
                    "rank": rank[test] + 1,
                    "n_a": a[test],
                    "n_b": b[test],
                    "frac_a": a[test] / na,
                    "frac_b": b[test] / nb,
                    "odds_ratio": [float(x[0]) for x in res],
                    "p_value": [float(x[1]) for x in res],
                }
            )
            ct["q_value"] = bh(ct.p_value.to_numpy())
            clone_tabs.append(ct.sort_values("p_value", kind="stable"))
    table = pd.DataFrame(rows)
    clones = pd.concat(clone_tabs, ignore_index=True) if clone_tabs else pd.DataFrame()
    g_rows = table[table.measure == "G"]
    summary = {
        "n_loci_tested": int(g_rows.p_value.notna().sum()),
        "min_p_G": float(g_rows.p_value.min()) if g_rows.p_value.notna().any() else np.nan,
        "max_abs_z_G": float(g_rows.z.abs().max()) if "z" in g_rows and g_rows.z.notna().any() else np.nan,
        "n_clones_tested": int(len(clones)),
        "n_clones_q05": int((clones.q_value < 0.05).sum()) if len(clones) else 0,
        "n_rows": n_rows,
        "n_units": int(len(air)),
    }
    params = {
        "loci": loci,
        "unit": unit,
        "singleton_only": singleton_only,
        "within": within,
        "n_perm": n_perm,
        "top": list(top),
        "rare_max": rare_max,
        "min_clone_units": min_clone_units,
        "min_units": min_units,
        "design": f"fixed-margin 2 x K tables (Patefield) of {'(spot, clone) units' if unit == 'spot_clone' else 'rows'}"
        " within strata",
        "rng_entropy": rng_entropy(gen),
    }
    return CompositionResult(name="composition", table=table, summary=summary, params=params, clones=clones)


def composition_power(
    air: pd.DataFrame,
    *,
    group: str | np.ndarray | pd.Series,
    reference: str | np.ndarray | pd.Series,
    g_grid: Sequence[float] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05),
    models: Sequence[str] = ("polyclonal", "oligoclonal", "resident"),
    n_rep: int = 20,
    n_perm: int = 199,
    alpha: float = 0.05,
    loci: str | Sequence[str] | None = None,
    unit: Unit = "spot_clone",
    singleton_only: bool = False,
    within: str | Sequence[str] | None = None,
    rare_max: int = 10,
    n_oligo: int = 20,
    n_resident: int = 50,
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
) -> TestResult:
    """Power of :func:`composition` to see a genuine fraction ``g`` of group-specific identities (spike-ins).

    In every replicate a fraction ``g`` of the group's units is relabelled with "genuine" identities and the group is
    compared with the reference exactly as in :func:`composition` (G-test, rare-clone share, top-10 share).

    Parameters
    ----------
    air, group, reference, loci, unit, singleton_only, within, spot_key, clone_key, locus_key, library_key
        As in :func:`composition`.
    g_grid
        Genuine fractions of the group's records.
    models
        ``"polyclonal"`` (every genuine record a new clone, e.g. naive B cells), ``"oligoclonal"`` (``n_oligo`` new
        clones with Zipf(1) sizes, e.g. germinal centres) or ``"resident"`` (the ``n_resident`` largest existing
        clones in a reshuffled, group-specific mix, e.g. plasma cells of the dominant clones).
    n_rep, n_perm, alpha
        Replicates per (model, g), null tables per test and the level.
    rare_max
        Rare-clone size limit.
    n_oligo, n_resident
        Clone numbers of the oligoclonal and resident models.
    rng
        Seed or generator; every (locus, model) gets a spawned child generator.

    Returns
    -------
    TestResult
        ``table``: one row per (locus, model, g) with ``k`` (relabelled records), ``n_a``, ``power_G``, ``power_rare``,
        ``power_top10`` and ``power_any_bonf`` (smallest of the three p-values × 3 <= alpha), ``n_rep``, ``n_perm``.

    Null model
    ----------
    As :func:`composition`: fixed-margin random labelling of the group membership of (spot, clone) units (within
    strata); the spike-ins are the planted alternative.

    References
    ----------
    Patefield 1981, doi:10.2307/2346669; Hope 1968, doi:10.1111/j.2517-6161.1968.tb00759.x.
    """
    S.validate_air(air, required=(clone_key,), numeric=())
    for m in models:
        if m not in ("polyclonal", "oligoclonal", "resident"):
            raise ValueError(f"unknown model {m!r}")
    gen = as_rng(rng)
    within = _as_list(within)
    in_a, in_b = _mask(air, group, "group"), _mask(air, reference, "reference")
    if (in_a & in_b).any():
        raise ValueError("group and reference overlap")
    air, in_a, in_b = _collapse_units(
        air, in_a, in_b, within, unit=unit, spot_key=spot_key, clone_key=clone_key, locus_key=locus_key,
        library_key=library_key,
    )  # fmt: skip
    has_clone = air[clone_key].notna().to_numpy()
    single = np.ones(len(air), bool)
    if singleton_only:
        skeys = [c for c in (library_key, spot_key) if c in air.columns] if unit == "spot_clone" else [spot_key]
        single = (air.groupby(skeys, sort=False, dropna=False)[spot_key].transform("size") == 1).to_numpy()
    st = _strata_codes(air, within)
    combos = [(lv, lm, m) for lv, lm in _locus_records(air, loci, locus_key) for m in models]
    children = gen.spawn(len(combos))
    rows = []
    for (locus, lm, model), child in zip(combos, children, strict=True):
        sel = lm & has_clone
        codes, uniq = pd.factorize(air[clone_key].to_numpy()[sel], sort=False)
        k0 = len(uniq)
        size = np.bincount(codes, minlength=k0)
        order = np.argsort(-size)
        ia, ib = (in_a & single)[sel], (in_b & single)[sel]
        ca0, cb = codes[ia], codes[ib]
        sa, sb = st[sel][ia], st[sel][ib]
        for g in g_grid:
            k = int(round(g * len(ca0)))
            if k < 1:
                continue
            p = np.empty((n_rep, 3))
            for rep in range(n_rep):
                ca = ca0.copy()
                idx = child.choice(len(ca), k, replace=False)
                if model == "polyclonal":
                    new = k0 + np.arange(k)
                elif model == "oligoclonal":
                    w = 1 / np.arange(1, n_oligo + 1)
                    new = k0 + child.choice(n_oligo, k, p=w / w.sum())
                else:
                    nr = min(n_resident, k0)
                    w = child.permutation(size[order[:nr]].astype(float))
                    new = order[:nr][child.choice(nr, k, p=w / w.sum())]
                ca[idx] = new
                kk = max(k0, int(ca.max()) + 1)
                sz = np.bincount(np.concatenate([codes, ca[idx]]), minlength=kk)
                feats = {
                    "rare": (sz <= rare_max).astype(float),
                    "top10": np.isin(np.arange(kk), order[:10]).astype(float),
                }
                r = _composition_core(ca, cb, sa, sb, kk, feats, n_perm, child)
                p[rep, 0] = mc_pvalue(r["G_null"], r["G"], alternative="greater")[0]
                for j, f in enumerate(("rare", "top10"), start=1):
                    _, _, d0, dn, centre = r["feats"][f]
                    p[rep, j] = mc_pvalue(np.abs(dn - centre), abs(d0 - centre), alternative="greater")[0]
            rows.append(
                {
                    "locus": locus,
                    "model": model,
                    "g": g,
                    "k": k,
                    "n_a": len(ca0),
                    "power_G": float((p[:, 0] <= alpha).mean()),
                    "power_rare": float((p[:, 1] <= alpha).mean()),
                    "power_top10": float((p[:, 2] <= alpha).mean()),
                    "power_any_bonf": float((p.min(axis=1) * 3 <= alpha).mean()),
                    "n_rep": n_rep,
                    "n_perm": n_perm,
                }
            )
    table = pd.DataFrame(rows)
    summary = {}
    if len(table):
        for (locus, model), t in table.groupby(["locus", "model"], sort=False):
            hit = t[t.power_G >= 0.8]
            summary[f"g80[{locus}|{model}]"] = float(hit.g.min()) if len(hit) else np.nan
    params = {
        "g_grid": list(g_grid),
        "models": list(models),
        "n_rep": n_rep,
        "n_perm": n_perm,
        "alpha": alpha,
        "unit": unit,
        "singleton_only": singleton_only,
        "within": within,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="composition_power", table=table, summary=summary, params=params)
