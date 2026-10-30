"""Data-driven read floor: at which read support do the loci of one clonotype behave like cells of one clone?

The cells of a genuine expanded clone carry the same partner chain (the same alpha chain for a TRB clonotype, the same
light chain for a heavy-chain clone), so the loci of one clonotype share a partner-chain clonotype more often than
random sets of loci do. A PCR chimera, the CDR3 of a highly expressed molecule copied onto another cell's barcode,
carries few reads and sits wherever its host molecule was, so its partner chains are those of unrelated cells. Below
a well-chosen read floor the loci of a clonotype share partners at chance level; above it they share them like cells
of one clone, and the floor is where that changes. Several floors are tested at once, so the suggested floor is
chosen after a multiple-testing correction across floors (Benjamini–Hochberg), not from raw p-values.

References
----------
Kebschull & Zador 2015, doi:10.1093/nar/gkv717 (PCR template switching creates chimeras); Howie et al. 2015,
doi:10.1126/scitranslmed.aac5624 (partner-chain co-occurrence); Besag & Diggle 1977, doi:10.2307/2346974 (Monte-Carlo
tests); Davison & Hinkley 1997, doi:10.1017/CBO9780511802843; Benjamini & Hochberg 1995,
doi:10.1111/j.2517-6161.1995.tb02031.x; Benjamini & Yekutieli 2001, doi:10.1214/aos/1013699998 (BH under positive
dependence).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from spaclone import _schema as S
from spaclone._utils import LowPowerWarning, RngLike, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy
from spaclone.pp import merge_loci
from spaclone.qc.colocalisation import _sink_records

__all__ = ["ReadFloorResult", "read_floor"]

logger = logging.getLogger("spaclone")

_SIDES = ("above", "below")


@dataclass(frozen=True)
class ReadFloorResult(TestResult):
    """Result of :func:`read_floor`.

    Attributes
    ----------
    loci
        The loci of the focal chain (one row each): ``library_id``, the clone column, ``x_um``, ``y_um``,
        ``n_spots``, ``reads_max`` (reads of the locus' best record, which decides its side of a floor) and
        ``n_partners`` (partner-chain clonotypes on its spots).
    """

    loci: pd.DataFrame | None = None


class _Sharing:
    """Partner sharing of the loci of one side of a floor, for any labelling of those loci."""

    def __init__(
        self,
        site_locus: np.ndarray,
        site_xy: np.ndarray,
        site_lib: np.ndarray,
        site_partner: tuple[np.ndarray, np.ndarray],
        n_partner: int,
        radius_um: float | None,
        remerge: bool,
    ) -> None:
        # site_locus: index (0..n_loci-1) of the selected locus of every selected site
        self.site_locus = site_locus
        self.n_sites = len(site_locus)
        self.n_loci = int(site_locus.max()) + 1 if len(site_locus) else 0
        self.sp_site, self.sp_partner = site_partner
        self.n_partner = max(int(n_partner), 1)
        self.remerge = bool(remerge and radius_um is not None)
        self.pa = self.pb = np.empty(0, np.int64)
        if self.remerge and self.n_sites > 1:
            pa, pb = [], []
            for lib in np.unique(site_lib):  # pairs of sites within the radius, never across libraries
                ix = np.flatnonzero(site_lib == lib)
                if len(ix) < 2:
                    continue
                pr = cKDTree(site_xy[ix]).query_pairs(radius_um, output_type="ndarray")
                if len(pr):
                    pa.append(ix[pr[:, 0]])
                    pb.append(ix[pr[:, 1]])
            if pa:
                self.pa, self.pb = np.concatenate(pa), np.concatenate(pb)

    def units(self, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit (locus after re-merging) of every site and the label of every unit."""
        site_lab = lab[self.site_locus]
        if self.remerge and len(self.pa):
            same = site_lab[self.pa] == site_lab[self.pb]
            adj = coo_matrix(
                (np.ones(int(same.sum()), np.int8), (self.pa[same], self.pb[same])), shape=(self.n_sites, self.n_sites)
            )
            n_unit, comp = connected_components(adj, directed=False)
        else:
            comp, n_unit = self.site_locus, self.n_loci
        first = np.full(n_unit, -1, np.int64)
        first[comp[::-1]] = np.arange(self.n_sites)[::-1]
        return comp, site_lab[first]

    def __call__(self, lab: np.ndarray, min_loci: int, min_partner_loci: int) -> tuple[int, int, int]:
        """(tested clonotypes, clonotypes sharing a partner in >= 2 of their loci, clonotypes with >= min_loci loci)."""
        comp, unit_lab = self.units(lab)
        n_lab = int(lab.max()) + 1 if len(lab) else 0
        n_units = np.bincount(unit_lab, minlength=n_lab)
        with_p = np.zeros(n_lab, np.int64)
        maxc = np.zeros(n_lab, np.int64)
        if len(self.sp_site):
            up = np.unique(comp[self.sp_site] * np.int64(self.n_partner) + self.sp_partner)  # distinct (unit, partner)
            u_of, p_of = up // self.n_partner, up % self.n_partner
            with_p = np.bincount(unit_lab[np.unique(u_of)], minlength=n_lab)
            keys, cnt = np.unique(unit_lab[u_of] * np.int64(self.n_partner) + p_of, return_counts=True)
            np.maximum.at(maxc, keys // self.n_partner, cnt)
        multi = n_units >= min_loci
        tested = multi & (with_p >= min_partner_loci)
        shared = tested & (maxc >= 2)
        return int(tested.sum()), int(shared.sum()), int(multi.sum())


def read_floor(
    air: pd.DataFrame,
    *,
    locus: str = "TRB",
    partner_locus: str = "TRA",
    floors: Sequence[float] = (2, 5, 10, 20, 50, 100),
    radius_um: float | None = 15.0,
    sides: Sequence[str] = _SIDES,
    sink_min_clones: int | None = 20,
    min_loci: int = 2,
    min_partner_loci: int = 2,
    remerge: bool = True,
    n_perm: int = 200,
    alpha: float = 0.05,
    multiplicity: Literal["bh"] | None = "bh",
    rng: RngLike = None,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
) -> ReadFloorResult:
    """Partner-chain sharing of the loci of one clonotype, observed vs shuffled, at a range of read floors.

    Records of ``locus`` are collapsed to (library, spot, clone) sites (reads = the best record) and merged into loci
    (:func:`spaclone.pp.merge_loci`: same-clone sites within ``radius_um``, single linkage). A locus is on the
    ``"above"`` side of a floor when its best record has at least that many reads (the TR1 membership rule), else on
    the ``"below"`` side. For every floor and side, a clonotype is *tested* when it has at least ``min_loci`` loci of
    that side, at least ``min_partner_loci`` of which hold a partner-chain clonotype on their spots, and it *shares*
    when one partner clonotype is found at two or more of its loci. The statistic is the share of tested clonotypes
    that share.

    Parameters
    ----------
    air
        Receptor table with both chains: spot, coordinates (µm), locus, clone (clonotype) and read count per record
        (e.g. valid contigs with ``consensus_count``). Records with a missing clone or read count are dropped.
    locus, partner_locus
        The focal chain whose loci are tested and the partner chain looked up on their spots.
    floors
        Read floors (reads of a locus' best record).
    radius_um
        Locus radius for :func:`~spaclone.pp.merge_loci`. ``None`` uses the sites themselves as units, without any
        merging (the archive's bin-level diagnostic): shuffled sites of one clonotype that land on the same spot then
        count that spot's partners twice.
    sides
        Which sides of each floor to test: ``"above"`` (loci with >= floor reads) and / or ``"below"``.
    sink_min_clones
        Remove records on barcode sinks first: spots holding at least this many distinct clones (all loci, per
        library; :func:`spaclone.pp.barcode_sinks`). ``None`` keeps them. A sink holds copies of the abundant clones of
        both chains, so an abundant clonotype on several sinks "shares" the abundant partners there: on simulated
        chips without any genuine record, four sinks made the 2-read floor look informative in 38% of chips (0%
        without them).
    min_loci, min_partner_loci
        Clonotype filters, see above. ``min_partner_loci=0`` counts every clonotype with ``min_loci`` loci (the TR4
        definition of a matching alpha chain among expanded clonotypes).
    remerge
        Re-apply the locus rule to every shuffled data set: loci that receive the same clonotype and lie within
        ``radius_um`` of each other become one locus, as they would have in real data. ``False`` keeps the loci as
        they are (the TR4 chance model).
    n_perm
        Number of label shuffles per floor and side (a :class:`~spaclone._utils.LowPowerWarning` says when
        ``1 / (n_perm + 1)`` cannot pass Benjamini–Hochberg across the tested floors on its own).
    alpha
        Level used for ``summary["suggested_floor"]``.
    multiplicity
        ``"bh"`` (default): the ``"above"`` sides of the floors are one family; a floor is suggested only if its
        ``"above"`` side passes Benjamini–Hochberg at ``alpha`` across the tested floors (``q_value``). ``None``: raw
        p-values (the legacy rule; with six floors a floor is then suggested on 16–25% of chips without any genuine
        record).
    rng
        Seed or generator; consumed floor by floor, side by side, library by library (one permutation each). The
        result is a deterministic function of the data and the generator.
    spot_key, x_key, y_key, clone_key, locus_key, library_key, reads_key
        Column names.

    Returns
    -------
    ReadFloorResult
        ``table``: one row per (floor, side) with ``n_loci``, ``n_clonotypes``, ``n_multi`` (clonotypes with
        ``min_loci`` loci), ``n_tested``, ``n_shared``, ``statistic`` (share sharing), ``null_mean``, ``null_sd``,
        ``null_lo``, ``null_hi``, ``null_n_shared`` (mean number of sharing clonotypes in the shuffles, the chance
        expectation of ``n_shared``), ``ratio``, ``z``, ``p_value`` (one-sided, greater), ``q_value``
        (Benjamini–Hochberg across the floors of the same side) and ``n_perm``. ``null``: the shuffled shares,
        ``(n_perm, n_rows)``. ``summary["suggested_floor"]``: the largest floor whose ``"below"`` side is testable and
        does not share more than chance (p > ``alpha``) while its ``"above"`` side does (q <= ``alpha`` with
        ``multiplicity="bh"``, p <= ``alpha`` with ``None``); NaN if none ("no floor": read the table).
        ``summary["suggested_floor_unadjusted"]`` gives the legacy rule for comparison.

    Null model
    ----------
    Random labelling of loci: the clonotype labels of the loci of one side of a floor are permuted among those loci,
    inside each library. Every locus keeps its position, its spots and the partner-chain clonotypes on them, and
    every clonotype keeps its number of loci; so a clonotype's loci become a random set of loci of the same read
    support. With ``remerge`` the locus rule is applied again to each shuffled data set. Every (floor, side) has its
    own shuffles; the ``"above"`` sets of the floors are nested and their statistics positively dependent, under
    which Benjamini–Hochberg keeps its level (Benjamini & Yekutieli 2001).

    Notes
    -----
    Barcode sinks hold copies of the abundant clones of both chains, so the loci of an abundant clonotype on several
    sinks share the abundant partner clonotypes there; they are removed first (``sink_min_clones``).

    Legacy setting: ``multiplicity=None, sink_min_clones=None`` (raw p-values; sinks kept).
    Reproduces ``04_external_validation_2026-10/workflow/diag/stereoir_TR_chimera.py`` with ``radius_um=None`` (bins with
    >= 20 reads: 8.2% share a TRA clonotype vs 0.9% shuffled; bins with < 20 reads: 8.6% vs 7.2%; with sites in the
    same order and the same generator the shuffles are identical draw for draw), and the matching-alpha test of
    ``TR4_expanded_clones.py`` with ``radius_um=15``, ``floors=(20,)``, ``sides=("above",)``, ``min_loci=3``,
    ``min_partner_loci=0`` and ``remerge=False``.

    References
    ----------
    Kebschull & Zador 2015, doi:10.1093/nar/gkv717; Howie et al. 2015, doi:10.1126/scitranslmed.aac5624; Besag &
    Diggle 1977, doi:10.2307/2346974.

    Examples
    --------
    >>> res = spaclone.qc.read_floor(tcr_contigs, locus="TRB", partner_locus="TRA", rng=0)  # doctest: +SKIP
    >>> res.table[["side", "floor", "statistic", "null_mean", "p_value"]]  # doctest: +SKIP
    """
    S.validate_air(air, required=(spot_key, x_key, y_key, clone_key, locus_key, reads_key), numeric=(x_key, y_key))
    sides = tuple(sides)
    bad = [s for s in sides if s not in _SIDES]
    if bad or not sides:
        raise ValueError(f"sides must be a non-empty subset of {_SIDES}, got {sides}")
    floors = [float(f) for f in floors]
    if not floors:
        raise ValueError("floors must not be empty")
    if radius_um is not None and not (np.isfinite(radius_um) and radius_um >= 0):
        raise ValueError("radius_um must be None or a finite number >= 0")
    if n_perm < 1 or min_loci < 2 or min_partner_loci < 0:
        raise ValueError("need n_perm >= 1, min_loci >= 2 and min_partner_loci >= 0")
    if locus == partner_locus:
        raise ValueError("locus and partner_locus must differ")
    if multiplicity not in ("bh", None):
        raise ValueError(f"multiplicity must be 'bh' or None, got {multiplicity!r}")
    gen = as_rng(rng)

    df = air.loc[air[clone_key].notna()]
    lib = df[library_key].astype(str).to_numpy() if library_key in df.columns else np.full(len(df), "", object)
    loc = df[locus_key].astype(str).to_numpy()
    n_sink_spots = n_sink_records = 0
    if sink_min_clones is not None:
        if sink_min_clones < 1:
            raise ValueError("sink_min_clones must be >= 1 (or None to keep sinks)")
        on_sink, n_sink_spots = _sink_records(lib, df[spot_key].to_numpy(), loc, df[clone_key].to_numpy(),
                                             int(sink_min_clones))  # fmt: skip
        n_sink_records = int(on_sink.sum())
        if n_sink_records:
            logger.info("read_floor: %d records on %d barcode sinks removed", n_sink_records, n_sink_spots)
            df, lib, loc = df.loc[~on_sink], lib[~on_sink], loc[~on_sink]
    reads = pd.to_numeric(df[reads_key], errors="coerce").to_numpy(float)
    focal = (loc == locus) & np.isfinite(reads)
    if ((loc == locus) & ~np.isfinite(reads)).any():
        logger.info(
            "read_floor: %d %s records without a read count dropped", int(((loc == locus) & ~focal).sum()), locus
        )
    rec = pd.DataFrame(
        {
            "_lib": lib[focal],
            spot_key: df[spot_key].to_numpy()[focal],
            clone_key: df[clone_key].to_numpy()[focal],
            x_key: df[x_key].to_numpy(float)[focal],
            y_key: df[y_key].to_numpy(float)[focal],
            reads_key: reads[focal],
        }
    )
    if rec.empty:
        raise ValueError(f"no {locus} records with a clone and a read count")
    # sites: one row per (library, spot, clone) in order of first appearance, reads of the best record
    site_id = rec.groupby(["_lib", spot_key, clone_key], sort=False).ngroup().to_numpy()
    first = np.full(int(site_id.max()) + 1, -1, np.int64)
    first[site_id[::-1]] = np.arange(len(rec))[::-1]
    sites = rec.iloc[first].reset_index(drop=True)
    sites[reads_key] = pd.Series(rec[reads_key].to_numpy()).groupby(site_id).max().to_numpy()
    # loci
    if radius_um is None:
        site_locus = np.arange(len(sites))
        loci = sites.assign(n_spots=1, reads_max=sites[reads_key])
    else:
        loci, member = merge_loci(
            sites, radius_um=radius_um, clone_key=clone_key, by="_lib", spot_key=spot_key, x_key=x_key, y_key=y_key,
            reads_key=reads_key,
        )  # fmt: skip
        site_locus = member.to_numpy()
    locus_reads = loci["reads_max"].to_numpy(float)
    locus_lib = loci["_lib"].astype(str).to_numpy()
    # labels: (library, clonotype) codes of the loci
    locus_code = pd.MultiIndex.from_arrays([locus_lib, loci[clone_key].to_numpy()]).factorize()[0].astype(np.int64)
    # partner clonotypes on every site's spot
    part = loc == partner_locus
    pkey = pd.MultiIndex.from_arrays([lib[part], df[spot_key].to_numpy()[part]])
    pcode = pd.factorize(df[clone_key].to_numpy()[part])[0].astype(np.int64)
    skey = pd.MultiIndex.from_arrays([sites["_lib"].to_numpy(), sites[spot_key].to_numpy()])
    spot_index = skey.unique()
    p_spot = spot_index.get_indexer(pkey)
    keep_p = p_spot >= 0
    sp = pd.DataFrame({"s": p_spot[keep_p], "p": pcode[keep_p]}).drop_duplicates()
    s_of_site = spot_index.get_indexer(skey)
    # site -> partner entries (a spot's partners are shared by all sites on it)
    order = np.argsort(sp.s.to_numpy(), kind="stable")
    sp_s, sp_p = sp.s.to_numpy()[order], sp.p.to_numpy()[order]
    ptr = np.searchsorted(sp_s, np.arange(len(spot_index) + 1))
    lens = ptr[s_of_site + 1] - ptr[s_of_site]
    e_site = np.repeat(np.arange(len(sites)), lens)
    e_pos = np.repeat(ptr[s_of_site] - (np.cumsum(lens) - lens), lens) + np.arange(int(lens.sum()))
    e_partner = sp_p[e_pos]
    n_partner = int(pcode.max()) + 1 if len(pcode) else 0
    site_xy = np.column_stack([sites[x_key].to_numpy(float), sites[y_key].to_numpy(float)])
    site_lib = sites["_lib"].astype(str).to_numpy()
    n_part_locus = np.zeros(len(loci), np.int64)
    if len(e_site):
        lp = np.unique(site_locus[e_site] * np.int64(max(n_partner, 1)) + e_partner)
        n_part_locus = np.bincount(lp // max(n_partner, 1), minlength=len(loci))

    rows: list[dict[str, Any]] = []
    nulls: list[np.ndarray] = []
    for f in floors:
        for side in sides:
            sel = locus_reads >= f if side == "above" else locus_reads < f
            sel_loci = np.flatnonzero(sel)
            row = {"floor": f, "side": side, "n_loci": len(sel_loci)}
            new_index = np.full(len(loci), -1, np.int64)
            new_index[sel_loci] = np.arange(len(sel_loci))
            in_sel = new_index[site_locus] >= 0
            sel_sites = np.flatnonzero(in_sel)
            site_new = np.full(len(sites), -1, np.int64)
            site_new[sel_sites] = np.arange(len(sel_sites))
            ep = in_sel[e_site]
            engine = _Sharing(
                new_index[site_locus[sel_sites]],
                site_xy[sel_sites],
                site_lib[sel_sites],
                (site_new[e_site[ep]], e_partner[ep]),
                n_partner,
                radius_um,
                remerge,
            )
            lab0 = pd.factorize(locus_code[sel_loci])[0].astype(np.int64)
            row["n_clonotypes"] = int(lab0.max()) + 1 if len(lab0) else 0
            null = np.full(n_perm, np.nan)
            null_shared = np.full(n_perm, np.nan)
            if len(sel_loci) < 2:
                rows.append({**row, "n_multi": 0, "n_tested": 0, "n_shared": 0, "statistic": np.nan, "n_perm": n_perm})
                nulls.append(null)
                continue
            n_t, n_s, n_m = engine(lab0, min_loci, min_partner_loci)
            groups = [np.flatnonzero(locus_lib[sel_loci] == v) for v in sorted(pd.unique(locus_lib[sel_loci]))]
            for b in range(n_perm):
                lab = lab0.copy()
                for g in groups:
                    lab[g] = lab0[g[gen.permutation(len(g))]]
                t_b, s_b, _ = engine(lab, min_loci, min_partner_loci)
                null[b] = s_b / t_b if t_b else np.nan
                null_shared[b] = s_b
            stat = n_s / n_t if n_t else np.nan
            if np.isfinite(null).sum() >= 2:
                mu, sd, lo, hi = (float(v[0]) for v in null_summary(null))
            else:  # nothing testable in the shuffles
                mu = sd = lo = hi = np.nan
            rows.append(
                {
                    **row,
                    "n_multi": n_m,
                    "n_tested": n_t,
                    "n_shared": n_s,
                    "statistic": stat,
                    "null_mean": mu,
                    "null_sd": sd,
                    "null_lo": lo,
                    "null_hi": hi,
                    "null_n_shared": float(np.mean(null_shared)),
                    "ratio": stat / mu if mu > 0 else np.nan,
                    "z": (stat - mu) / sd if sd > 0 else np.nan,
                    "p_value": float(mc_pvalue(null, stat, alternative="greater")[0]) if n_t and mu == mu else np.nan,
                    "n_perm": n_perm,
                }
            )
            nulls.append(null)
    cols = [
        "floor", "side", "n_loci", "n_clonotypes", "n_multi", "n_tested", "n_shared", "statistic", "null_mean",
        "null_sd", "null_lo", "null_hi", "null_n_shared", "ratio", "z", "p_value", "q_value", "n_perm",
    ]  # fmt: skip
    table = pd.DataFrame(rows).reindex(columns=cols)
    for side in sides:  # BH across the floors of one side
        m = (table.side == side).to_numpy()
        table.loc[m, "q_value"] = bh(table.loc[m, "p_value"].to_numpy(dtype=float))
    m_above = int(table.loc[table.side == "above", "p_value"].notna().sum())
    if multiplicity == "bh" and "below" in sides and m_above > 1 and 1 / (n_perm + 1) > alpha / m_above:
        warnings.warn(
            f"read_floor: the smallest p-value 1/{n_perm + 1} cannot pass Benjamini-Hochberg at {alpha} across "
            f"{m_above} floors on its own; raise n_perm to at least {int(np.ceil(m_above / alpha))}",
            LowPowerWarning,
            stacklevel=2,
        )
    t = table.set_index(["side", "floor"])

    def _suggest(adjusted: bool) -> float:
        ok = []
        for f in floors:
            if ("above", f) in t.index and ("below", f) in t.index:
                a, b = t.loc[("above", f)], t.loc[("below", f)]
                pa = a.q_value if adjusted else a.p_value
                if pa <= alpha and b.n_tested > 0 and b.p_value > alpha:
                    ok.append(f)
        return float(max(ok)) if ok else np.nan

    summary: dict[str, Any] = {
        "n_loci": len(loci),
        "n_clonotypes": int(locus_code.max()) + 1,
        "suggested_floor": _suggest(multiplicity == "bh"),
        "suggested_floor_unadjusted": _suggest(False),
        "multiplicity": multiplicity,
        "n_sink_spots": n_sink_spots,
        "n_sink_records": n_sink_records,
    }
    for (side, f), r in t.iterrows():
        summary[f"share[{side}|{f:g}]"] = r.statistic
        summary[f"null[{side}|{f:g}]"] = r.null_mean
    params = {
        "locus": locus,
        "partner_locus": partner_locus,
        "floors": floors,
        "radius_um": radius_um,
        "sides": sides,
        "sink_min_clones": sink_min_clones,
        "min_loci": min_loci,
        "min_partner_loci": min_partner_loci,
        "remerge": remerge,
        "n_perm": n_perm,
        "alpha": alpha,
        "multiplicity": multiplicity,
        "design": "Permute(clone label among the loci of one side of a floor, within library)",
        "alternative": "greater",
        "rng_entropy": rng_entropy(gen),
    }
    loci_out = pd.DataFrame(
        {
            "library_id": locus_lib,
            clone_key: loci[clone_key].to_numpy(),
            x_key: loci[x_key].to_numpy(float),
            y_key: loci[y_key].to_numpy(float),
            "n_spots": loci["n_spots"].to_numpy(),
            "reads_max": locus_reads,
            "n_partners": n_part_locus,
        }
    )
    return ReadFloorResult(
        name="read_floor", table=table, summary=summary, params=params, null=np.column_stack(nulls), loci=loci_out
    )
