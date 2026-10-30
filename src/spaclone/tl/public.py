"""Public and convergent clonotypes across individuals, and their generation probability.

A clonotype found in two individuals cannot be the same cells: it was made twice, by convergent recombination (public
receptors are usually short, germline-like and of high generation probability, Pgen) or it is an artefact. Two
artefacts are common in repertoire data and are warned about here:

* single-linkage clone definitions (the Dandelion-like ``vj_length_identity`` rule) chain similar junctions, so a large
  clone with many variants has many chances to link to a junction seen in another individual; check with exact
  clonotypes (``clone_key`` = V + J + junction);
* light chains: kappa and lambda junctions are short and little diverse, so unrelated B cells share them; a shared
  light-chain junction is not a public clone.

:func:`public_top_clones` asks whether a sample's largest clones are public more often than its other clones, against
k clones drawn at random from the sample. :func:`pgen` wraps OLGA (optional dependency, extra ``tcr``).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from spaclone import _schema as S
from spaclone._utils import RngLike, SpacloneWarning, TestResult, as_rng, bh, mc_pvalue, null_summary, rng_entropy
from spaclone.pp.clones import gene_name
from spaclone.tl.repertoire import _keys, clone_units

__all__ = ["OLGA_MODELS", "pgen", "public_clonotypes", "public_top_clones"]

#: OLGA default model per (species, chain).
OLGA_MODELS: dict[tuple[str, str], str] = {
    ("mouse", "TRB"): "mouse_T_beta",
    ("mouse", "TRA"): "mouse_T_alpha",
    ("mouse", "IGH"): "mouse_B_heavy",
    ("mouse", "IGK"): "mouse_B_kappa",
    ("mouse", "IGL"): "mouse_B_lambda",
    ("human", "TRB"): "human_T_beta",
    ("human", "TRA"): "human_T_alpha",
    ("human", "IGH"): "human_B_heavy",
    ("human", "IGK"): "human_B_kappa",
    ("human", "IGL"): "human_B_lambda",
}
_VDJ = {"TRB", "IGH", "TRD"}
_MODELS: dict[str, Any] = {}


def _sample_order(values: pd.Series) -> list:
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().unique())
        return [c for c in values.cat.categories if c in present]
    return sorted(values.dropna().unique())


def _olga_model(name: str) -> Any:
    if name in _MODELS:
        return _MODELS[name]
    from pathlib import Path

    import olga
    import olga.generation_probability as olga_pgen
    import olga.load_model as load_model

    path = Path(olga.__file__).parent / "default_models" / name
    args = (
        str(path / "model_params.txt"),
        str(path / "V_gene_CDR3_anchors.csv"),
        str(path / "J_gene_CDR3_anchors.csv"),
    )
    if name.endswith(("_beta", "_heavy")):
        g = load_model.GenomicDataVDJ()
        g.load_igor_genomic_data(*args)
        m = load_model.GenerativeModelVDJ()
        m.load_and_process_igor_model(str(path / "model_marginals.txt"))
        model = olga_pgen.GenerationProbabilityVDJ(m, g)
    else:
        g = load_model.GenomicDataVJ()
        g.load_igor_genomic_data(*args)
        m = load_model.GenerativeModelVJ()
        m.load_and_process_igor_model(str(path / "model_marginals.txt"))
        model = olga_pgen.GenerationProbabilityVJ(m, g)
    _MODELS[name] = model
    return model


def _pgen_one(row: tuple[str, str, str, str]) -> tuple[float, float, int, int]:
    from olga.utils import gene_to_num_str

    name, v, j, junction = row
    model = _olga_model(name)
    vk = isinstance(v, str) and gene_to_num_str(v, "V") in model.V_mask_mapping
    jk = isinstance(j, str) and gene_to_num_str(j, "J") in model.J_mask_mapping
    p_vj = model.compute_aa_CDR3_pgen(junction, v if vk else None, j if jk else None, print_warnings=False)
    p_aa = model.compute_aa_CDR3_pgen(junction, None, None, print_warnings=False)
    return float(p_vj), float(p_aa), int(vk), int(jk)


def _pgen_impl(
    table: pd.DataFrame, species: str, locus_key: str, v_key: str, j_key: str, junction_key: str, n_jobs: int
) -> pd.DataFrame:
    try:
        import olga  # noqa: F401
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("pgen needs OLGA: pip install 'spaclone[tcr]' (or pip install olga)") from err
    missing = [c for c in (locus_key, v_key, j_key, junction_key) if c not in table.columns]
    if missing:
        raise KeyError(f"pgen: column(s) {missing} not in the table")
    model = table[locus_key].astype("string").str.upper().map(lambda c: OLGA_MODELS.get((species, c)))
    rows = pd.DataFrame(
        {
            "model": model.to_numpy(dtype=object),
            "v": gene_name(table[v_key]).to_numpy(dtype=object, na_value=None),
            "j": gene_name(table[j_key]).to_numpy(dtype=object, na_value=None),
            "junction": table[junction_key].astype("string").to_numpy(dtype=object, na_value=None),
        },
        index=table.index,
    )
    ok = rows["model"].notna() & rows["junction"].notna()
    uniq = rows.loc[ok].drop_duplicates()
    work = list(uniq.itertuples(index=False, name=None))
    if n_jobs > 1 and len(work) > 1:
        from multiprocessing import Pool

        with Pool(n_jobs) as pool:
            res = pool.map(_pgen_one, work, chunksize=max(1, len(work) // (4 * n_jobs)))
    else:
        res = [_pgen_one(w) for w in work]
    vals = pd.DataFrame(res, columns=["pgen_vj", "pgen_aa", "v_known", "j_known"], index=uniq.index)
    look = pd.concat([uniq, vals], axis=1)
    out = rows.merge(look, on=["model", "v", "j", "junction"], how="left")
    out.index = table.index
    return out[["pgen_vj", "pgen_aa", "v_known", "j_known", "model"]]


def pgen(
    table: pd.DataFrame,
    *,
    species: Literal["mouse", "human"] = "mouse",
    locus_key: str = S.LOCUS,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Generation probability (OLGA) of every receptor chain in a table.

    Parameters
    ----------
    table
        Rows with a chain, a V call, a J call and an amino-acid junction (the conserved C to the F / W, OLGA's CDR3).
    species
        ``"mouse"`` or ``"human"``; the OLGA default model of the chain is used (:data:`OLGA_MODELS`).
    locus_key, v_key, j_key, junction_key
        Column names. Calls are reduced to the gene of the first call (allele dropped).
    n_jobs
        Worker processes (each loads the models once; on platforms that start workers by spawning, call from a
        ``if __name__ == "__main__":`` block).

    Returns
    -------
    pandas.DataFrame
        Indexed like ``table``: ``pgen_vj`` = P(junction, V gene, J gene), ``pgen_aa`` = P(junction) over all genes,
        ``v_known`` / ``j_known`` (1 if the model knows the gene; otherwise ``pgen_vj`` marginalises over all V or J
        genes, as the archive's TR2 did) and ``model``. Rows whose chain has no model, or without a junction, get NaN.

    Raises
    ------
    ImportError
        If OLGA is not installed (``pip install 'spaclone[tcr]'``).

    References
    ----------
    Sethna et al. 2019, doi:10.1093/bioinformatics/btz035 (OLGA); Marcou et al. 2018,
    doi:10.1038/s41467-018-02832-w (IGoR models).
    """
    return _pgen_impl(table, species, locus_key, v_key, j_key, junction_key, n_jobs)


def _mode_by(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    d = df.dropna(subset=[value])
    cnt = d.groupby([key, value], sort=False, observed=True).size().rename("_n").reset_index()
    cnt = cnt.sort_values([key, "_n", value], ascending=[True, False, True]).drop_duplicates(key)
    return cnt.set_index(key)[value]


def public_clonotypes(
    air: pd.DataFrame,
    *,
    sample_key: str,
    min_samples: int = 2,
    clone_key: str = S.CLONE,
    unit: Literal["locus", "spot", "record"] = "spot",
    pgen: bool | Literal["public"] = False,
    species: Literal["mouse", "human"] = "mouse",
    n_jobs: int = 1,
    radius_um: float = 15.0,
    junction_key: str = S.JUNCTION_AA,
    junction_nt_key: str = S.JUNCTION,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> pd.DataFrame:
    """Clonotypes shared between individuals (samples), with their junction variants and optional Pgen.

    Parameters
    ----------
    air
        Receptor table pooling several individuals, with a clone column whose ids are comparable across samples (for
        :func:`spaclone.pp.define_clones`, keep the sample out of ``by``).
    sample_key
        Column naming the individual (one mouse, one donor). Sections of one individual should share a sample value.
    min_samples
        Samples a clonotype must be found in to count as public.
    clone_key
        Clonotype column: an exact clonotype (V + J + junction), a single-linkage clone id, or a junction column to
        compare junctions alone.
    unit
        Unit of the size columns (see :func:`spaclone.tl.clone_units`); presence needs one unit.
    pgen
        ``True``: OLGA Pgen of every clonotype's main chain (V, J and junction, see :func:`pgen`); ``"public"``: of the
        public ones only; ``False``: none.
    species
        Species of the OLGA models.
    n_jobs
        Worker processes for Pgen.
    radius_um
        Linkage distance of locus units in µm.
    junction_key, junction_nt_key, v_key, j_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names; the junction, nucleotide junction and gene columns are optional.

    Returns
    -------
    pandas.DataFrame
        One row per clonotype: the clone, ``n_samples``, ``samples`` (``;``-joined, in sample order: categorical order
        if the sample column is categorical, else sorted), ``public``, ``n_units``, ``max_units`` (units in its
        largest sample), ``n_records``, the main chain, ``junction`` (most common), ``n_junctions`` (distinct amino-acid
        junctions in the clone), ``n_junction_nt`` (distinct nucleotide junctions: convergent recombination when the
        amino-acid junction is the same), ``v_gene``, ``j_gene`` and, with ``pgen``, ``pgen_vj`` / ``pgen_aa``.
        Sorted by decreasing ``n_samples`` and ``n_units``.

    Warns
    -----
    SpacloneWarning
        When light chains (IGK / IGL) are present, or when public clones group several junctions (single linkage).

    References
    ----------
    Venturi et al. 2006, doi:10.1073/pnas.0608907103 (convergent recombination drives TCR sharing); Elhanati et al.
    2018, doi:10.1111/imr.12665 (sharing predicted by recombination statistics); Briney et al. 2019,
    doi:10.1038/s41586-019-0879-y (public antibody clonotypes); Sethna et al. 2019, doi:10.1093/bioinformatics/btz035
    (OLGA).
    """
    if sample_key not in air.columns:
        raise KeyError(f"sample column {sample_key!r} not in the table")
    if min_samples < 2:
        raise ValueError("min_samples must be >= 2")
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    cu = clone_units(air, unit=unit, by=[sample_key], radius_um=radius_um, **keys)
    u = cu.units
    order = _sample_order(air[sample_key])
    rank = {s: i for i, s in enumerate(order)}
    per = u.groupby([clone_key, sample_key], observed=True, sort=False).agg(units=("unit_id", "size"))
    per = per.reset_index()
    per["_r"] = per[sample_key].map(rank)
    per = per.sort_values([clone_key, "_r"])
    g = per.groupby(clone_key, sort=False, observed=True)
    tab = pd.DataFrame(
        {
            "n_samples": g.size(),
            "samples": g[sample_key].agg(lambda v: ";".join(map(str, v))),
            "n_units": g["units"].sum(),
            "max_units": g["units"].max(),
        }
    )
    tab.index.name = clone_key
    rec = air.loc[cu.membership.index]
    tab["n_records"] = rec.groupby(clone_key, observed=True).size().reindex(tab.index).to_numpy()
    tab["public"] = tab["n_samples"] >= min_samples
    extra: dict[str, pd.Series] = {}
    if locus_key in rec.columns:
        extra[locus_key] = _mode_by(rec, clone_key, locus_key)
    if junction_key in rec.columns:
        jx = rec[[clone_key]].assign(_j=rec[junction_key].astype("string").to_numpy(dtype=object, na_value=None))
        extra["junction"] = _mode_by(jx, clone_key, "_j")
        extra["n_junctions"] = jx.dropna(subset=["_j"]).groupby(clone_key, observed=True)["_j"].nunique()
    if junction_nt_key in rec.columns:
        nt = rec[[clone_key]].assign(_n=rec[junction_nt_key].astype("string").to_numpy(dtype=object, na_value=None))
        extra["n_junction_nt"] = nt.dropna(subset=["_n"]).groupby(clone_key, observed=True)["_n"].nunique()
    for col, name in ((v_key, "v_gene"), (j_key, "j_gene")):
        if col in rec.columns:
            gg = rec[[clone_key]].assign(_g=gene_name(rec[col]).to_numpy(dtype=object, na_value=None))
            extra[name] = _mode_by(gg, clone_key, "_g")
    for name, s in extra.items():
        tab[name] = s.reindex(tab.index).to_numpy()
    tab = tab.reset_index()
    tab = tab.sort_values(["n_samples", "n_units", clone_key], ascending=[False, False, True]).reset_index(drop=True)
    if locus_key in rec.columns and rec[locus_key].astype("string").str.upper().isin(["IGK", "IGL"]).any():
        warnings.warn(
            "light-chain records (IGK / IGL) present: kappa and lambda junctions are shared by unrelated B cells, so a "
            "light chain found in several individuals is not evidence of a public clone; use heavy chains",
            SpacloneWarning,
            stacklevel=2,
        )
    if "n_junctions" in tab.columns and (tab.loc[tab["public"], "n_junctions"] > 1).any():
        warnings.warn(
            "public clones group several junctions: single-linkage clone definitions chain similar junctions, so large "
            "clones link to other individuals more easily; repeat with exact clonotypes (V + J + junction)",
            SpacloneWarning,
            stacklevel=2,
        )
    if pgen:
        need = [c for c in (locus_key, "v_gene", "j_gene", "junction") if c not in tab.columns]
        if need:
            raise KeyError(f"pgen needs the chain, V, J and junction columns; missing {need}")
        sel = tab["public"] if pgen == "public" else pd.Series(True, index=tab.index)
        p = _pgen_impl(tab.loc[sel], species, locus_key, "v_gene", "j_gene", "junction", n_jobs)
        tab["pgen_vj"] = p["pgen_vj"].reindex(tab.index)
        tab["pgen_aa"] = p["pgen_aa"].reindex(tab.index)
    return tab


def public_top_clones(
    air: pd.DataFrame,
    *,
    sample_key: str,
    top_k: int = 10,
    n_draws: int = 10_000,
    min_samples: int = 2,
    clone_key: str = S.CLONE,
    unit: Literal["locus", "spot", "record"] = "spot",
    rng: RngLike = None,
    radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    reads_key: str = S.READS,
    locus_id_key: str = S.LOCUS_ID,
) -> TestResult:
    """Are a sample's largest clones found in other individuals more often than its other clones?

    Parameters
    ----------
    air
        Receptor table pooling several individuals (see :func:`public_clonotypes`).
    sample_key
        Column naming the individual.
    top_k
        Number of largest clones per sample (by units; ties broken by records, then by clone id).
    n_draws
        Random draws of ``top_k`` clones per sample.
    min_samples
        A clone is public when found in at least this many samples.
    clone_key
        Clonotype column.
    unit
        Unit of the clone sizes (see :func:`spaclone.tl.clone_units`).
    rng
        Seed or generator; samples are visited in sample order (categorical order if categorical, else sorted) and
        each gets ``n_draws`` draws ``rng.choice(n_clones, top_k, replace=False)`` in turn.
    radius_um
        Linkage distance of locus units in µm.
    spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key
        Column names.

    Returns
    -------
    TestResult
        ``table``: one row per sample and a pooled row (``sample = "all"``, the sum over samples, null draws summed
        draw by draw) with ``n_clones``, ``n_public``, ``top_k``, ``statistic`` (public clones among the top k),
        ``expected`` (k * public / clones), ``null_mean``, ``null_sd``, ``null_lo`` / ``null_hi`` (2.5% and 97.5%),
        ``z``, ``p_value`` (Monte-Carlo, more sharing), ``p_hypergeom`` (exact upper tail of the hypergeometric null,
        per sample), ``q_value`` (BH over samples) and ``n_perm``.

    Notes
    -----
    Null model: within each sample, the top-k label is given to k clones drawn uniformly at random, without
    replacement, from the sample's clones; the clones' public status, sizes and samples stay fixed. The number of public
    clones among them is then hypergeometric. A large clone carries more junction variants, so under single-linkage
    clone definitions it has more chances to link to another individual; repeat with exact clonotypes before reading
    an excess as biology.

    References
    ----------
    Phipson & Smyth 2010, doi:10.2202/1544-6115.1585; Benjamini & Hochberg 1995,
    doi:10.1111/j.2517-6161.1995.tb02031.x; Venturi et al. 2006, doi:10.1073/pnas.0608907103.
    """
    if sample_key not in air.columns:
        raise KeyError(f"sample column {sample_key!r} not in the table")
    if top_k < 1 or n_draws < 1:
        raise ValueError("top_k and n_draws must be >= 1")
    gen = as_rng(rng)
    keys = _keys(clone_key, spot_key, x_key, y_key, locus_key, library_key, reads_key, locus_id_key)
    u = clone_units(air, unit=unit, by=[sample_key], radius_um=radius_um, **keys).units
    size = u.groupby([sample_key, clone_key], observed=True).agg(size=("unit_id", "size"), rec=("n_records", "sum"))
    size = size.reset_index()
    n_samp = size.groupby(clone_key, observed=True)[sample_key].nunique()
    size["public"] = size[clone_key].map(n_samp).to_numpy() >= min_samples
    rows, nulls = [], []
    for s in _sample_order(air[sample_key]):
        k = size[size[sample_key] == s].sort_values([clone_key])
        k = k.sort_values(["size", "rec"], ascending=False)
        sh = k["public"].to_numpy(dtype=np.int64)
        n = len(sh)
        kk = min(top_k, n)
        obs = int(sh[:kk].sum())
        nl = np.array([sh[gen.choice(n, kk, replace=False)].sum() for _ in range(n_draws)], dtype=float)
        nulls.append(nl)
        big_k = int(sh.sum())
        rows.append(
            {
                "sample": s,
                "n_clones": n,
                "n_public": big_k,
                "top_k": kk,
                "statistic": obs,
                "expected": kk * big_k / n if n else np.nan,
                "p_hypergeom": float(hypergeom.sf(obs - 1, n, big_k, kk)) if n else np.nan,
            }
        )
    null = np.column_stack(nulls + [np.sum(nulls, axis=0)]) if nulls else np.empty((n_draws, 0))
    rows.append(
        {
            "sample": "all",
            "n_clones": sum(r["n_clones"] for r in rows),
            "n_public": sum(r["n_public"] for r in rows),
            "top_k": sum(r["top_k"] for r in rows),
            "statistic": sum(r["statistic"] for r in rows),
            "expected": sum(r["expected"] for r in rows),
            "p_hypergeom": np.nan,
        }
    )
    table = pd.DataFrame(rows)
    mu, sd, lo, hi = null_summary(null)
    table["null_mean"], table["null_sd"], table["null_lo"], table["null_hi"] = mu, sd, lo, hi
    with np.errstate(divide="ignore", invalid="ignore"):
        table["z"] = (table["statistic"] - table["null_mean"]) / table["null_sd"]
    table["p_value"] = mc_pvalue(null, table["statistic"].to_numpy(dtype=float), alternative="greater")
    q = np.full(len(table), np.nan)
    q[:-1] = bh(table["p_value"].to_numpy()[:-1])
    table["q_value"] = q
    table["n_perm"] = n_draws
    cols = ["sample", "n_clones", "n_public", "top_k", "statistic", "expected", "null_mean", "null_sd", "null_lo"]
    table = table[[*cols, "null_hi", "z", "p_value", "p_hypergeom", "q_value", "n_perm"]]
    pooled = table.iloc[-1]
    summary = {
        "samples": len(table) - 1,
        "top_public": int(pooled["statistic"]),
        "expected": float(pooled["null_mean"]),
        "p_value": float(pooled["p_value"]),
    }
    params = {
        "sample_key": sample_key,
        "top_k": top_k,
        "n_draws": n_draws,
        "min_samples": min_samples,
        "clone_key": clone_key,
        "unit": unit,
        "rng_entropy": rng_entropy(gen),
    }
    return TestResult(name="public_top_clones", table=table, summary=summary, params=params, null=null)
