"""Dandelion clone calls on spatial receptor tables, made reproducible and mapped back to the records.

:func:`find_clones` runs ``dandelion.tl.find_clones`` (Stephenson et al. 2021; Suo et al. 2024) on a spatial receptor
table and writes the clone of every record back into the table, so that loci, maps, hotspots and scans run on
Dandelion's clones. :func:`compare_clones` measures the agreement of two clone calls per chain (adjusted Rand index,
split and merged clones), e.g. Dandelion against :func:`spaclone.pp.define_clones`.

Two facts about Dandelion matter for reproducible clone calls:

* **Version.** sc-dandelion 1.0 (October 2026) groups junctions by connected components (single linkage) within V
  gene, J gene and junction length. Up to 0.5.x a junction was grouped with its direct neighbours only, the
  neighbourhoods overlapped, and a record took the last of its neighbourhoods in size order: a chain A ~ B ~ C (A
  and C too far apart) is one clone in 1.0 and two in 0.5 (B goes with C, A stays alone). 1.0 therefore calls fewer,
  larger clones. With records as units and the default identity, Dandelion >= 1.0 gives the same partition as
  ``spaclone.pp.define_clones(method="vj_length_identity")``. ``dandelion.__version__`` is unreliable (it reports
  ``0.1.dev9`` for both releases), so the version is read from the package metadata and recorded.
* **Hash seed.** Clone numbering, and in 0.5.x clone membership, follows Python's iteration order over sets of
  strings, which changes between sessions unless ``PYTHONHASHSEED`` is fixed. :func:`find_clones` therefore runs
  Dandelion in a child process with ``PYTHONHASHSEED=0`` (``deterministic=True``, the default) and the parent's
  import path, so the same table always gives the same ids.

What is a cell? Dandelion pairs heavy and light chains within a cell. Spots and beads hold several cells and a cell
spreads over several spots, so the default ``unit="record"`` makes every record its own "cell" and Dandelion
clusters each chain on its own (as the archive's S4 did for IGH). ``unit="spot"`` or a segmented-cell column asks
Dandelion to refine clones by the light chain within the unit, which is only meaningful when the unit is one cell.

sc-dandelion is optional: ``pip install 'sc-dandelion>=0.5.7'`` (the ``spaclone[dandelion]`` extra; 0.5.x also
needs ``matplotlib<3.9``, which it does not declare).

References
----------
Stephenson et al. 2021, doi:10.1038/s41591-021-01329-2 (Dandelion); Suo et al. 2024, doi:10.1038/s41587-023-01734-7
(Dandelion VDJ processing); Gupta et al. 2017, doi:10.4049/jimmunol.1601850 (clonal grouping by junction
similarity); Hubert & Arabie 1985, doi:10.1007/BF01908075 (adjusted Rand index).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import SpacloneWarning

__all__ = [
    "DandelionVersionWarning",
    "compare_clones",
    "dandelion_version",
    "find_clones",
    "to_dandelion_table",
]

logger = logging.getLogger("spaclone")

#: Columns Dandelion needs to build its metadata.
REQUIRED = ("sequence_id", "cell_id", "locus", "productive", "v_call", "d_call", "j_call", "c_call", "umi_count")
_IG = ("IGH", "IGK", "IGL")
_TR = ("TRA", "TRB", "TRG", "TRD")


class DandelionVersionWarning(SpacloneWarning):
    """sc-dandelion < 1.0 was used: its clone calls are order-dependent and differ from 1.0 and from spaclone's."""


def dandelion_version(python: str | os.PathLike | None = None) -> str | None:
    """Installed sc-dandelion version from the package metadata (``dandelion.__version__`` is unreliable).

    Parameters
    ----------
    python
        Interpreter to ask (default: this one).

    Returns
    -------
    str or None
        e.g. ``"1.0.1"``; None when sc-dandelion is not installed there.
    """
    if python is None:
        try:
            return metadata.version("sc-dandelion")
        except metadata.PackageNotFoundError:
            return None
    code = (
        "import importlib.metadata as m\n"
        "try:\n    print(m.version('sc-dandelion'))\nexcept m.PackageNotFoundError:\n    print('')"
    )
    r = subprocess.run([str(python), "-c", code], check=False, capture_output=True, text=True, timeout=120)
    v = r.stdout.strip()
    return v or None


def _major(version: str | None) -> int:
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return 0


def _flag(values: pd.Series) -> np.ndarray:
    """Boolean view of a productive column: True for True, 1, "T", "TRUE", "Y", "YES" (any case)."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    text = values.astype(str).str.strip().str.upper()
    return text.isin(["T", "TRUE", "1", "1.0", "Y", "YES"]).to_numpy(dtype=bool)


def to_dandelion_table(
    air: pd.DataFrame,
    *,
    unit: str = "record",
    radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    v_key: str = S.V_CALL,
    d_key: str = "d_call",
    j_key: str = S.J_CALL,
    c_key: str = S.C_CALL,
    junction_aa_key: str = S.JUNCTION_AA,
    junction_key: str = S.JUNCTION,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> pd.DataFrame:
    """The receptor records as a Dandelion (AIRR) table with unique ids and a cell column.

    Parameters
    ----------
    air
        Receptor table.
    unit
        Dandelion's "cell": ``"record"`` (every record; chains are clustered independently), ``"spot"`` (all records
        of a spot, with the library), ``"locus"`` (same-clone spots within ``radius_um``, needs ``clone_key``) or the
        name of a column (e.g. segmented cells from :func:`spaclone.pp.assign_to_cells`).
    radius_um
        Locus radius for ``unit="locus"``.
    spot_key, x_key, y_key, clone_key, locus_key, library_key, v_key, d_key, j_key, c_key, junction_aa_key, junction_key, productive_key, reads_key
        Column names.

    Returns
    -------
    pandas.DataFrame
        One row per usable record (IG / TR locus, V call, J call and a junction): ``sequence_id`` (``"r<row>"``, the
        row position in ``air``; unique, so contig-id collisions between libraries cannot mix cells), ``cell_id``,
        ``locus``, ``v_call``, ``d_call``, ``j_call``, ``c_call``, ``junction``, ``junction_aa``, ``productive``
        (``"T"`` / ``"F"``), ``umi_count`` (reads, or 1), plus ``sequence_alignment`` / ``germline_alignment`` when
        present. ``attrs["spaclone_rows"]`` holds the row positions.
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a receptor table, got {type(air).__name__}")
    for c in (locus_key, v_key, j_key):
        if c not in air.columns:
            raise KeyError(f"to_dandelion_table: column {c!r} not in the table")
    if junction_aa_key not in air.columns and junction_key not in air.columns:
        raise KeyError(f"to_dandelion_table: need {junction_aa_key!r} or {junction_key!r}")
    tab = air.reset_index(drop=True)
    rows = np.arange(len(tab), dtype=np.int64)

    def col(name: str, default: Any = "") -> pd.Series:
        if name in tab.columns:
            return tab[name].astype(object).where(tab[name].notna(), default)
        return pd.Series([default] * len(tab), dtype=object)

    if unit == "record":
        cell = pd.Series([f"r{i}" for i in rows], dtype=object)
    elif unit == "spot":
        if spot_key not in tab.columns:
            raise KeyError(f"unit='spot' needs the column {spot_key!r}")
        cell = tab[spot_key].astype(str)
        if library_key in tab.columns:
            cell = tab[library_key].astype(str) + ":" + cell
    elif unit == "locus":
        from spaclone.pp.loci import merge_loci

        ok = tab[clone_key].notna() if clone_key in tab.columns else None
        if ok is None:
            raise KeyError(f"unit='locus' needs the clone column {clone_key!r} (loci are merged per clone)")
        _, member = merge_loci(
            tab[ok],
            radius_um=radius_um,
            clone_key=clone_key,
            spot_key=spot_key,
            x_key=x_key,
            y_key=y_key,
            locus_key=locus_key,
            library_key=library_key,
            reads_key=reads_key,
        )
        cell = pd.Series([None] * len(tab), dtype=object)
        cell[ok.to_numpy()] = ["locus_" + str(v) for v in member.to_numpy()]
    else:
        if unit not in tab.columns:
            raise KeyError(f"unit must be 'record', 'spot', 'locus' or a column name; {unit!r} is not a column")
        cell = tab[unit].astype(object)
    out = pd.DataFrame(
        {
            "sequence_id": [f"r{i}" for i in rows],
            "cell_id": cell.to_numpy(dtype=object),
            "locus": tab[locus_key].astype(str).str.upper().to_numpy(),
            "v_call": col(v_key).to_numpy(),
            "d_call": col(d_key).to_numpy(),
            "j_call": col(j_key).to_numpy(),
            "c_call": col(c_key).to_numpy(),
            "junction": col(junction_key).to_numpy(),
            "junction_aa": col(junction_aa_key).to_numpy(),
            "productive": np.where(_flag(tab[productive_key]) if productive_key in tab.columns else True, "T", "F"),
            "umi_count": (
                tab[reads_key].astype(float).fillna(1.0).to_numpy() if reads_key in tab.columns else np.ones(len(tab))
            ),
        }
    )
    for extra in ("sequence_alignment", "germline_alignment", "v_call_genotyped"):
        if extra in tab.columns:
            out[extra] = col(extra).to_numpy()
    usable = (
        out.locus.isin(_IG + _TR)
        & (out.v_call.astype(str) != "")
        & (out.j_call.astype(str) != "")
        & ((out.junction_aa.astype(str) != "") | (out.junction.astype(str) != ""))
        & pd.Series(cell.notna().to_numpy())
    ).to_numpy(dtype=bool)
    out = out.loc[usable].reset_index(drop=True)
    out.attrs["spaclone_rows"] = rows[usable].tolist()
    return out


_CHILD = r"""
import json, sys, warnings
import pandas as pd
warnings.simplefilter("ignore")
inp, outp, params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
import importlib.metadata as md
import dandelion as ddl
version = md.version("sc-dandelion")
tab = pd.read_parquet(inp)
kw = {}
import inspect
if "verbose" in inspect.signature(ddl.Dandelion.__init__).parameters:
    kw["verbose"] = False
vdj = ddl.Dandelion(tab, **kw)
fkw = dict(identity=params["identity"], by_alleles=params["by_alleles"], key_added=params["key_added"])
if params["key"] is not None:
    fkw["key"] = params["key"]
if "verbose" in inspect.signature(ddl.tl.find_clones).parameters:
    fkw["verbose"] = False
res = ddl.tl.find_clones(vdj, **fkw)
vdj = res if res is not None else vdj
data = vdj.data if hasattr(vdj, "data") else vdj._data
out = pd.DataFrame({"sequence_id": data["sequence_id"].astype(str).to_numpy(),
                    "clone": data[params["key_added"]].astype(str).to_numpy()})
out.to_parquet(outp, index=False)
print(json.dumps({"version": version}))
"""


def _run_child(
    table: pd.DataFrame, params: dict[str, Any], python: str, timeout: float | None
) -> tuple[pd.DataFrame, str]:
    with tempfile.TemporaryDirectory(prefix="spaclone_dandelion_") as tmp:
        inp, outp = Path(tmp) / "in.parquet", Path(tmp) / "out.parquet"
        table.to_parquet(inp, index=False)
        env = dict(os.environ, PYTHONHASHSEED="0")
        if Path(python).resolve() == Path(sys.executable).resolve():
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)  # the parent's import path, also under -I
        else:
            env.pop("PYTHONPATH", None)  # another environment: its own packages only
        r = subprocess.run(
            [str(python), "-s", "-c", _CHILD, str(inp), str(outp), json.dumps(params)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Dandelion failed in the child process ({python}):\n{r.stderr[-3000:]}")
        info = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
        return pd.read_parquet(outp), info["version"]


def _run_here(table: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    import inspect

    try:
        import dandelion as ddl
    except ImportError as e:
        raise ImportError(
            "spaclone.ext.dandelion needs sc-dandelion: pip install 'sc-dandelion>=0.5.7' (spaclone[dandelion])"
        ) from e
    version = dandelion_version() or "unknown"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"verbose": False} if "verbose" in inspect.signature(ddl.Dandelion.__init__).parameters else {}
        vdj = ddl.Dandelion(table, **kw)
        fkw: dict[str, Any] = {
            "identity": params["identity"],
            "by_alleles": params["by_alleles"],
            "key_added": params["key_added"],
        }
        if params["key"] is not None:
            fkw["key"] = params["key"]
        if "verbose" in inspect.signature(ddl.tl.find_clones).parameters:
            fkw["verbose"] = False
        res = ddl.tl.find_clones(vdj, **fkw)
    vdj = res if res is not None else vdj
    data = vdj.data
    out = pd.DataFrame(
        {
            "sequence_id": data["sequence_id"].astype(str).to_numpy(),
            "clone": data[params["key_added"]].astype(str).to_numpy(),
        }
    )
    return out, version


def find_clones(
    air: pd.DataFrame,
    *,
    unit: str = "record",
    identity: float | Mapping[str, float] = 0.85,
    key: str | Mapping[str, str] | None = None,
    by_alleles: bool = False,
    key_added: str = "dandelion_clone_id",
    deterministic: bool = True,
    python: str | os.PathLike | None = None,
    timeout: float | None = None,
    radius_um: float = 15.0,
    spot_key: str = S.SPOT,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    locus_key: str = S.LOCUS,
    library_key: str = S.LIBRARY,
    v_key: str = S.V_CALL,
    d_key: str = "d_call",
    j_key: str = S.J_CALL,
    c_key: str = S.C_CALL,
    junction_aa_key: str = S.JUNCTION_AA,
    junction_key: str = S.JUNCTION,
    productive_key: str = S.PRODUCTIVE,
    reads_key: str = S.READS,
) -> pd.DataFrame:
    """Dandelion clone calls (``dandelion.tl.find_clones``) on a spatial receptor table, one clone per record.

    Parameters
    ----------
    air
        Receptor table with V calls, J calls and junctions.
    unit
        Dandelion's cell: ``"record"`` (default; chains clustered on their own), ``"spot"``, ``"locus"`` or a
        segmented-cell column (see :func:`to_dandelion_table` and the module notes).
    identity, key, by_alleles
        Passed to ``dandelion.tl.find_clones``: junction identity (Dandelion's default 0.85; in 1.x a dict per
        library type ``{"ig": ..., "tr-ab": ..., "tr-gd": ...}`` is accepted), the junction column (``None`` =
        Dandelion's default: ``junction_aa``; in 1.x nucleotide ``junction`` with identity 1 for TCRs) and whether V
        and J alleles must match.
    key_added
        Column written with the clone id.
    deterministic
        Run Dandelion in a child process with ``PYTHONHASHSEED=0`` (same table, same ids in every session). With
        ``False`` it runs in this process and ids may change between sessions.
    python
        Interpreter of the child process (default this one, with this process's import path), e.g. an environment
        with another Dandelion version.
    timeout
        Seconds before the child process is stopped.
    radius_um, spot_key, x_key, y_key, clone_key, locus_key, library_key, v_key, d_key, j_key, c_key, junction_aa_key, junction_key, productive_key, reads_key
        Units and column names (see :func:`to_dandelion_table`).

    Returns
    -------
    pandas.DataFrame
        A copy of ``air`` with ``key_added`` (Dandelion's clone id; missing for records Dandelion could not use) and
        ``attrs["spaclone"]["dandelion"]``: version, algorithm (``"connected components"`` for >= 1.0,
        ``"overlapping neighbourhoods"`` before), unit, parameters, ``PYTHONHASHSEED``.

    Raises
    ------
    ImportError
        When sc-dandelion is not installed (in-process run).
    RuntimeError
        When Dandelion fails in the child process (its error is included).

    Warns
    -----
    DandelionVersionWarning
        When sc-dandelion < 1.0 was used.

    Examples
    --------
    >>> air = find_clones(air)  # doctest: +SKIP
    >>> loci, member = spaclone.pp.merge_loci(air.dropna(subset=["dandelion_clone_id"]),
    ...                                       clone_key="dandelion_clone_id")  # doctest: +SKIP
    """
    table = to_dandelion_table(
        air,
        unit=unit,
        radius_um=radius_um,
        spot_key=spot_key,
        x_key=x_key,
        y_key=y_key,
        clone_key=clone_key,
        locus_key=locus_key,
        library_key=library_key,
        v_key=v_key,
        d_key=d_key,
        j_key=j_key,
        c_key=c_key,
        junction_aa_key=junction_aa_key,
        junction_key=junction_key,
        productive_key=productive_key,
        reads_key=reads_key,
    )
    rows = np.asarray(table.attrs.pop("spaclone_rows"), dtype=np.int64)
    params = {
        "identity": dict(identity) if isinstance(identity, Mapping) else float(identity),
        "key": dict(key) if isinstance(key, Mapping) else key,
        "by_alleles": bool(by_alleles),
        "key_added": "clone_id",
    }
    out = air.copy()
    if len(table) == 0:
        out[key_added] = pd.array([None] * len(air), dtype="string")
        return out
    if deterministic:
        if python is None and dandelion_version() is None:
            raise ImportError(
                "spaclone.ext.dandelion needs sc-dandelion: pip install 'sc-dandelion>=0.5.7' (spaclone[dandelion])"
            )
        res, version = _run_child(table, params, str(python or sys.executable), timeout)
    else:
        if python is not None:
            raise ValueError("python= needs deterministic=True (a child process)")
        res, version = _run_here(table, params)
    look = pd.Series(res["clone"].to_numpy(dtype=object), index=res["sequence_id"].to_numpy(dtype=object))
    clone = look.reindex(table["sequence_id"].to_numpy(dtype=object)).to_numpy(dtype=object)
    clone = np.where(pd.isna(clone) | (clone == "") | (clone == "nan") | (clone == "None"), None, clone)
    col = np.full(len(air), None, dtype=object)
    col[rows] = clone
    out[key_added] = pd.array(col, dtype="string")
    major = _major(version)
    info = {
        "version": version,
        "algorithm": "connected components" if major >= 1 else "overlapping neighbourhoods",
        "unit": unit,
        "identity": params["identity"],
        "key": params["key"],
        "by_alleles": bool(by_alleles),
        "deterministic": bool(deterministic),
        "PYTHONHASHSEED": "0" if deterministic else os.environ.get("PYTHONHASHSEED"),
        "n_records": len(table),
        "n_clones": int(pd.Series(clone).nunique()),
    }
    meta = dict(out.attrs.get("spaclone", {})) if isinstance(out.attrs.get("spaclone"), dict) else {}
    meta["dandelion"] = info
    out.attrs["spaclone"] = meta
    if major < 1:
        warnings.warn(
            f"sc-dandelion {version} (< 1.0) groups each junction with its direct neighbours only and assigns "
            "records in order, so its clones depend on the order of the data and differ from Dandelion >= 1.0 and "
            "from spaclone.pp.define_clones (single linkage); prefer sc-dandelion >= 1.0.",
            DandelionVersionWarning,
            stacklevel=2,
        )
    logger.info("Dandelion %s: %d records, %d clones (%s)", version, len(table), info["n_clones"], unit)
    return out


def compare_clones(
    air: pd.DataFrame,
    *,
    a: str = "dandelion_clone_id",
    b: str = S.CLONE,
    by: str | Sequence[str] | None = S.LOCUS,
) -> pd.DataFrame:
    """Agreement of two clone calls on the same records, per chain.

    Parameters
    ----------
    air
        Receptor table with both clone columns.
    a, b
        The two clone columns (e.g. Dandelion's and :func:`spaclone.pp.define_clones`'s).
    by
        Strata compared separately (default the chain); ``None`` compares all records together.

    Returns
    -------
    pandas.DataFrame
        One row per stratum: ``n_records`` (records with both calls), ``n_clones_a``, ``n_clones_b``,
        ``adjusted_rand`` (1 = the same partition), ``identical`` (the same partition), ``n_split`` (clones of ``b``
        spread over several clones of ``a``), ``n_merged`` (clones of ``a`` holding several clones of ``b``),
        ``largest_a``, ``largest_b`` (records in the largest clone).
    """
    from sklearn.metrics import adjusted_rand_score

    for c in (a, b):
        if c not in air.columns:
            raise KeyError(f"compare_clones: column {c!r} not in the table")
    by_cols = [] if by is None else ([by] if isinstance(by, str) else list(by))
    ok = air[a].notna() & air[b].notna()
    d = air.loc[ok, [*by_cols, a, b]].astype({a: object, b: object})
    groups = d.groupby(by_cols, sort=True, observed=True) if by_cols else [((), d)]
    rows = []
    for k, g in groups:
        ka, kb = pd.factorize(g[a])[0], pd.factorize(g[b])[0]
        pairs = pd.DataFrame({"a": ka, "b": kb}).drop_duplicates()
        n_split = int((pairs.groupby("b").size() > 1).sum())
        n_merged = int((pairs.groupby("a").size() > 1).sum())
        row = dict(zip(by_cols, k if isinstance(k, tuple) else (k,), strict=True))
        row.update(
            n_records=len(g),
            n_clones_a=int(ka.max() + 1) if len(g) else 0,
            n_clones_b=int(kb.max() + 1) if len(g) else 0,
            adjusted_rand=float(adjusted_rand_score(kb, ka)) if len(g) else np.nan,
            identical=bool(n_split == 0 and n_merged == 0),
            n_split=n_split,
            n_merged=n_merged,
            largest_a=int(np.bincount(ka).max()) if len(g) else 0,
            largest_b=int(np.bincount(kb).max()) if len(g) else 0,
        )
        rows.append(row)
    return pd.DataFrame(rows)
