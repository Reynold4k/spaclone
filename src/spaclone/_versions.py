"""What is installed: :func:`show_versions` and :func:`check_install`.

:func:`show_versions` lists the versions of Python, spaclone, its dependencies, the optional packages and the external
tools (IgBLAST, Change-O, IgPhyML, R with the Immcantation and sciCSR packages), for bug reports and methods sections.
:func:`check_install` says which optional features work in this environment and gives, for each missing one, the exact
command that enables it.

External tools are looked up in this order: an explicit path, the environment variable named in the table below, the
directory of the running Python (programs that pip installed into this environment), then ``PATH``.

=====================  ==========================================================================================
Variable               Meaning
=====================  ==========================================================================================
``SPACLONE_IGBLAST``   ``igblastn`` executable, or an IgBLAST installation directory (``bin/igblastn`` inside)
``SPACLONE_CHANGEO``   directory holding ``MakeDb.py``, ``DefineClones.py`` and ``CreateGermlines.py``
``SPACLONE_IGPHYML``   ``igphyml`` executable
``SPACLONE_RSCRIPT``   ``Rscript`` executable (R with alakazam, shazam, tigger, dowser, scoper)
``SPACLONE_SCICSR_*``  ``SPACLONE_SCICSR_RSCRIPT`` / ``SPACLONE_SCICSR_PYTHON``: the separate environment of the original
                       sciCSR (R 4.3 + Seurat 4; Python 3.8 + CellRank 1.5), when it is not the R above
=====================  ==========================================================================================
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

__all__ = ["FEATURES", "Feature", "InstallReport", "check_install", "external_env", "find_tool", "show_versions"]

#: Where spaclone is installed from: its name on PyPI (``pip install spaclone``).
SOURCE = "spaclone"

#: Optional dependencies of each extra, used when the package metadata are not available (running from a source
#: tree). ``tests/test_versions.py`` checks that this mirrors ``pyproject.toml``.
EXTRAS_FALLBACK: dict[str, tuple[str, ...]] = {
    "bam": ("pysam>=0.22",),
    "images": ("tifffile>=2023.1",),
    "immcantation": ("changeo>=1.3.5",),
    "scirpy": ("scirpy>=0.20", "awkward>=2", "mudata>=0.3"),
    "squidpy": ("squidpy>=1.6",),
    "spatialdata": ("spatialdata>=0.2", "shapely>=2"),
    "dandelion": ("sc-dandelion>=0.5.7",),
    "scicsr": (
        "cellrank>=2,<2.3; python_version < '3.12'",
        "cellrank>=2.3; python_version >= '3.12'",
        "deeptime>=0.4",
        "pysam>=0.22",
    ),
    "threadfin": ("threadfin>=4.5",),
    "leiden": ("leidenalg>=0.10",),
    "tcr": ("olga>=1.3",),
    "datasets": ("pooch>=1.7",),
    "fast": ("numba>=0.59",),
}
#: Extras that only group other extras.
UMBRELLA_EXTRAS: dict[str, tuple[str, ...]] = {
    "scverse": ("scirpy", "squidpy", "spatialdata"),
    "full": (
        "bam",
        "dandelion",
        "datasets",
        "fast",
        "images",
        "immcantation",
        "leiden",
        "scicsr",
        "scverse",
        "tcr",
        "threadfin",
    ),
}
#: Environment variables that point to external tools.
TOOL_ENV = {
    "igblastn": "SPACLONE_IGBLAST",
    "MakeDb.py": "SPACLONE_CHANGEO",
    "DefineClones.py": "SPACLONE_CHANGEO",
    "CreateGermlines.py": "SPACLONE_CHANGEO",
    "igphyml": "SPACLONE_IGPHYML",
    "Rscript": "SPACLONE_RSCRIPT",
}
#: R packages of the Immcantation suite used by :mod:`spaclone.ext.immcantation`.
IMMCANTATION_R = ("alakazam", "shazam", "tigger", "dowser", "scoper", "airr")


@dataclass(frozen=True)
class Feature:
    """One optional feature: a spaclone extra, an external program or an R package set.

    Attributes
    ----------
    name
        Short label.
    kind
        ``"python"`` (a spaclone extra), ``"tool"`` (executables) or ``"R"`` (R packages).
    requires
        Import names (python), executable names (tool) or R package names (R).
    used_by
        What it enables in spaclone.
    extra
        The spaclone extra that installs it (python features).
    install
        Command that installs it (tools and R; python features take the requirements of their extra).
    dists
        Distribution names whose versions are reported (default: ``requires``).
    """

    name: str
    kind: Literal["python", "tool", "R"]
    requires: tuple[str, ...]
    used_by: str
    extra: str | None = None
    install: str = ""
    dists: tuple[str, ...] = field(default_factory=tuple)


#: Every optional feature that :func:`check_install` reports, in report order.
FEATURES: tuple[Feature, ...] = (
    Feature(
        "scirpy",
        "python",
        ("scirpy", "awkward", "mudata"),
        "spaclone.ext.scirpy; io.to_scirpy / io.from_scirpy; tl.clonotype_network(backend='scirpy')",
        extra="scirpy",
    ),
    Feature("squidpy", "python", ("squidpy",), "spaclone.ext.squidpy; gex.spatially_variable_genes", extra="squidpy"),
    Feature(
        "SpatialData",
        "python",
        ("spatialdata", "shapely"),
        "io.to_spatialdata; fast polygon lookup in pp.assign_to_cells",
        extra="spatialdata",
    ),
    Feature(
        "Dandelion",
        "python",
        ("dandelion",),
        "spaclone.ext.dandelion (Dandelion clone calls on spatial records)",
        extra="dandelion",
        dists=("sc-dandelion",),
    ),
    Feature(
        "CellRank",
        "python",
        ("cellrank", "deeptime"),
        "backend='cellrank' of the sciCSR transition models (spaclone.ext.scicsr, spaclone.tl transitions)",
        extra="scicsr",
    ),
    Feature("pysam", "python", ("pysam",), "BAM input of the sciCSR germline-transcript counts", extra="bam"),
    Feature(
        "Threadfin", "python", ("threadfin",), "spaclone.ext.threadfin (Threadfin made spatial)", extra="threadfin"
    ),
    Feature(
        "leidenalg", "python", ("leidenalg",), "scanpy's leidenalg backend (gex.cluster uses igraph)", extra="leiden"
    ),
    Feature("OLGA", "python", ("olga",), "tl.pgen; tl.public_clonotypes(pgen=True)", extra="tcr"),
    Feature("pooch", "python", ("pooch",), "downloads in spaclone.datasets", extra="datasets"),
    Feature("tifffile", "python", ("tifffile",), "io.read_tissue_mask (TIFF masks)", extra="images"),
    Feature("numba", "python", ("numba",), "nothing required (scanpy already installs it)", extra="fast"),
    Feature(
        "IgBLAST",
        "tool",
        ("igblastn",),
        "bcr.run_igblast; ext.immcantation.igblast",
        install="conda install -c conda-forge -c bioconda igblast   (or https://ncbi.github.io/igblast/)",
    ),
    Feature(
        "Change-O",
        "tool",
        ("MakeDb.py", "DefineClones.py", "CreateGermlines.py"),
        "ext.immcantation.makedb / define_clones / create_germlines",
        extra="immcantation",
        install="pip install 'changeo>=1.3.5'   (the spaclone[immcantation] extra; or conda install -c bioconda changeo)",
    ),
    Feature(
        "IgPhyML",
        "tool",
        ("igphyml",),
        "ext.immcantation.dowser_trees(build='igphyml')",
        install="conda install -c conda-forge -c bioconda igphyml",
    ),
    Feature(
        "R + Immcantation",
        "R",
        IMMCANTATION_R,
        "ext.immcantation: SHazaM thresholds and mutations, TIgGER genotypes, dowser trees",
        install=(
            "bash scripts/install_spaclone_env.sh   (or in R: BiocManager::install('pwalign'); "
            "install.packages(c('alakazam', 'shazam', 'tigger', 'dowser', 'scoper')))"
        ),
    ),
    Feature(
        "R + sciCSR",
        "R",
        ("sciCSR",),
        "the R bridge of spaclone.ext.scicsr (the original sciCSR package)",
        install="bash scripts/install_spaclone_env.sh --with-scicsr   (R package from github.com/Fraternalilab/sciCSR)",
    ),
)


def _applies(requirement: str, extra: str | None) -> tuple[str, bool]:
    """(requirement without its marker, whether its marker holds here for ``extra``)."""
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging comes with pip / scanpy
        spec, _, marker = requirement.partition(";")
        return spec.strip(), "python_version" not in marker
    req = Requirement(requirement)
    ok = req.marker is None or req.marker.evaluate({"extra": extra or ""})
    spec = req.name + (f"[{','.join(sorted(req.extras))}]" if req.extras else "") + str(req.specifier)
    return spec, bool(ok)


def _requirements() -> tuple[list[str], dict[str, list[str]]]:
    """Core requirements and the requirements of every extra that apply to this Python (metadata, else fallback)."""
    try:
        reqs = md.requires("spaclone") or []
    except md.PackageNotFoundError:
        reqs = []
    core: list[str] = []
    extras: dict[str, list[str]] = {}
    for r in reqs:
        m = re.search(r"""extra\s*==\s*["']([^"']+)["']""", r)
        name = m.group(1) if m else None
        spec, ok = _applies(r, name)
        if spec.startswith("spaclone"):
            continue
        if name is None:
            if ok:
                core.append(spec)
        else:
            extras.setdefault(name, [])
            if ok:
                extras[name].append(spec)
    for k, v in EXTRAS_FALLBACK.items():  # extras that the installed metadata do not know (older installs)
        if k not in extras:
            extras[k] = [spec for spec, ok in (_applies(r, k) for r in v) if ok]
    if not core:
        core = [
            "anndata>=0.10",
            "scanpy>=1.10",
            "h5py>=3.8",
            "igraph>=0.11",
            "matplotlib>=3.7",
            "numpy>=1.24",
            "pandas>=2.0",
            "pyarrow>=12",
            "scikit-learn>=1.3",
            "scipy>=1.11",
        ]
    return core, extras


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0]


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _quote(spec: str) -> str:
    return f"'{spec}'" if re.search(r"[<>=!~]", spec) else spec


def find_tool(name: str, path: str | os.PathLike | None = None) -> Path | None:
    """Locate an external program.

    Parameters
    ----------
    name
        Executable name, e.g. ``"igblastn"``, ``"MakeDb.py"``, ``"igphyml"``, ``"Rscript"``.
    path
        Explicit location: the executable itself or a directory holding it (``bin/`` is also searched).

    Returns
    -------
    pathlib.Path or None
        The executable, or None when it is not found. The lookup order is ``path``, the environment variable of
        :data:`TOOL_ENV` (an executable or a directory), the directory of the running Python (programs installed by
        pip into this environment, also when it is not activated), then ``PATH``.
    """
    for cand in (path, os.environ.get(TOOL_ENV.get(name, ""), "") or None):
        if cand is None:
            continue
        p = Path(cand).expanduser()
        for q in (p, p / name, p / "bin" / name):
            if q.is_file() and (q.name == name or q == p) and os.access(q, os.X_OK):
                return q
    here = Path(sys.executable).parent / name  # scripts of this environment, also when it is not activated
    if here.is_file() and os.access(here, os.X_OK):
        return here
    w = shutil.which(name)
    return Path(w) if w else None


def external_env() -> dict[str, str]:
    """Environment for external programs: this one without the variables that tie a child to this Python.

    ``PYTHONPATH`` / ``PYTHONHOME`` would load this environment's packages into Change-O's own Python, and
    ``PYTHONNOUSERSITE`` would hide the packages of a Change-O installed with ``pip install --user``.
    """
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE"):
        env.pop(k, None)
    return env


def _tool_check(name: str, exe: Path) -> tuple[bool, str | None, str]:
    """(runs, version, error): run the program's version command once."""
    if name == "igphyml":
        return True, None, ""  # igphyml prints its whole help text; report the path only
    args = {"igblastn": ["-version"], "Rscript": ["--version"]}.get(name, ["--version"])
    try:
        r = subprocess.run(
            [str(exe), *args], check=False, capture_output=True, text=True, timeout=60, env=external_env()
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, None, f"{type(e).__name__}: {e}"[:300]
    text = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        last = [ln for ln in text.strip().splitlines() if ln.strip()]
        return False, None, (last[-1] if last else f"exit {r.returncode}")[:300]
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return True, (m.group(1) if m else None), ""


def _tool_version(name: str, exe: Path) -> str | None:
    return _tool_check(name, exe)[1]


def _r_packages(rscript: Path, packages: tuple[str, ...]) -> dict[str, str | None]:
    """Versions of R packages (None if not installed), from one Rscript call."""
    code = (
        "for (p in commandArgs(TRUE)) cat(p, tryCatch(as.character(packageVersion(p)), "
        "error = function(e) 'NA'), '\\n', sep = '\\t')"
    )
    try:
        r = subprocess.run(
            [str(rscript), "--vanilla", "-e", code, *packages],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=external_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return dict.fromkeys(packages)
    out: dict[str, str | None] = dict.fromkeys(packages)
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 2 and parts[0] in out:
            out[parts[0]] = None if parts[1] == "NA" else parts[1]
    return out


def _import_ok(module: str) -> tuple[bool, str]:
    """Import a module; (True, "") or (False, first line of the error)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 - any failure means the feature is unusable; report it
        msg = f"{type(e).__name__}: {e}".splitlines()[0]
        return False, msg[:300]
    return True, ""


@dataclass(frozen=True)
class InstallReport:
    """Result of :func:`check_install`.

    Attributes
    ----------
    table
        One row per feature: ``feature``, ``kind``, ``available`` (``"yes"``, ``"no"``, ``"partial"`` or
        ``"broken"``: installed but the import fails or the program does not run), ``version``, ``extra``,
        ``enable`` (the command that installs it), ``used_by`` and ``detail`` (path of an external tool, or the
        error).
    spaclone_version
        Version of spaclone.
    """

    table: pd.DataFrame
    spaclone_version: str

    @property
    def ok(self) -> bool:
        """Every reported feature is available."""
        return bool((self.table["available"] == "yes").all())

    @property
    def missing(self) -> pd.DataFrame:
        """The rows of features that are not fully available."""
        return self.table[self.table["available"] != "yes"]

    def __repr__(self) -> str:
        t = self.table
        n_ok = int((t["available"] == "yes").sum())
        lines = [f"spaclone {self.spaclone_version}: {n_ok} of {len(t)} optional features available", ""]
        w = max(len(s) for s in t["feature"]) + 2
        for r in t.itertuples(index=False):
            ver = f" {r.version}" if r.version else ""
            lines.append(f"  {r.feature:<{w}}{r.available:<8}{ver}")
            if r.available == "broken" and r.detail:
                lines.append(f"  {'':<{w}}  import error: {r.detail}")
        miss = t[t["available"] != "yes"]
        if len(miss):
            lines += ["", "To enable:"]
            for r in miss.itertuples(index=False):
                lines.append(f"  {r.feature:<{w}}{r.enable}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        cols = ["feature", "available", "version", "enable", "used_by"]
        return f"<p>spaclone {self.spaclone_version}</p>" + self.table[cols].to_html(index=False)


def _spaclone_version() -> str:
    return _version("spaclone") or "0+unknown"


def check_install(
    *,
    deep: bool = True,
    external: bool = True,
    r_packages: bool = True,
    tools: dict[str, str | os.PathLike] | None = None,
) -> InstallReport:
    """Which optional features work here, and the exact command that enables each missing one.

    Parameters
    ----------
    deep
        Import every optional Python package (a few seconds), so that a package that is installed but broken (for
        example sc-dandelion 0.5.x with matplotlib >= 3.9) is reported as ``"broken"`` rather than available. With
        ``False`` only the presence of the packages is checked.
    external
        Look for the external tools (IgBLAST, Change-O, IgPhyML, Rscript).
    r_packages
        Ask R (one ``Rscript`` call) for the versions of the Immcantation packages and sciCSR.
    tools
        Explicit locations of external tools, ``{name: path}`` with names as in :data:`TOOL_ENV` (``"igblastn"``,
        ``"MakeDb.py"``, ``"igphyml"``, ``"Rscript"``, ...; ``"Rscript (sciCSR)"`` for the R holding sciCSR); a
        directory also works.

    Returns
    -------
    InstallReport
        Printable report (one line per feature, then the commands for the missing ones) with ``.table``,
        ``.missing`` and ``.ok``.

    Examples
    --------
    >>> import spaclone
    >>> spaclone.check_install()  # doctest: +SKIP
    >>> spaclone.check_install().missing[["feature", "enable"]]  # doctest: +SKIP
    """
    _, extras = _requirements()
    tools = dict(tools or {})
    rows = []
    rscripts: dict[str, Path | None] = {}
    r_versions: dict[Path, dict[str, str | None]] = {}
    for f in FEATURES:
        if f.kind != "R" or not external:
            continue
        given = (
            tools.get("Rscript (sciCSR)") or os.environ.get("SPACLONE_SCICSR_RSCRIPT")
            if f.name == "R + sciCSR"
            else None
        )
        exe = find_tool("Rscript", given or tools.get("Rscript"))
        rscripts[f.name] = exe
        if exe is not None and r_packages and exe not in r_versions:
            names = tuple(dict.fromkeys(p for g in FEATURES if g.kind == "R" for p in g.requires))
            r_versions[exe] = _r_packages(exe, names)
    for f in FEATURES:
        row: dict[str, Any] = {"feature": f.name, "kind": f.kind, "extra": f.extra or "", "used_by": f.used_by}
        if f.kind == "python":
            specs = extras.get(f.extra or "", [])
            row["enable"] = "pip install " + " ".join(_quote(s) for s in specs) if specs else ""
            present = [importlib.util.find_spec(m) is not None for m in f.requires]
            dists = f.dists or f.requires
            ver = _version(dists[0])
            row["version"] = ver or ""
            detail = ""
            if all(present):
                status = "yes"
                if deep:
                    for m in f.requires:
                        good, msg = _import_ok(m)
                        if not good:
                            status, detail = "broken", f"{m}: {msg}"
                            break
            else:
                status = "partial" if any(present) else "no"
            row.update(available=status, detail=detail)
        elif f.kind == "tool":
            row["enable"] = f.install
            if not external:
                row.update(available="not checked", version="", detail="")
            else:
                found = {t: find_tool(t, tools.get(t)) for t in f.requires}
                n = sum(p is not None for p in found.values())
                status = "yes" if n == len(found) else ("partial" if n else "no")
                first_name, first = next(((t, p) for t, p in found.items() if p is not None), (None, None))
                ver, detail = None, str(first) if first else ""
                if first is not None:
                    runs, ver, err = _tool_check(first_name, first)
                    if not runs:
                        status, detail = "broken", f"{first}: {err}"
                row.update(available=status, version=ver or "", detail=detail)
        else:
            row["enable"] = f.install
            if not external or not r_packages:
                row.update(available="not checked", version="", detail="")
            elif rscripts.get(f.name) is None:
                row.update(
                    available="no",
                    version="",
                    detail="Rscript not found (set SPACLONE_RSCRIPT or put R on PATH)",
                )
            else:
                rscript = rscripts[f.name]
                vers = {p: r_versions.get(rscript, {}).get(p) for p in f.requires}
                n = sum(v is not None for v in vers.values())
                status = "yes" if n == len(vers) else ("partial" if n else "no")
                missing = [p for p, v in vers.items() if v is None]
                shown = ", ".join(f"{p} {v}" for p, v in vers.items() if v is not None)
                row.update(
                    available=status,
                    version=shown,
                    detail=f"{rscript}" + (f"; missing: {', '.join(missing)}" if missing else ""),
                )
        rows.append(row)
    cols = ["feature", "kind", "available", "version", "extra", "enable", "used_by", "detail"]
    return InstallReport(pd.DataFrame(rows, columns=cols), _spaclone_version())


def show_versions(*, external: bool = True, as_dict: bool = False) -> dict[str, Any] | None:
    """Print the versions of Python, spaclone, its dependencies, the optional packages and the external tools.

    Parameters
    ----------
    external
        Also report IgBLAST, Change-O, IgPhyML and R with its packages (runs each program once).
    as_dict
        Return the versions as a dictionary instead of printing them.

    Returns
    -------
    dict or None
        With ``as_dict=True``: ``{"python": ..., "platform": ..., "spaclone": ..., "dependencies": {...},
        "optional": {...}, "external": {...}}``; missing packages have the value None.

    Examples
    --------
    >>> import spaclone
    >>> spaclone.show_versions()  # doctest: +SKIP
    """
    core, extras = _requirements()
    deps = {_dist_name(r): _version(_dist_name(r)) for r in core}
    opt_names = sorted({_dist_name(r) for specs in extras.values() for r in specs}, key=str.lower)
    optional = {d: _version(d) for d in opt_names}
    ext: dict[str, Any] = {}
    if external:
        for name in ("igblastn", "MakeDb.py", "DefineClones.py", "CreateGermlines.py", "igphyml", "Rscript"):
            exe = find_tool(name)
            ext[name] = None if exe is None else {"version": _tool_version(name, exe), "path": str(exe)}
        if ext.get("Rscript"):
            names = tuple(dict.fromkeys(p for f in FEATURES if f.kind == "R" for p in f.requires))
            ext["R packages"] = _r_packages(Path(ext["Rscript"]["path"]), names)
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "spaclone": _spaclone_version(),
        "dependencies": deps,
        "optional": optional,
        "external": ext,
    }
    if as_dict:
        return info
    w = max([len(k) for k in [*deps, *optional, *ext]] + [12]) + 2
    out = [
        f"{'spaclone':<{w}}{info['spaclone']}",
        f"{'python':<{w}}{info['python']}",
        f"{'platform':<{w}}{info['platform']}",
        "",
        "dependencies",
    ]
    out += [f"  {k:<{w}}{v or 'not installed'}" for k, v in deps.items()]
    out += ["", "optional"]
    out += [f"  {k:<{w}}{v or '-'}" for k, v in optional.items()]
    if external:
        out += ["", "external tools"]
        for k, v in ext.items():
            if k == "R packages":
                out.append(f"  {'R packages':<{w}}" + ", ".join(f"{p} {x or '-'}" for p, x in v.items()))
            elif v is None:
                out.append(f"  {k:<{w}}-")
            else:
                out.append(f"  {k:<{w}}{v['version'] or ''}  ({v['path']})".replace("  ()", ""))
    print("\n".join(out))
    return None
