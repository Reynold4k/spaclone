"""Write ``docs/api.md`` from the public names of every spaclone module.

The public names of a module are its ``__all__``. A subpackage (``spaclone.io``, ``spaclone.bcr``, ...) re-exports
its submodules with ``from .x import *``; its public names are the names of those submodules' ``__all__`` that the
package really exports (the same object), grouped by submodule. Every public name appears exactly once; the one-line
summaries come from the docstrings through ``sphinx.ext.autosummary``.

Usage (from the repository root)::

    python docs/_scripts/make_api.py           # write docs/api.md
    python docs/_scripts/make_api.py --check   # exit 1 when docs/api.md is out of date
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
import types
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
API = DOCS / "api.md"

# Top level, in reading order (every other name of ``spaclone.__all__`` that is not a module is appended).
TOP = ["read", "run", "TestResult", "validate_air", "check_install", "show_versions"]

# (namespace, anchor, heading, description, [(title, [submodules])]); submodules not listed are appended.
PACKAGES = [
    (
        "io",
        "readers-io",
        "Readers and storage",
        "Every reader returns a {class}`~spaclone.io.SpatialAIR`: the receptor table with coordinates in µm, the "
        "matching spot-level GEX when there is one, and the reader's metadata (source unit and conversion factor). "
        "{func}`spaclone.read` (the same function as `spaclone.io.read`) reads a GEX source and a receptor table of "
        "any supported tool in one call; {func}`spaclone.io.sniff` says what a file is.",
        [
            ("Read a section", ["paired", "_container"]),
            ("What is this file?", ["sniff"]),
            ("Receptor tables of any tool", ["receptors", "airr"]),
            ("Spatial expression sources", ["gexio"]),
            ("Technology readers", ["airspace", "mixcr", "cellcalls"]),
            ("Stereo-seq molecule-level inputs", ["stereo"]),
            ("Cell-type labels and barcodes", ["annotations", "barcodes"]),
            ("Format registry", ["formats"]),
            ("scirpy, SpatialData and storage", ["interop", "store"]),
        ],
    ),
    (
        "pp",
        "receptor-preprocessing-pp",
        "Receptor preprocessing",
        "Receptor QC metrics and filtering, deduplication, barcode sinks, contamination flags, clones, loci (inferred "
        "cells), aggregation to spots, bins or cells, and coordinates.",
        [
            ("QC metrics", ["metrics"]),
            ("Filtering and validity", ["filtering", "validity"]),
            ("Deduplication and error variants", ["dedup", "variants"]),
            ("Barcode sinks", ["sinks"]),
            ("Contamination", ["contamination"]),
            ("Clones and loci", ["clones", "loci"]),
            ("Aggregation and cells", ["aggregate"]),
            ("Coordinates and sections", ["coords"]),
        ],
    ),
    (
        "gex",
        "transcriptome-gex",
        "Transcriptome",
        "Spot-level transcriptome: QC on and off the tissue, scanpy-based preprocessing with spatial defaults, cell "
        "types, Ig constant-gene RNA, regions with instances, region profiles, spatial domains, pseudobulk DE with "
        "region instances as replicates, and spatial statistics.",
        [
            ("QC, tissue and bins", ["qc"]),
            ("Normalisation, clustering and embedding", ["workflow"]),
            ("Cell types", ["annotate"]),
            ("Ig RNA", ["ig"]),
            ("Regions and region profiles", ["regions", "profile"]),
            ("Spatial domains", ["domains"]),
            ("Differential expression", ["de"]),
            ("Spatial statistics", ["spatial"]),
        ],
    ),
    (
        "qc",
        "location-trust-qc",
        "Location trust",
        "Is a receptor molecule or call where its cell was? {func}`spaclone.run` runs every applicable test and "
        "returns a {class}`~spaclone.qc.TrustReport` with two-level verdicts (molecule and clone) from pre-stated "
        "criteria; each test can also be called on its own.",
        [
            ("Trust report and criteria", ["report"]),
            ("Molecule level", ["same_spot", "umi_match", "offtissue"]),
            ("Cell and clone level", ["celltype", "colocalisation", "pairing"]),
            ("Read floor", ["read_floor"]),
        ],
    ),
    (
        "bcr",
        "b-cell-receptor-biology-bcr",
        "B-cell receptor biology",
        "Isotypes and class switching, somatic hypermutation, mutation spectrum and selection, lineage trees, and "
        "the two end-to-end workflows: CSR/SHM hotspots with their overlap regions, and lineage trees on tissue.",
        [
            ("Workflows", ["workflow", "lineage_workflow"]),
            ("Isotypes", ["isotype"]),
            ("Class switching", ["csr"]),
            ("CSR potential and SHM score (sciCSR)", ["csr_dynamics"]),
            ("Somatic hypermutation", ["shm"]),
            ("Mutation spectrum and selection", ["spectrum", "selection"]),
            ("Lineage trees", ["lineage"]),
        ],
    ),
    (
        "tl",
        "concentration-and-repertoire-tl",
        "Concentration, transitions and repertoire",
        "Share maps and map-level hotspots, the clonal scan, tissue structures and niches, sharing between regions, "
        "lineage tests, Markov transition models with transition path theory (sciCSR on tissue), and repertoire "
        "statistics.",
        [
            ("Share maps and hotspots", ["maps"]),
            ("Clonal scan", ["scan"]),
            ("Tissue structures", ["structures"]),
            ("Niches and sharing between regions", ["niche", "sharing"]),
            ("Lineage tests", ["lineage"]),
            ("Transition models and transition path theory", ["transitions"]),
            ("Repertoire", ["repertoire"]),
            ("Clonotype networks and public clonotypes", ["network", "public"]),
            ("Clone dispersion", ["dispersion"]),
            ("Invariant TCR chains", ["tcr"]),
        ],
    ),
    (
        "pl",
        "plotting-pl",
        "Plotting",
        "Every plot takes `ax=None` (when it draws one panel) and returns the Axes or the Figure; none calls "
        "`show()` or `savefig()`. Spatial maps draw y downwards, like an image of the section.",
        [
            ("Hotspot atlas and class-switch map", ["atlas", "csr_map"]),
            ("Lineage trees", ["lineage_gallery"]),
            ("Transitions on tissue", ["csr_flow"]),
            ("Threadfin clone states", ["clone_states"]),
            ("Location trust", ["report", "placement"]),
            ("Records and QC", ["records", "qc", "molecule"]),
            ("Maps, niches and clones", ["maps", "clones"]),
            ("B-cell receptor maps", ["bcr"]),
            ("Transcriptome", ["gex"]),
            ("Repertoire", ["repertoire"]),
        ],
    ),
    (
        "sim",
        "simulation-sim",
        "Simulation",
        "Simulated sections with known truth for every record, and the planted-truth benchmark.",
        [("Simulated sections", ["synthetic"]), ("Planted-truth benchmark", ["plant"])],
    ),
    (
        "datasets",
        "example-data-datasets",
        "Example data",
        "A synthetic section (no download) and the six AIR-SPACE lymph nodes of GEO GSE286452 (Mock, D3, D7, D10, "
        "D14, D21), downloaded once and verified by size and sha256.",
        [("", ["fetch"])],
    ),
]

NULLS = (
    "nulls",
    "null-designs-nulls",
    "Null designs",
    "A null design is an object that randomises a receptor table in a stated way: {class}`~spaclone.nulls.Permute` "
    "(random labelling) or {class}`~spaclone.nulls.Relocate` (random placement). {func}`~spaclone.nulls.placebo` "
    "re-runs any statistic on many randomised tables, and {func}`~spaclone.nulls.compare_designs` does it under "
    "several designs at once.",
)

EXT = [
    (
        "scicsr",
        "sciCSR",
        "the original sciCSR R package: definitions, export, runs and agreement (`spaclone[scicsr]`).",
    ),
    ("threadfin", "Threadfin", "clone-state profiles on spatial units, mapped back to tissue (`spaclone[threadfin]`)."),
    (
        "scirpy",
        "scirpy",
        "clonotypes, expansion, diversity, overlap and V(D)J usage on spatial units (`spaclone[scverse]`).",
    ),
    (
        "squidpy",
        "squidpy",
        "neighbourhood enrichment, co-occurrence and autocorrelation of receptor labels (`spaclone[scverse]`).",
    ),
    (
        "dandelion",
        "Dandelion",
        "reproducible Dandelion clone calls mapped back to the records (`spaclone[dandelion]`).",
    ),
    (
        "immcantation",
        "Immcantation",
        "IgBLAST, Change-O, SHazaM and dowser round trips (programs installed separately).",
    ),
]

# One line per namespace for the table at the top.
CONTENTS = {
    "io": "readers of every mainstream format (coordinates in µm), format detection, scirpy / SpatialData, storage",
    "pp": "receptor QC metrics, filtering, deduplication, clones, loci, barcode sinks, aggregation",
    "gex": "transcriptome QC, clustering, cell types, Ig RNA, regions, domains, DE, spatial statistics",
    "qc": "location-trust tests and the trust report: is a receptor molecule or call where its cell was?",
    "bcr": "isotypes, class switching, somatic hypermutation, lineage trees, the hotspot and lineage workflows",
    "tl": "share maps, hotspots, clonal scan, niches, transition models (TPT), repertoire statistics",
    "pl": "plots; each takes `ax=None` and returns the Axes or Figure",
    "sim": "simulated sections with known truth, planted-truth benchmarks",
    "datasets": "example data (download and cache)",
    "nulls": "null designs (random labelling, random placement) and the placebo engine",
}


def para(text: str) -> str:
    """Wrap a paragraph at the documentation's line length."""
    return textwrap.fill(text, width=118, break_long_words=False, break_on_hyphens=False)


INTRO = """\
# API reference

<!-- Generated by docs/_scripts/make_api.py from the __all__ of every public module; edit that script, not this file. -->

Every public function, class and constant, grouped by namespace and then by task, with the first line of its
docstring. Functions are called through their namespace, for example `spaclone.pp.filter_records` or
`spaclone.bcr.hotspot_workflow`. Data objects (receptor tables, AnnData) are positional arguments; everything else is
keyword-only. Coordinates and distances are in micrometres, and distance arguments end in `_um`. Every test returns a
{class}`~spaclone.TestResult` (or a subclass) and states its null model in the docstring.
"""


def public_names(modname: str) -> list[tuple[str, str]]:
    """``(name, defining submodule)`` of every public name of a module, in ``__all__`` order."""
    mod = importlib.import_module(modname)
    if not hasattr(mod, "__path__"):  # a plain module
        return [(n, modname) for n in mod.__all__]
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    subs = [
        n.module
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.level == 1 and any(a.name == "*" for a in n.names)
    ]
    out: dict[str, str] = {}
    for sub in subs:
        smod = importlib.import_module(f"{modname}.{sub}")
        for n in getattr(smod, "__all__", ()):
            if getattr(mod, n, None) is getattr(smod, n) and n not in out:
                out[n] = sub
    leaks = [
        n
        for n in dir(mod)
        if not n.startswith("_") and n not in out and not isinstance(getattr(mod, n), types.ModuleType)
    ]
    if leaks:
        raise SystemExit(f"{modname}: public attributes outside every __all__: {leaks}")
    return list(out.items())


def kind_order(mod: types.ModuleType, name: str) -> int:
    """Functions first, then classes (results, warnings), then constants."""
    obj = getattr(mod, name)
    if inspect.isclass(obj):
        return 1
    if callable(obj):
        return 0
    return 2


def autosummary(names: list[str]) -> list[str]:
    """An ``autosummary`` block (one stub page per name in ``generated/``)."""
    return [
        "```{eval-rst}",
        ".. autosummary::",
        "   :toctree: generated",
        "   :nosignatures:",
        "",
        *(f"   {n}" for n in names),
        "```",
        "",
    ]


def section(
    full: str, anchor: str, heading: str, text: str, groups: list[tuple[str, list[str]]]
) -> tuple[list[str], list[str]]:
    """The section of one namespace: heading, module directive and one autosummary block per group."""
    mod = importlib.import_module(full)
    names = public_names(full)
    by_sub: dict[str, list[str]] = {}
    for n, sub in names:
        by_sub.setdefault(sub, []).append(n)
    listed = [s for _, subs in groups for s in subs]
    unknown = [s for s in by_sub if s not in listed]
    groups = list(groups) + [(s.replace("_", " ").strip().capitalize(), [s]) for s in unknown]
    lines = [
        f"({anchor})=",
        f"## {heading} (`{full}`)",
        "",
        para(text),
        "",
        "```{eval-rst}",
        f".. module:: {full}",
        "```",
        "",
    ]
    done: list[str] = []
    for title, subs in groups:
        group = [n for s in subs for n in by_sub.get(s, [])]
        if not group:
            continue
        group.sort(key=lambda n: kind_order(mod, n))  # stable: __all__ order within a kind
        if title:
            lines += [f"### {title}", ""]
        lines += autosummary(group)
        done += group
    return lines, [f"{full}.{n}" for n in done]


def build() -> tuple[str, list[str], dict[str, int]]:
    """The text of ``api.md``, the documented names and the count per namespace."""
    import spaclone

    counts: dict[str, int] = {}
    documented: list[str] = []
    table = [
        "| Namespace | Contents |",
        "|---|---|",
        "| [`spaclone`](#api-top-level) | read, run, the result class, table validation, install checks |",
    ]
    body: list[str] = []

    top = [n for n in spaclone.__all__ if not isinstance(getattr(spaclone, n), (types.ModuleType, str))]
    top = [n for n in TOP if n in top] + [n for n in top if n not in TOP]
    body += [
        "(api-top-level)=",
        "## Top level (`spaclone`)",
        "",
        para(
            "`spaclone.read` reads any supported pair of files into a {class}`~spaclone.io.SpatialAIR`; `spaclone.run` "
            "runs QC, filtering and every applicable location-trust test in one call."
        ),
        "",
        "```{eval-rst}",
        ".. module:: spaclone",
        "```",
        "",
        *autosummary(top),
    ]
    documented += [f"spaclone.{n}" for n in top]
    counts["spaclone"] = len(top)

    for ns, anchor, heading, text, groups in PACKAGES:
        lines, names = section(f"spaclone.{ns}", anchor, heading, text, groups)
        body += lines
        documented += names
        counts[f"spaclone.{ns}"] = len(names)
        table.append(f"| [`spaclone.{ns}`](#{anchor}) | {CONTENTS[ns]} |")

    ns, anchor, heading, text = NULLS
    lines, names = section(f"spaclone.{ns}", anchor, heading, text, [("", [f"spaclone.{ns}"])])
    body += lines
    documented += names
    counts[f"spaclone.{ns}"] = len(names)
    table.append(f"| [`spaclone.{ns}`](#{anchor}) | {CONTENTS[ns]} |")

    ext = importlib.import_module("spaclone.ext")
    missing = sorted(set(ext.__all__) - {m for m, _, _ in EXT})
    if missing:
        raise SystemExit(f"spaclone.ext modules without an entry in EXT: {missing}")
    body += [
        "(ecosystem-bridges-ext)=",
        "## Ecosystem bridges (`spaclone.ext`)",
        "",
        para(
            "Each bridge imports its package only when called, so `import spaclone` never needs them; install the "
            "extra named below. See {doc}`ecosystem` for worked examples."
        ),
        "",
        "```{eval-rst}",
        ".. module:: spaclone.ext",
        "```",
        "",
    ]
    table.append(
        "| [`spaclone.ext`](#ecosystem-bridges-ext) | bridges to sciCSR, Threadfin, scirpy, squidpy, Dandelion and Immcantation |"
    )
    for m, title, what in EXT:
        full = f"spaclone.ext.{m}"
        mod = importlib.import_module(full)
        names = sorted(mod.__all__, key=lambda n: kind_order(mod, n))
        body += [
            f"### {title} (`{full}`)",
            "",
            f"Bridge to {what}",
            "",
            "```{eval-rst}",
            f".. module:: {full}",
            "```",
            "",
        ]
        body += autosummary(names)
        documented += [f"{full}.{n}" for n in names]
        counts[full] = len(names)

    text = INTRO + "\n" + "\n".join(table) + "\n\n" + "\n".join(body).rstrip("\n") + "\n"
    return text, documented, counts


def main(argv: list[str]) -> int:
    """Write (or ``--check``) ``docs/api.md``."""
    text, documented, counts = build()
    dup = sorted({n for n in documented if documented.count(n) > 1})
    if dup:
        raise SystemExit(f"names listed twice: {dup}")
    lower: dict[str, str] = {}
    for n in documented:  # generated/<name>.rst must not collide on case-insensitive file systems
        if n.lower() in lower:
            print(f"warning: stub files differ only in case: {lower[n.lower()]} / {n}", file=sys.stderr)
        lower[n.lower()] = n
    if "--check" in argv:
        current = API.read_text(encoding="utf-8") if API.exists() else ""
        if current != text:
            print(f"{API} is out of date: run python docs/_scripts/make_api.py", file=sys.stderr)
            return 1
        print(f"{API} is up to date ({len(documented)} public names)")
        return 0
    API.write_text(text, encoding="utf-8")
    for k, v in counts.items():
        print(f"{k:28s} {v:4d}")
    print(f"{'total':28s} {len(documented):4d}  -> {API}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
