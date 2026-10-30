"""spaclone: end-to-end analysis of spatial immune-receptor data.

Readers (``io``), receptor preprocessing and QC (``pp``), the transcriptome (``gex``), location-trust tests (``qc``),
B-cell receptor biology (``bcr``), repertoire and spatial concentration statistics (``tl``), plotting (``pl``),
simulation (``sim``), and explicit null designs (``nulls``). ``spaclone.run`` runs QC, filtering and every applicable
trust test in one call.
"""

from importlib.metadata import PackageNotFoundError, version

from . import bcr, datasets, ext, gex, io, nulls, pl, pp, qc, sim, tl
from ._run import run
from ._schema import validate_air
from ._utils import TestResult
from ._versions import check_install, show_versions
from .io import read

try:
    __version__ = version("spaclone")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "TestResult",
    "bcr",
    "check_install",
    "datasets",
    "ext",
    "gex",
    "io",
    "nulls",
    "pl",
    "pp",
    "qc",
    "read",
    "run",
    "show_versions",
    "sim",
    "tl",
    "validate_air",
    "__version__",
]
