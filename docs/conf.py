"""Sphinx configuration for spaclone."""

import importlib
import logging
import pkgutil
import pprint
import sys
from collections import namedtuple
from importlib.metadata import version as _version

from sphinx.pycode import ModuleAnalyzer

project = "spaclone"
author = "Chen Satoshi"
copyright = "2026, Chen Satoshi"
release = _version("spaclone")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
autosummary_generate = True
autodoc_member_order = "bysource"
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_custom_sections = [("Null model", "notes_style")]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = "off"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
html_theme = "sphinx_book_theme"
html_title = "spaclone"
html_theme_options = {"repository_url": "https://github.com/Reynold4k/spaclone", "use_repository_button": True}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

_TUPLEGETTER = type(namedtuple("_Probe", "field").field)


class _InheritedDataclassAnnotations(logging.Filter):
    """Drop one known false positive of sphinx-autodoc-typehints, and nothing else.

    The result classes are dataclasses that inherit from ``spaclone.TestResult``. The ``__init__`` that
    ``dataclasses`` writes for a subclass carries the inherited annotations (``dict[str, Any]``) as strings and
    resolves them in the subclass's module, which does not import ``Any``, so the extension cannot evaluate them.
    Users never call these constructors and the class signatures are unaffected. Any other unresolvable annotation
    still fails the strict build.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        if getattr(record, "subtype", None) != "forward_reference" or len(args) < 3:
            return True
        modname, _, attr = str(args[0]).rpartition(".")
        try:
            cls = getattr(importlib.import_module(modname), attr)
        except (ImportError, AttributeError, ValueError):
            return True
        test_result = importlib.import_module("spaclone").TestResult
        inherited = isinstance(cls, type) and issubclass(cls, test_result)
        return not (inherited and str(args[2]) == "name 'Any' is not defined")


def _skip_namedtuple_fields(app, what, name, obj, skip, options):
    """Leave out NamedTuple field accessors.

    Their generated "Alias for field number N" text would duplicate the entries of the class docstring's Attributes
    section.
    """
    if isinstance(obj, _TUPLEGETTER):
        return True
    return None


def _defining_module(full_name: str, obj) -> str:
    """Return the submodule that defines a constant re-exported by a namespace.

    ``spaclone.qc.CRITERIA`` lives in ``spaclone.qc.report``; the ``#:`` comment there is its documentation.
    """
    modname, _, attr = full_name.rpartition(".")
    pkg = sys.modules.get(modname)
    if pkg is None or not hasattr(pkg, "__path__"):
        return modname
    for info in pkgutil.iter_modules(pkg.__path__):
        sub = importlib.import_module(f"{modname}.{info.name}")
        if attr in getattr(sub, "__all__", ()) and getattr(sub, attr, None) is obj:
            return sub.__name__
    return modname


def _document_constant(app, what, name, obj, options, lines):
    """Document a constant by its ``#:`` comment and show its value, pretty-printed.

    autodoc looks for the comment only in the namespace ``__init__`` and would otherwise print the docstring of the
    value's type (``dict``, ``tuple``, ...).
    """
    if what != "data":
        return
    attr = name.rpartition(".")[2]
    try:
        comment = ModuleAnalyzer.for_module(_defining_module(name, obj)).find_attr_docs().get(("", attr))
    except Exception:  # noqa: BLE001 - no source available: fall back to no description
        comment = None
    lines[:] = [*comment] if comment else []
    text = pprint.pformat(obj, width=100, sort_dicts=False)
    if "\n" not in text and len(text) <= 90 and "``" not in text:
        lines += ["", f"Value: ``{text}``", ""]
    else:
        lines += ["", "Value:", "", ".. code-block:: python", "", *("   " + line for line in text.splitlines()), ""]


def setup(app):
    """Connect the hooks above."""
    app.connect("autodoc-skip-member", _skip_namedtuple_fields)
    app.connect("autodoc-process-docstring", _document_constant, priority=900)
    logging.getLogger("sphinx.sphinx_autodoc_typehints._warnings").addFilter(_InheritedDataclassAnnotations())

# stub files whose names differ from another public name only in case (clash on case-insensitive file systems)
autosummary_filename_map = {
    "spaclone.gex.IG_GENES": "spaclone.gex.IG_GENES-constant",
    "spaclone.gex.PROFILE_MARKERS": "spaclone.gex.PROFILE_MARKERS-constant",
    "spaclone.tl.INVARIANT_TCR": "spaclone.tl.INVARIANT_TCR-constant",
}
