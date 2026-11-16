"""Small helpers of the interactive figures: the call written into a file, argument checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def describe(obj: Any) -> str:
    """Short description of an argument for the call recorded in a figure (data objects by type and size)."""
    if obj is None or isinstance(obj, bool | int | float | str):
        return repr(obj)
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {len(obj):,} x {obj.shape[1]}>"
    if isinstance(obj, pd.Series):
        return f"<Series {len(obj):,}>"
    if isinstance(obj, np.ndarray):
        return f"<array {'x'.join(str(s) for s in obj.shape)}>"
    if hasattr(obj, "n_obs") and hasattr(obj, "n_vars"):
        return f"<AnnData {obj.n_obs:,} x {obj.n_vars:,}>"
    if isinstance(obj, dict):
        return (
            "{"
            + ", ".join(f"{k!r}: {describe(v)}" for k, v in list(obj.items())[:8])
            + (", ..." if len(obj) > 8 else "")
            + "}"
        )
    if isinstance(obj, list | tuple):
        inner = ", ".join(describe(v) for v in list(obj)[:8]) + (", ..." if len(obj) > 8 else "")
        return f"[{inner}]" if isinstance(obj, list) else f"({inner})"
    name = type(obj).__name__
    lib = None
    meta = getattr(obj, "meta", None)
    if isinstance(meta, dict):
        lib = meta.get("library_id")
    return f"<{name}{' ' + str(lib) if lib else ''}>"


def call_text(func: str, *args: Any, **kwargs: Any) -> str:
    """``func(<args>, key=value, ...)`` with data objects described briefly and None-valued keywords left out."""
    parts = [describe(a) for a in args]
    parts += [f"{k}={describe(v)}" for k, v in kwargs.items() if v is not None]
    return f"{func}({', '.join(parts)})"


def check_style(style: str | None) -> str:
    """``"dark"`` or ``"light"`` (None means light, as for the static maps)."""
    if style is None:
        return "light"
    if style not in ("dark", "light"):
        raise ValueError(f"style must be 'dark' or 'light', got {style!r}")
    return style
