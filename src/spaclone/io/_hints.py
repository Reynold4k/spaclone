"""Message helpers of the readers: examples as plain strings and suggestions for a missing column."""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from typing import Any

__all__: list[str] = []


def examples(values: Iterable[Any], n: int = 3) -> str:
    """Up to ``n`` values as a list of plain strings, ``['a', 'b']`` (never ``np.str_('a')``)."""
    out: list[str] = []
    for v in values:
        if len(out) >= n:
            break
        out.append(str(v))
    return repr(out)


def column_hint(missing: str, columns: Sequence[Any], *, keywords: Sequence[str] = (), max_all: int = 60) -> str:
    """Every column when there are few, else the columns closest to ``missing`` (by name or by a keyword)."""
    cols = [str(c) for c in columns if str(c) and not str(c).startswith("Unnamed:")]
    if len(cols) <= max_all:
        return "columns: " + ", ".join(repr(c) for c in cols)
    close = difflib.get_close_matches(missing, cols, n=5, cutoff=0.5)
    close += [c for c in cols if c not in close and any(k in c.lower() for k in keywords)]
    if not close:
        return f"{len(cols)} columns, none close to {missing!r}; first ones: " + ", ".join(repr(c) for c in cols[:20])
    return f"closest of the {len(cols)} columns: " + ", ".join(repr(c) for c in close[:8])


def candidates(columns: Sequence[Any], keywords: Sequence[str]) -> list[str]:
    """Columns whose lower-case name contains one of ``keywords``."""
    return [str(c) for c in columns if any(k in str(c).lower() for k in keywords)]
