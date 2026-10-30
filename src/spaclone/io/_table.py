"""Plumbing for delimited text tables: compression, byte-order mark, line endings, comment lines and the header.

Receptor tools write their tables in many small variants of CSV / TSV: gzip or plain, with or without a UTF-8 byte-order
mark, CRLF line endings, metadata lines before the header (``#FileFormat=GEMv0.1``, BD Rhapsody's ``##`` lines), a
header that itself starts with ``#`` (TRUST4 reports), quoted names, or a leading unnamed index column written by
pandas. :func:`peek_table` finds the header and the delimiter from the first lines only; :func:`read_text_table` then
reads the whole table as strings with the same settings.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import lzma
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__: list[str] = []

_MAGIC = {b"\x1f\x8b": "gzip", b"BZh": "bz2", b"\xfd7zXZ": "xz", b"PK\x03\x04": "zip"}
_HDF5 = b"\x89HDF\r\n\x1a\n"
_DELIMS = ("\t", ",", ";", "|")


@dataclass(frozen=True)
class TableHead:
    """What :func:`peek_table` learnt about a delimited text file.

    Attributes
    ----------
    path
        The file.
    columns
        Header names, stripped of quotes, surrounding blanks, a byte-order mark and a leading ``#``. A leading empty
        name (a pandas index column) is kept as ``""``.
    sep
        Delimiter (tab, comma, semicolon or ``|``).
    skiprows
        Lines before the header (metadata / comment lines).
    comments
        Those lines (at most 50), without line endings.
    header_hash
        The header line started with ``#`` (TRUST4 reports, VDJtools).
    index_column
        The first column has no name: an index written by pandas ``to_csv``.
    compression
        ``"gzip"``, ``"bz2"``, ``"xz"``, ``"zip"`` or ``None`` (from the file's magic bytes, not its suffix).
    bom
        The text starts with a UTF-8 byte-order mark.
    crlf
        Lines end with CRLF.
    sample
        Up to ``n`` data rows as strings (empty strings for empty fields).
    """

    path: str
    columns: tuple[str, ...]
    sep: str
    skiprows: int
    comments: tuple[str, ...]
    header_hash: bool
    index_column: bool
    compression: str | None
    bom: bool
    crlf: bool
    sample: pd.DataFrame = field(repr=False)

    def evidence(self) -> list[str]:
        """Human-readable notes on the layout (used in :func:`spaclone.io.sniff`)."""
        out = [f"{'tab' if self.sep == chr(9) else repr(self.sep)}-delimited, {len(self.columns)} columns"]
        if self.compression:
            out.append(f"{self.compression}-compressed (magic bytes)")
        if self.bom:
            out.append("UTF-8 byte-order mark")
        if self.crlf:
            out.append("CRLF line endings")
        if self.skiprows:
            out.append(f"{self.skiprows} metadata line(s) before the header, e.g. {self.comments[0][:60]!r}")
        if self.header_hash:
            out.append("header line starts with '#'")
        if self.index_column:
            out.append("leading unnamed index column (ignored)")
        return out


def binary_kind(path: str | Path) -> str | None:
    """``"HDF5"``, ``"Parquet"`` or ``"zarr"`` when a path is a binary container, else ``None``.

    HDF5 (AnnData ``.h5ad`` / ``.h5mu``, 10x ``.h5``, GEF, Dandelion ``.h5ddl``) and Parquet from their magic bytes
    (HDF5 also after a user block of 512, 1024 or 2048 bytes), zarr from the store markers of a directory.
    """
    p = Path(path)
    if p.is_dir():
        return "zarr" if any((p / m).exists() for m in (".zgroup", ".zattrs", "zarr.json")) else None
    try:
        with open(p, "rb") as fh:
            start = fh.read(2056)
    except OSError:
        return None
    if any(start[o : o + 8] == _HDF5 for o in (0, 512, 1024, 2048)):
        return "HDF5"
    return "Parquet" if start.startswith(b"PAR1") else None


def compression_of(path: str | Path) -> str | None:
    """Compression of a file from its magic bytes (``None`` for plain text)."""
    with open(path, "rb") as fh:
        start = fh.read(6)
    for magic, name in _MAGIC.items():
        if start.startswith(magic):
            return name
    return None


def open_text(path: str | Path, compression: str | None = None) -> io.TextIOBase:
    """Open a (possibly compressed) text file for reading, keeping CRLF line endings visible (``newline=""``)."""
    comp = compression_of(path) if compression is None else compression
    if comp == "gzip":
        raw: Any = gzip.open(path, "rb")
    elif comp == "bz2":
        raw = bz2.open(path, "rb")
    elif comp == "xz":
        raw = lzma.open(path, "rb")
    elif comp == "zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"{path}: a zip archive must hold exactly one table, found {len(names)}")
        raw = zf.open(names[0])
    else:
        raw = open(path, "rb")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")


def _split(line: str, sep: str) -> list[str]:
    return next(csv.reader([line], delimiter=sep, quotechar='"'))


def _clean_name(name: str) -> str:
    return name.replace("﻿", "").strip().strip('"').strip("'").strip()


def _best_sep(header: str, following: list[str]) -> str:
    """Delimiter: the candidate that splits the header into the most fields with the same count on the next lines."""
    best, best_n = "\t", 1
    for sep in _DELIMS:
        if sep not in header:
            continue
        try:
            n = len(_split(header, sep))
        except csv.Error:
            continue
        if n <= 1:
            continue
        rows = [r for r in following[:20] if r.strip()]
        same = sum(len(_split(r, sep)) in (n, n + 1) for r in rows) if rows else 1
        if (rows and same < max(1, len(rows) // 2)) or n <= best_n:
            continue
        best, best_n = sep, n
    return best


def _is_metadata(line: str, nxt: list[str]) -> bool:
    """A leading line that is not the header: ``##...``, ``#key=value``, or a ``#`` line unlike the next row."""
    s = line.lstrip("﻿")
    if not s.strip():
        return True
    if not s.startswith("#"):
        return False
    if s.startswith("##"):
        return True
    body = s[1:]
    if "=" in body and not any(d in body for d in ("\t", ",")):
        return True
    data = [r for r in nxt if r.strip() and not r.startswith("#")]
    if not data:
        return False
    sep = _best_sep(body, data)
    n_head = len(_split(body, sep)) if sep in body else 1
    n_data = len(_split(data[0], sep)) if sep in data[0] else 1
    return n_head < 2 or n_data not in (n_head, n_head + 1)


def peek_table(path: str | Path, *, n: int = 500) -> TableHead:
    """Find the header, the delimiter and the layout of a delimited text table from its first lines.

    Parameters
    ----------
    path
        CSV / TSV, optionally gzip / bz2 / xz / zip compressed (detected from magic bytes).
    n
        Data rows to keep in ``sample``.

    Returns
    -------
    TableHead

    Raises
    ------
    ValueError
        If the file is a directory, a binary container (HDF5, Parquet, zarr), empty, or holds no delimited header.
    """
    path = Path(path)
    kind = binary_kind(path)
    if kind is not None:
        what = "a zarr store" if kind == "zarr" else f"a{'n' if kind == 'HDF5' else ''} {kind} file"
        raise ValueError(f"{path.name} is {what}, not a delimited text table")
    if path.is_dir():
        raise ValueError(f"{path} is a directory, not a delimited text table")
    comp = compression_of(path)
    with open_text(path, comp) as fh:
        lines: list[str] = []
        for _ in range(n + 60):
            line = fh.readline()
            if not line:
                break
            lines.append(line)
    if not lines:
        raise ValueError(f"{path} is empty")
    crlf = lines[0].endswith("\r\n")
    bom = lines[0].startswith("﻿")
    stripped = [ln.rstrip("\r\n") for ln in lines]
    stripped[0] = stripped[0].lstrip("﻿")
    k = 0
    while k < len(stripped) - 1 and k < 50 and _is_metadata(stripped[k], stripped[k + 1 : k + 25]):
        k += 1
    header = stripped[k]
    header_hash = header.startswith("#")
    body = header[1:] if header_hash else header
    data = stripped[k + 1 :]
    sep = _best_sep(body, data)
    cols = [_clean_name(c) for c in _split(body, sep)] if body else []
    if not cols or (len(cols) == 1 and not cols[0]):
        raise ValueError(f"{path}: no header line found")
    index_column = cols[0] in ("", "Unnamed: 0")
    if index_column:
        cols[0] = ""
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):  # duplicated names get pandas-style suffixes (name.1, name.2)
        if c and c in seen:
            seen[c] += 1
            cols[i] = f"{c}.{seen[c]}"
        else:
            seen.setdefault(c, 0)
    rows = [_split(r, sep) for r in data[:n] if r.strip()]
    width = len(cols)
    rows = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in rows]
    sample = pd.DataFrame(rows, columns=[c if c else f"_col{i}" for i, c in enumerate(cols)], dtype=object)
    return TableHead(
        path=str(path),
        columns=tuple(cols),
        sep=sep,
        skiprows=k,
        comments=tuple(stripped[: min(k, 50)]),
        header_hash=header_hash,
        index_column=index_column,
        compression=comp,
        bom=bom,
        crlf=crlf,
        sample=sample,
    )


def read_text_table(head: TableHead, *, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the whole table described by ``head`` as strings (empty fields missing, ``"NA"`` kept as text).

    Parameters
    ----------
    head
        From :func:`peek_table`.
    usecols
        Columns to keep (header names as in ``head.columns``); default all named columns.

    Returns
    -------
    pandas.DataFrame
        Columns named as in ``head.columns`` (the unnamed index column dropped), all values ``str`` or missing.
    """
    names = [c if c else f"_col{i}" for i, c in enumerate(head.columns)]
    want = [c for c in names if not c.startswith("_col")] if usecols is None else list(usecols)
    with open_text(head.path, head.compression) as fh:
        df = pd.read_csv(
            fh,
            sep=head.sep,
            skiprows=head.skiprows + 1,
            header=None,
            names=names,
            usecols=want,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            quotechar='"',
            engine="c",
            index_col=False,
        )
    return df
