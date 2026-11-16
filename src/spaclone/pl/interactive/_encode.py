"""Binary blocks of an interactive figure: typed arrays, quantised values, varints, compression and base64.

Every block is a dict ``{"t": dtype, "n": length, "d": base64}`` with optional ``"z": 1`` (zlib-compressed, ``"r"`` =
raw byte length), ``"q": [offset, step]`` (quantised: value = offset + code * step; code ``"nan"`` = missing) and
``"delta": 1`` for varints that store running differences. The figure engine decodes them in the browser
(``engine.js``: ``decodeBlob``); :func:`decode` here is the same decoder in Python, used by the tests to check that
every encoding round-trips.
"""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np

#: Little-endian numpy dtypes of the block types.
DTYPES: dict[str, str] = {"u1": "u1", "i1": "i1", "u2": "<u2", "i2": "<i2", "u4": "<u4", "i4": "<i4", "f4": "<f4",
                          "f8": "<f8"}  # fmt: skip
#: Blocks at least this large (bytes) are compressed when compression saves at least 10%.
COMPRESS_MIN = 4096
#: zlib level: 6 is zlib's default (good ratio, fast); the output is the same bytes for the same input and zlib.
LEVEL = 6


def _pack(raw: bytes, meta: dict[str, Any], compress: bool | None) -> dict[str, Any]:
    out = dict(meta)
    data = raw
    if compress is None:
        compress = len(raw) >= COMPRESS_MIN
    if compress and raw:
        c = zlib.compress(raw, LEVEL)
        if len(c) <= 0.9 * len(raw):
            data = c
            out["z"] = 1
            out["r"] = len(raw)
    out["d"] = base64.b64encode(data).decode("ascii")
    return out


def code_dtype(n_codes: int) -> str:
    """Smallest unsigned block type holding codes ``0 .. n_codes`` (the last one is kept for missing values)."""
    if n_codes < 255:
        return "u1"
    if n_codes < 65535:
        return "u2"
    return "u4"


def encode_array(values: Any, dtype: str, *, compress: bool | None = None) -> dict[str, Any]:
    """A block holding ``values`` as ``dtype`` (``u1``, ``u2``, ``u4``, ``i1``, ``i2``, ``i4``, ``f4``, ``f8``).

    Integers must fit the type (checked); floats are rounded to ``f4`` / ``f8``. Non-finite floats stay NaN (infinities
    become NaN so that no figure carries an infinite coordinate).
    """
    if dtype not in DTYPES:
        raise ValueError(f"unknown block type {dtype!r}")
    a = np.asarray(values)
    if dtype in ("f4", "f8"):
        a = np.asarray(a, dtype=float)
        a = np.where(np.isfinite(a), a, np.nan)
    else:
        a = np.asarray(a)
        if a.size:
            info = np.iinfo(np.dtype(DTYPES[dtype]))
            if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                raise ValueError("integer block with non-finite values")
            lo, hi = (a.min(), a.max())
            if lo < info.min or hi > info.max:
                raise ValueError(f"values {lo}..{hi} do not fit {dtype}")
    arr = np.ascontiguousarray(a.ravel(), dtype=np.dtype(DTYPES[dtype]))
    return _pack(arr.tobytes(), {"t": dtype, "n": int(arr.size)}, compress)


def encode_quantised(values: Any, *, bits: int = 16, compress: bool | None = None) -> dict[str, Any]:
    """Floats stored as ``bits``-bit codes between their minimum and maximum (the top code marks missing values).

    The decoded value is ``offset + code * step`` with ``step = (max - min) / (2**bits - 2)``: the error is at most
    ``step / 2``. Constant columns decode exactly.
    """
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    a = np.asarray(values, dtype=float).ravel()
    fin = np.isfinite(a)
    top = (1 << bits) - 1
    lo = float(a[fin].min()) if fin.any() else 0.0
    hi = float(a[fin].max()) if fin.any() else 0.0
    step = (hi - lo) / (top - 1) if hi > lo else 1.0
    codes = np.full(a.size, top, dtype=np.int64)
    codes[fin] = np.clip(np.rint((a[fin] - lo) / step), 0, top - 1).astype(np.int64)
    blk = encode_array(codes, "u1" if bits == 8 else "u2", compress=compress)
    blk["q"] = [lo, step]
    blk["nan"] = top
    return blk


def varint_bytes(values: Any) -> bytes:
    """Unsigned LEB128 encoding of non-negative integers (7 bits per byte, high bit = more bytes follow)."""
    v = np.asarray(values, dtype=np.uint64).ravel()
    if v.size == 0:
        return b""
    nb = np.ones(v.size, dtype=np.int64)
    t = v >> np.uint64(7)
    while t.any():
        nb += t > 0
        t = t >> np.uint64(7)
    pos = np.concatenate([[0], np.cumsum(nb)[:-1]])
    buf = np.zeros(int(nb.sum()), dtype=np.uint8)
    t = v.copy()
    for k in range(int(nb.max())):
        m = nb > k
        byte = (t[m] & np.uint64(0x7F)).astype(np.uint8)
        more = (nb[m] > k + 1).astype(np.uint8) << 7
        buf[pos[m] + k] = byte | more
        t[m] = t[m] >> np.uint64(7)
    return buf.tobytes()


def encode_varints(values: Any, *, delta: bool = False, compress: bool | None = None) -> dict[str, Any]:
    """A block of unsigned varints (with ``delta=True``: the differences of a non-decreasing sequence)."""
    v = np.asarray(values, dtype=np.int64).ravel()
    if delta:
        d = np.diff(v, prepend=0)
        if v.size and (d < 0).any():
            raise ValueError("delta varints need a non-decreasing sequence")
        v = d
    if v.size and v.min() < 0:
        raise ValueError("varints must be non-negative")
    blk = _pack(varint_bytes(v), {"t": "v", "n": int(v.size)}, compress)
    if delta:
        blk["delta"] = 1
    return blk


def encode_bytes(raw: bytes, *, compress: bool | None = None) -> dict[str, Any]:
    """A block of raw bytes (``"t": "raw"``)."""
    return _pack(bytes(raw), {"t": "raw", "n": len(raw)}, compress)


def block_bytes(blk: dict[str, Any]) -> int:
    """Characters the block takes in the figure (its base64 text plus a little JSON)."""
    return len(blk.get("d", "")) + 40


# ------------------------------------------------------------------------------------------------ the decoder


def _raw(blk: dict[str, Any]) -> bytes:
    data = base64.b64decode(blk.get("d", ""))
    if blk.get("z"):
        data = zlib.decompress(data)
        if "r" in blk and len(data) != blk["r"]:
            raise ValueError("decompressed length differs from the stored one")
    return data


def _read_varints(data: bytes, n: int, delta: bool) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    p = 0
    acc = 0
    for i in range(n):
        v = 0
        sh = 0
        while True:
            b = data[p]
            p += 1
            v |= (b & 0x7F) << sh
            sh += 7
            if not b & 0x80:
                break
        if delta:
            acc += v
            out[i] = acc
        else:
            out[i] = v
    return out


def decode(blk: dict[str, Any]) -> np.ndarray:
    """Decode a block (the same rules as the browser engine): numbers as numpy, quantised blocks as float."""
    data = _raw(blk)
    t = blk["t"]
    if t == "v":
        return _read_varints(data, int(blk["n"]), bool(blk.get("delta")))
    if t == "raw":
        return np.frombuffer(data, dtype=np.uint8).copy()
    arr = np.frombuffer(data, dtype=np.dtype(DTYPES[t]), count=int(blk["n"])).copy()
    if "q" in blk:
        lo, step = blk["q"]
        nan = blk.get("nan", -1)
        out = lo + arr.astype(float) * step
        out[arr == nan] = np.nan
        return out.astype(np.float32)
    return arr
