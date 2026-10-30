"""Stereo-seq (SAW) inputs for the molecule-level tests: whole-chip bin1 GEX, the tissue mask and the GEX saturation curve.

Stereo-seq positions are DNB (bin1) indices with a pitch of 0.5 µm (the ``resolution`` attribute of a ``.gef`` file, in
nm). The raw readers return DNB indices *with the conversion factor attached* (``meta["um_per_dnb"]``); everything
derived here (:class:`DistanceGrid`) is in micrometres. The readers load only what the molecule tests need: per-DNB
molecule (MID) totals for the whole chip, on and off tissue so that empty glass can serve as a negative control, and
the DNBs of a few named genes.

References
----------
Chen et al. 2022, doi:10.1016/j.cell.2022.04.003 (Stereo-seq; SAW ``.gef`` layout: ``geneExp/bin1/expression`` sorted
by gene, with per-gene ``offset`` / ``count`` in ``geneExp/bin1/gene``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np

from spaclone import _schema as S

__all__ = [
    "UM_PER_DNB",
    "DistanceGrid",
    "StereoGEX",
    "p_det_from_saturation",
    "read_gef_bin1",
    "read_stereo_gef",
    "read_tissue_mask",
    "signed_distance_map",
]

logger = logging.getLogger("spaclone")

#: Stereo-seq DNB pitch in micrometres (bin1).
UM_PER_DNB = S.DNB_UM
_SHIFT = np.int64(32)
_LOW = np.int64((1 << 32) - 1)


class StereoGEX(NamedTuple):
    """Whole-chip bin1 GEX of a SAW ``.gef``, reduced to what the molecule-level tests need.

    Attributes
    ----------
    x, y
        DNB indices (int64) of every DNB that holds at least one GEX molecule (MID), on or off tissue.
    total
        MIDs on each of these DNBs, all genes (int64).
    genes
        ``{gene: (x, y, count)}`` for each requested gene: the DNBs holding at least one MID of it and the MIDs there.
        Requested genes that are absent from the file are present with empty arrays and listed in
        ``meta["genes_missing"]``.
    meta
        ``path``, ``sn`` (chip), ``resolution_nm``, ``um_per_dnb`` (the factor to micrometres), ``source_unit``,
        ``min_x`` ... ``max_y`` (chip extent in DNBs), ``n_entries`` (gene × DNB rows read), ``n_zero_entries``
        (rows with 0 MIDs, dropped), ``n_occupied``, ``genes_missing``.
    """

    x: np.ndarray
    y: np.ndarray
    total: np.ndarray
    genes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    meta: dict[str, Any]


def _attr(v: Any) -> Any:
    a = np.asarray(v)
    if a.dtype.kind in "SO":
        a = a.astype(str)
    return a.reshape(-1)[0].item() if a.size == 1 else a.tolist()


def _reduce_by_key(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique keys and the summed counts per key (int64)."""
    if len(keys) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    sums = np.add.reduceat(counts[order].astype(np.int64, copy=False), starts)
    return sk[starts], sums


def read_gef_bin1(path: str | Path, genes: Sequence[str] | None = None, *, chunk_rows: int = 1 << 24) -> StereoGEX:
    """Read the whole-chip bin1 GEX of a SAW ``raw.gef``: per-DNB MID totals and the DNBs of a few genes.

    Off-tissue DNBs are included (use the ``raw.gef``, not ``tissue.gef``), so that empty glass can be used as a
    negative control. The expression table is read in chunks of ``chunk_rows`` rows and reduced to one key and one
    count per row (about 9 bytes per gene × DNB row while the totals are summed); nothing else of the file is loaded.

    Parameters
    ----------
    path
        SAW ``*.raw.gef`` (HDF5 with groups ``geneExp/bin1/expression`` and ``geneExp/bin1/gene``).
    genes
        Gene names whose DNBs are returned (e.g. the immunoglobulin constant genes). ``None`` returns totals only.
        A name listed more than once in the file (several gene ids) is summed over its ids.
    chunk_rows
        Rows of the expression table read at a time.

    Returns
    -------
    StereoGEX
        DNB indices in DNB units; multiply by ``meta["um_per_dnb"]`` (0.5) for micrometres.

    Notes
    -----
    Rows with a count of 0 (not written by SAW in the files we have seen) are dropped and counted in
    ``meta["n_zero_entries"]``: a DNB or gene with no molecule is not occupied.
    """
    path = Path(path)
    want = list(dict.fromkeys(genes)) if genes is not None else []
    with h5py.File(path, "r") as h:
        expr = h["geneExp/bin1/expression"]
        gtab = h["geneExp/bin1/gene"][:]
        attrs = {k: _attr(v) for k, v in expr.attrs.items()}
        sn = _attr(h.attrs["sn"]) if "sn" in h.attrs else None
        n = int(expr.shape[0])
        keys = np.empty(n, np.int64)
        cnt = np.empty(n, expr.dtype["count"])
        for s in range(0, n, chunk_rows):
            blk = expr[s : s + chunk_rows]
            keys[s : s + len(blk)] = (blk["x"].astype(np.int64) << _SHIFT) | blk["y"].astype(np.int64)
            cnt[s : s + len(blk)] = blk["count"]
            del blk
        n_zero = int((cnt == 0).sum())
        if n_zero:
            ok = cnt > 0
            keys, cnt = keys[ok], cnt[ok]
        uk, tot = _reduce_by_key(keys, cnt)
        del keys, cnt
        names = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in gtab["geneName"]])
        offs = gtab["offset"].astype(np.int64)
        cnts = gtab["count"].astype(np.int64)
        gene_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        missing = []
        for g in want:
            idx = np.flatnonzero(names == g)
            if len(idx) == 0:
                missing.append(g)
                gene_data[g] = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
                continue
            parts = [expr[int(offs[i]) : int(offs[i] + cnts[i])] for i in idx]
            blk = np.concatenate(parts) if len(parts) > 1 else parts[0]
            gk = (blk["x"].astype(np.int64) << _SHIFT) | blk["y"].astype(np.int64)
            gc = blk["count"].astype(np.int64)
            keep = gc > 0
            gk, gc = _reduce_by_key(gk[keep], gc[keep])
            gene_data[g] = (gk >> _SHIFT, gk & _LOW, gc)
    if missing:
        logger.warning("read_gef_bin1: gene(s) %s not in %s; treated as never detected", missing, path.name)
    res_nm = float(attrs.get("resolution", 500))
    meta = {
        "path": str(path),
        "sn": sn,
        "technology": "Stereo-seq",
        "resolution_nm": res_nm,
        "um_per_dnb": res_nm / 1000.0,
        "source_unit": "DNB",
        "min_x": int(attrs.get("minX", 0)),
        "max_x": int(attrs["maxX"]) if "maxX" in attrs else int((uk >> _SHIFT).max(initial=0)),
        "min_y": int(attrs.get("minY", 0)),
        "max_y": int(attrs["maxY"]) if "maxY" in attrs else int((uk & _LOW).max(initial=0)),
        "n_entries": n,
        "n_zero_entries": n_zero,
        "n_occupied": len(uk),
        "genes_missing": missing,
    }
    return StereoGEX(x=uk >> _SHIFT, y=uk & _LOW, total=tot, genes=gene_data, meta=meta)


#: Alias of :func:`read_gef_bin1`: same arguments, same result.
read_stereo_gef = read_gef_bin1


def read_tissue_mask(path: str | Path, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a tissue mask image (e.g. SAW ``*_tissue_cut.tif``) as a boolean array indexed ``[y, x]`` in DNBs.

    Parameters
    ----------
    path
        Image readable by ``tifffile`` (optional dependency ``spaclone[images]``); non-zero pixels are tissue.
    shape
        Expected ``(n_y, n_x)`` of the chip, e.g. ``(max_y + 1, max_x + 1)`` from :func:`read_gef_bin1`. A mask stored
        transposed is transposed; any other shape raises ``ValueError``.

    Returns
    -------
    numpy.ndarray
        Boolean, C-contiguous; ``mask[y, x]`` is True on tissue.
    """
    try:
        import tifffile
    except ImportError as e:
        raise ImportError("reading a tissue mask image needs tifffile: pip install 'spaclone[images]'") from e

    m = np.asarray(tifffile.imread(str(path)))
    if m.ndim != 2:
        raise ValueError(f"tissue mask must be a 2-D image, got shape {m.shape}")
    m = m > 0
    if shape is not None:
        shape = (int(shape[0]), int(shape[1]))
        if m.shape == shape:
            pass
        elif m.shape == shape[::-1]:
            m = m.T
        else:
            raise ValueError(f"tissue mask {m.shape} matches neither {shape} nor its transpose")
    return np.ascontiguousarray(m)


@dataclass(frozen=True, eq=False)
class DistanceGrid:
    """Signed distance to the tissue edge on a regular grid, in micrometres (negative inside tissue).

    Attributes
    ----------
    values
        ``(n_y, n_x)`` float32 array; ``values[i, j]`` is the signed distance of the grid cell whose lower-left corner
        is at ``(origin_um[0] + j * pixel_um, origin_um[1] + i * pixel_um)``.
    pixel_um
        Side of a grid cell in micrometres.
    origin_um
        Position of the lower-left corner of cell ``(0, 0)``.
    """

    values: np.ndarray
    pixel_um: float
    origin_um: tuple[float, float] = field(default=(0.0, 0.0))

    def _index(self, x_um: Any, y_um: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_um, dtype=float)
        y = np.asarray(y_um, dtype=float)
        ny, nx = self.values.shape
        ix = np.clip(np.floor((x - self.origin_um[0]) / self.pixel_um), 0, nx - 1).astype(np.intp)
        iy = np.clip(np.floor((y - self.origin_um[1]) / self.pixel_um), 0, ny - 1).astype(np.intp)
        return iy, ix

    def __call__(self, x_um: Any, y_um: Any) -> np.ndarray:
        """Signed distance (µm) at the given positions; positions off the grid take the nearest edge cell."""
        iy, ix = self._index(x_um, y_um)
        return self.values[iy, ix].astype(float)

    def in_tissue(self, x_um: Any, y_um: Any) -> np.ndarray:
        """Whether the positions lie on tissue (signed distance <= 0)."""
        return self(x_um, y_um) <= 0

    def area_um2(self, bins_um: Sequence[float]) -> np.ndarray:
        """Area (µm²) of the grid whose signed distance falls in each half-open bin ``[bins[k], bins[k + 1])``."""
        e = np.asarray(bins_um, dtype=float)
        idx = np.searchsorted(e, self.values.ravel(), side="right") - 1
        ok = (idx >= 0) & (idx < len(e) - 1)
        return np.bincount(idx[ok], minlength=len(e) - 1).astype(float) * self.pixel_um**2

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_y, n_x)`` of the grid."""
        return tuple(self.values.shape)  # type: ignore[return-value]

    @property
    def area_inside_um2(self) -> float:
        """Tissue area of the grid (µm²)."""
        return float((self.values <= 0).sum()) * self.pixel_um**2

    @property
    def area_outside_um2(self) -> float:
        """Off-tissue area of the grid (µm²)."""
        return float((self.values > 0).sum()) * self.pixel_um**2

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Centres (µm) of all grid cells, flattened in ``values.ravel()`` order."""
        ny, nx = self.values.shape
        xs = self.origin_um[0] + (np.arange(nx) + 0.5) * self.pixel_um
        ys = self.origin_um[1] + (np.arange(ny) + 0.5) * self.pixel_um
        gx, gy = np.meshgrid(xs, ys)
        return gx.ravel(), gy.ravel()


def signed_distance_map(
    mask: np.ndarray,
    *,
    pixel_um: float = UM_PER_DNB,
    downsample: int = 5,
    origin_um: tuple[float, float] = (0.0, 0.0),
) -> DistanceGrid:
    """Signed distance to the edge of a tissue mask, in micrometres (negative inside tissue).

    The mask is first reduced to blocks of ``downsample`` × ``downsample`` pixels (a block is tissue when at least half
    of its pixels are; remainder rows and columns are dropped), then the Euclidean distance transform gives, for tissue
    blocks, minus the distance to the nearest background block and, for background blocks, the distance to the nearest
    tissue block (both in block units × ``downsample`` × ``pixel_um``). With the defaults this is the 2.5 µm (bin5) grid
    of the archived Stereo-XCR-seq analyses, value for value.

    Parameters
    ----------
    mask
        Boolean ``mask[y, x]`` (True = tissue), e.g. from :func:`read_tissue_mask`.
    pixel_um
        Side of a mask pixel in µm (0.5 for a Stereo-seq DNB mask).
    downsample
        Block size in pixels (1 = no reduction; the transform then needs about 16 bytes per pixel).
    origin_um
        Position of the lower-left corner of mask pixel ``(0, 0)``.

    Returns
    -------
    DistanceGrid
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D")
    d = int(downsample)
    if d < 1:
        raise ValueError("downsample must be >= 1")
    if d > 1:
        ny, nx = (m.shape[0] // d) * d, (m.shape[1] // d) * d
        blocks = m[:ny, :nx].reshape(ny // d, d, nx // d, d).sum(axis=(1, 3), dtype=np.int32)
        m = 2 * blocks >= d * d
    if m.all() or not m.any():
        raise ValueError("the mask is all tissue or all background; there is no edge to measure distances to")
    step = d * float(pixel_um)
    signed = np.where(m, -ndimage.distance_transform_edt(m), ndimage.distance_transform_edt(~m)) * step
    return DistanceGrid(
        values=signed.astype(np.float32), pixel_um=step, origin_um=(float(origin_um[0]), float(origin_um[1]))
    )


def p_det_from_saturation(path: str | Path) -> dict[str, Any]:
    """Probability that a GEX library molecule is sequenced, from the SAW saturation curve (``REPORT/saturation.json``).

    SAW subsamples a fraction ``f`` of the GEX reads and counts the unique captures ``U(f)``. A Michaelis–Menten curve
    ``U = U_max f / (K + f)`` saturates slowly, so its asymptote over-estimates the library size and
    ``p_det = U(1) / U_max`` is conservative (low). A Gamma–Poisson capture model ``U = M [1 − (1 + f μ / k)^(−k)]``
    is fitted as well; its ``p_det_nb = U(1) / M`` is reported only when identifiable (``M`` below 50 × ``U(1)``).

    Parameters
    ----------
    path
        SAW ``saturation.json`` (key ``"All"`` with lists ``sample`` = fractions and ``bar_umi`` = unique captures).

    Returns
    -------
    dict
        ``p_det`` (Michaelis–Menten, the conservative default for :func:`spaclone.qc.same_spot`), ``se`` (delta-method
        standard error of the fit; it ignores model uncertainty, compare ``p_det_nb``), ``model``,
        ``unique_full_depth``, ``sequencing_saturation``, ``mm_umax``, ``mm_k``, ``rmse_rel_mm``, ``nb_m``, ``nb_mu``,
        ``nb_k``, ``rmse_rel_nb``, ``p_det_nb``, ``fraction``, ``unique``.

    Notes
    -----
    ``p_det`` here is the chance that a molecule present in the library is seen at all; the chance that a genuine
    receptor molecule's own transcript is seen *given that its DNB holds GEX* is higher, so bounds on located fractions
    computed with this ``p_det`` are conservative. The fits use the curve in file order, as the archived analysis
    (workflow A4) did, so the numbers match it exactly.

    References
    ----------
    Chen et al. 2022, doi:10.1016/j.cell.2022.04.003 (SAW saturation report).
    """
    from scipy.optimize import curve_fit

    with open(path) as fh:
        j = json.load(fh)["All"]
    f = np.asarray(j["sample"], dtype=float)
    u = np.asarray(j["bar_umi"], dtype=float)
    top = int(np.argmax(f))
    u1 = float(u[top])

    def mm(f, um, k):
        return um * f / (k + f)

    def nb(f, m, mu, k):
        return m * (1 - (1 + f * mu / k) ** (-k))

    (um, k_mm), cov = curve_fit(mm, f, u, p0=(u1 * 1.5, 0.5), maxfev=200000)
    p_mm = u1 / um
    se_mm = float(u1 / um**2 * np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else float("nan")
    sat = j.get("bar_y1")
    out: dict[str, Any] = {
        "p_det": float(p_mm),
        "se": se_mm,
        "model": "michaelis-menten",
        "unique_full_depth": u1,
        "sequencing_saturation": float(np.asarray(sat, dtype=float)[top]) if sat is not None else None,
        "mm_umax": float(um),
        "mm_k": float(k_mm),
        "rmse_rel_mm": float(np.sqrt(np.mean((mm(f, um, k_mm) - u) ** 2)) / u1),
    }
    try:
        (m_nb, mu_nb, k_nb), _ = curve_fit(
            nb, f, u, p0=(u1 * 1.3, 3.0, 1.0), bounds=([u1, 1e-3, 1e-3], [u1 * 100, 1e3, 1e3]), maxfev=200000
        )
        out.update(
            {
                "nb_m": float(m_nb),
                "nb_mu": float(mu_nb),
                "nb_k": float(k_nb),
                "rmse_rel_nb": float(np.sqrt(np.mean((nb(f, m_nb, mu_nb, k_nb) - u) ** 2)) / u1),
                "p_det_nb": float(u1 / m_nb) if m_nb < 50 * u1 else None,
            }
        )
    except RuntimeError:  # pragma: no cover - the fit did not converge
        out.update({"nb_m": None, "nb_mu": None, "nb_k": None, "rmse_rel_nb": None, "p_det_nb": None})
    out["fraction"] = f.tolist()
    out["unique"] = u.tolist()
    return out
