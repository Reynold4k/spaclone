"""Spatial gene-expression sources: identification, loading and the micrometre unit of their coordinates.

:func:`read_gex` loads a spot / bin / bead / cell expression matrix from any supported source and returns it with its
coordinates in source units and the unit decision. The unit is set automatically **only from technology facts**:

* coordinates written by a spaclone reader (``uns["spaclone"]["um_per_unit"]``) are already µm;
* Space Ranger: Visium HD ``microns_per_pixel`` (or ``bin_size_um / spot_diameter_fullres``) from the scale factors;
  Visium SD from the array geometry (spots two array columns apart are 100 µm apart, measured on
  ``array_row`` / ``array_col`` and the full-resolution pixel coordinates; ``spot_diameter_fullres`` is only a
  "theoretical 65 µm spot" and differs between Space Ranger versions);
* Stereo-seq DNB, ``resolution`` nm per unit (500 nm = 0.5 µm), when the source is certainly Stereo-seq: a GEF / GEM
  file, an AnnData written by stereopy / SAW (``uns["bin_type"]`` with ``uns["resolution"]``), or obs names that spell
  their own integer coordinates in a Stereo-seq convention (``"x-y"``, ``"x_y"``, ``"sample:x_y"`` on a SAW bin
  lattice, or stereopy's packed integer ``x << 32 | y``);
* Xenium cell centroids (``x_centroid`` / ``y_centroid``) are µm (10x Xenium output specification).

Anything else raises an error that names ``unit_um=``. Coordinates come from ``obsm["spatial"]`` (the scverse
convention), else from a single unambiguous alternative (``obsm["X_spatial"]``, obs ``x`` / ``y``, ``x_centroid`` /
``y_centroid``, ``imagecol`` / ``imagerow``, ``sdimx`` / ``sdimy``, ``array_col`` / ``array_row``); several that
disagree raise an error that names ``spatial_key=``.

References
----------
10x Genomics, Space Ranger spatial outputs (tissue_positions.csv, scalefactors_json.json; Visium spot 55 µm, Visium HD
binned outputs); 10x Genomics, Xenium output file specification (cells.parquet, cell_feature_matrix.h5); Chen et al.
2022, doi:10.1016/j.cell.2022.04.003 (Stereo-seq, DNB pitch 500 nm; SAW GEF / GEM formats); Fang et al. 2024,
doi:10.1101/2024.11.07.622402 (stereopy).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "GEX_FORMATS",
    "SAW_BIN_SIZES",
    "detect_unit",
    "gex_format",
    "read_10x_h5",
    "read_gef",
    "read_gem",
    "read_gex",
    "read_slideseq",
    "read_spaceranger",
    "read_xenium",
    "source_coordinates",
]

logger = logging.getLogger("spaclone")

#: Bin sizes (in DNB) of SAW / stereopy square bins; obs names that spell coordinates count as Stereo-seq ids only
#: on one of these lattices.
SAW_BIN_SIZES: tuple[int, ...] = (1, 5, 10, 20, 25, 50, 100, 150, 200)
_VISIUM_SPOT_UM = 55.0
_DNB_NM = 500.0
_HD_ID = re.compile(r"^s_(\d{3})um_(\d{5})_(\d{5})(-\d+)?$")
#: Supported GEX sources: ``gex_format=`` value → description.
GEX_FORMATS: dict[str, str] = {
    "anndata": "AnnData object or .h5ad (scanpy / squidpy, stereopy / SAW, Seurat-converted, Curio, Giotto, AIR-SPACE)",
    "spaceranger": "Space Ranger Visium SD folder (filtered_feature_bc_matrix.h5 + spatial/tissue_positions[_list].csv)",
    "visium_hd": "Space Ranger Visium HD bin folder (binned_outputs/square_XXXum: matrix + tissue_positions.parquet)",
    "xenium": "Xenium output folder (cell_feature_matrix.h5 + cells.parquet / cells.csv.gz)",
    "stereo_gef": "Stereo-seq SAW square-bin .gef (geneExp/bin1) — needs bin_size",
    "stereo_cellbin_gef": "Stereo-seq SAW .cellbin.gef (cellBin/cell, cellExp, gene)",
    "stereo_gem": "Stereo-seq GEM / GEM.gz (geneID x y MIDCount [CellID]) — needs bin_size unless cell bins",
    "slideseq": "Slide-seq bead tables (BeadLocationsForR.csv + MappedDGEForR.csv) — needs unit_um",
    "spatialdata": "SpatialData .zarr (one table)",
    "seurat_rds": "Seurat .rds (convert to .h5ad first; the error gives the snippet)",
}
_SEURAT_SNIPPET = (
    "library(Seurat); obj <- readRDS('object.rds')\n"
    "SeuratDisk::SaveH5Seurat(obj, 'object.h5Seurat'); SeuratDisk::Convert('object.h5Seurat', dest = 'h5ad')\n"
    "# or: sceasy::convertFormat(obj, from = 'seurat', to = 'anndata', outFile = 'object.h5ad')"
)
_STEREOPY_SNIPPET = (
    "import stereo as st\n"
    "data = st.io.read_stereo_h5ad('file.h5ad')\n"
    "st.io.stereo_to_anndata(data, flavor='scanpy', output='file.anndata.h5ad')"
)


# ------------------------------------------------------------------------------------------------- identification


def _is_rds(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            head = fh.read(4)
        if head[:2] == b"\x1f\x8b":
            with gzip.open(p, "rb") as fh:
                head = fh.read(4)
        return head[:2] in (b"X\n", b"A\n", b"B\n") or p.suffix.lower() == ".rds"
    except OSError:
        return False


def _find(folder: Path, *names: str) -> Path | None:
    for n in names:
        hits = sorted(folder.glob(n))
        if hits:
            return hits[0]
    return None


def gex_format(source: Any) -> tuple[str, list[str]]:
    """Identify a GEX source (see :data:`GEX_FORMATS`) and the evidence.

    Parameters
    ----------
    source
        AnnData, path of a file or folder.

    Returns
    -------
    tuple
        ``(format name, evidence)``.

    Raises
    ------
    ValueError
        If the source matches no supported layout, or a folder holds several candidates (e.g. a Visium HD ``outs``
        folder with several bin sizes).
    """
    if isinstance(source, ad.AnnData):
        return "anndata", ["AnnData object"]
    if hasattr(source, "tables") and hasattr(source, "points"):
        return "spatialdata", ["SpatialData object"]
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"GEX source {p} not found")
    if p.is_dir():
        if (p / "zarr.json").exists() or (p / ".zgroup").exists() or (p / "tables").is_dir():
            return "spatialdata", [f"{p.name}: zarr store with tables/"]
        if (p / "binned_outputs").is_dir():
            bins = sorted(d.name for d in (p / "binned_outputs").iterdir() if d.is_dir())
            raise ValueError(
                f"{p} is a Visium HD outs folder with bins {bins}; pass one bin folder as gex, e.g. "
                f"{p / 'binned_outputs' / (bins[1] if len(bins) > 1 else bins[0])}"
            )
        mat = _find(p, "filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5")
        if mat is not None and (p / "spatial" / "tissue_positions.parquet").exists():
            return "visium_hd", [f"{mat.name} + spatial/tissue_positions.parquet (Visium HD bins)"]
        if mat is not None and _find(p / "spatial", "tissue_positions.csv", "tissue_positions_list.csv"):
            return "spaceranger", [f"{mat.name} + spatial/tissue_positions*.csv (Visium SD)"]
        if _find(p, "*cell_feature_matrix.h5") and _find(p, "*cells.parquet", "*cells.csv.gz", "*cells.csv"):
            return "xenium", ["cell_feature_matrix.h5 + cells.parquet / cells.csv.gz (Xenium)"]
        if _find(p, "*BeadLocationsForR.csv*", "*bead_locations*", "*matched_bead_locations*"):
            return "slideseq", ["Slide-seq bead location table"]
        if (p / "outs").is_dir():
            return gex_format(p / "outs")
        raise ValueError(
            f"{p}: no supported GEX layout (Space Ranger SD/HD, Xenium, Slide-seq, SpatialData). Pass the file itself "
            "(.h5ad, .gef, .gem) or gex_format=."
        )
    name = p.name.lower()
    if name.endswith((".gem", ".gem.gz")):
        return "stereo_gem", [f"{p.name}: GEM file name"]
    if name.endswith(".gef"):
        with h5py.File(p, "r") as h:
            if "cellBin" in h:
                return "stereo_cellbin_gef", ["GEF with cellBin/"]
            if "geneExp" in h:
                return "stereo_gef", ["GEF with geneExp/bin1"]
        raise ValueError(f"{p}: a .gef without geneExp/ or cellBin/")
    if name.endswith((".rds", ".rda")) or _is_rds(p):
        return "seurat_rds", ["R serialised object"]
    if name.endswith((".h5ad", ".h5")):
        with h5py.File(p, "r") as h:
            keys = set(h.keys())
            if {"exp_matrix", "position"} <= keys or ({"cells", "genes", "exp_matrix"} <= keys):
                raise ValueError(
                    f"{p} is a stereopy StereoExpData file, not an AnnData. Convert it once:\n{_STEREOPY_SNIPPET}"
                )
            if {"obs", "var"} <= keys or "X" in keys:
                return "anndata", ["h5ad (AnnData)"]
            if "matrix" in keys and "spatial" in {q.name for q in p.parent.iterdir() if q.is_dir()}:
                return gex_format(p.parent)
            if "matrix" in keys:
                raise ValueError(
                    f"{p} is a 10x feature-barcode matrix without positions; pass the Space Ranger folder that holds "
                    "it and spatial/ (or the Xenium output folder)"
                )
    if name.endswith((".csv", ".csv.gz", ".txt", ".txt.gz")) and "bead" in name:
        return "slideseq", ["bead location table"]
    raise ValueError(f"{p}: unrecognised GEX file; supported: {', '.join(GEX_FORMATS)} (pass gex_format=)")


# ------------------------------------------------------------------------------------------------- 10x matrices


def _plain(v: Any) -> Any:
    """An HDF5 attribute as str / number / list of str."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [x.decode() if isinstance(x, bytes) else (x.item() if hasattr(x, "item") else x) for x in v.ravel()]
    return v.item() if hasattr(v, "item") else v


def read_10x_h5(path: str | Path, *, gex_only: bool = True) -> ad.AnnData:
    """Read a 10x feature-barcode matrix (``*_feature_bc_matrix.h5`` / ``cell_feature_matrix.h5``) with h5py.

    Parameters
    ----------
    path
        HDF5 file with group ``matrix`` (``barcodes``, ``data``, ``indices``, ``indptr``, ``shape``,
        ``features/{id,name,feature_type}``).
    gex_only
        Keep only features of type ``"Gene Expression"`` (Xenium control probes and codewords are dropped).

    Returns
    -------
    anndata.AnnData
        Raw integer counts in ``X`` (CSR), barcodes as obs names, gene symbols as var names (made unique) with
        ``var["gene_ids"]`` and ``var["feature_types"]``; ``uns["10x_h5"]`` holds the file attributes (chemistry,
        software version, library ids).
    """
    with h5py.File(path, "r") as h:
        m = h["matrix"]
        shape = tuple(int(v) for v in m["shape"][:])
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
        bcs = [b.decode() for b in m["barcodes"][:]]
        f = m["features"]
        ids = [b.decode() for b in f["id"][:]]
        names = [b.decode() for b in f["name"][:]]
        ftype = [b.decode() for b in f["feature_type"][:]] if "feature_type" in f else ["Gene Expression"] * len(ids)
        attrs = {k: _plain(v) for k, v in h.attrs.items()}
    var = pd.DataFrame({"gene_ids": ids, "feature_types": ftype}, index=pd.Index(names, dtype=str))
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=pd.Index(bcs, dtype=str)), var=var)
    if gex_only:
        a = a[:, a.var["feature_types"].to_numpy() == "Gene Expression"].copy()
    a.var_names_make_unique()
    a.uns["10x_h5"] = attrs
    return a


def _positions(folder: Path) -> tuple[pd.DataFrame, str]:
    sp = folder / "spatial"
    pq_ = sp / "tissue_positions.parquet"
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    if pq_.exists():
        return pd.read_parquet(pq_).set_index("barcode"), "spatial/tissue_positions.parquet"
    csv1 = sp / "tissue_positions.csv"
    if csv1.exists():
        return pd.read_csv(csv1).set_index("barcode"), "spatial/tissue_positions.csv (Space Ranger >= 2.0, header)"
    csv0 = sp / "tissue_positions_list.csv"
    if csv0.exists():
        return (
            pd.read_csv(csv0, header=None, names=cols).set_index("barcode"),
            "spatial/tissue_positions_list.csv (Space Ranger 1.x, headerless; 10x column order)",
        )
    raise FileNotFoundError(f"{sp}: no tissue_positions.parquet / tissue_positions.csv / tissue_positions_list.csv")


def read_spaceranger(path: str | Path, *, matrix: str = "filtered") -> ad.AnnData:
    """Read a Space Ranger Visium SD output folder or a Visium HD bin folder.

    Parameters
    ----------
    path
        Folder with ``filtered_feature_bc_matrix.h5`` (or ``raw_...``) and ``spatial/`` (an SD ``outs`` folder, or a
        Visium HD ``binned_outputs/square_XXXum`` folder).
    matrix
        ``"filtered"`` (spots / bins under tissue) or ``"raw"`` (all of them, off-tissue included).

    Returns
    -------
    anndata.AnnData
        Raw counts; ``obs`` with ``in_tissue``, ``array_row``, ``array_col``; ``obsm["spatial"]`` = full-resolution
        pixels ``(pxl_col_in_fullres, pxl_row_in_fullres)`` as in scanpy / squidpy; ``uns["spatial"][library]
        ["scalefactors"]`` from ``scalefactors_json.json`` (the unit: see :func:`detect_unit`).
    """
    p = Path(path)
    if (p / "outs").is_dir() and not (p / "spatial").is_dir():
        p = p / "outs"
    h5 = p / f"{matrix}_feature_bc_matrix.h5"
    if not h5.exists():
        raise FileNotFoundError(f"{h5} not found (matrix='filtered' or 'raw')")
    a = read_10x_h5(h5)
    pos, src = _positions(p)
    pos.index = pos.index.astype(str)
    miss = ~a.obs_names.isin(pos.index)
    if miss.any():
        raise ValueError(f"{int(miss.sum())} matrix barcodes are not in {src} (e.g. {str(a.obs_names[miss][0])!r})")
    pos = pos.loc[a.obs_names]
    for c in ("in_tissue", "array_row", "array_col"):
        if c in pos:
            a.obs[c] = pos[c].to_numpy()
    a.obs["in_tissue"] = a.obs["in_tissue"].astype(bool)
    a.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    sf_path = p / "spatial" / "scalefactors_json.json"
    sf = json.loads(sf_path.read_text()) if sf_path.exists() else {}
    libs = a.uns.get("10x_h5", {}).get("library_ids") or [p.name]
    lib = str(libs[0]) if isinstance(libs, list) else str(libs)
    a.uns["spatial"] = {lib: {"scalefactors": sf, "metadata": {"source": str(p), "positions": src}}}
    return a


def read_xenium(path: str | Path) -> ad.AnnData:
    """Read a Xenium output folder: ``cell_feature_matrix.h5`` and ``cells.parquet`` (or ``cells.csv.gz``).

    Parameters
    ----------
    path
        Xenium output folder (file names may carry a dataset prefix, as on the 10x download pages).

    Returns
    -------
    anndata.AnnData
        Gene-expression features only; ``obs`` = the cell table; ``obsm["spatial"]`` = ``x_centroid`` /
        ``y_centroid`` in µm (10x: "x_centroid / y_centroid: cell centroid in µm"); ``uns["spaclone_unit"]`` records
        the unit fact.
    """
    p = Path(path)
    h5 = _find(p, "cell_feature_matrix.h5", "*cell_feature_matrix.h5")
    cells = _find(p, "cells.parquet", "*cells.parquet", "cells.csv.gz", "*cells.csv.gz", "*cells.csv")
    if h5 is None or cells is None:
        raise FileNotFoundError(f"{p}: a Xenium folder needs cell_feature_matrix.h5 and cells.parquet / cells.csv.gz")
    a = read_10x_h5(h5)
    tab = pd.read_parquet(cells) if cells.suffix == ".parquet" else pd.read_csv(cells)
    tab["cell_id"] = tab["cell_id"].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    tab = tab.set_index("cell_id")
    if not a.obs_names.isin(tab.index).all():
        raise ValueError(f"{p}: matrix barcodes and cell_id of {cells.name} differ")
    tab = tab.loc[a.obs_names]
    a.obs = tab.copy()
    a.obsm["spatial"] = tab[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    a.uns["spaclone_unit"] = {"um_per_unit": 1.0, "rule": "Xenium cells: x_centroid / y_centroid in µm (10x spec)"}
    return a


# ------------------------------------------------------------------------------------------------- Stereo-seq


def _attr(h: Any, key: str, default: Any = None) -> Any:
    if key not in h.attrs:
        return default
    v = np.asarray(h.attrs[key])
    v = v.reshape(-1)[0] if v.size else default
    return v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)


def _packed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, np.int64) << 32) | np.asarray(y, np.int64)).astype(str).astype(object)


def _names(gtab: np.ndarray, field: str) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in gtab[field]]


def read_gef(path: str | Path, *, bin_size: int | None = None, chunk_rows: int = 1 << 24) -> ad.AnnData:
    """Read a Stereo-seq SAW ``.gef``: square bins of ``bin_size`` DNB from ``geneExp/bin1``, or the cell bins.

    Parameters
    ----------
    path
        ``*.raw.gef`` / ``*.tissue.gef`` (square bins) or ``*.cellbin.gef``.
    bin_size
        Square-bin side in DNB (e.g. 20 = 10 µm); required for square-bin files, ignored for cell bins.
    chunk_rows
        Expression rows read at a time.

    Returns
    -------
    anndata.AnnData
        Raw counts; obs names = stereopy's packed integer ``x << 32 | y`` of the bin's lower corner (square bins:
        ``x // bin_size * bin_size``) or of the cell centre; ``obs["x"]`` / ``obs["y"]`` and ``obsm["spatial"]`` in
        DNB; ``uns["bin_type"]``, ``uns["bin_size"]``, ``uns["resolution"]`` (nm) and ``uns["sn"]`` as stereopy
        writes them; gene symbols as var names (unique) with ``var["gene_ids"]``.
    """
    p = Path(path)
    with h5py.File(p, "r") as h:
        sn = _attr(h, "sn", None)
        if "cellBin" in h:
            cb = h["cellBin"]
            cell = cb["cell"][:]
            exp = cb["cellExp"][:]
            gtab = cb["gene"][:]
            res = float(_attr(h, "resolution", _DNB_NM))
            n_genes = len(gtab)
            indptr = np.concatenate([[0], np.cumsum(cell["geneCount"].astype(np.int64))])
            X = sparse.csr_matrix(
                (exp["count"].astype(np.float32), exp["geneID"].astype(np.int64), indptr), shape=(len(cell), n_genes)
            )
            x, y = cell["x"].astype(np.int64), cell["y"].astype(np.int64)
            obs = pd.DataFrame({"x": x, "y": y, "cell_id_gef": cell["id"].astype(np.int64)})
            for c in ("dnbCount", "area"):
                if c in cell.dtype.names:
                    obs[c] = cell[c].astype(np.int64)
            names = _packed(x, y)
            if len(set(names)) != len(names):
                names = obs["cell_id_gef"].astype(str).to_numpy(dtype=object)
            bin_type, bsize = "cell_bins", 1
        else:
            if bin_size is None:
                raise ValueError(
                    f"{p.name} is a square-bin GEF: pass bin_size= (bin side in DNB, e.g. 20 for bin20 = 10 µm, "
                    "50 for bin50); the AIRR spot ids of StereoIR are bin20"
                )
            bsize = int(bin_size)
            g = h["geneExp/bin1"]
            gtab = g["gene"][:]
            expr = g["expression"]
            res = float(_attr(expr, "resolution", _attr(h, "resolution", _DNB_NM)))
            cnt_field = "count" if "count" in gtab.dtype.names else "geneCount"
            gene_of_row = np.repeat(np.arange(len(gtab), dtype=np.int32), gtab[cnt_field].astype(np.int64))
            keys, genes, vals = [], [], []
            for s in range(0, int(expr.shape[0]), chunk_rows):
                blk = expr[s : s + chunk_rows]
                k = ((blk["x"].astype(np.int64) // bsize * bsize) << 32) | (blk["y"].astype(np.int64) // bsize * bsize)
                keys.append(k)
                genes.append(gene_of_row[s : s + len(blk)])
                vals.append(blk["count"].astype(np.float32))
            k = np.concatenate(keys) if keys else np.zeros(0, np.int64)
            uk, code = np.unique(k, return_inverse=True)
            X = sparse.csr_matrix(
                (np.concatenate(vals) if vals else np.zeros(0, np.float32),
                 (code, np.concatenate(genes) if genes else np.zeros(0, np.int32))),
                shape=(len(uk), len(gtab)),
            )  # fmt: skip
            X.sum_duplicates()
            x, y = uk >> 32, uk & ((1 << 32) - 1)
            obs = pd.DataFrame({"x": x, "y": y})
            names = uk.astype(str).astype(object)
            bin_type = "bins"
    gname = "geneName" if "geneName" in gtab.dtype.names else "geneID"
    var = pd.DataFrame({"gene_ids": _names(gtab, "geneID")}, index=pd.Index(_names(gtab, gname), dtype=str))
    obs.index = pd.Index(names, dtype=str)
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names_make_unique()
    a.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.int64)
    a.uns.update(
        {"bin_type": bin_type, "bin_size": bsize, "resolution": int(res), "sn": str(sn or p.name.split(".")[0])}
    )
    return a


def read_gem(path: str | Path, *, bin_size: int | None = None) -> ad.AnnData:
    """Read a Stereo-seq GEM / GEM.gz (``geneID x y MIDCount`` [``ExonCount``] [``CellID`` / ``label``]).

    Parameters
    ----------
    path
        SAW GEM (``#`` metadata lines are skipped).
    bin_size
        Square-bin side in DNB; required unless the file has a ``CellID`` (or old ``label``) column, which gives cell
        bins (position = mean DNB coordinate of the cell).

    Returns
    -------
    anndata.AnnData
        Same conventions as :func:`read_gef` (packed integer obs names, DNB coordinates, stereopy ``uns`` keys).
    """
    from spaclone.io._table import peek_table

    head = peek_table(path, n=5)
    cols = list(head.columns)
    cnt = next((c for c in ("MIDCount", "MIDCounts", "UMICount") if c in cols), None)
    if not {"geneID", "x", "y"} <= set(cols) or cnt is None:
        raise ValueError(f"{path}: a GEM needs geneID, x, y and MIDCount columns; got {cols}")
    cell = next((c for c in ("CellID", "cell_id", "label") if c in cols), None)
    gene_col = "geneName" if "geneName" in cols else "geneID"
    use = ["geneID", "x", "y", cnt] + ([cell] if cell else []) + (["geneName"] if gene_col == "geneName" else [])
    df = pd.read_csv(path, sep=head.sep, skiprows=head.skiprows, usecols=use,
                     dtype={"geneID": str, "geneName": str, "x": np.int64, "y": np.int64, cnt: np.int64})  # fmt: skip
    if cell:
        df = df[df[cell].astype(str) != "0"]
        key, uniq = pd.factorize(df[cell].astype(str))
        pos = df.assign(_k=key).groupby("_k")[["x", "y"]].mean()
        x, y = np.round(pos["x"].to_numpy()).astype(np.int64), np.round(pos["y"].to_numpy()).astype(np.int64)
        names = np.asarray(uniq, dtype=object)
        bin_type, bsize = "cell_bins", 1
    else:
        if bin_size is None:
            raise ValueError(f"{Path(path).name} is a square-bin GEM: pass bin_size= (bin side in DNB, e.g. 20)")
        bsize = int(bin_size)
        k = ((df["x"].to_numpy() // bsize * bsize) << 32) | (df["y"].to_numpy() // bsize * bsize)
        uk, key = np.unique(k, return_inverse=True)
        x, y = uk >> 32, uk & ((1 << 32) - 1)
        names = uk.astype(str).astype(object)
        bin_type = "bins"
    gk, genes = pd.factorize(df[gene_col].astype(str))
    X = sparse.csr_matrix((df[cnt].to_numpy(np.float32), (key, gk)), shape=(len(names), len(genes)))
    X.sum_duplicates()
    a = ad.AnnData(X=X, obs=pd.DataFrame({"x": x, "y": y}, index=pd.Index(names, dtype=str)),
                   var=pd.DataFrame(index=pd.Index(genes, dtype=str)))  # fmt: skip
    a.var_names_make_unique()
    a.obsm["spatial"] = np.column_stack([x, y]).astype(np.int64)
    a.uns.update(
        {"bin_type": bin_type, "bin_size": bsize, "resolution": int(_DNB_NM), "sn": Path(path).name.split(".")[0]}
    )
    return a


def read_slideseq(path: str | Path) -> ad.AnnData:
    """Read Slide-seq bead tables: ``BeadLocationsForR.csv`` (barcodes, xcoord, ycoord) + ``MappedDGEForR.csv``.

    The bead coordinates are puck-image pixels; their size is not stored in the files, so :func:`spaclone.read` needs
    ``unit_um=`` (Slide-seqV2 publications use 0.65 µm per pixel).

    Parameters
    ----------
    path
        Folder with the two tables (or the bead location file next to the DGE).

    Returns
    -------
    anndata.AnnData
        Beads × genes with ``obsm["spatial"]`` in pixels.
    """
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    loc = _find(folder, "*BeadLocationsForR.csv*", "*bead_locations*.csv*", "*matched_bead_locations*")
    dge = _find(folder, "*MappedDGEForR.csv*", "*.digital_expression.txt*", "*DGE*.csv*")
    if loc is None or dge is None:
        raise FileNotFoundError(f"{folder}: Slide-seq needs BeadLocationsForR.csv and MappedDGEForR.csv")
    b = pd.read_csv(loc)
    b.columns = [c.strip().lower() for c in b.columns]
    bc = next(c for c in ("barcodes", "barcode") if c in b.columns)
    b = b.set_index(bc)
    m = pd.read_csv(dge, index_col=0)
    beads = [c for c in m.columns if c in b.index]
    a = ad.AnnData(X=sparse.csr_matrix(m[beads].to_numpy(np.float32).T),
                   obs=pd.DataFrame(index=pd.Index(beads, dtype=str)), var=pd.DataFrame(index=m.index.astype(str)))  # fmt: skip
    a.obsm["spatial"] = b.loc[beads, ["xcoord", "ycoord"]].to_numpy(dtype=float)
    return a


def _read_spatialdata(source: Any) -> ad.AnnData:
    try:
        import spatialdata as sd
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError("reading SpatialData needs 'pip install spaclone[spatialdata]'") from e
    sdata = source if hasattr(source, "tables") else sd.read_zarr(source)
    names = list(sdata.tables.keys())
    if len(names) != 1:
        raise ValueError(f"the SpatialData object holds {len(names)} tables {names}; pass sdata.tables[name] as gex")
    return sdata.tables[names[0]]


def read_gex(
    source: Any, *, gex_format: str = "auto", bin_size: int | None = None
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Load a GEX source (any of :data:`GEX_FORMATS`) as an AnnData with coordinates in source units.

    Parameters
    ----------
    source
        AnnData, ``.h5ad``, Space Ranger SD folder, Visium HD bin folder, Xenium folder, ``.gef``, ``.gem``,
        Slide-seq folder or SpatialData object / ``.zarr``.
    gex_format
        ``"auto"`` or a key of :data:`GEX_FORMATS`.
    bin_size
        Square-bin size in DNB for Stereo-seq GEF / GEM.

    Returns
    -------
    tuple
        The AnnData and a record ``{"format", "evidence", "file"}``.

    Raises
    ------
    ValueError
        Unrecognised source, a stereopy-native h5ad or Seurat ``.rds`` (with the conversion snippet), a square-bin
        GEF / GEM without ``bin_size``.
    """
    if gex_format == "auto":
        fmt, ev = gex_format_detect(source)
    else:
        if gex_format not in GEX_FORMATS:
            raise ValueError(f"unknown gex_format {gex_format!r}; known: {', '.join(GEX_FORMATS)}")
        fmt, ev = gex_format, [f"given: gex_format={gex_format!r}"]
    file = None if isinstance(source, ad.AnnData) or hasattr(source, "tables") else str(source)
    if fmt == "anndata":
        a = source if isinstance(source, ad.AnnData) else ad.read_h5ad(source)
    elif fmt in ("spaceranger", "visium_hd"):
        a = read_spaceranger(source)
    elif fmt == "xenium":
        a = read_xenium(source)
    elif fmt in ("stereo_gef", "stereo_cellbin_gef"):
        a = read_gef(source, bin_size=bin_size)
    elif fmt == "stereo_gem":
        a = read_gem(source, bin_size=bin_size)
    elif fmt == "slideseq":
        a = read_slideseq(source)
    elif fmt == "spatialdata":
        a = _read_spatialdata(source)
    else:  # seurat_rds
        rscript = shutil.which("Rscript")
        raise ValueError(
            f"{source} is an R object (Seurat .rds); spaclone reads AnnData. Convert it once in R"
            f"{' (Rscript found at ' + rscript + ')' if rscript else ''}:\n{_SEURAT_SNIPPET}\nthen pass the .h5ad."
        )
    return a, {"format": fmt, "evidence": ev, "file": file}


def gex_format_detect(source: Any) -> tuple[str, list[str]]:
    """Alias of :func:`gex_format` (kept separate so that ``read_gex(gex_format=...)`` can shadow the name)."""
    return gex_format(source)


# ------------------------------------------------------------------------------------------------- coordinates


_PAIRS = (
    ("x", "y"),
    ("X", "Y"),
    ("x_centroid", "y_centroid"),
    ("imagecol", "imagerow"),
    ("sdimx", "sdimy"),
    ("center_x", "center_y"),
    ("array_col", "array_row"),
)


def source_coordinates(adata: ad.AnnData, spatial_key: str = "auto") -> tuple[np.ndarray, str, list[str]]:
    """Coordinates of every unit in source units, where they came from, and the alternatives seen.

    Parameters
    ----------
    adata
        The GEX.
    spatial_key
        ``"auto"`` (``obsm["spatial"]``, else one unambiguous alternative), an ``obsm`` key, or ``"obs:x,y"`` for two
        obs columns.

    Returns
    -------
    tuple
        ``(xy (n, 2) float, source, other candidates)``.

    Raises
    ------
    KeyError, ValueError
        No coordinates, non-finite values, or several disagreeing candidates (name ``spatial_key=``).
    """
    cands: list[tuple[str, np.ndarray]] = []
    if spatial_key.startswith("obs:"):
        xk, yk = spatial_key[4:].split(",")
        cands.append((f"obs[{xk!r}], obs[{yk!r}]", adata.obs[[xk, yk]].to_numpy(dtype=float)))
    elif spatial_key != "auto":
        if spatial_key not in adata.obsm:
            raise KeyError(f"read: the GEX has no obsm[{spatial_key!r}] (obsm keys: {list(adata.obsm)})")
        cands.append((f"obsm[{spatial_key!r}]", np.asarray(adata.obsm[spatial_key], dtype=float)))
    else:
        for k in ("spatial", "X_spatial"):
            if k in adata.obsm:
                cands.append((f"obsm[{k!r}]", np.asarray(adata.obsm[k], dtype=float)))
        for xk, yk in _PAIRS:
            if xk in adata.obs.columns and yk in adata.obs.columns:
                try:
                    cands.append((f"obs[{xk!r}], obs[{yk!r}]", adata.obs[[xk, yk]].to_numpy(dtype=float)))
                except (TypeError, ValueError):
                    continue
    if not cands:
        raise KeyError(
            "read: the GEX has no coordinates (obsm['spatial'], obsm['X_spatial'] or an obs pair such as x / y, "
            "x_centroid / y_centroid, imagecol / imagerow, array_col / array_row); pass spatial_key="
        )
    src, xy = cands[0]
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"read: {src} must have two columns (x, y)")
    xy = xy[:, :2]
    others = [s for s, v in cands[1:]]
    if spatial_key == "auto" and not src.startswith("obsm['spatial']") and len(cands) > 1:
        differ = [s for s, v in cands[1:] if v.shape[0] != xy.shape[0] or not np.allclose(v[:, :2], xy, equal_nan=True)]
        if differ:
            raise ValueError(
                f"read: several coordinate sets and no obsm['spatial']: {src} and {', '.join(differ)} differ; pass "
                "spatial_key= (an obsm key, or 'obs:xcol,ycol')"
            )
    if not np.isfinite(xy).all():
        raise ValueError(f"read: {src} holds NaN or infinite coordinates")
    return xy, src, others


# ------------------------------------------------------------------------------------------------- units


def _grid_pitch(v: np.ndarray) -> int | None:
    u = np.unique(v)
    if len(u) < 2 or not np.array_equal(u, np.round(u)) or np.abs(u).max() > 2**52:
        return None
    return int(np.gcd.reduce(np.diff(u).astype(np.int64)))


def _id_convention(ids: np.ndarray, xy: np.ndarray) -> str | None:
    """The Stereo-seq id convention in which ``ids`` spell their own integer coordinates, else None."""
    if not len(ids) or not np.array_equal(xy, np.round(xy)):
        return None
    take = np.linspace(0, len(ids) - 1, min(len(ids), 5000)).astype(int)
    s = pd.Series(np.asarray(ids, dtype=object)[take]).astype(str)
    x, y = xy[take, 0].astype(np.int64), xy[take, 1].astype(np.int64)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and (x >= 0).all() and (y >= 0).all():
        packed = (x << 32) | y
        if (
            np.array_equal(num.to_numpy(dtype=np.float64), packed.astype(np.float64))
            and (s.astype("int64").to_numpy() == packed).all()
        ):
            return "packed integer x << 32 | y (stereopy)"
    for sep, label in (("-", "'x-y'"), ("_", "'x_y'")):
        parts = s.str.replace(r"^[^:]*:", "", regex=True).str.split(sep, regex=False)
        if (parts.str.len() == 2).all():
            px = pd.to_numeric(parts.str[0], errors="coerce").to_numpy()
            py = pd.to_numeric(parts.str[1], errors="coerce").to_numpy()
            if np.array_equal(px, x) and np.array_equal(py, y):
                return label + (" with a 'sample:' prefix" if s.str.contains(":").any() else "")
    return None


def _visium_sd_unit(adata: ad.AnnData, xy: np.ndarray, lib: str, spot_px: float) -> tuple[float, str, str]:
    """µm per full-resolution pixel of Visium SD, measured on the 100 µm spot array.

    Spots of one array row two array columns apart are 100 µm apart (centre to centre). ``spot_diameter_fullres`` is
    not used: it is the diameter of a "theoretical 65 µm spot" in the 10x docs and measures 65-70 µm against the
    array pitch across Space Ranger versions.
    """
    if not {"array_row", "array_col"} <= set(adata.obs.columns):
        raise ValueError(
            f"read: Visium library {lib!r} has scale factors but no obs array_row / array_col, so the pixel size "
            "cannot be measured on the 100 µm spot array; keep those columns (Space Ranger tissue_positions) or pass "
            f"unit_um= (spot_diameter_fullres = {spot_px:g} px is a theoretical 65 µm spot per 10x, approximate)"
        )
    o = pd.DataFrame({"r": adata.obs["array_row"].to_numpy(), "c": adata.obs["array_col"].to_numpy(),
                      "x": xy[:, 0], "y": xy[:, 1]})  # fmt: skip
    if "library_id" in adata.obs.columns:
        o["lib"] = adata.obs["library_id"].astype(str).to_numpy()
    else:
        o["lib"] = ""
    key = o.set_index(["lib", "r", "c"])
    nb = o.assign(c=o.c + 2).set_index(["lib", "r", "c"])
    both = key.index.intersection(nb.index)
    if len(both) < 3:
        raise ValueError(f"read: too few neighbouring Visium spots to measure the pixel size of {lib!r}; pass unit_um=")
    a, b = key.loc[both, ["x", "y"]].to_numpy(float), nb.loc[both, ["x", "y"]].to_numpy(float)
    d = np.hypot(*(a - b).T)
    med = float(np.median(d))
    if not med > 0 or np.median(np.abs(d - med)) / med > 0.02:
        raise ValueError(f"read: Visium spot spacing of {lib!r} is irregular in obsm['spatial']; pass unit_um=")
    f = 100.0 / med
    return (
        f,
        "Visium full-resolution pixel",
        (
            f"Visium SD array: spots two array columns apart are 100 µm apart; median {med:.4g} px over {len(d)} pairs "
            f"of library {lib!r}"
        ),
    )


def detect_unit(
    adata: ad.AnnData, xy: np.ndarray, *, coordinates_from: str = "obsm['spatial']"
) -> tuple[float, float, str, str]:
    """µm per coordinate unit along x and y, the unit's name and the rule that decided it (``unit_um="auto"``).

    Parameters
    ----------
    adata
        The GEX (its ``uns`` and obs names are inspected).
    xy
        Its coordinates in source units.
    coordinates_from
        Where ``xy`` came from (an ``obs`` pair of ``x_centroid`` / ``y_centroid`` triggers the Xenium rule).

    Returns
    -------
    tuple
        ``(sx, sy, unit name, rule)``.

    Raises
    ------
    ValueError
        When no technology fact fixes the unit; the message names ``unit_um=`` and what was seen.
    """
    sp = adata.uns.get("spaclone")
    if isinstance(sp, Mapping) and "um_per_unit" in sp:
        return 1.0, 1.0, "µm", "already in µm (uns['spaclone'] written by a spaclone reader)"
    su = adata.uns.get("spaclone_unit")
    if isinstance(su, Mapping) and "um_per_unit" in su:
        f = float(su["um_per_unit"])
        return f, f, "µm" if f == 1.0 else "source unit", str(su.get("rule", "reader fact"))
    vis = adata.uns.get("spatial")
    if isinstance(vis, Mapping) and coordinates_from.startswith("obsm['spatial']"):
        found = []
        for lib, d in vis.items():
            sf = d.get("scalefactors", {}) if isinstance(d, Mapping) else {}
            if "microns_per_pixel" in sf:
                found.append((float(sf["microns_per_pixel"]), "Visium HD full-resolution pixel",
                               f"uns['spatial'][{lib!r}] microns_per_pixel"))  # fmt: skip
            elif "bin_size_um" in sf and "spot_diameter_fullres" in sf:
                found.append((float(sf["bin_size_um"]) / float(sf["spot_diameter_fullres"]),
                              "Visium HD full-resolution pixel", f"bin_size_um / spot_diameter_fullres of {lib!r}"))  # fmt: skip
            elif "spot_diameter_fullres" in sf:
                hd = pd.Series(adata.obs_names[:200]).str.match(_HD_ID.pattern).any()
                if hd:
                    m = _HD_ID.match(str(adata.obs_names[0]))
                    if m:
                        um = float(m.group(1))
                        found.append((um / float(sf["spot_diameter_fullres"]), "Visium HD full-resolution pixel",
                                      f"{m.group(1)} µm bin ids / spot_diameter_fullres of {lib!r}"))  # fmt: skip
                    continue
                found.append(_visium_sd_unit(adata, xy, lib, float(sf["spot_diameter_fullres"])))
        vals = {round(f, 12) for f, _, _ in found}
        if len(vals) > 1:
            raise ValueError(
                f"read: the GEX holds {len(found)} Visium libraries with different scale factors ({sorted(vals)}); "
                "read one section at a time or pass unit_um="
            )
        if found:
            f, name, rule = found[0]
            return f, f, name, f"{rule} = {f:g}"
    bt, res = adata.uns.get("bin_type"), adata.uns.get("resolution")
    if bt is not None and res is not None:
        bt = bt.decode() if isinstance(bt, bytes) else str(np.asarray(bt).reshape(-1)[0])
        if bt in ("bins", "cell_bins"):
            f = float(np.asarray(res).reshape(-1)[0]) / 1000.0
            bs = adata.uns.get("bin_size")
            return (f, f, f"Stereo-seq DNB ({bt}{', bin' + str(int(np.asarray(bs).reshape(-1)[0])) if bs is not None else ''})",
                    f"stereopy / SAW AnnData: uns['bin_type'] = {bt!r}, uns['resolution'] = {float(np.asarray(res).reshape(-1)[0]):g} nm")  # fmt: skip
    conv = _id_convention(np.asarray(adata.obs_names, dtype=object), xy)
    if conv is not None:
        px, py = _grid_pitch(xy[:, 0]), _grid_pitch(xy[:, 1])
        pitch = px if px == py else None
        if conv.startswith("packed") or (pitch in SAW_BIN_SIZES):
            lab = f"bin{pitch}" if pitch else "cell bins"
            return (
                _DNB_NM / 1000.0,
                _DNB_NM / 1000.0,
                f"Stereo-seq DNB ({lab}, {(pitch or 1) * 0.5:g} µm)",
                f"obs names spell the integer coordinates as {conv}"
                + (f" on a {pitch}-unit lattice: Stereo-seq {lab} in DNB units (0.5 µm)" if pitch else
                   ": Stereo-seq DNB units (0.5 µm)"),
            )  # fmt: skip
    if coordinates_from.startswith("obs['x_centroid']") or (
        {"x_centroid", "y_centroid"} <= set(adata.obs.columns)
        and {"transcript_counts", "control_probe_counts"} & set(adata.obs.columns)
        and coordinates_from.startswith("obsm['spatial']")
        and np.allclose(adata.obs[["x_centroid", "y_centroid"]].to_numpy(float), xy)
    ):
        return 1.0, 1.0, "µm", "Xenium cell table: x_centroid / y_centroid are µm (10x Xenium output specification)"
    px = _grid_pitch(xy[:, 0])
    hint = (
        f" Integer coordinates on a {px}-unit lattice: if these are Stereo-seq DNB coordinates pass unit_um=0.5."
        if px
        else ""
    )
    raise ValueError(
        "read: the unit of the GEX coordinates cannot be established from the data (no spaclone, Space Ranger "
        f"scale-factor or stereopy metadata, no Stereo-seq coordinate ids; {coordinates_from} spans "
        f"x {xy[:, 0].min():g}..{xy[:, 0].max():g}, y {xy[:, 1].min():g}..{xy[:, 1].max():g}). Pass unit_um= "
        "(µm per coordinate unit; 1 if they are µm)." + hint + " Known values: Stereo-seq DNB 0.5; Curio / AIR-SPACE "
        "bead units 0.65 (as spaclone.io.read_airspace uses); Visium full-resolution pixel 55 / spot_diameter_fullres."
    )


#: Plausible nearest-neighbour spacing of spatial units in µm (Stereo-seq DNB 0.5 µm up to Visium SD / legacy ST spots
#: 100-200 µm) and the widest plausible section in µm (5 cm).
SPACING_RANGE_UM = (0.2, 300.0)
MAX_SECTION_UM = 50_000.0


def _unit_check(xy_um: Any, *, argument: str = "unit_um", n_max: int = 20_000) -> dict[str, Any]:
    """Whether unit positions in µm look like a tissue section: nearest-neighbour spacing and width.

    Returns ``{"spacing_um", "width_um", "plausible", "message"}``; ``message`` names ``argument`` when the spacing (the
    median nearest-neighbour distance between distinct positions) lies outside :data:`SPACING_RANGE_UM` or the
    section is wider than :data:`MAX_SECTION_UM`.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy_um, dtype=float).reshape(len(xy_um), -1)[:, :2] if len(xy_um) else np.zeros((0, 2))
    xy = np.unique(xy[np.isfinite(xy).all(axis=1)], axis=0)
    out: dict[str, Any] = {"spacing_um": None, "width_um": None, "plausible": True, "message": ""}
    if len(xy) < 2:
        return out
    pts = xy if len(xy) <= n_max else xy[np.linspace(0, len(xy) - 1, n_max).astype(int)]
    d, _ = cKDTree(xy).query(pts, k=2)
    spacing, width = float(np.median(d[:, 1])), float(np.ptp(xy, axis=0).max())
    lo, hi = SPACING_RANGE_UM
    bad = []
    if not lo <= spacing <= hi:
        bad.append(f"units {spacing:.3g} µm apart (nearest neighbours; expected {lo:g}-{hi:g} µm)")
    if width > MAX_SECTION_UM:
        bad.append(f"a section {width / 1e4:.3g} cm wide (expected < {MAX_SECTION_UM / 1e4:g} cm)")
    out.update(spacing_um=spacing, width_um=width, plausible=not bad)
    if bad:
        out["message"] = (
            f"the coordinates in µm give {' and '.join(bad)}: check {argument} (µm per coordinate unit, e.g. 0.5 for "
            "Stereo-seq DNB, 0.65 for Curio / AIR-SPACE beads, 1 when the coordinates are µm)"
        )
    return out


_ENSEMBL = re.compile(r"^ENS([A-Z]*)G\d{11}")


def _species_of_genes(genes: Any, *, min_genes: int = 50) -> tuple[str | None, str]:
    """``"mouse"`` / ``"human"`` from the gene names when the evidence is clear, else ``None``; with the evidence.

    Ensembl gene ids decide by their prefix (``ENSMUSG`` mouse, ``ENSG`` human). Gene symbols decide by their case:
    mouse symbols are capitalised (``Cd19``, ``Ighg1``), human symbols upper case (``CD19``, ``IGHG1``). A species is
    named only from at least ``min_genes`` names when >= 80% of them follow one convention and <= 10% the other.
    """
    names = [str(g) for g in genes]
    ens = [m.group(1) for m in map(_ENSEMBL.match, names) if m]
    if len(ens) >= min_genes and len(ens) >= 0.8 * len(names):
        mus, hs = sum(p == "MUS" for p in ens) / len(ens), sum(p == "" for p in ens) / len(ens)
        for sp, share in (("mouse", mus), ("human", hs)):
            if share >= 0.9:
                return (
                    sp,
                    f"{share:.0%} of {len(ens):,} Ensembl gene ids start with {'ENSMUSG' if sp == 'mouse' else 'ENSG'}",
                )
        return None, "Ensembl gene ids of several species"
    letters = [[c for c in n if c.isalpha()] for n in names]
    sym = [x for x in letters if len(x) >= 2]
    if len(sym) < min_genes:
        return None, f"too few gene symbols ({len(sym)}) to tell the species"
    upper = sum(all(c.isupper() for c in x) for x in sym) / len(sym)
    title = sum(x[0].isupper() and all(c.islower() for c in x[1:]) for x in sym) / len(sym)
    if title >= 0.8 and upper <= 0.1:
        return "mouse", f"{title:.0%} of {len(sym):,} gene symbols are capitalised (mouse, e.g. 'Cd19')"
    if upper >= 0.8 and title <= 0.1:
        return "human", f"{upper:.0%} of {len(sym):,} gene symbols are upper case (human, e.g. 'CD19')"
    return None, f"gene-name case is not clear ({title:.0%} capitalised, {upper:.0%} upper case)"
