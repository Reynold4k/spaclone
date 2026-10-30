"""Example data: the six public AIR-SPACE lymph nodes from GEO (downloaded once, verified by sha256) and a synthetic chip.

The AIR-SPACE files (Jiang et al., PNAS 2026; GEO GSE286452; mouse popliteal lymph node after Vaccinia infection) are
the bead AnnData of every bead and the per-read MiXCR clone table of a sample. ``"Mock"`` is the uninfected node,
``"D3"``, ``"D7"``, ``"D10"``, ``"D14"`` and ``"D21"`` the days after infection (germinal centres from about day 7;
clones shared between germinal centres at day 21). Downloading needs the optional ``pooch`` package
(``pip install "spaclone[datasets]"``); files already in the cache are verified without it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from spaclone._utils import RngLike
from spaclone.io._container import SpatialAIR

__all__ = ["AIRSPACE_FILES", "airspace", "data_dir", "load_airspace", "synthetic_example"]

logger = logging.getLogger("spaclone")

_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8727nnn"
#: AIR-SPACE GEO files per sample (Mock, then days after infection): exact HTTPS URL, sha256 and size in bytes
#: (URLs and sizes checked against GEO with HEAD requests, sha256 of the downloaded files; 2026-10-30).
AIRSPACE_FILES: dict[str, dict[str, dict[str, Any]]] = {
    "Mock": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727829/suppl/GSM8727829_adata_Mock_with_image_raw_counts.h5ad",
            "sha256": "7d3f2ca1ea35ae83ae7ab3c35f5c4326dbf1de0ef6e5d905857efcd93f850977",
            "size": 67388834,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727835/suppl/GSM8727835_Mock_ovp_clone_table.txt.gz",
            "sha256": "34f024e6059d907db879bc39026824c079b3cd8e0a5952983fad76cb15f102bc",
            "size": 2014880,
        },
    },
    "D3": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727830/suppl/GSM8727830_adata_D3_with_image_raw_counts.h5ad",
            "sha256": "54d5e8b9a44eb088c2e202c78474f18230f8eb1eaf6a66c9eb46aab6525f52d0",
            "size": 96464214,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727836/suppl/GSM8727836_D3_ovp_clone_table.txt.gz",
            "sha256": "8de402293eac1d49e8c8b6dd73b348a206acd107bdfe6f4001fe303d1081cec0",
            "size": 3573749,
        },
    },
    "D7": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727831/suppl/GSM8727831_adata_D7_with_image_raw_counts.h5ad",
            "sha256": "b7c8a827402f2de83c08c63e998d043f0708aa9c3736c68e65d6abca47329965",
            "size": 96009730,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727837/suppl/GSM8727837_D7_ovp_clone_table.txt.gz",
            "sha256": "caa87c2099a56f0cc75f8e04acfd1d834305ad69c277545417ae3a51d55e510b",
            "size": 2817398,
        },
    },
    "D10": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727832/suppl/GSM8727832_adata_D10_with_image_raw_counts.h5ad",
            "sha256": "6dcec6ce0024ab21d260e73017ae7d3d36960ed4381e6469617ccbed43f76380",
            "size": 104516822,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727838/suppl/GSM8727838_D10_ovp_clone_table.txt.gz",
            "sha256": "ea39c8b331ad05d49594f4059e885e5dc9ad722f13b230d5674cafc6fd199232",
            "size": 5362977,
        },
    },
    "D14": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727833/suppl/GSM8727833_adata_D14_with_image_raw_counts.h5ad",
            "sha256": "4dc7bf758c32627311500c20df9aeb9a9b3c422bc24f6bedc346e47c73a4875c",
            "size": 90476350,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727839/suppl/GSM8727839_D14_ovp_clone_table.txt.gz",
            "sha256": "e237cfee453a10b7738f28480147846b2f787392ade05ebb0495381a072ff8ed",
            "size": 3576088,
        },
    },
    "D21": {
        "h5ad": {
            "url": f"{_GEO}/GSM8727834/suppl/GSM8727834_adata_D21_with_image_raw_counts.h5ad",
            "sha256": "98c3bc268a10f93bf51f14608b1b3be90c1158d7fb8f6de618788ac2f612461a",
            "size": 75076938,
        },
        "clone_table": {
            "url": f"{_GEO}/GSM8727840/suppl/GSM8727840_D21_ovp_clone_table.txt.gz",
            "sha256": "464355582e74d7782dc9305446adb466a4c7ab676ffd89e3bfdc601f8ccf38b2",
            "size": 2843163,
        },
    },
}


def data_dir(path: str | Path | None = None) -> Path:
    """The cache folder: ``path``, else ``$SPACLONE_DATA_DIR``, else the user cache (``pooch.os_cache``)."""
    if path is not None:
        return Path(path)
    env = os.environ.get("SPACLONE_DATA_DIR")
    if env:
        return Path(env)
    try:
        import pooch
    except ImportError:
        return Path.home() / ".cache" / "spaclone"
    return Path(pooch.os_cache("spaclone"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(entry: dict[str, Any], folder: Path, download: bool, progressbar: bool) -> Path:
    name = entry["url"].rsplit("/", 1)[1]
    path = folder / name
    if path.exists() and path.stat().st_size == entry["size"] and _sha256(path) == entry["sha256"]:
        return path
    if not download:
        state = "is corrupt (size or sha256 differs)" if path.exists() else "is not in the cache"
        raise FileNotFoundError(f"{name} {state}: {path}; call with download=True to fetch it from GEO")
    try:
        import pooch
    except ImportError as err:
        raise ImportError(
            'downloading example data needs pooch: pip install "spaclone[datasets]" (or pip install pooch)'
        ) from err
    folder.mkdir(parents=True, exist_ok=True)
    out = pooch.retrieve(
        url=entry["url"], known_hash=f"sha256:{entry['sha256']}", fname=name, path=folder, progressbar=progressbar
    )
    return Path(out)


def airspace(
    sample: str = "D21",
    *,
    data_dir: str | Path | None = None,
    download: bool = True,
    progressbar: bool = False,
) -> tuple[Path, Path]:
    """Paths to the bead AnnData and the clone table of one AIR-SPACE GEO sample.

    Parameters
    ----------
    sample
        ``"Mock"``, ``"D3"``, ``"D7"``, ``"D10"``, ``"D14"`` or ``"D21"`` (see :data:`AIRSPACE_FILES`; 69–110 MB
        per sample).
    data_dir
        Cache folder (default :func:`data_dir`).
    download
        Download missing or corrupt files from GEO (needs ``pooch``); ``False`` only verifies the cache.
    progressbar
        Show a download progress bar (needs ``tqdm``).

    Returns
    -------
    tuple of pathlib.Path
        ``(h5ad, clone_table)``, both verified by size and sha256.

    Raises
    ------
    FileNotFoundError
        With ``download=False``, when a file is missing or corrupt.

    Examples
    --------
    >>> h5ad, clones = airspace("D21")  # doctest: +SKIP
    """
    if sample not in AIRSPACE_FILES:
        raise ValueError(f"unknown AIR-SPACE sample {sample!r}; available: {sorted(AIRSPACE_FILES)}")
    folder = globals()["data_dir"](data_dir)
    files = AIRSPACE_FILES[sample]
    h5 = _fetch(files["h5ad"], folder, download, progressbar)
    ct = _fetch(files["clone_table"], folder, download, progressbar)
    return h5, ct


def load_airspace(
    sample: str = "D21",
    *,
    data_dir: str | Path | None = None,
    download: bool = True,
    **kwargs: Any,
) -> SpatialAIR:
    """Fetch one AIR-SPACE sample and read it with :func:`spaclone.io.read_airspace`.

    Parameters
    ----------
    sample, data_dir, download
        See :func:`airspace`.
    **kwargs
        Passed to :func:`spaclone.io.read_airspace` (e.g. ``loci=("IGH", "IGK", "IGL")``, ``keep_x=False``).

    Returns
    -------
    SpatialAIR
        Molecules (every bead kept, off-tissue beads flagged) and the bead GEX.
    """
    from spaclone.io.airspace import read_airspace

    h5, ct = airspace(sample, data_dir=data_dir, download=download)
    return read_airspace(h5, ct, **kwargs)


def synthetic_example(*, rng: RngLike = 0, **kwargs: Any) -> SpatialAIR:
    """A small synthetic section with known truth, no download (:func:`spaclone.sim.synthetic_chip`, seed 0).

    Parameters
    ----------
    rng
        Seed or generator (default 0: the same example every time).
    **kwargs
        Passed to :func:`spaclone.sim.synthetic_chip`.

    Returns
    -------
    SpatialAIR
        Receptor molecules with ``truth`` columns, the spot GEX and the truth tables in ``meta``.

    Examples
    --------
    >>> sa = synthetic_example()
    >>> sa.air.truth.value_counts().index.tolist()
    ['swapped', 'genuine', 'sink']
    """
    from spaclone.sim.synthetic import synthetic_chip

    return synthetic_chip(rng=rng, **kwargs)
