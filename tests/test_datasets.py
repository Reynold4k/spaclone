import hashlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spaclone import datasets, pl
from spaclone.datasets import fetch
from spaclone.io import SpatialAIR

# ------------------------------------------------------------------------------------------------------ registry


def test_registry_entries_are_exact_geo_files():
    """All six AIR-SPACE nodes (GSE286452): bead counts GSM8727829-34, clone tables GSM8727835-40, in time order."""
    assert list(datasets.AIRSPACE_FILES) == ["Mock", "D3", "D7", "D10", "D14", "D21"]
    for i, (sample, files) in enumerate(datasets.AIRSPACE_FILES.items()):
        assert set(files) == {"h5ad", "clone_table"}
        for kind, e in files.items():
            assert e["url"].startswith("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8727nnn/GSM")
            name = e["url"].rsplit("/", 1)[1]
            gsm = re.match(r"(GSM\d+)_", name).group(1)
            assert f"/{gsm}/suppl/{name}" in e["url"] and f"_{sample}_" in name
            assert gsm == f"GSM{8727829 + i + (6 if kind == 'clone_table' else 0)}"
            assert name.endswith(".h5ad" if kind == "h5ad" else "_ovp_clone_table.txt.gz")
            assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) and e["size"] > 1_000_000
    hashes = [e["sha256"] for files in datasets.AIRSPACE_FILES.values() for e in files.values()]
    assert len(set(hashes)) == 12


def _fake(monkeypatch, tmp_path, content=b"spaclone test file\n"):
    name = "GSM0000001_Mock_ovp_clone_table.txt.gz"
    entry = {
        "url": f"https://example.invalid/{name}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    reg = {
        "Mock": {"h5ad": dict(entry, url="https://example.invalid/GSM0000002_adata_Mock.h5ad"), "clone_table": entry}
    }
    monkeypatch.setattr(fetch, "AIRSPACE_FILES", reg)
    return name, content


def test_cached_files_are_verified_without_download(monkeypatch, tmp_path):
    name, content = _fake(monkeypatch, tmp_path)
    (tmp_path / name).write_bytes(content)
    (tmp_path / "GSM0000002_adata_Mock.h5ad").write_bytes(content)
    h5, ct = datasets.airspace("Mock", data_dir=tmp_path, download=False)
    assert ct.name == name and h5.exists()
    (tmp_path / name).write_bytes(content[:-1] + b"X")
    with pytest.raises(FileNotFoundError, match="corrupt"):
        datasets.airspace("Mock", data_dir=tmp_path, download=False)


def test_missing_files_without_download_raise(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="not in the cache"):
        datasets.airspace("Mock", data_dir=tmp_path, download=False)
    with pytest.raises(ValueError, match="unknown AIR-SPACE sample"):
        datasets.airspace("D99", data_dir=tmp_path, download=False)


def test_download_without_pooch_explains_the_extra(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "pooch", None)
    with pytest.raises(ImportError, match=r"spaclone\[datasets\]"):
        datasets.airspace("Mock", data_dir=tmp_path, download=True)


def test_data_dir_order(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACLONE_DATA_DIR", str(tmp_path / "env"))
    assert datasets.data_dir() == tmp_path / "env"
    assert datasets.data_dir(tmp_path / "arg") == tmp_path / "arg"


def test_synthetic_example_needs_no_download():
    a, b = datasets.synthetic_example(), datasets.synthetic_example()
    assert isinstance(a, SpatialAIR)
    pd.testing.assert_frame_equal(a.air, b.air)
    assert set(a.air.truth) == {"genuine", "swapped", "sink"}


# --------------------------------------------------------------------------------------------- pl.spatial_records


def test_spatial_records_draws_and_returns_axes():
    sa = datasets.synthetic_example(size_um=300.0, tissue_radius_um=120.0, spot_um=4.0, sink_size=30)
    ax = pl.spatial_records(sa)
    assert ax.get_aspect() == 1.0 and len(ax.collections) >= 3  # glass, tissue, BCR, TCR
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[0].startswith("BCR (") and labels[1].startswith("TCR (")
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # y drawn downwards
    plt.close(ax.figure)
    fig, ax2 = plt.subplots()
    out = pl.spatial_records(sa.air, sa.gex, color="truth", style="dark", highlight=["genuine"], ax=ax2)
    assert out is ax2
    assert matplotlib.colors.to_hex(fig.get_facecolor()) == "#000000"
    assert out.get_legend() is None  # one highlighted category: no legend box
    plt.close(fig)


def test_spatial_records_categories_and_errors():
    air = pd.DataFrame(
        {
            "x_um": np.arange(12.0),
            "y_um": np.zeros(12),
            "locus": ["IGH"] * 6 + ["TRB"] * 6,
            "grp": list("abcdefghijkl"),
            "truth": ["sink", "genuine", "swapped"] * 4,
        }
    )
    ax = pl.spatial_records(air, color="grp", max_categories=4, background="none", scale_bar_um=5)
    labels = [t.get_text().split(" ")[0] for t in ax.get_legend().get_texts()]
    assert labels[-1] == "other" and len(labels) == 4
    plt.close(ax.figure)
    ax = pl.spatial_records(air, color="truth", background=np.array([[0.0, 1.0], [5.0, 2.0]]))
    assert [t.get_text().split(" ")[0] for t in ax.get_legend().get_texts()] == ["genuine", "swapped", "sink"]
    plt.close(ax.figure)
    with pytest.raises(KeyError):
        pl.spatial_records(air, color="nope")
    with pytest.raises(ValueError, match="style"):
        pl.spatial_records(air, style="neon")
