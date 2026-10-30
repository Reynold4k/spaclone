"""Tests of spaclone.pl.lineage_tree / lineage_gallery (figures of bcr.lineage_workflow results)."""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from spaclone import bcr, pl, tl


def _builder():
    """The synthetic lineage table of test_bcr_lineage_workflow.py (one planted lineage + a related clone)."""
    path = Path(__file__).with_name("test_bcr_lineage_workflow.py")
    spec = importlib.util.spec_from_file_location("_lineage_synth", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._lineage_table


def _gex(seed: int = 0):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    n = 6000
    xy = np.column_stack([rng.uniform(0, 4000, n), rng.uniform(0, 1200, n)])
    lab = np.where(xy[:, 0] < 1200, "B_cells", np.where(xy[:, 0] < 2500, "T_cells", "Erythrocyte"))
    lab[rng.random(n) < 0.1] = "Neutrophils"
    obs = pd.DataFrame({"cell2loc_anno": lab}, index=[f"s{i}" for i in range(n)])
    return ad.AnnData(obs=obs, obsm={"spatial": xy})


@pytest.fixture(scope="module")
def table():
    return _builder()()


@pytest.fixture(scope="module")
def result(table):
    return bcr.lineage_workflow(table, _gex(), min_tips=5, rng=1)


@pytest.fixture(scope="module")
def result_share(table, result):
    """Workflow result with a CSR HotspotResult (switched share map) from the loci of the first run."""
    rec = result.records
    loci = result.loci.copy()
    sw = rec.groupby("locus_id").isotype.agg(lambda s: (s == "IgG").mean() >= 0.5)
    loci["switched"] = loci.locus_id.map(sw).astype(bool)
    hs = tl.hotspots(loci, mark="switched", sigma_um=100.0, n_perm=19, min_neff=1.0, rng=0)
    return bcr.lineage_workflow(table, _gex(), hotspots=hs, min_tips=5, rng=1)


def _texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _no_cjk(texts):
    return not any(0x2E80 <= ord(ch) <= 0x9FFF or 0xAC00 <= ord(ch) <= 0xD7AF for t in texts for ch in t)


def _overlaps(fig, items):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(name, it.get_window_extent(r)) for name, it in items]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                bad.append((boxes[i][0], boxes[j][0]))
    return bad


def test_lineage_tree_exact(result):
    tid = result.drawn("exact").tree_id.iloc[0]
    fig = pl.lineage_tree(result, tid, grouping="exact", style="dark")
    texts = _texts(fig)
    assert texts[0] if texts else False
    assert any(t.startswith("IGHV1-1  IGHJ1  CARDYFGNW") for t in texts)
    assert any("17 tips" in t for t in texts)
    assert "tree distance from germline (mutations)" in texts
    assert "position along the section (mm)" in texts
    assert {"tip isotype", "tip location", "map"} <= set(texts)
    assert _no_cjk(texts)
    assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    plt.close(fig)


def test_lineage_tree_related_lists_members(result):
    sel = result.drawn("related")
    tid = sel.tree_id.iloc[0]
    fig = pl.lineage_tree(result, tid, grouping="related", style="light")
    texts = _texts(fig)
    assert any("2 related exact clones" in t for t in texts)
    assert any("CARDYFGNW (17)" in t and "CARDYFGDW (3)" in t for t in texts)
    assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    plt.close(fig)


def test_layout_has_no_overlapping_text(result):
    tid = result.drawn("related").tree_id.iloc[0]
    fig = pl.lineage_tree(result, tid, grouping="related")
    items = [(f"text{k}", t) for k, t in enumerate(fig.texts)] + [
        (f"legend{k}", lg) for k, lg in enumerate(fig.legends)
    ]
    assert _overlaps(fig, items) == []
    plt.close(fig)


def test_share_background_and_colorbar(result_share):
    assert result_share.maps["share"] is not None
    tid = result_share.drawn("exact").tree_id.iloc[0]
    fig = pl.lineage_tree(result_share, tid, grouping="exact")
    assert "switched share (IgG+IgA)" in _texts(fig)
    plt.close(fig)
    fig = pl.lineage_tree(result_share, tid, grouping="exact", background="tissue")
    assert "switched share (IgG+IgA)" not in _texts(fig)
    plt.close(fig)


def test_gallery_writes_files_and_index(result, tmp_path):
    paths = pl.lineage_gallery(result, tmp_path / "gal", formats=("png", "pdf"), dpi=40)
    names = [p.name for p in paths]
    assert names[-1] == "lineage_gallery.csv"
    assert {"lineage_exact_01.png", "lineage_exact_01.pdf", "lineage_related_01.png"} <= set(names)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    idx = pd.read_csv(paths[-1])
    n_drawn = int(result.selection.drawn.sum())
    assert len(idx) == n_drawn
    assert list(idx.columns[:4]) == ["file", "grouping", "rank", "tree_id"]
    assert set(idx.file) <= set(names)
    # same result, same files
    again = pl.lineage_gallery(result, tmp_path / "gal2", dpi=40)
    assert [p.name for p in again] == [n for n in names if not n.endswith(".pdf")]


def test_errors(result):
    with pytest.raises(KeyError, match="no exact tree"):
        pl.lineage_tree(result, "IGHV0|IGHJ0|CXW", grouping="exact")
    tid = result.drawn("exact").tree_id.iloc[0]
    with pytest.raises(ValueError, match="switched-share map"):
        pl.lineage_tree(result, tid, background="share")
    with pytest.raises(ValueError, match="style"):
        pl.lineage_tree(result, tid, style="neon")


def test_legend_lists_only_location_classes_the_workflow_computed(result):
    import copy

    res = copy.copy(result)
    res.maps = {k: v for k, v in (result.maps or {}).items() if k not in ("compartments", "hotspots")}
    tid = res.drawn("exact").tree_id.iloc[0]
    fig = pl.lineage_tree(res, tid, grouping="exact")
    texts = _texts(fig)
    assert not [t for t in texts if t.startswith(("follicle (0)", "T zone (0)", "red pulp (0)", "CSR hotspot (0)"))]
    plt.close(fig)
