"""Tests of spaclone.pl.hotspot_atlas, hotspot_overview and region_zoom on small hand-made workflow results."""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import spaclone.pl as pl
from spaclone.tl import Grid, HotspotResult, Raster, ShareMap

FAST = {"figsize": (8.0, 4.5), "dpi": 50}


def _disc(grid, cx, cy, r):
    gx, gy = grid.centres()
    return (gx[:, None] - cx) ** 2 + (gy[None, :] - cy) ** 2 <= r**2


def _hotspots(name, grid, tissue, labels, seed):
    rng = np.random.default_rng(seed)
    share = np.clip(0.2 + 0.6 * (labels > 0) + rng.normal(0, 0.05, grid.shape), 0, 1)
    share = np.where(tissue, share, np.nan)
    sm = ShareMap(grid, share, np.full(grid.shape, np.nan), share - 0.2, np.where(tissue, 50.0, 0.0), tissue, 0.2,
                  500, 100, {"sigma_um": 30.0, "min_neff": 10.0})  # fmt: skip
    ids = np.setdiff1d(np.unique(labels), [0])
    tab = pd.DataFrame({"hotspot": ids, "direction": "hot"})
    return HotspotResult(name=name, table=tab, map=sm, labels=Raster(labels, grid), threshold={"hot": 3.0})


def fake_result(n_regions=3, *, seed=0, angle_deg=0.0, extra=True, gex=True):
    """An elongated section (900 x 300 µm) with ``n_regions`` CSR hotspots that overlap SHM hotspots."""
    rng = np.random.default_rng(seed)
    grid = Grid((0.0, 0.0), 10.0, (100, 100))
    gx, gy = grid.centres()
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    a = np.radians(angle_deg)
    u = (X - 500) * np.cos(a) + (Y - 500) * np.sin(a)
    v = -(X - 500) * np.sin(a) + (Y - 500) * np.cos(a)
    tissue = (u / 450) ** 2 + (v / 150) ** 2 <= 1
    csr = np.zeros(grid.shape, dtype=np.int64)
    shm = np.zeros(grid.shape, dtype=np.int64)
    centres = []
    for k in range(n_regions):
        t = -330 + k * (660 / max(n_regions - 1, 1)) if n_regions > 1 else 0.0
        cx, cy = 500 + t * np.cos(a), 500 + t * np.sin(a)
        centres.append((cx, cy))
        csr[_disc(grid, cx, cy, 35)] = k + 1
        shm[_disc(grid, cx + 12, cy, 30)] = k + 1
    if extra:  # a CSR hotspot that touches no SHM hotspot
        csr[_disc(grid, 500 - 60 * np.sin(a), 500 + 100 * np.cos(a), 20) & tissue] = n_regions + 1
    n = 800
    pts = np.column_stack([rng.uniform(0, 1000, 4 * n), rng.uniform(0, 1000, 4 * n)])
    i, j, ins = grid.index(pts[:, 0], pts[:, 1])
    pts = pts[ins & tissue[i, j]][:n]
    iso = rng.choice(["IgM", "IgD", "IgG", "IgA", "unknown"], len(pts), p=[0.5, 0.2, 0.15, 0.1, 0.05])
    sw = pd.array(np.where(iso == "unknown", None, np.isin(iso, ["IgG", "IgA"])), dtype="boolean")
    ev = rng.random(len(pts)) < 0.7
    mut = pd.array(np.where(ev, rng.random(len(pts)) < 0.3, None), dtype="boolean")
    loci = pd.DataFrame({"locus_id": np.arange(len(pts)), "clone_id": [f"c{k % 97}" for k in range(len(pts))],
                         "x_um": pts[:, 0], "y_um": pts[:, 1], "isotype_class": pd.array(iso, dtype="string"),
                         "switched": sw, "mutated": mut, "evaluable": ev})  # fmt: skip
    regions = pd.DataFrame({"region_id": np.arange(1, n_regions + 1), "csr_hotspots": [[k + 1] for k in range(n_regions)],
                            "shm_hotspots": [[k + 1] for k in range(n_regions)],
                            "x_um": [c[0] for c in centres], "y_um": [c[1] for c in centres],
                            "switched_share": rng.uniform(0.3, 0.6, n_regions),
                            "mutated_share": rng.uniform(0.2, 0.5, n_regions),
                            "gex_label": ["ASC", "GC B", "ASC", "plasma cell", "ASC", "GC B"][:n_regions]})  # fmt: skip
    res = types.SimpleNamespace(loci=loci, grid=grid, tissue=Raster(tissue, grid), csr=_hotspots("csr", grid, tissue, csr, 1),
                                shm=_hotspots("shm", grid, tissue, shm, 2), regions=regions,
                                params={"min_excess": 3})  # fmt: skip
    if gex:
        ux, uy = np.meshgrid(np.arange(5, 1000, 20.0), np.arange(5, 1000, 20.0), indexing="ij")
        ux, uy = ux.ravel(), uy.ravel()
        ui, uj, uin = grid.index(ux, uy)
        keep = uin & tissue[ui, uj]
        ids = pd.Index([f"u{k}" for k in range(int(keep.sum()))])
        units = pd.DataFrame({"x_um": ux[keep], "y_um": uy[keep]}, index=ids)
        score = pd.Series(rng.normal(0, 1, len(ids)), index=ids)
        score[rng.random(len(ids)) < 0.4] = np.nan  # units without a score (non-B)
        res.module_maps = {"ASC": score, "GC B": score * 0.5, "plasma cell": score + 1}
        res.gex_units = units
        th = np.linspace(0, 2 * np.pi, 40)
        cx, cy = centres[0] if centres else (500.0, 500.0)
        res.gc_gex = pd.DataFrame({"contour_id": 1, "x_um": cx + 30 + 20 * np.cos(th), "y_um": cy + 20 * np.sin(th)})
    return res


def _texts(fig):
    legends = [ax.get_legend() for ax in fig.axes] + list(fig.legends)
    return (
        [t.get_text() for ax in fig.axes for t in ax.texts]
        + [t.get_text() for t in fig.texts]
        + [t.get_text() for leg in legends if leg is not None for t in leg.get_texts()]
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_atlas_axes_per_region_count(n):
    fig = pl.hotspot_atlas(fake_result(n), **FAST)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2 + 3 * n
    if n == 0:
        assert any("No region where a CSR hotspot overlaps" in t for t in _texts(fig))
    else:
        titles = [t for t in _texts(fig) if t.startswith("CSR · region")]
        assert len(titles) == n
    plt.close(fig)


def test_atlas_more_than_four_regions_shows_first_three_along_the_section():
    res = fake_result(6)
    fig = pl.hotspot_atlas(res, **FAST)
    assert len(fig.axes) == 2 + 9
    titles = sorted(t for t in _texts(fig) if t.startswith("CSR · region"))
    assert [t.split("·")[1].strip() for t in titles] == ["region 1", "region 2", "region 3"]
    assert any("3 of 6 regions shown" in t for t in _texts(fig))
    plt.close(fig)


def test_atlas_region_choice_and_errors():
    res = fake_result(4)
    fig = pl.hotspot_atlas(res, regions=[4, 2], **FAST)
    titles = [t for t in _texts(fig) if t.startswith("CSR · region")]
    assert len(titles) == 2 and {t.split("·")[1].strip() for t in titles} == {"region 2", "region 4"}
    plt.close(fig)
    with pytest.raises(KeyError):
        pl.hotspot_atlas(res, regions=[9], **FAST)
    with pytest.raises(ValueError):
        pl.hotspot_atlas(fake_result(6), regions=[1, 2, 3, 4, 5], **FAST)
    with pytest.raises(ValueError):
        pl.hotspot_atlas(res, style="neon", **FAST)


def test_atlas_colours_are_fixed_by_category():
    figs = [pl.hotspot_atlas(fake_result(3, seed=s), **FAST) for s in (0, 5)]
    for fig in figs:
        csr_zoom = fig.axes[2]  # first region, CSR row
        cols = {to_hex(c) for coll in csr_zoom.collections for c in np.atleast_2d(coll.get_facecolors())}
        assert "#ff3b3b" in cols and "#ffa31a" in cols  # IgG red, IgA orange, whatever else is on the panel
        assert "#8c8c8c" in cols  # IgM / IgD grey
        plt.close(fig)


def test_atlas_is_deterministic():
    res = fake_result(2)
    a = pl.hotspot_atlas(res, **FAST)
    b = pl.hotspot_atlas(res, **FAST)
    for ax_a, ax_b in zip(a.axes, b.axes, strict=True):
        assert ax_a.get_xlim() == ax_b.get_xlim() and ax_a.get_ylim() == ax_b.get_ylim()
    a.canvas.draw()
    b.canvas.draw()
    assert np.array_equal(np.asarray(a.canvas.buffer_rgba()), np.asarray(b.canvas.buffer_rgba()))
    plt.close(a)
    plt.close(b)


def test_section_is_rotated_long_axis_horizontal_and_image_convention():
    res = fake_result(2, angle_deg=55.0)
    fig = pl.hotspot_overview(res, **FAST)
    ax = fig.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert abs(x1 - x0) > 2 * abs(y1 - y0)
    assert y0 > y1  # y increases downwards
    plt.close(fig)
    fig = pl.hotspot_overview(res, rotate=False, invert_y=False, **FAST)
    ax = fig.axes[0]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    plt.close(fig)


def test_overview_and_region_zoom_return_types():
    res = fake_result(3)
    fig = pl.hotspot_overview(res, **FAST)
    assert isinstance(fig, matplotlib.figure.Figure) and len(fig.axes) == 2
    plt.close(fig)
    fig = pl.region_zoom(res, 2, **FAST)
    assert isinstance(fig, matplotlib.figure.Figure) and len(fig.axes) == 3
    plt.close(fig)
    ax = pl.region_zoom(res, 1, channels="gex", **FAST)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)
    _, axes = plt.subplots(1, 2)
    fig = pl.region_zoom(res, 3, channels=("csr", "shm"), ax=list(axes))
    assert fig is axes[0].figure
    plt.close(fig)
    with pytest.raises(KeyError):
        pl.region_zoom(res, 99, **FAST)
    with pytest.raises(ValueError):
        pl.region_zoom(res, 1, channels="tcr", **FAST)


def test_light_style_and_mapping_input_and_missing_optional_fields():
    res = fake_result(2, gex=False)
    fig = pl.hotspot_atlas(vars(res), style="light", **FAST)  # a mapping works like the result object
    assert to_hex(fig.get_facecolor()) == "#ffffff"
    assert any("Hotspot ·" in t for t in _texts(fig))
    assert any("no score map" in t or "no GEX profile" in t for t in _texts(fig))
    plt.close(fig)
    res.shm = None
    fig = pl.hotspot_atlas(res, **FAST)
    assert any("no SHM map" in t for t in _texts(fig))
    plt.close(fig)


def test_atlas_without_an_shm_map_shows_the_csr_map_alone():
    res = fake_result(2, gex=False)
    res.shm = None
    res.regions = res.regions.iloc[:0]
    res.messages = ["no V alignments (v_sequence_alignment / v_germline_alignment, ...): SHM maps skipped"]
    fig = pl.hotspot_atlas(res, **FAST)
    assert len(fig.axes) == 1 and fig.axes[0].get_position().width > 0.9  # no half figure of empty outline
    texts = " ".join(_texts(fig))
    assert (
        "No SHM map in the result" in texts
        and "have no V alignments" in texts
        and "no SHM map in the result" not in texts
    )
    plt.close(fig)


def test_never_shows_or_saves(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("plot functions must not show or save")

    monkeypatch.setattr(plt, "show", boom)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", boom)
    res = fake_result(1)
    for fig in (pl.hotspot_atlas(res, **FAST), pl.hotspot_overview(res, **FAST), pl.region_zoom(res, 1, **FAST)):
        plt.close(fig)


def test_regions_without_ids_are_numbered_along_the_section():
    res = fake_result(3)
    res.regions = res.regions.drop(columns="region_id").iloc[::-1].reset_index(drop=True)
    fig = pl.hotspot_atlas(res, **FAST)
    xs = []
    for ax in fig.axes[2::3]:
        lbl = [t.get_text() for t in ax.texts if t.get_text().startswith("CSR")][0]
        xs.append((ax.get_position().x0, lbl.split("·")[1].strip()))
    assert [lbl for _, lbl in sorted(xs)] == ["region 1", "region 2", "region 3"]
    plt.close(fig)


def _paired_section(seed=0, nx=48, ny=32, centre=(160.0, 160.0, 50.0)):
    """A bin20 GEX section (DNB coordinates) and IGH contigs with one planted switched + mutated ASC disc."""
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(seed)
    markers = {"plasmablast/ASC": ["Jchain", "Xbp1", "Prdm1", "Irf4", "Mzb1"], "T": ["Cd3e", "Cd3g", "Cd3d", "Cd4"],
               "GC B": ["Aicda", "S1pr2", "Rgs13", "Gcsam"]}  # fmt: skip
    genes = [g for v in markers.values() for g in v] + ["Actb", "Malat1"]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    xd, yd = 1000 + 20 * ii, 2000 + 20 * jj
    ids = np.array([f"{a}-{b}" for a, b in zip(xd, yd, strict=True)], dtype=object)
    hot = (10.0 * ii - centre[0]) ** 2 + (10.0 * jj - centre[1]) ** 2 < centre[2] ** 2
    lam = np.full((len(ids), len(genes)), 0.15)
    lam[:, genes.index("Actb")], lam[:, genes.index("Malat1")] = 15.0, 10.0
    for g in markers["plasmablast/ASC"]:
        lam[hot, genes.index(g)] += 3.0
    a = ad.AnnData(X=sparse.csr_matrix(rng.poisson(lam).astype(np.float32)), obs=pd.DataFrame(index=ids),
                   var=pd.DataFrame(index=genes))  # fmt: skip
    a.obsm["spatial"] = np.column_stack([xd, yd]).astype(np.int64)
    rb = np.repeat(np.arange(len(ids)), rng.poisson(np.where(hot, 2.5, 0.6)))
    m = len(rb)
    u = rng.random(m)
    c_call = np.where(hot[rb], np.select([u < 0.3, u < 0.75], ["IGHM*01", "IGHG1*01"], "IGHA*01"),
                      np.select([u < 0.75, u < 0.9, u < 0.97], ["IGHM*01", "IGHD*01", "IGHG1*01"], "IGHA*01"))  # fmt: skip
    mutated = rng.random(m) < np.where(hot[rb], 0.65, 0.05)
    alleles = [f"IGHV1-{k}*01" for k in (26, 31, 53, 64)]
    germ = {al: "".join(rng.choice(list("ACGT"), 260)) for al in alleles}
    v = rng.choice(alleles, m)
    seq = []
    for al, k in zip(v, np.where(mutated, 3 + rng.poisson(3, m), 0), strict=True):
        s = np.array(list(germ[al]))
        at = rng.choice(260, size=k, replace=False)
        s[at] = np.where(s[at] != "A", "A", "C")  # k true mismatches
        seq.append("".join(s))
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    air = pd.DataFrame({"cell_id": ids[rb], "locus": "IGH", "v_call": v, "j_call": "IGHJ4*01", "c_call": c_call,
                        "junction_aa": ["C" + "".join(rng.choice(aa, rng.integers(8, 13))) + "W" for _ in range(m)],
                        "productive": "T", "consensus_count": rng.choice([12, 40, 80], m),
                        "v_sequence_alignment": seq, "v_germline_alignment": [germ[al] for al in v]})  # fmt: skip
    return a, air


def test_workflow_to_atlas_end_to_end():
    """io.read -> bcr.hotspot_workflow -> hotspot_atlas / region_zoom / csr_map on a planted synthetic section."""
    import spaclone as sc

    a, air = _paired_section()
    res = sc.bcr.hotspot_workflow(sc.read(a, air), rng=0, sigma_um=30.0, n_perm=99, n_perm_region=199,
                                  clone_null=False, gc_call=False)  # fmt: skip
    assert len(res.regions) >= 1
    fig = pl.hotspot_atlas(res, **FAST)
    n = min(len(res.regions), 4 if len(res.regions) <= 4 else 3)
    assert len(fig.axes) == 2 + 3 * n
    texts = _texts(fig)
    assert any(t.startswith("CSR · region 1") and "switched" in t for t in texts)
    assert any(t.startswith("SHM ·") and "mutated" in t for t in texts)
    assert any(t.startswith("Hotspot · ASC") for t in texts)  # "plasmablast/ASC" shown as the archive does
    assert any("≥ 3 excess V mutations" in t for t in texts)  # params["min_excess"]
    assert fig.axes[4].get_images(), "the module score map is drawn in the first region's GEX zoom"
    plt.close(fig)
    rid = int(res.regions.region_id.iloc[0])
    fig = pl.region_zoom(res, rid, score_mask=res.units.region_id > 0, **FAST)
    assert len(fig.axes) == 3
    plt.close(fig)
    fig = pl.csr_map(res, n_perm=19, **FAST)
    assert any("Isotype mix by region" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_clone_level_p_line():
    """One short line per region with its clone-level map p-values; nothing without them or with clone_p=False."""
    res = fake_result(3)
    res.regions["csr_clone_p"] = [1 / 1001, 0.0183, np.nan]
    res.regions["shm_clone_p"] = [0.4123, 1.0, np.nan]
    res.params = {"min_excess": 3, "n_perm": 1000}
    fig = pl.hotspot_atlas(res, **FAST)
    lines = [t for t in _texts(fig) if t.startswith("clone-level p")]
    assert lines == ["clone-level p: CSR < 0.001 · SHM = 0.412", "clone-level p: CSR = 0.0183 · SHM = 1"]
    gex_tiles = [ax for ax in fig.axes if any(t.get_text().startswith("Hotspot") for t in ax.texts)]
    assert all(
        t.get_position()[1] < 0.5 for ax in gex_tiles for t in ax.texts if t.get_text().startswith("clone-level p")
    )  # bottom of the GEX tile, away from the title
    plt.close(fig)
    fig = pl.hotspot_atlas(res, clone_p=False, **FAST)
    assert not [t for t in _texts(fig) if t.startswith("clone-level p")]
    plt.close(fig)
    ax = pl.region_zoom(res, 2, channels="csr", **FAST)
    assert [t.get_text() for t in ax.texts if t.get_text().startswith("clone-level p")] == [
        "clone-level p: CSR = 0.0183 · SHM = 1"
    ]
    plt.close(ax.figure)
    fig = pl.hotspot_atlas(fake_result(2), **FAST)  # no clone-level columns: no line
    assert not [t for t in _texts(fig) if t.startswith("clone-level p")]
    plt.close(fig)
