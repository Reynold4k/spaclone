import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse

from spaclone import TestResult, pl
from spaclone.ext import threadfin as tfx


def _light(seed=0, side=24, names="auto"):
    """A ThreadfinSpatialResult-shaped object without running Threadfin (the plots use its tables only)."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    x, y = ii.ravel() * 10.0, jj.ravel() * 10.0
    n = x.size
    red = x >= side * 5.0
    tz = (~red) & (x < 60) & (y < 80)
    ctype = np.where(red, "Erythrocyte", np.where(tz, "T_cells", "B_cells"))
    ids = [f"s{i}" for i in range(n)]
    gex = ad.AnnData(
        X=sparse.csr_matrix(np.ones((n, 2), np.float32)), obs=pd.DataFrame({"ct": pd.Categorical(ctype)}, index=ids)
    )
    gex.obsm["spatial"] = np.c_[x, y]
    gex.obs["reg"] = red & (y < 60)
    ctx = tfx.spatial_context(gex, cell_type_key="ct", regions={"hotspot": "reg"}, names=names)
    clone = np.full(n, None, dtype=object)
    sizes = [12, 9, 8, 6, 5, 3, 3, 2, 2, 2] + [1] * 20
    order = rng.permutation(n)
    k = 0
    for c, m in enumerate(sizes):
        clone[order[k : k + m]] = f"IGHV1-{c}|IGHJ1|CARW{c}W"
        k += m
    pos = np.where(red, -0.3, 0.3) + rng.normal(0, 0.1, n)
    u = pd.DataFrame({"x_um": x, "y_um": y, "clone_id": clone}, index=pd.Index(ids, name="unit_id"))
    u["clonal"] = u.clone_id.notna()
    size = u.clone_id.map(u.clone_id.value_counts())
    u["profiled"] = size.fillna(0).to_numpy() >= 2
    for j in (1, 2, 3):
        u[f"axis_{j}"] = pos if j == 1 else rng.normal(0, 0.1, n)
        cm = u[u.profiled].groupby("clone_id")[f"axis_{j}"].mean()
        u[f"clone_axis_{j}"] = u.clone_id.map(cm).where(u.profiled)
    u["programme"] = np.where(u.profiled, np.where(u.clone_axis_1 > 0, "P1", "P2"), None)
    smp = ctx.sample(x, y)
    for c in smp.columns:
        u[c] = smp[c].to_numpy()
    u["n_spots"] = 1
    pr = u[u.profiled]
    c = pr.groupby("clone_id").agg(
        n_units=("x_um", "size"),
        axis_1=("clone_axis_1", "first"),
        axis_2=("clone_axis_2", "first"),
        axis_3=("clone_axis_3", "first"),
    )
    axes = pd.DataFrame(
        {"axis": [1, 2, 3], "variance": [1.0, 0.2, 0.1], "share": [0.6, 0.12, 0.06], "share_all": [0.6, 0.12, 0.06]}
    )
    params = {"clone_key": "clone_id", "n_axes": 3, "units": "spot", "features": "expression"}
    return tfx.ThreadfinSpatialResult(threadfin=None, units=u, clones=c, axes=axes, params=params, context=ctx)


@pytest.fixture(autouse=True)
def _no_output(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("plot functions must not save or show")

    monkeypatch.setattr(Figure, "savefig", boom)
    monkeypatch.setattr(plt, "show", boom)
    yield
    plt.close("all")


@pytest.mark.parametrize("color", ["clone_axis", "axis", "programme", "compartment", "dist_t_zone_um", "n_spots"])
@pytest.mark.parametrize("style", ["dark", "light"])
def test_clone_state_map_colours(color, style):
    res = _light()
    ax = pl.clone_state_map(res, color=color, style=style, end_labels=("red-pulp end", "white-pulp end"))
    assert isinstance(ax, Axes)
    bg = "#000000" if style == "dark" else "#ffffff"
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == bg
    assert len(ax.collections) >= 2  # background + coloured units (+ outlines)


def test_clone_state_map_options_and_errors():
    res = _light()
    fig, ax = plt.subplots()
    out = pl.clone_state_map(
        res,
        ax=ax,
        background=False,
        outlines=False,
        regions=False,
        colorbar=False,
        legend=False,
        scalebar_um=None,
        invert_y=True,
        title="t",
    )
    assert out is ax and ax.get_ylim()[0] > ax.get_ylim()[1]
    with pytest.raises(KeyError):
        pl.clone_state_map(res, color="nope")
    with pytest.raises(ValueError):
        pl.clone_state_map(res, style="neon")
    with pytest.raises(ValueError):
        pl.clone_state_map(res, axis=7)
    with pytest.raises(TypeError):
        pl.clone_state_map(res.units)


def _fake_niche_test(res, n_bins):
    rows = []
    for c in ("T zone", "follicle", "red pulp"):
        for b in range(n_bins):
            rows.append(
                {"kind": "share", "bin": f"D{b + 1}", "category": c, "null_mean": 0.3, "null_lo": 0.2, "null_hi": 0.4}
            )
    return TestResult(name="niche_by_axis", table=pd.DataFrame(rows))


def test_clone_axis_profile_with_null_and_markers():
    res = _light()
    expr = pd.DataFrame(
        np.random.default_rng(1).random((len(res.units), 3)), index=res.units.index, columns=["Hbb-bs", "Cd3e", "Cr2"]
    )
    fig = pl.clone_axis_profile(res, _fake_niche_test(res, 4), n_bins=4, expression=expr, style="dark")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3  # shares, marker heat map, its colour bar
    fig2 = pl.clone_axis_profile(res, n_bins=4, style="light")
    assert len(fig2.axes) == 1
    grouped = pl.clone_axis_profile(res, n_bins=4, expression=expr, genes={"red": ["Hbb-bs"], "T": ["Cd3e"]})
    assert grouped.axes[1].get_yticklabels()[0].get_text() in ("red", "T")


def test_spanning_clones_plot():
    res = _light()
    fig = pl.spanning_clones(res, n_top=2, style="dark")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3  # two panels and the colour bar
    cid = res.clones.index[0]
    t = pd.DataFrame(
        {
            "clone_id": [cid],
            "n_units": [12],
            "spanning": [True],
            "axis_range": [0.5],
            "p_range": [0.2],
            "cross_um": [40.0],
            "p_cross": [0.01],
        }
    )
    fig = pl.spanning_clones(res, TestResult(name="spanning_clones", table=t), style="light")
    head = fig.axes[0].get_title(loc="left")
    assert "axis range 0.50 (p 0.2)" in head and "ends 40 µm (p 0.01)" in head
    with pytest.raises(KeyError):
        pl.spanning_clones(res, clones=["nope"])


def test_clone_state_map_follows_the_house_orientation():
    res = _light()
    ax = pl.clone_state_map(res, style="dark")
    assert ax.yaxis_inverted()
    assert ax.figure.get_dpi() >= 150  # a figure made here is slide-sized and sharp


def _legend_entries(ax):
    """{label: colour} of the legend above a map."""
    leg = ax.get_legend()
    return {
        t.get_text(): matplotlib.colors.to_hex(h.get_color())
        for h, t in zip(leg.legend_handles, leg.get_texts(), strict=True)
    }


@pytest.mark.parametrize("style", ["dark", "light"])
def test_outline_names_follow_the_context_names(style):
    """'White pulp' only where the context names a red pulp (a spleen); tissue-neutral names otherwise."""
    from spaclone.pl._colors import REGION_COLORS

    spleen = _light()
    assert spleen.context.params["naming"] == "spleen"  # the right half is erythroid: a red pulp
    assert "white pulp (B-rich + T zones)" in _legend_entries(pl.clone_state_map(spleen, style=style))
    node = _light(names="neutral")  # e.g. a lymph node: the area outside the B and T zones is not red pulp
    texts = list(_legend_entries(pl.clone_state_map(node, style=style)))
    assert "B-rich + T zones" in texts and "T zone" in texts
    assert not any("pulp" in t for t in texts)
    entries = _legend_entries(pl.clone_state_map(node, color="compartment", colorbar=False, style=style))
    third = [t for t in entries if t.startswith("non-B/T area")]
    assert third and not any("pulp" in t for t in entries)
    assert entries[third[0]] == REGION_COLORS[style]["red pulp"]  # same place, same colour under either name
    fig = pl.clone_axis_profile(node, n_bins=4, style=style)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "non-B/T area" in labels and labels == [c for c in ("T zone", "follicle", "non-B/T area") if c in labels]


def test_clone_axis_profile_without_compartments_keeps_the_expression_panel(caplog):
    import dataclasses

    base = _light()
    res = dataclasses.replace(base, units=base.units.drop(columns=["compartment"]))
    expr = pd.DataFrame({"Cd3e": np.arange(len(res.units), dtype=float)}, index=res.units.index)
    with caplog.at_level("INFO", logger="spaclone"):
        fig = pl.clone_axis_profile(res, n_bins=4, expression=expr)
    assert len(fig.axes) == 2  # the heatmap and its colour bar, no compartment panel
    assert "compartment" in caplog.text
    with pytest.raises(ValueError, match="nothing to draw"):
        pl.clone_axis_profile(res, n_bins=4)
