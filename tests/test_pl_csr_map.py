"""Tests of spaclone.pl.csr_map and isotype_composition on small loci tables and the synthetic chip."""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import spaclone as sc
import spaclone.pl as pl
from spaclone.tl import Grid, Raster

FAST = {"figsize": (8.0, 4.5), "dpi": 50}
SUBCLASSES = ["IgM", "IgD", "IgG3", "IgG1", "IgG2b", "IgG2c", "IgA"]


def loci_table(n=1500, *, seed=0, subclasses=SUBCLASSES, p=None):
    """Loci on an elongated section; switched subclasses concentrate in two foci; clones of 1-4 loci."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1200, n)
    y = rng.uniform(0, 400, n)
    focus = ((x - 300) ** 2 + (y - 200) ** 2 < 80**2) | ((x - 900) ** 2 + (y - 200) ** 2 < 80**2)
    p = p or [0.55, 0.25, 0.02, 0.06, 0.05, 0.03, 0.04]
    pf = np.array([0.15, 0.05, 0.05, 0.3, 0.15, 0.1, 0.2])
    sub = np.where(
        focus,
        rng.choice(subclasses, n, p=pf / pf.sum()),
        rng.choice(subclasses, n, p=np.array(p) / np.sum(p)),
    )
    cls = pd.Series(sub).str.slice(0, 3).to_numpy()
    clone = np.repeat(np.arange(n), rng.integers(1, 5, n))[:n]
    return pd.DataFrame({"x_um": x, "y_um": y, "clone_id": [f"c{c}" for c in clone],
                         "isotype_class": pd.array(cls, dtype="string"),
                         "isotype_subclass": pd.array(sub, dtype="string"),
                         "switched": pd.array(np.isin(cls, ["IgG", "IgA", "IgE"]), dtype="boolean"),
                         "compartment": np.where(x < 600, "follicle", "red pulp")})  # fmt: skip


def _rows(ax):
    """Bar labels from top to bottom."""
    pairs = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()], strict=True), reverse=True)
    return [lab for _, lab in pairs]


def _titles(fig):
    """Map titles (the bold text of each title band), in panel order."""
    return [t.get_text() for ax in fig.axes for t in ax.texts if getattr(t, "_spaclone_right", None) is not None]


def test_csr_map_panels_follow_the_locus_order():
    loci = loci_table()
    fig = pl.csr_map(loci, sigma_um=40.0, n_perm=19, composition=None, **FAST)
    iso_titles = [t.split(" · ")[0] for t in _titles(fig) if t.startswith("Ig")]
    assert iso_titles == ["IgM", "IgD", "IgG3", "IgG1", "IgG2b", "IgG2a/c", "IgA"]
    # switched share, dominant class, 7 isotype maps, transitions panel
    assert len(fig.axes) == 2 + 7 + 1
    plt.close(fig)


def test_csr_map_levels_transitions_and_composition():
    loci = loci_table(seed=1)
    fig = pl.csr_map(loci, level="class", transitions="matrix", n_perm=19, sigma_um=40.0, **FAST)
    iso_titles = [t.split(" · ")[0] for t in _titles(fig) if t.startswith("Ig")]
    assert iso_titles == ["IgM", "IgD", "IgG", "IgA"]
    assert len(fig.axes) == 2 + 4 + 1 + 1  # + transitions matrix + composition bars (compartment column)
    plt.close(fig)
    fig = pl.csr_map(loci, transitions=None, composition=None, sigma_um=40.0, **FAST)
    assert len(fig.axes) == 2 + 7
    plt.close(fig)
    with pytest.raises(ValueError):
        pl.csr_map(loci, transitions="sankey", **FAST)
    with pytest.raises(ValueError):
        pl.csr_map(loci, level="gene", **FAST)


def test_csr_map_uses_a_given_transition_test(monkeypatch):
    loci = loci_table(seed=2)
    test = sc.bcr.csr_transitions(loci, isotype_key="isotype_subclass", unit_key=None, n_perm=19, rng=0)

    def boom(*a, **k):
        raise AssertionError("the given test must be drawn, not recomputed")

    monkeypatch.setattr(sc.bcr, "csr_transitions", boom)
    fig = pl.csr_map(loci, transition_test=test, composition=None, sigma_um=40.0, **FAST)
    plt.close(fig)


def test_isotype_colours_are_fixed_by_name():
    dark, light = pl.ISOTYPE_COLORS["dark"], pl.ISOTYPE_COLORS["light"]
    assert dark["IgM"] == "#7ecfff" and dark["IgG"] == "#ff3b3b" and dark["IgA"] == "#ffa31a"
    assert {k: dark[k] for k in ("IgM", "IgD", "IgG", "IgA", "IgE")} == {
        k: sc.pl.bcr.ISOTYPE_COLORS_DARK[k] for k in ("IgM", "IgD", "IgG", "IgA", "IgE")
    }
    assert set(dark) == set(light)
    a = pl.isotype_composition(loci_table(seed=3), by="compartment", **FAST)
    no_igm = loci_table(seed=3, subclasses=SUBCLASSES[1:] + ["IgM"], p=[0.25, 0.02, 0.06, 0.05, 0.03, 0.04, 0.0])
    no_igm = no_igm[no_igm.isotype_class != "IgM"]
    b = pl.isotype_composition(no_igm, by="compartment", **FAST)

    def colours(ax):
        return {c.get_label(): to_hex(c.patches[0].get_facecolor()) for c in ax.containers}

    ca, cb = colours(a), colours(b)
    assert "IgM" in ca and "IgM" not in cb
    for name in cb:  # the survivors keep their colour when IgM is absent
        assert cb[name] == ca[name] == dark[name if name != "IgG2a/c" else "IgG2c"]
    plt.close(a.figure)
    plt.close(b.figure)


def test_isotype_composition_groupings():
    loci = loci_table(seed=4)
    ax = pl.isotype_composition(loci, by="compartment", **FAST)
    assert _rows(ax) == ["follicle", "red pulp"]
    shares = np.add.reduce([np.array([p.get_width() for p in c.patches]) for c in ax.containers])
    assert np.allclose(shares, 1.0)
    plt.close(ax.figure)
    grid = Grid((0.0, 0.0), 10.0, (120, 40))
    codes = np.zeros(grid.shape, dtype=int)
    codes[60:] = 1
    ax = pl.isotype_composition(loci, by=Raster(codes, grid, categories=("left", "right")), level="class", **FAST)
    assert _rows(ax) == ["left", "right"]
    plt.close(ax.figure)
    with pytest.raises(KeyError):
        pl.isotype_composition(loci, by="no_such_column", **FAST)


def test_isotype_composition_by_region_of_a_workflow_result():
    loci = loci_table(seed=5)
    grid = Grid((0.0, 0.0), 10.0, (120, 40))
    gx, gy = grid.centres()
    lab = np.zeros(grid.shape, dtype=np.int64)
    lab[((gx[:, None] - 300) ** 2 + (gy[None, :] - 200) ** 2) < 80**2] = 1
    lab[((gx[:, None] - 900) ** 2 + (gy[None, :] - 200) ** 2) < 80**2] = 2
    csr = types.SimpleNamespace(labels=Raster(lab, grid), table=pd.DataFrame({"hotspot": [1, 2], "direction": "hot"}))
    regions = pd.DataFrame({"region_id": [1, 2], "csr_hotspots": [[1], [2]], "shm_hotspots": [[], []]})
    res = types.SimpleNamespace(loci=loci, grid=grid, tissue=Raster(np.ones(grid.shape, bool), grid), csr=csr,
                                regions=regions)  # fmt: skip
    ax = pl.isotype_composition(res, **FAST)
    assert _rows(ax) == ["region 1", "region 2", "outside regions"]
    plt.close(ax.figure)


def test_csr_map_from_receptor_records_of_the_synthetic_chip():
    sa = sc.sim.synthetic_chip(rng=0, n_sinks=0, tcr=False, genuine_fraction=0.9)
    fig = pl.csr_map(sa.air, n_perm=19, sigma_um=30.0, composition=None, **FAST)
    names = [t.split(" · ")[0] for t in _titles(fig) if t.startswith("Ig")]
    assert names[:2] == ["IgM", "IgD"] and "IgA" in names
    plt.close(fig)


def test_csr_map_is_deterministic():
    loci = loci_table(seed=6)
    a = pl.csr_map(loci, n_perm=19, rng=1, composition=None, sigma_um=40.0, **FAST)
    b = pl.csr_map(loci, n_perm=19, rng=1, composition=None, sigma_um=40.0, **FAST)
    a.canvas.draw()
    b.canvas.draw()
    assert np.array_equal(np.asarray(a.canvas.buffer_rgba()), np.asarray(b.canvas.buffer_rgba()))
    plt.close(a)
    plt.close(b)
