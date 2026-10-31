"""Defaults of pl.clone_map: local links only, a fitted scale bar instead of axes, a visible tissue background."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

from spaclone import pl


def _section(width_um=3000.0, height_um=1500.0, seed=0):
    """Background units plus two clones: ``a`` in two tight groups 1.2 mm apart, ``b`` in one tight group."""
    rng = np.random.default_rng(seed)
    n = 3000
    bg = pd.DataFrame({"x_um": rng.uniform(0, width_um, n), "y_um": rng.uniform(0, height_um, n),
                       "clone_id": [f"bg{i}" for i in range(n)]})  # fmt: skip
    c = np.array([width_um / 2, height_um / 2])
    groups = [("a", c - [600.0, 0.0]), ("a", c + [600.0, 0.0]), ("b", c + [0.0, 300.0])]
    parts = [pd.DataFrame({"x_um": xy[0] + rng.uniform(-40, 40, 12), "y_um": xy[1] + rng.uniform(-40, 40, 12),
                           "clone_id": cid}) for cid, xy in groups]  # fmt: skip
    air = pd.concat([bg, *parts], ignore_index=True)
    air["spot_id"] = [f"s{i}" for i in range(len(air))]
    air["locus"] = "IGH"
    return air


def _segments(ax):
    lc = [c for c in ax.collections if isinstance(c, LineCollection)]
    return [np.asarray(s) for c in lc for s in c.get_segments()]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_clone_map_draws_local_links_by_default():
    """The 1.2 mm link that joins clone a's two groups is left out; max_link_um=None draws the whole tree."""
    air = _section()
    ax = pl.clone_map(air, ["a", "b"])
    seg = _segments(ax)
    lengths = np.array([np.hypot(*(s[0] - s[1])) for s in seg])
    assert len(seg) == (24 - 2) + (12 - 1)  # every tree link but the one between a's groups
    assert lengths.max() <= 250.0
    full = _segments(pl.clone_map(air, ["a", "b"], max_link_um=None))
    assert len(full) == (24 - 1) + (12 - 1)
    assert max(np.hypot(*(s[0] - s[1])) for s in full) > 1000.0


def test_clone_map_scale_bar_replaces_the_axes_and_fits_the_view():
    air = _section()
    ax = pl.clone_map(air, ["a", "b"])
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
    assert [t.get_text() for t in ax.texts] == ["500 µm"]
    bar = ax.lines[-1].get_xdata()
    assert bar[1] - bar[0] == pytest.approx(500.0)
    small = _section(width_um=600.0, height_um=400.0, seed=1)  # 500 µm would cover most of the view
    ax = pl.clone_map(small, ["b"])
    x0, x1 = ax.get_xlim()
    bar = ax.lines[-1].get_xdata()
    assert bar[1] - bar[0] <= 0.25 * (x1 - x0) and [t.get_text() for t in ax.texts] == ["100 µm"]
    ax = pl.clone_map(air, ["a"], scalebar_um=2000)  # longer than a quarter of the 3 mm view: 500 µm
    assert [t.get_text() for t in ax.texts] == ["500 µm"]
    ax = pl.clone_map(air, ["a"], scalebar_um=None)  # no bar: the axes stay
    assert len(ax.get_xticks()) > 0 and not ax.texts and ax.get_xlabel()


def test_clone_map_background_reads_and_own_figure_uses_its_width():
    air = _section()
    ax = pl.clone_map(air, ["a"])
    bg = ax.collections[0]
    assert bg.get_sizes()[0] == 2.0 and len(bg.get_offsets()) == len(air)
    assert ax.get_position().x0 < 0.06 and ax.get_position().x1 > 0.94  # no margin kept for hidden tick labels
    ax = pl.clone_map(air, ["a"], scalebar_um=None)
    assert ax.get_position().x0 > 0.1  # with axes, room for the tick labels
