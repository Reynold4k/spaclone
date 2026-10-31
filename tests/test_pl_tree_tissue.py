"""pl.tree_tissue: the colour legend sits above the map, outside it, so it never covers a tip."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, pl


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _tree_and_tips():
    """A six-tip tree whose tips sit in every corner of an elongated section (where an in-map legend would go)."""
    trees = bcr.read_trees({"c1": "(((a:1,b:2):1,(c:1,d:2):1):1,(e:2,f:1):2,Germline:0);"})
    tips = pd.DataFrame(
        {
            "sequence_id": list("abcdef"),
            "x_um": [60.0, 1940.0, 60.0, 1940.0, 1000.0, 1900.0],
            "y_um": [40.0, 40.0, 560.0, 560.0, 300.0, 120.0],
            "isotype": ["IgM", "IgG1", "IgA", "IgG2b", "IgD", "IgE"],
        }
    )
    rng = np.random.default_rng(0)
    background = pd.DataFrame({"x_um": rng.uniform(0, 2000, 4000), "y_um": rng.uniform(0, 600, 4000)})
    return trees["c1"], tips, background


@pytest.mark.parametrize("style", ["dark", "light"])
@pytest.mark.parametrize("invert_y", [True, False])
def test_tree_tissue_legend_never_covers_a_tip(style, invert_y):
    tree, tips, background = _tree_and_tips()
    fig = pl.tree_tissue(tree, tips, background=background, style=style, invert_y=invert_y)
    ax_t, ax_m = fig.axes
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = ax_m.get_legend()
    box = leg.get_window_extent(r)
    assert box.y0 >= ax_m.get_window_extent(r).y1 - 1  # above the map
    assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1  # inside the figure
    tips_px = ax_m.collections[-1].get_offset_transform().transform(ax_m.collections[-1].get_offsets())
    assert len(tips_px) == len(tips)
    inside = (
        (tips_px[:, 0] >= box.x0) & (tips_px[:, 0] <= box.x1) & (tips_px[:, 1] >= box.y0) & (tips_px[:, 1] <= box.y1)
    )
    assert not inside.any()
    assert leg.get_title().get_text() == "isotype"
    assert [t.get_text() for t in leg.get_texts()] == list(tips["isotype"])
    assert not box.overlaps(ax_t.get_window_extent(r))
