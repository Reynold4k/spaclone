"""Tests of spaclone.pl.spatial_records: highlighted categories survive the ``max_categories`` cut."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spaclone import pl
from spaclone.pl._colors import NEUTRAL, SLOTS


def _clones(seed: int = 0) -> pd.DataFrame:
    """Ten frequent clones (40 records each, ``f0``-``f9``) and two rare ones (``r1``: 3 records, ``r2``: 2)."""
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(10) for _ in range(40)] + ["r1"] * 3 + ["r2"] * 2
    n = len(names)
    return pd.DataFrame(
        {"x_um": rng.uniform(0, 400, n), "y_um": rng.uniform(0, 300, n), "locus": "IGH", "clone_id": names}
    )


def _legend(ax) -> dict[str, tuple[int, str]]:
    """``{label: (count, colour)}`` of the legend."""
    out = {}
    for h, t in zip(ax.get_legend().legend_handles, ax.get_legend().get_texts(), strict=True):
        name, count = t.get_text().rsplit(" (", 1)
        out[name] = (int(count.rstrip(")").replace(",", "")), mcolors.to_hex(h.get_markerfacecolor()))
    return out


def _layers(ax) -> list[tuple[int, str, float]]:
    """``(points, colour, marker area)`` of every scatter layer."""
    return [
        (len(c.get_offsets()), mcolors.to_hex(c.get_facecolors()[0]), float(c.get_sizes()[0]))
        for c in ax.collections
        if len(c.get_offsets())
    ]


@pytest.mark.parametrize("style", ["dark", "light"])
def test_highlighted_rare_categories_keep_their_own_colour(style):
    air = _clones()
    grey = NEUTRAL[style]
    ax = pl.spatial_records(air, color="clone_id", highlight=["r2", "f3", "r1"], background="none", style=style)
    leg = _legend(ax)
    # the two rare clones are outside the 8 most frequent values, yet each is drawn in its own slot colour
    assert set(leg) == {"f3", "r1", "r2"}
    assert {k: v[0] for k, v in leg.items()} == {"f3": 40, "r1": 3, "r2": 2}
    colours = [v[1] for v in leg.values()]
    assert len(set(colours)) == 3 and grey not in colours and set(colours) <= set(SLOTS[style])
    layers = _layers(ax)
    assert (len(air) - 45, grey) in [(n, c) for n, c, _ in layers]  # the rest in grey underneath
    for name in ("r1", "r2"):
        assert (leg[name][0], leg[name][1]) in [(n, c) for n, c, _ in layers]
    plt.close("all")


def test_without_highlight_the_most_frequent_values_keep_the_colours():
    ax = pl.spatial_records(_clones(), color="clone_id", background="none", max_categories=4)
    leg = _legend(ax)
    assert list(leg) == ["f0", "f1", "f2", "other"]  # ties broken by category order; r1 and r2 pooled
    assert leg["other"] == (7 * 40 + 5, NEUTRAL["dark"])
    plt.close("all")


def test_more_highlights_than_colours_pool_the_rarest_highlighted_values():
    air = _clones()
    ax = pl.spatial_records(
        air, color="clone_id", highlight=["f1", "f2", "r1", "r2"], max_categories=3, background="none"
    )
    leg = _legend(ax)
    # two colours for the most frequent highlighted values, the other highlighted values pooled as "other"
    assert list(leg) == ["f1", "f2", "other"] and leg["other"] == (5, NEUTRAL["dark"])
    assert (len(air) - 85, NEUTRAL["dark"]) in [(n, c) for n, c, _ in _layers(ax)]  # unhighlighted: grey, below
    plt.close("all")


def test_highlighted_records_stay_visible_over_many_grey_ones():
    air = _clones()
    ax = pl.spatial_records(air, color="clone_id", highlight=["r1", "r2"], background="none")
    sizes = {n: s for n, _, s in _layers(ax)}
    assert sizes[3] == sizes[2] > sizes[len(air) - 5]  # highlighted larger than the grey records
    plt.close("all")
    ax = pl.spatial_records(air, color="clone_id", highlight=["r1", "r2"], background="none", size=2.0)
    assert {s for _, _, s in _layers(ax)} == {2.0}  # an explicit size applies to every record
    plt.close("all")
    ax = pl.spatial_records(air, color="receptor", highlight=["BCR"], background="none")
    assert ax.get_legend() is None and _layers(ax)[0][:2] == (len(air), SLOTS["dark"][0])
    plt.close("all")
