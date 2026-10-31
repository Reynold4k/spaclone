import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spaclone import pl
from spaclone.pp import qc_metrics, read_support_curve


@pytest.fixture
def air():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "spot_id": rng.integers(0, 150, n).astype(str),
            "x_um": 0.0,
            "y_um": 0.0,
            "locus": rng.choice(["IGH", "IGK", "TRB"], n),
            "clone_id": rng.integers(0, 40, n).astype(str),
            "junction_aa": rng.choice(["CARDYW", "CQQYF", "CAR*W"], n),
            "productive": rng.choice(["T", "F"], n, p=[0.9, 0.1]),
            "consensus_count": np.r_[rng.geometric(0.7, n // 2), np.round(rng.lognormal(3, 0.4, n - n // 2))].astype(
                float
            ),
        }
    ).assign(x_um=lambda d: d.spot_id.astype(int) % 15 * 10.0, y_um=lambda d: d.spot_id.astype(int) // 15 * 10.0)


def test_each_plot_returns_axes(air):
    for fn, arg in (
        (pl.read_support, air),
        (pl.read_support, read_support_curve(air, by="locus")),
        (pl.valid_shares, air),
        (pl.valid_shares, qc_metrics(air, by="locus")),
        (pl.clones_per_spot, air),
        (pl.spot_qc_map, air),
    ):
        fig, ax = plt.subplots()
        out = fn(arg, ax=ax)
        assert out is ax
        plt.close(fig)
    ax = pl.spot_qc_map(qc_metrics(air), color="valid_share", rotate=False)
    assert ax.get_xlabel() == "x (µm)"
    ax = pl.spot_qc_map(qc_metrics(air), color="valid_share")  # default: long axis horizontal when elongated
    assert ax.get_xlabel() in ("x (µm)", "µm along the long axis")
    plt.close("all")
    with pytest.raises(KeyError):
        pl.spot_qc_map(air, color="nothing")


def test_qc_panel_returns_figure(air):
    fig = pl.qc_panel(air)
    assert len(fig.axes) >= 4
    plt.close(fig)
    fig = pl.qc_panel(air.drop(columns="consensus_count"))
    plt.close(fig)


def _contrast(a, b):
    """WCAG contrast ratio of two colours."""
    from matplotlib.colors import to_rgb

    def lum(c):
        rgb = np.array(to_rgb(c))
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(lin @ [0.2126, 0.7152, 0.0722])

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _lattice(nx=120, ny=50, pitch=10.0, seed=0):
    """Spot table of a lattice whose records follow a geometric law (most spots hold one record, as on real chips)."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    n = ii.size
    records = np.minimum(rng.geometric(0.55, n), 30).astype(float)
    records[-1] = 30.0  # one busy spot: counts span 1 to 30
    return pd.DataFrame(
        {
            "spot_id": [f"s{k}" for k in range(n)],
            "x_um": ii.ravel() * pitch,
            "y_um": jj.ravel() * pitch,
            "n_records": records,
        }
    )


@pytest.mark.parametrize("style", [None, "light", "dark"])
def test_spot_qc_map_low_end_stays_visible_on_its_surface(style):
    """The commonest value (one record) is drawn in a colour that stands out from the surface, in both styles."""
    from matplotlib.colors import to_hex

    from spaclone.pl._colors import CHROME

    sp = _lattice()
    ax = pl.spot_qc_map(sp, style=style, rotate=False)
    mode = "dark" if style == "dark" else "light"
    surface = CHROME[mode]["bg"]
    assert to_hex(ax.get_facecolor()) == surface
    sc = ax.collections[0]
    low = sc.cmap(sc.norm(1.0))  # the colour of a single-record spot
    assert _contrast(low, surface) >= 2.0
    assert sc.norm.vmin == 1.0 and sc.norm.vmax == sp.n_records.max()


def test_spot_qc_map_dark_style_paints_axes_figure_and_ink():
    from matplotlib.colors import to_hex

    from spaclone.pl._colors import CHROME

    fig, ax = plt.subplots()
    fig.set_facecolor("#ffffff")
    out = pl.spot_qc_map(_lattice(), ax=ax, style="dark")
    assert out is ax
    assert to_hex(ax.get_facecolor()) == to_hex(fig.get_facecolor()) == "#000000"
    ink2 = CHROME["dark"]["ink2"]
    assert to_hex(ax.xaxis.label.get_color()) == ink2 and to_hex(ax.yaxis.label.get_color()) == ink2
    cb = fig.axes[-1]
    assert cb.get_ylabel() == "n_records" and to_hex(cb.yaxis.label.get_color()) == ink2
    fig.canvas.draw()
    assert all(to_hex(t.get_color()) == ink2 for t in cb.get_yticklabels())
    light = pl.spot_qc_map(_lattice())  # the default stays white
    assert to_hex(light.get_facecolor()) == "#ffffff"
    with pytest.raises(ValueError, match="style"):
        pl.spot_qc_map(_lattice(), style="neon")
    plt.close("all")


def test_spot_qc_map_marker_is_as_wide_as_the_spot_pitch():
    """Spots tile the section: the default disc is as wide as the median nearest-neighbour spacing, on the axes."""
    for pitch in (10.0, 25.0):
        ax = pl.spot_qc_map(_lattice(pitch=pitch), rotate=False)
        ax.apply_aspect()
        x0, x1 = ax.get_xlim()
        pt_per_um = ax.get_position().width * ax.figure.get_figwidth() * 72.0 / abs(x1 - x0)
        diameter = float(np.sqrt(ax.collections[0].get_sizes()[0]))
        assert diameter == pytest.approx(pitch * pt_per_um, rel=1e-6)
    ax = pl.spot_qc_map(_lattice(), s=3.0)  # an explicit size is kept
    assert ax.collections[0].get_sizes()[0] == 3.0
    plt.close("all")


def test_spot_qc_map_colour_bar_reads_counts():
    """A log colour bar of counts is labelled 1, 2, 5, 10, 20 (not powers of ten only)."""
    ax = pl.spot_qc_map(_lattice(), rotate=False)
    fig = ax.figure
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.axes[-1].get_yticklabels() if t.get_text()]
    assert {"1", "2", "5", "10", "20"} <= set(labels)
    plt.close("all")
