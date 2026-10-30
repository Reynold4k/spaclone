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
