"""Tests of spaclone.pl.csr_flow: flow fields on tissue, flux matrices, stationary distributions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spaclone import pl, tl


@pytest.fixture(scope="module")
def model_and_flux():
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(10) * 10.0, np.arange(10) * 10.0)
    u = pd.DataFrame({"x_um": xs.ravel(), "y_um": ys.ravel()})
    u["t"] = u.x_um / 90 + rng.normal(0, 0.05, len(u))
    u["region"] = np.where(u.x_um < 30, "follicle", np.where(u.x_um < 60, "GC", "red pulp"))
    m = tl.transition_model(u, time_key="t", k=6, states="region")
    res = tl.tpt_flux(m, source="follicle", target="red pulp", n_random=9, rng=0)
    return m, res


@pytest.mark.parametrize("kind", ["stream", "grid", "units"])
def test_csr_flow_kinds(model_and_flux, kind):
    m, _ = model_and_flux
    ax = pl.csr_flow(m, kind=kind, style="dark")
    assert ax.get_xlabel() == "" and not ax.get_xticks().size  # a tissue map: no axes, a scale bar
    assert ax._spaclone_frame is not None
    lo, hi = ax.get_ylim()
    assert lo > hi  # image convention by default: y increases downwards
    assert ax.figure.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    plt.close(ax.figure)


def test_csr_flow_colour_by_committor_and_table_input(model_and_flux):
    m, res = model_and_flux
    ax = pl.csr_flow(m, color=res.units.q_plus.to_numpy(), label="forward committor", style=None, invert_y=False)
    lo, hi = ax.get_ylim()
    assert lo < hi
    plt.close(ax.figure)
    d = tl.transition_displacement(m)
    tab = d.assign(csr_pot=d.time)
    fig, axs = plt.subplots(1, 2)
    out = pl.csr_flow(tab, ax=axs[0], kind="grid", colorbar=False, title="CSR potential")
    assert out is axs[0] and out.get_title(loc="left") == "CSR potential"
    with pytest.raises(ValueError, match="kind"):
        pl.csr_flow(m, kind="spiral", ax=axs[1])
    plt.close(fig)


def test_flux_matrix_and_stationary(model_and_flux):
    _, res = model_and_flux
    ax = pl.flux_matrix(res, value="net", order=["follicle", "GC", "red pulp"], mask_improbable=True)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["follicle", "GC", "red pulp"]
    plt.close(ax.figure)
    ax = pl.flux_matrix(res, value="gross", style="dark", use="p_value")
    plt.close(ax.figure)
    ax = pl.stationary_distribution(res, style="dark")
    assert [t.get_text() for t in ax.get_xticklabels()] == list(res.states.index)
    plt.close(ax.figure)
    nonull = tl.tpt_flux(model_and_flux[0], source="follicle", target="red pulp", n_random=0)
    ax = pl.flux_matrix(nonull)
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="order"):
        pl.flux_matrix(res, order=["follicle", "mantle"])


def test_stationary_distribution_of_a_transition_model(model_and_flux):
    m, res = model_and_flux
    ax = pl.stationary_distribution(m, style="dark")  # the chain's own stationary distribution, lumped by state
    names = [t.get_text() for t in ax.get_xticklabels()]
    heights = dict(zip(names, (b.get_height() for b in ax.patches), strict=True))
    pi = pd.Series(m.pi).groupby(np.asarray(m.states.astype(str))).sum()
    assert set(names) == set(pi.index) and np.allclose([heights[k] for k in pi.index], pi.to_numpy())
    np.testing.assert_allclose([heights[k] for k in res.states.index], res.states.stationary, atol=1e-9)
    plt.close(ax.figure)
    with pytest.raises(TypeError, match="tl.tpt_flux"):
        pl.stationary_distribution(pd.DataFrame({"a": [1.0]}))


def test_csr_flow_labels_the_csr_potential_and_keeps_streamlines_on_the_tissue(model_and_flux):
    m, _ = model_and_flux
    d = tl.transition_displacement(m)
    tab = d.assign(csr_pot=d.time)
    ax = pl.csr_flow(tab, kind="stream")  # colour bar label of the time key csr_pot
    assert "CSR potential" in [t.get_text() for a in ax.child_axes for t in a.texts]  # the inset colour bar
    plt.close(ax.figure)
    hole = tab[~((tab.x_um > 25) & (tab.x_um < 65) & (tab.y_um > 25) & (tab.y_um < 65))]  # no tissue in the middle
    ax = pl.csr_flow(hole, kind="stream", n_grid=10, s=0)
    from matplotlib.collections import LineCollection

    seg = np.concatenate([np.asarray(c.get_segments()).reshape(-1, 2) for c in ax.collections
                          if isinstance(c, LineCollection)])  # fmt: skip
    assert len(seg) and not ((seg[:, 0] > 37) & (seg[:, 0] < 53) & (seg[:, 1] > 37) & (seg[:, 1] < 53)).any()
    plt.close(ax.figure)


def test_csr_flow_draws_region_labels_with_a_legend(model_and_flux):
    m, _ = model_and_flux
    regions = np.asarray(m.states.astype(str))
    ax = pl.csr_flow(m, color=regions, kind="stream")
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts and all("(" in t for t in texts)
    plt.close(ax.figure)
