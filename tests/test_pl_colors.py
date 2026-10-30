"""The palette (spaclone.pl._colors): validator floors for every categorical set, monotone ramps, one source."""

import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from spaclone.pl import _colors as C

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])  # fmt: skip
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])  # fmt: skip
_MACHADO = {  # Machado, Oliveira & Fernandes 2009, severity 1.0 (as the dataviz validator)
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
}  # fmt: skip
_SURFACE = {"dark": "#000000", "light": "#ffffff"}


def _lin(hexes):
    rgb = np.array([[int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes])
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _oklab(hexes, kind=None):
    lin = _lin(hexes)
    if kind is not None:
        lin = np.clip(lin @ _MACHADO[kind].T, 0, 1)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def _contrast(hexes, surface):
    y = _lin(hexes) @ [0.2126, 0.7152, 0.0722]
    ys = float((_lin([surface]) @ [0.2126, 0.7152, 0.0722])[0])
    hi, lo = np.maximum(y, ys), np.minimum(y, ys)
    return (hi + 0.05) / (lo + 0.05)


def _worst(cols, pairs, kind=None):
    lab = _oklab(cols, kind)
    return min(100 * float(np.linalg.norm(lab[i] - lab[j])) for i, j in pairs)


def _all_pairs(n):
    return list(itertools.combinations(range(n), 2))


_SETS = {
    "region": lambda m: list(C.REGION_COLORS[m].values()),
    "isotype classes": lambda m: [C.ISOTYPE_COLORS[m][k] for k in C.ISOTYPE_CLASSES],
    "hotspot marks": lambda m: [C.MARKS[m][k] for k in ("csr", "shm", "gcb")],
    "receptors": lambda m: list(C.RECEPTOR_COLORS[m].values()),
    "lineage locations": lambda m: [C.LOCATION_COLORS[m][k] for k in ("CSR hotspot", "follicle", "T zone", "red pulp")],
    "outline slots": lambda m: list(C.OUTLINE_SLOTS[m]),
    "first three slots": lambda m: list(C.SLOTS[m][:3]),
}


@pytest.mark.parametrize("mode", ["dark", "light"])
@pytest.mark.parametrize("name", sorted(_SETS))
def test_every_categorical_set_keeps_every_pair_apart(name, mode):
    """All-pairs floors of the dataviz validator: Delta E (OKLab x 100) >= 15 with full colour vision, >= 8 under
    protanopia and deuteranopia; every colour >= 3:1 on its surface (black or white) except the light slots.
    """
    cols = _SETS[name](mode)
    assert len(set(cols)) == len(cols)
    pairs = _all_pairs(len(cols))
    assert _worst(cols, pairs) >= 15.0, (name, mode)
    for kind in ("protan", "deutan"):
        assert _worst(cols, pairs, kind) >= 8.0, (name, mode, kind)
    if name != "first three slots":  # light aqua sits below 3:1 on white: categorical maps always carry a legend
        assert _contrast(cols, _SURFACE[mode]).min() >= 3.0, (name, mode)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_slots_keep_neighbours_apart(mode):
    cols = list(C.SLOTS[mode])
    pairs = [(i, i + 1) for i in range(len(cols) - 1)]
    assert _worst(cols, pairs) >= 15.0
    assert min(_worst(cols, pairs, k) for k in ("protan", "deutan")) >= 8.0
    assert C.REGION_COLORS[mode]["GC"] == C.SLOTS[mode][1]  # follicle / GC keep the first two slots
    assert C.REGION_COLORS[mode]["follicle"] == C.SLOTS[mode][0]


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_isotype_subclasses_are_an_ordinal_ramp(mode):
    """Inside IgG the subclasses step in lightness along the locus, light to dark, each step >= 8 under CVD."""
    pal = C.ISOTYPE_COLORS[mode]
    ladder = [pal[k] for k in ("IgG3", "IgG1", "IgG2b", "IgG2a/c")]
    L = _oklab(ladder)[:, 0]
    assert np.all(np.diff(L) < -0.06)
    for kind in ("protan", "deutan"):
        assert _worst(ladder, [(i, i + 1) for i in range(3)], kind) >= 8.0


@pytest.mark.parametrize("mode", ["dark", "light"])
@pytest.mark.parametrize("name", sorted(C.RAMPS))
def test_sequential_ramps_are_monotone(name, mode):
    """On black a ramp rises from the surface to a luminous end; on white it falls from near-white to a deep end."""
    L = _oklab(list(C.RAMPS[name][mode]))[:, 0]
    step = np.diff(L)
    assert np.all(step < 0) if mode == "light" else np.all(step > 0), (name, mode, L)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_diverging_has_a_grey_centre_and_symmetric_arms(mode):
    stops = list(C.DIVERGING[mode])
    lab = _oklab(stops)
    mid = len(stops) // 2
    assert np.hypot(*lab[mid, 1:]) < 0.01  # neutral grey, no hue at the midpoint
    left, right = lab[:mid, 0][::-1], lab[mid + 1 :, 0]
    assert np.allclose(left, right, atol=0.02)  # equal lightness steps per arm
    away = np.diff(np.r_[lab[mid, 0], right])
    assert np.all(away > 0) if mode == "dark" else np.all(away < 0)


def test_ink_on_fills():
    assert C.ink_on("#ffd23f") == C.CHROME["light"]["ink"]  # yellow takes dark text
    assert C.ink_on("#2a78d6") == "#ffffff"
    assert C.ink_on("#0d366b") == "#ffffff"


def test_every_plot_colour_comes_from_the_palette_module():
    """No plot module defines a colour of its own: hex literals live in pl/_colors.py only."""
    pl_dir = Path(C.__file__).parent
    hexes = re.compile(r"#[0-9a-fA-F]{6}\b")
    offenders = {
        p.name: hexes.findall(p.read_text(encoding="utf-8"))
        for p in sorted(pl_dir.glob("*.py"))
        if p.name != "_colors.py"
    }
    assert {k: v for k, v in offenders.items() if v} == {}


# ------------------------------------------------------------------------------- one orientation for every map


def test_section_frame_turns_the_long_axis_horizontal_and_starts_at_zero():
    from spaclone.pl._frame import SectionFrame

    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1000, 4000)
    xy = np.c_[200 + t * np.cos(0.6), 300 + t * np.sin(0.6)] + rng.normal(0, 40, (4000, 2))  # a tilted strip
    fr = SectionFrame(xy)
    p = fr(xy)
    assert fr.rotated and p[:, 0].std() > 5 * p[:, 1].std()
    assert np.allclose(p.min(axis=0), 0.0)
    assert np.allclose(np.linalg.norm(p[1:] - p[:-1], axis=1), np.linalg.norm(xy[1:] - xy[:-1], axis=1))  # rigid
    disc = rng.normal(0, 100, (2000, 2))  # a round section keeps its orientation
    assert not SectionFrame(disc).rotated and np.allclose(SectionFrame(disc)(disc), disc)
    assert not SectionFrame(xy, rotate=False).rotated
    q = SectionFrame(xy, rotate=90.0)(xy)
    assert np.allclose(np.ptp(q, axis=0), np.ptp(xy, axis=0)[::-1])


def test_every_map_rotates_an_elongated_section_and_draws_y_down():
    import matplotlib

    matplotlib.use("Agg")
    import anndata as ad
    import matplotlib.pyplot as plt
    import pandas as pd

    from spaclone import pl

    rng = np.random.default_rng(1)
    n = 3000
    t = rng.uniform(0, 1500, n)
    xy = np.c_[t * np.cos(0.5), t * np.sin(0.5)] + rng.normal(0, 60, (n, 2))
    air = pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "locus": "IGH", "spot_id": [f"s{i}" for i in range(n)],
                        "clone_id": rng.choice(list("abcde"), n), "isotype": rng.choice(["IgM", "IgG"], n)})  # fmt: skip
    gex = ad.AnnData(obs=pd.DataFrame({"cell_type": rng.choice(["B", "T"], n)}, index=air.spot_id.to_numpy()))
    gex.obsm["spatial"] = xy
    axes = [
        pl.spatial_records(air, background="none"),
        pl.gex_map(gex, "cell_type"),
        pl.clone_map(air),
        pl.isotype_map(air),
        pl.spot_qc_map(air),
    ]
    for ax in axes:
        assert ax.yaxis_inverted()
        fr = ax._spaclone_frame
        assert fr.rotated and abs(abs(np.degrees(fr.angle)) - np.degrees(0.5)) < 5
    plt.close("all")


def test_maps_frame_the_main_tissue_and_log_a_detached_speck(caplog):
    """A stray cluster of records far from the section neither stretches the view nor covers the tissue."""
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from spaclone import pl
    from spaclone.pl._frame import main_tissue

    rng = np.random.default_rng(3)
    sec = np.c_[rng.uniform(0, 4000, 12000), rng.uniform(0, 1500, 12000)]
    speck = np.c_[rng.normal(2500, 20, 60), rng.normal(6000, 20, 60)]  # 60 records 4.5 mm off the section
    xy = np.vstack([sec, speck])
    m = main_tissue(xy)
    assert m[:12000].all() and not m[12000:].any()
    island = np.c_[rng.uniform(4200, 4400, 300), rng.uniform(0, 300, 300)]  # 200 µm off the section's right end
    m = main_tissue(np.vstack([sec, island, speck]))
    assert m[:12300].all() and not m[12300:].any()  # an island of the section stays, the far speck goes
    lattice = np.stack(np.meshgrid(np.arange(0, 3000, 100.0), np.arange(0, 2000, 100.0)), -1).reshape(-1, 2)
    assert main_tissue(lattice).all()  # a 100 µm spot lattice (Visium) is one piece, not single spots
    assert main_tissue(xy[-40:]).all()  # too few points to crop
    air = pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "locus": "IGH", "spot_id": [f"s{i}" for i in range(len(xy))],
                        "clone_id": rng.choice(list("abc"), len(xy))})  # fmt: skip
    with caplog.at_level(logging.INFO, logger="spaclone"):
        ax = pl.clone_map(air)
    assert "60 of 12,060 points" in caplog.text and "crop=False" in caplog.text
    lo, hi = sorted(ax.get_ylim())
    assert hi - lo < 2000  # the 1.5 mm section, not the speck 6 mm away
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert ax.get_legend().get_window_extent(r).y0 >= ax.get_window_extent(r).y1 - 1  # legend above the map
    assert fig.get_figheight() < 0.75 * fig.get_figwidth()  # the figure follows the section
    plt.close(fig)
    lo, hi = sorted(pl.clone_map(air, crop=False).get_ylim())
    assert hi - lo > 5000
    for ax in (pl.spatial_records(air, background="none"), pl.isotype_map(air.assign(isotype="IgM")),
               pl.spot_qc_map(air)):  # fmt: skip
        lo, hi = sorted(ax.get_ylim())
        assert hi - lo < 2000
    plt.close("all")


def test_clone_map_ranks_heavy_chain_clones_by_default():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from spaclone import pl

    rng = np.random.default_rng(2)
    rows = [("IGK", "k1", 300), ("IGL", "l1", 250), ("IGH", "h1", 40), ("IGH", "h2", 30), ("IGH", "h3", 20)]
    air = pd.concat(
        [pd.DataFrame({"locus": lo, "clone_id": c, "x_um": rng.uniform(0, 500, m), "y_um": rng.uniform(0, 500, m)})
         for lo, c, m in rows], ignore_index=True,
    )  # fmt: skip
    air["spot_id"] = [f"s{i}" for i in range(len(air))]
    labels = [t.get_text().split(" · ")[0] for t in pl.clone_map(air).get_legend().get_texts()]
    assert labels == ["h1", "h2", "h3"]
    labels = [t.get_text().split(" · ")[0] for t in pl.clone_map(air, loci=None).get_legend().get_texts()]
    assert labels == ["k1", "l1", "h1"]
    plt.close("all")
