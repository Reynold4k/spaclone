"""Plot conventions shared by every spaclone plot: fixed colours by name, one y-axis convention, legends, pl.placebo."""

import itertools

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import bcr, nulls, pl, tl
from spaclone.pl._colors import SLOTS, locus_colors


def _hex(c):
    return mcolors.to_hex(c).lower()


def _legend_colours(ax):
    """{label without the count: colour} of a legend whose handles are markers, collections or lines."""
    leg = ax.get_legend()
    out = {}
    for h, t in zip(leg.legend_handles, leg.get_texts(), strict=True):
        name = t.get_text().split(" (")[0].split(" · ")[0].removeprefix("clone ")
        if hasattr(h, "get_markerfacecolor") and h.get_marker() not in (None, "None", ""):
            out[name] = _hex(h.get_markerfacecolor())
        elif hasattr(h, "get_facecolor"):
            out[name] = _hex(np.atleast_2d(h.get_facecolor())[0])
        else:
            out[name] = _hex(h.get_color())
    return out


def _cells(seed=0, n=600):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, size=(n, 2))
    ct = rng.choice(["B", "GC_B", "ASC", "T", "myeloid", "zeta", "alpha", "unassigned"], n)
    a = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(1, size=(n, 2)).astype(np.float32)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(ct)}, index=[f"u{i}" for i in range(n)]),
        var=pd.DataFrame(index=["Cd3e", "Ms4a1"]),
    )
    a.obsm["spatial"] = xy
    return a


def _records(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n)],
            "x_um": rng.uniform(0, 200, n),
            "y_um": rng.uniform(0, 200, n),
            "locus": rng.choice(["IGH", "IGK", "TRA", "TRB"], n),
            "truth": rng.choice(["genuine", "swapped", "sink"], n, p=[0.3, 0.6, 0.1]),
            "clone_id": rng.integers(0, 50, n).astype(str),
            "isotype": rng.choice(["IgM", "IgG", "IgA"], n),
        }
    )


# ------------------------------------------------------------------------------------- (a) colours follow the name


def test_truth_and_locus_colours_do_not_depend_on_the_values_present():
    air = _records()
    full = _legend_colours(pl.spatial_records(air, color="truth", background="none", style="light"))
    swapped_only = _legend_colours(
        pl.spatial_records(air[air.truth != "genuine"], color="truth", background="none", style="light")
    )
    assert full["genuine"] == SLOTS["light"][0] and full["swapped"] == SLOTS["light"][1]
    assert swapped_only["swapped"] == full["swapped"] and swapped_only["sink"] == full["sink"]
    tcr = _legend_colours(pl.spatial_records(air[air.locus == "TRB"].head(5).assign(
        locus=["TRB", "TRB", "TRA", "TRB", "TRA"]), color="locus", background="none", style="dark"))  # fmt: skip
    assert tcr == {"TRA": locus_colors("dark")["TRA"], "TRB": locus_colors("dark")["TRB"]}
    assert pl.LOCUS_COLORS == locus_colors("light")  # one chain colour across pl.qc, records, molecule plots
    plt.close("all")


def test_other_columns_take_slots_by_category_and_subsets_keep_them():
    air = _records().assign(grp=lambda d: np.where(d.x_um < 100, "b", "c"))
    air.loc[air.index[:20], "grp"] = "a"
    full = _legend_colours(pl.spatial_records(air, color="grp", background="none", style="light"))
    assert full == {"a": SLOTS["light"][0], "b": SLOTS["light"][1], "c": SLOTS["light"][2]}  # sorted values
    cat = air.assign(grp=pd.Categorical(air.grp, categories=["c", "b", "a"]))
    whole = _legend_colours(pl.spatial_records(cat, color="grp", background="none", style="light"))
    part = _legend_colours(pl.spatial_records(cat[cat.grp != "c"], color="grp", background="none", style="light"))
    assert whole["c"] == SLOTS["light"][0] and part == {"b": whole["b"], "a": whole["a"]}  # declared categories
    own = _legend_colours(
        pl.spatial_records(air, color="grp", background="none", palette={"a": "#000000"}, style="light")
    )
    assert own == full | {"a": "#000000"}
    plt.close("all")


def test_gex_map_colours_are_fixed_by_class_name():
    a = _cells()
    full = _legend_colours(pl.gex_map(a, "cell_type", style="light"))
    masked = _legend_colours(
        pl.gex_map(a, "cell_type", mask=~a.obs.cell_type.isin(["alpha", "B"]).to_numpy(), style="light")
    )
    assert full["B"] == pl.HOUSE_COLORS["B"][1] and full["myeloid"] == pl.HOUSE_COLORS["myeloid"][1]
    assert masked["zeta"] == full["zeta"] and "alpha" not in masked  # classes without a house colour keep theirs
    assert list(full)[-1] == "unassigned"  # neutral classes are listed last
    plt.close("all")


# ------------------------------------------------------------------------------- (b) folded classes appear once


def test_gex_map_groups_list_and_draw_a_folded_class_once():
    a = _cells()
    ax = pl.gex_map(a, "cell_type", groups={"B lineage": ["B", "GC_B", "ASC"]})
    labels = [t.get_text().split(" (")[0] for t in ax.get_legend().get_texts()]
    assert labels.count("B lineage") == 1 and not {"B", "GC_B", "ASC"} & set(labels)
    assert len(labels) == len(set(labels))
    n_lineage = int(a.obs.cell_type.isin(["B", "GC_B", "ASC"]).sum())
    assert f"B lineage ({n_lineage:,})" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(ax.collections) == len(labels)  # one scatter per class, none drawn twice
    plt.close("all")


# --------------------------------------------------------------------------- (c) germinal centre vs ASC colours


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


def _oklab(hexes, kind=None):
    rgb = np.array([[int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    if kind is not None:
        lin = np.clip(lin @ _MACHADO[kind].T, 0, 1)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def test_region_palette_keeps_every_pair_apart_in_both_modes():
    """The validator's all-pairs floors: Delta E (OKLab x 100) >= 15 with full colour vision, >= 8 under protanopia
    and deuteranopia, for every pair of region colours, in both modes.
    """
    from spaclone.pl._colors import REGION_COLORS

    for mode in ("dark", "light"):
        cols = list(REGION_COLORS[mode].values())
        assert len(set(cols)) == 5
        for kind, floor in ((None, 15.0), ("protan", 8.0), ("deutan", 8.0)):
            lab = _oklab(cols, kind)
            d = 100 * np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
            assert d[np.triu_indices(len(cols), 1)].min() >= floor, (mode, kind)
    assert REGION_COLORS["dark"]["GC"] == SLOTS["dark"][1] and REGION_COLORS["light"]["GC"] == SLOTS["light"][1]


@pytest.mark.parametrize("style", ["dark", None])
def test_gex_regions_use_the_region_palette_and_cell_maps_keep_theirs(style):
    from spaclone.pl._colors import CELL_COLORS, REGION_COLORS

    mode = "dark" if style == "dark" else "light"
    a = _cells()
    a.obs["region"] = pd.Categorical(np.resize(["GC", "ASC", "T zone", "follicle", "red pulp", "other"], a.n_obs))
    a.obs["region_id"] = np.nan
    cols = _legend_colours(pl.gex_regions(a, style=style))
    assert {k: cols[k] for k in REGION_COLORS[mode]} == REGION_COLORS[mode]
    cells = _legend_colours(pl.gex_map(a, "cell_type", style=style))  # the ASC cell class stays amber
    assert cells["ASC"] == CELL_COLORS["ASC"][0 if mode == "dark" else 1] != cols["ASC"]
    plt.close("all")


# ------------------------------------------------------------------------------------ (d) one y-axis convention


def _tree_and_tips():
    trees = bcr.read_trees({"c1": "((a:1,b:2):1,c:3,Germline:0);"})
    tips = pd.DataFrame(
        {"sequence_id": ["a", "b", "c"], "x_um": [0.0, 50.0, 100.0], "y_um": [0.0, 80.0, 20.0],
         "isotype": ["IgM", "IgG", "IgG"]}
    )  # fmt: skip
    return trees["c1"], tips


def _hotspot_result():
    rng = np.random.default_rng(3)
    n = 900
    t = pd.DataFrame({"x_um": rng.uniform(0, 300, n), "y_um": rng.uniform(0, 300, n)})
    t["clone_id"] = (np.arange(n) // 3).astype(str)  # three units per clone
    hot = np.hypot(t.x_um - 80, t.y_um - 80) < 50
    t["mark"] = rng.random(n) < np.where(hot, 0.7, 0.1)
    return t, tl.hotspots(t, mark="mark", sigma_um=25.0, n_perm=19, blocks="clone_id", min_neff=3, rng=1)


def test_every_spatial_map_draws_y_downwards_unless_asked():
    air = _records()
    a = _cells()
    a.obs["region"] = pd.Categorical(np.resize(["GC", "ASC"], a.n_obs))
    tree, tips = _tree_and_tips()
    _, hs = _hotspot_result()
    loci = air.assign(mutated=air.index % 3 == 0, switched=air.isotype != "IgM")
    for flag in (True, False):
        axes = [
            pl.spatial_records(air, background="none", invert_y=flag),
            pl.gex_map(a, "cell_type", invert_y=flag),
            pl.gex_map(a, "Cd3e", invert_y=flag),
            pl.gex_regions(a, instance_key=None, invert_y=flag),
            pl.isotype_map(air, invert_y=flag),
            pl.share_map(hs, invert_y=flag),
            pl.csr_share_map(hs, units=loci, invert_y=flag),
            pl.shm_share_map(hs, units=loci, invert_y=flag),
            pl.spot_qc_map(air, invert_y=flag),
            pl.tree_tissue(tree, tips, invert_y=flag).axes[1],
            pl.qc_panel(air.drop(columns="truth"), invert_y=flag).axes[3],
            pl.clone_map(air, invert_y=flag, scalebar_um=50),
        ]
        assert [ax.yaxis_inverted() for ax in axes] == [flag] * len(axes)
        plt.close("all")


def test_clone_map_colours_follow_the_named_order_and_never_cycle():
    air = _records(n=600, seed=4).assign(clone_id=lambda d: np.resize(list("abcdefghij"), len(d)))
    ax = pl.clone_map(air, ["c", "a"], background=False, style="light")
    assert _legend_colours(ax) == {"c": SLOTS["light"][0], "a": SLOTS["light"][1]}
    assert ax.get_legend().get_title().get_text() == "" and ax.yaxis_inverted()
    own = _legend_colours(pl.clone_map(air, ["a"], palette={"a": "#000000"}, style="dark"))
    assert own == {"a": "#000000"}
    ten = _legend_colours(pl.clone_map(air, list("abcdefghij"), style="light"))  # ten clones: eight slots, then grey
    assert len({ten[c] for c in "abcdefgh"}) == 8 and ten["i"] == ten["j"] == pl.clones.STYLES["light"]["muted"]
    top = _legend_colours(
        pl.clone_map(air, style="light")
    )  # default: the three largest clones, in the all-pairs-safe slots
    assert list(top.values()) == list(SLOTS["light"][:3])
    plt.close("all")


# ------------------------------------------------------------------------- (e) titles are not repeated in legends


def test_plot_title_is_not_repeated_as_legend_or_colourbar_title():
    a = _cells()
    ax = pl.gex_map(a, "cell_type")
    assert ax.get_title(loc="left") == "cell_type" and ax.get_legend().get_title().get_text() == ""
    ax = pl.gex_map(a, "cell_type", title="")
    assert ax.get_legend().get_title().get_text() == "cell_type"
    ax = pl.gex_map(a, "Cd3e", title="Cd3e (log-normalised)")
    assert ax.figure.axes[-1].get_ylabel() == ""  # the colour bar does not repeat the gene name
    ax = pl.gex_regions(a.copy(), region_key="cell_type", instance_key=None)
    assert ax.get_legend().get_title().get_text() == ""
    plt.close("all")


# --------------------------------------------------------------------------- (f) knee / inflection labels in qc_panel


def test_qc_panel_text_never_overlaps():
    rng = np.random.default_rng(0)
    n = 3000
    air = pd.DataFrame(
        {
            "spot_id": rng.integers(0, 800, n).astype(str),
            "locus": rng.choice(["IGH", "IGK", "IGL", "TRA", "TRB"], n),
            "clone_id": rng.integers(0, 300, n).astype(str),
            "junction_aa": "CARDYW",
            "productive": True,
            "consensus_count": np.r_[rng.geometric(0.6, n // 2), np.round(rng.lognormal(2.5, 0.5, n - n // 2))],
        }
    ).assign(x_um=lambda d: d.spot_id.astype(int) % 40 * 10.0, y_um=lambda d: d.spot_id.astype(int) // 40 * 10.0)
    fig = pl.qc_panel(air)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes[:2]:  # read support, valid shares
        texts = [t for t in ax.texts if t.get_text()]
        leg = ax.get_legend()
        texts += [t for t in (leg.get_texts() if leg else []) if t.get_text()]
        boxes = [t.get_window_extent(r) for t in texts]
        for b1, b2 in itertools.combinations(boxes, 2):
            assert not b1.overlaps(b2)
        if leg is not None and ax is not fig.axes[0]:  # the valid-share key sits above the bars
            assert leg.get_window_extent(r).y0 >= ax.get_window_extent(r).y1 - 1
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(t.startswith("IGH: knee") and "inflection" in t for t in labels)
    plt.close("all")


# ------------------------------------------------------------------------------------------------- (g) pl.placebo


def _placebo_results():
    rng = np.random.default_rng(1)
    air = pd.DataFrame({"x_um": rng.uniform(0, 100, 300), "clone_id": rng.integers(0, 30, 300).astype(str)})
    air.loc[air.clone_id == "0", "x_um"] = rng.uniform(0, 5, int((air.clone_id == "0").sum()))

    def spread(d):
        return float(d[d.clone_id == "0"].x_um.std())

    def vec(d):
        return {"spread": spread(d), "n_clones": float(d.clone_id.nunique()), "mean_x": float(d.x_um.mean())}

    design = nulls.Permute(["clone_id"])
    one = nulls.placebo(spread, air, design, n_perm=99, alternative="less", rng=0)
    many = nulls.placebo(vec, air, design, n_perm=99, alternative="less", rng=0)
    return one, many, air, design, spread


def test_placebo_histogram_shows_the_null_the_observed_value_and_p():
    one, many, air, design, spread = _placebo_results()
    ax = pl.placebo(one)
    assert sum(p.get_height() for p in ax.patches) == one.null.shape[0]  # every placebo draw in a bar
    obs_line = [ln for ln in ax.lines if np.allclose(ln.get_xdata(), one.observed[0])]
    assert obs_line  # a vertical line at the observed value
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert f"p = {one.p_value:.3g}" in labels and "(less)" in labels and "99 draws" in labels
    assert ax.get_xlabel() == "spread" and ax.get_ylabel() == "placebo data sets"
    ax2 = pl.placebo(many, component="n_clones")  # integer draws: one bar per value
    assert pl.placebo(many, component=1).get_xlabel() == ax2.get_xlabel() == "n_clones"
    plt.close("all")


def test_placebo_strips_standardise_each_component():
    one, many, *_ = _placebo_results()
    fig, ax = plt.subplots()
    assert pl.placebo(many, ax=ax, style="dark") is ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["spread", "n_clones", "mean_x"]
    notes = [t.get_text() for t in ax.texts]
    assert notes[0].startswith("p = ") and notes[1] == notes[2] == "no placebo variance"  # constant under shuffles
    assert _hex(fig.get_facecolor()) == "#ffffff"  # a figure the caller owns is not repainted
    with pytest.raises(ValueError, match="component"):
        pl.placebo(many, kind="hist")
    with pytest.raises(KeyError):
        pl.placebo(many, component="nope")
    with pytest.raises(ValueError, match="keep_null"):
        pl.placebo(nulls.placebo(lambda d: 1.0, _placebo_results()[2], nulls.Permute(["clone_id"]), n_perm=5,
                                 keep_null=False, rng=0))  # fmt: skip
    plt.close("all")


# ---------------------------------------------------------------------------- (h) permutation-studentised z maps


def test_share_maps_draw_the_permutation_z_map_by_default():
    t, hs = _hotspot_result()
    assert hs.params["standardise"] == "permutation"
    ax = pl.share_map(hs)
    im = ax.images[-1]
    img = np.asarray(im.get_array(), dtype=float).transpose(1, 0, 2)  # RGBA on the grid
    valid = hs.map.eligible & np.isfinite(hs.z_map.values)
    np.testing.assert_allclose(img[valid], im.get_cmap()(im.norm(hs.z_map.values[valid])))
    tissue = pl.maps._display_tissue(hs.map)
    assert valid.any() and (img[tissue][:, 3] == 1).all()  # the map covers the whole tissue: no holes
    assert ax.figure.axes[-1].get_ylabel() == "z (permutation-studentised)"
    ax = pl.share_map(hs, layer="share")
    im = ax.images[-1]
    img = np.asarray(im.get_array(), dtype=float).transpose(1, 0, 2)
    elig = hs.map.eligible
    np.testing.assert_allclose(img[elig], im.get_cmap()(im.norm(hs.map.share[elig])))
    for fn in (pl.csr_share_map, pl.shm_share_map):
        ax = fn(hs)
        assert ax.images[-1].get_cmap().name.startswith("spaclone_diverging")
        assert fn(hs, layer="share").images[-1].get_cmap().name == "spaclone_bcr"
    binomial = tl.hotspots(t, mark="mark", sigma_um=25.0, n_perm=19, min_neff=3, rng=1)
    assert binomial.params["standardise"] == "binomial"
    assert pl.share_map(binomial).figure.axes[-1].get_ylabel() == "share (smoothed)"
    with pytest.raises(ValueError, match="z_perm"):
        pl.share_map(hs.map, layer="z_perm")
    plt.close("all")


# ------------------------------------------------------------------------------- other categorical colour maps


def test_placement_and_spectratype_colours_follow_the_level():
    from spaclone._utils import TestResult

    tab = pd.DataFrame(
        {
            "locus": ["IGH", "IGH", "IGK", "IGK"],
            "band": ["0-20", "20-100", "0-20", "20-100"],
            "statistic": [0.1, 0.2, 0.15, 0.25],
            "null_mean": [0.1, 0.18, 0.1, 0.2],
            "null_lo": [0.05, 0.1, 0.05, 0.1],
            "null_hi": [0.15, 0.25, 0.15, 0.3],
        }
    )
    res = TestResult(name="colocalisation", table=tab)
    full = _legend_colours(pl.placement(res, x="locus", hue="band", style="light"))
    part = _legend_colours(
        pl.placement(res, x="locus", hue="band", query="band == '20-100'", style="light")
    )  # one level left
    assert full["band = 20-100"] == part["band = 20-100"] == SLOTS["light"][1]
    loci = _legend_colours(pl.placement(res.table, x="band", hue="locus", style="light"))
    assert loci == {
        "locus = IGH": SLOTS["light"][0],
        "locus = IGK": SLOTS["light"][1],
        "observed": loci["observed"],
        "null mean and 95% interval": loci["null mean and 95% interval"],
    }
    spec = pd.DataFrame({"region": np.repeat(["GC", "ASC", "zone 9"], 3), "length": np.tile([12, 13, 14], 3),
                         "count": 1, "share": 1 / 3})  # fmt: skip
    a = {ln.get_label(): _hex(ln.get_color()) for ln in pl.spectratype(spec).lines}
    b = {ln.get_label(): _hex(ln.get_color()) for ln in pl.spectratype(spec[spec.region != "GC"]).lines}
    assert a["GC"] == pl.HOUSE_COLORS["GC"][1] and a["ASC"] == b["ASC"] == pl.HOUSE_COLORS["ASC"][1]
    regions = spec.assign(region=spec.region.replace({"zone 9": "follicle"}))  # a region vocabulary
    r = {ln.get_label(): _hex(ln.get_color()) for ln in pl.spectratype(regions).lines}
    from spaclone.pl._colors import REGION_COLORS

    assert r == {k: REGION_COLORS["light"][k] for k in ("GC", "ASC", "follicle")}
    plt.close("all")


def test_trust_report_compact_card_lists_one_row_per_test():
    class _Report:
        verdict = "Clone positions supported."
        status = {"overall": "supported", "molecule": "limited", "clone": "signal"}

        @staticmethod
        def tests():
            return pd.DataFrame(
                {
                    "test": ["own_type_share", "colocalisation", "colocalisation", "pairing"],
                    "locus": ["BCR", "IGH", "IGK", "IGH/IGK"],
                    "library_id": "D21",
                    "ratio": [1.1, 21.3, 12.4, 3.6],
                    "q_value": [0.3, 0.008, np.nan, 0.008],
                    "verdict": ["not detected", "signal", "info", "signal"],
                }
            )

    for style in ("dark", "light"):
        fig = pl.trust_report(_Report(), style=style, compact=True, width=6)
        texts = [t.get_text() for t in fig.texts]
        assert {"limited", "signal"} <= set(texts) and "21×" in texts and "12×" not in texts  # heavy chain row only
        assert sum(t.startswith("q ") for t in texts) == 3
        plt.close(fig)
