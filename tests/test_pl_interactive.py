"""Interactive figures: encodings, structure, determinism, selection numbers, notebook display, robust inputs."""

import base64
import json
import re
import shutil
import subprocess
import warnings
import zlib
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import spaclone
from spaclone import bcr, pl
from spaclone.pl import interactive as I
from spaclone.pl.interactive import _encode as E
from spaclone.pl.interactive import _html

# ------------------------------------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def syn():
    return spaclone.datasets.synthetic_example(sequences=True)


@pytest.fixture(scope="module")
def hot(syn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bcr.hotspot_workflow(syn)


def _records(n=500, seed=0, **extra):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n)],
            "x_um": rng.uniform(0, 300, n),
            "y_um": rng.uniform(0, 150, n),
            "locus": rng.choice(["IGH", "IGK", "TRB"], n),
            "clone_id": rng.choice([f"c{i}" for i in range(40)], n),
            "c_call": rng.choice(["IGHM", "IGHG1", "IGHA", "IGKC", None], n),
            "consensus_count": rng.integers(1, 20, n).astype(float),
        }
    )
    for k, v in extra.items():
        d[k] = v
    return d


def _units(n=800, seed=0, genes=("Cd3e", "Ms4a1", "Jchain"), dense=False):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.2, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(
        X=X if dense else sparse.csr_matrix(X),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(["B", "T", "ASC"], n)),
                "total_counts": X.sum(1) + 5,
                "in_tissue": True,
            },  # fmt: skip
            index=[f"u{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=list(genes)),
    )
    a.obsm["spatial"] = rng.uniform(0, 300, size=(n, 2))
    return a


def _blob_ids(spec):
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and re.fullmatch(r"b\d+", o):
            out.add(o)

    walk({k: v for k, v in spec.items() if k not in ("blobs", "tables")})
    return out


def _check_spec(fig):
    """Every referenced block exists, layer ids are unique, the spec is JSON without NaN."""
    d = fig.spec
    refs = _blob_ids(d)
    assert refs <= set(d["blobs"]), refs - set(d["blobs"])
    ids = [L["id"] for p in d["panels"] for L in p["layers"]]
    assert len(ids) == len(set(ids))
    json.dumps(_html._clean(d), allow_nan=False, default=_html._default)
    return d


# ------------------------------------------------------------------------------------------------ encodings


@pytest.mark.parametrize("dtype", ["u1", "u2", "u4", "i1", "i2", "i4", "f4", "f8"])
@pytest.mark.parametrize("compress", [False, True])
def test_blocks_round_trip_exactly(dtype, compress):
    rng = np.random.default_rng(1)
    info = np.iinfo(np.dtype(E.DTYPES[dtype])) if dtype[0] in "ui" else None
    if info is not None:
        v = rng.integers(info.min, int(info.max) + 1, 5000, dtype=np.int64)
    else:
        v = rng.normal(0, 1e3, 5000)
        v[::97] = np.nan
    blk = E.encode_array(v, dtype, compress=compress)
    back = E.decode(blk)
    want = np.asarray(v, dtype=np.dtype(E.DTYPES[dtype]))
    np.testing.assert_array_equal(back, want)
    if compress:
        assert blk.get("z") == 1 or len(base64.b64decode(blk["d"])) == want.nbytes


def test_quantised_blocks_stay_within_half_a_step():
    v = np.r_[np.linspace(-3, 7, 10001), np.nan, np.inf]
    blk = E.encode_quantised(v, bits=16)
    back = E.decode(blk).astype(float)
    step = blk["q"][1]
    fin = np.isfinite(v)
    assert np.all(np.abs(back[fin] - v[fin]) <= step / 2 + 1e-6)
    assert np.isnan(back[~fin]).all()
    const = E.decode(E.encode_quantised(np.full(10, 2.5)))
    np.testing.assert_array_equal(const, np.full(10, 2.5, dtype=np.float32))


def test_varints_match_leb128_and_round_trip():
    assert E.varint_bytes([0, 127, 128, 300]) == bytes([0x00, 0x7F, 0x80, 0x01, 0xAC, 0x02])
    idx = np.sort(np.random.default_rng(2).choice(10**7, 2000, replace=False))
    blk = E.encode_varints(idx, delta=True)
    np.testing.assert_array_equal(E.decode(blk), idx)
    with pytest.raises(ValueError):
        E.encode_varints([3, 1], delta=True)


def test_integer_blocks_refuse_values_that_do_not_fit():
    with pytest.raises(ValueError):
        E.encode_array([256], "u1")
    with pytest.raises(ValueError):
        E.encode_array([np.nan], "u2")


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node.js not found: the engine's decoders are tested in the browser suite")
def test_engine_decoders_agree_with_python(tmp_path):
    """The engine's own inflate, varints, base64 and polygon test, run by node.js on data written by Python."""
    rng = np.random.default_rng(3)
    cases = []
    for k, n in enumerate((0, 1, 7, 1000, 70000)):
        raw = rng.integers(0, 4 if k % 2 else 256, n, dtype=np.uint8).tobytes()
        for level in (0, 1, 6, 9):
            cases.append(
                {"z": base64.b64encode(zlib.compress(raw, level)).decode(), "raw": base64.b64encode(raw).decode()}
            )
    idx = np.sort(rng.choice(10**6, 500, replace=False))
    var = {"d": base64.b64encode(E.varint_bytes(np.diff(idx, prepend=0))).decode(), "n": len(idx), "want": idx.tolist()}
    poly = [[0, 0], [10, 0], [10, 10], [5, 3], [0, 10]]
    pts = rng.uniform(-1, 11, size=(400, 2)).tolist()
    (tmp_path / "data.json").write_text(json.dumps({"cases": cases, "var": var, "poly": poly, "pts": pts}))
    eng = (Path(spaclone.pl.interactive.__file__).parent / "_static" / "engine.js").read_text()
    script = f"""
const vm = require('vm'); const fs = require('fs');
const doc = {{readyState: 'complete', querySelectorAll: () => [], createElement: () => ({{getContext: () => null}})}};
const ctx = {{window: {{document: doc}}, document: doc}}; ctx.window.window = ctx.window;
vm.createContext(ctx); vm.runInContext({json.dumps(eng)}.replace('typeof window !== "undefined" ? window : this', 'window'), ctx);
const E = ctx.window.SpacloneFigure; const d = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "data.json"))}, 'utf8'));
const out = {{inflate: d.cases.map(c => Buffer.from(E.inflate(E.b64(c.z), 0)).toString('base64') === c.raw),
  varints: Array.from(E.varints(E.b64(d.var.d), d.var.n, true)).every((v, i) => v === d.var.want[i]),
  pip: d.pts.map(p => E.pip(p[0], p[1], d.poly.map(q => q[0]), d.poly.map(q => q[1])))}};
console.log(JSON.stringify(out));
"""
    (tmp_path / "t.js").write_text(script)
    res = subprocess.run([NODE, str(tmp_path / "t.js")], capture_output=True, text=True, timeout=120, check=True)
    out = json.loads(res.stdout)
    assert all(out["inflate"]) and out["varints"]
    from spaclone.pl.interactive._figure import _crossing

    xs, ys = np.array(poly, float).T
    p = np.array(pts)
    np.testing.assert_array_equal(np.array(out["pip"]), _crossing(p[:, 0], p[:, 1], xs, ys))


# ------------------------------------------------------------------------------------------------ the file


def test_saved_file_is_self_contained_and_stamped(syn, tmp_path):
    fig = I.explore(syn, genes=5)
    path = fig.save(tmp_path / "fig")
    text = path.read_text(encoding="utf-8")
    assert path.suffix == ".html"
    assert not re.search(r"<script[^>]+src=", text)  # nothing loaded: the engine is inside
    assert not re.search(r"<link[^>]+href=", text)
    assert f"spaclone {_html.package_version()}" in text
    assert "spaclone.pl.explore(" in text and _html.NOTICE in text
    assert text.count('type="application/json"') == 1
    assert _html.engine_version() == _html.package_version()


def test_same_input_same_bytes(syn):
    a = I.explore(syn, genes=5).to_html()
    b = I.explore(syn, genes=5).to_html()
    assert a == b


def test_engine_file_is_plain_ascii():
    eng = (Path(I.__file__).parent / "_static" / "engine.js").read_bytes()
    assert all(b < 128 for b in eng)
    assert b"__VERSION__" not in eng


def test_cdn_and_separate_engine_file(syn, tmp_path):
    fig = I.spatial_records(syn.air.head(300))
    assert "cdn.jsdelivr.net/gh/Reynold4k/spaclone@v" in fig.to_html(include_js="cdn")
    p = fig.save(tmp_path / "a.html", include_js="spaclone-figure.js")
    assert (tmp_path / "spaclone-figure.js").read_text() == _html.engine_js()
    assert '<script src="spaclone-figure.js">' in p.read_text()
    with pytest.raises(ValueError):
        fig.to_html(include_js="http://example.org/x")


def test_notebook_bundle_carries_picture_and_figure(syn):
    fig = I.spatial_records(syn.air.head(400), syn.gex)
    b = fig._repr_mimebundle_()
    assert set(b) >= {"text/html", "image/png", "text/plain"}
    assert base64.b64decode(b["image/png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert "spc-nb-picture" in b["text/html"] and "srcdoc" in b["text/html"]
    old = I.options.notebook_max_mb
    try:
        I.options.notebook_max_mb = 0.0
        b2 = fig._repr_mimebundle_()
        assert "srcdoc" not in b2["text/html"] and "fig.save" in b2["text/html"]
    finally:
        I.options.notebook_max_mb = old


def test_size_warning(syn, tmp_path):
    fig = I.spatial_records(syn.air.head(200))
    old = I.options.warn_mb
    try:
        I.options.warn_mb = 0.0
        with pytest.warns(I.FigureSizeWarning):
            fig.save(tmp_path / "w.html")
    finally:
        I.options.warn_mb = old


# ------------------------------------------------------------------------------------------------ the explorer


def test_explorer_structure(syn):
    fig = I.explore(syn, genes=10)
    d = _check_spec(fig)
    ids = [c["id"] for c in d["controls"]]
    assert ids[:2] == ["color", "gene"] and "clone" in ids
    color = next(c for c in d["controls"] if c["id"] == "color")
    vals = [o["value"] for o in color["options"]]
    assert {"isotype", "locus", "clone", "cell_type", "gene"} <= set(vals)
    assert color["value"] == "isotype"
    g = next(iter(d["genes"].values()))
    names = d["tables"][g["names"]]
    assert len(names) >= 10 and g["counts"] is True
    assert repr(fig).startswith("<InteractiveFigure 'spaclone_explorer'")


def test_explorer_samples_switch_and_frames(syn):
    other = spaclone.datasets.synthetic_example(rng=7)
    fig = I.explore({"A": syn, "B 小鼠": other}, genes=3)
    d = _check_spec(fig)
    s = next(c for c in d["controls"] if c["id"] == "sample")
    assert [o["value"] for o in s["options"]] == ["A", "B 小鼠"]
    p = d["panels"][0]
    assert p["home_by"] == "sample" and set(p["home_map"]) == {"A", "B 小鼠"}
    assert set(d["frames"]) == {"A", "B 小鼠"}
    for L in p["layers"]:
        assert L.get("when", {}).get("sample") in ("A", "B 小鼠", None)


def test_explorer_genes_counts_exact(syn):
    from spaclone.pl.interactive import _genes

    fig = I.explore(syn, genes=["Jchain", "ighm", "no_such_gene"])
    sp = fig._spec
    key = next(iter(sp.genes))
    names = sp.tables[sp.genes[key]["names"]]
    assert names == ["Jchain", "Ighm"]
    g = syn.gex
    keep = g.obs["in_tissue"].to_numpy(bool)
    col = np.asarray(g[keep][:, "Jchain"].X.todense()).ravel()
    np.testing.assert_array_equal(_genes.values(sp, key, "Jchain", norm=False), col)


def test_selection_numbers_equal_pandas(syn):
    fig = I.explore(syn, genes=0)
    sp = fig._spec
    lay = next(L for L in fig.spec["panels"][0]["layers"] if L.get("select") and L["color"].get("kind") == "cat")
    x = sp.arrays[lay["x"]].astype(float)
    y = sp.arrays[lay["y"]].astype(float)
    box = tuple(float(np.quantile(v, q)) + 1e-3 for v, q in ((x, 0.3), (x, 0.7), (y, 0.25), (y, 0.6)))  # off the points
    s = fig.summary(box=box, state={"color": "isotype"})
    inside = (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])
    sec_air = syn.air[np.isfinite(syn.air[["x_um", "y_um"]].to_numpy(float)).all(1)].reset_index(drop=True)
    assert s.loc[s.quantity == "count", "value"].item() == inside.sum()
    assert s.loc[s.quantity == "clones", "value"].item() == sec_air.loc[inside, "clone_id"].nunique()
    iso = spaclone.pl.interactive._section.isotype_classes(sec_air)[inside].dropna()
    iso = iso[iso != "unknown"]
    want = iso.value_counts(normalize=True)
    for cls, share in want.items():
        got = s.loc[s.quantity == f"isotype (heavy chains): {cls}", "value"].item()
        assert got == pytest.approx(share, abs=1e-12)
    poly = [(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])]
    s2 = fig.summary(polygon=poly, state={"color": "isotype"})
    assert s2.loc[s2.quantity == "count", "value"].item() == inside.sum()


def test_static_functions_keep_their_signatures():
    import inspect

    for name in ("share_map", "spatial_records", "gex_map", "hotspot_atlas", "lineage_tree"):
        assert "interactive" not in inspect.signature(getattr(pl, name)).parameters
    assert pl.share_map.__module__ == "spaclone.pl.maps"
    assert pl.explore is I.explore


# ------------------------------------------------------------------------------------------------ orientation


def test_maps_use_the_static_orientation(syn):
    """The interactive maps rotate and orient a section exactly as the static map of the same function."""
    import matplotlib.pyplot as plt

    from spaclone.pl._frame import frame_of

    air = syn.air.head(2000)
    for static, inter, pts in (
        (lambda: pl.spatial_records(air, syn.gex), lambda: I.spatial_records(air, syn.gex), air[["x_um", "y_um"]]),
        (lambda: pl.clone_map(air), lambda: I.clone_map(air), air[["x_um", "y_um"]]),
    ):
        ax = static()
        fr = frame_of(ax)
        plt.close(ax.figure)
        fig = inter()
        sp = fig._spec
        lay = next(L for L in fig.spec["panels"][0]["layers"] if L["type"] == "points" and L["n"] == len(air))
        got = np.c_[sp.arrays[lay["x"]], sp.arrays[lay["y"]]]
        want = fr(pts.to_numpy(float))
        np.testing.assert_allclose(got, want.astype(np.float32), rtol=0, atol=1e-3)
        assert fig.spec["panels"][0]["invert_y"] is True


# ------------------------------------------------------------------------------------------------ robust inputs


def test_robust_inputs(tmp_path):
    r = _records(400)
    r.loc[::17, "x_um"] = np.nan
    r.loc[::23, "consensus_count"] = np.inf
    fig = I.explore(r)
    assert any("without coordinates" in n["text"] for n in fig.spec["notes"])
    _check_spec(fig)
    one = _records(50, locus=["IGH"] * 50, clone_id=["7"] * 50)
    _check_spec(I.explore(one))
    many = _records(3000, clone_id=[f"{i:04d}" for i in range(3000)])
    d = _check_spec(I.spatial_records(many, color="clone_id"))
    assert len(next(c for c in d["controls"] if c["id"] == "color")["options"]) >= 2
    long = _records(100, locus=["IGH"] * 100)
    long["label"] = ["x" * 300 + str(i % 3) for i in range(100)]
    _check_spec(I.spatial_records(long, color="label"))
    cat = _records(200)
    cat["locus"] = pd.Categorical(cat["locus"])
    cat["clone_id"] = cat["clone_id"].astype("string")
    _check_spec(I.explore(cat))
    with pytest.raises(KeyError):
        I.spatial_records(r, color="nope")
    empty = r.iloc[:0]
    fig = I.explore({"none": (empty, _units(50))})
    _check_spec(fig)


def test_odd_gene_names_and_matrix_kinds(tmp_path):
    names = ("A/B", "<script>x</script>", "Gène-1", "123", "Cd3e")
    for kind in ("sparse", "dense", "backed"):
        a = _units(300, genes=names, dense=(kind == "dense"))
        if kind == "backed":
            p = tmp_path / "u.h5ad"
            a.write_h5ad(p)
            a = ad.read_h5ad(p, backed="r")
        fig = I.gex_map(a, "<script>x</script>", genes=list(names))
        d = _check_spec(fig)
        tid = next(iter(d["genes"].values()))["names"]
        assert d["tables"][tid] == list(names)
        html = fig.to_html(picture=False)
        assert "<script>x</script>" not in html  # escaped inside the JSON
    with pytest.raises(KeyError):
        I.gex_map(_units(10), "not a gene")


def test_scaled_data_is_refused_with_a_reason():
    a = _units(100, dense=True)
    a.X = a.X - 1.0
    with pytest.raises(ValueError, match="non-negative"):
        I.gex_map(a, "Cd3e")


def test_thinning_is_visible(syn):
    fig = I.explore(syn, max_points=1000, genes=0)
    assert any("random 1,000 of" in n["text"] for n in fig.spec["notes"])
    n = max(L["n"] for L in fig.spec["panels"][0]["layers"] if L["type"] == "points")
    assert n <= 1000


# ------------------------------------------------------------------------------------------------ every family


def test_point_maps_build(syn):
    g = syn.gex[syn.gex.obs["in_tissue"].to_numpy(bool)]
    for fig in (
        I.spatial_records(syn, color="truth", highlight=["genuine"]),
        I.isotype_map(spaclone.bcr.isotype(syn.air)[lambda d: d.isotype.notna()]),
        I.clone_map(syn.air, n_top=2),
        I.gex_map(g, "cell_type", genes=5),
        I.gex_map(g, "Jchain", genes=None),
        I.spot_qc_map(syn.air),
    ):
        d = _check_spec(fig)
        assert d["panels"][0]["kind"] == "map"
        png = fig.to_png()
        assert png[:4] == b"\x89PNG"


def test_hotspot_family_builds(hot):
    rid = hot.regions.region_id.iloc[0] if len(hot.regions) else None
    figs = [I.share_map(hot.csr), I.csr_share_map(hot.csr, units=hot.loci, isotype_key="isotype_class"),
            I.hotspot_overview(hot), I.hotspot_atlas(hot), I.csr_map(hot, n_perm=19)]  # fmt: skip
    if hot.shm is not None:
        figs.append(I.shm_share_map(hot.shm, units=hot.loci))
    if rid is not None:
        figs.append(I.region_zoom(hot, rid))
    for fig in figs:
        d = _check_spec(fig)
        assert any(L["type"] == "image" for p in d["panels"] for L in p["layers"])
    d = figs[0].spec
    layers = next(c for c in d["controls"] if c["id"] == "layer")
    assert {o["value"] for o in layers["options"]} >= {"share", "expected", "z", "n_eff"}


def test_plot_family_builds(syn):
    tab = pd.DataFrame({"group": ["GC"] * 50 + ["ASC"] * 50, "reference": "rest", "gene": [f"g{i}" for i in range(100)],
                        "log2fc": np.linspace(-4, 4, 100), "p_value": np.logspace(-12, 0, 100), "q_value": np.logspace(-10, 0, 100)})  # fmt: skip
    d = _check_spec(I.gex_volcano(tab, group="ASC"))
    assert next(c for c in d["controls"] if c["id"] == "group")["value"] == "ASC"
    trans = bcr.csr_transitions(syn.air, n_perm=19, rng=0)
    _check_spec(I.csr_transitions(trans))
    div = spaclone.tl.diversity(syn.air, by="locus", rng=0, n_perm=19)
    _check_spec(I.diversity(div))
    exp = spaclone.tl.clonal_expansion(syn.air, by="locus")
    _check_spec(I.clonal_expansion(exp))
    _check_spec(I.read_support(syn.air))


def test_flow_and_trust_report_build(syn):
    from spaclone.tl import tpt_flux, transition_model

    g = syn.gex
    rec = syn.air[syn.air.spot_id.astype(str).isin(g.obs_names)]
    units = bcr.csr_potential(rec, unit_key="spot_id", species="mouse")
    units["region"] = g.obs["compartment"].astype(str).reindex(units.index).fillna("other").to_numpy()
    model = transition_model(units, time_key="csr_pot", graph="spatial", states="region", rng=0)
    _check_spec(I.csr_flow(model))
    _check_spec(I.stationary_distribution(model))
    states = list(pd.unique(units["region"]))
    flux = tpt_flux(model, source=states[0], target=states[-1], rng=0, n_random=9, n_bootstrap=9)
    _check_spec(I.flux_matrix(flux))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = spaclone.run(syn)
    _check_spec(I.trust_report(rep))


def test_repertoire_null_and_niche_plots_build(syn):
    from spaclone import nulls, tl

    air = syn.air.copy()
    air["region"] = syn.gex.obs["compartment"].astype(str).reindex(air.spot_id.astype(str)).fillna("other").to_numpy()
    _check_spec(I.overlap(tl.overlap(air, by="region", unit="spot")))

    def stat(d):
        return {"n_clones": float(d.clone_id.nunique()), "mean_x": float(d.x_um.mean())}

    pb = nulls.placebo(stat, air.head(1500), nulls.Permute(["clone_id"]), n_perm=19, rng=0)
    d = _check_spec(I.placebo(pb))
    comp = next(c for c in d["controls"] if c["id"] == "component")
    assert [o["label"] for o in comp["options"]] == ["n_clones", "mean_x"] and comp["value"] == "0"
    one = nulls.placebo(
        lambda t: float(t.clone_id.nunique()), air.head(800), nulls.Permute(["clone_id"]), n_perm=19, rng=0
    )
    _check_spec(I.placebo(one))
    small = air[air.clone_id.isin(air.clone_id.value_counts().index[:12])]
    _check_spec(I.clonal_scan(tl.clonal_scan(small, min_units=10, n_perm=19, rng=0, units="records", radii_um=(30,))))
    air["mark"] = air["region"].eq(air["region"].iloc[0])
    air["side"] = np.where(air.x_um < air.x_um.median(), "left", "right")
    _check_spec(I.niche_profile(tl.niche_profile(air, group="mark", compartment="side", n_perm=19, rng=0)))
    air["d"] = air.x_um - air.x_um.median()
    prof = tl.niche_profile(air, group="mark", distance_um="d", bands_um=[-300, -100, 0, 100, 300], n_perm=19, rng=0)
    _check_spec(I.niche_profile(prof, ratio=False))


#: Public plotting functions without an interactive form in this release (listed in docs/interactive.md).
STATIC_ONLY = {"clone_axis_profile", "clone_state_map", "spanning_clones", "mst_edges", "isotype_composition",
               "gex_dotplot", "lineage_gallery", "density_profile", "same_spot_multiplicity", "placement",
               "clones_per_spot", "qc_panel", "valid_shares", "clone_dispersion", "gene_usage", "spectratype"}  # fmt: skip


def test_every_plot_has_an_interactive_form_or_is_listed_static():
    import inspect

    funcs = {n for n in dir(pl) if not n.startswith("_") and inspect.isfunction(getattr(pl, n))} - {"explore"}
    forms = set(I.__all__)
    assert funcs - forms == STATIC_ONLY
    assert forms - funcs == {"explore", "InteractiveFigure", "Options", "options", "FigureSizeWarning"}
    doc = Path(__file__).resolve().parents[1] / "docs" / "interactive.md"
    if doc.exists():
        text = doc.read_text(encoding="utf-8")
        static = text[text.index("Static only in 1.1.0") :]
        assert all(f"spaclone.pl.{n}`" in static for n in STATIC_ONLY)
        assert all(f"spaclone.pl.{n}`" in text[: text.index("Static only in 1.1.0")] for n in funcs & forms)


@pytest.mark.slow
def test_lineage_family_builds(syn):
    lin = bcr.lineage_workflow(syn, rng=0)
    d = _check_spec(I.lineage_tree(lin))
    assert {p["id"] for p in d["panels"]} == {"tree", "map"}
    tid = lin.drawn("exact").tree_id.iloc[0]
    tips = lin.tree_tips(tid, grouping="exact").rename(columns={"tree_label": "sequence_id"})
    _check_spec(I.tree_tissue(lin.trees["exact"][tid], tips))


def test_check_install_lists_interactive_figures():
    rep = spaclone.check_install(deep=False, external=False)
    row = rep.table[rep.table.feature == "interactive figures"]
    assert len(row) == 1 and row.available.item() == "yes"
    spaclone.check_install(require="interactive")
