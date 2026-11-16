"""Interactive figures in real headless browsers (Playwright: Chromium, Firefox and WebKit, the engine of Safari).

Skipped when Playwright or a browser is not installed or cannot start (WebKit needs system libraries that some
Linux machines lack). CI runs them in the ``browser`` job (``python -m playwright install --with-deps chromium firefox
webkit``); locally: ``pytest -m browser``.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import spaclone
from spaclone.pl import interactive as I

pytestmark = pytest.mark.browser

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")

#: Content-process sandbox off for Firefox: some HPC nodes forbid the namespaces it needs (local files only).
FF_ENV = dict.fromkeys(("MOZ_DISABLE_CONTENT_SANDBOX", "MOZ_DISABLE_GMP_SANDBOX", "MOZ_DISABLE_RDD_SANDBOX", "MOZ_DISABLE_SOCKET_PROCESS_SANDBOX", "MOZ_DISABLE_GPU_SANDBOX", "MOZ_DISABLE_UTILITY_SANDBOX"), "1")  # fmt: skip


@pytest.fixture(scope="module")
def pw():
    try:
        with sync_api.sync_playwright() as p:
            yield p
    except Exception as e:  # noqa: BLE001 - no driver
        pytest.skip(f"Playwright cannot start: {e}")


def _launch(pw, name, args=()):
    try:
        if name == "firefox":
            return pw.firefox.launch(
                env=dict(os.environ, **FF_ENV), firefox_user_prefs={"security.sandbox.content.level": 0}
            )
        return getattr(pw, name).launch(args=list(args))
    except Exception as e:  # noqa: BLE001 - the browser is not installed
        pytest.skip(f"{name} is not available: {str(e).splitlines()[0]}")


@pytest.fixture(scope="module", params=["chromium", "firefox", "webkit"])
def browser(pw, request):
    b = _launch(pw, request.param)
    yield b
    b.close()


@pytest.fixture(scope="module")
def chromium(pw):
    b = _launch(pw, "chromium")
    yield b
    b.close()


@pytest.fixture(scope="module")
def syn():
    return spaclone.datasets.synthetic_example(sequences=True)


@pytest.fixture(scope="module")
def explorer_file(syn, tmp_path_factory):
    other = spaclone.datasets.synthetic_example(rng=7, genuine_fraction=0.1)
    fig = I.explore({"A": syn, "B": other}, genes=["Jchain", "Ighm", "Cd3e", "Ms4a1"])
    path = tmp_path_factory.mktemp("fig") / "explore.html"
    fig.save(path)
    return fig, path


def _open(browser, path, width=1300, height=850):
    page = browser.new_page(viewport={"width": width, "height": height})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(path.as_uri())
    page.wait_for_selector("[data-spc-ready]", timeout=60_000)
    assert page.get_attribute("[data-spc-ready]", "data-spc-ready") == "1", errors
    page.wait_for_timeout(200)
    return page, errors


FIG = "document.querySelector('[data-spc-ready]').__spc"
NONBLANK = """(() => { const c = document.querySelector('.spc-panel canvas'); const x = c.getContext('2d');
  const d = x.getImageData(0, 0, c.width, c.height).data; let n = 0;
  for (let i = 0; i < d.length; i += 16) if (d[i] + d[i + 1] + d[i + 2] > 60 && !(d[i] > 250 && d[i+1] > 250 && d[i+2] > 250)) n++;
  return n; })()"""


def test_loads_offline_without_errors(browser, explorer_file):
    _, path = explorer_file
    page, errors = _open(browser, path)
    assert not errors
    assert page.evaluate(NONBLANK) > 500
    assert page.evaluate(f"{FIG}.panels[0].drawnPoints") > 1000
    page.close()


def test_box_zoom_keeps_aspect_and_reset_restores(browser, explorer_file):
    _, path = explorer_file
    page, _ = _open(browser, path)
    k0 = page.evaluate(f"{FIG}.view({FIG}.panels[0]).k")
    page.click('button[data-tool="zoom"]')
    box = page.locator(".spc-panel canvas").first.bounding_box()
    x, y = box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.4
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 120, y + 60, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    t = page.evaluate(
        f"(() => {{ const p = {FIG}.panels[0]; const t = p.tf(); return [t.sx, t.sy, {FIG}.view(p).k]; }})()"
    )
    assert t[2] > 3 * k0
    assert abs(t[0]) == pytest.approx(abs(t[1]))  # one scale on both axes: 1 µm is as long across as down
    page.click('button[data-tool="pan"]')
    page.locator(".spc-panel canvas").first.dblclick(position={"x": 30, "y": 30})
    page.wait_for_timeout(300)
    assert page.evaluate(f"{FIG}.view({FIG}.panels[0]).k") == pytest.approx(k0)
    page.close()


def test_sample_and_layer_switch_change_the_drawing(browser, explorer_file):
    _, path = explorer_file
    page, errors = _open(browser, path)
    n_a = page.evaluate(f"{FIG}.panels[0].drawnPoints")
    leg_a = page.inner_text(".spc-legendbox")
    page.select_option('[data-control="sample"] select', "B")
    page.wait_for_timeout(300)
    assert page.evaluate(f"{FIG}.state.sample") == "B"
    assert page.evaluate(f"{FIG}.panels[0].drawnPoints") != n_a
    assert page.inner_text(".spc-legendbox") != leg_a
    page.select_option('[data-control="color"] select', "cell_type")
    page.wait_for_timeout(300)
    assert "cell type" in page.inner_text(".spc-legendbox").lower()
    assert not errors
    page.close()


def test_gene_search_finds_and_rejects(browser, explorer_file):
    _, path = explorer_file
    page, errors = _open(browser, path)
    inp = page.locator('[data-control="gene"] input')
    inp.click()
    inp.fill("JCH")
    page.wait_for_timeout(150)
    assert "Jchain" in page.inner_text('[data-control="gene"] .spc-list')
    inp.press("Enter")
    page.wait_for_timeout(300)
    assert page.evaluate(f"{FIG}.state.gene") == "Jchain"
    assert page.evaluate(f"{FIG}.state.color") == "gene"
    lab = page.get_attribute(".spc-cbar", "data-label")
    dom = page.get_attribute(".spc-cbar", "data-domain")
    assert lab.startswith("Jchain")
    inp.fill("no-such-gene")
    inp.press("Enter")
    page.wait_for_timeout(200)
    assert "spc-bad" in (inp.get_attribute("class") or "")
    assert "no gene" in page.inner_text('[data-control="gene"]')
    assert page.evaluate(f"{FIG}.state.gene") == "Jchain"
    inp.fill("Ighm")
    inp.press("Enter")
    page.wait_for_timeout(300)
    assert page.get_attribute(".spc-cbar", "data-domain") != dom
    assert not errors
    page.close()


def test_selection_numbers_equal_python(browser, explorer_file):
    fig, path = explorer_file
    page, errors = _open(browser, path)
    page.click('button[data-tool="select"]')
    box = page.locator(".spc-panel canvas").first.bounding_box()
    x, y = box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.3
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 200, y + 160, steps=10)
    page.mouse.up()
    page.wait_for_timeout(300)
    sel = page.evaluate(
        f"(() => {{ const f = {FIG}; return {{xs: f.selection.xs, ys: f.selection.ys, stats: f.lastStats}}; }})()"
    )
    js = sel["stats"][0]
    py = fig.summary(
        box=(min(sel["xs"]), max(sel["xs"]), min(sel["ys"]), max(sel["ys"])), state={"sample": "A", "color": "isotype"}
    )
    assert js["n"] == py.loc[py.quantity == "count", "value"].item() > 0
    rows = {r["label"]: r for r in js["rows"]}
    assert rows["clones"]["value"] == py.loc[py.quantity == "clones", "value"].item()
    for it in rows["isotype (heavy chains)"]["items"]:
        q = py.loc[py.quantity == f"isotype (heavy chains): {it['name']}", "value"].item()
        assert it["share"] == pytest.approx(q, abs=1e-12)
    assert rows["reads per record (mean)"]["value"] == pytest.approx(
        py.loc[py.quantity == "reads per record (mean)", "value"].item(), rel=1e-12
    )
    page.click('button[data-tool="lasso"]')
    pts = [(x + 20, y + 20), (x + 220, y + 40), (x + 160, y + 200), (x + 10, y + 150)]
    page.mouse.move(*pts[0])
    page.mouse.down()
    for p_ in pts[1:]:
        page.mouse.move(*p_, steps=6)
    page.mouse.up()
    page.wait_for_timeout(300)
    sel = page.evaluate(
        f"(() => {{ const f = {FIG}; return {{xs: f.selection.xs, ys: f.selection.ys, stats: f.lastStats}}; }})()"
    )
    py = fig.summary(polygon=np.c_[sel["xs"], sel["ys"]], state={"sample": "A", "color": "isotype"})
    assert sel["stats"][0]["n"] == py.loc[py.quantity == "count", "value"].item()
    assert "selection" in page.inner_text(".spc-stats").lower()
    assert not errors
    page.close()


def test_hover_and_legend_toggle(browser, explorer_file):
    _, path = explorer_file
    page, _ = _open(browser, path)
    hit = page.evaluate(f"""(() => {{ const f = {FIG}, p = f.panels[0], L = p.visibleLayers().find(l => l.select);
      const x = f.blob(L.x), y = f.blob(L.y), t = p.tf(); const i = Math.floor(L.n / 3);
      return [x[i] * t.sx + t.tx, y[i] * t.sy + t.ty]; }})()""")
    box = page.locator(".spc-panel canvas").first.bounding_box()
    page.mouse.move(box["x"] + hit[0], box["y"] + hit[1])
    page.wait_for_timeout(250)
    assert page.is_visible(".spc-tip")
    assert "clone" in page.inner_text(".spc-tip")
    n0 = page.evaluate(f"{FIG}.panels[0].drawnPoints")
    page.locator(".spc-legendbox button.spc-li").first.click()
    page.wait_for_timeout(250)
    assert page.evaluate(f"{FIG}.panels[0].drawnPoints") < n0
    page.close()


def test_works_without_webgl(pw, explorer_file):
    """The engine draws on a 2D canvas: no WebGL, so a browser without it shows the same figure."""
    b = _launch(pw, "chromium", args=["--disable-webgl", "--disable-3d-apis", "--disable-gpu"])
    _, path = explorer_file
    page, errors = _open(b, path)
    assert page.evaluate("!!document.createElement('canvas').getContext('webgl')") is False
    assert page.evaluate(NONBLANK) > 500 and not errors
    b.close()


def test_twenty_figures_on_one_page(chromium, syn, tmp_path):
    """Twenty figures in one document (as a notebook shows them): every one starts and draws."""
    from spaclone.pl.interactive import _html

    figs = [I.spatial_records(syn.air.sample(800, random_state=k), syn.gex.copy()[::4]) for k in range(20)]
    for embed in ("iframe", "inline"):
        body = "".join(_html.notebook(f.spec, png=None, height=360, embed=embed) for f in figs)
        page_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body style='background:#222'>{body}</body></html>"
        path = tmp_path / f"twenty_{embed}.html"
        path.write_text(page_html, encoding="utf-8")
        page = chromium.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda e, errs=errors: errs.append(str(e)))
        page.goto(path.as_uri())
        page.wait_for_timeout(6000)
        if embed == "iframe":
            ready = page.evaluate("""Array.from(document.querySelectorAll('iframe')).filter(f => {
              try { return f.contentDocument.querySelector('[data-spc-ready="1"]'); } catch (e) { return false; } }).length""")
        else:
            ready = page.evaluate("document.querySelectorAll('[data-spc-ready=\"1\"]').length")
        assert ready == 20, (embed, ready, errors[:3])
        assert not errors
        page.close()


@pytest.mark.slow
def test_large_sections_stay_fluid(chromium, tmp_path):
    """A 150,000-unit section and a 500,000-record figure: load and redraw times (written to timings.json)."""
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(0)
    out = {}
    n_units = 150_000
    xy = rng.uniform(0, 6000, size=(n_units, 2)) * [1.0, 0.6]
    g = ad.AnnData(X=sparse.random(n_units, 50, density=0.05, format="csr", random_state=0, dtype=np.float32) * 10,
                   obs=pd.DataFrame({"cell_type": pd.Categorical(rng.choice(list("ABCDEFG"), n_units))}, index=[f"u{i}" for i in range(n_units)]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(50)]))  # fmt: skip
    g.X.data = np.ceil(g.X.data)
    g.obsm["spatial"] = xy
    n_rec = 500_000
    air = pd.DataFrame({"spot_id": [f"s{i}" for i in range(n_rec)], "x_um": rng.uniform(0, 6000, n_rec), "y_um": rng.uniform(0, 3600, n_rec),
                        "locus": rng.choice(["IGH", "IGK", "TRB"], n_rec), "clone_id": rng.integers(0, 50_000, n_rec).astype(str),
                        "c_call": rng.choice(["IGHM", "IGHG1", "IGHA", None], n_rec), "consensus_count": rng.integers(1, 30, n_rec).astype(float)})  # fmt: skip
    for name, fig in (
        ("units_150k", I.gex_map(g, "cell_type", genes=20)),
        ("records_500k", I.spatial_records(air, color="locus")),
    ):
        path = tmp_path / f"{name}.html"
        t = time.time()
        fig.save(path, picture=False)
        build = time.time() - t
        page = chromium.new_page(viewport={"width": 1440, "height": 900})
        t = time.time()
        page.goto(path.as_uri())
        page.wait_for_selector("[data-spc-ready]", timeout=120_000)
        load = time.time() - t
        ms = page.evaluate(f"""(() => {{ const f = {FIG}, p = f.panels[0], v = f.view(p), out = [];
          for (let k = 0; k < 6; k++) {{ const t0 = performance.now(); p.zoomAt([p.w / 2, p.h / 2], k % 2 ? 0.5 : 2); p.settle(); out.push(performance.now() - t0); }}
          return out; }})()""")
        out[name] = {"file_mb": round(path.stat().st_size / 1e6, 2), "build_s": round(build, 2), "load_s": round(load, 2),
                     "redraw_ms": [round(v, 1) for v in ms]}  # fmt: skip
        assert np.median(ms) < 400, out[name]
        page.close()
    (tmp_path / "timings.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def _broken(fig):
    """The figure's specification with something the engine cannot read (the panels are not a list)."""
    d = dict(fig.spec)
    d["panels"] = 5
    return d


@pytest.mark.parametrize("where", ["page", "iframe", "inline"])
def test_a_figure_that_cannot_start_shows_why_and_its_picture(browser, syn, tmp_path, where):
    """Never a blank box: the reason and the static picture, in a saved page and in both notebook embeddings."""
    from spaclone.pl.interactive import _html

    fig = I.spatial_records(syn.air.head(400), color="locus")
    png = fig.to_png()
    if where == "page":
        html = _html.page(_broken(fig), fallback=png)
    else:
        body = _html.notebook(_broken(fig), png=png, height=420, embed=where)
        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body style='margin:12px'>{body}</body></html>"
    path = tmp_path / f"broken_{where}.html"
    path.write_text(html, encoding="utf-8")
    page = browser.new_page(viewport={"width": 1100, "height": 800})
    page.goto(path.as_uri())
    reason = ".spc-error" if where == "page" else ".spc-nb-error"
    page.wait_for_selector(reason, timeout=30_000)
    text = page.inner_text(reason)
    assert "could not start (" in text and text.split("could not start (", 1)[1][:1] not in ("", ")")  # with a reason
    pic = ".spc-fallback-img" if where == "page" else ".spc-nb-picture"
    shown = page.evaluate(f"""(() => {{ const i = document.querySelector('{pic}'); const r = i.getBoundingClientRect();
      return i.complete && i.naturalWidth > 0 && getComputedStyle(i).display !== 'none' && r.width > 100 && r.height > 50; }})()""")
    assert shown
    assert page.evaluate("document.querySelectorAll('[data-spc-ready=\"1\"]').length") == 0
    assert page.evaluate("document.querySelectorAll('.spc-panel').length") == 0  # nothing half built is left
    page.close()


def _small_tree():
    """A three-tip tree (one internal node) and its tips at five places of a section."""
    from spaclone import bcr

    parent = np.array([-1, 0, 1, 1, 0, 0])
    length = np.array([0.0, 2.0, 1.0, 3.0, 4.0, 0.0])
    labels = [None, None, "t1", "t2", "t3", bcr.GERMLINE]
    tree = bcr.LineageTree("c1", parent, length, labels)
    tips = pd.DataFrame({"sequence_id": ["t1", "t1", "t2", "t3", "t3"], "x_um": [100.0, 400.0, 900.0, 1500.0, 1700.0],
                         "y_um": [200.0, 650.0, 300.0, 800.0, 120.0], "isotype": ["IgM", "IgM", "IgG1", "IgG1", "IgA"]})  # fmt: skip
    return tree, tips


def test_tree_and_tissue_highlight_each_other(browser, tmp_path):
    """Hover a tip of the tree: its records are ringed on the section; hover a record: its tip is ringed in the tree."""
    tree, tips = _small_tree()
    path = tmp_path / "tree.html"
    I.tree_tissue(tree, tips).save(path)
    page, errors = _open(browser, path)
    where = (
        "((pid) => { const f = " + FIG + ", p = f.panels.find(q => q.spec.id === pid);"
        " const L = p.visibleLayers().find(l => l.type === 'points' && l.key_group);"
        " const x = f.blob(L.x), y = f.blob(L.y), t = p.tf(), r = p.canvas.getBoundingClientRect();"
        " return [r.left + x[0] * t.sx + t.tx, r.top + y[0] * t.sy + t.ty]; })"
    )
    count = f"((pid) => {FIG}.panels.find(q => q.spec.id === pid).hlCount)"
    ids = page.evaluate(f"{FIG}.panels.map(p => p.spec.id)")
    tree_id = next(i for i in ids if i != "map")
    x, y = page.evaluate(where, tree_id)
    page.mouse.move(x, y)
    page.wait_for_timeout(250)
    assert page.evaluate(count, "map") >= 1  # tree tip -> its records on the section
    page.mouse.move(5, 5)
    x, y = page.evaluate(where, "map")
    page.mouse.move(x, y)
    page.wait_for_timeout(250)
    assert page.evaluate(count, tree_id) >= 1  # record -> its tip in the tree
    assert not errors
    page.close()
