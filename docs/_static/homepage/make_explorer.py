"""The interactive explorer of the six public AIR-SPACE lymph nodes: the demo file and the README picture.

Usage (from the repository root; public AIR-SPACE data only, GEO GSE286452)::

    python docs/_static/homepage/make_explorer.py OUT_DIR

writes ``OUT_DIR/spaclone_airspace_explorer.html`` (the demo: Mock to day 21 with their CSR / SHM hotspots, 300
variable genes plus markers) and ``docs/_static/homepage/09_airspace_explorer.png`` (the README picture: day 21 by
isotype class with a box selection, drawn by headless Chromium; needs ``pip install playwright`` and
``python -m playwright install chromium``). The demo file is not part of the repository or of the package.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import spaclone
from spaclone import bcr, gex, io

HERE = Path(__file__).resolve().parent
DAYS = ["Mock", "D3", "D7", "D10", "D14", "D21"]


def build(out_dir: Path) -> Path:
    """Read the six nodes, call regions, run the hotspot workflow with its defaults, save the explorer."""
    warnings.filterwarnings("ignore")
    data = {day: io.read_airspace(*spaclone.datasets.airspace(day), alignments=True) for day in DAYS}
    for sa in data.values():
        gex.call_regions(sa.gex, kind="germinal_centre")
        gex.call_regions(sa.gex, kind="plasma_cell")
    hot = {day: bcr.hotspot_workflow(sa) for day, sa in data.items()}
    fig = spaclone.pl.explore(data, hotspots=hot, genes=300, title="AIR-SPACE mouse lymph nodes, Mock to day 21")
    return fig.save(out_dir / "spaclone_airspace_explorer.html")


def picture(html: Path, png: Path) -> None:
    """Day 21 by isotype class, a box selection with its counts; 1600 x 900 px."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-dev-shm-usage"])
        page = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        page.goto(html.resolve().as_uri())
        page.wait_for_selector('[data-spc-ready="1"]', timeout=120_000)
        page.select_option('[data-control="sample"] select', "D21")
        page.wait_for_timeout(400)
        page.click('button[data-tool="select"]')
        box = page.locator(".spc-panel canvas").first.bounding_box()
        x, y = box["x"] + box["width"] * 0.36, box["y"] + box["height"] * 0.22
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + box["width"] * 0.17, y + box["height"] * 0.3, steps=12)
        page.mouse.up()
        page.wait_for_timeout(500)
        page.click('button[data-tool="pan"]')
        page.mouse.move(5, 5)
        page.wait_for_timeout(300)
        page.screenshot(path=str(png))
        b.close()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    html = build(out)
    print(f"{html}: {html.stat().st_size / 1e6:.1f} MB")
    picture(html, HERE / "09_airspace_explorer.png")
    print(HERE / "09_airspace_explorer.png")
