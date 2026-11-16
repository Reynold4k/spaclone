"""The :class:`InteractiveFigure` object: show, save, notebook display, and selection statistics in Python."""

from __future__ import annotations

import base64
import logging
import tempfile
import warnings
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spaclone._utils import SpacloneWarning
from spaclone.pl.interactive import _html, _png
from spaclone.pl.interactive._spec import Spec

logger = logging.getLogger("spaclone")

__all__ = ["FigureSizeWarning", "InteractiveFigure", "options"]


class FigureSizeWarning(SpacloneWarning):
    """An interactive figure is larger than :attr:`Options.warn_mb`."""


@dataclass
class Options:
    """Settings of interactive figures; the settings in use are ``spaclone.pl.interactive.options``.

    For example ``spaclone.pl.interactive.options.embed = "inline"`` or ``...options.notebook_max_mb = 50``.

    Attributes
    ----------
    embed
        How a notebook shows a figure: ``"auto"`` (default: in an iframe of its own, isolated from the notebook's
        styles, else in the notebook page when the frontend blocks such frames), ``"iframe"`` or ``"inline"``. The
        static picture is shown until the figure has started, and stays where scripts do not run.
    notebook_max_mb
        Largest figure shown interactively inside a notebook output (default 20 MB); a larger one shows its static
        picture and a line saying to save it with :meth:`InteractiveFigure.save`, so that the notebook file stays small
        and Jupyter's output-rate limit is never hit.
    warn_mb
        :meth:`InteractiveFigure.save` warns (:class:`FigureSizeWarning`) above this size (default 50 MB).
    picture
        Put the static picture (PNG of the first view) into notebook outputs and saved files (default True).
    """

    embed: str = "auto"
    notebook_max_mb: float = 20.0
    warn_mb: float = 50.0
    picture: bool = True


#: The settings in use.
options = Options()


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython comes with every notebook
        return False
    ip = get_ipython()
    return ip is not None and hasattr(ip, "kernel")


def _crossing(px: np.ndarray, py: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Even-odd crossing test of points against a polygon: the same arithmetic as the figure engine (``pip``)."""
    inside = np.zeros(len(px), dtype=bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        yi, yj, xi, xj = ys[i], ys[j], xs[i], xs[j]
        cond = (yi > py) != (yj > py)
        if cond.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = (xj - xi) * (py[cond] - yi) / (yj - yi) + xi
            hit = np.zeros(len(px), dtype=bool)
            hit[cond] = px[cond] < xc
            inside ^= hit
        j = i
    return inside


class InteractiveFigure:
    """An interactive figure: one self-contained HTML document with its data inside.

    Made by :func:`spaclone.pl.explore` and the functions of :mod:`spaclone.pl.interactive`. It opens in any current
    browser by double-click, works offline, needs no Python kernel, and is what a notebook shows inline (with a static
    picture for viewers that do not run scripts).

    Examples
    --------
    >>> fig = spaclone.pl.explore(sa)  # doctest: +SKIP
    >>> fig.save("section.html")  # doctest: +SKIP
    >>> fig  # in a notebook: the interactive figure  # doctest: +SKIP
    """

    def __init__(self, spec: Spec, *, name: str = "figure") -> None:
        self._spec = spec
        self.name = name
        self._dict: dict[str, Any] | None = None
        self._png: bytes | None = None
        self._html: dict[tuple, str] = {}

    # ---- content
    @property
    def spec(self) -> dict[str, Any]:
        """The figure specification (JSON-ready dictionary)."""
        if self._dict is None:
            self._dict = self._spec.to_dict()
            self._dict["file_stem"] = self.name
        return self._dict

    def to_png(self) -> bytes:
        """PNG of the first view (drawn with matplotlib from the same specification)."""
        if self._png is None:
            self._png = _png.render(self._spec, self.spec)
        return self._png

    def to_html(self, *, include_js: str = "inline", picture: bool | None = None) -> str:
        """The figure as a standalone HTML document.

        Parameters
        ----------
        include_js
            ``"inline"`` (default: the figure engine inside the file, works offline), ``"cdn"`` (the engine of this
            spaclone release from jsDelivr: about 100 kB smaller, needs the network) or the relative path of a ``.js``
            file (written by :meth:`save` next to the HTML file; for many figures in one folder).
        picture
            Embed the static picture of the first view, shown if the figure cannot start (default
            :attr:`Options.picture`).
        """
        pic = options.picture if picture is None else bool(picture)
        key = (include_js, pic)
        if key not in self._html:
            self._html[key] = _html.page(self.spec, fallback=self.to_png() if pic else None, include_js=include_js)
        return self._html[key]

    @property
    def nbytes(self) -> int:
        """Size of the saved file in bytes (engine inline, with the static picture)."""
        return len(self.to_html().encode("utf-8"))

    def save(self, path: str | Path, *, include_js: str = "inline", picture: bool | None = None) -> Path:
        """Write the figure to one HTML file.

        Parameters
        ----------
        path
            File name (``.html`` is added when missing).
        include_js, picture
            As in :meth:`to_html`. With a ``.js`` path, the engine file is written there too (relative to the HTML
            file) unless it exists with the same content.

        Returns
        -------
        pathlib.Path
            The file written. Its name, size and number of embedded genes are printed; a :class:`FigureSizeWarning`
            is issued above :attr:`Options.warn_mb`.
        """
        p = Path(path)
        if p.suffix.lower() not in (".html", ".htm"):
            p = p.with_name(p.name + ".html")
        text = self.to_html(include_js=include_js, picture=picture)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        if include_js not in ("inline", "cdn"):
            js = p.parent / include_js
            body = _html.engine_js()
            if not js.exists() or js.read_text(encoding="utf-8") != body:
                js.parent.mkdir(parents=True, exist_ok=True)
                js.write_text(body, encoding="utf-8")
        mb = p.stat().st_size / 1e6
        genes = self._spec.genes
        n_genes = len(self._spec.tables[next(iter(genes.values()))["names"]]) if genes else 0
        print(f"wrote {p} ({mb:.1f} MB" + (f", {n_genes:,} genes)" if n_genes else ")"))
        logger.debug("wrote %s (%.2f MB)", p, mb)
        if mb > options.warn_mb:
            warnings.warn(
                f"{p.name} is {mb:.1f} MB: large for e-mail and slow to open; fewer genes (genes=...) or fewer points "
                "(max_points=...) make it smaller",
                FigureSizeWarning,
                stacklevel=2,
            )
        return p

    def show(self) -> None:
        """Display the figure: inline in a notebook, otherwise in the default web browser (from a temporary file)."""
        if _in_notebook():
            from IPython.display import display

            display(self)
            return
        folder = Path(tempfile.mkdtemp(prefix="spaclone-figure-"))
        path = self.save(folder / f"{self.name}.html")
        webbrowser.open(path.as_uri())

    # ---- notebooks
    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        png = self.to_png() if options.picture else None
        out: dict[str, Any] = {"text/plain": repr(self)}
        if png:
            out["image/png"] = base64.b64encode(png).decode("ascii")
        size_mb = self._spec.nbytes() / 1e6
        if size_mb > options.notebook_max_mb:
            b64 = out.get("image/png", "")
            img = f'<img src="data:image/png;base64,{b64}" style="max-width:100%">' if b64 else ""
            out["text/html"] = (
                f"{img}<p style='font:12px sans-serif;color:#888'>This interactive figure holds {size_mb:.0f} MB of data, "
                f"more than spaclone.pl.interactive.options.notebook_max_mb ({options.notebook_max_mb:g}); the picture "
                f"is its first view. Save it with fig.save('{self.name}.html') and open the file.</p>"
            )
            return out
        out["text/html"] = _html.notebook(self.spec, png=png, height=self._height(), embed=options.embed)
        return out

    def _height(self) -> int:
        lay = self.spec.get("layout", {})
        return int(np.clip(lay.get("height_hint", 620), 280, 1000))

    def __repr__(self) -> str:
        d = self.spec
        n_pan = len(d.get("panels", []))
        ctl = ", ".join(c.get("label") or c["id"] for c in d.get("controls", []) if not str(c["id"]).startswith("_"))
        size = self._spec.nbytes() / 1e6
        return (f"<InteractiveFigure {self.name!r}: {d.get('title', '')!s} · {n_pan} panel{'s' if n_pan != 1 else ''}"
                f"{' · controls: ' + ctl if ctl else ''} · data {size:.2f} MB>")  # fmt: skip

    # ---- selection statistics
    def summary(
        self,
        *,
        box: tuple[float, float, float, float] | None = None,
        polygon: Any = None,
        panel: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """What lies inside a region of a map: the numbers the figure's selection panel shows for the same region.

        Parameters
        ----------
        box
            ``(x0, x1, y0, y1)`` in the panel's display coordinates (the figure's axes: micrometres after the section's
            rotation for tissue maps).
        polygon
            ``(k, 2)`` corners in display coordinates (a free-shape selection).
        panel
            Panel id (default: the first panel with selectable points).
        state
            Control values (e.g. ``{"sample": "D21"}``; default: the first view).

        Returns
        -------
        pandas.DataFrame
            One row per number: ``layer`` (the layer's name), ``unit``, ``quantity`` (``"count"``, a distinct count
            such as ``"clones"``, ``"<label>: <category>"`` for shares, or the label of a mean), ``value`` and ``n``
            (units the value is computed over).
        """
        if (box is None) == (polygon is None):
            raise ValueError("give either box=(x0, x1, y0, y1) or polygon=[(x, y), ...]")
        d = self.spec
        st = _png.initial_state(d)
        st.update(state or {})
        if box is not None:
            x0, x1, y0, y1 = (float(v) for v in box)
            xs = np.array([min(x0, x1), max(x0, x1), max(x0, x1), min(x0, x1)])
            ys = np.array([min(y0, y1), min(y0, y1), max(y0, y1), max(y0, y1)])
            is_box = True
        else:
            pg = np.asarray(polygon, dtype=float).reshape(-1, 2)
            xs, ys = pg[:, 0], pg[:, 1]
            is_box = False
        panels = d.get("panels", [])
        if panel is None:
            cand = [p for p in panels if any(L.get("select") for L in p.get("layers", []))]
            if not cand:
                raise ValueError("this figure has no selectable points")
            pnl = cand[0]
        else:
            pnl = next((p for p in panels if p["id"] == panel), None)
            if pnl is None:
                raise KeyError(f"no panel {panel!r} (panels: {[p['id'] for p in panels]})")
        arrays = self._spec.arrays
        rows = []
        for L in _png.visible_layers(pnl, st, set(d.get("layers_off", []))):
            if L.get("type") != "points" or not L.get("select"):
                continue
            stt = d["stats"][L["select"]]
            x = np.asarray(arrays[L["x"]], dtype=np.float64)
            y = np.asarray(arrays[L["y"]], dtype=np.float64)
            with np.errstate(invalid="ignore"):
                m = (x >= xs.min()) & (x <= xs.max()) & (y >= ys.min()) & (y <= ys.max())
            if not is_box and m.any():
                idx = np.flatnonzero(m)
                m[idx] = _crossing(x[idx], y[idx], xs, ys)
            inside = np.flatnonzero(m)
            unit = stt.get("unit", "points")
            lname = str(L.get("name") or L["id"])
            rows.append({"layer": lname, "unit": unit, "quantity": "count", "value": float(len(inside)),
                         "n": len(inside)})  # fmt: skip
            for q in stt.get("rows", []):
                kind = q["kind"]
                if kind == "distinct":
                    c = np.asarray(arrays[q["codes"]], dtype=np.int64)[inside]
                    skip = q.get("skip", -1)
                    rows.append({"layer": lname, "unit": unit, "quantity": q["label"],
                                 "value": float(len(np.unique(c[c != skip]))), "n": len(inside)})  # fmt: skip
                elif kind == "shares":
                    c = np.asarray(arrays[q["codes"]], dtype=np.int64)[inside]
                    names = q["names"]
                    keep = (c < len(names)) & ~np.isin(c, q.get("skip", []))
                    cnt = np.bincount(c[keep], minlength=len(names))[: len(names)]
                    tot = int(keep.sum())
                    order = np.argsort(-cnt, kind="stable")
                    for k in order:  # every category, as the figure's own numbers (it shows those above 0)
                        rows.append({"layer": lname, "unit": unit, "quantity": f"{q['label']}: {names[k]}",
                                     "value": float(cnt[k] / tot) if tot else 0.0, "n": tot})  # fmt: skip
                else:
                    v = np.asarray(arrays[q["values"]], dtype=np.float64)[inside]
                    fin = ~np.isnan(v)
                    total = float(np.cumsum(v[fin])[-1]) if fin.any() else 0.0  # sequential, as the engine sums
                    if kind == "sum":
                        val = total
                    else:
                        val = total / int(fin.sum()) if fin.any() else float("nan")
                    rows.append({"layer": lname, "unit": unit, "quantity": q["label"], "value": val,
                                 "n": int(fin.sum())})  # fmt: skip
        return pd.DataFrame(rows, columns=["layer", "unit", "quantity", "value", "n"])
