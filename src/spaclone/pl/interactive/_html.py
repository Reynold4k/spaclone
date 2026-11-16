"""HTML documents of interactive figures: a standalone page and the notebook embedding.

A saved figure is one HTML file: the styles, the figure specification with its data (JSON in a
``<script type="application/json">`` element), the figure engine (``_static/engine.js``, spaclone's own code) and,
optionally, a static PNG of the first view, shown when the engine cannot run. Nothing is fetched from the network
unless ``include_js="cdn"``.

In a notebook the output is a picture first (PNG, in the HTML and as ``image/png``), replaced by the interactive figure
once it has started: frontends that strip scripts (an untrusted notebook, GitHub's viewer, nbconvert to PDF) keep the
picture.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

#: Where ``include_js="cdn"`` loads the engine from: the file of the tagged release on GitHub, served by jsDelivr.
CDN = "https://cdn.jsdelivr.net/gh/Reynold4k/spaclone@v{version}/src/spaclone/pl/interactive/_static/engine.js"
#: Licence notice of the embedded drawing code (spaclone's own; no third-party code is embedded).
NOTICE = (
    "Drawing code: the spaclone figure engine (part of spaclone, MIT licence, copyright Chen Zhu), "
    "https://github.com/Reynold4k/spaclone. No third-party code is embedded."
)


def package_version() -> str:
    """Version of the installed spaclone (the engine's own version when spaclone is not installed)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("spaclone")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return engine_version()


@lru_cache(maxsize=1)
def engine_js() -> str:
    """The figure engine (JavaScript)."""
    return resources.files("spaclone.pl.interactive").joinpath("_static/engine.js").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def engine_css() -> str:
    """The figure styles (CSS)."""
    return resources.files("spaclone.pl.interactive").joinpath("_static/engine.css").read_text(encoding="utf-8")


def engine_version() -> str:
    """Version written in the engine file (``var VERSION = "..."``)."""
    m = re.search(r'var VERSION = "([^"]+)"', engine_js())
    if not m:  # pragma: no cover - the package file is broken
        raise RuntimeError("the spaclone figure engine has no version")
    return m.group(1)


def png_bytes(rgba: np.ndarray) -> bytes:
    """PNG of an ``(rows, columns, 3 or 4)`` array (floats 0..1 or uint8), without metadata (same bytes every time)."""
    from PIL import Image

    a = np.asarray(rgba)
    if a.dtype != np.uint8:
        a = np.clip(np.rint(np.nan_to_num(a, nan=0.0) * 255), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", compress_level=9)
    return buf.getvalue()


def png_data_uri(rgba: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgba)).decode("ascii")


def _json(spec: dict[str, Any]) -> str:
    """JSON safe inside a ``<script>`` element (no ``</``, no ``<!--``, no line separators)."""
    s = json.dumps(spec, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_default)
    return s.replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serialisable in a figure: {type(o).__name__}")


def _clean(o: Any) -> Any:
    """Non-finite floats in the specification become null (JSON has no NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def figure_id(spec_json: str) -> str:
    return "spc-" + hashlib.sha1(spec_json.encode("utf-8")).hexdigest()[:12]


def _comment(spec: dict[str, Any]) -> str:
    call = html.escape(str(spec.get("call", "")), quote=False).replace("--", "- -")
    return (
        "<!--\n"
        f"  spaclone {spec.get('version', '')} interactive figure: {html.escape(str(spec.get('title', '')), quote=False)}\n"
        f"  made by: {call}\n"
        "  The data and the drawing code are inside this file: it opens offline and sends nothing anywhere.\n"
        f"  {NOTICE}\n"
        "-->\n"
    )


def _engine_tag(include_js: str, version: str) -> str:
    if include_js == "inline":
        return "<script>" + engine_js().replace("</script", "<\\/script") + "</script>"
    if include_js == "cdn":
        return f'<script src="{CDN.format(version=version)}" crossorigin="anonymous"></script>'
    if include_js.endswith(".js"):
        return f'<script src="{html.escape(include_js)}"></script>'
    raise ValueError("include_js must be 'inline', 'cdn' or the relative path of a .js file")


def root_html(spec: dict[str, Any], *, mode: str, fallback: bytes | None, spec_json: str | None = None) -> str:
    """The figure's container: fallback picture, the specification and the mount point."""
    sj = spec_json if spec_json is not None else _json(_clean(spec))
    fid = figure_id(sj)
    title = html.escape(str(spec.get("title", "spaclone figure")))
    parts = [f'<div class="spc-root spc-{spec.get("style", "dark")}" id="{fid}" data-spc-engine="{engine_version()}" '
             f'data-spc-mode="{mode}">']  # fmt: skip
    if fallback:
        b64 = base64.b64encode(fallback).decode("ascii")
        parts.append(f'<img class="spc-fallback spc-fallback-img" alt="{title} (static view)" '
                     f'src="data:image/png;base64,{b64}">')  # fmt: skip
    parts.append("<noscript><p style=\"font:13px sans-serif;color:#888\">This interactive figure needs JavaScript; "
                 "the picture is its first view.</p></noscript>")  # fmt: skip
    parts.append(f'<script type="application/json" class="spc-spec">{sj}</script>')
    parts.append("</div>")
    return "".join(parts)


def page(spec: dict[str, Any], *, fallback: bytes | None = None, include_js: str = "inline", mode: str = "page") -> str:
    """A standalone HTML document of the figure."""
    sj = _json(_clean(spec))
    version = str(spec.get("version", ""))
    light = " spc-page-light" if spec.get("style") == "light" else ""
    title = html.escape(str(spec.get("title", "spaclone figure")) or "spaclone figure")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="generator" content="spaclone {html.escape(version)}">\n'
        f"<title>{title}</title>\n"
        f"<style>{engine_css()}</style>\n</head>\n"
        f'<body class="spc-page{light}">\n'
        f"{_comment(spec)}"
        f"{root_html(spec, mode=mode, fallback=fallback, spec_json=sj)}\n"
        f"{_engine_tag(include_js, version)}\n"
        "</body>\n</html>\n"
    )


def notebook(spec: dict[str, Any], *, png: bytes | None, height: int, embed: str = "auto") -> str:
    """The ``text/html`` output of a figure in a notebook.

    The output starts as the static picture. ``embed="auto"`` (default) then puts the full page into an
    ``<iframe srcdoc>`` (its own document: no style of the notebook reaches it) and, if that has not started after a
    few seconds (a frontend that blocks such frames), mounts the figure in the notebook page instead; the picture
    stays until one of them has started. ``"iframe"`` and ``"inline"`` use only one way. Frontends that do not run
    scripts keep the picture.
    """
    if embed not in ("auto", "iframe", "inline"):
        raise ValueError("embed must be 'auto', 'iframe' or 'inline'")
    doc = page(spec, fallback=None, mode="embed")
    fid = figure_id(doc)
    alt = html.escape(str(spec.get("title", "spaclone figure")))
    img = ""
    if png:
        b64 = base64.b64encode(png).decode("ascii")
        img = (f'<img class="spc-nb-picture" alt="{alt} (static view)" src="data:image/png;base64,{b64}" '
               'style="display:block;max-width:100%;height:auto">')  # fmt: skip
    src = json.dumps(doc).replace("</", "<\\/").replace("<!--", "<\\!--")
    use_iframe = "true" if embed in ("auto", "iframe") else "false"
    # a frame only (no inline fallback): after 20 s without a start, a line under the picture says so
    wait = (
        "setTimeout(function(){if(!done)fail('the figure did not start in this viewer');},20000);"
        if embed == "iframe"
        else ""
    )
    use_inline = "true" if embed in ("auto", "inline") else "false"
    script = (
        "<script>(function(){"
        f"var box=document.getElementById({json.dumps(fid)});if(!box||box.getAttribute('data-done'))return;"
        "box.setAttribute('data-done','1');"
        f"var doc={src},pic=box.querySelector('.spc-nb-picture'),done=false,f=null,inl=false;"
        "function hidePic(){if(pic)pic.style.display='none';}"
        # a figure that cannot start: the picture stays and a line says why
        "function fail(m){if(done)return;done=true;inl=true;if(f){f.remove();f=null;}if(pic)pic.style.display='block';"
        "var d=document.createElement('div');d.className='spc-nb-error';"
        "d.style.cssText='font:12px/1.4 sans-serif;color:#b9471a;margin:4px 0';"
        "d.textContent='The interactive figure could not start ('+m+')'+(pic?'; the picture above is its static view.':'.');"
        "box.appendChild(d);}"
        "function show(h){if(done||!f)return;done=true;f.style.cssText='width:100%;border:0;display:block;height:'+"
        f"(h&&h>50?h:{int(height)})+'px';hidePic();}}"
        "function inline(){if(done||inl)return;inl=true;if(f){f.remove();f=null;}"
        "try{var t=new DOMParser().parseFromString(doc,'text/html'),st=t.querySelector('style'),"
        "root=t.querySelector('.spc-root'),sc=t.querySelectorAll('script:not([type])');"
        "var wrap=document.createElement('div');wrap.style.cssText='position:relative;width:100%';"
        "if(st)wrap.appendChild(document.importNode(st,true));var r=document.importNode(root,true);wrap.appendChild(r);"
        "box.appendChild(wrap);var s=document.createElement('script');s.textContent=sc[sc.length-1].textContent;"
        "wrap.appendChild(s);var n=0,tm=setInterval(function(){var a=r.getAttribute('data-spc-ready');"
        "if(a==='1'){done=true;hidePic();clearInterval(tm);}else if(a==='error'||++n>80){clearInterval(tm);"
        "var why=a==='error'?(r.getAttribute('data-spc-error')||'unknown error'):'it did not start within 12 s';"
        "wrap.remove();fail(why);}},150);}catch(e){fail(e&&e.message?e.message:String(e));}}"
        "if(" + use_iframe + "){f=document.createElement('iframe');"
        f"f.setAttribute('title',{json.dumps(str(spec.get('title', 'spaclone figure')))});"
        "f.setAttribute('allow','clipboard-write');"
        f"f.style.cssText='width:100%;height:{int(height)}px;border:0;display:block;position:absolute;left:0;top:0;opacity:0;pointer-events:none';"
        "function fit(){try{var d=f&&f.contentDocument;if(!d)return;var r=d.querySelector('[data-spc-ready]');"
        "if(r&&r.getAttribute('data-spc-ready')==='1'){show(r.offsetHeight);f.style.height=r.offsetHeight+'px';}"
        "else if(r&&r.getAttribute('data-spc-ready')==='error'){fail(r.getAttribute('data-spc-error')||'unknown error');}}"
        "catch(e){}}"
        "window.addEventListener('message',function(e){var m=e.data;if(!m||!f||e.source!==f.contentWindow)return;"
        "if(m.spaclone==='ready')show(m.height);else if(m.spaclone==='error')fail(m.message||'unknown error');"
        "else if(m.spaclone==='height'&&done&&m.height>50)f.style.height=m.height+'px';});"
        "f.addEventListener('load',function(){fit();var n=0,t=setInterval(function(){fit();if(++n>40)clearInterval(t);},250);});"
        "f.srcdoc=doc;box.appendChild(f);}"
        "if("
        + use_inline
        + ")setTimeout(inline,"
        + ("9000" if embed == "auto" else "0")
        + ");"
        + wait
        + "})();</script>"
    )
    return f'<div class="spc-embed" id="{fid}" style="position:relative;width:100%;min-height:40px">{img}{script}</div>'
