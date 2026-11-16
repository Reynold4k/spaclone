"""Interactive forms of the point maps: records, isotypes, clones, transcriptome maps, regions and spot QC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import counts_desc
from spaclone.pl import _colors as C
from spaclone.pl.interactive import _genes, _spec
from spaclone.pl.interactive._figure import InteractiveFigure
from spaclone.pl.interactive._section import ISO_NAMES, Section
from spaclone.pl.interactive._util import call_text, check_style

__all__ = ["clone_map", "gex_map", "gex_regions", "isotype_map", "spatial_records", "spot_qc_map"]


def _map_spec(title: str, call: str, style: str) -> tuple[_spec.Spec, dict[str, Any]]:
    spec = _spec.Spec(style=style, title=title, call=call)
    panel = spec.panel("map", "map", area=[0, 0, 1, 1], invert_y=True)
    return spec, panel


def _finish(
    spec: _spec.Spec, panel: dict[str, Any], sec: Section, name: str, *, invert_y: bool = True
) -> InteractiveFigure:
    panel["home"] = sec.home
    panel["invert_y"] = bool(invert_y)
    h = sec.home
    asp = max((h[1] - h[0]) / max(h[3] - h[2], 1e-9), 0.4)
    spec.layout = {"cols": 1, "rows": 1, "aspect": float(np.clip(asp * 1.04, 0.75, 3.2)), "height_hint": 600}
    for n in sec.notes:
        spec.note(n)
    return InteractiveFigure(spec, name=name)


def _categories(values: Any, vocab: Sequence[str], max_categories: int, highlight: Sequence[Any] | None) -> tuple:
    """Values with rare categories pooled into "other" (the static maps' rule) and the names kept, in vocabulary order."""
    vals = pd.Series(np.asarray(values, dtype=object)).astype("string").fillna("NA").to_numpy(dtype=object)
    present = pd.Series(vals[vals != "NA"]).value_counts()
    ranked = [v for v in vocab if v in present.index]
    if highlight is not None:
        hl = {str(h) for h in highlight}
        ranked = [v for v in ranked if v in hl]
    k = max(1, min(int(max_categories), 8))
    if len(ranked) <= k:
        keep = ranked
    else:
        pos = {v: i for i, v in enumerate(vocab)}
        top = set(sorted(ranked, key=lambda v: (-present[v], pos[v]))[: k - 1])
        keep = [v for v in ranked if v in top]
    out = np.where(np.isin(vals, keep) | (vals == "NA"), vals, "other")
    names = list(keep) + [c for c in ("other", "NA") if (out == c).any()]
    return out, names


def spatial_records(
    air: Any,
    gex: Any = None,
    *,
    color: str | None = "receptor",
    palette: Mapping[str, str] | None = None,
    highlight: Sequence[Any] | None = None,
    background: Any = "gex",
    style: str = "dark",
    alpha: float = 0.85,
    max_categories: int = 8,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.spatial_records`: receptor records on the section.

    The same arguments as the static map (``size``, ``legend``, ``scale_bar_um`` and ``ax`` do not apply: markers follow
    the zoom, the legend and the scale bar are always there). A switcher offers the other colourings the table allows
    (BCR / TCR, chain, isotype class, truth, and ``color``); hover a record for its clone, chain, isotype, V / J, CDR3
    and reads; legend entries show or hide a category; box and free-shape selections count the records inside.

    Parameters
    ----------
    air, gex, color, palette, highlight, background, style, alpha, max_categories, rotate, crop, invert_y, title
        As in :func:`spaclone.pl.spatial_records`.
    max_points
        Most records drawn; above it a random subset (noted in the figure). ``None``: all.
    x_key, y_key, locus_key
        Column names.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.io._container import SpatialAIR

    style = check_style(style)
    if isinstance(air, SpatialAIR):
        gex = air.gex if gex is None else gex
        air = air.air
    if not isinstance(air, pd.DataFrame):
        raise TypeError("spatial_records: air must be a receptor table (DataFrame) or a SpatialAIR")
    for c in (x_key, y_key):
        if c not in air.columns:
            raise KeyError(f"spatial_records: column {c!r} not in the receptor table")
    if color is not None and color != "receptor" and color not in air.columns:
        raise KeyError(f"spatial_records: column {color!r} not in the receptor table")
    call = call_text("spaclone.pl.interactive.spatial_records", air, gex, color=color, highlight=highlight, style=style)
    spec, panel = _map_spec(title or "receptor records", call, style)
    bg_xy = None
    if isinstance(background, str):
        if background not in ("gex", "none"):
            raise ValueError("background must be 'gex', 'none' or positions")
        use_gex = gex if background == "gex" else None
    else:
        use_gex = None
        bg_xy = (background[[x_key, y_key]].to_numpy(dtype=float) if isinstance(background, pd.DataFrame)
                 else np.asarray(background, dtype=float).reshape(-1, 2))  # fmt: skip
    # the static map frames the section on every background spot (tissue and glass) or on the records
    frame_pts: Any = "units"
    if use_gex is not None and "spatial" in getattr(use_gex, "obsm", {}):
        frame_pts = np.asarray(use_gex.obsm["spatial"], dtype=float)[:, :2]
    elif bg_xy is not None and len(bg_xy):
        frame_pts = bg_xy
    else:
        frame_pts = air[[x_key, y_key]].to_numpy(dtype=float)
    sec = Section(spec, panel, None, air, use_gex, style=style, rotate=rotate, crop=crop, in_tissue=False,
                  max_points=max_points, frame_points=frame_pts, x_key=x_key, y_key=y_key, locus_key=locus_key)  # fmt: skip
    mode = sec.mode
    ch = C.CHROME[mode]
    if sec.n_units:  # tissue and glass, as in the static map
        g = use_gex
        tis = (pd.array(g.obs["in_tissue"], dtype="boolean").fillna(False).to_numpy(dtype=bool)[sec.unit_rows]
               if "in_tissue" in g.obs.columns else np.ones(sec.n_units, dtype=bool))  # fmt: skip
        enc, _, _ = _spec.cat_color(
            spec, np.where(tis, "tissue", "glass"), {"tissue": ch["tissue_strong"], "glass": ch["glass"]}
        )
        sec.add(_spec.points(spec, None, xy_ids=sec.unit_ids(), color=enc, size=sec.unit_size(), shape=sec.unit_shape,
                             hover=False, name="tissue"))  # fmt: skip
    elif bg_xy is not None and len(bg_xy):
        sec.add(_spec.points(spec, _geo_apply(sec, bg_xy), color={"kind": "const", "color": ch["tissue_strong"]},
                             size={"px": 2.0}, hover=False, name="tissue"))  # fmt: skip
    a = sec.air
    options: list[tuple[str, str]] = []
    first = color if color is not None else "none"
    if a is None or not len(a):
        spec.note("no records with coordinates", warn=True)
        return _finish(spec, panel, sec, "spatial_records", invert_y=invert_y)
    loc = a[locus_key].astype("string").str.upper() if locus_key in a.columns else None

    def add(value: str, label: str, vals: Any, colors: Mapping[str, str], names: Sequence[str], hkey: str) -> None:
        options.append((value, label))
        cols = dict(colors) | dict(palette or {})
        hidden = [n for n in names if highlight is not None and value == color and n not in {str(h) for h in highlight}]
        sec.records_cat(vals, cols, title=label, names=list(names), when={"color": value}, hkey=hkey, hidden=hidden,
                        alpha=alpha)  # fmt: skip

    if loc is not None:
        rec = np.where(
            loc.isin(S.BCR_LOCI).fillna(False), "BCR", np.where(loc.isin(S.TCR_LOCI).fillna(False), "TCR", "other")
        )
        add("receptor", "receptor", rec, dict(C.RECEPTOR_COLORS[mode]) | {"other": C.NEUTRAL[mode]},
            [n for n in ("BCR", "TCR", "other") if (rec == n).any()], "receptor")  # fmt: skip
        vals, names = _categories(loc.to_numpy(dtype=object), list(S.ALL_LOCI), max_categories, None)
        add(
            locus_key,
            "chain",
            vals,
            C.locus_colors(mode) | {"other": C.NEUTRAL[mode], "NA": C.NEUTRAL[mode]},
            names,
            "locus",
        )
    iso = sec.isotypes()
    if iso.notna().any():
        vals = iso.fillna("other chain").to_numpy(dtype=object)
        names = [n for n in ISO_NAMES + ["other chain"] if (vals == n).any()]
        add("isotype", "isotype class", vals, {n: C.isotype_color(n, mode) for n in ISO_NAMES} | {"other chain": ch["unassigned"]},
            names, "isotype")  # fmt: skip
    for col in ([color] if color not in (None, "receptor", locus_key, "isotype") else []) + (
        ["truth"] if "truth" in a.columns and color != "truth" else []
    ):
        if col not in a.columns:
            continue
        from spaclone.pl.records import _FIXED_SLOT, FIXED_ORDER

        vocab = [v for v in C.vocabulary(a[col], order=FIXED_ORDER.get(col)) if v not in ("nan", "None", "<NA>", "NaN")]
        vals, names = _categories(
            a[col].to_numpy(dtype=object), vocab, max_categories, highlight if col == color else None
        )
        slot_of = _FIXED_SLOT.get(col, {})
        named = {v: C.SLOTS[mode][slot_of[v]] for v in vocab if v in slot_of}
        cols = C.fixed_colors(vocab if len(vocab) <= 8 else [n for n in names if n not in ("other", "NA")], mode=mode,
                              named=named)  # fmt: skip
        cols |= {"other": C.NEUTRAL[mode], "NA": C.NEUTRAL[mode]}
        add(col, col, vals, cols, names, f"col:{col}")
    if color is None:
        options.append(("none", "records"))
        sec.add(_spec.points(spec, None, xy_ids=sec.record_ids(), color={"kind": "const", "color": C.RECEPTOR_COLORS[mode]["BCR"]},
                             size=sec.record_size(), alpha=alpha, hover=sec.record_hover(), when={"color": "none"},
                             select=sec.record_stats()))  # fmt: skip
    values = [v for v, _ in options]
    if first not in values:
        first = values[0]
    spec.control(
        "color", "select", label="colour", options=[{"value": v, "label": lab} for v, lab in options], value=first
    )
    return _finish(spec, panel, sec, "spatial_records", invert_y=invert_y)


def _geo_apply(sec: Section, xy: np.ndarray) -> np.ndarray:
    from spaclone.pl.interactive import _geo

    return _geo.apply(sec.m, xy)


def isotype_map(
    units: pd.DataFrame,
    *,
    isotype_key: str = "isotype",
    x_key: str = S.X,
    y_key: str = S.Y,
    style: str = "dark",
    palette: Mapping[str, str] | None = None,
    order: Sequence[str] = ("unknown", "IgM", "IgD", "IgE", "IgG", "IgA"),
    background: pd.DataFrame | None = None,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.isotype_map`: units coloured by isotype, each class can be shown or hidden.

    Parameters
    ----------
    units, isotype_key, x_key, y_key, style, palette, order, background, rotate, crop, invert_y, title
        As in :func:`spaclone.pl.isotype_map` (``order``: later classes are drawn on top).
    max_points
        Most units drawn (a random subset above it, noted in the figure). ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pl.bcr import _colors_for

    style = check_style(style)
    for c in (isotype_key, x_key, y_key):
        if c not in units.columns:
            raise KeyError(f"isotype_map: column {c!r} not in the table")
    call = call_text("spaclone.pl.interactive.isotype_map", units, isotype_key=isotype_key, style=style)
    spec, panel = _map_spec(title or "isotype", call, style)
    bxy = background[[x_key, y_key]].to_numpy(dtype=float) if background is not None else None
    fp = bxy if bxy is not None and len(bxy) else units[[x_key, y_key]].to_numpy(dtype=float)
    sec = Section(spec, panel, None, units, None, style=style, rotate=rotate, crop=crop, max_points=max_points,
                  frame_points=fp, x_key=x_key, y_key=y_key)  # fmt: skip
    ch = C.CHROME[sec.mode]
    if bxy is not None and len(bxy):
        sec.add(_spec.points(spec, _geo_apply(sec, bxy), color={"kind": "const", "color": ch["tissue_strong"]},
                             size={"um": max(sec.spacing, 2.0), "min": 1.0, "max": 40}, hover=False, name="tissue"))  # fmt: skip
    a = sec.air
    if a is None or not len(a):
        spec.note("no units with coordinates", warn=True)
        return _finish(spec, panel, sec, "isotype_map", invert_y=invert_y)
    iso = a[isotype_key].astype("string").fillna("unknown").astype(str)
    present = list(pd.unique(iso))
    ranked = [v for v in present if v not in order] + [v for v in order if v in present]
    cols = _colors_for(ranked, palette, style, C.vocabulary(iso))
    hover = {"title": _spec.hover_cat(spec, iso.to_numpy(dtype=object), "isotype"), "rows": [
        r for r in sec.record_hover()["rows"] if r.get("label") != "isotype"]}  # fmt: skip
    enc, names, codes = _spec.cat_color(spec, iso.to_numpy(dtype=object), cols, names=ranked)
    stack = np.argsort(np.array([ranked.index(v) for v in iso]), kind="stable")  # later classes on top
    sid = "isotypes"
    cid, _, _ = spec.codes(iso.to_numpy(dtype=object), ranked)
    spec.stats[sid] = {"unit": "units", "rows": [{"kind": "shares", "label": "isotype", "codes": cid, "names": ranked,
                                                  "colors": [cols[n] for n in ranked]}]}  # fmt: skip
    lay = _spec.points(spec, None, xy_ids=sec.record_ids(), color=enc, size=sec.record_size(), hover=hover, order=stack,
                       select=sid, name="isotype")  # fmt: skip
    sec.add(lay)
    counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
    _spec.legend_cat(spec, isotype_key, names, enc["colors"], counts, layer=lay["id"])
    return _finish(spec, panel, sec, "isotype_map", invert_y=invert_y)


def clone_map(
    air: pd.DataFrame,
    clones: str | Sequence[str] | None = None,
    *,
    n_top: int = 3,
    loci: str | Sequence[str] | None = "heavy",
    links: bool = True,
    max_link_um: float | None = 250.0,
    background: bool = True,
    style: str = "dark",
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    x_key: str = S.X,
    y_key: str = S.Y,
    clone_key: str = S.CLONE,
    spot_key: str = S.SPOT,
    locus_key: str = S.LOCUS,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.clone_map`: a searchable clone list over every unit.

    The clones of ``clones`` (or the ``n_top`` largest) are drawn first; type in the search box to add any clone of
    the table (up to eight, each in its own colour; the largest clones are listed first), remove one with its ×.
    Links join a clone's distinct positions along its minimum spanning tree (computed in the browser), up to
    ``max_link_um``.

    Parameters
    ----------
    air, clones, n_top, loci, links, max_link_um, background, style
        As in :func:`spaclone.pl.clone_map`.
    rotate, crop, invert_y, x_key, y_key, clone_key, spot_key, locus_key
        As in :func:`spaclone.pl.clone_map`.
    title
        Figure title.
    max_points
        Most records drawn (a random subset above it, noted). ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    if clone_key not in air.columns:
        raise KeyError(f"clone_map: column {clone_key!r} not in the table")
    d = air[air[clone_key].notna()]
    if clones is None:
        pool = d
        if loci is not None and locus_key in d.columns:
            want = set(S.HEAVY_LOCI) if loci == "heavy" else ({loci} if isinstance(loci, str) else set(loci))
            sel = d[locus_key].astype(str).str.upper().isin({str(v).upper() for v in want}).to_numpy()
            pool = d[sel] if sel.any() else d
        key = [spot_key, clone_key] if spot_key in pool.columns else [x_key, y_key, clone_key]
        # as the static clone_map: most spots first, ties in the order the clones appear
        chosen = [str(c) for c in counts_desc(pool.drop_duplicates(key)[clone_key]).index[:n_top]]
    else:
        chosen = [str(clones)] if isinstance(clones, str) else [str(c) for c in clones]
    call = call_text("spaclone.pl.interactive.clone_map", air, clones=clones, n_top=n_top, style=style)
    spec, panel = _map_spec(title or "clones", call, style)
    sec = Section(spec, panel, None, air, None, style=style, rotate=rotate, crop=crop, max_points=max_points,
                  frame_points=air[[x_key, y_key]].to_numpy(dtype=float), x_key=x_key, y_key=y_key,
                  locus_key=locus_key, clone_key=clone_key)  # fmt: skip
    if not sec.n_records:
        spec.note("no records with coordinates", warn=True)
        return _finish(spec, panel, sec, "clone_map", invert_y=invert_y)
    lay, tid, aux = sec.records_highlight(control="clone", links=links, max_link_um=max_link_um)
    if not background:
        spec.layers_off.append("others")
    names = spec.tables[tid]
    missing = [c for c in chosen if c not in set(names)]
    if missing:
        spec.note(f"not in the table: {', '.join(missing[:5])}", warn=True)
    spec.control("clone", "search", label="clones", source=tid, aux=aux, placeholder="add a clone", noun="clone",
                 nouns="clones", multi=True, max=8, colors=list(C.SLOTS[sec.mode]),
                 value=[c for c in chosen if c in set(names)][:8])  # fmt: skip
    spec.legend.append({"kind": "highlight", "title": "clones", "control": "clone", "colors": list(C.SLOTS[sec.mode]),
                        "other": "every other record" if background else None, "other_color": C.CHROME[sec.mode]["muted"],
                        "empty": "add a clone in the search box"})  # fmt: skip
    return _finish(spec, panel, sec, "clone_map", invert_y=invert_y)


def gex_map(
    adata: Any,
    color: str,
    *,
    style: str = "dark",
    palette: Mapping[str, str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    layer: str | None = None,
    mask: Any = None,
    genes: int | Sequence[str] | str | None = 300,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    spatial_key: str = "spatial",
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.gex_map`: any ``obs`` column or gene, with a gene picker.

    The switcher lists the categorical and numeric ``obs`` columns; the gene picker offers ``genes`` (see
    :func:`spaclone.pl.explore`), always including ``color`` when it is a gene.

    Parameters
    ----------
    adata, color, style, palette, groups, vmin, vmax, layer, mask, rotate, crop, invert_y, title, spatial_key
        As in :func:`spaclone.pl.gex_map` (``mask``: only these units are drawn).
    genes
        Genes the picker offers (a number of variable genes, a list, ``"all"``, or None).
    max_points
        Most units drawn (a random subset above it, noted). ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    if spatial_key not in adata.obsm:
        raise KeyError(f"gex_map: adata.obsm has no {spatial_key!r}")
    is_obs = color in adata.obs.columns
    if not is_obs and color not in adata.var_names:
        raise KeyError(f"{color!r} is neither an obs column nor a gene")
    sub = adata if mask is None else adata[np.asarray(mask, dtype=bool)]
    call = call_text("spaclone.pl.interactive.gex_map", adata, color, genes=genes, style=style)
    spec, panel = _map_spec(title if title is not None else str(color), call, style)
    sec = Section(spec, panel, None, None, sub, style=style, rotate=rotate, crop=crop, in_tissue=False,
                  max_points=max_points, spatial_key=spatial_key)  # fmt: skip
    if not sec.n_units:
        spec.note("no units with coordinates", warn=True)
        return _finish(spec, panel, sec, "gex_map", invert_y=invert_y)
    obs = sub.obs
    options: list[tuple[str, str]] = []
    cat_cols = [c for c in obs.columns if (isinstance(obs[c].dtype, pd.CategoricalDtype) or obs[c].dtype == object
                                           or pd.api.types.is_bool_dtype(obs[c]) or pd.api.types.is_string_dtype(obs[c]))
                and 1 <= obs[c].nunique() <= 60]  # fmt: skip
    num_cols = [c for c in obs.columns if pd.api.types.is_numeric_dtype(obs[c]) and not pd.api.types.is_bool_dtype(obs[c])
                and c not in cat_cols]  # fmt: skip
    stats_cols = [(c, c) for c in cat_cols[:3]]
    for c in cat_cols:
        vals = None
        if groups and c == color:
            folded = {str(m): str(n) for n, ms in groups.items() for m in ms}
            raw = obs[c].astype(object).to_numpy()
            vals = np.array([folded.get(str(v), v) for v in raw], dtype=object)
        lay = sec.units_cat(c, title=c, when={"color": f"obs:{c}"}, stats=stats_cols, hkey=f"obs:{c}", values=vals,
                            extra_names=list(groups) if (groups and c == color) else None)  # fmt: skip
        if palette and c == color:
            lay["color"]["colors"] = [
                palette.get(n, cc) for n, cc in zip(lay["color"]["names"], lay["color"]["colors"], strict=True)
            ]
        options.append((f"obs:{c}", c))
    for c in num_cols:
        v = pd.to_numeric(obs[c], errors="coerce").to_numpy(dtype=float)[sec.unit_rows]
        lay = sec.units_num(v, title=c, ramp=C.RAMPS["seq"][sec.mode], when={"color": f"obs:{c}"}, stats=stats_cols,
                            quantiles=(0.01, 0.99))  # fmt: skip
        if c == color and (vmin is not None or vmax is not None):
            dom = lay["color"]["domain"]
            lay["color"]["domain"] = [vmin if vmin is not None else dom[0], vmax if vmax is not None else dom[1]]
        options.append((f"obs:{c}", c))
    gene_list: list[str] = []
    if genes not in (None, 0) or not is_obs:
        want = genes if genes not in (None, 0) else []
        gene_list, _ = _genes.select_genes([(sub, sec.unit_rows)], want, layer=layer) if want else ([], {})
        if not is_obs and color not in gene_list:
            gene_list = [color] + gene_list
    if gene_list:
        _genes.encode(spec, "_", sub, sec.unit_rows, gene_list, layer=layer, unit_word=sec.unit_word)
        sec.units_gene("_", when={"color": "gene"}, stats=stats_cols)
        options.append(("gene", "gene"))
        spec.control("gene", "search", label="gene", source={"genes": "_"}, placeholder=f"gene ({len(gene_list):,})",
                     noun="gene", nouns="genes", also={"color": "gene"}, value=None if is_obs else color)  # fmt: skip
        if spec.genes["_"].get("counts"):
            spec.control("_norm", "select", label="values", options=[{"value": "norm", "label": "log1p(counts per 10k)"},
                         {"value": "counts", "label": "counts"}], value="norm", when={"color": "gene"})  # fmt: skip
    first = f"obs:{color}" if is_obs else "gene"
    spec.controls.insert(0, {"id": "color", "kind": "select", "label": "show",
                             "options": [{"value": v, "label": lab} for v, lab in options], "value": first})  # fmt: skip
    return _finish(spec, panel, sec, "gex_map", invert_y=invert_y)


def gex_regions(
    adata: Any,
    *,
    region_key: str = "region",
    instance_key: str | None = "region_id",
    style: str = "dark",
    palette: Mapping[str, str] | None = None,
    label_instances: bool = True,
    rotate: bool | float = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    spatial_key: str = "spatial",
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.gex_regions`: called regions with their instance numbers.

    Hover a unit for its region and instance; legend entries show or hide a region class; selections count units per
    region class.

    Parameters
    ----------
    adata, region_key, instance_key, style, palette, label_instances, rotate, invert_y, title, spatial_key
        As in :func:`spaclone.pl.gex_regions`.
    max_points
        Most units drawn. ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    style = check_style(style)
    if region_key not in adata.obs.columns:
        raise KeyError(f"gex_regions: obs has no {region_key!r} (run spaclone.gex.call_regions first)")
    call = call_text("spaclone.pl.interactive.gex_regions", adata, region_key=region_key, style=style)
    spec, panel = _map_spec(title or region_key, call, style)
    sec = Section(spec, panel, None, None, adata, style=style, rotate=rotate, crop=True, in_tissue=False,
                  max_points=max_points, spatial_key=spatial_key)  # fmt: skip
    if not sec.n_units:
        spec.note("no units with coordinates", warn=True)
        return _finish(spec, panel, sec, "gex_regions", invert_y=invert_y)
    lay = sec.units_cat(region_key, title=region_key, stats=[(region_key, "region")])
    if palette:
        lay["color"]["colors"] = [
            palette.get(n, c) for n, c in zip(lay["color"]["names"], lay["color"]["colors"], strict=True)
        ]
    if label_instances and instance_key is not None and instance_key in adata.obs.columns:
        ids = adata.obs[instance_key].astype(object).to_numpy()[sec.unit_rows]
        ok = ~pd.isna(ids)
        if ok.any():
            frame = pd.DataFrame({"id": ids[ok].astype(str), "x": sec.uxy[ok, 0], "y": sec.uxy[ok, 1]})
            cen = frame.groupby("id", sort=False)[["x", "y"]].mean()
            labels = [rid.split("|")[-1].rsplit("_", 1)[-1] for rid in cen.index]
            tl = _spec.text(spec, cen.to_numpy(), labels, size=11.5, weight="700")
            tl["okey"] = "instances"
            sec.add(tl)
            _spec.legend_lines(
                spec, None, [("instance numbers", C.CHROME[sec.mode]["ink"], "line")], layers=["instances"]
            )
    return _finish(spec, panel, sec, "gex_regions", invert_y=invert_y)


def spot_qc_map(
    data: Any,
    *,
    color: str = "n_records",
    style: str | None = None,
    log: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    title: str | None = None,
    max_points: int | None = 500_000,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> InteractiveFigure:
    """Interactive form of :func:`spaclone.pl.spot_qc_map`: a per-spot QC metric, with a switcher over the metrics.

    Parameters
    ----------
    data, color, style, log, rotate, crop, invert_y, x_key, y_key
        As in :func:`spaclone.pl.spot_qc_map` (counts on a log colour scale with ``log=True``).
    title
        Figure title.
    max_points
        Most spots drawn. ``None``: all.

    Returns
    -------
    InteractiveFigure
    """
    from spaclone.pp.metrics import QCMetrics, qc_metrics

    style = check_style(style)
    if isinstance(data, QCMetrics):
        sp = data.spots
    elif isinstance(data, pd.DataFrame) and "n_records" in data.columns and color in data.columns:
        sp = data
    else:
        sp = qc_metrics(data, by=()).spots
    if color not in sp.columns:
        raise KeyError(f"spot_qc_map: column {color!r} not in the spot table")
    call = call_text("spaclone.pl.interactive.spot_qc_map", data, color=color, style=style)
    spec, panel = _map_spec(title or "spot QC", call, style)
    sec = Section(spec, panel, None, sp.reset_index(), None, style=style, rotate=rotate, crop=crop, max_points=max_points,
                  frame_points=sp[[x_key, y_key]].to_numpy(dtype=float), x_key=x_key, y_key=y_key)  # fmt: skip
    a = sec.air
    metrics = [c for c in ("n_records", "n_reads", "n_clones", "n_loci", "valid_share", "productive_share", "is_sink")
               if c in a.columns] + ([color] if color not in ("n_records", "n_reads", "n_clones", "n_loci", "valid_share",
                                                              "productive_share", "is_sink") else [])  # fmt: skip
    ramp = C.RAMPS["blue_ordinal"][sec.mode]
    spot_hover = {"title": _spec.hover_cat(spec, a[S.SPOT] if S.SPOT in a.columns else a.index.astype(str), "spot"),
                  "rows": [_spec.hover_num(spec, pd.to_numeric(a[c], errors="coerce"), c) for c in metrics if c != "is_sink"]}  # fmt: skip
    sid = "spots"
    spec.stats[sid] = {"unit": "spots", "rows": [{"kind": "sum" if c.startswith("n_") else "mean", "label": f"{c} (sum)" if c.startswith("n_") else f"{c} (mean)",
                                                  "values": spec.blob(pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float), "f4")}
                                                 for c in metrics if c != "is_sink"]}  # fmt: skip
    size = {"um": float(sec.spacing), "min": 1.2, "max": 80.0}
    for c in metrics:
        v = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        lg = bool(log and c.startswith("n_") and np.nanmax(v) > 0) if len(v) else False
        enc, dom = _spec.num_color(spec, v, ramp, scale="log" if lg else "lin", quantiles=(0.0, 1.0) if lg else (0.0, 1.0),
                                   domain=(max(float(np.nanmin(v[v > 0])), 1e-9), float(np.nanmax(v))) if lg else None)  # fmt: skip
        lay = _spec.points(spec, None, xy_ids=sec.record_ids(), color=enc, size=size, hover=spot_hover, when={"color": c},
                           select=sid, name=c)  # fmt: skip
        sec.add(lay)
        _spec.legend_colorbar(spec, c, ramp, dom, scale="log" if lg else "lin", when={"color": c})
    spec.control("color", "select", label="metric", options=[{"value": c, "label": c} for c in metrics], value=color)
    return _finish(spec, panel, sec, "spot_qc_map", invert_y=invert_y)
