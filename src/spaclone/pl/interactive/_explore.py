"""The explorer: one or several sections with a sample switcher, a layer switcher, a gene picker and a clone search."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone.pl import _colors as C
from spaclone.pl.interactive import _genes, _spec
from spaclone.pl.interactive._figure import InteractiveFigure
from spaclone.pl.interactive._hotspots import HotspotLayers
from spaclone.pl.interactive._section import ISO_NAMES, Section
from spaclone.pl.interactive._util import call_text, check_style

logger = logging.getLogger("spaclone")

__all__ = ["explore"]

#: Layers of the switcher: (value, label, needs) in menu order.
LAYERS = (
    ("isotype", "isotype class", "iso"),
    ("switched", "switched / unswitched", "iso"),
    ("receptor", "BCR / TCR", "locus"),
    ("locus", "chain", "locus"),
    ("clone", "clones (search)", "clone"),
    ("reads", "reads per record", "reads"),
    ("csr", "switched share (CSR map)", "csr"),
    ("shm", "mutated share (SHM map)", "shm"),
    ("cell_type", "cell type", "cell_type"),
    ("region", "regions (transcriptome)", "region"),
    ("gene", "gene expression", "genes"),
    ("umis", "UMIs per unit", "umis"),
)


def _samples(data: Any, samples: Sequence[str] | None, sample_key: str) -> list[tuple[str, Any, Any, Any]]:
    """``(name, air, gex, hotspot result)`` per sample from the accepted inputs."""
    from spaclone.io import SpatialAIR

    def one(obj: Any) -> tuple[Any, Any, Any]:
        if isinstance(obj, SpatialAIR):
            return obj.air, obj.gex, None
        if isinstance(obj, pd.DataFrame):
            return obj, None, None
        if hasattr(obj, "obsm") and hasattr(obj, "obs"):
            return None, obj, None
        if hasattr(obj, "loci") and hasattr(obj, "grid"):  # a hotspot workflow result
            rec = getattr(obj, "records", None)
            return (rec if isinstance(rec, pd.DataFrame) else None), getattr(obj, "gex", None), obj
        if isinstance(obj, tuple) and len(obj) in (2, 3):
            return (obj[0], obj[1], obj[2] if len(obj) == 3 else None)
        raise TypeError(
            "explore() takes a SpatialAIR, a receptor table, an AnnData, a hotspot workflow result, or a dict / list of "
            f"them (one per sample); got {type(obj).__name__}"
        )

    out: list[tuple[str, Any, Any, Any]] = []
    if isinstance(data, Mapping):
        for k, v in data.items():
            a, g, h = one(v)
            out.append((str(k), a, g, h))
    elif isinstance(data, list | tuple) and data and not (isinstance(data, tuple) and len(data) in (2, 3) and
                                                         isinstance(data[0], pd.DataFrame | type(None))):  # fmt: skip
        for i, v in enumerate(data):
            a, g, h = one(v)
            name = None
            if hasattr(v, "meta") and isinstance(v.meta, Mapping):
                name = v.meta.get("library_id")
            out.append((str(name) if name else f"sample {i + 1}", a, g, h))
    else:
        a, g, h = one(data)
        name = None
        if hasattr(data, "meta") and isinstance(data.meta, Mapping):
            name = data.meta.get("library_id")
        if a is not None and g is None and sample_key in a.columns and a[sample_key].nunique() > 1:
            for lib, sub in a.groupby(a[sample_key].astype(str), sort=False):
                out.append((str(lib), sub, None, None))
        else:
            out.append((str(name) if name else "section", a, g, h))
    if not out:
        raise ValueError("explore(): no sample given")
    names = [n for n, *_ in out]
    if len(set(names)) < len(names):  # duplicated names: number them
        seen: dict[str, int] = {}
        renamed = []
        for n, *rest in out:
            seen[n] = seen.get(n, 0) + 1
            renamed.append((n if names.count(n) == 1 else f"{n} ({seen[n]})", *rest))
        out = renamed
    if samples is not None:
        by = {n: (n, a, g, h) for n, a, g, h in out}
        missing = [s for s in samples if str(s) not in by]
        if missing:
            raise KeyError(f"samples {missing} not in the data (samples: {list(by)})")
        out = [by[str(s)] for s in samples]
    return out


def explore(
    data: Any,
    *,
    hotspots: Any = None,
    samples: Sequence[str] | None = None,
    color: str | None = None,
    genes: int | Sequence[str] | str | None = 300,
    layer: str | None = None,
    cell_type_key: str | None = None,
    region_key: str = "region",
    style: str = "dark",
    title: str | None = None,
    max_points: int | None = 500_000,
    in_tissue: bool = True,
    rotate: bool | float = True,
    crop: bool = True,
    invert_y: bool = True,
    sample_key: str = S.LIBRARY,
    x_key: str = S.X,
    y_key: str = S.Y,
    locus_key: str = S.LOCUS,
    clone_key: str = S.CLONE,
) -> InteractiveFigure:
    """Explore sections interactively: switch samples and layers, pick a gene, search clones, select regions.

    One self-contained HTML figure (see :class:`~spaclone.pl.interactive.InteractiveFigure`): the section, a sample
    switcher when several sections are given, a layer switcher (isotype class, switched state, BCR / TCR, chain,
    clones, reads; the CSR and SHM share maps with their hotspots and regions when ``hotspots`` are given; cell type,
    regions, gene expression and UMIs of the transcriptome), a searchable gene picker and a clone search. Drag a box or
    a free shape to count the records (or units) inside: clones, isotype and chain shares, mean reads.

    Parameters
    ----------
    data
        A :class:`~spaclone.io.SpatialAIR` (records and transcriptome), a receptor table (one sample per
        ``sample_key`` value), an AnnData (transcriptome only), a :class:`~spaclone.bcr.HotspotWorkflowResult`, or a
        dict ``{name: one of these}`` / a list of them for several samples (the switcher keeps their order).
    hotspots
        :func:`spaclone.bcr.hotspot_workflow` result(s) adding the CSR and SHM maps, their hotspots and regions: one
        result, or ``{name: result}`` per sample. Nothing is computed here.
    samples
        Names of the samples to show, in this order (default: all).
    color
        Layer shown first (a value of the switcher: ``"isotype"``, ``"switched"``, ``"receptor"``, ``"locus"``,
        ``"clone"``, ``"reads"``, ``"csr"``, ``"shm"``, ``"cell_type"``, ``"region"``, ``"gene"``, ``"umis"``);
        default the isotype class when records carry constant-gene calls, else the chain, else the cell type.
    genes
        Genes the picker offers: a number ``k`` (default 300: the ``k`` most variable genes of the drawn units plus the
        immune marker and immunoglobulin genes found), a list of names, ``"all"``, or ``0`` / ``None`` (no picker).
        Each gene adds its detected units to the file; :attr:`~InteractiveFigure.nbytes` gives the size. See
        :mod:`spaclone.pl.interactive._genes` for the storage and its precision (integer counts exactly).
    layer
        AnnData layer of the gene values (default ``X``).
    cell_type_key
        ``obs`` column of cell types (default ``"cell_type"``, else ``"cell2loc_anno"``, when present).
    region_key
        ``obs`` column of region labels (:func:`spaclone.gex.call_regions`).
    style
        ``"dark"`` (the house style) or ``"light"``.
    title
        Figure title.
    max_points
        Most records (and units) drawn per sample; above it a random subset is drawn and counted, with a note in the
        figure. ``None`` draws everything.
    in_tissue
        Draw only units with ``obs["in_tissue"]`` True (when that column exists).
    rotate, crop, invert_y
        Orientation of the sections, as in every spaclone map.
    sample_key, x_key, y_key, locus_key, clone_key
        Column names.

    Returns
    -------
    InteractiveFigure
        Show it in a notebook, ``.show()`` it, or ``.save("figure.html")``.

    Examples
    --------
    >>> import spaclone
    >>> fig = spaclone.pl.explore(spaclone.datasets.synthetic_example(), genes=20)  # no download
    >>> [c["id"] for c in fig.spec["controls"]]  # layer, gene picker, values (log1p CP10k or counts), clone search
    ['color', 'gene', '_norm', 'clone']
    >>> s = fig.summary(box=(0, 500, 0, 500))  # what a box selection of the figure counts
    >>> int(s.loc[s.quantity == "count", "value"].iloc[0])  # records in that box
    6767

    Several sections in one file, with a sample switcher (AIR-SPACE lymph nodes, downloaded once):

    >>> days = {d: spaclone.io.read_airspace(*spaclone.datasets.airspace(d)) for d in ("D7", "D14", "D21")}  # doctest: +SKIP
    >>> spaclone.pl.explore(days, genes=200).save("airspace.html")  # doctest: +SKIP
    """
    style = check_style(style)
    mode = style
    ch = C.CHROME[mode]
    items = _samples(data, samples, sample_key)
    if hotspots is not None:
        hot_map = (
            dict(hotspots) if isinstance(hotspots, Mapping) else ({items[0][0]: hotspots} if len(items) == 1 else None)
        )
        if hot_map is None:
            raise ValueError("with several samples, pass hotspots={sample name: result}")
        unknown = [k for k in hot_map if k not in {n for n, *_ in items}]
        if unknown:
            raise KeyError(f"hotspots for unknown samples {unknown}")
        items = [(n, a, g, hot_map.get(n, h)) for n, a, g, h in items]
    multi = len(items) > 1
    call = call_text("spaclone.pl.explore", data, hotspots=hotspots, samples=samples, color=color, genes=genes,
                     style=style, title=title, max_points=max_points)  # fmt: skip
    spec = _spec.Spec(style=style, title=title or ("spaclone explorer" if multi else f"spaclone explorer · {items[0][0]}"),
                      call=call)  # fmt: skip
    panel = spec.panel("map", "map", area=[0, 0, 1, 1], invert_y=bool(invert_y), home_by="sample" if multi else None)
    sections: dict[str, Section] = {}
    avail: dict[str, set[str]] = {}
    for name, air, gex, _ in items:
        key = name if multi else None
        sec = Section(spec, panel, key, air, gex, style=style, rotate=rotate, crop=crop, in_tissue=in_tissue,
                      max_points=max_points, x_key=x_key, y_key=y_key, locus_key=locus_key, clone_key=clone_key)  # fmt: skip
        sections[name] = sec
        has = set()
        if sec.n_records:
            a = sec.air
            if locus_key in a.columns:
                has.add("locus")
            if sec.isotypes().notna().any():
                has.add("iso")
            if clone_key in a.columns:
                has.add("clone")
            if S.READS in a.columns and pd.to_numeric(a[S.READS], errors="coerce").notna().any():
                has.add("reads")
        if sec.n_units:
            obs = sec.gex.obs
            ct = cell_type_key or next((c for c in ("cell_type", "cell2loc_anno") if c in obs.columns), None)
            if ct is not None and ct in obs.columns:
                has.add("cell_type")
                sec.ct_key = ct
            if region_key in obs.columns:
                has.add("region")
            if "total_counts" in obs.columns:
                has.add("umis")
            has.add("units")
        avail[name] = has
    for name, *_rest, hot in items:
        if hot is not None:
            avail[name] |= {"csr", "shm"}
    # genes
    unit_sets = [(sec.gex, sec.unit_rows) for sec in sections.values() if sec.n_units]
    gene_names: list[str] = []
    gene_report: dict[str, Any] = {"n_genes": 0}
    if unit_sets and genes not in (None, 0):
        gene_names, gene_report = _genes.select_genes(unit_sets, genes, layer=layer)
        if gene_report.get("missing"):
            logger.warning("explore: genes not in the data: %s", ", ".join(gene_report["missing"][:20]))
    if gene_names:
        for name, sec in sections.items():
            if sec.n_units:
                _genes.encode(spec, name, sec.gex, sec.unit_rows, gene_names, layer=layer, unit_word=sec.unit_word)
                avail[name].add("genes")
    everything = set().union(*avail.values())
    options = [(v, lab) for v, lab, need in LAYERS if need in everything]
    if not options:
        raise ValueError("explore(): nothing to draw (no records with coordinates and no units with obsm['spatial'])")
    values = [v for v, _ in options]
    first = (
        color
        if color is not None
        else next((v for v in ("isotype", "locus", "cell_type", "gene", "umis") if v in values), values[0])
    )
    if first not in values:
        raise ValueError(f"color={first!r} is not available here; choose from {values}")
    rec_modes = [v for v in values if v in ("isotype", "switched", "receptor", "locus", "clone", "reads")]
    map_modes = [v for v in values if v in ("csr", "shm")]
    if multi:
        spec.control("sample", "select", label="sample", options=[{"value": n, "label": n} for n, *_ in items],
                     value=items[0][0])  # fmt: skip
    spec.control(
        "color", "select", label="show", options=[{"value": v, "label": lab} for v, lab in options], value=first
    )
    if "genes" in everything:
        spec.control("gene", "search", label="gene", source={"genes_by": "sample"} if multi else {"genes": items[0][0]},
                     placeholder=f"gene ({len(gene_names):,})", noun="gene", nouns="genes", also={"color": "gene"},
                     value=None)  # fmt: skip
        g0 = next(iter(spec.genes.values()))
        if g0.get("counts"):
            spec.control("_norm", "select", label="values", options=[{"value": "norm", "label": "log1p(counts per 10k)"},
                         {"value": "counts", "label": "counts"}], value="norm", when={"color": "gene"})  # fmt: skip
    if "clone" in everything:
        spec.control("clone", "search", label="clone", source={"by": "sample", "tables": {}, "aux": {}} if multi else None,
                     placeholder="clone id", noun="clone", nouns="clones", multi=True, max=8,
                     colors=list(C.SLOTS[mode]), also={"color": "clone"}, reset_on=["sample"] if multi else None,
                     value=[])  # fmt: skip
    clone_ctl = next((c for c in spec.controls if c["id"] == "clone"), None)
    homes = {}
    notes_when = []
    for name, sec in sections.items():
        key = name if multi else None
        bw = {"sample": key} if multi else {}
        has = avail[name]
        homes[name] = sec.home
        for n_ in sec.notes:
            spec.note(f"{name}: {n_}" if multi else n_, when=bw or None)
        # tissue under everything
        hot = next(h for n, *_r, h in items if n == name)
        hl = HotspotLayers(spec, panel, hot, sec.m, style=style, base_when=bw, key=name) if hot is not None else None
        if sec.n_units:
            sec.units_tissue(when={"color": rec_modes + map_modes} if (rec_modes or map_modes) else None)
        if hl is not None and map_modes:
            hl.tissue(when={"color": map_modes})
            for w in ("csr", "shm"):
                if w in map_modes and hl.share_map(w, when={"color": w}) is None:
                    notes_when.append(
                        {"when": {**bw, "color": w}, "text": f"no {w.upper()} map in this sample's result"}
                    )
            hl.add_edge()
        # records
        if sec.n_records:
            iso = sec.isotypes()
            mode_ = mode
            if "isotype" in values and "iso" in has:
                cols = {n: C.isotype_color(n, mode_) for n in ISO_NAMES} | {"other chain": ch["unassigned"]}
                vals = iso.fillna("other chain").to_numpy(dtype=object)
                sec.records_cat(vals, cols, title="isotype class", names=ISO_NAMES + ["other chain"],
                                when={"color": "isotype"}, hkey="isotype", hidden=["other chain"])  # fmt: skip
            if "switched" in values and "iso" in has:
                sw = np.where(iso.isin(["IgG", "IgA", "IgE"]).fillna(False), "switched (IgG, IgA, IgE)",
                              np.where(iso.isin(["IgM", "IgD"]).fillna(False), "unswitched (IgM, IgD)",
                                       np.where(iso.notna(), "no isotype call", "other chain")))  # fmt: skip
                cols = {"switched (IgG, IgA, IgE)": C.MARKS[mode_]["csr"], "unswitched (IgM, IgD)": C.ISOTYPE_COLORS[mode_]["IgM"],
                        "no isotype call": C.ISOTYPE_COLORS[mode_]["unknown"], "other chain": ch["unassigned"]}  # fmt: skip
                sec.records_cat(sw, cols, title="switched", when={"color": "switched"}, hkey="switched",
                                hidden=["other chain"])  # fmt: skip
            if "receptor" in values and "locus" in has:
                loc = sec.air[locus_key].astype("string").str.upper()
                rec = np.where(
                    loc.isin(S.BCR_LOCI).fillna(False),
                    "BCR",
                    np.where(loc.isin(S.TCR_LOCI).fillna(False), "TCR", "other"),
                )
                cols = dict(C.RECEPTOR_COLORS[mode_]) | {"other": C.NEUTRAL[mode_]}
                sec.records_cat(rec, cols, title="receptor", names=["BCR", "TCR", "other"], when={"color": "receptor"},
                                hkey="receptor")  # fmt: skip
            if "locus" in values and "locus" in has:
                loc = sec.air[locus_key].astype("string").str.upper()
                names = [lo for lo in S.ALL_LOCI if (loc == lo).any()] + (
                    ["other"] if (~loc.isin(S.ALL_LOCI)).any() else []
                )
                cols = C.locus_colors(mode_) | {"other": C.NEUTRAL[mode_]}
                vals = np.where(loc.isin(S.ALL_LOCI).fillna(False), loc.fillna("other"), "other")
                sec.records_cat(vals, cols, title="chain", names=names, when={"color": "locus"}, hkey="locus")
            if "clone" in values and "clone" in has:
                _, tid, aux = sec.records_highlight(control="clone", when={"color": "clone"})
                if multi:
                    clone_ctl["source"]["tables"][name] = tid
                    clone_ctl["source"]["aux"][name] = aux
                else:
                    clone_ctl["source"], clone_ctl["aux"] = tid, aux
                spec.legend.append({"kind": "highlight", "title": "clones", "control": "clone", "colors": list(C.SLOTS[mode]),
                                    "other": "every record", "other_color": ch["muted"], "when": {**bw, "color": "clone"},
                                    "empty": "search a clone id above (the largest clones come first)"})  # fmt: skip
                notes_when.append(
                    {
                        "when": {**bw, "color": "clone"},
                        "text": "search a clone above: up to 8, each in its own colour; links join a clone's units (minimum spanning tree, links up to 250 µm)",
                    }
                )
            if "reads" in values and "reads" in has:
                reads = pd.to_numeric(sec.air[S.READS], errors="coerce").to_numpy(dtype=float)
                sec.records_num(reads, title="reads per record", ramp=C.RAMPS["blue_ordinal"][mode_], scale="log",
                                when={"color": "reads"}, quantiles=(0.0, 0.99))  # fmt: skip
        # overlays on every layer: hotspot outlines, regions
        if hl is not None:
            for w in ("csr", "shm"):
                hl.hotspot_outline(w, okey=f"{w}_hotspots")
            reg, _boxes = hl.regions(okey="regions")
            items_ = [("CSR hotspot", hl.pal["csr"], "line"), ("SHM hotspot", hl.pal["shm"], "line")]
            keys = ["csr_hotspots", "shm_hotspots"]
            if reg is not None:
                items_.append(("region (click to zoom)", hl.pal["csr"], "box"))
                keys.append("regions")
            _spec.legend_lines(spec, "outlines", items_, when=bw or None, layers=keys)
        # units coloured
        if sec.n_units:
            stats_cols = [(c_, lab) for c_, lab in ((getattr(sec, "ct_key", None), "cell type"), (region_key, "region"))
                          if c_ is not None and c_ in sec.gex.obs.columns]  # fmt: skip
            if "cell_type" in values and "cell_type" in has:
                sec.units_cat(
                    sec.ct_key, title="cell type", when={"color": "cell_type"}, stats=stats_cols, hkey="cell_type"
                )
            if "region" in values and "region" in has:
                sec.units_cat(region_key, title="region", when={"color": "region"}, stats=stats_cols, hkey="region")
            if "gene" in values and "genes" in has:
                sec.units_gene(name, when={"color": "gene"}, stats=stats_cols)
            if "umis" in values and "umis" in has:
                tc = pd.to_numeric(sec.gex.obs["total_counts"], errors="coerce").to_numpy(dtype=float)[sec.unit_rows]
                sec.units_num(tc, title="UMIs per unit", ramp=C.RAMPS["blue_ordinal"][mode], scale="log",
                              when={"color": "umis"}, stats=stats_cols, hover_label="UMIs", quantiles=(0.01, 0.99))  # fmt: skip
        for v, lab, need in LAYERS:
            if v in values and need not in has and need not in ("genes",):
                notes_when.append({"when": {**bw, "color": v}, "text": f"no {lab} for this sample"})
            elif v == "gene" and "gene" in values and "genes" not in has:
                notes_when.append({"when": {**bw, "color": v}, "text": "no transcriptome for this sample"})
    if multi:
        panel["home_map"] = homes
    else:
        panel["home"] = next(iter(homes.values()))
    if notes_when:
        panel["notes_when"] = notes_when
    # layout: aspect of the widest section
    asp = [max((h[1] - h[0]) / max(h[3] - h[2], 1e-9), 0.4) for h in homes.values()]
    spec.layout = {"cols": 1, "rows": 1, "aspect": float(np.clip(np.median(asp) * 1.04, 0.75, 3.2)), "height_hint": 640}
    if gene_names:
        spec.note(f"genes: {gene_report.get('rule', '')} · {len(gene_names):,} genes embedded (genes= changes this)")
    subtitle = []
    if multi:
        subtitle.append(f"{len(items)} samples")
    nrec = [s.n_records for s in sections.values() if s.n_records]
    nunit = [s.n_units for s in sections.values() if s.n_units]
    if nrec:
        subtitle.append(
            f"{min(nrec):,}–{max(nrec):,} receptor records"
            if multi and min(nrec) != max(nrec)
            else f"{nrec[0]:,} receptor records"
        )
    if nunit:
        subtitle.append(
            f"{min(nunit):,}–{max(nunit):,} transcriptome units"
            if multi and min(nunit) != max(nunit)
            else f"{nunit[0]:,} transcriptome units"
        )
    spec.subtitle = " · ".join(subtitle)
    return InteractiveFigure(spec, name="spaclone_explorer")
