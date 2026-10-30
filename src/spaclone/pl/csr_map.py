"""Spatial class-switch map: the tissue counterpart of SciCSR's class-switch views on single-cell embeddings.

:func:`csr_map` draws one 16:9 figure of where the heavy-chain isotypes are and how lineages switch:

* one small map per isotype, in the order of the constant genes on the IgH locus (mouse IgM, IgD, IgG3, IgG1,
  IgG2b, IgG2a/c, IgE, IgA; human IgM, IgD, IgG3, IgG1, IgA1, IgG2, IgG4, IgE, IgA2): the smoothed share of loci of
  that isotype among loci with an isotype, from :func:`spaclone.tl.share_map` (one call per isotype, with the kernel,
  grid and tissue of the CSR map);
* the switched share (IgG + IgA + IgE) with its CSR hotspots outlined, as in :func:`hotspot_atlas`;
* the dominant downstream isotype: at each place the switched class (IgG, IgA or IgE) that holds most switched loci,
  brighter where more loci are switched;
* class switches within clones from :func:`spaclone.bcr.csr_transitions`: arcs along the locus from an upstream to a
  downstream isotype found in one clone (width = number of clones), or an upstream x downstream matrix.

:func:`isotype_composition` draws the isotype mix of loci as stacked bars by region (the CSR/SHM overlap regions of a
workflow result), compartment or any other grouping.

Isotype colours are fixed by name (:data:`ISOTYPE_COLORS`, defined in :mod:`spaclone.pl._colors`) and are the house
isotype colours of the whole package (:func:`spaclone.pl.isotype_map`, :func:`spaclone.pl.hotspot_atlas`): the hue
names the class (IgM light blue, IgD blue, IgG red, IgE violet, IgA orange) and, inside IgG (and human IgA), the
lightness names the subclass along the locus, light to dark (an ordinal ramp, always read with labels: panel titles,
bar labels, legend). Checked with the dataviz palette validator: the five classes are told apart for every pair
(normal vision ΔE ≥ 15.7, protan/deutan ΔE ≥ 8.6, in both modes); neighbouring subclasses differ by ΔE ≥ 8 under
simulated colour-vision deficiency. Functions return the Figure (or the Axes) and never call ``show()`` or
``savefig()``.

References
----------
Ng et al. 2023, sciCSR, Nat Methods, doi:10.1038/s41592-023-02060-1 (class-switch recombination from single-cell RNA);
Stavnezer, Guikema & Schrader 2008, doi:10.1146/annurev.immunol.26.021607.090248 (locus order);
Horns et al. 2016, doi:10.7554/eLife.16578 (isotypes within lineages).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spaclone import _schema as S
from spaclone.pl._colors import CHROME, ISOTYPE_COLORS, RAMPS, ink_on, isotype_color, mode_of
from spaclone.pl.atlas import (
    _FS,
    _REF_W,
    _AtlasData,
    _cmap,
    _draw_fill,
    _draw_image,
    _draw_lines,
    _field,
    _Frame,
    _labels_array,
    _mask_fill,
    _mask_lines,
    _nearest_fill,
    _note,
    _opaque_rim,
    _set_view,
    _share_map,
    _stroke,
    _style,
)

__all__ = ["ISOTYPE_COLORS", "csr_map", "isotype_composition"]

logger = logging.getLogger("spaclone")

#: Locus order of the panels (5' to 3'); allelic genes share a position (mouse IgG2a / IgG2c).
_ORDER = {
    "mouse": {
        "subclass": ["IgM", "IgD", "IgG3", "IgG1", "IgG2b", "IgG2a/c", "IgE", "IgA"],
        "class": ["IgM", "IgD", "IgG", "IgE", "IgA"],
    },
    "human": {
        "subclass": ["IgM", "IgD", "IgG3", "IgG1", "IgA1", "IgG2", "IgG4", "IgE", "IgA2"],
        "class": ["IgM", "IgD", "IgG", "IgE", "IgA"],
    },
}
_SWITCHED = ("IgG", "IgA", "IgE")
_GENE = {"IGHM": "IgM", "IGHD": "IgD", "IGHG": "IgG", "IGHA": "IgA", "IGHE": "IgE"}


def _iso_color(name: str, style: str | None) -> str:
    """Fixed colour of an isotype name (class colour for unknown subclasses, grey for anything else)."""
    return isotype_color(name, mode_of(style))


def _norm_name(v: Any) -> Any:
    """``IGHG2B*01`` / ``IGHG2B`` / ``IgG2b`` -> ``IgG2b``; ``IGHG`` -> ``IgG``; missing stays missing."""
    if not isinstance(v, str) or not v.strip() or v.strip().lower() in ("unknown", "nan", "na", "none", "*"):
        return pd.NA
    t = v.strip().split("*")[0]
    if t.upper().startswith("IGH"):
        up = t.upper()
        if up in _GENE:
            return _GENE[up]
        cls = _GENE.get(up[:4])
        if cls is None:
            return pd.NA
        return cls + up[4:].lower()
    return t


# ----------------------------------------------------------------------------------------------- input handling


class _CSRInputs:
    """Units (one row per locus) with class, subclass and switched calls, and the map geometry."""

    def __init__(
        self,
        data: Any,
        *,
        species: str,
        csr: Any,
        grid: Any,
        tissue: Any,
        sigma_um: float | None,
        min_neff: float | None,
        pixel_um: float,
        isotype_key: str | None,
        subclass_key: str | None,
        clone_key: str,
        x_key: str,
        y_key: str,
    ) -> None:
        from spaclone.tl import Grid, Raster

        self.result = None
        if isinstance(data, pd.DataFrame):
            units = data
        else:
            loci = _field(data, "loci")
            if not isinstance(loci, pd.DataFrame):
                air = _field(data, "air")
                if isinstance(air, pd.DataFrame):
                    units = air
                else:
                    raise TypeError("csr_map needs a workflow result (with .loci), a loci table or a receptor table")
            else:
                units = loci
                self.result = data
        units = self._as_loci(units, species=species, clone_key=clone_key, x_key=x_key, y_key=y_key)
        if self.result is not None:
            units = self._subclass_from_records(units, _field(self.result, "records"), species)
        self.csr = csr if csr is not None else _field(self.result, "csr")
        sm = _share_map(self.csr)
        self.grid = grid if grid is not None else _field(self.result, "grid") or (sm.grid if sm is not None else None)
        params = _field(self.result, "params", {}) or {}
        sm_params = getattr(sm, "params", {}) if sm is not None else {}
        self.sigma_um = float(
            sigma_um if sigma_um is not None else sm_params.get("sigma_um", params.get("sigma_um", 50.0))
        )
        self.min_neff = float(min_neff if min_neff is not None else sm_params.get("min_neff", 10.0))
        x = units[x_key].to_numpy(dtype=float)
        y = units[y_key].to_numpy(dtype=float)
        if self.grid is None:
            self.grid = Grid.from_points(x, y, pixel_um=pixel_um)
        t = tissue if tissue is not None else _field(self.result, "tissue")
        if t is None and sm is not None:
            t = np.asarray(sm.eligible, dtype=bool) | (np.nan_to_num(np.asarray(sm.n_eff, dtype=float)) > 0)
        if t is None:  # loci only: the tissue is where the kernel sees enough units
            occ = self.grid.rasterize(x, y)
            r = max(1.0, self.sigma_um / self.grid.pixel_um)
            dens = ndi.gaussian_filter(occ, r) * (2 * np.pi * r * r)
            t = ndi.binary_fill_holes(dens >= max(self.min_neff, 1.0) / 2)
        arr = np.asarray(getattr(t, "values", t), dtype=bool)
        self.tissue = Raster(arr, self.grid, name="tissue")
        self.on_grid = self.grid.index(x, y)[2]
        self.units = units
        self.x_key, self.y_key, self.clone_key = x_key, y_key, clone_key
        self.species = species
        self.isotype_key = isotype_key
        self.subclass_key = subclass_key

    @staticmethod
    def _as_loci(units: pd.DataFrame, *, species: str, clone_key: str, x_key: str, y_key: str) -> pd.DataFrame:
        """Loci from a receptor table (records of one cell merged, isotype called per locus) or the table itself."""
        for c in (x_key, y_key):
            if c not in units.columns:
                raise KeyError(f"csr_map: column {c!r} not in the table")
        has_iso = any(c in units.columns for c in ("isotype_class", "isotype", "isotype_subclass"))
        is_records = S.SPOT in units.columns and S.LOCUS_ID not in units.columns and S.C_CALL in units.columns
        if is_records and not has_iso:
            from spaclone.bcr import call_isotype, isotype
            from spaclone.pp import merge_loci

            air = units
            if S.LOCUS in air.columns:
                air = air[air[S.LOCUS].astype("string") == "IGH"]
            air = air[air[clone_key].notna()]
            loci, member = merge_loci(air, radius_um=15.0, clone_key=clone_key, x_key=x_key, y_key=y_key)
            air = isotype(air.assign(**{S.LOCUS_ID: member.reindex(air.index).to_numpy()}), species=species)
            calls = call_isotype(air, by="locus")
            sub = (
                air.dropna(subset=["isotype_subclass"])
                .groupby([S.LOCUS_ID, "isotype_subclass"])
                .size()
                .rename("n")
                .reset_index()
                .sort_values([S.LOCUS_ID, "n"], ascending=[True, False])
                .drop_duplicates(S.LOCUS_ID)
                .set_index(S.LOCUS_ID)
                .isotype_subclass
            )
            loci = loci.set_index(S.LOCUS_ID, drop=False)
            loci["isotype_class"] = calls.isotype.reindex(loci.index).astype("string")
            loci["isotype_subclass"] = sub.reindex(loci.index).astype("string")
            loci["switched"] = calls.switched.reindex(loci.index).astype("boolean")
            return loci.reset_index(drop=True)
        if not has_iso and S.C_CALL in units.columns:
            from spaclone.bcr import isotype

            return isotype(units, species=species)
        return units

    @staticmethod
    def _subclass_from_records(units: pd.DataFrame, records: Any, species: str) -> pd.DataFrame:
        """Per-locus subclass from the result's records when the loci table has none.

        A locus takes the most frequent subclass among its records (ties: the 5'-most gene on the locus).
        """
        if any(c in units.columns for c in ("isotype_subclass", "subclass", "sub")):
            return units
        if not isinstance(records, pd.DataFrame) or S.LOCUS_ID not in units.columns:
            return units
        if S.LOCUS_ID not in records.columns or "isotype_subclass" not in records.columns:
            return units
        from spaclone.bcr.isotype import CONSTANT_ORDER

        order = CONSTANT_ORDER[species]
        d = pd.DataFrame(
            {
                "locus": records[S.LOCUS_ID].to_numpy(dtype=object),
                "sub": records["isotype_subclass"].astype("string").to_numpy(dtype=object),
            }
        ).dropna()
        d = d[d["sub"].isin(list(order))]
        if not len(d):
            return units
        cnt = d.groupby(["locus", "sub"]).size().rename("n").reset_index()
        cnt["pos"] = cnt["sub"].map(order)
        top = (
            cnt.sort_values(["locus", "n", "pos"], ascending=[True, False, True])
            .drop_duplicates("locus")
            .set_index("locus")["sub"]
        )
        sub = units[S.LOCUS_ID].map(top)
        return units.assign(isotype_subclass=pd.array(sub.to_numpy(dtype=object), dtype="string"))

    def labels(self, level: str) -> tuple[pd.Series, pd.Series, str]:
        """Per unit: the class, the label at the chosen level (NA = no isotype), and the level used."""
        u = self.units
        ck = self.isotype_key or next((c for c in ("isotype_class", "isotype", "iso_main") if c in u.columns), None)
        sk = self.subclass_key or next((c for c in ("isotype_subclass", "subclass", "sub") if c in u.columns), None)
        cls = (
            u[ck].map(_norm_name).astype("string")
            if ck is not None
            else pd.Series(pd.NA, index=u.index, dtype="string")
        )
        cls = cls.where(cls.isin(["IgM", "IgD", "IgG", "IgA", "IgE"]))
        sub = (
            u[sk].map(_norm_name).astype("string")
            if sk is not None
            else pd.Series(pd.NA, index=u.index, dtype="string")
        )
        if ck is None and sk is not None:  # derive the class from the subclass
            cls = sub.str.slice(0, 3).where(sub.str.slice(0, 3).isin(["IgM", "IgD", "IgG", "IgA", "IgE"]))
        order = _ORDER[self.species]["subclass"]
        if level == "auto":
            multi = cls.isin(["IgG"] + (["IgA"] if self.species == "human" else []))
            has_sub = sub.notna() & sub.isin(order + ["IgG2a", "IgG2c"])
            level = "subclass" if multi.any() and (has_sub & multi).sum() >= 0.5 * multi.sum() else "class"
        if level == "class":
            return cls, cls.copy(), "class"
        lab = sub.copy()
        if self.species == "mouse":
            lab = lab.replace({"IgG2a": "IgG2a/c", "IgG2c": "IgG2a/c"})
        single = cls.isin(["IgM", "IgD", "IgE"] + (["IgA"] if self.species == "mouse" else []))
        lab = lab.where(lab.isin(order), pd.NA)
        lab = lab.where(~single | lab.notna(), cls)
        no_sub = lab.isna() & cls.notna()
        lab = lab.where(~no_sub, cls + " (no subclass)")
        return cls, lab.astype("string"), "subclass"


# --------------------------------------------------------------------------------------------------- the maps


def _panel_names(lab: pd.Series, species: str, level: str, min_loci: int) -> list[str]:
    counts = lab.value_counts()
    order = _ORDER[species][level]
    names: list[str] = []
    for name in order:
        cls = name[:3]
        extra = f"{cls} (no subclass)"
        if level == "subclass" and extra in counts.index and extra not in names and counts[extra] >= min_loci:
            names.append(extra)
        if counts.get(name, 0) >= min_loci:
            names.append(name)
    return names


def _ramp(color: str, style: str | None) -> list[str]:
    """One-hue ramp of an isotype colour: near-background to the colour to a light tint (dark style)."""
    from matplotlib.colors import to_hex, to_rgb

    c = np.array(to_rgb(color))
    if style == "dark":
        bg, tip = np.zeros(3), np.ones(3)
        stops = [0.86 * bg + 0.14 * c, 0.45 * bg + 0.55 * c, c, 0.45 * c + 0.55 * tip]
    else:
        bg, tip = np.ones(3), np.zeros(3)
        stops = [0.9 * bg + 0.1 * c, 0.45 * bg + 0.55 * c, c, 0.55 * c + 0.45 * tip]
    return [to_hex(np.clip(s, 0, 1)) for s in stops]


def _share_rgba(sm: Any, tissue: np.ndarray, ramp: Sequence[str]) -> tuple[np.ndarray, float, float]:
    """RGBA image of a share map on its eligible pixels (2nd-99.5th percentile colour limits).

    The image is opaque one pixel beyond the eligible pixels; draw it clipped to :func:`_eligible_fill`.
    """
    from matplotlib.colors import Normalize

    vals = np.asarray(sm.share, dtype=float)
    elig = np.asarray(sm.eligible, dtype=bool) & tissue & np.isfinite(vals)
    fin = vals[elig]
    lo, hi = np.quantile(fin, [0.02, 0.995]) if len(fin) else (0.0, 1.0)
    hi = float(hi) if hi > lo else float(lo) + 1e-9
    rgba = _cmap(ramp, "spaclone_csr_iso")(Normalize(float(lo), hi)(_nearest_fill(vals, elig)))
    rgba[..., 3] = _opaque_rim(elig)
    return rgba, float(lo), hi


def _eligible_fill(sm: Any, tissue: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Smoothed outline of a share map's eligible pixels (the clip of its image)."""
    vals = np.asarray(sm.share, dtype=float)
    elig = np.asarray(sm.eligible, dtype=bool) & tissue & np.isfinite(vals)
    return _mask_fill(elig, sm.grid, 0.7)


def _share_maps(inp: _CSRInputs, lab: pd.Series, names: Sequence[str]) -> dict[str, Any]:
    """One :func:`spaclone.tl.share_map` per isotype: the share of loci of that isotype among loci with one."""
    from spaclone.tl import share_map

    ok = lab.notna().to_numpy() & inp.on_grid
    tab = pd.DataFrame(
        {
            S.X: inp.units[inp.x_key].to_numpy(dtype=float)[ok],
            S.Y: inp.units[inp.y_key].to_numpy(dtype=float)[ok],
        }
    )
    lv = lab.to_numpy(dtype=object)[ok]
    out = {}
    for name in names:
        mark = lv == name
        if not mark.any() or mark.all():
            out[name] = None
            continue
        try:
            out[name] = share_map(
                tab,
                mark=mark,
                sigma_um=inp.sigma_um,
                grid=inp.grid,
                tissue=inp.tissue,
                min_neff=inp.min_neff,
            )
        except ValueError as err:  # a mark the kernel cannot vary (e.g. all units of one isotype)
            logger.info("csr_map: no share map for %s (%s)", name, err)
            out[name] = None
    return out


def _dominant_rgba(
    class_maps: Mapping[str, Any], tissue: np.ndarray, style: str | None, pal: Mapping[str, Any]
) -> tuple[np.ndarray | None, list[str]]:
    """Hue = switched class with the largest smoothed share, lightness = switched share (2nd-99.5th percentile)."""
    from matplotlib.colors import to_rgb

    present = [c for c in _SWITCHED if class_maps.get(c) is not None]
    if not present:
        return None, []
    stack = np.stack([np.nan_to_num(np.asarray(class_maps[c].share, dtype=float)) for c in present])
    elig = np.zeros(tissue.shape, dtype=bool)
    for c in present:
        elig |= np.asarray(class_maps[c].eligible, dtype=bool)
    elig &= tissue
    total = stack.sum(0)
    dom = stack.argmax(0)
    fin = total[elig]
    lo, hi = np.quantile(fin, [0.02, 0.995]) if len(fin) else (0.0, 1.0)
    t = np.clip((total - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.8
    cols = np.stack([np.array(to_rgb(_iso_color(c, style))) for c in present])
    rgb_dom = cols[dom]
    if style == "dark":
        base = np.array(to_rgb(CHROME["dark"]["floor"]))
        tip = 0.75 * rgb_dom + 0.25
        rgb = np.where(
            t[..., None] < 0.75,
            base + (rgb_dom - base) * (t[..., None] / 0.75),
            rgb_dom + (tip - rgb_dom) * ((t[..., None] - 0.75) / 0.25),
        )
    else:
        base = np.array(to_rgb(CHROME["light"]["floor"]))
        rgb = base + (rgb_dom - base) * t[..., None]
    rgba = np.zeros((*tissue.shape, 4))
    for k in range(3):
        rgba[..., k] = _nearest_fill(rgb[..., k], elig)
    rgba[..., 3] = _opaque_rim(elig)
    return rgba, present


# ---------------------------------------------------------------------------------------------- transitions


def _transitions(inp: _CSRInputs, cls: pd.Series, lab: pd.Series, level: str, n_perm: int, rng: Any) -> Any:
    from spaclone.bcr import csr_transitions

    if inp.clone_key not in inp.units.columns:
        raise KeyError(f"csr_map: clone column {inp.clone_key!r} not in the loci table")
    if level == "subclass":
        raw = lab.astype("string").replace({"IgG2a/c": "IgG2c"})
        key_vals = raw.where(~raw.str.contains("no subclass", regex=False).fillna(False))
    else:
        key_vals = cls
    tab = pd.DataFrame(
        {
            inp.clone_key: inp.units[inp.clone_key].to_numpy(dtype=object),
            "_csr_label": key_vals.to_numpy(dtype=object),
        }
    ).dropna()
    return csr_transitions(
        tab,
        clone_key=inp.clone_key,
        species=inp.species,
        level=level if level in ("subclass", "class") else "class",
        isotype_key="_csr_label",
        unit_key=None,
        n_perm=n_perm,
        rng=rng,
    )


def _arc_panel(
    ax: Any,
    table: pd.DataFrame,
    nodes: Sequence[str],
    counts: Mapping[str, int],
    style: str | None,
    pal: Mapping[str, Any],
    s: float,
    alpha_q: float = 0.05,
) -> list[Any]:
    """Arcs along the locus from upstream to downstream isotypes found in one clone (width = number of clones)."""
    from matplotlib.patches import FancyArrowPatch

    ax.set_facecolor(pal["bg"])
    ax.set_axis_off()
    pos = {n: k for k, n in enumerate(nodes)}
    k = len(nodes)
    ax.set_xlim(-0.65, k - 0.35)
    ax.set_ylim(-1.0, 3.2)
    texts: list[Any] = []
    ax.plot([0, k - 1], [0, 0], color=pal["muted"], lw=0.8 * s, zorder=1, solid_capstyle="round")
    t = table[table.n_clones_both > 0].copy()
    t["_u"] = t.upstream.map(lambda v: "IgG2a/c" if v in ("IgG2a", "IgG2c") else v)
    t["_d"] = t.downstream.map(lambda v: "IgG2a/c" if v in ("IgG2a", "IgG2c") else v)
    t = t[t._u.isin(pos) & t._d.isin(pos) & (t._u != t._d)]
    rad = -0.85
    if len(t):
        t = t.groupby(["_u", "_d"], as_index=False).agg(n=("n_clones_both", "sum"), q=("q_value", "min"))
        nmax = float(t.n.max())
        for r in t.sort_values("n").itertuples():
            x0, x1 = pos[r._1], pos[r._2]
            lw = (0.8 + 6.5 * np.sqrt(r.n / nmax)) * s
            sig = np.isfinite(r.q) and r.q <= alpha_q
            ax.add_patch(
                FancyArrowPatch(
                    (x0, 0.08),
                    (x1, 0.08),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle=f"-|>,head_length={0.45 + 0.1 * lw / s},head_width={0.25 + 0.06 * lw / s}",
                    mutation_scale=6 * s,
                    lw=lw,
                    color=_iso_color(r._2, style),
                    alpha=0.95 if sig else 0.55,
                    zorder=2 + r.n / nmax,
                    shrinkA=3 * s,
                    shrinkB=3 * s,
                )
            )
        scale = _arc_scale(ax)
        for r in t.sort_values("n", ascending=False).head(3).itertuples():
            x0, x1 = pos[r._1], pos[r._2]
            h = 0.5 * abs(rad) * abs(x1 - x0) * scale
            texts.append(
                ax.text((x0 + x1) / 2, 0.08 + h, f"{int(r.n):,}", ha="center", va="center", color=pal["fg"],
                        fontsize=_FS["small"] * s, zorder=10, path_effects=_stroke(pal, 3.0 * s))
            )  # fmt: skip
    vmax = max(counts.values()) if counts and max(counts.values()) > 0 else 1
    for n in nodes:
        c = counts.get(n, 0)
        x = pos[n]
        size = (40 + 260 * np.sqrt(c / vmax)) * s**2 if c else 40 * s**2
        ax.scatter([x], [0], s=size, c=_iso_color(n, style) if c else pal["bg"],
                   edgecolors=pal["bg"] if c else pal["muted"], linewidths=(2.0 if c else 1.0) * s, zorder=6)  # fmt: skip
        y = -0.32 if x % 2 == 0 else -0.62
        texts.append(
            ax.text(x, y, n.replace(" (no subclass)", "*"), ha="center", va="top", color=pal["fg"] if c else pal["muted"],
                    fontsize=_FS["small"] * s, fontweight="bold" if c else "normal", zorder=7)
        )  # fmt: skip
    return texts


def _arc_scale(ax: Any) -> float:
    """Data-y per data-x in display space (arcs are drawn in display coordinates)."""
    ax.apply_aspect()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (abs(y1 - y0) / bb.height) / (abs(x1 - x0) / bb.width)


def _matrix_panel(
    ax: Any,
    table: pd.DataFrame,
    nodes: Sequence[str],
    style: str | None,
    pal: Mapping[str, Any],
    s: float,
    alpha_q: float = 0.05,
) -> list[Any]:
    """Bubble matrix in sciCSR's flux-matrix convention: from (upstream) on x, to (downstream) on y.

    Upper triangle only; colour = clones holding both isotypes, size = significance against random labelling
    (-log10 q, capped at q = 0.01).
    """
    from matplotlib.colors import Normalize

    t = table.copy()
    t["upstream"] = t.upstream.map(lambda v: "IgG2a/c" if v in ("IgG2a", "IgG2c") else v)
    t["downstream"] = t.downstream.map(lambda v: "IgG2a/c" if v in ("IgG2a", "IgG2c") else v)
    t = t.groupby(["upstream", "downstream"], as_index=False).agg(n=("n_clones_both", "sum"), q=("q_value", "min"))
    names = [n for n in nodes if n in set(t.upstream) | set(t.downstream)]
    idx = {n: i for i, n in enumerate(names)}
    t = t[t.upstream.isin(idx) & t.downstream.isin(idx)]
    t = t[[idx[u] < idx[d] for u, d in zip(t.upstream, t.downstream, strict=True)]]
    k = len(names)
    cmap = _cmap(RAMPS["blue"][mode_of(style)], "spaclone_csr_flux")
    nmax = float(t.n.max()) if len(t) and t.n.max() > 0 else 1.0
    norm = Normalize(0, np.sqrt(nmax))
    q = np.clip(np.nan_to_num(t.q.to_numpy(dtype=float), nan=1.0), 0.01, 1.0)
    sig = -np.log10(q) / 2.0  # 0 (q = 1) .. 1 (q <= 0.01)
    x = [idx[u] for u in t.upstream]
    y = [idx[d] for d in t.downstream]
    ax.set_facecolor(pal["bg"])
    for v in range(k):
        ax.axhline(v, color=pal["tissue"], lw=0.6 * s, zorder=0)
        ax.axvline(v, color=pal["tissue"], lw=0.6 * s, zorder=0)
    ax.scatter(x, y, s=(30 + 300 * sig) * s**2, c=cmap(norm(np.sqrt(t.n.to_numpy(dtype=float)))),
               edgecolors=pal["bg"], linewidths=1.5 * s, zorder=3)  # fmt: skip
    texts = []
    for xi, yi, ni in zip(x, y, t.n.to_numpy(), strict=True):
        texts.append(
            ax.text(xi + 0.22, yi + 0.22, f"{int(ni):,}", ha="left", va="bottom", fontsize=_FS["small"] * 0.85 * s,
                    color=pal["fg"], zorder=4, path_effects=_stroke(pal, 2.0 * s))
        )  # fmt: skip
    from matplotlib.lines import Line2D

    keys = [(1.0, "q = 1"), (0.1, "q = 0.1"), (0.01, "q \u2264 0.01")]  # size key, in the empty upper triangle
    hh = [Line2D([], [], ls="", marker="o", ms=np.sqrt((30 + 300 * (-np.log10(qv) / 2)) * s**2), color=pal["muted"],
                 label=lab_) for qv, lab_ in keys]  # fmt: skip
    leg = ax.legend(handles=hh, loc="upper right", frameon=False, fontsize=_FS["small"] * 0.9 * s, labelcolor=pal["fg"],
                    handletextpad=0.5, labelspacing=1.5, borderaxespad=0.2)  # fmt: skip
    leg.set_zorder(20)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylim(-0.6, k - 0.4)
    ax.set_xticks(range(k), names, rotation=90, ha="center", va="bottom", fontsize=_FS["small"] * s,
                  color=pal["fg"])  # fmt: skip
    ax.xaxis.tick_top()
    ax.set_yticks(range(k), names, fontsize=_FS["small"] * s, color=pal["fg"])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("from (upstream, 5')", color=pal["muted"], fontsize=_FS["small"] * s)
    ax.set_ylabel("to (downstream, 3')", color=pal["muted"], fontsize=_FS["small"] * s)
    return texts


# --------------------------------------------------------------------------------------------------- layout


def _grid_layout(
    n_maps: int, sec_aspect: float, W: float, H: float, left_w: float, title_in: float, gap_in: float
) -> tuple[int, int, float, float]:
    """Columns and rows of the map grid and the map width / height (inches) that make maps largest."""
    best = (1, n_maps, 0.0, 0.0)
    for ncol in range(1, n_maps + 1):
        nrow = int(np.ceil(n_maps / ncol))
        w = (left_w * W - (ncol + 1) * gap_in) / ncol
        h = w / sec_aspect
        if nrow * (h + title_in + gap_in) > H * 0.98:
            h = H * 0.98 / nrow - title_in - gap_in
            w = h * sec_aspect
        if w > best[2] + 1e-9:
            best = (ncol, nrow, w, h)
    return best


def _fmt_pct(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0%"
    return f"{v:.1%}" if v < 0.01 else f"{v:.0%}"


def _fmt_range(lo: float, hi: float) -> tuple[str, str]:
    d = 1 if hi < 0.095 else 0
    return f"{lo * 100:.{d}f}", f"{hi * 100:.{d}f}%"


def _title_band(
    ax: Any,
    title: str,
    pal: Mapping[str, Any],
    s: float,
    *,
    swatch: str | None = None,
    ramp: Sequence[str] | None = None,
    lo: float | None = None,
    hi: float | None = None,
    keys: Sequence[tuple[str, str]] = (),
) -> list[Any]:
    """Title band above a map; returns its texts (for the overlap check).

    The title is bold, on the left, after an optional colour dot; on the right sits either a mini colour ramp with
    its limits or colour keys.
    """
    fig = ax.figure
    pos = ax.get_position(original=True)
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    fs = _FS["label"] * 0.92 * s
    y = 1.0 + 0.07 / h_in
    out = []
    x = 0.0
    if swatch is not None:
        ax.scatter([0.075 / w_in], [y + 0.045 / h_in], s=42 * s**2, c=swatch, transform=ax.transAxes,
                   clip_on=False, lw=0, zorder=10)  # fmt: skip
        x = 0.17 / w_in
    title_text = ax.text(x, y, title, transform=ax.transAxes, ha="left", va="bottom", color=pal["fg"], fontsize=fs,
                         fontweight="bold", clip_on=False)  # fmt: skip
    title_text._spaclone_right = 1.0  # left edge of the right-hand keys, in axes fraction (set below)
    out.append(title_text)
    small = _FS["small"] * 0.95 * s
    if ramp is not None and lo is not None and hi is not None:
        lo_t, hi_t = _fmt_range(lo, hi)
        bar_w, bar_h = 0.55 * s / w_in, 0.075 * s / h_in
        hi_w = (0.075 * len(hi_t) * s + 0.05) / w_in
        x1 = 1.0 - hi_w
        _ramp_strip(ax, ramp, (x1 - bar_w, y + 0.035 / h_in, bar_w, bar_h))
        lo_text = ax.text(x1 - bar_w - 0.04 / w_in, y, lo_t, transform=ax.transAxes, ha="right", va="bottom",
                          color=pal["muted"], fontsize=small, clip_on=False)  # fmt: skip
        out.append(lo_text)
        title_text._spaclone_right = x1 - bar_w - (0.04 + 0.075 * len(lo_t) * s) / w_in
        out.append(ax.text(x1 + 0.04 / w_in, y, hi_t, transform=ax.transAxes, ha="left", va="bottom",
                           color=pal["muted"], fontsize=small, clip_on=False))  # fmt: skip
    if keys:
        xr = 1.0
        for label, col in reversed(list(keys)):
            t = ax.text(xr, y, label, transform=ax.transAxes, ha="right", va="bottom", color=pal["fg"], fontsize=small,
                        clip_on=False)  # fmt: skip
            out.append(t)
            xr -= (0.075 * len(label) * s + 0.06) / w_in
            ax.scatter([xr - 0.02 / w_in], [y + 0.045 / h_in], s=30 * s**2, marker="s", c=col, transform=ax.transAxes,
                       clip_on=False, lw=0, zorder=10)  # fmt: skip
            xr -= (0.17 * s) / w_in
        title_text._spaclone_right = xr
    return out


def _ramp_strip(ax: Any, ramp: Sequence[str], rect: tuple[float, float, float, float], n: int = 48) -> None:
    """A colour ramp drawn as ``n`` touching cells in axes coordinates (no extra Axes)."""
    from matplotlib.collections import PolyCollection

    x0, y0, w, h = rect
    xs = x0 + w * np.arange(n + 1) / n
    verts = [[(xs[k], y0), (xs[k + 1], y0), (xs[k + 1], y0 + h), (xs[k], y0 + h)] for k in range(n)]
    cols = _cmap(ramp, "spaclone_csr_key")((np.arange(n) + 0.5) / n)
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors=cols, linewidths=0.3,
                                     transform=ax.transAxes, clip_on=False, zorder=10))  # fmt: skip


def _fit_band(fig: Any, texts: Sequence[Any], min_scale: float = 0.7) -> None:
    """Give every map title one common size that clears the keys on its right.

    Titles with a shorter alternative switch to it first; a title that still does not fit at the smallest size is
    written on two lines, and truncated only when even that does not fit.
    """
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    except (AttributeError, NotImplementedError, RuntimeError):  # a canvas without a renderer: keep the sizes
        return
    bands = []
    for t in texts:
        if t.axes is None or getattr(t, "_spaclone_right", None) is None:
            continue
        ab = t.axes.get_window_extent(r)
        others = [o for o in texts if o.axes is t.axes and o is not t]
        right = min([o.get_window_extent(r).x0 for o in others] + [ab.x0 + t._spaclone_right * ab.width])
        bands.append((t, min(right, ab.x1) - 0.1 * fig.dpi))
    if not bands:
        return

    def need(t: Any, right: float) -> float:
        bb = t.get_window_extent(r)
        return (right - bb.x0) / max(bb.x1 - bb.x0, 1e-9)

    for t, right in bands:  # shorter wording where the full title would need shrinking
        for alt in getattr(t, "_spaclone_alts", ()):
            if need(t, right) >= 1:
                break
            t.set_text(alt)
    scale = min(1.0, max(min_scale, min(need(t, right) for t, right in bands)))
    for t, _ in bands:
        t.set_fontsize(t.get_fontsize() * scale)
    for t, right in bands:
        if need(t, right) < 1:  # two lines (the band grows upwards) before any truncation
            words = t.get_text().split(" ")
            for k in sorted(range(1, len(words)), key=lambda k: abs(2 * k - len(words))):
                t.set_text(" ".join(words[:k]) + "\n" + " ".join(words[k:]))
                if need(t, right) >= 1:
                    break
            else:
                t.set_text(" ".join(words))
        for _ in range(40):
            if need(t, right) >= 1 or len(t.get_text()) <= 6:
                break
            t.set_text(t.get_text()[:-2].rstrip() + "\u2026")


def _composition_table(lab: pd.Series, groups: pd.Series, names: Sequence[str], order: Sequence[str] | None):
    keep = lab.notna().to_numpy() & groups.notna().to_numpy()
    ct = pd.crosstab(groups[keep].astype(str), lab[keep].astype(str)).reindex(columns=list(names), fill_value=0)
    idx = list(order) if order is not None else list(dict.fromkeys(groups[keep].astype(str)))
    ct = ct.reindex([g for g in idx if g in ct.index]).fillna(0)
    return ct


def _draw_bars(
    ax: Any,
    ct: pd.DataFrame,
    names: Sequence[str],
    style: str | None,
    pal: Mapping[str, Any],
    s: float,
    *,
    legend: bool,
    xlabel: bool = True,
) -> None:
    """Stacked horizontal bars of shares (rows of ``ct``), segments in locus order with a background-coloured gap."""
    n = ct.sum(axis=1)
    share = ct.div(n.replace(0, np.nan), axis=0).fillna(0)
    ax.set_facecolor(pal["bg"])
    width_in = ax.get_position(original=True).width * ax.figure.get_figwidth()
    fs_in = _FS["small"] * s / 72
    y = np.arange(len(share))[::-1].astype(float)
    left = np.zeros(len(share))
    for name in names:
        w = share[name].to_numpy()
        col = _iso_color(name, style)
        ax.barh(y, w, left=left, height=0.62, color=col, edgecolor=pal["bg"], linewidth=1.5 * s, label=name)
        ink = ink_on(col)
        for yi, li, wi in zip(y, left, w, strict=True):
            label = f"{wi:.0%}"
            # only where the label fits inside its segment with a margin (otherwise legend + n carry it)
            if wi * width_in >= 0.62 * fs_in * len(label) + 0.08:
                ax.text(li + wi / 2, yi, label, ha="center", va="center", fontsize=_FS["small"] * s, color=ink)
        left += w
    for yi, ni in zip(y, n.to_numpy(), strict=True):
        ax.text(1.015, yi, f"n = {int(ni):,}", ha="left", va="center", fontsize=_FS["small"] * s, color=pal["muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(share) - 0.4)
    ax.set_yticks(y, list(share.index), fontsize=_FS["legend"] * s, color=pal["fg"])
    ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"], fontsize=_FS["small"] * s, color=pal["muted"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if xlabel:
        ax.set_xlabel("share of loci with an isotype (5' → 3')", color=pal["muted"], fontsize=_FS["small"] * s)
    if legend and names:
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=min(len(names), 4), frameon=False,
                        fontsize=_FS["small"] * s, labelcolor=pal["fg"], handlelength=1.0, handletextpad=0.4,
                        columnspacing=1.0, borderaxespad=0.2)  # fmt: skip
        leg.set_zorder(20)


def _locus_nodes(names: Sequence[str], species: str) -> list[str]:
    """Locus positions to draw: every expressed gene of the species, plus the "(no subclass)" panels present."""
    out = []
    for n in _ORDER[species]["subclass"]:
        extra = f"{n[:3]} (no subclass)"
        if extra in names and extra not in out:
            out.append(extra)
        out.append(n)
    return out


# --------------------------------------------------------------------------------------------------- public


def csr_map(
    data: Any,
    *,
    species: Literal["mouse", "human"] = "mouse",
    level: Literal["auto", "subclass", "class"] = "auto",
    style: str | None = "dark",
    figsize: tuple[float, float] = (16.0, 9.0),
    dpi: float = 200,
    transitions: Literal["arcs", "matrix"] | None = "arcs",
    transition_test: Any = None,
    n_perm: int = 199,
    rng: Any = 0,
    composition: Any = "auto",
    csr: Any = None,
    grid: Any = None,
    tissue: Any = None,
    sigma_um: float | None = None,
    min_neff: float | None = None,
    pixel_um: float = 10.0,
    min_loci: int = 1,
    rotate: bool | float = True,
    invert_y: bool = True,
    isotype_key: str | None = None,
    subclass_key: str | None = None,
    clone_key: str = S.CLONE,
    x_key: str = S.X,
    y_key: str = S.Y,
    notes: bool = False,
) -> Any:
    """Isotype maps along the IgH constant locus, switched share, dominant switched class and switches within clones.

    The spatial counterpart of SciCSR's class-switch views on single-cell embeddings.

    Parameters
    ----------
    data
        A :class:`spaclone.bcr.HotspotWorkflowResult` (``loci``, ``grid``, ``tissue``, ``csr``, ``regions``,
        ``params``; the subclass of a locus comes from its ``records`` when the loci table has none), a loci table (``x_um``, ``y_um``, ``clone_id``, ``isotype_class`` and/or ``isotype_subclass``,
        ``switched``), or a receptor table with ``c_call`` (records are merged into loci with
        :func:`spaclone.pp.merge_loci`, 15 µm, and called with :func:`spaclone.bcr.call_isotype`).
    species
        IgH locus order of the panels and of the transitions.
    level
        ``"subclass"`` (constant genes: IgG3, IgG1, ...), ``"class"`` (IgM, IgD, IgG, IgE, IgA) or ``"auto"``
        (subclass when at least half of the IgG loci, and for human the IgA loci, carry a subclass).
    style
        ``"dark"`` (house style) or ``"light"`` / None.
    figsize, dpi
        Figure size (inches) and resolution.
    transitions
        ``"arcs"`` (arcs along the locus), ``"matrix"`` (bubble matrix in sciCSR's flux-matrix layout) or None.
    transition_test
        A :func:`spaclone.bcr.csr_transitions` result to draw; by default the test is run here on the loci
        (``unit_key=None``: one row per cell; ``n_perm`` shuffles; ``rng``).
    n_perm, rng
        Shuffles and seed of that test (only used when ``transition_test`` is None).
    composition
        Stacked bars of the isotype mix under the transitions: ``"auto"`` (by region for a workflow result with
        regions, else by a ``compartment`` column when present, else none), None, or any ``by`` of
        :func:`isotype_composition`.
    csr
        CSR hotspots (:class:`~spaclone.tl.HotspotResult`) to outline; default ``data.csr``. Without it the switched
        share is mapped without hotspots.
    grid, tissue
        Grid and tissue of the maps; default those of the result (or of ``csr``), else a 10-µm grid over the loci
        and the area where the kernel sees enough loci.
    sigma_um, min_neff
        Kernel and eligibility of the isotype maps; default those of the CSR map (``csr.map.params``), else 50 µm
        and 10 effective loci.
    pixel_um
        Pixel of a grid made here.
    min_loci
        Isotypes with fewer loci get no panel (they stay on the locus diagram as hollow nodes).
    rotate, invert_y
        As in :func:`hotspot_atlas`.
    isotype_key, subclass_key
        Class and subclass columns (default ``isotype_class`` / ``isotype``, ``isotype_subclass``); names such as
        ``IGHG2B`` or ``IgG2b`` are accepted.
    clone_key, x_key, y_key
        Clone and coordinate columns.
    notes
        Also print the reading notes (what hue, brightness, arc width and opacity encode) in the figure. Off by
        default: figures carry key labels only; the encodings are described here and in the docs.

    Returns
    -------
    matplotlib.figure.Figure
        Axes in order: the switched-share map, the dominant switched class map, one map per isotype (locus order),
        then the transitions panel and the composition bars (when drawn). The colour key of each map (its 2nd and
        99.5th percentile) is drawn above it.

    Notes
    -----
    Every map is a kernel share (Nadaraya–Watson) over loci with the same Gaussian, grid and eligibility; the colour
    scale of each runs from its 2nd to its 99.5th percentile on eligible pixels (key above the map). A locus with
    an IgG call but no subclass is shown in its own panel ("IgG (no subclass)") at subclass level. In the
    transitions panel an arc from an upstream to a downstream isotype counts the clones holding loci of both
    (switching runs 5' to 3' only, so the downstream isotype is the later state); arcs whose count exceeds random
    labelling of cells (``q <= 0.05``) are opaque.

    References
    ----------
    Ng et al. 2023, doi:10.1038/s41592-023-02060-1; Kelsall & Diggle 1998, doi:10.1111/1467-9876.00128; Horns et al.
    2016, doi:10.7554/eLife.16578.
    """
    import matplotlib.pyplot as plt

    if transitions not in ("arcs", "matrix", None):
        raise ValueError("transitions must be 'arcs', 'matrix' or None")
    if level not in ("auto", "subclass", "class"):
        raise ValueError("level must be 'auto', 'subclass' or 'class'")
    if species not in _ORDER:
        raise ValueError(f"species must be one of {sorted(_ORDER)}")
    pal = _style(style)
    inp = _CSRInputs(
        data,
        species=species,
        csr=csr,
        grid=grid,
        tissue=tissue,
        sigma_um=sigma_um,
        min_neff=min_neff,
        pixel_um=pixel_um,
        isotype_key=isotype_key,
        subclass_key=subclass_key,
        clone_key=clone_key,
        x_key=x_key,
        y_key=y_key,
    )
    cls, lab, lvl = inp.labels(level)
    names = _panel_names(lab, species, lvl, min_loci)
    iso_maps = _share_maps(inp, lab, names)
    class_maps = (
        _share_maps(inp, cls, list(_SWITCHED)) if lvl == "subclass" else {c: iso_maps.get(c) for c in _SWITCHED}
    )
    tissue_arr = np.asarray(inp.tissue.values, dtype=bool)
    frame = _Frame(inp.grid, tissue_arr, rotate)
    fill = _mask_fill(tissue_arr, inp.grid, 1.5)
    edge = _mask_lines(tissue_arr, inp.grid, 1.5)
    box = frame.section_box(tissue_arr)
    sec_aspect = (box[1] - box[0]) / max(box[3] - box[2], 1e-9)
    sm_sw = _switched_map(inp, cls)
    hot = _labels_array(inp.csr, inp.grid) if inp.csr is not None else None
    hot_lines = _mask_lines(hot > 0, inp.grid, 0.7) if hot is not None else []

    test, test_err = transition_test, None
    if transitions is not None and test is None:
        try:
            test = _transitions(inp, cls, lab, lvl, n_perm, rng)
        except (ValueError, KeyError) as err:
            test_err = str(err)
    by = _auto_composition(inp, composition)

    W, H = figsize
    s = W / _REF_W
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=pal["bg"])
    side = transitions is not None or by is not None
    left_w = 0.70 if side else 1.0
    title_in, gap_in = 0.34 * s, 0.10 * s
    n_maps = 2 + len(names)
    ncol, nrow, mw, mh = _grid_layout(n_maps, sec_aspect, W, H, left_w - 0.005, title_in, gap_in)
    block_w = ncol * mw + (ncol - 1) * gap_in
    block_h = nrow * (mh + title_in) + (nrow - 1) * gap_in
    x_start = (left_w * W - block_w) / 2
    y_top = H - max(0.06 * s, (H - block_h) / 2)
    texts: list[Any] = []
    axes = []
    panels = [("switched", None), ("dominant", None)] + [("iso", n) for n in names]
    for k, (kind, name) in enumerate(panels):
        r, c = divmod(k, ncol)
        x = x_start + c * (mw + gap_in)
        y = y_top - (r + 1) * (mh + title_in) - r * gap_in
        ax = fig.add_axes((x / W, y / H, mw / W, mh / H))
        axes.append(ax)
        ax.set_facecolor(pal["bg"])
        _set_view(ax, box, invert_y)
        _draw_fill(ax, frame, fill, pal["tissue"], z=-1)
        if kind == "switched":
            if sm_sw is not None:
                rgba, lo, hi = _share_rgba(sm_sw, tissue_arr, pal["csr_ramp"])
                _draw_image(ax, frame, inp.grid, rgba, z=0, clip=_eligible_fill(sm_sw, tissue_arr))
                texts += _title_band(ax, "Switched share", pal, s, ramp=pal["csr_ramp"], lo=lo, hi=hi)
            else:
                texts += _title_band(ax, "Switched share", pal, s)
            _draw_lines(ax, frame, hot_lines, pal["csr"], 1.1 * s, 5)
        elif kind == "dominant":
            rgba, present = _dominant_rgba(class_maps, tissue_arr, style, pal)
            if rgba is not None:
                ref = next(class_maps[k] for k in present)
                _draw_image(ax, frame, inp.grid, rgba, z=0, clip=_eligible_fill(ref, tissue_arr))
            _draw_lines(ax, frame, hot_lines, pal["csr"], 0.8 * s, 5)
            keys = [(c, _iso_color(c, style)) for c in present]
            band = _title_band(ax, "Dominant switched class", pal, s, keys=keys)
            band[0]._spaclone_alts = ("Dominant class",)  # then two lines, never a bare "Dominant"
            texts += band
        else:
            sm = iso_maps.get(name)
            n_iso = int((lab == name).sum())
            share = float(n_iso) / max(int(lab.notna().sum()), 1)
            title = f"{name} · {_fmt_pct(share)} of loci"
            color = _iso_color(name, style)
            ramp = _ramp(color, style)
            rgba, lo, hi = _share_rgba(sm, tissue_arr, ramp) if sm is not None else (None, 0.0, 0.0)
            if rgba is not None and hi - lo > 1e-6:
                _draw_image(ax, frame, inp.grid, rgba, z=0, clip=_eligible_fill(sm, tissue_arr))
                band = _title_band(ax, title, pal, s, swatch=color, ramp=ramp, lo=lo, hi=hi)
            else:  # no loci, or too few for a share that varies: say so instead of a 0–0% scale
                band = _title_band(ax, title, pal, s, swatch=color)
                _note(ax, "no loci" if n_iso == 0 else "too few loci for a map", pal, _FS["small"] * s)
            band[0]._spaclone_alts = (f"{name} · {_fmt_pct(share)}", name)
            texts += band
        _draw_lines(ax, frame, edge, pal["tissue_edge"], 0.8 * s, 4)
        _set_view(ax, box, invert_y)
    if side:
        texts += _side_column(fig, axes, inp, cls, lab, lvl, names, test, test_err, transitions, by, style, pal, s,
                              hot_lines=bool(hot_lines), notes=notes)  # fmt: skip
    _fit_band(fig, texts)
    _keep_inside(fig, [t for t in texts if t.axes is None])
    return fig


def _keep_inside(fig: Any, texts: Sequence[Any], min_scale: float = 0.75) -> None:
    """Shrink (then truncate) figure-level texts that would cross the right edge of the figure."""
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    except (AttributeError, NotImplementedError, RuntimeError):  # a canvas without a renderer: keep the sizes
        return
    right = fig.bbox.x1 - 0.06 * fig.dpi
    for t in texts:
        fs0 = t.get_fontsize()
        for _ in range(40):
            if t.get_window_extent(r).x1 <= right or len(t.get_text()) <= 6:
                break
            if t.get_fontsize() > fs0 * min_scale:
                t.set_fontsize(t.get_fontsize() * 0.95)
            else:
                t.set_text(t.get_text()[:-2].rstrip() + "\u2026")


def _switched_map(inp: _CSRInputs, cls: pd.Series) -> Any:
    """The CSR map of the result when it is on the same grid, else a share map of the switched calls."""
    from spaclone.tl import share_map

    sm = _share_map(inp.csr)
    if sm is not None and getattr(sm, "grid", None) == inp.grid:
        return sm
    u = inp.units
    sw = u["switched"] if "switched" in u.columns else cls.isin(list(_SWITCHED)).where(cls.notna())
    sw = pd.array(sw, dtype="boolean")
    ok = ~np.asarray(pd.isna(sw)) & inp.on_grid
    mark = np.asarray(sw[ok], dtype=bool)
    if not mark.any() or mark.all():
        return None
    tab = pd.DataFrame({S.X: u[inp.x_key].to_numpy(dtype=float)[ok], S.Y: u[inp.y_key].to_numpy(dtype=float)[ok]})
    return share_map(tab, mark=mark, sigma_um=inp.sigma_um, grid=inp.grid, tissue=inp.tissue, min_neff=inp.min_neff)


def _auto_composition(inp: _CSRInputs, composition: Any) -> Any:
    if composition is None:
        return None
    if isinstance(composition, str) and composition == "auto":
        reg = _field(inp.result, "regions")
        if isinstance(reg, pd.DataFrame) and len(reg):
            return "region"
        for c in ("compartment", "comp"):
            if c in inp.units.columns:
                return c
        return None
    return composition


def _side_column(
    fig: Any,
    axes: list[Any],
    inp: _CSRInputs,
    cls: pd.Series,
    lab: pd.Series,
    lvl: str,
    names: Sequence[str],
    test: Any,
    test_err: str | None,
    transitions: str | None,
    by: Any,
    style: str | None,
    pal: Mapping[str, Any],
    s: float,
    *,
    hot_lines: bool,
    notes: bool = False,
) -> list[Any]:
    """Right-hand column, top down: class switches within clones, the isotype mix by group, then the numbers.

    Blocks are stacked with a cursor (figure fraction); the composition bars take only the height left above the
    numbers and are left out when there is no room for them.
    """
    x0, x1 = 0.715, 0.99
    line_h = 0.028
    small = _FS["small"] * 0.92 * s
    texts: list[Any] = []
    # the numbers at the bottom (their top is the floor of the blocks above)
    n_iso = int(cls.notna().sum())
    sw = int((cls.isin(list(_SWITCHED)) & cls.notna()).sum())
    lines = [f"{n_iso:,} loci with an isotype \u00b7 {sw / max(n_iso, 1):.0%} switched"]
    if test is not None and test.summary.get("n_clones_multi") is not None:
        lines.append(f"{int(test.summary['n_clones_multi']):,} clones hold loci of \u2265 2 isotypes")
    if notes:
        lines.append("dominant class: hue = most frequent switched class,")
        lines.append(("  brightness" if style == "dark" else "  colour strength") + " = switched share")
    if hot_lines:
        lines.append("yellow outline: CSR hotspot")
    floor = 0.04 + len(lines) * line_h + 0.02
    for k, ln in enumerate(lines):
        texts.append(fig.text(x0, 0.04 + (len(lines) - 1 - k) * line_h, ln, color=pal["fg"] if k == 0 else pal["muted"],
                              fontsize=small, ha="left", va="bottom"))  # fmt: skip
    y = 0.975
    if transitions is not None:
        texts.append(fig.text(x0, y, "Class switches within clones", color=pal["fg"], fontsize=_FS["label"] * s,
                              fontweight="bold", ha="left", va="top"))  # fmt: skip
        texts.append(fig.text(x0, y - 0.03, "IgH constant locus, 5' \u2192 3'", color=pal["muted"],
                              fontsize=_FS["small"] * s, ha="left", va="top"))  # fmt: skip
        y -= 0.05
        nodes = _ORDER[inp.species][lvl] if lvl == "class" else _locus_nodes(names, inp.species)
        counts = {n: int((lab == n).sum()) for n in nodes}
        if transitions == "arcs":
            h = 0.31
            ax = fig.add_axes((x0, y - h, x1 - x0, h))
            y -= h
        else:  # room above for the vertical "from" labels and below for the axis title
            h = 0.25
            ax = fig.add_axes((x0 + 0.06, y - 0.075 - h, x1 - x0 - 0.075, h))
            y -= 0.075 + h + 0.035
        axes.append(ax)
        cap: list[str] = []
        if test is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no class switches to test" + (f"\n({test_err})" if test_err else ""),
                    transform=ax.transAxes, ha="center", va="center", color=pal["muted"], fontsize=_FS["small"] * s)  # fmt: skip
        elif transitions == "arcs":
            texts += _arc_panel(ax, test.table, nodes, counts, style, pal, s)
            cap = ["arc: upstream \u2192 downstream isotype in one clone; width = clones",
                   "opaque: more clones than random labelling of cells (q \u2264 0.05)"]  # fmt: skip
        else:
            texts += _matrix_panel(ax, test.table, nodes, style, pal, s)
            cap = ["colour: clones holding both isotypes",
                   "size: significance against random labelling of cells (-log10 q)"]  # fmt: skip
        y -= 0.012
        for ln in cap if notes else []:
            texts.append(fig.text(x0, y, ln, color=pal["muted"], fontsize=small, ha="left", va="top"))
            y -= line_h
    if by is not None:
        try:
            groups, order = _groups(inp.result if inp.result is not None else inp.units, inp, by, True, inp.x_key,
                                    inp.y_key)  # fmt: skip
        except (KeyError, ValueError) as err:
            logger.info("csr_map: no composition bars (%s)", err)
            groups = None
        ct = None
        if groups is not None:
            ct = _composition_table(lab, groups, names, order)
            ct = ct[ct.sum(axis=1) > 0]
        head_y = y - 0.03
        legend_h = 0.03 * int(np.ceil(len(names) / 4)) + 0.015
        room = head_y - 0.035 - legend_h - (floor + 0.04)  # 0.04: the 0 / 50% / 100% labels under the bars
        if ct is not None and len(ct) and room >= 0.03 * min(len(ct), 2):
            head = "Isotype mix by region" if isinstance(by, str) and by == "region" else "Isotype mix"
            texts.append(fig.text(x0, head_y, head, color=pal["fg"], fontsize=_FS["label"] * 0.92 * s,
                                  fontweight="bold", ha="left", va="top"))  # fmt: skip
            h = min(0.045 * len(ct), room)
            top = head_y - 0.035 - legend_h
            bx = fig.add_axes((x0 + 0.075, top - h, x1 - x0 - 0.135, h))
            axes.append(bx)
            _draw_bars(bx, ct, names, style, pal, s * 0.92, legend=True, xlabel=False)
        elif ct is not None and len(ct):
            logger.info("csr_map: no room for the composition bars; draw them with pl.isotype_composition")
    return texts


def isotype_composition(
    data: Any,
    *,
    by: Any = "region",
    species: Literal["mouse", "human"] = "mouse",
    level: Literal["auto", "subclass", "class"] = "auto",
    ax: Any = None,
    style: str | None = "dark",
    figsize: tuple[float, float] | None = None,
    dpi: float = 200,
    outside: bool = True,
    min_loci: int = 1,
    legend: bool = True,
    isotype_key: str | None = None,
    subclass_key: str | None = None,
    x_key: str = S.X,
    y_key: str = S.Y,
) -> Any:
    """Isotype mix of loci as stacked bars, one bar per region, compartment or group, segments in locus order.

    Parameters
    ----------
    data
        A workflow result (``loci`` and, for ``by="region"``, ``regions`` and the CSR hotspot labels) or a loci
        table.
    by
        ``"region"`` (the CSR/SHM overlap regions of the result, plus the loci outside every region with
        ``outside=True``), a column of the loci table (e.g. ``"compartment"``), a categorical
        :class:`~spaclone.tl.Raster` sampled at the loci (e.g. :func:`spaclone.tl.compartments`), or one value per
        locus.
    species, level, isotype_key, subclass_key
        As in :func:`csr_map`.
    ax
        Axes to draw on; a new figure if None.
    style
        ``"dark"`` or ``"light"`` / None.
    figsize, dpi
        Size and resolution of a new figure (default 9 in wide, 0.55 in per bar).
    outside
        ``by="region"``: add a bar for the loci outside every region.
    min_loci
        Groups with fewer loci with an isotype are left out.
    legend
        Isotype legend above the bars.
    x_key, y_key
        Coordinate columns (``by="region"`` or a raster).

    Returns
    -------
    matplotlib.axes.Axes
        Bars from 0 to 100%; the number of loci with an isotype is written after each bar, and each share inside
        its segment when the label fits there (the legend names the rest).
    """
    import matplotlib.pyplot as plt

    pal = _style(style)
    inp = _CSRInputs(
        data, species=species, csr=None, grid=None, tissue=None, sigma_um=None, min_neff=None, pixel_um=10.0,
        isotype_key=isotype_key, subclass_key=subclass_key, clone_key=S.CLONE, x_key=x_key, y_key=y_key,
    )  # fmt: skip
    _, lab, lvl = inp.labels(level)
    groups, order = _groups(data, inp, by, outside, x_key, y_key)
    names = _panel_names(lab, species, lvl, 1)
    ct = _composition_table(lab, groups, names, order)
    ct = ct[ct.sum(axis=1) >= max(min_loci, 1)]
    if ax is None:
        fs = figsize or (9.0, 1.3 + 0.55 * max(len(ct), 1))
        fig, ax = plt.subplots(figsize=fs, dpi=dpi, facecolor=pal["bg"])
        fig.subplots_adjust(left=0.19, right=0.86, top=1 - 0.75 / fs[1] if legend else 0.95, bottom=0.55 / fs[1])
    s = min(ax.figure.get_figwidth() / 9.0, 1.4)
    _draw_bars(ax, ct, names, style, pal, s, legend=legend)
    return ax


def _groups(data: Any, inp: _CSRInputs, by: Any, outside: bool, x_key: str, y_key: str) -> tuple[pd.Series, list]:
    """Group label of every locus (NA = left out) and the display order of the groups."""
    u = inp.units
    idx = u.index
    if isinstance(by, str) and by == "region":
        if inp.result is None:
            raise ValueError("by='region' needs a workflow result with regions")
        ad = _AtlasData(inp.result, style="dark", rotate=False, invert_y=True, x_key=x_key, y_key=y_key)
        g = pd.Series(pd.NA, index=idx, dtype="string")
        i, j, ins = ad.grid.index(u[x_key].to_numpy(dtype=float), u[y_key].to_numpy(dtype=float))
        assigned = np.zeros(len(u), dtype=bool)
        order = []
        for rid in sorted(ad.regions.index, key=lambda r: (str(type(r)), r)):
            m = ad.region_mask(rid)
            hit = ins & m[i, j] & ~assigned
            g[hit] = f"region {rid}"
            order.append(f"region {rid}")
            assigned |= hit
        if outside:
            g[ins & ~assigned & ad.tissue[i, j]] = "outside regions"
            order.append("outside regions")
        return g, order
    if isinstance(by, str):
        if by not in u.columns:
            raise KeyError(f"isotype_composition: column {by!r} not in the loci table")
        col = u[by]
        if isinstance(col.dtype, pd.CategoricalDtype):
            return col.astype("string"), [str(c) for c in col.cat.categories]
        g = col.astype("string")
        return g, sorted(pd.unique(g.dropna()), key=lambda v: str(v).lower())
    if hasattr(by, "sample_categories") and getattr(by, "categories", None) is not None:
        vals = by.sample_categories(u[x_key].to_numpy(dtype=float), u[y_key].to_numpy(dtype=float), fill="outside")
        return pd.Series(vals, index=idx, dtype="string").replace("outside", pd.NA), list(by.categories)
    arr = np.asarray(by, dtype=object)
    if arr.shape != (len(u),):
        raise ValueError("by must be 'region', a column name, a categorical Raster or one value per locus")
    g = pd.Series(arr, index=idx, dtype="string")
    return g, sorted(pd.unique(g.dropna()), key=lambda v: str(v).lower())
