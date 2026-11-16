"""One section in an interactive map: its frame, the transcriptome units, the receptor records and their layers.

:class:`Section` adds layers for one sample to a map panel: units (beads, bins, spots) as tissue or coloured by a
cell type, a region label, a gene or a number; receptor records coloured by receptor family, chain, isotype class,
switched state, a highlighted clone or a number; with hover text, legends and selection statistics. Colours come from
:mod:`spaclone.pl._colors` (the colours of the static maps). Every layer of a sample carries ``when={sample: key}`` so
that the sample switcher shows one section at a time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from spaclone import _schema as S
from spaclone._utils import counts_desc
from spaclone.pl import _colors as C
from spaclone.pl.interactive import _encode as E
from spaclone.pl.interactive import _geo as G
from spaclone.pl.interactive import _spec as P

#: Isotype classes in locus order, then "unknown" (a heavy chain without a usable constant call).
ISO_NAMES = ["IgM", "IgD", "IgG", "IgE", "IgA", "unknown"]
#: Most categories drawn in their own colours for a free categorical column (the static maps' rule).
MAX_CATEGORIES = 8


def _when(base: Mapping[str, Any] | None, **kw: Any) -> dict[str, Any]:
    out = dict(base or {})
    out.update({k: v for k, v in kw.items() if v is not None})
    return out


def isotype_classes(air: pd.DataFrame, c_key: str = S.C_CALL, locus_key: str = S.LOCUS) -> pd.Series:
    """Isotype class of every heavy-chain record (missing for other chains), from an ``isotype``-like column or C calls."""
    for col in ("isotype_class", "isotype"):
        if col in air.columns:
            s = air[col].astype("string")
            s = s.replace({"IGHM": "IgM", "IGHD": "IgD", "IGHG": "IgG", "IGHA": "IgA", "IGHE": "IgE"})
            cls = s.str.extract(r"^(IgM|IgD|IgG|IgE|IgA|unknown)", expand=False)
            if cls.notna().any():
                return cls.astype("string").reset_index(drop=True)
    if c_key not in air.columns:
        return pd.Series(pd.NA, index=range(len(air)), dtype="string")
    from spaclone.bcr.isotype import isotype

    sub = air[[c_key] + ([locus_key] if locus_key in air.columns else [])]
    return (
        isotype(sub, species=None, c_key=c_key, locus_key=locus_key)["isotype"].astype("string").reset_index(drop=True)
    )


class Section:
    """Layers of one sample on a map panel.

    Parameters
    ----------
    spec, panel
        The figure and the map panel.
    key
        Sample key (state value of the sample switcher); ``None`` for a figure of one section without a switcher.
    air
        Receptor records (``x_um``, ``y_um``; ``locus``, ``clone_id``, ``c_call`` ... when present) or None.
    gex
        Spot-level AnnData (``obsm["spatial"]`` in µm) or None.
    frame_points
        ``"units"`` (the drawn units fix the orientation; the records when there are none) or an ``(n, 2)`` array.
    """

    def __init__(
        self,
        spec: P.Spec,
        panel: dict[str, Any],
        key: str | None,
        air: pd.DataFrame | None,
        gex: Any,
        *,
        style: str = "dark",
        rotate: bool | float = True,
        crop: bool = True,
        in_tissue: bool = True,
        max_points: int | None = 500_000,
        frame_points: Any = "units",
        x_key: str = S.X,
        y_key: str = S.Y,
        locus_key: str = S.LOCUS,
        clone_key: str = S.CLONE,
        spatial_key: str = "spatial",
        unit_shape: str | None = None,
    ) -> None:
        self.spec, self.panel, self.key = spec, panel, key
        self.mode = "dark" if style == "dark" else "light"
        self.ch = C.CHROME[self.mode]
        self.base_when = {"sample": key} if key is not None else {}
        self.x_key, self.y_key, self.locus_key, self.clone_key = x_key, y_key, locus_key, clone_key
        self.notes: list[str] = []
        self.unit_word = "units"
        # ---- units
        self.gex = None
        self.unit_rows = np.zeros(0, dtype=np.int64)
        uxy = np.zeros((0, 2))
        if gex is not None and spatial_key in getattr(gex, "obsm", {}):
            xy = np.asarray(gex.obsm[spatial_key], dtype=float)[:, :2]
            keep = np.isfinite(xy).all(axis=1)
            if in_tissue and "in_tissue" in gex.obs.columns:
                tis = pd.array(gex.obs["in_tissue"], dtype="boolean").fillna(False).to_numpy(dtype=bool)
                n_off = int((keep & ~tis).sum())
                keep &= tis
                if n_off:
                    self.notes.append(f"{n_off:,} units outside the tissue are not drawn (in_tissue=False shows them)")
            rows = np.flatnonzero(keep)
            sub = G.thin(len(rows), max_points, seed=1)
            if sub is not None:
                self.notes.append(
                    f"units: a random {len(sub):,} of {len(rows):,} are drawn (max_points=None draws all)"
                )
                rows = rows[sub]
            self.gex = gex
            self.unit_rows = rows
            uxy = xy[rows]
            tech = str((getattr(gex, "uns", {}) or {}).get("spaclone", {}).get("technology", "")).lower()
            sq = (
                any(t in tech for t in ("stereo", "visium hd", "bin")) if unit_shape is None else unit_shape == "square"
            )
            self.unit_shape = "square" if sq else "circle"
            self.unit_word = (
                "bins" if sq else ("beads" if "curio" in tech or "slide" in tech or "air-space" in tech else "units")
            )
        elif gex is not None:
            self.notes.append(f"the transcriptome has no obsm[{spatial_key!r}]: units are not drawn")
        # ---- records
        self.air = None
        rxy = np.zeros((0, 2))
        if air is not None and len(air):
            for c in (x_key, y_key):
                if c not in air.columns:
                    raise KeyError(f"column {c!r} not in the receptor table (pass x_key / y_key)")
            a = air.reset_index(drop=True)
            xy = a[[x_key, y_key]].to_numpy(dtype=float)
            ok = np.isfinite(xy).all(axis=1)
            if (~ok).any():
                self.notes.append(f"{int((~ok).sum()):,} records without coordinates are not drawn")
            idx = np.flatnonzero(ok)
            sub = G.thin(len(idx), max_points, seed=0)
            if sub is not None:
                self.notes.append(
                    f"records: a random {len(sub):,} of {len(idx):,} are drawn and counted in selections "
                    "(max_points=None draws all)"
                )
                idx = idx[sub]
            self.air = a.iloc[idx].reset_index(drop=True)
            rxy = xy[idx]
        # ---- frame and display coordinates
        if isinstance(frame_points, str):
            fp = uxy if len(uxy) else rxy
        else:
            fp = np.asarray(frame_points, dtype=float).reshape(-1, 2)
        self.frame = G.frame_for(fp, rotate, crop)
        self.m = G.affine(self.frame)
        self.uxy = G.apply(self.m, uxy)
        self.rxy = G.apply(self.m, rxy)
        allp = self.uxy if len(self.uxy) else self.rxy
        self.home = G.home_box(allp, self.frame if len(fp) else None)
        self.spacing = G.spacing_um(self.uxy) if len(self.uxy) else (G.spacing_um(self.rxy) if len(self.rxy) else 10.0)
        self._uids: tuple[str, str] | None = None
        self._rids: tuple[str, str] | None = None
        self._rhover: dict[str, Any] | None = None
        self._uhover: dict[str, Any] | None = None
        fkey = key if key is not None else "_"
        spec.frames[fkey] = G.inverse(self.frame)

    # ---- shared blocks
    @property
    def n_units(self) -> int:
        return len(self.uxy)

    @property
    def n_records(self) -> int:
        return len(self.rxy)

    def unit_ids(self) -> tuple[str, str]:
        if self._uids is None:
            self._uids = self.spec.xy(self.uxy)
        return self._uids

    def record_ids(self) -> tuple[str, str]:
        if self._rids is None:
            self._rids = self.spec.xy(self.rxy)
        return self._rids

    def unit_size(self) -> dict[str, float]:
        return {"um": float(self.spacing) * (1.0 if self.unit_shape == "square" else 1.05), "min": 0.9, "max": 120.0}

    def record_size(self) -> dict[str, float]:
        s = max(self.spacing * 0.9, 3.0) if self.n_units else max(self.spacing, 3.0)
        return {"um": float(s), "min": 2.2 if self.n_records < 20000 else 1.6, "max": 60.0}

    def add(self, layer: dict[str, Any]) -> dict[str, Any]:
        self.panel["layers"].append(layer)
        return layer

    # ---- hover
    def record_hover(self) -> dict[str, Any]:
        if self._rhover is not None:
            return self._rhover
        a, sp = self.air, self.spec
        rows = []
        title = None
        if self.clone_key in a.columns:
            title = P.hover_cat(sp, "clone " + a[self.clone_key].astype("string").fillna("NA").astype(str), "clone")
        if self.locus_key in a.columns:
            rows.append(P.hover_cat(sp, a[self.locus_key], "chain"))
        iso = self.isotypes()
        if iso.notna().any():
            sub = a["isotype_subclass"] if "isotype_subclass" in a.columns else None
            show = sub.astype("string").fillna(iso) if sub is not None else iso
            rows.append(P.hover_cat(sp, show.to_numpy(dtype=object), "isotype"))
        for col, lab in ((S.V_CALL, "V"), (S.J_CALL, "J")):
            if col in a.columns:
                rows.append(P.hover_cat(sp, a[col].astype("string").str.split(",").str[0], lab))
        if S.JUNCTION_AA in a.columns and len(a) <= 200_000:
            rows.append(P.hover_cat(sp, a[S.JUNCTION_AA], "CDR3"))
        for col, lab in (("shm_mutations", "V mutations"), ("mutations", "V mutations")):
            if col in a.columns:
                rows.append(P.hover_num(sp, pd.to_numeric(a[col], errors="coerce"), lab))
                break
        if S.READS in a.columns:
            rows.append(P.hover_num(sp, pd.to_numeric(a[S.READS], errors="coerce"), "reads"))
        if S.SPOT in a.columns and len(a) <= 200_000:
            rows.append(P.hover_cat(sp, a[S.SPOT], "spot"))
        self._rhover = {"title": title, "rows": rows} if title else {"rows": rows}
        return self._rhover

    def unit_hover(self, extra: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
        if self._uhover is None:
            g, sp, r = self.gex, self.spec, self.unit_rows
            rows = []
            for col, lab in (("cell_type", "cell type"), ("cell2loc_anno", "cell type"), ("region", "region"),
                             ("region_id", "region instance")):  # fmt: skip
                if col in g.obs.columns:
                    rows.append(P.hover_cat(sp, g.obs[col].to_numpy(dtype=object)[r], lab))
            if "total_counts" in g.obs.columns:
                rows.append(
                    P.hover_num(sp, pd.to_numeric(g.obs["total_counts"], errors="coerce").to_numpy()[r], "UMIs")
                )
            title = P.hover_cat(
                sp, np.asarray(g.obs_names)[r], self.unit_word[:-1] if self.unit_word.endswith("s") else "unit"
            )
            self._uhover = {"title": title, "rows": rows}
        out = dict(self._uhover)
        if extra:
            out["rows"] = list(extra) + list(self._uhover["rows"])
        return out

    # ---- records
    def isotypes(self) -> pd.Series:
        if not hasattr(self, "_iso"):
            self._iso = (
                isotype_classes(self.air, locus_key=self.locus_key)
                if self.air is not None
                else pd.Series([], dtype="string")
            )
        return self._iso

    def record_stats(self) -> str:
        sid = f"rec:{self.key if self.key is not None else '_'}"
        if sid in self.spec.stats:
            return sid
        a, sp = self.air, self.spec
        rows: list[dict[str, Any]] = []
        if self.clone_key in a.columns:
            cid, names, codes = sp.codes(a[self.clone_key])
            rows.append({"kind": "distinct", "label": "clones", "codes": cid, "skip": len(names)})
        iso = self.isotypes()
        if iso.notna().any():
            cid, names, _ = sp.codes(iso.to_numpy(dtype=object), ISO_NAMES)
            cols = [C.isotype_color(n, self.mode) for n in names]
            rows.append({"kind": "shares", "label": "isotype (heavy chains)", "codes": cid, "names": names,
                         "colors": cols, "skip": [names.index("unknown")]})  # fmt: skip
        if self.locus_key in a.columns:
            loci = [lo for lo in S.ALL_LOCI if lo in set(a[self.locus_key].astype(str))]
            cid, names, _ = sp.codes(a[self.locus_key].astype("string").str.upper(), loci)
            lc = C.locus_colors(self.mode)
            rows.append({"kind": "shares", "label": "chain", "codes": cid, "names": names,
                         "colors": [lc.get(n, C.NEUTRAL[self.mode]) for n in names]})  # fmt: skip
        if S.READS in a.columns:
            rows.append({"kind": "mean", "label": "reads per record (mean)",
                         "values": sp.blob(pd.to_numeric(a[S.READS], errors="coerce").to_numpy(dtype=float), "f4"),
                         "fmt": None})  # fmt: skip
        self.spec.stats[sid] = {"unit": "records", "rows": rows}
        return sid

    def records_cat(
        self,
        values: Any,
        colors: Mapping[str, str],
        *,
        title: str,
        names: Sequence[str] | None = None,
        when: Mapping[str, Any] | None = None,
        hkey: str | None = None,
        hidden: Sequence[str] = (),
        alpha: float = 0.9,
        legend: bool = True,
        stack: bool = True,
        lid: str | None = None,
    ) -> dict[str, Any]:
        """Records coloured by a categorical value (large groups drawn underneath)."""
        enc, names, codes = P.cat_color(self.spec, values, colors, names=names)
        counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
        order = None
        if stack:  # large groups underneath, small ones stay visible; missing values first
            rank = np.argsort(np.argsort(-counts, kind="stable"), kind="stable")
            prio = np.where(codes < len(names), len(names) - rank[np.minimum(codes, len(names) - 1)], -1)
            order = np.argsort(prio, kind="stable")
        lay = P.points(self.spec, None, xy_ids=self.record_ids(), color=enc, size=self.record_size(), alpha=alpha,
                       hover=self.record_hover(), when=_when(self.base_when, **(when or {})), name=title,
                       select=self.record_stats(), order=order, lid=lid, hkey=hkey)  # fmt: skip
        self.add(lay)
        k = hkey or lay["id"]
        for h in hidden:
            if h in names:
                entry = {"layer": k, "cat": names.index(h)}
                if entry not in self.spec.hidden:
                    self.spec.hidden.append(entry)
        if legend:
            P.legend_cat(self.spec, title, names, enc["colors"], counts, layer=k, when=lay.get("when"))
        return lay

    def records_highlight(
        self, *, control: str = "clone", when: Mapping[str, Any] | None = None, title: str = "clone", links: bool = True,
        max_link_um: float | None = 250.0,
    ) -> tuple[dict[str, Any], str, str]:  # fmt: skip
        """Records of chosen clones in the categorical slots, the others grey; clone links (minimum spanning tree)."""
        a, sp = self.air, self.spec
        heavy = (
            a[self.locus_key].astype("string").str.upper().isin(S.HEAVY_LOCI).fillna(False).to_numpy()
            if self.locus_key in a.columns
            else np.ones(len(a), bool)
        )
        ids = a[self.clone_key].astype("string")
        key_cols = [S.SPOT, self.clone_key] if S.SPOT in a.columns else [self.x_key, self.y_key, self.clone_key]
        pool = a[heavy] if heavy.any() else a
        size = counts_desc(pool.drop_duplicates(key_cols)[self.clone_key].astype("string"))  # ties: first seen first
        rest = counts_desc(ids[~ids.isin(size.index)].dropna())
        ranked = list(size.index) + list(rest.index)
        n_rec = ids.value_counts()
        loc = a[self.locus_key].astype("string") if self.locus_key in a.columns else pd.Series("", index=a.index)
        main_locus = (
            pd.DataFrame({"c": ids, "l": loc}).dropna().groupby("c")["l"].agg(lambda s: counts_desc(s).index[0])
        )
        aux = [f"{main_locus.get(c, '')} · {int(size.get(c, 0)):,} spots · {int(n_rec.get(c, 0)):,} records".lstrip(" ·")
               for c in ranked]  # fmt: skip
        tid = sp.table(ranked)
        pos = {c: i for i, c in enumerate(ranked)}
        codes = np.asarray(
            pd.Index(ranked, dtype=object).get_indexer(pd.Index(ids.astype(object), dtype=object)), dtype=np.int64
        )
        codes[codes < 0] = len(ranked)
        del pos
        cid = sp.blob(codes, E.code_dtype(len(ranked)))
        slots = list(C.SLOTS[self.mode])
        enc = {"kind": "highlight", "by": control, "codes": cid, "table": tid, "colors": slots}
        w = _when(self.base_when, **(when or {}))
        rs = self.record_size()
        # every record small and grey underneath (as the static clone map), the chosen clones larger on top
        bg = P.points(sp, None, xy_ids=self.record_ids(), color={"kind": "const", "color": self.ch["muted"]},
                      size={"um": rs["um"] * 0.6, "min": 1.0, "max": rs["max"] * 0.6}, alpha=0.5, hover=self.record_hover(),
                      when=w, name="every record", select=self.record_stats(), okey="others")  # fmt: skip
        self.add(bg)
        lay = P.points(sp, None, xy_ids=self.record_ids(), color=enc, size={"um": rs["um"] * 1.5, "min": rs["min"] * 1.9,
                       "max": rs["max"] * 1.2}, hover=self.record_hover(), when=w, name=title,
                       needs_any=control)  # fmt: skip
        self.add(lay)
        if links:
            self.add({"id": sp.layer_id(), "type": "links", "x": self.record_ids()[0], "y": self.record_ids()[1],
                      "n": self.n_records, "codes": cid, "table": tid, "by": control, "colors": slots, "width": 0.9,
                      "alpha": 0.85, "max_um": max_link_um, "when": w})  # fmt: skip
        return lay, tid, sp.table(aux)

    def records_num(
        self, values: Any, *, title: str, ramp: Sequence[str], scale: str = "lin", fmt: str | None = None,
        when: Mapping[str, Any] | None = None, domain: tuple[float, float] | None = None,
        quantiles: tuple[float, float] = (0.01, 0.99), note: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        enc, dom = P.num_color(self.spec, values, ramp, scale=scale, domain=domain, quantiles=quantiles,
                               nan=self.ch["unassigned"])  # fmt: skip
        lay = P.points(self.spec, None, xy_ids=self.record_ids(), color=enc, size=self.record_size(),
                       hover=self.record_hover(), when=_when(self.base_when, **(when or {})), name=title,
                       select=self.record_stats())  # fmt: skip
        self.add(lay)
        P.legend_colorbar(self.spec, title, ramp, dom, fmt=fmt, scale=scale, when=lay.get("when"), note=note)
        return lay

    # ---- units
    def unit_stats(self, cat_cols: Sequence[tuple[str, str]] = ()) -> str:
        sid = f"unit:{self.key if self.key is not None else '_'}"
        if sid in self.spec.stats:
            return sid
        g, sp, r = self.gex, self.spec, self.unit_rows
        rows: list[dict[str, Any]] = []
        for col, lab in cat_cols:
            vals = g.obs[col].to_numpy(dtype=object)[r]
            names = C.vocabulary(g.obs[col])
            cid, names, _ = sp.codes(vals, names)
            cols = self.cat_colors(names, column=col)
            rows.append(
                {"kind": "shares", "label": lab, "codes": cid, "names": names, "colors": [cols[n] for n in names]}
            )
        if "total_counts" in g.obs.columns:
            rows.append({"kind": "mean", "label": "UMIs per unit (mean)",
                         "values": sp.blob(pd.to_numeric(g.obs["total_counts"], errors="coerce").to_numpy(dtype=float)[r], "f4")})  # fmt: skip
        self.spec.stats[sid] = {"unit": self.unit_word, "rows": rows}
        return sid

    def cat_colors(self, names: Sequence[str], *, column: str | None = None) -> dict[str, str]:
        named = C.named_colors(names, self.mode, column=column)
        grey = {n: self.ch["unassigned"] for n in names if n in C.GREY_LABELS}
        return C.fixed_colors(names, mode=self.mode, named=named, neutral=grey)

    def units_tissue(self, *, when: Mapping[str, Any] | None = None, color: str | None = None) -> dict[str, Any]:
        """Units drawn as tissue (one neutral colour) under the records."""
        lay = P.points(self.spec, None, xy_ids=self.unit_ids(), color={"kind": "const", "color": color or self.ch["tissue_strong"]},
                       size=self.unit_size(), shape=self.unit_shape, when=_when(self.base_when, **(when or {})),
                       name="tissue", hover=False)  # fmt: skip
        return self.add(lay)

    def units_cat(
        self, column: str, *, title: str, when: Mapping[str, Any] | None = None, stats: Sequence[tuple[str, str]] = (),
        hkey: str | None = None, values: Any = None, extra_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        g, r = self.gex, self.unit_rows
        col = g.obs[column]
        vocab = C.vocabulary(col)
        vals = col.to_numpy(dtype=object)[r] if values is None else np.asarray(values, dtype=object)[r]
        if extra_names:
            vocab = list(dict.fromkeys([*map(str, extra_names), *vocab]))
        present = set(pd.Series(vals).dropna().astype(str))
        names = [v for v in vocab if v in present]
        if len(names) > MAX_CATEGORIES * 4:
            top = counts_desc(pd.Series(vals).dropna().astype(str)).index[: MAX_CATEGORIES - 1]
            vals = np.where(pd.Series(vals).astype(str).isin(top), vals, "other")
            names = [n for n in names if n in set(top)] + ["other"]
        cols = self.cat_colors(names, column=column)
        enc, names, codes = P.cat_color(self.spec, vals, cols, names=names, nan=self.ch["unassigned"])
        counts = np.bincount(codes, minlength=len(names) + 1)[: len(names)]
        lay = P.points(self.spec, None, xy_ids=self.unit_ids(), color=enc, size=self.unit_size(), shape=self.unit_shape,
                       hover=self.unit_hover(), when=_when(self.base_when, **(when or {})), name=title,
                       select=self.unit_stats(stats), hkey=hkey)  # fmt: skip
        self.add(lay)
        P.legend_cat(self.spec, title, names, enc["colors"], counts, layer=hkey or lay["id"], shape="square",
                     when=lay.get("when"))  # fmt: skip
        return lay

    def units_num(
        self, values: Any, *, title: str, ramp: Sequence[str], scale: str = "lin", fmt: str | None = None,
        when: Mapping[str, Any] | None = None, stats: Sequence[tuple[str, str]] = (), hover_label: str | None = None,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> dict[str, Any]:  # fmt: skip
        v = np.asarray(values, dtype=float)
        enc, dom = P.num_color(self.spec, v, ramp, scale=scale, quantiles=quantiles, nan=self.ch["unassigned"])
        extra = [P.hover_num(self.spec, v, hover_label or title, fmt=fmt)]
        lay = P.points(self.spec, None, xy_ids=self.unit_ids(), color=enc, size=self.unit_size(), shape=self.unit_shape,
                       hover=self.unit_hover(extra), when=_when(self.base_when, **(when or {})), name=title,
                       select=self.unit_stats(stats))  # fmt: skip
        self.add(lay)
        P.legend_colorbar(self.spec, title, ramp, dom, fmt=fmt, scale=scale, when=lay.get("when"))
        return lay

    def units_gene(
        self, genes_key: str, *, control: str = "gene", when: Mapping[str, Any] | None = None,
        stats: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:  # fmt: skip
        ramp = list(C.RAMPS["seq"][self.mode])
        lay = P.points(self.spec, None, xy_ids=self.unit_ids(), color={"kind": "gene", "genes": genes_key, "gene": control,
                       "ramp": ramp}, size=self.unit_size(), shape=self.unit_shape, hover=self.unit_hover(),
                       when=_when(self.base_when, **(when or {})), name="gene", select=self.unit_stats(stats))  # fmt: skip
        self.add(lay)
        sec = {"kind": "genebar", "genes": genes_key, "gene": control, "ramp": ramp, "unit_word": self.unit_word,
               "when": lay.get("when"), "empty": "type a gene name above"}  # fmt: skip
        self.spec.legend.append(sec)
        return lay
