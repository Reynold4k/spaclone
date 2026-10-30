"""``spaclone.io.sniff``: what a file, folder or object is, how spaclone would read it, and what you must still say.

:func:`sniff` runs the same detection as the readers on a preview (the first rows of a table, the metadata of an
AnnData) and reports the format and its evidence, the column mapping, the meaning of the count columns, the junction
test, the coordinate unit, the annotation candidates, and the arguments still required (each with the reason). It
never reads a whole large table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from spaclone.io.formats import RECEPTOR_FORMATS

__all__ = ["SniffResult", "sniff"]


@dataclass
class SniffResult:
    """What :func:`sniff` found. ``print()`` it for a readable summary.

    Attributes
    ----------
    source
        What was sniffed.
    kind
        ``"receptor"``, ``"gex"``, ``"receptor+gex"``, ``"store"`` or ``"unknown"``.
    format
        Format name (``format=`` / ``gex_format=`` value), or ``None``.
    tool
        Tool and versions that write it.
    evidence
        Why this format.
    columns
        Canonical field → source column (receptor tables).
    counts
        Meaning of each count column.
    junction
        The conserved-anchor test on the preview rows.
    units
        Coordinate source and unit decision (GEX).
    annotations
        Cell-type label / proportion candidates (GEX).
    required
        ``{argument: reason}`` still needed to read it.
    notes
        Further remarks.
    """

    source: str
    kind: str = "unknown"
    format: str | None = None
    tool: str | None = None
    evidence: list[str] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)
    counts: dict[str, str] = field(default_factory=dict)
    junction: dict[str, Any] = field(default_factory=dict)
    units: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    required: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing else must be given."""
        return self.format is not None and not self.required

    def __str__(self) -> str:
        out = [
            f"{self.source}",
            f"  kind: {self.kind}; format: {self.format or '?'}" + (f" ({self.tool})" if self.tool else ""),
        ]
        for e in self.evidence:
            out.append(f"  evidence: {e}")
        if self.columns:
            out.append("  columns: " + ", ".join(f"{k} <- {v}" for k, v in self.columns.items()))
        for k, v in self.counts.items():
            out.append(f"  counts: {k}: {v}")
        if self.junction:
            out.append(f"  junction: {self.junction.get('decision')}")
        for k, v in self.units.items():
            out.append(f"  {k}: {v}")
        if self.annotations:
            out.append(f"  annotations: {self.annotations}")
        for n in self.notes:
            out.append(f"  note: {n}")
        if self.required:
            for k, v in self.required.items():
                out.append(f"  REQUIRED {k}=  ({v})")
        else:
            out.append("  nothing else required")
        return "\n".join(out)

    __repr__ = __str__


def _sniff_gex(obj: Any, res: SniffResult) -> None:
    import numpy as np

    from spaclone.io.annotations import detect_annotations
    from spaclone.io.gexio import detect_unit, gex_format, read_gex, source_coordinates

    name, ev = gex_format(obj)
    res.kind, res.format, res.evidence = "gex", name, list(ev)
    if name == "seurat_rds":
        res.required["gex"] = "a Seurat .rds must be converted to .h5ad first (see the error of spaclone.read)"
        return
    if name in ("stereo_gef", "stereo_gem"):
        res.required["bin_size"] = "square-bin Stereo-seq data: give the bin side in DNB (e.g. 20 = 10 µm)"
        res.units = {"unit": "Stereo-seq DNB (0.5 µm; resolution attribute of the GEF)"}
        return
    if isinstance(obj, str | Path) and Path(obj).suffix == ".h5ad":
        import anndata as ad

        a = ad.read_h5ad(obj, backed="r")
    else:
        a, _ = read_gex(obj, gex_format=name)
    try:
        xy, src, others = source_coordinates(a)
        res.units["coordinates"] = src + (f" (also: {', '.join(others)})" if others else "")
        try:
            sx, sy, uname, rule = detect_unit(a, np.asarray(xy, dtype=float), coordinates_from=src)
            res.units["unit"] = f"{sx:g} µm per unit ({uname}): {rule}"
        except ValueError as e:
            res.required["unit_um"] = str(e).split(". Pass")[0]
    except (KeyError, ValueError) as e:
        res.required["spatial_key"] = str(e)
    ann = detect_annotations(a)
    res.annotations = ann.record()
    if getattr(a, "isbacked", False):
        a.file.close()


def sniff(source: Any) -> SniffResult:
    """Identify a receptor table or GEX source and list the arguments still needed to read it.

    Parameters
    ----------
    source
        A file, folder, ``DataFrame``, AnnData / MuData or Dandelion object.

    Returns
    -------
    SniffResult
        Printable; ``result.ok`` is True when :func:`spaclone.read` / :func:`spaclone.io.read_receptors` need
        nothing more.

    Examples
    --------
    >>> print(sniff("filtered_contig_annotations.csv"))  # doctest: +SKIP
    """
    from spaclone.io._table import peek_table
    from spaclone.io.receptors import _anndata_source, _finish_table, receptor_format

    res = SniffResult(source=str(source) if isinstance(source, str | Path) else type(source).__name__)
    if isinstance(source, str | Path):
        p = Path(source)
        if p.is_dir() and (p / "meta.json").exists() and (p / "air.parquet").exists():
            res.kind, res.format, res.evidence = "store", "spaclone_store", ["meta.json + air.parquet"]
            return res
        is_gex = p.is_dir() or p.name.lower().endswith((".gef", ".gem", ".gem.gz", ".rds"))
        if p.suffix.lower() in (".h5ad", ".h5mu", ".h5"):
            from spaclone.io.paired import _receptor_in_h5

            kind = _receptor_in_h5(p) if p.suffix.lower() != ".h5" else None
            if kind is None:
                is_gex = True
            else:
                res.kind, res.format = "receptor+gex", kind
                res.evidence.append({"scirpy": "obsm['airr'] (scirpy)", "scirpy_legacy": "IR_* obs columns (old scirpy)",
                                     "cell_calls": "per-cell call columns (Stereo-XCR-seq LUAD)"}[kind])  # fmt: skip
                return res
        if is_gex:
            try:
                _sniff_gex(p, res)
            except (ValueError, FileNotFoundError, KeyError) as e:
                res.required["gex_format"] = str(e)
            return res
    elif not isinstance(source, pd.DataFrame) and (
        hasattr(source, "obsm") or hasattr(getattr(source, "mod", None), "items")
    ):
        try:
            df, name, ev = _anndata_source(source)
            res.kind, res.format, res.evidence = "receptor+gex", name, ev
            return res
        except ValueError:
            _sniff_gex(source, res)
            return res
    # receptor tables
    try:
        fmt, ev = receptor_format(source)
    except ValueError as e:
        res.kind = "unknown"
        res.required["format"] = str(e)
        return res
    res.kind, res.format, res.tool, res.evidence = "receptor", fmt.name, fmt.tool, list(ev)
    if fmt.layout == "special":
        res.notes.append(fmt.notes)
        return res
    if isinstance(source, pd.DataFrame):
        sample = source.head(500).astype(object)
    elif Path(source).suffix.lower() == ".json":
        from spaclone.io.receptors import _read_10x_json

        sample = _read_10x_json(Path(source)).head(500)
    else:
        sample = peek_table(source, n=500).sample
    for attempt in range(3):
        try:
            tab = _finish_table(sample, fmt, ev, columns=None, count_type=res.required and "reads" or None,
                                junction="auto" if "junction" not in res.required else "junction")  # fmt: skip
            break
        except ValueError as e:
            msg = str(e)
            if "count_type" in msg and "count_type" not in res.required:
                res.required["count_type"] = msg
            elif "junction=" in msg and "junction" not in res.required:
                res.required["junction"] = msg
            else:
                res.required["columns"] = msg
                return res
            if attempt == 2:
                return res
    io = tab.attrs.get("spaclone", {}).get("io", {})
    res.columns = dict(io.get("columns", {}))
    res.counts = {k: v for k, v in io.get("counts", {}).items() if k != "count_type"}
    res.junction = dict(io.get("junction", {}))
    if not io.get("cell_ids"):
        res.notes.append("no cell / spot barcode column: a bulk repertoire cannot be placed on GEX units")
    if fmt.name in RECEPTOR_FORMATS and RECEPTOR_FORMATS[fmt.name].notes:
        res.notes.append(RECEPTOR_FORMATS[fmt.name].notes)
    return res


def _together(results: list[SniffResult]) -> list[SniffResult]:
    """Sniff results of files that are read together, as :func:`spaclone.read` reads them.

    An AIR-SPACE clone table supplies the unit of its bead h5ad: ``spaclone.read(bead_h5ad, clone_table)`` uses the
    Curio bead unit, so ``unit_um`` is not required for the h5ad then.
    """
    from spaclone.io.airspace import CURIO_UM

    if any(r.format == "airspace_ovp" for r in results):
        for r in results:
            if r.kind == "gex" and set(r.required) == {"unit_um"} and r.source.lower().endswith(".h5ad"):
                del r.required["unit_um"]
                r.units["unit"] = (
                    f"{CURIO_UM:g} µm per unit (AIR-SPACE / Curio bead unit, applied with the clone table)"
                )
                r.notes.append("read together with the AIR-SPACE clone table: spaclone.read(bead_h5ad, clone_table)")
    return results
