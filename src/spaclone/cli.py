"""Command-line interface.

``spaclone run`` writes a location-trust report for one sample; ``spaclone sniff`` says what a file is and what must
still be given to read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORMATS = ("auto", "airr", "stereo-xcr", "airspace", "cell-calls", "spatialair")


def _read_auto(args: argparse.Namespace):
    """``--format auto``: the receptor format (and the GEX, with ``--gex``) from documented signatures."""
    from pathlib import Path

    from spaclone import io

    kw = {k: v for k, v in (("spot_key", args.spot_key), ("bin_size", args.bin_size), ("count_type", args.count_type))
          if v is not None}  # fmt: skip
    if args.unit_um is not None:
        kw["unit_um"] = args.unit_um
    if args.gex:
        return io.read(args.gex, args.input, **kw), True
    p = Path(args.input)
    if args.clone_table:  # AIR-SPACE: spaclone.read checks the barcode overlap and the unit
        return io.read(
            args.input, args.clone_table, **({"unit_um": args.unit_um} if args.unit_um is not None else {})
        ), False
    if p.is_dir() or p.suffix.lower() in (".h5ad", ".h5mu"):
        return io.read(args.input, **kw), False
    fmt, _ = io.receptor_format(p)
    if fmt.name == "stereo_xcr_meta":
        return io.read_stereo_xcr(p), False
    if fmt.name == "airr":
        return io.read_airr(p), False
    raise SystemExit(f"{p.name} is a {fmt.tool}: give the spatial GEX of the section with --gex")


def _read(args: argparse.Namespace):
    from spaclone import io

    if args.format == "auto":
        return _read_auto(args)[0]
    if args.format == "airr":
        return io.read_airr(args.input)
    if args.format == "stereo-xcr":
        return io.read_stereo_xcr(args.input)
    if args.format == "airspace":
        if not args.clone_table:
            raise SystemExit("--format airspace needs --clone-table (the *_ovp_clone_table.txt.gz)")
        return io.read_airspace(args.input, args.clone_table, **({"scale_um": args.unit_um} if args.unit_um else {}))
    if args.format == "cell-calls":
        import anndata as ad

        return io.from_cell_calls(ad.read_h5ad(args.input))
    return io.read(args.input)


def _cmd_run(args: argparse.Namespace) -> int:
    import anndata as ad
    import matplotlib

    matplotlib.use("Agg")
    import spaclone

    try:  # an argument or input problem: the message says what to change
        data = _read(args)
        gex = ad.read_h5ad(args.gex) if args.gex and args.format != "auto" else None
        report = spaclone.run(
            data,
            gex,
            species=args.species,
            min_reads=args.min_reads,
            on_tissue=True if args.on_tissue else None,
            cell_type_key=args.cell_type_key,
            own_types=args.own_types,
            options=args.options,
            n_perm=args.n_perm,
            rng=args.seed,
        )
    except ValueError as e:
        raise SystemExit(f"spaclone run: {e}") from None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.to_markdown(out / "report.md")
    report.tests().to_csv(out / "tests.csv", index=False)
    (out / "methods.txt").write_text(report.methods() + "\n", encoding="utf-8")
    fig = report.plot()
    fig.savefig(out / "report.png", dpi=200, facecolor=fig.get_facecolor())
    print(f"Verdict: {report.verdict}")
    print(report.verdict_molecule)
    print(report.verdict_clone)
    if not report.verdict.startswith("No evidence"):  # the no-evidence verdict already carries the power statement
        print(report.power)
    print(f"written: {out / 'report.md'}, tests.csv, methods.txt, report.png")
    return 0


def _cmd_sniff(args: argparse.Namespace) -> int:
    """Sniff every file; files given together are judged as ``spaclone.read`` reads them (exit 1 if any needs more)."""
    from spaclone import io
    from spaclone.io.sniff import _together

    rc = 0
    for res in _together([io.sniff(f) for f in args.files]):
        print(res)
        rc = rc or (0 if res.ok else 1)
    return rc


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    from spaclone import __version__

    parser = argparse.ArgumentParser(prog="spaclone", description="End-to-end analysis of spatial BCR/TCR data.")
    parser.add_argument("--version", action="version", version=f"spaclone {__version__}")
    sub = parser.add_subparsers(dest="command")
    r = sub.add_parser("run", help="QC, filtering and every applicable location-trust test, written as a report")
    r.add_argument("--format", default="auto", choices=FORMATS, help="input format (default auto: detected)")
    r.add_argument(
        "--input", required=True, help="receptor file (AIRR TSV, MiXCR meta table, h5ad, or a spaclone.io.write folder)"
    )
    r.add_argument("--clone-table", help="AIR-SPACE clone table (with --format airspace)")
    r.add_argument(
        "--gex",
        help="spatial GEX of the section (.h5ad, Space Ranger / Visium HD / Xenium folder, .gef, .gem); with "
        "--format auto it is read together with --input by spaclone.read",
    )
    r.add_argument("--unit-um", type=float, default=None, help="µm per GEX coordinate unit (default: detected)")
    r.add_argument("--spot-key", default=None, help="receptor column naming the GEX unit (default cell_id)")
    r.add_argument("--bin-size", type=int, default=None, help="Stereo-seq square-bin side in DNB for .gef / .gem")
    r.add_argument("--count-type", default=None, choices=("reads", "umis", "templates", "cells"),
                   help="what an unlabelled receptor count column counts")  # fmt: skip
    r.add_argument("--species", default="mouse", choices=("mouse", "human"))
    r.add_argument("--min-reads", default=None, help="read floor (a number, or 'auto')")
    r.add_argument("--on-tissue", action="store_true", help="keep only records whose in_tissue column is true")
    r.add_argument(
        "--cell-type-key",
        default="auto",
        help="GEX obs column of cell types for the own-type share (default auto: obs['cell_type'], else the only "
        "standard cell-type label column, else the test is skipped with the candidates listed)",
    )
    r.add_argument(
        "--own-types",
        default="auto",
        help='own cell types per receptor: auto (B / T lineage vocabulary) or JSON, e.g. \'{"BCR": ["B_cells"], '
        '"TCR": ["T_cells"]}\' (a string value is a regular expression)',
    )
    r.add_argument(
        "--options",
        default=None,
        help='per-step options as JSON, e.g. \'{"pairing": {"min_units": 10}}\'',
    )
    r.add_argument("--n-perm", type=int, default=200, help="minimum permutations per Monte-Carlo test")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True, help="output folder")
    sn = sub.add_parser("sniff", help="say what a file is, how spaclone reads it and what must still be given")
    sn.add_argument("files", nargs="+", help="receptor tables, GEX files or folders")
    args = parser.parse_args(argv)
    if args.command == "sniff":
        return _cmd_sniff(args)
    if args.command == "run":
        if args.min_reads not in (None, "auto"):
            args.min_reads = float(args.min_reads)
        if args.own_types != "auto":
            try:
                args.own_types = json.loads(args.own_types)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--own-types is not valid JSON: {e}") from e
            ok = isinstance(args.own_types, dict) and set(args.own_types) <= {"BCR", "TCR"}
            if not ok or not all(isinstance(v, str | list) for v in args.own_types.values()):
                raise SystemExit(
                    '--own-types must be auto or a JSON object, e.g. {"BCR": ["B_cells"], "TCR": ["T_cells"]}'
                )
        if args.options:
            try:
                args.options = json.loads(args.options)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--options is not valid JSON: {e}") from e
            if not isinstance(args.options, dict) or not all(isinstance(v, dict) for v in args.options.values()):
                raise SystemExit('--options must be a JSON object of objects, e.g. {"pairing": {"min_units": 10}}')
        return _cmd_run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
