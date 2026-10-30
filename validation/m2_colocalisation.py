"""M2 validation: same-clone co-localisation (spaclone.qc.colocalisation) against the archive's X5 test T3.

Archive: $SPACLONE_ARCHIVE/workflow/X5_depth_tests.py, function t3 -> results/DEPTH/t3_clone_position.csv.
Units as X5: LUAD (GSE328481, 11 chips) = CellBin cells with an IGH clone call (IGHcdr3aa), clone = IGHcdr3aa;
AIR-SPACE (GSE286452, 6 lymph nodes) = (tissue bead, MiXCR cloneId) units of IGH molecules (deduplicated by
barcode + UMI), coordinates = bead x, y * 0.65. Depth strata = Ig constant-gene RNA of the unit in the bins 0, 1-2,
3-9, 10-29, 30-99, 100-299, >= 300. Bands 0-100, 0-20, 20-100 um, clones of 2-50 units, same-bead pairs excluded,
200 shuffles per null.

Run 1 (legacy-equivalent: units='records', sink_min_clones=None): strata as integer codes (np.digitize, as X5) and
one generator np.random.default_rng(0) consumed in X5's order (LUAD chips sorted by file name, then AIR-SPACE
Mock..D21; depth-matched before whole-sample), so the shuffles are identical draw for draw: observed shares, null
summaries and p-values must match exactly.
Run 2 (package strata labels, legacy units): strata as spaclone.nulls.bin_labels strings (the same partition, visited
in another order), own generator: observed shares exact, nulls within Monte-Carlo error.
Run 3 (improved, 2026-10-30 review fix M4; rows with match = 'improved'): 20-100 um, depth-matched, (a) barcode sinks
(>= 20 clones per bead) removed first with (spot, clone) units, which is what spaclone.run uses for beads and cells,
and (b) the function default for sub-cellular spots, loci of 15 um (merging same-clone cells/beads within 15 um, so
the clone-size window counts cells). 'archived' = archived ratio, 'package' = the ratio under the improved design.
Also writes the README-banner unit tables m2_banner_units_AIRSPACE_D14.parquet and m2_banner_units_LUAD_D06047A2.parquet
(one row per (spot, clone) unit used by the test).
Output: validation/results/m2_colocalisation.csv (quantity, archived, package, match, note)
Usage : python -I m2_colocalisation.py
"""

import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spaclone
from spaclone import nulls, qc

E = Path(os.environ["SPACLONE_ARCHIVE"])  # archived analysis of the public data sets (not distributed)
H5L = E / "StereoXCR_LUAD_GSE328481" / "raw"
RAW = E / "AIRSPACE_GSE286452" / "raw"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [(0.0, 100.0), (0.0, 20.0), (20.0, 100.0)]
EDGES = [1, 3, 10, 30, 100, 300]  # X5: np.digitize(ig, BINS[1:-1])
N_PERM = 200
LEGACY = {"units": "records", "sink_min_clones": None}
AIR = {
    "Mock": ("GSM8727829", "GSM8727835"),
    "D3": ("GSM8727830", "GSM8727836"),
    "D7": ("GSM8727831", "GSM8727837"),
    "D10": ("GSM8727832", "GSM8727838"),
    "D14": ("GSM8727833", "GSM8727839"),
    "D21": ("GSM8727834", "GSM8727840"),
}
IG_MOUSE = ["Igkc", "Iglc1", "Iglc2", "Iglc3", "Ighm", "Ighd", "Ighg1", "Ighg2b", "Ighg2c", "Ighg3", "Igha", "Ighe"]
BANNER = {
    ("LUAD", "D06047A2"): "m2_banner_units_LUAD_D06047A2.parquet",
    ("AIR-SPACE", "D14"): "m2_banner_units_AIRSPACE_D14.parquet",
}
T0 = time.time()
rows = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add(quantity, archived, package, note="", atol=0.0, rtol=0.0):
    a, p = float(archived), float(package)
    match = (np.isnan(a) and np.isnan(p)) or abs(a - p) <= atol + rtol * abs(a)
    rows.append({"quantity": quantity, "archived": archived, "package": package, "match": bool(match), "note": note})


def gsum(A, X, genes):
    idx = [A.var_names.get_loc(g) for g in genes if g in A.var_names]
    return np.asarray(X[:, idx].sum(1)).ravel() if idx else np.zeros(A.n_obs)


def units_table(spot, x_um, y_um, clone, ig, sample):
    return pd.DataFrame(
        {
            "spot_id": np.asarray(spot),
            "x_um": np.asarray(x_um, float),
            "y_um": np.asarray(y_um, float),
            "locus": "IGH",
            "clone_id": np.asarray(clone, dtype=object),
            "library_id": sample,
            "ig_code": np.digitize(np.asarray(ig, float), EDGES),
            "ig_bin": nulls.bin_labels(np.asarray(ig, float), nulls.IG_RNA_BINS),
        }
    )


def inputs():
    """Yield (dataset, sample, air table) in X5's order."""
    for f in sorted(H5L.glob("GSM*.h5ad")):
        chip = f.stem.split("_")[1]
        A = ad.read_h5ad(f)
        X = A.X.tocsc()
        igh = [
            g
            for g in A.var_names
            if g in ("IGHM", "IGHD", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHE")
        ]
        iglc = [g for g in A.var_names if g.startswith("IGLC") and not g.startswith("IGLCOR")]
        ig = gsum(A, X, ["IGKC", *iglc, *igh])
        o = A.obs
        call = o.IGHcdr3aa.notna().to_numpy()
        air = units_table(
            A.obs_names.to_numpy()[call],
            o.x.to_numpy(float)[call] * 0.5,
            o.y.to_numpy(float)[call] * 0.5,
            o.IGHcdr3aa.astype(object).to_numpy()[call],
            ig[call],
            chip,
        )
        del X, A
        yield "LUAD", chip, air
    for s, (g_gex, g_air) in AIR.items():
        A = ad.read_h5ad(next(RAW.glob(f"{g_gex}_adata_{s}_*.h5ad")))
        X = A.X.tocsc() if hasattr(A.X, "tocsc") else A.X
        B = pd.DataFrame({"barcode": A.obs.barcode.to_numpy(), "ig": gsum(A, X, IG_MOUSE)})
        del X, A
        R = pd.read_parquet(E / "results" / "AIRSPACE" / f"region_maps_{s}.parquet")
        U = R[["barcode", "x", "y", "region"]].merge(B, on="barcode")
        U["x_um"], U["y_um"] = U.x * 0.65, U.y * 0.65
        M = pd.read_csv(
            next(RAW.glob(f"{g_air}_{s}_ovp_clone_table.txt.gz")),
            sep="\t",
            low_memory=False,
            usecols=["barcode", "umi", "receptor_type", "cloneId"],
        )
        M = M[M.receptor_type == "IGH"].drop_duplicates(["barcode", "umi"])
        mol = M.merge(U[["barcode", "x_um", "y_um", "ig"]], on="barcode")  # inner join keeps the molecules' order
        yield "AIR-SPACE", s, units_table(mol.barcode, mol.x_um, mol.y_um, mol.cloneId.astype(str), mol.ig, s)


def main():
    log(f"spaclone {spaclone.__version__}")
    arch = pd.read_csv(E / "results" / "DEPTH" / "t3_clone_position.csv")
    arch["design"] = arch["null"].map({"depth_matched": "stratified", "whole_sample": "whole_sample"})
    gen_legacy = np.random.default_rng(0)  # X5's generator
    gen_default = np.random.default_rng(20261030)
    gen_improved = np.random.default_rng(20261031)  # run 3 has its own draws, so run 2 is unchanged
    exact = {"observed": "statistic", "null_mean": "null_mean", "null_lo": "null_lo", "null_hi": "null_hi"}
    for dataset, sample, air in inputs():
        r1 = qc.colocalisation(air, strata=["ig_code", "ig_bin"], bands_um=BANDS, n_perm=N_PERM, rng=gen_legacy,
                               **LEGACY)  # fmt: skip
        r2 = qc.colocalisation(air, strata="ig_bin", bands_um=BANDS, n_perm=N_PERM, rng=gen_default, **LEGACY)
        if (dataset, sample) in BANNER:
            r1.units.to_parquet(OUT / BANNER[(dataset, sample)], index=False)
        a = arch[(arch.dataset == dataset) & (arch["sample"] == sample)]
        t1, t2 = r1.table, r2.table
        first = t1.iloc[0]
        add(f"{dataset} {sample} units", a.units.iat[0], first.n_units, "(spot, clone) units", atol=0)
        add(f"{dataset} {sample} clones of 2-50 units", a.clones_2_50.iat[0], first.n_clones_in_window, atol=0)
        for _, ar in a.iterrows():
            key = (t1.design == ar.design) & (t1.band == ar.band_um)
            p1, p2 = t1[key].iloc[0], t2[(t2.design == ar.design) & (t2.band == ar.band_um)].iloc[0]
            tag = f"{dataset} {sample} {ar['null']} {ar.band_um} um"
            for col, pc in exact.items():
                add(f"{tag} {col}", ar[col], p1[pc], "legacy-equivalent run (same draws)", atol=1e-15, rtol=1e-12)
            add(f"{tag} ratio", ar.ratio, p1.ratio, "legacy-equivalent run (same draws)", atol=1e-12, rtol=1e-9)
            add(f"{tag} p_greater", ar.p_greater, p1.p_value, "legacy-equivalent run (same draws)", atol=1e-12)
            add(f"{tag} observed (default strata labels)", ar.observed, p2.statistic, "package default run", atol=1e-15,
                rtol=1e-12)  # fmt: skip
            sd_arch = (ar.null_hi - ar.null_lo) / 3.92  # the archive kept the 95% range, not the sd
            mc = 4.5 * np.hypot(p2.null_sd, sd_arch) / np.sqrt(N_PERM) + 1e-12
            add(f"{tag} null_mean (default strata labels)", ar.null_mean, p2.null_mean,
                f"package default run, other draws; tolerance 4.5 MC s.e. = {mc:.2g}", atol=mc)  # fmt: skip
            add(f"{tag} expected (analytic) vs archived null mean", ar.null_mean, p1.expected,
                "package analytic expectation (ratio of expectations); tolerance 4.5 MC s.e. of the archive",
                atol=4.5 * sd_arch / np.sqrt(N_PERM) + 1e-12)  # fmt: skip
        a20 = a[(a.design == "stratified") & (a.band_um == "20-100")].iloc[0]
        for label, kw in (("units=records, sinks removed", {"units": "records", "sink_min_clones": 20}),
                          ("units=loci (15 um), sinks removed", {})):  # fmt: skip
            r3 = qc.colocalisation(air, within="ig_bin", bands_um=((20.0, 100.0),), n_perm=N_PERM, rng=gen_improved,
                                   **kw)  # fmt: skip
            t3 = r3.table[r3.table.design == "stratified"].iloc[0]
            rows.append({"quantity": f"{dataset} {sample} depth_matched 20-100 um ratio ({label})", "archived": a20.ratio,
                         "package": t3.ratio, "match": "improved",
                         "note": f"improved design (review M4); {t3.n_units} units, {t3.n_clones_in_window} clones in the "
                                 f"window, p {t3.p_value:.3g}; {r3.summary['n_sink_spots']} sink spots removed"})  # fmt: skip
        log(f"{dataset} {sample}: {first.n_units} units; " + "; ".join(
            f"{r.design[:5]} {r.band}: obs {r.statistic:.4g} null {r.null_mean:.4g} p {r.p_value:.3g}"
            for r in t1.itertuples() if r.band != "0-100"))  # fmt: skip
    R = pd.DataFrame(rows, columns=["quantity", "archived", "package", "match", "note"])
    R.to_csv(OUT / "m2_colocalisation.csv", index=False)
    scored = R[R.match.isin([True, False])].assign(match=lambda d: d.match.astype(bool))
    legacy = scored[scored.note.str.startswith("legacy")]
    log(
        f"{int(scored.match.sum())} / {len(scored)} quantities match; legacy-equivalent run {int(legacy.match.sum())} / "
        f"{len(legacy)}; {int((R.match == 'improved').sum())} improved-design rows"
    )
    print(scored[~scored.match].to_string(index=False), flush=True)
    return 0 if scored.match.all() else 1


if __name__ == "__main__":
    sys.exit(main())
