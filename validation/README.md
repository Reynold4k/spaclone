# Validation scripts and results

These scripts recompute, with spaclone functions, the numbers of the analyses that the package was built from, and
compare them with the archived values. The method and the totals are described in the documentation
(`docs/validation.md`).

- `m<k>_<name>.py`: one script per validated part (k = 1 to 9, one group per development module).
- `m<k>_<name>.sbatch`: the SLURM job that ran it.
- `results/m<k>_<name>.csv`: the output, one row per checked quantity, with the columns `quantity`, `archived`,
  `package`, `match` and `note`. `match` is `True` (equal, or within the tolerance stated in the note), `False`
  (differs; the note gives the reason), `new` or `improved` (in `m1_*` and `m2_*`: a number the archive did not
  compute, or the number of a deliberately improved method), or empty (information rows without a pass/fail). Two
  files have other columns: `m8_lineage_per_tree_E.csv` / `m8_lineage_per_tree_L.csv` (per-tree comparison with
  dowser). `m2_banner_units_*.parquet` are per-unit tables for figures, written by `m2_colocalisation.py`.

spaclone was developed from a 2026 analysis that also used unpublished data; those data and their numbers are not
distributed. The validation here uses public data only, plus simulated data. Four tables keep only their rows on
public or simulated data, and the scripts that wrote them are not included because they also read the unpublished
data: `results/m2_own_type_share.csv`, `results/m4_scan.csv` (the calibration rows, also in
`results/m4_scan_part_calibration.csv`), `results/m5_readers.csv` and `results/m6_filtering.csv`.

The scripts are kept as they were run, except that their fixed paths on the project storage of the HPC cluster were
replaced by environment variables: `SPACLONE_ARCHIVE` (folder of the archived analysis of the public data sets),
`SPACLONE_DEMO` (`m5_benchmark.py` only: the archived run of the synthetic demo) and, in the SLURM files,
`SPACLONE_ROOT` (folder with the Python environment `env/` and the scripts in `validation/`; logs go to `logs/` of the
submission folder). The archived analyses are part of a separate analysis archive that is not public, so the scripts
do not run without it.

The StereoIR mouse-spleen files (Stereo-seq 5′ immune-repertoire demonstration data, for which no public description
was found) are not included. The other data sets are public: AIR-SPACE mouse lymph nodes (GEO GSE286452) and
Stereo-XCR-seq human lung adenocarcinoma (GEO GSE328481, processed cell-level files).
