#!/bin/bash
#SBATCH --job-name=m1_airspace_umi
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:05:00
#SBATCH --output=logs/m1_airspace_umi_match_%j.log
# M1 validation of qc.umi_match on the archived AIR-SPACE X4 tables (D10, D14).
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
$ROOT/env/bin/python -I $ROOT/validation/m1_airspace_umi_match.py
