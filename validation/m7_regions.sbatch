#!/bin/bash
#SBATCH --job-name=m7_regions
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
# Optional: SPACLONE_DEV=<src dir> runs a development copy (exported by the caller).
unset PYTHONPATH
R="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
cd "$R/validation"
"$R/env/bin/python" -I "$R/validation/m7_regions.py"
