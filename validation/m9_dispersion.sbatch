#!/bin/bash
#SBATCH --job-name=m9_dispersion
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/m9_dispersion_%j.log
# Optional: SPACLONE_DEV_SRC=<src dir> (exported by sbatch --export) runs a development copy of spaclone.
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m9_dispersion.py"
