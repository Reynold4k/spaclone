#!/bin/bash
# 2026-10-30: long is closed by dec_maint until 12-04; sapphire with a realistic time (F2)
#SBATCH --job-name=m2_colocalisation
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=logs/m2_colocalisation_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m2_colocalisation.py"
