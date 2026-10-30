#!/bin/bash
#SBATCH --job-name=m8_spectrum_selection
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=logs/m8_spectrum_selection_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
cd "$ROOT/validation"
"$ROOT/env/bin/python" -I "$ROOT/validation/m8_spectrum_selection.py"
