#!/bin/bash
#SBATCH --job-name=m9_airspace_repertoire
#SBATCH --partition=sapphire
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=logs/m9_airspace_repertoire_%j.log
# Optional: SPACLONE_DEV_SRC=<src dir> (exported by sbatch --export) runs a development copy of spaclone.
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m9_airspace_repertoire.py"
