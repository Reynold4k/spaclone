#!/bin/bash
#SBATCH --job-name=m8_airspace_shm
#SBATCH --partition=sapphire
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/m8_airspace_shm_%j.log
# Optional: SPACLONE_SRC=<src dir> (sbatch --export=ALL,SPACLONE_SRC=...) runs a development copy of spaclone.
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m8_airspace_shm.py"
