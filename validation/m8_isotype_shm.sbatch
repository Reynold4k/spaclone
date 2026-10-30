#!/bin/bash
#SBATCH --job-name=m8_isotype_shm
#SBATCH --partition=sapphire
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/m8_isotype_shm_%j.log
# Usage: sbatch m8_isotype_shm.sbatch            (package tree)
#        SPACLONE_SRC=<dev src> sbatch --export=ALL,SPACLONE_SRC m8_isotype_shm.sbatch   (scratch copy)
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
cd "$ROOT/validation"
"$ROOT/env/bin/python" -I "$ROOT/validation/m8_isotype_shm.py"
