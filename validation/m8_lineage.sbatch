#!/bin/bash
#SBATCH --job-name=m8_lineage
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/m8_lineage_%j.log
# Usage: sbatch m8_lineage.sbatch   (package tree; SPACLONE_SRC=<dev src> with --export=ALL,SPACLONE_SRC for a copy)
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
cd "$ROOT/validation"
"$ROOT/env/bin/python" -I "$ROOT/validation/m8_lineage.py" 4
