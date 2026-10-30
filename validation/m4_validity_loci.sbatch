#!/bin/bash
#SBATCH --job-name=m4_validity_loci
#SBATCH --partition=long
#SBATCH --time=31-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=logs/m4_validity_loci_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m4_validity_loci.py"
