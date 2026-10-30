#!/bin/bash
# M4 lineage validation (L7 / L11 statistics from the archived tip tables): sbatch --account=<acct> m4_lineage.sbatch
#SBATCH --job-name=m4_lineage
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m4_lineage.py"
