#!/bin/bash
#SBATCH --job-name=m9_stereoir_tcr
#SBATCH --partition=long
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/m9_stereoir_tcr_%j.log
# Optional: SPACLONE_DEV_SRC=<src dir> (exported by sbatch --export) runs a development copy of spaclone.
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m9_stereoir_tcr.py" "$SLURM_CPUS_PER_TASK"
