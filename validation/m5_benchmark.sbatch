#!/bin/bash
#SBATCH --job-name=m5_benchmark
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/m5_benchmark_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m5_benchmark.py" 4
