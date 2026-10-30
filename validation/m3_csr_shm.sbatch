#!/bin/bash
#SBATCH --job-name=m3_csr_shm
#SBATCH --partition=sapphire
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/m3_csr_shm_%j.log
# long is closed by the December maintenance (until 2026-12-04): sapphire / cascade with a realistic time.
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m3_csr_shm.py"
