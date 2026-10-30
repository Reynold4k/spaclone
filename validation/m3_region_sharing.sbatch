#!/bin/bash
#SBATCH --job-name=m3_region_sharing
#SBATCH --partition=long
# project rule is --time=31-00:00:00; long accepts only > 30 days, and a 31-day job cannot start before the
# dec_maint reservation of the long nodes (2026-11-30 to 2026-12-04) has passed, so 30 days + 5 min until then
#SBATCH --time=30-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/m3_region_sharing_%j.log
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
"$ROOT/env/bin/python" -I "$ROOT/validation/m3_region_sharing.py"
