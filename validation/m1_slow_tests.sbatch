#!/bin/bash
#SBATCH --job-name=m1_slow_tests
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:05:00
#SBATCH --output=logs/m1_slow_tests_%j.log
# M1 test files including the slow calibration tests (pytest -m slow and the default ones).
unset PYTHONPATH
ROOT="${SPACLONE_ROOT:?set SPACLONE_ROOT to the folder with env/ and validation/}"
cd "$ROOT/spaclone"
"$ROOT/env/bin/python" -I -m pytest -p no:cacheprovider tests/test_sinks.py tests/test_same_spot.py tests/test_offtissue.py tests/test_umi_match.py tests/test_io_stereo.py -m "slow or not slow" -q --durations=10
