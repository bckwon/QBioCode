#!/usr/bin/env bash
#BSUB -J admet_test_results
#BSUB -q normal
#BSUB -m "zu-a100-c08-03"
#BSUB -n 4
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -M 16000
#BSUB -o logs/admet/test_results_%J.out
#BSUB -e logs/admet/test_results_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

echo "Job ${LSB_JOBID:-local} | Host: $(hostname) | $(date)"

# Build test_results.csv from existing test-split ModelResults.csv files.
# This replaces 04_test_inference.py which requires fitted model checkpoints
# that were not saved during the qprofiler sweep.
python experiments/admet_benchmark/04b_build_test_results.py \
    --results-dir results/admet_config \
    --valid-dir   results/admet_config \
    --output-dir  results/admet_benchmark/test_results

echo "Done: $(date) | Exit: $?"
