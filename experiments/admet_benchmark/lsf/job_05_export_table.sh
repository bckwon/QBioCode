#!/usr/bin/env bash
#BSUB -J admet_export_table
#BSUB -q normal
#BSUB -m "zu-a100-c08-03"
#BSUB -n 1
#BSUB -R "span[hosts=1] rusage[mem=8000]"
#BSUB -M 8000
#BSUB -o logs/admet/export_table_%J.out
#BSUB -e logs/admet/export_table_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

echo "Job ${LSB_JOBID:-local} | Host: $(hostname) | $(date)"

python experiments/admet_benchmark/05_export_performance_table.py \
    --test-results results/admet_benchmark/test_results/test_results.csv \
    --metadata     data/admet/metadata.json \
    --output-dir   results/admet_benchmark/tables

echo "Done: $(date) | Exit: $?"
