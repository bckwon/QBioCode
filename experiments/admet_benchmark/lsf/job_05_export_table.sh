#!/usr/bin/env bash
#==============================================================================
# LSF Job: Phase 5c — Export Performance Tables
#
# Produces clean, publication-ready CSV and ASCII tables from test-set
# inference results.
#
# Run AFTER job_04_test_inference.sh completes.
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_05_export_table.sh
#
# Estimated wall time: < 5 min (pure pandas post-processing)
#==============================================================================

#BSUB -J admet_export_table
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -W 00:30
#BSUB -o logs/admet/export_table_%J.out
#BSUB -e logs/admet/export_table_%J.err
#BSUB -cwd /dccstor/cardiac/QBioCode

set -euo pipefail

REPO_ROOT="/dccstor/cardiac/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Job ID     : ${LSB_JOBID:-local}"
echo "Host       : $(hostname)"
echo "Start time : $(date)"
echo "======================================================"

cd "${REPO_ROOT}"

${PYTHON} experiments/admet_benchmark/05_export_performance_table.py \
    --test-results  results/admet_benchmark/test_results/test_results.csv \
    --metadata      data/admet/metadata.json \
    --output-dir    results/admet_benchmark/tables \
    --primary-metric auroc \
    2>&1 | tee "${LOG_DIR}/export_table_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
