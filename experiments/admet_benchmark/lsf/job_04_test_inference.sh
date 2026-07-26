#!/usr/bin/env bash
#==============================================================================
# LSF Job: Phase 5b — Test-Set Inference
#
# Loads best checkpoints for all (endpoint × featurizer × model) combinations
# and evaluates on TDC canonical test splits.
#
# Run AFTER job_02_qprofiler_array.sh completes.
# (Can run in parallel with job_03_qsage.sh)
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_04_test_inference.sh
#
# Estimated wall time: ~30 min
#==============================================================================

#BSUB -J admet_test_inference
#BSUB -q normal
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -o logs/admet/test_inference_%J.out
#BSUB -e logs/admet/test_inference_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Job ID     : ${LSB_JOBID:-local}"
echo "Host       : $(hostname)"
echo "Start time : $(date)"
echo "======================================================"

cd "${REPO_ROOT}"

${PYTHON} experiments/admet_benchmark/04_test_inference.py \
    --data-dir        data/admet \
    --checkpoint-dir  results/admet_benchmark/checkpoints \
    --output-dir      results/admet_benchmark/test_results \
    --featurizers     ecfp4 maccs rdkit200 \
    2>&1 | tee "${LOG_DIR}/test_inference_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
