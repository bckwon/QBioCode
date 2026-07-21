#!/usr/bin/env bash
#==============================================================================
# LSF Job: Phase 1 — ADMET Dataset Preparation
#
# Downloads all 22 TDC ADMET endpoints, applies binarization, and
# featurizes with ECFP4, MACCS, and RDKit200.
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_01_prepare.sh
#
# Estimated wall time: ~30 min (mostly TDC network I/O + RDKit compute)
#==============================================================================

#BSUB -J admet_prepare
#BSUB -q normal
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -W 02:00
#BSUB -o logs/admet/prepare_%J.out
#BSUB -e logs/admet/prepare_%J.err
#BSUB -cwd /dccstor/cardiac/QBioCode

set -euo pipefail

REPO_ROOT="/dccstor/cardiac/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "Repo root    : ${REPO_ROOT}"
echo "Python       : ${PYTHON}"
echo "======================================================"

cd "${REPO_ROOT}"

${PYTHON} experiments/admet_benchmark/01_prepare_admet_datasets.py \
    --data-dir      data/admet \
    --featurizers   ecfp4 maccs rdkit200 \
    --qml-cap       300 \
    --seed          42 \
    2>&1 | tee "${LOG_DIR}/prepare_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
