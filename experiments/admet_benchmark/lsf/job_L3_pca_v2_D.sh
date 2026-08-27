#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep V2 — Batch D (8 tasks on c05-01)
#
# Fixes vs 93560:
#   1. Exclusive GPU per task (j_exclusive=yes) — no GPU contention deadlock
#   2. Direct log redirect, no tee pipe — no pipe buffer stall
#   3. One array per node — tasks don't compete for GPU
#
# Tasks: K=16 Bioavail/maccs+rdkit200, BBB×3, PPBR×3
#==============================================================================
#BSUB -J pca_v2_D[1-8]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -m "zu-a100-c05-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=yes"
#BSUB -o /dev/null
#BSUB -e /dev/null
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

TASKS=(
    ""
    "16:::Bioavailability_Ma:::maccs"
    "16:::Bioavailability_Ma:::rdkit200"
    "16:::BBB_Martins:::ecfp4"
    "16:::BBB_Martins:::maccs"
    "16:::BBB_Martins:::rdkit200"
    "16:::PPBR_AstraZeneca:::ecfp4"
    "16:::PPBR_AstraZeneca:::maccs"
    "16:::PPBR_AstraZeneca:::rdkit200"
)

TASK="${TASKS[${LSB_JOBINDEX}]}"
K="${TASK%%:::*}"
REST="${TASK#*:::}"
ENDPOINT="${REST%%:::*}"
FEAT="${REST##*:::}"

INPUT_DIR="${REPO_ROOT}/data/admet_pca_input/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_pca_${K}.yaml"
DATA_TYPE="${ENDPOINT}_${FEAT}_qsvc_pca${K}"
LOG="${REPO_ROOT}/logs/admet/l3_pca_${ENDPOINT}_${FEAT}_k${K}_${LSB_JOBID:-local}.log"

{
echo "======================================================"
echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}]  $(hostname)  $(date)"
echo "K=${K}  Endpoint=${ENDPOINT}  Feat=${FEAT}"
echo "======================================================"

PYTHONPATH="${REPO_ROOT}" qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     1

EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
} > "${LOG}" 2>&1
