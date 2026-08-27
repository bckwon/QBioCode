#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep V2 — Batch D2 (4 tasks on b05-01)
#
# Replaces D[5-8] (95665[5-8]) which never produced logs on c05-01 (silent
# failure — node accepted jobs but did not execute scripts).
#
# Tasks (all K=16):
#   BBB_Martins/rdkit200, PPBR_AstraZeneca/ecfp4, PPBR_AstraZeneca/maccs,
#   PPBR_AstraZeneca/rdkit200
#==============================================================================
#BSUB -J pca_v2_D2[1-4]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -m "zu-a100-b05-01"
#BSUB -gpu "num=1:mode=shared"
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
