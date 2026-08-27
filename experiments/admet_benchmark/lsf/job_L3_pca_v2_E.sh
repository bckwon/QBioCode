#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep V2 — Batch E (2 missing K=16 tasks)
#
# Covers the 2 tasks not completed by any prior run:
#   [1] Clearance_Hepatocyte_AZ/maccs  — C2[2] exit 255 (immediate failure)
#   [2] CYP2C9_Substrate_CarbonMangels/rdkit200 — C[1] killed mid-UMAP
#==============================================================================
#BSUB -J pca_v2_E[1-2]
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
    "16:::Clearance_Hepatocyte_AZ:::maccs"
    "16:::CYP2C9_Substrate_CarbonMangels:::rdkit200"
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
