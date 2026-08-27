#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep V2 — Batch C (8 tasks on c05-03)
#
# Fixes vs 93560:
#   1. mode=shared (j_exclusive=yes removed — incompatible with shared slots)
#      Node isolation via -m pinning provides GPU separation between arrays.
#   2. Direct log redirect, no tee pipe — no pipe buffer stall
#   3. One array per node — tasks don't compete for GPU across arrays
#
# Tasks: K=16 CYP2C9_Sub/rdkit200, Clearance×3, CYP2D6_Sub×3, Bioavail/ecfp4
#==============================================================================
#BSUB -J pca_v2_C[1-8]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -m "zu-a100-c08-04"
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
    "16:::CYP2C9_Substrate_CarbonMangels:::rdkit200"
    "16:::Clearance_Hepatocyte_AZ:::ecfp4"
    "16:::Clearance_Hepatocyte_AZ:::maccs"
    "16:::Clearance_Hepatocyte_AZ:::rdkit200"
    "16:::CYP2D6_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2D6_Substrate_CarbonMangels:::maccs"
    "16:::CYP2D6_Substrate_CarbonMangels:::rdkit200"
    "16:::Bioavailability_Ma:::ecfp4"
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
