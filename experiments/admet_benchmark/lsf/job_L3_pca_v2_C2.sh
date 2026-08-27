#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep V2 — Batch C2 (7 tasks, 5 on c08-04, 2 on c05-02)
#
# Replaces C (96059) which deadlocked: all 8 tasks landed on GPU3 (njobs=8).
# Fix: 7 individual single-task job arrays, spread across two nodes with
# confirmed free GPU slots. One task per GPU — no contention.
#
# Tasks (all K=16):
#   c08-04 [1-5]: Clearance×3, CYP2D6_Sub/ecfp4, CYP2D6_Sub/maccs
#   c05-02 [6-7]: CYP2D6_Sub/rdkit200, Bioavailability_Ma/ecfp4
# (CYP2C9_Sub/rdkit200 already done from prior run)
#==============================================================================
#BSUB -J pca_v2_C2[1-7]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
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
    "16:::Clearance_Hepatocyte_AZ:::ecfp4"
    "16:::Clearance_Hepatocyte_AZ:::maccs"
    "16:::Clearance_Hepatocyte_AZ:::rdkit200"
    "16:::CYP2D6_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2D6_Substrate_CarbonMangels:::maccs"
    "16:::CYP2D6_Substrate_CarbonMangels:::rdkit200"
    "16:::Bioavailability_Ma:::ecfp4"
)

# Route tasks to specific nodes based on available GPU slots
# [1-5] → c08-04 (5 free slots confirmed), [6-7] → c05-02 (lightly loaded)
NODE_MAP=("" "zu-a100-c08-04" "zu-a100-c08-04" "zu-a100-c08-04" "zu-a100-c08-04" "zu-a100-c08-04" "zu-a100-c05-02" "zu-a100-c05-02")

TASK="${TASKS[${LSB_JOBINDEX}]}"
TARGET_NODE="${NODE_MAP[${LSB_JOBINDEX}]}"

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
echo "Target node hint: ${TARGET_NODE}"
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
