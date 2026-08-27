#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA kN RESUBMIT — stalled K=16 priority endpoint tasks
#
# Resubmits the 13 tasks that silently hung on c08:
#   K=16: CYP2C9_Substrate × 3 feats (ecfp4, maccs, rdkit200)  [rdkit200 done, skip]
#         Clearance_Hepatocyte_AZ × 3 feats
#         CYP2D6_Substrate × 3 feats                            [rdkit200 done, skip]
#         BBB_Martins × 3 feats
#         PPBR_AstraZeneca × 3 feats
#
# Pinned to c06-01 only.
#==============================================================================
#BSUB -J pca_kN_resub[1-13]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -m "zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/l3_pca_kN_resub_%I_%J.out
#BSUB -e logs/admet/l3_pca_kN_resub_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r

set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

# Explicit task list — 13 stalled K=16 tasks (skip rdkit200 ones already done)
TASKS=(
    ""
    "16:::CYP2C9_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2C9_Substrate_CarbonMangels:::maccs"
    "16:::Clearance_Hepatocyte_AZ:::ecfp4"
    "16:::Clearance_Hepatocyte_AZ:::maccs"
    "16:::Clearance_Hepatocyte_AZ:::rdkit200"
    "16:::CYP2D6_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2D6_Substrate_CarbonMangels:::maccs"
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

INPUT_DIR="${REPO_ROOT}/data/admet/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_pca_${K}.yaml"
DATA_TYPE="${ENDPOINT}_${FEAT}_qsvc_pca${K}"
LOG="${REPO_ROOT}/logs/admet/l3_pca_${ENDPOINT}_${FEAT}_k${K}_${LSB_JOBID:-local}.log"

echo "======================================================"
echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}]  $(hostname)  $(date)"
echo "K=${K}  Endpoint=${ENDPOINT}  Feat=${FEAT}"
echo "======================================================"

PYTHONPATH="${REPO_ROOT}" qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     1 \
    2>&1 | tee "${LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
