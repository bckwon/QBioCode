#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA K=4 RESUBMIT — stalled large-Veith CYP tasks
#
# Resubmits the 15 tasks that silently hung on c08 (QSVC kernel deadlock):
#   CYP1A2_Veith  × 3 feats
#   CYP2C19_Veith × 3 feats
#   CYP2C9_Veith  × 3 feats
#   CYP2D6_Veith  × 3 feats
#   CYP3A4_Veith  × 3 feats
#
# Pinned to c06-01 only — consistently stable for large-kernel QSVC.
#==============================================================================
#BSUB -J pca_k4_resub[1-15]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -m "zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/l3_pca_k4_resub_%I_%J.out
#BSUB -e logs/admet/l3_pca_k4_resub_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r

set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

# Explicit task list — 15 stalled tasks from 88161[10-15,19-21,25-27,31-33]
TASKS=(
    ""
    "CYP1A2_Veith:::ecfp4"
    "CYP1A2_Veith:::maccs"
    "CYP1A2_Veith:::rdkit200"
    "CYP2C19_Veith:::ecfp4"
    "CYP2C19_Veith:::maccs"
    "CYP2C19_Veith:::rdkit200"
    "CYP2C9_Veith:::ecfp4"
    "CYP2C9_Veith:::maccs"
    "CYP2C9_Veith:::rdkit200"
    "CYP2D6_Veith:::ecfp4"
    "CYP2D6_Veith:::maccs"
    "CYP2D6_Veith:::rdkit200"
    "CYP3A4_Veith:::ecfp4"
    "CYP3A4_Veith:::maccs"
    "CYP3A4_Veith:::rdkit200"
)

TASK="${TASKS[${LSB_JOBINDEX}]}"
ENDPOINT="${TASK%%:::*}"
FEAT="${TASK##*:::}"
K=4

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
