#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L3 PCA sweep FINAL RESUBMIT — 36 missing tasks
#
# All previously stalled tasks resubmitted with file_dataset=train_qml.csv
# (patched in admet_qsvc_pca_{K}.yaml — avoids full train.csv QSVC kernel).
# Pinned to c06-01 only for stability.
#
# Tasks:
#   K=4:  CYP1A2_Veith×3, CYP2C19_Veith×3, CYP2C9_Veith×3,
#          CYP2D6_Veith×3, CYP3A4_Veith×3, DILI/ecfp4, Bioavailability_Ma/rdkit200
#   K=12: PPBR_AstraZeneca/rdkit200
#   K=16: CYP2C9_Sub×3, Clearance×3, CYP2D6_Sub×3, Bioavail×3, BBB×3, PPBR×3
#==============================================================================
#BSUB -J pca_final[1-36]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -m "zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/l3_pca_final_%I_%J.out
#BSUB -e logs/admet/l3_pca_final_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r

set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

TASKS=(
    ""
    "4:::Bioavailability_Ma:::rdkit200"
    "4:::CYP1A2_Veith:::ecfp4"
    "4:::CYP1A2_Veith:::maccs"
    "4:::CYP1A2_Veith:::rdkit200"
    "4:::CYP2C19_Veith:::ecfp4"
    "4:::CYP2C19_Veith:::maccs"
    "4:::CYP2C19_Veith:::rdkit200"
    "4:::CYP2C9_Veith:::ecfp4"
    "4:::CYP2C9_Veith:::maccs"
    "4:::CYP2C9_Veith:::rdkit200"
    "4:::CYP2D6_Veith:::ecfp4"
    "4:::CYP2D6_Veith:::maccs"
    "4:::CYP2D6_Veith:::rdkit200"
    "4:::CYP3A4_Veith:::ecfp4"
    "4:::CYP3A4_Veith:::maccs"
    "4:::CYP3A4_Veith:::rdkit200"
    "4:::DILI:::ecfp4"
    "12:::PPBR_AstraZeneca:::rdkit200"
    "16:::CYP2C9_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2C9_Substrate_CarbonMangels:::maccs"
    "16:::CYP2C9_Substrate_CarbonMangels:::rdkit200"
    "16:::Clearance_Hepatocyte_AZ:::ecfp4"
    "16:::Clearance_Hepatocyte_AZ:::maccs"
    "16:::Clearance_Hepatocyte_AZ:::rdkit200"
    "16:::CYP2D6_Substrate_CarbonMangels:::ecfp4"
    "16:::CYP2D6_Substrate_CarbonMangels:::maccs"
    "16:::CYP2D6_Substrate_CarbonMangels:::rdkit200"
    "16:::Bioavailability_Ma:::ecfp4"
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

INPUT_DIR="${REPO_ROOT}/data/admet/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_pca_${K}.yaml"
DATA_TYPE="${ENDPOINT}_${FEAT}_qsvc_pca${K}"
LOG="${REPO_ROOT}/logs/admet/l3_pca_${ENDPOINT}_${FEAT}_k${K}_${LSB_JOBID:-local}.log"

echo "======================================================"
echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}]  $(hostname)  $(date)"
echo "K=${K}  Endpoint=${ENDPOINT}  Feat=${FEAT}  (train_qml.csv only)"
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
