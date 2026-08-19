#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Classical-at-300 Ablation RESUBMIT (12 tasks)
#
# Resubmits the 12 tasks whose ModelResults.csv was left empty by the
# checkpoint-skip bug (stale batch dirs from the Aug-13 dry-run were present
# when job 76124 started, causing qprofiler-batch to skip all computation).
#
# Stale dirs have been deleted before this script was created.
# Tasks:
#   [1]  AMES / ecfp4
#   [2]  AMES / maccs
#   [3]  AMES / rdkit200
#   [4]  CYP1A2_Veith / ecfp4
#   [5]  CYP1A2_Veith / maccs
#   [6]  CYP1A2_Veith / rdkit200
#   [7]  Half_Life_Obach / ecfp4
#   [8]  Half_Life_Obach / maccs
#   [9]  Half_Life_Obach / rdkit200
#   [10] Lipophilicity_AstraZeneca / ecfp4
#   [11] Lipophilicity_AstraZeneca / maccs
#   [12] Lipophilicity_AstraZeneca / rdkit200
#
# Config: admet_classical300_config.yaml
#   - model: ['lr','rf','mlp','xgb','svc']
#   - embeddings: ['none','pca','umap']
#   - max_train_samples: 300 (uses train_qml.csv)
#
# Node selection: pinned to b05-02 and c06-01 (lowest load as of Aug 14).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02h2_classical300_resubmit12.sh
#==============================================================================

#BSUB -J admet_cl300_r12[1-12]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01"
#BSUB -o logs/admet/cl300_r12_%I_%J.out
#BSUB -e logs/admet/cl300_r12_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_classical300_config.yaml"

mkdir -p "${LOG_DIR}"

TASKS=(
    ""                                          #  0 — unused
    "AMES:::ecfp4"                              #  1
    "AMES:::maccs"                              #  2
    "AMES:::rdkit200"                           #  3
    "CYP1A2_Veith:::ecfp4"                      #  4
    "CYP1A2_Veith:::maccs"                      #  5
    "CYP1A2_Veith:::rdkit200"                   #  6
    "Half_Life_Obach:::ecfp4"                   #  7
    "Half_Life_Obach:::maccs"                   #  8
    "Half_Life_Obach:::rdkit200"                #  9
    "Lipophilicity_AstraZeneca:::ecfp4"         # 10
    "Lipophilicity_AstraZeneca:::maccs"         # 11
    "Lipophilicity_AstraZeneca:::rdkit200"      # 12
)

TASK="${TASKS[${LSB_JOBINDEX}]}"
ENDPOINT="${TASK%%:::*}"
FEAT="${TASK##*:::}"

INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}[${LSB_JOBINDEX:-1}]"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "Task index   : ${LSB_JOBINDEX} / 12"
echo "Endpoint     : ${ENDPOINT}"
echo "Featurizer   : ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Config       : ${CONFIG}"
echo "======================================================"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found in ${INPUT_DIR}"
    exit 1
fi

TRAIN_QML_LINES=$(wc -l < "${INPUT_DIR}/train_qml.csv")
echo "train_qml.csv lines: ${TRAIN_QML_LINES} (≤301 expected)"

cd "${REPO_ROOT}"

PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}_classical300" \
    --n-jobs     5 \
    2>&1 | tee "${LOG_DIR}/cl300_r12_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
