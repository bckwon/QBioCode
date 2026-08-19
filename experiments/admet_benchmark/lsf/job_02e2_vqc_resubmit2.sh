#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: VQC Resubmit (2 tasks)
#
# Resubmits the 2 CYP2C19_Veith tasks that were frozen during VQC/QNN fitting:
#
#   [1]  CYP2C19_Veith / ecfp4   (42835[7],  stalled 74.7h on c05-05)
#   [2]  CYP2C19_Veith / maccs   (44027,     stalled 63.8h on b05-01)
#
# These use the MAIN benchmark config (admet_config.yaml) which includes
# all QML models (qsvc, vqc, qnn, qensemble) + classical (lr, rf, mlp, xgb, svc).
#
# Node selection: pinned to b05-02 and c06-01 only (lowest load as of Aug 14).
# Explicitly excludes c05-05 (unreachable/hangs VQC) and c08-01 (100% load).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02e2_vqc_resubmit2.sh
#==============================================================================

#BSUB -J admet_vqc_resub[1-2]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -M 16000
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/vqc_resub_%I_%J.out
#BSUB -e logs/admet/vqc_resub_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

mkdir -p "${LOG_DIR}"

# Task list: both CYP2C19_Veith tasks
TASKS=(
    ""                             #  0 — unused (LSF is 1-based)
    "CYP2C19_Veith:::ecfp4"        #  1  (42835[7],  orig 37)
    "CYP2C19_Veith:::maccs"        #  2  (44027,     orig 38)
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
echo "Endpoint     : ${ENDPOINT}"
echo "Featurizer   : ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Config       : ${CONFIG}"
echo "======================================================"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train.csv" ]; then
    echo "[ERROR] train.csv not found in ${INPUT_DIR}"
    exit 1
fi

cd "${REPO_ROOT}"

PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     10 \
    2>&1 | tee "${LOG_DIR}/vqc_resub_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
