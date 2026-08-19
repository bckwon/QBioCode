#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4d-2 — QProfiler Sweep RETRY4 RESUBMIT (5 tasks)
#
# Resubmits the 5 retry4 tasks that were frozen on zu-a100-c08-01 (100% CPU
# saturated, zero throughput) for 28–66 hours:
#
#   [4]  CYP3A4_Veith / ecfp4           (41742 task 4,  orig 34)
#   [6]  CYP3A4_Veith / rdkit200        (41742 task 6,  orig 36)
#   [10] CYP2C9_Veith / ecfp4           (41742 task 10, orig 40)
#   [11] CYP2C9_Veith / maccs           (41742 task 11, orig 41)
#   [13] Clearance_Hepatocyte_AZ/rdkit200 (41742 task 13, orig 57)
#
# Node selection: pinned to b05-02 and c06-01 only (lowest load as of Aug 14).
# Avoids: c08-01 (100%), c08-02 (97%), c05-03 (99%), b05-01 (97%),
#         c08-04 (96%), c05-02 (55%), c05-01 (silently hangs), c05-04 (closed).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02d2_retry4_resubmit5.sh
#==============================================================================

#BSUB -J admet_retry4r5[1-5]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -M 16000
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/retry4r5_%I_%J.out
#BSUB -e logs/admet/retry4r5_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

mkdir -p "${LOG_DIR}"

# Explicit task list — the 5 frozen tasks from 41742
TASKS=(
    ""                                          #  0 — unused (LSF is 1-based)
    "CYP3A4_Veith:::ecfp4"                      #  1  (41742[4],  orig 34)
    "CYP3A4_Veith:::rdkit200"                   #  2  (41742[6],  orig 36)
    "CYP2C9_Veith:::ecfp4"                      #  3  (41742[10], orig 40)
    "CYP2C9_Veith:::maccs"                      #  4  (41742[11], orig 41)
    "Clearance_Hepatocyte_AZ:::rdkit200"        #  5  (41742[13], orig 57)
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
    2>&1 | tee "${LOG_DIR}/retry4r5_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
