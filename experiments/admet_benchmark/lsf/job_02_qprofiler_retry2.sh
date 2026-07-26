#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: QProfiler Retry 2 — 22 remaining tasks
#
# Covers tasks not completed by retry1 (32219).
# Excludes nodes zu-a100-c05-05, zu-a100-c06-01, zu-a100-c08-02 which
# repeatedly fail with "Cannot open your job file" (LSF scratch exhausted).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02_qprofiler_retry2.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry2[1-22]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-b05-01 zu-a100-b05-02 zu-a100-c05-01 zu-a100-c05-02 zu-a100-c05-03 zu-a100-c05-04 zu-a100-c08-01 zu-a100-c08-03 zu-a100-c08-04"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -W 04:00
#BSUB -o logs/admet/qprofiler_%I_%J.out
#BSUB -e logs/admet/qprofiler_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

mkdir -p "${LOG_DIR}"

# Explicit list of the 22 remaining (endpoint:::featurizer) pairs — 1-based index
TASKS=(
    ""                                          # index 0 unused (LSF is 1-based)
    "VDss_Lombardo:::ecfp4"                     #  1
    "VDss_Lombardo:::maccs"                     #  2
    "VDss_Lombardo:::rdkit200"                  #  3
    "CYP2C19_Veith:::ecfp4"                     #  4
    "CYP2C19_Veith:::maccs"                     #  5
    "CYP2C19_Veith:::rdkit200"                  #  6
    "CYP2D6_Veith:::ecfp4"                      #  7
    "CYP3A4_Veith:::maccs"                      #  8
    "CYP2C9_Veith:::maccs"                      #  9
    "CYP2C9_Veith:::rdkit200"                   # 10
    "Clearance_Hepatocyte_AZ:::ecfp4"           # 11
    "Clearance_Hepatocyte_AZ:::maccs"           # 12
    "Clearance_Hepatocyte_AZ:::rdkit200"        # 13
    "hERG:::ecfp4"                              # 14
    "hERG:::maccs"                              # 15
    "hERG:::rdkit200"                           # 16
    "AMES:::ecfp4"                              # 17
    "AMES:::maccs"                              # 18
    "AMES:::rdkit200"                           # 19
    "DILI:::ecfp4"                              # 20
    "DILI:::maccs"                              # 21
    "DILI:::rdkit200"                           # 22
)

TASK="${TASKS[${LSB_JOBINDEX}]}"
ENDPOINT="${TASK%%:::*}"
FEAT="${TASK##*:::}"

INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"

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

N_CSV=$(find "${INPUT_DIR}" -maxdepth 1 -name "train.csv" | wc -l)
if [ "${N_CSV}" -eq 0 ]; then
    echo "[ERROR] No train.csv found in ${INPUT_DIR}"
    exit 1
fi

cd "${REPO_ROOT}"

DATA_TYPE="${ENDPOINT}_${FEAT}"

# Ensure venv binaries are on PATH
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     10 \
    2>&1 | tee "${LOG_DIR}/qprofiler_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
