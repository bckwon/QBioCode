#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4c — QProfiler Sweep RETRY 3 (18 dead tasks)
#
# Re-submits the 18 tasks that died in arrays 40223 and 40830 due to:
#   - [Errno 28] No space left on device  →  zu-a100-c05-02 scratch full
#   - TERM_OWNER kill                     →  zu-a100-c05-05
#
# Node availability at time of submission (Mon Jul 27):
#   SAFE  (ok, GPU, sufficient free slots):
#     zu-a100-b05-01  (free: 86)
#     zu-a100-c05-01  (free: 22)
#     zu-a100-c05-04  (free: 14)
#     zu-a100-c06-01  (free: 32)
#     zu-a100-c08-03  (free: 25)
#   EXCLUDED (caused failures):
#     zu-a100-c05-02  — /tmp scratch full  → OSError [Errno 28]
#     zu-a100-c05-05  — TERM_OWNER kills
#   EXCLUDED (unavail):
#     zu-a100-c05-03, zu-a100-c08-02, zu-a100-c08-04
#   EXCLUDED (too few slots):
#     zu-a100-b05-02 (free: 4), zu-a100-c08-01 (free: 2)
#
# Dead tasks (18 total, 1-based indices matching the original task mapping):
#   [31] CYP2D6_Veith / ecfp4
#   [32] CYP2D6_Veith / maccs
#   [33] CYP2D6_Veith / rdkit200
#   [34] CYP3A4_Veith / ecfp4
#   [35] CYP3A4_Veith / maccs
#   [36] CYP3A4_Veith / rdkit200
#   [37] CYP1A2_Veith / ecfp4
#   [38] CYP1A2_Veith / maccs
#   [39] CYP1A2_Veith / rdkit200
#   [40] CYP2C9_Veith / ecfp4
#   [41] CYP2C9_Veith / maccs
#   [42] CYP2C9_Veith / rdkit200
#   [57] Clearance_Hepatocyte_AZ / rdkit200
#   [62] AMES / maccs
#   [63] AMES / rdkit200
#   [64] DILI / ecfp4
#   [65] DILI / maccs
#   [66] DILI / rdkit200
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02c_qprofiler_retry3.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry3[1-18]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-b05-01 zu-a100-c05-01 zu-a100-c05-04 zu-a100-c06-01 zu-a100-c08-03"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qprofiler_%I_%J.out
#BSUB -e logs/admet/qprofiler_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
# Auto-requeue on node failure (up to 3 attempts per element).
#BSUB -r
#BSUB -nr 3

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

mkdir -p "${LOG_DIR}"

# Explicit flat list of the 18 dead tasks — 1-based (index 0 unused).
# Each entry is "ENDPOINT:::FEATURIZER".
TASKS=(
    ""                                          # 0 — unused (LSF is 1-based)
    "CYP2D6_Veith:::ecfp4"                      #  1  (original idx 31)
    "CYP2D6_Veith:::maccs"                      #  2  (original idx 32)
    "CYP2D6_Veith:::rdkit200"                   #  3  (original idx 33)
    "CYP3A4_Veith:::ecfp4"                      #  4  (original idx 34)
    "CYP3A4_Veith:::maccs"                      #  5  (original idx 35)
    "CYP3A4_Veith:::rdkit200"                   #  6  (original idx 36)
    "CYP1A2_Veith:::ecfp4"                      #  7  (original idx 37)
    "CYP1A2_Veith:::maccs"                      #  8  (original idx 38)
    "CYP1A2_Veith:::rdkit200"                   #  9  (original idx 39)
    "CYP2C9_Veith:::ecfp4"                      # 10  (original idx 40)
    "CYP2C9_Veith:::maccs"                      # 11  (original idx 41)
    "CYP2C9_Veith:::rdkit200"                   # 12  (original idx 42)
    "Clearance_Hepatocyte_AZ:::rdkit200"        # 13  (original idx 57)
    "AMES:::maccs"                              # 14  (original idx 62)
    "AMES:::rdkit200"                           # 15  (original idx 63)
    "DILI:::ecfp4"                              # 16  (original idx 64)
    "DILI:::maccs"                              # 17  (original idx 65)
    "DILI:::rdkit200"                           # 18  (original idx 66)
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
    echo "Run job_01_prepare.sh first."
    exit 1
fi

N_CSV=$(find "${INPUT_DIR}" -maxdepth 1 -name "train.csv" | wc -l)
if [ "${N_CSV}" -eq 0 ]; then
    echo "[ERROR] No train.csv found in ${INPUT_DIR}"
    exit 1
fi

cd "${REPO_ROOT}"

DATA_TYPE="${ENDPOINT}_${FEAT}"

# Run qprofiler-batch for this single endpoint/featurizer
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
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
