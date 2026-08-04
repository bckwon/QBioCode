#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4d — QProfiler Sweep RETRY 4 (16 tasks)
#
# Re-submits the 16 tasks that were permanently PEND in array 41217 because
# the original node list became unavailable:
#   zu-a100-c05-01  → unavail
#   zu-a100-c06-01  → unavail
#   zu-a100-c08-03  → closed_Full
#   zu-a100-c05-02  → closed_Full
#
# Node availability at submission time (Tue Jul 29):
#   INCLUDED (ok, A100, meaningful free GPU slots):
#     zu-a100-c08-04  (free: 110)
#     zu-a100-b05-01  (free: 105)
#     zu-a100-c05-04  (free: 105)
#     zu-a100-c08-01  (free:  99)
#     zu-a100-c05-03  (free:  88)
#     zu-a100-c05-05  (free:  40)
#   EXCLUDED (unavail/full/too few):
#     zu-a100-c05-01  — unavail
#     zu-a100-c06-01  — unavail
#     zu-a100-c08-03  — closed_Full
#     zu-a100-c05-02  — closed_Full
#     zu-a100-b05-02  — free: 2 (too few)
#
# Tasks (16): indices 1–6, 9–18 from the 41217 task list,
# which map to the original ADMET task indices as follows:
#   [ 1] CYP2D6_Veith / ecfp4          (orig 31)
#   [ 2] CYP2D6_Veith / maccs           (orig 32)
#   [ 3] CYP2D6_Veith / rdkit200        (orig 33)
#   [ 4] CYP3A4_Veith / ecfp4           (orig 34)
#   [ 5] CYP3A4_Veith / maccs           (orig 35)
#   [ 6] CYP3A4_Veith / rdkit200        (orig 36)
#   [ 9] CYP1A2_Veith / rdkit200        (orig 39)
#   [10] CYP2C9_Veith / ecfp4           (orig 40)
#   [11] CYP2C9_Veith / maccs           (orig 41)
#   [12] CYP2C9_Veith / rdkit200        (orig 42)
#   [13] Clearance_Hepatocyte_AZ / rdkit200 (orig 57)
#   [14] AMES / maccs                   (orig 62)
#   [15] AMES / rdkit200                (orig 63)
#   [16] DILI / ecfp4                   (orig 64)
#   [17] DILI / maccs                   (orig 65)
#   [18] DILI / rdkit200                (orig 66)
# (Tasks 7 & 8 = CYP1A2_Veith ecfp4/maccs are still running in 41217.)
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02d_qprofiler_retry4.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry4[1-16]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-c08-04 zu-a100-b05-01 zu-a100-c05-04 zu-a100-c08-01 zu-a100-c05-03 zu-a100-c05-05"
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

# Explicit task list — 1-based (index 0 unused).
# Preserves the same endpoint/featurizer assignments as 41217,
# minus the two tasks (7 & 8) still running there.
TASKS=(
    ""                                          #  0 — unused (LSF is 1-based)
    "CYP2D6_Veith:::ecfp4"                      #  1  (41217 idx 1,  orig 31)
    "CYP2D6_Veith:::maccs"                      #  2  (41217 idx 2,  orig 32)
    "CYP2D6_Veith:::rdkit200"                   #  3  (41217 idx 3,  orig 33)
    "CYP3A4_Veith:::ecfp4"                      #  4  (41217 idx 4,  orig 34)
    "CYP3A4_Veith:::maccs"                      #  5  (41217 idx 5,  orig 35)
    "CYP3A4_Veith:::rdkit200"                   #  6  (41217 idx 6,  orig 36)
    "CYP1A2_Veith:::rdkit200"                   #  7  (41217 idx 9,  orig 39)
    "CYP2C9_Veith:::ecfp4"                      #  8  (41217 idx 10, orig 40)
    "CYP2C9_Veith:::maccs"                      #  9  (41217 idx 11, orig 41)
    "CYP2C9_Veith:::rdkit200"                   # 10  (41217 idx 12, orig 42)
    "Clearance_Hepatocyte_AZ:::rdkit200"        # 11  (41217 idx 13, orig 57)
    "AMES:::maccs"                              # 12  (41217 idx 14, orig 62)
    "AMES:::rdkit200"                           # 13  (41217 idx 15, orig 63)
    "DILI:::ecfp4"                              # 14  (41217 idx 16, orig 64)
    "DILI:::maccs"                              # 15  (41217 idx 17, orig 65)
    "DILI:::rdkit200"                           # 16  (41217 idx 18, orig 66)
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
