#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4e — QProfiler Sweep RETRY 5 (11 tasks)
#
# Re-submits the 11 endpoint/featurizer tasks that are orphaned after
# zu-a100-c05-02 went UNREACH (again) and killed tasks in arrays 40223
# and 40830.  The following tasks from those arrays are UNKWN and will
# not self-recover (auto-requeue would land on the same dead node):
#
#   40223 UNKWN: [21,22,23,24,25,26,27,28]
#   40830 UNKWN: [21,22,23,26,27,29,30,31,41,42,43,54]
#
# After deduplication and removing tasks already covered by still-running
# arrays (40830[25]=VDss/ecfp4, 41742[1-16]=CYP2D6/CYP3A4/CYP1A2/
# CYP2C9/Clearance/AMES/DILI), the 11 true orphans are:
#
#   [ 1] BBB_Martins / rdkit200                 (orig 21)
#   [ 2] PPBR_AstraZeneca / ecfp4               (orig 22)
#   [ 3] PPBR_AstraZeneca / maccs               (orig 23)
#   [ 4] PPBR_AstraZeneca / rdkit200            (orig 24)
#   [ 5] VDss_Lombardo / maccs                  (orig 26)
#   [ 6] VDss_Lombardo / rdkit200               (orig 27)
#   [ 7] CYP2C19_Veith / ecfp4                  (orig 28)
#   [ 8] CYP2C19_Veith / maccs                  (orig 29)
#   [ 9] CYP2C19_Veith / rdkit200               (orig 30)
#   [10] CYP2C9_Substrate_CarbonMangels / ecfp4 (orig 43)
#   [11] Half_Life_Obach / rdkit200              (orig 54)
#
# Node availability at submission time (Thu Jul 30):
#   INCLUDED (ok, meaningful free GPU slots):
#     zu-a100-c05-05  (free: 117)
#     zu-a100-c08-02  (free: 108)
#     zu-a100-c08-04  (free:  65)
#     zu-a100-c05-01  (free:  58)
#     zu-a100-c05-03  (free:  43)
#     zu-a100-b05-01  (free:  37)
#   EXCLUDED:
#     zu-a100-c05-02  — UNREACH (permanently excluded — full /tmp)
#     zu-a100-c08-01  — closed_Full
#     zu-a100-c08-03  — closed_Full
#     zu-a100-b05-02  — too few free slots (20)
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02e_qprofiler_retry5.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry5[1-11]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-c05-05 zu-a100-c08-02 zu-a100-c08-04 zu-a100-c05-01 zu-a100-c05-03 zu-a100-b05-01"
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
TASKS=(
    ""                                              #  0 — unused (LSF is 1-based)
    "BBB_Martins:::rdkit200"                        #  1  (orig 21)
    "PPBR_AstraZeneca:::ecfp4"                      #  2  (orig 22)
    "PPBR_AstraZeneca:::maccs"                      #  3  (orig 23)
    "PPBR_AstraZeneca:::rdkit200"                   #  4  (orig 24)
    "VDss_Lombardo:::maccs"                         #  5  (orig 26)
    "VDss_Lombardo:::rdkit200"                      #  6  (orig 27)
    "CYP2C19_Veith:::ecfp4"                         #  7  (orig 28)
    "CYP2C19_Veith:::maccs"                         #  8  (orig 29)
    "CYP2C19_Veith:::rdkit200"                      #  9  (orig 30)
    "CYP2C9_Substrate_CarbonMangels:::ecfp4"        # 10  (orig 43)
    "Half_Life_Obach:::rdkit200"                    # 11  (orig 54)
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
