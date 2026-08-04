#!/usr/bin/env bash
#==============================================================================
# LSF Job: Phase 4f — QProfiler Sweep RETRY 6 (1 task)
#
# Re-runs the single task that failed in array 42835:
#   42835[8] = CYP2C19_Veith / maccs  (orig task index 29)
#
# ROOT CAUSE OF FAILURE:
#   42835[8] was dispatched to zu-a100-c05-03 on Thu Jul 30 22:57:00.
#   LSF reported "Cannot open your job file: /u/bkwon/.lsbatch/..."
#   The job terminated in 1 second — a transient NFS staging failure on
#   c05-03 prevented the job script from being staged before execution.
#   The job never ran (0 QSVC/VQC/QNN logged). All 3 auto-requeues were
#   consumed by the same staging failure and the task expired.
#
# FIX:
#   - Submit as a plain (non-array) job so LSF re-stages the script fresh.
#   - Use nodes with the most free GPU slots right now (Aug 3 ~05:00 CEST):
#       zu-a100-c08-01  (free: 104)
#       zu-a100-c08-02  (free: 108)
#       zu-a100-b05-01  (free:  51)
#       zu-a100-c05-05  (free:  45)
#       zu-a100-c05-03  (free:  35)  <- same node but NFS is stable now
#       zu-a100-c05-01  (free:  21)
#   - Exclude: c05-02 (historically kills jobs), b05-02/c06-01/c08-03 (unavail)
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02f_qprofiler_retry6.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry6_CYP2C19_maccs
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-c08-01 zu-a100-c08-02 zu-a100-b05-01 zu-a100-c05-05 zu-a100-c05-03 zu-a100-c05-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qprofiler_CYP2C19_Veith_maccs_%J.out
#BSUB -e logs/admet/qprofiler_CYP2C19_Veith_maccs_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
# Auto-requeue on node failure (up to 3 attempts).
#BSUB -r
#BSUB -nr 3

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

ENDPOINT="CYP2C19_Veith"
FEAT="maccs"
INPUT_DIR="${REPO_ROOT}/data/admet/${ENDPOINT}/${FEAT}"

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}"
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
