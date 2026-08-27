#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L1 Multi-seed Classical@300 ablation
#
# 4 seeds × 22 endpoints × 3 featurisers = 264 tasks
# Each task runs classical models (lr/rf/mlp/xgb/svc) on
# data/admet_seeds/seed_{S}/{ep}/{feat}/train_qml.csv (~300 rows)
# and evaluates against the fixed TDC test.csv.
#
# Classical training is fast (~5 min/endpoint). No GPU required.
# Total: ~22 GPU-hours (CPU time only).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_L1_classical300_seeds.sh
#==============================================================================
#BSUB -J admet_cl300_seeds[1-264]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01 zu-a100-c08-02 zu-a100-c08-03"
# No GPU directive — classical models only
#BSUB -o logs/admet/l1_cl300_seeds_%I_%J.out
#BSUB -e logs/admet/l1_cl300_seeds_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

# Same task list as L1 QSVC — 264 entries
source experiments/admet_benchmark/lsf/l1_task_list.sh

TASK="${TASKS[${LSB_JOBINDEX}]}"
SEED="${TASK%%:::*}"
REST="${TASK#*:::}"
ENDPOINT="${REST%%:::*}"
FEAT="${REST##*:::}"

INPUT_DIR="${REPO_ROOT}/data/admet_seeds/seed_${SEED}/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_classical300_seed_${SEED}.yaml"
DATA_TYPE="${ENDPOINT}_${FEAT}_cl300_seed${SEED}"
LOG="${REPO_ROOT}/logs/admet/l1_cl300_${ENDPOINT}_${FEAT}_seed${SEED}_${LSB_JOBID:-local}.log"

echo "======================================================"
echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}]  $(hostname)  $(date)"
echo "Seed=${SEED}  Endpoint=${ENDPOINT}  Feat=${FEAT}  (classical@300)"
echo "Input: ${INPUT_DIR}"
echo "Config: ${CONFIG}"
echo "======================================================"

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found: ${INPUT_DIR}/train_qml.csv"
    exit 1
fi

PYTHONPATH="${REPO_ROOT}" qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     5 \
    2>&1 | tee "${LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
