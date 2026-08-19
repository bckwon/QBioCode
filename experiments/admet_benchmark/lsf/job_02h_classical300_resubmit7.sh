#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Classical-at-300 Ablation — Resubmit of 7 missing tasks
#
# Tasks: Caco2_Wang ×3, HIA_Hou ×3, Pgp_Broccatelli/ecfp4
# Reason for resubmit: original 76124[1-7] landed on zu-a100-c05-01 and
# produced zero output (node appeared healthy but jobs stuck with 0 CPU/MEM).
#
# Fixes applied vs original job:
#   - Explicit node list: only healthy, lightly-loaded A100 nodes
#   - Excludes: c05-01 (stuck), c05-02 (saturated 100%), c05-04 (closed),
#               c05-05 (unreachable), c08-01 (saturated 100%)
#   - Memory corrected to 32000 MB (observed peak ~28.6 GB in completed tasks)
#   - No wall-time limit
#   - No GPU directive (classical-only, no CUDA needed)
#==============================================================================

#BSUB -J admet_classical300_r2[1-7]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32000]"
#BSUB -m "zu-a100-b05-01 zu-a100-b05-02 zu-a100-c05-03 zu-a100-c06-01 zu-a100-c08-02 zu-a100-c08-03 zu-a100-c08-04"
#BSUB -o logs/admet/classical300_r2_%I_%J.out
#BSUB -e logs/admet/classical300_r2_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_classical300_config.yaml"

TASKS=(
    ""                              # 0 unused
    "Caco2_Wang:::ecfp4"            # 1
    "Caco2_Wang:::maccs"            # 2
    "Caco2_Wang:::rdkit200"         # 3
    "HIA_Hou:::ecfp4"               # 4
    "HIA_Hou:::maccs"               # 5
    "HIA_Hou:::rdkit200"            # 6
    "Pgp_Broccatelli:::ecfp4"       # 7
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
echo "Endpoint     : ${ENDPOINT} / ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Ablation     : classical_at_300 (resubmit r2 — was stuck on c05-01)"
echo "======================================================"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input dir not found: ${INPUT_DIR}"; exit 1
fi
if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv missing in ${INPUT_DIR}"; exit 1
fi

lines=$(wc -l < "${INPUT_DIR}/train_qml.csv")
echo "train_qml.csv: ${lines} lines (≤301 expected)"

cd "${REPO_ROOT}"
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}_classical300" \
    --n-jobs     5 \
    2>&1 | tee "${LOG_DIR}/classical300_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
