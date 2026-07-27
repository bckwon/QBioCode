#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4b — QProfiler Sweep RETRY (indices 21–66)
#
# Re-submits the 46 array elements that never ran or were killed by LSF due to
# the GPFS/node outage (Jul 22-24).  The hard pin to zu-a100-b05-02 has been
# removed; jobs are spread across all A100 nodes that are currently healthy
# (status=ok as of re-submission, excluding the two closed_Full nodes).
#
# Available A100 nodes (ok, not closed_Full) at time of submission:
#   zu-a100-b05-01  (free slots: 126)
#   zu-a100-b05-02  (free slots:   3)  — running existing array 40223
#   zu-a100-c05-02  (free slots: 121)
#   zu-a100-c05-04  (free slots: 116)
#   zu-a100-c05-05  (free slots: 121)
#   zu-a100-c06-01  (free slots: 120)
#   zu-a100-c08-01  (free slots:  57)
#   zu-a100-c08-02  (free slots: 126)
#   zu-a100-c08-03  (free slots:  89)
#   zu-a100-c08-04  (free slots: 125)
# Excluded (closed_Full): zu-a100-c05-01, zu-a100-c05-03
#
# GPU mode: shared + j_exclusive=no — matches the original sweep.
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02b_qprofiler_retry.sh
#
# Estimated wall time per job: ~2.5 h (QML on simulator, 10 models × 3 splits)
# Total compute: ~46 jobs × 2.5 h ≈ 115 core-hours
#==============================================================================

#BSUB -J admet_qprofiler_retry[21-66]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-b05-01 zu-a100-b05-02 zu-a100-c05-02 zu-a100-c05-04 zu-a100-c05-05 zu-a100-c06-01 zu-a100-c08-01 zu-a100-c08-02 zu-a100-c08-03 zu-a100-c08-04"
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

# Build ordered list of all (endpoint, featurizer) pairs — must match array size.
# The task index mapping is IDENTICAL to job_02_qprofiler_array.sh so that
# LSB_JOBINDEX 21..66 resolves to the same endpoint/featurizer as the original.
ENDPOINTS=(
    Caco2_Wang HIA_Hou Pgp_Broccatelli Bioavailability_Ma
    Lipophilicity_AstraZeneca Solubility_AqSolDB BBB_Martins
    PPBR_AstraZeneca VDss_Lombardo
    CYP2C19_Veith CYP2D6_Veith CYP3A4_Veith CYP1A2_Veith CYP2C9_Veith
    CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels
    CYP3A4_Substrate_CarbonMangels
    Half_Life_Obach Clearance_Hepatocyte_AZ
    hERG AMES DILI
)
FEATURIZERS=(ecfp4 maccs rdkit200)

# Build flat task list: task_index → (endpoint, featurizer)
TASKS=()
for EP in "${ENDPOINTS[@]}"; do
    for FEAT in "${FEATURIZERS[@]}"; do
        TASKS+=("${EP}:::${FEAT}")
    done
done

# LSF array index is 1-based; subtract 1 to index into 0-based TASKS array
TASK_IDX=$(( LSB_JOBINDEX - 1 ))
TASK="${TASKS[${TASK_IDX}]}"
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
