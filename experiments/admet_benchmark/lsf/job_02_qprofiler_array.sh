#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Phase 4 — QProfiler Sweep (one job per endpoint × featurizer)
#
# Runs qprofiler-batch for each of the 22 ADMET endpoints × 3 featurizers
# as an LSF job array (66 independent jobs).  Each job processes one
# endpoint/featurizer directory.
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02_qprofiler_array.sh
#
# Estimated wall time per job: ~2.5 h (QML on simulator, 10 models × 3 splits)
# Total compute: ~66 jobs × 2.5 h = 165 core-hours
#==============================================================================

#BSUB -J admet_qprofiler[1-66]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qprofiler_%I_%J.out
#BSUB -e logs/admet/qprofiler_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
# Auto-requeue on job failure (covers node crashes, scratch errors, UNKWN).
# Up to 3 requeue attempts per array element before giving up.
#BSUB -r
#BSUB -nr 3
# Pin to the proven-healthy GPU node zu-a100-b05-02 (shared mode confirmed working).
#BSUB -R "hname=='zu-a100-b05-02'"

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_config.yaml"

mkdir -p "${LOG_DIR}"

# Build ordered list of all (endpoint, featurizer) pairs — must match array size
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

# LSF array index is 1-based
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
