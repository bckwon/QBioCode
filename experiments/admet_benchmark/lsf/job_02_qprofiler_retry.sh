#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: QProfiler Retry — 40 missing endpoint × featurizer tasks
#
# Auto-generated to cover tasks not completed by the original array (32033).
# The TASKS array below maps LSF index [1-40] directly to the missing pairs.
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02_qprofiler_retry.sh
#==============================================================================

#BSUB -J admet_qprofiler_retry[1-40]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=yes"
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

# Explicit list of the 40 missing (endpoint:::featurizer) pairs — 1-based index
TASKS=(
    ""                                                    # index 0 unused (LSF is 1-based)
    "HIA_Hou:::maccs"                                     #  1
    "Lipophilicity_AstraZeneca:::rdkit200"                #  2
    "Solubility_AqSolDB:::ecfp4"                          #  3
    "Solubility_AqSolDB:::maccs"                          #  4
    "Solubility_AqSolDB:::rdkit200"                       #  5
    "BBB_Martins:::ecfp4"                                 #  6
    "BBB_Martins:::rdkit200"                              #  7
    "PPBR_AstraZeneca:::ecfp4"                            #  8
    "PPBR_AstraZeneca:::maccs"                            #  9
    "PPBR_AstraZeneca:::rdkit200"                         # 10
    "VDss_Lombardo:::ecfp4"                               # 11
    "VDss_Lombardo:::maccs"                               # 12
    "VDss_Lombardo:::rdkit200"                            # 13
    "CYP2C19_Veith:::ecfp4"                               # 14
    "CYP2C19_Veith:::maccs"                               # 15
    "CYP2C19_Veith:::rdkit200"                            # 16
    "CYP2D6_Veith:::ecfp4"                                # 17
    "CYP2D6_Veith:::maccs"                                # 18
    "CYP2D6_Veith:::rdkit200"                             # 19
    "CYP3A4_Veith:::ecfp4"                                # 20
    "CYP3A4_Veith:::maccs"                                # 21
    "CYP2C9_Veith:::ecfp4"                                # 22
    "CYP2C9_Veith:::maccs"                                # 23
    "CYP2C9_Veith:::rdkit200"                             # 24
    "CYP2C9_Substrate_CarbonMangels:::ecfp4"              # 25
    "CYP2D6_Substrate_CarbonMangels:::rdkit200"           # 26
    "CYP3A4_Substrate_CarbonMangels:::ecfp4"              # 27
    "CYP3A4_Substrate_CarbonMangels:::maccs"              # 28
    "Clearance_Hepatocyte_AZ:::ecfp4"                     # 29
    "Clearance_Hepatocyte_AZ:::maccs"                     # 30
    "Clearance_Hepatocyte_AZ:::rdkit200"                  # 31
    "hERG:::ecfp4"                                        # 32
    "hERG:::maccs"                                        # 33
    "hERG:::rdkit200"                                     # 34
    "AMES:::ecfp4"                                        # 35
    "AMES:::maccs"                                        # 36
    "AMES:::rdkit200"                                     # 37
    "DILI:::ecfp4"                                        # 38
    "DILI:::maccs"                                        # 39
    "DILI:::rdkit200"                                     # 40
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

# Ensure venv binaries are on PATH so qprofiler-batch is found regardless
# of how this job was submitted
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
