#!/usr/bin/env bash
#BSUB -J admet_qsvc_f4[1-4]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c08-02"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qsvc_f4_%I_%J.out
#BSUB -e logs/admet/qsvc_f4_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2
set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"
mkdir -p "${LOG_DIR}"
TASKS=(
    ""
    "Lipophilicity_AstraZeneca:::rdkit200"
    "PPBR_AstraZeneca:::rdkit200"
    "VDss_Lombardo:::ecfp4"
    "VDss_Lombardo:::maccs"
)
TASK="${TASKS[${LSB_JOBINDEX}]}"
ENDPOINT="${TASK%%:::*}"; FEAT="${TASK##*:::}"
INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"
echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}] | Host: $(hostname) | $(date)"
echo "Endpoint: ${ENDPOINT} | Featurizer: ${FEAT}"
cd "${REPO_ROOT}"
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" qprofiler-batch \
  --input-dir "${INPUT_DIR}" --config "${CONFIG}" \
  --data-type "${DATA_TYPE}" --n-jobs 2 \
  2>&1 | tee "${LOG_DIR}/qsvc_f4_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"
EXIT_CODE=${PIPESTATUS[0]}
echo "Done: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
