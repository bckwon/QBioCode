#!/usr/bin/env bash
#BSUB -J qsvc_vdss_maccs
#BSUB -q normal
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01 zu-a100-c08-02"
#BSUB -n 1
#BSUB -R "span[hosts=1] rusage[mem=8000]"
#BSUB -M 8000
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qsvc_vdss_maccs_%J.out
#BSUB -e logs/admet/qsvc_vdss_maccs_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"

ENDPOINT="VDss_Lombardo"
FEAT="maccs"
INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"

echo "Job ${LSB_JOBID:-local} | Host: $(hostname) | $(date)"
echo "Endpoint: ${ENDPOINT} | Featurizer: ${FEAT}"

cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

qprofiler-batch \
    --input-dir "${INPUT_DIR}" \
    --config    "${CONFIG}" \
    --data-type "${DATA_TYPE}" \
    --n-jobs 2 \
    2>&1 | tee "${LOG_DIR}/qsvc_vdss_maccs_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "Done: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
