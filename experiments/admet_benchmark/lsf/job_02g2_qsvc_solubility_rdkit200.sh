#!/usr/bin/env bash
#==============================================================================
# LSF Job: QSVC Rerun — Solubility_AqSolDB / rdkit200 (task 18 resubmit)
#
# Task 75734[18] crashed with:
#   ValueError: Input X contains infinity or a value too large for dtype('float64')
# on test.csv, column '42'.  Root cause: one molecule in the test split had
# an RDKit descriptor value of inf (likely a division-by-zero in SlogP or
# similar 3D descriptor).
#
# Fix applied (Aug 14 2026):
#   data/admet/Solubility_AqSolDB/rdkit200/test.csv — the single Inf value
#   in column '42' (row 171) was replaced with the column median (-0.014080).
#   The original file is backed up at test.csv.bak_inf.
#
# This script resubmits only this one task using the same admet_qsvc_config.yaml.
# Output goes to results/admet_qsvc_config/ (same tree as job 75734).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02g2_qsvc_solubility_rdkit200.sh
#==============================================================================

#BSUB -J admet_qsvc_sol_rdk
#BSUB -q normal
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qsvc_sol_rdkit200_%J.out
#BSUB -e logs/admet/qsvc_sol_rdkit200_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"

ENDPOINT="Solubility_AqSolDB"
FEAT="rdkit200"
INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "Endpoint     : ${ENDPOINT}"
echo "Featurizer   : ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Config       : ${CONFIG}"
echo "======================================================"
echo "Fix applied  : test.csv col '42' Inf → column median (-0.014080)"
echo "Models       : qsvc only"
echo "Embeddings   : pca, umap"
echo "======================================================"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found in ${INPUT_DIR}"
    exit 1
fi

# Sanity check: verify no Inf values remain in test.csv
echo "Pre-run Inf check on test.csv:"
"${REPO_ROOT}/.venv/bin/python" -c "
import pandas as pd, numpy as np
df = pd.read_csv('${INPUT_DIR}/test.csv')
feat_cols = [c for c in df.columns if c not in ('Y','split')]
n_inf = np.isinf(df[feat_cols].values).sum()
print(f'  test.csv Inf count: {n_inf}')
assert n_inf == 0, 'ERROR: Inf values still present in test.csv!'
print('  OK — no Inf values.')
"

cd "${REPO_ROOT}"

PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     2 \
    2>&1 | tee "${LOG_DIR}/qsvc_sol_rdkit200_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
