#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: QSVC Re-Run (post kernel-collapse fix)
#
# Re-runs ONLY the QSVC model across all 22 ADMET endpoints × 3 featurizers
# = 66 tasks after two fixes applied on 2026-08-04:
#
#   Fix 1 — compute_qsvc.py: MinMaxScaler(0,1) re-scale applied to X_train/
#            X_test BEFORE the quantum kernel.  ZZFeatureMap produces near-zero
#            off-diagonal kernel values for inputs outside [0,1] (concentration
#            of measure), causing the SVM to predict only the majority class.
#            PCA/UMAP output is zero-centred so always needs this rescaling.
#
#   Fix 2 — admet_config.yaml: C changed from 0.01 → 1.0.  C=0.01 is so
#            heavily regularised that no separating boundary can form even
#            when the kernel has some signal.
#
# Config used: qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml
#   - model: ['qsvc'] only
#   - embeddings: ['pca', 'umap']  ('none' always skipped — fingerprints >> 26 qubits)
#   - n_jobs: 2
#
# Results are written to a FRESH results/admet_qsvc_config/ directory tree
# (new Hydra timestamp per run).  Existing results for all other models
# (lr, rf, mlp, xgb, svc, vqc, qnn, qensemble, mmelon) are NEVER touched.
#
# Prerequisites:
#   - All job_02 array tasks should be complete (or at least not writing to
#     the same results dirs) before submitting this job.
#   - data/admet/{endpoint}/{featurizer}/ directories must exist (job_01 done).
#
# Submit from repo root after all job_02 arrays are finished:
#   bsub < experiments/admet_benchmark/lsf/job_02g_qsvc_rerun.sh
#
# Estimated wall time per task: ~2–6 h
#   (QSVC on pca+umap, ≤300 train samples, 8 qubits, simulator)
# Total compute: 66 tasks, parallelised over nodes
#==============================================================================

#BSUB -J admet_qsvc_rerun[1-66]
#BSUB -q normal
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qsvc_rerun_%I_%J.out
#BSUB -e logs/admet/qsvc_rerun_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
# Auto-requeue on node failure, up to 3 attempts
#BSUB -r
#BSUB -nr 3
# Exclude known-problematic nodes
#BSUB -R "hname!='zu-a100-c05-05'"

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"

mkdir -p "${LOG_DIR}"

# ── Task list: identical order to job_02_qprofiler_array.sh ──────────────────
# Index i → TASKS[i-1] maps array element to (endpoint, featurizer)
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

TASKS=()
for EP in "${ENDPOINTS[@]}"; do
    for FEAT in "${FEATURIZERS[@]}"; do
        TASKS+=("${EP}:::${FEAT}")
    done
done

TASK_IDX=$(( LSB_JOBINDEX - 1 ))
TASK="${TASKS[${TASK_IDX}]}"
ENDPOINT="${TASK%%:::*}"
FEAT="${TASK##*:::}"

INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}[${LSB_JOBINDEX:-1}]"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "Task index   : ${LSB_JOBINDEX} / 66"
echo "Endpoint     : ${ENDPOINT}"
echo "Featurizer   : ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Config       : ${CONFIG}"
echo "======================================================"
echo "Fix applied  : ZZFeatureMap MinMaxScaler(0,1) pre-rescaling + C=1.0"
echo "Models       : qsvc only"
echo "Embeddings   : pca, umap (no 'none' — skipped by qubit budget)"
echo "======================================================"

# ── Validate input ────────────────────────────────────────────────────────────
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    echo "Run job_01_prepare.sh first."
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found in ${INPUT_DIR}"
    echo "QSVC uses the QML-capped training split (≤300 samples)."
    exit 1
fi

cd "${REPO_ROOT}"

# ── Run QSVC-only sweep ───────────────────────────────────────────────────────
# qprofiler-batch creates a fresh timestamped results dir — no overwrite of
# existing results for other models.
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     2 \
    2>&1 | tee "${LOG_DIR}/qsvc_rerun_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
