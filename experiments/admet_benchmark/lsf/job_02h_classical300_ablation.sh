#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: Classical-at-300 Ablation  (job_02h)
#
# Purpose
# -------
# Runs ONLY classical ML models (lr, rf, mlp, xgb, svc) on the QML-capped
# training split (train_qml.csv, ≤300 samples) to isolate the DATA EFFECT
# from the MODEL EFFECT in the quantum-classical comparison.
#
# Scientific motivation
# ---------------------
# The main benchmark (job_02) trains classical models on the full dataset
# (up to 13,000 samples) while QML is capped at 300.  This ablation trains
# the same classical models on the same ≤300 samples QML uses.
# Comparing ablation AUC vs main-benchmark AUC:
#   (a) If ablation ≈ QML  →  data starvation explains the gap, not circuits.
#   (b) If ablation > QML  →  classical feature space (full fingerprint vs 8 PCA
#       components) provides additional advantage beyond sample count.
#   (c) If ablation < QML  →  genuine quantum data-efficiency signal.
#
# Config
# ------
# qbiocode/apps/qprofiler/configs/admet_classical300_config.yaml
#   - model: ['lr', 'rf', 'mlp', 'xgb', 'svc']  (no QML)
#   - file_dataset: 'ALL'  → batch runs train_qml.csv + test/valid/train
#   - embeddings: ['none', 'pca', 'umap']  (matches main benchmark)
#   - config_file_name: 'admet_classical300_config'
#
# Output location (fully isolated, no collision risk)
# ---------------------------------------------------
# results/admet_classical300_config/dataset={split}/simulator_{ts}/
#     ├── ModelResults.csv          ← per-model AUC/F1/MCC results
#     ├── RawDataEvaluation.csv     ← dataset statistics
#     └── .hydra/config.yaml        ← full config snapshot with ablation metadata
#
# Aggregated batch output (one file per task):
# results/{EP}_{FEAT}_classical300_batch_{ts}/ModelResults.csv
#
# Wall-time estimate
# ------------------
# Classical-only, 5 models × 3 embeddings × 4 splits = 60 cells per task.
# 300-sample training → each cell takes seconds to minutes.
# Estimated: 15–60 min per task (worst case: UMAP fit on large fingerprints).
#
# Tasks: 22 endpoints × 3 featurizers = 66 total
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02h_classical300_ablation.sh
#==============================================================================

#BSUB -J admet_classical300[1-66]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32000]"
#BSUB -o logs/admet/classical300_%I_%J.out
#BSUB -e logs/admet/classical300_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
# Auto-requeue on transient node failure (max 2 retries — tasks are fast so
# the cost of a spurious restart is low).
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_classical300_config.yaml"

mkdir -p "${LOG_DIR}"

# ── Task list: identical order to job_02g_qsvc_rerun.sh (22 endpoints × 3 feats) ──
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

# ── Header ───────────────────────────────────────────────────────────────────
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
echo "Ablation     : classical_at_300"
echo "Models       : lr rf mlp xgb svc (NO QML)"
echo "Train input  : train_qml.csv (≤300 samples — same as QML)"
echo "Test input   : test.csv (same test set as main benchmark)"
echo "Embeddings   : none pca umap"
echo "Output tree  : results/admet_classical300_config/"
echo "======================================================"

# ── Validate input ────────────────────────────────────────────────────────────
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found in ${INPUT_DIR}"
    echo "This file must exist (created by job_01_prepare.sh)."
    exit 1
fi

# Sanity-check: train_qml.csv should have ≤301 lines (1 header + ≤300 rows)
TRAIN_QML_LINES=$(wc -l < "${INPUT_DIR}/train_qml.csv")
if [ "${TRAIN_QML_LINES}" -gt 302 ]; then
    echo "[WARNING] train_qml.csv has ${TRAIN_QML_LINES} lines (expected ≤301). Proceeding."
fi
echo "train_qml.csv lines: ${TRAIN_QML_LINES} (≤301 expected)"

cd "${REPO_ROOT}"

# ── Run classical-only sweep ──────────────────────────────────────────────────
# qprofiler-batch processes all 4 CSV files in INPUT_DIR in parallel.
# The key file is train_qml.csv; train.csv/valid.csv/test.csv runs provide
# full-feature classical baselines in the same output dir for reference.
#
# Output: results/{DATA_TYPE}_classical300_batch_{ts}/ModelResults.csv
# The "_classical300_batch_" infix makes these dirs instantly distinguishable
# from the main benchmark ("_batch_") and QSVC rerun dirs.
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
