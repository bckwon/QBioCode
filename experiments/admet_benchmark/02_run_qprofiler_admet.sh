#!/usr/bin/env bash
# =============================================================================
# 02_run_qprofiler_admet.sh
# =============================================================================
# Phase 4 experiment script: run QProfiler across all 22 ADMET endpoints
# and 3 featurizers using qprofiler-batch.
#
# Usage (from repo root):
#   bash experiments/admet_benchmark/02_run_qprofiler_admet.sh
#
# Environment variables (override defaults):
#   DATA_DIR     - root data directory          (default: data/admet)
#   RESULTS_DIR  - output results directory     (default: results/admet_benchmark)
#   N_JOBS       - parallel jobs per endpoint   (default: 10)
#   BACKEND      - qiskit backend               (default: simulator)
#   FEATURIZERS  - space-separated list         (default: "ecfp4 maccs rdkit200")
#   ENDPOINTS    - space-separated list or ALL  (default: ALL)
# =============================================================================

set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/admet}"
RESULTS_DIR="${RESULTS_DIR:-results/admet_benchmark}"
N_JOBS="${N_JOBS:-10}"
BACKEND="${BACKEND:-simulator}"
FEATURIZERS="${FEATURIZERS:-ecfp4 maccs rdkit200}"
CONFIG="qbiocode/apps/qprofiler/configs/admet_config.yaml"

# All 22 TDC ADMET endpoints
ALL_ENDPOINTS=(
    Caco2_Wang HIA_Hou Pgp_Broccatelli Bioavailability_Ma
    Lipophilicity_AstraZeneca Solubility_AqSolDB BBB_Martins
    PPBR_AstraZeneca VDss_Lombardo
    CYP2C19_Veith CYP2D6_Veith CYP3A4_Veith CYP1A2_Veith CYP2C9_Veith
    CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels
    CYP3A4_Substrate_CarbonMangels
    Half_Life_Obach Clearance_Hepatocyte_AZ
    hERG AMES DILI
)

ENDPOINTS_TO_RUN=("${ALL_ENDPOINTS[@]}")

mkdir -p "${RESULTS_DIR}"

echo "======================================================================"
echo "QBioCode-ADMET: QProfiler Sweep"
echo "======================================================================"
echo "Data dir    : ${DATA_DIR}"
echo "Results dir : ${RESULTS_DIR}"
echo "Config      : ${CONFIG}"
echo "N jobs      : ${N_JOBS}"
echo "Backend     : ${BACKEND}"
echo "Featurizers : ${FEATURIZERS}"
echo "Endpoints   : ${#ENDPOINTS_TO_RUN[@]} total"
echo "======================================================================"

TIMESTAMP=$(date -u +"%Y-%m-%d_%H_%M_%S")
TOTAL_RUNS=0
FAILED_RUNS=0

for ENDPOINT in "${ENDPOINTS_TO_RUN[@]}"; do
    for FEAT in ${FEATURIZERS}; do
        INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"

        if [ ! -d "${INPUT_DIR}" ]; then
            echo "[WARN] Skipping ${ENDPOINT}/${FEAT} — directory not found: ${INPUT_DIR}"
            continue
        fi

        # Count CSV files
        N_FILES=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.csv" | wc -l)
        if [ "${N_FILES}" -eq 0 ]; then
            echo "[WARN] Skipping ${ENDPOINT}/${FEAT} — no CSV files in ${INPUT_DIR}"
            continue
        fi

        DATA_TYPE="${ENDPOINT}_${FEAT}"
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Running: ${ENDPOINT} / ${FEAT}  (${N_FILES} CSV files)"
        echo "----------------------------------------------------------------------"

        # Build per-run config with overrides
        RUN_RESULTS="${RESULTS_DIR}/${DATA_TYPE}_${TIMESTAMP}"
        mkdir -p "${RUN_RESULTS}"

        qprofiler-batch \
            --input-dir "${INPUT_DIR}" \
            --config "${CONFIG}" \
            --data-type "${DATA_TYPE}" \
            --n-jobs "${N_JOBS}" \
            2>&1 | tee "${RUN_RESULTS}/qprofiler_batch.log" \
            && TOTAL_RUNS=$((TOTAL_RUNS + 1)) \
            || { echo "[ERROR] Failed: ${ENDPOINT}/${FEAT}"; FAILED_RUNS=$((FAILED_RUNS + 1)); }
    done
done

echo ""
echo "======================================================================"
echo "Sweep complete"
echo "  Total runs  : ${TOTAL_RUNS}"
echo "  Failed runs : ${FAILED_RUNS}"
echo "  Results in  : ${RESULTS_DIR}"
echo "======================================================================"
