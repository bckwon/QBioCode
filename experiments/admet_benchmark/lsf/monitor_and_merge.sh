#!/usr/bin/env bash
#==============================================================================
# monitor_and_merge.sh — Poll job completion and run merge when ready
#
# Usage (run from repo root in a terminal or screen session):
#   bash experiments/admet_benchmark/lsf/monitor_and_merge.sh
#
# What it does:
#   1. Every 30 minutes, checks whether 75734[16,17,61,62,63] and 76371
#      (Solubility/rdkit200 resubmit) have all completed.
#   2. Once all QSVC tasks are done, runs 06_merge_qsvc_results.py.
#   3. Reports the merged result count.
#==============================================================================

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs/admet"

QSVC_JOB=75734
QSVC_TASKS=(16 17 61 62 63)
SOL_JOB=76371  # Solubility_AqSolDB/rdkit200 single-task resubmit

echo "=============================================="
echo "ADMET QSVC completion monitor"
echo "Started: $(date)"
echo "Watching: ${QSVC_JOB}[${QSVC_TASKS[*]}] + ${SOL_JOB}"
echo "=============================================="

check_done() {
    local job=$1 task=$2
    local log="${LOG_DIR}/qsvc_rerun_${task}_${job}.out"
    if [ -f "$log" ] && grep -q "total run time" "$log" 2>/dev/null; then
        return 0
    fi
    return 1
}

check_sol_done() {
    # Single job — check by job ID directly
    local stat
    stat=$(bjobs "${SOL_JOB}" 2>/dev/null | awk 'NR==2{print $3}' || echo "DONE")
    if [ "$stat" = "DONE" ] || [ "$stat" = "EXIT" ] || ! bjobs "${SOL_JOB}" 2>/dev/null | grep -q "${SOL_JOB}"; then
        return 0
    fi
    return 1
}

POLL_INTERVAL=1800  # 30 minutes

while true; do
    echo ""
    echo "--- $(date) ---"

    all_done=true

    for task in "${QSVC_TASKS[@]}"; do
        if check_done "$QSVC_JOB" "$task"; then
            echo "  [${task}] DONE"
        else
            echo "  [${task}] still running"
            all_done=false
        fi
    done

    if check_sol_done; then
        echo "  [sol_rdkit200] DONE"
    else
        echo "  [sol_rdkit200] still running"
        all_done=false
    fi

    if $all_done; then
        echo ""
        echo "All QSVC tasks complete! Running merge..."
        cd "${REPO_ROOT}"
        "${PYTHON}" experiments/admet_benchmark/06_merge_qsvc_results.py \
            2>&1 | tee "${LOG_DIR}/merge_qsvc_$(date +%Y%m%d_%H%M%S).log"
        echo ""
        echo "Merge complete. Checking result counts..."
        find results/admet_config/ -name "ModelResults.csv" \
            -exec grep -l "qsvc" {} \; 2>/dev/null | wc -l || true
        echo "Done at $(date)"
        break
    fi

    echo "  Sleeping ${POLL_INTERVAL}s..."
    sleep "${POLL_INTERVAL}"
done
