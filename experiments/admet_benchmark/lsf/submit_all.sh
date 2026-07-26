#!/usr/bin/env bash
#==============================================================================
# submit_all.sh — QBioCode-ADMET Full Pipeline Orchestrator
#==============================================================================
# Submits all pipeline jobs in sequence using LSF dependency expressions
# (-w "done(JOB_ID)") so each phase starts only when its predecessor
# succeeds cleanly.
#
# Dependency chain:
#
#   job_01_prepare          (Phase 1 — data download + featurization)
#        ↓  done()
#   job_02_qprofiler[1-66]  (Phase 4 — QProfiler sweep, job array)
#        ↓  done()
#       ┌──────────────────────┐
#   job_03_qsage          job_04_test_inference   (parallel)
#        ↓  done()               ↓  done()
#       └───────────┬────────────┘
#               job_05_export_table
#
# Usage (from repo root):
#   bash experiments/admet_benchmark/lsf/submit_all.sh
#
# Options (environment variables):
#   DRY_RUN=1                  — print commands without submitting
#   QUEUE=night                — override queue (default: normal)
#   NOTIFY_EMAIL=you@host.com  — add -u/-N to all jobs
#==============================================================================

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LSF_DIR="${REPO_ROOT}/experiments/admet_benchmark/lsf"
LOG_DIR="${REPO_ROOT}/logs/admet"
QUEUE="${QUEUE:-normal}"
DRY_RUN="${DRY_RUN:-0}"
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# submit_job SCRIPT [DEPENDENCY]
#
# Submits SCRIPT via bsub and prints ONLY the numeric job ID to stdout.
# All diagnostic messages go to stderr so they are never captured by $().
#
# DEPENDENCY is the raw LSF -w expression, e.g. "done(12345)" or
# "done(12345) && done(67890)".  Pass "" or omit for no dependency.
#
# Bug fix vs previous version:
#   The old function mixed echo (stdout) with the job-ID echo (stdout), so
#   JOB_ID=$(submit ...) captured multi-line garbage instead of just the
#   numeric ID, breaking every downstream -w "done(${JOB_ID})" expression.
#   Fix: all diagnostic output now goes to >&2 (stderr); only the bare
#   numeric job ID is written to stdout.  bsub args are passed directly
#   (no eval) so quoting is never an issue.
# ---------------------------------------------------------------------------
submit_job() {
    local script="$1"
    local dependency="${2:-}"

    # Build bsub argument list — no eval, no string concatenation with quotes
    local -a bsub_args=(-q "${QUEUE}")

    if [ -n "${dependency}" ]; then
        bsub_args+=(-w "${dependency}")
    fi
    if [ -n "${NOTIFY_EMAIL}" ]; then
        bsub_args+=(-u "${NOTIFY_EMAIL}" -N)
    fi

    echo "  ── Submitting: $(basename "${script}")" >&2
    if [ -n "${dependency}" ]; then
        echo "     Dependency: ${dependency}" >&2
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [DRY RUN] bsub ${bsub_args[*]} < ${script}" >&2
        # Print a fake numeric ID to stdout so the caller's $() capture works
        echo "9999999"
        return 0
    fi

    # Submit: bsub reads job script from stdin, args are separate tokens
    local raw_output
    raw_output=$(bsub "${bsub_args[@]}" < "${script}" 2>&1)

    # Extract numeric job ID from: "Job <NNNN> is submitted to queue <normal>."
    local job_id
    job_id=$(echo "${raw_output}" | grep -oP '(?<=Job <)\d+(?=>)')

    if [ -z "${job_id}" ]; then
        echo "  [ERROR] bsub did not return a job ID. Full output:" >&2
        echo "${raw_output}" >&2
        exit 1
    fi

    echo "  Job submitted: ID=${job_id}" >&2
    echo "  Full output  : ${raw_output}" >&2

    # Only the bare job ID goes to stdout — this is what $() captures
    echo "${job_id}"
}

# ---------------------------------------------------------------------------
# Banner (all to stdout — user sees it in the terminal)
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "QBioCode-ADMET Pipeline Orchestrator"
echo "======================================================================"
echo "Repo root  : ${REPO_ROOT}"
echo "Queue      : ${QUEUE}"
echo "Dry run    : ${DRY_RUN}"
echo "Log dir    : ${LOG_DIR}"
[ -n "${NOTIFY_EMAIL}" ] && echo "Notify     : ${NOTIFY_EMAIL}"
echo "======================================================================"
echo ""
echo "Submission sequence:"
echo "  [1] job_01_prepare.sh          — data download + featurization"
echo "  [2] job_02_qprofiler_array.sh  — QProfiler sweep (array [1-66])"
echo "  [3] job_03_qsage.sh            — QSage training       (after [2])"
echo "  [4] job_04_test_inference.sh   — test-set inference   (after [2])"
echo "  [5] job_05_export_table.sh     — performance tables   (after [3]+[4])"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Data preparation (no dependency)
# ---------------------------------------------------------------------------
echo "── Step 1: Data Preparation ──────────────────────────────────────────"
JOB1_ID=$(submit_job "${LSF_DIR}/job_01_prepare.sh")
echo "  Job 1 ID: ${JOB1_ID}"

# ---------------------------------------------------------------------------
# Step 2 — QProfiler array
# Waits for job 1 to DONE (all tasks succeeded).
# For a job array, "done(JOBID)" means every array element finished OK.
# ---------------------------------------------------------------------------
echo ""
echo "── Step 2: QProfiler Sweep (array [1-66]) ────────────────────────────"
JOB2_ID=$(submit_job "${LSF_DIR}/job_02_qprofiler_array.sh" "done(${JOB1_ID})")
echo "  Job 2 ID: ${JOB2_ID}"

# ---------------------------------------------------------------------------
# Steps 3 & 4 — QSage + test inference (both wait for full array to finish)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 3: QSage Training ────────────────────────────────────────────"
JOB3_ID=$(submit_job "${LSF_DIR}/job_03_qsage.sh" "done(${JOB2_ID})")
echo "  Job 3 ID: ${JOB3_ID}"

echo ""
echo "── Step 4: Test-Set Inference ────────────────────────────────────────"
JOB4_ID=$(submit_job "${LSF_DIR}/job_04_test_inference.sh" "done(${JOB2_ID})")
echo "  Job 4 ID: ${JOB4_ID}"

# ---------------------------------------------------------------------------
# Step 5 — Export tables (waits for BOTH step 3 AND step 4)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 5: Export Performance Tables ────────────────────────────────"
JOB5_ID=$(submit_job "${LSF_DIR}/job_05_export_table.sh" \
    "done(${JOB3_ID}) && done(${JOB4_ID})")
echo "  Job 5 ID: ${JOB5_ID}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "All jobs submitted successfully."
echo ""
printf "  %-12s %-10s %s\n" "Phase" "Job ID" "Dependency"
printf "  %-12s %-10s %s\n" "─────────────" "──────────" "────────────────────"
printf "  %-12s %-10s %s\n" "prepare"     "${JOB1_ID}" "(none)"
printf "  %-12s %-10s %s\n" "qprofiler"   "${JOB2_ID}" "done(${JOB1_ID})"
printf "  %-12s %-10s %s\n" "qsage"       "${JOB3_ID}" "done(${JOB2_ID})"
printf "  %-12s %-10s %s\n" "test_infer"  "${JOB4_ID}" "done(${JOB2_ID})"
printf "  %-12s %-10s %s\n" "export_tbl"  "${JOB5_ID}" "done(${JOB3_ID}) && done(${JOB4_ID})"
echo ""
echo "Monitor with:"
echo "  bjobs -u \$(whoami)             # all your jobs"
echo "  bjobs -J admet_qprofiler       # watch array progress"
echo "  bjobs -J admet_qprofiler[1]    # single array element"
echo "  bpeek <JOB_ID>                 # peek at running job stdout"
echo "  bkill <JOB_ID>                 # cancel a job"
echo ""
echo "Logs in    : ${LOG_DIR}/"
echo "Results in : ${REPO_ROOT}/results/admet_benchmark/"
echo "======================================================================"
