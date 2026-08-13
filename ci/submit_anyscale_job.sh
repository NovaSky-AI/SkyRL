#!/usr/bin/env bash
# Submit an Anyscale job, retrying only when the cluster never acquired its GPUs.
#
# On-demand 4xL4 (g6.12xlarge) capacity in us-east-1 is frequently exhausted. When
# that happens the Anyscale control plane cycles availability zones, gives up, and
# terminates the cluster with "failed to acquire min nodes" -- the job goes from
# STARTING straight to FAILED without the entrypoint ever executing. That is a
# capacity problem, not a test problem, so we resubmit instead of failing CI.
#
# Reaching RUNNING is NOT that signal. The CLI collapses Anyscale's internal HA job
# states onto a handful of user-facing ones, and ERRORED / CLEANING_UP / RESTARTING
# all map to RUNNING (see HA_JOB_STATE_TO_JOB_STATE in anyscale/job/_private/job_sdk.py)
# because ERRORED is transient when retries remain. So a cluster that dies during
# provisioning still reports STARTING -> RUNNING for the few seconds it spends tearing
# itself down, and `job wait --state RUNNING` exits 0 on it.
#
# What actually distinguishes the two cases is whether Anyscale ever created a *job
# run*: a run exists only once the cluster came up and the entrypoint was submitted to
# it. So the gate is applied after the job settles -- if it failed without a run, it
# never got GPUs and we resubmit; if it failed with one, the tests really failed and
# we report it as-is.
#
# Usage: ci/submit_anyscale_job.sh <config-file> <job-name> <run-timeout-s> [start-timeout-s]
#
# <job-name> must be unique to the invocation. `job status` and `job wait` resolve a
# name to the most recently created job carrying it, so a name shared with a
# concurrent run would let the two poll each other's jobs. The CI workflows get this
# from `github.run_id` plus `github.run_attempt`; anything else calling this script
# is responsible for its own unique suffix.
#
# Env overrides:
#   ANYSCALE_CLOUD           cloud to submit to (default sky-anyscale-aws-us-east-1)
#   CAPACITY_MAX_ATTEMPTS    submissions before giving up (default 10)
#   CAPACITY_RETRY_DELAY_S   sleep between attempts (default 300)

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <config-file> <job-name> <run-timeout-s> [start-timeout-s]" >&2
    exit 2
fi

CONFIG_FILE="$1"
JOB_NAME="$2"
RUN_TIMEOUT_S="$3"
# Ceiling on time spent in STARTING. Defaults to the run timeout, i.e. no deadline
# of its own, because the failure we retry for terminates the cluster by itself --
# `job wait` sees the terminal state and returns immediately, well under any bound
# set here. A job still in STARTING is instead waiting on an autoscaler node or an
# image pull, and resubmitting neither conjures capacity nor un-restarts the pull.
START_TIMEOUT_S="${4:-$RUN_TIMEOUT_S}"

CLOUD="${ANYSCALE_CLOUD:-sky-anyscale-aws-us-east-1}"
MAX_ATTEMPTS="${CAPACITY_MAX_ATTEMPTS:-5}"
RETRY_DELAY_S="${CAPACITY_RETRY_DELAY_S:-300}"

# Current state of a job by name. `job status` resolves the most recently created
# job with that name, which is why each attempt gets a unique name below.
job_state() {
    anyscale job status --cloud "$CLOUD" --name "$1" --json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state", "UNKNOWN"))' 2>/dev/null \
        || echo "UNKNOWN"
}

# Number of job runs Anyscale created for a job, or "unknown" if the status call
# failed. A run is created when the entrypoint is submitted to a live cluster, so
# zero runs means provisioning never finished.
job_run_count() {
    anyscale job status --cloud "$CLOUD" --name "$1" --json 2>/dev/null \
        | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("runs") or []))' 2>/dev/null \
        || echo "unknown"
}

# True when the entrypoint produced output, i.e. the cluster really did come up.
# Fallback for when the run count is unavailable -- `job logs` needs a different
# credential type than `job status` on some tokens, hence not using it as primary.
entrypoint_produced_logs() {
    local logs
    logs="$(anyscale job logs --cloud "$CLOUD" --name "$1" --head --max-lines 5 2>/dev/null || true)"
    [[ -n "${logs//[[:space:]]/}" ]]
}

# True when the cluster came up far enough to execute the entrypoint. Guards against
# both misreading a fast entrypoint crash as a capacity failure and misreading a
# capacity failure's transient RUNNING as a real test failure.
entrypoint_ran() {
    local runs
    runs="$(job_run_count "$1")"
    if [[ "$runs" == "unknown" ]]; then
        entrypoint_produced_logs "$1"
        return
    fi
    [[ "$runs" -gt 0 ]]
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    run_name="$JOB_NAME"
    if [[ "$attempt" -gt 1 ]]; then
        run_name="${JOB_NAME}-retry${attempt}"
    fi

    echo "--- Anyscale job attempt ${attempt}/${MAX_ATTEMPTS}: ${run_name}"
    anyscale job submit -f "$CONFIG_FILE" --name "$run_name" --timeout "$RUN_TIMEOUT_S"

    started=1
    anyscale job wait --cloud "$CLOUD" --name "$run_name" \
        --state RUNNING --timeout "$START_TIMEOUT_S" || started=0

    if [[ "$started" -eq 1 ]]; then
        # Provisional -- see the note at the top on RUNNING being reported for a
        # cluster that is actually tearing down after failing to provision.
        echo "Job ${run_name} reached RUNNING, waiting for it to finish."
        if anyscale job wait --cloud "$CLOUD" --name "$run_name" --timeout "$RUN_TIMEOUT_S"; then
            exit 0
        fi
    fi

    state="$(job_state "$run_name")"

    # A job can pass through RUNNING between two 10s `job wait` polls, so a missed
    # RUNNING with a terminal SUCCEEDED still means we got GPUs.
    if [[ "$state" == "SUCCEEDED" ]]; then
        echo "Job ${run_name} succeeded."
        exit 0
    fi

    if entrypoint_ran "$run_name"; then
        echo "Job ${run_name} failed (state: ${state}) but the entrypoint ran -- real failure, not retrying." >&2
        exit 1
    fi

    echo "Job ${run_name} failed (state: ${state}) without ever running its entrypoint:" >&2
    echo "treating this as a GPU capacity failure." >&2
    # A `job wait` timeout leaves the job running -- it only stops the client
    # polling. Terminate before resubmitting so we never leave a second cluster
    # competing for the same scarce instance type (or silently running tests
    # whose result nobody reads).
    case "$state" in
        SUCCEEDED | FAILED | TERMINATED) ;;
        *) anyscale job terminate --cloud "$CLOUD" --name "$run_name" || true ;;
    esac
    if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
        echo "Resubmitting in ${RETRY_DELAY_S}s ..." >&2
        sleep "$RETRY_DELAY_S"
    fi
done

echo "Gave up after ${MAX_ATTEMPTS} attempts without ever acquiring GPUs." >&2
exit 1
