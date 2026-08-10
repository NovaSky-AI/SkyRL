#!/usr/bin/env bash
# Submit an Anyscale job, retrying only when the cluster never acquired its GPUs.
#
# On-demand 4xL4 (g6.12xlarge) capacity in us-east-1 is frequently exhausted. When
# that happens the Anyscale control plane cycles availability zones, gives up, and
# terminates the cluster with "failed to acquire min nodes" -- the job goes from
# STARTING straight to FAILED without the entrypoint ever executing. That is a
# capacity problem, not a test problem, so we resubmit instead of failing CI.
#
# Reaching RUNNING is the signal that provisioning succeeded, so `anyscale job wait
# --state RUNNING` is the gate: it exits 0 on RUNNING, and non-zero if the job hits
# a terminal state first. Anything that fails *after* RUNNING is a real test failure
# and is reported as-is.
#
# Usage: ci/submit_anyscale_job.sh <config-file> <job-name> <run-timeout-s> [start-timeout-s]
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

# True when the entrypoint produced output, i.e. the cluster really did come up.
# Guards against misreading a fast entrypoint crash as a capacity failure.
entrypoint_produced_logs() {
    local logs
    logs="$(anyscale job logs --cloud "$CLOUD" --name "$1" --head --max-lines 5 2>/dev/null || true)"
    [[ -n "${logs//[[:space:]]/}" ]]
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    run_name="$JOB_NAME"
    if [[ "$attempt" -gt 1 ]]; then
        run_name="${JOB_NAME}-retry${attempt}"
    fi

    echo "--- Anyscale job attempt ${attempt}/${MAX_ATTEMPTS}: ${run_name}"
    anyscale job submit -f "$CONFIG_FILE" --name "$run_name" --timeout "$RUN_TIMEOUT_S"

    started=1
    anyscale job wait --cloud "$CLOUD" --name "$run_name" \
        --state RUNNING --timeout "$START_TIMEOUT_S" || started=0

    state="$(job_state "$run_name")"

    if [[ "$started" -eq 0 ]]; then
        # A job can pass through RUNNING between two 10s `job wait` polls, so a
        # missed RUNNING with a terminal SUCCEEDED still means we got GPUs.
        if [[ "$state" == "SUCCEEDED" ]]; then
            echo "Job ${run_name} succeeded before RUNNING was observed."
            exit 0
        fi
        if entrypoint_produced_logs "$run_name"; then
            echo "Job ${run_name} failed (state: ${state}) but the entrypoint ran -- real failure, not retrying." >&2
            exit 1
        fi
        echo "Job ${run_name} never reached RUNNING (state: ${state}) and the entrypoint never ran:" >&2
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
        continue
    fi

    echo "Job ${run_name} reached RUNNING -- GPUs acquired, waiting for the tests to finish."
    anyscale job wait --cloud "$CLOUD" --name "$run_name" --timeout "$RUN_TIMEOUT_S"
    exit 0
done

echo "Gave up after ${MAX_ATTEMPTS} attempts without ever acquiring GPUs." >&2
exit 1
