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
# A failed CLI call is not that signal either. `_resolve_to_job_model` rewrites any
# exception out of `get_job` -- including the `ValueError: Rate limit exceeded (user
# scope)` that concurrent GPU workflows provoke -- into `RuntimeError: Job with name
# '<name>' was not found.` A rate-limited `job wait` returns instantly and looks just
# like a cluster that died in STARTING. Hence the retries below, and the rule that we
# only resubmit once the previous job is confirmed terminal.
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
#   CAPACITY_MAX_ATTEMPTS    submissions before giving up (default 5)
#   CAPACITY_RETRY_DELAY_S   sleep between attempts (default 300)
#   ANYSCALE_API_ATTEMPTS    tries per control-plane call (default 6)
#   ANYSCALE_JOBS_FILE       where submitted job names are recorded, for
#                            ci/terminate_anyscale_jobs.sh to clean up after a cancel

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
MAX_ATTEMPTS="${CAPACITY_MAX_ATTEMPTS:-10}"
RETRY_DELAY_S="${CAPACITY_RETRY_DELAY_S:-300}"
API_ATTEMPTS="${ANYSCALE_API_ATTEMPTS:-6}"
# Tries at getting the job into a terminal state before we refuse to resubmit.
TERMINATE_ATTEMPTS="${ANYSCALE_TERMINATE_ATTEMPTS:-5}"
# Consecutive unreadable status polls before we stop waiting on a job.
UNKNOWN_POLL_LIMIT="${ANYSCALE_UNKNOWN_POLL_LIMIT:-10}"

API_ERR_LOG="$(mktemp)"

# Every name we submit, so a cancelled run can still be cleaned up. Cancelling a
# workflow (`cancel-in-progress`, the UI button, a job timeout) kills this script,
# not the cluster it is waiting on -- the GPUs stay allocated until someone
# terminates the job.
JOBS_FILE="${ANYSCALE_JOBS_FILE:-${RUNNER_TEMP:-/tmp}/anyscale-submitted-jobs.txt}"
: > "$JOBS_FILE"

CURRENT_RUN_NAME=""

# Best-effort terminate on the way out. The runner allows only a short grace period
# after signalling, so this gets two quick tries rather than the usual backoff; the
# `if: cancelled()` cleanup step in the workflow is the reliable backstop.
on_signal() {
    trap - INT TERM
    if [[ -n "$CURRENT_RUN_NAME" ]]; then
        echo "Interrupted -- terminating ${CURRENT_RUN_NAME}." >&2
        anyscale job terminate --cloud "$CLOUD" --name "$CURRENT_RUN_NAME" \
            || anyscale job terminate --cloud "$CLOUD" --name "$CURRENT_RUN_NAME" \
            || echo "Terminate failed; ci/terminate_anyscale_jobs.sh must finish the job off." >&2
    fi
    exit 143
}

trap 'rm -f "$API_ERR_LOG"' EXIT
trap on_signal INT TERM

# Retry a control-plane call with exponential backoff; the rate limit advertises a
# retry-after of a few seconds. Only stdout is forwarded, since callers parse it as JSON.
anyscale_retry() {
    local out delay=5 i
    for ((i = 1; i <= API_ATTEMPTS; i++)); do
        if out="$("$@" 2>"$API_ERR_LOG")"; then
            printf '%s' "$out"
            return 0
        fi
        if ((i < API_ATTEMPTS)); then
            echo "  control-plane call failed (${i}/${API_ATTEMPTS}), retrying in ${delay}s: $*" >&2
            sleep "$delay"
            delay=$((delay * 2))
            if ((delay > 60)); then delay=60; fi
        fi
    done
    echo "  control-plane call failed after ${API_ATTEMPTS} attempts: $*" >&2
    cat "$API_ERR_LOG" >&2
    return 1
}

# Echoes "<state> <run-count>". A run is created once the entrypoint is submitted to a
# live cluster, so zero runs means provisioning never finished. Both come from a single
# status call to halve the request count. Unreadable status is "UNKNOWN unknown".
job_status() {
    local json
    if ! json="$(anyscale_retry anyscale job status --cloud "$CLOUD" --name "$1" --json)"; then
        echo "UNKNOWN unknown"
        return 0
    fi
    printf '%s' "$json" | python3 -c '
import json, sys

try:
    doc = json.load(sys.stdin)
    print(doc.get("state") or "UNKNOWN", len(doc.get("runs") or []))
except Exception:
    print("UNKNOWN unknown")
'
}

job_state() {
    local state runs
    read -r state runs <<<"$(job_status "$1")"
    echo "$state"
}

is_terminal() {
    case "$1" in
        SUCCEEDED | FAILED | TERMINATED) return 0 ;;
        *) return 1 ;;
    esac
}

# True when the entrypoint produced output, i.e. the cluster really did come up.
# Fallback for when the run count is unavailable -- `job logs` needs a different
# credential type than `job status` on some tokens, hence not using it as primary.
entrypoint_produced_logs() {
    local logs
    logs="$(anyscale_retry anyscale job logs --cloud "$CLOUD" --name "$1" --head --max-lines 5 || true)"
    [[ -n "${logs//[[:space:]]/}" ]]
}

# True when the cluster came up far enough to execute the entrypoint. Guards against
# both misreading a fast entrypoint crash as a capacity failure and misreading a
# capacity failure's transient RUNNING as a real test failure.
entrypoint_ran() {
    local runs="$1" name="$2"
    if [[ "$runs" == "unknown" ]]; then
        entrypoint_produced_logs "$name"
    else
        [[ "$runs" -gt 0 ]]
    fi
}

# Wait for a job to settle, resuming when the CLI call rather than the job is what
# failed: `job wait` exits non-zero for a terminal state, its own timeout and a
# rate-limited lookup alike. Returns 0 if <target> (empty = any terminal state) is hit.
wait_for_state() {
    local name="$1" target="$2" timeout_s="$3"
    local deadline=$((SECONDS + timeout_s))
    local remaining state unreadable=0
    while true; do
        remaining=$((deadline - SECONDS))
        if ((remaining <= 0)); then
            return 1
        fi
        if [[ -n "$target" ]]; then
            anyscale job wait --cloud "$CLOUD" --name "$name" \
                --state "$target" --timeout "$remaining" && return 0
        else
            anyscale job wait --cloud "$CLOUD" --name "$name" --timeout "$remaining" && return 0
        fi
        state="$(job_state "$name")"
        if is_terminal "$state"; then
            if [[ -z "$target" ]]; then
                return 0
            fi
            return 1
        fi
        # The wait call broke (STARTING/RUNNING) or the control plane is unreachable
        # (UNKNOWN). Neither means the job is dead, so keep waiting -- but bound the
        # unreadable case so a down control plane can't hold us for the full timeout.
        if [[ "$state" == "UNKNOWN" ]]; then
            unreadable=$((unreadable + 1))
            if ((unreadable >= UNKNOWN_POLL_LIMIT)); then
                echo "Job ${name} state unreadable ${unreadable}x in a row; stopping the wait." >&2
                return 1
            fi
        else
            unreadable=0
        fi
        echo "Wait on ${name} returned early in state ${state}; resuming." >&2
        sleep 15
    done
}

# Drive a job to a terminal state and verify it got there. Required before a resubmit:
# a `job wait` timeout only stops the client polling, and `job terminate` can no-op
# under a rate limit.
ensure_terminal() {
    local name="$1" state i
    for ((i = 1; i <= TERMINATE_ATTEMPTS; i++)); do
        state="$(job_state "$name")"
        if is_terminal "$state"; then
            return 0
        fi
        anyscale_retry anyscale job terminate --cloud "$CLOUD" --name "$name" >/dev/null || true
        sleep 15
    done
    state="$(job_state "$name")"
    is_terminal "$state"
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    run_name="$JOB_NAME"
    if [[ "$attempt" -gt 1 ]]; then
        run_name="${JOB_NAME}-retry${attempt}"
    fi

    echo "--- Anyscale job attempt ${attempt}/${MAX_ATTEMPTS}: ${run_name}"
    # Recorded before submitting: a submit that times out client-side may still have
    # created the job, and an unrecorded job is one nobody will clean up.
    echo "$run_name" >> "$JOBS_FILE"
    CURRENT_RUN_NAME="$run_name"
    anyscale job submit -f "$CONFIG_FILE" --name "$run_name" --timeout "$RUN_TIMEOUT_S"

    if wait_for_state "$run_name" RUNNING "$START_TIMEOUT_S"; then
        # Provisional -- see the note at the top on RUNNING being reported for a
        # cluster that is actually tearing down after failing to provision.
        echo "Job ${run_name} reached RUNNING, waiting for it to finish."
        wait_for_state "$run_name" "" "$RUN_TIMEOUT_S" || true
    fi

    read -r state runs <<<"$(job_status "$run_name")"

    # A job can pass through RUNNING between two 10s `job wait` polls, so a missed
    # RUNNING with a terminal SUCCEEDED still means we got GPUs.
    if [[ "$state" == "SUCCEEDED" ]]; then
        echo "Job ${run_name} succeeded."
        exit 0
    fi

    if entrypoint_ran "$runs" "$run_name"; then
        echo "Job ${run_name} failed (state: ${state}) but the entrypoint ran -- real failure, not retrying." >&2
        exit 1
    fi

    echo "Job ${run_name} failed (state: ${state}) without ever running its entrypoint:" >&2
    echo "treating this as a GPU capacity failure." >&2

    if ensure_terminal "$run_name"; then
        CURRENT_RUN_NAME=""
    else
        echo "Could not confirm ${run_name} is terminated; refusing to resubmit and" >&2
        echo "leave a second cluster competing for the same instance type." >&2
        exit 1
    fi

    if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
        echo "Resubmitting in ${RETRY_DELAY_S}s ..." >&2
        sleep "$RETRY_DELAY_S"
    fi
done

echo "Gave up after ${MAX_ATTEMPTS} attempts without ever acquiring GPUs." >&2
exit 1
