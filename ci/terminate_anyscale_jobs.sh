#!/usr/bin/env bash
# Terminate the Anyscale jobs submit_anyscale_job.sh left behind.
#
# Cancelling a workflow run -- `cancel-in-progress`, the UI button, a job timeout --
# stops the runner, not the cluster. The submitting script traps INT/TERM and tries to
# clean up itself, but the runner only allows a short grace period before SIGKILL, so
# this runs as an `if: cancelled()` step where GitHub guarantees it gets to finish.
#
# Reads the names recorded by submit_anyscale_job.sh. No file means nothing was ever
# submitted, which is a normal outcome, not an error.
#
# Usage: ci/terminate_anyscale_jobs.sh [jobs-file]

set -uo pipefail

CLOUD="${ANYSCALE_CLOUD:-sky-anyscale-aws-us-east-1}"
JOBS_FILE="${1:-${ANYSCALE_JOBS_FILE:-${RUNNER_TEMP:-/tmp}/anyscale-submitted-jobs.txt}}"

if [[ ! -s "$JOBS_FILE" ]]; then
    echo "No Anyscale jobs recorded in ${JOBS_FILE}; nothing to clean up."
    exit 0
fi

failed=0
# fd 3, so the anyscale calls below can't swallow the list from stdin.
while read -r name <&3; do
    [[ -n "$name" ]] || continue

    state="$(anyscale job status --cloud "$CLOUD" --name "$name" --json 2>/dev/null \
        | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("state") or "UNKNOWN")
except Exception:
    print("UNKNOWN")' 2>/dev/null || echo UNKNOWN)"

    case "$state" in
        SUCCEEDED | FAILED | TERMINATED)
            echo "${name}: already ${state}."
            continue
            ;;
    esac

    # UNKNOWN included: a status call that fails under a rate limit is not evidence
    # the job is gone, and terminating an already-dead job is harmless.
    echo "${name}: state ${state}, terminating."
    for attempt in 1 2 3; do
        if anyscale job terminate --cloud "$CLOUD" --name "$name"; then
            continue 2
        fi
        sleep $((attempt * 5))
    done
    echo "${name}: could not terminate -- check it by hand." >&2
    failed=1
done 3< "$JOBS_FILE"

exit "$failed"
