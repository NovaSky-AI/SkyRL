#!/usr/bin/env bash
# Run ROCm compatibility smoke tests (GPU node or container required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPORT_DIR="${REPORT_DIR:-/tmp/skyrl-rocm-smoke}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPORT_DIR}/smoke_${STAMP}.json"

SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

python integrations/rocm_amd/smoke_test.py --all --model "$SMOKE_MODEL" --report "$REPORT"
EXIT=$?

if [[ $EXIT -eq 0 ]]; then
  echo "All checks passed. See $REPORT"
else
  echo "Some checks failed. See $REPORT"
fi

exit $EXIT
