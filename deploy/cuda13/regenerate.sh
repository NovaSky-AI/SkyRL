#!/usr/bin/env bash
# Regenerate requirements-megatron.txt from the CUDA-13 resolution of the
# root project (run on a branch whose root pyproject/uv.lock are the CUDA-13
# stack). Run from the repository root.
set -euo pipefail

REQ=deploy/cuda13/requirements-megatron.txt

uv export --extra megatron --no-hashes --no-emit-project --emit-index-url -o "$REQ"

python3 - "$REQ" <<'EOF'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines(True)
out, skip_via = [], False
for line in lines:
    # Only the flashinfer cu130 index is used by these pins; other emitted
    # extra indexes (cu128/cu129/...) would reintroduce ambiguity.
    if line.startswith("--extra-index-url") and "flashinfer.ai/whl/cu130" not in line:
        continue
    # The editable gym path is cwd-sensitive; install.sh installs it (and
    # skyrl itself) explicitly with absolute paths instead.
    if line.startswith("-e ./skyrl-gym"):
        skip_via = True
        continue
    if skip_via and line.strip().startswith("# via"):
        skip_via = False
        continue
    skip_via = False
    out.append(line)
open(path, "w", encoding="utf-8").writelines(out)
EOF

echo "regenerated $REQ"
