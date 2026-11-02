#!/usr/bin/env bash
# Submit a focused vLLM-upgrade GPU job to the SkyRL CI cloud (NovaSky org).
#
# Usage, from the repo root:
#   bash .agents/skills/skyrl-vllm-upgrade/scripts/submit_focused.sh <fsdp|megatron> <tag> [group...]
#
# <tag> makes the job name unique (query jobs by the printed ID, not by name).
# Groups default to the spec's entrypoint; pass them to run a subset or a rerun.
# HF_TOKEN is taken from the environment, else from .env.ray, and passed with --env
# so it is never written to a file.
set -euo pipefail

flavor=$1
tag=$2
shift 2
dir=.agents/skills/skyrl-vllm-upgrade/scripts
spec=$dir/anyscale_focused_$flavor.yaml

# In a git worktree, .env.ray (untracked) lives in the main checkout.
main_root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
for env_file in .env.ray "$main_root/.env.ray"; do
    if [[ -z "${HF_TOKEN:-}" && -f "$env_file" ]]; then
        HF_TOKEN=$(grep '^HF_TOKEN=' "$env_file" | cut -d= -f2-)
    fi
done
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set" >&2; exit 1; }

cmd=()
if [[ $# -gt 0 ]]; then
    cmd=(-- bash "$dir/run_focused_gpu.sh" "$@")
fi

# Large untracked local artifacts must not be uploaded with working_dir. The CLI echoes
# the job's env vars, so its output is redacted.
anyscale job submit -f "$spec" --name "vllm-upgrade-$flavor-$tag" --timeout 21000 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -e '*.whl' -e 'exp_logs/' -e 'agents_artifacts/' -e 'literature_review/' \
    ${cmd[@]+"${cmd[@]}"} 2>&1 | sed -E "s/'HF_TOKEN': '[^']*'/'HF_TOKEN': '<redacted>'/"
