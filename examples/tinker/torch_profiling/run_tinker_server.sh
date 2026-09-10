#!/usr/bin/env bash

set -euo pipefail

# Tinker API server with torch profiling enabled.
#
# --torch-profiler turns on /start_profiling, /stop_profiling and
# /profiling_status. Without it those endpoints return 404. export_dir may also
# be a cloud URI (s3://, gs://, gcs://), in which case each closed window is
# uploaded and the local copy dropped.
#
# FSDP needs cpu_offload=true whenever the policy is under offload management,
# which includes colocate_all=false since colocate_policy_ref defaults to true.
# The default manual offload path moves parameters with torch.utils.swap_tensors,
# which fails while the profiler holds references to them.

DEFAULT_BACKEND_CONFIG='{"trainer.placement.colocate_all": false, "trainer.placement.policy_num_gpus_per_node": 1, "trainer.policy.fsdp_config.cpu_offload": true, "trainer.policy.model.lora.max_loras": 4}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

EXPORT_DIR="${EXPORT_DIR:-gs://sumanth-test/skyrl_traces_test/}"
TORCH_PROFILER="{\"export_dir\": \"${EXPORT_DIR}\", \"ranks\": [0]}"

export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

uv run --extra tinker --extra fsdp --with ray==2.56 -m skyrl.tinker.api \
  --base-model "Qwen/Qwen3-0.6B" \
  --backend fsdp \
  --port 8000 \
  --backend-config "$BACKEND_CONFIG" \
  --torch-profiler "$TORCH_PROFILER" \
  "$@"
