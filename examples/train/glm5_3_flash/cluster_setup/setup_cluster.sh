#!/bin/bash
# Prepare a freshly started Ray cluster (Anyscale workspace) to run SkyRL's Megatron backend:
# populate a shared uv cache, copy it to every node's local disk, then check EFA and measure
# inter-node NCCL bandwidth. Idempotent; rerun after every workspace restart (local disks are
# wiped, the shared cache is not).
#
# Run on the head node from the SkyRL repo (any directory inside it works):
#   bash examples/train/glm5_3_flash/cluster_setup/setup_cluster.sh
#
# Settings (environment):
#   SHARED_UV_CACHE   uv cache on storage every node mounts   (default: /shared/hackskyrl/uv_cache)
#   UV_EXTRAS         extras to sync                           (default: "megatron dev")
#   EXPECTED_GPUS     GPUs to wait for in `ray status`         (default: 64)
#   SKIP_PROBE=1      skip the NCCL bandwidth probe
# Needs `python3` with `ray` importable on the head node (true for Anyscale Ray images), uv, rsync.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
SHARED_UV_CACHE="${SHARED_UV_CACHE:-/shared/hackskyrl/uv_cache}"
UV_EXTRAS="${UV_EXTRAS:-megatron dev}"
EXPECTED_GPUS="${EXPECTED_GPUS:-64}"
LOG_DIR="${LOG_DIR:-$REPO/logs/cluster_setup}"
mkdir -p "$LOG_DIR"
step() { echo; echo "=== $(date +%T) $*"; }

step "1/5 uv sync into the shared cache ($SHARED_UV_CACHE)"
# The shared cache persists across restarts; uv must copy (not hardlink) into it from local disk.
# Offline first: re-resolving from the network is slow and GitHub release downloads can fail.
extras=(); for e in $UV_EXTRAS; do extras+=(--extra "$e"); done
mkdir -p "$SHARED_UV_CACHE"
(cd "$REPO" && UV_CACHE_DIR="$SHARED_UV_CACHE" UV_LINK_MODE=copy UV_OFFLINE=1 uv sync "${extras[@]}") \
  > "$LOG_DIR/uv_sync.log" 2>&1 \
  || (cd "$REPO" && UV_CACHE_DIR="$SHARED_UV_CACHE" UV_LINK_MODE=copy uv sync "${extras[@]}") \
  >> "$LOG_DIR/uv_sync.log" 2>&1
tail -1 "$LOG_DIR/uv_sync.log"

step "2/5 wait for $EXPECTED_GPUS GPUs in the Ray cluster"
for _ in $(seq 1 120); do
  gpus=$(ray status 2>/dev/null | grep -oE "/[0-9.]+ GPU" | head -1 | tr -dc '0-9.' || true)
  [ "${gpus%.*}" = "$EXPECTED_GPUS" ] && break
  echo "  GPUs: ${gpus:-0}/$EXPECTED_GPUS, waiting"; sleep 15
done
[ "${gpus%.*}" = "$EXPECTED_GPUS" ] || { echo "cluster never reached $EXPECTED_GPUS GPUs"; exit 1; }

step "3/5 copy the uv cache to every node's local disk (GPU workers in parallel, then the head)"
# Every `uv run --isolated` (each Ray worker and the job driver on the head) builds an env from
# the cache; from network storage that is a ~60 GB copy per process.
python3 "$HERE/seed_uv_cache.py" "$SHARED_UV_CACHE" "$HOME/.cache/uv"

step "4/5 EFA in a worker pod"
python3 - <<'PY'
import subprocess, ray
ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
@ray.remote(num_gpus=1)
def check():
    cmd = ("ls /dev/infiniband 2>/dev/null | tr '\n' ' '; echo; "
           "/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM 2>&1 | grep -c 'domain:'")
    return ray.util.get_node_ip_address(), subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.split("\n")
ip, (devs, domains, *_) = ray.get(check.remote())
print(f"  {ip}: devices [{devs.strip()}], efa RDM domains: {domains.strip()}")
if not devs.strip():
    print("  EFA NOT MOUNTED: add vpc.amazonaws.com/efa to the worker group (see efa_compute_config.yaml)")
PY

if [ "${SKIP_PROBE:-0}" = "1" ]; then exit 0; fi
step "5/5 NCCL all-to-all across nodes (1 GPU on each of 2 nodes, then 8 on each of 2)"
cd "$REPO"
for cfg in "1 2" "8 2"; do
  log="$LOG_DIR/nccl_probe_${cfg// /x}.log"
  # Local cache from step 3 (UV_CACHE_DIR unset), so the env builds in seconds.
  UV_OFFLINE=1 timeout 900 uv run --isolated --extra megatron python "$HERE/nccl_a2a_probe.py" $cfg 512 \
    > "$log" 2>&1 || echo "  probe failed or timed out, see $log"
  echo "  ${cfg// /x}: $(grep -o 'RESULT.*' "$log" | tail -1)"
  echo "    transport: $(grep -ohE 'NET/(OFI|IB|Socket)[^|]{0,60}' "$log" | sort | uniq -c | sort -rn | head -2 | tr -s ' ' | tr '\n' ';')"
done
echo
echo "EFA works when the transport shows NET/OFI with provider efa and bandwidth is tens of GB/s"
echo "per GPU. NET/Socket at ~1-2 GB/s per GPU means NCCL fell back to TCP."
