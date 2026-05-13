#!/bin/bash
# One-time setup for RunPod Slurm cluster access via SkyPilot.
# Requires: ~/.ssh/runpod_key (ask team for the key)
#
# This script handles EVERYTHING needed for Slurm training:
#   1. Creates ~/.slurm/config (SSH connection to Slurm controller)
#   2. Ensures 'slurm' is in allowed_clouds in ~/.sky/config.yaml
#   3. Adds slurm.cluster_configs (workdir, gpu_partition_map) to ~/.sky/config.yaml
#   4. Restarts the SkyPilot API server to pick up the new config
#   5. Verifies connectivity and GPU detection
#
# The SkyPilot API server ignores allowed_clouds and cluster_configs from
# the project-level .sky.yaml, so these MUST be in ~/.sky/config.yaml.
set -euo pipefail

if [ ! -f ~/.ssh/runpod_key ]; then
  echo "ERROR: ~/.ssh/runpod_key not found. Ask the team for the RunPod SSH key."
  exit 1
fi

# 1. SSH config for Slurm controller
echo "[1/5] Setting up ~/.slurm/config..."
mkdir -p ~/.slurm
cat > ~/.slurm/config <<'EOF'
Host runpod-cluster
    HostName 31.24.80.22
    Port 16198
    User root
    IdentityFile ~/.ssh/runpod_key
    StrictHostKeyChecking no
EOF

# 2. Ensure slurm is in allowed_clouds
echo "[2/5] Checking allowed_clouds..."
if [ ! -f ~/.sky/config.yaml ]; then
  mkdir -p ~/.sky
  cat > ~/.sky/config.yaml <<'EOF'
allowed_clouds:
  - slurm
  - gcp
  - runpod
EOF
  echo "  Created ~/.sky/config.yaml with slurm enabled"
elif ! grep -q '^\s*- slurm' ~/.sky/config.yaml; then
  # Add slurm to allowed_clouds list
  sed -i.bak '/allowed_clouds:/a\  - slurm' ~/.sky/config.yaml && rm -f ~/.sky/config.yaml.bak
  echo "  Added slurm to allowed_clouds"
else
  echo "  slurm already in allowed_clouds"
fi

# 3. Add slurm.cluster_configs
echo "[3/5] Adding slurm.cluster_configs..."
if grep -q 'cluster_configs:' ~/.sky/config.yaml 2>/dev/null && grep -q 'runpod-cluster:' ~/.sky/config.yaml 2>/dev/null; then
  echo "  cluster_configs.runpod-cluster already exists — skipping"
elif grep -q '^slurm:' ~/.sky/config.yaml 2>/dev/null; then
  # slurm section exists but no cluster_configs — append under it
  sed -i.bak '/^slurm:/a\  cluster_configs:\n    runpod-cluster:\n      workdir: /workspace\n      gpu_partition_map:\n        H200: gpu' ~/.sky/config.yaml && rm -f ~/.sky/config.yaml.bak
  echo "  Added cluster_configs under existing slurm section"
else
  cat >> ~/.sky/config.yaml <<'EOF'

slurm:
  cluster_configs:
    runpod-cluster:
      workdir: /workspace
      gpu_partition_map:
        H200: gpu
EOF
  echo "  Added slurm.cluster_configs"
fi

# 4. Restart API server to pick up config
echo "[4/5] Restarting SkyPilot API server..."
sky api stop 2>/dev/null || true
sky api start 2>&1 | tail -3

# 5. Verify
echo "[5/5] Verifying..."
echo "  SSH connectivity..."
ssh -F ~/.slurm/config runpod-cluster "sinfo -N" || { echo "FAIL: Cannot connect to Slurm controller"; exit 1; }
echo "  SkyPilot detection..."
sky check slurm 2>&1 | grep -E "Slurm:|enabled|disabled"
echo ""
echo "Done. Run 'sky show-gpus --cloud slurm' to see available GPUs."
