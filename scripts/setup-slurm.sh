#!/bin/bash
# One-time setup for RunPod Slurm cluster access via SkyPilot.
# Requires: ~/.ssh/runpod_key (ask team for the key)
set -euo pipefail

if [ ! -f ~/.ssh/runpod_key ]; then
  echo "ERROR: ~/.ssh/runpod_key not found. Ask the team for the RunPod SSH key."
  exit 1
fi

mkdir -p ~/.slurm
cat > ~/.slurm/config <<'EOF'
Host runpod-cluster
    HostName 31.24.80.22
    Port 16198
    User root
    IdentityFile ~/.ssh/runpod_key
    StrictHostKeyChecking no
EOF

echo "Verifying SSH connectivity..."
ssh -F ~/.slurm/config runpod-cluster "sinfo -N" || { echo "FAIL: Cannot connect to Slurm controller"; exit 1; }

echo "Verifying SkyPilot detection..."
sky check slurm 2>&1 | tail -5

echo "Done. Run 'sky show-gpus --cloud slurm' to see available GPUs."
