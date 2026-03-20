#!/bin/bash
set -euo pipefail

# ==============================================================================
# Phase 1-2: Setup & Sanity Check for H200 clusters
# Run this on EVERY node before training.
# ==============================================================================

echo "============================================"
echo " Phase 1: Hardware & Environment Verification"
echo "============================================"

# GPU info
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Total GPUs: $NUM_GPUS"

# CUDA
echo -e "\n=== CUDA ==="
nvcc --version 2>/dev/null || echo "nvcc not found (check CUDA toolkit)"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

# NCCL / network
echo -e "\n=== Network ==="
echo "Hostname: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"
ibstat 2>/dev/null | grep -E "State|Rate" | head -8 || echo "No InfiniBand (using TCP)"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-not set}"

# NVLink topology
echo -e "\n=== GPU Topology ==="
nvidia-smi topo -m 2>/dev/null | head -15

echo ""
echo "============================================"
echo " Phase 1b: Install SkyRL"
echo "============================================"

# Clone if needed
if [ ! -d "$HOME/SkyRL" ]; then
    echo "Cloning SkyRL..."
    git clone https://github.com/NovaSky-AI/SkyRL.git "$HOME/SkyRL"
fi

cd "$HOME/SkyRL"
git fetch origin main
git checkout main
git pull origin main

# Check Python
echo -e "\n=== Python ==="
which python3
python3 --version
which uv || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)

# Create venv and install
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.12 venv..."
    uv venv --python 3.12 .venv
fi

echo "Installing dependencies (this may take several minutes for transformer-engine/flash-attn)..."
uv sync --extra megatron 2>&1 | tail -5

# Install transformers 5.x for GLM-4.7-Flash
echo "Installing transformers 5.x..."
uv pip install --python .venv/bin/python "transformers>=5.0.0" 2>&1 | tail -3

# Install test deps
uv pip install --python .venv/bin/python pytest pytest-timeout 2>&1 | tail -2

echo ""
echo "============================================"
echo " Phase 1c: Verify Key Imports"
echo "============================================"

.venv/bin/python -c "
import torch
print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
import transformers; print(f'transformers {transformers.__version__}')
import vllm; print(f'vllm {vllm.__version__}')
import megatron.core; print(f'megatron-core {megatron.core.__version__}')
from megatron.bridge import AutoBridge; print('AutoBridge OK')

# Verify GLM-4.7-Flash config loads
from transformers import AutoConfig
c = AutoConfig.from_pretrained('zai-org/GLM-4.7-Flash', trust_remote_code=True)
print(f'GLM-4.7-Flash: model_type={c.model_type}, experts={c.n_routed_experts}, topk={c.num_experts_per_tok}, layers={c.num_hidden_layers}')
print()
print('All imports OK!')
"

# Prepare GSM8K data
echo ""
echo "=== Preparing GSM8K data ==="
mkdir -p "$HOME/data/gsm8k"
if [ ! -f "$HOME/data/gsm8k/train.parquet" ]; then
    .venv/bin/python examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
else
    echo "GSM8K data already exists"
fi

echo ""
echo "============================================"
echo " Phase 2: Single-Node Functional Test"
echo "============================================"

# Start ray on this node
.venv/bin/ray stop --force 2>/dev/null || true
sleep 2
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_memory_usage_threshold=1.0
.venv/bin/ray start --head 2>&1 | tail -2

echo "Running Megatron checkpoint save/load test..."
.venv/bin/python -m pytest tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py \
    -k "megatron and not lora and not fully_reshardable" \
    -v --timeout=600 -x 2>&1 | tail -20

echo ""
echo "Running Megatron fully_reshardable checkpoint test..."
.venv/bin/python -m pytest tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py \
    -k "megatron_fully_reshardable" \
    -v --timeout=600 -x 2>&1 | tail -20

.venv/bin/ray stop --force 2>/dev/null

echo ""
echo "============================================"
echo " SETUP COMPLETE"
echo "============================================"
echo "GPUs: $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "Next: run the training script for your cluster size"
