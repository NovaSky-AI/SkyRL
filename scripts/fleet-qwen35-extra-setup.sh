#!/usr/bin/env bash
# Qwen3.5-specific dependencies (sourced by fleet-common-setup.sh via --extra-setup)
#
# Installs: transformers 5.3.0, flash-attn 2.8.3 wheel, CUDA toolkit (nvcc), causal-conv1d
# Writes: $HOME/.cuda_env (sourced at run time for FlashInfer JIT)

# Upgrade transformers to 5.3.0 for Qwen3.5-MoE (model_type=qwen3_5_moe).
# - Qwen3.5 launched Feb 2026; all 4.x releases predate it.
# - 5.1.0 doesn't register qwen3_5_moe in AUTO_CONFIG_MAPPING.
# - 5.3.0 is the first stable release with full qwen3_5_moe support.
# - Do NOT install from git main (renamed layer_type_validation, breaks vLLM 0.17).
uv pip install -U "transformers==5.3.0"

# flash-attn 2.8.3 prebuilt wheel for torch 2.10 + CUDA 12 (training forward/backward)
uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

python -c "import torch; import torchvision; print(f'torch={torch.__version__}, torchvision={torchvision.__version__}')"

# --- CUDA toolkit for FlashInfer JIT (GatedDeltaNet kernels) ---
# pip CUDA packages are incomplete (missing nv/target headers); use NVIDIA apt repo instead
CUDA_HOME=""
for d in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
  if [ -x "$d/bin/nvcc" ]; then
    CUDA_HOME="$d"
    break
  fi
done
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
  NVCC_PATH=$(command -v nvcc)
  CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
fi
if [ -z "$CUDA_HOME" ]; then
  echo "nvcc not found on system. Installing CUDA toolkit from NVIDIA apt repo..."
  sudo apt-get update -qq
  UBUNTU_VER=$(lsb_release -rs 2>/dev/null | tr -d '.' || echo "2204")
  KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "Installing CUDA keyring from $KEYRING_URL"
  wget -qO /tmp/cuda-keyring.deb "$KEYRING_URL" 2>&1 || curl -sLo /tmp/cuda-keyring.deb "$KEYRING_URL"
  file /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends cuda-nvcc-12-8 libcublas-dev-12-8 cuda-nvrtc-dev-12-8
  CUDA_HOME="/usr/local/cuda-12.8"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

# Write cuda_env for run phase (fleet-common-run.sh sources this via --cuda-env)
echo "export CUDA_HOME=$CUDA_HOME" > "$HOME/.cuda_env"
echo "export PATH=$CUDA_HOME/bin:\$PATH" >> "$HOME/.cuda_env"

# causal-conv1d: required for GatedDeltaNet fast CUDA kernels in Qwen3.5-MoE.
# Without it, fla-core falls back to a naive PyTorch implementation that crashes
# with cudaErrorIllegalAddress on multi-node FSDP2 (Xid 31 MMU fault).
# Must be built from source (needs nvcc + g++) — install AFTER CUDA toolkit setup.
uv pip install "causal-conv1d>=1.6.0"
python -c "import causal_conv1d; print(f'causal-conv1d OK: {causal_conv1d.__version__}')"

# Verify pinned packages survived dependency resolution
python -c "import transformers; assert transformers.__version__ == '5.3.0', f'Expected 5.3.0 got {transformers.__version__}'"
# Ensure torch 2.10.0 — uv pip install can downgrade it during transitive resolution
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "torch version after setup: $TORCH_VER"
if [[ "$TORCH_VER" != 2.10.0* ]]; then
  echo "WARNING: torch was downgraded to $TORCH_VER, reinstalling 2.10.0+cu128"
  pip install --force-reinstall --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; assert torch.__version__.startswith('2.10.0'), f'Expected 2.10.0 got {torch.__version__}'"
python -c "import torch; import flash_attn_2_cuda; print('flash_attn CUDA extension OK')"
