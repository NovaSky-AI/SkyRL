#!/bin/bash

module purge
module load gcc/14.2.0
module load cuda/12.8

export UV_VENV="$SCRATCH/venvs/skyrl"
uv venv -p 3.12 "$UV_VENV"
source "$UV_VENV/bin/activate"

# Torch + GCC runtime + CUDA on loader path
export TORCH_LIB="$UV_VENV/lib/python3.12/site-packages/torch/lib"
GCC_ROOT="$(dirname "$(dirname "$(which gcc)")")"
export LD_LIBRARY_PATH="$GCC_ROOT/lib64:$GCC_ROOT/lib:$TORCH_LIB:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv pip install --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir \
  "torch==2.7.0"

cd $SCRATCH
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL
git checkout arm  # Warning: will download ~1GB of wheels
cd $SCRATCH

# Install wheels into uv env
uv pip install --no-index --find-links "$SCRATCH/SkyRL/wheels/aarch64-cu128-torch2.7/" --no-deps \
 vllm flash_attn "flashinfer-python==0.2.6.post1"

# Sanity check torch & vllm imports
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm ok")
PY

# We installed vllm without deps -- now install the deps
uv pip install -r $SCRATCH/SkyRL/wheels/aarch64-cu128-torch2.7/requirements-vllm.txt

cd SkyRL/skyrl-train

# Install gym and train packages
uv pip install -e ../skyrl-gym --no-deps 
uv pip install -e . --no-deps   
uv pip install -r $SCRATCH/SkyRL/wheels/aarch64-cu128-torch2.7/requirements-skyrl.txt

export VLLM_USE_V1=1

# Start ray
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
ray start --head || true


# Note: This constructs a GSM8k dataset, and you can optionally specify the output directory with: --output_dir
python examples/gsm8k/gsm8k_dataset.py


# IMPT: 
# 1) update the shell file with the correct number of available GPUs (and any other training configuration updates)
# 2) Update the shell file’s uv command to the following in order to avoid picking up x86 deps: uv run --active --no-project --no-sync -m skyrl_train.entrypoints.main_base \
bash examples/gsm8k/run_gsm8k.sh


