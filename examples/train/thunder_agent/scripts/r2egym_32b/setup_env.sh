#!/usr/bin/env bash
set -euo pipefail

# Build and validate the Python environment for the ThunderAgent R2EGym 32B
# recipe. This intentionally does not use `uv sync --extra fsdp`: the recipe
# needs explicit torch/vLLM pins that match the cluster CUDA driver.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ACTION="${1:-sync}"

if [ -z "${RECIPE_HOME:-}" ]; then
  if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    RECIPE_HOME="/scratch/$USER/skyrl-thunder-agent"
  else
    RECIPE_HOME="$HOME/.cache/skyrl-thunder-agent"
  fi
fi

RECIPE_VENV="${RECIPE_VENV:-$RECIPE_HOME/venvs/skyrl-ta-pr-core-vllm0201-cu129}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$RECIPE_HOME/uv-cache/vllm0201-cu129}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_DEV="${INSTALL_DEV:-0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
VLLM_CU129_WHEEL_URL="${VLLM_CU129_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl}"

mkdir -p "$(dirname "$RECIPE_VENV")" "$UV_CACHE_DIR"

sync_env() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "Missing required command: uv" >&2
    exit 1
  fi

  cd "$REPO_ROOT"
  export UV_CACHE_DIR
  export FLASH_ATTENTION_SKIP_CUDA_BUILD="${FLASH_ATTENTION_SKIP_CUDA_BUILD:-TRUE}"

  uv venv --python "$PYTHON_VERSION" "$RECIPE_VENV"
  local python_bin="$RECIPE_VENV/bin/python"

  uv pip install \
    --python "$python_bin" \
    --torch-backend "$TORCH_BACKEND" \
    --compile-bytecode \
    "$VLLM_CU129_WHEEL_URL" \
    ray==2.51.1 \
    omegaconf hydra-core datasets tensorboard func_timeout accelerate \
    torchdata peft debugpy tensordict jaxtyping polars s3fs pybind11 \
    wandb cloudpathlib hf_transfer harbor mini-swe-agent litellm skyrl-gym

  uv pip install --python "$python_bin" \
    "thunderagent @ git+https://github.com/ThunderAgent-org/ThunderAgent.git"

  uv pip install --python "$python_bin" --no-deps --editable "$REPO_ROOT"

  if [ "$INSTALL_DEV" = "1" ]; then
    uv pip install --python "$python_bin" pytest pytest-asyncio pytest-forked
  fi
}

check_env() {
  local python_bin="$RECIPE_VENV/bin/python"
  if [ ! -x "$python_bin" ]; then
    echo "Recipe Python does not exist: $python_bin" >&2
    echo "Run: RECIPE_VENV='$RECIPE_VENV' bash '$SCRIPT_DIR/setup_env.sh' sync" >&2
    exit 1
  fi

  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - <<'PY'
import importlib
required = ["ray", "torch", "vllm", "fastapi", "uvicorn", "omegaconf", "harbor", "litellm"]
versions = {}
for name in required:
    mod = importlib.import_module(name)
    versions[name] = getattr(mod, "__version__", "ok")
    print(f"{name}={versions[name]}")

expected = {
    "torch": "2.11.0",
    "vllm": "0.20.1",
    "ray": "2.51.1",
}
for name, expected_prefix in expected.items():
    actual = versions[name]
    if not str(actual).startswith(expected_prefix):
        raise SystemExit(f"{name} version mismatch: expected {expected_prefix}*, got {actual}")

from examples.train.thunder_agent.main_harbor_thunder_agent import HarborThunderAgentFullyAsyncExp
from examples.train.thunder_agent.skyrl_integration.generator import ThunderAgentHarborGenerator
from examples.train.thunder_agent.skyrl_integration.remote_inference_client import ThunderAgentRemoteInferenceClient
from examples.train_integrations.harbor.dataset import HarborTaskDataset
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap

print("pr_core_thunder_agent_imports=ok")
PY

  if [ -x "$RECIPE_VENV/bin/ray" ]; then
    "$RECIPE_VENV/bin/ray" --version
  else
    echo "Missing Ray CLI: $RECIPE_VENV/bin/ray" >&2
    exit 1
  fi
}

case "$ACTION" in
  sync)
    sync_env
    check_env
    echo
    echo "Environment ready."
    echo "export PYTHON_BIN=\"$RECIPE_VENV/bin/python\""
    echo "export RAY_BIN=\"$RECIPE_VENV/bin/ray\""
    ;;
  check)
    check_env
    ;;
  print)
    echo "export PYTHON_BIN=\"$RECIPE_VENV/bin/python\""
    echo "export RAY_BIN=\"$RECIPE_VENV/bin/ray\""
    ;;
  *)
    echo "Usage: bash $0 [sync|check|print]" >&2
    exit 1
    ;;
esac
