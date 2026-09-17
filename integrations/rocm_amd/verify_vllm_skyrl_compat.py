#!/usr/bin/env python3
"""Ensure installed vLLM can satisfy SkyRL's inference server imports."""
import importlib
import sys

# Load the PyTorch ROCm runtime before vLLM and its optional compiler plugins.
import torch  # noqa: F401
import vllm


def _import_check(module_name: str, attrs: list[str]) -> str | None:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - surface the full import failure
        return f"{module_name}: {exc}"
    for attr in attrs:
        if not hasattr(mod, attr):
            return f"{module_name} missing attribute {attr}"
    return None


def main() -> int:
    print(f"vllm={vllm.__version__} path={vllm.__file__}")
    print(f"torch.hip={getattr(torch.version, 'hip', None)}")

    checks = [
        ("vllm.entrypoints.openai.api_server", ["build_app", "init_app_state", "create_server_socket"]),
        ("vllm.engine.async_llm_engine", ["AsyncLLMEngine"]),
        ("vllm.engine.arg_utils", ["AsyncEngineArgs"]),
    ]
    for module_name, attrs in checks:
        err = _import_check(module_name, attrs)
        if err:
            print(f"FAIL: {err}")
            return 1

    try:
        from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
        from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
            VLLMServerActor,
        )
        from skyrl.train.config import SkyRLTrainConfig

        build_vllm_cli_args(SkyRLTrainConfig())
        print(f"VLLMServerActor={VLLMServerActor.__name__}")
    except Exception as exc:  # noqa: BLE001 - report the complete compatibility failure
        print(f"FAIL: SkyRL vLLM server imports: {exc}")
        return 1

    print("PASS: vLLM SkyRL API compatibility")
    return 0


if __name__ == "__main__":
    sys.exit(main())
