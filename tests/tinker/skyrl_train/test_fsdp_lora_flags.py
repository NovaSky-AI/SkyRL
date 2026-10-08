"""Tinker's train_attn / train_mlp must change which modules get an adapter.

The assertion is structural, not statistical: with ``train_mlp=False`` the MLP
LoRA tensors are absent from the exported adapter, so comparing key sets proves
the flags reached PEFT exactly. Mirrors the acceptance criteria of the
tinker-cookbook's own ``tests/weights/test_partial_lora_lifecycle.py``.

The marker sets below are written out rather than imported from
``skyrl.backends.skyrl_train.utils.lora_targets`` on purpose: a test that
classifies modules with the same helper the implementation uses would pass even
if that helper were wrong.

FSDP serves exactly one adapter per server lifetime (multi-tenant LoRA is
Megatron-only), so each case gets its own server. Needs 2 GPUs: one for the FSDP
policy worker, one for the vLLM engine that the adapter export syncs into. Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest \\
    pytest -s tests/tinker/skyrl_train/test_fsdp_lora_flags.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager

import pytest

cuda_device_count = 0
try:  # pragma: no cover - import guard
    import torch

    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    cuda_device_count = 0

pytestmark = pytest.mark.skipif(
    cuda_device_count < 2,
    reason="LoRA flag tests need 2 GPUs: an FSDP policy worker plus the vLLM engine the adapter syncs into",
)

tinker = pytest.importorskip("tinker")
from safetensors.torch import load_file  # noqa: E402
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
LORA_RANK = 8

# Leaf module names for each group in a Qwen3-style decoder. Deliberately
# duplicated from the implementation -- see the module docstring.
ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}


@contextmanager
def _api_server(port: int, lora_sync_path: str):
    backend_config = {
        "strategy": "fsdp",
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.colocate_all": False,
        "trainer.policy.model.lora.lora_sync_path": lora_sync_path,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        cmd = [
            "uv", "run", "--isolated",
            "--extra", "tinker",
            "--extra", "fsdp",
            "-m", "skyrl.tinker.api",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--base-model", BASE_MODEL,
            "--backend", "fsdp",
            "--backend-config", json.dumps(backend_config),
            "--database-url", f"sqlite:///{db_path}",
        ]  # fmt: skip
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                ok = wait_for_condition(lambda: _server_is_up(port), timeout_sec=180, poll_interval_sec=2)
                if not ok:
                    with open(log_path) as f:
                        print(f"=== Server failed to start ===\n{f.read()}")
                    pytest.fail("Tinker API server did not come up in time")
                yield
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _server_is_up(port: int) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"http://0.0.0.0:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


def _make_datum(tokenizer, prompt: str, completion: str):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


def _adapter_leaf_names(lora_sync_path: str) -> set[str]:
    """Leaf module names that received a LoRA adapter, read off the export."""
    weights_file = os.path.join(lora_sync_path, "adapter_model.safetensors")
    assert os.path.exists(weights_file), f"no adapter exported to {lora_sync_path}"
    leaves = set()
    for key in load_file(weights_file):
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part.startswith("lora_") and i > 0:
                leaves.add(parts[i - 1])
                break
    return leaves


def _train_one_step_and_export(port: int, *, train_attn: bool, train_mlp: bool) -> None:
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{port}/", api_key=TINKER_API_KEY)
    client = sc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=False,
    )
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()
    # Weight sync is what writes the PEFT adapter files we then inspect.
    client.save_weights_and_get_sampling_client(name=f"lora_flags_{train_attn}_{train_mlp}")


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "expected", "port"),
    [
        pytest.param(True, True, ATTN_LEAVES | MLP_LEAVES, 8031, id="both"),
        pytest.param(True, False, ATTN_LEAVES, 8032, id="attn_only"),
        pytest.param(False, True, MLP_LEAVES, 8033, id="mlp_only"),
    ],
)
def test_flags_select_which_modules_are_adapted(train_attn, train_mlp, expected, port):
    with tempfile.TemporaryDirectory() as sync_dir:
        with _api_server(port, sync_dir):
            _train_one_step_and_export(port, train_attn=train_attn, train_mlp=train_mlp)
            leaves = _adapter_leaf_names(sync_dir)

    assert leaves == expected, f"adapted {sorted(leaves)}, expected {sorted(expected)}"
    # lm_head is never adapted: PEFT's "all-linear" excludes the output layer,
    # which is why train_unembed is rejected rather than honoured here.
    assert "lm_head" not in leaves


def test_adapter_config_records_the_narrowed_targets():
    """adapter_config.json is what vLLM reads back, so it must agree too."""
    with tempfile.TemporaryDirectory() as sync_dir:
        with _api_server(8034, sync_dir):
            _train_one_step_and_export(8034, train_attn=True, train_mlp=False)
            with open(os.path.join(sync_dir, "adapter_config.json")) as f:
                config = json.load(f)

    assert config["r"] == LORA_RANK
    assert config["peft_type"] == "LORA"
    assert set(config["target_modules"]) == ATTN_LEAVES
