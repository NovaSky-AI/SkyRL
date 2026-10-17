"""Tinker's train_attn / train_mlp must change which modules get an adapter on Megatron.

The assertions are structural: the exported adapter's tensors and its
``adapter_config.json`` must cover exactly the requested group, and vLLM must
serve the adapter. As in ``test_fsdp_lora_flags.py``, the expected module names
are written out here instead of being imported from
``skyrl.backends.skyrl_train.utils.lora_targets``, so a wrong group table cannot
pass by being compared with itself.

All tenants on a Megatron server are cloned from the first one's template, so
each flag combination gets its own server. Needs 2 GPUs: one for the Megatron
policy worker, one for the vLLM engine that the adapter export syncs into. Run:
  uv run --isolated --extra tinker --extra megatron --with pytest \\
    pytest -s tests/tinker/skyrl_train/test_megatron_lora_flags.py
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
    reason="LoRA flag tests need 2 GPUs: a Megatron policy worker plus the vLLM engine the adapter syncs into",
)

tinker = pytest.importorskip("tinker")
from safetensors.torch import load_file  # noqa: E402
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

DENSE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MOE_MODEL = "eatang/qwen3-moe-tiny-random"
TINKER_API_KEY = "tml-dummy"
LORA_RANK = 8

# HF leaf names of each group in a Qwen3 decoder, which is what the export uses.
ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}


@contextmanager
def _api_server(port: int, lora_sync_path: str, base_model: str = DENSE_MODEL):
    backend_config = {
        "strategy": "megatron",
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.colocate_all": False,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
        # Adapter sync, so vLLM serves each tenant's adapter by name and the
        # PEFT files this test reads are written.
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "trainer.policy.model.lora.max_loras": 2,
        "trainer.policy.model.lora.max_cpu_loras": 2,
        "trainer.policy.model.lora.lora_sync_path": lora_sync_path,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        cmd = [
            "uv", "run", "--isolated",
            "--extra", "tinker",
            "--extra", "megatron",
            "-m", "skyrl.tinker.api",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--base-model", base_model,
            "--backend", "megatron",
            "--backend-config", json.dumps(backend_config),
            "--database-url", f"sqlite:///{db_path}",
        ]  # fmt: skip
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                ok = wait_for_condition(lambda: _server_is_up(port), timeout_sec=300, poll_interval_sec=2)
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


def _find_export(lora_sync_path: str, filename: str, model_id: str | None = None) -> str:
    """Locate an exported adapter file; the worker writes one subdirectory per model id."""
    matches = [
        os.path.join(root, filename)
        for root, _dirs, files in os.walk(lora_sync_path)
        if filename in files and (model_id is None or model_id in root.split(os.sep))
    ]
    assert matches, f"no {filename} exported anywhere under {lora_sync_path}"
    assert len(matches) == 1, f"expected one {filename}, found {matches}"
    return matches[0]


def _adapter_keys(lora_sync_path: str, model_id: str | None = None) -> list[str]:
    return list(load_file(_find_export(lora_sync_path, "adapter_model.safetensors", model_id)))


def _leaf_names(keys: list[str]) -> set[str]:
    """Leaf module names that received a LoRA adapter, read off the exported tensor names."""
    leaves = set()
    for key in keys:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part.startswith("lora_") and i > 0:
                leaves.add(parts[i - 1])
                break
    return leaves


def _service_client(port: int):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{port}/", api_key=TINKER_API_KEY)


def _create(port: int, base_model: str, *, train_attn: bool, train_mlp: bool):
    return _service_client(port).create_lora_training_client(
        base_model=base_model, rank=LORA_RANK, train_attn=train_attn, train_mlp=train_mlp, train_unembed=False
    )


def _train_step(client) -> None:
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()


def _sample_tokens(client, sampler) -> list[int]:
    prompt_ids = client.get_tokenizer().encode("Question: 1+1?\nAnswer:", add_special_tokens=True)
    out = sampler.sample(
        prompt=tinker_types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker_types.SamplingParams(max_tokens=8, temperature=0.0, top_k=1, seed=0),
    ).result()
    return list(out.sequences[0].tokens)


def test_attention_only_adapter_and_its_tenants():
    """An attention-only adapter exports only attention modules and is served by vLLM.

    A second tenant with the same flags shares the runtime; one asking for the MLP
    group is rejected, because every tenant is a copy of the first one's template.
    """
    with tempfile.TemporaryDirectory() as sync_dir:
        with _api_server(8041, sync_dir):
            first = _create(8041, DENSE_MODEL, train_attn=True, train_mlp=False)
            _train_step(first)
            sampler = first.save_weights_and_get_sampling_client(name="attn_only")
            tokens = _sample_tokens(first, sampler)
            leaves = _leaf_names(_adapter_keys(sync_dir, first.model_id))
            with open(_find_export(sync_dir, "adapter_config.json", first.model_id)) as f:
                adapter_config = json.load(f)

            second = _create(8041, DENSE_MODEL, train_attn=True, train_mlp=False)
            _train_step(second)

            with pytest.raises(Exception) as exc:
                _create(8041, DENSE_MODEL, train_attn=False, train_mlp=True)

    assert tokens, "vLLM returned no tokens from the attention-only adapter"
    assert leaves == ATTN_LEAVES, f"adapted {sorted(leaves)}, expected {sorted(ATTN_LEAVES)}"
    assert set(adapter_config["target_modules"]) == ATTN_LEAVES
    assert adapter_config["r"] == LORA_RANK
    assert "signature mismatch" in str(exc.value).lower()


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "expected", "port"),
    [
        pytest.param(False, True, MLP_LEAVES, 8042, id="mlp_only"),
        pytest.param(True, True, ATTN_LEAVES | MLP_LEAVES, 8043, id="both"),
    ],
)
def test_flags_select_which_modules_are_adapted(train_attn, train_mlp, expected, port):
    """Each remaining flag combination exports exactly the expected modules, and vLLM samples from it."""
    with tempfile.TemporaryDirectory() as sync_dir:
        with _api_server(port, sync_dir):
            client = _create(port, DENSE_MODEL, train_attn=train_attn, train_mlp=train_mlp)
            _train_step(client)
            sampler = client.save_weights_and_get_sampling_client(name=f"lora_flags_{train_attn}_{train_mlp}")
            tokens = _sample_tokens(client, sampler)
            leaves = _leaf_names(_adapter_keys(sync_dir, client.model_id))

    assert tokens, "vLLM returned no tokens from the adapter"
    assert leaves == expected, f"adapted {sorted(leaves)}, expected {sorted(expected)}"


def test_mlp_only_adapter_covers_the_moe_experts():
    """On an MoE model the MLP group includes the routed experts, which FSDP cannot adapt."""
    with tempfile.TemporaryDirectory() as sync_dir:
        with _api_server(8044, sync_dir, base_model=MOE_MODEL):
            client = _create(8044, MOE_MODEL, train_attn=False, train_mlp=True)
            _train_step(client)
            sampler = client.save_weights_and_get_sampling_client(name="moe_mlp_only")
            tokens = _sample_tokens(client, sampler)
            keys = _adapter_keys(sync_dir, client.model_id)

    assert tokens, "vLLM returned no tokens from the MoE adapter"
    assert keys, "the MoE adapter exported no tensors"
    outside_mlp = [key for key in keys if ".mlp." not in key]
    assert not outside_mlp, f"MLP-only adapter has tensors outside the MLP: {outside_mlp[:3]}"
    assert any(".mlp.experts." in key for key in keys), f"no expert tensors among {keys[:5]}"
