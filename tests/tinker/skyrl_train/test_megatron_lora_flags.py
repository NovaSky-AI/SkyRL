"""Tinker's train_attn / train_mlp must change which modules get an adapter on Megatron.

The adapter is read back the way a user exports their LoRA: save sampler weights,
download the checkpoint archive through the REST API, and read its
``adapter_config.json``. Its ``target_modules`` lists the modules that got an adapter,
so comparing them to the expected group proves the flags reached Megatron-Bridge
exactly. As in ``test_fsdp_lora_flags.py``, the expected names are written out rather
than imported from ``skyrl.backends.skyrl_train.utils.lora_targets``, so a wrong group
table cannot pass by being compared with itself.

All tenants on a Megatron server are cloned from the first one's template, so each
flag combination gets its own server. Needs 2 GPUs: one for the Megatron policy
worker, one for the vLLM engine the sampler weights sync into. Run:
  uv run --isolated --extra tinker --extra megatron --with pytest \\
    pytest -s tests/tinker/skyrl_train/test_megatron_lora_flags.py
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlparse

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
from safetensors import safe_open  # noqa: E402
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import start_api_server  # noqa: E402

DENSE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MOE_MODEL = "eatang/qwen3-moe-tiny-random"
TINKER_API_KEY = "tml-dummy"
LORA_RANK = 8

# HF leaf names of each group in a Qwen3 decoder, which is what the export uses.
# Deliberately duplicated from the implementation -- see the module docstring.
ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}

BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    # Adapter sync: vLLM serves each tenant's adapter by name, and the sampler
    # checkpoint archive is that PEFT adapter.
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.model.lora.max_loras": 2,
    "trainer.policy.model.lora.max_cpu_loras": 2,
}


def _megatron_server(port: int, base_model: str = DENSE_MODEL):
    return start_api_server(
        port=port,
        backend="megatron",
        base_model=base_model,
        backend_config=BACKEND_CONFIG,
        extras=("tinker", "megatron"),
        wait_for_up=True,
        teardown_timeout=15.0,
    )


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


def _service_client(port: int):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{port}/", api_key=TINKER_API_KEY)


def _create(service_client, base_model: str, *, train_attn: bool, train_mlp: bool):
    return service_client.create_lora_training_client(
        base_model=base_model, rank=LORA_RANK, train_attn=train_attn, train_mlp=train_mlp, train_unembed=False
    )


def _train_step(client) -> None:
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()


def _save_and_export(service_client, training_client, name: str) -> tuple[str, dict, list[str]]:
    """Save sampler weights and download them as a user exports a LoRA.

    Returns:
        The checkpoint's tinker path, the archive's ``adapter_config.json``, and the
        tensor names in its ``adapter_model.safetensors``.
    """
    path = training_client.save_weights_for_sampler(name=name).result().path
    parsed = urlparse(path)
    run_id, checkpoint_id = parsed.netloc, parsed.path.lstrip("/")
    url = service_client.create_rest_client().get_checkpoint_archive_url(run_id, checkpoint_id).result().url
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive = os.path.join(tmp_dir, "checkpoint.tar")
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp_dir, filter="data")
        with open(os.path.join(tmp_dir, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        with safe_open(os.path.join(tmp_dir, "adapter_model.safetensors"), framework="pt") as weights:
            tensor_names = list(weights.keys())
    return path, adapter_config, tensor_names


def _adapted_leaves(adapter_config: dict) -> set[str]:
    return {name.split(".")[-1] for name in adapter_config["target_modules"]}


def _sample_tokens(training_client, sampler) -> list[int]:
    prompt_ids = training_client.get_tokenizer().encode("Question: 1+1?\nAnswer:", add_special_tokens=True)
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
    with _megatron_server(8041):
        service_client = _service_client(8041)
        first = _create(service_client, DENSE_MODEL, train_attn=True, train_mlp=False)
        _train_step(first)
        path, adapter_config, _ = _save_and_export(service_client, first, name="attn_only")
        tokens = _sample_tokens(first, first.create_sampling_client(path))

        second = _create(service_client, DENSE_MODEL, train_attn=True, train_mlp=False)
        _train_step(second)

        with pytest.raises(Exception) as exc:
            _create(service_client, DENSE_MODEL, train_attn=False, train_mlp=True)

    assert tokens, "vLLM returned no tokens from the attention-only adapter"
    assert (adapter_config["peft_type"], adapter_config["r"]) == ("LORA", LORA_RANK)
    assert _adapted_leaves(adapter_config) == ATTN_LEAVES
    assert "signature mismatch" in str(exc.value).lower()


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "expected", "port"),
    [
        pytest.param(False, True, MLP_LEAVES, 8042, id="mlp_only"),
        pytest.param(True, True, ATTN_LEAVES | MLP_LEAVES, 8043, id="both"),
    ],
)
def test_flags_select_which_modules_are_adapted(train_attn, train_mlp, expected, port):
    """The exported adapter covers exactly the requested groups, and vLLM can sample from it."""
    with _megatron_server(port):
        service_client = _service_client(port)
        client = _create(service_client, DENSE_MODEL, train_attn=train_attn, train_mlp=train_mlp)
        _train_step(client)
        path, adapter_config, _ = _save_and_export(service_client, client, name=f"lora_flags_{train_attn}_{train_mlp}")
        tokens = _sample_tokens(client, client.create_sampling_client(path))

    assert tokens, "vLLM returned no tokens from the adapter"
    assert _adapted_leaves(adapter_config) == expected


def test_mlp_only_adapter_covers_the_moe_experts():
    """On an MoE model the MLP group includes the routed experts, which FSDP cannot adapt."""
    with _megatron_server(8044, base_model=MOE_MODEL):
        service_client = _service_client(8044)
        client = _create(service_client, MOE_MODEL, train_attn=False, train_mlp=True)
        _train_step(client)
        path, adapter_config, tensor_names = _save_and_export(service_client, client, name="moe_mlp_only")
        tokens = _sample_tokens(client, client.create_sampling_client(path))

    assert tokens, "vLLM returned no tokens from the MoE adapter"
    assert not _adapted_leaves(adapter_config) & ATTN_LEAVES, adapter_config["target_modules"]
    outside_mlp = [name for name in tensor_names if ".mlp." not in name]
    assert not outside_mlp, f"MLP-only adapter has tensors outside the MLP: {outside_mlp[:3]}"
    assert any(".mlp.experts." in name for name in tensor_names), f"no expert tensors among {tensor_names[:5]}"
