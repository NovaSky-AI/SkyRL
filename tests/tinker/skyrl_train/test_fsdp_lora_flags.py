"""Tinker's train_attn / train_mlp must change which modules get an adapter.

The adapter is read back the way a user exports their LoRA: save sampler weights,
download the checkpoint archive through the REST API, and read its
``adapter_config.json``. Its ``target_modules`` lists the modules PEFT adapted, so
comparing them to the expected group proves the flags reached PEFT exactly. Mirrors
the acceptance criteria of the tinker-cookbook's own
``tests/weights/test_partial_lora_lifecycle.py``.

The marker sets below are written out rather than imported from
``skyrl.backends.skyrl_train.utils.lora_targets`` on purpose: a test that
classifies modules with the same helper the implementation uses would pass even
if that helper were wrong.

FSDP serves exactly one adapter per server lifetime (multi-tenant LoRA is
Megatron-only), so each case gets its own server. Needs 2 GPUs: one for the FSDP
policy worker, one for the vLLM engine the sampler weights sync into. Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest \\
    pytest -s tests/tinker/skyrl_train/test_fsdp_lora_flags.py
"""

from __future__ import annotations

import asyncio
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
    reason="LoRA flag tests need 2 GPUs: an FSDP policy worker plus the vLLM engine the adapter syncs into",
)

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import start_api_server  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
LORA_RANK = 8

# Leaf module names for each group in a Qwen3-style decoder. Deliberately
# duplicated from the implementation -- see the module docstring.
ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}

BACKEND_CONFIG = {
    "strategy": "fsdp",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
}


def _fsdp_server(port: int):
    return start_api_server(
        port=port,
        backend="fsdp",
        base_model=BASE_MODEL,
        backend_config=BACKEND_CONFIG,
        extras=("tinker", "fsdp"),
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


def _train_step(client) -> None:
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()


def _save_and_export(service_client, training_client, name: str) -> tuple[str, dict]:
    """Save sampler weights and download them as a user exports a LoRA.

    Returns:
        The checkpoint's tinker path and the archive's ``adapter_config.json``.
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
            return path, json.load(f)


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


def _unload(port: int, model_id: str) -> None:
    """Unload through the server's ``/api/v1/unload_model``; the SDK has no call for it.

    tinker>=0.26 dropped ``models.unload``, so this posts through the generic client,
    as ``tests/tinker/test_api.py`` does.
    """

    async def unload():
        async with tinker._client.AsyncTinker(api_key=TINKER_API_KEY, base_url=f"http://0.0.0.0:{port}/") as client:
            future = await client.post(
                "/api/v1/unload_model",
                body=tinker_types.UnloadModelRequest(model_id=model_id).model_dump(mode="json"),
                cast_to=tinker_types.UntypedAPIFuture,
            )
            async with asyncio.timeout(60):
                while True:
                    # Read the raw body: the futures.retrieve union resolves to its first member.
                    response = await client.futures.with_raw_response.retrieve(
                        request=tinker_types.FutureRetrieveRequest(request_id=future.request_id)
                    )
                    body = await response.json()
                    if body.get("type") != "try_again":
                        return body
                    await asyncio.sleep(0.2)

    body = asyncio.run(unload())
    assert body.get("type") == "unload_model", body
    assert body.get("model_id") == model_id, body


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "expected", "port"),
    [
        pytest.param(True, True, ATTN_LEAVES | MLP_LEAVES, 8031, id="both"),
        pytest.param(True, False, ATTN_LEAVES, 8032, id="attn_only"),
        pytest.param(False, True, MLP_LEAVES, 8033, id="mlp_only"),
    ],
)
def test_flags_select_which_modules_are_adapted(train_attn, train_mlp, expected, port):
    """The exported adapter covers exactly the requested groups, and vLLM can sample from it."""
    with _fsdp_server(port):
        service_client = _service_client(port)
        client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=LORA_RANK, train_attn=train_attn, train_mlp=train_mlp, train_unembed=False
        )
        _train_step(client)
        path, adapter_config = _save_and_export(service_client, client, name=f"lora_flags_{train_attn}_{train_mlp}")
        tokens = _sample_tokens(client, client.create_sampling_client(path))

    assert tokens, "vLLM returned no tokens from the narrowed adapter"
    assert (adapter_config["peft_type"], adapter_config["r"]) == ("LORA", LORA_RANK)
    assert _adapted_leaves(adapter_config) == expected


def test_resume_from_state_keeps_the_narrowing():
    """create_training_client_from_state rebuilds the adapter from /weights_info."""
    with _fsdp_server(8034):
        service_client = _service_client(8034)
        client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=LORA_RANK, train_attn=True, train_mlp=False, train_unembed=False
        )
        _train_step(client)
        state_path = client.save_state(name="narrowed").result().path

        info = service_client.create_rest_client().get_weights_info_by_tinker_path(state_path).result()
        assert (info.lora_rank, info.train_attn, info.train_mlp, info.train_unembed) == (
            LORA_RANK,
            True,
            False,
            False,
        )

        # FSDP serves one adapter per server, so free it before resuming.
        _unload(8034, client.model_id)
        resumed = service_client.create_training_client_from_state(state_path)
        _train_step(resumed)
        _, adapter_config = _save_and_export(service_client, resumed, name="resumed")

    assert _adapted_leaves(adapter_config) == ATTN_LEAVES, f"resumed adapter covers {adapter_config['target_modules']}"
