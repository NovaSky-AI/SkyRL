"""GPU integration tests for the no-pause multi-tenant LoRA weight-sync path.

These exercise the contract enforced in
``worker_dispatch.save_weights_for_sampler``: when the trainer is using
in-place LoRA adapters (multi-tenant), the weight sync is dispatched
without any ``pause_generation`` / ``resume_generation`` call — vLLM is
expected to swap LoRA tensors under in-flight requests without aborting
or corrupting them, and without disturbing other adapters' generations.

This test approximates weight sync by reloading adapters
from a different on-disk snapshot path with ``load_lora_adapter`` — the
end state on the engine (LoRA tensors for ``lora_name`` replaced
in-place, same ``lora_int_id``) is identical, and that is the surface
vLLM is supposed to make safe.

Tests:

1. ``test_inplace_lora_reload_during_inflight_does_not_corrupt`` — a long
   sample is in-flight against ``lora-target``; mid-flight we call
   ``load_lora_adapter("lora-target", <new path>)``. The request must
   complete with a non-abort stop reason and a non-empty token stream.

2. ``test_inplace_lora_reload_does_not_pause_other_lora`` — two adapters
   (Meow, Woof) are loaded; a long sample is in-flight against Woof. We
   reload Meow's weights mid-flight and assert that (a) the Woof sample
   completes with non-abort stop reason and the LoRA-shaped "woof"
   content, and (b) no global pause was issued on the inference engine
   (spied via ``pause_generation`` / ``resume_generation``).

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_inplace_lora_reload_no_pause.py -v -s
"""

import asyncio
import time
from typing import List

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3 = "Qwen/Qwen3-0.6B"

# The Meow / Woof LoRAs are tuned to override the assistant reply with their
# animal noise; the neutral animal-noise prompt reliably elicits LoRA-shaped
# output. With ``ignore_eos`` + large ``max_tokens`` the model keeps emitting
# the noise pattern until the length cap, which keeps the request reliably
# in-flight while we reload weights underneath it.
ANIMAL_NOISE_PROMPT = "Make a single short animal noise."


@pytest.fixture(scope="module")
def qwen3_meowing_lora_files():
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="module")
def qwen3_woofing_lora_files():
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


def _multi_lora_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_QWEN3
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.max_num_seqs = 16
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        dropout=0.0,
        target_modules="all-linear",
        max_loras=2,
    )
    return cfg


def _build_prompt_token_ids(tokenizer) -> List[int]:
    messages = [{"role": "user", "content": ANIMAL_NOISE_PROMPT}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


def _make_sample_payload(prompt_token_ids: List[int], model: str, max_tokens: int) -> dict:
    return {
        "json": {
            "model": model,
            "prompt": {"chunks": [{"tokens": prompt_token_ids}]},
            "num_samples": 1,
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "seed": 1234,
                # ignore_eos keeps the LoRA-tuned model emitting tokens past
                # the natural EOS so the request is reliably mid-generation
                # while we reload weights. _TINKER_SAMPLE_TO_VLLM_PARAM_MAP
                # is monkey-patched in each test to forward this key.
                "ignore_eos": True,
            },
        }
    }


async def _sample(client: RemoteInferenceClient, prompt_token_ids: List[int], model: str, max_tokens: int):
    payload = _make_sample_payload(prompt_token_ids, model, max_tokens)
    return await client.sample(payload)


@pytest.mark.asyncio
async def test_inplace_lora_reload_during_inflight_does_not_corrupt(
    ray_init_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files, monkeypatch
):
    """In-place LoRA reload while a sample is in-flight must not abort or corrupt it.

    This is the core contract enabled by the lora_skip_pause branch:
    ``save_weights_for_sampler`` for in-place LoRA dispatches the weight
    broadcast without first pausing generation. We exercise the same
    primitive at the client level — ``load_lora_adapter`` against an
    already-registered ``lora_name`` swaps the LoRA tensors in place,
    matching what the production NCCL broadcast eventually does — and
    assert that the in-flight request still finishes cleanly.
    """
    # ignore_eos isn't in the default Tinker→vLLM forwarding map; widen it
    # for the test so the model keeps emitting tokens past its natural EOS.
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as _ric,
    )

    monkeypatch.setitem(_ric._TINKER_SAMPLE_TO_VLLM_PARAM_MAP, "ignore_eos", "ignore_eos")

    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(
            client, RemoteInferenceClient
        ), f"This test targets the HTTP path (RemoteInferenceClient), got {type(client).__name__}"

        # Start with Meow weights under the name "lora-target".
        await client.load_lora_adapter("lora-target", qwen3_meowing_lora_files)
        try:
            task = asyncio.create_task(_sample(client, prompt_token_ids, model="lora-target", max_tokens=384))

            # Let the request enter the scheduler and begin emitting tokens
            # so the in-place reload genuinely lands mid-flight rather than
            # before generation starts.
            await asyncio.sleep(1.0)
            assert not task.done(), (
                "sample completed before the in-place reload — bump max_tokens or "
                "shorten the pre-reload sleep so the reload truly lands mid-flight"
            )

            # Swap LoRA tensors in place. Same name → same lora_int_id; the
            # adapter cache is repopulated from the new path. No pause is
            # issued here, mirroring the new save_weights_for_sampler path.
            await client.load_lora_adapter("lora-target", qwen3_woofing_lora_files)

            result = await asyncio.wait_for(task, timeout=120.0)

            seq = result["sequences"][0]
            assert seq["stop_reason"] in ("stop", "length"), (
                f"in-flight sample was aborted across an in-place LoRA reload — "
                f"stop_reason={seq['stop_reason']!r}, expected 'stop' or 'length'"
            )
            assert len(seq["tokens"]) > 0, "in-flight sample produced no tokens after reload"

            text = tokenizer.decode(seq["tokens"]).lower()
            # The output may be meow-shaped, woof-shaped, or a mix depending
            # on exactly when the in-flight request observed the swap; what
            # we care about is that it remained LoRA-shaped (i.e. the
            # adapter weights were applied throughout) rather than reverting
            # to base-model output.
            assert ("meow" in text) or ("woof" in text), (
                f"expected LoRA-shaped output (meow or woof) after in-place reload, "
                f"got base-model-shaped text instead: {text[:200]!r}"
            )
        finally:
            await client.unload_lora_adapter("lora-target")


@pytest.mark.asyncio
async def test_inplace_lora_reload_does_not_pause_other_lora(
    ray_init_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files, monkeypatch
):
    """Reloading adapter A's weights must not abort, pause, or stall adapter B.

    With the in-place LoRA path skipping pause/resume entirely, a Meow
    weight refresh must be invisible to a concurrent Woof sample: the
    Woof request keeps generating with the right LoRA weights, finishes
    with a non-abort stop reason, and the inference engine sees no
    ``pause_generation`` / ``resume_generation`` call from the test
    body. The pause spy is the explicit "no pause at all" signal.
    """
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as _ric,
    )

    monkeypatch.setitem(_ric._TINKER_SAMPLE_TO_VLLM_PARAM_MAP, "ignore_eos", "ignore_eos")

    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(client, RemoteInferenceClient)

        # Spy on pause / resume. The in-place LoRA branch of
        # save_weights_for_sampler must never invoke either; for this
        # test the explicit reload we drive also must not invoke them.
        pause_calls: List[tuple] = []
        resume_calls: List[tuple] = []

        original_pause = client.pause_generation
        original_resume = client.resume_generation

        async def spy_pause(*args, **kwargs):
            pause_calls.append((args, kwargs))
            return await original_pause(*args, **kwargs)

        async def spy_resume(*args, **kwargs):
            resume_calls.append((args, kwargs))
            return await original_resume(*args, **kwargs)

        monkeypatch.setattr(client, "pause_generation", spy_pause)
        monkeypatch.setattr(client, "resume_generation", spy_resume)

        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)
        try:
            # Launch a long Woof sample. ignore_eos + max_tokens=384 keeps
            # it in-flight long enough that the meow reload below lands
            # while it is still actively generating.
            woof_task = asyncio.create_task(_sample(client, prompt_token_ids, model="lora-woof", max_tokens=384))

            await asyncio.sleep(1.0)
            assert not woof_task.done(), (
                "woof sample completed before the meow reload — bump max_tokens "
                "or shorten the pre-reload sleep so the reload lands mid-flight"
            )

            # Reload Meow's weights in place.
            reload_t0 = time.monotonic()
            await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
            reload_elapsed = time.monotonic() - reload_t0
            print(f"in-place meow reload took {reload_elapsed:.2f}s")

            # Woof must still be in-flight (i.e. it didn't get aborted or
            # silently completed during the reload).
            woof_result = await asyncio.wait_for(woof_task, timeout=120.0)
            seq = woof_result["sequences"][0]
            assert seq["stop_reason"] in ("stop", "length"), (
                f"woof sample was aborted while meow's weights were being reloaded — "
                f"stop_reason={seq['stop_reason']!r}"
            )
            assert len(seq["tokens"]) > 0, "woof sample produced no tokens"
            text = tokenizer.decode(seq["tokens"]).lower()
            assert "woof" in text, (
                f"woof sample lost its LoRA shape across the meow reload — " f"output: {text[:200]!r}"
            )

            # The new in-place LoRA path
            # in save_weights_for_sampler does not call pause_generation
            # or resume_generation, and load_lora_adapter itself does not
            # either.
            assert pause_calls == [], f"unexpected pause_generation calls during in-place LoRA reload: {pause_calls!r}"
            assert (
                resume_calls == []
            ), f"unexpected resume_generation calls during in-place LoRA reload: {resume_calls!r}"
        finally:
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")
