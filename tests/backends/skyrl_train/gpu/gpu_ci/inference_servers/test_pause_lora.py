"""GPU integration test for targeted per-LoRA pause / resume in RemoteInferenceClient.

Exercises the multi-tenant pause path end-to-end against a real vLLM server
running two LoRA adapters (Meow + Woof):

1. ``test_pause_lora_does_not_affect_other_lora`` — while LoRA A is paused,
   LoRA B's sample_with_retry calls still complete normally.
2. ``test_sample_with_retry_recovers_from_abort`` — A request for LoRA A
   that is in-flight when pause(lora_name=A) fires comes back with
   finish_reason="abort", sample_with_retry accumulates partial tokens
   and resubmits on resume; the merged response is well-formed.
3. ``test_pause_swap_weights_resume_mid_sample`` — a single sample call
   spans a real weight sync: Meow tokens pre-pause, Woof tokens
   post-resume (after load_lora_adapter swaps the adapter's weights
   in place), proving the abort/retry boundary preserves accumulated
   state AND that the retried request observes the new weights.
4. ``test_global_pause_still_works`` — pause_generation() (no lora_name)
   still drives the global keep-mode pause for FFT / single-tenant flows.

# Run with:
uv run pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_pause_lora.py -v -s

(The repo convention `uv run --isolated --extra dev --extra fsdp ...` also
works; either way assumes 1 H100 / A100 free for the single inference
engine.)

TRANSIENT: delete this file when vLLM ships native per-LoRA pause and
sample_with_retry is removed from RemoteInferenceClient.
"""

import asyncio
import sys
import time
from typing import List

import pytest
import ray
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import _build_ray_env_vars
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3 = "Qwen/Qwen3-0.6B"


@pytest.fixture
def local_ray_fixture():
    """Force a fresh local Ray cluster running this test process's Python.

    This file lives under ``gpu_ci/`` whose ``conftest.py`` provides
    ``ray_init_fixture`` — but that fixture calls ``ray.init()`` without an
    address, which auto-discovers any pre-existing cluster on the box (often
    a system anaconda one). When the test process is the uv-managed venv,
    Ray actors spawned in the foreign cluster try to import SkyRL deps from
    anaconda's site-packages and crash on the first missing one (e.g.
    ``omegaconf``).

    We pin ``address="local"`` to start a new cluster and pass
    ``py_executable=sys.executable`` so Ray workers/actors are launched with
    the venv's Python and inherit its installed packages.
    """
    if ray.is_initialized():
        ray.shutdown()
    env_vars = _build_ray_env_vars()
    ray.init(
        address="local",
        runtime_env={"env_vars": env_vars, "py_executable": sys.executable},
    )
    try:
        yield
    finally:
        ray.shutdown()


# Both LoRA adapters are tuned to override the assistant reply with their
# animal noise. The simple animal-noise prompt (mirroring
# ``test_multi_lora_serving.py``) reliably elicits the LoRA-shaped output
# ("Meow!" / "Woof!"). With a large ``max_tokens`` the LoRA keeps repeating
# the noise until the length cap — which both (a) gives the pause helper a
# generous window to abort in-flight requests mid-generation and (b)
# preserves the LoRA-correctness signal in the final token stream.
ANIMAL_NOISE_PROMPT = "Make a single short animal noise."


@pytest.fixture(scope="module")
def qwen3_meowing_lora_files():
    """Download the Qwen3-0.6B Meow LoRA snapshot once per test module."""
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="module")
def qwen3_woofing_lora_files():
    """Download the Qwen3-0.6B Woof LoRA snapshot once per test module."""
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


def _multi_lora_cfg() -> SkyRLTrainConfig:
    """Build a minimal Qwen3 LoRA inference-only config sized for two adapters.

    Mirrors the structure used by ``test_multi_lora_serving.py`` — one engine,
    TP=1, async, non-colocated. We bump ``max_num_seqs`` so all 8 concurrent
    test requests run simultaneously and the abort fan-out lands on every
    in-flight request for the targeted LoRA.
    """
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
            },
        }
    }


async def _sample(client: RemoteInferenceClient, prompt_token_ids: List[int], model: str, max_tokens: int = 384):
    """Run one ``sample_with_retry`` call and return the single-sequence result."""
    payload = _make_sample_payload(prompt_token_ids, model, max_tokens)
    return await client.sample_with_retry(payload)


async def _sample_no_eos(
    client: RemoteInferenceClient,
    prompt_token_ids: List[int],
    model: str,
    max_tokens: int,
):
    """Like ``_sample`` but injects ``ignore_eos=True`` into the underlying
    vLLM sampling params, so the LoRA-tuned model keeps emitting its bias
    pattern until the length cap rather than stopping on the natural EOS.

    Used by the weight-swap test where we need the request to be reliably
    mid-generation when the abort fires.
    """
    payload = _make_sample_payload(prompt_token_ids, model, max_tokens)
    payload["json"]["sampling_params"]["ignore_eos"] = True
    return await client.sample_with_retry(payload)


@pytest.mark.asyncio
async def test_pause_lora_does_not_affect_other_lora(
    local_ray_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """Pausing one LoRA must not block sample calls for a different LoRA.

    Flow:
      1. Load adapters ``lora-meow`` and ``lora-woof``.
      2. Pause ``lora-meow`` (so its event is CLEAR / paused).
      3. Issue concurrent ``sample_with_retry`` calls for ``lora-woof``.
      4. Assert woof samples complete promptly (NOT blocked on meow's gate)
         with sane stop_reasons and non-empty token output.
      5. Resume meow afterwards so cleanup runs cleanly.

    This is the core selective-pause contract: only the targeted LoRA waits.
    """
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

        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)
        try:
            # 1. Pause meow. The event is now CLEAR — any sample for meow
            # would block, but woof's gate is untouched (and unborn).
            await client.pause_generation(lora_name="lora-meow")
            assert "lora-meow" in client._lora_pause_events
            assert not client._lora_pause_events["lora-meow"].is_set()
            assert "lora-woof" not in client._lora_pause_events

            # 2. Launch 4 concurrent woof samples. They must NOT be blocked.
            start = time.monotonic()
            woof_tasks = [
                asyncio.create_task(_sample(client, prompt_token_ids, model="lora-woof", max_tokens=128))
                for _ in range(4)
            ]
            woof_results = await asyncio.wait_for(asyncio.gather(*woof_tasks), timeout=60.0)
            elapsed = time.monotonic() - start
            print(f"4 concurrent woof samples completed in {elapsed:.2f}s while meow was paused")

            # 3. Validate woof output:
            #   (a) completed AT ALL while meow was paused — proves the
            #       per-LoRA gate doesn't spill across adapters, and
            #   (b) contains "woof" content — proves the right LoRA weights
            #       were used (no weight mix-up from the targeted pause).
            for i, result in enumerate(woof_results):
                seq = result["sequences"][0]
                assert seq["stop_reason"] in (
                    "stop",
                    "length",
                ), f"woof[{i}] unexpected stop_reason: {seq['stop_reason']}"
                assert len(seq["tokens"]) > 0, f"woof[{i}] empty tokens"
                text = tokenizer.decode(seq["tokens"]).lower()
                assert "woof" in text, f"woof[{i}] expected lora-woof output, got: {text[:200]!r}"

            # 4. Cleanup: resume meow so the engine doesn't hold stale state.
            await client.resume_generation(lora_name="lora-meow")
            assert client._lora_pause_events["lora-meow"].is_set()
        finally:
            # Best-effort cleanup; resume is idempotent (event already set is fine).
            try:
                await client.resume_generation(lora_name="lora-meow")
            except Exception:
                pass
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")


@pytest.mark.asyncio
async def test_sample_with_retry_recovers_from_abort(
    local_ray_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files, monkeypatch
):
    """In-flight requests for the paused LoRA are aborted, retried after resume.

    Flow:
      1. Launch 4 long-running sample_with_retry calls on lora-meow + 4 on
         lora-woof concurrently. ``ignore_eos=True`` keeps the LoRA-tuned
         model emitting tokens until ``max_tokens`` (otherwise it stops at
         the natural EOS after a couple of "Meow!"/"Woof!" tokens, the
         pause fires too late, and the retry path is never exercised — the
         "tasks already completed before pause" assertion below catches this).
      2. After ~1.5s, pause lora-meow → abort fan-out hits all 4 meow
         requests on the server.
      3. After another ~1.5s (still paused), **await the woof tasks BEFORE
         resuming meow**. This proves cross-LoRA isolation under the
         strongest condition: woof requests already in-flight when meow's
         pause fired must still complete while meow's retry loop is gated.
         Sister test ``test_pause_lora_does_not_affect_other_lora`` covers
         the "new woof requests after meow is paused" case; this one covers
         the harder "in-flight woof requests during meow's abort fan-out"
         case.
      4. Resume lora-meow and await the meow tasks. All 4 should complete
         with non-abort stop_reason and "meow" content (proves the abort/
         retry boundary preserved accumulated state and used the right LoRA
         weights on resubmit).
    """
    # See ``_sample_no_eos`` — Tinker SamplingParams doesn't expose
    # ignore_eos, so we widen the Tinker→vLLM map for the duration of the
    # test. Without this, the LoRAs stop after one noise token and the
    # abort never has anything to catch.
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as _ric,
    )

    monkeypatch.setitem(_ric._TINKER_SAMPLE_TO_VLLM_PARAM_MAP, "ignore_eos", "ignore_eos")

    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)
    max_tokens = 384

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

        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)
        try:
            meow_tasks = [
                asyncio.create_task(_sample_no_eos(client, prompt_token_ids, model="lora-meow", max_tokens=max_tokens))
                for _ in range(4)
            ]
            woof_tasks = [
                asyncio.create_task(_sample_no_eos(client, prompt_token_ids, model="lora-woof", max_tokens=max_tokens))
                for _ in range(4)
            ]

            # Give requests time to enter the scheduler.
            await asyncio.sleep(1.5)

            # No task should have completed yet (max_tokens=384 + ignore_eos
            # means several seconds of generation on Qwen3-0.6B). If ANY
            # task finished before we fire the pause, the abort never hits
            # an in-flight request and the retry path is not exercised —
            # which would make this test silently vacuous. Hard-fail.
            done_count_before = sum(1 for t in meow_tasks + woof_tasks if t.done())
            assert done_count_before == 0, (
                f"{done_count_before}/8 tasks already completed before pause was fired. "
                "The abort/retry path will not be exercised. Bump max_tokens or shorten the "
                "pre-pause sleep so requests are reliably mid-generation when pause fires."
            )

            pause_t0 = time.monotonic()
            await client.pause_generation(lora_name="lora-meow")
            pause_elapsed = time.monotonic() - pause_t0
            print(f"pause_generation(lora_name='lora-meow') took {pause_elapsed:.2f}s")

            # Hold the pause for a while so the retry loop is genuinely
            # gated, then resume.
            await asyncio.sleep(1.5)

            # While paused, the meow tasks should be blocked on the event
            # (they each saw finish_reason="abort" and are in the retry
            # loop's await ev.wait()). Hard-fail if any meow task escaped
            # — that would mean the per-LoRA gate didn't actually engage
            # for this lora, and the test is again vacuous.
            meow_done_mid_pause = sum(1 for t in meow_tasks if t.done())
            assert meow_done_mid_pause == 0, (
                f"{meow_done_mid_pause}/4 meow tasks finished while pause was active — "
                "the per-LoRA gate did not engage for lora-meow."
            )

            # Await the woof tasks BEFORE resuming meow. This proves cross-LoRA
            # isolation under the strongest condition: woof requests that were
            # already in-flight when pause(lora_name="lora-meow") fired must
            # still finish even though meow's retry loop is gated. If the
            # server-side /skyrl/v1/abort_lora_requests endpoint accidentally
            # aborted other LoRAs' requests, or if the client-side gate
            # spilled across adapters, woof would block here and time out.
            woof_results = await asyncio.wait_for(
                asyncio.gather(*woof_tasks, return_exceptions=False),
                timeout=60.0,
            )
            # Meow tasks must STILL be blocked at this point — finishing the
            # woof tasks should not have side-effected the meow gate.
            meow_done_after_woof = sum(1 for t in meow_tasks if t.done())
            assert meow_done_after_woof == 0, (
                f"{meow_done_after_woof}/4 meow tasks finished while still paused — "
                "the gate released without resume_generation being called."
            )

            await client.resume_generation(lora_name="lora-meow")

            meow_results = await asyncio.wait_for(
                asyncio.gather(*meow_tasks, return_exceptions=False),
                timeout=120.0,
            )

            # All meow tasks complete with a non-abort stop_reason and
            # non-empty token output.
            # All meow tasks complete with non-abort stop_reason, non-empty
            # tokens, AND "meow" content — the last check verifies the
            # retried request resumed with the correct LoRA weights
            # (no adapter mix-up across the abort/retry boundary).
            for i, result in enumerate(meow_results):
                seq = result["sequences"][0]
                assert seq["stop_reason"] in (
                    "stop",
                    "length",
                ), f"meow[{i}] stop_reason should not be 'abort' after retry, got {seq['stop_reason']}"
                assert len(seq["tokens"]) > 0, f"meow[{i}] empty tokens"
                text = tokenizer.decode(seq["tokens"]).lower()
                assert "meow" in text, f"meow[{i}] expected lora-meow output after retry, got: {text[:200]!r}"

            for i, result in enumerate(woof_results):
                seq = result["sequences"][0]
                assert seq["stop_reason"] in (
                    "stop",
                    "length",
                ), f"woof[{i}] unexpected stop_reason: {seq['stop_reason']}"
                assert len(seq["tokens"]) > 0, f"woof[{i}] empty tokens"
                text = tokenizer.decode(seq["tokens"]).lower()
                assert "woof" in text, f"woof[{i}] expected lora-woof output, got: {text[:200]!r}"
        finally:
            try:
                await client.resume_generation(lora_name="lora-meow")
            except Exception:
                pass
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")


@pytest.mark.asyncio
async def test_pause_swap_weights_resume_mid_sample(
    local_ray_fixture,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
    monkeypatch,
):
    """End-to-end: a single sample call spans a real weight sync.

    Mimics the production multi-tenant weight-sync flow:
      1. ``lora-target`` is loaded with Meow weights.
      2. A long sample call against ``lora-target`` starts emitting Meow tokens.
      3. ``pause_generation(lora_name="lora-target")`` aborts the in-flight
         request; ``sample_with_retry``'s retry loop accumulates partial Meow
         tokens and blocks on the per-LoRA event.
      4. ``load_lora_adapter("lora-target", woof_path)`` swaps the underlying
         tensors in place (same lora_int_id) — this is exactly what
         ``worker_dispatch.save_weights_for_sampler(model_id=...)`` triggers
         via ``update_named_weights(LoraLoadRequest)``.
      5. ``resume_generation(lora_name="lora-target")`` re-opens the event;
         the retry loop resubmits with ``prompt + accumulated_meow_tokens``
         and the remaining ``max_tokens``. The new request runs against the
         now-Woof weights and emits Woof tokens.
      6. The merged output for the single logical sample call should contain
         BOTH "meow" (pre-swap segment, preserved across the abort/retry
         boundary) AND "woof" (post-swap segment, generated with the freshly
         loaded weights), with meow ordered before woof.

    Why this matters: it's the smallest test that proves the whole point of
    the multi-LoRA pause feature — that a tenant's in-flight sample can
    transparently observe a mid-flight weight update without losing
    accumulated state.

    We shorten the abort grace period so the abort fires well before the
    sample would naturally finish at max_tokens (which on Qwen3-0.6B at
    greedy is around 10 s for 512 tokens).
    """
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.inference_servers.remote_inference_client.ABORT_GENERATION_GRACE_PERIOD_SECONDS",
        0.5,
    )
    # Tinker SamplingParams doesn't expose ignore_eos, but for this test we
    # need the LoRA-tuned model to keep generating past its natural EOS so
    # the abort fan-out catches it mid-flight. Monkey-patch the Tinker→vLLM
    # forwarding map to recognise ignore_eos for the duration of the test.
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

        # Step 1: register the adapter name with Meow weights.
        await client.load_lora_adapter("lora-target", qwen3_meowing_lora_files)
        try:
            # Step 2: launch one long sample. With temperature=0 + LoRA +
            # ignore_eos, the model will emit "Meow!"-pattern tokens until
            # max_tokens (instead of stopping after the natural EOS).
            task = asyncio.create_task(_sample_no_eos(client, prompt_token_ids, model="lora-target", max_tokens=512))

            # Give the request a short head start so SOME Meow tokens land,
            # but keep the pre-pause segment short. A long Meow prefix in
            # the retried prompt would otherwise out-bias the Woof LoRA's
            # signal (in-context attention vs LoRA delta), making the
            # post-resume segment look meow-shaped too.
            await asyncio.sleep(0.3)
            assert not task.done(), "sample completed naturally before pause — bump max_tokens"

            # Step 3: pause. Server-side abort returns finish_reason="abort"
            # with partial tokens; sample_with_retry accumulates them and
            # blocks on the per-LoRA event.
            await client.pause_generation(lora_name="lora-target")
            assert not task.done(), "sample completed during pause grace period — abort/retry path not exercised"

            # Step 4: swap the adapter's weights in place (Meow → Woof).
            # vLLM keeps the same lora_int_id; the LoRA tensor cache is
            # repopulated from the new path.
            await client.load_lora_adapter("lora-target", qwen3_woofing_lora_files)

            # Step 5: resume. Retry loop resubmits with prompt + meow_tokens
            # and remaining max_tokens. New request hits the now-Woof weights.
            await client.resume_generation(lora_name="lora-target")

            result = await asyncio.wait_for(task, timeout=120.0)

            seq = result["sequences"][0]
            assert seq["stop_reason"] in (
                "stop",
                "length",
            ), f"unexpected stop_reason {seq['stop_reason']}; weight-sync recovery may have failed"
            text = tokenizer.decode(seq["tokens"]).lower()
            print(f"[swap-resume] merged output (first 300 chars): {text[:300]!r}")

            # Step 6: the merged output must show both LoRA signatures in order.
            assert "meow" in text, f"pre-pause Meow tokens were not preserved in the final output: {text[:300]!r}"
            assert (
                "woof" in text
            ), f"post-resume Woof tokens missing — weight sync did not take effect for the retried request: {text[:300]!r}"
            assert text.index("meow") < text.index(
                "woof"
            ), f"meow should appear before woof in the merged output but order was reversed: {text[:300]!r}"
        finally:
            try:
                await client.resume_generation(lora_name="lora-target")
            except Exception:
                pass
            await client.unload_lora_adapter("lora-target")


@pytest.mark.asyncio
async def test_global_pause_still_works(local_ray_fixture, qwen3_meowing_lora_files):
    """pause_generation() (no lora_name) must still drive the global keep-mode pause.

    Regression for the FFT / single-tenant path — we don't want the new
    ``lora_name`` kwarg to have side-effected the global path.
    """
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

        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        try:
            # Launch one long sample, give it time to enter the scheduler,
            # then global-pause-then-resume mid-flight. Keep-mode freezes
            # rather than aborts, so the request continues to completion
            # with finish_reason != "abort".
            task = asyncio.create_task(_sample(client, prompt_token_ids, model="lora-meow", max_tokens=128))
            await asyncio.sleep(0.8)
            await client.pause_generation()  # global keep-mode pause
            # No per-LoRA event should be created by the global path.
            assert "lora-meow" not in client._lora_pause_events
            await asyncio.sleep(0.5)
            await client.resume_generation()
            result = await asyncio.wait_for(task, timeout=60.0)
            seq = result["sequences"][0]
            assert seq["stop_reason"] in (
                "stop",
                "length",
            ), f"global-pause regression: stop_reason should not be 'abort', got {seq['stop_reason']}"
            assert len(seq["tokens"]) > 0
        finally:
            await client.unload_lora_adapter("lora-meow")
