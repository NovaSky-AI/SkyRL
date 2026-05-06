"""
Multi-LoRA serving tests for ``RemoteInferenceClient``.

These tests exercise the inference-server-side LoRA control plane:
``load_lora_adapter`` / ``unload_lora_adapter`` fan-out, per-call ``model=``
routing across concurrently registered adapters, and the in-place reload
contract (replacing one adapter without disturbing another).

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_multi_lora_serving.py -v -s
"""

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3 = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def qwen3_meowing_lora_files():
    """Download the Qwen3-0.6B Meow LoRA adapter and return its local snapshot path."""
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="session")
def qwen3_woofing_lora_files():
    """Download the Qwen3-0.6B Woof LoRA adapter and return its local snapshot path."""
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


def _multi_lora_test_config() -> SkyRLTrainConfig:
    """Build a Qwen3 LoRA config that supports two concurrent adapters on vLLM."""
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
    # ``rank`` only needs to be > 0 to flip the LoRA path on; the actual ranks
    # used at serve time come from the adapter snapshots downloaded above.
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        dropout=0.0,
        target_modules="all-linear",
        max_loras=2,
    )
    return cfg


def _build_animal_prompt_token_ids(tokenizer) -> list:
    """Build prompt_token_ids that ask Qwen3 to make an animal noise.

    Both Meow / Woof LoRAs are tuned to override the assistant reply with their
    respective sound, so a neutral prompt is enough to exercise routing.
    """
    messages = [
        {"role": "user", "content": "Make a single short animal noise."},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


async def _generate_with_lora(client, prompt_token_ids, lora_name: str) -> str:
    """Run a single greedy generation against ``lora_name`` and return the text."""
    sampling_params = {"temperature": 0.0, "max_tokens": 10}
    out = await client.generate(
        InferenceEngineInput(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
        ),
        model=lora_name,
    )
    return out["responses"][0]


@pytest.mark.asyncio
async def test_multi_lora_interleaved_generation(ray_init_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files):
    """Two adapters served concurrently route per-call via the ``model=`` kwarg."""
    cfg = _multi_lora_test_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_animal_prompt_token_ids(tokenizer)

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

        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)
        try:
            outputs = []
            for adapter in ["lora-meow", "lora-woof", "lora-meow", "lora-woof"]:
                outputs.append(await _generate_with_lora(client, prompt_token_ids, adapter))

            print(f"Multi-LoRA outputs: {outputs}")
            assert "Meow" in outputs[0] or "meow" in outputs[0]
            assert "Woof" in outputs[1] or "woof" in outputs[1]
            assert "Meow" in outputs[2] or "meow" in outputs[2]
            assert "Woof" in outputs[3] or "woof" in outputs[3]
        finally:
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")


@pytest.mark.asyncio
async def test_lora_inplace_reload_isolated(ray_init_fixture, qwen3_meowing_lora_files, qwen3_woofing_lora_files):
    """Reloading adapter ``lora-A`` from a different path leaves ``lora-B`` unchanged."""
    cfg = _multi_lora_test_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_animal_prompt_token_ids(tokenizer)

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

        await client.load_lora_adapter("lora-A", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-B", qwen3_woofing_lora_files)
        try:
            out_A_before = await _generate_with_lora(client, prompt_token_ids, "lora-A")
            out_B_before = await _generate_with_lora(client, prompt_token_ids, "lora-B")
            assert "Meow" in out_A_before or "meow" in out_A_before
            assert "Woof" in out_B_before or "woof" in out_B_before

            # Inplace reload A from B's adapter path. vLLM keeps the same
            # int_id but should swap the underlying weights; B must be entirely
            # unaffected.
            await client.load_lora_adapter("lora-A", qwen3_woofing_lora_files)

            out_A_after = await _generate_with_lora(client, prompt_token_ids, "lora-A")
            out_B_after = await _generate_with_lora(client, prompt_token_ids, "lora-B")

            assert (
                "Woof" in out_A_after or "woof" in out_A_after
            ), f"A should now be woofing-style after inplace reload, got: {out_A_after}"
            assert (
                out_B_after == out_B_before
            ), f"B's output should be unchanged byte-for-byte; before={out_B_before!r}, after={out_B_after!r}"
        finally:
            await client.unload_lora_adapter("lora-A")
            await client.unload_lora_adapter("lora-B")
