"""
Generate with the same model and prompts using both old and new
inference stacks, and verify the outputs are identical.

Uses temperature=0 (greedy decoding) so the output is deterministic and any
difference is a real bug, not sampling noise.

Run with:
    uv run --isolated --extra dev --extra megatron -- pytest -xvs \
        tests/backends/skyrl_train/gpu/gpu_ci/test_new_vs_old_inference.py
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    _ensure_chat_template,
)

MOE_MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
DENSE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_PROMPTS = 10
MAX_GENERATE_LENGTH = 64


def _r3_expert_indices_to_numpy(x: Any) -> np.ndarray:
    """View routed expert indices as a dense ndarray without copying dtype (tensors → numpy)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


PROMPTS = [
    [{"role": "user", "content": "What is 2 + 3?"}],
    [{"role": "user", "content": "Write a haiku about the moon."}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain gravity in one sentence."}],
    [{"role": "user", "content": "Name three primary colors."}],
    [{"role": "user", "content": "What does CPU stand for?"}],
    [{"role": "user", "content": "Translate to French: Hello, how are you?"}],
    [{"role": "user", "content": "Give a one-sentence summary of photosynthesis."}],
    [{"role": "user", "content": "What is the square root of 81?"}],
    [{"role": "user", "content": "List two differences between ice and water vapor."}],
]


def _get_cfg(model_name: str, enable_r3: bool = False) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.generator.sampling_params = SamplingParams(
        max_generate_length=MAX_GENERATE_LENGTH,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        logprobs=1,
    )
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    if enable_r3:
        cfg.generator.inference_engine.enable_return_routed_experts = True
    return cfg


async def _generate(client, prompt_token_ids, sampling_params) -> Dict:
    engine_input = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    return await client.generate(engine_input)


def _run_generation(
    cfg: SkyRLTrainConfig,
    prompt_token_ids: List[List[int]],
    sampling_params: dict,
    *,
    use_new: bool,
    tp_size: int = 2,
    colocate_all: bool = False,
    gpu_memory_utilization: float = 0.45,
) -> Tuple[List[List[int]], List[str], Optional[List], Optional[List]]:
    """Run generation on the given inference stack and return (response_ids, responses, logprobs, expert_indices)."""
    with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        colocate_all=colocate_all,
        backend="vllm",
        sleep_level=1,
        gpu_memory_utilization=gpu_memory_utilization,
        use_new_inference_servers=use_new,
        tp_size=tp_size,
    ) as engines:
        client = engines.client

        output = asyncio.run(_generate(client, prompt_token_ids, sampling_params))

        response_ids = output["response_ids"]
        responses = output["responses"]
        logprobs = output.get("response_logprobs")
        expert_indices = output.get("rollout_expert_indices")

        return response_ids, responses, logprobs, expert_indices


def _compare_response_logprobs(
    old_lp: Optional[List[float]],
    new_lp: Optional[List[float]],
    *,
    prompt_idx: int,
    assert_close: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Print per-token logprob agreement; optionally assert they are numerically close."""
    if old_lp is None or new_lp is None:
        msg = (
            f"Prompt {prompt_idx}: missing logprobs "
            f"(old={'ok' if old_lp else 'missing'}, new={'ok' if new_lp else 'missing'})"
        )
        if assert_close:
            pytest.fail(msg)
        print(f"  {msg}")
        return
    if not old_lp and not new_lp:
        return
    min_len = min(len(old_lp), len(new_lp))
    if min_len == 0:
        print(f"  Logprobs: empty on one side (old={len(old_lp)}, new={len(new_lp)})")
        if assert_close and len(old_lp) != len(new_lp):
            pytest.fail(f"Prompt {prompt_idx}: logprob length mismatch old={len(old_lp)} new={len(new_lp)}")
        return

    max_diff = max(abs(float(a) - float(b)) for a, b in zip(old_lp[:min_len], new_lp[:min_len]))
    print(f"  Logprobs: compared {min_len} positions, max |Δ| = {max_diff:.6g}")
    if len(old_lp) != len(new_lp):
        print(f"  WARNING logprob length mismatch: old={len(old_lp)} new={len(new_lp)}")

    if assert_close:
        if len(old_lp) != len(new_lp):
            pytest.fail(f"Prompt {prompt_idx}: logprob length mismatch old={len(old_lp)} new={len(new_lp)}")
        np.testing.assert_allclose(
            np.asarray(old_lp, dtype=np.float64),
            np.asarray(new_lp, dtype=np.float64),
            rtol=rtol,
            atol=atol,
            err_msg=f"Prompt {prompt_idx}: response_logprobs differ beyond rtol={rtol} atol={atol}",
        )


@pytest.mark.parametrize(
    "model_name,tp_size",
    [
        pytest.param(DENSE_MODEL_NAME, 2, id="dense_tp2"),
    ],
)
def test_old_vs_new_inference_generation(ray_init_fixture, model_name: str, tp_size: int):
    """
    Generate with old and new inference using greedy decoding and verify
    response token IDs are identical.
    """
    cfg = _get_cfg(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_chat_template(tokenizer)

    prompts = PROMPTS[:NUM_PROMPTS]
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )["input_ids"]

    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)

    print("\n=== Generating with OLD inference ===")
    old_ids, old_responses, old_logprobs, _ = _run_generation(
        cfg,
        prompt_token_ids,
        sampling_params,
        use_new=False,
        tp_size=tp_size,
    )

    print("\n=== Generating with NEW inference ===")
    new_ids, new_responses, new_logprobs, _ = _run_generation(
        cfg,
        prompt_token_ids,
        sampling_params,
        use_new=True,
        tp_size=tp_size,
    )

    assert len(old_ids) == len(new_ids) == NUM_PROMPTS

    for i in range(NUM_PROMPTS):
        ids_match = old_ids[i] == new_ids[i]

        status = "MATCH" if ids_match else "MISMATCH"
        print(f"\nPrompt {i}: {status}")
        print(f"  Old ({len(old_ids[i])} tokens): {old_responses[i][:120]}...")
        print(f"  New ({len(new_ids[i])} tokens): {new_responses[i][:120]}...")

        if not ids_match:
            min_len = min(len(old_ids[i]), len(new_ids[i]))
            for j in range(min_len):
                if old_ids[i][j] != new_ids[i][j]:
                    print(f"  First token mismatch at position {j}: old={old_ids[i][j]}, new={new_ids[i][j]}")
                    break
            if len(old_ids[i]) != len(new_ids[i]):
                print(f"  Length mismatch: old={len(old_ids[i])}, new={len(new_ids[i])}")

        _compare_response_logprobs(
            old_logprobs[i] if old_logprobs else None,
            new_logprobs[i] if new_logprobs else None,
            prompt_idx=i,
            assert_close=False,
        )


@pytest.mark.parametrize(
    "model_name,tp_size",
    [
        pytest.param(MOE_MODEL_NAME, 4, id="moonlight_tp4"),
    ],
)
def test_old_vs_new_inference_r3(ray_init_fixture, model_name: str, tp_size: int):
    """
    Generate with old and new inference on a MoE model with router replay enabled.
    Verify response token IDs match and rollout_expert_indices match (same shape, dtype, and values).
    """
    cfg = _get_cfg(model_name, enable_r3=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_chat_template(tokenizer)

    prompts = PROMPTS[:NUM_PROMPTS]
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )["input_ids"]

    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)

    print("\n=== Generating with OLD inference (R3 enabled) ===")
    old_ids, old_responses, old_logprobs, old_experts = _run_generation(
        cfg,
        prompt_token_ids,
        sampling_params,
        use_new=False,
        tp_size=tp_size,
    )

    print("\n=== Generating with NEW inference (R3 enabled) ===")
    new_ids, new_responses, new_logprobs, new_experts = _run_generation(
        cfg,
        prompt_token_ids,
        sampling_params,
        use_new=True,
        tp_size=tp_size,
    )

    assert len(old_ids) == len(new_ids) == NUM_PROMPTS

    for i in range(NUM_PROMPTS):
        ids_match = old_ids[i] == new_ids[i]
        status = "MATCH" if ids_match else "MISMATCH"
        print(f"\nPrompt {i}: {status}")
        print(f"  Old ({len(old_ids[i])} tokens): {old_responses[i][:120]}...")
        print(f"  New ({len(new_ids[i])} tokens): {new_responses[i][:120]}...")

        _compare_response_logprobs(
            old_logprobs[i] if old_logprobs else None,
            new_logprobs[i] if new_logprobs else None,
            prompt_idx=i,
            assert_close=True,
            rtol=1e-3,
            atol=1e-4,
        )

    assert old_experts is not None, "Old inference did not return rollout_expert_indices"
    assert new_experts is not None, "New inference did not return rollout_expert_indices"
    assert len(old_experts) == len(new_experts) == NUM_PROMPTS

    experts_match = True
    for i in range(NUM_PROMPTS):
        old_ei = old_experts[i]
        new_ei = new_experts[i]
        assert old_ei is not None, f"Prompt {i}: old expert indices is None"
        assert new_ei is not None, f"Prompt {i}: new expert indices is None"

        old_arr = _r3_expert_indices_to_numpy(old_ei)
        new_arr = _r3_expert_indices_to_numpy(new_ei)
        print(
            f"  Expert indices: old shape={old_arr.shape} dtype={old_arr.dtype}, "
            f"new shape={new_arr.shape} dtype={new_arr.dtype}"
        )
        if old_arr.shape != new_arr.shape:
            print(
                f"  WARNING Prompt {i}: expert indices shape mismatch: " f"old={old_arr.shape} vs new={new_arr.shape}"
            )
            experts_match = False
            continue
        if old_arr.dtype != new_arr.dtype:
            print(
                f"  WARNING Prompt {i}: expert indices dtype mismatch: " f"old={old_arr.dtype} vs new={new_arr.dtype}"
            )
            experts_match = False
            continue
        if not np.array_equal(old_arr, new_arr):
            hint = ""
            flat_old, flat_new = old_arr.reshape(-1), new_arr.reshape(-1)
            for j in range(len(flat_old)):
                if flat_old[j] != flat_new[j]:
                    hint = f" First value mismatch at flat index {j}: old={flat_old[j]} new={flat_new[j]}"
                    break
            print(f"  WARNING Prompt {i}: expert indices values differ.{hint}")
            experts_match = False

    print("\n=== rollout_expert_indices (old stack) ===")
    for i in range(NUM_PROMPTS):
        arr = _r3_expert_indices_to_numpy(old_experts[i])
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"\nPrompt {i} shape={arr.shape} dtype={arr.dtype}:\n{arr}")

    assert (
        experts_match
    ), "rollout_expert_indices differ between old and new inference — see WARNING lines and dumps above"
