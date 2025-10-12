import jax
import numpy as np

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import types


class FutureStub:
    """Minimal stub with request_id (engine only reads this attribute)."""

    def __init__(self, request_id: int):
        self.request_id = request_id


def make_fwd_bwd_input(token_lists: list[list[int]]):
    samples = []
    for tokens in token_lists:
        targets = tokens[1:] + [0]
        weights = [1] * len(tokens)
        samples.append(
            {
                "model_input": {"chunks": [{"tokens": tokens}]},
                "loss_fn_inputs": {
                    "target_tokens": {"data": targets},
                    "weights": {"data": weights},
                },
            }
        )
    payload = {"forward_backward_input": {"data": samples}}
    return types.ForwardBackwardInput.model_validate(payload)


def test_adapter_gradient_calculation():
    config = EngineConfig(
        base_model="Qwen/Qwen3-0.6B",
        checkpoints_base_path="",
        max_lora_adapters=8,
        max_lora_rank=32,
    )
    engine = TinkerEngine(config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    # Create two LoRA adapters
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Adapter1 samples (fixed across both rounds)
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    # Adapter2 samples (round 1: 2 samples; round 2: 4 samples)
    a2_input1 = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    reqs_round1 = [
        (FutureStub(101), adapter1_id, a1_input),
        (FutureStub(102), adapter2_id, a2_input1),
    ]

    # Process round 1 batch
    engine.process_forward_backward_batch(reqs_round1)

    grads_A1_round1 = jax.tree.map(lambda x: x.copy(), engine.accumulated_grads[adapter1_id])

    # Clear stored grads so we can run another fwd/bwd without optimizer update.
    engine.accumulated_grads[adapter1_id] = None
    engine.accumulated_grads[adapter2_id] = None

    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    a2_input2 = make_fwd_bwd_input([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    reqs_round2 = [
        (FutureStub(201), adapter1_id, a1_input),
        (FutureStub(202), adapter2_id, a2_input2),
    ]

    # Process round 2 batch
    engine.process_forward_backward_batch(reqs_round2)

    grads_A1_round2 = jax.tree.map(lambda x: x.copy(), engine.accumulated_grads[adapter1_id])

    def _assert_mostly_close(a, b, rtol=1e-3, atol=1e-3, min_match_pct=99.0):
        a_arr = np.array(a)
        b_arr = np.array(b)

        # Check how many elements are close
        matches = np.isclose(a_arr, b_arr, rtol=rtol, atol=atol)
        match_pct = 100.0 * np.sum(matches) / a_arr.size
        if match_pct < min_match_pct:

            # Show statistics about mismatches
            diff = np.abs(a_arr - b_arr)
            rel_diff = np.abs((a_arr - b_arr) / (np.abs(b_arr) + 1e-10))
            failing = ~matches
            raise AssertionError(
                f"Only {match_pct}% of elements match (required: {min_match_pct}%)\n"
                f"  Max absolute diff: {np.max(diff[failing])}\n"
                f"  Max relative diff: {np.max(rel_diff[failing])}\n"
                f"  Mean of mismatches: {np.mean(diff[failing])}"
            )

    jax.tree.map(
        lambda a, b: _assert_mostly_close(a, b, rtol=1e-3, atol=1e-2, min_match_pct=99.0),
        grads_A1_round1,
        grads_A1_round2,
    )
