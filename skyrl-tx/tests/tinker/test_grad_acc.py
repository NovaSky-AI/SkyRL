import os
import jax
import jax.numpy as jnp
import numpy as np

from tx.tinker.engine import TinkerEngine
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


def _mean_grads_from_acc(acc_slot: dict):
    """Convert accumulator (sum, denom) -> mean grads tree."""
    assert acc_slot["grad_sum"] is not None and acc_slot["denominator"] > 0
    denom = acc_slot["denominator"]
    return jax.tree.map(lambda g: g / jnp.asarray(denom, dtype=g.dtype), acc_slot["grad_sum"])


def _assert_tree_allclose(t1, t2, rtol=1e-3, atol=1e-3):
    leaves1 = jax.tree.leaves(t1)
    leaves2 = jax.tree.leaves(t2)
    assert len(leaves1) == len(leaves2), "Gradient trees differ in structure/leaf count"
    for a, b in zip(leaves1, leaves2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def _assert_outputs_close(res_a: dict, res_b: dict, rtol=1e-4, atol=1e-4):
    assert set(res_a.keys()) == set(res_b.keys()), "Request IDs differ between runs"
    for rid in res_a.keys():
        oa = res_a[rid]
        ob = res_b[rid]
        la = oa.loss_fn_outputs
        lb = ob.loss_fn_outputs
        assert len(la) == len(lb), f"Sample count mismatch in request {rid}"
        for sa, sb in zip(la, lb):
            ea = np.asarray(sa["elementwise_loss"]["data"], dtype=np.float32)
            eb = np.asarray(sb["elementwise_loss"]["data"], dtype=np.float32)
            np.testing.assert_allclose(ea, eb, rtol=rtol, atol=atol)

            pa = np.asarray(sa["logprobs"]["data"], dtype=np.float32)
            pb = np.asarray(sb["logprobs"]["data"], dtype=np.float32)
            np.testing.assert_allclose(pa, pb, rtol=rtol, atol=atol)


def test_micro_batching_equivalence_gradients_and_outputs():
    """
    Verifies that micro-batching (e.g., TX_MICRO_BATCH_SIZE=4) produces:
      - the same per-adapter MEAN gradients as a fused run (no micro-batching),
      - identical per-token outputs (losses/logprobs) within tolerance,
      - correct denominators per adapter.
    """
    # Build engine and two adapters.
    engine = TinkerEngine(
        base_model_name="Qwen/Qwen3-0.6B",
        checkpoints_base_path="",
        max_lora_adapters=8,
        max_lora_rank=32,
    )

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Fused batch with 6 total examples: 2 for adapter1, 4 for adapter2.
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2 samples
    a2_input = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )  # 4 samples

    reqs = [
        (FutureStub(1001), adapter1_id, a1_input),
        (FutureStub(1002), adapter2_id, a2_input),
    ]

    # --- Run 1: micro-batching enabled (choose a non-divisible last chunk: 6 -> 4 + 2)
    prev_env = os.environ.get("TX_MICRO_BATCH_SIZE")
    os.environ["TX_MICRO_BATCH_SIZE"] = "4"

    res_micro = engine.process_forward_backward_batch(reqs)
    acc_micro_a1 = engine.accumulated_grads[adapter1_id]
    acc_micro_a2 = engine.accumulated_grads[adapter2_id]
    mean_micro_a1 = _mean_grads_from_acc(acc_micro_a1)
    mean_micro_a2 = _mean_grads_from_acc(acc_micro_a2)

    # Sanity on denominators with micro-batching
    assert acc_micro_a1["denominator"] == 2
    assert acc_micro_a2["denominator"] == 4

    # Reset accumulators (no optimizer step)
    engine.accumulated_grads[adapter1_id] = {"grad_sum": None, "denominator": 0}
    engine.accumulated_grads[adapter2_id] = {"grad_sum": None, "denominator": 0}

    # --- Run 2: fused (no micro-batching; env<=0 -> full batch as one micro)
    os.environ["TX_MICRO_BATCH_SIZE"] = "0"

    res_full = engine.process_forward_backward_batch(reqs)
    acc_full_a1 = engine.accumulated_grads[adapter1_id]
    acc_full_a2 = engine.accumulated_grads[adapter2_id]
    mean_full_a1 = _mean_grads_from_acc(acc_full_a1)
    mean_full_a2 = _mean_grads_from_acc(acc_full_a2)

    # Denominators should be identical in fused run
    assert acc_full_a1["denominator"] == 2
    assert acc_full_a2["denominator"] == 4

    # Compare MEAN gradients (should match within tolerance)
    _assert_tree_allclose(mean_micro_a1, mean_full_a1, rtol=1e-3, atol=1e-3)
    _assert_tree_allclose(mean_micro_a2, mean_full_a2, rtol=1e-3, atol=1e-3)

    # Compare per-token outputs (losses/logprobs) across runs
    _assert_outputs_close(res_micro, res_full, rtol=1e-4, atol=1e-4)

    # Cleanup env
    if prev_env is None:
        os.environ.pop("TX_MICRO_BATCH_SIZE", None)
    else:
        os.environ["TX_MICRO_BATCH_SIZE"] = prev_env
