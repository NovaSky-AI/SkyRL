"""Multi-LoRA tests for the *colocated* Megatron path (``colocate_all=True``).

Sibling of ``test_multi_lora_megatron.py``, which pins
``trainer.placement.colocate_all=False``. Colocation is the SkyRL-Train
default (and what the Tinker API assumes when the operator passes no
override), and it changes the memory lifecycle underneath every adapter
swap:

  * ``WorkerDispatch._prepare_for_weight_sync`` offloads the optimizer and
    the DDP grad buffers before syncing weights to vLLM.
  * ``forward``/``forward_from_staged`` only backload the *model*, so the
    optimizer + grad buffers stay offloaded across a pure forward.
  * Megatron's ``offload_grad_buffers`` frees ``grad_data`` outright
    (``storage().resize_(0)``) and ``restore_grad_buffers`` zero-fills it.

So on this path a swap runs against partially-offloaded state, and any
grads the live adapter accumulated but has not yet consumed in
``optim_step`` are destroyed by the offload. Both are exercised below.

Run with:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_multi_lora_megatron_colocated.py
"""

from __future__ import annotations

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="multi-LoRA Megatron tests require at least one CUDA GPU")

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.skyrl_train.test_multi_lora_megatron import (  # noqa: E402
    BASE_MODEL,
    TINKER_API_KEY,
    _api_server,
    _make_datum,
    _sample_greedy,
)

TEST_PORT = 8021

# Same tiny config as the non-colocated module, but colocated: the policy
# worker and the single vLLM engine share one GPU, so the dispatch offloads
# policy state to CPU around every sample.
BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": True,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.model.lora.max_loras": 4,
    "trainer.policy.model.lora.max_cpu_loras": 4,
    "generator.inference_engine.num_engines": 1,
    "generator.inference_engine.tensor_parallel_size": 1,
    "generator.inference_engine.gpu_memory_utilization": 0.3,
}


@pytest.fixture(scope="module")
def server():
    with _api_server(TEST_PORT, BACKEND_CONFIG) as proc:
        yield proc


@pytest.fixture
def service_client(server):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)


def test_sample_non_live_adapter_colocated(service_client):
    """Minimal repro: sampling an adapter that isn't the live one.

    After ``create_lora_training_client`` for A then B, A is the live
    adapter on the workers and (under colocation) the optimizer + grad
    buffers are offloaded. ``save_weights_for_sampler(B)`` then swaps
    A -> B while ``grad_data`` has been freed, so the AdapterStore's
    snapshot copies from a zero-sized CUDA storage:

        cpu_grad_data[...].copy_(buf.grad_data)
        torch.AcceleratorError: CUDA error: invalid argument

    No training required — the offload after ``init_model`` is enough.
    """
    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = a.get_tokenizer()

    tokens = _sample_greedy(b, name="colo_b0", tok=tok, prompt="Question: 2+2?\nAnswer:")
    assert len(tokens) > 0


def test_pending_grads_survive_other_tenant_sample(service_client):
    """A's un-consumed grads must survive another tenant's sample.

    The Tinker request stream interleaves tenants at request granularity, so
    ``A.forward_backward`` / ``A.optim_step`` can straddle a ``B.sample``.
    Under colocation that sample offloads the grad buffers, and Megatron's
    grad offload *drops* the grads (reload zero-fills). If they aren't parked
    in A's adapter slot, A's ``optim_step`` applies an all-zero gradient.

    Measured against an undisturbed control adapter C that runs the same
    single step from pristine on the same data: A's post-step loss has to
    land on C's. Asserting only ``post < pre`` on A is not enough — Adam
    still applies weight decay to a zero gradient, which drifts the loss by
    ~1e-6 and would satisfy a bare inequality while the step itself was a
    no-op (the real step moves it by ~1e-2).
    """
    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    c = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = a.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    lr = tinker_types.AdamParams(learning_rate=1e-2)

    def _loss(out):
        return sum(sum(o["elementwise_loss"].data) for o in out.loss_fn_outputs)

    # A accumulates grads but does not step yet.
    pre_a = _loss(a.forward_backward(data, "cross_entropy").result())

    # B samples in the gap: offloads optimizer + grad buffers, swaps A -> B.
    _sample_greedy(b, name="colo_pending_b", tok=tok, prompt="Question: 2+2?\nAnswer:")

    # A steps on the grads it accumulated before B intervened.
    a.optim_step(lr).result()
    post_a = _loss(a.forward_backward(data, "cross_entropy").result())

    # C: same step, no sample in the gap.
    pre_c = _loss(c.forward_backward(data, "cross_entropy").result())
    c.optim_step(lr).result()
    post_c = _loss(c.forward_backward(data, "cross_entropy").result())

    improvement_c = pre_c - post_c
    print(
        f"\n[pending_grads] A: pre={pre_a!r} post={post_a!r} Δ={post_a - pre_a:.6e}\n"
        f"[pending_grads] C: pre={pre_c!r} post={post_c!r} Δ={post_c - pre_c:.6e}"
    )

    assert pre_a == pre_c, f"A and C were not equally pristine: {pre_a!r} vs {pre_c!r}"
    assert improvement_c > 0, f"control adapter C did not learn (pre={pre_c!r}, post={post_c!r}); test is inconclusive"
    # Tolerance scales with the control's own step, so this stays meaningful
    # whatever the tiny model happens to do.
    assert abs(post_a - post_c) <= 0.05 * improvement_c, (
        f"A's step diverged from the undisturbed control: A post={post_a!r}, C post={post_c!r} "
        f"(C improved by {improvement_c:.6e}). A's pending grads were dropped by the grad-buffer "
        "offload that B's sample triggered."
    )


def test_two_adapters_train_and_sample_colocated(service_client):
    """Colocated analog of ``test_two_adapters_sample_independently``.

    Full interleaving of training and sampling across two tenants: every
    sample round-trips through offload -> swap -> broadcast -> offload, and
    every training request backloads and swaps back.
    """
    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = a.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
    sample_prompt = "Question: 2+2?\nAnswer:"

    for _ in range(2):
        a.forward_backward(data, "cross_entropy").result()
        a.optim_step(tinker_types.AdamParams(learning_rate=1e-2)).result()
    tokens_a_first = _sample_greedy(a, name="colo_a1", tok=tok, prompt=sample_prompt)

    b.forward_backward(data, "cross_entropy").result()
    b.optim_step(tinker_types.AdamParams(learning_rate=5e-2)).result()
    tokens_b = _sample_greedy(b, name="colo_b1", tok=tok, prompt=sample_prompt)

    a.forward_backward(data, "cross_entropy").result()
    a.optim_step(tinker_types.AdamParams(learning_rate=1e-2)).result()
    tokens_a_continued = _sample_greedy(a, name="colo_a2", tok=tok, prompt=sample_prompt)
    print(
        f"\n[colocated_independently] tokens_a_first={tokens_a_first}\n"
        f"[colocated_independently] tokens_b      ={tokens_b}\n"
        f"[colocated_independently] tokens_a_cont ={tokens_a_continued}"
    )

    assert tokens_a_continued != tokens_a_first, (
        f"A's tokens did not change after one more optim_step (A={tokens_a_continued!r}, "
        f"prior={tokens_a_first!r}). A's optimizer state may have been wiped by B."
    )
    assert tokens_a_continued != tokens_b, (
        f"A's continued sample matches B's sample: A={tokens_a_continued!r}, B={tokens_b!r}. "
        "B's adapter sync may have clobbered A's slot on vLLM."
    )
