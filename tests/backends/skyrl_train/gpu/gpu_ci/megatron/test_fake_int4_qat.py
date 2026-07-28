"""GPU integration tests for the fake-INT4 QAT TEGroupedLinear hook.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_fake_int4_qat.py

These protect the feature against megatron-core / TransformerEngine version bumps.
The dangerous failure mode is SILENT: if a TE refactor stops routing
``_get_weight_tensors()`` into the grouped GEMM (or starts caching weights), the
hook keeps "installing" fine but no longer affects compute, and the train/infer
gap quietly returns. The canary test therefore asserts the hook's *effect on the
GEMM output*, not just its installation.
"""

import inspect

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.workers.megatron import fake_int4_qat as fq_mod
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import MegatronWorker
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

MOE_MODEL_NAME = "Qwen/Qwen3-30B-A3B"
GS = 32


# ---------------------------------------------------------------------------
# Tripwires: fail loudly (and readably) when megatron/TE refactor the surfaces
# the monkeypatch depends on.
# ---------------------------------------------------------------------------


@pytest.mark.megatron
def test_te_grouped_linear_still_exposes_get_weight_tensors():
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    assert hasattr(TEGroupedLinear, "_get_weight_tensors"), (
        "TEGroupedLinear._get_weight_tensors is gone: the fake-INT4 QAT monkeypatch "
        "(skyrl.backends.skyrl_train.workers.megatron.fake_int4_qat) no longer has an attach point. "
        "Find where the new TE version fetches weights in GroupedLinear.forward and re-target the patch."
    )
    fwd_src = inspect.getsource(te_grouped_linear.GroupedLinear.forward)
    assert "_get_weight_tensors" in fwd_src, (
        "transformer_engine GroupedLinear.forward no longer calls _get_weight_tensors(): "
        "the fake-INT4 QAT patch would install but silently stop affecting the expert GEMMs. "
        "Re-target the patch to the new weight-fetch path before bumping TE."
    )


@pytest.mark.megatron
def test_maybe_setup_fake_int4_qat_wiring():
    """The worker helper: BF16 redirect is independent of the STE install."""

    class _Stub:
        _rank = 0
        cfg = SkyRLTrainConfig()

    stub = _Stub()
    fq_cfg = stub.cfg.trainer.policy.model.fake_int4_qat
    # SkyRLTrainConfig nests trainer.*; the worker reads self.cfg.policy (=cfg.trainer.policy)
    stub.cfg = stub.cfg.trainer

    # disabled + no redirect
    assert MegatronWorker._maybe_setup_fake_int4_qat(stub) is None
    assert not fq_mod._installed

    # disabled + redirect: path returned, STE NOT installed (INT4-served baseline)
    fq_cfg.bf16_base_path = "/data/bf16-masters"
    assert MegatronWorker._maybe_setup_fake_int4_qat(stub) == "/data/bf16-masters"
    assert not fq_mod._installed

    # enabled: STE installed once, path still returned
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    orig = TEGroupedLinear._get_weight_tensors
    try:
        fq_cfg.enabled = True
        assert MegatronWorker._maybe_setup_fake_int4_qat(stub) == "/data/bf16-masters"
        assert fq_mod._installed
        assert TEGroupedLinear._get_weight_tensors is not orig
    finally:
        TEGroupedLinear._get_weight_tensors = orig
        fq_mod._installed = False


# ---------------------------------------------------------------------------
# Canary: the patched weights must actually flow into the grouped GEMM.
# Runs in a ray task so the process-global patch and torch.distributed state
# cannot leak into other tests.
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _run_hook_effect_canary():
    import os

    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEGroupedLinear,
    )
    from megatron.core.transformer import TransformerConfig

    from skyrl.backends.skyrl_train.workers.megatron.fake_int4_qat import (
        fake_int4_quantize_ste,
        install_fake_int4_qat,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    torch.cuda.set_device(0)

    num_gemms, in_f, out_f, tokens = 4, 128, 64, 32
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=in_f,
        num_attention_heads=4,
        num_moe_experts=num_gemms,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
    )

    def make_module(seed):
        torch.manual_seed(seed)
        return TEColumnParallelGroupedLinear(
            num_gemms=num_gemms,
            input_size=in_f,
            output_size=out_f,
            config=cfg,
            init_method=cfg.init_method,
            bias=False,
            skip_bias_add=True,
            is_expert=True,
        ).cuda()

    module = make_module(0)
    torch.manual_seed(1)
    x = torch.randn(tokens, in_f, dtype=torch.bfloat16, device="cuda")
    m_splits = [tokens // num_gemms] * num_gemms
    masters = [getattr(module, f"weight{i}").data.clone() for i in range(num_gemms)]

    with torch.no_grad():
        out_unpatched, _ = module(x, m_splits)

        # reference: the same UNpatched GEMM, with weights replaced by their fake-quant
        ref_module = make_module(0)
        for i in range(num_gemms):
            getattr(ref_module, f"weight{i}").data.copy_(fake_int4_quantize_ste(masters[i], GS, 7.5, -8.0))
        out_reference, _ = ref_module(x, m_splits)

        orig = TEGroupedLinear._get_weight_tensors
        try:
            install_fake_int4_qat(group_size=GS, scale_divisor=7.5, q_min=-8.0)
            out_patched, _ = module(x, m_splits)

            # 1) the hook has an effect at all (catches silent no-op)
            effect = not torch.equal(out_patched, out_unpatched)
            # 2) and the effect is exactly quantization: bitwise-equal to the
            #    same TE kernel run on pre-quantized weights
            exact = torch.equal(out_patched, out_reference)

            # 3) weights are re-read every forward (catches future TE caching):
            #    mutate a master in place and expect the output to track it
            getattr(module, f"weight{0}").data.mul_(2.0)
            out_mutated, _ = module(x, m_splits)
            tracks_mutation = not torch.equal(out_mutated, out_patched)
        finally:
            TEGroupedLinear._get_weight_tensors = orig

    torch.distributed.destroy_process_group()
    return effect, exact, tracks_mutation


@pytest.mark.megatron
def test_fake_int4_hook_affects_grouped_gemm(ray_init_fixture):
    effect, exact, tracks_mutation = ray.get(_run_hook_effect_canary.remote())
    assert effect, "patched forward == unpatched forward: the fake-quant hook is a silent no-op"
    assert exact, (
        "patched forward != TE GEMM on pre-quantized weights: the hook is quantizing "
        "something other than what the GEMM consumes (weight-fetch path changed?)"
    )
    assert tracks_mutation, (
        "output ignored an in-place master-weight update: TE appears to cache weights "
        "across forwards, so QAT would train against stale quantized experts"
    )


# ---------------------------------------------------------------------------
# Worker-level end-to-end: enabling the flag must change the policy logprobs
# through SkyRL's real init path on a (truncated) MoE model.
# ---------------------------------------------------------------------------


def _moe_forward_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MOE_MODEL_NAME
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.logger = "console"
    cfg.trainer.gradient_checkpointing_use_reentrant = True
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False  # forward-only test, no inference engine
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.ref_num_gpus_per_node = 4
    mcfg = cfg.trainer.policy.megatron_config
    mcfg.tensor_model_parallel_size = 4
    mcfg.expert_model_parallel_size = 4
    mcfg.expert_tensor_parallel_size = 1
    if mcfg.transformer_config_kwargs is None:
        mcfg.transformer_config_kwargs = dict()
    mcfg.transformer_config_kwargs["num_layers"] = 2  # truncate the 30B MoE for CI
    validate_cfg(cfg)
    return cfg


async def _run_moe_forward(qat_enabled: bool):
    from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
        get_test_training_batch,
    )

    cfg = _moe_forward_cfg()
    cfg.trainer.policy.model.fake_int4_qat.enabled = qat_enabled
    batch = get_test_training_batch(4)
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )
    refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
    output = WorkerOutput.cat(actor_group.actor_infos, ray.get(refs))
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_megatron_forward_qat_on_off(ray_init_fixture):
    """QAT must change expert compute end-to-end, and only moderately (STE, not corruption)."""
    try:
        logprobs_off = await _run_moe_forward(qat_enabled=False)
        ray.shutdown()
        ray_init_for_tests()
        logprobs_on = await _run_moe_forward(qat_enabled=True)
    finally:
        ray.shutdown()

    assert logprobs_off.shape == logprobs_on.shape
    assert not torch.equal(logprobs_off, logprobs_on), (
        "logprobs identical with fake-INT4 QAT on/off: the hook did not reach the expert GEMMs "
        "through the worker init path (megatron module structure or install wiring changed?)"
    )
    mean_abs = (logprobs_on - logprobs_off).abs().mean().item()
    assert mean_abs < 1.0, f"QAT moved logprobs implausibly far (mean |diff| = {mean_abs:.3f})"
