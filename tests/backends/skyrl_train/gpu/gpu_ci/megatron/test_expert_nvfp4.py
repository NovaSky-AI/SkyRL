"""GPU tests for expert_nvfp4 on Megatron routed experts (Blackwell only; skipped elsewhere).

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_expert_nvfp4.py

The failure mode to guard against is silent: a TE or Megatron refactor that routes expert weights
around ``general_grouped_gemm`` leaves the hook installed but without effect. The model test
therefore checks the hook's effect on outputs and gradients against a reference model whose expert
weights are pre-rounded, bit for bit. The weight round trip must decode exactly the rollouts' bytes (FlashInfer
``fp4_quantize`` per expert with vLLM's global scale, and vLLM's own online quantizer for absmax). The activation
tests check the per-token round trip against FlashInfer's (the one vLLM's W4A4 MoE runs) and the model with rounded
expert inputs against a reference; each runs in a fresh process, since the hook is installed once per process.
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (
    BLOCK_SCALE_FOUR_OVER_SIX,
    BLOCK_SCALES,
)

NUM_EXPERTS = 4
NUM_LAYERS = 2


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _decode(q: torch.Tensor, sf: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 ``(e2m1 * sf) * scale`` of codes ``q (M, K/2)``, linear E4M3 scales ``sf (M, K/16)`` and per-row
    decode scales ``scale (M,)``."""
    lut = torch.tensor(_E2M1, device=q.device)
    q = q.view(torch.uint8)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).reshape(q.shape[0], -1)
    s = sf.view(torch.float8_e4m3fn).float().reshape(q.shape[0], -1)[:, : v.shape[1] // 16]
    return (v * s.repeat_interleave(16, dim=1)) * scale.float().reshape(-1, 1)


def _flashinfer_env(block_scale: str):
    import contextlib
    import os

    from skyrl.backends.skyrl_train.kernels.nvfp4_quant import flashinfer_nvfp4_env

    four_over_six_env = flashinfer_nvfp4_env(True)

    @contextlib.contextmanager
    def ctx():
        saved = {k: os.environ.pop(k, None) for k in four_over_six_env}
        if block_scale == "four_over_six":
            os.environ.update(four_over_six_env)
        try:
            yield
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    return ctx()


def _rollout_quantize(w: torch.Tensor, block_scale: str):
    """The rollouts' NVFP4 bytes of one ``(N, K)`` expert weight: vLLM's online global scale
    (``_quantize_moe_weight_to_nvfp4``) and FlashInfer ``fp4_quantize`` with that scalar scale (the weight sync and
    vLLM's online re-quantization). ``(codes, linear scales, weight_scale_2)``."""
    from flashinfer import fp4_quantize

    lo, hi = w.aminmax()
    amax = torch.maximum(lo.abs(), hi.abs()).float().clamp_min(1e-8).reshape(1)
    gs = (6.0 * 448.0) / amax
    with _flashinfer_env(block_scale):
        q, sf = fp4_quantize(w, gs, 16, False, False)
        torch.cuda.synchronize()
    return q.view(torch.uint8), sf.view(torch.uint8).reshape(w.shape[0], -1), (1.0 / gs).float()


def _rollout_grid(w: torch.Tensor, block_scale: str) -> torch.Tensor:
    """The rollouts' dequantized NVFP4 weight, bf16."""
    q, sf, alpha = _rollout_quantize(w, block_scale)
    return _decode(q, sf, alpha.expand(w.shape[0])).bfloat16()


def _test_weights() -> list[torch.Tensor]:
    torch.manual_seed(0)
    gaussian = torch.randn(512, 2048, device="cuda") * 0.02
    # Heavy-tailed rows with outliers, so blocks span many E4M3 scales and both 4/6 candidates win.
    heavy = torch.distributions.StudentT(2.0).sample((512, 2048)).cuda() * torch.logspace(-4, -1, 512).cuda()[:, None]
    return [gaussian.bfloat16(), heavy.bfloat16()]


def _build_model(seed: int, fuse_expert_wgrad: bool):
    from megatron.core import tensor_parallel
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer import TransformerConfig

    tensor_parallel.model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_grouped_gemm=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend="unfused",
        gradient_accumulation_fusion=False,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
    model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=256, max_sequence_length=64).cuda()
    if fuse_expert_wgrad:
        # TE accumulates the experts' wgrad straight into main_grad (gradient_accumulation_fusion for
        # TE modules); model-wide it would also need APEX's fused-wgrad extension for the output layer.
        from skyrl.backends.skyrl_train.workers.megatron.quantization import (
            expert_nvfp4,
        )

        for _, module in expert_nvfp4.routed_expert_grouped_linears([model]):
            module.fuse_wgrad_accumulation = True
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False, overlap_grad_reduce=False)
    return DistributedDataParallel(config=config, ddp_config=ddp_config, module=model)


def _pre_round(model_chunks, block_scale: str) -> None:
    """Round the routed-expert weights of ``model_chunks`` in place (the reference models)."""
    from skyrl.backends.skyrl_train.workers.megatron.quantization import (
        expert_nvfp4,
    )

    with torch.no_grad():
        for _, module in expert_nvfp4.routed_expert_grouped_linears(model_chunks):
            weights = module._get_weight_tensors()
            for w, y in zip(weights, expert_nvfp4.fake_nvfp4_quantize_weights(weights, block_scale)):
                w.copy_(y)


def _forward_backward(model, tokens):
    """Logits and every parameter's gradient (Megatron's DDP accumulates them into ``main_grad``:
    TE writes it directly for fused-wgrad experts, DDP's grad hooks add it otherwise)."""
    position_ids = torch.arange(tokens.shape[1], device="cuda")[None].expand_as(tokens)
    logits = model(input_ids=tokens, position_ids=position_ids, attention_mask=None)
    logits.float().square().mean().backward()
    grads = {name: p.main_grad.detach().clone() for name, p in model.named_parameters()}
    return logits.detach(), grads


@ray.remote(num_gpus=1)
def _run_checks():
    import os

    if torch.cuda.get_device_capability()[0] < 10:
        return None

    from megatron.core import parallel_state as mpu

    from skyrl.backends.skyrl_train.workers.megatron.quantization import (
        expert_nvfp4,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    results = {}

    # 1) The round trip (all experts in one call) against the rollouts' grid, per expert, bit for bit; experts held
    # apart (a stacked copy) and back to back in one buffer, highest first as Megatron's DDP lays them out (a view).
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        _quantize_moe_weight_to_nvfp4,
    )

    from skyrl.backends.skyrl_train.kernels.nvfp4_quant import fake_experts

    weights = _test_weights()
    buffer = torch.empty(len(weights) * weights[0].numel(), dtype=torch.bfloat16, device="cuda")
    in_buffer = [buffer[i * weights[0].numel() : (i + 1) * weights[0].numel()].view(weights[0].shape) for i in range(len(weights))][::-1]  # fmt: skip
    for w, b in zip(weights, in_buffer):
        b.copy_(w)
    for block_scale in BLOCK_SCALES:
        for layout, ws in (("apart", weights), ("buffer", in_buffer)):
            before = dict(fake_experts.STACK_STATS)
            fused = expert_nvfp4.fake_nvfp4_quantize_weights(ws, block_scale)
            results[f"stack_views/{layout}"] = fake_experts.STACK_STATS["views"] - before["views"]
            for i, (w, y) in enumerate(zip(ws, fused)):
                ref = _rollout_grid(w, block_scale)
                results[f"mismatch/{block_scale}/{layout}/{i}"] = int(
                    (y.view(torch.int16) != ref.view(torch.int16)).sum()
                )
                results[f"changed/{block_scale}/{layout}/{i}"] = (y != w).float().mean().item()
            packed = fake_experts.pack_expert_weights(ws, block_scale, keep_linear=True)
            results[f"pack_dequant_mismatch/{block_scale}/{layout}"] = sum(
                int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(packed.dequantize(), fused)
            )
    # absmax: the bytes vLLM's own online quantizer (scaled_fp4_quant per expert) produces.
    w3 = torch.stack(weights)
    vq, vsf, vs2 = _quantize_moe_weight_to_nvfp4(w3)
    gs, alpha = fake_experts.global_scales(w3)
    q, sf = fake_experts.quantize(w3, gs, "absmax")
    results["vllm_absmax_bytes_differ"] = int(
        not (torch.equal(q, vq.view(torch.uint8)) and torch.equal(sf, vsf.view(torch.uint8).view(sf.shape)))
    )
    results["vllm_alpha_differ"] = int(not torch.equal(alpha, vs2))
    print("fake-NVFP4 QAT quantizer:", results, flush=True)

    # 2) The hook on a real Megatron MoE model, with and without fused wgrad accumulation.
    expert_nvfp4.install_grouped_gemm_hook(BLOCK_SCALE_FOUR_OVER_SIX)
    tokens = torch.randint(0, 256, (2, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    for fused_wgrad in (False, True):
        raw = _build_model(seed=0, fuse_expert_wgrad=fused_wgrad)
        qat = _build_model(seed=0, fuse_expert_wgrad=fused_wgrad)
        ref = _build_model(seed=0, fuse_expert_wgrad=fused_wgrad)
        count = expert_nvfp4.register_routed_expert_weights([qat], torch.bfloat16)
        linears = expert_nvfp4.routed_expert_grouped_linears([ref])
        _pre_round([ref], BLOCK_SCALE_FOUR_OVER_SIX)
        logits_raw, _ = _forward_backward(raw, tokens)
        logits_qat, grads_qat = _forward_backward(qat, tokens)
        logits_ref, grads_ref = _forward_backward(ref, tokens)
        results[f"fused_wgrad={fused_wgrad}/count"] = count
        results[f"fused_wgrad={fused_wgrad}/linears"] = sorted(name for name, _ in linears)
        results[f"fused_wgrad={fused_wgrad}/effect"] = not torch.equal(logits_qat, logits_raw)
        results[f"fused_wgrad={fused_wgrad}/logits_exact"] = torch.equal(logits_qat, logits_ref)
        results[f"fused_wgrad={fused_wgrad}/grad_mismatches"] = sorted(
            n for n in grads_qat if not torch.equal(grads_qat[n], grads_ref[n])
        )

    torch.distributed.destroy_process_group()
    return results


@pytest.mark.megatron
def test_expert_nvfp4_on_blackwell(ray_init_fixture):
    results = ray.get(_run_checks.remote())
    if results is None:
        pytest.skip("NVFP4 needs Blackwell (SM100+)")
    print(results)

    for key, value in results.items():
        if key.startswith("changed/"):
            assert value > 0.5, f"{key}: the NVFP4 round trip left {1 - value:.1%} of weights unchanged"
        if key.startswith(("mismatch/", "pack_dequant_mismatch/")):
            assert value == 0, f"{key}: {value} weights differ from the rollouts' NVFP4 grid"
    assert results["stack_views/apart"] == 0 and results["stack_views/buffer"] == 1, results
    assert results["vllm_absmax_bytes_differ"] == 0 and results["vllm_alpha_differ"] == 0, results

    for fused_wgrad in (False, True):
        linears = results[f"fused_wgrad={fused_wgrad}/linears"]
        assert len(linears) == NUM_LAYERS * 2, f"routed-expert grouped linears found: {linears}"
        assert not any("shared_experts" in name for name in linears), linears
        assert results[f"fused_wgrad={fused_wgrad}/count"] == NUM_LAYERS * 2 * NUM_EXPERTS
        assert results[
            f"fused_wgrad={fused_wgrad}/effect"
        ], "QAT logits == raw logits: the grouped-GEMM hook is a silent no-op"
        assert results[
            f"fused_wgrad={fused_wgrad}/logits_exact"
        ], "QAT forward != the same model run on pre-rounded expert weights"
        assert not results[f"fused_wgrad={fused_wgrad}/grad_mismatches"], (
            "QAT gradients != the pre-rounded model's (dgrad must use the rounded weights, wgrad is "
            f"straight-through): {results[f'fused_wgrad={fused_wgrad}/grad_mismatches']}"
        )


def _flashinfer_per_token(x: torch.Tensor, block_scale: str) -> torch.Tensor:
    """vLLM's W4A4 MoE input quantization (``quantize_nvfp4_per_token_input``), dequantized in torch, under the
    FlashInfer env SkyRL exports to the engines for ``expert_nvfp4_block_scale`` (read per launch)."""
    from flashinfer import SfLayout, nvfp4_quantize

    with _flashinfer_env(block_scale):
        q, sf, scale = nvfp4_quantize(x, 1.0 / 2688.0, sfLayout=SfLayout.layout_linear, per_token_activation=True)
        torch.cuda.synchronize()
    return _decode(q, sf, scale.reshape(-1)).to(x.dtype)


_ACTIVATION_CASES = ("swiglu", "row_scales", "student_t", "zero_rows", "one_row")


def _test_activations() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    a, b = torch.randn(1024, 2048, device="cuda"), torch.randn(1024, 2048, device="cuda")
    rows = torch.randn(1024, 6144, device="cuda") * torch.logspace(-6, 3, 1024, device="cuda")[:, None]
    zeros = torch.randn(64, 256, device="cuda")
    zeros[3] = 0
    zeros[7, :48] = 0
    return {
        "swiglu": (torch.nn.functional.silu(a) * b).bfloat16(),
        "row_scales": rows.bfloat16(),
        "student_t": torch.distributions.StudentT(2.0).sample((1024, 2048)).cuda().bfloat16(),
        "zero_rows": zeros.bfloat16(),
        "one_row": torch.randn(1, 2048, device="cuda").bfloat16(),
    }


@ray.remote(num_gpus=1, max_calls=1)
def _run_activation_checks(port: int):
    import os

    if torch.cuda.get_device_capability()[0] < 10:
        return None

    from megatron.core import parallel_state as mpu

    from skyrl.backends.skyrl_train.workers.megatron.quantization import (
        expert_nvfp4,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    results = {}

    # 1) The per-token round trip against FlashInfer's bytes decoded in torch (bits, so also the sign of zero).
    for bs in BLOCK_SCALES:
        for name, x in _test_activations().items():
            ours = expert_nvfp4.fake_nvfp4_quantize_activations(x, bs)
            ref = _flashinfer_per_token(x, bs)
            results[f"vs_flashinfer/{bs}/{name}"] = int((ours.view(torch.int16) != ref.view(torch.int16)).sum())
            results[f"changed/{bs}/{name}"] = (ours != x).float().mean().item()
            results[f"inplace_differs/{bs}/{name}"] = not torch.equal(
                expert_nvfp4.fake_nvfp4_quantize_activations(x.clone(), bs, inplace=True), ours
            )
            results[f"rel_err/{bs}/{name}"] = ((ours.float() - x.float()).norm() / x.float().norm()).item()
    empty = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
    results["empty_shape"] = tuple(expert_nvfp4.fake_nvfp4_quantize_activations(empty).shape)

    # 2) The model: QAT with activations, and a reference with pre-rounded weights whose expert inputs are
    # rounded by the same straight-through hook.
    block_scale = BLOCK_SCALE_FOUR_OVER_SIX
    expert_nvfp4.install_grouped_gemm_hook(block_scale)
    tokens = torch.randint(0, 256, (2, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    qat = _build_model(seed=0, fuse_expert_wgrad=True)
    expert_nvfp4.register_routed_expert_weights([qat], torch.bfloat16)
    results["hooked"] = expert_nvfp4.register_routed_expert_input_rounding([qat])
    logits, grads = _forward_backward(qat, tokens)
    weights_only = _build_model(seed=0, fuse_expert_wgrad=True)
    _pre_round([weights_only], block_scale)
    logits_weights_only, _ = _forward_backward(weights_only, tokens)
    results["effect"] = not torch.equal(logits, logits_weights_only)
    ref = _build_model(seed=0, fuse_expert_wgrad=True)
    _pre_round([ref], block_scale)
    expert_nvfp4.register_routed_expert_input_rounding([ref])
    logits_ref, grads_ref = _forward_backward(ref, tokens)
    results["logits_exact"] = torch.equal(logits, logits_ref)
    results["grad_mismatches"] = sorted(n for n in grads if not torch.equal(grads[n], grads_ref[n]))

    torch.distributed.destroy_process_group()
    return results


@pytest.mark.megatron
def test_expert_nvfp4_activations_on_blackwell(ray_init_fixture):
    quantized = ray.get(_run_activation_checks.remote(29513))
    if quantized is None:
        pytest.skip("NVFP4 needs Blackwell (SM100+)")
    print(quantized)

    for key, value in quantized.items():
        if key.startswith("vs_flashinfer/"):
            assert value == 0, f"{key}: {value} elements differ from vLLM's per-token NVFP4 grid"
        if key.startswith("inplace_differs/"):
            assert not value, f"{key}: the in-place round trip differs from the out-of-place one"
        if key.startswith("changed/") and not key.endswith("zero_rows"):
            assert value > 0.5, f"{key}: the round trip left {1 - value:.1%} of activations unchanged"
    for name in _ACTIVATION_CASES:
        assert quantized[f"rel_err/four_over_six/{name}"] <= quantized[f"rel_err/absmax/{name}"] * 1.0001, name
    assert quantized["empty_shape"] == (0, 256)
    assert quantized["hooked"] == NUM_LAYERS * 2
    assert quantized["effect"], "rounding the expert inputs did not change the logits"
    assert quantized["logits_exact"], "QAT forward != the reference with pre-rounded weights and inputs"
    assert not quantized["grad_mismatches"], quantized["grad_mismatches"]


def _unswizzle(sf: torch.Tensor, rows: int, k_blocks: int) -> torch.Tensor:
    """128x4-swizzled E4M3 bytes -> (rows, k_blocks) fp32 scales."""
    lin = sf.reshape(rows // 128, k_blocks // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, k_blocks)
    return lin.view(torch.float8_e4m3fn).float()


def _unswizzle_bytes(sf: torch.Tensor, rows: int, k_blocks: int) -> torch.Tensor:
    """128x4-swizzled scale bytes -> (rows, k_blocks) bytes."""
    return sf.reshape(rows // 128, k_blocks // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, k_blocks)


def _dequant_packed(q: torch.Tensor, sf: torch.Tensor, scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """fp32 e2m1 * sf * scale (per row) of packed codes ``q (rows, k/2)`` with 128x4-swizzled scales."""
    return _decode(q, _unswizzle_bytes(sf, rows, k // 16), scale.reshape(rows))


@ray.remote(num_gpus=1, max_calls=1)
def _run_fp4_forward_checks(port: int):
    import os

    if torch.cuda.get_device_capability()[0] < 10:
        return None

    import transformer_engine.pytorch as te
    from megatron.core import parallel_state as mpu

    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fake_experts import (
        fake_quantize_expert_weights,
        pack_expert_weights,
    )
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fp4_grouped_gemm import (
        tile_layout,
    )
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.per_token import (
        flashinfer_per_token,
        pack_per_token,
    )
    from skyrl.backends.skyrl_train.workers.megatron.quantization import (
        expert_nvfp4,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    results = {}
    block_scale = BLOCK_SCALE_FOUR_OVER_SIX
    torch.manual_seed(0)

    # 1) Packed operands are the fake-quant round trips.
    weights = [(torch.randn(512, 256, device="cuda") * 0.02 * 3.0**e).bfloat16() for e in range(4)]
    wp = pack_expert_weights(weights, block_scale)
    fake_w = expert_nvfp4.fake_nvfp4_quantize_weights(weights, block_scale)
    results["packed_weights_mismatch"] = sum(
        (_dequant_packed(wp.q[e], wp.sf[e], wp.alpha[e].expand(512), 512, 256).bfloat16() != fake_w[e]).sum().item()
        for e in range(4)
    )
    # The packed weights' dequantization (linear scales) is the fake round trip, bit for bit
    # (heavy tails, zero blocks, a zero expert, both block-scale modes).
    heavy = torch.distributions.StudentT(2.0).sample((4, 256, 512)).cuda() * torch.logspace(-6, 0, 256).cuda()[:, None]
    heavy[1, :, :64] = 0
    heavy[2] = 0
    for bs in BLOCK_SCALES:
        for name, ws in (("gaussian", weights), ("heavy", [w.bfloat16() for w in heavy.unbind(0)])):
            ours = pack_expert_weights(ws, bs, keep_linear=True).dequantize()
            ref = fake_quantize_expert_weights(ws, bs)
            results[f"dequant_packed_mismatch/{bs}/{name}"] = sum(
                int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(ours, ref)
            )
    m_splits = [300, 0, 37, 700]
    for bs in BLOCK_SCALES:
        x = (
            torch.randn(sum(m_splits), 256, device="cuda")
            * torch.logspace(-2, 1, sum(m_splits), device="cuda")[:, None]
        )
        x = x.bfloat16()
        layout = tile_layout(m_splits, x.device)
        xr = x.clone()
        q, sf, pts = pack_per_token(xr, layout.dst, layout.padded_rows, bs == "four_over_six", write_back=True)
        fake_x = expert_nvfp4.fake_nvfp4_quantize_activations(x, bs)
        results[f"write_back_mismatch/{bs}"] = int((xr.view(torch.int16) != fake_x.view(torch.int16)).sum())
        deq = _dequant_packed(q, sf, pts, layout.padded_rows, 256)[layout.dst.long()]
        results[f"packed_act_mismatch/{bs}"] = (deq.bfloat16() != fake_x).sum().item()
        # The operand holds FlashInfer's bytes of each row at its padded row.
        fq, fsf, fpts = flashinfer_per_token(x, bs == "four_over_six")
        rows = layout.dst.long()
        results[f"fi_operand_codes_mismatch/{bs}"] = int((q[rows] != fq).sum())
        results[f"fi_operand_sf_mismatch/{bs}"] = int((_unswizzle_bytes(sf, layout.padded_rows, 16)[rows] != fsf).sum())
        results[f"fi_operand_pts_mismatch/{bs}"] = int((pts[rows] != fpts).sum())

    # 2) A TE GroupedLinear with fp4_forward: forward vs the fp32 product of the exact round trips,
    # backward vs bf16 GEMMs of the fake-quant operands (dgrad and wgrad are unchanged).
    expert_nvfp4.install_grouped_gemm_hook(block_scale, fp4_forward=True)
    layer = te.GroupedLinear(4, 256, 512, bias=False, params_dtype=torch.bfloat16, device="cuda")
    ws = layer._get_weight_tensors()
    for w in ws:
        expert_nvfp4._ROUTED_EXPERT_WEIGHTS[id(w)] = w
    # As register_routed_expert_input_rounding(..., pack_fp4=True) does for routed experts.
    layer.register_forward_pre_hook(expert_nvfp4.pack_input_pre_hook(block_scale))
    x = (torch.randn(sum(m_splits), 256, device="cuda") * 0.5).bfloat16()
    x_in = x.clone().requires_grad_()
    out = layer(x_in * 1, m_splits)
    dy = torch.randn_like(out)
    out.backward(dy)
    xq = expert_nvfp4.fake_nvfp4_quantize_activations(x, block_scale)
    wq_fake = expert_nvfp4.fake_nvfp4_quantize_weights(ws, block_scale)
    wp = pack_expert_weights(ws, block_scale)
    wq, wsf, alpha = wp.q, wp.sf, wp.alpha
    order = list(range(4))[::-1] if wp.reversed else list(range(4))
    layout = tile_layout(m_splits, x.device)
    q, sf, pts = pack_per_token(x.clone(), layout.dst, layout.padded_rows, True, write_back=False)
    x_exact = _dequant_packed(q, sf, pts, layout.padded_rows, 256)[layout.dst.long()]
    ref, ref_dx, ref_dw = [], [], []
    for e, xs, gs in zip(range(4), torch.split(x_exact, m_splits), torch.split(dy, m_splits)):
        w_exact = _dequant_packed(wq[order[e]], wsf[order[e]], alpha[order[e]].expand(512), 512, 256)
        ref.append((xs.double() @ w_exact.double().t()).float())
        ref_dx.append(gs.float() @ wq_fake[e].float())
        ref_dw.append(gs.float().t() @ torch.split(xq, m_splits)[e].float())
    ref = torch.cat(ref)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    results["fwd_rel_vs_exact"] = rel(out.detach(), ref)
    results["fwd_identical_vs_exact"] = (out.detach() == ref.bfloat16()).float().mean().item()
    results["fwd_rel_vs_fake"] = rel(
        out.detach(), torch.cat([a @ w.t() for a, w in zip(torch.split(xq, m_splits), wq_fake)])
    )
    results["dgrad_rel"] = rel(x_in.grad, torch.cat(ref_dx))

    results["wgrad_rel"] = max(rel(w.grad, r) for w, r in zip(ws, ref_dw) if r.abs().sum() > 0)

    # 3) Under a checkpoint (the trainer's full recompute), with an input TE cannot take as the rows of one
    # tensor (a slice of a larger buffer): the backward of the recompute must run (the input is rounded by
    # the pre-hook, before TE takes the views it saves) and match the plain input's, bit for bit.
    from megatron.core import tensor_parallel

    tensor_parallel.model_parallel_cuda_manual_seed(0)

    def run(sliced: bool):
        xs = x.clone().requires_grad_()
        for w in ws:
            w.grad = None

        def f(t):
            t = torch.cat([t * 1, t[:7] * 1])[: t.shape[0]] if sliced else t * 1
            return layer(t, m_splits)

        y = tensor_parallel.checkpoint(f, False, xs)
        y.backward(dy)
        return y.detach(), xs.grad, [w.grad.clone() for w in ws]

    try:
        plain, sliced = run(False), run(True)
        results["checkpoint_sliced_equal"] = (
            torch.equal(plain[0], sliced[0])
            and torch.equal(plain[1], sliced[1])
            and all(torch.equal(a, b) for a, b in zip(plain[2], sliced[2]))
        )
    except RuntimeError as e:
        results["checkpoint_sliced_equal"] = f"{type(e).__name__}: {e}"[:300]
    torch.distributed.destroy_process_group()
    return results


@pytest.mark.megatron
def test_expert_nvfp4_fp4_forward_on_blackwell(ray_init_fixture):
    results = ray.get(_run_fp4_forward_checks.remote(29515))
    if results is None:
        pytest.skip("NVFP4 tensor cores need Blackwell (SM100+)")
    print(results)
    assert results["packed_weights_mismatch"] == 0
    for key, value in results.items():
        if key.startswith("dequant_packed_mismatch/"):
            assert value == 0, f"{key}: {value} dequantized packed weights differ (bits) from the round trip"
    for bs in ("absmax", "four_over_six"):
        assert results[f"write_back_mismatch/{bs}"] == 0, bs
        assert results[f"packed_act_mismatch/{bs}"] == 0, bs
        for part in ("codes", "sf", "pts"):
            assert results[f"fi_operand_{part}_mismatch/{bs}"] == 0, (bs, part)
    # One bf16 rounding of the exact fp32 product; the fake path also rounds its operands to bf16.
    assert results["fwd_rel_vs_exact"] < 3e-3 and results["fwd_identical_vs_exact"] > 0.95, results
    assert results["fwd_rel_vs_fake"] < 1e-2, results
    assert results["dgrad_rel"] < 1e-2 and results["wgrad_rel"] < 1e-2, results
    assert results["checkpoint_sliced_equal"] is True, results["checkpoint_sliced_equal"]


def w4a4_alltoall_worker(rank: int, world: int, port: int, queue) -> None:
    """One rank of an EP=2 Megatron MoE model on the alltoall dispatcher (SkyRL's default) with
    ``install_expert_nvfp4(mode="w4a4")`` as the worker runs it, global layer 1 kept BF16, under full recompute.
    w4a4 rounds the per-token expert inputs and runs the fprops on NVFP4 tensor cores. Reference: a copy of
    the model with the NVFP4 layer's expert weights pre-rounded and its expert inputs rounded by a plain bf16 pre-hook
    (no grouped-GEMM hook), i.e. bf16 GEMMs of exactly the fake-quant operands."""
    import os

    import torch.nn.functional as F

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer import TransformerConfig

    from skyrl.backends.skyrl_train.workers.megatron.quantization import (
        expert_nvfp4,
    )

    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=world)
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    torch.manual_seed(0)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=4,
        moe_ffn_hidden_size=256,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend="unfused",
        gradient_accumulation_fusion=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)

    def build():
        return GPTModel(config=config, transformer_layer_spec=spec, vocab_size=256, max_sequence_length=256).cuda()

    model, ref, wref, raw = build(), build(), build(), build()
    # The same weights in all four (modules hold process groups, so no deepcopy).
    for other in (ref, wref, raw):
        other.load_state_dict(model.state_dict())
    bs = BLOCK_SCALE_FOUR_OVER_SIX
    results = {"registered": expert_nvfp4.install_expert_nvfp4([model], "w4a4", bs, bf16_layers=[1])}
    results["input_hooks"] = len(expert_nvfp4._INPUT_ROUNDING_MODULES)

    # Reference: layer 0's experts pre-rounded, their inputs rounded by a bf16 pre-hook; layer 1 untouched.
    def round_(x):
        expert_nvfp4.fake_nvfp4_quantize_activations(x, bs, inplace=True)

    # wref: the weights pre-rounded only (what w4a16 computes), to show the inputs are rounded too.
    for m, inputs_too in ((ref, True), (wref, False)):
        for name, module in expert_nvfp4.routed_expert_grouped_linears([m], bf16_layers=[1]):
            weights = module._get_weight_tensors()
            with torch.no_grad():
                for w, y in zip(weights, expert_nvfp4.fake_nvfp4_quantize_weights(weights, bs)):
                    w.copy_(y)
            if inputs_too:
                module.register_forward_pre_hook(expert_nvfp4.round_input_pre_hook(round_))
    results["ref_modules"] = len(expert_nvfp4.routed_expert_grouped_linears([ref], bf16_layers=[1]))

    g = torch.Generator(device="cuda").manual_seed(rank)
    tokens = torch.randint(0, 256, (2, 128), generator=g, device="cuda")
    position_ids = torch.arange(tokens.shape[1], device="cuda")[None].expand_as(tokens)

    def train_step(m):
        m.zero_grad(set_to_none=True)
        m.train()
        logits = m(input_ids=tokens, position_ids=position_ids, attention_mask=None)
        logits.float().square().mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    logits, grads = train_step(model)
    logits_ref, grads_ref = train_step(ref)
    logits_raw, _ = train_step(raw)
    logits_wref, _ = train_step(wref)
    results["operand_left_over"] = expert_nvfp4._fprop_operand is not None
    results["rel/logits_vs_ref"] = rel(logits, logits_ref)
    results["rel/logits_vs_raw"] = rel(logits, logits_raw)
    results["rel/logits_vs_weights_only"] = rel(logits, logits_wref)
    results["rel/grads_vs_ref_max"] = max(rel(grads[n], grads_ref[n]) for n in grads_ref)
    results["grads_finite"] = all(torch.isfinite(gr).all().item() for gr in grads.values())
    if rank == 0:
        queue.put(results)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@ray.remote(num_gpus=2, max_calls=1)
def _run_w4a4_alltoall_checks(port: int):
    if torch.cuda.get_device_capability()[0] != 10:
        return None
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.start_processes(w4a4_alltoall_worker, args=(2, port, queue), nprocs=2, start_method="spawn")
    return queue.get()


@pytest.mark.megatron
def test_expert_nvfp4_w4a4_on_alltoall_on_blackwell(ray_init_fixture):
    results = ray.get(_run_w4a4_alltoall_checks.remote(29571))
    if results is None:
        pytest.skip("NVFP4 needs SM100")
    print(results)
    # Layer 0 only (layer 1 is BF16): 4 local experts x (fc1, fc2) weights, both GEMM inputs hooked.
    assert results["registered"] == 8 and results["input_hooks"] == 2 and results["ref_modules"] == 2, results
    assert not results["operand_left_over"], results
    assert results["grads_finite"], results
    # The NVFP4 fprop accumulates the exact products in fp32; the reference rounds the dequantized operands' GEMM
    # to bf16: close, and much closer than the unrounded model or the weights-only one (the inputs are rounded).
    assert results["rel/logits_vs_ref"] < 1e-2, results
    assert results["rel/logits_vs_raw"] > 3 * results["rel/logits_vs_ref"], results
    assert results["rel/logits_vs_weights_only"] > 2 * results["rel/logits_vs_ref"], results
    assert results["rel/grads_vs_ref_max"] < 0.1, results


def bf16_layers_pp_worker(rank: int, world: int, port: int, queue) -> None:
    """PP=4 with a custom ``pipeline_model_parallel_layout``: each stage registers exactly its NVFP4 layers' experts,
    by the global ``TransformerLayer.layer_number`` (the HF layer the vLLM ignore list and the weight sync use)."""
    import os

    import torch.nn.functional as F

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from skyrl.backends.skyrl_train.workers.megatron.quantization import expert_nvfp4

    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=world)
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_size=world,
        pipeline_model_parallel_layout="Et|tt|t|ttL",
        attention_backend="unfused",
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=256,
        max_sequence_length=128,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    ).cuda()
    bf16_layers = [0, 5]
    nvfp4_moe = {id(m) for _, m in expert_nvfp4.nvfp4_modules([model], MoELayer, bf16_layers)}
    nvfp4_layers = sorted(layer.layer_number - 1 for layer in model.decoder.layers if id(layer.mlp) in nvfp4_moe)
    all_layers = sorted(m.layer_number - 1 for m in model.decoder.layers)
    count = expert_nvfp4.register_routed_expert_weights([model], torch.bfloat16, bf16_layers)
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, (all_layers, nvfp4_layers, count))
    if rank == 0:
        queue.put(gathered)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@ray.remote(num_gpus=4, max_calls=1)
def _run_bf16_layers_pp_checks(port: int):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.start_processes(bf16_layers_pp_worker, args=(4, port, queue), nprocs=4, start_method="spawn")
    return queue.get()


@pytest.mark.megatron
def test_bf16_layers_follow_the_global_layer_number_under_pp(ray_init_fixture):
    stages = ray.get(_run_bf16_layers_pp_checks.remote(29581))
    print(stages)
    assert [s[0] for s in stages] == [[0], [1, 2], [3], [4, 5]], stages
    assert [s[1] for s in stages] == [[], [1, 2], [3], [4]], stages
    # 4 experts x (fc1, fc2) per NVFP4 layer.
    assert [s[2] for s in stages] == [0, 16, 8, 8], stages
