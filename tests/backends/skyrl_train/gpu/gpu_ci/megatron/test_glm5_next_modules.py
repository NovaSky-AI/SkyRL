"""GLM-5.3-Flash Megatron building blocks vs. the HF ``glm5_next`` reference implementation.

Single-GPU checks of the two modules SkyRL adds on top of megatron-core for GLM-5.3-Flash:

- ``HyperConnectionModule`` (mHC, backport of Megatron-LM main) against
  ``Glm5NextTextHyperConnection`` -- mapping outputs and the n-stream residual update;
- ``KimiDeltaAttention`` (KDA) against ``Glm5NextTextLinearAttention`` on packed sequences.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_glm5_next_modules.py
"""

import os

import pytest
import ray
import torch

pytestmark = pytest.mark.megatron


def _init_single_rank_megatron():
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)


@ray.remote(num_gpus=1)
def _hyper_connection_parity():
    from megatron.core.transformer import TransformerConfig
    from transformers import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextHyperConnection,
    )

    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
    from skyrl.backends.skyrl_train.workers.megatron.mcore_ext.hyper_connection import (
        HyperConnectionModule,
    )

    _init_single_rank_megatron()
    torch.manual_seed(0)
    hidden, n, seq, batch = 64, 4, 5, 2

    # GLM normalizes the flattened streams with a standard RMSNorm using rms_norm_eps (1e-5); the
    # streams below are scaled like the real model's embeddings (per-token rms ~ 1e-2), where the
    # epsilon placement is material.
    hf_cfg = Glm5NextTextConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        hc_mult=n,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    hf = Glm5NextTextHyperConnection(hf_cfg).cuda()
    with torch.no_grad():
        hf.fn.normal_(0, 0.02)
        hf.base.normal_(0, 0.5)
        hf.scale.uniform_(0.5, 1.5)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        add_bias_linear=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
    )
    for name, value in (
        ("enable_mhc_connections", True),
        ("mhc_num_residual_streams", n),
        ("mhc_sinkhorn_iterations", 20),
        ("mhc_init_gating_factor", 0.01),
        ("use_fused_mhc", False),
        ("mhc_fused_backend", "auto"),
        ("mhc_norm_eps", 1e-5),
        ("mhc_norm_eps_inside_sqrt", True),
    ):
        setattr(cfg, name, value)
    with torch.device("cuda"):
        mod = HyperConnectionModule(cfg, layer_number=1)
    with torch.no_grad():
        mod.mapping_proj.weight.copy_(hf.fn)
        mod.bias.copy_(hf.base)
        mod.alpha_pre.copy_(hf.scale[0:1])
        mod.alpha_post.copy_(hf.scale[1:2])
        mod.alpha_res.copy_(hf.scale[2:3])

    streams = 0.01 * torch.randn(seq, batch, n, hidden, device="cuda")  # [s, b, n, C]
    update = 0.01 * torch.randn(seq, batch, hidden, device="cuda")  # single-stream sub-layer output
    with torch.no_grad():
        post, comb, collapsed = hf(streams.permute(1, 0, 2, 3).contiguous())  # HF: [B, S, H, D]
        hf_next = post.unsqueeze(-1) * update.permute(1, 0, 2).unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), streams.permute(1, 0, 2, 3)
        )
        aggregated, h_res, h_post = mod(streams.view(seq, batch, n * hidden))
        meg_next = mod.fused_h_res_h_post_bda(
            h_res, streams.view(seq, batch, n * hidden), h_post, (update, None), 0.0, False, False
        ).view(seq, batch, n, hidden)

    def max_diff(a, b):
        return (a - b).abs().max().item()

    return {
        "aggregated": max_diff(aggregated, collapsed.permute(1, 0, 2)),
        "h_post": max_diff(h_post, post.permute(1, 0, 2)),
        "h_res": max_diff(h_res, comb.permute(1, 0, 2, 3)),
        "next_streams": max_diff(meg_next, hf_next.permute(1, 0, 2, 3)),
        "expand": max_diff(
            HyperConnectionModule.input_expand(update, n).view(seq, batch, n, hidden),
            update.unsqueeze(2).expand(-1, -1, n, -1),
        ),
        "contract": max_diff(
            HyperConnectionModule.output_contract(streams.view(seq, batch, n * hidden), n), streams.mean(2)
        ),
    }


@ray.remote(num_gpus=1)
def _kda_parity():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import build_module
    from transformers import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextLinearAttention,
    )

    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
    from skyrl.backends.skyrl_train.workers.megatron.glm5_next.layer_specs import (
        get_kda_module_spec,
    )

    _init_single_rank_megatron()
    torch.manual_seed(0)
    hidden, heads, head_dim, kernel = 128, 4, 32, 4
    dtype = torch.bfloat16

    hf_cfg = Glm5NextTextConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        indexer_types=["full"],
        linear_attn_config={
            "num_heads": heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": kernel,
            "gate_lower_bound": -5.0,
            "kda_layers": [0],
            "full_attn_layers": [],
        },
        rms_norm_eps=1e-5,
        hidden_act="silu",
    )
    hf = Glm5NextTextLinearAttention(hf_cfg, layer_idx=0).cuda()
    with torch.no_grad():
        for proj in (
            hf.q_proj,
            hf.k_proj,
            hf.v_proj,
            hf.o_proj,
            hf.b_proj,
            hf.g_a_proj,
            hf.g_b_proj,
            hf.forget_gate.f_a_proj,
            hf.forget_gate.f_b_proj,
        ):
            proj.weight.normal_(0, 0.05)
        hf.conv1d.weight.normal_(0, 0.3)
        hf.forget_gate.A_log.copy_(torch.empty(heads).uniform_(1, 16).log())
        hf.forget_gate.dt_bias.uniform_(-3.0, -1.0)
        hf.o_norm.weight.uniform_(0.8, 1.2)
    # The real model keeps projections/conv/norm in bf16 and A_log/dt_bias in fp32.
    hf = hf.to(dtype)
    hf.forget_gate.A_log.data = hf.forget_gate.A_log.data.float()
    hf.forget_gate.dt_bias.data = hf.forget_gate.dt_bias.data.float()

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        num_query_groups=4,
        linear_num_value_heads=heads,
        linear_num_key_heads=heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=kernel,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        bf16=True,
        params_dtype=dtype,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
    )
    cfg.kda_gate_lower_bound = -5.0
    mod = build_module(
        get_kda_module_spec(TESpecProvider()),
        config=cfg,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    )
    qkv_dim = heads * head_dim
    with torch.no_grad():
        for meg_name, hf_param in (
            ("q_proj", hf.q_proj.weight),
            ("k_proj", hf.k_proj.weight),
            ("v_proj", hf.v_proj.weight),
            ("f_a_proj", hf.forget_gate.f_a_proj.weight),
            ("f_b_proj", hf.forget_gate.f_b_proj.weight),
            ("g_a_proj", hf.g_a_proj.weight),
            ("g_b_proj", hf.g_b_proj.weight),
            ("b_proj", hf.b_proj.weight),
            ("o_proj", hf.o_proj.weight),
        ):
            getattr(mod, meg_name).weight.copy_(hf_param)
        # HF fuses the q/k/v convolutions into one depthwise conv over [q | k | v] channels.
        q_w, k_w, v_w = torch.split(hf.conv1d.weight, [qkv_dim] * 3, dim=0)
        mod.q_conv1d.weight.copy_(q_w)
        mod.k_conv1d.weight.copy_(k_w)
        mod.v_conv1d.weight.copy_(v_w)
        mod.A_log.copy_(hf.forget_gate.A_log)
        mod.dt_bias.copy_(hf.forget_gate.dt_bias)
        mod.o_norm.weight.copy_(hf.o_norm.weight)

    lengths = [37, 50]
    xs = [torch.randn(1, length, hidden, device="cuda", dtype=dtype) for length in lengths]
    with torch.no_grad():
        hf_out = torch.cat([hf(x) for x in xs], dim=1)[0].float()  # [t, C]
        packed = torch.cat(xs, dim=1).transpose(0, 1).contiguous()  # [t, 1, C]
        cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        meg_out, bias = mod(packed, packed_seq_params=params)
    assert bias is None
    meg_out = meg_out[:, 0].float()
    diff = (meg_out - hf_out).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "ref_mean_abs": hf_out.abs().mean().item(),
    }


def test_hyper_connection_matches_hf(ray_init_fixture):
    """The backported mHC module reproduces the HF glm5_next hyper-connection in fp32."""
    diffs = ray.get(_hyper_connection_parity.remote())
    print(f"mHC max abs diffs vs HF: {diffs}")
    for name, value in diffs.items():
        assert value < 1e-5, f"{name}: {value}"


def test_kda_matches_hf(ray_init_fixture):
    """KDA on packed sequences reproduces per-sequence HF Glm5NextTextLinearAttention (bf16)."""
    stats = ray.get(_kda_parity.remote())
    print(f"KDA vs HF: {stats}")
    assert stats["mean_abs_diff"] < 0.02 * max(stats["ref_mean_abs"], 1e-3), stats
    assert stats["max_abs_diff"] < 0.2 * max(stats["ref_mean_abs"], 1e-3) + 1e-2, stats
