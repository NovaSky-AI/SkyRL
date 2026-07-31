from types import SimpleNamespace

import pytest
import torch

import skyrl.backends.skyrl_train.quantization.nvfp4 as nvfp4_module
from skyrl.backends.skyrl_train.quantization import (
    SERIALIZED_WEIGHT_PREFIX,
    get_quantized_model_layout,
    iter_serialized_weight_tensors,
)
from skyrl.backends.skyrl_train.quantization.nvfp4 import (
    NVFP4_4OVER6_FP8_MAX,
    NVFP4_BLOCK_SIZE,
    Nvfp4ExpertStrategy,
)
from skyrl.backends.skyrl_train.quantization.vllm_nvfp4 import (
    _materialize_expanded_parameter,
    install_vllm_nvfp4_per_token_patch,
)


@pytest.fixture(autouse=True)
def _disable_modelopt_fp4_backend_probe(monkeypatch):
    import modelopt.torch.quantization.qtensor.nvfp4_tensor as nvfp4_tensor

    monkeypatch.setattr(nvfp4_tensor, "fp4_compatible", lambda: False)


def _dequantize(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    return NVFP4QTensor(shape, torch.bfloat16, packed).dequantize(
        scale=block_scale,
        double_scale=global_scale,
        block_sizes={-1: NVFP4_BLOCK_SIZE},
        dtype=torch.float32,
    )


def _reference_nvfp4_four_over_six_scales(weight: torch.Tensor):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    weight_fp32 = weight.float()
    reduce_dims = (-2, -1) if weight.ndim == 3 else None
    global_scale = (
        weight_fp32.abs().amax(dim=reduce_dims, keepdim=weight.ndim == 3) / (6.0 * NVFP4_4OVER6_FP8_MAX)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    blocks = weight_fp32.view(*weight.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)

    def candidate(multiplier: float):
        block_scale = (block_amax * multiplier / (6.0 * global_scale)).clamp(
            min=2**-9,
            max=torch.finfo(torch.float8_e4m3fn).max,
        )
        block_scale = block_scale.to(torch.float8_e4m3fn)
        dequant_scale = block_scale.float() * global_scale
        q_values = NVFP4QTensor._cast_fp4(blocks / dequant_scale.unsqueeze(-1))
        fp4_values = NVFP4QTensor.get_e2m1_values(weight.device)[q_values.long()].float()
        error = ((blocks - fp4_values * dequant_scale.unsqueeze(-1)) ** 2).sum(dim=-1)
        return block_scale, error

    scale_m6, error_m6 = candidate(1.0)
    scale_m4, error_m4 = candidate(1.5)
    block_scale = torch.where(error_m4 < error_m6, scale_m4.float(), scale_m6.float())
    return block_scale.to(torch.float8_e4m3fn), global_scale


def _reference_nvfp4_cast(weight: torch.Tensor):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    block_scale, global_scale = _reference_nvfp4_four_over_six_scales(weight)
    quantized, block_scale, global_scale = NVFP4QTensor.quantize(
        weight.contiguous(),
        NVFP4_BLOCK_SIZE,
        weights_scaling_factor=block_scale,
        weights_scaling_factor_2=global_scale,
    )
    global_scale = global_scale.reshape(weight.shape[0]) if weight.ndim == 3 else global_scale.reshape(())
    return quantized._quantized_data.contiguous(), block_scale.contiguous(), global_scale.contiguous()


def test_nvfp4_four_over_six_selects_no_worse_mse_than_m6():
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    torch.manual_seed(0)
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    global_scale = weight.abs().amax().float() / (6.0 * NVFP4_4OVER6_FP8_MAX)
    m6, m6_scale, _ = NVFP4QTensor.quantize(
        weight,
        NVFP4_BLOCK_SIZE,
        weights_scaling_factor_2=global_scale,
    )
    q_weight, block_scale, selected_global_scale = _reference_nvfp4_cast(weight)

    m6_dequant = _dequantize(m6._quantized_data, m6_scale, global_scale, weight.shape)
    selected_dequant = _dequantize(q_weight, block_scale, selected_global_scale, weight.shape)

    assert torch.mean((weight.float() - selected_dequant) ** 2) <= torch.mean(
        (weight.float() - m6_dequant) ** 2
    )
    assert torch.any(block_scale.view(torch.uint8) != m6_scale.view(torch.uint8))


def test_batched_nvfp4_four_over_six_uses_per_expert_global_scales():
    torch.manual_seed(1)
    weight = torch.randn(3, 16, 64, dtype=torch.bfloat16)
    weight[1].mul_(4)
    weight[2].mul_(0.25)

    packed, block_scale, global_scale = _reference_nvfp4_cast(weight)

    expected_global_scale = weight.abs().amax(dim=(-2, -1)).float() / (6.0 * NVFP4_4OVER6_FP8_MAX)
    assert packed.shape == (3, 16, 32)
    assert packed.dtype == torch.uint8
    assert block_scale.shape == (3, 16, 4)
    assert block_scale.dtype == torch.float8_e4m3fn
    assert torch.allclose(global_scale, expected_global_scale)
    assert torch.isfinite(_dequantize(packed, block_scale, global_scale[:, None, None], weight.shape)).all()


def test_nvfp4_humans_recipe_runtime_env():
    strategy = Nvfp4ExpertStrategy(
        backward_override="dequantized",
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        four_over_six_scope="all",
        four_over_six_e4m3_use_256_scope="all",
        four_over_six_error_mode="MSE",
        four_over_six_error_use_fast_math=True,
        disable_fp4_quant_fast_math=True,
    )

    env = strategy.build_runtime_env()

    assert env["NVTE_BACKWARD_OVERRIDE"] == "dequantized"
    assert env["NVTE_NVFP4_DISABLE_RHT"] == "1"
    assert env["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] == "1"
    assert env["NVTE_NVFP4_DISABLE_2D_QUANTIZATION"] == "1"
    assert env["NVTE_NVFP4_ROW_SCALED_ACTIVATION"] == "1"
    assert env["NVTE_NVFP4_4OVER6"] == "all"
    assert env["NVTE_NVFP4_4OVER6_E4M3_USE_256"] == "all"
    assert env["NVTE_NVFP4_4OVER6_ERR_MODE"] == "MSE"
    assert env["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] == "1"
    assert env["SKYRL_VLLM_NVFP4_PER_TOKEN_ACTIVATION"] == "1"
    assert env["FLASHINFER_NVFP4_4OVER6"] == "1"
    assert env["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] == "1"
    assert env["FLASHINFER_NVFP4_4OVER6_ERR_MODE"] == "MSE"
    assert env["FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH"] == "1"
    assert env["FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"] == "1"
    assert env["TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"] == "1"


def test_nvfp4_gate_and_up_weights_share_global_scales(monkeypatch):
    torch.manual_seed(2)
    gate_up = torch.randn(3, 32, 64, dtype=torch.bfloat16)
    gate_up[:, :16].mul_(4)

    def fake_quantize_pair(gate, up, **_kwargs):
        global_scale = torch.maximum(
            gate.float().abs().amax(dim=(-2, -1)),
            up.float().abs().amax(dim=(-2, -1)),
        )

        def payload(weight):
            return (
                torch.zeros((*weight.shape[:-1], weight.shape[-1] // 2), dtype=torch.uint8),
                torch.ones(
                    (*weight.shape[:-1], weight.shape[-1] // NVFP4_BLOCK_SIZE),
                    dtype=torch.float8_e4m3fn,
                ),
                global_scale.clone(),
            )

        return payload(gate), payload(up)

    monkeypatch.setattr(nvfp4_module, "_quantize_nvfp4_pair", fake_quantize_pair)
    strategy = Nvfp4ExpertStrategy(four_over_six_scope="weights", row_scaled_activation=True)

    emitted = dict(
        iter_serialized_weight_tensors(
            "model.layers.0.mlp.experts.gate_up_proj",
            gate_up,
            torch.bfloat16,
            get_quantized_model_layout("qwen3_moe"),
            strategy,
            model_type="qwen3_moe",
        )
    )

    prefix = f"{SERIALIZED_WEIGHT_PREFIX}{strategy.mode}:model.layers.0.mlp.experts"
    assert torch.equal(
        emitted[f"{prefix}.gate_proj.weight_scale_2"],
        emitted[f"{prefix}.up_proj.weight_scale_2"],
    )
    assert torch.equal(
        emitted[f"{prefix}.gate_proj.input_scale"],
        torch.ones(3, dtype=torch.float32),
    )


def test_nvfp4_high_precision_tail_is_excluded_from_training_and_rollout():
    strategy = Nvfp4ExpertStrategy(high_precision_last_layers=2, num_layers=4)

    assert strategy.should_quantize(
        "routed_expert_gate",
        "model.layers.1.mlp.experts.gate_proj.weight",
        (8, 16),
    )
    assert not strategy.should_quantize(
        "routed_expert_gate",
        "model.layers.2.mlp.experts.gate_proj.weight",
        (8, 16),
    )

    quantization_config = strategy.vllm_quantization_config(
        None,
        SimpleNamespace(num_hidden_layers=4),
        get_quantized_model_layout("qwen3_moe"),
    )
    assert "*.layers.2.mlp.experts*" in quantization_config["ignore"]
    assert "*.layers.3.mlp.experts*" in quantization_config["ignore"]

    emitted = dict(
        iter_serialized_weight_tensors(
            "model.layers.2.mlp.experts.gate_up_proj",
            torch.randn(3, 32, 64, dtype=torch.bfloat16),
            torch.bfloat16,
            get_quantized_model_layout("qwen3_moe"),
            strategy,
            model_type="qwen3_moe",
        )
    )
    prefix = f"{SERIALIZED_WEIGHT_PREFIX}{strategy.mode}:model.layers.2.mlp.experts"
    assert emitted[f"{prefix}.gate_proj.weight"].dtype == torch.bfloat16
    assert emitted[f"{prefix}.up_proj.weight"].dtype == torch.bfloat16
    assert all("weight_scale" not in name for name in emitted)


def test_vllm_nvfp4_patch_accepts_unpacked_ep_workspace(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
        TrtLlmNvFp4ExpertsMonolithic,
    )

    monkeypatch.setenv("SKYRL_VLLM_NVFP4_PER_TOKEN_ACTIVATION", "1")
    install_vllm_nvfp4_per_token_patch()
    experts = object.__new__(TrtLlmNvFp4ExpertsModular)
    experts.per_token_activation = True
    experts.hidden_dim = 64

    assert experts.expects_unquantized_inputs
    assert experts.workspace_shapes(2, 0, 64, 2, 8, 1, None, None) == ((0,), (0,), (2, 64))
    parallel_config = SimpleNamespace(use_all2all_kernels=False, enable_eplb=False)
    assert not TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(parallel_config)


def test_vllm_nvfp4_patch_materializes_expanded_activation_scales():
    layer = torch.nn.Module()
    input_scale = torch.nn.Parameter(torch.ones(1).expand(3), requires_grad=False)
    layer.register_parameter("w13_input_scale", input_scale)

    assert input_scale.stride() == (0,)
    _materialize_expanded_parameter(layer, "w13_input_scale")
    input_scale.data.copy_(torch.tensor([1.0, 2.0, 3.0]))

    assert input_scale.stride() == (1,)
    assert torch.equal(input_scale, torch.tensor([1.0, 2.0, 3.0]))
