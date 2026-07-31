"""NVFP4 expert training and serialized weight encoding."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, Literal

import torch

from .base import QuantizationStrategy, QuantizedModelLayout, WeightCategory

SERIALIZED_NVFP4 = "serialized_nvfp4"
NVFP4_BLOCK_SIZE = 16
NVFP4_GLOBAL_SCALE_DENOMINATOR = 6.0 * 448.0
NVFP4_4OVER6_FP8_MAX = 256.0
NVFP4_TE_ROW_ALIGNMENT = 16
Nvfp4FourOverSixScope = Literal["none", "weights", "activations", "all"]
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")

EXPERT_ONLY_NVFP4_IGNORED_MODULES = (
    "*.self_attn.*",
    "*.linear_attn.*",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.down_proj",
    "*.mlp.shared_expert*",
    "*lm_head*",
    "*.visual.*",
    "mtp.*",
)


def build_nvfp4_te_recipe(*, persistent: bool) -> dict:
    """Build Transformer Engine settings for NVFP4 modules."""

    recipe = {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {
            "fp4_quantization_recipe": "nvfp4",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
        },
        "evaluation_recipe": {
            "fp4_quantization_recipe": "nvfp4",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
        },
    }
    if persistent:
        recipe["training_recipe"]["fp4_param"] = True
        recipe["evaluation_recipe"]["fp4_param"] = True
    return recipe


def configure_nvfp4_provider(provider) -> None:
    """Set Megatron provider fields required by expert NVFP4."""

    if getattr(provider, "quant_recipe", None) is not None:
        raise ValueError("Expert NVFP4 conflicts with an existing quant_recipe")
    if getattr(provider, "fp8", None) not in (None, False):
        raise ValueError("Expert NVFP4 conflicts with global FP8 configuration")
    if getattr(provider, "fp4", None) not in (None, False):
        raise ValueError("Expert NVFP4 conflicts with global FP4 configuration")
    if not provider.moe_grouped_gemm:
        raise ValueError("Expert NVFP4 requires moe_grouped_gemm=true")
    if provider.moe_router_dtype != "fp32":
        raise ValueError("Expert NVFP4 requires moe_router_dtype=fp32")

    provider.fp4 = "e2m1"
    provider.fp4_recipe = "nvfp4"


def validate_nvfp4_hardware() -> None:
    """Require a Blackwell GPU with native NVFP4 support."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"Expert NVFP4 requires SM100 or SM103, got SM{capability[0]}{capability[1]}")


def get_serialized_nvfp4_quantization_config(
    *,
    per_token_activation: bool = False,
    additional_ignored_modules: Sequence[str] = (),
) -> dict:
    """Return ModelOpt NVFP4 settings for expert-only loading."""

    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4" if per_token_activation else "W4A16_NVFP4",
        "group_size": NVFP4_BLOCK_SIZE,
        "ignore": [*EXPERT_ONLY_NVFP4_IGNORED_MODULES, *additional_ignored_modules],
    }


def nvfp4_scale_names_for_weight(name: str) -> tuple[str, str]:
    """Return checkpoint names for NVFP4 block and global scales."""

    if not name.endswith(".weight"):
        raise ValueError(f"NVFP4 scales can only be derived from .weight tensors: {name}")
    prefix = name[: -len(".weight")]
    return f"{prefix}.weight_scale", f"{prefix}.weight_scale_2"


def _nvfp4_four_over_six_scales(
    weight: torch.Tensor,
    nvfp4_qtensor,
    *,
    batched_experts: bool,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the lower-MSE M=6 or M=4 scale for every 16-value block."""

    weight_fp32 = weight.float()
    if global_scale is None:
        if batched_experts:
            global_amax = weight_fp32.abs().amax(dim=(-2, -1), keepdim=True)
        else:
            global_amax = weight_fp32.abs().amax()
        global_scale = (global_amax / (6.0 * NVFP4_4OVER6_FP8_MAX)).clamp_min(
            torch.finfo(torch.float32).tiny
        )

    blocks = weight_fp32.view(*weight.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)

    def candidate(multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
        block_scale = (
            block_amax * multiplier / (6.0 * global_scale)
        ).clamp(min=2**-9, max=torch.finfo(torch.float8_e4m3fn).max)
        block_scale = block_scale.to(torch.float8_e4m3fn)
        dequant_scale = block_scale.float() * global_scale
        q_values = nvfp4_qtensor._cast_fp4(blocks / dequant_scale.unsqueeze(-1))
        fp4_values = nvfp4_qtensor.get_e2m1_values(weight.device)[q_values.long()].float()
        error = ((blocks - fp4_values * dequant_scale.unsqueeze(-1)) ** 2).sum(dim=-1)
        return block_scale, error

    scale_m6, error_m6 = candidate(1.0)
    scale_m4, error_m4 = candidate(1.5)
    selected_scale = torch.where(error_m4 < error_m6, scale_m4.float(), scale_m6.float())
    return selected_scale.to(torch.float8_e4m3fn), global_scale


def _nvfp4_global_decode_scale(global_amax: torch.Tensor, e4m3_max: int) -> torch.Tensor:
    """Match Transformer Engine's NVFP4 global decode-scale contract."""

    encode_scale = torch.div(float(e4m3_max) * 6.0, global_amax.float())
    encode_scale = torch.minimum(
        encode_scale,
        torch.tensor(torch.finfo(torch.float32).max, device=global_amax.device),
    )
    encode_scale = torch.where(encode_scale == 0, torch.ones_like(encode_scale), encode_scale)
    return torch.reciprocal(encode_scale)


def _pad_nvfp4_rows(weight: torch.Tensor) -> torch.Tensor:
    """Pad rows to Transformer Engine's rowwise quantizer alignment."""

    pad_rows = (-weight.shape[0]) % NVFP4_TE_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


def _nvfp4_te_quantize_2d(
    weight: torch.Tensor,
    *,
    four_over_six: bool,
    e4m3_max: int,
    error_mode: Literal["MAE", "MSE"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one matrix with Transformer Engine's production NVFP4 kernel."""

    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    rows, cols = weight.shape
    if cols % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_BLOCK_SIZE}, got {cols}")
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=four_over_six,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=error_mode,
        with_random_sign_mask=False,
    )
    quantized = quantizer.quantize(_pad_nvfp4_rows(weight.contiguous()))
    packed = quantized._rowwise_data[:rows, : cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[:rows, : cols // NVFP4_BLOCK_SIZE].contiguous()
    global_amax = quantized._amax_rowwise.reshape(-1)[0]
    return (
        packed,
        block_scale.view(torch.float8_e4m3fn),
        _nvfp4_global_decode_scale(global_amax, e4m3_max),
    )


def _nvfp4_te_quantize(
    weight: torch.Tensor,
    *,
    four_over_six: bool,
    e4m3_max: int,
    error_mode: Literal["MAE", "MSE"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight or independent 3D expert matrices with TE."""

    if weight.ndim == 2:
        return _nvfp4_te_quantize_2d(
            weight,
            four_over_six=four_over_six,
            e4m3_max=e4m3_max,
            error_mode=error_mode,
        )
    if weight.ndim != 3:
        raise ValueError(f"NVFP4 expects a 2D or 3D tensor, got shape={tuple(weight.shape)}")
    outputs = [
        _nvfp4_te_quantize_2d(
            expert,
            four_over_six=four_over_six,
            e4m3_max=e4m3_max,
            error_mode=error_mode,
        )
        for expert in weight.unbind(0)
    ]
    packed, block_scale, global_scale = zip(*outputs)
    return torch.stack(packed), torch.stack(block_scale), torch.stack(global_scale)


def _nvfp4_te_quantize_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    four_over_six: bool,
    e4m3_max: int,
    error_mode: Literal["MAE", "MSE"],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Quantize gate/up together so each expert shares one global scale."""

    if first.ndim != second.ndim or first.shape[:-2] != second.shape[:-2] or first.shape[-1] != second.shape[-1]:
        raise ValueError(
            "NVFP4 gate/up pair requires matching batch and input dimensions, "
            f"got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.ndim == 2:
        first_rows = first.shape[0]
        packed, block_scale, global_scale = _nvfp4_te_quantize_2d(
            torch.cat((first, second), dim=0),
            four_over_six=four_over_six,
            e4m3_max=e4m3_max,
            error_mode=error_mode,
        )
        return (
            (packed[:first_rows].contiguous(), block_scale[:first_rows].contiguous(), global_scale.clone()),
            (packed[first_rows:].contiguous(), block_scale[first_rows:].contiguous(), global_scale.clone()),
        )
    if first.ndim != 3:
        raise ValueError(f"NVFP4 gate/up pair expects 2D or 3D weights, got ndim={first.ndim}")
    pairs = [
        _nvfp4_te_quantize_pair(
            gate,
            up,
            four_over_six=four_over_six,
            e4m3_max=e4m3_max,
            error_mode=error_mode,
        )
        for gate, up in zip(first.unbind(0), second.unbind(0))
    ]
    first_outputs, second_outputs = zip(*pairs)

    def stack(outputs):
        packed, block_scale, global_scale = zip(*outputs)
        return torch.stack(packed), torch.stack(block_scale), torch.stack(global_scale)

    return stack(first_outputs), stack(second_outputs)


def _nvfp4_cast(
    weight: torch.Tensor,
    *,
    batched_experts: bool,
    four_over_six: bool,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight to packed E2M1 data with E4M3 and FP32 scales."""

    expected_ndim = 3 if batched_experts else 2
    if weight.ndim != expected_ndim:
        raise ValueError(
            f"NVFP4 expects a {expected_ndim}D tensor when batched_experts={batched_experts}, "
            f"got shape={tuple(weight.shape)}"
        )
    if weight.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires the last dimension to be divisible by {NVFP4_BLOCK_SIZE}, "
            f"got shape={tuple(weight.shape)}"
        )

    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    except ImportError as exc:
        raise RuntimeError("Serialized NVFP4 requires nvidia-modelopt") from exc

    weight = weight.detach().contiguous()
    block_scale = None
    if four_over_six:
        block_scale, global_scale = _nvfp4_four_over_six_scales(
            weight,
            NVFP4QTensor,
            batched_experts=batched_experts,
            global_scale=global_scale,
        )
    elif global_scale is None and batched_experts:
        global_scale = (
            weight.abs().amax(dim=(-2, -1), keepdim=True).float() / NVFP4_GLOBAL_SCALE_DENOMINATOR
        )
    elif global_scale is None:
        global_scale = None

    quantized, block_scale, global_scale = NVFP4QTensor.quantize(
        weight,
        NVFP4_BLOCK_SIZE,
        weights_scaling_factor=block_scale,
        weights_scaling_factor_2=global_scale,
    )
    packed_weight = quantized._quantized_data
    if batched_experts:
        global_scale = global_scale.reshape(weight.shape[0])
    else:
        global_scale = global_scale.reshape(())
    return packed_weight.contiguous(), block_scale.contiguous(), global_scale.contiguous()


def nvfp4_cast_to_fp4(
    weight: torch.Tensor,
    *,
    four_over_six: bool = False,
    e4m3_max: int = int(NVFP4_4OVER6_FP8_MAX),
    error_mode: Literal["MAE", "MSE"] = "MSE",
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to serialized NVFP4 tensors."""

    if weight.is_cuda and global_scale is None:
        return _nvfp4_te_quantize(
            weight,
            four_over_six=four_over_six,
            e4m3_max=e4m3_max if four_over_six else 448,
            error_mode=error_mode,
        )
    return _nvfp4_cast(
        weight,
        batched_experts=False,
        four_over_six=four_over_six,
        global_scale=global_scale,
    )


def batched_nvfp4_cast_to_fp4(
    weight: torch.Tensor,
    *,
    four_over_six: bool = False,
    e4m3_max: int = int(NVFP4_4OVER6_FP8_MAX),
    error_mode: Literal["MAE", "MSE"] = "MSE",
    expert_batch_size: int = 8,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert weight to serialized NVFP4 tensors."""

    if weight.ndim != 3:
        raise ValueError(f"Batched NVFP4 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")
    if weight.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires the last dimension to be divisible by {NVFP4_BLOCK_SIZE}, "
            f"got shape={tuple(weight.shape)}"
        )
    if weight.is_cuda and global_scale is None:
        return _nvfp4_te_quantize(
            weight,
            four_over_six=four_over_six,
            e4m3_max=e4m3_max if four_over_six else 448,
            error_mode=error_mode,
        )

    num_experts, rows, cols = weight.shape
    packed_weight = torch.empty((num_experts, rows, cols // 2), dtype=torch.uint8, device=weight.device)
    block_scale = torch.empty(
        (num_experts, rows, cols // NVFP4_BLOCK_SIZE),
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    output_global_scale = torch.empty(num_experts, dtype=torch.float32, device=weight.device)

    for start in range(0, num_experts, expert_batch_size):
        end = min(start + expert_batch_size, num_experts)
        q_chunk, block_chunk, global_chunk = _nvfp4_cast(
            weight[start:end],
            batched_experts=True,
            four_over_six=four_over_six,
            global_scale=None if global_scale is None else global_scale[start:end],
        )
        packed_weight[start:end].copy_(q_chunk)
        block_scale[start:end].copy_(block_chunk)
        output_global_scale[start:end].copy_(global_chunk)
    return packed_weight, block_scale, output_global_scale


@dataclass(frozen=True)
class Nvfp4ExpertStrategy(QuantizationStrategy):
    """Apply NVFP4 to routed expert weights."""

    backward_override: Literal["high_precision", "dequantized"] | None = None
    disable_rht: bool = False
    disable_stochastic_rounding: bool = False
    disable_2d_quantization: bool = False
    row_scaled_activation: bool = False
    four_over_six_scope: Nvfp4FourOverSixScope = "none"
    four_over_six_e4m3_use_256_scope: Nvfp4FourOverSixScope = "all"
    four_over_six_error_mode: Literal["MAE", "MSE"] = "MSE"
    four_over_six_error_use_fast_math: bool = False
    disable_fp4_quant_fast_math: bool = False
    high_precision_last_layers: int = 0
    num_layers: int | None = None
    expert_batch_size: int = 8

    mode: ClassVar[str] = SERIALIZED_NVFP4
    quantized_categories: ClassVar[frozenset[WeightCategory]] = frozenset(
        {"routed_expert_gate", "routed_expert_up", "routed_expert_down"}
    )
    receiver_suffixes: ClassVar[dict[str, str]] = {
        ".weight": "",
        ".weight_scale": "_scale",
        ".weight_scale_2": "_scale_2",
        ".input_scale": "_input_scale",
    }
    supported_model_types: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}
    )
    reject_unknown_routed_experts: ClassVar[bool] = True
    required_model_dtype: ClassVar[str | None] = "bfloat16"
    vllm_quantization: ClassVar[str] = "modelopt_fp4"

    @classmethod
    def from_config(cls, config: Any, *, num_layers: int | None = None) -> "Nvfp4ExpertStrategy":
        """Build a strategy from expert NVFP4 model configuration."""

        return cls(
            backward_override=config.backward_override,
            disable_rht=config.disable_rht,
            disable_stochastic_rounding=config.disable_stochastic_rounding,
            disable_2d_quantization=config.disable_2d_quantization,
            row_scaled_activation=config.row_scaled_activation,
            four_over_six_scope=config.four_over_six_scope,
            four_over_six_e4m3_use_256_scope=config.four_over_six_e4m3_use_256_scope,
            four_over_six_error_mode=config.four_over_six_error_mode,
            four_over_six_error_use_fast_math=config.four_over_six_error_use_fast_math,
            disable_fp4_quant_fast_math=config.disable_fp4_quant_fast_math,
            high_precision_last_layers=config.high_precision_last_layers,
            num_layers=num_layers,
            expert_batch_size=config.expert_batch_size,
        )

    def _is_high_precision_layer(self, name: str) -> bool:
        if self.high_precision_last_layers == 0:
            return False
        if self.num_layers is None:
            raise RuntimeError("NVFP4 high-precision layer filtering requires the model layer count")
        match = _LAYER_INDEX_PATTERN.search(name)
        return bool(match and int(match.group(1)) >= self.num_layers - self.high_precision_last_layers)

    def should_quantize(
        self,
        category: WeightCategory | None,
        name: str,
        shape: Sequence[int],
    ) -> bool:
        """Keep configured final routed-expert layers in BF16."""

        return super().should_quantize(category, name, shape) and not self._is_high_precision_layer(name)

    def build_te_recipe(self, *, persistent: bool) -> dict:
        """Build Transformer Engine NVFP4 settings."""

        return build_nvfp4_te_recipe(persistent=persistent)

    def configure_megatron_provider(self, provider) -> None:
        """Set provider fields required by NVFP4 training."""

        required_recipe_fields = set()
        if self.row_scaled_activation:
            required_recipe_fields.add("row_scaled_activation")
        if self.four_over_six_scope != "none":
            required_recipe_fields.update(
                {
                    "nvfp4_4over6",
                    "nvfp4_4over6_e4m3_use_256",
                    "nvfp4_4over6_err_mode",
                }
            )
        if self.backward_override is not None:
            required_recipe_fields.add("backward_override")
        if required_recipe_fields:
            from transformer_engine.common.recipe import NVFP4BlockScaling

            recipe_fields = set(getattr(NVFP4BlockScaling, "__dataclass_fields__", ()))
            missing_fields = required_recipe_fields - recipe_fields
            if missing_fields:
                raise RuntimeError(
                    f"Transformer Engine NVFP4 recipe does not support fields: {sorted(missing_fields)}"
                )
        configure_nvfp4_provider(provider)
        if self.high_precision_last_layers:
            if self.high_precision_last_layers >= provider.num_layers:
                raise ValueError(
                    "expert_nvfp4.high_precision_last_layers must be smaller than the model layer count, "
                    f"got {self.high_precision_last_layers} for {provider.num_layers} layers"
                )
            provider.first_last_layers_bf16 = True
            provider.num_layers_at_start_in_bf16 = 0
            provider.num_layers_at_end_in_bf16 = self.high_precision_last_layers

    def build_runtime_env(self) -> dict[str, str]:
        """Return Transformer Engine NVFP4 process settings."""

        env = {
            "NVTE_NVFP4_DISABLE_RHT": str(int(self.disable_rht)),
            "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": str(int(self.disable_stochastic_rounding)),
            "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": str(int(self.disable_2d_quantization)),
            "NVTE_NVFP4_ROW_SCALED_ACTIVATION": str(int(self.row_scaled_activation)),
            "NVTE_NVFP4_4OVER6": self.four_over_six_scope,
            "NVTE_NVFP4_4OVER6_E4M3_USE_256": self.four_over_six_e4m3_use_256_scope,
            "NVTE_NVFP4_4OVER6_ERR_MODE": self.four_over_six_error_mode,
            "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": str(int(self.four_over_six_error_use_fast_math)),
            "SKYRL_VLLM_NVFP4_PER_TOKEN_ACTIVATION": str(int(self.row_scaled_activation)),
            "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": str(int(self.disable_fp4_quant_fast_math)),
            "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": str(int(self.disable_fp4_quant_fast_math)),
        }
        if self.backward_override is not None:
            env["NVTE_BACKWARD_OVERRIDE"] = self.backward_override
        if self.four_over_six_scope in {"activations", "all"}:
            env.update(
                {
                    "FLASHINFER_NVFP4_4OVER6": "1",
                    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": str(
                        int(self.four_over_six_e4m3_use_256_scope in {"activations", "all"})
                    ),
                    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": self.four_over_six_error_mode,
                    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": str(
                        int(self.four_over_six_error_use_fast_math)
                    ),
                }
            )
        return env

    def prepare_serialization(
        self,
        weights: Sequence[tuple[str, torch.Tensor]],
    ) -> dict[str, Any]:
        """Quantize fused gate/up weights together with one global scale."""

        gate = next(((name, tensor) for name, tensor in weights if name.endswith(".gate_proj.weight")), None)
        up = next(((name, tensor) for name, tensor in weights if name.endswith(".up_proj.weight")), None)
        if gate is None or up is None:
            return {}

        gate_name, gate_tensor = gate
        up_name, up_tensor = up
        if self._is_high_precision_layer(gate_name):
            return {}
        if gate_tensor.ndim != up_tensor.ndim:
            raise ValueError("Fused gate and up weights must have matching dimensions")
        if gate_tensor.is_cuda:
            four_over_six = self.four_over_six_scope in {"weights", "all"}
            e4m3_max = (
                int(NVFP4_4OVER6_FP8_MAX)
                if self.four_over_six_e4m3_use_256_scope in {"weights", "all"}
                else 448
            )
            gate_output, up_output = _nvfp4_te_quantize_pair(
                gate_tensor,
                up_tensor,
                four_over_six=four_over_six,
                e4m3_max=e4m3_max,
                error_mode=self.four_over_six_error_mode,
            )
            return {gate_name: gate_output, up_name: up_output}
        reduce_dims = (-2, -1) if gate_tensor.ndim == 3 else None
        keepdim = gate_tensor.ndim == 3
        gate_amax = gate_tensor.float().abs().amax(dim=reduce_dims, keepdim=keepdim)
        up_amax = up_tensor.float().abs().amax(dim=reduce_dims, keepdim=keepdim)
        denominator = (
            6.0 * NVFP4_4OVER6_FP8_MAX
            if self.four_over_six_scope in {"weights", "all"}
            else NVFP4_GLOBAL_SCALE_DENOMINATOR
        )
        global_scale = (torch.maximum(gate_amax, up_amax) / denominator).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        return {gate_name: global_scale, up_name: global_scale}

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
        context: Any = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield a packed NVFP4 weight and its two scale tensors."""

        quantize_weights_with_four_over_six = self.four_over_six_scope in {"weights", "all"}
        e4m3_max = (
            int(NVFP4_4OVER6_FP8_MAX)
            if self.four_over_six_e4m3_use_256_scope in {"weights", "all"}
            else 448
        )
        if isinstance(context, tuple) and len(context) == 3:
            q_weight, block_scale, global_scale = context
        elif batched_experts:
            q_weight, block_scale, global_scale = batched_nvfp4_cast_to_fp4(
                tensor,
                four_over_six=quantize_weights_with_four_over_six,
                e4m3_max=e4m3_max,
                error_mode=self.four_over_six_error_mode,
                expert_batch_size=self.expert_batch_size,
                global_scale=context,
            )
        else:
            q_weight, block_scale, global_scale = nvfp4_cast_to_fp4(
                tensor,
                four_over_six=quantize_weights_with_four_over_six,
                e4m3_max=e4m3_max,
                error_mode=self.four_over_six_error_mode,
                global_scale=context,
            )
        block_scale_name, global_scale_name = nvfp4_scale_names_for_weight(name)
        yield name, q_weight
        yield block_scale_name, block_scale
        yield global_scale_name, global_scale
        if self.row_scaled_activation:
            yield name[: -len(".weight")] + ".input_scale", torch.ones_like(global_scale)

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: QuantizedModelLayout,
    ) -> dict[str, Any]:
        """Return ModelOpt W4A16 NVFP4 settings for vLLM."""

        del inference_config, layout
        num_layers = getattr(hf_config, "num_hidden_layers", None) or getattr(hf_config, "num_layers", None)
        ignored_modules = []
        if self.high_precision_last_layers:
            if num_layers is None:
                raise ValueError("NVFP4 final-layer BF16 filtering requires num_hidden_layers in the HF config")
            if self.high_precision_last_layers >= num_layers:
                raise ValueError("NVFP4 high_precision_last_layers must be smaller than the model layer count")
            ignored_modules.extend(
                f"*.layers.{layer_idx}.mlp.experts*"
                for layer_idx in range(num_layers - self.high_precision_last_layers, num_layers)
            )
        return get_serialized_nvfp4_quantization_config(
            per_token_activation=self.row_scaled_activation,
            additional_ignored_modules=ignored_modules,
        )
