"""Routed-expert NVFP4 training through Transformer Engine's NVFP4 recipe (humans& "4-bitter Lesson").

Only ``*.experts.linear_fc1/fc2`` (TE GroupedLinear) run in NVFP4; a per-module megatron-core quant recipe keeps
every other module BF16, and ``num_layers_at_{start,end}_in_bf16`` (Megatron's ``first_last_layers_bf16``, the
same knobs as FP8 training) keep the routed experts of the first / last N layers BF16. The
recipe knobs (per-token activations, 4/6, dequantized backward, ...) come from the ``NVTE_NVFP4_*`` env that
``nvfp4_runtime_env`` forwards to every actor; TE reads it when it builds ``NVFP4BlockScaling()``.
Adapted from NovaSky-AI/SkyRL#1967 (``quantization/{nvfp4,megatron}.py``).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _te_recipe_config() -> dict:
    # The same recipe in train and eval mode: the trainer's logprob forwards must match the sampler too.
    # override_nonquantized_autocast stays False: megatron-core wraps every layer in the fp4 context except the
    # ``first_last_layers_bf16`` layers, and the experts there must follow that BF16 context, as on the sampler.
    recipe = {
        "fp4_quantization_recipe": "nvfp4",
        "override_quantized_autocast": True,
        "override_nonquantized_autocast": False,
    }
    return {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": recipe,
        "evaluation_recipe": dict(recipe),
    }


def build_expert_nvfp4_quant_recipe() -> dict:
    high_precision = {"override_quantized_autocast": True}
    return {
        "configs": {
            "nvfp4": _te_recipe_config(),
            "high_precision": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": high_precision,
                "evaluation_recipe": dict(high_precision),
            },
        },
        # First match wins: routed expert fc1/fc2 -> NVFP4, everything else -> BF16.
        "matchers": {
            "routed_fc1": {"type": "glob", "pattern": "*.experts.linear_fc1", "config": "nvfp4", "enabled": True},
            "routed_fc2": {"type": "glob", "pattern": "*.experts.linear_fc2", "config": "nvfp4", "enabled": True},
            "default": {"type": "glob", "pattern": "*", "config": "high_precision", "enabled": True},
        },
    }


def configure_expert_nvfp4(provider) -> None:
    """Set the Megatron provider fields for routed-expert NVFP4.

    BF16 first / last layers come from the provider's ``first_last_layers_bf16`` /
    ``num_layers_at_{start,end}_in_bf16``, which ``MegatronConfig`` folds into ``transformer_config_kwargs``.
    """
    from megatron.core.quantization.quant_config import RecipeConfig
    from transformer_engine.common.recipe import NVFP4BlockScaling

    from skyrl.backends.skyrl_train.weight_sync.nvfp4 import nvfp4_bf16_layers

    if torch.cuda.is_available() and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError(f"expert_nvfp4 requires SM100/SM103, got {torch.cuda.get_device_capability()}")
    if getattr(provider, "quant_recipe", None) is not None:
        raise ValueError("expert_nvfp4 conflicts with an existing quant_recipe")
    if getattr(provider, "fp8", None) not in (None, False) or getattr(provider, "fp4", None) not in (None, False):
        raise ValueError("expert_nvfp4 conflicts with a global fp8/fp4 setting")
    if not provider.moe_grouped_gemm:
        raise ValueError("expert_nvfp4 requires moe_grouped_gemm=true")
    if getattr(provider, "expert_tensor_parallel_size", 1) not in (None, 1):
        # Each expert's global (per-tensor) scale must come from the whole matrix, as on the sampler.
        raise ValueError("expert_nvfp4 requires expert_tensor_parallel_size=1")

    provider.fp4 = "e2m1"
    provider.fp4_recipe = "nvfp4"
    provider.quant_recipe = RecipeConfig.from_config_dict(build_expert_nvfp4_quant_recipe())
    # Validates the range; the trainer's BF16 layers are applied by Megatron's first_last_layers_bf16 itself.
    bf16_layers = nvfp4_bf16_layers(provider.num_layers, provider)

    recipe = NVFP4BlockScaling()
    logger.info(
        "expert_nvfp4: routed experts in NVFP4, layers %s BF16; TE recipe: row_scaled_activation=%s "
        "4over6=%s e4m3_use_256=%s err_mode=%s backward_override=%s rht=%s sr=%s 2d=%s",
        bf16_layers,
        recipe.row_scaled_activation,
        recipe.nvfp4_4over6,
        recipe.nvfp4_4over6_e4m3_use_256,
        recipe.nvfp4_4over6_err_mode,
        recipe.backward_override,
        recipe.fp4_quant_fwd_weight.random_hadamard_transform,
        recipe.fp4_quant_fwd_weight.stochastic_rounding,
        recipe.fp4_quant_fwd_weight.fp4_2d_quantization,
    )
