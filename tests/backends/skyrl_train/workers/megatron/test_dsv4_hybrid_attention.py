"""CPU tests for the ``dsv4_hybrid`` (DeepSeek-V4 / V4.1 compressed sparse attention) handling in the Megatron wrapper
and worker: unpacked-row alignment to the compression ratios, no explicit model attention mask, and no ``core_attn``
selective recompute. Other models must stay unchanged.

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_dsv4_hybrid_attention.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.megatron


def _wrapper(variant=None, ratios=None, mlp_chunks=1):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        MegatronModelWrapper,
    )

    model_config = MagicMock()
    model_config.experimental_attention_variant = variant
    model_config.csa_compress_ratios = ratios
    model_config.mlp_chunks_for_training = mlp_chunks
    model_config.sequence_parallel = False
    model_config.calculate_per_token_loss = False
    skyrl_config = MagicMock()
    skyrl_config.trainer.remove_microbatch_padding = True
    with patch(
        "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
        return_value=model_config,
    ):
        return MegatronModelWrapper(config=skyrl_config, actor_module=[MagicMock()], actor_optimizer=None)


@pytest.mark.parametrize(
    "ratios, mlp_chunks, expected",
    [
        ([0, 4, 128], 1, 128),  # DeepSeek-V4: sliding-window-only (0), CSA (4), HCA (128)
        ([0, 1, 2], 1, 2),  # DeepSeek-V4.1
        ([0, 4, 6], 1, 12),  # a multiple of every ratio, not just the largest
        ([0, 4, 128], 3, 384),  # combined with the mlp_chunks alignment
        ([0], 1, 1),
        (None, 1, 1),
    ],
)
def test_dsv4_hybrid_aligns_to_every_compression_ratio(ratios, mlp_chunks, expected):
    wrapper = _wrapper("dsv4_hybrid", ratios, mlp_chunks)
    assert wrapper._implicit_causal_mask_only
    assert wrapper._seq_align == expected


def test_other_models_keep_the_mlp_chunk_alignment_only():
    wrapper = _wrapper(None, [4, 128], mlp_chunks=3)
    assert not wrapper._implicit_causal_mask_only
    assert wrapper._seq_align == 3


def test_dsv4_hybrid_passes_no_explicit_attention_mask():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert _wrapper("dsv4_hybrid", [0, 4, 128])._model_attention_mask(mask) is None


def test_other_models_get_the_te_attention_mask():
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        to_te_attention_mask,
    )

    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert torch.equal(_wrapper(None)._model_attention_mask(mask), to_te_attention_mask(mask))


def _provider(variant, granularity, modules):
    return SimpleNamespace(
        experimental_attention_variant=variant, recompute_granularity=granularity, recompute_modules=modules
    )


@pytest.mark.parametrize(
    "modules, expected_granularity, expected_modules",
    [
        (["core_attn"], None, []),  # nothing left: recompute off
        (["core_attn", "moe", "mhc"], "selective", ["moe", "mhc"]),
        (["moe", "layernorm"], "selective", ["moe", "layernorm"]),  # no core_attn: unchanged
    ],
)
def test_dsv4_hybrid_drops_core_attn_recompute(modules, expected_granularity, expected_modules):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        drop_unsupported_recompute_modules,
    )

    provider = _provider("dsv4_hybrid", "selective", list(modules))
    drop_unsupported_recompute_modules(provider)
    assert provider.recompute_granularity == expected_granularity
    assert provider.recompute_modules == expected_modules


@pytest.mark.parametrize(
    "variant, granularity",
    [(None, "selective"), ("dsa", "selective"), ("dsv4_hybrid", "full")],
)
def test_core_attn_recompute_kept_elsewhere(variant, granularity):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        drop_unsupported_recompute_modules,
    )

    provider = _provider(variant, granularity, ["core_attn", "moe"])
    drop_unsupported_recompute_modules(provider)
    assert provider.recompute_granularity == granularity
    assert provider.recompute_modules == ["core_attn", "moe"]


def test_unsupported_recompute_modules_come_from_the_variant_table(monkeypatch):
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    # A new variant only needs a table entry; every listed module is dropped, the rest kept.
    monkeypatch.setitem(megatron_worker._UNSUPPORTED_RECOMPUTE_MODULES, "other_variant", {"core_attn", "mla_up_proj"})
    provider = _provider("other_variant", "selective", ["core_attn", "moe", "mla_up_proj"])
    megatron_worker.drop_unsupported_recompute_modules(provider)
    assert provider.recompute_granularity == "selective"
    assert provider.recompute_modules == ["moe"]
