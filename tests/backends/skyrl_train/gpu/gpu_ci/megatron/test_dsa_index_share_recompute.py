"""Regression test for the DSA index-share recompute patch.

Ported from the test added in NVIDIA/Megatron-LM#6793 so SkyRL verifies the
underlying issue is actually fixed by
``skyrl.backends.skyrl_train.patches.megatron.patch_dsa_index_share``, rather
than only that the patch applies cleanly.

The scenario is two outstanding non-packed checkpointed forwards followed by
their two recomputes (F1 -> F2 -> B1 -> B2). Without the patch both forwards
share one holder (hung off ``attention_mask``), so F2 overwrites F1's top-k
support and B1 recomputes against the wrong forward's state -- the assertion
below sees ``[2, 2]`` instead of ``[1, 2]``.

Needs no GPU, but does need the ``megatron`` extra, so it lives with the rest of
the Megatron GPU CI suite.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_dsa_index_share_recompute.py
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_index_share import (
    patch_dsa_index_share,
)

pytestmark = pytest.mark.megatron


@pytest.fixture(scope="module")
def patched_megatron():
    """Apply the backport, skipping if this megatron-core does not need it."""
    if not patch_dsa_index_share():
        pytest.skip(
            "DSA index-share patch did not apply -- megatron-core either already "
            "contains NVIDIA/Megatron-LM#6793 or no longer matches the 0.20.0 source form"
        )


def _build_attention():
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAttention,
        DSAttentionSubmodules,
    )

    config = SimpleNamespace(
        dsa_indexer_topk=8,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=1,
        kv_channels=16,
    )
    return DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
        pg_collection=SimpleNamespace(),
    )


def test_nonpacked_checkpointed_forwards_keep_index_share_holders_isolated(patched_megatron, monkeypatch):
    """Each checkpointed forward recomputes against its own top-k support."""
    from megatron.core import recompute
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    attention = _build_attention()
    recomputed_supports = []

    class HolderLayer:
        layer_number = 2

        def __call__(self, hidden_states, attention_mask, packed_seq_params, **_kwargs):
            topk_holder = attention._get_index_share_topk_holder(packed_seq_params, attention_mask)
            length_holder = attention._get_index_share_topk_length_holder(packed_seq_params, attention_mask)
            if torch.is_grad_enabled():
                recomputed_supports.append((topk_holder[1].clone(), length_holder[1].clone()))
            else:
                topk_holder[1] = hidden_states.detach().to(torch.int64)
                length_holder[1] = hidden_states.detach().to(torch.int32)
            return hidden_states

    block = SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_indexer_topk_freq=4,
            fp8=False,
            fp4=False,
            distribute_saved_activations=False,
            recompute_method="block",
            recompute_num_layers=1,
        ),
        layers=[HolderLayer()],
        num_layers_per_pipeline_rank=1,
        pg_collection=SimpleNamespace(tp=None),
    )
    checkpoint_calls = []

    def fake_checkpoint(function, _distribute_saved_activations, *args):
        checkpoint_calls.append((function, args))
        with torch.no_grad():
            return function(*args)

    monkeypatch.setattr("megatron.core.recompute.tensor_parallel.checkpoint", fake_checkpoint)
    shared_attention_mask = torch.empty(1)

    # F1 then F2: both stage their top-k support before either is recomputed.
    for value in (1.0, 2.0):
        recompute.checkpointed_forward(
            block,
            hidden_states=torch.tensor([value]),
            attention_mask=shared_attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_emb=torch.empty(1),
            attention_bias=None,
            packed_seq_params=None,
            use_inner_quantization_context=False,
        )

    # B1 then B2.
    for function, args in checkpoint_calls:
        with torch.enable_grad():
            function(*args)

    assert [support.item() for support, _length in recomputed_supports] == [1, 2]
    assert [length.item() for _support, length in recomputed_supports] == [1, 2]

    # The shared mask must never have been used as a carrier.
    assert not hasattr(shared_attention_mask, DSAttention._HOLDER_ATTR)
    assert not hasattr(shared_attention_mask, DSAttention._LENGTH_HOLDER_ATTR)


def test_packed_forward_still_uses_packed_seq_params(patched_megatron):
    """The packed path is untouched: PackedSeqParams stays the carrier."""
    attention = _build_attention()
    packed_seq_params = SimpleNamespace()

    carrier = attention._get_index_share_carrier(packed_seq_params, torch.empty(1))

    assert carrier is packed_seq_params


def test_patch_is_idempotent(patched_megatron):
    """Re-applying is a no-op that still reports success."""
    from megatron.core import recompute

    before = recompute.checkpointed_forward
    assert patch_dsa_index_share() is True
    assert recompute.checkpointed_forward is before


def test_importers_see_the_patched_function(patched_megatron):
    """TransformerBlock bound the function by name, so it must be rebound too."""
    from megatron.core import recompute
    from megatron.core.transformer import transformer_block

    assert transformer_block.checkpointed_forward is recompute.checkpointed_forward
