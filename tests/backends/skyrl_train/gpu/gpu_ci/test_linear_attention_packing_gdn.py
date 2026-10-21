"""Packed Gated DeltaNet sequences must not leak recurrent state across trajectories.

Before the fix, a datum's per-token logprobs differ depending on which unrelated sequences it was
packed with (SkyRL #1525). Attention derives the boundaries from position_ids; Gated DeltaNet only
sees them if the wrapper forwards them explicitly.
"""

import pytest
import torch

pytest.importorskip("causal_conv1d")
pytest.importorskip("fla")

from skyrl.backends.skyrl_train.utils.linear_attention_packing import (  # noqa: E402
    gated_delta_net_layers,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,  # noqa: E402
)

GDN_MODEL = "Qwen/Qwen3.5-0.8B"
CONTROL_MODEL = "Qwen/Qwen3-0.6B"
SEQ_LEN = 16
# Calibrated on Qwen3.5 bf16: packing noise stays well below this, a cross-sequence leak reads as >1.
MAX_DELTA = 0.5
MEAN_DELTA = 0.1


def _build_wrapper(pretrain_or_model, *, packing):
    wrapper = HFModelWrapper(
        pretrain_or_model=pretrain_or_model,
        use_flash_attention_2=True,
        bf16=True,
        sequence_parallel_size=1,
        remove_microbatch_padding=packing,
        language_model_only=True,
    )
    wrapper.model.eval()
    wrapper.model.to("cuda")
    return wrapper


def _forward(wrapper, sequences, num_actions):
    return wrapper(sequences.cuda(), num_actions, torch.ones_like(sequences).cuda())


def _packed_vs_standalone(wrapper, seq_len, seed):
    """Max/mean |delta| for sequence B: forwarded alone vs packed after a different sequence A."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(1, 1000, (1, seq_len), generator=generator)
    b = torch.randint(1, 1000, (1, seq_len), generator=generator)
    packed = torch.cat([a, b], dim=0)
    num_actions = seq_len - 1

    standalone = _forward(wrapper, b, num_actions)[0]
    batched = _forward(wrapper, packed, num_actions)[1]

    delta = (batched - standalone).abs()
    return float(delta.max()), float(delta.mean())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed GDN kernels need a GPU")
def test_packed_gated_delta_net_matches_standalone_forward():
    wrapper = _build_wrapper(GDN_MODEL, packing=True)
    assert gated_delta_net_layers(wrapper.model), f"{GDN_MODEL} has no Gated DeltaNet layers"

    max_delta, mean_delta = _packed_vs_standalone(wrapper, SEQ_LEN, seed=0)

    assert max_delta <= MAX_DELTA, f"cross-sequence leak: max |delta| {max_delta:.4f}"
    assert mean_delta <= MEAN_DELTA, f"cross-sequence leak: mean |delta| {mean_delta:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed GDN kernels need a GPU")
def test_unpacked_gated_delta_net_is_the_sanity_control():
    wrapper = _build_wrapper(GDN_MODEL, packing=False)

    max_delta, _ = _packed_vs_standalone(wrapper, SEQ_LEN, seed=0)

    assert max_delta <= MAX_DELTA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed kernels need a GPU")
def test_pure_attention_control_packing_is_clean():
    wrapper = _build_wrapper(CONTROL_MODEL, packing=True)
    assert not gated_delta_net_layers(wrapper.model)

    max_delta, mean_delta = _packed_vs_standalone(wrapper, SEQ_LEN, seed=0)

    assert max_delta <= MAX_DELTA
    assert mean_delta <= MEAN_DELTA
