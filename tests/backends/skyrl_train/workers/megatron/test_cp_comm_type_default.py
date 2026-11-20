"""``default_cp_comm_type_for_sliding_window``: the context-parallel comm type picked for Beam.

Transformer Engine has no ring (``p2p``) context-parallel path for sliding-window attention, so
the default must be ``a2a`` -- or ``all_gather`` when the per-TP-rank head counts do not divide by
the CP size, which ``a2a`` requires.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.megatron

megatron_worker = pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")


def _config(cp, tp=1, heads=64, kv_heads=8, window=(4096, 0), cp_comm_type=None):
    return SimpleNamespace(
        context_parallel_size=cp,
        tensor_model_parallel_size=tp,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        window_size=window,
        cp_comm_type=cp_comm_type,
    )


@pytest.mark.parametrize(
    "tp, cp, expected",
    [
        (1, 2, "a2a"),  # 64 / 8 heads, both divisible by 2
        (4, 2, "a2a"),  # 16 / 2 heads per TP rank
        (8, 2, "all_gather"),  # 1 KV head per TP rank: not divisible by CP=2
        (4, 4, "all_gather"),  # 2 KV heads per TP rank: not divisible by CP=4
    ],
)
def test_defaults_to_a2a_only_when_heads_divide(tp, cp, expected):
    config = _config(cp=cp, tp=tp)
    megatron_worker.default_cp_comm_type_for_sliding_window(config)
    assert config.cp_comm_type == expected


def test_explicit_choice_and_no_cp_are_left_alone():
    config = _config(cp=2, cp_comm_type="all_gather")
    megatron_worker.default_cp_comm_type_for_sliding_window(config)
    assert config.cp_comm_type == "all_gather"
    config = _config(cp=1)
    megatron_worker.default_cp_comm_type_for_sliding_window(config)
    assert config.cp_comm_type is None
    config = _config(cp=2, window=None)
    megatron_worker.default_cp_comm_type_for_sliding_window(config)
    assert config.cp_comm_type is None


@pytest.mark.parametrize("ring", ["p2p", "a2a+p2p"])
def test_ring_choices_are_rejected(ring):
    with pytest.raises(ValueError, match="sliding-window"):
        megatron_worker.default_cp_comm_type_for_sliding_window(_config(cp=2, cp_comm_type=ring))
