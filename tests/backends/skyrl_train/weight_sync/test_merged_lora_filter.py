"""Merged-LoRA weight sync sends only what training changes (``lora_config.sync_lora_targets_only``).

The filter itself, the forced level-1 sleep that keeps vLLM's unsent weights valid, and (with the
vLLM wheel) the filtered ``MegatronWeightSource`` export.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.merged_lora_filter import (
    MergedLoraSyncFilter,
    select_sync_names,
    sync_unit,
)

L = "decoder.layers.{}."
# A Megatron qwen4_exp-like layer: LoRA (all-linear) wraps the attention / MLP linears.
NAMES = [
    "embedding.word_embeddings.weight",
    *[
        name.format(i)
        for i in (0, 1)
        for name in (
            L + "attn_hyper_connection.input_mix_weight_down.weight",
            L + "self_attention.linear_qkv.to_wrap.weight",
            L + "self_attention.linear_qkv.to_wrap.layer_norm_weight",
            L + "self_attention.q_layernorm.weight",
            L + "self_attention.indexer.index_qk_proj.weight",
            L + "self_attention.linear_proj.to_wrap.weight",
            L + "mlp_hyper_connection.block_inject_weight.weight",
            L + "mlp.router.weight",
            L + "mlp.experts.linear_fc1.to_wrap.weight0",
            L + "mlp.experts.linear_fc1.to_wrap.weight1",
            L + "mlp.experts.linear_fc2.weight0",
            L + "mlp.experts.linear_fc2.weight1",
            L + "mlp.shared_experts.linear_fc1.to_wrap.weight",
        )
    ],
    "decoder.final_layernorm.input_mix_weight_up.weight",
    "output_layer.weight",
]


@pytest.mark.parametrize(
    "name, unit",
    [
        ("decoder.layers.3.self_attention.linear_qkv.to_wrap.weight", "decoder.layers.3.self_attention."),
        ("decoder.layers.3.mlp.experts.linear_fc2.weight17", "decoder.layers.3.mlp."),
        ("decoder.layers.3.attn_hyper_connection.hc_norm.weight", "decoder.layers.3.attn_hyper_connection."),
        ("mtp.layers.0.transformer_layer.mlp.router.weight", "mtp.layers.0.transformer_layer."),
        (
            "vision_model.decoder.layers.2.self_attention.linear_qkv.weight",
            "vision_model.decoder.layers.2.self_attention.",
        ),
        ("output_layer.to_wrap.weight", "output_layer."),
        ("output_layer.weight", "output_layer."),
        ("embedding.word_embeddings.weight", "embedding.word_embeddings."),
        ("decoder.final_layernorm.weight", "decoder.final_layernorm."),
    ],
)
def test_sync_unit(name, unit):
    assert sync_unit(name) == unit


def test_only_blocks_with_lora_targets_are_kept():
    keep = select_sync_names(NAMES, [n for n in NAMES if ".to_wrap." in n])
    dropped = set(NAMES) - keep
    # Frozen modules outside any wrapped block stay with vLLM's checkpoint copies.
    assert dropped == {
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.input_mix_weight_up.weight",
        "output_layer.weight",
        *(
            n.format(i)
            for i in (0, 1)
            for n in (
                L + "attn_hyper_connection.input_mix_weight_down.weight",
                L + "mlp_hyper_connection.block_inject_weight.weight",
            )
        ),
    }
    # Frozen siblings inside a wrapped block ride along: vLLM's FusedMoE holds fc1 AND fc2 of every
    # expert, so sending fc1 alone would reload the layer with an uninitialized fc2.
    for i in (0, 1):
        assert L.format(i) + "mlp.experts.linear_fc2.weight0" in keep
        assert L.format(i) + "mlp.router.weight" in keep
        assert L.format(i) + "self_attention.linear_qkv.to_wrap.layer_norm_weight" in keep


def test_attention_only_lora_skips_the_mlp_blocks():
    changed = [n for n in NAMES if "self_attention.linear_qkv.to_wrap" in n]
    keep = select_sync_names(NAMES, changed)
    assert keep == {n for n in NAMES if ".self_attention." in n}


class _Bridge:
    def __init__(self, tasks):
        self.tasks = tasks
        self.calls = 0

    def get_conversion_tasks(self, module):
        self.calls += 1
        return self.tasks


def _task(name, weight):
    return SimpleNamespace(global_param_name=name, param_weight=weight)


def _frozen():
    return torch.nn.Parameter(torch.zeros(1), requires_grad=False)


def test_filter_adds_trainable_params_and_buffers_and_resolves_once():
    tasks = [_task(n, _frozen()) for n in NAMES]
    # A user-unfrozen module and a persistent buffer (e.g. a router's expert_bias) also change.
    tasks[-1] = _task("output_layer.weight", torch.nn.Parameter(torch.zeros(1)))
    tasks.append(_task("decoder.layers.5.mlp.router.expert_bias", torch.zeros(1)))
    tasks.append(_task("decoder.layers.5.mlp.router.weight", _frozen()))
    tasks.append(_task("decoder.layers.6.mlp.router.weight", None))  # held by another PP stage
    bridge = _Bridge(tasks)
    keep = MergedLoraSyncFilter(bridge, object())

    assert keep("output_layer.weight")
    assert keep("decoder.layers.5.mlp.router.expert_bias") and keep("decoder.layers.5.mlp.router.weight")
    assert not keep("decoder.layers.6.mlp.router.weight")
    assert not keep("embedding.word_embeddings.weight")
    assert keep("decoder.layers.1.mlp.experts.linear_fc2.weight1")
    assert bridge.calls == 1


def test_filter_unions_what_other_ranks_hold(monkeypatch):
    """Under PP/EP a rank cannot see every param's requires_grad: the changed set is all-gathered."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def all_gather_object(out, obj, group=None):
        out[0] = obj
        out[1] = ["decoder.layers.6.mlp.router.weight"]  # trainable on the stage that holds it

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    tasks = [_task(n, _frozen()) for n in NAMES] + [_task("decoder.layers.6.mlp.router.weight", None)]
    keep = MergedLoraSyncFilter(_Bridge(tasks), object())
    assert keep("decoder.layers.6.mlp.router.weight")


def test_filter_refuses_a_model_without_lora():
    keep = MergedLoraSyncFilter(_Bridge([_task("decoder.layers.0.mlp.linear_fc1.weight", _frozen())]), object())
    with pytest.raises(ValueError, match="would send nothing"):
        keep("decoder.layers.0.mlp.linear_fc1.weight")


def _sleep_levels(**client_kwargs):
    import asyncio

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    client = RemoteInferenceClient(
        proxy_url="http://proxy", server_urls=["http://s0"], model_name="m", data_parallel_size=1, **client_kwargs
    )
    sent = []

    async def pause(*args, **kwargs):
        return {}

    async def call_all(path, params=None, **kwargs):
        sent.append((path, params))
        return {}

    client.pause = pause
    client._call_all_servers = call_all
    asyncio.run(client.sleep())
    return client, [params["level"] for path, params in sent if path == "/sleep"]


@pytest.mark.parametrize(
    "client_kwargs, level",
    [
        ({}, "2"),
        ({"uses_lora_weight_sync": True}, "1"),
        ({"keep_weights_on_sleep": True}, "1"),
    ],
)
def test_partial_sync_forces_level_one_sleep(client_kwargs, level):
    client, levels = _sleep_levels(**client_kwargs)
    assert levels == [level]
    assert client.sleep_keeps_weights == (level == "1")


@pytest.mark.vllm
def test_megatron_source_exports_the_filtered_tasks_in_one_call():
    pytest.importorskip("vllm", reason="sources.py imports vLLM's WeightSource")
    from skyrl.backends.skyrl_train.weight_sync.sources import MegatronWeightSource

    class _ExportingBridge(_Bridge):
        def __init__(self, tasks):
            super().__init__(tasks)
            self.exported = []

        def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
            self.exported.append([t.global_param_name for t in conversion_tasks])
            return iter([(t.global_param_name, torch.zeros(2)) for t in conversion_tasks])

    bridge = _ExportingBridge([_task(n, _frozen()) for n in NAMES])
    source = MegatronWeightSource(bridge, object(), torch.bfloat16, param_filter=MergedLoraSyncFilter(bridge, object()))
    sent = [name for name, _ in source]
    assert [m.name for m in source.metadata()] == sent
    assert "embedding.word_embeddings.weight" not in sent and "output_layer.weight" not in sent
    assert sent == [n for n in NAMES if n in select_sync_names(NAMES, [x for x in NAMES if ".to_wrap." in x])]
    assert all(names == sent for names in bridge.exported)  # HF order kept, one export per pass
