"""CPU tests for ``patch_routed_experts_pp_relay`` (R3 with vLLM pipeline parallelism).

Two capturers stand in for two PP stages: stage 0 captures its layers and sends the prefix, stage 1
installs it and captures the rest, and the last stage must then hold every layer.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

pytestmark = pytest.mark.vllm

NUM_LAYERS, TOP_K, NUM_EXPERTS, MAX_TOKENS = 27, 6, 64, 32


@pytest.fixture(scope="module")
def relay():
    from skyrl.backends.skyrl_train.patches.vllm import (
        patch_routed_experts_pp_relay as relay,
    )

    status = relay._relay_status()
    if status != "patchable":
        pytest.skip(f"installed vLLM is {status!r} for the PP relay patch")
    relay.apply_pp_relay_worker_patch()
    relay.apply_pp_relay_config_patch()
    assert relay._WORKER_PATCHED and relay._CONFIG_PATCHED
    return relay


@pytest.fixture
def make_capturer(relay, monkeypatch):
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    monkeypatch.setattr(rec, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(rec, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None))

    def make(pp_rank, pp_size):
        monkeypatch.setattr(
            ps,
            "get_pp_group",
            lambda: SimpleNamespace(rank_in_group=pp_rank, world_size=pp_size, is_last_rank=pp_rank == pp_size - 1),
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_total_num_hidden_layers=lambda: NUM_LAYERS,
                get_num_experts=lambda: NUM_EXPERTS,
                get_num_experts_per_tok=lambda: TOP_K,
                hf_text_config=SimpleNamespace(model_type="test"),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_rank=0, tensor_parallel_size=1, pipeline_parallel_size=pp_size
            ),
        )
        return rec.RoutedExpertsCapturer(max_num_batched_tokens=MAX_TOKENS, vllm_config=config)

    return make


def _routes(num_tokens, seed):
    """Distinct top-k expert ids per (token, layer); layer 0 is dense (all zero), as in Moonlight."""
    gen = torch.Generator().manual_seed(seed)
    routes = torch.rand(num_tokens, NUM_LAYERS, NUM_EXPERTS, generator=gen).topk(TOP_K, -1).indices.to(torch.int32)
    routes[:, 0] = 0
    return routes


def _capture_stage(capturer, routes):
    for layer in range(max(capturer.start_layer, 1), capturer.end_layer):
        capturer.capture(layer, routes[:, layer].to(torch.int64))


@pytest.mark.parametrize("pp_size,partition", [(2, None), (2, "1,26"), (2, "20,7"), (4, None), (4, "2,9,8,8")])
def test_relay_reassembles_every_layer(make_capturer, monkeypatch, pp_size, partition):
    if partition:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    stages = [make_capturer(rank, pp_size) for rank in range(pp_size)]
    for prev, nxt in zip(stages, stages[1:]):
        assert prev.end_layer == nxt.start_layer
    assert stages[0].start_layer == 0 and stages[-1].end_layer == NUM_LAYERS

    num_tokens = 9
    routes = _routes(num_tokens, 0)
    sent = None
    for stage in stages:
        if sent is not None:
            stage.load_pp_routes(sent)
        _capture_stage(stage, routes)
        if stage is not stages[-1]:
            sent = stage.get_pp_routes(num_tokens)
            assert sent.dtype == torch.int32 and sent.is_contiguous()
            assert sent.shape == (num_tokens, stage.end_layer, TOP_K)
            # A copy: the next step overwrites the capture buffer while the send is in flight.
            assert sent.data_ptr() != stage.device_buffer.data_ptr()
    torch.testing.assert_close(stages[-1].device_buffer[:num_tokens], routes, rtol=0, atol=0)


def test_without_relay_earlier_layers_are_lost(make_capturer):
    """The failure the relay fixes: the last stage alone holds zeros for stage 0's layers."""
    last = make_capturer(1, 2)
    routes = _routes(5, 1)
    _capture_stage(last, routes)
    assert not last.device_buffer[:5, 1 : last.start_layer].any()


def test_single_stage_is_unchanged(make_capturer):
    capturer = make_capturer(0, 1)
    assert not capturer.is_pipeline_stage
    assert (capturer.start_layer, capturer.end_layer) == (0, NUM_LAYERS)


def test_load_rejects_partition_mismatch_and_dtype(make_capturer):
    stage1 = make_capturer(1, 2)
    with pytest.raises(ValueError, match="layers"):
        stage1.load_pp_routes(torch.zeros(4, stage1.start_layer + 1, TOP_K, dtype=torch.int32))
    with pytest.raises(TypeError):
        stage1.load_pp_routes(torch.zeros(4, stage1.start_layer, TOP_K, dtype=torch.int64))
    with pytest.raises(ValueError, match="rows"):
        stage1.load_pp_routes(torch.zeros(MAX_TOKENS + 1, stage1.start_layer, TOP_K, dtype=torch.int32))


def test_receive_defers_to_comm_postprocess_and_pops_key(relay, make_capturer):
    """The load runs inside ``wait_for_comm`` after the TP all-gather postprocess, not earlier."""
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

    stage1 = make_capturer(1, 2)
    routes = _routes(4, 2)[:, : stage1.start_layer].contiguous()
    events = []
    tensors = {"hidden_states": torch.zeros(4, 2), "routed_experts": torch.full((2, 7), -1, dtype=torch.int32)}

    class Handle:
        def wait(self):
            events.append("wait")

    def all_gather():
        events.append("gather")
        tensors["routed_experts"] = routes.clone()

    pending = AsyncIntermediateTensors(tensors, comm_handles=[Handle()], comm_postprocess=[all_gather])
    relay._install_receive(pending, stage1, num_tokens=4)
    assert events == []  # the receive still overlaps input preparation
    assert list(pending.tensors) == ["hidden_states"]  # first access, as in the model runner
    assert events == ["wait", "gather"]
    torch.testing.assert_close(stage1.device_buffer[:4, : stage1.start_layer], routes, rtol=0, atol=0)

    ready = IntermediateTensors({"hidden_states": torch.zeros(4, 2), "routed_experts": routes.clone()})
    relay._install_receive(ready, stage1, num_tokens=4)
    assert "routed_experts" not in ready.tensors

    with pytest.raises(AssertionError, match="did not relay"):
        relay._install_receive(IntermediateTensors({"hidden_states": torch.zeros(4, 2)}), stage1, num_tokens=4)


def _fake_vllm_config(pp_size=2, dcp_size=1):
    return SimpleNamespace(
        aux_output_config=SimpleNamespace(enabled=True),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(runner_type="generate", is_moe=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size, decode_context_parallel_size=dcp_size, prefill_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )


def test_config_patch_lifts_only_the_pp_rejection(relay):
    from vllm.config.vllm import VllmConfig

    patched = VllmConfig._verify_aux_output_compatibility
    with pytest.raises(ValueError, match="pipeline parallelism"):
        patched.__wrapped__(_fake_vllm_config(pp_size=2))

    config = _fake_vllm_config(pp_size=2)
    patched(config)
    assert config.parallel_config.pipeline_parallel_size == 2  # restored
    with pytest.raises(ValueError, match="context parallelism"):
        patched(_fake_vllm_config(pp_size=2, dcp_size=2))


_PLUGIN_PROBE = """
import types
from vllm.config.vllm import VllmConfig
from vllm.plugins import load_general_plugins
load_general_plugins()  # what the engine core does before re-validating the config
cfg = types.SimpleNamespace(
    aux_output_config=types.SimpleNamespace(enabled=True), use_v2_model_runner=True,
    model_config=types.SimpleNamespace(runner_type="generate", is_moe=True),
    cache_config=types.SimpleNamespace(enable_prefix_caching=True), speculative_config=None,
    parallel_config=types.SimpleNamespace(pipeline_parallel_size=2, decode_context_parallel_size=1,
                                          prefill_context_parallel_size=1),
    kv_transfer_config=None,
)
try:
    VllmConfig._verify_aux_output_compatibility(cfg)
    print("ACCEPTED")
except ValueError as e:
    print("REJECTED", e)
"""


@pytest.mark.parametrize("env_value, expected", [("1", "ACCEPTED"), (None, "REJECTED")])
def test_general_plugin_installs_config_patch_in_a_fresh_vllm_process(relay, env_value, expected):
    """The engine core re-validates the config in its own process: the `vllm.general_plugins` entry point must
    install the config patch there when (and only when) the server actor set SKYRL_R3_PP_RELAY=1."""
    import os
    import subprocess
    import sys
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="vllm.general_plugins")}
    assert eps.get("skyrl_r3_pp_relay", "").endswith("patch_routed_experts_pp_relay:vllm_general_plugin")

    env = {k: v for k, v in os.environ.items() if k not in (relay.PP_RELAY_ENV, "VLLM_PLUGINS")}
    if env_value is not None:
        env[relay.PP_RELAY_ENV] = env_value
    out = subprocess.run([sys.executable, "-c", _PLUGIN_PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert expected in out.stdout, out.stdout + out.stderr
