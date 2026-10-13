"""CPU checks for the Model Runner V2 pipeline-parallel R3 runtime patch."""

import multiprocessing
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm")
pytestmark = pytest.mark.vllm


def _construct_config_in_spawned_process(model_path):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model=model_path,
        skip_tokenizer_init=True,
        enable_return_routed_experts=True,
        pipeline_parallel_size=2,
        distributed_executor_backend="ray",
        enforce_eager=True,
    )
    config = args.create_engine_config()
    assert config.use_v2_model_runner
    assert config.parallel_config.pipeline_parallel_size == 2

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    with pytest.raises(ValueError, match="incompatible with pipeline parallelism"):
        args.create_engine_config()


def test_pp_patch_loaded_in_spawned_engine_process(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("vLLM EngineArgs configuration needs a CUDA platform")
    from transformers import DeepseekV3Config

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    DeepseekV3Config().to_json_file(tmp_path / "config.json")
    process = multiprocessing.get_context("spawn").Process(
        target=_construct_config_in_spawned_process, args=(str(tmp_path),)
    )
    process.start()
    try:
        process.join(timeout=60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


@pytest.fixture
def patched_runner(monkeypatch):
    from pydantic.dataclasses import rebuild_dataclass
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from skyrl.backends.skyrl_train.patches.vllm.patch_routed_experts_pp import (
        apply_routed_experts_pp_patch,
    )

    monkeypatch.setattr(VllmConfig, "__post_init__", VllmConfig.__post_init__)
    monkeypatch.setattr(GPUModelRunner, "execute_model", lambda self, *args, **kwargs: self.output)
    apply_routed_experts_pp_patch()
    apply_routed_experts_pp_patch()
    yield GPUModelRunner.execute_model
    monkeypatch.undo()
    rebuild_dataclass(VllmConfig, force=True)


def test_pp_forward_relays_routes(patched_runner):
    from vllm.sequence import IntermediateTensors
    from vllm.v1.outputs import RoutedExpertsTensors
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState

    local = torch.zeros((2, 4, 2), dtype=torch.int32)
    local[:, 2:] = 5
    earlier = torch.zeros_like(local)
    earlier[:, :2] = 3
    expected = local + earlier
    routes = RoutedExpertsTensors(routing_data=local, slot_mapping=torch.tensor([0, 1]))
    state = ExecuteModelState(
        **{name: routes if name == "routed_experts" else None for name in ExecuteModelState._fields}
    )
    original_output = IntermediateTensors({"hidden_states": torch.ones((2, 4))})
    runner = SimpleNamespace(
        use_pp=True, execute_model_state=state, routed_experts_capturer=Mock(), output=original_output
    )

    output = patched_runner(runner, None, IntermediateTensors({"routed_experts": earlier}))
    assert torch.count_nonzero(local[:, :2]) == 0
    assert torch.count_nonzero(earlier[:, 2:]) == 0
    local.zero_()
    earlier.zero_()
    torch.testing.assert_close(runner.execute_model_state.routed_experts.routing_data, expected)
    torch.testing.assert_close(output["routed_experts"], expected)
    assert "routed_experts" not in original_output.tensors
    assert output["hidden_states"] is original_output["hidden_states"]
    assert runner.execute_model_state.routed_experts.slot_mapping is routes.slot_mapping


def test_pp_route_shape_mismatch(patched_runner):
    from vllm.sequence import IntermediateTensors

    local = torch.zeros((2, 4, 2), dtype=torch.int32)
    runner = SimpleNamespace(
        use_pp=True,
        execute_model_state=SimpleNamespace(routed_experts=SimpleNamespace(routing_data=local)),
        routed_experts_capturer=Mock(),
        output=None,
    )
    previous = IntermediateTensors({"routed_experts": torch.zeros((1, 4, 2), dtype=torch.int32)})

    with pytest.raises(ValueError, match="PP routed expert shapes must match"):
        patched_runner(runner, None, previous)
