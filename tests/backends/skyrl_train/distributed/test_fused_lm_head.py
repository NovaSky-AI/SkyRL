import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.fused_lm_head import (
    call_model_with_fused_lm_head,
    fused_lm_head_output_processor,
)


@pytest.fixture
def hybrid_model_type(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    fp8_utils = types.ModuleType("megatron.core.fp8_utils")
    models = types.ModuleType("megatron.core.models")
    hybrid = types.ModuleType("megatron.core.models.hybrid")
    hybrid_model = types.ModuleType("megatron.core.models.hybrid.hybrid_model")
    pipeline_parallel = types.ModuleType("megatron.core.pipeline_parallel")
    activation_offload = types.ModuleType("megatron.core.pipeline_parallel.fine_grained_activation_offload")
    utils = types.ModuleType("megatron.core.utils")

    class HybridModel:
        def forward(self): ...

    class OffloadInterface:
        @staticmethod
        def mark_not_offloadable(parameter):
            parameter.offloading_activation = False

    hybrid_model.HybridModel = HybridModel
    activation_offload.FineGrainedActivationOffloadingInterface = OffloadInterface
    fp8_utils.is_mxfp8_output_proj_active = lambda config: config.mxfp8_output_projection
    utils.unwrap_model = lambda model: model.module
    megatron.core = core
    core.models = models
    core.pipeline_parallel = pipeline_parallel
    models.hybrid = hybrid
    hybrid.hybrid_model = hybrid_model
    pipeline_parallel.fine_grained_activation_offload = activation_offload
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.fp8_utils", fp8_utils)
    monkeypatch.setitem(sys.modules, "megatron.core.models", models)
    monkeypatch.setitem(sys.modules, "megatron.core.models.hybrid", hybrid)
    monkeypatch.setitem(sys.modules, "megatron.core.models.hybrid.hybrid_model", hybrid_model)
    monkeypatch.setitem(sys.modules, "megatron.core.pipeline_parallel", pipeline_parallel)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.pipeline_parallel.fine_grained_activation_offload",
        activation_offload,
    )
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    return HybridModel


def _make_hybrid(
    hybrid_model_type,
    *,
    post_process=True,
    mxfp8_output_projection=False,
    use_mup=False,
    fine_grained_activation_offloading=False,
    mtp_num_layers=None,
):
    model = hybrid_model_type()
    model.post_process = post_process
    model.config = SimpleNamespace(
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        mtp_num_layers=mtp_num_layers,
        mxfp8_output_projection=mxfp8_output_projection,
        use_mup=use_mup,
    )
    model.mtp_process = mtp_num_layers is not None
    model.share_embeddings_and_output_weights = False
    output_parameter = SimpleNamespace(offloading_activation=True)
    model.output_layer = SimpleNamespace(weight="lm-head-weight", parameters=lambda: iter((output_parameter,)))
    return model


def test_hybrid_processes_hidden_states_and_restores_post_process(hybrid_model_type):
    hybrid = _make_hybrid(hybrid_model_type, fine_grained_activation_offloading=True)
    hybrid.share_embeddings_and_output_weights = True
    hybrid.shared_embedding_or_output_weight = Mock(return_value="shared-weight")
    wrapped_model = Mock(module=hybrid, side_effect=lambda *_args, **_kwargs: "hidden-states")
    processor = Mock(return_value="processed")

    output = call_model_with_fused_lm_head(
        wrapped_model,
        "input",
        output_processor=processor,
        output_processor_context={},
        packed_seq_params="packed",
    )

    assert output == "processed"
    assert hybrid.post_process is True
    assert next(hybrid.output_layer.parameters()).offloading_activation is False
    wrapped_model.assert_called_once_with("input", packed_seq_params="packed")
    processor.assert_called_once_with(
        hidden_states="hidden-states",
        output_layer=hybrid.output_layer,
        output_weight="shared-weight",
        context={},
    )


def test_hybrid_pipeline_stage_skips_output_processor(hybrid_model_type):
    hybrid = _make_hybrid(hybrid_model_type, post_process=False)
    wrapped_model = Mock(module=hybrid, return_value="pipeline-hidden-states")
    processor = Mock()

    output = call_model_with_fused_lm_head(
        wrapped_model,
        "input",
        output_processor=processor,
        output_processor_context={},
    )

    assert output == "pipeline-hidden-states"
    wrapped_model.assert_called_once_with("input")
    processor.assert_not_called()


def test_native_model_uses_output_processor_hook(hybrid_model_type):
    class NativeInnerModel:
        config = SimpleNamespace(mxfp8_output_projection=False, use_mup=False)

        def forward(self, *, output_processor=None, output_processor_context=None): ...

    processor = Mock()
    context = {}
    model = Mock(module=NativeInnerModel(), return_value="native-output")

    output = call_model_with_fused_lm_head(
        model,
        "input",
        output_processor=processor,
        output_processor_context=context,
    )

    assert output == "native-output"
    model.assert_called_once_with(
        "input",
        output_processor=processor,
        output_processor_context=context,
    )


def test_hybrid_restores_post_process_when_forward_fails(hybrid_model_type):
    hybrid = _make_hybrid(hybrid_model_type)
    wrapped_model = Mock(module=hybrid, side_effect=RuntimeError("forward failed"))

    with pytest.raises(RuntimeError, match="forward failed"):
        call_model_with_fused_lm_head(
            wrapped_model,
            "input",
            output_processor=Mock(),
            output_processor_context={},
        )

    assert hybrid.post_process is True


@pytest.mark.parametrize(
    "mxfp8,use_mup,mtp_num_layers,match",
    [
        (True, False, None, "MXFP8"),
        (False, True, None, "MuP"),
        (False, False, 1, "MTP"),
    ],
)
def test_hybrid_rejects_unsupported_output_paths(hybrid_model_type, mxfp8, use_mup, mtp_num_layers, match):
    hybrid = _make_hybrid(
        hybrid_model_type,
        mxfp8_output_projection=mxfp8,
        use_mup=use_mup,
        mtp_num_layers=mtp_num_layers,
    )
    wrapped_model = Mock(module=hybrid)

    with pytest.raises(NotImplementedError, match=match):
        call_model_with_fused_lm_head(
            wrapped_model,
            "input",
            output_processor=Mock(),
            output_processor_context={},
        )
    wrapped_model.assert_not_called()


@pytest.mark.parametrize(
    "sequence_parallel,allreduce_dgrad",
    [(True, False), (False, True)],
)
def test_fused_lm_head_uses_tensor_parallel_grad_path(monkeypatch, sequence_parallel, allreduce_dgrad):
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    gather = Mock(side_effect=lambda tensor, **_kwargs: tensor)
    copy = Mock(side_effect=lambda tensor, **_kwargs: tensor)
    tensor_parallel.gather_from_sequence_parallel_region = gather
    tensor_parallel.copy_to_tensor_model_parallel_region = copy
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)

    hidden_states = torch.ones((2, 1, 3))
    output_layer = SimpleNamespace(
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=sequence_parallel,
        tp_group="tp-group",
        weight=torch.ones(1),
    )

    output = fused_lm_head_output_processor(hidden_states=hidden_states, output_layer=output_layer, context={})

    assert torch.equal(output, hidden_states.transpose(0, 1))
    if sequence_parallel:
        gather.assert_called_once_with(
            hidden_states,
            tensor_parallel_output_grad=True,
            group="tp-group",
        )
        copy.assert_not_called()
    else:
        copy.assert_called_once_with(hidden_states, group="tp-group")
        gather.assert_not_called()
