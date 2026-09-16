from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_nccl.plan import (
    build_lora_nccl_plan,
    pack_lora_nccl_bucket_into,
    unpack_lora_nccl_bucket,
)
from skyrl.backends.skyrl_train.weight_sync.lora_nccl.transport import (
    _LoRAConsumerAssembler,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    build_lora_consumer_plan,
)

pytest.importorskip("vllm", reason="LoRA scaling contracts target vLLM LoRA layers")
pytestmark = pytest.mark.vllm


def test_bridge_export_to_vllm_activation_preserves_exact_normalized_deltas(monkeypatch, tmp_path):
    pytest.importorskip("transformer_engine.pytorch", reason="Requires the Megatron runtime extra")
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import (
        AdapterWeightConversionTask,
        MegatronModelBridge,
        WeightConversionTask,
    )
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.config.lora import LoRAConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.models.interfaces import SupportsLoRA

    class BridgeForTest(MegatronModelBridge):
        def provider_bridge(self, hf_pretrained):
            raise NotImplementedError

        def mapping_registry(self):
            return MegatronMappingRegistry()

    class LocalMapping:
        tp_rank = 0
        tp_size = 1
        ep_rank = 0
        ep_size = 1
        is_expert = False

        def maybe_dequantize(self, tensor):
            return tensor

    class ReceiverModel(torch.nn.Module, SupportsLoRA):
        packed_modules_mapping = {}

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(architectures=["ScalingContractModel"])
            self.dense_proj = ReplicatedLinear(8, 6, bias=False, disable_tp=True)
            self.shared_expert_proj = ReplicatedLinear(8, 6, bias=False, disable_tp=True)
            self.routed_expert_proj = ReplicatedLinear(8, 6, bias=False, disable_tp=True)
            self.nonintegral_proj = ReplicatedLinear(8, 6, bias=False, disable_tp=True)

    alpha = 16
    configured_rank = 32
    effective_ranks = {
        "dense_proj": 32,
        "shared_expert_proj": 32,
        "routed_expert_proj": 4,
        "nonintegral_proj": 7,
    }
    source_factors = {}
    tasks = []
    for index, (name, effective_rank) in enumerate(effective_ranks.items()):
        a = (torch.arange(effective_rank * 8, dtype=torch.float32).reshape(effective_rank, 8) + 3 + index) / 100
        b = (torch.arange(6 * effective_rank, dtype=torch.float32).reshape(6, effective_rank) + 7 + index) / 120
        source_factors[name] = (a, b)
        mapping = LocalMapping()
        tasks.append(
            AdapterWeightConversionTask(
                global_base_prefix=name,
                adapter_key=None,
                alpha=alpha,
                dim=effective_rank,
                linear_in_task=WeightConversionTask(
                    param_name=f"{name}.linear_in",
                    global_param_name=f"{name}.linear_in",
                    mapping=mapping,
                    param_weight=a,
                ),
                linear_out_task=WeightConversionTask(
                    param_name=f"{name}.linear_out",
                    global_param_name=f"{name}.linear_out",
                    mapping=mapping,
                    param_weight=b,
                ),
            )
        )

    bridge = BridgeForTest()
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *args, **kwargs: {"adapter": tasks},
    )
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda _registry, prefix, _adapter_key, _suffix: [f"base_model.model.{prefix}.weight"],
    )
    records = list(
        AutoBridge.export_local_adapter_weights(
            SimpleNamespace(_model_bridge=bridge),
            [SimpleNamespace(config=SimpleNamespace())],
        )
    )
    source_tensors, sources = extract_lora_bridge_sources(records, configured_rank=configured_rank)
    source_layout = LoRABridgeSourceLayout("adapter", sources)
    source_before = {key: tensor.clone() for key, tensor in source_tensors.items()}

    def init_cpu_punica(self, *_args):
        self.supports_mm = False
        self.punica_wrapper_mapping = {"language_model": object()}

    monkeypatch.setattr(LoRAModelManager, "_init_punica_wrapper", init_cpu_punica)
    monkeypatch.setattr("vllm.lora.model_manager.PIN_MEMORY", False)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=(tmp_path / "gloo-init").as_uri(),
            backend="gloo",
        )
        initialize_model_parallel(1, 1, backend="gloo")
        try:
            manager = LoRAModelManager(
                ReceiverModel(),
                1,
                configured_rank,
                32,
                LoRAConfig(
                    max_lora_rank=configured_rank,
                    max_cpu_loras=2,
                    max_loras=2,
                    lora_dtype=torch.bfloat16,
                ),
                torch.device("cpu"),
                config,
            )
            receiver_plan = manager.get_local_adapter_plan(
                PEFTHelper(
                    r=configured_rank,
                    lora_alpha=alpha,
                    target_modules=list(effective_ranks),
                )
            )
            consumer_plan = build_lora_consumer_plan(source_layout, receiver_plan)
            transport_plan = build_lora_nccl_plan({0: consumer_plan}, 1 << 20)
            assembler = _LoRAConsumerAssembler(consumer_plan, "cpu")
            for bucket in transport_plan.buckets:
                packed = torch.empty(bucket.source_bytes // 4, dtype=torch.float32)
                pack_lora_nccl_bucket_into(bucket, source_tensors, packed)
                for pull, received in unpack_lora_nccl_bucket(bucket, packed).items():
                    exact = source_tensors[pull.source_slice.key][pull.source_slice.indices]
                    assert torch.equal(received.view(torch.uint8), exact.contiguous().view(torch.uint8))
                    received_before = received.clone()
                    assembler.copy(pull, received)
                    assert torch.equal(received.view(torch.uint8), received_before.view(torch.uint8))
            factors = assembler.finish()

            source_pointers = {tensor.untyped_storage().data_ptr() for tensor in source_tensors.values()}
            for name, effective_rank in effective_ranks.items():
                component_scales = {
                    source.component: source.value_scale
                    for source in source_layout.sources
                    if any(f".{name}." in hf_name for hf_name in source.hf_param_names)
                }
                assert component_scales == {
                    "linear_in": (1, 1),
                    "linear_out": (
                        Fraction(configured_rank, effective_rank).numerator,
                        Fraction(configured_rank, effective_rank).denominator,
                    ),
                }
                staged_a, staged_b = (component[0] for component in factors[name])
                source_a, source_b = source_factors[name]
                torch.testing.assert_close(
                    staged_a[:effective_rank],
                    source_a.to(torch.bfloat16),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    staged_b[:, :effective_rank],
                    (source_b * (configured_rank / effective_rank)).to(torch.bfloat16),
                    rtol=0,
                    atol=0,
                )
                assert torch.count_nonzero(staged_a[effective_rank:]) == 0
                assert torch.count_nonzero(staged_b[:, effective_rank:]) == 0
                assert staged_a.untyped_storage().data_ptr() not in source_pointers
                assert staged_b.untyped_storage().data_ptr() not in source_pointers

            assert manager.add_local_adapter(1, receiver_plan, factors)
            assert manager.activate_adapter(1)
            for name, effective_rank in effective_ranks.items():
                active_a, active_b = manager.modules[name]._get_lora_shard_buffers(0)[0]
                actual_delta = active_b[:, :configured_rank].float() @ active_a[:configured_rank].float()
                source_a, source_b = source_factors[name]
                trainer_delta = (alpha / effective_rank) * (source_b @ source_a)
                torch.testing.assert_close(actual_delta, trainer_delta, rtol=0.02, atol=0.02)

            routed_a, routed_b = source_factors["routed_expert_proj"]
            correction = configured_rank / 4
            trainer_delta = (alpha / 4) * (routed_b @ routed_a)
            omitted = (alpha / configured_rank) * (routed_b @ routed_a)
            doubled = (alpha / configured_rank) * ((routed_b * correction * correction) @ routed_a)
            assert torch.linalg.vector_norm(trainer_delta) / torch.linalg.vector_norm(omitted) == pytest.approx(8)
            assert torch.linalg.vector_norm(doubled) / torch.linalg.vector_norm(trainer_delta) == pytest.approx(8)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()

    for key, tensor in source_tensors.items():
        assert tensor.dtype is torch.float32
        assert torch.equal(tensor.view(torch.uint8), source_before[key].view(torch.uint8))
