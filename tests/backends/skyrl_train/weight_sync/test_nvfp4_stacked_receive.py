"""CPU tests for the stacked NVFP4 receiver against vLLM's layerwise-reload internals (needs the vllm wheel).

``fast_path_plan`` decides when one ``copy_`` per parameter is provably what the per-expert loader would write;
``process_layer_with_direct_copy`` runs vLLM's own layerwise materialize / kernel-tensor restore around it. The
GPU test (``gpu_ci/test_nvfp4_stacked_sync.py``) checks the end result against the per-expert path on a real
engine; these pin the decision logic and the reload bookkeeping on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="the receiver drives vLLM's layerwise reload internals")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync import nvfp4_stacked_receive as recv  # noqa: E402, I001
from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked import STACKED_LEAVES  # noqa: E402

E, N, H = 4, 16, 32


class FakeModelOptNvFp4FusedMoE:
    use_global_sf = True


def _stacks():
    gen = torch.Generator().manual_seed(0)

    def rnd(shape, dtype):
        return (
            torch.randint(0, 120, shape, generator=gen).to(torch.uint8).view(dtype)
            if dtype != torch.float32
            else (torch.rand(shape, generator=gen) + 0.5)
        )

    return {
        "w13_weight": rnd((E, 2 * N, H // 2), torch.uint8),
        "w13_weight_scale": rnd((E, 2 * N, H // 16), torch.float8_e4m3fn),
        "w13_weight_scale_2": rnd((E, 2), torch.float32),
        "w13_input_scale": torch.ones(E, 2),
        "w2_weight": rnd((E, H, N // 2), torch.uint8),
        "w2_weight_scale": rnd((E, H, N // 16), torch.float8_e4m3fn),
        "w2_weight_scale_2": rnd((E,), torch.float32),
        "w2_input_scale": torch.ones(E),
    }


class FakeExperts(torch.nn.Module):
    """Stand-in for vLLM's ModelOpt NVFP4 RoutedExperts at EP2 (this rank owns ``owned`` global experts)."""

    def __init__(self, owned=(2, 3), tp_size=1, global_sf=True, extra_param=False):
        super().__init__()
        self.global_num_experts = E
        self.local_num_experts = len(owned)
        self._g2l = {g: i for i, g in enumerate(owned)}
        self.quant_method = FakeModelOptNvFp4FusedMoE()
        self.quant_method.use_global_sf = global_sf
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=tp_size), tp_shard_with_padding=False, is_act_and_mul=True
        )
        self.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
        for leaf, t in _stacks().items():
            rows = E if (global_sf and leaf.endswith("input_scale")) else len(owned)
            p = torch.nn.Parameter(torch.zeros((rows, *t.shape[1:]), dtype=t.dtype), requires_grad=False)
            self.register_parameter(leaf, p)
        if extra_param:
            self.register_parameter("w13_bias", torch.nn.Parameter(torch.zeros(len(owned), 2 * N)))

    def _map_global_expert_id_to_local_expert_id(self, g):
        return self._g2l.get(g, -1)


def test_plan_for_ep2_rank():
    plan, why = recv.fast_path_plan(FakeExperts(), _stacks())
    assert why == ""
    index, sources = plan
    assert index.tolist() == [2, 3]
    assert sources["w13_input_scale"] == "global" and sources["w13_weight"] == "local"


@pytest.mark.parametrize(
    "module, reason",
    [
        (FakeExperts(tp_size=2), "tp_size"),
        (FakeExperts(extra_param=True), "other loadable"),
        (FakeExperts(global_sf=False), "w13_input_scale"),  # params sized local, stack global -> shapes differ
    ],
)
def test_plan_falls_back(module, reason):
    if reason == "w13_input_scale":
        module.quant_method.use_global_sf = True  # params local-sized but the method claims global scales
    plan, why = recv.fast_path_plan(module, _stacks())
    assert plan is None and reason in why


def test_plan_rejects_non_bijective_expert_map():
    module = FakeExperts()
    module._g2l = {0: 0, 1: 0}
    plan, why = recv.fast_path_plan(module, _stacks())
    assert plan is None and "expert map" in why


def test_direct_copy_through_vllm_layerwise_reload():
    """Under vLLM's real reload bookkeeping, the direct-copy layer process lands the bytes in the kernel tensors."""
    from vllm.model_executor.model_loader.reload import layerwise

    module = FakeExperts()
    kernel = {n: p for n, p in module.named_parameters()}
    layerwise.record_metadata_for_reloading(module)
    layerwise.initialize_layerwise_reload(module)
    info = layerwise.get_layerwise_info(module)
    assert info.can_load() and all(p.is_meta for p in module.parameters())

    stacks = _stacks()
    plan, why = recv.fast_path_plan(module, stacks)
    assert plan is not None, why
    recv.process_layer_with_direct_copy(module, info, lambda: recv.copy_stacks_into(module, stacks, plan))

    assert not info.can_load()  # reset: finalize_layerwise_reload will skip this layer
    for leaf in STACKED_LEAVES:
        param = getattr(module, leaf)
        assert param is kernel[leaf]  # the original (cudagraph-referenced) tensors are back in place
        expected = stacks[leaf] if leaf.endswith("input_scale") else stacks[leaf][2:]
        a, b = param.data, expected
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), leaf
    layerwise.finalize_layerwise_reload(module, SimpleNamespace(dtype=torch.bfloat16))
