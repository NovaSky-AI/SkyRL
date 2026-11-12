"""Inkling2 MXFP8 rollout experts (``SKYRL_INKLING2_ROLLOUT_EXPERTS=mxfp8``) load BF16 weights.

``Inkling2MXFP8Experts`` quantizes each BF16 expert with ``mx_cast_to_fp8`` and writes it in the
branch's layout. Its parameters and its output must be bitwise those of the branch's own MXFP8
MoE loaded (with the branch's loaders) from the same codes and scales, whether the layer arrives
whole, as this rank's experts, or one expert at a time (weight sync).

Needs a Blackwell GPU and the Inkling2 vLLM build (``pyproject.toml``), with Triton >= 3.8.

    uv run --isolated --extra dev --extra megatron pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_inkling2_mxfp8_experts.py
"""

import importlib.util
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

if importlib.util.find_spec("vllm.models.inkling2") is None:
    pytest.skip("needs a vLLM with Inkling2 (vllm.models.inkling2)", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("Inkling2 MXFP8 experts need an SM100 GPU", allow_module_level=True)

from vllm.models.inkling2.moe import Inkling2MoE  # noqa: E402

from skyrl.backends.skyrl_train.patches.vllm.inkling2.model import (  # noqa: E402
    _BF16MoE,
    _MXFP8MoE,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import (  # noqa: E402
    mx_cast_to_fp8,
)

pytestmark = pytest.mark.b200

CONFIG = SimpleNamespace(
    hidden_size=512,
    intermediate_size=256,
    n_routed_experts=16,
    n_shared_experts=0,
    num_experts_per_tok=8,
    route_scale=8.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    use_gate_bias=True,
)
SHAPES = {
    "w13": (2 * CONFIG.intermediate_size, CONFIG.hidden_size),
    "w2": (CONFIG.hidden_size, CONFIG.intermediate_size),
}


@pytest.fixture(scope="module")
def dist_init():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{path}", local_rank=0, backend="nccl"
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


def _bf16_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    weights = {
        "gate.weight": (torch.randn(CONFIG.n_routed_experts, CONFIG.hidden_size, device="cuda") * 0.05).bfloat16(),
        "gate.bias": torch.randn(CONFIG.n_routed_experts, device="cuda") * 0.01,
    }
    for name, (n, k) in SHAPES.items():
        # Spread magnitudes over blocks so the scales differ between blocks and experts.
        spread = torch.exp2(torch.randint(-6, 3, (CONFIG.n_routed_experts, n, k // 32, 1), device="cuda").float())
        w = torch.randn(CONFIG.n_routed_experts, n, k // 32, 32, device="cuda") * spread * 0.02
        weights[f"experts.{name}_weight"] = w.flatten(-2).bfloat16()
    return weights


def _branch_moe(bf16: dict[str, torch.Tensor]) -> Inkling2MoE:
    """The branch's MXFP8 MoE, loaded by its own loaders from ``mx_cast_to_fp8`` codes."""
    loaded = dict(bf16)
    for name in SHAPES:
        codes, scales = zip(*(mx_cast_to_fp8(w) for w in bf16[f"experts.{name}_weight"]))
        loaded[f"experts.{name}_weight"] = torch.stack(codes)
        loaded[f"experts.{name}_weight_scale"] = torch.stack(scales)
    with torch.device("cuda"):
        moe = Inkling2MoE(CONFIG, "mxfp8")
    for name, param in moe.named_parameters():
        param.weight_loader(param, loaded[name])
    return moe


def _rollout_moe(bf16: dict[str, torch.Tensor], how: str) -> _MXFP8MoE:
    with torch.device("cuda"):
        moe = _MXFP8MoE(CONFIG, "mxfp8")
    for name, param in moe.named_parameters():
        if name.endswith("_weight_scale"):
            continue
        weight = bf16[name]
        if not getattr(param, "skyrl_bf16_source", False):
            param.weight_loader(param, weight)
        elif how == "layer":
            param.weight_loader(param, weight)
        elif how == "local":
            param.weight_loader(param, weight, local=True)
        else:
            # Weight sync order: experts one at a time, out of order.
            for e in reversed(range(CONFIG.n_routed_experts)):
                param.weight_loader(param, weight[e], e)
    return moe


@pytest.mark.parametrize("how", ["layer", "local", "expert"])
def test_bf16_load_matches_branch_layout(dist_init, how):
    bf16 = _bf16_weights()
    ref = dict(_branch_moe(bf16).named_parameters())
    got = dict(_rollout_moe(bf16, how).named_parameters())
    assert ref.keys() == got.keys()
    for name in ref:
        assert ref[name].dtype == got[name].dtype, name
        assert torch.equal(
            ref[name].view(torch.uint8) if ref[name].element_size() == 1 else ref[name],
            got[name].view(torch.uint8) if got[name].element_size() == 1 else got[name],
        ), name


@pytest.mark.parametrize("tokens", [5, 300])
def test_forward_matches_branch_and_bf16(dist_init, tokens):
    bf16 = _bf16_weights()
    branch, rollout = _branch_moe(bf16), _rollout_moe(bf16, "expert")
    x = torch.randn(tokens, CONFIG.hidden_size, device="cuda").bfloat16()
    out = rollout(x)
    assert torch.equal(out, branch(x))

    # And close to the plugin's BF16 layer on the same BF16 weights: MXFP8 weights and
    # activations (3 mantissa bits) give a few percent relative error, a layout bug ~100%.
    with torch.device("cuda"):
        bf16_moe = _BF16MoE(CONFIG, "bf16")
    for name, param in bf16_moe.named_parameters():
        param.weight_loader(param, bf16[name]) if hasattr(param, "weight_loader") else param.data.copy_(bf16[name])
    ref = bf16_moe(x).float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.08, rel


def test_scales_are_not_loadable(dist_init):
    with torch.device("cuda"):
        moe = _MXFP8MoE(CONFIG, "mxfp8")
    param = moe.experts.w13_weight_scale
    with pytest.raises(ValueError, match="derive their scales"):
        param.weight_loader(param, torch.zeros_like(param))


@pytest.mark.parametrize("cls", ["mxfp8", "bf16"])
@pytest.mark.parametrize("tokens", [5, 300])
def test_routed_experts_capture(dist_init, cls, tokens):
    """Rollout router replay (R3): with a capture callback bound (as vLLM's routed-experts capturer
    does), each layer reports the experts it routed each token to, and its output is unchanged."""
    bf16 = _bf16_weights()
    if cls == "mxfp8":
        moe = _rollout_moe(bf16, "layer")
    else:
        with torch.device("cuda"):
            moe = _BF16MoE(CONFIG, "bf16")
        for name, param in moe.named_parameters():
            param.weight_loader(param, bf16[name]) if hasattr(param, "weight_loader") else param.data.copy_(bf16[name])
    x = torch.randn(tokens, CONFIG.hidden_size, device="cuda").bfloat16()
    expected = moe(x)

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCaptureSource,
    )

    assert isinstance(moe, RoutedExpertsCaptureSource)
    captured = []
    moe.capture_fn = captured.append
    assert torch.equal(moe(x), expected)
    (ids,) = captured
    assert ids.shape == (tokens, CONFIG.num_experts_per_tok)
    # The gate's selection: sigmoid scores plus the bias, top-k (exact ties aside).
    scores = torch.sigmoid(x.float() @ bf16["gate.weight"].float().T) + bf16["gate.bias"]
    ref = scores.topk(CONFIG.num_experts_per_tok, dim=-1).indices
    assert torch.equal(ids.long().sort(-1).values, ref.sort(-1).values)
