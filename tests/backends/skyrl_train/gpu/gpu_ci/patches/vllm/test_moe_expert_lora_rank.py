"""GPU tests for ``generator.inference_engine.scale_moe_lora_rank`` on a tiny random Qwen3-MoE checkpoint.

The adapter has rank ``LORA_RANK`` on attention and the dense MLP and rank ``LORA_RANK // TOP_K`` on
every expert, the layout ``normalize_moe_lora`` exports. It is served by an engine with vLLM's default
LoRA slots and by engines with ``scale_moe_lora_rank=true``, and compared to the same adapter merged
into the base weights.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_moe_expert_lora_rank.py -v -s
"""

import json

import pytest
import pytest_asyncio
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerHTTPError,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.gpu_ci.patches.vllm.lora_slot_probe import (
    LORA_SLOT_PROBE_WORKER_EXTENSION_CLS,
)
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

TOKENIZER = "Qwen/Qwen3-0.6B"
LORA_RANK = 16
TOP_K = 4
EXPERT_RANK = LORA_RANK // TOP_K
NUM_PROMPTS = 4
PROMPT_LEN = 48
CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    # Matches the Qwen3 tokenizer saved next to the weights.
    "vocab_size": 151936,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 3,
    # Layer 0 has a dense MLP, layers 1-2 are MoE.
    "mlp_only_layers": [0],
    "decoder_sparse_step": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": TOP_K,
    "norm_topk_prob": True,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _shapes(config: dict) -> dict[str, tuple[int, ...]]:
    hidden, head_dim = config["hidden_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * head_dim, config["num_key_value_heads"] * head_dim
    shapes = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "lm_head.weight": (config["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}"
        shapes |= {
            f"{p}.input_layernorm.weight": (hidden,),
            f"{p}.post_attention_layernorm.weight": (hidden,),
            f"{p}.self_attn.q_proj.weight": (q, hidden),
            f"{p}.self_attn.k_proj.weight": (kv, hidden),
            f"{p}.self_attn.v_proj.weight": (kv, hidden),
            f"{p}.self_attn.o_proj.weight": (hidden, q),
            f"{p}.self_attn.q_norm.weight": (head_dim,),
            f"{p}.self_attn.k_norm.weight": (head_dim,),
        }
        if i in config["mlp_only_layers"]:
            inter = config["intermediate_size"]
            shapes |= {
                f"{p}.mlp.gate_proj.weight": (inter, hidden),
                f"{p}.mlp.up_proj.weight": (inter, hidden),
                f"{p}.mlp.down_proj.weight": (hidden, inter),
            }
            continue
        inter = config["moe_intermediate_size"]
        shapes[f"{p}.mlp.gate.weight"] = (config["num_experts"], hidden)
        for e in range(config["num_experts"]):
            shapes |= {
                f"{p}.mlp.experts.{e}.gate_proj.weight": (inter, hidden),
                f"{p}.mlp.experts.{e}.up_proj.weight": (inter, hidden),
                f"{p}.mlp.experts.{e}.down_proj.weight": (hidden, inter),
            }
    return shapes


def _write_adapter(path, base: dict, expert_rank: int, generator: torch.Generator) -> dict:
    """Write a PEFT adapter with rank ``expert_rank`` on the experts; return the merged weights."""
    merged, adapter = dict(base), {}
    for name, weight in base.items():
        module = name.removesuffix(".weight")
        if module.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        rank = expert_rank if ".experts." in module else LORA_RANK
        out_features, in_features = weight.shape
        lora_a = (torch.randn(rank, in_features, generator=generator) * in_features**-0.5).bfloat16()
        # Non-null B, sized so the delta is a sizable fraction of the base weight.
        lora_b = (torch.randn(out_features, rank, generator=generator) * 0.5 * rank**-0.5).bfloat16()
        adapter[f"base_model.model.{module}.lora_A.weight"] = lora_a
        adapter[f"base_model.model.{module}.lora_B.weight"] = lora_b
        # lora_alpha == r below, so the scale is 1 on every module.
        merged[name] = weight + lora_b.float() @ lora_a.float()
    path.mkdir()
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": TARGET_MODULES,
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config))
    save_file(adapter, str(path / "adapter_model.safetensors"))
    return merged


def _write_model(path, weights: dict, tokenizer) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    save_file({k: v.bfloat16().contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    tokenizer.save_pretrained(str(path))


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Base and merged checkpoints, the normalized adapter, and one whose experts exceed the slot rank."""
    root = tmp_path_factory.mktemp("qwen3moe_tiny")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    generator = torch.Generator().manual_seed(0)
    base = {}
    for name, shape in _shapes(CONFIG).items():
        if name.endswith("norm.weight"):
            base[name] = torch.ones(shape)
        elif name == "model.embed_tokens.weight":
            base[name] = torch.randn(shape, generator=generator)
        else:
            base[name] = torch.randn(shape, generator=generator) * shape[-1] ** -0.5
    merged = _write_adapter(root / "adapter", base, EXPERT_RANK, generator)
    _write_adapter(root / "adapter_larger", base, 2 * EXPERT_RANK, generator)
    _write_model(root / "base", base, tokenizer)
    _write_model(root / "merged", merged, tokenizer)
    return {name: root / name for name in ("base", "merged", "adapter", "adapter_larger")}


@pytest.fixture(scope="module")
def prompts():
    generator = torch.Generator().manual_seed(1)
    return torch.randint(3, 1000, (NUM_PROMPTS, PROMPT_LEN), generator=generator).tolist()


def _cfg(model_path, enable_lora: bool, scale_moe_lora_rank: bool = False) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=LORA_RANK if enable_lora else 0, alpha=LORA_RANK)
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False
    cfg.trainer.policy.megatron_config.lora_config.normalize_moe_lora = True
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.num_engines = 1
    ie_cfg.run_engines_locally = True
    ie_cfg.tensor_parallel_size = 1
    ie_cfg.scale_moe_lora_rank = scale_moe_lora_rank
    return cfg


def _engines(cfg: SkyRLTrainConfig, enforce_eager: bool = True) -> InferenceEngineState:
    cfg.generator.inference_engine.enforce_eager = enforce_eager
    return InferenceEngineState.create(
        cfg=cfg,
        sleep_level=1,
        gpu_memory_utilization=0.3,
        enable_lora=cfg.trainer.policy.model.lora.rank > 0,
        engine_init_kwargs={"max_model_len": 256, "worker_extension_cls": LORA_SLOT_PROBE_WORKER_EXTENSION_CLS},
    )


async def _prompt_logprobs(client, prompts, model=None) -> torch.Tensor:
    logprobs = []
    for ids in prompts:
        body = {
            "prompt": {"chunks": [{"tokens": ids}]},
            "num_samples": 1,
            "sampling_params": {"temperature": 0.0, "max_tokens": 1},
            "include_prompt_logprobs": True,
        }
        if model is not None:
            body["model"] = model
        result = await client.sample({"json": body})
        logprobs.extend(result["prompt_logprobs"][1:])
    return torch.tensor(logprobs)


async def _slot_ranks(client) -> dict:
    responses = await client._call_all_servers("/collective_rpc", {"method": "skyrl_test_lora_slot_ranks"})
    (response,) = responses.values()
    return response["body"]["results"][0]


def _ranks(slot_ranks: dict, experts: bool) -> set:
    return set().union(*(r for name, r in slot_ranks.items() if name.endswith(".experts") == experts))


@pytest_asyncio.fixture(scope="class")
async def reference(class_scoped_ray_init_fixture, checkpoints, prompts):
    """Prompt logprobs of the base and merged checkpoints without LoRA, and of the adapter in vLLM's
    default slots (every slot at ``LORA_RANK``)."""
    out = {}
    for kind in ("base", "merged"):
        async with _engines(_cfg(checkpoints[kind], enable_lora=False)) as engines:
            out[kind] = await _prompt_logprobs(engines.client, prompts)
    async with _engines(_cfg(checkpoints["base"], enable_lora=True)) as engines:
        out["default_slot_ranks"] = await _slot_ranks(engines.client)
        await engines.client.load_lora_adapter("adapter", str(checkpoints["adapter"]))
        out["default_slots"] = await _prompt_logprobs(engines.client, prompts, model="adapter")
    return out


@pytest.mark.vllm
@pytest.mark.asyncio(loop_scope="class")
class TestScaleMoeLoRARank:
    async def test_default_slots_match_merged_weights(self, reference):
        """Baseline: the adapter is zero-padded into ``LORA_RANK`` expert slots and reproduces the merged weights."""
        assert _ranks(reference["default_slot_ranks"], experts=True) == {LORA_RANK}
        base_vs_merged = (reference["base"] - reference["merged"]).abs().mean().item()
        lora_vs_merged = (reference["default_slots"] - reference["merged"]).abs().mean().item()
        print(f"default slots: |lora - merged| {lora_vs_merged:.4f}, |base - merged| {base_vs_merged:.4f}")
        # The adapter must change the model, and LoRA must reproduce the merged weights up to bf16 rounding.
        assert base_vs_merged > 0.1
        assert lora_vs_merged < 0.1 * base_vs_merged

    @pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "cuda_graphs"])
    async def test_expert_slots_at_expert_rank_match_merged_weights(
        self, checkpoints, prompts, reference, enforce_eager
    ):
        cfg = _cfg(checkpoints["base"], enable_lora=True, scale_moe_lora_rank=True)
        async with _engines(cfg, enforce_eager) as engines:
            slot_ranks = await _slot_ranks(engines.client)
            await engines.client.load_lora_adapter("adapter", str(checkpoints["adapter"]))
            with_lora = await _prompt_logprobs(engines.client, prompts, model="adapter")

        assert _ranks(slot_ranks, experts=True) == {EXPERT_RANK}
        assert _ranks(slot_ranks, experts=False) == {LORA_RANK}
        base_vs_merged = (reference["base"] - reference["merged"]).abs().mean().item()
        lora_vs_merged = (with_lora - reference["merged"]).abs().mean().item()
        slots_diff = (with_lora - reference["default_slots"]).abs()
        print(
            f"eager={enforce_eager}: |lora - merged| {lora_vs_merged:.4f}, |base - merged| {base_vs_merged:.4f}, "
            f"|expert-rank slots - default slots| mean {slots_diff.mean().item():.2e} max {slots_diff.max().item():.2e}"
        )
        assert lora_vs_merged < 0.1 * base_vs_merged
        if enforce_eager:
            # Same as the (eager) default-slots run up to bf16 accumulation order: the MoE-LoRA kernels
            # pick their tile config by rank. CUDA graph runs are only compared to the merged weights.
            assert slots_diff.mean().item() < 2e-3

    async def test_rejects_expert_adapter_above_slot_rank(self, checkpoints):
        cfg = _cfg(checkpoints["base"], enable_lora=True, scale_moe_lora_rank=True)
        async with _engines(cfg) as engines:
            with pytest.raises(InferenceServerHTTPError, match="above the MoE expert LoRA slot rank"):
                await engines.client.load_lora_adapter("larger", str(checkpoints["adapter_larger"]))
