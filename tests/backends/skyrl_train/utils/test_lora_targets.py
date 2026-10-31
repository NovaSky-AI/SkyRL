"""Group-narrowing for LoRA target modules (Tinker's train_attn / train_mlp).

Uses a synthetic module tree rather than a real checkpoint: the names below
mirror what `Qwen3ForCausalLM` exposes, so the test stays offline and fast while
still covering the paths `resolve_lora_target_modules` classifies on. The Megatron
backend's `"all-linear"` expansion and its model checks need no model and are
tested directly.
"""

import pytest
import torch.nn as nn

from skyrl.backends.skyrl_train.utils.lora_targets import (
    check_lora_group_flags,
    check_megatron_lora_group_support,
    lora_module_group,
    megatron_all_linear_target_modules,
    resolve_critic_lora_target_modules,
    resolve_lora_target_modules,
)


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()
        self.input_layernorm = nn.LayerNorm(4)


class _Inner(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList(_Layer() for _ in range(num_layers))


class _CausalLM(nn.Module):
    """Stand-in for a decoder-only HF model, including the tied-off lm_head."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = _Inner(num_layers)
        self.lm_head = nn.Linear(4, 16, bias=False)


class _LinearExperts(nn.Module):
    """A sparse block exposing expert projections as ``nn.Linear``.

    Mirrors Mixtral's pre-transformers-5 layout (``block_sparse_moe.experts.N``).
    This is the only shape in which the ``experts`` entry of
    ``LORA_MLP_PARENT_MODULES`` decides anything -- everywhere else an ``mlp``
    ancestor already classifies the path.
    """

    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.gate = nn.Linear(4, num_experts, bias=False)  # router
        self.experts = nn.ModuleList(_Mlp() for _ in range(num_experts))


class _MoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.block_sparse_moe = _LinearExperts()


class _MoEInner(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList(_MoELayer() for _ in range(num_layers))


class _MoECausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _MoEInner()
        self.lm_head = nn.Linear(4, 16, bias=False)


ATTN_NAMES = ["k_proj", "o_proj", "q_proj", "v_proj"]
MLP_NAMES = ["down_proj", "gate_proj", "up_proj"]


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("model.layers.0.self_attn.q_proj", "attn"),
        ("model.layers.3.self_attn.o_proj", "attn"),
        ("model.layers.0.linear_attn.in_proj_k", "attn"),
        ("model.layers.0.mlp.gate_proj", "mlp"),
        ("model.layers.0.mlp.experts.0.w1", "mlp"),
        # `experts` only decides the group when it is not under an `mlp` parent:
        # the case above passes on "mlp" alone, so it cannot catch a regression here.
        ("model.layers.0.block_sparse_moe.experts.0.w1", "mlp"),
        ("model.layers.0.block_sparse_moe.gate", None),
        ("lm_head", "unembed"),
        ("model.embed_tokens", "unembed"),
        ("model.layers.0.input_layernorm", None),
        ("model.norm", None),
    ],
)
def test_lora_module_group_classification(module_name: str, expected: str | None) -> None:
    """Each module path maps to the group its parent module names, or to none."""
    assert lora_module_group(module_name) == expected


def test_both_flags_set_passes_target_modules_through() -> None:
    """The default path must stay byte-identical to pre-flag behaviour."""
    model = _CausalLM()
    assert resolve_lora_target_modules(model, "all-linear", train_attn=True, train_mlp=True) == "all-linear"
    explicit = ["q_proj"]
    assert resolve_lora_target_modules(model, explicit, train_attn=True, train_mlp=True) is explicit


def test_attn_only_selects_attention_projections() -> None:
    """train_mlp=False narrows the targets to the attention projections."""
    resolved = resolve_lora_target_modules(_CausalLM(), "all-linear", train_attn=True, train_mlp=False)
    assert resolved == ATTN_NAMES


def test_mlp_only_selects_mlp_projections() -> None:
    """train_attn=False narrows the targets to the MLP projections."""
    resolved = resolve_lora_target_modules(_CausalLM(), "all-linear", train_attn=False, train_mlp=True)
    assert resolved == MLP_NAMES


def test_groups_are_disjoint_and_cover_every_linear() -> None:
    """attn + mlp must partition the linears that "all-linear" would adapt.

    PEFT's ``all-linear`` skips the output layer, so ``lm_head`` is excluded on
    both sides -- which is why train_unembed has nothing to narrow here.
    """
    model = _CausalLM()
    attn = set(resolve_lora_target_modules(model, "all-linear", train_attn=True, train_mlp=False))
    mlp = set(resolve_lora_target_modules(model, "all-linear", train_attn=False, train_mlp=True))
    assert attn.isdisjoint(mlp)

    all_linears = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and lora_module_group(name) != "unembed"
    }
    assert attn | mlp == all_linears


def test_both_flags_false_is_rejected() -> None:
    """With both groups off there is nothing to train, so the request raises."""
    with pytest.raises(ValueError, match="at least one of `train_attn` or `train_mlp`"):
        resolve_lora_target_modules(_CausalLM(), "all-linear", train_attn=False, train_mlp=False)


def test_narrowing_an_explicit_module_list_is_rejected() -> None:
    """Narrowing applies only to the all-linear default, not to an explicit module list."""
    with pytest.raises(ValueError, match='narrow `trainer.policy.model.lora.target_modules="all-linear"`'):
        resolve_lora_target_modules(_CausalLM(), ["q_proj"], train_attn=True, train_mlp=False)


def test_narrowing_an_adapter_to_initialize_from_is_rejected() -> None:
    """The adapter at init_adapter_path decides which modules are adapted, so it cannot be narrowed."""
    check_lora_group_flags("all-linear", True, True, init_adapter_path="/tmp/adapter")
    with pytest.raises(ValueError, match="init_adapter_path"):
        check_lora_group_flags("all-linear", True, False, init_adapter_path="/tmp/adapter")


def test_model_without_the_requested_group_is_rejected() -> None:
    """An empty group raises, and the error points to target_modules as the workaround."""

    class _NoMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

    with pytest.raises(ValueError, match="No mlp linear layers found") as exc:
        resolve_lora_target_modules(_NoMlp(), "all-linear", train_attn=False, train_mlp=True)
    assert "target_modules" in str(exc.value)


def test_linear_experts_are_selected_as_mlp() -> None:
    """Expert projections join the MLP group when they are ``nn.Linear``.

    Dropping ``experts`` from LORA_MLP_PARENT_MODULES would leave this model with
    an empty MLP group, so this is the test that holds the entry in place.
    """
    resolved = resolve_lora_target_modules(_MoECausalLM(), "all-linear", train_attn=False, train_mlp=True)
    assert resolved == MLP_NAMES


def test_router_beside_the_experts_is_in_neither_group() -> None:
    """A router outside any `mlp` parent is unclassified, so narrowing leaves it out.

    A router under an `mlp` parent would be classified as MLP, as `"all-linear"` would adapt it.
    """
    model = _MoECausalLM()
    attn = resolve_lora_target_modules(model, "all-linear", train_attn=True, train_mlp=False)
    mlp = resolve_lora_target_modules(model, "all-linear", train_attn=False, train_mlp=True)
    assert "gate" not in set(attn) | set(mlp)


def test_fused_moe_experts_cannot_be_adapted() -> None:
    """Current transformers stores expert projections as fused 3-D ``nn.Parameter``.

    PEFT adapts ``nn.Linear`` modules, so those experts cannot carry an adapter at
    all. Narrowing to the MLP group then fails loudly rather than silently handing
    back an attention-only adapter. Pins the assumption against a layout change.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
        )
    )

    assert not [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and lora_module_group(name) == "mlp"
    ]
    assert any("experts" in name for name, _ in model.named_parameters())

    with pytest.raises(ValueError, match="No mlp linear layers found"):
        resolve_lora_target_modules(model, "all-linear", train_attn=False, train_mlp=True)


# Megatron-Core names written out rather than imported from lora_targets, so a wrong
# entry in MEGATRON_ALL_LINEAR_TARGETS cannot pass by being compared with itself.
@pytest.mark.parametrize(
    ("lora_type", "include_gdn_in_proj", "expected"),
    [
        ("lora", True, ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]),
        ("lora", False, ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "out_proj"]),
        (
            "canonical_lora",
            True,
            [
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_proj",
                "linear_fc1_up",
                "linear_fc1_gate",
                "linear_fc2",
                "in_proj",
                "out_proj",
            ],
        ),
    ],
)
def test_megatron_all_linear_with_both_groups_is_unchanged(
    lora_type: str, include_gdn_in_proj: bool, expected: list[str]
) -> None:
    """Both flags give the list Megatron used before the flags existed, in the same order."""
    assert megatron_all_linear_target_modules(lora_type, True, True, include_gdn_in_proj) == expected


@pytest.mark.parametrize(
    ("lora_type", "train_attn", "include_gdn_in_proj", "expected"),
    [
        ("lora", True, True, ["linear_qkv", "linear_proj", "in_proj", "out_proj"]),
        ("lora", True, False, ["linear_qkv", "linear_proj", "out_proj"]),
        ("lora", False, True, ["linear_fc1", "linear_fc2"]),
        ("lora", False, False, ["linear_fc1", "linear_fc2"]),
        ("canonical_lora", True, True, ["linear_q", "linear_k", "linear_v", "linear_proj", "in_proj", "out_proj"]),
        ("canonical_lora", False, True, ["linear_fc1_up", "linear_fc1_gate", "linear_fc2"]),
    ],
)
def test_megatron_narrowing_keeps_one_group(
    lora_type: str, train_attn: bool, include_gdn_in_proj: bool, expected: list[str]
) -> None:
    """Each narrowing keeps exactly its group's names; the GDN `in_proj` exclusion still applies."""
    names = megatron_all_linear_target_modules(lora_type, train_attn, not train_attn, include_gdn_in_proj)
    assert names == expected


def test_megatron_unknown_lora_type_is_rejected() -> None:
    """A lora_type with no name list raises instead of returning an empty target list."""
    with pytest.raises(ValueError, match="Unknown Megatron lora_type"):
        megatron_all_linear_target_modules("artur", True, True, True)


@pytest.mark.parametrize(
    ("has_vision_tower", "multi_latent_attention", "train_attn", "train_mlp", "error"),
    [
        (False, False, True, False, None),
        (False, False, False, True, None),
        (True, False, True, True, None),
        (True, False, True, False, "vision-language"),
        (True, False, False, True, "vision-language"),
        (False, True, True, True, None),
        (False, True, False, True, None),
        (False, True, True, False, "multi-latent attention"),
    ],
    ids=[
        "dense_attn_only",
        "dense_mlp_only",
        "vlm_both",
        "vlm_attn_only",
        "vlm_mlp_only",
        "mla_both",
        "mla_mlp_only",
        "mla_attn_only",
    ],
)
def test_megatron_group_support(
    has_vision_tower: bool, multi_latent_attention: bool, train_attn: bool, train_mlp: bool, error: str | None
) -> None:
    """Narrowing is rejected on a vision tower, and attention-only on MLA; everything else passes."""
    kwargs = dict(has_vision_tower=has_vision_tower, multi_latent_attention=multi_latent_attention)
    if error is None:
        check_megatron_lora_group_support(train_attn, train_mlp, **kwargs)
        return
    with pytest.raises(ValueError, match=error) as exc:
        check_megatron_lora_group_support(train_attn, train_mlp, **kwargs)
    assert "target_modules" in str(exc.value)


def test_a_name_shared_across_groups_is_rejected() -> None:
    """GPT-BigCode names both the MLP and the attention output projection `c_proj`.

    PEFT matches target names by suffix, so returning `c_proj` for the MLP group
    would also adapt `attn.c_proj`.
    """
    from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM

    model = GPTBigCodeForCausalLM(GPTBigCodeConfig(n_layer=1, n_embd=32, n_head=4, vocab_size=64))
    with pytest.raises(ValueError, match="would also adapt"):
        resolve_lora_target_modules(model, "all-linear", train_attn=False, train_mlp=True)


def test_unsupported_architectures_raise() -> None:
    """Real models outside Llama/Qwen-style naming raise rather than build a wrong or empty adapter.

    One per cause: GPT-2's projections are ``Conv1D``, GPT-NeoX's attention lives under
    `attention`, and Qwen2.5-VL has a vision tower. Each error names `target_modules`.
    """
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    tiny = dict(num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=4, vocab_size=64)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32)
    cases = [
        (GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=4, vocab_size=64)), False, "No mlp linear"),
        (GPTNeoXForCausalLM(GPTNeoXConfig(**tiny)), True, "No attn linear"),
        (
            Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(text_config=tiny, vision_config=vision)),
            True,
            "vision-language",
        ),
    ]
    for model, train_attn, match in cases:
        with pytest.raises(ValueError, match=match) as exc:
            resolve_lora_target_modules(model, "all-linear", train_attn=train_attn, train_mlp=not train_attn)
        assert "target_modules" in str(exc.value)


def test_all_linear_is_matched_in_any_case() -> None:
    """PEFT accepts the shorthand in any case, so narrowing must too."""
    resolved = resolve_lora_target_modules(_CausalLM(), "ALL-LINEAR", train_attn=True, train_mlp=False)
    assert resolved == ATTN_NAMES


class _CriticModel(nn.Module):
    """Stand-in for SkyRL's critic: a decoder with a scalar value head and no lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()
        self.value_head = nn.Linear(4, 1, bias=False)


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "group"),
    [(True, False, ATTN_NAMES), (False, True, MLP_NAMES)],
    ids=["attn_only", "mlp_only"],
)
def test_narrowed_critic_keeps_its_value_head(train_attn: bool, train_mlp: bool, group: list[str]) -> None:
    """The value head is in neither group, so it must be added back or it stays frozen."""
    assert lora_module_group("value_head") is None
    resolved = resolve_critic_lora_target_modules(_CriticModel(), "all-linear", train_attn, train_mlp, "value_head")
    assert resolved == [*group, "value_head"]


def test_default_critic_targets_are_unchanged() -> None:
    """With both flags on, the critic passes target_modules through unchanged."""
    resolved = resolve_critic_lora_target_modules(_CriticModel(), "all-linear", True, True, "value_head")
    assert resolved == "all-linear"
