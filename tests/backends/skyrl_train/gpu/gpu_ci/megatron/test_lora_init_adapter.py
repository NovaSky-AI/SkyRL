"""``trainer.policy.model.lora.init_adapter_path``: a PEFT adapter resumes exactly on Megatron.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_lora_init_adapter.py

A tiny random dense Qwen3.6-style checkpoint (hybrid GatedDeltaNet + gated attention, untied
``lm_head``, VL ``model.language_model`` namespace; see ``create_qwen36_dense_tiny``) gets a
Tinker-shaped PEFT adapter: separate ``q/k/v_proj`` adapters, ``linear_attn.in_proj_q/k/v``
(Tinker's split of HF's fused ``in_proj_qkv``) plus ``in_proj_z`` but no ``in_proj_b/a``,
``gate/up/down_proj``, ``o_proj``/``out_proj`` and ``lm_head``, all with non-zero ``lora_B``.

The reference is HuggingFace + ``peft`` on the same adapter (after the ``in_proj_qkv`` fold,
which real ``peft`` must also accept). The policy worker initialized from the adapter has to
score fixed token sequences like the reference, and unlike the base model, so a fused Megatron
module built at the summed rank with a block-diagonal ``lora_B`` (``linear_qkv`` <- q/k/v,
``in_proj`` <- qkv/z, ``linear_fc1`` <- gate/up) is checked end to end.
"""

import json
import os
import shutil
import tempfile

import pytest
import ray
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from skyrl.backends.skyrl_train.distributed.lora_import import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    PEFT_PREFIX,
    normalize_peft_adapter,
    read_peft_adapter,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.megatron_tiny_models.create_qwen36_dense_tiny import (
    create_qwen36_dense_tiny,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_lora_models import (
    _trainer_logprobs,
)
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

pytestmark = pytest.mark.megatron

MODEL_DIR = os.environ.get("SKYRL_TEST_QWEN36_TINY_DIR", "/tmp/qwen36-dense-tiny-random")
RANK = 8
ALPHA = 16
BATCH_SIZE = 4
SEQ_LEN = 48
RESPONSE_LEN = 24
# bf16 HF vs. the trainer on a random tiny model; the adapter moves logprobs by far more.
PARITY_THRESHOLD = 2e-2
ADAPTER_EFFECT_MIN = 1e-1


def _text_config(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return config.get_text_config()


def build_tinker_style_adapter(model_dir: str, out_dir: str, seed: int = 0) -> str:
    """Write a PEFT adapter with the module set and key names Tinker's converter emits."""
    text = _text_config(model_dir)
    hidden = text.hidden_size
    head_dim = text.head_dim
    q_out = text.num_attention_heads * head_dim * (2 if getattr(text, "attn_output_gate", False) else 1)
    kv_out = text.num_key_value_heads * head_dim
    attn_in = text.num_attention_heads * head_dim
    key_dim = text.linear_num_key_heads * text.linear_key_head_dim
    value_dim = text.linear_num_value_heads * text.linear_value_head_dim
    inter = text.intermediate_size

    shapes = {}  # module -> (out, in)
    for i, layer_type in enumerate(text.layer_types):
        prefix = f"model.language_model.layers.{i}"
        if layer_type == "full_attention":
            shapes[f"{prefix}.self_attn.q_proj"] = (q_out, hidden)
            shapes[f"{prefix}.self_attn.k_proj"] = (kv_out, hidden)
            shapes[f"{prefix}.self_attn.v_proj"] = (kv_out, hidden)
            shapes[f"{prefix}.self_attn.o_proj"] = (hidden, attn_in)
        else:
            shapes[f"{prefix}.linear_attn.in_proj_q"] = (key_dim, hidden)
            shapes[f"{prefix}.linear_attn.in_proj_k"] = (key_dim, hidden)
            shapes[f"{prefix}.linear_attn.in_proj_v"] = (value_dim, hidden)
            shapes[f"{prefix}.linear_attn.in_proj_z"] = (value_dim, hidden)
            shapes[f"{prefix}.linear_attn.out_proj"] = (hidden, value_dim)
        shapes[f"{prefix}.mlp.gate_proj"] = (inter, hidden)
        shapes[f"{prefix}.mlp.up_proj"] = (inter, hidden)
        shapes[f"{prefix}.mlp.down_proj"] = (hidden, inter)
    shapes["lm_head"] = (text.vocab_size, hidden)

    g = torch.Generator().manual_seed(seed)
    state = {}
    for module, (out, in_features) in shapes.items():
        state[f"{PEFT_PREFIX}{module}{LORA_A_SUFFIX}"] = (0.1 * torch.randn(RANK, in_features, generator=g)).to(
            torch.bfloat16
        )
        state[f"{PEFT_PREFIX}{module}{LORA_B_SUFFIX}"] = (0.1 * torch.randn(out, RANK, generator=g)).to(torch.bfloat16)
    config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": model_dir,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": RANK,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": sorted({m.rsplit(".", 1)[-1] for m in shapes}),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_dir


def write_normalized_adapter(adapter_dir: str, out_dir: str) -> str:
    """Write the adapter as plain PEFT (``in_proj_q/k/v`` folded into ``in_proj_qkv``)."""
    adapter = normalize_peft_adapter(read_peft_adapter(adapter_dir))
    state = {}
    for module, piece in adapter.pieces.items():
        state[f"{PEFT_PREFIX}{module}{LORA_A_SUFFIX}"] = piece.lora_a.contiguous()
        state[f"{PEFT_PREFIX}{module}{LORA_B_SUFFIX}"] = piece.lora_b.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter.config, f, indent=2)
    return out_dir


def make_inputs(model_dir: str, seed: int = 1) -> TrainingInputBatch:
    vocab = _text_config(model_dir).vocab_size
    g = torch.Generator().manual_seed(seed)
    sequences = torch.randint(0, vocab, (BATCH_SIZE, SEQ_LEN), generator=g)
    ones = torch.ones(BATCH_SIZE, RESPONSE_LEN, dtype=torch.long)
    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
            "loss_mask": ones,
            "response_mask": ones,
        }
    )
    data.metadata = {"response_length": RESPONSE_LEN}
    return data


@torch.no_grad()
def hf_response_logprobs(model_dir: str, adapter_dir: str | None, data: TrainingInputBatch) -> torch.Tensor:
    """Per-token logprobs of the last ``response_length`` tokens from HF (+ PEFT)."""
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.cuda().eval()
    sequences = data["sequences"].cuda()
    logits = model(input_ids=sequences, attention_mask=data["attention_mask"].cuda()).logits.float()
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    result = logprobs[:, -data.metadata["response_length"] :].cpu()
    del model, logits
    torch.cuda.empty_cache()
    return result


def _policy_config(strategy: str, adapter_dir: str | None) -> SkyRLTrainConfig:
    """LoRA policy config; ``adapter_dir=None`` builds the same adapters at their zero init."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = strategy
    cfg.trainer.policy.model.path = MODEL_DIR
    cfg.trainer.logger = "console"
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    # VL checkpoint: train only the language model (GPTModel + GDN thd path on Megatron).
    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.generator.inference_engine.language_model_only = True
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=RANK, alpha=ALPHA, init_adapter_path=adapter_dir)
    if strategy == "fsdp":
        cfg.trainer.policy.fsdp_config.fsdp_size = 1
    validate_cfg(cfg)
    return cfg


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


@pytest.fixture(scope="module")
def adapter_dirs():
    create_qwen36_dense_tiny(MODEL_DIR)
    root = tempfile.mkdtemp(prefix="skyrl_lora_init_adapter_")
    tinker_style = build_tinker_style_adapter(MODEL_DIR, os.path.join(root, "tinker_style"))
    normalized = write_normalized_adapter(tinker_style, os.path.join(root, "normalized"))
    yield tinker_style, normalized
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def reference(adapter_dirs):
    """(inputs, HF base logprobs, HF+adapter logprobs); the adapter must matter."""
    _, normalized = adapter_dirs
    data = make_inputs(MODEL_DIR)
    base = hf_response_logprobs(MODEL_DIR, None, data)
    adapted = hf_response_logprobs(MODEL_DIR, normalized, data)
    effect = _mean_abs(base, adapted)
    print(f"HF: adapter moves response logprobs by {effect:.4f} (mean abs)")
    assert effect > ADAPTER_EFFECT_MIN, "the test adapter must change the model noticeably"
    return data, base, adapted


def _policy_logprobs(cfg: SkyRLTrainConfig, data: TrainingInputBatch) -> torch.Tensor:
    with ray_init():
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1)
        return _trainer_logprobs(policy, data)


@pytest.mark.parametrize("strategy", ["megatron"])
def test_policy_initialized_from_peft_adapter_matches_hf(strategy, adapter_dirs, reference):
    """The imported adapter adds no error beyond the backend's own bf16 noise against HF.

    The noise floor is the same policy with fresh (zero) adapters against the HF base model;
    the imported policy must sit within a small multiple of it from HF + adapter, and far
    from the base model.
    """
    tinker_style, _ = adapter_dirs
    data, hf_base, hf_adapted = reference

    fresh = _policy_logprobs(_policy_config(strategy, None), data)
    noise_floor = _mean_abs(fresh, hf_base)

    trainer = _policy_logprobs(_policy_config(strategy, tinker_style), data)
    parity = _mean_abs(trainer, hf_adapted)
    distance_from_base = _mean_abs(trainer, hf_base)
    print(
        f"{strategy}: imported vs HF+adapter {parity:.4f}, fresh vs HF base {noise_floor:.4f}, "
        f"imported vs HF base {distance_from_base:.4f} (mean abs logprob)"
    )
    assert parity < max(
        PARITY_THRESHOLD, 2 * noise_floor
    ), f"{strategy} policy does not reproduce the adapter: {parity:.4f} (noise floor {noise_floor:.4f})"
    assert distance_from_base > ADAPTER_EFFECT_MIN, f"{strategy} policy ignored the adapter: {distance_from_base:.4f}"


# ---------------------------------------------------------------------------
# Merged weight sync (Megatron merge_lora=True, the default)
# ---------------------------------------------------------------------------


class _MergedExportPolicyWorkerBase(MegatronPolicyWorkerBase):
    def export_merged_hf_weights(self) -> dict:
        """The HF-format weights the default (merged) weight sync publishes."""
        return {
            name: tensor.detach().to("cpu")
            for name, tensor in self.bridge.export_hf_weights(self.actor_module, cpu=True, show_progress=False)
        }


_MergedExportPolicyWorker = ray.remote(num_gpus=1)(_MergedExportPolicyWorkerBase)


@torch.no_grad()
def hf_merged_state_dict(model_dir: str, adapter_dir: str) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    return {name: tensor.detach().to("cpu") for name, tensor in merged.state_dict().items()}


def test_megatron_merged_export_matches_peft_merge(adapter_dirs):
    """``merge_lora=true`` publishes base + imported adapter, like ``peft``'s ``merge_and_unload``."""
    tinker_style, normalized = adapter_dirs
    reference = hf_merged_state_dict(MODEL_DIR, normalized)
    cfg = _policy_config("megatron", tinker_style)
    with ray_init():
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, worker_cls=_MergedExportPolicyWorker)
        exported = ray.get(policy.async_run_ray_method("pass_through", "export_merged_hf_weights"))[0]

    adapted_bases = {f"{m}.weight" for m in read_peft_adapter(normalized).pieces}
    missing = sorted(adapted_bases - set(exported))
    assert not missing, f"merged export lacks adapted base weights: {missing[:8]}"
    # Only the adapted base weights carry the merge; the other tensors are the bridge's
    # plain conversion (e.g. the GDN norm kept in bf16), which this test does not pin.
    worst_other = 0.0
    for name, tensor in exported.items():
        if name not in reference:
            continue
        ref = reference[name].float()
        if name in adapted_bases:
            torch.testing.assert_close(tensor.float(), ref, rtol=2e-2, atol=2e-3, msg=lambda m, n=name: f"{n}: {m}")
        else:
            worst_other = max(worst_other, (tensor.float() - ref).abs().max().item())
    print(
        f"megatron merged export: {len(adapted_bases)} adapted base weights match peft merge; "
        f"max abs diff on the other {len(exported) - len(adapted_bases)} tensors {worst_other:.5f}"
    )


# ---------------------------------------------------------------------------
# Resumed training: the imported (widened) adapters train, the base stays frozen
# ---------------------------------------------------------------------------

# Large enough that one Adam step visibly moves bf16 adapter weights.
TRAIN_LR = 1e-3


class _TrainProbePolicyWorkerBase(MegatronPolicyWorkerBase):
    def _named_parameters(self):
        for index, chunk in enumerate(self.actor_module):
            for name, parameter in chunk.named_parameters():
                yield f"chunk{index}.{name}", parameter

    def snapshot_parameters(self) -> dict:
        """``{name: (requires_grad, cpu copy)}`` for every parameter."""
        return {
            name: (parameter.requires_grad, parameter.detach().float().cpu().clone())
            for name, parameter in self._named_parameters()
        }

    def adapter_grad_norms(self) -> dict:
        """Gradient norm per adapter parameter, read from the DDP grad buffer before ``optim_step``."""
        norms = {}
        for name, parameter in self._named_parameters():
            if ".adapter." not in name:
                continue
            grad = getattr(parameter, "main_grad", None)
            if grad is None:
                grad = parameter.grad
            norms[name] = None if grad is None else grad.float().norm().item()
        return norms

    def lora_import_state(self) -> tuple:
        """The import plan / adapter the worker still holds (both hold full CPU adapter tensors)."""
        return self._lora_import_plan, self._lora_import_adapter


_TrainProbePolicyWorker = ray.remote(num_gpus=1)(_TrainProbePolicyWorkerBase)


def _adapter_target(name: str) -> str:
    """``chunk0.module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight`` -> ``linear_qkv``."""
    return name.split(".adapter.", 1)[0].rsplit(".", 1)[-1]


def _sft_batch(data: TrainingInputBatch) -> TrainingInputBatch:
    zeros = torch.zeros(BATCH_SIZE, RESPONSE_LEN)
    batch = TrainingInputBatch(
        {
            **{key: data[key] for key in ("sequences", "attention_mask", "loss_mask", "response_mask")},
            "action_log_probs": zeros,
            "base_action_log_probs": zeros,
            "rollout_logprobs": zeros,
            "advantages": zeros,
        }
    )
    batch.metadata = {**data.metadata, "global_step": 0}
    return batch


def test_megatron_trains_imported_adapter(adapter_dirs, reference):
    """One SFT step on the imported adapter: every widened adapter (incl. ``output_layer``) gets a
    finite non-zero gradient and moves, the frozen base does not, and the trained tokens gain logprob.
    """
    tinker_style, _ = adapter_dirs
    data, _, _ = reference
    cfg = _policy_config("megatron", tinker_style)
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.policy.optimizer_config.lr = TRAIN_LR
    cfg.trainer.policy.optimizer_config.weight_decay = 0.0
    validate_cfg(cfg)

    with ray_init():
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, worker_cls=_TrainProbePolicyWorker)
        import_state = ray.get(policy.async_run_ray_method("pass_through", "lora_import_state"))[0]
        before = ray.get(policy.async_run_ray_method("pass_through", "snapshot_parameters"))[0]
        logprobs_before = _trainer_logprobs(policy, data)

        results = ray.get(
            policy.async_run_ray_method("mesh", "forward_backward", _sft_batch(data), loss_fn="cross_entropy")
        )
        grad_norms = ray.get(policy.async_run_ray_method("pass_through", "adapter_grad_norms"))[0]
        ray.get(policy.async_run_ray_method("pass_through", "optim_step"))

        after = ray.get(policy.async_run_ray_method("pass_through", "snapshot_parameters"))[0]
        logprobs_after = _trainer_logprobs(policy, data)

    assert import_state == (None, None), "the worker keeps the imported adapter tensors after loading them"
    assert all(torch.isfinite(torch.tensor(r.metrics["loss"])) for r in results), [r.metrics for r in results]

    adapter_names = sorted(name for name in before if ".adapter." in name)
    targets = {_adapter_target(name) for name in adapter_names}
    # Tinker's module set lands on these fused / renamed Megatron leaves.
    expected_targets = {"linear_qkv", "linear_proj", "in_proj", "out_proj", "linear_fc1", "linear_fc2", "output_layer"}
    assert expected_targets <= targets, f"missing adapter targets: {sorted(expected_targets - targets)}"

    not_trainable = [name for name in adapter_names if not before[name][0]]
    assert not not_trainable, f"adapter parameters frozen: {not_trainable[:8]}"
    trainable_base = [name for name, (requires_grad, _) in before.items() if requires_grad and ".adapter." not in name]
    assert not trainable_base, f"base parameters trainable under LoRA: {trainable_base[:8]}"

    bad_grads = [name for name in adapter_names if grad_norms.get(name) is None or not grad_norms[name] > 0.0]
    assert not bad_grads, f"adapter parameters without a finite non-zero gradient: {bad_grads[:8]}"
    nonfinite = [name for name in adapter_names if not torch.isfinite(torch.tensor(grad_norms[name]))]
    assert not nonfinite, f"adapter parameters with non-finite gradients: {nonfinite[:8]}"

    unchanged = [name for name in adapter_names if torch.equal(before[name][1], after[name][1])]
    assert not unchanged, f"adapter parameters the optimizer step left unchanged: {unchanged[:8]}"
    nonfinite_weights = [name for name in adapter_names if not torch.isfinite(after[name][1]).all()]
    assert not nonfinite_weights, f"adapter parameters non-finite after the step: {nonfinite_weights[:8]}"

    moved_base = [
        name for name in before if ".adapter." not in name and not torch.equal(before[name][1], after[name][1])
    ]
    assert not moved_base, f"frozen base parameters changed: {moved_base[:8]}"

    gain = (logprobs_after - logprobs_before).mean().item()
    print(
        f"megatron resumed training: {len(adapter_names)} adapter params over {sorted(targets)} trained, "
        f"{len(before) - len(adapter_names)} base params frozen; mean response logprob gain {gain:.4f}"
    )
    assert gain > 0, f"one SFT step did not raise the trained tokens' logprobs: {gain:.4f}"
