"""Shard-local EMA of effective Megatron weights, including merged LoRA deltas."""

import torch
from megatron.bridge.peft.lora_merge import LoRAMerge
from megatron.core.utils import unwrap_model


@torch.no_grad()
def effective_parameters(model):
    """Yield base names/weights, merging TP adapters using Bridge's own rules.

    Merging A/B first is essential: EMA(A) @ EMA(B) is not EMA(A @ B).
    Replicated vision LinearAdapters use the same merge with tp_size=1.
    """
    model = unwrap_model(model)
    replacements = {}
    excluded = set()
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        if hasattr(module, "to_wrap") and hasattr(module, "adapter"):
            adapter = module.adapter
            if not all(hasattr(adapter, k) for k in ("linear_in", "linear_out", "alpha", "dim")):
                raise ValueError(f"Unsupported Megatron SDPO adapter: {name}")
            replacements[prefix + "to_wrap.weight"] = (
                prefix + "weight",
                module.to_wrap.weight,
                adapter,
                None if isinstance(module.to_wrap, torch.nn.Linear) else adapter.tp_group,
            )
            excluded.update(prefix + "adapter." + n for n, _ in adapter.named_parameters())
        elif all(hasattr(module, k) for k in ("linear_in", "linear_out", "weight", "alpha", "dim")):
            replacements[prefix + "weight"] = (
                prefix + "weight",
                module.weight,
                module,
                None,
            )
            excluded.update(
                prefix + side + "." + n
                for side in ("linear_in", "linear_out")
                for n, _ in getattr(module, side).named_parameters()
            )
    for name, parameter in model.named_parameters():
        if name in excluded:
            continue
        if name in replacements:
            clean, base, adapter, tp_group = replacements[name]
            value = LoRAMerge().merge(
                base.detach().float(),
                adapter.linear_out.weight.detach().float(),
                adapter.linear_in.weight.detach().float(),
                adapter.alpha,
                adapter.dim,
                tp_group=tp_group,
            )
            yield clean, value
        else:
            yield name.replace(".to_wrap.", "."), parameter.detach()


class MegatronEMATeacher:
    """update_rate 0 is a frozen teacher: no FP32 accumulator and no weights in checkpoints."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.updates = 0
        self.model.eval().requires_grad_(False)
        self.weights = {}
        if config.update_rate:
            self.weights = {n: p.detach().float().cpu().clone() for n, p in unwrap_model(model).named_parameters()}
        if config.storage == "cpu":
            self.offload_to_cpu()

    def offload_to_cpu(self):
        self.model.cpu()

    def backload_to_gpu(self):
        self.model.cuda(torch.cuda.current_device())

    @torch.no_grad()
    def after_update(self, student):
        if not self.config.update_rate:
            return
        seen = set()
        target = dict(unwrap_model(self.model).named_parameters())
        for name, value in effective_parameters(student):
            if name not in self.weights or self.weights[name].shape != value.shape or name in seen:
                raise ValueError(f"Teacher/student parameter mismatch: {name}")
            seen.add(name)
            self.weights[name].lerp_(value.float().cpu(), self.config.update_rate)
            target[name].copy_(self.weights[name])
        if seen != self.weights.keys():
            raise ValueError(f"Missing teacher parameters: {self.weights.keys() - seen}")
        self.updates += 1

    def state_dict(self):
        return {
            "weights": self.weights,
            "updates": self.updates,
            "update_rate": self.config.update_rate,
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["update_rate"] != self.config.update_rate or state["weights"].keys() != self.weights.keys():
            raise ValueError("Incompatible Megatron SDPO teacher checkpoint")
        target = dict(unwrap_model(self.model).named_parameters())
        for name, weight in state["weights"].items():
            if weight.shape != self.weights[name].shape:
                raise ValueError(f"Teacher checkpoint shape mismatch: {name}")
            self.weights[name].copy_(weight)
            target[name].copy_(weight)
        self.updates = state["updates"]
