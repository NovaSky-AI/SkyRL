"""Teacher providers own scoring and state, independently of feedback and loss.

Hosted teachers are a separate FSDP module on the policy worker.
- `self` scores the student
- `ema` loads a separate copy of the policy checkpoint and lerps
- `frozen` is `ema` with update_rate 0: a separate copy that never updates
"""

from abc import ABC, abstractmethod

import torch

from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    load_fsdp2_model_to_gpu,
    offload_fsdp2_model_to_cpu,
)

from .utils import response_logits


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _reshard(module):
    # Explicitly reshard all modules before reading/writing local parameter
    # shards, including the root (which can stay unsharded after a forward).
    for submodule in reversed(list(module.modules())):
        if hasattr(submodule, "reshard"):
            submodule.reshard()


def effective_student_parameters(student):
    """Yield base-model names and effective weights without mutating LoRA layers.

    One layer's delta is materialized at a time; the teacher stays adapter-free.
    PEFT computes its own scaling/orientation, including tensor parallel dispatch.
    """
    if not getattr(student, "peft_config", None):
        yield from student.named_parameters()
        return
    from peft.tuners.lora import LoraLayer

    base = student.get_base_model()
    layers = {name: module for name, module in base.named_modules() if isinstance(module, LoraLayer)}
    for name, parameter in base.named_parameters():
        if any(part.startswith("lora_") for part in name.split(".")):
            continue
        clean_name = name.replace(".base_layer.", ".")
        layer_name, _, field = clean_name.rpartition(".")
        layer = layers.get(layer_name)
        value = parameter.detach()
        if layer is not None and field == "weight" and not layer.disable_adapters:
            if layer.merged:
                raise ValueError("SDPO EMA expects unmerged student LoRA adapters")
            for adapter in layer.active_adapters:
                value = value + layer.get_delta_weight(adapter)
        yield clean_name, value


def _matching_local(value, target):
    if hasattr(value, "redistribute") and hasattr(target, "placements"):
        value = value.redistribute(device_mesh=target.device_mesh, placements=target.placements)
    return _local(value)


class TeacherProvider(ABC):
    def offload_to_cpu(self):
        """Release persistent accelerator state when rollout engines wake up."""

    def backload_to_gpu(self):
        """Restore the configured training storage after rollout engines sleep."""

    @abstractmethod
    def score(self, student, sequences, attention_mask, response_length, **model_inputs):
        """Return detached response logits in the teacher's comparison space."""

    @abstractmethod
    def after_update(self, student):
        """Called only after a successful student optimizer update."""

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state, student):
        pass


class SelfTeacher(TeacherProvider):
    @torch.no_grad()
    def score(self, student, sequences, attention_mask, response_length, **model_inputs):
        training = student.training
        try:
            student.eval()
            return response_logits(student, sequences, attention_mask, response_length, **model_inputs).detach()
        finally:
            _reshard(student)
            student.train(training)

    def after_update(self, student):
        pass

    def state_dict(self):
        return {"kind": "self"}

    def load_state_dict(self, state, student):
        if state != {"kind": "self"}:
            raise ValueError("Checkpoint teacher does not match teacher.kind=self")


class HostedTeacher(TeacherProvider):
    """Eval teacher on the policy worker; no optimizer / student clone."""

    def __init__(self, teacher_model, storage="cpu"):
        if teacher_model is None:
            raise ValueError("Hosted teacher requires a separate teacher model")
        self.teacher_model = teacher_model
        self.storage = storage
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        if storage == "cpu":
            self.offload_to_cpu()

    @property
    def weights(self):
        _reshard(self.teacher_model)
        return {
            name: _local(parameter).detach().cpu().clone() for name, parameter in self.teacher_model.named_parameters()
        }

    def offload_to_cpu(self):
        offload_fsdp2_model_to_cpu(self.teacher_model, empty_cache=torch.cuda.is_available())

    def backload_to_gpu(self):
        if self.storage != "cuda":
            return
        load_fsdp2_model_to_gpu(self.teacher_model)

    @torch.no_grad()
    def score(self, student, sequences, attention_mask, response_length, **model_inputs):
        moved = False
        if self.storage == "cpu" and sequences.device.type == "cuda":
            load_fsdp2_model_to_gpu(self.teacher_model)
            moved = True
        try:
            self.teacher_model.eval()
            return response_logits(
                self.teacher_model,
                sequences,
                attention_mask,
                response_length,
                **model_inputs,
            ).detach()
        finally:
            _reshard(self.teacher_model)
            if moved:
                offload_fsdp2_model_to_cpu(self.teacher_model, empty_cache=False)

    def after_update(self, student):
        pass


class EMATeacher(HostedTeacher):
    """Hosted copy of the policy checkpoint; local shards track an EMA of the student.

    Resume currently requires the same sharding topology.
    """

    def __init__(self, teacher_model, update_rate, storage="cpu"):
        super().__init__(teacher_model, storage)
        self.update_rate = update_rate
        self.updates = 0

    @torch.no_grad()
    def after_update(self, student):
        if student is self.teacher_model:
            raise ValueError("EMA teacher requires a separate teacher model")
        if not self.update_rate:
            return
        _reshard(self.teacher_model)
        _reshard(student)
        teacher_params = dict(self.teacher_model.named_parameters())
        seen = set()
        for name, value in effective_student_parameters(student):
            if name not in teacher_params:
                raise ValueError(f"EMA teacher has no parameter {name}")
            parameter = teacher_params[name]
            teacher_local = _local(parameter)
            source = _matching_local(value, parameter)
            if source.shape != teacher_local.shape:
                raise ValueError(f"EMA shard shape mismatch for {name}")
            teacher_local.lerp_(
                source.to(device=teacher_local.device, dtype=teacher_local.dtype),
                self.update_rate,
            )
            seen.add(name)
        if seen != teacher_params.keys():
            raise ValueError("EMA teacher parameter names must match the student")
        self.updates += 1

    def state_dict(self):
        state = {"kind": "ema", "update_rate": self.update_rate, "updates": self.updates}
        if self.update_rate:  # a resume rebuilds a frozen teacher from its path
            state["weights"] = self.weights
        return state

    def load_state_dict(self, state, student):
        if state.get("kind") != "ema" or state.get("update_rate") != self.update_rate:
            raise ValueError("Checkpoint teacher kind/update_rate differs from configuration")
        self.updates = state["updates"]
        if not self.update_rate:
            return
        _reshard(self.teacher_model)
        params = dict(self.teacher_model.named_parameters())
        weights = state["weights"]
        if weights.keys() != params.keys() or any(weights[n].shape != _local(p).shape for n, p in params.items()):
            raise ValueError("EMA checkpoint parameter shards do not match this model/topology")
        with torch.no_grad():
            for name, param in params.items():
                local = _local(param)
                local.copy_(weights[name].to(device=local.device, dtype=local.dtype))


def build_teacher(student, config, teacher_model=None):
    if config.kind == "self":
        return SelfTeacher()
    if config.kind not in ("ema", "frozen"):
        raise ValueError(f"Unsupported teacher provider: {config.kind}")
    if teacher_model is None or teacher_model is student:
        raise ValueError(f"{config.kind} teacher requires a separate teacher model")
    return EMATeacher(teacher_model, config.update_rate, config.storage)
