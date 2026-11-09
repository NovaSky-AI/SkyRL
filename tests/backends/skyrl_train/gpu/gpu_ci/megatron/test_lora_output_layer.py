"""Megatron LoRA with ``output_layer`` in ``target_modules``.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_lora_output_layer.py

``GPTModel._postprocess`` reads ``output_layer.gather_output`` and toggles
``output_layer.sequence_parallel``; Megatron-Bridge's adapter wrapper does not proxy them,
so without the passthrough the first forward raises
``'LoRALinear' object has no attribute 'gather_output'``. The test runs a forward with the
fresh (zero) adapters, perturbs only the output layer's LoRA B (every other adapter stays at
zero), and checks the logprobs move, so the output-layer adapter itself is on the compute path.
"""

import pytest
import ray

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLLoraConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_lora_models import (
    _trainer_logprobs,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models import (
    get_test_actor_config,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
    perturb_lora_b,
)

pytestmark = pytest.mark.megatron

MODEL_NAME = "Qwen/Qwen3-0.6B"


class _PerturbPolicyWorkerBase(MegatronPolicyWorkerBase):
    def perturb_output_layer_lora_b(self, multiplier: float) -> dict:
        named_parameters = (
            (f"chunk{index}.{name}", parameter)
            for index, chunk in enumerate(self.actor_module)
            for name, parameter in chunk.named_parameters()
            if "output_layer" in name
        )
        return perturb_lora_b(named_parameters, multiplier=multiplier)

    def output_layer_adapter_param_names(self) -> list:
        return sorted(
            name
            for chunk in self.actor_module
            for name, _ in chunk.named_parameters()
            if "output_layer" in name and ".adapter." in name
        )


_PerturbPolicyWorker = ray.remote(num_gpus=1)(_PerturbPolicyWorkerBase)


def test_lora_on_output_layer_changes_logprobs():
    cfg = get_test_actor_config(MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=8, alpha=16, target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "output_layer"]
    )
    validate_cfg(cfg)
    data = make_dummy_training_batch(batch_size=2, seq_len=32, num_actions=16)

    with ray_init():
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, worker_cls=_PerturbPolicyWorker)
        adapter_params = ray.get(policy.async_run_ray_method("pass_through", "output_layer_adapter_param_names"))[0]
        assert any(name.endswith("linear_in.weight") for name in adapter_params), adapter_params
        assert any(name.endswith("linear_out.weight") for name in adapter_params), adapter_params

        before = _trainer_logprobs(policy, data)
        perturbed = ray.get(policy.async_run_ray_method("pass_through", "perturb_output_layer_lora_b", 30.0))[0]
        after = _trainer_logprobs(policy, data)

    # Only the output layer's B was changed; every other adapter is still zero.
    assert perturbed["changed_tensors"] == 1, perturbed
    moved = (before - after).abs().mean().item()
    print(
        f"output_layer LoRA: {len(adapter_params)} adapter params; perturbing only its B moved logprobs by {moved:.4f}"
    )
    assert moved > 1e-3, "perturbing only the output_layer adapter must change the logprobs"
