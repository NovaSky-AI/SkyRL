"""Megatron worker instrumentation for packing gradient comparisons."""

import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)


# Transformer Engine can request TF32 directly through cuBLAS. Disable it for
# FP32 controls; this does not change the packed BF16 attention/GEMM kernels.
@ray.remote(num_gpus=1, runtime_env={"env_vars": {"NVIDIA_TF32_OVERRIDE": "0"}})
class PackingParityWorker(MegatronPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps: int = 1e9):
        super().init_model(model_path, num_training_steps)
        self._grads_synced = False
        if not self.cfg.bf16:
            assert not self.provider.bf16
            assert all(
                parameter.dtype == torch.float32 for module in self.actor_module for parameter in module.parameters()
            )

    def gradient_samples(self):
        """Sample every trainable parameter after grad sync, before optimizer clipping and updates."""
        # Unsynced main_grad holds per-rank partials (e.g. sequence-parallel layernorm shards)
        # that depend on the packing layout. The sync is not idempotent, so optim_step skips it.
        self.model.run_pending_grad_sync()
        self._grads_synced = True
        samples = {}
        for chunk_index, module in enumerate(self.actor_module):
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    gradient = parameter.main_grad.flatten()
                    stride = max(1, gradient.numel() // 1024)
                    samples[f"{chunk_index}.{name}"] = gradient[::stride][:1024].float().cpu().clone()
        return samples

    def optim_step(self):
        if not self._grads_synced:
            return super().optim_step()
        run_pending_grad_sync = self.model.run_pending_grad_sync
        self.model.run_pending_grad_sync = lambda: None
        try:
            return super().optim_step()
        finally:
            self.model.run_pending_grad_sync = run_pending_grad_sync
            self._grads_synced = False
