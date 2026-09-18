"""Shared vLLM layerwise-reload lifecycle for SkyRL's vLLM worker-extension classes.

Provides `LayerwiseReloadWorkerMixin`, the start/finish bracket that
`new_inference_worker_wrap.NewInferenceWorkerWrap` uses to run vLLM's layerwise
reload once per weight sync rather than once per chunk.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _empty_cuda_cache_rocm() -> None:
    """Release unused ROCm cached blocks after full-weight sync."""
    is_rocm = torch.version.hip is not None
    if not torch.cuda.is_available() or not is_rocm:
        return

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


class LayerwiseReloadWorkerMixin:
    """Bracket a multi-chunk weight sync with one vLLM layerwise-reload init/finalize.

    `skyrl_start_weight_update` initializes the layerwise reload once; each chunk then loads
    its weights raw; `skyrl_finish_weight_update` finalizes once over the whole weight set.
    A per-chunk `reload_weights` is the wrong approach: it re-finalizes on every call
    and restores layers absent from that chunk, corrupting a multi-chunk sync.
    """

    vllm_config: "VllmConfig"
    model_runner: "GPUModelRunner"
    model_config: "ModelConfig"
    device: torch.device

    # NOTE: named with a `skyrl_` prefix to avoid colliding with vLLM's own
    # Worker.start_weight_update / finish_weight_update (added in vllm-project/vllm
    # #39212, merge e3b65a5, shipped in vLLM 0.22.0+). vLLM injects the
    # worker-extension class as a *base* of Worker and asserts the extension
    # defines no attribute already present on Worker, so same-named methods abort
    # engine init. The skyrl_-prefixed variants keep SkyRL's IPC weight-sync path
    # (and the MoE set_current_vllm_config wrapping) intact alongside vLLM's native API.
    def skyrl_start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "skyrl_start_weight_update called while a weight update is "
                "already active. Call skyrl_finish_weight_update first."
            )
        if getattr(self, "_weight_update_active", False):
            raise RuntimeError("vLLM native weight update is already active. Call finish_weight_update first.")

        # MXFP8 TRT-LLM MoE prepare re-derives a fixed per-layer weight/scale
        # relocation on every sync. Replace it with a learned,
        # bitwise-validated permutation cache; falls back to the original on
        # any mismatch. No-op for non-MXFP8 wires.
        from skyrl.backends.skyrl_train.inference_servers.trtllm_moe_prepare_cache import (
            install as install_trtllm_moe_prepare_cache,
        )

        install_trtllm_moe_prepare_cache()

        if is_checkpoint_format:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True
        # vLLM's native /update_weights endpoint checks these flags before
        # calling the configured WeightTransferEngine. Mirroring them lets
        # SkyRL keep its patched layerwise start/finish while using native
        # update_weights for transports such as checkpoint-delta.
        self._is_checkpoint_format = is_checkpoint_format
        self._weight_update_active = True

    def skyrl_finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_ipc calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_start_weight_update must be called before skyrl_finish_weight_update.")

        # The sharded_rdt engine defers its GPU post-processing (scatter/quant/
        # kernel-copy) to background threads during update, so drain it here —
        # before finalize, which needs every layer fully loaded + reset. No-op
        # for the ipc/nccl engines (they process synchronously per chunk).
        engine = getattr(self, "weight_transfer_engine", None)
        if engine is not None and getattr(engine, "defers_processing", False):
            drain_pending = getattr(engine, "drain_pending", None)
            if drain_pending is not None:
                drain_pending()

        if self._skyrl_is_checkpoint_format:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        # Serialized FP8 sync ships no KV/attention scale calibration, so scales
        # are 1.0 by contract. Re-assert it after every sync: vLLM 0.26 corrupts
        # them at boot (compressed-tensors copies dummy-load placeholders
        # verbatim) and after level-2 wake (init_fp8_kv_scales resets only the
        # k/v tensors — q wakes as 0.0 and the float mirrors keep garbage),
        # which serves NaN (quantized-Q path) or silently wrong logprobs
        # (bf16-Q path) under kv_cache_dtype=fp8_*.
        # Gated on a dummy-weight boot: this mixin also serves engines started
        # from a real FP8 checkpoint, whose calibrated k/v scales must survive
        # a sync that does not carry replacements for them.
        from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
            booted_without_checkpoint_weights,
            normalize_serialized_fp8_kv_scales,
        )

        if booted_without_checkpoint_weights():
            normalize_serialized_fp8_kv_scales(self.model_runner)

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True
        self._weight_update_active = False
        self._is_checkpoint_format = True
        _empty_cuda_cache_rocm()
