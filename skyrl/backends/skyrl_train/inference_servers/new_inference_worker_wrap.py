"""
vLLM Worker Extension for native weight sync with chunked transfer support.

This module provides NewInferenceWorkerWrap, a vLLM worker extension that
enables chunked weight updates from training to inference using the
start/update/finish lifecycle:

    start_weight_update   ->  one or more update_weights_chunk  ->  finish_weight_update

This separates the layerwise reload initialization/finalization from individual
chunk transfers, allowing weights to be sent in bounded-memory chunks rather
than all at once.

Used only with the new inference path (_SKYRL_USE_NEW_INFERENCE=1).

TODO: Once https://github.com/vllm-project/vllm/pull/39212 lands, vLLM will
natively support start_weight_update / update_weights / finish_weight_update
on GPUWorker with dedicated HTTP endpoints. At that point this worker extension
can be removed and SkyRL can call the native endpoints directly instead of
routing through /collective_rpc.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

import torch

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap:
    """
    vLLM worker extension for chunked weight sync (new inference path).

    Provides a three-phase weight update protocol via collective_rpc:
        1. start_weight_update: Prepare model for receiving weights
        2. update_weights_chunk: Receive and load one chunk of weights
        3. finish_weight_update: Finalize the model after all chunks

    Attributes accessed from the host GPUWorker (via mixin inheritance):
        self.weight_transfer_engine
        self.model_runner
        self.model_config
        self.device
    """

    def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_chunk calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )

        if is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True

    def update_weights_chunk(self, update_info: dict) -> None:
        """
        Receive and load a single chunk of weights.

        Delegates to the weight transfer engine's receive_weights, using
        model.load_weights for checkpoint format or direct param.copy_ for
        kernel format. Can be called multiple times between start and finish.

        Args:
            update_info: Dict with backend-specific update info (names,
                dtypes, shapes, IPC handles or NCCL packed flag, etc.)
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights_chunk.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. " "Please set weight_transfer_config to enable weight transfer."
            )

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with torch.device(self.device):
            if self._skyrl_is_checkpoint_format:
                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=model.load_weights,
                )
            else:

                def load_weights_direct(
                    weights: list[tuple[str, torch.Tensor]],
                ) -> None:
                    for name, weight in weights:
                        param = model.get_parameter(name)
                        param.copy_(weight)

                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=load_weights_direct,
                )

        torch.accelerator.synchronize()

    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_chunk calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        if self._skyrl_is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True
