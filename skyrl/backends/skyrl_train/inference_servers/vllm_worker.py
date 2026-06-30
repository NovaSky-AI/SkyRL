"""
vLLM Worker Extension for SkyRL weight synchronization.

This module provides WorkerWrap, a vLLM worker extension class that enables
efficient NCCL-based and CUDA IPC-based weight updates from the training
process to inference workers.

TODO: This will be removed once vLLM natively supports weight sync APIs.
See: https://github.com/vllm-project/vllm/issues/31848

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap
"""

import warnings

import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
)

# Path to this worker extension class for use in CLI args (derived from module path)
VLLM_WORKER_EXTENSION_CLS = f"{__name__}.WorkerWrap"


class WorkerWrap(LayerwiseReloadWorkerMixin):
    """
    vLLM worker extension for SkyRL weight synchronization.

    This class is injected into vLLM workers via --worker-extension-cls and
    provides methods that can be called via engine.collective_rpc() to
    coordinate weight updates across all TP/PP workers.

    Methods:
        init_weight_update_communicator: Initialize the weight receiver
        skyrl_start_weight_update: Begin a sync; initialize vLLM layerwise reload once
        load_weights: Receive and load one chunk of weights from trainer
        skyrl_finish_weight_update: End a sync; finalize vLLM layerwise reload once
        teardown_weight_receiver: Clean up weight receiver resources
    """

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker."""
        return args, kwargs

    def init_weight_update_communicator(self, init_info: bytes):
        """
        Initialize weight update communicator from init info.

        Args:
            init_info: Pickled bytes of WeightSyncInitInfo from the sender.
        """
        import pickle

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"

        # Unpickle init_info to restore the original object type
        assert isinstance(init_info, bytes), f"Expected bytes, got {type(init_info).__name__}"
        init_info = pickle.loads(init_info)

        strategy_cls = init_info.strategy_type()

        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            # TODO(haochen): we should get rid of this flag and override existing receiver.
            if init_info.override_existing_receiver:
                self._weight_receiver.teardown()
                self._weight_receiver = None
            else:
                warnings.warn(
                    "Detected an existing weight receiver. "
                    "For overriding, use `generator.inference_engine.override_existing_update_group=enable`"
                )
                return

        self._weight_receiver = strategy_cls.create_receiver(init_info)

    def load_weights(self, request: bytes) -> None:
        """
        Load one chunk of weights using the receiver.

        Called via collective_rpc from the weight loader, once per chunk.
        When the sender brackets the sync with skyrl_start_weight_update / skyrl_finish_weight_update,
        the chunk is loaded raw and the single finalize runs vLLM's post-load weight
        processing exactly once over the whole weight set.
        Without a bracket, it falls back to a self-contained reload_weights
        (initialize + load + finalize in this one call), correct when the call
        carries the whole model so finalize sees every layer and restores none.

        Args:
            request: Pickled bytes of WeightUpdateRequest.
        """
        import pickle

        from vllm.config import set_current_vllm_config

        # Unpickle request to restore the original object type
        assert isinstance(request, bytes), f"Expected bytes, got {type(request).__name__}"
        request = pickle.loads(request)

        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        weight_update_bracketed = getattr(self, "_skyrl_weight_update_active", False)
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            if weight_update_bracketed:
                self.model_runner.model.load_weights(weights=weight_list)
            else:
                self.model_runner.reload_weights(weights_iterator=iter(weight_list))

        if weight_update_bracketed:
            # Finish consuming IPC-backed tensors before the sender drops them on
            # its next barrier; matches NewInferenceWorkerWrap.update_weights_ipc
            torch.accelerator.synchronize()

        for weight in weight_list:
            del weight

    def teardown_weight_receiver(self):
        """Clean up weight receiver resources."""
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            warnings.warn("No weight receiver to teardown")
            return
        self._weight_receiver.teardown()
