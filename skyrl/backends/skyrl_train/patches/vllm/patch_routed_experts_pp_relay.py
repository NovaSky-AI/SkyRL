"""Runtime patch: routed-experts capture (R3) with vLLM pipeline parallelism.

vLLM rejects ``--enable-return-routed-experts`` with ``pipeline_parallel_size > 1``: every PP stage
captures only the MoE layers it owns, and only the last stage's capture buffer reaches the output,
so the routes of every earlier stage's layers would be lost. This patch relays the accumulated
routes of layers ``[0, end_layer)`` from each stage to the next inside the PP
``IntermediateTensors`` (key ``"routed_experts"``). The receiving stage installs them in its
capture buffer before its forward pass, so the last stage holds every layer and the unchanged
AuxOutput path returns them.

It targets vLLM builds with the AuxOutput connector (vLLM >= 0.31; Model Runner V2 only, which
AuxOutput requires). It is a no-op when the installed vLLM already relays routes natively.

Install points:
  * :func:`apply_pp_relay_config_patch` in the process that builds the ``VllmConfig`` (the server
    actor), before the engine config is created: drops only the PP > 1 rejection.
  * :func:`apply_pp_relay_worker_patch` in every worker process, before the model runner creates
    the AuxOutput connector (``new_inference_worker_wrap`` imports it at module import time).

Remove once the pinned vLLM includes the upstream relay.
"""

from __future__ import annotations

import functools
from typing import Any

from loguru import logger

PP_ROUTED_EXPERTS_KEY = "routed_experts"
# Set by the server actor when R3 runs with PP > 1. vLLM processes inherit it, and the `vllm.general_plugins`
# entry point below reads it to install the config patch in each of them.
PP_RELAY_ENV = "SKYRL_R3_PP_RELAY"
_CONFIG_PATCHED = False
_WORKER_PATCHED = False


def _relay_status() -> str:
    """``"native"`` if vLLM relays routes itself, ``"patchable"`` if this module can add the relay,
    ``"unsupported"`` otherwise (no AuxOutput connector, e.g. vLLM 0.30.x, or vLLM missing)."""
    try:
        from vllm.config.vllm import VllmConfig
        from vllm.distributed.aux_output_connector.worker import (
            AuxOutputWorkerConnector,
        )
        from vllm.model_executor.layers.fused_moe import routed_experts_capturer
    except ImportError:
        return "unsupported"
    if hasattr(AuxOutputWorkerConnector, "relay_pp_routes"):
        return "native"
    if not hasattr(VllmConfig, "_verify_aux_output_compatibility") or not hasattr(
        routed_experts_capturer, "RoutedExpertsCapturer"
    ):
        return "unsupported"
    return "patchable"


def vllm_supports_routed_experts_with_pp() -> bool:
    """True if R3 works with inference PP > 1: natively, or through this patch."""
    return _relay_status() in ("native", "patchable")


# ----------------------------------------------------------------------------- config
def apply_pp_relay_config_patch() -> bool:
    """Drop vLLM's PP > 1 rejection for R3, keeping every other AuxOutput check.

    ``VllmConfig._verify_aux_output_compatibility`` reads ``pipeline_parallel_size`` only for that
    rejection, so the wrapper presents PP=1 to it for the duration of the call.
    """
    global _CONFIG_PATCHED
    if _CONFIG_PATCHED or _relay_status() != "patchable":
        return False
    from vllm.config.vllm import VllmConfig

    original = VllmConfig._verify_aux_output_compatibility

    @functools.wraps(original)
    def _verify_aux_output_compatibility(self) -> None:
        parallel_config = self.parallel_config
        pp_size = parallel_config.pipeline_parallel_size
        if pp_size == 1:
            return original(self)
        parallel_config.pipeline_parallel_size = 1
        try:
            return original(self)
        finally:
            parallel_config.pipeline_parallel_size = pp_size

    VllmConfig._verify_aux_output_compatibility = _verify_aux_output_compatibility
    _CONFIG_PATCHED = True
    logger.info("Installed the R3 pipeline-parallel config patch")
    return True


def vllm_general_plugin() -> None:
    """``vllm.general_plugins`` entry point: install the config patch in every vLLM process.

    The engine core re-runs ``VllmConfig.__post_init__`` (and so ``_verify_aux_output_compatibility``) in
    its own process at the end of its startup handshake, so patching only the process that builds the
    config is not enough. vLLM loads general plugins in each process, including the engine core, before
    that check. A no-op unless the server actor set ``SKYRL_R3_PP_RELAY=1``.
    """
    import os

    if os.environ.get(PP_RELAY_ENV) == "1":
        apply_pp_relay_config_patch()


# ----------------------------------------------------------------------------- worker
def _get_pp_routes(self, num_tokens: int) -> Any:
    """Routes of layers ``[0, end_layer)`` for the next stage, as a contiguous int32 copy (the
    capture buffer is overwritten by the next step while the send may still be in flight)."""
    import torch

    return self.device_buffer[:num_tokens, : self.end_layer].clone(memory_format=torch.contiguous_format)


def _load_pp_routes(self, routes: Any) -> None:
    """Install the previous stages' routes (layers ``[0, start_layer)``) before this stage's
    forward pass. The copy is enqueued on the current stream, which also runs the forward pass or
    replays its CUDA graph, so this stage's own layers are written after it."""
    if routes.dtype != self.device_buffer.dtype:
        raise TypeError(f"pipeline-parallel routed experts must be {self.device_buffer.dtype}, got {routes.dtype}")
    num_tokens, num_layers, top_k = routes.shape
    if num_layers != self.start_layer or top_k != self.device_buffer.shape[2]:
        raise ValueError(
            f"pipeline-parallel routed experts cover {num_layers} layers x top-{top_k}, expected the "
            f"{self.start_layer} layers before this stage's start layer x top-{self.device_buffer.shape[2]}"
        )
    if num_tokens > self.device_buffer.shape[0]:
        raise ValueError(
            f"pipeline-parallel routed experts have {num_tokens} rows, capture buffer holds "
            f"{self.device_buffer.shape[0]}"
        )
    self.device_buffer[:num_tokens, : self.start_layer].copy_(routes)


def _patch_capturer() -> None:
    import vllm.distributed.aux_output_connector.worker as aux_worker
    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec
    from vllm.distributed.utils import get_pp_indices

    cls = rec.RoutedExpertsCapturer
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Global layer range of this stage: the partition ``make_layers`` uses
        # (including VLLM_PP_LAYER_PARTITION).
        num_layers = self.device_buffer.shape[1]
        pp_group = parallel_state.get_pp_group()
        self.is_pipeline_stage = pp_group.world_size > 1
        if self.is_pipeline_stage:
            self.start_layer, self.end_layer = get_pp_indices(num_layers, pp_group.rank_in_group, pp_group.world_size)
        else:
            self.start_layer, self.end_layer = 0, num_layers

    cls.__init__ = __init__
    cls.get_pp_routes = _get_pp_routes
    cls.load_pp_routes = _load_pp_routes

    original_bind = rec.bind_routed_experts_capturer

    @functools.wraps(original_bind)
    def bind_routed_experts_capturer(model, capturer):
        if getattr(capturer, "is_pipeline_stage", False) is not True:
            return original_bind(model, capturer)
        from vllm.model_executor.layers.fused_moe.layer import MoERunner

        # The relay forwards layers by position, so every captured layer must lie in this
        # stage's range; a model that partitions differently fails here, at startup.
        for module in model.modules():
            if isinstance(module, (rec.RoutedExpertsCaptureSource, MoERunner)):
                if not capturer.start_layer <= module.layer_id < capturer.end_layer:
                    raise ValueError(
                        f"Routed-experts layer {module.layer_id} is outside this pipeline stage's layer "
                        f"range [{capturer.start_layer}, {capturer.end_layer})."
                    )
        try:
            return original_bind(model, capturer)
        except ValueError as e:
            if "No supported MoE router found" not in str(e):
                raise
            # A stage may hold only dense layers; it still relays the earlier stages' routes.
            logger.info(
                f"No MoE router in pipeline stage layers [{capturer.start_layer}, {capturer.end_layer}); "
                "relaying only"
            )

    # The connector module imports the symbol by name.
    rec.bind_routed_experts_capturer = bind_routed_experts_capturer
    aux_worker.bind_routed_experts_capturer = bind_routed_experts_capturer


def _patch_connector() -> None:
    """Only TP rank 0 of the last PP stage owns the AuxOutput data plane (store, host buffer).
    Earlier stages never produce output; drop the store they would otherwise keep."""
    import vllm.distributed.parallel_state as parallel_state
    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

    original_init = AuxOutputWorkerConnector.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not parallel_state.get_pp_group().is_last_rank and self._store is not None:
            self.close()
            self._buffer = None

    AuxOutputWorkerConnector.__init__ = __init__


def _install_receive(intermediate_tensors: Any, capturer: Any, num_tokens: int) -> None:
    """Load the relayed routes on the stream that runs the forward pass, without forcing the
    pending receive early.

    For an ``AsyncIntermediateTensors`` whose receive is still in flight, the load is appended to
    its ``_comm_postprocess``: it runs at the first ``.tensors`` access in the model runner (after
    the receive handles are waited and the TP all-gather is done), before the forward pass. The
    dict is read with ``object.__getattribute__`` because ``.tensors`` would re-enter the wait.
    """

    def load(tensors: dict) -> None:
        routes = tensors.pop(PP_ROUTED_EXPERTS_KEY, None)
        assert routes is not None, "previous pipeline stage did not relay routed experts"
        assert (
            routes.shape[0] == num_tokens
        ), f"relayed routed experts cover {routes.shape[0]} tokens, expected {num_tokens}"
        capturer.load_pp_routes(routes)

    get = object.__getattribute__
    try:
        waited = get(intermediate_tensors, "_comm_waited")
    except AttributeError:
        waited = True
    if waited:
        load(intermediate_tensors.tensors)
        return
    postprocess = get(intermediate_tensors, "_comm_postprocess")
    if postprocess is None:
        postprocess = []
        object.__setattr__(intermediate_tensors, "_comm_postprocess", postprocess)
    postprocess.append(lambda: load(get(intermediate_tensors, "tensors")))


def _patch_model_runner() -> None:
    import vllm.distributed.parallel_state as parallel_state
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    @functools.wraps(original)
    def execute_model(self, scheduler_output, intermediate_tensors=None, *args, **kwargs):
        connector = getattr(self, "aux_output_connector", None)
        pp_group = parallel_state.get_pp_group()
        active = connector is not None and pp_group.world_size > 1 and not kwargs.get("dummy_run", False)
        if not active:
            return original(self, scheduler_output, intermediate_tensors, *args, **kwargs)
        capturer = connector._capturer
        num_tokens = scheduler_output.total_num_scheduled_tokens
        if intermediate_tensors is not None:
            _install_receive(intermediate_tensors, capturer, num_tokens)
        output = original(self, scheduler_output, intermediate_tensors, *args, **kwargs)
        if isinstance(output, IntermediateTensors) and not pp_group.is_last_rank:
            # A new object: the runner's output may alias CUDA graph output buffers.
            assert PP_ROUTED_EXPERTS_KEY not in output.tensors
            output = IntermediateTensors(output.tensors | {PP_ROUTED_EXPERTS_KEY: capturer.get_pp_routes(num_tokens)})
        return output

    GPUModelRunner.execute_model = execute_model


def apply_pp_relay_worker_patch() -> bool:
    """Install the relay in this worker process. Idempotent; returns True if installed."""
    global _WORKER_PATCHED
    if _WORKER_PATCHED or _relay_status() != "patchable":
        return False
    _patch_capturer()
    _patch_connector()
    _patch_model_runner()
    _WORKER_PATCHED = True
    logger.info("Installed the R3 pipeline-parallel relay patch")
    return True
