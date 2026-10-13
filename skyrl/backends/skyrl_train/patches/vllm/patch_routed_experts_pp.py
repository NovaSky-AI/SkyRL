"""Runtime patch for PP routed expert collection in vLLM 0.30 Model Runner V2.

Each stage captures into the global layer layout, with zeros for other stages.
Forward the stable snapshots alongside hidden_states/residual through the
existing PP send/recv. Each stage adds its disjoint layer routes, and the final stage
copies the complete snapshot to CPU.

Remove when the pinned vLLM gathers PP routes natively. Model Runner V1 remains
unsupported with PP routing replay.
"""

import inspect
import textwrap
from functools import wraps

from loguru import logger

_PP_GUARD = """            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with "
                    "pipeline parallelism (PP > 1)."
                )"""


def apply_routed_experts_pp_patch() -> bool:
    """Install the V2 collection and narrow the old config guard once per process."""
    try:
        import vllm.config.vllm as config_module
        from vllm.sequence import IntermediateTensors
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return False

    original = GPUModelRunner.execute_model
    if getattr(original, "_skyrl_relays_pp_routes", False):
        return False

    # The AuxOutput API requires the native vLLM change instead of this runtime patch.
    if not hasattr(GPUModelRunner, "init_routed_experts_capturer"):
        return False

    config_cls = config_module.VllmConfig
    config_source = inspect.getsource(config_cls.__post_init__)
    if _PP_GUARD not in config_source:
        return False
    config_source = textwrap.dedent(
        config_source.replace(
            _PP_GUARD,
            _PP_GUARD.replace(
                "pipeline_parallel_size > 1:",
                "pipeline_parallel_size > 1 and not self.use_v2_model_runner:",
            ),
            1,
        )
    )
    namespace = {}
    exec(compile(config_source, "<skyrl-patch:vllm-pp-routes>", "exec"), config_module.__dict__, namespace)

    @wraps(original)
    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, *args, **kwargs):
        pp_routed_experts = None
        if (
            self.use_pp
            and not dummy_run
            and self.routed_experts_capturer is not None
            and intermediate_tensors is not None
        ):
            pp_routed_experts = intermediate_tensors["routed_experts"]
        output = original(self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)
        state = self.execute_model_state
        if self.use_pp and not dummy_run and state is not None and state.routed_experts is not None:
            routes = state.routed_experts
            # vLLM 0.30's runner already snapshots the local routed expert IDs.
            local = routes.routing_data
            if pp_routed_experts is not None:
                if pp_routed_experts.shape != local.shape:
                    raise ValueError(
                        "PP routed expert shapes must match: "
                        f"previous={tuple(pp_routed_experts.shape)}, local={tuple(local.shape)}"
                    )
                routes = routes._replace(routing_data=local + pp_routed_experts)
                self.execute_model_state = state._replace(routed_experts=routes)
            if isinstance(output, IntermediateTensors):
                # Add expert IDs without mutating the model's output dictionary.
                return IntermediateTensors(output.tensors | {"routed_experts": routes.routing_data})
        return output

    execute_model._skyrl_relays_pp_routes = True
    GPUModelRunner.execute_model = execute_model
    config_cls.__post_init__ = wraps(config_cls.__post_init__)(namespace["__post_init__"])
    # Pydantic stores the post-init callback in its compiled dataclass schema.
    # Replacing the method alone leaves actual VllmConfig construction unchanged.
    from pydantic.dataclasses import rebuild_dataclass

    rebuild_dataclass(config_cls, force=True)
    logger.info("Installed Model Runner V2 pipeline-parallel routed-experts collection")
    return True
