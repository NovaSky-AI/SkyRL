"""Compose the run-scoped sampling service used by generators."""

from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.train.config import GeneratorConfig
from skyrl.train.sampling_service import SamplingClient, SamplingService
from skyrl.utils.adaptive_concurrency import (
    EngineLoadConcurrencyPolicy,
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


def build_sampling_concurrency_controller(
    generator_cfg: GeneratorConfig,
) -> SamplingConcurrencyController | None:
    """Build the controller owned by the run-scoped sampling service."""

    concurrency_cfg = generator_cfg.sampling_concurrency
    if not concurrency_cfg.enabled:
        return None
    if concurrency_cfg.policy == "fixed":
        policy = FixedConcurrencyPolicy()
    elif concurrency_cfg.policy == "engine_load":
        if not generator_cfg.inference_engine.run_engines_locally:
            raise ValueError("sampling concurrency policy 'engine_load' requires SkyRL-managed local vLLM engines")
        if generator_cfg.inference_engine.enable_pd:
            raise ValueError(
                "sampling concurrency policy 'engine_load' does not yet support prefill/decode role attribution"
            )
        policy = EngineLoadConcurrencyPolicy(
            min_limit=concurrency_cfg.min_limit,
            max_limit=concurrency_cfg.max_limit,
        )
    else:
        raise ValueError(f"Unknown sampling concurrency policy: {concurrency_cfg.policy}")

    return SamplingConcurrencyController(
        policy=policy,
        initial_limit=concurrency_cfg.initial_limit,
    )


def build_sampling_client(
    inference_engine_client: InferenceEngineInterface,
    generator_cfg: GeneratorConfig,
) -> SamplingClient:
    """Wrap one backend in the long-lived service and its lightweight client."""

    controller = build_sampling_concurrency_controller(generator_cfg)
    feedback_producer = None
    if controller is not None and isinstance(controller.policy, EngineLoadConcurrencyPolicy):
        from skyrl.train.utils.vllm_metrics_scraper import VLLMEngineFeedbackProducer

        server_urls = getattr(inference_engine_client, "server_urls", None)
        if not server_urls:
            raise ValueError(f"{type(inference_engine_client).__name__} does not expose managed-vLLM metrics endpoints")
        feedback_producer = VLLMEngineFeedbackProducer(
            controller,
            model_server_urls=list(server_urls),
        )

    service = SamplingService(
        inference_engine_client,
        controller=controller,
        feedback_producer=feedback_producer,
    )
    if controller is not None:
        logger.info(
            "Server-side sampling concurrency enabled with policy={} and initial_limit={}",
            type(controller.policy).__name__,
            controller.current_limit,
        )
    return SamplingClient(service, inference_engine_client)
