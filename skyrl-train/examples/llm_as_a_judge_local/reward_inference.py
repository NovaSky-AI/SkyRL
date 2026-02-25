"""
Frozen reward model inference using SkyRL's InferenceEngineClient.

This module provides a **self-contained** frozen-model inference layer
for the ``llm_as_a_judge_local`` example.  No modifications to SkyRL
core are required.

Classes
-------
1. **FrozenRewardInferenceClient** — a subclass of ``InferenceEngineClient``
   that creates vLLM engines *without* weight-sync capabilities.  It adds
   ``score()`` and ``score_batch()`` methods for text-level reward scoring
   while inheriting load balancing, prompt routing, and multi-engine fan-out
   from the base class.

2. **RewardInferenceService** — a Ray actor that wraps the client, making
   it accessible from any Ray worker (e.g., environments running on other
   nodes).

Helper
------
``create_frozen_vllm_engines()`` — a standalone function that creates vLLM
inference engines without the ``worker_extension_cls`` used for weight sync.
This avoids any dependency on a ``frozen_model`` flag in core SkyRL while
still leveraging placement-group GPU scheduling and the
``RayWrappedInferenceEngine`` wrapper.

Benefits over a plain vLLM subprocess approach:

- Automatic load balancing across multiple engines
- Placement-group GPU scheduling (managed by Ray)
- Proper Ray lifecycle management (no stale subprocesses or port conflicts)
- Easy scale-out: just increase ``num_engines``
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import ray
from omegaconf import OmegaConf

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInterface

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Local engine creation for frozen models (no core changes needed)
# ═══════════════════════════════════════════════════════════════════


def create_frozen_vllm_engines(
    num_engines: int,
    pretrain: str,
    tensor_parallel_size: int = 1,
    model_dtype: str = "auto",
    seed: int = 42,
    enable_prefix_caching: bool = True,
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 4096,
    max_num_seqs: int = 256,
) -> List[InferenceEngineInterface]:
    """Create frozen vLLM inference engines (no weight sync, no sleep).

    This is a **self-contained** engine creation function designed for
    frozen models (reward models, LLM judges, verifiers) that never need
    weight synchronisation.  It mirrors the relevant parts of the core
    ``create_ray_wrapped_inference_engines()`` but intentionally omits:

    - ``worker_extension_cls`` (weight sync)
    - ``enable_sleep_mode`` (always ``False``)
    - LoRA / data-parallel / pipeline-parallel support

    The engines are still full ``InferenceEngineInterface`` instances
    wrapped in ``RayWrappedInferenceEngine``, so they inherit load
    balancing, multi-node placement-group scheduling, and all other
    ``InferenceEngineClient`` features.

    Args:
        num_engines: Number of independent vLLM engines to create.
        pretrain: HuggingFace model name or local path.
        tensor_parallel_size: Tensor-parallel GPUs per engine.
        model_dtype: Model dtype (``"auto"``, ``"bfloat16"``, …).
        seed: Random seed for reproducibility.
        enable_prefix_caching: Enable vLLM prefix caching.
        enforce_eager: Disable CUDA graphs for reliability.
        gpu_memory_utilization: GPU memory fraction.
        max_model_len: Maximum sequence length for KV cache allocation.
            If ``None``, vLLM infers from the model config (can be very
            large and waste GPU memory).  Set explicitly to limit VRAM.
        max_num_batched_tokens: Maximum tokens per batch.
        max_num_seqs: Maximum concurrent sequences.

    Returns:
        List of ``InferenceEngineInterface`` (``RayWrappedInferenceEngine``).
    """
    from ray.util.placement_group import (
        PlacementGroupSchedulingStrategy,
        placement_group,
    )

    from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
        RayWrappedInferenceEngine,
    )
    from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMRayActor
    from skyrl_train.utils import (
        get_all_env_variables,
        get_ray_pg_ready_with_timeout,
        ray_noset_visible_devices,
    )

    noset_visible_devices = ray_noset_visible_devices(
        ray.get(get_all_env_variables.remote())
    )

    # For TP=1 use the lightweight "uni" executor; TP>1 needs "ray"
    distributed_executor_backend = (
        "uni" if tensor_parallel_size == 1 else "ray"
    )
    num_gpus_per_actor = 1 if tensor_parallel_size == 1 else 0
    per_engine_gpu_count = tensor_parallel_size

    # Create a placement group to pack all engine GPUs together
    bundles = [
        {"GPU": 1, "CPU": 1}
        for _ in range(num_engines * per_engine_gpu_count)
    ]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    logger.info(
        f"Creating {num_engines} frozen vLLM engine(s) "
        f"(TP={tensor_parallel_size}, no weight sync) for: {pretrain}"
    )

    actors = []
    for i in range(num_engines):
        base_pg_index = i * per_engine_gpu_count
        bundle_indices = (
            list(range(base_pg_index, base_pg_index + per_engine_gpu_count))
            if per_engine_gpu_count > 1
            else None
        )
        sched = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=base_pg_index,
        )

        # Build optional kwargs (only pass max_model_len if explicitly set,
        # otherwise let vLLM infer from the model config)
        extra_kwargs: Dict[str, Any] = {}
        if max_model_len is not None:
            extra_kwargs["max_model_len"] = max_model_len

        engine = AsyncVLLMRayActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=sched,
        ).remote(
            model=pretrain,
            enforce_eager=enforce_eager,
            # NOTE: No worker_extension_cls — frozen models don't sync weights
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            distributed_executor_backend=distributed_executor_backend,
            seed=seed + i,
            enable_prefix_caching=enable_prefix_caching,
            dtype=model_dtype,
            trust_remote_code=True,
            vllm_v1_disable_multiproc=True,
            gpu_memory_utilization=gpu_memory_utilization,
            bundle_indices=bundle_indices,
            num_gpus=per_engine_gpu_count,
            enable_sleep_mode=False,
            noset_visible_devices=noset_visible_devices,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_logprobs=1,
            **extra_kwargs,
        )
        actors.append(engine)

    return [RayWrappedInferenceEngine(a) for a in actors]


# ═══════════════════════════════════════════════════════════════════
# Subclass of InferenceEngineClient for frozen reward models
# ═══════════════════════════════════════════════════════════════════


class FrozenRewardInferenceClient(InferenceEngineClient):
    """InferenceEngineClient for frozen reward models.

    Creates vLLM inference engines **without** weight-sync capabilities
    (via ``create_frozen_vllm_engines``) and adds a text-level scoring API
    on top of the standard ``generate()`` method.

    Key features inherited from ``InferenceEngineClient``:

    - ``generate()`` — batch generation with load-balanced prompt routing
    - ``_select_engine_idx()`` — random / consistent-hash engine selection
    - ``_run_on_all_engines()`` — fan-out utility for all engines
    - ``reset_prefix_cache()`` — cache management

    Added by this subclass:

    - ``score()`` — single chat prompt → text response
    - ``score_batch()`` — batch scoring with automatic engine fan-out

    Usage::

        client = FrozenRewardInferenceClient(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            num_engines=2,
        )
        response = await client.score([{"role": "user", "content": "..."}])
    """

    def __init__(
        self,
        model: str,
        num_engines: int = 1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = True,
        seed: int = 42,
    ):
        from transformers import AutoTokenizer

        # 1. Load reward model tokenizer (independent from policy tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # 2. Create frozen vLLM engines via our local helper
        #    (no dependency on core frozen_model flag)
        engines = create_frozen_vllm_engines(
            num_engines=num_engines,
            pretrain=model,
            tensor_parallel_size=tensor_parallel_size,
            model_dtype="auto",
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_model_len,
            max_num_seqs=max_num_seqs,
        )

        # 3. Build a minimal DictConfig to satisfy InferenceEngineClient.__init__
        #    We disable the HTTP endpoint since scoring goes through Ray, not HTTP.
        minimal_cfg = OmegaConf.create(
            {
                "generator": {
                    "served_model_name": model,
                    "backend": "vllm",
                    "enable_http_endpoint": False,
                    "http_endpoint_host": "0.0.0.0",
                    "http_endpoint_port": 0,
                },
                "trainer": {"policy": {"model": {"path": model}}},
            }
        )

        # 4. Initialize the base class (sets self.engines, self.tokenizer, etc.)
        super().__init__(engines, tokenizer, minimal_cfg)

        # Store a dedicated reference for the reward model tokenizer
        self.reward_tokenizer = tokenizer
        logger.info(
            f"FrozenRewardInferenceClient ready: "
            f"{len(engines)} engine(s) for {model}"
        )

    # ── Scoring API (text-in / text-out) ──────────────────────────

    async def score(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Score a single chat prompt and return the raw text response.

        Args:
            messages: Chat messages in OpenAI format, e.g.
                      ``[{"role": "user", "content": "..."}]``
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response (caller parses it for the score).
        """
        prompt_ids = self.reward_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        result = await self.generate(
            {
                "prompts": None,
                "prompt_token_ids": [prompt_ids],
                "sampling_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "session_ids": None,
            }
        )
        return result["responses"][0]

    async def score_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """Score a batch of chat prompts with automatic load balancing.

        Each prompt is independently routed to an engine via
        ``route_prompts_to_engines``.  With *N* engines, up to *N*
        requests are processed in parallel.
        """
        all_prompt_ids = [
            self.reward_tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
            )
            for msgs in messages_batch
        ]

        result = await self.generate(
            {
                "prompts": None,
                "prompt_token_ids": all_prompt_ids,
                "sampling_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "session_ids": None,
            }
        )
        return result["responses"]

    # ── Lifecycle overrides (frozen models don't sleep) ───────────

    async def sleep(self, *args: Any, **kwargs: Any):
        """No-op: frozen reward models are always active."""
        pass

    async def wake_up(self, *args: Any, **kwargs: Any):
        """No-op: frozen reward models are always active."""
        pass

    def health_check(self) -> bool:
        """Simple check that engines exist."""
        return len(self.engines) > 0


# ═══════════════════════════════════════════════════════════════════
# Ray actor wrapper (example-specific glue for cross-process access)
# ═══════════════════════════════════════════════════════════════════

# NOTE: Do NOT import loguru at module level.  Ray's cloudpickle
# serialises the module scope when exporting @ray.remote classes, and
# loguru's logger holds open file handles that cannot be pickled.


@ray.remote(num_gpus=0, max_concurrency=64)
class RewardInferenceService:
    """Named Ray actor exposing ``FrozenRewardInferenceClient``.

    Environments look up this actor by name and call ``score()``::

        service = ray.get_actor("reward_inference_service")
        reply = ray.get(service.score.remote(messages))

    The actor itself requests **0 GPUs** — the frozen vLLM engines it
    creates internally are separate Ray actors that each claim their own
    GPU via placement groups.
    """

    def __init__(
        self,
        model: str,
        num_engines: int = 1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        enable_prefix_caching: bool = True,
    ):
        self.client = FrozenRewardInferenceClient(
            model=model,
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
        )
        self._model_name = model
        print(
            f"[RewardInferenceService] Ready: "
            f"{model} × {num_engines} engine(s)"
        )

    async def score(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Score a single chat prompt via the frozen inference client."""
        return await self.client.score(messages, temperature, max_tokens)

    async def score_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """Score a batch of chat prompts with automatic load balancing."""
        return await self.client.score_batch(
            messages_batch, temperature, max_tokens
        )

    def get_model_name(self) -> str:
        """Return the reward model name."""
        return self._model_name

    def health_check(self) -> bool:
        """Check that engines are alive."""
        return self.client.health_check()
