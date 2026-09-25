"""Harbor trials, captured by skycap.

The sibling ``harbor`` integration asks Harbor for per-turn token ids
(``collect_rollout_details``), which is why it has to ban summarization: a
rewritten history breaks the harness's token accounting. Here Harbor runs
unmodified, in text space. Each trial is pointed at its own skycap trajectory
URL; skycap renders every prompt, calls the engine with token ids, and keeps a
context graph, so a rewritten history is a branch rather than a hole and
summarization is allowed.

Per trial: create a trajectory, point the agent at ``trajectory.base_url``, run
it, and ``finish`` with the reward to get one sample per path. ``compose``
turns those into the step-wise ``GeneratorOutput``.
"""

import asyncio
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import litellm
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from skycap import CapturePool
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import build_vllm_cache_salt
from skyrl.train.utils.rate_limiter import create_rate_limiter

from .compose import TrialOutcome, compose

litellm.suppress_debug_info = True

# Attempts per trial. A Harbor failure is often environmental (a sandbox that didn't come up).
MAX_NUM_RETRIES_PER_TRIAL = 2

#: skycap authenticates nothing, but LiteLLM won't build a client without a key.
PLACEHOLDER_API_KEY = "skycap"


class HarborSkycapGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: Any,
        harbor_cfg: Dict[str, Any],
        capture_urls: List[str],
        inference_engine_client: Any = None,
    ) -> None:
        """
        Args:
            generator_cfg: the run's generator config.
            harbor_cfg: Harbor's ``TrialConfig`` template.
            capture_urls: the skycap servers to spread trajectories over.
            inference_engine_client: read for its ``weight_version``, which keys the prefix-cache salt.
        """
        if not getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "HarborSkycapGenerator emits one row per captured path, grouped per rollout the step-wise way. "
                "Set generator.step_wise_trajectories=true."
            )
        if getattr(generator_cfg, "merge_stepwise_output", False):
            raise ValueError(
                "Set generator.merge_stepwise_output=false: each row is already a complete multi-turn path, and "
                "prefix merging could fuse two paths that merely share a prefix."
            )
        if getattr(generator_cfg.inference_engine, "enable_return_routed_experts", False):
            raise ValueError(
                "HarborSkycapGenerator doesn't support R3 yet: SkyRL's trainer refuses routed experts with "
                "step-wise output. skycap still records them. Set "
                "generator.inference_engine.enable_return_routed_experts=false."
            )
        self.generator_cfg = generator_cfg
        self.capture_urls = list(capture_urls)
        self.inference_engine_client = inference_engine_client
        served = generator_cfg.inference_engine.served_model_name
        assert served is not None and "/" not in served, "served_model_name must be set, without '/'"
        self._served_model_name = served

        self._template = deepcopy(harbor_cfg)
        agent = self._template.setdefault("agent", {})
        agent["model_name"] = f"hosted_vllm/{served}"
        kwargs = agent.setdefault("kwargs", {})
        # skycap has the tokens exactly; asking Harbor for them too is what forces the sibling to ban summarization.
        kwargs.pop("collect_rollout_details", None)
        self._rate_limiter = create_rate_limiter(getattr(generator_cfg, "rate_limit", None))

    def _cache_salt(self) -> Optional[str]:
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        return build_vllm_cache_salt(weight_version, self._served_model_name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(f"Prompt count ({len(prompts)}) doesn't match trajectory_ids ({len(trajectory_ids)})")
        metadata = input_batch.get("batch_metadata")
        step = getattr(metadata, "global_step", None)
        cache_salt = self._cache_salt()

        outcomes: List[Optional[TrialOutcome]] = [None] * len(prompts)
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        # A pool per batch: its HTTP session belongs to this event loop.
        async with CapturePool(self.capture_urls) as pool:

            async def worker(index: int, prompt: ConversationType, trajectory_id: TrajectoryID) -> None:
                outcomes[index] = await self._trial(pool, prompt, trajectory_id, cache_salt, step)
                progress.update(1)

            try:
                async with asyncio.TaskGroup() as group:
                    for index, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                        group.create_task(worker(index, prompt, trajectory_id))
            finally:
                progress.close()

        return compose(
            outcomes,
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
            top_k=self.generator_cfg.sampling_params.top_k,
        )

    async def _trial(
        self,
        pool: CapturePool,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
    ) -> TrialOutcome:
        """One rollout, retried on unknown errors. Never raises: one failure must not cancel the batch."""
        started = time.monotonic()
        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            try:
                outcome = await self._attempt(pool, prompt, trajectory_id, cache_salt, step, attempt)
            except Exception as error:  # noqa: BLE001 - retried, then masked
                logger.warning(f"{prefix} failed: {type(error).__name__}: {error}")
                continue
            outcome.e2e_time = time.monotonic() - started
            if outcome.stop_reason != "error":
                return outcome
            logger.warning(f"{prefix} produced nothing to train on")
        return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error", e2e_time=time.monotonic() - started)

    async def _attempt(
        self,
        pool: CapturePool,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        attempt: int,
    ) -> TrialOutcome:
        """One attempt on its own trajectory, so a retry never shares a graph with the attempt it replaces."""
        meta = {
            "instance_id": str(trajectory_id.instance_id),
            "repetition_id": trajectory_id.repetition_id,
            "step": step,
            "attempt": attempt,
        }
        async with pool.trajectory(meta) as trajectory:
            config = self._trial_config(prompt, trajectory.base_url, cache_salt)
            async with self._rate_limiter:
                results = await (await Trial.create(TrialConfig.model_validate(config))).run()

            exception = results.exception_info.exception_type if results.exception_info else None
            if exception == "AgentTimeoutError":
                # Masked, not retried, as the sibling does.
                reward, stop_reason = 0.0, "agent_timeout"
            elif exception == "ContextLengthExceededError":
                # Trains with reward 0, as the sibling does.
                reward, stop_reason = 0.0, "context_length"
            elif not results.verifier_result:
                reward, stop_reason = 0.0, "error"
                logger.warning(f"Trajectory {trajectory_id} has no verifier result: {results.exception_info}")
            else:
                reward, stop_reason = float(results.verifier_result.rewards["reward"]), "complete"
            finished = await trajectory.finish({"reward": reward, "stop_reason": stop_reason})

        if finished.status != "finished":
            # The trajectory failed inside skycap (e.g. an unattributable prompt): its samples may miss a turn.
            logger.warning(f"Trajectory {trajectory_id}: skycap status {finished.status!r}, not training on it")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error")
        return TrialOutcome(
            trajectory_id=trajectory_id, samples=finished.samples, reward=reward, stop_reason=stop_reason
        )

    def _trial_config(self, prompt: ConversationType, base_url: str, cache_salt: Optional[str]) -> Dict[str, Any]:
        config = deepcopy(self._template)
        config["task"] = {"path": prompt}
        kwargs = config["agent"]["kwargs"]
        kwargs["api_base"] = base_url
        llm_kwargs = kwargs.setdefault("llm_kwargs", {})
        # Terminus-2 takes `api_base` itself but passes a key only through `llm_kwargs`.
        llm_kwargs["api_key"] = PLACEHOLDER_API_KEY
        if cache_salt is not None:
            # LiteLLM merges `extra_body` into the request body, where skycap reads `cache_salt` and forwards it.
            extra_body = llm_kwargs.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
            extra_body["cache_salt"] = cache_salt
        return config
