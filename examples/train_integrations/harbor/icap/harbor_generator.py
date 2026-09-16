"""Harbor trials through inference-capture's token proxy.

The difference from the sibling ``harbor`` integration is what Harbor is asked
to do. There it runs with ``collect_rollout_details=True`` and emits per-turn
token IDs itself, which is why that integration has to ban summarization:
compaction breaks the harness's own token accounting.

Here Harbor runs in text space, unmodified. It is handed a trajectory URL and a
key and nothing else changes. The proxy renders the prompt, calls the engine
with token IDs, and keeps a message graph -- so a rewritten history is a branch
rather than a hole, and summarization is allowed.

Two hooks:

* once per run, a ``CaptureService`` beside the inference engine, with the
  engine registered as a ``tokens`` target;
* once per trial, create a trajectory, point Harbor at it, finish it, export.

The exported branches become a ``GeneratorOutput`` in ``compose``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from tqdm.asyncio import tqdm

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from skyrl.train.generators.base import (
    ConversationType,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)

from .compose import compose

logger = logging.getLogger(__name__)


@dataclass
class TrialOutcome:
    """One completed trial: what capture recorded, plus what only Harbor knows."""

    trajectory_id: TrajectoryID
    # One `token-samples` row per root-to-leaf branch, straight from capture.
    rows: List[Dict[str, Any]]
    reward: float = 0.0
    # "complete" | "context_length" | "agent_timeout" | "error" | "length".
    stop_reason: str = "complete"
    generation_time: Optional[float] = None


class ICapHarborGenerator(GeneratorInterface):
    """Run Harbor trials against a capture token proxy."""

    def __init__(
        self,
        generator_cfg: Any,
        harbor_trial_config: Dict[str, Any],
        inference_engine_client: Any,
        capture_service: Any,
        *,
        project: str,
        target_name: str = "policy",
        max_retries: int = 2,
    ) -> None:
        self.generator_cfg = generator_cfg
        self.inference_engine_client = inference_engine_client
        self.capture = capture_service
        self.project = project
        self.target_name = target_name
        # Total attempts per trial, not extra ones. A Harbor failure is often
        # environmental -- a sandbox that did not come up -- which is why the
        # sibling integration retries too.
        self.max_retries = max_retries
        # A caller-supplied trajectory id names one trial for the life of the
        # capture database, but SkyRL's TrajectoryID is only unique within a
        # step: instance 0, repetition 0 comes round again every step and on
        # every re-run. Without something per-run in the name, step 2 collides
        # with step 1 and the batch dies on a 409. See `_session_id`.
        self.run_id = uuid4().hex[:8]
        self._harbor_trial_config_template = harbor_trial_config

        # Harbor is not asked to collect token ids here: the proxy has them
        # exactly, and asking twice is how the other integration ends up
        # banning summarization.
        agent_kwargs = self._harbor_trial_config_template.setdefault("agent", {}).setdefault("kwargs", {})
        agent_kwargs.pop("collect_rollout_details", None)

    # -- policy version ----------------------------------------------------
    def _cache_salt(self) -> Optional[str]:
        """Prefix-cache salt keyed on the current weights.

        Rollouts from different weights must not share the engine's prefix
        cache. The salt is per trajectory rather than per target because the
        weights move every training step while the target stays put.
        """
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        version = getattr(self.inference_engine_client, "weight_version", None)
        return None if version is None else str(version)

    # -- the interface -----------------------------------------------------
    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(f"prompt count ({len(prompts)}) does not match trajectory_ids " f"({len(trajectory_ids)})")

        cache_salt = self._cache_salt()
        step = getattr(input_batch.get("batch_metadata"), "global_step", None)
        outcomes: List[Optional[TrialOutcome]] = [None] * len(prompts)

        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def worker(index: int, prompt: ConversationType, trajectory_id: TrajectoryID) -> None:
            outcomes[index] = await self._trial(prompt, trajectory_id, cache_salt, step)
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as group:
                for index, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    group.create_task(worker(index, prompt, trajectory_id))
        finally:
            progress.close()

        # `_trial` does not raise, so every slot is filled -- but a masked
        # placeholder beats an `AttributeError` inside `compose` naming the
        # wrong culprit if that ever stops being true.
        settled = [
            outcome or self._masked(trajectory_ids[index], "error", time.monotonic())
            for index, outcome in enumerate(outcomes)
        ]
        return compose(
            [outcome.rows for outcome in settled],
            trajectory_ids=[outcome.trajectory_id for outcome in settled],
            rewards=[outcome.reward for outcome in settled],
            stop_reasons=[outcome.stop_reason for outcome in settled],
            step_wise=getattr(self.generator_cfg, "step_wise_trajectories", True),
            generation_times=[outcome.generation_time or 0.0 for outcome in settled],
        )

    # -- one trial ---------------------------------------------------------
    async def _trial(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
    ) -> TrialOutcome:
        """One rollout, with retries. **Never raises.**

        Trials run in a ``TaskGroup``, which cancels every sibling when any task
        raises -- so an exception escaping here would lose the whole step, not
        one rollout. A trial that cannot be completed is masked instead, exactly
        as the sibling integration masks one, and ``compose`` drops it.
        """
        started = time.monotonic()
        attempts = max(1, self.max_retries)
        last_error: Optional[BaseException] = None

        for attempt in range(attempts):
            try:
                return await self._attempt(prompt, trajectory_id, cache_salt, step, attempt, started)
            except TimeoutError:
                # The agent ran out of time. Retrying buys nothing and costs a
                # sandbox, so mask it, as the sibling integration does.
                return self._masked(trajectory_id, "agent_timeout", started)
            except Exception as error:
                last_error = error
                logger.warning(
                    "trial %s attempt %d/%d failed: %s", trajectory_id, attempt + 1, attempts, error
                )

        logger.error("trial %s failed %d times, masking: %s", trajectory_id, attempts, last_error)
        return self._masked(trajectory_id, "error", started)

    def _masked(self, trajectory_id: TrajectoryID, stop_reason: str, started: float) -> TrialOutcome:
        """No rows, so `compose` masks it -- rather than training on a guess."""
        return TrialOutcome(
            trajectory_id=trajectory_id,
            rows=[],
            reward=0.0,
            stop_reason=stop_reason,
            generation_time=time.monotonic() - started,
        )

    async def _attempt(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        attempt: int,
        started: float,
    ) -> TrialOutcome:
        """One attempt, on its own trajectory. Raises if it did not complete.

        A retry gets a **fresh** trajectory rather than reusing the failed one:
        capture's graph is append-only, so a second attempt against the same
        name would interleave two rollouts into one record.
        """
        from inference_capture.sdk import create_trajectory

        labels = [f"step-{step}"] if step is not None else []
        upstream: Dict[str, Any] = {"body": {"cache_salt": cache_salt}} if cache_salt else {}

        trajectory = await asyncio.to_thread(
            create_trajectory,
            project=self.project,
            target=self.target_name,
            # Naming the trajectory is also how the engine's session key is
            # chosen, so capture's session and SkyRL's are the same one.
            trajectory_id=_session_id(trajectory_id, run_id=self.run_id, step=step, attempt=attempt),
            upstream=upstream,
            labels=labels,
            endpoint=self.capture.base_url,
        )

        reward, stop_reason = 0.0, "error"
        try:
            config = _with_api_base(self._harbor_trial_config_template, trajectory)
            results = await self._run_harbor(config, prompt)
        except TimeoutError:
            stop_reason = "agent_timeout"
            raise
        else:
            reward = float(results.get("reward", 0.0))
            stop_reason = results.get("stop_reason", "complete")
        finally:
            # Record the outcome even on the way out. The graph of a failed
            # attempt is still worth having, and finishing releases the lease.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    trajectory.finish, annotations={"reward": reward, "stop_reason": stop_reason}
                )

        rows = await asyncio.to_thread(trajectory.export, "token-samples")
        return TrialOutcome(
            trajectory_id=trajectory_id,
            rows=rows,
            reward=reward,
            stop_reason=stop_reason,
            generation_time=time.monotonic() - started,
        )

    async def _run_harbor(self, config: Dict[str, Any], prompt: ConversationType) -> Dict[str, Any]:
        """Run one Harbor trial. The config already points at the capture route.

        Only two things come back that capture cannot know: the reward, and why
        the agent stopped. Everything else about the rollout is in the graph.
        """
        from copy import deepcopy

        config = deepcopy(config)
        config["task"] = {"path": prompt}
        trial = await Trial.create(TrialConfig.model_validate(config))
        results = await trial.run()

        exception = results.exception_info.exception_type if results.exception_info else None
        if exception == "AgentTimeoutError":
            # Masked, not retried, and not trained on -- the same as the
            # sibling integration treats it.
            raise TimeoutError("harbor reported AgentTimeoutError")
        if exception == "ContextLengthExceededError":
            # Trainable with reward 0, again matching the sibling.
            return {"reward": 0.0, "stop_reason": "context_length"}
        if not results.verifier_result:
            raise RuntimeError(f"trial produced no verifier result: {results.exception_info}")

        return {
            "reward": float(results.verifier_result.rewards["reward"]),
            "stop_reason": "complete",
        }


def _session_id(
    trajectory_id: TrajectoryID, *, run_id: str, step: Optional[int], attempt: int = 0
) -> str:
    """A trajectory name that is unique, and a URL path segment.

    Four parts, each earning its place:

    * ``run_id`` -- capture keeps a named trajectory forever, so a second run
      of the same batch would collide with the first;
    * ``step`` -- the same instance and repetition come round every training
      step, against different weights;
    * the SkyRL trajectory id -- so capture's session and SkyRL's are legibly
      the same one, which is the point of naming it at all;
    * ``attempt``, only when there has been one. A retry needs its own
      trajectory, because capture's graph is append-only and reusing the name
      would interleave two rollouts into one record.

    Non-alphanumerics are folded to ``-`` because the name becomes a path
    segment on the trajectory's own route.
    """
    raw = trajectory_id.to_string() if hasattr(trajectory_id, "to_string") else str(trajectory_id)
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in raw)
    name = f"{run_id}-s{'x' if step is None else step}-{safe}"
    return name if attempt == 0 else f"{name}-a{attempt}"


def _with_api_base(template: Dict[str, Any], trajectory: Any) -> Dict[str, Any]:
    """Point one trial at its trajectory. The only change Harbor sees.

    The key goes in ``llm_kwargs``, not beside ``api_base``. Terminus-2 takes
    ``api_base`` as its own parameter but has no ``api_key`` one: it forwards
    ``llm_kwargs`` to the LiteLLM constructor and swallows anything else. An
    ``api_key`` set next to ``api_base`` is therefore accepted, ignored, and
    the trajectory route answers 401 -- with nothing in the trial config to
    suggest why.
    """
    import copy

    config = copy.deepcopy(template)
    kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
    kwargs["api_base"] = trajectory.base_url
    llm_kwargs = kwargs.setdefault("llm_kwargs", {})
    if not isinstance(llm_kwargs, dict):
        raise TypeError("harbor agent kwargs.llm_kwargs must be a mapping")
    llm_kwargs["api_key"] = trajectory.api_key
    return config
