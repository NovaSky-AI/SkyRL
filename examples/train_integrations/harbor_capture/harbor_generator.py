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
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from tqdm.asyncio import tqdm

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
    # One `token_samples` row per root-to-leaf branch, straight from capture.
    rows: List[Dict[str, Any]]
    reward: float = 0.0
    # "complete" | "context_length" | "agent_timeout" | "error" | "length".
    stop_reason: str = "complete"
    generation_time: Optional[float] = None


class HarborCaptureGenerator(GeneratorInterface):
    """Run Harbor trials against a capture token proxy."""

    def __init__(
        self,
        generator_cfg: Any,
        harbor_trial_config: Dict[str, Any],
        inference_engine_client: Any,
        capture_endpoint: str,
        *,
        project: str,
        run_id: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        from skyrl_capture.sdk import CaptureClient

        self.generator_cfg = generator_cfg
        self.inference_engine_client = inference_engine_client
        self.capture_endpoint = capture_endpoint
        # One client for the generator's life. `create_trajectory` would make
        # a throwaway one per trial otherwise, and a connection pool per
        # rollout is how a long run runs out of file descriptors.
        self.capture_client = CaptureClient(capture_endpoint)
        self.project = project
        self.run_id_label = run_id
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

        _require_grouped_output(generator_cfg)

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
            # `step_wise` is deliberately not passed: every trainable path is
            # emitted either way now, grouped by trajectory id with the last
            # marked. What the run still needs is the matching generator
            # configuration -- `step_wise_trajectories=true` so one rollout may
            # emit several rows, and `merge_stepwise_output=false` so complete
            # paths are not re-merged as if they were sequential turns.
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
        from skyrl_capture.sdk import create_trajectory

        session = _session_id(trajectory_id, run_id=self.run_id, step=step, attempt=attempt)
        trajectory = await asyncio.to_thread(
            create_trajectory,
            project=self.project,
            # First-class dimensions rather than annotations, because every
            # question about a run groups by them.
            run_id=self.run_id_label,
            task_id=_task_id(trajectory_id),
            step=step,
            # Naming the trajectory is also how the engine's session key is
            # chosen, so capture's session and SkyRL's are the same one.
            trajectory_id=session,
            client=self.capture_client,
        )

        reward, stop_reason = 0.0, "error"
        envelope: Dict[str, Any] = {}
        failure: Optional[BaseException] = None
        try:
            config = _with_capture_route(
                self._harbor_trial_config_template, trajectory, cache_salt=cache_salt
            )
            results = await self._run_harbor(config, prompt)
        except TimeoutError as error:
            stop_reason = "agent_timeout"
            failure = error
            raise
        except BaseException as error:
            failure = error
            raise
        else:
            reward = float(results.get("reward", 0.0))
            stop_reason = results.get("stop_reason", "complete")
        finally:
            if failure is not None:
                # Say why, on the trajectory, before trying to finish it.
                # `annotate` is a metadata edit and does not wait for turns to
                # settle, so it lands even when `finish` cannot -- and an
                # abandoned agent leaves a generation running, which is exactly
                # when finishing is refused. Without this the record carries no
                # trace of what went wrong and reads as merely unfinished.
                await _explain_failure(trajectory, stop_reason, failure)
            try:
                envelope = await asyncio.to_thread(
                    trajectory.finish,
                    annotations={"reward": reward, "stop_reason": stop_reason},
                    format="token_samples",
                )
            except Exception as error:
                # On the success path a failed finish is a failed attempt:
                # nothing committed, so there is nothing to train from.
                #
                # On the failure path it must not speak. The trial already
                # failed for a reason the caller acts on -- a timeout is masked
                # and not retried -- and letting a finish error replace it
                # turns that into a generic failure and buys a retry the
                # timeout policy exists to avoid. The reason is annotated
                # above, and the record is completed by capture's sweep.
                if failure is None:
                    raise
                logger.warning(
                    "could not finish %s after %s: %s", trajectory.id, stop_reason, error
                )
            finally:
                # The engine's session outlives the trajectory unless someone
                # says otherwise, and a router that keeps one per rollout
                # leaks prefix-cache slots for the length of the run.
                await _release_session(self.inference_engine_client, session)

        rows = _trainable_rows(envelope, trajectory_id=trajectory_id)
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


def _require_grouped_output(generator_cfg: Any) -> None:
    """Refuse to start unless the run is configured for grouped output.

    One capture trajectory can export several complete paths -- a summarizing
    agent produces them by design -- and this generator emits one row per
    path. SkyRL expresses "several rows, one rollout, one advantage" with its
    step-wise machinery, so the run has to be configured for it.

    Checked here rather than at the first batch because the failure is
    otherwise expensive and indirect: `step_wise_trajectories=false` makes the
    trainer assert one response per prompt, and it does that *after* a full
    batch of Harbor trials has been run and thrown away. Raising at
    construction costs nothing and names the override.
    """
    if not getattr(generator_cfg, "step_wise_trajectories", False):
        raise ValueError(
            "this generator emits one row per captured path, so a rollout that branched "
            "produces more rows than prompts. Set generator.step_wise_trajectories=true. "
            "The rows are complete multi-turn samples grouped by trajectory id, not "
            "sequential turns -- step-wise is being reused for its grouping."
        )
    if getattr(generator_cfg, "merge_stepwise_output", False):
        raise ValueError(
            "set generator.merge_stepwise_output=false. Prefix merging recombines "
            "sequential per-turn rows where prompt[i] + response[i] is a prefix of "
            "prompt[i+1]. These rows are already complete multi-turn paths, so merging "
            "would at best be redundant and at worst fuse two distinct paths that "
            "merely look like a prefix of one another."
        )


#: capture authenticates nothing on the way in, but most provider SDKs refuse
#: to construct a client without a key. This is that placeholder.
PLACEHOLDER_API_KEY = "skyrl-capture"


def _task_id(trajectory_id: TrajectoryID) -> Optional[str]:
    """Which task this attempt attempted.

    A GRPO group is the repetitions of one instance, so the instance is the
    task and the repetition is not part of it. `task_id` is an indexed column
    on a listing, which is what makes "show me this group" a query.
    """
    instance = getattr(trajectory_id, "instance_id", None)
    return None if instance is None else str(instance)


def _with_capture_route(
    template: Dict[str, Any], trajectory: Any, *, cache_salt: Optional[str]
) -> Dict[str, Any]:
    """Point one trial at its trajectory. The only change Harbor sees.

    The key goes in ``llm_kwargs``, not beside ``api_base``. Terminus-2 takes
    ``api_base`` as its own parameter but has no ``api_key`` one: it forwards
    ``llm_kwargs`` to the LiteLLM constructor and swallows anything else. An
    ``api_key`` set next to ``api_base`` is therefore accepted, ignored, and
    the trajectory route answers 401 -- with nothing in the trial config to
    suggest why.

    ``cache_salt`` rides in ``extra_body``, which LiteLLM merges into the
    request body. capture carries fields it does not recognise through to the
    upstream protocol, and `SkyRLTitoProtocol` is what knows this one means a
    prefix-cache key. It is per trajectory because the weights move every
    training step while the engine stays put.
    """
    import copy

    config = copy.deepcopy(template)
    kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
    kwargs["api_base"] = trajectory.base_url
    llm_kwargs = kwargs.setdefault("llm_kwargs", {})
    if not isinstance(llm_kwargs, dict):
        raise TypeError("harbor agent kwargs.llm_kwargs must be a mapping")
    llm_kwargs["api_key"] = PLACEHOLDER_API_KEY
    if cache_salt:
        extra_body = llm_kwargs.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            raise TypeError("harbor agent kwargs.llm_kwargs.extra_body must be a mapping")
        extra_body["cache_salt"] = cache_salt
    return config


async def _explain_failure(trajectory: Any, stop_reason: str, error: BaseException) -> None:
    """Record why a trial failed, on the trajectory itself.

    Best effort, and deliberately separate from `finish`: a metadata edit
    applies to a live trajectory without waiting for its turns to settle, so it
    succeeds in the one case that matters -- an agent that gave up mid-call,
    leaving a generation in flight that refuses the finish behind it.

    The label is what a listing filters on; the annotation is what a reader
    needs. Both travel with the record when it is eventually compiled, because
    they are in the journal before the finish is attempted.
    """
    try:
        await asyncio.to_thread(
            trajectory.annotate,
            failed=True,
            failure_reason=stop_reason,
            failure_detail=f"{type(error).__name__}: {error}"[:500],
        )
        await asyncio.to_thread(trajectory.tag, "failed", stop_reason)
    except Exception as annotate_error:
        logger.warning("could not annotate the failure of %s: %s", trajectory.id, annotate_error)


async def _release_session(engine_client: Any, session_id: str) -> None:
    """Let the router drop this rollout's session.

    Best effort on purpose: the trajectory is already committed by the time
    this runs, so a router that has gone away must not turn a finished rollout
    into a failed one. What it costs when it fails is a prefix-cache slot.
    """
    finish = getattr(engine_client, "finish_session", None)
    if finish is None:
        return
    try:
        result = finish(session_id)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("could not release engine session %s", session_id, exc_info=True)


def _trainable_rows(envelope: Dict[str, Any], *, trajectory_id: TrajectoryID) -> List[Dict[str, Any]]:
    """The rows this attempt may train on, or none at all.

    Integrity is a property of the whole capture, not of a row: a trajectory
    with a gap, an unconfirmed delivery or an unfinished commit has rows that
    look perfectly well-formed and are missing a turn that happened. Training
    on the ones that survived would be training on a conversation that never
    took place, so the whole attempt is refused instead.
    """
    status = envelope.get("status")
    if status != "finished":
        logger.warning("refusing rows from %s: status is %r", trajectory_id, status)
        return []
    records = envelope.get("records") or []
    for record in records:
        capture = record.get("capture") or {}
        if capture and not _capture_is_sound(capture):
            logger.warning("refusing rows from %s: %s", trajectory_id, capture)
            return []
    return list(records)


def _capture_is_sound(capture: Dict[str, Any]) -> bool:
    return (
        bool(capture.get("complete", True))
        and not capture.get("calls_missing")
        and not capture.get("delivery_uncertain")
        and not capture.get("recovery_uncertain")
        and not capture.get("errors")
    )
