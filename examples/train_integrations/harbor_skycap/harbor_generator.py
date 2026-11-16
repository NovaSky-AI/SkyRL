"""Harbor trials, captured by skycap.

The sibling ``harbor`` integration asks Harbor for per-turn token ids
(``collect_rollout_details``), which is why it has to ban summarization: a
rewritten history breaks the harness's token accounting. Here Harbor runs
unmodified, in text space. Each trial is pointed at its own skycap trajectory
URL; skycap renders every prompt, calls the engine with token ids, and keeps a
context graph, so a rewritten history is a branch rather than a hole and
summarization is allowed.

Per trial: create a trajectory, point the agent at it, run it, and ``finish``
with the reward to get a sample per path the ``train_paths`` rule picks.
``compose`` turns those into the step-wise ``GeneratorOutput``.

Terminus-2 calls the model from this process and gets ``trajectory.base_url``.
An installed agent (mini-swe-agent, Claude Code, ...) calls it from inside its
sandbox, so it gets the trajectory's route on the server's exposed URL
(``trajectory.exposed_base_url``, with ``skycap.exposure``), through the
sandbox environment that LiteLLM- and OpenAI-based agents read.
"""

import asyncio
import importlib
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

import litellm
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from skycap import CapturePool, PathRuleError
from skycap.paths import load_rule
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    StoredRolloutGroup,
    TrajectoryID,
)
from skyrl.train.generators.skycap_store import (
    CapturedFinish,
    StoredCapturePool,
    StoredPools,
)
from skyrl.train.generators.utils import build_vllm_cache_salt
from skyrl.train.utils.rate_limiter import create_rate_limiter

from ..harbor.trial_metrics import TrialAttempts, trial_metrics
from .compose import TrialChecks, TrialOutcome, compose
from .record_index import RecordLog

litellm.suppress_debug_info = True

# Attempts per trial. A Harbor failure is often environmental (a sandbox that didn't come up).
MAX_NUM_RETRIES_PER_TRIAL = 2

#: How a failed trial's exception shows it ran out of context. Harbor raises its own two types for agents it
#: classifies; an installed agent that exits on skycap's 400 (code ``context_length_exceeded``, which LiteLLM
#: may raise as a plain ``BadRequestError``) is only a ``NonZeroAgentExitCodeError``. Its message keeps just the
#: first 1000 characters of the agent's output, which end before the error, so the end of the agent's own log in
#: the trial directory is searched too. Rich wraps that log, possibly inside the code.
CONTEXT_LENGTH_EXCEPTIONS = {"ContextLengthExceededError", "ContextWindowExceededError"}
CONTEXT_LENGTH_OUTPUT = re.compile(r"context\s*_length\s*_exceeded|ContextWindowExceededError")
#: How much of the end of each agent log is searched.
AGENT_LOG_TAIL_BYTES = 64 * 1024


def _ran_out_of_context(results: Any) -> bool:
    """Whether a trial's agent stopped on a context-length error: by Harbor's exception, its message or the agent's log."""
    exception = results.exception_info.exception_type if results.exception_info else None
    if exception in CONTEXT_LENGTH_EXCEPTIONS:
        return True
    if exception != "NonZeroAgentExitCodeError":
        return False
    if CONTEXT_LENGTH_OUTPUT.search(results.exception_info.exception_message or ""):
        return True
    uri = urlparse(getattr(results, "trial_uri", None) or "")
    if uri.scheme != "file":
        return False
    for log in sorted(Path(url2pathname(uri.path), "agent").glob("*.txt")):
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - AGENT_LOG_TAIL_BYTES))
            if CONTEXT_LENGTH_OUTPUT.search(f.read().decode(errors="replace")):
                return True
    return False


#: skycap authenticates nothing, but LiteLLM won't build a client without a key.
PLACEHOLDER_API_KEY = "skycap"


class HarborSkycapGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: Any,
        harbor_cfg: Dict[str, Any],
        pool: CapturePool,
        inference_engine_client: Any = None,
        train_paths: str = "all",
        records: Optional[RecordLog] = None,
        include_images: bool = False,
        check_first_batch: bool = True,
        stored_pools: Optional[StoredPools] = None,
        support_id_bits: Optional[int] = None,
    ) -> None:
        """
        Args:
            generator_cfg: the run's generator config.
            harbor_cfg: Harbor's ``TrialConfig`` template.
            pool: the client ``generate()`` spreads trajectories over, bringing their samples back here
                (``SkycapServers.http_pool()``).
            inference_engine_client: read for its ``weight_version``, which keys the prefix-cache salt.
            train_paths: the skycap path rule every trajectory is finished with: ``all``, ``final``, or a custom
                rule's ``"pkg.module:function"``, which the servers must have been started with.
            records: where every trajectory opened is logged, per phase, for the W&B record index.
            include_images: the model takes images; each row carries its path's ``pixel_values`` and ``image_grid_thw``.
            check_first_batch: raise when not one trajectory of the first ``generate`` or ``generate_to_ray``
                call has anything to train (``skycap.check_first_batch``).
            stored_pools: the same servers' actor clients (``SkycapServers.stored_pools()``). With them,
                ``generate_to_ray`` finishes and composes each group on one server and leaves it in that
                node's object store.
            support_id_bits: the policy vocabulary's id width (``support_id_bits(vocab_size)``), at
                which a stored group's sample-support member ids are packed.
        """
        # Imported here too, so a bad import path fails at startup rather than at the first finish.
        load_rule(train_paths)
        self.train_paths = train_paths
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
        self.generator_cfg = generator_cfg
        self._routed_experts = bool(getattr(generator_cfg.inference_engine, "enable_return_routed_experts", False))
        self.pool = pool
        self.stored_pools = stored_pools
        self.support_id_bits = support_id_bits
        self.inference_engine_client = inference_engine_client
        self.records = records
        served = generator_cfg.inference_engine.served_model_name
        if served is None or "/" in served:
            raise ValueError("generator.inference_engine.served_model_name must be set, without '/'")
        self._served_model_name = served

        self._template = deepcopy(harbor_cfg)
        agent = self._template.setdefault("agent", {})
        self._in_sandbox = runs_in_sandbox(agent)
        self._warned_unexposed = False
        if self._in_sandbox:
            # Harbor's mini-swe-agent refuses to start unless this process's environment holds a key for the
            # model's provider, and ``hosted_vllm`` has none it knows; skycap checks no key.
            os.environ.setdefault("MSWEA_API_KEY", PLACEHOLDER_API_KEY)
        agent["model_name"] = f"hosted_vllm/{served}"
        kwargs = agent.setdefault("kwargs", {})
        # skycap has the tokens exactly; asking Harbor for them too is what forces the sibling to ban summarization.
        kwargs.pop("collect_rollout_details", None)
        self.include_images = include_images
        self._check_first_batch = check_first_batch
        self._first_batch_done = False
        self._rate_limiter = create_rate_limiter(getattr(generator_cfg, "rate_limit", None))

    async def close(self) -> None:
        """Close the pools' HTTP sessions. The generator can't generate afterwards."""
        await self.pool.close()
        if self.stored_pools is not None:
            await self.stored_pools.close()

    def _cache_salt(self) -> Optional[str]:
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        return build_vllm_cache_salt(weight_version, self._served_model_name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Run the batch over every server and compose it here (eval, unstreamed training)."""
        outcomes, metrics = await self._trials(input_batch, self.pool, [], disable_tqdm)
        output = compose(outcomes, **self._compose_options())
        output["rollout_metrics"].update(metrics)
        return output

    async def generate_to_ray(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> StoredRolloutGroup:
        if self.stored_pools is None:
            return await super().generate_to_ray(input_batch, disable_tqdm=disable_tqdm)
        if self.include_images:
            raise ValueError("a stored group carries no vision features: skycap.include_images needs generate()")
        return await self.stored_pools.run(lambda pool: self._generate_stored(pool, input_batch, disable_tqdm))

    async def _generate_stored(
        self, pool: StoredCapturePool, input_batch: GeneratorInput, disable_tqdm: bool
    ) -> StoredRolloutGroup:
        finished: List[str] = []
        try:
            outcomes, metrics = await self._trials(input_batch, pool, finished, disable_tqdm)
        except BaseException:
            # Nothing will be composed: drop the samples the server cached for this group.
            pool.discard(finished)
            raise
        composed = {o.capture_id for o in outcomes}
        discard = [trajectory_id for trajectory_id in finished if trajectory_id not in composed]
        group = await pool.compose_group(outcomes, discard, self._compose_options(), self.support_id_bits)
        group.group_fields["rollout_metrics"].update(metrics)
        return group

    def _compose_options(self) -> Dict[str, Any]:
        return dict(
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
            top_k=self.generator_cfg.sampling_params.top_k,
            sample_support=getattr(self.generator_cfg.inference_engine, "enable_return_sample_support_set", False),
            routed_experts=self._routed_experts,
            include_images=self.include_images,
        )

    async def _trials(
        self, input_batch: GeneratorInput, pool: CapturePool, finished: List[str], disable_tqdm: bool
    ) -> tuple[List[TrialOutcome], Dict[str, Any]]:
        """Run every trial of the batch on ``pool``, recording each trajectory it finishes in ``finished``."""
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(f"Prompt count ({len(prompts)}) doesn't match trajectory_ids ({len(trajectory_ids)})")
        metadata = input_batch.get("batch_metadata")
        step = getattr(metadata, "global_step", None)
        phase = getattr(metadata, "training_phase", "train")
        cache_salt = self._cache_salt()

        outcomes: List[Optional[TrialOutcome]] = [None] * len(prompts)
        attempts = [TrialAttempts() for _ in prompts]
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def worker(index: int, prompt: ConversationType, trajectory_id: TrajectoryID) -> None:
            outcomes[index] = await self._trial(
                pool, finished, prompt, trajectory_id, cache_salt, step, phase, attempts[index]
            )
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as group:
                for index, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    group.create_task(worker(index, prompt, trajectory_id))
        finally:
            progress.close()
        if self._check_first_batch and not self._first_batch_done:
            self._first_batch_done = True
            # A trial with something to train names its capture, whether its samples came back or stayed on the server.
            if not any(outcome.capture_id is not None for outcome in outcomes):
                raise RuntimeError(
                    f"No trajectory in the first batch ({len(outcomes)}) has anything to train; each trial's "
                    f"exception.txt and agent logs are under {self._template.get('trials_dir')}. "
                    "skycap.check_first_batch=false skips this check."
                )
        return outcomes, trial_metrics(attempts)

    async def _trial(
        self,
        pool: CapturePool,
        finished: List[str],
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        phase: str,
        attempts: TrialAttempts,
    ) -> TrialOutcome:
        """One rollout, retried on unknown errors. Never raises: one failure must not cancel the batch.

        A path rule that raises is not retried: the rule runs on what the trial captured, so another run of the
        trial would mostly pay for the sandbox and the agent again to fail the same way. The rollout is masked.
        """
        started = time.monotonic()
        missing_routes = False
        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            attempts.start()
            try:
                outcome = await self._attempt(
                    pool, finished, prompt, trajectory_id, cache_salt, step, phase, attempt, attempts
                )
            except PathRuleError as error:
                logger.error(f"{prefix}: path rule {self.train_paths!r} failed, not retrying: {error}")
                attempts.fail(error)
                break
            except Exception as error:  # noqa: BLE001 - retried, then masked
                logger.warning(f"{prefix} failed: {type(error).__name__}: {error}")
                attempts.fail(error)
                continue
            outcome.e2e_time = time.monotonic() - started
            if outcome.stop_reason != "error":
                return outcome
            missing_routes = missing_routes or outcome.missing_routes
            logger.warning(f"{prefix} produced nothing to train on")
        return TrialOutcome(
            trajectory_id=trajectory_id,
            stop_reason="error",
            e2e_time=time.monotonic() - started,
            missing_routes=missing_routes,
        )

    async def _attempt(
        self,
        pool: CapturePool,
        finished: List[str],
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        phase: str,
        attempt: int,
        attempts: TrialAttempts,
    ) -> TrialOutcome:
        """One attempt on its own trajectory, so a retry never shares a graph with the attempt it replaces."""
        meta = {
            "task": str(prompt),
            "instance_id": str(trajectory_id.instance_id),
            "repetition_id": trajectory_id.repetition_id,
            "step": step,
            "attempt": attempt,
        }
        trajectory = None
        try:
            # The trajectory opens only once the trial may run: one queued behind the rate limiter for longer
            # than skycap.ttl would otherwise be abandoned as idle before its agent's first call.
            async with self._rate_limiter, pool.trajectory(meta, paths=self.train_paths) as trajectory:
                finished.append(trajectory.id)
                config = self._trial_config(prompt, self._agent_url(trajectory), cache_salt, trajectory.api_key)
                results = await (await Trial.create(TrialConfig.model_validate(config))).run()

                attempts.record(results)
                exception = results.exception_info.exception_type if results.exception_info else None
                if exception == "AgentTimeoutError":
                    # Masked, not retried, as the sibling does.
                    reward, stop_reason = 0.0, "agent_timeout"
                elif await asyncio.to_thread(_ran_out_of_context, results):
                    # Trains with reward 0, as the sibling does.
                    reward, stop_reason = 0.0, "context_length"
                elif not results.verifier_result:
                    reward, stop_reason = 0.0, "error"
                    logger.warning(f"Trajectory {trajectory_id} has no verifier result: {results.exception_info}")
                else:
                    reward, stop_reason = float(results.verifier_result.rewards["reward"]), "complete"
                await trajectory.finish({"reward": reward, "stop_reason": stop_reason})
        finally:
            # Every attempt that got a trajectory, including one that raised: the pool finished it.
            if self.records is not None and trajectory is not None:
                self.records.add(phase, trajectory_id, attempt, trajectory)

        result = trajectory.result
        if result.status != "finished":
            # The trajectory failed inside skycap (e.g. an unattributable prompt): its samples may miss a turn.
            logger.warning(f"Trajectory {trajectory_id}: skycap status {result.status!r}, not training on it")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error")
        # A stored pool's finish brings back the checks; samples brought back are checked here.
        checks = result.checks if isinstance(result, CapturedFinish) else TrialChecks.of(result.samples)
        if not checks.trainable:
            # Nothing generated (e.g. the harness failed before its first call): retried, then masked,
            # so the trial's reward doesn't enter the group without tokens behind it.
            logger.warning(f"Trajectory {trajectory_id}: skycap captured no trainable tokens")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error")
        # Overlong filtering clears a context-length trial's loss mask, so it trains nothing and needs no routes.
        filtered = stop_reason == "context_length" and self.generator_cfg.apply_overlong_filtering
        if self._routed_experts and not filtered and checks.missing_routes:
            # skycap drops a path's routes when any node on it lacks them; replaying the rest would
            # train those tokens on routes the rollout never took. Retried, then masked.
            logger.warning(f"Trajectory {trajectory_id}: a trained path has no routed experts")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error", missing_routes=True)
        return TrialOutcome(
            trajectory_id=trajectory_id,
            samples=[] if isinstance(result, CapturedFinish) else result.samples,
            reward=reward,
            stop_reason=stop_reason,
            unbridged_calls=result.unbridged_calls,
            capture_id=trajectory.id,
        )

    def _agent_url(self, trajectory: Any) -> str:
        """Where the agent calls its trajectory: on the server's exposed URL when it runs in its sandbox."""
        if self._in_sandbox and trajectory.exposed_base_url is not None:
            return trajectory.exposed_base_url
        if self._in_sandbox and not self._warned_unexposed:
            # Fine for a sandbox on this network (Docker); a remote one can't reach the server's own URL.
            logger.warning(
                f"{self._template['agent'].get('name')} runs inside its sandbox, but skycap isn't exposed "
                "(skycap.exposure.type=none): it gets the server's own URL, which a remote sandbox can't reach"
            )
            self._warned_unexposed = True
        return trajectory.base_url

    def _trial_config(
        self,
        prompt: ConversationType,
        base_url: str,
        cache_salt: Optional[str],
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The trial's Harbor config. ``api_key`` is the trajectory's own key (``skycap.require_api_key``)."""
        config = deepcopy(self._template)
        config["task"] = {"path": prompt}
        key = api_key or PLACEHOLDER_API_KEY
        if self._in_sandbox:
            # Installed agents read their endpoint and key from the sandbox's environment, not from agent kwargs.
            # mini-swe-agent sends MSWEA_API_KEY over a provider's key, so it carries the trajectory's key too.
            env = config["agent"].setdefault("env", {})
            for provider in ("OPENAI", "HOSTED_VLLM"):
                env[f"{provider}_API_BASE"] = base_url
                env[f"{provider}_API_KEY"] = key
            # Codex and Harbor's other OpenAI-SDK-style agents read OPENAI_BASE_URL, not the LiteLLM names above.
            env["OPENAI_BASE_URL"] = base_url
            env["MSWEA_API_KEY"] = key
        kwargs = config["agent"]["kwargs"]
        kwargs["api_base"] = base_url
        llm_kwargs = kwargs.setdefault("llm_kwargs", {})
        # Terminus-2 takes `api_base` itself but passes a key only through `llm_kwargs`.
        llm_kwargs["api_key"] = key
        if cache_salt is not None:
            # LiteLLM merges `extra_body` into the request body, where skycap reads `cache_salt` and forwards it.
            extra_body = llm_kwargs.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
            extra_body["cache_salt"] = cache_salt
        return config


def runs_in_sandbox(agent: Dict[str, Any]) -> bool:
    """Whether Harbor's agent ``agent`` (a ``TrialConfig.agent``) is installed in the sandbox and calls the model
    from there, as mini-swe-agent and Claude Code do, rather than from this process, as Terminus-2 does."""
    # As Harbor picks the agent (AgentFactory.create_agent_from_config): a known name wins over an import path.
    name, import_path = agent.get("name"), agent.get("import_path")
    if name is not None and name in AgentName.values():
        cls = AgentFactory.get_agent_class(AgentName(name))
    elif import_path:
        module, _, attribute = import_path.partition(":")
        cls = getattr(importlib.import_module(module), attribute)
    else:
        return False
    return isinstance(cls, type) and issubclass(cls, BaseInstalledAgent)
