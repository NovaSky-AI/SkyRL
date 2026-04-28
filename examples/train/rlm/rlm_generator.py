"""
RLM-specific generator: extends SkyRLGymGenerator via four hooks plus a thin
``generate`` wrapper that resolves batch-level RLM overrides once.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)

from skyrl.backends.skyrl_train.inference_engines.openrouter_engine import OpenRouterInferenceEngine
from skyrl.train.generators.base import TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    AgentLoopState,
    SkyRLGymGenerator,
    StepWiseOutput,
    TrajectoryOutput,
)


@dataclass
class _RLMRolloutContext:
    """Per-rollout (per-tree-node) state for an RLM rollout.

    One entry lives in ``RLMGymGenerator.active_rollouts`` keyed by ``rid`` for
    the lifetime of an ``agent_loop`` invocation. Roots and children share the
    same shape; the parent->children direction is reconstructed by scanning
    ``active_rollouts`` for matching ``parent_rid`` (insertion order preserves
    registration order, which preserves child order).
    """

    rid: str
    env_class: str                            # the env id this rollout is using (used to recurse children with the same env)
    trajectory_id: Optional[str]              # shared across the whole tree
    parent_rid: Optional[str]                 # None for root
    depth: int                                # 0 for root, +1 per level
    child_index: Optional[int]                # None for root; assigned at registration
    output: Optional[Union["StepWiseOutput", "TrajectoryOutput"]] = None


class RLMGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator extended for the RLM environment.

    Lives entirely in user code (``examples/train/rlm/``). Plugs into the base
    via three hooks: ``_setup_env_extras``, ``_finalize_episode``, and
    ``_call_inference_engine``.

    Per-rollout state — including children and judge scores — lives in
    ``self.active_rollouts``, a dict keyed by an opaque ``rid`` minted in
    ``_setup_env_extras`` and threaded through ``env_extras["rlm_rollout_id"]``.
    Each parent and each child rollout gets its own entry; children link back
    to their parent via ``parent_rid``, and the parent->children direction is
    reconstructed at finalize time by scanning ``active_rollouts``. The whole
    subtree is popped in the root's ``_finalize_episode``.

    Children are inlined into the root's ``step_outputs`` at the end of
    ``_finalize_episode`` (root branch only) so the base flattener in
    ``generate()`` picks them up without needing a flatten hook override.

    Subclasses building new RLM-style tasks should add their env id to
    ``RLM_ENV_CLASSES`` so the hooks below recognize it.
    """

    # Env-class ids this generator should treat as RLM-shaped. Subclass and
    # extend if you register a new BaseRLMEnv subclass with a different id.
    RLM_ENV_CLASSES: frozenset = frozenset({"evidence_rlm", "multipaper_evidence_rlm"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-rollout registry keyed by rid. Populated in _setup_env_extras /
        # _run_child; the whole subtree is popped in the root's
        # _finalize_episode. CPython dict ops on distinct keys are atomic so no
        # lock is needed for set/get/pop; do not iterate concurrently with
        # mutation.
        self.active_rollouts: Dict[str, _RLMRolloutContext] = {}
        # Optional frozen engine used by non-root rollouts when
        # ``generator_cfg.child_openrouter_model`` is set. ``None`` means
        # children inherit the policy engine.
        self.frozen_inference_engine: Optional[InferenceEngineInterface] = self._build_frozen_inference_engine()

    # ------------------------------------------------------------------
    # Hook 1: env-extras setup (runs inside agent_loop, before env construction)
    # ------------------------------------------------------------------

    def _setup_env_extras(
        self,
        env_class: str,
        env_extras: Dict[str, Any],
        prompt: ConversationType,
        sampling_params: Optional[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        trajectory_id: Optional[TrajectoryID],
    ) -> Dict[str, Any]:
        """Register this rollout in ``self.active_rollouts`` and inject hook callbacks.

        Two entry paths:
        - Root rollout: no ``rlm_rollout_id`` on ``env_extras`` yet. Mint a fresh
          rid and register a new root context.
        - Child rollout: ``_run_child`` has already registered the child context
          and stamped ``env_extras["rlm_rollout_id"]`` before calling
          ``agent_loop``. Reuse the existing context; just (re)inject callbacks
          since the child's ``env_extras`` was stripped of parent callables.
        """
        if env_class not in self.RLM_ENV_CLASSES:
            return env_extras

        env_extras = dict(env_extras)

        rid = env_extras.get("rlm_rollout_id")
        if rid is None:
            # Root: mint id and register fresh context.
            rid = uuid.uuid4().hex[:8]
            env_extras["rlm_rollout_id"] = rid
            self.active_rollouts[rid] = _RLMRolloutContext(
                rid=rid,
                env_class=env_class,
                trajectory_id=trajectory_id.to_string() if trajectory_id is not None else None,
                parent_rid=None,
                depth=0,
                child_index=None,
            )
        # else: child context already registered by _run_child; nothing to add here.

        env_extras["depth"] = self.active_rollouts[rid].depth

        enable_child_agents = getattr(self.generator_cfg, "enable_child_agents", True)

        loop = asyncio.get_running_loop()
        if "lm_callback" not in env_extras:
            # In-REPL llm_query() always goes through the frozen engine when
            # configured; otherwise falls back to the policy engine.
            env_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params)
            if enable_child_agents:
                env_extras["subcall_fn"] = self._make_subcall_fn(
                    loop, env_extras, sampling_params, max_tokens, max_input_length, rid,
                )

        return env_extras

    def _build_frozen_inference_engine(self) -> Optional[InferenceEngineInterface]:
        """Build the frozen OpenRouter engine if ``child_openrouter_model`` is set.

        Shared across all non-root rollouts. ``None`` means children inherit
        the policy engine.
        """
        model = getattr(self.generator_cfg, "child_openrouter_model", None)
        if model is None:
            return None
        return OpenRouterInferenceEngine(
            model=model,
            tokenizer=self.tokenizer,
            base_url=getattr(self.generator_cfg, "child_openrouter_base_url", "https://openrouter.ai/api/v1"),
        )

    # ------------------------------------------------------------------
    # Hook 3: per-rollout inference engine dispatch
    # ------------------------------------------------------------------

    async def _call_inference_engine(
        self,
        engine_input: InferenceEngineInput,
        env_extras: Dict[str, Any],
    ) -> InferenceEngineOutput:
        """Route non-root rollouts to ``self.frozen_inference_engine`` if configured.

        Reads ``rlm_rollout_id`` off ``env_extras`` and looks up the context in
        ``self.active_rollouts``. Roots always run on the policy engine; only
        non-root nodes route through the frozen engine.
        """
        rid = env_extras.get("rlm_rollout_id")
        ctx = self.active_rollouts.get(rid) if rid else None
        if ctx is not None and ctx.parent_rid is not None and self.frozen_inference_engine is not None:
            return await self.frozen_inference_engine.generate(engine_input)
        return await super()._call_inference_engine(engine_input, env_extras)

    # ------------------------------------------------------------------
    # Hook 3: post-episode finalize
    # ------------------------------------------------------------------

    async def _finalize_episode(
        self,
        env,
        env_class: str,
        agent_loop_state: AgentLoopState,
        agent_loop_output,
        env_extras: Dict[str, Any],
        prompt: ConversationType,
        trajectory_id: Optional[TrajectoryID],
    ) -> Dict[str, Any]:
        """Finalize one node of the rollout tree.

        Non-root nodes do the minimum: stamp their own per-step ``rlm_metadata``
        from fields already on their context, and stash ``agent_loop_output``
        on their context for the root to pick up.

        The root does the structural work: walk the tree via ``_dfs_descendants``
        (DFS preorder by registration order), inline descendants' step_outputs,
        promote the trajectory-level summary onto the root's final step,
        compute ``child_rlm_metrics`` over the whole tree, then pop the entire
        subtree from ``self.active_rollouts``.
        """
        if env_class not in self.RLM_ENV_CLASSES:
            return await super()._finalize_episode(
                env, env_class, agent_loop_state, agent_loop_output, env_extras, prompt, trajectory_id
            )

        env_metrics = env.get_metrics()

        rid = env_extras.get("rlm_rollout_id")
        ctx = self.active_rollouts.get(rid) if rid else None
        if ctx is None:
            # Defensive: should not happen if _setup_env_extras ran. Fall back to
            # base behavior so we don't crash a training run.
            logger.warning(f"_finalize_episode: missing context for rid={rid!r}")
            return env_metrics

        # Local: stamp per-step rlm_metadata using fields already on ctx.
        if isinstance(agent_loop_output, StepWiseOutput):
            for step_index, step in enumerate(agent_loop_output.step_outputs):
                step.env_metrics["rlm_metadata"] = {
                    "trajectory_id": ctx.trajectory_id,
                    "depth": ctx.depth,
                    "child_index": ctx.child_index,
                    "step_index": step_index,
                }

        # Stash output for an ancestor to read.
        ctx.output = agent_loop_output

        # Non-root: done. Parent / root will inline us.
        if ctx.parent_rid is not None:
            return env_metrics

        # ---- Root branch: tree-walk and collapse ----
        descendants = self._dfs_descendants(rid)  # preorder, excludes root

        # Inline descendants' step_outputs ahead of the root's. Root's last step
        # remains the final entry so is_last_step still marks the trajectory end.
        if isinstance(agent_loop_output, StepWiseOutput) \
           and getattr(self.generator_cfg, "train_child_trajectories", False) \
           and descendants:
            children_flat: List[TrajectoryOutput] = []
            for d in descendants:
                if d.output is None or not isinstance(d.output, StepWiseOutput):
                    continue
                children_flat.extend(d.output.step_outputs)
            if children_flat:
                agent_loop_output.step_outputs = children_flat + agent_loop_output.step_outputs

        # Tear down the whole subtree (root + every descendant) in one pass.
        for r in [rid, *(d.rid for d in descendants)]:
            self.active_rollouts.pop(r, None)

        # Drop callables and the (potentially huge) context payload off env_extras
        # so it's JSON-serializable for dump_per_dataset_eval_results.
        for k in ("lm_callback", "subcall_fn", "rlm_rollout_id"):
            env_extras.pop(k, None)

        return env_metrics

    def _dfs_descendants(self, rid: str) -> List[_RLMRolloutContext]:
        """DFS preorder over descendants of ``rid``, in registration order. Excludes ``rid`` itself."""
        children_by_parent: Dict[str, List[str]] = {}
        for r, ctx in self.active_rollouts.items():
            if ctx.parent_rid is not None:
                children_by_parent.setdefault(ctx.parent_rid, []).append(r)

        out: List[_RLMRolloutContext] = []
        stack: List[str] = list(reversed(children_by_parent.get(rid, [])))
        while stack:
            cur = stack.pop()
            ctx = self.active_rollouts.get(cur)
            if ctx is None:
                continue
            out.append(ctx)
            stack.extend(reversed(children_by_parent.get(cur, [])))
        return out

    # ==================================================================
    # RLM-specific helpers (called by hooks above)
    # ==================================================================

    def _make_lm_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        sampling_params: Optional[Dict[str, Any]],
    ) -> Callable[[List[str]], List[str]]:
        """Sync callback that dispatches batched text prompts to an inference engine.

        Safe to call from a non-async thread (e.g. inside a REPL ``exec()``).
        Routes through ``self.frozen_inference_engine`` when configured; falls
        back to the policy engine otherwise.
        """
        target_engine = self.inference_engine_client

        async def _generate(prompts: List[str]) -> List[str]:
            token_ids = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
                for p in prompts
            ]
            engine_input = InferenceEngineInput(
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
            )
            output = await target_engine.generate(engine_input)
            return output["responses"]

        def callback(prompts: List[str]) -> List[str]:
            future = asyncio.run_coroutine_threadsafe(_generate(prompts), loop)
            return future.result(timeout=300)

        return callback

    def _make_subcall_fn(
        self,
        loop: asyncio.AbstractEventLoop,
        env_extras: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        parent_rid: str,
    ) -> Callable[[str], str]:
        """Build a sync ``subcall_fn`` exposed to the parent's REPL as ``rlm_query``.

        Each invocation registers a child ``_RLMRolloutContext`` in
        ``self.active_rollouts`` (linked to ``parent_rid``), then spawns a child
        ``agent_loop``. The child's ``_setup_env_extras`` sees the pre-stamped
        ``rlm_rollout_id`` and reuses the existing context. The child's
        ``_finalize_episode`` parks its output on its context; the root's
        finalize collects it via the tree.

        Task-specific per-child attribution (e.g. paper_id) is the env's job:
        the child env reads ``extras["parent_context"]`` and surfaces whatever
        it wants via its own ``get_metrics()``.
        """
        # Decode the parent's context once so children can reverse-lookup their
        # own attribution against it (e.g. paper_id from paper text).
        parent_context: Any = None
        extra_info = env_extras.get("extra_info") or {}
        if isinstance(extra_info, dict):
            ctx_raw = extra_info.get("context_text")
            if isinstance(ctx_raw, str):
                try:
                    parent_context = json.loads(ctx_raw)
                except (json.JSONDecodeError, ValueError):
                    parent_context = ctx_raw
            else:
                parent_context = ctx_raw

        async def _run_child(prompt: str, context=None) -> str:
            parent_ctx = self.active_rollouts.get(parent_rid)
            if parent_ctx is None:
                # Parent torn down between subcall registration and execution —
                # should not happen, but degrade gracefully.
                logger.warning(f"_run_child: parent context {parent_rid!r} is gone")
                return ""

            child_rid = uuid.uuid4().hex[:8]
            child_ctx = _RLMRolloutContext(
                rid=child_rid,
                env_class=parent_ctx.env_class,
                trajectory_id=parent_ctx.trajectory_id,
                parent_rid=parent_rid,
                depth=parent_ctx.depth + 1,
                child_index=sum(1 for c in self.active_rollouts.values() if c.parent_rid == parent_rid),
            )
            self.active_rollouts[child_rid] = child_ctx

            # Build child env_extras: strip parent callables (child's
            # _setup_env_extras will re-inject), pre-stamp child_rid so the
            # hook reuses the registered context.
            child_extras = {k: v for k, v in env_extras.items() if k not in ("lm_callback", "subcall_fn")}
            child_extras["rlm_rollout_id"] = child_rid
            if parent_context is not None:
                child_extras["parent_context"] = parent_context

            if context is not None:
                child_extra_info = dict(child_extras.get("extra_info", {}) or {})
                if isinstance(context, str):
                    child_extra_info["context_text"] = context
                else:
                    child_extra_info["context_text"] = json.dumps(context)
                child_extras["extra_info"] = child_extra_info
                # Children aren't scored independently; clear evidence-based reward_spec
                # so the child's env builds its own (per-context) reward.
                child_extras["reward_spec"] = {"ground_truth": None}

            result = await self.agent_loop(
                prompt=[{"role": "user", "content": prompt}],
                env_class=parent_ctx.env_class,
                env_extras=child_extras,
                max_tokens=max_tokens,
                max_input_length=max_input_length,
                sampling_params=sampling_params,
            )

            if isinstance(result, StepWiseOutput):
                child_env_metrics = result.step_outputs[-1].env_metrics if result.step_outputs else {}
            else:
                child_env_metrics = result.env_metrics or {}
            return child_env_metrics.get("final_answer") or ""

        def subcall_fn(prompt: str, context=None) -> str:
            future = asyncio.run_coroutine_threadsafe(_run_child(prompt, context=context), loop)
            return future.result(timeout=600)

        return subcall_fn
