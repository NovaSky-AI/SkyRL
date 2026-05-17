"""
RLM-specific generator: extends SkyRLGymGenerator via two hooks plus a thin
``generate`` wrapper that resolves batch-level RLM overrides once.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
)

from examples.train.rlm.openrouter_client import OpenRouterInferenceClient
from skyrl.train.generators.base import TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
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
    env_class: str  # the env id this rollout is using (used to recurse children with the same env)
    trajectory_id: Optional[str]  # shared across the whole tree
    parent_rid: Optional[str]  # None for root
    depth: int  # 0 for root, +1 per level
    child_index: Optional[int]  # None for root; assigned at registration
    output: Optional["StepWiseOutput"] = None


class RLMGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator extended for the RLM environment.

    Lives entirely in user code (``examples/train/rlm/``). Plugs into the base
    via two hooks:

    * ``_setup_env_extras`` — register the rollout context, inject callbacks.
    * ``_post_process_agent_loop_output`` — stamp per-step RLM metadata, stash
      output on the context, inline child trajectories (whose rewards were
      already assigned by their own ``agent_loop`` calls), and tear down the
      rollout tree.  Runs after the base class assigns per-step rewards, so
      the upstream reward code needs no modification.

    Per-rollout state lives in ``self.active_rollouts``, a dict keyed by an
    opaque ``rid`` minted in ``_setup_env_extras``.  Each parent and each child
    rollout gets its own entry; children link back via ``parent_rid``, and the
    parent->children direction is reconstructed by scanning ``active_rollouts``.
    The whole subtree is popped in the root's ``_post_process_agent_loop_output``.

    Subclasses building new RLM-style tasks should add their env id to
    ``RLM_ENV_CLASSES`` so the hooks below recognize it.
    """

    # Env-class ids this generator should treat as RLM-shaped. Subclass and
    # extend if you register a new BaseRLMEnv subclass with a different id.
    RLM_ENV_CLASSES: frozenset = frozenset({"evidence_rlm", "multipaper_evidence_rlm"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.generator_cfg.step_wise_trajectories:
            raise ValueError(
                "RLMGymGenerator requires step_wise_trajectories=True. "
                "Multiple code paths assume step-wise mode (ephemeral "
                "user-prompt injection, output type assertions, child "
                "trajectory inlining). Non-step-wise mode is not supported."
            )
        # Per-rollout registry keyed by rid. Populated in _setup_env_extras /
        # _run_child; the whole subtree is popped in the root's
        # _finalize_episode. _rollout_lock serialises child registration so
        # the count-then-insert in _register_rollout is atomic.
        self.active_rollouts: Dict[str, _RLMRolloutContext] = {}
        self._rollout_lock = threading.Lock()
        self.frozen_inference_engine: Optional[OpenRouterInferenceClient] = self._build_frozen_inference_engine()

    def _build_frozen_inference_engine(self) -> Optional[OpenRouterInferenceClient]:
        """Build a frozen OpenRouter client for in-REPL ``llm_query`` calls.

        Returns ``None`` when ``frozen_openrouter_model`` is not set, in which
        case ``_make_lm_callback`` falls back to the policy engine.
        """
        model = getattr(self.generator_cfg, "frozen_openrouter_model", None)
        if model is None:
            return None
        return OpenRouterInferenceClient.from_model(model=model, tokenizer=self.tokenizer)

    # ------------------------------------------------------------------
    # Hook 1: env-extras setup (runs inside agent_loop, before env construction)
    # ------------------------------------------------------------------

    def _setup_env_extras(
        self,
        env_class: str,
        env_extras: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
        trajectory_id: Optional[TrajectoryID],
    ) -> Dict[str, Any]:
        """Register this rollout in ``active_rollouts`` and inject hook callables.

        Sole entry point for registration. Roots arrive with no extras keys set
        and get a fresh root context. Children arrive with ``_rlm_parent_rid``
        stamped by ``_run_child`` (consumed and stripped here) and get a child
        context that inherits ``env_class`` / ``trajectory_id`` from the parent.
        """
        if env_class not in self.RLM_ENV_CLASSES:
            return env_extras

        env_extras = dict(env_extras)

        parent_rid = env_extras.pop("_rlm_parent_rid", None)
        ctx = self._register_rollout(env_class, trajectory_id, parent_rid)
        env_extras["rlm_rollout_id"] = ctx.rid
        env_extras["depth"] = ctx.depth

        loop = asyncio.get_running_loop()
        env_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params, ctx.rid)
        if getattr(self.generator_cfg, "enable_child_agents", True):
            env_extras["subcall_fn"] = self._make_subcall_fn(loop, env_extras, sampling_params, ctx.rid)

        return env_extras

    # ------------------------------------------------------------------
    # Hook 2: post-reward output assembly
    # ------------------------------------------------------------------

    def _post_process_agent_loop_output(
        self,
        agent_loop_output,
        env_extras: Dict[str, Any],
        trajectory_id: Optional[TrajectoryID],
    ):
        """Stamp RLM metadata, stash output, inline children, and tear down the tree.

        Runs after the base class assigns per-step rewards.  Children already
        carry their own rewards from their independent ``agent_loop`` calls.
        """
        rid = env_extras.get("rlm_rollout_id")
        ctx = self.active_rollouts.get(rid) if rid else None
        if ctx is None:
            return agent_loop_output

        assert isinstance(
            agent_loop_output, StepWiseOutput
        ), f"RLMGymGenerator requires step_wise_trajectories=True, got {type(agent_loop_output).__name__}"

        for step_index, step in enumerate(agent_loop_output.step_outputs):
            step.env_metrics["rlm_metadata"] = {
                "trajectory_id": ctx.trajectory_id,
                "depth": ctx.depth,
                "child_index": ctx.child_index,
                "step_index": step_index,
            }

        # Mark this rollout's final step as a trajectory boundary for observability.
        # Note that this is different from is_last_step, which is used for training.
        if agent_loop_output.step_outputs:
            agent_loop_output.step_outputs[-1].env_metrics["is_trajectory_boundary"] = True

        ctx.output = agent_loop_output

        # Non-root: parent/root will inline us later.
        if ctx.parent_rid is not None:
            for k in ("lm_callback", "subcall_fn"):
                env_extras.pop(k, None)
            return agent_loop_output

        # ---- Root: coalesce children and tear down ----
        descendants = self._dfs_descendants(rid)

        if getattr(self.generator_cfg, "train_child_trajectories", False) and descendants:
            children_flat: List[TrajectoryOutput] = []
            for d in descendants:
                if d.output is None:
                    continue
                assert isinstance(
                    d.output, StepWiseOutput
                ), f"Child rollout output must be StepWiseOutput, got {type(d.output).__name__}"
                children_flat.extend(d.output.step_outputs)
            if children_flat:
                agent_loop_output.step_outputs = children_flat + agent_loop_output.step_outputs

        for r in [rid, *(d.rid for d in descendants)]:
            self.active_rollouts.pop(r, None)

        for k in ("lm_callback", "subcall_fn"):
            env_extras.pop(k, None)
        return agent_loop_output

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
        rid: Optional[str],
    ) -> Callable[[List[str]], List[str]]:
        """Sync callback that dispatches batched text prompts to an inference engine.

        Safe to call from a non-async thread (e.g. inside a REPL ``exec()``).
        Uses ``frozen_inference_engine`` when configured; otherwise falls back
        to the policy engine.
        """
        target_engine = self.frozen_inference_engine or self.inference_engine_client

        async def _generate(prompts: List[str]) -> List[str]:
            if rid is not None and rid not in self.active_rollouts:
                # Owner rollout was torn down — bail before hitting the engine.
                # Firing now risks landing on a partially-woken vLLM during
                # sync_weights and crashing the EngineCore.
                raise asyncio.CancelledError(f"rollout {rid} torn down before lm_query")
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
            try:
                return future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                # Cancel the underlying asyncio task so it doesn't keep firing HTTP
                # requests after we've given up — orphans queued on a sleeping engine
                # crash vLLM on partial wake.
                future.cancel()
                raise

        return callback

    def _make_subcall_fn(
        self,
        loop: asyncio.AbstractEventLoop,
        env_extras: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
        parent_rid: str,
    ) -> Callable[[str], str]:
        """Build a sync ``subcall_fn`` exposed to the parent's REPL as ``rlm_query``.

        Each invocation spawns a child ``agent_loop`` with ``_rlm_parent_rid``
        set on its extras; the child's ``_setup_env_extras`` consumes that
        sentinel and registers the child context (linked to ``parent_rid``).
        The child's ``_finalize_episode`` parks its output on its context;
        the root's finalize collects it via the tree.
        """

        async def _run_child(prompt: str, context=None) -> str:
            parent_ctx = self.active_rollouts.get(parent_rid)
            if parent_ctx is None:
                logger.warning(
                    f"_run_child: parent rollout {parent_rid!r} was never registered. Returning from subcall early."
                )
                return ""

            child_extras = dict(env_extras)
            child_extras["_rlm_parent_rid"] = parent_rid

            if context is not None:
                child_extra_info = dict(child_extras.get("extra_info", {}) or {})
                if isinstance(context, str):
                    child_extra_info["context_text"] = context
                else:
                    child_extra_info["context_text"] = json.dumps(context)
                child_extras["extra_info"] = child_extra_info

            result = await self.agent_loop(
                prompt=[{"role": "user", "content": prompt}],
                env_class=parent_ctx.env_class,
                env_extras=child_extras,
                max_tokens=self.generator_cfg.sampling_params.max_generate_length,
                max_input_length=self.generator_cfg.max_input_length,
                sampling_params=sampling_params,
            )

            assert isinstance(
                result, StepWiseOutput
            ), f"Child agent_loop must return StepWiseOutput, got {type(result).__name__}"
            child_env_metrics = result.step_outputs[-1].env_metrics if result.step_outputs else {}
            return child_env_metrics.get("final_answer") or ""

        def subcall_fn(prompt: str, context=None) -> str:
            future = asyncio.run_coroutine_threadsafe(_run_child(prompt, context=context), loop)
            try:
                return future.result(timeout=600)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise

        return subcall_fn

    def _register_rollout(
        self,
        env_class: str,
        trajectory_id: Optional[TrajectoryID],
        parent_rid: Optional[str],
    ) -> "_RLMRolloutContext":
        """Create and register an _RLMRolloutContext, returning it.

        Root rollouts (parent_rid is None) get a fresh context with depth 0.
        Child rollouts inherit env_class/trajectory_id from their parent.
        """
        rid = uuid.uuid4().hex[:8]
        if parent_rid is None:
            tid = trajectory_id.to_string() if trajectory_id is not None else None
            ctx = _RLMRolloutContext(
                rid=rid,
                env_class=env_class,
                trajectory_id=tid,
                parent_rid=None,
                depth=0,
                child_index=None,
            )
            self.active_rollouts[rid] = ctx
        else:
            with self._rollout_lock:
                parent = self.active_rollouts[parent_rid]
                child_index = sum(1 for c in self.active_rollouts.values() if c.parent_rid == parent_rid)
                ctx = _RLMRolloutContext(
                    rid=rid,
                    env_class=parent.env_class,
                    trajectory_id=parent.trajectory_id,
                    parent_rid=parent_rid,
                    depth=parent.depth + 1,
                    child_index=child_index,
                )
                self.active_rollouts[rid] = ctx
        return ctx
