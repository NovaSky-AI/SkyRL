"""
RLM-specific generator: extends SkyRLGymGenerator via four hooks plus a thin
``generate`` wrapper that resolves batch-level RLM overrides once.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    AgentLoopState,
    SkyRLGymGenerator,
    StepWiseOutput,
    TrajectoryOutput,
)


_RUBRIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "RubricScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "precision_score": {
                    "type": "integer",
                    "description": "1-10: how tight and accurate the spans are (no extraneous sentences, no off-topic padding)",
                },
                "recall_score": {
                    "type": "integer",
                    "description": "1-10: how thoroughly the evidence covers what is needed to answer the question",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the scores",
                },
            },
            "required": ["precision_score", "recall_score", "reasoning"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class _RLMRolloutContext:
    """Per-rollout (per-tree-node) state for an RLM rollout.

    One entry lives in ``RLMGymGenerator.active_rollouts`` keyed by ``rid`` for
    the lifetime of an ``agent_loop`` invocation. Roots and children share the
    same shape; the tree is reconstructed from ``parent_rid`` / ``children_rids``.
    """

    rid: str
    env_class: str                            # the env id this rollout is using (used to recurse children with the same env)
    trajectory_id: Optional[str]              # shared across the whole tree
    parent_rid: Optional[str]                 # None for root
    depth: int                                # 0 for root, +1 per level
    child_index: Optional[int]                # None for root; assigned at registration
    children_rids: List[str] = field(default_factory=list)
    output: Optional[Union["StepWiseOutput", "TrajectoryOutput"]] = None
    judge_scores: Dict[str, Any] = field(default_factory=dict)
    call_record: Optional[Dict[str, Any]] = None  # set on a child by its parent's _run_child
    child_engine: Optional[InferenceEngineInterface] = None
    child_system_prompt: Optional[str] = None


class RLMGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator extended for the RLM environment.

    Lives entirely in user code (``examples/train/rlm/``). Plugs into the base
    via three hooks: ``_setup_env_extras``, ``_finalize_episode``, and
    ``_call_inference_engine``. A thin ``generate`` wrapper resolves the
    batch-level RLM overrides (``child_system_prompt``) once and stashes them
    on ``self`` for the hooks.

    Per-rollout state — including children, judge scores, and the per-rollout
    inference engine override — lives in ``self.active_rollouts``, a dict keyed
    by an opaque ``rid`` minted in ``_setup_env_extras`` and threaded through
    ``env_extras["rlm_rollout_id"]``. Each parent and each child rollout gets
    its own entry; children link back to their parent via ``parent_rid`` and
    parents track children via ``children_rids``. The whole subtree is popped
    in the root's ``_finalize_episode``.

    Children are inlined into the root's ``step_outputs`` at the end of
    ``_finalize_episode`` (root branch only) so the base flattener in
    ``generate()`` picks them up without needing a flatten hook override.

    Subclasses building new RLM-style tasks should add their env id to
    ``RLM_ENV_CLASSES`` so the hooks below recognize it.
    """

    # Env-class ids this generator should treat as RLM-shaped. Subclass and
    # extend if you register a new BaseRLMEnv subclass with a different id.
    RLM_ENV_CLASSES: frozenset = frozenset({"evidence_rlm"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set per-call by ``generate``; read by hooks. Reset in finally.
        self._batch_rlm_overrides: Dict[str, Optional[str]] = {}
        # Per-rollout registry keyed by rid. Populated in _setup_env_extras /
        # _run_child; the whole subtree is popped in the root's
        # _finalize_episode. CPython dict ops on distinct keys are atomic so no
        # lock is needed for set/get/pop; do not iterate concurrently with
        # mutation.
        self.active_rollouts: Dict[str, _RLMRolloutContext] = {}

    # ------------------------------------------------------------------
    # generate(): thin wrapper that resolves batch-level overrides once
    # ------------------------------------------------------------------

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        rlm_cfg = getattr(self.skyrl_gym_cfg, "rlm", None)
        overrides: Dict[str, Optional[str]] = {"child_system_prompt": None}
        if rlm_cfg is not None:
            cfg_dict = vars(rlm_cfg) if hasattr(rlm_cfg, "__dict__") else dict(rlm_cfg)
            overrides["child_system_prompt"] = cfg_dict.get("child_system_prompt")

        self._batch_rlm_overrides = overrides
        try:
            return await super().generate(input_batch, disable_tqdm)
        finally:
            self._batch_rlm_overrides = {}

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
        env_extras["step_wise"] = self.generator_cfg.step_wise_trajectories

        rid = env_extras.get("rlm_rollout_id")
        if rid is None:
            # Root: mint id and register fresh context.
            rid = uuid.uuid4().hex[:8]
            env_extras["rlm_rollout_id"] = rid
            self.active_rollouts[rid] = _RLMRolloutContext(
                rid=rid,
                env_class=env_class,
                trajectory_id=None,  # filled in by _finalize_episode (it has trajectory_id)
                parent_rid=None,
                depth=0,
                child_index=None,
                child_system_prompt=self._batch_rlm_overrides.get("child_system_prompt"),
                child_engine=self._build_child_engine_if_configured(),
            )
        # else: child context already registered by _run_child; nothing to add here.

        judge_reward_model = getattr(self.generator_cfg, "judge_reward_model", None)
        enable_child_agents = getattr(self.generator_cfg, "enable_child_agents", True)

        loop = asyncio.get_running_loop()
        if "lm_callback" not in env_extras:
            # Non-root nodes route in-REPL llm_query() through child_engine when configured;
            # roots stay on the policy engine. Mirrors the _call_inference_engine routing.
            ctx = self.active_rollouts[rid]
            lm_engine = ctx.child_engine if ctx.parent_rid is not None else None
            env_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params, engine=lm_engine)
            if enable_child_agents:
                env_extras["subcall_fn"] = self._make_subcall_fn(
                    loop, env_extras, sampling_params, max_tokens, max_input_length, rid,
                )

        if judge_reward_model:
            env_extras["judge_reward_fn"] = self._make_judge_reward_fn(loop, env_extras, prompt, rid)

        return env_extras

    def _build_child_engine_if_configured(self) -> Optional[InferenceEngineInterface]:
        """Build a per-rollout OpenRouter engine if ``child_openrouter_model`` is set.

        Returned engine is shared by all children spawned from this root (and
        their descendants). ``None`` means children inherit the policy engine.
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
        """Route to the per-rollout child engine if this rollout is a child.

        Reads ``rlm_rollout_id`` off ``env_extras`` and looks up the context in
        ``self.active_rollouts``. Children carry a ``child_engine`` reference
        (an ``OpenRouterInferenceEngine``) when ``child_openrouter_model`` is
        configured. Roots have ``child_engine`` set too — but they shouldn't
        use it for their *own* generation, only pass it down. Hence: only
        non-root nodes route through it.
        """
        rid = env_extras.get("rlm_rollout_id")
        ctx = self.active_rollouts.get(rid) if rid else None
        if ctx is not None and ctx.parent_rid is not None and ctx.child_engine is not None:
            return await ctx.child_engine.generate(engine_input)
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

        Non-root nodes do the minimum: fold judge_scores into env_metrics, stamp
        their own per-step ``rlm_metadata`` from fields already on their context,
        and stash ``agent_loop_output`` on their context for the root to pick up.

        The root does the structural work: walk the tree via ``children_rids``
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

        # trajectory_id is only available here (not in _setup_env_extras), so
        # backfill it onto the context. For children, _run_child already copied
        # the parent's trajectory_id at registration time; this only matters for
        # roots, where it's None until now.
        tid_str = trajectory_id.to_string() if trajectory_id is not None else None
        if ctx.trajectory_id is None:
            ctx.trajectory_id = tid_str

        # Local: judge scores merge into env_metrics for every node.
        if ctx.judge_scores:
            env_metrics.update(ctx.judge_scores)

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

        # Whole-tree call records → child_rlm_metrics.
        call_records = [d.call_record for d in descendants if d.call_record is not None]
        if call_records:
            from .envs.evidence_rewards import compute_child_rlm_metrics

            reward_spec = env_extras.get("reward_spec") or {}
            evidence = reward_spec.get("evidence") or []
            extra_info = env_extras.get("extra_info") or {}
            ctx_raw = extra_info.get("context_text") if isinstance(extra_info, dict) else None
            parent_context: Dict[str, str] = {}
            if isinstance(ctx_raw, str):
                try:
                    decoded = json.loads(ctx_raw)
                    if isinstance(decoded, dict):
                        parent_context = decoded
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(ctx_raw, dict):
                parent_context = ctx_raw

            if isinstance(evidence, list) and evidence and isinstance(evidence[0], dict) and parent_context:
                child_rlm_metrics = compute_child_rlm_metrics(call_records, evidence, parent_context)
                env_metrics.update(child_rlm_metrics)
                if isinstance(agent_loop_output, StepWiseOutput) and agent_loop_output.step_outputs:
                    agent_loop_output.step_outputs[-1].env_metrics.update(child_rlm_metrics)

        # Promote trajectory-level summary onto root's final step.
        if isinstance(agent_loop_output, StepWiseOutput) and agent_loop_output.step_outputs:
            agent_loop_output.step_outputs[-1].env_metrics["rlm_metadata"].update({
                "final_answer": env_metrics.get("final_answer"),
                "turns_used":   env_metrics.get("turns_used"),
                "evidence":     (env_extras.get("reward_spec") or {}).get("evidence"),
                "num_children": len(ctx.children_rids),
            })

        # Tear down the whole subtree (root + every descendant) in one pass.
        for r in [rid, *(d.rid for d in descendants)]:
            self.active_rollouts.pop(r, None)

        # Drop callables and the (potentially huge) context payload off env_extras
        # so it's JSON-serializable for dump_per_dataset_eval_results.
        extra_info = env_extras.get("extra_info")
        if isinstance(extra_info, dict):
            extra_info.pop("context_text", None)
        for k in ("lm_callback", "subcall_fn", "judge_reward_fn", "rlm_rollout_id"):
            env_extras.pop(k, None)
        reward_spec = env_extras.get("reward_spec")
        if isinstance(reward_spec, dict):
            reward_spec.pop("reward_fn", None)

        return env_metrics

    def _dfs_descendants(self, rid: str) -> List[_RLMRolloutContext]:
        """DFS preorder over descendants of ``rid``, in registration order. Excludes ``rid`` itself."""
        out: List[_RLMRolloutContext] = []
        stack: List[str] = list(reversed(self.active_rollouts[rid].children_rids))
        while stack:
            cur = stack.pop()
            ctx = self.active_rollouts.get(cur)
            if ctx is None:
                continue
            out.append(ctx)
            stack.extend(reversed(ctx.children_rids))
        return out

    # ==================================================================
    # RLM-specific helpers (called by hooks above)
    # ==================================================================

    def _make_lm_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        sampling_params: Optional[Dict[str, Any]],
        engine: Optional[InferenceEngineInterface] = None,
    ) -> Callable[[List[str]], List[str]]:
        """Sync callback that dispatches batched text prompts to an inference engine.

        Safe to call from a non-async thread (e.g. inside a REPL ``exec()``).
        ``engine`` defaults to ``self.inference_engine_client``; child rollouts pass
        the OpenRouter engine when ``child_openrouter_model`` is set.
        """
        target_engine = engine if engine is not None else self.inference_engine_client

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

    def _make_judge_reward_fn(
        self,
        loop: asyncio.AbstractEventLoop,
        env_extras: Dict[str, Any],
        prompt: ConversationType,
        rid: str,
    ) -> Callable[[str], float]:
        """Score the final answer with an LLM judge. Replaces F1 reward when judge_reward_model is set."""
        import ast as _ast
        import json as _json
        import textwrap as _textwrap

        question = " ".join(msg["content"] for msg in prompt if msg.get("content"))

        reward_spec = env_extras.get("reward_spec", {})
        raw_evidence = reward_spec.get("evidence") or []
        gt_strings: List[str] = []
        for ev in raw_evidence:
            if isinstance(ev, str):
                gt_strings.append(ev)
            elif isinstance(ev, dict):
                for sel in ev.get("selections", []):
                    text = sel.get("text", "").strip()
                    if text:
                        gt_strings.append(text)

        model = getattr(self.generator_cfg, "judge_reward_model", None)
        base_url = getattr(self.generator_cfg, "judge_reward_base_url", "https://api.openai.com/v1").rstrip("/")

        def _judge(final_answer: str) -> float:
            import httpx as _httpx

            try:
                predicted = _ast.literal_eval(final_answer)
                if isinstance(predicted, str):
                    predicted = [predicted]
                elif isinstance(predicted, (list, tuple)):
                    predicted = [s if isinstance(s, str) else str(s) for s in predicted]
                else:
                    predicted = [str(predicted)]
            except (ValueError, SyntaxError):
                predicted = [s.strip() for s in final_answer.split("\n\n") if s.strip()]

            gt_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(gt_strings)) or "(none)"
            pred_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(predicted)) or "(none)"

            user_msg = _textwrap.dedent(f"""\
                You are evaluating predicted evidence extractions against ground truth evidence for a question.
                Treat the ground truth evidence as a perfect 10/10 reference for all dimensions.

                Question: {question}

                Ground truth evidence (reference — treat as 10/10 on all dimensions):
                {gt_block}

                Predicted evidence:
                {pred_block}

                Score the predicted evidence on two dimensions (1-10 each):

                PRECISION SCORE — are the spans tight and free of off-topic padding?
                  10 — every span contains only directly relevant sentences; no extraneous setup, headers, or filler
                   9 — essentially tight; one trivially redundant phrase but no real noise
                   8 — minor padding (1-2 extra sentences) but core content is accurate
                   7 — a few extra sentences that are related but not strictly necessary
                   6 — noticeable extraneous text in some spans, but the relevant parts are present
                   5 — roughly half the content is relevant; half is filler or tangential
                   4 — spans are significantly bloated with irrelevant surrounding text
                   3 — small fraction of each span is on-topic; most is irrelevant context
                   2 — most of each span is irrelevant; the relevant fragment is buried
                   1 — extractions are almost entirely off-topic or wrong

                RECALL SCORE — do the predicted spans collectively cover what is needed to answer the question?
                  10 — all key facts from the ground truth are present; nothing important missing
                   9 — all key facts present; only a trivially minor detail absent
                   8 — most key facts covered; one minor detail from the ground truth absent
                   7 — core answer present; a couple of supporting details from the ground truth missing
                   6 — core answer present but a meaningful portion of the ground truth evidence is missing
                   5 — about half the ground truth evidence is covered; half is missing
                   4 — partial answer only; several important aspects from the ground truth are absent
                   3 — a few relevant facts retrieved but most of the ground truth evidence is missing
                   2 — barely touches the question; most of the ground truth evidence is missing
                   1 — no relevant content retrieved

                Provide a brief reasoning string (2-4 sentences) explaining the scores.
            """)

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set when using judge_reward_model")

            result = None
            for attempt in range(5):
                if attempt > 0:
                    time.sleep(2 ** attempt)
                try:
                    with _httpx.Client(timeout=60) as client:
                        resp = client.post(
                            f"{base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": user_msg}],
                                "temperature": 0,
                                "response_format": _RUBRIC_RESPONSE_FORMAT,
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()
                except Exception as e:
                    if attempt == 4:
                        logger.warning(f"Judge reward model failed after 5 attempts: {e}")
                        return 0.0
                    logger.warning(f"Judge reward model attempt {attempt + 1} failed: {e}, retrying...")
                    continue

                try:
                    result = _json.loads(data["choices"][0]["message"]["content"])
                    break
                except _json.JSONDecodeError:
                    if attempt == 4:
                        return 0.0
                    continue

            precision = result.get("precision_score", 0) / 10.0
            recall = result.get("recall_score", 0) / 10.0
            ctx = self.active_rollouts.get(rid)
            if ctx is not None:
                ctx.judge_scores["judge_precision"] = precision
                ctx.judge_scores["judge_recall"] = recall
            return (precision + recall) / 2.0

        def judge_reward_fn(final_answer: str) -> float:
            return _judge(final_answer)

        return judge_reward_fn

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

        The child's call record (paper_id / final_answer / had_final_answer) is
        written onto the child's own context, not into a closure list. The root
        gathers all descendants' records via DFS.

        When ``child_openrouter_model`` is set, the parent's context carries a
        ``child_engine``; we propagate it down so descendants share one engine.
        """
        # Reverse lookup: paper text → paper_id, for child-call attribution.
        context_to_pid: Dict[str, str] = {}
        extra_info = env_extras.get("extra_info") or {}
        if isinstance(extra_info, dict):
            ctx_raw = extra_info.get("context_text")
            if isinstance(ctx_raw, str):
                try:
                    decoded = json.loads(ctx_raw)
                    if isinstance(decoded, dict):
                        context_to_pid = {v: k for k, v in decoded.items()}
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(ctx_raw, dict):
                context_to_pid = {v: k for k, v in ctx_raw.items()}

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
                child_index=len(parent_ctx.children_rids),
                child_engine=parent_ctx.child_engine,
                child_system_prompt=parent_ctx.child_system_prompt,
            )
            self.active_rollouts[child_rid] = child_ctx
            parent_ctx.children_rids.append(child_rid)

            # Build child env_extras: strip parent callables (child's
            # _setup_env_extras will re-inject), pre-stamp child_rid so the
            # hook reuses the registered context.
            child_extras = {k: v for k, v in env_extras.items() if k not in ("lm_callback", "subcall_fn")}
            child_extras["rlm_rollout_id"] = child_rid

            if child_ctx.child_system_prompt:
                child_extras["custom_system_prompt"] = child_ctx.child_system_prompt
            if context is not None:
                child_extra_info = dict(child_extras.get("extra_info", {}) or {})
                if isinstance(context, str):
                    child_extra_info["context_text"] = context
                else:
                    child_extra_info["context_text"] = json.dumps(context)
                child_extras["extra_info"] = child_extra_info
                # Children aren't scored independently; clear evidence-based reward_spec
                # and drop parent's custom_tools so the child's env.init() rebuilds them
                # closing over the child's own context.
                child_extras["reward_spec"] = {"ground_truth": None}
                child_extras.pop("custom_tools", None)

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
            child_final = child_env_metrics.get("final_answer")

            # Attribution record lives on the child's context; root's finalize
            # gathers them via DFS over the tree.
            child_ctx.call_record = {
                "paper_id": context_to_pid.get(context) if isinstance(context, str) else None,
                "final_answer": child_final,
                "had_final_answer": child_final is not None,
            }
            return child_final or ""

        def subcall_fn(prompt: str, context=None) -> str:
            future = asyncio.run_coroutine_threadsafe(_run_child(prompt, context=context), loop)
            return future.result(timeout=600)

        return subcall_fn
