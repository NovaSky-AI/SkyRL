"""
RLM-specific generator: extends SkyRLGymGenerator via four hooks plus a thin
``generate`` wrapper that resolves batch-level RLM overrides once.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)

from .openrouter_engine import OpenRouterInferenceEngine
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


class RLMGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator extended for the RLM environment.

    Lives entirely in user code (``examples/train/rlm/``). Plugs into the base
    via three hooks: ``_setup_env_extras``, ``_finalize_episode``, and
    ``_call_inference_engine``. A thin ``generate`` wrapper resolves the
    batch-level RLM overrides (``child_system_prompt``) once and stashes them
    on ``self`` for the hooks. Child rollouts that should hit an external model
    (when ``generator.child_openrouter_model`` is set) get an
    ``OpenRouterInferenceEngine`` built per-rollout in ``_setup_env_extras``,
    stashed in ``env_extras["rlm_setup_state"]``, and consumed by
    ``_call_inference_engine`` for that rollout's generation calls. Children
    are inlined into ``step_outputs`` at the end of ``_finalize_episode`` so
    the base flattener in ``generate()`` picks them up without needing a
    flatten hook override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set per-call by ``generate``; read by hooks. Reset in finally.
        self._batch_rlm_overrides: Dict[str, Optional[str]] = {}

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
        """Inject lm_callback / subcall_fn / judge_reward_fn for RLM envs.

        Stashes ``judge_scores`` / ``child_results`` / ``child_call_tracker`` into
        ``env_extras["rlm_setup_state"]`` — a per-rollout handoff to
        ``_finalize_episode``. Lives on ``env_extras`` (not ``self``) so concurrent
        rollouts can't stomp on each other's state.
        """
        if env_class != "rlm":
            return env_extras

        env_extras = dict(env_extras)
        env_extras["step_wise"] = self.generator_cfg.step_wise_trajectories

        judge_reward_model = getattr(self.generator_cfg, "judge_reward_model", None)
        enable_child_agents = getattr(self.generator_cfg, "enable_child_agents", True)

        child_results: List[StepWiseOutput] = []
        child_call_tracker: List[Dict[str, Any]] = []
        judge_scores: Dict[str, Any] = {}

        loop = asyncio.get_running_loop()
        if "lm_callback" not in env_extras:
            env_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params)
            if enable_child_agents:
                env_extras["subcall_fn"] = self._make_subcall_fn(
                    loop, env_extras, sampling_params, max_tokens, max_input_length,
                    child_results=child_results,
                    child_call_tracker=child_call_tracker,
                )

        if judge_reward_model:
            env_extras["judge_reward_fn"] = self._make_judge_reward_fn(loop, env_extras, prompt, judge_scores)

        # Per-rollout handoff to _finalize_episode. Lives on env_extras (not self) so
        # concurrent rollouts each have their own copy and can't overwrite each other's.
        env_extras["rlm_setup_state"] = {
            "child_results": child_results,
            "child_call_tracker": child_call_tracker,
            "judge_scores": judge_scores,
        }
        return env_extras

    # ------------------------------------------------------------------
    # Hook 3: per-rollout inference engine dispatch
    # ------------------------------------------------------------------

    async def _call_inference_engine(
        self,
        engine_input: InferenceEngineInput,
        env_extras: Dict[str, Any],
    ) -> InferenceEngineOutput:
        """Use the per-rollout engine override stashed in ``env_extras`` if present.

        ``_run_child`` puts an ``OpenRouterInferenceEngine`` here for child rollouts
        when ``child_openrouter_model`` is configured, so children hit the external
        API while the parent stays on the policy engine.
        """
        engine = env_extras.get("inference_engine_override")
        if engine is None:
            return await super()._call_inference_engine(engine_input, env_extras)
        return await engine.generate(engine_input)

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
        # For non-RLM rollouts, fall back to the base behavior.
        if env_class != "rlm":
            return await super()._finalize_episode(
                env, env_class, agent_loop_state, agent_loop_output, env_extras, prompt, trajectory_id
            )

        env_metrics = env.get_metrics()

        setup = env_extras.get("rlm_setup_state") or {}
        child_results: List[StepWiseOutput] = setup.get("child_results", [])
        child_call_tracker: List[Dict[str, Any]] = setup.get("child_call_tracker", [])
        judge_scores: Dict[str, Any] = setup.get("judge_scores", {})

        # Track which steps came from which child (by index in child_results) so we can
        # stamp depth/child_index metadata onto each step's env_metrics below.
        child_index_by_step: Dict[int, int] = {}  # id(step_output) → child_index

        if isinstance(agent_loop_output, StepWiseOutput):
            # Inline children into step_outputs so the base flattener (in generate())
            # picks them up without needing a hook override. Children come first per
            # parent, sharing the parent's trajectory_id; the parent's last step
            # remains the final entry, so is_last_step still marks the trajectory end.
            if getattr(self.generator_cfg, "train_child_trajectories", False) and child_results:
                children_flat: List[TrajectoryOutput] = []
                for child_idx, child in enumerate(child_results):
                    for step in child.step_outputs:
                        child_index_by_step[id(step)] = child_idx
                        children_flat.append(step)
                agent_loop_output.step_outputs = children_flat + agent_loop_output.step_outputs

        if judge_scores:
            env_metrics.update(judge_scores)

        if child_call_tracker:
            from skyrl_gym.envs.rlm.evidence_tools import compute_child_rlm_metrics

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
                child_rlm_metrics = compute_child_rlm_metrics(child_call_tracker, evidence, parent_context)
                env_metrics.update(child_rlm_metrics)
                if isinstance(agent_loop_output, StepWiseOutput) and agent_loop_output.step_outputs:
                    agent_loop_output.step_outputs[-1].env_metrics.update(child_rlm_metrics)

        # Stamp per-step rlm_metadata onto each step's env_metrics so the upstream eval
        # dump (dump_per_dataset_eval_results) carries enough info to reconstruct the
        # parent/child structure of each rollout from the JSONL alone. Trajectory-level
        # fields (final_answer, turns_used) live only on the parent's last step row.
        if isinstance(agent_loop_output, StepWiseOutput) and agent_loop_output.step_outputs:
            tid_str = trajectory_id.to_string() if trajectory_id is not None else None
            num_children_steps = len(agent_loop_output.step_outputs) - sum(
                1 for s in agent_loop_output.step_outputs if id(s) not in child_index_by_step
            )
            parent_step_index = 0
            child_step_index_by_idx: Dict[int, int] = {}
            for step in agent_loop_output.step_outputs:
                child_idx = child_index_by_step.get(id(step))
                meta: Dict[str, Any] = {"trajectory_id": tid_str}
                if child_idx is not None:
                    meta["depth"] = 1
                    meta["child_index"] = child_idx
                    meta["step_index"] = child_step_index_by_idx.get(child_idx, 0)
                    child_step_index_by_idx[child_idx] = child_step_index_by_idx.get(child_idx, 0) + 1
                else:
                    meta["depth"] = 0
                    meta["child_index"] = None
                    meta["step_index"] = parent_step_index
                    parent_step_index += 1
                step.env_metrics["rlm_metadata"] = meta

            # Promote trajectory-level summary onto the parent's final step.
            agent_loop_output.step_outputs[-1].env_metrics["rlm_metadata"].update(
                {
                    "final_answer": env_metrics.get("final_answer"),
                    "turns_used": env_metrics.get("turns_used"),
                    "evidence": (env_extras.get("reward_spec") or {}).get("evidence"),
                    "num_children": len(child_results),
                }
            )

        # Drop the (potentially huge) context payload and internal-only callables from
        # env_extras now that the rollout is done. This makes env_extras safe to JSON-
        # serialize in the upstream eval dump (dump_per_dataset_eval_results).
        extra_info = env_extras.get("extra_info")
        if isinstance(extra_info, dict):
            extra_info.pop("context_text", None)
        for k in ("lm_callback", "subcall_fn", "judge_reward_fn", "rlm_setup_state", "inference_engine_override"):
            env_extras.pop(k, None)
        # reward_spec.reward_fn is a callable injected by some flows; same treatment.
        reward_spec = env_extras.get("reward_spec")
        if isinstance(reward_spec, dict):
            reward_spec.pop("reward_fn", None)

        return env_metrics

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
        judge_scores: Optional[Dict[str, Any]] = None,
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
            if judge_scores is not None:
                judge_scores["judge_precision"] = precision
                judge_scores["judge_recall"] = recall
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
        child_results: Optional[List] = None,
        child_call_tracker: Optional[List] = None,
    ) -> Callable[[str], str]:
        """Build a sync ``subcall_fn`` exposed to the parent's REPL as ``rlm_query``.

        Spawns a child agent_loop with stripped extras (no ``lm_callback``/``subcall_fn``
        from the parent) so the base hook re-injects fresh ones. When
        ``child_openrouter_model`` is set, the child's agent_loop and its
        ``lm_callback`` are both routed through an ``OpenRouterInferenceEngine``,
        leaving the policy engine for the depth-0 agent.
        """
        use_openrouter = getattr(self.generator_cfg, "child_openrouter_model", None) is not None
        # Build the per-rollout external engine for children. None means children inherit the policy engine.
        child_engine: Optional[InferenceEngineInterface] = (
            OpenRouterInferenceEngine(
                model=getattr(self.generator_cfg, "child_openrouter_model"),
                tokenizer=self.tokenizer,
                base_url=getattr(self.generator_cfg, "child_openrouter_base_url", "https://openrouter.ai/api/v1"),
            )
            if use_openrouter
            else None
        )
        child_system_prompt = self._batch_rlm_overrides.get("child_system_prompt")

        # Reverse lookup: paper text → paper_id, for child-call attribution.
        context_to_pid: Dict[str, str] = {}
        if child_call_tracker is not None:
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
            child_prompt = [{"role": "user", "content": prompt}]
            child_extras = {k: v for k, v in env_extras.items() if k not in ("lm_callback", "subcall_fn")}
            child_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params, engine=child_engine)
            # Stash the per-rollout engine override on env_extras so the child's
            # _call_inference_engine hook picks it up. None means "use the policy engine".
            if child_engine is not None:
                child_extras["inference_engine_override"] = child_engine
            else:
                child_extras.pop("inference_engine_override", None)
            if child_system_prompt:
                child_extras["custom_system_prompt"] = child_system_prompt
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
                prompt=child_prompt,
                env_class="rlm",
                env_extras=child_extras,
                max_tokens=max_tokens,
                max_input_length=max_input_length,
                sampling_params=sampling_params,
            )
            if isinstance(result, StepWiseOutput):
                if child_results is not None:
                    child_results.append(result)
                child_env_metrics = result.step_outputs[-1].env_metrics if result.step_outputs else {}
            else:
                child_env_metrics = result.env_metrics or {}
            child_final = child_env_metrics.get("final_answer")
            if child_call_tracker is not None:
                paper_id = context_to_pid.get(context) if isinstance(context, str) else None
                child_call_tracker.append({
                    "paper_id": paper_id,
                    "final_answer": child_final,
                    "had_final_answer": child_final is not None,
                })
            return child_final or ""

        def subcall_fn(prompt: str, context=None) -> str:
            future = asyncio.run_coroutine_threadsafe(_run_child(prompt, context=context), loop)
            return future.result(timeout=600)

        return subcall_fn
