"""
RLM-specific generator: extends SkyRLGymGenerator via four hooks plus a thin
``generate`` wrapper that resolves batch-level RLM overrides once.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
)
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


def _write_rlm_rollout(
    rollout_output_dir: str,
    prompt: ConversationType,
    env_extras: Dict[str, Any],
    env_metrics: Dict[str, Any],
    parent_step_outputs: List[TrajectoryOutput],
    child_step_outputs: List[List[TrajectoryOutput]],
    tokenizer,
    trajectory_id_str: Optional[str] = None,
) -> None:
    """Write one JSON file per trajectory (parent + children) into rollout_output_dir."""
    os.makedirs(rollout_output_dir, exist_ok=True)

    tid = trajectory_id_str or "unknown"
    tid_safe = tid.replace("/", "_").replace(":", "-")

    reward_spec = env_extras.get("reward_spec", {})

    child_refs = []
    for idx, child in enumerate(child_step_outputs):
        if not child:
            continue
        child_record = {
            "trajectory_id": f"child_{tid}_{idx}",
            "parent_trajectory_id": tid,
            "child_index": idx,
            "turns": [
                {
                    "input": tokenizer.decode(s.prompt_ids, skip_special_tokens=False),
                    "output": tokenizer.decode(s.response_ids, skip_special_tokens=False),
                }
                for s in child
            ],
            "reward": (child[-1].env_metrics or {}).get("reward") if child else None,
            "stop_reason": child[-1].stop_reason if child else None,
            "final_answer": (child[-1].env_metrics or {}).get("final_answer") if child else None,
        }
        child_filename = f"child_{tid_safe}_{idx}.json"
        with open(os.path.join(rollout_output_dir, child_filename), "w") as f:
            f.write(json.dumps(child_record, ensure_ascii=False, indent=2))
        child_refs.append(child_filename)

    parent_record = {
        "trajectory_id": tid,
        "reward": env_metrics.get("reward", 0.0),
        "turns_used": env_metrics.get("turns_used", 0),
        "stop_reason": parent_step_outputs[-1].stop_reason if parent_step_outputs else None,
        "final_answer": env_metrics.get("final_answer"),
        "evidence": reward_spec.get("evidence"),
        "turns": [
            {
                "input": tokenizer.decode(s.prompt_ids, skip_special_tokens=False),
                "output": tokenizer.decode(s.response_ids, skip_special_tokens=False),
            }
            for s in parent_step_outputs
        ],
        "child_files": child_refs,
    }
    with open(os.path.join(rollout_output_dir, f"traj_{tid_safe}.json"), "w") as f:
        f.write(json.dumps(parent_record, ensure_ascii=False, indent=2))


class RLMGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator extended for the RLM environment.

    Lives entirely in user code (``examples/train/rlm/``). Plugs into the base
    via four hooks: ``_setup_env_extras``, ``_call_inference``,
    ``_finalize_episode``, ``_flatten_step_wise_outputs`` (and its logprobs
    sister). A thin ``generate`` wrapper resolves the batch-level RLM overrides
    (``custom_system_prompt``, ``child_system_prompt``, ``rollout_output_dir``,
    ``include_children``) once and stashes them on ``self`` for the hooks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openrouter_usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "requests": 0}
        # Set per-call by ``generate``; read by hooks. Reset in finally.
        self._batch_rlm_overrides: Dict[str, Optional[str]] = {}
        self._include_children: bool = True
        # Per-call setup state stashed by ``_setup_env_extras`` so
        # ``_finalize_episode`` can read judge_scores / child_results /
        # child_call_tracker without re-passing them through the loop body.
        self._rlm_setup_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # generate(): thin wrapper that resolves batch-level overrides once
    # ------------------------------------------------------------------

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False, include_children: bool = True) -> GeneratorOutput:
        rlm_cfg = getattr(self.skyrl_gym_cfg, "rlm", None)
        overrides: Dict[str, Optional[str]] = {
            "custom_system_prompt": None,
            "child_system_prompt": None,
            "rollout_output_dir": None,
        }
        if rlm_cfg is not None:
            cfg_dict = vars(rlm_cfg) if hasattr(rlm_cfg, "__dict__") else dict(rlm_cfg)
            overrides["custom_system_prompt"] = cfg_dict.get("custom_system_prompt")
            overrides["child_system_prompt"] = cfg_dict.get("child_system_prompt")
            base_dir = cfg_dict.get("rollout_output_dir")
            if base_dir:
                batch_metadata = input_batch.get("batch_metadata")
                if batch_metadata and batch_metadata.training_phase == "eval":
                    step = batch_metadata.global_step
                    step_label = f"eval_step_{step}" if step is not None else "eval"
                    overrides["rollout_output_dir"] = os.path.join(base_dir, step_label)
                else:
                    overrides["rollout_output_dir"] = base_dir

        self._batch_rlm_overrides = overrides
        self._include_children = include_children
        try:
            return await super().generate(input_batch, disable_tqdm)
        finally:
            self._batch_rlm_overrides = {}
            self._include_children = True

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

        Stashes ``judge_scores`` / ``child_results`` / ``child_call_tracker`` on
        ``self._rlm_setup_state`` so ``_finalize_episode`` can read them.
        """
        # Reset per-call state so non-RLM rollouts don't leak prior values.
        self._rlm_setup_state = None

        if env_class != "rlm":
            return env_extras

        env_extras = dict(env_extras)
        env_extras["step_wise"] = self.generator_cfg.step_wise_trajectories

        judge_reward_model = getattr(self.generator_cfg, "judge_reward_model", None)
        enable_child_agents = getattr(self.generator_cfg, "enable_child_agents", True)
        custom_system_prompt = self._batch_rlm_overrides.get("custom_system_prompt")

        if custom_system_prompt:
            env_extras["custom_system_prompt"] = custom_system_prompt

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

        self._rlm_setup_state = {
            "child_results": child_results,
            "child_call_tracker": child_call_tracker,
            "judge_scores": judge_scores,
        }
        return env_extras

    # ------------------------------------------------------------------
    # Hook 2: inference dispatch (one generation call per turn)
    # ------------------------------------------------------------------

    async def _call_inference(
        self,
        agent_loop_state: AgentLoopState,
        chat_history: ConversationType,
        ephemeral_user_message: Optional[Dict[str, str]],
        current_sampling_params: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
        session_id: str,
        external_generate_fn: Optional[Callable] = None,
    ) -> Tuple[str, List[int], str, Optional[List[float]], Optional[Any]]:
        """Route through ``external_generate_fn`` (e.g. OpenRouter) when set."""
        if external_generate_fn is not None:
            messages = list(chat_history) + ([ephemeral_user_message] if ephemeral_user_message else [])
            output = await external_generate_fn(messages, current_sampling_params)
            output_ids = self.tokenizer.encode(output, add_special_tokens=False)
            return output, output_ids, "stop", None, None

        return await super()._call_inference(
            agent_loop_state,
            chat_history,
            ephemeral_user_message,
            current_sampling_params,
            sampling_params,
            session_id,
            external_generate_fn=external_generate_fn,
        )

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

        if hasattr(env, "set_chat_history"):
            env.set_chat_history(copy.deepcopy(agent_loop_state.chat_history))
        env_metrics = env.get_metrics()

        setup = self._rlm_setup_state or {}
        child_results: List[StepWiseOutput] = setup.get("child_results", [])
        child_call_tracker: List[Dict[str, Any]] = setup.get("child_call_tracker", [])
        judge_scores: Dict[str, Any] = setup.get("judge_scores", {})

        if isinstance(agent_loop_output, StepWiseOutput):
            agent_loop_output.child_outputs = child_results

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

        rollout_output_dir = self._batch_rlm_overrides.get("rollout_output_dir")
        if rollout_output_dir and isinstance(agent_loop_output, StepWiseOutput):
            parent_steps = agent_loop_output.step_outputs
            child_steps = [co.step_outputs for co in (agent_loop_output.child_outputs or [])]
            tid_str = trajectory_id.to_string() if trajectory_id is not None else None
            await self._run_in_executor_if_available(
                functools.partial(_write_rlm_rollout, trajectory_id_str=tid_str),
                rollout_output_dir,
                prompt,
                env_extras,
                env_metrics,
                parent_steps,
                child_steps,
                self.tokenizer,
            )

        return env_metrics

    # ------------------------------------------------------------------
    # Hook 4: step-wise output flattening (children-first)
    # ------------------------------------------------------------------

    def _flatten_step_wise_outputs(self, all_outputs, trajectory_ids, env_classes):
        train_child_trajectories = getattr(self.generator_cfg, "train_child_trajectories", False)
        include_children = self._include_children

        responses, rewards, stop_reasons, loss_masks = [], [], [], []
        prompt_token_ids, env_metrics = [], []
        is_last_step, out_trajectory_ids, out_env_classes = [], [], []
        step_metadata = []

        for i, output in enumerate(all_outputs):
            include_children_for_output = (
                train_child_trajectories
                and include_children
                and output.child_outputs
            )

            # Children first: all is_last_step=False, share parent's trajectory_id.
            if include_children_for_output:
                for child_idx, child_output in enumerate(output.child_outputs):
                    for j, step_output in enumerate(child_output.step_outputs):
                        responses.append(step_output.response_ids)
                        rewards.append(step_output.reward)
                        stop_reasons.append(step_output.stop_reason)
                        loss_masks.append(step_output.loss_mask)
                        prompt_token_ids.append(step_output.prompt_ids)
                        env_metrics.append(step_output.env_metrics)
                        is_last_step.append(False)
                        out_trajectory_ids.append(trajectory_ids[i])
                        out_env_classes.append(env_classes[i])
                        step_metadata.append({"depth": 1, "child_index": child_idx, "step_index": j})

            for j, step_output in enumerate(output.step_outputs):
                responses.append(step_output.response_ids)
                rewards.append(step_output.reward)
                stop_reasons.append(step_output.stop_reason)
                loss_masks.append(step_output.loss_mask)
                prompt_token_ids.append(step_output.prompt_ids)
                env_metrics.append(step_output.env_metrics)
                is_last_step.append(j == len(output.step_outputs) - 1)
                out_trajectory_ids.append(trajectory_ids[i])
                out_env_classes.append(env_classes[i])
                step_metadata.append({"depth": 0, "child_index": None, "step_index": j})

        return (
            responses, rewards, stop_reasons, loss_masks,
            prompt_token_ids, env_metrics,
            is_last_step, out_trajectory_ids, out_env_classes, step_metadata,
        )

    def _flatten_step_wise_logprobs(self, all_outputs):
        train_child_trajectories = getattr(self.generator_cfg, "train_child_trajectories", False)
        include_children = self._include_children

        rollout_logprobs: List[Optional[List[float]]] = []
        for output in all_outputs:
            if train_child_trajectories and include_children and output.child_outputs:
                for child in output.child_outputs:
                    rollout_logprobs += [s.rollout_logprobs for s in child.step_outputs]
            rollout_logprobs += [s.rollout_logprobs for s in output.step_outputs]
        return rollout_logprobs

    # ==================================================================
    # RLM-specific helpers (called by hooks above)
    # ==================================================================

    async def _openrouter_generate(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call an OpenAI-compatible endpoint (e.g. OpenRouter) and return the assistant message."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for child_openrouter_model support.  Install with: pip install httpx")

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set when using child_openrouter_model")

        model = getattr(self.generator_cfg, "child_openrouter_model", None)
        base_url = getattr(self.generator_cfg, "child_openrouter_base_url", "https://openrouter.ai/api/v1").rstrip("/")

        params = sampling_params or {}
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0),
            "max_tokens": params.get("max_generate_length", 1024),
            "reasoning": {"effort": "none"},
        }
        if params.get("additional_kwargs"):
            body.update(params["additional_kwargs"])

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    usage = data.get("usage", {})
                    if usage:
                        self.openrouter_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        self.openrouter_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        self.openrouter_usage["requests"] += 1
                        details = usage.get("prompt_tokens_details") or {}
                        self.openrouter_usage["cached_tokens"] += details.get("cached_tokens", 0)
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter API call failed after 3 attempts: {last_exc}") from last_exc

    def _make_openrouter_lm_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        sampling_params: Optional[Dict[str, Any]],
    ) -> Callable[[List[str]], List[str]]:
        async def _generate(prompts: List[str]) -> List[str]:
            tasks = [
                self._openrouter_generate([{"role": "user", "content": p}], sampling_params)
                for p in prompts
            ]
            return list(await asyncio.gather(*tasks))

        def callback(prompts: List[str]) -> List[str]:
            future = asyncio.run_coroutine_threadsafe(_generate(prompts), loop)
            return future.result(timeout=300)

        return callback

    def _make_lm_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        sampling_params: Optional[Dict[str, Any]],
    ) -> Callable[[List[str]], List[str]]:
        """Sync callback that dispatches batched text prompts to the inference engine.

        Safe to call from a non-async thread (e.g. inside a REPL ``exec()``).
        """
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
            output = await self.inference_engine_client.generate(engine_input)
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
        ``child_openrouter_model`` is set, both the child's main loop and its
        ``lm_callback`` route through OpenRouter, leaving the policy engine for
        the depth-0 agent.
        """
        use_openrouter = getattr(self.generator_cfg, "child_openrouter_model", None) is not None
        external_generate_fn = self._openrouter_generate if use_openrouter else None
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
            if use_openrouter:
                child_extras["lm_callback"] = self._make_openrouter_lm_callback(loop, sampling_params)
            else:
                child_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params)
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
                external_generate_fn=external_generate_fn,
            )
            if child_results is not None and isinstance(result, StepWiseOutput):
                child_results.append(result)
            child_env_metrics = result.env_metrics if isinstance(result.env_metrics, dict) else {}
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
