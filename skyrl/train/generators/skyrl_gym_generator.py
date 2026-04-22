"""
This file implements ``SkyRLGymGenerator``, an implementation of the `GeneratorInterface` that
uses SkyRL-Gym as the environment.

For details, see https://docs.skyrl.ai/docs/tutorials/skyrl_gym_generator
"""

import asyncio
import copy
import functools
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import torch
from loguru import logger
from tqdm.asyncio import tqdm

import skyrl_gym
from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
)
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    apply_overlong_filtering,
    get_custom_chat_template,
    get_generation_prompt_ids,
    get_rollout_metrics,
)
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


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
class TrajectoryOutput:
    """Output from a single agent_loop execution."""

    response_ids: List[int]
    reward: Union[List[float], float]
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]
    env_metrics: Dict[str, Any]
    rollout_expert_indices: Optional[List[List[List[int]]]] = None
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


@dataclass
class StepWiseOutput:
    """Output from a single agent_loop execution for step-wise training."""

    step_outputs: List[TrajectoryOutput]
    child_outputs: List["StepWiseOutput"] = field(default_factory=list)

    @property
    def response_ids(self) -> List[int]:
        """Concatenated response_ids across all steps (used by _run_child to decode child output)."""
        ids: List[int] = []
        for step in self.step_outputs:
            ids.extend(step.response_ids)
        return ids

    @property
    def env_metrics(self) -> Dict[str, Any]:
        """Env metrics from the last step (used by _run_child for child chat history)."""
        if self.step_outputs:
            return self.step_outputs[-1].env_metrics
        return {}


@dataclass
class AgentLoopState:
    chat_history: ConversationType
    input_ids: List[int]
    loss_mask: List[int]
    rollout_logprobs: Optional[List[float]]
    response_end_idx: Optional[int]
    done: bool
    rollout_expert_indices: Optional[List[List[List[int]]]] = None


@dataclass
class TurnOutput:
    output: str
    output_ids: List[int]
    output_logprobs: Optional[List[float]]
    new_obs: ConversationType
    obs_ids: List[int]
    rollout_expert_indices: Optional[List[List[List[int]]]]  # [seq_len, layer_num, topk]
    reward: Optional[float]
    added_eos: bool = False

    def get_turn_rollout_expert_indices(self) -> Optional[List[List[List[int]]]]:
        """
        Get rollout inference indices for this turn's tokens (output tokens + observation tokens).

        Returns indices for generated output tokens, with padding entries (all 0)
        for any manually-added EOS token and observation tokens
        Returns None if rollout_expert_indices is None.
        """
        if self.rollout_expert_indices is None:
            return None
        if not self.rollout_expert_indices:
            return self.rollout_expert_indices
        layer_num = len(self.rollout_expert_indices[0])
        topk = len(self.rollout_expert_indices[0][0]) if layer_num > 0 else 0
        pad_entry = [[0] * topk for _ in range(layer_num)]
        indices = list(self.rollout_expert_indices)
        if self.added_eos:
            indices.append(pad_entry)
        indices.extend(pad_entry for _ in range(len(self.obs_ids)))
        return indices

    def get_turn_loss_mask(self) -> List[int]:
        """
        Get loss mask for this turn's tokens.

        Returns:
            List[int]: Loss mask where 1 indicates tokens to include in loss computation and 0 indicates
                tokens to exclude. If `added_eos` is True, the EOS token is masked out (set to 0).
                Observation tokens are always masked out (set to 0).
        """
        # if `added_eos` is `True`, then  the EOS token was not generated and only added in the
        # `agent_loop` function. For consistency with other entities like logprobs , we ignore it in the loss
        # mask
        return ([1] * len(self.output_ids) if not self.added_eos else [1] * (len(self.output_ids) - 1) + [0]) + [
            0
        ] * len(self.obs_ids)

    def get_turn_rollout_logprobs(self) -> Optional[List[float]]:
        """
        Get rollout logprobs for this turn's tokens.

        Returns:
            Optional[List[float]]: Logprobs for output tokens followed by dummy values (0.0) for
                observation tokens. Returns None if output_logprobs is None.
        """
        if not self.output_logprobs:
            return None
        return self.output_logprobs + [0.0] * len(self.obs_ids)



def _write_rlm_rollout(
    rollout_output_dir: str,
    prompt: ConversationType,
    env_extras: Dict[str, Any],
    env_metrics: Dict[str, Any],
    parent_step_outputs: List["TrajectoryOutput"],
    child_step_outputs: List[List["TrajectoryOutput"]],
    tokenizer,
    trajectory_id_str: Optional[str] = None,
) -> None:
    """Write one JSON file per trajectory (parent + children) into rollout_output_dir."""
    os.makedirs(rollout_output_dir, exist_ok=True)

    tid = trajectory_id_str or "unknown"
    # Sanitize so it's safe as a filename
    tid_safe = tid.replace("/", "_").replace(":", "-")

    reward_spec = env_extras.get("reward_spec", {})

    # Write each child as its own file: child_{parent_tid}_{idx}.json
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
        child_path = os.path.join(rollout_output_dir, child_filename)
        with open(child_path, "w") as f:
            f.write(json.dumps(child_record, ensure_ascii=False, indent=2))
        child_refs.append(child_filename)

    # Write the parent trajectory file: traj_{tid}.json
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
    parent_path = os.path.join(rollout_output_dir, f"traj_{tid_safe}.json")
    with open(parent_path, "w") as f:
        f.write(json.dumps(parent_record, ensure_ascii=False, indent=2))


class SkyRLGymGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: GeneratorConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_turns = generator_cfg.max_turns
        self.batched = generator_cfg.batched
        self.use_conversation_multi_turn = generator_cfg.use_conversation_multi_turn
        # optionally use custom chat template to get loss masks (i.e. for Qwen3)
        self.custom_chat_template = get_custom_chat_template(generator_cfg.chat_template)
        # get generation prompt ids for the tokenizer if needed
        self.generation_prompt_ids = get_generation_prompt_ids(tokenizer) if self.use_conversation_multi_turn else None
        if self.skyrl_gym_cfg.max_env_workers > 0:
            self.env_executor = ThreadPoolExecutor(
                max_workers=self.skyrl_gym_cfg.max_env_workers, thread_name_prefix="skyrl-gym-env-"
            )
        else:
            self.env_executor = None

        self.openrouter_usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "requests": 0}

        self._validate_cfg(generator_cfg)

        # base_conversation is used when `use_conversation_multi_turn==True and custom_chat_template==None` to
        # correctly format and tokenize observations into `observation_ids`.
        # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
        self.base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        self.base_conversation_token_ids = tokenizer.apply_chat_template(
            self.base_conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            **self.generator_cfg.chat_template_kwargs,
        )
        # We remove tokens after the last EOS token so that it can be captured in `observation_ids`.
        # For details, see https://docs.skyrl.ai/docs/tutorials/skyrl_gym_generator#multi-turn-tokenization-and-ti-to
        if self.tokenizer.eos_token_id in self.base_conversation_token_ids:
            last_eos_token_index = (
                len(self.base_conversation_token_ids)
                - 1
                - self.base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
            )
            self.base_conversation_token_ids = self.base_conversation_token_ids[: last_eos_token_index + 1]

    def _validate_cfg(self, generator_cfg: GeneratorConfig):
        if len(generator_cfg.chat_template_kwargs) and generator_cfg.batched:
            raise ValueError(
                "`chat_template_kwargs` is not compatible with `batched=True` since the chat templating is handled by the inference engine"
            )

        if self.generator_cfg.step_wise_trajectories:
            if self.batched:
                raise ValueError("`step_wise_trajectories` doesn't support `batched=True`")

            if self.custom_chat_template is not None:
                raise ValueError(
                    f"`step_wise_trajectories` doesn't support custom chat template, got {generator_cfg.chat_template}"
                )

            if self.generator_cfg.inference_engine.enable_return_routed_experts:
                raise ValueError("`step_wise_trajectories` doesn't support `enable_return_routed_experts=True`")

            if not self.use_conversation_multi_turn:
                raise ValueError("`step_wise_trajectories` doesn't support `use_conversation_multi_turn=False`")

    async def _run_in_executor_if_available(self, func, *args, **kwargs):
        if (executor := self.env_executor) is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # OpenRouter / OpenAI-compatible external generation
    # ------------------------------------------------------------------

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

        model = self.generator_cfg.child_openrouter_model
        base_url = self.generator_cfg.child_openrouter_base_url.rstrip("/")

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
        """Build an lm_callback that routes through OpenRouter instead of the inference engine."""

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

    # ------------------------------------------------------------------
    # Inference-engine lm_callback (default)
    # ------------------------------------------------------------------

    def _make_lm_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        sampling_params: Optional[Dict[str, Any]],
    ) -> Callable[[List[str]], List[str]]:
        """Build a sync callback that dispatches batched text prompts to the inference engine.

        The callback can be called from a non-async thread (e.g. inside REPL exec()).
        It blocks until all responses are returned.
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
        """Build a sync callable that scores the final answer using an LLM judge via OpenRouter.

        The callable replaces the F1 reward when judge_reward_model is set.
        Score = ((precision + recall) / 2) ** 2, where each dimension is normalized to [0, 1].
        """
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

        model = self.generator_cfg.judge_reward_model
        base_url = self.generator_cfg.judge_reward_base_url.rstrip("/")

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
            overall = (precision + recall) / 2.0
            return overall

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
        child_histories: Optional[List] = None,
        child_results: Optional[List] = None,
        child_call_tracker: Optional[List] = None,
    ) -> Callable[[str], str]:
        """Build a sync subcall_fn that runs a full child agent_loop (child RLMEnv).

        The child env gets the same lm_callback as the parent so it can make LLM
        calls, but does NOT get a subcall_fn — preventing unbounded recursion.

        When ``child_openrouter_model`` is configured, both the child's main
        generation loop and its ``lm_callback`` (for ``llm_query``) are routed
        through the OpenRouter API so that only the top-level (depth-0) agent
        uses the policy inference engine.

        child_histories: if provided, each child's chat_history is appended to this
        list after the child completes (used for rollout logging).
        child_call_tracker: if provided, each child call's metadata (paper_id,
        final_answer, had_final_answer) is appended for aggregate metric computation.
        """
        use_openrouter = self.generator_cfg.child_openrouter_model is not None
        external_generate_fn = self._openrouter_generate if use_openrouter else None

        # Build reverse lookup from paper text → paper_id for child call tracking.
        _context_to_pid: Dict[str, str] = {}
        if child_call_tracker is not None:
            _extra_info = env_extras.get("extra_info") or {}
            if isinstance(_extra_info, dict):
                _ctx_raw = _extra_info.get("context_text")
                if isinstance(_ctx_raw, str):
                    try:
                        _decoded = json.loads(_ctx_raw)
                        if isinstance(_decoded, dict):
                            _context_to_pid = {v: k for k, v in _decoded.items()}
                    except (json.JSONDecodeError, ValueError):
                        pass
                elif isinstance(_ctx_raw, dict):
                    _context_to_pid = {v: k for k, v in _ctx_raw.items()}

        async def _run_child(prompt: str, context=None) -> str:
            child_prompt = [{"role": "user", "content": prompt}]
            # Strip lm_callback/subcall_fn so the child cannot recurse further
            child_extras = {k: v for k, v in env_extras.items() if k not in ("lm_callback", "subcall_fn")}
            # Re-inject lm_callback — via OpenRouter when configured, else via the inference engine
            if use_openrouter:
                child_extras["lm_callback"] = self._make_openrouter_lm_callback(loop, sampling_params)
            else:
                child_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params)
            # If a child-specific system prompt is configured, use it as the child's custom_system_prompt
            child_system_prompt = child_extras.pop("child_system_prompt", None)
            if child_system_prompt:
                child_extras["custom_system_prompt"] = child_system_prompt
            # Children don't write their own rollout files; parent handles everything
            child_extras.pop("rollout_output_dir", None)
            # If the parent passed a per-child context, override the child's context_text
            # so the child REPL sees this value as its `context` variable instead of
            # inheriting the parent's full context.
            if context is not None:
                child_extra_info = dict(child_extras.get("extra_info", {}) or {})
                if isinstance(context, str):
                    child_extra_info["context_text"] = context
                else:
                    import json as _json
                    child_extra_info["context_text"] = _json.dumps(context)
                child_extras["extra_info"] = child_extra_info
                # Clear parent's evidence-based reward_spec for children — the child
                # is not scored independently; the parent aggregates results.
                child_extras["reward_spec"] = {"ground_truth": None}
                # Drop parent's custom_tools so env.init() builds fresh ones
                # that close over the child's own context instead.
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
            if child_histories is not None:
                child_chat_history = result.env_metrics.get("chat_history") if isinstance(result.env_metrics, dict) else None
                child_histories.append(child_chat_history)
            if child_results is not None and isinstance(result, StepWiseOutput):
                child_results.append(result)
            child_env_metrics = result.env_metrics if isinstance(result.env_metrics, dict) else {}
            child_final = child_env_metrics.get("final_answer")
            if child_call_tracker is not None:
                paper_id = _context_to_pid.get(context) if isinstance(context, str) else None
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

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
        external_generate_fn: Optional[Callable] = None,
    ) -> Union[TrajectoryOutput, StepWiseOutput]:
        """
        Multi-turn generation loop that executes a single trajectory.

        Note:
            We ensure token-in-token-out generation. With two exceptions:
            - When calling Env.step() and BaseTextEnvStepOutput["postprocessed_action"] is not None.
              This will likely be deprecated soon.
            - When custom_chat_template = True and use_conversation_multi_turn = True. We always
              re-tokenize the entire chat history every turn and at the end. This is used for cases
              like removing Qwen3 thinking tokens in non-last-round assistant message.

        Args:
            prompt: ConversationType
            env_extras: Dict[str, Any]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
            external_generate_fn: Optional async callable ``(messages, sampling_params) -> str``.
                When provided, each generation step calls this function (e.g. OpenRouter)
                instead of the local inference engine.  The returned text is tokenized with
                the policy tokenizer so that loss-mask / step-wise bookkeeping still works.
        Returns:
            response_ids: List[int]
            reward: Union[float, List[float]]
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
            rollout_logprobs: Optional[List[float]]
        """
        # NOTE: `custom_chat_template` was mainly for getting accurate loss masks for thinking models.
        # This is no longer needed now given that step wise training is supported
        # TODO (sumanthrh): This path can be deprecated
        retokenize_chat_history = self.use_conversation_multi_turn and self.custom_chat_template

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_extras["step_wise"] = self.generator_cfg.step_wise_trajectories

        # For RLM envs, inject lm_callback, subcall_fn, and optionally judge_reward_fn.
        # We only inject lm_callback/subcall_fn if not already present (allows caller to override).
        # subcall_fn is skipped when enable_child_agents=False (single-paper mode): the top-level
        # agent never calls rlm_query_batched, and omitting subcall_fn keeps the shorter repl_timeout.
        # judge_reward_fn replaces the F1 reward when judge_reward_model is set.
        child_histories: Optional[List] = None
        child_results: List[StepWiseOutput] = []
        child_call_tracker: List[Dict[str, Any]] = []
        judge_scores: Dict[str, Any] = {}
        _needs_rlm_setup = env_class == "rlm" and (
            "lm_callback" not in env_extras or self.generator_cfg.judge_reward_model
        )
        if _needs_rlm_setup:
            loop = asyncio.get_running_loop()
            env_extras = dict(env_extras)  # don't mutate caller's dict
            if "lm_callback" not in env_extras:
                env_extras["lm_callback"] = self._make_lm_callback(loop, sampling_params)
                if self.generator_cfg.enable_child_agents:
                    child_histories = []
                    child_results = []
                    env_extras["subcall_fn"] = self._make_subcall_fn(
                        loop, env_extras, sampling_params, max_tokens, max_input_length,
                        child_histories=child_histories,
                        child_results=child_results,
                        child_call_tracker=child_call_tracker,
                    )
            if self.generator_cfg.judge_reward_model:
                judge_scores = {}
                env_extras["judge_reward_fn"] = self._make_judge_reward_fn(loop, env_extras, prompt, judge_scores)

        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )

        # Instantiate chat_history and chat_end_index, which are only used if `retokenize_chat_history==True`.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, init_info = await self._run_in_executor_if_available(env.init, chat_history)
        # next_user_message is the ephemeral per-turn user prompt: appended before inference but
        # never stored in chat_history so it doesn't pollute the training context.
        next_user_message = init_info.get("next_user_message") if init_info else None
        initial_chat_history_length = len(chat_history)
        _init_chat_for_inference = chat_history + ([next_user_message] if next_user_message else [])
        initial_input_ids = self.tokenizer.apply_chat_template(
            _init_chat_for_inference,
            # If retokenize_chat_history==True, avoid including the generation prompt in both the
            # prompt_ids and response_ids due to how `response_encodings["input_ids"]` works.
            add_generation_prompt=not retokenize_chat_history,
            chat_template=self.custom_chat_template if retokenize_chat_history else None,
            tokenize=True,
            return_dict=False,
            **self.generator_cfg.chat_template_kwargs,
        )

        initial_prompt_length = len(initial_input_ids)
        loss_mask = []  # this excludes the prompt
        rollout_logprobs = None

        # `sampling_params` if provided is a dict in the format expected by the inference engine backend
        # we cast default config to a dict for consistency
        current_sampling_params: dict = (
            sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
        )

        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, Optional[int]]] = []

        is_step_wise = self.generator_cfg.step_wise_trajectories

        agent_loop_output = StepWiseOutput(step_outputs=[]) if is_step_wise else None

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None
        agent_loop_state = AgentLoopState(
            chat_history=chat_history,
            input_ids=initial_input_ids,
            loss_mask=[],
            rollout_logprobs=[] if get_logprobs else None,
            response_end_idx=None,
            done=False,
        )

        _turn_num = 0
        while not agent_loop_state.done:
            _turn_num += 1

            if len(agent_loop_state.input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 1. Generate output
            _t_tokenize = time.perf_counter()
            if is_step_wise or retokenize_chat_history:
                # re-apply whole chat template so length check is correct.
                # Append next_user_message ephemerally — it is used for inference but not stored in chat_history.
                _chat_for_inference = chat_history + ([next_user_message] if next_user_message else [])
                agent_loop_state.input_ids = self.tokenizer.apply_chat_template(
                    _chat_for_inference,
                    chat_template=self.custom_chat_template if retokenize_chat_history else None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.generator_cfg.chat_template_kwargs,
                )
                agent_loop_state.loss_mask = []
                agent_loop_state.rollout_logprobs = None
                if len(agent_loop_state.input_ids) > max_input_length:
                    stop_reason = "length"
                    break
            _tokenize_s = time.perf_counter() - _t_tokenize
            _prompt_tokens = len(agent_loop_state.input_ids)

            _t_infer = time.perf_counter()
            if external_generate_fn is not None:
                # Route through an external API (e.g. OpenRouter) instead of the
                # policy inference engine.  We send the full message list and
                # tokenize the returned text so step-wise bookkeeping still works.
                _ext_msgs = list(chat_history) + ([next_user_message] if next_user_message else [])
                output = await external_generate_fn(_ext_msgs, current_sampling_params)
                output_ids = self.tokenizer.encode(output, add_special_tokens=False)
                stop_reason = "stop"
                response_logprobs = None
                rollout_expert_indices = None
            else:
                engine_input = InferenceEngineInput(
                    prompt_token_ids=[agent_loop_state.input_ids], session_ids=[session_id], sampling_params=sampling_params
                )
                engine_output = await self.inference_engine_client.generate(engine_input)
                output = engine_output["responses"][0]
                output_ids = engine_output["response_ids"][0]
                stop_reason = engine_output["stop_reasons"][0]
                response_logprobs = engine_output.get("response_logprobs", None)
                rollout_expert_indices = engine_output.get("rollout_expert_indices", None)
                if response_logprobs is not None:
                    response_logprobs = response_logprobs[0]
                    if self.custom_chat_template is not None:
                        raise ValueError("Response Logprobs bookkeeping is not supported with custom chat template")

            _inference_s = time.perf_counter() - _t_infer

            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices[0]
                if self.custom_chat_template is not None:
                    raise ValueError("Rollout expert indices bookkeeping is not supported with custom chat template")
            # Append eos when sampling_params.stop is not None. Does not affect 3.a as chat templates add eos_token.
            # sampling_params is not None for eval, but None for training (which uses engine.sampling_params which are from cfg)
            stop_strs = current_sampling_params.get("stop", None)
            added_eos = False
            if (
                stop_strs is not None
                and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn
                and self.use_conversation_multi_turn
            ):
                if output.endswith(tuple(stop_strs)) and output_ids[-1] != self.tokenizer.eos_token_id:
                    output_ids.append(self.tokenizer.eos_token_id)
                    # dummy logprobs for EOS token id. It will be loss masked with 0 in TurnOutput.get_turn_loss_mask
                    if response_logprobs is not None:
                        response_logprobs.append(0.0)
                    added_eos = True

            # 2. Environment step
            _t_env = time.perf_counter()
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            _env_step_s = time.perf_counter() - _t_env
            new_obs = env_step_output["observations"]
            next_user_message = env_step_output.get("next_user_message")
            step_reward: float = env_step_output["reward"]
            agent_loop_state.done = env_step_output["done"]


            if env_step_output.get("postprocessed_action", None) is not None:
                # TODO(Charlie): come back to this, we should deprecate postprocessed action
                logger.warning(
                    "WARNING: postprocessed action may violate token-in-token-out. Ideally you "
                    "post-process it in the token space rather than string space. "
                    "A better solution coming soon."
                )
                output = env_step_output["postprocessed_action"]
                output_ids = self.tokenizer.encode(output, add_special_tokens=False)

            obs_ids = self.get_obs_ids_from_obs(new_obs, agent_loop_state.done)

            # final turn output containing generated response and environment observations
            turn_output = TurnOutput(
                output=output,
                output_ids=output_ids,
                output_logprobs=response_logprobs,
                new_obs=new_obs,
                reward=step_reward,
                obs_ids=obs_ids,
                added_eos=added_eos,
                rollout_expert_indices=rollout_expert_indices,
            )

            if turn_output.rollout_expert_indices is not None and agent_loop_state.rollout_expert_indices is None:
                agent_loop_state.rollout_expert_indices = []

            if is_step_wise:
                # current response + observation ids
                turn_response_ids = turn_output.output_ids + turn_output.obs_ids
                turn_prompt_ids = agent_loop_state.input_ids

                # agent loop only tracks loss mask and rollout logprobs for this turn with step_wise training
                turn_loss_mask = turn_output.get_turn_loss_mask()
                turn_response_logprobs: Optional[List[float]] = turn_output.get_turn_rollout_logprobs()

                _step_latency = {
                    "prompt_tokens": _prompt_tokens,
                    "output_tokens": len(output_ids),
                    "tokenize_s": _tokenize_s,
                    "inference_s": _inference_s,
                    "env_step_s": _env_step_s,
                    "repl_exec_s": env_step_output.get("metadata", {}).get("repl_exec_s"),
                }
                per_step_output = TrajectoryOutput(
                    response_ids=turn_response_ids,
                    reward=step_reward,
                    loss_mask=turn_loss_mask,
                    prompt_ids=turn_prompt_ids,
                    rollout_logprobs=turn_response_logprobs,
                    stop_reason=stop_reason,
                    env_metrics={**(env.get_metrics() if agent_loop_state.done else {}), "latency": _step_latency},
                    rollout_expert_indices=turn_output.get_turn_rollout_expert_indices(),
                )
                agent_loop_output.step_outputs.append(per_step_output)

            # 3. Update states: input ids, loss_mask, chat_history, etc.
            # Three ways of managing input
            if retokenize_chat_history:
                # a. custom chat template
                agent_loop_state = self._update_agent_state_by_retokenizing_chat_history(agent_loop_state, turn_output)
            elif self.use_conversation_multi_turn:
                # b. Token-in-token-out. Follow multi-turn chat history format.
                agent_loop_state = self._update_agent_loop_state_with_multiturn_chat_template(
                    agent_loop_state, turn_output
                )
            else:
                # c. Token-in-token-out. All steps/observations are appended to a single assistant message.
                agent_loop_state = self._update_agent_loop_state_with_singleturn_chat_template(
                    agent_loop_state, turn_output
                )

            per_step_rewards.append((step_reward, agent_loop_state.response_end_idx))

        # Get environment-specific metrics after the episode is done.
        # For RLM envs, store the full chat_history on the env before calling get_metrics()
        # so it's available in env_metrics for rollout logging.
        if env_class == "rlm" and hasattr(env, "set_chat_history"):
            env.set_chat_history(copy.deepcopy(agent_loop_state.chat_history))
        env_metrics = env.get_metrics()

        # Attach child results to the output before writing rollout files.
        if env_class == "rlm" and isinstance(agent_loop_output, StepWiseOutput):
            agent_loop_output.child_outputs = child_results

        # Merge judge sub-scores (precision, recall) into env_metrics for WandB logging.
        if judge_scores:
            env_metrics.update(judge_scores)

        # Compute child RLM aggregate metrics (paper selection F1, submission rate, etc.)
        if env_class == "rlm" and child_call_tracker:
            from skyrl_gym.envs.rlm.evidence_tools import compute_child_rlm_metrics

            reward_spec = env_extras.get("reward_spec") or {}
            evidence = reward_spec.get("evidence") or []
            _extra_info = env_extras.get("extra_info") or {}
            _ctx_raw = _extra_info.get("context_text") if isinstance(_extra_info, dict) else None
            parent_context: Dict[str, str] = {}
            if isinstance(_ctx_raw, str):
                try:
                    _decoded = json.loads(_ctx_raw)
                    if isinstance(_decoded, dict):
                        parent_context = _decoded
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(_ctx_raw, dict):
                parent_context = _ctx_raw

            if isinstance(evidence, list) and evidence and isinstance(evidence[0], dict) and parent_context:
                child_rlm_metrics = compute_child_rlm_metrics(child_call_tracker, evidence, parent_context)
                env_metrics.update(child_rlm_metrics)
                if isinstance(agent_loop_output, StepWiseOutput) and agent_loop_output.step_outputs:
                    agent_loop_output.step_outputs[-1].env_metrics.update(child_rlm_metrics)

        # Write rollout JSONL for RLM envs when rollout_output_dir is configured.
        rollout_output_dir = env_extras.get("rollout_output_dir")
        if env_class == "rlm" and rollout_output_dir and isinstance(agent_loop_output, StepWiseOutput):
            parent_steps = agent_loop_output.step_outputs
            child_steps = [co.step_outputs for co in (agent_loop_output.child_outputs or [])]
            _tid_str = trajectory_id.to_string() if trajectory_id is not None else None
            await self._run_in_executor_if_available(
                functools.partial(_write_rlm_rollout, trajectory_id_str=_tid_str),
                rollout_output_dir,
                prompt,
                env_extras,
                env_metrics,
                parent_steps,
                child_steps,
                self.tokenizer,
            )

        # Close the environment
        await self._run_in_executor_if_available(env.close)

        prompt_ids = agent_loop_state.input_ids[:initial_prompt_length]
        rollout_logprobs = None
        rollout_expert_indices_out = None
        response_ids = None

        # Prepare the final loss_mask, response_ids and rollout_logprobs .
        # We remove the final observation messages /token IDs here
        # Note that during the agent loop, we still add the final observation messages/ tokens because we terminate the agent loop if the input length
        # exceeds the maximum
        if retokenize_chat_history:
            response_encodings = self.tokenizer.apply_chat_template(
                agent_loop_state.chat_history[
                    initial_chat_history_length : len(agent_loop_state.chat_history) - len(new_obs)
                ],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
                **self.generator_cfg.chat_template_kwargs,
            )
            loss_mask = response_encodings["assistant_masks"]
            response_ids = response_encodings["input_ids"]
        elif not self.generator_cfg.step_wise_trajectories:
            assert not any(
                agent_loop_state.loss_mask[agent_loop_state.response_end_idx - initial_prompt_length + 1 :]
            ), "loss_mask at index after response end should be all 0"
            loss_mask = agent_loop_state.loss_mask[: agent_loop_state.response_end_idx - initial_prompt_length + 1]
            response_ids = agent_loop_state.input_ids[initial_prompt_length : agent_loop_state.response_end_idx + 1]
            if agent_loop_state.rollout_logprobs is not None:
                rollout_logprobs = agent_loop_state.rollout_logprobs[
                    : agent_loop_state.response_end_idx - initial_prompt_length + 1
                ]
            if agent_loop_state.rollout_expert_indices is not None:
                rollout_expert_indices_out = agent_loop_state.rollout_expert_indices[
                    : agent_loop_state.response_end_idx + 1
                ]
            # fix index for per_step_rewards
            per_step_rewards = [(reward, idx - initial_prompt_length) for reward, idx in per_step_rewards]
            assert len(loss_mask) == len(
                response_ids
            ), f"loss_mask and response_ids should have the same length, got {len(loss_mask)} and {len(response_ids)}"

        appended_eos_token = False
        if not self.use_conversation_multi_turn:
            assert response_ids is not None and loss_mask is not None
            if stop_reason != "length" and response_ids and response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
                # TODO(Charlie): this should be 0? Otherwise logprobs will be extremely off. But if it is loss
                # masked with 0, why bother adding it?
                loss_mask.append(1)
                if rollout_logprobs is not None:
                    rollout_logprobs.append(0.0)
                if rollout_expert_indices_out is not None and rollout_expert_indices_out:
                    layer_num = len(rollout_expert_indices_out[0])
                    topk = len(rollout_expert_indices_out[0][0]) if layer_num > 0 else 0
                    rollout_expert_indices_out.append([[0] * topk for _ in range(layer_num)])
                appended_eos_token = True

        if self.generator_cfg.step_wise_trajectories:
            for per_step_output, (reward, resp_end_idx) in zip(agent_loop_output.step_outputs, per_step_rewards):
                per_token_reward = [0.0] * len(per_step_output.response_ids)
                per_token_reward[resp_end_idx] = float(reward)
                # in-place update to per-token reward
                per_step_output.reward = per_token_reward
        else:
            reward_out = self._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token)

            agent_loop_output = TrajectoryOutput(
                response_ids=response_ids,
                reward=reward_out,
                stop_reason=stop_reason,
                loss_mask=loss_mask,
                prompt_ids=prompt_ids,
                rollout_logprobs=rollout_logprobs,
                env_metrics=env_metrics,
                rollout_expert_indices=rollout_expert_indices_out,
            )

        return agent_loop_output

    def _build_per_token_rewards(
        self, per_step_rewards: List[Tuple[float, Optional[int]]], response_ids: List[int], appended_eos_token: bool
    ) -> Union[float, List[float]]:
        """
        Build reward output from per-step rewards.

        Args:
            per_step_rewards: List of (reward, response_end_token_idx) tuples for each step
            response_ids: List of response token IDs
            appended_eos_token: Whether an EOS token was manually appended at the end

        Returns:
            Union[float, List[float]]: If custom_chat_template is used, returns the last step's reward (float).
                Otherwise, returns a list of token-level rewards (List[float]).
        """
        if self.custom_chat_template:
            # TODO(Charlie): Currently, the possible response truncation will not affect the reward
            # in the if branch, but some final rewards may be lost in the else branch. Fix this
            # when we support turn-level rewards for the `retokenize_chat_history` codepath.
            reward_out = per_step_rewards[-1][0]
        else:
            # Build token-level rewards placed at assistant turn boundaries
            token_level_rewards: List[float] = [0.0] * len(response_ids)
            for i, (step_reward, idx) in enumerate(per_step_rewards):
                assert step_reward is not None
                if idx >= len(response_ids):
                    break
                if appended_eos_token and i == len(per_step_rewards) - 1:
                    # NOTE(Charlie): If we appended the eos token, we need to place
                    # the reward at the last token (the manually appended eos token)
                    # rather than the last turn's assistant-generated token. This matches
                    # the logic in trainer.py::postprocess_generator_output when rewards are List[float].
                    token_level_rewards[-1] = step_reward
                else:
                    token_level_rewards[idx] += step_reward
            reward_out = token_level_rewards
        return reward_out

    def get_obs_ids_from_obs(self, new_obs: ConversationType, is_done: bool) -> List[int]:
        """
        Returns observation token ids from observation messages for a turn.

        Args:
            new_obs: Observation messages from the environment step
            is_done: Whether the agent loop has terminated

        Returns:
            List[int]: Observation token IDs. For multi-turn mode, includes chat template formatting.
                For single-turn mode, returns directly encoded observation tokens.
        """
        if self.use_conversation_multi_turn:
            # 2. apply chat template for observations, also generate generation prompt for next turn
            obs_ids_to_add = []
            if len(new_obs) > 0:
                # For Qwen, this will generate `\n<|user|>Some observation<|im_end|>\n`. Note that the
                # first `\n` is generated since we stripped it in ``base_conversation_token_ids``.
                obs_ids_to_add = self.tokenizer.apply_chat_template(
                    [*self.base_conversation, *new_obs],
                    add_generation_prompt=not is_done,
                    tokenize=True,
                    return_dict=False,
                    **self.generator_cfg.chat_template_kwargs,
                )[len(self.base_conversation_token_ids) :]
            elif not is_done:
                obs_ids_to_add = self.generation_prompt_ids
        else:
            # Build observation token ids (encoded directly, not using chat template)
            # no generation prompt is added in this case
            obs_ids_to_add = []
            if len(new_obs) > 0:
                for obs in new_obs:
                    obs_tokens = self.tokenizer.encode(obs["content"], add_special_tokens=False)
                    obs_ids_to_add.extend(obs_tokens)
        return obs_ids_to_add

    def _update_chat_history(
        self,
        chat_history: ConversationType,
        output: str,
        new_obs: ConversationType,
    ) -> ConversationType:
        """
        Update chat history with assistant response and new observations.

        Args:
            chat_history: Current conversation history
            output: Assistant's response text
            new_obs: New observation messages from the environment

        Returns:
            ConversationType: Updated chat history with assistant response and observations appended.
                The EOS token is removed from output if present, as it will be reapplied by the chat template.
        """
        # remove eos token from end of output if it exists, since it will be reapplied by the chat template
        if output.endswith(self.tokenizer.eos_token):
            output = output[: -len(self.tokenizer.eos_token)]

        # Add assistant response to chat history
        chat_history += [{"role": "assistant", "content": output}]

        # Add observations to chat history
        if len(new_obs) > 0:
            chat_history += new_obs
        return chat_history

    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        envs = []
        init_prompts = []
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = await self._run_in_executor_if_available(env.init, prompt)
            init_prompts.append(init_prompt)
            envs.append(env)

        # for consistency, use token-in-token-out
        prompt_token_ids = self.tokenizer.apply_chat_template(
            init_prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        engine_input = InferenceEngineInput(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        engine_output = await self.inference_engine_client.generate(engine_input)
        outputs = engine_output["responses"]
        responses = engine_output["response_ids"]
        stop_reasons = engine_output["stop_reasons"]
        logprobs = engine_output.get("response_logprobs", None)
        raw_rollout_expert_indices = engine_output.get("rollout_expert_indices", None)

        truncated_responses = []
        rewards = []
        loss_masks = []
        env_metrics = []
        truncated_logprobs: Optional[List[List[float]]] = [] if logprobs is not None else None
        truncated_indices: Optional[List] = [] if raw_rollout_expert_indices is not None else None

        for i, (output, response, env, env_class) in enumerate(zip(outputs, responses, envs, env_classes)):
            # step on environment and compute reward
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            reward = env_step_output["reward"]
            rewards.append(reward)

            if len(response) > max_tokens:
                response = response[:max_tokens]
            loss_masks.append([1] * len(response))
            truncated_responses.append(response)
            if logprobs is not None:
                sample_logprobs = logprobs[i][: len(response)]
                truncated_logprobs.append(sample_logprobs)
            if raw_rollout_expert_indices is not None:
                sample_indices = raw_rollout_expert_indices[i]
                prompt_len = len(prompt_token_ids[i])
                truncated_indices.append(sample_indices[: prompt_len + len(response)])

            # Get environment-specific metrics
            env_metrics.append(env.get_metrics())
            # Close the environment
            await self._run_in_executor_if_available(env.close)

        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes, loss_masks)

        if self.generator_cfg.apply_overlong_filtering:
            # set loss mask to 0 if the stop reason is not "stop"
            loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": truncated_responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": truncated_logprobs,
            "rollout_expert_indices": truncated_indices,
        }

        return generator_output

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False, include_children: bool = True) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
            disable_tqdm: bool
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        if self.generator_cfg.step_wise_trajectories:
            assert trajectory_ids is not None, "`trajectory_ids` is a required field for step wise training"
        sampling_params: Optional[dict] = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        if self.batched:
            return await self.generate_batched(prompts, env_classes, env_extras, max_tokens, sampling_params)

        for i in range(len(prompts)):
            if env_classes[i] == "rlm":
                rlm_config = getattr(self.skyrl_gym_cfg, "rlm", None)
                if rlm_config is not None:
                    cfg = vars(rlm_config) if hasattr(rlm_config, "__dict__") else rlm_config
                    custom_prompt = cfg.get("custom_system_prompt")
                    if custom_prompt:
                        env_extras[i]["custom_system_prompt"] = custom_prompt
                    child_prompt = cfg.get("child_system_prompt")
                    if child_prompt:
                        env_extras[i]["child_system_prompt"] = child_prompt
                    rollout_output_dir = cfg.get("rollout_output_dir")
                    if rollout_output_dir:
                        batch_metadata = input_batch.get("batch_metadata")
                        if batch_metadata and batch_metadata.training_phase == "eval":
                            step = batch_metadata.global_step
                            step_label = f"eval_step_{step}" if step is not None else "eval"
                            env_extras[i]["rollout_output_dir"] = os.path.join(rollout_output_dir, step_label)

        # Async agent loop to generate trajectories in parallel.
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i] if trajectory_ids is not None else None,
                )
            )

        all_outputs = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
            disable=disable_tqdm,
        )

        if self.generator_cfg.step_wise_trajectories:
            responses = []
            rewards = []
            stop_reasons = []
            loss_masks = []
            prompt_token_ids = []
            env_metrics = []
            is_last_step = []
            out_trajectory_ids = []
            out_env_classes = []
            step_metadata = []
            for i, output in enumerate(all_outputs):
                include_children_for_output = (
                    self.generator_cfg.train_child_trajectories
                    and include_children
                    and output.child_outputs
                )

                # Append child steps first (all is_last_step=False, share
                # parent's trajectory_id).  They inherit the parent's GRPO
                # advantage via the cumsum broadcast.
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

                # Always append parent steps.
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
            env_classes = out_env_classes
        else:
            responses = [output.response_ids for output in all_outputs]
            rewards = [output.reward for output in all_outputs]
            stop_reasons = [output.stop_reason for output in all_outputs]
            loss_masks = [output.loss_mask for output in all_outputs]
            prompt_token_ids = [output.prompt_ids for output in all_outputs]
            env_metrics = [output.env_metrics for output in all_outputs]
            is_last_step = None
            out_trajectory_ids = None
            step_metadata = None

        has_vision_features = any(getattr(output, "pixel_values", None) is not None for output in all_outputs)
        pixel_values = (
            [getattr(output, "pixel_values", None) for output in all_outputs] if has_vision_features else None
        )
        image_grid_thw = (
            [getattr(output, "image_grid_thw", None) for output in all_outputs] if has_vision_features else None
        )

        if sampling_params is not None:
            # sampling params will be a dict in the format of the inference engine backend
            get_logprobs = sampling_params.get("logprobs", None) is not None
        else:
            get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        if get_logprobs:
            if self.generator_cfg.step_wise_trajectories:
                rollout_logprobs = []
                for output in all_outputs:
                    # Match flattening order: children first, then all parent steps
                    if self.generator_cfg.train_child_trajectories and include_children and output.child_outputs:
                        for child in output.child_outputs:
                            rollout_logprobs += [s.rollout_logprobs for s in child.step_outputs]
                    rollout_logprobs += [s.rollout_logprobs for s in output.step_outputs]
            else:
                rollout_logprobs = [output.rollout_logprobs for output in all_outputs]
        else:
            rollout_logprobs = None

        if self.generator_cfg.inference_engine.enable_return_routed_experts:
            rollout_expert_indices = [output.rollout_expert_indices for output in all_outputs]
        else:
            rollout_expert_indices = None

        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes, loss_masks)

        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            # set loss mask to 0 if the stop reason is not "stop"
            loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": out_trajectory_ids,
            "rollout_expert_indices": rollout_expert_indices,
            "is_last_step": is_last_step,
            "env_metrics": env_metrics,
            "step_metadata": step_metadata,
        }
        if has_vision_features:
            generator_output["pixel_values"] = pixel_values
            generator_output["image_grid_thw"] = image_grid_thw

        return generator_output

    def _zero_reward_if_not_stop(
        self, rewards: List[Union[float, List[float]]], stop_reasons: List[str]
    ) -> List[Union[float, List[float]]]:
        """
        Sets the reward to 0 if the stop reason is not "stop".

        This can be useful in cases where the LLM generation was truncated or aborted, but the environment still assigns non-zero reward.
        Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response,
        we typically don't want to reward it. This is a general setting for all environments.

        Args:
            rewards: List of rewards (can be float or List[float] for per-token rewards)
            stop_reasons: List of stop reasons for each trajectory

        Returns:
            List[Union[float, List[float]]]: Modified rewards with non-"stop" cases set to 0
        """
        for i, stop_reason in enumerate(stop_reasons):
            if stop_reason != "stop":
                if isinstance(rewards[i], list):
                    rewards[i] = [0.0] * len(rewards[i])
                else:
                    rewards[i] = 0.0
        return rewards

    # ----------------------------------------------------------------------------
    # Three methods of managing chat history and input ids in `agent_loop()`
    # ----------------------------------------------------------------------------
    def _update_agent_state_by_retokenizing_chat_history(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the chat history and input ids given a new model response and observation by retokenizing
        the entire chat history. Hence token-in-token-out is not followed.

        This method is used when `use_conversation_multi_turn=True` and `custom_chat_template` is set.
        It re-tokenizes the entire chat history every turn, which is useful for cases like removing
        Qwen3 thinking tokens in non-last-round assistant messages.

        Args:
            agent_loop_state: Current agent loop state containing chat history and input IDs
            turn_output: Turn output containing the model's response and new observations

        Returns:
            AgentLoopState: Updated agent loop state with retokenized chat history and input IDs.
                Note: loss_mask, response_end_idx, and rollout_logprobs are set to None as they
                are computed at the end with the custom chat template.
        """
        assert self.use_conversation_multi_turn and self.custom_chat_template

        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        # `loss_mask` is computed at the end with `custom_chat_template`
        agent_loop_state.loss_mask = None
        # untracked state
        agent_loop_state.response_end_idx = None
        # `logprobs` are not computed because retokenizing breaks token-in-token-out
        agent_loop_state.rollout_logprobs = None
        # indices are not meaningful when retokenizing
        agent_loop_state.rollout_expert_indices = None
        return agent_loop_state

    def _update_agent_loop_state_with_multiturn_chat_template(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is True. It assumes that the input to the LLM is formatted as a list of messages, with observations
        stored in user messages.

        For example (using the Qwen 2.5 chat template), a trajectory for multi-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...
        <|im_end|>
        <|im_start|>user
                            turn 1 env observation goes here
                            <observation>...</observation>
        <|im_end|>
        ...

        The chat template is applied without tokenization before and after the chat history is appended to
        in order to get new token ids in the chat template format (but without re-tokenizing the entire chat history every turn).

        Args:
            agent_loop_state: Current agent loop state containing chat history, input IDs, loss mask, etc.
            turn_output: Turn output containing the model's response, output IDs, logprobs, and observations

        Returns:
            AgentLoopState: Updated agent loop state with appended turn IDs, loss mask, and logprobs.
                For step-wise training, only response_end_idx is updated; loss_mask and rollout_logprobs
                are set to None as they are tracked per-step.
        """
        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        loss_mask_for_turn = turn_output.get_turn_loss_mask()
        rollout_logprobs_for_turn = turn_output.get_turn_rollout_logprobs()

        # use the raw rollout expert indices without any appending of observation tokens
        # this will be overwritten each turn, so we don't need to append observation tokens to it
        rollout_expert_indices_for_turn = turn_output.rollout_expert_indices

        if self.generator_cfg.step_wise_trajectories:
            # cumulative input_ids is not tracked for step wise training
            agent_loop_state.response_end_idx = len(turn_output.output_ids) - 1
            # no running loss_mask, `rollout_logprobs`, or `rollout_expert_indices` are tracked for step-wise training
            agent_loop_state.loss_mask = None
            agent_loop_state.rollout_logprobs = None
            agent_loop_state.rollout_expert_indices = None
        else:
            # Directly append turn output
            turn_ids = turn_output.output_ids + turn_output.obs_ids
            agent_loop_state.response_end_idx = len(agent_loop_state.input_ids) + len(turn_output.output_ids) - 1
            agent_loop_state.input_ids += turn_ids
            agent_loop_state.loss_mask += loss_mask_for_turn
            if agent_loop_state.rollout_logprobs is not None and rollout_logprobs_for_turn is not None:
                agent_loop_state.rollout_logprobs += rollout_logprobs_for_turn
            if agent_loop_state.rollout_expert_indices is not None and rollout_expert_indices_for_turn is not None:
                # overwrite the existing rollout inference indices, since the inference engine should
                # return the expert indices for the entire sequence including each turn's input
                # and the final response should not have an observation appended to it
                agent_loop_state.rollout_expert_indices = rollout_expert_indices_for_turn

        return agent_loop_state

    def _update_agent_loop_state_with_singleturn_chat_template(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is False. It assumes that the input to the LLM is a list of token ids
        and that the multi-turn conversation happens in a single assistant message.

        For example (using the Qwen 2.5 chat template), a trajectory for single-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...

                            turn 1 env observation goes here
                            <observation>...</observation>

                            turn 2 model response goes here:
                            <think>... </think>
                            ...

        Args:
            agent_loop_state: Current agent loop state containing chat history, input IDs, loss mask, etc.
            turn_output: Turn output containing the model's response, output IDs, logprobs, and observations

        Returns:
            AgentLoopState: Updated agent loop state with appended turn IDs, loss mask, and logprobs.
                The EOS token is removed from response tokens (if present) since we are continuing
                the current assistant message. Observations are encoded directly without chat template formatting.
        """
        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        obs_ids_to_add = turn_output.obs_ids

        # Remove EOS token from response tokens since we are continuing the current assistant message
        new_resp_tokens = turn_output.output_ids.copy()
        if new_resp_tokens and new_resp_tokens[-1] == self.tokenizer.eos_token_id:
            new_resp_tokens = new_resp_tokens[:-1]

        turn_ids = new_resp_tokens + obs_ids_to_add
        loss_mask_for_turn = [1] * len(new_resp_tokens) + [0] * len(obs_ids_to_add)
        rollout_logprobs_for_turn = None
        if turn_output.output_logprobs is not None:
            # For response tokens, use actual logprobs
            # for obs tokens, use dummy values
            rollout_logprobs_for_turn = turn_output.output_logprobs[: len(new_resp_tokens)] + [0.0] * len(
                obs_ids_to_add
            )

        # Directly append turn output
        agent_loop_state.response_end_idx = len(agent_loop_state.input_ids) + len(new_resp_tokens) - 1
        agent_loop_state.input_ids += turn_ids
        agent_loop_state.loss_mask += loss_mask_for_turn
        if agent_loop_state.rollout_logprobs is not None and rollout_logprobs_for_turn is not None:
            agent_loop_state.rollout_logprobs += rollout_logprobs_for_turn
        if (
            self.generator_cfg.inference_engine.enable_return_routed_experts
            and turn_output.rollout_expert_indices is not None
        ):
            # overwrite the existing rollout inference indices, since the inference engine should
            # return the expert indices for the entire sequence including each turn's input and observation tokens
            # and the final response should not have an observation appended to it
            agent_loop_state.rollout_expert_indices = turn_output.rollout_expert_indices

        return agent_loop_state


if __name__ == "__main__":
    """
    Standalone entrypoint: generate one trajectory for a single dataset row.

    Two task types:
      - Single-paper (default): uses alphaXiv/rlm-sft-Qwen3.5-9B-v1 with
        single-paper parquet data from ~/data/rlm/validation.parquet.
      - Multi-paper (--multi_paper): uses alphaXiv/rlm-sft-multi-9b-v1-epoch-4
        with multi-paper parquet data from ~/data/multi-paper/validation.parquet,
        multipaper system prompts, and eval-tuned sampling params. All defaults
        can be overridden by explicit CLI args.

    Two engine modes:
      1. Cold start  – spins up a fresh vLLM engine via Ray (slower first run).
      2. Hot connect  – pass --vllm_url <host:port> to reuse a running vLLM
         server, skipping model load entirely.  Start the server once with:

           vllm serve alphaXiv/rlm-sft-Qwen3.5-9B-v1 \
               --host 0.0.0.0 --port 8000 \
               --dtype bfloat16 \
               --max-model-len 32768 --gpu-memory-utilization 0.95 \
               --language-model-only --enable-prefix-caching \
               --cudagraph-capture-sizes 1

         Single-paper example:

           python -m skyrl.train.generators.skyrl_gym_generator \
               --vllm_url localhost:8000 --row_idx 0 \
               "data.val_data=['$HOME/data/rlm/validation.parquet']" \
               environment.env_class=rlm \
               generator.max_turns=10 \
               trainer.policy.model.path=alphaXiv/rlm-sft-Qwen3.5-9B-v1 \
               trainer.max_prompt_length=32768 \
               generator.max_input_length=32768 \
               generator.eval_sampling_params.max_generate_length=1024 \
               generator.eval_sampling_params.temperature=1.0 \
               generator.chat_template_kwargs.enable_thinking=false

         Multi-paper example (uses built-in defaults, just pass --multi_paper):

           python -m skyrl.train.generators.skyrl_gym_generator \
               --multi_paper --vllm_url localhost:8000 --row_idx 0

         Or override the data dir / model:

           python -m skyrl.train.generators.skyrl_gym_generator \
               --multi_paper --vllm_url localhost:8000 --row_idx 0 \
               "data.val_data=['$HOME/data/multi-paper/test.parquet']" \
               trainer.policy.model.path=my-org/my-multi-paper-model
    """
    import json
    import sys
    import time as _time

    import asyncio

    from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.generators.base import TrajectoryID
    from skyrl.train.utils.utils import validate_generator_cfg
    from skyrl.utils.tok import get_tokenizer  # noqa: E402

    _t0 = _time.perf_counter()

    # ---- Strip custom flags from argv before handing rest to config ----
    argv = sys.argv[1:]

    row_idx = 0
    if "--row_idx" in argv:
        pos = argv.index("--row_idx")
        row_idx = int(argv[pos + 1])
        argv = argv[:pos] + argv[pos + 2:]

    vllm_url: str | None = None
    if "--vllm_url" in argv:
        pos = argv.index("--vllm_url")
        vllm_url = argv[pos + 1]
        argv = argv[:pos] + argv[pos + 2:]

    multi_paper = False
    if "--multi_paper" in argv:
        pos = argv.index("--multi_paper")
        multi_paper = True
        argv = argv[:pos] + argv[pos + 1:]

    dump_eval = False
    if "--dump_eval" in argv:
        pos = argv.index("--dump_eval")
        dump_eval = True
        argv = argv[:pos] + argv[pos + 1:]

    if multi_paper:
        # Prepend multi-paper defaults; explicit user args that follow will override.
        _mp_defaults = [
            "trainer.policy.model.path=alphaXiv/rlm-sft-multi-9b-v1-epoch-4",
            "data.val_data=['~/data/multi-paper/validation.parquet']",
            "environment.env_class=rlm",
            "environment.skyrl_gym.rlm.custom_system_prompt=multipaper",
            "environment.skyrl_gym.rlm.child_system_prompt=multipaper_child",
            "generator.max_turns=10",
            "generator.eval_sampling_params.max_generate_length=4096",
            "generator.eval_sampling_params.temperature=0.7",
            "generator.eval_sampling_params.top_p=0.8",
            "generator.eval_sampling_params.top_k=20",
            "generator.eval_sampling_params.min_p=0.0",
            "generator.eval_sampling_params.repetition_penalty=1.0",
            "generator.eval_sampling_params.additional_kwargs.presence_penalty=1.5",
            "trainer.max_prompt_length=32768",
            "generator.max_input_length=32768",
            "generator.chat_template_kwargs.enable_thinking=false",
            "generator.child_openrouter_model=openai/gpt-5.4-nano",
        ]
        argv = _mp_defaults + argv
        logger.info("Multi-paper mode: injected default overrides (model, data, prompts, sampling)")

    cfg = SkyRLTrainConfig.from_cli_overrides(argv)

    # ---- Fast defaults for 1-GPU dev loop ----
    ie = cfg.generator.inference_engine
    ie.enforce_eager = False
    ie.num_engines = 1
    ie.tensor_parallel_size = 1
    ie.pipeline_parallel_size = 1
    ie.data_parallel_size = 1
    ie.max_num_seqs = 8
    ie.gpu_memory_utilization = 0.95
    ie.ray_actor_max_concurrency = 1
    ie.engine_init_kwargs.setdefault("language_model_only", True)
    ie.engine_init_kwargs.setdefault("async_scheduling", True)
    ie.engine_init_kwargs["compilation_config"] = {"cudagraph_capture_sizes": [1]}
    cfg.generator.batched = False
    cfg.generator.step_wise_trajectories = True
    cfg.trainer.placement.colocate_all = False

    validate_generator_cfg(cfg)

    # ---- Tokenizer (needed for chat template / decoding) ----
    tokenizer = get_tokenizer(
        cfg.trainer.policy.model.path,
        trust_remote_code=True,
        use_fast=not cfg.trainer.disable_fast_tokenizer,
        padding_side="left",
    )

    # ---- Load single row directly from parquet (no PromptDataset overhead) ----
    import os
    import pyarrow.parquet as pq

    _parquet_path = os.path.expanduser(cfg.data.val_data[0])
    _table = pq.read_table(_parquet_path)
    assert row_idx < len(_table), (
        f"--row_idx {row_idx} out of range (parquet has {len(_table)} rows)"
    )
    _row = {col: _table.column(col)[row_idx].as_py() for col in _table.column_names}
    messages = _row.pop("prompt")
    env_class = _row.pop("env_class", None) or cfg.environment.env_class
    env_extras = _row
    uid = str(row_idx)
    logger.info(f"Config + row load: {_time.perf_counter() - _t0:.1f}s")

    async def _run() -> None:

        _t1 = _time.perf_counter()

        if vllm_url is not None:
            # ---- Hot path: connect to an already-running vLLM server ----
            from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
                InferenceEngineClient,
            )
            from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
                RemoteInferenceEngine,
            )

            engine = RemoteInferenceEngine(
                url=vllm_url,
                model_name=cfg.trainer.policy.model.path,
                engine_backend="vllm",
                tokenizer=tokenizer,
                tp_size=1, pp_size=1, dp_size=1, ep_size=1,
            )
            inference_engine_client = InferenceEngineClient(
                engines=[engine],
                tokenizer=tokenizer,
                model_path=cfg.trainer.policy.model.path,
                lora_cfg=cfg.trainer.policy.model.lora,
                inference_engine_cfg=ie,
            )
            logger.info(f"Connected to vLLM at {vllm_url}: {_time.perf_counter() - _t1:.1f}s")
        else:
            # ---- Cold path: spin up vLLM via Ray ----
            from skyrl.train.entrypoints.main_base import BasePPOExp
            from skyrl.train.utils.utils import initialize_ray

            initialize_ray(cfg)

            class _Exp(BasePPOExp):
                def get_train_dataset(self):
                    return None
                def get_eval_dataset(self):
                    return None

            exp = _Exp(cfg)
            inference_engine_client = exp.get_inference_client()
            await inference_engine_client.wake_up()
            logger.info(f"vLLM engine cold-started: {_time.perf_counter() - _t1:.1f}s")

        generator = SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend,
            cfg.generator.eval_sampling_params,
        )

        # Inject RLM config into env_extras (mirrors what generate() does)
        if env_class == "rlm":
            rlm_config = getattr(cfg.environment.skyrl_gym, "rlm", None)
            if rlm_config is not None:
                _rlm_cfg = vars(rlm_config) if hasattr(rlm_config, "__dict__") else rlm_config
                if _rlm_cfg.get("custom_system_prompt"):
                    env_extras["custom_system_prompt"] = _rlm_cfg["custom_system_prompt"]
                if _rlm_cfg.get("child_system_prompt"):
                    env_extras["child_system_prompt"] = _rlm_cfg["child_system_prompt"]

        max_tokens = cfg.generator.eval_sampling_params.max_generate_length
        max_input_length = cfg.generator.max_input_length

        _task_label = "multi-paper" if multi_paper else "single-paper"
        logger.info(f"[{_task_label}] Row {row_idx} | uid={uid} | env_class={env_class}")

        _t2 = _time.perf_counter()
        agent_output = await generator.agent_loop(
            prompt=messages,
            env_class=env_class,
            env_extras=env_extras,
            max_tokens=max_tokens,
            max_input_length=max_input_length,
            sampling_params=sampling_params,
            trajectory_id=TrajectoryID(instance_id=uid, repetition_id=0),
        )
        logger.info(f"agent_loop() took {_time.perf_counter() - _t2:.1f}s")

        import pathlib
        _traj_prefix = "mp_trajectory" if multi_paper else "trajectory"
        traj_root = pathlib.Path("trajectories")
        existing = sorted(traj_root.glob(f"{_traj_prefix}_*")) if traj_root.exists() else []
        next_id = len(existing) + 1
        traj_dir = traj_root / f"{_traj_prefix}_{next_id}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        def _save_stepwise_output(sw_output, traj_dir, depth=0, child_idx=None):
            """Recursively save parent + child step outputs with depth in filename."""
            n_saved = 0
            for step_idx, step_output in enumerate(sw_output.step_outputs):
                prompt_text = tokenizer.decode(step_output.prompt_ids)
                response_text = tokenizer.decode(step_output.response_ids)
                is_last = step_idx == len(sw_output.step_outputs) - 1

                step_data = {
                    "step": step_idx + 1,
                    "depth": depth,
                    "child_index": child_idx,
                    "is_last": is_last,
                    "stop_reason": step_output.stop_reason,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "prompt_token_ids": step_output.prompt_ids,
                    "response_token_ids": step_output.response_ids,
                    "latency": step_output.env_metrics.get("latency", {}),
                }

                if child_idx is not None:
                    filename = f"depth-{depth}_child-{child_idx}_step_{step_idx + 1}.json"
                else:
                    filename = f"depth-{depth}_step_{step_idx + 1}.json"

                (traj_dir / filename).write_text(json.dumps(step_data, indent=2, ensure_ascii=False))
                n_saved += 1

            for ci, child_output in enumerate(sw_output.child_outputs or []):
                n_saved += _save_stepwise_output(child_output, traj_dir, depth=depth + 1, child_idx=ci)

            return n_saved

        n_parent = len(agent_output.step_outputs)
        n_children = len(agent_output.child_outputs) if hasattr(agent_output, "child_outputs") else 0
        total_saved = _save_stepwise_output(agent_output, traj_dir)

        # Save rollout metrics from the last step's env_metrics + OpenRouter pricing
        _PRICING = {
            "openai/gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
            "openai/gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
        }
        last_metrics = agent_output.step_outputs[-1].env_metrics if agent_output.step_outputs else {}
        usage = generator.openrouter_usage
        if usage["requests"] > 0:
            model_name = cfg.generator.child_openrouter_model
            pricing = _PRICING.get(model_name, _PRICING["openai/gpt-5.4-mini"])
            if model_name not in _PRICING:
                logger.warning(f"No pricing entry for {model_name}, falling back to gpt-5.4-mini rates")

            prompt_tok = usage["prompt_tokens"]
            cached_tok = usage["cached_tokens"]
            uncached_tok = prompt_tok - cached_tok
            completion_tok = usage["completion_tokens"]
            cost_input = uncached_tok / 1_000_000 * pricing["input"]
            cost_cached = cached_tok / 1_000_000 * pricing["cached_input"]
            cost_output = completion_tok / 1_000_000 * pricing["output"]
            cost_total = cost_input + cost_cached + cost_output
            last_metrics["openrouter"] = {
                "requests": usage["requests"],
                "prompt_tokens": prompt_tok,
                "cached_tokens": cached_tok,
                "uncached_tokens": uncached_tok,
                "completion_tokens": completion_tok,
                "cost_input_usd": round(cost_input, 6),
                "cost_cached_usd": round(cost_cached, 6),
                "cost_output_usd": round(cost_output, 6),
                "cost_total_usd": round(cost_total, 6),
                "model": model_name,
                "pricing_per_million": pricing,
            }
            logger.info(
                f"OpenRouter usage: {usage['requests']} requests, "
                f"{uncached_tok:,} uncached + {cached_tok:,} cached input tok, "
                f"{completion_tok:,} output tok → ${cost_total:.4f}"
            )
        if last_metrics:
            meta_path = traj_dir / "metadata.json"
            meta_path.write_text(json.dumps(last_metrics, indent=2))
            logger.info(f"Rollout metrics:\n{json.dumps(last_metrics, indent=2)}")

        logger.info(
            f"Wrote {total_saved} files to {traj_dir} "
            f"({n_parent} parent steps, {n_children} child rollouts)"
        )

        if dump_eval:
            from collections import defaultdict as _defaultdict

            eval_dump_dir = traj_dir / "eval_dump"
            eval_dump_dir.mkdir(parents=True, exist_ok=True)

            rlm_entries: dict[str, list] = _defaultdict(list)

            def _collect_for_eval(sw_output, depth=0, child_idx=None):
                if depth == 0:
                    rlm_key = "root_rlm"
                else:
                    rlm_key = f"child_rlm_{child_idx}"
                for j, step_out in enumerate(sw_output.step_outputs):
                    rlm_entries[rlm_key].append({
                        "step_index": j,
                        "depth": depth,
                        "child_index": child_idx,
                        "is_last_step": j == len(sw_output.step_outputs) - 1,
                        "stop_reason": step_out.stop_reason,
                        "input_prompt": tokenizer.decode(step_out.prompt_ids),
                        "output_response": tokenizer.decode(step_out.response_ids),
                        "latency": step_out.env_metrics.get("latency", {}),
                        "env_class": env_class,
                    })
                for ci, child_out in enumerate(sw_output.child_outputs or []):
                    _collect_for_eval(child_out, depth=depth + 1, child_idx=ci)

            _collect_for_eval(agent_output)

            manifest = {}
            for rlm_key, entries in sorted(rlm_entries.items()):
                filename = eval_dump_dir / f"{rlm_key}.jsonl"
                with open(filename, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest[rlm_key] = {
                    "file": str(filename.name),
                    "num_steps": len(entries),
                    "depth": entries[0]["depth"],
                    "child_index": entries[0]["child_index"],
                }
                logger.info(f"  {rlm_key}: {len(entries)} steps → {filename.name}")

            manifest_path = eval_dump_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            logger.info(f"Eval dump written to {eval_dump_dir} ({len(rlm_entries)} RLM files + manifest)")

        logger.info(f"Total wall time: {_time.perf_counter() - _t0:.1f}s")

    asyncio.run(_run())
