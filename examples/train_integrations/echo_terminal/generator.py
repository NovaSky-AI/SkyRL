from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics

from .chat_template import choose_obs_role
from .interaction import AddedMessageSpan, TerminalInteraction
from .parsers import check_format_warnings, get_parser

logger = logging.getLogger(__name__)


def truncate_output(output: str, max_chars: int = 6000, strategy: str = "start_end") -> str:
    if len(output) <= max_chars:
        return output
    original_len = len(output)
    if strategy == "start":
        msg = f"\n[Output truncated: showing first {max_chars} of {original_len} characters]"
        return output[:max_chars] + msg
    if strategy == "end":
        msg = f"[Output truncated: showing last {max_chars} of {original_len} characters]\n"
        return msg + output[-max_chars:]
    half = max_chars // 2
    msg = f"\n[Output truncated: showing first {half} and last {half} of {original_len} characters]\n"
    return output[:half] + msg + output[-half:]


def strip_thinking(text: str) -> str:
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result.strip()


@dataclass
class TerminalTrajectoryOutput:
    trajectory_id: TrajectoryID
    prompt_token_ids: list[int]
    response_ids: list[int]
    loss_masks: list[int]
    world_loss_masks: list[int]
    world_warning_masks: list[int]
    world_env_masks: list[int]
    world_full_observation_count: int
    rollout_logprobs: list[float]
    world_model_only: bool = False
    reward: float = 0.0
    correct: bool = False
    stop_reason: str = "error"
    metrics: dict[str, float | int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TerminalAgentGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        max_seq_len: int,
    ) -> None:
        if getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError("TerminalAgentGenerator is non-step-wise. Set generator.step_wise_trajectories=false.")
        self.generator_cfg = generator_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._parser_name = generator_cfg.parser_name
        self._parser = get_parser(self._parser_name)
        self._obs_role = choose_obs_role(tokenizer, candidates=("tool", "user"))
        world_loss_target = generator_cfg.world_loss_target
        if world_loss_target is None:
            world_loss_target = "full_observation" if generator_cfg.world_loss_on_format_warn else "env_only"
        if world_loss_target not in {"full_observation", "env_only", "warning_only", "warning_plus_env"}:
            raise ValueError(f"Unsupported world_loss_target={generator_cfg.world_loss_target!r}")
        self.world_loss_target = world_loss_target
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        self._turn_end_tokens = [self._im_end_id, newline_ids[-1]] if self._im_end_id is not None else newline_ids[-1:]
        self._think_start_id: int | None = None
        self._think_end_id: int | None = None
        if generator_cfg.thinking_handling != "keep_all":
            self._init_thinking_token_ids()

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        env_extras = input_batch.get("env_extras")
        trajectory_ids = input_batch.get("trajectory_ids")
        sampling_params = input_batch.get("sampling_params") or {}
        if env_extras is None or trajectory_ids is None:
            raise ValueError("TerminalAgentGenerator requires env_extras and trajectory_ids.")
        if not (len(prompts) == len(env_extras) == len(trajectory_ids)):
            raise ValueError("prompts, env_extras, and trajectory_ids must have the same length.")

        from .harbor_environment import HarborEnvironmentProvider

        provider = HarborEnvironmentProvider(
            docker_memory_mb=self.generator_cfg.docker_memory_mb,
            docker_cpus=self.generator_cfg.docker_cpus,
            base_temp_dir=Path(self.generator_cfg.base_temp_dir) if self.generator_cfg.base_temp_dir else None,
            max_concurrent_builds=self.generator_cfg.max_concurrent_builds,
            max_build_retries=self.generator_cfg.max_build_retries,
        )
        unique_envs, prompt_idx_by_instance = self._prepare_unique_envs(env_extras, trajectory_ids)
        for extra, trajectory_id in zip(env_extras, trajectory_ids):
            extra["_prompt_idx"] = prompt_idx_by_instance[trajectory_id.instance_id]

        outputs: list[TerminalTrajectoryOutput | None] = [None] * len(prompts)
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Terminal Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        try:
            await provider.prepare_batch(unique_envs, num_generations=self.generator_cfg.n_samples_per_prompt)
            semaphore = asyncio.Semaphore(self.generator_cfg.agent_max_concurrency)

            async def _worker(idx: int) -> None:
                async with semaphore:
                    try:
                        outputs[idx] = await asyncio.wait_for(
                            self._run_one(
                                provider,
                                prompts[idx],
                                env_extras[idx],
                                trajectory_ids[idx],
                                sampling_params,
                            ),
                            timeout=self.generator_cfg.agent_timeout,
                        )
                    except asyncio.TimeoutError:
                        outputs[idx] = self._failure_output(
                            prompts[idx],
                            env_extras[idx],
                            trajectory_ids[idx],
                            "timeout",
                        )
                    except Exception as exc:
                        logger.exception("Terminal rollout failed for %s: %s", trajectory_ids[idx], exc)
                        outputs[idx] = self._failure_output(
                            prompts[idx],
                            env_extras[idx],
                            trajectory_ids[idx],
                            "error",
                        )
                    finally:
                        progress.update(1)

            async with asyncio.TaskGroup() as tg:
                for idx in range(len(prompts)):
                    tg.create_task(_worker(idx))
        finally:
            progress.close()
            await provider.cleanup_batch()

        return self._build_generator_output([o for o in outputs if o is not None])

    def _prepare_unique_envs(
        self,
        env_extras: list[dict[str, Any]],
        trajectory_ids: list[TrajectoryID],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        unique_envs: list[dict[str, Any]] = []
        prompt_idx_by_instance: dict[str, int] = {}
        for extra, tid in zip(env_extras, trajectory_ids):
            if tid.instance_id in prompt_idx_by_instance:
                continue
            prompt_idx_by_instance[tid.instance_id] = len(unique_envs)
            unique_envs.append(extra)
        return unique_envs, prompt_idx_by_instance

    async def _run_one(
        self,
        provider: HarborEnvironmentProvider,
        prompt: list[dict[str, Any]],
        env_extra: dict[str, Any],
        trajectory_id: TrajectoryID,
        sampling_params: dict[str, Any],
    ) -> TerminalTrajectoryOutput:
        prompt_token_ids = list(env_extra.get("prompt_token_ids") or self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        ))
        interaction = TerminalInteraction(
            prompt_id=int(trajectory_id.instance_id) if str(trajectory_id.instance_id).isdigit() else 0,
            completion_id=trajectory_id.repetition_id,
            prompt_messages=copy.deepcopy(prompt),
            prompt_token_ids=prompt_token_ids,
            metadata={
                "path": env_extra.get("path", ""),
                "data_source": env_extra.get("data_source"),
            },
        )
        environment = None
        setup_complete = False
        try:
            environment = await provider.create(env_extra)
            await environment.setup()
            setup_complete = True
            await self._agent_loop(interaction, trajectory_id.to_string(), environment, sampling_params)
        except Exception as exc:
            stop_reason = "env_setup_error" if not setup_complete else "error"
            logger.warning("Environment or agent failed for %s: %s", trajectory_id, exc, exc_info=True)
            self._mark_failure(interaction, stop_reason, exc)
        finally:
            if environment is not None:
                await environment.cleanup()

        self._ensure_non_empty_completion(interaction)
        return TerminalTrajectoryOutput(
            trajectory_id=trajectory_id,
            prompt_token_ids=interaction.prompt_token_ids,
            response_ids=interaction.completion_token_ids,
            loss_masks=interaction.completion_masks,
            world_loss_masks=interaction.completion_observation_masks,
            world_warning_masks=interaction.completion_warning_masks,
            world_env_masks=interaction.completion_env_output_masks,
            world_full_observation_count=int(interaction.metadata.get("full_observation_body_count", 0)),
            rollout_logprobs=interaction.completion_logprobs,
            world_model_only=bool(env_extra.get("world_model_only", False)),
            reward=interaction.reward,
            correct=interaction.correct,
            stop_reason=str(interaction.metadata.get("stop_reason", "error")),
            metrics=interaction.metrics,
            metadata=dict(interaction.metadata),
        )

    async def _agent_loop(
        self,
        interaction: TerminalInteraction,
        session_id: str,
        environment: HarborEnvironment,
        sampling_params: dict[str, Any],
    ) -> None:
        sampling_params = dict(sampling_params)
        sampling_params.setdefault("logprobs", 1)
        stop_reason = "error"
        t_run_start = time.monotonic()
        turn_traces: list[dict[str, Any]] = []
        rollout_context_ids: list[int] | None = None
        if self.generator_cfg.thinking_handling == "strip_rollout":
            rollout_context_ids = list(interaction.prompt_token_ids)

        turn = -1
        for turn in range(self.generator_cfg.max_turns):
            total_len = len(interaction.token_ids)
            if not self._has_token_budget(interaction, 0):
                stop_reason = "max_total_tokens"
                break
            current_len = len(rollout_context_ids) if rollout_context_ids is not None else total_len
            if current_len >= self.generator_cfg.max_context_tokens:
                stop_reason = "max_context_tokens"
                break

            generation_context = rollout_context_ids if rollout_context_ids is not None else interaction.token_ids
            max_tokens = self.generator_cfg.max_tokens_per_generation
            if self.generator_cfg.max_total_tokens is not None:
                max_tokens = min(max_tokens, self.generator_cfg.max_total_tokens - total_len)
            if max_tokens <= 0:
                stop_reason = "max_total_tokens"
                break

            t_gen = time.monotonic()
            token_ids, logprobs = await self._generate_tokens(
                generation_context,
                sampling_params,
                session_id=session_id,
                seed=interaction.completion_id,
                max_tokens=max_tokens,
            )
            generate_sec = time.monotonic() - t_gen
            token_ids = token_ids[:max_tokens]
            logprobs = logprobs[:max_tokens]

            if self.generator_cfg.thinking_handling == "strip_all":
                add_ids, add_logprobs = self._strip_thinking_from_tokens(token_ids, logprobs)
            else:
                add_ids, add_logprobs = token_ids, logprobs
            if not self._has_token_budget(interaction, len(add_ids)):
                stop_reason = "max_total_tokens"
                break

            interaction.append_assistant(add_ids, add_logprobs or [0.0] * len(add_ids), self.tokenizer)
            if rollout_context_ids is not None:
                clean_ids, _ = self._strip_thinking_from_tokens(token_ids)
                rollout_context_ids.extend(clean_ids)

            response_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            format_warnings, format_violations = check_format_warnings(
                response_text,
                tags=self._parser.format_tags,
                parser_name=self._parser_name,
            )
            text_for_parsing = strip_thinking(response_text)
            parse_result = self._parser.parse_response(text_for_parsing)
            parse_result.warnings = format_warnings
            parse_result.violations = format_violations

            if parse_result.error:
                turn_traces.append(
                    self._turn_trace(turn, generate_sec, len(token_ids), 0.0, True, False, current_len, format_violations)
                )
                await self._add_observation(interaction, "", parse_result.error, rollout_context_ids)
                continue

            commands = self._select_commands(parse_result.commands)
            t_exec = time.monotonic()
            terminal_output = await self._execute_commands(environment, commands)
            exec_sec = time.monotonic() - t_exec
            turn_traces.append(
                self._turn_trace(
                    turn,
                    generate_sec,
                    len(token_ids),
                    exec_sec,
                    False,
                    parse_result.is_done,
                    current_len,
                    format_violations,
                )
            )

            if parse_result.is_done:
                stop_reason = "done"
                break
            if not self._has_token_budget(interaction, 0):
                stop_reason = "max_total_tokens"
                break
            warnings_prefix = ""
            if self.generator_cfg.add_format_warn and format_warnings:
                warnings_prefix = "WARNINGS:\n" + "\n".join(f"- {w}" for w in format_warnings) + "\n\n"
            if not commands:
                await self._add_observation(
                    interaction,
                    warnings_prefix,
                    "No commands provided. Please provide commands to execute or set done.",
                    rollout_context_ids,
                )
                continue
            await self._add_observation(interaction, warnings_prefix, terminal_output, rollout_context_ids)
        else:
            stop_reason = "max_turns"

        num_turns = min(turn + 1, self.generator_cfg.max_turns) if turn >= 0 else 0
        interaction.metadata["stop_reason"] = stop_reason
        interaction.metadata["num_agent_turns"] = num_turns

        total_generate_sec = sum(t.get("generate_sec", 0.0) for t in turn_traces)
        total_exec_sec = sum(t.get("exec_sec", 0.0) for t in turn_traces)
        total_generate_tokens = sum(t.get("generate_tokens", 0) for t in turn_traces)
        if stop_reason == "max_total_tokens":
            interaction.reward = 0.0
            interaction.correct = False
            verifier_sec = 0.0
            verifier_error = None
        else:
            t_verify = time.monotonic()
            verifier_reward, verifier_error = await environment.run_verifier(timeout=self.generator_cfg.verifier_timeout)
            verifier_sec = time.monotonic() - t_verify
            interaction.reward = self._compute_reward(verifier_reward, interaction)
            interaction.correct = verifier_reward >= self.generator_cfg.correct_threshold

        interaction.metadata["trace"] = {
            "agent_run_sec": time.monotonic() - t_run_start,
            "verifier_sec": verifier_sec,
            "total_generate_sec": total_generate_sec,
            "total_exec_sec": total_exec_sec,
            "total_tokens": len(interaction.token_ids),
            "total_prompt_tokens": len(interaction.prompt_token_ids),
            "total_completion_tokens": len(interaction.completion_token_ids),
            "total_generate_tokens": total_generate_tokens,
            "turns": turn_traces,
        }
        metrics: dict[str, float | int] = {f"stop_reason/{stop_reason}": 1, "num_agent_turns": num_turns}
        if verifier_error:
            metrics[f"verifier_error/{verifier_error}"] = 1
        for trace in turn_traces:
            for violation in trace.get("format_violations", []):
                metrics[f"format/{violation}"] = metrics.get(f"format/{violation}", 0) + 1
        metrics["parse_errors"] = sum(1 for t in turn_traces if t.get("parse_error"))
        metrics["format_violations_total"] = sum(
            value for key, value in metrics.items() if key.startswith("format/") and isinstance(value, int)
        )
        interaction.metrics = metrics

    async def _generate_tokens(
        self,
        prompt_token_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str,
        seed: int,
        max_tokens: int,
    ) -> tuple[list[int], list[float]]:
        params = dict(sampling_params)
        params["max_tokens"] = max_tokens
        params.setdefault("logprobs", 1)
        params["seed"] = seed
        result = await self.inference_engine_client.generate(
            {
                "prompts": None,
                "prompt_token_ids": [prompt_token_ids],
                "sampling_params": params,
                "session_ids": [session_id],
                "skip_detokenize": True,
            }
        )
        token_ids = result["response_ids"][0]
        logprobs = result.get("response_logprobs")
        if logprobs is None or logprobs[0] is None:
            return token_ids, [0.0] * len(token_ids)
        return token_ids, list(logprobs[0])

    def _get_message_delta_tokens(self, interaction: TerminalInteraction, role: str, text: str) -> list[int]:
        before_ids = self.tokenizer.apply_chat_template(
            interaction.messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        after_ids = self.tokenizer.apply_chat_template(
            interaction.messages + [{"role": role, "content": text}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        return after_ids[len(before_ids) :]

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        shared = 0
        for a_tok, b_tok in zip(a, b):
            if a_tok != b_tok:
                break
            shared += 1
        return shared

    @staticmethod
    def _common_suffix_len(a: list[int], b: list[int], prefix_len: int = 0) -> int:
        max_suffix = min(len(a), len(b)) - prefix_len
        shared = 0
        while shared < max_suffix and a[len(a) - 1 - shared] == b[len(b) - 1 - shared]:
            shared += 1
        return shared

    def _get_observation_content_spans(
        self,
        interaction: TerminalInteraction,
        role: str,
        warning_text: str,
        env_text: str,
    ) -> tuple[int, int, int]:
        empty_delta = self._get_message_delta_tokens(interaction, role, "")
        full_delta = self._get_message_delta_tokens(interaction, role, warning_text + env_text)

        prefix_len = self._common_prefix_len(empty_delta, full_delta)
        suffix_len = self._common_suffix_len(empty_delta, full_delta, prefix_len=prefix_len)
        content_end = len(full_delta) - suffix_len

        if not warning_text:
            return prefix_len, prefix_len, content_end

        warning_delta = self._get_message_delta_tokens(interaction, role, warning_text)
        warning_prefix_len = self._common_prefix_len(empty_delta, warning_delta)
        warning_suffix_len = self._common_suffix_len(empty_delta, warning_delta, prefix_len=warning_prefix_len)
        warning_content_end = len(warning_delta) - warning_suffix_len

        if warning_prefix_len != prefix_len:
            raise ValueError(
                f"Inconsistent observation-message prefix spans: empty/full={prefix_len}, "
                f"empty/warning={warning_prefix_len}."
            )

        return prefix_len, warning_content_end, content_end

    def _apply_mask_indices(self, mask: list[int], indices: list[int]) -> None:
        if self.generator_cfg.max_world_model_tokens is not None:
            indices = indices[: self.generator_cfg.max_world_model_tokens]
        for idx in indices:
            mask[idx] = 1

    async def _add_observation(
        self,
        interaction: TerminalInteraction,
        warning_text: str,
        env_text: str,
        rollout_context_ids: list[int] | None,
    ) -> None:
        text = warning_text + env_text
        if self.generator_cfg.max_total_tokens is not None:
            overhead = len(self._turn_end_tokens) + 10
            remaining = self.generator_cfg.max_total_tokens - len(interaction.token_ids) - overhead
            if remaining <= 0:
                return
            obs_token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if obs_token_count > remaining:
                ratio = remaining / obs_token_count
                warning_text, env_text, text = self._truncate_observation_segments(
                    warning_text,
                    env_text,
                    max(0, int(len(text) * ratio * 0.9)),
                )

        turn_end = self._append_turn_end(interaction)
        if rollout_context_ids is not None:
            rollout_context_ids.extend(turn_end)
        content_start_offset, warning_end_offset, content_end_offset = self._get_observation_content_spans(
            interaction,
            self._obs_role,
            warning_text,
            env_text,
        )
        span, token_ids = self._add_message(interaction, self._obs_role, text, add_generation_prompt=True)

        warning_start = span.message_start + content_start_offset
        warning_end = span.message_start + warning_end_offset
        env_start = warning_end
        env_end = span.message_start + content_end_offset
        interaction.metadata["full_observation_body_count"] = (
            int(interaction.metadata.get("full_observation_body_count", 0))
            + max(0, content_end_offset - content_start_offset)
        )

        if not (span.message_start <= warning_start <= warning_end <= env_start <= env_end <= span.message_end):
            raise ValueError(
                "Invalid observation spans: "
                f"target={self.world_loss_target}, offsets=({content_start_offset}, {warning_end_offset}, "
                f"{content_end_offset}), span={span}"
            )

        for i in range(warning_start, warning_end):
            interaction.completion_warning_masks[i] = 1
        for i in range(env_start, env_end):
            interaction.completion_env_output_masks[i] = 1

        if self.world_loss_target == "full_observation":
            obs_indices = list(range(span.message_start, span.message_end))
        elif self.world_loss_target == "warning_only":
            obs_indices = list(range(warning_start, warning_end))
        elif self.world_loss_target == "warning_plus_env":
            obs_indices = list(range(warning_start, env_end))
        else:
            obs_indices = list(range(env_start, env_end))
        self._apply_mask_indices(interaction.completion_observation_masks, obs_indices)

        if rollout_context_ids is not None:
            rollout_context_ids.extend(token_ids)
        return span

    def _add_message(
        self,
        interaction: TerminalInteraction,
        role: Literal["system", "user", "tool"],
        text: str,
        add_generation_prompt: bool,
    ) -> tuple[AddedMessageSpan, list[int]]:
        before_ids = self.tokenizer.apply_chat_template(
            interaction.messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        after_message_ids = self.tokenizer.apply_chat_template(
            interaction.messages + [{"role": role, "content": text}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        after_full_ids = self.tokenizer.apply_chat_template(
            interaction.messages + [{"role": role, "content": text}],
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
        )
        message_token_ids = after_message_ids[len(before_ids) :]
        token_ids = after_full_ids[len(before_ids) :]
        interaction.append_tokens(token_ids, [0] * len(token_ids), [0.0] * len(token_ids), role=role, text=text)
        completion_start = len(interaction.completion_token_ids) - len(token_ids)
        message_end = completion_start + len(message_token_ids)
        return (
            AddedMessageSpan(
                completion_start=completion_start,
                completion_end=completion_start + len(token_ids),
                message_start=completion_start,
                message_end=message_end,
                generation_prompt_start=message_end,
                generation_prompt_end=completion_start + len(token_ids),
            ),
            token_ids,
        )

    def _append_turn_end(self, interaction: TerminalInteraction) -> list[int]:
        tail = interaction.completion_token_ids[-len(self._turn_end_tokens) :]
        if tail == self._turn_end_tokens:
            return []
        if tail and self._im_end_id is not None and tail[-1] == self._im_end_id:
            tokens = self._turn_end_tokens[1:]
        else:
            tokens = self._turn_end_tokens
        interaction.completion_token_ids.extend(tokens)
        interaction.completion_masks.extend([0] * len(tokens))
        interaction.completion_observation_masks.extend([0] * len(tokens))
        interaction.completion_warning_masks.extend([0] * len(tokens))
        interaction.completion_env_output_masks.extend([0] * len(tokens))
        interaction.completion_logprobs.extend([0.0] * len(tokens))
        return tokens

    async def _execute_commands(self, environment: HarborEnvironment, commands: list) -> str:
        if not commands:
            return ""
        outputs = []
        for parsed_cmd in commands:
            raw_text = parsed_cmd.raw
            if parsed_cmd.error:
                outputs.append(f"Command '{raw_text}' skipped due to parse error: {parsed_cmd.error}")
                continue
            if parsed_cmd.name != "bash":
                outputs.append(
                    f"Command '{raw_text}' skipped due to Unknown tool '{parsed_cmd.name}'. Only 'bash' is supported."
                )
                continue
            cmd = parsed_cmd.arguments.get("command", "")
            timeout = parsed_cmd.arguments.get("timeout", 30)
            if not cmd:
                outputs.append(
                    f"Command '{raw_text}' skipped due to no command provided. Please provide a command in correct format to execute."
                )
                continue
            try:
                result = await environment.exec(cmd, timeout=float(timeout))
            except (RuntimeError, TimeoutError) as exc:
                outputs.append(f"Command '{cmd}' timed out after {timeout} seconds.\n\n(exit_code=-1)")
                logger.warning("Command failed/timed out: %r (%s)", cmd, exc)
                continue
            except Exception as exc:
                outputs.append(f"Command '{cmd}' execution error: {exc}\n\n(exit_code=-1)")
                continue
            raw_output = result.stdout or ""
            if result.stderr:
                raw_output += f"\n{result.stderr}"
            truncated = truncate_output(
                raw_output,
                max_chars=self.generator_cfg.max_terminal_output_chars,
                strategy=self.generator_cfg.terminal_output_truncation,
            )
            status = "executed successfully" if result.return_code == 0 else "failed"
            outputs.append(f"Command '{cmd}' {status}. Output: {truncated}\n\n(exit_code={result.return_code})")
        return "\n\n".join(outputs)

    def _select_commands(self, commands: list) -> list:
        if self.generator_cfg.command_selection == "all":
            return commands
        if self.generator_cfg.command_selection == "last":
            return commands[-self.generator_cfg.max_commands_per_turn :]
        return commands[: self.generator_cfg.max_commands_per_turn]

    def _compute_reward(self, verifier_reward: float, interaction: TerminalInteraction) -> float:
        total_reward = verifier_reward
        completion_len = len(interaction.completion_token_ids)
        if (
            self.generator_cfg.length_penalty_coef > 0
            and completion_len > self.generator_cfg.length_penalty_threshold
        ):
            excess = completion_len - self.generator_cfg.length_penalty_threshold
            total_reward -= self.generator_cfg.length_penalty_coef * (excess / 1000)
        return float(total_reward)

    def _mark_failure(
        self,
        interaction: TerminalInteraction,
        stop_reason: str,
        exc: Exception | None = None,
    ) -> None:
        interaction.metadata["stop_reason"] = stop_reason
        if exc is not None:
            interaction.metadata["error_type"] = type(exc).__name__
            interaction.metadata["error_message"] = str(exc)
        interaction.reward = 0.0
        interaction.correct = False
        metrics: dict[str, float | int] = {f"stop_reason/{stop_reason}": 1}
        if stop_reason == "env_setup_error":
            metrics["env_setup_errors"] = 1
            if exc is not None:
                interaction.metadata["env_setup_error"] = str(exc)
        interaction.metrics = metrics
        self._ensure_non_empty_completion(interaction)

    def _failure_output(
        self,
        prompt: list[dict[str, Any]],
        env_extra: dict[str, Any],
        trajectory_id: TrajectoryID,
        stop_reason: str,
    ) -> TerminalTrajectoryOutput:
        prompt_token_ids = list(env_extra.get("prompt_token_ids") or self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        ))
        eos_id = self.tokenizer.eos_token_id or 0
        return TerminalTrajectoryOutput(
            trajectory_id=trajectory_id,
            prompt_token_ids=prompt_token_ids,
            response_ids=[eos_id],
            loss_masks=[0],
            world_loss_masks=[0],
            world_warning_masks=[0],
            world_env_masks=[0],
            world_full_observation_count=0,
            rollout_logprobs=[0.0],
            world_model_only=bool(env_extra.get("world_model_only", False)),
            reward=0.0,
            correct=False,
            stop_reason=stop_reason,
            metrics={f"stop_reason/{stop_reason}": 1},
            metadata={
                "stop_reason": stop_reason,
                "path": env_extra.get("path", ""),
                "data_source": env_extra.get("data_source"),
            },
        )

    def _ensure_non_empty_completion(self, interaction: TerminalInteraction) -> None:
        if interaction.completion_token_ids:
            return
        eos_id = self.tokenizer.eos_token_id or 0
        interaction.completion_token_ids = [eos_id]
        interaction.completion_masks = [0]
        interaction.completion_observation_masks = [0]
        interaction.completion_warning_masks = [0]
        interaction.completion_env_output_masks = [0]
        interaction.completion_logprobs = [0.0]

    def _has_token_budget(self, interaction: TerminalInteraction, num_tokens: int) -> bool:
        if self.generator_cfg.max_total_tokens is None:
            return True
        return len(interaction.token_ids) + num_tokens <= self.generator_cfg.max_total_tokens

    def _init_thinking_token_ids(self) -> None:
        unk_id = self.tokenizer.unk_token_id
        think_start = self.tokenizer.convert_tokens_to_ids("<think>")
        think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        if think_start is not None and think_start != unk_id:
            self._think_start_id = think_start
        if think_end is not None and think_end != unk_id:
            self._think_end_id = think_end

    def _strip_thinking_from_tokens(
        self,
        token_ids: list[int],
        logprobs: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None]:
        if self._think_start_id is None:
            return token_ids, logprobs
        clean_ids: list[int] = []
        clean_logprobs: list[float] | None = [] if logprobs is not None else None
        in_thinking = False
        for i, token_id in enumerate(token_ids):
            if token_id == self._think_start_id:
                in_thinking = True
                continue
            if in_thinking and self._think_end_id is not None and token_id == self._think_end_id:
                in_thinking = False
                continue
            if not in_thinking:
                clean_ids.append(token_id)
                if clean_logprobs is not None:
                    clean_logprobs.append(logprobs[i])
        return clean_ids, clean_logprobs

    @staticmethod
    def _truncate_observation_segments(warning_text: str, env_text: str, max_chars: int) -> tuple[str, str, str]:
        full_text = warning_text + env_text
        if len(full_text) <= max_chars:
            return warning_text, env_text, full_text
        truncated_full = full_text[:max_chars]
        warning_len = min(len(warning_text), len(truncated_full))
        return truncated_full[:warning_len], truncated_full[warning_len:], truncated_full

    @staticmethod
    def _turn_trace(
        turn: int,
        generate_sec: float,
        generate_tokens: int,
        exec_sec: float,
        parse_error: bool,
        is_done: bool,
        context_len: int,
        format_violations: list[str],
    ) -> dict[str, Any]:
        return {
            "turn": turn,
            "generate_sec": generate_sec,
            "generate_tokens": generate_tokens,
            "exec_sec": exec_sec,
            "parse_error": parse_error,
            "is_done": is_done,
            "context_len": context_len,
            "format_violations": format_violations,
        }

    def _build_generator_output(self, trajectory_outputs: list[TerminalTrajectoryOutput]) -> GeneratorOutput:
        prompt_token_ids = [t.prompt_token_ids for t in trajectory_outputs]
        response_ids = [t.response_ids for t in trajectory_outputs]
        rewards = [t.reward for t in trajectory_outputs]
        loss_masks = [t.loss_masks for t in trajectory_outputs]
        stop_reasons = [t.stop_reason for t in trajectory_outputs]
        rollout_logprobs = [t.rollout_logprobs for t in trajectory_outputs]
        world_loss_masks = [t.world_loss_masks for t in trajectory_outputs]
        world_warning_masks = [t.world_warning_masks for t in trajectory_outputs]
        world_env_masks = [t.world_env_masks for t in trajectory_outputs]
        world_full_observation_counts = [t.world_full_observation_count for t in trajectory_outputs]
        world_model_only = [t.world_model_only for t in trajectory_outputs]
        correct = [t.correct for t in trajectory_outputs]
        trajectory_metadata = [t.metadata for t in trajectory_outputs]
        rollout_metrics = get_rollout_metrics(response_ids, rewards)
        for key in set().union(*(t.metrics.keys() for t in trajectory_outputs)):
            values = [t.metrics.get(key, 0) for t in trajectory_outputs]
            rollout_metrics[f"generate/{key}"] = sum(values)
        rollout_metrics["generate/avg_num_turns"] = (
            sum(t.metrics.get("num_agent_turns", 0) for t in trajectory_outputs) / len(trajectory_outputs)
            if trajectory_outputs
            else 0
        )
        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "world_loss_masks": world_loss_masks,
            "world_warning_masks": world_warning_masks,
            "world_env_masks": world_env_masks,
            "world_full_observation_counts": world_full_observation_counts,
            "world_model_only": world_model_only,
            "correct": correct,
            "trajectory_metadata": trajectory_metadata,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": [t.trajectory_id for t in trajectory_outputs],
            "rollout_expert_indices": None,
            "is_last_step": None,
            "pixel_values": None,
            "image_grid_thw": None,
        }
