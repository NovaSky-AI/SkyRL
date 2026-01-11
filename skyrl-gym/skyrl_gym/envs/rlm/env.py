import json
import os
import re
import tempfile
import itertools
import uuid
from typing import Optional

from omegaconf import DictConfig
from openai import OpenAI

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.rlm import utils


class RLMEnvironment(BaseTextEnv):
    """
    RLM Loop Environment for SkyRL Gym
    """

    def __init__(
        self,
        *,
        max_turns: int = 10,
        max_tokens: int = 1024,
        max_repl_output_chars: int = 4000,
    ):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self.globals_dict: dict = {}
        self.locals_dict: dict = {}

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_repl_output_chars = max_repl_output_chars
        self._reset_repl_namespace()

        self.lm_client: OpenAI | None = None
        self.model: str | None = None
        self.init_prompt: str | None = None

    def _reset_repl_namespace(self) -> None:
        self.globals_dict = {
            "__builtins__": utils.SAFE_BUILTINS.copy(),
            "__name__": "__main__",
            "llm_query": self.llm_query,
        }

    def _context_metadata_prompt(self) -> str:
        ctx = self.locals_dict.get("context", None)
        ctx_type = type(ctx).__name__
        if isinstance(ctx, str):
            total = len(ctx)
            preview = [min(2000, total)]
        elif isinstance(ctx, (list, tuple)):
            total = sum(len(str(x)) for x in ctx)
            preview = [len(str(x)) for x in ctx[:10]]
        elif isinstance(ctx, dict):
            total = sum(len(str(k)) + len(str(v)) for k, v in itertools.islice(ctx.items(), 1000))
            preview = [len(ctx)]
        else:
            total = len(str(ctx)) if ctx is not None else 0
            preview = []

        return f"Your context is a {ctx_type} with {total} total characters. " f"(Preview lens: {preview})"

    def _engine_setup(
        self,
        env_cfg: DictConfig,
    ):
        """Setup the inference engine."""

        openai_api_key = env_cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set (as parameter, in env_cfg, or as environment variable)")

        base_url = env_cfg.base_url
        model = env_cfg.model
        init_prompt = env_cfg.init_prompt
        if not base_url or not model or not init_prompt:
            raise ValueError("env_cfg must include base_url, model, and init_prompt")

        self.lm_client = OpenAI(base_url=base_url, api_key=openai_api_key)
        self.model = model
        self.init_prompt = init_prompt

        # Ensure helper function is available inside the REPL.
        self.globals_dict["llm_query"] = self.llm_query

    def llm_query(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Recursive LM call available inside the REPL."""
        assert self.lm_client is not None, "Inference engine not setup; call _engine_setup(...) first."
        used_model = model or self.model
        assert used_model is not None, "Model not configured; call _engine_setup(...) first."

        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        return (
            self.lm_client.chat.completions.create(
                model=used_model,
                messages=msgs,
            )
            .choices[0]
            .message.content
        )

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict]:
        self.turns = 0
        self._reset_repl_namespace()
        self.locals_dict = {}
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        assert self.lm_client is not None, "Inference engine not setup; call _engine_setup(...) first."
        assert self.model is not None, "Model not configured; call _engine_setup(...) first."
        assert self.init_prompt is not None, "System prompt not configured; call _engine_setup(...) first."

        first_turn_safeguard = (
            "You have not inspected `context` yet. Your first step should be to use a <repl> block "
            "to peek at the structure of `context` (e.g., type, length, small preview). "
            "Do NOT provide a final answer yet.\n\n"
        )
        messages = [
            {"role": "system", "content": self.init_prompt},
            {"role": "assistant", "content": self._context_metadata_prompt()},
            {"role": "user", "content": first_turn_safeguard + action},
        ]

        for i in range(self.max_turns):
            self.turns += 1

            response = self.lm_client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})

            has_final_answer = bool(re.search(r"\bFINAL_VAR\b|\bFINAL\s*\(", response_text, re.IGNORECASE))

            # Find ALL <repl> blocks in the response
            code_matches = re.findall(r"<repl>(.*?)</repl>", response_text, re.DOTALL | re.IGNORECASE)

            utils.print_box(
                f"MODEL OUTPUT (turn {self.turns}/{self.max_turns})",
                response_text,
                color="cyan",
            )
            if not code_matches and not has_final_answer:
                utils.print_box(
                    "NO REPL BLOCKS DETECTED",
                    "Model did not emit any <repl>...</repl> blocks this turn.",
                    color="yellow",
                )

            all_results: list[str] = []
            last_execution_result: dict | None = None
            for idx, code in enumerate(code_matches, start=1):
                code = code.strip()
                utils.print_box(f"EXECUTING REPL CODE (block {idx})", code, color="magenta")
                execution_result = self._execute_code(code)
                last_execution_result = execution_result

                stdout = execution_result.get("stdout", "") or ""
                stderr = execution_result.get("stderr", "") or ""
                stdout, stdout_trunc = utils.truncate_text(stdout, self.max_repl_output_chars)
                stderr, stderr_trunc = utils.truncate_text(stderr, self.max_repl_output_chars)

                result_message: list[str] = []
                if stdout:
                    suffix = "\n[stdout truncated]" if stdout_trunc else ""
                    result_message.append(f"Output:\n{stdout}{suffix}")
                if stderr:
                    suffix = "\n[stderr truncated]" if stderr_trunc else ""
                    result_message.append(f"Error:\n{stderr}{suffix}")
                if not execution_result.get("stdout") and not execution_result.get("stderr"):
                    result_message.append("Code executed successfully (no output)")

                result_text = "\n".join(result_message)
                all_results.append(result_text)

                utils.print_box(
                    f"REPL RESULT (block {idx})",
                    result_text,
                    color="green" if not execution_result.get("stderr") else "red",
                )

            observation_content = "\n\n".join(all_results)
            observation_content, obs_trunc = utils.truncate_text(observation_content, self.max_repl_output_chars * 2)
            if obs_trunc:
                observation_content += "\n[observation truncated]"

            if not observation_content and not has_final_answer:
                observation_content = (
                    "No <repl>...</repl> blocks were found in your last message. "
                    "If you need to run Python, wrap it in <repl> tags. "
                    'If you are done, respond with FINAL("...") or FINAL_VAR(name).'
                )
            messages.append({"role": "user", "content": observation_content})

            if i == 0:
                messages.append(
                    {"role": "user", "content": "Continue. Use <repl> blocks as needed, or finish with FINAL(...)."}
                )

            done = self.turns >= self.max_turns or has_final_answer
            final_answer_value = None
            if has_final_answer:
                final_answer_value = self._extract_final_answer(response_text)

            if done:
                break

        final_observations = messages[-3:] if len(messages) >= 3 else messages

        last_execution_result = last_execution_result or {}
        metadata = {
            "execution_time": last_execution_result.get("execution_time", 0),
            "has_error": bool(last_execution_result.get("stderr", "")),
            "has_final_answer": bool(has_final_answer),
            "turns": self.turns,
            "total_messages": len(messages),
        }
        if final_answer_value is not None:
            metadata["final_answer"] = final_answer_value

        return BaseTextEnvStepOutput(
            observations=final_observations,
            reward=0.0,
            done=done,
            postprocessed_action=response_text,
            metadata=metadata,
        )

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment's local variables."""
        self.locals_dict["context"] = context_payload

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
        else:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context_payload, f, ensure_ascii=False)

        self.locals_dict["context_path"] = context_path

    def execute_code(self, code: str) -> dict:
        """Public wrapper: execute Python in the REPL namespace."""
        return self._execute_code(code)

    def _extract_final_answer(self, response: str) -> Optional[str]:
        final_var_match = re.search(
            r"FINAL_VAR\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
            response,
            re.IGNORECASE,
        )
        if final_var_match:
            variable_name = final_var_match.group(1)
            if variable_name in self.locals_dict:
                value = self.locals_dict[variable_name]
                return str(value)
            return f"Error: Variable '{variable_name}' not found in environment"

        final_answer_match = re.search(
            r'FINAL\s*\(\s*["\'](.+?)["\']\s*\)',
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if final_answer_match:
            return final_answer_match.group(1)

        final_direct_match = re.search(r"FINAL\s*\(\s*(.+?)\s*\)", response, re.IGNORECASE)
        if final_direct_match:
            return final_direct_match.group(1)

        return None

    def _execute_code(self, code: str) -> dict:
        """Code Execution in REPL Environment"""
        import io
        import sys
        import time

        start_time = time.perf_counter()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()

        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf

            combined = {**self.globals_dict, **self.locals_dict}
            exec(code, combined, combined)

            for key, value in combined.items():
                if key not in self.globals_dict and not key.startswith("_"):
                    self.locals_dict[key] = value

            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
        except Exception as e:
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        execution_time = time.perf_counter() - start_time
        return {
            "stdout": stdout,
            "stderr": stderr,
            "locals": self.locals_dict.copy(),
            "execution_time": execution_time,
        }

    def close(self):
        if getattr(self, "temp_dir", None) and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.globals_dict.clear()
        self.locals_dict.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        self.close()
