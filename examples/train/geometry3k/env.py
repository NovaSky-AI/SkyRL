"""
Multi-turn environment for Geometry-3K multi-modal math problems.

The model sees a geometry diagram + question, reasons, and can call a
``calc_score`` tool to check its answer against the ground truth. The
environment returns feedback so the model can iteratively refine.

Adapted from the Slime RL environment:
https://github.com/THUDM/slime/blob/8efb1166c8e4bb4c810059b0161cbd06bcfbb6cf/examples/geo3k_vlm_multi_turn/env_geo3k.py
"""

import json
import logging
import re
from typing import Any, Dict, List

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .math_utils import extract_boxed_answer, grade_answer_from_boxed

logger = logging.getLogger(__name__)

# Matches JSON payload between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SUPPORTED_TOOL_NAMES = {"calc_score"}


class Geometry3kEnv(BaseTextEnv):
    """
    Multi-turn environment for Geometry-3K visual math problems.

    Interaction protocol:
        1. Model receives image + question prompt (via init).
        2. Model reasons then emits a tool call:
           <tool_call>{"name": "calc_score", "arguments": {"answer": "..."}}</tool_call>
        3. Env scores the answer and returns feedback as an observation.
        4. Episode ends when: answer is correct, max_turns reached, or no tool call found.

    Rewards:
        - Intermediate turns: 0.0
        - Final turn correct: 1.0
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = str(extras["reward_spec"]["ground_truth"])

        self.max_turns = int(extras.get("max_turns", 3))
        self.correct = False

    def _extract_tool_call(self, text: str) -> Dict[str, Any] | None:
        """Parse the latest tool call payload from ``<tool_call>`` tags."""
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None

        raw_json = matches[-1].group(1).strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Failed to decode tool call payload: %s", raw_json)
            return None

        func_field = payload.get("function")
        func_dict = func_field if isinstance(func_field, dict) else {}
        name = payload.get("name") or func_dict.get("name")
        arguments = payload.get("arguments") or func_dict.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_answer(self, answer: str) -> float:
        """Score an answer against ground truth using math grading utilities."""
        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")

        for candidate in candidates:
            try:
                if grade_answer_from_boxed(candidate, self.ground_truth):
                    return 1.0
            except Exception:
                continue
        return 0.0

    def _extract_answer_from_text(self, text: str) -> str | None:
        """Extract answer: prefer \\boxed{}, fall back to last non-empty line."""
        boxed = extract_boxed_answer(text)
        if boxed:
            return str(boxed).strip()
        for line in reversed(text.splitlines()):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:512]
        return None

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        """Build contextual feedback for the model."""
        is_final_warning = self.max_turns >= 2 and self.turns >= self.max_turns - 1

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )

        if is_final_warning:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                "Since you only have one chance to answer, don't call tool again. "
                "Provide your final answer in the form: Answer: \\boxed{{$Answer}}"
            )

        return (
            f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
            "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        is_final_turn = self.turns >= self.max_turns

        tool_call = self._extract_tool_call(action)

        # No tool call detected - try to extract answer from text directly.
        if not tool_call:
            answer_text = self._extract_answer_from_text(action)
            if answer_text:
                score = self._score_answer(answer_text)
                self.correct = score == 1.0
            return BaseTextEnvStepOutput(
                observations=[],
                reward=1.0 if self.correct else 0.0,
                done=True,
                metadata={"ground_truth": self.ground_truth, "tool_call": False},
            )

        # Validate tool name.
        name = (tool_call.get("name") or "").strip()
        if name not in SUPPORTED_TOOL_NAMES:
            feedback = (
                f"Tool `{name}` is not supported. "
                'Call `calc_score` via <tool_call>{{"name": "calc_score", "arguments": {{"answer": "<your_answer>"}}}}</tool_call> '
                "to check your solution."
            )
            return BaseTextEnvStepOutput(
                observations=[] if is_final_turn else [{"role": "user", "content": feedback}],
                reward=0.0,
                done=is_final_turn,
                metadata={"ground_truth": self.ground_truth, "tool_call": True, "unsupported_tool": name},
            )

        # Extract and score the answer.
        arguments = tool_call.get("arguments") or {}
        raw_answer = arguments.get("answer", "")
        parsed_answer = str(raw_answer).strip() if raw_answer else ""

        if not parsed_answer:
            feedback = (
                "Tool call detected but no `answer` was provided. "
                'Call `calc_score` via <tool_call>{{"name": "calc_score", "arguments": {{"answer": "<your_answer>"}}}}</tool_call> '
                "to check your solution."
            )
            return BaseTextEnvStepOutput(
                observations=[] if is_final_turn else [{"role": "user", "content": feedback}],
                reward=0.0,
                done=is_final_turn,
                metadata={"ground_truth": self.ground_truth, "tool_call": True, "answer_missing": True},
            )

        score = self._score_answer(parsed_answer)
        self.correct = score == 1.0

        # Episode ends if correct or final turn.
        done = self.correct or is_final_turn
        reward = score if done else 0.0

        if done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={
                    "ground_truth": self.ground_truth,
                    "tool_call": True,
                    "score": score,
                    "answer": parsed_answer,
                },
            )

        # Intermediate turn - return feedback.
        feedback = self._build_tool_feedback(score, parsed_answer)
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=0.0,
            done=False,
            metadata={"ground_truth": self.ground_truth, "tool_call": True, "score": score, "answer": parsed_answer},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "acc": 1.0 if self.correct else 0.0,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        n = len(metrics)
        avg_turns = sum(float(m.get("turns", 0)) for m in metrics) / n
        avg_acc = sum(float(m.get("acc", 0)) for m in metrics) / n
        return {"avg_turns": avg_turns, "avg_acc": avg_acc}
