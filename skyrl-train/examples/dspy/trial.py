from __future__ import annotations
import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
# ---------------------------
# Result containers
# ---------------------------
@dataclass
class AgentResult:
    """Holds the raw DSPy program output plus any metadata we want to pass downstream."""
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class VerifierResult:
    """Minimal verifier result shape expected by your generator code."""
    rewards: Dict[str, float]
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
@dataclass
class TrialResults:
    """Returned by Trial.run()."""
    agent_result: Optional[AgentResult] = None
    verifier_result: Optional[VerifierResult] = None
    traces: List[Dict[str, Any]] = field(default_factory=list)
    exception_info: Optional[str] = None
# ---------------------------
# Config
# ---------------------------
@dataclass
class TrialConfig:
    """
    dspy_program: a dspy.Module instance (or anything callable like program(**kwargs))
    example: an input example (dict-like or object with attrs)
    verifier: optional callable(pred, kwargs, example) -> VerifierResult|dict|float|bool
    reward_key: if verifier returns dict rewards, we read this key as the scalar reward
    """
    dspy_program: Any
    example: Any
    verifier: Optional[Callable[..., Any]] = None
    reward_key: str = "reward"
# ---------------------------
# Trial
# ---------------------------
class Trial:
    """
    Trial executes a DSPy program on an example and collects a supervised-style trace:
      - formatted input messages (adapter.format)
      - formatted finetune messages (adapter.format_finetune_data)['messages']
      - completion = assistant last message content
      - reward = float(total)
    Trace collection is done here (not inside the dspy program).
    """
    def __init__(self, config: TrialConfig):
        self.cfg = config
        self.program = config.dspy_program
        self.example = config.example
        # We store one (or many) traces from this trial.
        self.traces: List[Dict[str, Any]] = []
        self.reward_fn = None
    # -------- utilities --------
    # -------- public API --------
    async def run(self) -> TrialResults:
        kwargs = self._example_to_kwargs(self.example)
        results = TrialResults()
        try:
            # 1) Run DSPy program
            pred = self.program(kwargs)
            # 2) Verify (optional)
            final_reward = await self.reward_fn(self.example, pred)
            self.program.update_reward(final_reward)

            # 4) Collect trace
            trace = self.program_collect_trace()
            return results
        except Exception as e:
            results.exception_info = f"{type(e).__name__}: {e}"
            return results
            