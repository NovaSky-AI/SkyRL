from __future__ import annotations
import asyncio
import inspect
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import dspy
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
    reward: Optional[AgentResult] = None
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
# ---------------------------
# Config
# ---------------------------
@dataclass
class TrialConfig:
    """
    Configuration for a single Trial execution.
    """
    dspy_program: dspy.Module
    example: Any
    reward_fn: Optional[Callable[[Any, Any], float]] = None
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
        self.reward_fn = config.reward_fn
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
            # self.program.update_reward(final_reward)

            # 4) Collect trace
            # We need to put the dspy 
            chat_history = self.program.collect_trace(pred, kwargs)
            results.chat_history = chat_history
            results.reward = final_reward
            return results
        except Exception as e:
            results.exception_info = f"{type(e).__name__}: {e}"
            return results
            