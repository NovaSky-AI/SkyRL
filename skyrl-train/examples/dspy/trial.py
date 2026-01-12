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

class DSPyTimeoutError(asyncio.TimeoutError):
    pass

class VerifierTimeoutError(asyncio.TimeoutError):
    pass

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
    traces: List[Dict[str, Any]] = field(default_factory=list)
    exception_info: Optional[str] = None
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
    final_reward_fn: Optional[Callable[[Any, Any], float]] = None
    local_reward_fn: Optional[Callable[[Any, Any], float]] = None
    alpha: float = 0
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
        # Instantiate the program if it's a class, otherwise use it directly
        if inspect.isclass(config.dspy_program):
            self.program = config.dspy_program()
        else:
            self.program = config.dspy_program
        
        self.example = config.example
        self.final_reward_fn = config.final_reward_fn
        self.local_reward_fn = config.local_reward_fn
        self.alpha = config.alpha

    # -------- public API --------
    async def run(self) -> TrialResults:
        
        # kwargs = self._example_to_kwargs(self.example)
        results = TrialResults()
        try:
            inference_timeout_seconds = 600
            verification_timeout_seconds = 60
            assert inspect.iscoroutinefunction(self.program.forward), "program.forward must be a coroutine function"
            try:
                pred = await asyncio.wait_for(self.program.forward(self.example), timeout=inference_timeout_seconds)
            except asyncio.TimeoutError as e:
                raise DSPyTimeoutError(f"TimeoutError: program.forward exceeded {inference_timeout_seconds} seconds")

            final_reward = 0
            # 2) Verify (optional)
            if self.final_reward_fn is not None:
                # Prepare input for reward_fn (it expects a dict with prompt, task_id, is_stdin)
                # reward_input = kwargs.copy()
                # Check if reward_fn is async
                
                assert inspect.iscoroutinefunction(self.final_reward_fn), "final_reward_fn must be a coroutine function"

                try:
                    print(f"[Trial] running final_reward_fn for task_id: {self.example.get('task_id')}")
                    final_reward = await asyncio.wait_for(self.final_reward_fn(self.example, pred), timeout=verification_timeout_seconds)
                    print(f"[Trial] final_reward_fn finished for task_id: {self.example.get('task_id')}. Got {final_reward}")
                except asyncio.TimeoutError as e:
                    raise VerifierTimeoutError(f"TimeoutError: final_reward_fn exceeded {verification_timeout_seconds} seconds")
                # Update program reward if method exists
                # if hasattr(self.program, "update_reward"):
                #     self.program.update_reward(final_reward)

            # 3) Collect trace
            trace, trace_pred = self.program.collect_trace(self.example, pred)
            
            assert inspect.iscoroutinefunction(self.local_reward_fn), "local_reward_fn must be a coroutine function"
            
            if self.alpha == 0:
                local_reward = 0
            else:
                try:
                    print(f"[Trial] running local_reward_fn for task_id: {self.example.get('task_id')}")
                    local_reward = await asyncio.wait_for(self.local_reward_fn(self.example, trace_pred), timeout=verification_timeout_seconds) if self.local_reward_fn is not None else 0
                    print(f"[Trial] local_reward_fn finished for task_id: {self.example.get('task_id')}. Got {local_reward}")
                except asyncio.TimeoutError as e:
                    raise VerifierTimeoutError(f"TimeoutError: local_reward_fn exceeded {verification_timeout_seconds} seconds")
            
            # For handling the case where there is only one module in the program.
            reward = (1 - self.alpha) * final_reward + self.alpha * local_reward
            results.reward = AgentResult(
                output=reward,
                metadata={"reward_value": reward}
            )
            
            if trace:
                results.traces = trace if isinstance(trace, list) else [trace]
            
            return results
        except Exception as e:
            results.exception_info = f"{type(e).__name__}: {e}"
            return results
            