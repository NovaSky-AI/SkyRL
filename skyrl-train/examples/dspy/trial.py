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
    reward_fn: Optional[Callable[[Any, Any], float]] = None
    lm: Optional[Any] = None  # DSPy LM object
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
        
        # Set LM on the program if provided
        if config.lm is not None:
            if hasattr(self.program, "set_lm"):
                self.program.set_lm(config.lm)
            else:
                # If the program doesn't have set_lm, try setting on sub-programs
                # This handles cases like NaiveCodeGenerator_dspy which has stdin_prog and functional_prog
                if hasattr(self.program, "stdin_prog"):
                    self.program.stdin_prog.set_lm(config.lm)
                if hasattr(self.program, "functional_prog"):
                    self.program.functional_prog.set_lm(config.lm)
        
        self.example = config.example
        self.reward_fn = config.reward_fn
    # -------- utilities --------
    def _example_to_kwargs(self, example: Any) -> Dict[str, Any]:
        """
        Convert an example to keyword arguments for the DSPy program.
        
        Handles multiple example formats:
        - ConversationType (list of messages): extracts prompt from first user message
        - dict: extracts 'prompt' and 'is_stdin' keys
        - dspy.Example: extracts 'prompt' and 'is_stdin' attributes
        - Other: attempts to access as attributes or dict keys
        
        Returns:
            Dict with 'prompt' (str) and 'is_stdin' (bool) keys.
        """
        kwargs = {}
        
        # Handle ConversationType (list of messages)
        if isinstance(example, list) and len(example) > 0:
            # Extract prompt from first user message
            first_message = example[0]
            if isinstance(first_message, dict) and "content" in first_message:
                kwargs["prompt"] = first_message["content"]
            else:
                raise ValueError(f"Expected first message to be a dict with 'content' key, got: {first_message}")
            # Default is_stdin to False for ConversationType (can't extract from messages)
            kwargs["is_stdin"] = False
        
        # Handle dict
        elif isinstance(example, dict):
            if "prompt" in example:
                kwargs["prompt"] = example["prompt"]
            else:
                raise ValueError(f"Example dict must have 'prompt' key, got keys: {list(example.keys())}")
            
            # Get is_stdin from dict or infer from tests
            if "is_stdin" in example:
                kwargs["is_stdin"] = example["is_stdin"]
            elif "tests" in example:
                # Import here to avoid circular imports
                from .dataset import _has_test_type
                kwargs["is_stdin"] = _has_test_type(example["tests"], "stdin")
            else:
                kwargs["is_stdin"] = False
        
        # Handle dspy.Example
        elif hasattr(example, "prompt"):
            kwargs["prompt"] = example.prompt
            
            # Get is_stdin from attribute or infer from tests
            if hasattr(example, "is_stdin"):
                kwargs["is_stdin"] = example.is_stdin
            elif hasattr(example, "tests"):
                from .dataset import _has_test_type
                kwargs["is_stdin"] = _has_test_type(example.tests, "stdin")
            else:
                kwargs["is_stdin"] = False
        
        # Fallback: try to access as attributes or dict-like
        else:
            # Try to get prompt
            if hasattr(example, "prompt"):
                kwargs["prompt"] = example.prompt
            elif isinstance(example, Mapping) and "prompt" in example:
                kwargs["prompt"] = example["prompt"]
            else:
                raise ValueError(f"Cannot extract 'prompt' from example of type {type(example)}")
            
            # Try to get is_stdin
            if hasattr(example, "is_stdin"):
                kwargs["is_stdin"] = example.is_stdin
            elif isinstance(example, Mapping) and "is_stdin" in example:
                kwargs["is_stdin"] = example["is_stdin"]
            elif hasattr(example, "tests"):
                from .dataset import _has_test_type
                kwargs["is_stdin"] = _has_test_type(example.tests, "stdin")
            elif isinstance(example, Mapping) and "tests" in example:
                from .dataset import _has_test_type
                kwargs["is_stdin"] = _has_test_type(example["tests"], "stdin")
            else:
                kwargs["is_stdin"] = False
        
        return kwargs
    # -------- public API --------
    async def run(self) -> TrialResults:
        kwargs = self._example_to_kwargs(self.example)
        results = TrialResults()
        try:
            # import pdb; pdb.set_trace()
            # 1) Run DSPy program
            pred = self.program(**kwargs)
            
            # 2) Verify (optional)
            if self.reward_fn is not None:
                # Prepare input for reward_fn (it expects a dict with prompt, task_id, is_stdin)
                reward_input = kwargs.copy()
                # Check if reward_fn is async
                if asyncio.iscoroutinefunction(self.reward_fn):
                    final_reward = await self.reward_fn(reward_input, pred)
                else:
                    final_reward = self.reward_fn(reward_input, pred)
                
                # Store reward as AgentResult
                results.reward = AgentResult(
                    output=final_reward,
                    metadata={"reward_value": final_reward}
                )
                
                # Update program reward if method exists
                if hasattr(self.program, "update_reward"):
                    self.program.update_reward(final_reward)

            # 3) Collect trace
            if hasattr(self.program, "collect_trace"):
                trace = self.program.collect_trace(kwargs, pred)
                if trace:
                    results.traces = trace if isinstance(trace, list) else [trace]
            
            return results
        except Exception as e:
            results.exception_info = f"{type(e).__name__}: {e}"
            return results
            