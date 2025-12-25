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
    # -------- utilities --------
    @staticmethod
    def _example_to_kwargs(example: Any) -> Dict[str, Any]:
        """
        Accepts:
          - dict / Mapping
          - dataclass / pydantic / simple object with attributes
        """
        if example is None:
            return {}
        if isinstance(example, Mapping):
            return dict(example)
        # Try vars() for objects with __dict__
        try:
            d = vars(example)
            if isinstance(d, dict):
                return dict(d)
        except TypeError:
            pass
        # Fallback: grab public attrs (best-effort)
        out: Dict[str, Any] = {}
        for k in dir(example):
            if k.startswith("_"):
                continue
            try:
                v = getattr(example, k)
            except Exception:
                continue
            if callable(v):
                continue
            out[k] = v
        return out
    @staticmethod
    def _get_adapter_and_signature(program: Any):
        """
        Your snippet uses:
          self.adapter.format(signature=self.original_sig, inputs=kwargs, demos=[])
          self.adapter.format_finetune_data(signature=self.original_sig, inputs=kwargs, outputs=pred, demos=[])['messages']
        Here we try to locate:
          - adapter (program.adapter)
          - signature (program.original_sig OR program.signature OR program._signature)
        """
        adapter = getattr(program, "adapter", None)
        if adapter is None:
            raise AttributeError(
                "DSPy program has no `.adapter`. "
                "To use trace collection via adapter.format/format_finetune_data, "
                "attach an adapter to the program (e.g., program.adapter = ...)."
            )
        sig = (
            getattr(program, "original_sig", None)
            or getattr(program, "signature", None)
            or getattr(program, "_signature", None)
        )
        if sig is None:
            raise AttributeError(
                "DSPy program has no `.original_sig` / `.signature` / `._signature`. "
                "Trial needs a signature object for adapter.format."
            )
        return adapter, sig
    @staticmethod
    def _assistant_content_from_messages(all_messages: List[Dict[str, Any]]) -> str:
        if not all_messages:
            return ""
        last = all_messages[-1]
        return str(last.get("content", ""))
    async def _call_program(self, kwargs: Dict[str, Any]) -> Any:
        """
        DSPy modules are typically synchronous. We run them in a thread so Trial.run is async.
        If your program is already async-callable, we await it directly.
        """
        # Preferred: program(**kwargs)
        call = self.program
        if inspect.iscoroutinefunction(getattr(call, "__call__", None)):
            return await call(**kwargs)
        # Some dspy.Module have forward(**kwargs) and __call__ delegates, but we keep it simple:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: call(**kwargs))
    async def _call_verifier(self, pred: Any, kwargs: Dict[str, Any]) -> Optional[VerifierResult]:
        v = self.cfg.verifier
        if v is None:
            return None
        # Support sync or async verifier.
        if inspect.iscoroutinefunction(v):
            out = await v(pred=pred, kwargs=kwargs, example=self.example)
        else:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, lambda: v(pred=pred, kwargs=kwargs, example=self.example))
        # Normalize common return types into VerifierResult
        if isinstance(out, VerifierResult):
            return out
        if isinstance(out, dict):
            # Allow either {"reward": x, ...} or {"rewards": {...}}
            if "rewards" in out and isinstance(out["rewards"], dict):
                rewards = {k: float(v) for k, v in out["rewards"].items()}
                passed = bool(out.get("passed", True))
                details = dict(out.get("details", {})) if isinstance(out.get("details", {}), dict) else {}
                return VerifierResult(rewards=rewards, passed=passed, details=details)
            # Treat it as rewards dict directly
            rewards = {k: float(v) for k, v in out.items() if _is_number(v)}
            passed = bool(out.get("passed", True)) if "passed" in out else True
            return VerifierResult(rewards=rewards, passed=passed, details={})
        if isinstance(out, (int, float)):
            return VerifierResult(rewards={self.cfg.reward_key: float(out)}, passed=True, details={})
        if isinstance(out, bool):
            return VerifierResult(rewards={self.cfg.reward_key: 1.0 if out else 0.0}, passed=out, details={})
        # Unknown shape; mark as failed but keep info
        return VerifierResult(rewards={self.cfg.reward_key: 0.0}, passed=False, details={"raw": repr(out)})
    def _collect_trace(self, kwargs: Dict[str, Any], pred: Any, total_reward: float) -> Dict[str, Any]:
        adapter, sig = self._get_adapter_and_signature(self.program)
        inp_messages = adapter.format(
            signature=sig,
            inputs=kwargs,
            demos=[],  # TODO: Add support for demos
        )
        ft = adapter.format_finetune_data(
            signature=sig,
            inputs=kwargs,
            outputs=pred,
            demos=[],  # TODO: Add support for demos
        )
        all_messages = ft["messages"]
        trace = {
            "messages": inp_messages,
            "completion": {
                "role": "assistant",
                "content": self._assistant_content_from_messages(all_messages),
            },
            "reward": float(total_reward),
        }
        return trace
    # -------- public API --------
    async def run(self) -> TrialResults:
        kwargs = self._example_to_kwargs(self.example)
        results = TrialResults()
        try:
            # 1) Run DSPy program
            pred = await self._call_program(kwargs)
            # 2) Verify (optional)
            verifier_result = await self._call_verifier(pred, kwargs)
            results.verifier_result = verifier_result
            # 3) Decide scalar reward for trace
            total = 0.0
            if verifier_result is not None:
                # Pick reward_key if present, else fall back to sum of rewards.
                if self.cfg.reward_key in verifier_result.rewards:
                    total = float(verifier_result.rewards[self.cfg.reward_key])
                else:
                    total = float(sum(verifier_result.rewards.values()))
            # 4) Collect trace
            trace = self._collect_trace(kwargs=kwargs, pred=pred, total_reward=total)
            self.traces.append(trace)
            results.traces = list(self.traces)
            # 5) Build agent_result metadata: your generator expects agent_result.metadata['all_messages']
            # We mirror adapter.format_finetune_data()['messages'] under 'all_messages'.
            adapter, sig = self._get_adapter_and_signature(self.program)
            all_messages = adapter.format_finetune_data(
                signature=sig,
                inputs=kwargs,
                outputs=pred,
                demos=[],
            )["messages"]
            results.agent_result = AgentResult(
                output=pred,
                metadata={
                    "all_messages": all_messages,
                    "inp_messages": adapter.format(signature=sig, inputs=kwargs, demos=[]),
                    "trace": trace,
                },
            )
            return results
        except Exception as e:
            results.exception_info = f"{type(e).__name__}: {e}"
            return results
            
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)