"""Smoke test for the patched FleetTaskEnv reward-gating logic.

Runs WITHOUT a real Fleet env: we duplicate the small `_apply_taste_reward`
helper that the diff installs on FleetTaskEnv (lifted verbatim from the diff
body) and exercise it against stubbed `score_trajectory_async` callables.

Reward shape under test:
    effective_taste = max(taste_floor, taste_score)   (1.0 on judge fail/None)
    reward          = verifier_reward * effective_taste

Cases:
  (a) success + pretty taste (v=1, t=1.0, floor=0.1) -> R=1.0
  (b) success + mid taste    (v=1, t=0.5, floor=0.1) -> R=0.5
  (c) success + ugly taste   (v=1, t=0.0, floor=0.1) -> R=0.1 (floor)
  (d) failure + pretty taste (v=0, t=1.0)            -> R=0.0 (gated to 0)
  (e) failure + ugly taste   (v=0, t=0.0)            -> R=0.0
  (f) judge timeout + success                         -> R=verifier (1.0)
  (g) judge exception + success                       -> R=verifier (1.0)
  (h) SKYRL_TASTE_DISABLED=1 + success                -> R=verifier (1.0)

Prints PASS/FAIL per test. Exits 0 if all pass, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Reproduce the helper installed by env.py.diff. Keep this in sync with the
# diff body (search for "async def _apply_taste_reward" in env.py.diff).
# -----------------------------------------------------------------------------

logger = logging.getLogger("smoke_test")


class _FakeFleetTaskEnv:
    """Minimal stand-in for FleetTaskEnv with the attributes the helper reads."""

    def __init__(self, floor: float, timeout_s: float, judge_async):
        self.taste_floor = floor
        self.taste_judge_timeout_s = timeout_s
        self.task_key = "smoke-task-1"
        self.task_config = {"prompt": "Send an email to bob@example.com saying hi"}
        self.chat_history = [
            {"role": "system", "content": "you are a CU agent"},
            {"role": "user", "content": "Send an email..."},
            {"role": "assistant", "content": "I will click Compose."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Now I type the address."},
        ]
        self.last_verifier_reward: Optional[float] = None
        self.last_taste_reward: Optional[float] = None
        self.last_effective_taste: Optional[float] = None
        self.last_taste_judge_failed: bool = False
        # Inject the stubbed judge in place of the real package.
        self._judge_async = judge_async

    async def _apply_taste_reward(self, verifier_reward: float, episode_done: bool) -> float:
        # Body lifted from env.py.diff (kept tight).
        if not episode_done:
            return verifier_reward

        self.last_verifier_reward = float(verifier_reward)
        self.last_taste_reward = None
        self.last_effective_taste = None
        self.last_taste_judge_failed = False

        score_trajectory_async = self._judge_async

        actions = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in self.chat_history
            if m.get("role") == "assistant"
        ]
        task_text = self.task_config.get("prompt", "")
        outcome = bool(self.last_verifier_reward >= 1.0)

        taste_score: Optional[float]
        try:
            taste_score = await asyncio.wait_for(
                score_trajectory_async(task_text, actions, outcome),
                timeout=self.taste_judge_timeout_s,
            )
        except asyncio.TimeoutError:
            self.last_taste_judge_failed = True
            taste_score = None
        except Exception:
            self.last_taste_judge_failed = True
            taste_score = None

        if taste_score is None:
            self.last_effective_taste = 1.0
            return verifier_reward

        taste_score = max(0.0, min(1.0, float(taste_score)))
        self.last_taste_reward = taste_score
        effective_taste = max(self.taste_floor, taste_score)
        self.last_effective_taste = effective_taste
        return verifier_reward * effective_taste


# -----------------------------------------------------------------------------
# Stubbed judges
# -----------------------------------------------------------------------------


def _judge_returning(value: float):
    async def _inner(task: str, actions, outcome: bool) -> float:
        return value
    return _inner


async def _judge_returns_none_if_disabled(task: str, actions, outcome: bool) -> Optional[float]:
    # Mimics the SKYRL_TASTE_DISABLED=1 short-circuit in skyrl_gym.taste.
    if os.environ.get("SKYRL_TASTE_DISABLED") == "1":
        return None
    return 1.0


async def _judge_slow(task: str, actions, outcome: bool) -> float:
    await asyncio.sleep(5.0)
    return 0.9


async def _judge_raises(task: str, actions, outcome: bool) -> float:
    raise RuntimeError("simulated API outage")


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


def _ok(name: str) -> None:
    print(f"PASS: {name}")


def _fail(name: str, msg: str) -> None:
    print(f"FAIL: {name} -> {msg}")


async def _check(name: str, env: _FakeFleetTaskEnv, verifier: float, expected: float,
                 *, expect_failed: bool = False) -> int:
    r = await env._apply_taste_reward(verifier_reward=verifier, episode_done=True)
    ok = abs(r - expected) < 1e-9 and env.last_taste_judge_failed is expect_failed
    if ok:
        _ok(name)
        return 0
    _fail(name, f"r={r} expected={expected} failed={env.last_taste_judge_failed} "
                f"verifier={env.last_verifier_reward} taste={env.last_taste_reward} "
                f"effective={env.last_effective_taste}")
    return 1


async def run() -> int:
    failures = 0
    floor = 0.1

    # (a) success + pretty taste -> 1.0
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_returning(1.0))
    failures += await _check("a_success_pretty_v1_t1_floor0.1_R1.0", env, 1.0, 1.0)

    # (b) success + mid taste -> 0.5
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_returning(0.5))
    failures += await _check("b_success_mid_v1_t0.5_floor0.1_R0.5", env, 1.0, 0.5)

    # (c) success + ugly taste -> floor (0.1)
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_returning(0.0))
    failures += await _check("c_success_ugly_v1_t0_floor0.1_R0.1", env, 1.0, 0.1)

    # (d) failure + pretty taste -> 0.0 (the hack is closed)
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_returning(1.0))
    failures += await _check("d_failure_pretty_v0_t1_R0.0_HACK_CLOSED", env, 0.0, 0.0)

    # (e) failure + ugly taste -> 0.0
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_returning(0.0))
    failures += await _check("e_failure_ugly_v0_t0_R0.0", env, 0.0, 0.0)

    # (f) judge timeout + success -> verifier (1.0), failed=True
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=0.05, judge_async=_judge_slow)
    failures += await _check("f_timeout_success_R_eq_verifier_1.0", env, 1.0, 1.0,
                             expect_failed=True)

    # (g) judge exception + success -> verifier (1.0), failed=True
    env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0, judge_async=_judge_raises)
    failures += await _check("g_exception_success_R_eq_verifier_1.0", env, 1.0, 1.0,
                             expect_failed=True)

    # (h) SKYRL_TASTE_DISABLED=1 + success -> verifier (1.0), failed=False
    os.environ["SKYRL_TASTE_DISABLED"] = "1"
    try:
        env = _FakeFleetTaskEnv(floor=floor, timeout_s=2.0,
                                judge_async=_judge_returns_none_if_disabled)
        failures += await _check("h_disabled_env_var_R_eq_verifier_1.0", env, 1.0, 1.0,
                                 expect_failed=False)
        # Extra invariant: effective_taste should be 1.0 in the disabled path.
        if env.last_effective_taste != 1.0:
            _fail("h_disabled_env_var_R_eq_verifier_1.0",
                  f"effective_taste={env.last_effective_taste} expected 1.0")
            failures += 1
    finally:
        del os.environ["SKYRL_TASTE_DISABLED"]

    return failures


if __name__ == "__main__":
    failures = asyncio.run(run())
    if failures == 0:
        print("\nALL SMOKE TESTS PASSED (8/8)")
        sys.exit(0)
    else:
        print(f"\n{failures} TEST(S) FAILED")
        sys.exit(1)
