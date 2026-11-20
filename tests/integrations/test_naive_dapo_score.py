"""Regression test: `naive_dapo.compute_score` must call `math_equal` with valid kwargs."""

import inspect
import os

import pytest

if os.name != "posix":
    pytest.skip("naive_dapo relies on POSIX SIGALRM timeouts", allow_module_level=True)

pytest.importorskip("pylatexenc")
pytest.importorskip("sympy")
pytest.importorskip("skyrl_agent")


def test_pi_branch_passes_valid_kwargs_to_math_equal(monkeypatch):
    """A misspelled kwarg (`tiemout`) raised TypeError, which the bare `except` turned into
    reward 0.0 for every answer containing `\\pi` that the first-stage grader rejected."""
    from skyrl_agent.tasks.verifiers import naive_dapo
    from skyrl_agent.tasks.verifiers.prime_math.grader import math_equal as real_math_equal

    real_signature = inspect.signature(real_math_equal)
    pis_seen = []

    def checked_math_equal(*args, **kwargs):
        real_signature.bind(*args, **kwargs)  # raises TypeError on an unknown kwarg
        pis_seen.append(kwargs["pi"])
        return True

    monkeypatch.setattr(naive_dapo, "grade_answer", lambda pred, gt: (False, pred))
    monkeypatch.setattr(naive_dapo, "math_equal", checked_math_equal)

    result = naive_dapo.compute_score("Answer: \\pi", "\\pi", {})

    assert result["score"] == 1.0
    assert pis_seen, "math_equal was never reached in the \\pi branch"
