"""Grade a SWE-bench eval log the swebench way. Exits 0 when the instance is resolved.

Runs inside the task's sandbox, from the task's ``tests/test.sh``:

    grade.py <instance config.json> <eval log> <report.json>

A log whose test run swebench can't find is unresolved. Repos swebench grades fail-only (a test passes
unless the log says it failed) are refused: there a test run that never happened would score as solved.
"""

import json
import sys

from swebench.harness.constants import (
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
from swebench.harness.test_spec.test_spec import make_test_spec


def main() -> int:
    config, log, out = sys.argv[1:4]
    with open(config) as f:
        row = json.load(f)
    spec = make_test_spec(row, namespace="swebench")
    report = {"instance_id": spec.instance_id, "resolved": False}
    if spec.repo in FAIL_ONLY_REPOS:
        report["error"] = f"{spec.repo} is graded fail-only, which can't tell a crashed test run from a pass"
    else:
        status, found = get_logs_eval(spec, log)
        report["tests_found"] = found
        if found:
            gold = {KEY_INSTANCE_ID: spec.instance_id, FAIL_TO_PASS: spec.FAIL_TO_PASS, PASS_TO_PASS: spec.PASS_TO_PASS}
            tests = get_eval_tests_report(status, gold, eval_type=EvalType.PASS_AND_FAIL)
            report["tests"] = tests
            report["resolved"] = get_resolution_status(tests) == ResolvedStatus.FULL.value
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("RESOLVED" if report["resolved"] else f"UNRESOLVED {report.get('error', '')}")
    return 0 if report["resolved"] else 1


if __name__ == "__main__":
    sys.exit(main())
