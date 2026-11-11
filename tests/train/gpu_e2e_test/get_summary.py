import argparse
import operator
import sys
import time

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--project_name", type=str, required=True)
parser.add_argument("--wait_for_finished", action="store_true")

parser.add_argument(
    "--asserts",
    nargs="+",
    type=str,
    required=True,
    help="List of assertions to make on summary metrics. Format;a <metric_name> <operator> <threshold>. Example: --asserts eval/all/avg_score >= 0.72 generate/max_num_tokens <= 1000",
)

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
    "<=": operator.le,
    ">=": operator.ge,
}

args = parser.parse_args()
assertions = [assertion.strip().split() for assertion in args.asserts]
metric_names = [metric_name for metric_name, _, _ in assertions]

# W&B's Api.runs() caches results on the Api instance. Query with a fresh
# instance while the run's summary is still syncing.
deadline = time.monotonic() + 180
while True:
    api = wandb.Api()
    runs = api.runs(f"{args.project_name}", filters={"display_name": args.run_name}, order="-created_at")
    matched_run = next(iter(runs), None)
    if matched_run is None and not args.wait_for_finished:
        raise ValueError(f"Run {args.run_name} not found in project {args.project_name}")
    if matched_run is not None and not args.wait_for_finished:
        break
    if matched_run is not None and matched_run.state in {"failed", "crashed", "killed"}:
        raise ValueError(f"Run {args.run_name} ended with state {matched_run.state}")
    if (
        matched_run is not None
        and matched_run.state == "finished"
        and all(name in matched_run.summary_metrics for name in metric_names)
    ):
        break
    if time.monotonic() >= deadline:
        raise ValueError(f"Run {args.run_name} did not finish with all requested summary metrics within 180 seconds")
    time.sleep(10)

for metric_name, operator_str, threshold in assertions:
    threshold = float(threshold)
    if metric_name not in matched_run.summary_metrics:
        raise ValueError(f"Metric {metric_name} not found in run {args.run_name}")

    metric_value = matched_run.summary_metrics[metric_name]
    if not OPERATORS[operator_str](metric_value, threshold):
        print(f"Metric {metric_name} is not {operator_str} threshold {threshold}: {metric_value}")
        sys.exit(1)
    else:
        print(f"Metric {metric_name} is {operator_str} threshold {threshold}: {metric_value}")
print("All assertions passed!")
