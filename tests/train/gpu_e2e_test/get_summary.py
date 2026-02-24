import wandb
import argparse
import operator

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--project_name", type=str, required=True)

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

api = wandb.Api()
# get latest run with the run name
# get all runs in the project in the order of latest to oldest
runs = api.runs(f"{args.project_name}", order="-created_at", per_page=50)
# pages are fetched lazily
matched_run = None
for run in runs:
    if run.name == args.run_name:
        matched_run = run
        break


if matched_run is None:
    raise ValueError(f"Run {args.run_name} not found in project {args.project_name}")

for assertion in args.asserts:
    metric_name, operator_str, threshold = assertion.lstrip().rstrip().split()
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
