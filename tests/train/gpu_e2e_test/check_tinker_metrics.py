import argparse
import json
import operator
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--metrics-file", type=str, required=True)
parser.add_argument(
    "--asserts",
    nargs="+",
    type=str,
    required=True,
    help="Assertions on the max value across all logged steps. Format: <metric_name> <operator> <threshold>. Example: --asserts reward/total >= 0.1",
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

max_values: dict[str, float] = {}
with open(args.metrics_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for key, value in record.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            value = float(value)
            if key not in max_values or value > max_values[key]:
                max_values[key] = value

failed = False
for assertion in args.asserts:
    metric_name, operator_str, threshold = assertion.strip().split()
    threshold = float(threshold)
    if metric_name not in max_values:
        print(f"Metric {metric_name} not found in {args.metrics_file}")
        failed = True
        continue
    metric_value = max_values[metric_name]
    if not OPERATORS[operator_str](metric_value, threshold):
        print(f"Metric {metric_name} (max={metric_value}) is not {operator_str} {threshold}")
        failed = True
    else:
        print(f"Metric {metric_name} (max={metric_value}) is {operator_str} {threshold}")

if failed:
    sys.exit(1)
print("All assertions passed!")
