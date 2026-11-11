import argparse
import operator
import time

import wandb

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
    "<=": operator.le,
    ">=": operator.ge,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--wait_for_finished", action="store_true")
    parser.add_argument(
        "--asserts",
        nargs="+",
        type=str,
        required=True,
        help="Assertions such as 'eval/all/avg_score >= 0.72'.",
    )
    args = parser.parse_args()
    assertions = [assertion.strip().split() for assertion in args.asserts]
    metric_names = [name for name, _, _ in assertions]

    deadline = time.monotonic() + 180
    while True:
        # Api.runs() caches results on the Api instance, so create one per poll.
        runs = wandb.Api().runs(f"{args.project_name}", filters={"display_name": args.run_name}, order="-created_at")
        matched_run = next(iter(runs), None)
        if matched_run is None and not args.wait_for_finished:
            raise ValueError(f"Run {args.run_name} not found in project {args.project_name}")
        if matched_run is not None and not args.wait_for_finished:
            metric_values = matched_run.summary_metrics
            break
        if matched_run is not None and matched_run.state in {"failed", "crashed", "killed"}:
            raise ValueError(f"Run {args.run_name} ended with state {matched_run.state}")

        if matched_run is not None and matched_run.state == "finished":
            step_rows = matched_run.scan_history(keys=["trainer/global_step", "trainer/total_training_steps"])
            final_steps = [
                row["trainer/global_step"]
                for row in step_rows
                if row["trainer/global_step"] >= row["trainer/total_training_steps"]
            ]
            if final_steps:
                final_step = max(final_steps)
                metric_rows = matched_run.scan_history(keys=["trainer/global_step", *metric_names])
                final_rows = [row for row in metric_rows if row["trainer/global_step"] == final_step]
                if final_rows:
                    metric_values = final_rows[-1]
                    break

        if time.monotonic() >= deadline:
            raise ValueError(f"Run {args.run_name} did not publish final training metrics within 180 seconds")
        time.sleep(10)

    for metric_name, operator_str, threshold in assertions:
        threshold = float(threshold)
        if metric_name not in metric_values:
            raise ValueError(f"Metric {metric_name} not found in run {args.run_name}")
        metric_value = metric_values[metric_name]
        if not OPERATORS[operator_str](metric_value, threshold):
            print(f"Metric {metric_name} is not {operator_str} threshold {threshold}: {metric_value}")
            return 1
        print(f"Metric {metric_name} is {operator_str} threshold {threshold}: {metric_value}")
    print("All assertions passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
