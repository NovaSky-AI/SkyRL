"""Summarize per-engine counters saved by the offloading generation example."""

import json
from pathlib import Path
import sys

from prometheus_client.parser import text_string_to_metric_families


def counters(text):
    """Sum each sample name across engine-local metric labels."""
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


def main():
    directory = Path(sys.argv[1])
    metadata = json.loads((directory / "result.json").read_text())
    snapshots = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    engines = []
    for index, terminal in enumerate(snapshots[-1]["engines"]):
        initial = counters(snapshots[0]["engines"][index]["metrics"])
        final = counters(terminal["metrics"])
        deltas = {
            name: value - initial.get(name, 0)
            for name, value in final.items()
            if name.endswith(("_total", "_count", "_sum"))
        }
        if any(value < 0 for value in deltas.values()):
            raise ValueError("A counter reset during the evaluation.")
        preemptions = deltas.get("vllm:num_preemptions_total")
        engines.append(
            {
                "url": terminal["url"],
                "preemptions": preemptions,
                "counters": deltas,
                "note": "An absent counter is unavailable; inspect raw scrapes and engine logs before inferring zero.",
            }
        )
    metadata["engines"] = engines
    metadata["preemptions"] = (
        sum(engine["preemptions"] for engine in engines)
        if all(engine["preemptions"] is not None for engine in engines)
        else None
    )
    (directory / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
