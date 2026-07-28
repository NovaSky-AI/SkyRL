from copy import deepcopy

import pytest

from skyrl.benchmarks.bench_fsdp_multi_lora_tinker import (
    build_cases,
    compare_results,
    summarize_samples,
)


def test_build_cases_distinguishes_underfilled_and_saturated_workloads():
    cases = build_cases(
        adapter_counts=[2],
        workloads=["underfilled", "saturated"],
        sequence_length=512,
        max_tokens_per_microbatch=4096,
    )

    assert cases[0].examples_per_adapter == 1
    assert cases[0].tokens_per_step == 1024
    assert cases[1].examples_per_adapter == 8
    assert cases[1].tokens_per_step == 8192


def test_build_cases_rejects_a_token_cap_smaller_than_one_sequence():
    with pytest.raises(ValueError, match="at least sequence-length"):
        build_cases([2], ["saturated"], sequence_length=1024, max_tokens_per_microbatch=512)


def test_summarize_samples_uses_median_latency_for_throughput():
    summary = summarize_samples([4.0, 1.0, 2.0], tokens_per_step=200)

    assert summary["median_seconds"] == 2.0
    assert summary["p95_seconds"] == 4.0
    assert summary["tokens_per_second"] == 100.0


def test_compare_results_reports_grouped_speedup():
    single = {
        "config": {"implementation": "single"},
        "output_path": "single.json",
        "cases": [
            {
                "case": {
                    "workload": "underfilled",
                    "active_adapters": 4,
                    "examples_per_adapter": 1,
                    "sequence_length": 512,
                    "max_tokens_per_microbatch": 4096,
                },
                "total": {"tokens_per_second": 600.0},
                "peak_gpu_memory_mib": 26000,
            }
        ],
    }
    concurrent = deepcopy(single)
    concurrent["config"]["implementation"] = "concurrent"
    concurrent["output_path"] = "concurrent.json"
    concurrent["cases"][0]["total"]["tokens_per_second"] = 1080.0
    concurrent["cases"][0]["peak_gpu_memory_mib"] = 34000

    comparison = compare_results(single, concurrent)

    assert comparison["cases"][0]["speedup"] == pytest.approx(1.8)
    assert comparison["cases"][0]["single_peak_gpu_memory_mib"] == 26000
    assert comparison["cases"][0]["concurrent_peak_gpu_memory_mib"] == 34000
