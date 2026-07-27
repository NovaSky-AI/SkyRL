"""Benchmark BF16 and expert-only MXFP8 vLLM serving on DAPO prompts.

Run:
    uv run --isolated --with modal modal run --detach \
        examples/train/megatron/modal_vllm_mxfp8_concurrency_benchmark.py \
        --mode both
"""

from __future__ import annotations

import json
import os
import pathlib

import modal

APP_NAME = "skyrl-vllm-mxfp8-concurrency-benchmark"
MODEL = "Qwen/Qwen3-30B-A3B"
DAPO_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
REMOTE_REPO = "/root/SkyRL"
HF_HOME = "/root/hf-cache"
DATA_ROOT = "/root/data/vllm-mxfp8-concurrency"
RESULT_ROOT = "/root/results/vllm-mxfp8-concurrency"
PORT = 8000
GPU = os.environ.get("MODAL_GPU", "B200:8")


def _repo_root() -> pathlib.Path:
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        candidate = start if start.is_dir() else start.parent
        for path in (candidate, *candidate.parents):
            if (path / "pyproject.toml").exists() and (path / "skyrl").exists():
                return path
    raise RuntimeError("Run the benchmark from a SkyRL checkout")


repo_root = _repo_root()
hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("skyrl-vllm-mxfp8-concurrency-data", create_if_missing=True)
result_volume = modal.Volume.from_name("skyrl-vllm-mxfp8-concurrency-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    .pip_install("huggingface-hub", "datasets", "jinja2>=3.1", "matplotlib", "transformers>=5.6.1,<=5.8.0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_HOME,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "MPLBACKEND": "Agg",
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{REMOTE_REPO}/.venv",
            "VLLM_USE_FLASHINFER_MOE_FP16": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(
        str(repo_root),
        REMOTE_REPO,
        copy=True,
        ignore=[".venv", ".git", "**/__pycache__"],
    )
    .workdir(REMOTE_REPO)
    .run_commands("uv sync --extra megatron", gpu="any")
    .run_commands(f"rm -rf {HF_HOME}")
)

app = modal.App(APP_NAME)


def _run_command(command: list[str]) -> None:
    import subprocess

    subprocess.run(command, cwd=REMOTE_REPO, check=True)


@app.function(
    image=image,
    volumes={HF_HOME: hf_volume, DATA_ROOT: data_volume},
    timeout=2 * 60 * 60,
)
def prepare_assets(max_prompts: int, max_prompt_tokens: int) -> str:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(MODEL)
    hf_volume.commit()

    os.makedirs(DATA_ROOT, exist_ok=True)
    output_path = f"{DATA_ROOT}/dapo-prompts-{max_prompt_tokens}.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dataset = load_dataset(DAPO_DATASET, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)

    prompts: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        messages = row["prompt"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]

        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)

        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(formatted, add_special_tokens=False)
        if len(prompt_tokens) < 4 or len(prompt_tokens) > max_prompt_tokens:
            continue
        prompts.append(formatted)
        if len(prompts) >= max_prompts:
            break

    if len(prompts) < max_prompts:
        raise RuntimeError(f"DAPO produced only {len(prompts)} prompts within {max_prompt_tokens} tokens")

    with open(output_path, "w", encoding="utf-8") as output_file:
        for prompt in prompts:
            output_file.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    data_volume.commit()
    print(f"Prepared {len(prompts)} unique DAPO prompts at {output_path}")
    return output_path


def _server_command(mode: str, max_model_len: int) -> list[str]:
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "--extra",
        "megatron",
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        MODEL,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "8",
        "--data-parallel-backend",
        "mp",
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--gpu-memory-utilization",
        "0.6",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1024",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--trust-remote-code",
    ]
    if mode == "mxfp8":
        command.extend(["--quantization", "online", "--quantization-config", '{"moe":"mxfp8"}'])
    return command


def _wait_for_server(process, log_path: str, timeout_s: int = 1800) -> None:
    import time
    import urllib.request

    deadline = time.monotonic() + timeout_s
    health_url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"vLLM exited with code {process.returncode}\n{log_text[-20_000:]}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s; log={log_path}")


def _bench_command(
    dataset_path: str,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    result_dir: str,
    result_filename: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "--extra",
        "megatron",
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--endpoint",
        "/v1/completions",
        "--model",
        MODEL,
        "--dataset-name",
        "custom",
        "--dataset-path",
        dataset_path,
        "--custom-output-len",
        str(output_len),
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--ignore-eos",
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--logprobs",
        "1",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--seed",
        "0",
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_filename,
    ]


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        HF_HOME: hf_volume,
        DATA_ROOT: data_volume,
        RESULT_ROOT: result_volume,
    },
    timeout=24 * 60 * 60,
)
def benchmark_mode(
    mode: str,
    dataset_path: str,
    concurrencies: list[int],
    output_len: int,
    min_requests: int,
    request_multiplier: int,
    max_prompt_tokens: int,
) -> dict:
    import signal
    import subprocess

    if mode not in {"bf16", "mxfp8"}:
        raise ValueError("mode must be bf16 or mxfp8")

    mode_result_dir = f"{RESULT_ROOT}/{mode}"
    os.makedirs(mode_result_dir, exist_ok=True)
    server_log_path = f"{mode_result_dir}/server.log"
    max_model_len = max_prompt_tokens + output_len

    with open(server_log_path, "w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            _server_command(mode, max_model_len),
            cwd=REMOTE_REPO,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        _wait_for_server(process, server_log_path)

        warmup_concurrency = min(32, max(concurrencies))
        warmup_prompts = max(32, warmup_concurrency)
        with open(dataset_path, encoding="utf-8") as source_file:
            prompt_lines = source_file.readlines()
        prompt_offset = 0
        warmup_dataset_path = f"{mode_result_dir}/warmup-prompts.jsonl"
        with open(warmup_dataset_path, "w", encoding="utf-8") as warmup_file:
            warmup_file.writelines(prompt_lines[prompt_offset : prompt_offset + warmup_prompts])
        prompt_offset += warmup_prompts
        warmup_command = _bench_command(
            warmup_dataset_path,
            min(32, output_len),
            warmup_concurrency,
            warmup_prompts,
            mode_result_dir,
            "warmup.json",
        )
        _run_command(warmup_command)

        points = []
        for concurrency in concurrencies:
            num_prompts = max(min_requests, request_multiplier * concurrency)
            if prompt_offset + num_prompts > len(prompt_lines):
                raise RuntimeError(f"Need {prompt_offset + num_prompts} prompts, found {len(prompt_lines)}")
            result_filename = f"{mode}-c{concurrency}.json"
            point_dataset_path = f"{mode_result_dir}/prompts-c{concurrency}.jsonl"
            with open(point_dataset_path, "w", encoding="utf-8") as point_dataset_file:
                point_dataset_file.writelines(prompt_lines[prompt_offset : prompt_offset + num_prompts])
            prompt_offset += num_prompts
            _run_command(
                _bench_command(
                    point_dataset_path,
                    output_len,
                    concurrency,
                    num_prompts,
                    mode_result_dir,
                    result_filename,
                )
            )
            with open(f"{mode_result_dir}/{result_filename}", encoding="utf-8") as result_file:
                result = json.load(result_file)
            if result["completed"] != num_prompts or result.get("failed", 0):
                raise RuntimeError(
                    f"{mode} concurrency={concurrency}: completed={result['completed']} "
                    f"failed={result.get('failed', 0)} expected={num_prompts}"
                )
            point = {
                "concurrency": concurrency,
                "num_prompts": num_prompts,
                "output_throughput": result["output_throughput"],
                "request_throughput": result["request_throughput"],
                "mean_ttft_ms": result["mean_ttft_ms"],
                "median_ttft_ms": result["median_ttft_ms"],
                "p99_ttft_ms": result["p99_ttft_ms"],
                "mean_tpot_ms": result["mean_tpot_ms"],
                "median_tpot_ms": result["median_tpot_ms"],
                "p99_tpot_ms": result["p99_tpot_ms"],
                "mean_e2el_ms": result["mean_e2el_ms"],
                "p99_e2el_ms": result["p99_e2el_ms"],
            }
            points.append(point)
            print(json.dumps({"mode": mode, **point}, sort_keys=True))

        summary = {
            "mode": mode,
            "model": MODEL,
            "dataset": DAPO_DATASET,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 8,
            "data_parallel_size": 8,
            "output_len": output_len,
            "max_prompt_tokens": max_prompt_tokens,
            "points": points,
        }
        with open(f"{mode_result_dir}/summary.json", "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
        result_volume.commit()
        return summary
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result_volume.commit()


@app.function(
    image=image,
    volumes={RESULT_ROOT: result_volume},
    timeout=30 * 60,
)
def render_results(results: list[dict]) -> dict:
    import csv

    import matplotlib.pyplot as plt

    by_mode = {result["mode"]: result for result in results}
    concurrencies = [point["concurrency"] for point in results[0]["points"]]
    os.makedirs(RESULT_ROOT, exist_ok=True)

    rows = []
    for concurrency in concurrencies:
        row = {"concurrency": concurrency}
        for mode, result in by_mode.items():
            point = next(point for point in result["points"] if point["concurrency"] == concurrency)
            row[f"{mode}_output_throughput"] = point["output_throughput"]
            row[f"{mode}_mean_ttft_ms"] = point["mean_ttft_ms"]
            row[f"{mode}_mean_tpot_ms"] = point["mean_tpot_ms"]
        if "bf16" in by_mode and "mxfp8" in by_mode:
            bf16 = row["bf16_output_throughput"]
            row["gain_percent"] = 100 * (row["mxfp8_output_throughput"] / bf16 - 1)
        rows.append(row)

    csv_path = f"{RESULT_ROOT}/comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(9, 5))
    width = 0.38
    positions = list(range(len(concurrencies)))
    modes = [mode for mode in ("bf16", "mxfp8") if mode in by_mode]
    for mode_index, mode in enumerate(modes):
        offsets = [position + (mode_index - (len(modes) - 1) / 2) * width for position in positions]
        throughputs = [point["output_throughput"] for point in by_mode[mode]["points"]]
        axis.bar(offsets, throughputs, width=width, label=mode.upper())
    axis.set_title("Qwen3-30B-A3B vLLM rollout throughput")
    axis.set_xlabel("Maximum concurrent requests")
    axis.set_ylabel("Output tokens per second")
    axis.set_xticks(positions, [str(value) for value in concurrencies])
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plot_path = f"{RESULT_ROOT}/comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    combined_path = f"{RESULT_ROOT}/comparison.json"
    with open(combined_path, "w", encoding="utf-8") as combined_file:
        json.dump({"results": results, "comparison": rows}, combined_file, indent=2)
    result_volume.commit()
    return {"rows": rows, "csv": csv_path, "plot": plot_path, "json": combined_path}


def _print_comparison(rendered: dict) -> None:
    rows = rendered["rows"]
    has_both = "gain_percent" in rows[0]
    if has_both:
        print("| Concurrent requests | BF16 output tok/s | MXFP8 output tok/s | Gain |")
        print("|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row['concurrency']} | {row['bf16_output_throughput']:.1f} | "
                f"{row['mxfp8_output_throughput']:.1f} | {row['gain_percent']:+.1f}% |"
            )
    else:
        mode = "bf16" if "bf16_output_throughput" in rows[0] else "mxfp8"
        print(f"| Concurrent requests | {mode.upper()} output tok/s |")
        print("|---:|---:|")
        for row in rows:
            print(f"| {row['concurrency']} | {row[f'{mode}_output_throughput']:.1f} |")
    print(json.dumps({key: value for key, value in rendered.items() if key != "rows"}, indent=2))


@app.local_entrypoint()
def main(
    mode: str = "both",
    concurrencies: str = "1,16,32,128,256",
    output_len: int = 512,
    max_prompt_tokens: int = 512,
    min_requests: int = 64,
    request_multiplier: int = 2,
) -> None:
    if mode not in {"bf16", "mxfp8", "both"}:
        raise ValueError("mode must be bf16, mxfp8, or both")
    concurrency_values = [int(value) for value in concurrencies.split(",") if value]
    if not concurrency_values or any(value <= 0 for value in concurrency_values):
        raise ValueError("concurrencies must contain positive integers")
    if output_len <= 0 or max_prompt_tokens <= 0:
        raise ValueError("output_len and max_prompt_tokens must be positive")
    if min_requests <= 0 or request_multiplier <= 0:
        raise ValueError("min_requests and request_multiplier must be positive")

    warmup_prompts = 32
    benchmark_prompts = sum(max(min_requests, request_multiplier * value) for value in concurrency_values)
    dataset_path = prepare_assets.remote(warmup_prompts + benchmark_prompts, max_prompt_tokens)
    modes = ("bf16", "mxfp8") if mode == "both" else (mode,)
    results = [
        benchmark_mode.remote(
            selected_mode,
            dataset_path,
            concurrency_values,
            output_len,
            min_requests,
            request_multiplier,
            max_prompt_tokens,
        )
        for selected_mode in modes
    ]
    rendered = render_results.remote(results)
    _print_comparison(rendered)
