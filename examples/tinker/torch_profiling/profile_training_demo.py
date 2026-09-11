"""Demo of torch profiling a running SkyRL Tinker server.

Profiling is driven over plain HTTP rather than through the Tinker SDK, so a
long-lived server can be profiled at any point without a restart. The client
below trains normally and brackets a few optim steps with /start_profiling and
/stop_profiling.

Usage:
    # Terminal 1: start a SkyRL Tinker API server with profiling enabled
    bash examples/tinker/torch_profiling/run_tinker_server.sh

    # Terminal 2
    TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with transformers \\
        python examples/tinker/torch_profiling/profile_training_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import tinker
from tinker import types

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=2, help="Optim steps to run before profiling starts.")
    p.add_argument("--profiled-steps", type=int, default=3, help="Optim steps to run while profiling.")
    p.add_argument("--global-step", type=int, default=0, help="Names the trace directory; see --tag.")
    p.add_argument("--tag", default="demo", help="Appended to the trace directory name.")
    return p.parse_args()


def profiler_request(base_url: str, path: str, payload: dict | None = None) -> dict:
    """Call a profiling endpoint, raising with the server's message on failure."""
    url = f"{base_url.rstrip('/')}{path}"
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        # Long timeout: /stop_profiling blocks until the final window has been
        # written, including its upload when export_dir is a cloud path.
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"{path} failed with HTTP {e.code}: {detail}") from e


def make_datum(tokenizer, prompt: str, completion: str) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


def train_step(client, data, learning_rate: float = 1e-4) -> None:
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(types.AdamParams(learning_rate=learning_rate)).result()


def main() -> None:
    args = parse_args()

    service_client = tinker.ServiceClient(base_url=f"{args.base_url}/", api_key=args.api_key)
    client = service_client.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    tokenizer = client.get_tokenizer()
    data = [make_datum(tokenizer, "Question: 1+1?\nAnswer:", " 2")]

    print(f"status before: {profiler_request(args.base_url, '/profiling_status')}")

    for step in range(args.warmup_steps):
        train_step(client, data)
        print(f"warmup step {step + 1}/{args.warmup_steps} done (not profiled)")

    # The profiler advances one step per optim_step for this model_id only, so
    # the schedule below is expressed in optim steps: record the next
    # `profiled_steps` after a single warmup step, then idle until stopped.
    started = profiler_request(
        args.base_url,
        "/start_profiling",
        {
            "model_id": client.model_id,
            "global_step": args.global_step,
            "export_path_extra": args.tag,
            "schedule_options": {
                "skip_first": 0,
                "wait": 0,
                "warmup": 1,
                "active": args.profiled_steps,
                "repeat": 1,
            },
            "profile_options": {"activities": ["cpu", "cuda"], "with_stack": False},
            # Set to re-run with the same global_step; otherwise a non-empty
            # target directory is rejected so an earlier capture is not lost.
            "overwrite": False,
        },
    )
    print(f"profiling started, traces -> {started['export_path']}")

    # One extra step so the window closes: a trace is written on the transition
    # out of the last active step, not during it.
    for step in range(args.profiled_steps + 2):
        train_step(client, data)
        print(f"profiled step {step + 1}/{args.profiled_steps + 2} done")

    print(f"status during: {profiler_request(args.base_url, '/profiling_status')}")

    profiler_request(args.base_url, "/stop_profiling", {"model_id": client.model_id})
    print(f"profiling stopped, traces written to {started['export_path']}")
    print("open the *.pt.trace.json files in chrome://tracing, Perfetto, or HolisticTraceAnalysis")


if __name__ == "__main__":
    main()
