"""Time a fixed-input LoRA training loop against a native SkyRL Tinker server."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import httpx
import tinker
from tinker import types


def build_datum(tokens: list[int], context: int) -> types.Datum:
    """Repeat a token fixture into exactly `context` input/target positions."""
    if not tokens or context < 2:
        raise ValueError("a nonempty token fixture and context >= 2 are required")
    sequence = (tokens * ((context + 1 + len(tokens) - 1) // len(tokens)))[: context + 1]
    return types.Datum(
        model_input=types.ModelInput.from_ints(sequence[:-1]),
        loss_fn_inputs={"target_tokens": sequence[1:], "weights": [1.0] * context},
    )


@contextmanager
def measure(report, name: str):
    """Persist phase boundaries even when a request fails."""
    started = time.perf_counter()
    record = {"phase": name, "started_unix": time.time(), "status": "running"}
    report.write(json.dumps(record) + "\n")
    report.flush()
    try:
        yield record
        record["status"] = "completed"
    finally:
        if record["status"] == "running":
            record["status"] = "failed"
        record["seconds"] = time.perf_counter() - started
        report.write(json.dumps(record, allow_nan=False) + "\n")
        report.flush()
        print(json.dumps(record, allow_nan=False), flush=True)


def check_training_result(result, context: int, batch_size: int) -> None:
    if len(result.loss_fn_outputs) != batch_size:
        raise ValueError("forward/backward returned the wrong datum count")
    for output in result.loss_fn_outputs:
        values = output["logprobs"].data
        if len(values) != context or not all(math.isfinite(value) for value in values):
            raise ValueError("forward/backward must return one finite logprob per scored position")
    if not all(math.isfinite(value) for value in result.metrics.values()):
        raise ValueError("non-finite training metric")


def unload_model(base_url: str, model_id: str) -> None:
    """Use SkyRL's HTTP unload endpoint; the public SDK has no unload method."""
    deadline = time.monotonic() + 120
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=35) as client:
        response = client.post("api/v1/unload_model", json={"model_id": model_id})
        response.raise_for_status()
        request_id = response.json()["request_id"]
        while time.monotonic() < deadline:
            response = client.post("api/v1/retrieve_future", json={"request_id": request_id})
            if response.status_code == 408:
                continue
            response.raise_for_status()
            result = response.json()
            if result["type"] != "unload_model" or result["model_id"] != model_id:
                raise RuntimeError(f"unexpected unload result: {result}")
            return
    raise TimeoutError(f"unload did not finish for {model_id}; inspect the server before reusing it")


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # This example targets the local, unauthenticated SkyRL API only.
    service = tinker.ServiceClient(base_url=args.base_url, api_key="skyrl-local")
    with (args.output_dir / "phases.jsonl").open("w") as report:
        with measure(report, "create_model") as record:
            trainer = service.create_lora_training_client(base_model=args.model_path, rank=32, seed=0)
            record["model_id"] = trainer.model_id
        try:
            info = trainer.get_info()
            if not info.is_lora or info.lora_rank != 32:
                raise ValueError("expected a rank-32 LoRA training client")
            tokenizer = trainer.get_tokenizer()
            tokens = tokenizer.encode(args.text_file.read_text(), add_special_tokens=False)
            datum = build_datum(tokens, args.context)
            fixture = json.dumps(
                {
                    "model_input": datum.model_input.model_dump(mode="json"),
                    "loss_fn_inputs": {
                        key: {"data": value.data, "dtype": value.dtype, "shape": value.shape}
                        for key, value in datum.loss_fn_inputs.items()
                    },
                }
            )
            (args.output_dir / "datum.json").write_text(fixture + "\n")
            (args.output_dir / "run.json").write_text(
                json.dumps(
                    {
                        "model": info.model_dump(mode="json"),
                        "tinker_version": tinker.__version__,
                        "context": args.context,
                        "batch_size": args.batch_size,
                        "backwards_per_step": 2,
                        "steps": args.steps,
                        "learning_rate": args.learning_rate,
                        "loss_fn": "cross_entropy",
                        "datum_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
            data = [datum] * args.batch_size
            prompt = types.ModelInput.from_ints(tokens[:128])
            for step in range(args.steps):
                for backward in range(2):
                    with measure(report, f"step_{step}/backward_{backward}") as record:
                        result = trainer.forward_backward(data, "cross_entropy").result()
                        check_training_result(result, args.context, args.batch_size)
                        record["scored_tokens"] = args.context * args.batch_size
                        record["metrics"] = result.metrics
                with measure(report, f"step_{step}/optimizer") as record:
                    result = trainer.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result()
                    norm = result.metrics["skyrl.ai/grad_norm"]
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError(f"expected a finite nonzero gradient norm, got {norm}")
                    record["metrics"] = result.metrics
                with measure(report, f"step_{step}/publication"):
                    sampler = trainer.save_weights_and_get_sampling_client()
                with measure(report, f"step_{step}/sample") as record:
                    result = sampler.sample(
                        prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=8, temperature=0)
                    ).result()
                    if len(result.sequences) != 1 or not result.sequences[0].tokens:
                        raise ValueError("sampling returned no sequence")
                    record["output_tokens"] = result.sequences[0].tokens
            with measure(report, "checkpoint") as record:
                record["path"] = trainer.save_state("full-context-final").result().path
        finally:
            with measure(report, "unload"):
                unload_model(args.base_url, trainer.model_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--text-file", type=Path, required=True, help="Fixed text repeated to fill the training context"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for inputs, phase timings and metrics"
    )
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.context < 2 or args.batch_size < 1 or args.steps < 2:
        parser.error("context >= 2, batch-size >= 1 and steps >= 2 are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    if not args.text_file.is_file():
        parser.error("text-file must exist")
    run(args)


if __name__ == "__main__":
    main()
