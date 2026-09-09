"""Profile dataset-independent, full-context GSPO through SkyRL's Tinker API."""

import argparse
from contextlib import contextmanager
import hashlib
from itertools import cycle, islice
import json
import math
from pathlib import Path
import time

import httpx
import tinker
from tinker import types

SEED_TEXTS = (
    "A river flows past a stone bridge. Trees grow along the bank and birds gather in the branches. ",
    "Calculate the area of a rectangle: multiply its length by its width. Explain each arithmetic operation. ",
)


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # This example targets the local, unauthenticated SkyRL API only.
    service = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
    with (args.output_dir / "phases.jsonl").open("w") as report:
        with measure_phase(report, "create_model") as record:
            trainer = service.create_lora_training_client(base_model=args.model_path, rank=32, seed=0)
            record["model_id"] = trainer.model_id
        try:
            datums = prepare_full_context_inputs(trainer, args, report)
            prompt = types.ModelInput.from_ints(datums[0].model_input.to_ints()[:128])
            publish_and_sample(trainer, prompt, report, "initial")
            for step in range(args.steps):
                step_phase = "warmup" if step == 0 else f"step_{step}"
                with measure_phase(report, step_phase):
                    batch = score_reference_batch(trainer, datums, args, report, step)
                    with measure_phase(report, f"{step_phase}/forward_backward") as record:
                        result = trainer.forward_backward(batch, "gspo").result()
                        check_training_result(result, args.context, args.batch_size)
                        record["scored_tokens"] = args.context * args.batch_size
                        record["metrics"] = result.metrics
                    with measure_phase(report, f"{step_phase}/optimizer") as record:
                        result = trainer.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result()
                        norm = result.metrics["skyrl.ai/grad_norm"]
                        if not math.isfinite(norm) or norm <= 0:
                            raise ValueError(f"expected a finite nonzero gradient norm, got {norm}")
                        record["metrics"] = result.metrics
                    publish_and_sample(trainer, prompt, report, step_phase)
            with measure_phase(report, "checkpoint") as record:
                record["path"] = trainer.save_state("full-context-final").result().path
        finally:
            with measure_phase(report, "unload"):
                unload_model(args.base_url, trainer.model_id)


def build_full_context_datum(seed_tokens: list[int], context_length: int) -> types.Datum:
    """Repeat seed tokens to fill the context, plus one token for shifted targets."""
    if not seed_tokens or context_length < 2:
        raise ValueError("a nonempty token fixture and context >= 2 are required")
    sequence = list(islice(cycle(seed_tokens), context_length + 1))
    input_tokens = sequence[:-1]
    target_tokens = sequence[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": [1.0] * context_length},
    )


def build_gspo_batch(datums: list[types.Datum], reference) -> list[types.Datum]:
    """Attach frozen old-policy scores and a sequence-constant synthetic advantage."""
    return [
        types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                **datum.loss_fn_inputs,
                "logprobs": output["logprobs"],
                "advantages": [1.0 if index % 2 == 0 else -1.0] * len(datum.model_input.to_ints()),
            },
        )
        for index, (datum, output) in enumerate(zip(datums, reference.loss_fn_outputs, strict=True))
    ]


def serialize_batch(data: list[types.Datum]) -> str:
    return json.dumps(
        [
            {
                "model_input": datum.model_input.model_dump(mode="json"),
                "loss_fn_inputs": {
                    key: {"data": value.data, "dtype": value.dtype, "shape": value.shape}
                    for key, value in datum.loss_fn_inputs.items()
                },
            }
            for datum in data
        ],
        allow_nan=False,
    )


@contextmanager
def measure_phase(report, phase_name: str):
    """Persist phase boundaries even when a request fails."""
    started = time.perf_counter()
    record = {"phase": phase_name, "started_unix": time.time(), "status": "running"}
    report.write(json.dumps(record) + "\n")
    report.flush()
    try:
        yield record
        record["status"] = "completed"
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        raise
    finally:
        if record["status"] == "running":
            record["status"] = "failed"
        record["seconds"] = time.perf_counter() - started
        report.write(json.dumps(record, allow_nan=False) + "\n")
        report.flush()
        print(json.dumps(record, allow_nan=False), flush=True)


def check_training_result(result, context: int, batch_size: int) -> None:
    if len(result.loss_fn_outputs) != batch_size:
        raise ValueError("model pass returned the wrong datum count")
    for output in result.loss_fn_outputs:
        values = output["logprobs"].data
        if len(values) != context or not all(math.isfinite(value) for value in values):
            raise ValueError("model pass must return one finite logprob per scored position")
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


def prepare_full_context_inputs(trainer, args, report) -> list[types.Datum]:
    with measure_phase(report, "prepare_inputs") as record:
        info = trainer.get_info()
        if not info.is_lora or info.lora_rank != 32:
            raise ValueError("expected a rank-32 LoRA training client")
        tokenizer = trainer.get_tokenizer()
        datums = [
            build_full_context_datum(tokenizer.encode(seed, add_special_tokens=False), args.context)
            for seed in SEED_TEXTS
        ]
        if datums[0].model_input.to_ints() == datums[1].model_input.to_ints():
            raise ValueError("opposite-advantage fixtures must not have identical token inputs")
        datums = [datums[index % len(datums)] for index in range(args.batch_size)]
        fixture = serialize_batch(datums)
        (args.output_dir / "datums.json").write_text(fixture + "\n")
        record["input_positions"] = [len(datum.model_input.to_ints()) for datum in datums]
        record["scored_positions"] = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in datums]
        metadata = {
            "model": info.model_dump(mode="json"),
            "tinker_version": tinker.__version__,
            "context": args.context,
            "batch_size": args.batch_size,
            "backwards_per_step": 1,
            "steps": args.steps,
            "warmup_steps": 1,
            "measured_steps": args.steps - 1,
            "publications": args.steps + 1,
            "learning_rate": args.learning_rate,
            "loss_fn": "gspo",
            "loss_fn_config": None,
            "reference_source": "trainer.forward before each optimizer update",
            "advantages": [1.0, -1.0],
            "datums_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
        }
        (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return datums


def score_reference_batch(trainer, datums, args, report, step: int) -> list[types.Datum]:
    phase_prefix = "warmup" if step == 0 else f"step_{step}"
    with measure_phase(report, f"{phase_prefix}/reference") as record:
        # cross_entropy here is a forward-only scoring request, never an update.
        reference = trainer.forward(datums, "cross_entropy").result()
        check_training_result(reference, args.context, args.batch_size)
        batch = build_gspo_batch(datums, reference)
        record["scored_tokens"] = args.context * args.batch_size
    (args.output_dir / f"step_{step}_batch.json").write_text(serialize_batch(batch) + "\n")
    return batch


def publish_and_sample(trainer, prompt, report, phase_prefix: str) -> None:
    with measure_phase(report, f"{phase_prefix}/publication"):
        sampler = trainer.save_weights_and_get_sampling_client()
    with measure_phase(report, f"{phase_prefix}/sample") as record:
        result = sampler.sample(
            prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=8, temperature=0)
        ).result()
        if len(result.sequences) != 1 or not result.sequences[0].tokens:
            raise ValueError("sampling returned no sequence")
        record["output_tokens"] = result.sequences[0].tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for inputs, phase timings and metrics"
    )
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2, help="Total full-context sequences per update")
    parser.add_argument("--steps", type=int, default=3, help="Total updates: one warmup, then measured updates")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.context < 2 or args.batch_size < 2 or args.steps < 2:
        parser.error("context >= 2, batch-size >= 2 and steps >= 2 are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
