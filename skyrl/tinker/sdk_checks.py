"""SDK-only training check stages reusable by service-owning clients."""

import hashlib
import json
import math
import time
from contextlib import contextmanager
from itertools import cycle, islice

import httpx
import tinker
from tinker import types

SEED_TEXTS = (
    "A river flows past a stone bridge. Trees grow along the bank and birds gather in the branches. ",
    "Calculate the area of a rectangle: multiply its length by its width. Explain each arithmetic operation. ",
)


def create_training_client(service, model_path: str, report):
    with measure_phase(report, "create_model") as record:
        trainer = service.create_lora_training_client(base_model=model_path, rank=32, seed=0)
        record["model_id"] = trainer.model_id
    return trainer


def train_batch(trainer, batch, args, report, phase: str) -> None:
    with measure_phase(report, f"{phase}/forward_backward") as record:
        result = trainer.forward_backward(batch, "gspo").result()
        check_training_result(result, args.context, args.batch_size)
        record.update(scored_tokens=args.context * args.batch_size, metrics=result.metrics)


def update_optimizer(trainer, learning_rate: float, report, phase: str) -> None:
    with measure_phase(report, f"{phase}/optimizer") as record:
        result = trainer.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
        norm = result.metrics["skyrl.ai/grad_norm"]
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"expected a finite nonzero gradient norm, got {norm}")
        record["metrics"] = result.metrics


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
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = client.post("api/v1/retrieve_future", json={"request_id": request_id}, timeout=remaining)
            except httpx.ReadTimeout as error:
                raise TimeoutError(f"unload did not finish for {model_id} within its polling budget") from error
            if response.status_code == 408:
                continue
            response.raise_for_status()
            result = response.json()
            if result["type"] != "unload_model" or result["model_id"] != model_id:
                raise RuntimeError(f"unexpected unload result: {result}")
            return
    raise TimeoutError(f"unload did not finish for {model_id}; inspect the server before reusing it")


def prepare_full_context_inputs(trainer, args, report, tokenizer=None) -> list[types.Datum]:
    with measure_phase(report, "prepare_inputs") as record:
        info = trainer.get_info()
        # Older SkyRL revisions omit these optional SDK fields.
        if info.is_lora is False or info.lora_rank not in (None, 32):
            raise ValueError("expected a rank-32 LoRA training client")
        if tokenizer is None:
            tokenizer = trainer.get_tokenizer()
        datums = [
            build_full_context_datum(tokenizer.encode(seed, add_special_tokens=False), args.context)
            for seed in SEED_TEXTS
        ]
        if datums[0].model_input.to_ints() == datums[1].model_input.to_ints():
            raise ValueError("opposite-advantage fixtures must not have identical token inputs")
        datums = [datums[index % len(datums)] for index in range(args.batch_size)]
        fixture = serialize_batch(datums) + "\n"
        (args.output_dir / "datums.json").write_text(fixture)
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


def publish_and_sample(trainer, prompt, report, phase_prefix: str):
    with measure_phase(report, f"{phase_prefix}/publication"):
        sampler = trainer.save_weights_and_get_sampling_client()
    with measure_phase(report, f"{phase_prefix}/sample") as record:
        result = sampler.sample(
            prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=8, temperature=0)
        ).result()
        if len(result.sequences) != 1 or not result.sequences[0].tokens:
            raise ValueError("sampling returned no sequence")
        record["output_tokens"] = result.sequences[0].tokens
    return sampler
