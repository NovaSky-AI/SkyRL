"""Direct Tinker smoke client for SkyRL's AMD Tinker server example.

Run this in a separate shell from this directory after starting
``run_tinker_server_amd.sh``. The script intentionally avoids task frameworks
and exercises the public Tinker SDK directly against SkyRL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tinker
from tinker import types


DEFAULT_BASE_URL = "http://localhost:9000"
DEFAULT_API_KEY = "tml-dummy"
SMOKE_LORA_RANK = 32
SMOKE_STEPS = 4
SMOKE_LEARNING_RATE = 3.0e-5
SMOKE_SAMPLE_TOKENS = 32
SMOKE_TEMPERATURE = 0.0

SMOKE_EXAMPLES = [
    ("Reply with only this word: amber", "amber"),
    ("Reply with only this word: basalt", "basalt"),
    ("Reply with only this word: cobalt", "cobalt"),
    ("Reply with only this word: delta", "delta"),
    ("Reply with only this word: ember", "ember"),
    ("Reply with only this word: forest", "forest"),
    ("Reply with only this word: granite", "granite"),
    ("Reply with only this word: harbor", "harbor"),
    ("Reply with only this word: indigo", "indigo"),
    ("Reply with only this word: jasmine", "jasmine"),
    ("Reply with only this word: kernel", "kernel"),
    ("Reply with only this word: lantern", "lantern"),
    ("Reply with only this word: marble", "marble"),
    ("Reply with only this word: nickel", "nickel"),
    ("Reply with only this word: orchid", "orchid"),
    ("Reply with only this word: prism", "prism"),
]


@dataclass(frozen=True)
class EncodedDatum:
    prompt: str
    target: str
    prompt_tokens: list[int]
    target_tokens: list[int]
    datum: types.Datum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed AMD Tinker LoRA smoke test against a SkyRL server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY))
    parser.add_argument(
        "--base-model",
        default=os.environ.get("TINKER_HELLO_BASE_MODEL"),
        help="Base model to request. Defaults to the first model advertised by the server.",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def fetch_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(url, headers={"accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_health(base_url: str) -> None:
    health_url = f"{base_url}/api/v1/healthz"
    try:
        payload = fetch_json(health_url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SkyRL Tinker health check failed at {health_url}: {exc}") from exc

    if payload.get("status") != "ok":
        raise RuntimeError(f"Unexpected health response from {health_url}: {payload}")
    print(f"[1/7] health ok ({health_url})", flush=True)


def discover_base_model(client: tinker.ServiceClient, requested: str | None) -> str:
    capabilities = client.get_server_capabilities()
    supported = [model.model_name for model in capabilities.supported_models if model.model_name]

    if requested:
        if supported and requested not in supported:
            print(
                f"      warning: requested {requested!r}, but server advertised {supported!r}",
                flush=True,
            )
        base_model = requested
    else:
        if not supported:
            raise RuntimeError("Server did not advertise any supported base models")
        base_model = supported[0]

    print(f"[2/7] model {base_model}", flush=True)
    return base_model


def encode_prompt(tokenizer: Any, user_message: str) -> list[int]:
    messages = [{"role": "user", "content": user_message}]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            tokens = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
            return list(tokens)
        except Exception as exc:
            print(f"      warning: chat template failed, falling back to plain text: {exc}", flush=True)

    return list(tokenizer.encode(f"User: {user_message}\nAssistant:", add_special_tokens=False))


def encode_text(tokenizer: Any, text: str) -> list[int]:
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if not tokens:
        raise ValueError(f"text produced no tokens: {text!r}")
    return tokens


def tensor_i64(values: list[int]) -> types.TensorData:
    return types.TensorData(data=values, dtype="int64", shape=[len(values)])


def tensor_f32(values: list[float]) -> types.TensorData:
    return types.TensorData(data=values, dtype="float32", shape=[len(values)])


def make_cross_entropy_datum(prompt_tokens: list[int], target_tokens: list[int]) -> types.Datum:
    model_input_tokens = prompt_tokens + target_tokens[:-1]
    return types.Datum(
        model_input=types.ModelInput.from_ints(model_input_tokens),
        loss_fn_inputs={
            "target_tokens": tensor_i64(target_tokens),
            "weights": tensor_f32([1.0] * len(target_tokens)),
        },
    )


def build_datums(tokenizer: Any) -> list[EncodedDatum]:
    encoded = []
    for prompt, target in SMOKE_EXAMPLES:
        prompt_tokens = encode_prompt(tokenizer, prompt)
        target_tokens = encode_text(tokenizer, target)
        encoded.append(
            EncodedDatum(
                prompt=prompt,
                target=target,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                datum=make_cross_entropy_datum(prompt_tokens, target_tokens),
            )
        )
    return encoded


def metric_value(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def sample_once(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    label: str,
) -> None:
    sampling_params = types.SamplingParams(
        max_tokens=SMOKE_SAMPLE_TOKENS,
        seed=1,
        temperature=SMOKE_TEMPERATURE,
        top_p=1.0,
        top_k=-1,
    )
    sample_result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    sequence = sample_result.sequences[0]
    completion = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    print(f"      {label}.stop_reason={sequence.stop_reason}", flush=True)
    print(f"      {label}.text={completion!r}", flush=True)


def run() -> int:
    args = parse_args()
    base_url = normalize_base_url(args.base_url)

    check_health(base_url)

    service_client = tinker.ServiceClient(base_url=base_url, api_key=args.api_key)
    base_model = discover_base_model(service_client, args.base_model)

    print(f"[3/7] creating LoRA training client rank={SMOKE_LORA_RANK}", flush=True)
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=SMOKE_LORA_RANK,
        seed=1,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
        user_metadata={"example": "skyrl-amd-tinker-smoke"},
    )
    tokenizer = training_client.get_tokenizer()

    encoded_datums = build_datums(tokenizer)
    datums = [item.datum for item in encoded_datums]
    total_prompt_tokens = sum(len(item.prompt_tokens) for item in encoded_datums)
    total_target_tokens = sum(len(item.target_tokens) for item in encoded_datums)
    print(
        "[4/7] built smoke batch "
        f"examples={len(datums)} prompt_tokens={total_prompt_tokens} target_tokens={total_target_tokens}",
        flush=True,
    )

    print("[5/7] syncing initial weights and sampling once", flush=True)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    sample_once(sampling_client, tokenizer, encoded_datums[0].prompt_tokens, "pre_train_sample")

    print(
        f"[6/7] training steps={SMOKE_STEPS} loss=cross_entropy lr={SMOKE_LEARNING_RATE}",
        flush=True,
    )
    for step in range(1, SMOKE_STEPS + 1):
        forward_backward_future = training_client.forward_backward(datums, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=SMOKE_LEARNING_RATE))

        forward_backward_result = forward_backward_future.result()
        optim_result = optim_future.result()
        fb_metrics = forward_backward_result.metrics
        optim_metrics = optim_result.metrics
        print(
            f"      step {step}/{SMOKE_STEPS} "
            f"tokens={metric_value(fb_metrics, 'num_tokens:sum')} "
            f"loss={metric_value(fb_metrics, 'total_loss:sum')} "
            f"grad_norm={metric_value(optim_metrics, 'skyrl.ai/grad_norm')} "
            f"lr={metric_value(optim_metrics, 'skyrl.ai/learning_rate')}",
            flush=True,
        )

    print("[7/7] syncing trained weights and sampling once", flush=True)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    sample_once(sampling_client, tokenizer, encoded_datums[0].prompt_tokens, "post_train_sample")

    print("PASS", flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"tinker_hello_world failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
