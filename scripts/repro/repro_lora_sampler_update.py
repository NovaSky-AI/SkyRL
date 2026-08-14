"""Verify that consecutive Megatron LoRA updates reach vLLM sampling.

The probe compares greedy token logprobs before training, after update 1, and after
update 2. Equality at either boundary isolates a no-op export or stale reload.
"""

import argparse
import asyncio
import math

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

TRAIN_TEXT = "The quick brown fox jumps over the lazy dog. " * 40
PROBE_TEXT = "Question: What is 2 + 2? Answer:"


async def greedy_logprobs(client, prompt_tokens: list[int]) -> list[float]:
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=24, temperature=0.0),
    )
    return [round(float(x), 6) for x in result.sequences[0].logprobs]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--base-model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=args.base_url)
    training = await service.create_lora_training_client_async(
        base_model=args.base_model, rank=args.rank
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokens = tokenizer.encode(TRAIN_TEXT)[:512]
    n = len(tokens) - 1
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=tokens[1:], dtype="int64", shape=[n]),
            "weights": TensorData(data=[1.0] * n, dtype="float32", shape=[n]),
        },
    )
    prompt_tokens = tokenizer.encode(PROBE_TEXT)

    series = []
    losses = []
    sampler = await training.save_weights_and_get_sampling_client_async()
    series.append(await greedy_logprobs(sampler, prompt_tokens))
    for update in (1, 2):
        backward = await training.forward_backward(
            [datum], loss_fn="cross_entropy"
        ).result_async(timeout=1800)
        loss = float(backward.metrics.get("total_loss:sum", float("nan")))
        losses.append(loss)
        step = await training.optim_step(
            types.AdamParams(learning_rate=args.lr)
        ).result_async(timeout=600)
        sampler = await training.save_weights_and_get_sampling_client_async()
        series.append(await greedy_logprobs(sampler, prompt_tokens))
        print(
            f"update={update} loss={loss} "
            f"grad_norm={step.metrics.get('skyrl.ai/grad_norm')}"
        )

    print("initial:", series[0][:8])
    print("update1:", series[1][:8])
    print("update2:", series[2][:8])
    if not all(math.isfinite(loss) and loss > 0.0 for loss in losses):
        print("FAIL: cross-entropy produced no finite positive loss")
        return 1
    if series[0] == series[1]:
        print("FAIL: first exported adapter had no effect on sampling")
        return 1
    if series[1] == series[2]:
        print("FAIL: second adapter reload remained on stale weights")
        return 1
    print("PASS: both optimizer updates changed vLLM sampling")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
