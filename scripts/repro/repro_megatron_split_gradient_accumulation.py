"""Reproduce Tinker/Megatron gradient loss across split forward_backward calls.

The split client receives two gradient-producing requests followed by a zero-weight
request. The combined client receives the same examples in one request. Their probe
losses must match after one optimizer step.
"""

import argparse
import asyncio

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData


def make_datum(tokenizer, prompt: str, completion: str, weight: float = 1.0) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    targets = tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * (len(prompt_tokens) - 1) + [weight] * (len(completion_tokens) + 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=targets, dtype="int64", shape=[len(targets)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


async def loss(client, datum: types.Datum) -> float:
    output = await client.forward_backward([datum], loss_fn="cross_entropy").result_async(timeout=1800)
    return sum(float(x) for item in output.loss_fn_outputs for x in item["elementwise_loss"].data)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=args.base_url)
    split = await service.create_lora_training_client_async(base_model=args.base_model, rank=args.rank)
    combined = await service.create_lora_training_client_async(base_model=args.base_model, rank=args.rank)
    tokenizer = split.get_tokenizer()
    first = make_datum(tokenizer, "Question: 1+1? Answer:", " 2")
    second = make_datum(tokenizer, "Question: 2+2? Answer:", " 4")
    final_zero = make_datum(tokenizer, "Question: 0+0? Answer:", " 0", weight=0.0)

    for datum in (first, second, final_zero):
        await split.forward_backward([datum], loss_fn="cross_entropy").result_async(timeout=1800)
    split_step = await split.optim_step(types.AdamParams(learning_rate=args.lr)).result_async(timeout=600)

    await combined.forward_backward([first, second, final_zero], loss_fn="cross_entropy").result_async(timeout=1800)
    combined_step = await combined.optim_step(types.AdamParams(learning_rate=args.lr)).result_async(timeout=600)

    probe = make_datum(tokenizer, "Question: 3+3? Answer:", " 6")
    split_loss = await loss(split, probe)
    combined_loss = await loss(combined, probe)
    delta = abs(split_loss - combined_loss)
    split_grad_norm = float(split_step.metrics.get("skyrl.ai/grad_norm") or 0.0)
    combined_grad_norm = float(combined_step.metrics.get("skyrl.ai/grad_norm") or 0.0)
    print(f"split_grad_norm={split_grad_norm}")
    print(f"combined_grad_norm={combined_grad_norm}")
    print(f"split_loss={split_loss:.9f} combined_loss={combined_loss:.9f} delta={delta:.9e}")
    if combined_grad_norm <= 0.0:
        print("FAIL: the combined request produced no trainable gradients")
        return 1
    if split_grad_norm <= 0.0:
        print("FAIL: split requests lost all trainable gradients")
        return 1
    if delta > 1e-5:
        print("FAIL: split requests did not produce the combined update")
        return 1
    print("PASS: every split request contributed to the optimizer update")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
