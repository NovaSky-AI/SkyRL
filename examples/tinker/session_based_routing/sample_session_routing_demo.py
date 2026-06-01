"""Demo of explicit ``seq_id`` for trajectory-pinned routing on SkyRL's Tinker API.

Usage:
    # Terminal 1: start a SkyRL Tinker API server with the SkyRL-Train backend
    bash examples/tinker/session_based_routing/run_tinker_server.sh

    # Terminal 2
    TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with transformers \\
        python examples/tinker/session_based_routing/sample_session_routing_demo.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import tinker
from tinker import types
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from transformers import AutoTokenizer


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS_PER_TURN = 32
logger = logging.getLogger("sample_session_routing_demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-trajectories", type=int, default=4)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


async def asample_explicit(
    holder,
    *,
    sampling_session_id: str,
    seq_id: int,
    prompt: types.ModelInput,
    sampling_params: types.SamplingParams,
) -> types.SampleResponse:
    """Submit one /api/v1/asample with caller-controlled ``sampling_session_id`` + ``seq_id``.

    ``SamplingClient.sample`` does not expose ``seq_id``, so the demo builds the
    raw ``SampleRequest`` and dispatches via ``client.sampling.asample`` directly.
    """
    request = types.SampleRequest(
        sampling_session_id=sampling_session_id,
        seq_id=seq_id,
        num_samples=1,
        prompt=prompt,
        sampling_params=sampling_params,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )
    logger.info("dispatch routing-key=%s:%d", sampling_session_id, seq_id)

    start = time.time()
    with holder.aclient(ClientConnectionPoolType.SAMPLE) as client:
        untyped_future = await client.sampling.asample(request=request, max_retries=0)
    return await _APIFuture(
        types.SampleResponse,
        holder,
        untyped_future,
        request_start_time=start,
        request_type="Sample",
    )


async def run_trajectory(
    holder,
    tokenizer,
    *,
    sampling_session_id: str,
    trajectory_idx: int,
    num_turns: int,
    seed: int,
) -> list[str]:
    """Multi-turn rollout where every turn reuses ``seq_id=trajectory_idx`` so the
    whole trajectory lands on the same backend (warm prefix cache)."""
    chat = [{"role": "user", "content": f"Tell me a one-sentence story about trajectory #{trajectory_idx}."}]
    outputs: list[str] = []
    for turn in range(num_turns):
        prompt_tokens = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        result = await asample_explicit(
            holder,
            sampling_session_id=sampling_session_id,
            seq_id=trajectory_idx,
            prompt=types.ModelInput.from_ints(list(prompt_tokens)),
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOKENS_PER_TURN,
                temperature=0.7,
                seed=seed + 100 * trajectory_idx + turn,
            ),
        )
        tokens = list(result.sequences[0].tokens)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        outputs.append(text)
        chat.append({"role": "assistant", "content": text})
        chat.append({"role": "user", "content": "Continue."})
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    # rank=0 picks the FFT path on the SkyRL-Train backend (no LoRA).
    training_client = service_client.create_lora_training_client(base_model=args.model, rank=0)
    sampling_client = training_client.save_weights_and_get_sampling_client(name="routing_demo")
    # _sampling_session_id is the only way to read the SDK-issued session id at
    # this version; the public API does not expose it.
    sampling_session_id: str = sampling_client._sampling_session_id
    holder = sampling_client.holder

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(
        "base_model=%s sampling_session_id=%s num_trajectories=%d turns=%d",
        args.model,
        sampling_session_id,
        args.num_trajectories,
        args.turns,
    )

    try:
        # The SDK's holder owns its own event loop; submit onto it and block.
        futures = [
            holder.run_coroutine_threadsafe(
                run_trajectory(
                    holder,
                    tokenizer,
                    sampling_session_id=sampling_session_id,
                    trajectory_idx=i,
                    num_turns=args.turns,
                    seed=args.seed,
                )
            ).future()
            for i in range(args.num_trajectories)
        ]
        results = [f.result() for f in futures]
    finally:
        service_client.holder.close()

    for i, turns in enumerate(results):
        logger.info("trajectory=%d turns=%d first_turn=%r", i, len(turns), turns[0][:80] if turns else "")


if __name__ == "__main__":
    main()
