"""Score the fixed GLM probe and capture replay routes in the same completion."""

import base64
import io
import math

import numpy as np


async def score_with_routes(client, sequences, model):
    await client.reset_prefix_cache()
    scores, routes = [], []
    for tokens in sequences:
        result = await client.completion(
            {
                "json": {
                    "model": model,
                    "prompt": tokens,
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "n": 1,
                    "stream": False,
                    "prompt_logprobs": 0,
                    "add_special_tokens": False,
                    "return_token_ids": True,
                    "routed_experts_prompt_start": 0,
                }
            }
        )
        if len(result["choices"]) != 1:
            raise ValueError("Expected one completion for each fixed probe")
        choice = result["choices"][0]
        if choice["prompt_token_ids"] != tokens:
            raise ValueError("Completion changed the fixed prompt tokens")
        values = choice["prompt_logprobs"]
        if len(values) != len(tokens) or values[0] is not None:
            raise ValueError("Prompt logprobs must cover the unchanged prompt, starting with None")
        selected = [values[index][str(token)]["logprob"] for index, token in enumerate(tokens[1:], 1)]
        if any(not math.isfinite(value) or value == -9999 for value in selected):
            raise ValueError("Nonfinite or clamped prompt logprob")
        captured = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"], validate=True)), allow_pickle=False)
        if captured.shape != (len(tokens), 78, 8):
            raise ValueError(f"Expected full GLM prompt routes [tokens, 78, 8], got {captured.shape}")
        if not np.issubdtype(captured.dtype, np.integer) or np.any(captured < 0) or np.any(captured >= 256):
            raise ValueError("GLM routes must contain integer expert IDs in [0, 256)")
        scores.extend(selected)
        routes.append(captured)
    return scores, routes
