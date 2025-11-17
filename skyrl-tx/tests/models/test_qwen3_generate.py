import os
import tempfile
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tx.models import Qwen3Config, Qwen3ForCausalLM
from tx.tinker import types
from tx.utils.models import load_safetensors


@pytest.fixture
def mesh_configs():
    """Generate mesh configurations to test."""
    num_devices = len(jax.devices())
    configs = [
        {"name": "basic", "dp": 1, "tp": 1, "fsdp": False},
        {"name": "tp", "dp": 1, "tp": num_devices, "fsdp": False},
        {"name": "fsdp", "dp": num_devices, "tp": 1, "fsdp": True},
    ]

    return configs


@pytest.mark.parametrize("mesh_config_idx", [0, 1, 2])
def test_qwen3_generate(mesh_config_idx, mesh_configs):
    """Test batched text generation with KV caching matches HuggingFace."""
    mesh_config = mesh_configs[mesh_config_idx]
    print(
        f"\nTesting with mesh configuration: {mesh_config['name']} (dp={mesh_config['dp']}, tp={mesh_config['tp']}, fsdp={mesh_config['fsdp']})"
    )

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["My name is", "The capital of France is", "Test stopping"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Generate with HuggingFace (reference)
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Generate with our implementation
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        base_config = PretrainedConfig.from_pretrained(model_name)
        config = Qwen3Config(
            base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True, fsdp=mesh_config["fsdp"]
        )

        mesh = jax.make_mesh((mesh_config["dp"], mesh_config["tp"]), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        sampling_params = [
            types.SamplingParams(max_tokens=10, temperature=0.0, seed=42),
            types.SamplingParams(max_tokens=20, temperature=0.0, seed=42),
            types.SamplingParams(max_tokens=50, temperature=0.0, seed=42, stop=[6149]),
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
        )

        # Compare generated tokens
        for i, (our_tokens, hf_tokens, sampling_param) in enumerate(
            zip(result.generated_ids, hf_output.sequences, sampling_params)
        ):
            prompt_length = batch.input_ids.shape[1]
            hf_tokens_truncated = hf_tokens[prompt_length : prompt_length + sampling_param.max_tokens].tolist()

            if sampling_param.stop:
                assert result.stop_reasons[i] == "stop"
                assert our_tokens[-1] in sampling_param.stop
                # We need to truncate it manually here since if we use the `eos_token_id`
                # in huggingface generate, it will pad the sequence with padding tokens
                hf_tokens_truncated = hf_tokens_truncated[: len(our_tokens)]

            assert our_tokens == hf_tokens_truncated, (
                f"Generated tokens for request {i} don't match HuggingFace. "
                f"Ours: {our_tokens}, HF: {hf_tokens_truncated}"
            )

        # Compare logprobs for sampled tokens
        for i, (our_tokens, our_logprobs) in enumerate(zip(result.generated_ids, result.logprobs)):
            # Compute expected logprobs from HF scores
            for step_idx, (token_id, our_logprob) in enumerate(zip(our_tokens, our_logprobs)):
                hf_logits = hf_output.scores[step_idx][i]
                hf_logprobs = torch.nn.functional.log_softmax(hf_logits, dim=-1)
                expected_logprob = float(hf_logprobs[token_id])

                assert np.isclose(our_logprob, expected_logprob, rtol=5e-3, atol=5e-3), (
                    f"Request {i}, step {step_idx}: Logprob mismatch. "
                    f"Ours: {our_logprob}, HF: {expected_logprob}, diff: {abs(our_logprob - expected_logprob)}"
                )


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Skip speed test in CI due to memory limits")
@pytest.mark.parametrize("mesh_config_idx", [0, 1, 2])
def test_qwen3_generate_speed(mesh_config_idx, mesh_configs):
    """Profile batched text generation with KV caching."""
    mesh_config = mesh_configs[mesh_config_idx]
    print(
        f"\nTesting with mesh configuration: {mesh_config['name']} (dp={mesh_config['dp']}, tp={mesh_config['tp']}, fsdp={mesh_config['fsdp']})"
    )

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(
        base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True, fsdp=mesh_config["fsdp"]
    )

    inputs = [
        "Why do humans need sleep and what happens when we dream",
        "Explain the meaning of life and consciousness",
        "Describe the process of photosynthesis in plants",
        "How do airplanes fly through the air efficiently",
        "What are black holes and how are they formed",
        "Tell me about the solar system and its planets",
        "Explain the difference between AI and machine learning",
        "How does the human brain process language",
        "What is quantum computing and how does it work",
    ]

    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        mesh = jax.make_mesh((mesh_config["dp"], mesh_config["tp"]), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)
        sampling_params = [types.SamplingParams(max_tokens=50, temperature=0.0, seed=42) for i in range(len(inputs))]

        # Warmup
        model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
        )

        runs = 1
        times = []

        for i in range(runs):
            start = time.perf_counter()
            result = model.generate(
                batch.input_ids.numpy(),
                batch.attention_mask.numpy(),
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()

        total_new_tokens = len(result.generated_ids) * 50

    print(f"Generation stats (50 tokens, {runs} runs):")
    print(f"Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Min/Max: {times.min()*1000:.2f} / {times.max()*1000:.2f} ms")
    print(f"New tokens/sec: {total_new_tokens / mean_time:.2f}")
