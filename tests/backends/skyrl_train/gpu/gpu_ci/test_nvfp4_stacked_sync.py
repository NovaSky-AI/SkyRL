"""Stacked NVFP4 expert sync (``expert_nvfp4_stacked_sync``) must leave vLLM bitwise where the per-expert sync does.

Builds a tiny random Qwen3-MoE checkpoint, boots vLLM exactly as the expert-NVFP4 sampler boots (ModelOpt NVFP4
expert-only, dummy weights, per-token activations, FlashInfer TRT-LLM MoE), then on every worker runs the receive
side of a weight sync twice over the same NVFP4 expert bytes: once as the per-expert stream, once stacked. Each
pass is what the receive engine does (``initialize_layerwise_reload`` -> ``_load_checkpoint_weights`` in small
batches, so stacked layers span batches -> ``finalize_layerwise_reload``). After each pass it digests every
parameter and buffer and runs greedy decoding with logprobs. Both must match exactly, and the stacked pass must
have taken the direct path on every NVFP4 layer.

The NVFP4 bytes come from a deterministic stand-in quantizer (valid E2M1 / E4M3 bytes, no TE needed): the
receiver is under test here, not TE; both paths get the same bytes.

Needs SM100 (NVFP4 TRT-LLM MoE). Run:

    uv run --isolated --extra dev --extra megatron pytest -s \\
        tests/backends/skyrl_train/gpu/gpu_ci/test_nvfp4_stacked_sync.py
"""

import gc
import json
import os

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_blackwell_or_newer,
)

pytestmark = pytest.mark.skipif(not is_blackwell_or_newer(), reason="NVFP4 TRT-LLM MoE needs SM100+")

NUM_LAYERS, NUM_EXPERTS, HIDDEN, INTER, VOCAB = 2, 8, 1024, 512, 2048
PROMPT = [11, 257, 1023, 5, 77, 900, 1500, 3, 64, 2000, 12, 18]


def _write_tiny_qwen3_moe(path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        moe_intermediate_size=INTER,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump({}, f)


def _fake_quantize(weight, settings):
    """Deterministic, input-dependent, finite NVFP4 bytes (packed E2M1, positive E4M3 block scales)."""
    rows, cols = weight.shape
    w = weight.float()
    packed = (w[:, : cols // 2].abs() * 97 + w[:, cols // 2 :].abs() * 31).to(torch.int64).remainder(256)
    scale = w.abs().reshape(rows, cols // 16, 16).amax(-1).clamp(0.25, 2.0).to(torch.float8_e4m3fn)
    decode = (w.abs().amax() * 1e-3 + 1e-3).reshape(()).float()
    return packed.to(torch.uint8).contiguous(), scale.contiguous(), decode


def _worker_sync(worker, mode, seed):
    """Runs inside each vLLM worker: one receive-side sync in ``mode`` ("per_expert" | "stacked")."""
    import hashlib
    import itertools

    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
    )

    from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
        _load_checkpoint_weights,
    )
    from skyrl.backends.skyrl_train.weight_sync import nvfp4, nvfp4_stacked
    from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked_receive import (
        Nvfp4StackedReceiver,
    )

    nvfp4.quantize_nvfp4_matrix = _fake_quantize
    nvfp4_stacked.quantize_nvfp4_matrix = _fake_quantize
    model = worker.model_runner.model
    device = worker.device
    settings = nvfp4.Nvfp4WeightQuantSettings(
        use_4over6=False, e4m3_max=448, err_mode="MAE", row_scaled_activation=True, backward_override=None
    )
    if mode == "stacked":
        serializer = nvfp4_stacked.Nvfp4StackedExpertSerializer(settings, num_experts=NUM_EXPERTS)
    else:
        serializer = nvfp4.Nvfp4ExpertSerializer(settings)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    stream = []
    for layer in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            stream.append((f"{base}.gate_proj.weight", torch.randn(INTER, HIDDEN, generator=gen)))
            stream.append((f"{base}.up_proj.weight", torch.randn(INTER, HIDDEN, generator=gen)))
            stream.append((f"{base}.down_proj.weight", torch.randn(HIDDEN, INTER, generator=gen)))

    with torch.device(device):
        initialize_layerwise_reload(model)
        receiver = Nvfp4StackedReceiver(model, fallback_load=lambda w: model.load_weights(weights=w))
        batch = []
        for name, tensor in stream:
            for out_name, out in serializer(name, tensor.to(device=device, dtype=torch.bfloat16)):
                batch.append((out_name, out.contiguous()))
                if len(batch) >= 5:  # small batches: stacked layers span several load_weights calls
                    _load_checkpoint_weights(model, batch, nvfp4_stacked_receiver=receiver)
                    batch = []
        serializer.finish()
        if batch:
            _load_checkpoint_weights(model, batch, nvfp4_stacked_receiver=receiver)
        receiver.finish()
        finalize_layerwise_reload(model, worker.model_config)
    torch.cuda.synchronize()

    digest = {}
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        data = tensor.detach().contiguous().reshape(-1)
        h = hashlib.blake2b(digest_size=16)
        if data.numel():
            h.update(memoryview(data.view(torch.uint8).cpu().numpy()))
        digest[name] = f"{tuple(tensor.shape)}|{tensor.dtype}|{h.hexdigest()}"
    return {"fast": receiver.layers_fast, "fallback": receiver.layers_fallback, "digest": digest}


def _greedy(llm):
    from vllm import SamplingParams

    out = llm.generate(
        [{"prompt_token_ids": PROMPT}],
        SamplingParams(temperature=0.0, max_tokens=16, logprobs=0, ignore_eos=True),
        use_tqdm=False,
    )[0].outputs[0]
    return list(out.token_ids), [lp[t].logprob for lp, t in zip(out.logprobs, out.token_ids)]


@pytest.mark.parametrize("tp,ep", [(1, False), (2, True)], ids=["tp1", "tp2_ep2"])
def test_stacked_sync_matches_per_expert_bitwise(tmp_path, tp, ep):
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    import sys

    import cloudpickle
    from vllm import LLM

    from skyrl.backends.skyrl_train.weight_sync.nvfp4 import (
        get_nvfp4_vllm_engine_overrides,
        nvfp4_runtime_env,
    )

    # The worker function and its helpers travel by value: the workers cannot import this test module.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    os.environ.update(nvfp4_runtime_env(dict(os.environ)))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # collective_rpc with a callable
    model_dir = str(tmp_path / "tiny_qwen3_moe")
    _write_tiny_qwen3_moe(model_dir)
    overrides = get_nvfp4_vllm_engine_overrides(NUM_LAYERS, 0)
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        seed=0,
        quantization=overrides["quantization"],
        load_format=overrides["load_format"],
        quantization_config=overrides["quantization_config"],
        hf_overrides=overrides["hf_overrides"],
        moe_backend="flashinfer_trtllm",
    )
    try:
        runs = {}
        for mode in ("per_expert", "stacked", "per_expert"):
            per_worker = llm.collective_rpc(_worker_sync, args=(mode, 1234))
            runs.setdefault(mode, []).append((per_worker, _greedy(llm)))

        (ref_workers, ref_gen), (ref2_workers, ref2_gen) = runs["per_expert"]
        ((stk_workers, stk_gen),) = runs["stacked"]
        for w in stk_workers:
            assert w["fast"] == NUM_LAYERS and w["fallback"] == 0, (w["fast"], w["fallback"])
        for w in ref_workers:
            assert w["fast"] == 0 and w["fallback"] == 0
        for rank, (a, b, c) in enumerate(zip(ref_workers, stk_workers, ref2_workers)):
            assert a["digest"].keys() == b["digest"].keys()
            diff = [k for k in a["digest"] if a["digest"][k] != b["digest"][k]]
            assert not diff, f"rank {rank}: {len(diff)} tensors differ after the stacked sync, e.g. {diff[:5]}"
            assert a["digest"] == c["digest"], f"rank {rank}: per-expert sync is not reproducible"
        assert ref_gen == stk_gen == ref2_gen, (ref_gen, stk_gen, ref2_gen)
        assert all(lp == lp for lp in ref_gen[1]), "NaN logprobs"
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
