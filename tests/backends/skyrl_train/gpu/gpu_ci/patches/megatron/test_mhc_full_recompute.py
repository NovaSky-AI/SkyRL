"""Full activation recompute on GLM-5.3-Flash's mHC layers must not change the loss or gradients.

megatron-core refuses mHC with ``recompute_granularity="full"``; SkyRL bypasses that one check for
its own ``HyperConnectionTransformerLayer`` (``Glm5NextModelProvider.finalize`` in ``patches/megatron/glm5_next/provider.py``).
This builds the same random-init GLM-5.3-Flash slice twice through the worker's own
``init_configs`` / ``make_megatron_module`` -- once without recompute, once with full recompute
(uniform, one layer per chunk) -- copies the weights across, and compares one forward/backward.

The slice is ``eatang/GLM-5.3-Flash-4layer``'s config (2 KDA + 2 NoPE-MLA/DSA layers, 1 dense + 3
MoE, mHC on every block) cut to 16 routed experts so it fits one GPU; weights are random. The
sequence is longer than ``index_topk`` so the DSA layers run the k-pool selection, which a
replayed forward must reproduce exactly.

Grouped-GEMM / fla kernels are not bitwise deterministic, so the recompute error is compared to
the run-to-run noise of each mode (a second step of the same model on the same batch) rather
than to zero: over the whole model, and per parameter for tensors large enough that their
gradient is not noise-dominated.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_mhc_full_recompute.py
"""

import json
import os
import statistics

import pytest
import ray
import torch

pytestmark = pytest.mark.megatron

HF_REPO = "eatang/GLM-5.3-Flash-4layer"
NUM_EXPERTS = 16
SEQ_LEN = 3072  # > index_topk (2048): exercises the k-pool top-k selection


def _init_single_rank_megatron():
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _make_model_dir(root: str) -> str:
    from huggingface_hub import hf_hub_download

    model_dir = os.path.join(root, "glm5p3_flash_4layer_tiny")
    os.makedirs(model_dir, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = hf_hub_download(HF_REPO, name)
        with open(src, "rb") as f_in, open(os.path.join(model_dir, name), "wb") as f_out:
            f_out.write(f_in.read())
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["text_config"]["n_routed_experts"] = NUM_EXPERTS
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return model_dir


def _build(model_dir: str, recompute_kwargs: dict):
    from types import SimpleNamespace

    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker
    from skyrl.train.config.config import MegatronConfig

    # Weights come from the no-recompute model below; nothing to load from model_dir.
    megatron_worker.SKYRL_MEGATRON_RANDOM_INIT = True

    megatron_config = MegatronConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        mtp_num_layers=0,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="none",
    )
    transformer_config_kwargs = {
        "gradient_accumulation_fusion": False,
        "dsa_kernel_backend": "tilelang",
        **recompute_kwargs,
    }
    # Only what init_configs reads or writes on the worker.
    worker = SimpleNamespace(
        cfg=SimpleNamespace(
            gradient_checkpointing=True,
            mtp=None,
            algorithm=SimpleNamespace(enable_sample_support_replay=False),
        ),
        strategy=SimpleNamespace(),
    )
    megatron_worker.MegatronWorker.init_configs(
        worker,
        model_dir,
        megatron_config,
        {},
        transformer_config_kwargs,
        bf16=True,
        flash_attn=True,  # the trainer default; the GPU test env exports NVTE_FUSED_ATTN=0
        language_model_only=True,
    )
    model = megatron_worker.MegatronWorker.make_megatron_module(worker, wrap_with_ddp=False, bf16=True)[0]
    return model, worker.provider


def _step(model, input_ids, position_ids, attention_mask, labels):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    logits = model(input_ids, position_ids, attention_mask)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.item(), grads, peak


def _global_rel_err(a: dict, b: dict) -> float:
    num = sum(((a[n] - b[n]) ** 2).sum() for n in b)
    den = sum((b[n] ** 2).sum() for n in b)
    return (num / den).sqrt().item()


def _rel_err(a: dict, b: dict) -> dict:
    assert a.keys() == b.keys()
    return {n: ((a[n] - b[n]).norm() / b[n].norm().clamp_min(1e-12)).item() for n in a}


@ray.remote(num_gpus=1)
def _full_recompute_parity(root: str):
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        to_te_attention_mask,
    )

    _init_single_rank_megatron()
    model_dir = _make_model_dir(root)

    torch.manual_seed(0)
    base_model, base_provider = _build(model_dir, {"recompute_granularity": None})
    full_model, full_provider = _build(
        model_dir,
        {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1},
    )
    full_model.load_state_dict(base_model.state_dict(), strict=True)

    torch.manual_seed(1)
    vocab = base_provider.vocab_size
    input_ids = torch.randint(0, vocab, (1, SEQ_LEN), device="cuda")
    labels = torch.randint(0, vocab, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)
    attention_mask = to_te_attention_mask(torch.ones(1, SEQ_LEN, dtype=torch.bool, device="cuda"))

    loss_a, grads_a, peak_base = _step(base_model, input_ids, position_ids, attention_mask, labels)
    loss_b, grads_b, _ = _step(base_model, input_ids, position_ids, attention_mask, labels)
    loss_f, grads_f, peak_full = _step(full_model, input_ids, position_ids, attention_mask, labels)
    loss_f2, grads_f2, _ = _step(full_model, input_ids, position_ids, attention_mask, labels)

    # Run-to-run noise within each mode, and the difference across modes.
    noise_base = _rel_err(grads_b, grads_a)
    noise_full = _rel_err(grads_f2, grads_f)
    recompute = _rel_err(grads_f, grads_a)
    noise = {n: max(noise_base[n], noise_full[n]) for n in recompute}
    global_err = {
        "recompute": _global_rel_err(grads_f, grads_a),
        "noise_base": _global_rel_err(grads_b, grads_a),
        "noise_full": _global_rel_err(grads_f2, grads_f),
    }
    return {
        "base_granularity": base_provider.recompute_granularity,
        "full_granularity": full_provider.recompute_granularity,
        "full_method": full_provider.recompute_method,
        "full_num_layers": full_provider.recompute_num_layers,
        "decoder_granularity": getattr(full_model, "module", full_model).decoder.config.recompute_granularity,
        "losses": (loss_a, loss_b, loss_f, loss_f2),
        "noise": noise,
        "recompute": recompute,
        "global_err": global_err,
        "numel": {n: g.numel() for n, g in grads_a.items()},
        "peak_gb": (peak_base / 2**30, peak_full / 2**30),
    }


@ray.remote(num_gpus=1)
def _full_recompute_still_validated(root: str):
    """Only the mHC guard is bypassed: megatron-core's other full-recompute checks still run."""
    _init_single_rank_megatron()
    model_dir = _make_model_dir(root)
    try:
        _build(model_dir, {"recompute_granularity": "full", "recompute_method": None, "recompute_num_layers": 1})
    except ValueError as e:
        return str(e)
    return None


def test_mhc_full_recompute_matches_no_recompute(ray_init_fixture, tmp_path):
    r = ray.get(_full_recompute_parity.remote(str(tmp_path)))
    print(json.dumps({k: v for k, v in r.items() if k not in ("noise", "recompute")}, indent=2))
    worst = sorted(r["recompute"].items(), key=lambda kv: -kv[1])[:5]
    print("worst recompute rel err:", worst)
    print("max noise rel err:", max(r["noise"].values()))

    # Full recompute actually stayed on (not downgraded to selective) and reached the decoder.
    assert r["base_granularity"] is None
    assert (r["full_granularity"], r["full_method"], r["full_num_layers"]) == ("full", "uniform", 1)
    assert r["decoder_granularity"] == "full"

    loss_a, loss_b, loss_f, loss_f2 = r["losses"]
    loss_noise = max(abs(loss_b - loss_a), abs(loss_f2 - loss_f))
    assert abs(loss_f - loss_a) <= max(4 * loss_noise, 1e-4), r["losses"]

    # Whole-model gradient: no further from the reference than a rerun of either mode.
    g = r["global_err"]
    print("global grad rel err:", g)
    assert g["recompute"] <= 1.5 * max(g["noise_base"], g["noise_full"]), g

    # Per parameter, the recompute error should look like one more rerun: ratio ~1.
    ratios = {n: r["recompute"][n] / max(r["noise"][n], 1e-12) for n in r["recompute"]}
    assert statistics.median(ratios.values()) <= 1.5, sorted(ratios.items(), key=lambda kv: -kv[1])[:5]
    # Small tensors (the per-stream mHC alpha scalars / biases) have near-zero, noise-dominated
    # gradients whose error ratio is heavy-tailed; hold every large tensor to it individually.
    for name, ratio in ratios.items():
        if r["numel"][name] >= 4096:
            assert ratio <= 4, (name, r["recompute"][name], r["noise"][name])

    # And it saves activation memory, which is the point.
    peak_base, peak_full = r["peak_gb"]
    assert peak_full < 0.75 * peak_base, r["peak_gb"]


def test_other_full_recompute_checks_still_apply(ray_init_fixture, tmp_path):
    msg = ray.get(_full_recompute_still_validated.remote(str(tmp_path)))
    assert msg is not None and "recompute_method" in msg, msg
