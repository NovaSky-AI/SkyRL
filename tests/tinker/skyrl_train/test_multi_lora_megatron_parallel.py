"""Multi-GPU version of PR #2008's cross-request accumulation test.

Runs the comparison from ``test_forward_backward_accumulates_across_requests``
(three chunked forward_backward calls vs one combined call, final chunk
zero-weighted) at DP=2, where a per-call gradient sync corrupts the result.

DP=2 is the only topology kept here, because it is the only one that fails
without the accompanying fix. The full sweep was run once during review:

    topology   |split - combined|
    dp1        0.00e+00   <- also covered by the PR's own test
    dp2        1.18e-02
    tp2        0.00e+00
    pp2        0.00e+00

TP/PP come out clean only because LoRA trains no layernorm or embedding
parameters, so the ReduceOp.SUM all-reduces in finalize_model_grads have
nothing to touch; a full fine-tune at TP/PP > 1 double-counts them the same
way. Re-add those topologies to TOPOLOGIES if this ever covers FFT.

DP1 was bit-exact despite the split and combined paths having different chunk
counts and microbatch groupings, which establishes that the comparison has no
floating-point noise floor -- any non-zero delta at DP2 is a real gradient
error, not reassociation.

Run with:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_multi_lora_megatron_parallel.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() >= 3)
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="parallel multi-LoRA tests need >= 3 CUDA GPUs")

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402
from tests.tinker.skyrl_train.test_multi_lora_megatron import (  # noqa: E402
    BASE_MODEL,
    TINKER_API_KEY,
    _server_is_up,
)


def _config(num_gpus: int, tp: int, pp: int) -> dict:
    return {
        "strategy": "megatron",
        "trainer.placement.policy_num_gpus_per_node": num_gpus,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.colocate_all": False,
        "trainer.policy.megatron_config.tensor_model_parallel_size": tp,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": pp,
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "trainer.policy.model.lora.max_loras": 4,
        "trainer.policy.model.lora.max_cpu_loras": 4,
    }


# (label, port, config). Kept as a list so topologies are easy to re-add;
# note TP > 1 needs `transformer_config_kwargs.should_pad_vocab=True` because
# tiny-Qwen3's vocab (151669) isn't divisible by 2.
TOPOLOGIES = [
    ("dp2", 8022, _config(num_gpus=2, tp=1, pp=1)),
]


def _start_server(port: int, cfg: dict, log_path: str, db_path: str):
    cmd = [
        "uv",
        "run",
        "--isolated",
        "--extra",
        "tinker",
        "--extra",
        "megatron",
        "-m",
        "skyrl.tinker.api",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--base-model",
        BASE_MODEL,
        "--backend",
        "megatron",
        "--backend-config",
        json.dumps(cfg),
        "--database-url",
        f"sqlite:///{db_path}",
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    ok = wait_for_condition(lambda: _server_is_up(port), timeout_sec=300, poll_interval_sec=2)
    if not ok:
        proc.terminate()
        with open(log_path) as f:
            print(f"=== server ({port}) failed to start ===\n{f.read()[-8000:]}")
        pytest.fail(f"Tinker API server on port {port} did not come up")
    return proc, log_file


def _make_datum(tokenizer, prompt: str, completion: str, completion_weight: float = 1.0):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [completion_weight] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [completion_weight]},
    )


@pytest.mark.parametrize("label,port,cfg", TOPOLOGIES, ids=[t[0] for t in TOPOLOGIES])
def test_zero_weight_chunks_are_noops(label, port, cfg, request):
    """Control for the DP result: appending gradient-free chunks must change nothing.

    Both clients see the exact same real data in their first (and only
    gradient-producing) call, so sharding, microbatch grouping and FP
    summation order are identical between them. The only difference is
    that ``padded`` issues two extra all-zero forward_backward calls
    before its optim_step. A correct implementation is bit-identical;
    any delta is the extra per-call grad sync re-reducing gradients that
    were already reduced.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        proc, log_file = _start_server(port + 100, cfg, log_path, db_path)
        try:
            sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{port + 100}/", api_key=TINKER_API_KEY)
            padded = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
            plain = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
            tok = padded.get_tokenizer()

            real = [
                _make_datum(tok, "Question: 1+1?\nAnswer:", " 2"),
                _make_datum(tok, "Question: 5+3?\nAnswer:", " 8"),
            ]
            zeros = [
                _make_datum(tok, "Question: 0+0?\nAnswer:", " 0", completion_weight=0.0),
                _make_datum(tok, "Question: 9+0?\nAnswer:", " 9", completion_weight=0.0),
            ]

            padded.forward_backward(real, "cross_entropy").result()
            padded.forward_backward(zeros, "cross_entropy").result()
            padded.forward_backward(zeros, "cross_entropy").result()
            padded.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

            plain.forward_backward(real, "cross_entropy").result()
            plain.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

            probe = [
                _make_datum(tok, "Question: 3+3?\nAnswer:", " 6"),
                _make_datum(tok, "Question: 4+4?\nAnswer:", " 8"),
            ]
            padded_out = padded.forward_backward(probe, "cross_entropy").result()
            plain_out = plain.forward_backward(probe, "cross_entropy").result()
            padded_loss = sum(sum(o["elementwise_loss"].data) for o in padded_out.loss_fn_outputs)
            plain_loss = sum(sum(o["elementwise_loss"].data) for o in plain_out.loss_fn_outputs)

            print(
                f"\n[{label}/zero-chunks] padded={padded_loss!r} plain={plain_loss!r} "
                f"|Δ|={abs(padded_loss - plain_loss):.6e}"
            )

            assert padded_loss == pytest.approx(
                plain_loss, abs=1e-6
            ), f"[{label}] gradient-free chunks changed the update: padded={padded_loss}, plain={plain_loss}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_file.close()


@pytest.mark.parametrize("label,port,cfg", TOPOLOGIES, ids=[t[0] for t in TOPOLOGIES])
def test_forward_backward_accumulates_across_requests_parallel(label, port, cfg, request):
    """Same assertion as the PR's test, at DP/TP/PP > 1.

    Two datums per chunk so every DP rank gets real work in the split path.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        proc, log_file = _start_server(port, cfg, log_path, db_path)
        try:
            sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{port}/", api_key=TINKER_API_KEY)
            split = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
            combined = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
            tok = split.get_tokenizer()

            chunk1 = [
                _make_datum(tok, "Question: 1+1?\nAnswer:", " 2"),
                _make_datum(tok, "Question: 5+3?\nAnswer:", " 8"),
            ]
            chunk2 = [
                _make_datum(tok, "Question: 2+2?\nAnswer:", " 4"),
                _make_datum(tok, "Question: 7+1?\nAnswer:", " 8"),
            ]
            chunk3 = [
                _make_datum(tok, "Question: 0+0?\nAnswer:", " 0", completion_weight=0.0),
                _make_datum(tok, "Question: 9+0?\nAnswer:", " 9", completion_weight=0.0),
            ]
            assert not any(chunk3[0].loss_fn_inputs["weights"].data)

            for chunk in (chunk1, chunk2, chunk3):
                split.forward_backward(chunk, "cross_entropy").result()
            split.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

            combined.forward_backward(chunk1 + chunk2 + chunk3, "cross_entropy").result()
            combined.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

            probe = [
                _make_datum(tok, "Question: 3+3?\nAnswer:", " 6"),
                _make_datum(tok, "Question: 4+4?\nAnswer:", " 8"),
            ]
            split_out = split.forward_backward(probe, "cross_entropy").result()
            combined_out = combined.forward_backward(probe, "cross_entropy").result()
            split_loss = sum(sum(o["elementwise_loss"].data) for o in split_out.loss_fn_outputs)
            combined_loss = sum(sum(o["elementwise_loss"].data) for o in combined_out.loss_fn_outputs)

            print(
                f"\n[{label}] split={split_loss!r} combined={combined_loss!r} "
                f"|Δ|={abs(split_loss - combined_loss):.6e}"
            )

            assert split_loss == pytest.approx(
                combined_loss, abs=1e-6
            ), f"[{label}] separate requests produced a different update: split={split_loss}, combined={combined_loss}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_file.close()
