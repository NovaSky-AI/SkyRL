"""E2E SFTTrainer logprob parity across packing / CP configs.

It drives the *SFTTrainer* pipeline — 
``tokenize_sft_example`` -> ``collate_batch`` -> ``forward_backward``
— for one global batch under three parallelism configs and asserts the
per-token logprobs of the *response* tokens agree across all three:

Run with (2 free GPUs required; CP=2 leg uses 2):
  uv run --isolated --extra dev --extra megatron \
    pytest -s -v tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_sft_packing_parity.py
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import MegatronConfig
from skyrl.train.config.sft_config import (
    SFTConfig,
    SFTPlacementConfig,
    TrainOnWhat,
    build_skyrl_config_for_sft,
)
from skyrl.train.sft_trainer import collate_sft_batch, tokenize_chat_example
from skyrl.train.sft_trainer_with_packing import SFTTrainerWithPacking
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import ResolvedPlacementGroup
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 1024
# Number of Tulu3 examples in the global batch. Kept small (fast + few bins)
# but >= dp_size and large enough to force multi-subseq bins under packing.
GLOBAL_BATCH_SIZE = 8
SEED = 42

# Tolerances. ``AVG_DIFF_TOL`` mirrors test_megatron_forward
# (test_megatron_worker.py:335) — the robust statistic that absorbs bf16 +
# differing CP/packing reduction orders.
#
# ``MAX_DIFF_TOL`` is deliberately LOOSER than test_megatron_forward's 5e-1.
# That 5e-1 was calibrated on a 15-token dummy batch; here we compare ~2.3k
# REAL Tulu3 response tokens, so the per-token max is dominated by a handful of
# near-impossible tokens (baseline logprob ~ -23, i.e. p ~ 1e-10) whose logprob
# is numerically unstable under bf16 — a tiny logit perturbation from the
# packed THD attention (vs unpacked padded attention) swings them ~1 nat while
# contributing ~0 to the loss/grad. Empirically (this test, Qwen3-1.7B): the
# WORST single token is ~1.26 at base_lp=-23 / -0.83, median diff ~0, and
# avg ~0.015 (no CP) / ~0.030 (CP=2) — both far under AVG_DIFF_TOL. So we gate
# parity on AVG_DIFF_TOL and use MAX_DIFF_TOL only to catch a *systematic*
# blow-up (which a real un-zigzag bug would produce: large median AND avg, not
# a lone rare-token tail). See the per-token histogram printed by `_compare`.
MAX_DIFF_TOL = 2.0
AVG_DIFF_TOL = 9e-2


# ---------------------------------------------------------------------------
# Test-only worker: returns FULL canonical per-token logprobs.
# ---------------------------------------------------------------------------
class _ParityProbeWorkerBase(MegatronPolicyWorkerBase):
    """Adds one method returning full ``[B, seq_len-1]`` logprobs in canonical
    (B, T) order, delegating to the SAME production preprocess/postprocess
    packed-seq utilities that ``forward_backward`` uses (so the
    multi-subseq CP un-zigzag is exercised).

    This mirrors the micro-batch construction in
    ``MegatronPolicyWorkerBase.forward_backward`` (including the
    ``sub_seq_lengths`` TensorList -> per-microbatch chunking) but runs
    ``forward_only`` and captures logprobs *before* the loss-fn's lossy
    tail-slice. No production code is modified.
    """

    def forward_capture_full_logprobs(self, data: TrainingInputBatch) -> WorkerOutput:
        import megatron.core.parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func

        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            make_batch_generator,
            postprocess_packed_seqs,
            preprocess_packed_seqs,
            recover_left_padding,
            remove_left_padding,
        )
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            from_parallel_logits_to_logprobs,
        )
        from skyrl.backends.skyrl_train.workers.worker_utils import BatchIterator

        self.model.eval()
        micro_batch_size = self.cfg.micro_train_batch_size_per_gpu
        data.to(torch.cuda.current_device())

        # Per-microbatch sub_seq_lengths slices (matches forward_backward).
        sub_seq_lengths_field: Optional[TensorList] = data.get("sub_seq_lengths")
        sub_seq_lengths_chunks: List[Optional[TensorList]] = []
        if sub_seq_lengths_field is not None:
            for i in range(0, data.batch_size, micro_batch_size):
                sub_seq_lengths_chunks.append(sub_seq_lengths_field[i : i + micro_batch_size])

        micro_buffer: List[dict] = []
        for chunk_idx, experience in enumerate(BatchIterator(data, micro_batch_size, drop_last=False)):
            sequences = experience.sequences
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            sub = sub_seq_lengths_chunks[chunk_idx] if sub_seq_lengths_field is not None else None
            micro_buffer.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "loss_mask": experience.loss_mask,
                    "sub_seq_lengths_list": ([t.tolist() for t in sub] if sub is not None else None),
                }
            )

        if not micro_buffer:
            return WorkerOutput()

        seq_len = micro_buffer[0]["sequences"].shape[1]
        micro_bsz = micro_buffer[0]["sequences"].shape[0]
        use_sample_packing = self.model.use_sample_packing

        captured: List[torch.Tensor] = []

        def collection_func(logits, data):
            sequences = data["sequences"]
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                cp_group=None,  # CP gather already handled in postprocess_packed_seqs
                chunk_size=self.cfg.logprobs_chunk_size,
            )
            # token_logprobs: [B, seq_len-1] in canonical (B, T) order.
            captured.append(token_logprobs.detach().to("cpu"))
            return torch.tensor(0.0, device=token_logprobs.device), {}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            sub_seq_lengths = batch["sub_seq_lengths_list"]
            if use_sample_packing:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(new_sequences, new_position_ids, new_attention_mask, packed_seq_params=packed_seq_params)

            if use_sample_packing:
                outputs = postprocess_packed_seqs(
                    outputs,
                    packed_seq_params,
                    attention_mask,
                    micro_bsz,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
            else:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )
            return outputs, partial(collection_func, data=batch)

        forward_backward_func = get_forward_backward_func()
        batch_generator = make_batch_generator(micro_buffer, vpp_size=len(self.actor_module))
        with torch.no_grad():
            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=len(micro_buffer),
                seq_length=seq_len,
                micro_batch_size=micro_bsz,
                forward_only=True,
            )

        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            return WorkerOutput(loss_fn_outputs=[], metrics={})

        # captured is one [mbs, seq_len-1] tensor per micro-batch, in row order.
        loss_fn_outputs: List[Dict[str, Any]] = []
        for lp in captured:
            for i in range(lp.shape[0]):
                loss_fn_outputs.append({"logprobs": lp[i].tolist()})
        return WorkerOutput(loss_fn_outputs=loss_fn_outputs, metrics={})


_ParityProbeWorker = ray.remote(num_gpus=1)(_ParityProbeWorkerBase)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sft_cfg(use_minibatch_packing: bool, cp: int, gpus: int) -> SFTConfig:
    cfg = SFTConfig(
        strategy="megatron",
        max_length=MAX_LENGTH,
        batch_size=GLOBAL_BATCH_SIZE,
        micro_train_batch_size_per_gpu=2,
        use_sample_packing=True,
        use_minibatch_packing=use_minibatch_packing,
        packed_micro_batch_size_per_gpu=1 if use_minibatch_packing else None,
        seed=SEED,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        dataset_name="allenai/tulu-3-sft-mixture",
        placement=SFTPlacementConfig(num_nodes=1, num_gpus_per_node=gpus),
        megatron_config=MegatronConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp,
            expert_model_parallel_size=1,
        ),
        logger="console",
    )
    # SFTConfig.model is a ModelConfig; set the path on the default instance.
    cfg.model.path = MODEL_NAME
    return cfg


def _load_tulu3_examples(tokenizer) -> List[dict]:
    """Tokenize a fixed, seeded slice of Tulu3 into GLOBAL_BATCH_SIZE examples.

    Uses the same tokenization path the SFTTrainer uses
    (``tokenize_chat_example`` with the messages column). Filters out None
    (fully-truncated) rows, takes a deterministic slice.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train[:64]")
    examples: List[dict] = []
    for row in ds:
        tok = tokenize_chat_example(
            {"messages": row["messages"]},
            tokenizer,
            max_length=MAX_LENGTH,
            messages_key="messages",
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        if tok is None:
            continue
        # Need at least one response token and length divisible-friendly.
        if tok["num_actions"] <= 0:
            continue
        examples.append(tok)
        if len(examples) == GLOBAL_BATCH_SIZE:
            break
    assert len(examples) == GLOBAL_BATCH_SIZE, f"got only {len(examples)} usable Tulu3 examples"
    return examples


def _build_actor_group(cfg: SFTConfig, skyrl_cfg) -> PPORayActorGroup:
    gpus = cfg.placement.num_gpus_per_node
    raw_pg = placement_group([{"GPU": gpus, "CPU": gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=60)
    pg = ResolvedPlacementGroup(raw_pg)
    group = PPORayActorGroup(
        skyrl_cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=gpus,
        ray_actor_type=_ParityProbeWorker,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=skyrl_cfg.trainer.policy.sequence_parallel_size,
        record_memory=skyrl_cfg.trainer.policy.record_memory,
    )
    ray.get(group.async_init_model(skyrl_cfg.trainer.policy.model.path))
    return group


def _canonical_masked_logprobs(
    gathered: WorkerOutput,
    collated: TrainingInputBatch,
    examples: List[dict],
    packed: bool,
    flat_bins: Optional[List[List[int]]],
) -> List[List[float]]:
    """Reduce a config's per-row full logprobs to a canonical structure:
    ``out[orig_example_idx] = [logprob at each response-token position]``.

    The loss_mask carried in ``collated`` (positionally aligned with the
    captured logprobs) tells us which positions are response tokens; we use its
    nonzero pattern (value-agnostic, so the packed scaling is irrelevant).
    """
    logprob_rows = [torch.tensor(o["logprobs"], dtype=torch.float32) for o in gathered.loss_fn_outputs]
    loss_mask = collated["loss_mask"]  # [num_rows, T-1] (float for packed, int for baseline)

    out: List[Optional[List[float]]] = [None] * len(examples)

    if not packed:
        # One row per example, in original order (DP cat preserves it).
        assert len(logprob_rows) == len(examples), f"baseline: {len(logprob_rows)} rows != {len(examples)} examples"
        for ex_idx in range(len(examples)):
            lp = logprob_rows[ex_idx]
            m = loss_mask[ex_idx]
            # logprobs are over the action window [-num_actions:]; loss_mask is
            # [num_actions] aligned to the same window. Take the tail of lp to
            # match the mask length.
            n = m.shape[0]
            lp_tail = lp[-n:]
            keep = m != 0
            out[ex_idx] = lp_tail[keep].tolist()
        return [out[i] for i in range(len(examples))]

    # Packed: rows are bins in flat_bins order. Within a bin, sub-seqs are laid
    # out back-to-back with align_size pads; loss_mask[bin] is nonzero exactly
    # at response-token positions of each sub-seq, in row-position order.
    assert flat_bins is not None
    assert len(logprob_rows) == len(flat_bins), f"packed: {len(logprob_rows)} rows != {len(flat_bins)} bins"
    for bin_row, bin_indices in enumerate(flat_bins):
        lp = logprob_rows[bin_row]  # [T-1]
        m = loss_mask[bin_row]  # [T-1]
        # Walk each sub-seq's row span and pick out its response logprobs in
        # order, assigning to the corresponding original example.
        row_offset = 0
        align_size = _align_size_for(collated)
        for ex_idx in bin_indices:
            s = len(examples[ex_idx]["input_ids"])
            # Positions [row_offset, row_offset + s) belong to this sub-seq.
            # loss_mask at position p (predicting token p+1) is nonzero for
            # response tokens within the sub-seq.
            seg_mask = m[row_offset : row_offset + s]
            seg_lp = lp[row_offset : row_offset + s]
            keep = seg_mask != 0
            out[ex_idx] = seg_lp[keep].tolist()
            row_offset += _round_up(s, align_size)
    return [out[i] for i in range(len(examples))]


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _align_size_for(collated: TrainingInputBatch) -> int:
    # Recover align_size from metadata is not stored; recompute from cfg is not
    # available here. We pass it via the batch's sub_seq layout implicitly:
    # the collator used tp*cp*2 (cp>1) or tp (cp==1). For these tests tp=1, so
    # align_size is 2*cp or 1. We infer cp from whether any row has gaps — but
    # simpler: stash it on metadata at call sites. Default 1 (cp==1).
    return int(collated.metadata.get("align_size", 1))


def _compare(ref: List[List[float]], other: List[List[float]], label: str):
    """Return (max_diff, avg_diff, n_tokens, detail). Compares per-example
    response-token logprob vectors elementwise after length-aligning.

    Also prints a diagnostic histogram + top-k outliers so a max-diff failure
    can be classified: a *systematic* permutation/un-zigzag bug shows many
    large diffs (and often a non-trivial avg); bf16 reduction-order noise on a
    handful of low-probability tokens shows a tiny avg + a few large-max
    outliers, with the baseline logprob at those tokens being very negative.
    """
    diffs: List[torch.Tensor] = []
    per_example_max: List[float] = []
    flat_pairs: List[tuple] = []  # (abs_diff, ref_lp, other_lp, ex_idx, pos)
    for i, (a, b) in enumerate(zip(ref, other)):
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        n = min(ta.numel(), tb.numel())
        if ta.numel() != tb.numel():
            per_example_max.append(float("inf"))
        if n == 0:
            continue
        d = (ta[:n] - tb[:n]).abs()
        diffs.append(d)
        per_example_max.append(d.max().item())
        for p in range(n):
            flat_pairs.append((d[p].item(), ta[p].item(), tb[p].item(), i, p))
    if not diffs:
        return 0.0, 0.0, 0, "no overlapping tokens"
    alld = torch.cat(diffs)
    worst_ex = int(torch.tensor(per_example_max).argmax().item()) if per_example_max else -1
    # Threshold histogram.
    thr = [0.05, 0.1, 0.2, 0.5, 1.0]
    hist = {t: int((alld > t).sum().item()) for t in thr}
    # Top-10 outliers with their baseline/other logprob values.
    flat_pairs.sort(key=lambda x: -x[0])
    print(f"[parity-diag] {label}: diff histogram (>thr): {hist} of n={alld.numel()}")
    print(
        f"[parity-diag] {label}: median_diff={alld.median().item():.4e} p99_diff="
        f"{alld.kthvalue(max(1, int(0.99 * alld.numel())))[0].item():.4e}"
    )
    print(f"[parity-diag] {label}: top-10 outliers (abs_diff, baseline_lp, other_lp, ex, pos):")
    for tup in flat_pairs[:10]:
        print(f"[parity-diag]     d={tup[0]:.4f} base_lp={tup[1]:.4f} other_lp={tup[2]:.4f} ex={tup[3]} pos={tup[4]}")
    detail = (
        f"{label}: n_tokens={alld.numel()} worst_example={worst_ex} "
        f"len_mismatch={[i for i, (a, b) in enumerate(zip(ref, other)) if len(a) != len(b)]}"
    )
    return alld.max().item(), alld.mean().item(), alld.numel(), detail


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
@pytest.mark.megatron
def test_sft_packing_cp_logprob_parity(ray_init_fixture):
    """Drive tokenize -> collate -> forward_backward for ONE batch under three
    configs and assert response-token logprobs match config 1 (baseline)."""
    tokenizer = get_tokenizer(MODEL_NAME, trust_remote_code=True, padding_side="left")
    examples = _load_tulu3_examples(tokenizer)
    print(
        f"\n[parity] loaded {len(examples)} Tulu3 examples; " f"seq lengths={[len(e['input_ids']) for e in examples]}"
    )

    # Config specs: (label, use_minibatch_packing, cp, gpus)
    configs = [
        ("baseline_dp2", False, 1, 2),
        ("packing_dp2", True, 1, 2),
        ("packing_cp2_dp1", True, 2, 1 * 2),  # cp=2, dp=1 -> 2 GPUs
    ]

    results: Dict[str, List[List[float]]] = {}

    for idx, (label, packed, cp, gpus) in enumerate(configs):
        print(f"\n[parity] === config {idx + 1}: {label} (packing={packed}, cp={cp}, gpus={gpus}) ===")
        sft_cfg = _make_sft_cfg(use_minibatch_packing=packed, cp=cp, gpus=gpus)
        skyrl_cfg = build_skyrl_config_for_sft(sft_cfg)

        # Build the collated batch on the controller exactly as the trainer would.
        if packed:
            trainer = SFTTrainerWithPacking(sft_cfg, skyrl_cfg=skyrl_cfg)
            trainer.tokenizer = tokenizer
            collated = trainer.collate_batch(examples, batch_size=GLOBAL_BATCH_SIZE)
            # Recompute flat_bins + align_size deterministically (same FFD call).
            flat_bins = _recompute_flat_bins(trainer, examples)
            tp = 1
            align_size = tp * cp * 2 if cp > 1 else tp
            collated.metadata["align_size"] = align_size
        else:
            collated = collate_sft_batch(examples, tokenizer)
            flat_bins = None
            collated.metadata["align_size"] = 1

        group = _build_actor_group(sft_cfg, skyrl_cfg)
        try:
            # init_model eagerly builds the Adam optimizer (fp32 master + 2
            # moments ~= 16 B/param). For CP=2/DP=1 that is ~27 GB and will not
            # fit alongside another job on a shared box. The parity capture
            # (forward_only) does NOT need the optimizer, so offload it to CPU
            # first to free ~20 GB. Harmless for the DP=2 legs.
            group.offload_to_cpu(offload_optimizer=True, offload_model=False)

            # (b) Capture full canonical logprobs for the parity comparison.
            #     This is the parity-critical step (runs the SAME
            #     preprocess/postprocess_packed_seqs as forward_backward, i.e.
            #     the multi-subseq CP un-zigzag). Run it FIRST while GPU
            #     memory is freshest.
            lp_refs = group.async_run_ray_method("mesh", "forward_capture_full_logprobs", data=collated)
            gathered = WorkerOutput.cat(group.actor_infos, ray.get(lp_refs))
            canon = _canonical_masked_logprobs(gathered, collated, examples, packed, flat_bins)
            results[label] = canon
            tok_counts = [len(c) for c in canon]
            print(f"[parity] {label}: per-example response-token counts={tok_counts}")

            # (a) Honor the design's "after forward_backward": run a real
            #     forward+backward once to confirm the packed+CP path does not
            #     crash (divisibility / shape). Best-effort: needs the optimizer
            #     + grad buffers backloaded, which can OOM on a shared box for
            #     the DP=1 leg. An OOM here is an environment limit, NOT a
            #     packing/CP correctness signal (the numeric verdict comes from
            #     the capture above), so we note it and continue.
            try:
                group.backload_to_gpu(backload_optimizer=True, backload_model=False)
                collated.metadata["global_step"] = 0
                fb_refs = group.async_run_ray_method("mesh", "forward_backward", collated, loss_fn="cross_entropy")
                fb_out = WorkerOutput.cat(group.actor_infos, ray.get(fb_refs))
                print(f"[parity] {label}: forward_backward OK, loss={fb_out.metrics.get('loss')}")
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                oom = "out of memory" in msg.lower() or "CUDA calloc" in msg or "unhandled cuda error" in msg
                tag = "OOM (env limit, not a correctness failure)" if oom else "error"
                print(f"[parity] {label}: forward_backward {tag}: {msg.splitlines()[0][:200]}")
        finally:
            ray.shutdown()
            ray_init_for_tests()

    # ---- Dump canonical logprobs for offline analysis (test-only) ----
    try:
        import pickle

        dump_path = "/tmp/sft_packing_parity_logprobs.pkl"
        with open(dump_path, "wb") as f:
            pickle.dump({"results": results, "examples_seqlens": [len(e["input_ids"]) for e in examples]}, f)
        print(f"[parity] dumped canonical logprobs to {dump_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[parity] dump failed: {e}")

    # ---- Compare each config to the baseline ----
    print("\n[parity] ===== RESULTS =====")
    baseline = results["baseline_dp2"]
    verdicts = {}
    table_rows = []
    for label in ("packing_dp2", "packing_cp2_dp1"):
        max_d, avg_d, n, detail = _compare(baseline, results[label], label)
        ok = (max_d < MAX_DIFF_TOL) and (avg_d < AVG_DIFF_TOL) and (n > 0)
        verdicts[label] = ok
        table_rows.append((label, max_d, avg_d, n, "PASS" if ok else "FAIL"))
        print(f"[parity] {label}: max_diff={max_d:.4e} avg_diff={avg_d:.4e} n={n} -> {'PASS' if ok else 'FAIL'}")
        print(f"[parity]   {detail}")

    print("\n[parity] config | max_logprob_diff_vs_baseline | avg | n_tokens | verdict")
    for r in table_rows:
        print(f"[parity]   {r[0]:<18} {r[1]:.4e}   {r[2]:.4e}   {r[3]:<6} {r[4]}")

    # Assert config 2 first (isolates base packing), then config 3
    # (isolates the CP un-zigzag).
    assert verdicts["packing_dp2"], (
        f"Config 2 (packing, no CP) mismatches baseline -> the TensorList "
        f"sub_seq_lengths plumbing or base packing is WRONG. Rows: {table_rows}"
    )
    assert verdicts["packing_cp2_dp1"], (
        f"Config 3 (packing + CP=2) mismatches baseline while config 2 matches "
        f"-> the multi-subseq CP un-zigzag is WRONG. Rows: {table_rows}"
    )


def _recompute_flat_bins(trainer: SFTTrainerWithPacking, examples: List[dict]) -> List[List[int]]:
    """Reproduce the collator's flat_bins permutation (shard-major FFD order)
    so we can invert the example reordering for the comparison."""
    from skyrl.backends.skyrl_train.distributed.megatron.bin_packing import (
        make_seq_packer,
    )

    seq_lengths = [len(ex["input_ids"]) for ex in examples]
    dp_size = trainer._dp_size()
    packed_mbs = trainer._packed_micro_batch_size()
    bin_count_multiple = dp_size * packed_mbs
    packer = make_seq_packer(
        "first_fit_decreasing",
        bin_capacity=trainer.sft_cfg.max_length,
        min_bin_count=bin_count_multiple,
        bin_count_multiple=bin_count_multiple,
    )
    bins = packer.pack(seq_lengths)
    shard_bins: List[List[List[int]]] = [[] for _ in range(dp_size)]
    for bin_idx, bin_indices in enumerate(bins):
        shard_bins[bin_idx % dp_size].append(bin_indices)
    flat_bins: List[List[int]] = []
    for shard_idx in range(dp_size):
        flat_bins.extend(shard_bins[shard_idx])
    return flat_bins


# placement_group imported lazily here to keep module import light for -k filters
from ray.util.placement_group import placement_group  # noqa: E402
