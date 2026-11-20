"""SkyRL's real training step for the hard-label MTP draft loss at TP=2 x PP=2 x CP=2 == ``process_mtp_loss``.

``test_mtp_hard_label_loss_vs_native.py`` checks the hard-label loss op by op at TP=1/2 (its docstring explains why
the two masks line up). Here ``MegatronModelWrapper.forward_backward_mini_batch`` itself runs, under Megatron's
pipeline schedule and DDP, on the packed layout CP needs. So the parts that only matter across ranks run for real: the
last stage laying out the token ids its MTP block re-embeds (``stage_needs_input_ids``), the fused LM head's hand-off
of the heads' sequence-parallel hidden states, and the CP-aware targets (``target_shift``). Layouts on 8 GPUs (DP
fills the rest): TP=2 x PP=2 x CP=2, plus PP=2 and CP=2 alone to localize a failure, and PP=4 with virtual pipeline
stages (VPP=2: the MTP block is in the last chunk of the last rank). Both MTP output paths (materialized logits,
Triton fused LM head), each with and without logprob chunking. The batch runs as one microbatch, and split as the
policy worker splits it with ``micro_train_batch_size_per_gpu=1``: one sample per microbatch, then the all-masked
padding microbatches token-based batching appends to even out DP ranks. All of it on TE's fused attention backend
and, as SkyRL runs ``flash_attn=True`` (``NVTE_FUSED_ATTN=0``), on FlashAttention. The distribution draft loss, which
runs only without PP and CP, must give the same metric and MTP gradient however the batch is split. A step whose MTP
capture records nothing must raise rather than train without the draft loss.

The reference is ``process_mtp_loss`` at TP = PP = CP = 1 on the same weights and the whole batch
(``compare_hard_label_vs_native``, which also checks SkyRL against it there), not at the same layout: under CP it
divides each rank's loss by that rank's own token count, while SkyRL's draft loss is a token mean over the whole
mini-batch. So every microbatch split must reproduce it: the MTP gradient, and the ``mtp_loss`` metric once reduced
over microbatches and DP ranks as the worker reduces it. Megatron's distributed checkpointing moves the weights, and
the reference MTP gradient saved in their place, to each layout, so every rank compares its own shard.

TE has no fp32 attention under CP, so the core attention runs in fp16 on both sides; everything else is fp32. The
policy loss gets zero advantages, so its gradient is exactly zero and the draft loss is the only gradient source.

Run (needs 8 GPUs on one node)::
    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_draft_loss_tp_pp_cp.py
"""

import functools
import os
import shutil
import tempfile
from collections import defaultdict

import pytest
import torch
from megatron.core import dist_checkpointing, tensor_parallel
from megatron.core import parallel_state as mpu
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnBackend

from skyrl.backends.skyrl_train.distributed.megatron import (
    fused_linear_logprob_triton,
)
from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
    MegatronStrategy,
)
from skyrl.backends.skyrl_train.mtp import draft_loss, hidden_capture
from skyrl.backends.skyrl_train.mtp.native_loss_patch import disable_native_mtp_loss
from skyrl.backends.skyrl_train.patches.te.patch_fa2_head_dim import (
    patch_fa2_head_dim_allowlist,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_model_wrapper as wrapper_mod,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    all_reduce_metrics,
    reduce_metrics,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import MTPLossType
from skyrl.train.utils.utils import _apply_mtp_config
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_mtp_hard_label_loss_vs_native import (
    CHUNKS,
    EXPECTED_NUM_TOKENS,
    HEADS,
    NUM_ACTIONS,
    SEQ,
    _build_tiny_mtp_gpt,
    _make_rl_batch,
    _param_group,
    compare_hard_label_vs_native,
    init_megatron,
    run_on_one_node,
    teardown_megatron,
)

WORLD_SIZE = 8
NUM_LAYERS = 8  # one trunk layer per model chunk at PP=4 x VPP=2
# (TP, PP, CP, VPP). VPP runs at PP=4: at PP=2 a rank's previous and next stage are the same rank, and megatron-core's
# shape exchange (p2p_communication._communicate_shapes) then pairs a forward shape with a backward one, so the
# interleaved schedule fails whenever packed microbatches differ in length, MTP or not.
PARALLEL_LAYOUTS = ((2, 2, 2, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 4, 1, 2))
# Rows of _make_rl_batch in each microbatch, and how many padding microbatches follow them (8 microbatches in all:
# the interleaved schedule wants a multiple of PP).
SPLITS = {"one_microbatch": ([[0, 1, 2, 3]], 0), "per_sample": ([[0], [1], [2], [3]], 4)}
# The interleaved (VPP) schedule needs at least PP microbatches.
CASES = [(layout, split) for layout in PARALLEL_LAYOUTS for split in SPLITS if layout[3] == 1 or split == "per_sample"]
LAYOUTS = list(dict.fromkeys(layout for layout, _ in CASES))
ATTENTION_BACKENDS = (AttnBackend.fused, AttnBackend.flash)
# The distribution draft loss runs only without PP / CP and on materialized logits.
DISTRIBUTION_LAYOUT = (2, 1, 1, 1)

_te_core_attention = TEDotProductAttention.forward


def _half_core_attention(self, query, key, value, *args, **kwargs):
    """TE's attention backends that support CP take bf16 / fp16 only; fp16 keeps more of fp32's precision."""
    return _te_core_attention(self, query.half(), key.half(), value.half(), *args, **kwargs).float()


def _layout_id(layout) -> str:
    return "tp{}-pp{}-cp{}-vpp{}".format(*layout)


def _init_layout(tp: int, pp: int, cp: int, vpp: int, seed: int = 1234):
    mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp if vpp > 1 else None,
        context_parallel_size=cp,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _build_chunks(device, attention_backend, vpp: int) -> list:
    """This rank's model chunks, one per virtual pipeline stage."""
    vp_stages = range(vpp) if vpp > 1 else [None]
    return [_build_tiny_mtp_gpt(device, attention_backend, NUM_LAYERS, vp_stage=vp_stage) for vp_stage in vp_stages]


def _load_chunks(chunks: list, ckpt_dir: str):
    """Load every chunk in one call, as megatron-core does: each call must cover the checkpoint's tensors."""
    state = dist_checkpointing.load({f"model{i}": m.sharded_state_dict() for i, m in enumerate(chunks)}, ckpt_dir)
    for i, model in enumerate(chunks):
        model.load_state_dict(state[f"model{i}"])


def _trainer_cfg(head: str, chunk: int, loss_type: MTPLossType = MTPLossType.HARD_LABEL):
    """The draft loss on the packed layout, configured as the policy worker gets it."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.fused_lm_head_logprob = head == "fused_triton"
    cfg.trainer.fused_lm_head_logprob_backend = "triton"
    cfg.trainer.policy.megatron_config.mtp_loss_chunk_size = chunk
    cfg.trainer.algorithm.use_kl_loss = False
    # The wrapper gets the "regular" PPO loss below, which checks the config names it (the default is "dppo").
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_type = loss_type
    # Not 1, so a dropped weight cannot pass; a power of two, like the reference's MTP_LOSS_SCALING_FACTOR, so the
    # fp16 attention backward rounds the same on both sides.
    cfg.trainer.mtp.loss_weight = 0.25
    _apply_mtp_config(cfg)
    return cfg.trainer


def _micro_batches(batch, split: str) -> list:
    """``batch`` as the policy worker hands its microbatches to ``forward_backward_mini_batch``: on the host.

    A padding microbatch is as token-based batching builds one: a single real token, loss mask zero.
    """
    rows, num_padding = SPLITS[split]
    host = {k: v.cpu() for k, v in batch.items()}
    micros = [{k: v[r] for k, v in host.items()} for r in rows]
    for _ in range(num_padding):
        pad = {k: torch.zeros_like(v[:1]) for k, v in host.items()}
        pad["attention_mask"][:, 0] = 1
        micros.append(pad)
    for i, m in enumerate(micros):
        zeros = torch.zeros_like(m["loss_mask"])
        m.update(
            num_actions=NUM_ACTIONS,
            old_action_log_probs=zeros,
            base_action_log_probs=None,
            advantages=zeros,  # the policy loss then puts exactly zero gradient on every parameter
            rollout_action_logprobs=None,
            response_mask=None,
            rollout_expert_indices=None,
            num_microbatches=len(micros),
            num_real_microbatches=len(rows),
            is_padding_batch=i >= len(rows),
        )
    return micros


def _train_step(
    ddps, chunks, split, head, chunk, batch, ref_loss, ref_grads, native=None, loss_type=MTPLossType.HARD_LABEL
):
    """One ``forward_backward_mini_batch`` and the grad sync ``optim_step`` runs, compared with the reference.

    Returns the comparison and this step's MTP gradient (``d(loss)/d(theta)``, the loss weight divided out).
    """
    trainer_cfg = _trainer_cfg(head, chunk, loss_type)
    micros = _micro_batches(batch, split)
    wrapper = wrapper_mod.MegatronModelWrapper(trainer_cfg, ddps, policy_loss_fn=PolicyLossRegistry.get("regular"))
    aligned, orig_align = [], draft_loss.mtp_action_aligned

    def align_spy(*args, **kwargs):
        lp, valid = orig_align(*args, **kwargs)
        aligned.append((lp.detach(), valid.bool()))
        return lp, valid

    draft_loss.mtp_action_aligned = align_spy
    try:
        metrics = wrapper.forward_backward_mini_batch(
            [dict(m) for m in micros], seq_len=SEQ, micro_batch_size=micros[0]["sequences"].shape[0]
        )
    finally:
        draft_loss.mtp_action_aligned = orig_align
    wrapper.run_pending_grad_sync()

    weight = trainer_cfg.policy.megatron_config.mtp_loss_weight
    sq = defaultdict(float, dict.fromkeys(("mtp", "trunk", "embedding", "output_layer"), 0.0))
    max_rel, worst, sq_diff, sq_ref, mtp_grads = 0.0, None, 0.0, 0.0, {}
    for i, model in enumerate(chunks):
        for n, p in model.named_parameters():
            sq[_param_group(n)] += float(p.main_grad.double().pow(2).sum())
            if _param_group(n) == "mtp":
                mtp_grads[(i, n)] = p.main_grad / weight
            if (i, n) in ref_grads:
                ref_g = ref_grads[(i, n)]
                diff = mtp_grads[(i, n)] - ref_g
                rel = diff.abs().max().item() / max(ref_g.abs().max().item(), 1e-30)
                if rel > max_rel:
                    max_rel, worst = rel, n
                sq_diff += float(diff.double().pow(2).sum())
                sq_ref += float(ref_g.double().pow(2).sum())
    mtp_grads = {k: g.clone() for k, g in mtp_grads.items()}
    for ddp in ddps:
        ddp.zero_grad_buffer()
    # The worker's reduction: over its real microbatches, then over the DP group.
    status = reduce_metrics(
        {"mtp_loss": [m["mtp_loss"] for m, mb in zip(metrics, micros) if not mb["is_padding_batch"]]},
        sum_loss_metrics=True,
    )
    dp_group = mpu.get_data_parallel_group(with_context_parallel=False)
    strategy = MegatronStrategy(trainer_cfg.policy.megatron_config)
    loss = all_reduce_metrics(status, strategy, group=dp_group, sum_loss_metrics=True)["mtp_loss"]
    r = dict(
        loss=loss,
        loss_abs_diff=abs(loss - ref_loss),
        norms={k: v**0.5 for k, v in sq.items()},
        mtp_grad_max_rel_diff=max_rel,
        mtp_grad_max_rel_diff_param=worst,
        mtp_grad_rel_l2_diff=(sq_diff / sq_ref) ** 0.5 if ref_grads else None,
    )
    if native is not None and aligned:  # the last stage: each microbatch's scores, against Megatron's for its rows
        assert len(aligned) == len(micros), (len(aligned), len(micros))
        ref_ce, ref_mask = native["ce"][:, -NUM_ACTIONS:], native["mask"][:, -NUM_ACTIONS:].bool()
        real = list(zip(aligned, SPLITS[split][0]))
        r.update(
            mask_equal=all(torch.equal(valid, ref_mask[rows]) for (_, valid), rows in real),
            num_tokens=[int(valid.sum()) for _, valid in aligned],
            per_token_ce_max_abs_diff=max(float((-lp - ref_ce[rows])[valid].abs().max()) for (lp, valid), rows in real),
        )
    return r, mtp_grads


def compare_training_step_vs_native(ckpt_dir: str, attention_backend, cases=CASES) -> dict:
    """Score one RL batch with SkyRL's training step for each (layout, split) against the TP=PP=CP=1 reference.

    Collective over the world (8 ranks for the layouts above). ``ckpt_dir`` is an empty directory every rank sees.
    """
    device = torch.device("cuda")
    batch = _make_rl_batch(device)
    weights_dir, grads_dir = os.path.join(ckpt_dir, "weights"), os.path.join(ckpt_dir, "grads")
    if torch.distributed.get_rank() == 0:
        os.makedirs(weights_dir)
        os.makedirs(grads_dir)
    torch.distributed.barrier()

    # The reference: process_mtp_loss on the whole model, every rank a DP replica.
    _init_layout(1, 1, 1, 1)
    (model,) = _build_chunks(device, attention_backend, vpp=1)
    ref = compare_hard_label_vs_native(
        model, batch, num_actions=NUM_ACTIONS, layouts=("packed",), chunks=CHUNKS[:1], keep_native=True
    )
    native = ref.pop("native")["packed"]
    dist_checkpointing.save(model.sharded_state_dict(), weights_dir)
    # Save d(mean CE)/d(theta) in the weights' place: loaded at another layout, it is each rank's shard.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in native["mtp_grads"]:
                p.copy_(native["mtp_grads"][n] / ref["scale"])
            else:
                p.zero_()
    dist_checkpointing.save(model.sharded_state_dict(), grads_dir)
    out = {"reference": dict(loss=native["loss"], num_tokens=native["num_tokens"], tp1=ref)}
    del model

    disable_native_mtp_loss()  # as the policy worker does before its first forward
    for layout in dict.fromkeys(layout for layout, _ in cases):
        _init_layout(*layout)
        chunks = _build_chunks(device, attention_backend, vpp=layout[3])
        _load_chunks(chunks, grads_dir)
        ref_grads = {
            (i, n): p.detach().clone()
            for i, model in enumerate(chunks)
            for n, p in model.named_parameters()
            if _param_group(n) == "mtp"
        }
        _load_chunks(chunks, weights_dir)
        # Without the distributed optimizer every rank keeps its parameters' whole reduced gradient. Chunks after the
        # first get no bucketing, as megatron-core wraps them.
        ddps = [
            DistributedDataParallel(m.config, DistributedDataParallelConfig(), m, disable_bucketing=i > 0)
            for i, m in enumerate(chunks)
        ]
        try:
            results = {
                f"{split}/{head}/chunk{chunk}": _train_step(
                    ddps, chunks, split, head, chunk, batch, native["loss"], ref_grads, native
                )[0]
                for case_layout, split in cases
                if case_layout == layout
                for head in HEADS
                for chunk in CHUNKS
            }
        except Exception as e:
            e.add_note(f"at (TP, PP, CP, VPP) = {layout}, rank {torch.distributed.get_rank()}")
            raise
        out[layout] = dict(owns_mtp=bool(ref_grads), cases=results)
        del ddps, chunks

    # The distribution draft loss: every split must reproduce the whole batch's metric and MTP gradient.
    _init_layout(*DISTRIBUTION_LAYOUT)
    chunks = _build_chunks(device, attention_backend, vpp=1)
    _load_chunks(chunks, weights_dir)
    (model,) = chunks
    ddps = [DistributedDataParallel(model.config, DistributedDataParallelConfig(), model)]
    step = functools.partial(_train_step, ddps, [model], head="logits", chunk=CHUNKS[0], batch=batch)
    whole, whole_grads = step("one_microbatch", ref_loss=0.0, ref_grads={}, loss_type=MTPLossType.DISTRIBUTION)
    split, _ = step("per_sample", ref_loss=whole["loss"], ref_grads=whole_grads, loss_type=MTPLossType.DISTRIBUTION)
    out["distribution"] = dict(owns_mtp=bool(whole_grads), whole=whole, per_sample=split)
    return out


def _training_step_rank(world_size: int, rank: int, init_method: str, *, attention_backend) -> dict:
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=init_method, world_size=world_size, rank=rank)
    # This process only: fp16 core attention. As the worker sets it, flash_attn=True exports NVTE_FUSED_ATTN=0; the
    # fused backend, which SkyRL uses otherwise, needs it on (the test env turns it off).
    TEDotProductAttention.forward = _half_core_attention
    os.environ["NVTE_FUSED_ATTN"] = "1" if attention_backend == AttnBackend.fused else "0"
    patch_fa2_head_dim_allowlist()  # as the worker does before it builds the model
    # IEEE fp32 matmuls, as compare_hard_label_vs_native sets for its own runs.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = True
    ckpt_dir = [tempfile.mkdtemp() if rank == 0 else None]  # the ranks share one node
    torch.distributed.broadcast_object_list(ckpt_dir)
    try:
        out = compare_training_step_vs_native(ckpt_dir[0], attention_backend)
        torch.distributed.barrier()
    finally:  # no collective on the error path: the error reaches run_on_one_node, which kills the blocked peers
        if rank == 0:
            shutil.rmtree(ckpt_dir[0], ignore_errors=True)
    teardown_megatron()
    return out


@pytest.fixture(scope="module", params=ATTENTION_BACKENDS, ids=lambda backend: backend.name)
def training_step(request, module_scoped_ray_init_fixture):
    """``compare_training_step_vs_native`` on 8 ranks of one node, per attention backend; one result per rank."""
    backend = request.param
    results = run_on_one_node(functools.partial(_training_step_rank, attention_backend=backend), WORLD_SIZE)
    ref = results[0]["reference"]
    print(f"\n===== {backend.name} reference (TP=PP=CP=1): loss {ref['loss']}, {ref['num_tokens']} tokens =====")
    for case, r in ref["tp1"]["skyrl"].items():
        print(f"SkyRL there, {case}: {r}")
    for layout in LAYOUTS:
        for rank, res in enumerate(results):
            if res[layout]["owns_mtp"]:
                print(f"\n===== {backend.name} TP/PP/CP/VPP={layout} rank {rank} (owns the MTP layer) =====")
                for case, r in res[layout]["cases"].items():
                    print(f"{case}: {r}")
    for rank, res in enumerate(results):
        print(f"\n===== {backend.name} distribution loss, {_layout_id(DISTRIBUTION_LAYOUT)} rank {rank} =====")
        print(res["distribution"])
    return results


@pytest.mark.megatron
@pytest.mark.parametrize("layout,split", CASES, ids=lambda x: _layout_id(x) if isinstance(x, tuple) else x)
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_mtp_hard_label_training_step_matches_process_mtp_loss(training_step, layout, split, head, chunk):
    ref = training_step[0]["reference"]
    runs = [(rank, res[layout]) for rank, res in enumerate(training_step)]
    # The MTP layer lives on the last stage, one TP x CP x DP group: 8 / PP ranks.
    assert sum(run["owns_mtp"] for _, run in runs) == WORLD_SIZE // layout[1]
    # Without CP the forward is bitwise the reference's; the fp16 attention backward moves the worst MTP gradient
    # entry by up to ~1.8e-4. CP splits that attention across ranks: per-token CE moves by up to ~7.5e-4, the loss
    # by ~4.5e-5 and the worst gradient entry by ~1.1e-3.
    loss_tol, ce_tol, grad_tol = (2e-4, 3e-3, 5e-3) if layout[2] > 1 else (1e-5, 1e-5, 1e-3)
    for rank, run in runs:
        r = run["cases"][f"{split}/{head}/chunk{chunk}"]
        # The mtp_loss metric every stage reports (broadcast from the last stage), reduced as the worker does.
        assert r["loss_abs_diff"] < loss_tol, (rank, r)
        if not run["owns_mtp"]:
            continue
        # Same targets and mask per microbatch, exactly; padding microbatches score nothing.
        assert r["mask_equal"], (rank, r)
        assert sum(r["num_tokens"]) == ref["num_tokens"] == EXPECTED_NUM_TOKENS, (rank, r)
        # Same per-token CE and MTP-layer gradient shard, once the loss weight is divided out.
        assert r["per_token_ce_max_abs_diff"] < ce_tol, (rank, r)
        assert r["mtp_grad_max_rel_diff"] < grad_tol, (rank, r)


@pytest.mark.megatron
@pytest.mark.parametrize("layout", LAYOUTS, ids=_layout_id)
def test_mtp_hard_label_training_step_gradient_reaches_only_the_head(training_step, layout):
    for rank, res in enumerate(training_step):
        for case, r in res[layout]["cases"].items():
            assert (r["norms"]["mtp"] > 0) == res[layout]["owns_mtp"], (rank, case, r)
            for group in ("trunk", "embedding", "output_layer"):
                assert r["norms"][group] == 0.0, (rank, case, group, r)


@pytest.mark.megatron
def test_mtp_distribution_training_step_is_a_mini_batch_token_mean(training_step):
    """The distribution draft loss, one sample per microbatch plus padding microbatches, reproduces the whole batch as
    one microbatch: its mtp_loss metric (reduced as the worker does) and its MTP gradient."""
    for rank, res in enumerate(training_step):
        run = res["distribution"]
        assert run["owns_mtp"], (rank, run)
        assert run["per_sample"]["loss_abs_diff"] < 1e-5, (rank, run)
        assert run["per_sample"]["mtp_grad_max_rel_diff"] < 1e-3, (rank, run)
        for case in ("whole", "per_sample"):
            assert run[case]["norms"]["mtp"] > 0, (rank, case, run)
            for group in ("trunk", "embedding", "output_layer"):
                assert run[case]["norms"][group] == 0.0, (rank, case, group, run)


def _missing_heads_rank(world_size: int, rank: int, init_method: str, *, loss_type: MTPLossType):
    """One training step at TP = PP = CP = 1 whose MTP capture records nothing, as if the block's forward never ran.
    Returns the AssertionError's message, or None if the step raised none."""
    device = init_megatron(world_size, rank, init_method)
    # forward_step then hands loss_func no head outputs (this process only; run_on_one_node kills it afterwards).
    hidden_capture.MTPHiddenCapture.compute_student_hidden_states = lambda self: None
    disable_native_mtp_loss()
    try:
        model = _build_tiny_mtp_gpt(device)
        ddps = [DistributedDataParallel(model.config, DistributedDataParallelConfig(), model)]
        trainer_cfg = _trainer_cfg("logits", CHUNKS[0], loss_type)
        wrapper = wrapper_mod.MegatronModelWrapper(trainer_cfg, ddps, policy_loss_fn=PolicyLossRegistry.get("regular"))
        micros = _micro_batches(_make_rl_batch(device), "one_microbatch")
        try:
            wrapper.forward_backward_mini_batch(micros, seq_len=SEQ, micro_batch_size=micros[0]["sequences"].shape[0])
        except AssertionError as e:
            return str(e)
        return None
    finally:
        teardown_megatron()


@pytest.mark.megatron
@pytest.mark.parametrize(
    ("loss_type", "missing"),
    [
        (MTPLossType.HARD_LABEL, "no MTP head outputs for the hard-label draft loss"),
        (MTPLossType.DISTRIBUTION, "no MTP head logits for the distribution draft loss"),
    ],
)
def test_mtp_training_step_fails_without_the_heads_outputs(module_scoped_ray_init_fixture, loss_type, missing):
    (message,) = run_on_one_node(functools.partial(_missing_heads_rank, loss_type=loss_type), 1)
    assert message is not None and missing in message, message
    assert "were not captured on the last pipeline stage" in message, message
