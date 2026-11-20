"""SkyRL's hard-label MTP draft loss == Megatron-Core's ``process_mtp_loss``, but only the head gets gradient.

On one tiny fp32 ``GPTModel`` with one MTP layer, the same token ids and the same RL loss mask (the last
``NUM_ACTIONS`` columns are "generated"):

1. **Same loss.** ``mtp_hard_label_draft_loss`` (depth 1) equals the value
   ``process_mtp_loss`` logs to ``MTPLossLoggingHelper``, ``sum(mask * CE) / num_tokens``, token by token:
   same targets, mask, per-token CE and denominator.
2. **Same gradient on the MTP layer's own parameters**, once Megatron's ``mtp_loss_scaling_factor /
   mtp_num_layers`` (applied through ``MTPLossAutoScaler``) is divided out.
3. **No gradient anywhere else.** SkyRL leaves the decoder trunk, the embedding and the output layer at
   exactly zero gradient; Megatron's stock loss trains all three. We also record ``mtp_detach_heads=True``,
   which in the pinned megatron-core detaches the block's trunk input, the re-embedding and the output
   weight, isolating the head as SkyRL does.

Covered: the materialized-logits and fused-LM-head (Triton, IEEE fp32) paths, packed (THD) and de-padded
inputs, each with and without logprob chunking, at TP=1 and at TP=2 with sequence parallelism (on by default
in SkyRL whenever TP > 1), checked on every rank. Both sides run their real code: SkyRL with
``disable_native_mtp_loss()`` as in production, Megatron with that patch undone. ``compare_hard_label_vs_native``
is a plain function, so it can also be pointed at a pretrained MTP checkpoint.

Why the masks line up. Take one sequence's real tokens ``x_0 .. x_{L-1}`` and let ``g_j = 1`` mark
trainable generated tokens (the RL loss mask on its own token; the prompt is 0).

* Megatron, ``labels=None`` (the RL path), takes the mask in token coordinates and rolls labels and mask
  in lockstep, so depth 1 has ``mtp_labels[t] = x_{t+2}`` and ``mask[t] = g_{t+2}`` for ``t <= L-3``,
  and zero-filled ``mask[L-2] = mask[L-1] = 0``: the targets ``j in [2, L-1]`` with ``g_j = 1``. Packed
  rolls run per segment over ``cu_seqlens_q_padded``.
* SkyRL, ``target_shift = 2``: column ``s`` scores ``x_{s+2}``, and ``mtp_action_aligned`` stops at
  ``end = cols - 1``, dropping the two columns whose targets wrap (Megatron's zero-filled slots). Action
  token ``j`` counts iff ``g_j = 1`` and position ``j - 2`` is a real token of the same row: the same
  target set and denominator. ``j = 1`` (after a 1-token prompt, row 2 below) has no head position on
  either side.

A caller that passes ``labels`` (the SFT convention) must pass the mask already in label coordinates.
SkyRL's production forward passes neither, so the native loss, were it not disabled, would score every
real token, prompt included.

Run (needs 2 GPUs on one node)::
    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_hard_label_loss_vs_native.py
"""

import functools
import socket
import sys
from contextlib import contextmanager

import pytest
import ray
import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt import gpt_model as gpt_model_mod
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer import multi_token_prediction as mtp_mod
from megatron.core.transformer.enums import AttnBackend
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.distributed.megatron import (
    fused_linear_logprob_triton,
)
from skyrl.backends.skyrl_train.distributed.megatron.fused_lm_head import (
    call_model_with_fused_lm_head,
    fused_lm_head_output_processor,
)
from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
    to_te_attention_mask,
)
from skyrl.backends.skyrl_train.mtp import draft_loss
from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
from skyrl.backends.skyrl_train.mtp.hidden_capture import maybe_capture_mtp_hidden
from skyrl.backends.skyrl_train.mtp.native_loss_patch import (
    _skyrl_skip_native_mtp_loss,
    disable_native_mtp_loss,
)
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_model_wrapper as wrapper_mod,
)
from skyrl.train.config.config import MTPLossType
from skyrl.train.fused_lm_head import FusedLmHeadBackend
from skyrl.train.utils import get_ray_pg_ready_with_timeout

VOCAB = 512
SEQ = 24
NUM_ACTIONS = 12
HIDDEN = 128  # the Triton fused LM-head kernel needs hidden_size % 128 == 0
MTP_LOSS_SCALING_FACTOR = 0.25  # not Megatron's 0.1 default, so a missing rescale cannot pass
LAYOUTS = ("packed", "unpacked")
HEADS = ("logits", "fused_triton")
CHUNKS = (1024, 8)  # production default (one chunk here) and a size that splits the sequence
RANKS = ((1, 0), (2, 0), (2, 1))  # (TP size, TP rank): every rank of a TP=1 and a TP=2 group
# Depth-1 targets that count, per row of _make_rl_batch: 11 (12 generated, one masked) + 9 + 11 (the first
# generated token follows a 1-token prompt: no position two back) + 2 (last real token masked).
EXPECTED_NUM_TOKENS = 11 + 9 + 11 + 2


def _make_rl_batch(device):
    """A left-padded prompt + right-padded response batch, in SkyRL's ``[batch, seq]`` layout.

    Rows cover: a response that runs to the last column, with an interior masked token (row 0); left
    and right padding (row 1); a 1-token prompt, whose first generated token has no position two back
    (row 2); a response whose last real token is masked out (row 3).
    """

    gen = torch.Generator().manual_seed(7)
    batch = 4
    prompt_start = SEQ - NUM_ACTIONS
    # (first real column, number of real response tokens, loss-mask zeros within the response)
    rows = [(0, 12, [5]), (4, 9, []), (11, 12, []), (2, 3, [2])]
    sequences = torch.zeros(batch, SEQ, dtype=torch.long)
    attention_mask = torch.zeros(batch, SEQ, dtype=torch.long)
    loss_mask = torch.zeros(batch, NUM_ACTIONS, dtype=torch.float32)
    for r, (start, n_resp, masked) in enumerate(rows):
        end = prompt_start + n_resp
        sequences[r, start:end] = torch.randint(1, VOCAB, (end - start,), generator=gen)
        attention_mask[r, start:end] = 1
        loss_mask[r, :n_resp] = 1.0
        loss_mask[r, masked] = 0.0
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "sequences": sequences.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
        "loss_mask": loss_mask.to(device),
    }


def _param_group(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("embedding."):
        return "embedding"
    if name.startswith("output_layer."):
        return "output_layer"
    if name.startswith("decoder."):
        return "trunk"
    return "other"


def init_megatron(tp: int, rank: int, init_method: str, seed: int = 1234):
    """NCCL world of ``tp`` processes, one GPU each, as TP = ``tp``, PP = CP = 1 parallel state."""

    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=init_method, world_size=tp, rank=rank)
    mpu.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)
    return torch.device("cuda")


def teardown_megatron():
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()


def compare_hard_label_vs_native(
    model, batch, *, num_actions, layouts=LAYOUTS, heads=HEADS, chunks=CHUNKS, keep_native=False
) -> dict:
    """Score ``batch`` with Megatron's native MTP loss and with SkyRL's hard-label draft loss; compare them.

    ``model`` is a bare Megatron ``GPTModel`` with one MTP layer, on this GPU, in train mode; under TP it
    is this rank's shard and every rank calls this with the same ``batch``. ``batch`` holds SkyRL's
    ``[batch, seq]`` ``sequences`` / ``attention_mask`` / ``position_ids`` and the ``[batch, num_actions]``
    ``loss_mask``. Gradients are compared per rank, on this rank's shard.

    Megatron runs first, with stock ``process_mtp_loss`` and ``MTPLossAutoScaler``, once as configured
    and once with ``mtp_detach_heads=True``. SkyRL then runs with ``disable_native_mtp_loss()`` applied as
    in production, and the patch is undone afterwards. Only the MTP-layer gradients are kept as tensors;
    every other parameter group is reduced to its norm on the spot. With ``keep_native``, ``out["native"]``
    also holds each layout's stock Megatron run as tensors: ``mtp_grads``, and ``ce`` / ``mask`` / ``labels``
    in ``[batch, seq]`` target coordinates.
    """

    native_process_mtp_loss = mtp_mod.process_mtp_loss
    assert native_process_mtp_loss is not _skyrl_skip_native_mtp_loss, "SkyRL patch leaked into this worker"
    assert gpt_model_mod.process_mtp_loss is native_process_mtp_loss
    assert model.config.mtp_num_layers == 1, model.config.mtp_num_layers

    @contextmanager
    def skyrl_native_loss_patch():
        """``disable_native_mtp_loss()`` as production applies it, undone on exit."""
        autoscaler = mtp_mod.MTPLossAutoScaler
        had_apply = "apply" in vars(autoscaler)
        orig_apply = vars(autoscaler).get("apply")
        bound = []
        for mod in list(sys.modules.values()):
            try:
                if getattr(mod, "process_mtp_loss", None) is native_process_mtp_loss:
                    bound.append(mod)
            except Exception:
                continue
        disable_native_mtp_loss()
        try:
            yield
        finally:
            for mod in bound:
                mod.process_mtp_loss = native_process_mtp_loss
            mtp_mod.process_mtp_loss = native_process_mtp_loss
            if had_apply:
                autoscaler.apply = orig_apply
            else:
                del autoscaler.apply
            assert gpt_model_mod.process_mtp_loss is native_process_mtp_loss

    device = batch["sequences"].device
    named_params = list(model.named_parameters())
    groups_present = sorted({_param_group(n) for n, _ in named_params})
    assert groups_present == ["embedding", "mtp", "output_layer", "trunk"], groups_present
    configs = [m.config for m in model.modules() if isinstance(getattr(m, "config", None), TransformerConfig)]

    sequences, attention_mask = batch["sequences"], batch["attention_mask"]
    position_ids, loss_mask = batch["position_ids"], batch["loss_mask"]
    seq_len = sequences.shape[1]
    am_bool = attention_mask.to(bool)

    def model_inputs(layout):
        """The trunk inputs SkyRL's forward_step builds for this layout (CP=1)."""
        if layout == "packed":
            ids, psp = preprocess_packed_seqs(sequences, am_bool, pre_process=True)
            pos = preprocess_packed_seqs(position_ids, am_bool, pre_process=True)[0]
            targets = wrapper_mod._pack_sequence_values(sequences, am_bool, psp)
            return dict(ids=ids, pos=pos, new_am=None, psp=psp, packed_targets=targets)
        ids, new_am, pos = remove_left_padding(sequences, am_bool, position_ids, pre_process=True)
        return dict(ids=ids, pos=pos, new_am=new_am, psp=None, packed_targets=None)

    def to_model_layout(values, inp):
        """Place a ``[batch, seq]`` tensor in the layout the model sees ([1, T] packed or de-padded)."""
        if inp["psp"] is not None:
            return wrapper_mod._pack_sequence_values(values, am_bool, inp["psp"])
        return remove_left_padding(values, am_bool, position_ids, pre_process=True)[0]

    def head_to_target_coords(values, inp):
        """Move a per-head-position tensor ``[t]`` to its depth-1 target ``[t + 2]``, back in ``[batch, seq]``.

        Shifting the flat packed buffer crosses segment seams; that is only correct if Megatron zeroed each
        segment's last two slots, which is part of what the mask comparison checks.
        """
        shifted = torch.zeros_like(values)
        shifted[..., 2:] = values[..., :-2]
        if inp["psp"] is None:
            return recover_left_padding(shifted, inp["new_am"], am_bool, seq_len)
        cu = inp["psp"].cu_seqlens_q_padded.to(torch.long)
        # Packed slot of row r's real column c: segment start cu[r] + c's 0-based ordinal among the row's real
        # tokens (cumsum - 1). Only the am_bool entries are read.
        packed_idx = cu[:-1].unsqueeze(1) + am_bool.long().cumsum(dim=1) - 1
        out = torch.zeros(sequences.shape, dtype=values.dtype, device=device)
        out[am_bool] = shifted[0, packed_idx[am_bool]]
        return out

    def collect_grads():
        """(MTP-layer grads, per-group grad norms, per-group "any grad tensor at all")."""
        mtp_grads, sq, connected = {}, {k: 0.0 for k in groups_present}, {k: False for k in groups_present}
        for n, p in named_params:
            if p.grad is None:
                continue
            group = _param_group(n)
            sq[group] += float(p.grad.detach().double().pow(2).sum())
            connected[group] = True
            if group == "mtp":
                mtp_grads[n] = p.grad.detach().clone()
        return mtp_grads, {k: v**0.5 for k, v in sq.items()}, connected

    def megatron_native(layout, detach_heads):
        assert gpt_model_mod.process_mtp_loss is native_process_mtp_loss
        assert mtp_mod.MTPLossAutoScaler.apply.__name__ == "apply"
        inp = model_inputs(layout)
        full_mask = torch.zeros(sequences.shape, dtype=torch.float32, device=device)
        full_mask[:, -num_actions:] = loss_mask
        token_mask = to_model_layout(full_mask, inp)  # labels=None: the mask goes in token coordinates

        for c in configs:
            c.mtp_detach_heads = detach_heads
        model.zero_grad(set_to_none=True)
        mtp_mod.MTPLossLoggingHelper.clean_metrics_in_tracker()
        mtp_mod.MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=device))

        seen = {}
        orig_counts = mtp_mod._compute_mtp_acceptance_counts

        def counts_spy(mtp_logits, mtp_labels, layer_loss_mask, *args, **kwargs):
            seen["labels"] = mtp_labels.detach().clone()
            seen["mask"] = layer_loss_mask.detach().clone()
            return orig_counts(mtp_logits, mtp_labels, layer_loss_mask, *args, **kwargs)

        def lm_loss_spy(labels, logits):
            ce = type(model).compute_language_model_loss(model, labels, logits)
            seen["ce"] = ce.detach().clone()
            return ce

        mtp_mod._compute_mtp_acceptance_counts = counts_spy
        model.compute_language_model_loss = lm_loss_spy  # only process_mtp_loss calls it (labels=None)
        try:
            logits = model(
                inp["ids"],
                inp["pos"],
                to_te_attention_mask(inp["new_am"]),
                packed_seq_params=inp["psp"],
                loss_mask=token_mask,
            )
            # The main loss contributes exactly nothing; MTPLossAutoScaler injects the MTP gradient.
            (logits.sum() * 0.0).backward()
            del logits
        finally:
            mtp_mod._compute_mtp_acceptance_counts = orig_counts
            del model.compute_language_model_loss
            for c in configs:
                c.mtp_detach_heads = False
        tracker = mtp_mod.MTPLossLoggingHelper.tracker
        mtp_grads, norms, _ = collect_grads()
        model.zero_grad(set_to_none=True)
        return dict(
            loss=float(tracker["loss_values"][0]),
            num_tokens=float(tracker["total_values"][0]),
            mask=head_to_target_coords(seen["mask"].float(), inp),
            labels=head_to_target_coords(seen["labels"], inp),
            ce=head_to_target_coords(seen["ce"].float(), inp),
            mtp_grads=mtp_grads,
            norms=norms,
        )

    def skyrl_hard_label(layout, head, chunk):
        assert gpt_model_mod.process_mtp_loss is _skyrl_skip_native_mtp_loss
        inp = model_inputs(layout)
        fused = head == "fused_triton"
        model.zero_grad(set_to_none=True)
        mtp_mod.MTPLossLoggingHelper.clean_metrics_in_tracker()
        data = {"attention_mask": attention_mask, "sub_seq_lengths_list": None}

        # forward_step's MTP capture (megatron_model_wrapper.py).
        with maybe_capture_mtp_hidden(model, True) as capture:
            args = (inp["ids"], inp["pos"], to_te_attention_mask(inp["new_am"]))
            if fused:
                op_ctx = {}
                call_model_with_fused_lm_head(
                    model,
                    *args,
                    packed_seq_params=inp["psp"],
                    output_processor=fused_lm_head_output_processor,
                    output_processor_context=op_ctx,
                )
                data["lm_head_weight"] = op_ctx["lm_head_weight"]
            else:
                model(*args, packed_seq_params=inp["psp"])
            student_hidden = capture.compute_student_hidden_states()
            student_model = capture.model

        def depad(t):
            return recover_left_padding(t, inp["new_am"], am_bool, seq_len)

        hidden_list = logits_list = None
        if fused:
            hidden_list = [
                fused_lm_head_output_processor(hidden_states=h, output_layer=student_model.output_layer)
                for h in student_hidden
            ]
            if inp["psp"] is None:
                hidden_list = [depad(h) for h in hidden_list]
        else:
            logits_list = project_mtp_hidden_to_logits(student_hidden, student_model)
            if inp["psp"] is None:
                logits_list = [depad(sl) for sl in logits_list]
        # The batch is the whole mini-batch, so its own valid-token counts make the loss its token mean.
        token_counts = draft_loss.mtp_draft_token_counts(
            [{"attention_mask": attention_mask, "num_actions": num_actions, "loss_mask": loss_mask}],
            len(student_hidden),
            loss_type=MTPLossType.HARD_LABEL,
            packed=inp["psp"] is not None,
        )

        aligned = []
        orig_align = draft_loss.mtp_action_aligned

        def align_spy(*args, **kwargs):
            lp, valid = orig_align(*args, **kwargs)
            aligned.append((lp.detach().clone(), valid.detach().clone()))
            return lp, valid

        draft_loss.mtp_action_aligned = align_spy
        try:
            loss = draft_loss.mtp_hard_label_draft_loss(
                hidden_list if fused else logits_list,
                fused=fused,
                lm_head_weight=data.get("lm_head_weight"),
                sequences=sequences,
                attention_mask=data["attention_mask"],
                sub_seq_lengths=data["sub_seq_lengths_list"],
                num_actions=num_actions,
                loss_mask=loss_mask,
                packed_seq_params=inp["psp"],
                packed_targets=inp["packed_targets"],
                tp_grp=mpu.get_tensor_model_parallel_group(),
                tp_rank=mpu.get_tensor_model_parallel_rank(),
                chunk_size=chunk,
                fused_backend=FusedLmHeadBackend.TRITON,
                token_counts=token_counts,
            )
        finally:
            draft_loss.mtp_action_aligned = orig_align
        loss.backward()
        del hidden_list, logits_list, student_hidden
        tracker = mtp_mod.MTPLossLoggingHelper.tracker
        native_ran = bool("loss_values" in tracker and tracker["loss_values"].abs().sum() > 0)
        mtp_grads, norms, connected = collect_grads()
        model.zero_grad(set_to_none=True)
        lp, valid = aligned[0]
        return dict(
            loss=float(loss),
            lp=lp,
            valid=valid,
            mtp_grads=mtp_grads,
            norms=norms,
            connected=connected,
            native_ran=native_ran,
        )

    # IEEE fp32 everywhere a matmul differs between the two sides (output layer vs Triton kernel).
    prev_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    prev_force_ieee = fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = True
    scale = model.config.mtp_loss_scaling_factor / model.config.mtp_num_layers
    out = {"megatron": {}, "skyrl": {}, "scale": scale}
    try:
        megatron = {
            layout: {
                "stock": megatron_native(layout, detach_heads=False),
                "detach_heads": megatron_native(layout, detach_heads=True),
            }
            for layout in layouts
        }
        with skyrl_native_loss_patch():
            skyrl = {
                (layout, head, chunk): skyrl_hard_label(layout, head, chunk)
                for layout in layouts
                for head in heads
                for chunk in chunks
            }
        assert mtp_mod.MTPLossAutoScaler.apply.__name__ == "apply"
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
        fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = prev_force_ieee

    for layout, runs in megatron.items():
        out["megatron"][layout] = {
            variant: dict(loss=r["loss"], num_tokens=r["num_tokens"], norms=r["norms"]) for variant, r in runs.items()
        }
        # Megatron's mask has no mass outside the generated region.
        out["megatron"][layout]["mask_mass_outside_actions"] = float(
            runs["stock"]["mask"][:, :-num_actions].abs().sum()
        )
    for (layout, head, chunk), r in skyrl.items():
        ref = megatron[layout]["stock"]
        valid = r["valid"].bool()
        ce_diff = (-r["lp"] - ref["ce"][:, -num_actions:])[valid].abs()

        def mtp_grad_diff(meg):
            """Max abs diff; max over params of max|diff| / max|grad|, and which param; global L2 rel diff."""
            max_abs, max_rel, worst, sq_diff, sq_ref = 0.0, 0.0, None, 0.0, 0.0
            for n, g_sky in r["mtp_grads"].items():
                g_meg = meg["mtp_grads"][n] / scale  # d(CE mean)/d(theta_mtp)
                diff = g_sky - g_meg
                d = diff.abs().max().item()
                rel = d / max(g_meg.abs().max().item(), 1e-30)
                max_abs = max(max_abs, d)
                if rel > max_rel:
                    max_rel, worst = rel, n
                sq_diff += float(diff.double().pow(2).sum())
                sq_ref += float(g_meg.double().pow(2).sum())
            return max_abs, max_rel, worst, (sq_diff / sq_ref) ** 0.5

        assert r["mtp_grads"].keys() == ref["mtp_grads"].keys()
        abs_stock, rel_stock, worst_stock, l2_stock = mtp_grad_diff(ref)
        abs_detach, rel_detach, _, _ = mtp_grad_diff(megatron[layout]["detach_heads"])
        out["skyrl"][f"{layout}/{head}/chunk{chunk}"] = dict(
            loss=r["loss"],
            megatron_loss=ref["loss"],
            loss_abs_diff=abs(r["loss"] - ref["loss"]),
            loss_rel_diff=abs(r["loss"] - ref["loss"]) / abs(ref["loss"]),
            num_tokens=float(r["valid"].sum()),
            megatron_num_tokens=ref["num_tokens"],
            mask_equal=bool(torch.equal(ref["mask"][:, -num_actions:].bool(), valid)),
            labels_equal=bool(torch.equal(ref["labels"][:, -num_actions:][valid], sequences[:, -num_actions:][valid])),
            per_token_ce_max_abs_diff=float(ce_diff.max()),
            mtp_grad_max_abs_diff=abs_stock,
            mtp_grad_max_rel_diff=rel_stock,
            mtp_grad_max_rel_diff_param=worst_stock,
            mtp_grad_rel_l2_diff=l2_stock,
            mtp_grad_max_abs_diff_vs_detach=abs_detach,
            mtp_grad_max_rel_diff_vs_detach=rel_detach,
            norms=r["norms"],
            connected=r["connected"],
            native_ran=r["native_ran"],
        )
    if keep_native:
        out["native"] = {layout: runs["stock"] for layout, runs in megatron.items()}
    return out


def _build_tiny_mtp_gpt(device, attention_backend=AttnBackend.unfused, num_layers=2, vp_stage=None):
    """This rank's part of the model (its ``vp_stage`` chunk under VPP) for the current TP / PP / CP / VPP layout."""
    tp = mpu.get_tensor_model_parallel_world_size()
    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=HIDDEN,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.05,
        params_dtype=torch.float32,
        bf16=False,
        fp16=False,
        gradient_accumulation_fusion=False,
        # fp32 attention runs on TE's unfused backend (flash / fused need bf16 or fp16), the default here; say
        # so, since megatron-core pins the NVTE_* backend env vars from this field.
        attention_backend=attention_backend,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=MTP_LOSS_SCALING_FACTOR,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        # As SkyRL sets them: SP whenever TP > 1, and stage-to-stage shapes sent per microbatch (packed).
        sequence_parallel=tp > 1,
        pipeline_dtype=torch.float32,
        variable_seq_lengths=True,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    torch.manual_seed(0)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=layer_spec,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
        mtp_block_spec=get_gpt_mtp_block_spec(cfg, layer_spec, use_transformer_engine=True, vp_stage=vp_stage),
        pre_process=mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
        post_process=mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
        vp_stage=vp_stage,
    ).to(device)
    return model.train()


@ray.remote(num_gpus=1)
class _ParityRank:
    def endpoint(self):
        with socket.socket() as s:
            s.bind(("", 0))
            return ray.util.get_node_ip_address(), s.getsockname()[1]

    def run(self, fn, world_size: int, rank: int, master_addr: str, master_port: int):
        return fn(world_size, rank, f"tcp://{master_addr}:{master_port}")


def run_on_one_node(fn, world_size: int) -> list:
    """``fn(world_size, rank, init_method)`` on ``world_size`` ranks of one node, one process and GPU each."""
    pg = placement_group([{"GPU": world_size, "CPU": world_size}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=120)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    ranks = [_ParityRank.options(scheduling_strategy=strategy).remote() for _ in range(world_size)]
    try:
        master_addr, master_port = ray.get(ranks[0].endpoint.remote())
        refs = [r.run.remote(fn, world_size, i, master_addr, master_port) for i, r in enumerate(ranks)]
        pending = refs
        while pending:  # raise the first rank's error rather than wait on peers blocked in a collective
            done, pending = ray.wait(pending, num_returns=1)
            ray.get(done)
        return ray.get(refs)
    finally:
        for r in ranks:
            ray.kill(r)
        remove_placement_group(pg)


def _tp_parity(tp: int, rank: int, init_method: str) -> dict:
    device = init_megatron(tp, rank, init_method)
    try:
        model = _build_tiny_mtp_gpt(device)
        return compare_hard_label_vs_native(model, _make_rl_batch(device), num_actions=NUM_ACTIONS)
    finally:
        teardown_megatron()


@pytest.fixture(scope="module")
def parity_by_tp(module_scoped_ray_init_fixture):
    """``compare_hard_label_vs_native`` on every rank of a TP group, one process and GPU per rank, once per TP size."""

    @functools.cache
    def run(tp: int) -> list:
        results = run_on_one_node(_tp_parity, tp)
        for rank, res in enumerate(results):
            print(f"\n===== TP={tp} rank {rank}: SkyRL hard-label MTP loss vs Megatron process_mtp_loss =====")
            for layout, runs in res["megatron"].items():
                print(f"megatron[{layout}]: {runs}")
            for case, r in res["skyrl"].items():
                print(f"skyrl[{case}]: {r}")
        return results

    return run


@pytest.fixture(scope="module", params=RANKS, ids=lambda tp_rank: "tp{}-rank{}".format(*tp_rank))
def parity(request, parity_by_tp):
    tp, rank = request.param
    return {**parity_by_tp(tp)[rank], "tp": tp}


@pytest.mark.megatron
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_mtp_hard_label_loss_matches_process_mtp_loss(parity, layout, head, chunk):
    r = parity["skyrl"][f"{layout}/{head}/chunk{chunk}"]
    assert not r["native_ran"], "process_mtp_loss ran on the SkyRL side: the native-loss patch is not in effect"
    assert parity["megatron"][layout]["mask_mass_outside_actions"] == 0.0
    # Same targets, same mask, same denominator -- exactly.
    assert r["mask_equal"], "SkyRL's valid mask != Megatron's depth-1 loss mask (moved to target coordinates)"
    assert r["labels_equal"], "Megatron's depth-1 labels are not the tokens SkyRL scores"
    assert r["num_tokens"] == r["megatron_num_tokens"] == EXPECTED_NUM_TOKENS
    # Same per-token CE and loss, in fp32.
    assert r["per_token_ce_max_abs_diff"] < 1e-5, r
    assert r["loss_abs_diff"] < 1e-5 and r["loss_rel_diff"] < 1e-6, r
    # Same gradient on the MTP layer's own parameters, once Megatron's scale is divided out. At TP>1 the
    # cross-rank reductions reorder fp32 sums: the worst tensor's max diff reaches ~1.4e-5 (L2 ~2.6e-6).
    grad_tol = 1e-5 if parity["tp"] == 1 else 5e-5
    assert r["mtp_grad_max_rel_diff"] < grad_tol, r
    assert r["mtp_grad_max_rel_diff_vs_detach"] < grad_tol, r


@pytest.mark.megatron
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mtp_hard_label_loss_gradient_reaches_only_the_head(parity, layout):
    for head in HEADS:
        for chunk in CHUNKS:
            r = parity["skyrl"][f"{layout}/{head}/chunk{chunk}"]
            assert r["norms"]["mtp"] > 0, r
            # No autograd path at all from the draft loss into the policy's parameters.
            for group in ("trunk", "embedding", "output_layer"):
                assert r["norms"][group] == 0.0 and not r["connected"][group], (head, chunk, group, r)
    stock = parity["megatron"][layout]["stock"]["norms"]
    detach = parity["megatron"][layout]["detach_heads"]["norms"]
    # The stock native loss trains the policy: trunk, embedding and output layer all get gradient.
    for group in ("mtp", "trunk", "embedding", "output_layer"):
        assert stock[group] > 0, (group, stock)
    # mtp_detach_heads=True detaches the output weight, the block's trunk input and the re-embedding.
    assert detach["mtp"] > 0, detach
    for group in ("trunk", "embedding", "output_layer"):
        assert detach[group] == 0.0, (group, detach)
