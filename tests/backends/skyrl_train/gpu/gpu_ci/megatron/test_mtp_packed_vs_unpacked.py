"""Localize Bug B: the MTP soft-CE is ~2.5 on a clean single UNPADDED sequence but
inflates to ~44 (TP=1) on a real batched, LEFT-PADDED RL batch -- far above ln(V)=12.4,
so the draft head looks *confidently wrong* on real training data.

The existing `test_mtp_replay_vs_native` calls the model DIRECTLY
(`self.actor_module[0](sequences, position_ids, attention_mask)`), i.e. the BSHD path.
Real training defaults to `remove_microbatch_padding=True`, which routes through
`preprocess_packed_seqs` (THD packing). So the clean probe is structurally blind to the
packed path. This probe runs BOTH on the SAME content and reports soft-CE at the real
positions for each, plus replay-vs-native divergence under packing:

  - unpacked soft-CE ~= packed soft-CE  -> bug is NOT the packed forward (look loss-side).
  - packed soft-CE >> unpacked, replay ~= native  -> Megatron's THD MTP forward is the
    culprit (we feed it packed inputs the MTP path mishandles).
  - packed soft-CE >> unpacked, replay != native   -> our REPLAY diverges under packing.

Run::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_packed_vs_unpacked.py
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3.5-2B"


class _ProbeMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def probe_packed_vs_unpacked(self, token_ids_list) -> dict:
        import torch
        import megatron.core.parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            recover_left_padding,
            remove_left_padding,
        )
        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _mtp_layer_offset,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import (
            build_teacher_logits,
            draft_soft_ce,
            shift_mask_for_mtp,
        )

        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        if getattr(host, "mtp", None) is None:
            return {"error": "no mtp head built", "host": type(host).__name__}
        num_layers = int(getattr(host.config, "mtp_num_layers", 1) or 1)
        device = torch.cuda.current_device()

        def to_bsv(x):
            if x.dim() != 3:
                return x
            return x if x.shape[0] != 1 or x.shape[1] == 1 else x  # [1,S,V] already

        tp_grp = mpu.get_tensor_model_parallel_group()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def _buggy_shift_mask(mask, k):
            # Pre-fix behaviour: roll only, no source-side AND (leaks left-pad boundary).
            shift = k + 1
            rolled = torch.roll(mask, shifts=-shift, dims=1)
            rolled[:, -shift:] = 0
            return rolled

        # ---- soft-CE on a [B,S,V] main + per-depth student set ----
        # vp_group exercises the real vocab-parallel path when TP>1 (mirrors loss_func).
        def soft_ce_over_depths(main_logits, students, mask, mask_fn, vp=False):
            out = []
            for k, st in enumerate(students):
                teacher = build_teacher_logits(main_logits, k, detach=True)
                lm = mask_fn(mask, k)
                out.append(
                    draft_soft_ce(
                        st, teacher, lm,
                        vocab_parallel_group=tp_grp if vp else None,
                    ).item()
                )
            return out

        # =========================================================================
        # (1) UNPACKED baseline: run each sequence alone (no padding), BSHD path.
        # =========================================================================
        # Skip the raw single-seq baseline under SP/TP>1: it bypasses remove_left_padding's
        # seq-length padding to a TP multiple, so odd-length seqs trip the SP scatter. The batched
        # path below uses remove_left_padding and is the one that matters for the TP>1 (158) case.
        unpacked_soft_ce = []
        for ids in token_ids_list if tp_size == 1 else []:
            seq = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
            S = seq.shape[1]
            pos = torch.arange(S, device=device).unsqueeze(0)
            am = torch.ones_like(seq)
            cap = MTPHiddenCapture(self.actor_module[0], detach_trunk=True)
            with torch.no_grad(), cap.capture():
                out = self.actor_module[0](seq, pos, am)
                st_hidden = cap.compute_student_hidden_states()
            main = to_bsv(out).float()
            students = [s.float() for s in project_mtp_hidden_to_logits(st_hidden, host)]
            unpacked_soft_ce.append(
                soft_ce_over_depths(main, students, am.float(), shift_mask_for_mtp, vp=tp_size > 1)[0]
            )

        # =========================================================================
        # (2) BATCHED LEFT-PADDED, non-packed (remove_left_padding) -- the ACTUAL
        # Qwen3.5 training path (GDN can't pack -> REMOVE_MICROBATCH_PADDING=false).
        # =========================================================================
        B = len(token_ids_list)
        L = max(len(x) for x in token_ids_list)
        sequences = torch.zeros(B, L, device=device, dtype=torch.long)
        attention_mask = torch.zeros(B, L, device=device, dtype=torch.long)
        for i, ids in enumerate(token_ids_list):
            n = len(ids)
            sequences[i, L - n :] = torch.tensor(ids, device=device)  # LEFT padding
            attention_mask[i, L - n :] = 1
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        am_bool = attention_mask.to(bool)
        is_first = mpu.is_pipeline_first_stage(ignore_virtual=True)
        new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
            sequences, am_bool, position_ids, pre_process=is_first
        )
        is_last = mpu.is_pipeline_last_stage(ignore_virtual=True)

        def depad(t):
            return recover_left_padding(t, new_attention_mask, am_bool, L, post_process=is_last)

        # capture in-forward NATIVE mtp output via hook, alongside our replay
        inforward = {}
        mtp = host.mtp
        hh = mtp.register_forward_hook(lambda _m, _i, o: inforward.__setitem__("out", o.detach()))
        cap = MTPHiddenCapture(self.actor_module[0], detach_trunk=True)
        try:
            with torch.no_grad(), cap.capture():
                out = self.actor_module[0](new_sequences, new_position_ids, new_attention_mask)
                replay_hidden = cap.compute_student_hidden_states()
        finally:
            hh.remove()

        main_packed = depad(out).float()  # [B,L,V] (forward_step de-pads the model output directly)
        replay_students = [depad(s).float() for s in project_mtp_hidden_to_logits(replay_hidden, host)]

        total = 1 + _mtp_layer_offset(mtp) + num_layers
        native_hidden = list(torch.chunk(inforward["out"], total, dim=0))[-num_layers:]
        native_students = [depad(s).float() for s in project_mtp_hidden_to_logits(native_hidden, host)]

        mask_f = attention_mask.float()
        vp = tp_size > 1
        replay_fixed = soft_ce_over_depths(main_packed, replay_students, mask_f, shift_mask_for_mtp, vp=vp)
        replay_buggy = soft_ce_over_depths(main_packed, replay_students, mask_f, _buggy_shift_mask, vp=vp)
        native_fixed = soft_ce_over_depths(main_packed, native_students, mask_f, shift_mask_for_mtp, vp=vp)

        # replay-vs-native divergence at REAL positions only (pad positions are zero-filled).
        m3 = am_bool.unsqueeze(-1)
        max_abs = max(
            ((r - n).abs() * m3).max().item() for r, n in zip(replay_students, native_students)
        )
        pad_frac = 1.0 - (attention_mask.sum().item() / (B * L))

        return {
            "host": type(host).__name__,
            "mtp_num_layers": num_layers,
            "tp_size": tp_size,
            "batch_shape": (B, L),
            "pad_fraction": round(pad_frac, 3),
            "unpacked_soft_ce_per_seq": [round(x, 3) for x in unpacked_soft_ce],
            "batched_soft_ce_replay_FIXED_mask": [round(x, 3) for x in replay_fixed],
            "batched_soft_ce_replay_BUGGY_mask": [round(x, 3) for x in replay_buggy],
            "batched_soft_ce_native_FIXED_mask": [round(x, 3) for x in native_fixed],
            "max_abs_replay_minus_native": max_abs,
        }


_ProbePolicyWorker = ray.remote(num_gpus=1)(_ProbeMegatronPolicyWorker)


def _cfg():
    import os

    tp = int(os.environ.get("PROBE_TP", "1"))
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = tp
    cfg.trainer.placement.ref_num_gpus_per_node = tp
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    cfg.trainer.mtp.loss_type = "soft_ce"
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
def test_mtp_packed_vs_unpacked(ray_init_fixture):
    import os

    tp = int(os.environ.get("PROBE_TP", "1"))
    cfg = _cfg()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Coherent passages of VERY different lengths -> heavy left padding in the batch (stresses the
    # left-pad mask leak: the shorter the seq, the larger its pad fraction).
    long_txt = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. In 1969, Apollo 11 landed humans on the Moon. "
        "Water boils at one hundred degrees Celsius at sea level. The capital of France is Paris."
    )
    short_txt = "The quick brown fox jumps over the lazy dog near the river bank at dawn."
    tiny_txt = "Paris is the capital of France."
    ids_long = tok(long_txt, add_special_tokens=False)["input_ids"][:96]
    ids_short = tok(short_txt, add_special_tokens=False)["input_ids"][:40]
    ids_tiny = tok(tiny_txt, add_special_tokens=False)["input_ids"][:10]
    token_ids_list = [ids_long, ids_short, ids_tiny]

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=tp, cfg=cfg
        )
        res = ray.get(
            policy.async_run_ray_method("pass_through", "probe_packed_vs_unpacked", token_ids_list)
        )[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    print("\n===== MTP packed vs unpacked =====")
    for k, v in res.items():
        print(f"{k:36s}: {v}")
    print("==================================")
    assert "error" not in res, res
