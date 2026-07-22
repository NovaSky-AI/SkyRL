"""Isolate the `mtp_loss ~158` anomaly: is our decoupled REPLAY of the MTP head
broken, or is Megatron's NATIVE MTP forward itself wrong for Qwen3.5?

Context: vLLM drafts this pretrained MTP head at ~0.8 acceptance (head is GOOD), yet
our training-side soft-CE reports ~158 nats (>> uniform ln(V)=12.4) -> the head looks
confidently wrong on the training side. Two hypotheses:
  (a) our replay (capture mtp inputs -> re-run on detached hidden -> project) diverges
      from Megatron's in-forward MTP output, or
  (b) Megatron's native MTP forward is itself wrong (so even process_mtp_loss is high).

This probe runs ONE real forward on the policy worker and, for MTP depth 0:
  - captures the IN-FORWARD mtp output chunk (a forward hook) == what process_mtp_loss
    projects -> in-forward logits
  - runs our replay -> replay logits
  - projects both through the shared output layer
  - reports: max|replay - in-forward| (should be ~0 if the replay is faithful),
    hard-CE of each vs ground-truth seq[t+2], and our soft-CE (teacher = rolled main).

Interpretation:
  - replay ~= in-forward AND both hard-CE high  -> (b) Megatron MTP forward bug.
  - replay != in-forward (replay hard-CE high, in-forward low) -> (a) replay bug.

Run::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_replay_vs_native.py
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
    def probe_replay_vs_native(self, token_ids=None, seq_len: int = 96) -> dict:
        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )

        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        if getattr(host, "mtp", None) is None:
            return {"error": "no mtp head built", "host": type(host).__name__}

        device = torch.cuda.current_device()
        # Use REAL tokenized text (passed from the driver) so the model is confident -- a
        # confidently-wrong MTP head shows up as high CE, which a garbage ramp cannot reveal.
        if token_ids is not None:
            ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            seq_len = ids.shape[0]
        else:
            ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)  # [1, S]
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Capture the IN-FORWARD mtp output via a forward hook (the thing
        # process_mtp_loss splits + projects), alongside our replay capture.
        inforward = {}

        def _out_hook(_m, _inp, out):
            inforward["out"] = out.detach()

        mtp = host.mtp
        h = mtp.register_forward_hook(_out_hook)
        capture = MTPHiddenCapture(self.actor_module[0])
        try:
            with torch.no_grad(), capture.capture():
                outputs = self.actor_module[0](sequences, position_ids, attention_mask)
                replay_hidden = capture.compute_student_hidden_states()  # list per depth
        finally:
            h.remove()

        # Normalize a 3-D logits tensor to [B, S, V] (B=1 here). project_mtp_hidden_to_logits
        # already returns [B, S, V]; the main model's output orientation is model-dependent.
        def to_bsv(x):
            if x.dim() != 3:
                return x
            if x.shape[0] == 1:
                return x
            if x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        main_logits = to_bsv(outputs)

        # Split the in-forward output the same way compute_student_hidden_states does.
        from skyrl.backends.skyrl_train.mtp.hidden_capture import _mtp_layer_offset

        num_layers = int(getattr(host.config, "mtp_num_layers", 1) or 1)
        total = 1 + _mtp_layer_offset(mtp) + num_layers
        inforward_hidden = list(torch.chunk(inforward["out"], total, dim=0))[-num_layers:]

        replay_logits = project_mtp_hidden_to_logits(replay_hidden, host)[0]  # [B,S,V/tp]
        native_logits = project_mtp_hidden_to_logits(inforward_hidden, host)[0]

        rl = to_bsv(replay_logits).float()
        nl = to_bsv(native_logits).float()
        ml = main_logits.float()

        # ground-truth target for depth-0 MTP: seq[t+2]
        gt = torch.roll(sequences, shifts=-2, dims=1)
        valid = seq_len - 2
        gt = gt[:, :valid]

        def hard_ce(logits):
            lp = torch.log_softmax(logits[:, :valid].float(), dim=-1)
            return -lp.gather(-1, gt.unsqueeze(-1)).squeeze(-1).mean().item()

        # soft CE: teacher = main rolled left by 1 (policy dist over seq[t+2])
        teacher = torch.roll(ml, shifts=-1, dims=1)[:, :valid]
        tprob = torch.softmax(teacher, dim=-1)
        r_logp = torch.log_softmax(rl[:, :valid], dim=-1)
        n_logp = torch.log_softmax(nl[:, :valid], dim=-1)
        soft_ce_replay = -(tprob * r_logp).sum(-1).mean().item()
        soft_ce_native = -(tprob * n_logp).sum(-1).mean().item()

        replay_argmax_match = (rl[:, :valid].argmax(-1) == gt).float().mean().item()
        native_argmax_match = (nl[:, :valid].argmax(-1) == gt).float().mean().item()
        # Main model's own next-token (t+1) prediction -- sanity that the trunk is fine on this text.
        nxt = torch.roll(sequences, -1, 1)[:, : seq_len - 1]
        main_logp = torch.log_softmax(ml[:, : seq_len - 1].float(), dim=-1)
        hard_ce_main_t1 = -main_logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1).mean().item()
        main_argmax_match = (ml[:, : seq_len - 1].argmax(-1) == nxt).float().mean().item()

        return {
            "hard_ce_main_t+1": hard_ce_main_t1,
            "host": type(host).__name__,
            "vocab_shard": tuple(rl.shape),
            "max_abs_replay_minus_native": (rl - nl).abs().max().item(),
            "hard_ce_replay": hard_ce(rl),
            "hard_ce_native": hard_ce(nl),
            "soft_ce_replay": soft_ce_replay,
            "soft_ce_native": soft_ce_native,
            "replay_argmax_match_t+2": replay_argmax_match,
            "native_argmax_match_t+2": native_argmax_match,
            "main_argmax_match_t+1": main_argmax_match,
            "replay_logit_absmax": rl.abs().max().item(),
            "native_logit_absmax": nl.abs().max().item(),
            "main_logit_absmax": ml.abs().max().item(),
        }


_ProbePolicyWorker = ray.remote(num_gpus=1)(_ProbeMegatronPolicyWorker)


def _cfg():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    # This probe compares the replayed vs in-forward MTP output on a single unpacked sequence;
    # packing is irrelevant here and the Qwen3.5 VL bridge rejects it (it packs internally).
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
def test_mtp_replay_vs_native(ray_init_fixture):
    cfg = _cfg()

    # Real coherent text so the model is confident (a confidently-wrong MTP head -> high CE).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The derivative of x squared is two x. In 1969, "
        "Apollo 11 landed the first humans on the Moon. Water boils at one hundred degrees Celsius "
        "at sea level. The capital of France is Paris, a city on the river Seine."
    )
    token_ids = tok(passage, add_special_tokens=False)["input_ids"][:128]

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_replay_vs_native", token_ids))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    print("\n===== MTP replay vs native =====")
    for k, v in res.items():
        print(f"{k:32s}: {v}")
    print("================================")

    assert "error" not in res, res
    # A faithful replay must match the in-forward MTP output.
    assert res["max_abs_replay_minus_native"] < 1e-2, (
        f"replay diverges from in-forward MTP output (max_abs="
        f"{res['max_abs_replay_minus_native']:.4f}) -> replay-input bug"
    )
