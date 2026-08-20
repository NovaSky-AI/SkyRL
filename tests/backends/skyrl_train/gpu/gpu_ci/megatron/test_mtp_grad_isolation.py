"""MTP decoupling ground truth on a real Megatron worker (Qwen3.5-2B, 1 GPU).

Asserts the two invariants the whole decoupled-MTP design rests on, against the real
DistributedOptimizer + DDP stack (CPU unit tests fake that stack and cannot see this):

  1. The draft head is TRAINABLE: every ``.mtp.`` param has ``requires_grad=True`` and is inside
     the policy optimizer's param groups (shared-buffer design; no separate MTP optimizer).
  2. The draft loss is ISOLATED: a backward pass of the draft loss alone (both the full soft-CE
     and the top-k path used by the 9B/MiMo recipes) puts gradient ONLY in ``.mtp.`` params --
     nothing in the embedding (== tied lm_head), the decoder trunk, or the final norm.

Diagnoses the "policy grad_norm 20x inflated / head silently frozen" class of regressions directly.

uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_grad_isolation.py
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


def _is_mtp_name(name: str) -> bool:
    return ".mtp." in name or name.startswith("mtp.")


class _ProbeMTPGradIsolation(MegatronPolicyWorkerBase):
    def probe(self, token_ids=None, seq_len: int = 64) -> dict:
        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import (
            build_teacher_logits,
            draft_soft_ce,
            draft_soft_ce_topk,
        )

        chunk = self.actor_module[0]
        host = _resolve_mtp_host(_unwrap_model(chunk))
        if getattr(host, "mtp", None) is None:
            return {"error": f"no mtp head built (host={type(host).__name__})"}

        named = list(chunk.named_parameters())
        mtp_named = [(n, p) for n, p in named if _is_mtp_name(n)]

        # -- (1) trainability + optimizer membership ------------------------------------------
        opt_model_ids = set()
        for sub in getattr(self.optimizer, "chained_optimizers", None) or [self.optimizer]:
            for attr in ("model_float16_groups", "model_fp32_groups", "float16_groups", "fp32_from_fp32_groups"):
                for group in getattr(sub, attr, None) or []:
                    opt_model_ids.update(id(p) for p in group)

        res = {
            "host": type(host).__name__,
            "n_params": len(named),
            "n_requires_grad": sum(p.requires_grad for _, p in named),
            "n_opt_model_params": len(opt_model_ids),
            "n_mtp_params": len(mtp_named),
            "n_mtp_requires_grad": sum(p.requires_grad for _, p in mtp_named),
            "n_mtp_in_optimizer": sum(1 for _, p in mtp_named if id(p) in opt_model_ids),
            "mtp_names": sorted(n for n, _ in mtp_named),
            "mtp_names_in_optimizer": sorted(n for n, p in mtp_named if id(p) in opt_model_ids),
        }

        # -- (2) draft-loss gradient isolation ------------------------------------------------
        device = torch.cuda.current_device()
        if token_ids is not None:
            # Real coherent text: the pretrained head then agrees with the teacher, so the head grad
            # norm here approximates a real run's (soft-CE grads track distribution mismatch).
            ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            seq_len = ids.shape[0]
        else:
            ids = (torch.arange(seq_len, device=device) * 7 + 3) % 10000
        sequences = ids.unsqueeze(0)
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = torch.ones(1, seq_len, device=device)

        def grad_norm_of(param) -> float:
            g = getattr(param, "main_grad", None)
            if g is None:
                g = param.grad
            return 0.0 if g is None else float(torch.norm(g.float(), 2))

        def run_backward(loss_fn_name: str) -> dict:
            chunk.zero_grad_buffer()
            for _, p in named:
                p.grad = None
            capture = MTPHiddenCapture(chunk)
            with capture.capture():
                outputs = chunk(sequences, position_ids, attention_mask)
                student_hidden = capture.compute_student_hidden_states()
            student_logits = project_mtp_hidden_to_logits(student_hidden, capture.model)[0]
            main_logits = outputs if outputs.shape[0] == 1 else outputs.transpose(0, 1)
            teacher_src = main_logits.detach()
            if loss_fn_name == "topk":
                # The 9B/MiMo recipes: top-k on the UN-rolled teacher, rolled inside the loss.
                loss = draft_soft_ce_topk(student_logits.float(), teacher_src.float(), mask, k=64, roll_shift=1)
            else:
                teacher = build_teacher_logits(teacher_src.float(), 0)
                loss = draft_soft_ce(student_logits.float(), teacher, mask)
            loss.backward()

            per_group = {"mtp": 0.0, "embedding": 0.0, "decoder": 0.0, "other": 0.0}
            for n, p in named:
                g = grad_norm_of(p)
                if _is_mtp_name(n):
                    key = "mtp"
                elif "embedding" in n or "embed_tokens" in n:
                    key = "embedding"
                elif ".layers." in n:
                    key = "decoder"
                else:
                    key = "other"
                per_group[key] = max(per_group[key], g)
            per_group["loss"] = float(loss)
            return per_group

        res["grads_soft_ce"] = run_backward("full")
        res["grads_topk"] = run_backward("topk")

        # -- (3) the separate grad-clip is installed on the real optimizer ----------------------
        # init_model installs it (mtp heads active); the step override lives on the INSTANCE. Its
        # per-rank main-param mapping is recounted here without reinstalling.
        from skyrl.backends.skyrl_train.mtp.grad_clip import (
            _mtp_main_params,
            _mtp_model_param_ids,
            _sub_optimizers,
        )

        res["clip_installed"] = "step" in vars(self.optimizer)
        ids = _mtp_model_param_ids(self.actor_module)
        res["n_clip_local"] = sum(len(_mtp_main_params(s, ids)) for s in _sub_optimizers(self.optimizer))
        return res


_ProbeWorker = ray.remote(num_gpus=1)(_ProbeMTPGradIsolation)


def _cfg(tp: int, num_gpus: int):
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.placement.ref_num_gpus_per_node = num_gpus
    # Match the spec-decode recipes: no sample packing (the VL-bridge GDN path rejects it).
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    validate_cfg(cfg)
    return cfg


# TP=4 mirrors the 9B/MiMo recipes (sequence parallel on, vocab-sharded draft loss); TP=1 is the
# minimal case; TP4xDP2 mirrors the full 8-GPU recipe (the DistributedOptimizer DP-shards param
# ownership, which coverage checks must account for).
@pytest.mark.megatron
@pytest.mark.parametrize("tp,dp", [(1, 1), (4, 1), (4, 2)])
def test_mtp_grad_isolation(ray_init_fixture, tp, dp):
    num_gpus = tp * dp
    cfg = _cfg(tp, num_gpus)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbeWorker
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The derivative of x squared is two x. In 1969, "
        "Apollo 11 landed the first humans on the Moon. Water boils at one hundred degrees Celsius "
        "at sea level. The capital of France is Paris, a city on the river Seine."
    )
    token_ids = tok(passage, add_special_tokens=False)["input_ids"]
    # Sequence parallel (TP>1) scatters the sequence dim: length must divide by TP.
    token_ids = token_ids[: (len(token_ids) // (tp * 2)) * (tp * 2)]

    try:
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=num_gpus, cfg=cfg
        )
        results = ray.get(policy.async_run_ray_method("pass_through", "probe", token_ids))
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    for rank, res in enumerate(results):
        print(f"\n===== MTP grad isolation (TP={tp}, DP={dp}) rank {rank} =====")
        for k, v in res.items():
            print(f"{k:24s}: {v}")
        assert "error" not in res, res

    res0 = results[0]
    # (1) The head is trainable and (globally) inside the shared policy optimizer. The
    # DistributedOptimizer shards param OWNERSHIP across DP ranks, so a single rank may
    # legitimately own none (or a subset) of the head's params -- coverage is the RANK UNION.
    assert res0["n_mtp_params"] > 0
    assert res0["n_mtp_requires_grad"] == res0["n_mtp_params"], "MTP head params are frozen -- head cannot train"
    covered = set().union(*(r["mtp_names_in_optimizer"] for r in results))
    missing = set(res0["mtp_names"]) - covered
    assert not missing, (
        f"MTP head params missing from the policy optimizer on EVERY rank: {sorted(missing)} "
        f"(per-rank counts: {[r['n_mtp_in_optimizer'] for r in results]}) -- those weights never train"
    )

    # (2) The draft loss's gradient reaches ONLY the head. A leak into the embedding (== tied
    # lm_head) or the decoder trunk perturbs the policy and inflates policy/grad_norm.
    for res in results:
        for path in ("grads_soft_ce", "grads_topk"):
            g = res[path]
            assert g["mtp"] > 0, f"{path}: head got no gradient -- draft loss is not training the head"
            for group in ("embedding", "decoder", "other"):
                assert g[group] == 0.0, (
                    f"{path}: draft-loss gradient leaked into {group} (norm={g[group]:.4f}) -- "
                    "the detach chain is broken; this is what inflates policy/grad_norm"
                )

    # (3) The per-group clip is installed on every rank, and the head's main params are covered by
    # the DP-rank union (a single rank owning zero is normal under DistributedOptimizer sharding).
    assert all(r["clip_installed"] for r in results), "separate MTP grad clip not installed on some rank"
    assert sum(r["n_clip_local"] for r in results) > 0
