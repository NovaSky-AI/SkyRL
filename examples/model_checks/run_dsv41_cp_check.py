"""DeepSeek-V4.1 context-parallel correctness check: trainer logprobs of a fixed batch under a given CP size.

Scores ``--num-seqs`` deterministic random-token sequences of ``--seq-len`` tokens with the Megatron policy (no
LoRA, forward only, routing computed by the router) and saves the per-token logprobs, so runs with different
``--cp`` / ``--packed`` can be compared offline (``--compare a.pt b.pt``). Use a real-weight checkpoint (e.g. a
layer slice): every parallel layout then loads identical weights, which a random init does not guarantee.

    python -m examples.model_checks.run_dsv41_cp_check --model <dir> --cp 2 --packed 1 --out /tmp/cp2.pt
    python -m examples.model_checks.run_dsv41_cp_check --compare /tmp/cp1.pt /tmp/cp2.pt
"""

import argparse

import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray, validate_cfg

CheckWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)


def build_config(args) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    t = cfg.trainer
    t.strategy = "megatron"
    t.policy.model.path = args.model
    t.placement.colocate_all = False
    t.placement.policy_num_nodes = 1
    t.placement.policy_num_gpus_per_node = args.gpus
    t.placement.ref_num_nodes = 1
    t.placement.ref_num_gpus_per_node = args.gpus
    t.policy.inference_only_init = not args.train
    t.micro_forward_batch_size_per_gpu = 1
    t.remove_microbatch_padding = bool(args.packed)
    t.fused_lm_head_logprob = True
    t.fused_lm_head_logprob_backend = "triton"
    t.logger = "console"
    t.algorithm.max_seq_len = args.seq_len + 8
    m = t.policy.megatron_config
    m.tensor_model_parallel_size = 1
    m.pipeline_model_parallel_size = 1
    m.context_parallel_size = args.cp
    m.expert_model_parallel_size = args.ep
    m.expert_tensor_parallel_size = 1
    m.mtp_num_layers = 0
    m.moe_grouped_gemm = True
    m.moe_token_dispatcher_type = "alltoall"
    if args.r3:
        m.moe_enable_routing_replay = True
        cfg.generator.inference_engine.enable_return_routed_experts = True
    # megatron-core's packed DSv4 attention requires the cuDNN DSA backend; set it for every run so packed and
    # unpacked runs use the same kernels.
    m.transformer_config_kwargs["dsa_kernel_backend"] = args.dsa_backend
    if args.cp > 1:
        m.transformer_config_kwargs["attention_cp_layout"] = "contiguous"
        m.transformer_config_kwargs["linear_cp_layout"] = "contiguous"
    if args.train:
        # LoRA (as the RL recipe) and the stateful full recompute, so CP's backward paths run: the compressed-KV
        # gather's reduce-scatter, the boundary exchange's gradient, the layer replays.
        from examples.model_checks.run_dsv41_logprobs import LORA_TARGET_MODULES

        lora = t.policy.model.lora
        lora.rank, lora.alpha, lora.target_modules = 32, 32, LORA_TARGET_MODULES
        m.lora_config.merge_lora = False
        m.transformer_config_kwargs["recompute_granularity"] = "full"
        m.transformer_config_kwargs["recompute_method"] = "uniform"
        m.transformer_config_kwargs["recompute_num_layers"] = 1
        m.transformer_config_kwargs["recompute_modules"] = None
        t.micro_train_batch_size_per_gpu = 1
        t.policy_mini_batch_size = args.num_seqs
        t.train_batch_size = args.num_seqs
    else:
        t.policy.model.lora.rank = 0
    e = cfg.generator.inference_engine
    e.run_engines_locally = True
    e.tensor_parallel_size = 1
    e.num_engines = 1
    validate_cfg(cfg)
    return cfg


def make_batch_r3(num_seqs, seq_len, vocab, n_layers, n_experts, topk, seed: int = 0) -> TrainingInputBatch:
    """As ``make_batch`` plus deterministic random expert routes for every token, replayed by the routers (R3):
    routing is then identical across parallel layouts, so CP runs can be compared without top-k flips."""
    import numpy as np

    from skyrl.train.dataset.preprocess import (
        convert_prompts_responses_to_batch_tensors,
        make_router_padding_mask,
    )

    g = torch.Generator().manual_seed(seed)
    seqs = torch.randint(1000, min(vocab, 100000), (num_seqs, seq_len), generator=g)
    rng = np.random.default_rng(seed)
    routes = [
        np.argsort(rng.random((seq_len, n_layers, n_experts)), axis=-1)[..., :topk].astype(np.int16)
        for _ in range(num_seqs)
    ]
    prompts = [s[:1].tolist() for s in seqs]
    responses = [s[1:].tolist() for s in seqs]
    seqs_t, attn, resp_mask, rewards, loss_masks, lp_t, routes_t, _ = convert_prompts_responses_to_batch_tensors(
        0, prompts, responses, [[0.0] * len(r) for r in responses], [[1] * len(r) for r in responses],
        [[0.0] * len(r) for r in responses], routes, None,
    )
    zeros = torch.zeros_like(resp_mask, dtype=torch.float32)
    batch = TrainingInputBatch(
        {
            "sequences": seqs_t,
            "attention_mask": attn,
            "response_mask": resp_mask,
            "loss_mask": loss_masks,
            "rewards": rewards,
            "rollout_logprobs": lp_t,
            "rollout_expert_indices": routes_t,
            "router_padding_mask": make_router_padding_mask(attn, [seq_len] * num_seqs),
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "advantages": zeros.clone(),
        }
    )
    batch.metadata = {"response_length": resp_mask.size(1)}
    return batch


def make_batch(num_seqs: int, seq_len: int, vocab: int, seed: int = 0) -> TrainingInputBatch:
    g = torch.Generator().manual_seed(seed)
    seqs = torch.randint(1000, min(vocab, 100000), (num_seqs, seq_len), generator=g)
    num_actions = seq_len - 1
    attn = torch.ones(num_seqs, seq_len, dtype=torch.long)
    resp = torch.ones(num_seqs, num_actions, dtype=torch.long)
    zeros = torch.zeros(num_seqs, num_actions, dtype=torch.float32)
    batch = TrainingInputBatch(
        {
            "sequences": seqs,
            "attention_mask": attn,
            "response_mask": resp,
            "loss_mask": resp.clone(),
            "rewards": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "rollout_expert_indices": None,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "advantages": zeros.clone(),
        }
    )
    batch.metadata = {"response_length": num_actions}
    return batch


def run(args) -> None:
    cfg = build_config(args)
    initialize_ray(cfg)
    policy = PPORayActorGroup(
        cfg.trainer, num_nodes=1, num_gpus_per_node=args.gpus, ray_actor_type=CheckWorker, sequence_parallel_size=1
    )
    ray.get(policy.async_init_model(args.model))
    if args.r3:
        # Route geometry from the args: the driver may not see a node-local checkpoint dir.
        batch = make_batch_r3(args.num_seqs, args.seq_len, args.vocab, args.n_layers, args.n_experts, args.topk)
    else:
        batch = make_batch(args.num_seqs, args.seq_len, args.vocab)
    out = WorkerOutput.cat(policy.actor_infos, ray.get(policy.async_run_ray_method("mesh", "forward", data=batch)))
    logprobs = loss_fn_outputs_to_tensor(out.loss_fn_outputs, key="logprobs").float().cpu()
    result = {"logprobs": logprobs, "args": vars(args)}
    if args.train:
        ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch, loss_fn="cross_entropy"))
        grad_norms = ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
        result["grad_norm"] = float(grad_norms[0])
        print(f"[cp_check] cp={args.cp} grad_norm={result['grad_norm']:.8f}", flush=True)
    torch.save(result, args.out)
    finite = torch.isfinite(logprobs).all().item()
    print(f"[cp_check] cp={args.cp} packed={args.packed} saved {tuple(logprobs.shape)} finite={finite} "
          f"mean={logprobs.mean():.5f} -> {args.out}", flush=True)
    ray.shutdown()


def compare(a_path: str, b_path: str) -> None:
    ra, rb = torch.load(a_path), torch.load(b_path)
    a, b = ra["logprobs"].double(), rb["logprobs"].double()
    if "grad_norm" in ra and "grad_norm" in rb:
        print(f"[cp_check] grad_norm {ra['grad_norm']:.8f} vs {rb['grad_norm']:.8f} "
              f"(rel diff {abs(ra['grad_norm'] - rb['grad_norm']) / max(abs(ra['grad_norm']), 1e-12):.2e})")
    d = (a - b).abs()
    worst = d.flatten().argmax().item()
    print(f"[cp_check] {a_path} vs {b_path}: shape {tuple(a.shape)} mean|d| {d.mean():.3e} max|d| {d.max():.3e} "
          f"(at flat {worst}) p99 {d.flatten().quantile(0.99):.3e}; per-seq mean|d| "
          f"{[round(x, 5) for x in d.mean(dim=1).tolist()]}")
    pos = d.mean(dim=0)
    n = pos.numel()
    print("[cp_check] mean|d| by position quarter:", [f"{pos[i * n // 4:(i + 1) * n // 4].mean():.3e}" for i in range(4)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/mnt/local_storage/hackskyrl/models/dsv41_flash_l9_q")
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--packed", type=int, default=0)
    p.add_argument("--dsa-backend", default="cudnn")
    p.add_argument("--n-layers", type=int, default=9, help="--r3: checkpoint num_hidden_layers")
    p.add_argument("--n-experts", type=int, default=384, help="--r3: n_routed_experts")
    p.add_argument("--topk", type=int, default=6, help="--r3: num_experts_per_tok")
    p.add_argument("--r3", type=int, default=0, help="1: replay fixed random expert routes (layout-independent)")
    p.add_argument("--train", type=int, default=0, help="1: LoRA + stateful recompute, also report the grad norm")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--vocab", type=int, default=129280)
    p.add_argument("--out", default="/tmp/cp_check.pt")
    p.add_argument("--compare", nargs=2, default=None)
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)
