"""Check Nemotron-120B logprobs on 8 trainer + 8 inference GPUs, without an optimizer.

uv run --isolated --extra megatron -m examples.model_checks.run_logprobs
uv run --isolated --extra megatron -m examples.model_checks.run_logprobs --full-ft

Start Ray across both nodes; LoRA export requires a shared /shared mount.
"""

import argparse
import asyncio
import base64
import io
import math
from contextlib import asynccontextmanager

import numpy as np
import ray
import torch

from examples.model_checks.logprob_checks import (
    build_probe_sequences,
    check_agreement,
    compare_logprobs,
    perturb_adapters,
)
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.inference_servers.setup import (
    build_new_inference_client,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.routed_experts import (
    compact_routed_expert_indices,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)
from skyrl.train.utils.utils import initialize_ray
from skyrl.utils.tok import get_tokenizer


@asynccontextmanager
async def open_runtime(cfg, tokenizer):
    if ray.is_initialized():
        raise RuntimeError("Run in a fresh driver on an owned Ray cluster")
    try:
        initialize_ray(cfg)
        client, setup = build_new_inference_client(cfg, tokenizer)
        try:
            policy = PPORayActorGroup(
                cfg.trainer,
                num_nodes=cfg.trainer.placement.policy_num_nodes,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                ray_actor_type=ray.remote(LoRALogprobWorker),
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            ray.get(policy.async_init_model(cfg.trainer.policy.model.path))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "init_weight_sync_state",
                    client,
                    cfg.generator.inference_engine,
                )
            )
            yield policy, client
        finally:
            try:
                await client.aclose()
            finally:
                if setup.router is not None:
                    setup.router.shutdown()
                for group in setup.server_groups:
                    group.shutdown()
    finally:
        # Disconnect this driver and its non-detached actors, not the cluster.
        ray.shutdown()


class LoRALogprobWorker(MegatronPolicyWorkerBase):
    def perturb_test_adapter(self, multiplier=10):
        parameters = (
            (f"chunk{index}.{name}", parameter)
            for index, chunk in enumerate(self.actor_module)
            for name, parameter in chunk.named_parameters()
        )
        return perturb_adapters(parameters, multiplier=multiplier)


def build_batch(sequences, pad_token_id, routes=None):
    responses = [tokens[1:] for tokens in sequences]
    masks = [[1] * len(tokens) for tokens in responses]
    tokens, attention, response, rewards, loss_mask, _, replay_routes, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id, [[tokens[0]] for tokens in sequences], responses, masks, masks, rollout_expert_indices=routes
    )
    batch = TrainingInputBatch(
        {
            "sequences": tokens,
            "attention_mask": attention,
            "response_mask": response,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "rollout_expert_indices": replay_routes,
            "rollout_logprobs": torch.zeros_like(loss_mask),
            "action_log_probs": torch.zeros_like(loss_mask),
            "base_action_log_probs": torch.zeros_like(loss_mask),
            "advantages": torch.zeros_like(loss_mask),
        }
    )
    if routes is not None:
        batch["router_padding_mask"] = make_router_padding_mask(attention, [len(route) for route in routes])
    batch.metadata = {"response_length": response.shape[1]}
    return batch


def score_trainer(policy, batch):
    results = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch, loss_fn="cross_entropy"))
    output = WorkerOutput.cat(policy.actor_infos, results)
    lengths = batch["response_mask"].sum(dim=1).tolist()
    assert [len(row["logprobs"]) for row in output.loss_fn_outputs] == lengths
    # The loss path already removes padding from each sample's outputs.
    scores = [score for row in output.loss_fn_outputs for score in row["logprobs"]]
    if not all(map(math.isfinite, scores)):
        raise ValueError("nonfinite trainer logprobs")
    return scores


async def score_sampler(client, sequences, model, capture_routes=False):
    await client.reset_prefix_cache()
    scores, routes = [], [] if capture_routes else None
    for tokens in sequences:
        request = {
            "model": model,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 1.0,
            "n": 1,
            "stream": False,
            "prompt_logprobs": 0,
            "add_special_tokens": False,
            "return_token_ids": True,
        }
        if capture_routes:
            request["routed_experts_prompt_start"] = 0
        result = await client.completion({"json": request})
        assert len(result["choices"]) == 1
        choice = result["choices"][0]
        assert choice["prompt_token_ids"] == tokens
        values = choice["prompt_logprobs"]
        assert len(values) == len(tokens) and values[0] is None
        selected = [values[index][str(token)]["logprob"] for index, token in enumerate(tokens[1:], 1)]
        assert all(math.isfinite(value) and value != -9999 for value in selected)
        scores.extend(selected)
        if capture_routes:
            captured = compact_routed_expert_indices(
                np.load(io.BytesIO(base64.b64decode(choice["routed_experts"], validate=True)), allow_pickle=False)
            )
            assert captured.shape[0] == len(tokens) and all(captured.shape[1:])
            routes.append(captured)
    return scores, routes


async def publish(policy, client, cfg):
    await client.pause_generation()
    try:
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "broadcast_to_inference_engines",
                client,
                cfg.generator.inference_engine,
            )
        )
    finally:
        await client.resume_generation()


async def check_logprobs(policy, client, cfg, tokenizer, report=None):
    """Publish → score → perturb → verify stale receiver → publish → score again."""
    sequences = build_probe_sequences(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    lora = cfg.trainer.policy.model.lora.rank > 0
    replay = cfg.trainer.policy.megatron_config.moe_enable_routing_replay
    assert replay == cfg.generator.inference_engine.enable_return_routed_experts
    model = resolve_policy_model_name(cfg) if lora else client.model_name
    colocated = cfg.trainer.placement.colocate_all
    scores = {} if report is None else report
    scores["tokens"] = sequences
    for phase in ["zero", "perturbed"] if lora else ["full_ft"]:
        if phase == "perturbed":
            ray.get(policy.async_run_ray_method("pass_through", "perturb_test_adapter"))
        current = scores[phase] = {}
        if colocated:
            # Offload the trainer base before waking the receiver for adapter export.
            policy.offload_to_cpu(offload_optimizer=True, offload_model=lora)
            await client.wake_up(tags=["weights"])
        if phase == "perturbed":
            if colocated:
                await client.wake_up(tags=["kv_cache"])
            current["stale"], _ = await score_sampler(client, sequences, model, replay)
            assert compare_logprobs(scores["zero"]["inference"], current["stale"])["max_abs"] <= 1e-6
            if colocated:
                await client.sleep()
                await client.wake_up(tags=["weights"])
        await publish(policy, client, cfg)
        if colocated:
            if not lora:
                policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await client.wake_up(tags=["kv_cache"])
        current["inference"], routes = await score_sampler(client, sequences, model, replay)
        current["repeat"], repeat_routes = await score_sampler(client, sequences, model, replay)
        if replay:
            current.update(
                routes=[route.tolist() for route in routes], repeat_routes=[r.tolist() for r in repeat_routes]
            )
        if colocated:
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)
        current["trainer"] = score_trainer(policy, build_batch(sequences, pad_id, routes))
        difference = compare_logprobs(current["trainer"], current["inference"])
        print(f"{phase}: {difference}", flush=True)
        check_agreement(difference, mean_atol=0.05, max_atol=0.5)
        assert compare_logprobs(current["inference"], current["repeat"])["max_abs"] <= 1e-6
        if replay:
            assert all(np.array_equal(route, repeat) for route, repeat in zip(routes, repeat_routes, strict=True))
    if lora:
        for backend in ("trainer", "inference"):
            assert compare_logprobs(scores["zero"][backend], scores["perturbed"][backend])["max_abs"] > 1e-6
    return scores


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ft", action="store_true")
    args = parser.parse_args()
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.logger = "console"
    parallel = cfg.trainer.policy.megatron_config
    parallel.tensor_model_parallel_size = 8
    parallel.expert_model_parallel_size = 8
    parallel.expert_tensor_parallel_size = 1
    parallel.lora_config.merge_lora = False
    cfg.trainer.policy.model.lora.rank = 0 if args.full_ft else 8
    cfg.trainer.policy.model.lora.alpha = 16
    cfg.trainer.policy.model.lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    cfg.trainer.policy.model.lora.lora_sync_path = "/shared/skyrl-logprob-adapter"
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.tensor_parallel_size = 8
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.engine_init_kwargs = {"max_model_len": 4096}
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        await check_logprobs(policy, client, cfg, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
