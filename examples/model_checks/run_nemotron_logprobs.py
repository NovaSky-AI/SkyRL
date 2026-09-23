"""Check Nemotron-120B logprobs on 8 trainer + 8 inference GPUs, without an optimizer.

uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs
uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs --full-ft

Start Ray across both nodes; LoRA export requires a shared /shared mount.
"""

import argparse
import asyncio
import math
from contextlib import asynccontextmanager

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
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
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


def perturb_trainer(policy, multiplier=10):
    return ray.get(policy.async_run_ray_method("pass_through", "perturb_test_adapter", multiplier))


class LoRALogprobWorker(MegatronPolicyWorkerBase):
    def perturb_test_adapter(self, multiplier=10):
        parameters = (
            (f"chunk{index}.{name}", parameter)
            for index, chunk in enumerate(self.actor_module)
            for name, parameter in chunk.named_parameters()
        )
        return perturb_adapters(parameters, multiplier=multiplier)


def build_batch(sequences, pad_token_id):
    responses = [tokens[1:] for tokens in sequences]
    masks = [[1] * len(tokens) for tokens in responses]
    tokens, attention, response, rewards, loss_mask, _, replay_routes, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id, [[tokens[0]] for tokens in sequences], responses, masks, masks
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


async def score_sampler(client, sequences, model):
    await client.reset_prefix_cache()
    scores = []
    for tokens in sequences:
        result = await client.sample(
            {
                "json": {
                    "model": model,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                    "sampling_params": {"max_tokens": 1, "temperature": 1.0},
                    "num_samples": 1,
                    "prompt_logprobs": True,
                }
            }
        )
        values = result["prompt_logprobs"]
        assert values is not None and len(values) == len(tokens)
        assert values[0] is None and all(value is not None for value in values[1:])
        if not all(map(math.isfinite, values[1:])):
            raise ValueError("nonfinite sampler logprobs")
        scores.extend(values[1:])
    return scores


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


async def check_logprobs(policy, client, cfg, tokenizer):
    sequences = build_probe_sequences(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_id)
    lora = cfg.trainer.policy.model.lora.rank > 0
    model = resolve_policy_model_name(cfg) if lora else client.model_name
    colocated = cfg.trainer.placement.colocate_all
    scores = {}
    for phase in ["zero", "perturbed"] if lora else ["full_ft"]:
        if phase == "perturbed":
            perturb_trainer(policy)
        trainer = score_trainer(policy, batch)
        if colocated:
            policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
            await client.wake_up(tags=["weights"])
        await publish(policy, client, cfg)
        if colocated:
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await client.wake_up(tags=["kv_cache"])
        inference = await score_sampler(client, sequences, model)
        repeat = await score_sampler(client, sequences, model)
        difference = compare_logprobs(trainer, inference)
        print(f"{phase}: {difference}", flush=True)
        check_agreement(difference, mean_atol=0.05, max_atol=0.5)
        assert compare_logprobs(inference, repeat)["max_abs"] <= 1e-6
        scores[phase] = {"trainer": trainer, "inference": inference}
        if colocated:
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)
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
