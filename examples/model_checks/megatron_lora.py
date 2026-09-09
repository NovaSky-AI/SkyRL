"""Native SkyRL calls used by the standalone LoRA score diagnostic."""

from contextlib import asynccontextmanager
import math

import ray
import torch

from examples.model_checks.lora_logprobs import perturb_adapters
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.inference_servers.setup import build_new_inference_client
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import initialize_ray


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
                    "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
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
    tokens, attention, response, rewards, loss_mask, _, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id, [[tokens[0]] for tokens in sequences], responses, masks, masks
    )
    batch = TrainingInputBatch(
        {
            "sequences": tokens,
            "attention_mask": attention,
            "response_mask": response,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "rollout_expert_indices": None,
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
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
    finally:
        await client.resume_generation()
