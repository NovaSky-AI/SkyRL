"""Score one sequence, two identical copies, and a repeat on a pristine Megatron trainer."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import ray
import torch

from examples.model_checks.batch_sensitivity import (
    compare_batches,
    compare_rows,
    load_fixture,
)
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import initialize_ray
from skyrl.utils.tok import get_tokenizer


class BatchSensitivityWorker(MegatronPolicyWorkerBase):
    def score_zero_adapter(self, data):
        counts = [
            torch.count_nonzero(parameter.detach())
            for chunk in self.actor_module
            for name, parameter in chunk.named_parameters()
            if ".adapter.linear_out." in name
        ]
        if not counts:
            raise RuntimeError("No LoRA output parameters found")
        if int(torch.stack(counts).sum().cpu()) != 0:
            raise RuntimeError("Expected zero-initialized LoRA output parameters")
        # Preserve the observed production logprob path, rather than SFT forward-only.
        return self.forward(data, loss_fn=None)


def build_batch(tokens, copies, pad_token_id):
    responses = [tokens[1:]] * copies
    masks = [[1] * (len(tokens) - 1)] * copies
    sequences, attention, response, rewards, loss_mask, _, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id, [[tokens[0]]] * copies, responses, masks, masks
    )
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
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


def score_batch(policy, batch, copies, positions):
    results = ray.get(policy.async_run_ray_method("mesh", "score_zero_adapter", data=batch))
    output = WorkerOutput.cat(policy.actor_infos, results)
    rows = [item["logprobs"] for item in output.loss_fn_outputs]
    if len(rows) != copies or any(len(row) != positions for row in rows):
        raise ValueError("Returned scores do not match requested rows and scored positions")
    for row in rows:
        compare_rows(row, row)
    return rows


def write_report(output_dir, report):
    temporary = output_dir / "batch-sensitivity.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output_dir / "batch-sensitivity.json")


def load_config(args, positions):
    overrides = json.loads(args.backend_config.read_text())
    overrides["trainer.policy.model.path"] = str(args.model.resolve())
    overrides["trainer.log_path"] = str(args.output_dir / "runtime-logs")
    overrides["trainer.policy.torch_profiler_config"] = {"enable": False}
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    trainer = cfg.trainer
    parallel = trainer.policy.megatron_config
    placement = trainer.placement
    if trainer.strategy != "megatron" or not trainer.bf16:
        raise ValueError("Expected a BF16 Megatron trainer")
    if trainer.policy.model.lora.rank <= 0:
        raise ValueError("Expected a positive LoRA rank")
    if parallel.pipeline_model_parallel_size != 1 or parallel.context_parallel_size != 1:
        raise ValueError("Batch sensitivity requires PP1 and CP1")
    if placement.policy_num_nodes * placement.policy_num_gpus_per_node != parallel.tensor_model_parallel_size:
        raise ValueError("Batch sensitivity requires DP1: trainer GPU count must equal TP")
    if trainer.remove_microbatch_padding:
        raise ValueError("Batch sensitivity requires rectangular padded batches")
    if trainer.max_tokens_per_microbatch < 2 * positions:
        raise ValueError("Token budget must keep both duplicate rows in one microbatch")
    return cfg, overrides


def run(args, report):
    fixture = load_fixture(args.fixture, args.model)
    tokens = fixture["tokens"]
    cfg, overrides = load_config(args, len(tokens))
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")
    single = build_batch(tokens, 1, pad)
    duplicate = build_batch(tokens, 2, pad)
    for key in ("sequences", "attention_mask", "response_mask", "loss_mask"):
        if not torch.equal(single[key].expand_as(duplicate[key]), duplicate[key]):
            raise ValueError(f"Single and duplicate batches differ in {key}")
    report.update(
        fixture=fixture,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        backend_config=overrides,
        input_shape=list(single["sequences"].shape),
        scored_positions=len(tokens) - 1,
        scores={},
        seconds={},
        scoring_path="MegatronPolicyWorkerBase.forward(loss_fn=None)",
    )
    if ray.is_initialized():
        raise RuntimeError("Run in a fresh driver on an owned Ray allocation")
    try:
        initialize_ray(cfg)
        trainer = cfg.trainer
        placement = trainer.placement
        policy = PPORayActorGroup(
            trainer,
            num_nodes=placement.policy_num_nodes,
            num_gpus_per_node=placement.policy_num_gpus_per_node,
            ray_actor_type=ray.remote(BatchSensitivityWorker),
            sequence_parallel_size=trainer.policy.sequence_parallel_size,
            record_memory=trainer.policy.record_memory,
        )
        started = perf_counter()
        ray.get(policy.async_init_model(cfg.trainer.policy.model.path))
        report["seconds"]["initialize"] = perf_counter() - started
        for name, batch, copies in (("single", single, 1), ("duplicate", duplicate, 2), ("repeat", single, 1)):
            started = perf_counter()
            report["scores"][name] = score_batch(policy, batch, copies, len(tokens) - 1)
            report["seconds"][name] = perf_counter() - started
            write_report(args.output_dir, report)
        report["comparisons"] = compare_batches(report["scores"])
    finally:
        try:
            write_report(args.output_dir, report)
        finally:
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local HF snapshot at the fixture's pinned revision")
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "diagnostic_only": True}
    try:
        run(args, report)
        report["completed"] = True
    finally:
        write_report(args.output_dir, report)


if __name__ == "__main__":
    main()
