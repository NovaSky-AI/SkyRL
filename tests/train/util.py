# utility functions used for CPU tests

import threading

import torch
import torch.distributed as dist

from skyrl.train.config import (
    AlgorithmConfig,
    DataConfig,
    DataLoaderConfig,
    GeneratorConfig,
    InferenceEngineConfig,
    SamplingParams,
    SkyRLTrainConfig,
    TrainerConfig,
)


class ThreadedAllReduce:
    """Stands in for ``torch.distributed.all_reduce`` across ``world_size`` threads, one per rank."""

    def __init__(self, world_size: int):
        # The timeout turns a rank that died before reaching the collective into a failure, not a hang.
        self._barrier = threading.Barrier(world_size, timeout=60)
        self._lock = threading.Lock()
        self._contributions = []

    def __call__(self, tensor, op=dist.ReduceOp.SUM, group=None):
        with self._lock:
            self._contributions.append(tensor.clone())
        self._barrier.wait()
        stacked = torch.stack(self._contributions)
        reduced = stacked.amax(dim=0) if op == dist.ReduceOp.MAX else stacked.sum(dim=0)
        if self._barrier.wait() == 0:
            self._contributions.clear()
        self._barrier.wait()
        tensor.copy_(reduced)


def example_dummy_config():
    # TODO (sumanthrh): Cleanup overrides
    trainer_cfg = TrainerConfig(
        project_name="unit-test",
        run_name="test-run",
        logger="tensorboard",
        micro_train_batch_size_per_gpu=2,
        train_batch_size=2,
        eval_batch_size=2,
        update_epochs_per_batch=1,
        epochs=1,
        max_prompt_length=20,
        remove_microbatch_padding=False,
        seed=42,
        resume_mode="none",
        algorithm=AlgorithmConfig(
            advantage_estimator="grpo",
            kl_estimator_type="k1",
            use_kl_loss=True,
            kl_loss_coef=0.0,
            loss_reduction="token_mean",
            grpo_norm_by_std=True,
        ),
    )
    generator_cfg = GeneratorConfig(
        sampling_params=SamplingParams(max_generate_length=20),
        n_samples_per_prompt=1,
        batched=False,
        max_turns=1,
        inference_engine=InferenceEngineConfig(),
    )
    # No dataloader worker processes for CPU tests. The default (8, derived in
    # SkyRLTrainConfig.__post_init__) spawns eight fresh interpreters -- ~0.5 GB each once they
    # import torch and skyrl -- the moment a test iterates a dataloader over a handful of dummy rows.
    # StatefulDataLoader keeps that iterator alive with the trainer, so a test that leaves the loop
    # early (a crash mid-step, an early stop) pins several GB until the cyclic GC runs. On a 16 GB CI
    # runner that pushed the node past Ray's 95% memory threshold and its monitor killed the registry
    # actors, failing unrelated tests in test_ppo_utils.py.
    data_cfg = DataConfig(dataloader=DataLoaderConfig(num_workers=0))
    cfg = SkyRLTrainConfig(trainer=trainer_cfg, generator=generator_cfg, data=data_cfg)

    return cfg
