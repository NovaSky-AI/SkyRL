"""Weight-source iteration bench: decompose sync cost WITHOUT vLLM/engine.

Spawns a multinode Megatron policy (LoRA optional), builds the RDT WeightSource
on every rank, and times full iterations split into buckets:
  expert_gather / lora_merge / bridge_nonexpert.
Usage:
  uv run --isolated --extra megatron python tests/backends/skyrl_train/gpu/source_bench.py \
      <model> <num_nodes> <gpus_per_node> <tp> <pp> <ep> <lora_rank> <iters>
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import SkyRLLoraConfig
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

model, nn, gpn, tp, pp, ep, lora_rank, iters = sys.argv[1:9]

cfg = SkyRLTrainConfig()
cfg.trainer.policy.model.path = model
cfg.trainer.strategy = "megatron"
if int(lora_rank) > 0:
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=int(lora_rank), alpha=int(lora_rank))
cfg.trainer.policy.megatron_config.tensor_model_parallel_size = int(tp)
cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = int(pp)
cfg.trainer.policy.megatron_config.expert_model_parallel_size = int(ep)
cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1

ray_init_for_tests()
grp = init_worker_with_type(
    "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=int(gpn), num_nodes=int(nn), cfg=cfg
)
# inject bench as a method and run on all ranks
res = ray.get(grp.async_run_ray_method("pass_through", "rdt_source_bench"))
print("BENCH RESULTS (rank0):", res[0])
for i, r in enumerate(res):
    print(f"rank{i}:", r)
