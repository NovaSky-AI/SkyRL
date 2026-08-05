"""Weight-source iteration bench: decompose sync cost WITHOUT vLLM/engine.

Spawns a multinode Megatron policy (LoRA optional), builds the RDT WeightSource
on every rank, and times full iterations split into buckets:
  expert_gather / lora_merge / bridge_nonexpert.
Usage:
  uv run --isolated --extra megatron python tests/backends/skyrl_train/gpu/source_bench.py \
      <model> <num_nodes> <gpus_per_node> <tp> <pp> <ep> <lora_rank> <iters> [--pp-local]

Default legs are stacked-expert vs plain-bridge (see MegatronWorker
rdt_source_bench). With ``--pp-local`` the legs become gather-to-all vs PP-local
gather instead, including the byte check that the groups a stage serves PP-locally
carry the same tensors it would have gathered to all (rdt_pp_local_bench).
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
method = "rdt_pp_local_bench" if "--pp-local" in sys.argv else "rdt_source_bench"
res = ray.get(grp.async_run_ray_method("pass_through", method))
print(f"BENCH RESULTS ({method}, rank0):", res[0])
for i, r in enumerate(res):
    print(f"rank{i}:", r)
if method == "rdt_pp_local_bench":
    bad = [i for i, r in enumerate(res) if not r.get("check", {}).get("ok")]
    print("PP-LOCAL BYTE CHECK:", "PASS on all ranks" if not bad else f"FAILED on ranks {bad}")
