"""
uv run --isolated --extra dev --extra fsdp pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/test_meta_init.py

Multi-rank FSDP2 meta-init correctness under sequence parallelism.
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.utils import get_free_port
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPRefWorkerBase
from skyrl.train.config import AlgorithmConfig, RefConfig, TrainerConfig

# Use a model that does not tie its embeddings (shares one weight tensor between
# the input embedding and the output `lm_head`), since `FSDPRefWorkerBase.init_model`
# gates meta-init on `not tie_word_embeddings`, leading to a 'tied' model (e.g. Qwen3-0.6B)
# skipping meta-init entirely, so it can't catch the regression we're testing against
# https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/modeling_utils.py#L2582
MODEL_NAME = "llamafactory/tiny-random-Llama-3"
SERVER_HOST = "127.0.0.1"
WORLD_SIZE = 2
SP_SIZE = 2
SEQ_LEN = 128


@ray.remote(num_gpus=1)
class MetaInitProbeRefWorker(FSDPRefWorkerBase):
    def record_inv_freq(self) -> list[dict]:
        out: list[dict] = []
        for name, buf in self.model.model.named_buffers():
            if not name.endswith("inv_freq") or name.endswith("original_inv_freq"):
                # `original_inv_freq` is transformers' pre-cast backup of `inv_freq`.
                # It has the same data, so skip it to avoid duplicate records
                continue
            out.append(
                {
                    "name": name,
                    "n_nan": int(torch.isnan(buf).sum().item()),
                    "dtype": str(buf.dtype),
                    "first5": buf.detach().float().cpu().tolist()[:5],
                }
            )
        return out

    def forward_and_count_nan(self, sequences: torch.Tensor) -> int:
        sequences = sequences.to("cuda")
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            log_probs = self.model(sequences, sequences.shape[1] - 1, torch.ones_like(sequences))
        return int(torch.isnan(log_probs).sum().item())


@pytest.mark.usefixtures("ray_init_fixture")
@pytest.mark.parametrize("bf16", [True, False])
def test_meta_init_inv_freq_finite_under_sp(bf16: bool) -> None:
    """Meta-init under SP=2 must leave every rank's rotary `inv_freq` finite and its
    forward NaN-free, for both bf16 (which triggers the dtype mismatch) and fp32."""
    cfg = TrainerConfig(
        algorithm=AlgorithmConfig(temperature=0.1),  # Placeholder non-None temperature
        ref=RefConfig(sequence_parallel_size=SP_SIZE),
        bf16=bf16,
    )
    # All WORLD_SIZE worker actors must agree on `MASTER_PORT`; pick once on the driver
    master_port = get_free_port()

    actors = [
        MetaInitProbeRefWorker.remote(
            cfg=cfg,
            world_size=WORLD_SIZE,
            rank=r,
            local_rank=0,
            master_addr=SERVER_HOST,
            master_port=master_port,
            sequence_parallel_size=SP_SIZE,
        )
        for r in range(WORLD_SIZE)
    ]
    # Stand up the NCCL process group + device mesh, then build the FSDP2 model
    ray.get([a.init_worker_process_group.remote() for a in actors])
    ray.get([a.init_model.remote(MODEL_NAME) for a in actors])

    inv_freq_records = ray.get([a.record_inv_freq.remote() for a in actors])
    for rank, records in enumerate(inv_freq_records):
        assert records, f"rank {rank}: no inv_freq buffers found — test expects a model with rotary embeddings"
        for record in records:
            assert record["dtype"] == "torch.float32", (
                f"rank {rank}: {record['name']} is {record['dtype']} after FSDP init, expected fp32 "
                f"(non-rank-0 meta-init cast it to bf16, diverging from rank-0's fp32). Checked in addition to "
                f"the finiteness assert below, which a finite-but-garbage value (e.g. ~2e7) would pass."
            )
            assert record["n_nan"] == 0, (
                f"rank {rank}: {record['name']} non-finite after FSDP init "
                f"(n_nan={record['n_nan']}, dtype={record['dtype']}, first5={record['first5']})"
            )

    sequences = torch.randint(10, 10_000, (1, SEQ_LEN), dtype=torch.long)
    nan_counts = ray.get([a.forward_and_count_nan.remote(sequences) for a in actors])
    for rank, n_nan in enumerate(nan_counts):
        assert n_nan == 0, f"rank {rank}: log_probs has {n_nan} NaN positions under SP={SP_SIZE} (bf16={bf16})"
