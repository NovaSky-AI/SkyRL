"""Iteration-order consistency test for ``MegatronWeightExtractor``.

When ``enable_bucketing=True``, ``extract_weights`` (the producer that streams
parameter tensors to the inference engine) and ``get_weight_metadata`` (the
consumer's ``update_info`` source) must yield the same parameters, in the
same order, with the same count. Any divergence between the two methods
breaks downstream weight-sync consumers that rely on positional alignment
between the two streams.

This test loads a Megatron ref worker for two parametrizations -- a
multimodal MoE model that exercises the bucketed grouped-export path and a
small dense model that serves as a non-MoE sanity check -- builds a fresh
``MegatronWeightExtractor`` per rank with bucketing enabled, calls
``get_weight_metadata`` and ``extract_weights`` on it, and asserts that the
two iteration sequences are identical (same length and same per-position
``name``).

Run with::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_extractor_consistency.py

"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronRefWorkerBase,
    MegatronWeightExtractor,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import str_to_torch_dtype, validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type


class _ProbeMegatronRefWorker(MegatronRefWorkerBase):
    """Test-only ``MegatronRefWorkerBase`` subclass that exposes a probe of
    ``MegatronWeightExtractor`` iteration sequences.

    The probe is added on the test side (rather than on the production
    ``MegatronRefWorkerBase``) so production code stays free of test-only
    instrumentation.
    """

    def probe_extractor_iteration_sequences(self, dtype_str: str) -> dict:
        """Return per-rank ``get_weight_metadata`` and ``extract_weights``
        name sequences captured from a fresh ``MegatronWeightExtractor``."""
        dtype = str_to_torch_dtype(dtype_str)

        extractor = MegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            enable_bucketing=True,
            bucket_size_threshold_GB=1.0,
            training_dtype=torch.bfloat16,
        )

        metadata = extractor.get_weight_metadata(dtype)
        meta_names = list(metadata["names"])

        extract_names: list[str] = []
        for chunk in extractor.extract_weights(dtype):
            extract_names.extend(chunk.names)
            del chunk

        return {
            "meta_names": meta_names,
            "extract_names": extract_names,
        }


_ProbeRefWorker = ray.remote(num_gpus=1)(_ProbeMegatronRefWorker)


def _make_ref_cfg(model_name: str) -> SkyRLTrainConfig:
    """Build a minimal Megatron ref-worker config for the consistency check."""
    is_moe = "A3B" in model_name or "MoE" in model_name
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.ref_num_gpus_per_node = 4
    cfg.trainer.ref.megatron_config.tensor_model_parallel_size = 2
    # Pipeline parallelism halves per-rank parameter storage by sharding
    # transformer layers across 2 stages. Combined with tp=2 and ep=2, this
    # yields ~8.75 B params per rank for the 35B-A3B MoE model under the L4 24 GB budget.
    cfg.trainer.ref.megatron_config.pipeline_model_parallel_size = 2 if is_moe else 1
    cfg.trainer.ref.megatron_config.expert_model_parallel_size = 2 if is_moe else 1
    cfg.trainer.ref.megatron_config.expert_tensor_parallel_size = 1
    if cfg.trainer.ref.megatron_config.transformer_config_kwargs is None:
        cfg.trainer.ref.megatron_config.transformer_config_kwargs = dict()
    cfg.trainer.ref.megatron_config.transformer_config_kwargs["fp8"] = "e4m3"
    if is_moe:
        cfg.trainer.gradient_checkpointing_use_reentrant = True
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(
            "Qwen/Qwen3.5-35B-A3B",
            id="qwen3_5_35b_a3b_mm_moe",
        ),
        pytest.param("Qwen/Qwen2.5-1.5B-Instruct", id="qwen2_5_1_5b_dense"),
    ],
)
def test_megatron_extractor_iteration_order_consistency(ray_init_fixture, model_name):
    """Per rank, assert ``get_weight_metadata`` and ``extract_weights``
    yield the same parameter names in the same order with the same count."""
    cfg = _make_ref_cfg(model_name)

    # Monkey-patch the production ``RefWorker`` symbol so
    # ``init_worker_with_type`` (which does ``importlib.import_module +
    # getattr(module, "RefWorker")`` at call time) picks up the probe-augmented
    # subclass instead. Restored unconditionally in ``finally``.
    _orig_ref_worker = _megatron_worker_mod.RefWorker
    _megatron_worker_mod.RefWorker = _ProbeRefWorker

    try:
        ref = init_worker_with_type(
            "ref",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=4,
            cfg=cfg,
        )
        results = ray.get(ref.async_run_ray_method("pass_through", "probe_extractor_iteration_sequences", "bfloat16"))
        assert results, "expected at least one Megatron ref rank"

        for rank_idx, result in enumerate(results):
            meta_names = result["meta_names"]
            extract_names = result["extract_names"]
            assert len(meta_names) > 0, f"[rank {rank_idx}] empty iteration sequence"
            assert len(meta_names) == len(extract_names), (
                f"[rank {rank_idx}] count divergence: "
                f"get_weight_metadata yielded {len(meta_names)} params, "
                f"extract_weights yielded {len(extract_names)}"
            )
            # First-divergence index for a useful failure message.
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(meta_names, extract_names)) if a != b),
                None,
            )
            assert first_diff is None, (
                f"[rank {rank_idx}] order divergence at index {first_diff}: "
                f"metadata={meta_names[first_diff]!r}, "
                f"extract={extract_names[first_diff]!r}"
            )
            print(
                f"[rank {rank_idx}] iteration sequences match: N={len(meta_names)} params, "
                f"first={meta_names[0]!r}, last={meta_names[-1]!r},"
            )
    finally:
        _megatron_worker_mod.RefWorker = _orig_ref_worker
