"""Channel-agreement test for the Megatron ``WeightSource``.

``metadata()`` declares what iteration will yield, and the trainer engine sizes
the worker's receive buffers -- and, in packed mode, cuts its chunk boundaries --
from it. A source that disagrees splits the stream differently on each side.
``NCCLTrainerWeightTransferEngine._checked_iter`` catches that at runtime and
names the first divergent parameter; this test gets there first.

Two parametrizations: a multimodal MoE model that exercises the grouped expert
export, and a small dense model as a non-MoE sanity check.

Run with::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_weight_source.py

"""

import pytest
import ray

from skyrl.backends.skyrl_train.weight_sync.sources import MegatronWeightSource
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronRefWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type


class _ProbeMegatronRefWorker(MegatronRefWorkerBase):
    """Exposes a probe of the Megatron ``WeightSource``'s two channels.

    Test-side rather than on the production worker, so production code stays free
    of test-only instrumentation.
    """

    def probe_source_channel_agreement(self, dtype_str: str) -> dict:
        """Return this rank's ``metadata()`` and iteration name sequences.

        ``metadata()`` caches its dry export and iteration runs a second one, so
        this also covers the two exports agreeing.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        source = MegatronWeightSource(self.bridge, self.actor_module, str_to_torch_dtype(dtype_str))

        meta = source.metadata()
        meta_names = [m.name for m in meta]
        meta_shapes = [list(m.shape) for m in meta]

        iter_names: list[str] = []
        iter_shapes: list[list[int]] = []
        for name, tensor in source:
            iter_names.append(name)
            iter_shapes.append(list(tensor.shape))
            del tensor

        return {
            "meta_names": meta_names,
            "meta_shapes": meta_shapes,
            "iter_names": iter_names,
            "iter_shapes": iter_shapes,
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
    cfg.trainer.ref.megatron_config.pipeline_model_parallel_size = 2 if is_moe else 1
    cfg.trainer.ref.megatron_config.expert_model_parallel_size = 2 if is_moe else 1
    cfg.trainer.ref.megatron_config.expert_tensor_parallel_size = 1
    if cfg.trainer.ref.megatron_config.transformer_config_kwargs is None:
        cfg.trainer.ref.megatron_config.transformer_config_kwargs = dict()
    cfg.trainer.ref.megatron_config.transformer_config_kwargs["fp8"] = "e4m3"
    # Cap MoE layers to fit the L4 24 GB budget; parameter iteration order
    # (the only thing this test checks) is preserved with any num_layers > 0.
    # MTP layers hit an attention-mask-type assertion in this ref-only setup.
    if is_moe:
        cfg.trainer.ref.megatron_config.transformer_config_kwargs["num_layers"] = 2
        cfg.trainer.ref.megatron_config.transformer_config_kwargs["mtp_num_layers"] = 0
    if is_moe:
        cfg.trainer.gradient_checkpointing_use_reentrant = True
    if "qwen3.5" in model_name.lower():  # use LM only path for qwen3.5
        cfg.trainer.ref.language_model_only = True
        cfg.generator.inference_engine.language_model_only = True
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
def test_megatron_source_channel_agreement(ray_init_fixture, model_name):
    """Per rank, assert ``metadata()`` and iteration yield the same parameter
    names and shapes, in the same order, with the same count."""
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
        results = ray.get(ref.async_run_ray_method("pass_through", "probe_source_channel_agreement", "bfloat16"))
        assert results, "expected at least one Megatron ref rank"

        for rank_idx, result in enumerate(results):
            meta_names = result["meta_names"]
            iter_names = result["iter_names"]
            assert len(meta_names) > 0, f"[rank {rank_idx}] empty iteration sequence"
            assert len(meta_names) == len(iter_names), (
                f"[rank {rank_idx}] count divergence: "
                f"metadata() declared {len(meta_names)} params, "
                f"iteration yielded {len(iter_names)}"
            )
            # First-divergence index for a useful failure message.
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(
                            zip(meta_names, result["meta_shapes"]),
                            zip(iter_names, result["iter_shapes"]),
                        )
                    )
                    if a != b
                ),
                None,
            )
            assert first_diff is None, (
                f"[rank {rank_idx}] channel divergence at index {first_diff}: "
                f"metadata()={meta_names[first_diff]!r} {result['meta_shapes'][first_diff]}, "
                f"iteration={iter_names[first_diff]!r} {result['iter_shapes'][first_diff]}"
            )
            print(
                f"[rank {rank_idx}] channels agree: N={len(meta_names)} params, "
                f"first={meta_names[0]!r}, last={meta_names[-1]!r},"
            )
    finally:
        _megatron_worker_mod.RefWorker = _orig_ref_worker
