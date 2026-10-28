"""SkyRL entrypoint for the CholecT50 GRPO (RLVR) run.

Mirrors ``examples/train/geometry3k/geometry3k_entrypoint.py``: register the
custom env class, then run the standard SkyRL PPO/GRPO experiment from the
Hydra-style CLI overrides in ``run_cholec_neuron.sh``.
"""
import multiprocessing as mp
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

mp.set_start_method("spawn", force=True)


def apply_vision_rope_device_patch() -> None:
    """Fix a Qwen3.5 vision-tower RoPE device mismatch.

    ``Qwen3_5VisionRotaryEmbedding`` registers ``inv_freq`` as a *non-persistent*
    buffer (``persistent=False``). Under FSDP ``cpu_offload`` that buffer is left on
    CPU while activations are on the accelerator, and the vision RoPE forward does
    ``position_ids * self.inv_freq`` without moving the buffer to the tensor's device
    — unlike the *text* rotary embedding, which already does ``.to(x.device)``. This
    monkeypatch makes ``inv_freq`` follow ``position_ids.device``, so the same
    mismatch cannot occur on ``neuron`` either.
    """
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35

    cls = getattr(_qwen35, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is None:
        print("[cholec] WARN: Qwen3_5VisionRotaryEmbedding not found; skipping patch", flush=True)
        return

    # transformers 5.18 changed the vision RoPE forward to ``forward(self, x,
    # position_ids)`` and to return ``(cos, sin)``. Rather than reimplement that (and
    # drift from upstream), keep the original forward and only move the buffer it
    # reads, which is the single device-unsafe step this patch exists for.
    _orig_forward = cls.forward

    @torch.no_grad()
    def _forward(self, x, position_ids):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
        return _orig_forward(self, x, position_ids)

    cls.forward = _forward
    print("[cholec] patched Qwen3_5VisionRotaryEmbedding.forward (device-safe inv_freq)", flush=True)


def apply_neuron_vision_interp_patch() -> None:
    """Force Qwen3.5 vision position-interpolation precompute onto CPU.

    ``get_vision_interpolation_indices_and_weights`` uses ``torch.repeat_interleave``
    with a dynamic ``repeats`` tensor. neuronx-cc cannot compile that op (Torch-MLIR
    lowering fails / BIRCodeGenLoop crashes), so the small ``(total_thw, n_taps)``
    index/weight tensors are computed on CPU and moved back. Same fix the TRL
    trainium2 leg ships (``_patch_vision_interp``); eager mode already falls back
    to CPU for the same op, so this is a sync, not a correctness, change.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as _mq

    _orig = getattr(_mq, "get_vision_interpolation_indices_and_weights", None)
    if _orig is None:
        print("[cholec] WARN: get_vision_interpolation_indices_and_weights not found; skipping patch", flush=True)
        return

    def _patched(grid_thw, *args, **kwargs):
        dev = grid_thw.device
        if dev.type == "neuron":
            indices, weights = _orig(grid_thw.to("cpu"), *args, **kwargs)
            return indices.to(dev), weights.to(dev)
        return _orig(grid_thw, *args, **kwargs)

    _mq.get_vision_interpolation_indices_and_weights = _patched
    print("[cholec] patched get_vision_interpolation_indices_and_weights (repeat_interleave on CPU)", flush=True)


def apply_neuron_vision_seqlens_patch() -> None:
    """Force the dynamic ``repeat_interleave`` in the vision seqlens helper onto CPU.

    ``get_vision_cu_seqlens`` builds per-frame attention boundaries with
    ``torch.repeat_interleave(h*w, grid_thw[:, 0])`` -- a dynamic ``repeats`` tensor
    that neuronx-cc cannot lower (its internal ``index_select`` bounds-check raises
    "No active NKI context"). ``get_vision_attention_seqlens`` calls this helper via the
    ``transformers.vision_utils`` module global, so patching the helper there covers the
    Qwen3.5 vision tower. The tiny ``(num_frames+1,)`` result is computed on CPU and moved
    back; eager mode already falls back for the same op, so this is a sync, not a
    correctness, change.
    """
    import transformers.vision_utils as _vu

    _orig = getattr(_vu, "get_vision_cu_seqlens", None)
    if _orig is None:
        print("[cholec] WARN: get_vision_cu_seqlens not found; skipping patch", flush=True)
        return

    def _patched(grid_thw, merge_temporal=False, kwargs=None):
        dev = grid_thw.device
        if dev.type == "neuron":
            return _orig(grid_thw.to("cpu"), merge_temporal=merge_temporal, kwargs=kwargs).to(dev)
        return _orig(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)

    _vu.get_vision_cu_seqlens = _patched
    print("[cholec] patched get_vision_cu_seqlens (repeat_interleave on CPU)", flush=True)


@ray.remote(num_cpus=1)
def cholec_entrypoint(cfg: SkyRLTrainConfig):
    apply_vision_rope_device_patch()
    apply_neuron_vision_interp_patch()
    apply_neuron_vision_seqlens_patch()
    register(
        id="cholec",
        entry_point="examples.train.cholec.env:CholecEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(cholec_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
