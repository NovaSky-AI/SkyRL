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
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35

    def _forward(self, position_ids):
        inv_freq = self.inv_freq.to(position_ids.device)
        return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)

    cls = getattr(_qwen35, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is None:
        print("[cholec] WARN: Qwen3_5VisionRotaryEmbedding not found; skipping patch", flush=True)
        return
    cls.forward = _forward
    print("[cholec] patched Qwen3_5VisionRotaryEmbedding.forward (device-safe inv_freq)", flush=True)


@ray.remote(num_cpus=1)
def cholec_entrypoint(cfg: SkyRLTrainConfig):
    apply_vision_rope_device_patch()
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
