"""Hosted Fireworks specialization of SkyRL's SFT trainer."""

from __future__ import annotations

import asyncio

from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.backends.fireworks.sft import FireworksSFTDispatch
from skyrl.train.sft_trainer import SFTTrainer


class FireworksSFTTrainer(SFTTrainer):
    """Reuse the native SFT loop while Fireworks owns model computation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fireworks_runtime: FireworksRuntime | None = None
        self._ray_gpu_monitor = None
        self._num_training_gpus = 1

    def _init_tracker(self) -> None:
        self.tracker = None

    def _init_workers(self) -> None:
        if self.is_vlm:
            raise ValueError("Fireworks SFT currently supports text-only models")
        runtime = FireworksRuntime.connect(
            config=self.sft_cfg.fireworks,
            tokenizer=self.tokenizer,
            tokenizer_model=self.sft_cfg.model.path,
            lora_rank=self.sft_cfg.model.lora.rank,
            learning_rate=self.sft_cfg.optimizer_config.lr,
            create_deployment=False,
        )
        self._fireworks_runtime = runtime
        try:
            self.dispatch = FireworksSFTDispatch(
                runtime,
                self.sft_cfg.fireworks,
                self.sft_cfg.optimizer_config,
            )
            super()._init_tracker()
        except BaseException:
            asyncio.run(runtime.close())
            self._fireworks_runtime = None
            raise

    def _validate_batch_parallelism(self) -> None:
        if self.sft_cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def save_checkpoint(self) -> str:
        raise NotImplementedError("Fireworks SFT does not support persistent checkpoints")

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if self._fireworks_runtime is not None:
                asyncio.run(self._fireworks_runtime.close())
                self._fireworks_runtime = None
