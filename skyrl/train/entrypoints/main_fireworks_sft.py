"""Direct entrypoint for SFT on a hosted Fireworks trainer."""

from __future__ import annotations

import sys

from loguru import logger

from skyrl.train.config import (
    SFTConfig,
    build_skyrl_config_for_sft,
    validate_fireworks_sft_cfg,
)
from skyrl.train.fireworks_sft_trainer import FireworksSFTTrainer


def run(cfg: SFTConfig) -> dict | None:
    """Run the native SFT loop without attaching to Ray."""

    validate_fireworks_sft_cfg(cfg)
    trainer = FireworksSFTTrainer(cfg, skyrl_cfg=build_skyrl_config_for_sft(cfg))
    try:
        trainer.setup()
        trainer.train()
        return trainer.export_final_model()
    except Exception as exc:
        if trainer.tracker is not None:
            trainer.tracker.log_exception(exc, step=trainer.global_step)
        else:
            logger.error(f"Fireworks SFT failed before tracker initialization:\n{exc}")
        raise
    finally:
        trainer.shutdown()


def main() -> None:
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
