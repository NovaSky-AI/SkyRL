"""ThunderAgent-aware generator wrappers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from skyrl.train.generators.base import ConversationType, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator


class ThunderAgentSkyRLGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator that releases ThunderAgent programs on trajectory exit."""

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ):
        try:
            return await super().agent_loop(
                prompt,
                env_class,
                env_extras,
                max_tokens,
                max_input_length,
                sampling_params=sampling_params,
                trajectory_id=trajectory_id,
            )
        finally:
            if trajectory_id is None:
                return

            release_program = getattr(self.inference_engine_client, "release_program", None)
            if release_program is None:
                return

            program_id = trajectory_id.to_string()
            try:
                await release_program(program_id)
            except Exception:
                logger.exception("Failed to release ThunderAgent program {}", program_id)
