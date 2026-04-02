from typing import TYPE_CHECKING

from .base import GeneratorInput, GeneratorInterface, GeneratorOutput

if TYPE_CHECKING:
    from .output_builders import (
        RetokenizedTrajectory,
        TokenizedTrajectory,
        build_generator_output_from_messages,
        build_generator_output_from_tokenized,
    )
    from .skyrl_gym_generator import SkyRLGymGenerator

__all__ = [
    "GeneratorInterface",
    "GeneratorInput",
    "GeneratorOutput",
    "RetokenizedTrajectory",
    "TokenizedTrajectory",
    "build_generator_output_from_messages",
    "build_generator_output_from_tokenized",
    "SkyRLGymGenerator",
]


def __getattr__(name: str):
    if name == "SkyRLGymGenerator":
        from .skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator

    if name in {
        "RetokenizedTrajectory",
        "TokenizedTrajectory",
        "build_generator_output_from_messages",
        "build_generator_output_from_tokenized",
    }:
        from .output_builders import (
            RetokenizedTrajectory,
            TokenizedTrajectory,
            build_generator_output_from_messages,
            build_generator_output_from_tokenized,
        )

        return {
            "RetokenizedTrajectory": RetokenizedTrajectory,
            "TokenizedTrajectory": TokenizedTrajectory,
            "build_generator_output_from_messages": build_generator_output_from_messages,
            "build_generator_output_from_tokenized": build_generator_output_from_tokenized,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
