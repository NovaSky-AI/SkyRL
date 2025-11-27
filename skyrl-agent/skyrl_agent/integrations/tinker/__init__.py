from .tinker_backend import TinkerBackend, TinkerGeneratorOutput, TinkerGeneratorInput
from ..base import register_backend, BackendSpec

register_backend(
    "tinker",
    BackendSpec(
        infer_backend_cls=TinkerBackend,
        generator_output_cls=TinkerGeneratorOutput,
        generator_input_cls=TinkerGeneratorInput,
    ),
)
