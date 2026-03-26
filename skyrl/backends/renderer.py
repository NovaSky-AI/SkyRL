from skyrl.tinker.types import ModelInput, RenderedModelInput


def render_model_input(model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
    return [
        RenderedModelInput(
            prompt_ids=[tok for chunk in mi.chunks for tok in (chunk.tokens if hasattr(chunk, "tokens") else [])]
        )
        for mi in model_inputs
    ]
