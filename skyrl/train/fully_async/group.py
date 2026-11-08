"""The unit of work exchanged between generation workers and the training loop."""

from dataclasses import dataclass
from typing import Any, List, Optional

from skyrl.train.generators.base import GeneratorOutput


@dataclass
class GeneratedOutputGroup:
    """
    The GeneratorOutput for a single group of rollouts, along with the metadata.

    Attributes:
        generator_output (GeneratorOutput): The GeneratorOutput for a single group of rollouts.
            That is, the output to the same prompt, but `n_samples_per_prompt` of it.

        uid (str): The uid of the group. Underlyingly, it is the index of the data in train_dataloader.dataset.

        global_step_when_scheduled (int): The global step when the group was scheduled for generation,
            used for validating the staleness control.

        group_completion_time_s (Optional[float]): Wall-clock time (seconds) for the whole group to
            finish generation, i.e. the time for the slowest trajectory in the group to complete.
            None if generation timing was not captured.

        prompts (Optional[List[Any]]): The generator input prompts (in OpenAI message format) for this
            group, one per trajectory (parallel to ``generator_output["response_ids"]``). Retained so
            the trajectory logger can render prompt + response, mirroring the synchronous trainer which
            logs ``generator_input["prompts"]``. None if not captured.
    """

    generator_output: GeneratorOutput
    uid: str
    global_step_when_scheduled: int
    group_completion_time_s: Optional[float] = None
    prompts: Optional[List[Any]] = None
