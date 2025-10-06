"""
uv run --isolated --extra dev pytest tests/cpu/test_eval.py
"""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from skyrl_train.evaluate import evaluate
from skyrl_train.generators.base import GeneratorInterface, GeneratorOutput
from tests.cpu.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class DummyGenerator(GeneratorInterface):
    def __init__(self, output: GeneratorOutput):
        self.output = output
        self.seen_inputs = []

    async def generate(self, input_batch):
        self.seen_inputs.append(input_batch)
        return self.output


@pytest.mark.asyncio
async def test_evaluate_computes_expected_metrics(dummy_config, tmp_path):
    cfg = dummy_config
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)

    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question-1"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        },
        {
            "prompt": [{"role": "user", "content": "question-2"}],
            "env_class": "custom_env",
            "env_extras": {"data_source": "dataset/b"},
            "uid": "uid-2",
        },
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    generator = DummyGenerator(generator_output)

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"

    metrics = await evaluate(
        eval_dataloader=eval_dataloader,
        generator=generator,
        cfg=cfg,
        global_step=5,
        tokenizer=tokenizer,
    )

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    assert len(generator.seen_inputs) == 1
    seen_batch = generator.seen_inputs[0]
    assert seen_batch["prompts"] == [prompt["prompt"] for prompt in prompts_batch]
    assert seen_batch["env_classes"] == ["gsm8k", "custom_env"]
    assert seen_batch["env_extras"] == [prompt["env_extras"] for prompt in prompts_batch]
    assert seen_batch["batch_metadata"].training_phase == "eval"
