"""
uv run --isolated --extra dev pytest tests/train/test_eval.py
"""

import json
from unittest.mock import MagicMock

import pytest

from skyrl.train.config import EnvironmentConfig, SamplingParams
from skyrl.train.evaluate import evaluate
from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from tests.train.util import example_dummy_config


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
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.eval_sampling_params = SamplingParams(
        max_generate_length=20,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        logprobs=None,
        stop=None,
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = EnvironmentConfig(env_class="gsm8k")
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


# ---------------------------------------------------------------------------
# Step-wise vs plain generation through the same `evaluate()`
# ---------------------------------------------------------------------------


def _configure_eval(cfg, tmp_path, *, step_wise: bool, dump_results: bool = False):
    """Same setup as ``test_evaluate_computes_expected_metrics``, parameterized on generation mode."""
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.eval_sampling_params = SamplingParams(
        max_generate_length=20,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        logprobs=None,
        stop=None,
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.step_wise_trajectories = step_wise
    cfg.environment = EnvironmentConfig(env_class="gsm8k")
    cfg.trainer.dump_eval_results = dump_results
    cfg.trainer.export_path = str(tmp_path)
    return cfg


_PROMPTS_BATCH = [
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


def _plain_output() -> GeneratorOutput:
    """One row per prompt: uid-1 scores 1.0, uid-2 scores 0.0."""
    return {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }


def _step_wise_output() -> GeneratorOutput:
    """One row per step: uid-1 takes two steps (0.0, then 1.0 on the last), uid-2 takes one (0.0).

    Scoring every step would give dataset_a an avg_score of 0.5; scoring last steps only gives
    1.0, so the expectations in the tests below discriminate between the two.
    """
    return {
        "prompt_token_ids": [[101], [101, 201], [102]],
        "response_ids": [[201], [202], [203]],
        "rewards": [0.0, 1.0, 0.0],
        "loss_masks": [[1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop"],
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID("uid-1", 0), TrajectoryID("uid-1", 0), TrajectoryID("uid-2", 0)],
        "is_last_step": [False, True, True],
    }


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"
    return tokenizer


@pytest.mark.asyncio
async def test_evaluate_step_wise_scores_last_step_only(dummy_config, tmp_path):
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=True)
    generator = DummyGenerator(_step_wise_output())

    metrics = await evaluate(
        eval_dataloader=DummyStatefulDataLoader([_PROMPTS_BATCH]),
        generator=generator,
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
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
    assert generator.seen_inputs[0]["batch_metadata"].training_phase == "eval"


@pytest.mark.asyncio
@pytest.mark.parametrize("step_wise", [False, True])
async def test_evaluate_reports_turn_counts_to_trajectory_logger(dummy_config, tmp_path, step_wise):
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=step_wise)
    output = _step_wise_output() if step_wise else _plain_output()
    trajectory_logger = MagicMock()

    await evaluate(
        eval_dataloader=DummyStatefulDataLoader([_PROMPTS_BATCH]),
        generator=DummyGenerator(output),
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
        trajectory_logger=trajectory_logger,
    )

    trajectory_logger.log.assert_called_once()
    kwargs = trajectory_logger.log.call_args.kwargs
    # One logged sample per trajectory in both modes. Only step-wise overrides the logger's
    # turn count, with the number of steps each trajectory took.
    assert len(kwargs["prompts"]) == 2
    assert kwargs["generator_output"]["rewards"] == [1.0, 0.0]
    assert kwargs["num_turns_list"] == ([2, 1] if step_wise else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("step_wise", [False, True])
async def test_evaluate_surfaces_rollout_metrics(dummy_config, tmp_path, step_wise):
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=step_wise)
    output = _step_wise_output() if step_wise else _plain_output()

    metrics = await evaluate(
        eval_dataloader=DummyStatefulDataLoader([_PROMPTS_BATCH]),
        generator=DummyGenerator(output),
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
    )

    assert "eval/all/generate/avg_num_tokens" in metrics


@pytest.mark.asyncio
async def test_evaluate_step_wise_dump_keeps_every_step(dummy_config, tmp_path):
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=True, dump_results=True)

    await evaluate(
        eval_dataloader=DummyStatefulDataLoader([_PROMPTS_BATCH]),
        generator=DummyGenerator(_step_wise_output()),
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
    )

    dump_dir = tmp_path / "dumped_evals" / "global_step_5_evals"
    rows_a = [json.loads(line) for line in (dump_dir / "dataset_a.jsonl").read_text().splitlines()]
    rows_b = [json.loads(line) for line in (dump_dir / "dataset_b.jsonl").read_text().splitlines()]
    # The dump is the debugging artifact: it keeps every step, not just the scored last steps.
    assert [row["score"] for row in rows_a] == [0.0, 1.0]
    assert [row["score"] for row in rows_b] == [0.0]
    assert (dump_dir / "aggregated_results.jsonl").exists()
