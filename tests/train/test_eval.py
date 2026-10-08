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
    def __init__(self, output: GeneratorOutput | list[GeneratorOutput]):
        # A list is returned one element per `generate()` call, in order (clamped to the last);
        # a single output is returned on every call.
        self.outputs = list(output) if isinstance(output, list) else [output]
        self.seen_inputs = []

    async def generate(self, input_batch):
        self.seen_inputs.append(input_batch)
        return self.outputs[min(len(self.seen_inputs), len(self.outputs)) - 1]


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
async def test_evaluate_step_wise_counts_turns_per_repetition(dummy_config, tmp_path):
    """With ``eval_n_samples_per_prompt > 1`` the repetitions of one prompt share a uid (which is
    what pass@n groups by) but are distinct trajectories: each last-step row must report its own
    repetition's step count, not the sum over the prompt."""
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=True)
    cfg.generator.eval_n_samples_per_prompt = 2
    # One prompt, two repetitions: rep 0 takes three steps (reward on its last), rep 1 takes two.
    output: GeneratorOutput = {
        "prompt_token_ids": [[101], [101, 201], [101, 201, 202], [101], [101, 203]],
        "response_ids": [[201], [202], [204], [203], [205]],
        "rewards": [0.0, 0.0, 1.0, 0.0, 0.0],
        "loss_masks": [[1]] * 5,
        "stop_reasons": ["stop"] * 5,
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID("uid-1", 0)] * 3 + [TrajectoryID("uid-1", 1)] * 2,
        "is_last_step": [False, False, True, False, True],
    }
    trajectory_logger = MagicMock()

    metrics = await evaluate(
        eval_dataloader=DummyStatefulDataLoader([[_PROMPTS_BATCH[0]]]),
        generator=DummyGenerator(output),
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
        trajectory_logger=trajectory_logger,
    )

    kwargs = trajectory_logger.log.call_args.kwargs
    assert len(kwargs["prompts"]) == 2
    assert kwargs["num_turns_list"] == [3, 2]
    # pass@n still groups the two repetitions under their shared prompt id.
    assert metrics["eval/dataset_a/pass_at_2"] == pytest.approx(1.0)
    assert metrics["eval/dataset_a/avg_score"] == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# Two batches: `_EvalRows.extend()` and cross-batch `concatenate_generator_outputs`
# ---------------------------------------------------------------------------


def _prompts(*rows):
    """One prompt per (uid, data_source) pair, with the default env class."""
    return [
        {
            "prompt": [{"role": "user", "content": f"question-{uid}"}],
            "env_class": None,
            "env_extras": {"data_source": data_source},
            "uid": uid,
        }
        for uid, data_source in rows
    ]


def _plain_output_for(rewards) -> GeneratorOutput:
    n = len(rewards)
    return {
        "prompt_token_ids": [[100 + i] for i in range(n)],
        "response_ids": [[200 + i] for i in range(n)],
        "rewards": list(rewards),
        "loss_masks": [[1]] * n,
        "stop_reasons": ["stop"] * n,
        "rollout_logprobs": None,
    }


def _step_wise_output_for(trajectories) -> GeneratorOutput:
    """``trajectories``: (uid, per-step rewards) in order; each trajectory is a contiguous block."""
    out = _plain_output_for([r for _, rewards in trajectories for r in rewards])
    out["trajectory_ids"] = [TrajectoryID(uid, 0) for uid, rewards in trajectories for _ in rewards]
    out["is_last_step"] = [i == len(rewards) - 1 for _, rewards in trajectories for i in range(len(rewards))]
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("step_wise", [False, True])
async def test_evaluate_two_batches_keeps_rows_aligned_across_batches(dummy_config, tmp_path, step_wise):
    """Two dataloader batches: exercises ``_EvalRows.extend()`` and cross-batch
    ``concatenate_generator_outputs``. Datasets alternate within each batch, so the per-dataset
    metrics only come out right if batch 2's rows are attributed to batch 2's prompts."""
    cfg = _configure_eval(dummy_config, tmp_path, step_wise=step_wise, dump_results=True)
    batch_1 = _prompts(("uid-1", "dataset/a"), ("uid-2", "dataset/b"))
    batch_2 = _prompts(("uid-3", "dataset/a"), ("uid-4", "dataset/b"))
    if step_wise:
        # uid-1: two steps, reward on the last; uid-2: one step. uid-3: one step; uid-4: two steps.
        outputs = [
            _step_wise_output_for([("uid-1", [0.0, 1.0]), ("uid-2", [0.0])]),
            _step_wise_output_for([("uid-3", [0.0]), ("uid-4", [0.0, 1.0])]),
        ]
    else:
        outputs = [_plain_output_for([1.0, 0.0]), _plain_output_for([0.0, 1.0])]
    generator = DummyGenerator(outputs)
    trajectory_logger = MagicMock()

    metrics = await evaluate(
        eval_dataloader=DummyStatefulDataLoader([batch_1, batch_2]),
        generator=generator,
        cfg=cfg,
        global_step=5,
        tokenizer=_tokenizer(),
        trajectory_logger=trajectory_logger,
    )

    # One generate() per batch, each with that batch's prompts.
    assert [len(inp["prompts"]) for inp in generator.seen_inputs] == [2, 2]
    assert generator.seen_inputs[1]["prompts"] == [p["prompt"] for p in batch_2]

    # Per-dataset metrics span both batches: a = {uid-1: 1.0, uid-3: 0.0}, b = {uid-2: 0.0, uid-4: 1.0}.
    for key in (
        "eval/dataset_a/avg_score",
        "eval/dataset_b/avg_score",
        "eval/all/avg_score",
        "eval/all/pass_at_1",
    ):
        assert metrics[key] == pytest.approx(0.5), key
    assert "eval/all/generate/avg_num_tokens" in metrics  # rollout metrics re-aggregated across batches

    # One logged sample per trajectory across both batches, in row order.
    kwargs = trajectory_logger.log.call_args.kwargs
    assert len(kwargs["prompts"]) == 4
    assert kwargs["generator_output"]["rewards"] == [1.0, 0.0, 0.0, 1.0]
    assert kwargs["num_turns_list"] == ([2, 1, 1, 2] if step_wise else None)

    # The dump attributes every row (every step when step-wise) to its own batch's dataset.
    dump_dir = tmp_path / "dumped_evals" / "global_step_5_evals"

    def _scores(name):
        return [json.loads(line)["score"] for line in (dump_dir / f"{name}.jsonl").read_text().splitlines()]

    if step_wise:
        assert _scores("dataset_a") == [0.0, 1.0, 0.0]  # uid-1 (2 steps) then uid-3 (1 step)
        assert _scores("dataset_b") == [0.0, 0.0, 1.0]  # uid-2 (1 step) then uid-4 (2 steps)
    else:
        assert _scores("dataset_a") == [1.0, 0.0]
        assert _scores("dataset_b") == [0.0, 1.0]
