"""
uv run --isolated --extra dev pytest tests/train/test_evaluator_hooks.py
"""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.evaluator import StandaloneEvaluator
from skyrl.train.generators.base import BatchMetadata, GeneratorInterface, GeneratorOutput, TrajectoryID
from skyrl.train.trainer import RayPPOTrainer
from tests.train.util import example_dummy_config


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


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "prompt": [{"role": "user", "content": "question"}],
            "env_class": None,
            "env_extras": {"data_source": "test"},
            "uid": "uid-1",
        }

    def collate_fn(self, batch):
        return batch


class SimpleLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class PrepareCalled(RuntimeError):
    pass


def _load_skyrl_agent_trainer_cls():
    module_path = (
        Path(__file__).resolve().parents[2] / "skyrl-agent" / "skyrl_agent" / "integrations" / "skyrl_train" / "trainer.py"
    )
    spec = importlib.util.spec_from_file_location("skyrl_agent_skyrl_train_trainer", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SkyRLAgentPPOTrainer


@pytest.mark.asyncio
async def test_standalone_evaluator_uses_prepare_hook():
    cfg = example_dummy_config()
    cfg.generator.inference_engine.backend = "vllm"
    cfg.trainer.dump_eval_results = False

    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        }
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])
    generator = DummyGenerator(
        {
            "prompt_token_ids": [[101]],
            "response_ids": [[201]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "stop_reasons": ["stop"],
            "rollout_logprobs": None,
        }
    )
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"

    class HookedEvaluator(StandaloneEvaluator):
        def prepare_generator_input(self, prompts, training_phase, global_step):
            return (
                {
                    "prompts": [[{"role": "user", "content": "hooked"}]],
                    "env_classes": ["gsm8k"],
                    "env_extras": [{"data_source": "dataset/a"}],
                    "sampling_params": None,
                    "trajectory_ids": None,
                    "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
                },
                ["uid-1"],
            )

    metrics = await HookedEvaluator(cfg=cfg, generator=generator, tokenizer=tokenizer).evaluate(
        eval_dataloader=eval_dataloader,
        global_step=7,
    )

    assert generator.seen_inputs[0]["prompts"] == [[{"role": "user", "content": "hooked"}]]
    assert metrics["eval/all/avg_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_standalone_evaluator_uses_eval_metadata_hook():
    cfg = example_dummy_config()
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.eval_n_samples_per_prompt = 2
    cfg.trainer.dump_eval_results = False

    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        }
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])
    generator = DummyGenerator(
        {
            "prompt_token_ids": [[101], [102]],
            "response_ids": [[201], [202]],
            "rewards": [1.0, 0.0],
            "loss_masks": [[1], [1]],
            "stop_reasons": ["stop", "stop"],
            "rollout_logprobs": None,
        }
    )
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"

    class HookedEvaluator(StandaloneEvaluator):
        def prepare_generator_input(self, prompts, training_phase, global_step):
            return (
                {
                    "prompts": [prompts[0]["prompt"], prompts[0]["prompt"]],
                    "env_classes": ["gsm8k", "gsm8k"],
                    "env_extras": [{"data_source": "dataset/a"}],
                    "sampling_params": None,
                    "trajectory_ids": [
                        TrajectoryID(instance_id="uid-1", repetition_id=0),
                        TrajectoryID(instance_id="uid-1", repetition_id=1),
                    ],
                    "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
                },
                ["uid-1", "uid-1"],
            )

        def get_eval_metadata(self, generator_input, uids, generator_output):
            return (
                ["gsm8k", "gsm8k"],
                [{"data_source": "dataset/a"}, {"data_source": "dataset/a"}],
                ["uid-1", "uid-1"],
            )

    metrics = await HookedEvaluator(cfg=cfg, generator=generator, tokenizer=tokenizer).evaluate(
        eval_dataloader=eval_dataloader,
        global_step=7,
    )

    assert metrics["eval/dataset_a/avg_score"] == pytest.approx(0.5)
    assert metrics["eval/all/pass_at_2"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_trainer_train_uses_prepare_generator_input_hook():
    cfg = example_dummy_config()
    cfg.trainer.eval_interval = 0

    class HookedTrainer(RayPPOTrainer):
        def __init__(self, *args, **kwargs):
            self.prepare_calls = []
            super().__init__(*args, **kwargs)

        def _build_train_dataloader_and_compute_training_steps(self):
            self.train_dataloader = SimpleLoader(
                [
                    [
                        {
                            "prompt": [{"role": "user", "content": "question"}],
                            "env_class": None,
                            "env_extras": {},
                            "uid": "uid-1",
                        }
                    ]
                ]
            )
            self.total_training_steps = 1

        def _remove_tail_data(self, prompts):
            return prompts

        def prepare_generator_input(self, prompts, training_phase, global_step):
            self.prepare_calls.append((prompts, training_phase, global_step))
            raise PrepareCalled

    trainer = HookedTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=MagicMock(),
        generator=MagicMock(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.init_weight_sync_state = MagicMock()
    trainer.dispatch.save_weights_for_sampler = AsyncMock()

    with pytest.raises(PrepareCalled):
        await trainer.train()

    assert trainer.prepare_calls
    assert trainer.prepare_calls[0][1] == "train"


@pytest.mark.asyncio
async def test_trainer_generate_uses_validation_hook():
    cfg = example_dummy_config()
    input_batch = {
        "prompts": [[{"role": "user", "content": "question"}]],
        "env_classes": ["gsm8k"],
        "env_extras": [{}],
        "sampling_params": None,
        "trajectory_ids": None,
        "batch_metadata": BatchMetadata(global_step=1, training_phase="train"),
    }
    generator_output = {
        "prompt_token_ids": [[101]],
        "response_ids": [[201]],
        "rewards": [1.0],
        "loss_masks": [[1]],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        "rollout_metrics": {"test_metric": 1.0},
    }

    class HookedTrainer(RayPPOTrainer):
        def _build_train_dataloader_and_compute_training_steps(self):
            self.train_dataloader = None
            self.total_training_steps = None

        def validate_generator_output(self, input_batch, generator_output):
            self.validated = (input_batch, generator_output)

    trainer = HookedTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=MagicMock(),
        generator=DummyGenerator(generator_output),
    )

    result = await trainer.generate(input_batch)

    assert result == generator_output
    assert trainer.validated[0] == input_batch
    assert trainer.validated[1] == generator_output


def test_skyrl_agent_eval_metadata_expands_prompt_level_extras():
    SkyRLAgentPPOTrainer = _load_skyrl_agent_trainer_cls()

    trainer = object.__new__(SkyRLAgentPPOTrainer)
    trainer.cfg = example_dummy_config()

    generator_input = {
        "prompts": [
            [{"role": "user", "content": "question-1"}],
            [{"role": "user", "content": "question-2"}],
        ],
        "env_classes": ["env-a", "env-a", "env-b", "env-b"],
        "env_extras": [{"data_source": "dataset/a"}, {"data_source": "dataset/b"}],
        "sampling_params": None,
        "trajectory_ids": [
            TrajectoryID(instance_id="uid-1", repetition_id=0),
            TrajectoryID(instance_id="uid-1", repetition_id=1),
            TrajectoryID(instance_id="uid-2", repetition_id=0),
            TrajectoryID(instance_id="uid-2", repetition_id=1),
        ],
        "batch_metadata": BatchMetadata(global_step=1, training_phase="eval"),
    }
    generator_output = {
        "prompt_token_ids": [[101], [102], [103], [104]],
        "response_ids": [[201], [202], [203], [204]],
        "rewards": [1.0, 0.0, 0.5, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop", "stop"],
        "rollout_logprobs": None,
        "trajectory_ids": None,
    }
    uids = ["uid-1", "uid-1", "uid-2", "uid-2"]

    env_classes, env_extras, expanded_uids = trainer.get_eval_metadata(generator_input, uids, generator_output)

    assert env_classes == ["env-a", "env-a", "env-b", "env-b"]
    assert env_extras == [
        {"data_source": "dataset/a"},
        {"data_source": "dataset/a"},
        {"data_source": "dataset/b"},
        {"data_source": "dataset/b"},
    ]
    assert expanded_uids == uids


@pytest.mark.asyncio
async def test_eval_only_entrypoint_uses_get_evaluator_hook():
    cfg = example_dummy_config()
    cfg.generator.inference_engine.enable_http_endpoint = True
    cfg.trainer.placement.colocate_all = False

    class DummyInferenceClient:
        async def wake_up(self):
            return None

    class DummyTracker:
        def __init__(self):
            self.logged = None

        def log(self, payload, step, commit):
            self.logged = (payload, step, commit)

    class DummyEvaluator:
        def __init__(self):
            self.called = False

        async def evaluate(self, eval_dataloader, global_step):
            self.called = True
            return {"eval/all/avg_score": 1.0}

    class HookedEvalOnlyEntrypoint(EvalOnlyEntrypoint):
        def get_eval_dataset(self):
            return DummyDataset()

        def get_inference_client(self):
            return DummyInferenceClient()

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            return object()

        def get_evaluator(self, cfg, tokenizer, generator):
            self.evaluator = DummyEvaluator()
            return self.evaluator

        def get_tracker(self):
            self.tracker_instance = DummyTracker()
            return self.tracker_instance

    with patch("skyrl.train.entrypoints.main_base.get_tokenizer", return_value=MagicMock()):
        exp = HookedEvalOnlyEntrypoint(cfg)

    result = await exp.run()

    assert result == {"eval/all/avg_score": 1.0}
    assert exp.evaluator.called is True
    assert exp.tracker_instance.logged == (result, 0, True)
