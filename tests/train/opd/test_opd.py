"""CPU tests for on-policy distillation (skyrl/train/opd).

Run with:
uv run --isolated --extra dev pytest tests/train/opd/test_opd.py
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.opd.config import (
    OPDExpConfig,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    FireworksTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
    _extract_echoed_logprobs,
)
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.opd.utils import (
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    pad_teacher_logprobs,
    split_generator_input,
)
from skyrl.train.trainer import RayPPOTrainer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTeacher(TeacherLogprobClient):
    """Deterministic teacher: logprob of token id t is -t/100. Optional jitter and in-flight tracking."""

    def __init__(self, max_concurrency: int = 32, jitter: float = 0.0, delay: float = 0.0):
        super().__init__(max_concurrency=max_concurrency)
        self.jitter = jitter
        self.delay = delay
        self.calls: List[tuple] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False
        self._n = 0

    async def aclose(self) -> None:
        self.closed = True

    async def _compute_logprobs(self, prompt_ids, response_ids):
        self.calls.append((tuple(prompt_ids), tuple(response_ids)))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            self._n += 1
            offset = self.jitter * (self._n % 2)  # alternate answers when jitter > 0
            return [-t / 100.0 + offset for t in response_ids]
        finally:
            self.in_flight -= 1


class WrongLengthTeacher(TeacherLogprobClient):
    async def _compute_logprobs(self, prompt_ids, response_ids):
        return [0.0] * (len(response_ids) + 1)


# ---------------------------------------------------------------------------
# opd_utils
# ---------------------------------------------------------------------------


def test_pad_teacher_logprobs_right_aligns_and_pads_rows():
    response_mask = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 1]])  # last row is a padding row
    out = pad_teacher_logprobs([[-1.0, -2.0], [-3.0, -4.0, -5.0]], response_mask, pad_size=1)
    assert out.shape == (3, 3)
    assert out[0].tolist() == [0.0, -1.0, -2.0]
    assert out[1].tolist() == [-3.0, -4.0, -5.0]
    assert out[2].tolist() == out[0].tolist()  # padding row copies row 0


def test_pad_teacher_logprobs_rejects_misaligned_rows():
    response_mask = torch.tensor([[0, 1, 1]])
    with pytest.raises(ValueError, match="teacher logprobs"):
        pad_teacher_logprobs([[-1.0]], response_mask, pad_size=0)
    with pytest.raises(ValueError, match="rows"):
        pad_teacher_logprobs([[-1.0, -2.0]], response_mask, pad_size=1)


def test_apply_opd_to_advantages_pure_and_mixed():
    student = torch.tensor([[-1.0, -2.0, -3.0]])
    teacher = torch.tensor([[-1.5, -1.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    zero_adv = torch.zeros(1, 3)

    adv, metrics = apply_opd_to_advantages(zero_adv, student, teacher, mask, kl_coef=2.0)
    # -coef * (student - teacher) * mask
    assert adv.tolist() == [[-1.0, 2.0, 0.0]]
    assert metrics["opd/reverse_kl"] == pytest.approx((0.5 + -1.0) / 2)
    assert metrics["opd/reverse_kl_abs_max"] == pytest.approx(1.0)
    assert metrics["opd/kl_coef"] == 2.0

    rl_adv = torch.tensor([[0.7, 0.7, 0.7]])
    mixed, _ = apply_opd_to_advantages(rl_adv, student, teacher, mask, kl_coef=1.0)
    assert torch.allclose(mixed, torch.tensor([[0.7 - 0.5, 0.7 + 1.0, 0.7]]))


def test_apply_opd_to_advantages_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_opd_to_advantages(torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 3), torch.ones(1, 3), 1.0)


# ---------------------------------------------------------------------------
# teacher_client base class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_client_empty_response_short_circuits():
    teacher = FakeTeacher()
    assert await teacher.compute_logprobs([1, 2], []) == []
    assert teacher.calls == []


@pytest.mark.asyncio
async def test_base_client_length_invariant():
    with pytest.raises(RuntimeError, match="returned 3 logprobs for 2"):
        await WrongLengthTeacher().compute_logprobs([1], [5, 6])


@pytest.mark.asyncio
async def test_base_client_limits_concurrency():
    teacher = FakeTeacher(max_concurrency=2, delay=0.05)
    await asyncio.gather(*(teacher.compute_logprobs([1], [i]) for i in range(6)))
    assert teacher.max_in_flight == 2


@pytest.mark.asyncio
async def test_self_test_accepts_deterministic_and_rejects_noisy():
    assert await FakeTeacher().self_test([1], [2, 3], n=4, max_abs_diff=0.01) == 0.0
    with pytest.raises(RuntimeError, match="not reproducible"):
        await FakeTeacher(jitter=0.3).self_test([1], [2, 3], n=4, max_abs_diff=0.05)


# ---------------------------------------------------------------------------
# FireworksTeacherClient
# ---------------------------------------------------------------------------


def _content_response(ids: List[int], logprobs: List[float]) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "token_ids": ids,
                "logprobs": {"content": [{"token_id": t, "logprob": lp} for t, lp in zip(ids, logprobs)]},
            }
        ]
    }


def _legacy_response(ids: List[int], logprobs: List[float]) -> Dict[str, Any]:
    return {"choices": [{"token_ids": ids, "logprobs": {"token_ids": ids, "token_logprobs": logprobs}}]}


def test_extract_echoed_logprobs_both_shapes():
    assert _extract_echoed_logprobs(_content_response([7, 8], [-0.1, -0.2])["choices"][0]) == ([7, 8], [-0.1, -0.2])
    assert _extract_echoed_logprobs(_legacy_response([7, 8], [-0.1, -0.2])["choices"][0]) == ([7, 8], [-0.1, -0.2])
    with pytest.raises(RuntimeError, match="no logprobs"):
        _extract_echoed_logprobs({"token_ids": [1]})


def test_fireworks_client_rejects_v1_base_url_and_missing_key():
    with pytest.raises(ValueError, match="server root"):
        FireworksTeacherClient("m", api_key="k", base_url="https://api.fireworks.ai/v1")
    with pytest.raises(ValueError, match="API key"):
        FireworksTeacherClient("m", api_key="")


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["content", "legacy"])
async def test_fireworks_request_and_parse(shape):
    client = FireworksTeacherClient("accounts/fireworks/models/x", api_key="k")
    captured: Dict[str, Any] = {}

    async def fake_post(body):
        captured.update(body)
        make = _content_response if shape == "content" else _legacy_response
        return make([30, 31, 32], [-0.5, -0.25, -0.125])

    client._post = fake_post  # type: ignore[method-assign]
    out = await client.compute_logprobs([10, 11], [30, 31, 32])

    assert out == [-0.5, -0.25, -0.125]
    assert captured["prompt"] == [10, 11, 30, 31, 32]
    assert captured["max_tokens"] == 0
    assert captured["echo_last"] == 3
    assert captured["logprobs"] is True
    assert captured["return_token_ids"] is True
    assert captured["temperature"] == 1.0


@pytest.mark.asyncio
async def test_fireworks_id_mismatch_raises():
    client = FireworksTeacherClient("accounts/fireworks/models/x", api_key="k")

    async def fake_post(body):
        return _content_response([99, 31], [-0.5, -0.25])

    client._post = fake_post  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="different token ids"):
        await client.compute_logprobs([10], [30, 31])


# ---------------------------------------------------------------------------
# VLLMTeacherClient
# ---------------------------------------------------------------------------


def _vllm_response(prompt_len: int, ids: List[int], logprobs: List[float]) -> Dict[str, Any]:
    """A vLLM completions choice with prompt_logprobs: None at position 0, then {id: {"logprob"}} per
    token (JSON keys are strings). Each entry also carries an alternative, as a k>0 server would,
    to show the lookup by sent id ignores extra entries."""
    entries: List[Any] = [None] + [{"7": {"logprob": -9.0, "rank": 1}} for _ in range(prompt_len - 1)]
    entries += [
        {"7": {"logprob": -0.01, "rank": 1}, str(tid): {"logprob": lp, "rank": 2}} for tid, lp in zip(ids, logprobs)
    ]
    return {"choices": [{"text": "x", "logprobs": None, "prompt_logprobs": entries}]}


def test_vllm_client_rejects_missing_model_and_urls():
    with pytest.raises(ValueError, match="model name"):
        VLLMTeacherClient("", server_urls=["http://a:8000"])
    with pytest.raises(ValueError, match="server url"):
        VLLMTeacherClient("m", server_urls=[])


@pytest.mark.asyncio
async def test_vllm_request_parse_and_round_robin():
    client = VLLMTeacherClient("teacher", server_urls=["http://a:8000/", "http://b:8000"])
    seen: Dict[str, Any] = {}

    async def fake_post(body):
        seen.update(body)
        return _vllm_response(prompt_len=2, ids=[30, 31, 32], logprobs=[-0.5, -0.25, -0.125])

    client._post = fake_post  # type: ignore[method-assign]
    out = await client.compute_logprobs([10, 11], [30, 31, 32])

    assert out == [-0.5, -0.25, -0.125]
    assert seen["prompt"] == [10, 11, 30, 31, 32]
    assert seen["model"] == "teacher" and seen["max_tokens"] == 1 and seen["prompt_logprobs"] == 0
    assert [client._next_url() for _ in range(3)] == [
        "http://a:8000/v1/completions",
        "http://b:8000/v1/completions",
        "http://a:8000/v1/completions",
    ]


@pytest.mark.asyncio
async def test_vllm_wrong_token_id_and_short_response_raise():
    client = VLLMTeacherClient("teacher", server_urls=["http://a:8000"])

    async def scored_a_different_token(body):
        return _vllm_response(prompt_len=1, ids=[99], logprobs=[-1.0])  # we sent 30

    client._post = scored_a_different_token  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="tokenizer"):
        await client.compute_logprobs([10], [30])

    async def too_short(body):
        return {"choices": [{"prompt_logprobs": [None]}]}

    client._post = too_short  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="prompt logprobs"):
        await client.compute_logprobs([10], [30])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_opd_config_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    cfg = OPDExpConfig.from_cli_overrides(
        [
            "trainer.teacher.model=accounts/fireworks/models/gpt-oss-120b",
            "trainer.algorithm.opd.kl_coef=0.5",
            "trainer.algorithm.opd.use_task_reward=true",
        ]
    )
    assert cfg.trainer.algorithm.use_kl_loss is False
    assert cfg.trainer.algorithm.policy_loss_type == "importance_sampling"
    assert cfg.trainer.teacher.backend == "fireworks"
    assert cfg.trainer.algorithm.opd.kl_coef == 0.5
    assert cfg.trainer.algorithm.opd.use_task_reward is True
    validate_opd_cfg(cfg)  # no error


@pytest.mark.parametrize(
    "overrides, message",
    [
        ([], "trainer.teacher.model must be set"),
        (["trainer.teacher.model=m", "trainer.teacher.backend=bogus"], "backend must be one of"),
        (["trainer.teacher.model=m", "trainer.teacher.backend=vllm"], "server_urls"),
        (["trainer.teacher.model=m", "trainer.algorithm.zero_variance_filter=true"], "zero_variance_filter"),
        (["trainer.teacher.model=m", "trainer.algorithm.advantage_batch_normalize=true"], "advantage_batch_normalize"),
        (["trainer.teacher.model=m", "trainer.algorithm.policy_loss_type=rollout_is"], "old-logprob forward"),
        (["trainer.teacher.model=m", "trainer.algorithm.opd.kl_coef=-1"], "kl_coef"),
        (["trainer.teacher.model=m", "generator.step_wise_trajectories=true"], "step_wise"),
    ],
)
def test_validate_opd_cfg_rejects(monkeypatch, overrides, message):
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    cfg = OPDExpConfig.from_cli_overrides(overrides)
    with pytest.raises(ValueError, match=message):
        validate_opd_cfg(cfg)


def test_validate_opd_cfg_requires_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m"])
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        validate_opd_cfg(cfg)


# ---------------------------------------------------------------------------
# OPDTrainer: per-group generate + scoring, then the three consumers of teacher_logprobs
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.decode.side_effect = lambda ids, **kwargs: "decoded"
    tok.eos_token_id = 4
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    return tok


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def _batch(num_prompts: int, n: int, phase: str) -> GeneratorInput:
    """The layout prepare_generator_input produces: prompt-major, n rows per prompt."""
    rows = [(p, r) for p in range(num_prompts) for r in range(n)]
    return {
        "prompts": [[{"role": "user", "content": f"q{p}"}] for p, _ in rows],
        "env_classes": ["gsm8k"] * len(rows),
        "env_extras": [{"answer": str(p)} for p, _ in rows],
        "sampling_params": {"temperature": 1.0},
        "trajectory_ids": [TrajectoryID(instance_id=str(p), repetition_id=r) for p, r in rows],
        "batch_metadata": BatchMetadata(global_step=1, training_phase=phase),
    }


class FakeGenerator(GeneratorInterface):
    """Response ids encode (prompt, repetition); records every call; can hold a prompt's group open."""

    def __init__(self, hold: Optional[Dict[int, asyncio.Event]] = None):
        self.calls: List[GeneratorInput] = []
        self.hold = hold or {}  # prompt index -> event its group waits for before returning

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        self.calls.append(input_batch)
        prompts = [int(prompt[0]["content"][1:]) for prompt in input_batch["prompts"]]
        repetitions = [tid.repetition_id for tid in input_batch["trajectory_ids"]]
        for p in sorted(set(prompts)):
            if p in self.hold:
                await asyncio.wait_for(self.hold[p].wait(), timeout=5)
        return {
            "prompt_token_ids": [[1, p] for p in prompts],
            "response_ids": [[10 + p, 20 + r, 4] for p, r in zip(prompts, repetitions)],
            "rewards": [float(r) for r in repetitions],
            "loss_masks": [[1, 1, 1] for _ in prompts],
            "stop_reasons": ["stop"] * len(prompts),
            "rollout_metrics": {"generate/custom_avg": float(len(prompts))},
            "rollout_logprobs": None,
            "trajectory_ids": input_batch["trajectory_ids"],
        }


class TqdmAwareFakeGenerator(FakeGenerator):
    """Takes disable_tqdm, like SkyRLGymGenerator.generate."""

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        self.disable_tqdm = disable_tqdm
        return await super().generate(input_batch)


def _trainer(
    use_task_reward: bool,
    kl_coef: float = 1.0,
    tokenizer=None,
    generator: Optional[GeneratorInterface] = None,
    teacher: Optional[TeacherLogprobClient] = None,
) -> OPDTrainer:
    cfg = OPDExpConfig()
    cfg.trainer.train_batch_size = 1  # prompts per batch; the two rows below share one uid
    cfg.trainer.eval_batch_size = 1
    cfg.trainer.policy_mini_batch_size = 1
    cfg.trainer.resume_mode = "none"
    cfg.trainer.epochs = 1
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.algorithm.opd.use_task_reward = use_task_reward
    cfg.trainer.algorithm.opd.kl_coef = kl_coef
    trainer = OPDTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=generator or FakeGenerator(),
        teacher_client=teacher or FakeTeacher(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.get_lcm_dp_size.return_value = 1
    return trainer


def test_split_generator_input_slices_rows_and_shares_batch_fields():
    batch = _batch(2, 2, "train")
    groups = split_generator_input(batch, 2)
    assert len(groups) == 2
    assert groups[1]["prompts"] == batch["prompts"][2:]
    assert groups[1]["env_extras"] == batch["env_extras"][2:]
    assert groups[1]["trajectory_ids"] == batch["trajectory_ids"][2:]
    assert groups[1]["sampling_params"] is batch["sampling_params"]
    assert groups[1]["batch_metadata"] is batch["batch_metadata"]

    batch["env_extras"] = None
    batch["trajectory_ids"] = None
    assert split_generator_input(batch, 4)[0]["env_extras"] is None
    with pytest.raises(ValueError, match="groups of 3"):
        split_generator_input(batch, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("generator_cls", [FakeGenerator, TqdmAwareFakeGenerator])
async def test_trainer_generate_scores_each_group_as_it_finishes(generator_cls, tokenizer):
    """Group 0 must be scored while group 1 is still generating; rows come back in input order."""
    release_group1 = asyncio.Event()
    generator = generator_cls(hold={1: release_group1})
    teacher = FakeTeacher()
    trainer = _trainer(use_task_reward=True, tokenizer=tokenizer, generator=generator, teacher=teacher)

    async def release_group1_once_group0_is_scored():
        # Group 1 cannot return before this fires, so the first two teacher calls must be group 0's.
        # A trainer that scored only after all generation finished would deadlock here (5 s timeout).
        while len(teacher.calls) < 2:
            await asyncio.sleep(0.001)
        release_group1.set()

    releaser = asyncio.create_task(release_group1_once_group0_is_scored())
    out = await trainer.generate(_batch(2, 2, "train"))
    await releaser

    assert [len(call["prompts"]) for call in generator.calls] == [2, 2]
    if generator_cls is TqdmAwareFakeGenerator:
        assert generator.disable_tqdm is True
    assert out["response_ids"] == [[10, 20, 4], [10, 21, 4], [11, 20, 4], [11, 21, 4]]
    assert out[TEACHER_LOGPROBS_KEY] == [[-t / 100.0 for t in row] for row in out["response_ids"]]
    assert [tid.to_string() for tid in out["trajectory_ids"]] == ["0_0", "0_1", "1_0", "1_1"]
    assert "rollout_metrics" not in out
    assert trainer.all_metrics["generate/custom_avg"] == 2.0  # per-group metrics re-aggregated
    assert trainer.all_metrics["opd/teacher_time_exposed"] >= 0.0
    assert trainer.all_metrics["opd/teacher_time_per_group_mean"] >= 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["returns", "raises"])
async def test_trainer_train_closes_teacher_client(outcome, tokenizer):
    """The teacher's sessions are closed when the training loop ends, and when it fails."""
    teacher = FakeTeacher()
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, teacher=teacher)
    base_train = AsyncMock(side_effect=RuntimeError("boom") if outcome == "raises" else None)

    with patch.object(RayPPOTrainer, "train", base_train):
        if outcome == "raises":
            with pytest.raises(RuntimeError, match="boom"):
                await trainer.train()
        else:
            await trainer.train()

    base_train.assert_awaited_once()
    assert teacher.closed


@pytest.mark.asyncio
async def test_trainer_generate_leaves_eval_alone(tokenizer):
    generator = FakeGenerator()
    teacher = FakeTeacher()
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=generator, teacher=teacher)

    out = await trainer.generate(_batch(2, 2, "eval"))

    assert len(generator.calls) == 1 and len(generator.calls[0]["prompts"]) == 4
    assert teacher.calls == []
    assert TEACHER_LOGPROBS_KEY not in out


def _generator_output() -> Dict[str, Any]:
    return {
        "prompt_token_ids": [[1, 2], [1, 2]],
        "response_ids": [[3, 4, 5], [6, 7]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1, 1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
        TEACHER_LOGPROBS_KEY: [[-0.3, -0.4, -0.5], [-0.6, -0.7]],
    }


def test_trainer_pure_mode_zeroes_rewards_after_metrics():
    trainer = _trainer(use_task_reward=False)
    out, uids = trainer.postprocess_generator_output(_generator_output(), ["u", "u"])
    assert out["rewards"] == [[0.0, 0.0, 0.0], [0.0, 0.0]]
    assert trainer.all_metrics["reward/avg_raw_reward"] == pytest.approx(0.5)  # logged from the real rewards


def test_trainer_mixed_mode_keeps_rewards():
    trainer = _trainer(use_task_reward=True)
    out, _ = trainer.postprocess_generator_output(_generator_output(), ["u", "u"])
    assert out["rewards"] == [[0.0, 0.0, 1.0], [0.0, 0.0]]


def test_trainer_end_to_end_advantages_pure(tokenizer):
    """Pure OPD through the real conversion + estimator: advantages == -kl_coef * (student - teacher) * mask."""
    trainer = _trainer(use_task_reward=False, kl_coef=2.0, tokenizer=tokenizer)
    out, uids = trainer.postprocess_generator_output(_generator_output(), ["u", "u"])
    batch = trainer.convert_to_training_input(out, uids)

    assert torch.allclose(batch[TEACHER_LOGPROBS_KEY], batch_teacher(batch))  # right-aligned

    student = torch.tensor([[-1.0, -1.0, -1.0], [0.0, -2.0, -2.0]])
    batch["action_log_probs"] = student
    batch["values"] = None
    batch = trainer.compute_advantages_and_returns(batch)

    expected = -2.0 * (student - batch_teacher(batch)) * batch["loss_mask"]
    assert torch.allclose(batch["advantages"], expected)
    assert TEACHER_LOGPROBS_KEY not in batch  # popped, never shipped to workers
    assert "opd/reverse_kl" in trainer.all_metrics


def batch_teacher(batch) -> torch.Tensor:
    return torch.tensor([[-0.3, -0.4, -0.5], [0.0, -0.6, -0.7]])


def test_trainer_missing_teacher_field_is_loud():
    trainer = _trainer(use_task_reward=False)
    out = _generator_output()
    out.pop(TEACHER_LOGPROBS_KEY)
    with pytest.raises(RuntimeError, match="teacher_logprobs"):
        trainer.convert_to_training_input(out, ["u", "u"])
