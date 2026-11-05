"""CPU tests for on-policy distillation (skyrl/train/opd).

Run with:
uv run --isolated --extra dev pytest tests/train/opd/test_opd.py
"""

import asyncio
import base64
import copy
import io
import math
import random
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from loguru import logger

from skyrl.backends.skyrl_train.utils.ppo_utils import (
    apply_loss_reduction_to_advantages_minibatch,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SampleSupportColumn,
)
from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.opd.config import (
    OPDConfig,
    OPDExpConfig,
    TeacherEngineConfig,
    teacher_logprob_width,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    SkyRLTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
    parse_scoring_response,
    scoring_request,
)
from skyrl.train.opd.teacher_launch import _teacher_start_port
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.opd.utils import (
    ROLLOUT_CORRECT_KEY,
    STUDENT_LOGPROB_FLOOR,
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    candidate_matrix,
    pad_teacher_logprobs,
    split_generator_input,
)
from skyrl.train.trainer import RayPPOTrainer
from tests.backends.skyrl_train.sample_support_helpers import csr, dense

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _encode(array) -> str:
    """A base64 ``.npy`` float32 array, as vLLM sends ``prompt_token_id_logprobs``."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _id_scores(ids) -> np.ndarray:
    """The fake teacher's logprob of token id t, -t/100; ``-1`` padding scores -inf, as in vLLM."""
    ids = np.asarray(ids)
    return np.where(ids >= 0, -ids / 100.0, -np.inf).astype(np.float32)


def _answer(request: Dict[str, Any]) -> Dict[str, Any]:
    """vLLM's answer to a scoring request, from ``_id_scores``."""
    return {"prompt_token_id_logprobs": _encode(_id_scores(request["sampling_params"]["prompt_logprob_token_ids"]))}


def _width(request: Dict[str, Any]) -> int:
    return np.asarray(request["sampling_params"]["prompt_logprob_token_ids"]).shape[1]


class FakeTeacher(TeacherLogprobClient):
    """Answers like vLLM with ``_id_scores``. Records each request; optional delay and in-flight tracking."""

    def __init__(self, max_concurrency: int = 32, delay: float = 0.0):
        super().__init__(max_concurrency=max_concurrency)
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def _post_generate(self, request):
        self.calls.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return _answer(request)
        finally:
            self.in_flight -= 1


class CannedTeacher(TeacherLogprobClient):
    """Answers every request with ``response``, or raises it."""

    def __init__(self, response: Union[Dict[str, Any], Exception]):
        super().__init__()
        self.response = response

    async def _post_generate(self, request):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


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

    adv, metrics = apply_opd_to_advantages(zero_adv, student, teacher, mask, OPDConfig(kl_coef=2.0))
    # -coef * (student - teacher) * mask
    assert adv.tolist() == [[-1.0, 2.0, 0.0]]
    assert metrics["opd/reverse_kl"] == pytest.approx((0.5 + -1.0) / 2)
    assert metrics["opd/reverse_kl_abs_max"] == pytest.approx(1.0)
    assert metrics["opd/kl_coef"] == 2.0

    rl_adv = torch.tensor([[0.7, 0.7, 0.7]])
    mixed, _ = apply_opd_to_advantages(rl_adv, student, teacher, mask, OPDConfig(kl_coef=1.0))
    assert torch.allclose(mixed, torch.tensor([[0.7 - 0.5, 0.7 + 1.0, 0.7]]))


def test_apply_opd_to_advantages_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_opd_to_advantages(torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 3), torch.ones(1, 3), OPDConfig())


def _logprobs(*probabilities: float) -> torch.Tensor:
    return torch.tensor([probabilities]).log()


def test_apply_opd_to_advantages_without_stabilization_is_unchanged():
    # Bit for bit what the term was before the stabilization settings existed.
    generator = torch.Generator().manual_seed(0)
    student, teacher = -torch.rand(4, 7, generator=generator) * 5, -torch.rand(4, 7, generator=generator) * 5
    advantages = torch.randn(4, 7, generator=generator)
    mask = (torch.rand(4, 7, generator=generator) > 0.3).float()

    new, metrics = apply_opd_to_advantages(advantages, student, teacher, mask, OPDConfig(kl_coef=0.7))

    reverse_kl = torch.where(mask.bool(), student - teacher, torch.zeros_like(student))
    assert torch.equal(new, advantages - 0.7 * reverse_kl)
    assert metrics["opd/divergence_clip_fraction"] == 0.0
    assert metrics["opd/advantage_clip_fraction"] == 0.0


@pytest.mark.parametrize(
    "opd",
    [
        OPDConfig(),
        OPDConfig(divergence_clip=1.0),
        OPDConfig(advantage_clip_max=5.0),
        OPDConfig(advantage_log_compress=True),
    ],
)
def test_an_infinite_student_logprob_stays_finite(opd):
    # Unfloored, -inf gives an inf term, and exp(-inf) * -inf = NaN slips past the divergence clip.
    student = torch.tensor([[float("-inf"), -1.0]])
    teacher = torch.tensor([[-2.0, -2.0]])

    new, metrics = apply_opd_to_advantages(torch.zeros(1, 2), student, teacher, torch.ones(1, 2), opd)

    assert torch.isfinite(new).all()
    assert all(np.isfinite(value) for value in metrics.values())


def test_an_infinite_student_logprob_is_floored():
    student = torch.tensor([[float("-inf")]])
    new, _ = apply_opd_to_advantages(torch.zeros(1, 1), student, torch.tensor([[-2.0]]), torch.ones(1, 1), OPDConfig())
    assert new.item() == -2.0 - STUDENT_LOGPROB_FLOOR


def test_divergence_clip_drops_only_confident_disagreements():
    # p log(p / q) per token: 0.9 log 9 = 1.98 (clipped at tau=1), 0.5 log 1.25 = 0.11 (kept),
    # 1e-4 log 1e4 = 9e-4 (a large log-ratio on an unlikely token, kept), and 0.1 log(1/9) < 0 (never clipped).
    student, teacher = _logprobs(0.9, 0.5, 1e-4, 0.1), _logprobs(0.1, 0.4, 1e-8, 0.9)
    mask = torch.ones(1, 4)
    plain, _ = apply_opd_to_advantages(torch.zeros(1, 4), student, teacher, mask, OPDConfig())

    clipped, metrics = apply_opd_to_advantages(
        torch.zeros(1, 4), student, teacher, mask, OPDConfig(divergence_clip=1.0)
    )

    assert clipped[0, 0] == 0.0
    assert torch.equal(clipped[0, 1:], plain[0, 1:])
    assert metrics["opd/divergence_clip_fraction"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ({"advantage_clip_min": -1.0}, [-1.0, -0.5, 0.5, 3.0]),
        ({"advantage_clip_max": 1.0}, [-3.0, -0.5, 0.5, 1.0]),
        ({"advantage_clip_min": -1.0, "advantage_clip_max": 1.0}, [-1.0, -0.5, 0.5, 1.0]),
    ],
)
def test_advantage_clip_bounds_the_log_ratio(bounds, expected):
    log_ratio = torch.tensor([[-3.0, -0.5, 0.5, 3.0]])  # log q - log p per token
    student = torch.full_like(log_ratio, -1.0)

    new, metrics = apply_opd_to_advantages(
        torch.zeros(1, 4), student, student + log_ratio, torch.ones(1, 4), OPDConfig(**bounds)
    )

    assert new[0].tolist() == pytest.approx(expected)
    assert metrics["opd/advantage_clip_fraction"] == pytest.approx(
        sum(a != e for a, e in zip([-3, -0.5, 0.5, 3], expected)) / 4
    )


def test_log_compression_keeps_kl_coef_outside():
    log_ratio = torch.tensor([[-3.0, 0.0, 0.5, 3.0]])
    student = torch.full_like(log_ratio, -1.0)

    compressed, metrics = apply_opd_to_advantages(
        torch.zeros(1, 4),
        student,
        student + log_ratio,
        torch.ones(1, 4),
        OPDConfig(kl_coef=2.0, advantage_log_compress=True),
    )

    assert compressed[0].tolist() == pytest.approx(
        [-2.0 * np.log1p(3.0), 0.0, 2.0 * np.log1p(0.5), 2.0 * np.log1p(3.0)]
    )
    assert metrics["opd/advantage_clip_fraction"] == 0.0


def test_stabilization_leaves_the_task_advantage_and_masked_tokens_alone():
    student, teacher = _logprobs(0.9, 0.9), _logprobs(1e-3, 1e-3)  # log q - log p = -6.8 on both tokens
    rl = torch.tensor([[5.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0]])

    new, metrics = apply_opd_to_advantages(
        rl, student, teacher, mask, OPDConfig(advantage_clip_min=-0.5, divergence_clip=10.0)
    )

    assert new.tolist() == [[4.5, 5.0]]  # the task advantage is not clipped; the masked token gets no term
    assert metrics["opd/advantage_clip_fraction"] == 1.0  # over the one masked-in token


def test_log_ratio_percentiles_cover_masked_tokens_only():
    log_ratio = torch.arange(100.0).reshape(1, 100) / 10
    student = torch.zeros(1, 100)
    mask = torch.ones(1, 100)
    mask[0, :50] = 0

    _, metrics = apply_opd_to_advantages(torch.zeros(1, 100), student, student + log_ratio, mask, OPDConfig())
    low, high = np.quantile(log_ratio[0, 50:].numpy(), [0.01, 0.99])
    assert metrics["opd/log_ratio_p01"] == pytest.approx(low)
    assert metrics["opd/log_ratio_p99"] == pytest.approx(high)

    _, empty = apply_opd_to_advantages(
        torch.zeros(1, 100), student, student + log_ratio, torch.zeros(1, 100), OPDConfig()
    )
    assert "opd/log_ratio_p01" not in empty and "opd/log_ratio_p99" not in empty


# ---------------------------------------------------------------------------
# teacher_client base class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_client_empty_response_short_circuits():
    teacher = FakeTeacher()
    scores = await teacher.compute_logprobs([1, 2], np.empty((0, 3), dtype=np.int32))
    assert scores.logprobs.shape == (0, 3)
    assert teacher.calls == []


@pytest.mark.asyncio
async def test_base_client_scores_the_candidate_matrix():
    teacher = FakeTeacher()
    candidates = np.array([[5, 7, -1], [6, 8, 9]], dtype=np.int32)

    scores = await teacher.compute_logprobs([1, 2], candidates)

    np.testing.assert_array_equal(scores.logprobs, _id_scores(candidates))
    assert teacher.calls == [scoring_request([1, 2], candidates)]


@pytest.mark.asyncio
async def test_base_client_rejects_a_non_matrix():
    with pytest.raises(ValueError, match=r"\[tokens, width\]"):
        await FakeTeacher().compute_logprobs([1], np.array([5, 6], dtype=np.int32))


@pytest.mark.asyncio
async def test_base_client_limits_concurrency():
    teacher = FakeTeacher(max_concurrency=2, delay=0.05)
    await asyncio.gather(*(teacher.compute_logprobs([1], np.array([[i]], dtype=np.int32)) for i in range(6)))
    assert teacher.max_in_flight == 2


def test_scoring_request_windows_the_prefill_to_the_response():
    candidates = np.array([[30, 7], [31, -1], [32, 9]], dtype=np.int32)

    request = scoring_request([10, 11], candidates)

    assert request["token_ids"] == [10, 11, 30, 31, 32]
    params = request["sampling_params"]
    # Row i scores sequence token start + i + 1, so rows 0..2 are the three response tokens.
    assert params["prompt_logprob_start"] == 1
    assert params["prompt_logprob_token_ids"] == candidates.tolist()
    assert params["max_tokens"] == 1 and params["temperature"] == 1.0 and params["detokenize"] is False
    assert "prompt_logprobs" not in params


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "no prompt_token_id_logprobs"),
        ({"prompt_token_id_logprobs": _encode(np.zeros((2, 1)))}, r"expected \(2, 2\) scores, got \(2, 1\)"),
        ({"prompt_token_id_logprobs": _encode([[-0.5, -1.0], [np.nan, -1.0]])}, "non-finite"),
        ({"prompt_token_id_logprobs": _encode([[-0.5, np.inf], [-0.5, -1.0]])}, "non-finite"),
        ({"prompt_token_id_logprobs": _encode([[-0.5, -np.inf], [-0.5, -1.0]])}, "non-finite"),
    ],
)
def test_parse_scoring_response_rejects_malformed_answers(response, message):
    candidates = np.array([[5, 6], [7, 8]], dtype=np.int32)
    with pytest.raises(RuntimeError, match=message):
        parse_scoring_response(response, candidates)


def test_parse_scoring_response_accepts_minus_inf_on_padding():
    candidates = np.array([[5, -1]], dtype=np.int32)

    scores = parse_scoring_response({"prompt_token_id_logprobs": _encode([[-0.5, -np.inf]])}, candidates)

    assert scores.logprobs.dtype == np.float32
    assert scores.logprobs[0, 0] == -0.5 and np.isneginf(scores.logprobs[0, 1])


@pytest.mark.asyncio
async def test_preflight_scores_one_token_at_the_width():
    teacher = FakeTeacher()
    await teacher.preflight(17)
    (request,) = teacher.calls
    assert request["token_ids"] == [0, 0] and _width(request) == 17


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [{}, RuntimeError("teacher HTTP 400: prompt_logprob_token_ids requires the V2 model runner")],
)
async def test_preflight_names_what_the_server_needs(response):
    with pytest.raises(RuntimeError, match=r"teacher preflight failed: .*V2 model runner and --max-logprobs >= 17"):
        await CannedTeacher(response).preflight(17)


# ---------------------------------------------------------------------------
# VLLMTeacherClient
# ---------------------------------------------------------------------------


class FakeResponse:
    request_info = SimpleNamespace(real_url="http://router:9000/inference/v1/generate")
    history = ()

    def __init__(self, status, payload=None):
        self.status, self.payload = status, payload

    async def text(self):
        return "busy" if self.status >= 500 else "bad request"

    async def json(self):
        return self.payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    closed = False

    def __init__(self, responses: List[FakeResponse]):
        self.urls: List[str] = []
        self.responses = responses

    def post(self, url, json):
        self.urls.append(url)
        return self.responses.pop(0)


def _with_session(client: VLLMTeacherClient, session: FakeSession) -> None:
    async def fake_get_session():
        return session

    client._get_session = fake_get_session  # type: ignore[method-assign]


def test_vllm_client_rejects_missing_model_and_bad_url():
    with pytest.raises(ValueError, match="model name"):
        VLLMTeacherClient("", server_url="http://a:8000")
    with pytest.raises(ValueError, match="server url"):
        VLLMTeacherClient("m", server_url="")
    with pytest.raises(ValueError, match="trailing /v1"):
        VLLMTeacherClient("m", server_url="http://a:8000/v1/")


@pytest.mark.asyncio
async def test_vllm_client_sends_the_scoring_request_with_its_model():
    client = VLLMTeacherClient("teacher", server_url="http://a:8000/")
    assert client.server_url == "http://a:8000"
    seen: Dict[str, Any] = {}

    async def fake_post(body):
        seen.update(body)
        return _answer(body)

    client._post = fake_post  # type: ignore[method-assign]
    candidates = np.array([[30, 7], [31, -1]], dtype=np.int32)
    scores = await client.compute_logprobs([10, 11], candidates)

    np.testing.assert_array_equal(scores.logprobs, _id_scores(candidates))
    assert seen == {**scoring_request([10, 11], candidates), "model": "teacher"}


@pytest.mark.asyncio
async def test_vllm_client_retries_on_the_same_endpoint():
    # One endpoint per teacher: a retry backs off and goes to the same URL; balancing replicas is the
    # endpoint's (router's) job.
    client = VLLMTeacherClient("teacher", server_url="http://router:9000", max_retries=1)
    session = FakeSession([FakeResponse(503), FakeResponse(200, {"prompt_token_id_logprobs": _encode([[-0.5]])})])
    _with_session(client, session)

    scores = await client.compute_logprobs([10], np.array([[30]], dtype=np.int32))

    assert scores.logprobs.tolist() == [[-0.5]]
    assert session.urls == ["http://router:9000/inference/v1/generate"] * 2


@pytest.mark.asyncio
async def test_vllm_client_does_not_retry_a_client_error():
    # A 4xx is the server refusing the request, e.g. an id outside the teacher's vocabulary or a width
    # above --max-logprobs; retrying cannot help.
    client = VLLMTeacherClient("teacher", server_url="http://router:9000", max_retries=3)
    session = FakeSession([FakeResponse(400)])
    _with_session(client, session)

    with pytest.raises(RuntimeError, match="teacher HTTP 400 .*bad request"):
        await client.compute_logprobs([10], np.array([[30]], dtype=np.int32))
    assert len(session.urls) == 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_opd_config_defaults_and_overrides():
    cfg = OPDExpConfig.from_cli_overrides(
        [
            "trainer.teacher.model=Qwen/Qwen3-32B",
            "trainer.algorithm.opd.kl_coef=0.5",
            "trainer.algorithm.opd.use_task_reward=true",
        ]
    )
    assert cfg.trainer.algorithm.use_kl_loss is False
    assert cfg.trainer.algorithm.policy_loss_type == "importance_sampling"
    assert cfg.trainer.teacher.backend == "skyrl"  # the job launches the teacher by default
    assert cfg.trainer.algorithm.opd.kl_coef == 0.5
    assert cfg.trainer.algorithm.opd.use_task_reward is True
    validate_opd_cfg(cfg)  # no error


def test_teacher_engine_block_keeps_its_defaults_under_overrides():
    # from_cli_overrides rebuilds a nested block from its field type whenever any key inside it is
    # overridden, so the teacher's changed defaults have to live on TeacherEngineConfig itself.
    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m", "trainer.teacher.inference_engine.num_engines=2"])
    ie_cfg = cfg.trainer.teacher.inference_engine
    assert isinstance(ie_cfg, TeacherEngineConfig)
    assert ie_cfg.num_engines == 2
    assert ie_cfg.enable_prefix_caching is False
    assert ie_cfg.gpu_memory_utilization == 0.9
    assert ie_cfg.enable_ray_prometheus_stats is False
    assert ie_cfg != TeacherEngineConfig()
    assert OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m"]).trainer.teacher.inference_engine == (
        TeacherEngineConfig()
    )
    validate_opd_cfg(cfg)


def test_validate_opd_cfg_accepts_each_backend_with_its_own_fields():
    validate_opd_cfg(
        OPDExpConfig.from_cli_overrides(
            [
                "trainer.teacher.model=Qwen/Qwen3-32B",
                "trainer.teacher.inference_engine.tensor_parallel_size=4",
                "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=4096",
            ]
        )
    )
    validate_opd_cfg(
        OPDExpConfig.from_cli_overrides(
            ["trainer.teacher.model=m", "trainer.teacher.backend=vllm", "trainer.teacher.server_url=http://a:8000"]
        )
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ([], "trainer.teacher.model must be set"),
        (["trainer.teacher.model=m", "trainer.teacher.backend=bogus"], "backend must be one of"),
        (["trainer.teacher.model=m", "trainer.teacher.backend=fireworks"], "backend must be one of"),  # removed
        (["trainer.teacher.model=m", "trainer.teacher.backend=vllm"], "requires trainer.teacher.server_url"),
        (
            ["trainer.teacher.model=m", "trainer.teacher.backend=vllm", "trainer.teacher.server_url=http://a:8000/v1"],
            "trailing /v1",
        ),
        # The two backends are exclusive: the other backend's fields are a config error, not a no-op.
        (
            ["trainer.teacher.model=m", "trainer.teacher.server_url=http://a:8000"],
            "server_url is for backend='vllm'",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.run_engines_locally=false"],
            "run_engines_locally must be true",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.external_server_urls=['http://a:8000']"],
            "external_server_urls does not apply",
        ),
        (["trainer.teacher.model=m", "trainer.teacher.inference_engine.enable_pd=true"], "enable_pd does not apply"),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.enable_return_routed_experts=true"],
            "enable_return_routed_experts does not apply",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.enable_return_sample_support_set=true"],
            "enable_return_sample_support_set does not apply",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.speculative_config={method: mtp}"],
            "speculative_config does not apply",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.candidate_top_k=4"],
            "candidate_top_k does not apply",
        ),
        (
            ["trainer.teacher.model=m", "trainer.teacher.inference_engine.fp8_weight_sync_exclude_modules=['*.mlp.*']"],
            "fp8_weight_sync_exclude_modules does not apply",
        ),
        (["trainer.teacher.model=m", "trainer.teacher.inference_engine.num_engines=0"], "at least one GPU"),
        (
            [
                "trainer.teacher.model=m",
                "trainer.teacher.backend=vllm",
                "trainer.teacher.server_url=http://a:8000",
                "trainer.teacher.inference_engine.num_engines=2",
            ],
            "inference_engine is for backend='skyrl'",
        ),
        (["trainer.teacher.model=m", "trainer.algorithm.zero_variance_filter=true"], "zero_variance_filter"),
        (["trainer.teacher.model=m", "trainer.algorithm.advantage_batch_normalize=true"], "advantage_batch_normalize"),
        (["trainer.teacher.model=m", "trainer.algorithm.policy_loss_type=rollout_is"], "old-logprob forward"),
        (
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.policy_loss_type=cispo",
                "trainer.algorithm.cispo.cispo_anchor=rollout",
            ],
            "cispo_anchor='old'",
        ),
        (["trainer.teacher.model=m", "trainer.algorithm.advantage_estimator=gae"], "needs a task reward"),
        (["trainer.teacher.model=m", "trainer.algorithm.opd.kl_coef=-1"], "kl_coef"),
        (["trainer.teacher.model=m", "trainer.algorithm.opd.divergence_clip=0"], "divergence_clip must be positive"),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.advantage_clip_min=0.5"],
            "advantage_clip_min must be <= 0",
        ),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.advantage_clip_max=-0.5"],
            "advantage_clip_max must be >= 0",
        ),
        (
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.opd.advantage_clip_max=1.0",
                "trainer.algorithm.opd.advantage_log_compress=true",
            ],
            "are alternatives",
        ),
        (["trainer.teacher.model=m", "generator.step_wise_trajectories=true"], "step_wise"),
        (["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=1"], "0 or at least 2"),
        (["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=-4"], "0 or at least 2"),
        (
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.opd.top_k=4",
                "trainer.algorithm.enable_sample_support_replay=true",
                "generator.inference_engine.enable_return_sample_support_set=true",
                "generator.sampling_params.top_k=8",
            ],
            "sample-support replay renormalizes it",
        ),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=4", "generator.sampling_params.top_k=50"],
            "top_k=-1",
        ),
        (
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.opd.top_k=4",
                "generator.sampling_params.repetition_penalty=1.1",
            ],
            "repetition_penalty=1",
        ),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=4", "generator.vision_language_generator=true"],
            "vision_language_generator",
        ),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=4", "generator.sampling_params.min_p=0.05"],
            "min_p=0",
        ),
        (
            ["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=4", "trainer.algorithm.policy_loss_type=regular"],
            "needs policy_loss_type='importance_sampling'",
        ),
    ],
)
def test_validate_opd_cfg_rejects(overrides, message):
    cfg = OPDExpConfig.from_cli_overrides(overrides)
    with pytest.raises(ValueError, match=message):
        validate_opd_cfg(cfg)


@pytest.mark.parametrize(
    "overrides, warns",
    [
        (["trainer.teacher.model=m"], True),
        (["trainer.teacher.model=m", "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=4096"], False),
        (
            ["trainer.teacher.model=m", "trainer.teacher.backend=vllm", "trainer.teacher.server_url=http://a:8000"],
            False,
        ),
    ],
)
def test_validate_opd_cfg_warns_when_launched_teacher_max_model_len_is_unset(overrides, warns):
    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        validate_opd_cfg(OPDExpConfig.from_cli_overrides(overrides))
    finally:
        logger.remove(handler_id)
    assert any("max_model_len is not set" in message for message in messages) == warns


def test_validate_opd_cfg_accepts_zero_advantage_bounds():
    validate_opd_cfg(
        OPDExpConfig.from_cli_overrides(
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.opd.advantage_clip_min=0",
                "trainer.algorithm.opd.advantage_clip_max=0",
                "trainer.algorithm.opd.divergence_clip=0.5",
            ]
        )
    )
    # The divergence clip combines with either advantage regulation.
    validate_opd_cfg(
        OPDExpConfig.from_cli_overrides(
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.opd.divergence_clip=0.5",
                "trainer.algorithm.opd.advantage_log_compress=true",
            ]
        )
    )


def test_validate_opd_cfg_turns_on_candidate_capture_for_top_k():
    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=16"])
    validate_opd_cfg(cfg)
    ie_cfg = cfg.generator.inference_engine
    assert ie_cfg.enable_return_sample_support_set is True and ie_cfg.candidate_top_k == 16
    assert ie_cfg.enable_return_sample_support_logprobs is True
    assert teacher_logprob_width(cfg) == 17

    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m"])
    validate_opd_cfg(cfg)
    ie_cfg = cfg.generator.inference_engine
    assert ie_cfg.enable_return_sample_support_set is False and ie_cfg.candidate_top_k is None
    assert ie_cfg.enable_return_sample_support_logprobs is False
    assert teacher_logprob_width(cfg) == 1


def test_validate_opd_cfg_accepts_gae_in_mixed_mode_and_old_anchored_cispo():
    validate_opd_cfg(
        OPDExpConfig.from_cli_overrides(
            [
                "trainer.teacher.model=m",
                "trainer.algorithm.advantage_estimator=gae",
                "trainer.algorithm.opd.use_task_reward=true",
                "trainer.algorithm.policy_loss_type=cispo",
                "trainer.algorithm.cispo.cispo_anchor=old",
            ]
        )
    )


# ---------------------------------------------------------------------------
# Launching the teacher (backend="skyrl")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # the student's launched servers take num_engines * dp windows of 100 ports from 8000
        (["generator.inference_engine.num_engines=4", "generator.inference_engine.data_parallel_size=2"], 8800),
        (["generator.inference_engine.num_engines=1"], 8100),
        # a PD student indexes the same number of windows
        (
            [
                "generator.inference_engine.num_engines=3",
                "generator.inference_engine.enable_pd=true",
                "generator.inference_engine.num_prefill=1",
            ],
            8300,
        ),
        # an external student launches nothing
        (
            [
                "generator.inference_engine.num_engines=4",
                "generator.inference_engine.run_engines_locally=false",
                "generator.inference_engine.external_server_urls=['http://s:8000']",
            ],
            8000,
        ),
        (
            ["generator.inference_engine.num_engines=4", "generator.inference_engine.external_proxy_url=http://p:9000"],
            8000,
        ),
    ],
)
def test_teacher_start_port_is_past_the_students_windows(overrides, expected):
    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m"] + overrides)
    assert _teacher_start_port(cfg) == expected


@pytest.mark.parametrize(
    "engine_cap, top_k, expected",
    [(20, 0, 20), (20, 16, 20), (20, 32, 33), (64, 32, 64), (-1, 32, -1)],
)
def test_launched_teacher_raises_max_logprobs_to_the_scoring_width(monkeypatch, engine_cap, top_k, expected):
    from skyrl.train.opd.teacher_launch import launch_teacher

    # Parsed first: the config's own validation imports the real utils module.
    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m", f"trainer.algorithm.opd.top_k={top_k}"])
    built = {}
    # Stand-ins for the vLLM-side modules, which need vllm and vllm_router to import.
    setup = types.ModuleType("setup")
    utils = types.ModuleType("utils")

    def fake_impl(ie_cfg, *, make_cli_args, **kwargs):
        built["args"] = make_cli_args()
        return "client", "server-setup"

    setup.build_new_inference_client_impl = fake_impl
    utils.build_frozen_vllm_cli_args = lambda model, ie_cfg, seed: Namespace(max_logprobs=engine_cap)
    monkeypatch.setitem(sys.modules, "skyrl.backends.skyrl_train.inference_servers.setup", setup)
    monkeypatch.setitem(sys.modules, "skyrl.backends.skyrl_train.inference_servers.utils", utils)

    assert launch_teacher(cfg) == ("client", "server-setup")
    assert built["args"].max_logprobs == expected


class FakeRemoteInferenceClient:
    """The two RemoteInferenceClient methods SkyRLTeacherClient uses."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self.torn_down = False

    async def score_tokens(self, request):
        self.requests.append(request)
        return _answer(request)

    async def teardown(self):
        self.torn_down = True


@pytest.mark.asyncio
async def test_skyrl_teacher_client_scores_through_score_tokens():
    client = FakeRemoteInferenceClient()
    teacher = SkyRLTeacherClient(client, server_setup="server-setup", max_concurrency=4)
    assert teacher.client is client
    assert teacher.server_setup == "server-setup"
    candidates = np.array([[3, 9], [4, -1]], dtype=np.int32)

    scores = await teacher.compute_logprobs([1, 2], candidates)

    np.testing.assert_array_equal(scores.logprobs, _id_scores(candidates))
    # score_tokens names the model the servers were launched under.
    assert client.requests == [scoring_request([1, 2], candidates)]

    await teacher.aclose()
    assert client.torn_down


def test_opd_exp_launches_the_teacher_for_the_skyrl_backend(monkeypatch):
    import skyrl.train.entrypoints.main_opd as main_opd

    cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=Qwen/Qwen3-32B", "trainer.teacher.max_concurrency=8"])
    fake_client = FakeRemoteInferenceClient()
    launched = []

    def fake_launch(launch_cfg):
        launched.append(launch_cfg)
        return fake_client, "server-setup"

    monkeypatch.setattr(main_opd, "launch_teacher", fake_launch)
    exp = main_opd.OPDExp.__new__(main_opd.OPDExp)  # skip BasePPOExp.__init__ (tokenizer, datasets)
    exp.cfg = cfg

    teacher = exp.get_teacher_client()
    assert isinstance(teacher, SkyRLTeacherClient)
    assert teacher.client is fake_client
    assert launched == [cfg]
    assert teacher.server_setup == "server-setup"  # the client holds the deployment, keeping its actors alive


@pytest.mark.parametrize("fails", [False, True])
def test_opd_exp_preflights_the_teacher_before_any_model_loads(monkeypatch, fails):
    import skyrl.train.entrypoints.main_opd as main_opd

    events: List[Any] = []

    class ProbedTeacher(FakeTeacher):
        async def preflight(self, width):
            events.append(("preflight", width))
            if fails:
                raise RuntimeError("teacher preflight failed: no V2 runner")

        async def aclose(self):
            events.append("aclose")

    monkeypatch.setattr(main_opd.OPDExp, "get_teacher_client", lambda self: ProbedTeacher())
    monkeypatch.setattr(main_opd.BasePPOExp, "_setup_trainer", lambda self: events.append("setup") or "trainer")
    exp = main_opd.OPDExp.__new__(main_opd.OPDExp)
    exp.cfg = OPDExpConfig.from_cli_overrides(["trainer.teacher.model=m", "trainer.algorithm.opd.top_k=4"])

    if fails:
        with pytest.raises(RuntimeError, match="preflight failed"):
            exp._setup_trainer()
        assert events == [("preflight", 5), "aclose"]
    else:
        assert exp._setup_trainer() == "trainer"
        assert events == [("preflight", 5), "aclose", "setup"]


def test_opd_exp_keeps_url_backends_unlaunched(monkeypatch):
    import skyrl.train.entrypoints.main_opd as main_opd

    monkeypatch.setattr(main_opd, "launch_teacher", lambda cfg: pytest.fail("must not launch"))
    exp = main_opd.OPDExp.__new__(main_opd.OPDExp)
    exp.cfg = OPDExpConfig.from_cli_overrides(
        ["trainer.teacher.model=m", "trainer.teacher.backend=vllm", "trainer.teacher.server_url=http://a:8000"]
    )
    assert isinstance(exp.get_teacher_client(), VLLMTeacherClient)


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


class SupportFakeGenerator(FakeGenerator):
    """Also returns each row's captured top-2 with sampler logprobs, as SkyRLGymGenerator does under capture.

    The last token (EOS) has no captured members.
    """

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        output = await super().generate(input_batch)
        output[SAMPLE_SUPPORT_FIELD] = [
            csr([[50, 51], [52, -1], [-1, -1]], logprobs=[[-0.5, -1.0], [-0.25, 0.0], [0.0, 0.0]])
            for _ in output["response_ids"]
        ]
        return output


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
    top_k: int = 0,
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
    cfg.trainer.algorithm.opd.top_k = top_k
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


def test_candidate_matrix_puts_the_sampled_token_first():
    plain = candidate_matrix([30, 31])
    assert plain.dtype == np.int32
    np.testing.assert_array_equal(plain, [[30], [31]])
    support = csr([[5, 6], [-1, -1], [7, -1]])
    np.testing.assert_array_equal(candidate_matrix([30, 31, 32], support), [[30, 5, 6], [31, -1, -1], [32, 7, -1]])
    with pytest.raises(ValueError, match="2 candidate rows for 3 response tokens"):
        candidate_matrix([30, 31, 32], support[:2])


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
    for logprobs, response_ids in zip(out[TEACHER_LOGPROBS_KEY], out["response_ids"]):
        np.testing.assert_array_equal(logprobs, _id_scores(response_ids))
    assert {_width(request) for request in teacher.calls} == {1}  # top_k=0 scores the sampled token only
    assert [tid.to_string() for tid in out["trajectory_ids"]] == ["0_0", "0_1", "1_0", "1_1"]
    assert "rollout_metrics" not in out
    assert trainer.all_metrics["generate/custom_avg"] == 2.0  # per-group metrics re-aggregated
    assert trainer.all_metrics["opd/teacher_time_exposed"] >= 0.0
    assert trainer.all_metrics["opd/teacher_time_per_group_mean"] >= 0.0


@pytest.mark.asyncio
async def test_trainer_top_k_scores_the_captured_candidates(tokenizer):
    teacher = FakeTeacher()
    trainer = _trainer(
        use_task_reward=False, tokenizer=tokenizer, generator=SupportFakeGenerator(), teacher=teacher, top_k=2
    )

    out = await trainer.generate(_batch(1, 2, "train"))

    assert {_width(request) for request in teacher.calls} == {3}
    rows = out[SAMPLE_SUPPORT_FIELD][0]
    # [sampled, candidates...]; the EOS, with no captured members, stays empty.
    np.testing.assert_array_equal(dense(rows), [[10, 50, 51], [20, 52, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(rows.logprobs, [0.0, -0.5, -1.0, 0.0, -0.25])
    np.testing.assert_array_equal(rows.member_values, _id_scores(rows.token_ids))
    np.testing.assert_array_equal(out[TEACHER_LOGPROBS_KEY][0], _id_scores([10, 20, 4]))

    # The third column reaches the trainer batch.
    out, uids = trainer.postprocess_generator_output(out, ["0", "0"])
    batch = trainer.convert_to_training_input(out, uids)
    assert len(batch[SAMPLE_SUPPORT_FIELD].columns) == 3


@pytest.mark.asyncio
async def test_trainer_top_k_needs_a_generator_that_forwards_support(tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=FakeGenerator(), top_k=2)
    with pytest.raises(RuntimeError, match="opd.top_k > 0 needs 'rollout_sample_support'"):
        await trainer.generate(_batch(1, 2, "train"))


class ShortSupportFakeGenerator(SupportFakeGenerator):
    """Returns one support entry fewer than it returns responses."""

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        output = await super().generate(input_batch)
        output[SAMPLE_SUPPORT_FIELD] = output[SAMPLE_SUPPORT_FIELD][:-1]
        return output


@pytest.mark.asyncio
async def test_trainer_top_k_refuses_support_that_does_not_cover_every_trajectory(tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=ShortSupportFakeGenerator(), top_k=2)
    with pytest.raises(ValueError, match="zip"):
        await trainer.generate(_batch(1, 2, "train"))


@pytest.mark.asyncio
async def test_trainer_top_k_zero_leaves_a_generators_support_alone(tokenizer):
    # Support captured for another purpose, such as replay, keeps its rows and scores only the sampled token.
    teacher = FakeTeacher()
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=SupportFakeGenerator(), teacher=teacher)

    out = await trainer.generate(_batch(1, 2, "train"))

    assert {_width(request) for request in teacher.calls} == {1}
    assert out[SAMPLE_SUPPORT_FIELD][0].member_values is None
    np.testing.assert_array_equal(dense(out[SAMPLE_SUPPORT_FIELD][0]), [[50, 51], [52, -1], [-1, -1]])


# SupportFakeGenerator's rows under FakeTeacher (log q(t) = -t/100): every row is [sampled, 50, 51] then
# [sampled, 52]; the EOS has none. No sampled token is a candidate.
TOP_K_C = [0.0 + math.exp(-1.0) * (-0.51 + 1.0), math.exp(-0.25) * (-0.52 + 0.25), 0.0]
# The candidates' a(v) = log q - log p: 50 and 51 in the first rows, 52 in the second.
TOP_K_A = [-0.50 + 0.5, -0.51 + 1.0, -0.52 + 0.25]


async def _top_k_batch(trainer):
    out = await trainer.generate(_batch(1, 2, "train"))
    out, uids = trainer.postprocess_generator_output(out, ["0", "0"])
    batch = trainer.convert_to_training_input(copy.deepcopy(out), uids)
    batch["action_log_probs"] = torch.tensor([[-1.0, -2.0, -3.0], [-1.5, -2.5, -0.5]])
    batch["values"] = None
    return out, uids, batch


@pytest.mark.asyncio
@pytest.mark.parametrize("use_task_reward", [False, True])
async def test_trainer_top_k_advantages_and_logit_weights(use_task_reward, tokenizer):
    trainer = _trainer(
        use_task_reward=use_task_reward, kl_coef=2.0, tokenizer=tokenizer, generator=SupportFakeGenerator(), top_k=2
    )
    out, uids, batch = await _top_k_batch(trainer)
    student = batch["action_log_probs"].clone()
    # The estimator's own advantages: zero in pure mode, GRPO's in mixed mode.
    base = trainer.convert_to_training_input(copy.deepcopy(out), uids)
    base["action_log_probs"], base["values"] = student.clone(), None
    rl_advantages = RayPPOTrainer.compute_advantages_and_returns(trainer, base)["advantages"]

    batch = trainer.compute_advantages_and_returns(batch)

    teacher = torch.tensor([[-0.10, -0.20, -0.04], [-0.10, -0.21, -0.04]])
    # No sampled token is a candidate, so each keeps its tail term; the top-K term is the worker's.
    torch.testing.assert_close(batch["advantages"], rl_advantages + 2.0 * (teacher - student))
    if use_task_reward:
        assert rl_advantages.abs().sum() > 0
    # The member values are now the worker's weights: slot 0, the sampled token, gets 0, each candidate
    # kl_coef * a(v). The worker multiplies by the current pi(v); C (the metric) uses the sampler's p(v).
    w_50, w_51, w_52 = (2.0 * a for a in TOP_K_A)
    weights = [0.0, w_50, w_51, 0.0, w_52]
    np.testing.assert_allclose(
        batch[SAMPLE_SUPPORT_FIELD].columns[SampleSupportColumn.MEMBER_VALUES].numpy(), weights * 2, atol=1e-6
    )
    assert trainer.all_metrics["opd/sampled_in_topk_fraction"] == 0.0
    expected_mass = (math.exp(-0.5) + math.exp(-1.0) + math.exp(-0.25)) / 3
    assert trainer.all_metrics["opd/topk_mass_mean"] == pytest.approx(expected_mass, abs=1e-6)
    assert trainer.all_metrics["opd/topk_weight_abs_mean"] == pytest.approx(
        2.0 * (abs(TOP_K_C[0]) + abs(TOP_K_C[1])) / 3, abs=1e-6
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("loss_reduction", ["token_mean", "sequence_mean", "seq_mean_token_sum_norm", "prompt_mean"])
async def test_trainer_top_k_scales_the_logit_weights_like_the_advantages(loss_reduction, tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=SupportFakeGenerator(), top_k=2)
    trainer.cfg.trainer.algorithm.loss_reduction = loss_reduction
    trainer.cfg.trainer.algorithm.max_seq_len = 16
    _, _, batch = await _top_k_batch(trainer)
    batch["loss_mask"] = torch.tensor([[1, 1, 1], [0, 1, 1]])  # unequal token counts per row
    batch = trainer.compute_advantages_and_returns(batch)
    advantages = batch["advantages"].clone()
    weights = batch[SAMPLE_SUPPORT_FIELD].columns[SampleSupportColumn.MEMBER_VALUES].clone()

    batch = trainer._normalize_advantages(batch, [(0, 2)], prompt_boundaries=[(0, 2)])

    def reduce(values):
        return apply_loss_reduction_to_advantages_minibatch(
            values,
            batch["loss_mask"],
            loss_reduction,
            micro_batch_size=trainer.cfg.trainer.micro_train_batch_size_per_gpu,
            max_seq_len=16,
            prompt_boundaries=[(0, 2)],
        )

    # The advantages are the base normalization's, untouched.
    assert torch.equal(batch["advantages"], reduce(advantages))
    scale = reduce(torch.ones(2, 3))
    # Members of trajectory b's row t take scale[b, t]: rows are [b, t] = (0, 0), (0, 1), (1, 0), (1, 1).
    member_scale = torch.stack([scale[0, 0]] * 3 + [scale[0, 1]] * 2 + [scale[1, 0]] * 3 + [scale[1, 1]] * 2)
    torch.testing.assert_close(
        batch[SAMPLE_SUPPORT_FIELD].columns[SampleSupportColumn.MEMBER_VALUES], weights * member_scale
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
async def test_the_old_logprob_forward_gets_support_only_under_replay(replay, tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=SupportFakeGenerator(), top_k=2)
    trainer.cfg.trainer.algorithm.enable_sample_support_replay = replay
    _, _, batch = await _top_k_batch(trainer)
    sent = []

    def fake_forward(model, data, key, mini_batch_boundaries):
        sent.append(set(data.keys()))
        return torch.zeros(batch["loss_mask"].shape)

    trainer._execute_forward_pass = fake_forward
    trainer.fwd_logprobs_values_reward(batch)

    assert sent and all((SAMPLE_SUPPORT_FIELD in keys) == replay for keys in sent)


@pytest.mark.asyncio
async def test_trainer_top_k_zero_leaves_the_base_normalization_alone(tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer, generator=SupportFakeGenerator())
    _, _, batch = await _top_k_batch(trainer)
    batch = trainer.compute_advantages_and_returns(batch)
    support = batch[SAMPLE_SUPPORT_FIELD]

    batch = trainer._normalize_advantages(batch, [(0, 2)])

    assert batch[SAMPLE_SUPPORT_FIELD] is support
    assert len(support.columns) == 2


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


def test_trainer_logs_the_rollout_metrics_from_the_real_rewards(tokenizer):
    out = _generator_output()
    out["stop_reasons"] = ["length", "stop"]
    trainer = _trainer(use_task_reward=False, kl_coef=2.0, tokenizer=tokenizer)

    out, uids = trainer.postprocess_generator_output(out, ["u", "u"])
    batch = trainer.convert_to_training_input(out, uids)
    batch["action_log_probs"] = torch.tensor([[-1.0, -1.0, -1.0], [0.0, -2.0, -2.0]])
    batch["values"] = None
    trainer.compute_advantages_and_returns(batch)

    assert trainer.all_metrics["opd/truncation_rate"] == 0.5
    assert trainer.all_metrics["opd/repetition_rate"] == 0.0  # both rollouts are too short to check
    # Pure mode zeroed the rewards, but row 0 (reward 1.0) still counts as correct: log q - log p averages
    # 0.6 on it and 1.35 on row 1's two trained tokens.
    assert trainer.all_metrics["opd/informativeness"] == pytest.approx(0.6 - 1.35)
    assert ROLLOUT_CORRECT_KEY not in batch  # popped, never shipped to workers


def test_trainer_repetition_rate_counts_looping_token_ids():
    out = _generator_output()
    rng = random.Random(0)
    out["response_ids"] = [[17, 42, 99] * 1_000, [rng.randrange(150_000) for _ in range(3_000)]]
    out["loss_masks"] = [[1] * 3_000, [1] * 3_000]
    trainer = _trainer(use_task_reward=True)

    trainer.postprocess_generator_output(out, ["u", "u"])

    assert trainer.all_metrics["opd/repetition_rate"] == 0.5  # the loop, not the varied rollout


def test_trainer_padding_rows_do_not_count_toward_informativeness(tokenizer):
    trainer = _trainer(use_task_reward=False, tokenizer=tokenizer)
    trainer.dispatch.get_lcm_dp_size.return_value = 4  # two rows padded to four

    out, uids = trainer.postprocess_generator_output(_generator_output(), ["u", "u"])
    batch = trainer.convert_to_training_input(out, uids)

    assert batch[ROLLOUT_CORRECT_KEY].tolist() == [True, False, False, False]
    # Padding rows copy row 0's tensors, a correct rollout, but their loss mask is all zero.
    batch["action_log_probs"] = torch.tensor([[-1.0, -1.0, -1.0], [0.0, -2.0, -2.0]] + [[-1.0, -1.0, -1.0]] * 2)
    batch["values"] = None
    trainer.compute_advantages_and_returns(batch)
    assert trainer.all_metrics["opd/informativeness"] == pytest.approx(0.6 - 1.35)


def test_trainer_needs_correctness_from_postprocess():
    trainer = _trainer(use_task_reward=False)
    with pytest.raises(RuntimeError, match="rollout_correct"):
        trainer.convert_to_training_input(_generator_output(), ["u", "u"])


def test_trainer_skips_truncation_without_stop_reasons():
    out = _generator_output()
    out["stop_reasons"] = None
    trainer = _trainer(use_task_reward=False)

    trainer.postprocess_generator_output(out, ["u", "u"])

    assert "opd/truncation_rate" not in trainer.all_metrics
    assert trainer.all_metrics["opd/repetition_rate"] == 0.0


def batch_teacher(batch) -> torch.Tensor:
    return torch.tensor([[-0.3, -0.4, -0.5], [0.0, -0.6, -0.7]])


def test_trainer_missing_teacher_field_is_loud():
    trainer = _trainer(use_task_reward=False)
    out = _generator_output()
    out.pop(TEACHER_LOGPROBS_KEY)
    with pytest.raises(RuntimeError, match="teacher_logprobs"):
        trainer.convert_to_training_input(out, ["u", "u"])
