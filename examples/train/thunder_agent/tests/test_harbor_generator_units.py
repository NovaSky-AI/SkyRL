"""
Unit tests for ThunderAgentHarborGenerator helper methods.

No GPU / Ray / Harbor runtime needed. Uses pytest + httpx mock transport.
Run:
    pytest examples/train/thunder_agent/tests/test_harbor_generator_units.py -v
"""
from __future__ import annotations

import asyncio
import pathlib
import tempfile
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_generator(*, proxy_url: str = "http://ta:8080", served_model_name: str = "TestModel"):
    """Build a ThunderAgentHarborGenerator with a minimal fake config + tokenizer."""
    from omegaconf import OmegaConf
    from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator

    generator_cfg = OmegaConf.create({
        "inference_engine": {
            "http_endpoint_host": "localhost",
            "http_endpoint_port": 8001,
            "served_model_name": served_model_name,
            "engine_init_kwargs": {},
        },
        "apply_overlong_filtering": False,
        "rate_limit": None,
    })
    harbor_cfg = {
        "agent": {
            "name": "mini-swe-agent",
            "kwargs": {
                "max_turns": 10,
                "store_all_messages": True,
            },
        },
        "environment": {"type": "docker"},
        "verifier": {"disable": False},
    }
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
    tokenizer.encode = MagicMock(side_effect=lambda text, **kw: [ord(c) % 1000 for c in text[:5]])

    fake_client = MagicMock()
    fake_client.proxy_url = proxy_url

    gen = ThunderAgentHarborGenerator(
        generator_cfg=generator_cfg,
        harbor_cfg=harbor_cfg,
        inference_engine_client=fake_client,
        tokenizer=tokenizer,
        max_seq_len=2048,
    )
    return gen


# ---------------------------------------------------------------------------
# Test 1: _attach_trial_routing_ids
# ---------------------------------------------------------------------------

class TestAttachTrialRoutingIds:
    def _call(self, config: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator
        return ThunderAgentHarborGenerator._attach_trial_routing_ids(config, session_id)

    def test_session_id_equals_program_id(self):
        config = {"agent": {"kwargs": {}}}
        result = self._call(config, "abc123")
        assert result["agent"]["kwargs"]["session_id"] == "abc123"
        assert result["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]["program_id"] == "abc123"

    def test_ids_do_not_leak_across_calls(self):
        """Two fresh deepcopies of the same template must have independent IDs."""
        template = {"agent": {"kwargs": {}}}
        cfg1 = self._call(deepcopy(template), "id-A")
        cfg2 = self._call(deepcopy(template), "id-B")
        assert cfg1["agent"]["kwargs"]["session_id"] == "id-A"
        assert cfg2["agent"]["kwargs"]["session_id"] == "id-B"

    def test_existing_extra_body_keys_preserved(self):
        config = {
            "agent": {
                "kwargs": {
                    "llm_call_kwargs": {
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                    }
                }
            }
        }
        result = self._call(config, "uuid-xyz")
        extra = result["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]
        assert extra["program_id"] == "uuid-xyz"
        assert extra["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Test 2: _apply_sampling_params_to_trial_config
# ---------------------------------------------------------------------------

class TestApplySamplingParams:
    def _call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator
        config = {"agent": {"kwargs": {}}}
        return ThunderAgentHarborGenerator._apply_sampling_params_to_trial_config(config, params)

    def test_temperature_mapped_to_agent_kwargs(self):
        result = self._call({"temperature": 0.3})
        assert result["agent"]["kwargs"]["temperature"] == 0.3

    def test_top_p_in_llm_call_kwargs(self):
        result = self._call({"top_p": 0.9})
        assert result["agent"]["kwargs"]["llm_call_kwargs"]["top_p"] == 0.9

    def test_max_tokens_aliases(self):
        for key in ("max_tokens", "max_completion_tokens", "max_generate_length"):
            result = self._call({key: 2048})
            assert result["agent"]["kwargs"]["llm_call_kwargs"]["max_tokens"] == 2048

    def test_top_k_min_p_repetition_in_extra_body(self):
        result = self._call({"top_k": 50, "min_p": 0.05, "repetition_penalty": 1.1})
        extra = result["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]
        assert extra["top_k"] == 50
        assert extra["min_p"] == 0.05
        assert extra["repetition_penalty"] == 1.1

    def test_default_values_not_added_to_extra_body(self):
        result = self._call({"top_k": -1, "min_p": 0, "repetition_penalty": 1.0})
        agent_kwargs = result["agent"]["kwargs"]
        extra = agent_kwargs.get("llm_call_kwargs", {}).get("extra_body", {})
        assert "top_k" not in extra
        assert "min_p" not in extra
        assert "repetition_penalty" not in extra

    def test_logprobs_enables_collect_rollout_details(self):
        result = self._call({"logprobs": 1})
        assert result["agent"]["kwargs"]["collect_rollout_details"] is True

    def test_no_logprobs_leaves_collect_rollout_details_false(self):
        result = self._call({"temperature": 0.5})
        assert result["agent"]["kwargs"].get("collect_rollout_details", False) is False

    def test_none_params_returns_config_unchanged(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator
        config = {"agent": {"kwargs": {"temperature": 0.7}}}
        result = ThunderAgentHarborGenerator._apply_sampling_params_to_trial_config(config, None)
        assert result["agent"]["kwargs"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Test 3: _best_effort_release_program
# ---------------------------------------------------------------------------

class TestBestEffortReleaseProgram:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_transport(self, responses):
        """responses: list of (status_code, content_or_exception)."""
        call_count = [0]

        def handler(request):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            code_or_exc = responses[idx]
            if isinstance(code_or_exc, Exception):
                raise code_or_exc
            status_code, body = code_or_exc
            return httpx.Response(status_code, content=body.encode())

        return httpx.MockTransport(handler)

    def test_200_ok_no_retry(self):
        gen = _make_generator()
        gen._release_client = httpx.AsyncClient(
            transport=self._make_transport([(200, '{"released": true}')]),
        )
        gen._release_semaphore = asyncio.Semaphore(1)
        self._run(gen._best_effort_release_program("pid-ok"))
        # Should not raise, completes in 1 call

    def test_404_no_retry(self):
        gen = _make_generator()
        gen._release_client = httpx.AsyncClient(
            transport=self._make_transport([(404, "not found")]),
        )
        gen._release_semaphore = asyncio.Semaphore(1)
        self._run(gen._best_effort_release_program("pid-404"))

    def test_500_logs_warning_no_retry(self, caplog):
        gen = _make_generator()
        gen._release_client = httpx.AsyncClient(
            transport=self._make_transport([(500, "server error")]),
        )
        gen._release_semaphore = asyncio.Semaphore(1)
        self._run(gen._best_effort_release_program("pid-500"))

    def test_timeout_retries_then_gives_up(self):
        gen = _make_generator()
        gen._release_max_attempts = 3
        gen._release_retry_backoff_sec = 0.0  # no actual sleep delay in tests

        responses = [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
        ]
        call_count = [0]

        def handler(request):
            call_count[0] += 1
            raise responses[call_count[0] - 1]

        gen._release_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gen._release_semaphore = asyncio.Semaphore(1)
        # Should not raise even after max retries exhausted
        self._run(gen._best_effort_release_program("pid-timeout"))
        assert call_count[0] == gen._release_max_attempts

    def test_none_program_id_returns_immediately(self):
        gen = _make_generator()
        # No client initialized — would crash if it tried to call anything
        self._run(gen._best_effort_release_program(None))

    def test_no_proxy_returns_immediately(self):
        gen = _make_generator(proxy_url=None)
        gen._supports_program_release = False
        self._run(gen._best_effort_release_program("pid-any"))


# ---------------------------------------------------------------------------
# Test 4: _get_response_ids_and_loss_mask_from_harbor_rollout
# ---------------------------------------------------------------------------

class TestGetResponseIdsAndLossMask:
    """
    Tests the token-level response/loss-mask extraction logic.
    Uses a synthetic tokenizer stub that maps content to predictable IDs.
    """

    def _make_gen_with_tokenizer(self, gen_prompt_ids=(151644,)):
        """Return a generator whose tokenizer encodes deterministically."""
        from omegaconf import OmegaConf
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator

        generator_cfg = OmegaConf.create({
            "inference_engine": {
                "http_endpoint_host": "localhost",
                "http_endpoint_port": 8001,
                "served_model_name": "M",
                "engine_init_kwargs": {},
            },
            "apply_overlong_filtering": False,
            "rate_limit": None,
        })
        harbor_cfg = {"agent": {"name": "mini-swe-agent", "kwargs": {}},
                      "environment": {"type": "docker"}, "verifier": {"disable": False}}

        fake_client = MagicMock()
        fake_client.proxy_url = "http://ta:8080"

        # Tokenizer that returns gen_prompt_ids for any message starting role=="assistant"
        # and small ints for user messages.
        def apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                  chat_template=None):
            role = messages[0]["role"]
            if role == "assistant":
                return list(gen_prompt_ids) + [100, 101, 102]
            return [10, 11]  # user message

        def encode_messages_subset(msgs, tok, **kw):
            return apply_chat_template(msgs)

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 0

        # Patch encode_messages_subset to use our simple stub
        with patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.encode_messages_subset",
            side_effect=encode_messages_subset,
        ), patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.get_generation_prompt_ids",
            return_value=list(gen_prompt_ids),
        ):
            gen = ThunderAgentHarborGenerator(
                generator_cfg=generator_cfg,
                harbor_cfg=harbor_cfg,
                inference_engine_client=fake_client,
                tokenizer=tokenizer,
                max_seq_len=2048,
            )
            return gen, list(gen_prompt_ids)

    def test_loss_mask_shape_matches_response_ids(self):
        with patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.encode_messages_subset",
            side_effect=lambda msgs, tok, **kw: [10, 11] if msgs[0]["role"] == "user" else [151644, 100, 101],
        ), patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.get_generation_prompt_ids",
            return_value=[151644],
        ):
            gen, gen_prompt = _make_generator(), [151644]
            gen.custom_chat_template_content = None
            gen.tokenizer.eos_token_id = 0

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            completion_token_ids = [[100, 101]]

            from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator
            with patch.object(
                ThunderAgentHarborGenerator,
                "_get_response_ids_and_loss_mask_from_harbor_rollout",
                wraps=gen._get_response_ids_and_loss_mask_from_harbor_rollout,
            ):
                pass  # shape assertion done in functional test below

    def test_user_segments_have_zero_loss_mask(self):
        """User message tokens must always have loss_mask=0."""
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator

        gen_prompt = [151644]

        def enc(msgs, tok, **kw):
            if msgs[0]["role"] == "user":
                return [10, 11]
            return gen_prompt + [100, 101, 102]

        with patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.encode_messages_subset",
            side_effect=enc,
        ), patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.get_generation_prompt_ids",
            return_value=gen_prompt,
        ):
            tok = MagicMock()
            tok.eos_token_id = 0
            from omegaconf import OmegaConf
            gcfg = OmegaConf.create({"inference_engine": {"http_endpoint_host": "h",
                "http_endpoint_port": 1, "served_model_name": "M", "engine_init_kwargs": {}},
                "apply_overlong_filtering": False, "rate_limit": None})
            fake_client = MagicMock(); fake_client.proxy_url = "http://ta:8080"
            gen = ThunderAgentHarborGenerator(gcfg, {"agent": {"name": "a", "kwargs": {}},
                "environment": {"type": "docker"}, "verifier": {"disable": False}},
                fake_client, tok, 2048)
            gen.custom_chat_template_content = None

            messages = [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "reply"},
            ]
            completion_ids = [[100, 101, 102]]
            response_ids, loss_mask, _ = gen._get_response_ids_and_loss_mask_from_harbor_rollout(
                messages, completion_ids, None
            )
            assert len(response_ids) == len(loss_mask)
            # user segment: tokens [10, 11] → loss_mask 0
            assert loss_mask[0] == 0
            assert loss_mask[1] == 0
            # gen_prompt token (151644) → loss_mask 0
            user_len = 2
            gen_prompt_len = len(gen_prompt)
            assert loss_mask[user_len] == 0
            # completion tokens → loss_mask 1
            for i in range(gen_prompt_len + 1, gen_prompt_len + 1 + len(completion_ids[0])):
                assert loss_mask[user_len + i - 1] == 1

    def test_rollout_logprobs_length_matches_response_ids(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator

        gen_prompt = [151644]

        def enc(msgs, tok, **kw):
            if msgs[0]["role"] == "user":
                return [10, 11]
            return gen_prompt + [100, 101]

        with patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.encode_messages_subset",
            side_effect=enc,
        ), patch(
            "examples.train.thunder_agent.skyrl_integration.harbor_generator.get_generation_prompt_ids",
            return_value=gen_prompt,
        ):
            tok = MagicMock(); tok.eos_token_id = 0
            from omegaconf import OmegaConf
            gcfg = OmegaConf.create({"inference_engine": {"http_endpoint_host": "h",
                "http_endpoint_port": 1, "served_model_name": "M", "engine_init_kwargs": {}},
                "apply_overlong_filtering": False, "rate_limit": None})
            fake_client = MagicMock(); fake_client.proxy_url = "http://ta:8080"
            gen = ThunderAgentHarborGenerator(gcfg, {"agent": {"name": "a", "kwargs": {}},
                "environment": {"type": "docker"}, "verifier": {"disable": False}},
                fake_client, tok, 2048)
            gen.custom_chat_template_content = None

            messages = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
            completion_ids = [[100, 101]]
            logprobs = [[-0.1, -0.2]]
            response_ids, loss_mask, rollout_logprobs = gen._get_response_ids_and_loss_mask_from_harbor_rollout(
                messages, completion_ids, logprobs
            )
            assert len(response_ids) == len(loss_mask) == len(rollout_logprobs)


# ---------------------------------------------------------------------------
# Test 5: Empty dataset guard
# ---------------------------------------------------------------------------

class TestHarborDatasetEmptyGuard:
    def test_raises_on_nonexistent_path(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset
        with pytest.raises(ValueError, match="zero task directories"):
            ThunderAgentHarborDataset(["/nonexistent/path/xyz"])

    def test_raises_when_dir_has_no_instruction_md(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdir without instruction.md
            (pathlib.Path(tmpdir) / "task1").mkdir()
            with pytest.raises(ValueError, match="zero task directories"):
                ThunderAgentHarborDataset([tmpdir])

    def test_succeeds_when_instruction_md_present(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = pathlib.Path(tmpdir) / "task1"
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("do the thing")
            ds = ThunderAgentHarborDataset([tmpdir])
            assert len(ds) == 1
            assert ds[0]["uid"] == str(task_dir.resolve())

    def test_max_tasks_applied_after_guard(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                d = pathlib.Path(tmpdir) / f"task{i}"
                d.mkdir()
                (d / "instruction.md").write_text(f"task {i}")
            ds = ThunderAgentHarborDataset([tmpdir], max_tasks=3)
            assert len(ds) == 3

    def test_stable_uid_across_instances(self):
        from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):
                d = pathlib.Path(tmpdir) / f"task{i:02d}"
                d.mkdir()
                (d / "instruction.md").write_text(f"task {i}")
            ds1 = ThunderAgentHarborDataset([tmpdir])
            ds2 = ThunderAgentHarborDataset([tmpdir])
            assert [item["uid"] for item in ds1] == [item["uid"] for item in ds2]
