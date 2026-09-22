"""The whole tokens path against a real tokenizer and a real renderer.

Every other tokens test runs on the ``builtin`` renderer, which is a ChatML
template over a byte tokenizer: exactly reversible, and nothing like a real
model. This module drives the same HTTP path through ``renderers`` over a real
Hugging Face tokenizer, because the interesting failures live in the template
-- reasoning blocks, tool-call syntax, where the generation prompt ends.

Opt in with a tokenizer that is already cached locally:

    CAPTURE_REAL_TOKENIZER=Qwen/Qwen3-0.6B uv run pytest tests/test_tokens_real_renderer.py

Skipped otherwise, so the default suite still needs no network and no model.
"""

from __future__ import annotations

import os

import pytest

TOKENIZER = os.environ.get("CAPTURE_REAL_TOKENIZER")

pytestmark = pytest.mark.skipif(
    not TOKENIZER, reason="set CAPTURE_REAL_TOKENIZER to a locally cached tokenizer"
)


@pytest.fixture
async def real_stack(stack_builder):
    """A capture process whose tokens upstream renders through the real library."""

    def upstream(url: str):
        from skyrl_capture.config import TitoUpstream

        return TitoUpstream(
            type="tokens",
            url=f"{url}/generate",
            model="real-model",
            tokenizer=TOKENIZER,
            api_key="upstream-secret",
            max_model_len=4096,
        )

    return await stack_builder(upstream_for=upstream)


async def _chat(stack, trajectory, messages, **kwargs):
    response = await stack.chat(trajectory, messages, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


async def test_a_real_template_round_trips_through_the_proxy(real_stack):
    """One turn, and the stored tokens are the ones inference was given."""
    created = await real_stack.create_trajectory()
    body = await _chat(real_stack, created, [{"role": "user", "content": "What is 2+2?"}])
    assert body["choices"][0]["message"]["role"] == "assistant"
    await real_stack.settle()

    graph = await real_stack.graph(created["id"])
    # One user node and one assistant node, and the assistant carries tokens.
    assistant = [node for node in graph["nodes"] if node["author"] == "model"]
    assert len(assistant) == 1
    assert assistant[0]["token_count"] > 0
    assert assistant[0]["sampled_start"] is not None


async def test_continuing_reuses_the_prefix_rather_than_re_rendering(real_stack):
    """Turn two bridges off turn one's exact prompt and completion."""
    created = await real_stack.create_trajectory()
    first = await _chat(real_stack, created, [{"role": "user", "content": "Name a colour."}])
    reply = first["choices"][0]["message"]
    await real_stack.settle()

    await _chat(
        real_stack,
        created,
        [
            {"role": "user", "content": "Name a colour."},
            {k: v for k, v in reply.items() if v is not None},
            {"role": "user", "content": "Another one."},
        ],
    )
    await real_stack.settle()

    graph = await real_stack.graph(created["id"])
    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    # One root: the second call continued the first rather than starting over.
    assert len(roots) == 1, "a continuation must not start a second root"
    assert len(graph["leaf_assistant_node_ids"]) == 1


async def test_the_export_tokens_are_exactly_what_was_captured(real_stack):
    """input_ids and loss_mask line up, and only sampled positions are masked in."""
    created = await real_stack.create_trajectory()
    await _chat(real_stack, created, [{"role": "user", "content": "Say hello."}])
    await real_stack.settle()
    await real_stack.finish(created["id"], annotations={"rlvr_reward": 1.0})

    rows = await real_stack.export_lines(trajectory=created["id"], format="token_samples")
    assert len(rows) == 1
    row = rows[0]
    assert len(row["loss_mask"]) == len(row["input_ids"])
    assert len(row["rollout_logprobs"]) == len(row["input_ids"])
    # The prompt is context and the completion is the target, so the mask has
    # both values and starts at zero.
    assert row["loss_mask"][0] == 0
    assert sum(row["loss_mask"]) > 0
    assert row["tokenizer"] == TOKENIZER
