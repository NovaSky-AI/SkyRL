"""The tokens a renderer produces must not move when `transformers` moves.

`pyproject.toml` caps the `tokens` extra at `transformers<6`. The cap used to be
`<5`, on the belief that the pinned `renderers` revision imported a path 5.x had
removed. It did not -- every symbol still resolves on 5.16.1, and rendering is
byte-identical on both -- so the cap was lifted.

What the cap was really protecting is not the import. It is that a capture
rollout and a trainer rollout render the *same tokens*; if a `transformers`
release changed the output by even one token, training would silently be
attributed against a prompt that was never sent.

So this pins the output instead of the version. The golden is a SHA over the
rendered token ids for a fixed conversation -- system, user, an assistant turn
carrying a reasoning block, and a tool result -- which is the shape where
template differences actually show up.

If this fails after an upgrade, do not re-record the golden until you know which
tokens changed and whether the trainer's renderer changed with it.

Opt in with a tokenizer that is already cached locally, as with the other real
renderer tests:

    CAPTURE_REAL_TOKENIZER=Qwen/Qwen3-0.6B uv run pytest tests/test_renderer_parity.py
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

TOKENIZER = os.environ.get("CAPTURE_REAL_TOKENIZER")

pytestmark = pytest.mark.skipif(
    not TOKENIZER, reason="set CAPTURE_REAL_TOKENIZER to a locally cached tokenizer"
)

# Deliberately includes a `<think>` block: `thinking_retention="all"` is what
# keeps an earlier assistant turn rendering as the tokens that were sampled, and
# it is the setting most likely to be affected by a template change.
CONVERSATION = [
    {"role": "system", "content": "You are a terminal agent."},
    {"role": "user", "content": "List the files in /app and explain."},
    {"role": "assistant", "content": "<think>\nI should run ls.\n</think>\n\nRunning ls now."},
    {"role": "user", "content": "total 0\ndrwxr-xr-x 2 root root 40 Jan 1 00:00 ."},
]

# tokenizer -> (sha256 of the rendered token ids, token count).
# Recorded on transformers 4.57.6 and confirmed identical on 5.16.1.
GOLDEN = {
    "Qwen/Qwen3-0.6B": ("e7e97a963d08b92672294515374bd81a5aeea1d2647a8a040d86196481eea166", 66),
    "Qwen/Qwen3-4B-Instruct-2507": (
        "e7e97a963d08b92672294515374bd81a5aeea1d2647a8a040d86196481eea166",
        66,
    ),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": (
        "e7e97a963d08b92672294515374bd81a5aeea1d2647a8a040d86196481eea166",
        66,
    ),
}


def _render(tokenizer: str) -> list[int]:
    from renderers import AutoRendererConfig, create_renderer_pool

    # The same configuration `tito/renderer.py` builds its pool with.
    pool = create_renderer_pool(
        tokenizer, AutoRendererConfig(thinking_retention="all"), size=1
    )
    return list(pool.render(CONVERSATION).token_ids)


def test_rendered_tokens_match_the_recorded_golden() -> None:
    if TOKENIZER not in GOLDEN:
        pytest.skip(
            f"no golden recorded for {TOKENIZER!r}; add one to GOLDEN after checking "
            "the tokens against the trainer's renderer"
        )
    expected_sha, expected_len = GOLDEN[TOKENIZER]
    token_ids = _render(TOKENIZER)
    actual_sha = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
    assert len(token_ids) == expected_len, (
        f"{TOKENIZER} now renders {len(token_ids)} tokens, was {expected_len}. "
        "A transformers or renderers upgrade changed the prompt."
    )
    assert actual_sha == expected_sha, (
        f"{TOKENIZER} renders the same number of tokens but different ones. "
        "Find out which before re-recording."
    )


def test_transformers_still_exposes_what_renderers_imports() -> None:
    """The symbols the `<5` cap was wrongly believed to protect.

    Cheaper to fail here, naming the symbol, than inside a half-imported module
    with "cannot import name 'AutoTokenizer'".
    """
    import transformers

    assert hasattr(transformers, "PreTrainedTokenizer")
    assert hasattr(transformers, "PreTrainedTokenizerFast")
    assert hasattr(transformers, "AutoTokenizer")
    assert hasattr(transformers, "AutoConfig")

    from transformers.feature_extraction_utils import BatchFeature  # noqa: F401
    from transformers.models.auto.tokenization_auto import get_tokenizer_config  # noqa: F401
    from transformers.tokenization_utils import PreTrainedTokenizer  # noqa: F401
