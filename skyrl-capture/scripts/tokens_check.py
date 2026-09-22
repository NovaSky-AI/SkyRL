"""Prove that token capture stores exactly what inference saw.

Run it:

    uv run python scripts/tokens_check.py                      # no setup at all
    uv run python scripts/tokens_check.py --tokenizer Qwen/Qwen3-0.6B

Nothing here is mocked except the inference call itself, which has to be, so
the tokens are known. Rendering, prefix matching, bridging, attribution and
commit are the real code paths a live trajectory takes.

The claim being checked, after every single turn:

    the tokens stored along the path from the root to the assistant node
    are exactly the prompt that was sent, followed by the completion that
    came back

If that holds, an export's `input_ids` are what the model actually saw. If it
ever fails, they are a reconstruction, and training on them is training on a
guess. The last scenario corrupts a prompt on purpose, so you can see the
check refuse it rather than take it on faith that it would.
"""

from __future__ import annotations

import argparse
import sys

from skyrl_capture.tito.renderer import build_renderer
from skyrl_capture.tito.trace import TokenTrace
from skyrl_capture.tito.types import ModelTurnResult, TokenError

FAILURES: list[str] = []
TURN = [0]


class Engine:
    """Stands in for the inference endpoint, so the tokens are known."""

    def __init__(self, renderer, tokenizer_name: str) -> None:
        self.renderer = renderer
        self._encode = self._make_encoder(tokenizer_name)
        self.stop = renderer.get_stop_token_ids()[0]

    @staticmethod
    def _make_encoder(name: str):
        if name == "builtin":
            from skyrl_capture.tito.renderer import ByteTokenizer

            return ByteTokenizer().encode
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    def sample(self, text: str) -> tuple[int, ...]:
        return tuple(self._encode(text)) + (self.stop,)


def turn(trace, engine, messages, reply, *, tools=None, label="", corrupt_at=None):
    """One complete proxy cycle, then the invariant check."""
    trace.set_text_renderer(engine.renderer.decode, engine.renderer.turn_start_token())
    pending = trace.prepare_turn(messages, tools=tools)

    bridged = None
    if pending.bridge_transition_id is not None:
        previous_prompt = trace.transition_prompt_ids(pending.bridge_transition_id)
        previous_completion = trace.transition_completion_ids(pending.bridge_transition_id)
        tail = list(messages)[len(pending.matched_node_ids) :]
        bridged = engine.renderer.bridge(previous_prompt, previous_completion, tail, tools=tools)
    rendered = bridged if bridged is not None else engine.renderer.render(messages, tools=tools)

    prompt = list(rendered.token_ids)
    if corrupt_at is not None:
        prompt[corrupt_at] = 99_999

    completion = engine.sample(reply)
    TURN[0] += 1
    result = ModelTurnResult(
        prompt_token_ids=tuple(prompt),
        prompt_message_indices=tuple(rendered.message_indices),
        reused_prefix_length=rendered.reused_prefix_length,
        completion_ids=completion,
        completion_logprobs=tuple(-0.1 for _ in completion),
        assistant_message=engine.renderer.parse_response(completion, tools=tools),
        stop_reason="stop",
        model="demo",
    )
    commit = trace.commit(pending, result, exchange_id=f"ex_{TURN[0]:04d}")

    stored = tuple(trace.tokens_for(trace.path_to(commit.assistant_node_id)))
    expected = tuple(rendered.token_ids) + completion
    ok = stored == expected
    if not ok:
        FAILURES.append(label)
    print(
        f"    [{'ok' if ok else 'FAIL':>4}] {label:<38} "
        f"{'reused ' + str(rendered.reused_prefix_length) + ' tokens' if bridged else 'full render':<22} "
        f"prompt={len(rendered.token_ids):<6} stored==sent+returned: {ok}"
    )
    return commit, result


def heading(text: str) -> None:
    print(f"\n  {text}\n  {'-' * len(text)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        default="builtin",
        help="'builtin' needs nothing; a HF name uses the real renderer (downloads once)",
    )
    args = parser.parse_args()

    renderer = build_renderer(tokenizer=args.tokenizer, model="demo")
    engine = Engine(renderer, args.tokenizer)
    real = args.tokenizer != "builtin"
    print(f"\n  renderer: {renderer.name}")
    print("  checking after every turn that stored tokens == prompt sent + completion returned")

    heading("1. An agent feeding its own replies back")
    print("     The ordinary case. Each turn should reuse the previous prefix")
    print("     instead of re-rendering the conversation from scratch.")
    trace = TokenTrace("demo-basic")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    _, first = turn(trace, engine, messages, "4", label="turn 1")
    messages += [first.assistant_message, {"role": "user", "content": "And 3+3?"}]
    _, second = turn(trace, engine, messages, "6", label="turn 2, continued")
    messages += [second.assistant_message, {"role": "user", "content": "And 4+4?"}]
    turn(trace, engine, messages, "8", label="turn 3, continued")
    print(f"     -> {len(trace.nodes())} nodes, {len(trace.leaves())} leaf: one conversation, no duplication")

    heading("2. The client edits what the model said, then continues from the edit")
    print("     Both versions have to survive: the model's original is what was")
    print("     sampled, the edit is what the next turn actually saw.")
    trace = TokenTrace("demo-repair")
    messages = [{"role": "user", "content": "How many retries?"}]
    turn(trace, engine, messages, "It retries four times.", label="model says four")
    repaired = {"role": "assistant", "content": "It retries three times."}
    turn(
        trace, engine,
        messages + [repaired, {"role": "user", "content": "And the backoff?"}],
        "Doubling from 100ms.", label="client substitutes three",
    )
    print(f"     -> {len(trace.leaves())} leaves: it forked rather than overwriting the original")

    heading("3. A rewritten history (compaction)")
    print("     The transcript is thrown away and replaced by a summary. The")
    print("     shared opening is kept once; the rest is a separate branch.")
    trace = TokenTrace("demo-compaction")
    base = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Start."}]
    _, one = turn(trace, engine, base, "Step one done.", label="turn 1")
    conversation = base + [one.assistant_message, {"role": "user", "content": "Summarize."}]
    _, summary = turn(trace, engine, conversation, "Did step one.", label="turn 2, summarize")
    rebuilt = [
        base[0],
        {"role": "user", "content": f"So far: {summary.assistant_message['content']}"},
        {"role": "user", "content": "Continue."},
    ]
    turn(trace, engine, rebuilt, "Step two done.", label="turn 3, rebuilt context")
    print(f"     -> {len(trace.leaves())} leaves: the rebuilt context is its own branch")

    if real:
        heading("4. A reasoning model, and what the client sends back")
        print("     This is the one that costs you if you get it wrong.")
        trace = TokenTrace("demo-reasoning")
        messages = [{"role": "user", "content": "What is 17*23?"}]
        _, reasoned = turn(
            trace, engine, messages,
            "<think>\n17*20=340, 17*3=51, so 391\n</think>\n\n391",
            label="turn 1, model thinks",
        )
        kept = reasoned.assistant_message
        print(f"       parsed reasoning_content: {kept.get('reasoning_content', '(none)')!r}")
        turn(
            trace, engine,
            messages + [kept, {"role": "user", "content": "Times 2?"}],
            "782", label="turn 2, reasoning replayed",
        )
        stripped = {k: v for k, v in kept.items() if k != "reasoning_content"}
        turn(
            trace, engine,
            messages + [stripped, {"role": "user", "content": "Times 2?"}],
            "782", label="turn 2, reasoning DROPPED",
        )
        print("     -> both are exact, but only the replayed one reused its prefix.")
        print("        Dropping reasoning forks the trajectory and re-renders everything.")
    else:
        heading("4. A reasoning model (skipped)")
        print("     Needs a real tokenizer: --tokenizer Qwen/Qwen3-0.6B")

    heading("5. The check has teeth")
    print("     Everything above passed. That is only meaningful if the check can")
    print("     fail, so here one token of an otherwise valid prompt is altered")
    print("     before commit -- the kind of thing a renderer bug would do.")
    trace = TokenTrace("demo-corrupt")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    _, clean = turn(trace, engine, messages, "4", label="a clean turn first")
    follow = messages + [clean.assistant_message, {"role": "user", "content": "And 3+3?"}]
    try:
        turn(trace, engine, follow, "6", label="one prompt token altered", corrupt_at=4)
        print("    [FAIL] the corrupted prompt was accepted")
        FAILURES.append("corruption not detected")
    except TokenError as error:
        print(f"    [  ok] refused before storing anything: {error}")
        FAILURES[:] = [f for f in FAILURES if f != "one prompt token altered"]
    turn(trace, engine, follow, "6", label="the same turn, uncorrupted")

    print()
    if FAILURES:
        print(f"  FAILURES: {', '.join(FAILURES)}\n")
        return 1
    print("  every turn stored exactly the tokens inference was given and returned,")
    print("  and a deliberately corrupted prompt was refused.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
