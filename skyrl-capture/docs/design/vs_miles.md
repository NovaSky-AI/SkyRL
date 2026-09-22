# How our token capture differs from Miles

Miles is an RL post-training framework from LMSYS (forked from slime). Its
token-in/token-out path is described in
[No Token Left Behind](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/)
and implemented in [radixark/miles](https://github.com/radixark/miles). We solve
the same problem in `skyrl_capture.tito`. This note is a comparison of the
two, written against Miles at the revision cloned in September 2026.

Both systems agree on the goal, and both state it the same way:

> The tokens the trainer sees must be exactly the tokens the engine saw and
> produced. No detokenize/retokenize round trip in between.

We disagree about how to get there. The disagreement is one decision deep, and
almost everything else follows from it.

## The short answer

**Miles keeps the model's Jinja chat template as the source of truth and
recovers token boundaries by diffing two renders. We replace the Jinja template
with a renderer that emits token boundaries directly, so there is nothing to
diff.**

Miles renders the conversation without the new message, renders it with the new
message, checks the second string starts with the first, and tokenizes the
difference:

```python
# miles/utils/chat_template_utils/tito_tokenizer.py
text_without = self.apply_chat_template(base_messages, add_generation_prompt=False, tools=tools)
text_with    = self.apply_chat_template(base_messages + appended_messages, ...)
if not text_with.startswith(text_without):
    raise ValueError(f"rendered suffix diff failed for {roles}")
return self._encode_text(text_with[len(text_without):])
```

The result is spliced onto a per-session token buffer. Because the template is
opaque, the splice point has to be patched per model family — that is what
`merge_tokens` is for.

We call a renderer that returns, for every token, the index of the message it
came from:

```python
# src/skyrl_capture/tito/renderer.py
rendered = self._pool.render(list(messages), tools=..., add_generation_prompt=True)
return RenderedPrompt(
    token_ids=tuple(rendered.token_ids),
    message_indices=tuple(rendered.message_indices),   # -1 == template scaffold
)
```

`attribute_prompt_tokens` then cuts the prompt into one token block per message.
No second render, no string comparison, no subtraction.

## Why we can do that

The renderer is Prime Intellect's `renderers` library, which is not a Jinja
template engine. It is seventeen hand-written renderers — Qwen3, Qwen3.5,
Qwen3.6, Qwen3-VL, GLM-4.5, GLM-5, Kimi-K2, Kimi-K2.5, Llama-3, DeepSeek-V3,
Nemotron-3, MiniMax-M2, gpt-oss, and others — each of which builds a turn by
emitting one message at a time.

From gpt_oss.py, on bridging:

> Each new message is rendered in isolation via `enc.render(m)` — same
> primitive `render()` uses — so the bridge can never drift from the full
> re-render.

That is the whole trick. If the renderer builds the sequence message by message,
it knows which tokens came from which message as it goes. Attribution is a
by-product of construction, not something recovered afterwards. The same is true
of `sampled_mask` (did the model actually emit this token, or is it scaffolding
wrapped around the model's output?) and `is_content` (message body vs. role
tags).

A Jinja template cannot tell you any of this. It takes a whole message list and
returns one string. Diffing is the only way to get boundaries back out of it,
and diffing is what Miles does.

## What this does to model-specific code

It does **not** eliminate per-model work. It moves it and changes what it has to
get right. This is the honest version of the comparison.

**Miles** has 15 `TITOTokenizer` subclasses, 4 of which override the splice
logic, plus 9 bundled "fixed" Jinja templates that patch the model's own
template so it renders append-only. Each override exists to repair a boundary
the diff cannot see:

| Family | Patch | Reason |
| --- | --- | --- |
| Qwen3 | Append a `\n` token to the prefix if it ends in `<|im_end|>` | The engine stops at `<|im_end|>`; the template emits `<|im_end|>\n` |
| GLM-4.7 | Strip the last prefix token if it is `<|user|>` or `<|observation|>` | Those are both stop tokens *and* next-message start tokens, so the boundary is ambiguous |
| MiniMax-M2.5 | Own `merge_tokens` | Same class of boundary quirk |
| DeepSeek-V4 | Own `tokenize_additional_messages` | The dummy-prefix render trick does not work for this template at all |
| Nemotron-3, Kimi-K2.5, Inkling | Template kwargs, fixed Jinja, custom parsing | Templates that drop reasoning or re-pack content |

There is also a per-family whitelist of which roles may be appended
(`allowed_append_roles`), because whether a render stays append-only depends on
what role comes next. Most families support `{tool}`, `{tool, user}`, or
`{tool, user, system}` — not the full four.

**Ours** has per-model code too: the ~16 renderers. But a renderer implements the
model's format, which is a thing you can read the spec for and test directly. It
does not have to anticipate how a *diff* of that format will go wrong. And the
boundary case Miles patches by hand is something the renderer already owns:

```python
# renderers/base.py
def trim_to_turn_close(previous_prompt_ids, previous_completion_ids,
                       close_token_ids, *, synthesize_close=None):
    """Return the longest prefix of prev_prompt + prev_completion that ends at a
    turn-close token, or None if none exists and synthesize_close is not
    provided."""
```

The renderer knows its own close tokens, so it trims to a real turn boundary
instead of guessing, and synthesizes the close when the previous turn was cut
off at `max_tokens`. Where Miles appends a newline for Qwen3, the Qwen3 renderer
just knows where its turn ends.

Crucially, when it cannot prove it preserved the prefix, it returns `None`
rather than a guess. A `None` costs us a full re-render. It never costs us a
wrong token.

## The cost of our approach

If `renderers` has no hand-coded renderer for a model, it falls back to
`DefaultRenderer`, which wraps `apply_chat_template`. And `DefaultRenderer` gets
attribution the only way an opaque template allows:

```python
# renderers/default.py
for idx, message in enumerate(messages):
    cur_ids = self._apply(messages[: idx + 1], tools=tools)
    new_tokens = cur_ids[prev_len:]
    message_indices.extend([idx] * len(new_tokens))
    prev_len = len(cur_ids)
```

That is Miles's mechanism, done worse: N renders per call instead of two, no
`sampled_mask`, and `bridge_to_next_turn` returns `None` unconditionally, so
every turn is a full render.

So the trade is real:

- **Miles works on any model with a Jinja template**, best-effort, and degrades
  by logging non-critical mismatches.
- **We work excellently on models with a renderer and poorly on models
  without one.** Adding a model means writing a renderer, which is more work
  than writing a `merge_tokens` override.

For our use case — a small number of models under active RL training — that is
the right side of the trade. For a framework that has to accept whatever
checkpoint a customer brings, Miles's choice is defensible.

## Divergence: hard-fail vs. fork

The second real difference. What happens when the client sends something that is
not a clean continuation?

Miles rejects it:

```python
# miles/utils/chat_template_utils/message_matcher_hub/funcs.py
def assert_messages_append_only_with_allowed_role(stored_messages, new_messages, ...):
    """Assert new_messages is an append-only extension of stored_messages."""
    ...
    raise ValueError(f"message mismatch at index {i} ...")
```

Append-only is enforced at three levels — message list, template render, and
token sequence — and a violation is an error the agent has to deal with. Their
`LinearTrajectory` allows rolling back at most one assistant step
(`MAX_ASSISTANT_ROLLBACK_STEPS = 1`, with a comment saying it is hardcoded for
now).

We branch. `prepare_turn` finds the longest matching prefix in message space,
looks for a bridge transition at that point, and if the client's history
diverges — a reworded assistant turn, a stripped reasoning block, a compaction,
an edited earlier message — we render fully and commit a new branch off the
divergence point. Nothing fails. The graph grows a fork.

This matters because the behaviour is common, not exotic. From
`docs/design/tokens_log.md`, every one of these forks rather than errors:

| Scenario | What the client did |
| --- | --- |
| Reasoning dropped | Stripped `reasoning_content`, as most SDKs do |
| Client repair | Edited the assistant message |
| Tools widened | Same messages, new tool set |
| Mid-history edit | Rewrote an earlier message |
| Compaction | Replaced the transcript with a summary |
| Late system prompt | Added a system message on turn two |

Miles's v2 `TrajectoryNode` is a forest too, so they are not strictly linear
anymore — but their forest is still append-only by construction ("The forest is
append-only and `seq` is the only ordering key"), and branches come from the
agent deliberately re-generating, not from the client's history drifting.

We also tag every node with `author`: `"model"` if the tokens were sampled,
`"client"` if the message was replayed in. Miles has a related distinction —
"session-generated assistant responses create checkpoints; client-injected
assistant messages remain prompt history" — but it is a property of the
checkpoint, not a label carried on every node into the export.

## Storage: turn snapshots vs. message deltas

Miles stores, per node, a full root-to-node token snapshot:

```python
# miles/rollout/session/v2/tree_trajectory.py
token_ids: list[int]          # full root->node snapshot
completion_span: tuple[int, int]   # this node's sampled completion within token_ids
```

One node per model generation. On a branchy tree the shared prefix is stored
once per branch, so total memory is O(nodes x depth).

We store one node per *message*, holding only that message's token delta, and a
shared prefix is stored once for all branches that pass through it. Node
granularity is finer than Miles's because we have per-token attribution — that
is what the whole mechanism buys us, beyond avoiding the diff.

It also changes where the data lives. Miles keeps sessions in the rollout
process. We persist the graph to PostgreSQL and object storage, so a trajectory
can outlive a proxy process; on a cache miss we reload from Postgres, and if
that fails the worst case is a new branch, never wrong tokens.

## Verification

Miles builds the sequence, then checks it against a canonical full re-render
with `TokenSeqComparator`, which segments both sequences on special tokens and
classifies each difference:

- `SPECIAL_TOKEN_COUNT`, `SPECIAL_TOKEN_TYPE`, `NON_ASSISTANT_TEXT` — must be
  zero, these are bugs.
- `ASSISTANT_TEXT` — "Expected and non-severe: assistant tokens are inherited
  directly from the pretokenized prefix across turns, so they may not match the
  chat template's canonical tokenization."

That last tolerance is correct and is the point of TITO — when the sampled
tokens and the canonical retokenization disagree, the canonical one is wrong.
But it means their verification cannot be exact, so it has to categorize.

We do not compare against a re-render, because we never re-render a committed
turn. `_validate` checks that the reused prefix length equals the bridge
transition's `prompt + completion` length exactly:

```python
# src/skyrl_capture/tito/trace.py
# This length check is the whole prefix check. The tokens behind it are the
# ones these nodes were committed from -- handed to the bridge and returned
# unchanged -- so agreeing on where the prefix ends is agreeing on the prefix.
```

Plus a 64-token boundary probe in the renderer (`_prefix_survived`), which
catches the one case the contract does not cover: `trim_to_turn_close` keeping
less than we handed it. A full token-by-token audit is available behind
`TOKENS_AUDIT_PREFIX=1` and is off by default because it is O(context) per turn.

Different philosophies. Theirs verifies the output; ours constrains the
construction and checks the one thing construction cannot guarantee. Theirs
catches more classes of bug at runtime. Ours is cheaper and refuses rather than
reports.

## Engine contract

| | Miles | Ours |
| --- | --- | --- |
| Engine | SGLang, and a specific branch — the code says "Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue" | Any endpoint implementing the token wire |
| Request | OpenAI chat completions with `input_ids` overridden, plus forced `logprobs=True`, `return_meta_info=True`, `no_stop_trim=False` | `prompt_token_ids`, `sampling_params`, `session_ids` |
| Response | `choice.meta_info.output_token_logprobs`, unzipped into IDs | `response_ids`, `response_logprobs`, `stop_reasons`, `rollout_expert_indices` |
| Adding a wire | Edit the session server | Register an upstream plugin (`upstream/plugins.py`) |
| Logprobs | Required | Required — a response without them is an error, not a partial success |
| MoE routed experts | `return_routed_experts`, for R3 replay | `rollout_expert_indices`, must cover prompt + completion or be rejected |
| Streaming | Fake: backend call is non-streaming, response re-rendered as one SSE chunk | Same: `build_stream_frames` synthesizes SSE from the completed response |

Both arrived at synthesized streaming for the same reason — TITO needs the whole
completion before it can attribute anything.

## Scope

These are not the same product, and a fair comparison has to say so.

Miles's TITO is a component of an RL trainer. It lives next to Megatron and
SGLang, `collect_samples` assembles training samples on the session server, and
the token path exists to feed the trainer in the same process tree.

Ours is a standalone capture service. The same proxy has a text mode for
ordinary observation, the graph is persisted and queryable, there is a UI, and
the training data comes out through an export (`token_samples`, carrying
`input_ids`, `loss_mask`, `rollout_logprobs`, `rollout_expert_indices`) that
loads without this service in the loop. Trajectory identity is a URL route
(`/route/<trajectory-id>/...`), not a session ID in the request body, so an
unmodified client points its base URL at us and needs no other change.

One consequence worth naming: because we are not in the trainer, a capture
failure has nowhere to escalate. So the tokens route commits to the graph
*before* responding, and if the commit fails the caller still gets its
completion — marked `x-capture-status: poisoned` — and the trajectory is
stopped. Miles can raise at the rollout layer and let the framework handle it.

## Summary

| | Miles | Ours |
| --- | --- | --- |
| Boundary mechanism | Diff two Jinja renders under a synthetic dummy prefix | Per-token message attribution from the renderer |
| Source of truth for format | Model's Jinja template (sometimes a patched copy) | Hand-coded renderer per model family |
| Model-specific surface | 15 TITO subclasses, 4 splice overrides, 9 fixed Jinja templates, per-family append-role whitelist | 17 renderers; no splice patches |
| Unsupported model | Works, best-effort | Falls back to Jinja: N renders per call, no bridging |
| Splice boundary | Patched by hand per family | Renderer trims to its own turn-close token |
| Client sends divergent history | Hard error | New branch |
| Node granularity | One model generation | One message |
| Prefix storage | Full snapshot per node | Delta per node, shared prefix stored once |
| Verification | Compare against canonical re-render, classify mismatches | Length identity against the bridge + 64-token boundary probe; full audit opt-in |
| Persistence | In rollout process | PostgreSQL + object store |
| Coupling | SGLang branch, Megatron, in-trainer | Any token wire, standalone proxy |
| Failure mode | Raise to the framework | Commit before responding; poison and stop the trajectory |

## Worth borrowing

Two things they have that we do not:

1. **A runtime comparator.** `TokenSeqComparator` classifies differences by
   segment type, so "the special tokens moved" and "an assistant string
   retokenized differently" are separate signals. Our `TOKENS_AUDIT_PREFIX` was
   all-or-nothing and told you only that something diverged. Their
   categorization would make a failure diagnosable instead of just detectable.

   **[BUILT]** `tito/compare.py`. The classes are ours rather than theirs --
   `length`, `scaffold`, `sampled`, `given` -- and every one is a bug, where
   Miles must tolerate its `ASSISTANT_TEXT` class because it compares against a
   canonical re-render. A report names the class, the node, the prompt offset
   and a decoded window either side; counts are on `/healthz` under
   `tokens.audit_failures`. `TOKENS_AUDIT_PREFIX=report` audits without
   refusing.

2. **Two verification layers, CPU and GPU.** They run a fast token-level check
   and a separate end-to-end check under real model inference. We have the fast
   layer. The second one catches the class of bug where the tokens are
   self-consistent and still wrong.

   **[BUILT]** `skyrl-capture verify`. It re-feeds a finished record's prompts
   to the engine greedily and compares the completions, which checks both that
   the prompt is the one the engine saw and that the engine accepts it at all.
   Run against a real model it is the GPU layer; run against the mock engine,
   whose completion is a deterministic function of its prompt, it is the same
   claim at CI cost, and it runs on every push. The GPU invocation is written
   down in [verification.md](../verification.md) and has not been executed.

One thing we should not borrow: the per-family append-role whitelist. It exists
because their render can drift depending on what role comes next, which is a
constraint the diff mechanism imposes. We do not have the constraint, so we
should not inherit the configuration surface that manages it.
