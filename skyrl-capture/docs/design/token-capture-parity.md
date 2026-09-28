# Token-capture parity with SkyRL

SkyRL calls this **TITO** (token-in/token-out). That name appears below only
where it refers to their implementation; ours is `tokens` mode throughout.

The PRD's Phase 4 describes token capture at a level that omits several
requirements the reference implementation in
[SkyRL PR #2143](https://github.com/NovaSky-AI/SkyRL/pull/2143) treats as
load-bearing. Because the entire value of token capture is that the stored
tokens are *exactly* what inference saw, a near-miss here silently corrupts
training data rather than producing a visible error.

This document is the checklist that `skyrl_capture.tito` is built against.
Each row states the requirement, why it matters, and where it is implemented.

## Requirements the PRD does not state

| # | Requirement | Why it matters | Implementation |
| --- | --- | --- | --- |
| 1 | The renderer must return **per-token message attribution** (`message_indices`), with `-1` for template scaffold tokens. | Without it there is no way to split a rendered prompt into per-message node deltas. This is the mechanism that makes one-node-per-message possible in token space. | `tito/renderer.py`, `RenderedPrompt.message_indices` |
| 2 | Leading scaffold tokens belong to the **following** message; trailing scaffold after the last message is the **assistant generation prompt**. | Misattribution shifts token boundaries and breaks prefix reuse on the next turn. | `tito/trace.py::attribute_prompt_tokens` |
| 3 | An assistant node stores a `sampled_start` boundary: scaffold tokens first, then the sampled completion. | The assistant node is not purely sampled. Training must mask the scaffold; `prompt_token_ids` for the call is *prior nodes + scaffold*. | `tito/trace.py::TokenNode.sampled_start` |
| 4 | Per-token `sampled_mask` and `logprobs`, with `0.0` logprobs on non-sampled positions. | Gives the loss mask and rollout logprobs directly, with no re-alignment at export time. | `tito/trace.py::TokenNode.__init__` |
| 5 | Optional per-token **routed expert IDs** (MoE), covering prompt *and* completion. | Required for MoE training signal; must align to the full sequence or be rejected. | `tito/trace.py`, `routed_experts` |
| 6 | `tools` participate in prefix identity via a **`tools_hash`**, and a prefix may only be reused when the tools hash matches. | Chat templates render tool schemas into the prompt, so the same messages with different tools are a different token sequence. | `tito/trace.py::prepare_turn` |
| 7 | Prefix reuse must anchor on a **bridge transition**: a completed model call whose assistant node terminates it, with a matching tools hash. | Reuse must resume from a real inference boundary, not an arbitrary message node. | `tito/trace.py::find_bridge_transition` |
| 8 | `reused_prefix_length` must equal the bridge's `prompt + completion` length exactly, and the re-rendered prefix must be byte-identical. A mismatch forces a **full render and a new branch**. | This is the fidelity check that prevents mixing tokenizations. | `tito/trace.py::TokenTrace._validate` |
| 9 | Two-stage prefix matching: match in **message space** first, then tighten using the **actual rendered prompt token IDs**. | Identical messages can have different tokenizations; only the token-exact match may be reused. | `tito/trace.py::longest_message_prefix`, `longest_exact_prefix` |
| 10 | Turns within a trajectory are **serialized**, and a prepared turn carries a graph revision that is rejected if stale. | Concurrent commits to one trace would interleave prefixes and corrupt attribution. | `tito/trace.py` revision check, per-trajectory turn lock |
| 11 | Committing a message that already exists as an identical child (message **and** tokens **and** `sampled_start`) reuses that node instead of duplicating it. | Makes retries idempotent without creating phantom forks. | `tito/trace.py::find_exact_child` |
| 12 | On export, each sampled node is trainable **at most once** across branches. | Forks share assistant nodes; training them once per branch double-counts the same sampled tokens. | `export/formats.py` |
| 13 | `max_tokens` is clamped against `max_model_len - prompt_tokens`, and the renderer's stop token IDs are passed to inference. | Prevents context overflow and wrong stop behaviour for the model's own format. | `tito/engine.py::build_sampling_params` |
| 14 | Per-turn `stop_reason`, `model`, and canonicalized `sampling_params` are recorded. | Needed for overlong filtering, error masking, and reproducibility. | `tito/trace.py::Transition` |

## Deliberate divergences

* **Fail-closed capture.** The PRD requires capture to be fail-open: a capture
  failure must never fail a successful inference response. That is right for
  text-mode *observation*. It is wrong for token capture, where the captured token
  deltas are the authoritative training record and the proxy itself produced
  them. So the tokens route commits before responding. When that commit fails the
  caller still gets its completion -- the engine produced it, and an attribution
  failure is not an inference failure -- marked with `x-capture-status: poisoned`,
  and the trajectory is stopped. Stopping it is the point: the failed turn is not
  in the graph, so a next turn would render its assistant message from the
  client's replayed history and record model-generated tokens as `author: client`.
  Text-mode durability stays fail-open in both modes.
* **Persistence.** SkyRL keeps the trace in process memory for the lifetime of
  one trial. Here the graph is persisted to PostgreSQL and object storage, so
  node identity is a string ID rather than a list index, and commits are
  idempotent by `(trajectory, parent, delta_hash)`.
* **Composer scope.** SkyRL's composer emits its own `GeneratorOutput`. The
  equivalent here is the `token_samples` export. It carries the same values
  (`loss_mask`, `rollout_logprobs`, `rollout_expert_indices`) plus the
  trajectory and branch identifiers, so it can be loaded without this service
  in the loop. It differs in one way deliberately: rather than a
  `prompt`/`response` pair it emits one `input_ids` sequence with the mask over
  it, because on a multi-turn path everything after the first generation is
  "response", later user and tool messages included, and the split then names
  nothing a consumer can act on.
