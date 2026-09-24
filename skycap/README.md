# skycap

Trajectory capture for RL rollouts. A harness points its unchanged OpenAI client
at a per-trajectory URL. skycap records every model call into a context graph —
one node per message, where resamples, subagents, compaction and harness edits
are forks — and returns one training sample per root-to-leaf path.

skycap is its own package inside this repository and does not depend on
`skyrl`. It is being landed in layers; see the PRs titled `[skycap][k/N]`.

## Develop

```bash
cd skycap
uv sync
uv run pytest
```

Formatting and lint are the repository's (`bash format.sh` from the root).
