# Development tooling

Workflows for people changing capture, kept out of the `skyrl-capture` package
because they are not part of the product loop
(`serve` → `run` → `view`/`annotate` → `export`). Nothing here is installed by
`pip install skyrl-capture`; all of it is run from a checkout.

```
tools/
  mock_server.py     a runnable mock inference server
  bench/             capture-on against capture-off, through the same proxy
  verify/            re-feed a record's prompts to the engine
```

| Command | What it does |
| --- | --- |
| `python -m tools.verify.cli` | Re-feeds a finished record's prompts to the engine and compares the completions. The second [verification layer](../docs/verification.md). |
| `python -m tools.bench.cli` | Compares capture-on against capture-off through the same proxy. See [benchmarks.md](../docs/benchmarks.md). |
| `python -m tools.mock_server` | A mock upstream to run any of this against, and what the quickstart points at. |

```bash
uv run python -m tools.verify.cli --record ./traces --run-id run-a \
  --engine-url http://127.0.0.1:8000/generate --model policy

uv run python -m tools.bench.cli --requests 4000 --concurrency 8 --processes 3
```

`--help` works on each. The fixture generator `scripts/verify_e2e.py` captures a
short tokens-mode run for the verifier to check.

The mock upstream itself lives in `tests/support/`, with the fixtures that are
its main consumer; `tools/mock_server.py` is the uvicorn wrapper that makes it
a command. Neither is importable from the installed package, which is the
point: what ships should describe the product rather than the laboratory it
was built in.
