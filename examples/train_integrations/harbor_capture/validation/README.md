# Validation

End-to-end checks for this integration, on the path a training run uses,
without taking a gradient step. Unlike `tests/integrations/harbor_capture/`,
these need a live engine, a capture service and Harbor sandboxes -- so they
are run by hand, not in CI.

## The claim

> every token the trainer trains on is a token the engine sampled from

Stated in absolute terms, not against the sibling integration. The two build
different prompts from turn two onward and each is faithful to its own run, so
a bit-for-bit comparison between them measures nothing.

## Scripts

| | |
|---|---|
| `rollout_phase.py` | a rollout phase: prompts x group, through the generator and `compose` |
| `token_fidelity.py` | trained tokens == tokens the engine sampled from |
| `validate_compose.py` | the batch `compose` hands the trainer holds together |
| `classify_parity.py` | capture's rendering vs the engine's, in three buckets |
| `render_parity.py` | the same over every captured exchange |

## Running

Engine -- SkyRL's own server, so the rollout path is the training path:

    VLLM_BATCH_INVARIANT=1 .venv/bin/python -m \
      skyrl.backends.skyrl_train.inference_servers.vllm_server_actor \
      --model Qwen/Qwen3-4B-Instruct-2507 --tensor-parallel-size 1 \
      --host 127.0.0.1 --port 9500 --max-model-len 65536 \
      --gpu-memory-utilization 0.85 --enable-auto-tool-choice \
      --tool-call-parser hermes

Batch invariance makes sampled tokens independent of how requests batch
together, so a difference between two runs means a real difference rather than
a scheduler artefact. It uses slower kernels, so **performance numbers need a
run with it off**.

Capture, from SkyRL's environment -- the upstream plugin imports SkyRL's
`generate_wire`, so it cannot load from capture's own venv:

    PYTHONPATH=$PWD uv run --with-editable /path/to/inference-capture \
      skyrl-capture serve --host 0.0.0.0 --port 8080 --mode tokens \
      --upstream-type skyrl --upstream-url http://127.0.0.1:9500 \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 --max-model-len 65536 \
      --record-dir /tmp/harbor-capture \
      --upstream-module examples.train_integrations.harbor_capture.upstream

Rollouts (`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` for Harbor's sandboxes,
`HARBOR_TASKS` for the task directory):

    python validation/rollout_phase.py --tasks 4 --group-size 8 --turns 8 \
      --timeout 180 --concurrency 16

Then:

    python validation/token_fidelity.py /tmp/harbor-capture
    python validation/validate_compose.py /tmp/harbor-capture <run-id>

## Results, 2026-09-22

Qwen3-4B, code_contests via Terminus-2, across two runs (256 and 32 rollouts):

    312 trajectories, 856 sampled turns
      trained tokens == tokens the engine sampled from : 856/856
      compose structural checks                        : 0 failures

    rollout-93aef74a: 32 rollouts -> 32 rows in 181s
      rows with gradient : 32/32
      stop reasons       : complete 32
      sessions released  : 32
      groups with spread : 3/4

`sessions released` equal to the rollout count is what says a timeout is
masked rather than retried; an earlier run released 278 for 256 because a
finish error was replacing the `TimeoutError` behind it.

`groups with spread` is what says a task set would train: a group whose
rewards are all equal gives GRPO no signal.

## What this does not cover

* **No tool-schema traffic.** Terminus-2 drives the shell through its own text
  protocol, so `tools` is empty on every turn. Two rendering divergences found
  separately -- the OpenAI `{"type": "function", ...}` wrapper, and key order
  from capture's canonicalisation -- are covered only by fixtures.
  `../templates/qwen3_renderer_aligned.jinja` aligns the engine to capture on
  both, for experiments that need the confound removed. It is not a production
  setting: in tokens mode capture renders the prompt itself, so the engine's
  chat template is not used on this path at all.
* **No branching.** `--summarize` has not been run, so compaction -- the thing
  the sibling structurally cannot represent -- is untested end to end.
* **The finish sweep has not fired in a live run.** It was verified against 26
  real orphans by hand; the runs since produced no timeouts, so it had nothing
  to do. A short `--timeout` would force the case.
