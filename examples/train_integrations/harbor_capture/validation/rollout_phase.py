"""A rollout phase, exactly as training runs one, minus the gradient step.

Same generator, same capture proxy, same `/skyrl/v1/generate` endpoint, same
`compose`. What is missing is only the optimizer: nothing here trains, and
nothing about how a rollout is produced is simplified for the sake of running
it standalone.

    python rollout_phase.py --tasks 2 --turns 12 --timeout 180

Needs a capture service already serving (see README) and
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in the environment for Harbor's sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import yaml

from examples.train_integrations.harbor_capture.harbor_generator import (  # noqa: E402
    HarborCaptureGenerator,
)
from skyrl.train.generators.base import TrajectoryID  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rollout")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1].parent / "harbor/harbor_trial_config/default.yaml"
TASKS = Path(os.environ.get("HARBOR_TASKS", "/home/ray/default/work_skyrl/work_icap/data/harbor"))
#: Where a run writes its summary. One file per run id.
RESULTS = Path(os.environ.get("CAPTURE_RESULTS", Path(__file__).resolve().parent / "results"))


class GeneratorConfig:
    """The generator settings this integration requires.

    Not a stand-in for SkyRL's config object: these are read by name and the
    generator refuses to build without the first two, so naming them here is
    the same contract a training run satisfies through Hydra.
    """

    step_wise_trajectories = True
    merge_stepwise_output = False
    use_cache_salt = False
    max_turns = 32


class EngineClient:
    """The part of the inference-engine client a rollout touches.

    A training run passes the real one, which also carries weight versions and
    releases router sessions. With no trainer there are no weight versions, and
    the session release is the one thing worth keeping honest -- a router that
    keeps a session per rollout leaks prefix-cache slots.
    """

    weight_version = None

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.released: list[str] = []

    def finish_session(self, session_id: str) -> None:
        self.released.append(session_id)


def trial_config(model: str, *, turns: int, timeout: int, summarize: bool) -> dict:
    with open(DEFAULT_CONFIG) as handle:
        config = yaml.safe_load(handle)
    agent = config.setdefault("agent", {})
    agent["override_timeout_sec"] = timeout
    kwargs = agent.setdefault("kwargs", {})
    kwargs["max_turns"] = turns
    # Capture owns token accounting; asking Harbor for it too is what forces
    # the sibling integration to ban summarization.
    kwargs.pop("collect_rollout_details", None)
    kwargs["enable_summarize"] = summarize
    config.setdefault("environment", {})["type"] = "modal"
    # LiteLLM's `hosted_vllm/` prefix takes exactly one slash, so the served
    # name cannot itself contain one -- a training run sets
    # `served_model_name` to a slashless label for the same reason. capture
    # sends its own configured model upstream regardless of what the client
    # asks for, so this name is only a routing label for LiteLLM.
    agent["model_name"] = f"hosted_vllm/{model}"
    config["trials_dir"] = "/tmp/icap-validation-trials"
    return config


def pick_tasks(count: int) -> list[str]:
    found = sorted(p for p in TASKS.iterdir() if (p / "task.toml").is_file())
    if not found:
        raise SystemExit(f"no harbor tasks under {TASKS}")
    return [str(p) for p in found[:count]]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=1, help="distinct prompts")
    parser.add_argument("--group-size", type=int, default=1, help="rollouts per prompt")
    parser.add_argument("--concurrency", type=int, default=16, help="sandboxes in flight")
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--summarize", action="store_true", help="let the agent compact its history")
    parser.add_argument("--capture", default=os.environ.get("CAPTURE_ENDPOINT", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="policy", help="slashless routing label for LiteLLM")
    parser.add_argument("--run-id", default=None, help="defaults to rollout-<short uuid>")
    parser.add_argument("--out", default=None, help="defaults to results/<run-id>.json")
    args = parser.parse_args()

    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set; Harbor cannot start a sandbox")

    prompts = pick_tasks(args.tasks)
    # One rollout per (prompt, repetition). The repetitions of a prompt are
    # its GRPO group: they share an `instance_id`, which is what the trainer
    # groups by when it normalises rewards.
    # One run id per invocation. Trajectory ids are unique already, but the
    # run is what a listing groups by -- and reusing one across invocations
    # makes two runs look like one, which is exactly the question a rollout
    # phase is inspected to answer.
    run_id = args.run_id or f"rollout-{uuid4().hex[:8]}"
    out = args.out or str(RESULTS / f"{run_id}.json")

    tasks, ids = [], []
    for prompt in prompts:
        for repetition in range(args.group_size):
            tasks.append(prompt)
            ids.append(TrajectoryID(instance_id=Path(prompt).name, repetition_id=repetition))

    logger.info(
        "run %s: %d prompts x %d group = %d rollouts, max_turns=%d, timeout=%ds, "
        "summarize=%s, concurrency=%d",
        run_id, len(prompts), args.group_size, len(tasks), args.turns, args.timeout,
        args.summarize, args.concurrency,
    )
    budget = len(tasks) * args.timeout / 3600 * 0.055
    logger.info("worst case ~%.0f sandbox-minutes, about $%.2f if every rollout "
                "runs to its timeout", len(tasks) * args.timeout / 60, budget)

    generator = HarborCaptureGenerator(
        generator_cfg=GeneratorConfig(),
        harbor_trial_config=trial_config(args.model, turns=args.turns,
                                         timeout=args.timeout, summarize=args.summarize),
        inference_engine_client=EngineClient(args.capture),
        capture_endpoint=args.capture,
        project="icap-validation",
        run_id=run_id,
    )

    started = time.perf_counter()
    # `generate` puts every trial in one TaskGroup, so without this every
    # rollout would ask Modal for a sandbox at the same moment and every turn
    # of all of them would queue behind one engine.
    gate = asyncio.Semaphore(args.concurrency)
    unlimited = generator._trial

    async def limited(*call, **named):
        async with gate:
            return await unlimited(*call, **named)

    generator._trial = limited

    output = await generator.generate(
        {
            "prompts": tasks,
            "trajectory_ids": ids,
            "env_classes": [None] * len(tasks),
            "env_extras": [None] * len(tasks),
        },
        disable_tqdm=False,
    )
    elapsed = time.perf_counter() - started

    from collections import Counter

    rows = len(output["response_ids"])
    trainable = sum(sum(mask) for mask in output["loss_masks"])
    live = sum(1 for mask in output["loss_masks"] if any(mask))
    stops = Counter(output["stop_reasons"] or [])
    rewards = list(output["rewards"])
    print(f"\n{len(tasks)} rollouts -> {rows} rows in {elapsed:.0f}s "
          f"({elapsed / max(1, len(tasks)):.1f}s per rollout)")
    print(f"  rows with gradient : {live}/{rows}")
    print(f"  trainable tokens   : {trainable:,}")
    print(f"  stop reasons       : {dict(stops)}")
    print(f"  reward mean/max    : {sum(rewards) / max(1, len(rewards)):.3f} / {max(rewards):.3f}")
    print(f"  sessions released  : {len(generator.inference_engine_client.released)}")
    # A group with no reward spread teaches GRPO nothing; worth seeing.
    groups = {}
    for identifier, reward in zip(output["trajectory_ids"], rewards):
        groups.setdefault(identifier.instance_id, []).append(reward)
    varied = sum(1 for values in groups.values() if len(set(values)) > 1)
    print(f"  groups with spread : {varied}/{len(groups)}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "run_id": run_id,
        "tasks": tasks,
        "seconds": elapsed,
        "rows": rows,
        "trainable_tokens": trainable,
        "rewards": list(output["rewards"]),
        "stop_reasons": list(output["stop_reasons"] or []),
        "is_last_step": list(output["is_last_step"] or []),
        "prompt_lengths": [len(p) for p in output["prompt_token_ids"]],
        "response_lengths": [len(r) for r in output["response_ids"]],
    }, indent=2))
    print(f"\nrun_id {run_id}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
