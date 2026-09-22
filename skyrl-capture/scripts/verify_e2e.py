"""Capture a short tokens-mode run into a record, for `tools/verify/`.

The setup half of the end-to-end verification layer: it produces the artifact
the check runs against. Kept as a script rather than folded into the command
because verification's input is a *finished* record, and making one is a
separate job from checking one.

    python -m tools.mock_server --port 9199 &
    python scripts/verify_e2e.py
    uv run python -m tools.verify.cli --record ./ci-traces --run-id ci \
      --engine-url http://127.0.0.1:9199/generate --model ci-policy

Against a real engine, point `ENGINE_URL` and `TOKENIZER` at it; the rest is
the same. See docs/verification.md.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("LANG", "C.utf8")
os.environ.setdefault("LC_ALL", "C.utf8")

from skyrl_capture.config import TitoUpstream, load_config  # noqa: E402
from skyrl_capture.sdk import CaptureClient, create_trajectory  # noqa: E402
from skyrl_capture.service import CaptureService  # noqa: E402

ENGINE_URL = os.environ.get("ENGINE_URL", "http://127.0.0.1:9199/generate")
TOKENIZER = os.environ.get("TOKENIZER", "builtin")
RECORD = Path(os.environ.get("RECORD_DIR", "./ci-traces"))
TASKS = ("add-index", "fix-migration")


def turn(trajectory, messages: list[dict], *, max_tokens: int = 8) -> dict:
    body = json.dumps({"model": "ci", "messages": messages, "max_tokens": max_tokens}).encode()
    request = urllib.request.Request(
        f"{trajectory.base_url}/chat/completions",
        data=body,
        # Capture authenticates nothing on the way in; the placeholder is
        # here because an OpenAI-shaped client always sends one.
        headers={"authorization": "Bearer unused", "content-type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]


def main() -> int:
    config = load_config().with_overrides(
        upstream=TitoUpstream(
            type="tokens", url=ENGINE_URL, model="ci-policy",
            tokenizer=TOKENIZER, max_model_len=8192,
        ),
        record_dir=RECORD,
    )
    service = CaptureService(config=config, port=8166)
    service.start(blocking=False)
    client = CaptureClient(service.base_url)
    try:
        for step, task in enumerate(TASKS):
            trajectory = create_trajectory(
                project="ci", run_id="ci", task_id=task, step=step, client=client
            )
            history = [{"role": "user", "content": f"Work on {task}."}]
            reply = turn(trajectory, history)
            history.append({key: value for key, value in reply.items() if value is not None})
            history.append({"role": "user", "content": "Summarise."})
            turn(trajectory, history)
            trajectory.finish(annotations={"reward": 1.0})
    finally:
        client.close()
        service.stop()
    import asyncio

    from skyrl_capture.reader.records import RecordReader

    async def committed() -> list[str]:
        reader = RecordReader(RECORD)
        await reader.refresh()
        return await reader.finished_trajectory_ids(project="ci", run_id=None)

    written = asyncio.run(committed())
    print(f"captured {len(written)} trajectories into {RECORD}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
