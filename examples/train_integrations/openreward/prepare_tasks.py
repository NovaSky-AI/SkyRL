"""Fetch tasks from OpenReward environments and build a SkyRL-compatible Parquet dataset.

For each task:
  1. Creates a temporary session to get the initial prompt and available tools
  2. Formats tool specs into a system prompt
  3. Saves as a Parquet row with columns: prompt, env_class, env_name, split, task_index

Usage:
    python prepare_tasks.py \
        --env "GeneralReasoning/WhoDunit" \
        --split train \
        --output tasks.parquet \
        [--max-tasks 100]
    
    # Multiple environments:
    python prepare_tasks.py \
        --env "GeneralReasoning/WhoDunit" \
        --env "GeneralReasoning/CTF" \
        --split train \
        --output tasks.parquet
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Any

from openreward import AsyncOpenReward

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = (
    "You are an agent that takes actions in a stateful environment to achieve a goal.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'
)


def format_tool_spec(spec: Any) -> str:
    """Format a single ToolSpec as JSON for embedding in the system prompt."""
    return json.dumps(
        {
            "name": spec.name,
            "arguments_schema": spec.input_schema,
            "description": spec.description,
        }
    )


async def fetch_tasks_for_env(
    client: AsyncOpenReward,
    env_name: str,
    split: str,
    max_tasks: int | None = None,
) -> list[dict]:
    """Fetch tasks from one environment, create temp sessions to get prompts + tools."""
    env = client.environments.get(name=env_name)

    # Get task count
    num_tasks = await env.num_tasks(split)
    effective = min(max_tasks, num_tasks) if max_tasks is not None else num_tasks
    logger.info(f"{env_name} [{split}]: {num_tasks} tasks total, fetching {effective}")

    rows = []
    for idx in range(effective):
        try:
            async with env.session(split=split, index=idx) as session:
                # Get initial prompt and tools
                prompt_blocks = await session.get_prompt()
                tools = await session.list_tools()

            user_text = "".join(b.text for b in prompt_blocks if b.type == "text")
            tools_str = "\n".join(format_tool_spec(t) for t in tools)
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_str)

            row = {
                "prompt": json.dumps(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ]
                ),
                "env_class": "openreward",
                "env_name": env_name,
                "split": split,
                "task_index": idx,
            }
            rows.append(row)

            if (idx + 1) % 10 == 0:
                logger.info(f"  fetched {idx + 1}/{effective}")

        except Exception as e:
            logger.warning(f"  failed to fetch task {idx} from {env_name}: {e}")
            continue

    logger.info(f"  {env_name}: {len(rows)} rows prepared")
    return rows


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenReward()
    all_rows = []

    for env_name in args.env:
        rows = await fetch_tasks_for_env(
            client=client,
            env_name=env_name,
            split=args.split,
            max_tasks=args.max_tasks,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No tasks fetched. Check environment names and API key.")
        return

    # Write output
    output = args.output
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if output.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "prompt": [r["prompt"] for r in all_rows],
                "env_class": [r["env_class"] for r in all_rows],
                "env_name": [r["env_name"] for r in all_rows],
                "split": [r["split"] for r in all_rows],
                "task_index": [r["task_index"] for r in all_rows],
            }
        )
        pq.write_table(table, output)
    else:
        # JSONL fallback
        with open(output, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")

    logger.info(f"Wrote {len(all_rows)} rows to {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenReward tasks for SkyRL training")
    parser.add_argument("--env", action="append", required=True, help="OpenReward environment name (repeatable)")
    parser.add_argument("--split", default="train", help="Task split (default: train)")
    parser.add_argument("--output", default="tasks.parquet", help="Output file (.parquet or .jsonl)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per environment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
