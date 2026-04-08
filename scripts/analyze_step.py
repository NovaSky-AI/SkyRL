#!/usr/bin/env python3
"""Pretty-print trajectory step files produced by the skyrl_gym_generator __main__ entrypoint.

Supports two file naming conventions:
  Old format:  step_1.json, step_2.json, ...
  New format:  depth-0_step_1.json, depth-1_child-0_step_1.json, ...
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def c(text: str, *styles: str) -> str:
    prefix = "".join(COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{COLORS['reset']}"


def hr(char="─", width=88) -> str:
    return c(char * width, "dim")


def section_header(title: str) -> str:
    pad = 2
    inner = f" {title} "
    side = (88 - len(inner) - 2 * pad) // 2
    line = "─" * side
    return c(f"{'─' * pad}{line}┤ ", "dim") + c(title, "bold", "cyan") + c(f" ├{line}{'─' * pad}", "dim")


def parse_chatml_turns(text: str) -> list[dict]:
    """Split ChatML-formatted text into turns with role and content."""
    turns = []
    parts = text.split("<|im_start|>")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        role_end = part.find("\n")
        if role_end == -1:
            role = part.rstrip("<|im_end|>").strip()
            content = ""
        else:
            role = part[:role_end].strip()
            content = part[role_end + 1:]
        content = content.replace("<|im_end|>", "").strip()
        turns.append({"role": role, "content": content})
    return turns


def format_role(role: str) -> str:
    role_styles = {
        "system": ("magenta", "bold"),
        "user": ("green", "bold"),
        "assistant": ("blue", "bold"),
    }
    styles = role_styles.get(role, ("white", "bold"))
    return c(f"  [{role.upper()}]", *styles)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = (max_chars - 40) // 2
    return text[:half] + c(f"\n  ... ({len(text) - max_chars:,} chars truncated) ...\n", "dim", "yellow") + text[-half:]


def load_step(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "step": data["step"],
        "depth": data.get("depth", 0),
        "child_index": data.get("child_index"),
        "is_last": data["is_last"],
        "stop_reason": data["stop_reason"],
        "prompt_text": data["prompt_text"],
        "response_text": data["response_text"],
        "latency": data.get("latency", {}),
    }


_DEPTH_CHILD_RE = re.compile(r"depth-(\d+)_(?:child-(\d+)_)?step_(\d+)\.json")
_LEGACY_RE = re.compile(r"step_(\d+)\.json")


def _step_sort_key(step: dict) -> tuple:
    """Sort steps by (depth, child_index, step_number)."""
    ci = step["child_index"] if step["child_index"] is not None else -1
    return (step["depth"], ci, step["step"])


def load_trajectory_dir(traj_dir: Path) -> tuple[list[dict], dict[int, dict[int | None, list[dict]]], dict | None]:
    """Load all step files from a trajectory directory.

    Returns:
        parent_steps: flat list of depth-0 steps (sorted by step number)
        children: dict mapping depth -> {child_index -> sorted list of steps}
        metadata: optional metadata dict
    """
    all_steps: list[dict] = []

    for f in traj_dir.iterdir():
        if not f.suffix == ".json" or f.name == "metadata.json":
            continue
        m = _DEPTH_CHILD_RE.match(f.name)
        if m:
            all_steps.append(load_step(f))
            continue
        m = _LEGACY_RE.match(f.name)
        if m:
            all_steps.append(load_step(f))

    all_steps.sort(key=_step_sort_key)

    parent_steps = [s for s in all_steps if s["depth"] == 0]
    children: dict[int, dict[int | None, list[dict]]] = {}
    for s in all_steps:
        if s["depth"] == 0:
            continue
        depth = s["depth"]
        ci = s["child_index"]
        children.setdefault(depth, {}).setdefault(ci, []).append(s)

    metadata = None
    meta_path = traj_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return parent_steps, children, metadata


def print_step(step: dict, max_prompt_chars: int, max_user_chars: int, max_response_chars: int, show_full_prompt: bool):
    """Print a single step file nicely."""
    prompt_text = step["prompt_text"]
    response_text = step["response_text"]
    latency = step["latency"]
    depth = step["depth"]
    child_index = step["child_index"]

    if child_index is not None:
        title = f"DEPTH {depth} / CHILD {child_index} / STEP {step['step']}"
        box_color = "yellow"
    else:
        title = f"STEP {step['step']}"
        box_color = "cyan"
    if step["is_last"]:
        title += "  (FINAL)"

    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", box_color))
    print(c(f"  ║{'':^86}║", "bold", box_color))
    print(c("  ║", "bold", box_color) + c(f"{title:^86}", "bold", "white") + c("║", "bold", box_color))
    print(c(f"  ║{'':^86}║", "bold", box_color))
    print(c(f"  ╚{'═' * 86}╝", "bold", box_color))
    print()

    print(section_header("Metadata"))
    print()

    sr = step["stop_reason"]
    sr_colored = c(sr, "green" if sr == "stop" else "yellow", "bold")

    meta_rows = [
        ("Depth", str(depth)),
    ]
    if child_index is not None:
        meta_rows.append(("Child Index", str(child_index)))
    meta_rows += [
        ("Step", str(step["step"])),
        ("Is Last", c("YES", "bold", "green") if step["is_last"] else c("no", "dim")),
        ("Stop Reason", sr_colored),
        ("Prompt Length", f"{len(prompt_text):,} chars"),
        ("Response Length", f"{len(response_text):,} chars"),
    ]
    if latency:
        inf_s = latency.get("inference_s")
        env_s = latency.get("env_step_s")
        if inf_s is not None:
            meta_rows.append(("Inference Time", f"{inf_s:.1f}s"))
        if env_s is not None:
            meta_rows.append(("Env Step Time", f"{env_s:.1f}s"))

    for label, value in meta_rows:
        print(f"  {c(label + ':', 'bold'):>38s}  {value}")
    print()

    combined = prompt_text + response_text
    turns = parse_chatml_turns(combined)

    if show_full_prompt:
        print(section_header("Full Conversation"))
        print()
        for turn in turns:
            print(format_role(turn["role"]))
            content = turn["content"]
            if turn["role"] == "user":
                content = truncate(content, max_user_chars)
            for line in content.split("\n"):
                print(f"    {line}")
            print()
    else:
        prompt_turns = parse_chatml_turns(prompt_text)
        response_turns = parse_chatml_turns(response_text)

        print(section_header("Prompt"))
        print()
        for turn in prompt_turns:
            print(format_role(turn["role"]))
            content = turn["content"]
            if turn["role"] == "system":
                content = truncate(content, max_prompt_chars)
            elif turn["role"] == "user":
                content = truncate(content, max_user_chars)
            for line in content.split("\n"):
                print(f"    {line}")
            print()

        print(section_header("Response"))
        print()
        if response_turns:
            for turn in response_turns:
                print(format_role(turn["role"]))
                content = turn["content"]
                if turn["role"] == "user":
                    content = truncate(content, max_user_chars)
                else:
                    content = truncate(content, max_response_chars)
                for line in content.split("\n"):
                    print(f"    {line}")
                print()
        else:
            content = truncate(response_text, max_response_chars)
            for line in content.split("\n"):
                print(f"    {line}")
            print()

    print(hr())
    print()


def _print_steps_table(label: str, steps: list[dict], box_color: str = "cyan"):
    """Print a compact table of steps."""
    print(c(f"  {label}", "bold", box_color))
    print(f"    {'Step':>6}  {'Prompt':>14}  {'Response':>14}  {'Stop':>8}  {'Last':>6}")
    print(f"    {'─' * 6}  {'─' * 14}  {'─' * 14}  {'─' * 8}  {'─' * 6}")
    for step in steps:
        sr = step["stop_reason"]
        sr_colored = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
        is_last_str = c("YES", "bold", "green") if step["is_last"] else ""
        print(
            f"    {step['step']:>6}"
            f"  {len(step['prompt_text']):>8} chr"
            f"  {len(step['response_text']):>8} chr"
            f"  {sr_colored:>19}"
            f"  {is_last_str}"
        )
    print()


def print_trajectory_overview(
    parent_steps: list[dict],
    children: dict[int, dict[int | None, list[dict]]],
    metadata: dict | None,
):
    """Print a compact overview of all steps in a trajectory."""
    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c("  ║", "bold", "cyan") + c(f"{'TRAJECTORY OVERVIEW':^86}", "bold", "white") + c("║", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ╚{'═' * 86}╝", "bold", "cyan"))
    print()

    if metadata:
        print(section_header("Rollout Metrics"))
        print()
        key_metrics = [
            ("Reward", f"{metadata.get('environment/reward', metadata.get('reward', 0)):.4f}"),
            ("Turns Used", str(int(metadata.get("environment/turns_used", metadata.get("turns_used", 0))))),
            ("Final Value Set", c("YES", "bold", "green") if metadata.get("environment/final_value_set", metadata.get("final_value_set", 0)) > 0 else c("NO", "bold", "red")),
        ]
        if "generate/avg_num_tokens" in metadata:
            key_metrics += [
                ("Avg Response Tokens", f"{metadata.get('generate/avg_num_tokens', 0):,.0f}"),
                ("Min / Max Tokens", f"{metadata.get('generate/min_num_tokens', 0):,} / {metadata.get('generate/max_num_tokens', 0):,}"),
                ("Avg Assistant Tokens", f"{metadata.get('generate/avg_assistant_tokens', 0):,.0f}"),
            ]
        for label, value in key_metrics:
            print(f"  {c(label + ':', 'bold'):>38s}  {value}")
        print()

    n_children = sum(len(by_ci) for by_ci in children.values())

    print(section_header(f"Parent (depth 0) — {len(parent_steps)} steps"))
    print()
    _print_steps_table("Parent Agent", parent_steps)

    if children:
        for depth in sorted(children.keys()):
            by_ci = children[depth]
            print(section_header(f"Children (depth {depth}) — {len(by_ci)} child agent(s)"))
            print()
            for ci in sorted(by_ci.keys(), key=lambda x: x if x is not None else -1):
                child_steps = by_ci[ci]
                label = f"Child {ci}" if ci is not None else "Child"
                total_resp = sum(len(s["response_text"]) for s in child_steps)
                _print_steps_table(f"{label}  ({len(child_steps)} steps, {total_resp:,} resp chars)", child_steps, box_color="yellow")
    else:
        print(c("  No child rollouts found.", "dim"))
        print()

    print(hr())
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print trajectory step files from the skyrl_gym_generator entrypoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Print a single step file:
              python scripts/analyze_step.py trajectories/mp_trajectory_1/depth-0_step_1.json

              # Print all steps in a trajectory directory:
              python scripts/analyze_step.py trajectories/mp_trajectory_1/

              # Overview table only (no full conversation):
              python scripts/analyze_step.py trajectories/mp_trajectory_1/ --overview

              # Show specific parent steps from a directory:
              python scripts/analyze_step.py trajectories/mp_trajectory_1/ --steps 1 3

              # Show child rollouts only (all depths > 0):
              python scripts/analyze_step.py trajectories/mp_trajectory_1/ --children

              # Show a specific child's steps:
              python scripts/analyze_step.py trajectories/mp_trajectory_1/ --children --child-idx 0

              # Show full prompt without truncation:
              python scripts/analyze_step.py trajectories/mp_trajectory_1/depth-0_step_1.json --full
        """),
    )
    parser.add_argument("path", type=Path, help="Path to a step JSON file or a trajectory directory")
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="Which parent step numbers to display (default: all)")
    parser.add_argument("--children", action="store_true", help="Show child rollouts instead of parent steps")
    parser.add_argument("--child-idx", type=int, default=None, help="Filter to a specific child index (use with --children)")
    parser.add_argument("--max-prompt-chars", type=int, default=2000, help="Max chars for system prompt (default: 2000)")
    parser.add_argument("--max-user-chars", type=int, default=500, help="Max chars for user messages (default: 500)")
    parser.add_argument("--max-response-chars", type=int, default=50000, help="Max chars per response turn (default: 50000)")
    parser.add_argument("--full", action="store_true", help="Show full prompt without truncation")
    parser.add_argument("--overview", action="store_true", help="Only show the trajectory overview table")
    args = parser.parse_args()

    target = args.path

    if target.is_dir():
        parent_steps, children, metadata = load_trajectory_dir(target)
        total = len(parent_steps) + sum(len(s) for by_ci in children.values() for s in by_ci.values())
        if total == 0:
            print(c(f"  No step JSON files found in {target}", "red", "bold"))
            return

        n_children = sum(len(by_ci) for by_ci in children.values())
        print()
        parts = [f"{len(parent_steps)} parent steps"]
        if n_children:
            parts.append(f"{n_children} child rollout(s)")
        print(c(f"  Loaded {', '.join(parts)} from ", "dim") + c(str(target), "bold", "underline"))
        print()

        print_trajectory_overview(parent_steps, children, metadata)

        if not args.overview:
            if args.children:
                for depth in sorted(children.keys()):
                    for ci in sorted(children[depth].keys(), key=lambda x: x if x is not None else -1):
                        if args.child_idx is not None and ci != args.child_idx:
                            continue
                        for step in children[depth][ci]:
                            print_step(step, args.max_prompt_chars, args.max_user_chars, args.max_response_chars, args.full)
            else:
                step_filter = set(args.steps) if args.steps else None
                for step in parent_steps:
                    if step_filter and step["step"] not in step_filter:
                        continue
                    print_step(step, args.max_prompt_chars, args.max_user_chars, args.max_response_chars, args.full)

    elif target.is_file():
        step = load_step(target)
        depth = step["depth"]
        ci = step["child_index"]
        label = f"depth {depth}"
        if ci is not None:
            label += f", child {ci}"
        label += f", step {step['step']}"
        print()
        print(c(f"  Loaded ({label}) from ", "dim") + c(str(target), "bold", "underline"))
        print_step(step, args.max_prompt_chars, args.max_user_chars, args.max_response_chars, args.full)

    else:
        print(c(f"  Error: path not found: {target}", "red", "bold"))


if __name__ == "__main__":
    main()
