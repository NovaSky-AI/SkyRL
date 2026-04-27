#!/usr/bin/env python3
"""
Step through RLM eval JSONL examples turn by turn.

Usage:
    python step_through_eval.py <path_to_jsonl>

Keys:
    ENTER / SPACE  - next turn
    n              - next example
    p              - previous example
    q              - quit
"""

import json
import re
import sys
import termios
import tty
from pathlib import Path


ROLE_COLORS = {
    "system":    "\033[33m",   # yellow
    "user":      "\033[36m",   # cyan
    "assistant": "\033[32m",   # green
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
MAGENTA = "\033[35m"


def getch() -> str:
    """Read a single character from stdin without echoing."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def parse_chatml(text: str) -> list[dict]:
    """Parse a ChatML-formatted string into a list of {role, content} dicts."""
    turns = []
    # Split on <|im_start|>
    parts = re.split(r"<\|im_start\|>", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Remove trailing <|im_end|>
        part = re.sub(r"<\|im_end\|>\s*$", "", part)
        # First line is the role
        lines = part.split("\n", 1)
        role = lines[0].strip()
        content = lines[1] if len(lines) > 1 else ""
        turns.append({"role": role, "content": content})
    return turns


def load_examples(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def fmt_header(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def fmt_role(role: str) -> str:
    color = ROLE_COLORS.get(role, "")
    return f"{color}{BOLD}[{role.upper()}]{RESET}"


def clear_screen():
    print("\033[2J\033[H", end="", flush=True)


def print_turn(turn_idx: int, total_turns: int, turn: dict, example_idx: int, total_examples: int):
    clear_screen()
    print(fmt_header(f"Example {example_idx + 1}/{total_examples}  |  Turn {turn_idx + 1}/{total_turns}"))
    print(DIM + "─" * 80 + RESET)
    print(f"{fmt_role(turn['role'])}")
    print()
    content = turn["content"].strip()
    # Truncate very long content with a note
    max_chars = 3000
    if len(content) > max_chars:
        print(content[:max_chars])
        print(DIM + f"\n... [{len(content) - max_chars} more chars truncated] ..." + RESET)
    else:
        print(content)
    print()
    print(DIM + "─" * 80 + RESET)
    print(DIM + "ENTER/SPACE=next turn  n=next example  p=prev example  q=quit" + RESET)


def print_final(example_idx: int, total_examples: int, example: dict, output_response: str, total_turns: int = 0, turn_idx: int = 0):
    clear_screen()
    turn_label = f"Turn {turn_idx + 1}/{total_turns}  |  " if total_turns else ""
    print(fmt_header(f"Example {example_idx + 1}/{total_examples}  |  {turn_label}LAST TURN"))
    print(DIM + "─" * 80 + RESET)
    print(f"{ROLE_COLORS['assistant']}{BOLD}[ASSISTANT — LAST TURN]{RESET}")
    print()
    content = output_response.strip()
    max_chars = 3000
    if len(content) > max_chars:
        print(content[:max_chars])
        print(DIM + f"\n... [{len(content) - max_chars} more chars truncated] ..." + RESET)
    else:
        print(content)
    print()

    # Show reward info if available
    extras = example.get("env_extras", {})
    reward_spec = extras.get("reward_spec", {})
    evidence = reward_spec.get("evidence")
    ground_truth = reward_spec.get("ground_truth")
    stop_reason = example.get("stop_reason", "")

    print(DIM + "─" * 80 + RESET)
    print(f"{BOLD}Stop reason:{RESET} {stop_reason}")
    if ground_truth:
        print(f"\n{MAGENTA}{BOLD}Ground truth:{RESET}")
        gt = str(ground_truth)
        print(gt[:1000] + ("..." if len(gt) > 1000 else ""))
    if evidence:
        print(f"\n{MAGENTA}{BOLD}Evidence (reward):{RESET}")
        if isinstance(evidence, list):
            for i, e in enumerate(evidence):
                s = str(e)
                print(f"  [{i}] {s[:400]}" + ("..." if len(s) > 400 else ""))
        else:
            s = str(evidence)
            print(s[:800] + ("..." if len(s) > 800 else ""))
    print()
    print(DIM + "─" * 80 + RESET)
    print(DIM + "ENTER/SPACE/n=next example  p=prev example  q=quit" + RESET)


def run(path: str):
    examples = load_examples(path)
    if not examples:
        print("No examples found.")
        return

    total = len(examples)
    ex_idx = 0

    while True:
        example = examples[ex_idx]
        input_prompt = example.get("input_prompt", "")
        output_response = example.get("output_response", "")

        turns = parse_chatml(input_prompt)
        # output_response may itself be a multi-turn ChatML string, or just plain text.
        # Parse it; if it yields turns, extend; otherwise treat as a single assistant turn.
        if output_response.strip():
            out_turns = parse_chatml(output_response)
            if out_turns:
                turns.extend(out_turns)
            else:
                turns.append({"role": "assistant", "content": output_response})

        turn_idx = 0
        total_turns = len(turns)
        showing_final = False

        while True:
            if turn_idx < total_turns - 1:
                print_turn(turn_idx, total_turns, turns[turn_idx], ex_idx, total)
            else:
                # Last turn — show with reward info appended
                showing_final = True
                print_final(ex_idx, total, example, turns[turn_idx]["content"], total_turns, turn_idx)

            ch = getch()

            if ch in ("\r", "\n", " "):
                if turn_idx < total_turns - 1:
                    turn_idx += 1
                else:
                    # Advance to next example
                    ex_idx = (ex_idx + 1) % total
                    break
            elif ch in ("n", "N"):
                ex_idx = (ex_idx + 1) % total
                break
            elif ch in ("p", "P"):
                ex_idx = (ex_idx - 1) % total
                break
            elif ch in ("q", "Q", "\x03"):  # q or Ctrl-C
                clear_screen()
                print("Bye.")
                return


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
