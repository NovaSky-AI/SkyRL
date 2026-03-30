#!/usr/bin/env python3
"""Pretty-print rollout data from a dumped eval JSONL file."""

import argparse
import ast
import contextlib
import io
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
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
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


def parse_prompt_turns(prompt: str) -> list[dict]:
    """Split a ChatML-formatted prompt into turns."""
    turns = []
    parts = prompt.split("<|im_start|>")
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


_tokenizer = None

DEFAULT_MODEL = "alphaXiv/rlm-sft-Qwen3.5-9B-v1"


def load_tokenizer(model: str):
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def count_tokens(text: str) -> int:
    if _tokenizer is not None:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return max(1, int(len(text) / 3.5))


def _trajectory_f1(scores: list) -> float:
    """Get the actual trajectory F1: the last non-zero value in per-token rewards."""
    if not scores:
        return 0.0
    for v in reversed(scores):
        if v > 0:
            return v
    return 0.0


def _fmt_scores(scores: list) -> str:
    """Format a score list as a colored string using the actual trajectory F1."""
    if not scores:
        return c("no scores", "dim")
    f1 = _trajectory_f1(scores)
    if f1 >= 0.99:
        return c(f"F1={f1:.3f}  (exact)", "bold", "green")
    elif f1 > 0.0:
        return c(f"F1={f1:.3f}  (partial)", "yellow", "bold")
    else:
        return c(f"F1={f1:.3f}  (no match)", "bold", "red")


def _fmt_prf(p, r, f1) -> str:
    """Format P/R/F1 as a colored string."""
    if f1 is None:
        return c("—", "dim")
    color = "green" if f1 >= 0.99 else ("yellow" if f1 > 0.0 else "red")
    p_str = f"{p:.3f}" if p is not None else "—"
    r_str = f"{r:.3f}" if r is not None else "—"
    return c(f"F1={f1:.3f}", color, "bold") + c(f"  P={p_str}  R={r_str}", color)


def compute_prf(steps: list) -> tuple:
    """
    Recompute precision, recall, F1 for a trajectory by re-executing its REPL blocks.

    Returns (precision, recall, f1). If recomputation fails, precision and recall
    are None and f1 falls back to the stored per-token reward.
    """
    last = steps[-1]
    stored_f1 = _trajectory_f1(last["score"])

    # Only works for envs that store evidence + context_text
    try:
        from skyrl_gym.envs.rlm.evidence_tools import compute_metrics, make_tools
    except ImportError:
        return None, None, stored_f1

    extras = last["env_extras"]
    rs = extras.get("reward_spec", {})
    if isinstance(rs, str):
        rs = eval(rs)
    evidence = rs.get("evidence") if isinstance(rs, dict) else None
    if not evidence:
        return None, None, stored_f1

    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = eval(extra_info)
    ctx = extra_info.get("context_text", "") if isinstance(extra_info, dict) else ""
    if not ctx:
        return None, None, stored_f1

    # Build full conversation text
    full_conv = steps[0]["input_prompt"]
    for step in steps:
        full_conv += step["output_response"]

    # Use the LAST FINAL_VAR call (the first match is often in the system prompt example)
    final_var_matches = list(re.finditer(r'FINAL_VAR\((\w+)\)', full_conv))
    if not final_var_matches:
        return None, None, stored_f1
    varname = final_var_matches[-1].group(1)

    repl_blocks = re.findall(r'```repl\n(.*?)```', full_conv, re.DOTALL)
    if not repl_blocks:
        return None, None, stored_f1

    tools = make_tools(ctx)
    ns = {
        "search": tools["search"]["tool"],
        "extract_section": tools["extract_section"]["tool"],
        "SHOW_VARS": lambda: list(ns.keys()),
        "FINAL_VAR": lambda v: None,
        "context": ctx,
    }

    final_answer_str = None
    for block in repl_blocks:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(block, ns)  # noqa: S102
        except Exception:
            pass
        if "FINAL_VAR" in block:
            val = ns.get(varname)
            if val is not None:
                final_answer_str = str(val)
            break

    if final_answer_str is None:
        return None, None, stored_f1

    # Reproduce reward_fn logic from evidence_tools.make_reward_fn
    evidence_intervals = []
    for ev in evidence:
        idx = ctx.find(ev.strip())
        if idx != -1:
            evidence_intervals.append((idx, idx + len(ev.strip())))

    try:
        substrings = ast.literal_eval(final_answer_str)
        if isinstance(substrings, str):
            substrings = [substrings]
        elif not isinstance(substrings, list):
            substrings = [str(substrings)]
    except (ValueError, SyntaxError):
        substrings = [s.strip() for s in final_answer_str.split("\n\n") if s.strip()]

    retrieved_intervals = []
    for s in substrings:
        idx = ctx.find(s)
        if idx != -1:
            retrieved_intervals.append((idx, idx + len(s)))

    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    return metrics["precision"], metrics["recall"], metrics["f1"]


def print_step_wise_trajectory(traj_idx: int, steps: list[dict], max_response_chars: int, max_context_chars: int, show_context: bool, prf: tuple = None):
    """Print a grouped step-wise trajectory (multiple records = one trajectory)."""
    first = steps[0]
    last = steps[-1]
    extras = first["env_extras"]
    reward_spec = extras.get("reward_spec", {})
    if isinstance(reward_spec, str):
        reward_spec = eval(reward_spec)
    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = eval(extra_info)

    ground_truth = reward_spec.get("ground_truth", "N/A")
    max_turns = extras.get("max_turns", "N/A")
    context_text = extra_info.get("context_text", "")
    env_class = first["env_class"]
    data_source = first["data_source"]
    scores = last["score"]
    stop_reason = last["stop_reason"]
    if prf is None:
        prf = (None, None, _trajectory_f1(scores))

    # Combine all steps into the full conversation
    full_transcript = first["input_prompt"]
    for step in steps:
        full_transcript += step["output_response"]
    turns = parse_prompt_turns(full_transcript)
    assistant_turns = [t for t in turns if t["role"] == "assistant" and len(t["content"].strip()) > 10]
    repl_turns = [t for t in turns if t["role"] == "user" and t["content"].startswith("Code executed:")]

    total_response_chars = sum(len(s["output_response"]) for s in steps)
    total_response_tokens = sum(count_tokens(s["output_response"]) for s in steps)

    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ║", "bold", "cyan") + c(f"{'TRAJECTORY #' + str(traj_idx):^86}", "bold", "white") + c("║", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ╚{'═' * 86}╝", "bold", "cyan"))
    print()

    print(section_header("Metadata"))
    print()

    precision, recall, f1 = prf
    meta_rows = [
        ("Env Class", c(env_class, "bold")),
        ("Data Source", data_source),
        ("Stop Reason", c(stop_reason, "green" if stop_reason == "stop" else "yellow", "bold")),
        ("Ground Truth", c(str(ground_truth), "bold", "green")),
        ("Max Turns", str(max_turns)),
        ("Actual Steps", f"{len(steps)} steps, {len(assistant_turns)} assistant, {len(repl_turns)} REPL"),
        ("Total Generation", f"~{total_response_tokens:,} tokens ({total_response_chars:,} chars)"),
        ("── Metrics ──", ""),
        ("F1 / Precision / Recall", _fmt_prf(precision, recall, f1)),
    ]
    for label, value in meta_rows:
        print(f"  {c(label + ':', 'bold'):>38s}  {value}")

    print()
    print(section_header("Per-Step Lengths"))
    print()
    print(f"  {'Step':>6}  {'Prompt':>14}  {'Response':>14}  {'Cumulative':>14}  {'Stop':>8}")
    print(f"  {'─' * 6}  {'─' * 14}  {'─' * 14}  {'─' * 14}  {'─' * 8}")

    cumulative_tokens = 0
    for si, step in enumerate(steps):
        prompt_tok = count_tokens(step["input_prompt"])
        resp_tok = count_tokens(step["output_response"])
        cumulative_tokens += resp_tok
        sr = step["stop_reason"]
        sr_colored = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
        print(f"  {si + 1:>6}  {prompt_tok:>8} tok  {resp_tok:>8} tok  {cumulative_tokens:>8} tok  {sr_colored:>19}")

    print()
    init_prompt_tok = count_tokens(first["input_prompt"])
    final_prompt_tok = count_tokens(last["input_prompt"])
    print(f"  {c('Initial prompt:', 'dim')}  ~{init_prompt_tok:,} tokens ({len(first['input_prompt']):,} chars)")
    print(f"  {c('Final prompt:', 'dim')}   ~{final_prompt_tok:,} tokens ({len(last['input_prompt']):,} chars)")
    print(f"  {c('Context size:', 'dim')}   {len(context_text):,} chars")
    print()

    print(section_header("Conversation"))

    user_query = None
    for turn in turns:
        if turn["role"] == "user" and not turn["content"].startswith("[Execution Result]"):
            user_query = turn["content"]

    if user_query:
        print()
        print(f"  {c('Query:', 'bold', 'yellow')}  {user_query}")

    print()
    for turn in turns:
        print(format_role(turn["role"]))
        for line in turn["content"].split("\n"):
            print(f"    {line}")
        print()

    if show_context and context_text:
        print(section_header("Context (from extra_info)"))
        print()
        ctx = truncate(context_text, max_context_chars)
        for line in ctx.split("\n"):
            print(f"    {line}")
        print()

    print(hr())
    print()


def print_rollout(idx: int, record: dict, max_response_chars: int, max_context_chars: int, show_context: bool):
    extras = record["env_extras"]
    reward_spec = extras.get("reward_spec", {})
    if isinstance(reward_spec, str):
        reward_spec = eval(reward_spec)
    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = eval(extra_info)

    ground_truth = reward_spec.get("ground_truth", "N/A")
    max_turns = extras.get("max_turns", "N/A")
    context_text = extra_info.get("context_text", "")

    prompt = record["input_prompt"]
    response = record["output_response"]
    scores = record["score"]
    stop_reason = record["stop_reason"]
    env_class = record["env_class"]
    data_source = record["data_source"]

    full_transcript = prompt + response
    turns = parse_prompt_turns(full_transcript)

    assistant_turns = [t for t in turns if t["role"] == "assistant" and len(t["content"].strip()) > 10]
    repl_turns = [t for t in turns if t["role"] == "user" and t["content"].startswith("Code executed:")]

    prompt_tok = count_tokens(prompt)
    resp_tok = count_tokens(response)

    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ║", "bold", "cyan") + c(f"{'ROLLOUT #' + str(idx):^86}", "bold", "white") + c("║", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ╚{'═' * 86}╝", "bold", "cyan"))
    print()

    print(section_header("Metadata"))
    print()

    meta_rows = [
        ("Env Class", c(env_class, "bold")),
        ("Data Source", data_source),
        ("Stop Reason", c(stop_reason, "green" if stop_reason == "stop" else "yellow")),
        ("Ground Truth", c(str(ground_truth), "bold", "green")),
        ("Max Turns", str(max_turns)),
        ("Agent Turns", f"{len(assistant_turns)} assistant, {len(repl_turns)} REPL"),
        ("Prompt Length", f"~{prompt_tok:,} tokens ({len(prompt):,} chars)"),
        ("Response Length", f"~{resp_tok:,} tokens ({len(response):,} chars)"),
        ("Score (F1)", _fmt_scores(scores)),
    ]
    for label, value in meta_rows:
        print(f"  {c(label + ':', 'bold'):>38s}  {value}")

    print()
    print(section_header("Conversation"))

    user_query = None
    for turn in turns:
        if turn["role"] == "user" and not turn["content"].startswith("[Execution Result]"):
            user_query = turn["content"]

    if user_query:
        print()
        print(f"  {c('Query:', 'bold', 'yellow')}  {user_query}")

    print()
    for turn_num, turn in enumerate(turns):
        print(format_role(turn["role"]))
        content = turn["content"]
        if turn["role"] == "system":
            pass
        elif turn["role"] == "user" and turn["content"].startswith("[Execution Result]"):
            pass

        for line in content.split("\n"):
            print(f"    {line}")
        print()

    if show_context and context_text:
        print(section_header("Context (from extra_info)"))
        print()
        ctx = truncate(context_text, max_context_chars)
        for line in ctx.split("\n"):
            print(f"    {line}")
        print()

    print(hr())
    print()


def group_step_wise_trajectories(records: list[dict]) -> list[list[dict]]:
    """Group step-wise records into trajectories.

    Step-wise records for the same trajectory have strictly growing prompt
    lengths AND share a common prefix with the first record in the group.
    A new trajectory starts whenever the prompt doesn't extend the first
    record's prompt (i.e. it's a different conversation entirely).
    """
    if not records:
        return []

    trajectories = []
    current: list[dict] = [records[0]]
    first_prompt = records[0]["input_prompt"]

    for r in records[1:]:
        cur_prompt = r["input_prompt"]
        extends_first = (
            len(cur_prompt) > len(first_prompt)
            and cur_prompt[:len(first_prompt)] == first_prompt
        )
        if extends_first:
            current.append(r)
        else:
            trajectories.append(current)
            current = [r]
            first_prompt = cur_prompt
    if current:
        trajectories.append(current)
    return trajectories


def print_trajectory_summary(trajectories: list[list[dict]], all_prf: list[tuple] = None):
    """Print a compact summary table of all trajectories."""
    print(f"  {'#':>4}  {'Steps':>5}  {'Stop':>8}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Tot Gen':>10}  {'GT':>6}")
    print(f"  {'─' * 4}  {'─' * 5}  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 10}  {'─' * 6}")
    for ti, steps in enumerate(trajectories):
        n_steps = len(steps)
        sr = steps[-1]["stop_reason"]
        total_gen_tok = sum(count_tokens(s["output_response"]) for s in steps)
        extras = steps[0]["env_extras"]
        rs = extras.get("reward_spec", {})
        if isinstance(rs, str):
            rs = eval(rs)
        gt = rs.get("ground_truth", "?")
        sr_colored = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")

        p, r, f1 = all_prf[ti] if all_prf else (None, None, _trajectory_f1(steps[-1]["score"]))
        f1_color = "green" if f1 >= 0.99 else ("yellow" if f1 > 0.0 else "red")
        f1_str = c(f"{f1:.3f}", f1_color, "bold")
        p_str = c(f"{p:.3f}", f1_color) if p is not None else c("—", "dim")
        r_str = c(f"{r:.3f}", f1_color) if r is not None else c("—", "dim")

        print(f"  {ti:>4}  {n_steps:>5}  {sr_colored:>19}  {f1_str:>17}  {p_str:>17}  {r_str:>17}  {total_gen_tok:>6} tok  {str(gt):>6}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and pretty-print rollouts from a dumped eval JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/analyze_rollout.py eval.jsonl
              python scripts/analyze_rollout.py eval.jsonl --max-rollouts 5
              python scripts/analyze_rollout.py eval.jsonl --show-context --max-context-chars 2000
              python scripts/analyze_rollout.py eval.jsonl --offset 3 --max-rollouts 2
              python scripts/analyze_rollout.py --summary          # compact table of all trajectories
        """),
    )
    parser.add_argument("file", type=Path, nargs="?", default=Path("/root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl"), help="Path to the JSONL file (default: /root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl)")
    parser.add_argument("--max-rollouts", type=int, default=1, help="Number of rollouts/trajectories to display (default: 1)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rollouts/trajectories before printing (default: 0)")
    parser.add_argument("--max-response-chars", type=int, default=3000, help="Max chars to show per response (default: 3000)")
    parser.add_argument("--show-context", action="store_true", help="Also print the context_text from extra_info")
    parser.add_argument("--max-context-chars", type=int, default=2000, help="Max chars of context to show (default: 2000)")
    parser.add_argument("--scores-only", action="store_true", help="Only print a compact score summary for each raw record")
    parser.add_argument("--summary", action="store_true", help="Print a compact trajectory summary table with per-step lengths")
    parser.add_argument("--raw", action="store_true", help="Show raw per-record view instead of grouped trajectory view")
    parser.add_argument("--recompute-prf", action="store_true", help="Recompute precision/recall by re-executing REPL blocks (RLM env only)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"HuggingFace model for tokenization (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-tokenizer", action="store_true", help="Skip loading the tokenizer; use a rough chars/3.5 estimate instead")
    args = parser.parse_args()

    if not args.no_tokenizer:
        print(c(f"  Loading tokenizer: {args.model} ...", "dim"), end="", flush=True)
        load_tokenizer(args.model)
        print(c(" done", "dim"))

    if not args.file.exists():
        print(c(f"Error: file not found: {args.file}", "red", "bold"))
        return

    records = []
    with open(args.file) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    trajectories = group_step_wise_trajectories(records)
    is_step_wise = len(trajectories) < total

    print()
    print(c(f"  Loaded {total} records from ", "dim") + c(str(args.file), "bold", "underline"))
    if is_step_wise:
        print(c(f"  Grouped into {len(trajectories)} trajectories (step-wise, {total / len(trajectories):.0f} steps avg)", "dim"))
    print()

    # Precompute P/R/F1 for all trajectories if requested
    all_prf = None
    if args.recompute_prf and is_step_wise:
        print(c("  Computing P/R/F1 (re-executing REPL blocks)...", "dim"))
        all_prf = []
        for steps in trajectories:
            all_prf.append(compute_prf(steps))
        print()

    if args.scores_only:
        print(f"  {'#':>4}  {'Stop':>8}  {'GT':>6}  {'Env':>6}  {'Score':>12}  {'Prompt':>10}  {'Resp':>10}")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        for i in range(total):
            r = records[i]
            extras = r["env_extras"]
            rs = extras.get("reward_spec", {})
            if isinstance(rs, str):
                rs = eval(rs)
            gt = rs.get("ground_truth", "?")
            score_str = _fmt_scores(r["score"])
            p_tok = count_tokens(r["input_prompt"])
            r_tok = count_tokens(r["output_response"])
            print(f"  {i:>4}  {r['stop_reason']:>8}  {str(gt):>6}  {r['env_class']:>6}  {score_str:>23}  {p_tok:>6} tok  {r_tok:>6} tok")
        print()
        return

    if args.summary:
        if is_step_wise:
            print_trajectory_summary(trajectories, all_prf)
            for ti, steps in enumerate(trajectories):
                print(c(f"  Trajectory {ti}:", "bold"))
                print(f"    {'Step':>4}  {'Prompt':>10}  {'Response':>10}  {'Stop':>8}")
                print(f"    {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
                for si, s in enumerate(steps):
                    p = count_tokens(s["input_prompt"])
                    r = count_tokens(s["output_response"])
                    sr = s["stop_reason"]
                    sr_c = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
                    print(f"    {si + 1:>4}  {p:>6} tok  {r:>6} tok  {sr_c:>19}")
                print()
        else:
            print(f"  {'#':>4}  {'Turns':>18}  {'Tot Gen':>10}  {'Stop':>8}  {'Score'}")
            print(f"  {'─' * 4}  {'─' * 18}  {'─' * 10}  {'─' * 8}  {'─' * 20}")
            for i in range(total):
                r = records[i]
                full = r["input_prompt"] + r["output_response"]
                turns = parse_prompt_turns(full)
                n_sys = sum(1 for t in turns if t["role"] == "system")
                n_usr = sum(1 for t in turns if t["role"] == "user")
                n_ast = sum(1 for t in turns if t["role"] == "assistant")
                resp_tok = count_tokens(r["output_response"])
                sr = r["stop_reason"]
                sr_c = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
                score_str = _fmt_scores(r["score"])
                turns_str = f"{n_sys}s {n_usr}u {n_ast}a"
                print(f"  {i:>4}  {turns_str:>18}  {resp_tok:>6} tok  {sr_c:>19}  {score_str}")
            print()
        return

    if is_step_wise and not args.raw:
        start = args.offset
        end = min(start + args.max_rollouts, len(trajectories))
        if start >= len(trajectories):
            print(c(f"Offset {start} is beyond the {len(trajectories)} trajectories.", "red", "bold"))
            return
        print(c(f"  Showing trajectories {start}..{end - 1}  (of {len(trajectories)} total)", "dim"))
        print()
        for ti in range(start, end):
            prf = all_prf[ti] if all_prf else None
            print_step_wise_trajectory(ti, trajectories[ti], args.max_response_chars, args.max_context_chars, args.show_context, prf=prf)
    else:
        start = args.offset
        end = min(start + args.max_rollouts, total)
        if start >= total:
            print(c(f"Offset {start} is beyond the {total} records.", "red", "bold"))
            return
        print(c(f"  Showing rollouts {start}..{end - 1}  (of {total} total)", "dim"))
        print()
        for i in range(start, end):
            print_rollout(i, records[i], args.max_response_chars, args.max_context_chars, args.show_context)


if __name__ == "__main__":
    main()
