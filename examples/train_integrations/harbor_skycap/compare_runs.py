"""Diff a skycap run against a run of the sibling ``harbor`` integration, token for token.

The sibling asks vLLM for each turn's token ids (``collect_rollout_details``), so
its Harbor trials hold the exact prompt and completion of every turn: the
ground truth. For the same task under greedy sampling, skycap's record must
hold the same turns. This checks, per task:

- each turn's prompt and completion token ids are identical, and the logprobs agree;
- the training row skycap builds (the path's tokens, trained on the sampled
  ones) equals the baseline's turns merged into one sequence.

Greedy decoding on vLLM is not bitwise stable across batches, so a pair can
fork at some turn. Turns are compared up to the first difference, and where the
pair forks is reported, with the text on each side when ``--tokenizer`` is given.

    python -m examples.train_integrations.harbor_skycap.compare_runs \\
        /tmp/harbor/runs/<baseline> /tmp/harbor/runs/<skycap> [--tokenizer Qwen/Qwen3-4B-Instruct-2507]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Turn = Tuple[List[int], List[int], List[float]]  # prompt ids, completion ids, completion logprobs


def baseline_turns(run: Path) -> Dict[str, List[Turn]]:
    """Per task, the turns vLLM reported to the sibling integration."""
    out: Dict[str, List[Turn]] = {}
    for result in sorted(run.glob("trials/*/result.json")):
        trial = json.loads(result.read_text())
        details = (trial.get("agent_result") or {}).get("rollout_details") or []
        if len(details) != 1:
            continue  # failed, or more than one segment (summarization): not comparable
        d = details[0]
        out[trial["task_name"]] = list(zip(d["prompt_token_ids"], d["completion_token_ids"], d["logprobs"]))
    return out


def skycap_turns(run: Path) -> Dict[str, Tuple[List[Turn], List[int], List[int]]]:
    """Per task, skycap's turns, and the training row it builds (tokens, loss mask)."""
    from skycap import record

    out = {}
    for trajectory_id in record.list_ids(run / "skycap"):
        trajectory = record.load(run / "skycap", trajectory_id)
        paths = trajectory.graph.paths()
        if trajectory.status != "finished" or len(paths) != 1:
            continue
        tokens: List[int] = []
        mask: List[int] = []
        turns: List[Turn] = []
        for node in (trajectory.graph.nodes[i] for i in paths[0]):
            t = node.tokens
            if node.author == "model":
                start = t.sampled_start
                turns.append((tokens + t.token_ids[:start], t.token_ids[start:], t.logprobs[start:]))
                mask += [0] * start + [1] * (len(t.token_ids) - start)
            else:
                mask += [0] * len(t.token_ids)
            tokens = tokens + t.token_ids
        out[Path(trajectory.meta["task"]).name] = (turns, tokens, mask)
    return out


def merged_row(turns: List[Turn]) -> Tuple[List[int], List[int]]:
    """The baseline's turns as one sequence and its loss mask: what prefix merging trains on."""
    prompt, completion, _ = turns[-1]
    tokens = prompt + completion
    mask = [0] * len(tokens)
    for p, c, _ in turns:
        if tokens[: len(p) + len(c)] != p + c:
            return [], []  # a turn isn't a prefix of the last: the baseline's turns don't merge
        mask[len(p) : len(p) + len(c)] = [1] * len(c)
    return tokens, mask


def first_difference(a: List[int], b: List[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(base: Dict[str, List[Turn]], cap: Dict[str, Any], decode) -> List[Dict[str, Any]]:
    rows = []
    for task in sorted(set(base) & set(cap)):
        b_turns = base[task]
        c_turns, c_tokens, c_mask = cap[task]
        same, fork, logprob_diff = 0, None, 0.0
        for index, ((bp, bc, bl), (cp, cc, cl)) in enumerate(zip(b_turns, c_turns)):
            if bp != cp:
                fork = ("prompt", index, first_difference(bp, cp), bp, cp)
                break
            at = first_difference(bc, cc)
            shared = len(bc) if at is None else at  # logprobs are comparable up to the first different token
            logprob_diff = max([logprob_diff, *(abs(x - y) for x, y in zip(bl[:shared], cl[:shared]))])
            if at is not None:
                fork = ("completion", index, at, bc, cc)
                break
            same += 1
        b_tokens, b_mask = merged_row(b_turns)
        row = {
            "task": task,
            "turns": f"{len(b_turns)} / {len(c_turns)}",
            "identical_turns": same,
            "max_logprob_diff": logprob_diff,  # over the shared tokens
            "row_identical": bool(b_tokens) and (b_tokens, b_mask) == (c_tokens, c_mask),
            "fork": None,
        }
        if fork is not None:
            where, turn, at, bs, cs = fork
            row["fork"] = f"turn {turn} {where}, token {at}"
            if decode is not None:
                window = slice(max(0, at - 8), at + 8)
                row["fork"] += f": baseline {decode(bs[window])!r} vs skycap {decode(cs[window])!r}"
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("skycap", type=Path)
    parser.add_argument("--tokenizer", help="to show the text on each side of a fork")
    args = parser.parse_args()

    decode = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731

    base, cap = baseline_turns(args.baseline), skycap_turns(args.skycap)
    rows = compare(base, cap, decode)
    print(
        "| task | turns (baseline / skycap) | identical turns | max logprob diff before the fork | training row identical |"
    )
    print("| --- | --- | --- | --- | --- |")
    for r in rows:
        print(
            f"| {r['task']} | {r['turns']} | {r['identical_turns']} | {r['max_logprob_diff']:.2e} "
            f"| {'yes' if r['row_identical'] else 'no'} |"
        )
    print()
    turns = sum(int(r["turns"].split(" / ")[0]) for r in rows)
    identical = sum(r["identical_turns"] for r in rows)
    print(f"{len(rows)} tasks compared ({len(base)} baseline, {len(cap)} skycap trials usable)")
    print(f"{identical} / {turns} baseline turns token-identical in skycap")
    print(f"{sum(r['row_identical'] for r in rows)} / {len(rows)} training rows identical")
    for r in rows:
        if r["fork"]:
            print(f"  {r['task']}: forks at {r['fork']}")
    return 0 if rows and all(r["row_identical"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
