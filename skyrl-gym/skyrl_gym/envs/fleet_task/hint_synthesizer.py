"""LLM-synthesized hints for failed trajectories.

Analyzes the full failed trajectory + verifier errors and produces actionable
guidance via Claude Sonnet. Falls back to static build_hint_text() on failure.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Category tag for LLM-synthesized hints
CATEGORY_LLM = "llm_synthesized"
CATEGORY_STATIC = "static_fallback"
CATEGORY_LLM_FAILED = "llm_failed_static_fallback"

HINT_SYSTEM_PROMPT = """\
You are a debugging assistant for an AI agent that failed a task. \
Analyze the failed trajectory and verifier feedback, then provide \
2-5 sentences of actionable guidance for the agent's next attempt.

Rules:
- Be specific: reference exact actions that failed and why.
- Be actionable: tell the agent what to do differently, not just what went wrong.
- If the agent ran out of context/turns, suggest being more efficient (fewer unnecessary steps).
- If tool calls errored, explain the correct usage pattern.
- Do NOT repeat the task instructions verbatim.
- Do NOT say "the previous attempt failed" — the agent already knows that."""


def format_trajectory_for_hint(
    chat_history: List[Dict[str, Any]],
    max_turns: int = 15,
    max_msg_chars: int = 3000,
    max_total_chars: int = 150_000,
) -> str:
    """Format chat_history into readable text for LLM hint synthesis.

    Truncates to the last `max_turns` messages, caps individual messages,
    and enforces a total character budget.
    """
    if not chat_history:
        return "(empty trajectory)"

    # Take last N turns
    recent = chat_history[-max_turns:]
    parts = []
    total = 0

    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle list-type content (multimodal messages)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        text_parts.append("[image]")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "unknown_tool")
                        inp = str(block.get("input", ""))[:500]
                        text_parts.append(f"[tool_use: {name}({inp})]")
                    elif block.get("type") == "tool_result":
                        text_parts.append(f"[tool_result: {str(block.get('content', ''))[:500]}]")
                    else:
                        text_parts.append(str(block)[:200])
                else:
                    text_parts.append(str(block)[:200])
            content = "\n".join(text_parts)

        if isinstance(content, str) and len(content) > max_msg_chars:
            content = content[:max_msg_chars] + f"... [truncated, {len(content)} chars total]"

        line = f"[{role}]: {content}"
        if total + len(line) > max_total_chars:
            parts.append(f"... [trajectory truncated at {max_total_chars} chars]")
            break
        parts.append(line)
        total += len(line)

    return "\n\n".join(parts)


def format_verifier_feedback(
    verifier_stdout: Optional[str],
    verifier_error: Optional[str],
    tool_error_messages: Optional[List[str]],
) -> str:
    """Extract verifier errors/successes and tool errors into readable text."""
    import ast
    import re

    parts = []

    if verifier_stdout:
        err_match = re.search(
            r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
            verifier_stdout,
            re.DOTALL,
        )
        suc_match = re.search(
            r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n<<< SUCCESS_ACCUMULATOR <<<",
            verifier_stdout,
            re.DOTALL,
        )
        if err_match or suc_match:
            try:
                errors = ast.literal_eval(err_match.group(1)) if err_match else []
                successes = ast.literal_eval(suc_match.group(1)) if suc_match else []
            except Exception:
                errors, successes = [], []
            if successes:
                parts.append(f"Verifier checks PASSED ({len(successes)}):")
                for s in successes[:10]:
                    parts.append(f"  - {str(s)[:200]}")
            if errors:
                parts.append(f"Verifier checks FAILED ({len(errors)}):")
                for e in errors[:10]:
                    parts.append(f"  - {str(e)[:200]}")

    if verifier_error:
        parts.append(f"Verifier error: {verifier_error[:500]}")

    if tool_error_messages:
        unique = list(dict.fromkeys(tool_error_messages))[:10]
        parts.append("Tool errors encountered:")
        for e in unique:
            parts.append(f"  - {e[:300]}")

    return "\n".join(parts) if parts else "(no verifier feedback available)"


async def synthesize_hint(
    task_prompt: str,
    chat_history: List[Dict[str, Any]],
    verifier_stdout: Optional[str],
    verifier_error: Optional[str],
    tool_error_messages: Optional[List[str]],
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 30.0,
    static_fallback_fn=None,
) -> Tuple[str, str]:
    """Synthesize a hint from a failed trajectory using an LLM.

    Returns:
        (hint_text, hint_category) where category is one of
        CATEGORY_LLM, CATEGORY_STATIC, CATEGORY_LLM_FAILED.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, falling back to static hints")
        if static_fallback_fn:
            return static_fallback_fn(verifier_stdout, verifier_error, tool_error_messages), CATEGORY_STATIC
        return "The previous attempt failed. Try a different approach.", CATEGORY_STATIC

    trajectory_text = format_trajectory_for_hint(chat_history)
    verifier_text = format_verifier_feedback(verifier_stdout, verifier_error, tool_error_messages)

    user_message = f"""## Task
{task_prompt[:5000]}

## Agent Trajectory (last turns)
{trajectory_text}

## Verifier Feedback
{verifier_text}

Based on the trajectory and feedback above, provide 2-5 sentences of specific, actionable guidance for the agent's next attempt."""

    try:
        client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.3,
                system=HINT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ),
            timeout=timeout,
        )
        hint_text = response.content[0].text.strip()
        if hint_text:
            return hint_text, CATEGORY_LLM
        else:
            logger.warning("LLM returned empty hint, falling back to static")
    except asyncio.TimeoutError:
        logger.warning(f"LLM hint synthesis timed out after {timeout}s")
    except Exception as e:
        logger.warning(f"LLM hint synthesis failed: {e}")

    # Fallback to static hint
    if static_fallback_fn:
        return static_fallback_fn(verifier_stdout, verifier_error, tool_error_messages), CATEGORY_LLM_FAILED
    return "The previous attempt failed. Try a different approach.", CATEGORY_LLM_FAILED


async def synthesize_hints_batch(
    hint_requests: List[Dict[str, Any]],
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 30.0,
    max_concurrency: int = 20,
    static_fallback_fn=None,
) -> List[Tuple[str, str]]:
    """Synthesize hints for a batch of failed trajectories concurrently.

    Args:
        hint_requests: List of dicts with keys:
            - task_prompt: str
            - chat_history: List[Dict]
            - verifier_stdout: Optional[str]
            - verifier_error: Optional[str]
            - tool_error_messages: Optional[List[str]]
            - instance_id: str (for logging)
        model: LLM model to use
        timeout: per-request timeout
        max_concurrency: max concurrent LLM calls
        static_fallback_fn: fallback function for static hints

    Returns:
        List of (hint_text, hint_category) tuples, one per request.
    """
    if not hint_requests:
        return []

    sem = asyncio.Semaphore(max_concurrency)
    start = time.monotonic()

    async def _synth(req: Dict[str, Any]) -> Tuple[str, str]:
        async with sem:
            return await synthesize_hint(
                task_prompt=req["task_prompt"],
                chat_history=req.get("chat_history", []),
                verifier_stdout=req.get("verifier_stdout"),
                verifier_error=req.get("verifier_error"),
                tool_error_messages=req.get("tool_error_messages"),
                model=model,
                timeout=timeout,
                static_fallback_fn=static_fallback_fn,
            )

    results = await asyncio.gather(*[_synth(req) for req in hint_requests])

    elapsed = time.monotonic() - start
    n_llm = sum(1 for _, cat in results if cat == CATEGORY_LLM)
    n_fallback = sum(1 for _, cat in results if cat in (CATEGORY_STATIC, CATEGORY_LLM_FAILED))
    logger.info(
        f"Hint synthesis batch: {len(results)} total, {n_llm} LLM-synthesized, "
        f"{n_fallback} fallback, {elapsed:.1f}s elapsed"
    )

    return list(results)
