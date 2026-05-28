"""Utils for trajectory logging."""

import dataclasses
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from skyrl.train.utils.tracking import Tracking

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"


def _color_block_format_and_kwargs(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\\n<color>{p1}</color>\\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    kwargs: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        kwargs[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, kwargs


def pretty_print_example(
    logger: Any,
    prompt: List[Dict[str, Any]],
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt in OpenAI message format.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    reward_val = 0.0
    reward_str = "N/A"
    try:
        prompt_str = str(prompt)
        response_str = str(response)
        # --- Reward handling ---
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_kwargs = _color_block_format_and_kwargs(prompt_str, BASE_PROMPT_COLOR, "p")
        response_fmt, response_kwargs = _color_block_format_and_kwargs(response_str, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = "Example:\n" f"  Input: {prompt_fmt}\n" "  Output (Total Reward: {reward}):\n" f"{response_fmt}"

        # Merge all args for str.format
        format_kwargs = {**prompt_kwargs, **response_kwargs, "reward": reward_str}

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_kwargs)
    except Exception as e:
        print(f"Error pretty printing example, debug printing instead: {e}")
        print(f"Example:\n  Input: {prompt}\n  Output (Total Reward: {reward_str}):\n{response}")


@dataclasses.dataclass
class TrajectoryLogger:
    """Logs rollout samples to tracker backends as a table.

    Accepts a full ``GeneratorOutput``-shaped dict plus a parallel list of
    prompts, and derives the per-sample fields (``num_turns`` from the loss
    mask, the trajectory string from prompt + response tokens) internally.
    A caller with a better signal for ``num_turns`` (e.g. step-wise eval
    counting trajectory steps) can pass it in explicitly to override.

    Designed for subclassing. Override the granular pieces to customize:

      - :meth:`build_samples` -- which columns each row contains. If you
        change the tuple shape, also update :attr:`columns`.
      - :meth:`format_trajectory` -- how the trajectory string is rendered.
      - :meth:`count_assistant_turns` -- default loss-mask -> turn-count.
      - :meth:`log` -- top-level dispatch (e.g. add a new backend, change the
        wandb key, prepend different per-row metadata).

    All hooks accept ``**kwargs`` so callers can plumb extra fields through
    a subclass (e.g. ``data_source`` for a per-dataset column) without
    changing the base API.
    """

    columns: Tuple[str, ...] = ("step", "idx", "reward", "num_turns", "trajectory")
    sample_seed: int = 0
    """Seed for the random picks in :meth:`select_sample_indices`. Fixed so
    consecutive runs surface the same prompt/response pairs in the table."""

    def log(
        self,
        *,
        tracker: Optional["Tracking"],
        num_samples: int,
        prompts: List[Any],
        generator_output: Dict[str, Any],
        tokenizer: Any,
        global_step: Optional[int],
        num_turns_list: Optional[List[int]] = None,
        wandb_key: str,
        include_idx: bool = True,
        **kwargs: Any,
    ) -> None:
        """Build sample rows from a GeneratorOutput-shaped dict and dispatch.

        ``generator_output`` must contain at least ``response_ids``,
        ``rewards`` and ``loss_masks``. ``prompts`` is passed separately
        because it lives on :class:`GeneratorInput`, not output.

        ``tracker`` is the active :class:`Tracking` instance. Today only the
        wandb backend gets a trajectory table written (via
        :meth:`Tracking.log_samples_to_table`); other backends are a no-op.

        ``num_turns_list`` defaults to per-response assistant-span counts
        derived from ``loss_masks`` via :meth:`count_assistant_turns`. Pass
        an explicit value (e.g. trajectory step counts in step-wise eval)
        to override.

        Extra ``**kwargs`` are forwarded to :meth:`build_samples` for
        subclass extensibility.

        No-op when ``num_samples <= 0`` or the tracker is not a backend we
        know how to write to.
        """
        response_ids = generator_output.get("response_ids") or []
        if num_samples <= 0 or tracker is None or tracker.backend != "wandb" or not response_ids:
            return
        loss_masks = generator_output.get("loss_masks") or []
        rewards = generator_output.get("rewards") or []
        if num_turns_list is None:
            num_turns_list = [self.count_assistant_turns(m) for m in loss_masks]
        samples = self.build_samples(
            num_samples=num_samples,
            prompts=prompts,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            num_turns_list=num_turns_list,
            tokenizer=tokenizer,
            **kwargs,
        )
        if not samples:
            return
        # `global_step` may be None (eval-only context); the table API wants
        # a numeric step.
        step = 0 if global_step is None else global_step
        # ``build_samples`` always emits ``(idx, reward, num_turns, trajectory)``
        # tuples; drop the leading idx when the caller doesn't want it logged.
        columns = list(self.columns)
        if not include_idx:
            columns = [c for c in columns if c != "idx"]
            samples = [sample[1:] for sample in samples]
        # Prepend `step` to each row so rows from different calls remain
        # distinguishable in the accumulating table.
        tracker.log_samples_to_table(
            key=wandb_key,
            columns=columns,
            samples=[(step, *sample) for sample in samples],
            step=step,
        )

    def build_samples(
        self,
        *,
        num_samples: int,
        prompts: List[Any],
        response_ids: List[List[int]],
        rewards: List[float],
        loss_masks: List[List[int]],
        num_turns_list: List[int],
        tokenizer: Any,
        **kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """Build the per-row tuples to be written.

        Default shape: ``(idx, reward, num_turns, trajectory)`` (matching
        :attr:`columns` after the ``step`` column is prepended by :meth:`log`).
        ``idx`` is the position in the input arrays, *not* a sequential row
        number, so it points back to the original sample.

        Indices are chosen by :meth:`select_sample_indices`, which by default
        anchors on the min- and max-reward samples and fills the rest at
        random. Override either method to customize selection or column shape;
        ``**kwargs`` are whatever extra values the caller passed to
        :meth:`log`.
        """
        total = min(
            len(response_ids),
            len(prompts),
            len(rewards),
            len(loss_masks),
            len(num_turns_list),
        )
        # Per-token rewards arrive as lists; collapse to scalars so the
        # min/max picks and the wandb column are both well-typed.
        scalar_rewards = [float(sum(r)) if isinstance(r, list) else float(r) for r in rewards[:total]]
        indices = self.select_sample_indices(num_samples=num_samples, rewards=scalar_rewards, total=total)
        return [
            (
                i,
                scalar_rewards[i],
                num_turns_list[i],
                self.format_trajectory(prompts[i], response_ids[i], loss_masks[i], tokenizer, **kwargs),
            )
            for i in indices
        ]

    def select_sample_indices(
        self,
        *,
        num_samples: int,
        rewards: Sequence[float],
        total: int,
    ) -> List[int]:
        """Pick up to ``num_samples`` indices from ``[0, total)`` to log.

        Guarantees that the min- and max-reward samples are included when
        ``num_samples >= 2`` and ``total >= 2``; remaining slots are filled by
        random sampling without replacement. With ``num_samples == 1`` only
        the min-reward sample is kept (arbitrary choice when we can fit one).
        If all rewards tie, ``min`` and ``max`` may resolve to the same index;
        the duplicate is dropped and the rest of the budget goes to random
        picks. Returned indices are sorted ascending so the output table reads
        in input order.
        """
        if total <= 0 or num_samples <= 0:
            return []
        n = min(num_samples, total)
        if n >= total:
            return list(range(total))

        # Anchor on extremes. `min`/`max` return the first occurrence on ties,
        # which is fine -- we just need one of each.
        min_idx = min(range(total), key=lambda i: rewards[i])
        max_idx = max(range(total), key=lambda i: rewards[i])
        anchors: List[int] = []
        for cand in (min_idx, max_idx):
            if cand not in anchors:
                anchors.append(cand)
        anchors = anchors[:n]

        rest_needed = n - len(anchors)
        if rest_needed > 0:
            pool = [i for i in range(total) if i not in anchors]
            rng = random.Random(self.sample_seed)
            anchors.extend(rng.sample(pool, min(rest_needed, len(pool))))
        return sorted(anchors)

    def format_trajectory(
        self,
        prompt: Any,
        response_token_ids: Optional[List[int]],
        loss_mask: Optional[List[int]],
        tokenizer: Any,
        **kwargs: Any,
    ) -> str:
        """Render a trajectory as a human-readable string with role separators.

        The initial prompt (a list of ``{"role", "content"}`` chat messages,
        or a plain string) is rendered with one ``[ROLE]\\n{content}`` block
        per message. The generated response is then split into runs based on
        ``loss_mask`` -- mask=1 spans are ``[ASSISTANT]``, mask=0 spans are
        ``[USER/TOOL]`` (the mask alone can't distinguish the two). Override
        for env-specific formatting (e.g. parsing structured tool calls).
        """
        parts: List[str] = []
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            for msg in prompt:
                role = str(msg.get("role", "user")).upper()
                content = msg.get("content", "")
                parts.append(f"[{role}]\n{content}")
        elif prompt is not None:
            parts.append(f"[USER]\n{prompt}")

        if response_token_ids:
            if loss_mask and len(loss_mask) == len(response_token_ids):
                cur_role: Optional[str] = None
                cur_tokens: List[int] = []
                for tok, m in zip(response_token_ids, loss_mask):
                    new_role = "ASSISTANT" if m == 1 else "USER/TOOL"
                    if cur_role is None:
                        cur_role = new_role
                        cur_tokens = [tok]
                    elif new_role == cur_role:
                        cur_tokens.append(tok)
                    else:
                        decoded = tokenizer.decode(cur_tokens, skip_special_tokens=True)
                        parts.append(f"[{cur_role}]\n{decoded}")
                        cur_role = new_role
                        cur_tokens = [tok]
                if cur_tokens and cur_role is not None:
                    decoded = tokenizer.decode(cur_tokens, skip_special_tokens=True)
                    parts.append(f"[{cur_role}]\n{decoded}")
            else:
                decoded = tokenizer.decode(response_token_ids, skip_special_tokens=True)
                parts.append(f"[ASSISTANT]\n{decoded}")

        return "\n\n".join(parts)

    def count_assistant_turns(self, loss_mask: Optional[List[int]]) -> int:
        """Count contiguous 1-spans in ``loss_mask`` (each = one assistant turn).

        Default for the ``num_turns`` column when the caller doesn't supply
        one. Returns 0 for empty/None masks.
        """
        if not loss_mask:
            return 0
        turns = 0
        prev = 0
        for m in loss_mask:
            if m == 1 and prev == 0:
                turns += 1
            prev = m
        return turns
