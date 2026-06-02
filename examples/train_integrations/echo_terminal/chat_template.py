from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

_PROBE_SENTINEL = "OBSERVATION_PROBE_TOKEN_ABCXYZ_12345"
_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "You are a helpful agent with access to bash."},
    {"role": "user", "content": "Run `ls /tmp` and report the result."},
    {
        "role": "assistant",
        "content": (
            "<think>\nI should call the bash tool with the command.\n</think>\n\n"
            "<tool_call>\n<function=bash>\n<parameter=command>\nls /tmp\n"
            "</parameter>\n</function>\n</tool_call>"
        ),
    },
]


def check_role_roundtrip(
    tokenizer: PreTrainedTokenizerBase,
    role: str,
    *,
    template_kwargs: dict[str, Any] | None = None,
    sentinel: str = _PROBE_SENTINEL,
) -> tuple[bool, str]:
    template_kwargs = template_kwargs or {}
    try:
        before = tokenizer.apply_chat_template(
            _PROBE_MESSAGES,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        after = tokenizer.apply_chat_template(
            _PROBE_MESSAGES + [{"role": role, "content": sentinel}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
    except Exception as exc:
        return False, f"apply_chat_template raised {type(exc).__name__}: {exc}"

    if after[: len(before)] != before:
        return False, f"prefix mismatch when {role!r} was appended"
    delta = after[len(before) :]
    if not delta:
        return False, "empty delta"
    decoded = tokenizer.decode(delta, skip_special_tokens=False)
    if sentinel not in decoded:
        return False, f"sentinel content not found for role {role!r}"
    return True, ""


def choose_obs_role(
    tokenizer: PreTrainedTokenizerBase,
    candidates: tuple[str, ...] = ("tool", "user"),
    *,
    template_kwargs: dict[str, Any] | None = None,
) -> str:
    failures: list[str] = []
    for role in candidates:
        ok, reason = check_role_roundtrip(tokenizer, role, template_kwargs=template_kwargs)
        if ok:
            return role
        failures.append(f"  - {role!r}: {reason}")
    raise RuntimeError(
        f"None of {list(candidates)!r} produced a usable observation role for "
        f"tokenizer {tokenizer.name_or_path!r}. Failures:\n" + "\n".join(failures)
    )


def resolve_chat_template_path(template_path: str | Path) -> Path:
    path = Path(template_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        module_path = Path(__file__).resolve()
        candidates = [
            Path.cwd() / path,
            module_path.parent / path,
            module_path.parents[2] / path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Chat template not found at {template_path!r}; checked {candidates!r}")


def load_chat_template(tokenizer: PreTrainedTokenizerBase, template_path: str | Path) -> None:
    tokenizer.chat_template = resolve_chat_template_path(template_path).read_text()
