"""Collecting distillation data from a teacher and a student in one run.

The same task is put to two models, each working through it in a short loop.
That is the shape a distillation dataset is gathered in, and it is also the
case where capture has to be careful: when the two models answer identically --
which on easy turns they will -- their generations are still two samples, from
two models, and a dataset that means to train one of them or to distil one into
the other has to be able to tell them apart.

The first turn is pinned to the same reply on purpose, so nothing except the
model can account for a divergence there. Later turns differ as the two models
would.

    skyrl-capture run --project distill \
      -- python scripts/model_distill.py

    skyrl-capture view

Two roots, one per model, each carrying its own path. Exporting gives one row
per model, and `model` on the row is what a consumer filters on:

    skyrl-capture export --trajectory <id> --format text-samples --output samples.jsonl
    jq -r 'select(.model == "teacher-70b") | .path_id' samples.jsonl
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]

TEACHER = "teacher-70b"
STUDENT = "student-7b"

TASK = "Explain why the auth test fails."

# (prompt, teacher reply, student reply). The first is deliberately identical:
# it is the turn that would have collapsed into a single node.
TURNS = [
    (TASK, "The token expiry check is inverted.", "The token expiry check is inverted."),
    (
        "What is the minimal fix?",
        "Flip the comparison in verify_expiry and add a regression test.",
        "Change the comparison operator.",
    ),
]


def chat(model: str, messages: list[dict[str, str]], reply: str) -> str:
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": 128, "temperature": 0.2}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            # Mock-only: pins what each model "says" so the run is reproducible.
            "x-mock-reply": reply,
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def run(model: str, pick: int) -> list[str]:
    """Work the task with one model, carrying the conversation forward."""
    messages: list[dict[str, str]] = []
    answers: list[str] = []
    for prompt, *replies in TURNS:
        messages.append({"role": "user", "content": prompt})
        answer = chat(model, messages, replies[pick])
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)
    return answers


def main() -> None:
    teacher = run(TEACHER, pick=0)
    student = run(STUDENT, pick=1)

    for name, answers in ((TEACHER, teacher), (STUDENT, student)):
        print(f"{name}:")
        for answer in answers:
            print(f"    {answer}")

    assert teacher[0] == student[0], "turn 1 is pinned identical, so only the model differs"
    print(
        "\nturn 1 is word-for-word the same from both models."
        "\nthe graph should still show two roots, and the export two rows."
    )


if __name__ == "__main__":
    main()
