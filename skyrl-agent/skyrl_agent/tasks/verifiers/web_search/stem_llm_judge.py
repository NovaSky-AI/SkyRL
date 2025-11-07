"""
Minimal STEM judge: every answer is graded by an external LLM.

Prerequisite:
  - Set env var STEM_LLM_JUDGE_URL to an OpenAI-compatible base URL.
  - Launch the service with your preferred model beforehand, e.g.

       
"""

import os
import re
import requests
from typing import Tuple

# # ------------ Prompt template ------------------------------------------------
# JUDGE_TEMPLATE = """\
# You are a strict grader for university-level STEM problems.

# Question:
# {QUESTION}

# Reference Answer:
# {REFERENCE_ANSWER}

# Student Answer:
# {STUDENT_ANSWER}

# Carefully think and check whether the student answer is equivalent to the reference answer.
# You only need to refer to the reference answer to grade the student's answer. Sometimes the student's answer is expressed in a different way from the reference answer, but the meaning is the same, and you should still consider it correct. If they are not equivalent in mathematical sense, you should consider it incorrect.

# <Final Grade>: CORRECT or INCORRECT

# """


# ------------ Core LLM call --------------------------------------------------
def _llm_judge(question: str, student: str, reference: str, verbose: bool = False) -> bool:
    url_base = os.getenv("STEM_LLM_JUDGE_URL")
    if not url_base:
        raise EnvironmentError("STEM_LLM_JUDGE_URL not set")
    url = url_base.rstrip("/") + "/v1/chat/completions"

    # prompt = JUDGE_TEMPLATE.format(
    #     QUESTION=question,
    #     STUDENT_ANSWER=student,
    #     REFERENCE_ANSWER=reference,
    # )

    prompt = (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {reference}\n\n"
        f"### Student Answer: {student}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
    )

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]

    decision = _extract_final_decision(text)
    if decision is None:
        if verbose:
            print("[judge-warning] Unable to parse final decision; treating as incorrect.")
        score = 0.0
    else:
        score = float(decision)

    marker = "✅" if score == 1.0 else "❌"

    if verbose:
        print(marker * 50)
        print("student answer: ", student)
        print("gt: ", reference)
        import json as _json

        print(_json.dumps(data, indent=2, ensure_ascii=False))
        print(marker * 16 + " LLM Judge CONTENT " + marker * 16)
        print(text)
        print(marker * 16 + "End of LLM Judge Reply \n" + marker * 16)

    return score


_FINAL_DECISION_RE = re.compile(r"final\s*decision\s*(?:[:\-=])\s*(yes|no)\b", re.IGNORECASE)


def _extract_final_decision(text: str):
    """Return True for Yes, False for No, or None if no marker is found."""
    matches = list(_FINAL_DECISION_RE.finditer(text))
    if not matches:
        return None
    last_decision = matches[-1].group(1).lower()
    return last_decision == "yes"


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response):
    is_matched = False

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    return is_matched, response


# ------------ Public API -----------------------------------------------------
def compute_score(data_source: str, model_output: str, ground_truth: str, extra_info: dict) -> Tuple[bool, float, str]:
    """
    Arguments
    ---------
    model_output : str   – agent's raw answer
    ground_truth : str   – reference answer
    extra_info   : dict  – MUST contain key "question"

    Returns
    -------
    (is_correct, score, normalized_student_answer)
        score is 1.0 if correct, else 0.0
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    # Try to extract boxed answer; if not found, fall back to full output
    is_matched, extracted_model_output = match_answer(model_output)

    # Exact-match fast path: if dataset specifies exactMatch, do direct comparison
    answer_type = str(extra_info.get("answer_type", "")).lower()
    if answer_type == "exactmatch":
        student = extracted_model_output.strip()
        gt = ground_truth.strip()
        return 1.0 if student == gt else 0.0

    # Otherwise, require boxed for LLM judging (legacy behavior)
    question = extra_info["question"]
    if not is_matched:
        return 0.0
    else:
        try:
            is_correct = _llm_judge(question, extracted_model_output, ground_truth, verbose=False)
        except Exception as e:
            instance_id = extra_info.get("instance_id", "unknown") if isinstance(extra_info, dict) else "unknown"
            print(f"[judge-error] instance_id={instance_id} {e}")
            return 0.0
        return float(is_correct)


# ---------------- Demo -------------------------------------------------------
if __name__ == "__main__":
    demo_item = {
        "question": "A cash-strapped young professional offers to buy your car "
        "with four equal annual payments of $3,000, beginning 2 "
        "years from today. Assuming you can invest at 10% and want "
        "to receive $9,000 for the car, should you accept?",
        "answer": "$8,645.09",
    }
    agent_reply = "The answer is\\boxed{$8,645.09}"  # pretend this comes from your agent
    score = compute_score(
        data_source="",
        model_output=agent_reply,
        ground_truth=demo_item["answer"],
        extra_info={"question": demo_item["question"]},
    )
    print("score =", score)
