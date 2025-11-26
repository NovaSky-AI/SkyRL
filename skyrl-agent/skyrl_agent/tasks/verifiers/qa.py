# From https://github.com/PeterGriffinJin/Search-R1/blob/9828adecbbdb184333e4d8ca9d6b8bd10275da44/verl/utils/reward_score/qa_em.py#L36
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import litellm
import json

JUDGE_PROMPT_BROWSECOMP_OFFICIAL = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

JUDGE_PROMPT_RULER_OFFICIAL = """Does the [response] correctly answer the [question] based on [correct_answer]?

[question]: {question}
[correct_answer]: {correct_answer}
[response]: {response}

Answer 'yes' if the response matches the correct answer, 'no' otherwise.
""".strip()

ruler_answer_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "simple_judgment",
        "schema": {
            "type": "object",
            "properties": {
                "correct": {"type": "string", "enum": ["yes", "no"]},
            },
            "required": ["correct"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # remove \boxed{} if it exists
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def bool_mapping(s):
    """Map boolean strings to yes/no format"""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def contains_chinese(text):
    """Check if the given text contains Chinese characters"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3400" <= char <= "\u4dbf":
            return True
        if "\uf900" <= char <= "\ufaff":
            return True
    return False


def normalize_text(text: str) -> str:
    """Normalize text for F1 scoring"""
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def f1_score(answer_content, gt):
    """Compute F1 score between answer and ground truth"""
    answer_content = normalize_text(bool_mapping(answer_content))
    gt = normalize_text(bool_mapping(gt))

    if contains_chinese(gt):

        def parse_chinese_str(s):
            numbers = []
            for i, c in enumerate(s):
                if c.isdigit():
                    if i > 0 and s[i - 1].isdigit():
                        numbers[-1] = numbers[-1] + c
                    else:
                        numbers.append(c)
            for c in "0123456789，。 ,.-":
                s = s.replace(c, "")
            return set(list(s) + numbers)

        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(gt)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt.split())

    if not gt_tokens or not pred_tokens:
        return 0

    common_tokens = pred_tokens & gt_tokens
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)

    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def compute_score_f1(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Compute F1 score - handles both single targets and lists"""
    target = ground_truth["target"]

    if solution_str is None:
        return {"score": 0}

    # Handle numpy arrays and lists
    if hasattr(target, "tolist"):
        target = target.tolist()

    # Handle lists by computing F1 against each and taking max
    if isinstance(target, list):
        scores = []
        for gt in target:
            scores.append(f1_score(solution_str, gt))
        return {"score": max(scores) if scores else 0}

    # Single target case
    return {"score": f1_score(solution_str, target)}


def compute_score_em(solution_str, ground_truth, format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Solution string: {solution_str}")

    if solution_str is None:
        return {"score": 0}
    else:
        if em_check(solution_str, ground_truth["target"]):
            return {"score": score}
        else:
            return {"score": format_score}


# use llm as a verifier
def compute_score_browsecomp(solution_str, ground_truth, question):
    """The scoring function for LLM"""

    if isinstance(ground_truth["target"], list):
        assert len(ground_truth["target"]) == 1, "Only one correct answer is supported for browsecomp"
        ground_truth["target"] = ground_truth["target"][0]

    if solution_str is None:
        return {"score": 0}
    else:
        # use llm as a verifier
        prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
            question=question, response=solution_str, correct_answer=ground_truth["target"]
        )
        print(f"Prompt for judge model: {prompt}")
        response = litellm.completion(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            num_retries=5,
            response_format=extracted_answer_format_for_confidence,
        )
        print(f"Response from judge model: {response}")
        raw_content = response.choices[0].message["content"]
        raw_judge = json.loads(raw_content)
        judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""

        return {"score": 1 if judgement == "Correct" else 0}


# use llm as a verifier
def compute_score_ruler(solution_str, ground_truth, question):
    """The scoring function for LLM"""
    if solution_str is None:
        return {"score": 0}
    else:
        # use llm as a verifier
        prompt = JUDGE_PROMPT_RULER_OFFICIAL.format(
            question=question, response=solution_str, correct_answer=ground_truth
        )
        print(f"Prompt for judge model: {prompt}")
        response = litellm.completion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            num_retries=1,
            response_format=ruler_answer_format,
        )
        print(f"Response from judge model: {response}")
        raw_content = response.choices[0].message["content"]
        print(f"Raw content from judge model: {raw_content}")
        raw_judge = json.loads(raw_content)
        judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""

        return {"score": 1 if judgement == "Correct" else 0}
