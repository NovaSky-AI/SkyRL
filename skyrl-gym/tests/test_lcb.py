import pytest
import skyrl_gym
import json
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "model_response, tests, expected_reward",
    [
        # Correct code: second largest index
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                    {"input": "3\n3 2 1\n", "output": "2\n", "testtype": "stdin"},
                ]
            ),
            1.0,
        ),
        # Wrong logic: returns index of largest
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    print(A.index(max(A)) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
        # Missing main() call — runtime error
        (
            """```python
def main():
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

# forgot to call main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
    ],
)
def test_compute_score(model_response, tests, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": tests}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert step_output["reward"] == expected_reward


def _functional_tests(func_name, cases):
    metadata = {"func_name": func_name}
    return json.dumps(
        [{"input": inp, "output": out, "testtype": "functional", "metadata": metadata} for inp, out in cases]
    )


# Sum of the k largest numbers.
MAX_K_SUM_TESTS = _functional_tests("maxKSum", [("[3, 1, 4, 1, 5]\n2", "9"), ("[10, -2, 7]\n1", "10")])
# The k largest numbers, largest first.
TOP_K_TESTS = _functional_tests("topK", [("[3, 1, 4, 1, 5]\n2", "[5, 4]"), ("[10, -2, 7]\n1", "[10]")])
# Count of each character.
COUNT_CHARS_TESTS = _functional_tests("countChars", [('"aab"', '{"a": 2, "b": 1}'), ('"z"', '{"z": 1}')])
# Consecutive pairs.
PAIRS_TESTS = _functional_tests("pairs", [("[1, 2, 3, 4]", "[[1, 2], [3, 4]]"), ("[5, 6]", "[[5, 6]]")])


@pytest.mark.parametrize(
    "solution, tests, expected_reward",
    [
        (
            """
class Solution:
    def maxKSum(self, nums: List[int], k: int) -> int:
        return sum(sorted(nums, reverse=True)[:k])
""",
            MAX_K_SUM_TESTS,
            1.0,
        ),
        (
            """
class Solution:
    def maxKSum(self, nums: List[int], k: int) -> int:
        return sum(sorted(nums)[:k])
""",
            MAX_K_SUM_TESTS,
            0.0,
        ),
        # An object whose __eq__ always returns True must not pass.
        (
            """
class _AlwaysEqual:
    def __eq__(self, other):
        return True

class Solution:
    def maxKSum(self, nums: List[int], k: int) -> int:
        return _AlwaysEqual()
""",
            MAX_K_SUM_TESTS,
            0.0,
        ),
        # Same for an int subclass, which looks like an ordinary return value.
        (
            """
class _Z(int):
    def __eq__(self, other):
        return True
    __hash__ = int.__hash__

class Solution:
    def maxKSum(self, nums: List[int], k: int) -> int:
        return _Z(0)
""",
            MAX_K_SUM_TESTS,
            0.0,
        ),
        # A tuple where the expected output is a list still passes.
        (
            """
class Solution:
    def topK(self, nums: List[int], k: int) -> List[int]:
        return tuple(sorted(nums, reverse=True)[:k])
""",
            TOP_K_TESTS,
            1.0,
        ),
        # A Counter where the expected output is a dict still passes.
        (
            """
from collections import Counter

class Solution:
    def countChars(self, s: str) -> dict:
        return Counter(s)
""",
            COUNT_CHARS_TESTS,
            1.0,
        ),
        # A list of numpy integers where the expected output is a list still passes.
        (
            """
class Solution:
    def topK(self, nums: List[int], k: int) -> List[int]:
        return list(np.array(sorted(nums, reverse=True)[:k]))
""",
            TOP_K_TESTS,
            1.0,
        ),
        # Nested tuples where the expected output is a list of lists pass.
        (
            """
class Solution:
    def pairs(self, nums: List[int]) -> List[List[int]]:
        return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
""",
            PAIRS_TESTS,
            1.0,
        ),
        # A value with no JSON form, such as a set, fails.
        (
            """
class Solution:
    def topK(self, nums: List[int], k: int) -> List[int]:
        return set(sorted(nums, reverse=True)[:k])
""",
            TOP_K_TESTS,
            0.0,
        ),
    ],
)
def test_compute_score_call_based(solution, tests, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": tests}},
    )
    step_output = env.step("```python\n" + solution.strip("\n") + "\n```")
    assert step_output["reward"] == expected_reward
