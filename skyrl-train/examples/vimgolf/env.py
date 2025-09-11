import difflib
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict

import Levenshtein as levenshtein
import vimgolf.vimgolf
import vimgolf_gym
import vimgolf_gym.dataclasses
import vimgolf_gym.lib
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

# reference:
# - dataset: https://skyrl.readthedocs.io/en/latest/datasets/dataset-preparation.html
# - training: https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html

# TODO: make AI to use fewer keystrokes in each challenge


def assert_docker_privilege():
    assert shutil.which("docker"), "Docker not found in PATH"
    # assert user is in docker group or has permission to run docker without sudo
    assert (
        os.geteuid() == 0 or subprocess.run(["groups"], capture_output=True, text=True).stdout.find("docker") != -1
    ), "User does not have permission to run Docker commands"


def _create_diff_string(content_1: str, content_2: str):
    # Create a Differ object
    differ = difflib.Differ()

    # Compare the lines of the two files
    diff = differ.compare(content_1.splitlines(), content_2.splitlines())

    # Return the differences
    return "\n".join(diff)


def clip_value(value: float, min_value: float, max_value: float):
    assert min_value <= max_value
    return max(min_value, min(value, max_value))


def _calculate_edit_distance_score(input_text: str, output_text: str, buffer_str: str) -> float:
    """
    # should preprocess input_text and output_text with vimgolf.vimgolf.format_
    # clip input score and output score with in [0, 1]
    d_io = levenshtein.distance(input_text, output_text)
    d_ib = levenshtein.distance(input_text, buffer_str)
    d_bo = levenshtein.distance(buffer_str, output_text)
    input_score = 1 - (d_io - d_ib) / d_io
    input_score = clip_value(input_score, 0, 1)
    output_score = (d_io - d_bo) / d_io
    output_score = clip_value(output_score, 0, 1)
    score = input_score * output_score
    """
    input_text = vimgolf.vimgolf.format_(input_text)
    output_text = vimgolf.vimgolf.format_(output_text)
    buffer_str = vimgolf.vimgolf.format_(buffer_str)

    d_io = levenshtein.distance(input_text, output_text)

    if d_io == 0:
        # If input and output are the same, score is 1 if buffer matches, 0 otherwise.
        return 1.0 if buffer_str == output_text else 0.0

    d_ib = levenshtein.distance(input_text, buffer_str)
    d_bo = levenshtein.distance(buffer_str, output_text)
    input_score = 1 - (d_io - d_ib) / d_io
    input_score = clip_value(input_score, 0, 1)
    output_score = (d_io - d_bo) / d_io
    output_score = clip_value(output_score, 0, 1)
    score = input_score * output_score
    return score


def _verify_solution_and_get_feedback(solution: str, input_text: str, output_text: str):
    """
    # should preprocess input_text and output_text with vimgolf.vimgolf.format_
    # should test this function with empty solution and oracle solution
    """
    input_text = vimgolf.vimgolf.format_(input_text)
    output_text = vimgolf.vimgolf.format_(output_text)

    success = False
    buffer_str = None
    diff_str = None
    score = 0.0

    with tempfile.TemporaryDirectory() as tempdir:
        input_file = os.path.join(tempdir, "input.txt")
        output_file = os.path.join(tempdir, "output.txt")
        with open(input_file, "w") as f:
            f.write(input_text)
        with open(output_file, "w") as f:
            f.write(output_text)
        with vimgolf_gym.lib.VimGolfEnv(
            input_file=input_file,
            output_file=output_file,
            init_keys=solution,
            use_docker=True,
            log_buffer=True,
        ) as env:
            for _ in range(3):
                success = env.success
                if success:
                    break
                time.sleep(1)
            buffer = env.buffer
            buffer_str = buffer.decode(encoding="utf-8", errors="replace")
            diff_str = _create_diff_string(output_text, buffer_str)
            if not success:
                with open(output_file, "rb") as f:
                    expected_output = f.read()
                success = buffer == expected_output
                score = _calculate_edit_distance_score(
                    input_text=input_text, output_text=output_text, buffer_str=buffer_str
                )
            else:
                score = 1
    ret = dict(success=success, buffer=buffer_str, diff=diff_str, score=score)
    return ret


def _validate_solution(solution: str, input_text: str, output_text: str):
    input_text = vimgolf.vimgolf.format_(input_text)
    output_text = vimgolf.vimgolf.format_(output_text)
    custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge(input=input_text, output=output_text)
    ret = False
    with vimgolf_gym.make("vimgolf-custom", use_docker=True, custom_challenge=custom_challenge) as env:
        ret = env.verify_keys(solution)
    return ret


def get_last_non_empty_line(content: str):
    lines = content.splitlines()
    lines = [it.strip() for it in lines if it.strip()]
    if lines:
        return lines[-1]
    else:
        return ""


class _VimGolfBaseEnv(BaseTextEnv):
    """vimgolf base (initialized with input text, output text)"""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "input_text" in extras["reward_spec"], "input_text is required in reward_spec field"
        assert "output_text" in extras["reward_spec"], "output_text is required in reward_spec field"
        assert "detail" in extras["reward_spec"], "detail is required in reward_spec field"

        assert_docker_privilege()

        self.input_text = extras["reward_spec"]["input_text"]
        self.output_text = extras["reward_spec"]["output_text"]
        self.detail = extras["reward_spec"]["detail"]


class VimGolfSingleTurnEnv(_VimGolfBaseEnv):
    """vimgolf single turn (program interaction free)"""

    def _get_reward(self, solution: str) -> float:
        validated = _validate_solution(solution=solution, input_text=self.input_text, output_text=self.output_text)
        if validated:
            return 1.0
        else:
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        solution = get_last_non_empty_line(action)
        reward = self._get_reward(solution)

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

    def init(self, prompt: ConversationType) -> tuple[ConversationType, Dict[str, Any]]:

        system_prompt = {
            "role": "system",
            "content": f"""
Vimgolf is a game where you try to transform text using the fewest number of keystrokes in Vim.

Your task is to solve the following Vimgolf challenge with details:
  
Details:
  
{self.detail}

The input file wrapped in triple backticks:
  
```
{self.input_text}
```

The output file wrapped in triple backticks:
  
```
{self.output_text}
```

Your keystokes must be less than the length of output file. Do not naively copy and paste the output file. You must use Vim commands to transform the input file into the output file.

Here are some example solutions, for format demostration (all solutions shall be in one line):

iHello World<Esc>:wq<NL>

:%s/abcdef/defabc/g<NL>:wq<NL>

Your last line of response will be treated as solution. Do not wrap the solution around any marker (like triple backticks), just write it in plain style. Do not write it in multiline style. Do not write any comment or explanation. Do not write any other text. Just write the solution. If your solution contains multiple steps, you will concatenate these steps into one line, optionally using <NL> as separator, depending on the situation.

Example response:

I think the following solution is optimal:

iHello World<Esc>:s/World/Earth/g<NL>:wq<NL>

Please write your solution according to the rules and the example response:
""",
        }
        vimgolf_single_turn_prompt = [system_prompt]
        return vimgolf_single_turn_prompt, {}


class VimGolfMultiTurnEnv(_VimGolfBaseEnv):
    """vimgolf multi turn (program interaction free)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = 5

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        solution = get_last_non_empty_line(action)

        feedback = _verify_solution_and_get_feedback(
            solution=solution, input_text=self.input_text, output_text=self.output_text
        )

        is_correct = feedback["success"]
        score = feedback["score"]
        _ = feedback["buffer"]
        diff = feedback["diff"]

        done = self.turns >= self.max_turns or is_correct

        # Reward structure:
        # - Correct answer: 1.0
        # - Wrong answer: score
        # - No answer: 0.0

        if is_correct:
            reward = 1.0
        elif not solution:
            reward = 0.0
        else:
            reward = score

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True, metadata={"parsed_solution": solution}
            )

        # Give feedback for another attempt

        if solution:
            feedback = f"""
Your solution {solution} is incorrect. Please try again.

Score: {score} (1 for correct, 0 for wrong, intermediate values for partial correctness)

Diff between your solution and the expected output:

{diff}
"""
        else:
            feedback = "Please provide your solution at the last line of the response."

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=reward,
            done=False,
            metadata={"parsed_solution": solution},
        )


# TODO: override the system prompt of the dataset for vimgolf gym env, through super init method parameter changing, or create a dataset for vimgolf gym env


class VimGolfGymEnv(_VimGolfBaseEnv):
    """vimgolf gym (with program interaction)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = 25
        self._custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge(
            input=self.input_text, output=self.output_text
        )
        self.vimgolf_gym_env = vimgolf_gym.make(
            "vimgolf-custom", use_docker=True, log_buffer=True, custom_challenge=self._custom_challenge
        )

    def init(self, prompt: ConversationType) -> tuple[ConversationType, Dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        """
        vimgolf_gym_system_prompt = [
            {
                "role": "system",
                "content": f"""
You are in VimGolf-Gym, a Gym-like environment for solving Vimgolf challenges using the fewest Vim commands.

The environment is TermExec which accepts UTF-8 string with ANSI escape sequences. You can easily input keys like \x1b (Esc), \x0a (Enter), \x7f (Backspace) and so on. You can only interact with the TermExec environment, for solving the Vimgolf challenge.

The input text is:

```
{self.input_text}
```

The expected output text is:

```
{self.output_text}
```

Your action shall be wrapped within <termexec> and </termexec> tags, like:

<termexec>iHello World\x1b:wq\x0a</termexec>

TermExec will provide feedback after each action. We only accept one action at a time. Do not provide multiple actions in one response.
""",
            }
        ]
        return vimgolf_gym_system_prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        termexec_action = ""
        observations = []
        metadata = {}
        reward = 0
        try:
            termexec_action = self._extract_termexec_action_block(action)
        except ValueError as e:
            error_msg = e.args[0]
            observations.append({"role": "system", "content": error_msg})
        if termexec_action:
            reward = 0.5
            # execute the action and collect feedback
            self.vimgolf_gym_env.act(termexec_action)
            terminal_screen = self.vimgolf_gym_env.executor.display()
            buffer_str = self.vimgolf_gym_env.buffer.decode("utf-8", errors="replace")
            edit_distance_score = _calculate_edit_distance_score(
                input_text=self.input_text, output_text=self.output_text, buffer_str=buffer_str
            )

            terminal_feedback = f"""
You have performed a termexec action, now you have the feedback from the terminal:

Terminal Screen:

{terminal_screen}

Estimated edit distance score (1 for perfect match, 0 for worst match): {edit_distance_score}
"""
            observations.append({"role": "system", "content": terminal_feedback})
        else:
            # hint the agent to provide a valid action
            observations.append({"role": "system", "content": "Please provide a valid action. Your action is empty."})
        is_success = self.vimgolf_gym_env.success
        if is_success:
            reward = 1
            metadata["parsed_solution"] = self.vimgolf_gym_env.get_best_success_result().keys
        done = self.turns >= self.max_turns or is_success
        return BaseTextEnvStepOutput(observations=observations, reward=reward, done=done, metadata=metadata)

    @staticmethod
    def _extract_termexec_action_block(action: str) -> str:
        """extract the termexec action block from the action string"""
        if "<termexec>" not in action or "</termexec>" not in action:
            raise ValueError("Action must be wrapped within <termexec> and </termexec> tags")
        if action.count("<termexec>") > 1 or action.count("</termexec>") > 1:
            raise ValueError("Action must contain only one <termexec> and one </termexec> tag")
        return action.split("<termexec>")[1].split("</termexec>")[0]

    def close(self):
        """close the environment"""
        self.vimgolf_gym_env.close()
