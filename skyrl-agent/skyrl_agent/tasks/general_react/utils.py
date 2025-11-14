from typing import Dict, Any
import asyncio

from skyrl_agent.tasks.base import BaseTask


class GeneralReactTask(BaseTask):
    @classmethod
    async def initialize_runtime(cls):
        pass

    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> str:
        print(instance)
        # (TODO) Dacheng: A hack to make inference only compatible.
        # During inference, the key is "prompt"
        # During training, the key is "raw_prompt"
        if "raw_prompt" in instance:
            import numpy

            assert isinstance(
                instance.get("raw_prompt"), numpy.ndarray
            ), f"Raw prompt must be a numpy array, but got {type(instance.get('raw_prompt'))}"
            assert (
                len(instance.get("raw_prompt")) == 1
            ), f"Raw prompt must have only one element, but got {len(instance.get('raw_prompt'))}"
            prompt = list(instance.get("raw_prompt"))
        else:
            prompt = instance.get("prompt")

        print(f"Prompt: {prompt}")

        assert isinstance(prompt, list), f"Prompt must be a list, but got {type(prompt)}"

        # Check if there's already a system message
        has_system = any(msg.get("role") == "system" for msg in prompt)
        data_source = instance.get("data_source", "")

        if not has_system:
            # Prepend a dummy system prompt so that the agent can use tools by the convert_fncall_messages_to_non_fncall_messages function.
            # From Retool prompt: https://arxiv.org/pdf/2504.11536
            if data_source.startswith("math"):
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools and return the final answer inside the finish tool. \
                    If there are additional requirments such as the answer should be included inside \\boxed{}, please return the answer in the format of \
                    <function=finish> \
                    <parameter=answer>\\boxed{'The final answer goes here.'}</parameter> \
                    </function>",
                }
            elif data_source.startswith("codegen"):
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools and return the final answer inside the finish tool. \
                    Given a coding task, you can use the code interpreter tool to run code to test your solution. \
                    Please return the final code solution in the format of \
                    <function=finish> \
                    <parameter=answer>```python\n# The final answer goes here.\n```</parameter> \
                    </function>",
                }
            else:
                assert False, f"Data source {data_source} is not supported for ReAct agent."
            prompt = [system_prompt] + prompt

        print(f"Prompt after system prompt: {prompt}")
        return prompt

    @classmethod
    def complete_runtime(cls):
        pass

    @classmethod
    async def evaluate_result(
        cls, result: any, instance: any, data_source: str, instance_id: int, trajectory_id: int
    ) -> float:
        # print(f"Evaluating result: {result=} {instance=} {data_source=} {instance_id=} {trajectory_id=}")
        ground_truth = instance["reward_model"]["ground_truth"]
        extra_info = instance["extra_info"]
        print(f"Evaluating result: {result=}")
        if not result:
            return 0.0
        if data_source == "ToRL":
            from skyrl_agent.tasks.verifiers import torl

            return torl.compute_score(result, ground_truth)
        elif data_source.startswith("math"):
            from skyrl_agent.tasks.verifiers import naive_dapo

            print(f"Evaluating math task with data_source: {data_source}, got {result=} {ground_truth=} {extra_info=}")
            res = naive_dapo.compute_score(result, ground_truth, extra_info=extra_info)
            print(f"Evaluated math task with data_source: {data_source}, got {res=}")
            return res["score"]
        # code generation
        elif data_source.startswith("codegen"):
            from skyrl_agent.tasks.verifiers import coder1

            # print(f"Getting result {result}")
            print(f"Evaluating codegen task with data_source: {data_source}, got {result=}")
            res = coder1.compute_score(result, ground_truth, extra_info=extra_info)
            print(f"Evaluated codegen task with data_source: {data_source}, got {res=}")
            # print(f"Evaluating codegen task with data_source: {data_source}, got {score=} {extracted_model_output=}")
            print(f"Evaluating codegen task with data_source: {data_source}, got {res['score']=}")
            return res["score"]
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])


class DummyReactTask(GeneralReactTask):
    # For Dummy ReAct Agent that does not use tools
    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> str:
        prompt = instance.get("prompt")
        return prompt


if __name__ == "__main__":
    # Simple test
    # ToRL
    solution_str = "\\boxed{1}"
    ground_truth = "1"
    instance = {"reward_model": {"ground_truth": ground_truth}, "extra_info": {}}
    print(asyncio.run(GeneralReactTask.evaluate_result(solution_str, instance, "math", 0, 0)))
