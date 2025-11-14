import json
import copy
import os
from typing import Any, List, Dict
from collections import defaultdict

from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.base import AsyncInferBackend
from skyrl_agent.tools.base import TOOL_REGISTRY
from skyrl_agent.dispatcher.async_utils import call_sync_from_async
from .messages import (
    TOOL_CALL_PARSE_ERROR_GUIDANCE,
    NO_TOOL_CALL_DETECTED_GUIDANCE,
    TOOL_INVOCATION_ERROR_GUIDANCE,
)
from uuid import uuid4

from openhands.llm.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.core.exceptions import (
    FunctionCallConversionError,
    FunctionCallValidationError,
)


class ReActAgent:
    def __init__(
        self,
        traj_config: TrajectoryConfig,
        infer_engine: AsyncInferBackend,
        tokenizer: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.infer_engine = infer_engine
        self.sampling_params = traj_config.sampling_params

        self.max_prompt_length = traj_config.max_prompt_length
        self.qwen3_enable_thinking = traj_config.qwen3_enable_thinking
        self.qwen3_acc_thinking = traj_config.qwen3_acc_thinking

        self.instance_id = traj_config.instance_id
        self.trajectory_id = traj_config.trajectory_id
        self.max_iterations = traj_config.max_iterations

        self.step_count = 0
        self.messages: List[dict] = []
        self.tools = {}
        self.tool_params = []

        self.agent_id = uuid4().hex
        self._register_tools(traj_config.tools)

        self.prompt_token_len = 0
        self.response_token_len = 0

        # Debug and profiling flags/counters
        self._debug = bool(traj_config.debug_log)
        if not self._debug:
            self._debug = os.getenv("SKYAGENT_DEBUG_LOG", "0") == "1"
        self._profile_enabled = bool(traj_config.profile_tools)
        if not self._profile_enabled:
            self._profile_enabled = os.getenv("SKYAGENT_PROFILE_TOOLS", "0") == "1"
        self._tool_calls_total: int = 0
        self._tool_calls_by_name: Dict[str, int] = defaultdict(int)

    def _register_tools(self, tools: List[str]) -> None:
        """Register a list of tool instances."""
        print(f"[Register Tools] {tools}")
        for name in tools:
            if name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool '{name}'. Must be one of: {list(TOOL_REGISTRY)}")
            tool = TOOL_REGISTRY[name]()
            self.tools[tool.name] = tool
            self.tool_params.append(tool.get_tool_param())

    async def step(self):
        done = False
        finish_reason = None

        self.step_count += 1
        print(f"[Agent Step {self.step_count}] instance={self.instance_id} traj={self.trajectory_id}")

        formatted_messages = convert_fncall_messages_to_non_fncall_messages(
            self.messages, self.tool_params, add_in_context_learning_example=False
        )
        # print(f"[Agent Step {self.step_count}] Formatted messages: {formatted_messages}, messages: {self.messages}")

        input_ids = self.tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.qwen3_enable_thinking,
        )
        if len(self.messages) == 2:
            self.prompt_token_len = len(input_ids)
        self.response_token_len = len(input_ids) - self.prompt_token_len

        if self.response_token_len >= self.max_prompt_length:
            # raise ValueError(
            #     f"Input length {len(input_ids)} exceeds max prompt length {self.max_prompt_length}. "
            #     "Please reduce the input size or increase the max prompt length."
            # )
            # For now, we will just stop the agent if the input length exceeds the max prompt length.
            # print(f"[Agent Step] Input length {len(input_ids)} exceeds max prompt length {self.max_prompt_length}. Stopping agent.")
            done = True
            finish_reason = "CONTEXT_WINDOW_EXCEEDED"
            return done, finish_reason, None
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params["max_tokens"] = self.max_prompt_length - self.response_token_len

        try:
            response_str, stop_reason = await self.infer_engine.async_generate_ids(
                input_ids=input_ids,
                sampling_params=sampling_params,
                request_id=self.agent_id,
            )
            # assert False, "Dacheng: check whether needs to do call_sync_from_async here."
            # print(f"[Agent Step {self.step_count}] [LLM Response] {response_str} length {len(response_str)}")

            assistant_msg = {"role": "assistant", "content": response_str}
            self.messages.append(assistant_msg)
            if stop_reason == "length":
                print(f"[Agent Step] Stopping reason: {stop_reason}. Stopping agent.")
                done = True
                finish_reason = "CONTEXT_WINDOW_EXCEEDED"
                return done, finish_reason, None

            # Convert assistant text back into a structured tool call; if parsing fails
            # due to schema/type violations, guide the model to retry with correct format.
            try:
                fncall_messages = convert_non_fncall_messages_to_fncall_messages([assistant_msg], self.tool_params)
            except (FunctionCallValidationError, FunctionCallConversionError) as e:
                err = str(e)
                print(f"[Agent Step Error] Converter failed to parse tool call: {err}")
                guidance = TOOL_CALL_PARSE_ERROR_GUIDANCE.format(error=err)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"error": err}),
                    }
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": guidance,
                    }
                )
                return done, finish_reason, None

            # if no tools provided, we can just return the response
            if not self.tools:
                print(f"[Agent Step {self.step_count}] No tools provided, returning response.")
                done = True
                finish_reason = "FINISH"
                return done, finish_reason, response_str

            tool_call = fncall_messages[0].get("tool_calls", [None])[0]
            if tool_call is None:
                print(f"[Agent Step {self.step_count}] No tool call found in response")
                # print(f"[Agent Step {self.step_count}] Response content: {response_str[:1000]}")  # Hidden: may contain thinking

                # Check if response was likely truncated during a tool call
                if "<function=" in response_str and not response_str.strip().endswith("</function>"):
                    print("[ERROR] Tool call appears incomplete - likely truncated!")
                    print(f"[ERROR] Last 500 chars: {response_str[-500:]}")

                # Provide targeted guidance including the exact tag format required by the converter
                self.messages.append(
                    {
                        "role": "user",
                        "content": NO_TOOL_CALL_DETECTED_GUIDANCE,
                    }
                )
                return done, finish_reason, None
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
            if tool_name not in self.tools:
                self.messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"error": f"Tool '{tool_name}' not found."}),
                    }
                )
                return done, finish_reason, response_str

            tool = self.tools[tool_name]
            # print(f"[Tool Dispatch] Calling tool: {tool_name} with args: {tool_args}")
            try:
                output = await call_sync_from_async(
                    tool.call,
                    tool_args,
                    trajectory_id=self.trajectory_id,
                )
                # Successful tool call: record profiling stats if enabled
                if self._profile_enabled:
                    try:
                        self._tool_calls_total += 1
                        if tool_name:
                            self._tool_calls_by_name[tool_name] += 1
                    except Exception:
                        pass
            except Exception as e:
                # Tool invocation failed (likely invalid arguments) – append error and guide model to retry
                try:
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps({"error": str(e)}),
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
                except Exception:
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps({"error": "Tool failed with an exception."}),
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": TOOL_INVOCATION_ERROR_GUIDANCE,
                    }
                )
                return done, finish_reason, None
            # append observations to messages
            try:
                self.messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(output),
                        "tool_call_id": tool_call.get("id"),
                    }
                )
                if self._debug:
                    try:
                        out_str = json.dumps(output)
                    except Exception:
                        out_str = str(output)
                    out_str = out_str.replace("\n", " ")
                    if len(out_str) > 400:
                        out_str = out_str[:400] + "..."
                    print(f"[Tool Output Preview] {out_str}")
            except Exception as e:
                print(f"[Agent Step Error] Error appending tool output to messages: {str(e)}")
                self.messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}),
                        "tool_call_id": tool_call.get("id"),
                    }
                )
            # if it is a finish tool, we can stop the agent
            if tool_name == "finish":
                print("[Agent Step] Finish tool called. Stopping agent.")
                done = True
                finish_reason = "FINISH_TOOL"
                return done, finish_reason, output

            # For non-finish tools, continue the agent loop
            return done, finish_reason, response_str

        except Exception as e:
            print(f"[Agent Step Error] Error during step: {str(e)}")
            done = True
            finish_reason = f"error: {str(e)}"
            return done, finish_reason, None

    async def run(self, instruction: List[Dict]) -> List[str]:
        """Run the agent till the end with the provided user input."""
        self._init_message(instruction)
        result = None
        finish_reason = None
        while self.step_count < self.max_iterations:
            try:
                done, finish_reason, result = await self.step()
                if done:
                    break
            except Exception as e:
                finish_reason = f"error: {str(e)}"
                print(f"[Agent Run Error] Exception during step: {str(e)}")
                break
        else:  # If we exit the loop without hitting a break, it means we reached max iterations
            finish_reason = "max_iterations_reached"

        return finish_reason, result

    def get_messages(self) -> List[dict]:
        return convert_fncall_messages_to_non_fncall_messages(
            self.messages, self.tool_params, add_in_context_learning_example=False
        )
        # return self.messages

    def _init_message(self, instruction: List[Dict]) -> None:
        """Initialize the agent's message history with the provided instruction."""
        self.messages = []
        if not isinstance(instruction, list):
            raise ValueError("Instruction must be a list of messages.")

        for msg in instruction:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must be a dictionary with 'role' and 'content'.")
            self.messages.append(msg)

    # Expose profiling snapshot for upstream aggregation
    def get_tool_profile(self) -> Dict[str, Any]:
        if not self._profile_enabled:
            return None
        try:
            return {
                "tool_calls_total": int(self._tool_calls_total),
                "tool_calls_by_name": dict(self._tool_calls_by_name),
            }
        except Exception:
            return None


if __name__ == "__main__":
    # Example usage for testing
    from skyrl_agent.config.configuration_utils import TrajectoryConfig
    from skyrl_agent.integrations.openai import OpenAIBackend, OpenAIBackendConfig
    from transformers import AutoTokenizer
    import asyncio

    # Load tokenizer and model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define trajectory configuration
    traj_config = TrajectoryConfig(
        instance_id="test_instance",
        trajectory_id="test_trajectory",
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
        max_prompt_length=12048,
        qwen3_enable_thinking=True,
        tools=["finish", "code_interpreter"],
        max_iterations=5,
        agent_cls="skyrl_agent.agents.react.ReActAgent",  # Use ReActAgent for testing
    )

    backend_config = OpenAIBackendConfig(
        model_name=model_name,
        # change this to your desired url and port
        api_url="http://localhost:8000",
    )
    # TODO: model_name need not be in config
    infer_engine = OpenAIBackend(infer_engine=None, cfg=backend_config)

    # Create the ReAct agent
    # Test for with tools
    agent = ReActAgent(
        traj_config=traj_config,
        infer_engine=infer_engine,
        tokenizer=tokenizer,
    )

    # Define a sample instruction
    instruction = [
        {"content": "Please reason step by step, and put your final answer within \\boxed{}.", "role": "system"},
        {
            "content": "Points $A,B,C,D,E$ and $F$ lie, in that order, on $\\overline{AF}$, dividing it into five segments, each of length 1. Point $G$ is not on line $AF$. Point $H$ lies on $\\overline{GD}$, and point $J$ lies on $\\overline{GF}$. The line segments $\\overline{HC}, \\overline{JE},$ and $\\overline{AG}$ are parallel. Find $HC/JE$.",
            "role": "user",
        },
    ]

    # Run the agent
    finish_reason, result = asyncio.run(agent.run(instruction))

    print(agent.get_messages())
    print(f"Finish Reason: {finish_reason}")
    print(f"Result: {result}")
