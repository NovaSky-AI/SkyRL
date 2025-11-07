# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# %%
import ast
import json
from typing import Any

import regex as re

# from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


class Glm4MoeModelToolParser:
    def __init__(self):
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

        self._buffer = ""

    def extract_tool_calls(
        self,
        model_output: str,
    ):

        def _deserialize(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass

            try:
                return ast.literal_eval(value)
            except Exception:
                pass
            return value

        matched_tool_calls = self.func_call_regex.findall(model_output)
        # logger.debug("model_output: %s", model_output)
        try:
            tool_calls = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                tc_name = tc_detail.group(1)
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args)
                arg_dct = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    arg_val = _deserialize(arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append({"name": tc_name, "arguments": arg_dct})
        except Exception:
            return {"tools_called": False, "tool_calls": [], "content": model_output}
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return {"tools_called": True, "tool_calls": tool_calls, "content": content}
            return {"tools_called": False, "tool_calls": [], "content": model_output}


class Qwen3XMLToolParser:
    def __init__(self):

        # Add missing attributes for compatibility with serving_chat.py
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        self.func_call_regex = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

    def extract_tool_calls(
        self,
        model_output: str,
    ):

        result = self.func_call_regex.findall(model_output)

        if not result:
            return {"tools_called": False, "tool_calls": [], "content": model_output}

        else:
            tool_calls = []
            for tool_call in result:
                try:
                    evaled_tool_dict = json.loads(tool_call)
                except Exception:
                    try:
                        evaled_tool_dict = ast.literal_eval(tool_call)
                    except Exception:
                        evaled_tool_dict = None

                if evaled_tool_dict and evaled_tool_dict.get("name") and evaled_tool_dict.get("arguments"):
                    tool_calls.append(evaled_tool_dict)

                content = model_output[: model_output.find(self.tool_call_start_token)]

            return {"tools_called": bool(tool_calls), "tool_calls": tool_calls, "content": content}


# %%
# class Qwen25Detector():
#     """
#     Detector for Qwen 2.5 and Qwen 3 model function call format.

#     Format Structure:
#     ```
#     <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
#     ```

#     Key Components:
#     - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
#     - Function Call Object: JSON object with "name" and "arguments" fields

#     Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
#     """

#     def __init__(self):
#         """
#         Initializes the detector with necessary state variables.
#         """
#         super().__init__()
#         self.bot_token = "<tool_call>\n"
#         self.eot_token = "\n</tool_call>"
#         self.tool_call_separator = "\n"
#         self._normal_text_buffer = ""  # Buffer for handling partial end tokens

#     def has_tool_call(self, text: str) -> bool:
#         """Check if the text contains a Qwen 2.5 format tool call."""
#         return self.bot_token in text

#     def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
#         """
#         One-time parsing: Detects and parses tool calls in the provided text.

#         :param text: The complete text to parse.
#         :param tools: List of available tools.
#         :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
#         """
#         idx = text.find(self.bot_token)
#         normal_text = text[:idx].strip() if idx != -1 else text
#         if self.bot_token not in text:
#             return StreamingParseResult(normal_text=normal_text, calls=[])

#         # Find all <tool_call>\n...\n</tool_call> blocks
#         pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
#         match_result_list = re.findall(pattern, text, re.DOTALL)
#         calls = []
#         for match_result in match_result_list:
#             try:
#                 parsed_call = json.loads(match_result.strip())
#                 calls.extend(self.parse_base_json(parsed_call, tools))
#             except json.JSONDecodeError as e:
#                 logger.warning(
#                     f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
#                 )
#                 continue
#         return StreamingParseResult(normal_text=normal_text, calls=calls)
