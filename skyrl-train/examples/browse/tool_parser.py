import ast
import json
from typing import Any

import regex as re

# from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


class GLM4ToolParser:
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
            return {"tools_called": False, "tool_calls": []}
        else:
            if len(tool_calls) > 0:
                return {"tools_called": True, "tool_calls": tool_calls}
            return {"tools_called": False, "tool_calls": []}


class Qwen3ToolParser:
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
            return {"tools_called": False, "tool_calls": []}

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

            return {"tools_called": bool(tool_calls), "tool_calls": tool_calls}
