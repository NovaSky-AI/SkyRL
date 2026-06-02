import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedCommand:
    name: str = ""
    arguments: dict = field(default_factory=dict)
    error: str = ""
    raw: str = ""


@dataclass
class ParseResult:
    thinking: str = ""
    commands: list[ParsedCommand] = field(default_factory=list)
    is_done: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def check_format_warnings(
    raw_response: str,
    tags: list[str] | None = None,
    parser_name: str = "xml",
    thinking_enabled: bool = True,
) -> tuple[list[str], list[str]]:
    if not raw_response:
        return [], []
    if thinking_enabled and parser_name == "qwen35":
        raw_response = "<think>\n" + raw_response

    tags = tags or ["command", "action"]
    all_tags = ["think"] + [t for t in tags if t != "think"]
    tag_alternation = "|".join(re.escape(t) for t in all_tags)
    tag_re = re.compile(rf"<(/?)({tag_alternation})>")

    warnings: list[str] = []
    violations: list[str] = []

    if thinking_enabled:
        think_opens = len(re.findall(r"<think>", raw_response))
        if think_opens == 0:
            warnings.append("Missing <think> block")
            violations.append("missing_think")
        elif think_opens > 1:
            warnings.append(f"Found {think_opens} <think> blocks, expected 1")
            violations.append("multiple_think")

    for tag in all_tags:
        opens = len(re.findall(f"<{re.escape(tag)}>", raw_response))
        closes = len(re.findall(f"</{re.escape(tag)}>", raw_response))
        if opens > closes:
            warnings.append(f"Unclosed <{tag}> tag")
            violations.append("unclosed_tag")
        if re.findall(rf"<{re.escape(tag)}>\s*</{re.escape(tag)}>", raw_response):
            warnings.append(f"Empty <{tag}> block")
            violations.append("empty_block")

    open_tag = None
    for match in tag_re.finditer(raw_response):
        is_close = match.group(1) == "/"
        tag_name = match.group(2)
        if is_close:
            if open_tag == tag_name:
                open_tag = None
        elif open_tag is not None:
            warnings.append(f"<{tag_name}> nested inside unclosed <{open_tag}>")
            violations.append("nested_tags")
            break
        else:
            open_tag = tag_name

    first_open = re.search(rf"<(?:{tag_alternation})>", raw_response)
    if first_open and raw_response[: first_open.start()].strip():
        warnings.append("Text detected before first XML tag")
        violations.append("text_outside_tags")
    else:
        between_re = rf"</(?:{tag_alternation})>(.*?)<(?:{tag_alternation})>"
        for match in re.finditer(between_re, raw_response, re.DOTALL):
            if match.group(1).strip():
                warnings.append("Text detected between XML tags")
                violations.append("text_outside_tags")
                break

    last_close = None
    for match in re.finditer(rf"</(?:{tag_alternation})>", raw_response):
        last_close = match
    if last_close and raw_response[last_close.end() :].strip():
        warnings.append("Text detected after final XML tag")
        violations.append("text_outside_tags")

    return warnings, violations


class XMLParser:
    format_tags = ["command", "action"]

    def parse_response(self, response: str | None) -> ParseResult:
        if not response:
            return ParseResult(error="Empty response")
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        command_matches = re.findall(r"<command>(.*?)</command>", response, re.DOTALL)
        commands = [
            ParsedCommand(name="bash", arguments={"command": cmd.strip()}) for cmd in command_matches if cmd.strip()
        ]
        is_done = bool(re.search(r"<action>\s*done\s*</action>", response, re.IGNORECASE))
        if not commands and not is_done:
            return ParseResult(thinking=thinking, error="No <command> or <action>done</action> found in response.")
        return ParseResult(thinking=thinking, commands=commands, is_done=is_done)


class HFHermesParser:
    format_tags = ["tool_call"]
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def parse_response(self, response: str | None, extract_answer_tags_for_done: bool = False) -> ParseResult:
        if not response:
            return ParseResult(thinking="", error="Empty response")
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        is_done = bool(self._ANSWER_RE.search(response)) if extract_answer_tags_for_done else False
        matches = self._TOOL_CALL_RE.findall(response)
        if not matches:
            if is_done:
                return ParseResult(thinking=thinking, is_done=True)
            return ParseResult(thinking=thinking, error="No <tool_call> found in response.")
        commands = []
        for raw in matches:
            error = ""
            try:
                payload = json.loads(raw.strip())
            except json.JSONDecodeError:
                commands.append(ParsedCommand(error="Invalid JSON in tool_call", raw=raw.strip()))
                continue
            if isinstance(payload, dict):
                name = payload.get("name", "")
                args = payload.get("arguments", {})
            else:
                name, args, error = "unknown", {}, "Expected JSON object with 'name' and 'arguments' fields"
            if not isinstance(args, dict):
                args, error = {}, "Expected 'arguments' field to be a JSON object"
            if not extract_answer_tags_for_done and name == "done":
                is_done = True
            elif name:
                commands.append(ParsedCommand(name=name, arguments=args, error=error, raw=raw.strip()))
        if not commands and not is_done:
            return ParseResult(thinking=thinking, error="No valid tool calls or done signal found.")
        return ParseResult(thinking=thinking, commands=commands, is_done=is_done)


class Qwen35XMLParser:
    format_tags = ["tool_call"]
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
    _FUNCTION_RE = re.compile(r"<function=([\w.-]+)>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def parse_response(self, response: str | None, extract_answer_tags_for_done: bool = False) -> ParseResult:
        if not response:
            return ParseResult(thinking="", error="Empty response")
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        matches = self._TOOL_CALL_RE.findall(response)
        if not matches:
            return ParseResult(thinking=thinking, error="No <tool_call> found in response.")
        is_done = bool(self._ANSWER_RE.search(response)) if extract_answer_tags_for_done else False
        commands = []
        for raw in matches:
            func_match = self._FUNCTION_RE.search(raw)
            if not func_match:
                commands.append(ParsedCommand(error="Missing <function=...> tag", raw=raw.strip()))
                continue
            name = func_match.group(1)
            if not extract_answer_tags_for_done and name == "done":
                is_done = True
                continue
            params = {m.group(1): m.group(2).strip() for m in self._PARAM_RE.finditer(raw)}
            if params:
                commands.append(ParsedCommand(name=name, arguments=params, raw=raw.strip()))
            else:
                commands.append(ParsedCommand(name=name, arguments={}, error="No parameters found", raw=raw.strip()))
        if not commands and not is_done:
            return ParseResult(thinking=thinking, error="No valid tool calls or done signal found.")
        return ParseResult(thinking=thinking, commands=commands, is_done=is_done)


def get_parser(parser_name: str) -> HFHermesParser | Qwen35XMLParser | XMLParser:
    if parser_name == "hermes":
        return HFHermesParser()
    if parser_name == "xml":
        return XMLParser()
    if parser_name == "qwen35":
        return Qwen35XMLParser()
    raise ValueError(f"Unknown parser_name {parser_name!r}. Must be 'hermes', 'xml', or 'qwen35'.")
