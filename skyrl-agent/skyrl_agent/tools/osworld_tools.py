from .base import BaseTool, register_tool
from typing import Union, Optional, Tuple
import xml.etree.ElementTree as ET
import re
import tiktoken
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

@register_tool("osworld_action")
class OSWorldActionTool(BaseTool):
    name = "osworld_action"
    description = "Execute desktop automation actions using pyautogui code snippets. Provide Python code that uses pyautogui functions like click(), typewrite(), press(), hotkey(), scroll(), moveTo(), dragTo(), etc."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code snippet using pyautogui functions. Examples: 'pyautogui.click(500, 300)', 'pyautogui.typewrite(\"Hello World\")', 'pyautogui.press(\"enter\")', 'pyautogui.hotkey(\"ctrl\", \"c\")', 'time.sleep(2)'"
            }
        },
        "required": ["code"]
    }
    
    attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
    attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
    state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
    state_ns_windows = "https://accessibility.windows.example.org/ns/state"
    component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
    component_ns_windows = "https://accessibility.windows.example.org/ns/component"
    value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
    value_ns_windows = "https://accessibility.windows.example.org/ns/value"
    class_ns_windows = "https://accessibility.windows.example.org/ns/class"

    def call(self, params: Union[str, dict], runtime=None, **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}
        
        code = params.get("code", "").strip()
        
        if not code:
            return {"error": "Code parameter cannot be empty"}

        # Execute runtime.step with a 2-minute timeout
        try:
            obs, reward, done, info = runtime.step(code, 0.2)
        except Exception as e:
            return f"Action execution failed: {str(e)}, please try again, possibly with a different action."
        
        # Prefer accessibility tree when available; otherwise fall back to screenshot
        acc_tree = obs.get("accessibility_tree")
        if acc_tree:
            linearized_accessibility_tree = OSWorldActionTool.linearize_accessibility_tree(acc_tree, "ubuntu") # fixme: refer to the platform of the runtime/agent
            if linearized_accessibility_tree:
                linearized_accessibility_tree = OSWorldActionTool.trim_accessibility_tree(linearized_accessibility_tree, 10000) # fixme: add arguments for max tokens
            return "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(linearized_accessibility_tree)
        
        
        ## TODO(ys): untested visual tool response
        screenshot_bytes = obs.get("screenshot")
        if screenshot_bytes:
            encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{encoded}"
            return  [
                    {"type": "text", "text": "Here is the latest desktop screenshot after executing the action."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
        return {"error": "No observation available from runtime (both accessibility_tree and screenshot are missing)."}
    
    @staticmethod
    def parse_code_from_string(input_string):
        input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
        if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
            return [input_string.strip()]

        # This regular expression will match both ```code``` and ```python code```
        # and capture the `code` part. It uses a non-greedy match for the content inside.
        pattern = r"```(?:\w+\s+)?(.*?)```"
        # Find all non-overlapping matches in the string
        matches = re.findall(pattern, input_string, re.DOTALL)

        # The regex above captures the content inside the triple backticks.
        # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
        # so the code inside backticks can span multiple lines.

        # matches now contains all the captured code snippets

        codes = []

        for match in matches:
            match = match.strip()
            commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

            if match in commands:
                codes.append(match.strip())
            elif match.split('\n')[-1] in commands:
                if len(match.split('\n')) > 1:
                    codes.append("\n".join(match.split('\n')[:-1]))
                codes.append(match.split('\n')[-1])
            else:
                codes.append(match)

        return codes
    
    @staticmethod
    def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):
        
        if platform == "ubuntu":
            _attributes_ns = OSWorldActionTool.attributes_ns_ubuntu
            _state_ns = OSWorldActionTool.state_ns_ubuntu
            _component_ns = OSWorldActionTool.component_ns_ubuntu
            _value_ns = OSWorldActionTool.value_ns_ubuntu
        elif platform == "windows":
            _attributes_ns = OSWorldActionTool.attributes_ns_windows
            _state_ns = OSWorldActionTool.state_ns_windows
            _component_ns = OSWorldActionTool.component_ns_windows
            _value_ns = OSWorldActionTool.value_ns_windows
        else:
            raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

        filtered_nodes = OSWorldActionTool.filter_nodes(ET.fromstring(accessibility_tree), platform)
        linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

        # Linearize the accessibility tree nodes into a table format
        for node in filtered_nodes:
            if node.text:
                text = (
                    node.text if '"' not in node.text \
                        else '"{:}"'.format(node.text.replace('"', '""'))
                )

            elif node.get("{{{:}}}class".format(OSWorldActionTool.class_ns_windows), "").endswith("EditWrapper") \
                    and node.get("{{{:}}}value".format(_value_ns)):
                node_text = node.get("{{{:}}}value".format(_value_ns), "")
                text = (node_text if '"' not in node_text \
                            else '"{:}"'.format(node_text.replace('"', '""'))
                        )
            else:
                text = '""'

            linearized_accessibility_tree.append(
                "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                    node.tag, node.get("name", ""),
                    text,
                    node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(OSWorldActionTool.class_ns_windows), ""),
                    node.get("{{{:}}}description".format(_attributes_ns), ""),
                    node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                    node.get('{{{:}}}size'.format(_component_ns), "")
                )
            )

        return "\n".join(linearized_accessibility_tree)
    
    @staticmethod
    def filter_nodes(root: ET, platform="ubuntu", check_image=False):
        filtered_nodes = []

        for node in root.iter():
            if OSWorldActionTool.judge_node(node, platform, check_image):
                filtered_nodes.append(node)
                # print(ET.tostring(node, encoding="unicode"))

        return filtered_nodes
    
    @staticmethod
    def judge_node(node: ET, platform="ubuntu", check_image=False) -> bool:
        if platform == "ubuntu":
            _state_ns = OSWorldActionTool.state_ns_ubuntu
            _component_ns = OSWorldActionTool.component_ns_ubuntu
        elif platform == "windows":
            _state_ns = OSWorldActionTool.state_ns_windows
            _component_ns = OSWorldActionTool.component_ns_windows
        else:
            raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

        keeps: bool = node.tag.startswith("document") \
                    or node.tag.endswith("item") \
                    or node.tag.endswith("button") \
                    or node.tag.endswith("heading") \
                    or node.tag.endswith("label") \
                    or node.tag.endswith("scrollbar") \
                    or node.tag.endswith("searchbox") \
                    or node.tag.endswith("textbox") \
                    or node.tag.endswith("link") \
                    or node.tag.endswith("tabelement") \
                    or node.tag.endswith("textfield") \
                    or node.tag.endswith("textarea") \
                    or node.tag.endswith("menu") \
                    or node.tag in {"alert", "canvas", "check-box"
                        , "combo-box", "entry", "icon"
                        , "image", "paragraph", "scroll-bar"
                        , "section", "slider", "static"
                        , "table-cell", "terminal", "text"
                        , "netuiribbontab", "start", "trayclockwclass"
                        , "traydummysearchcontrol", "uiimage", "uiproperty"
                        , "uiribboncommandbar"
                                    }
        keeps = keeps and (
                platform == "ubuntu"
                and node.get("{{{:}}}showing".format(_state_ns), "false") == "true"
                and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
                or platform == "windows"
                and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
        ) \
                and (
                        node.get("{{{:}}}enabled".format(_state_ns), "false") == "true"
                        or node.get("{{{:}}}editable".format(_state_ns), "false") == "true"
                        or node.get("{{{:}}}expandable".format(_state_ns), "false") == "true"
                        or node.get("{{{:}}}checkable".format(_state_ns), "false") == "true"
                ) \
                and (
                        node.get("name", "") != "" or node.text is not None and len(node.text) > 0 \
                        or check_image and node.get("image", "false") == "true"
                )

        coordinates: Tuple[int, int] = eval(node.get("{{{:}}}screencoord".format(_component_ns), "(-1, -1)"))
        sizes: Tuple[int, int] = eval(node.get("{{{:}}}size".format(_component_ns), "(-1, -1)"))
        keeps = keeps and coordinates[0] >= 0 and coordinates[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
        return keeps
    
    @staticmethod
    def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(linearized_accessibility_tree)
        if len(tokens) > max_tokens:
            linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
            linearized_accessibility_tree += "[...]\n"
        return linearized_accessibility_tree