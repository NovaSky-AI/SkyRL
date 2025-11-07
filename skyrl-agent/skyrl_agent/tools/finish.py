from .base import BaseTool, register_tool
from typing import Union


@register_tool("finish")
class FinishTool(BaseTool):
    name = "finish"
    description = (
        "Signals the completion of the current task or conversation.\n\n"
        "Use this tool when:\n"
        "- You have successfully completed the requested task\n"
        "- You cannot proceed further due to technical limitations or missing information\n\n"
        "Always end by calling the finish tool.\n"
    )
    parameters = {
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Provide exactly one \\boxed{...} as the final answer.\n"
                    "You MUST ALWAYS use \\boxed{YOUR_ANSWER} format in the finish tool's answer parameter.\n"
                    "This applies to ALL types of answers:\n"
                    "  - Numbers: \\boxed{42}\n"
                    "  - Words: \\boxed{word}\n"
                    "  - Yes/No: \\boxed{Yes} or \\boxed{No}\n"
                    "  - Expressions: \\boxed{x^2 + 1}\n"
                    "  - Multiple choice: \\boxed{A} or \\boxed{B}\n"
                    "- NEVER use plain text without \\boxed{} in the finish tool\n"
                    "- This format is MANDATORY for ALL answers"
                ),
            },
        },
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        # If the parameters are valid, we can proceed to finish the task.
        answer = params.get("answer", "")
        # Enforce boxed format if missing (universal requirement per instruction)
        if isinstance(answer, str) and "\\boxed{" not in answer:
            answer = f"\\boxed{{{answer}}}"
        return answer
