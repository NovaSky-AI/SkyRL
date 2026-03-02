from typing import Dict, Any

from skyrl_train.generators.utils import get_custom_chat_template


def pop_openai_kwargs(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize & remove OpenAI-serving-only kwargs from engine_kwargs.
    """
    openai_kwargs: Dict[str, Any] = {}

    enable_auto_tools = engine_kwargs.pop("enable_auto_tools", engine_kwargs.pop("enable_auto_tool_choice", None))
    if enable_auto_tools is not None:
        openai_kwargs["enable_auto_tools"] = bool(enable_auto_tools)

    tool_parser = engine_kwargs.pop("tool_parser", engine_kwargs.pop("tool_call_parser", None))
    if tool_parser is not None:
        openai_kwargs["tool_parser"] = tool_parser

    reasoning_parser = engine_kwargs.pop("reasoning_parser", None)
    if reasoning_parser is not None:
        openai_kwargs["reasoning_parser"] = reasoning_parser

    # Since we use OpenAIServingChat() ourselves, which requires the content of
    # the chat template, not the path (unlike --chat-template in vllm serve CLI args)
    chat_template = engine_kwargs.pop("chat_template", None)
    if chat_template is not None:
        openai_kwargs["chat_template"] = get_custom_chat_template(
            chat_template_config={"source": "file", "name_or_path": chat_template}
        )
        assert openai_kwargs["chat_template"] is not None, "Failed to get custom chat template"

    return openai_kwargs
