from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs


def test_pop_openai_kwargs():
    """
    Test pop_openai_kwargs with both primary and alias.
    Ensure OpenAI kwargs are popped, non-OpenAI kwargs are kept.
    """
    engine_kwargs = {
        "enable_auto_tools": 1,
        "tool_parser": "json",
        "other": "keep",
    }
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": True, "tool_parser": "json"}
    assert engine_kwargs == {"other": "keep"}

    engine_kwargs = {"enable_auto_tool_choice": 0, "tool_call_parser": "proto"}
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": False, "tool_parser": "proto"}
    assert engine_kwargs == {}
