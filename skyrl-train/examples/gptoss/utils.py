from typing import List, Tuple, Any
from loguru import logger


def find_sublist_in_list(list1: List[Any], list2: List[Any]):
    assert len(list1) and len(list2)

    offset = 0
    while offset < len(list1):
        try:
            start_idx = list1.index(list2[0], offset)
        except ValueError:
            break
        if list1[start_idx : start_idx + len(list2)] == list2:
            return start_idx
        offset = start_idx + 1
    raise ValueError(f"{list2} not found in the given input")


def get_think_and_answer_ids(
    response_ids: List[int],
    think_start_token_ids: List[int],
    think_end_token_ids: List[int],
) -> Tuple[List[int], List[int]]:
    try:
        think_end_list_idx = find_sublist_in_list(response_ids, think_end_token_ids)
    except ValueError:
        # truncated response case
        try:
            _ = find_sublist_in_list(response_ids, think_start_token_ids)
        except ValueError:
            logger.info("MEM1: Think end and think start idx not found for response. Assuming no thinking in response")
            return [], response_ids
        logger.info("MEM1: Think end not found for response. Assuming no answer in response")
        return response_ids, []
    return (
        response_ids[: think_end_list_idx + len(think_end_token_ids)],
        response_ids[think_end_list_idx + len(think_end_token_ids) :],
    )


def get_think_and_answer_str(response_str: List[int], think_start="<think>", think_end="</think>") -> Tuple[str, str]:
    think_end_list_idx = response_str.find(think_end)
    if think_end_list_idx == -1:
        # truncated response case
        think_start_list_idx = response_str.find(think_start)
        if think_start_list_idx == -1:
            logger.info("MEM1: Think end and think start idx not found for response. Assuming no thinking in response")
            return "", response_str
        logger.info("MEM1: Think end not found for response. Assuming no thinking in response")
        return response_str, ""
    return response_str[: think_end_list_idx + len(think_end)], response_str[think_end_list_idx + len(think_end) :]
