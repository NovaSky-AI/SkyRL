
from typing import List, Tuple, Dict
from loguru import logger

def get_think_and_answer_ids(response_ids: List[int],  think_start_token_id: int, think_end_token_id: int,) -> Tuple[List[int], List[int]]:
    try: 
        think_end_list_idx = response_ids.index(think_end_token_id)
    except ValueError:
        # truncated response case
        try: 
            think_start_list_idx = response_ids.index(think_start_token_id)
        except ValueError:
            logger.info("MEM1: Think end and think start idx not found for response. Assuming no thinking in response")
            return [], response_ids
        logger.info("MEM1: Think end not found for response. Assuming no thinking in response")
        return response_ids, []
    return response_ids[:think_end_list_idx+1], response_ids[think_end_list_idx+1:]
    

def get_think_and_answer_str(response_str: List[int], think_start = "<think>", think_end = "</think>") -> Tuple[str, str]:
    think_end_list_idx = response_str.find(think_end)
    if think_end_list_idx == -1:
        # truncated response case
        think_start_list_idx = response_str.find(think_start)
        if think_start_list_idx == -1:
            logger.info("MEM1: Think end and think start idx not found for response. Assuming no thinking in response")
            return "", response_str
        logger.info("MEM1: Think end not found for response. Assuming no thinking in response")
        return response_str, ""
    return response_str[:think_end_list_idx+len(think_end)], response_str[think_end_list_idx+len(think_end):]