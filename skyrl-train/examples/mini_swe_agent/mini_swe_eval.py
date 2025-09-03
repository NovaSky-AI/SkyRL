from typing import TypedDict, Optional
import traceback
import uuid

from typing import Dict, Any
from loguru import logger
from minisweagent.run.extra.swebench import (
    get_sb_environment,
)


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


def evaluate_trajectory(instance: Dict[str, Any], model_patch, sweagent_config) -> MiniSWEEvaluationResult:
    instance_id = instance["instance_id"]

    ret = MiniSWEEvaluationResult(instance_id=instance_id, resolved=False, eval_error=None)

    env = None
    try:
        env = get_sb_environment(sweagent_config, instance)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}")
        return ret

    # apply git patch
    # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
    # In modern systems, this is typically ~ 1 MB, which is pretty generous.
    # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
    delimiter = f"PATCH_{uuid.uuid4().hex}"  # unlikely to collide with symbols in the patch
    command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"

    obs = env.execute(command)

    if obs["returncode"] != 0:
        ret["eval_error"] = obs["output"]
    else:
        # run eval script in-line
        eval_script = instance["eval_script"]
        eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
        obs = env.execute(eval_cmd)
        # use the return value
        ret["resolved"] = obs["returncode"] == 0
        # truncate to last 1000 for brevity
        ret["eval_error"] = obs["output"][-1000:] if not ret["resolved"] else None
    return ret
