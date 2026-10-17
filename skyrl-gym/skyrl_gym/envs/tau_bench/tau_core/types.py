# Adapted from Sierra's tau-bench (sierra-research/tau-bench).
# Converted from pydantic BaseModels to dataclasses to avoid adding a pydantic
# dependency to skyrl-gym. ``Task`` accepts (and ignores) extra keyword fields such
# as ``annotator`` that appear in the upstream task definitions.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


@dataclass
class Action:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Task:
    """A single tau-bench task. Implemented as a plain class so it tolerates extra
    keyword fields (e.g. ``annotator``) present in the upstream task definitions."""

    def __init__(
        self,
        user_id: str,
        actions: List[Action],
        instruction: str,
        outputs: List[str],
        **_ignored: Any,
    ) -> None:
        self.user_id = user_id
        self.actions = actions
        self.instruction = instruction
        self.outputs = outputs


@dataclass
class RewardOutputInfo:
    r_outputs: float
    outputs: Dict[str, bool]


@dataclass
class RewardActionInfo:
    r_actions: float
    gt_data_hash: str


@dataclass
class RewardResult:
    reward: float
    info: Union[RewardOutputInfo, RewardActionInfo]
    actions: List[Action]


@dataclass
class EnvInfo:
    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardResult] = None


@dataclass
class EnvResponse:
    observation: str
    reward: float
    done: bool
    info: EnvInfo


@dataclass
class EnvResetResponse:
    observation: str
    info: EnvInfo
