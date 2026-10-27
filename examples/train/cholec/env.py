"""Single-turn CholecT50 surgical triplet-recognition environment for SkyRL.

Port of the verl / Tinker reward to SkyRL's gymnasium env contract:

    reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)

The model is shown one endoscopic frame (the image lives in the prompt's
``image_url`` content parts) and emits a single completion listing
``instrument-verb-target`` triplets. ``step()`` parses that completion and returns
the blended F1 reward. Single-turn (``max_turns=1``): the episode ends after the
first — and only — response.
"""
import json
import re
from typing import Any, Dict, Optional, Set

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

INSTRUMENTS = {"grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"}
VERBS = {"grasp", "retract", "dissect", "coagulate", "clip", "cut",
         "aspirate", "irrigate", "pack", "null_verb"}
TARGETS = {"gallbladder", "cystic_plate", "cystic_duct", "cystic_artery",
           "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity",
           "liver", "adhesion", "omentum", "peritoneum", "gut", "specimen_bag",
           "null_target"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")  # <|im_end|>, <|im_start|>, <|image_pad|>, ...


def parse_triplets(text: Optional[str]) -> Optional[Set[str]]:
    """Parse a completion into a set of canonical 'instrument-verb-target' strings.

    Returns None for a malformed completion (-> reward 0), an empty set for an
    explicit 'none', else the parsed triplets. Any leaked <think>...</think> block
    is stripped first (thinking-mode footgun).
    """
    if text is None:
        return None
    t = _SPECIAL_RE.sub("", _THINK_RE.sub("", text)).strip()
    if t.lower() in ("", "none", "no action", "nothing", "no triplets"):
        return set()
    parts = [p.strip().strip(".;()[]").lower()
             for p in t.replace("\n", ",").split(",") if p.strip()]
    trips: Set[str] = set()
    for p in parts:
        comps = p.split("-")
        if len(comps) != 3:
            return None
        i, v, tg = (c.strip() for c in comps)
        if i not in INSTRUMENTS or v not in VERBS or tg not in TARGETS:
            return None
        trips.add(f"{i}-{v}-{tg}")
    return trips


def _f1(pred: Set[str], gt: Set[str]) -> float:
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def reward_single(pred: Optional[Set[str]], gt: dict) -> float:
    if pred is None:
        return 0.0
    triplet_f = _f1(pred, set(gt.get("triplets", [])))
    inst_f = _f1({t.split("-")[0] for t in pred}, set(gt.get("instruments", [])))
    verb_f = _f1({t.split("-")[1] for t in pred}, set(gt.get("verbs", [])))
    targ_f = _f1({t.split("-")[2] for t in pred}, set(gt.get("targets", [])))
    return 0.4 * triplet_f + 0.6 * (inst_f + verb_f + targ_f) / 3.0


class CholecEnv(BaseTextEnv):
    """Single-turn CholecT50 triplet recognition: image + instruction in, triplets out."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 1

        reward_spec = extras.get("reward_spec", {}) or {}
        gt = reward_spec.get("ground_truth", {})
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                gt = {}
        self.ground_truth = gt or {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        reward = reward_single(parse_triplets(action), self.ground_truth)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"ground_truth": self.ground_truth},
        )


if __name__ == "__main__":
    gt = {
        "triplets": ["grasper-retract-gallbladder", "hook-dissect-cystic_duct"],
        "instruments": ["grasper", "hook"],
        "verbs": ["retract", "dissect"],
        "targets": ["gallbladder", "cystic_duct"],
    }
    print("exact:", reward_single(parse_triplets("grasper-retract-gallbladder, hook-dissect-cystic_duct"), gt))
    print("one-of-two:", reward_single(parse_triplets("grasper-retract-gallbladder"), gt))
    print("thinking-leak:", reward_single(parse_triplets("<think>hmm</think>\ngrasper-retract-gallbladder, hook-dissect-cystic_duct"), gt))
    print("none:", reward_single(parse_triplets("none"), gt))
    print("malformed:", reward_single(parse_triplets("random garbage"), gt))
