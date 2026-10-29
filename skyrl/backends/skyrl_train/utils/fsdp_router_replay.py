"""Rollout routing replay (R3) for HF models trained with FSDP.

Same semantics as Megatron's ``RouterReplay`` (REPLAY_FORWARD): the MoE router uses the
rollout's top-k expert ids and computes their weights from its own probabilities
(``scores.gather(-1, ids)``), so the gate still gets gradients.

Model contract: a replay-capable MoE module declares the class attributes
``replay_topk_ids`` and ``replay_native_topk_ids``. Before a forward its
``replay_topk_ids`` is set to ``[B * S, top_k]`` ids in the flattened order of the
router's input, ``-1`` rows keeping the router's own top-k; the module stores its own
top-k in ``replay_native_topk_ids``. The module's global layer index (vLLM records routes
by layer number, dense layers have no record) is the single integer in its module name.

Rows: a (token, MoE layer) row is replayed when the token is real and has a captured
route (``router_padding_mask`` False). Rows a loss-active target depends on (every real
token before the last loss-active target) must be replayed, and every replayed row must
hold ``top_k`` distinct ids in range; anything else raises.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor

REPLAY_ATTR = "replay_topk_ids"
NATIVE_ATTR = "replay_native_topk_ids"


def find_replay_routers(model: nn.Module) -> dict[int, nn.Module]:
    """Map global layer index -> replay-capable MoE module."""
    routers: dict[int, nn.Module] = {}
    for name, module in model.named_modules():
        if not (hasattr(type(module), REPLAY_ATTR) and hasattr(type(module), NATIVE_ATTR)):
            continue
        numbers = [int(part) for part in name.split(".") if part.isdigit()]
        if len(numbers) != 1:
            raise ValueError(f"cannot read a layer index from router module name {name!r}")
        if numbers[0] in routers:
            raise ValueError(f"two replay routers for layer {numbers[0]}")
        routers[numbers[0]] = module
    return routers


@dataclass
class ReplayPlan:
    ids: torch.Tensor  # [B, S, len(layers), top_k] int64, -1 = not replayed
    replayed: torch.Tensor  # [B, S] bool
    real_tokens: int
    replayed_tokens: int


def build_replay_plan(
    rollout_expert_indices: PackedTensor,
    attention_mask: torch.Tensor,
    router_padding_mask: torch.Tensor,
    layers: list[int],
    num_experts: int,
    loss_mask: Optional[torch.Tensor] = None,
) -> ReplayPlan:
    """Scatter packed rollout routes onto the padded batch and validate them.

    ``rollout_expert_indices`` holds one segment per batch row covering that row's real
    tokens in order (``[sum(real_i), num_layers, top_k]``). ``loss_mask`` is the
    response-aligned ``[B, R]`` mask whose slot ``k`` scores sequence position ``S - R + k``.
    """
    if len(rollout_expert_indices.row_shape) != 2:
        raise ValueError(f"expected [tokens, layers, topk] routes, got {rollout_expert_indices!r}")
    num_layers, topk = rollout_expert_indices.row_shape
    bad_layers = [layer for layer in layers if not 0 <= layer < num_layers]
    if bad_layers:
        raise ValueError(f"MoE layers {bad_layers} are outside the {num_layers} recorded layers")
    if router_padding_mask.shape != attention_mask.shape:
        raise ValueError(
            f"router_padding_mask {tuple(router_padding_mask.shape)} does not match "
            f"attention_mask {tuple(attention_mask.shape)}"
        )
    real = attention_mask.bool()
    seg = rollout_expert_indices.sequence_lengths.to(real.device, torch.long)
    if len(seg) != real.shape[0] or not torch.equal(seg, real.sum(1)):
        raise ValueError(f"route segments {seg.tolist()} do not match real token counts {real.sum(1).tolist()}")

    B, S = real.shape
    layer_index = torch.tensor(layers, dtype=torch.long, device=rollout_expert_indices.device)
    ids = torch.full((B, S, len(layers), topk), -1, dtype=torch.long, device=real.device)
    ids[real] = rollout_expert_indices.values.index_select(1, layer_index).to(real.device, torch.long)
    replayed = real & ~router_padding_mask.to(real.device, torch.bool)
    ids[~replayed] = -1

    if loss_mask is not None:
        R = loss_mask.shape[1]
        active = loss_mask.to(real.device) > 0
        target_pos = torch.arange(S - R, S, device=real.device).expand(B, R)
        last_target = torch.where(active, target_pos, -1).amax(1)
        required = real & (torch.arange(S, device=real.device)[None] < last_target[:, None])
        missing = required & ~replayed
        if missing.any():
            b, s = missing.nonzero()[0].tolist()
            raise ValueError(
                f"{int(missing.sum())} tokens that loss-active targets depend on have no rollout route "
                f"(first: batch row {b}, position {s})"
            )

    rows = ids[replayed]  # [n, layers, topk]
    if rows.numel():
        out_of_range = (rows < 0) | (rows >= num_experts)
        if out_of_range.any():
            raise ValueError(
                f"{int(out_of_range.any(-1).sum())} replayed rows hold expert ids outside [0, {num_experts})"
            )
        ordered = rows.sort(-1).values
        dup = (ordered[..., 1:] == ordered[..., :-1]).any(-1)  # [n, layers]
        if dup.any():
            n, l = dup.nonzero()[0].tolist()  # noqa: E741
            b, s = replayed.nonzero()[n].tolist()
            raise ValueError(
                f"{int(dup.sum())} replayed (token, layer) rows repeat an expert id "
                f"(first: batch row {b}, position {s}, layer {layers[l]}: {rows[n, l].tolist()}); "
                "the rollout did not record this layer"
            )
    return ReplayPlan(ids, replayed, int(real.sum()), int(replayed.sum()))


def install_replay(
    routers: dict[int, nn.Module],
    plan: Optional[ReplayPlan],
    token_index: Optional[torch.Tensor] = None,
) -> None:
    """Set (or with ``plan=None`` clear) every router's replay ids.

    ``token_index`` selects the forward's tokens from the flattened ``[B * S]`` batch when
    padding is removed (``unpad_input``'s indices); otherwise the router sees ``B * S`` rows.
    """
    for i, layer in enumerate(sorted(routers)):
        module = routers[layer]
        setattr(module, NATIVE_ATTR, None)
        if plan is None:
            setattr(module, REPLAY_ATTR, None)
            continue
        ids = plan.ids[:, :, i].reshape(-1, plan.ids.shape[-1])
        if token_index is not None:
            ids = ids.index_select(0, token_index)
        setattr(module, REPLAY_ATTR, ids.contiguous())


def replay_metrics(routers: dict[int, nn.Module], plan: ReplayPlan) -> dict[str, float]:
    """Coverage and the share of replayed rows whose own top-k set differs from the rollout's.

    Replay hides routing mismatch, so the second number is the mismatch it covers up.
    """
    replayed_rows = mismatched = 0
    for module in routers.values():
        target = getattr(module, REPLAY_ATTR)
        native = getattr(module, NATIVE_ATTR)
        if target is None or native is None:
            raise RuntimeError("router replay was installed but a router did not run")
        use = target[:, 0] >= 0
        differs = (target.sort(-1).values != native.sort(-1).values).any(-1) & use
        replayed_rows += int(use.sum())
        mismatched += int(differs.sum())
    return {
        "router_replay/row_coverage": plan.replayed_tokens / max(plan.real_tokens, 1),
        "router_replay/native_mismatch_frac": mismatched / max(replayed_rows, 1),
    }
