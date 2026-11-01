"""Opt-in: send each token once per destination EP rank in the MoE all-to-all.

``MoEAlltoAllTokenDispatcher`` permutes tokens by expert and sends one copy per (token, expert)
pair, so a top-8 token crosses the EP all-to-all 8 times although its experts live on ~5 of 8
ranks (GLM-5.3-Flash: 288 experts, EP8). ``SKYRL_MOE_DEDUP_A2A=1`` instead:

- sender: permutes tokens by destination *rank* (one copy per (token, rank)) and sends, with each
  copy, the token's routing mask and probabilities over that rank's local experts;
- receiver: permutes the received copies by local expert, runs the experts unchanged, and sums
  each copy's expert outputs (they are already prob-weighted) before the return all-to-all;
- sender: sums the per-rank partial outputs of each token.

Dispatch and combine move ~1/3 fewer rows, forward and backward. The expert inputs are
identical; the combine sums a token's expert outputs in two stages (per rank, rounded to the
activation dtype, then across ranks) instead of one. Applies to the plain dropless alltoall
path (ETP1, fused permute, no shared-expert overlap); anything else runs the stock methods.
"""

import os

import torch
from loguru import logger

_APPLIED = False
_METHODS = (
    "dispatch_preprocess",
    "token_dispatch",
    "dispatch_postprocess",
    "combine_preprocess",
    "token_combine",
    "combine_postprocess",
)


def apply_moe_dedup_alltoall_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_MOE_DEDUP_A2A", "0").lower() not in ("1", "true"):
        return
    from megatron.core.tensor_parallel import all_to_all
    from megatron.core.transformer.moe.moe_utils import permute, unpermute
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    stock = {name: getattr(MoEAlltoAllTokenDispatcher, name) for name in _METHODS}

    def eligible(self) -> bool:
        config = self.config
        return (
            self.tp_size == 1
            and self.ep_size > 1
            and not self.drop_and_pad
            and not config.moe_router_padding_for_quantization
            and config.moe_permute_fusion
            and not config.batch_invariant_mode
            and self.shared_experts is None
        )

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        self._skyrl_dedup = eligible(self)
        if not self._skyrl_dedup:
            return stock["dispatch_preprocess"](self, hidden_states, routing_map, probs)
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.hidden_shape_before_permute = hidden_states.shape
        num_tokens = hidden_states.size(0)

        local_map = routing_map.view(num_tokens, self.ep_size, self.num_local_experts)
        rank_map = local_map.any(dim=-1)
        input_splits = rank_map.sum(dim=0)
        output_splits = torch.empty_like(input_splits)
        torch.distributed.all_to_all_single(output_splits, input_splits, group=self.ep_group)
        input_splits, output_splits = torch.stack([input_splits, output_splits]).tolist()
        self.input_splits, self.output_splits = input_splits, output_splits
        self._skyrl_rank_map = rank_map

        # permute orders copies by rank, then token: the same order as rank_map.T.nonzero().
        tokens, _, self._skyrl_send_map, _, _ = permute(
            hidden_states, rank_map, num_out_tokens=sum(input_splits), fused=True
        )
        rank_idx, token_idx = rank_map.t().nonzero(as_tuple=True)
        self._skyrl_send_mask = local_map[token_idx, rank_idx]
        copy_probs = probs.view(num_tokens, self.ep_size, self.num_local_experts)[token_idx, rank_idx]
        return tokens, copy_probs

    def token_dispatch(self, tokens, probs):
        if not self._skyrl_dedup:
            return stock["token_dispatch"](self, tokens, probs)
        splits = (self.output_splits, self.input_splits)
        recv_tokens = all_to_all(self.ep_group, tokens, *splits)
        recv_probs = all_to_all(self.ep_group, probs, *splits)
        self._skyrl_recv_mask = all_to_all(self.ep_group, self._skyrl_send_mask.to(torch.uint8), *splits).bool()
        self._skyrl_send_mask = None
        return recv_tokens, recv_probs

    def dispatch_postprocess(self, tokens, probs):
        if not self._skyrl_dedup:
            return stock["dispatch_postprocess"](self, tokens, probs)
        mask = self._skyrl_recv_mask
        tokens_per_expert = mask.sum(dim=0).cpu()
        self._skyrl_recv_shape = tokens.shape
        expert_tokens, expert_probs, self._skyrl_recv_map, _, _ = permute(
            tokens, mask, probs=probs, num_out_tokens=int(tokens_per_expert.sum()), fused=True
        )
        return expert_tokens, tokens_per_expert, expert_probs

    def combine_preprocess(self, hidden_states):
        if not self._skyrl_dedup:
            return stock["combine_preprocess"](self, hidden_states)
        output = unpermute(
            hidden_states,
            self._skyrl_recv_map,
            restore_shape=self._skyrl_recv_shape,
            routing_map=self._skyrl_recv_mask,
            fused=True,
        )
        self._skyrl_recv_map = self._skyrl_recv_mask = None
        return output

    def token_combine(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
        if not self._skyrl_dedup:
            return stock["token_combine"](self, hidden_states, async_finish, allocate_on_comm_stream)
        return all_to_all(self.ep_group, hidden_states, self.input_splits, self.output_splits)

    def combine_postprocess(self, hidden_states):
        if not self._skyrl_dedup:
            return stock["combine_postprocess"](self, hidden_states)
        output = unpermute(
            hidden_states,
            self._skyrl_send_map,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self._skyrl_rank_map,
            fused=True,
        )
        self._skyrl_send_map = self._skyrl_rank_map = None
        return output.view(self.hidden_shape)

    for name in _METHODS:
        setattr(MoEAlltoAllTokenDispatcher, name, locals()[name])
    _APPLIED = True
    logger.info("MoE: all-to-all sends each token once per destination EP rank")
