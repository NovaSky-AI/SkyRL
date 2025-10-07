# Reference: Mem1
from typing import List, Dict, Union, Literal, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math


def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1,
    padding_side: Literal["left", "right"] = "right",
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return (
            torch.cat(
                [
                    tensor,
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )
            if padding_side == "right"
            else torch.cat(
                [
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor,
                ],
                dim=dim,
            )
        )


def flatten_trajectory(traj, pad_token_id, max_prompt_length, max_response_length):
    """
    Returns:
      seq:       LongTensor [L]
      kind:      LongTensor [L]  (0=q,1=t,2=r,3=i)
      step:      LongTensor [L]
      prompt_len: int
    """
    segments = []
    kinds = []
    steps = []
    loss_masks = []
    prompt_len = 0

    def append_segment(tokens, kind_code, step_idx):
        t = torch.tensor(tokens, dtype=torch.long) if not isinstance(tokens, torch.Tensor) else tokens
        loss_mask_val = 1 if kind_code in [1, 2] else 0
        sel = t
        segments.append(sel)
        kinds.append(torch.full_like(sel, kind_code))
        steps.append(torch.full_like(sel, step_idx))
        if kind_code > 0:
            loss_masks.append(torch.full_like(sel, loss_mask_val))

    # 1) question segment
    append_segment(traj["q"], kind_code=0, step_idx=0)

    # 2) interaction steps t_j, r_j, (optional i_j)
    j = 0
    while j < traj["num_rounds"]:
        if f"t{j}" in traj:
            append_segment(traj[f"t{j}"], kind_code=1, step_idx=j)
        if f"r{j}" in traj:
            append_segment(traj[f"r{j}"], kind_code=2, step_idx=j)
        if f"i{j}" in traj:
            append_segment(traj[f"i{j}"], kind_code=3, step_idx=j)
        j += 1

    if max_prompt_length is not None:
        prompt_seq = segments[0][:max_prompt_length]
        prompt_kind = kinds[0][:max_prompt_length]
        prompt_step = steps[0][:max_prompt_length]
    else:
        prompt_seq = segments[0]
        prompt_kind = kinds[0]
        prompt_step = steps[0]

    prompt_len = len(prompt_seq)

    resp_seq = torch.cat(segments[1:], dim=0)
    resp_kind = torch.cat(kinds[1:], dim=0)
    resp_step = torch.cat(steps[1:], dim=0)
    resp_loss_mask = torch.cat(loss_masks, dim=0)

    if max_response_length is not None:
        resp_seq = resp_seq[:max_response_length]
        resp_kind = resp_kind[:max_response_length]
        resp_step = resp_step[:max_response_length]
        resp_loss_mask = resp_loss_mask[:max_response_length]

    seq = torch.cat([prompt_seq, resp_seq], dim=0)
    kind = torch.cat([prompt_kind, resp_kind], dim=0)
    step = torch.cat([prompt_step, resp_step], dim=0)

    return seq, kind, step, prompt_len, resp_loss_mask


# NOTE (sumanthrh): attention mask for observation is all zeros. Seems okay because we do teacher forcing for causal LM models and we don't train on logprobs for observation tokens
def make_attention_mask(kind, step):
    """
    kind: LongTensor [L], values in {0=q,1=t,2=r,3=i}
    step: LongTensor [L]

    Returns:
      mask:       BoolTensor [L,L]
      info_mask:  BoolTensor [L]   (False for kind==3)
    """
    L = kind.size(0)
    ki = kind.unsqueeze(1)  # [L,1]
    kj = kind.unsqueeze(0)  # [1,L]
    si = step.unsqueeze(1)  # [L,1]
    sj = step.unsqueeze(0)  # [1,L]

    mask = torch.zeros((L, L), dtype=torch.bool)

    # each block attends to itself
    mask = (ki == kj) & (si == sj)

    # t tokens at step j may attend to q and (r,i) tokens at step j-1
    mask |= (ki == 1) & (kj == 0)  # attend to q
    mask |= (ki == 1) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step

    # r tokens at step j may attend to q, (r,i) tokens at step j-1, and t tokens at the same step
    mask |= (ki == 2) & (kj == 0)  # attend to q
    mask |= (ki == 2) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step
    mask |= (ki == 2) & (sj == si) & (kj == 1)  # attend to t at same step

    # i tokens (external info) attend to themselves
    mask |= (ki == 3) & (kj == 3) & (si == sj)

    # info_mask is False only for external information (kind == 3)
    info_mask = kind.ne(3)

    # make sure it is causal mask
    mask = torch.tril(mask)

    return mask, info_mask


def make_attention_mask_custom(kind, step):
    """
    kind: LongTensor [L], values in {0=q,1=t,2=r,3=i}
    step: LongTensor [L]

    Returns:
      mask:       BoolTensor [L,L]
      info_mask:  BoolTensor [L]   (False for kind==3)
    """
    L = kind.size(0)
    ki = kind.unsqueeze(1)  # [L,1]
    kj = kind.unsqueeze(0)  # [1,L]
    si = step.unsqueeze(1)  # [L,1]
    sj = step.unsqueeze(0)  # [1,L]

    mask = torch.ones((L, L), dtype=torch.bool)

    # # each block attends to itself
    # mask = (ki == kj) & (si == sj)

    # # t tokens at step j may attend to q and (r,i) tokens at step j-1
    # mask |= (ki == 1) & (kj == 0)  # attend to q
    # mask |= (ki == 1) & ((kj == 2) | (kj == 3))  # attend to r,i all previous step

    # # r tokens at step j may attend to q, (r,i) tokens at step j-1, and t tokens at the same step
    # mask |= (ki == 2) & (kj == 0)  # attend to q
    # mask |= (ki == 2) & ((kj == 2) | (kj == 3))  # attend to r,i at all prevoius step
    # mask |= (ki == 2) & (sj == si) & (kj == 1)  # attend to t at same step

    # # i tokens (external info) attend to themselves
    # mask |= (ki == 3) & (kj == 3) & (si == sj)

    # # info_mask is False only for external information (kind == 3)
    info_mask = kind.ne(3)

    # make sure it is causal mask
    mask = torch.tril(mask)

    return mask, info_mask


def get_mask_mod(kind: "B, S", step: "B, S") -> callable:

    def my_mask_mod(b, h, q_idx, k_idx) -> bool:
        nonlocal kind, step

        ki = kind[b][q_idx]
        kj = kind[b][k_idx]

        si = step[b][q_idx]
        sj = step[b][k_idx]

        mask = (ki == kj) & (si == sj)
        # t tokens at step j may attend to q and (r,i) tokens at step j-1
        mask |= (ki == 1) & (kj == 0)  # attend to q
        mask |= (ki == 1) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step

        # r tokens at step j may attend to q, (r,i) tokens at step j-1, and t tokens at the same step
        mask |= (ki == 2) & (kj == 0)  # attend to q
        mask |= (ki == 2) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step
        mask |= (ki == 2) & (sj == si) & (kj == 1)  # attend to t at same step

        # i tokens (external info) attend to themselves
        mask |= (ki == 3) & (kj == 3) & (si == sj)

        mask &= q_idx >= k_idx
        return mask

    return my_mask_mod


def compose_final_output(
    trajectories: List[Dict[str, List[int]]],
    is_masked_out: Optional[List[bool]] = None,
    pad_token_id=0,
    max_prompt_length=None,
    max_response_length=None,
    max_length_padding: bool = False,
):
    # --- Flatten all trajectories ---
    results = [flatten_trajectory(traj, pad_token_id, max_prompt_length, max_response_length) for traj in trajectories]
    seqs, kinds, steps, prompt_lens, loss_masks = zip(*results)
    B = len(seqs)

    # --- Build batched prompts (left-padded) ---
    prompt_segments = [seq[:p] for seq, p in zip(seqs, prompt_lens)]
    # prompt_attention = [[1]*p for p in prompt_lens]
    rev_prompts = [seg.flip(0) for seg in prompt_segments]
    rev_padded = pad_sequence(rev_prompts, batch_first=True, padding_value=pad_token_id)
    # prompt_attention = pad_sequence(prompt_attention, batch_first=True, padding_value=0)
    prompts = rev_padded.flip(1)  # [B, P_max]
    if max_length_padding:
        prompts = pad_to_length(prompts, length=max_prompt_length, pad_value=pad_token_id, padding_side="left")
    # prompt_attention = prompt_attention.flip(1) # [B, P_max]

    # --- Build batched responses (right-padded) ---
    response_segments = [seq[p:] for seq, p in zip(seqs, prompt_lens)]
    # response_attn_segments = [[1]*p for p in prompt_lens]
    responses = pad_sequence(response_segments, batch_first=True, padding_value=pad_token_id)  # [B, R_max]
    loss_masks = pad_sequence(loss_masks, batch_first=True, padding_value=0)
    if max_length_padding:
        responses = pad_to_length(responses, length=max_response_length, pad_value=pad_token_id, padding_side="right")
        loss_masks = pad_to_length(loss_masks, length=max_response_length, pad_value=0, padding_side="right")
    # response_attn = pad_sequence(response_attn_segments, batch_first=True, padding_value=0) # [B, R_max]

    # # --- Check if prompts and responses exceed 8196 ---
    # # --- If so, truncate ---
    # if prompts.size(1) + responses.size(1) > 8196:
    #     max_length = 8196 - prompts.size(1)
    #     responses = responses[:, :max_length]

    # --- Concatenate ---
    input_ids = torch.cat([prompts, responses], dim=1)  # [B, S]
    # attention_mask = torch.cat([prompt_attention, response_attn], dim=1)  # [B, S]
    attention_mask = input_ids.ne(pad_token_id)  # [B, S]
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask  # [B, S]

    # --- Build 4D masks batch-wise ---
    masks = []
    info_masks = []
    P_max = prompts.size(1)
    for k, s, p_len in zip(kinds, steps, prompt_lens):
        small_mask, info_small = make_attention_mask(k, s)  # [L,L], [L]
        L = small_mask.size(0)
        S = input_ids.size(1)
        offset = P_max - p_len

        # make sure the small mask is not larger than the max allowed size
        L = min(L, S - offset)
        small_mask = small_mask[:L, :L]
        info_small = info_small[:L]

        big_mask = torch.zeros((S, S), dtype=torch.bool)
        big_mask[offset : offset + L, offset : offset + L] = small_mask
        im = torch.zeros((S,), dtype=torch.bool)
        im[offset : offset + L] = info_small

        masks.append(big_mask)
        info_masks.append(im)

    attention_mask_4d = torch.stack(masks, dim=0).unsqueeze(1)  # [B,1,S,S]
    attention_mask_4d = torch.where(attention_mask_4d.bool(), 0, -1e9)  # [B,1,S,S]
    info_mask = torch.stack(info_masks, dim=0)  # [B, S]

    if is_masked_out is not None:
        # mask out
        masked_out_inds = [i for i, masked_out in enumerate(is_masked_out) if masked_out]
        loss_masks[masked_out_inds, :] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "attention_mask_4d": attention_mask_4d,
        "response_mask": loss_masks,
        "prompts": prompts,
        "responses": responses,
    }


PAD_TO_MULTIPLE_OF = 128


def compose_final_output_flex_attn(
    trajectories: List[Dict[str, List[int]]],
    is_masked_out: Optional[List[bool]] = None,
    pad_token_id=0,
    max_prompt_length: int = None,
    max_response_length: int = None,
    max_length_padding: bool = False,
):

    # --- Flatten all trajectories ---
    results = [flatten_trajectory(traj, pad_token_id, max_prompt_length, max_response_length) for traj in trajectories]
    seqs, kinds, steps, prompt_lens, loss_masks = zip(*results)
    B = len(seqs)

    # --- Build batched prompts (left-padded) ---
    prompt_segments = [seq[:p] for seq, p in zip(seqs, prompt_lens)]
    # prompt_attention = [[1]*p for p in prompt_lens]
    rev_prompts = [seg.flip(0) for seg in prompt_segments]
    rev_padded = pad_sequence(rev_prompts, batch_first=True, padding_value=pad_token_id)
    # prompt_attention = pad_sequence(prompt_attention, batch_first=True, padding_value=0)
    prompts = rev_padded.flip(1)  # [B, P_max]
    if max_length_padding:
        assert max_prompt_length
        length = max_prompt_length
    else:
        length = math.ceil(prompts.size(1) / PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF

    prompts = pad_to_length(prompts, length=length, pad_value=pad_token_id, padding_side="left")
    left_paddings = [len(p) - len(seg) for p, seg in zip(prompts, prompt_segments)]
    # print("left paddings: ", left_paddings)
    # prompt_attention = prompt_attention.flip(1) # [B, P_max]

    # --- Build batched responses (right-padded) ---
    response_segments = [seq[p:] for seq, p in zip(seqs, prompt_lens)]
    # response_attn_segments = [[1]*p for p in prompt_lens]
    responses = pad_sequence(response_segments, batch_first=True, padding_value=pad_token_id)  # [B, R_max]
    loss_masks = pad_sequence(loss_masks, batch_first=True, padding_value=0)
    if max_length_padding:
        assert max_response_length
        length = max_response_length
    else:
        length = math.ceil(responses.size(1) / PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF
    responses = pad_to_length(responses, length=length, pad_value=pad_token_id, padding_side="right")
    loss_masks = pad_to_length(loss_masks, length=length, pad_value=0, padding_side="right")
    right_paddings = [len(r) - len(seg) for r, seg in zip(responses, response_segments)]
    # print("right paddings: ", right_paddings)
    # response_attn = pad_sequence(response_attn_segments, batch_first=True, padding_value=0) # [B, R_max]

    # # --- Check if prompts and responses exceed 8196 ---
    # # --- If so, truncate ---
    # if prompts.size(1) + responses.size(1) > 8196:
    #     max_length = 8196 - prompts.size(1)
    #     responses = responses[:, :max_length]

    # --- Concatenate ---
    input_ids = torch.cat([prompts, responses], dim=1)  # [B, S]
    # attention_mask = torch.cat([prompt_attention, response_attn], dim=1)  # [B, S]
    attention_mask = input_ids.ne(pad_token_id)  # [B, S]
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask  # [B, S]

    kinds_t = [F.pad(kind, (lp, rp), value=-1) for kind, lp, rp in zip(kinds, left_paddings, right_paddings)]
    # print("Kinds t before tensor: ", kinds_t)
    kinds_t = torch.stack(kinds_t)  # [B, S]

    steps_t = [F.pad(step, (lp, rp), value=-1) for step, lp, rp in zip(steps, left_paddings, right_paddings)]
    steps_t = torch.stack(steps_t)  # [B, S]

    # print("Steps t before tensor: ", steps_t)
    assert kinds_t.shape == input_ids.shape
    assert steps_t.shape == input_ids.shape

    # # --- Build 4D masks batch-wise ---
    # masks = []
    # info_masks = []
    # P_max = prompts.size(1)
    # for k, s, p_len, lp, rp in zip(kinds, steps, prompt_lens, left_paddings, right_paddings):
    #     small_mask, info_small = make_attention_mask(k, s)  # [L,L], [L]
    #     L = small_mask.size(0)
    #     S = input_ids.size(1)
    #     offset = P_max - p_len

    #     # make sure the small mask is not larger than the max allowed size
    #     L = min(L, S - offset)
    #     small_mask = small_mask[:L, :L]
    #     info_small = info_small[:L]

    #     big_mask = torch.zeros((S, S), dtype=torch.bool)
    #     big_mask[offset:offset+L, offset:offset+L] = small_mask
    #     im = torch.zeros((S,), dtype=torch.bool)
    #     im[offset:offset+L] = info_small

    #     masks.append(big_mask)
    #     info_masks.append(im)

    # attention_mask_4d = torch.stack(masks,   dim=0).unsqueeze(1)  # [B,1,S,S]
    # info_mask         = torch.stack(info_masks, dim=0)           # [B, S]

    # input_ids = pad_to_length(input_ids, length=pad_value, pad_value=pad_token_id, padding_side="right")

    print(
        f"final shapes after compose: input_ids; {input_ids.shape}, prompts: {prompts.shape}, responses : {responses.shape}, kinds: {kinds_t.shape}"
    )

    if is_masked_out is not None:
        # mask out
        masked_out_inds = [i for i, masked_out in enumerate(is_masked_out) if masked_out]
        loss_masks[masked_out_inds, :] = 0

    # TODO: add loss mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": loss_masks,
        "prompts": prompts,
        "responses": responses,
        "kinds": kinds_t,
        "steps": steps_t,
    }


# Example usage:
if __name__ == "__main__":
    # generate [512, 4096] trajectories
    trajectories = [
        {
            "q": [1, 2, 3],
            "t0": [201, 202],
            "r0": [301],
            "i0": [401, 402],
            "t1": [203],
            "r1": [302, 303],
            "i1": [302, 303],
            "t2": [204],
            "r2": [304, 305],
            "i2": [404, 405],
            "num_rounds": 3,
        },
        {
            "q": [1, 2, 3, 4],
            "t0": [],
            "r0": [301],
            "i0": [401, 402],
            "t1": [],
            "r1": [302, 303],
            "i1": [302, 303],
            "t2": [],
            "r2": [304, 305],
            "i2": [404, 405],
            "num_rounds": 3,
        },
        # { "q": [1,2,3], "t0":[213],      "r0":[313,314],"i0":[413],      "t1":[214],     "r1":[314],     "num_rounds": 2 },
    ]
    # trajectories = trajectories * 256
    batch = compose_final_output(trajectories, pad_token_id=0)
    batch = compose_final_output_flex_attn(trajectories, pad_token_id=0)

    print("input_ids shape:        ", batch["input_ids"].shape)  # [2, S]
    # print("attention_mask_4d shape:", batch["attention_mask_4d"].shape) # [2, 1, S, S]

    print("prompts shape:          ", batch["prompts"].shape)  # [2, P_max]
    print("responses shape:        ", batch["responses"].shape)  # [2, R_max]
