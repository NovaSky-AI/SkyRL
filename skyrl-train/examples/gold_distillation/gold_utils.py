"""
GOLD (General On-policy Logit Distillation) Utilities for Cross-Tokenizer Distillation.

Adapted from TRL's GOLDTrainer/ULDLoss implementation:
https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py

This module provides utilities for distilling knowledge between models with different
tokenizers by:
1. Re-tokenizing student-generated text with the teacher's tokenizer
2. Aligning token spans between student and teacher sequences
3. Computing a hybrid loss combining GKD (for matching tokens) and ULD (for non-matching tokens)
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def pad(tensors: list[torch.Tensor], padding_side: str = "right", padding_value: int = 0) -> torch.Tensor:
    """Pad a list of tensors to the same length."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            if padding_side == "right":
                t = F.pad(t, (0, pad_len), value=padding_value)
            else:
                t = F.pad(t, (pad_len, 0), value=padding_value)
        padded.append(t)
    return torch.stack(padded)


def build_teacher_inputs_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    completion_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Tokenize teacher prompts/completions and produce tensors ready for GOLD loss.

    Adapted from TRL's build_teacher_inputs_from_texts.

    Args:
        tokenizer: Teacher tokenizer
        prompt_texts: List of prompt strings
        completion_texts: List of completion strings

    Returns:
        teacher_input_ids: Padded input IDs [batch, seq_len]
        teacher_labels: Labels with prompt masked as -100 [batch, seq_len]
        teacher_attention_mask: Attention mask [batch, seq_len]
        teacher_prompt_length: Max prompt length in batch
    """
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_token_ids = tokenizer(completion_texts, add_special_tokens=False)["input_ids"]

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    prompt_lengths: list[int] = []

    for prompt_ids, completion_ids in zip(prompt_token_ids, completion_token_ids, strict=True):
        # Remove trailing EOS from prompt so completions can extend cleanly
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]

        prompt_lengths.append(len(prompt_ids))
        sequence = list(prompt_ids)
        sequence.extend(completion_ids)
        if eos_token_id is not None:
            sequence.append(eos_token_id)

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))

        labels = seq_tensor.clone()
        labels[: len(prompt_ids)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        labels_list.append(labels)

    teacher_input_ids = pad(
        sequences,
        padding_side="right",
        padding_value=pad_token_id if pad_token_id is not None else 0,
    )
    teacher_attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

    if eos_token_id is not None:
        for row in range(teacher_attention_mask.size(0)):
            valid = (
                teacher_input_ids[row] != pad_token_id
                if pad_token_id is not None
                else teacher_attention_mask[row].bool()
            )
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1]
                teacher_attention_mask[row, last_idx + 1 :] = False

    teacher_prompt_length = max(prompt_lengths) if prompt_lengths else 0

    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_prompt_length


def build_alignment_groups_from_ids(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_token_ids: list[int],
    teacher_token_ids: list[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Build alignment groups using a greedy substring-equality algorithm on decoded token pieces.

    Adapted from TRL's _build_alignment_groups_from_ids.

    Args:
        student_tokenizer: Student tokenizer
        teacher_tokenizer: Teacher tokenizer
        student_token_ids: List of student token IDs
        teacher_token_ids: List of teacher token IDs

    Returns:
        student_alignment_groups: List of student token index groups
        teacher_alignment_groups: List of teacher token index groups
    """

    def to_canonical_pieces(tok, ids):
        pieces = []
        prev = ""
        for k in range(len(ids)):
            # IMPORTANT: Do NOT skip special tokens - we need to align them too
            cur = tok.decode(ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # Extract the incremental addition (may include spaces/ZWJ/etc.)
            pieces.append(cur[len(prev) :])
            prev = cur
        return pieces

    s_pieces = to_canonical_pieces(student_tokenizer, student_token_ids)
    t_pieces = to_canonical_pieces(teacher_tokenizer, teacher_token_ids)

    i = j = 0
    s_buf = t_buf = ""
    s_group: list[int] = []
    t_group: list[int] = []
    s_groups: list[list[int]] = []
    t_groups: list[list[int]] = []

    def flush():
        if s_group and t_group:
            s_groups.append(s_group.copy())
            t_groups.append(t_group.copy())

    # Greedily accumulate pieces until substrings match, then flush
    while i < len(s_pieces) or j < len(t_pieces):
        if s_buf == t_buf and s_buf != "":
            flush()
            s_buf = t_buf = ""
            s_group = []
            t_group = []
            continue

        if s_buf == "" and i < len(s_pieces):
            s_buf += s_pieces[i]
            s_group.append(i)
            i += 1
            continue
        if t_buf == "" and j < len(t_pieces):
            t_buf += t_pieces[j]
            t_group.append(j)
            j += 1
            continue

        if len(s_buf) <= len(t_buf):
            if i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
            elif j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
        else:
            if j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
            elif i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1

    # Flush any remainder if both sides accumulated something
    if s_buf == t_buf and s_group and t_group:
        flush()
    elif s_group or t_group:
        # Handle remaining unmatched tokens by forcing a flush.
        # This ensures that if one sequence is longer than the other,
        # the remaining tokens are still captured in a final, possibly misaligned, group.
        s_groups.append(s_group.copy() if s_group else [])
        t_groups.append(t_group.copy() if t_group else [])

    return s_groups, t_groups


def merge_probabilities_with_alignment_groups(
    probs: torch.Tensor,
    alignment_groups: list[list[int]],
) -> torch.Tensor:
    """
    Merge probabilities based on alignment groups via element-wise product + renormalization.

    For multi-token groups, computes softmax(sum of log-probs), which is equivalent to
    normalizing the element-wise product of probability distributions.

    Adapted from TRL's _merge_probabilities_with_alignment_groups.

    Args:
        probs: Probability tensor [seq_len, vocab_size]
        alignment_groups: List of alignment groups (each group is a list of positions to merge)

    Returns:
        Merged probability tensor [num_groups, vocab_size]
    """
    if not alignment_groups:
        return probs

    vocab_size = probs.size(-1)
    target_len = len(alignment_groups)
    aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device, dtype=probs.dtype)

    for group_idx, group in enumerate(alignment_groups):
        if len(group) > 1:
            # Multiple tokens map to this group - merge via element-wise product + renormalization
            # This computes: softmax(log(p1) + log(p2) + ...) = normalized(p1 * p2 * ...)
            eps = 1e-8
            # Vectorized operation is more efficient than a loop
            logp = torch.log(probs[group].clamp_min(eps)).sum(dim=0)
            aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
        elif len(group) == 1:
            aligned_probs[group_idx] = probs[group[0]]
        else:
            # No tokens map to this group
            aligned_probs[group_idx] = torch.zeros_like(probs[0])

    return aligned_probs
