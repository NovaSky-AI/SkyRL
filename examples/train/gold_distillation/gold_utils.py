"""
GOLD (General On-policy Logit Distillation) Utilities for Cross-Tokenizer Distillation.

Adapted from TRL's GOLDTrainer/ULDLoss implementation:
https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py

This module provides utilities for distilling knowledge between models with different
tokenizers by:
1. Re-tokenizing student-generated text with the teacher's tokenizer
2. Aligning token spans between student and teacher sequences
3. Computing a hybrid loss combining JSD (for matching tokens) and L1 (for non-matching tokens)
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

    This implementation uses list accumulation + torch.stack() for autograd compatibility,
    so gradients flow through the student probabilities.

    Args:
        probs: Probability tensor [seq_len, vocab_size]
        alignment_groups: List of alignment groups (each group is a list of positions to merge)

    Returns:
        Merged probability tensor [num_groups, vocab_size]
    """
    if not alignment_groups:
        return probs

    eps = 1e-8
    merged = []
    for group in alignment_groups:
        if len(group) > 1:
            # Multiple tokens → merge via element-wise product + renormalization
            logp = torch.log(probs[group].clamp_min(eps)).sum(dim=0)
            merged.append(torch.softmax(logp, dim=-1))
        elif len(group) == 1:
            merged.append(probs[group[0]])
        else:
            merged.append(torch.zeros_like(probs[0]))

    return torch.stack(merged)


def compute_vocabulary_mapping(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Compute matched/unmatched vocabulary index sets between student and teacher tokenizers.

    Uses Jaccard-style exact string matching on decoded single tokens, following TRL's approach.

    Args:
        student_tokenizer: Student model tokenizer
        teacher_tokenizer: Teacher model tokenizer

    Returns:
        student_matched_indices: Sorted list of student vocab indices that have a match in teacher
        teacher_matched_indices: Corresponding teacher vocab indices (same order as student_matched)
        student_unmatched_indices: Student vocab indices with no teacher match
        teacher_unmatched_indices: Teacher vocab indices with no student match
    """
    # Build token string → index mapping for teacher
    teacher_token_to_idx: dict[str, int] = {}
    for idx in range(teacher_tokenizer.vocab_size):
        token_str = teacher_tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        teacher_token_to_idx[token_str] = idx

    student_matched = []
    teacher_matched = []
    student_unmatched = []

    for idx in range(student_tokenizer.vocab_size):
        token_str = student_tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if token_str in teacher_token_to_idx:
            student_matched.append(idx)
            teacher_matched.append(teacher_token_to_idx[token_str])
        else:
            student_unmatched.append(idx)

    # Teacher unmatched = all teacher indices not in matched set
    matched_teacher_set = set(teacher_matched)
    teacher_unmatched = [idx for idx in range(teacher_tokenizer.vocab_size) if idx not in matched_teacher_set]

    return student_matched, teacher_matched, student_unmatched, teacher_unmatched


def compute_jsd_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Compute generalized Jensen-Shannon Divergence loss.

    JSD_beta(P, Q) = beta * KL(P || M) + (1 - beta) * KL(Q || M)
    where M = beta * P + (1 - beta) * Q.

    Special cases:
        beta=0.0: Forward KL (KL(teacher || student))
        beta=1.0: Reverse KL (KL(student || teacher))
        beta=0.5: Symmetric JSD

    Args:
        student_probs: Student probability distribution [..., vocab_size] (with grad)
        teacher_probs: Teacher probability distribution [..., vocab_size] (no grad)
        beta: Interpolation parameter

    Returns:
        Scalar JSD loss (mean over all positions)
    """
    eps = 1e-8

    if beta == 0.0:
        # Forward KL: KL(teacher || student) = sum(teacher * log(teacher / student))
        log_student = torch.log(student_probs.clamp_min(eps))
        log_teacher = torch.log(teacher_probs.clamp_min(eps))
        kl = (teacher_probs * (log_teacher - log_student)).sum(dim=-1)
        return kl.mean()
    elif beta == 1.0:
        # Reverse KL: KL(student || teacher) = sum(student * log(student / teacher))
        log_student = torch.log(student_probs.clamp_min(eps))
        log_teacher = torch.log(teacher_probs.clamp_min(eps))
        kl = (student_probs * (log_student - log_teacher)).sum(dim=-1)
        return kl.mean()
    else:
        # General JSD
        m = beta * student_probs + (1 - beta) * teacher_probs
        log_m = torch.log(m.clamp_min(eps))
        log_student = torch.log(student_probs.clamp_min(eps))
        log_teacher = torch.log(teacher_probs.clamp_min(eps))

        kl_student_m = (student_probs * (log_student - log_m)).sum(dim=-1)
        kl_teacher_m = (teacher_probs * (log_teacher - log_m)).sum(dim=-1)

        jsd = beta * kl_student_m + (1 - beta) * kl_teacher_m
        return jsd.mean()


def compute_gold_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_alignment_groups: list[list[int]],
    teacher_alignment_groups: list[list[int]],
    student_matched_indices: torch.Tensor,
    teacher_matched_indices: torch.Tensor,
    student_unmatched_indices: torch.Tensor,
    teacher_unmatched_indices: torch.Tensor,
    student_labels: torch.Tensor | None = None,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    beta: float = 0.0,
    matched_weight: float = 1.0,
    unmatched_weight: float = 1.0,
    distillation_weight: float = 1.0,
    crossentropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GOLD loss for a single sequence.

    The loss combines:
    1. JSD on matched (shared vocabulary) token probabilities
    2. L1 on sorted unmatched token probabilities
    3. Optional cross-entropy loss on student predictions

    Args:
        student_logits: Student logits [num_student_response_tokens, student_vocab_size] (with grad)
        teacher_logits: Teacher logits [num_teacher_response_tokens, teacher_vocab_size] (no grad)
        student_alignment_groups: List of student token index groups
        teacher_alignment_groups: List of teacher token index groups
        student_matched_indices: Tensor of student vocab indices that match teacher tokens
        teacher_matched_indices: Corresponding teacher vocab indices
        student_unmatched_indices: Student vocab indices with no teacher match
        teacher_unmatched_indices: Teacher vocab indices with no student match
        student_labels: Optional labels for cross-entropy [num_student_response_tokens]
        student_temperature: Temperature for student logits
        teacher_temperature: Temperature for teacher logits
        beta: JSD interpolation parameter
        matched_weight: Weight for JSD on matched tokens
        unmatched_weight: Weight for L1 on unmatched tokens
        distillation_weight: Weight for the distillation loss
        crossentropy_weight: Weight for the cross-entropy loss

    Returns:
        loss: Scalar loss value
        metrics: Dict of per-component loss values
    """
    # Convert logits to probabilities with temperature
    student_probs = F.softmax(student_logits / student_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / teacher_temperature, dim=-1)

    # Merge probabilities for aligned groups
    student_aligned = merge_probabilities_with_alignment_groups(student_probs, student_alignment_groups)
    teacher_aligned = merge_probabilities_with_alignment_groups(teacher_probs, teacher_alignment_groups)

    num_groups = min(student_aligned.size(0), teacher_aligned.size(0))
    if num_groups == 0:
        zero = student_logits.sum() * 0.0  # preserve grad graph
        return zero, {"jsd_loss": 0.0, "l1_loss": 0.0, "ce_loss": 0.0, "total_loss": 0.0}

    student_aligned = student_aligned[:num_groups]
    teacher_aligned = teacher_aligned[:num_groups]

    # Split into matched/unmatched vocabulary components
    # Matched: JSD on shared vocabulary tokens
    student_matched_probs = student_aligned[:, student_matched_indices]  # [groups, num_matched]
    teacher_matched_probs = teacher_aligned[:, teacher_matched_indices]  # [groups, num_matched]

    # Renormalize matched probabilities
    s_matched_sum = student_matched_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    t_matched_sum = teacher_matched_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    student_matched_normed = student_matched_probs / s_matched_sum
    teacher_matched_normed = teacher_matched_probs / t_matched_sum

    jsd_loss = compute_jsd_loss(student_matched_normed, teacher_matched_normed, beta=beta)

    # Unmatched: L1 on sorted probability mass
    student_unmatched_probs = student_aligned[:, student_unmatched_indices]  # [groups, num_s_unmatched]
    teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_indices]  # [groups, num_t_unmatched]

    # Sort each in descending order, pad to same length, then L1
    s_unmatched_sorted = student_unmatched_probs.sort(descending=True, dim=-1).values
    t_unmatched_sorted = teacher_unmatched_probs.sort(descending=True, dim=-1).values

    max_unmatched = max(s_unmatched_sorted.size(-1), t_unmatched_sorted.size(-1))
    if s_unmatched_sorted.size(-1) < max_unmatched:
        s_unmatched_sorted = F.pad(s_unmatched_sorted, (0, max_unmatched - s_unmatched_sorted.size(-1)))
    if t_unmatched_sorted.size(-1) < max_unmatched:
        t_unmatched_sorted = F.pad(t_unmatched_sorted, (0, max_unmatched - t_unmatched_sorted.size(-1)))

    l1_loss = F.l1_loss(s_unmatched_sorted, t_unmatched_sorted)

    # Combine distillation losses
    distill_loss = matched_weight * jsd_loss + unmatched_weight * l1_loss
    total_loss = distillation_weight * distill_loss

    # Optional cross-entropy loss
    ce_loss_val = 0.0
    if crossentropy_weight > 0.0 and student_labels is not None:
        ce_loss = F.cross_entropy(student_logits, student_labels, ignore_index=-100)
        total_loss = total_loss + crossentropy_weight * ce_loss
        ce_loss_val = ce_loss.item()

    metrics = {
        "jsd_loss": jsd_loss.item(),
        "l1_loss": l1_loss.item(),
        "ce_loss": ce_loss_val,
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics
