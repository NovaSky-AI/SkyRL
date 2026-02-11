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
import torch.nn as nn
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
        # Handle remaining unmatched tokens by forcing a flush
        if s_group or t_group:
            if not s_group:
                s_group = []
            if not t_group:
                t_group = []
            if s_group or t_group:
                s_groups.append(s_group.copy() if s_group else [])
                t_groups.append(t_group.copy() if t_group else [])

    return s_groups, t_groups


def merge_probabilities_with_alignment_groups(
    probs: torch.Tensor,
    alignment_groups: list[list[int]],
) -> torch.Tensor:
    """
    Merge probabilities based on alignment groups using geometric mean.

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
            # Multiple tokens map to this group - merge via geometric mean
            eps = 1e-8
            logp = torch.log(probs[group[0]].clamp_min(eps))
            for idx in group[1:]:
                if idx < probs.size(0):
                    logp = logp + torch.log(probs[idx].clamp_min(eps))
            aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
        elif len(group) == 1:
            aligned_probs[group_idx] = probs[group[0]]
        else:
            # No tokens map to this group
            aligned_probs[group_idx] = torch.zeros_like(probs[0])

    return aligned_probs


def generalized_jsd_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Generalized Jensen-Shannon Divergence loss.

    Adapted from TRL's GOLDTrainer.generalized_jsd_loss.

    Args:
        student_probs: Student probabilities [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        teacher_probs: Teacher probabilities [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        beta: Interpolation coefficient (default: 0.5)

    Returns:
        JSD loss scalar
    """
    student_log_probs = torch.log(student_probs.clamp_min(1e-8))
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-8))

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        # Mixture distribution
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
            dim=0,
        )

        # KL divergences
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    return jsd.sum() / jsd.size(0)


class ULDLoss(nn.Module):
    """
    Universal Logit Distillation Loss for cross-tokenizer distillation.

    Adapted from TRL's ULDLoss class.

    This loss handles models with different tokenizers by:
    1. Aligning token spans using greedy text matching
    2. Merging probabilities for aligned spans
    3. Computing hybrid loss: JSD for matched vocab tokens, sorted L1 for unmatched
    """

    def __init__(
        self,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        crossentropy_weight: float = 0.0,
        distillation_weight: float = 1.0,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        skip_student_eos: bool = False,
        skip_teacher_eos: bool = False,
        use_extended_uld: bool = True,
        use_hybrid_loss: bool = True,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.use_extended_uld = use_extended_uld
        self.use_hybrid_loss = use_hybrid_loss
        self.beta = beta
        self.ignore_index = -100

        # Initialize vocabulary mapping for hybrid loss
        self._vocab_mapping: dict[int, int] = {}
        self._teacher_matched_ids: set[int] = set()
        self._student_matched_ids: set[int] = set()
        self._initialize_vocabulary_mapping()

    def _initialize_vocabulary_mapping(self):
        """Initialize vocabulary mapping for hybrid ULD loss."""
        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()

        student_token_to_id = dict(student_vocab.items())

        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                self._vocab_mapping[teacher_id] = student_id
                self._teacher_matched_ids.add(teacher_id)
                self._student_matched_ids.add(student_id)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        student_input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ULD loss.

        Args:
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            student_labels: Student target labels [batch_size, seq_len]
            teacher_labels: Teacher target labels [batch_size, seq_len]
            student_input_ids: Student input token IDs [batch_size, seq_len]
            teacher_input_ids: Teacher input token IDs [batch_size, seq_len]

        Returns:
            Total loss (cross-entropy + distillation)
        """
        # Compute cross-entropy loss for student
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = torch.tensor(0.0, device=student_logits.device)

        # Compute distillation loss
        distillation_loss = self._compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
        )

        return crossentropy_loss + distillation_loss

    def _get_start_and_size_answers(self, answer_tensors: torch.Tensor) -> tuple[list[int], list[int]]:
        """Get start index and size of answer regions (non -100 labels)."""
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            answer_mask = answer.ne(self.ignore_index)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            valid_indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(valid_indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size

    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        student_input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Universal Logit Distillation loss with token mapping.
        """
        # Get answer regions
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]

        # Handle edge case where all answer sizes are 0
        if (
            not student_answer_size
            or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * student_logits.sum() * 1e-8

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            # Get answer regions for this batch item
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            if student_size <= 0 or teacher_size <= 0:
                loss_i = student_logits[i].sum() * 0.0
                distillation_losses.append(loss_i)
                continue

            # Extract answer logits
            student_answer_logits = student_logits[i, student_start : student_start + student_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]

            # Convert to probabilities
            student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)

            # Get token IDs for mapping
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()

            if self.use_extended_uld:
                # Build alignment groups directly from token ids using greedy text matching
                student_alignment_groups, teacher_alignment_groups = build_alignment_groups_from_ids(
                    self.student_tokenizer, self.teacher_tokenizer, student_token_ids, teacher_token_ids
                )

                # Merge student probabilities using student alignment groups
                student_aligned = merge_probabilities_with_alignment_groups(student_probs, student_alignment_groups)

                # Merge teacher probabilities using teacher alignment groups
                teacher_aligned = merge_probabilities_with_alignment_groups(teacher_probs, teacher_alignment_groups)
            else:
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]

            # Apply ULD loss computation
            if self.use_hybrid_loss and self._vocab_mapping:
                # Use hybrid approach: direct comparison for matched tokens, sorting for unmatched
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                # Original approach: sort all probabilities
                student_sorted = student_aligned.sort(dim=-1, descending=True).values
                teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

                # Pad vocabularies to same size
                student_vocab_size = student_sorted.size(-1)
                teacher_vocab_size = teacher_sorted.size(-1)
                max_vocab_size = max(student_vocab_size, teacher_vocab_size)

                if student_vocab_size < max_vocab_size:
                    student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
                if teacher_vocab_size < max_vocab_size:
                    teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

                # Compute L1 distance (ULD approach)
                aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
                aligned_loss /= student_aligned.size(0)  # Normalize by sequence length

            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return self.distillation_weight * distillation_loss

    def _compute_hybrid_uld_loss(
        self,
        student_aligned: torch.Tensor,
        teacher_aligned: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid ULD loss on aligned probability distributions.

        1. Directly compares probabilities for tokens with matching vocabulary entries (JSD)
        2. Uses sorting approach only for tokens with different vocabulary entries (L1)
        """
        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        # Convert sets to sorted tensors for indexing
        if self._teacher_matched_ids:
            teacher_matched_indices = torch.tensor(sorted(self._teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_indices = torch.tensor(
                [self._vocab_mapping[tid.item()] for tid in teacher_matched_indices], dtype=torch.long, device=device
            )
        else:
            teacher_matched_indices = torch.tensor([], dtype=torch.long, device=device)
            student_matched_indices = torch.tensor([], dtype=torch.long, device=device)

        # Create masks for unmatched tokens
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_indices) > 0:
            teacher_matched_mask[teacher_matched_indices] = True
            student_matched_mask[student_matched_indices] = True

        # 1. JSD loss for matched vocabulary tokens
        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_indices) > 0:
            teacher_matched_probs = teacher_aligned[:, teacher_matched_indices]
            student_matched_probs = student_aligned[:, student_matched_indices]
            matched_token_count = teacher_matched_probs.size(-1)
            matched_loss = generalized_jsd_loss(student_matched_probs, teacher_matched_probs, beta=self.beta)

        # 2. Sorted comparison loss for unmatched vocabulary tokens
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            # Pad to same size if needed
            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            unmatched_loss /= student_aligned.size(0)

        # 3. Adaptive weighting based on vocabulary overlap
        matched_weight = matched_token_count / max(1, teacher_vocab_size)
        unmatched_weight = 1.0 - matched_weight

        return matched_weight * matched_loss + unmatched_weight * unmatched_loss


def compute_per_token_gold_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    temperature: float = 1.0,
    use_extended_uld: bool = True,
) -> torch.Tensor:
    """
    Compute per-token GOLD loss for use as rewards in RL training.

    This is a simplified version that returns per-token losses that can be used
    as dense reward signals in the SkyRL training loop.

    Args:
        student_logits: Student logits [batch, seq_len, student_vocab]
        teacher_logits: Teacher logits [batch, seq_len, teacher_vocab]
        student_input_ids: Student input IDs [batch, seq_len]
        teacher_input_ids: Teacher input IDs [batch, seq_len]
        loss_mask: Mask for which tokens to include [batch, seq_len]
        student_tokenizer: Student tokenizer
        teacher_tokenizer: Teacher tokenizer
        temperature: Temperature for softmax
        use_extended_uld: Whether to use alignment-based ULD

    Returns:
        Per-token losses [batch, seq_len] (higher = worse match with teacher)
    """
    batch_size, seq_len = student_logits.shape[:2]
    device = student_logits.device

    per_token_losses = torch.zeros(batch_size, seq_len, device=device)

    for i in range(batch_size):
        # Get mask for this sample
        mask = loss_mask[i].bool()
        if not mask.any():
            continue

        # Get the valid token positions
        valid_positions = mask.nonzero(as_tuple=True)[0]
        start_pos = valid_positions[0].item()
        end_pos = valid_positions[-1].item() + 1

        # Extract logits for the response region
        student_resp_logits = student_logits[i, start_pos:end_pos]
        teacher_resp_logits = teacher_logits[i, start_pos:end_pos]

        # Convert to probabilities
        student_probs = F.softmax(student_resp_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_resp_logits / temperature, dim=-1)

        # Get token IDs for alignment
        student_token_ids = student_input_ids[i, start_pos:end_pos].tolist()
        teacher_token_ids = teacher_input_ids[i, start_pos:end_pos].tolist()

        if use_extended_uld and len(student_token_ids) > 0 and len(teacher_token_ids) > 0:
            # Build alignment groups
            student_groups, teacher_groups = build_alignment_groups_from_ids(
                student_tokenizer, teacher_tokenizer, student_token_ids, teacher_token_ids
            )

            if student_groups and teacher_groups:
                # Merge probabilities
                student_aligned = merge_probabilities_with_alignment_groups(student_probs, student_groups)
                teacher_aligned = merge_probabilities_with_alignment_groups(teacher_probs, teacher_groups)

                # Compute per-group L1 loss
                min_groups = min(student_aligned.size(0), teacher_aligned.size(0))
                for g in range(min_groups):
                    # Sort and compare
                    s_sorted = student_aligned[g].sort(descending=True).values
                    t_sorted = teacher_aligned[g].sort(descending=True).values

                    # Pad to same size
                    max_vocab = max(s_sorted.size(0), t_sorted.size(0))
                    if s_sorted.size(0) < max_vocab:
                        s_sorted = F.pad(s_sorted, (0, max_vocab - s_sorted.size(0)))
                    if t_sorted.size(0) < max_vocab:
                        t_sorted = F.pad(t_sorted, (0, max_vocab - t_sorted.size(0)))

                    group_loss = F.l1_loss(s_sorted, t_sorted, reduction="sum")

                    # Assign loss to student tokens in this group
                    if g < len(student_groups):
                        for tok_idx in student_groups[g]:
                            if tok_idx < (end_pos - start_pos):
                                per_token_losses[i, start_pos + tok_idx] = group_loss / max(1, len(student_groups[g]))
        else:
            # Simple per-token comparison (no alignment)
            min_len = min(student_probs.size(0), teacher_probs.size(0))
            for t in range(min_len):
                s_sorted = student_probs[t].sort(descending=True).values
                t_sorted = teacher_probs[t].sort(descending=True).values

                max_vocab = max(s_sorted.size(0), t_sorted.size(0))
                if s_sorted.size(0) < max_vocab:
                    s_sorted = F.pad(s_sorted, (0, max_vocab - s_sorted.size(0)))
                if t_sorted.size(0) < max_vocab:
                    t_sorted = F.pad(t_sorted, (0, max_vocab - t_sorted.size(0)))

                per_token_losses[i, start_pos + t] = F.l1_loss(s_sorted, t_sorted, reduction="sum")

    return per_token_losses
