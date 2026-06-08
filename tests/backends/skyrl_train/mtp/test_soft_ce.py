"""CPU unit tests for the decoupled MTP draft losses and loss combination.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_soft_ce.py
"""

import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.mtp.draft_loss_wrapper import (
    DraftLossConfig,
    combine_policy_and_draft_loss,
)
from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
    draft_hard_ce,
    draft_soft_ce,
    shift_mask_for_mtp,
)


def test_vocab_parallel_soft_ce_matches_reference(monkeypatch):
    # The memory-lean _VocabParallelSoftCrossEntropy (NeMo-RL-style einsum + in-place softmax) must
    # match the plain full-vocab soft CE in both forward and gradient. We stub the TP all-reduce to a
    # no-op so a single shard behaves like the full (un-sharded) vocab, exercising the kernel on CPU.
    import torch.distributed as dist

    from skyrl.backends.skyrl_train.mtp.soft_ce import _VocabParallelSoftCrossEntropy

    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: t)

    torch.manual_seed(0)
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)
    g_out = torch.randn(2, 4)

    loss = _VocabParallelSoftCrossEntropy.apply(student, teacher, object())
    loss.backward(g_out)
    got_loss, got_grad = loss.detach(), student.grad.detach()

    student2 = student.detach().clone().requires_grad_(True)
    ref = -(F.softmax(teacher, -1) * F.log_softmax(student2, -1)).sum(-1)
    ref.backward(g_out)

    assert torch.allclose(got_loss, ref.detach(), atol=1e-5)
    assert torch.allclose(got_grad, student2.grad, atol=1e-5)


def test_vocab_parallel_soft_ce_preserves_input_dtype(monkeypatch):
    # Backward must return a grad in the student logits' original dtype (e.g. bf16), not fp32.
    import torch.distributed as dist

    from skyrl.backends.skyrl_train.mtp.soft_ce import _VocabParallelSoftCrossEntropy

    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: t)

    student = torch.randn(1, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(1, 3, 5, dtype=torch.bfloat16)
    _VocabParallelSoftCrossEntropy.apply(student, teacher, object()).sum().backward()
    assert student.grad.dtype == torch.bfloat16


def test_draft_soft_ce_chunked_matches_unchunked():
    # Sequence-chunking (+ gradient checkpointing) must be numerically identical to the whole-sequence
    # loss in both value and gradient -- it only bounds activation memory.
    torch.manual_seed(0)
    student = torch.randn(2, 9, 7)
    teacher = torch.randn(2, 9, 7)
    mask = torch.ones(2, 9)
    mask[0, 5:] = 0  # partial mask to exercise the masked denominator across chunk boundaries

    s1 = student.clone().requires_grad_(True)
    loss_full = draft_soft_ce(s1, teacher, mask)
    loss_full.backward()

    s2 = student.clone().requires_grad_(True)
    loss_chunked = draft_soft_ce(s2, teacher, mask, chunk_size=4)  # 9 -> chunks of 4,4,1
    loss_chunked.backward()

    assert torch.allclose(loss_full, loss_chunked, atol=1e-6)
    assert torch.allclose(s1.grad, s2.grad, atol=1e-6)


def test_draft_hard_ce_chunked_matches_unchunked():
    torch.manual_seed(1)
    student = torch.randn(2, 9, 7)
    labels = torch.randint(0, 7, (2, 9))
    mask = torch.ones(2, 9)
    mask[1, 6:] = 0

    s1 = student.clone().requires_grad_(True)
    loss_full = draft_hard_ce(s1, labels, mask)
    loss_full.backward()

    s2 = student.clone().requires_grad_(True)
    loss_chunked = draft_hard_ce(s2, labels, mask, chunk_size=4)
    loss_chunked.backward()

    assert torch.allclose(loss_full, loss_chunked, atol=1e-6)
    assert torch.allclose(s1.grad, s2.grad, atol=1e-6)


def test_draft_soft_ce_chunk_size_larger_than_seq_is_noop():
    # chunk_size >= seq_len must behave exactly like the un-chunked path.
    torch.manual_seed(2)
    student = torch.randn(1, 4, 5, requires_grad=True)
    teacher = torch.randn(1, 4, 5)
    mask = torch.ones(1, 4)
    assert torch.allclose(
        draft_soft_ce(student, teacher, mask, chunk_size=999),
        draft_soft_ce(student, teacher, mask),
        atol=1e-6,
    )


def test_draft_soft_ce_topk_matches_reference():
    # Single-device (TP=1) top-k soft CE must equal a reference: distill the teacher's top-k tokens,
    # renormalized over that set. Checks both value and gradient (the custom backward scatters
    # softmax(student)-softmax(teacher) to the top-k columns).
    from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk

    torch.manual_seed(0)
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    mask = torch.ones(2, 5)
    mask[0, 3:] = 0
    k = 4

    s1 = student.clone().requires_grad_(True)
    got = draft_soft_ce_topk(s1, teacher, mask, k=k)
    got.backward()

    # Reference: gather student at teacher's top-k, softmax/CE over the k set.
    s2 = student.clone().requires_grad_(True)
    t_vals, t_idx = teacher.topk(k, dim=-1)
    s_vals = s2.gather(-1, t_idx)
    t_p = F.softmax(t_vals, dim=-1)
    ref_per_token = -(t_p * F.log_softmax(s_vals, dim=-1)).sum(-1)
    ref = (ref_per_token * mask).sum() / mask.sum()
    ref.backward()

    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.allclose(s1.grad, s2.grad, atol=1e-6)


def test_draft_soft_ce_topk_roll_shift_matches_prerolled():
    # roll_shift (top-k on the un-rolled policy logits, then roll the small [B,S,k] result) must equal
    # pre-rolling the full teacher then top-k with roll_shift=0 -- in both value and gradient. This is
    # the memory optimization that avoids the full [S, vocab] rolled-teacher copy.
    from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk

    torch.manual_seed(0)
    student = torch.randn(2, 6, 11)
    teacher = torch.randn(2, 6, 11)
    mask = torch.ones(2, 6)
    shift, k = 2, 4

    s1 = student.clone().requires_grad_(True)
    got = draft_soft_ce_topk(s1, teacher, mask, k=k, roll_shift=shift)
    got.backward()

    s2 = student.clone().requires_grad_(True)
    pre_rolled = torch.roll(teacher, shifts=-shift, dims=1)
    ref = draft_soft_ce_topk(s2, pre_rolled, mask, k=k, roll_shift=0)
    ref.backward()

    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.allclose(s1.grad, s2.grad, atol=1e-6)


def test_draft_soft_ce_topk_memory_is_topk_sized():
    # The forward must not materialize a full-vocab tensor: gradient is nonzero only at the k columns.
    from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk

    torch.manual_seed(1)
    student = torch.randn(1, 3, 20, requires_grad=True)
    teacher = torch.randn(1, 3, 20)
    draft_soft_ce_topk(student, teacher, torch.ones(1, 3), k=5).backward()
    # exactly k nonzero grad columns per token.
    assert int((student.grad != 0).sum(-1).max()) <= 5


def test_soft_ce_matches_reference():
    torch.manual_seed(0)
    student = torch.randn(2, 5, 7, requires_grad=True)
    teacher = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5)

    ref = -(F.softmax(teacher, -1) * F.log_softmax(student, -1)).sum(-1)
    ref_mm = (ref * mask).sum() / mask.sum()
    got = draft_soft_ce(student, teacher, mask)
    assert torch.allclose(got, ref_mm, atol=1e-6)


def test_soft_ce_gradient_is_softmax_difference():
    # d/d student of soft CE is softmax(student) - softmax(teacher), spread over the mask mean.
    torch.manual_seed(1)
    student = torch.randn(2, 4, 6, requires_grad=True)
    teacher = torch.randn(2, 4, 6)
    mask = torch.ones(2, 4)

    draft_soft_ce(student, teacher, mask).backward()
    n = mask.sum()
    expected = (F.softmax(student.detach(), -1) - F.softmax(teacher, -1)) * (mask.unsqueeze(-1) / n)
    assert torch.allclose(student.grad, expected, atol=1e-6)


def test_soft_ce_respects_mask():
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    # Masked-out position must not affect the loss value.
    teacher_alt = teacher.clone()
    teacher_alt[0, 1] = torch.randn(5)
    a = draft_soft_ce(student, teacher, mask)
    b = draft_soft_ce(student, teacher_alt, mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_hard_ce_matches_reference():
    torch.manual_seed(2)
    student = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.randint(0, 7, (2, 5))
    mask = torch.ones(2, 5)

    got = draft_hard_ce(student, labels, mask)
    ref = (-F.log_softmax(student, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()
    assert torch.allclose(got, ref, atol=1e-6)


def test_build_teacher_logits_rolls_and_detaches():
    ml = torch.arange(2 * 4 * 3, dtype=torch.float, requires_grad=True).reshape(2, 4, 3)
    t0 = build_teacher_logits(ml, mtp_layer_number=0)
    t1 = build_teacher_logits(ml, mtp_layer_number=1)
    assert torch.equal(t0, torch.roll(ml.detach(), -1, dims=1))
    assert torch.equal(t1, torch.roll(ml.detach(), -2, dims=1))
    assert not t0.requires_grad


def test_shift_mask_zeros_boundary():
    m = torch.ones(1, 4)
    assert shift_mask_for_mtp(m, 0).tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert shift_mask_for_mtp(m, 1).tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_combine_is_policy_plus_weighted_draft():
    torch.manual_seed(3)
    policy_loss = torch.tensor(2.0, requires_grad=True)
    student = torch.randn(2, 5, 7, requires_grad=True)
    main_logits = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5)
    cfg = DraftLossConfig(loss_weight=0.5, loss_type="soft_ce")

    combined, metrics = combine_policy_and_draft_loss(
        policy_loss, [student], main_logits, mask, cfg
    )
    # Recompute the draft term independently and check the combination identity.
    teacher = build_teacher_logits(main_logits, 0)
    draft = draft_soft_ce(student, teacher, shift_mask_for_mtp(mask, 0))
    assert torch.allclose(combined, policy_loss + 0.5 * draft, atol=1e-6)
    assert abs(metrics["mtp_loss"] - draft.item()) < 1e-6


def test_combine_hard_ce_uses_rolled_labels():
    torch.manual_seed(4)
    policy_loss = torch.tensor(1.0)
    student = torch.randn(1, 5, 7, requires_grad=True)
    main_logits = torch.randn(1, 5, 7)
    mask = torch.ones(1, 5)
    base_labels = torch.randint(0, 7, (1, 5))
    cfg = DraftLossConfig(loss_weight=1.0, loss_type="hard_ce")

    combined, _ = combine_policy_and_draft_loss(
        policy_loss, [student], main_logits, mask, cfg, hard_labels=base_labels
    )
    layer_labels = torch.roll(base_labels, shifts=-1, dims=1)
    draft = draft_hard_ce(student, layer_labels, shift_mask_for_mtp(mask, 0))
    assert torch.allclose(combined, policy_loss + draft, atol=1e-6)
