"""CPU unit tests for the decoupled MTP draft losses.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_soft_ce.py
"""

import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
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


def test_draft_losses_reject_attached_teacher():
    # The teacher is the policy's own distribution; an attached one would train the policy through the
    # draft loss. draft_soft_ce's non-vocab-parallel path uses plain ops and would propagate it
    # silently, so both losses assert the caller detached it.
    from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk

    student = torch.randn(1, 3, 6, requires_grad=True)
    attached_teacher = torch.randn(1, 3, 6, requires_grad=True)
    mask = torch.ones(1, 3)
    for loss_fn in (
        lambda: draft_soft_ce(student, attached_teacher, mask),
        lambda: draft_soft_ce_topk(student, attached_teacher, mask, k=3),
    ):
        with pytest.raises(AssertionError, match="must be detached"):
            loss_fn()
    # A detached teacher is accepted.
    draft_soft_ce(student, attached_teacher.detach(), mask)


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


def test_shift_mask_left_padded_does_not_leak_pad_source():
    # Bug A regression: a left-padded row [PAD PAD t0 t1 t2]. Rolling the mask left makes the
    # last pad slot (idx 1) point at the first real token, which the OLD code unmasked -> a
    # de-padded zero-logit (uniform) pad position leaked into the loss. The source-side AND must
    # keep only positions whose own token AND its t+shift target are real.
    m = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    # depth 0 (shift 1): valid sources are t0,t1 (targets t1,t2 real); t2 has no real target.
    assert shift_mask_for_mtp(m, 0).tolist() == [[0.0, 0.0, 1.0, 1.0, 0.0]]
    # depth 1 (shift 2): only t0 has a real target (t2); pad idx0/idx1 must stay 0.
    assert shift_mask_for_mtp(m, 1).tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]


def test_shift_mask_right_padded_is_unaffected():
    # Right padding never leaks (rolled mask is already a subset), so the fix is a no-op there.
    m = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    assert shift_mask_for_mtp(m, 0).tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]


def test_left_pad_zero_logits_do_not_inflate_loss():
    # End-to-end (loss-level) reproduction of Bug A: emulate the de-pad pipeline, which ZERO-fills
    # pad positions (postprocess_packed_seqs / recover_left_padding both use torch.zeros). With a
    # perfectly-aligned student at the real positions, the draft soft-CE must equal the teacher's
    # entropy over the real supervised positions -- the leaked uniform pad position must NOT inflate
    # it. Also asserts the leak (had it survived) is bounded by log(V), per the two-bug analysis.
    torch.manual_seed(0)
    V = 64
    pad = 2  # left padding
    real = 6
    seq = pad + real
    # Real-position teacher logits: moderately peaked so entropy << log(V).
    main_logits = torch.zeros(1, seq, V)
    main_logits[:, pad:, :] = torch.randn(1, real, V) * 3.0  # pad positions stay zero (de-pad fill)
    mask = torch.zeros(1, seq)
    mask[:, pad:] = 1.0

    # Perfectly-aligned student: student[t] == teacher target for depth 0 == main_logits[t+1].
    # Build it from the rolled teacher so soft-CE at real+aligned positions == teacher entropy.
    teacher = build_teacher_logits(main_logits, 0)  # roll(-1); pad positions stay zero
    student = teacher.clone()  # aligned where it matters; pad positions zero (uniform), like de-pad

    layer_mask = shift_mask_for_mtp(mask, 0)
    loss = draft_soft_ce(student, teacher, layer_mask)

    # Oracle: entropy of the teacher over exactly the source-AND-target-valid positions.
    valid = (mask[:, :].bool()) & (torch.roll(mask, -1, 1).bool())
    valid[:, -1:] = False
    tprob = F.softmax(teacher.float(), dim=-1)
    ent = -(tprob * torch.log_softmax(teacher.float(), -1)).sum(-1)
    oracle = (ent * valid).sum() / valid.sum()
    assert torch.allclose(loss, oracle, atol=1e-5), (loss.item(), oracle.item())
    # And the result is the true (low) entropy, nowhere near log(V).
    assert loss.item() < ent[valid].max().item() + 1e-4
    assert loss.item() < torch.log(torch.tensor(float(V))).item()


def test_unpadded_vocab_shard_width():
    from skyrl.backends.skyrl_train.mtp.soft_ce import unpadded_vocab_shard_width

    # Unknown true vocab -> no-op.
    assert unpadded_vocab_shard_width(None, 128, 0) == 128
    # No padding: every rank keeps its full shard.
    assert unpadded_vocab_shard_width(256, 128, 0) == 128
    assert unpadded_vocab_shard_width(256, 128, 1) == 128
    # 250 padded to 256 over TP=2: rank 0 full, rank 1 loses the 6-column tail.
    assert unpadded_vocab_shard_width(250, 128, 0) == 128
    assert unpadded_vocab_shard_width(250, 128, 1) == 122
    # Pathological: a rank whose entire shard is padding.
    assert unpadded_vocab_shard_width(100, 128, 1) == 0


def test_draft_soft_ce_topk_grad_dtype_matches_input():
    # The top-k backward must build its full-vocab grad buffer directly in the input dtype
    # (bf16), not fp32-then-cast — value-identical, half the transient at large vocab.
    from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk

    torch.manual_seed(0)
    student = torch.randn(1, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(1, 4, 9, dtype=torch.bfloat16)
    draft_soft_ce_topk(student, teacher, torch.ones(1, 4), k=3).backward()
    assert student.grad.dtype == torch.bfloat16
    assert int((student.grad != 0).sum(-1).max()) <= 3
