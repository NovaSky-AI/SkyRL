"""SDK probe alignment and reuse of native numerical assertions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.tinker import sdk_logprobs as checks


def make_probes():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [3, 7, 11])
    return checks.prepare_probes(SimpleNamespace(get_tokenizer=lambda: tokenizer))


def test_sampler_scores_match_shifted_training_targets_for_unequal_lengths():
    probes = make_probes()
    sampler = SimpleNamespace(
        compute_logprobs=Mock(
            side_effect=[
                SimpleNamespace(result=lambda: [None] + [-0.1] * 64),
                SimpleNamespace(result=lambda: [None] + [-0.2] * 128),
            ]
        )
    )
    assert checks.score_sampler(sampler, probes) == [-0.1] * 64 + [-0.2] * 128
    for call, datum in zip(sampler.compute_logprobs.call_args_list, probes, strict=True):
        assert call.args[0].to_ints()[1:] == datum.loss_fn_inputs["target_tokens"].data


def test_missing_sampler_scores_are_not_padded():
    sampler = SimpleNamespace(compute_logprobs=lambda prompt: SimpleNamespace(result=lambda: [None, -0.1]))
    with pytest.raises(ValueError, match="one score per token"):
        checks.score_sampler(sampler, make_probes())


@pytest.mark.parametrize("sampler_after,passes", [([-0.99, -2.04], True), ([-1.03, -1.98], False)])
def test_sdk_stages_use_native_update_direction_assertions(monkeypatch, sampler_after, passes):
    trainer_values = iter([[-1.0, -2.0], [-1.0, -2.0], [-0.98, -2.03]])
    sampler_values = iter([[-1.01, -2.01], [-1.01, -2.01], [-1.01, -2.01], sampler_after])
    monkeypatch.setattr(checks, "score_trainer", lambda *args: next(trainer_values))
    monkeypatch.setattr(checks, "score_sampler", lambda *args: next(sampler_values))
    report = {}
    checks.score_before_update(None, None, [], report, 0.05)
    checks.check_withheld_publication(None, None, [], report)
    if passes:
        checks.check_published_update(None, [], report, 0.05, 0.005)
        assert report["update_delta"]["mean_abs"] < 1e-10
    else:
        with pytest.raises(AssertionError):
            checks.check_published_update(None, [], report, 0.05, 0.005)
    assert report["updated"] == sampler_after


def test_explicit_tokenizer_preserves_probes_without_loading_remote_model_path():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [3, 7, 11])
    trainer = SimpleNamespace(get_tokenizer=Mock(return_value=tokenizer))
    expected = checks.prepare_probes(trainer)
    trainer.get_tokenizer = Mock(side_effect=AssertionError("remote path unavailable locally"))
    actual = checks.prepare_probes(trainer, tokenizer)
    trainer.get_tokenizer.assert_not_called()
    assert [datum.model_dump() for datum in actual] == [datum.model_dump() for datum in expected]
