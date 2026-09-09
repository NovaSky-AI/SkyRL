"""Fixed-token SDK stages around a caller-owned optimizer update."""

import math

from tinker import types

from skyrl.tinker.logprob_checks import (
    build_probe_sequences,
    check_policy_snapshot,
    check_updated_adapter,
)
from skyrl.tinker.logprob_checks import (
    check_withheld_publication as check_withheld_scores,
)


def prepare_probes(trainer):
    return [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
        )
        for tokens in build_probe_sequences(trainer.get_tokenizer())
    ]


def score_trainer(trainer, probes):
    result = trainer.forward(probes, "cross_entropy").result()
    if len(result.loss_fn_outputs) != len(probes):
        raise ValueError("wrong training datum count")
    scores = []
    for datum, output in zip(probes, result.loss_fn_outputs, strict=True):
        values = output["logprobs"].data
        if len(values) != len(datum.model_input.to_ints()) or not all(map(math.isfinite, values)):
            raise ValueError("wrong training token count or nonfinite logprobs")
        scores.extend(values)
    return scores


def score_sampler(sampler, probes):
    scores = []
    for datum in probes:
        tokens = datum.model_input.to_ints() + [datum.loss_fn_inputs["target_tokens"].data[-1]]
        result = sampler.compute_logprobs(types.ModelInput.from_ints(tokens)).result()
        if len(result) != len(tokens) or result[0] is not None:
            raise ValueError("sampler must return one score per token with an unscored first token")
        if any(value is None or not math.isfinite(value) for value in result[1:]):
            raise ValueError("missing or nonfinite sampler score")
        scores.extend(result[1:])
    return scores


def score_before_update(trainer, sampler, probes, report, atol):
    report.update(
        trainer_zero=score_trainer(trainer, probes),
        zero=score_sampler(sampler, probes),
        trainer_repeat=score_trainer(trainer, probes),
        repeat=score_sampler(sampler, probes),
    )
    check_policy_snapshot(report, atol)


def check_withheld_publication(trainer, sampler, probes, report):
    report["trainer_updated"] = score_trainer(trainer, probes)
    report["stale"] = score_sampler(sampler, probes)
    check_withheld_scores(report)


def check_published_update(sampler, probes, report, atol, delta_atol):
    report["updated"] = score_sampler(sampler, probes)
    check_updated_adapter(report, atol, delta_atol)
