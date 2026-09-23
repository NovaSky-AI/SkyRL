"""Compare fixed-token trainer and inference scores before and after LoRA perturbation."""

from examples.model_checks.logprob_checks import (
    build_probe_sequences,
    check_agreement,
    compare_logprobs,
)
from examples.model_checks.megatron_lora import (
    build_batch,
    perturb_trainer,
    publish,
    score_sampler,
    score_trainer,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name


async def check_logprobs(policy, client, cfg, tokenizer):
    sequences = build_probe_sequences(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_id)
    lora = cfg.trainer.policy.model.lora.rank > 0
    model = resolve_policy_model_name(cfg) if lora else client.model_name
    colocated = cfg.trainer.placement.colocate_all
    scores = {}
    for phase in ["zero", "perturbed"] if lora else ["full_ft"]:
        if phase == "perturbed":
            perturb_trainer(policy)
        trainer = score_trainer(policy, batch)
        if colocated:
            policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
            await client.wake_up(tags=["weights"])
        await publish(policy, client, cfg)
        if colocated:
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await client.wake_up(tags=["kv_cache"])
        inference = await score_sampler(client, sequences, model)
        repeat = await score_sampler(client, sequences, model)
        difference = compare_logprobs(trainer, inference)
        print(f"{phase}: {difference}", flush=True)
        check_agreement(difference, mean_atol=0.05, max_atol=0.5)
        assert compare_logprobs(inference, repeat)["max_abs"] <= 1e-6
        scores[phase] = {"trainer": trainer, "inference": inference}
        if colocated:
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)
    if lora:
        for backend in ("trainer", "inference"):
            assert compare_logprobs(scores["zero"][backend], scores["perturbed"][backend])["max_abs"] > 1e-6
    return scores
