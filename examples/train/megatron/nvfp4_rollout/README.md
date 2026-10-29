# NVFP4 rollouts: Qwen3-30B-A3B DAPO

`run_megatron_dapo_qwen3_30b_a3b_nvfp4.sh` is `../run_megatron_dapo_qwen3_30b_a3b.sh` on one 8xB200
node, with vLLM quantizing the routed experts to NVFP4 online:

```
generator.inference_engine.engine_init_kwargs.quantization=nvfp4_per_token
generator.inference_engine.engine_init_kwargs.kernel_config.moe_backend=flashinfer_trtllm
```

The Megatron trainer stays bf16. Every weight sync sends bf16 weights, and vLLM re-quantizes the
experts on reload. This covers the rollout side of
[humans& "The 4-bitter Lesson"](https://humansand.ai/blog/nvfp4-rl) only; none of its trainer-side
fixes (NVFP4 forward with a dequantized backward, 4/6 scaling, last ~15% of layers in bf16) are applied.

`SCORE_CENTERING=true` adds score centering (#2484): `policy_loss_type=rollout_is`,
`tis_ratio_type=null`, sample-support replay + score centering, `top_k=128`. That changes the loss
and the sampler too, so it is not a pure score-centering ablation against the plain NVFP4 arm.

## Colocated NVFP4 needs vLLM sleep level 1

On vLLM 0.32, `sleep(level=2)` breaks online NVFP4 MoE (`flashinfer_trtllm`): after wake + weight
reload every rollout is garbage (repetition until the length cap; AIME pass@32 0). SkyRL now clamps
the sleep level to 1 when `quantization` starts with `nvfp4` (`RemoteInferenceClient.effective_sleep_level`).

Standalone repro (Qwen3-30B-A3B-Base, TP8, eager, greedy):

| sequence | output |
|---|---|
| fresh load | correct |
| layerwise reload, no sleep | correct |
| sleep(1) + wake | correct |
| sleep(1) + wake + reload | correct |
| sleep(2) + wake + reload | **garbage** |
| sleep(2) + wake(weights) + reload + wake(kv_cache) | **garbage** |
| bf16: sleep(2) + wake + reload | correct |

Level 2 zeroes three tensors per MoE layer that are not rebuilt on reload: `routed_experts.g1_scale_c`
and `moe_quant_config._a1/_a2.alpha_or_gscale`. `make_nvfp4_moe_quant_config` builds the a1/a2 global
scales as non-parameter tensors in the weights pool, `Nvfp4OnlineMoEMethod._setup_kernel` only refreshes
`g1/g2_alphas` on reload, and `TrtLlmNvFp4Experts.process_weights_after_loading` then computes
`g1_scale_c = g1_alphas * a2_gscale = 0`. Restoring those tensors is not enough: after the restore all
non-KV GPU tensors reachable from the model match their pre-sleep checksums and the output is still
garbage, so more state is lost outside the model object (flashinfer kernel or a module-level cache).
Not root-caused further.

## Results (single seed, 10-11 steps)

Upstream main b88a3073, vLLM 0.32.0, flashinfer 0.7.0.post1. Hyperparameters unchanged from the base
recipe (batch 512x16, mini-batch 32, 8K responses, token TIS cap 2.0, `enforce_eager`). The bf16 arm
slept at level 2, both NVFP4 arms at level 1.

W&B project `sky-posttraining-uc-berkeley/nvfp4_rollout_repro`: bf16 `te8ahd2v` (steps 1-10) and
`p8yopqjs` (11+); NVFP4 `3ihka60y`; NVFP4 + score centering `v818vj4g`; the level-2 garbage run `iscnfmk5`.

Train accuracy = (avg_final_rewards + 1) / 2; logprob diff = `minibatch_rollout_logprobs_abs_diff_mean`.

| step | train acc bf16 / nvfp4 / nvfp4+SC | logprob diff | avg response length |
|---|---|---|---|
| 1 | .054 / .059 / .060 | .023 / .080 / .074 | 872 / 816 / 806 |
| 5 | .071 / .064 / .057 | .023 / .077 / .070 | 823 / 818 / 840 |
| 8 | .132 / .094 / .087 | .022 / .077 / .076 | 774 / 702 / 687 |
| 10 | .234 / .132 / .172 | .018 / .078 / .064 | 817 / 677 / 781 |
| 11 | .266 / - / .235 | .016 / - / - | 861 / - / 873 |

bf16 later: train acc .344 at step 15, .413 at step 19; logprob diff .012 by step 18.

AIME-2024, 32 samples per problem, mean accuracy / pass@32:

| step | bf16 | nvfp4 | nvfp4+SC |
|---|---|---|---|
| 0 | .007 / .13 | .006 / .17 | .006 / .10 |
| 5 | .024 / .30 | .006 / .20 | .011 / .23 |
| 10 | .156 / .47 | .055 / .43 | .116 / .33 |

Step time is ~57 min for every arm (generation ~19, forward logprobs ~5, policy training ~33), so
NVFP4 generation is not the bottleneck at this scale.

## Findings

1. Colocated NVFP4 needs sleep level 1 (above).
2. NVFP4 rollouts carry ~3.3x the rollout-vs-trainer logprob mismatch of bf16 (~.077 vs ~.023). It stays
   flat over training, while bf16's falls as the policy sharpens (.023 to .012).
3. Plain NVFP4 learns slower: by step 10 it is ~2-3 steps behind bf16 (train acc .132 vs .234,
   AIME .055 vs .156), and responses shrink faster (~650-680 tokens at steps 9-10 vs ~800).
4. Score centering closes part of the gap: train acc .172 at step 10 and .235 at step 11 (bf16: .234
   at step 10), AIME .116 at step 10, mismatch .064-.076. Its policy entropy reads lower (.53-.67 vs
   .80-.87), but `top_k=128` changes the sampling distribution, so entropy is not comparable across arms.
5. vLLM TP8 engine init occasionally hangs in `profile_run` (7 of 8 workers in `_dummy_run`, one idle in
   the shm broadcast dequeue); a relaunch fixes it.

Caveat: one seed and 10 steps. The gap shows in both train accuracy and AIME, but seed variance is
untested.
