# New Model Support in SkyRL

Assumptions: we have a vLLM PR with initial support for serving a model with tensor-parallel serving. Huggingface transformers also has a commit that supports the model type.

Goal: We want to provide E2E training support in SkyRL for that model using Megatron and vLLM, supporting both LoRA, full finetuning, and advanced parallelism strategies.
This should be done in stages, and we should aim to identify and fix gaps in upstream vLLM, Megatron-LM (megatron.core), and Megatron-Bridge.

An example PR will have the file structure of https://github.com/NovaSky-AI/SkyRL/pull/2179, where all patches to vLLM, Megatron-LM (megatron.core), and Megatron-Bridge are in the `skyrl/backends/skyrl_train/patches/` folder,
and will be able to be removed by an agent once the upstream dependency gaps are resolved. It will also have an example GSM8K full finetuning and LoRA script on a minimal number of GPU nodes, and 
an example DAPO script again with full finetuning and LoRA on minimal nodes.

Other requirements: a user provided WANDB_API_KEY so that the runs are logged and tracked, a gh token or gh auth to push to a PR branch, and a huggingface token to create the slice for stage 0.

# Stages
## Stage 0: Model Architecture translation to Megatron, and roundtrip logprobs test
Goal: Create a test_megatron_models.py and test_megatron_lora_models.py test on a N layer slice of the model (where N is max(4, number of layers to include all layer types/patterns), like if we have N dense layers before MoE, and some pattern of linear vs quadratic/sparse attention).

Details: In this stage, we should create a representative slice of the model to add to CI (with the model slice being live on huggingface), to ensure that the Megatron <> vLLM round trip weight sync and generation match each other. We should also run the test with the full model weights but we don't need to add that full size test to the final PR. This should run and pass on the h100 ci on a 4xH100 slice. This will require:

Megatron Side:
1. Megatron-Core model component definition
2. Megatron-Bridge HF <> GPT/HybridModel mapping with LoRA included

vLLM Side:
1. Basic TP serving and weight reloading (full layerwise weight reloading correct for full finetuning)
2. LoRA mapping implemented for new model

This step does not include running a backward pass, since the roundtrip logprob test just requires a megatron and vLLM forward pass. The compute requirements are just whatever is needed to do BF16 forward passes. We should use a wheel built off of the vllm branch, and ideally keep megatron-core/megatron-bridge versions pinned to whatever is current and implement all megatron layers on top of the patches/megatron folder. We should compile a list of issues to raise with vllm, megatron-core, and megatron-bridge to help upstream stable model support.

## Stage 1: Basic E2E test with GSM8K: LoRA and full finetuning
Goal: Verify E2E training works on GSM8K with minimal bells and whistles

Details: In this stage, we should get standard synchronous GSM8K training running with both LoRA and full finetuning. We might not see reward increase, since GSM8K is largely maxxed out, but we should see some signal on things like logprob diff, and just validate that e2e training works. We should include a script on a minimal set of nodes, like we have for `examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh`. For inference parallelism, we should just use TP, and for training parallelism, we should keep it minimal (ideally just TP/EP should just be enough, but depending on model size we may need more). We should just run BF16 training and serving here.

## Stage 2: Medium sized E2E test with DAPO: LoRA and full finetuning
Goal: Verify E2E training on DAPO showing reward signal and R3 + checkpointing. This showing promising reward signal should be enough to establish initial stable model support

Details: Models should show learning signal on the basic sync DAPO recipe with LoRA with around ~8K context (this can vary depending on how verbose the model is). Even if the model is maxxed out on AIME, it should show reward signal from the DAPO recipe's overlong buffer reducing context length over training. The outcome for this stage should be scripts like `examples/train/glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_2node.sh` and `examples/train/glm5_3_flash/run_dapo_glm5p3_flash_fullft_sync_8node.sh` that show meaningful reward increase and shorter responses over the course of ~20 steps. We can still keep just whatever minimal parallelisms for train and inference, and use BF16 training and serving at this point. We should enable R3 for both LoRA and full finetuning runs to reduce logprob mismatch.

In this step we should also verify checkpoint save and resume is working correctly for LoRA and full finetuning. This involves adapting `tests/backends/skyrl_train/gpu/gpu_ci/test_trainer_full_checkpointing.py` with the new model, and verifying checkpointing works for both LoRA and full finetuning. 

## Stage 3: Production readiness and performance
Goal: Identify gaps in performance knobs (i.e. quantization)

### Training performance optimization
Goals: improve training throughput and enable long context training on the megatron side for the new model implementation.

Details: This will involve things like making sure that optimized kernels (i.e. SparseMLA in Tilelang for GLM 5.3 Flash) are properly enabled for new model shapes, making sure that advanced megatron perf features (i.e. overlap grad reduce, full activation recompute, optimizer offload/precision aware optimizer etc.) are supported for the new model arch.

### Quantization
Goals: enable full MXFP8 training at parity with BF16 training.

Details: We should add a new model spec defining what modules are quantized for a given model in `skyrl/backends/skyrl_train/weight_sync/fp8/models` - see the qwen35 example there. Then establish a BF16 baseline, and get MXFP8 training working E2E showing a matching training curve.


