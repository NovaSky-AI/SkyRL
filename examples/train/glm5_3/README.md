# GLM-5.3 DAPO LoRA

This BF16 recipe uses four 8×B300 nodes. Each node alternates between a TP8/EP8
trainer replica and a TP8/EP8 vLLM engine. It uses R3 routing replay, rank-32 LoRA
with shared expert adapters, and in-memory adapter synchronization.

Use SkyRL's Megatron installation and start a Ray cluster across the nodes.
The same checkout, model and data must be available on every node. `RUN_ROOT`
must be on shared storage; a node-local directory is not sufficient.

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
RUN_ROOT=/shared/glm53-dapo RAY_ADDRESS=auto \
  bash examples/train/glm5_3/run_dapo_glm5p3_lora.sh
```

`MODEL_PATH`, `DATA_DIR`, and `NUM_NODES` can be overridden in the environment.
Additional config overrides can be passed as arguments. Changing the node count
requires checking batch divisibility and available memory.

Each step collects 128 prompts × 16 responses (2,048 trajectories), then performs
four optimizer updates with 32-prompt minibatches. The 20-step run has 80 updates;
the learning rate warms up over the first 40 updates. AIME evaluation runs before
training and every five steps, with 32 responses per problem. Responses are capped
at 8,192 tokens, with DAPO's soft length penalty above 4,096 tokens and loss masking
for unfinished responses. Dynamic sampling is disabled.

This is a current-main adaptation of the full-model learning run, not its exact
historical configuration: it uses current main and in-memory synchronization,
and evaluates the base policy in the same run. The historical run used file-based
synchronization. This updated recipe has been configuration-checked but has not yet
been rerun end to end on GPUs; memory headroom must include adapter loading.

For GLM-5.3 Flash, use the existing [two-node DAPO LoRA recipe](../glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_2node.sh).
It has separate topology, LoRA, and optimization settings; it is not the exact
Tinker API configuration used for the published Flash learning curve.
