# Rollout Routing Replay (R3) with the Tinker API

Rollout Routing Replay (R3) stabilizes RL training of Mixture-of-Experts (MoE) models by
replaying the inference engine's expert selections during the Megatron training forward pass.
See the [R3 docs](../../../docs/content/docs/tinker/router_replay.mdx) and
[Ma et al., 2025](https://arxiv.org/abs/2510.11370).

## Run

```bash
# Terminal 1: start a SkyRL Tinker server with R3 enabled (MoE + Megatron, 8 GPUs)
bash examples/tinker/router_replay/run_tinker_server_megatron.sh

# Terminal 2: sample with routing capture, then forward_backward replaying it
TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with requests --with transformers \
    python examples/tinker/router_replay/router_replay_demo.py
```

The demo prints the mean `|train_logprob - rollout_logprob|` per prompt. To see the effect,
run the server again with `"trainer.policy.megatron_config.moe_enable_routing_replay": false`
in the backend config and compare — the mismatch is markedly larger without R3.

## How it works

- The server enables `trainer.policy.megatron_config.moe_enable_routing_replay=true`, which
  also turns on routed-experts capture in vLLM.
- Each `sample` response carries a `routed_experts` field (`[num_tokens - 1, num_layers, top_k]`).
- The client passes it back in `loss_fn_inputs["routed_experts"]` on `forward_backward`, and the
  Megatron backend replays it so training activates the same experts as inference.
