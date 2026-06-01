# Tinker Session-based Routing Example

This example shows how to exercise SkyRL's deterministic sampling-session routing — pinning a multi-turn trajectory to one inference engine by reusing a fixed `seq_id` across its sample calls.

Setup:
- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- backend: `--backend fsdp` (full fine-tune via `rank=0`)
- N parallel trajectories × T turns, each trajectory's turns share `seq_id=trajectory_idx`


## 1. Start the Tinker API server

```bash
bash examples/tinker/session_based_routing/run_tinker_server.sh
```

The launcher already includes server-side defaults for colocated placement, four inference engines, and micro-batching. To override them, pass your own `BACKEND_CONFIG=...`:

```bash
BACKEND_CONFIG='{
  "trainer.placement.policy_num_gpus_per_node": 2,
  "generator.inference_engine.num_engines": 2
}' bash examples/tinker/session_based_routing/run_tinker_server.sh
```

## 2. Run the routing demo client

```bash
TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with transformers \
    python examples/tinker/session_based_routing/sample_session_routing_demo.py \
    --num-trajectories 4 --turns 3
```

Each `/api/v1/asample` dispatch is logged on both sides with the same routing key:

```
client: dispatch routing-key=sampling_<id>:<seq_id>
server: [sticky-routing] dispatch idx=<i> model=<...> session_id=sampling_<id>:<seq_id>
```

All turns of trajectory `i` share `session_id=sampling_<id>:i` and therefore land on the same backend; distinct trajectories spread across backends.

## Notes

- The demo uses `service_client.create_lora_training_client(..., rank=0)` to bootstrap the policy via the SkyRL-Train backend, then `save_weights_and_get_sampling_client(...)` to obtain a sampling session. Everything after that bypasses `SamplingClient.sample` and calls `client.sampling.asample` directly so the caller controls `seq_id`.
- See [docs/content/docs/tinker/session_based_routing.mdx](../../../docs/content/docs/tinker/session_based_routing.mdx) for the feature design and routing-key contract.
