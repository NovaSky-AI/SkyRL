# Expert-parallel all-to-all bench

Standalone tools for the MoE expert-parallel (EP) dispatch path. They need no model. They replay
the routes of a real batch, dumped with `trainer.dump_data_batch=true` and router replay on
(`{export_path}/dumped_data/global_step_{N}_training_input.pkl`).

| File | What it does | Needs |
|---|---|---|
| `moe_imbalance.py` | Load imbalance and cross-node rows from a dump | CPU |
| `make_synthetic.py` | Writes a synthetic dump to test the scripts | CPU |
| `extract_routes.py` | Dump -> one traffic matrix per all-to-all call (`routes.npz`) | CPU |
| `a2a_bench.py` | Replays `routes.npz` with random payload: stock `all_to_all_single` against point-to-point sends, for real / uniform / intra-node / cross-node traffic | one process per EP rank |
| `two_hop.py` | `DirectDispatch` (one row per token and expert, as Megatron's `alltoall` dispatcher) and `TwoHopDispatch` (one copy per destination node, then a fan-out inside the node) | library |
| `two_hop_bench.py` | Checks both dispatchers against an oracle on every call and times each phase | one process per EP rank |

All assume TP1, CP1, PP1, expert tensor parallel size 1, and that EP rank `r` is on node
`r // gpus_per_node`. Run from this directory.

```bash
# Correctness on CPU: 8 processes as 2 pseudo-nodes of 4
uv run --isolated --no-project --with torch --with numpy torchrun --nproc-per-node 8 two_hop_bench.py \
  --synthetic-tokens 2000 --num-experts 64 --topk 4 --hidden-size 32 --gpus-per-node 4 \
  --backend-kind gloo --device cpu --dtype float32 --transport p2p

# Timing on two 8-GPU nodes with real routes. Run on each node with the same rendezvous endpoint.
torchrun --nnodes 2 --nproc-per-node 8 --rdzv-endpoint HOST:PORT two_hop_bench.py --dump DUMP.pkl \
  --dp 16 --num-experts 512 --max-tokens-per-microbatch 10752 --hidden-size 2048 --layers 0,1 --calls 0
```

`two_hop_bench.py` prints `check: ... PASS`, the dispatch and combine time of each dispatcher (the
slowest rank per call), and the time of each phase. `--transport a2a` uses NCCL all-to-all inside
node groups and same-index groups; `--transport p2p` uses point-to-point sends and works on any
backend.
