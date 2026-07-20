# Delta Weight Sync Examples

These examples run non-colocated GSM8K training with checkpoint-delta weight
sync. The trainer publishes compressed XOR deltas to a shared location, the
inference side pulls and applies them into a local checkpoint before pausing
generation, and the paused phase reloads from that prepared checkpoint.

Prepare GSM8K data first:

```bash
uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
```

For Google Cloud Storage:

```bash
SYNC_DIR=gs://<bucket>/<prefix>/$(date +%Y%m%d_%H%M%S) \
bash examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_gcs.sh
```

For NFS or another shared POSIX filesystem:

```bash
SYNC_DIR=/mnt/shared_storage/skyrl-delta-sync/$(date +%Y%m%d_%H%M%S) \
bash examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_nfs.sh
```

For Qwen3.5-35B-A3B DAPO on two 8-GPU nodes:

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
SYNC_DIR=gs://<bucket>/<prefix>/$(date +%Y%m%d_%H%M%S) \
bash examples/train/delta_weight_sync/run_dapo_qwen3.5_35b_a3b_delta.sh
```

Use a unique `SYNC_DIR` per run. For `gs://` paths, GCS credentials and the
`gcloud` CLI must be available on the trainer and inference workers. For
`s3://` paths, install `s5cmd` on those workers. For shared filesystem paths,
all workers must see the same path.
