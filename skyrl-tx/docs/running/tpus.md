# Running on TPUs

Currently we only have instructions how to run `tx` on a single TPU VM. Multi-node instructions will be added later.

## Setting up the TPU VM

First start the TPU VM:

```bash
gcloud compute tpus tpu-vm create <TPU_NAME> --project=<PROJECT> --zone=<ZONE> --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --scopes=https://www.googleapis.com/auth/cloud-platform.read-only,https://www.googleapis.com/auth/devstorage.read_write --network=<NETWORK> --subnetwork=<SUBNETWORK> --spot
```

After the VM is started, you can ssh into it via

```bash
gcloud compute tpus tpu-vm ssh <TPU_NAME>
```

## Setting up tx

Once you are logged into the VM, install `uv` and clone the `tx` repository with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tx-project/tx
cd tx
```

## Starting the training

Next, download the dataset with

```bash
uv run --with huggingface_hub hf download Qwen/Qwen3-4B --local-dir /tmp/qwen3
```

You can then start the training with

```bash
uv run --extra tpu --with jinja2 tx train --model Qwen/Qwen3-4B --dataset HuggingFaceH4/ultrachat_200k --loader tx.loaders.chat --split train_sft --output-dir /tmp/ultrachat --batch-size 8 --load-checkpoint-path /tmp/qwen3 --tp-size 8
```

Note that at the beginning the training is a little slow since the JIT compiler needs to compile kernels for the various shapes.

See the full set of options of `tx` in [the CLI reference](../reference.md).

You can visualize TPU usage with

```bash
uv run --with libtpu --with git+https://github.com/google/cloud-accelerator-diagnostics/#subdirectory=tpu_info tpu-info
```
