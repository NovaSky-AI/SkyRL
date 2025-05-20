# SkyRL-v0 Reproduction Scripts

We provide exact scripts to reproduce our results for SkyRL-Agent-7B-v0, SkyRL-Agent-8B-v0, SkyRL-Agent-14B-v0. 

## Pre-requisite

### Installation

Make sure to have followed the installation commands in [INSTALL.md](../../../INSTALL.md). 

### Start Ray
Start ray in your cluster following the guide: https://docs.ray.io/en/latest/ray-core/starting-ray.html 

### Data preparation

We provide the datasets we used on HuggingFace: https://huggingface.co/novasky-ai 

We used [NovaSky-AI/SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data) for training both SkyRL-Agent-8B-v0 and SkyRL-Agent-14B-v0.
We used [NovaSky-AI/SkyRL-v0-80-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-80-data) (first stage) and [NovaSky-AI/SkyRL-v0-220-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-220-data) (second stage) to train SkyRL-Agent-7B-v0.
Make sure to download the dataset and update the path in `DATA_PATH` in the script, e.g. via

```bash
uv run huggingface-cli download NovaSky-AI/SkyRL-v0-293-data --repo-type dataset --local-dir <path_to_swegym_dataset>
```

### Setup Environment variables

We use a [`.env`](../../../.env.swebench) file to pass environment variables to all the processes created by Ray. Make sure to set `WANDB_API_KEY`, `ALLHANDS_API_KEY` and `SANDBOX_REMOTE_RUNTIME_API_URL`.

## Running the scripts

For  SkyRL-Agent-14B-v0 Thinking mode, use [run_skyrl_agent_qwen14b_t.sh](./run_skyrl_agent_qwen14b_t.sh) and for SkyRL-Agent-8B-v0 Non-Thinking Mode, use [run_skyrl_agent_qwen8b_nt.sh](./run_skyrl_agent_qwen8b_nt.sh). For SkyRL-Agent-7B-v0, use [run_skyrl_agent_oh7b_s1.sh](./run_skyrl_agent_oh7b_s1.sh) and [run_skyrl_agent_oh7b_s2.sh](./run_skyrl_agent_oh7b_s2.sh).
The scripts are meant to be run from the git root. The default settings are meant for a 8xH200 node. For a 8xH100 node, we recommend the following modifications:

```bash
TP_SIZE=2
SP_SIZE=2
NNODES=2
```
