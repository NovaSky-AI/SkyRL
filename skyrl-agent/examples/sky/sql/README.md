# SkyRL-SQL Reproduction Scripts

We provide the exact scripts to reproduce our results for SkyRL-SQL-7B. The implementation is based on [SearchR1](https://github.com/PeterGriffinJin/Search-R1) implementation of agent loop.  

## Pre-requisite

### Installation

Make sure to have followed the installation commands in [INSTALL.md](../../../INSTALL.md). 

### Start Ray
Start ray in your cluster following the guide: https://docs.ray.io/en/latest/ray-core/starting-ray.html. 

### Data preparation

We provide the dataset we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data 
Make sure to download the dataset and update the path in `DATA_PATH` in the script. 

```bash 
huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data --local-dir <path_to_dir> --repo-type dataset
```

### Setup Environment variables

We use a [`.env.sql`](../../../.env.sql) file to pass environment variables to all the processes created by Ray. Make sure to set `WANDB_API_KEY` (other env vars like `ALLHANDS_API_KEY` are only used for the Swe-Bench task).

### DB environment 

Make sure to setup the database files needed for training. We use the database from [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL/edit/main/train_and_evaluate/README.md). 

You can download the datasets from:
- [ModelScope-OmniSQL-datasets](https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary)
- [HuggingFace-OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets)

The datasets include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

Unzip `data.zip` in this folder, and set the corresponding `DB_PATH` in the training script below. You can e.g. download and unzip the data by running

```bash
huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir <path_to_file.zip>
unzip <path_to_file.zip>
```


## Running the scripts 

We provide a script [run_skyrl_sql.sh](./run_skyrl_sql.sh) for reproducing our results for SkyRL-SQL-7B.

## Evaluation 
Reference our fork of the [OmniSQL repo](https://github.com/lynnliu030/OmniSQL/tree/main/evaluate_skysql) for evaluating the model. 