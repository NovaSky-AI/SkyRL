## Guide: Verifiers (Environments Hub) + SkyRL

This directory holds the workflow to train on Environments Hub environments with SkyRL.

To start training, follow three simple steps:
1) Install the environment from Environments Hub.
2) Prepare the environment's training and validation datasets.
3) Launch training!

Start by following the SkyRL [installation instructions](https://skyrl.readthedocs.io/en/latest/getting-started/installation.html), then enter the `skyrl-train` directory:
```bash
cd SkyRL/skyrl-train
```

### 1) Install the environment
First, specify your desired environment from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments):
```bash
export ENV_ID="will/wordle"
```

Then, install the environment, which adds it to the `uv` project:
```bash
uv run integrations/verifiers/install_environment.py $ENV_ID
```

### 2) Prepare the dataset
Next, load the environment's dataset and convert to SkyRL format:
```bash
uv run --isolated --with verifiers \
  python integrations/verifiers/prepare_dataset.py \
  --env_id $ENV_ID
```

`prepare_dataset.py` has additional optional parameters:
  - `--output_dir`: directory to place datasets (default: `~/data/{env_id}`)
  - `--num_train`: number of training samples to generate (-1 for no limit)
  - `--num_eval`: number of validation samples to generate (-1 for no limit)

Notes on dataset generation:
- This script will generate the following Parquet files under `output_dir`:
  - `train.parquet`
  - `validation.parquet`, if included in the environment
- For issues in loading the dataset, see the Troubleshooting section below.

### 3) Launch training
Open `run_verifiers.sh`, which specifies the training configuration parameters and is the primary interface for launching training runs.

Modify the commonly-edited training settings as needed:
```bash
ENV_ID="will/wordle"
DATA_DIR="$HOME/data/$ENV_ID"
NUM_GPUS=1
LOGGER="wandb"
```

Finally, launch your training run:

```bash
bash integrations/verifiers/run_verifiers.sh
```

All training parameters can be modified in `run_verifiers.sh`, such as the model choice (`trainer.policy.model.path`), GRPO group size (`generator.n_samples_per_prompt`), or training batch size (`trainer.train_batch_size`).

See all available training configuration parameters in `ppo_base_config.yaml`.


## Troubleshooting

For issues with SkyRL or the integration with Verifiers, please [open an Issue](https://github.com/NovaSky-AI/SkyRL/issues/new). 


### Datasets
Verifiers environments can handle dataset splits in different ways. Some environments require passing a `dataset_split` argument to `load_environment()` (e.g., to specify `train` vs `test`), others implement both `vf_env.load_dataset()` and `vf_env.load_eval_dataset()`. 

The implementation in `prepare_dataset.py` assumes the latter approach to get datasets, which may be incorrect for some environments. Please modify `prepare_dataset.py` as needed to extract and prepare the correct datasets.


## TODOs and Limitations
We welcome any contributions to help resolve the remaining tasks!
* Make it easier to specify different Verifiers environments used for training and validation.
* Make it smoother to specify which dataset splits to use (ie, resolve the challenge specified in the `Troubleshooting` section.)
* Consider plumbing Verifiers-specific configuration to the VerifiersGenerator for easy override. For example: `zero_truncated_completions` and `mask_truncated_completions`.
