# SkyRL-Train: A modular, performant RL framework for post-training LLMs

SkyRL-Train is a highly modular, performant RL framework for post-training LLMs. With a focus on modularity, SkyRL makes it easy to prototype new training algorithms, environments, and execution plans—without compromising usability or speed. 

SkyRL-train is **for users who want to modify anything:**

- **Quickly develop new environments** without modifying or understanding the training code .
- **Modify the training execution plan** such as model placement, colocation or disaggregation of training and generation, and async RL.
- **Implement custom trajectory generation** specific to your use-case, such as custom sampling methods, tree search, etc.
- … make any other flexible modifications to the RL workflow!

## Quick Start

A quick start guide is provided below.

### Requirements

The only requirements are:

- CUDA version >=12.4
- [uv](https://docs.astral.sh/uv/)

If you're running on an existing Ray cluster, make sure to use Ray 2.44.0 and Python 3.12. If not, proceed with the installation instructions below.


First, clone the repository:

```bash
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-train
```

Then, create a new virtual environment and install the dependencies:

```bash
# creates a venv at .venv/
uv sync --extra vllm 
source .venv/bin/activate
```

Then, prepare the dataset:

```bash
uv run -- python examples/gsm8k/gsm8k_dataset.py
```

Finally, before training, make sure to configure Ray to use `uv`:

```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# or add to your .bashrc
# echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc
```

You should now be able to run our example script (assumes at least 4 GPUs):

```bash
export WANDB_API_KEY=<your wandb api key>
bash examples/gsm8k/run_gsm8k.sh
```

For detailed installation instructions, as well as more examples, please refer to our [documentation](https://skyrl.readthedocs.io/en/latest/)

# Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!


