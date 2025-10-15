Launching with Skypilot
=====================================

Skypilot
-------

`SkyPilot <https://docs.skypilot.co/en/latest>`_ is a system to run, manage,
and scale AI workloads on any infrastructure. It handles all of the logic of
provisioning, setting up, and tearing down clusters on any cloud.

Setup Skypilot
--------------

SkyPilot works across AWS, GCP, Azure, OCI, Lambda, Nebius, and more. The
commands below use AWS as a concrete example. You can swap ``infra`` and accelerator
names (for example ``infra: nebius`` with ``accelerators: L40:2``) to target a
different cloud.

.. code-block:: bash

    git clone https://github.com/NovaSky-AI/SkyRL.git
    cd SkyRL

    # Install uv if you have not already.
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a dedicated env for the SkyPilot CLI and install the stable build.
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install "skypilot[aws]"  # replace aws with the clouds you plan to use

    # Validate your configuration.
    sky check aws

Run SkyRL with SkyPilot
-----------------------

To interact with SkyPilot we need to write a yaml defining the job. For the
full set of options allowed see `SkyPilot docs <https://docs.skypilot.co/en/latest/reference/yaml-spec.html>`_.

.. note::

   Docker image ``erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8`` depends on
   features that have not been released in the stable SkyPilot builds as of
   October 8, 2025. Until SkyPilot publishes v0.10.3 or later, rely on the
   default base image resolved by SkyPilot and install extra dependencies inside
   ``setup``. See `skypilot-org/skypilot#7181 <https://github.com/skypilot-org/skypilot/pull/7181>`_.

.. code-block:: yaml

  # Run this from the repository root after cloning SkyRL.
  resources:
    infra: aws           # replace this with what cloud you want to launch on
    accelerators: L40S:4 # 4x 48 GB GPUs; adjust to the SKU that matches your quota
    memory: 64+          # every node has at least 64 GB memory
    ports: 6479          # expose port for ray dashboard
  #  network_tier: best # when using multiple nodes, communication can become a bottleneck

  num_nodes: 1         # cluster size

  # Set to a git repository 
  workdir:
    url: https://github.com/NovaSky-AI/SkyRL.git
    ref: main

  # Set secrets
  secrets:
    WANDB_API_KEY: null

  envs:
    LOGGER: "wandb"  # change to "console" to print to stdout
    INFERENCE_BACKEND: "vllm"
    # INFERENCE_BACKEND: "sglang"

  # Commands run on each node of the remote cluster to set up the environment (e.g., install dependencies). These are run directly inside Docker.
  setup: |
    cd skyrl-train
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv sync --extra vllm
    uv pip install wandb
    uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

  # If you already have processed the data locally, delete the above line

  # The actual task commands to be executed on the remote cluster.
  # This script will first start the Ray cluster (different ray start commands are executed on Head and Worker nodes).
  # Then, your training script will only be run on the Head node (SKYPILOT_NODE_RANK == 0).
  run: |
    set -euo pipefail

    cd skyrl-train
    source .venv/bin/activate

    TMP_DIR="$HOME/skyrl-tmp"
    mkdir -p "$TMP_DIR"
    export TMPDIR="$TMP_DIR"

    read -r head_ip _ <<< "$SKYPILOT_NODE_IPS"
    DATA_DIR="$HOME/data/gsm8k"

    # Login to Weights & Biases once the secrets are available.
    uv run -- python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

    wait_for_ray() {
      local address=$1
      for _ in $(seq 1 24); do
        if ray status --address "$address" >/dev/null 2>&1; then
          return 0
        fi
        sleep 5
      done
      echo "Ray cluster at $address failed to become ready" >&2
      return 1
    }

    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
    if [ "$SKYPILOT_NODE_RANK" = "0" ]; then
      if ! ray status --address 127.0.0.1:6479 >/dev/null 2>&1; then
        ray start --head --disable-usage-stats --port 6479
      fi
      wait_for_ray 127.0.0.1:6479
      uv run --isolated --extra "$INFERENCE_BACKEND" -m skyrl_train.entrypoints.main_base \
        data.train_data="['${DATA_DIR}/train.parquet']" \
        data.val_data="['${DATA_DIR}/validation.parquet']" \
        trainer.algorithm.advantage_estimator="grpo" \
        trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
        trainer.placement.colocate_all=true \
        trainer.strategy=fsdp2 \
        trainer.placement.policy_num_gpus_per_node="$SKYPILOT_NUM_GPUS_PER_NODE" \
        trainer.placement.ref_num_gpus_per_node="$SKYPILOT_NUM_GPUS_PER_NODE" \
        trainer.placement.ref_num_nodes="$SKYPILOT_NUM_NODES" \
        trainer.placement.policy_num_nodes="$SKYPILOT_NUM_NODES" \
        generator.num_inference_engines="$SKYPILOT_NUM_GPUS_PER_NODE" \
        generator.inference_engine_tensor_parallel_size=1 \
        trainer.epochs=20 \
        trainer.eval_batch_size=1024 \
        trainer.eval_before_train=true \
        trainer.eval_interval=5 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=1024 \
        trainer.policy_mini_batch_size=256 \
        trainer.micro_forward_batch_size_per_gpu=64 \
        trainer.micro_train_batch_size_per_gpu=64 \
        trainer.ckpt_interval=10 \
        trainer.max_prompt_length=512 \
        generator.sampling_params.max_generate_length=1024 \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.algorithm.use_kl_loss=true \
        generator.backend="$INFERENCE_BACKEND" \
        generator.run_engines_locally=true \
        generator.weight_sync_backend=nccl \
        generator.async_engine=true \
        generator.batched=true \
        environment.env_class=gsm8k \
        generator.n_samples_per_prompt=5 \
        generator.gpu_memory_utilization=0.8 \
        trainer.logger="$LOGGER" \
        trainer.project_name="gsm8k" \
        trainer.run_name="gsm8k_test" \
        trainer.resume_mode=null \
        trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt"
    else
      if ! ray status --address "$head_ip:6479" >/dev/null 2>&1; then
        ray start --address "$head_ip:6479" --disable-usage-stats
      fi
      wait_for_ray "$head_ip:6479"
    fi

    echo "Node setup and Ray start script finished for rank ${SKYPILOT_NODE_RANK}."


You can launch this yaml with
``sky launch -c skyrl skyrl_train/examples/gsm8k/gsm8k-skypilot.yaml --secret WANDB_API_KEY="1234"``.
After it launches, you can easily access the cluster with ``ssh skyrl``. To
terminate the cluster simply run ``sky down skyrl``.

Launch Verification Views
-------------------------

Use the following reference views to confirm the environment and job status:

.. figure:: images/skypilot-dashboard.jpeg
   :alt: SkyPilot Dashboard showing the gsm8k cluster ready state
   :width: 80%

   SkyPilot Dashboard after ``sky launch`` reports the cluster as healthy.

.. figure:: images/skypilot-ray-logs.png
   :alt: Terminal logs from ``sky logs skyrl`` showing GRPO training progress
   :width: 80%

   ``sky logs`` streaming Ray task updates confirms Ray and SkyRL workers are active.

.. figure:: images/skypilot-wandb.jpeg
   :alt: Weights & Biases dashboard capturing the gsm8k_test run metrics
   :width: 80%

   Weights & Biases dashboard provides live metrics and checkpoints for the run.
