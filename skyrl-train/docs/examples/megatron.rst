Megatron Backend for 5D Parallelism
===================================

SkyRL supports NVIDIA's `Megatron-Core <https://developer.nvidia.com/megatron-core>`_ library as an RL training backend, inheriting support for 5D parallelism (tensor+sequence, pipeline, context, expert, and data parallelism), and optimized performance for large scale models.

We provide example scripts for running efficient large scale MoE training with models like ``Qwen3-30B-A3B`` using Megatron in the `examples/megatron <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/megatron>`_ directory.
For details on configuring the Megatron backend, and enabling checkpointing, see :ref:`megatron-configurations`, and :ref:`megatron-checkpointing`.

Comparison to FSDP
------------------
We show performance comparisons for the Megatron and FSDP2 backends on the Search-R1 task (4K max context length) for various model sizes in the table below. Training speed for small scale dense models with Megatron
is similar to the FSDP2 backend, and Megatron enables high throughput training for larger scale MoE models where FSDP is no longer feasible as the only parallelism strategy.

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 20 20 10

   * - Model
     - Backend
     - Compute
     - Policy Training Time (s)
     - Forward Pass Time (s)
     - Avg Num Tokens
   * - Qwen2.5-3B-Instruct
     - Megatron
     - 8xH100
     - 48
     - 39
     - 658
   * - Qwen2.5-3B-Instruct
     - FSDP2
     - 8xH100
     - 42
     - 28
     - 658
   * - Qwen2.5-7B-Instruct
     - Megatron
     - 8xH100
     - 93
     - 46
     - 819
   * - Qwen2.5-7B-Instruct
     - FSDP2
     - 8xH100
     - 100
     - 33
     - 834
   * - Qwen3-30B-A3B
     - Megatron
     - 4x8xH100
     - 189
     - 145
     - 1158

For all experiments, we used a train batch size of 512. For Qwen2.5 3B and 7B Megatron was configured with only data parallel=8. 
For Qwen3-30B-A3B Megatron was configured with DP=4, TP=2, and EP=8, and the FSDP2 backend was unable to complete a training step due to memory constraints. 
All statistics shown were averaged over the first 10 steps of training. Micro batch sizes were tuned to be the max possible for each backend.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: images/search-r1-3b.svg
         :width: 400px
         :align: center
     - .. image:: images/search-r1-30b.svg
         :width: 400px
         :align: center

.. centered:: Left: Matching Qwen2.5-3B-Instruct reward curves for Megatron and FSDP2. Right: Qwen3-30B-A3B reward curve for Megatron (330 steps on 4 8xH100 nodes over 4 days).

A script for running the Qwen3-30B-A3B experiment can be found `here <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/megatron/run_search_megatron.sh>`_. 
Additionally, we provide a script for running basic GSM8K training on Qwen3-235B-A22B with Megatron `here <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/megatron/run_megatron_qwen3-235b-a22b.sh>`_. 
Note that although training at 100B+ scale using Megatron is currently possible, we are in the process of further optimizing peformance.

For more details on configuring the Megatron backend, and enabling checkpointing, see :ref:`megatron-configurations`, and :ref:`megatron-checkpointing`.

.. _megatron-installation:

Installation
------------

Setting up the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To get started, you can follow the instructions for installing via Docker in the :doc:`../getting-started/installation` page, but using the ``erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8-megatron`` image instead of the default image.

This ensures that the necessary dependencies needed for Megatron (i.e. ``TransformerEngine``) are installed and don't need to be built on each node for each run, which can be time consuming.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~
After following the installation instructions, set the following environment variables for the ``TransformerEngine`` dependency to be correctly picked up by the uv + ray integration (since we currently exclude it from the pyproject.toml file to avoid building it on each node):

.. code-block:: bash

    export SKYRL_PYTHONPATH_EXPORT=1
    # where TransformerEngine is installed (via pip) on your machine
    export PYTHONPATH="/home/ray/anaconda3/lib/python3.12/site-packages"

Flash Attention
~~~~~~~~~~~~~~~
Next, in order to use flash attention with the megatron backend, you must use ``flash_attn`` version ``2.7.4.post1`` or lower for compatibility with ``TransformerEngine==2.5.0``.
You can replace the ``flash-attn`` wheel in the ``pyproject.toml`` file with the following to use the ``2.7.4.post1`` release, and you can find wheels for other versions `here <https://github.com/Dao-AILab/flash-attention/releases>`_.

.. code-block:: bash

    flash-attn = { url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" }


Configuration
-------------
We provide the following options for fully configuring the Megatron backend, exposing the underlying Megatron optimizer, DDP, and model config objects
for advanced users to fully take advantage of all of Megatron-Core's feature flags. For more details, see the :ref:`megatron-configurations` section.

.. code-block:: yaml
    :caption: ``skyrl_train/config/megatron/policy.yaml``

    # @package megatron_config.policy
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    expert_model_parallel_size: 1
    expert_tensor_parallel_size: null

    ddp_config: # pass-through config to Megatron's `DistributedDataParallelConfig` object
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/distributed/distributed_data_parallel_config.py#L8
      ...
    optimizer_config_kwargs: # pass-through kwargs to Megatron's `OptimizerConfig` object
      # any overlapping arguments with those we attempt to resolve in trainer.policy.optimizer_config will be overridden by the values here
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/optimizer/optimizer_config.py#L12
      ...
    transformer_config_kwargs: # pass-through kwargs to the Megatron's `TransformerConfig` object
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/transformer/transformer_config.py#L33
      ...


These default values can be overridden by passing in the corresponding arguments to ``trainer.policy.megatron_config`` in the launch script.

Parallelism Resources
----------------------
Understanding and configuring parallelism strategies for large models can be challenging.
Some helpful resources for understanding and tuning large scale parallelism strategies can be found at the `Huggingface Ultra-Scale Playbook <https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=finding_the_best_training_configuration>`_, 
the `The Mesh Parallelism Zoo <https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/>`_, and the `Visualizing 6-D Parallelism <https://main-horse.github.io/posts/visualizing-6d>`_.

Below, we show a diagram displaying how all 5 parallelism strategies - tensor, pipeline, context, expert, and data parallelism - can be utilized in SkyRL, as well as how dispatching data across these parallel groups works.

.. image:: images/parallelism.svg


