Switching Training Backends
=========================================
With SkyRL-Train, you can switch between different training backends with minimal changes to your training script.

Currently, we support the following training backends:

- DeepSpeed
- FSDP
- FSDP2

To switch to a different backend, simply set the ``trainer.strategy`` parameter to the desired backend. We use the ``fsdp2`` backend by default.

Prerequisites
-------------
First, make sure you are familiar with the standard setup process for running GRPO training. See :doc:`../getting-started/quickstart` for more details.

Running the Examples
---------------------

We provide baseline examples for GRPO training on GSM8K for each of these backends starting from the basic quickstart example.

.. code-block:: bash

    uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
        trainer.algorithm.advantage_estimator="grpo" \
        data.train_data="['$HOME/data/gsm8k/train.parquet']" \
        data.val_data="['$HOME/data/gsm8k/validation.parquet']" \
        trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
        ... # Other parameters (see `examples/gsm8k/run_gsm8k.sh` for more)

DeepSpeed
~~~~~~~~~
The only configuration change required for DeepSpeed is to set the ``trainer.strategy`` parameter to ``deepspeed``.

.. code-block:: bash

    # + trainer.strategy=deepspeed
    bash examples/training_backends/deepspeed/run_deepspeed.sh

Additionally, you can tune `deepspeed specific configurations <https://www.deepspeed.ai/docs/config-json/>`_ as shown below:

.. code-block:: bash

    # enable offloading of model parameters to CPU during the forward pass
    deepspeed_config.train.zero_optimization.offload_param.device=cpu \

    # enable offloading of optimizer state to CPU during the forward pass
    deepspeed_config.train.zero_optimization.offload_optimizer.device=cpu \

    # sub group size for ZeRO3-Offload (default is 1e9)
    deepspeed_config.train.zero_optimization.sub_group_size=1e8 \

    # Number of elements reduced/allreduced at a time (default is 5e8)
    deepspeed_config.train.zero_optimization.reduce_bucket_size=1e8 \

You can also refer to additional details at :ref:`deepspeed-configurations`, and 
you can find the full set of default values we provide for these configurations at :code_link:`skyrl_train/config/deepspeed_config/train.yaml` 
and :code_link:`skyrl_train/config/deepspeed_config/eval.yaml`.

FSDP and FSDP2
~~~~~~~~~~~~~~

To switch to FSDP or FSDP2, set the ``trainer.strategy`` parameter to ``fsdp`` or ``fsdp2`` respectively.

.. code-block:: bash

    # + trainer.strategy=fsdp
    bash examples/training_backends/fsdp/run_fsdp.sh

.. code-block:: bash

    # + trainer.strategy=fsdp2
    bash examples/training_backends/fsdp/run_fsdp2.sh

Additionally, you can tune `FSDP specific configurations <https://pytorch.org/docs/stable/fsdp.html>`_ as shown below:

.. code-block:: bash

    # enable offloading of model parameters to CPU during the forward pass for the ref model
    trainer.ref.fsdp_config.cpu_offload=true \

You can refer to the full set of FSDP configurations at :ref:`fsdp-configurations`.

.. note:: 
    ``cpu_offload`` cannot be enabled for the policy or critic model with FSDP1, since gradient accumulation outside ``no_sync`` mode is not supported with CPU offloading. 
    See the limitations section in `FSDP docs <https://docs.pytorch.org/docs/stable/fsdp.html>`_ for more details.
