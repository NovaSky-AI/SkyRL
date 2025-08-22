FlashRL + SkyRL: Training with Quantized Rollouts
=================================================

In this example, we walk through how to train a model with quantized rollouts using `FlashRL <https://fengyao.notion.site/flash-rl>`_ and SkyRL.

What is FlashRL?
----------------

`FlashRL <https://fengyao.notion.site/flash-rl>`_ is a novel method that provides the first open-source RL recipe with quantized (Int8, FP8) rollout generation while preserving downstream performance. FlashRL consists of two main components:

- Truncated Importance Sampling (TIS): In scalable RL frameworks, policy model and rollout are typically managed by different libraries/ frameworks (FSDP and vLLM, resp.), which leads to a mismatch between the probability distributions. TIS is a technique that solves the rollout and training mismatch problem by applying a token-level correction factor (based on the importance-sampling ratio) to the policy loss. 
- Online Quantization Support: While vLLM has support for inference with quantized weights, it is tricky to use this for RL training. FlashRL also has patches for vLLM to support weight updates for FP8 and Int8 during training. 


FlashRL + SkyRL
---------------

SkyRL now has a native integration with FlashRL. Currently, we support training with Int8 quantization as well as `online FP8 quantization <https://docs.vllm.ai/en/v0.9.2/features/quantization/fp8.html#online-dynamic-quantization>`_ in vLLM. 


.. warning::

   FlashRL integration only supports single-turn training at the moment.


How does it work?
~~~~~~~~~~~~~~~~~~

At a high level, we sample generations from the inference engine with quantized weights (Int8, FP8). We then compute advantages and model losses. We apply the TIS correction factor to the policy loss to account for the difference in rollout and training probability distributions. On weight update, we sync the weights (in half precision) with the inference engine layer by layer. These weights are then quantized to the appropriate format (Int8, FP8) before loading.

Examples
--------

We provide examples for training with FP8 and Int8 rollouts for DAPO. The FlashRL related files are in the :code_link:`examples/flash_rl/` folder. 

For FP8, you simply need to specify ``FLASHRL_CONFIG=fp8_vllm`` in your environment variables. 

For Int8, we need to provide calibration data. We leverage the provided calibration data from FlashRL for ``Qwen/Qwen2.5-0.5B-Instruct`` and ``Qwen/Qwen-32B`` models. You can simply specify the appropriate ``FLASHRL_CONFIG`` in your environment variables. See :ref:`flashrl-config` for more details on how this works.

- 0.5B: ``FLASHRL_CONFIG=LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-quantized.w8a8-RedHatAI/flashrl_config.yaml``
- 32B: ``FLASHRL_CONFIG=LiyuanLucasLiu/Qwen2.5-32B-quantized.w8a8/flashrl_config.yaml``


Training Qwen2.5-32B with Int8 rollouts on the DAPO recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, prepare and save the dataset in a chosen ``DATA_DIR`` by running:

.. code-block:: bash
    # execute from skyrl-train directory

    DATA_DIR="$HOME/data/dapo" bash examples/algorithms/dapo/prepare_dapo_data.sh

We highlight some important training parameters configured for FlashRL from our example configuration at :code_link:`examples/flash_rl/run_dapo_repro_flashrl_32b_int8.sh`:

.. code-block:: bash
    :caption: Training configuration at ``examples/flash_rl/run_dapo_repro_flashrl_32b_int8.sh``

    # path for dataset (.parquet files) containing the prompts and metadata for each question
    DATA_DIR="$HOME/data/dapo"

    # TIS parameters
    USE_TIS=true
    TIS_IMP_RATIO_CAP=8.0

    uv run --isolated --extra flashrl --env-file examples/flash_rl/.env.int8 -m examples.flash_rl.main_dapo_flashrl \
        ...
        trainer.algorithm.use_tis=$USE_TIS \
        trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
        generator.sampling_params.logprobs=0 \
        ...

Here, we've configured training to use TIS with the importance sampling ratio cap of 8.0. ``generator.sampling_params.logprobs=0`` ensures that logprobs for the chosen tokens are returned by the inference engine, which is required for TIS. Note that for making sure the FlashRL patches are applied for vLLM, we use the ``FLASHRL_CONFIG`` environment variable in ``examples/flash_rl/.env.int8``: 

.. code-block:: bash
    :caption: Environment variables at ``examples/flash_rl/.env.int8``

    FLASHRL_CONFIG=LiyuanLucasLiu/Qwen2.5-32B-quantized.w8a8/flashrl_config.yaml
    # FLASHRL_LOGGING_LEVEL=DEBUG <--- optional
    ...


For a more lightweight example, we also provide scripts for training on Qwen2.5-0.5B-Instruct with Int8 rollouts at :code_link:`examples/flash_rl/run_dapo_repro_flashrl_0.5b_int8.sh`.


Training with FP8
~~~~~~~~~~~~~~~~~~

The configuration is similar to the Int8 example. The only difference is the value for ``FLASHRL_CONFIG`` in ``examples/flash_rl/.env.0.5b_fp8``. We provide a script for training Qwen2.5-0.5B-Instruct with FP8 rollouts  at :code_link:`examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_fp8.sh`.


.. _flashrl-config:

What does the ``FLASHRL_CONFIG`` do?
------------------------------------

We use a custom vLLM wheel (in the ``--flashrl`` extra) to apply some patches from FlashRL. 
The ``FLASHRL_CONFIG`` is used to customize vLLM initialization as well as weight syncing behavior. 

For FP8, this is simply a string (``fp8_vllm``) while for Int8, this is a path to a YAML file (either locally, accessible to all nodes in your Ray cluster, or a file path on the HuggingFace Hub). 

For Qwen2.5-0.5B-Instruct, the ``FLASHRL_CONFIG`` is ``LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-quantized.w8a8-RedHatAI/flashrl_config.yaml`` which contains the following:

.. code-block:: yaml
    :caption: ``LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-quantized.w8a8-RedHatAI/flashrl_config.yaml``

    configs:
      - distributed_executor_backend: external_launcher # ignored in SkyRL - We use the ray backend for vLLM
        fn: int8 # dictates the quantization type
        load_format: auto
        model: LiyuanLucasLiu/Qwen2-0.5B-Instruct-quantized.w8a8-RedHatAI # custom model path passed to vLLM at init - weights are loaded directly in int8
        profile: LiyuanLucasLiu/Qwen2-0.5B-Instruct-quantized.w8a8-RedHatAI/profile.pt # calibration profile for Qwen's weights, used during weight syncing

While most parameters are self-explanatory, the ``profile`` parameter is used to specify the calibration profile for Qwen's weights. This is used during weight syncing, when the policy model sends weights in half precision (bfloat16) to the inference engine. This profile is used to quantize the weights in bfloat16 to int8 before loading.

.. warning::

   FlashRL integration is experimental. While generation times can improve for large models with quantization, we've observed that the time spent in weight syncing is much higher with FlashRL for FP8. This negates some of the benefits of FP8 inference. The slowdown is primarily due to slow weight quantization in vLLM's ``process_weights_after_loading`` function and we are working on improving this.

   We recommend to use Int8 quantization over FP8 if possible.
