FlashRL + SkyRL: Training with FP8 Rollouts
===========================================

In this example, we walk through how to train a model with FP8 rollouts using `FlashRL <https://fengyao.notion.site/flash-rl>`_ and SkyRL.

We provide an example for training Qwen2.5-1.5B-Instruct and Qwen3-32B with FP8 rollouts. 

What is FlashRL?
----------------

FlashRL is a novel method that provides the first RL recipe with quantized rollout generation while preserving downstream performance. FlashRL consists of two main components:

- Truncated Importance Sampling (TIS): In scalable RL frameworks, policy model and rollout are typically managed by different libraries/ frameworks (FSDP and vLLM, resp.), which leads to a mismatch between the probability distributions. TIS is a technique that solves the rollout and training mismatch problem by applying a token-level correction factor (based on the importance-sampling ratio) to the policy loss. 
- Online Quantization Support: While vLLM has support for inference with quantized weights, it is tricky to use this for RL training. FlashRL also has patches for vLLM to support weight updates for FP8 and Int8 during training. 


FlashRL + SkyRL
---------------

SkyRL now supports an initial integration with FlashRL. Currently, we only support training with `online FP8 quantization <https://docs.vllm.ai/en/v0.9.2/features/quantization/fp8.html#online-dynamic-quantization>`_ in vLLM. You should simply specify ``FLASHRL_CONFIG=fp8_vllm`` in your environment variables and use the ``--extra flashrl`` flag when running the training script.


.. warning::

   FlashRL integration only supports single-turn training at the moment.


How does it work?
~~~~~~~~~~~~~~~~~~

We pass `quantization=fp8`  flag to the vLLM engine at initialization time. This means that the weights are loaded as usual in half precision and then quantized down to fp8. During training, generations are sampled as usual, and in this case, sampled from quantized weights. Since we use online quantization, the scale factor used for quantizing activations are computed on the fly by vLLM internally. 

The sampled rollouts are then used to compute the policy loss. We further apply the TIS correction factor to the policy loss and then update the policy model weights. These weights, in half precision, are then synced with the inference engine layer by layer. These are then loaded and quantized down to fp8 similar to how we quantized the weights at initialization. 


Example
--------

We provide two examples for training with FP8 rollouts for DAPO: one for training Qwen2.5-1.5B-Instruct and one for Qwen3-32B. The FlashRL related files are in ``skyrl_train/examples/flash_rl/`` folder. 


.. code-block:: bash
    :caption: Training configuration at ``skyrl_train/examples/flash_rl/run_dapo_flashrl.sh``

    # path for dataset (.parquet files) containing the prompts and metadata for each question
    DATA_DIR="$HOME/data/gsm8k"

    uv run --isolated --extra flashrl --env-file examples/flash_rl/.env.flashrl -m examples.flash_rl.main_dapo_flashrl \
        ...
        trainer.algorithm.use_tis=true \
        trainer.algorithm.tis_imp_ratio_cap=2.0 \
        ...

Here, we've configured training to use TIS with the importance sampling ratio cap of 2.0. Note that for making sure the FlashRL patches are applied for vLLM, we use the ``FLASHRL_CONFIG`` env var in ``examples/flash_rl/.env.flashrl``:

.. code-block:: bash
    :caption: Environment variables at ``examples/flash_rl/.env.flashrl``

    FLASHRL_CONFIG=fp8_vllm
    ...

.. warning::

   FlashRL integration is experimental. While generation times can improve for large models with quantization, we've observed that the time spent in weight syncing is much higher with FlashRL for fp8. This negates some of the benefits of fp8 inference. The slowdown is primarily due to slow weight quantization in vLLM's ``process_weights_after_loading`` function and we are actively working on improving this.