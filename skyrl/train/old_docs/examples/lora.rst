LoRA Training in SkyRL
======================

This guide demonstrates how to use Low-Rank Adaptation (LoRA) for efficient reinforcement learning training in SkyRL. LoRA allows you to train large language models with significantly reduced memory footprint and computational requirements by only updating a small set of parameters.

What is LoRA?
-------------

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. 

Configuration
-------------

LoRA can be configured for both policy and critic models in the training configuration. Here's how to set it up:

.. code-block:: yaml

    trainer:
      policy:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
          lora:
            rank: 32              # LoRA rank (higher = more parameters)
            alpha: 32             # LoRA scaling parameter
            dropout: 0            # LoRA dropout rate
            lora_sync_path: "/tmp/skyrl_lora_sync"  # Path for adapter sync
            target_modules: "all-linear"  # Apply to all linear layers OR
            # specify specific modules as a list
            exclude_modules: null  # Modules to exclude from LoRA
      critic:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
          lora:
            rank: 32
            alpha: 32
            dropout: 0
            target_modules: "all-linear"
            exclude_modules: null

Key Parameters
~~~~~~~~~~~~~~

- **``rank``**: The rank of the low-rank decomposition. Higher values mean more trainable parameters but also more memory usage. Common values are 8, 16, 32, or 64.
- **``alpha``**: The scaling parameter for LoRA. Often set equal to rank, but can be tuned independently.
- **``dropout``**: Dropout rate applied to LoRA layers. Helps with regularization.
- **``lora_sync_path``**: Directory path where LoRA adapters are saved and synchronized between training and inference engines.
- **``target_modules``**: Specifies which modules to apply LoRA to. By default, LoRA is applied to all linear layers in the model - you can customize which modules to target by providing a list of specific module names.
- **``exclude_modules``**: Specifies which modules to exclude from LoRA application. By default, no modules are excluded.

Running LoRA Training
---------------------

Here's a complete example of running LoRA training on GSM8K:

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

First, prepare your dataset:

.. code-block:: bash

   uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

Training Script
~~~~~~~~~~~~~~~

Create a training script with LoRA configuration:

.. code-block:: bash

   #!/bin/bash
   set -x

   DATA_DIR="$HOME/data/gsm8k"
   NUM_GPUS=4
   LOGGER="wandb"  # change to "console" to print to stdout
   INFERENCE_BACKEND="vllm"

   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
     data.train_data="['$DATA_DIR/train.parquet']" \
     data.val_data="['$DATA_DIR/validation.parquet']" \
     trainer.algorithm.advantage_estimator="grpo" \
     trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
     trainer.policy.model.lora.rank=32 \
     trainer.policy.model.lora.alpha=32 \
     trainer.policy.model.lora.lora_sync_path="/tmp/skyrl_lora_sync" \
     trainer.strategy=fsdp2 \
     trainer.placement.colocate_all=true \
     trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
     trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
     generator.num_inference_engines=$NUM_GPUS \
     generator.inference_engine_tensor_parallel_size=1 \
     trainer.train_batch_size=128 \
     trainer.policy_mini_batch_size=128 \
     trainer.micro_forward_batch_size_per_gpu=64 \
     trainer.micro_train_batch_size_per_gpu=64 \
     trainer.ckpt_interval=10 \
     generator.sampling_params.max_generate_length=1024 \
     trainer.policy.optimizer_config.lr=3.0e-5 \
     trainer.algorithm.use_kl_loss=true \
     generator.backend=$INFERENCE_BACKEND \
     generator.batched=true \
     environment.env_class=gsm8k \
     generator.n_samples_per_prompt=4 \
     trainer.logger="$LOGGER" \
     trainer.project_name="gsm8k_0.5b_lora" \
     trainer.run_name="gsm8k_0.5b_lora_test" \
     trainer.ckpt_path="$HOME/ckpts/gsm8k_0.5b_lora_ckpt"

Launch Training 
~~~~~~~~~~~~~~~

Set up your WandB API key and run the training:

.. code-block:: bash

   export WANDB_API_KEY=your_wandb_api_key
   bash examples/lora/run_qwen2_5_0.5b_gsm8k_grpo_lora.sh

Configuration Tips
------------------

We recommend looking at the blog `LoRA Without Regret <https://thinkingmachines.ai/blog/lora/>`_ for more details and practical guidance on setting LoRA hyperparameters. Here are the key takeaways:

1. **Learning rate:** Use roughly 10× higher learning rate for LoRA than for full fine-tuning; LoRA is more tolerant of large learning rates, especially for shorter runs.

2. **Rank:** Choose a rank large enough to match dataset complexity (e.g., 32–64 for most RL fine-tuning tasks). LoRA performs best when not capacity-limited.

3. **Layer coverage:** Apply LoRA to *all* layers, particularly MLP/MoE layers — attention-only LoRA tends to underperform.


Current Limitations
-------------------

SkyRL's LoRA implementation has the following current limitations:

1. **Disk-based synchronization**: LoRA adapters are saved to disk and reloaded rather than synchronized in-memory. 

4. **Single adapter per model**: Currently, only one LoRA adapter can be active per model at a time.

These limitations are being addressed in future releases, with plans for in-memory synchronization and improved adapter management.
