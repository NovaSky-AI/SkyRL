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
      critic:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
          lora:
            rank: 32
            alpha: 32
            dropout: 0

Key Parameters
~~~~~~~~~~~~~~

- **``rank``**: The rank of the low-rank decomposition. Higher values mean more trainable parameters but also more memory usage. Common values are 8, 16, 32, or 64.
- **``alpha``**: The scaling parameter for LoRA. Often set equal to rank, but can be tuned independently.
- **``dropout``**: Dropout rate applied to LoRA layers. Helps with regularization.
- **``lora_sync_path``**: Directory path where LoRA adapters are saved and synchronized between training and inference engines.

Target Modules
~~~~~~~~~~~~~~

By default, LoRA is applied to all linear layers in the model. You can customize which modules to target:

.. code-block:: yaml

    trainer:
      target_modules: "all-linear"  # Apply to all linear layers
      # OR specify specific modules:
      exclude_modules: null  # Modules to exclude from LoRA

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
     trainer.epochs=20 \
     trainer.eval_batch_size=1024 \
     trainer.eval_before_train=false \
     trainer.eval_interval=5 \
     trainer.update_epochs_per_batch=1 \
     trainer.train_batch_size=1024 \
     trainer.policy_mini_batch_size=256 \
     trainer.micro_forward_batch_size_per_gpu=64 \
     trainer.micro_train_batch_size_per_gpu=64 \
     trainer.ckpt_interval=10 \
     trainer.max_prompt_length=512 \
     generator.sampling_params.max_generate_length=1024 \
     trainer.policy.optimizer_config.lr=3.0e-5 \
     trainer.algorithm.use_kl_loss=true \
     generator.backend=$INFERENCE_BACKEND \
     generator.run_engines_locally=true \
     generator.weight_sync_backend=nccl \
     generator.async_engine=true \
     generator.batched=true \
     environment.env_class=gsm8k \
     generator.n_samples_per_prompt=5 \
     generator.gpu_memory_utilization=0.8 \
     trainer.logger="$LOGGER" \
     trainer.project_name="gsm8k_0.5b_lora" \
     trainer.run_name="gsm8k_0.5b_lora_test" \
     trainer.resume_mode=null \
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

What's Next?
------------

Now that you understand LoRA training in SkyRL, you might want to explore:

- :doc:`../configuration/config`: Complete configuration reference
- :doc:`ppo`: Standard PPO training without LoRA
- :doc:`../tutorials/async`: Asynchronous training patterns
