LLM as a Judge for GSM8K
=========================================

This example demonstrates how to train a model using LLM as a judge for reward computation on the GSM8K dataset. Instead of using rule-based reward functions, this approach leverages an LLM (like GPT-4o-mini) to evaluate the quality of generated solutions.

The implementation provides a custom environment that uses an LLM judge to compare predicted solutions against ground truth answers, offering more nuanced evaluation than simple exact matching.

Task Overview
-------------

In this task, the agent is given a math problem and must generate a solution that ends with the final answer in the format `#### <number>`. The reward is computed by an LLM judge that evaluates:

1. Whether the predicted solution ends with the correct format (`#### <number>`)
2. Whether the final answer matches the ground truth exactly

The LLM judge provides a binary reward (0 or 1) based on these criteria.

Dataset Preparation
-------------------

To download and prepare the dataset, run the following script:

.. code-block:: bash

   uv run examples/llm_as_a_judge/gsm8k_dataset_judge.py --output_dir $HOME/data/gsm8k_llm_judge

This script downloads the GSM8K dataset, extracts ground truth answers, and formats it for the LLM judge environment.

Environment Implementation
---------------------------

The LLM judge environment is implemented in ``examples/llm_as_a_judge/llm_judge_env.py``. We use the OpenAI API to access the LLM judge.

The environment sends the following prompt to the judge:

.. code-block:: text

   You are a strict math evaluation assistant.

   Compare the following **gold** and **predicted** math solutions. Your job is to determine if the predicted solution is mathematically correct and if the predicted solution ends with a line of the form:

   #### <number>

   You must only give a score of "1" if:
   - The final line of the predicted solution **ends with `#### <number>`**, and
   - The number **matches the final answer in the gold solution** exactly.

   Instructions:
   - You may provide internal reasoning or explanation before giving your final judgment.
   - Your final judgment must appear as a separate line at the end of your response, in the format:

   ### Final Score: 1

   or

   ### Final Score: 0

   Do not include any explanation after the final score.

Installation and Setup
----------------------

1. **Set Environment Variables**: Add your API keys to `.env.llm_judge` file.

   .. code-block:: bash

      OPENAI_API_KEY=your_openai_api_key
      WANDB_API_KEY=your_wandb_api_key # optional

2. **Verify Dataset**: Make sure your dataset is properly prepared:

   .. code-block:: bash

      ls $HOME/data/gsm8k_llm_judge/
      # Should show: train.parquet  validation.parquet

Training Configuration
----------------------

The training configuration uses GRPO with colocated training and generation. Key parameters include:

.. code-block:: bash
   :caption: Training configuration at ``examples/llm_as_a_judge/run_llm_judge.sh``

   # Data and model paths
   DATA_DIR="$HOME/data/gsm8k_llm_judge"
   CKPT_PATH="$HOME/ckpts/llm_judge"

   # Hardware configuration
   NUM_GPUS=4
   NUM_INFERENCE_ENGINES=4
   TP_SIZE=1

   uv run --isolated --extra vllm --env-file .env.llm_judge -m examples.llm_as_a_judge.main_llm_judge \
     # Data configuration
     data.train_data="['$DATA_DIR/train.parquet']" \
     data.val_data="['$DATA_DIR/validation.parquet']" \
     
     # Algorithm and training
     trainer.algorithm.advantage_estimator="grpo" \
     trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
     trainer.epochs=20 \
     trainer.train_batch_size=32 \
     trainer.policy_mini_batch_size=32 \
     
     # Placement and strategy
     trainer.placement.colocate_all=true \
     trainer.strategy=fsdp2 \
     trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
     
     # Generator configuration
     generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
     generator.inference_engine_tensor_parallel_size=$TP_SIZE \
     generator.backend=vllm \
     generator.n_samples_per_prompt=5 \
     
     # Environment and LLM judge configuration
     environment.env_class=llm_as_a_judge \
     environment.skyrl_gym.llm_as_a_judge.model="gpt-4o-mini" \
     
     # Other parameters (see the `examples/llm_as_a_judge/run_llm_judge.sh` for the full script)
     ...


Launching Your Training Run
---------------------------

Now you can launch your training run with the following command:

.. code-block:: bash

    bash examples/llm_as_a_judge/run_llm_judge.sh

The training will use the LLM judge to evaluate each generated solution.

What's Next?
------------

Now that you've seen how to use LLM as a judge for reward computation, you might want to explore:

- :doc:`ppo`: Compare with rule-based PPO training on GSM8K
- :doc:`multi_turn_text2sql`: Explore multi-turn training with async rollouts
- :doc:`search`: Learn about multi-turn search agent training
- :doc:`../tutorials/new_env`: Learn how to build your own custom environments