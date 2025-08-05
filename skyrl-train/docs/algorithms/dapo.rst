DAPO
====

The `DAPO <https://arxiv.org/abs/2503.14476>`_ (Decoupled Clip and Dynamic Sampling Policy Optimization) algorithm consists of the following components on top of a GRPO baseline:

- **Clip-Higher**: Promotes the diversity of the system and avoids entropy collapse;
- **Dynamic Sampling**: Improves training efficiency and stability;
- **Token-Level Policy Gradient Loss**: Critical in long-CoT RL scenarios;
- **Overlong Filtering**: Reduces reward noise and stabilizes training.

In this guide, we walk through how to enable each of these components in SkyRL. We provide a simple example script for training DAPO on GSM8K in :code_link:`examples/algorithm/dapo/run_dapo_gsm8k.sh`.

Clip-Higher
~~~~~~~~~~~
To use clip-higher, you can simply configure ``trainer.algorithm.eps_clip_high`` separately from ``trainer.algorithm.eps_clip_low``.

.. code-block:: yaml

    trainer:
      algorithm:
        eps_clip_low: 0.2
        eps_clip_high: 0.27

Dynamic Sampling
~~~~~~~~~~~~~~~~
In DAPO style dynamic sampling, we sample rollouts until we have a full batch with non-zero advantages (meaning that we have a non-zero std deviation of rewards for the n rollouts for a given prompt). 

To configure DAPO style dynamic sampling, you can set ``trainer.algorithm.dynamic_sampling.type`` to ``filter`` and configure ``trainer.algorithm.dynamic_sampling.max_sample_batches`` to the maximum number of batches to sample.
If ``max_sample_batches`` is exceeded, SkyRL-Train will attempt to exit the training loop, and if ``max_sample_batches`` is -1, SkyRL-Train will sample until a full batch with non-zero advantages is accumulated.

.. code-block:: yaml

    trainer:
      algorithm:
        dynamic_sampling:
          type: filter
          max_sample_batches: 30

Token-Level Policy Gradient Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DAPO uses token-level policy gradient loss, which can be enabled by setting ``trainer.algorithm.loss_reduction`` to ``token_mean``. This is the default setting in SkyRL-Train.

.. code-block:: yaml
    
    trainer:
      algorithm:
        loss_reduction: "token_mean" 

Overlong Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To enable overlong filtering, which sets loss mask to be all zeros for responses that do not output a stop token (i.e. responses that are too long), you can set ``generator.apply_overlong_filtering`` to ``true``.

.. code-block:: yaml

    generator:
      apply_overlong_filtering: true
