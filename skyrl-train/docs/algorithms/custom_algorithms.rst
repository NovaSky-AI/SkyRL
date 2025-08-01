Implementing Custom Algorithms
=============================

SkyRL-Train provides a registry system for easily implementing custom algorithms (advantage estimators, policy loss) without modifying the core codebase. 
Full documentation for the registry system can be found in the :doc:`registry API <../api/registry>`.
Example scripts of using the registry can be found in at :code_link:`examples/algorithm/`.

Registering a Custom Advantage Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can register custom advantage estimators using either a decorator or the registry directly:

.. code-block:: python

   from skyrl_train.utils.ppo_utils import register_advantage_estimator, AdvantageEstimatorRegistry
   import torch

   # Using the decorator
   @register_advantage_estimator("simple_baseline")
   def compute_simple_baseline_advantage(
        token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, **kwargs
    ):
        with torch.no_grad():
            response_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)

            # Simple baseline: use the mean reward across the batch
            baseline = response_rewards.mean()
            advantages = (response_rewards - baseline) * response_mask
            returns = advantages.clone()

            return advantages, returns

   # Or register directly
   def another_estimator(**kwargs):
       # Implementation here
       pass

   AdvantageEstimatorRegistry.register("direct_registration", another_estimator)

Registering a Custom Policy Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, you can register custom policy loss functions:

.. code-block:: python

   from skyrl_train.utils.ppo_utils import register_policy_loss, PolicyLossRegistry

   @register_policy_loss("reinforce")
   def compute_reinforce_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
       # Your custom policy loss implementation (like REINFORCE)
       loss = (-log_probs * advantages).mean()
       # return loss and clip ratio
       return loss, 0.0

Ray Distribution
~~~~~~~~~~~~~~~~

The registry system handles Ray actor synchronization when Ray is initialized. Functions registered on one process will be available to all Ray actors:

.. code-block:: python

   import ray
   from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry, sync_registries

   # Register a function on the main process
   AdvantageEstimatorRegistry.register("shared_function", my_function)

   # After Ray is initialized, we sync the registries to a named ray actor (in utils/utils.py::initialize_ray)
   ray.init()
   sync_registries()
   
   @ray.remote(num_cpus=1)
    def skyrl_entrypoint(cfg: DictConfig):
        # Function is now available on all Ray actors
        available_functions = AdvantageEstimatorRegistry.list_available()

        exp = BasePPOExp(cfg)
        exp.run()



