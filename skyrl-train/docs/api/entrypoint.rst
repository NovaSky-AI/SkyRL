Entrypoint API
==============

The main entrypoint is the `BasePPOExp` class which runs the main training loop.

.. autoclass:: skyrl_train.entrypoints.main_base.BasePPOExp
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

The second entrypoint (evaluation-only) is the `EvalOnlyEntrypoint` class which runs evaluation without training.

.. autoclass:: skyrl_train.entrypoints.main_generate.EvalOnlyEntrypoint
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance: