Development Guide
=================

Pre-requisities
---------------

Follow the :doc:`installation guide <installation>`. Make sure that the installation works with our :doc:`quick start example <quickstart>`.


Modifying the code
------------------

- Are you adding a new environment or task? Follow the :doc:`new task tutorial <../tutorials/new_env>`. You would only need to add your custom code to a folder in `examples/` similar to `examples/multiply <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply>`_

- Are you modifying the training code (ex: adding a new algorithm, changing the training loop etc)? You would modify the code in :code_link:`skyrl_train`. 

- Are you modifying the existing environment code (ex: adding a custom method for all ``Env`` classes, improving the ``SearchEnv`` implementation)? You would modify the code in  `skyrl-gym <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym/>`_. Note that for adding a new environment or task, you do not have to modify the ``skyrl-gym`` package. 


Running tests
-------------

For running tests you should use the `dev` `extra <https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras>`_ for additional dependencies.

CPU tests
~~~~~~~~~

.. code-block:: bash

    uv run --isolated --extra dev --extra vllm pytest tests/cpu

GPU tests
~~~~~~~~~

The GPU tests require a node with atleast 8 GPUs. They have been tested on a 8xH100 node, but should work even on 8xA100 nodes. 

.. code-block:: bash

    uv run --isolated --extra dev --extra vllm pytest tests/gpu

