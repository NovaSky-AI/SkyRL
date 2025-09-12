Troubleshooting
===============


Placement Group Timeouts
-------------------------

In SkyRL, we use Ray placement groups to request resources for different actors. In Ray clusters that autoscale with KubeRay, placement group creation can take a long time since the cluster might have to add a new node, pull the relevant image and start the container, etc. 
You can use the ``SKYRL_RAY_PG_TIMEOUT_IN_S`` environment variable (Used in the ``.env`` file passed to the ``uv run`` command with ``--env-file``) to increase the timeout for placement group creation (By default, this is 60 seconds)

Multi-node Training
-------------------

For multi-node training, it is helpful to first confirm that your cluster is properly configured. We provide a script at ``scripts/multi_node_nccl_test.py`` to test multi-node communication.

To run the script, you can use the following command:

.. code-block:: bash

   uv run --isolated --env-file .env scripts/multi_node_nccl_test.py --num-nodes 2

.env is optional, but it is recommended to use for configuring environment variables.

Note on ``LD_LIBRARY_PATH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS). We've seen issues with `uv` where the ``LD_LIBRARY_PATH`` is not exported even if it is set in the ``.env`` file. It is recommended to set the ``SKYRL_LD_LIBRARY_PATH_EXPORT=1`` in the ``.env`` file and set ``LD_LIBRARY_PATH`` directly in the current shell.



