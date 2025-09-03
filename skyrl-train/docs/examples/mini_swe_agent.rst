SkyRL + Mini-SWE-Agent: Training a SWE-Agent for SWE-Bench

========================================================

In this example, we walk through how to train a SWE-Agent on the SWE-Bench task by leveraging [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent)

Dataset preparation
-------------------

For training, we use `SWE-Gym <https://huggingface.co/SWE-Gym>`_, and more specifically the subset of SWE-Gym in https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data

Execute the following command: 

.. code-block:: bash
    # execute from skyrl-train directory

    uv run --isolated examples/mini_swe_agent/preprocess.py --output_dir ~/data/swe_gym


Training
---------

Prerequisites: Ensure that you have the required environment backend installed for generating trajectories with Mini-SWE-Agent. By default, we use `apptainer <https://apptainer.org/docs/admin/main/index.html#>`_. This can be modified in :code_link:`examples/mini_swe_agent/swebench.yaml` 

.. code-block:: bash
    # execute from skyrl-train directory

    bash examples/mini_swe_agent/run_mini_swe.sh


TODO (sumanthrh, pcmorritz): Fill this in
