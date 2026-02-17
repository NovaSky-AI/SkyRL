SkyRL + Mini-SWE-Agent: Training a SWE-Agent for SWE-Bench
===========================================================

In this example, we walk through a simple example on how to train a SWE-Agent on the SWE-Bench task by leveraging `Mini-SWE-Agent <https://github.com/SWE-agent/mini-swe-agent>`_


How does it work?
------------------

The Mini-SWE-Agent integration with SkyRL can be found in :code_link:`examples/mini_swe_agent/`. We implement a custom generator ``MiniSweAgentGenerator`` to use Mini-SWE-Agent to generate trajectories for the SWE-Bench task. 


.. code-block:: python

    class MiniSweAgentGenerator(SkyRLGymGenerator):
        
        async def generate_trajectory(self, prompt, ...): 
            ...

        async def generate(self, generator_input: GeneratorInput) -> GeneratorOutput:
            ...
            prompts = generator_input["prompts"]
            env_extras = generator_input["env_extras"]
            tasks = []
            for i in range(len(prompts)):
                tasks.append(
                    self.generate_trajectory(
                        prompts[i],
                        env_extras[i],
                    )
                )

            all_outputs = await asyncio.gather(*tasks)
            ...

In ``generate_trajectory`` we start a Ray task to generate a trajectory and evaluate it for the given instance. More concretely, this consists of the following:

1. Generation:
    - Initialize a sandbox / environment for the instance
    - Generate a trajectory in this environment with Mini-SWE-Agent. For inference, we configure Mini-SWE-Agent to use the HTTP endpoint provided by SkyRL.
    - Store the generated git patch after generation completes.
2. Evaluation: 
    - Initialize a fresh environment for the given instance using the given backend.
    - Apply the model's git patch to the working directory in the environment.
    - Run the evaluation script for the instance. If the script runs successfully, the instance is considered to be resolved, and unresolved otherwise.

By running this workflow as a Ray task, we are also able to scale up generation across all the nodes in the Ray cluster. 


 At a high level, the code looks as follows:

.. code-block:: python

    @ray.remote(num_cpus=0.01)
    def init_and_run(instance: dict, litellm_model_name: str, sweagent_config: dict, data_source: str, ...):
        model = get_model(litellm_model_name, sweagent_config.get("model", {}))
        error = None
        try:
            env = get_sb_environment(sweagent_config, instance, data_source)
            agent = DefaultAgent(model, env, **sweagent_config.get("agent", {}))
            exit_status, model_patch = agent.run(instance["problem_statement"])
            eval_result = evaluate_trajectory(instance, model_patch, sweagent_config, data_source)
        except Exception as e:
            error = str(e)
        return agent.messages, eval_result, error

    class MiniSweAgentGenerator(SkyRLGymGenerator):
        async def generate_trajectory(self, prompt, env_extras, ...): 
            messages, eval_result, error = init_and_run.remote(env_extras["instance"], ...)
            ...

Note that the full implementation has some additional logic for configuration and error handling, and can be found in :code_link:`examples/mini_swe_agent/mini_swe_generator.py`.


Dataset preparation
-------------------

For training, we use `SWE-Gym <https://huggingface.co/SWE-Gym>`_, and more specifically the subset of SWE-Gym in `SumanthRH/SWE-Gym-Subset <https://huggingface.co/datasets/SumanthRH/SWE-Gym-Subset>`_.

Execute the following command: 

.. code-block:: bash

    # execute from skyrl-train directory
    uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset


Training
---------

Prerequisites: Ensure that you have the required environment backend installed for generating trajectories with Mini-SWE-Agent. By default, we use `Podman <https://podman.io/docs>`_. This can be modified in :code_link:`examples/mini_swe_agent/swebench.yaml` 

We provide two example scripts: One for Qwen3-8B model and another for the `Qwen/Qwen3-Coder-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>`_ model. While the first script for Qwen3-8B requires a single 8xH100 node, the script for the 30B model requires 2 8xH100 nodes for training.

.. code-block:: bash

    # execute from skyrl-train directory
    bash examples/mini_swe_agent/run_mini_swe_8B.sh
    # or for 30B:
    # bash examples/mini_swe_agent/run_mini_swe_30B.sh


Tips
~~~~~

- If you notice too many errors such as ``ValueError: The decoder prompt (length xxxx) is longer than the maximum model length`` in the logs, this means that the LLM is hitting context length limits. Training can still proceed as usual, but if there are too many such errors per batch, then you should either increase the sequence length (increase ``max_input_length`` and ``max_generate_length``) or reduce the number of steps in the ``swebench.yaml`` file.
- The task can sometimes be too difficult for the base model. For convenience, we log the list of rewards in a batch. If the rewards are all zeros, then the batch is too hard. If you notice too many such batches in your dataset, you should either (1) filter your data to have a better mix of easy and hard samples to promote learning (2) choose a stronger base model or (3) increase ``step_limit`` in ``swebench.yaml``. We've noticed that SWE-Gym can be hard (i.e most 0 rewards) for the Qwen3-8B with the given settings. The choice of the available tools can also affect performance (in Mini-SWE-Agent, agents have one tool - bash commands)
- If you notice errors like "Error during evaluation [Errno 7] Argument list too long: 'podman'" , this is because the evaluation logic currently applies the model's git patch in-line, and for very large git patches, you will hit system ``ARG_MAX`` limits. On modern systems, this maximum is ~ 1 MB, which is very generous. We thus make a simple assumption that large patches that exceed this limit are meant to be incorrect.
- If running podman within a container, you might hit errors due to insufficient UIDs. While the training logs will only have a brief error message: ``Command '['podman', 'run', '-d', '--name', 'minisweagent-e7fbf68e', '-w', '/testbed', '--rm', 'docker://swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-24026:latest', 'sleep', '2h']' returned non-zero exit status 125.``, more information is available if you run the command with ``--log-level=debug``. Make sure that your user namespace has sufficient UIDs and GIDs. You have two options on Linux-based machines:

    1. Edit the ``/etc/subuid`` and ``/etc/subgid`` files to use a large range such as ``100000-1100000``.
    2. Set ``ignore_chown_errors=true`` in the containers.conf file for Podman as described `here <https://github.com/containers/podman/blob/737108ba04277731534eb718b5406e7a8406f8f4/docs/source/markdown/podman.1.md?plain=1#L467>`_.
