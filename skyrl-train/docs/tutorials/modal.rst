Using Modal for SkyRL Training
=================================

Modal Labs
----------

`Modal Labs <https://modal.com/docs>`_ is a serverless platform for running containerized AI workloads with 
minimal infrastructure overhead. Modal provides seamless access to GPU resources without the complexity of 
managing cloud infrastructure directly.

.. note::
   If you're new to Modal? Complete the `Modal quickstart <https://modal.com/docs/guide>`_ first.

Overview
--------

The SkyRL Modal integration allows you to:

- **Run SkyRL commands** from your local repository inside Modal's cloud infrastructure
- **Use pre-configured containers** based on the SkyRL base image (``novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8``)
- **Mount your local SkyRL repository** to ``/root/SkyRL`` in the container
- **Access persistent storage** for data and checkpoints at ``/root/data`` and ``/home/ray/data``
- **Initialize Ray clusters** automatically for distributed computing
- **Stream live output** from your commands back to your local terminal

Prerequisites
-------------

Before using the Modal integration, ensure you have the following set up:

1. **Install Modal into your local environment**:

   .. code-block:: bash

       pip install modal

.. note:: For instructions on setting up the base environment required to use SkyRL, refer to `SkyRL Base Environment Setup <https://skyrl.readthedocs.io/en/latest/getting-started/installation.html#base-environment>`_

Ensure your environment is properly configured before proceeding with the Modal integration steps below.

2. **Set up Modal authentication**:

   .. code-block:: bash

       modal setup
  
.. note:: This will prompt you to authorize Modal access to your account. Refer to the `Modal authentication setup <https://modal.com/docs/guide/authentication>`_ docs.


3. **Clone the SkyRL repository**:

.. code-block:: bash

    git clone https://github.com/NovaSky-AI/SkyRL.git
    cd SkyRL/skyrl-train


Basic Usage
-----------

Navigate to the Modal integration directory and run the following commands:

.. code-block:: bash

    cd integrations/modal
    export MODAL_APP_NAME="your-app-name-here"
    modal run main.py --command "your-command-here"

The ``--command`` parameter accepts any command you want to run in the SkyRL container environment.


Test GPU Availability
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    modal run main.py --command "nvidia-smi"

This command verifies that GPU resources are available and properly configured in the Modal container.


Run Training Script
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    modal run main.py --command "bash examples/gsm8k/run_generation_gsm8k.sh"

This executes a complete training pipeline using the GSM8K dataset. You can also run other bash scripts through this setup.

Run from Subdirectory
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    modal run main.py --command "uv run search/run_skyrl_train_search.py"

Commands automatically run from the appropriate directory within the SkyRL repository structure.

Resource Configuration
~~~~~~~~~~~~~~~~~~~~~~

The default resource configuration provides:

- **GPU**: 1x NVIDIA L4 GPU
- **Timeout**: 3600 seconds (1 hour)

To modify resources, edit the ``@app.function()`` decorator in ``main.py``:

.. code-block:: python

    @app.function(
        image=image,
        gpu="A100:1",  # Change GPU type/count
        volumes=volume,
        timeout=7200,  # Change timeout (in seconds)
    )

.. tip::
   Check `Modal's GPU options <https://modal.com/docs/guide/gpu>`_ for more customization.

How It Works
------------

Image Creation
~~~~~~~~~~~~~~

The ``create_modal_image()`` function pulls the SkyRL base Docker image (``novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8``) 
and sets required environment variables (``SKYRL_REPO_ROOT``). It then mounts your local SkyRL repository to ``/root/SkyRL`` in 
the container and excludes unnecessary files (``.venv``, ``.git``, ``__pycache__``, etc.) for faster uploads.

Volume Management
~~~~~~~~~~~~~~~~~

The ``create_modal_volume()`` function creates or attaches a persistent volume named ``"skyrl-data"``. 
It then mounts the volume at ``/root/data`` for data persistence across runs and creates a symlink at ``/home/ray/data`` 
pointing to ``/root/data`` for compatibility from your local machine.

Command Execution
~~~~~~~~~~~~~~~~~

The ``run_script()`` function changes to the SkyRL repository directory (``/root/SkyRL/skyrl-train``), which 
ensures ``skyrl-gym`` is available for package dependencies, starts a Ray cluster and executes your command with 
live output streaming to your local terminal.

Data Persistence
----------------

Data stored in ``/root/data`` (or ``/home/ray/data``) persists across Modal runs through the attached volume. This persistence is useful for:

- **Dataset storage**: Generated datasets remain available for subsequent training runs
- **Model checkpoints**: Training checkpoints are preserved between sessions
- **Intermediate results**: Cached computations and processed data persist
- **Logs and artifacts**: Training logs and evaluation results are retained

Example workflow using persistent data:

.. code-block:: bash

    # Generate dataset (stored persistently)
    modal run main.py --command "uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
    
    # Run training using the persisted dataset
    modal run main.py --command "bash examples/gsm8k/run_generation_gsm8k.sh"
    
    # Resume training from checkpoint in subsequent runs
    modal run main.py --command "bash examples/gsm8k/resume_training.sh"

.. tip::
   Use ``modal volume ls`` to check volumes and ``modal shell`` for interfacing with the volume.


Long-Running Jobs
~~~~~~~~~~~~~~~~~

For extended training sessions, increase the timeout:

.. code-block:: python

    # In main.py, modify the function decorator
    @app.function(
        image=image,
        gpu="A100:1",
        volumes=volume,
        timeout=14400,  # 4 hours
    )

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

Configure multiple GPUs for larger models:

.. code-block:: python

    @app.function(
        image=image,
        gpu="A100:4",  # 4x A100 GPUs
        volumes=volume,
        timeout=7200,
    )


Refer to the official `Modal platform documentation <https://modal.com/docs>`_ and the `SkyRL repository <https://github.com/NovaSky-AI/SkyRL>`_.
