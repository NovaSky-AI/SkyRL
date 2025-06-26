Installation
============

Requirements
------------
- CUDA version >=12.4
- `uv <https://docs.astral.sh/uv/>`_

We use `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. We also make use of the `uv` and `ray` integration to manage dependencies for ray workers. 

If you're running on an existing Ray cluster, make sure to use Ray 2.44.0 and Python 3.12.


Docker (recommended)
---------------------

We provide a docker image with the base dependencies ``sumanthrh/skyrl-train-ray-2.44.0-py3.12-cu12.4`` for quick setup. 

1. Make sure to have `NVIDIA Container Runtime <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ installed.

2. You can launch the container using the following command:

.. code-block:: bash

    docker run -it  --runtime=nvidia --gpus all sumanthrh/skyrl-train-ray-2.44.0-py3.12-cu12.4 --name skyrl-train

3. Inside the launched container, setup the latest version of the project:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd skyrl-train
    uv sync --extra vllm
    source .venv/bin/activate


That is it! You should now be to able to run our :doc:`quick start example <quickstart>`.

Install without Dockerfile
--------------------------

For installation without the Dockerfile, make sure you meet the pre-requisities: 

- CUDA 12.4
- `uv <https://docs.astral.sh/uv/>`_
- `ray <https://docs.ray.io/en/latest/>`_ 2.44.0

All project dependencies are managed by `uv`.

Clone the repo and `cd` into the `skyrl` directory:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd skyrl-train 

Base environment
~~~~~~~~~~~~~~~~

We recommend having a base virtual environment for the project.

With ``uv``: 

.. code-block:: bash

    uv venv --python 3.12 <path_to_venv>

If ``<path_to_venv>`` is not specified, the virtual environment will be created in the current directory at ``.venv``.

.. tip::
    Because of how Ray ships content in the `working directory <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html>`_, we recommend that the base environment is created *outside* the package directory.

Then activate the virtual environment and install the dependencies.

.. code-block:: bash

    source <path_to_venv>/bin/activate
    uv sync --extra vllm

With ``conda``: 

.. code-block:: bash

    conda create -n skyrl-train python=3.12
    conda activate skyrl-train

You should now be to able to run our :doc:`quick start example <quickstart>`.


Development 
-----------

For development, make sure to use ``--extra dev`` so that the dev dependencies are included.

.. code-block:: bash

    uv run --extra dev pytest -s tests/cpu