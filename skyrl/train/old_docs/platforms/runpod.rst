RunPod
======

`RunPod <https://www.runpod.io>`_ is an end-to-end AI cloud that simplifies building and deploying models. You can easily launch a GPU pod in seconds and run training jobs or inference workloads.


Use the following steps to run SkyRL-Train on Runpod. You can customize them to your needs.
These instructions were tested end-to-end on a single A40 GPU on Runpod, training on the
GSM8K dataset. It's a quick, minimal path to your first training run.

First, install Miniconda:

.. code-block:: bash

    cd $HOME
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all

Close the terminal and reopen it to activate the base conda environment. Then run the following
snippet. The NUMA installation follows :ref:`system-dependencies`.

.. code-block:: bash

    cd $HOME
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    # Optionally, export your WANDB_API_KEY
    echo "export WANDB_API_KEY=YOUR_WANDB_API_KEY" >> ~/.bashrc

    # Sometimes Runpod places `.cache` under `/workspace`, which can be slow.
    # We set them under `~/` here.
    mkdir -p "$HOME/.cache"
    echo 'export UV_CACHE_DIR="$HOME/.cache/uv"' >> ~/.bashrc
    echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc

    # ---------------
    # Install numactl (libnuma)
    # ---------------
    # Get the source
    cd $HOME
    wget https://github.com/numactl/numactl/releases/download/v2.0.16/numactl-2.0.16.tar.gz
    tar xzf numactl-2.0.16.tar.gz
    cd numactl-2.0.16

    # Build to a local prefix
    ./configure --prefix=$HOME/.local
    make
    make install
    cd $HOME

    # Point the compiler and linker to it
    echo "export CPATH=$HOME/.local/include:$CPATH" >> ~/.bashrc
    echo "export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

Close the terminal and reopen it. Then launch a basic GSM8K training run with the following
commands. For more, see :doc:`quickstart`.

.. code-block:: bash

    cd $HOME
    git clone https://github.com/NovaSky-AI/SkyRL
    cd SkyRL/skyrl-train
    pip install uv
    uv run --isolated python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
    NUM_GPUS=1 LOGGER=console bash examples/gsm8k/run_gsm8k.sh
