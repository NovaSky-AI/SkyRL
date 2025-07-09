Search
=========

We provide scripts to reproduce our results for training a multi-turn search agent using the dataset and recipe from `Search-R1 <https://arxiv.org/pdf/2503.09516>`_.

Pre-requisites
--------------

Make sure to have followed the installation commands in :ref:`installation <installation>`. 


Start Ray
---------

Start ray in your cluster following the guide: https://docs.ray.io/en/latest/ray-core/starting-ray.html. 

Data Preparation
----------------

We provide a script to download the dataset from huggingface, and preprocess it to run in SkyRL-Gym.

.. code-block:: bash

    uv run --isolated examples/search/searchr1_dataset.py --local_dir ~/data/searchR1

Start the Search Engine
------------------------

Retriever environments 
~~~~~~~~~~~~~~~~~~~~~~

```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

## Download the Index
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

## Prepare Datasets 
```bash
python examples/search/searchr1_dataset.py --local_dir $local_dir
```

## Start the Local Flat e5 Retrieval Server 

GPU version 
```bash
conda activate retriever

bash examples/search/retriever/retrieval_launch.sh 
```