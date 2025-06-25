SkyRL-SQL
=========

We provide scripts to reproduce the results for `SkyRL-SQL-7B <https://novasky-ai.notion.site/skyrl-sql>`_ using SkyRL-train and SkyGym.


Pre-requisites 
---------------

Make sure to have followed the installation commands in :ref:`installation <installation>`. 


Start Ray
---------

Start ray in your cluster following the guide: https://docs.ray.io/en/latest/ray-core/starting-ray.html. 


Data Preparation
----------------


We provide the dataset we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data 
Make sure to download the dataset and update the path in `DATA_PATH` in the script. 

.. code-block:: bash

    huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset


DB environment 
---------------

Make sure to setup the database files needed for training. We use the database from `OmniSQL <https://github.com/RUCKBReasoning/OmniSQL/edit/main/train_and_evaluate/README.md>`_. 

You can download the datasets from:
- `ModelScope-OmniSQL-datasets <https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary>`_
- `HuggingFace-OmniSQL-datasets <https://huggingface.co/datasets/seeklhy/OmniSQL-datasets>`_



The datasets include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

Unzip `data.zip` in this folder, and set the corresponding `DB_PATH` in the training script below. You can download and unzip the data by running

.. code-block:: bash

    huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir <path_to_file.zip>
    unzip <path_to_file.zip>

Running the scripts 
-------------------

We provide a script `run_skyrl_sql.sh <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/text_to_sql/run_skyrl_sql.sh>`_ for reproducing the results for SkyRL-SQL-7B. Make sure to substitute the `DB_PATH`  and `DATA_PATH` variables with your own.

.. code-block:: bash
    export WANDB_API_KEY=<wandb-api-key>
    bash examples/skyrl-sql/run_skyrl_sql.sh



