## Downloading SQL Data

We provide the dataset we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data-newfmt 
You can download the dataset by running the following command

```bash
    huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
```

## DB environment 

Make sure to setup the database files needed for training. We use the database from [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL/edit/main/train_and_evaluate/README.md)

You can download the datasets from:
- [ModelScope-OmniSQL-datasets](https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary)
- [HuggingFace-OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets)



The datasets include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

Unzip `data.zip` in this folder, and set the corresponding `DB_PATH` in the training script below. You can download and unzip the data by running

```bash
    huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir <path_to_file.zip>
    unzip <path_to_file.zip>
```