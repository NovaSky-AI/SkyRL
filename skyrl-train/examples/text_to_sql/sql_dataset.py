import os

from datasets import Dataset, DatasetDict
import argparse

THINKING_SYSTEM_SINGLE_TURN = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analisying questions, 
summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.

Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- If you have 0 turns left, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm? 
    Database Schema:
    Table: animals
    - id (INTEGER, PRIMARY KEY)
    - species (TEXT)
    - age (INTEGER)
    - name (TEXT)

<think>I am querying how many pigs are in the farm. Since the question asks for how many pigs, I can use a SELECT COUNT() statement to query from the animals table where species is pig.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
"""
THINKING_SYSTEM = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analisying questions, 
summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.


Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Based on this observation, you can think again and refine.
- The returned dataframe will be truncated in 50 rows if observation is too long. 
- If you find no further exploration is needed or reaches max turns, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm? 
    Database Schema:
    Table: animals
    - id (INTEGER, PRIMARY KEY)
    - species (TEXT)
    - age (INTEGER)
    - name (TEXT)

<think>I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
+----------+
| COUNT(*) |
+----------+
|   12     |
+----------+
</observation>
<think>The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
"""

short_system_prompt = THINKING_SYSTEM  # use THINKING_SYSTEM_SINGLE_TURN if you want to use single turn generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sql/653_skyrl_sql.json")
    parser.add_argument("--validation_input", type=str, default="data/sql/validation_spider_dev.json")
    parser.add_argument("--output_dir", type=str, default="data/sql")
    args = parser.parse_args()

    # Load the JSON file directly as a Dataset
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_json(args.input),
            "validation": Dataset.from_json(args.validation_input),
        }
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            """Transform each dataset example into the required format"""
            user_content = (
                "{db_details}:"
                + example["schema"]
                + ";\n {external_knowledge}: "
                + example["external_knowledge"]
                + ";\n {question}: "
                + example["question"]
            )

            data_source = "synsql" if split == "train" else "spider"  # use spider-dev for validation

            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "env_class": "text2sql",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": example["sql"],
                },
                # Custom fields specific to the SynSQL dataset:
                "db_id": example["db_id"],
                "data": example["data"],
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    # Process the dataset for train (all data)
    dataset_dict["train"] = dataset_dict["train"].map(function=make_map_fn("train"), with_indices=True)
    dataset_dict["validation"] = dataset_dict["validation"].map(function=make_map_fn("validation"), with_indices=True)

    print(
        f"Generated {len(dataset_dict['train'])} train examples and {len(dataset_dict['validation'])} validation examples"
    )
    dataset_dict["train"].to_parquet(os.path.join(args.output, "train.parquet"))
    dataset_dict["validation"].to_parquet(os.path.join(args.output, "validation.parquet"))
