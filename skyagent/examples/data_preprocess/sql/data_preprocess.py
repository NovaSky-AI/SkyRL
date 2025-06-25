import os

from datasets import Dataset
import argparse
from tqdm import tqdm
import json 


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

short_system_prompt = THINKING_SYSTEM

if __name__=='__main__':
    parser=argparse.ArgumentParser()    
    parser.add_argument('--input',type=str, default='600_synsql.json')
    parser.add_argument('--output', type=str, default='dataset/SkyRL-SQL-653-data')
    args=parser.parse_args()

        
    # read through the json file 
    output_dataset=[]
    split = 'train'
    i = 0 
    with open(args.input, 'r') as f:
        input_dataset = json.load(f)
            
        for data_entry in tqdm(input_dataset):
            cur_data = {
                "data_source": "synsql",
                "prompt": [{
                    "role": "system",
                    "content": short_system_prompt
                },
                {
                    "role": "user",
                    "content": "{db_details}:"+ data_entry["schema"] + ";\n {external_knowledge}: " + data_entry["external_knowledge"] + ";\n {question}: " + data_entry["question"],
                }],
                "ability": 'synsql',
                "db_id": data_entry['db_id'],
                "data": data_entry['data'], 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry['sql'],
                },
                "extra_info": {
                    'split': 'dummy',
                    'index': 0
                }
            }
            output_dataset.append(cur_data)
            i += 1 
            
        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)
        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
    
    split = 'validation'
    output_dataset = [] 
    with open(args.input, 'r') as f:
        input_dataset = json.load(f)
            
        for data_entry in tqdm(input_dataset):
            cur_data = {
                "data_source": "synsql",
                "prompt": [{
                    "role": "system",
                    "content": short_system_prompt
                },
                {
                    "role": "user",
                    "content": "{db_details}:"+ data_entry["schema"] + "; {question}: " + data_entry["question"],
                }],
                "ability": 'synsql',
                "db_id": data_entry['db_id'],
                "data": data_entry['data'], 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry['sql'],
                },
                "extra_info": {
                    'split': 'dummy',
                    'index': 0
                }
            }
            output_dataset.append(cur_data)
            break 
            
        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)
        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
