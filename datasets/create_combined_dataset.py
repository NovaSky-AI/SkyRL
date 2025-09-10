import json
import random
import pandas as pd
import os
from typing import Dict, List, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

discovery_instructions = """

"""

qr_instructions = """

"""

def create_base_prompt(context: str, question: str, dataset_locations: str, metadata_section: str, answer_format: str) -> str:
    prompt = f"""
You are an expert data scientist and data analyst who tackles analytical challenges through systematic thinking and thorough investigation. 
For each task, you will receive a question along with file paths to relevant data and background information. Your analysis should make full use of these data sources.

TASK: Analyze the provided dataset to answer the query.

DATASET INFORMATION:
{context}

{metadata_section}

QUERY: {question}

DATASET LOCATIONS (use full paths):
{dataset_locations}

INSTRUCTIONS:
1. Load and explore the dataset(s) using Python (use the dataset locations above)
2. If multiple datasets are provided, analyze each one to find out what is relevant and consider how they relate to each other
3. Perform data preprocessing, statistical analysis and, when appropriate, apply additional methods such as regression modeling, hypothesis testing, time-series, or spatial analysis, or feature engineering to identify relationships between variables.
4. Where simple statistics are insufficient or the data does not contain enough information itself, attempt more advanced or alternative approaches (e.g., using regression, dimensionality reduction, robustness checks, or combining datasets).
5. Provide insights and conclusions based on your analysis and answer the query in your final answer section.
6. Do one step at a time. Explore the data and then answer the query.
7. Do not use plotting libraries. You cannot see the plots.
8. When workflow tags are provided, you should use them to guide your analysis.
9. Only form your final answer when you have enough evidence.
10. Code execution is continuous - variables and data loaded in previous steps remain available for subsequent analysis. Do not need to reload the same dataset or variables.


You MUST use the following format for your response. Each step must follow this exact structure:

<reasoning>
Write clear reasoning about what you plan to do next and why. Be specific about your analytical approach.
</reasoning>
<python>
Write executable Python code here. Each code block should do ONE specific task.
Code must be complete and runnable. Include all necessary imports.
</python>
<information>
The output/results from your Python code will appear here.
This section is read-only - you cannot write here.
</information>

Repeat these blocks for each analysis step. When you reach your conclusion, you should follow this structure:

<reasoning>
Write clear reasoning about how you came up with your final answer.
</reasoning>
{answer_format}
"""
    return prompt

def create_qr_prompt(item: Dict[str, Any]) -> str:
    context = item['context']
    question = item['question']
    data_paths = item['data']
    metadata = item['metadata']
    
    dataset_locations = '\n'.join(data_paths) if data_paths else 'No dataset provided'
    
    # Build QR-specific metadata section
    keywords = metadata['keywords']
    question_type = metadata['question_type']
    reference = metadata['reference']
    
    metadata_section = f"""Keywords: {', '.join(keywords)}
Question Type: {question_type}
Reference: {reference}"""
    
    # QR-specific answer format
    answer_format = """<answer>
Write your final answer here. Requirements:
1. Direct and concise answer to the query
2. Derived from the provided dataset
3. If numerical, provide the exact value as requested in the question (e.g., rounded to the nearest hundredth as specified)
</answer>"""
    
    return create_base_prompt(context, question, dataset_locations, metadata_section, answer_format)

def create_discovery_prompt(item: Dict[str, Any]) -> str:
    context = item['context']
    question = item['question']
    data_paths = item['data']
    metadata = item['metadata']
    
    dataset_locations = '\n'.join(data_paths) if data_paths else 'No dataset provided'
    
    # Build DiscoveryBench-specific metadata section
    domain = metadata['domain']
    domain_knowledge = metadata['domain_knowledge']
    workflow_tags = metadata['workflow_tags']
    columns_info = metadata['columns_info']
    dataset_descriptions = metadata['dataset_descriptions']
    
    metadata_lines = []
    if domain:
        metadata_lines.append(f"DOMAIN: {domain}")
    
    metadata_lines.append("DATASET DESCRIPTIONS:")
    metadata_lines.append(chr(10).join(dataset_descriptions) if dataset_descriptions else "No dataset descriptions available")
    
    metadata_lines.append("COLUMNS:")
    metadata_lines.append(columns_info)
    
    if domain_knowledge:
        metadata_lines.append(f"DOMAIN KNOWLEDGE: {domain_knowledge}")
    if workflow_tags:
        metadata_lines.append(f"WORKFLOW TAGS: {workflow_tags}")
    
    # Only add workflow if key exists
    if 'workflow' in metadata:
        workflow = metadata['workflow']
        metadata_lines.append("WORKFLOW:")
        metadata_lines.append(workflow)
    
    metadata_section = '\n\n'.join(metadata_lines)
    
    # DiscoveryBench-specific answer format
    answer_format = """<answer>
Write your final scientific hypothesis here. Requirements:
1. Direct and concise answer to the query
2. Derived from the provided dataset
3. Clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance.
</answer>"""
    
    return create_base_prompt(context, question, dataset_locations, metadata_section, answer_format)

def process_dataset(items: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    processed = []
    
    for idx, item in enumerate(items):
        if dataset_name == "qrdata":
            user_content = create_qr_prompt(item)
        elif dataset_name == "discoverybench":
            user_content = create_discovery_prompt(item)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        system_prompt = {
            "role": "system",
            "content": "You are a data scientist who is expert in data analysis and data modeling. You can analyze datasets and generate scientific hypotheses."
        }
        
        data = {
            "data_source": dataset_name,
            "prompt": [
                system_prompt,
                {
                    "role": "user", 
                    "content": user_content
                }
            ],
            "env_class": "allocated_code",
            "reward_spec": {
                "method": "rule",
                "ground_truth": str(item['answer'])
            },
            "extra_info": {
                "question": item['question'],
                "answer": item['answer'],
                "context": item['context'],
                "data_paths": item['data'],
                "metadata": item['metadata'],
                "index": idx,
                "original_item": item
            }
        }
        processed.append(data)
    
    return processed

def create_train_val_split(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data

def main():
    random.seed(41)
    
    print("Loading QR data...")
    qr_data = load_json_data('/data/fan/clone/SkyRL/datasets/qrdata.json')
    print(f"Loaded {len(qr_data)} QR examples")
    
    print("Loading DiscoveryBench data...")
    discovery_data = load_json_data('/data/fan/clone/SkyRL/datasets/discoverybench.json')
    print(f"Loaded {len(discovery_data)} DiscoveryBench examples")
    
    print("Processing QR data...")
    processed_qr = process_dataset(qr_data, "qrdata")
    
    print("Processing DiscoveryBench data...")
    processed_discovery = process_dataset(discovery_data, "discoverybench")
    
    print("Creating QR splits...")
    qr_train, qr_val = create_train_val_split(processed_qr)
    
    print("Creating DiscoveryBench splits...")
    discovery_train, discovery_val = create_train_val_split(processed_discovery)
    
    print("Combining datasets...")
    all_train = qr_train + discovery_train
    all_val = qr_val + discovery_val
    
    random.shuffle(all_train)
    random.shuffle(all_val)
    
    print("Creating output directory...")
    output_dir = '/data/fan/clone/SkyRL/datasets/data_parquet'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving datasets...")
    train_df = pd.DataFrame(all_train)
    val_df = pd.DataFrame(all_val)
    
    train_path = os.path.join(output_dir, "combined_train.parquet")
    val_path = os.path.join(output_dir, "combined_validation.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved {len(all_train)} training samples to {train_path}")
    print(f"Saved {len(all_val)} validation samples to {val_path}")
    print(f"QR training samples: {len(qr_train)}")
    print(f"QR validation samples: {len(qr_val)}")
    print(f"DiscoveryBench training samples: {len(discovery_train)}")
    print(f"DiscoveryBench validation samples: {len(discovery_val)}")
    
    if all_train:
        print("\n" + "="*80)
        print("SAMPLE TRAINING EXAMPLE:")
        print("="*80)
        sample = random.choice(all_train)
        print(f"Data source: {sample['data_source']}")
        print(f"Question: {sample['extra_info']['question']}")
        print("\nFULL PROMPT:")
        print("-" * 80)
        print(sample['prompt'][1]['content'])
        print("-" * 80)
        print("\nANSWER:")
        print(sample['extra_info']['answer'])
        print("="*80)

if __name__ == "__main__":
    main()
