import sys
from pathlib import Path
import os

from datetime import datetime
import json
import argparse
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

import dspy
import wandb
from assertion.livecodebench.program import NaiveCodeGenerator
from assertion.livecodebench.data import lcb_data
from assertion.evaluator import Evaluator
from assertion.livecodebench.live_code_bench_execute import check_correctness
from assertion.livecodebench.utils import post_process_code
from assertion.eval_utils import ProgressLogger

from concurrent.futures import ProcessPoolExecutor


dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
# Example CLI:
# python evals/eval_ranker_original_cli.py \
#   --dataset test \
#   --gen_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
#   --prog_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
#   --gen_port 7001 \
#   --prog_port 7000 \
#   --num_workers 20

###########################################################
#   CHECK FUNCTION
###########################################################

def check_pass(example, pred):
    timeout = 50
    result = check_correctness(
        example,
        post_process_code(pred.code),
        timeout=timeout,
        is_extracted=True,
        fast_check=True,
    )
    return result["passed"]


###########################################################
#   CREATE LM OBJECTS (PER PROCESS)
###########################################################d

def make_lm(model_name: str, port: int):
    model = f"openai/{model_name}"
    print(f"Making LM for {model_name} on port {port}")
    lm = dspy.LM(
        model=model,
        api_base=f"http://localhost:{port}/v1",
        api_key="fake-key",
        temperature=1,
        model_type="chat",
        cache=False,
        # max_tokens=2,
    )

    return lm

def health_check(model_name: str, port: int):
    lm = make_lm(model_name, port)
    response = lm(messages=[{"role": "user", 
                             "content": "Say this is a test message."}])
    
    assert response is not None
    
    print(f"Health check response: {response}")

###########################################################
#   WORKER PROCESS: RUN ONE EXPERIMENT
###########################################################

def run_eval(prog_model, prog_port, dataset="test", num_workers=10, progress_logger=None):
    # Get the dataset based on the dataset argument
    trainset, testset = lcb_data()
    
    if dataset == "train":
        evalset = trainset
    elif dataset == "test":
        evalset = testset
    elif dataset == "both":
        evalset = trainset + testset
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'train' or 'test'")

    # Create model instances
    prog_lm = make_lm(prog_model, prog_port)

    # Create program instance
    prog = NaiveCodeGenerator(prog_lm=prog_lm)
    
    evaluator = Evaluator(
        program=prog,
        evalset=evalset,
        metric_fn=check_pass,
        num_workers=num_workers,
        debug=False,
        progress_logger=progress_logger,
    )

    all_results = evaluator.evaluate()
    
    evaluator.close()
    return all_results

def get_model_name(model_name):
    if model_name == "Qwen/Qwen2.5-Coder-7B-Instruct":
        return "7B"
    elif model_name == "Qwen/Qwen2.5-Coder-1.5B-Instruct":
        return "1point5B"
    elif model_name == "Qwen/Qwen2.5-Coder-3B-Instruct":
        return "3B"
    elif model_name == "Harryllh/lcb_test_generator_1.5b_400steps":
        return "1point5B_400steps"
    elif model_name == "Harryllh/lcb_test_generator_1.5b_230steps":
        return "1point5B_230steps"
    elif model_name == "Harryllh/lcb_test_generator_1.5b_100steps":
        return "1point5B_100steps"
    elif model_name == "my_lora":
        return "lora_15b_100steps"
    elif model_name == "Harryllh/lcb_test_generator_3b_200steps":
        return "3B_200steps"
    elif model_name == "Harryllh/lcb_prog_generator_1.5b_100steps":
        return "1point5B_100steps"
    elif model_name == "Harryllh/lcb_prog_generator_3b_100steps":
        return "3B_100steps"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
def save_results(all_results, args):
    label = f"prog_{get_model_name(args.prog_model)}_{args.dataset}"
    result_path = f"{args.base_path}/{label}.json"

    print(f"Saving results to {result_path}")
    
    json_results = {"time_stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    "results": {i: [] for i in range(len(all_results))}}
    
    for cur_idx, cur_results in enumerate(all_results):
        for example, pred, score in cur_results:
            example_dict = {k: getattr(example, k, None) for k in ['prompt', 'task_id', 'is_stdin']}
            pred_dict = {k: getattr(pred, k, None) for k in ['code'] if hasattr(pred, k)}

            json_results["results"][cur_idx].append({
                "example": example_dict,
                "pred": pred_dict,
                "score": score,
            })

    with open(result_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Compute avg score from all results
    final_score_list = []
    
    for cur_results in all_results:
        for example, pred, score in cur_results:
            final_score_list.append(score)
            
    final_metric = sum(final_score_list) / len(final_score_list)
        
    print(f"Final Metric for {label}: {final_metric}")

###########################################################
#   MAIN ENTRY: PARSE CLI ARGUMENTS AND RUN
###########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NaiveCodeGenerator")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["train", "test", "both"],
        help="Dataset split to use: 'train' or 'test' or 'both'"
    )
    parser.add_argument(
        "--prog_model",
        type=str,
        required=True,
        help="Model name for the program LM"
    )
    parser.add_argument(
        "--prog_port",
        type=int,
        required=True,
        help="Port number for the program model API"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of worker processes for evaluation (default: 10)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/baseline/",
        help="Base path for saving results (default: /work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/baseline/)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for evaluation (default: 1)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to store log files for each process (default: ./logs)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Name of the experiment (default: '')"
    )

    args = parser.parse_args()
    
    print("Starting evaluation...")
    
    print(f"Number of runs: {args.num_runs}")
    
    print(f"Health checking {args.prog_model} on port {args.prog_port}")
    health_check(args.prog_model, args.prog_port)
    
    def single_run(i):
        label = f"prog_{get_model_name(args.prog_model)}_{args.dataset}"

        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = log_dir / f"evaluation_run_{i}_{label}.log"

        logging.root.handlers = []
        
        logging.basicConfig(
            level=logging.INFO,
            # level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Process {i}: Logging to {log_filename}")
        
        # Initialize wandb for this process
        wandb_run = wandb.init(
            project="assertion-proj",
            name=f"{args.experiment_name}_{label}_run_{i}",
            config={
                "prog_model": args.prog_model,
                "dataset": args.dataset,
                "prog_port": args.prog_port,
                "num_workers": args.num_workers,
                "run_id": i,
            },
            settings=wandb.Settings(start_method="fork"),
        )
        
        progress_logger = ProgressLogger(wandb_run=wandb_run)
        
        result = run_eval(
            prog_model=args.prog_model,
            prog_port=args.prog_port,
            dataset=args.dataset,
            num_workers=args.num_workers,
            progress_logger=progress_logger,
        )
        
        return i, result
    
    all_results = [None] * args.num_runs  
    
    with ProcessPoolExecutor(max_workers=args.num_runs) as ex:
        print(f"Starting {args.num_runs} runs...")
        futures = {ex.submit(single_run, i): i for i in range(args.num_runs)}

        for future in futures:
            idx, result = future.result()
            all_results[idx] = result
        
    assert len(all_results) == args.num_runs

    save_results(all_results, args)
