#!/usr/bin/env python3
import argparse
import os

from .program import CodeGeneratorWithRankerAssertTestGen
from .data import lcb_data
from assertion.trainer import LMManager, AssertionTrainer, ArborLM
from .utils import post_process_code
from .live_code_bench_execute import check_correctness
from assertion.evaluator import Evaluator
import dspy


def check_pass(example, pred):
    timeout = 50
    result_dict = check_correctness(
        example,
        post_process_code(pred.code),
        timeout,
        is_extracted=True,
        fast_check=True,
    )
    return result_dict["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--arbor_port", type=int, default=8002)
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--num_programs", type=int, default=5)
    args = parser.parse_args()

    # -------------------------------------------------------
    # Environment Variables
    # -------------------------------------------------------
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "999999999"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # -------------------------------------------------------
    # Load Data
    # -------------------------------------------------------
    trainset, testset = lcb_data()

    # -------------------------------------------------------
    # LM Specs / LM Manager
    # -------------------------------------------------------
    arbor_port = args.arbor_port
    lm_specs = [
        ("prog_lm", "Qwen/Qwen2.5-Coder-7B-Instruct", 7000, "vllm"),
        ("generator_lm", args.model, arbor_port, "arbor"),
    ]

    lm_manager = LMManager(lm_specs)
    generator_lm = lm_manager.get_lm("generator_lm")
    prog_lm = lm_manager.get_lm("prog_lm")

    # Disable DSPy cache/logs
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.disable_logging()

    # -------------------------------------------------------
    # Program Instance
    # -------------------------------------------------------
    prog = CodeGeneratorWithRankerAssertTestGen(
        prog_lm=prog_lm,
        generator_lm=generator_lm,
        num_tests=args.num_tests,
        num_programs=args.num_programs,
    )

    # -------------------------------------------------------
    # Trainer
    # -------------------------------------------------------
    trainer = AssertionTrainer(
        program=prog,
        trainset=trainset,
        lm_manager=lm_manager,
        assert_fn=check_pass,
        batch_size=args.batch_size,
    )

    # -------------------------------------------------------
    # Run Training
    # -------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
