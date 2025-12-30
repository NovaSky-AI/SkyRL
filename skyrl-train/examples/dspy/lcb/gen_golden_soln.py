import argparse
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

import dspy
from data import lcb_data
from utils import GenerateLCBcodestdin
from lcb_utils import reduce_preds
from program import GenerateLCBcodestdin, GenerateLCBcodefunctional
from live_code_bench_execute import check_correctness
from utils import post_process_code

load_dotenv()



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


def main(mode: str):
    """
    mode: 'train' or 'test'
    """

    # initialize model and data
    lm = dspy.LM("openai/gpt-5.1")
    trainset, testset = lcb_data()
    dataset = trainset if mode == "train" else testset

    stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
    functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)

    gold_preds = {}

    failed_examples = []
    for example in tqdm(dataset, desc=f"Running {mode} set"):
        task_id = example.task_id
        is_stdin = example.is_stdin
        prompt = example.prompt

        prog_generator = stdin_prog if is_stdin else functional_prog
        prog_generator.set_lm(lm)

        

        for retry in range(3):
            gold_pred = prog_generator(prompt=prompt)
            gold_pred = reduce_preds([gold_pred])[0]

            result_dict = check_correctness(
                example,
                post_process_code(gold_pred.code),
                60,
                is_extracted=True,
                fast_check=True,
            )
            passed = result_dict["passed"]
            if passed:
                break
            else:
                print("failed")
                error_message = ' '.join(result_dict['maybe_error_messages'])
                if len(error_message) > 10000:
                    print("error message too long")
                    continue
                prompt = prompt + "\n" + "Here's a failed test case: " + "\n" + "code: " + gold_pred.code + "\n" + "error message: " + ' '.join(error_message)

        if retry == 2:
            failed_examples.append(task_id)
        gold_preds[task_id] = gold_pred
    # Save output
    output_name = f"gold_preds_{mode}.pkl"
    with open(output_name, "wb") as f:
        pickle.dump(gold_preds, f)

    print(f"\n✅ Finished {mode} mode. Saved results to {output_name}")
    print("failed examples: ", len(failed_examples))
    print("failed examples: ", failed_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LCB generation in train or test mode.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Choose whether to run on the training or test set.",
    )
    args = parser.parse_args()

    main(args.mode)
