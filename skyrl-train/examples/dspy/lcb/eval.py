from .program import CodeGeneratorWithRanker_Original
from .data import lcb_data
from assertion.evaluator import Evaluator
from .live_code_bench_execute import check_correctness
from .utils import post_process_code
import dspy
from concurrent.futures import ProcessPoolExecutor, as_completed



dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

###########################################################
#   CHECK FUNCTION
###########################################################

def check_pass(example, pred):
    timeout = 50
    result = check_correctness(
        example,
        post_process_code(pred.code),
        timeout,
        is_extracted=True,
        fast_check=True,
    )
    return result["passed"]


###########################################################
#   CREATE LM OBJECTS (PER PROCESS)
###########################################################

def make_lm(port_lm):
    lm = dspy.LM(
        model="openai/my_lora",
        api_base=f"http://localhost:{port_lm}/v1",
        api_key="fake-key",
        temperature=0.7,
        model_type="chat",
        cache=False,
        # max_tokens=16000,
    )

    return lm

def make_qwen(port_qwen):
    qwen = dspy.LM(
        model="openai/Qwen/Qwen2.5-Coder-7B-Instruct",
        api_base=f"http://localhost:{port_qwen}/v1",
        api_key="fake-key",
        temperature=0.7,
        model_type="chat",
        cache=False,
        # max_tokens=16000,
    )

    return qwen


###########################################################
#   WORKER PROCESS: RUN ONE EXPERIMENT
###########################################################

def run_one_prog(label, port_lm=None, port_qwen=None):
    trainset, testset = lcb_data()

    # Re-create program instance per worker
    if label == "prog_qwen":
        print("single qwen")
        qwen = make_qwen(port_qwen)
        prog = CodeGeneratorWithRanker_Original(prog_lm=qwen, generator_lm=qwen)

    elif label == "prog_lm":
        print("single trained lm")
        lm = make_lm(port_lm)
        prog = CodeGeneratorWithRanker_Original(prog_lm=lm, generator_lm=lm)

    elif label == "prog_mixed":
        lm = make_lm(port_lm)
        qwen = make_qwen(port_qwen)
        prog = CodeGeneratorWithRanker_Original(prog_lm=qwen, generator_lm=lm)

    else:
        raise ValueError(f"Unknown program label: {label}")

    evaluator = Evaluator(
        program=prog,
        evalset=testset,
        metric_fn=check_pass,
        num_workers=10,
        debug=False,
    )

    results = evaluator.evaluate()
    scores = [score for (_, _, score) in results if isinstance(score, (int, float))]
    avg_score = sum(scores) / len(scores)

    return label, avg_score


###########################################################
#   MAIN ENTRY: START 3 PROCESSES
###########################################################

if __name__ == "__main__":
    # Choose whatever ports you want

    label, avg = run_one_prog("prog_mixed", port_lm=7000, port_qwen=7001)
    # label, avg = run_one_prog("prog_qwen", port_qwen=7001)
    print(f"Final Metric for {label}: {avg}")
    # config = [
    #     ("prog_qwen",   7000, 7001),   # (label, lm_port, qwen_port)
    #     ("prog_lm",     7002, 7003),
    #     ("prog_mixed",  7004, 7005),
    # ]

    # with ProcessPoolExecutor(max_workers=len(config)) as executor:
    #     futures = {
    #         executor.submit(run_one_prog, label, port_lm, port_qwen): label
    #         for (label, port_lm, port_qwen) in config
    #     }

    #     for fut in as_completed(futures):
    #         label, avg = fut.result()
    #         print(f"Final Metric for {label}: {avg}")
