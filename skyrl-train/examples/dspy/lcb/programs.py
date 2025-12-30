import dspy
from .utils import GenerateLCBcodestdin, GenerateLCBcodefunctional, post_process_code, assert_test, assert_test_multiple, assert_dummy
from .lcb_utils import post_process_tests_inputs, reduce_preds, check_test
from .utils import assert_test
from dspy.adapters.xml_adapter import XMLAdapter
from collections import Counter
import json
import time

TIMEOUT_CONSTANT = 6
NUM_TESTS = 5
TEMPARATURE_STEP = 0.001


class GenerateTests_func_inputs(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate a complete set of potential inputs to test an AI-generated solution to the coding problem. Cover: (i) Edge cases, such as empty string or arrays, (ii) Complex and diffucult inputs, but do not include very long inputs. (iii) Other ones that can maximize the chance of catching a bug.  Provide the input and output in JSON format as follows: \
        {"input": <example_input>, "output": <expected_output>} Ensure the input and output match the types and structure expected for the problem. \
        Do not include any additional text or explanations, just the JSON object.'
    )

class GenerateTests_std_inputs(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
            desc="Generate some potential input and output pairs to test an AI-generated solution to the coding problem in the format of 'Input: input Output: output'. Assume input will be used via stdin and output is caputred stdout. Include (1) Edge cases, such as empty string or arrays, (3) Complex and diffucult inputs, but make sure do not include very long inputs. (3) Other ones that can maximize the chance of catching a bug. Do not give anything else than the input output pair"
    )
    # note: original instruction: "Generate a complete set of potential input and output pair as a list to test an AI-generated solution to the coding problem.  Assume input will be used via stdin and output is caputred stdout. Include (1) Edge cases, such as empty string or arrays, (3) Complex and diffucult inputs, but make sure do not include very long inputs. (3) Other ones that can maximize the chance of catching a bug. Do not give anything else than the input output pair, in the format of Input: input Output: output."

class GenerateTest_std_input(dspy.Signature):
    prompt = dspy.InputField(format=str)
    generated_tests = dspy.InputField(format=list, desc="A list of previously generated tests to avoid duplicates")
    test = dspy.OutputField(
            desc="Assume input will be used via stdin and output is caputred stdout. Include (1) Edge cases, such as empty string or arrays, (3) Complex and diffucult inputs, but make sure do not include very long inputs. (3) Other ones that can maximize the chance of catching a bug. Do not give anything else than the input output pair. Make sure to output in the format of Input: input Output: output."
    )


class GenerateLCBcodestdin(dspy.Signature):
    """ """

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                      the function should take stdin as input and print the output. Simply call the function after the definition. DO NOT give me anything else! "
    )


class GenerateLCBcodefunctional(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                     DO NOT include anything other than function body! Give me only the function itself! "
    )


class NaiveCodeGenerator(dspy.Module):
    def __init__(self, cot=False):
        if cot:
            self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
            self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        else:
            self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
            self.functional_prog = dspy.Predict(GenerateLCBcodefunctional)
        
        self.adapter = XMLAdapter()

    def forward(self, **kwargs):
        prompt = kwargs.get("prompt")
        is_stdin = kwargs.get("is_stdin")

        prog = self.stdin_prog if is_stdin else self.functional_prog

        pred = prog(prompt=prompt)
        return pred
    
    def collect_trace(self, kwargs, pred):
        # Determine which signature to use based on is_stdin
        is_stdin = kwargs.get("is_stdin")
        original_sig = GenerateLCBcodestdin if is_stdin else GenerateLCBcodefunctional

        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=original_sig,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        # Extract user and assistant messages

        chat_history = [None, None, None]

        for msg in all_messages:
            if msg.get("role") == "system":
                chat_history[0] = {
                    "role": "system",
                    "content": msg['content']
                }
            if msg.get("role") == "user":
                chat_history[1] = {
                    "role": "user",
                    "content": msg['content']
                }
            elif msg.get("role") == "assistant":
                chat_history[2] = {
                    "role": "assistant",
                    "content": msg['content']
                }
        return chat_history


class CodeGeneratorWithRanker(dspy.Module):
    def __init__(
        self,
        temperature: float = 0.7,
        n=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None
    ):
        # assert prog_lm is not None, "prog_lm must be provided"
        # assert generator_lm is not None, "generator_lm must be provided"
        self.prog_lm = prog_lm
        self.generator_lm = generator_lm

        # self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
        self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
        # self.stdin_prog.set_lm(prog_lm)
        # self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog = dspy.Predict(GenerateLCBcodefunctional)
        # self.functional_prog.set_lm(prog_lm)
        # self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs)
        self.test_generator = dspy.Predict(GenerateTests_func_inputs)
        # self.test_generator.set_lm(generator_lm)
        # self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs)
        self.test_generator_stdin = dspy.Predict(GenerateTests_std_inputs)
        # self.test_generator_stdin.set_lm(generator_lm)
        # TODO: use the golden program to check against the test being generated. Or use a teacher model to generate the "golden solution" to test if the tests are valid. Check wether the output of the test is the same as the output of the solution.

        self.log_file = log_file
        self.num_generations = n
        import datetime

        now = datetime.datetime.now()
        self.init_time = now.strftime("%Y%m%d%H%M%S")
        self.temperature = temperature

        self.cache_execution = cache_execution
        self.num_tests = 5
        self.single_test_generator = dspy.ChainOfThought(GenerateTest_std_input)
        # self.single_test_generator = dspy.Predict(GenerateTest_std_input)
        # self.single_test_generator.set_lm(generator_lm)

    def forward(self, example):
        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin if is_stdin else self.test_generator
        
        print(f"[Program] Generating tests for task_id: {task_id}")
        raw_tests = dspy.Prediction(tests="")
        for retry in range(3):
            try:
                start = time.time()
                raw_tests = generator(prompt=prompt)
                end = time.time()
                print(f"[Program] Time taken to generate tests: {end - start}")
                break
            except Exception as e:
                print(f"[Program] Error generating tests for example {example.task_id}: {e}")
                print(f"[Program] Retrying {retry + 1} out of 3")
                continue
            
        tests = post_process_tests_inputs(raw_tests.tests, is_stdin)

        self.prog = self.stdin_prog if is_stdin else self.functional_prog

        print(f"[Program] Generating programs for task_id: {task_id}")
        if len(tests) == 0:
            print(f"No tests found for prompt: {prompt.splitlines()[0]}")
            return self.prog(prompt=prompt)

        preds = [
            self.prog(
                prompt=prompt,
                config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i)),
            )
            for i in range(self.num_generations)
        ]

        
        preds = reduce_preds(preds)

        tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
        tests_counter = Counter(tests_as_strings)
        tests = [
            {"test": json.loads(test_str), "count": count}
            for test_str, count in tests_counter.items()
        ]


        preds_pass = [
            list(
                map(
                    lambda test: test["count"]
                    if check_test(
                        [test["test"]], pred, 0, prompt, "dummy", runtime_debug=True
                    )[0]
                    else 0,
                    tests,
                )
            )
            for pred in preds
        ]

        
        zipped_stats = sorted(zip([p.code for p in preds], map(sum, preds_pass)), key=lambda x: x[1])
        sorted_arr =  {task_id: zipped_stats}
        if self.log_file:
            # with lock:
            self.log_file.write(json.dumps(sorted_arr) + "\n")

        preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[ 
            0
        ]

        pred = preds_pass_rank
        
        return dspy.Prediction(code=pred)
    
class CodeGeneratorWithRanker_prog(CodeGeneratorWithRanker):
    def collect_trace(self, kwargs, pred):
        is_stdin = kwargs.get("is_stdin")
        original_sig = GenerateLCBcodestdin if is_stdin else GenerateLCBcodefunctional

        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=original_sig,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        # Extract user and assistant messages

        chat_history = [None, None, None]

        for msg in all_messages:
            if msg.get("role") == "system":
                chat_history[0] = {
                    "role": "system",
                    "content": msg['content']
                }
            if msg.get("role") == "user":
                chat_history[1] = {
                    "role": "user",
                    "content": msg['content']
                }
            elif msg.get("role") == "assistant":
                chat_history[2] = {
                    "role": "assistant",
                    "content": msg['content']
                }
        return chat_history


class CodeGeneratorWithRanker_test(CodeGeneratorWithRanker):
    def forward(self, example):
        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin if is_stdin else self.test_generator
        
        print(f"[Program] Generating tests for task_id: {task_id}")
        self.raw_tests = dspy.Prediction(tests="")
        for retry in range(3):
            try:
                start = time.time()
                self.raw_tests = generator(prompt=prompt)
                end = time.time()
                print(f"[Program] Time taken to generate tests: {end - start}")
                break
            except Exception as e:
                print(f"[Program] Error generating tests for example {example.task_id}: {e}")
                print(f"[Program] Retrying {retry + 1} out of 3")
                continue
            
        tests = post_process_tests_inputs(self.raw_tests.tests, is_stdin)

        self.prog = self.stdin_prog if is_stdin else self.functional_prog

        print(f"[Program] Generating programs for task_id: {task_id}")
        if len(tests) == 0:
            print(f"No tests found for prompt: {prompt.splitlines()[0]}")
            return self.prog(prompt=prompt)

        preds = [
            self.prog(
                prompt=prompt,
                config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i)),
            )
            for i in range(self.num_generations)
        ]

        
        preds = reduce_preds(preds)

        tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
        tests_counter = Counter(tests_as_strings)
        tests = [
            {"test": json.loads(test_str), "count": count}
            for test_str, count in tests_counter.items()
        ]


        preds_pass = [
            list(
                map(
                    lambda test: test["count"]
                    if check_test(
                        [test["test"]], pred, 0, prompt, "dummy", runtime_debug=True
                    )[0]
                    else 0,
                    tests,
                )
            )
            for pred in preds
        ]

        
        zipped_stats = sorted(zip([p.code for p in preds], map(sum, preds_pass)), key=lambda x: x[1])
        sorted_arr =  {task_id: zipped_stats}
        if self.log_file:
            # with lock:
            self.log_file.write(json.dumps(sorted_arr) + "\n")

        preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[ 
            0
        ]

        pred = preds_pass_rank
        
        return dspy.Prediction(code=pred)

    def collect_trace(self, kwargs, pred):
        is_stdin = kwargs.get("is_stdin")
        original_sig = GenerateTests_std_inputs if is_stdin else GenerateTests_func_inputs

        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=original_sig,
                                inputs=kwargs,
                                outputs=self.raw_tests,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        # Extract user and assistant messages

        chat_history = [None, None, None]

        for msg in all_messages:
            if msg.get("role") == "system":
                chat_history[0] = {
                    "role": "system",
                    "content": msg['content']
                }
            if msg.get("role") == "user":
                chat_history[1] = {
                    "role": "user",
                    "content": msg['content']
                }
            elif msg.get("role") == "assistant":
                chat_history[2] = {
                    "role": "assistant",
                    "content": msg['content']
                }
        return chat_history

    
    