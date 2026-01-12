import dspy
from .utils import GenerateLCBcodestdin, GenerateLCBcodefunctional, post_process_code, assert_test, assert_test_multiple, assert_dummy
from .lcb_utils import post_process_tests_inputs, reduce_preds, check_test
from .utils import assert_test
from dspy.adapters.xml_adapter import XMLAdapter
from collections import Counter
import json
import time
import asyncio

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
        return chat_history, None


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
        self.adapter = XMLAdapter()

    def forward(self, **kwargs):
        prompt = kwargs.get("prompt")
        canonical_solution = kwargs.get("canonical_solution")
        task_id = kwargs.get("task_id")
        test = kwargs.get("test")
        entry_point = kwargs.get("entry_point")
        is_stdin = kwargs.get("is_stdin")

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
    def __init__(self):
        super().__init__()
        test_gen_lm = dspy.LM(
            model="openai/Qwen/Qwen2.5-Coder-3B-Instruct",
            api_base="http://0.0.0.0:8002/v1",
            api_key="fake-key",
            temperature=1.0,
            model_type="chat",
            max_tokens=4096,
            cache=False,
        )
        self.test_generator_stdin.set_lm(test_gen_lm)
        self.test_generator.set_lm(test_gen_lm)

        # TODO: change the prog lm to something else
        self.stdin_prog.set_lm(test_gen_lm)
        self.functional_prog.set_lm(test_gen_lm)
    def collect_trace(self, kwargs, pred):
        is_stdin = kwargs.get("is_stdin")
        prog_gen = self.stdin_prog if is_stdin else self.functional_prog

        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=prog_gen.predictors()[0].signature,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        # Extract user and assistant messages

        return all_messages, None


class CodeGeneratorWithRanker_test(CodeGeneratorWithRanker):
    def __init__(self):
        super().__init__()
        test_gen_lm = dspy.LM(
            model="openai/Qwen/Qwen2.5-Coder-1.5B-Instruct",
            api_base="http://0.0.0.0:8002/v1",
            api_key="fake-key",
            temperature=1.0,
            model_type="chat",
            max_tokens=4096,
            cache=False,
        )
        self.test_generator_stdin.set_lm(test_gen_lm)
        self.test_generator.set_lm(test_gen_lm)

        prog_gen_lm = dspy.LM(
            model="openai/Qwen/Qwen2.5-Coder-7B-Instruct",
            api_base="http://0.0.0.0:8002/v1",
            api_key="fake-key",
            temperature=1.0,
            model_type="chat",
            max_tokens=4096,
            cache=False,
        )
        self.stdin_prog.set_lm(test_gen_lm)
        self.functional_prog.set_lm(test_gen_lm)
        
    async def _check_one(self, test_obj, pred, prompt):
        # test_obj is {"test": ..., "count": ...}
        ok = await asyncio.to_thread(
            check_test,
            [test_obj["test"]],
            pred,
            0,
            prompt,
            "dummy",
            runtime_debug=True,
        )
        # check_test returns something like (passed_bool, ...)
        return test_obj["count"] if ok[0] else 0
        
    async def forward(self, example):
        prompt = example.get("prompt")  
        task_id = example.get("task_id")
        is_stdin = example.get("is_stdin")

        generator = self.test_generator_stdin if is_stdin else self.test_generator
        
        print(f"[Program] Generating tests for task_id: {task_id}")
        self.raw_tests = dspy.Prediction(tests="")
        for retry in range(3):
            try:
                start = time.time()
                self.raw_tests = await generator.acall(prompt=prompt)
                end = time.time()
                print(f"[Program] Time taken to generate tests: {end - start}")
                break
            except Exception as e:
                print(f"[Program] Error generating tests for example {task_id}: {e}")
                print(f"[Program] Retrying {retry + 1} out of 3")
                continue
            
        print(f"[Program] tests generated for task_id: {task_id}")
        tests = post_process_tests_inputs(self.raw_tests.tests, is_stdin)

        self.prog = self.stdin_prog if is_stdin else self.functional_prog

        print(f"[Program] Generating programs for task_id: {task_id}")
        if len(tests) == 0:
            print(f"No tests found for prompt: {prompt.splitlines()[0]}")
            return await self.prog.acall(prompt=prompt)

        preds = [
            await self.prog.acall(
                prompt=prompt,
                config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i)),
            )
            for i in range(self.num_generations)
        ]
        print(f"[Program] programs generated for task_id: {task_id}")

        
        preds = reduce_preds(preds)

        tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
        tests_counter = Counter(tests_as_strings)
        tests = [
            {"test": json.loads(test_str), "count": count}
            for test_str, count in tests_counter.items()
        ]


        print(f"[Program] running tests for task_id: {task_id}")
        start = time.time()
        # preds_pass = [
        #     list(
        #         map(
        #             lambda test: test["count"]
        #             if check_test(
        #                 [test["test"]], pred, 0, prompt, "dummy", runtime_debug=True
        #             )[0]
        #             else 0,
        #             tests,
        #         )
        #     )
        #     for pred in preds
        # ]

        preds_pass = []
        for pred in preds:
            tasks = [self._check_one(t, pred, prompt) for t in tests]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            preds_pass.append(list(results))

            
        end = time.time()
        print(f"[Program] Time taken to run tests: {end - start}")
        print(f"[Program] tests finished for task_id: {task_id}")
        
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
        generator = self.test_generator_stdin if is_stdin else self.test_generator

        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=generator.predictors()[0].signature,
                                inputs=kwargs,
                                outputs=self.raw_tests,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        # Extract user and assistant messages

        return all_messages, self.raw_tests

    
    