import json
import dspy.evaluate
from .utils import (
    check_correctness,
    unsafe_lcb_runTests,
)
from collections import Counter
import dspy
from .utils import GenerateLCBcodestdin, GenerateLCBcodefunctional, post_process_code, assert_test, assert_test_multiple, assert_dummy
from .lcb_utils import post_process_tests_inputs, reduce_preds, check_test
from .utils import assert_test
from dspy.adapters.xml_adapter import XMLAdapter

# Optional import for assertion module (only used by some classes, not NaiveCodeGenerator_dspy)
try:
    from assertion.assertion_chain import Assertion_module, Assertion_variable
except ImportError:
    # Define dummy classes if assertion module is not available
    class Assertion_module:
        def __init__(self, *args, **kwargs):
            raise ImportError("assertion module not available. Install it or use classes that don't require it.")
    
    class Assertion_variable:
        def __init__(self, *args, **kwargs):
            raise ImportError("assertion module not available. Install it or use classes that don't require it.")

try:
    from dspy.adapters.xml_adapter import XMLAdapter
except ImportError:
    # Fallback to dspy.Adapter if XMLAdapter is not available
    XMLAdapter = dspy.Adapter

from collections import defaultdict
import time

TIMEOUT_CONSTANT = 6
# os.environ["DSP_NOTEBOOK_CACHEDIR"] = "./human_eval_dspy_cache"

# Settings for prompt without demo
# NUM_SAMPLES = 50
# TEMPARATURE_BASE = 0.0
# TEMPARATURE_STEP = 0.01
# NUM_TESTS = 5

# Settings for prompt with demo
# lock = threading.Lock()

NUM_SAMPLES = 20
# TEMPARATURE_BASE = 0.7
TEMPARATURE_STEP = 0.001
NUM_TESTS = 5


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
    def __init__(self, cot=False, prog_lm=None):
        if cot:
            self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
            self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        else:
            self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
            self.functional_prog = dspy.Predict(GenerateLCBcodefunctional)
        
        self.prog_lm = prog_lm
        self.stdin_prog.set_lm(prog_lm)
        self.functional_prog.set_lm(prog_lm)

    def forward(self, example):
        prompt = example.prompt
        is_stdin = example.is_stdin

        prog = self.stdin_prog if is_stdin else self.functional_prog
        pred = prog(prompt=prompt)
        return pred
    
    def copy(self):
        return self.__class__(
            prog_lm=self.prog_lm
            )



class NaiveCodeGenerator_dspy(dspy.Module):
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
        is_stdin = kwargs.get("is_stdin", False)
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


class CodeGeneratorWithRanker_Original(dspy.Module):
    def __init__(
        self,
        temperature: float = 0.7,
        n=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None
    ):
        assert prog_lm is not None, "prog_lm must be provided"
        assert generator_lm is not None, "generator_lm must be provided"
        self.prog_lm = prog_lm
        self.generator_lm = generator_lm

        # self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
        self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
        self.stdin_prog.set_lm(prog_lm)
        # self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog = dspy.Predict(GenerateLCBcodefunctional)
        self.functional_prog.set_lm(prog_lm)
        # self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs)
        self.test_generator = dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm)
        # self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs)
        self.test_generator_stdin = dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm)
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
        self.single_test_generator.set_lm(generator_lm)

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
    
    def copy(self):
        return self.__class__(
            prog_lm=self.prog_lm,
            generator_lm=self.generator_lm
            )



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
        assert prog_lm is not None, "prog_lm must be provided"
        assert generator_lm is not None, "generator_lm must be provided"
        self.prog_lm = prog_lm
        self.generator_lm = generator_lm

        self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
        self.stdin_prog.set_lm(prog_lm.lm)
        self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog.set_lm(prog_lm.lm)
        self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm.lm)
        self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm.lm)
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
        self.single_test_generator.set_lm(generator_lm.lm)

    def forward(self, example):
        
        # is_stdin = has_test_type(example.public_test_cases, "stdin")

        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        # print(list(self.pre_computed_tests.values())[0], len(list(self.pre_computed_tests.values())[0]))
        # if task_id in self.cache_execution:
        #      zipped_stats = self.cache_execution[task_id]
        #    print(len(self.cache_execution[task_id]))
        #assert False
        # else:

        generated_tests = []
        for i in range(self.num_tests):
            # new_test = self.single_test_generator(prompt=prompt, generated_tests=generated_tests, config=dict(temperature=self.temperature + (0.01 * i)))
            # import pdb; pdb.set_trace()
            # new_test = self.single_test_generator(prompt=prompt, config=dict(temperature=1))
            new_test = self.single_test_generator(prompt=prompt, config=dict(temperature=self.temperature + (0.01 * i)))
            
            generated_tests.append(new_test.test)
        
        

        raw_tests = ''.join(generated_tests)
        tests = post_process_tests_inputs(raw_tests, is_stdin)



        # generator = self.test_generator_stdin if is_stdin else self.test_generator
        
        # # raw_tests = generator(prompt=prompt, config=dict(temperature=self.temperature + (0.01 * r)))

        # raw_tests = generator(prompt=prompt, config=dict(temperature=self.temperature + (0.01)))

        # tests = post_process_tests_inputs(raw_tests.tests, is_stdin)

        self.prog = self.stdin_prog if is_stdin else self.functional_prog
        # tests = self.pre_computed_tests[task_id]

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

        # print(zipped_stats)
        preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[ 
            0
        ]#.code

        pred = preds_pass_rank
        
        return dspy.Prediction(code=pred)
    
    def copy(self):
        return self.__class__(
            prog_lm=self.prog_lm,
            generator_lm=self.generator_lm
            )




class CodeGeneratorWithRankerAssertTestGen(dspy.Module):
    def __init__(
        self,
        temperature: float = 0.7,
        num_tests=5,
        num_programs=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None,
    ):
        # super().__init__(
        #     temperature=temperature,
        #     log_file=log_file,
        #     cache_execution=cache_execution,
        #     prog_lm=prog_lm,
        #     generator_lm=generator_lm
        # )
        self.generator_lm = generator_lm
        self.prog_lm = prog_lm
        self.num_tests = num_tests
        self.num_programs = num_programs

        self.test_generator = dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm.lm)
        self.test_generator_assertion = Assertion_module(self.test_generator, max_retries=8, topk=1)
        self.test_generator_assertion.add_assertion(assert_test_multiple)

        self.test_generator_stdin = dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm.lm)
        self.test_generator_stdin_assertion = Assertion_module(self.test_generator_stdin, max_retries=8, topk=1)
        self.test_generator_stdin_assertion.add_assertion(assert_test_multiple)

        self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
        self.stdin_prog.set_lm(prog_lm.lm)
        # self.stdin_prog_assertion = Assertion_module(self.stdin_prog, max_retries=3, topk=2)
        # self.stdin_prog_assertion.add_assertion(assert_prog)
        # self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog.set_lm(prog_lm.lm)
        # self.functional_prog_assertion = Assertion_module(self.functional_prog, max_retries=3, topk=2)
        # self.functional_prog_assertion.add_assertion(assert_prog)
    
    def forward(self, example):
        
        # is_stdin = has_test_type(example.public_test_cases, "stdin")

        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        print("is stdin !!!!!!!!!!!!!", is_stdin)

        generator = self.test_generator_stdin_assertion if is_stdin else self.test_generator_assertion

        prompt_a = Assertion_variable(prompt, 1)
        task_id_a = Assertion_variable(task_id, 1)
        is_stdin_a = Assertion_variable(is_stdin, 1)


        triples = []

        # try:
        triples = generator(prompt=prompt_a, task_id=task_id_a, is_stdin=is_stdin_a)
        # except Exception as e:
        #     print("error: ", e)

        self.prog = self.stdin_prog if is_stdin else self.functional_prog

        # if triples == []:
        #     import pdb; pdb.set_trace()
        #     return [[self.prog(prompt=prompt, config=dict(temperature=self.temperature)), id]]


        # import pdb; pdb.set_trace()

        preds = []
        for i in range(self.num_programs):
            for retry in range(3):
                try:
                    preds.append(self.prog(prompt=prompt, config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i))))
                    break
                except Exception as e:
                    print("error: ", e)
                    continue

        preds = reduce_preds(preds)

        # import pdb; pdb.set_trace()

        outputs = []
    
        for score, raw_tests, id in triples:
            tests = []
            try:
                tests = post_process_tests_inputs(raw_tests.tests, is_stdin)
            except Exception as e:
                print("error: ", e)

            if len(tests) == 0:
                print(f"No tests found for prompt: {prompt.splitlines()[0]}")
                for retry in range(3):
                    try:
                        preds.append(self.prog(prompt=prompt, config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i))))
                        break
                    except Exception as e:
                        print("error: ", e)
                        continue
                # outputs.append([self.prog(prompt=prompt, id=id), id])

            tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
            tests_counter = Counter(tests_as_strings)
            tests = [
                {"test": json.loads(test_str), "count": count}
                for test_str, count in tests_counter.items()
            ]

            print("Selecting outputs...")
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

            print("selection done!!!")

            zipped_stats = sorted(zip([p.code for p in preds], map(sum, preds_pass)), key=lambda x: x[1])
            sorted_arr =  {task_id: zipped_stats}
            if self.log_file:
                # with lock:
                self.log_file.write(json.dumps(sorted_arr) + "\n")

            # print(zipped_stats)
            preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[ 
                0
            ]#.code

            pred = preds_pass_rank
            
            outputs.append([dspy.Prediction(code=pred), id])
        

        if outputs == []:
            print("EMPTY OUTPUTS!!!")
        return outputs
    
    def update_reward(self, rewards, ids):
        # defensivg coding here. Calling updates to both modules. If id cannot be found in a module, reward update is not going to take effect.
        self.test_generator_stdin_assertion.update_reward(rewards, ids)
        self.test_generator_assertion.update_reward(rewards, ids)
    
    def get_trace(self):
        trace_dict = defaultdict(list)
        generator_trace = self.test_generator_assertion.get_trace()
        generator_stdin_trace = self.test_generator_stdin_assertion.get_trace()
        print(f"generator_trace size: {len(generator_trace)}. generator_trace_stdin size: {len(generator_stdin_trace)}")
        trace_dict[self.generator_lm] = generator_trace + generator_stdin_trace

        return trace_dict
    
    def get_current_rewards(self):
        return self.test_generator_stdin_assertion.get_current_rewards() + self.test_generator_assertion.get_current_rewards()
    
    def update_model(self, lm_manager):
        """
        Swap in a new query LM and a new infer LM, updating all sub‐modules.
        """
        print("before:", self.generator_lm.lm, self.generator_lm.lm.model)
        new_generator_lm = lm_manager.get_lm('generator_lm')
        new_prog_lm = lm_manager.get_lm('prog_lm')
        self.generator_lm = new_generator_lm
        self.prog_lm = new_prog_lm

        self.test_generator.set_lm(self.generator_lm.lm)
        self.test_generator_stdin.set_lm(self.generator_lm.lm)
        # import pdb; pdb.set_trace()

        # self.stdin_prog.set_lm(self.generator_lm.lm)
        # self.functional_prog.set_lm(self.generator_lm.lm)
        # self.stdin_prog.set_lm(self.prog_lm.lm)
        # self.functional_prog.set_lm(self.prog_lm.lm)

        print("after:", self.generator_lm.lm, self.generator_lm.lm.model)
    
    def reset(self):
        self.test_generator_assertion.reset()
        self.test_generator_stdin_assertion.reset()


class CodeGeneratorWithRankerAssertTestGen_single(dspy.Module):
    def __init__(
        self,
        temperature: float = 0.7,
        num_tests=5,
        num_programs=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None,
        use_chain_of_thought=False,
    ):
        print("use_chain_of_thought: ", use_chain_of_thought)
        self.use_chain_of_thought = use_chain_of_thought
        self.generator_lm = generator_lm
        self.num_tests = num_tests
        self.num_programs = num_programs

        self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm.lm)
        self.test_generator_assertion = Assertion_module(self.test_generator, max_retries=8, topk=1)
        self.test_generator_assertion.add_assertion(assert_test_multiple)

        self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm.lm)
        self.test_generator_stdin_assertion = Assertion_module(self.test_generator_stdin, max_retries=8, topk=1)
        self.test_generator_stdin_assertion.add_assertion(assert_test_multiple)

    
    def forward(self, example):
        
        # is_stdin = has_test_type(example.public_test_cases, "stdin")

        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin_assertion if is_stdin else self.test_generator_assertion

        prompt_a = Assertion_variable(prompt, 1)
        task_id_a = Assertion_variable(task_id, 1)
        is_stdin_a = Assertion_variable(is_stdin, 1)

        # try:
        triples = generator(prompt=prompt_a, task_id=task_id_a, is_stdin=is_stdin_a)

        return []
    
    def update_reward(self, rewards, ids):
        # defensivg coding here. Calling updates to both modules. If id cannot be found in a module, reward update is not going to take effect.
        self.test_generator_stdin_assertion.update_reward(rewards, ids)
        self.test_generator_assertion.update_reward(rewards, ids)
    
    def get_trace(self):
        trace_dict = defaultdict(list)
        generator_trace = self.test_generator_assertion.get_trace()
        generator_stdin_trace = self.test_generator_stdin_assertion.get_trace()
        print(f"generator_trace size: {len(generator_trace)}. generator_trace_stdin size: {len(generator_stdin_trace)}")
        trace_dict[self.generator_lm] = generator_trace + generator_stdin_trace

        return trace_dict
    
    def get_current_rewards(self):
        return self.test_generator_stdin_assertion.get_current_rewards() + self.test_generator_assertion.get_current_rewards()
    
    def update_model(self, lm_manager):
        """
        Swap in a new query LM and a new infer LM, updating all sub‐modules.
        """
        print("before:", self.generator_lm.lm, self.generator_lm.lm.model)
        new_generator_lm = lm_manager.get_lm('generator_lm')
        self.generator_lm = new_generator_lm

        self.test_generator.set_lm(self.generator_lm.lm)
        self.test_generator_stdin.set_lm(self.generator_lm.lm)

        print("after:", self.generator_lm.lm, self.generator_lm.lm.model)
    
    def reset(self):
        self.test_generator_assertion.reset()
        self.test_generator_stdin_assertion.reset()



class CodeGeneratorWithRankerAssertTestGen_single_dspy(dspy.Module):
    def __init__(
        self,
        temperature: float = 0.7,
        num_tests=5,
        num_programs=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None,
        use_chain_of_thought=False,
    ):
        print("use_chain_of_thought: ", use_chain_of_thought)
        self.use_chain_of_thought = use_chain_of_thought
        self.generator_lm = generator_lm
        self.num_tests = num_tests
        self.num_programs = num_programs

        self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm)

        self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm)

    
    def forward(self, **kwargs):
        
        prompt = kwargs.get("prompt")
        is_stdin = kwargs.get("is_stdin")

        generator = self.test_generator_stdin if is_stdin else self.test_generator

        return generator(prompt=prompt)



class CodeGeneratorWithRankerAssertProgGen(CodeGeneratorWithRanker):
    def __init__(
        self,
        temperature: float = 0.7,
        num_tests=5,
        num_programs=5,
        log_file=None,
        cache_execution=None,
        prog_lm=None,
        generator_lm=None,
    ):
        super().__init__(
            temperature=temperature,
            log_file=log_file,
            cache_execution=cache_execution,
            prog_lm=prog_lm,
            generator_lm=generator_lm
        )
        self.num_tests = num_tests
        self.num_programs = num_programs

        self.test_generator = dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm.lm)

        self.test_generator_stdin = dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm.lm)

        self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
        self.stdin_prog.set_lm(prog_lm.lm)
        self.stdin_prog_assertion = Assertion_module(self.stdin_prog, max_retries=3, topk=8)
        # self.stdin_prog_assertion = Assertion_module(self.stdin_prog, max_retries=3, topk=2)
        # self.stdin_prog_assertion.add_assertion(assert_prog)
        # self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.functional_prog.set_lm(prog_lm.lm)
        self.functional_prog_assertion = Assertion_module(self.functional_prog, max_retries=3, topk=8)
        
        # self.functional_prog_assertion = Assertion_module(self.functional_prog, max_retries=3, topk=2)
        # self.functional_prog_assertion.add_assertion(assert_prog)
    
    def forward(self, example):
        
        # is_stdin = has_test_type(example.public_test_cases, "stdin")

        prompt = example.prompt
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin if is_stdin else self.test_generator

        tests = []
        for retry in range(3):
            try:
                raw_tests = generator(prompt=prompt)
                tests = post_process_tests_inputs(raw_tests.tests, is_stdin)

                tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
                tests_counter = Counter(tests_as_strings)
                tests = [
                    {"test": json.loads(test_str), "count": count}
                    for test_str, count in tests_counter.items()
                ]

                break
            except Exception as e:
                print("error: ", e)
                continue

        self.prog = self.stdin_prog if is_stdin else self.functional_prog                    
        
        prompt_a = Assertion_variable(prompt, 1)
        tests_a = Assertion_variable(tests, 1)

        pred = self.prog(prompt=prompt_a, tests=tests_a)
        print("Pred: ", pred)
        return pred

    def update_reward(self, rewards, ids):
        # defensivg coding here. Calling updates to both modules. If id cannot be found in a module, reward update is not going to take effect.
        self.stdin_prog_assertion.update_reward(rewards, ids)
        self.functional_prog_assertion.update_reward(rewards, ids)
        
    def get_trace(self):
        trace_dict = defaultdict(list)
        generator_stdin_trace = self.stdin_prog_assertion.get_trace()
        generator_trace = self.functional_prog_assertion.get_trace()
        print(f"generator_trace size: {len(generator_trace)}. generator_trace_stdin size: {len(generator_stdin_trace)}")
        trace_dict[self.generator_lm] = generator_trace + generator_stdin_trace

        return trace_dict
    
    def get_current_rewards(self):
        return self.functional.get_current_rewards() + self.stdin_prog_assertion.get_current_rewards()
    
    def update_model(self, lm_manager):
        """
        Swap in a new query LM and a new infer LM, updating all sub‐modules.
        """
        print("before:", self.generator_lm.lm, self.generator_lm.lm.model)
        
        new_generator_lm = lm_manager.get_lm('generator_lm')
        new_prog_lm = lm_manager.get_lm('prog_lm')
        
        self.generator_lm = new_generator_lm
        self.prog_lm = new_prog_lm

        self.stdin_prog.set_lm(self.generator_lm.lm)
        self.functional_prog.set_lm(self.generator_lm.lm)
        
        self.stdin_prog.set_lm(self.prog_lm.lm)
        self.functional_prog.set_lm(self.prog_lm.lm)

        print("after:", self.generator_lm.lm, self.generator_lm.lm.model)
    
    def reset(self):
        self.stdin_prog_assertion.reset()
        self.functional.reset()
        
class NaiveTestGenerator(dspy.Module):
    def __init__(
        self,
        generator_lm=None,
        use_chain_of_thought=False,
    ):

        self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm)

        self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm)

    def forward(self, example):
        prompt = example.prompt
        task_id = example.task_id
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin if is_stdin else self.test_generator

        for retry in range(3):
            try:
                pred = generator(prompt=prompt)
                return pred
            except Exception as e:
                print("[NaiveTestGenerator] Error generating tests: ", e)
                continue

        return 0