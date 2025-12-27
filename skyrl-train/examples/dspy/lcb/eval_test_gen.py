import dspy
from utils import assert_test_multiple_test, assert_test_multiple
from program import GenerateTests_func_inputs, GenerateTests_std_inputs, CodeGeneratorWithRanker
from data import lcb_data
from tqdm import tqdm

class CodeGeneratorWithRankerAssertTestGen_eval(dspy.Module):
    def __init__(
        self,
        generator_lm=None,
        split="test",
        use_chain_of_thought=False,
    ):

        self.split = split
        if split == "test":
            self.reward_fn = assert_test_multiple_test
        else:
            self.reward_fn = assert_test_multiple
        self.test_generator = dspy.ChainOfThought(GenerateTests_func_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_func_inputs)
        self.test_generator.set_lm(generator_lm)

        self.test_generator_stdin = dspy.ChainOfThought(GenerateTests_std_inputs) if use_chain_of_thought else dspy.Predict(GenerateTests_std_inputs)
        self.test_generator_stdin.set_lm(generator_lm)


    
    def forward(self, example):

        prompt = example.prompt
        canonical_solution = example.canonical_solution
        task_id = example.task_id
        test = example.test
        entry_point = example.entry_point
        is_stdin = example.is_stdin

        generator = self.test_generator_stdin if is_stdin else self.test_generator

        for retry in range(3):
            try:
                pred = generator(prompt=prompt)
                _, score = self.reward_fn(pred, prompt=prompt, task_id=task_id, is_stdin=is_stdin)
                return score
            except Exception as e:
                print("error: ", e)
                continue

        return 0



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

def make_qwen7b(port_qwen):
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

def make_qwen8b(port_qwen):
    qwen = dspy.LM(
        model="openai/Qwen/Qwen3-8B",
        api_base=f"http://localhost:{port_qwen}/v1",
        api_key="fake-key",
        temperature=0.7,
        model_type="chat",
        cache=False,
        # max_tokens=16000,
    )

    return qwen

lm = make_qwen8b(7001)
split = "train"
prog = CodeGeneratorWithRankerAssertTestGen_eval(generator_lm=lm, split=split, use_chain_of_thought=True)

trainset,testset = lcb_data()

rewards = []
for example in tqdm(testset if split == "test" else trainset):
    score = prog(example)
    rewards.append(score)
    # break

print(f"Average Reward: {sum(rewards) / len(rewards)}")