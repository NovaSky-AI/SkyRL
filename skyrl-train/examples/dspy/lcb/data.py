import dspy
from datasets import load_dataset
import random
import json
import pickle
import os
import base64
import zlib
from tqdm import tqdm

def modify_entry(entry, idx):
    entry["question_id"] = idx

    # Ensure lists exist
    entry["private_tests"]["input"] = list(entry["private_tests"].get("input", []))
    entry["private_tests"]["output"] = list(entry["private_tests"].get("output", []))
    entry["generated_tests"]["input"] = list(entry["generated_tests"].get("input", []))
    entry["generated_tests"]["output"] = list(entry["generated_tests"].get("output", []))

    # Extend private tests with generated tests
    entry["private_tests"]["input"].extend(entry["generated_tests"]["input"])
    entry["private_tests"]["output"].extend(entry["generated_tests"]["output"])

    # Convert to desired format
    private_test_cases = []
    for inp, out in zip(entry["private_tests"]["input"], entry["private_tests"]["output"]):
        private_test_cases.append({
            "input": inp.strip(),  # Ensure clean formatting
            "output": out.strip(),
            "testtype": "stdin"  # Set test type as "stdin"
        })

    # Apply the new format
    entry["private_test_cases"] = json.dumps(private_test_cases)
    entry["public_test_cases"] = json.dumps(entry.pop("public_tests"))  # Store as JSON string

    return entry


# def map_to_dspy_example(row):
#     return {
#         "prompt": row["description"],
#         "test": json.loads(row["private_test_cases"]),
#         "entry_point": "",
#         "canonical_solution": "",  # seems like live code bench lite does not have this field
#         "task_id": row["question_id"],
#         "is_stdin": True,
#         "public_test_cases": row["public_test_cases"],
#     }


def map_to_dspy_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "canonical_solution": "",  # seems like live code bench lite does not have this field
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
    }

def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def update_dataset_in_place(
    dataset,
):  ## helper functions to translate the private test cases
    

    # for entry in tqdm(dataset):
    for i, entry in enumerate(dataset):
        # print("before: ", type(entry["private_test_cases"]))
        # import pdb; pdb.set_trace()
        try:
            # Translate the private test cases
            decoded_private_test_cases = translate_private_test_cases(
                entry["private_test_cases"]
            )
            # Update the entry in place
            entry["private_test_cases"] = decoded_private_test_cases
            # dataset[i]["private_test_cases"] = decoded_private_test_cases
        except Exception as e:
            print(e)
        
        # print("after: ", type(entry["private_test_cases"]))
        

def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False



def get_split(split="test"):
    pkl_path = f"live_code_bench_dataset_{split}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    
    # lcb_codegen = load_dataset(
    #             "deepmind/code_contests", split=split, trust_remote_code=True
    #             )
    
    lcb_codegen = load_dataset(
            "livecodebench/code_generation_lite", version_tag="release_v2", split=split, trust_remote_code=True
            )

    # filtered_lcb_codegen_list = [entry for entry in lcb_codegen if of_difficulty(entry, args.difficulty)]
    filtered_lcb_codegen_list = [entry for entry in lcb_codegen]

    extracted_tests = {}
    for entry in filtered_lcb_codegen_list:
        extracted_tests[entry["question_id"]] = json.loads(entry["public_test_cases"])

    update_dataset_in_place(filtered_lcb_codegen_list)  ## decode the private test cases
    

    ## extract the private test cases for calculating pass at 20
    extracted_private_tests = {}
    for entry in filtered_lcb_codegen_list:
        extracted_private_tests[entry["question_id"]] = entry["private_test_cases"]


    live_code_bench_dataset = [
        dspy.Example(**map_to_dspy_example(row)).with_inputs(
            "prompt", "test", "entry_point", "canonical_solution", "task_id", "is_stdin", "public_test_cases"
        )
        for row in filtered_lcb_codegen_list
    ]


    # lcb_codegen = load_dataset(
    #             "deepmind/code_contests", split="train", trust_remote_code=True
    #             )

    # lcb_codegen = lcb_codegen.map(modify_entry, with_indices=True)

    # random.seed(41)
    # print(f"Before: {len(lcb_codegen)}")
    # filtered_lcb_codegen_list = lcb_codegen

    # extracted_tests = {}
    # for entry in filtered_lcb_codegen_list:
    #     extracted_tests[entry["question_id"]] = json.loads(entry["public_test_cases"])

    # # update_dataset_in_place(filtered_lcb_codegen_list)  ## decode the private test cases

    # ## extract the private test cases for calculating pass at 20
    # extracted_private_tests = {}
    # for entry in filtered_lcb_codegen_list:
    #     extracted_private_tests[entry["question_id"]] =  json.loads(entry["private_test_cases"])


    # live_code_bench_dataset = [
    #     dspy.Example(**map_to_dspy_example(row)).with_inputs(
    #         "prompt", "test", "entry_point", "canonical_solution", "task_id", "is_stdin", "public_test_cases"
    #     )
    #     for row in filtered_lcb_codegen_list
    # ]

    with open(f"live_code_bench_dataset_{split}.pkl", "wb") as f:
        print("Saving dataset...")
        pickle.dump(live_code_bench_dataset, f)

    return live_code_bench_dataset


def lcb_data():
    # return None, get_split("test")
    testset = get_split("test")
    return testset[:400], testset[400:]
