from datasets import load_dataset
import dspy


def papillon_data():
    pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")
    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")

    examples = [
        dspy.Example({
            "target_response": x["target_response"],
            "user_query": x["user_query"],
            "pii_str": x["pii_units"]
        }).with_inputs("user_query")
        for x in pupa_new["train"]
    ]

    trainset, testset = examples[:300], examples[300:]  
    print(f"Loaded {len(trainset)} training examples, and {len(testset)} test examples.")
    return trainset, testset