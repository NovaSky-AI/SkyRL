import uuid
import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset

CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names

def banking77_data():
    kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

    trainset = [
        dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label], task_id=uuid.uuid4()).with_inputs("text", "hint")
        for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:1000]    
    ]
    validationset = [
        dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label], task_id=uuid.uuid4()).with_inputs("text", "hint")
        for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[1000: 1200]
    ]
    
    random.Random(0).shuffle(trainset)
    
    return trainset, validationset




    
