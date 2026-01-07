import random
import ujson
import dspy
import os
import pickle
import tqdm
import tarfile
import bm25s
import Stemmer
from datasets import load_dataset
from .utils import _get_hover_data_dir
from dspy.datasets import DataLoader

def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))

def prepare_corpus(input_path: str = "wiki.abstracts.2017.jsonl") -> None:
    hover_dir = _get_hover_data_dir()

    # Always keep all artifacts (jsonl, archive, pickle) under ~/data/hover
    jsonl_path = os.path.join(hover_dir, os.path.basename(input_path))
    archive_path = os.path.join(hover_dir, "wiki.abstracts.2017.tar.gz")
    pkl_path = os.path.join(hover_dir, "bm25_retriever.pkl")

    if os.path.exists(pkl_path):
        return

    if not os.path.exists(jsonl_path):
        from dspy.utils import download

        # Download to the current working directory using the default behavior.
        # Then move the resulting archive into the hover data directory so that
        # everything is centralized under ~/data/hover.
        remote_url = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
        local_archive_name = "wiki.abstracts.2017.tar.gz"
        download(remote_url)

        if os.path.exists(local_archive_name):
            os.makedirs(hover_dir, exist_ok=True)
            os.replace(local_archive_name, archive_path)

        # Extract the downloaded archive into the hover data directory
        if os.path.exists(archive_path):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=hover_dir)

    corpus = []
    with open(jsonl_path, "r") as f:
        for line in tqdm.tqdm(f):
            line = ujson.loads(line)
            corpus.append(f"{line['title']} | {' '.join(line['text'])}")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "retriever": retriever,
                "corpus": corpus,  # optional, if you need original texts
                "corpus_tokens": corpus_tokens,
            },
            fh,
        )

def hover_data():

    kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
    hover = DataLoader().from_huggingface(dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs)

    hpqa_ids = set()
    hover = [
        dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
        for x in hover
        if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
    ]

    random.Random(0).shuffle(hover)
    trainset, devset, testset = hover[:600], hover[600:900], hover[900:]

    print("Preparing corpus...")
    prepare_corpus("wiki.abstracts.2017.jsonl")
    print("Corpus prepared.")

    return trainset + devset, testset