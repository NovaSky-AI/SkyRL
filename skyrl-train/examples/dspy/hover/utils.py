from typing import List, Dict, Any
import dspy
import os
import pickle
import bm25s
import Stemmer

def _get_hover_data_dir() -> str:
    """Return the directory where all hover-related data should be stored."""
    hover_dir = os.path.join(os.path.expanduser("~"), "data", "hover")
    os.makedirs(hover_dir, exist_ok=True)
    return hover_dir

class BM25Searcher:
    _shared_retriever = None
    _shared_corpus = None

    def __init__(self):
        if BM25Searcher._shared_retriever is None:
            pkl_path = os.path.join(_get_hover_data_dir(), "bm25_retriever.pkl")
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)

            BM25Searcher._shared_retriever = data["retriever"]
            BM25Searcher._shared_corpus = data["corpus"]

        self.retriever = BM25Searcher._shared_retriever
        self.corpus = BM25Searcher._shared_corpus
        self.stemmer = Stemmer.Stemmer("english")

    def search(self, query: str, k: int) -> List[str]:
        tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=False,
        )

        results, scores = self.retriever.retrieve(
            tokens, k=k, n_threads=1, show_progress=False
        )

        return [
            self.corpus[doc]
            for doc, score in zip(results[0], scores[0])
        ]


async def hover_query_reward_fn(example, pred):
    return 0

async def hover_final_reward_fn(example, pred, trace=None):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)

# async def hover_final_reward_fn(example, pred):
#     gold_titles = set(
#         map(
#             dspy.evaluate.normalize_text,
#             [doc["key"] for doc in example["supporting_facts"]],
#         )
#     )
#     found_titles = set(
#         map(
#             dspy.evaluate.normalize_text,
#             pred.titles
#         )
#     )
#     return gold_titles.issubset(found_titles)
    
