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


from sentence_transformers import SentenceTransformer, util
import numpy as np
print("loading sentence transformer model...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
print("loading sentence transformer done")

async def hover_query_reward_fn(example: list[str], preds, st_model=sentence_model, avg_weight=0.7, min_weight=0.3):
    return 0
    queries = [p.followup_search_query for p in preds]
    if len(queries) <= 1:
        # One query is trivially non-repetitive
        return 1.0

    # Embed and normalize
    emb = st_model.encode(queries, normalize_embeddings=True)  # [N, D]

    # Compute centroid
    centroid = emb.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)

    # Cosine similarity of each query to centroid
    sims = emb @ centroid  # [N], in [-1, 1], usually [0, 1]

    avg_sim = float(sims.mean())
    min_sim = float(sims.min())

    # "How similar are these queries overall?"
    similarity_score = (
        avg_weight * avg_sim +
        min_weight * min_sim
    )
    similarity_score = float(np.clip(similarity_score, 0.0, 1.0))

    # INVERT → distinctiveness
    distinctiveness_score = 1.0 - similarity_score
    import pdb; pdb.set_trace()
    return distinctiveness_score

async def hover_query_reward_fn_2(example, preds):
    return

# async def assert_no_duplicate_notes(example, trace):
    
#     unique_notes = set(pred.new_notes)
#     if not len(unique_notes) == len(pred.new_notes):
#         return "Ensure there's no duplicate notes", 0
#     return None, 1

# async def hover_query_reward_fn(example, pred):
#     return 0

async def hover_final_reward_fn(example, pred, trace=None):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)
